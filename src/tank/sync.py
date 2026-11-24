"""Publish immutable batch fragments and their pinned strategy sources."""
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from . import evaluation, versions
from .eval_plan import RUN_ID, publish_runtime, read_runtime_archive, shared_runtime, validate_manifest, version_ids
from .eval_store import database, read_manifest
from .workspace import read_json, write_json

PART_BYTES = 32 * 1024 * 1024
REMOTE_TIMEOUTS = {"ls-remote": 30, "fetch": 300, "push": 300}


def _store_fragment(payload, directory, digest):
    if payload.stat().st_size <= PART_BYTES:
        target = directory / (digest + ".jsonl.gz")
        payload.replace(target)
        return target
    target = directory / (digest + ".parts")
    if not target.exists():
        staged = payload.parent / "split"
        staged.mkdir()
        with payload.open("rb") as stream:
            index = 0
            while part := stream.read(PART_BYTES):
                (staged / f"{index:08d}.bin").write_bytes(part)
                index += 1
        try:
            staged.rename(target)
        except OSError:
            if not target.is_dir():
                raise
    return target


def _fragments(directory):
    return sorted((*directory.glob("*.jsonl.gz"), *directory.glob("*.parts")))


def _unpack_fragment(fragment, temp):
    if fragment.suffix != ".parts":
        return fragment
    payload = temp / "matches.jsonl.gz"
    with payload.open("wb") as out:
        for index, part in enumerate(sorted(fragment.iterdir())):
            if part.name != f"{index:08d}.bin" or part.is_symlink():
                raise ValueError(f"结果分段无效 {part.name}")
            with part.open("rb") as stream:
                shutil.copyfileobj(stream, out)
    return payload


def _copy_version(root, version):
    if version == "baseline":
        return
    versions.verify(root, version)
    source = versions.snapshot_data(root, version).parent
    dest = root / "published/versions" / version
    if dest.is_dir():
        if versions._version_id(versions._file_hashes(dest / "files")) != version:
            raise ValueError(f"已共享的版本不完整 {version}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staged = Path(tmp) / version
        shutil.copytree(source, staged)
        (staged / ".gitattributes").write_bytes(b"files/** -text\n")
        try:
            staged.rename(dest)
        except OSError:
            if not dest.is_dir():
                raise


def publish(root, run_id):
    with database(root) as db:
        manifest, _ = read_manifest(db, run_id)
    ids = version_ids(manifest)
    for version in sorted(ids):
        _copy_version(root, version)
    engine = manifest["engine"]
    runtime = publish_runtime(root, engine) if "source_sha256" in engine else None
    dest = root / "published/runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        payload = Path(tmp) / "matches.jsonl.gz"
        result = evaluation.export_run(root, run_id, payload)
        with payload.open("rb") as f:
            key = hashlib.file_digest(f, "sha256").hexdigest()
        fragment = _store_fragment(payload, dest, key)
    write_json(dest / "manifest.json", manifest)
    return {"run_id": run_id, "versions": sorted(ids), "completed": result["completed"],
            "fragment": str(fragment.relative_to(root)), "runtime": str(runtime.relative_to(root)) if runtime else None,
            "state": "ready_for_git"}


def restore(root, run_id):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("批次 ID 格式错误")
    dest = root / "published/runs" / run_id
    manifest = read_json(dest / "manifest.json")
    validate_manifest(manifest)
    if manifest["run_id"] != run_id:
        raise ValueError("批次目录与清单不匹配")
    for version in version_ids(manifest) - {"baseline"}:
        versions.verify(root, version)
    engine = manifest["engine"]
    if "source_sha256" in engine:
        read_runtime_archive(shared_runtime(root, engine), engine)
    files = _fragments(dest)
    if not files:
        raise ValueError("找不到批次结果文件")
    for fragment in files:
        with tempfile.TemporaryDirectory() as tmp:
            payload = _unpack_fragment(fragment, Path(tmp))
            with payload.open("rb") as f:
                if hashlib.file_digest(f, "sha256").hexdigest() != fragment.name.split(".")[0]:
                    raise ValueError(f"结果文件校验失败 {fragment.name}")
            evaluation.import_run(root, payload, expected_manifest=manifest)
    return evaluation.status(root, run_id)


def list_published(root):
    return [f.parent.name for f in sorted((root / "published/runs").glob("run-*/manifest.json"))]


def list_remote(root):
    refs = _git(root, "ls-remote", "--heads", "origin", "refs/heads/experiments/*")
    ids = set()
    for line in refs.splitlines():
        parts = line.split()[1].split("/")
        if len(parts) == 5 and RUN_ID.fullmatch(parts[3]):
            ids.add(parts[3])
    return sorted(ids)


def _stop_remote(process):
    if os.name == "nt":
        taskkill = Path(os.environ["SYSTEMROOT"]) / "System32/taskkill.exe"
        try:
            subprocess.run(
                [str(taskkill), "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def _run_remote(command, env, flags, operation, timeout):
    # Remote helpers can inherit capture pipes, so use files to keep timeout cleanup bounded.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(
            command, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            creationflags=flags, start_new_session=os.name != "nt",
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                _stop_remote(process)
                raise RuntimeError(f"Git {operation} 超过 {timeout:g} 秒") from error
        stdout.seek(0)
        stderr.seek(0)
        return subprocess.CompletedProcess(command, code, stdout.read(), stderr.read())


def _git(root, *args):
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    if sys.prefix != sys.base_prefix:
        env["VIRTUAL_ENV"] = sys.prefix
    config = ["-c", "core.longpaths=true"] if os.name == "nt" else []
    command = ["git", *config, "-C", str(root), *args]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    timeout = REMOTE_TIMEOUTS.get(args[0])
    if timeout is None:
        proc = subprocess.run(command, env=env, capture_output=True, creationflags=flags)
    else:
        proc = _run_remote(command, env, flags, args[0], timeout)
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    return proc.stdout.decode("utf-8", "replace").strip()


def branch_for(batch):
    digest = Path(batch["fragment"]).name.split(".")[0]
    return f"experiments/{batch['run_id']}/{digest}"


def _verify_prepared_fragment(root, fragment):
    with tempfile.TemporaryDirectory(dir=root / ".local") as temporary:
        payload = _unpack_fragment(root / fragment, Path(temporary))
        with payload.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != fragment.name.split(".")[0]:
                raise ValueError(f"结果片段校验失败 {fragment.name}")


def push(root, run_id, *, prepared=None):
    batch = publish(root, run_id) if prepared is None else prepared
    if batch["run_id"] != run_id:
        raise ValueError("发布片段与批次不匹配")
    fragment = Path(batch["fragment"])
    ref = branch_for(batch)
    if _git(root, "ls-remote", "--heads", "origin", ref):
        return batch | {"branch": ref, "state": "published"}
    if prepared is not None:
        _verify_prepared_fragment(root, fragment)
    paths = [fragment, fragment.with_name("manifest.json")]
    if batch["runtime"] is not None:
        paths.append(Path(batch["runtime"]))
    paths.extend(Path("published/versions") / v for v in batch["versions"] if v != "baseline")
    with tempfile.TemporaryDirectory(dir=root / ".local") as temp:
        checkout = Path(temp) / "share"
        _git(root, "worktree", "add", "--quiet", "--detach", str(checkout), "HEAD")
        try:
            if (checkout / ".githooks/pre-commit").is_file():
                from .checks import CheckError, _sdk_wheel

                try:
                    _sdk_wheel(root, checkout)
                except CheckError as error:
                    raise ValueError(str(error)) from error
            for path in paths:
                target = checkout / path
                target.parent.mkdir(parents=True, exist_ok=True)
                if (root / path).is_dir():
                    shutil.copytree(root / path, target, dirs_exist_ok=True)
                else:
                    shutil.copyfile(root / path, target)
            _git(checkout, "add", "--force", "--", *map(str, paths))
            if _git(checkout, "diff", "--cached", "--name-only"):
                _git(checkout, "commit", "-m", f"chore：发布实验 {run_id}")
            _git(checkout, "push", "origin", f"HEAD:refs/heads/{ref}")
        finally:
            if not checkout.resolve().is_relative_to((root / ".local").resolve()):
                raise ValueError(f"发布工作区位置无效 {checkout}")
            _git(root, "worktree", "remove", "--force", str(checkout))
    return batch | {"branch": ref, "state": "published"}


def fetch(root, run_id):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("批次 ID 格式错误")
    refs = _git(root, "ls-remote", "--heads", "origin", f"refs/heads/experiments/{run_id}/*")
    if not refs:
        raise ValueError("远端没有这个已发布批次")
    scratch = root / ".local"
    scratch.mkdir(exist_ok=True)
    for line in refs.splitlines():
        sha, ref = line.split()
        _git(root, "fetch", "origin", ref)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            checkout = Path(temp) / "fetch"
            _git(root, "worktree", "add", "--quiet", "--detach", str(checkout), sha)
            try:
                relative = Path("published/runs") / run_id
                manifest = read_json(checkout / relative / "manifest.json")
                validate_manifest(manifest)
                if manifest["run_id"] != run_id:
                    raise ValueError("远端批次标识不匹配")
                ids = version_ids(manifest)
                for version in ids - {"baseline"}:
                    versions.verify(checkout, version)
                    path = Path("published/versions") / version
                    if not (root / path).is_dir():
                        target = root / path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(dir=target.parent) as staging:
                            copied = Path(staging) / version
                            shutil.copytree(checkout / path, copied)
                            copied.rename(target)
                engine = manifest["engine"]
                if "source_sha256" in engine:
                    source = shared_runtime(checkout, engine)
                    target = shared_runtime(root, engine)
                    read_runtime_archive(source, engine)
                    if target.is_file():
                        read_runtime_archive(target, engine)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(dir=target.parent) as staging:
                            draft = Path(staging) / target.name
                            shutil.copyfile(source, draft)
                            draft.replace(target)
                dest = root / relative
                dest.mkdir(parents=True, exist_ok=True)
                manifest_path = dest / "manifest.json"
                if manifest_path.exists() and read_json(manifest_path) != manifest:
                    raise ValueError(f"远端清单与本机批次冲突 {run_id}")
                write_json(manifest_path, manifest)
                for source in _fragments(checkout / relative):
                    target = dest / source.name
                    if not target.exists():
                        if source.suffix == ".parts":
                            with tempfile.TemporaryDirectory(dir=dest) as tmp:
                                staged = Path(tmp) / "split"
                                shutil.copytree(source, staged, symlinks=True)
                                staged.rename(target)
                            continue
                        with tempfile.NamedTemporaryFile(dir=dest, delete=False) as out, source.open("rb") as incoming:
                            shutil.copyfileobj(incoming, out)
                        Path(out.name).replace(target)
            finally:
                _git(root, "worktree", "remove", str(checkout))
    return restore(root, run_id)
