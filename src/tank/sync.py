"""Publish immutable batch fragments and their pinned strategy sources."""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import evaluation, versions
from .eval_plan import RUN_ID, publish_runtime, read_runtime_archive, shared_runtime, validate_manifest
from .eval_store import database, read_manifest
from .workspace import read_json, write_json

PART_BYTES = 32 * 1024 * 1024


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
    ids = {*manifest["candidates"], manifest["baseline"], *manifest["suite"]["opponents"]}
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
    for version in {*manifest["candidates"], manifest["baseline"], *manifest["suite"]["opponents"]} - {"baseline"}:
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


def _git(root, *args):
    env = os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    if sys.prefix != sys.base_prefix:
        env["VIRTUAL_ENV"] = sys.prefix
    config = ["-c", "core.longpaths=true"] if os.name == "nt" else []
    proc = subprocess.run(["git", *config, "-C", str(root), *args], env=env, capture_output=True,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    return proc.stdout.decode("utf-8", "replace").strip()


def push(root, run_id):
    batch = publish(root, run_id)
    fragment = Path(batch["fragment"])
    ref = f"experiments/{run_id}/{fragment.name.split('.')[0]}"
    if _git(root, "ls-remote", "--heads", "origin", ref):
        return batch | {"branch": ref, "state": "published"}
    paths = [fragment, fragment.with_name("manifest.json")]
    if batch["runtime"] is not None:
        paths.append(Path(batch["runtime"]))
    paths.extend(Path("published/versions") / v for v in batch["versions"] if v != "baseline")
    with tempfile.TemporaryDirectory(dir=root / ".local") as temp:
        checkout = Path(temp) / "share"
        _git(root, "worktree", "add", "--quiet", "--detach", str(checkout), "HEAD")
        try:
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
            _git(root, "worktree", "remove", str(checkout))
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
                ids = {*manifest["candidates"], manifest["baseline"], *manifest["suite"]["opponents"]}
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
