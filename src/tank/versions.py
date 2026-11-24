import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from .workspace import read_json, write_json


BASELINE = {"id": "baseline", "name": "基础练习人机", "runtime": "sdk",
            "entrypoint": "seedcup.game.bots:BaselineBot", "note": ""}


def _version_dir(root, version_id):
    if not re.fullmatch(r"v-[0-9a-f]{64}", version_id):
        raise ValueError(f"无效的策略版本 {version_id}")
    for parent in (root / ".local/versions", root / "published/versions"):
        path = parent / version_id
        if path.is_dir():
            return path
    raise ValueError(f"找不到策略版本 {version_id}")


def snapshot_data(root: Path, version_id: str) -> Path:
    return _version_dir(root, version_id) / "files"


def describe(root: Path, version_id: str) -> dict:
    if version_id == "baseline":
        return BASELINE.copy()
    return read_json(_version_dir(root, version_id) / "version.json")


def list_versions(root: Path) -> list:
    ids = {p.parent.name for source in (".local", "published")
           for p in (root / source / "versions").glob("v-*/version.json")}
    return [BASELINE.copy(), *({k: v for k, v in describe(root, key).items() if k != "files"}
                              for key in sorted(ids))]


def _file_hashes(path):
    values = {}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"策略目录含符号链接 {file.name}")
        if file.is_file():
            with file.open("rb") as stream:
                values[file.relative_to(path).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return values


def compare(root, first, second):
    before, after = (describe(root, key) for key in (first, second))
    left, right = (record.get("files", {}) for record in (before, after))
    return {"from": first, "to": second, "added": sorted(right.keys() - left),
            "removed": sorted(left.keys() - right),
            "modified": sorted(key for key in left.keys() & right if left[key] != right[key]),
            "entrypoints": [before["entrypoint"], after["entrypoint"]]}


def _version_id(files):
    payload = json.dumps(files, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return "v-" + hashlib.sha256(payload.encode()).hexdigest()


def verify(root, version_id):
    folder = snapshot_data(root, version_id)
    files = _file_hashes(folder)
    record = describe(root, version_id)
    spec = read_json(folder / "submission.json")
    if (_version_id(files) != version_id or record["id"] != version_id or record["files"] != files
            or any(record[field] != spec[field] for field in ("runtime", "entrypoint"))):
        raise ValueError(f"快照内容校验失败 {version_id}")


def create(root, strategy, note=""):
    base = (root / "strategies").resolve()
    source = (base / strategy).resolve()
    if source.parent != base or not (source / "submission.json").is_file():
        raise ValueError("请选择 strategies 下含 submission.json 的策略目录")
    return create_from_directory(root, source, source.name, note)


def create_from_directory(root, source, name, note=""):
    from seedcup.server.evaluation_manifest import manifest
    parent = root / ".local/versions"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as temp:
        staged = Path(temp) / "snapshot"
        dest = staged / "files"
        shutil.copytree(source, dest, symlinks=True, ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", ".env", ".env.*", ".local", ".venv"))
        files = _file_hashes(dest)
        spec = read_json(dest / "submission.json")
        manifest((dest / "submission.json").read_bytes(), set(files))
        key = _version_id(files)
        record = {"id": key, "name": name, "note": note, "files": files,
                  "runtime": spec["runtime"], "entrypoint": spec["entrypoint"]}
        write_json(staged / "version.json", record)
        try:
            staged.rename(parent / key)
        except OSError:
            if not (parent / key / "version.json").is_file():
                raise
            verify(root, key)
    return describe(root, key)


def restore(root, version_id, name):
    verify(root, version_id)
    base = (root / "strategies").resolve()
    target = (base / name).resolve()
    if target.parent != base:
        raise ValueError("目标必须位于 strategies 目录下")
    shutil.copytree(snapshot_data(root, version_id), target)
    return {"strategy": name, "version_id": version_id}


def main_version(root):
    path = root / ".local/main.json"
    return read_json(path)["id"] if path.is_file() else "baseline"


def set_main(root, version_id):
    info = describe(root, version_id)
    write_json(root / ".local/main.json", {"id": info["id"]})
    return info


def load_bot(root: Path, version_id: str):
    if version_id == "baseline":
        from seedcup.game.bots import BaselineBot
        return BaselineBot()
    from .process_bot import ProcessBot
    return ProcessBot(root, snapshot_data(root, version_id), version_id)
