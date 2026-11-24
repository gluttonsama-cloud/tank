"""Run repository training scripts as recorded local background jobs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

from .training import TrainingStore
from .versions import create_from_directory, snapshot_data
from .workspace import read_json, write_json


def _job_dir(root: Path, job_id: str) -> Path:
    if not re.fullmatch(r"train-[0-9a-f]{32}", job_id):
        raise ValueError("训练作业 ID 格式错误")
    return root / ".local/training/jobs" / job_id


def _freeze(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    with target.open("rb") as copied:
        return hashlib.file_digest(copied, "sha256").hexdigest()


def _freeze_python(source: Path, target: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(source.rglob("*.py")):
        if not path.resolve().is_relative_to(source):
            raise ValueError(f"源文件不在仓库目录内 {path}")
        relative = path.relative_to(source)
        hashes[relative.as_posix()] = _freeze(path, target / relative)
    return hashes


def _command_environment(workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("SEEDCUP_USERNAME", "SEEDCUP_PASSWORD", "TANK_LAN_TOKEN"):
        env.pop(name, None)
    return env | {"PYTHONPATH": str(workspace / "workbench"), "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def start(root: Path, script: str, config_id: str, inputs: list[Path], params: dict) -> dict:
    root = root.resolve()
    store = TrainingStore(root)
    config = store.read_config(config_id)
    source = root / "training"
    entry = (source / script).resolve()
    if entry.suffix != ".py" or not entry.is_relative_to(source) or not entry.is_file():
        raise ValueError("请选择 training/ 下的 Python 训练脚本")
    if not isinstance(params, dict):
        raise ValueError("params 必须为 JSON 对象")
    json.dumps(params, allow_nan=False)
    input_paths = [file.resolve() for file in inputs]
    for file in input_paths:
        if not file.is_file():
            raise ValueError(f"训练输入文件不存在 {file}")

    job_id = "train-" + uuid4().hex
    folder = _job_dir(root, job_id)
    folder.mkdir(parents=True)
    created = {"job_id": job_id, "status": "queued", "created_at": time.time()}
    write_json(folder / "queued.json", created)
    script_name = entry.relative_to(source).as_posix()
    try:
        sources = _freeze_python(source, folder / "training")
        workbench = _freeze_python(Path(__file__).resolve().parent, folder / "workbench/tank")
        records = []
        for index, file in enumerate(input_paths):
            relative = Path("inputs") / f"{index:03d}" / file.name
            digest = _freeze(file, folder / relative)
            records.append({"source": str(file), "path": relative.as_posix(), "sha256": digest})
        request = {
            "schema_version": 1, "job_id": job_id, "script": script_name, "config_id": config_id,
            "config": config, "params": params, "inputs": records, "output": "output",
            "sources": sources, "workbench": workbench,
            "runtime": {"python": sys.version, "numpy": importlib.metadata.version("numpy"),
                        "seedcup": importlib.metadata.version("seedcup")},
        }
        write_json(folder / "request.json", request)
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with (folder / "worker.log").open("ab") as log:
            worker = subprocess.Popen(
                [sys.executable, "-m", "tank.jobs", str(root), job_id], cwd=folder,
                env=_command_environment(folder), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=flags, start_new_session=os.name != "nt",
            )
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        write_json(folder / "finished.json", created | {"status": "failed", "error": str(error), "finished_at": time.time()})
        raise
    write_json(folder / "launch.json", {"pid": worker.pid})
    return created | {"config_id": config_id, "script": script_name}


def _output_path(folder: Path, name: str, field: str) -> Path:
    path = (folder / name).resolve()
    if path == folder or not path.is_relative_to(folder):
        raise ValueError(f"{field} 必须指向 output/ 内的文件或目录")
    return path


def _register_output(root: Path, request: dict, output: Path) -> dict:
    result = read_json(output / "result.json")
    weights = _output_path(output, result["weights"], "weights")
    store = TrainingStore(root)
    config_id = request["config_id"]
    if store.read_config(config_id) != request["config"]:
        raise ValueError(f"训练配置发生变化 {config_id}")
    if not weights.is_file():
        raise ValueError(f"找不到导出权重 {result['weights']}")
    step = result["step"]
    if type(step) is not int or step < 0:
        raise ValueError("step 必须为非负整数")
    snapshot_id = None
    if "strategy" in result:
        strategy = _output_path(output, result["strategy"], "strategy")
        if not weights.is_relative_to(strategy):
            raise ValueError("导出的策略目录必须包含权重")
        snapshot = create_from_directory(root, strategy, f"训练 {Path(request['script']).stem}", f"第 {step} 步")
        snapshot_id = snapshot["id"]
        weights = snapshot_data(root, snapshot_id) / weights.relative_to(strategy)
    checkpoint_id = store.register_checkpoint(config_id, weights, step=step, snapshot_id=snapshot_id)
    return {"checkpoint_id": checkpoint_id, "snapshot_id": snapshot_id, "metrics": result.get("metrics", {})}


def _run(root: Path, job_id: str) -> int:
    folder = _job_dir(root, job_id)
    state = read_json(folder / "queued.json") | {"status": "running", "started_at": time.time(), "pid": os.getpid()}
    write_json(folder / "running.json", state)
    try:
        request_file = folder / "request.json"
        request = read_json(request_file)
        output = folder / request["output"]
        output.mkdir()
        script = folder / "training" / request["script"]
        process = subprocess.run([sys.executable, "-B", "-u", str(script), str(request_file)],
                                 cwd=folder, env=_command_environment(folder), stdin=subprocess.DEVNULL,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if process.returncode:
            raise RuntimeError(f"训练脚本退出码 {process.returncode}")
        result = _register_output(root, request, output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        traceback.print_exc()
        write_json(folder / "finished.json", state | {"status": "failed", "error": str(error), "finished_at": time.time()})
        return 1
    write_json(folder / "finished.json", state | result | {"status": "completed", "finished_at": time.time()})
    return 0


def _process_exists(pid: int) -> bool:
    if os.name == "nt":
        from ctypes import WinDLL, byref, wintypes

        kernel = WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def status(root: Path, job_id: str) -> dict:
    folder = _job_dir(root, job_id)
    # Publish each state once so concurrent readers never hold open a file that must be replaced.
    finished, running, queued = (folder / f"{name}.json" for name in ("finished", "running", "queued"))
    state = read_json(finished if finished.is_file() else running if running.is_file() else queued)
    launch = folder / "launch.json"
    if state["status"] in ("queued", "running"):
        pid = state.get("pid")
        if pid is None and launch.is_file():
            pid = read_json(launch)["pid"]
        if pid is not None and not _process_exists(pid):
            state = read_json(finished) if finished.is_file() else state | {
                "status": "interrupted", "error": "训练作业进程已退出",
            }
    request = folder / "request.json"
    if request.is_file():
        spec = read_json(request)
        state |= {"script": spec["script"], "config_id": spec["config_id"], "request": str(request)}
    return state


def logs(root: Path, job_id: str, lines: int = 50) -> dict:
    if lines < 1:
        raise ValueError("lines 必须为正整数")
    path = _job_dir(root, job_id) / "worker.log"
    chunks = []
    with path.open("rb") as stream:
        end = position = stream.seek(0, os.SEEK_END)
        remaining = lines + 1
        while remaining > 0 and position > max(0, end - 256_000):
            length = min(8192, position - max(0, end - 256_000))
            position -= length
            stream.seek(position)
            block = stream.read(length)
            chunks.append(block)
            remaining -= block.count(b"\n")
    contents = b"".join(reversed(chunks)).decode("utf-8", errors="replace").splitlines()
    return {"job_id": job_id, "path": str(path), "lines": contents[-lines:]}


def list_jobs(root: Path, limit: int = 30) -> list[dict]:
    from heapq import nlargest
    if limit < 1:
        raise ValueError("limit 必须为正整数")
    statuses = (root / ".local/training/jobs").glob("train-*/queued.json")
    recent = nlargest(limit, statuses, key=lambda path: path.stat().st_mtime_ns)
    results = [status(root, path.parent.name) for path in recent]
    return sorted(results, key=lambda item: item["created_at"], reverse=True)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="jobs_action", required=True)
    config = commands.add_parser("configure", help="登记训练配置")
    config.add_argument("--teacher", help="教师策略版本")
    config.add_argument("--opponent", action="append", required=True, metavar="VERSION=WEIGHT", help="陪练与采样权重，可重复")
    config.add_argument("--seed", type=int, required=True, help="陪练采样种子")
    job = commands.add_parser("start", help="启动后台训练")
    job.add_argument("script", help="相对于 training/ 的 Python 脚本")
    job.add_argument("--config", required=True, help="训练配置 ID")
    job.add_argument("--input", action="append", default=[], type=Path, help="训练输入文件，可重复")
    job.add_argument("--params", default="{}", help="传给训练脚本的 JSON 参数")
    status_parser = commands.add_parser("status", help="查询训练状态")
    status_parser.add_argument("job_id")
    log_parser = commands.add_parser("logs", help="读取训练日志")
    log_parser.add_argument("job_id")
    log_parser.add_argument("--lines", type=int, default=50)
    commands.add_parser("list", help="列出最近的训练作业").add_argument("--limit", type=int, default=30)
    for command in commands.choices.values():
        command.add_argument("--json", action="store_true", help="输出 JSON")


def dispatch(root: Path, args: argparse.Namespace):
    if args.jobs_action == "configure":
        pool = {}
        for item in args.opponent:
            version, separator, weight = item.partition("=")
            if not separator or not version or version in pool:
                raise ValueError(f"--opponent 应为不重复的 VERSION=WEIGHT，收到 {item}")
            pool[version] = int(weight)
        config_id = TrainingStore(root).register_config(
            teacher_snapshot=args.teacher, opponent_pool=pool, sampling_seed=args.seed,
        )
        return {"config_id": config_id}
    if args.jobs_action == "start":
        return start(root, args.script, args.config, args.input, json.loads(args.params))
    if args.jobs_action == "status":
        return status(root, args.job_id)
    if args.jobs_action == "logs":
        return logs(root, args.job_id, args.lines)
    return list_jobs(root, args.limit)


if __name__ == "__main__":
    sys.exit(_run(Path(sys.argv[1]), sys.argv[2]))
