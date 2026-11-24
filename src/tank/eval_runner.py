"""Run one official match in its own seeded, interruptible process."""

from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .eval_plan import engine_fields, engine_matches, runner_identity, runtime_directory


class MatchFailure(RuntimeError):
    def __init__(self, message, *, timeout=False):
        super().__init__(message)
        self.timeout = timeout


def _stop_match(process, log):
    if os.name == "nt":
        subprocess.run(
            [str(Path(os.environ["SYSTEMROOT"]) / "System32" / "taskkill.exe"), "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_isolated(root, spec, timeout, replay, log_path, cancelled=lambda: False):
    seed = spec["strategy_seed"]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=log_path.parent) as directory, log_path.open("ab") as log:
        request = Path(directory) / "request.json"
        output = Path(directory) / "result.json"
        request.write_text(json.dumps(spec | {"replay": replay}), encoding="utf-8")
        env = os.environ.copy()
        env.pop("TANK_LAN_TOKEN", None)
        process = subprocess.Popen(
            [sys.executable, "-P", "-B", "-m", "tank.eval_runner", str(root), str(request), str(output)],
            cwd=root,
            env=env | {
                "PYTHONHASHSEED": str(seed), "PYTHONUTF8": "1", "TANK_STRATEGY_SEED": str(seed),
                "PYTHONPATH": str(runtime_directory(root, spec)),
            },
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            start_new_session=os.name != "nt",
        )
        try:
            while process.poll() is None:
                if cancelled():
                    return None
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise MatchFailure(f"对局运行超过 {timeout:g} 秒", timeout=True)
                try:
                    process.wait(timeout=min(remaining, 0.25))
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if process.poll() is None:
                _stop_match(process, log)
        if process.returncode:
            raise MatchFailure(f"对局进程退出码 {process.returncode}，日志 {log_path}")
        response = json.loads(output.read_text(encoding="utf-8"))
        if "timeout" in response:
            raise MatchFailure(response["timeout"], timeout=True)
        return response["match"], time.monotonic() - started


def _run_match(root, settings):
    import numpy

    from .simulation import run_match
    from .versions import load_bot

    engine = engine_fields(settings)
    if not engine_matches(runner_identity(root, cpp="cpp_sha256" in engine), engine):
        raise ValueError("对局子进程的 SDK、依赖或运行器与批次清单不同")
    seed = settings["strategy_seed"]
    random.seed(seed)
    numpy.random.seed(seed)
    bots = [load_bot(root, settings[side]) for side in ("red", "blue")]
    try:
        return run_match(*bots, settings["seed"], settings["game_no"], replay=settings["replay"])
    finally:
        for bot in bots:
            close = getattr(bot, "close", None)
            if close is not None:
                close()


if __name__ == "__main__":
    root_arg, request_arg, output_arg = map(Path, sys.argv[1:])
    spec_arg = json.loads(request_arg.read_text(encoding="utf-8"))
    try:
        response = {"match": _run_match(root_arg, spec_arg)}
    except TimeoutError as error:
        response = {"timeout": str(error)}
    output_arg.write_text(json.dumps(response, allow_nan=False), encoding="utf-8")
