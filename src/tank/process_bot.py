"""Run a local strategy with the competition's JSON-lines protocol."""
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from seedcup.server.evaluation_manifest import manifest

from .versions import _file_hashes, _version_id


CPP_FLAGS = ("-std=c++17", "-O2")


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def compiler_fingerprint():
    compiler = shutil.which("g++")
    if compiler is None:
        raise ValueError("C++ 策略需要 g++")
    executable = Path(compiler).resolve()
    with executable.open("rb") as binary:
        executable_hash = hashlib.file_digest(binary, "sha256").hexdigest()
    fingerprint = {
        "compiler": executable_hash,
        "flags": CPP_FLAGS,
        "target": subprocess.check_output([compiler, "-dumpmachine"], timeout=10, creationflags=NO_WINDOW).decode().strip(),
        "version": subprocess.check_output([compiler, "-dumpfullversion"], timeout=10, creationflags=NO_WINDOW).decode().strip(),
    }
    identity = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    return compiler, identity


def _read_lines(stream, replies):
    try:
        while line := stream.readline(4097):
            replies.put(line)
            if len(line) > 4096:
                break
    finally:
        replies.put(b"")


class ProcessBot:
    def __init__(self, root, folder, version_id, *, match_seconds=15):
        self.root = root
        self.folder = folder
        self.version_id = version_id
        self.match_seconds = match_seconds
        self.process = None
        self.scratch = None
        self.replies = queue.SimpleQueue()

    def command(self, source):
        paths = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()}
        spec = manifest((source / "submission.json").read_bytes(), paths)
        entry = spec["entrypoint"]
        if spec["runtime"] == "python":
            cmd = [sys.executable, "-B", "-u"]
            if spec["class"]:
                return cmd + [str(Path(__file__).with_name("_sdk_bot.py")), entry + ":" + spec["class"]]
            return cmd + [entry]
        compiler, compiler_id = compiler_fingerprint()
        build = self.root / ".local/build" / self.version_id / compiler_id
        build.mkdir(parents=True, exist_ok=True)
        binary = build / ("bot.exe" if os.name == "nt" else "bot")
        if not binary.is_file():
            with tempfile.TemporaryDirectory(dir=build) as tmp:
                target = Path(tmp) / binary.name
                subprocess.run([compiler, *CPP_FLAGS, entry, *spec["sources"], "-o", str(target)],
                               cwd=source, check=True, timeout=45, creationflags=NO_WINDOW)
                try:
                    os.link(target, binary)
                except FileExistsError:
                    pass  # Another shard published the same compiled snapshot.
        return [str(binary)]

    def on_match_start(self, info):
        local = self.root / ".local"
        local.mkdir(exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="match-", dir=local)
        cwd = Path(self.scratch.name)
        shutil.copytree(self.folder, cwd, dirs_exist_ok=True)
        if self.version_id.startswith("v-") and _version_id(_file_hashes(cwd)) != self.version_id:
            raise ValueError(f"对局策略快照内容校验失败 {self.version_id}")
        command = self.command(cwd)
        self.started = time.monotonic()
        logs = self.root / ".local/logs"
        logs.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix="strategy-", suffix=".log", dir=logs, delete=False) as log:
            self.log = Path(log.name)
            env = {k:v for k,v in os.environ.items() if k.upper() in {
                "PATH","SYSTEMROOT","WINDIR","TEMP","TMP","HOME","USERPROFILE","LANG","LC_ALL",
                "TANK_STRATEGY_SEED","PYTHONHASHSEED","VIRTUAL_ENV","SYSTEMDRIVE","PROGRAMDATA",
                "LOCALAPPDATA","APPDATA"}}
            env["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=log, creationflags=NO_WINDOW)
        threading.Thread(target=_read_lines, args=(self.process.stdout, self.replies), daemon=True).start()
        self.send({"type": "match_start", **info})

    def send(self, data):
        try:
            self.process.stdin.write(json.dumps(data, separators=(",", ":")).encode() + b"\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.close(failed=True)
            raise RuntimeError(f"策略进程已退出，日志 {self.log}") from exc

    def decide(self, obs):
        start = time.monotonic()
        self.send({"type": "obs", **obs})
        deadline = self.started + 3 if obs["frame"] == 0 else start + 0.1
        if self.match_seconds is not None:
            deadline = min(deadline, self.started + self.match_seconds)
        try:
            raw = self.replies.get(timeout=max(0, deadline - time.monotonic()))
            if not raw or len(raw) > 4096:
                raise ValueError(f"动作缺失或过长，日志 {self.log}")
            action = json.loads(raw)
            if action.get("type") != "action" or action.get("frame") != obs["frame"]:
                raise ValueError("策略返回了错误的动作帧")
            return action
        except queue.Empty as exc:
            self.close(failed=True)
            raise TimeoutError(f"策略第 {obs['frame']} 帧超时，日志 {self.log}") from exc
        except (ValueError, AttributeError):
            self.close(failed=True)
            raise

    def on_match_end(self, result):
        self.send({"type": "match_end", "result": result})
        try:
            self.process.stdin.close()
            if self.process.wait(timeout=1) != 0:
                raise RuntimeError(f"策略退出异常，日志 {self.log}")
        finally:
            self.close()

    def close(self, failed=False):
        if self.process is None:
            if self.scratch is not None:
                self.scratch.cleanup()
                self.scratch = None
            return
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()
        self.process = None
        self.scratch.cleanup()
        self.scratch = None
        if not failed and self.log.stat().st_size == 0:
            self.log.unlink()
