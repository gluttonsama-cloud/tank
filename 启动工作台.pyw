"""双击启动 Windows 本地工作台。"""
import argparse
import ctypes
from http.client import HTTPConnection, HTTPException
import json
import os
from pathlib import Path
import subprocess
import time
import webbrowser


def running(port, root):
    connection = HTTPConnection("127.0.0.1", port, timeout=2)
    try:
        try:
            connection.connect()
        except (ConnectionRefusedError, TimeoutError):
            return False
        connection.request("GET", "/api/workspace")
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"端口 {port} 已被其他服务占用。")
        data = json.loads(response.read(8192))
        if not isinstance(data, dict) or data.get("application") != "tank":
            raise RuntimeError(f"端口 {port} 已被其他服务占用。")
        active = Path(data["root"])
        if active != root:
            raise RuntimeError(f"端口 {port} 正在运行另一个工作台：{active}\n请用 --port 指定其他端口。")
        return True
    except (HTTPException, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"端口 {port} 未返回工作台信息。") from error
    finally:
        connection.close()


def launch(port, open_browser):
    root = Path(__file__).resolve().parent
    if not running(port, root):
        python = root / ".venv/Scripts/python.exe"
        if not python.exists():
            raise RuntimeError("请先按照根目录 README.md 安装运行环境。")
        logs = root / ".local/logs/workbench.log"
        logs.parent.mkdir(parents=True, exist_ok=True)
        with logs.open("ab") as log:
            process = subprocess.Popen(
                [str(python), "-m", "tank", "web", "--port", str(port)], cwd=root,
                env=os.environ | {"TANK_ROOT": str(root), "PYTHONUTF8": "1"},
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            )
        deadline = time.monotonic() + 20
        while not running(port, root):
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"工作台未能启动。日志位于 {logs}")
            time.sleep(0.15)
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}/#strategy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        launch(args.port, not args.no_browser)
    except (OSError, RuntimeError) as error:
        if args.no_browser:
            raise SystemExit(str(error))
        ctypes.windll.user32.MessageBoxW(None, str(error), "启动工作台", 0x10)
        raise SystemExit(1)
