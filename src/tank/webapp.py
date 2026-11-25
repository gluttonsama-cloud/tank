import json
import re
import sqlite3
import threading
import time
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from seedcup.client.viewer import TYPES, WEB, static_file

from . import evaluation, online, publication, sync, versions
from .cli import execute, list_runs, parser
from .eval_plan import list_suites
from .workspace import read_json


def overview(root):
    from .catalog import projects
    scores = root / ".local/online/team-scores.json"
    team = read_json(scores) if scores.exists() else {}
    return {"projects": projects(root), "team_scores": {key: team.get(key) for key in ("best", "recent", "refreshed_at")},
            "versions": versions.list_versions(root), "main": versions.main_version(root),
            "sources": sorted(p.parent.name for p in (root / "strategies").glob("*/submission.json")),
            "suites": [s["name"] for s in list_suites(root)], "runs": list_runs(root)}


class WorkbenchHTTP(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, root):
        self.root = root
        self.replays = {}
        self.guard = threading.Lock()
        self.next_sync = 0
        self.sync_error = None
        self.sync_running = False
        self.stop_sync = threading.Event()
        self.team_sync = {"last_received_at": None, "error": None}
        super().__init__(addr, Handler)

    def sync_loop(self, interval=300):
        while not self.stop_sync.is_set():
            self.refresh_scores()
            errors = []
            for operation in (publication.reconcile, sync.receive_updates):
                try:
                    operation(self.root)
                    if operation is sync.receive_updates:
                        self.team_sync["last_received_at"] = time.time()
                except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
                    errors.append(str(error))
            self.team_sync["error"] = "\n".join(errors) or None
            self.stop_sync.wait(interval)

    def refresh_scores(self):
        if not online.account(self.root):
            return
        with self.guard:
            if self.sync_running or time.monotonic() < self.next_sync:
                return
            self.sync_running = True
            self.next_sync = time.monotonic() + 300
        threading.Thread(target=self._sync_scores, daemon=True).start()

    def _sync_scores(self):
        from . import online
        try:
            online.refresh(self.root)
            self.sync_error = None
        except (ValueError, OSError) as error:
            self.sync_error = str(error)
        finally:
            with self.guard:
                self.sync_running = False


class Handler(BaseHTTPRequestHandler):
    def _reply(self, data, content_type="application/json; charset=utf-8", status=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _local(self):
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in hosts or (origin and origin != "http://" + host):
            self._reply({"error":"仅限本机访问"}, status=403)
            return False
        return True

    def do_GET(self):
        if not self._local():
            return
        url = urlsplit(self.path)
        root = self.server.root
        try:
            if url.path == "/api/workspace":
                return self._reply({"application": "tank", "root": str(root.resolve())})
            if url.path == "/api/overview":
                self.server.refresh_scores()
                return self._reply(overview(root) | {"account_username": (online.account(root) or {}).get("username"), "team_sync": dict(self.server.team_sync),
                                                     "sync_error": self.server.sync_error,
                                                     "score_sync_running": self.server.sync_running})
            if url.path == "/api/online/practice":
                query = parse_qs(url.query)
                if "version" not in query:
                    raise ValueError("缺少参数 version")
                return self._reply(online.history(root, query["version"][0]))
            if url.path == "/replay":
                query = parse_qs(url.query, keep_blank_values=True)
                if "online" in query:
                    match_id = query["online"][0]
                    if not re.fullmatch(r"[1-9][0-9]*", match_id):
                        raise ValueError("官网对局 ID 需为正整数")
                    # The SDK player requires a numeric URL. A leading zero separates
                    # stable official match IDs from the per-process local replay IDs.
                    index = f"0{match_id}"
                else:
                    case = query["case"][0]
                    path = evaluation.replay(root, case)["replay_path"]
                    with self.server.guard:
                        index = len(self.server.replays) + 1
                        self.server.replays[index] = Path(path)
                self.send_response(303)
                self.send_header("Location", f"/replay/{index}")
                self.end_headers()
                return
            if url.path.startswith("/api/replays/"):
                from seedcup.game.replay import load
                index = url.path.removeprefix("/api/replays/")
                if re.fullmatch(r"0[1-9][0-9]*", index):
                    path = online.practice_replay_path(root, int(index[1:]))
                    if not path.is_file():
                        return self._reply({"detail": "本机未找到这场官网练习回放"}, status=404)
                else:
                    path = self.server.replays[int(index)]
                return self._reply(load(path))
            if url.path.startswith("/replay/") and url.path[8:].isdigit():
                file = WEB / "replay.html"
            elif url.path.startswith("/static/"):
                file = static_file(url.path[8:])
            else:
                file = {"/": root / "web/index.html", "/app.js": root / "web/app.js",
                        "/style.css": root / "web/style.css"}.get(url.path)
            if file is None or not file.is_file():
                return self._reply({"error":"未找到"}, status=404)
            return self._reply(file.read_bytes(), TYPES[file.suffix])
        except (KeyError, IndexError, OSError, ValueError, RuntimeError) as exc:
            self._reply({"error":str(exc)}, status=400)

    def do_POST(self):
        if not self._local():
            return
        if self.path not in ("/api/command", "/api/account") or self.headers.get_content_type() != "application/json":
            return self._reply({"error":"请求格式错误"}, status=400)
        try:
            size = int(self.headers["Content-Length"])
            if not 0 < size < 65536:
                raise ValueError("请求大小超出范围")
            payload = json.loads(self.rfile.read(size))
            if self.path == "/api/account":
                return self._reply(online.save_account(self.server.root, payload["username"], payload["password"]))
            argv = payload["args"]
            if not isinstance(argv, list) or not argv or not all(isinstance(v,str) for v in argv):
                raise ValueError("args 须为命令参数列表")
            if argv[0] in ("check", "web", "lan", "screen") or argv[:2] in (["eval","import"], ["eval","export"]):
                raise ValueError("此操作请通过 CLI 运行")
            data = execute(self.server.root, parser().parse_args(argv))
            self._reply({"data":data})
        except (SystemExit, KeyError, TypeError, OSError, ValueError, RuntimeError) as exc:
            self._reply({"error":str(exc)}, status=HTTPStatus.BAD_REQUEST)


def serve(root, port=8765, open_browser=False):
    with WorkbenchHTTP(("127.0.0.1", port), root) as server:
        url = f"http://127.0.0.1:{server.server_port}"
        print(url, flush=True)
        if open_browser:
            threading.Timer(0.1, partial(webbrowser.open, url)).start()
        worker = threading.Thread(target=server.sync_loop, name="team-sync", daemon=True)
        worker.start()
        try:
            server.serve_forever()
        finally:
            server.stop_sync.set()
