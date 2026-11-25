"""Run the team's foreground practice queue from one SeedCup client."""

from contextlib import ExitStack, suppress
from getpass import getpass, GetPassWarning
import hmac
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
import json
import os
from pathlib import Path
import re
from secrets import token_urlsafe
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Condition, Thread
import time
import traceback
from uuid import uuid4
import warnings

from . import lan, online, practice_priority, versions
from .eval_plan import json_text, required
from .eval_store import shard_lock
from .process_bot import ProcessBot
from .workspace import read_json, write_json


TOKEN_ENV = "TANK_PRACTICE_TOKEN"
PORT = 8767
MAX_REPLAY_BYTES = 32 * 1024 * 1024
MAX_HTTP_BYTES = 64 * 1024
MAX_RELAY_RESPONSE = 2 * 1024 * 1024
STEP_WAIT = 2
RELAY_ATTACH_TIMEOUT = 6
POLL_SECONDS = 2
STOP_GRACE_SECONDS = 3
VERSION_ID = re.compile(r"v-[0-9a-f]{64}")
FAILED = (OSError, ValueError, RuntimeError, subprocess.SubprocessError)
NETWORK_ERRORS = (OSError, HTTPException)


def _folder(root):
    return root / ".local/online/practice-host"


def _token_path(root):
    return _folder(root) / "token.json"


def _token(root, *, create=False):
    token = os.environ.get(TOKEN_ENV)
    if token:
        return token.encode("utf-8")
    path = _token_path(root)
    if not path.is_file():
        if not create:
            raise ValueError(f"请用 tank sample host-use 设置组内练习令牌，或设置 {TOKEN_ENV}")
        write_json(path, {"token": token_urlsafe(32)})
    token = required(read_json(path), "token")
    if not isinstance(token, str) or not token:
        raise ValueError("本机练习令牌文件的 token 无效")
    return token.encode("utf-8")


def _validated_id(value, pattern, label):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{label}格式无效")
    return value


def _practice_spec(data):
    version_id, level, games, seed = (required(data, field) for field in ("version_id", "level", "games", "seed"))
    _validated_id(version_id, VERSION_ID, "策略快照 ID")
    if level not in online.PRACTICE_LEVELS:
        raise ValueError("level 不是有效的官网练习档位")
    if type(games) is not int or games not in (1, 2):
        raise ValueError("games 必须为 1 或 2")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
        raise ValueError("seed 必须为 0 到 2147483647 的整数或 null")
    return {"version_id": version_id, "level": level, "games": games, "seed": seed}


def _snapshot_exists(root, version_id):
    return any((root / source / "versions" / version_id).is_dir() for source in (".local", "published"))


def _tag(request_id):
    return f"host:{request_id}"


class Relay:
    def __init__(self):
        self.changed = Condition()
        self.sequence = 0
        self.ack = 0
        self.output = None
        self.event = None
        self.contact = None
        self.closed = False

    def attach(self):
        def client_ready():
            return self.contact is not None and time.monotonic() - self.contact < STEP_WAIT + 1

        with self.changed:
            self.changed.wait_for(lambda: self.closed or client_ready(), RELAY_ATTACH_TIMEOUT)
            if self.closed or not client_ready():
                raise online.OnlineError("发起工作台未连接，无法转发策略动作")

    def step(self, ack, output):
        with self.changed:
            if type(ack) is not int or ack < self.ack or ack > self.sequence:
                raise ValueError("策略转发序号不匹配")
            if ack != self.ack:
                kind = self.event["kind"]
                if kind == "obs":
                    if not isinstance(output, dict) or set(output) != {"forward", "turn", "fire"}:
                        raise ValueError("策略动作字段无效")
                elif output is not None:
                    raise ValueError("只有 OBS 可以返回动作")
                self.ack, self.output = ack, output
            elif output != self.output:
                raise ValueError("相同转发序号的动作不一致")
            self.contact = time.monotonic()
            self.changed.notify_all()
            self.changed.wait_for(lambda: self.closed or self.sequence > ack, STEP_WAIT)
            return {"event": self.event if not self.closed and self.sequence > ack else None, "closed": self.closed}

    def exchange(self, kind, payload, timeout):
        with self.changed:
            if self.closed:
                raise online.OnlineError("发起工作台已断开策略转发")
            self.sequence += 1
            self.event = {"sequence": self.sequence, "kind": kind, "payload": payload}
            self.changed.notify_all()
            self.changed.wait_for(lambda: self.closed or self.ack == self.sequence, timeout)
            if self.closed or self.ack != self.sequence:
                raise online.OnlineError("发起工作台未及时响应策略转发")
            return self.output

    def close(self):
        with self.changed:
            self.closed = True
            self.changed.notify_all()


class RemoteBot:
    def __init__(self, relay):
        self.relay = relay

    def on_match_start(self, message):
        self.relay.exchange("start", message, 6)

    def decide(self, message):
        return self.relay.exchange("obs", message, 3.5 if message["frame"] == 0 else 0.25)

    def on_match_end(self, result):
        self.relay.exchange("end", result, 2)

    def close(self):
        self.relay.close()


class Coordinator:
    def __init__(self, root, token, stop_flag=None):
        self.root = root
        self.token = token
        self.ready = Condition()
        self.nonces = {}
        self.next_nonce_sweep = 0
        self.relays = {}
        self.pending = ExitStack()
        self.pending_team = None
        self.worker = Thread(target=self._run, name="practice-host")
        self.stopping = False
        self.stop_flag = stop_flag
        folder = _folder(root)
        folder.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(folder / "queue.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT NOT NULL UNIQUE, spec TEXT NOT NULL,
                status TEXT NOT NULL, result TEXT, error TEXT
            )
        """)
        with self.db:
            for row in self.db.execute("SELECT request_id, spec FROM requests WHERE status = 'running'").fetchall():
                spec = json.loads(row["spec"])
                result = self._saved_result(row["request_id"], spec)
                self._finish(row["request_id"], "interrupted", result, "主机在练习期间停止，请核对官网对局后再发起新的练习")

    def __enter__(self):
        return self

    def start(self):
        self.worker.start()

    def __exit__(self, *_):
        self.stop()
        if self.worker.ident is not None:
            self.worker.join()
        self.pending.close()
        self.db.close()

    def stop(self):
        with self.ready:
            self.stopping = True
            self.ready.notify()

    def stop_requested(self):
        return self.stopping or (self.stop_flag is not None and self.stop_flag.is_file())

    def _saved_result(self, request_id, spec):
        folder = self.root / ".local/online/practice" / spec["version_id"] / spec["level"]
        matches = []
        for path in sorted(folder.glob("*.json")):
            record = online._read_practice_record(path)
            if record.get("case_tag") == _tag(request_id):
                matches.append(record)
        return {"version_id": spec["version_id"], "level": spec["level"], "matches": matches,
                "room": matches[0]["room"] if matches else None}

    def _finish(self, request_id, status, result, error):
        self.db.execute(
            "UPDATE requests SET status = ?, result = ?, error = ? WHERE request_id = ?",
            (status, json_text(result), error, request_id),
        )

    def _next(self):
        return self.db.execute("SELECT * FROM requests WHERE status = 'queued' ORDER BY rowid LIMIT 1").fetchone()

    def _mark_pending(self, team_id):
        if self.pending_team != team_id:
            self.pending.close()
            self.pending_team = None
            self.pending.enter_context(practice_priority.queue_pending(self.root, team_id))
            self.pending_team = team_id

    def _run(self):
        while True:
            with self.ready:
                self.ready.wait_for(lambda: self.stopping or self._next() is not None)
                if self.stop_requested():
                    return
                row = self._next()
                request_id, spec = row["request_id"], json.loads(row["spec"])
                relay = self.relays.setdefault(request_id, Relay()) if spec.pop("relay", False) else None
                with self.db:
                    self.db.execute("UPDATE requests SET status = 'running' WHERE request_id = ?", (request_id,))
            try:
                client = online.OnlineClient.login(self.root)
                self._mark_pending(client.user["id"])
                bot = {}
                if relay is not None:
                    relay.attach()
                    bot["bot_factory"] = lambda: RemoteBot(relay)
                result = online.practice(self.root, **spec, client=client, case_tag=_tag(request_id), **bot)
                status, failure = "completed", None
            except FAILED as error:
                result = self._saved_result(request_id, spec)
                status, failure = "failed", str(error)
                if getattr(error, "room", None) is not None:
                    result["room"] = error.room
                if getattr(error, "close_error", None) is not None:
                    result["close_error"] = error.close_error
                if isinstance(error, online.RequestUncertain):
                    status = "interrupted"
                    failure = f"{failure}，请核对官网对局后再发起新的练习"
            except Exception as error:
                # Preserve the queue and expose unexpected implementation failures in the host console.
                traceback.print_exc()
                result = self._saved_result(request_id, spec)
                status, failure = "interrupted", f"练习主机异常（{type(error).__name__}），请核对官网对局"
            with self.ready:
                with self.db:
                    self._finish(request_id, status, result, failure)
                if relay is not None:
                    relay.close()
                    del self.relays[request_id]
                if self.stop_requested() or self._next() is None:
                    self.pending.close()
                    self.pending_team = None

    def verify(self, action, timestamp, nonce, signature, body):
        if (not re.fullmatch(r"[0-9]{10}", timestamp) or not lan.NONCE.fullmatch(nonce)
                or not re.fullmatch(r"[0-9a-f]{64}", signature)):
            return False
        now, issued = time.time(), int(timestamp)
        if abs(now - issued) > lan.SIGNATURE_AGE:
            return False
        expected = lan._signature(self.token, "request", f"{action}:{timestamp}", nonce, body)
        if not hmac.compare_digest(signature, expected):
            return False
        with self.ready:
            if now >= self.next_nonce_sweep:
                self.nonces = {key: expiry for key, expiry in self.nonces.items() if expiry >= now}
                self.next_nonce_sweep = now + 30
            if nonce in self.nonces:
                return False
            self.nonces[nonce] = issued + lan.SIGNATURE_AGE
        return True

    def _response(self, request_id, row):
        payload = {"request_id": request_id, "status": row["status"], "relay": json.loads(row["spec"]).get("relay", False)}
        if row["error"] is not None:
            payload["error"] = row["error"]
        if row["result"] is not None:
            result = json.loads(row["result"])
            for record in result["matches"]:
                record.pop("replay_path", None)
                record["replay_saved"] = online.practice_replay_path(self.root, record["id"]).is_file()
            payload["result"] = result
        return payload

    def handle(self, action, request):
        request_id = _validated_id(required(request, "request_id"), lan.NONCE, "练习请求 ID")
        with self.ready:
            row = self.db.execute("SELECT * FROM requests WHERE request_id = ?", (request_id,)).fetchone()
            if action == "/practice/submit":
                if set(request) - {"request_id", "version_id", "level", "games", "seed"}:
                    raise ValueError("练习请求只接受快照 ID 和练习参数")
                spec = _practice_spec(request)
                if row is not None:
                    if spec != _practice_spec(json.loads(row["spec"])):
                        raise ValueError("该练习请求 ID 对应不同的快照或参数")
                    return self._response(request_id, row)
                if self.stop_requested():
                    raise ValueError("练习主机正在停止")
                version_id = spec["version_id"]
                spec["relay"] = not _snapshot_exists(self.root, version_id)
                if self.stop_requested():
                    raise ValueError("练习主机正在停止")
                with self.db:
                    self.db.execute(
                        "INSERT INTO requests (request_id, spec, status) VALUES (?, ?, 'queued')",
                        (request_id, json_text(spec)),
                    )
                self.ready.notify()
                return {"request_id": request_id, "status": "queued", "relay": spec["relay"]}
            if row is None:
                return {"request_id": request_id, "status": "not_found"}
            if action == "/practice/result":
                return self._response(request_id, row)
            if action in ("/practice/step", "/practice/cancel"):
                if not json.loads(row["spec"]).get("relay"):
                    raise ValueError("此请求使用主机快照")
                if row["status"] not in ("queued", "running"):
                    return self._response(request_id, row)
                relay = self.relays.setdefault(request_id, Relay())
                if action == "/practice/cancel":
                    relay.close()
                    return {"request_id": request_id, "status": row["status"]}
                status = row["status"]
            else:
                match_id = required(request, "match_id")
                result = json.loads(row["result"]) if row["result"] is not None else None
                record = next((item for item in result["matches"] if item["id"] == match_id), None) if result else None
                if type(match_id) is not int or record is None:
                    raise ValueError("此请求中没有该官网对局")
        if action == "/practice/step":
            return {"request_id": request_id, "status": status} | relay.step(required(request, "ack"), required(request, "output"))
        path = online.practice_replay_path(self.root, match_id)
        if not path.is_file():
            saved = online._read_practice_record(online._practice_path(self.root, record))
            if saved.get("case_tag") != _tag(request_id) or saved["result"] != record["result"]:
                raise ValueError("主机上的官网对局记录与练习请求不一致")
            online._archive_practice_replay(self.root, online.OnlineClient.login(self.root), saved, retry_pending=True)
        if path.stat().st_size > MAX_REPLAY_BYTES:
            raise ValueError("官网回放超过单次回传大小上限")
        return {"replay": lan._encode(path.read_bytes())}


class _ServerStopped(Exception):
    pass


def _server(coordinator, host, port):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def setup(self):
            self.request.settimeout(30)
            super().setup()

        def log_message(self, pattern, *args):
            pass

        def do_POST(self):
            if self.path not in ("/practice/submit", "/practice/result", "/practice/replay", "/practice/step", "/practice/cancel"):
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_HTTP_BYTES:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            nonce = self.headers.get("X-Tank-Nonce", "")
            if len(body) != length or not coordinator.verify(
                self.path, self.headers.get("X-Tank-Time", ""), nonce,
                self.headers.get("X-Tank-Signature", ""), body,
            ):
                self.send_error(403)
                return
            try:
                request = json.loads(body)
                if not isinstance(request, dict):
                    raise ValueError("请求必须为 JSON 对象")
                response, status = coordinator.handle(self.path, request), 200
            except (OSError, ValueError) as error:
                response, status = {"error": str(error)}, 400
            payload = json_text(response).encode()
            with suppress(BrokenPipeError, ConnectionResetError):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Tank-Signature", lan._signature(coordinator.token, "response", str(status), nonce, payload))
                self.end_headers()
                self.wfile.write(payload)

    class Server(ThreadingHTTPServer):
        daemon_threads = False
        drain_deadline = None

        def service_actions(self):
            if coordinator.stop_requested():
                coordinator.stop()
                if coordinator.worker.is_alive():
                    return
                if self.drain_deadline is None:
                    # The requester's next poll can still import the completed practice.
                    self.drain_deadline = time.monotonic() + STOP_GRACE_SECONDS
                elif time.monotonic() >= self.drain_deadline:
                    raise _ServerStopped

    return Server((host, port), Handler)


def _server_info(root):
    return _folder(root) / "server.json"


def _stop_flag(root, instance):
    _validated_id(instance, lan.NONCE, "练习主机实例 ID")
    return _folder(root) / f"stop-{instance}"


def serve(root, *, host="127.0.0.1", port=PORT, token=None, instance=None):
    instance = instance if instance is not None else uuid4().hex
    stop_flag = _stop_flag(root, instance)
    with shard_lock(_folder(root), "server") as acquired:
        if not acquired:
            raise ValueError("此工作台已启动练习主机")
        info_path = _server_info(root)
        info_path.unlink(missing_ok=True)
        info = {"host": host, "instance": instance}
        if token is None:
            token = _token(root, create=True)
            if not os.environ.get(TOKEN_ENV):
                info["token_path"] = str(_token_path(root))
                print(f"组内令牌文件 {info['token_path']}", file=sys.stderr, flush=True)
        try:
            with Coordinator(root, token, stop_flag) as coordinator, _server(coordinator, host, port) as server:
                coordinator.start()
                info["port"] = server.server_port
                write_json(info_path, info)
                print(f"练习主机正在监听 {host}:{server.server_port}", file=sys.stderr, flush=True)
                try:
                    server.serve_forever()
                except (KeyboardInterrupt, _ServerStopped):
                    pass
                finally:
                    coordinator.stop()
        finally:
            info_path.unlink(missing_ok=True)
            stop_flag.unlink(missing_ok=True)
    return _public_info(info, "stopped")


def _public_info(info, status):
    return {"status": status, **{key: value for key, value in info.items() if key != "instance"}}


def launch(root, *, host="127.0.0.1", port=PORT):
    folder = _folder(root)
    info_path = _server_info(root)
    with shard_lock(folder, "launch") as launching:
        if not launching:
            raise ValueError("练习主机正在启动")
        with shard_lock(folder, "server") as available:
            if not available:
                if not info_path.is_file():
                    raise ValueError("练习主机正在启动")
                info = read_json(info_path)
                return _public_info(info, "already_running")
        log_path = folder / "server.log"
        instance = uuid4().hex
        args = [sys.executable, "-B", "-m", "tank.practice_host", host, str(port), instance]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with log_path.open("ab") as log:
            process = subprocess.Popen(args, cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                       creationflags=flags, start_new_session=os.name != "nt",
                                       env=os.environ | {"PYTHONIOENCODING": "utf-8"})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and process.poll() is None:
            if info_path.is_file():
                info = read_json(info_path)
                if info["instance"] == instance:
                    return _public_info(info, "running") | {"log": str(log_path)}
            time.sleep(0.1)
    raise ValueError(f"未确认练习主机已启动，请查看 {log_path}")


def stop(root):
    folder, info_path = _folder(root), _server_info(root)
    with shard_lock(folder, "server") as available:
        if available:
            return {"status": "stopped"}
        if not info_path.is_file():
            raise ValueError("练习主机正在启动，请稍后停止")
        info = read_json(info_path)
        _stop_flag(root, info["instance"]).touch()
        return _public_info(info, "stopping")


def _active_host_info(root):
    info_path = _server_info(root)
    if not info_path.is_file():
        return None
    with shard_lock(_folder(root), "server") as available:
        if available:
            return None
        try:
            return read_json(info_path)
        except FileNotFoundError:
            return None


def host_running(root):
    info = _active_host_info(root)
    return info is not None and not _stop_flag(root, info["instance"]).is_file()


def _config(root):
    return _folder(root) / "client.json"


def is_remote(root):
    path = _config(root)
    if not path.is_file() or _active_host_info(root) is not None:
        return False
    host = lan.Client(read_json(path)["url"], b"").host
    if host == "localhost":
        return False
    try:
        return not ip_address(host).is_loopback
    except ValueError:
        return True


def configure(root, url, *, prompt=False, replace_token=False):
    path = _config(root)
    if url is None:
        path.unlink(missing_ok=True)
        return {"mode": "local"}
    url = lan.Client(url, b"").url
    if replace_token and os.environ.get(TOKEN_ENV):
        raise ValueError(f"请先取消环境变量 {TOKEN_ENV}，再更换保存的练习令牌")
    if prompt and not os.environ.get(TOKEN_ENV) and (replace_token or not _token_path(root).is_file()):
        if not sys.stdin.isatty():
            raise ValueError(f"请在交互终端输入组内练习令牌，或设置 {TOKEN_ENV}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", GetPassWarning)
                token = getpass("组内练习令牌：")
        except (GetPassWarning, EOFError) as error:
            raise ValueError(f"无法读取令牌，请设置 {TOKEN_ENV}") from error
        if not token:
            raise ValueError("组内练习令牌不能为空")
        write_json(_token_path(root), {"token": token})
    write_json(path, {"url": url})
    return {"mode": "remote", "url": url}


def _receipt_path(root, request_id):
    return _folder(root) / "client" / f"{request_id}.json"


def _request(client, action, body):
    for delay in (0, 1, 2):
        if delay:
            time.sleep(delay)
        try:
            response = client.request(action, body)
        except NETWORK_ERRORS as error:
            if delay == 2:
                request_id = body["request_id"]
                raise ValueError(f"未收到练习主机响应，请运行 tank sample host-result {request_id}") from error
        else:
            if action != "/practice/replay" and response["request_id"] != body["request_id"]:
                raise ValueError("练习主机返回了不同的请求 ID")
            return response


class _RelayClient:
    def __init__(self, remote):
        self.connection = HTTPConnection(remote.host, remote.port, timeout=STEP_WAIT + 3)
        self.token = remote.token

    def request(self, path, request):
        body = json_text(request).encode()
        nonce, timestamp = uuid4().hex, str(int(time.time()))
        self.connection.request("POST", path, body, {
            "Content-Type": "application/json", "X-Tank-Time": timestamp, "X-Tank-Nonce": nonce,
            "X-Tank-Signature": lan._signature(self.token, "request", f"{path}:{timestamp}", nonce, body),
        })
        response = self.connection.getresponse()
        payload = response.read(MAX_RELAY_RESPONSE + 1)
        signature = response.getheader("X-Tank-Signature", "")
        expected = lan._signature(self.token, "response", str(response.status), nonce, payload)
        if len(payload) > MAX_RELAY_RESPONSE:
            raise ValueError("练习主机返回的策略帧超过大小上限")
        if not signature.isascii() or not hmac.compare_digest(signature, expected):
            raise ValueError("策略转发响应未通过认证")
        data = json.loads(payload)
        if response.status != 200:
            raise ValueError(data["error"])
        if data["request_id"] != request["request_id"]:
            raise ValueError("策略转发返回了不同的练习请求 ID")
        return data

    def step(self, request):
        try:
            return self.request("/practice/step", request)
        except NETWORK_ERRORS:
            self.connection.close()
            return self.request("/practice/step", request)


def _relay_practice(root, remote, request_id, version_id, response):
    relay = _RelayClient(remote)
    step = {"request_id": request_id, "ack": 0, "output": None}
    bot = None
    try:
        while response["status"] in ("queued", "running"):
            response = relay.step(step)
            event = response.get("event")
            if event is None:
                if response.get("closed"):
                    time.sleep(0.05)
                continue
            sequence, kind, payload = (event[field] for field in ("sequence", "kind", "payload"))
            if sequence != step["ack"] + 1:
                raise ValueError("练习主机返回了不同的策略转发序号")
            output = None
            if kind == "start" and bot is None:
                bot = ProcessBot(root, versions.snapshot_data(root, version_id), version_id, match_seconds=None)
                bot.on_match_start(payload)
            elif kind == "obs" and bot is not None:
                output, invalid = online.parse_action(bot.decide(payload))
                if invalid:
                    raise online.OnlineError(f"策略第 {payload['frame']} 帧动作字段类型或范围无效")
            elif kind == "end" and bot is not None:
                bot.on_match_end(payload)
                bot = None
            else:
                raise ValueError("练习主机返回了无效的策略转发事件")
            step.update(ack=sequence, output=output)
        return response
    except NETWORK_ERRORS as error:
        raise ValueError(f"策略转发连接已断开，请运行 tank sample host-result {request_id}") from error
    finally:
        relay.connection.close()
        if response["status"] in ("queued", "running"):
            with suppress(*NETWORK_ERRORS, ValueError):
                relay.request("/practice/cancel", {"request_id": request_id})
            relay.connection.close()
        if bot is not None:
            bot.close(failed=True)


def _import_result(root, client, request_id, response, spec, *, retry_replay=False, match_id=None):
    result = response["result"]
    version_id, level = spec["version_id"], spec["level"]
    if result["version_id"] != version_id or result["level"] != level:
        raise ValueError("练习主机返回了不同的策略或档位")
    downloads = []
    for record in result["matches"]:
        replay_id = record["id"]
        if (type(replay_id) is not int or replay_id < 1 or record["version_id"] != version_id
                or record["level"] != level or record["seed"] != spec["seed"]):
            raise ValueError("练习主机返回了不匹配的对局记录")
        replay = online.practice_replay_path(root, replay_id)
        saved_on_host = record.pop("replay_saved")
        if replay.is_file():
            record["replay_path"] = str(replay)
            record.pop("replay_error", None)
        elif match_id in (None, replay_id) and (saved_on_host or retry_replay):
            downloads.append((record, replay, saved_on_host))
        write_json(online._practice_path(root, record), record)
    for record, replay, saved_on_host in downloads:
        try:
            data = _request(client, "/practice/replay", {"request_id": request_id, "match_id": record["id"]})
        except ValueError as error:
            if saved_on_host:
                raise
            record["replay_error"] = str(error)
            write_json(online._practice_path(root, record), record)
            continue
        content = lan._decode(data["replay"])
        if len(content) > MAX_REPLAY_BYTES:
            raise ValueError("官网回放超过单次回传大小上限")
        replay.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=replay.parent) as directory:
            staged = Path(directory) / replay.name
            staged.write_bytes(content)
            staged.replace(replay)
        record["replay_path"] = str(replay)
        record.pop("replay_error", None)
        write_json(online._practice_path(root, record), record)


def _read_result(root, client, request_id, spec, *, match_id=None):
    response = _request(client, "/practice/result", {"request_id": request_id})
    if "result" in response:
        _import_result(root, client, request_id, response, spec, retry_replay=True, match_id=match_id)
    return response


def result(root, request_id, *, match_id=None):
    _validated_id(request_id, lan.NONCE, "练习请求 ID")
    path = _receipt_path(root, request_id)
    if not path.is_file():
        raise ValueError(f"在发起练习的工作台查询请求 {request_id}")
    receipt = read_json(path)
    client = lan.Client(receipt["url"], _token(root))
    return _read_result(root, client, request_id, receipt["spec"], match_id=match_id)


def replay(root, match_id):
    if type(match_id) is not int or match_id < 1:
        raise online.OnlineError("官网对局 ID 需为正整数")
    paths = list((root / ".local/online/practice").glob(f"*/*/{match_id}.json"))
    if len(paths) == 1:
        record = online._read_practice_record(paths[0])
        tag = record.get("case_tag", "")
        if tag.startswith("host:") and lan.NONCE.fullmatch(tag[5:]):
            replay_path = online.practice_replay_path(root, match_id)
            if replay_path.is_file():
                return record | {"replay_path": str(replay_path)}
            if _receipt_path(root, tag[5:]).is_file():
                result(root, tag[5:], match_id=match_id)
                record = online._read_practice_record(paths[0])
                if replay_path.is_file():
                    return record
                raise online.OnlineError(record.get("replay_error", "练习主机暂未取得官网回放"))
    return online.replay(root, match_id)


def practice(root, version_id, *, client=None, games=1, seed=None, level="rookie"):
    config = _config(root)
    local_host = _active_host_info(root) if client is None else None
    if client is not None or (local_host is None and not config.is_file()):
        return online.practice(root, version_id, client=client, games=games, seed=seed, level=level)
    spec = _practice_spec({"version_id": version_id, "games": games, "level": level, "seed": seed})
    versions.verify(root, version_id)
    if local_host is not None:
        host = local_host["host"]
        if host in ("", "0.0.0.0"):
            host = "127.0.0.1"
        url = f"http://{host}:{local_host['port']}"
    else:
        url = read_json(config)["url"]
    remote = lan.Client(url, _token(root))
    request_id = uuid4().hex
    write_json(_receipt_path(root, request_id), {"url": remote.url, "spec": spec})
    print(f"练习请求 {request_id}", file=sys.stderr, flush=True)
    request = spec | {"request_id": request_id}
    response = _request(remote, "/practice/submit", request)
    if response["relay"]:
        response = _relay_practice(root, remote, request_id, version_id, response)
    while response["status"] in ("queued", "running"):
        time.sleep(POLL_SECONDS)
        response = _request(remote, "/practice/result", {"request_id": request_id})
    if "result" in response:
        _import_result(root, remote, request_id, response, spec)
    if response["status"] == "completed":
        return response["result"] | {"request_id": request_id}
    reason = response.get("error", "找不到主机练习请求")
    raise ValueError(f"{reason}（练习请求 {request_id}）")


def add_arguments(commands):
    server = commands.add_parser("host-serve", help="启动团队官网练习主机")
    server.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    server.add_argument("--port", type=int, default=PORT)
    server.add_argument("--background", action="store_true", help="在后台运行")
    halt = commands.add_parser("host-stop", help="完成当前练习后停止主机")
    use = commands.add_parser("host-use", help="选择本工作台的官网练习主机")
    use.add_argument("url", nargs="?", help=f"例如 http://192.168.1.10:{PORT}")
    use.add_argument("--local", action="store_true", help="改用本机进行官网练习")
    use.add_argument("--replace-token", action="store_true", help="更换保存的组内令牌")
    query = commands.add_parser("host-result", help="查询练习请求并导入已完成的对局")
    query.add_argument("request_id")
    for command in (server, halt, use, query):
        command.add_argument("--json", action="store_true")


def handles(op):
    return op in ("host-serve", "host-stop", "host-use", "host-result")


def dispatch(root, args):
    if args.op == "host-serve":
        runner = launch if args.background else serve
        return runner(root, host=args.host, port=args.port)
    if args.op == "host-stop":
        return stop(root)
    if args.op == "host-use":
        if args.local == (args.url is not None):
            raise ValueError("请选择主机地址或 --local")
        if args.local and args.replace_token:
            raise ValueError("更换令牌时请填写练习主机地址")
        return configure(root, args.url, prompt=True, replace_token=args.replace_token)
    return result(root, args.request_id)


if __name__ == "__main__":
    from .workspace import project_root

    serve(project_root(), host=sys.argv[1], port=int(sys.argv[2]), instance=sys.argv[3])
