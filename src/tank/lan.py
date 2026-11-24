"""Lease an evaluation shard and merge authenticated LAN results."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import http.client
import json
import os
import re
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
from urllib.parse import urlsplit
from uuid import uuid4

from . import evaluation
from .eval_plan import json_text, match_specs, required, shard_for
from .eval_store import cached_matches, database, local_dir, queued_shards, read_manifest, shard_lock
from .workspace import read_json, write_json

TOKEN_ENV = "TANK_LAN_TOKEN"
MAX_REQUEST_BYTES = 96 * 1024 * 1024
SIGNATURE_AGE = 300
WORKER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
NONCE = re.compile(r"[0-9a-f]{32}")


def _token():
    value = os.environ.get(TOKEN_ENV)
    if not value:
        raise ValueError(f"请设置本机环境变量 {TOKEN_ENV}")
    return value.encode("utf-8")


def _worker_name(name):
    if not isinstance(name, str) or not WORKER_NAME.fullmatch(name):
        raise ValueError("工作机名称须由字母、数字、点、下划线或连字符组成")
    return name


def _signature(token, label, details, nonce, body):
    body_digest = hashlib.sha256(body).hexdigest()
    message = f"{label}\n{details}\n{nonce}\n{body_digest}".encode("ascii")
    return hmac.digest(token, message, "sha256").hex()


def _encode(bundle):
    return base64.b64encode(bundle).decode("ascii")


def _decode(data):
    if not isinstance(data, str):
        raise ValueError("bundle 必须为 Base64 字符串")
    try:
        return base64.b64decode(data, validate=True)
    except binascii.Error as error:
        raise ValueError("bundle 的 Base64 编码错误") from error


class Coordinator:
    def __init__(self, root, run_id, token, lease_seconds, *, clock=time.time):
        if type(lease_seconds) is not int or lease_seconds < 3:
            raise ValueError("lease_seconds 须为大于或等于 3 的整数")
        self.root = root
        self.run_id = run_id
        self.token = token
        self.lease_seconds = lease_seconds
        self.clock = clock
        self.mutex = Lock()
        self.recent_nonces = {}
        self.folder = local_dir(root) / run_id
        with database(root) as db:
            self.manifest, _ = read_manifest(db, run_id)

    def verify(self, action, timestamp, nonce, signature, body):
        if not timestamp.isascii() or not timestamp.isdecimal() or len(timestamp) != 10 or not NONCE.fullmatch(nonce):
            return False
        if not signature.isascii() or len(signature) != 64:
            return False
        now = time.time()
        issued = int(timestamp)
        if abs(now - issued) > SIGNATURE_AGE:
            return False
        expected = _signature(self.token, "request", f"{action}:{timestamp}", nonce, body)
        if not hmac.compare_digest(signature, expected):
            return False
        with self.mutex:
            self.recent_nonces = {key: expiry for key, expiry in self.recent_nonces.items() if expiry >= now}
            if nonce in self.recent_nonces:
                return False
            self.recent_nonces[nonce] = issued + SIGNATURE_AGE
        return True

    def _pending_shards(self):
        specs = match_specs(self.manifest)
        with database(self.root) as db:
            _, run = read_manifest(db, self.run_id)
            if run["cancelled"]:
                return []
            cached = cached_matches(db, specs, summary=True)
            failed = {
                row[0] for row in db.execute("SELECT case_id FROM failures WHERE run_id = ?", (self.run_id,))
            }
            queued = queued_shards(db, self.run_id)
        complete = {
            case_id for case_id, match in cached.items() if match["has_replay"] or not self.manifest["replay"]
        }
        pending = {
            shard_for(specs[case_id], self.manifest["shards"]) for case_id in specs.keys() - complete - failed
        }
        active = evaluation._running_shards(self.root, self.manifest, specs)
        return sorted(pending - queued - set(active))

    def handle(self, action, request):
        worker = _worker_name(required(request, "worker"))
        with self.mutex:
            with database(self.root) as db:
                ledger = {
                    row["lease_id"]: dict(row)
                    for row in db.execute("SELECT * FROM lan_leases WHERE run_id = ?", (self.run_id,))
                }
            if action == "/claim":
                return self._claim(worker, ledger)
            lease_id = required(request, "lease_id")
            if not isinstance(lease_id, str) or not NONCE.fullmatch(lease_id):
                raise ValueError("领取记录 ID 格式错误")
            lease = ledger.get(lease_id)
            if lease is None or lease["worker"] != worker:
                raise ValueError("找不到该工作机的领取记录")
            if action == "/renew":
                return self._renew(lease)
            return self._submit(lease, required(request, "bundle"))

    def _claim(self, worker, ledger):
        now = self.clock()
        pending = self._pending_shards()
        active = {
            lease["shard"]: lease
            for lease in ledger.values()
            if not lease["submitted"] and lease["expires_at"] > now and lease["shard"] in pending
        }
        target = next((lease for lease in active.values() if lease["worker"] == worker), None)
        if target is None:
            shard = next((item for item in pending if item not in active), None)
            if shard is None:
                return {"assignment": None, "status": evaluation.status(self.root, self.run_id)["status"]}
            lease_id = uuid4().hex
            target = {
                "lease_id": lease_id, "run_id": self.run_id, "shard": shard, "worker": worker,
                "expires_at": now + self.lease_seconds, "submitted": False,
            }
            with database(self.root) as db:
                db.execute(
                    "INSERT INTO lan_leases (run_id, lease_id, shard, worker, expires_at) VALUES (?, ?, ?, ?, ?)",
                    (self.run_id, lease_id, shard, worker, target["expires_at"]),
                )
        with TemporaryDirectory(dir=self.folder) as temp:
            name = Path(temp) / "plan.jsonl.gz"
            evaluation.export_run(self.root, self.run_id, name, shard=target["shard"])
            return {"assignment": target, "lease_seconds": self.lease_seconds, "bundle": _encode(name.read_bytes())}

    def _renew(self, lease):
        now = self.clock()
        with database(self.root) as db:
            _, run = read_manifest(db, self.run_id)
            if run["cancelled"]:
                return {"cancelled": True}
            if lease["submitted"] or lease["expires_at"] <= now:
                raise ValueError("分片领取已结束或超时")
            expiry = now + self.lease_seconds
            db.execute("UPDATE lan_leases SET expires_at = ? WHERE lease_id = ?", (expiry, lease["lease_id"]))
        return {"expires_at": expiry}

    def _submit(self, lease, bundle):
        with TemporaryDirectory(dir=self.folder) as temp:
            name = Path(temp) / "results.jsonl.gz"
            name.write_bytes(_decode(bundle))
            result = evaluation.import_run(self.root, name, expected_assignment=(self.run_id, lease["shard"]))
        with database(self.root) as db:
            db.execute("UPDATE lan_leases SET submitted = 1 WHERE lease_id = ?", (lease["lease_id"],))
        status = evaluation.status(self.root, self.run_id)
        return {"shard": lease["shard"], "status": status | {
            "imported_cases": result["imported_cases"], "imported_failures": result["imported_failures"],
        }}


def _server(coordinator, host, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, pattern, *args):
            pass

        def do_POST(self):
            if self.path not in ("/claim", "/renew", "/submit"):
                self.send_error(404)
                return
            self.connection.settimeout(30)
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_REQUEST_BYTES:
                self.send_error(413)
                return
            body = self.rfile.read(length)
            timestamp = self.headers.get("X-Tank-Time", "")
            nonce = self.headers.get("X-Tank-Nonce", "")
            signature = self.headers.get("X-Tank-Signature", "")
            if not coordinator.verify(self.path, timestamp, nonce, signature, body):
                self.send_error(403)
                return
            try:
                request = json.loads(body)
                if not isinstance(request, dict):
                    raise ValueError("请求必须为 JSON 对象")
                response = coordinator.handle(self.path, request)
                response_status = 200
            except (ValueError, EOFError, OSError) as error:
                response = {"error": str(error)}
                response_status = 400
            payload = json_text(response).encode("utf-8")
            self.send_response(response_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header(
                "X-Tank-Signature",
                _signature(coordinator.token, "response", str(response_status), nonce, payload),
            )
            self.end_headers()
            self.wfile.write(payload)

    return ThreadingHTTPServer((host, port), Handler)


def serve(root, run_id, *, host="127.0.0.1", port=8765, lease_seconds=180, token=None):
    coordinator = Coordinator(root, run_id, token if token is not None else _token(), lease_seconds)
    with shard_lock(coordinator.folder / "lan", 0) as acquired:
        if not acquired:
            raise ValueError(f"批次 {run_id} 的协调服务已在运行")
        with _server(coordinator, host, port) as server:
            print(f"{run_id} 正在监听 {host}:{server.server_port}", file=sys.stderr, flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    return {"run_id": run_id, "status": "stopped"}


class Client:
    def __init__(self, url, token):
        target = urlsplit(url)
        if (
            target.scheme != "http" or target.hostname is None or target.username is not None
            or target.path not in ("", "/") or target.query or target.fragment
        ):
            raise ValueError("协调地址格式为 http://主机:端口")
        self.host = target.hostname
        self.port = target.port or 80
        self.url = f"http://{target.netloc}"
        self.token = token

    def request(self, action, request):
        data = json_text(request).encode("utf-8")
        if len(data) > MAX_REQUEST_BYTES:
            raise ValueError("分片超过单次回传大小上限")
        nonce = uuid4().hex
        timestamp = str(int(time.time()))
        client = http.client.HTTPConnection(self.host, self.port, timeout=30)
        try:
            client.request("POST", action, data, {
                "Content-Type": "application/json", "X-Tank-Time": timestamp, "X-Tank-Nonce": nonce,
                "X-Tank-Signature": _signature(self.token, "request", f"{action}:{timestamp}", nonce, data),
            })
            response = client.getresponse()
            payload = response.read(MAX_REQUEST_BYTES + 1)
            status_code = response.status
            signature = response.getheader("X-Tank-Signature", "")
        finally:
            client.close()
        if len(payload) > MAX_REQUEST_BYTES:
            raise ValueError("协调服务返回的分片超过大小上限")
        expected = _signature(self.token, "response", str(status_code), nonce, payload)
        if not signature.isascii() or not hmac.compare_digest(signature, expected):
            raise ValueError("协调服务响应未通过认证，请核对地址与令牌")
        result = json.loads(payload)
        if status_code != 200:
            raise ValueError(result["error"])
        return result


def _submit_outbox(root, client, worker, folder):
    receipts = []
    for metadata_path in sorted(folder.glob("*.json")):
        metadata = read_json(metadata_path)
        if metadata["url"] != client.url or metadata["worker"] != worker:
            continue
        bundle = metadata_path.with_suffix(".jsonl.gz")
        evaluation.export_run(root, metadata["run_id"], bundle, shard=metadata["shard"])
        response = client.request("/submit", {
            "worker": worker, "lease_id": metadata["lease_id"], "bundle": _encode(bundle.read_bytes()),
        })
        metadata_path.unlink()
        bundle.unlink()
        receipts.append(response)
    return receipts


def _process_assignment(root, client, worker, claimed, outbox):
    assignment = claimed["assignment"]
    run_id, shard = assignment["run_id"], assignment["shard"]
    lease_id = assignment["lease_id"]
    if not isinstance(lease_id, str) or not NONCE.fullmatch(lease_id):
        raise ValueError("领取记录 ID 格式错误")
    with TemporaryDirectory(dir=outbox) as temp:
        plan = Path(temp) / "plan.jsonl.gz"
        plan.write_bytes(_decode(claimed["bundle"]))
        evaluation.import_run(root, plan, expected_assignment=(run_id, shard))
    metadata = outbox / f"{lease_id}.json"
    write_json(metadata, {
        "url": client.url, "worker": worker, "lease_id": lease_id, "run_id": run_id, "shard": shard,
    })
    stop = Event()

    def renew():
        while not stop.wait(claimed["lease_seconds"] / 3):
            try:
                renewed = client.request("/renew", {"worker": worker, "lease_id": lease_id})
                if renewed.get("cancelled"):
                    evaluation.cancel(root, run_id)
                    return
            except (OSError, ValueError, http.client.HTTPException):
                # Work and save locally while the coordinator is offline.
                continue

    heartbeat = Thread(target=renew, daemon=True)
    heartbeat.start()
    try:
        evaluation.resume(root, run_id, shard=shard, background=False)
    finally:
        stop.set()
        heartbeat.join()


def work(root, url, *, name=None, max_shards=1, token=None):
    worker = _worker_name(name if name is not None else socket.gethostname())
    if type(max_shards) is not int or max_shards < 1:
        raise ValueError("max_shards 必须是正整数")
    client = Client(url, token if token is not None else _token())
    outbox = local_dir(root) / "lan-outbox"
    outbox.mkdir(exist_ok=True)
    with shard_lock(outbox, 0) as acquired:
        if not acquired:
            raise ValueError("这台工作机已在领取分片")
        receipts = _submit_outbox(root, client, worker, outbox)
        for _ in range(max_shards):
            claimed = client.request("/claim", {"worker": worker})
            if claimed["assignment"] is None:
                return {"worker": worker, "status": claimed["status"], "receipts": receipts}
            _process_assignment(root, client, worker, claimed, outbox)
            receipts.extend(_submit_outbox(root, client, worker, outbox))
    return {"worker": worker, "status": receipts[-1]["status"]["status"], "receipts": receipts}


def add_arguments(parser):
    commands = parser.add_subparsers(dest="lan_action", required=True)
    server = commands.add_parser("serve", help="为一个批次分配局域网分片")
    server.add_argument("run_id")
    server.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    server.add_argument("--port", type=int, default=8765)
    server.add_argument("--lease-seconds", type=int, default=180)
    worker = commands.add_parser("work", help="领取、运行并回传分片")
    worker.add_argument("url", help="例如 http://192.168.1.10:8765")
    worker.add_argument("--name", help="工作机名称，默认使用本机主机名")
    worker.add_argument("--max-shards", type=int, default=1)


def dispatch(root, args):
    if args.lan_action == "serve":
        return serve(root, args.run_id, host=args.host, port=args.port, lease_seconds=args.lease_seconds)
    return work(root, args.url, name=args.name, max_shards=args.max_shards)
