import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from tank import online, versions


def _snapshot(root: Path, code: str) -> str:
    strategy = root / "strategies/demo"
    strategy.mkdir(parents=True)
    (strategy / "submission.json").write_text(
        '{"runtime":"python","entrypoint":"bot.py:Demo"}', encoding="utf-8",
    )
    (strategy / "bot.py").write_text(code, encoding="utf-8", newline="\n")
    return versions.create(root, "demo")["id"]


BOT = """\
from seedcup import Bot
import numpy as np
class Demo(Bot):
    def decide(self, obs):
        return {"forward": float(np.float64(0)), "turn": 0, "fire": False}
"""


def test_pack_reproducible_and_preflights_extracted_python(tmp_path):
    key = _snapshot(tmp_path, BOT)
    result = online.pack(tmp_path, key)
    path = Path(result["path"])
    data = path.read_bytes()
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["trial"]["frames"] > 0
    assert 0 <= result["trial"]["cpu_seconds"] < 10
    with ZipFile(path) as archive:
        assert archive.namelist() == ["bot.py", "submission.json"]
        assert archive.read("bot.py").decode() == BOT
    assert online.pack(tmp_path, key)["sha256"] == result["sha256"]
    versions.verify(tmp_path, key)


def test_pack_rejects_dependency_available_only_locally(tmp_path):
    key = _snapshot(tmp_path, "import pytest\n" + BOT)
    with pytest.raises((RuntimeError, ValueError), match="日志"):
        online.pack(tmp_path, key)
    assert not list(tmp_path.glob(".local/online/packages/*.zip"))
    logs = list(tmp_path.glob(".local/logs/*.log"))
    assert len(logs) == 1 and "No module named 'pytest'" in logs[0].read_text(encoding="utf-8")


def test_pack_rejects_excess_cpu(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    monkeypatch.setattr(online, "_cpu_seconds", lambda process: 10.01)
    with pytest.raises(online.OnlineError, match="10 秒 CPU"):
        online.pack(tmp_path, key)


class _MockClient:
    user = {"id": 8, "username": "team008"}
    username = "local-test-user"
    password = "local-test-password"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def request(self, method, path, **kwargs):
        self.requests.append((method, path, kwargs))
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def _listing(*items):
    return {"window": {"automatic_evaluation": True}, "submissions": list(items)}


def test_submit_timeout_reconciles_same_zip_without_second_post(tmp_path, monkeypatch):
    archive = tmp_path / "submission.zip"
    archive.write_bytes(b"archive content")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    package = {"version_id": "test-key", "path": str(archive), "sha256": digest}
    monkeypatch.setattr(online, "pack", lambda root, key: package)
    record = {"version": "custom", "id": 5, "sha256": digest, "evaluation": {"status": "queued"}}
    finished = record | {"evaluation": {"status": "succeeded", "score": 123, "breakdown": {"kill": 50}}}
    client = _MockClient([_listing(), online.RequestUncertain("timeout"), _listing(record), _listing(finished)])

    assert online.submit(tmp_path, "test-key", version="custom", client=client)["id"] == 5
    assert online.status(tmp_path, 5, client=client)["evaluation"] == finished["evaluation"]
    assert [method for method, _, _ in client.requests] == ["GET", "POST", "GET", "GET"]
    post = client.requests[1][2]
    assert post["content_type"].startswith("multipart/form-data; boundary=")
    assert b"archive content" in post["body"]
    saved = json.loads((tmp_path / ".local/online/submissions/8-5.json").read_text(encoding="utf-8"))
    assert saved["version_id"] == "test-key"
    assert saved["sha256"] == digest


def test_submit_checks_version_conflict_after_timeout(tmp_path, monkeypatch):
    archive = tmp_path / "submission.zip"
    archive.write_bytes(b"archive content")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    package = {"version_id": "test-key", "path": str(archive), "sha256": digest}
    monkeypatch.setattr(online, "pack", lambda root, key: package)
    client = _MockClient([
        _listing(), online.RequestUncertain("timeout"), _listing({"version": "custom", "sha256": "b" * 64}),
    ])

    with pytest.raises(online.OnlineError, match="不同的 ZIP"):
        online.submit(tmp_path, "test-key", version="custom", client=client)
    assert [method for method, _, _ in client.requests] == ["GET", "POST", "GET"]


class _MockSocket:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass

    async def recv(self):
        return json.dumps(next(self.messages))

    async def send(self, message):
        self.sent.append(json.loads(message))


def test_practice_runs_room_and_closes_without_reconnect(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    team = {"id": 8}
    result = {"winner": 0}
    socket = _MockSocket([
        {"type": "welcome", "team": team},
        {"type": "match_start", "kind": "room", "match_id": 100},
        {"type": "obs", "frame": 0, "match_id": 100},
        {"type": "match_end", "match_id": 100, "result": result},
    ])
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: socket)
    client = _MockClient([{"code": "AB12CD"}, {"ok": True}, {"ok": True}])

    progress = online.practice(tmp_path, key, client=client)

    assert progress["matches"][0]["result"] == result
    assert progress["matches"][0]["replay_url"].endswith("/replay/100")
    assert [path for _, path, _ in client.requests] == [
        online.ROOMS, f"{online.ROOMS}/AB12CD/start", f"{online.ROOMS}/AB12CD/close",
    ]
    assert socket.sent[1] == {"type": "action", "match_id": 100, "frame": 0,
                              "forward": 0.0, "turn": 0.0, "fire": False}


def test_practice_stops_when_connection_is_replaced(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    socket = _MockSocket([
        {"type": "welcome", "team": {"id": 8}}, {"type": "error", "code": "replaced"},
    ])
    calls = []

    def connect(*args, **kwargs):
        calls.append(args)
        return socket

    monkeypatch.setattr(online.websockets, "connect", connect)
    client = _MockClient([{"code": "AB12CD"}, {"ok": True}, {"ok": True}])

    with pytest.raises(online.OnlineError, match="连接被替换"):
        online.practice(tmp_path, key, client=client)
    assert len(calls) == 1
