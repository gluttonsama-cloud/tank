import gzip
import hashlib
import json
from pathlib import Path
from threading import Thread
from urllib.request import urlopen
from zipfile import ZipFile

import pytest

from tank import catalog, online, versions
from tank.cli import execute, parser
from tank.webapp import WorkbenchHTTP
from tank.workspace import read_json, write_json


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
    finished = record | {"evaluation": {"status": "completed", "score": 123, "breakdown": {"kill": 50}}}
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


@pytest.mark.parametrize("archived", [False, True])
def test_submit_before_evaluation_is_visible(tmp_path, monkeypatch, archived):
    archive = tmp_path / "submission.zip"
    archive.write_bytes(b"archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    package = {"version_id": "snapshot", "path": str(archive), "sha256": digest}
    monkeypatch.setattr(online, "pack", lambda root, key: package)
    record = {"id": 5, "version": "custom", "sha256": digest, "created_at": 42}
    completed = record | {"evaluation": {"status": "completed", "score": 123}}
    version_id = _snapshot(tmp_path, BOT)
    assert online._submission_label(tmp_path, version_id) == "demo v1"
    record["version"] = completed["version"] = "tank-old" if archived else "demo v1"
    listing = _listing(*([record] if archived else [])) | {"rank_submission_enabled": True}
    client = _MockClient([listing, *([] if archived else [{"submission": record}]), {},
                          _listing(record), _listing(record), _listing(completed)])

    result = online.submit(tmp_path, version_id, client=client)
    assert result["id"] == 5
    path = tmp_path / ".local/online/submissions/8-5.json"
    assert catalog._raw_submission(path)["status"] == "awaiting_evaluation"
    assert online.status(tmp_path, 5, client=client)["evaluation"] is None
    assert online.status(tmp_path, 5, client=client)["evaluation"]["score"] == 123
    posts = [path for method, path, _ in client.requests if method == "POST"]
    assert posts == ([] if archived else [online.SUBMISSIONS]) + [online.SUBMISSIONS + "/5/evaluate"]


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


def _replay_payload(level, result):
    return {
        "format": online.FORMAT_VERSION,
        "match": {
            "id": 100, "kind": "room", "room": "AB12CD",
            "red": {"id": None, "name": online.PRACTICE_NAMES[level], "bot": True},
            "blue": {"id": 8, "name": "Team", "bot": False},
        },
        "map": {"seed": 42},
        "frames": [[0]],
        "result": result,
    }


def _practice_socket(level, result):
    return _MockSocket([
        {"type": "welcome", "team": {"id": 8}},
        {"type": "match_start", "kind": "room", "match_id": 100, "side": "blue",
         "opponent": online.PRACTICE_NAMES[level], "map": {"seed": 42}},
        {"type": "obs", "frame": 0, "match_id": 100},
        {"type": "match_end", "match_id": 100, "result": result},
    ])


@pytest.mark.parametrize("level", online.PRACTICE_LEVELS)
def test_practice_runs_room_and_closes_without_reconnect(tmp_path, monkeypatch, level):
    key = _snapshot(tmp_path, BOT)
    result = {"winner": 0, "scores": [{"total": 100}, {"total": 20}]}
    socket = _practice_socket(level, result)
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: socket)
    payload = _replay_payload(level, result)
    raw = gzip.compress(json.dumps(payload).encode())
    pending = [online.RemoteError(404, "回放还在生成，稍后再试")] if level == "rookie" else []
    client = _MockClient([{"code": "AB12CD"}, {"ok": True}, {"ok": True}, *pending, raw])

    progress = online.practice(tmp_path, key, client=client, level=level)
    assert client.requests[0][2]["fields"]["level"] == level
    record = read_json(tmp_path / ".local/online/practice" / key / level / "100.json")
    assert record["score"] == 20 and record["level"] == level
    assert record["version_id"] == key and record["side"] == "blue" and record["map_seed"] == 42

    assert progress["matches"][0]["result"] == result
    assert progress["matches"][0]["replay_url"].endswith("/replay/100")
    replay_path = tmp_path / ".local/online/replays/100.json.gz"
    assert record["replay_path"] == progress["matches"][0]["replay_path"] == str(replay_path)
    assert replay_path.read_bytes() == raw
    assert [path for _, path, _ in client.requests] == [
        online.ROOMS, f"{online.ROOMS}/AB12CD/start", f"{online.ROOMS}/AB12CD/close",
        *(["/api/replays/100"] * (len(pending) + 1)),
    ]
    assert client.requests[-1] == ("GET", "/api/replays/100", {"compressed": True})
    assert socket.sent[1] == {"type": "action", "match_id": 100, "frame": 0,
                              "forward": 0.0, "turn": 0.0, "fire": False}


def test_replay_failure_keeps_score_and_can_retry_from_cli(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    result = {"winner": 0, "scores": [{"total": 100}, {"total": 20}]}
    socket = _practice_socket("boss", result)
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: socket)
    client = _MockClient([{"code": "AB12CD"}, {"ok": True}, {"ok": True}, online.RequestUncertain("超时")])

    record = online.practice(tmp_path, key, client=client, level="boss")["matches"][0]

    assert record["score"] == 20 and record["replay_error"] == "超时"
    path = tmp_path / ".local/online/practice" / key / "boss/100.json"
    saved = read_json(path)
    assert saved["result"] == result and "replay_error" not in saved and "replay_path" not in saved

    # Older workbench records did not include team, room or resolved map seed.
    for name in ("team_id", "room", "map_seed", "opponent"):
        del saved[name]
    write_json(path, saved)
    cache = versions.snapshot_data(tmp_path, key) / "__pycache__"
    cache.mkdir()
    (cache / "bot.cpython-312.pyc").write_bytes(b"runtime cache from an older practice")
    raw = gzip.compress(json.dumps(_replay_payload("boss", result)).encode())
    monkeypatch.setattr(online.OnlineClient, "login", lambda root: _MockClient([raw]))
    imported = execute(tmp_path, parser().parse_args(["online", "replay", "100", "--json"]))
    assert imported["team_id"] == 8 and imported["version_id"] == key
    assert Path(imported["replay_path"]).read_bytes() == raw
    assert read_json(path) == imported


@pytest.mark.parametrize(("field", "value", "error"), [
    ("id", 101, "不一致"),
    ("kind", "ladder", "不一致"),
    ("room", "ZX98YZ", "不一致"),
    ("blue", {"id": 99, "bot": False}, "出生侧"),
    ("red", {"id": 9, "bot": False, "name": online.PRACTICE_NAMES["boss"]}, "练习对手"),
    ("red", {"id": None, "bot": True, "name": online.PRACTICE_NAMES["strong"]}, "练习对手"),
])
def test_replay_rejects_mismatched_official_match(tmp_path, field, value, error):
    key = _snapshot(tmp_path, BOT)
    result = {"winner": 0, "scores": [{"total": 100}, {"total": 20}]}
    path = tmp_path / ".local/online/practice" / key / "boss/100.json"
    record = {"id": 100, "version_id": key, "level": "boss", "side": "blue",
              "score": 20, "result": result, "seed": 42, "room": "AB12CD", "team_id": 8}
    write_json(path, record)
    payload = _replay_payload("boss", result)
    payload["match"][field] = value
    client = _MockClient([gzip.compress(json.dumps(payload).encode())])

    with pytest.raises(online.OnlineError, match=error):
        online.replay(tmp_path, 100, client=client)
    assert not (tmp_path / ".local/online/replays/100.json.gz").exists()
    assert read_json(path) == record


def test_practice_history_uses_selected_version_and_most_recent_ids(tmp_path, monkeypatch):
    version = f"v-{'a' * 64}"
    other = f"v-{'b' * 64}"
    for match_id in range(94, 106):
        level = "boss" if match_id % 2 else "rookie"
        write_json(online._practice_path(tmp_path, {"id": match_id, "level": level, "version_id": version}),
                   {"id": match_id, "level": level, "version_id": version, "score": 46.4})
    write_json(tmp_path / ".local/online/practice" / other / "strong/120.json",
               {"id": 120, "level": "strong", "version_id": other, "score": 100})
    archived = online.practice_replay_path(tmp_path, 104)
    archived.parent.mkdir(parents=True)
    archived.touch()
    monkeypatch.setattr(online.OnlineClient, "login", lambda root: pytest.fail("history must be offline"))

    matches = execute(tmp_path, parser().parse_args(["online", "history", version, "--json"]))
    assert [match["id"] for match in matches] == list(range(105, 95, -1))
    assert matches[0]["replay_url"] == f"{online.SITE}/replay/105"
    assert matches[1] == {"id": 104, "level": "rookie", "score": 46.4, "replay_saved": True,
                          "replay_url": "/replay?online=104"}
    assert [match["id"] for match in online.history(tmp_path, other)] == [120]
    with pytest.raises(online.OnlineError, match="快照 ID"):
        online.history(tmp_path, "../other")


def test_official_replay_viewer_keeps_stable_url_after_restart(tmp_path, monkeypatch):
    official = online.practice_replay_path(tmp_path, 1)
    official.parent.mkdir(parents=True)
    official.write_bytes(gzip.compress(b'{"match":{"id":1},"frames":[[0]]}'))
    local = tmp_path / "local.json"
    write_json(local, {"match": {"id": "local"}})
    monkeypatch.setattr("tank.webapp.evaluation.replay", lambda root, case: {"replay_path": local})

    for restart in (False, True):
        with WorkbenchHTTP(("127.0.0.1", 0), tmp_path) as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                if not restart:
                    with urlopen(base + "/replay?case=local-case", timeout=5) as response:
                        assert response.url == base + "/replay/1"
                    with urlopen(base + "/api/replays/1", timeout=5) as response:
                        assert json.load(response)["match"]["id"] == "local"
                with urlopen(base + "/replay?online=1", timeout=5) as response:
                    assert response.url == base + "/replay/01"
                    assert b"/static/replay.js" in response.read()
                with urlopen(base + "/api/replays/01", timeout=5) as response:
                    assert json.load(response)["match"]["id"] == 1
            finally:
                server.shutdown()
                worker.join()


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


def _completed(submission_id, digest, score):
    return {
        "id": submission_id, "version": f"official-{submission_id}", "sha256": digest,
        "created_at": float(submission_id), "download_url": None,
        "evaluation": {
            "status": "completed", "score": score, "error": None,
            "breakdown": {"kill": 0, "damage": score, "survive": 0, "time": 0},
        },
    }


def test_refresh_uses_one_listing_and_links_old_and_new_official_records(tmp_path):
    version = _snapshot(tmp_path, BOT)
    packed = online.pack(tmp_path, version)
    best = _completed(1, "b" * 64, 5600)
    recent = _completed(2, packed["sha256"], 5600)
    unlinked = _completed(3, "c" * 64, 4000) | {"created_at": 1.5}
    pending = _completed(4, "d" * 64, 6000)
    pending["evaluation"]["status"] = "running"
    invalid = _completed(5, "e" * 64, 7000)
    del invalid["evaluation"]["breakdown"]["time"]
    fake_total = _completed(6, "f" * 64, 8000)
    fake_total["evaluation"]["score"] = 7000
    listing = _listing(pending, recent, best, invalid, unlinked, fake_total) | {"best_id": 1}
    client = _MockClient([listing, listing])
    (tmp_path / ".local/online/submissions").mkdir()
    write_json(tmp_path / ".local/online/submissions/8-1.json", best | {"team_id": 8, "version_id": version})

    summary = online.refresh(tmp_path, client=client)

    assert client.requests == [("GET", online.SUBMISSIONS, {})]
    assert summary["best"]["id"] == 1 and summary["best"]["version_id"] == version
    assert summary["recent"]["id"] == 2 and summary["recent"]["version_id"] == version
    assert all(item["games"] == 40 for item in summary["submissions"])
    assert {item["id"]: item["version_id"] for item in summary["submissions"]} == {1: version, 2: version, 3: None}
    assert read_json(tmp_path / ".local/online/team-scores.json") == summary
    saved = read_json(tmp_path / ".local/online/submissions/8-2.json")
    assert saved["link_source"] == "archive_sha256"
    Path(packed["path"]).unlink()
    assert online.refresh(tmp_path, client=client)["recent"]["version_id"] == version


def test_refresh_leaves_unknown_or_conflicting_link_unassigned(tmp_path, monkeypatch):
    version = _snapshot(tmp_path, BOT)
    row = _completed(1, "a" * 64, 42)
    client = _MockClient([_listing(row) | {"best_id": 1}])
    path = tmp_path / ".local/online/submissions/8-1.json"
    prior = row | {"team_id": 8, "version_id": version, "sha256": "b" * 64}
    write_json(path, prior)
    monkeypatch.setattr(online, "_archive_version", lambda *args: pytest.fail("a conflicting record must not fall back"))

    summary = online.refresh(tmp_path, client=client)

    assert summary["best"]["version_id"] is None
    assert read_json(path) == prior
