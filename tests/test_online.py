import gzip
import hashlib
import json
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
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
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))
    payload = _replay_payload(level, result)
    raw = gzip.compress(json.dumps(payload).encode())
    pending = [online.RemoteError(404, "回放还在生成，稍后再试")] if level == "rookie" else []
    client = _MockClient([*_practice_responses(_FOREGROUND_SDK), *pending, raw])

    progress = online.practice(tmp_path, key, client=client, level=level)
    assert next(fields["fields"]["level"] for method, path, fields in client.requests
                if method == "POST" and path == online.ROOMS) == level
    record = read_json(tmp_path / ".local/online/practice" / key / level / "100.json")
    assert record["score"] == 20 and record["level"] == level
    assert record["version_id"] == key and record["side"] == "blue" and record["map_seed"] == 42

    assert progress["matches"][0]["result"] == result
    assert progress["matches"][0]["replay_url"].endswith("/replay/100")
    replay_path = tmp_path / ".local/online/replays/100.json.gz"
    assert record["replay_path"] == progress["matches"][0]["replay_path"] == str(replay_path)
    assert replay_path.read_bytes() == raw
    assert [path for method, path, _ in client.requests if method == "POST"] == [
        online.ROOMS, f"{online.ROOMS}/AB12CD/start", f"{online.ROOMS}/AB12CD/close"]
    assert sum(path == "/api/replays/100" for _, path, _ in client.requests) == len(pending) + 1
    assert client.requests[-1] == ("GET", "/api/replays/100", {"compressed": True})
    assert socket.sent[1] == {"type": "action", "match_id": 100, "frame": 0,
                              "forward": 0.0, "turn": 0.0, "fire": False}


def test_replay_failure_keeps_score_and_can_retry_from_cli(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    result = {"winner": 0, "scores": [{"total": 100}, {"total": 20}]}
    socket = _practice_socket("boss", result)
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: socket)
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))
    client = _MockClient([*_practice_responses(_FOREGROUND_SDK), online.RequestUncertain("超时")])

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
    client = _MockClient([*_practice_responses(_FOREGROUND_SDK, finish=False), _room_list("playing"),
                          _official_state("other-client", _room_match("EF34GH"))])
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))

    with pytest.raises(online.OnlineError, match="连接被替换") as failure:
        online.practice(tmp_path, key, client=client)
    assert len(calls) == 1
    assert "暂缓关闭" in failure.value.close_error
    assert not any(path.endswith("/close") for _, path, _ in client.requests)


_COLLECTION_SDK = "tank-workbench/collector/" + "0" * 32
_FOREGROUND_SDK = "tank-workbench/practice/" + "0" * 32


def _official_state(sdk=None, match=None, *, in_event=False):
    return {"user": {"id": 8}, "client": {"online": sdk is not None, "sdk": sdk},
            "match": match, "in_event": in_event}


def _room_match(code="AB12CD", game=0, match_id=100):
    return {"id": match_id, "room": code, "kind": "room", "game": game}


def _room_list(status, code="AB12CD", matches=()):
    return {"rooms": [{"code": code, "status": status, "matches": list(matches)}]}


def _practice_responses(sdk, *, finish=True):
    own = _official_state(sdk)
    setup = [_official_state(), _official_state(), {"rooms": []}, own,
             {"code": "AB12CD", "is_owner": True}, own, {"ok": True}]
    if finish:
        setup += [_official_state(sdk, _room_match()), _room_list("finished"), own, {"ok": True}]
    return setup


@pytest.mark.parametrize("practice", [online.practice, online.collect_practice])
@pytest.mark.parametrize("state", [_official_state("other-client"), _official_state(in_event=True),
                                   _official_state(match={"id": 909})])
def test_practice_defers_before_connecting_when_team_is_busy(tmp_path, monkeypatch, state, practice):
    key = _snapshot(tmp_path, BOT)
    client = _MockClient([state])
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: pytest.fail("connected"))

    with pytest.raises(online.PracticeDeferred) as failure:
        practice(tmp_path, key, client=client)

    assert failure.value.matches == []
    assert client.requests == [("GET", "/api/me", {})]


@pytest.mark.parametrize("practice", [online.practice, online.collect_practice])
def test_practice_rechecks_before_sending_login(tmp_path, monkeypatch, practice):
    key = _snapshot(tmp_path, BOT)
    client = _MockClient([_official_state(), _official_state("other-client")])
    socket = _MockSocket([])
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: socket)

    with pytest.raises(online.PracticeDeferred, match="已有在线客户端"):
        practice(tmp_path, key, client=client)
    assert not socket.sent


def test_foreground_start_conflict_does_not_abort_the_other_match(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    socket = _MockSocket([{"type": "welcome", "team": {"id": 8}}])
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: socket)
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))
    state = _official_state(_FOREGROUND_SDK, _room_match("EF34GH"))
    responses = _practice_responses(_FOREGROUND_SDK, finish=False)
    responses[-1] = online.RemoteError(409, "队伍正在打另一场对局")
    client = _MockClient([*responses, _room_list("waiting"), state])

    with pytest.raises(online.RemoteError, match="正在打另一场对局") as failure:
        online.practice(tmp_path, key, client=client)
    assert "暂缓关闭" in failure.value.close_error
    assert not any(path.endswith("/close") for _, path, _ in client.requests)


@pytest.mark.parametrize("replaced", [False, True])
def test_collection_keeps_partial_receipt_and_closes_only_its_own_room(tmp_path, monkeypatch, replaced):
    key = _snapshot(tmp_path, BOT)
    final = {"winner": 0, "scores": [{"total": 100}, {"total": 20}]}
    socket = _practice_socket("boss", final)
    if replaced:
        socket.messages = iter([*socket.messages, {"type": "error", "code": "replaced"}])
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: socket)
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))
    own = _official_state(_COLLECTION_SDK)
    active = _official_state(_COLLECTION_SDK, _room_match())
    external = _official_state("other-client", _room_match("EF34GH", match_id=200))
    raw = gzip.compress(json.dumps(_replay_payload("boss", final)).encode())
    responses = [_official_state(), _official_state(), {"rooms": [{"code": "EF34GH"}]}, own,
                 {"code": "AB12CD", "is_owner": True}, own, {"ok": True}, active]
    client = _MockClient([*responses, _room_list("playing"), external, raw] if replaced
                         else [*responses, _room_list("finished"), own, {"ok": True}, raw])
    tag = "sample-" + "f" * 32 + "-15"
    kwargs = {"client": client, "level": "boss", "seed": 42, "case_tag": tag}

    if replaced:
        with pytest.raises(online.PracticeDeferred, match="连接被替换") as failure:
            online.collect_practice(tmp_path, key, games=2, **kwargs)
        assert failure.value.room == "AB12CD"
        assert "暂缓关闭" in failure.value.close_error
        records = failure.value.matches
    else:
        records = online.collect_practice(tmp_path, key, games=1, **kwargs)["matches"]

    assert len(records) == 1 and records[0]["case_tag"] == tag and records[0]["score"] == 20
    receipt = read_json(tmp_path / ".local/sampling/receipts" / tag / "100.json")
    assert receipt | {"replay_path": records[0]["replay_path"]} == records[0]
    assert client.requests[4][2]["fields"]["public"] is False
    endpoints = [path for _, path, _ in client.requests]
    assert f"{online.ROOMS}/EF34GH/close" not in endpoints
    assert (f"{online.ROOMS}/AB12CD/close" in endpoints) is not replaced
    assert (tmp_path / ".local/online/collection-room-8.json").is_file() is replaced


def test_foreground_clears_a_confirmed_orphan_before_signing_in(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    pending = tmp_path / ".local/online/collection-room-8.json"
    write_json(pending, {"code": "EF34GH"})
    final = {"winner": 0, "scores": [{"total": 100}, {"total": 20}]}
    old_match = _official_state(match=_room_match("EF34GH", match_id=99))
    raw = gzip.compress(json.dumps(_replay_payload("rookie", final)).encode())
    client = _MockClient([_room_list("playing", "EF34GH"), old_match, {"ok": True},
                          *_practice_responses(_FOREGROUND_SDK), raw])
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))

    def connect(*args, **kwargs):
        assert ("POST", f"{online.ROOMS}/EF34GH/close", {}) in client.requests
        assert not pending.exists()
        return _practice_socket("rookie", final)

    monkeypatch.setattr(online.websockets, "connect", connect)
    tag = "host:" + "d" * 32
    result = online.practice(tmp_path, key, client=client, case_tag=tag)
    assert result["room"] == "AB12CD"
    assert result["matches"][0]["case_tag"] == tag


def test_foreground_leaves_orphan_while_another_client_is_using_the_team(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    pending = tmp_path / ".local/online/collection-room-8.json"
    write_json(pending, {"code": "EF34GH"})
    client = _MockClient([{"rooms": [{"code": "EF34GH"}]}, _official_state("other-client")])
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: pytest.fail("connected"))

    with pytest.raises(online.PracticeDeferred, match="暂缓关闭"):
        online.practice(tmp_path, key, client=client)
    assert pending.is_file()
    assert [method for method, _, _ in client.requests] == ["GET", "GET"]


@pytest.mark.parametrize("already_closed", [False, True])
def test_foreground_recovers_start_request_that_never_reached_official(tmp_path, monkeypatch, already_closed):
    key = _snapshot(tmp_path, BOT)
    pending = tmp_path / ".local/online/collection-room-8.json"
    write_json(pending, {"code": "EF34GH", "games": 2, "start": "requested"})
    if already_closed:
        inspect = [{"rooms": []}, _official_state(), {"matches": []}]
    else:
        inspect = [_room_list("waiting", "EF34GH"), _official_state()]
    final = {"winner": 0, "scores": [{"total": 100}, {"total": 20}]}
    raw = gzip.compress(json.dumps(_replay_payload("rookie", final)).encode())
    closed_state = [{"rooms": []}, _official_state(), {"matches": []}]
    close = [] if already_closed else [{"ok": True}, *closed_state, *closed_state]
    client = _MockClient([*inspect, *inspect, *close, *_practice_responses(_FOREGROUND_SDK), raw])
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))

    def connect(*args, **kwargs):
        closed = ("POST", f"{online.ROOMS}/EF34GH/close", {}) in client.requests
        assert closed is not already_closed and not pending.exists()
        return _practice_socket("rookie", final)

    monkeypatch.setattr(online.websockets, "connect", connect)
    assert len(online.practice(tmp_path, key, client=client)["matches"]) == 1


def test_collection_leaves_unknown_room_untouched_after_lost_create_response(tmp_path, monkeypatch):
    key = _snapshot(tmp_path, BOT)
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: _practice_socket("rookie", {}))
    client = _MockClient([_official_state(), _official_state(), {"rooms": []},
                          _official_state(_COLLECTION_SDK), online.RequestUncertain("官网回复超时")])
    with pytest.raises(online.PracticeDeferred, match="官网回复超时"):
        online.collect_practice(tmp_path, key, client=client)

    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: pytest.fail("connected"))
    client = _MockClient([{"rooms": [{"code": "EF34GH"}]}])
    with pytest.raises(online.PracticeDeferred, match="未收到确认"):
        online.collect_practice(tmp_path, key, client=client)
    assert client.requests == [("GET", online.ROOMS, {})]


def test_collection_rejects_path_like_tags_before_requests(tmp_path):
    with pytest.raises(online.OnlineError, match="标识格式无效"):
        online.collect_practice(tmp_path, "unused", client=_MockClient([]), case_tag="../practice")


def test_remote_bot_needs_no_host_snapshot_and_finished_score_survives_failed_cleanup(tmp_path, monkeypatch):
    key = "v-" + "7" * 64
    tag = "host:" + "1" * 32
    final = {"winner": 0, "scores": [{"total": 100}, {"total": 20}]}
    path = tmp_path / ".local/online/practice" / key / "rookie/100.json"
    socket = _practice_socket("rookie", final)
    monkeypatch.setattr(online.websockets, "connect", lambda *args, **kwargs: socket)
    monkeypatch.setattr(online, "uuid4", lambda: SimpleNamespace(hex="0" * 32))
    monkeypatch.setattr(versions, "snapshot_data", lambda *args: pytest.fail("read host strategy source"))
    raw = gzip.compress(json.dumps(_replay_payload("rookie", final)).encode())
    client = _MockClient([*_practice_responses(_FOREGROUND_SDK), raw])
    calls = []

    class RemoteBot:
        def on_match_start(self, message):
            calls.append(message["type"])

        def decide(self, message):
            calls.append(message["type"])
            return {"forward": 0, "turn": 0, "fire": False}

        def on_match_end(self, result):
            assert read_json(path)["result"] == result
            calls.append("match_end")
            raise RuntimeError("远程策略收尾失败")

        def close(self):
            calls.append("close")

    with pytest.raises(RuntimeError, match="收尾失败") as failure:
        online.practice(tmp_path, key, client=client, case_tag=tag, bot_factory=RemoteBot)
    assert calls == ["match_start", "obs", "match_end", "close"]
    assert failure.value.matches == [read_json(path)]
    assert read_json(path)["case_tag"] == tag
    assert online.practice_replay_path(tmp_path, 100).read_bytes() == raw


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
