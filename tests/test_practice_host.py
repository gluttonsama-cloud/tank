from contextlib import contextmanager
import gzip
from pathlib import Path
from threading import Event, Thread
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tank import lan, online, practice_host, practice_priority, versions
from tank.workspace import read_json, write_json


TOKEN = b"test practice host token"


@pytest.fixture(autouse=True)
def foreground_account(monkeypatch):
    monkeypatch.setattr(online.OnlineClient, "login", lambda root: SimpleNamespace(user={"id": 57}))


def _snapshot(root, code="# fixed source\n"):
    strategy = root / "strategies/demo"
    strategy.mkdir(parents=True)
    (strategy / "submission.json").write_text('{"runtime":"python","entrypoint":"bot.py"}', encoding="utf-8")
    (strategy / "bot.py").write_text(code, encoding="utf-8")
    return versions.create(root, "demo")["id"]


@contextmanager
def _service(root):
    with practice_host.Coordinator(root, TOKEN) as coordinator, practice_host._server(coordinator, "127.0.0.1", 0) as server:
        coordinator.start()
        worker = Thread(target=server.serve_forever)
        worker.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            yield lan.Client(url, TOKEN)
        finally:
            server.shutdown()
            worker.join()


def _submit(version_id, *, seed=1):
    return {"request_id": uuid4().hex, "version_id": version_id, "level": "rookie", "games": 1, "seed": seed}


def _store_match(root, spec, tag, match_id):
    record = {"version_id": spec["version_id"], "level": spec["level"], "seed": spec["seed"],
              "id": match_id, "room": "AAAAAA", "score": 11.5, "case_tag": tag, "result": {}}
    write_json(online._practice_path(root, record), record)
    replay = online.practice_replay_path(root, match_id)
    replay.parent.mkdir(parents=True, exist_ok=True)
    replay.write_bytes(gzip.compress(b"official replay"))
    return record | {"replay_path": str(replay)}


def _wait(client, request_id):
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        response = client.request("/practice/result", {"request_id": request_id})
        if response["status"] not in ("queued", "running"):
            return response
        time.sleep(0.02)
    pytest.fail(f"练习请求未完成 {request_id}")


def test_known_snapshot_downloads_result_after_duplicate_submit(tmp_path, monkeypatch):
    host, player = (tmp_path / name for name in ("host", "player"))
    key = _snapshot(player)
    assert _snapshot(host) == key
    monkeypatch.setenv(practice_host.TOKEN_ENV, TOKEN.decode())
    monkeypatch.setenv("SEEDCUP_PASSWORD", "local-only-secret")
    monkeypatch.setattr(practice_host, "POLL_SECONDS", 0.01)
    calls = []

    def run(root, **spec):
        assert root == host
        assert practice_priority.foreground_pending(root, spec["client"].user["id"])
        versions.verify(root, spec["version_id"])
        calls.append(spec)
        match = _store_match(root, spec, spec["case_tag"], len(calls))
        return {"version_id": key, "level": spec["level"], "room": match["room"], "matches": [match]}

    monkeypatch.setattr(online, "practice", run)
    send = lan.Client.request
    interrupted = False

    def lose_ack(self, action, request):
        nonlocal interrupted
        assert "local-only-secret" not in str(request)
        response = send(self, action, request)
        if action == "/practice/submit" and response["status"] == "queued" and not interrupted:
            interrupted = True
            assert _wait(self, request["request_id"])["status"] == "completed"
            raise ConnectionError("模拟丢失已排队响应")
        return response

    monkeypatch.setattr(lan.Client, "request", lose_ack)
    with _service(host) as client:
        practice_host.configure(player, client.url)
        first = practice_host.practice(player, key, seed=42)
        assert practice_host.result(player, first["request_id"])["status"] == "completed"
        path = Path(first["matches"][0]["replay_path"])
        assert path.is_relative_to(player)
        assert gzip.decompress(path.read_bytes()) == b"official replay"
        assert online.history(player, key)[0]["score"] == 11.5

        second = practice_host.practice(player, key, seed=43)
        assert second["request_id"] != first["request_id"]
    assert [call["seed"] for call in calls] == [42, 43]


def test_host_enqueues_in_acceptance_order_and_checks_id_conflicts(tmp_path, monkeypatch):
    key = _snapshot(tmp_path)
    first, second, third = (_submit(key, seed=seed) for seed in (1, 2, 3))
    active = Event()
    release = Event()
    seeds = []

    def run(root, **spec):
        assert practice_priority.foreground_pending(root, spec["client"].user["id"])
        seeds.append(spec["seed"])
        if spec["seed"] == 1:
            active.set()
            assert release.wait(4)
        match = _store_match(root, spec, spec["case_tag"], spec["seed"])
        return {"version_id": key, "level": spec["level"], "room": match["room"], "matches": [match]}

    monkeypatch.setattr(online, "practice", run)
    with _service(tmp_path) as client:
        assert client.request("/practice/submit", first)["status"] == "queued"
        try:
            assert active.wait(4)
            assert client.request("/practice/submit", second)["status"] == "queued"
            assert client.request("/practice/submit", third)["status"] == "queued"
            assert client.request("/practice/submit", first)["status"] == "running"
            with pytest.raises(ValueError, match="不同的快照或参数"):
                client.request("/practice/submit", first | {"seed": 7})
        finally:
            release.set()
        assert _wait(client, third["request_id"])["status"] == "completed"
        with pytest.raises(ValueError, match="没有该官网对局"):
            client.request("/practice/replay", {"request_id": second["request_id"], "match_id": 1})
    assert seeds == [1, 2, 3]


def test_queue_reserves_authenticated_team_before_waiting_for_remote_strategy(tmp_path, monkeypatch):
    team_ids = iter((57, 99))
    monkeypatch.setattr(online.OnlineClient, "login", lambda root: SimpleNamespace(user={"id": next(team_ids)}))
    first_waiting, release = Event(), Event()
    observed = []

    def attach(relay):
        observed.append(tuple(practice_priority.foreground_pending(tmp_path, team) for team in (57, 99)))
        if len(observed) == 1:
            first_waiting.set()
            assert release.wait(4)
        raise online.OnlineError("结束模拟练习")

    monkeypatch.setattr(practice_host.Relay, "attach", attach)
    with _service(tmp_path) as client:
        first, second = (_submit("v-" + "a" * 64) for _ in range(2))
        client.request("/practice/submit", first)
        try:
            assert first_waiting.wait(4)
            client.request("/practice/submit", second)
        finally:
            release.set()
        assert _wait(client, second["request_id"])["status"] == "failed"
    assert observed == [(True, False), (False, True)]


def test_host_rejects_bad_token_and_snapshot_upload(tmp_path, monkeypatch):
    player, host = tmp_path / "player", tmp_path / "host"
    key = _snapshot(player)
    request = _submit(key)

    def unexpected_run(*_, **__):
        pytest.fail("无效请求不得进入官网练习")

    monkeypatch.setattr(online, "practice", unexpected_run)
    with _service(host) as client:
        with pytest.raises(ValueError, match="未通过认证"):
            lan.Client(client.url, b"different").request("/practice/submit", request)
        with pytest.raises(ValueError, match="只接受快照 ID 和练习参数"):
            client.request("/practice/submit", request | {"snapshot": {"files": {"bot.py": "ignored"}}})
        assert client.request("/practice/result", {"request_id": request["request_id"]})["status"] == "not_found"
    assert not (host / ".local/versions" / key).exists()


def test_unknown_snapshot_runs_only_at_requester_and_recovers_missing_replay(tmp_path, monkeypatch):
    host, player = (tmp_path / name for name in ("host", "player"))
    code = f"""\
import json
from pathlib import Path
import sys

for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "obs":
        forward = 1 if Path.cwd().is_relative_to(Path({str(player)!r})) else -1
        print(json.dumps({{"type": "action", "frame": message["frame"], "forward": forward, "turn": 0, "fire": False}}))
    elif message["type"] == "match_end":
        break
"""
    key = _snapshot(player, code)
    monkeypatch.setenv(practice_host.TOKEN_ENV, TOKEN.decode())
    frames = []

    def run(root, *, bot_factory, **spec):
        assert root == host
        assert practice_priority.foreground_pending(root, spec["client"].user["id"])
        assert not (root / ".local/versions" / key).exists()
        bot = bot_factory()
        bot.on_match_start({"match_id": 1, "side": "red"})
        for frame in range(5):
            action = bot.decide({"frame": frame})
            assert action == {"forward": 1, "turn": 0, "fire": False}
            frames.append(frame)
        bot.on_match_end({})
        match = _store_match(root, spec, spec["case_tag"], 1)
        online.practice_replay_path(root, 1).unlink()
        return {"version_id": key, "level": spec["level"], "room": match["room"], "matches": [match]}

    def archive(root, client, record, *, retry_pending):
        assert root == host and record["id"] == 1 and record["case_tag"].startswith("host:")
        return _store_match(root, record, record["case_tag"], 1)

    monkeypatch.setattr(online, "practice", run)
    monkeypatch.setattr(online, "_archive_practice_replay", archive)
    with _service(host) as client:
        practice_host.configure(player, client.url)
        completed = practice_host.practice(player, key, seed=42)
        assert "replay_path" not in completed["matches"][0]
        assert online.history(player, key)[0]["score"] == 11.5
        replay = practice_host.replay(player, completed["matches"][0]["id"])
        assert Path(replay["replay_path"]).is_relative_to(player)
        assert gzip.decompress(Path(replay["replay_path"]).read_bytes()) == b"official replay"
    assert frames == list(range(5))
    assert not list(host.glob(".local/versions/*"))


def test_unknown_snapshot_without_attached_client_never_starts_official_practice(tmp_path, monkeypatch):
    remote_key = "v-" + "a" * 64
    monkeypatch.setattr(practice_host, "RELAY_ATTACH_TIMEOUT", 0.10)

    def unexpected_run(*_, **__):
        pytest.fail("转发客户端未挂上，不得开官网房间")

    monkeypatch.setattr(online, "practice", unexpected_run)
    with _service(tmp_path) as client:
        request = _submit(remote_key)
        assert client.request("/practice/submit", request)["relay"]
        terminal = _wait(client, request["request_id"])
        assert terminal["status"] == "failed"
        assert "发起工作台未连接" in terminal["error"]


def test_interrupted_request_recovers_saved_match_without_relaunch(tmp_path, monkeypatch):
    key = _snapshot(tmp_path)
    request = _submit(key)
    request_id = request["request_id"]
    with practice_host.Coordinator(tmp_path, TOKEN) as old:
        assert old.handle("/practice/submit", request)["status"] == "queued"
        with old.db:
            old.db.execute("UPDATE requests SET status = 'running' WHERE request_id = ?", (request_id,))
    _store_match(tmp_path, request, f"host:{request_id}", 99)
    _store_match(tmp_path, request, "unrelated", 100)

    def unexpected_run(*_, **__):
        pytest.fail("主机重启不得重复启动状态未知的对局")

    monkeypatch.setattr(online, "practice", unexpected_run)
    with _service(tmp_path) as client:
        response = client.request("/practice/submit", request)
        assert response["status"] == "interrupted"
        assert [match["id"] for match in response["result"]["matches"]] == [99]


def test_local_config_allows_host_loopback_and_restores_direct_practice(tmp_path, monkeypatch):
    assert not practice_host.is_remote(tmp_path)
    practice_host.configure(tmp_path, "http://practice.example:8767")
    assert practice_host.is_remote(tmp_path)
    practice_host.configure(tmp_path, "http://127.0.0.1:8767")
    assert not practice_host.is_remote(tmp_path)
    assert practice_host.configure(tmp_path, None) == {"mode": "local"}
    monkeypatch.setattr(online, "practice", lambda root, version_id, **spec: (root, version_id, spec["seed"]))
    assert practice_host.practice(tmp_path, "local-key", seed=8) == (tmp_path, "local-key", 8)


def test_client_saves_token_and_env_overrides_without_reprompt(tmp_path, monkeypatch):
    monkeypatch.delenv(practice_host.TOKEN_ENV, raising=False)
    monkeypatch.setattr(practice_host.sys.stdin, "isatty", lambda: True)
    answers = iter(("initial-token", "replacement-token"))
    monkeypatch.setattr(practice_host, "getpass", lambda _: next(answers))
    url = "http://practice.example:8767"
    assert practice_host.configure(tmp_path, url, prompt=True)["url"] == url
    assert practice_host._token(tmp_path) == b"initial-token"
    practice_host.configure(tmp_path, url, prompt=True)
    practice_host.configure(tmp_path, url, prompt=True, replace_token=True)
    assert practice_host._token(tmp_path) == b"replacement-token"
    monkeypatch.setenv(practice_host.TOKEN_ENV, "working-token")
    assert practice_host._token(tmp_path) == b"working-token"
    assert read_json(practice_host._token_path(tmp_path))["token"] == "replacement-token"


def _wait_for_file(path):
    deadline = time.monotonic() + 8
    while not path.is_file():
        if time.monotonic() >= deadline:
            pytest.fail(f"主机未写入监听状态 {path}")
        time.sleep(0.02)
    return read_json(path)


def test_background_process_confirms_listen_and_does_not_spawn_duplicate(tmp_path, monkeypatch):
    (tmp_path / "resource").mkdir()
    (tmp_path / "resource/SHA256SUMS").touch()
    monkeypatch.setenv("TANK_ROOT", str(tmp_path))
    monkeypatch.delenv(practice_host.TOKEN_ENV, raising=False)
    launched = practice_host.launch(tmp_path, port=0)
    try:
        assert launched["status"] == "running"
        assert launched["port"] > 0
        token_path = Path(launched["token_path"])
        token = read_json(token_path)["token"]
        assert len(token) >= 32
        assert token not in str(launched)
        client = lan.Client(f"http://127.0.0.1:{launched['port']}", token.encode())
        assert client.request("/practice/result", {"request_id": uuid4().hex})["status"] == "not_found"
        with monkeypatch.context() as patch:
            def unexpected_spawn(*_, **__):
                pytest.fail("后台服务已在监听，不应额外创建进程")

            patch.setattr(practice_host.subprocess, "Popen", unexpected_spawn)
            running = practice_host.launch(tmp_path, port=0)
            assert running["status"] == "already_running"
            assert running["port"] == launched["port"]
    finally:
        assert practice_host.stop(tmp_path)["status"] == "stopping"
        info_path = practice_host._server_info(tmp_path)
        deadline = time.monotonic() + 8
        while info_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not info_path.is_file()
    assert token not in Path(launched["log"]).read_text(encoding="utf-8")


def test_stop_finishes_active_request_and_preserves_queued_requests(tmp_path, monkeypatch):
    key = _snapshot(tmp_path)
    first, second = _submit(key), _submit(key)
    started = Event()
    release = Event()
    monkeypatch.setattr(practice_host, "STOP_GRACE_SECONDS", 0.2)

    def run(root, **spec):
        assert spec["case_tag"] == f"host:{first['request_id']}"
        started.set()
        assert release.wait(6)
        match = _store_match(root, spec, spec["case_tag"], 1)
        return {"version_id": key, "level": spec["level"], "room": match["room"], "matches": [match]}

    monkeypatch.setattr(online, "practice", run)
    worker = Thread(target=practice_host.serve, kwargs={"root": tmp_path, "port": 0, "token": TOKEN})
    worker.start()
    info = _wait_for_file(practice_host._server_info(tmp_path))
    client = lan.Client(f"http://{info['host']}:{info['port']}", TOKEN)
    try:
        assert client.request("/practice/submit", first)["status"] == "queued"
        assert started.wait(4)
        assert client.request("/practice/submit", second)["status"] == "queued"
        assert practice_host.host_running(tmp_path)
        assert practice_host.stop(tmp_path)["status"] == "stopping"
        assert not practice_host.host_running(tmp_path)
        with pytest.raises(ValueError, match="练习主机正在停止"):
            client.request("/practice/submit", _submit(key))
        assert worker.is_alive()
        release.set()
        assert _wait(client, first["request_id"])["status"] == "completed"
        assert client.request("/practice/result", {"request_id": second["request_id"]})["status"] == "queued"
    finally:
        release.set()
        practice_host.stop(tmp_path)
        worker.join(timeout=7)
    assert not worker.is_alive()


def test_local_workbench_joins_same_queue_as_remote_peers(tmp_path, monkeypatch):
    key = _snapshot(tmp_path)
    first, second = _submit(key, seed=1), _submit(key, seed=2)
    started, release, local_queued = Event(), Event(), Event()
    runs, local_result = [], []
    monkeypatch.setenv(practice_host.TOKEN_ENV, TOKEN.decode())
    monkeypatch.setattr(practice_host, "STOP_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(practice_host, "POLL_SECONDS", 0.02)

    def run(root, **spec):
        seed = spec["seed"]
        runs.append(seed)
        if seed == 1:
            started.set()
            assert release.wait(6)
        match = _store_match(root, spec, spec["case_tag"], seed)
        return {"version_id": key, "level": spec["level"], "room": match["room"], "matches": [match]}

    send = lan.Client.request

    def observe_local(self, action, request):
        response = send(self, action, request)
        if action == "/practice/submit" and request["seed"] == 3:
            assert response["status"] == "queued"
            local_queued.set()
        return response

    monkeypatch.setattr(online, "practice", run)
    monkeypatch.setattr(lan.Client, "request", observe_local)
    worker = Thread(target=practice_host.serve, kwargs={"root": tmp_path, "host": "0.0.0.0", "port": 0, "token": TOKEN})
    worker.start()
    info = _wait_for_file(practice_host._server_info(tmp_path))
    client = lan.Client(f"http://127.0.0.1:{info['port']}", TOKEN)
    local = Thread(target=lambda: local_result.append(practice_host.practice(tmp_path, key, seed=3)))
    try:
        assert client.request("/practice/submit", first)["status"] == "queued"
        assert started.wait(4)
        assert client.request("/practice/submit", second)["status"] == "queued"
        local.start()
        assert local_queued.wait(4)
        release.set()
        local.join(timeout=6)
        assert local_result[0]["matches"][0]["seed"] == 3
    finally:
        release.set()
        practice_host.stop(tmp_path)
        worker.join(timeout=7)
    assert runs == [1, 2, 3]
