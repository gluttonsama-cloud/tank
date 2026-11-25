import gzip
import json
import time
from types import SimpleNamespace

import pytest
from seedcup.game import Game, generate_map
from seedcup.game.replay import ReplayRecorder

from tank import online, practice_host, sampling
from tank import sampling_data as data
from tank.workspace import read_json, write_json


def replay(mid, seed, side="red", *, rival=None, team_id=1):
    own = {"id": team_id, "name": "采样方", "bot": False}
    other = (
        {"id": None, "name": online.PRACTICE_NAMES["boss"], "bot": True}
        if rival is None
        else {"id": rival, "name": "对方", "bot": False}
    )
    red, blue = (own, other) if side == "red" else (other, own)
    game_map = generate_map(seed)
    meta = {"id": mid, "kind": "room", "room": "ABC123", "game": int(side == "blue"), "red": red, "blue": blue}
    game = Game(game_map)
    rec = ReplayRecorder(game_map, meta)
    rec.add(game.snapshot(), [])
    fx = game.step([{"forward": 0, "turn": 0, "fire": False} for _ in range(2)])
    rec.add(game.snapshot(), fx)
    return gzip.compress(json.dumps(rec.to_dict({"ticks": 1})).encode(), mtime=0)


def test_map_split_identity_and_enrich(tmp_path):
    a = data.ingest(tmp_path, replay(5, 612, rival=7), "team:7")
    b = data.ingest(tmp_path, replay(6, 612, "blue", rival=7), "team:7")
    assert [a["target_side"], b["target_side"]] == ["blue", "red"]
    assert a["split"] == b["split"] and a["grid_sha256"] == b["grid_sha256"]
    assert a["version_id"] is None
    with pytest.raises(ValueError, match="没有指定对手"):
        data.ingest(tmp_path, replay(7, 612, rival=7), "team:8")
    assert not online.practice_replay_path(tmp_path, 7).exists()
    with pytest.raises(ValueError, match="内容已变更"):
        data.ingest(tmp_path, replay(5, 613, rival=7), "team:7")
    out = tmp_path / "train.json"
    assert data.dataset(tmp_path, "team:7", split=a["split"], output=out)["maps"] == 1
    assert [r["id"] for r in read_json(out)] == [5, 6]


def test_passive_pull_only_downloads_finished_matching_games_once(tmp_path):
    rows = [
        {
            "id": mid,
            "kind": "room",
            "map_seed": 19,
            "red_id": 1,
            "blue_id": 7,
            "red_name": "本队",
            "blue_name": "对方",
            "status": status,
            "started_at": 123,
        }
        for mid, status in ((39, "finished"), (40, "running"))
    ]

    class Client:
        def __init__(self):
            self.user = {"id": 1}
            self.fetched = 0

        def request(self, method, path, **kwargs):
            assert method == "GET"
            if path == "/api/team/matches":
                return {"matches": rows}
            assert path == "/api/replays/39"
            self.fetched += 1
            return replay(39, 19, rival=7)

    client = Client()
    assert data.pull(tmp_path, "team:7", client=client)["ids"] == [39]
    assert data.pull(tmp_path, "team:7", client=client)["ids"] == [] and client.fetched == 1
    online.practice_replay_path(tmp_path, 39).unlink()
    assert data.pull(tmp_path, "team:7", client=client)["ids"] == [39] and client.fetched == 2
    with pytest.raises(ValueError, match="对手"):
        data.pull(tmp_path, "team:1", client=client)


def test_legacy_practice_binds_to_official_result(tmp_path):
    game = replay(61, 19)
    listing = {"id": 61, "map_seed": 19, "kind": "room", "red_id": 1, "blue_id": None, "started_at": 123}
    old = {"id": 61, "side": "red", "seed": None, "level": "boss", "result": {"ticks": 1}, "version_id": "old"}
    assert data.ingest(tmp_path, game, "boss", listing=listing, practice=old)["version_id"] == "old"
    with pytest.raises(ValueError, match="旧练习记录"):
        data.ingest(tmp_path, game, "boss", listing=listing, practice=old | {"result": {"ticks": 2}})


def test_sampling_requires_its_own_credentials_and_runs_without_foreground_server(tmp_path):
    write_json(online.account_path(tmp_path), {"username": "main", "password": "test"})
    with pytest.raises(ValueError, match="sample account"):
        sampling.check_host(tmp_path)
    write_json(online.sampling_account_path(tmp_path), {"username": "sample", "password": "test", "team_id": 2})
    sampling.check_host(tmp_path)
    practice_host.configure(tmp_path, "http://192.0.2.1:8767")
    with pytest.raises(ValueError, match="共享练习主机上"):
        sampling.check_host(tmp_path)


def test_sampling_authentication_never_uses_the_foreground_account(tmp_path, monkeypatch):
    calls = []

    def authenticate(username, password):
        calls.append(username)
        if password == "invalid":
            raise online.RemoteError(401, "登录失败")
        return SimpleNamespace(user={"id": 2, "must_change": username == "new"})

    monkeypatch.setattr(online, "OnlineClient", authenticate)
    write_json(online.account_path(tmp_path), {"username": "main", "password": "primary"})
    with pytest.raises(online.OnlineError, match="sample account"):
        online.sampling_client(tmp_path)
    assert online.save_sampling_account(tmp_path, "sample", "valid") == {"username": "sample", "team_id": 2}
    with pytest.raises(online.OnlineError, match="初始密码"):
        online.save_sampling_account(tmp_path, "new", "initial")
    assert online.sampling_client(tmp_path).user["id"] == 2
    path = online.sampling_account_path(tmp_path)
    write_json(path, read_json(path) | {"password": "invalid"})
    with pytest.raises(online.RemoteError, match="登录失败"):
        data.pull(tmp_path, "boss")
    assert calls == ["sample", "new", "sample", "sample"]


def test_missing_old_team_replay_reports_match_before_opening_another_room(tmp_path):
    with pytest.raises(online.OnlineError, match="tank online replay 50"):
        online.download_sample_replay(tmp_path, {"id": 50, "team_id": 1}, SimpleNamespace(user={"id": 2}))


def test_paused_between_sides_resumes_without_duplicate_training_rows(tmp_path, monkeypatch):
    key = "sample-" + "b" * 32
    v = "v-" + "2" * 64
    plan = {
        "mode": "active",
        "target": "boss",
        "width": 1,
        "maps": 1,
        "probes": ["peek"],
        "versions": {"peek": v},
        "seed": 802,
        "interval": 1,
    }
    with data.database(tmp_path) as db:
        db.execute(
            "INSERT INTO runs(id,plan,state,updated) VALUES (?,?,?,?)",
            (key, json.dumps(plan), "queued", time.time()),
        )
    team_id = 1
    monkeypatch.setattr(online, "sampling_client", lambda root: SimpleNamespace(user={"id": team_id}))
    paused = False

    def fake_room(root, version, *, client, case_tag, **kwargs):
        nonlocal paused

        def finish(mid, side):
            path = online.practice_replay_path(root, mid)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(replay(mid, kwargs["seed"], side, team_id=client.user["id"]))
            record = {
                "id": mid,
                "side": side,
                "version_id": version,
                "team_id": client.user["id"],
                "map_seed": kwargs["seed"],
                "room": "ABC123",
                "case_tag": case_tag,
                "level": "boss",
            }
            write_json(root / ".local/sampling/receipts" / case_tag / f"{mid}.json", record)
            return record

        if not paused:
            paused = True
            error = online.PracticeDeferred("队友优先")
            error.matches = [finish(50, "red")]
            raise error
        return {"matches": [finish(51, "red"), finish(52, "blue")]}

    monkeypatch.setattr(online, "collect_practice", fake_room)
    assert sampling.drive(tmp_path, key, plan, lambda: paused) == "paused"
    assert sampling.status(tmp_path, key)["games"] == 1
    team_id = 2
    assert sampling.drive(tmp_path, key, plan, lambda: False) == "completed"
    out = tmp_path / "actual.json"
    assert data.dataset(tmp_path, "boss", run=key, output=out)["games"] == 2
    entries = read_json(out)
    assert [e["id"] for e in entries] == [50, 52] and all(e["probe"] == "peek" for e in entries)
    assert [e["players"][side]["id"] for e, side in zip(entries, ("red", "blue"))] == [1, 2]


def test_restart_accepts_matching_tag_only(tmp_path):
    key = "sample-" + "4" * 32
    folder = tmp_path / ".local/online/practice/snapshot/boss"
    write_json(folder / "80.json", {"id": 80, "case_tag": key + "-3"})
    write_json(folder / "81.json", {"id": 81, "case_tag": key + "-4"})
    assert list(sampling.receipts(tmp_path, key, 3, "snapshot", "boss", recover=True)) == [80]


def test_rotation_keeps_matched_controls_and_distinct_seeds():
    plan = {
        "probes": None,
        "rotation": sampling.ROTATION,
        "width": 3,
        "versions": {p: p for p in sampling.probes.catalog()},
        "seed": 2**31 - 1,
    }
    groups = [sampling.case(plan, n) for n in range(24)]
    assert len({g[0] for g in groups}) == 8
    assert {"approach", "silent"}.issubset(g[1] for g in groups[:3])
    assert all(0 <= g[0] < 2**31 for g in groups)


def test_boss_start_freezes_pressure_and_late_game_mix(tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "check_host", lambda root: None)
    monkeypatch.setattr(sampling.probes, "snapshots", lambda root, names: {name: name for name in names})
    monkeypatch.setattr(sampling, "launch", lambda root, key, foreground: key)

    def open_plan(target, names=None):
        key = sampling.start(tmp_path, target=target, maps=4, seed=123, names=names)
        return sampling.record(tmp_path, key)["plan"]

    plan = open_plan("boss")
    cycle = [[sampling.case(plan, step)[1] for step in range(start, start + 3)] for start in range(0, 12, 3)]
    assert cycle[0] == ["approach", "silent", "pulse"]
    assert all("approach" in group for group in cycle)
    assert sum("lancer" in group for group in cycle) == 3
    assert {name for group in cycle for name in group} == set(sampling.probes.catalog())
    assert open_plan("rookie")["rotation"] == [list(group) for group in sampling.ROTATION]
    custom = open_plan("boss", ["peek"])
    assert custom["rotation"] is None and sampling.case(custom, 1)[1] == "peek"
