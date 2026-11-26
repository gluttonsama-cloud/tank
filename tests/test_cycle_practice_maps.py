import gzip
import hashlib
import json

import pytest
from seedcup.game import SIDES, Game, generate_map, rules
from seedcup.game.runner import match_info

from tank import cycle_pve, cycle_store, sampling, sampling_data, versions

PARENT = "sample-" + "1" * 32


def _first_seed(tier="boss", number=1):
    digest = hashlib.sha256(f"pve-accept:new:old:{number}:{tier}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 2**31


def _root(root):
    with sampling_data.database(root) as db:
        plan = {"mode": "active", "maps": None, "width": 1, "interval": 10, "target": "boss",
                "seed": 100, "probes": ["parent"], "versions": {"parent": "old"}}
        db.execute("INSERT INTO runs(id,plan,state,updated) VALUES(?,?,'scheduled',1)",
                   (PARENT, json.dumps(plan)))
    with cycle_store.database(root) as db:
        db.execute("INSERT INTO lines VALUES('pve',1)")


def _gz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = gzip.compress(json.dumps(data).encode(), mtime=0)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


@pytest.mark.parametrize("collision", ["holdout", "online_replay", "sdk_capture", "local_train", "duplicate"])
def test_gate_selects_five_unused_layouts_and_replays_same_reservation(tmp_path, monkeypatch, collision):
    from training import cycle_train

    seed = _first_seed()
    attempted = cycle_pve._gate_grids(seed)
    occupied = attempted[seed]
    game_map = generate_map(seed)
    if collision == "holdout":
        with sampling_data.database(tmp_path) as db:
            db.execute("INSERT INTO sampling_holdouts VALUES (?)", (occupied,))
    elif collision == "online_replay":
        _gz(tmp_path / ".local/online/replays/1.json.gz", {"map": game_map.to_dict()})
    elif collision == "sdk_capture":
        _gz(tmp_path / ".local/sampling/observations/2.json.gz", {"start": match_info(game_map, 0, "boss")})
    elif collision == "local_train":
        monkeypatch.setattr(cycle_train, "local_training_grids", lambda root: {occupied})
    else:
        regular = cycle_pve._gate_grids

        def duplicate(value):
            grids = regular(value)
            if value == seed:
                grids[next(reversed(grids))] = occupied
            return grids

        monkeypatch.setattr(cycle_pve, "_gate_grids", duplicate)

    first = cycle_pve._gate_maps(tmp_path, PARENT, "new", "old", 1, "boss")
    new_seed, grids, source, run = first
    assert new_seed != seed
    assert source == "fresh"
    assert run is None
    assert len(set(grids.values())) == 5
    assert occupied not in grids.values()
    with sampling_data.database(tmp_path) as db:
        assert set(grids.values()) <= {row[0] for row in db.execute("SELECT grid FROM sampling_holdouts")}
    monkeypatch.setattr(cycle_pve, "_archived_and_local_grids",
                        lambda root: pytest.fail("repeat reservation scanned archived games"))
    assert cycle_pve._gate_maps(tmp_path, PARENT, "new", "old", 1, "boss") == first
    with pytest.raises(ValueError, match="地图预留"):
        cycle_pve._gate_maps(tmp_path, PARENT, "different", "old", 1, "boss")


def test_adopts_legacy_boss_batch_after_schedule_has_started(tmp_path, monkeypatch):
    _root(tmp_path)
    monkeypatch.setattr(versions, "verify", lambda *args: None)
    seed = _first_seed()
    original = sampling.enqueue(tmp_path, PARENT, strategies=[("candidate", "new"), ("incumbent", "old")],
                                maps=5, repeats=1, seed=seed, target="boss")
    with sampling_data.database(tmp_path) as db:
        db.execute("UPDATE runs SET state='running',step=1 WHERE id=?", (original["id"],))
        db.execute("INSERT INTO sampling_holdouts VALUES (?)",
                   (cycle_pve._gate_grids(seed)[seed],))
    monkeypatch.setattr(cycle_pve, "_archived_and_local_grids",
                        lambda root: pytest.fail("an existing batch must reuse its original plan"))
    monkeypatch.setattr(sampling, "enqueue", lambda *a, **k: pytest.fail("existing batch must not requeue"))
    report = cycle_pve.gate(tmp_path, PARENT, "new", "old", 1)
    assert report["batches"] == {"boss": original["id"]}
    assert report["map_reservations"]["boss"]["source"] == "legacy"
    assert report["map_reservations"]["boss"]["seeds"] == list(cycle_pve._gate_grids(seed))
    assert sampling.record(tmp_path, original["id"])["step"] == 1


def _capture_sdk(root, batch, mid, step, side, seed, version, tier):
    game_map = generate_map(seed)
    game = Game(game_map, tick=rules.GAME_TICKS - 1)
    for index, tank in enumerate(game.tanks):
        tank.x, tank.y = rules.ZONE_CX + (-1 if index == 0 else 1), rules.ZONE_CY
        if index == side:
            tank.hp = 90 if version == "new" else 60
        else:
            tank.hp = 75

    act = {"forward": 0.0, "turn": 0.0, "fire": False}
    frames = [[game.observe(side) | {"type": "obs", "match_id": mid}, act]]
    game.step([act, act])
    result = game.result_summary()
    practice = {"version_id": version, "level": tier, "seed": seed, "team_id": 57,
                "room": f"room-{mid}", "case_tag": f"{batch}-{step}"}
    data = {"format": "seedcup-sdk-observations-v1", "practice": practice, "frames": frames,
            "start": match_info(game_map, side, tier, match_id=mid, kind="room", game=side),
            "end": {"type": "match_end", "match_id": mid, "side": SIDES[side], "result": result}}
    file = f".local/sampling/observations/{mid}.json.gz"
    checksum = _gz(root / file, data)
    return practice | {"id": mid, "side": SIDES[side], "map_seed": seed, "result": result,
                       "observation_file": file, "observation_sha256": checksum}


def test_four_tier_gate_scores_archived_sdk_matches(tmp_path, monkeypatch):
    _root(tmp_path)
    monkeypatch.setattr(versions, "verify", lambda *args: None)
    seen = set()
    mid = 45000
    for target in ("boss", "rookie", "mid", "strong"):
        report = cycle_pve.gate(tmp_path, PARENT, "new", "old", 2)
        assert report["status"] == "collecting"
        assert report["tier"] == target
        batch = report["batches"][target]
        map_plan = report["map_reservations"][target]
        assert map_plan["source"] == "fresh"
        assert seen.isdisjoint(map_plan["grid_sha256"])
        seen.update(map_plan["grid_sha256"])
        plan = sampling.record(tmp_path, batch)["plan"]
        for step in range(plan["maps"] * plan["width"]):
            seed, _, version = sampling.assigned_case(tmp_path, batch, plan, step)
            receipts = []
            for side in range(2):
                mid += 1
                receipts.append(_capture_sdk(tmp_path, batch, mid, step, side, seed, version, target))
            assert sampling.save_cases(tmp_path, batch, step, plan, receipts, None)
        sampling.update(tmp_path, batch, "completed")
        assert sampling.status(tmp_path, batch)["sdk_games"] == 20
    report = cycle_pve.gate(tmp_path, PARENT, "new", "old", 2)
    assert report["status"] == "accepted"
    assert report["games"] == 80
    assert report["layout_deltas"] == {tier: [30.0] * 5 for tier in ("boss", "rookie", "mid", "strong")}
    assert report["estimated_40_game_gain"] == 1200
    with sampling_data.database(tmp_path) as db:
        assert len(seen) == 20
        assert seen == {row[0] for row in db.execute("SELECT grid FROM sampling_holdouts")}


def test_small_unstable_boss_gain_does_not_queue_other_tiers(tmp_path, monkeypatch):
    gains = [2.4, 51.5, 15.0, -10.4, -56.35]
    def boss_maps(root, parent, candidate, incumbent, number, tier):
        assert tier == "boss"
        return 0, dict.fromkeys(range(5)), "fresh", "completed-boss"
    monkeypatch.setattr(cycle_pve, "_gate_maps", boss_maps)
    monkeypatch.setattr(sampling, "status", lambda *_: {"id":"completed-boss","state":"completed","games":20})
    scores = {(i, version, side): 100 + (gain if version == "new" else 0)
              for i, gain in enumerate(gains) for version in ("old","new") for side in ("red","blue")}
    monkeypatch.setattr(cycle_pve, "_scores", lambda *_: scores)
    report = cycle_pve.gate(tmp_path, PARENT, "new", "old", 1)
    assert report["status"] == "rejected" and report["boss_mean_gain_per_game"] > 0
    assert report["boss_ci95_per_game"][0] < 0
