import json

from tank import sampling
from tank import sampling_adaptive as adaptive
from tank import sampling_data as data
from tank.workspace import write_json


def policy():
    names = ["specialist", "lancer", "recontact", "peek", "retreat", "late-zone", "approach", "silent", "pulse"]
    return {"start_step": 0, "anchors": names[:2], "versions": {k: k for k in names}}


def events(value):
    return dict.fromkeys(adaptive.EVENTS, value)


def test_selection_changes_with_realized_events_and_counts_layouts_once():
    p = policy()
    def pick(winner):
        rows = [(str(i), name, events(name == winner))
                for name in p["versions"] for i in range(6)]
        return adaptive.select_map(p, 1, rows * 2)

    for winner in ["recontact", "peek"]:
        m = pick(winner)
        assert m["picks"][1]["probe"] == winner
        assert m["training_layouts"] == 6 and m["training_games"] == 108
        assert m["event_layouts"] == dict.fromkeys(adaptive.EVENTS, 6)
    for index, control in [(7, "silent"), (15, "pulse")]:
        m = adaptive.select_map(p, index, [])
        assert [r["probe"] for r in m["picks"]][1:] == ["approach", control]


def test_only_training_layouts_feed_selection(tmp_path):
    entries = [(1, "train", "a"), (2, "development", "b"), (3, "blind", "c"), (4, "train", "d")]
    with data.database(tmp_path) as db:
        db.execute("INSERT INTO sampling_holdouts VALUES ('d')")
        for mid, split, grid in entries:
            row = {"split": split, "grid_sha256": grid, "version_id": "lancer", "path": "absent"}
            cov = {"encounters_v1": events(True)} if mid == 1 else {}
            db.execute("INSERT INTO examples VALUES (?,?,?,?)", (mid, "boss", json.dumps(row), json.dumps(cov)))
    assert adaptive.history(tmp_path) == [("a", "lancer", events(True))]


def test_new_holdout_or_snapshot_during_selection_recomputes_before_freeze(tmp_path, monkeypatch):
    key = "sample-" + "f" * 32
    plan = {"seed": 4, "width": 3, "adaptive": policy()}
    with data.database(tmp_path) as db:
        db.execute("INSERT INTO runs(id,plan,state,updated) VALUES (?,?,?,0)", (key, json.dumps(plan), "running"))
    calls = []

    def observations(root):
        if not calls:
            plan["adaptive"]["versions"]["specialist"] = "new"
            with data.database(root) as db:
                db.execute("UPDATE runs SET plan=? WHERE id=?", (json.dumps(plan), key))
                db.execute("INSERT INTO sampling_holdouts VALUES ('reserved')")
            calls.append(1)
            return [("reserved", "lancer", events(True))]
        calls.append(2)
        return []

    monkeypatch.setattr(adaptive, "history", observations)
    m = adaptive.decision(tmp_path, key, 0)
    assert calls == [1, 2] and m["training_games"] == 0 and m["picks"][0]["version_id"] == "new"


def test_continuous_default_and_migration_freeze_complete_maps(tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "check_host", lambda _: None)
    monkeypatch.setattr(sampling, "launch", lambda root, key, foreground: key)
    monkeypatch.setattr(sampling.probes, "snapshots", lambda root, names: {k: k for k in names})
    monkeypatch.setattr(sampling.versions, "create", lambda *args: {"id": "specialist"})
    monkeypatch.setattr(sampling.versions, "verify", lambda *args: None)
    key = sampling.start(tmp_path, target="boss", seed=500)
    p = sampling.record(tmp_path, key)["plan"]
    assert p["adaptive"]["start_step"] == 0
    sample = sampling.assigned_case(tmp_path, key, p, 0)
    assert sampling.status(tmp_path, key)["map_plan"]["picks"][0]["version_id"] == sample[2]
    del p["adaptive"]
    p["versions"].pop("recontact")
    with data.database(tmp_path) as db:
        db.execute("DELETE FROM sampling_maps WHERE run=?", (key,))
        db.execute("UPDATE runs SET plan=?,step=2 WHERE id=?", (json.dumps(p), key))
    protected = tmp_path / "split.json"
    write_json(protected, {"dev": [{"grid_sha256": "1" * 64}], "blind": [{"grid_sha256": "2" * 64}]})
    before = sampling.case(p, 2)
    sampling.adapt(tmp_path, key, "official-build", protected)
    upgraded = sampling.record(tmp_path, key)["plan"]
    assert sampling.assigned_case(tmp_path, key, upgraded, 2) == before
    assert upgraded["adaptive"]["start_step"] == 3
    first_map = [sampling.assigned_case(tmp_path, key, upgraded, s) for s in (3, 4, 5)]
    sampling.adapt(tmp_path, key, "next-build")
    updated = sampling.record(tmp_path, key)["plan"]
    assert [sampling.assigned_case(tmp_path, key, updated, s) for s in (3, 4, 5)] == first_map
    assert sampling.assigned_case(tmp_path, key, updated, 6)[2] == "lancer"
    assert sampling.assigned_case(tmp_path, key, updated, 9)[2] == "next-build"
    custom = sampling.start(tmp_path, target="boss", seed=500, names=["peek"])
    assert "adaptive" not in sampling.record(tmp_path, custom)["plan"]


def test_finite_experiment_and_coverage_sampler_alternate_complete_maps(tmp_path, monkeypatch):
    from itertools import count
    from types import SimpleNamespace

    ticks = count(0, 60)
    monkeypatch.setattr(sampling, "time", SimpleNamespace(time=lambda: next(ticks), monotonic=lambda: next(ticks),
                                                          sleep=lambda _: None))
    monkeypatch.setattr(sampling, "check_host", lambda _: None)
    monkeypatch.setattr(sampling, "launch", lambda root, key, foreground: key)
    monkeypatch.setattr(sampling.probes, "snapshots", lambda root, names: {k: k for k in names})
    monkeypatch.setattr(sampling.versions, "create", lambda *args: {"id": "specialist"})
    monkeypatch.setattr(sampling.versions, "verify", lambda *args: None)
    monkeypatch.setattr(sampling.online, "sampling_client", lambda _: object())
    monkeypatch.setattr(sampling, "receipts", lambda *args, **kwargs: {})
    parent = sampling.start(tmp_path, target="boss", seed=4071)
    batch = sampling.enqueue(tmp_path, parent, strategies=[("a", "version-a"), ("b", "version-b")],
                             maps=2, repeats=2, seed=900)["id"]
    cases = []
    def archived(root, key, step, plan, records, client):
        cases.append(key)
        return True
    monkeypatch.setattr(sampling, "save_cases", archived)
    sampling.drive(tmp_path, parent, sampling.record(tmp_path, parent)["plan"], lambda: cases.count(parent) == 6)
    assert cases == ([batch] * 4 + [parent] * 3) * 2
    assert sampling.record(tmp_path, batch)["state"] == "completed"


def test_paired_tier_experiments_keep_separate_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "check_host", lambda _: None)
    monkeypatch.setattr(sampling, "launch", lambda root, key, foreground: key)
    monkeypatch.setattr(sampling.probes, "snapshots", lambda *args: {"peek": "frozen"})
    monkeypatch.setattr(sampling.versions, "verify", lambda *args: None)
    parent = sampling.start(tmp_path, target="boss", names=["peek"], seed=4)
    pair = {"strategies": [("A", "fixed-a"), ("B", "fixed-b")], "maps": 1, "repeats": 1, "seed": 17}
    mid = sampling.enqueue(tmp_path, parent, target="mid", **pair)["id"]
    assert sampling.enqueue(tmp_path, parent, target="mid", **pair)["id"] == mid
    boss = sampling.enqueue(tmp_path, parent, **pair)["id"]
    assert mid != boss and sampling.record(tmp_path, mid)["plan"]["target"] == "mid"
    assert sampling.record(tmp_path, boss)["plan"]["target"] == "boss"
    assert sampling.record(tmp_path, parent)["plan"]["target"] == "boss"
