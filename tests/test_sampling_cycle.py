import json
from collections import Counter

from tank import sampling_adaptive as adaptive
from tank import sampling_data as data
from tank import sampling_opportunities as opportunities
from tank.sampling_window_timing import WindowTiming


def policy():
    names = ["specialist", "lancer", "recontact", "retreat", "silent", "peek", "late-zone",
             *adaptive.ARMS, adaptive.RESERVED_PROBE]
    return {"start_step": 0, "cycle_start_index": 0, "anchors": names[:2], "versions": dict(zip(names, names))}


def test_frozen_eight_training_maps_interleaved_with_holdouts(tmp_path, monkeypatch):
    key = "sample-" + "8" * 32
    start = 123
    splits = ["blind", "train", "development", *(["train"] * 7), "holdout"]
    map_splits = {start + i * 104729: (str(i), s) for i, s in enumerate(splits)}
    calls = []
    monkeypatch.setattr(adaptive, "layout_assignment", lambda root, seed: map_splits[seed])
    monkeypatch.setattr(opportunities, "history", lambda root: calls.append(1) or [])
    with data.database(tmp_path) as db:
        plan = {"width": 3, "seed": start, "adaptive": policy()}
        db.execute("INSERT INTO runs(id,plan,state,updated) VALUES (?,?,?,0)", (key, json.dumps(plan), "running"))
    maps = [adaptive.decision(tmp_path, key, i) for i in range(len(splits))]
    assert [adaptive.decision(tmp_path, key, i) for i in range(len(splits))] == maps
    train = [m for m in maps if m["planned_split"] == "train"]
    assert len(calls) == 8
    assert Counter(m["kind"] for m in train) == {"gap": 4, "intervention": 2, "explore": 1, "repeat": 1}
    assert [m["picks"][0]["probe"] for m in train] == ["specialist", "lancer"] * 4
    assert [m["sequence"] for m in train] == list(range(1, 9))
    assert all(m["cycle"] == 1 and len(m["picks"]) == 3 for m in train)
    assert all(p["probe"] != adaptive.RESERVED_PROBE for m in train for p in m["picks"])
    assert maps[0]["picks"][2]["probe"] == adaptive.RESERVED_PROBE
    assert all("opportunity_layouts" not in m for m in maps if m["kind"] == "fixed")
    for m in train:
        if m["kind"] == "repeat":
            reference, repeat, _ = m["picks"]
            assert repeat["version_id"] == reference["version_id"] and repeat["repeat"] == 2
        elif m["kind"] == "intervention":
            assert {p["probe"] for p in m["picks"][1:]} == set(adaptive.ARMS)


def test_gap_selection_uses_observable_results_on_distinct_layouts():
    def observed(outcome):
        return {"first_close": {"outcome": "stayed"}, "first_close_hit": {"outcome": outcome}}
    rows = [(str(i), n, observed("withdrawn" if n == "peek" else "ended"))
            for i in range(6) for n in policy()["versions"] if n != adaptive.RESERVED_PROBE]
    m = adaptive.opportunity_plan(policy(), 0, rows * 3, 123, "g", "train")
    assert m["picks"][1]["probe"] == "peek"
    assert m["opportunity_layouts"]["first_close_hit"] == {"eligible": 6, "withdrawn": 6, "stayed": 0, "ended": 6}


def test_online_trace_requires_official_acknowledgement_of_the_trigger_frame():
    grid = [[1] * 7, [1, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [1] * 7]
    state = {"x": 2., "y": 1.5, "heading": 0., "alive": True, "hp": 100, "reload": 0.}
    obs = {"frame": 8, "self": state, "enemy": state | {"x": 4.}, "last_action": None}
    for fire in (False, True):
        trace = WindowTiming({"map": {"grid": grid}})
        trace.observe(obs, {"fire": fire})
        assert trace.record["source_frame"] == 8 and trace.record["confirmed_uses"] == 0
        for current, source, applied in [(9, 7, 8), (10, 8, 9), (11, 8, 10), (12, 8, 10)]:
            ack = {"frame": applied, "source_frame": source, "fire": fire}
            trace.observe(obs | {"frame": current, "last_action": ack}, {"fire": False})
        assert trace.record == {"source_frame": 8, "command_fire": fire, "first_applied_frame": 9,
            "last_applied_frame": 10, "confirmed_uses": 2, "applied_fire": fire}
