from types import SimpleNamespace

import pytest

from tank import cycle, cycle_ladder, cycle_pve, cycle_release, cycle_store


def result(**overrides):
    return dict(
        {
            "round": 1,
            "signature": "one",
            "candidate": "new",
            "incumbent": "old",
            "accepted": True,
        },
        **overrides,
    )


def enable(root, line):
    with cycle_store.database(root) as db:
        db.execute("INSERT INTO lines VALUES(?,1)", (line,))


def test_official_pve_submit_never_selects_ladder(tmp_path, monkeypatch):
    enable(tmp_path, "pve")
    cycle_store.put(tmp_path, "pve", incumbent="old", score=8000, sampler="parent")
    monkeypatch.setattr(cycle_pve, "gate", lambda *args, **kwargs: {"status": "accepted"})
    monkeypatch.setattr(
        cycle_release,
        "official_score",
        lambda *args, **kwargs: {"state": "completed", "submission": 2, "score": 8200},
    )
    monkeypatch.setattr(
        cycle_ladder,
        "request_activation",
        lambda *args: pytest.fail("PvE changed ladder"),
    )
    cycle._process_result(tmp_path, "pve", result(), "old", None)
    assert cycle_store.get(tmp_path, "pve", "incumbent") == "new"
    assert cycle_store.history(tmp_path, "pve")[0]["outcome"] == "promoted"


@pytest.mark.parametrize("score,expected_calls", [(4332.99, 0), (4333, 1)])
def test_pvp_qualification_and_pending(tmp_path, monkeypatch, score, expected_calls):
    enable(tmp_path, "pvp")
    calls = []
    monkeypatch.setattr(
        cycle_release,
        "official_score",
        lambda *args, **kwargs: {"state": "completed", "submission": 7, "score": score},
    )

    def pending(root, version, submission, **kwargs):
        calls.append(submission)
        return {"active_version": "old", "activation": {"status": "pending"}}

    monkeypatch.setattr(cycle_ladder, "request_activation", pending)
    cycle._process_result(tmp_path, "pvp", result(), "old", 3)
    assert len(calls) == expected_calls
    hist = cycle_store.history(tmp_path, "pvp")
    assert not hist if expected_calls else hist[0]["outcome"] == "rejected"


def test_frozen_collection_map_resume(tmp_path):
    plan = {
        "width": 2,
        "seed": 123,
        "cycle": {"roster": {"incumbent": "A", "candidate": "B"}},
    }
    first = cycle_pve.collection_case(tmp_path, "parent", plan, 0)
    plan["cycle"]["roster"]["candidate"] = "C"
    second = cycle_pve.collection_case(tmp_path, "parent", plan, 1)
    assert {first[2], second[2]} == {"A", "B"} and first[0] == second[0]
    assert {cycle_pve.collection_case(tmp_path, "parent", plan, i)[2] for i in (2, 3)} == {"A", "C"}


def test_external_replacement_and_duplicate_recovery(tmp_path):
    enable(tmp_path, "pvp")
    cycle_store.put(tmp_path, "pvp", selection_target=5)
    cycle._process_result(tmp_path, "pvp", result(), "other", 8)
    assert cycle_store.get(tmp_path, "pvp", "selection_target") is None
    prior = cycle_store.history(tmp_path, "pvp")
    assert prior[0]["outcome"] == "superseded"
    cycle_store.put(tmp_path, "pvp", selection_target=9)
    cycle_store.finish(tmp_path, "pvp", result(), "promoted", "旧操作重试", changes={"selection_target": 5})
    assert cycle_store.history(tmp_path, "pvp") == prior
    assert cycle_store.get(tmp_path, "pvp", "selection_target") == 9


def test_gate_reserves_maps_before_sampling(tmp_path):
    _, maps, source, _ = cycle_pve._gate_maps(tmp_path, "parent", "new", "old", 1, "boss")
    assert source == "fresh"
    with cycle_pve.sampling_data.database(tmp_path) as db:
        held = {r[0] for r in db.execute("SELECT grid FROM sampling_holdouts")}
    assert len(maps) == 5 and set(maps.values()) <= held


def test_report_advances_without_new_replays(tmp_path):
    args = SimpleNamespace(op="report", line="pvp", since=0)
    first = cycle.dispatch(tmp_path, args)
    cycle_store.finish(tmp_path, "pvp", result(), "rejected", "本地未提升")
    args.since = first["as_of"]
    second = cycle.dispatch(tmp_path, args)
    assert len(second["pvp"]["history"]) == 1
    args.since = second["as_of"]
    assert cycle.dispatch(tmp_path, args)["pvp"]["history"] == []


def test_late_completed_round_is_resumed_and_recorded_once(tmp_path):
    cycle_store.finish(tmp_path, "pve", result(candidate=None), "superseded", "工作进程已中断")
    cycle_store.resume_completed_result(tmp_path, "pve", 1)
    assert not cycle_store.history(tmp_path, "pve")
    cycle_store.finish(tmp_path, "pve", result(), "promoted", "正式成绩提升", changes={"incumbent": "new"})
    (row,) = cycle_store.history(tmp_path, "pve")
    assert row["outcome"] == "promoted" and row["prior_decision"]["outcome"] == "superseded"
    assert cycle_store.get(tmp_path, "pve", "incumbent") == "new"


def test_single_slot_legacy_sampler_still_pairs_both_versions(tmp_path):
    plan = {
        "width": 1,
        "seed": 501,
        "cycle": {"from_step": 10, "roster": {"incumbent": "A", "candidate": "B", "reference": "R"}},
    }
    first, second = [cycle_pve.collection_case(tmp_path, "legacy", plan, i) for i in (10, 11)]
    assert first[0] == second[0] and {first[2], second[2]} == {"A", "B"}


def test_equal_tier_weight_and_map_pairing():
    m = cycle_pve.paired_interval({"boss": [3] * 5, "strong": [1] * 5, "mid": [-1] * 5, "rookie": [-1] * 5})
    assert m == {"estimated_40_game_gain": 20, "ci95": [20, 20]}


def test_rollback_needs_comparable_teams_and_sufficient_series():
    games = []
    for opponent in range(4):
        for idx in range(3):
            for version, t, outcome in (("old", 10, 1), ("new", 100, -1)):
                games.append(
                    {
                        "version_id": version,
                        "started_at": t + idx,
                        "ended_at": t + idx + 0.5,
                        "outcome": outcome,
                        "status": "finished",
                        "opponent_id": opponent,
                    }
                )
    assert cycle_release.comparable_series(games, "old", "new", 80)["rollback"]
    assert not cycle_release.comparable_series(games[:-1], "old", "new", 80)["rollback"]
    assert not cycle_release.comparable_series(games[6:], "old", "new", 80)["rollback"]


def test_three_failed_rounds_only_pause_training_on_same_line(tmp_path, monkeypatch):
    for i in range(3):
        cycle_store.finish(tmp_path, "pvp", result(round=i + 1), "rejected", "未提升", signature="one")
    monkeypatch.setattr(cycle, "_signature", lambda _: "one")
    monkeypatch.setattr(cycle, "_round", lambda *args: pytest.fail("unchanged search repeated"))
    cycle._experiment(tmp_path, "pvp", "old", 1, "pve")
    assert "等待新打法" in cycle_store.get(tmp_path, "pvp", "phase")
    assert not cycle_store.history(tmp_path, "pve")


def test_start_resume_and_independent_pause(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cycle, "_launch", lambda _, line: calls.append(line))
    cycle.dispatch(tmp_path, SimpleNamespace(op="start", line="all"))
    cycle.dispatch(tmp_path, SimpleNamespace(op="pause", line="pve"))
    cycle.resume_enabled(tmp_path)
    assert calls == ["pvp", "pve", "pvp"]
    assert cycle_store.enabled(tmp_path, "pvp")
    assert not cycle_store.enabled(tmp_path, "pve")
