import json
import time
from pathlib import Path

import pytest

from tank import evaluation, screening, versions
from tank.eval_plan import SCORE_ITEMS
from tank.eval_runner import MatchFailure

from test_evaluation import workspace


def add_suite(root, name, seed):
    (root / "suites" / f"{name}.json").write_text(
        json.dumps({"version": 1, "opponents": ["baseline"], "seeds": [seed], "repeats": 1}),
        encoding="utf-8",
    )


def candidates_for(root):
    scenario = root / "strategies" / "starter"
    records = []
    for marker in ("low", "high", "medium"):
        (scenario / "marker.txt").write_text(marker, encoding="utf-8")
        records.append(versions.create(root, "starter")["id"])
    return records


def test_screening_advances_equal_budgets_and_only_survivors(tmp_path, monkeypatch):
    root, _ = workspace(tmp_path / "stages")
    low, high, medium = candidates_for(root)
    add_suite(root, "round2", 84)
    add_suite(root, "holdout", 123)
    score = {"baseline": 0, high: 3, medium: 2, low: 1}
    observed = []

    def match(root, spec, timeout, replay, log, cancelled):
        observed.append((spec["seed"], spec["red"], spec["blue"]))
        red, blue = score[spec["red"]], score[spec["blue"]]
        by_side = [red, blue]
        scores = [dict.fromkeys(SCORE_ITEMS, 0) | {"damage": points, "total": points} for points in by_side]
        winner = -1 if red == blue else by_side.index(max(by_side))
        return {"scores": by_side, "winner": winner, "result": {"scores": scores, "winner": winner}}, 0.1

    monkeypatch.setattr(evaluation, "run_isolated", match)
    stages = [("one", 2), ("round2", 1), ("holdout", 1)]
    result = screening.screen(root, [low, high, medium], "baseline", stages, background=False)
    assert result["status"] == "completed"
    assert result["selected"] == [high]
    assert [round["selected"] for round in result["rounds"]] == [[high, medium], [high], [high]]
    assert [[seed for seed, red, blue in observed if policy in (red, blue)] for policy in (low, high, medium)] == [
        [42, 42], [42, 42, 84, 84, 123, 123], [42, 42, 84, 84],
    ]
    fresh = screening.status(root, result["id"])
    assert fresh["current_round"]["suite"]["seeds"] == [123]
    with pytest.raises(ValueError, match="最后一轮须使用独立地图"):
        screening.screen(root, [low, high], "baseline", [("one", 1), ("one", 1)], background=False)


def test_incomplete_pair_waits_for_resume_before_selecting(tmp_path, monkeypatch):
    root, candidate = workspace(tmp_path / "resume")
    add_suite(root, "holdout", 123)
    calls = []

    def match(root, spec, timeout, replay, log, cancelled):
        from test_evaluation import fake_match

        calls.append(spec)
        if len(calls) == 4:
            raise MatchFailure("测试超时", timeout=True)
        return fake_match(root, spec, timeout, replay, log, cancelled)

    monkeypatch.setattr(evaluation, "run_isolated", match)
    result = screening.screen(root, [candidate], None, [("one", 1), ("holdout", 1)], background=False)
    assert result["status"] == "incomplete"
    assert len(result["rounds"]) == 1
    assert "selected" not in result["rounds"][0]
    assert result["current_round"]["timed_out"] == 1

    resumed = screening.resume(root, result["id"], background=False)
    assert resumed["status"] == "completed"
    assert resumed["rounds"][0]["run_id"] == result["rounds"][0]["run_id"]
    assert calls[4] == calls[3]
    assert len(calls) == 9


def test_screening_background_returns_id_and_finishes_sdk_rounds(tmp_path):
    root, candidate = workspace(tmp_path / "background")
    add_suite(root, "holdout", 123)
    started = screening.screen(root, [candidate], "baseline", [("one", 1), ("holdout", 1)])
    assert started["status"] == "queued"
    assert started["rounds"] == []
    screen_id = started["id"]
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        result = screening.status(root, screen_id)
        if result["status"] not in ("queued", "running"):
            break
        time.sleep(0.2)
    else:
        pytest.fail(f"筛选耗时过长，状态 {result}")
    assert result["status"] == "completed", Path(result["log_path"]).read_text(encoding="utf-8")
    assert result["selected"] == [candidate]
    assert result["current_round"]["suite"]["seeds"] == [123]
    assert len(result["rounds"]) == 2
