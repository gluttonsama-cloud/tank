import shutil
import time
from pathlib import Path

import pytest

from tank import evaluation, versions
from tank.eval_plan import SCORE_ITEMS
from tank.eval_runner import MatchFailure
from tank.eval_store import database


SOURCE = Path(__file__).resolve().parents[1]


def workspace(folder):
    for name in ("resource", "suites"):
        (folder / name).mkdir(parents=True)
    shutil.copy(SOURCE / "resource" / "SHA256SUMS", folder / "resource")
    _, wheel = (SOURCE / "resource" / "SHA256SUMS").read_text(encoding="utf-8").split()
    shutil.copy(SOURCE / "resource" / wheel, folder / "resource")
    (folder / "suites" / "one.json").write_text(
        '{"version":1,"opponents":["baseline"],"seeds":[42],"repeats":1}', encoding="utf-8",
    )
    shutil.copy(SOURCE / "suites" / "quick.json", folder / "suites")
    shutil.copytree(SOURCE / "strategies" / "starter", folder / "strategies" / "starter")
    return folder, versions.create(folder, "starter")["id"]


def official_like(spec):
    by_side = [2 if spec[side] != "baseline" else 1 for side in ("red", "blue")]
    scores = [{name: points if name == "damage" else 0 for name in SCORE_ITEMS} | {"total": points}
              for points in by_side]
    winner = -1 if by_side[0] == by_side[1] else by_side.index(max(by_side))
    return {"scores": by_side, "winner": winner, "result": {"scores": scores, "winner": winner}}


def fake_match(root, spec, timeout, replay, log, cancelled):
    return official_like(spec), 0.01


def only_run(root):
    with database(root) as db:
        return db.execute("SELECT run_id FROM runs").fetchone()[0]


def test_interrupted_pair_excludes_delta_and_resumes_only_missing(tmp_path, monkeypatch):
    root, candidate = workspace(tmp_path / "interrupted")
    calls = []

    def interrupted(root, spec, timeout, replay, log, cancelled):
        calls.append(spec)
        if len(calls) == 4:
            raise KeyboardInterrupt()
        return fake_match(root, spec, timeout, replay, log, cancelled)

    monkeypatch.setattr(evaluation, "run_isolated", interrupted)
    with pytest.raises(KeyboardInterrupt):
        evaluation.start(root, [candidate], "baseline", "one", background=False)
    run_id = only_run(root)
    assert evaluation.status(root, run_id)["completed"] == 3
    comparison, = evaluation.compare(root, run_id)["rows"]
    assert comparison["completed_pairs"] == 0
    assert comparison["scored_games"] in (1, 2)
    assert comparison["total_score"] == 2 * comparison["scored_games"]
    assert comparison["total_delta"] is None
    assert "score_delta" not in comparison
    assert all("score_delta" not in row for row in evaluation.cases(root, run_id)["rows"])

    retried = []

    def finish(root, spec, timeout, replay, log, cancelled):
        retried.append(spec)
        return fake_match(root, spec, timeout, replay, log, cancelled)

    monkeypatch.setattr(evaluation, "run_isolated", finish)
    assert evaluation.resume(root, run_id, background=False)["status"] == "completed"
    assert retried == calls[3:]
    complete, = evaluation.compare(root, run_id)["rows"]
    assert complete["completed_pairs"] == 1
    assert complete["score_delta"] == 1
    assert (complete["total_score"], complete["scored_games"], complete["scheduled_games"], complete["total_delta"]) == (4, 2, 2, 2)
    assert complete["components"]["damage"] == 2

    fresh, _ = workspace(tmp_path / "uninterrupted")
    full_run = evaluation.start(fresh, [candidate], "baseline", "one", background=False)
    assert evaluation.compare(fresh, full_run["run_id"])["rows"] == [complete]
    reuse = evaluation.start(root, [candidate], "baseline", "one", background=False)
    assert reuse["completed"] == 4
    assert len(retried) == 5


def test_retry_targets_failures(tmp_path, monkeypatch):
    root, candidate = workspace(tmp_path / "retry")
    calls = []

    def fail_and_interrupt(root, spec, timeout, replay, log, cancelled):
        calls.append(spec)
        if len(calls) == 1:
            raise MatchFailure("模拟超时", timeout=True)
        raise KeyboardInterrupt()

    monkeypatch.setattr(evaluation, "run_isolated", fail_and_interrupt)
    with pytest.raises(KeyboardInterrupt):
        evaluation.start(root, [candidate], "baseline", "one", background=False)
    run_id = only_run(root)
    assert evaluation.status(root, run_id)["timed_out"] == 1

    retried = []

    def finish(root, spec, timeout, replay, log, cancelled):
        retried.append(spec)
        return fake_match(root, spec, timeout, replay, log, cancelled)

    monkeypatch.setattr(evaluation, "run_isolated", finish)
    result = evaluation.retry(root, run_id, background=False)
    assert result["failed"] == 0
    assert result["completed"] == 1
    assert retried == calls[:1]
    assert evaluation.resume(root, run_id, background=False)["status"] == "completed"
    assert retried[1] == calls[1]


def test_disconnected_shards_merge_idempotently(tmp_path, monkeypatch):
    root, candidate = workspace(tmp_path / "coordinator")
    worker, _ = workspace(tmp_path / "worker")
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)

    run = evaluation.start(root, [candidate], "baseline", "quick", shards=2, prepare_only=True)
    run_id = run["run_id"]
    plan = tmp_path / "shard.jsonl.gz"
    evaluation.export_run(root, run_id, plan, shard=1)
    assert evaluation.import_run(worker, plan)["completed"] == 0

    assert evaluation.resume(root, run_id, shard=0, background=False)["completed"] == 4
    assert evaluation.resume(worker, run_id, shard=1, background=False)["completed"] == 4
    evaluation.export_run(worker, run_id, plan, shard=1)
    merged = evaluation.import_run(root, plan)
    assert merged["status"] == "completed"
    assert merged["completed"] == 8
    assert evaluation.import_run(root, plan)["completed"] == 8
    assert evaluation.compare(root, run_id)["rows"][0]["completed_pairs"] == 2


def test_background_official_direct_pair_and_replay(tmp_path):
    root, candidate = workspace(tmp_path / "official")
    plan = evaluation.start(root, [candidate], "baseline", "one", mode="direct", replay=True)
    run_id = plan["run_id"]
    try:
        finish_by = time.monotonic() + 30
        while time.monotonic() < finish_by:
            state = evaluation.status(root, run_id)
            if state["status"] == "completed" and not state["active_shards"]:
                break
            if state["failed"]:
                pytest.fail(str(evaluation.logs(root, run_id)))
            time.sleep(0.15)
        else:
            pytest.fail(str(evaluation.logs(root, run_id)))
    finally:
        evaluation.cancel(root, run_id)

    red, blue = evaluation.cases(root, run_id)["rows"]
    assert (red["red"], red["blue"]) == (candidate, "baseline")
    assert (blue["red"], blue["blue"]) == ("baseline", candidate)
    assert red["has_replay"] and blue["has_replay"]
    repeated = evaluation.reproduce(root, blue["case_id"])
    assert repeated["matches_recorded_result"] and repeated["matches_recorded_replay"]
    assert Path(repeated["replay_path"]).is_file()


def test_shard_failure_sync_preserves_reason_and_yields_to_completed_case(tmp_path, monkeypatch):
    worker, candidate = workspace(tmp_path / "worker")
    coordinator, _ = workspace(tmp_path / "coordinator")
    calls = []

    def timeout_once(root, spec, timeout, replay, log, cancelled):
        calls.append(spec)
        if len(calls) == 1:
            raise MatchFailure("策略第 7 帧超时", timeout=True)
        return fake_match(root, spec, timeout, replay, log, cancelled)

    monkeypatch.setattr(evaluation, "run_isolated", timeout_once)
    run = evaluation.start(worker, [candidate], "baseline", "one", background=False)
    run_id = run["run_id"]
    bundle = tmp_path / "partial.jsonl.gz"
    evaluation.export_run(worker, run_id, bundle, shard=0)
    merged = evaluation.import_run(coordinator, bundle)
    assert (merged["completed"], merged["failed"], merged["timed_out"]) == (3, 1, 1)
    red = evaluation.cases(coordinator, run_id, side="red")["rows"][0]
    assert red["failures"] == [{"case_id": red["case_id"], "kind": "timeout", "message": "策略第 7 帧超时"}]

    assert evaluation.retry(coordinator, run_id, background=False)["status"] == "completed"
    stale = evaluation.import_run(coordinator, bundle)
    assert (stale["completed"], stale["failed"], stale["timed_out"], stale["imported_failures"]) == (4, 0, 0, 0)
