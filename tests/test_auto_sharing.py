import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tank import cli, evaluation, publication, sync
from tank.eval_plan import match_specs
from tank.eval_store import cached_matches, database, read_manifest, save_match

from test_evaluation import fake_match, workspace
from test_sharing import git


def repository(tmp_path, *, hook=False):
    root, candidate = workspace(tmp_path / "work")
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    git(root, "init", "-b", "main")
    git(root, "config", "user.email", "tests@localhost")
    git(root, "config", "user.name", "Auto sharing tests")
    (root / ".gitignore").write_text("/.local/\n/resource/*.whl\n", encoding="utf-8")
    git(root, "add", ".gitignore", "strategies", "suites", "resource/SHA256SUMS")
    if hook:
        _, wheel = (root / "resource/SHA256SUMS").read_text(encoding="utf-8").split()
        script = root / ".githooks/pre-commit"
        script.parent.mkdir()
        script.write_bytes(f"#!/bin/sh\ntest -s 'resource/{wheel}'\n".encode())
        git(root, "add", "--chmod=+x", ".githooks/pre-commit")
    git(root, "commit", "-m", "fixture")
    if hook:
        git(root, "config", "core.hooksPath", ".githooks")
    git(root, "remote", "add", "origin", str(remote))
    git(root, "push", "origin", "main")
    return root, candidate, remote


def wait_share(root, run_id, desired="published"):
    until = time.monotonic() + 40
    while time.monotonic() < until:
        run = evaluation.status(root, run_id)
        sharing = run["sharing"]
        if sharing["state"] == desired:
            return sharing
        if sharing["state"] == "failed":
            pytest.fail(str(sharing["error"]))
        if run["failed"]:
            pytest.fail(str(evaluation.logs(root, run_id)))
        time.sleep(0.15)
    log = root / ".local/eval" / run_id / "publication/worker.log"
    pytest.fail(log.read_text(encoding="utf-8") if log.is_file() else str(run))


def branches(remote):
    refs = subprocess.check_output(
        ["git", "-C", str(remote), "for-each-ref", "--format=%(refname:short)", "refs/heads/experiments"],
        text=True,
    )
    return refs.splitlines()


def test_background_and_cached_batches_share_once_while_drafts_stay_local(tmp_path):
    root, candidate, remote = repository(tmp_path, hook=True)
    start = cli.parser().parse_args([
        "eval", "start", "--candidates", candidate, "--mode", "direct", "--suite", "quick",
        "--shards", "2", "--share",
    ])
    started = cli.execute(root, start)
    run_id = started["run_id"]
    first = wait_share(root, run_id)
    assert first["attempts"] == 1
    assert branches(remote) == [first["branch"]]

    draft = evaluation.start(root, [candidate], "baseline", "quick", mode="direct", prepare_only=True)
    assert draft["status"] == "completed"
    assert draft["sharing"] is None
    assert branches(remote) == [first["branch"]]

    cached = evaluation.start(root, [candidate], "baseline", "quick", mode="direct", prepare_only=True, share=True)
    assert cached["status"] == "completed"
    second = wait_share(root, cached["run_id"])
    assert second["attempts"] == 1
    assert set(branches(remote)) == {first["branch"], second["branch"]}
    assert set(sync.list_remote(root)) == {run_id, cached["run_id"]}
    assert evaluation.share_retry(root, run_id)["sharing"]["attempts"] == 1


def test_share_intent_stays_on_coordinator_and_final_offline_import_publishes(tmp_path, monkeypatch):
    root, candidate, remote = repository(tmp_path)
    worker, _ = workspace(tmp_path / "worker")
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    started = evaluation.start(root, [candidate], "baseline", "quick", shards=2, prepare_only=True, share=True)
    run_id = started["run_id"]
    assert started["sharing"]["state"] == "waiting"
    bundle = tmp_path / "shard.jsonl.gz"
    evaluation.export_run(root, run_id, bundle, shard=1)
    assert evaluation.import_run(worker, bundle)["sharing"] is None

    evaluation.resume(root, run_id, shard=0, background=False)
    assert evaluation.status(root, run_id)["sharing"]["state"] == "waiting"
    with pytest.raises(ValueError, match="未完成"):
        evaluation.share_retry(root, run_id)
    evaluation.resume(worker, run_id, shard=1, background=False)
    evaluation.export_run(worker, run_id, bundle, shard=1)
    assert branches(remote) == []
    merged = evaluation.import_run(root, bundle)
    assert merged["status"] == "completed"
    evaluation.import_run(root, bundle)
    published = wait_share(root, run_id)
    assert published["attempts"] == 1
    assert branches(remote) == [published["branch"]]
    assert evaluation.status(worker, run_id)["sharing"] is None


def test_failed_delivery_remembers_the_artifact_and_retries_same_branch(tmp_path, monkeypatch):
    root, candidate, remote = repository(tmp_path)
    missing = tmp_path / "unreachable.git"
    git(root, "remote", "set-url", "origin", str(missing))
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    started = evaluation.start(root, [candidate], "baseline", "one", background=False, share=True)
    run_id = started["run_id"]
    failed = wait_share(root, run_id, desired="failed")
    assert failed["attempts"] == 1
    assert failed["error"]
    assert failed["branch"]
    assert branches(remote) == []
    with database(root) as db:
        manifest, _ = read_manifest(db, run_id)
        specs = match_specs(manifest)
        case_id = next(iter(specs))
        result = cached_matches(db, {case_id: specs[case_id]})[case_id]["result"]
        save_match(db, case_id, specs[case_id], result | {"replay": {"frames": [[1]]}}, 1)
        state = db.execute("SELECT prepared FROM auto_publications WHERE run_id = ?", (run_id,)).fetchone()
        assert sync.branch_for(json.loads(state["prepared"])) == failed["branch"]
    git(root, "remote", "set-url", "origin", str(remote))
    retry = cli.parser().parse_args(["eval", "share-retry", run_id, "--json"])
    assert cli.execute(root, retry)["sharing"]["attempts"] == 2
    published = wait_share(root, run_id)
    assert published["branch"] == failed["branch"]
    assert branches(remote) == [failed["branch"]]


@pytest.mark.parametrize("source", ["local", "import"])
def test_another_batch_fills_shared_run_without_status_polling(tmp_path, monkeypatch, source):
    root, candidate, remote = repository(tmp_path)
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    prepared = evaluation.start(root, [candidate], "baseline", "one", prepare_only=True, share=True)
    share_id = prepared["run_id"]
    assert prepared["sharing"]["state"] == "waiting"
    if source == "local":
        assert evaluation.start(root, [candidate], "baseline", "one", background=False)["sharing"] is None
    else:
        worker, candidate = workspace(tmp_path / "drive")
        plain = evaluation.start(worker, [candidate], "baseline", "one", background=False)
        archive = tmp_path / "plain.jsonl.gz"
        evaluation.export_run(worker, plain["run_id"], archive)
        assert evaluation.import_run(root, archive)["sharing"] is None

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with database(root) as db:
            state = db.execute("SELECT state, attempts, branch, error FROM auto_publications WHERE run_id = ?",
                               (share_id,)).fetchone()
        if state["state"] in {"published", "failed"}:
            break
        time.sleep(0.1)
    assert state["state"] == "published", dict(state)
    assert state["attempts"] == 1
    assert branches(remote) == [state["branch"]]


def test_status_recovers_committed_results_after_interruption_once(tmp_path, monkeypatch):
    root, candidate, remote = repository(tmp_path)
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)

    def interrupted(*args):
        raise KeyboardInterrupt

    prepared = evaluation.start(root, [candidate], "baseline", "one", prepare_only=True, share=True)
    run_id = prepared["run_id"]
    with monkeypatch.context() as patch:
        patch.setattr(evaluation, "_publish_after_results", interrupted)
        with pytest.raises(KeyboardInterrupt):
            evaluation.resume(root, run_id, background=False)
    with database(root) as db:
        assert db.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 4
        assert db.execute("SELECT state FROM auto_publications WHERE run_id = ?", (run_id,)).fetchone()[0] == "waiting"
    assert branches(remote) == []
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: evaluation.status(root, run_id), range(8)))
    assert all(result["status"] == "completed" and result["sharing"]["attempts"] == 1 for result in results)
    published = wait_share(root, run_id)
    assert branches(remote) == [published["branch"]]


def test_remote_timeout_records_share_failure_without_rerunning_matches(tmp_path, monkeypatch):
    root, candidate, remote = repository(tmp_path)
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    monkeypatch.setattr(publication, "_spawn", publication._work)
    real_popen = subprocess.Popen

    def hanging_remote(command, **kwargs):
        if "ls-remote" in command:
            command = [sys.executable, "-c", "import time; time.sleep(60)"]
        return real_popen(command, **kwargs)

    with monkeypatch.context() as patch:
        patch.setitem(sync.REMOTE_TIMEOUTS, "ls-remote", 0.2)
        patch.setattr(sync.subprocess, "Popen", hanging_remote)
        result = evaluation.start(root, [candidate], "baseline", "one", background=False, share=True)
    assert result["status"] == "completed"
    assert result["sharing"]["state"] == "failed"
    assert result["sharing"]["error"] == "Git ls-remote 超过 0.2 秒"
    assert branches(remote) == []
    recovered = evaluation.share_retry(root, result["run_id"])
    assert recovered["sharing"]["attempts"] == 2
    assert recovered["sharing"]["state"] == "published"
    assert recovered["sharing"]["branch"] == result["sharing"]["branch"]


def test_reconcile_publishes_only_registered_complete_batches(tmp_path, monkeypatch):
    root, candidate, remote = repository(tmp_path)
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    share = evaluation.start(root, [candidate], "baseline", "one", prepare_only=True, share=True)
    draft = evaluation.start(root, [candidate], "baseline", "one", prepare_only=True)
    assert publication.reconcile(root) == []
    with database(root) as db:
        manifest, _ = read_manifest(db, draft["run_id"])
        for case_id, spec in match_specs(manifest).items():
            result, elapsed = fake_match(root, spec, 120, False, None, None)
            save_match(db, case_id, spec, result, elapsed)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: publication.reconcile(root), range(4)))

    with database(root) as db:
        state = db.execute("SELECT attempts FROM auto_publications WHERE run_id = ?", (share["run_id"],)).fetchone()
        assert state["attempts"] == 1
    published = wait_share(root, share["run_id"])
    assert published["attempts"] == 1
    assert branches(remote) == [published["branch"]]
    assert publication.reconcile(root) == []
    assert evaluation.status(root, draft["run_id"])["sharing"] is None


@pytest.mark.parametrize("error,auto", [
    ("Git ls-remote 超过 0.2 秒", True),
    ("fatal: unable to access origin: Connection reset by peer", True),
    ("git@host: Permission denied (publickey).", False),
    ("fatal: '/unreachable.git' does not appear to be a git repository", False),
])
def test_reconcile_retries_only_temporary_git_failures(tmp_path, monkeypatch, error, auto):
    root, candidate, remote = repository(tmp_path)
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    monkeypatch.setattr(publication, "_spawn", publication._work)
    real_git = sync._git

    def fail_remote(folder, *args):
        if args[0] == "ls-remote":
            raise RuntimeError(error)
        return real_git(folder, *args)

    with monkeypatch.context() as patch:
        patch.setattr(sync, "_git", fail_remote)
        initial = evaluation.start(root, [candidate], "baseline", "one", background=False, share=True)
    run_id = initial["run_id"]
    failed = initial["sharing"]
    assert failed["state"] == "failed" and failed["attempts"] == 1
    assert publication.reconcile(root) == []
    with database(root) as db:
        db.execute("UPDATE auto_publications SET updated_at = ? WHERE run_id = ?",
                   (time.time() - publication.RETRY_DELAYS[0] - 1, run_id))

    reconciled = publication.reconcile(root)

    if auto:
        assert reconciled[0]["sharing"]["state"] == "published"
        assert reconciled[0]["sharing"]["attempts"] == 2
        assert branches(remote) == [failed["branch"]]
    else:
        assert reconciled == []
        assert evaluation.status(root, run_id)["sharing"]["attempts"] == 1
        assert branches(remote) == []
    assert publication.reconcile(root) == []


def test_network_retry_backoff_stops_after_three_retries():
    row = {"state": "failed", "attempts": 1, "updated_at": 0, "error": "Git push 超过 300 秒"}
    for index, delay in enumerate(publication.RETRY_DELAYS, 1):
        row["attempts"] = index
        assert not publication._retry_due(row, delay - 1)
        assert publication._retry_due(row, delay)
    row["attempts"] += 1
    assert not publication._retry_due(row, 50_000)


def test_reconcile_lost_push_response_reuses_remote_ref(tmp_path, monkeypatch):
    root, candidate, remote = repository(tmp_path)
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    monkeypatch.setattr(publication, "_spawn", publication._work)
    real_git = sync._git
    pushes = []

    def lose_response(folder, *args):
        result = real_git(folder, *args)
        if args[0] == "push":
            pushes.append(args)
            raise RuntimeError("send-pack: Connection reset by peer")
        return result

    monkeypatch.setattr(sync, "_git", lose_response)
    initial = evaluation.start(root, [candidate], "baseline", "one", background=False, share=True)
    failed = initial["sharing"]
    assert failed["state"] == "failed"
    assert branches(remote) == [failed["branch"]]
    with database(root) as db:
        db.execute("UPDATE auto_publications SET updated_at = ? WHERE run_id = ?",
                   (time.time() - publication.RETRY_DELAYS[0] - 1, initial["run_id"]))

    assert publication.reconcile(root)[0]["sharing"]["state"] == "published"
    assert branches(remote) == [failed["branch"]]
    assert len(pushes) == 1
