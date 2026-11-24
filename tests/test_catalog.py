from pathlib import Path

from tank import catalog, evaluation, versions
from tank.eval_plan import SCORE_ITEMS, make_manifest, match_specs
from tank.eval_store import add_manifest, database, save_match
from tank.workspace import write_json

from test_evaluation import workspace


def declare(root, project_id, *, kind="rule", sources=(), training=()):
    write_json(root / "projects" / f"{project_id}.json", {
        "name": project_id, "kind": kind, "sources": list(sources), "training": list(training),
    })


def inventory(root):
    return {project["id"]: project for project in catalog.projects(root)}


def test_projects_discover_sources_legacy_snapshots_and_training_jobs(tmp_path):
    root, starter = workspace(tmp_path / "workspace")
    declare(root, "rules", sources=["starter"])
    declare(root, "planned", kind="rl")
    declare(root, "learned", kind="imitation", training=["fit_demo.py"])
    source = root / "strategies" / "starter"
    (source / "revision.txt").write_text("legacy", encoding="utf-8")
    legacy = versions.create_from_directory(root, source, "已归档策略")["id"]
    (source / "revision.txt").write_text("trained", encoding="utf-8")
    learned = versions.create_from_directory(root, source, "训练 fit_demo")["id"]
    (source / "revision.txt").write_text("historical", encoding="utf-8")
    historical = versions.create_from_directory(root, source, "训练 fit_demo")["id"]
    write_json(source.with_name("draft") / "submission.json", {"entrypoint": "bot.py:Draft"})
    for number, script, status in [(1, "fit_demo.py", "completed"), (2, "fit_demo.py", "queued"),
                                    (3, "extras/fit.py", "running")]:
        job_id = f"train-{number:032x}"
        path = root / ".local/training/jobs" / job_id
        state = {"job_id": job_id, "created_at": number, "status": "queued"}
        write_json(path / "queued.json", state)
        write_json(path / "request.json", {"script": script, "config_id": "test-config"})
        if status == "completed":
            write_json(path / "finished.json", state | {"status": status, "snapshot_id": learned})
        if status == "running":
            write_json(path / "running.json", state | {"status": status})

    data = inventory(root)
    assert {v["id"] for v in data["rules"]["versions"]} == {starter}
    assert {v["id"] for v in data["learned"]["versions"]} == {learned, historical}
    assert all(v["project_id"] == "learned" for v in data["learned"]["versions"])
    assert data["learned"]["training"]["latest"]["status"] == "queued"
    assert data["learned"]["training"]["active"] == 1
    assert data["training:extras/fit.py"]["training"]["latest"]["status"] == "running"
    assert data["source:已归档策略"]["versions"][0]["id"] == legacy
    assert data["source:draft"]["sources"] == ["draft"]
    assert data["planned"]["local"] == {"latest": None, "best_by_condition": []}
    assert data["planned"]["official"] == {"best": None, "latest": None}
    assert data["planned"]["training"] == {"latest": None, "active": 0}
    assert not any(v["id"] == "baseline" for project in data.values() for v in project["versions"])
    assert not (root / ".local/eval").exists()


def add_run(root, number, candidate, suite, *, timeout=120, seeds=None, points=6.34):
    run_id = f"run-{number:032x}"
    manifest = make_manifest(root, run_id, [candidate], "baseline", suite, 1, "opponent", timeout, False)
    with database(root) as db:
        add_manifest(db, manifest)
        db.execute("UPDATE runs SET attempted = 1, created_at = ? WHERE run_id = ?", (number, run_id))
        for case_id, spec in match_specs(manifest).items():
            if seeds is not None and spec["seed"] not in seeds:
                continue
            scores = [points if spec[side] == candidate else 1 for side in ("red", "blue")]
            winner = -1 if scores[0] == scores[1] else scores.index(max(scores))
            detail = [dict.fromkeys(SCORE_ITEMS, 0) | {"damage": score, "total": score} for score in scores]
            save_match(db, case_id, spec, {"scores": scores, "winner": winner,
                                          "result": {"scores": detail, "winner": winner}}, 0.1)
    return run_id


def test_local_totals_only_rank_under_equal_conditions_and_cache_complete_runs(tmp_path, monkeypatch):
    root, first = workspace(tmp_path / "workspace")
    declare(root, "rules", sources=["starter"])
    source = root / "strategies/starter"
    (source / "revision.txt").write_text("new", encoding="utf-8")
    second = versions.create(root, "starter")["id"]
    first_run = add_run(root, 1, first, "one", points=6.34)
    second_run = add_run(root, 2, second, "one", points=8.62)
    different_timeout = add_run(root, 3, first, "one", timeout=30, points=6.34)
    partial = add_run(root, 4, first, "quick", seeds={42}, points=6.34)
    compared = []
    real_compare = evaluation.compare

    def compare(root, run, **filters):
        compared.append(run)
        return real_compare(root, run, **filters)

    monkeypatch.setattr(evaluation, "compare", compare)
    record = inventory(root)["rules"]["local"]
    best = record["best_by_condition"]
    assert [result["run_id"] for result in best] == [different_timeout, second_run]
    assert len({result["condition_id"] for result in best}) == 2
    assert best[1]["total_score"] == 17.24
    assert record["latest"]["version_id"] == first
    assert record["latest"]["run_id"] == partial
    assert record["latest"]["total_score"] == 12.68
    assert record["latest"]["scored_games"] == 2
    assert record["latest"]["scheduled_games"] == 4
    assert record["latest"]["opponents"] == ["baseline"]
    assert record["latest"]["status"] == "incomplete"
    assert set(compared) == {first_run, second_run, different_timeout, partial}

    compared.clear()
    assert inventory(root)["rules"]["local"] == record
    assert compared == [partial]
    add_run(root, 4, first, "quick", points=6.34)
    compared.clear()
    updated = inventory(root)["rules"]["local"]
    assert compared == [partial]
    assert updated["latest"]["total_score"] == 25.36
    assert updated["latest"]["status"] == "completed"
    compared.clear()
    inventory(root)
    assert compared == []
    monkeypatch.setattr(catalog, "_comparison_revision", lambda: "different-comparison")
    inventory(root)
    assert set(compared) == {first_run, second_run, different_timeout, partial}


def submission(root, version_id, number, *, score, status="completed", team_id=8):
    row = {
        "id": number, "team_id": team_id, "created_at": number + 0.5, "version": f"entry-{number}",
        "version_id": version_id, "sha256": f"{number:064x}",
    }
    parts = dict.fromkeys(SCORE_ITEMS, 0) | {"damage": score} if score is not None else None
    write_json(root / ".local/online/submissions" / f"{team_id}-{number}.json", row | {
        "evaluation": {"status": status, "score": score, "breakdown": parts},
    })
    return {key: value for key, value in row.items() if key != "team_id"} | {
        "score": score, "breakdown": parts, "games": 40,
    }


def test_official_uses_refreshed_valid_40_game_totals_and_preserves_latest_status(tmp_path):
    root, version = workspace(tmp_path / "workspace")
    declare(root, "rules", sources=["starter"])
    older = submission(root, version, 1, score=733.25)
    newest = submission(root, version, 2, score=547.67)
    submission(root, version, 3, score=2000)
    unbound = submission(root, version, 4, score=1876) | {"version_id": None}
    before_refresh = inventory(root)["rules"]["official"]
    assert before_refresh["best"]["id"] == 3
    submission(root, version, 5, score=None, status="queued")
    submission(root, version, 6, score=2999, team_id=87)
    write_json(root / ".local/online/team-scores.json", {
        "team_id": 8, "refreshed_at": 20, "best_id": 4, "submissions": [unbound, newest, older],
    })

    official = inventory(root)["rules"]["official"]
    assert official["best"]["id"] == older["id"]
    assert official["best"]["score"] == 733.25
    assert official["best"]["games"] == 40
    assert official["best"]["breakdown"]["damage"] == 733.25
    assert official["latest"]["id"] == 5
    assert official["latest"]["status"] == "queued"
    assert official["latest"]["score"] is None
