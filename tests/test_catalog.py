import hashlib
import shutil

import pytest

from tank import catalog, evaluation, versions
from tank.eval_plan import SCORE_ITEMS, make_manifest, match_specs
from tank.eval_store import add_manifest, database, read_manifest, save_match
from tank.workspace import write_json

from test_evaluation import workspace
from test_sharing import git


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


def add_scores(root, number, scores_by_version, suite, *, timeout=120, seeds=None, replay=False):
    run_id = f"run-{number:032x}"
    manifest = make_manifest(root, run_id, list(scores_by_version), "baseline", suite, 1, "opponent", timeout, replay)
    with database(root) as db:
        add_manifest(db, manifest)
        db.execute("UPDATE runs SET attempted = 1, created_at = ? WHERE run_id = ?", (number, run_id))
        for case_id, spec in match_specs(manifest).items():
            if seeds is not None and spec["seed"] not in seeds:
                continue
            scores = [scores_by_version.get(spec[side], 1) for side in ("red", "blue")]
            winner = -1 if scores[0] == scores[1] else scores.index(max(scores))
            detail = [dict.fromkeys(SCORE_ITEMS, 0) | {"damage": score, "total": score} for score in scores]
            result = {"scores": scores, "winner": winner, "result": {"scores": detail, "winner": winner}}
            if replay:
                result["replay"] = {"frames": []}
            save_match(db, case_id, spec, result, 0.1)
    return run_id


def add_run(root, number, candidate, suite, *, points=6.34, **options):
    return add_scores(root, number, {candidate: points}, suite, **options)


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
    project = inventory(root)["rules"]
    record = project["local"]
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
    history = {snapshot["id"]: snapshot["local"] for snapshot in project["versions"]}
    assert history[first]["latest"]["run_id"] == partial
    assert [item["run_id"] for item in history[first]["by_condition"]] == [partial, different_timeout, first_run]
    assert [item["run_id"] for item in history[second]["by_condition"]] == [second_run]
    assert history[first]["by_condition"][-1]["condition_id"] == history[second]["by_condition"][0]["condition_id"]

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


def test_version_condition_prefers_completion_and_keeps_latest_partial_progress(tmp_path):
    root, version = workspace(tmp_path / "workspace")
    declare(root, "rules", sources=["starter"])
    complete = add_run(root, 1, version, "quick")
    partial = add_run(root, 2, version, "quick", replay=True, seeds={42}, timeout=120.0)
    write_json(root / "suites/longer.json", {
        "version": 1, "opponents": ["baseline"], "seeds": [99, 100, 101, 102], "repeats": 1,
    })
    fuller = add_run(root, 3, version, "longer", seeds={99, 100, 101})
    newest = add_run(root, 4, version, "longer", replay=True, seeds={99, 100})

    history = inventory(root)["rules"]["versions"][0]["local"]
    assert history["latest"]["run_id"] == newest
    assert history["latest"]["status"] == "incomplete"
    assert history["latest"]["scored_games"] == 4
    assert len(history["by_condition"]) == 2
    by_suite = {record["suite"]: record for record in history["by_condition"]}
    assert by_suite["quick"]["run_id"] == complete
    assert by_suite["quick"]["status"] == "completed"
    assert by_suite["longer"]["run_id"] == fuller
    assert by_suite["longer"]["scored_games"] == 6
    with database(root) as db:
        assert db.execute("SELECT 1 FROM catalog_comparisons WHERE run_id = ?", (partial,)).fetchone() is None


def test_project_model_versions_keep_attribution_across_shared_batch(tmp_path, monkeypatch):
    root, _ = workspace(tmp_path / "workspace")
    source = root / "strategies" / "starter"
    ids = []
    for number, model in enumerate(("model-a", "model-b"), 1):
        script = f"{model}/fit.py"
        declare(root, model, kind="imitation", training=[script])
        (source / "revision.txt").write_text(model, encoding="utf-8")
        key = versions.create_from_directory(root, source, "训练 fit")["id"]
        ids.append(key)
        job_id = f"train-{number:032x}"
        job_path = root / ".local/training/jobs" / job_id
        info = {"job_id": job_id, "created_at": number}
        write_json(job_path / "queued.json", info | {"status": "queued"})
        write_json(job_path / "finished.json", info | {"status": "completed", "snapshot_id": key})
        write_json(job_path / "request.json", {"script": script, "config_id": "fixture"})
    first, second = ids
    run_id = add_scores(root, 1, {first: 7, second: 9}, "one")
    score_a = submission(root, first, 1, score=150.04)
    score_b = submission(root, second, 2, score=222.13)
    submission(root, first, 3, score=None, status="running")
    unknown = submission(root, first, 4, score=999) | {"version_id": None}
    write_json(root / ".local/online/team-scores.json", {
        "team_id": 8, "refreshed_at": 20, "best_id": 4, "submissions": [unknown, score_b, score_a],
    })
    calls = []
    real_compare = evaluation.compare

    def compare(root, selected):
        calls.append(selected)
        return real_compare(root, selected)

    monkeypatch.setattr(evaluation, "compare", compare)
    data = inventory(root)
    model_a, = data["model-a"]["versions"]
    model_b, = data["model-b"]["versions"]
    assert (model_a["id"], model_b["id"]) == (first, second)
    result_a, = model_a["local"]["by_condition"]
    result_b, = model_b["local"]["by_condition"]
    assert result_a["run_id"] == result_b["run_id"] == run_id
    assert result_a["condition_id"] == result_b["condition_id"]
    assert (result_a["total_score"], result_b["total_score"]) == (14, 18)
    assert result_a["opponents"] == ["baseline"]
    assert result_a["rules_version"]
    assert model_a["official"]["best"]["score"] == 150.04
    assert model_a["official"]["latest"]["status"] == "running"
    assert model_b["official"]["best"]["score"] == 222.13
    assert model_b["official"]["latest"]["id"] == 2
    assert model_b["official"]["best"]["games"] == 40
    assert calls == [run_id]
    assert inventory(root) == data
    assert calls == [run_id]


def test_experiment_metadata_connects_new_checkout_to_original_project(tmp_path, monkeypatch):
    source, _ = workspace(tmp_path / "source")
    project_id = "remote-model"
    write_json(source / "projects/remote-model.json", {
        "name": "对手预测模型", "kind": "imitation", "sources": [], "training": ["models/fit.py"],
    })
    strategy = source / "strategies/starter"
    (strategy / "revision.txt").write_text("trained policy", encoding="utf-8")
    version = versions.create_from_directory(source, strategy, "训练 fit")["id"]
    run_id = add_run(source, 1, version, "one", points=8.25)

    with monkeypatch.context() as patch:
        def unrelated(*args, **kwargs):
            pytest.fail("project label export must not enumerate unrelated snapshots or compare results")

        patch.setattr(evaluation, "compare", unrelated)
        patch.setattr(versions, "list_versions", unrelated)
        labels = catalog.project_labels(source, [version, "baseline"])
    assert labels == {version: [{"id": project_id, "name": "对手预测模型", "kind": "imitation", "number": 1}]}

    archive = tmp_path / "experiment-files"
    run_folder = archive / "published/runs" / run_id
    run_folder.mkdir(parents=True)
    files = archive / "published/versions" / version
    shutil.copytree(versions.snapshot_data(source, version).parent, files)
    (files / ".gitattributes").write_bytes(b"files/** -text\n")
    with database(source) as db:
        manifest, _ = read_manifest(db, run_id)
    write_json(run_folder / "manifest.json", manifest)
    temp_fragment = run_folder / "results.jsonl.gz"
    evaluation.export_run(source, run_id, temp_fragment)
    key = hashlib.sha256(temp_fragment.read_bytes()).hexdigest()
    fragment = temp_fragment.rename(run_folder / f"{key}.jsonl.gz")
    write_json(run_folder / f"{key}.projects.json", labels)
    git(archive, "init", "-b", "main")
    git(archive, "config", "user.name", "Catalog test")
    git(archive, "config", "user.email", "test@localhost")
    git(archive, "add", "published")
    git(archive, "commit", "-m", "shared experiment")
    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", str(archive), str(fresh))
    assert not (fresh / "projects").exists()
    assert not (fresh / "training").exists()
    versions.verify(fresh, version)
    evaluation.import_run(fresh, fresh / fragment.relative_to(archive), expected_manifest=manifest)

    project, = catalog.projects(fresh)
    assert (project["id"], project["name"], project["kind"]) == (project_id, "对手预测模型", "imitation")
    assert project["sources"] == project["training_scripts"] == []
    snapshot, = project["versions"]
    assert snapshot["id"] == version
    result, = snapshot["local"]["by_condition"]
    assert (result["run_id"], result["total_score"], result["scored_games"]) == (run_id, 16.5, 2)
    assert catalog.project_labels(fresh, [version]) == labels

    write_json(fresh / "projects/remote-model.json", {
        "name": "本机项目名称", "kind": "hybrid", "sources": [], "training": [],
    })
    project, = catalog.projects(fresh)
    assert (project["name"], project["kind"]) == ("本机项目名称", "hybrid")
    assert project["versions"][0]["id"] == version
    assert project["versions"][0]["local"]["by_condition"][0]["run_id"] == run_id


def test_project_label_validator_allows_only_batch_snapshots_and_display_metadata():
    version = "v-" + "a" * 64
    unrelated = "v-" + "b" * 64
    valid = {"id": "train:model/subdir", "name": "训练模型", "kind": "rl"}
    assert catalog.validate_project_labels({version: [valid]}, [version, "baseline"]) == {version: [valid]}
    with pytest.raises(ValueError, match="批次外快照"):
        catalog.validate_project_labels({unrelated: [valid]}, [version])
    with pytest.raises(ValueError, match="需要 id、name、kind"):
        catalog.validate_project_labels({version: [valid | {"training_scripts": ["remote.py"]}]}, [version])
    with pytest.raises(ValueError, match="项目 ID 重复"):
        catalog.validate_project_labels({version: [valid, valid | {"name": "同一 ID"}]}, [version])
