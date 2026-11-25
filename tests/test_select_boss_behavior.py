from hashlib import sha256
from pathlib import Path

import pytest
from seedcup.game import rules
from seedcup.game.replay import FORMAT_VERSION, save

from tank.workspace import write_json
from training import select_boss_behavior as selection
from training.boss_behavior_matches import common_alive_distances


def add_replay(root, replay_id, seed, grid, **fields):
    path = root / f".local/online/replays/{replay_id}.json.gz"
    replay = {"format": FORMAT_VERSION, "rules": rules.as_dict(),
              "match": {"id": replay_id}, "map": {"seed": seed, "grid": grid}}
    save(path, replay | fields)
    return {"id": replay_id, "seed": seed, "sha256": sha256(path.read_bytes()).hexdigest()}


def add_practice(root, replay_id, candidate, seed, side):
    write_json(root / f".local/online/practice/{candidate}/boss/{replay_id}.json",
               {"id": replay_id, "seed": seed, "side": side, "version_id": candidate})
    add_replay(root, replay_id, seed, f"grid-{seed}")


def test_common_alive_distances_use_same_window_for_all_models():
    def frame(tick, distance, blue_alive=1):
        return [tick, [0, 0, 0, 100, 1], [distance, 0, 0, 100, blue_alive], [], []]

    official = {"match": {"id": 5}, "frames": [frame(0, 10), frame(1, 4), frame(2, 30)]}
    early_death = {"frames": [frame(0, 10), frame(1, 8), frame(2, 25, 0)]}
    survives = {"frames": [frame(0, 10), frame(1, 6), frame(2, 40)]}
    assert common_alive_distances(official, [early_death, survives]) == [
        {"frames": 2, "official_median_distance_cells": 7, "local_median_distance_cells": 9},
        {"frames": 2, "official_median_distance_cells": 7, "local_median_distance_cells": 8},
    ]


def test_eval_map_check_deduplicates_source_reads_and_can_exclude_holdout(tmp_path, monkeypatch):
    add_practice(tmp_path, 101, "probe", 11, "red")
    add_practice(tmp_path, 102, "probe", 11, "blue")
    train = add_replay(tmp_path, 1, 1, "training-grid")
    holdout = add_replay(tmp_path, 2, 2, "grid-11")
    monkeypatch.setattr(selection.versions, "snapshot_data", lambda root, model: root / model)
    for model in ("reference", "trial"):
        write_json(tmp_path / model / "training.json", {
            "train": [train], "holdout": [holdout],
        })
    read_bytes = Path.read_bytes
    reads = []

    def tracked_read(path):
        reads.append(path.name)
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read)
    groups = selection.practice_groups(tmp_path, [101, 102])
    selection.check_eval_maps(tmp_path, ["reference", "trial"], groups, False)
    assert reads == ["101.json.gz", "102.json.gz", "1.json.gz"]
    with pytest.raises(ValueError, match="holdout.*地图网格重复"):
        selection.check_eval_maps(tmp_path, ["reference", "trial"], groups, True)
    write_json(tmp_path / "trial/training.json", {
        "train": [{"id": 102, "seed": 11}], "holdout": [],
    })
    with pytest.raises(ValueError, match="trial.*train.*种子或 ID 重复"):
        selection.check_eval_maps(tmp_path, ["reference", "trial"], groups, False)


@pytest.mark.parametrize(("field", "message"), [("format", "格式"), ("rules", "规则")])
@pytest.mark.parametrize("split", ["evaluation", "train"])
def test_eval_map_check_rejects_other_replay_rules_and_formats(tmp_path, monkeypatch, split, field, message):
    add_practice(tmp_path, 101, "probe", 11, "red")
    add_practice(tmp_path, 102, "probe", 11, "blue")
    groups = selection.practice_groups(tmp_path, [101, 102])
    models = ["baseline"]
    replay_id = 101
    if split == "evaluation":
        add_replay(tmp_path, replay_id, 11, "grid-11", **{field: None})
    else:
        replay_id = 1
        source = add_replay(tmp_path, replay_id, 1, "training-grid", **{field: None})
        monkeypatch.setattr(selection.versions, "snapshot_data", lambda root, model: root / model)
        write_json(tmp_path / "reference/training.json", {"train": [source], "holdout": []})
        models.append("reference")
    with pytest.raises(ValueError, match=f"回放 {replay_id}.*{message}"):
        selection.check_eval_maps(tmp_path, models, groups, False)


@pytest.mark.parametrize("missing", [False, True])
def test_eval_map_check_requires_original_training_file_digest(tmp_path, monkeypatch, missing):
    add_practice(tmp_path, 101, "probe", 11, "red")
    add_practice(tmp_path, 102, "probe", 11, "blue")
    source = add_replay(tmp_path, 1, 1, "training-grid")
    if missing:
        del source["sha256"]
    else:
        add_replay(tmp_path, 1, 1, "training-grid", frames=[])
    monkeypatch.setattr(selection.versions, "snapshot_data", lambda root, model: root / model)
    write_json(tmp_path / "reference/training.json", {"train": [source], "holdout": []})
    groups = selection.practice_groups(tmp_path, [101, 102])
    with pytest.raises(ValueError, match="缺少.*SHA-256" if missing else "SHA-256.*不符"):
        selection.check_eval_maps(tmp_path, ["reference"], groups, False)


def test_final_gate_rejects_reselecting_and_seen_development_layout(tmp_path):
    development = {"baseline": "baseline", "recommended_model": "frozen", "ids": [99]}
    with pytest.raises(ValueError, match="已选候选"):
        selection.selection_report(tmp_path, "baseline", ["other"], [1, 2], development=development)
    for invalid in (development | {"ids": []}, development | {"development_ids": [98]}):
        with pytest.raises(ValueError, match="原始开发回放"):
            selection.selection_report(tmp_path, "baseline", ["frozen"], [1, 2], development=invalid)
    add_practice(tmp_path, 1, "probe", 10, "red")
    add_practice(tmp_path, 2, "probe", 10, "blue")
    add_replay(tmp_path, 99, 999, "grid-10")
    with pytest.raises(ValueError, match="开发回放 99"):
        selection.selection_report(tmp_path, "baseline", ["frozen"], [1, 2], development=development)


def test_map_and_candidate_weights_and_duration_guard():
    def game(candidate, seed, distance, duration):
        return {"candidate": candidate, "seed": seed, "errors": {
            "median_distance_abs_error_cells": distance, "duration_abs_error_s": duration,
            "boss_death_mismatch": False,
        }}

    baseline = selection.summarize_errors([
        *[game("probe-a", 1, 8, 0)] * 4, game("probe-b", 1, 4, 0), game("probe-b", 2, 12, 40),
    ])
    trial = selection.summarize_errors([
        *[game("probe-a", 1, 6, 6)] * 4, game("probe-b", 1, 2, 0), game("probe-b", 2, 10, 30),
    ])
    assert baseline["overall"]["median_distance_mae_cells"] == 9
    assert baseline["by_candidate"]["probe-b"]["median_distance_mae_cells"] == 8
    assert trial["overall"]["duration_mae_s"] == 16.5
    decision = selection.promotion_decision(baseline, trial)
    assert decision["checks"] == {"distance_improved": True, "duration_not_increased": True,
                                  "boss_death_error_not_increased": True,
                                  "each_candidate_duration_within_limit": False}
    assert decision["eligible"] is False


def test_report_replays_each_id_once_per_model_and_picks_smallest_error(tmp_path, monkeypatch):
    definitions = [(1, "a", 41, "red"), (2, "a", 41, "blue"),
                   (3, "a", 41, "red"), (4, "a", 41, "blue"),
                   (5, "b", 42, "red"), (6, "b", 42, "blue")]
    for record in definitions:
        add_practice(tmp_path, *record)
    candidates = ["good", "best", "death-regression", "tied"]
    monkeypatch.setattr(selection.versions, "snapshot_data", lambda root, model: root / model)
    for model in candidates:
        write_json(tmp_path / model / "training.json", {"train": [], "holdout": []})

    calls = []

    def matched(root, models, candidates, seeds, *, replay_ids, include_common_alive):
        assert root == tmp_path and include_common_alive
        calls.append((candidates, seeds, replay_ids))
        distance = {"baseline": 6, "good": 7, "best": 8, "death-regression": 9, "tied": 6}
        official = {"seconds": 20, "boss": {"hp": 1}}
        return {"games": [{"replay_id": replay_id, "candidate": candidates[0], "model": model, "seed": seeds[0],
                           "official": official, "local": {"seconds": 20, "boss": {"hp": model != "death-regression"}},
                           "common_alive": {"official_median_distance_cells": 10,
                                            "local_median_distance_cells": distance[model]}}
                          for replay_id in replay_ids for model in models]}

    monkeypatch.setattr(selection, "matched_games", matched)
    report = selection.selection_report(tmp_path, "baseline", candidates, [6, 3, 4, 5, 2, 1])
    assert calls == [(["a"], [41], [1, 2]), (["a"], [41], [3, 4]), (["b"], [42], [5, 6])]
    assert len(report["games"]) == 30
    assert report["recommended_model"] == "best"
    assert report["models"]["death-regression"]["decision"]["checks"]["boss_death_error_not_increased"] is False
    assert report["models"]["tied"]["decision"]["eligible"] is False
