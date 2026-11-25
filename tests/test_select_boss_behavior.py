from hashlib import sha256
from pathlib import Path

import pytest
from seedcup.game import generate_map, rules
from seedcup.game.maps import Spawn
from seedcup.game.replay import FORMAT_VERSION, save

from tank.workspace import write_json
from training import boss_behavior_matches as matches, select_boss_behavior as selection
from training.boss_behavior_matches import common_alive_distances, compare_movement_starts, movement_start


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


def test_movement_start_uses_spawn_displacement_and_reports_censoring():
    red, blue = Spawn(2.5, 4.5, 0), Spawn(20.5, 5.5, 180)
    frames = [[tick, [red.x, red.y], [blue.x, blue.y]] for tick in range(31)]
    frames[1][1][0] += 0.05
    frames[2][1][0] += 0.06
    red_start, blue_start = movement_start(frames, 1, red), movement_start(frames, 2, blue)
    assert red_start == {"seconds": 0.2, "censored": False, "censor_lower_bound_s": None,
                         "displacement_at_3s_cells": 0}
    assert blue_start == {"seconds": None, "censored": True, "censor_lower_bound_s": 3,
                          "displacement_at_3s_cells": 0}
    assert movement_start(frames[:3], 1, red)["displacement_at_3s_cells"] is None
    assert compare_movement_starts(red_start, blue_start) == {"abs_error_s": None, "abs_error_lower_bound_s": 2.8}
    assert compare_movement_starts(blue_start, red_start) == {"abs_error_s": None, "abs_error_lower_bound_s": 2.8}
    assert compare_movement_starts(blue_start, blue_start) == {"abs_error_s": None, "abs_error_lower_bound_s": None}


def test_shrink_diagnostics_exclude_pre_shrink_and_any_teams_dead_frames():
    start = rules.ZONE_START_TICKS

    def frame(tick, margin, red_alive=1):
        radius = rules.zone_radius(tick)
        return [tick, [0, 0, 0, 100, red_alive], [rules.ZONE_CX + radius - margin, rules.ZONE_CY, 0, 100, 1], [], []]

    official = {"match": {"id": 2}, "frames": [frame(start, 100), frame(start + 1, 1), frame(start + 2, -1),
                                                frame(start + 3, 10)]}
    early_end = {"frames": [frame(start, 30), frame(start + 1, -2), frame(start + 2, -3), frame(start + 3, 20, 0)]}
    survives = {"frames": [frame(start, 10), frame(start + 1, 2), frame(start + 2, 3), frame(start + 3, -5)]}
    first, second = common_alive_distances(official, [early_end, survives], boss_side=1)
    assert first["frames"] == second["frames"] == 3
    assert first["shrink"] == pytest.approx({"frames": 2, "margin_mae_cells": 2.5, "official_outside_s": 0.1,
                                             "local_outside_s": 0.2, "official_inside_local_outside_s": 0.1})
    assert second["shrink"] == pytest.approx({"frames": 2, "margin_mae_cells": 2.5, "official_outside_s": 0.1,
                                              "local_outside_s": 0, "official_inside_local_outside_s": 0})
    [unobserved] = common_alive_distances(official, [{"frames": [frame(start, 3)]}], boss_side=1)
    assert unobserved["shrink"] == {"frames": 0, "margin_mae_cells": None, "official_outside_s": None,
                                    "local_outside_s": None, "official_inside_local_outside_s": None}


def test_paired_match_diagnostics_reuse_replays_independently_of_third_version(monkeypatch, tmp_path):
    game_map = generate_map(11).to_dict()
    red, blue = game_map["spawns"]
    shrink_start = rules.ZONE_START_TICKS

    def tank(spawn, dx=0, alive=True):
        return [spawn["x"] + dx, spawn["y"], spawn["heading"], 100 if alive else 0, alive, 0, 0, 0, 0, 0, 0, 0]

    def replay(move_tick, last_tick):
        frames = [[tick, tank(red, alive=tick < last_tick), tank(blue, dx=-int(tick >= move_tick)), [], []]
                  for tick in range(last_tick + 1)]
        return {
            "match": {"id": 1}, "map": game_map, "frames": frames,
            "result": {"ticks": last_tick, "scores": [{"total": 0}, {"total": 0}],
                       "reason": "kill", "zone_deaths": [False, False]},
        }

    official = replay(1, shrink_start + 4)
    locals_by_model = {"baseline": replay(2, shrink_start + 3), "trial": replay(1, shrink_start + 4),
                       "early": replay(100, 10)}
    calls = []

    def run_match(red, blue, seed, game_no, *, replay):
        assert red == "probe" and seed == 11 and game_no == 0 and replay
        calls.append(blue)
        local = locals_by_model[blue]
        return {"replay": local, "result": local["result"], "scores": [0, 0]}

    monkeypatch.setattr(matches, "official_cases", lambda *args, **kwargs: {("probe", 11, 0): [({"id": 1}, 0)]})
    monkeypatch.setattr(matches, "load", lambda path: official)
    monkeypatch.setattr(matches.versions, "verify", lambda root, model: None)
    monkeypatch.setattr(matches.versions, "load_bot", lambda root, model: model)
    monkeypatch.setattr(matches, "run_match", run_match)
    report = matches.matched_games(tmp_path, ["baseline", "trial", "early"], ["probe"], [11],
                                   include_common_alive=True, movement_baseline="baseline")
    assert calls == ["baseline", "trial", "early"]
    baseline, trial, early = report["games"]
    assert all(not row["movement"]["boss_start_observed_in_all_versions"] for row in report["games"])
    assert all(row["common_alive"]["shrink"]["frames"] == 0 for row in report["games"])
    assert "with_baseline" not in baseline["movement"]
    trial_pair = trial["movement"]["with_baseline"]
    assert trial_pair["boss_start_observed_in_pair"]
    assert trial_pair["baseline"]["frames"] == trial_pair["candidate"]["frames"] == shrink_start + 3
    assert trial_pair["baseline"]["shrink"]["frames"] == trial_pair["candidate"]["shrink"]["frames"] == 2
    early_pair = early["movement"]["with_baseline"]
    assert not early_pair["boss_start_observed_in_pair"]
    assert early_pair["baseline"]["shrink"]["frames"] == early_pair["candidate"]["shrink"]["frames"] == 0


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


def movement_game(candidate, seed, start_error, shrink_error, *, distance_error=0, shared_start=True, local_censored=False):
    observed = shrink_error is not None
    return {
        "candidate": candidate, "seed": seed,
        "official": {"seconds": 50, "boss": {"hp": 1, "movement_start": {"censored": False}}},
        "local": {"seconds": 50, "boss": {"hp": 1, "movement_start": {"censored": local_censored}}},
        "errors": {"median_distance_abs_error_cells": distance_error, "duration_abs_error_s": 0,
                   "boss_death_mismatch": False},
        "movement": {"start": {"boss": {"abs_error_s": start_error}}, "boss_start_observed_in_all_versions": shared_start,
                     "official_boss_zone_died": False, "local_boss_zone_died": local_censored},
        "common_alive": {
            "shrink": {"frames": 10 if observed else 0, "margin_mae_cells": shrink_error}
            | dict.fromkeys(selection.OUTSIDE_SECONDS, 0 if observed else None)
        },
    }


def test_movement_gate_uses_valid_maps_and_keeps_censored_game_coverage():
    baseline_games = [
        *[movement_game("a", 1, 2, 2, distance_error=5)] * 3,
        movement_game("b", 1, 4, None, distance_error=5),
        movement_game("b", 2, 8, 6, distance_error=5),
        movement_game("b", 3, 10, None, distance_error=5, shared_start=False, local_censored=True),
    ]
    valid_trial = [
        *[movement_game("a", 1, 2, 2)] * 3,
        movement_game("b", 1, 2, None),
        movement_game("b", 2, 8, 6),
        movement_game("b", 3, 20, None, shared_start=False),
    ]
    baseline = selection.summarize_errors(baseline_games, include_movement=True)
    trial = selection.summarize_errors(valid_trial, include_movement=True)
    assert baseline["overall"]["movement"] == {
        "boss_start_mae_s": 5.5, "boss_shrink_margin_mae_cells": 4,
        "start_valid_game_count": 5, "start_valid_map_count": 2,
        "shrink_valid_game_count": 4, "shrink_valid_map_count": 2, "shrink_common_frame_count": 40,
        "official_started_local_censored_game_count": 1, "official_boss_zone_death_count": 0,
        "local_boss_zone_death_count": 1, **dict.fromkeys(selection.OUTSIDE_SECONDS, 0),
    }
    assert selection.promotion_decision(baseline, trial, movement_gate=True)["eligible"]

    only_censor_gain = selection.summarize_errors([
        *[row | {"errors": row["errors"] | {"median_distance_abs_error_cells": 0}} for row in baseline_games[:-1]],
        movement_game("b", 3, 2, None, shared_start=False),
    ], include_movement=True)
    decision = selection.promotion_decision(baseline, only_censor_gain, movement_gate=True)
    assert decision["movement_delta"]["official_started_local_censored_game_count"] == -1
    assert not decision["checks"]["movement_improved"]
    assert not decision["eligible"]

    lost_coverage = selection.summarize_errors([
        *valid_trial[:3], movement_game("a", 3, None, None, shared_start=False, local_censored=True), *valid_trial[3:],
    ], include_movement=True)
    decision = selection.promotion_decision(baseline, lost_coverage, movement_gate=True)
    assert decision["movement_delta"]["official_started_local_censored_game_count"] == 0
    assert decision["candidate_start_censored_game_count_delta"] == {"a": 1, "b": -1}
    assert decision["checks"]["boss_start_censoring_not_increased"] is False
    assert decision["eligible"] is False

    without_shrink = selection.summarize_errors([movement_game("a", 1, 1, None),
                                                 movement_game("b", 1, 1, None)], include_movement=True)
    assert without_shrink["overall"]["movement"]["boss_shrink_margin_mae_cells"] is None
    decision = selection.promotion_decision(baseline, without_shrink, movement_gate=True)
    assert decision["checks"]["movement_has_valid_shrink"] is False
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

    def matched(root, models, candidates, seeds, *, replay_ids, include_common_alive, movement_baseline=None):
        assert root == tmp_path and include_common_alive
        calls.append((candidates, seeds, replay_ids))
        distance = {"baseline": 6, "good": 5, "best": 8, "death-regression": 9, "tied": 6}
        gate_distance = distance | {"good": 9, "tied": 7}
        absent = "death-regression" in models
        games = []
        for replay_id in replay_ids:
            for model in models:
                censored = model == "death-regression"
                start_error = None if censored else 0 if model in ("good", "tied") else 1
                game = movement_game(candidates[0], seeds[0], start_error, None if absent else 1,
                                     shared_start=not absent, local_censored=censored)
                game.update({"replay_id": replay_id, "model": model})
                game["local"]["boss"]["hp"] = not censored
                game["common_alive"].update({"official_median_distance_cells": 10,
                                             "local_median_distance_cells": distance[model]})
                if movement_baseline is not None and model != movement_baseline:
                    pair_game = movement_game(candidates[0], seeds[0], start_error, None if censored else 1)
                    pair = {
                        "boss_start_observed_in_pair": not censored,
                        "baseline": pair_game["common_alive"] | {
                            "official_median_distance_cells": 10, "local_median_distance_cells": gate_distance["baseline"],
                        },
                        "candidate": pair_game["common_alive"] | {
                            "official_median_distance_cells": 10, "local_median_distance_cells": gate_distance[model],
                        },
                    }
                    game["movement"]["with_baseline"] = pair
                games.append(game)
        return {"games": games}

    monkeypatch.setattr(selection, "matched_games", matched)
    report = selection.selection_report(tmp_path, "baseline", candidates, [6, 3, 4, 5, 2, 1])
    assert calls == [(["a"], [41], [1, 2]), (["a"], [41], [3, 4]), (["b"], [42], [5, 6])]
    assert len(report["games"]) == 30
    assert report["recommended_model"] == "best"
    assert report["models"]["death-regression"]["decision"]["checks"]["boss_death_error_not_increased"] is False
    assert report["models"]["tied"]["decision"]["eligible"] is False
    report = selection.selection_report(tmp_path, "baseline", candidates, [6, 3, 4, 5, 2, 1], movement_gate=True)
    assert report["recommended_model"] == "good"
    summaries = report["models"]
    assert summaries["best"]["decision"]["checks"]["movement_improved"] is False
    assert summaries["tied"]["decision"]["eligible"] is True
    comparison = summaries["good"]["comparison_with_baseline"]
    assert summaries["good"]["overall"]["movement"]["shrink_valid_game_count"] == 0
    assert comparison["candidate"]["overall"]["movement"]["shrink_valid_game_count"] == 6
    assert comparison["candidate"]["overall"]["median_distance_mae_cells"] == 1
    assert comparison["baseline"]["overall"]["median_distance_mae_cells"] == 4
    without_third = selection.selection_report(tmp_path, "baseline", ["good"], [1, 2, 3, 4, 5, 6], movement_gate=True)
    assert summaries["good"]["decision"] == without_third["models"]["good"]["decision"]
    assert comparison == without_third["models"]["good"]["comparison_with_baseline"]

    add_practice(tmp_path, 7, "b", 43, "red")
    add_practice(tmp_path, 8, "b", 43, "blue")
    final = selection.selection_report(tmp_path, "baseline", ["good"], [7, 8], development=report)
    assert final["movement_gate"] is True
    assert final["require_unseen"] is True
    assert final["recommended_model"] == "good"
