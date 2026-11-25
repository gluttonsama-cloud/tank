"""在同一批官网终极练习回放上比较本地陪练版本。"""

import argparse
import gzip
import json
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import fmean

from seedcup.game import rules
from seedcup.game.replay import FORMAT_VERSION

from tank import versions
from tank.workspace import project_root, read_json, write_json

if __package__:
    from .boss_behavior_matches import matched_games
else:
    from boss_behavior_matches import matched_games


METRICS = {
    "median_distance_mae_cells": "median_distance_abs_error_cells",
    "duration_mae_s": "duration_abs_error_s",
    "boss_death_error_rate": "boss_death_mismatch",
}
MOVEMENT_METRICS = ("boss_start_mae_s", "boss_shrink_margin_mae_cells")
MOVEMENT_COUNTS = ("start_valid_game_count", "shrink_valid_game_count", "shrink_common_frame_count",
                   "official_started_local_censored_game_count", "official_boss_zone_death_count",
                   "local_boss_zone_death_count")
OUTSIDE_SECONDS = ("official_outside_s", "local_outside_s", "official_inside_local_outside_s")
MAX_CANDIDATE_DURATION_REGRESSION_S = 5
FLOAT_TOLERANCE = 1e-12


def practice_groups(root, replay_ids):
    directory = root / ".local/online/practice"
    groups = defaultdict(lambda: {"red": [], "blue": []})
    for replay_id in replay_ids:
        paths = list(directory.glob(f"*/boss/{replay_id}.json"))
        if len(paths) != 1:
            raise ValueError(f"回放 {replay_id} 必须对应一条本机终极练习记录，已找到 {len(paths)} 条")
        path = paths[0]
        candidate = path.parent.parent.name
        record = read_json(path)
        if record["id"] != replay_id or record.get("version_id", candidate) != candidate:
            raise ValueError(f"回放 {replay_id} 的练习记录与文件路径不一致")
        seed = record.get("map_seed", record["seed"])
        side = record["side"]
        if side not in ("red", "blue"):
            raise ValueError(f"回放 {replay_id} 的出生侧无效 {side}")
        groups[candidate, seed][side].append(replay_id)
    return groups


def check_eval_maps(root, models, groups, require_unseen, development_ids=()):
    replay_dir = root / ".local/online/replays"
    maps = {}
    expected_rules = rules.as_dict()

    def replay_map(replay_id):
        if replay_id not in maps:
            compressed = (replay_dir / f"{replay_id}.json.gz").read_bytes()
            replay = json.loads(gzip.decompress(compressed))
            if replay.get("format") != FORMAT_VERSION:
                raise ValueError(f"回放 {replay_id} 的格式与当前官方 SDK 不一致")
            if rules.VARIANT or replay.get("rules") != expected_rules:
                raise ValueError(f"回放 {replay_id} 的规则与当前官方 SDK 不一致")
            if replay["match"]["id"] != replay_id:
                raise ValueError(f"回放 {replay_id} 与回放文件中的对局 ID 不一致")
            maps[replay_id] = replay["map"], sha256(compressed).hexdigest()
        return maps[replay_id]

    eval_ids = {replay_id for sides in groups.values() for ids in sides.values() for replay_id in ids}
    eval_seeds, eval_grids = {}, {}
    for (_, seed), sides in groups.items():
        for replay_id in sides["red"] + sides["blue"]:
            game_map, _ = replay_map(replay_id)
            if game_map["seed"] != seed:
                raise ValueError(f"回放 {replay_id} 的地图种子与练习记录不符")
            eval_seeds[seed] = replay_id
            eval_grids[game_map["grid"]] = replay_id

    for replay_id in development_ids:
        game_map, _ = replay_map(replay_id)
        if game_map["seed"] in eval_seeds or game_map["grid"] in eval_grids:
            raise ValueError(f"新图验证与开发回放 {replay_id} 使用了相同地图")

    for model in models:
        if model == "baseline":
            continue
        training = read_json(versions.snapshot_data(root, model) / "training.json")
        for split in (("train", "holdout") if require_unseen else ("train",)):
            for source in training[split]:
                source_id = source["id"]
                seed = source["seed"]
                if source_id in eval_ids or seed in eval_seeds:
                    replay_id = source_id if source_id in eval_ids else eval_seeds[seed]
                    raise ValueError(f"评估回放 {replay_id} 与 {model} 的 {split} 回放 {source_id} 种子或 ID 重复")
                expected_digest = source.get("sha256")
                if not expected_digest:
                    raise ValueError(f"{model} 的 {split} 回放 {source_id} 缺少原文件 SHA-256 摘要")
                game_map, digest = replay_map(source_id)
                if digest != expected_digest:
                    raise ValueError(f"{model} 的 {split} 回放 {source_id} 原文件 SHA-256 与训练清单不符")
                if game_map["seed"] != seed:
                    raise ValueError(f"{model} 的 {split} 回放 {source_id} 与训练清单的种子不符")
                if game_map["grid"] in eval_grids:
                    replay_id = eval_grids[game_map["grid"]]
                    raise ValueError(f"评估回放 {replay_id} 与 {model} 的 {split} 回放 {source_id} 地图网格重复")


def game_errors(game):
    distance = game["common_alive"]
    official, local = game["official"], game["local"]
    ticks = round(abs(official["seconds"] - local["seconds"]) * rules.TICK_HZ)
    official_died = official["boss"]["hp"] <= 0
    local_died = local["boss"]["hp"] <= 0
    return {"median_distance_abs_error_cells": abs(distance["official_median_distance_cells"] -
                                                  distance["local_median_distance_cells"]),
            "duration_abs_error_s": ticks / rules.TICK_HZ,
            "official_boss_died": official_died, "local_boss_died": local_died,
            "boss_death_mismatch": official_died != local_died}


def mean_or_none(values):
    observed = [value for value in values if value is not None]
    return fmean(observed) if observed else None


def sum_or_none(values):
    observed = [value for value in values if value is not None]
    return sum(observed) if observed else None


def summarize_movement(games):
    starts = [game["movement"]["start"]["boss"]["abs_error_s"] for game in games
              if game["movement"]["boss_start_observed_in_all_versions"]]
    shrink = [game["common_alive"]["shrink"] for game in games]
    return {
        "boss_start_mae_s": mean_or_none(starts),
        "boss_shrink_margin_mae_cells": mean_or_none(row["margin_mae_cells"] for row in shrink),
        "start_valid_game_count": len(starts),
        "shrink_valid_game_count": sum(row["frames"] > 0 for row in shrink),
        "shrink_common_frame_count": sum(row["frames"] for row in shrink),
        "official_started_local_censored_game_count": sum(
            not game["official"]["boss"]["movement_start"]["censored"]
            and game["local"]["boss"]["movement_start"]["censored"]
            for game in games
        ),
        "official_boss_zone_death_count": sum(game["movement"]["official_boss_zone_died"] for game in games),
        "local_boss_zone_death_count": sum(game["movement"]["local_boss_zone_died"] for game in games),
    } | {key: sum_or_none(row[key] for row in shrink) for key in OUTSIDE_SECONDS}


def summarize_errors(games, *, include_movement=False):
    grouped_games = defaultdict(list)
    for game in games:
        grouped_games[game["candidate"], game["seed"]].append(game)

    by_candidate_map = []
    maps, candidates = defaultdict(list), defaultdict(list)
    for (candidate, seed), rows in sorted(grouped_games.items()):
        group = {"candidate": candidate, "seed": seed, "game_count": len(rows)} | {
            metric: fmean(row["errors"][error] for row in rows) for metric, error in METRICS.items()
        }
        if include_movement:
            group["movement"] = summarize_movement(rows)
        by_candidate_map.append(group)
        maps[seed].append(group)
        candidates[candidate].append(group)

    def combine(groups, *, across_maps=False):
        result = {"game_count": sum(group["game_count"] for group in groups)} | {
            metric: fmean(group[metric] for group in groups) for metric in METRICS
        }
        if include_movement:
            rows = [group["movement"] for group in groups]
            movement = {key: mean_or_none(row[key] for row in rows) for key in MOVEMENT_METRICS}
            movement.update({key: sum(row[key] for row in rows) for key in MOVEMENT_COUNTS})
            movement.update({key: sum_or_none(row[key] for row in rows) for key in OUTSIDE_SECONDS})
            if across_maps:
                movement.update({
                    "start_valid_map_count": sum(row["start_valid_game_count"] > 0 for row in rows),
                    "shrink_valid_map_count": sum(row["shrink_valid_game_count"] > 0 for row in rows),
                })
            result["movement"] = movement
        return result

    by_map = [{"seed": seed, "candidate_count": len(groups)} | combine(groups)
              for seed, groups in sorted(maps.items())]
    by_candidate = {candidate: {"map_count": len(groups)} | combine(groups, across_maps=True)
                    for candidate, groups in sorted(candidates.items())}
    return {"overall": {"map_count": len(by_map)} | combine(by_map, across_maps=True), "by_map": by_map,
            "by_candidate": by_candidate, "by_candidate_map": by_candidate_map}


def paired_summaries(baseline_games, candidate_games):
    baseline_by_id = {game["replay_id"]: game for game in baseline_games}
    paired_games = {"baseline": [], "candidate": []}
    for candidate in candidate_games:
        pair = candidate["movement"]["with_baseline"]
        baseline = baseline_by_id[candidate["replay_id"]]
        for role, game in (("baseline", baseline), ("candidate", candidate)):
            movement = game["movement"] | {"boss_start_observed_in_all_versions": pair["boss_start_observed_in_pair"]}
            paired_game = game | {"common_alive": pair[role], "movement": movement}
            paired_game["errors"] = game_errors(paired_game)
            paired_games[role].append(paired_game)
    return {role: summarize_errors(games, include_movement=True) for role, games in paired_games.items()}


def promotion_decision(baseline, trial, *, movement_gate=False):
    delta = {key: trial["overall"][key] - baseline["overall"][key] for key in METRICS}
    by_candidate = {key: row["duration_mae_s"] - baseline["by_candidate"][key]["duration_mae_s"]
                    for key, row in trial["by_candidate"].items()}
    checks = {
        "distance_improved": delta["median_distance_mae_cells"] < 0,
        "duration_not_increased": delta["duration_mae_s"] <= FLOAT_TOLERANCE,
        "boss_death_error_not_increased": delta["boss_death_error_rate"] <= FLOAT_TOLERANCE,
        "each_candidate_duration_within_limit": all(
            diff <= MAX_CANDIDATE_DURATION_REGRESSION_S + FLOAT_TOLERANCE for diff in by_candidate.values()
        ),
    }
    decision = {"checks": checks, "overall_delta": delta, "candidate_duration_delta_s": by_candidate}
    if movement_gate:
        reference, candidate = baseline["overall"]["movement"], trial["overall"]["movement"]
        movement_delta = {
            key: candidate[key] - reference[key] if candidate[key] is not None and reference[key] is not None else None
            for key in MOVEMENT_METRICS
        }
        start_delta, shrink_delta = (movement_delta[key] for key in MOVEMENT_METRICS)
        censored_key = "official_started_local_censored_game_count"
        censored_by_candidate = {
            key: row["movement"][censored_key] - baseline["by_candidate"][key]["movement"][censored_key]
            for key, row in trial["by_candidate"].items()
        }
        decision["movement_delta"] = movement_delta | {censored_key: candidate[censored_key] - reference[censored_key]}
        decision["candidate_start_censored_game_count_delta"] = censored_by_candidate
        checks.update({
            "movement_has_valid_shrink": reference["shrink_valid_map_count"] > 0 and candidate["shrink_valid_map_count"] > 0,
            "boss_start_error_not_increased": start_delta is not None and start_delta <= FLOAT_TOLERANCE,
            "boss_shrink_margin_error_not_increased": shrink_delta is not None and shrink_delta <= FLOAT_TOLERANCE,
            "movement_improved": any(diff is not None and diff < -FLOAT_TOLERANCE for diff in movement_delta.values()),
            "boss_start_censoring_not_increased": all(diff <= 0 for diff in censored_by_candidate.values()),
        })
    return {"eligible": all(checks.values())} | decision


def selection_report(root, baseline, models, replay_ids, require_unseen=False, development=None, *, movement_gate=False):
    if not models or len({baseline, *models}) != len(models) + 1:
        raise ValueError("基准与待比较模型必须是不同版本，且至少提供一个待比较模型")
    if not replay_ids or any(replay_id < 1 for replay_id in replay_ids) or len(set(replay_ids)) != len(replay_ids):
        raise ValueError("官网回放 ID 必须为不重复的正整数")

    variants = [baseline, *models]
    development_ids = []
    if development is not None:
        if baseline != development["baseline"] or models != [development["recommended_model"]]:
            raise ValueError("最终验证只比较开发报告中的基准和已选候选")
        development_movement_gate = development.get("movement_gate", False)
        if type(development_movement_gate) is not bool:
            raise ValueError("开发报告的 movement_gate 必须是布尔值")
        movement_gate = movement_gate or development_movement_gate
        require_unseen = True
        development_ids = development["ids"]
        if (development.get("development_ids") or not development_ids
                or not all(type(i) is int and i > 0 for i in development_ids)):
            raise ValueError("需要含原始开发回放 ID 的开发报告")
    groups = practice_groups(root, replay_ids)
    pairs = []
    for (candidate, seed), sides in sorted(groups.items()):
        red, blue = sides["red"], sides["blue"]
        if not red or len(red) != len(blue):
            raise ValueError(f"策略 {candidate} 地图 {seed} 需要成对的红蓝回放，已找到红方 {len(red)} 局、蓝方 {len(blue)} 局")
        pairs.extend((candidate, seed, [a, b]) for a, b in zip(sorted(red), sorted(blue), strict=True))
    check_eval_maps(root, variants, groups, require_unseen, development_ids)

    games = []
    movement_options = {"movement_baseline": baseline} if movement_gate else {}
    for candidate, seed, ids in pairs:
        games.extend(matched_games(root, variants, [candidate], [seed], replay_ids=ids,
                                  include_common_alive=True, **movement_options)["games"])
    by_model = defaultdict(list)
    for game in games:
        game["errors"] = game_errors(game)
        by_model[game["model"]].append(game)
    summaries = {model: summarize_errors(by_model[model], include_movement=True) for model in variants}
    distance_key = "median_distance_mae_cells"
    ranking_distances = {}
    for model in models:
        summary = summaries[model]
        reference, trial = summaries[baseline], summary
        if movement_gate:
            comparison = paired_summaries(by_model[baseline], by_model[model])
            summary["comparison_with_baseline"] = comparison
            reference, trial = comparison["baseline"], comparison["candidate"]
        summary["decision"] = promotion_decision(reference, trial, movement_gate=movement_gate)
        ranking_distances[model] = trial["overall"][distance_key]
    eligible = [model for model in models if summaries[model]["decision"]["eligible"]]
    recommendation = min(eligible, default=baseline, key=lambda model: (ranking_distances[model], model))
    return {"rules": rules.as_dict()["version"], "baseline": baseline, "recommended_model": recommendation,
            "movement_gate": movement_gate,
            "require_unseen": require_unseen, "development_ids": development_ids,
            "ids": sorted(replay_ids), "models": summaries, "games": games,
            "method": {
                "distance": "每局取官网与所有本地版本中双方均存活的共同帧，分别计算距离中位数，再取官网与本地中位数的绝对差。",
                "movement_start": "相对各自出生位位移首次超过 0.05 格为起步；此前结束则右删失，记录观测时长及可知误差下界。Boss 起步误差只计入所有版本与官网均观测到起步的局。",
                "shrink": "从 SDK 首次减小圈半径的帧起，在官网与全部本地版本中双方都存活的共同帧上比较 Boss 的安全余量，即圈半径减去到圈心的距离。无缩圈共同帧的局不计缩圈误差，圈死按整局结算。",
                "movement_gate": "每个候选单独和基准、官网对齐。起步误差只计入三者均观测到起步的局。距离和缩圈误差使用三个回放中红蓝双方均存活的共同帧。结果见候选的 comparison_with_baseline。终验沿用开发报告已启用的运动门槛。",
                "aggregation": "先对相同策略与地图的各局取均值；总体按地图内策略等权和地图间等权依次取均值。策略结果对其地图等权。",
                "data_isolation": "核对回放格式、规则和训练清单中的原文件 SHA-256，检查评估集的 ID、地图种子和网格是否与训练集重叠。启用 require_unseen 时也检查留出集。",
                "tiebreak": ("符合全部阈值的版本按各自配对比较的总体距离误差从小到大排序，同分按版本 ID 排序。"
                             if movement_gate else "符合全部阈值的版本按全组共同帧的总体距离误差从小到大排序，同分按版本 ID 排序。"),
            },
            "thresholds": {"median_distance_mae_delta_cells": {"lt": 0},
                           "duration_mae_delta_s": {"lte": 0}, "boss_death_error_rate_delta": {"lte": 0},
                           "each_candidate_duration_mae_delta_s": {"lte": MAX_CANDIDATE_DURATION_REGRESSION_S},
                           "nonincrease_comparison_tolerance": FLOAT_TOLERANCE,
                           "movement_gate": {
                               "boss_start_mae_delta_s": {"lte": 0},
                               "boss_shrink_margin_mae_delta_cells": {"lte": 0},
                               "requires_movement_improvement": True,
                               "shrink_valid_map_count": {"gte": 1},
                               "each_candidate_official_started_local_censored_game_count_delta": {"lte": 0},
                           }}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="基准陪练版本 ID")
    parser.add_argument("--models", nargs="+", required=True, help="待比较的陪练版本 ID")
    parser.add_argument("--ids", nargs="+", type=int, required=True, help="本机归档的官网终极练习回放 ID")
    parser.add_argument("--require-unseen", action="store_true", help="同时检查与选模留出地图无重叠")
    parser.add_argument("--movement-gate", action="store_true", help="同时要求起步误差与缩圈余量误差不增加且至少一项改善")
    parser.add_argument("--development-report", type=Path, help="锁定开发候选、排除开发地图并沿用已启用的运动门槛")
    parser.add_argument("--output", type=Path, required=True, help="选模报告 JSON 路径")
    args = parser.parse_args()
    try:
        development = read_json(args.development_report) if args.development_report else None
        report = selection_report(project_root(), args.baseline, args.models, args.ids, args.require_unseen, development,
                                  movement_gate=args.movement_gate)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
