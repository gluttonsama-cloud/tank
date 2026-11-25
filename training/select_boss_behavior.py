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


def summarize_errors(games):
    grouped_games = defaultdict(list)
    for game in games:
        grouped_games[game["candidate"], game["seed"]].append(game["errors"])

    by_candidate_map = []
    maps, candidates = defaultdict(list), defaultdict(list)
    for (candidate, seed), rows in sorted(grouped_games.items()):
        group = {"candidate": candidate, "seed": seed, "game_count": len(rows)} | {
            metric: fmean(row[error] for row in rows) for metric, error in METRICS.items()
        }
        by_candidate_map.append(group)
        maps[seed].append(group)
        candidates[candidate].append(group)

    def combine(groups):
        return {"game_count": sum(group["game_count"] for group in groups)} | {
            metric: fmean(group[metric] for group in groups) for metric in METRICS
        }

    by_map = [{"seed": seed, "candidate_count": len(groups)} | combine(groups)
              for seed, groups in sorted(maps.items())]
    by_candidate = {candidate: {"map_count": len(groups)} | combine(groups)
                    for candidate, groups in sorted(candidates.items())}
    return {"overall": {"map_count": len(by_map)} | combine(by_map), "by_map": by_map,
            "by_candidate": by_candidate, "by_candidate_map": by_candidate_map}


def promotion_decision(baseline, trial):
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
    return {"eligible": all(checks.values()), "checks": checks, "overall_delta": delta,
            "candidate_duration_delta_s": by_candidate}


def selection_report(root, baseline, models, replay_ids, require_unseen=False, development=None):
    if not models or len({baseline, *models}) != len(models) + 1:
        raise ValueError("基准与待比较模型必须是不同版本，且至少提供一个待比较模型")
    if not replay_ids or any(replay_id < 1 for replay_id in replay_ids) or len(set(replay_ids)) != len(replay_ids):
        raise ValueError("官网回放 ID 必须为不重复的正整数")

    variants = [baseline, *models]
    development_ids = []
    if development is not None:
        if baseline != development["baseline"] or models != [development["recommended_model"]]:
            raise ValueError("最终验证只比较开发报告中的基准和已选候选")
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
    for candidate, seed, ids in pairs:
        games.extend(matched_games(root, variants, [candidate], [seed], replay_ids=ids,
                                  include_common_alive=True)["games"])
    by_model = defaultdict(list)
    for game in games:
        game["errors"] = game_errors(game)
        by_model[game["model"]].append(game)
    summaries = {model: summarize_errors(by_model[model]) for model in variants}
    for model in models:
        summaries[model]["decision"] = promotion_decision(summaries[baseline], summaries[model])
    eligible = [model for model in models if summaries[model]["decision"]["eligible"]]
    recommendation = min(eligible, default=baseline,
                         key=lambda model: (summaries[model]["overall"]["median_distance_mae_cells"], model))
    return {"rules": rules.as_dict()["version"], "baseline": baseline, "recommended_model": recommendation,
            "require_unseen": require_unseen, "development_ids": development_ids,
            "ids": sorted(replay_ids), "models": summaries, "games": games,
            "method": {
                "distance": "每局取官网与所有本地版本中双方均存活的共同帧，分别计算距离中位数，再取官网与本地中位数的绝对差。",
                "aggregation": "先对相同策略与地图的各局取均值；总体按地图内策略等权和地图间等权依次取均值。策略结果对其地图等权。",
                "data_isolation": "核对回放格式、规则和训练清单中的原文件 SHA-256，检查评估集的 ID、地图种子和网格是否与训练集重叠。启用 require_unseen 时也检查留出集。",
                "tiebreak": "符合全部阈值的版本按总体距离误差从小到大排序，同分按版本 ID 排序。",
            },
            "thresholds": {"median_distance_mae_delta_cells": {"lt": 0},
                           "duration_mae_delta_s": {"lte": 0}, "boss_death_error_rate_delta": {"lte": 0},
                           "each_candidate_duration_mae_delta_s": {"lte": MAX_CANDIDATE_DURATION_REGRESSION_S},
                           "nonincrease_comparison_tolerance": FLOAT_TOLERANCE}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="基准陪练版本 ID")
    parser.add_argument("--models", nargs="+", required=True, help="待比较的陪练版本 ID")
    parser.add_argument("--ids", nargs="+", type=int, required=True, help="本机归档的官网终极练习回放 ID")
    parser.add_argument("--require-unseen", action="store_true", help="同时检查与选模留出地图无重叠")
    parser.add_argument("--development-report", type=Path, help="按开发报告锁定候选并排除已用于选模的地图")
    parser.add_argument("--output", type=Path, required=True, help="选模报告 JSON 路径")
    args = parser.parse_args()
    try:
        development = read_json(args.development_report) if args.development_report else None
        report = selection_report(project_root(), args.baseline, args.models, args.ids, args.require_unseen, development)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
