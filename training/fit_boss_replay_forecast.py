"""Fit a short-horizon predictor on verified official boss practice replays."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import hypot
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from seedcup.game import SIDES, GameMap, generate_map, rules
from seedcup.game.bots import clear_shot
from seedcup.game.replay import FORMAT_VERSION

from tank.online import PRACTICE_NAMES
from tank.sampling_data import layout_hash
from tank.workspace import project_root, read_json, write_json

if __package__:
    from .boss_forecast import basis, state_vector
    from .boss_specialist_policy import flight_ticks
    from .fit_boss_sparring import replay_observation
else:
    from boss_forecast import basis, state_vector
    from boss_specialist_policy import flight_ticks
    from fit_boss_sparring import replay_observation


MOVE_STEP = rules.SPEED * rules.DT
SPLITS = ("train", "development", "blind")
DIAGNOSTIC_SEED = 630720971
PRIOR_JOBS = (
    "train-c177458faaef46e5ae5f63270c23cb98",
    "train-83b51e2a98c9466099eb2686ad49f6b3",
)
# Development and final layouts recorded in the coverage, probe-mix, range and movement plans.
BEHAVIOR_RESERVED_SEEDS = (
    422440491,
    833501530,
    833710988,
    833815717,
    833920446,
    834025175,
    834129904,
    834234633,
    903414189,
    903833105,
    903937834,
    904042563,
    904147292,
    904252021,
    904356750,
)


@dataclass(frozen=True)
class Samples:
    features: np.ndarray
    displacement: np.ndarray
    reach: np.ndarray
    shooter: np.ndarray
    fired: np.ndarray


def official_boss_side(meta):
    """Resolve the sole registered practice bot and the opposite numbered human team."""
    if (
        type(meta["id"]) is not int
        or meta["id"] <= 0
        or meta["kind"] != "room"
        or meta["public"] is not True
        or not meta["room"]
    ):
        return None
    for side in SIDES:
        boss, human = meta[side], meta[SIDES[1 - SIDES.index(side)]]
        if (
            boss["bot"] is True
            and boss["id"] is None
            and boss["name"] == PRACTICE_NAMES["boss"]
            and human["bot"] is False
            and type(human["id"]) is int
            and human["id"] > 0
        ):
            return side
    return None


def archived_replay(root, entry):
    path = root / ".local/online/replays" / f"{entry['id']}.json.gz"
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"回放 {entry['id']} 的 SHA-256 与采样索引不一致")
    replay = json.loads(gzip.decompress(raw))
    meta, game_map = replay["match"], replay["map"]
    target_side = official_boss_side(meta)
    if (
        target_side is None
        or meta["id"] != entry["id"]
        or entry["target"] != "boss"
        or any(meta[side] != entry["players"][side] for side in SIDES)
        or target_side != entry["target_side"]
        or meta["game"] != entry["game_no"]
        or game_map["seed"] != entry["seed"]
        or layout_hash(game_map) != entry["grid_sha256"]
    ):
        raise ValueError(f"回放 {entry['id']} 的官网房间、名单或地图与采样索引不一致")
    if replay["format"] != FORMAT_VERSION or replay["rules"] != rules.as_dict():
        raise ValueError(f"回放 {entry['id']} 格式或规则与官方 SDK 不一致")
    return replay


def reservations(root, baseline, plans, extra_seeds):
    """Include the actual historical replay layouts as well as SDK-generated reservations."""
    seeds = {
        DIAGNOSTIC_SEED,
        *BEHAVIOR_RESERVED_SEEDS,
        *extra_seeds,
        *baseline["train_maps"],
        *baseline["validation_maps"],
    }
    seeds.update(read_json(root / "suites/boss-specialist.json")["seeds"])
    for plan in plans:
        maps = read_json(plan)["maps"]
        seeds.update(maps["validation"])
        seeds.update(maps["holdout"])
    grids = set()
    manifest = root / "published/datasets/boss-behavior-20261125.json"
    data = read_json(manifest)
    with ZipFile(manifest.with_suffix(".zip")) as archive:
        for match_id in data["development"] + data["blind"]:
            game_map = json.loads(gzip.decompress(archive.read(f"online/replays/{match_id}.json.gz")))["map"]
            seeds.add(game_map["seed"])
            grids.add(layout_hash(game_map))
    grids.update(layout_hash(generate_map(seed).to_dict()) for seed in seeds)
    return seeds, grids


def grouped_entries(entries, excluded_seeds, excluded_grids, *, expand_development=False):
    groups = {split: defaultdict(lambda: defaultdict(lambda: defaultdict(list))) for split in SPLITS}
    by_seed, by_grid, final_by_seed = {}, {}, {}
    seen, skipped = set(), Counter()
    # Import the layout of every archived reserved seed before filtering any split.
    excluded_grids = excluded_grids | {e["grid_sha256"] for e in entries if e["seed"] in excluded_seeds}
    for entry in entries:
        match_id, seed, grid, split = (entry[key] for key in ("id", "seed", "grid_sha256", "split"))
        if match_id in seen:
            raise ValueError(f"回放 {match_id} 重复")
        seen.add(match_id)
        if seed in excluded_seeds or grid in excluded_grids:
            skipped["reserved"] += 1
            continue
        if split == "train" and entry.get("eligible_for_training") is False:
            skipped["collection_excluded"] += 1
            continue
        if by_seed.setdefault(seed, split) != split or by_grid.setdefault(grid, split) != split:
            raise ValueError(f"地图种子 {seed} 或布局 {grid} 跨分组")
        if entry["version_id"] is None:
            skipped["unknown_strategy"] += 1
            continue
        # Replenish development after excluding the previous experiments' reserved maps.
        if expand_development and split == "train" and int(grid[8:16], 16) % 10 == 0:
            split = "development"
        if final_by_seed.setdefault(seed, split) != split:
            raise ValueError(f"重分组后地图种子 {seed} 跨分组")
        groups[split][grid][entry["version_id"]][entry["target_side"]].append(entry)
    return groups, skipped


def replay_samples(replay, target_side):
    """Compute predictor inputs from the decision frame and its predecessor only."""
    frames = replay["frames"]
    if len(frames) != replay["result"]["ticks"] + 1 or any(i != frame[0] for i, frame in enumerate(frames)):
        raise ValueError(f"回放 {replay['match']['id']} 帧缺失或不连续")
    enemy = SIDES.index(target_side)
    side = 1 - enemy
    walls = GameMap.from_dict(replay["map"]).grid
    rows, fired = [], []
    for tick in range(1, len(frames) - 1):
        frame = frames[tick]
        own, other = frame[side + 1], frame[enemy + 1]
        x, y = other[:2]
        if own[5] * rules.DT > 0.5 or not own[4] or not other[4] or not clear_shot(walls, *own[:2], x, y):
            continue
        horizon = min(16, flight_ticks(hypot(own[0] - x, own[1] - y)))
        target_tick = tick + horizon
        if target_tick >= len(frames):
            continue
        target = frames[target_tick][enemy + 1]
        if not target[4]:
            continue
        prev = frames[tick - 1][enemy + 1]
        obs = replay_observation(frame, side)
        features, c, s = state_vector(obs, x - prev[0], y - prev[1], horizon)
        dx, dy = target[0] - x, target[1] - y
        sx, sy = own[0] - x, own[1] - y
        rows.append(
            (
                *features,
                dx * c + dy * s,
                dy * c - dx * s,
                horizon * MOVE_STEP,
                sx * c + sy * s,
                sy * c - sx * s,
            )
        )
        fired.append(frames[tick + 1][side + 1][10] > own[10])
    if not rows:
        return None
    array = np.asarray(rows)
    return Samples(array[:, :-5], array[:, -5:-3], array[:, -3:-2], array[:, -2:], np.asarray(fired))


def load_games(root, split, skipped):
    games = {}
    for grid, strategies in split.items():
        for strategy, sides in strategies.items():
            if set(sides) != set(SIDES):
                skipped["missing_paired_side"] += sum(len(entries) for entries in sides.values())
                continue
            pair = {}
            for side, entries in sides.items():
                pair[side] = []
                for entry in entries:
                    replay = archived_replay(root, entry)
                    samples = replay_samples(replay, side)
                    if samples is None:
                        skipped["no_samples"] += 1
                        continue
                    pair[side].append((entry, samples))
            if any(not rows for rows in pair.values()):
                skipped["missing_paired_side"] += sum(len(rows) for rows in pair.values())
                continue
            for side, rows in pair.items():
                games[grid, strategy, side] = rows
    return games


def weighted_games(games, *, fired_only=False):
    strata = defaultdict(lambda: defaultdict(dict))
    for (grid, strategy, side), rows in games.items():
        selected = [(entry, samples) for entry, samples in rows if not fired_only or samples.fired.any()]
        if selected:
            strata[grid][strategy][side] = selected
    for strategies in strata.values():
        for sides in strategies.values():
            for rows in sides.values():
                mass = 1 / len(strategies) / len(sides) / len(rows)
                for entry, samples in rows:
                    yield entry, samples, mass


def fit_candidates(train, baseline):
    projection, bias, baseline_coefs = (np.asarray(baseline[key]) for key in ("projection", "bias", "coef"))
    mats, labels, weights = [], [], []
    for _, samples, mass in weighted_games(train):
        mats.append(basis(samples.features, projection, bias) * samples.reach)
        labels.append(samples.displacement)
        weights.append(np.full(len(samples.features), mass / len(samples.features)))
    x, y, w = map(np.concatenate, (mats, labels, weights))
    normal = x.T @ (w[:, None] * x)
    target = x.T @ (w[:, None] * y)
    candidates = {}
    for ridge in (0.1, 1.0, 5.0):
        coefs = np.linalg.solve(normal + ridge * np.eye(len(normal)), target + ridge * baseline_coefs)
        candidates[f"ridge_{ridge:g}"] = coefs
    return projection, bias, {"v4": baseline_coefs, **candidates}


def prediction_metrics(displacement, shooter, predicted):
    distance = np.linalg.norm(predicted - displacement, axis=1)
    aim = predicted - shooter
    target = displacement - shooter
    lateral = np.abs(aim[:, 0] * target[:, 1] - aim[:, 1] * target[:, 0]) / np.linalg.norm(aim, axis=1)
    return np.column_stack((distance, distance <= rules.HIT_RADIUS, lateral, lateral <= rules.HIT_RADIUS))


METRICS = ("mae_cells", "within_hit_radius", "lateral_mae_cells", "within_lateral_hit_radius")


def metrics(games, projection, bias, models):
    # Average within game, then repeats, sides, strategies and layouts in that order.
    accum = defaultdict(lambda: defaultdict(list))
    for subset in ("near_ready", "fired"):
        for entry, samples, mass in weighted_games(games, fired_only=subset == "fired"):
            selected = samples.fired if subset == "fired" else slice(None)
            reach = samples.reach[selected]
            af = basis(samples.features[selected], projection, bias)
            for name, coefs in models.items():
                prediction = af @ coefs
                prediction *= reach / np.maximum(1.0, np.linalg.norm(prediction, axis=1, keepdims=True))
                values = prediction_metrics(
                    samples.displacement[selected], samples.shooter[selected], prediction
                )
                accum[name, subset][entry["grid_sha256"]].append((values.mean(axis=0), mass, len(values)))
    result = {name: {} for name in models}
    for (name, subset), groups in accum.items():
        by_grid = {}
        for grid, rows in groups.items():
            values, weights, counts = zip(*rows, strict=True)
            mean = np.average(values, axis=0, weights=weights)
            by_grid[grid] = dict(zip(METRICS, mean.tolist(), strict=True)) | {"samples": sum(counts)}
        average = np.asarray([[row[key] for key in METRICS] for row in by_grid.values()]).mean(axis=0)
        result[name][subset] = dict(zip(METRICS, average.tolist(), strict=True)) | {
            "samples": sum(row["samples"] for row in by_grid.values()),
            "maps": len(by_grid),
            "by_grid": by_grid,
        }
    return result


def split_sources(games):
    return [
        {
            "id": entry["id"],
            "seed": entry["seed"],
            "grid_sha256": entry["grid_sha256"],
            "target_side": entry["target_side"],
            "version_id": entry["version_id"],
            "sha256": entry["sha256"],
            "samples": len(samples.features),
            "fired_samples": int(samples.fired.sum()),
        }
        for entry, samples, _ in weighted_games(games)
    ]


def comparison(candidate, reference):
    report = {}
    for subset in ("near_ready", "fired"):
        diffs = {
            key: [
                cand[key] - reference[subset]["by_grid"][grid][key]
                for grid, cand in candidate[subset]["by_grid"].items()
            ]
            for key in METRICS
        }
        n = len(diffs[METRICS[0]])
        indices = np.random.default_rng(20261125).integers(n, size=(4000, n))
        report[subset] = {
            key: {
                "delta": float(np.mean(values)),
                "map_bootstrap_95": np.quantile(
                    np.asarray(values)[indices].mean(axis=1), [0.025, 0.975]
                ).tolist(),
            }
            for key, values in diffs.items()
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="tank sample export 的终极档清单")
    parser.add_argument("--baseline", type=Path, default=Path("strategies/boss-specialist/forecast.json"))
    parser.add_argument("--exclude-plan", type=Path, action="append", help="原 CEM 作业的 plan.json，可重复")
    parser.add_argument("--exclude-seed", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True, help="写出 CEM 可加载的 forecast.json")
    parser.add_argument("--report", type=Path, required=True, help="写出逐地图与逐回放报告")
    args = parser.parse_args()
    root = project_root()
    baseline = read_json(args.baseline)
    plans = args.exclude_plan or [
        root / ".local/training/jobs" / job / "output/plan.json" for job in PRIOR_JOBS
    ]
    excluded_seeds, excluded_grids = reservations(root, baseline, plans, args.exclude_seed)
    groups, skipped = grouped_entries(
        read_json(args.manifest), excluded_seeds, excluded_grids, expand_development=True
    )
    games = {split: load_games(root, groups[split], skipped) for split in SPLITS}
    if not all(games.values()):
        raise ValueError("三个地图分组均须有可用的成对样本")
    projection, bias, candidates = fit_candidates(games["train"], baseline)
    development = metrics(games["development"], projection, bias, candidates)
    # Choose by shot-frame lateral error. The blind maps only see the frozen winner and v4.
    chosen = min(
        (name for name in candidates if name != "v4"),
        key=lambda name: (
            development[name]["fired"]["lateral_mae_cells"],
            development[name]["near_ready"]["lateral_mae_cells"],
        ),
    )
    blind = metrics(games["blind"], projection, bias, {name: candidates[name] for name in ("v4", chosen)})
    sources = {split: split_sources(rows) for split, rows in games.items()}
    reference = sha256(args.baseline.read_bytes()).hexdigest()
    model = {
        "projection": projection.tolist(),
        "bias": bias.tolist(),
        "coef": candidates[chosen].tolist(),
        "source": "official_boss_practice_replays",
        "candidate": chosen,
        "reference_sha256": reference,
    }
    report = {
        "rules": rules.as_dict()["version"],
        "candidate": chosen,
        "reference_sha256": reference,
        "manifest_sha256": sha256(args.manifest.read_bytes()).hexdigest(),
        "development_expansion": "indexed train layouts with int(grid_sha256[8:16], 16) % 10 == 0",
        "excluded_seeds": sorted(excluded_seeds),
        "excluded_grids": sorted(excluded_grids),
        "skipped_replays": skipped,
        "sources": sources,
        "development": development,
        "blind": blind,
        "blind_delta": comparison(blind[chosen], blind["v4"]),
    }
    write_json(args.output, model)
    write_json(args.report, report)
    print(
        json.dumps(
            {
                "candidate": chosen,
                "sources": {split: len(rows) for split, rows in sources.items()},
                "blind_delta": report["blind_delta"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
