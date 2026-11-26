"""Score frozen style policies on the two-tier, paired local SeedCup holdout."""

import argparse
import hashlib
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import NamedTuple

import numpy as np
from seedcup import __version__ as sdk_version
from seedcup.game import Game, generate_map, parse_action, rules
from seedcup.game.bots import make_bot
from seedcup.game.runner import match_info
from seedcup.server.evaluation_scoring import summarize

from tank import versions
from tank.workspace import project_root, read_json, write_json

if __package__:
    from .boss_specialist_policy import load_strategy
else:
    from boss_specialist_policy import load_strategy


class Case(NamedTuple):
    tier: str
    map_seed: int
    side: int
    opponent_seed: int
    contestant_seed: int


TIERS = ("sdk_rookie", "v31_boss_surrogate")
BLAS_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


class LocalBot:
    """Keep the shared Python and NumPy RNGs independent inside a worker."""

    def __init__(self, seed, factory, *args):
        self.python = random.Random(seed).getstate()
        self.numpy = np.random.RandomState(seed % 2**32).get_state()
        self.bot = self.call(factory, *args)

    def call(self, method, *args):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        result = method(*args)
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        return result


def play(task):
    name, folder, surrogate, case = task
    side = case.side
    opponent_side = 1 - side
    game_map = generate_map(case.map_seed)
    game = Game(game_map)
    if case.tier == "sdk_rookie":
        opponent = LocalBot(case.opponent_seed, make_bot, "rookie", case.opponent_seed)
    else:
        opponent = LocalBot(case.opponent_seed, load_strategy(surrogate))
    contestant = LocalBot(case.contestant_seed, load_strategy(folder))
    opponent.call(opponent.bot.on_match_start, match_info(game_map, opponent_side, "参赛队伍"))
    info = match_info(game_map, side, "官方人机")
    del info["map"]["seed"]
    contestant.call(contestant.bot.on_match_start, {"type": "match_start", **info})
    while not game.over:
        own_obs = {"type": "obs", **game.observe(side)}
        own, own_bad = parse_action(contestant.call(contestant.bot.decide, own_obs))
        foe, foe_bad = parse_action(opponent.call(opponent.bot.decide, game.observe(opponent_side)))
        game.violations[side] += own_bad
        game.violations[opponent_side] += foe_bad
        game.step([own, foe] if side == 0 else [foe, own])
    result = game.result_summary()
    contestant.call(contestant.bot.on_match_end, result | {"side": info["side"]})
    row = case._asdict() | {"policy": name, "result": result, "violations": game.violations}
    if name == "router":
        row["router"] = {"active": contestant.bot.active, "switch_frame": contestant.bot.detector.switch_frame}
    return row


def score(row):
    return row["result"]["scores"][row["side"]]["total"]


def team_metrics(rows):
    cents = summarize(rows)
    return {
        "games": len(rows),
        "total": cents["score_cents"] / 100,
        "mean": cents["score_cents"] / (100 * len(rows)),
        "minimum": min(map(score, rows)),
        "wins": sum(r["result"]["winner"] == r["side"] for r in rows),
        "invalid_actions": sum(r["violations"][r["side"]] for r in rows),
        "breakdown": cents["breakdown"],
    }


def map_deltas(rows, first, second):
    scores = {(r["map_seed"], r["side"], r["policy"]): score(r) for r in rows}
    return {
        seed: fmean(scores[seed, side, first] - scores[seed, side, second] for side in (0, 1))
        for seed in sorted({r["map_seed"] for r in rows})
    }


def routing_metrics(rows):
    routed = [r["router"] for r in rows if r["policy"] == "router"]
    frames = [r["switch_frame"] for r in routed if r["switch_frame"] is not None]
    return {"switched_games": len(frames), "games": len(routed),
            "switch_rate": len(frames) / len(routed), "switch_frames": sorted(frames),
            "finished_in_B": sum(r["active"] == "B" for r in routed)}


def paired(deltas):
    rng = np.random.default_rng(830612)
    draws = []
    for layer in deltas:
        values = np.array(list(layer.values()))
        samples = rng.choice(values, size=(10000, len(values)))
        draws.append(samples.sum(axis=1))
    return {
        "mean_delta": fmean(delta for layer in deltas for delta in layer.values()),
        "map_bootstrap_95_percent": np.quantile(
            np.sum(draws, axis=0) / sum(map(len, deltas)), (0.025, 0.975)
        ).tolist(),
        "map_mean_deltas": deltas,
    }


def analyze(rows):
    tiers = {tier: [r for r in rows if r["tier"] == tier] for tier in TIERS}
    names = sorted({r["policy"] for r in rows})
    has_router = "router" in names
    oracle = {}
    result = {"per_tier": {}, "overall": {}}
    for tier, subset in tiers.items():
        metrics = {name: team_metrics([r for r in subset if r["policy"] == name]) for name in names}
        oracle[tier] = max(("A", "B"), key=lambda name: metrics[name]["total"])
        result["per_tier"][tier] = {
            "policies": metrics,
            "map_count": len({r["map_seed"] for r in subset}),
            "fixed_per_tier_oracle": oracle[tier],
            "B_minus_A": paired([map_deltas(subset, "B", "A")]),
        }
        if has_router:
            result["per_tier"][tier]["routing"] = routing_metrics(subset)
            for label, comparator in (("A", "A"), ("B", "B"), ("oracle", oracle[tier])):
                result["per_tier"][tier][f"router_minus_{label}"] = paired([
                    map_deltas(subset, "router", comparator)
                ])
    result["overall"]["policies"] = {name: team_metrics([r for r in rows if r["policy"] == name]) for name in names}
    result["overall"]["fixed_per_tier_oracle"] = team_metrics([r for r in rows if r["policy"] == oracle[r["tier"]]])
    result["overall"]["B_minus_A"] = paired([map_deltas(subset, "B", "A") for subset in tiers.values()])
    if has_router:
        for name in ("A", "B", "oracle"):
            result["overall"][f"router_minus_{name}"] = paired([
                map_deltas(subset, "router", oracle[tier] if name == "oracle" else name)
                for tier, subset in tiers.items()
            ])
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    parser.add_argument("--router", type=Path, help="Frozen router submission directory")
    parser.add_argument("--cases", type=Path, help="Paired game manifest")
    parser.add_argument("--out", type=Path, help="Directory for reference scores and reports")
    parser.add_argument("--router-report", default="with-router.json", help="Filename for the router comparison")
    args = parser.parse_args()
    for name in BLAS_THREADS:
        os.environ[name] = "1"
    if rules.VARIANT:
        raise RuntimeError("The benchmark requires the official SDK rules")
    root = project_root()
    out = args.out or root / ".local/style-switch/local-trials"
    experiment = Path(__file__).resolve().parent / "experiments/style-switch-v1"
    plan_path = args.cases or experiment / "cases.json"
    cases = [Case(**row) for row in read_json(plan_path)]
    map_hashes = {str(seed): hashlib.sha256(bytes(generate_map(seed).grid)).hexdigest()
                  for seed in sorted({case.map_seed for case in cases})}
    if len(set(map_hashes.values())) != len(map_hashes):
        raise ValueError("The holdout contains duplicate layouts")
    ids = read_json(experiment / "pilot.json")["policies"]
    surrogate_id, = read_json(root / "suites/boss-specialist.json")["opponents"]
    for version_id in (*ids.values(), surrogate_id):
        versions.verify(root, version_id)
    surrogate = versions.snapshot_data(root, surrogate_id)
    inputs = {
        "cases_sha256": digest(plan_path), "rules": rules.RULES_VERSION, "sdk": sdk_version,
        "policies": ids, "surrogate_v31": surrogate_id, "map_hashes": map_hashes,
    }
    fixed_path = out / "fixed.json"
    if args.router is None:
        tasks = [(name, versions.snapshot_data(root, version_id), surrogate, case)
                 for name, version_id in ids.items() for case in cases]
        result_path = fixed_path
        rows = []
    else:
        frozen = read_json(fixed_path)
        if frozen["inputs"] != inputs:
            raise ValueError("Fixed-policy results were collected with different benchmark inputs")
        inputs["router"] = versions._version_id(versions._file_hashes(args.router))
        result_path = out / args.router_report
        rows = frozen["games"]
        tasks = [("router", args.router, surrogate, case) for case in cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, row in enumerate(pool.map(play, tasks), start=1):
            rows.append(row)
            if count % 8 == 0 or count == len(tasks):
                print(f"{count}/{len(tasks)}", flush=True)
    summary = analyze(rows)
    write_json(result_path, {"inputs": inputs, "summary": summary, "games": rows})
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
