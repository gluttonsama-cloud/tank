"""Score local SeedCup policies by observed hits, remaining health and SDK result."""

import json
import random
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path
from statistics import fmean

import numpy as np
from seedcup.game import SIDES

from tank.simulation import run_match
from tank.workspace import project_root, read_json, write_json

if __package__:
    from .boss_local_policy import DEFAULT, NAMES, make_local_policy
    from .boss_specialist_policy import load_strategy
    from .fit_boss_specialist import (
        Settings,
        evaluate,
        freeze_snapshots,
        map_splits,
        population,
        refit,
        worker_pool,
    )
else:
    from boss_local_policy import DEFAULT, NAMES, make_local_policy
    from boss_specialist_policy import load_strategy
    from fit_boss_specialist import (
        Settings,
        evaluate,
        freeze_snapshots,
        map_splits,
        population,
        refit,
        worker_pool,
    )


SOURCE = Path(__file__).resolve()
POLICY_FILES = (
    "boss_local_policy.py",
    "boss_local_world.py",
    "boss_local_aim.py",
    "boss_specialist_policy.py",
)


def play_game(base, opponent, task):
    name, params, seed, side = task
    random.seed(seed)
    np.random.seed(seed)
    policy = base if params is None else make_local_policy(base, opponent, params)
    players = [policy(), opponent()]
    if side:
        players.reverse()
    r = run_match(*players, seed, 0)["result"]
    own, foe = r["stats"][side], r["stats"][1 - side]
    return {
        "candidate": name,
        "seed": seed,
        "side": SIDES[side],
        "score": r["scores"][side]["total"],
        "win": r["winner"] == side,
        "shots": own["shots"],
        "hits": own["hits"],
        "hp": own["hp"],
        "damaged": foe["hits"],
        "shell_damage_taken": foe["damage"],
        "healed": own["healed"],
        "zone_damage": own["zone_damage"],
        "clean": foe["hits"] == 0 and own["zone_damage"] == 0,
        "kill": r["winner"] == side and r["reason"] == "kill",
        "seconds": r["seconds"],
    }


def run_worker(folder):
    wire = sys.stdout
    with redirect_stdout(sys.stderr):
        base = load_strategy(folder / "snapshots/base")
        opponent = load_strategy(folder / "snapshots/opponent")
        for line in sys.stdin:
            row = play_game(base, opponent, json.loads(line))
            print(json.dumps(row, allow_nan=False), file=wire, flush=True)


def summary(games):
    shots, hits = sum(g["shots"] for g in games), sum(g["hits"] for g in games)
    killed = [g["seconds"] for g in games if g["kill"]]
    score, hp = (fmean(g[k] for g in games) for k in ("score", "hp"))
    hit_rate = hits / shots if shots else 0.0
    hits_per_game = hits / len(games)
    wins = sum(g["win"] for g in games)
    # Aggregate the actual discharges. An inactive tank must not benefit from high precision.
    utility = (
        score
        + 0.40 * hp
        + 20 * wins / len(games)
        + 2 * hits_per_game
        - 650 * max(0.0, 0.945 - hit_rate)
        - 22 * max(0.0, 8.3 - hits_per_game)
    )
    return {
        "games": len(games),
        "shots": shots,
        "hits": hits,
        "hit_rate": hit_rate,
        "hits_per_game": hits_per_game,
        "score": score,
        "hp": hp,
        "wins": wins,
        "hits_taken": fmean(g["damaged"] for g in games),
        "zone_damage": fmean(g["zone_damage"] for g in games),
        "clean": sum(g["clean"] for g in games),
        "gun_wins": len(killed),
        "gun_win_seconds": fmean(killed) if killed else None,
        "utility": utility,
    }


def scores(rows, names):
    return {name: summary([g for g in rows if g["candidate"] == name]) for name in names}


def meets_target(candidate, baseline):
    return (
        candidate["hit_rate"] >= 0.90
        and candidate["hits_per_game"] >= 8
        and candidate["wins"] >= baseline["wins"]
        and candidate["hp"] > baseline["hp"]
    )


def train(play, out, cfg, splits, seed):
    rng = np.random.default_rng(seed)
    mean, sigma = np.asarray(DEFAULT), np.full(len(DEFAULT), cfg.initial_sigma)
    incumbent = ("initial", DEFAULT)
    finalists = {incumbent[0]: incumbent[1]}
    for index, maps in enumerate(splits["train"], 1):
        members = population(rng, index, cfg, mean, sigma, incumbent)
        rows = evaluate(play, out, f"train-{index:02d}", members, maps)
        metrics = scores(rows, members)
        ranked = sorted(members, key=lambda k: metrics[k]["utility"], reverse=True)
        incumbent = (ranked[0], members[ranked[0]])
        finalists.update({key: members[key] for key in ranked[: cfg.finalists_per_generation]})
        mean, sigma = refit(mean, sigma, [members[k] for k in ranked[: cfg.elites]], cfg)
        write_json(
            out / "generations" / f"{index:02d}.json",
            {"best": incumbent[0], "params": members, "metrics": metrics},
        )
    return finalists


ENTRY = """from pathlib import Path
import json
from boss_specialist_policy import load_strategy
from boss_local_policy import make_local_policy

ROOT = Path(__file__).resolve().parent
Specialist = make_local_policy(load_strategy(ROOT / "base"), load_strategy(ROOT / "replica"),
                               json.loads((ROOT / "weights.json").read_text(encoding="utf-8"))["params"])
"""


def export(folder, key, params, plan, metrics):
    out = folder / "output"
    strategy = out / "strategy"
    strategy.mkdir()
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(folder / "snapshots/base", strategy / "base", ignore=ignore)
    shutil.copytree(folder / "snapshots/opponent", strategy / "replica", ignore=ignore)
    for name in POLICY_FILES:
        shutil.copy2(SOURCE.with_name(name), strategy / name)
    (strategy / "bot.py").write_text(ENTRY, encoding="utf-8", newline="\n")
    write_json(strategy / "submission.json", {"runtime": "python", "entrypoint": "bot.py:Specialist"})
    write_json(strategy / "weights.json", {"param_names": NAMES, "params": params})
    write_json(strategy / "training.json", dict(plan, selected=key, metrics=metrics))
    write_json(
        out / "result.json",
        {
            "weights": "strategy/weights.json",
            "strategy": "strategy",
            "step": metrics["train_episodes"],
            "metrics": metrics,
        },
    )


def main(request_path):
    folder = request_path.parent
    out = folder / "output"
    request = read_json(request_path)
    if request["inputs"]:
        raise ValueError("本地专项使用配置中冻结的基座和陪练")
    cfg = Settings.from_params(request["params"])
    seed = request["config"]["sampling_seed"]
    snapshots = freeze_snapshots(project_root(), folder, request["config"])
    splits = map_splits(seed, cfg)
    train_count = cfg.generations * cfg.population * cfg.maps_per_generation * 2
    plan = {
        "snapshots": snapshots,
        "seed": seed,
        "maps": splits,
        "settings": vars(cfg),
        "param_names": NAMES,
        "source_sha256": {
            n: request["sources"][n] for n in (*POLICY_FILES, SOURCE.name, "fit_boss_specialist.py")
        },
        "method": "model_predictive_control_and_episodic_cem",
        "expected_train_episodes": train_count,
    }
    write_json(out / "plan.json", plan)
    with worker_pool(folder, cfg.workers, SOURCE) as play:
        finalists = train(play, out, cfg, splits, seed)
        rows = evaluate(play, out, "validation", finalists, splits["validation"])
        val = scores(rows, finalists)

        def rank(name):
            r = val[name]
            return (r["hit_rate"] >= 0.90 and r["hits_per_game"] >= 8, r["utility"])

        best = max(finalists, key=rank)
        write_json(out / "selection.json", {"selected": best, "params": finalists, "metrics": val})
        control = {best: finalists[best], "v4": None}
        blind = evaluate(play, out, "holdout", control, splits["holdout"])
    result = scores(blind, control)
    metrics = {
        "selected": best,
        "passed": meets_target(result[best], result["v4"]),
        "train_episodes": train_count,
        "validation_episodes": len(rows),
        "holdout_episodes": len(blind),
        "validation": val[best],
        "holdout": result,
    }
    write_json(out / "holdout.json", result)
    export(folder, best, finalists[best], plan, metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        run_worker(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]))
