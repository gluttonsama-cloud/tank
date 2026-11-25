"""Search a residual policy with official SeedCup matches against a frozen opponent.

The elite refit follows de Boer et al., https://people.smp.uq.edu.au/DirkKroese/ps/aortut.pdf.
As in Salimans et al., https://arxiv.org/abs/1703.03864, independent workers return
episodic scores. Here each generation shares the same SeedCup maps and side assignments.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import isfinite
from pathlib import Path
from queue import SimpleQueue
from statistics import fmean

import numpy as np
from seedcup.game import SIDES, generate_map, rules

from tank import versions
from tank.simulation import run_match
from tank.workspace import project_root, read_json, write_json

if __package__:
    from .boss_specialist_policy import (
        DEFAULT_PARAMS,
        PARAM_NAMES,
        load_strategy,
        make_policy,
    )
else:
    from boss_specialist_policy import (
        DEFAULT_PARAMS,
        PARAM_NAMES,
        load_strategy,
        make_policy,
    )


BASE = "base"
WARM = "warm"
RESIDUAL = Path(__file__).with_name("boss_specialist_policy.py")
FORECAST = Path(__file__).with_name("boss_forecast.py")
SCORE_FIELDS = ("total", "damage", "kill", "survive", "time")
BOT_SOURCE = '''from pathlib import Path
import json

from boss_specialist_policy import load_strategy, make_policy

ROOT = Path(__file__).resolve().parent
PARAMS = json.loads((ROOT / "weights.json").read_text(encoding="utf-8"))["params"]
'''
FORECAST_SOURCE = '''from boss_forecast import make_predictor

PREDICTOR = make_predictor(json.loads((ROOT / "forecast.json").read_text(encoding="utf-8")))
BossSpecialist = make_policy(load_strategy(ROOT / "stock"), PARAMS, predictor=PREDICTOR)
'''
PLAIN_SOURCE = 'BossSpecialist = make_policy(load_strategy(ROOT / "stock"), PARAMS)\n'


@dataclass(frozen=True)
class Settings:
    generations: int = 15
    population: int = 24
    elites: int = 6
    finalists_per_generation: int = 2
    maps_per_generation: int = 4
    validation_maps: int = 12
    holdout_maps: int = 24
    workers: int = min(8, os.cpu_count() or 1)
    initial_sigma: float = 0.55
    sigma_floor: float = 0.12
    smoothing: float = 0.65

    @classmethod
    def from_params(cls, params):
        if unknown := params.keys() - cls.__dataclass_fields__.keys():
            raise ValueError(f"未知训练参数 {', '.join(sorted(unknown))}")
        settings = cls(**params)
        limits = {"generations": 1, "population": 2, "elites": 1, "finalists_per_generation": 1,
                  "maps_per_generation": 1, "validation_maps": 1, "holdout_maps": 1, "workers": 1}
        for name, minimum in limits.items():
            value = getattr(settings, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} 必须为至少 {minimum} 的整数")
        if settings.elites >= settings.population:
            raise ValueError("elites 必须小于 population")
        if settings.finalists_per_generation > settings.population:
            raise ValueError("finalists_per_generation 必须小于或等于 population")
        for name in ("initial_sigma", "sigma_floor", "smoothing"):
            value = getattr(settings, name)
            if type(value) not in (int, float) or not 0 < value <= 1:
                raise ValueError(f"{name} 必须在 (0, 1] 内")
        if settings.sigma_floor > settings.initial_sigma:
            raise ValueError("sigma_floor 必须小于或等于 initial_sigma")
        return settings


def map_splits(seed, settings):
    rng = random.Random(f"{seed}:maps")
    used_seeds, used_grids = set(), set()

    def take(count):
        maps = []
        while len(maps) < count:
            map_seed = rng.randrange(2**31)
            if map_seed in used_seeds:
                continue
            grid = generate_map(map_seed).grid
            if grid not in used_grids:
                used_seeds.add(map_seed)
                used_grids.add(grid)
                maps.append(map_seed)
        return maps

    return {"train": [take(settings.maps_per_generation) for _ in range(settings.generations)],
            "validation": take(settings.validation_maps), "holdout": take(settings.holdout_maps)}


def freeze_snapshots(root, folder, config):
    pool = config["opponent_pool"]
    if len(pool) != 1:
        raise ValueError("opponent_pool 必须只有一个固定陪练快照")
    snapshots = {"base": config["teacher_snapshot"], "opponent": next(iter(pool))}
    for role, snapshot_id in snapshots.items():
        if not snapshot_id or snapshot_id == "baseline":
            raise ValueError(f"{role} 必须使用已创建的 Python 策略快照")
        versions.verify(root, snapshot_id)
        source = versions.snapshot_data(root, snapshot_id)
        copy = folder / "snapshots" / role
        shutil.copytree(source, copy)
        if versions._version_id(versions._file_hashes(copy)) != snapshot_id:
            raise ValueError(f"作业内的 {role} 快照校验失败 {snapshot_id}")
    return snapshots


def training_inputs(request):
    result = {}
    for record in request["inputs"]:
        name = Path(record["path"]).name
        if name not in ("forecast.json", "weights.json"):
            raise ValueError(f"未知训练输入 {name}，可使用 forecast.json 或 weights.json")
        if name in result:
            raise ValueError(f"训练输入重复 {name}")
        result[name] = record
    return result


def load_warm_params(folder, source):
    content = (folder / source["path"]).read_bytes()
    if sha256(content).hexdigest() != source["sha256"]:
        raise ValueError("weights.json 的 SHA-256 与冻结输入记录不一致")
    record = json.loads(content)
    if not isinstance(record, dict) or record.get("param_names") != list(PARAM_NAMES):
        raise ValueError("weights.json 的 param_names 必须与当前策略一致")
    params = record.get("params")
    if not isinstance(params, list) or len(params) != len(PARAM_NAMES):
        raise ValueError(f"weights.json 的 params 必须包含 {len(PARAM_NAMES)} 项")
    for name, value in zip(PARAM_NAMES, params, strict=True):
        if type(value) not in (int, float) or not -1 <= value <= 1 or not isfinite(value):
            raise ValueError(f"weights.json 的参数 {name} 必须为 [-1, 1] 内的有限数值")
    return tuple(float(value) for value in params)


def load_predictor(path):
    if __package__:
        from .boss_forecast import make_predictor
    else:
        from boss_forecast import make_predictor
    return make_predictor(read_json(path))


def population(rng, generation, settings, mean, sigma, incumbent, *, warm_params=None):
    members = {} if incumbent is None else {incumbent[0]: incumbent[1]}
    seen = set(members.values()) | {tuple(DEFAULT_PARAMS)}
    if warm_params is not None:
        seen.add(warm_params)
    while len(members) < settings.population:
        params = tuple(np.clip(rng.normal(mean, sigma), -1, 1).tolist())
        if params in seen:
            continue
        members[f"g{generation:02d}-p{len(members):02d}"] = params
        seen.add(params)
    return members


def refit(mean, sigma, elites, settings):
    elite_vectors = np.asarray(elites)
    weight = settings.smoothing
    return (mean * (1 - weight) + elite_vectors.mean(axis=0) * weight,
            np.maximum(settings.sigma_floor, sigma * (1 - weight) + elite_vectors.std(axis=0) * weight))


def play_episode(base_class, opponent_class, task, predictor=None):
    candidate, params, seed, side = task
    random.seed(seed)
    np.random.seed(seed)
    policy = make_policy(base_class, params) if predictor is None else make_policy(base_class, params, predictor=predictor)
    bots = [policy(), opponent_class()]
    if side == 1:
        bots.reverse()
    # Use the same game number on the two sides to hold the opponent's match RNG fixed.
    result = run_match(*bots, seed, 0)["result"]
    return {"candidate": candidate, "seed": seed, "side": SIDES[side], "game_no": 0,
            "win": None if result["winner"] == -1 else result["winner"] == side,
            "ticks": result["ticks"], "reason": result["reason"],
            "score": result["scores"][side], "opponent_score": result["scores"][1 - side],
            "stats": result["stats"][side], "opponent_stats": result["stats"][1 - side]}


def run_worker(folder):
    wire = sys.stdout
    with redirect_stdout(sys.stderr):
        base = load_strategy(folder / "snapshots/base")
        opponent = load_strategy(folder / "snapshots/opponent")
        forecast = training_inputs(read_json(folder / "request.json")).get("forecast.json")
        predictor = load_predictor(folder / forecast["path"]) if forecast is not None else None
        for line in sys.stdin:
            result = play_episode(base, opponent, json.loads(line), predictor)
            print(json.dumps(result, allow_nan=False), file=wire, flush=True)


@contextmanager
def worker_pool(folder, count):
    children = SimpleQueue()
    processes = []
    thread = threading.local()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")

    def bind():
        thread.process = children.get_nowait()

    def play(task):
        process = thread.process
        process.stdin.write(json.dumps(task) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"对局 worker {process.pid} 终止，见 worker.log")
        return json.loads(line)

    executor = ThreadPoolExecutor(max_workers=count, initializer=bind)
    completed = False
    try:
        for _ in range(count):
            process = subprocess.Popen(
                [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", str(folder)],
                cwd=folder, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            processes.append(process)
            children.put(process)
        yield lambda tasks: executor.map(play, tasks)
        completed = True
    finally:
        if not completed:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
        executor.shutdown(wait=True, cancel_futures=True)
        for process in processes:
            process.stdin.close()
            process.stdout.close()
            process.wait()


def describe_scores(rows):
    def average(field):
        return {key: fmean(row[field][key] for row in rows) for key in SCORE_FIELDS}

    def accuracy(field):
        shots = sum(row[field]["shots"] for row in rows)
        hits = sum(row[field]["hits"] for row in rows)
        return {"shots": shots, "hits": hits, "hit_rate": hits / shots if shots else None}

    score = average("score")
    opponent = average("opponent_score")
    return {"episodes": len(rows), "score": score, "opponent_score": opponent,
            "margin": score["total"] - opponent["total"], "wins": sum(row["win"] is True for row in rows),
            "losses": sum(row["win"] is False for row in rows), "draws": sum(row["win"] is None for row in rows),
            "firing": accuracy("stats"), "opponent_firing": accuracy("opponent_stats")}


def report(rows, *, by_map=False):
    candidates = defaultdict(list)
    for row in rows:
        candidates[row["candidate"]].append(row)
    stats = {candidate: describe_scores(games) for candidate, games in candidates.items()}
    if by_map:
        for candidate, games in candidates.items():
            maps = defaultdict(list)
            for row in games:
                maps[row["seed"]].append(row)
            stats[candidate]["by_map"] = {seed: describe_scores(pair) for seed, pair in maps.items()}
    return stats


def evaluate(play, output, stage, candidates, seeds):
    tasks = [(candidate, params, seed, side)
             for seed in seeds for side in range(2) for candidate, params in candidates.items()]
    started = time.monotonic()
    print(json.dumps({"phase": stage, "completed": 0, "episodes": len(tasks)}), flush=True)
    rows = []
    for completed, row in enumerate(play(tasks), start=1):
        rows.append(row)
        if completed % 64 == 0 and completed < len(tasks):
            print(json.dumps({"phase": stage, "completed": completed, "episodes": len(tasks)}), flush=True)
    write_json(output / "episodes" / f"{stage}.json", rows)
    print(json.dumps({"phase": stage, "completed": len(rows), "episodes": len(tasks),
                      "seconds": round(time.monotonic() - started, 1)}), flush=True)
    return rows


def run_search(play, output, splits, seed, settings, warm_params=None):
    rng = np.random.default_rng(seed % (2**64))
    mean = np.asarray(DEFAULT_PARAMS if warm_params is None else warm_params, dtype=float)
    sigma = np.full_like(mean, settings.initial_sigma)
    incumbent = None if warm_params is None else (WARM, warm_params)
    finalists = {} if warm_params is None else {WARM: warm_params}
    history = []
    for generation, seeds in enumerate(splits, start=1):
        members = population(rng, generation, settings, mean, sigma, incumbent, warm_params=warm_params)
        rows = evaluate(play, output, f"train-{generation:02d}", members, seeds)
        scores = report(rows)
        ranked = sorted(members, key=lambda key: -scores[key]["score"]["total"])
        for name in ranked[:settings.finalists_per_generation]:
            if members[name] not in finalists.values():
                finalists[name] = members[name]
        incumbent = (ranked[0], members[ranked[0]])
        mean, sigma = refit(mean, sigma, [members[key] for key in ranked[:settings.elites]], settings)
        entry = {"generation": generation, "episodes": len(rows), "mean": mean.tolist(), "sigma": sigma.tolist(),
                 "candidates": [{"id": key, "params": members[key], "metrics": scores[key]} for key in ranked]}
        history.append(entry)
        write_json(output / "generations" / f"{generation:02d}.json", entry)
    return finalists, history


def paired_delta(best, base):
    return fmean(game["score"]["total"] - base["by_map"][seed]["score"]["total"]
                 for seed, game in best["by_map"].items())


def compare_candidates(play, output, stage, candidates, seeds):
    # When the supplied warm policy is zero, it and the base share their official episodes.
    unique, aliases = {}, {}
    for name, vector in candidates.items():
        aliases[name] = unique.setdefault(tuple(vector), name)
    rows = evaluate(play, output, stage, {name: vector for vector, name in unique.items()}, seeds)
    scores = report(rows, by_map=True)
    return {name: scores[primary] for name, primary in aliases.items()}, len(rows)


def rank_finalists(selection):
    return sorted((key for key in selection if key != BASE),
                  key=lambda key: (-selection[key]["score"]["total"], key != WARM))


def export_policy(folder, output, plan, selected, params, metrics, forecast):
    strategy = output / "strategy"
    strategy.mkdir()
    shutil.copytree(folder / "snapshots/base", strategy / "stock")
    shutil.copyfile(RESIDUAL, strategy / RESIDUAL.name)
    if forecast is not None:
        shutil.copyfile(FORECAST, strategy / FORECAST.name)
        shutil.copyfile(folder / forecast["path"], strategy / "forecast.json")
    source = BOT_SOURCE + (FORECAST_SOURCE if forecast is not None else PLAIN_SOURCE)
    (strategy / "bot.py").write_text(source, encoding="utf-8", newline="\n")
    write_json(strategy / "submission.json", {"runtime": "python", "entrypoint": "bot.py:BossSpecialist"})
    write_json(strategy / "weights.json", {"param_names": PARAM_NAMES, "params": params})
    write_json(strategy / "training.json", plan | {"selected": selected, "metrics": metrics})
    write_json(output / "result.json", {"weights": "strategy/weights.json", "strategy": "strategy",
                                       "step": plan["settings"]["generations"], "metrics": metrics})


def main(request_path):
    folder = request_path.parent
    request = read_json(request_path)
    output = folder / request["output"]
    settings = Settings.from_params(request["params"])
    inputs = training_inputs(request)
    forecast = inputs.get("forecast.json")
    warm = inputs.get("weights.json")
    warm_params = load_warm_params(folder, warm) if warm is not None else None
    source_names = [Path(__file__).name, RESIDUAL.name]
    if forecast is not None:
        load_predictor(folder / forecast["path"])
        source_names.append(FORECAST.name)
    snapshots = freeze_snapshots(project_root(), folder, request["config"])
    seed = request["config"]["sampling_seed"]
    splits = map_splits(seed, settings)
    plan = {"schema_version": 1, "method": "cem", "objective": "mean_official_own_total",
            "rules": rules.as_dict()["version"], "runtime": request["runtime"], "config_id": request["config_id"],
            "seed": seed, "snapshots": snapshots, "settings": asdict(settings), "maps": splits,
            "source_sha256": {name: request["sources"][name] for name in source_names},
            "expected_train_episodes": settings.generations * settings.population * settings.maps_per_generation * 2}
    if forecast is not None:
        plan["forecast_sha256"] = forecast["sha256"]
    if warm is not None:
        plan["warm_start_sha256"] = warm["sha256"]
    write_json(output / "plan.json", plan)
    with worker_pool(folder, settings.workers) as play:
        finalists, history = run_search(play, output, splits["train"], seed, settings, warm_params)
        finalists[BASE] = DEFAULT_PARAMS
        selection, validation_episodes = compare_candidates(play, output, "validation", finalists, splits["validation"])
        rank = rank_finalists(selection)
        selected = rank[0]
        write_json(output / "selection.json", {"selected": selected, "ranking": rank,
                                               "params": finalists, "metrics": selection})
        control = {WARM: warm_params} if warm_params is not None else {}
        control.update({selected: finalists[selected], BASE: DEFAULT_PARAMS})
        holdout, holdout_episodes = compare_candidates(play, output, "holdout", control, splits["holdout"])
    holdout_record = {"selected": selected, "metrics": holdout}
    if warm_params is not None:
        holdout_record["warm_paired_score_delta"] = paired_delta(holdout[WARM], holdout[BASE])
    write_json(output / "holdout.json", holdout_record)
    metrics = {"selected": selected, "train_episodes": sum(entry["episodes"] for entry in history),
               "validation_episodes": validation_episodes, "holdout_episodes": holdout_episodes}
    for phase, rows in (("validation", selection), ("holdout", holdout)):
        metrics[phase] = {
            "candidate": {key: value for key, value in rows[selected].items() if key != "by_map"},
            "base": {key: value for key, value in rows[BASE].items() if key != "by_map"},
            "paired_score_delta": paired_delta(rows[selected], rows[BASE]),
        }
        if warm_params is not None:
            metrics[phase]["warm"] = {key: value for key, value in rows[WARM].items() if key != "by_map"}
            metrics[phase]["warm_paired_score_delta"] = paired_delta(rows[WARM], rows[BASE])
    export_policy(folder, output, plan, selected, finalists[selected], metrics, forecast)
    print(json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        run_worker(Path(sys.argv[2]))
    else:
        main(Path(sys.argv[1]).resolve())
