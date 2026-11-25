"""Fit and test a visible-state motion forecast in local SDK games against a frozen opponent."""

import argparse
import random
from concurrent.futures import ProcessPoolExecutor
from math import hypot
from pathlib import Path

import numpy as np
from seedcup import Bot
from seedcup.game import rules
from seedcup.game.bots import clear_shot, wall_grid

from tank import versions
from tank.simulation import run_match
from tank.workspace import project_root, write_json

if __package__:
    from .boss_forecast import basis, state_vector
    from .boss_specialist_policy import load_strategy
else:
    from boss_forecast import basis, state_vector
    from boss_specialist_policy import load_strategy


def setup(root, base, opponent):
    global agent_factory, opponent_factory

    agent_factory = load_strategy(versions.snapshot_data(root, base))
    opponent_factory = load_strategy(versions.snapshot_data(root, opponent))


class Record(Bot):
    def __init__(self, bot):
        self.bot = bot
        self.xy = []
        self.samples = []

    def on_match_start(self, info):
        self.walls = wall_grid(info)
        self.bot.on_match_start(info)

    def decide(self, obs):
        me, en = obs["self"], obs["enemy"]
        x, y = en["x"], en["y"]
        tick = obs["frame"]
        if tick and tick % 2 == 0 and me["reload"] <= .5 and clear_shot(self.walls, me["x"], me["y"], x, y):
            horizon = min(16, max(1, round(hypot(me["x"]-x, me["y"]-y)/(rules.SHELL_SPEED*rules.DT))))
            vx, vy = x - self.xy[-1][0], y - self.xy[-1][1]
            feat, c, s = state_vector(obs, vx, vy, horizon)
            self.samples.append((tick, horizon, c, s, feat))
        self.xy.append((x, y))
        return self.bot.decide(obs)

    def on_match_end(self, result):
        self.bot.on_match_end(result)


def rollout(case):
    seed, side = case
    recorder = Record(agent_factory())
    bots = [recorder, opponent_factory()]
    if side:
        bots.reverse()
    run_match(*bots, seed, side)
    rows = []
    for tick, horizon, c, s, feat in recorder.samples:
        if tick + horizon >= len(recorder.xy):
            continue
        x0, y0 = recorder.xy[tick]
        x1, y1 = recorder.xy[tick + horizon]
        dx, dy = x1 - x0, y1 - y0
        reach = horizon * rules.SPEED * rules.DT
        rows.append([*feat, (dx*c + dy*s)/reach, (dy*c - dx*s)/reach, reach])
    return rows


def fit(train, val, seed):
    x, y = train[:, :-3], train[:, -3:-1]
    rng = np.random.default_rng(seed)
    projection = rng.normal(0., .65, (x.shape[1], 96))
    bias = rng.uniform(-1.5, 1.5, 96)
    a = basis(x, projection, bias)
    coef = np.linalg.solve(a.T @ a + 5*np.eye(a.shape[1]), a.T @ y)
    x, y, reach = val[:, :-3], val[:, -3:-1], val[:, -1:]
    preds = basis(x, projection, bias) @ coef
    preds /= np.maximum(1., np.linalg.norm(preds, axis=1, keepdims=True))

    def errors(prediction):
        dist = np.linalg.norm((prediction - y)*reach, axis=1)
        return {"mae_cells": float(dist.mean()), "within_hit_radius": float((dist <= rules.HIT_RADIUS).mean())}

    return {"projection": projection.tolist(), "bias": bias.tolist(), "coef": coef.tolist()}, {
        "train_samples": len(train), "validation_samples": len(val), "forecast": errors(preds),
        "constant_velocity": errors(x[:, :2]), "stationary": errors(0.),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026112511)
    args = parser.parse_args()
    root = project_root()
    for version in (args.base, args.opponent):
        versions.verify(root, version)
    maps = random.Random(args.seed).sample(range(1610000000, 1620000000), 60)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=setup,
                             initargs=(root, args.base, args.opponent)) as pool:
        batches = list(pool.map(rollout, [(m, s) for m in maps for s in (0, 1)]))
    train, val = (np.asarray([row for game in group for row in game]) for group in (batches[:96], batches[96:]))
    model, metrics = fit(train, val, args.seed)
    write_json(args.output, model | {"base": args.base, "opponent": args.opponent, "metrics": metrics,
                                     "seed": args.seed, "train_maps": maps[:48], "validation_maps": maps[48:]})
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
