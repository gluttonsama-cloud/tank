"""Inference for the NumPy demonstration model."""

import json
from math import atan2, cos, hypot, radians, sin
from pathlib import Path

import numpy as np
from seedcup import Bot


def features(observation):
    me = observation["self"]
    enemy = observation["enemy"]
    dx, dy = enemy["x"] - me["x"], enemy["y"] - me["y"]
    angle = atan2(dy, dx) - radians(me["heading"])
    return (1, dx / 32, dy / 24, sin(angle), cos(angle), hypot(dx, dy) / 40,
            me["hp"] / 100, enemy["hp"] / 100, me["reload"], float(observation["collision"]))


class DemoPolicy(Bot):
    def __init__(self):
        folder = Path(__file__).resolve().parent
        self.weights = np.load(folder / "weights.npy", allow_pickle=False)
        self.threshold = json.loads((folder / "policy.json").read_text(encoding="utf-8"))["fire_threshold"]

    def decide(self, observation):
        forward, turn, fire = np.asarray(features(observation)) @ self.weights
        return {"forward": float(np.clip(forward, -1, 1)), "turn": float(np.clip(turn, -1, 1)),
                "fire": bool(fire >= self.threshold)}
