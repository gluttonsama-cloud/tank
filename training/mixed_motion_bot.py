"""Replay/SDK movement clone. Inference needs only NumPy and the game SDK."""
from pathlib import Path

import numpy as np
from seedcup.game import rules

from boss_movement import MovementSparring
from sdk_motion_policy import MovementFeatures


def predict_movement(features, forest):
    nodes = forest["roots"].copy()
    for _ in range(int(forest["depth"])):
        right = features[forest["feature"][nodes]] > forest["threshold"][nodes]
        nodes = np.where(right, forest["right"][nodes], forest["left"][nodes])
    votes = forest["value"][nodes].reshape(2, -1, 6).mean(axis=1)
    modes = votes[:, :3].argmax(axis=1)
    axes = np.arange(2)
    return np.clip(votes[axes, modes + 3] / votes[axes, modes], -1, 1)


class MixedMotionSparring(MovementSparring):
    def __init__(self):
        super().__init__()
        with np.load(Path(__file__).with_name("mixed_motion.npz"), allow_pickle=False) as archive:
            self.forest = dict(archive)

    def on_match_start(self, info):
        super().on_match_start(info)
        self.motion_features = MovementFeatures(info)
        self.previous_obs = None

    def decide(self, obs):
        action = super().decide(obs)
        previous = self.previous_obs
        self.previous_obs = obs
        if obs["frame"] < 2 * rules.TICK_HZ:
            return action
        features = self.motion_features.features(obs, previous)
        forward, turn = predict_movement(features, self.forest)
        action["forward"] = float(forward)
        if not action["fire"]:
            action["turn"] = float(turn)
        return action
