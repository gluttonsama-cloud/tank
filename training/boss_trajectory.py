"""Use a replay-trained motion forecast to rank the current SDK maneuvers."""
from pathlib import Path

import numpy as np
from seedcup.game import rules

if __package__:
    from .boss_forecast import basis
    from .boss_movement import MovementSparring
    from .boss_sparring_policy import TURN_LOWER, TURN_UPPER, decode, predict
else:
    from boss_forecast import basis
    from boss_movement import MovementSparring
    from boss_sparring_policy import TURN_LOWER, TURN_UPPER, decode, predict


HORIZONS = (3, rules.TICK_HZ, 2 * rules.TICK_HZ)
REACH = np.asarray(HORIZONS, dtype=np.float32) * (rules.SPEED * rules.DT)
TURN_STEP = rules.TURN_RATE * rules.DT
COMMITMENT = np.minimum(np.arange(1, HORIZONS[-1] + 1), HORIZONS[0])
DISPLACEMENT_WEIGHTS = np.array((1., .8, .6), dtype=np.float32)
ONSET_WEIGHTS = np.array((1., .25, .05), dtype=np.float32)
REPLAY_STEERING_WEIGHT = 1.5


def forecast(features, weights):
    x = np.clip((features - weights["mean"]) / weights["scale"], -5, 5)
    result = basis(x, weights["projection"], weights["bias"]) @ weights["coef"]
    positions = result[..., :6].reshape(*result.shape[:-1], len(HORIZONS), 2)
    positions /= np.maximum(1., np.linalg.norm(positions, axis=-1, keepdims=True))
    onset = np.clip(result[..., 6:], 0., 1.)
    total = onset.sum(axis=-1, keepdims=True)
    onset /= np.maximum(1., total)
    onset[..., 0] += np.maximum(0., 1. - total[..., 0])
    return positions, onset


def maneuver_displacements(speeds, turns):
    # SDK movement turns before advancing. Hold steering for three ticks, then coast.
    headings = np.deg2rad(turns[..., None] * (TURN_STEP * COMMITMENT))
    dx = np.cumsum(np.cos(headings), axis=-1)[..., np.asarray(HORIZONS) - 1] / HORIZONS
    dy = np.cumsum(np.sin(headings), axis=-1)[..., np.asarray(HORIZONS) - 1] / HORIZONS
    return speeds[:, None, None, None] * np.stack((dx, dy), axis=-1)[..., None, :, :, :]


def rank_maneuvers(scores, weights, positions, onset, *, displacement_weights=DISPLACEMENT_WEIGHTS):
    turns = np.clip(weights["turn_centers"] + scores[..., 15:20], TURN_LOWER, TURN_UPPER)
    paths = maneuver_displacements(weights["speeds"], turns)
    displacement_cost = np.sum((paths - positions[..., None, None, :, :]) ** 2, axis=-1)
    path_cost = displacement_cost @ displacement_weights
    turn_preference = (onset[..., 2::2] - onset[..., 1::2]) @ ONSET_WEIGHTS
    turn_gain = turns * turn_preference[..., None]
    original = scores[..., :15].reshape(*scores.shape[:-1], 3, 5)
    ranked = original + REPLAY_STEERING_WEIGHT * (turn_gain[..., None, :] - path_cost)
    result = scores.copy()
    result[..., :15] = ranked.reshape(*scores.shape[:-1], 15)
    return result


class TrajectorySparring(MovementSparring):
    def __init__(self):
        super().__init__()
        with np.load(Path(__file__).with_name("trajectory.npz"), allow_pickle=False) as weights:
            self.trajectory = dict(weights)

    def decide(self, obs):
        if obs["frame"] < 2 * rules.TICK_HZ:
            return super().decide(obs)
        features = self.features(obs)
        scores = predict(features, self.weights)
        forward, turn, fire = decode(features, scores, self.weights)
        if not fire:
            positions, onset = forecast(features, self.trajectory)
            scores = rank_maneuvers(scores, self.weights, positions, onset)
            forward, turn, fire = decode(features, scores, self.weights)
        action = {"forward": float(forward), "turn": float(-turn if self.features.flip else turn),
                  "fire": bool(fire)}
        return self.evade(obs, action)
