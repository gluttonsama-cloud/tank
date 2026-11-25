"""Replay action retrieval for launch and shrinking-zone movement."""
from math import hypot

import numpy as np
from seedcup.game import rules

if __package__:
    from .boss_sparring_policy import BossSparring, decode, predict
else:
    from boss_sparring_policy import BossSparring, decode, predict


def phase_context(features, phase):
    # ManeuverFeatures uses the same mirrored, tank-relative axes on both sides.
    indices = ([29, 30, 32, 33, 19, 20, 27] if phase == "opening"
               else [32, 33, 15, 16, 17, 19, 20, 7, 8, 9, 14, 3])
    parts = features[..., indices].copy()
    if phase == "zone":
        parts[..., 4] = np.clip(parts[..., 4] * 5, -2, 2)  # Four cells per unit of margin.
    return np.concatenate((parts, features[..., 35:47]), axis=-1)


def near_zone_edge(tick, margin):
    # Project only the circle radius, without assuming how the tank will drive.
    shrink = rules.zone_radius(tick) - rules.zone_radius(tick + 2 * rules.TICK_HZ)
    return tick >= rules.ZONE_START_TICKS and margin - shrink < 2.5


def fit_memory(games, phases):
    memory = {}
    for phase in phases:
        contexts, actions = [], []
        for features, labels in games:
            ticks = np.arange(len(features))
            if phase == "opening":
                near_spawn = np.hypot((features[:, 1] - features[0, 1]) * rules.W,
                                      (features[:, 2] - features[0, 2]) * rules.H) < 1.5
                mask = (ticks < 2 * rules.TICK_HZ) & near_spawn
            else:
                mask = [near_zone_edge(int(t), f[17] * 20) for t, f in enumerate(features)]
            valid = np.asarray(mask) & np.isfinite(labels[:, 0])
            contexts.append(phase_context(features[valid], phase))
            actions.append(labels[valid, :2])
        memory[phase + "_context"] = np.concatenate(contexts)
        memory[phase + "_action"] = np.concatenate(actions)
        if not len(memory[phase + "_action"]):
            raise ValueError(f"训练回放中没有 {phase} 的可识别移动动作")
    return memory


def retrieved_action(weights, phase, features):
    offsets = weights[phase + "_context"] - phase_context(features, phase)
    distances = np.einsum("ij,ij->i", offsets, offsets)
    k = min(9, len(distances))
    nearest = np.argpartition(distances, k - 1)[:k]
    actions = weights[phase + "_action"][nearest]
    votes = 1 / (.05 + distances[nearest])
    # Vote for a joint maneuver so opposite directions do not average into a stop.
    speed = np.where(abs(actions[:, 0]) < .08, 0, np.sign(actions[:, 0])).astype(int)
    steering = np.where(abs(actions[:, 1]) < .2, 0, np.sign(actions[:, 1])).astype(int)
    modes = (speed + 1) * 3 + steering + 1
    chosen = modes == np.bincount(modes, weights=votes, minlength=9).argmax()
    return np.average(actions[chosen], axis=0, weights=votes[chosen])


class MovementSparring(BossSparring):
    def on_match_start(self, info):
        super().on_match_start(info)
        self.spawn = None
        self.launched = False
        self.recover_launch = False

    def decide(self, obs):
        features = self.features(obs)
        forward, turn, fire = decode(features, predict(features, self.weights), self.weights)
        tick, me = obs["frame"], obs["self"]
        if self.spawn is None:
            self.spawn = me["x"], me["y"]
        displacement = hypot(me["x"] - self.spawn[0], me["y"] - self.spawn[1])
        self.launched |= displacement > .05
        if rules.TICK_HZ // 2 <= tick < 2 * rules.TICK_HZ and not self.launched:
            self.recover_launch = True
        phase = None
        if self.recover_launch and tick < 2 * rules.TICK_HZ and displacement < 1.5:
            phase = "opening"
        elif ("zone_action" in self.weights and near_zone_edge(tick, features[17] * 20)
              and forward * features[33] <= .1):
            phase = "zone"
        if phase:
            forward, steering = retrieved_action(self.weights, phase, features)
            if not fire:
                turn = steering
        action = {"forward": float(forward), "turn": float(-turn if self.features.flip else turn), "fire": bool(fire)}
        return self.evade(obs, action) if self.evade else action
