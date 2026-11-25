"""SDK observations and NumPy inference for the replay-trained sparring bot."""

from collections import deque
from math import atan2, cos, degrees, hypot, radians, sin, tau
from pathlib import Path

import numpy as np
from seedcup import Bot
from seedcup.game import rules
from seedcup.game.bots import clear_shot, mirror_obs, ray_wall, wall_grid


def angle_delta(target, current):
    return (target - current + 180) % 360 - 180


class Features:
    def __init__(self, info):
        self.flip = info["side"] == "blue"
        self.walls = wall_grid(info)
        self.links = {}
        for index, wall in enumerate(self.walls):
            if wall == 1:
                x, y = index % rules.W, index // rules.W
                self.links[x, y] = [(a, b) for a, b in ((x+1, y), (x, y+1), (x-1, y), (x, y-1))
                                    if 0 <= a < rules.W and 0 <= b < rules.H and self.walls[b * rules.W + a] == 1]
        self.distances = {}
        self.history = deque(maxlen=5)

    def path_features(self, x, y, target_x, target_y, h):
        source, goal = (int(x), int(y)), (int(target_x), int(target_y))
        if goal not in self.links:
            goal = min(self.links, key=lambda p: (p[0]+0.5-target_x)**2 + (p[1]+0.5-target_y)**2)
        if goal not in self.distances:
            distances = {goal: 0}
            pending = deque((goal,))
            while pending:
                node = pending.popleft()
                for neighbor in self.links[node]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + 1
                        pending.append(neighbor)
            self.distances[goal] = distances
        distances = self.distances[goal]
        if source == goal or source not in distances:
            direction = atan2(target_y - y, target_x - x) - h
            return sin(direction), cos(direction), 0.0
        step_x, step_y = min(self.links[source], key=distances.__getitem__)
        direction = atan2(step_y + 0.5 - y, step_x + 0.5 - x) - h
        return sin(direction), cos(direction), distances[source] / 40

    def __call__(self, obs):
        if self.flip:
            obs = mirror_obs(obs)
        me, enemy, zone = obs["self"], obs["enemy"], obs["zone"]
        # Match the public replay's precision in both extraction and live inference.
        x, y, ex, ey = (round(v, 2) for v in (me["x"], me["y"], enemy["x"], enemy["y"]))
        heading, enemy_heading = round(me["heading"], 1), round(enemy["heading"], 1)
        h = radians(heading)
        c, s = cos(h), sin(h)

        def relative(vx, vy):
            return c*vx + s*vy, c*vy - s*vx

        current = (x, y, ex, ey, heading, enemy_heading)
        previous = self.history[-1] if self.history else current
        early = self.history[0] if self.history else current
        dx, dy = ex - x, ey - y
        distance = hypot(dx, dy)
        bearing = atan2(dy, dx) - h
        zx, zy = zone["x"] - x, zone["y"] - y
        center = atan2(zy, zx) - h
        frames = max(1, len(self.history))
        evx, evy = (ex - early[2]) / frames, (ey - early[3]) / frames
        flight_ticks = min(14, distance / (rules.SHELL_SPEED * rules.DT))
        ax, ay = ex + evx*flight_ticks, ey + evy*flight_ticks
        visible = clear_shot(self.walls, x, y, ex, ey)
        aim_clear = clear_shot(self.walls, x, y, ax, ay)
        if not aim_clear and visible:
            ax, ay, aim_clear = ex, ey, True
        lead = atan2(ay - y, ax - x) - h
        step = rules.TURN_RATE * rules.DT
        aim = angle_delta(degrees(lead), 0) / step
        aim_margin = degrees(atan2(rules.HIT_RADIUS, max(1, hypot(ax - x, ay - y)))) / step
        self.history.append(current)

        incoming = []
        for shell in obs["shells"]:
            if shell["mine"]:
                continue
            sx, sy = round(shell["x"], 2) - x, round(shell["y"], 2) - y
            vx, vy = round(shell["dx"], 3), round(shell["dy"], 3)
            along = -(sx*vx + sy*vy)
            lateral = sx*vy - sy*vx
            if 0 < along < 14 and abs(lateral) < 3:
                forward, sideways = relative(vx, vy)
                incoming.append((along / 14, lateral / 3, forward, sideways))
        threats = sorted(incoming)[:2]
        threats += [(1.0, 0.0, 0.0, 0.0)] * (2 - len(threats))

        return np.asarray([
            obs["frame"] / rules.GAME_TICKS, x / rules.W, y / rules.H,
            me["hp"] / rules.HP, enemy["hp"] / rules.HP,
            min(5, me["still"]) / 5, float(me["repairing"]), sin(bearing), cos(bearing), distance / 40,
            sin(lead), cos(lead), sin(radians(enemy_heading) - h), cos(radians(enemy_heading) - h),
            float(visible),
            sin(center), cos(center), (zone["radius"] - hypot(zx, zy)) / 20,
            (zone["radius"] - hypot(zone["x"] - ex, zone["y"] - ey)) / 20,
            *relative((x - previous[0]) * 4, (y - previous[1]) * 4),
            *relative((ex - previous[2]) * 4, (ey - previous[3]) * 4),
            *relative((x - early[0]) * 4 / frames, (y - early[1]) * 4 / frames),
            *relative(evx * 4, evy * 4),
            angle_delta(heading, previous[4]) / (rules.TURN_RATE * rules.DT),
            angle_delta(enemy_heading, previous[5]) / (rules.TURN_RATE * rules.DT),
            *self.path_features(x, y, ex, ey, h),
            *self.path_features(x, y, zone["x"], zone["y"], h),
            *(ray_wall(self.walls, x, y, cos(a), sin(a), 4) / 4 for a in (h + k*tau/12 for k in range(12))),
            *threats[0], *threats[1],
            me["reload"] / (rules.RELOAD_TICKS * rules.DT), aim, aim_margin, float(aim_clear),
        ], dtype=np.float32)


def predict(features, weights):
    x = (features - weights["mean"]) / weights["scale"]
    hidden = np.tanh(x @ weights["w1"] + weights["b1"])
    return hidden @ weights["w2"] + weights["b2"]


def decode(features, scores, weights):
    forward = weights["speeds"][np.argmax(scores[..., :3], axis=-1)]
    turn = np.clip(scores[..., 3], -1, 1)
    reload, aim, margin, clear = np.moveaxis(features[..., -4:], -1, 0)
    ticks = np.rint(reload * rules.RELOAD_TICKS)
    aim_turn = np.clip(aim, -1, 1)
    # Late in reload, blend toward the lead predicted from visible enemy movement.
    aim_weight = clear * np.maximum(0, (5 - ticks) / 5)
    wants_fire = (ticks <= 1) & (scores[..., 4] >= 0) & (clear == 1)
    turn = np.where(wants_fire, aim_turn, (1 - aim_weight) * turn + aim_weight * aim_turn)
    fire = wants_fire & (np.abs(aim - aim_turn) <= margin)
    return forward, turn, fire


class BossSparring(Bot):
    def __init__(self):
        with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as weights:
            self.weights = dict(weights)

    def on_match_start(self, info):
        self.features = Features(info)

    def decide(self, obs):
        features = self.features(obs)
        forward, turn, fire = decode(features, predict(features, self.weights), self.weights)
        return {"forward": float(forward), "turn": float(-turn if self.features.flip else turn),
                "fire": bool(fire)}
