"""SDK observations and NumPy inference for the replay-trained sparring bot."""

from collections import deque
from dataclasses import replace
from math import atan2, cos, degrees, hypot, radians, sin, tau
from pathlib import Path

import numpy as np
from seedcup import Bot
from seedcup.game import Game, GameMap, rules
from seedcup.game.bots import clear_shot, mirror_obs, ray_wall, wall_grid
from seedcup.game.engine import STILL, Shell, Tank, ray_circle
from seedcup.game.maps import Spawn


def angle_delta(target, current):
    return (target - current + 180) % 360 - 180


TURN_BOUNDS = np.array([-.9, -.075, .075, .9], dtype=np.float32)
TURN_LOWER = np.concatenate(([-1], TURN_BOUNDS))
TURN_UPPER = np.concatenate((TURN_BOUNDS, [1]))
EVADE_TICKS = 5


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


class ManeuverFeatures(Features):
    def __init__(self, info):
        super().__init__(info)
        self.motion = deque(maxlen=20)
        self.sight_run = 0
        self.stall_run = 0

    def __call__(self, obs):
        base = super().__call__(obs)
        x, y, ex, ey, heading, enemy_heading = self.history[-1]
        h = radians(heading)
        c, s = cos(h), sin(h)
        position = (x, y, ex, ey, heading, enemy_heading)
        context = []
        for lag in (4, 10, 20):
            width = min(lag, len(self.motion))
            old = self.motion[-width] if width else position
            window = max(1, width) * rules.SPEED * rules.DT
            for old_x, old_y, new_x, new_y in ((old[0], old[1], x, y), (old[2], old[3], ex, ey)):
                vx, vy = (new_x - old_x) / window, (new_y - old_y) / window
                context.extend((c*vx + s*vy, c*vy - s*vx))
            context.extend(angle_delta(current, previous) / 180
                           for current, previous in zip(position[4:], old[4:], strict=True))
        previous = self.motion[-1] if self.motion else position
        self.motion.append(position)
        if base[14]:
            self.sight_run = max(0, self.sight_run) + 1
        else:
            self.sight_run = min(0, self.sight_run) - 1
        self.stall_run = self.stall_run + 1 if hypot(x - previous[0], y - previous[1]) < .02 else 0

        # A shell stopped by a wall cannot reach the current tank center.
        threats = []
        horizon = rules.SHELL_SPEED
        shells = mirror_obs(obs)["shells"] if self.flip else obs["shells"]
        for shell in shells:
            if shell["mine"]:
                continue
            sx, sy = round(shell["x"], 2), round(shell["y"], 2)
            vx, vy = round(shell["dx"], 3), round(shell["dy"], 3)
            dx, dy = x - sx, y - sy
            along = dx*vx + dy*vy
            lateral = -dx*vy + dy*vx
            if not (0 < along < horizon and abs(lateral) < 2):
                continue
            if ray_wall(self.walls, sx, sy, vx, vy, along) < along - rules.HIT_RADIUS:
                continue
            threats.append((along / horizon, lateral / 2, c*vx+s*vy, c*vy-s*vx))
        threats = sorted(threats)[:2]
        threats += [(1.0, 0.0, 0.0, 0.0)] * (2 - len(threats))
        zone = obs["zone"]
        future_margin = (rules.zone_radius(obs["frame"] + 30) - hypot(x - zone["x"], y - zone["y"])) / 20
        enemy_aim = atan2(y - ey, x - ex) - radians(enemy_heading)
        return np.concatenate((base[:-4], np.asarray([
            *context, np.clip(self.sight_run / 20, -1, 1), min(self.stall_run, 20) / 20,
            future_margin, sin(enemy_aim), cos(enemy_aim), *threats[0], *threats[1],
        ], dtype=np.float32), base[-4:]))


def predict(features, weights):
    x = (features - weights["mean"]) / weights["scale"]
    hidden = np.tanh(x @ weights["w1"] + weights["b1"])
    if "w3" in weights:
        hidden = np.tanh(hidden @ weights["w2"] + weights["b2"])
        return hidden @ weights["w3"] + weights["b3"]
    return hidden @ weights["w2"] + weights["b2"]


def decode(features, scores, weights):
    if "turn_centers" in weights:
        return decode_maneuver(features, scores, weights)
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


def decode_maneuver(features, scores, weights):
    joint = scores[..., :15].reshape(*scores.shape[:-1], 3, 5)
    # Marginalize the speed over possible turns, then pick its most likely turn.
    probability = np.exp(joint - joint.max(axis=(-2, -1), keepdims=True))
    speed = np.argmax(probability.sum(axis=-1), axis=-1)
    turn_scores = np.take_along_axis(joint, speed[..., None, None], axis=-2).squeeze(-2)
    turn_index = np.argmax(turn_scores, axis=-1)
    turn = weights["turn_centers"][turn_index] + np.take_along_axis(
        scores[..., 15:20], turn_index[..., None], axis=-1).squeeze(-1)
    turn = np.clip(turn, TURN_LOWER[turn_index], TURN_UPPER[turn_index])
    reload, aim, margin, clear = np.moveaxis(features[..., -4:], -1, 0)
    ticks = np.rint(reload * rules.RELOAD_TICKS)
    aim_turn = np.clip(aim, -1, 1)
    wants_fire = (ticks <= 1) & (scores[..., 20] >= 0) & (clear == 1)
    fire = wants_fire & (np.abs(aim - aim_turn) <= margin)
    turn = np.where(fire, aim_turn, turn)
    return weights["speeds"][speed], turn, fire


class EvadeIncoming:
    def __init__(self, info, walls):
        self.grid = bytes(value for row in info["map"]["grid"] for value in row)
        self.walls = walls
        self.map = None

    def __call__(self, obs, action):
        me, enemy = obs["self"], obs["enemy"]
        shells = []
        threatened = False
        shell_travel = EVADE_TICKS * rules.SHELL_SPEED * rules.DT
        tank_travel = EVADE_TICKS * rules.SPEED * rules.DT
        for n, shell in enumerate(obs["shells"], start=1):
            if shell["mine"]:
                continue
            x, y, vx, vy = (shell[k] for k in ("x", "y", "dx", "dy"))
            if hypot(x - me["x"], y - me["y"]) > shell_travel + tank_travel + rules.HIT_RADIUS:
                continue
            impact = ray_circle(x, y, vx, vy, me["x"], me["y"], rules.HIT_RADIUS)
            if impact <= shell_travel and ray_wall(self.walls, x, y, vx, vy, shell_travel) >= impact:
                threatened = True
            shells.append(Shell(n, 1, x, y, vx, vy))
        if not threatened:
            return action

        tanks = [Tank(i, obs["x"], obs["y"], obs["heading"], alive=obs["alive"])
                 for i, obs in enumerate((me, enemy))]
        if self.map is None:
            spawns = tuple(Spawn(t.x, t.y, t.h) for t in tanks)
            self.map = GameMap(0, self.grid, sum(self.grid) / len(self.grid), -1, spawns)

        def safe(candidate):
            game = Game(self.map, tanks=[replace(t) for t in tanks], shells=[replace(sh) for sh in shells])
            for _ in range(EVADE_TICKS):
                game.step([candidate, STILL])
            return game.tanks[0].hp == rules.HP

        current = action | {"fire": False}
        if safe(current):
            return action
        speed, turn = action["forward"], action["turn"]
        alternatives = [(v, t) for v in (-1., 1.) for t in dict.fromkeys((turn, -1., 0., 1.))
                        if (v, t) != (speed, turn)]
        for v, t in sorted(alternatives, key=lambda a: (a[1] - turn)**2 + (a[0] - speed)**2):
            candidate = {"forward": v, "turn": t, "fire": False}
            if safe(candidate):
                return candidate | {"fire": action["fire"] and t == turn}
        return action


class BossSparring(Bot):
    def __init__(self):
        with np.load(Path(__file__).with_name("weights.npz"), allow_pickle=False) as weights:
            self.weights = dict(weights)

    def on_match_start(self, info):
        extractor = ManeuverFeatures if "turn_centers" in self.weights else Features
        self.features = extractor(info)
        self.evade = EvadeIncoming(info, self.features.walls) if "evade_incoming" in self.weights else None

    def decide(self, obs):
        features = self.features(obs)
        forward, turn, fire = decode(features, predict(features, self.weights), self.weights)
        action = {"forward": float(forward), "turn": float(-turn if self.features.flip else turn),
                  "fire": bool(fire)}
        return self.evade(obs, action) if self.evade else action
