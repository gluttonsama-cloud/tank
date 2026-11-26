"""Movement features shared by offline extraction and SDK inference."""

from collections import deque
from math import atan2, cos, hypot, radians, sin, tau

import numpy as np
from seedcup.game import rules
from seedcup.game.bots import clear_shot, ray_wall, wall_grid
from seedcup.game.engine import ang_diff
from seedcup.game.maps import cell_of


MOVE_STEP = rules.SPEED * rules.DT
TURN_STEP = rules.TURN_RATE * rules.DT
LOCAL_RANGE = 8.0
WALL_RANGE = 4.0
SHELL_RANGE = rules.SHELL_SPEED
THREAT_RANGE = .5 * rules.SHELL_SPEED
THREAT_WIDTH = rules.HIT_RADIUS + MOVE_STEP
FEATURE_NAMES = (
    "elapsed", "self_x", "self_y", "heading_sin", "heading_cos", "hp", "enemy_hp",
    "alive", "enemy_alive", "range", "near_range", "bearing_sin", "bearing_cos",
    "enemy_heading_sin", "enemy_heading_cos", "enemy_aim_sin", "enemy_aim_cos",
    "zone_radius", "zone_margin", "enemy_zone_margin", "zone_sin", "zone_cos",
    "has_previous", "self_forward", "self_lateral", "enemy_forward", "enemy_lateral",
    "self_turn", "enemy_turn", "clear_shot",
    *(f"wall_{degrees}" for degrees in range(0, 360, 45)),
    "enemy_path_sin", "enemy_path_cos", "enemy_path_distance",
    "zone_path_sin", "zone_path_cos", "zone_path_distance", "threat",
    *(f"shell_{slot}_{component}" for slot in (1, 2)
      for component in ("arrival", "miss", "forward", "lateral")),
)
FEATURE_COUNT = len(FEATURE_NAMES)


def observed_pose(tank):
    # Replays round positions to 0.01 cells and headings to 0.1 degrees.
    return round(tank["x"], 2), round(tank["y"], 2), round(tank["heading"], 1) % 360


class MovementFeatures:
    """Cache map navigation per match. Callers supply the preceding SDK observation."""

    def __init__(self, info):
        self.walls = wall_grid(info)
        self.links = []
        for pos, cell in enumerate(self.walls):
            x, y = pos % rules.W, pos // rules.W
            neighbors = ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))
            self.links.append(tuple(b * rules.W + a for a, b in neighbors
                                    if cell == 1 and 0 <= a < rules.W and 0 <= b < rules.H
                                    and self.walls[b * rules.W + a] == 1))
        self.distances = {}

    def is_wall(self, x, y):
        return not (0 <= x < rules.W and 0 <= y < rules.H) or self.walls[y * rules.W + x] == 2

    def path(self, x, y, tx, ty, heading):
        sx, sy = cell_of(x, y)
        gx, gy = cell_of(tx, ty)
        start, goal = sy * rules.W + sx, gy * rules.W + gx
        if start == goal or self.is_wall(sx, sy) or self.is_wall(gx, gy):
            return _direct_path(x, y, tx, ty, heading)
        if goal not in self.distances:
            distances = [-1] * len(self.links)
            distances[goal] = 0
            pending = deque((goal,))
            while pending:
                cell = pending.popleft()
                for neighbor in self.links[cell]:
                    if distances[neighbor] < 0:
                        distances[neighbor] = distances[cell] + 1
                        pending.append(neighbor)
            self.distances[goal] = distances
        distances = self.distances[goal]
        if distances[start] < 0:
            return _direct_path(x, y, tx, ty, heading)
        first = min(self.links[start], key=distances.__getitem__)
        angle = atan2(first // rules.W + .5 - y, first % rules.W + .5 - x) - heading
        return sin(angle), cos(angle), distances[start] / 40

    def extract(self, obs, prev):
        return _movement_features(obs, prev, self)

    def features(self, obs, prev):
        return self.extract(obs, prev)[0]


def _direct_path(x, y, tx, ty, heading):
    dx, dy = tx - x, ty - y
    angle = atan2(dy, dx) - heading
    return sin(angle), cos(angle), hypot(dx, dy) / 40


def _movement_features(obs, prev, geometry):
    me, enemy, zone = obs["self"], obs["enemy"], obs["zone"]
    x, y, heading = observed_pose(me)
    ex, ey, enemy_heading = observed_pose(enemy)
    h = radians(heading)
    c, s = cos(h), sin(h)

    def relative(dx, dy):
        return c * dx + s * dy, c * dy - s * dx

    frame = obs["frame"]
    prev_valid = (prev is not None and prev["frame"] == frame - 1
                  and prev["self"]["alive"] and prev["enemy"]["alive"] and me["alive"] and enemy["alive"])
    if prev_valid:
        px, py, past_heading = observed_pose(prev["self"])
        pex, pey, past_enemy_heading = observed_pose(prev["enemy"])
        velocity = (*relative((x - px) / MOVE_STEP, (y - py) / MOVE_STEP),
                    *relative((ex - pex) / MOVE_STEP, (ey - pey) / MOVE_STEP),
                    ang_diff(heading, past_heading) / TURN_STEP,
                    ang_diff(enemy_heading, past_enemy_heading) / TURN_STEP)
    else:
        velocity = (0.,) * 6

    dx, dy = ex - x, ey - y
    distance = hypot(dx, dy)
    bearing = atan2(dy, dx)
    eh = radians(enemy_heading)
    zx, zy = zone["x"], zone["y"]
    center = atan2(zy - y, zx - x) - h
    radius = round(zone["radius"], 3)

    incoming = []
    threat = False
    for shell in obs["shells"]:
        if shell["mine"]:
            continue
        sx, sy = round(shell["x"], 2), round(shell["y"], 2)
        vx, vy = round(shell["dx"], 3), round(shell["dy"], 3)
        scale = hypot(vx, vy)
        vx, vy = vx / scale, vy / scale
        along, miss = (x - sx) * vx + (y - sy) * vy, (y - sy) * vx - (x - sx) * vy
        if not (0 < along < SHELL_RANGE and abs(miss) < 2):
            continue
        if geometry:
            reach = max(0, along - rules.HIT_RADIUS)
            if reach and ray_wall(geometry.walls, sx, sy, vx, vy, reach) < reach - 1e-6:
                continue
        threat |= along <= THREAT_RANGE and abs(miss) <= THREAT_WIDTH
        incoming.append((along / SHELL_RANGE, miss / 2, *relative(vx, vy)))
    nearest = sorted(incoming)[:2]
    nearest += [(1., 0., 0., 0.)] * (2 - len(nearest))

    if geometry:
        visible = clear_shot(geometry.walls, x, y, ex, ey)
        walls = tuple(ray_wall(geometry.walls, x, y, cos(h + k*tau/8), sin(h + k*tau/8), WALL_RANGE)
                      / WALL_RANGE for k in range(8))
        path = geometry.path
    else:
        visible, walls, path = True, (1.,) * 8, _direct_path

    feature = np.asarray([
        frame / rules.GAME_TICKS, x / rules.W, y / rules.H, s, c,
        me["hp"] / rules.HP, enemy["hp"] / rules.HP, me["alive"], enemy["alive"],
        distance / 40, min(distance / LOCAL_RANGE, 1), sin(bearing-h), cos(bearing-h),
        sin(eh-h), cos(eh-h), sin(bearing + tau/2 - eh), cos(bearing + tau/2 - eh),
        radius / rules.ZONE_R0, (radius - hypot(zx - x, zy - y)) / 20,
        (radius - hypot(zx - ex, zy - ey)) / 20, sin(center), cos(center),
        prev_valid, *velocity, visible, *walls, *path(x, y, ex, ey, h), *path(x, y, zx, zy, h),
        threat, *nearest[0], *nearest[1],
    ], dtype=np.float32)
    return feature, threat


def movement_features(obs, prev, *, geometry=None):
    """Return features in world-coordinate SDK steering convention."""
    return _movement_features(obs, prev, geometry)[0]
