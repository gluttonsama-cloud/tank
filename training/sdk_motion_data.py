"""Extract the boss-visible movement observations common to replay and SDK captures."""

from math import cos, floor, radians, sin

import numpy as np
from seedcup.game import SIDES, GameMap, rules
from seedcup.game.engine import ang_diff
from seedcup.game.runner import match_info

if __package__:
    from .sdk_motion_policy import FEATURE_COUNT, MOVE_STEP, TURN_STEP, MovementFeatures, observed_pose
else:
    from sdk_motion_policy import FEATURE_COUNT, MOVE_STEP, TURN_STEP, MovementFeatures, observed_pose


TANK_FIELDS = ("x", "y", "heading", "hp", "alive")
SHELL_FIELDS = ("x", "y", "dx", "dy")
POSITION_TOLERANCE = .0101  # Two positions quantized to 0.01 cells can differ by this much.


def match_context(payload, source, side):
    """Return SDK opening information from the requested player's side."""
    index = SIDES.index(side)
    if source == "replay":
        game_map = GameMap.from_dict(payload["map"])
        meta = payload["match"]
        return match_info(game_map, index, meta[SIDES[1 - index]]["name"], game=meta["game"])
    if source != "sdk":
        raise ValueError(f"未知移动样本来源 {source}")
    start = payload["start"]
    if side == start["side"]:
        return start
    spawn = start["spawn"]
    return {"side": side, "map": start["map"], "rules": start["rules"], "game": start["game"],
            "spawn": {"x": start["map"]["width"] - spawn["x"], "y": spawn["y"],
                      "heading": (180 - spawn["heading"]) % 360}}


def iter_views(payload, source, side):
    """Yield each observed frame ID and the movement subset of the boss SDK view."""
    index = SIDES.index(side)
    if source == "replay":
        for tick, red, blue, shells, _ in payload["frames"]:
            own, other = (red, blue) if index == 0 else (blue, red)
            yield tick, {
                "frame": tick,
                "self": dict(zip(TANK_FIELDS, (*own[:4], bool(own[4])), strict=True)),
                "enemy": dict(zip(TANK_FIELDS, (*other[:4], bool(other[4])), strict=True)),
                "shells": [{"x": x, "y": y, "dx": dx, "dy": dy, "mine": owner == index}
                           for x, y, dx, dy, owner in shells],
                "zone": {"x": rules.ZONE_CX, "y": rules.ZONE_CY, "radius": round(rules.zone_radius(tick), 3)},
            }
    elif source == "sdk":
        swap = payload["start"]["side"] != side
        own, other = ("enemy", "self") if swap else ("self", "enemy")
        for obs, _ in payload["frames"]:
            yield obs["frame"], {
                "frame": obs["frame"],
                "self": {key: obs[own][key] for key in TANK_FIELDS},
                "enemy": {key: obs[other][key] for key in TANK_FIELDS},
                "shells": [{key: shell[key] for key in SHELL_FIELDS} | {"mine": shell["mine"] != swap}
                           for shell in obs["shells"]],
                "zone": {key: obs["zone"][key] for key in ("x", "y", "radius")},
            }
    else:
        raise ValueError(f"未知移动样本来源 {source}")


def _blocked_standstill(geometry, x, y, c, s, enemy):
    # A full forward or reverse step can be canceled by a wall or simultaneous tank movement.
    radius = rules.RADIUS + POSITION_TOLERANCE
    ex, ey, _ = observed_pose(enemy)
    for direction in (-1, 1):
        nx, ny = x + direction * c * MOVE_STEP, y + direction * s * MOVE_STEP
        if (nx - ex)**2 + (ny - ey)**2 < (2 * radius + MOVE_STEP)**2:
            return True
        for cy in range(floor(ny - radius), floor(ny + radius) + 1):
            for cx in range(floor(nx - radius), floor(nx + radius) + 1):
                if geometry.is_wall(cx, cy):
                    wx, wy = max(cx - nx, 0, nx - cx - 1), max(cy - ny, 0, ny - cy - 1)
                    if wx*wx + wy*wy < radius*radius:
                        return True
    return False


def _both_alive(obs):
    return obs["self"]["alive"] and obs["enemy"]["alive"]


def _motion_intervals(payload, source, side, geometry):
    result = payload["result"] if source == "replay" else payload["end"]["result"]
    terminal = result["ticks"]
    views = iter_views(payload, source, side)
    _, obs = next(views, (None, None))
    prev = None
    for after_frame, after in views:
        frame = obs["frame"]
        if after_frame <= frame:
            raise ValueError(f"移动样本帧号未递增 {frame} -> {after_frame}")
        if after_frame >= terminal:
            break
        if after_frame == frame + 1 and _both_alive(obs) and _both_alive(after):
            x, y, heading = observed_pose(obs["self"])
            nx, ny, next_heading = observed_pose(after["self"])
            dx, dy = nx - x, ny - y
            c, s = cos(radians(next_heading)), sin(radians(next_heading))
            if max(abs(dx), abs(dy)) <= POSITION_TOLERANCE:
                forward = (None if _blocked_standstill(geometry, x, y, c, s, obs["enemy"]) else 0.)
            else:
                # The SDK updates heading before translation. Project even when a wall clips one axis.
                forward = max(-1., min(1., (c * dx + s * dy) / MOVE_STEP))
            if forward is not None:
                turn = max(-1., min(1., ang_diff(next_heading, heading) / TURN_STEP))
                yield frame, obs, prev, np.asarray((forward, turn), dtype=np.float32)
        prev, obs = (obs if after_frame == frame + 1 and _both_alive(obs) else None), after


def iter_motion_intervals(payload, source, side):
    """Yield (frame, observation, previous observation, [forward, turn]) for identifiable motion."""
    yield from _motion_intervals(payload, source, side, MovementFeatures(match_context(payload, source, side)))


def motion_rows(payload, source, side):
    """Return row-aligned movement features, observed SDK controls, and visible projectile risk."""
    geometry = MovementFeatures(match_context(payload, source, side))
    size = len(payload["frames"])
    x = np.empty((size, FEATURE_COUNT), dtype=np.float32)
    y = np.empty((size, 2), dtype=np.float32)
    threat = np.empty(size, dtype=bool)
    count = 0
    for count, (_, obs, prev, action) in enumerate(_motion_intervals(payload, source, side, geometry), start=1):
        x[count - 1], threat[count - 1] = geometry.extract(obs, prev)
        y[count - 1] = action
    return x[:count], y[:count], threat[:count]
