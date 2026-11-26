"""Route an early match from the rookie specialist to the boss specialist."""

import sys
from functools import cache
from hashlib import sha256
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_file_location
from json import loads
from math import cos, hypot, radians, sin
from pathlib import Path

from seedcup import Bot
from seedcup.game import rules
from seedcup.game.bots import ray_wall, wall_grid


PACKAGE = Path(__file__).resolve().parent


class _ReadOnlyLoader(SourceFileLoader):
    def set_data(self, path, data, *, _mode=0o666):
        # Keep the vendored snapshot content hashes independent of imports.
        pass


@cache
def policy_loader(boss_folder):
    source = boss_folder.resolve() / "boss_specialist_policy.py"
    name = "_style_router_loader_" + sha256(str(source).encode()).hexdigest()
    spec = spec_from_file_location(name, source, loader=_ReadOnlyLoader(name, str(source)))
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.load_strategy


def policy_match_info(info):
    return {
        "side": info["side"],
        "map": {key: info["map"][key] for key in ("width", "height", "grid")},
        "spawn": info["spawn"],
        "rules": info["rules"],
    }


class StyleDetector:
    """Use sustained opponent drive while a shot is geometrically available."""

    def __init__(self, walls, config):
        self.walls = walls
        self.config = config
        self.previous = None
        self.drive_streak = 0
        self.switch_frame = None
        self.trace = []

    def _clear_lane(self, enemy, own, zone):
        x, y, _ = enemy
        dx, dy = own[0] - x, own[1] - y
        distance = hypot(dx, dy)
        if distance < 2 * rules.RADIUS + 2 * rules.SPEED * rules.DT:
            return False
        if hypot(x - zone[0], y - zone[1]) >= zone[2] - 1.05:
            return False
        ux, uy = dx / distance, dy / distance
        # Both offset rays must be clear through the full segment. This margin
        # keeps a corner grazing the centre ray from supplying evidence.
        return all(
            ray_wall(self.walls, x - uy * off, y + ux * off, ux, uy, distance) >= distance
            for off in (-0.04, 0.04)
        )

    def observe(self, obs):
        """Return the switch frame, or None while policy A remains active."""
        frame = obs["frame"]
        if self.switch_frame is not None or frame > self.config["latest_frame"]:
            return self.switch_frame

        own, enemy, zone = obs["self"], obs["enemy"], obs["zone"]
        now = (round(enemy["x"], 2), round(enemy["y"], 2), round(enemy["heading"], 1))
        own_xy = (round(own["x"], 2), round(own["y"], 2))
        clear = enemy["alive"] and own["alive"] and self._clear_lane(
            now, own_xy, (zone["x"], zone["y"], zone["radius"])
        )

        forward = None
        if self.previous is not None:
            before_frame, before, before_clear = self.previous
            if before_frame + 1 == frame and before_clear and clear:
                heading = radians(now[2])
                forward = (now[0] - before[0]) * cos(heading) + (now[1] - before[1]) * sin(heading)
        self.previous = (frame, now, clear)
        self.drive_streak = self.drive_streak + 1 if forward is not None and forward >= self.config["min_drive"] else 0

        if self.drive_streak >= self.config["consecutive_frames"]:
            self.switch_frame = frame
        self.trace.append({"frame": frame, "clear_lane": bool(clear),
                           "forward_cells": round(forward, 4) if forward is not None else None,
                           "drive_streak": self.drive_streak, "switch": self.switch_frame == frame})
        return self.switch_frame


class StyleRouter(Bot):
    def __init__(self):
        config = loads((PACKAGE / "detector.json").read_text(encoding="utf-8"))
        self.config = config
        load = policy_loader(PACKAGE / "policy_b")
        self.a = load(PACKAGE / "policy_a")()
        self.b = load(PACKAGE / "policy_b")()

    def on_match_start(self, info):
        info = policy_match_info(info)
        self.detector = StyleDetector(wall_grid(info), self.config)
        self.active = "A"
        self.a.on_match_start(info)
        self.b.on_match_start(info)

    def decide(self, obs):
        if self.active == "B" or self.detector.observe(obs) is not None:
            self.active = "B"
            return self.b.decide(obs)
        if obs["frame"] < self.config["latest_frame"]:
            # Frozen B v7 updates the base policy and world before its search.
            # Its discarded lookahead uses independent branches of the world.
            super(type(self.b), self.b).decide(obs)
            self.b._local_world.update(obs)
        return self.a.decide(obs)

    def on_match_end(self, result):
        self.a.on_match_end(result)
        self.b.on_match_end(result)
