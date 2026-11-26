"""Transferable short-range aiming adjustment for a packaged SeedCup policy."""

from json import loads
from math import atan2, degrees, hypot
from pathlib import Path

from cycle_base import BasePolicy
from seedcup.game import rules
from seedcup.game.bots import clear_shot, wall_grid

BLEND = loads(Path(__file__).with_name("cycle_ability.json").read_text(encoding="utf-8"))["close_aim_blend"]
TURN_STEP = rules.TURN_RATE * rules.DT


class CyclePolicy(BasePolicy):
    def on_match_start(self, info):
        super().on_match_start(info)
        self._cycle_walls = wall_grid(info)

    def decide(self, obs):
        action = super().decide(obs)
        if not action["fire"]:
            return action
        own, enemy = obs["self"], obs["enemy"]
        dx, dy = enemy["x"] - own["x"], enemy["y"] - own["y"]
        if (not 0 < hypot(dx, dy) <= 3 or not enemy["alive"]
                or not clear_shot(self._cycle_walls, own["x"], own["y"], enemy["x"], enemy["y"])):
            return action
        heading = own["heading"] + TURN_STEP * action["turn"]
        error = (degrees(atan2(dy, dx)) - heading + 180) % 360 - 180
        turn = action["turn"] + BLEND * error / TURN_STEP
        return action | {"turn": max(-1, min(1, turn))}
