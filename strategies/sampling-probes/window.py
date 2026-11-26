"""Approach with matched navigation and vary only the first eligible fire bit."""
from math import hypot

from navigation import clamp
from seedcup.game import rules
from seedcup.game.bots import clear_shot
from tactics import TURN_PER_TICK, SamplingTactic
from window_eligibility import CLOSE_CELLS, heading_error, shot_opportunity


class WindowHold(SamplingTactic):
    fire_at_opportunity = False

    def on_match_start(self, info):
        super().on_match_start(info)
        self.qualified = False

    def decide(self, obs):
        own, enemy = obs["self"], obs["enemy"]
        source, target = (tuple(state[key] for key in ("x", "y", "heading", "hp", "alive"))
                          for state in (own, enemy))
        if not self.qualified:
            opportunity = shot_opportunity(
                self.walls, source, target, turn_per_tick=TURN_PER_TICK,
                reload_ticks=round(own["reload"] * rules.TICK_HZ),
            )
            if opportunity is not None:
                self.qualified = True
                return {"forward": 0, "turn": opportunity["turn"], "fire": self.fire_at_opportunity}

        x, y = source[:2]
        ex, ey = target[:2]
        distance = hypot(ex - x, ey - y)
        if clear_shot(self.walls, x, y, ex, ey):
            diff = heading_error(source, target)
            if distance <= CLOSE_CELLS:
                return {"forward": 0, "turn": clamp(diff / TURN_PER_TICK, 1), "fire": False}
            if all(self.free(x + (ex - x) * step / distance, y + (ey - y) * step / distance)
                   for step in (.25, .5, .75)):
                return {"forward": .85 if abs(diff) < 45 else 0,
                        "turn": clamp(diff / TURN_PER_TICK, 1), "fire": False}
        return self.move(own, target[:2])


class WindowFire(WindowHold):
    fire_at_opportunity = True
