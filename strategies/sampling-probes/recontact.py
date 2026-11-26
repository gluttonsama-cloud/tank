"""Sampling tactic that alternates close pressure and separation."""
from math import hypot

from navigation import angle, clamp, delta
from seedcup.game import rules
from seedcup.game.bots import clear_shot
from tactics import TURN_PER_TICK, SamplingTactic


class Recontact(SamplingTactic):
    """Press at close range, then open enough space to approach again."""

    def on_match_start(self, info):
        super().on_match_start(info)
        self.pressure_until = None
        self.withdraw_until = None
        self.destination = None
        self.fired = False

    def decide(self, obs):
        own, enemy = obs["self"], obs["enemy"]
        x, y = own["x"], own["y"]
        ex, ey = enemy["x"], enemy["y"]
        frame = obs["frame"]
        distance = hypot(ex - x, ey - y)
        zone = obs["zone"]
        center = zone["x"], zone["y"]
        zone_limit = rules.zone_radius(frame + rules.TICK_HZ) - .3

        if self.pressure_until is not None and frame >= self.pressure_until:
            self.pressure_until = None
            self.withdraw_until = frame + 5 * rules.TICK_HZ
            self.destination = self.withdrawal_cell(x, y, ex, ey, center, zone_limit)

        if self.withdraw_until is not None:
            if distance >= 5.5 or frame >= self.withdraw_until:
                self.withdraw_until = None
            else:
                if frame % rules.TICK_HZ == 0:
                    self.destination = self.withdrawal_cell(x, y, ex, ey, center, zone_limit)
                return self.move(own, self.destination)

        if hypot(x - center[0], y - center[1]) > zone_limit:
            return self.move(own, center)
        if not clear_shot(self.walls, x, y, ex, ey):
            return self.move(own, (ex, ey))

        if distance <= 2.75 and self.pressure_until is None:
            self.pressure_until = frame + 2 * rules.TICK_HZ
            self.fired = False
        diff = delta(angle(ex - x, ey - y), own["heading"])
        turn = clamp(diff / TURN_PER_TICK, 1)
        aim_error = abs(diff - turn * TURN_PER_TICK)
        if distance > 2.5 and not all(
            self.free(x + (ex - x) * step / distance, y + (ey - y) * step / distance)
            for step in (.25, .5, .75)
        ):
            return self.move(own, (ex, ey))
        fire = (self.pressure_until is not None and distance <= 3
                and not self.fired and own["reload"] == 0 and aim_error < 2)
        self.fired |= fire
        speed = 1 if distance > 2.5 and aim_error < 45 else 0
        return {"forward": speed, "turn": turn, "fire": fire}

    def withdrawal_cell(self, x, y, ex, ey, center, zone_limit):
        destination = self.nearest_cell(x, y, lambda cx, cy:
            hypot(cx - ex, cy - ey) >= 5.5
            and hypot(cx - center[0], cy - center[1]) <= zone_limit)
        return destination or center
