"""Sampling approaches that change cover and delay returning to the zone."""
from collections import deque
from math import hypot

from seedcup.game import rules
from seedcup.game.bots import clear_shot

from navigation import Lancer, angle, clamp, delta


TURN_PER_TICK = rules.TURN_RATE * rules.DT


class SamplingTactic(Lancer):
    def nearest_cell(self, x, y, qualifies):
        origin = int(x), int(y)
        queue = deque([origin])
        seen = {origin}
        while queue:
            cell = queue.popleft()
            center = cell[0] + .5, cell[1] + .5
            if qualifies(*center):
                return center
            for neighbor in self.links[cell]:
                if neighbor not in seen:
                    queue.append(neighbor)
                    seen.add(neighbor)
        return None

    def move(self, own, destination):
        x, y = own["x"], own["y"]
        tx, ty = self.waypoint(x, y, *destination)
        diff = delta(angle(tx - x, ty - y), own["heading"])
        speed = min(.85, 4 * hypot(tx - x, ty - y))
        if abs(diff) > 90:
            diff = delta(diff + 180, 0)
            speed = -speed
        return {"forward": speed if abs(diff) < 45 else 0,
                "turn": clamp(diff / TURN_PER_TICK, 1), "fire": False}


class Peek(SamplingTactic):
    """Expose the tank, take a shot, and retrace the path to cover."""

    def on_match_start(self, info):
        super().on_match_start(info)
        self.cover = None
        self.exposed_until = None
        self.rest_until = 0
        self.withdrawing = False
        self.has_fired = False

    def decide(self, obs):
        own, enemy = obs["self"], obs["enemy"]
        x, y, heading = own["x"], own["y"], own["heading"]
        ex, ey = enemy["x"], enemy["y"]
        frame = obs["frame"]
        zone = obs["zone"]
        center = zone["x"], zone["y"]
        zone_limit = rules.zone_radius(frame + rules.RELOAD_TICKS) - .4
        if hypot(x - center[0], y - center[1]) > zone_limit:
            self.cover = None
            self.withdrawing = False
            self.exposed_until = None
            return self.move(own, center)
        visible = clear_shot(self.walls, x, y, ex, ey)
        if not visible:
            if self.withdrawing or self.exposed_until is not None:
                self.rest_until = frame + rules.RELOAD_TICKS
            self.withdrawing = False
            self.exposed_until = None
            if frame >= self.rest_until:
                self.cover = x, y
                return self.move(own, (ex, ey))
            return {"forward": 0, "turn": 0, "fire": False}

        if self.exposed_until is None and not self.withdrawing:
            self.exposed_until = frame + rules.TICK_HZ
            self.has_fired = False
        if self.withdrawing or self.has_fired or frame >= self.exposed_until:
            # A moving opponent can invalidate cover stored on the previous approach.
            if self.cover is None or clear_shot(self.walls, *self.cover, ex, ey):
                self.cover = self.nearest_cell(x, y, lambda cx, cy:
                    hypot(cx - center[0], cy - center[1]) <= zone_limit
                    and not clear_shot(self.walls, cx, cy, ex, ey))
            if self.cover is not None:
                self.withdrawing = True
                return self.move(own, self.cover)

        diff = delta(angle(ex - x, ey - y), heading)
        turn = clamp(diff / TURN_PER_TICK, 1)
        fire = own["reload"] == 0 and abs(diff - turn * TURN_PER_TICK) < 2
        self.has_fired |= fire
        return {"forward": 0, "turn": turn, "fire": fire}


class LateZone(SamplingTactic):
    """Wait at mid-radius and return after shrinking reaches the tank."""

    def on_match_start(self, info):
        super().on_match_start(info)
        self.hold = None
        self.returning = False

    def decide(self, obs):
        own, enemy = obs["self"], obs["enemy"]
        x, y = own["x"], own["y"]
        zone = obs["zone"]
        center = zone["x"], zone["y"]
        if self.hold is None:
            # Midpoint of the shrink schedule leaves time to reach the final circle.
            radius = (zone["radius"] + zone["final_radius"]) / 2
            ex, ey = enemy["x"], enemy["y"]
            self.hold = self.nearest_cell(x, y, lambda cx, cy:
                radius <= hypot(cx - center[0], cy - center[1]) < radius + 2
                and not clear_shot(self.walls, cx, cy, ex, ey))
            if self.hold is None:
                self.hold = x, y
        self.returning |= not own["in_zone"]
        destination = center if self.returning else self.hold
        if hypot(x - destination[0], y - destination[1]) < .18:
            return {"forward": 0, "turn": 0, "fire": False}
        return self.move(own, destination)
