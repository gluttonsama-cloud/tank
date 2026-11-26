"""Held-out opponent that alternates tangential movement around the target."""
from math import hypot

from navigation import Lancer, angle, clamp, delta
from seedcup.game import rules
from seedcup.game.bots import clear_shot

_ORBIT_CELLS = 5
_ORBIT_INTERVAL = 4 * rules.TICK_HZ
_TURN_PER_TICK = rules.TURN_RATE * rules.DT


class HoldoutOrbit(Lancer):
    def decide(self, obs):
        own, enemy = obs["self"], obs["enemy"]
        x, y = own["x"], own["y"]
        ex, ey = enemy["x"], enemy["y"]
        frame, zone = obs["frame"], obs["zone"]
        center = zone["x"], zone["y"]
        if hypot(x - center[0], y - center[1]) > rules.zone_radius(frame + _ORBIT_INTERVAL) - .5:
            return self.drive(own, self.waypoint(x, y, *center))

        dx, dy = ex - x, ey - y
        distance = hypot(dx, dy)
        phase, tick = divmod(frame, _ORBIT_INTERVAL)
        if (distance <= _ORBIT_CELLS + 1 and tick >= _ORBIT_INTERVAL - rules.TICK_HZ
                and clear_shot(self.walls, x, y, ex, ey)):
            diff = delta(angle(dx, dy), own["heading"])
            turn = clamp(diff / _TURN_PER_TICK, 1)
            return {"forward": 0, "turn": turn,
                    "fire": abs(diff - turn * _TURN_PER_TICK) < 2}

        # Add a radial correction to the tangent to stay near the orbit distance.
        if 0 < distance <= _ORBIT_CELLS + 1:
            radial = clamp(distance - _ORBIT_CELLS, 1)
            direction = -1 if phase % 2 else 1
            vx = (radial * dx - direction * dy) / distance
            vy = (radial * dy + direction * dx) / distance
            if all(self.free(x + vx * step, y + vy * step) for step in (.25, .5, .75)):
                return self.drive(own, (x + vx, y + vy))
        return self.drive(own, self.waypoint(x, y, ex, ey))

    def drive(self, own, destination):
        diff = delta(angle(destination[0] - own["x"], destination[1] - own["y"]), own["heading"])
        speed = 1
        if abs(diff) > 90:
            diff = delta(diff + 180, 0)
            speed = -speed
        return {"forward": speed if abs(diff) < 55 else 0,
                "turn": clamp(diff / _TURN_PER_TICK, 1), "fire": False}
