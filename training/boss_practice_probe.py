"""Controlled game opponents for recording SeedCup practice-room behaviour."""
from math import atan2, degrees, hypot

from seedcup.game.bots import clear_shot
from navigation import Lancer


class Probe(Lancer):
    drive = 1
    shoot = True

    def decide(self, obs):
        own, enemy = obs["self"], obs["enemy"]
        x, y = own["x"], own["y"]
        ex, ey = enemy["x"], enemy["y"]
        visible = clear_shot(self.walls, x, y, ex, ey)
        if visible:
            wx, wy = ex, ey
            speed = self.drive * .85
        else:
            wx, wy = self.waypoint(x, y, ex, ey)
            speed = min(.85, 4 * hypot(wx - x, wy - y))
        diff = (degrees(atan2(wy - y, wx - x)) - own["heading"] + 180) % 360 - 180
        turn = max(-1, min(1, diff / 18))
        return {"turn": turn, "forward": speed if abs(diff) < 50 else 0,
                "fire": self.shoot and visible and abs(diff - turn * 18) < 2}


class Approach(Probe):
    pass


class SilentApproach(Probe):
    shoot = False


class Retreat(Probe):
    drive = -1
