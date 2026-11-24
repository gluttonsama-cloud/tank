"""可直接用于本地对战与提交包预检的起始策略。"""
from math import atan2, degrees

from seedcup import Bot


class Starter(Bot):
    def decide(self, obs):
        me, enemy = obs["self"], obs["enemy"]
        heading = degrees(atan2(enemy["y"] - me["y"], enemy["x"] - me["x"]))
        delta = (heading - me["heading"] + 180) % 360 - 180
        return {"forward": 0.6, "turn": max(-1, min(1, delta / 18)), "fire": abs(delta) < 18}
