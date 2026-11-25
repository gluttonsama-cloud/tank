"""Controlled on/off windows using the same movement as Approach."""
from seedcup.game import rules

from paired import Approach


class Pulse(Approach):
    def decide(self, obs):
        action = super().decide(obs)
        phase = obs["frame"] // (2 * rules.RELOAD_TICKS)
        return action | {"fire": action["fire"] and phase % 2 == 0}
