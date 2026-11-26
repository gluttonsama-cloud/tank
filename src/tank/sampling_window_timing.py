"""Capture the official SDK's acknowledgement of the first eligible probe command."""
from seedcup.game import rules
from seedcup.game.bots import wall_grid

from ._window_eligibility import shot_opportunity


class WindowTiming:
    def __init__(self, match):
        self.walls = wall_grid(match)
        self.record = {"source_frame": None, "command_fire": None, "first_applied_frame": None,
                       "last_applied_frame": None, "confirmed_uses": 0, "applied_fire": None}

    def observe(self, message, action):
        trace = self.record
        if trace["source_frame"] is None:
            own, enemy = (tuple(message[side][k] for k in ("x", "y", "heading", "hp", "alive"))
                          for side in ("self", "enemy"))
            eligible = shot_opportunity(self.walls, own, enemy, turn_per_tick=rules.TURN_RATE * rules.DT,
                                        reload_ticks=round(message["self"]["reload"] * rules.TICK_HZ))
            if eligible is not None:
                trace.update(source_frame=message["frame"], command_fire=action["fire"])
            return
        applied = message.get("last_action")
        if (applied and applied.get("source_frame") == trace["source_frame"]
                and applied["frame"] != trace["last_applied_frame"]):
            if trace["confirmed_uses"] == 0:
                trace.update(first_applied_frame=applied["frame"], applied_fire=applied["fire"])
            trace.update(last_applied_frame=applied["frame"], confirmed_uses=trace["confirmed_uses"] + 1)
