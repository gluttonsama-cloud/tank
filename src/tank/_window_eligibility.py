"""Shared first-shot qualification for sampling bots and replay analysis."""
from math import atan2, degrees, hypot

from seedcup.game.bots import clear_shot

CLOSE_CELLS = 3
MAX_POST_TURN_ERROR_DEG = 2


def heading_error(own, enemy):
    bearing = degrees(atan2(enemy[1] - own[1], enemy[0] - own[0]))
    return (bearing - own[2] + 180) % 360 - 180


def shot_opportunity(walls, own, enemy, *, turn_per_tick, reload_ticks):
    """State tuples are (x, y, heading, hp, alive); reload is in SDK ticks."""
    if not (own[4] and enemy[4]) or reload_ticks > 1:
        return None
    distance = hypot(own[0] - enemy[0], own[1] - enemy[1])
    if distance > CLOSE_CELLS:
        return None
    aim_error = heading_error(own, enemy)
    turn = max(-1, min(1, aim_error / turn_per_tick))
    error_after_turn = abs(aim_error - turn * turn_per_tick)
    if error_after_turn >= MAX_POST_TURN_ERROR_DEG:
        return None
    if not clear_shot(walls, *own[:2], *enemy[:2]):
        return None
    return {"distance": distance, "probe_heading_error_deg": aim_error,
            "post_turn_error_deg": error_after_turn, "turn": turn}
