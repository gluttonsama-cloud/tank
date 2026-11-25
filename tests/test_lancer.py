import pytest
from seedcup.game import Game, generate_map, rules
from seedcup.game.engine import Shell
from seedcup.game.runner import match_info

from strategies.lancer.bot import Lancer


@pytest.mark.parametrize("side", [0, 1])
def test_lancer_dodges_toward_safety_at_the_zone_boundary(side):
    game_map = generate_map(1200500144)
    game = Game(game_map)
    game.tick = rules.ZONE_END_TICKS
    me, enemy = game.tanks[side], game.tanks[1 - side]
    me.x, me.y, me.h = (13.08, 12.45, 0) if side == 0 else (18.92, 12.45, 180)
    enemy.x, enemy.y = rules.ZONE_CX, rules.ZONE_CY
    dx = -1 if side == 0 else 1
    game.shells = [Shell(0, enemy.side, me.x - 2.5 * dx, me.y, dx, 0)]
    bot = Lancer()
    bot.on_match_start(match_info(game_map, side, "对手", game=0))

    actions = [{"forward": 0, "turn": 0, "fire": False}] * 2
    actions[side] = bot.decide(game.observe(side))
    start_x = me.x
    game.step(actions)

    assert abs(me.x - rules.ZONE_CX) < abs(start_x - rules.ZONE_CX)
    assert me.zone_damage == 0
