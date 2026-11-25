from copy import deepcopy
from math import inf

import pytest
from seedcup.game import Game, GameMap, rules
from seedcup.game.engine import STILL, Shell
from seedcup.game.maps import Spawn

from training.boss_local_aim import AimSettings, _trial, choose


class Branch:
    def __init__(self, game, side, opponent):
        self.game = game
        self.own_side = side
        self.opponent = opponent
        self.events = []

    def step(self, action):
        actions = [None, None]
        actions[self.own_side] = action
        actions[1 - self.own_side] = self.opponent(self.game.observe(1 - self.own_side))
        self.events = self.game.step(actions)
        return self.game


class World:
    def __init__(self, game, side=0, opponent=lambda obs: STILL):
        self.game, self.side, self.opponent = game, side, opponent
        self.branches = 0

    def branch(self):
        self.branches += 1
        return Branch(deepcopy(self.game), self.side, self.opponent)


def arena(own, enemy, side=0, walls=()):
    blocked = set(walls)
    grid = bytes(int((x, y) in blocked) for y in range(rules.H) for x in range(rules.W))
    spawns = (Spawn(*own), Spawn(*enemy))
    if side:
        spawns = spawns[::-1]
    return Game(GameMap(0, grid, sum(grid) / len(grid), 0, spawns))


@pytest.mark.parametrize("side", (0, 1))
def test_choose_fires_from_post_move_muzzle_without_mutating_world(side):
    game = arena((6.5, 8.5, 6.0), (8.5, 8.5, 180.0), side)
    game.tanks[side].reload = 1
    world = World(game, side, lambda obs: {"forward": 0.0, "turn": 1.0, "fire": True})
    action = choose(world, game.observe(side), STILL | {"forward": 1.0})

    assert action["fire"]
    assert game.tick == 0
    assert game.tanks[side].shots == 0
    branch = world.branch()
    branch.step(action)
    assert branch.game.tanks[side].hits == 1


def test_choose_models_enemy_responding_to_new_shell():
    game = arena((6.5, 8.5, 0.0), (13.5, 8.5, 90.0))

    def dodge(obs):
        return STILL | {"forward": 1.0} if obs["shells"] else STILL

    world = World(game, opponent=dodge)
    settings = AimSettings(budget_ms=500.0)
    assert _trial(world, STILL | {"fire": True}, 10, inf, None) is None
    action = choose(world, game.observe(0), STILL, settings)

    assert action["fire"]
    assert action["turn"] > 0
    assert _trial(world, action, 10, inf, None) >= settings.min_hit_margin


def test_choose_does_not_fire_into_wall():
    game = arena((5.5, 8.5, 0.0), (8.5, 8.5, 180.0), walls=[(7, y) for y in range(rules.H)])
    world = World(game)

    assert not choose(world, game.observe(0), STILL | {"fire": True})["fire"]


def test_hit_from_old_shell_does_not_validate_new_miss():
    game = arena((5.5, 8.5, 0.0), (8.5, 8.5, 180.0), walls=[(7, y) for y in range(rules.H)])
    game.shells = [Shell(1, 0, 8.0, 8.5, 1.0, 0.0)]
    game._shell_id = 2
    world = World(game)
    firing = STILL | {"fire": True}
    branch = world.branch()
    branch.step(firing)

    assert branch.game.tanks[0].hits == 1
    assert branch.game.tanks[0].shots == 1
    assert not branch.game.shells
    assert _trial(world, firing, 1, inf, None) is None


def test_prepare_turn_while_reloading_without_simulating_shots():
    game = arena((6.5, 8.5, 22.0), (8.5, 8.5, 180.0))
    game.tanks[0].reload = 2
    world = World(game)
    action = choose(world, game.observe(0), STILL)

    assert action["turn"] < 0
    assert not action["fire"]
    assert world.branches == 0
