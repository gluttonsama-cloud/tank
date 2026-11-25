from collections import deque
from types import SimpleNamespace

import pytest
from seedcup.game import Game, GameMap, generate_map, rules
from seedcup.game.engine import STILL, Shell
from seedcup.game.maps import Spawn
from seedcup.game.runner import match_info

from training.boss_local_world import World

SHOOT = STILL | {"fire": True}
DRIVE = STILL | {"forward": 1.0}


class Replica:
    def __init__(self, info, action=STILL):
        self.info = info
        self.action = action
        self.features = SimpleNamespace(history=deque(maxlen=5), motion=deque(maxlen=20))
        self.evade = SimpleNamespace(frame=None)

    def decide(self, obs):
        self.features.history.append(obs["frame"])
        self.features.motion.append((obs["self"]["x"], obs["enemy"]["x"]))
        self.evade.frame = obs["frame"]
        return self.action.copy()


@pytest.mark.parametrize("own_side", [0, 1])
def test_world_restores_public_state_and_flip(own_side):
    game = Game(generate_map(715), tick=3)
    tank = game.tanks[own_side]
    tank.reload, tank.still, tank.collided = 4, 2, True
    game.last_actions[own_side] = SHOOT | {"frame": 2, "source_frame": 2}
    game.shells = [Shell(11, 0, 9.68, 7.11, 1, 0), Shell(18, 1, 23.34, 7.11, -1, 0)]
    info = match_info(game.map, own_side, "陪练", game=2)
    models = []

    def make_replica(enemy_info):
        model = Replica(enemy_info, DRIVE)
        models.append(model)
        return model

    world = World(info, make_replica)
    obs = game.observe(own_side)
    restored, enemy_action = world.update(obs)

    enemy_side = 1 - own_side
    expected_info = match_info(game.map, enemy_side, "本地策略", game=2)
    assert models[0].info == expected_info
    assert restored.observe(own_side) == obs
    assert enemy_action == DRIVE
    assert world.update(obs) == (restored, enemy_action)
    assert list(models[0].features.history) == [3]

    after = world.branch().step(STILL)
    assert after.last_actions[own_side]["forward"] == 0.0
    assert after.last_actions[enemy_side]["forward"] == 1.0


def test_branch_model_and_game_are_isolated():
    game = Game(generate_map(93))
    world = World(match_info(game.map, 0, "陪练"), Replica)
    root, _ = world.update(game.observe(0))

    moving = world.branch()
    stationary = world.branch()
    moving.step(DRIVE)
    saved = moving.fork()
    moving.step(DRIVE)
    stationary.step(STILL)
    saved.step(STILL)

    assert moving.game.tanks[0].x != saved.game.tanks[0].x
    assert stationary.observation()["self"]["x"] == root.tanks[0].x
    assert world.game.tick == 0
    assert list(world._model.features.history) == [0]
    assert list(moving._model.features.history) == list(saved._model.features.history) == [0, 1]
    assert list(stationary._model.features.history) == [0]
    assert moving._model.features.motion is not saved._model.features.motion
    assert world._model.evade.frame == stationary._model.evade.frame == 0
    assert moving._model.evade.frame == saved._model.evade.frame == 1


def test_official_start_packet_needs_only_the_public_grid():
    game = Game(generate_map(93))
    info = match_info(game.map, 0, "陪练")
    del info["map"]["seed"]
    world = World(info, Replica)
    world.update(game.observe(0))
    assert world.branch().step(STILL).map.grid == game.map.grid


@pytest.mark.parametrize(
    "own_x,wall,expect_visible", [(8.5, False, True), (10.5, False, False), (8.5, True, False)]
)
def test_enemy_reload_from_new_or_immediately_vanished_shell(own_x, wall, expect_visible):
    grid = bytes(int(wall and x == 11 and y == 5) for y in range(rules.H) for x in range(rules.W))
    game_map = GameMap(4, grid, sum(grid) / len(grid), -1, (Spawn(own_x, 5.5, 0), Spawn(12.0, 5.5, 180)))
    game = Game(game_map)
    predicted_action = SHOOT if wall else STILL
    world = World(match_info(game_map, 0, "陪练"), lambda info: Replica(info, predicted_action))
    world.update(game.observe(0))

    game.step([STILL, SHOOT])
    assert bool(game.shells) == expect_visible
    restored, _ = world.update(game.observe(0))
    assert restored.tanks[1].reload == rules.RELOAD_TICKS
    if expect_visible:
        assert restored.shells[0].owner == 1

    for tick in range(1, rules.RELOAD_TICKS):
        game.step([STILL, STILL])
        restored, _ = world.update(game.observe(0))
        assert restored.tanks[1].reload == rules.RELOAD_TICKS - tick
