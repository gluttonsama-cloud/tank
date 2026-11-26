from dataclasses import replace

import numpy as np
import pytest
from seedcup.game import Game, generate_map, rules
from seedcup.game.engine import STILL, Shell
from seedcup.game.maps import Spawn
from seedcup.game.runner import match_info

from training.sdk_motion_data import iter_motion_intervals, iter_views, match_context, motion_rows
from training.sdk_motion_policy import FEATURE_NAMES, MovementFeatures, movement_features


def recorded_game(blue_actions, *, wall=None, blue_spawn=Spawn(13.5, 6.5, 180.), shells=()):
    grid = bytearray(rules.W * rules.H)
    if wall is not None:
        x, y = wall
        grid[y * rules.W + x] = 1
    game_map = replace(generate_map(12), grid=bytes(grid), spawns=(Spawn(4.5, 6.5, 0.), blue_spawn))
    game = Game(game_map, shells=list(shells))
    replay = {"match": {"red": {"name": "probe"}, "blue": {"name": "boss"}, "game": 0},
              "map": game_map.to_dict(), "frames": [], "result": {"ticks": len(blue_actions)}}
    capture = {"start": match_info(game_map, 0, "boss", game=0), "frames": [],
               "end": {"result": replay["result"]}}
    for action in (*blue_actions, None):
        replay["frames"].append(game.snapshot() + [[]])
        capture["frames"].append([game.observe(0), STILL])
        if action is not None:
            game.step([STILL, action])
    return replay, capture


def test_replay_and_capture_recover_boss_actions_and_match_live_features():
    commands = [STILL | {"forward": .6, "turn": -.2}, STILL, STILL | {"forward": -.8}, STILL]
    replay, capture = recorded_game(commands, shells=(Shell(1, 0, 9.7, 6.5, 1., 0.),
                                                      Shell(2, 1, 16., 6.5, 1., 0.)))
    features, actions, threat = motion_rows(replay, "replay", "blue")
    sdk_features, sdk_actions, sdk_threat = motion_rows(capture, "sdk", "blue")
    np.testing.assert_allclose(features, sdk_features, atol=.04)
    np.testing.assert_allclose(actions, sdk_actions, atol=.04)
    np.testing.assert_array_equal(threat, sdk_threat)
    assert actions == pytest.approx(np.asarray(((.6, -.2), (0., 0.), (-.8, 0.))), abs=.04)
    assert threat[0]

    _, first_obs = next(iter_views(capture, "sdk", "blue"))
    assert first_obs["self"]["heading"] == 180.
    assert [shell["mine"] for shell in first_obs["shells"]] == [False, True]
    geometry = MovementFeatures(match_context(capture, "sdk", "blue"))
    for row, (_, obs, prev, _) in zip(sdk_features, iter_motion_intervals(capture, "sdk", "blue"), strict=True):
        np.testing.assert_array_equal(row, movement_features(obs, prev, geometry=geometry))


def test_motion_labels_exclude_blocked_standstill_gaps_and_terminal_transitions():
    wall = (7, 6)
    spawn = Spawn(6.61, 6.5, 0.)
    replay, capture = recorded_game([STILL | {"forward": 1.}, *([STILL | {"forward": -1.}] * 2),
                                    STILL, STILL, STILL], wall=wall, blue_spawn=spawn)
    capture["frames"].pop(2)
    assert [i for i, *_ in iter_motion_intervals(replay, "replay", "blue")] == [1, 2, 3, 4]
    intervals = list(iter_motion_intervals(capture, "sdk", "blue"))
    assert [i for i, *_ in intervals] == [3, 4]
    assert intervals[0][2] is None
    x, y, threat = motion_rows(capture, "sdk", "blue")
    assert x[:, FEATURE_NAMES.index("has_previous")].tolist() == [0, 1]
    np.testing.assert_array_equal(y, np.zeros((2, 2)))
    assert not threat.any()


def test_death_of_either_tank_masks_adjacent_intervals():
    _, capture = recorded_game([STILL] * 4)
    for tank in ("self", "enemy"):
        capture["frames"][2][0][tank]["alive"] = False
        assert [i for i, *_ in iter_motion_intervals(capture, "sdk", "blue")] == [0]
        capture["frames"][2][0][tank]["alive"] = True


def test_wall_between_shell_and_tank_blocks_threat():
    replay, _ = recorded_game([STILL] * 2, wall=(11, 6), shells=(Shell(1, 0, 9., 6.5, 1., 0.),))
    _, obs = next(iter_views(replay, "replay", "blue"))
    threat_index = FEATURE_NAMES.index("threat")
    assert movement_features(obs, None)[threat_index] == 1
    geometry = MovementFeatures(match_context(replay, "replay", "blue"))
    assert geometry.extract(obs, None)[1] is False
