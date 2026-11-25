import json
from pathlib import Path

import numpy as np
import pytest
from seedcup.game import Game, GameMap, generate_map, rules
from seedcup.game.bots import wall_grid
from seedcup.game.engine import STILL, Shell
from seedcup.game.maps import Spawn
from seedcup.game.replay import FORMAT_VERSION, ReplayRecorder
from seedcup.game.runner import match_info


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "training"))
    import fit_boss_sparring

    return fit_boss_sparring


def test_forward_labels_follow_sdk_wall_sliding(trainer):
    grid = bytes(x == 5 and y == 5 for y in range(rules.H) for x in range(rules.W))
    game_map = GameMap(1, grid, 0, 0, (Spawn(4.61, 5.5, 0), Spawn(25.5, 5.5, 180)))
    idle = {"forward": 0, "turn": 0, "fire": False}
    game = Game(game_map)
    _, own, enemy, _ = game.snapshot()
    game.step([idle | {"forward": 1}, idle])
    _, next_own, _, _ = game.snapshot()
    assert game.observe(0)["collision"]
    assert np.isnan(trainer.recover_forward(game_map, own, enemy, next_own))

    game = Game(game_map)
    game.tanks[0].h = 45
    _, own, enemy, _ = game.snapshot()
    game.step([idle | {"forward": 1}, idle])
    _, next_own, _, _ = game.snapshot()
    assert game.observe(0)["collision"]
    assert next_own[0] == own[0] and next_own[1] > own[1]
    assert trainer.recover_forward(game_map, own, enemy, next_own) == pytest.approx(1, abs=0.03)

    game.step([idle, idle])
    _, stationary, _, _ = game.snapshot()
    assert trainer.recover_forward(game_map, next_own, enemy, stationary) == 0


@pytest.mark.parametrize("feature_name", ["Features", "ManeuverFeatures"])
def test_extract_matches_observations_and_blue_action_timing(trainer, feature_name):
    game_map = generate_map(73129)
    game = Game(game_map)
    game.tanks[1].h = 5
    meta = {"id": 123, "red": {"bot": False, "name": "red"},
            "blue": {"bot": True, "name": trainer.BOSS_NAME}}
    recorder = ReplayRecorder(game_map, meta)
    feature_type = getattr(trainer, feature_name)
    features = feature_type(match_info(game_map, 1, ""))
    expected = []
    idle = {"forward": 0, "turn": 0, "fire": False}
    recorder.add(game.snapshot(), [])
    for tick in range(rules.RELOAD_TICKS + 1):
        expected.append(features(game.observe(1)))
        fire = tick in (0, rules.RELOAD_TICKS)
        # The current-frame heading crosses zero on the first action.
        action = {"forward": 0, "turn": -1, "fire": fire}
        fx = game.step([idle, action])
        recorder.add(game.snapshot(), fx)
    x, y = trainer.extract(recorder.to_dict({"ticks": rules.RELOAD_TICKS + 1}), feature_type)
    np.testing.assert_allclose(x, expected, atol=1e-6)
    np.testing.assert_allclose(y[:, 1], 1, atol=0.01)
    assert y[0, 2] == y[-1, 2] == 1
    assert np.isnan(y[1:-1, 2]).all()


def test_split_rejects_map_leakage_and_unlisted_replays(trainer, tmp_path):
    def add(replay_id, seed, grid):
        path = tmp_path / f"{replay_id}.json"
        replay = {"match": {"id": replay_id}, "format": FORMAT_VERSION,
                  "rules": rules.as_dict(), "map": {"seed": seed, "grid": grid}}
        path.write_text(json.dumps(replay), encoding="utf-8")
        return {"path": path.name, "sha256": "digest"}

    train, holdout = add(1, 42, "A"), add(2, 43, "B")
    request = {"params": {"train_ids": [1], "holdout_ids": [2]}, "inputs": [train, holdout]}
    splits = trainer.load_splits(tmp_path, request)
    assert [[replay["match"]["id"] for replay, _ in split] for split in splits] == [[1], [2]]
    add(2, 44, "A")
    with pytest.raises(ValueError, match="不同地图"):
        trainer.load_splits(tmp_path, request)
    with pytest.raises(ValueError, match="回放 ID"):
        trainer.load_splits(tmp_path, request | {"inputs": [train]})


def test_decode_checks_reload_and_reachable_aim(trainer):
    ready = 1 / rules.RELOAD_TICKS
    # [normalized reload, target turn, hit margin, clear shot]
    features = np.array([[ready, -.7, .1, 1], [2 * ready, -.7, .1, 1],
                         [ready, 1.5, .1, 1], [ready, -.7, .1, 0]])
    scores = np.tile([0, -1, 3, .2, .8], (len(features), 1))
    forward, turn, fire = trainer.decode(features, scores, {"speeds": np.array([-1, 0, 1])})
    np.testing.assert_allclose(forward, [1, 1, 1, 1])
    np.testing.assert_allclose(turn, [-.7, -.34, 1, .2])
    assert fire.tolist() == [True, False, False, False]


def test_maneuver_turn_stays_in_selected_mode(trainer):
    features = np.array([[0.5, -.7, .1, 1], [0.5, -.7, .1, 1]])
    scores = np.zeros((2, 21))
    scores[0, 0], scores[1, 14] = 3, 3
    scores[:, 15], scores[:, 19] = .9, -.9
    weights = {"speeds": np.array([-1, 0, 1]), "turn_centers": np.array([-1, -.2, 0, .2, 1])}
    forward, turn, fire = trainer.decode(features, scores, weights)
    np.testing.assert_allclose(forward, [-1, 1])
    np.testing.assert_allclose(turn, [-.9, .9])
    assert not fire.any()


@pytest.mark.parametrize("covered", [False, True])
def test_evasion_only_changes_a_threatened_action(trainer, covered):
    from boss_sparring_policy import EVADE_TICKS, EvadeIncoming

    grid = bytes(covered and x == 8 and y == 12 for y in range(rules.H) for x in range(rules.W))
    game_map = GameMap(1, grid, 0, 0, (Spawn(10, 12.5, 90), Spawn(20, 12.5, 180)))
    game = Game(game_map, shells=[Shell(1, 1, 7, 12.5, 1, 0)])
    info = match_info(game_map, 0, "")
    evade = EvadeIncoming(info, wall_grid(info))
    action = evade(game.observe(0), STILL)
    assert (action == STILL) == covered
    for _ in range(EVADE_TICKS):
        game.step([action, STILL])
    assert game.tanks[0].hp == rules.HP
