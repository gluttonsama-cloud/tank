import json

import pytest
from seedcup import Bot
from seedcup.game import Game, GameMap, rules
from seedcup.game.engine import STILL, Shell
from seedcup.game.maps import Spawn
from seedcup.game.runner import match_info

from training.boss_specialist_policy import (
    DEFAULT_PARAMS,
    PARAM_NAMES,
    load_strategy,
    make_policy,
)


class Idle(Bot):
    def decide(self, obs):
        return STILL.copy()


def params(**overrides):
    return tuple(overrides.get(name, value) for name, value in zip(PARAM_NAMES, DEFAULT_PARAMS, strict=True))


def arena(own=(8.5, 12.5, 0), enemy=(15.5, 12.5, 180), walls=()):
    grid = bytes((x, y) in walls for y in range(rules.H) for x in range(rules.W))
    return Game(GameMap(7, grid, sum(grid) / len(grid), 0, (Spawn(*own), Spawn(*enemy))))


def start(game, **overrides):
    bot = make_policy(Idle, params(**overrides))()
    bot.on_match_start(match_info(game.map, 0, "test", game=0))
    return bot


def test_zero_policy_preserves_the_original_class_and_actions():
    class Stateful(Bot):
        def on_match_start(self, info):
            self.count = 0

        def decide(self, obs):
            self.count += 1
            return {"turn": self.count / 10, "forward": -0.3, "fire": self.count % 2 == 0}

    def unused_predictor(*_):
        pytest.fail("Zero residuals accessed the predictor")

    assert make_policy(Stateful, DEFAULT_PARAMS, predictor=unused_predictor) is Stateful
    baseline, candidate = Stateful(), make_policy(Stateful, DEFAULT_PARAMS)()
    for bot in (baseline, candidate):
        bot.on_match_start({})
    for tick in range(4):
        obs = {"frame": tick}
        assert candidate.decide(obs) == baseline.decide(obs)


@pytest.mark.parametrize("value", [1.1, float("nan")])
def test_parameters_are_validated_at_the_policy_boundary(value):
    with pytest.raises(ValueError, match="lead"):
        make_policy(Idle, params(lead=value))
    with pytest.raises(ValueError, match=f"Expected {len(PARAM_NAMES)}"):
        make_policy(Idle, ())


def test_lead_and_visible_trigger_change_the_shot_candidate():
    game = arena()
    bot = start(game, aim=1, lead=1, trigger=1)
    assert bot.decide(game.observe(0))["fire"] is True
    game.step([STILL, STILL])
    game.tanks[1].y += .25
    action = bot.decide(game.observe(0))
    assert action["turn"] > .15
    assert action["fire"] is True

    blocked = arena(walls={(11, 12)})
    blocked_bot = start(blocked, aim=1, lead=1, trigger=1)
    assert blocked_bot.decide(blocked.observe(0)) == STILL


def test_predictor_proposal_uses_observed_displacement_and_changes_aim():
    seen = []

    def predictor(obs, vx, vy, flight):
        seen.append((obs["frame"], vx, vy, flight))
        return obs["enemy"]["x"], obs["enemy"]["y"] - 1.0

    game = arena()
    bot = make_policy(Idle, params(aim=1, curvature=1), predictor=predictor)()
    info = match_info(game.map, 0, "test", game=0)
    bot.on_match_start(info)
    bot.decide(game.observe(0))
    game.step([STILL, STILL])
    game.tanks[1].y += .25
    assert bot.decide(game.observe(0))["turn"] < 0
    assert seen == [(0, 0, 0, 6), (1, 0, .25, 6)]


def test_dodge_avoids_a_visible_projectile_through_the_sdk():
    def with_shell():
        game = arena(own=(10.5, 12.5, 90))
        game.shells.append(Shell(1, 1, 6.9, 12.5, 1, 0))
        return game

    baseline, game = with_shell(), with_shell()
    bot = start(game, dodge=1)
    for _ in range(4):
        game.step([bot.decide(game.observe(0)), STILL])
        baseline.step([STILL, STILL])
    assert game.tanks[0].hp == rules.HP
    assert baseline.tanks[0].hp == rules.HP - rules.DAMAGE


def test_zone_movement_heads_toward_the_safe_area():
    game = arena(own=(12.85, 12.5, 0), enemy=(20.5, 12.5, 180))
    game.tick = rules.ZONE_END_TICKS
    bot = start(game, zone=1)
    action = bot.decide(game.observe(0))
    game.step([action, STILL])
    assert action["forward"] > 0
    assert game.tanks[0].x > 12.85


def test_load_strategy_keeps_same_named_snapshot_entries_distinct(tmp_path):
    def write_snapshot(value):
        folder = tmp_path / str(value)
        folder.mkdir()
        (folder / "submission.json").write_text(json.dumps({"runtime": "python", "entrypoint": "bot.py:Strategy"}))
        (folder / "bot.py").write_text(
            "from __future__ import annotations\n"
            "from dataclasses import dataclass\n"
            "from seedcup import Bot\n\n"
            "@dataclass\n"
            "class Strategy(Bot):\n"
            f"    value: float = {value}\n"
            "    def decide(self, obs):\n"
            "        return {'forward': self.value, 'turn': 0., 'fire': False}\n",
            encoding="utf-8",
        )
        return folder

    first_path = write_snapshot(.25)
    second_path = write_snapshot(.75)
    first, second = load_strategy(first_path), load_strategy(second_path)
    assert first is load_strategy(first_path)
    assert first().decide({})["forward"] == .25
    assert second().decide({})["forward"] == .75


def test_load_strategy_isolates_sibling_imports_throughout_the_match(tmp_path, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    def write_snapshot(name, value):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "submission.json").write_text(json.dumps({"runtime": "python", "entrypoint": "bot.py:Strategy"}))
        (folder / "snapshot_value.py").write_text(f"VALUE = {value}\n", encoding="utf-8")
        (folder / "snapshot_helper.py").write_text("from snapshot_value import VALUE\n", encoding="utf-8")
        (folder / "bot.py").write_text(
            "from importlib import import_module\n"
            "from seedcup import Bot\n"
            "from snapshot_helper import VALUE\n"
            "class Strategy(Bot):\n"
            "    def __init__(self):\n"
            "        self.helper = import_module('snapshot_helper')\n"
            "    def decide(self, obs):\n"
            "        import snapshot_helper\n"
            "        from . import snapshot_value\n"
            "        assert snapshot_helper is self.helper\n"
            "        assert snapshot_value.VALUE == VALUE\n"
            "        return {'forward': snapshot_helper.VALUE, 'turn': 0., 'fire': False}\n",
            encoding="utf-8",
        )
        return folder

    first = load_strategy(write_snapshot("base", .25))
    second = load_strategy(write_snapshot("sparring", .75))
    first_bot, second_bot = first(), second()
    assert first_bot.decide({})["forward"] == .25
    assert second_bot.decide({})["forward"] == .75
    assert first_bot.decide({})["forward"] == .25
    assert not list(tmp_path.rglob("*.pyc"))
