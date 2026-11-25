from seedcup.game import Game, generate_map, rules
from seedcup.game.runner import match_info

from training import fit_boss_local as fit
from training.boss_local_policy import prospective_shell
from training.boss_specialist_policy import Arena


def test_metrics_count_real_shots_and_penalize_inactivity():
    game = {
        "hits": 10,
        "shots": 10,
        "win": True,
        "score": 200.0,
        "hp": 68.0,
        "damaged": 4,
        "zone_damage": 0.0,
        "clean": False,
        "kill": True,
        "seconds": 40.0,
    }
    r = fit.summary([game, game | {"hits": 1, "shots": 2, "kill": False, "seconds": 90.0}])
    assert r["hit_rate"] == 11 / 12
    assert r["gun_wins"] == 1 and r["gun_win_seconds"] == 40.0
    assert fit.summary([game | {"hits": 1, "shots": 1}])["utility"] < fit.summary([game])["utility"]
    assert fit.summary([game | {"shots": 0, "hits": 0}])["hit_rate"] == 0.0


def test_holdout_requires_accuracy_damage_wins_and_health():
    reference = {"hit_rate": 0.28, "hits_per_game": 8.5, "wins": 90, "hp": 48}
    candidate = reference | {"hit_rate": 0.90, "hp": 49}
    assert fit.meets_target(candidate, reference)
    for change in ({"hit_rate": 0.899}, {"hits_per_game": 7.9}, {"wins": 89}, {"hp": 48}):
        assert not fit.meets_target(candidate | change, reference)


def test_anticipated_barrel_uses_simultaneous_sdk_turn_and_motion():
    from seedcup.game.engine import STILL

    game = Game(generate_map(777))
    arena = Arena(match_info(game.map, 0, "test"), route_home=False)
    own, foe = game.tanks
    action = {"forward": 1.0, "turn": 0.25, "fire": True}
    shell = prospective_shell(arena, foe, (foe.x, foe.y), action, (own.x, own.y))
    events = game.step([STILL, action])
    fire = next(e for e in events if e["type"] == "fire")
    assert (round(shell["x"], 2), round(shell["y"], 2)) == (fire["x"], fire["y"])
    assert shell["mine"] is False and rules.TICK_HZ == 10
