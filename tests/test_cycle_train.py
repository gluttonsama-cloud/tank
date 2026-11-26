import os
import shutil
import sqlite3
from dataclasses import asdict

import pytest
from seedcup.game import generate_map

from tank import sampling_data, versions
from tank.workspace import read_json, write_json
from training import cycle_matches, cycle_protocol, cycle_train
from training.boss_specialist_policy import load_strategy


def test_series_uses_uncapped_damage_only_when_wins_tie(monkeypatch):
    def game(candidate, opponent, case):
        assert (candidate, opponent) == ("candidate", "opponent")
        # Five wins each, then a draw. The damage could exceed either capped score.
        winner = case.side if case.game_no < 6 else 1 - case.side if case.game_no < 11 else -1
        stats = [{"damage": 130 - 10 * (side == case.side)} for side in (0, 1)]
        return {"result": {"winner": winner, "stats": stats}}

    monkeypatch.setattr(cycle_matches, "_game", game)
    games = tuple(cycle_matches.GameCase(i, i % 2, 10, 11, i) for i in range(1, 12))
    trial = cycle_matches.Trial("frozen-style", "candidate", "opponent", games)
    result = cycle_matches.play(trial)
    assert result["wins"] == [5, 5]
    assert result["actual_damage"] == [1320, 1430]
    assert result["series_points"] == 0
    assert cycle_matches.series_verdict([6, 5], [10, 500]) == 1
    assert cycle_matches.series_verdict([5, 5], [500, 500]) == .5


def test_bo11_conditions_pair_every_map_with_swapped_opening():
    cases = cycle_train._conditions({"style": "frozen"}, 8, "bo11", cycle_train._seed_source(34))
    assert len(cases) == 16
    for (style, first), (second_style, second) in zip(cases[::2], cases[1::2], strict=True):
        assert style == second_style == "style"
        assert len(first) == len(second) == 11
        assert [g.side for g in first] == [i % 2 for i in range(11)]
        for a, b in zip(first, second, strict=True):
            assert (a.map_seed, a.game_no, a.contestant_seed, a.opponent_seed) == (
                b.map_seed, b.game_no, b.contestant_seed, b.opponent_seed)
            assert a.side == 1 - b.side


def test_pve_cross_line_new_plan_uses_paired_bo11_and_legacy_remains_single_game():
    plan = {"pve": {"sdk-rookie": "sdk:rookie"}, "pvp": {"lancer-v7": "lancer", "router-v2": "router"}}
    report = {"line": "pve", "incumbent": "previous", "map_seed_namespace": 55, "inputs": plan}
    _, _, legacy = cycle_train._report_cases(report)
    assert len(legacy) == 16 and all(len(games) == 1 for _, games in legacy)
    plan["cross_series_groups"] = 8
    _, _, paired = cycle_train._report_cases(report)
    assert len(paired) == 32 and all(len(games) == 11 for _, games in paired)


def test_gameplay_identity_ignores_transfer_plans_and_captures_series_rules(tmp_path):
    for file in cycle_protocol.GAME_NODES:
        shutil.copyfile(cycle_protocol.SOURCE / file, tmp_path / file)
    original = cycle_protocol._gameplay_sources(tmp_path)
    runner = tmp_path / "cycle_matches.py"
    source = runner.read_text(encoding="utf-8")
    runner.write_text(source + "\ndef unrelated_test_schedule():\n    return 5\n", encoding="utf-8")
    assert cycle_protocol._gameplay_sources(tmp_path) == original
    runner.write_text(source.replace("max(wins) == 6", "max(wins) == 5"), encoding="utf-8")
    assert cycle_protocol._gameplay_sources(tmp_path) != original


def test_bootstrap_resamples_group_means():
    result = cycle_matches.paired_bootstrap({"style-a": [1.] * 8, "style-b": [0.] * 8}, 67)
    assert result["mean_delta"] == .5
    assert result["ci95"] == [.5, .5]


def test_series_bootstrap_rejects_mixed_conditions_and_uses_swapped_opening_pairs():
    def raw(games, points):
        return {"style": "incumbent", "games": [asdict(game) for game in games], "series_points": points}

    cases = cycle_train._conditions({"incumbent": "frozen"}, 8, "bo11", cycle_train._seed_source(34))
    candidate = [raw(games[:6], 1.) for _, games in cases]
    incumbent = [raw(games[:11], 0.) for _, games in cases]
    pooled, direct = cycle_matches.paired_series(candidate, incumbent, candidate, 54)
    assert pooled["group_deltas"] == {"incumbent": [1.] * 8}
    assert direct["ci95"] == [.5, .5]
    incumbent[0]["games"][0]["map_seed"] += 1
    with pytest.raises(ValueError, match="配对地图"):
        cycle_matches.paired_series(candidate, incumbent, candidate, 54)


def test_single_ability_snapshot_can_be_applied_to_official_shaped_bundle(tmp_path):
    source = tmp_path / "policy"
    source.mkdir()
    base = source / "bot.py"
    base.write_text("from seedcup import Bot\nclass Duelist(Bot):\n    def decide(self, obs):\n"
                    "        return {'forward': 1, 'turn': 0, 'fire': False}\n", encoding="utf-8")
    write_json(source / "submission.json", {"runtime": "python", "entrypoint": "bot.py:Duelist"})
    original = versions.create_from_directory(tmp_path, source, "original")["id"]
    cycle_train._ability(source, .3)
    candidate = versions.create_from_directory(tmp_path, source, "candidate")["id"]
    versions.verify(tmp_path, candidate)
    assert load_strategy(versions.snapshot_data(tmp_path, candidate)).__name__ == "CyclePolicy"
    assert versions.compare(tmp_path, original, candidate)["modified"] == ["submission.json"]
    assert read_json(source / "cycle_ability.json") == {cycle_train.ABILITY: .3}
    cycle_train._ability(source, .3)
    assert versions.create_from_directory(tmp_path, source, "candidate")["id"] == candidate


def test_official_focus_is_only_mined_into_train(tmp_path, monkeypatch):
    seed = 1034911
    grid = sampling_data.sdk_layout_hash({"map": {"grid": generate_map(seed).rows()}})
    focus = [{"event": "zone_displacement", "target": "boss", "entries": [
        {"game_id": 2, "split": "blind", "grid_sha256": grid, "map_seed": seed},
        {"game_id": 4, "split": "train", "grid_sha256": grid, "map_seed": seed},
    ]}]
    monkeypatch.setattr(cycle_train.cycle_experience, "summary", lambda *_, **__: {"training_focus": focus})
    feedback = cycle_train._feedback(tmp_path, "pve", 4)
    cases = cycle_train._feedback_cases(feedback, "pve", {"v31": "frozen"}, 1)
    assert [(style, game[0].side, game[0].map_seed) for style, game in cases] == [
        ("v31", 0, seed), ("v31", 1, seed)]


def test_worker_capacity_respects_pause_and_runtime_limit(tmp_path):
    out = tmp_path / "round"
    write_json(out / "workers.json", {"workers": 4})
    assert cycle_train._capacity(tmp_path, "pvp", out, 8) == 4
    db_path = tmp_path / ".local/cycle/control.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE lines(line TEXT PRIMARY KEY,enabled INTEGER)")
        db.execute("INSERT INTO lines VALUES('pvp',0)")
    with pytest.raises(cycle_train.TrainingPaused):
        cycle_train._capacity(tmp_path, "pvp", out, 8)


def test_pause_then_resume_only_runs_unfinished_pairs(tmp_path, monkeypatch):
    game_ids = []

    def play(trial):
        game, = trial.games
        game_ids.append(game.map_seed)
        return {"style": trial.style, "games": [asdict(game)]}

    class Pool:
        def map(self, function, trials, *, chunksize):
            return map(function, trials)

    def pause():
        if game_ids:
            raise cycle_train.TrainingPaused
        return 1

    monkeypatch.setattr(cycle_matches, "play", play)
    out = tmp_path / "checkpoint.json"
    trials = [cycle_matches.Trial("style", "sdk:rookie", "sdk:baseline", (cycle_matches.GameCase(i, 0, 1, 2, 1),))
              for i in range(3)]
    with pytest.raises(cycle_train.TrainingPaused):
        cycle_matches.run_trials(Pool(), trials, pause, out)
    assert game_ids == [0]
    assert cycle_matches.run_trials(Pool(), trials, lambda: 2, out) == read_json(out)["trials"]
    assert game_ids == [0, 1, 2]
    with pytest.raises(ValueError, match="策略快照、协议或条件"):
        cycle_matches.run_trials(Pool(), trials[::-1], lambda: 2, out)


def test_cache_binding_includes_actual_policy_bytes_and_protocol(tmp_path, monkeypatch):
    def play(trial):
        return {"style": trial.style, "games": [asdict(game) for game in trial.games]}

    class Pool:
        def map(self, function, trials, *, chunksize):
            return map(function, trials)

    monkeypatch.setattr(cycle_matches, "play", play)
    source = tmp_path / "policy"
    source.mkdir()
    script = source / "bot.py"
    script.write_text("EXAMPLE = 1", encoding="utf-8")
    game = cycle_matches.GameCase(112, 1, 100, 200, 1)
    trial = cycle_matches.Trial("style", str(source), "sdk:rookie", (game,))
    log = tmp_path / "log.json"
    cycle_matches.run_trials(Pool(), [trial], lambda: 1, log, protocol={"runner": "a"})
    with pytest.raises(ValueError, match="策略快照、协议或条件"):
        cycle_matches.run_trials(Pool(), [trial], lambda: 1, log, protocol={"runner": "b"})
    script.write_text("EXAMPLE = 2", encoding="utf-8")
    with pytest.raises(ValueError, match="策略快照、协议或条件"):
        cycle_matches.run_trials(Pool(), [trial], lambda: 1, log, protocol={"runner": "a"})


def test_profile_timing_uses_actual_applied_sdk_action_reuse(tmp_path):
    obs = tmp_path / ".local/sampling/observations/21.json.gz"
    obs.parent.mkdir(parents=True)
    obs.touch()
    os.utime(obs, (110, 110))
    write_json(tmp_path / ".local/online/practice/version/rookie/21.json", {
        "result": {"seconds": 6}, "sdk_timing": {"confirmed_actions": 55, "reused_actions": 4}})
    assert cycle_train.sdk_action_lag(tmp_path, 100, 111)["reuse_rate"] == 4 / 55
    assert cycle_train.sdk_action_lag(tmp_path, 105, 111)["status"] == "not_observed"
