from math import atan2, degrees

import pytest
from seedcup.game import Game, GameMap, generate_map, rules
from seedcup.game.engine import Shell
from seedcup.game.maps import Spawn
from seedcup.game.replay import ReplayRecorder, save
from tank.workspace import write_json

from training.boss_behavior_matches import matched_games, verify_unseen
from training.compare_boss import official_cases
from training.evaluate_boss_behavior import ReplayTrace, branch, evaluate_trace, static_intercept
from training.fit_boss_sparring import BOSS_NAME


IDLE = {"forward": 0, "turn": 0, "fire": False}
DRIVE = IDLE | {"forward": 1}


def open_map(wall=None):
    grid = bytes((x, y) == wall for y in range(rules.H) for x in range(rules.W))
    return GameMap(91, grid, 0, 0, (Spawn(10, 10, 90), Spawn(20, 10, 180)))


def replay_with_incoming_shell(action):
    game_map = open_map()
    meta = {"id": 123, "game": 0, "red": {"bot": True, "name": BOSS_NAME},
            "blue": {"bot": False, "name": "对手"}}
    recorder = ReplayRecorder(game_map, meta)
    game = Game(game_map, shells=[Shell(1, 1, 5.5, 10, 1, 0)])
    recorder.add(game.snapshot(), [])
    for _ in range(24):
        effects = game.step([action, IDLE])
        recorder.add(game.snapshot(), effects)
    return ReplayTrace(recorder.to_dict({"ticks": 24}))


class StillBot:
    def on_match_start(self, info):
        pass

    def decide(self, observation):
        return IDLE


def test_event_tracks_one_shell_and_scores_actual_and_simulated_outcomes():
    trace = replay_with_incoming_shell(DRIVE)
    events, _ = trace.threats()
    assert len(events) == 1
    assert events[0]["tick"] == 0 and events[0]["reference_hit"] is False
    points, details = branch(trace, StillBot, 0, (5, 10, 20), events, 0)
    assert points[5]["position_error_cells"] == pytest.approx(1.25)
    assert details[0]["predicted_hit"] is True
    assert details[0]["reference_perpendicular_cells"] == pytest.approx(1.25)
    assert details[0]["predicted_perpendicular_cells"] == 0
    recovered, calibrated = branch(trace, None, 0, (5, 10, 20), events, 0)
    assert recovered[20]["position_error_cells"] == pytest.approx(0)
    assert recovered[20]["signed_range_error_cells"] == pytest.approx(0)
    assert calibrated[0]["predicted_hit"] is False

    struck = replay_with_incoming_shell(IDLE)
    hits, coverage = struck.threats()
    assert len(hits) == 1 and hits[0]["reference_hit"] is True
    assert coverage["reference_hits_unassigned_to_observed_shell"] == 0


def test_threat_test_accounts_for_wall():
    shell = Shell(1, 1, 5.5, 10.5, 1, 0)
    assert static_intercept(open_map(), shell, 10, 10.5) is not None
    assert static_intercept(open_map(wall=(8, 10)), shell, 10, 10.5) is None


def test_continuous_graze_between_substeps_is_not_a_threat():
    game_map = open_map()
    shell = Shell(1, 1, 9.9, 10.41, 1, 0)
    game = Game(game_map, shells=[shell])
    game.tanks[0].x = 11
    assert static_intercept(game_map, shell, 11, 10) is None
    for _ in range(5):
        game.step([IDLE, IDLE])
    assert game.tanks[0].hp == rules.HP


def test_new_shot_hit_is_not_attributed_to_old_shell_sparking_nearby():
    wall_map = open_map(wall=(10, 10))
    heading = degrees(atan2(-0.6, 0.8)) % 360
    game_map = GameMap(91, wall_map.grid, 0, 0, (Spawn(9.80, 9.61, 0), Spawn(9.39, 10.43, heading)))
    meta = {"id": 124, "game": 0, "red": {"bot": True, "name": BOSS_NAME},
            "blue": {"bot": False, "name": "对手"}}
    recorder = ReplayRecorder(game_map, meta)
    game = Game(game_map, shells=[Shell(1, 1, 8.65, 10.02, 1, 0)], _shell_id=2)
    recorder.add(game.snapshot(), [])
    actions = [(IDLE | {"forward": 0.96}, IDLE), (IDLE, IDLE | {"fire": True}), *[(IDLE, IDLE)] * 3]
    for pair in actions:
        effects = game.step(list(pair))
        recorder.add(game.snapshot(), effects)
    replay = recorder.to_dict({"ticks": 5})
    assert [effect[0] for effect in replay["frames"][2][4]] == ["fire", "spark", "hit"]
    trace = ReplayTrace(replay)
    events, coverage = trace.threats()
    assert len(events) == 1 and events[0]["reference_hit"] is False
    assert coverage["reference_hits_unassigned_to_observed_shell"] == 1
    assert coverage["reference_ambiguous_impact_frames"] == 0
    _, predicted = branch(trace, None, 0, (5,), events, 0)
    assert predicted[0]["predicted_hit"] is False

    # Missing spark data makes the remaining hit unattributable to the old projectile.
    replay["frames"][2][4].pop(1)
    uncertain, coverage = ReplayTrace(replay).threats()
    assert len(uncertain) == 1 and uncertain[0]["reference_hit"] is None
    assert coverage["reference_ambiguous_impact_frames"] == 1


def test_every_sample_restarts_from_recorded_state_with_past_only():
    trace = replay_with_incoming_shell(DRIVE)
    instances = []

    class RecordingBot(StillBot):
        def on_match_start(self, info):
            self.calls = []
            instances.append(self.calls)

        def decide(self, observation):
            self.calls.append((observation["frame"], observation["self"]["y"]))
            return IDLE

    samples, _, _ = evaluate_trace(trace, RecordingBot, stride_frames=10, history_frames=3)
    assert [row["tick"] for row in samples if row["horizon_ticks"] == 5] == [0, 10]
    assert [t for t, _ in instances[1][:6]] == [7, 8, 9, 10, 11, 12]
    assert [y for _, y in instances[1][:6]] == [11.75, 12, 12.25, 12.5, 12.5, 12.5]


def add_practice(root, candidate, replay_id, side, game_map):
    write_json(root / ".local/online/practice" / candidate / "boss" / f"{replay_id}.json", {
        "id": replay_id, "seed": game_map["seed"], "side": side,
    })
    replay = {"match": {"id": replay_id, "game": ("red", "blue").index(side)}, "map": game_map}
    save(root / ".local/online/replays" / f"{replay_id}.json.gz", replay)
    return replay


def test_compare_uses_only_selected_ids_and_one_game_per_side(tmp_path):
    game_map = generate_map(42).to_dict()
    for replay_id, side in ((101, "red"), (102, "blue"), (103, "red")):
        add_practice(tmp_path, "candidate", replay_id, side, game_map)
    cases = official_cases(tmp_path, ["candidate"], [42], replay_ids=[103, 102])
    assert [played[0][0]["id"] for played in cases.values()] == [102, 103]
    legacy = official_cases(tmp_path, ["candidate"], [42])
    assert len(legacy["candidate", 42, 0]) == 2
    with pytest.raises(ValueError, match="只能指定一局"):
        official_cases(tmp_path, ["candidate"], [42], replay_ids=[101, 102, 103])
    with pytest.raises(ValueError, match="未匹配.*104"):
        official_cases(tmp_path, ["candidate"], [42], replay_ids=[101, 104])


def test_compare_checks_unseen_before_running_and_checks_grid(tmp_path, monkeypatch):
    from tank import versions

    monkeypatch.setattr(versions, "verify", lambda root, model: None)
    monkeypatch.setattr(versions, "snapshot_data", lambda root, model: root / model)
    game_map = generate_map(42).to_dict()
    save(tmp_path / ".local/online/replays/100.json.gz", {"map": game_map})
    write_json(tmp_path / "model/training.json", {"train": [], "holdout": [{"id": 100, "seed": 42}]})
    for replay_id, side in ((101, "red"), (102, "blue")):
        add_practice(tmp_path, "candidate", replay_id, side, game_map)
    with pytest.raises(ValueError, match="训练或选模数据重复"):
        matched_games(tmp_path, ["model"], ["candidate"], [42], replay_ids=[101, 102], require_unseen=True)

    same_grid = {"match": {"id": 201}, "map": game_map | {"seed": 43}}
    with pytest.raises(ValueError, match="回放 201 .*训练或选模数据重复"):
        verify_unseen(tmp_path, "model", [same_grid])
