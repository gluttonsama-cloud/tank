import shutil
from math import atan2, degrees, hypot
from pathlib import Path

import pytest
from seedcup import Bot
from seedcup.game import Game, GameMap, parse_action, rules
from seedcup.game.maps import Spawn
from seedcup.game.replay import FORMAT_VERSION, ReplayRecorder
from seedcup.game.runner import match_info

from tank import sampling_probes, versions
from tank.simulation import run_match


@pytest.fixture(scope="module")
def frozen_probes(tmp_path_factory):
    root = tmp_path_factory.mktemp("sampling")
    repo = Path(__file__).resolve().parents[1]
    for name in ("strategies/lancer", "strategies/sampling-probes"):
        shutil.copytree(repo / name, root / name)
    paired = "training/boss_practice_probe.py"
    (root / paired).parent.mkdir()
    shutil.copyfile(repo / paired, root / paired)
    return root, sampling_probes.snapshots(root, sampling_probes.catalog())


def test_probe_snapshots_are_distinct_and_reusable(frozen_probes):
    root, ids = frozen_probes
    assert versions.describe(root, ids["recontact"])["entrypoint"] == "recontact.py:Recontact"
    assert len(set(ids.values())) == len(ids)
    assert ids == sampling_probes.snapshots(root, ids)
    for version_id in ids.values():
        versions.verify(root, version_id)
    with pytest.raises(ValueError, match="invalid"):
        sampling_probes.snapshots(root, ["invalid"])


@pytest.fixture(scope="module")
def recordings(frozen_probes):
    root, ids = frozen_probes
    cases = {}
    for name, version_id in ids.items():
        probe = versions.load_bot(root, version_id)
        try:
            game = run_match(versions.load_bot(root, "baseline"), probe, 712101, 0, replay=True)
        finally:
            probe.close()
        cases[name] = game["replay"]
    return cases


def test_sdk_matches_exercise_catalog_tactics(recordings):
    counts = {name: sampling_probes.coverage(replay, "red") for name, replay in recordings.items()}
    approach, silent, retreat = (counts[name] for name in ("approach", "silent", "retreat"))
    assert approach["probe"]["shots"] > 0
    assert approach["probe"]["visible_approach_ticks"] > 0
    assert silent["probe"]["shots"] == 0
    assert silent["probe"]["visible_approach_ticks"] > 0
    assert retreat["probe"]["shots"] > 0
    assert retreat["probe"]["visible_retreat_ticks"] > 0
    assert approach["pair"]["close_ticks"] > retreat["pair"]["close_ticks"]
    assert counts["peek"]["pair"]["los_acquired"] >= 2
    assert counts["peek"]["pair"]["los_lost"] >= 2
    assert counts["peek"]["probe"]["shots"] > 0
    assert counts["lancer"]["probe"]["shots"] > 0
    late = counts["late-zone"]
    assert late["pair"]["shrink_ticks"] > 0 and late["pair"]["after_shrink_ticks"] > 0
    assert late["probe"]["outside_zone_ticks"] > 0
    assert late["probe"]["zone_exits"] == late["probe"]["zone_entries"] == 1
    assert late["probe"]["shots"] == 0


def test_coverage_counts_follow_explicit_target_side(recordings):
    replay = recordings["silent"]
    red = sampling_probes.coverage(replay, "red")
    blue = sampling_probes.coverage(replay, "blue")
    assert red["target"] == blue["probe"]
    assert blue["target"] == red["probe"]
    assert red["pair"] == blue["pair"]
    assert red["target"]["shots"] > 0
    assert blue["target"]["shots"] == red["target"]["hits_received"] == 0
    assert blue["target"]["hits_received"] == replay["frames"][-1][1][9]
    assert red["target"]["enemy_shell_ticks"] == 0
    assert blue["target"]["enemy_shell_ticks"] > 0
    with pytest.raises(ValueError, match="target_side"):
        sampling_probes.coverage(replay, "boss")

    batch = sampling_probes.aggregate_coverage((red, blue))
    assert batch["games"] == 2 and batch["ticks"] == red["ticks"] * 2
    assert batch["totals"]["target"]["shots"] == red["target"]["shots"]
    assert batch["games_with"]["target"]["shots"] == 1


def test_pulse_shots_stop_during_evenly_spaced_windows(recordings):
    replay = recordings["pulse"]
    window = round(2 * replay["rules"]["reload_seconds"] * replay["rules"]["tick_hz"])
    active_phases = {
        before[0] // window for before, after in zip(replay["frames"], replay["frames"][1:])
        if after[2][10] > before[2][10]
    }
    assert len(active_phases) >= 2
    assert all(phase % 2 == 0 for phase in active_phases)
    assert sampling_probes.coverage(replay, "red")["target"]["enemy_shell_ticks"] > 0


def test_projectile_conditioned_movement_uses_previous_frame_and_enemy_owner():
    game_map = GameMap(90, bytes(rules.W * rules.H), 0, 0,
                       (Spawn(10, 10, 359), Spawn(20, 10, 180)))
    recorder = ReplayRecorder(game_map, {})
    game = Game(game_map)
    recorder.add(game.snapshot(), [])
    idle = {"forward": 0, "turn": 0, "fire": False}
    turn_and_drive = idle | {"forward": 1, "turn": 1}
    pairs = [(idle, idle | {"fire": True}), (turn_and_drive, idle), (idle, idle)]
    for pair in pairs:
        effects = game.step(list(pair))
        recorder.add(game.snapshot(), effects)

    replay = recorder.to_dict(None)
    counts = sampling_probes.coverage(replay, "red")
    assert counts["probe"]["enemy_shell_ticks"] == 0
    assert counts["target"]["enemy_shell_ticks"] == 2
    assert counts["target"]["turn_ticks_with_enemy_shell"] == 1
    assert counts["target"]["lateral_move_ticks_with_enemy_shell"] == 1


class StationaryTarget(Bot):
    def decide(self, observation):
        return {"forward": 0, "turn": 0, "fire": False}


class TracedProbe(Bot):
    def __init__(self, probe):
        self.probe = probe
        self.inputs = []
        self.actions = []

    def on_match_start(self, info):
        self.probe.on_match_start(info)

    def decide(self, observation):
        action = self.probe.decide(observation)
        self.inputs.append(observation)
        self.actions.append({key: action[key] for key in ("turn", "forward", "fire")})
        return action

    def on_match_end(self, result):
        self.probe.on_match_end(result)


@pytest.mark.parametrize("seed", [712101, 123])
@pytest.mark.parametrize("probe_side", ["red", "blue"])
def test_window_pair_shares_observations_and_actions_until_first_opportunity(frozen_probes, seed, probe_side):
    root, versions_by_probe = frozen_probes
    runs = {}
    for name in ("window-fire", "window-hold"):
        probe = versions.load_bot(root, versions_by_probe[name])
        traced = TracedProbe(probe)
        target = StationaryTarget()
        bots = (traced, target) if probe_side == "red" else (target, traced)
        try:
            runs[name] = traced, run_match(*bots, seed, 0, replay=True)["replay"]
        finally:
            probe.close()

    fire, fire_replay = runs["window-fire"]
    hold, hold_replay = runs["window-hold"]
    anchor = sampling_probes.intervention_anchor(hold_replay, probe_side)
    assert anchor == sampling_probes.intervention_anchor(fire_replay, probe_side)
    frame = anchor["frame"]
    assert frame > 0
    assert 0 < anchor["distance"] <= 3
    assert anchor["post_turn_error_deg"] < 2
    assert fire.inputs[:frame + 1] == hold.inputs[:frame + 1]
    assert fire.actions[:frame] == hold.actions[:frame]
    assert fire.actions[frame] == hold.actions[frame] | {"fire": True}
    assert [i for i, action in enumerate(fire.actions) if action["fire"]] == [frame]
    assert not any(action["fire"] for action in hold.actions)
    assert fire_replay["frames"][:frame + 1] == hold_replay["frames"][:frame + 1]
    index = 1 if probe_side == "red" else 2
    assert fire_replay["frames"][frame + 1][index][10] == 1
    assert hold_replay["frames"][-1][index][10] == 0


def test_window_anchor_uses_pre_action_state_even_when_remote_reuses_action():
    game_map = GameMap(90, bytes(rules.W * rules.H), 0, 0,
                       (Spawn(10, 10, 0), Spawn(12.5, 10, 0)))
    game = Game(game_map)
    game.tanks[0].reload = 2
    recorder = ReplayRecorder(game_map, {"kind": "room"})
    recorder.add(game.snapshot(), [])
    idle = {"forward": 0, "turn": 0, "fire": False}
    effects = game.step([idle, idle])
    recorder.add(game.snapshot(), effects)
    assert sampling_probes.intervention_anchor(recorder.to_dict(None), "red") is None

    effects = game.step([idle, idle], source_frames=[0, 1])
    recorder.add(game.snapshot(), effects)
    assert game.observe(0)["last_action"]["source_frame"] == 0
    replay = recorder.to_dict(None)
    observed = sampling_probes.intervention_anchor(replay, "red")
    assert observed["frame"] == 1
    assert observed["probe_reload_ticks"] == 1
    assert observed["source"] == "replay_observation"
    assert observed["target_retreat_lanes"] == ["straight", "clockwise", "counterclockwise"]
    assert sampling_probes.intervention_anchor(replay, "blue") is None


def test_holdout_orbit_follows_alternating_directions_in_sdk(frozen_probes):
    root, ids = frozen_probes
    game_map = GameMap(90, bytes(rules.W * rules.H), 0, 0,
                       (Spawn(10, 10, 90), Spawn(15, 10, 180)))
    game = Game(game_map)
    target = StationaryTarget()
    probe = versions.load_bot(root, ids["holdout-orbit"])
    signed_motion = []
    try:
        probe.on_match_start(match_info(game_map, 0, "static"))
        for _ in range(8 * rules.TICK_HZ):
            own, enemy = game.tanks
            x, y = own.x, own.y
            observation = game.observe(0)
            action, violations = parse_action(probe.decide(observation))
            assert violations == 0
            game.step([action, target.decide(game.observe(1))])
            signed_motion.append((x - enemy.x) * (own.y - y) - (y - enemy.y) * (own.x - x))
    finally:
        probe.close()
    assert sum(signed_motion[:3 * rules.TICK_HZ]) < -5
    assert sum(signed_motion[4 * rules.TICK_HZ:7 * rules.TICK_HZ]) > 5
    assert game.tanks[0].shots == 2


class WithdrawingTarget(Bot):
    def on_match_start(self, info):
        self.hp = info["rules"]["hp"]
        self.withdraw_until = 0

    def decide(self, observation):
        own, enemy = observation["self"], observation["enemy"]
        dx, dy = enemy["x"] - own["x"], enemy["y"] - own["y"]
        diff = (degrees(atan2(dy, dx)) - own["heading"] + 180) % 360 - 180
        tick = observation["frame"]
        if own["hp"] < self.hp and hypot(dx, dy) <= 3:
            self.withdraw_until = tick + rules.RELOAD_TICKS
        self.hp = own["hp"]
        return {"forward": -1 if tick < self.withdraw_until and abs(diff) < 30 else 0,
                "turn": max(-1, min(1, diff / (rules.TURN_RATE * rules.DT))), "fire": False}


def _recontact_replay(frozen_probes, seed, probe_side, target):
    root, ids = frozen_probes
    probe = versions.load_bot(root, ids["recontact"])
    bots = (probe, target) if probe_side == "red" else (target, probe)
    try:
        return run_match(*bots, seed, 0, replay=True)["replay"]
    finally:
        probe.close()


@pytest.mark.parametrize("seed", [712101, 123])
@pytest.mark.parametrize("probe_side", ["red", "blue"])
def test_recontact_drives_close_separate_close_again_in_sdk(frozen_probes, seed, probe_side):
    replay = _recontact_replay(frozen_probes, seed, probe_side, StationaryTarget())
    probe_index = 1 if probe_side == "red" else 2
    target_side = "blue" if probe_side == "red" else "red"
    distances = [hypot(r[0] - b[0], r[1] - b[1]) for _, r, b, _, _ in replay["frames"]]
    close = next(i for i, distance in enumerate(distances) if distance < 3)
    separated = next(i for i in range(close + 1, len(distances)) if distances[i] > 5)
    assert any(distance < 3 for distance in distances[separated + 1:])
    assert replay["frames"][-1][probe_index][9] >= 2  # Probe hit the target twice.
    assert replay["frames"][-1][probe_index][11] == 0  # Valid SDK actions.
    assert sampling_probes.behavior_signals(replay, target_side) == {
        "reengage": True,
        "boss_withdraw": False,
        "disadvantaged_close": True,
        "late_close": False,
        "after_hit_withdraw": False,
    }
    assert sampling_probes.behavior_signals(replay, probe_side)["boss_withdraw"]


@pytest.mark.parametrize("probe_side", ["red", "blue"])
def test_recontact_exercises_target_withdrawal_after_sdk_projectile_hit(frozen_probes, probe_side):
    replay = _recontact_replay(frozen_probes, 123, probe_side, WithdrawingTarget())
    target_side = "blue" if probe_side == "red" else "red"
    signals = sampling_probes.behavior_signals(replay, target_side)
    assert signals["reengage"]
    assert signals["disadvantaged_close"]
    assert signals["after_hit_withdraw"]


def _behavior_frame(tick, red_x=10, blue_x=13, *, red_hp=100, blue_hits=0):
    # SDK tank fields are position, heading, HP, alive, reload, still, repair,
    # damage, hits, shots and violations.
    red = [red_x, 10, 0, red_hp, int(red_hp > 0), 0, 0, 0, 0, 0, 0, 0]
    blue = [blue_x, 10, 180, 100, 1, 0, 0, 0, 10 * blue_hits, blue_hits, blue_hits, 0]
    return [tick, red, blue, [], []]


def _behavior_signals(frames, side="red"):
    replay = {"format": FORMAT_VERSION, "rules": {"tick_hz": rules.TICK_HZ}, "frames": frames}
    return sampling_probes.behavior_signals(replay, side)


@pytest.mark.parametrize("target_gain, target_withdraws", [(0, False), (.99, False), (1, True)])
def test_behavior_uses_net_contribution_from_the_requested_side(target_gain, target_withdraws):
    frames = [
        _behavior_frame(tick, round(10 - target_gain * tick / 10, 2),
                        round(13 + (2 - target_gain) * tick / 10, 2))
        for tick in range(11)
    ]
    assert not _behavior_signals(frames)["reengage"]
    for tick in range(9, -1, -1):
        index, red, blue, _, _ = frames[tick]
        frames.append(_behavior_frame(20 - index, red[0], blue[0]))

    assert _behavior_signals(frames)["reengage"]
    assert _behavior_signals(frames)["boss_withdraw"] is target_withdraws
    assert _behavior_signals(frames, "blue")["boss_withdraw"]


def test_behavior_durations_use_alive_ticks_and_strictly_after_60_seconds():
    # Five separate near ticks, another five near ticks after a short break.
    frames = [
        _behavior_frame(tick, blue_x=13 if 1 <= tick <= 5 or 7 <= tick <= 11 else 15, red_hp=90)
        for tick in range(611)
    ]
    assert not _behavior_signals(frames[:11])["disadvantaged_close"]
    assert _behavior_signals(frames[:12])["disadvantaged_close"]
    assert not _behavior_signals(frames, "blue")["disadvantaged_close"]
    assert not _behavior_signals(frames[:610])["late_close"]
    assert _behavior_signals(frames)["late_close"]
    frames[-1] = _behavior_frame(610, blue_x=15, red_hp=0)
    assert not _behavior_signals(frames)["late_close"]


@pytest.mark.parametrize("far_tick, hits, expected", [(31, [1], True), (32, [1], False),
                                                      (32, [1, 16], True), (31, [], False)])
def test_behavior_after_hit_withdraw_uses_projectile_hit_and_response_deadline(far_tick, hits, expected):
    frames = [
        _behavior_frame(tick, red_x=10 - .25 * max(0, tick - far_tick + 8),
                        red_hp=100 if tick == 0 else 90,
                        blue_hits=sum(hit <= tick for hit in hits))
        for tick in range(far_tick + 1)
    ]
    signals = _behavior_signals(frames)
    assert signals["boss_withdraw"]
    assert signals["after_hit_withdraw"] is expected


def test_behavior_after_hit_withdraw_requires_target_movement_after_impact():
    frames = [_behavior_frame(0, blue_x=12.75)]
    frames.extend(
        _behavior_frame(tick + 1, 9.75 - tick * .075, 12.75 + tick * .125, red_hp=90, blue_hits=1)
        for tick in range(11)
    )
    assert not _behavior_signals(frames)["after_hit_withdraw"]
