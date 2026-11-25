import shutil
from pathlib import Path

import pytest
from seedcup.game import Game, GameMap, rules
from seedcup.game.maps import Spawn
from seedcup.game.replay import ReplayRecorder

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
