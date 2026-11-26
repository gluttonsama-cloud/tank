import gzip
import json
from hashlib import sha256

import numpy as np
import pytest
from seedcup.game import generate_map, rules
from seedcup.game.replay import FORMAT_VERSION

from tank.online import PRACTICE_NAMES
from tank.sampling_data import layout_hash
from training.boss_forecast import make_predictor
from training.fit_boss_replay_forecast import (
    Samples,
    archived_replay,
    fit_candidates,
    grouped_entries,
    metrics,
    official_boss_side,
    replay_samples,
    weighted_games,
)
from training.fit_boss_sparring import replay_observation


def match():
    return {
        "id": 31,
        "kind": "room",
        "public": True,
        "room": "P123",
        "game": 1,
        "red": {"id": 57, "bot": False, "name": PRACTICE_NAMES["boss"]},
        "blue": {"id": None, "bot": True, "name": PRACTICE_NAMES["boss"]},
    }


def entry(match_id=31, seed=12, grid="A", split="train", side="blue", strategy="example"):
    return {
        "id": match_id,
        "seed": seed,
        "grid_sha256": grid,
        "split": split,
        "target_side": side,
        "version_id": strategy,
    }


def test_official_room_bot_requires_identity_and_resolves_both_sides():
    meta = match()
    assert official_boss_side(meta) == "blue"
    assert official_boss_side(meta | {"red": meta["blue"], "blue": meta["red"]}) == "red"
    assert official_boss_side(meta | {"kind": "ladder"}) is None
    assert official_boss_side(meta | {"public": False}) is None
    assert official_boss_side(meta | {"blue": meta["red"]}) is None
    assert official_boss_side(meta | {"blue": meta["blue"] | {"id": 58}}) is None
    assert official_boss_side(meta | {"red": meta["blue"]}) is None


def test_archive_binds_digest_map_and_target_side(tmp_path):
    root = tmp_path / ".local/online/replays"
    root.mkdir(parents=True)
    meta, game_map = match(), generate_map(12).to_dict()
    replay = {"match": meta, "map": game_map, "format": FORMAT_VERSION, "rules": rules.as_dict()}
    raw = gzip.compress(json.dumps(replay).encode())
    path = root / "31.json.gz"
    path.write_bytes(raw)
    source = entry(grid=layout_hash(game_map)) | {
        "target": "boss",
        "sha256": sha256(raw).hexdigest(),
        "players": {side: meta[side] for side in ("red", "blue")},
        "game_no": meta["game"],
    }
    assert archived_replay(tmp_path, source) == replay
    with pytest.raises(ValueError, match="31"):
        archived_replay(tmp_path, source | {"target_side": "red"})
    path.write_bytes(raw + b"bad digest")
    with pytest.raises(ValueError, match="SHA-256"):
        archived_replay(tmp_path, source)


def test_split_checks_seed_and_layout_and_reserves_the_layout_of_excluded_seeds():
    train = entry()
    for overlap in (entry(32, seed=13, split="blind"), entry(32, grid="B", split="blind")):
        with pytest.raises(ValueError, match="跨分组"):
            grouped_entries([train, overlap], set(), set())
    groups, skipped = grouped_entries(
        [train, entry(32, seed=13, split="blind"), entry(33, seed=20, grid="C", split="blind")], {12}, set()
    )
    assert skipped["reserved"] == 2
    assert not groups["train"]
    assert list(groups["blind"]) == ["C"]


def test_collection_exclusions_apply_when_reading_a_full_manifest():
    groups, skipped = grouped_entries([
        entry() | {"eligible_for_training": False},
        entry(32, grid="B", seed=13),
        entry(33, grid="C", seed=14, split="blind") | {"eligible_for_training": False},
    ], set(), set())
    assert skipped["collection_excluded"] == 1
    assert set(groups["train"]) == {"B"} and set(groups["blind"]) == {"C"}


def training_replay(change_future=False):
    game_map = generate_map(12).to_dict()
    game_map["grid"] = "0" * len(game_map["grid"])
    frames = []
    for tick in range(14):
        red = [5.0, 5.0, 0.0, 100.0, 1, 0, 0, 0, 0, 0, int(tick >= 2), 0]
        y = 5 + 0.1 * tick if not change_future or tick < 2 else 5 - 0.1 * tick
        blue = [9.0, y, 0.0, 100.0, 1, 0, 0, 0, 0, 0, int(tick >= 5), 0]
        frames.append([tick, red, blue, [[6.1, 5.0, 1.0, 0.0, 0]] if tick else [], []])
    return {"match": match(), "map": game_map, "frames": frames, "result": {"ticks": frames[-1][0]}}


def test_sample_uses_current_and_previous_observations_and_marks_next_tick_shot():
    original, altered = training_replay(), training_replay(change_future=True)
    a, b = replay_samples(original, "blue"), replay_samples(altered, "blue")
    assert a.features[0] == pytest.approx(b.features[0])
    assert a.features[0, 1] == pytest.approx(0.1 / (rules.SPEED * rules.DT))
    assert a.displacement[0, 1] > 0 > b.displacement[0, 1]
    assert a.shooter[0] == pytest.approx((-4.0, -0.1))
    assert np.flatnonzero(a.fired).tolist() == [0]
    switched = replay_samples(original, "red")
    assert switched.features[0, 1] == 0
    assert np.flatnonzero(switched.fired).tolist() == [3]


def test_match_weights_average_repeats_sides_and_strategies():
    shots = Samples(
        np.zeros((2, 15)), np.zeros((2, 2)), np.ones((2, 1)), np.ones((2, 2)), np.array([True, False])
    )
    silent = Samples(shots.features, shots.displacement, shots.reach, shots.shooter, np.zeros(2, dtype=bool))
    games = {
        ("A", "a", "red"): [({"id": 1}, silent), ({"id": 2}, shots)],
        ("A", "a", "blue"): [({"id": 3}, shots)],
        ("A", "b", "red"): [({"id": 4}, silent)],
        ("A", "b", "blue"): [({"id": 5}, silent)],
        ("B", "b", "red"): [({"id": 6}, shots)],
        ("B", "b", "blue"): [({"id": 7}, shots)],
    }
    weights = {e["id"]: w for e, _, w in weighted_games(games)}
    assert weights == {1: 0.125, 2: 0.125, 3: 0.25, 4: 0.25, 5: 0.25, 6: 0.5, 7: 0.5}
    firing_weights = {e["id"]: w for e, _, w in weighted_games(games, fired_only=True)}
    assert firing_weights == {2: 0.5, 3: 0.5, 6: 0.5, 7: 0.5}


def test_fit_and_metrics_use_the_exported_runtime_predictions():
    replay = training_replay()
    blue = replay_samples(replay, "blue")
    red = replay_samples(replay, "red")
    games = {("A", "one", "blue"): [(entry(), blue)], ("A", "one", "red"): [(entry(32, side="red"), red)]}
    n = blue.features.shape[1]
    projection, bias, choices = fit_candidates(
        games,
        {"projection": np.zeros((n, 1)), "bias": [0.0], "coef": np.zeros((n + 2, 2))},
    )
    p = make_predictor({"projection": projection, "bias": bias, "coef": choices["ridge_0.1"]})
    report = metrics(games, projection, bias, {"fit": choices["ridge_0.1"]})
    obs = replay_observation(replay["frames"][1], 0)
    h = round(blue.reach[0, 0] / (rules.SPEED * rules.DT))
    ex, ey = p(obs, 0.0, 0.1, h)
    future = replay["frames"][1 + h][2][:2]
    red_obs = replay_observation(replay["frames"][4], 1)
    i = np.flatnonzero(red.fired)[0]
    h = round(red.reach[i, 0] / (rules.SPEED * rules.DT))
    ax, ay = p(red_obs, 0.0, 0.0, h)
    red_target = replay["frames"][4 + h][1][:2]
    shot_errors = [np.hypot(ex - future[0], ey - future[1]), np.hypot(ax - red_target[0], ay - red_target[1])]
    assert report["fit"]["fired"]["samples"] == 2
    assert report["fit"]["fired"]["mae_cells"] == pytest.approx(np.mean(shot_errors))
