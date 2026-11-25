import json
import shutil
import time
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from seedcup.game.bots import BaselineBot

from tank import jobs, versions
from tank.simulation import run_match
from tank.training import TrainingStore
from tank.workspace import read_json, write_json
from training import fit_boss_specialist as fit


def test_default_budget_and_reproducible_disjoint_map_allocation():
    settings = fit.Settings()
    assert settings.generations * settings.population * settings.maps_per_generation * 2 == 2880
    splits = fit.map_splits(2026112531, settings)
    assert splits == fit.map_splits(2026112531, settings)
    seeds = [seed for generation in splits["train"] for seed in generation] + splits["validation"] + splits["holdout"]
    assert len(seeds) == len(set(seeds)) == 96
    assert all(0 <= seed < 2**31 for seed in seeds)
    assert len({fit.generate_map(seed).grid for seed in seeds}) == len(seeds)


def test_cem_retains_the_elite_and_keeps_search_variance():
    settings = replace(fit.Settings(), population=4, elites=2)
    rng = np.random.default_rng(42)
    mean = np.asarray(fit.DEFAULT_PARAMS)
    sigma = np.full_like(mean, settings.initial_sigma)
    first = fit.population(rng, 1, settings, mean, sigma, None)
    elite = next(iter(first.items()))
    second = fit.population(rng, 2, settings, mean, sigma, elite)
    assert second[elite[0]] == elite[1]
    assert len(set(second.values())) == settings.population
    assert all(-1 <= param <= 1 for vector in second.values() for param in vector)
    assert tuple(fit.DEFAULT_PARAMS) not in second.values()
    repeated = [elite[1]] * settings.elites
    next_mean, next_sigma = fit.refit(mean, np.full_like(mean, settings.sigma_floor), repeated, settings)
    assert next_mean == pytest.approx(np.asarray(elite[1]) * settings.smoothing)
    assert next_sigma == pytest.approx(settings.sigma_floor)


def test_training_inputs_allow_one_frozen_file_of_each_kind():
    assert fit.training_inputs({"inputs": []}) == {}
    forecast = {"path": "inputs/000/forecast.json", "sha256": "abc"}
    weights = {"path": "inputs/001/weights.json", "sha256": "def"}
    assert fit.training_inputs({"inputs": [forecast, weights]}) == {"forecast.json": forecast, "weights.json": weights}
    with pytest.raises(ValueError, match="重复 weights.json"):
        fit.training_inputs({"inputs": [weights, weights]})
    with pytest.raises(ValueError, match="未知训练输入"):
        fit.training_inputs({"inputs": [{"path": "unknown.json"}]})


def _warm_input(folder, content):
    encoded = json.dumps(content).encode("utf-8")
    (folder / "weights.json").write_bytes(encoded)
    return {"path": "weights.json", "sha256": sha256(encoded).hexdigest()}


@pytest.mark.parametrize(("field", "value", "error"), [
    ("param_names", [], "param_names"),
    ("params", [0.0], "params"),
    ("params", [float("nan"), *fit.DEFAULT_PARAMS[1:]], "有限数值"),
    ("params", [float("inf"), *fit.DEFAULT_PARAMS[1:]], "有限数值"),
    ("params", [1.01, *fit.DEFAULT_PARAMS[1:]], "有限数值"),
    ("params", [True, *fit.DEFAULT_PARAMS[1:]], "有限数值"),
])
def test_warm_weights_require_matching_parameter_names_and_bounded_numbers(tmp_path, field, value, error):
    content = {"param_names": fit.PARAM_NAMES, "params": fit.DEFAULT_PARAMS} | {field: value}
    with pytest.raises(ValueError, match=error):
        fit.load_warm_params(tmp_path, _warm_input(tmp_path, content))


def test_warm_file_must_match_the_frozen_hash(tmp_path):
    record = _warm_input(tmp_path, {"param_names": fit.PARAM_NAMES, "params": fit.DEFAULT_PARAMS})
    assert fit.load_warm_params(tmp_path, record) == fit.DEFAULT_PARAMS
    with pytest.raises(ValueError, match="SHA-256"):
        fit.load_warm_params(tmp_path, record | {"sha256": "modified"})


def _scored_games(tasks):
    for name, _, seed, side in tasks:
        opponent = dict.fromkeys(fit.SCORE_FIELDS, 0)
        score = opponent | {"total": int(name != fit.WARM)}
        stats = {"shots": 0, "hits": 0}
        yield {"candidate": name, "seed": seed, "side": side, "score": score, "opponent_score": opponent,
               "stats": stats, "opponent_stats": stats, "win": None}


def test_warm_search_uses_center_keeps_the_candidate_and_prefers_it_on_a_tie(tmp_path):
    settings = replace(fit.Settings(), generations=2, population=4, elites=2, finalists_per_generation=1)
    warm = (.7,) * len(fit.PARAM_NAMES)
    finalists, history = fit.run_search(_scored_games, tmp_path, [[1], [2]], 42, settings, warm)
    first = {entry["id"]: entry["params"] for entry in history[0]["candidates"]}
    expected = np.clip(np.random.default_rng(42).normal(warm, settings.initial_sigma), -1, 1)
    assert first[fit.WARM] == finalists[fit.WARM] == warm
    assert first["g01-p01"] == pytest.approx(expected)
    assert fit.WARM not in {entry["id"] for entry in history[1]["candidates"]}
    assert len(set(finalists.values())) == len(finalists)

    score = {"score": {"total": 5}}
    assert fit.rank_finalists({"new": score, fit.WARM: score, fit.BASE: score}) == [fit.WARM, "new"]


def test_identical_warm_and_base_share_holdout_episodes(tmp_path):
    control = {fit.WARM: fit.DEFAULT_PARAMS, "new": (.7,) * len(fit.PARAM_NAMES), fit.BASE: fit.DEFAULT_PARAMS}
    metrics, episodes = fit.compare_candidates(_scored_games, tmp_path, "holdout", control, [1])
    assert set(metrics) == set(control)
    assert episodes == 4
    assert metrics[fit.WARM] == metrics[fit.BASE]
    assert fit.paired_delta(metrics[fit.WARM], metrics[fit.BASE]) == 0


def _make_snapshot(root, name, change=""):
    source = root / "strategies" / name
    source.mkdir(parents=True)
    (source / "bot.py").write_text(
        "from seedcup.game.bots import BaselineBot\n\nclass FrozenBot(BaselineBot):\n    pass\n" + change,
        encoding="utf-8",
    )
    write_json(source / "submission.json", {"runtime": "python", "entrypoint": "bot.py:FrozenBot"})
    return versions.create(root, name)["id"]


def test_reduced_official_job_pairs_sides_and_exports_a_standalone_bot(tmp_path, monkeypatch):
    monkeypatch.delenv("TANK_ROOT", raising=False)
    resource = tmp_path / "resource"
    resource.mkdir()
    (resource / "SHA256SUMS").touch()
    training = tmp_path / "training"
    training.mkdir()
    for name in ("fit_boss_specialist.py", "boss_specialist_policy.py"):
        shutil.copyfile(Path(fit.__file__).with_name(name), training / name)
    base = _make_snapshot(tmp_path, "base")
    opponent = _make_snapshot(tmp_path, "opponent", "# frozen opponent\n")
    config = TrainingStore(tmp_path).register_config(
        teacher_snapshot=base, opponent_pool={opponent: 1}, sampling_seed=42,
    )
    params = {"generations": 1, "population": 2, "elites": 1, "finalists_per_generation": 1,
              "maps_per_generation": 1, "validation_maps": 1, "holdout_maps": 1, "workers": 2}
    job_id = jobs.start(tmp_path, "fit_boss_specialist.py", config, [], params)["job_id"]
    deadline = time.monotonic() + 35
    while (result := jobs.status(tmp_path, job_id))["status"] in {"queued", "running"}:
        if time.monotonic() >= deadline:
            pytest.fail(f"训练作业超时 {jobs.logs(tmp_path, job_id)['lines']}")
        time.sleep(0.05)
    assert result["status"] == "completed", jobs.logs(tmp_path, job_id)["lines"]

    folder = Path(result["request"]).parent
    output = folder / "output"
    assert read_json(output / "plan.json")["snapshots"] == {"base": base, "opponent": opponent}
    assert result["metrics"]["train_episodes"] == result["metrics"]["holdout_episodes"] == 4
    assert result["metrics"]["validation_episodes"] == 4
    for stage in ("train-01", "validation", "holdout"):
        games = read_json(output / "episodes" / f"{stage}.json")
        assert len({game["seed"] for game in games}) == 1
        for candidate in {game["candidate"] for game in games}:
            assert {g["side"] for g in games if g["candidate"] == candidate} == {"red", "blue"}
    selection = read_json(output / "selection.json")
    selected = selection["selected"]
    assert selected == selection["ranking"][0]
    assert set(read_json(output / "holdout.json")["metrics"]) == {selected, fit.BASE}

    snapshot = result["snapshot_id"]
    versions.verify(tmp_path, snapshot)
    path = versions.snapshot_data(tmp_path, snapshot)
    assert read_json(path / "weights.json")["params"] == selection["params"][selected]
    assert (path / "stock/bot.py").read_bytes() == (versions.snapshot_data(tmp_path, base) / "bot.py").read_bytes()
    selected_game = next(game for game in games if game["candidate"] == selected and game["side"] == "red")
    bot = versions.load_bot(tmp_path, snapshot)
    try:
        match = run_match(bot, BaselineBot(), selected_game["seed"], selected_game["game_no"])
    finally:
        bot.close()
    assert match["result"]["scores"][0] == selected_game["score"]
