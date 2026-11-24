import hashlib
from pathlib import Path
import shutil
import time

import pytest
from seedcup.game.bots import BaselineBot

from tank import jobs, versions
from tank.simulation import run_match
from tank.training import TrainingStore, collect_demonstrations
from tank.workspace import read_json


def _config(root):
    return TrainingStore(root).register_config(
        teacher_snapshot="baseline", opponent_pool={"baseline": 1}, sampling_seed=2026,
    )


def _wait(root, job_id):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        result = jobs.status(root, job_id)
        if result["status"] not in ("queued", "running"):
            return result
        time.sleep(0.05)
    pytest.fail(f"训练作业超时 {jobs.logs(root, job_id)['lines']}")


def test_train_in_background_and_use_exported_snapshot(tmp_path):
    source = Path(__file__).resolve().parents[1] / "training"
    shutil.copytree(source, tmp_path / "training", ignore=shutil.ignore_patterns("__pycache__"))
    demonstration = tmp_path / "demo.json"
    collected = collect_demonstrations(BaselineBot(), BaselineBot(), 42, 0, match_id="demo",
                                       sides=("red", "blue"), output=demonstration,
                                       snapshots={"red": "baseline", "blue": "v-" + "a" * 64})
    digest = hashlib.sha256(demonstration.read_bytes()).hexdigest()
    config = _config(tmp_path)
    job = jobs.start(tmp_path, "fit_demo.py", config, [demonstration], {"ridge": 0.05})
    demonstration.unlink()

    result = _wait(tmp_path, job["job_id"])

    assert result["status"] == "completed", jobs.logs(tmp_path, job["job_id"])["lines"]
    assert result["metrics"]["samples"] == sum(sample["side"] == "red" for sample in collected["samples"])
    request = read_json(Path(result["request"]))
    assert request["params"] == {"ridge": 0.05}
    assert request["config"]["teacher_snapshot"] == "baseline"
    assert request["inputs"][0]["sha256"] == digest
    assert {"fit_demo.py", "demo_policy.py"} <= request["sources"].keys()
    snapshot_id = result["snapshot_id"]
    checkpoint = TrainingStore(tmp_path).read_checkpoint(result["checkpoint_id"])
    assert checkpoint["config_id"] == config
    assert checkpoint["snapshot_id"] == snapshot_id
    assert Path(checkpoint["weights_path"]).read_bytes() == (
        versions.snapshot_data(tmp_path, snapshot_id) / "weights.npy"
    ).read_bytes()
    versions.verify(tmp_path, snapshot_id)
    bot = versions.load_bot(tmp_path, snapshot_id)
    try:
        assert len(run_match(bot, BaselineBot(), 18, 0)["scores"]) == 2
    finally:
        bot.close()
    assert jobs.list_jobs(tmp_path)[0]["job_id"] == job["job_id"]
    assert "训练均方误差" in jobs.logs(tmp_path, job["job_id"], 1)["lines"][0]


def test_training_failure_retains_exit_code_and_log(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    (training / "fail.py").write_text("print('training failed')\nraise SystemExit(7)\n", encoding="utf-8")
    job = jobs.start(tmp_path, "fail.py", _config(tmp_path), [], {})

    result = _wait(tmp_path, job["job_id"])

    assert result["status"] == "failed"
    assert result["error"] == "训练脚本退出码 7"
    assert "training failed" in jobs.logs(tmp_path, job["job_id"])["lines"]


def test_configure_exposes_teacher_and_weighted_opponents(tmp_path):
    from argparse import ArgumentParser

    parser = ArgumentParser()
    jobs.add_arguments(parser)
    args = parser.parse_args(["configure", "--teacher", "baseline", "--opponent", "baseline=3", "--seed", "42"])
    config_id = jobs.dispatch(tmp_path, args)["config_id"]
    config = TrainingStore(tmp_path).read_config(config_id)
    assert config["teacher_snapshot"] == "baseline"
    assert config["opponent_pool"] == {"baseline": 3}
    assert config["sampling_seed"] == 42
