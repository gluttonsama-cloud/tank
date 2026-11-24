import hashlib

import pytest
from seedcup import Bot

from tank import versions
from tank.simulation import FrameEnvironment
from tank.training import TrainingStore, collect_demonstrations
from tank.workspace import read_json, write_json


class ReloadBot(Bot):
    def __init__(self, fire_on_reload):
        self.fire_on_reload = fire_on_reload

    def decide(self, obs):
        frame = obs["frame"]
        obs["self"]["reload"] = -1
        return {"forward": 0, "turn": 0, "fire": frame == 0 or (frame == 1 and self.fire_on_reload)}


def test_demonstrations_save_intended_fire_during_reload(tmp_path):
    frame_samples = []
    for fire in (True, False):
        path = tmp_path / f"fire-{fire}.json"
        demo = collect_demonstrations(
            ReloadBot(fire), ReloadBot(False), 42, 0,
            match_id=f"fire-{fire}", sides=("red",), output=path,
        )
        stored = read_json(path)
        assert stored == demo
        assert stored["match_info"]["red"]["game"] == 0
        assert set(sample["side"] for sample in stored["samples"]) == {"red"}
        sample = stored["samples"][1]
        assert sample["match_id"] == f"fire-{fire}"
        assert sample["frame"] == sample["observation"]["frame"] == 1
        assert sample["observation"]["self"]["reload"] > 0
        assert sample["action"] == {"forward": 0, "turn": 0, "fire": fire}
        assert stored["result"]["stats"][0]["shots"] == 1
        frame_samples.append(sample)

    assert frame_samples[0]["observation"] == frame_samples[1]["observation"]


def _snapshot(root, name):
    directory = root / "strategies" / name
    directory.mkdir(parents=True)
    (directory / "bot.py").write_text(
        f"# {name}\n"
        "from seedcup import Bot\n"
        "class Strategy(Bot):\n"
        "    def decide(self, obs):\n"
        "        return {'forward': 0, 'turn': 0, 'fire': False}\n",
        encoding="utf-8",
    )
    write_json(directory / "submission.json", {"runtime": "python", "entrypoint": "bot.py:Strategy"})
    return versions.create(root, name)["id"]


def test_checkpoints_trace_fixed_teacher_and_different_sampled_opponents(tmp_path):
    teacher = _snapshot(tmp_path, "teacher")
    sparring = _snapshot(tmp_path, "sparring")
    store = TrainingStore(tmp_path)
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"checkpoint")

    for opponent in (teacher, sparring):
        config_id = store.register_config(teacher_snapshot=teacher, opponent_pool={opponent: 1}, sampling_seed=7)
        env = FrameEnvironment()
        opening, recorded = store.start_episode(env, config_id, episode_index=0, seed=42, game_no=0,
                                               controlled_side="red")
        checkpoint_id = store.register_checkpoint(config_id, weights, step=100, snapshot_id=teacher)

        reopened = TrainingStore(tmp_path)
        checkpoint = reopened.read_checkpoint(checkpoint_id)
        config = reopened.read_config(checkpoint["config_id"])
        assert checkpoint["weights_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
        assert checkpoint["snapshot_id"] == config["teacher_snapshot"] == teacher
        assert config["opponent_pool"] == {opponent: 1}
        assert reopened.read_episode(opening.match_id) == recorded
        assert recorded["config_id"] == checkpoint["config_id"]
        assert recorded["opponent_snapshot"] == opponent


def test_pool_configuration_and_sampling_survive_reordering_and_restart(tmp_path):
    sparring = _snapshot(tmp_path, "sparring")
    store = TrainingStore(tmp_path)
    first = store.register_config(teacher_snapshot=None, opponent_pool={sparring: 4, "baseline": 1}, sampling_seed=22)
    second = store.register_config(teacher_snapshot=None, opponent_pool={"baseline": 1, sparring: 4}, sampling_seed=22)
    assert first == second
    settings = {"seed": 42, "game_no": 0, "controlled_side": "blue", "episode_index": 20}
    _, first_draw = store.start_episode(FrameEnvironment(), first, **settings)
    _, repeated_draw = TrainingStore(tmp_path).start_episode(FrameEnvironment(), second, **settings)
    assert first_draw["opponent_snapshot"] == repeated_draw["opponent_snapshot"]
    assert first_draw["opponent_snapshot"] in {sparring, "baseline"}
    assert first_draw["match_id"] != repeated_draw["match_id"]


def test_config_rejects_an_unregistered_snapshot(tmp_path):
    with pytest.raises(ValueError, match="teacher_snapshot"):
        TrainingStore(tmp_path).register_config(
            teacher_snapshot="v-" + "0" * 64,
            opponent_pool={"baseline": 1},
            sampling_seed=22,
        )
