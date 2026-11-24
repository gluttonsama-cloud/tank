"""示范采集与本机训练记录。"""

import hashlib
import json
import random
import re
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

from seedcup import Bot
from seedcup.game import SIDES

from .simulation import Frame, FrameEnvironment, run_match
from .versions import describe
from .workspace import read_json, write_json


class _DemonstrationBot(Bot):
    def __init__(self, bot: Bot, match_id: str, side: str, samples: list[dict]):
        self._bot = bot
        self._match_id = match_id
        self._side = side
        self._samples = samples
        self.info: dict

    def on_match_start(self, info: dict) -> None:
        self.info = deepcopy(info)
        self._bot.on_match_start(info)

    def decide(self, obs: dict) -> dict:
        observation = deepcopy(obs)
        action = self._bot.decide(obs)
        self._samples.append({
            "match_id": self._match_id,
            "frame": observation["frame"],
            "side": self._side,
            "observation": observation,
            "action": deepcopy({field: action[field] for field in ("forward", "turn", "fire")}),
        })
        return action

    def on_match_end(self, result: dict) -> None:
        self._bot.on_match_end(result)


def collect_demonstrations(
    red: Bot,
    blue: Bot,
    seed: int,
    game_no: int,
    *,
    match_id: str,
    sides: Sequence[str],
    output: Path,
    snapshots: dict[str, str] | None = None,
) -> dict:
    """采集指定一侧或两侧的示范，完成整局后原子写入一份 JSON。"""
    _require_reference(match_id, "match_id")
    if snapshots is not None and set(snapshots) != set(SIDES):
        raise ValueError("snapshots 需要 red 和 blue 双方版本")
    if not sides or any(side not in SIDES for side in sides) or len(sides) != len(set(sides)):
        raise ValueError("sides 必须包含 red、blue 中的一项或两项")
    samples: list[dict] = []
    recorders = {
        side: _DemonstrationBot(bot, match_id, side, samples)
        for side, bot in zip(SIDES, (red, blue), strict=True)
        if side in sides
    }
    match = run_match(recorders.get("red", red), recorders.get("blue", blue), seed, game_no)
    demonstration = {
        "schema_version": 1,
        "match_id": match_id,
        "seed": seed,
        "game_no": game_no,
        "match_info": {side: recorder.info for side, recorder in recorders.items()},
        "samples": samples,
        "result": match["result"],
    }
    if snapshots is not None:
        demonstration["snapshots"] = dict(snapshots)
    write_json(output, demonstration)
    return demonstration


def _require_reference(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须为非空字符串")


def _require_integer(value: int, field: str, *, minimum: int | None = None) -> None:
    if type(value) is not int:
        raise ValueError(f"{field} 必须为整数")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} 必须大于或等于 {minimum}")


class TrainingStore:
    """按训练配置、实际采样对局和检查点记录来源。"""

    def __init__(self, root: Path):
        self.root = root
        self.directory = root / ".local" / "training"

    def register_config(
        self, *, teacher_snapshot: str | None, opponent_pool: dict[str, int], sampling_seed: int,
    ) -> str:
        if teacher_snapshot is not None:
            self._require_snapshot(teacher_snapshot, "teacher_snapshot")
        if not isinstance(opponent_pool, dict) or not opponent_pool:
            raise ValueError("opponent_pool 必须包含至少一个对手快照及其权重")
        for snapshot, weight in opponent_pool.items():
            self._require_snapshot(snapshot, "opponent_pool 快照")
            _require_integer(weight, f"opponent_pool[{snapshot}]", minimum=1)
        _require_integer(sampling_seed, "sampling_seed")
        return self._register("configs", "trcfg", {
            "schema_version": 1,
            "teacher_snapshot": teacher_snapshot,
            "opponent_pool": dict(sorted(opponent_pool.items())),
            "sampling_seed": sampling_seed,
        })

    def read_config(self, config_id: str) -> dict:
        return self._read("configs", config_id, "trcfg-[0-9a-f]{64}", "config_id")

    def start_episode(
        self,
        env: FrameEnvironment,
        config_id: str,
        *,
        episode_index: int,
        seed: int,
        game_no: int,
        controlled_side: str,
    ) -> tuple[Frame, dict]:
        """从固定的对手池抽样并记录分配结果，返回首帧和对局记录。"""
        _require_integer(episode_index, "episode_index", minimum=0)
        if controlled_side not in SIDES:
            raise ValueError("controlled_side 必须为 red 或 blue")
        config = self.read_config(config_id)
        pool = config["opponent_pool"]
        rng = random.Random(f"{config['sampling_seed']}:{episode_index}")
        ticket = rng.randrange(sum(pool.values()))
        for opponent_snapshot, weight in pool.items():
            if ticket < weight:
                break
            ticket -= weight
        opening = env.reset(seed=seed, game_no=game_no)
        record = {
            "schema_version": 1,
            "match_id": opening.match_id,
            "config_id": config_id,
            "episode_index": episode_index,
            "seed": seed,
            "game_no": game_no,
            "controlled_side": controlled_side,
            "opponent_snapshot": opponent_snapshot,
        }
        try:
            write_json(self.directory / "episodes" / f"{opening.match_id}.json", record)
        except OSError:
            env.truncate("training_record_failed")
            raise
        return opening, record

    def read_episode(self, match_id: str) -> dict:
        return self._read("episodes", match_id, "[0-9a-f]{32}", "match_id")

    def register_checkpoint(
        self, config_id: str, weights: Path, *, step: int, snapshot_id: str | None = None,
    ) -> str:
        self.read_config(config_id)
        _require_integer(step, "step", minimum=0)
        if snapshot_id is not None:
            self._require_snapshot(snapshot_id, "snapshot_id")
        weights = weights.resolve()
        with weights.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        return self._register("checkpoints", "ckpt", {
            "schema_version": 1,
            "config_id": config_id,
            "step": step,
            "weights_path": str(weights),
            "weights_sha256": digest,
            "snapshot_id": snapshot_id,
        })

    def read_checkpoint(self, checkpoint_id: str) -> dict:
        return self._read("checkpoints", checkpoint_id, "ckpt-[0-9a-f]{64}", "checkpoint_id")

    def _require_snapshot(self, snapshot_id: str, field: str) -> None:
        _require_reference(snapshot_id, field)
        try:
            describe(self.root, snapshot_id)
        except ValueError as error:
            raise ValueError(f"{field}: {error}") from error

    def _register(self, folder: str, prefix: str, value: dict) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        record_id = f"{prefix}-{hashlib.sha256(payload).hexdigest()}"
        write_json(self.directory / folder / f"{record_id}.json", value)
        return record_id

    def _read(self, folder: str, record_id: str, pattern: str, field: str) -> dict:
        if not isinstance(record_id, str) or not re.fullmatch(pattern, record_id):
            raise ValueError(f"{field} 格式错误")
        return read_json(self.directory / folder / f"{record_id}.json")
