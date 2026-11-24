"""Run equal-budget selection rounds and an independent-map acceptance round."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

from . import evaluation, versions
from .eval_plan import get_suite, json_text
from .eval_store import database, local_dir, shard_lock, worker_lock

SCREEN_ID = re.compile(r"screen-[0-9a-f]{32}")
LAUNCH_GRACE_SECONDS = 30


def _folder(root, screen_id):
    return local_dir(root) / "screens" / screen_id


def _load(root, screen_id):
    if not isinstance(screen_id, str) or not SCREEN_ID.fullmatch(screen_id):
        raise ValueError("筛选 ID 格式错误")
    with database(root) as db:
        record = db.execute("SELECT record FROM screens WHERE screen_id = ?", (screen_id,)).fetchone()
    if record is None:
        raise ValueError(f"找不到筛选 {screen_id}")
    return json.loads(record["record"])


def _save(root, record):
    with database(root) as db:
        db.execute(
            "INSERT INTO screens VALUES (?, ?) ON CONFLICT(screen_id) DO UPDATE SET record = excluded.record",
            (record["id"], json_text(record)),
        )


def _plans(root, candidates, stages):
    if len(stages) < 2:
        raise ValueError("筛选需要初筛和验收两个阶段")
    plans, earlier_maps = [], set()
    count = len(candidates)
    for index, (name, keep) in enumerate(stages):
        suite = get_suite(root, name)
        if type(keep) is not int or not 1 <= keep <= count:
            raise ValueError(f"{name} 晋级数须介于 1 和 {count} 之间")
        if index == len(stages) - 1 and earlier_maps.intersection(suite["seeds"]):
            raise ValueError("最后一轮须使用独立地图")
        earlier_maps.update(suite["seeds"])
        plans.append({"suite": name, "suite_id": suite["suite_id"], "keep": keep})
        count = keep
    return plans


def status(root, screen_id):
    record = _load(root, screen_id)
    if record["status"] == "running":
        with shard_lock(_folder(root, screen_id), 0) as acquired:
            if acquired:
                record = _load(root, screen_id)
                if record["status"] == "running":
                    record |= {"status": "interrupted", "error": "筛选进程已退出，可继续筛选"}
    elif record["status"] == "queued" and time.time() - record["queued_at"] > LAUNCH_GRACE_SECONDS:
        record |= {"status": "interrupted", "error": "筛选进程未启动，可继续筛选"}
    if record["rounds"]:
        record["current_round"] = evaluation.status(root, record["rounds"][-1]["run_id"])
    return record


def screen(root, candidates, baseline, stages, *, shards=1, background=True):
    candidates = list(candidates)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("候选列表须为非空且无重复的版本")
    if baseline is None:
        baseline = versions.main_version(root)
    plans = _plans(root, candidates, stages)
    evaluation.estimate(root, candidates, baseline, plans[0]["suite"], shards=shards)
    record = {
        "id": "screen-" + uuid4().hex, "baseline": baseline, "plans": plans,
        "candidates": candidates, "rounds": [], "shards": shards,
    }
    return _launch(root, record, retry_failures=False, background=background)


def resume(root, screen_id, *, background=True):
    record = status(root, screen_id)
    if record["status"] == "completed":
        return record
    if record["status"] in ("queued", "running"):
        raise ValueError(f"筛选 {screen_id} 正在执行")
    record.pop("current_round", None)
    record.pop("error", None)
    return _launch(root, record, retry_failures=True, background=background)


def _launch(root, record, *, retry_failures, background):
    root = root.resolve()
    screen_id = record["id"]
    folder = _folder(root, screen_id)
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "worker.log"
    record |= {"status": "queued", "queued_at": time.time(), "log_path": str(log_path)}
    _save(root, record)
    if not background:
        _work(root, screen_id, retry_failures)
        return status(root, screen_id)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        with log_path.open("ab") as log:
            subprocess.Popen(
                [sys.executable, "-m", "tank.screening", str(root), screen_id, "retry" if retry_failures else "start"],
                cwd=root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                env=os.environ | {"PYTHONUTF8": "1"},
                creationflags=flags, start_new_session=os.name != "nt",
            )
    except OSError as error:
        _save(root, record | {"status": "failed", "error": str(error)})
        raise
    return record


def _advance(root, record, retry_failures):
    rounds = record["rounds"]
    candidates = record["candidates"]
    for index, plan in enumerate(record["plans"]):
        if index < len(rounds):
            entry = rounds[index]
            if "selected" in entry:
                candidates = entry["selected"]
                continue
            if retry_failures:
                evaluation.retry(root, entry["run_id"], background=False)
        else:
            if get_suite(root, plan["suite"])["suite_id"] != plan["suite_id"]:
                raise ValueError(f"方案 {plan['suite']} 已变更，请恢复原方案")
            run = evaluation.start(
                root, candidates, record["baseline"], plan["suite"], shards=record["shards"], prepare_only=True,
            )
            entry = {"run_id": run["run_id"], "candidates": candidates, **plan}
            rounds.append(entry)
            _save(root, record)
        run = evaluation.resume(root, entry["run_id"], background=False)
        if run["status"] != "completed":
            record["status"] = run["status"]
            _save(root, record)
            return
        rows = evaluation.compare(root, run["run_id"])["rows"]
        candidates = [row["candidate"] for row in rows[:plan["keep"]]]
        entry["selected"] = candidates
        _save(root, record)
    record.update(status="completed", selected=candidates)
    _save(root, record)


def _work(root, screen_id, retry_failures):
    with worker_lock(_folder(root, screen_id), 0) as acquired:
        if not acquired:
            return
        record = _load(root, screen_id)
        if record["status"] != "queued":
            return
        record["status"] = "running"
        _save(root, record)
        try:
            _advance(root, record, retry_failures)
        except KeyboardInterrupt:
            _save(root, record | {"status": "interrupted"})
            raise
        except (OSError, ValueError, RuntimeError) as error:
            traceback.print_exc()
            _save(root, record | {"status": "failed", "error": str(error)})


if __name__ == "__main__":
    root_arg, screen_arg, mode_arg = sys.argv[1:]
    _work(Path(root_arg), screen_arg, mode_arg == "retry")
