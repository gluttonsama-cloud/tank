"""SQLite records and process-owned shard locks."""

from __future__ import annotations

import gzip
import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager

from .eval_plan import RUN_ID, SCORE_ITEMS, json_text, required

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    manifest TEXT NOT NULL,
    created_at REAL NOT NULL,
    attempted INTEGER NOT NULL DEFAULT 0,
    cancelled INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS matches (
    case_id TEXT PRIMARY KEY,
    spec TEXT NOT NULL,
    result TEXT NOT NULL,
    elapsed REAL NOT NULL,
    replay BLOB
);
CREATE TABLE IF NOT EXISTS failures (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
CREATE TABLE IF NOT EXISTS lan_leases (
    run_id TEXT NOT NULL,
    lease_id TEXT PRIMARY KEY,
    shard INTEGER NOT NULL,
    worker TEXT NOT NULL,
    expires_at REAL NOT NULL,
    submitted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS screens (
    screen_id TEXT PRIMARY KEY,
    record TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS received_refs (
    ref TEXT PRIMARY KEY,
    sha TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS queued_shards (
    run_id TEXT NOT NULL,
    shard INTEGER NOT NULL,
    launch_id TEXT NOT NULL,
    queued_at REAL NOT NULL,
    PRIMARY KEY (run_id, shard)
);
"""


def local_dir(root):
    folder = root / ".local" / "eval"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@contextmanager
def database(root):
    db = sqlite3.connect(local_dir(root) / "results.sqlite3", timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.executescript(SCHEMA)
        with db:
            yield db
    finally:
        db.close()


def add_manifest(db, manifest):
    run_id = manifest["run_id"]
    encoded = json_text(manifest)
    known = db.execute("SELECT manifest FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if known and known["manifest"] != encoded:
        raise ValueError(f"批次 {run_id} 的清单冲突")
    db.execute("INSERT OR IGNORE INTO runs (run_id, manifest, created_at) VALUES (?, ?, ?)", (run_id, encoded, time.time()))


def read_manifest(db, run_id):
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("批次 ID 格式错误")
    row = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"找不到批次 {run_id}")
    return json.loads(row["manifest"]), row


def leased_shards(db, run_id):
    return {row[0] for row in db.execute(
        "SELECT DISTINCT shard FROM lan_leases WHERE run_id = ? AND submitted = 0 AND expires_at > ?",
        (run_id, time.time()),
    )}


def queued_shards(db, run_id):
    return {row[0] for row in db.execute(
        "SELECT shard FROM queued_shards WHERE run_id = ? AND queued_at > ?",
        (run_id, time.time() - 30),
    )}


def iter_matches(db, specs, *, include_replay=False, summary=False):
    keys = list(specs)
    columns = ", replay" if include_replay else ""
    if not summary:
        columns += ", result"
    batch_size = 50 if include_replay else 500
    for offset in range(0, len(keys), batch_size):
        batch = keys[offset:offset + batch_size]
        params = ",".join("?" for _ in batch)
        rows = db.execute(
            f"SELECT case_id, spec, elapsed, replay IS NOT NULL AS has_replay{columns}"
            f" FROM matches WHERE case_id IN ({params})", batch,
        )
        matches = {}
        for row in rows:
            case_id = row["case_id"]
            if json.loads(row["spec"]) != specs[case_id]:
                raise ValueError(f"对局 {case_id} 的开局参数冲突")
            record = dict(row)
            if not summary:
                record["result"] = json.loads(row["result"])
            matches[case_id] = record
        for case_id in batch:
            if case_id in matches:
                yield case_id, matches[case_id]


def cached_matches(db, specs, *, summary=False):
    return dict(iter_matches(db, specs, summary=summary))


def _score_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_result(result):
    winner = required(result, "winner")
    if type(winner) is not int or winner not in (-1, 0, 1):
        raise ValueError("winner 必须为 -1、0 或 1")
    scores = required(result, "scores")
    if not isinstance(scores, list) or len(scores) != 2 or not all(map(_score_number, scores)):
        raise ValueError("scores 必须包含红蓝双方的有限数值")
    official = required(result, "result")
    official_scores = required(official, "scores")
    if not isinstance(official_scores, list) or len(official_scores) != 2:
        raise ValueError("result.scores 必须包含红蓝双方成绩")
    if winner != required(official, "winner"):
        raise ValueError("winner 与官方结算不一致")
    for index, score in enumerate(official_scores):
        if scores[index] != required(score, "total"):
            raise ValueError("scores 与官方结算不一致")
        for name in SCORE_ITEMS:
            if not _score_number(required(score, name)):
                raise ValueError(f"result.scores.{name} 必须是有限数值")
    return {"winner": winner, "scores": scores, "result": official}


def save_match(db, case_id, spec, result, elapsed):
    replay = result.get("replay")
    if replay is not None and not isinstance(replay, dict):
        raise ValueError("replay 必须是 JSON 对象")
    if not _score_number(elapsed) or elapsed < 0:
        raise ValueError("elapsed 必须是有限非负数")
    encoded = json_text(validate_result(result))
    replay_blob = gzip.compress(json_text(replay).encode("utf-8"), mtime=0) if replay is not None else None
    spec_text = json_text(spec)
    db.execute(
        "INSERT OR IGNORE INTO matches VALUES (?, ?, ?, ?, ?)",
        (case_id, spec_text, encoded, elapsed, replay_blob),
    )
    known = db.execute("SELECT spec, result, replay FROM matches WHERE case_id = ?", (case_id,)).fetchone()
    if known["spec"] != spec_text or known["result"] != encoded or (
        known["replay"] is not None and replay_blob is not None
        and gzip.decompress(known["replay"]) != gzip.decompress(replay_blob)
    ):
        raise ValueError(f"对局 {case_id} 的成绩或回放冲突")
    if replay_blob is not None and known["replay"] is None:
        db.execute("UPDATE matches SET replay = ? WHERE case_id = ?", (replay_blob, case_id))


@contextmanager
def shard_lock(folder, shard):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"shard-{shard}.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import errno
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno != errno.EACCES:
                    raise
                yield False
                return
            try:
                yield True
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def worker_lock(folder, shard):
    deadline = time.monotonic() + 2
    while True:
        with shard_lock(folder, shard) as acquired:
            if acquired:
                yield True
                return
        if time.monotonic() >= deadline:
            yield False
            return
        time.sleep(0.05)
