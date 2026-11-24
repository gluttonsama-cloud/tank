"""Persist explicitly requested batch publication and run it in a local worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

from .eval_store import database, local_dir, shard_lock, worker_lock

SCHEMA = """
CREATE TABLE IF NOT EXISTS auto_publications (
    run_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    token TEXT,
    prepared TEXT,
    branch TEXT,
    error TEXT
)
"""
START_TIMEOUT = 30


def register(db, run_id):
    db.execute(SCHEMA)
    db.execute(
        "INSERT INTO auto_publications (run_id, state, updated_at) VALUES (?, 'waiting', ?)",
        (run_id, time.time()),
    )


def _read(db, run_id):
    db.execute(SCHEMA)
    return db.execute("SELECT * FROM auto_publications WHERE run_id = ?", (run_id,)).fetchone()


def waiting_runs(db):
    db.execute(SCHEMA)
    db.execute("CREATE INDEX IF NOT EXISTS auto_publications_by_state ON auto_publications (state)")
    return db.execute(
        """SELECT r.run_id, r.manifest FROM auto_publications AS p
           JOIN runs AS r ON r.run_id = p.run_id WHERE p.state = 'waiting'"""
    ).fetchall()


def _folder(root, run_id):
    return local_dir(root) / run_id / "publication"


def _summary(row):
    return {key: row[key] for key in ("state", "attempts", "updated_at", "branch", "error")}


def status(root, db, run_id):
    row = _read(db, run_id)
    if row is None:
        return None
    if row["state"] == "queued" and time.time() - row["updated_at"] > START_TIMEOUT:
        return _summary(row) | {"state": "failed", "error": "发布进程未启动"}
    if row["state"] == "publishing":
        with shard_lock(_folder(root, run_id), 0) as available:
            if available:
                row = _read(db, run_id)
                if row["state"] == "publishing":
                    return _summary(row) | {"state": "failed", "error": "发布进程已退出"}
    return _summary(row)


def _spawn(root, run_id, token):
    folder = _folder(root, run_id)
    folder.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with (folder / "worker.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "tank.publication", str(root), run_id, token],
            cwd=root, env=os.environ | {"PYTHONUTF8": "1"}, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, creationflags=flags, start_new_session=os.name != "nt",
        )


def enqueue(root, run_id, *, retry=False):
    root = root.resolve()
    with database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        share = status(root, db, run_id)
        eligible = {"waiting", "failed"} if retry else {"waiting"}
        if share is None or share["state"] not in eligible:
            return share
        token = uuid4().hex
        db.execute(
            """UPDATE auto_publications SET state = 'queued', attempts = attempts + 1,
                   token = ?, updated_at = ?, error = NULL WHERE run_id = ?""",
            (token, time.time(), run_id),
        )
    try:
        _spawn(root, run_id, token)
    except OSError as error:
        _finish(root, run_id, token, "failed", error=str(error))
    with database(root) as db:
        return status(root, db, run_id)


def _finish(root, run_id, token, state, *, branch=None, error=None):
    with database(root) as db:
        db.execute(
            """UPDATE auto_publications SET state = ?, branch = COALESCE(?, branch), error = ?, updated_at = ?
               WHERE run_id = ? AND token = ?""",
            (state, branch, error, time.time(), run_id, token),
        )


def _work(root, run_id, token):
    from . import evaluation, sync

    with worker_lock(_folder(root, run_id), 0) as acquired:
        if not acquired:
            return
        with database(root) as db:
            db.execute("BEGIN IMMEDIATE")
            row = _read(db, run_id)
            if row is None or row["state"] != "queued" or row["token"] != token:
                return
            db.execute(
                "UPDATE auto_publications SET state = 'publishing', updated_at = ? WHERE run_id = ?",
                (time.time(), run_id),
            )
        try:
            run = evaluation.status(root, run_id)
            if run["status"] != "completed":
                raise ValueError(f"批次 {run_id} 未完成")
            if row["prepared"] is not None:
                prepared = json.loads(row["prepared"])
            else:
                prepared = sync.publish(root, run_id)
                if prepared["completed"] != run["total"]:
                    raise ValueError(f"批次 {run_id} 的发布片段不完整")
                # Keep the first complete artifact so a retry targets the same remote reference.
                with database(root) as db:
                    db.execute(
                        "UPDATE auto_publications SET prepared = ?, branch = ? WHERE run_id = ? AND token = ?",
                        (json.dumps(prepared, ensure_ascii=False), sync.branch_for(prepared), run_id, token),
                    )
            result = sync.push(root, run_id, prepared=prepared)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            traceback.print_exc()
            _finish(root, run_id, token, "failed", error=str(error))
        else:
            _finish(root, run_id, token, "published", branch=result["branch"])


if __name__ == "__main__":
    root_arg, run_arg, token_arg = sys.argv[1:]
    _work(Path(root_arg), run_arg, token_arg)
