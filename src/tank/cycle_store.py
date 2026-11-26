"""Durable control flags, line state and experiment history."""

import json
import sqlite3
import time
from contextlib import contextmanager

LINES = ("pvp", "pve")


def folder(root):
    return root / ".local/cycle"


@contextmanager
def database(root):
    path = folder(root)
    path.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path / "control.sqlite3", timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS lines(line TEXT PRIMARY KEY,enabled INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS state(line TEXT, name TEXT, value TEXT NOT NULL, PRIMARY KEY(line,name));
            CREATE TABLE IF NOT EXISTS changes(
                line TEXT,round INTEGER,at REAL,outcome TEXT,record TEXT,PRIMARY KEY(line,round));
        """)
        with db:
            yield db
    finally:
        db.close()


def enabled(root, line):
    with database(root) as db:
        row = db.execute("SELECT enabled FROM lines WHERE line=?", (line,)).fetchone()
    return bool(row and row[0])


def get(root, line, name, default=None):
    with database(root) as db:
        row = db.execute("SELECT value FROM state WHERE line=? AND name=?", (line, name)).fetchone()
    return json.loads(row[0]) if row else default


def put(root, line, **values):
    with database(root) as db:
        db.executemany(
            "INSERT OR REPLACE INTO state VALUES(?,?,?)",
            [(line, k, json.dumps(v, allow_nan=False)) for k, v in values.items()],
        )


def round_dir(root, line, number):
    return folder(root) / line / "rounds" / f"{number:06d}"


def report_watermark(root):
    with database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        return time.time()


def history(root, line, limit=3, since=0, until=float("inf")):
    with database(root) as db:
        rows = db.execute(
            "SELECT record FROM changes WHERE line=? AND at>? AND at<=? AND outcome!='reopened' ORDER BY round DESC LIMIT ?",
            (line, since, until, limit),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def resume_completed_result(root, line, number):
    """A worker can finish after its interrupted round was superseded."""
    with database(root) as db:
        db.execute(
            """UPDATE changes SET outcome='reopened' WHERE line=? AND round=?
                      AND outcome='superseded' AND json_extract(record,'$.candidate') IS NULL""",
            (line, number),
        )


def finish(root, line, result, outcome, reason, *, changes=None, **extra):
    number = result["round"]
    report = {
        "round": number,
        "incumbent": result["incumbent"],
        "candidate": result["candidate"],
        "outcome": outcome,
        "reason": reason,
        "report": str(round_dir(root, line, number) / "result.json"),
        **extra,
    }
    with database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        report["at"] = time.time()
        new = db.execute(
            "INSERT OR IGNORE INTO changes VALUES(?,?,?,?,?)",
            (line, number, report["at"], outcome, json.dumps(report)),
        )
        if not new.rowcount:
            previous = json.loads(
                db.execute(
                    "SELECT record FROM changes WHERE line=? AND round=?",
                    (line, number),
                ).fetchone()[0]
            )
            if (
                previous["outcome"] != "superseded"
                or previous["candidate"] is not None
                or not result["candidate"]
            ):
                return previous
            report["prior_decision"] = previous
            db.execute(
                "UPDATE changes SET at=?,outcome=?,record=? WHERE line=? AND round=?",
                (report["at"], outcome, json.dumps(report), line, number),
            )
        values = {
            "phase": reason,
            "latest_candidate": result["candidate"],
            "gate": None,
            **(changes or {}),
        }
        if outcome == "promoted":
            values["last_change"] = report
        db.executemany(
            "INSERT OR REPLACE INTO state VALUES(?,?,?)",
            [(line, k, json.dumps(v, allow_nan=False)) for k, v in values.items()],
        )
    return report
