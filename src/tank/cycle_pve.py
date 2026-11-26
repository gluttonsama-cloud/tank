"""Paired official practice gates and migration of the existing sampler."""

import gzip
import hashlib
import json
import random
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from statistics import fmean
from uuid import uuid4

import numpy as np
from seedcup.game import generate_map

from . import sampling, sampling_data, versions


def _roster(incumbent, candidate, reference):
    roster = {}
    for name, version in (
        ("incumbent", incumbent),
        ("candidate", candidate),
        ("reference", reference),
    ):
        if version and version not in roster.values():
            roster[name] = version
    return roster


def collection_case(root, key, plan, step, *, freeze=True):
    width = max(2, plan["width"])
    start = plan["cycle"].get("from_step", 0)
    offset, pos = divmod(step - start, width)
    index = start // plan["width"] + offset
    seed = (plan["seed"] + index * 104729) % 2**31
    with sampling_data.database(root) as db:
        db.execute(
            "CREATE TABLE IF NOT EXISTS cycle_maps(run TEXT, idx INTEGER, picks TEXT, PRIMARY KEY(run,idx))"
        )
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT picks FROM cycle_maps WHERE run=? AND idx=?", (key, index)).fetchone()
        if row:
            picks = json.loads(row[0])
        else:
            picks = list(plan["cycle"]["roster"].items())[:width]
            random.Random(seed).shuffle(picks)
            if freeze:
                db.execute(
                    "INSERT INTO cycle_maps VALUES(?,?,?)",
                    (key, index, json.dumps(picks)),
                )
    name, version = picks[pos % len(picks)]
    return seed, name, version


def ensure_sampler(root, incumbent, candidate, reference, key=None):
    """Reconfigure only after retiring the old worker; keep its partial map and step."""
    roster = _roster(incumbent, candidate, reference)
    with sampling_data.database(root) as db:
        if key is None:
            rows = db.execute("""SELECT id FROM runs WHERE json_extract(plan,'$.mode')='active'
                AND json_extract(plan,'$.target')='boss' AND json_extract(plan,'$.maps') IS NULL
                ORDER BY paused,updated DESC LIMIT 1""").fetchone()
            key = rows[0] if rows else None
        if key is None:
            sampling.check_host(root)
            key = "sample-" + uuid4().hex
            plan = {
                "mode": "active",
                "maps": None,
                "target": "boss",
                "interval": 30,
                "width": max(2,len(roster)),
                "seed": random.SystemRandom().randrange(2**31),
                "versions": roster,
                "probes": list(roster),
                "cycle": {"from_step": 0, "roster": roster},
            }
            db.execute(
                "INSERT INTO runs(id,plan,state,updated) VALUES(?,?,'paused',?)",
                (key, json.dumps(plan), time.time()),
            )
    row = sampling.record(root, key)
    state = sampling.status(root, key)["state"]
    plan = row["plan"]
    previous = plan.get("cycle")
    if not previous and state in ("running", "waiting", "queued", "pausing"):
        if state != "pausing":
            sampling.pause(root, key)
        return key
    if not previous or previous["roster"] != roster:
        for version in set(roster.values()):
            versions.verify(root, version)
        with sampling_data.database(root) as db:
            db.execute("BEGIN IMMEDIATE")
            step, saved = db.execute("SELECT step,plan FROM runs WHERE id=?", (key,)).fetchone()
            plan = json.loads(saved)
            boundary = (step // plan["width"] + 1) * plan["width"]
            plan["cycle"] = {
                "from_step": previous["from_step"] if previous else boundary,
                "roster": roster,
            }
            db.execute("UPDATE runs SET plan=? WHERE id=?", (json.dumps(plan), key))
    if state in ("paused", "interrupted"):
        sampling.resume(root, key, False)
    elif state == "failed":
        raise ValueError(sampling.record(root, key)["message"])
    return key


def paired_interval(tiers):
    """Stratify by tier, draw independent layouts, keep red/blue together."""
    rng = np.random.default_rng(4011)
    draws = [rng.choice(v, size=(5000, len(v)), replace=True).mean(axis=1) for v in tiers.values()]
    return {
        "estimated_40_game_gain": 40 * fmean(map(fmean, tiers.values())),
        "ci95": (40 * np.quantile(np.mean(draws, axis=0), [0.025, 0.975])).tolist(),
    }


def _gate_grids(seed):
    return {value: sampling_data.layout_hash(generate_map(value).to_dict())
            for value in ((seed + i * 104729) % 2**31 for i in range(5))}


def _archived_and_local_grids(root):
    from training.cycle_train import local_training_grids

    from . import cycle_experience

    root = Path(root).resolve()
    grids = local_training_grids(root)
    if (root / ".local/cycle/experience.sqlite3").is_file():
        grids.update(cycle_experience.indexed_grids(root))
    # Search archive files directly so a capture waiting for the background index also counts.
    for folder in (".local/online/replays", ".local/cycle/ladder/replays"):
        for path in (root / folder).glob("*.json.gz"):
            with gzip.open(path, "rt", encoding="utf-8") as archive:
                grids.add(sampling_data.layout_hash(json.load(archive)["map"]))
    for path in (root / ".local/sampling/observations").glob("*.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as archive:
            grids.add(sampling_data.sdk_layout_hash(json.load(archive)["start"]))
    results = root / ".local/eval/results.sqlite3"
    if results.is_file():
        with closing(sqlite3.connect(results.as_uri() + "?mode=ro", uri=True)) as db:
            seeds = {row[0] for row in db.execute("SELECT DISTINCT json_extract(spec, '$.seed') FROM matches")}
        grids.update(sampling_data.layout_hash(generate_map(seed).to_dict()) for seed in seeds)
    return grids


def _gate_batch(db, parent, candidate, incumbent, tier, seed):
    rows = db.execute("""
        SELECT id, plan FROM runs WHERE json_extract(plan,'$.parent')=?
          AND json_extract(plan,'$.target')=? AND json_extract(plan,'$.seed')=?
        """, (parent, tier, seed))
    expected = {
        "mode": "active", "maps": 5, "width": 2, "repeats": 1,
        "probes": ["candidate", "incumbent"], "versions": {"candidate": candidate, "incumbent": incumbent},
    }
    matching = []
    for row in rows:
        plan = json.loads(row["plan"])
        if all(plan.get(key) == value for key, value in expected.items()):
            matching.append(row["id"])
    if len(matching) > 1:
        raise ValueError(f"档位 {tier} 的同轮配对评测记录重复")
    return matching[0] if matching else None


def _gate_maps(root, parent, candidate, incumbent, number, tier):
    """Reserve a whole paired map schedule, or adopt an already queued legacy schedule."""
    identity = (parent, candidate, incumbent)
    digest = hashlib.sha256(f"pve-accept:{candidate}:{incumbent}:{number}:{tier}".encode()).digest()
    original = int.from_bytes(digest[:4], "big") % 2**31

    def existing(db):
        row = db.execute("SELECT * FROM cycle_gate_maps WHERE round=? AND tier=?", (number, tier)).fetchone()
        if row is None:
            return None
        if (row["parent"], row["candidate"], row["incumbent"]) != identity:
            raise ValueError(f"第 {number} 轮 {tier} 的地图预留属于另一对策略或采样任务")
        grids = dict(json.loads(row["grids"]))
        if len(grids) != 5 or _gate_grids(row["seed"]) != grids:
            raise ValueError(f"第 {number} 轮 {tier} 的地图预留不符合当前 SDK")
        return row["seed"], grids, row["source"], _gate_batch(db, *identity, tier, row["seed"])

    def reserve(db, seed, grids, source, batch):
        db.execute("INSERT INTO cycle_gate_maps VALUES (?,?,?,?,?,?,?,?)",
                   (number, tier, *identity, seed, json.dumps(list(grids.items())), source))
        db.executemany("INSERT OR IGNORE INTO sampling_holdouts VALUES (?)", ((grid,) for grid in grids.values()))
        return seed, grids, source, batch

    with sampling_data.database(root) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS cycle_gate_maps(
            round INTEGER, tier TEXT, parent TEXT NOT NULL, candidate TEXT NOT NULL, incumbent TEXT NOT NULL,
            seed INTEGER NOT NULL, grids TEXT NOT NULL, source TEXT NOT NULL, PRIMARY KEY(round,tier))""")
        db.execute("BEGIN IMMEDIATE")
        saved = existing(db)
        if saved is not None:
            return saved
        legacy = _gate_batch(db, *identity, tier, original)
        if legacy is not None:
            return reserve(db, original, _gate_grids(original), "legacy", legacy)

    archived = _archived_and_local_grids(root)
    with sampling_data.database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        saved = existing(db)
        if saved is not None:
            return saved
        legacy = _gate_batch(db, *identity, tier, original)
        if legacy is not None:
            return reserve(db, original, _gate_grids(original), "legacy", legacy)
        for table in ("sampling_holdouts", "layouts", "examples", "sdk_observations"):
            expression = "grid" if table in ("sampling_holdouts", "layouts") else "json_extract(entry, '$.grid_sha256')"
            archived.update(row[0] for row in db.execute(f"SELECT DISTINCT {expression} FROM {table}"))
        rng = random.Random(digest)
        for seed in (original, *(rng.randrange(2**31) for _ in range(255))):
            grids = _gate_grids(seed)
            layouts = set(grids.values())
            if len(layouts) == 5 and layouts.isdisjoint(archived):
                return reserve(db, seed, grids, "fresh", None)
    raise ValueError(f"第 {number} 轮 {tier} 未找到五张独立的新评测地图")


def _scores(root, run, tier, grids):
    with sampling_data.database(root) as db:
        rows = db.execute(
            """SELECT s.id,s.side,o.entry AS sdk,e.entry AS replay FROM slots s
            LEFT JOIN sdk_observations o ON o.id=s.id AND o.target=?
            LEFT JOIN examples e ON e.id=s.id AND e.target=? WHERE s.run=?""",
            (tier, tier, run),
        ).fetchall()
    scores = {}
    for row in rows:
        if row["sdk"]:
            item = json.loads(row["sdk"])
            side, seed, result = item["side"], item["map_seed"], item["result"]
        elif row["replay"]:
            item = json.loads(row["replay"])
            raw = (root / item["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError(f"回放内容已变更 {row['id']}")
            side = "red" if item["target_side"] == "blue" else "blue"
            seed, result = item["seed"], json.loads(gzip.decompress(raw))["result"]
        else:
            raise ValueError(f"缺少采样对局 {row['id']}")
        key = seed, item["version_id"], side
        if side != row["side"] or key in scores or item["grid_sha256"] != grids.get(seed):
            raise ValueError(f"配对记录的地图、版本或出生侧不符 {row['id']}")
        scores[key] = result["scores"][("red", "blue").index(side)]["total"]
    return scores


def gate(root, parent, candidate, incumbent, number, *, allow_enqueue=lambda: True):
    tiers = {}
    batches = {}
    map_reservations = {}
    games = 0
    for tier in ("boss", "rookie", "mid", "strong"):
        seed, grids, source, existing = _gate_maps(root, parent, candidate, incumbent, number, tier)
        map_reservations[tier] = {"source": source, "seeds": list(grids), "grid_sha256": list(grids.values())}
        if existing is None and not allow_enqueue():
            return {"status":"paused","batches":batches,"games":games,"map_reservations":map_reservations}
        run = sampling.status(root, existing) if existing is not None else sampling.enqueue(
            root,
            parent,
            strategies=[("candidate", candidate), ("incumbent", incumbent)],
            maps=5,
            repeats=1,
            seed=seed,
            target=tier,
        )
        batches[tier] = run["id"]
        games += run["games"]
        if run["state"] != "completed":
            if run["state"] in ("failed", "paused"):
                raise ValueError(f"配对评测 {run['id']} {run['state']} {run['message']}")
            return {
                "status": "collecting",
                "tier": tier,
                "games": games,
                "batches": batches,
                "map_reservations": map_reservations,
            }
        scores = _scores(root, run["id"], tier, grids)
        tiers[tier] = [
            fmean(scores[m, candidate, s] - scores[m, incumbent, s] for s in ("red", "blue")) for m in grids
        ]
        if tier == "boss":
            boss_ci = [value / 40 for value in paired_interval(tiers)["ci95"]]
        if tier == "boss" and boss_ci[0] <= 0:
            return {
                "status": "rejected",
                "boss_mean_gain_per_game": fmean(tiers[tier]),
                "boss_ci95_per_game": boss_ci,
                "batches": batches,
                "layout_deltas": tiers,
                "games": games,
                "map_reservations": map_reservations,
            }
    metrics = paired_interval(tiers)
    return {
        "status": "accepted" if metrics["ci95"][0] > 0 else "rejected",
        "layout_deltas": tiers,
        "batches": batches,
        "games": games,
        "map_reservations": map_reservations,
        **metrics,
    }
