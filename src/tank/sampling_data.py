"""Index official replays by the player being imitated, with map-level splits."""
import gzip
import hashlib
import json
import re
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

from seedcup.game import SIDES, rules
from seedcup.game.replay import FORMAT_VERSION

from . import online
from . import sampling_opportunities as opportunities
from .sampling_probes import aggregate_coverage, coverage, intervention_anchor
from .workspace import read_json, write_json


@contextmanager
def database(root):
    folder = root / ".local/sampling"
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "samples.sqlite3", timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS layouts(grid TEXT PRIMARY KEY, split TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS examples(
                id INTEGER, target TEXT, entry TEXT NOT NULL, coverage TEXT NOT NULL, PRIMARY KEY(id,target));
            CREATE TABLE IF NOT EXISTS runs(
                id TEXT PRIMARY KEY, plan TEXT NOT NULL, state TEXT NOT NULL, step INTEGER NOT NULL DEFAULT 0,
                paused INTEGER NOT NULL DEFAULT 0, message TEXT NOT NULL DEFAULT '', updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS slots(
                run TEXT, step INTEGER, side TEXT, id INTEGER NOT NULL, PRIMARY KEY(run,step,side), UNIQUE(run,id));
            CREATE TABLE IF NOT EXISTS sampling_maps(
                run TEXT, map_index INTEGER, decision TEXT NOT NULL, PRIMARY KEY(run,map_index));
            CREATE TABLE IF NOT EXISTS sampling_holdouts(grid TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS sampling_reserved_versions(version TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS sdk_observations(
                id INTEGER PRIMARY KEY, target TEXT NOT NULL, entry TEXT NOT NULL);
        """)
        with db:
            yield db
    finally:
        db.close()


def target_key(value):
    if value not in online.PRACTICE_LEVELS and not re.fullmatch(r"team:[1-9][0-9]*", value):
        raise ValueError("对手使用练习档位或 team:<官网队伍ID>")
    return value


def target_side(meta, target):
    if target in online.PRACTICE_LEVELS:
        return next((s for s in SIDES if meta[s]["bot"] is True and
                     meta[s]["name"] == online.PRACTICE_NAMES[target]), None)
    team = int(target_key(target).split(":")[1])
    return next((s for s in SIDES if meta[s]["bot"] is False and meta[s]["id"] == team), None)


def initialize_splits(root):
    """Preserve the original experiment's train/development/blind map assignments."""
    manifest = root / "published/datasets/boss-behavior-20261125.json"
    if not manifest.is_file():
        return
    with database(root) as db:
        if db.execute("PRAGMA user_version").fetchone()[0] == 1:
            return
        meta = read_json(manifest)
        additional = [mid for batch in meta["added_train"].values() for mid in batch]
        by_split = {"train": meta["initial_train"] + additional,
                    "development": meta["development"], "blind": meta["blind"]}
        fixed = {}
        with ZipFile(manifest.with_suffix(".zip")) as archive:
            for split, ids in by_split.items():
                for mid in ids:
                    data = json.loads(gzip.decompress(archive.read(f"online/replays/{mid}.json.gz")))
                    key = layout_hash(data["map"])
                    if fixed.setdefault(key, split) != split:
                        raise ValueError(f"历史地图跨越多个分组 {mid}")
        db.executemany("INSERT OR IGNORE INTO layouts VALUES (?,?)", fixed.items())
        db.execute("PRAGMA user_version=1")


def layout_hash(game_map):
    return hashlib.sha256(game_map["grid"].encode("ascii")).hexdigest()


def sdk_layout_hash(start: dict) -> str:
    """Hash an SDK grid in the same row-major format as an official replay."""
    grid = "".join(str(int(cell)) for row in start["map"]["grid"] for cell in row)
    return layout_hash({"grid": grid})


def default_split(grid: str) -> str:
    return {0: "blind", 1: "development"}.get(int(grid[:8], 16) % 10, "train")


def _frozen_collection_case(db, practice):
    tag = (practice or {}).get("case_tag")
    if tag is None or not online._SAMPLE_TAG.fullmatch(tag):
        return False
    run, step = tag.rsplit("-", 1)
    return db.execute("""SELECT 1 FROM runs r JOIN sampling_maps m ON m.run=r.id
        AND m.map_index=CAST(? AS INTEGER)/json_extract(r.plan,'$.width')
        WHERE r.id=? AND json_extract(m.decision,'$.protocol')=2""", (step, run)).fetchone() is not None


def ingest(root, raw, target, *, match_id=None, listing=None, practice=None, probe=None, collection=None):
    """Validate identity before persisting raw bytes. A later snapshot can enrich a passive record."""
    target_key(target)
    replay = json.loads(gzip.decompress(raw))
    meta, game_map = replay["match"], replay["map"]
    mid = meta["id"]
    if type(mid) is not int or mid <= 0 or (match_id is not None and mid != match_id):
        raise ValueError(f"回放对局 ID 不符 {mid}")
    if meta["kind"] not in ("room", "ladder") or replay["format"] != FORMAT_VERSION or replay["rules"] != rules.as_dict():
        raise ValueError(f"对局 {mid} 的类型、规则或回放格式与采样设置不符")
    side = target_side(meta, target)
    if side is None:
        raise ValueError(f"对局 {mid} 中没有指定对手 {target}")
    other = SIDES[1 - SIDES.index(side)]
    if listing and (listing["id"] != mid or listing["map_seed"] != game_map["seed"] or listing["kind"] != meta["kind"]
                    or any(listing[f"{s}_id"] != meta[s]["id"] for s in SIDES)):
        raise ValueError(f"对局 {mid} 的名单或地图与官网记录不一致")
    if practice:
        if "team_id" not in practice:
            # Early workbench records predate room and team IDs. Bind them to the official result.
            if not listing or practice["result"] != replay["result"]:
                raise ValueError(f"旧练习记录 {mid} 与官网结果不符")
            practice = practice | {"team_id": listing[f"{other}_id"], "room": meta["room"],
                                   "map_seed": listing["map_seed"] if practice["seed"] is None else practice["seed"]}
        if (practice["id"] != mid or practice["side"] != other or practice["level"] != target
                or practice["team_id"] != meta[other]["id"] or practice["map_seed"] != game_map["seed"]
                or practice["room"] != meta["room"]):
            raise ValueError(f"对局 {mid} 与采样房记录不一致")
    if len(replay["frames"]) != replay["result"]["ticks"] + 1:
        raise ValueError(f"对局 {mid} 的回放不完整")
    digest = hashlib.sha256(raw).hexdigest()
    grid = layout_hash(game_map)
    path = online.practice_replay_path(root, mid)
    stats = coverage(replay, side)
    stats[opportunities.CACHE_KEY] = opportunities.extract(replay, side)
    if collection is not None:
        collection = collection | {"actual_grid": grid, "layout_match": grid == collection["planned_grid"]}
        if probe in ("window-fire", "window-hold"):
            collection["intervention_anchor"] = intervention_anchor(replay, other)
    initialize_splits(root)
    with database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        previous = db.execute("SELECT entry FROM examples WHERE id=? AND target=?", (mid, target)).fetchone()
        prior = json.loads(previous[0]) if previous else {}
        if prior and prior["sha256"] != digest:
            raise ValueError(f"同一官网对局的回放内容已变更 {mid}")
        db.execute("INSERT OR IGNORE INTO layouts VALUES (?,?)", (grid, default_split(grid)))
        split = db.execute("SELECT split FROM layouts WHERE grid=?", (grid,)).fetchone()[0]
        entry = {"id": mid, "target": target, "target_side": side, "seed": game_map["seed"],
                 "game_no": meta["game"], "players": {s: {k: meta[s][k] for k in ("id", "name", "bot")} for s in SIDES},
                 "started_at": listing["started_at"] if listing else prior.get("started_at"),
                 "grid_sha256": grid, "split": split, "sha256": digest, "rules": replay["rules"]["version"],
                 "path": path.relative_to(root).as_posix(), "probe": probe or prior.get("probe"),
                 "version_id": practice["version_id"] if practice else prior.get("version_id")}
        if collection is not None or "collection" in prior:
            entry["collection"] = collection if collection is not None else prior["collection"]
        elif prior.get("unassigned_collection") or _frozen_collection_case(db, practice):
            # Retrying a room can create another game on an already archived side.
            entry["unassigned_collection"] = True
        timing = (practice or {}).get("sdk_timing", prior.get("sdk_timing"))
        if timing is not None:
            entry["sdk_timing"] = timing
        if path.is_file():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"本机回放文件与来源不符 {mid}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
                staged = Path(tmp) / path.name
                staged.write_bytes(raw)
                staged.replace(path)
        db.execute("INSERT OR REPLACE INTO examples VALUES (?,?,?,?)",
                   (mid, target, json.dumps(entry), json.dumps(stats)))
    return entry


def pull(root, target, *, client=None):
    target_key(target)
    client = client or online.sampling_client(root)
    own_id = client.user["id"]
    if target == f"team:{own_id}":
        raise ValueError("请指定对手的官网队伍 ID")
    matches = online.matches(root, client=client)
    recent = [r["id"] for r in matches]
    with database(root) as db:
        placeholders = ",".join("?" for _ in recent)
        known = {r[0] for r in db.execute(f"SELECT id FROM examples WHERE target=? AND id IN ({placeholders})",
                                          (target, *recent))} if recent else set()
    added, pending = [], []
    for row in matches:
        mid = row["id"]
        if row["status"] != "finished" or (mid in known and online.practice_replay_path(root, mid).is_file()):
            continue
        sides = [s for s in SIDES if row[f"{s}_id"] == own_id]
        if len(sides) != 1:
            continue
        opponent = SIDES[1 - SIDES.index(sides[0])]
        opponent_id = row[f"{opponent}_id"]
        if target in online.PRACTICE_LEVELS:
            if opponent_id is not None or row[f"{opponent}_name"] != online.PRACTICE_NAMES[target]:
                continue
        elif target != f"team:{opponent_id}":
            continue
        saved = online.practice_replay_path(root, mid)
        try:
            raw = saved.read_bytes() if saved.is_file() else client.request("GET", f"/api/replays/{mid}", compressed=True)
        except online.RemoteError as error:
            if error.status != 404 or str(error) != "回放还在生成，稍后再试":
                raise
            pending.append(mid)
            continue
        saved_record = None
        if target in online.PRACTICE_LEVELS:
            path = next((root / ".local/online/practice").glob(f"*/{target}/{mid}.json"), None)
            if path is not None:
                saved_record = read_json(path)
        ingest(root, raw, target, match_id=mid, listing=row, practice=saved_record)
        added.append(mid)
    return {"target": target, "added": len(added), "ids": added, "pending": pending}


def dataset(root, target, *, run=None, split=None, output=None):
    target_key(target)
    query = f"SELECT e.entry,e.coverage,({opportunities.TRAIN_LAYOUT}) AS eligible_for_training"
    query += (",s.step,json_extract(e.coverage,'$.probe.shots') AS probe_shots" if run else "") + " FROM examples e"
    params = []
    if run:
        query += " JOIN slots s ON e.id=s.id AND s.run=?"
        params.append(run)
    query += " WHERE e.target=?"
    params.append(target)
    if split == "train":
        query += f" AND ({opportunities.TRAIN_LAYOUT})"
    elif split:
        query += " AND json_extract(e.entry,'$.split')=?"
        params.append(split)
    query += " ORDER BY e.id"

    def summarize(db, stream=None):
        maps = set()
        observations, updates = [], []
        rows = db.execute(query, params)
        pairs = {}
        if run:
            rows = rows.fetchall()
            pairs, pair_summary = opportunities.comparisons(
                json.loads(row["entry"]) | {"probe_shots": row["probe_shots"]} for row in rows)

        def counts():
            for index, row in enumerate(rows):
                entry = json.loads(row["entry"])
                grid = entry["grid_sha256"]
                maps.add(grid)
                replay_coverage = json.loads(row["coverage"])
                records = opportunities.for_entry(root, entry, replay_coverage.get(opportunities.CACHE_KEY), updates)
                if records is not None:
                    observations.append((grid, entry["version_id"], records))
                if stream is not None:
                    if "collection" in entry:
                        entry["repeat"] = entry["collection"]["repeat"]
                    elif run_plan and "repeats" in run_plan:
                        entry["repeat"] = 1 + row["step"] % run_plan["width"] // len(run_plan["probes"])
                    entry["eligible_for_training"] = bool(row["eligible_for_training"])
                    entry[opportunities.CACHE_KEY] = records
                    if entry["id"] in pairs:
                        entry["comparison"] = pairs[entry["id"]]
                    if index:
                        stream.write(",\n")
                    json.dump(entry, stream, ensure_ascii=False)
                yield replay_coverage

        stats = aggregate_coverage(counts())
        opportunities.save_cache(db, updates)
        result = {"target": target, "games": stats["games"], "maps": len(maps), "coverage": stats,
                  opportunities.CACHE_KEY: opportunities.summarize(observations),
                  "opportunities_unavailable_games": stats["games"] - len(observations)}
        if run:
            result["comparisons"] = pair_summary
        return result

    with database(root) as db:
        run_plan = None
        if run:
            row = db.execute("SELECT plan FROM runs WHERE id=?", (run,)).fetchone()
            if not row:
                raise ValueError(f"找不到采样任务 {run}")
            run_plan = json.loads(row["plan"])
        if output is None:
            return summarize(db)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            staged = Path(tmp) / "manifest.json"
            with staged.open("w", encoding="utf-8") as stream:
                stream.write("[\n")
                result = summarize(db, stream)
                stream.write("\n]\n")
            staged.replace(output)
    return result | {"path": str(output)}


def sdk_dataset(root: Path, target: str, *, run: str | None = None, split: str | None = None,
                output: Path | None = None) -> dict:
    """Export an index of the observation streams captured by the SDK client."""
    target_key(target)
    initialize_splits(root)
    query = "SELECT o.entry FROM sdk_observations o"
    params = [target]
    if run is not None:
        query += " JOIN slots s ON o.id=s.id AND s.run=?"
        params.insert(0, run)
    query += " WHERE o.target=? ORDER BY o.id"
    with database(root) as db:
        if run is not None and not db.execute("SELECT 1 FROM runs WHERE id=?", (run,)).fetchone():
            raise ValueError(f"找不到采样任务 {run}")
        layouts = dict(db.execute("SELECT grid,split FROM layouts"))
        holdouts = {row[0] for row in db.execute("SELECT grid FROM sampling_holdouts")}
        reserved = {row[0] for row in db.execute("SELECT version FROM sampling_reserved_versions")}
        entries = []
        for row in db.execute(query, params):
            entry = json.loads(row["entry"])
            if "grid_sha256" not in entry:
                # Early receipts predate the layout metadata. Recover it from the raw bytes.
                raw = (root / entry["observation_file"]).read_bytes()
                if hashlib.sha256(raw).hexdigest() != entry["observation_sha256"]:
                    raise ValueError(f"SDK 观测文件校验失败 {entry['id']}")
                entry["grid_sha256"] = sdk_layout_hash(json.loads(gzip.decompress(raw))["start"])
            grid = entry["grid_sha256"]
            actual_split = layouts.get(grid, default_split(grid))
            collection = entry.get("collection")
            if collection:
                collection.update(actual_grid=grid, layout_match=grid == collection["planned_grid"])
            eligible = (actual_split == "train" and grid not in holdouts and entry["version_id"] not in reserved
                        and entry["probe"] != "holdout-orbit" and not entry.get("unassigned_collection")
                        and (collection is None or (collection["planned_split"] == "train"
                                                    and collection["layout_match"])))
            matches_split = eligible if split == "train" else split is None or split == actual_split
            if not matches_split:
                continue
            entry.setdefault("target_side", SIDES[1 - SIDES.index(entry["side"])])
            entry.update(target=target, split=actual_split, eligible_for_training=eligible)
            entries.append(entry)
    result = {"source": "sdk", "target": target, "games": len(entries),
              "maps": len({entry["grid_sha256"] for entry in entries})}
    if output is not None:
        write_json(output, entries)
        result["path"] = str(output)
    return result
