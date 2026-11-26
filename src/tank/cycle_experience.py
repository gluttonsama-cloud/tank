"""Official game evidence for strategy iteration and targeted sampling."""

import gzip
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from itertools import pairwise
from math import cos, dist, isfinite, radians, sin
from pathlib import Path
from time import time

from seedcup.game import SIDES, GameMap
from seedcup.game.engine import ang_diff, line_of_sight
from seedcup.game.replay import FORMAT_VERSION

from . import sampling_data, versions
from .online import PRACTICE_NAMES
from .workspace import read_json

EVENTS = ("first_engagement", "consecutive_misses", "damage_exchange",
          "breaking_contact", "zone_displacement")
LEVEL_BY_NAME = {name: level for level, name in PRACTICE_NAMES.items()}
INDEX_VERSION = 5
EVENTS_PER_GAME = 3
MIN_TRAIN_MAPS = 4


@dataclass(slots=True)
class Frame:
    tick: int
    tanks: tuple
    shells: tuple
    radius: float
    fired: tuple = ()
    damage: tuple | None = None
    actions: dict | None = None


def _database(root):
    path = root / ".local/cycle/experience.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS games(
        id INTEGER PRIMARY KEY, fingerprint TEXT NOT NULL, report TEXT, error TEXT,
        ingested_at REAL NOT NULL DEFAULT 0)""")
    if "ingested_at" not in {row["name"] for row in db.execute("PRAGMA table_info(games)")}:
        with db:
            db.execute("BEGIN IMMEDIATE")
            if "ingested_at" not in {row["name"] for row in db.execute("PRAGMA table_info(games)")}:
                db.execute("ALTER TABLE games ADD COLUMN ingested_at REAL NOT NULL DEFAULT 0")
    return db


def _sampling_policy(db):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    layouts = dict(db.execute("SELECT grid, split FROM layouts"))
    holdouts, reserved = set(), set()
    if "sampling_holdouts" in tables:
        holdouts = {row[0] for row in db.execute("SELECT grid FROM sampling_holdouts")}
    if "sampling_reserved_versions" in tables:
        reserved = {row[0] for row in db.execute("SELECT version FROM sampling_reserved_versions")}
    return tables, layouts, holdouts, reserved


def _archive_index(root):
    indexed = defaultdict(dict)
    layouts, holdouts, reserved = {}, set(), set()
    path = root / ".local/sampling/samples.sqlite3"
    if path.is_file():
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db, db:
            db.execute("BEGIN")
            tables, layouts, holdouts, reserved = _sampling_policy(db)
            for mid, raw in db.execute("SELECT id, entry FROM examples ORDER BY id, target"):
                indexed[mid].setdefault("examples", []).append(json.loads(raw))
            if "sdk_observations" in tables:
                for mid, raw in db.execute("SELECT id, entry FROM sdk_observations ORDER BY id"):
                    indexed[mid]["sdk"] = json.loads(raw)
    for path in (root / ".local/online/replays").glob("*.json.gz"):
        name = path.name.removesuffix(".json.gz")
        if name.isdecimal() and int(name) > 0:
            indexed[int(name)]["replay"] = path.relative_to(root).as_posix()
    for path in sorted((root / ".local/online/practice").glob("*/*/*.json")):
        if path.stem.isdecimal() and int(path.stem) in indexed:
            indexed[int(path.stem)].setdefault("practice", []).append(path.relative_to(root).as_posix())
    for path in (root / ".local/cycle/ladder/games").glob("*.json"):
        if path.stem.isdecimal() and int(path.stem) > 0:
            indexed[int(path.stem)]["ladder"] = path.relative_to(root).as_posix()
    return indexed, layouts, holdouts, reserved


def _path_signature(root, path):
    try:
        stat = (root / path).stat()
    except FileNotFoundError:
        return [str(path), None]
    return [str(path), stat.st_size, stat.st_mtime_ns]


def _fingerprint(root, sources):
    files = [*sources.get("practice", []), *(entry["path"] for entry in sources.get("examples", []))]
    if "replay" in sources:
        files.append(sources["replay"])
    if "sdk" in sources:
        files.append(sources["sdk"]["observation_file"])
    if "ladder" in sources:
        files.append(sources["ladder"])
        try:
            ladder = read_json(root / sources["ladder"])
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        else:
            if ladder.get("replay_path"):
                files.append(ladder["replay_path"])
    payload = [INDEX_VERSION, sources, [_path_signature(root, path) for path in sorted(set(files))]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _read_payload(root, path, source, expected=None):
    path = root / path
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected is not None and checksum != expected:
        raise ValueError(f"{source} 文件校验失败 {path.name}")
    data = json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)
    return data, {"source": source, "path": path.relative_to(root).as_posix(), "sha256": checksum}


def _layout(grid, layouts, holdouts, reserved, collection, bindings, excluded):
    split = layouts.get(grid, sampling_data.default_split(grid))
    mismatch = any(item["planned_grid"] != grid or item["planned_split"] != "train" for item in collection)
    eligible = (split == "train" and grid not in holdouts and not excluded and not mismatch
                and all(item["id"] not in reserved for item in bindings.values()))
    return {"grid_sha256": grid, "split": split, "eligible_for_training": eligible}


def _bind(bindings, side, version_id, source, **metadata):
    if not version_id:
        return
    previous = bindings.get(side)
    if previous and previous["id"] != version_id:
        raise ValueError(f"{side} 方有冲突的版本归属")
    evidence = {"id": version_id, "reliability": "verified", "source": source} | metadata
    if not previous:
        bindings[side] = evidence


def _validate_replay(mid, replay):
    if replay["format"] != FORMAT_VERSION or replay["match"]["id"] != mid:
        raise ValueError(f"官网回放的格式或对局 ID 不符 {mid}")
    result, frames = replay["result"], replay["frames"]
    if not result or len(frames) != result["ticks"] + 1 or any(frame[0] != tick for tick, frame in enumerate(frames)):
        raise ValueError(f"官网回放缺少终局或帧号不连续 {mid}")


def _validate_sdk(mid, data, entry):
    start, end, practice = (data[key] for key in ("start", "end", "practice"))
    ticks = [obs["frame"] for obs, _ in data["frames"]]
    if (data["format"] != "seedcup-sdk-observations-v1"
            or start["match_id"] != mid or end["match_id"] != mid or entry["id"] != mid
            or entry["side"] != start["side"] or end["side"] != start["side"]
            or practice["team_id"] != entry["team_id"] or practice["version_id"] != entry["version_id"]
            or practice["level"] != entry["level"] or practice["room"] != entry["room"]
            or start["map"]["seed"] != entry["map_seed"] or end["result"] != entry["result"]
            or not ticks or not 0 <= ticks[0] <= ticks[-1] <= end["result"]["ticks"]
            or any(a >= b for a, b in pairwise(ticks))
            or any(obs["match_id"] != mid for obs, _ in data["frames"])):
        raise ValueError(f"SDK 观测与采样记录不符 {mid}")


def _replay_sources(root, mid, sources):
    ladder = read_json(root / sources["ladder"]) if "ladder" in sources else None
    if ladder is not None and (ladder["game_id"] != mid or ladder["status"] != "archived"):
        raise ValueError(f"天梯回放归档记录不完整 {mid}")
    paths = {}
    if "replay" in sources:
        paths[sources["replay"]] = ("official_replay", None)
    for entry in sources.get("examples", []):
        path = entry["path"]
        previous = paths.get(path, (None, None))[1]
        if previous and previous != entry["sha256"]:
            raise ValueError(f"采样表有不同的官网回放校验值 {mid}")
        paths[path] = ("official_replay", entry["sha256"])
    if ladder is not None:
        path = (root / ladder["replay_path"]).relative_to(root).as_posix()
        previous = paths.get(path, (None, None))[1]
        if previous and previous != ladder["replay_sha256"]:
            raise ValueError(f"天梯与采样表的官网回放校验值不一致 {mid}")
        paths[path] = ("ladder_replay", ladder["replay_sha256"])

    primary, provenance = None, []
    for path, (source, checksum) in paths.items():
        replay, evidence = _read_payload(root, path, source, checksum)
        _validate_replay(mid, replay)
        if primary is not None and primary != replay:
            raise ValueError(f"同一官网对局的回放内容不一致 {mid}")
        primary = replay
        provenance.append(evidence)
    return primary, ladder, provenance


def _practice_binding(root, mid, path, replay, bindings):
    entry = read_json(root / path)
    players = replay["match"]
    side = entry["side"]
    seed = entry["seed"] if "map_seed" not in entry else entry["map_seed"]
    other = SIDES[1 - SIDES.index(side)]
    if (entry["id"] != mid or entry["result"] != replay["result"]
            or (seed is not None and seed != replay["map"]["seed"])
            or not players[other]["bot"] or players[other]["name"] != PRACTICE_NAMES[entry["level"]]
            or players[side]["bot"] or entry.get("team_id", players[side]["id"]) != players[side]["id"]
            or entry.get("room", players["room"]) != players["room"]):
        raise ValueError(f"旧练习记录与官网回放不符 {mid} {path}")
    _bind(bindings, side, entry["version_id"], "practice_result", path=path)


def _replay_metadata(root, mid, replay, ladder, sources, bindings):
    meta, game_map = replay["match"], replay["map"]
    players = {side: meta[side] for side in SIDES}
    grid = sampling_data.layout_hash(game_map)
    for entry in sources.get("examples", []):
        if (entry["id"] != mid or entry["grid_sha256"] != grid or entry["seed"] != game_map["seed"]
                or sampling_data.target_side(meta, entry["target"]) != entry["target_side"]):
            raise ValueError(f"采样索引与官网回放不符 {mid}")
        # Only human-versus-bot practice entries carry a verified local version.
        if entry["target"] in PRACTICE_NAMES:
            side = SIDES[1 - SIDES.index(entry["target_side"])]
            _bind(bindings, side, entry["version_id"], "indexed_practice_replay", target=entry["target"])
    if ladder is not None:
        side = ladder["own_side"]
        other = SIDES[1 - SIDES.index(side)]
        if (meta["kind"] != "ladder" or meta[side]["bot"] or meta[other]["bot"]
                or meta[side]["id"] != ladder["team_id"] or meta[other]["id"] != ladder["opponent_id"]):
            raise ValueError(f"天梯归档的队伍或出生侧与回放不一致 {mid}")
        if ladder["submission_id"]:
            _bind(bindings, side, ladder["version_id"], "ladder_submission",
                  submission_id=ladder["submission_id"], attribution=ladder.get("attribution"))
    for path in sources.get("practice", []):
        _practice_binding(root, mid, path, replay, bindings)
    targets = {entry["target"] for entry in sources.get("examples", [])}
    targets.update(LEVEL_BY_NAME[player["name"]] for player in players.values()
                   if player["bot"] and player["name"] in LEVEL_BY_NAME)
    targets.update(f"team:{player['id']}" for player in players.values() if not player["bot"] and player["id"])
    return players, grid, sorted(targets)


def _zone_radius(rules, tick):
    zone = rules["zone"]
    time = tick / rules["tick_hz"]
    shrink = (time - zone["shrink_start"]) / (zone["shrink_end"] - zone["shrink_start"])
    return zone["start_radius"] + (zone["final_radius"] - zone["start_radius"]) * max(0, min(1, shrink))


def _replay_frames(data):
    for tick, red, blue, shells, effects in data["frames"]:
        yield Frame(tick, (tuple(red[:6]), tuple(blue[:6])), tuple(shell[4] for shell in shells),
                    _zone_radius(data["rules"], tick), tuple(side for effect, side, *_ in effects if effect == "fire"),
                    (red[8], blue[8]))


def _sdk_frames(data):
    side = SIDES.index(data["start"]["side"])
    last = None
    for obs, sent in data["frames"]:
        tick = obs["frame"]
        tanks = [tuple(tank[field] for field in ("x", "y", "heading", "hp", "alive"))
                 + (tank.get("reload"),) for tank in (obs["self"], obs["enemy"])]
        fired = (side,) if (last is not None and tick == last[0] + 1
                            and tanks[0][5] > last[1]) else ()
        yield Frame(tick, tuple(tanks if side == 0 else reversed(tanks)),
                    tuple(side if shell["mine"] else 1 - side for shell in obs["shells"]),
                    obs["zone"]["radius"], fired,
                    actions={SIDES[side]: {"sent": sent, "last_applied": obs["last_action"]}})
        last = tick, tanks[0][5]


def _sdk_map(start):
    data = start["map"]
    grid = bytes(int(cell) for row in data["grid"] for cell in row)
    spawn = start["spawn"]
    return GameMap.from_dict({"seed": data["seed"], "grid": "".join(map(str, grid)), "spawns": [
        spawn, {"x": data["width"] - spawn["x"], "y": spawn["y"], "heading": (180 - spawn["heading"]) % 360},
    ]})


def _distance(frame):
    return dist(frame.tanks[0][:2], frame.tanks[1][:2])


def _margin(frame, i, center):
    return frame.radius - dist(frame.tanks[i][:2], center)


def _damage(before, current, center):
    if current.damage is not None:
        return tuple(round(after - earlier, 2) for earlier, after in zip(before.damage, current.damage))
    # Inside the current and previous safe area, observed HP losses imply projectile damage.
    damage = []
    for i in range(2):
        inside = _margin(before, i, center) > .02 and _margin(current, i, center) > .02
        loss = before.tanks[i][3] - current.tanks[i][3]
        damage.append(round(max(0, loss), 2) if inside else 0)
    return tuple(reversed(damage))


def _event_snapshot(frames, index, tick_hz, center, sdk_by_tick, game_map):
    frame = frames[index]
    previous = frames[index - 1] if index else None
    sides = {}
    for i, side in enumerate(SIDES):
        tank = frame.tanks[i]
        state = {"pose": list(tank[:3]), "hp": tank[3], "alive": bool(tank[4]),
                 "reload_seconds": tank[5] / tick_hz if frame.damage is not None else tank[5],
                 "zone_margin": round(_margin(frame, i, center), 3)}
        action = {}
        if previous is not None and frame.tick == previous.tick + 1:
            last = previous.tanks[i]
            angle = radians(tank[2])
            along = (tank[0] - last[0]) * cos(angle) + (tank[1] - last[1]) * sin(angle)
            action["inferred_motion"] = {
                "forward_cells": round(along, 3),
                "heading_change_deg": round(ang_diff(tank[2], last[2]), 3),
            }
        if frame.damage is not None or i in frame.fired:
            action["shot_observed"] = i in frame.fired
            action["shot_evidence"] = "replay_fire_effect" if frame.damage is not None else "sdk_reload_increase"
        action.update(sdk_by_tick.get(frame.tick, {}).get(side, {}))
        if action:
            state["actions"] = action
        sides[side] = state
    return {"frame": frame.tick, "tanks": sides, "distance": round(_distance(frame), 3),
            "line_of_sight": line_of_sight(game_map, *frame.tanks[0][:2], *frame.tanks[1][:2]),
            "shells": {side: frame.shells.count(i) for i, side in enumerate(SIDES)},
            "zone_radius": round(frame.radius, 3)}


def _extract_events(frames, game_map, rules, sdk_by_tick):
    tick_hz = rules["tick_hz"]
    center = rules["zone"]["x"], rules["zone"]["y"]
    candidates = defaultdict(list)
    positions = {frame.tick: i for i, frame in enumerate(frames)}
    closed_at = None
    damage_at = [None, None]
    active = [None, None]
    misses = [[], []]
    outside_at = [None, None]

    def record(event, index, anchor, strength, **details):
        candidates[event].append((strength, anchor, index, details))

    def finish_misses(i):
        streak = misses[i]
        if len(streak) >= 3:
            record("consecutive_misses", streak[-1][1], streak[-3][0], len(streak),
                   side=SIDES[i], streak=len(streak),
                   last_three_fire_frames=[frames[start].tick for start, _ in streak[-3:]],
                   evidence="isolated_projectiles")
        streak.clear()

    for index, current in enumerate(frames):
        distance = _distance(current)
        both_alive = all(tank[4] for tank in current.tanks)
        previous = frames[index - 1] if index else None
        continuous = previous is not None and previous.tick + 1 == current.tick

        if not candidates["first_engagement"]:
            firing = bool(current.fired or current.shells)
            close = both_alive and distance <= 5 and line_of_sight(
                game_map, *current.tanks[0][:2], *current.tanks[1][:2])
            if firing or close:
                record("first_engagement", index, max(0, index - 1), 1,
                       trigger="projectile" if firing else "close_line_of_sight",
                       first_observation_is_frame_zero=frames[0].tick == 0,
                       uninterrupted_to_trigger=index == current.tick - frames[0].tick)

        if not continuous:
            for i in range(2):
                finish_misses(i)
            active = [None, None]
            outside_at, damage_at = [None, None], [None, None]
            closed_at = None

        if both_alive and distance <= 3:
            closed_at = index
        elif both_alive and closed_at is not None:
            earlier = frames[closed_at]
            if current.tick - earlier.tick > 3 * tick_hz:
                closed_at = None
            elif distance >= 5:
                contributions = [(dist(current.tanks[i][:2], earlier.tanks[1 - i][:2])
                                  - _distance(earlier) + distance
                                  - dist(earlier.tanks[i][:2], current.tanks[1 - i][:2])) / 2
                                 for i in range(2)]
                order = max(range(2), key=contributions.__getitem__)
                record("breaking_contact", index, closed_at, distance - _distance(earlier),
                       sides_movement_contribution={side: round(gain, 3) for side, gain in zip(SIDES, contributions)},
                       main_separating_side=SIDES[order])
                closed_at = None

        if not continuous:
            continue
        dealt = _damage(previous, current, center)
        for i, amount in enumerate(dealt):
            if amount > 0:
                damage_at[i] = index, amount
        if any(dealt) and all(item is not None for item in damage_at):
            first, second = damage_at
            if abs(frames[first[0]].tick - frames[second[0]].tick) <= 2 * tick_hz:
                record("damage_exchange", index, min(first[0], second[0]), first[1] + second[1],
                       damage_by_side={SIDES[0]: first[1], SIDES[1]: second[1]},
                       hit_frames={SIDES[0]: frames[first[0]].tick, SIDES[1]: frames[second[0]].tick},
                       damage_evidence="replay_damage_counter" if current.damage is not None else "sdk_in_zone_hp_loss")
                damage_at = [None, None]

        for i in range(2):
            side = SIDES[i]
            # Attribute only isolated shots. An overlap, an SDK gap or an unsafe
            # health observation invalidates a streak instead of guessing hits.
            shell_count = current.shells.count(i)
            if i in current.fired:
                if active[i] is not None or i in previous.shells or shell_count > 1:
                    active[i] = None
                    finish_misses(i)
                else:
                    active[i] = index
            if (active[i] is not None and current.damage is None
                    and min(_margin(current, 1 - i, center), _margin(previous, 1 - i, center)) <= .02):
                active[i] = None
                finish_misses(i)
            if active[i] is not None and (dealt[i] or shell_count == 0):
                fired_at = active[i]
                active[i] = None
                if dealt[i]:
                    finish_misses(i)
                else:
                    misses[i].append((fired_at, index))
            elif dealt[i]:
                finish_misses(i)

            if previous.radius >= rules["zone"]["start_radius"] and current.radius >= previous.radius:
                continue
            was_safe = _margin(previous, i, center) >= 0
            safe = _margin(current, i, center) >= 0
            if current.tanks[i][4] and was_safe and not safe:
                outside_at[i] = index
                record("zone_displacement", index, index - 1, 1,
                       side=side, transition="left_safe_zone",
                       zone_shrunk=current.radius < previous.radius)
            elif outside_at[i] is not None and (safe or not current.tanks[i][4]):
                entry = outside_at[i]
                outside_at[i] = None
                seconds = (current.tick - frames[entry].tick) / tick_hz
                record("zone_displacement", index, entry, 2 + seconds, side=side,
                       transition="reentered_safe_zone" if safe and current.tanks[i][4] else "died_outside",
                       seconds_outside=round(seconds, 2),
                       radial_progress=round(dist(frames[entry].tanks[i][:2], center)
                                             - dist(current.tanks[i][:2], center), 3))

    for i in range(2):
        finish_misses(i)
    events, counts = {}, {}
    for name, choices in candidates.items():
        counts[name] = len(choices)
        chosen = []
        for strength, anchor, index, details in sorted(choices, key=lambda v: (-v[0], v[2])):
            if any(abs(frames[index].tick - item["frame"]) <= tick_hz
                   and details.get("side") == item.get("side") for item in chosen):
                continue
            followup = positions.get(frames[index].tick + 2 * tick_hz)
            if followup is None and all(b.tick == a.tick + 1 for a, b in pairwise(frames[index:])):
                followup = len(frames) - 1
            snapshots = [_event_snapshot(frames, j, tick_hz, center, sdk_by_tick, game_map)
                         for j in dict.fromkeys((anchor, index, followup)) if j is not None]
            chosen.append({"frame": frames[index].tick, "anchor_frame": frames[anchor].tick,
                           "strength": round(strength, 3), "frames": snapshots, **details})
            if len(chosen) == EVENTS_PER_GAME:
                break
        events[name] = chosen
    return events, counts


def _game_report(root, mid, sources, layouts, holdouts, reserved):
    replay, ladder, provenance = _replay_sources(root, mid, sources)
    bindings, collections = {}, []
    excluded = False
    for entry in sources.get("examples", []):
        if "collection" in entry:
            collections.append(entry["collection"])
        excluded |= bool(entry.get("unassigned_collection") or entry.get("probe") == "holdout-orbit")
    sdk = None
    if "sdk" in sources:
        entry = sources["sdk"]
        sdk, evidence = _read_payload(root, entry["observation_file"], "sdk_observations", entry["observation_sha256"])
        _validate_sdk(mid, sdk, entry)
        start = sdk["start"]
        sdk_side = start["side"]
        sdk_grid = sampling_data.sdk_layout_hash(start)
        if entry.get("grid_sha256", sdk_grid) != sdk_grid:
            raise ValueError(f"SDK 观测地图不符 {mid}")
        _bind(bindings, sdk_side, entry["version_id"], "sdk_capture", case_tag=entry.get("case_tag"))
        if "collection" in entry:
            collections.append(entry["collection"])
        excluded |= bool(entry.get("unassigned_collection") or entry.get("probe") == "holdout-orbit")
        provenance.append(evidence | {"side": sdk_side, "observed_frames": len(sdk["frames"]),
                                      "missing_frame_count": sdk["end"]["result"]["ticks"] + 1 - len(sdk["frames"])})

    sdk_frames = list(_sdk_frames(sdk)) if sdk is not None else []
    if replay is not None:
        players, grid, targets = _replay_metadata(root, mid, replay, ladder, sources, bindings)
        if sdk is not None and (sdk_grid != grid or sdk["end"]["result"] != replay["result"]
                                or start["map"]["seed"] != replay["map"]["seed"]
                                or entry["team_id"] != players[sdk_side]["id"]):
            raise ValueError(f"SDK 观测与官网回放不一致 {mid}")
        game_map = GameMap.from_dict(replay["map"])
        rules, result = replay["rules"], replay["result"]
        kind = replay["match"]["kind"]
        frames = list(_replay_frames(replay))
    else:
        other = SIDES[1 - SIDES.index(sdk_side)]
        players = {sdk_side: {"id": entry["team_id"], "name": None, "bot": False},
                   other: {"id": None, "name": start["opponent"], "bot": True}}
        targets = [entry["level"], f"team:{entry['team_id']}"]
        grid, game_map = sdk_grid, _sdk_map(start)
        rules, result = start["rules"], sdk["end"]["result"]
        kind = start["kind"]
        frames = sdk_frames
    scope = "practice" if any(player["bot"] for player in players.values()) else "pvp"
    layout = _layout(grid, layouts, holdouts, reserved, collections, bindings, excluded)
    events, counts = _extract_events(frames, game_map, rules, {item.tick: item.actions for item in sdk_frames})
    report = {
        "game_id": mid, "scope": scope, "kind": kind, "targets": targets, "result": result,
        "sources": provenance, "players": players, "versions": bindings,
        "map": {"seed": game_map.seed, **layout}, "rules": rules["version"],
        "collections": collections, "collection_excluded": excluded, "event_counts": counts, "events": events,
        "observed_through_frame": frames[-1].tick,
    }
    if ladder is not None:
        report["ladder"] = {field: ladder[field] for field in ("series_id", "team_id", "opponent_id", "own_side", "role")}
    return report


def refresh(root, limit=128):
    """Index up to limit changed official games without contacting the service."""
    if type(limit) is not int or limit < 1:
        raise ValueError("limit 必须为正整数")
    root = Path(root).resolve()
    sources, layouts, holdouts, reserved = _archive_index(root)
    updated = 0
    errors = []
    with closing(_database(root)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        stored = {row["id"]: row for row in db.execute("""
            SELECT id, fingerprint, json_extract(report, '$.map') AS map,
                   json_extract(report, '$.collections') AS collections,
                   json_extract(report, '$.versions') AS versions,
                   json_extract(report, '$.collection_excluded') AS excluded
            FROM games""")}
        last = db.execute("SELECT max(ingested_at) FROM games").fetchone()[0]
        stamp = max(time(), (last or 0) + .000001)
        changed = []
        for mid, reports in sorted(sources.items(), reverse=True):
            fingerprint = _fingerprint(root, reports)
            old = stored.get(mid)
            if old is None or old["fingerprint"] != fingerprint:
                changed.append((mid, reports, fingerprint))
            elif old["map"] is not None:
                layout = json.loads(old["map"])
                current = _layout(layout["grid_sha256"], layouts, holdouts, reserved, json.loads(old["collections"]),
                                  json.loads(old["versions"]), old["excluded"])
                if layout != {"seed": layout["seed"], **current}:
                    changed.append((mid, reports, fingerprint))
        for mid, reports, fingerprint in changed[:limit]:
            try:
                report = _game_report(root, mid, reports, layouts, holdouts, reserved)
            except (OSError, EOFError, IndexError, KeyError, ValueError, TypeError) as error:
                errors.append({"game_id": mid, "error": str(error)})
                db.execute("INSERT OR REPLACE INTO games VALUES (?, ?, NULL, ?, ?)", (mid, fingerprint, str(error), stamp))
            else:
                db.execute("INSERT OR REPLACE INTO games VALUES (?, ?, ?, NULL, ?)",
                           (mid, fingerprint, json.dumps(report, ensure_ascii=False, allow_nan=False), stamp))
                updated += 1
        totals = db.execute("SELECT count(report), count(error) FROM games").fetchone()
    return {"indexed": updated, "errors": errors, "games": totals[0], "failed_games": totals[1],
            "remaining": max(0, len(changed) - limit)}


def _team_matchup(game):
    players = game["players"]
    return tuple(sorted(f"bot:{player['name']}" if player["bot"] else f"team:{player['id']}"
                        for player in players.values()))


def indexed_grids(root):
    """Collect official layouts to exclude from fresh simulator holdout maps."""
    with closing(_database(Path(root).resolve())) as db:
        return {row[0] for row in db.execute(
            "SELECT DISTINCT json_extract(report, '$.map.grid_sha256') FROM games WHERE report IS NOT NULL")}


def _version_labels(root):
    return {item["id"]: item for item in versions.list_versions(root)}


def _line_filter(line, game, labels):
    if line in ("all", "*", game["scope"], game["kind"]) or line in game["targets"]:
        return True
    aliases = {"pve": "practice"}
    if aliases.get(line) == game["scope"]:
        return True
    # A project's registered source names and snapshots are authoritative. Its
    # corpus also retains the observed competitors; only its versions are scored.
    return any(line in (item["id"], labels.get(item["id"], {}).get("name")) for item in game["versions"].values())


def _project_line(root, line):
    for path in (root / "projects").glob("*.json"):
        project = read_json(path)
        if line in (path.stem, project["name"], *project["sources"]):
            return set(project["sources"]), project.get("purpose")
    return {line}, None


def _version_rows(games, labels, line, source_names):
    stats = defaultdict(lambda: {"games": 0, "wins": 0, "draws": 0, "losses": 0, "maps": set(),
                                 "opponents": set(), "sides": Counter(), "scores": []})
    for game in games:
        for side, binding in game["versions"].items():
            key = binding["id"]
            label = labels.get(key, {})
            if source_names is not None and key != line and label.get("name") not in source_names:
                continue
            i = SIDES.index(side)
            opponent = game["players"][SIDES[1 - i]]
            matchup = f"team:{opponent['id']}" if not opponent["bot"] else LEVEL_BY_NAME.get(opponent["name"], opponent["name"])
            segment = stats[key, game["scope"], matchup]
            segment["games"] += 1
            winner = game["result"]["winner_side"]
            segment["wins" if winner == side else "draws" if winner is None else "losses"] += 1
            segment["maps"].add(game["map"]["grid_sha256"])
            segment["opponents"].add(matchup)
            segment["sides"][side] += 1
            segment["scores"].append(game["result"]["scores"][i]["total"])
    rows = []
    for (key, scope, target), item in stats.items():
        scores, maps = item.pop("scores"), item.pop("maps")
        item.pop("opponents")
        rows.append({"version_id": key, "strategy": labels.get(key, {}).get("name"),
                     "scope": scope, "target": target, **item, "maps": len(maps),
                     "mean_score": round(sum(scores) / len(scores), 2),
                     "minimum_score": min(scores)})
    return sorted(rows, key=lambda row: (-row["games"], row["version_id"], row["scope"], row["target"]))


def _representative(games, event, limit=6, examples_by_id=None):
    choices = [(game, example) for game in games for example in game["events"].get(event, [])]
    if examples_by_id is not None:
        choices = [(game, examples_by_id[game["game_id"]]) for game in games if game["game_id"] in examples_by_id]
    chosen, maps, matchups, ids, outcomes = [], Counter(), Counter(), set(), Counter()
    while choices and len(chosen) < limit:
        choices = [pair for pair in choices if pair[0]["game_id"] not in ids]
        if not choices:
            break
        game, example = min(choices, key=lambda pair: (
            matchups[_team_matchup(pair[0])], maps[pair[0]["map"]["grid_sha256"]],
            outcomes[pair[0]["result"]["winner_side"]], -pair[1]["strength"],
            -pair[0]["game_id"], pair[1]["frame"]))
        ids.add(game["game_id"])
        maps[game["map"]["grid_sha256"]] += 1
        matchups[_team_matchup(game)] += 1
        outcomes[game["result"]["winner_side"]] += 1
        chosen.append({"game_id": game["game_id"], "players": game["players"], "map": game["map"],
                       "versions": game["versions"], "sources": game["sources"],
                       "winner_side": game["result"]["winner_side"],
                       "score": {side: game["result"]["scores"][i]["total"] for i, side in enumerate(SIDES)},
                       "end_reason": game["result"]["reason"], **example})
    return chosen


def _perspective(game):
    ladder = game.get("ladder")
    if ladder:
        return ladder["own_side"]
    human = [side for side, player in game["players"].items() if not player["bot"] and player["id"]]
    return human[0] if len(human) == 1 else None


def _hp_drop(example, side):
    first, last = example["frames"][0], example["frames"][-1]
    return round(max(0, first["tanks"][side]["hp"] - last["tanks"][side]["hp"]), 2)


def _case(game, event):
    side = _perspective(game)
    if side is None:
        return None
    examples = [item for item in game["events"].get(event, []) if item.get("side", side) == side]
    if not examples:
        return None
    example = max(examples, key=lambda item: (_hp_drop(item, side), item["strength"]))
    winner = game["result"]["winner_side"]
    return example, side, _hp_drop(example, side), winner == side, winner is not None and winner != side


def _top_findings(games):
    findings = []
    for scope in ("pvp", "practice"):
        group = [game for game in games if game["scope"] == scope]
        for event in EVENTS:
            cases = [(game, case) for game in group if (case := _case(game, event)) is not None]
            costly = [(game, case) for game, case in cases if case[4] or (case[3] and case[2] >= 10)]
            if not costly:
                continue
            teams = {player["id"] for game, _ in costly for player in game["players"].values() if not player["bot"]}
            maps = {game["map"]["grid_sha256"] for game, _ in costly}
            examples = {game["game_id"]: case[0] for game, case in costly}
            frames = _representative([game for game, _ in costly], event, limit=3, examples_by_id=examples)
            findings.append({
                "event": event, "scope": scope, "games": len(cases), "distinct_teams": len(teams),
                "distinct_maps": len(maps), "losses": sum(case[4] for _, case in costly),
                "costly_wins": sum(case[3] for _, case in costly),
                "mean_hp_drop": round(sum(case[2] for _, case in costly) / len(costly), 2),
                "costly_win_hp_threshold": 10,
                "game_ids": [item["game_id"] for item in frames],
                "representative_frames": frames,
            })
    return sorted(findings, key=lambda finding: (-finding["distinct_teams"], -finding["distinct_maps"],
                  -finding["losses"], -finding["mean_hp_drop"], -finding["costly_wins"]))[:3]


def _opponent_split(opponent_id):
    # Hash only the official opponent's team ID so every complete series
    # against that team uses the same split across rounds and applications.
    key = hashlib.sha256(f"cycle-pvp-opponent-v1:{opponent_id}".encode()).digest()
    return {0: "test", 1: "dev"}.get(int.from_bytes(key[:8], "big") % 10, "train")


def _complete_series(root, games):
    indexed = defaultdict(dict)
    for game in games:
        if game.get("ladder"):
            indexed[game["ladder"]["series_id"]][game["game_id"]] = game
    complete = {}
    for series_id, members in indexed.items():
        record = root / ".local/cycle/ladder/series" / f"{series_id}.json"
        if not record.is_file():
            continue
        series = read_json(record)
        expected = {game["match_id"] for game in series["games"]}
        if (series["series_id"] == series_id and series["status"] == "finished" and expected == members.keys()
                and all(game["ladder"]["team_id"] == series["team_id"]
                        and game["ladder"]["opponent_id"] == series["opponent_id"] for game in members.values())):
            complete[series_id] = list(members.values())
    return complete


def _closed_gate_games(root, db, before_round):
    from . import cycle_store
    from .cycle_pve import _gate_batch

    if not (cycle_store.folder(root)/"control.sqlite3").exists() or not db.execute(
        "SELECT 1 FROM sqlite_master WHERE name='cycle_gate_maps'"
    ).fetchone():
        return {}
    closed = {(item["round"],item["candidate"],item["incumbent"]) for item in cycle_store.history(root,"pve",-1)
              if before_round is None or item["round"] < before_round}
    result = {}
    for number,parent,candidate,incumbent,tier,seed,grids in db.execute(
        "SELECT round,parent,candidate,incumbent,tier,seed,grids FROM cycle_gate_maps"
    ):
        if (number,candidate,incumbent) not in closed:
            continue
        batch = _gate_batch(db,parent,candidate,incumbent,tier,seed)
        if batch:
            layouts = set(dict(json.loads(grids)).values())
            for (mid,) in db.execute("SELECT id FROM slots WHERE run=?", (batch,)):
                result[mid] = (number, layouts)
    return result


def _training_partitions(root, games, before_round=None):
    """Partition PvE by layout and PvP by opponent, applying live held-out exclusions."""
    path = root / ".local/sampling/samples.sqlite3"
    released = {}
    if path.is_file():
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db, db:
            db.execute("BEGIN")
            db.row_factory = sqlite3.Row
            _, layouts, holdouts, reserved = _sampling_policy(db)
            released = _closed_gate_games(root,db,before_round)
    else:
        layouts, holdouts, reserved = {}, set(), set()
    pve = [game for game in games if game["scope"] == "practice"]
    eligible = {game["game_id"] for game in pve if game["map"]["eligible_for_training"]
                and _layout(game["map"]["grid_sha256"], layouts, holdouts, reserved, game["collections"],
                            game["versions"], game.get("collection_excluded"))["eligible_for_training"]}

    # A random layout split on a PvP-only map is irrelevant. A map actually
    # used by PvE for development/blind work is a cross-line exclusion.
    pve_holdouts = set(holdouts)
    for game in pve:
        grid = game["map"]["grid_sha256"]
        if (layouts.get(grid, game["map"]["split"]) != "train"
                or game.get("collection_excluded")
                or any(item["planned_grid"] == grid and item["planned_split"] != "train"
                       for item in game["collections"])
                or any(binding["id"] in reserved for binding in game["versions"].values())):
            pve_holdouts.add(grid)

    for game in pve:
        past = released.get(game["game_id"])
        if (past and game["map"]["grid_sha256"] in past[1] and not game.get("collection_excluded")
                and not game["collections"] and not any(v["id"] in reserved for v in game["versions"].values())):
            eligible.add(game["game_id"])
            game["map"].update(split="train",split_basis="completed_pve_gate",eligible_for_training=True)

    series = {}
    for series_id, members in _complete_series(root, games).items():
        isolated = any(game["map"]["grid_sha256"] in pve_holdouts or game.get("collection_excluded")
                       or any(binding["id"] in reserved for binding in game["versions"].values()) for game in members)
        opponent = members[0]["ladder"]["opponent_id"]
        split = _opponent_split(opponent)
        series[series_id] = {"opponent_id": opponent, "split": split, "isolated": bool(isolated)}
        if split == "train" and not isolated:
            eligible.update(game["game_id"] for game in members)

    for game in games:
        if ladder := game.get("ladder"):
            game["map"].update(split=_opponent_split(ladder["opponent_id"]), split_basis="opponent_team",
                               eligible_for_training=game["game_id"] in eligible)
    return eligible, series


def _training_focus(games, team_filter, series_filter, target_filter):
    groups = defaultdict(list)
    for game in games:
        if not game["map"]["eligible_for_training"] or game["map"]["split"] != "train":
            continue
        ladder = game.get("ladder")
        series_id = ladder["series_id"] if ladder else None
        if series_filter is not None and series_id not in series_filter:
            continue
        own_side = _perspective(game)
        if own_side is None:
            continue
        team_id = game["players"][own_side]["id"]
        if team_filter is not None and team_id != team_filter:
            continue
        opponent = game["players"][SIDES[1 - SIDES.index(own_side)]]
        target = LEVEL_BY_NAME.get(opponent["name"]) if opponent["bot"] else f"team:{opponent['id']}"
        if target_filter is not None and target != target_filter:
            continue
        for event in EVENTS:
            case = _case(game, event)
            if case is not None:
                item, _, lost_hp, won, lost = case
                entry = {"game_id": game["game_id"], "map_seed": game["map"]["seed"],
                         "grid_sha256": game["map"]["grid_sha256"], "split": game["map"]["split"],
                         "split_basis": game["map"].get("split_basis", "layout"),
                         "team_id": team_id, "opponent_id": opponent["id"], "series_id": series_id, "own_side": own_side,
                         "frame": item["frame"], "anchor_frame": item["anchor_frame"]}
                groups[game["scope"], target, event].append((lost, won and lost_hp >= 10, lost_hp, entry))
    focus = []
    for (scope, target, event), candidates in groups.items():
        entries, maps, series = [], set(), set()
        ranked = sorted(candidates, key=lambda case: (-case[0], -case[1], -case[2], -case[3]["game_id"]))
        for _, _, _, entry in ranked:
            grid, series_id = entry["grid_sha256"], entry["series_id"]
            if grid in maps or (series_id is not None and series_id in series):
                continue
            maps.add(grid)
            series.add(series_id)
            entries.append(entry)
            if len(entries) == 3:
                break
        focus.append({"scope": scope, "target": target, "event": event, "entries": entries,
                      "seen_maps": len({case[3]["grid_sha256"] for case in candidates}),
                      "losses": sum(case[0] for case in candidates),
                      "costly_wins": sum(case[1] for case in candidates)})
    return sorted(focus, key=lambda item: (-item["seen_maps"], -item["losses"], -item["costly_wins"],
                                           item["scope"], item["target"] or "", EVENTS.index(item["event"])))[:8]


def _sampling_needs(games, scope, target_filter):
    groups = defaultdict(list)
    for game in games:
        if game["scope"] == "practice":
            keys = [(game["scope"], target) for target in game["targets"] if target in PRACTICE_NAMES]
        else:
            ladder = game.get("ladder")
            if ladder and game["map"]["split"] != "train":
                continue
            keys = [("pvp", f"team:{ladder['opponent_id']}" if ladder else None)]
        for key in keys:
            groups[key].append(game)
    if scope in (None, "practice"):
        for target in PRACTICE_NAMES:
            groups.setdefault(("practice", target), [])
    pvp_train_target = target_filter is None or (
        target_filter.startswith("team:") and _opponent_split(int(target_filter.split(":")[1])) == "train")
    if scope in (None, "pvp") and pvp_train_target and not any(key[0] == "pvp" for key in groups):
        groups["pvp", target_filter] = []
    needs = []
    for (kind, target), group in groups.items():
        if target_filter is not None and target != target_filter:
            continue
        for name in EVENTS:
            eligible = [game for game in group if game["events"].get(name) and game["map"]["eligible_for_training"]]
            maps = sorted({game["map"]["grid_sha256"] for game in eligible})
            additional = MIN_TRAIN_MAPS - len(maps)
            if additional > 0:
                needs.append({"scope": kind, "target": target, "event": name, "seen_games": len(eligible),
                              "seen_maps": len(maps), "desired_maps": MIN_TRAIN_MAPS,
                              "additional_maps": additional, "preferred_split": "train",
                              "excluded_grid_sha256": maps, "priority": additional})
    return sorted(needs, key=lambda item: (-item["priority"], item["scope"], item["target"] or "", EVENTS.index(item["event"])))


def _pvp_split_summary(games, completed):
    groups = {}
    ladder_games = [game for game in games if game.get("ladder")]
    for split in ("train", "dev", "test"):
        split_games = [game for game in ladder_games if game["map"]["split"] == split]
        opponents = {game["ladder"]["opponent_id"] for game in split_games}
        series = {game["ladder"]["series_id"] for game in split_games}
        complete = [completed[key] for key in series & completed.keys()]
        groups[split] = {
            "opponents": len(opponents), "opponent_ids": sorted(opponents), "replay_games": len(split_games),
            "archived_series": len(series), "complete_series": len(complete),
            "isolated_series": sum(record["isolated"] for record in complete),
            "training_series": sum(split == "train" and not record["isolated"] for record in complete),
        }
    return groups


def report_watermark(root):
    with closing(_database(root)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        return time()


def summary(root, line="all", *, since=None, until=None, training_team_id=None, training_series_ids=None, before_round=None):
    """Return representative official evidence, scoped scores and training coverage gaps."""
    root = Path(root).resolve()
    cutoff = float(since) if since is not None else None
    if cutoff is not None and (not isfinite(cutoff) or cutoff < 0):
        raise ValueError("since 必须为非负的 UTC Unix 秒数")
    with closing(_database(root)) as db:
        records = list(db.execute("SELECT id, report, error, ingested_at FROM games WHERE ingested_at<=? ORDER BY id DESC",
                                  (float("inf") if until is None else until,)))
    games = [json.loads(row["report"]) | {"ingested_at": row["ingested_at"]} for row in records if row["report"] is not None]
    eligible, series = _training_partitions(root, games, before_round)
    failures = [{"game_id": row["id"], "error": row["error"]} for row in records
                if row["error"] is not None and (cutoff is None or row["ingested_at"] > cutoff)]
    labels = _version_labels(root)
    sources, purpose = _project_line(root, line)
    scope = {"pvp": "pvp", "pve": "practice", "practice": "practice", "ladder": "pvp"}.get(line)
    if scope is None:
        scope = {"pvp": "pvp", "pve": "practice"}.get(purpose)
    target_filter = None
    if line in PRACTICE_NAMES or line.startswith("team:"):
        target_filter = sampling_data.target_key(line)
        scope = "practice" if line in PRACTICE_NAMES else "pvp"
    if purpose is not None:
        selected = [game for game in games if scope is None or game["scope"] == scope]
    else:
        selected = [game for game in games if (scope is None or game["scope"] == scope)
                    and _line_filter(line, game, labels)]
    corpus = selected
    if cutoff is not None:
        selected = [game for game in corpus if game["ingested_at"] > cutoff]
    scoped_version = purpose is not None or (not selected and line not in ("all", "*", "pvp", "pve", "practice", "ladder"))
    history = _version_rows(selected, labels, line, sources if scoped_version else None)
    seen_versions = dict.fromkeys(row["version_id"] for row in history)
    changes = [{"version_id": key, "strategy": labels[key]["name"], "note": labels[key]["note"],
                "display_time": labels[key].get("display_time"),
                "tested_against": [{"scope": row["scope"], "target": row["target"], "games": row["games"],
                                    "wins": row["wins"], "mean_score": row["mean_score"]}
                                   for row in history if row["version_id"] == key]}
               for key in seen_versions if labels.get(key, {}).get("note")]
    scoped_games = {scope: [game for game in selected if game["scope"] == scope] for scope in ("pvp", "practice")}
    event_counts = {name: sum(game["event_counts"].get(name, 0) for game in selected) for name in EVENTS}
    return {
        "line": line, "games": len(selected), "maps": len({game["map"]["grid_sha256"] for game in selected}),
        "cursor": max((row["ingested_at"] for row in records), default=0), "since": cutoff,
        "total_games": len(corpus),
        "team_matchups": len({_team_matchup(game) for game in selected}),
        "sources": dict(Counter(source["source"] for game in selected for source in game["sources"])),
        "event_counts": event_counts,
        "representative_events": {name: _representative(selected, name) for name in EVENTS},
        "top_findings": _top_findings(selected),
        "version_stats": history,
        "previously_tried_changes": changes[:20],
        "sampling_needs": _sampling_needs(corpus, scope, target_filter),
        "training_focus": _training_focus([game for game in corpus if game["game_id"] in eligible],
                                           training_team_id, None if training_series_ids is None else set(training_series_ids),
                                           target_filter),
        "completed_gate_games": sum(g["map"].get("split_basis")=="completed_pve_gate" for g in corpus),
        "completed_gate_focus": _training_focus([g for g in corpus if g["map"].get("split_basis")=="completed_pve_gate"],
                                                training_team_id,None,target_filter) if line=="pve" else [],
        "pvp_splits": _pvp_split_summary(corpus, series),
        "coverage_by_scope": {name: {"games": len(group), "maps": len({game["map"]["grid_sha256"] for game in group}),
                                     "games_with_event": {event: sum(bool(game["events"].get(event)) for game in group)
                                                          for event in EVENTS}}
                              for name, group in scoped_games.items()},
        "failed_games": len(failures), "recent_errors": failures[:10],
    }
