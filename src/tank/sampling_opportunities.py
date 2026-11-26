"""Label the first SDK replay opportunity for each close-range reaction."""
import gzip
import hashlib
import json
from collections import Counter, defaultdict

from seedcup.game import SIDES

from .sampling_probes import (
    _BEHAVIOR_FAR,
    _BEHAVIOR_NEAR,
    _HIT_RESPONSE_SECONDS,
    _WITHDRAWAL_GAIN,
    _distance,
    _replay_inputs,
)

CACHE_KEY = "opportunities_v1"
EVENTS = ("first_close", "first_close_hit")
OUTCOMES = ("withdrawn", "stayed", "ended")
TRAIN_LAYOUT = """json_extract(e.entry,'$.split')='train'
    AND json_extract(e.entry,'$.grid_sha256') NOT IN (SELECT grid FROM sampling_holdouts)
    AND (json_extract(e.entry,'$.probe') IS NULL OR json_extract(e.entry,'$.probe')<>'holdout-orbit')
    AND NOT EXISTS (SELECT 1 FROM sampling_reserved_versions r WHERE r.version=json_extract(e.entry,'$.version_id'))
    AND json_extract(e.entry,'$.unassigned_collection') IS NULL
    AND (json_type(e.entry,'$.collection') IS NULL
         OR (json_extract(e.entry,'$.collection.planned_split')='train'
             AND json_extract(e.entry,'$.collection.layout_match')=1))"""
COMPARISON_ARMS = {"intervention": ("window-fire", "window-hold"), "repeat": ("reference", "repeat")}
INTERVENTION_TOLERANCES = {"distance_cells": .25, "probe_heading_deg": 10, "target_heading_deg": 10}


def _observe(replay, frames, target_index, trigger, tick_hz):
    probe_index = 3 - target_index
    previous = frames[trigger]
    target, probe = previous[target_index], previous[probe_index]
    distance = _distance(target, probe)
    deadline = trigger + _HIT_RESPONSE_SECONDS * tick_hz
    record = {
        "trigger_frame": trigger,
        "history_start_frame": max(0, trigger - tick_hz // 2),
        "deadline_frame": deadline,
        "trigger_distance_cells": distance,
        "trigger_target_hp": target[3],
        "trigger_probe_hp": probe[3],
    }
    outcome = "ended"
    end_reason = "missing_frames"
    end = trigger
    target_gain = 0.0
    result = replay["result"]
    finish_tick = result["ticks"] if result is not None else None
    finish_reason = result.get("reason", "match_finished") if result is not None else None
    for tick in range(trigger + 1, min(deadline + 1, len(frames))):
        current = frames[tick]
        if current[0] != tick:
            break
        end = tick
        target, probe = current[target_index], current[probe_index]
        if not (target[4] and probe[4]):
            end_reason = finish_reason if tick == finish_tick else "death"
            break
        last_target, last_probe = previous[target_index], previous[probe_index]
        prior_distance, distance = distance, _distance(target, probe)
        # Average the target's displacement with the probe fixed at each end of the tick.
        target_gain += (_distance(target, last_probe) - prior_distance
                        + distance - _distance(last_target, probe)) / 2
        if distance >= _BEHAVIOR_FAR and round(target_gain, 10) >= _WITHDRAWAL_GAIN:
            outcome = "withdrawn"
            break
        if tick == deadline:
            outcome = "stayed"
            break
        previous = current

    if outcome == "ended":
        record["end_reason"] = finish_reason if end == finish_tick else end_reason
    return record | {"end_frame": end, "outcome": outcome, "target_gain_cells": round(target_gain, 10)}


def extract(replay, target_side):
    """Record the first joint-alive trigger and up to three observable seconds after it."""
    target_index, tick_hz, frames = _replay_inputs(replay, target_side)
    probe_index = 3 - target_index
    records = dict.fromkeys(EVENTS)
    for tick, frame in enumerate(frames):
        if frame[0] != tick:
            break  # A later trigger can no longer be established as the first.
        target, probe = frame[target_index], frame[probe_index]
        if not (target[4] and probe[4]):
            break
        if _distance(target, probe) > _BEHAVIOR_NEAR:
            continue
        if records["first_close"] is None:
            records["first_close"] = _observe(replay, frames, target_index, tick, tick_hz)
        # SDK movement precedes shell impacts. The opponent's hit counter excludes zone damage.
        if tick > 0 and probe[9] > frames[tick - 1][probe_index][9]:
            records["first_close_hit"] = _observe(replay, frames, target_index, tick, tick_hz)
            break
    return records


def summarize(observations):
    """Count both games and independent layouts at each step of the denominator."""
    games = {event: dict.fromkeys(("qualified", *OUTCOMES), 0) for event in EVENTS}
    maps = {event: {kind: set() for kind in games[event]} for event in EVENTS}
    for grid, _, records in observations:
        for event, record in records.items():
            if record is not None:
                for kind in ("qualified", record["outcome"]):
                    games[event][kind] += 1
                    maps[event][kind].add(grid)
    return {event: {kind: {"games": count, "maps": len(maps[event][kind])}
                    for kind, count in counts.items()} for event, counts in games.items()}


def _anchor_differences(first, second):
    differences = {"distance_cells": round(abs(first["distance"] - second["distance"]), 4)}
    for side in ("probe", "target"):
        key = f"{side}_heading_error_deg"
        differences[f"{side}_heading_deg"] = round(abs((first[key] - second[key] + 180) % 360 - 180), 2)
    return differences


def _execution(games, anchor, *, expected_fire):
    if len(games) != 1:
        return {"confirmed": False, "reasons": ["case_missing" if not games else "case_ambiguous"]}
    timing = games[0].get("sdk_timing", {}).get("intervention")
    if timing is None:
        return {"confirmed": False, "reasons": ["sdk_evidence_missing"]}
    reasons = []
    if anchor is None:
        reasons.append("replay_anchor_missing")
    if timing["source_frame"] is None:
        reasons.append("sdk_observation_missing")
    elif anchor is not None and timing["source_frame"] != anchor["frame"]:
        reasons.append("source_frame_mismatch")
    if timing["command_fire"] is not expected_fire:
        reasons.append("command_mismatch")
    if timing["confirmed_uses"] <= 0:
        reasons.append("action_unconfirmed")
    elif timing["applied_fire"] is not expected_fire:
        reasons.append("applied_fire_mismatch")
    shots = games[0]["probe_shots"]
    if shots != int(expected_fire):
        reasons.append("replay_shots_mismatch")
    return {"confirmed": not reasons, "reasons": reasons, "replay_shots": shots}


def comparisons(entries):
    """Match archived collection cases within a map, probe side and official game number."""
    groups = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        collection = entry.get("collection")
        if collection is None:
            continue
        kind, role = collection.get("kind"), collection.get("role")
        if kind == "intervention" and role == "intervention":
            arm = entry["probe"]
        elif kind == "repeat" and role in COMPARISON_ARMS["repeat"]:
            arm = role
        else:
            continue
        probe_side = SIDES[1 - SIDES.index(entry["target_side"])]
        groups[kind, collection["run"], collection["map_index"], probe_side, entry["game_no"]][arm].append(entry)

    by_match = {}
    totals = {"intervention": Counter({"confirmed": 0, "coverage_only": 0}),
              "repeat": Counter({"matched": 0, "coverage_only": 0})}
    reasons_by_kind = {kind: Counter() for kind in COMPARISON_ARMS}
    geometry_matched, executed = 0, 0
    execution_reasons = Counter()
    for (kind, run, map_index, side, game_no), group in groups.items():
        arms = COMPARISON_ARMS[kind]
        reasons = [f"missing_{arm}" for arm in arms if arm not in group]
        reasons.extend(f"duplicate_{arm}" for arm in arms if len(group.get(arm, ())) > 1)
        matches = [entry for games in group.values() for entry in games]
        pair = {"kind": kind, "run": run, "map_index": map_index, "probe_side": side,
                "game_no": game_no, "match_ids": {arm: [e["id"] for e in group.get(arm, ())] for arm in arms}}
        grids = {entry["grid_sha256"] for entry in matches}
        if len(grids) != 1 or any(not entry["collection"]["layout_match"] for entry in matches):
            reasons.append("layout_mismatch")
        if kind == "intervention":
            anchors = {arm: group[arm][0]["collection"]["intervention_anchor"]
                       for arm in arms if len(group.get(arm, ())) == 1}
            reasons.extend(f"missing_anchor_{arm}" for arm, anchor in anchors.items() if anchor is None)
            if len(anchors) == len(arms) and all(anchors.values()):
                first, second = (anchors[arm] for arm in arms)
                differences = _anchor_differences(first, second)
                reasons.extend(f"{key}_mismatch" for key, limit in INTERVENTION_TOLERANCES.items()
                               if differences[key] > limit)
                if set(first["target_retreat_lanes"]) != set(second["target_retreat_lanes"]):
                    reasons.append("retreat_lanes_mismatch")
                pair["anchor_differences"] = differences
            execution = {arm: _execution(group.get(arm, ()), anchors.get(arm), expected_fire=arm == "window-fire")
                         for arm in arms}
            confirmed = all(result["confirmed"] for result in execution.values())
            pair.update(execution=execution, execution_confirmed=confirmed)
            pair["geometric_match"] = not reasons
            geometry_matched += pair["geometric_match"]
            executed += confirmed
            execution_reasons.update(sorted({reason for result in execution.values() for reason in result["reasons"]}))
            if not confirmed:
                reasons.append("execution_unconfirmed")
        elif all(len(group.get(arm, ())) == 1 for arm in arms):
            first, second = (group[arm][0]["version_id"] for arm in arms)
            if first is None or first != second:
                reasons.append("version_mismatch")
        passed = "confirmed" if kind == "intervention" else "matched"
        state = "coverage_only" if reasons else passed
        pair.update(state=state, reasons=reasons)
        totals[kind][state] += 1
        reasons_by_kind[kind].update(reasons)
        for entry in matches:
            by_match[entry["id"]] = pair
    summary = {kind: dict(counts) | {"groups": counts.total(), "reasons": dict(reasons_by_kind[kind])}
               for kind, counts in totals.items()}
    intervention = summary["intervention"]
    intervention.update(
        tolerances=INTERVENTION_TOLERANCES | {"require_same_retreat_lanes": True},
        geometry_matched=geometry_matched, execution_confirmed=executed, confirmed_pairs=intervention.pop("confirmed"),
        execution_reasons=dict(execution_reasons),
    )
    return by_match, summary


def for_entry(root, entry, cached, updates):
    """Use the archive index to backfill labels, or return None when raw data is unavailable."""
    if cached is not None:
        return cached
    try:
        raw = (root / entry["path"]).read_bytes()
    except FileNotFoundError:
        return None
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"本机回放文件与采样索引不符 {entry['id']}")
    records = extract(json.loads(gzip.decompress(raw)), entry["target_side"])
    updates.append((json.dumps(records), entry["id"], entry["target"], digest))
    return records


def save_cache(db, updates):
    if updates:
        db.executemany(f"""UPDATE examples SET coverage=json_set(coverage,'$.{CACHE_KEY}',json(?))
            WHERE id=? AND target=? AND json_extract(entry,'$.sha256')=?""", updates)


def history(root):
    """Read only non-reserved boss training layouts for sampling decisions."""
    from .sampling_data import database

    with database(root) as db:
        rows = db.execute(f"""SELECT e.entry,json_extract(e.coverage,'$.{CACHE_KEY}') AS cached FROM examples e
            WHERE e.target='boss' AND ({TRAIN_LAYOUT}) ORDER BY e.id""").fetchall()
        observations, updates = [], []
        for row in rows:
            entry = json.loads(row["entry"])
            cached = json.loads(row["cached"]) if row["cached"] is not None else None
            records = for_entry(root, entry, cached, updates)
            if records is not None:
                observations.append((entry["grid_sha256"], entry["version_id"], records))
        save_cache(db, updates)
    return observations
