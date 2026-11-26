"""Compare frozen A, B and the style router in paired official practice games."""

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from itertools import pairwise
from math import hypot
from pathlib import Path
from runpy import run_path
from statistics import mean

from seedcup.game import SIDES, rules
from seedcup.game.bots import wall_grid

from tank import online, sampling_data, versions
from tank.workspace import project_root, read_json, write_json


def inspect_capture(root, entry, level, label, plan, detector_type, config, run):
    raw = (root / entry["observation_file"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["observation_sha256"]:
        raise ValueError(f"SDK 观测哈希不符 {entry['id']}")
    capture = json.loads(gzip.decompress(raw))
    start, result = capture["start"], capture["end"]["result"]
    if (start["match_id"] != entry["id"] or capture["end"]["match_id"] != entry["id"]
            or capture["practice"]["version_id"] != (plan["router_version"] if label == "C" else plan["policies"][label])
            or start["opponent"] != online.PRACTICE_NAMES[level] or start["map"]["seed"] != plan["seed"]
            or start["side"] != entry["side"]):
        raise ValueError(f"对局条件不符 {entry['id']}")
    side = SIDES.index(start["side"])
    observations = [obs for obs, _ in capture["frames"]]
    detector = detector_type(wall_grid(start), config)
    for obs in observations:
        detector.observe(obs)
    movement, turns, moving = [], [], []
    for first, second in pairwise(observations):
        if second["frame"] > 5 * rules.TICK_HZ:
            break
        if second["frame"] != first["frame"] + 1:
            continue
        before, after = first["enemy"], second["enemy"]
        step = hypot(after["x"] - before["x"], after["y"] - before["y"])
        movement.append(step)
        moving.append(step > .025)
        turns.append(abs((after["heading"] - before["heading"] + 180) % 360 - 180))
    acknowledgements = [obs["last_action"] for obs in observations
                        if obs["last_action"] and obs["last_action"]["source_frame"] is not None]
    return {"id": entry["id"], "policy": label, "version": entry["version_id"], "run": run,
            "observation_sha256": entry["observation_sha256"], "level": level, "side": start["side"], "game": start["game"],
            "grid": sampling_data.sdk_layout_hash(start), "score": result["scores"][side]["total"],
            "shots": result["stats"][side]["shots"], "hits": result["stats"][side]["hits"],
            "seconds": result["seconds"], "received_frames": len(observations), "gate_frame": detector.switch_frame,
            "acknowledged_actions": len(acknowledgements),
            "reused_actions": sum(a["source_frame"] < a["frame"] for a in acknowledgements),
            "opening_observed_transitions": len(turns), "opening_enemy_moving_fraction": mean(moving) if moving else None,
            "opening_enemy_speed_cells_s": mean(movement) * rules.TICK_HZ if movement else None,
            "opening_enemy_abs_turn_deg_s": mean(turns) * rules.TICK_HZ if turns else None}


def report(root, plan):
    rows = []
    versions.verify(root, plan["router_version"])
    snapshot = versions.snapshot_data(root, plan["router_version"])
    detector_type = run_path(str(snapshot / "style_router.py"))["StyleDetector"]
    config = read_json(snapshot / "detector.json")
    by_version = {v: k for k, v in (plan["policies"] | {"C": plan["router_version"]}).items()}
    with sampling_data.database(root) as db:
        for level, run in (*plan["runs"].items(), *plan.get("extra_runs", {}).items()):
            for row in db.execute("""SELECT o.entry FROM sdk_observations o JOIN slots s ON s.id=o.id
                                     WHERE s.run=? AND o.target=? ORDER BY s.step,s.side""", (run, level)):
                entry = json.loads(row[0])
                rows.append(inspect_capture(root, entry, level, by_version[entry["version_id"]], plan, detector_type, config, run))
        for mid in plan["prior_boss_B"]:
            payload, run = db.execute("""SELECT o.entry, s.run FROM sdk_observations o JOIN slots s ON s.id=o.id
                                         WHERE o.id=? AND o.target='boss'""", (mid,)).fetchone()
            rows.append(inspect_capture(root, json.loads(payload), "boss", "B", plan, detector_type, config, run))
    if len({row["grid"] for row in rows}) > 1:
        raise ValueError("同种子回放出现不同地图布局")
    pairs = defaultdict(dict)
    for row in rows:
        if (row["side"], row["game"]) not in {("red", 0), ("blue", 1)}:
            raise ValueError(f"出生侧与场次不符 {row['id']}")
        group = pairs[row["level"], row["side"], row["game"]]
        if row["policy"] in group:
            raise ValueError(f"重复采样位置 {row['id']}")
        group[row["policy"]] = row["score"]
    summary = {}
    for level in ("rookie", "mid", "strong", "boss"):
        complete = [v for (tier, _, _), v in pairs.items() if tier == level and {"A", "B"} <= v.keys()]
        all_three = [v for v in complete if "C" in v]
        summary[level] = {"paired_sides": len(complete)}
        if len(complete) == 2:
            summary[level].update({name: round(mean(v[name] for v in complete), 3) for name in ("A", "B")})
            summary[level]["B_minus_A"] = round(mean(v["B"] - v["A"] for v in complete), 3)
        summary[level]["three_way_pairs"] = len(all_three)
        if len(all_three) == 2:
            summary[level]["C"] = round(mean(v["C"] for v in all_three), 3)
            for name in ("A", "B"):
                summary[level][f"C_minus_{name}"] = round(mean(v["C"] - v[name] for v in all_three), 3)
    return {"seed": plan["seed"], "matches": len(rows), "complete": all(v["three_way_pairs"] == 2
            for v in summary.values()), "per_tier": summary, "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = report(project_root(), read_json(args.plan))
    write_json(args.out, result)
    print(json.dumps({k: result[k] for k in ("matches", "complete", "per_tier")}))
