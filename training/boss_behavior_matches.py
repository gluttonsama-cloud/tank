"""同一策略、地图、出生侧和局号下的官网与 SDK 整局对比。"""

from math import atan2, degrees, hypot
from statistics import fmean, median

from seedcup.game import GameMap, rules
from seedcup.game.engine import ang_diff, line_of_sight
from seedcup.game.replay import load, save
from tank import versions
from tank.simulation import run_match
from tank.workspace import read_json

if __package__:
    from .compare_boss import official_cases
else:
    from compare_boss import official_cases


STATIONARY_DISTANCE = 0.025  # 大于连续两帧坐标舍入产生的最大欧氏误差。


def rate(count, total):
    return count / total if total else None


def summarize_match(replay, challenger_side, *, until=None):
    game_map = GameMap.from_dict(replay["map"])
    frames = replay["frames"] if until is None else replay["frames"][:until + 1]
    distances, visible, bearings = [], [], [[], []]
    for _, red, blue, _, _ in frames:
        dx, dy = blue[0] - red[0], blue[1] - red[1]
        distances.append(hypot(dx, dy))
        visible.append(line_of_sight(game_map, *red[:2], *blue[:2]))
        bearing = degrees(atan2(dy, dx))
        bearings[0].append(abs(ang_diff(bearing, red[2])))
        bearings[1].append(abs(ang_diff(bearing + 180, blue[2])))
    first_visible = next((i for i, value in enumerate(visible) if value), None)
    first_close = next((i for i, value in enumerate(distances) if value <= 4), None)
    near_min = min(distances)

    def summarize_side(side):
        states = [frame[side + 1] for frame in frames]
        other = [frame[2 - side] for frame in frames]
        steps = [hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(states, states[1:])]
        moving = [d for d in steps if d > STATIONARY_DISTANCE]
        hit_ticks = [i for i in range(1, len(frames)) for _ in range(other[i][9] - other[i - 1][9])]
        intervals = [b - a for a, b in zip(hit_ticks, hit_ticks[1:])]
        longest, chain = int(bool(hit_ticks)), int(bool(hit_ticks))
        for interval in intervals:
            chain = chain + 1 if interval <= 2 * rules.TICK_HZ else 1
            longest = max(longest, chain)
        visible_stationary = [d <= STATIONARY_DISTANCE for i, d in enumerate(steps) if visible[i]]
        shots, hits = states[-1][10], states[-1][9]
        return {
            "hp": states[-1][3],
            "shots": shots,
            "hits": hits,
            "hit_rate": rate(hits, shots),
            "damage_dealt": states[-1][8],
            "first_hit_received_s": hit_ticks[0] * rules.DT if hit_ticks else None,
            "distance_at_first_hit_received": distances[hit_ticks[0]] if hit_ticks else None,
            "hit_received_s": [i * rules.DT for i in hit_ticks],
            "max_hits_chain_gap_le_2s": longest,
            "stationary_ratio": rate(len(steps) - len(moving), len(steps)),
            "stationary_ratio_when_visible": fmean(visible_stationary) if visible_stationary else None,
            "speed_cells_s": fmean(steps) / rules.DT if steps else None,
            "moving_speed_cells_s": fmean(moving) / rules.DT if moving else None,
            "mean_bearing_error_when_visible_deg": (
                fmean(error for error, flag in zip(bearings[side], visible, strict=True) if flag)
                if first_visible is not None else None
            ),
            "hp_checkpoints": {str(i): states[i * rules.TICK_HZ][3]
                               for i in (5, 10, 15, 20, 30, 40, 60, 75, 90)
                               if i * rules.TICK_HZ < len(states)},
            "violations": states[-1][11],
        }

    return {
        "seconds": (len(frames) - 1) * rules.DT,
        "first_line_of_sight_s": first_visible * rules.DT if first_visible is not None else None,
        "first_within_4_cells_s": first_close * rules.DT if first_close is not None else None,
        "min_distance_cells": near_min,
        "median_distance_cells": median(distances),
        "visible_ratio": fmean(visible),
        "challenger": summarize_side(challenger_side),
        "boss": summarize_side(1 - challenger_side),
    }


def verify_unseen(root, model, replays):
    if model == "baseline":
        return
    training = read_json(versions.snapshot_data(root, model) / "training.json")
    sources = training["train"] + training["holdout"]
    ids = {source["id"] for source in sources}
    seeds = {source["seed"] for source in sources}
    grids = {load(root / ".local/online/replays" / f"{i}.json.gz")["map"]["grid"] for i in ids}
    for replay in replays:
        replay_id = replay["match"]["id"]
        game_map = replay["map"]
        if replay_id in ids or game_map["seed"] in seeds or game_map["grid"] in grids:
            raise ValueError(f"回放 {replay_id} 与 {model} 的训练或选模数据重复")


def common_alive_distances(official, local_replays):
    distances = [[] for _ in range(len(local_replays) + 1)]
    for frames in zip(official["frames"], *(replay["frames"] for replay in local_replays)):
        if any(frame[0] != frames[0][0] for frame in frames[1:]):
            raise ValueError(f"官网回放与本地复跑的帧号不一致 {official['match']['id']}")
        if not all(red[4] and blue[4] for _, red, blue, _, _ in frames):
            break
        for values, (_, red, blue, _, _) in zip(distances, frames, strict=True):
            values.append(hypot(blue[0] - red[0], blue[1] - red[1]))
    if not distances[0]:
        raise ValueError(f"官网回放与本地复跑没有双方存活的共同帧 {official['match']['id']}")
    reference = {"frames": len(distances[0]), "official_median_distance_cells": median(distances[0])}
    return [reference | {"local_median_distance_cells": median(values)} for values in distances[1:]]


def matched_games(root, models, candidates, seeds, save_dir=None, *, replay_ids=None,
                  require_unseen=False, include_common_alive=False):
    for keys in (models, candidates, seeds):
        if len(set(keys)) != len(keys):
            raise ValueError("快照或地图种子重复")
    for version_id in set(models + candidates) - {"baseline"}:
        versions.verify(root, version_id)
    cases = official_cases(root, candidates, seeds, replay_ids=replay_ids)
    replays = {r["id"]: load(root / ".local/online/replays" / f"{r['id']}.json.gz")
               for official in cases.values() for r, _ in official}
    if require_unseen:
        for model in models:
            verify_unseen(root, model, replays.values())
    games = []
    for (candidate, seed, side), official in cases.items():
        for record, game_no in official:
            replay_id = record["id"]
            replay = replays[replay_id]
            remote = summarize_match(replay, side) | {
                "reason": replay["result"]["reason"], "score": replay["result"]["scores"][side]["total"],
            }
            local_replays, rows = [], []
            for model in models:
                bots = [versions.load_bot(root, candidate), versions.load_bot(root, model)]
                if side:
                    bots.reverse()
                try:
                    match = run_match(*bots, seed, game_no, replay=True)
                finally:
                    for bot in bots:
                        if hasattr(bot, "close"):
                            bot.close()
                local_replay = match["replay"]
                if save_dir is not None:
                    save(save_dir / model / f"{replay_id}.json.gz", local_replay)
                until = min(match["result"]["ticks"], replay["result"]["ticks"])
                rows.append({
                    "replay_id": replay_id, "candidate": candidate, "model": model, "seed": seed,
                    "challenger_side": "blue" if side else "red", "game": game_no,
                    "official": remote,
                    "local": summarize_match(local_replay, side) | {
                        "reason": match["result"]["reason"], "score": match["scores"][side],
                    },
                    "same_duration": {
                        "official": summarize_match(replay, side, until=until),
                        "local": summarize_match(local_replay, side, until=until),
                    },
                })
                if include_common_alive:
                    local_replays.append(local_replay)
            if include_common_alive:
                for row, distance in zip(rows, common_alive_distances(replay, local_replays), strict=True):
                    row["common_alive"] = distance
            games.extend(rows)
    return {"rules": rules.as_dict()["version"], "models": models, "candidates": candidates,
            "seeds": seeds, "ids": sorted(replays), "require_unseen": require_unseen, "games": games}
