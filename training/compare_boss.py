"""Compare online boss scores with a frozen sparring model on matched maps and sides."""
import argparse
from itertools import combinations
from pathlib import Path
from statistics import fmean

from seedcup.game.maps import generate_map
from seedcup.game.replay import load
from tank import versions
from tank.simulation import run_match
from tank.workspace import project_root, read_json, write_json


def pair_results(by_version, games, seeds):
    def direction(value):
        return (value > 0) - (value < 0)

    pairs = []
    for a, b in combinations(by_version, 2):
        remote = by_version[a]["official"] - by_version[b]["official"]
        here = by_version[a]["local"] - by_version[b]["local"]
        per_map = []
        for seed in seeds:
            differences = {}
            for field in ("official", "local"):
                means = [fmean(g[field] for g in games if g["seed"] == seed and g["candidate"] == v)
                         for v in (a, b)]
                differences[field] = means[0] - means[1]
            per_map.append({"seed": seed, **differences})
        pairs.append({"a": a, "b": b, "official_delta": remote, "local_delta": here,
                      "order": "tied" if remote == 0 or here == 0 else "same" if remote*here > 0 else "reversed",
                      "official_order_stable_across_maps": len({direction(p["official"]) for p in per_map}) == 1,
                      "by_map": per_map})
    return pairs


def official_cases(root, candidates, seeds):
    cases = {}
    for candidate in candidates:
        folder = root / ".local/online/practice" / candidate / "boss"
        for record_path in sorted(folder.glob("*.json")):
            record = read_json(record_path)
            seed = record.get("map_seed", record["seed"])
            if seed not in seeds:
                continue
            replay = load(root / ".local/online/replays" / f"{record['id']}.json.gz")
            game_map = replay["map"]
            side = ("red", "blue").index(record["side"])
            if game_map["seed"] != seed:
                raise ValueError(f"回放地图与练习记录不符 {record['id']}")
            regenerated = generate_map(game_map["seed"]).to_dict()
            if any(regenerated[k] != game_map[k] for k in ("grid", "spawns")):
                raise ValueError(f"SDK 生成的地图与官网不同 {record['id']}")
            key = candidate, game_map["seed"], side
            cases.setdefault(key, []).append((record, replay["match"]["game"]))
    missing = [(v, seed, side) for v in candidates for seed in seeds
               for side in range(2) if (v, seed, side) not in cases]
    if missing:
        raise ValueError(f"官网换边对照缺 {len(missing)} 项，首项 {missing[0]}")
    return cases


def compare(root, opponent, candidates, seeds, fit_ids):
    if len(set(candidates)) < 2 or len(set(candidates)) != len(candidates):
        raise ValueError("至少选择两个不同的策略版本")
    if len(seeds) != len(set(seeds)):
        raise ValueError("地图种子重复")
    for key in candidates:
        versions.verify(root, key)
    if opponent != "baseline":
        versions.verify(root, opponent)
        training = read_json(versions.snapshot_data(root, opponent) / "training.json")
        recorded_ids = [item["id"] for item in training["train"] + training["holdout"]]
        if fit_ids and set(fit_ids) != set(recorded_ids):
            raise ValueError("指定的训练及选模对局与陪练快照不符")
        fit_ids = recorded_ids
    fit_maps = [load(root / ".local/online/replays" / f"{i}.json.gz")["map"] for i in fit_ids]
    if ({m["seed"] for m in fit_maps}.intersection(seeds) or
            {m["grid"] for m in fit_maps}.intersection(generate_map(s).to_dict()["grid"] for s in seeds)):
        raise ValueError("排名地图与训练或选模地图重复")
    cases = official_cases(root, candidates, seeds)
    games = []
    for (candidate, seed, side), played in cases.items():
        numbers = {n for _, n in played}
        if len(numbers) != 1:
            raise ValueError(f"同一出生侧存在不同局号 {seed} {side}")
        bots = [versions.load_bot(root, candidate), versions.load_bot(root, opponent)]
        if side:
            bots.reverse()
        try:
            local = run_match(*bots, seed, numbers.pop())["result"]
        finally:
            for bot in bots:
                if hasattr(bot, "close"):
                    bot.close()
        games.append({"candidate": candidate, "seed": seed, "side": side,
                      "ids": [r["id"] for r, _ in played],
                      "official": fmean(r["score"] for r, _ in played),
                      "local": local["scores"][side]["total"],
                      "official_seconds": fmean(r["result"]["seconds"] for r, _ in played),
                      "local_seconds": local["seconds"],
                      "official_zone_damage": fmean(r["result"]["stats"][side]["zone_damage"] for r, _ in played),
                      "local_zone_damage": local["stats"][side]["zone_damage"]})
    def mean(candidate, field):
        return fmean(g[field] for g in games if g["candidate"] == candidate)
    by_version = {v: {"note": versions.describe(root, v)["note"],
                      "official": mean(v, "official"), "local": mean(v, "local")} for v in candidates}
    return {"opponent": opponent, "fit_ids": fit_ids, "test_seeds": seeds,
            "by_version": by_version, "pairs": pair_results(by_version, games, seeds), "games": games}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--fit-ids", nargs="+", type=int, default=[],
                        help="额外核对训练及选模回放编号；默认读取陪练快照")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(project_root(), args.opponent, args.candidates, args.seeds, args.fit_ids)
    write_json(args.output, report)
    print(args.output)
