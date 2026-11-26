"""Calibrate the A/B router on archived boss play and SDK rookie games."""

import argparse
import shutil
import tempfile
from collections import Counter
from hashlib import sha256
from itertools import product
from pathlib import Path
from statistics import median

from seedcup.game import SIDES, Game, GameMap, generate_map, parse_action, rules
from seedcup.game.bots import BaselineBot, wall_grid
from seedcup.game.replay import FORMAT_VERSION, load
from seedcup.game.runner import match_info

from tank import versions
from tank.online import PRACTICE_NAMES
from tank.sampling_data import layout_hash
from tank.workspace import project_root, read_json, write_json
from training.fit_boss_sparring import replay_observation
from training.style_router import StyleDetector, policy_loader, policy_match_info

EXPERIMENT = Path(__file__).resolve().parent / "experiments/style-switch-v1"
LATEST_FRAME = 100
ARCHIVE_CUTOFF = 23053
CONFIG_GRID = tuple(
    {"latest_frame": LATEST_FRAME, "min_drive": drive, "consecutive_frames": frames}
    for drive, frames in product((0.10, 0.14, 0.18), (3, 5, 7))
)


def trace_replay(replay, challenger_side, config):
    """Play only public observation fields through the exact deployed detector."""
    game_map = GameMap.from_dict(replay["map"])
    if isinstance(challenger_side, str):
        challenger_side = SIDES.index(challenger_side)
    walls = wall_grid(match_info(game_map, challenger_side, ""))
    count = min(config["latest_frame"] + 1, replay["result"]["ticks"])
    observations = (replay_observation(frame, challenger_side)
                    for frame in replay["frames"][:count])
    return _trace(walls, observations, config)


def _trace(walls, observations, config):
    detector = StyleDetector(walls, config)
    for obs in observations:
        if detector.observe(obs) is not None:
            break
    return {"switch_frame": detector.switch_frame, "trace": detector.trace}


def _case(replay, challenger_side, source, version_id=None):
    game_map = GameMap.from_dict(replay["map"])
    frames = replay["frames"][:min(LATEST_FRAME + 1, replay["result"]["ticks"])]
    if any(frame[0] != i for i, frame in enumerate(frames)):
        raise ValueError(f"回放帧不连续 {replay['match']['id']}")
    return {
        "id": replay["match"]["id"], "seed": game_map.seed, "grid": layout_hash(replay["map"]),
        "side": challenger_side, "version": version_id, "source": source,
        "walls": wall_grid(match_info(game_map, challenger_side, "")),
        "observations": tuple(replay_observation(frame, challenger_side) for frame in frames),
    }


def _case_from_manifest(root, row, source, policy):
    side, seed = row["side"], row["seed"]
    if source == "sdk_rookie":
        game_map = generate_map(seed)
        game = Game(game_map)
        players = [policy(), BaselineBot(seed=row["opponent_seed"])]
        if side == 1:
            players.reverse()
        for index, bot in enumerate(players):
            info = match_info(game_map, index, "本地策略", game=side)
            bot.on_match_start(policy_match_info(info) if index == side else info)
        observations = []
        while not game.over:
            observations.append(game.observe(side))
            if game.tick == LATEST_FRAME:
                break
            actions = [parse_action(bot.decide(game.observe(index)))[0]
                       for index, bot in enumerate(players)]
            game.step(actions)
        case = row | {"source": source, "observations": observations,
                      "walls": wall_grid(match_info(game_map, side, ""))}
        grid = layout_hash(game_map.to_dict())
    else:
        path = root / ".local/online/replays" / f"{row['id']}.json.gz"
        if sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"回放哈希不符 {path.name}")
        replay = load(path)
        opponent = replay["match"][SIDES[1 - side]]
        if (replay["format"] != FORMAT_VERSION or replay["rules"] != rules.as_dict()
                or opponent["name"] != PRACTICE_NAMES[source.removeprefix("official_")]
                or not opponent["bot"] or replay["match"][SIDES[side]]["bot"]):
            raise ValueError(f"官网回放条件不符 {path.name}")
        case = _case(replay, side, source, row["version"])
        grid = case["grid"]
    if grid != row["grid"] or case["id"] != row["id"] or case["seed"] != seed:
        raise ValueError(f"地图或场次不符 {row['id']}")
    return case


def _evaluate(cases, config):
    rows = []
    for case in cases:
        traced = _trace(case["walls"], case["observations"], config)
        rows.append({k: case[k] for k in ("id", "seed", "grid", "side", "version", "source")}
                    | {"switch_frame": traced["switch_frame"]})
    frames = [row["switch_frame"] for row in rows if row["switch_frame"] is not None]
    return {"games": len(rows), "layouts": len({row["grid"] for row in rows}),
            "switches": len(frames), "median_switch_frame": median(frames) if frames else None,
            "rows": rows}


def _fit(boss, rookie):
    def key(result):
        config, positive, negative = result
        return (negative["switches"], -positive["switches"],
                positive["median_switch_frame"] or LATEST_FRAME + 1,
                -config["consecutive_frames"], -config["min_drive"])

    candidates = [(config, _evaluate(boss, config), _evaluate(rookie, config)) for config in CONFIG_GRID]
    selected = min(candidates, key=key)
    if selected[2]["switches"]:
        raise ValueError("校准范围内没有通过新手安全门槛的候选阈值")
    evidence = [{"config": config, "boss_switches": positive["switches"],
                 "sdk_rookie_switches": negative["switches"],
                 "boss_median_switch_frame": positive["median_switch_frame"]}
                for config, positive, negative in candidates]
    return (*selected, evidence)


def _freeze(root, output, policy_versions, config, calibration):
    if output.exists():
        raise FileExistsError(f"冻结目录已存在 {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="router-build-", dir=output.parent) as tmp:
        stage = Path(tmp) / "package"
        stage.mkdir()
        for letter, version in policy_versions.items():
            source = versions.snapshot_data(root, version)
            dest = stage / f"policy_{letter.lower()}"
            shutil.copytree(source, dest)
            if versions._version_id(versions._file_hashes(dest)) != version:
                raise ValueError(f"策略快照副本校验失败 {version}")
        shutil.copyfile(Path(__file__).with_name("style_router.py"), stage / "style_router.py")
        write_json(stage / "detector.json", config)
        write_json(stage / "calibration.json", calibration)
        write_json(stage / "submission.json", {"runtime": "python", "entrypoint": "style_router.py:StyleRouter"})
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".local/style-switch/router"))
    options = parser.parse_args()
    root = project_root()
    output = root / options.output
    pilot = read_json(EXPERIMENT / "pilot.json")
    local_cases = read_json(EXPERIMENT / "cases.json")
    excluded_seeds = {pilot["seed"], *(case["map_seed"] for case in local_cases)}
    for version in pilot["policies"].values():
        versions.verify(root, version)
    path_a, path_b = [versions.snapshot_data(root, pilot["policies"][letter]) for letter in ("A", "B")]
    policy = policy_loader(path_b)(path_a)
    splits = {split: {source: [_case_from_manifest(root, row, source, policy) for row in cases]
                      for source, cases in sources.items()}
              for split, sources in read_json(EXPERIMENT / "inputs.json").items()}
    layouts = {split: {c["grid"] for cases in sources.values() for c in cases}
               for split, sources in splits.items()}
    if layouts["train"] & layouts["holdout"] or any(
        c["seed"] in excluded_seeds for sources in splits.values() for cases in sources.values() for c in cases
    ):
        raise ValueError("校准、留出或整局验收地图重叠")
    config, train_boss, train_rookie, candidates = _fit(splits["train"]["official_boss"], splits["train"]["sdk_rookie"])
    calibration = {
        "config": config, "policy_versions": pilot["policies"], "sdk_rules": rules.RULES_VERSION,
        "archive_cutoff_inclusive": ARCHIVE_CUTOFF,
        "excluded_trials_seed_count": len(excluded_seeds),
        "fit_boss_by_challenger_version": dict(Counter(row["version"] for row in train_boss["rows"])),
        "evidence": {"coordinates": "rounded to replay precision before feature extraction",
                     "feature": "positive displacement along opponent heading with clear lanes in both frames",
                     "fallback": "stay with A when the streak is absent through the final eligible frame",
                     "time_seconds": LATEST_FRAME * rules.DT,
                     "classification": "switch evidence calibrated on boss versus SDK rookie only"},
        "selection_candidates": candidates,
        "train": {"official_boss": train_boss, "sdk_rookie": train_rookie},
        "holdout": {source: _evaluate(cases, config) for source, cases in splits["holdout"].items()},
    }
    _freeze(root, output, pilot["policies"], config, calibration)
    for split in ("train", "holdout"):
        for source, summary in calibration[split].items():
            print(split, source, {k: v for k, v in summary.items() if k != "rows"})
    print(output)


if __name__ == "__main__":
    main()
