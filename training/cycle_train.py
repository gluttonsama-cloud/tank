"""Independent local PvP and PvE CEM rounds with held-out paired evaluation."""

import os
import random
import re
import shutil
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from time import perf_counter, time

import numpy as np
from seedcup import __version__ as sdk_version
from seedcup.game import generate_map, rules
from seedcup.game.bots import available_tiers

from tank import (
    cycle_experience,
    practice_priority,
    sampling_data,
    sampling_probes,
    versions,
)
from tank.eval_plan import digest
from tank.eval_store import shard_lock, worker_lock
from tank.workspace import read_json, write_json

from .cycle_matches import (
    GameCase,
    Trial,
    game_metrics,
    open_pool,
    paired_games,
    paired_pool_series,
    paired_series,
    run_trials,
    search_utility,
)
from .cycle_protocol import equivalent_gameplay, search_protocol
from .cycle_protocol import game_protocol as _protocol
from .fit_boss_specialist import Settings, population, refit

ROUTER = "v-15e8911654f47c9b997e691e2e8ac68c5ae3db466a06d89503f89098bb7e4ca3"
LIVE_PVP = "v-a9092b1ae2c6bcab034e235ee27f36d20c416a3dd61f335e0a154bb2f9c56e4c"
LANCER = "v-2b9dfa9a5fa442edd0b7b4c223228034fce69d5b4e94e416d8f763244dc7e825"
OLD_MAIN = "v-0081349ee9d189d06bc534ecdac22f731b4908714c90b45644a1962aea0583f5"
WEAKNESS_PROBES = ("peek", "recontact", "holdout-orbit")
WORKER_LEVELS = (2, 4, 8, 12, 16)
SERIES_GROUPS = 8
TEST_MAPS = 12
SEARCH = Settings(generations=2, population=6, elites=2, maps_per_generation=3,
                  finalists_per_generation=1, validation_maps=5, initial_sigma=.26, sigma_floor=.09)
A_KNOBS = (
    ("zone_lookahead", r"(?m)(?<=^ZONE_LOOKAHEAD = )\d+", 30, 22),
    ("dodge_clearance", r"(?<=max\(0,)\d*\.\d+(?=-distance\)\*\(1-\.022\*i\)\))", .64, .15),
    ("contact_range", r"(?<=if dist < )\d+\.\d+(?= and cooldown <= \.2)", 2.5, 1),
)
ABILITY = "close_aim_blend"
FROZEN_INPUTS = "training/experiments/style-switch-v1/pilot.json"
EVENT_PARAMETERS = {
    "first_engagement": ("engage", "max_flight", "contact_range"),
    "consecutive_misses": ("max_flight", "prepare_weight", "min_hit_margin", ABILITY),
    "damage_exchange": ("dodge", "dodge_clearance", "anticipate"),
    "breaking_contact": ("engage", "spacing", "contact_range"),
    "zone_displacement": ("zone_lookahead", "dodge_clearance", "spacing"),
}


class TrainingPaused(Exception):
    """The daemon can resume the same round from its per-batch checkpoints."""


def _paths(root, records):
    for version in set(records.values()) - {v for v in records.values() if v.startswith("sdk:")}:
        versions.verify(root, version)
    return {key: value if value.startswith("sdk:") else str(versions.snapshot_data(root, value))
            for key, value in records.items()}


def _pve_pool(root):
    surrogate, = read_json(root / "suites/boss-specialist.json")["opponents"]
    return {f"sdk-{tier}": f"sdk:{tier}" for tier in available_tiers()} | {"v31": surrogate}


def _pvp_pool(root):
    path = root / ".local/cycle/pvp/opponents.json"
    with worker_lock(path.parent, "opponents") as acquired:
        if not acquired:
            raise RuntimeError("正在冻结 PvP 对手池")
        if path.exists():
            return read_json(path)
        first_plan = path.parent / "rounds/000001/plan.json"
        if first_plan.exists():
            opponents = read_json(first_plan)["pvp"]
        else:
            opponents = {"lancer-v7": LANCER, "router-v2": ROUTER, "old-main": OLD_MAIN}
            opponents.update(sampling_probes.snapshots(root, WEAKNESS_PROBES))
        write_json(path, opponents)
        return opponents


def _validate_router(root, pilot):
    versions.verify(root, ROUTER)
    folder = versions.snapshot_data(root, ROUTER)
    for side, version in pilot["policies"].items():
        versions.verify(root, version)
        source = versions.snapshot_data(root, version)
        if versions._file_hashes(folder / f"policy_{side.lower()}") != versions._file_hashes(source):
            raise ValueError(f"路由 v2 的子策略 {side} 与初始快照不符")


def _scalar(source, pattern):
    value, = re.findall(pattern, source)
    return float(value)


def _params(folder):
    a_source = (folder / "policy_a/bot.py").read_text(encoding="utf-8")
    a = [(_scalar(a_source, pattern) - default) / step for _, pattern, default, step in A_KNOBS]
    b = read_json(folder / "policy_b/weights.json")
    ability = read_json(folder / "cycle_ability.json")[ABILITY] * 2 if (folder / "cycle_ability.json").exists() else 0.
    return [name for name, *_ in A_KNOBS] + b["param_names"] + [ABILITY], tuple(a + b["params"] + [ability])


def _ability(folder, blend):
    ability_path = folder / "cycle_ability.json"
    if not ability_path.exists():
        manifest = read_json(folder / "submission.json")
        source, name = manifest["entrypoint"].split(":")
        module = ".".join(Path(source).with_suffix("").parts)
        (folder / "cycle_base.py").write_text(f"from {module} import {name} as BasePolicy\n", encoding="utf-8")
        shutil.copyfile(Path(__file__).with_name("cycle_policy.py"), folder / "cycle_policy.py")
        write_json(folder / "submission.json", manifest | {"entrypoint": "cycle_policy.py:CyclePolicy"})
    write_json(ability_path, {ABILITY: blend})


def _copy_package(source, directory):
    shutil.copytree(source, directory, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _variant(source, source_params, directory, vector):
    if tuple(vector) == source_params:
        return str(source)
    _copy_package(source, directory)
    file = directory / "policy_a/bot.py"
    a_source = file.read_bytes().decode("utf-8")
    prior_a_source = a_source
    for (_, pattern, default, step), normalized in zip(A_KNOBS, vector[:len(A_KNOBS)], strict=True):
        value = round(default + step * normalized, 6)
        if value == _scalar(a_source, pattern):
            continue
        # Preserve the source match shape and only substitute the tuned literal.
        literal = str(round(value)) if default == 30 else str(value)
        a_source = re.sub(pattern, literal, a_source)
    if a_source != prior_a_source:
        file.write_bytes(a_source.encode("utf-8"))
    weights = directory / "policy_b/weights.json"
    b_weights = read_json(weights)
    b_vector = list(vector[len(A_KNOBS):-1])
    if b_vector != b_weights["params"]:
        write_json(weights, b_weights | {"params": b_vector})
    if vector[-1] != source_params[-1]:
        _ability(directory, vector[-1] / 2)
    return str(directory)


@contextmanager
def _round_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "round.lock").open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"本轮训练已在运行：{out}") from error
        else:
            import fcntl

            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"本轮训练已在运行：{out}") from error
        yield


def _capacity(root, line, out, workers):
    db_path = root / ".local/cycle/control.sqlite3"
    if db_path.exists():
        with closing(sqlite3.connect(db_path)) as db:
            if db.execute("SELECT enabled FROM lines WHERE line=?", (line,)).fetchone() == (0,):
                raise TrainingPaused
    with shard_lock(root / ".local/cycle", "profile") as available:
        if not available:
            raise TrainingPaused
    path = out / "workers.json"
    capacity = read_json(path)["workers"] if path.exists() else workers
    if type(capacity) is not int or not 1 <= capacity <= WORKER_LEVELS[-1]:
        raise ValueError(f"workers 必须为 1–{WORKER_LEVELS[-1]} 的整数")
    return capacity


def _grid_hash(seed):
    return sampling_data.layout_hash(generate_map(seed).to_dict())


def _seed_source(seed, excluded=()):
    rng = random.Random(seed)
    grids = set(excluded)
    while True:
        value = rng.randrange(2**31)
        grid = _grid_hash(value)
        if grid in grids:
            continue
        grids.add(grid)
        yield (value, rng.getrandbits(32), rng.getrandbits(32))


def _conditions(opponents, count, mode, seeds):
    cases = []
    for style in opponents:
        for _ in range(count):
            matches = [next(seeds) for _ in range(11 if mode == "bo11" else 1)]
            for first_side in (0, 1):
                games = tuple(GameCase(seed, first_side ^ (i % 2), own, opp, i + 1)
                              for i, (seed, own, opp) in enumerate(matches))
                cases.append((style, games))
    return cases


def _trials(path, opponents, conditions):
    return [Trial(style, path, opponents[style], games) for style, games in conditions]


def _prior_rounds(root, line, before=None):
    folder = root / ".local/cycle" / line / "rounds"
    for path in sorted(folder.glob("[0-9]*/result.json"), reverse=True):
        if before is None or int(path.parent.name) < before:
            yield read_json(path)


def _feedback(root, line, round_number):
    previous = next(_prior_rounds(root, line, round_number), None)
    weak_styles, carryover = [], None
    if previous:
        style_metrics = previous["metrics"]["incumbent_by_style"]
        weak_styles = sorted(style_metrics, key=lambda style: search_utility(line, style_metrics[style]))[:2]
        carryover = {"candidate": previous["candidate"], "params": previous["params"]}
    donor_line = "pve" if line == "pvp" else "pvp"
    transfer = None
    for donor in _prior_rounds(root, donor_line):
        trial = donor["cross_line"]
        if ("evidence" in donor and trial["status"] == "tested" and trial["recipient_line"] == line
                and trial["utility_delta"] > 0 and not trial["after_metrics"]["invalid_actions"]):
            transfer = {"source": donor["candidate"], "round": donor["round"],
                        "recipient": trial["recipient"], "trial_variant": trial["variant"],
                        "value": trial["after"], "utility_delta": trial["utility_delta"]}
            break
    team = root / ".local/online/team-scores.json"
    team_filter = read_json(team).get("team_id") if line == "pvp" and team.is_file() else None
    evidence = cycle_experience.summary(root, line, training_team_id=team_filter,
                                       before_round=round_number if line=="pve" else None)
    focus = evidence["training_focus"]
    if line=="pve" and evidence["completed_gate_focus"]:
        past = evidence["completed_gate_focus"]
        focus = [next((item for item in past if item["target"]=="boss"),past[0]), *focus]
    return {"weak_styles": weak_styles, "carryover": carryover, "transfer": transfer, "experience": focus}


def _feedback_cases(feedback, line, opponents, generation):
    """Select observed train-only maps from diverse failure events for CEM."""
    cases, used = [], set()
    for focus in feedback.get("experience", []):
        style = f"sdk-{focus['target']}" if line == "pve" else "incumbent"
        if style not in opponents:
            style = "v31"
        entries = focus["entries"][generation - 1:] + focus["entries"][:generation - 1]
        for entry in entries:
            if entry["split"] == "train" and entry["grid_sha256"] not in used:
                map_seed = entry["map_seed"]
                if _grid_hash(map_seed) != entry["grid_sha256"]:
                    continue
                used.add(entry["grid_sha256"])
                rng = random.Random(f"{entry['game_id']}:{generation}")
                own, foe = rng.getrandbits(32), rng.getrandbits(32)
                cases.extend((style, (GameCase(map_seed, side, own, foe, 1),)) for side in (0, 1))
                break
        if len(cases) == 6:
            break
    return cases


def _game_comparison(candidates, mode, opponents, conditions, pool, capacity, out, protocol):
    metrics = {}
    for name, folder in candidates.items():
        rows = run_trials(pool, _trials(folder, opponents, conditions), capacity, out / f"{name}.json", protocol=protocol)
        summary = game_metrics(rows)
        metrics[name] = summary | {"utility": search_utility(mode, summary)}
    return metrics


def _search(folder, line, source, anchor, opponents, seeds, pool, capacity, feedback, protocol):
    seed = next(seeds)[0]
    rng = np.random.default_rng(seed)
    names, original = _params(source)
    mean, sigma = np.asarray(original), np.full(len(original), SEARCH.initial_sigma)
    focused = {name for focus in feedback.get("experience", []) if focus["entries"]
               for name in EVENT_PARAMETERS[focus["event"]]}
    for index, name in enumerate(names):
        if name in focused:
            sigma[index] *= 1.35
    leader = ("seed", original)
    vectors = {"seed": original}
    paths = {"seed": str(source), "incumbent": anchor}
    for generation in range(1, SEARCH.generations + 1):
        stage = folder / f"generation-{generation:02d}"
        members = population(rng, generation, SEARCH, mean, sigma, leader)
        if generation == 1:
            additions = {}
            if carryover := feedback.get("carryover"):
                additions["previous-candidate"] = tuple(carryover["params"][name] for name in names)
            if transfer := feedback["transfer"]:
                additions["transferred"] = (*original[:-1], transfer["value"] * 2)
            replaced = iter(reversed(tuple(members)))
            for key, warm in additions.items():
                if warm not in members.values():
                    del members[next(replaced)]
                    members[key] = warm
        for key, vector in members.items():
            if key not in paths:
                paths[key] = _variant(source, original, folder / "policies" / key, vector)
        cases = _conditions(opponents, SEARCH.maps_per_generation, "paired", seeds)
        weak = {key: opponents[key] for key in feedback["weak_styles"] if key in opponents}
        cases += _conditions(weak, 1, "paired", seeds)
        observed = _feedback_cases(feedback, line, opponents, generation)
        cases += observed
        metrics = _game_comparison({key: paths[key] for key in members}, line,
                                   opponents, cases, pool, capacity, stage, protocol)
        ranked = sorted(members, key=lambda key: metrics[key]["utility"], reverse=True)
        leader = (ranked[0], members[ranked[0]])
        vectors.update(members)
        mean, sigma = refit(mean, sigma, [members[key] for key in ranked[:SEARCH.elites]], SEARCH)
        write_json(stage / "metrics.json", {"parameters": members, "selected": leader[0], "metrics": metrics,
                                             "focused_parameters": sorted(focused),
                                             "observed_map_seeds": sorted({case[0].map_seed for _, case in observed})})
        for key in ranked:
            if key != "seed" and members[key][-1] != original[-1]:
                paths[f"dev-{generation:02d}"] = paths[key]
                vectors[f"dev-{generation:02d}"] = members[key]
                break
    finalists = {name: path for name, path in paths.items() if name.startswith("dev-")}
    dev_cases = _conditions(opponents, SEARCH.validation_maps, "paired", seeds)
    metrics = _game_comparison(finalists | {"incumbent": anchor}, line, opponents,
                               dev_cases, pool, capacity, folder / "dev", protocol)
    selected = max(finalists, key=lambda key: metrics[key]["utility"])
    write_json(folder / "dev/metrics.json", {"selected": selected, "metrics": metrics})
    return Path(finalists[selected]), vectors[selected]


def _style_metrics(results):
    return {name: game_metrics([row for row in results if row["style"] == name])
            for name in dict.fromkeys(row["style"] for row in results)}


def _acceptance(line, candidate, incumbent, opponents, seeds, pool, capacity, out, seed, protocol):
    mode = "bo11" if line == "pvp" else "paired"
    cases = _conditions(opponents, SERIES_GROUPS if line == "pvp" else TEST_MAPS, mode, seeds)
    new = run_trials(pool, _trials(candidate, opponents, cases), capacity, out / "candidate.json", protocol=protocol)
    old = run_trials(pool, _trials(incumbent, opponents, cases), capacity, out / "incumbent.json", protocol=protocol)
    duels = []
    if line == "pvp":
        direct = {"incumbent": incumbent}
        direct_cases = _conditions(direct, SERIES_GROUPS, mode, seeds)
        duels = run_trials(pool, _trials(candidate, direct, direct_cases), capacity, out / "direct.json", protocol=protocol)
    return _assess(line, new, old, duels, seed)


def _assess(line, new, old, duels, seed):
    new_metrics, old_metrics = game_metrics(new), game_metrics(old)
    metrics = {"candidate": new_metrics, "incumbent": old_metrics,
               "candidate_by_style": _style_metrics(new), "incumbent_by_style": _style_metrics(old)}
    if line == "pve":
        comparison = paired_games(new, old, seed)
        total_delta = round(new_metrics["total"] - old_metrics["total"], 2)
        return total_delta > 0 and not new_metrics["invalid_actions"], metrics | {
            "paired": comparison, "total_delta": total_delta}

    comparison, head_to_head = paired_series(new, old, duels, seed)
    direct_metrics = game_metrics(duels)
    groups = {len(values) for values in comparison["group_deltas"].values()} | {
        len(head_to_head["group_deltas"]["incumbent"])}
    if len(groups) != 1 or min(groups) < 8:
        raise ValueError("每类对手须至少有八组配对系列，且各类组数一致")
    accepted = (comparison["ci95"][0] > 0 and head_to_head["ci95"][0] > 0
                and not new_metrics["invalid_actions"] and not direct_metrics["invalid_actions"])
    return accepted, metrics | {"paired": comparison, "direct": head_to_head | {"metrics": direct_metrics},
                                "series_groups_per_style": next(iter(groups)), "max_games_per_series": 11}


def _cross_line(root, line, recipient, recipient_path, source_candidate, blend, opponents,
                seeds, pool, capacity, out, seed, protocol, cross_series_groups):
    policy_folder = out / "policy"
    existing = Path(recipient_path) / "cycle_ability.json"
    before = read_json(existing)[ABILITY] if existing.is_file() else 0.0
    if blend == before:
        return {"recipient": recipient, "source": source_candidate, "ability": ABILITY,
                "status": "already_present", "before": before, "after": blend}
    _copy_package(Path(recipient_path), policy_folder)
    _ability(policy_folder, blend)
    variant = versions.create_from_directory(root, policy_folder, "能力移植试验", "仅移植近距瞄准调整")["id"]
    mode = "bo11" if cross_series_groups else "paired"
    cases = _conditions(opponents, cross_series_groups or 4, mode, seeds)
    recipient_line = "pve" if line == "pvp" else "pvp"
    original = run_trials(pool, _trials(recipient_path, opponents, cases), capacity, out / "recipient.json", protocol=protocol)
    transfer = run_trials(pool, _trials(str(versions.snapshot_data(root, variant)), opponents, cases),
                          capacity, out / "transfer.json", protocol=protocol)
    return {"recipient_line": recipient_line, "recipient": recipient, "source": source_candidate,
            "ability": ABILITY, "variant": variant, "status": "tested", "before": before, "after": blend} | _transfer_metrics(
                recipient_line, original, transfer, seed, bool(cross_series_groups))


def _transfer_metrics(recipient_line, original, transfer, seed, bo11=False):
    original_metrics, transfer_metrics = game_metrics(original), game_metrics(transfer)
    delta = round(search_utility(recipient_line, transfer_metrics) -
                  search_utility(recipient_line, original_metrics), 6)
    result = {"before_metrics": original_metrics, "after_metrics": transfer_metrics, "utility_delta": delta}
    if not bo11:
        return result | {"paired_score": paired_games(transfer, original, seed), "mode": "single-games",
                         "evidence_level": "exploratory"}
    paired = paired_pool_series(transfer, original, seed)
    if min(map(len, paired["group_deltas"].values())) < 8:
        raise ValueError("能力移植每种对手须至少八组 BO11")
    return result | {"paired_series": paired, "mode": "bo11", "evidence_level": "recipient-bo11",
                     "transfer_passed": paired["ci95"][0] > 0 and not transfer_metrics["invalid_actions"]}


def _evidence_names(line, cross):
    names = ["test/candidate.json", "test/incumbent.json"]
    if line == "pvp":
        names.append("test/direct.json")
    if cross["status"] == "tested":
        names += ["cross-line/recipient.json", "cross-line/transfer.json"]
    return names


def _evidence(out, line, cross, protocol, prefix=Path()):
    names = _evidence_names(line, cross)
    files = {name: (prefix / name).as_posix() for name in names}
    return {"protocol": protocol, "files": files,
            "sha256": {name: sha256((out / file).read_bytes()).hexdigest() for name, file in files.items()}}


def verify_evidence(root, out, result):
    """Raise when a result cannot be traced to the exact current evaluation inputs."""
    evidence = result.get("evidence")
    if not evidence or not equivalent_gameplay(root, evidence["protocol"], _protocol(root)):
        raise ValueError("本地结果缺少同协议的策略快照和配对证据")
    line, cross = result["line"], result["cross_line"]
    policies = {"test/candidate.json": result["candidate"], "test/incumbent.json": result["incumbent"]}
    if line == "pvp":
        policies["test/direct.json"] = result["candidate"]
    if cross["status"] == "tested":
        policies["cross-line/recipient.json"] = cross["recipient"]
        policies["cross-line/transfer.json"] = cross["variant"]
    _paths(root, policies)
    if set(evidence["files"]) != policies.keys():
        raise ValueError("本地评测报告缺少配对记录")
    records = {}
    for name, expected in policies.items():
        file = out / evidence["files"][name]
        if sha256(file.read_bytes()).hexdigest() != evidence["sha256"][name]:
            raise ValueError(f"本地逐局记录校验失败：{name}")
        records[name] = read_json(file)
        record = records[name]["binding"]
        if record["protocol"] != evidence["protocol"] or record["contestants"] != [expected]:
            raise ValueError(f"本地逐局记录的策略或协议不一致：{name}")
        if name == "test/direct.json":
            expected_opponents = {"incumbent": result["incumbent"]}
        elif name.startswith("test/"):
            expected_opponents = result["inputs"][line]
        else:
            other = "pve" if line == "pvp" else "pvp"
            expected_opponents = result["inputs"][other]
            if other == "pvp":
                expected_opponents = {key: expected_opponents[key] for key in ("lancer-v7", "router-v2")}
        if record["opponents"] != expected_opponents:
            raise ValueError(f"本地逐局记录的对手快照不一致：{name}")
        _paths(root, expected_opponents)
    data = _paired_reports(records, "test", "candidate", "incumbent")
    duels = records["test/direct.json"]["trials"] if line == "pvp" else []
    accepted, metrics = _assess(line, *data, duels, result["map_seed_namespace"])
    if (metrics, accepted and result["candidate"] != result["incumbent"]) != (result["metrics"], result["accepted"]):
        raise ValueError("本地结论与已绑定的逐局结果不一致")
    if cross["status"] == "tested":
        original, transfer = _paired_reports(records, "cross-line", "recipient", "transfer")
        expected = _transfer_metrics(cross["recipient_line"], original, transfer, result["map_seed_namespace"],
                                     bool(result["inputs"].get("cross_series_groups")))
        annotation = {"mode", "evidence_level"} if not result["inputs"].get("cross_series_groups") else set()
        if any(cross.get(key, value if key in annotation else None) != value for key, value in expected.items()):
            raise ValueError("跨线试验与已绑定的逐局结果不一致")


def _paired_reports(records, parent, first, second):
    a, b = (records[f"{parent}/{key}.json"] for key in (first, second))
    if a["binding"]["schedule_sha256"] != b["binding"]["schedule_sha256"]:
        raise ValueError(f"配对安排不一致：{parent}")
    return a["trials"], b["trials"]


def _report_cases(report):
    """Recreate the complete frozen schedules, including games skipped after a BO11 clinch."""
    inputs, line = report["inputs"], report["line"]
    feedback = inputs.get("feedback", {"weak_styles": []})
    seeds = _round_seeds(report["map_seed_namespace"], inputs, feedback)
    _search_cases(line, inputs, report["incumbent"], seeds, feedback)
    pool = inputs[line]
    mode = "bo11" if line == "pvp" else "paired"
    count = inputs.get("series_groups", SERIES_GROUPS) if line == "pvp" else inputs.get("test_maps", TEST_MAPS)
    tests = _conditions(pool, count, mode, seeds)
    direct = _conditions({"incumbent": report["incumbent"]}, count, mode, seeds) if line == "pvp" else []
    other = "pve" if line == "pvp" else "pvp"
    cross_pool = inputs[other]
    if other == "pvp":
        cross_pool = {key: cross_pool[key] for key in ("lancer-v7", "router-v2")}
    cross_group_count = inputs.get("cross_series_groups", 0)
    return tests, direct, _conditions(cross_pool, cross_group_count or 4, "bo11" if cross_group_count else "paired", seeds)


def _search_cases(line, inputs, incumbent, seeds, feedback):
    search = inputs.get("search", asdict(SEARCH))
    pool = inputs[line]
    search_pool = pool if line == "pve" else pool | {"incumbent": incumbent}
    weak = {key: search_pool[key] for key in feedback["weak_styles"] if key in search_pool}
    cases = []
    next(seeds)
    for generation in range(1, search["generations"] + 1):
        cases.extend(_conditions(search_pool, search["maps_per_generation"], "paired", seeds))
        cases.extend(_conditions(weak, 1, "paired", seeds))
        cases.extend(_feedback_cases(feedback, line, search_pool, generation))
    return cases + _conditions(search_pool, search["validation_maps"], "paired", seeds)


def _round_seeds(seed, plan, feedback):
    excluded = set(plan.get("excluded_grids", ())) | {
        entry["grid_sha256"] for finding in feedback.get("experience", []) for entry in finding["entries"]}
    return _seed_source(seed, excluded)


def _round_seed(line, number):
    return int.from_bytes(sha256(f"cycle:{line}:{number}".encode()).digest()[:8], "big")


def local_training_grids(root):
    """Return every planned local train/dev layout, including unfinished rounds."""
    result = set()
    for line in ("pvp", "pve"):
        for path in (Path(root) / ".local/cycle" / line / "rounds").glob("*/plan.json"):
            plan = read_json(path)
            feedback = plan.get("feedback", {"weak_styles": []})
            seeds = _round_seeds(_round_seed(line, plan["round"]), plan, feedback)
            cases = _search_cases(line, plan, plan["incumbent"], seeds, feedback)
            result.update(_grid_hash(seed) for seed in {game.map_seed for _, games in cases for game in games})
    return result


def _done_revalidation(root, out, report, protocol, previous):
    names = _evidence_names(report["line"], report["cross_line"])
    archived = {name: read_json(out / previous.get("files", {}).get(name, name)) for name in names}
    for prefix in (out / "verified").glob("*"):
        if not all((prefix / name).is_file() for name in names):
            continue
        cached = {name: read_json(prefix / name) for name in names}
        bound_protocol = cached["test/candidate.json"]["binding"]["protocol"]
        if not equivalent_gameplay(root, bound_protocol, protocol):
            continue
        if any(cached[name]["trials"] != (entry["trials"] if isinstance(entry, dict) else entry)
               for name, entry in archived.items()):
            continue
        return prefix.relative_to(out), bound_protocol, {name: value["trials"] for name, value in cached.items()}
    return None


def revalidate(root, out, result=None, workers=4):
    """Replay the original holdout and transfer schedules on exact frozen policies."""
    root, out = Path(root).resolve(), Path(out).resolve()
    with _round_lock(out):
        path = out / "result.json"
        report = read_json(path)
        if result and any(result[key] != report[key] for key in ("line", "round", "incumbent", "candidate")):
            raise ValueError("结果与请求补验的轮次或策略不一致")
        protocol = _protocol(root)
        previous = report.get("evidence", {})
        cross = report["cross_line"]
        if cross["status"] == "tested" and "mode" not in cross:
            report = report | {"cross_line": cross | {"mode": "single-games", "evidence_level": "exploratory"}}
        if equivalent_gameplay(root, previous.get("protocol"), protocol):
            verify_evidence(root, out, report)
            if report["cross_line"] != cross:
                write_json(path, report)
            return report
        inputs, cross, line = report["inputs"], report["cross_line"], report["line"]
        if reused := _done_revalidation(root, out, report, protocol, previous):
            prefix, bound_protocol, completed = reused
            return _save_revalidation(root, out, report, prefix, bound_protocol, completed)
        pools = {key: _paths(root, inputs[key]) for key in ("pve", "pvp")}
        tests, direct, transfer_cases = _report_cases(report)
        runs = [(f"test/{key}.json", report[key], pools[line], tests) for key in ("candidate", "incumbent")]
        if line == "pvp":
            runs.append(("test/direct.json", report["candidate"], _paths(root, {"incumbent": report["incumbent"]}), direct))
        if cross["status"] == "tested":
            opponents = {style: pools[cross["recipient_line"]][style] for style, _ in transfer_cases}
            runs += [(f"cross-line/{name}.json", cross[key], opponents, transfer_cases)
                     for name, key in (("recipient", "recipient"), ("transfer", "variant"))]
        prefix = Path("verified") / protocol["runner_sha256"][:16]
        completed = {}
        try:
            with open_pool(workers) as pool:
                for name, version, opponents, cases in runs:
                    archived = read_json(out / previous.get("files", {}).get(name, name))
                    archived = archived["trials"] if isinstance(archived, dict) else archived
                    folder = _paths(root, {"policy": version})["policy"]
                    actual = run_trials(pool, _trials(folder, opponents, cases),
                                        lambda: min(workers, _capacity(root, line, out, workers)),
                                        out / prefix / name, protocol=protocol)
                    if actual != archived:
                        raise ValueError(f"旧报告与已绑定快照的配对复测不一致：{name}")
                    completed[name] = actual
        except TrainingPaused:
            return None
        return _save_revalidation(root, out, report, prefix, protocol, completed)


def _save_revalidation(root, out, report, prefix, protocol, completed):
    original = out / prefix / "legacy-result.json"
    if not original.exists():
        write_json(original, read_json(out / "result.json"))
    line, cross = report["line"], report["cross_line"]
    accepted, metrics = _assess(line, completed["test/candidate.json"], completed["test/incumbent.json"],
                                completed.get("test/direct.json", []), report["map_seed_namespace"])
    report |= {"metrics": metrics, "accepted": accepted and report["candidate"] != report["incumbent"],
               "evidence": _evidence(out, line, cross, protocol, prefix)}
    verify_evidence(root, out, report)
    write_json(out / "result.json", report)
    return report


def train(root, line, incumbent, round_number, workers, out, *, recipient_incumbent=None):
    """Run one resumable local round. Return None when cooperatively paused."""
    root, out = Path(root).resolve(), Path(out).resolve()
    if line not in ("pve", "pvp") or type(round_number) is not int or round_number < 1:
        raise ValueError("line 须为 pve 或 pvp，round_number 须为正整数")
    recipient = recipient_incumbent or (LIVE_PVP if line == "pve" else ROUTER)
    identity = {"line": line, "incumbent": incumbent, "round": round_number, "recipient": recipient}
    result_path, plan_path = out / "result.json", out / "plan.json"
    with _round_lock(out):
        if result_path.is_file():
            result = read_json(result_path)
            if any(result[key] != value for key, value in identity.items()):
                raise ValueError("该目录中的结果属于其他训练轮次")
            return result
        if rules.VARIANT:
            raise ValueError("训练须使用正式 SDK 规则")

        def capacity():
            return _capacity(root, line, out, workers)

        try:
            capacity()
            protocol = _protocol(root)
            search_identity = search_protocol()
            if plan_path.is_file():
                plan = read_json(plan_path)
                if (dict(plan, **identity) != plan or plan.get("search_protocol") != search_identity
                        or plan.get("protocol") != protocol
                        or (plan["rules"], plan["sdk"]) != (rules.RULES_VERSION, sdk_version)):
                    raise ValueError("训练轮次或测评协议变更，保留旧记录并开启新一轮")
            else:
                pilot = read_json(root / FROZEN_INPUTS)
                _validate_router(root, pilot)
                plan = identity | {"seed_policy": ROUTER, "initial_policies": pilot["policies"],
                                   "pvp": _pvp_pool(root), "pve": _pve_pool(root),
                                   "feedback": _feedback(root, line, round_number),
                                   "excluded_grids": sorted(cycle_experience.indexed_grids(root)),
                                   "rules": rules.RULES_VERSION, "sdk": sdk_version, "protocol": protocol,
                                   "search_protocol": search_identity, "search": asdict(SEARCH), "series_groups": SERIES_GROUPS,
                                   "test_maps": TEST_MAPS, "cross_series_groups": SERIES_GROUPS if line == "pve" else 0}
                write_json(plan_path, plan)
            inputs = _paths(root, {"incumbent": incumbent, "seed_policy": plan["seed_policy"], "recipient": recipient})
            incumbent_path, recipient_path = (inputs[key] for key in ("incumbent", "recipient"))
            pvp_pool, pve_pool = (_paths(root, plan[key]) for key in ("pvp", "pve"))
            opponents = pve_pool if line == "pve" else pvp_pool
            # Continue a trained line from its own promoted policy when available.
            incumbent_folder = Path(incumbent_path)
            source = incumbent_folder if (incumbent_folder / "policy_b/weights.json").is_file() else Path(inputs["seed_policy"])
            search_pool = opponents if line == "pve" else opponents | {"incumbent": incumbent_path}
            seed = _round_seed(line, round_number)
            feedback = plan.get("feedback", {"weak_styles": [], "transfer": None})
            seeds = _round_seeds(seed, plan, feedback)
            with open_pool(WORKER_LEVELS[-1]) as pool:
                path, vector = _search(out, line, source, incumbent_path, search_pool, seeds, pool, capacity, feedback, protocol)
                names, original = _params(source)
                candidate = versions.create_from_directory(root, path, f"cycle-{line}", "闭环对局搜索参数")["id"]
                candidate_path = str(versions.snapshot_data(root, candidate))
                accepted, metrics = _acceptance(line, candidate_path, incumbent_path, opponents,
                                                seeds, pool, capacity, out / "test", seed, protocol)
                cross_opponents = pve_pool if line == "pvp" else {key: pvp_pool[key] for key in ("lancer-v7", "router-v2")}
                cross = _cross_line(root, line, recipient, recipient_path, candidate, vector[-1] / 2,
                                    cross_opponents, seeds, pool, capacity, out / "cross-line", seed, protocol,
                                    plan["cross_series_groups"])
            capacity()
        except TrainingPaused:
            return None

        search_seed = incumbent if source == incumbent_folder else plan["seed_policy"]
        result = identity | {"candidate": candidate, "accepted": accepted and candidate != incumbent,
                             "metrics": metrics, "cross_line": cross, "seed_policy": search_seed,
                             "params": dict(zip(names, vector, strict=True)),
                             "before_params": dict(zip(names, original, strict=True)),
                             "inputs": plan, "map_seed_namespace": seed,
                             "evidence": _evidence(out, line, cross, protocol)}
        write_json(result_path, result)
        return result


def sdk_action_lag(root, start, end=None):
    """Summarize observed action reuse from practice records saved in the interval."""
    matches = confirmed = reused = 0
    end = time() if end is None else end
    for path in (root / ".local/sampling/observations").glob("*.json.gz"):
        finished = path.stat().st_mtime
        if not start <= finished <= end:
            continue
        for record_path in (root / ".local/online/practice").glob(f"*/*/{path.name.removesuffix('.gz')}"):
            report = read_json(record_path)
            timing = report.get("sdk_timing")
            if not timing or finished - report["result"]["seconds"] < start:
                continue
            matches += 1
            confirmed += timing["confirmed_actions"]
            reused += timing["reused_actions"]
            break
    return {"status": "observed" if confirmed else "not_observed", "matches": matches,
            "confirmed_actions": confirmed, "reused_actions": reused,
            "reuse_rate": reused / confirmed if confirmed else None}


def profile(root):
    """Benchmark identical cold and warm workloads at each supported worker count."""
    root = Path(root).resolve()
    with shard_lock(root / ".local/cycle", "profile") as acquired:
        if not acquired:
            raise RuntimeError("吞吐实测已在运行")
        return _profile(root)


def _profile_pause_reason(root, lag_baseline):
    cycle = root / ".local/cycle"
    db_path = cycle / "control.sqlite3"
    if db_path.exists():
        with closing(sqlite3.connect(db_path)) as db:
            enabled = [row[0] for row in db.execute("SELECT enabled FROM lines")]
        if enabled and not any(enabled):
            return "训练已全部暂停"
    team = root / ".local/online/team-scores.json"
    team_id = read_json(team).get("team_id") if team.is_file() else None
    if team_id is not None and practice_priority.foreground_pending(root, team_id):
        return "前台官网练习进行中"
    recent_lag = sdk_action_lag(root, time() - 120)["reuse_rate"]
    if recent_lag is not None and lag_baseline is not None and recent_lag - lag_baseline >= .02:
        return "SDK 动作沿用率上升"
    return None


def _profile(root):
    cycle = root / ".local/cycle"
    progress = cycle / "profile-progress.json"
    records = _pve_pool(root) | {"lancer-v7": LANCER}
    cases = _conditions(records, 8, "paired", _seed_source(1943054227))
    opponents = _paths(root, records)
    router = _paths(root, {"router": ROUTER})["router"]
    tasks = _trials(router, opponents, cases)
    identity = digest([_protocol(root), records, ROUTER, WORKER_LEVELS])
    previous = read_json(progress) if progress.exists() else None
    if previous and previous["identity"] == identity and previous["status"] != "completed":
        state = previous
    else:
        started_at = time()
        state = {"identity": identity, "profiles": [], "status": "ready", "active_workers": None,
                 "sdk_action_lag_baseline": sdk_action_lag(root, started_at - 600, started_at)}
    for line in ("pvp", "pve"):
        for lock_path in (cycle / line / "rounds").glob("*/round.lock"):
            try:
                with _round_lock(lock_path.parent):
                    pass
            except RuntimeError:
                state |= {"status": "paused", "reason": "等待本地对局批次结束", "active_workers": None}
                write_json(progress, state)
                return state

    rates, lag_baseline = state["profiles"], state["sdk_action_lag_baseline"]
    for count in WORKER_LEVELS[len(rates):]:
        state |= {"status": "running", "active_workers": count, "reason": None}
        write_json(progress, state)
        started = perf_counter()
        level_start = time()

        def capacity(level=count):
            if reason := _profile_pause_reason(root, lag_baseline["reuse_rate"]):
                raise TrainingPaused(reason)
            return level

        try:
            with open_pool(count) as pool:
                run_trials(pool, tasks[:count], capacity)
                warmed = perf_counter()
                rows = run_trials(pool, tasks, capacity)
                duration = perf_counter() - warmed
        except TrainingPaused as paused:
            state |= {"status": "paused", "reason": str(paused), "active_workers": None}
            write_json(progress, state)
            return state
        games = sum(len(row["games"]) for row in rows)
        observed = sdk_action_lag(root, level_start)
        lag_delta = (observed["reuse_rate"] - lag_baseline["reuse_rate"]
                     if observed["reuse_rate"] is not None and lag_baseline["reuse_rate"] is not None else None)
        rates.append({"workers": count, "games": games, "warmup_seconds": round(warmed - started, 3),
                      "seconds": round(duration, 3), "games_per_second": round(games / duration, 3),
                      "sdk_action_lag": observed, "lag_delta": lag_delta,
                      "eligible": (count <= 4 if lag_delta is None else lag_delta < .02),
                      "lag_validation": "pending" if lag_delta is None else "measured"})
        write_json(progress, state)
    eligible = [row for row in rates if row["eligible"]]
    workers = max(eligible, key=lambda row: row["games_per_second"])["workers"] if eligible else 1
    state |= {"workers": workers, "status": "completed", "active_workers": None}
    write_json(cycle / "profile.json", state)
    write_json(progress, state)
    return state
