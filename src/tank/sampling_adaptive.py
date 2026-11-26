"""Pick and freeze each new boss sampling map using observed training encounters."""
import gzip
import hashlib
import json
from collections import defaultdict

from seedcup.game import generate_map

from . import sampling_data as data
from . import sampling_probes as probes

EVENTS = ("reengage", "boss_withdraw", "after_hit_withdraw", "disadvantaged_close", "late_close")
CYCLE = ("gap", "intervention", "gap", "explore", "gap", "intervention", "gap", "repeat")
WINDOWS = ("first_close", "first_close_hit")
ARMS = ("window-fire", "window-hold")
RESERVED_PROBE = "holdout-orbit"


def history(root):
    with data.database(root) as db:
        rows = db.execute("""SELECT id,entry,json_extract(coverage,'$.encounters_v1') AS signals
            FROM examples WHERE target='boss' AND json_extract(entry,'$.split')='train'
            AND json_extract(entry,'$.grid_sha256') NOT IN (SELECT grid FROM sampling_holdouts)""").fetchall()
    updates, observations = [], []
    for row in rows:
        entry = json.loads(row["entry"])
        if row["signals"] is None:
            replay = json.loads(gzip.decompress((root / entry["path"]).read_bytes()))
            signals = probes.behavior_signals(replay, entry["target_side"])
            updates.append((json.dumps(signals), row["id"], entry["sha256"]))
        else:
            signals = json.loads(row["signals"])
        observations.append((entry["grid_sha256"], entry["version_id"], signals))
    if updates:
        with data.database(root) as db:
            db.executemany("""UPDATE examples SET coverage=json_set(coverage,'$.encounters_v1',json(?))
                WHERE target='boss' AND id=? AND json_extract(entry,'$.sha256')=?""", updates)
    return observations


def select_map(policy, sequence, observations):
    seen = defaultdict(set)
    hits = defaultdict(lambda: defaultdict(set))
    covered = {event: set() for event in EVENTS}
    layouts = set()
    for layout, version, signals in observations:
        layouts.add(layout)
        seen[version].add(layout)
        for event in EVENTS:
            if signals[event]:
                covered[event].add(layout)
                hits[version][event].add(layout)

    pool = policy["versions"]
    names = list(pool)
    # Exposure to the same layout from both sides counts once for coverage.
    def exposures(name):
        return len(seen[pool[name]])

    observed_pairs = sum(len(grids) for v, grids in seen.items() if v is not None)
    prior = {e: (1 + sum(len(hits[v][e]) for v in seen if v is not None)) / (2 + observed_pairs)
             for e in EVENTS}

    def chance(name, event):
        return (2 * prior[event] + len(hits[pool[name]][event])) / (2 + exposures(name))

    # Reward a second probe for events the earlier pick is unlikely to elicit.
    weights = {event: 1 / (1 + len(covered[event])) for event in EVENTS}
    picks = []

    def add(name, reason):
        picks.append({"probe": name, "version_id": pool[name], "reason": reason})
        for event in EVENTS:
            weights[event] *= 1 - chance(name, event)

    def remaining():
        used = {row["version_id"] for row in picks}
        return [name for name in names if pool[name] not in used]

    def gap_pick():
        available = remaining()
        tested = [name for name in available if exposures(name)]
        if not tested and "recontact" in available:
            return "recontact"
        return max(tested or available, key=lambda name: sum(chance(name, e) * weights[e] for e in EVENTS))

    add(min(policy["anchors"], key=exposures), "实战策略")
    if sequence % 8 == 7:
        add("approach", "同图开火对照")
        add("silent" if sequence % 16 == 7 else "pulse", "同图开火对照")
    else:
        add(gap_pick(), "补少见局面")
        if sequence % 3 == 0:
            candidates = [name for name in remaining() if name not in ("approach", "silent", "pulse")]
            add(min(candidates, key=exposures), "探索较少测试的打法")
        else:
            add(gap_pick(), "补少见局面")
    return {"picks": picks, "training_games": len(observations), "training_layouts": len(layouts),
            "event_layouts": {e: len(grids) for e, grids in covered.items()}}


def saved(root, run, index):
    with data.database(root) as db:
        row = db.execute("SELECT decision FROM sampling_maps WHERE run=? AND map_index=?", (run, index)).fetchone()
    return json.loads(row[0]) if row else None


def layout_assignment(root, seed):
    grid = data.layout_hash(generate_map(seed).to_dict())
    with data.database(root) as db:
        known = db.execute("SELECT split FROM layouts WHERE grid=?", (grid,)).fetchone()
        reserved = db.execute("SELECT 1 FROM sampling_holdouts WHERE grid=?", (grid,)).fetchone()
    split = known[0] if known else data.default_split(grid)
    return grid, "holdout" if reserved and split == "train" else split


def opportunity_plan(policy, sequence, rows, seed, grid, split):
    pool = policy["versions"]
    picks = []

    def add(name, role, repeat=1):
        picks.append({"probe": name, "version_id": pool[name], "role": role, "repeat": repeat})

    result = {"protocol": 2, "seed": seed, "planned_grid": grid, "planned_split": split, "picks": picks}
    if split != "train":
        for name in (*policy["anchors"], RESERVED_PROBE if split == "blind" else "recontact"):
            add(name, "reference" if name in policy["anchors"] else "fixed")
        return result | {"kind": "fixed"}

    kind = CYCLE[sequence % len(CYCLE)]
    anchor = policy["anchors"][sequence % len(policy["anchors"])]
    add(anchor, "reference")
    seen = defaultdict(set)
    outcomes = ("withdrawn", "stayed", "ended")
    cells = [(event, outcome) for event in WINDOWS for outcome in outcomes]
    covered = {cell: set() for cell in cells}
    by_version = defaultdict(lambda: defaultdict(set))
    for layout, version, opportunities in rows:
        seen[version].add(layout)
        for event in WINDOWS:
            opportunity = opportunities[event]
            if opportunity is not None:
                cell = (event, opportunity["outcome"])
                covered[cell].add(layout)
                by_version[version][cell].add(layout)

    result.update(kind=kind, sequence=sequence + 1, cycle=sequence // len(CYCLE) + 1,
                  training_games=len(rows), training_layouts=len({row[0] for row in rows}),
                  opportunity_layouts={event: {"eligible": len(set.union(*(covered[event, o] for o in outcomes))),
                      **{o: len(covered[event, o]) for o in outcomes}} for event in WINDOWS})
    candidates = [name for name in pool if name not in (*ARMS, RESERVED_PROBE)]

    def remaining():
        used = {p["version_id"] for p in picks}
        return [n for n in candidates if pool[n] not in used]

    # Use distinct exposed layouts as the denominator for each frozen probe.
    # A stopped window is visible in the ledger but earns no selection credit.
    cells_to_fill = [c for c in cells if c[1] != "ended"]
    exposures = sum(len(layouts) for version, layouts in seen.items() if version is not None)
    priors = {cell: (1 + sum(len(hits[cell]) for v, hits in by_version.items() if v is not None))
              / (2 + exposures) for cell in cells_to_fill}

    def chance(version, cell):
        return (len(by_version[version][cell]) + 2 * priors[cell]) / (len(seen[version]) + 2)

    def gap():
        weights = {c: 1 / (1 + len(covered[c])) for c in cells_to_fill}
        for cell in cells_to_fill:
            for p in picks:
                weights[cell] *= 1 - chance(p["version_id"], cell)

        def value(name):
            return sum(weight * chance(pool[name], cell) for cell, weight in weights.items())
        return max(remaining(), key=lambda name: (value(name), name == "recontact", name))

    if kind == "intervention":
        # A stable hash randomises order independently of both arms' outcomes.
        order = hashlib.sha256(str(seed).encode("ascii")).digest()[0] % 2
        for name in (ARMS[order], ARMS[1 - order]):
            add(name, "intervention")
    elif kind == "explore":
        for _ in range(2):
            add(min(remaining(), key=lambda n: (len(seen[pool[n]]), n)), "explore")
    elif kind == "repeat":
        add(anchor, "repeat", 2)
        add(gap(), "gap")
    else:
        for _ in range(2):
            add(gap(), "gap")
    return result


def _opportunity_decision(root, run, index, plan, seed):
    from . import sampling_opportunities

    policy = plan["adaptive"]
    data.initialize_splits(root)
    grid, split = layout_assignment(root, seed)
    if split != "train":
        return opportunity_plan(policy, 0, (), seed, grid, split)
    with data.database(root) as db:
        counts = db.execute("""SELECT COUNT(*),
            COUNT(*) FILTER (WHERE json_extract(decision,'$.planned_split')='train')
            FROM sampling_maps WHERE run=? AND map_index>=? AND map_index<?""",
            (run, policy["cycle_start_index"], index)).fetchone()
        if counts[0] != index - policy["cycle_start_index"]:
            raise ValueError("须先排定上一张采样地图")
        sequence = counts[1]
        prior = db.execute("""SELECT decision FROM sampling_maps WHERE run=?
            AND json_extract(decision,'$.sequence')=?""", (run, sequence + 1 - 2 * len(CYCLE))).fetchone()
    result = opportunity_plan(policy, sequence, sampling_opportunities.history(root), seed, grid, split)
    if prior and sequence % len(CYCLE) == 0:
        previous = json.loads(prior[0])["opportunity_layouts"]
        result["no_growth_two_cycles"] = [event for event in WINDOWS
            if result["opportunity_layouts"][event]["withdrawn"] <= previous[event]["withdrawn"]]
    return result


def _settings(db, run):
    return tuple(db.execute("""SELECT plan,(SELECT count(*) FROM sampling_holdouts),
                            (SELECT count(*) FROM sampling_reserved_versions) FROM runs WHERE id=?""",
                            (run,)).fetchone())


def decision(root, run, index):
    cached = saved(root, run, index)
    if cached is not None:
        return cached
    while True:
        with data.database(root) as db:
            settings = _settings(db, run)
        plan = json.loads(settings[0])
        policy = plan["adaptive"]
        seed = (plan["seed"] + index * 104729) % 2**31
        if "cycle_start_index" in policy and index >= policy["cycle_start_index"]:
            result = _opportunity_decision(root, run, index, plan, seed)
        else:
            result = select_map(policy, index - policy["start_step"] // plan["width"], history(root))
            result["seed"] = seed
        with data.database(root) as db:
            db.execute("BEGIN IMMEDIATE")
            if _settings(db, run) != settings:
                continue
            db.execute("INSERT OR IGNORE INTO sampling_maps VALUES (?,?,?)", (run, index, json.dumps(result)))
            row = db.execute("SELECT decision FROM sampling_maps WHERE run=? AND map_index=?", (run, index)).fetchone()
        return json.loads(row[0])
