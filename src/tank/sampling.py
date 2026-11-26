"""Resumable online practice campaigns and replay collection CLI."""
import argparse
import getpass
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

from . import online, versions
from . import sampling_adaptive as adaptive
from . import sampling_data as data
from . import sampling_probes as probes
from .eval_store import shard_lock
from .workspace import project_root, read_json

ROTATION = (("approach", "silent", "pulse"), ("approach", "retreat", "peek"),
            ("peek", "late-zone", "lancer"), ("retreat", "late-zone", "silent"))
BOSS_ROTATION = (("approach", "silent", "pulse"), ("approach", "lancer", "retreat"),
                 ("approach", "lancer", "peek"), ("approach", "lancer", "late-zone"))
RUN = re.compile(r"sample-[a-f0-9]{32}")


def location(root, key):
    if not RUN.fullmatch(key):
        raise ValueError("采样 ID 格式错误")
    return root / ".local/sampling/runs" / key


def record(root, key):
    location(root, key)
    with data.database(root) as db:
        row = db.execute("SELECT * FROM runs WHERE id=?", (key,)).fetchone()
    if row is None:
        raise ValueError(f"找不到采样任务 {key}")
    return dict(row) | {"plan": json.loads(row["plan"])}


def update(root, key, state, message=""):
    with data.database(root) as db:
        db.execute("UPDATE runs SET state=?,message=?,updated=? WHERE id=?", (state, message, time.time(), key))


def case(plan, step):
    count = plan["width"]
    index, pos = divmod(step, count)
    names = plan["probes"]
    if not names:
        rotation = plan["rotation"]
        names = rotation[index % len(rotation)]
    name = names[pos % len(names)]
    return (plan["seed"] + index * 104729) % 2**31, name, plan["versions"][name]


def adaptive_step(plan, step):
    return ("adaptive" in plan and step >= plan["adaptive"]["start_step"]
            and not cycle_step(plan, step))


def cycle_step(plan, step):
    return "cycle" in plan and step >= plan["cycle"]["from_step"]


def assigned_case(root, key, plan, step):
    if cycle_step(plan, step):
        from .cycle_pve import collection_case
        return collection_case(root, key, plan, step)
    if not adaptive_step(plan, step):
        return case(plan, step)
    index, pos = divmod(step, plan["width"])
    choice = adaptive.decision(root, key, index)
    pick = choice["picks"][pos]
    return choice["seed"], pick["probe"], pick["version_id"]


def status(root, key):
    row = record(root, key)
    plan = row["plan"]
    state = row["state"]
    if state in ("running", "waiting", "pausing") or (state == "queued" and time.time() - row["updated"] > 30):
        with shard_lock(location(root, key), 0) as free:
            if free:
                row = record(root, key)
                state = row["state"]
                if state in ("running", "waiting", "pausing", "queued"):
                    state = "paused" if row["paused"] else "interrupted"
    with data.database(root) as db:
        games, sdk_games = db.execute("""SELECT COUNT(*),COUNT(o.id) FROM slots s
            LEFT JOIN sdk_observations o ON o.id=s.id AND o.target=? WHERE s.run=?""",
                                      (plan["target"], key)).fetchone()
        child = None
        if plan["mode"] == "active" and plan["maps"] is None and state in ("running", "waiting"):
            child = db.execute("""SELECT id,state,message FROM runs WHERE json_extract(plan,'$.parent')=?
                               AND state IN ('running','waiting','pausing') ORDER BY updated DESC LIMIT 1""",
                               (key,)).fetchone()
    result = {"id": key, "mode": plan["mode"], "state": state, "target": plan["target"], "message": row["message"]}
    if child:
        with shard_lock(location(root, child["id"]), 0) as free:
            if not free:
                result.update(state="running", message="正在执行定向批次", active_batch=dict(child))
    if plan["mode"] == "active":
        result.update(games=games, sdk_games=sdk_games,
                      planned_games=plan["maps"] * plan["width"] * 2 if plan["maps"] else None)
        if "parent" in plan:
            result.update(parent=plan["parent"], repeats=plan["repeats"])
        if cycle_step(plan, row["step"]):
            from .cycle_pve import collection_case
            result["policy"] = "cycle-pve"
            result["map_seed"], result["probe"], _ = collection_case(root, key, plan, row["step"], freeze=False)
        elif "adaptive" in plan:
            start_index = plan["adaptive"].get("cycle_start_index")
            result["policy"] = "opportunity-v2" if start_index is not None else "coverage"
            if start_index is not None:
                result["opportunity_from_map"] = start_index + 1
            result["adaptive_from_step"] = plan["adaptive"]["start_step"]
            index = max(row["step"], plan["adaptive"]["start_step"]) // plan["width"]
            decision = adaptive.saved(root, key, index)
            if decision:
                result["map_plan"] = decision
        if state != "completed" and adaptive_step(plan, row["step"]):
            index, pos = divmod(row["step"], plan["width"])
            result["map_seed"] = (plan["seed"] + index * 104729) % 2**31
            if decision:
                result["probe"] = decision["picks"][pos]["probe"]
                if decision.get("protocol") == 2:
                    result["role"] = decision["picks"][pos]["role"]
                    result["repeat"] = decision["picks"][pos]["repeat"]
        elif state != "completed" and not cycle_step(plan, row["step"]):
            seed, name, _ = case(plan, row["step"])
            result.update(map_seed=seed, probe=name)
    return result


def enqueue(root, parent_id, *, strategies, maps, repeats, seed, target=None):
    parent = record(root, parent_id)["plan"]
    if parent["mode"] != "active" or parent["maps"] is not None:
        raise ValueError("请选择持续主动采样任务")
    if maps < 1 or repeats < 1 or not 0 <= seed < 2**31:
        raise ValueError("地图数和重复数须为正数，种子须在 0 至 2³¹-1 之间")
    target = parent["target"] if target is None else target
    if target not in online.PRACTICE_LEVELS:
        raise ValueError("定向采样请选择官网练习人机档位")
    selected = dict(strategies)
    if not selected or len(selected) != len(strategies) or len(set(selected.values())) != len(selected):
        raise ValueError("请为每种打法指定不同的名称和快照")
    for version in selected.values():
        versions.verify(root, version)
    plan = {"parent": parent_id, "mode": "active", "target": target, "seed": seed,
            "maps": maps, "probes": list(selected), "versions": selected, "repeats": repeats,
            "width": len(selected) * repeats, "interval": parent["interval"]}
    with data.database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        batches = db.execute("SELECT id,plan FROM runs WHERE json_extract(plan,'$.parent')=?", (parent_id,))
        key = next((row["id"] for row in batches if json.loads(row["plan"]) == plan), None)
        if key is None:
            key = "sample-" + uuid4().hex
            db.execute("INSERT INTO runs(id,plan,state,updated) VALUES (?,?,?,?)",
                       (key, json.dumps(plan), "scheduled", time.time()))
    return status(root, key)


def launch(root, key, foreground):
    if foreground:
        work(root, key)
    else:
        folder = location(root, key)
        folder.mkdir(parents=True, exist_ok=True)
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with (folder / "worker.log").open("ab") as log:
            subprocess.Popen([sys.executable, "-B", "-m", "tank.sampling", key], cwd=root, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=log, creationflags=flags, start_new_session=os.name != "nt")
    return status(root, key)


def check_host(root):
    from . import practice_host
    if practice_host.is_remote(root):
        raise ValueError("请在共享练习主机上启动主动采样；本机可运行 sample watch 收集已有回放")
    online.sampling_credentials(root)


def adaptive_policy(root, frozen_probes, specialist, start_step):
    pool = dict(frozen_probes)
    missing = [n for n in ("recontact", *adaptive.ARMS, adaptive.RESERVED_PROBE) if n not in pool]
    if missing:
        pool.update(probes.snapshots(root, missing))
    if specialist is None:
        specialist = versions.create(root, "boss-specialist", "常驻采样实战对手")["id"]
    else:
        versions.verify(root, specialist)
    if specialist in pool.values():
        raise ValueError("专项快照须与其他采样打法不同")
    return {"start_step": start_step, "cycle_start_index": 0, "anchors": ["specialist", "lancer"],
            "versions": {"specialist": specialist, **pool}}


def adapt(root, key, specialist=None, holdout_manifest=None):
    row = record(root, key)
    plan = row["plan"]
    if (plan["mode"] != "active" or plan["target"] != "boss" or plan["maps"] is not None
            or plan["probes"] is not None or plan["width"] != 3):
        raise ValueError("请选择使用默认打法的持续终极档采样任务")
    prior = plan.get("adaptive")
    pool = plan["versions"] if prior is None else {k: v for k, v in prior["versions"].items() if k != "specialist"}
    selected = specialist or (prior["versions"]["specialist"] if prior else None)
    policy = adaptive_policy(root, pool, selected, 0)
    reserved = set()
    if holdout_manifest is not None:
        manifest = read_json(holdout_manifest)
        reserved = {r["grid_sha256"] for split in ("dev", "blind") for r in manifest[split]}
        if any(not re.fullmatch(r"[0-9a-f]{64}", grid) for grid in reserved):
            raise ValueError("留出清单的 grid_sha256 无效")
    with data.database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("SELECT step,plan FROM runs WHERE id=?", (key,)).fetchone()
        plan = json.loads(current["plan"])
        old = plan.get("adaptive")
        if old and specialist is None:
            policy["versions"]["specialist"] = old["versions"]["specialist"]
        width = plan["width"]
        next_whole_map = (current["step"] + width - 1) // width
        if not old:
            # Legacy rooms can already have an unindexed receipt at slot one.
            next_whole_map = current["step"] // width + 1
        policy["start_step"] = old["start_step"] if old else next_whole_map * width
        last = db.execute("SELECT MAX(map_index) FROM sampling_maps WHERE run=?", (key,)).fetchone()[0]
        policy["cycle_start_index"] = (old["cycle_start_index"] if old and "cycle_start_index" in old
                                       else max(next_whole_map, last + 1 if last is not None else 0))
        plan["adaptive"] = policy
        db.executemany("INSERT OR IGNORE INTO sampling_holdouts VALUES (?)", ((grid,) for grid in reserved))
        db.execute("INSERT OR IGNORE INTO sampling_reserved_versions VALUES (?)",
                   (policy["versions"][adaptive.RESERVED_PROBE],))
        db.execute("UPDATE runs SET plan=? WHERE id=?", (json.dumps(plan), key))
    index = max(current["step"], policy["start_step"]) // plan["width"]
    adaptive.decision(root, key, index)
    return status(root, key)


def start(root, *, target, maps=None, seed=None, names=None, mode="active", interval=60, foreground=False):
    data.target_key(target)
    plan = {"mode": mode, "target": target, "interval": interval}
    if interval < 1:
        raise ValueError("interval 须为正数")
    if mode == "active":
        check_host(root)
        if target not in online.PRACTICE_LEVELS:
            raise ValueError("主动开房请选择官方人机档位；队伍对手使用 sample watch 或 sample import")
        if maps is not None and not 1 <= maps <= 2**31:
            raise ValueError("地图数须在 1 至 2³¹ 之间")
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big") % 2**31
        if not 0 <= seed < 2**31:
            raise ValueError("起始地图种子须在 0 至 2³¹-1 之间")
        if names is not None and (not names or len(names) != len(set(names))):
            raise ValueError("采样打法不能为空或重复")
        rotation = BOSS_ROTATION if target == "boss" else ROTATION
        by_coverage = target == "boss" and maps is None and names is None
        selected = list(probes.catalog()) if by_coverage else sorted({name for group in rotation for name in group})
        snaps = probes.snapshots(root, names or selected)
        plan.update(seed=seed, maps=maps, probes=names, rotation=None if names else rotation,
                    versions=snaps, width=len(names) if names else len(rotation[0]))
        if by_coverage:
            plan["adaptive"] = adaptive_policy(root, snaps, None, 0)
    key = "sample-" + uuid4().hex
    with data.database(root) as db:
        db.execute("INSERT INTO runs(id,plan,state,updated) VALUES (?,?,?,?)",
                   (key, json.dumps(plan), "queued", time.time()))
    try:
        return launch(root, key, foreground)
    except OSError as error:
        update(root, key, "failed", str(error))
        raise


def pause(root, key):
    current = status(root, key)
    if current["state"] in ("completed", "failed"):
        return current
    with data.database(root) as db:
        db.execute("""UPDATE runs SET paused=1,state=CASE WHEN state='scheduled' THEN 'paused' ELSE 'pausing' END,updated=?
                      WHERE id=? AND state NOT IN ('completed','failed')""", (time.time(), key))
    return status(root, key)


def resume(root, key, foreground):
    current = status(root, key)
    if current["state"] in ("scheduled", "queued", "running", "waiting", "pausing", "completed"):
        return current
    if current["mode"] == "active":
        check_host(root)
    if record(root, key)["plan"].get("parent"):
        with data.database(root) as db:
            db.execute("UPDATE runs SET paused=0,state='scheduled',message='',updated=? WHERE id=?",
                       (time.time(), key))
        return status(root, key)
    # A generation still holding the worker lock must finish releasing its room first.
    with shard_lock(location(root, key), 0) as free:
        if not free:
            return status(root, key)
        with data.database(root) as db:
            db.execute("UPDATE runs SET paused=0,state='queued',message='',updated=? WHERE id=?", (time.time(), key))
    return launch(root, key, foreground)


def receipts(root, key, step, version, level, *, recover=False):
    tag = f"{key}-{step}"
    items = {int(p.stem): read_json(p) for p in (root / ".local/sampling/receipts" / tag).glob("*.json")}
    if recover:
        for p in (root / ".local/online/practice" / version / level).glob("*.json"):
            r = read_json(p)
            if r.get("case_tag") == tag:
                items[r["id"]] = r
    return items


def save_cases(root, key, step, plan, records, client):
    seed, name, version = assigned_case(root, key, plan, step)
    options = {}
    if adaptive_step(plan, step):
        index, pos = divmod(step, plan["width"])
        decision = adaptive.saved(root, key, index)
        if decision.get("protocol") == 2:
            pick = decision["picks"][pos]
            options["collection"] = {"protocol": 2, "run": key, "map_index": index, "slot": pos + 1,
                "kind": decision["kind"], "role": pick["role"], "repeat": pick["repeat"],
                "cycle": decision.get("cycle"), "sequence": decision.get("sequence"),
                "planned_grid": decision["planned_grid"], "planned_split": decision["planned_split"]}
    with data.database(root) as db:
        seen = {r[0] for r in db.execute("SELECT side FROM slots WHERE run=? AND step=?", (key, step))}
    for r in sorted(records, key=lambda item: item["id"]):
        side = r["side"]
        if (r["case_tag"] != f"{key}-{step}" or r["version_id"] != version or r["level"] != plan["target"]
                or r["map_seed"] != seed or side not in ("red","blue")):
            raise ValueError(f"采样格与官网练习记录不符 {r['id']}")
        if side in seen:
            continue
        if (root / ".local/sampling/skipped" / f"{r['id']}.json").is_file():
            continue
        if "observation_file" in r:
            path = root / ".local/sampling/observations" / f"{r['id']}.json.gz"
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != r["observation_sha256"]:
                raise ValueError(f"SDK 观测文件校验失败 {r['id']}")
            trace = json.loads(gzip.decompress(raw))
            if (trace["start"]["match_id"] != r["id"] or trace["end"]["match_id"] != r["id"]
                    or trace["start"]["side"] != side or trace["start"]["map"]["seed"] != seed):
                raise ValueError(f"SDK 观测与采样记录不符 {r['id']}")
            grid = data.sdk_layout_hash(trace["start"])
            entry = r | options | {"probe": name, "frames": len(trace["frames"]),
                "grid_sha256": grid, "target_side": "blue" if side == "red" else "red"}
            if "collection" in entry:
                entry["collection"] = entry["collection"] | {"actual_grid": grid,
                    "layout_match": grid == entry["collection"]["planned_grid"]}
            with data.database(root) as db:
                db.execute("INSERT OR IGNORE INTO sdk_observations VALUES (?,?,?)",
                           (r["id"], plan["target"], json.dumps(entry)))
                db.execute("INSERT OR IGNORE INTO slots VALUES (?,?,?,?)", (key, step, side, r["id"]))
            seen.add(side)
            continue
        path = online.practice_replay_path(root, r["id"])
        if not path.is_file():
            online.download_sample_replay(root, r, client)
        data.ingest(root, path.read_bytes(), plan["target"], match_id=r["id"], practice=r, probe=name, **options)
        with data.database(root) as db:
            db.execute("INSERT OR IGNORE INTO slots VALUES (?,?,?,?)", (key, step, side, r["id"]))
        seen.add(side)
    return len(seen) == 2


def run_scheduled(root, parent, stop, client, *, one_map=False):
    with data.database(root) as db:
        pending = db.execute("""SELECT id FROM runs WHERE json_extract(plan,'$.parent')=?
            AND paused=0 AND state IN ('scheduled','running','waiting') ORDER BY updated,rowid
            LIMIT ?""", (parent, 1 if one_map else -1)).fetchall()
    for (key,) in pending:
        if stop():
            return
        with shard_lock(location(root, key), 0) as acquired:
            if not acquired:
                continue
            row = record(root, key)
            if row["paused"]:
                continue
            last_check, paused = 0., False

            def child_stop(batch=key):
                nonlocal last_check, paused
                if time.monotonic() >= last_check:
                    paused = bool(record(root, batch)["paused"])
                    last_check = time.monotonic() + .5
                return paused or stop()

            update(root, key, "running")
            try:
                state = drive(root, key, row["plan"], child_stop, client=client, yield_after_map=one_map)
            except Exception as error:  # noqa: BLE001 - 子批异常写入其日志，父采集继续。
                with (location(root, key) / "worker.log").open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
                update(root, key, "failed", str(error))
                continue
            if state in ("paused", "scheduled"):
                with data.database(root) as db:
                    db.execute("""UPDATE runs SET state=CASE WHEN paused=1 THEN 'paused' ELSE 'scheduled' END,
                                  message='',updated=? WHERE id=?""", (time.time(), key))
            else:
                update(root, key, state)


def drive(root, key, plan, stop, *, client=None, yield_after_map=False):
    step = record(root, key)["step"]
    recover = True
    failures = 0
    while not stop():
        if plan["mode"] == "active" and plan["maps"] is not None and step >= plan["maps"] * plan["width"]:
            return "completed"
        deferred = False
        try:
            if client is None:
                client = online.sampling_client(root)
            if plan["mode"] == "watch":
                result = data.pull(root, plan["target"], client=client)
                message = f"本轮新增 {result['added']} 场，{len(result['pending'])} 场等待回放"
                delay = plan["interval"]
            else:
                if plan["maps"] is None:
                    plan = record(root, key)["plan"]
                scheduled_turn = plan["maps"] is None and ("adaptive" not in plan or step % plan["width"] == 0)
                if scheduled_turn:
                    if adaptive_step(plan, step):
                        locked = adaptive.saved(root, key, step // plan["width"])
                        current = locked["picks"][0]["version_id"] if locked else None
                    else:
                        current = assigned_case(root, key, plan, step)[2]
                    if current is None or not receipts(root, key, step, current, plan["target"], recover=recover):
                        run_scheduled(root, key, stop, client, one_map="adaptive" in plan and not cycle_step(plan, step))
                        if stop():
                            return "paused"
                        plan = record(root, key)["plan"]
                seed, probe_name, version = assigned_case(root, key, plan, step)
                existing = receipts(root, key, step, version, plan["target"], recover=recover)
                if not save_cases(root, key, step, plan, existing.values(), client):
                    update(root, key, "running")
                    options = {"trace_window": True} if probe_name in adaptive.ARMS else {}
                    batch = online.collect_practice(root, version, client=client, games=2, seed=seed,
                                                    level=plan["target"], stop=stop, case_tag=f"{key}-{step}", **options)
                    if not save_cases(root, key, step, plan, batch["matches"], client):
                        raise ValueError("官网未返回完整换边记录")
                step += 1
                recover = False
                with data.database(root) as db:
                    db.execute("UPDATE runs SET step=?,updated=? WHERE id=?", (step, time.time(), key))
                if plan["maps"] is not None and step == plan["maps"] * plan["width"]:
                    return "completed"
                if yield_after_map and step % plan["width"] == 0:
                    return "scheduled"
                message, delay = "等待下一组换边", plan["interval"]
            failures = 0
        except online.PracticeDeferred as error:
            deferred = True
            ready = [r for r in error.matches if "observation_file" in r or online.practice_replay_path(root, r["id"]).is_file()]
            if ready:
                save_cases(root, key, step, plan, ready, client)
            message = str(error)
            delay = max(5, min(plan["interval"], 30))
        except (online.RequestUncertain, online.RemoteError) as error:
            deferred = True
            if isinstance(error, online.RemoteError):
                if error.status == 401 and failures == 0:
                    client = None
                elif error.status != 429 and not (error.status == 404 and str(error) == "回放还在生成，稍后再试"):
                    raise
            failures += 1
            message, delay = str(error), min(300, 10 * 2**min(failures-1, 5))
        update(root, key, "waiting", message)
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline and not stop():
            time.sleep(min(.5, max(0, deadline-time.monotonic())))
        if yield_after_map and deferred and not stop():
            return "scheduled"
    return "paused"


def work(root, key):
    folder = location(root, key)
    with shard_lock(folder, 0) as free:
        if not free:
            return
        row = record(root, key)
        if row["state"] == "completed":
            return
        plan = row["plan"]
        # A single active sampler per host and one watcher per target.
        slot = "active" if plan["mode"] == "active" else "watch-" + plan["target"].replace(":", "-")
        with shard_lock(folder.parent.parent / "workers" / slot, 0) as primary:
            if not primary:
                update(root, key, "failed", "同类采集进程已在运行，可稍后 resume")
                return
            next_check, paused = 0., False
            def stop():
                nonlocal next_check, paused
                if time.monotonic() >= next_check:
                    current = record(root, key)
                    paused = bool(current["paused"])
                    if "cycle" in current["plan"] and not paused:
                        from . import cycle_store
                        paused = not cycle_store.enabled(root,"pve")
                    next_check = time.monotonic()+.5
                return paused
            try:
                update(root, key, "running")
                state = drive(root, key, plan, stop)
            except Exception as error:
                update(root, key, "failed", str(error))
                raise
            update(root, key, state)


def add_arguments(parser):
    ops = parser.add_subparsers(dest="op", required=True)
    from . import practice_host
    practice_host.add_arguments(ops)
    for op in ("account", "probes", "freeze", "list", "start", "adapt", "enqueue", "watch", "status", "pause", "resume", "pull", "import", "export"):
        arg = ops.add_parser(op)
        arg.add_argument("--json", action="store_true")
        if op == "account":
            arg.add_argument("--username", required=True, help="采集账号学号")
        if op in ("start", "watch", "pull", "import", "export"):
            arg.add_argument("--target", default="boss", type=data.target_key, help="人机档位或 team:<队伍 ID>")
        if op in ("status", "pause", "resume", "adapt"):
            arg.add_argument("id")
        if op == "adapt":
            arg.add_argument("--specialist", help="固定常驻采集使用的终极人机专项快照")
            arg.add_argument("--holdout-manifest", type=Path, help="从选局统计中排除清单的 dev、blind 地图")
        if op in ("start", "watch", "resume"):
            arg.add_argument("--foreground", action="store_true")
        if op == "start":
            size = arg.add_mutually_exclusive_group()
            size.add_argument("--maps", type=int, default=16)
            size.add_argument("--continuous", action="store_true")
            arg.add_argument("--seed", type=int)
            arg.add_argument("--probes", nargs="+", choices=probes.catalog())
        if op == "enqueue":
            arg.add_argument("parent", help="持续采样任务 ID，换边完成后执行本批")
            arg.add_argument("--target", choices=online.PRACTICE_LEVELS, help="官网练习档位，默认沿用父任务")
            arg.add_argument("--strategy", dest="strategies", nargs=2, action="append", required=True,
                             metavar=("名称", "快照ID"))
            arg.add_argument("--maps", required=True, type=int)
            arg.add_argument("--repeats", type=int, default=1, help="每张图每种打法的换边组数")
            arg.add_argument("--seed", required=True, type=int)
        if op in ("start", "watch"):
            arg.add_argument("--interval", type=int, default=10 if op == "start" else 60, help="两轮之间的秒数")
        if op == "import":
            arg.add_argument("replay", type=Path)
        if op == "freeze":
            arg.add_argument("names", nargs="+", choices=probes.catalog())
        if op == "export":
            arg.add_argument("--run")
            arg.add_argument("--source", choices=("replay", "sdk"), default="replay")
            arg.add_argument("--split", choices=("train","development","blind"))
            arg.add_argument("--out", type=Path, help="输出样本清单")


def dispatch(root, args):
    from . import practice_host
    op = args.op
    if practice_host.handles(op):
        return practice_host.dispatch(root, args)
    if op == "account":
        return online.save_sampling_account(root, args.username, getpass.getpass("采集账号密码："))
    if op == "probes":
        return probes.catalog()
    if op == "freeze":
        return {name: {"version_id": version, "files": str(versions.snapshot_data(root, version)),
                       "entrypoint": versions.describe(root, version)["entrypoint"]}
                for name, version in probes.snapshots(root, args.names).items()}
    if op == "list":
        with data.database(root) as db:
            keys = [r[0] for r in db.execute("SELECT id FROM runs ORDER BY updated DESC LIMIT 30")]
        return [status(root, key) for key in keys]
    if op == "enqueue":
        return enqueue(root, args.parent, strategies=args.strategies, maps=args.maps, repeats=args.repeats,
                       seed=args.seed, target=args.target)
    if op == "adapt":
        return adapt(root, args.id, args.specialist, args.holdout_manifest)
    if op == "start":
        return start(root, target=args.target, maps=None if args.continuous else args.maps, names=args.probes,
                     seed=args.seed, interval=args.interval, foreground=args.foreground)
    if op == "watch":
        return start(root, target=args.target, mode="watch", interval=args.interval, foreground=args.foreground)
    if op == "pull":
        return data.pull(root, args.target)
    if op == "import":
        return data.ingest(root, args.replay.read_bytes(), args.target)
    if op == "export":
        export = data.sdk_dataset if args.source == "sdk" else data.dataset
        return export(root, args.target, run=args.run, split=args.split, output=args.out)
    if op == "status":
        return status(root, args.id)
    if op == "pause":
        return pause(root, args.id)
    return resume(root, args.id, args.foreground)


if __name__ == "__main__":
    worker = argparse.ArgumentParser()
    worker.add_argument("id")
    work(project_root(), worker.parse_args().id)
