"""Resumable online practice campaigns and replay collection CLI."""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from . import online, versions
from . import sampling_data as data
from . import sampling_probes as probes
from .eval_store import shard_lock
from .workspace import project_root, read_json

ROTATION = (("approach", "silent", "pulse"), ("approach", "retreat", "peek"),
            ("peek", "late-zone", "lancer"), ("retreat", "late-zone", "silent"))
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
    name = names[pos]
    return (plan["seed"] + index * 104729) % 2**31, name, plan["versions"][name]


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
        games = db.execute("SELECT COUNT(*) FROM slots WHERE run=?", (key,)).fetchone()[0]
    result = {"id": key, "mode": plan["mode"], "state": state, "target": plan["target"], "message": row["message"]}
    if plan["mode"] == "active":
        result.update(games=games, planned_games=plan["maps"] * plan["width"] * 2 if plan["maps"] else None)
        if state != "completed":
            seed, name, _ = case(plan, row["step"])
            result.update(map_seed=seed, probe=name)
    return result


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
    if not practice_host.host_running(root):
        raise ValueError("请先启动共享练习主机 tank sample host-serve，供队友排队")


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
        snaps = probes.snapshots(root, names or probes.catalog().keys())
        plan.update(seed=seed, maps=maps, probes=names, rotation=ROTATION if not names else None,
                    versions=snaps, width=len(names) if names else len(ROTATION[0]))
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
        db.execute("""UPDATE runs SET paused=1,state='pausing',updated=?
                      WHERE id=? AND state NOT IN ('completed','failed')""", (time.time(), key))
    return status(root, key)


def resume(root, key, foreground):
    current = status(root, key)
    if current["state"] in ("queued", "running", "waiting", "pausing", "completed"):
        return current
    if current["mode"] == "active":
        check_host(root)
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
    seed, name, version = case(plan, step)
    with data.database(root) as db:
        seen = {r[0] for r in db.execute("SELECT side FROM slots WHERE run=? AND step=?", (key, step))}
    for r in sorted(records, key=lambda item: item["id"]):
        side = r["side"]
        if (r["case_tag"] != f"{key}-{step}" or r["version_id"] != version or r["level"] != plan["target"]
                or r["map_seed"] != seed or r["team_id"] != client.user["id"] or side not in ("red","blue")):
            raise ValueError(f"采样格与官网练习记录不符 {r['id']}")
        if side in seen:
            continue
        path = online.practice_replay_path(root, r["id"])
        if not path.is_file():
            online.replay(root, r["id"], client=client)
        data.ingest(root, path.read_bytes(), plan["target"], match_id=r["id"], practice=r, probe=name)
        with data.database(root) as db:
            db.execute("INSERT OR IGNORE INTO slots VALUES (?,?,?,?)", (key, step, side, r["id"]))
        seen.add(side)
    return len(seen) == 2


def drive(root, key, plan, stop):
    client = None
    step = record(root, key)["step"]
    recover = True
    failures = 0
    while not stop():
        if plan["mode"] == "active" and plan["maps"] is not None and step >= plan["maps"] * plan["width"]:
            return "completed"
        try:
            if client is None:
                client = online.OnlineClient.login(root)
            if plan["mode"] == "watch":
                result = data.pull(root, plan["target"], client=client)
                message = f"本轮新增 {result['added']} 场，{len(result['pending'])} 场等待回放"
                delay = plan["interval"]
            else:
                seed, _, version = case(plan, step)
                existing = receipts(root, key, step, version, plan["target"], recover=recover)
                recover = False
                if not save_cases(root, key, step, plan, existing.values(), client):
                    update(root, key, "running")
                    batch = online.collect_practice(root, version, client=client, games=2, seed=seed,
                                                    level=plan["target"], stop=stop, case_tag=f"{key}-{step}")
                    if not save_cases(root, key, step, plan, batch["matches"], client):
                        raise ValueError("官网未返回完整换边记录")
                step += 1
                with data.database(root) as db:
                    db.execute("UPDATE runs SET step=?,updated=? WHERE id=?", (step, time.time(), key))
                if plan["maps"] is not None and step == plan["maps"] * plan["width"]:
                    return "completed"
                message, delay = "等待下一组换边", plan["interval"]
            failures = 0
        except online.PracticeDeferred as error:
            ready = [r for r in error.matches if online.practice_replay_path(root, r["id"]).is_file()]
            if ready:
                save_cases(root, key, step, plan, ready, client)
            message = str(error)
            delay = max(5, min(plan["interval"], 30))
        except (online.RequestUncertain, online.RemoteError) as error:
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
            from . import practice_host
            next_check, paused, host_stopped = 0., False, False
            def stop():
                nonlocal next_check, paused, host_stopped
                if time.monotonic() >= next_check:
                    paused = bool(record(root, key)["paused"])
                    if plan["mode"] == "active" and not paused:
                        host_stopped = not practice_host.host_running(root)
                    next_check = time.monotonic()+.5
                return paused or host_stopped
            try:
                update(root, key, "running")
                state = drive(root, key, plan, stop)
            except Exception as error:
                update(root, key, "failed", str(error))
                raise
            update(root, key, state, "共享练习主机已停止，启动主机后可 resume" if host_stopped else "")


def add_arguments(parser):
    ops = parser.add_subparsers(dest="op", required=True)
    from . import practice_host
    practice_host.add_arguments(ops)
    for op in ("probes", "freeze", "list", "start", "watch", "status", "pause", "resume", "pull", "import", "export"):
        arg = ops.add_parser(op)
        arg.add_argument("--json", action="store_true")
        if op in ("start", "watch", "pull", "import", "export"):
            arg.add_argument("--target", default="boss", type=data.target_key, help="人机档位或 team:<队伍 ID>")
        if op in ("status", "pause", "resume"):
            arg.add_argument("id")
        if op in ("start", "watch", "resume"):
            arg.add_argument("--foreground", action="store_true")
        if op == "start":
            size = arg.add_mutually_exclusive_group()
            size.add_argument("--maps", type=int, default=16)
            size.add_argument("--continuous", action="store_true")
            arg.add_argument("--seed", type=int)
            arg.add_argument("--probes", nargs="+", choices=probes.catalog())
        if op in ("start", "watch"):
            arg.add_argument("--interval", type=int, default=10 if op == "start" else 60, help="两轮之间的秒数")
        if op == "import":
            arg.add_argument("replay", type=Path)
        if op == "freeze":
            arg.add_argument("names", nargs="+", choices=probes.catalog())
        if op == "export":
            arg.add_argument("--run")
            arg.add_argument("--split", choices=("train","development","blind"))
            arg.add_argument("--out", type=Path, help="输出含快照、目标一侧和回放路径的训练清单")


def dispatch(root, args):
    from . import practice_host
    op = args.op
    if practice_host.handles(op):
        return practice_host.dispatch(root, args)
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
        return data.dataset(root, args.target, run=args.run, split=args.split, output=args.out)
    if op == "status":
        return status(root, args.id)
    if op == "pause":
        return pause(root, args.id)
    return resume(root, args.id, args.foreground)


if __name__ == "__main__":
    worker = argparse.ArgumentParser()
    worker.add_argument("id")
    work(project_root(), worker.parse_args().id)
