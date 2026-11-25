import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import versions
from .workspace import project_root


def parser():
    top = argparse.ArgumentParser(prog="tank", description="种子杯策略工作台")
    groups = top.add_subparsers(dest="group", required=True)
    from . import checks
    checks.add_arguments(groups.add_parser("check", help="检查改动"))
    web = groups.add_parser("web", help="打开本地工作台")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--open", action="store_true")
    from . import jobs, lan
    jobs.add_arguments(groups.add_parser("jobs", help="后台训练"))
    lan.add_arguments(groups.add_parser("lan", help="局域网分片"))
    screens = groups.add_parser("screen", help="分阶段筛选并独立验收").add_subparsers(dest="op", required=True)
    screen = screens.add_parser("start")
    screen.add_argument("--candidates", required=True)
    screen.add_argument("--baseline")
    screen.add_argument("--stages", required=True, help="方案:晋级数，以逗号分隔")
    screen.add_argument("--shards", type=int, default=1)
    for name in ("status", "resume"):
        screens.add_parser(name).add_argument("screen_id")
    for name, sub in screens.choices.items():
        sub.add_argument("--json", action="store_true")
        if name != "status":
            sub.add_argument("--foreground", action="store_true")
    operations = {
        "snapshot": ("create", "list", "show", "compare", "restore", "main"),
        "suite": ("list", "show", "create"),
        "eval": ("estimate", "start", "list", "status", "compare", "cases", "cancel", "resume",
                 "retry", "share-retry", "reproduce", "replay", "logs", "export", "import"),
        "sync": ("publish", "restore", "list", "fetch"),
        "online": ("pack", "practice", "submit", "status", "matches", "refresh"),
        "project": ("list",),
        "training": ("collect", "list"),
    }
    for group, ops in operations.items():
        commands = groups.add_parser(group).add_subparsers(dest="op", required=True)
        for op in ops:
            cmd = commands.add_parser(op)
            cmd.add_argument("--json", action="store_true", help="机器可读的 JSON")
            if group == "snapshot":
                if op == "create":
                    cmd.add_argument("--strategy", required=True)
                    cmd.add_argument("--note", default="")
                elif op != "list":
                    cmd.add_argument("version_id")
                    if op == "compare":
                        cmd.add_argument("other_id")
                    if op == "restore":
                        cmd.add_argument("--name", required=True)
            elif group == "eval":
                if op in ("start", "estimate"):
                    candidates = cmd.add_mutually_exclusive_group(required=True)
                    candidates.add_argument("--candidates", help="逗号分隔的快照 ID")
                    candidates.add_argument("--strategy", action="append", help="自动保存策略目录并测试，可重复")
                    cmd.add_argument("--baseline")
                    cmd.add_argument("--suite", default="quick")
                    cmd.add_argument("--shards", type=int, default=1)
                    cmd.add_argument("--mode", choices=("opponent", "direct"), default="opponent")
                    cmd.add_argument("--replay", action="store_true")
                    if op == "start":
                        cmd.add_argument("--timeout", type=float, default=120)
                        cmd.add_argument("--foreground", action="store_true")
                        cmd.add_argument("--prepare-only", action="store_true", help="只建批次，交给工作机执行")
                        cmd.add_argument("--share", action="store_true", help="完成后自动发布到 Git 实验分支")
                elif op == "list":
                    cmd.add_argument("--version", help="按参与对局的快照查询批次")
                    cmd.add_argument("--limit", type=int, default=30)
                elif op in ("reproduce", "replay"):
                    cmd.add_argument("case_id")
                elif op == "import":
                    cmd.add_argument("path", type=Path)
                elif op != "list":
                    cmd.add_argument("run_id")
                    if op in ("export", "resume", "retry"):
                        cmd.add_argument("--shard", type=int)
                    if op == "export":
                        cmd.add_argument("path", type=Path)
                    if op in ("cases", "compare"):
                        cmd.add_argument("--opponent")
                        cmd.add_argument("--side", choices=("red","blue"))
                    if op == "cases":
                        cmd.add_argument("--candidate")
                        cmd.add_argument("--limit", type=int, default=20)
                        cmd.add_argument("--sort")
            elif group == "sync" and op == "list":
                cmd.add_argument("--remote", action="store_true", help="列出队友已推送的批次")
            elif group == "sync":
                cmd.add_argument("run_id")
                if op == "publish":
                    cmd.add_argument("--push", action="store_true", help="发布到 Git 远端实验分支")
            elif group == "suite" and op in ("show", "create"):
                cmd.add_argument("name")
                if op == "create":
                    cmd.add_argument("--opponent", action="append", required=True)
                    cmd.add_argument("--seeds", required=True, help="地图种子，以逗号分隔")
                    cmd.add_argument("--repeats", type=int, default=1)
            elif group == "online" and op not in ("matches", "refresh"):
                cmd.add_argument("identifier")
                if op == "submit":
                    cmd.add_argument("--version")
                    cmd.add_argument("--note", default="")
                if op == "practice":
                    from .online import PRACTICE_LEVELS
                    cmd.add_argument("--level", choices=PRACTICE_LEVELS, default="rookie")
                    cmd.add_argument("--games", type=int, choices=(1,2), default=1)
                    cmd.add_argument("--seed", type=int)
            elif group == "training" and op == "collect":
                cmd.add_argument("--red", required=True)
                cmd.add_argument("--blue", default="baseline")
                cmd.add_argument("--side", choices=("red","blue","both"), default="red")
                cmd.add_argument("--seed", type=int, default=42)
                cmd.add_argument("--game-no", type=int, default=0)
    return top


def execute(root, args):
    if args.group == "screen":
        from . import screening
        if args.op == "status":
            return screening.status(root, args.screen_id)
        if args.op == "resume":
            return screening.resume(root, args.screen_id, background=not args.foreground)
        stages = []
        for value in args.stages.split(","):
            name, count = value.split(":")
            stages.append((name, int(count)))
        return screening.screen(root, args.candidates.split(","), args.baseline or versions.main_version(root),
                                stages, shards=args.shards, background=not args.foreground)
    if args.group == "jobs":
        from . import jobs
        return jobs.dispatch(root, args)
    group, op = args.group, args.op
    if group == "project":
        from .catalog import projects
        return projects(root)
    if group == "snapshot":
        if op == "compare":
            return versions.compare(root, args.version_id, args.other_id)
        if op == "create":
            return versions.create(root, args.strategy, args.note)
        if op == "list":
            return versions.list_versions(root)
        if op == "restore":
            return versions.restore(root, args.version_id, args.name)
        if op == "main":
            return versions.set_main(root, args.version_id)
        return versions.describe(root, args.version_id)
    if group == "suite":
        from .eval_plan import SUITE_NAME, list_suites, get_suite, validate_suite
        if op == "create":
            if not SUITE_NAME.fullmatch(args.name):
                raise ValueError("方案名只能包含小写字母、数字、连字符和下划线")
            suite = validate_suite({"version": 1, "opponents": args.opponent,
                                    "seeds": [int(v) for v in args.seeds.split(",")],
                                    "repeats": args.repeats}, args.name)
            for opponent in suite["opponents"]:
                versions.describe(root, opponent)
            folder = root / "suites"
            folder.mkdir(exist_ok=True)
            with (folder / (args.name + ".json")).open("x", encoding="utf-8") as output:
                json.dump(suite, output, indent=2)
                output.write("\n")
            return get_suite(root, args.name)
        return list_suites(root) if op == "list" else get_suite(root, args.name)
    if group == "eval":
        from . import evaluation as ev
        if op in ("estimate", "start"):
            ids = ([versions.create(root, name)["id"] for name in args.strategy] if args.strategy
                   else args.candidates.split(","))
            values = (root, ids, args.baseline or versions.main_version(root), args.suite)
            opts = dict(shards=args.shards, mode=args.mode)
            if op == "estimate":
                return ev.estimate(*values, **opts, replay=args.replay)
            return ev.start(*values, **opts, timeout=args.timeout, replay=args.replay,
                            background=not args.foreground, prepare_only=args.prepare_only, share=args.share)
        if op == "list":
            return list_runs(root, args.limit, version=args.version)
        if op == "import":
            return ev.import_run(root, args.path)
        if op == "export":
            return ev.export_run(root, args.run_id, args.path, shard=args.shard)
        if op in ("reproduce","replay"):
            return getattr(ev, op)(root, args.case_id)
        if op in ("retry", "resume"):
            return getattr(ev, op)(root, args.run_id, shard=args.shard)
        if op == "share-retry":
            return ev.share_retry(root, args.run_id)
        if op == "compare":
            return ev.compare(root, args.run_id, opponent=args.opponent, side=args.side)
        if op == "cases":
            return ev.cases(root, args.run_id, opponent=args.opponent, side=args.side,
                            candidate=args.candidate, sort=args.sort, limit=args.limit)
        return getattr(ev, op)(root, args.run_id)
    if group == "sync":
        from . import sync
        if op == "list":
            return sync.list_remote(root) if args.remote else sync.list_published(root)
        if op == "publish" and args.push:
            return sync.push(root, args.run_id)
        return getattr(sync, op)(root, args.run_id)
    if group == "training":
        from . import training
        if op == "list":
            return training_overview(root)
        from uuid import uuid4
        match_id = uuid4().hex
        path = root / ".local/training/demonstrations" / (match_id + ".json")
        bots = [versions.load_bot(root, v) for v in (args.red, args.blue)]
        try:
            data = training.collect_demonstrations(*bots, args.seed, args.game_no, match_id=match_id,
                      sides=["red","blue"] if args.side=="both" else [args.side], output=path,
                      snapshots={"red": args.red, "blue": args.blue})
        finally:
            for bot in bots:
                if hasattr(bot, "close"):
                    bot.close()
        return {"match_id": match_id, "samples": len(data["samples"]), "path": str(path)}
    if group == "online":
        from . import online
        if op == "refresh":
            return online.refresh(root)
        if op == "status":
            return online.status(root, int(args.identifier))
        if op == "submit":
            return online.submit(root, args.identifier, version=args.version, note=args.note)
        if op == "practice":
            return online.practice(root, args.identifier, games=args.games, seed=args.seed, level=args.level)
        return getattr(online, op)(root, *([] if op == "matches" else [args.identifier]))
    raise ValueError(f"未知命令 {group}")


def list_runs(root, limit=30, *, version=None):
    from .eval_store import database
    if limit < 1:
        raise ValueError("limit 必须为正整数")
    where, params = "", []
    if version is not None:
        where = """WHERE json_extract(manifest, '$.baseline') = ?
                   OR EXISTS (SELECT 1 FROM json_each(manifest, '$.candidates') WHERE value = ?)
                   OR EXISTS (SELECT 1 FROM json_each(manifest, '$.suite.opponents') WHERE value = ?)"""
        params = [version] * 3
    with database(root) as db:
        rows = db.execute(f"""SELECT run_id, created_at, json_extract(manifest, '$.suite.name') AS suite
                              FROM runs {where} ORDER BY created_at DESC LIMIT ?""", [*params, limit])
        return [{"run_id": row["run_id"], "created_at": row["created_at"],
                 "suite": row["suite"]} for row in rows]


def training_overview(root):
    folder = root / ".local/training"
    return {key: [p.stem for p in sorted((folder/key).glob("*.json"))]
            for key in ("configs", "episodes", "checkpoints", "demonstrations")}


def main(argv=None):
    args = parser().parse_args(argv)
    if args.group == "check":
        from .checks import run
        return run(args)
    try:
        root = project_root()
        if args.group == "lan":
            from . import lan
            data = lan.dispatch(root, args)
            print(json.dumps(data, ensure_ascii=False))
            return 0
        if args.group == "web":
            from .webapp import serve
            serve(root, args.port, args.open)
            return 0
        data = execute(root, args)
        print(json.dumps(data, ensure_ascii=False, indent=None if args.json else 2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
