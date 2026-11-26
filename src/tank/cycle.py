"""Run independent ladder and practice training lines on the workbench host."""

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

from . import cycle_pve, cycle_release, online, practice_priority, sampling
from . import cycle_store as store
from .eval_store import shard_lock, worker_lock
from .workspace import project_root, read_json, write_json

ROUTER = "v-15e8911654f47c9b997e691e2e8ac68c5ae3db466a06d89503f89098bb7e4ca3"


def _spawn(root, module, args, log_file):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    python = (
        os.path.join(os.path.dirname(sys.executable), "pythonw.exe") if os.name == "nt" else sys.executable
    )
    env = os.environ | {
        "PYTHONUTF8": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with log_file.open("ab") as log:
        return subprocess.Popen(
            [python, "-B", "-m", module, *args],
            cwd=root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            creationflags=flags,
            start_new_session=os.name != "nt",
        ).pid


def _launch(root, line):
    path = store.folder(root) / line
    with worker_lock(path, "launch") as acquired:
        if not acquired:
            return
        with shard_lock(path, "service") as free:
            if not free:
                return
        store.put(root, line, launch_at=time.time())
        _spawn(root, "tank.cycle", [line], path / "worker.log")


def resume_enabled(root):
    if not (store.folder(root) / "control.sqlite3").is_file():
        return
    from . import practice_host

    if practice_host.is_remote(root):
        return
    for line in store.LINES:
        if store.enabled(root, line):
            _launch(root, line)


def _signature(root):
    digest = hashlib.sha256()
    for path in sorted((root / "training").glob("cycle_*.py")):
        digest.update(path.read_bytes())
    pool = store.folder(root) / "pvp/opponents.json"
    if pool.exists():
        digest.update(pool.read_bytes())
    return digest.hexdigest()


def _update_opponents(root, previous):
    from training.cycle_train import FROZEN_INPUTS, _pvp_pool

    _pvp_pool(root)
    path = store.folder(root) / "pvp/opponents.json"
    with worker_lock(path.parent, "opponents") as acquired:
        if not acquired:
            raise RuntimeError("对手池正在更新")
        pool = read_json(path)
        pilots = read_json(root / FROZEN_INPUTS)["policies"]
        additions = {"basic-specialist-v7": pilots["A"], "boss-specialist-v7": pilots["B"]}
        if previous:
            additions["former-" + previous[2:14]] = previous
        for row in store.history(root, "pvp", -1):
            if row["outcome"] == "promoted":
                additions["past-main-" + str(row["round"])] = row["incumbent"]
        updated = pool | {k: v for k, v in additions.items() if v not in pool.values()}
        if updated != pool:
            write_json(path, updated)


def _idle_seconds():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class LastInput(ctypes.Structure):
        _fields_ = [("size", wintypes.UINT), ("tick", wintypes.DWORD)]

    last = LastInput()
    last.size = ctypes.sizeof(last)
    if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(last)):
        return ((ctypes.windll.kernel32.GetTickCount() - last.tick) & 0xFFFFFFFF) / 1000
    return None


def _foreground(root):
    team = (
        read_json(root / ".local/online/team-scores.json")
        if (root / ".local/online/team-scores.json").exists()
        else {}
    )
    idle = _idle_seconds()
    return (idle is not None and idle < 60) or (
        team.get("team_id") is not None and practice_priority.foreground_pending(root, team["team_id"])
    )


def _budget(root):
    profile = store.folder(root) / "profile.json"
    total = read_json(profile)["workers"] if profile.exists() else 4
    interactive = _foreground(root)
    with sampling.data.database(root) as db:
        recent = [
            json.loads(r[0]).get("sdk_timing")
            for r in db.execute("SELECT entry FROM sdk_observations ORDER BY id DESC LIMIT 10")
        ]
    ratios = [x["reused_actions"] / x["confirmed_actions"] for x in recent if x and x["confirmed_actions"]]
    delayed = len(ratios) >= 6 and sum(ratios[:2]) / 2 > sum(ratios[2:]) / len(ratios[2:]) + 0.02
    if interactive or delayed:
        total = min(4, max(1, total // 2))
    active = 0
    for line in store.LINES:
        if store.enabled(root, line):
            out = store.round_dir(root, line, store.get(root, line, "number", 1))
            with shard_lock(out, "fit") as available:
                active += int(not available or not (out / "result.json").is_file())
    return max(1, total // max(1, active))


def _round(root, line, live, number, recipient):
    out = store.round_dir(root, line, number)
    result_file = out / "result.json"
    request_file = out / "request.json"
    if not request_file.exists():
        write_json(
            request_file,
            {
                "line": line,
                "round": number,
                "incumbent": live,
                "recipient": recipient,
                "signature": _signature(root),
                "started_at": time.time(),
            },
        )
    request = read_json(request_file)
    verify_only = False
    if result_file.is_file():
        result = read_json(result_file)
        if any(result[k] != request[k] for k in ("incumbent", "recipient", "line", "round")):
            raise ValueError("训练结果与该轮登记的线路、轮次或快照不同")
        from training.cycle_train import verify_evidence

        try:
            verify_evidence(root, out, result)
        except ValueError:
            verify_only = True
            with shard_lock(store.folder(root), "profile") as free:
                if not free:
                    store.put(root, line, phase="等待吞吐测试完成后复核")
                    return None
        else:
            failure = out / "failure.json"
            if failure.exists():
                failure.replace(out / f"recovered-{failure.stat().st_mtime_ns}.json")
            store.resume_completed_result(root, line, number)
            return result | {"round": number, "signature": request["signature"]}
    budget = _budget(root)
    workers_file = out / "workers.json"
    if not workers_file.exists() or read_json(workers_file)["workers"] != budget:
        write_json(workers_file, {"workers": budget})
    failure = out / "failure.json"
    if failure.is_file():
        if not verify_only and request["signature"] != _signature(root):
            stale = {
                "round": number,
                "incumbent": request["incumbent"],
                "candidate": None,
                "signature": request["signature"],
            }
            _finish(
                root,
                line,
                stale,
                "superseded",
                "训练程序已更新，使用新轮次",
                error=read_json(failure)["error"],
            )
            return None
        raise ValueError(read_json(failure)["error"])
    from training.cycle_train import _round_lock

    try:
        with _round_lock(out):
            pass
    except RuntimeError as error:
        if "本轮训练已在运行" in str(error):
            return None
        raise
    with shard_lock(out, "fit") as free:
        if not free:
            return None
    _spawn(
        root,
        "tank.cycle_fit",
        [line, str(number), *(["verify"] if verify_only else [])],
        out / "training.log",
    )
    if verify_only:
        store.put(root, line, phase="复核本地配对成绩")
    return None


def _finish(root, line, result, outcome, reason, *, changes=None, **extra):
    metrics = result.get("metrics", {})
    comparison = metrics.get("paired", {})
    transfer = result.get("cross_line", {})
    return store.finish(
        root,
        line,
        result,
        outcome,
        reason,
        signature=result["signature"],
        changes=changes,
        local={k: comparison[k] for k in ("mean_delta", "ci95") if k in comparison},
        transfer={
            k: transfer[k] for k in ("ability", "status", "utility_delta", "recipient_line") if k in transfer
        },
        **extra,
    )


def _process_result(root, line, result, live, submission):
    if not store.enabled(root, line):
        return
    candidate, base = result["candidate"], result["incumbent"]
    if not result["accepted"]:
        return _finish(root, line, result, "rejected", "本地对照未达标")
    if live not in (base, candidate):
        return _finish(
            root,
            line,
            result,
            "superseded",
            "主力已更换，重新配对",
            changes={"selection_target": None},
        )
    if line == "pve":
        parent = store.get(root, line, "sampler")
        gate = cycle_pve.gate(
            root, parent, candidate, base, result["round"], allow_enqueue=lambda: store.enabled(root, line)
        )
        if gate["status"] == "paused":
            return
        store.put(root, line, gate=gate)
        if gate["status"] == "collecting":
            store.put(root, line, phase="官网同图换边")
            return
        if gate["status"] == "rejected":
            reason = "终极档增分不稳定，结束复测" if "boss_ci95_per_game" in gate else "官网四档配对增分未达标"
            return _finish(root, line, result, "rejected", reason, practice=gate)
    if not store.enabled(root, line):
        return
    official = cycle_release.official_score(root, candidate, line=line)
    if official["state"] == "failed":
        return _finish(
            root,
            line,
            result,
            "rejected",
            "官网评测失败",
            official=official,
            **({"practice": gate} if line == "pve" else {}),
        )
    if official["state"] != "completed":
        store.put(root, line, phase="等待正式 40 局", official=official)
        return
    score = official["score"]
    if line == "pve":
        high = store.get(root, line, "score")
        if high is not None and score <= high:
            return _finish(
                root,
                line,
                result,
                "rejected",
                "正式 40 局未超过主力",
                official=official,
                practice=gate,
            )
        return _finish(
            root,
            line,
            result,
            "promoted",
            "刷新人机主力",
            official=official,
            practice=gate,
            changes={"incumbent": candidate, "score": score},
        )
    if score < 4333:
        return _finish(root, line, result, "rejected", "正式人机分未达 4333", official=official)
    from . import cycle_ladder

    out = store.round_dir(root, line, result["round"])
    before = out / "activation.json"
    if not before.exists():
        if live != base:
            return _finish(root, line, result, "superseded", "生效策略由外部更换")
        write_json(
            before,
            {
                "previous": base,
                "previous_submission": submission,
                "current": candidate,
                "since": time.time(),
            },
        )
    store.put(root, line, selection_target=official["submission"])
    actual = cycle_ladder.request_activation(
        root,
        candidate,
        official["submission"],
        allow_activation=lambda: store.enabled(root, line),
        expected_active_version=base,
    )
    if actual["active_version"] != candidate:
        status = actual["activation"]["status"]
        if status in (
            "rejected",
            "series_not_activated",
            "changed",
            "active_changed",
            "cancelled",
            "superseded",
        ) or (
            status in ("series_void", "series_cancelled", "series_expired")
            and not actual["activation"]["can_retry"]
        ):
            return _finish(
                root,
                line,
                result,
                "superseded" if status in ("active_changed", "superseded", "cancelled") else "rejected",
                "官网换版未生效",
                activation=actual["activation"],
                changes={"selection_target": None},
            )
        store.put(root, line, phase="等待天梯换版结算")
        return
    return _finish(
        root,
        line,
        result,
        "promoted",
        "新版天梯策略已生效",
        official=official,
        changes={
            "review": read_json(before),
            "incumbent": candidate,
            "score": score,
            "selection_target": None,
        },
    )


def _experiment(root, line, incumbent, submission, recipient):
    history = store.history(root, line)
    if line == "pvp" and store.enabled(root, line):
        previous = store.get(root, line, "opponents_incumbent")
        if previous != incumbent:
            _update_opponents(root, previous)
            store.put(root, line, opponents_incumbent=incumbent)
    number = history[0]["round"] + 1 if history else 1
    if history and history[0]["outcome"] == "superseded" and history[0]["candidate"] is None:
        prior = history[0]["round"]
        if (store.round_dir(root, line, prior) / "result.json").is_file():
            number = prior
    signature = _signature(root)
    stalled = len(history) == 3 and all(
        r["outcome"] == "rejected" and r["incumbent"] == incumbent and r.get("signature") == signature
        for r in history
    )
    if stalled:
        store.put(root, line, phase="连续三轮未提升，等待新打法")
        return
    profile_path = store.folder(root) / "profile.json"
    progress = profile_path.with_name("profile-progress.json")
    recently_paused = (
        progress.exists()
        and time.time() - progress.stat().st_mtime < 120
        and read_json(progress).get("status") == "paused"
    )
    if not profile_path.exists() and number > 1 and not _foreground(root) and not recently_paused:
        others = [k for k in store.LINES if store.enabled(root, k)]
        if all((store.round_dir(root, k, 1) / "result.json").is_file() for k in others):
            with worker_lock(store.folder(root), "profile-launch") as acquired:
                if acquired:
                    with shard_lock(store.folder(root), "profile") as free:
                        running = not free
                    if not running and time.time() - store.get(root, "shared", "profile_launch", 0) > 10:
                        store.put(root, "shared", profile_launch=time.time())
                        _spawn(root, "tank.cycle", ["profile"], store.folder(root) / "profile.log")
            store.put(root, line, phase="测量训练吞吐")
            return
    store.put(root, line, number=number, phase="本地配对训练")
    result = _round(root, line, incumbent, number, recipient)
    if result is not None:
        store.put(root, line, latest_candidate=result["candidate"])
        _process_result(root, line, result, incumbent, submission)


def tick(root, line):
    if line == "pvp":
        from . import cycle_ladder

        live = cycle_ladder.tick(
            root,
            allow_challenge=lambda: store.enabled(root, line),
            pending_only=store.get(root, line, "selection_target"),
        )
        version = live["active_version"]
        public = {
            k: live.get(k)
            for k in (
                "team_id",
                "rank",
                "action",
                "series_seen",
                "pending_submission",
                "active_submission",
                "active_version",
                "archived_games",
                "replay_errors",
            )
        }
        known = [read_json(f) for f in (store.folder(root) / "ladder/series").glob("*.json")]
        outcomes = [
            r["outcome"]
            for r in known
            if version
            and r["version_id"] == version
            and r["team_id"] == live["team_id"]
            and r["ranked"]
            and r["status"] == "finished"
        ]
        record = {"wins": outcomes.count(1), "losses": outcomes.count(-1), "draws": outcomes.count(0)}
        store.put(
            root,
            line,
            ladder=public,
            incumbent=version,
            score=live.get("active_score") if version else None,
            record=record if version else None,
        )
        if not store.enabled(root, line):
            return
        if not version:
            store.put(root, line, phase="等待核实官网生效快照")
            return
        message = cycle_release.review_activation(root, live)
        if message:
            store.put(root, line, phase=message)
        else:
            _experiment(
                root,
                line,
                version,
                live["active_submission"],
                store.get(root, "pve", "incumbent") or ROUTER,
            )
    else:
        version = store.get(root, line, "incumbent")
        if version is None:
            records = online.refresh(root)["submissions"]
            best = max(
                (r for r in records if r["version_id"]),
                key=lambda r: r["score"],
                default=None,
            )
            version = best["version_id"] if best else ROUTER
            store.put(root, line, incumbent=version, score=best["score"] if best else None)
        policy = read_json(root / "training/experiments/style-switch-v1/pilot.json")["policies"]
        with worker_lock(store.folder(root) / line, "sampler-control") as acquired:
            if not acquired or not store.enabled(root, line):
                return
            parent = cycle_pve.ensure_sampler(
                root,
                version,
                store.get(root, line, "latest_candidate") or policy["B"],
                policy["A"],
                store.get(root, line, "sampler"),
            )
            store.put(root, line, sampler=parent)
        recipient = store.get(root, "pvp", "incumbent") or ROUTER
        if store.enabled(root, line):
            _experiment(root, line, version, None, recipient)


def _experience(root):
    from . import cycle_experience

    with shard_lock(store.folder(root), "experience-refresh") as free:
        if free and time.time() - store.get(root, "shared", "experience_at", 0) > 45:
            info = cycle_experience.refresh(root, limit=64)
            store.put(root, "shared", experience_at=time.time(), experience=info)


def service(root, line):
    loaded = _service_revision(root)
    with shard_lock(store.folder(root) / line, "service") as acquired:
        if not acquired:
            return
        while store.enabled(root, line) and _service_revision(root) == loaded:
            try:
                tick(root, line)
                _experience(root)
                store.put(root, line, error=None, updated=time.time())
            except Exception as error:  # noqa: BLE001 - 记录该线的故障，另一条线独立运行。
                traceback.print_exc()
                store.put(
                    root,
                    line,
                    error=f"{type(error).__name__}: {error}",
                    updated=time.time(),
                )
            for _ in range(15):
                if not store.enabled(root, line):
                    break
                time.sleep(1)
    if store.enabled(root, line):
        _launch(root, line)


def _service_revision(root):
    files = [
        *root.glob("src/tank/cycle*.py"),
        *root.glob("training/cycle*.py"),
        root / "src/tank/sampling.py",
    ]
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(files))


def line_status(root, line):
    with store.database(root) as db:
        flags = db.execute("SELECT enabled FROM lines WHERE line=?", (line,)).fetchone()
        rows = db.execute("SELECT name,value FROM state WHERE line=?", (line,)).fetchall()
    with shard_lock(store.folder(root) / line, "service") as free:
        values = {r[0]: json.loads(r[1]) for r in rows}
        if flags and flags[0]:
            status = (
                "running"
                if not free
                else ("starting" if time.time() - values.get("launch_at", 0) < 15 else "interrupted")
            )
        else:
            status = "paused" if flags else "stopped"
    return {"line": line, "status": status, **values}


def dashboard(root):
    if not (store.folder(root) / "control.sqlite3").is_file():
        return []
    items = []
    for line in store.LINES:
        s = line_status(root, line)
        items.append(
            {
                k: s.get(k)
                for k in (
                    "line",
                    "status",
                    "phase",
                    "incumbent",
                    "score",
                    "last_change",
                    "error",
                    "record",
                )
            }
        )
    return items


def add_arguments(parser):
    commands = parser.add_subparsers(dest="op", required=True)
    reconcile = commands.add_parser("reconcile", help="核对未确认的天梯请求")
    reconcile.add_argument("--clear", choices=("challenge", "activation"), help="核对后解除指定的未确认请求")
    reconcile.add_argument("--json", action="store_true")
    for op in ("start", "status", "pause", "resume", "report"):
        sub = commands.add_parser(op)
        sub.add_argument("line", choices=(*store.LINES, "all"), nargs="?", default="all")
        sub.add_argument("--json", action="store_true")
        if op == "report":
            sub.add_argument("--since", type=float, default=0, help="上次报告的 Unix 时间")


def dispatch(root, args):
    if args.op == "reconcile":
        from . import cycle_ladder

        return cycle_ladder.reconcile(root, clear=args.clear)
    lines = store.LINES if args.line == "all" else (args.line,)
    if args.op in ("start", "resume", "pause"):
        active = args.op != "pause"
        if active:
            from . import practice_host

            if practice_host.is_remote(root):
                raise ValueError("请在共享练习主机启动或恢复自动训练")
        with store.database(root) as db:
            db.executemany(
                "INSERT OR REPLACE INTO lines VALUES(?,?)",
                [(k, int(active)) for k in lines],
            )
        for line in lines:
            if active:
                if args.op == "resume":
                    number = store.get(root, line, "number", 1)
                    (store.round_dir(root, line, number) / "failure.json").unlink(missing_ok=True)
                    if line == "pve":
                        gate = store.get(root, line, "gate") or {}
                        for batch in gate.get("batches", {}).values():
                            if sampling.status(root, batch)["state"] in ("failed", "paused", "interrupted"):
                                sampling.resume(root, batch, False)
                        parent = store.get(root, line, "sampler")
                        if parent and sampling.status(root, parent)["state"] == "failed":
                            sampling.resume(root, parent, False)
                _launch(root, line)
            elif line == "pve":
                with worker_lock(store.folder(root) / line, "sampler-control") as acquired:
                    if not acquired:
                        raise ValueError("采样器正在交接，稍后重试暂停")
                    key = store.get(root, line, "sampler")
                    if key and not store.enabled(root, line):
                        sampling.pause(root, key)
    data = {line: line_status(root, line) for line in lines}
    if args.op == "report":
        from . import cycle_experience

        as_of = min(store.report_watermark(root), cycle_experience.report_watermark(root))
        for line in lines:
            evidence = cycle_experience.summary(root, line, since=args.since, until=as_of)
            data[line]["history"] = store.history(root, line, -1 if args.since else 12, args.since, as_of)
            data[line]["evidence"] = {
                k: evidence.get(k)
                for k in (
                    "cursor",
                    "games",
                    "maps",
                    "event_counts",
                    "previously_tried_changes",
                )
            }
            data[line]["evidence"]["examples"] = evidence.get("top_findings", [])
            if line == "pvp":
                data[line]["evidence"]["opponent_splits"] = evidence["pvp_splits"]
            data[line]["evidence"]["needs"] = evidence["sampling_needs"][:3]
        data["as_of"] = as_of
    return data


if __name__ == "__main__":
    root = project_root()
    if sys.argv[1] == "profile":
        from training.cycle_train import profile

        if not (store.folder(root) / "profile.json").is_file():
            profile(root)
    else:
        service(root, sys.argv[1])
