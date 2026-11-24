"""Create, execute, compare and transfer evaluation batches."""

from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from uuid import uuid4

from . import versions
from .eval_plan import (
    SCORE_ITEMS, engine_fields, freeze_runner, get_suite, json_text, list_suites, make_manifest, match_specs, pairings,
    required, shard_for, validate_manifest, version_ids,
)
from .eval_runner import MatchFailure, run_isolated
from .eval_store import (
    add_manifest, cached_matches, database, iter_matches, leased_shards, local_dir, queued_shards, read_manifest,
    save_match, shard_lock, worker_lock,
)
from .workspace import write_json

SIDES = ("red", "blue")
CASE_ID = re.compile(r"case-[0-9a-f]{64}")


def _run_folder(root, run_id):
    return local_dir(root) / run_id


def _plan(root, candidates, baseline, suite, shards, mode, timeout, replay):
    if baseline is None:
        baseline = versions.main_version(root)
    manifest = make_manifest(root, "run-" + uuid4().hex, candidates, baseline, suite, shards, mode, timeout, replay)
    for version_id in sorted(version_ids(manifest)):
        versions.describe(root, version_id)
    return manifest


def _actual_matches(manifest, records):
    return {
        case_id: item for case_id, item in records.items()
        if item["has_replay"] or not manifest["replay"]
    }


def _failure_rows(db, run_id):
    rows = db.execute("SELECT case_id, kind, message FROM failures WHERE run_id = ?", (run_id,))
    return {row["case_id"]: dict(row) for row in rows}


def _by_shard(manifest, specs, shard):
    if shard is None:
        return specs
    if type(shard) is not int or not 0 <= shard < manifest["shards"]:
        raise ValueError(f"shard 须为 0 到 {manifest['shards'] - 1} 的整数")
    return {
        case_id: spec for case_id, spec in specs.items()
        if shard_for(spec, manifest["shards"]) == shard
    }


def _forecast(manifest, specs, done):
    if not specs or len(specs) == len(done):
        return 0.0
    if not done:
        return None
    elapsed = fmean(row["elapsed"] for row in done.values())
    counts = defaultdict(int)
    for key in specs.keys() - done.keys():
        counts[shard_for(specs[key], manifest["shards"])] += 1
    return round(elapsed * max(counts.values()), 2)


def estimate(root, candidates, baseline, suite, *, shards=1, mode="opponent", replay=False):
    manifest = _plan(root, candidates, baseline, suite, shards, mode, 120, replay)
    specs = match_specs(manifest)
    with database(root) as db:
        cached = _actual_matches(manifest, cached_matches(db, specs, summary=True))
        history = db.execute(
            "SELECT AVG(elapsed) FROM matches WHERE json_extract(spec,'$.sdk_sha256') = ?"
            " AND json_extract(spec,'$.runner_sha256') = ? AND json_extract(spec,'$.cpp_sha256') IS ?",
            (manifest["engine"]["sdk_sha256"], manifest["engine"]["runner_sha256"], manifest["engine"].get("cpp_sha256")),
        ).fetchone()[0]
    counts = defaultdict(int)
    for key in specs.keys() - cached.keys():
        counts[shard_for(specs[key], shards)] += 1
    remaining = max(counts.values(), default=0)
    eta = round(remaining * history, 2) if remaining and history is not None else (None if remaining else 0.0)
    return {
        "suite": manifest["suite"], "baseline": manifest["baseline"], "candidates": manifest["candidates"],
        "total": len(specs), "cached": len(cached), "new": len(specs) - len(cached), "eta_seconds": eta,
        "shards": [{"shard": shard, "pending": count} for shard, count in sorted(counts.items())],
    }


def _verify_inputs(root, manifest):
    freeze_runner(root, manifest["engine"])
    for version_id in sorted(version_ids(manifest) - {"baseline"}):
        versions.verify(root, version_id)


def start(
    root, candidates, baseline, suite, *, shards=1, mode="opponent", timeout=120, replay=False,
    background=True, prepare_only=False,
):
    manifest = _plan(root, candidates, baseline, suite, shards, mode, timeout, replay)
    _verify_inputs(root, manifest)
    with database(root) as db:
        add_manifest(db, manifest)
    if prepare_only:
        return status(root, manifest["run_id"])
    return _launch(root, manifest, shard=None, retry_failures=False, background=background)


def _running_shards(root, manifest, specs):
    folder = _run_folder(root, manifest["run_id"])
    shards = sorted({shard_for(spec, manifest["shards"]) for spec in specs.values()})
    active = []
    for shard in shards:
        with shard_lock(folder, shard) as acquired:
            if not acquired:
                active.append(shard)
    return active


def status(root, run_id):
    with database(root) as db:
        manifest, run = read_manifest(db, run_id)
        specs = match_specs(manifest)
        done = _actual_matches(manifest, cached_matches(db, specs, summary=True))
        errors = {key: row for key, row in _failure_rows(db, run_id).items() if key not in done}
        lan = leased_shards(db, run_id)
        queued = queued_shards(db, run_id)
    active = sorted(set(_running_shards(root, manifest, specs)) | lan)
    if len(done) == len(specs):
        state = "completed"
    elif run["cancelled"]:
        state = "cancelling" if active else "cancelled"
    elif active:
        state = "running"
    elif queued:
        state = "queued"
    else:
        state = "incomplete" if run["attempted"] else "pending"
    return {
        "run_id": run_id, "status": state, "suite": manifest["suite"],
        "candidates": manifest["candidates"], "baseline": manifest["baseline"], "mode": manifest["mode"],
        "replay": manifest["replay"], "timeout": manifest["timeout"],
        "total": len(specs), "completed": len(done), "pending": len(specs) - len(done) - len(errors),
        "failed": len(errors), "timed_out": sum(row["kind"] == "timeout" for row in errors.values()),
        "eta_seconds": _forecast(manifest, specs, done), "shards": manifest["shards"], "active_shards": active,
        "lan_shards": sorted(lan),
    }


def _launch(root, manifest, *, shard, retry_failures, background):
    run_id = manifest["run_id"]
    selected = _by_shard(manifest, match_specs(manifest), shard)
    if _running_shards(root, manifest, selected):
        raise ValueError(f"批次 {run_id} 的指定分片正在运行")
    with database(root) as db:
        db.execute("BEGIN IMMEDIATE")
        done = _actual_matches(manifest, cached_matches(db, selected, summary=True))
        errors = _failure_rows(db, run_id)
        pending = selected.keys() - done.keys()
        pending = pending & errors.keys() if retry_failures else pending - errors.keys()
        shards = sorted({shard_for(selected[key], manifest["shards"]) for key in pending})
        if leased_shards(db, run_id).intersection(shards):
            raise ValueError(f"批次 {run_id} 有已由局域网工作机领取的分片")
        if queued_shards(db, run_id).intersection(shards):
            raise ValueError(f"批次 {run_id} 的指定分片已排队")
        db.execute("UPDATE runs SET attempted = 1, cancelled = 0 WHERE run_id = ?", (run_id,))
        if background:
            launch_ids = {index: uuid4().hex for index in shards}
            db.executemany(
                "INSERT OR REPLACE INTO queued_shards VALUES (?, ?, ?, ?)",
                [(run_id, index, launch_id, time.time()) for index, launch_id in launch_ids.items()],
            )
    if background:
        for index, launch_id in launch_ids.items():
            try:
                _spawn(root, run_id, index, retry_failures, launch_id)
            except OSError:
                with database(root) as db:
                    db.executemany(
                        "DELETE FROM queued_shards WHERE run_id = ? AND shard = ? AND launch_id = ?",
                        [(run_id, shard, queued_id) for shard, queued_id in launch_ids.items() if shard >= index],
                    )
                raise
    elif len(shards) == 1:
        _work(root, run_id, shards[0], retry_failures)
    elif shards:
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            work = [pool.submit(_work, root, run_id, index, retry_failures) for index in shards]
            try:
                for job in work:
                    job.result()
            except KeyboardInterrupt:
                cancel(root, run_id)
                raise
    return status(root, run_id)


def _log_path(root, run_id, shard):
    folder = _run_folder(root, run_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"shard-{shard}.log"


def _spawn(root, run_id, shard, retry_failures, launch_id):
    flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with _log_path(root, run_id, shard).open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "tank.evaluation", str(root), run_id, str(shard),
             "retry" if retry_failures else "resume", launch_id],
            cwd=root, env=os.environ | {"PYTHONUTF8": "1"},
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=flags, start_new_session=os.name != "nt",
        )


def _work(root, run_id, shard, retry_failures=False, launch_id=None):
    with worker_lock(_run_folder(root, run_id), shard) as locked:
        if not locked:
            return
        with database(root) as db:
            manifest, _ = read_manifest(db, run_id)
            if launch_id is not None:
                with db:
                    acquired = db.execute(
                        "DELETE FROM queued_shards WHERE run_id = ? AND shard = ? AND launch_id = ?",
                        (run_id, shard, launch_id),
                    ).rowcount
                if not acquired:
                    return
            freeze_runner(root, manifest["engine"])
            specs = _by_shard(manifest, match_specs(manifest), shard)
            log = _log_path(root, run_id, shard)

            def cancelled():
                return db.execute("SELECT cancelled FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]

            for case_id, spec in specs.items():
                if cancelled():
                    break
                if _actual_matches(manifest, cached_matches(db, {case_id: spec}, summary=True)):
                    continue
                failure = db.execute(
                    "SELECT 1 FROM failures WHERE run_id = ? AND case_id = ?", (run_id, case_id),
                ).fetchone()
                if bool(failure) != retry_failures:
                    continue
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(f"{case_id} {json_text(spec)}\n")
                try:
                    outcome = run_isolated(root, spec, manifest["timeout"], manifest["replay"], log, cancelled)
                except MatchFailure as error:
                    with db:
                        db.execute(
                            "INSERT OR REPLACE INTO failures VALUES (?, ?, ?, ?)",
                            (run_id, case_id, "timeout" if error.timeout else "error", str(error)),
                        )
                    continue
                if outcome is None:
                    break
                result, elapsed = outcome
                with db:
                    save_match(db, case_id, spec, result, elapsed)
                    if retry_failures:
                        db.execute("DELETE FROM failures WHERE run_id = ? AND case_id = ?", (run_id, case_id))


def cancel(root, run_id):
    with database(root) as db:
        read_manifest(db, run_id)
        db.execute("UPDATE runs SET cancelled = 1 WHERE run_id = ?", (run_id,))
        db.execute("DELETE FROM queued_shards WHERE run_id = ?", (run_id,))
    return status(root, run_id)


def _continue(root, run_id, shard, background, retry_failures):
    with database(root) as db:
        manifest, _ = read_manifest(db, run_id)
    _verify_inputs(root, manifest)
    return _launch(root, manifest, shard=shard, retry_failures=retry_failures, background=background)


def resume(root, run_id, *, shard=None, background=True):
    return _continue(root, run_id, shard, background, retry_failures=False)


def retry(root, run_id, *, shard=None, background=True):
    return _continue(root, run_id, shard, background, retry_failures=True)


def _side_index(side):
    if side is None:
        return None
    if side not in SIDES:
        raise ValueError("side 只支持 red 或 blue")
    return SIDES.index(side)


def _groups(manifest, done, *, candidate=None, opponent=None):
    direct = manifest["mode"] == "direct"
    for version_id, against, seed, repeat, focal, baseline in pairings(manifest):
        if candidate is not None and version_id != candidate:
            continue
        if opponent is not None and against != opponent:
            continue
        ids = {key for key, _ in focal + baseline}
        missing = sorted(key for key in ids if key not in done)
        yield (version_id, against, seed, repeat, focal, baseline, missing, direct)


def _score(row, side):
    result = row["result"]
    return result["scores"][side], result["winner"] == side, result["result"]["scores"][side]


def compare(root, run_id, *, opponent=None, side=None):
    side_index = _side_index(side)
    with database(root) as db:
        manifest, _ = read_manifest(db, run_id)
        specs = match_specs(manifest)
        done = _actual_matches(manifest, cached_matches(db, specs))
        failures = _failure_rows(db, run_id)
    grouped = defaultdict(list)
    for group in _groups(manifest, done, opponent=opponent):
        grouped[group[0]].append(group)
    rows = []
    sides = range(2) if side_index is None else (side_index,)
    for candidate, groups in grouped.items():
        candidate_scores, baseline_scores, candidate_wins = [], [], []
        candidate_items, baseline_items = defaultdict(list), defaultdict(list)
        used, expected_ids, missing_ids = 0, set(), set()
        for _, _, _, _, focal, baseline, missing, direct in groups:
            expected_ids.update(key for key, _ in focal + baseline)
            missing_ids.update(missing)
            if missing:
                continue
            used += 1
            for index in sides:
                score, win, items = _score(done[focal[index][0]], index)
                base_score, _, base_items = _score(done[baseline[index][0]], 1 - index if direct else index)
                candidate_scores.append(score)
                baseline_scores.append(base_score)
                candidate_wins.append(win)
                for name in SCORE_ITEMS:
                    candidate_items[name].append(items[name])
                    baseline_items[name].append(base_items[name])
        row = {
            "candidate": candidate, "completed_pairs": used, "total_pairs": len(groups),
            "missing_cases": len(missing_ids), "failed_cases": len(missing_ids & failures.keys()),
            "failure_rate": len(missing_ids & failures.keys()) / len(expected_ids),
            "status": "completed" if used == len(groups) else "incomplete",
        }
        if used:
            row.update(
                mean_score=fmean(candidate_scores), baseline_mean_score=fmean(baseline_scores),
                score_delta=fmean(a - b for a, b in zip(candidate_scores, baseline_scores, strict=True)),
                win_rate=fmean(candidate_wins),
                components={name: fmean(candidate_items[name]) for name in SCORE_ITEMS},
                component_deltas={
                    name: fmean(a - b for a, b in zip(candidate_items[name], baseline_items[name], strict=True))
                    for name in SCORE_ITEMS
                },
            )
        rows.append(row)
    rows.sort(key=lambda row: (row["status"] != "completed", -row.get("mean_score", 0), row["candidate"]))
    return {
        "run_id": run_id, "suite": manifest["suite"], "baseline": manifest["baseline"],
        "mode": manifest["mode"], "opponent": opponent, "side": side, "rows": rows,
    }


def cases(root, run_id, *, candidate=None, opponent=None, side=None, sort=None, limit=20):
    if sort not in (None, "score_delta"):
        raise ValueError("sort 只支持 score_delta")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit 必须是正整数")
    side_index = _side_index(side)
    with database(root) as db:
        manifest, _ = read_manifest(db, run_id)
        specs = match_specs(manifest)
        done = _actual_matches(manifest, cached_matches(db, specs))
        failures = _failure_rows(db, run_id)
    rows = []
    for version, against, seed, repeat, focal, baseline, missing, direct in _groups(
        manifest, done, candidate=candidate, opponent=opponent,
    ):
        for index, (case_id, spec) in enumerate(focal):
            if side_index is not None and index != side_index:
                continue
            baseline_id = baseline[index][0]
            row = {
                "case_id": case_id, "baseline_case_id": baseline_id, "candidate": version, "opponent": against,
                "seed": seed, "game_no": spec["game_no"], "repeat": repeat, "side": SIDES[index],
                "red": spec["red"], "blue": spec["blue"], "missing_case_ids": missing,
                "failures": [failures[key] for key in missing if key in failures],
            }
            if case_id in done:
                score, win, items = _score(done[case_id], index)
                row.update(score=score, won=win, components=items, has_replay=bool(done[case_id]["has_replay"]))
            if not missing:
                baseline_score, _, _ = _score(done[baseline_id], 1 - index if direct else index)
                row.update(baseline_score=baseline_score, score_delta=row["score"] - baseline_score)
            rows.append(row)
    if sort == "score_delta":
        rows.sort(key=lambda row: ("score_delta" not in row, row.get("score_delta", 0), row["case_id"]))
    return {"run_id": run_id, "suite_id": manifest["suite"]["suite_id"], "total": len(rows), "rows": rows[:limit]}


def _locate(db, case_id):
    if not isinstance(case_id, str) or not CASE_ID.fullmatch(case_id):
        raise ValueError("对局 ID 格式错误")
    result = db.execute("SELECT spec, result, replay FROM matches WHERE case_id = ?", (case_id,)).fetchone()
    if result is not None:
        return json.loads(result["spec"]), result
    for row in db.execute("SELECT manifest FROM runs"):
        spec = match_specs(json.loads(row["manifest"])).get(case_id)
        if spec is not None:
            return spec, None
    raise ValueError(f"找不到对局 {case_id}")


def reproduce(root, case_id, *, replay=True):
    with database(root) as db:
        spec, recorded = _locate(db, case_id)
    freeze_runner(root, engine_fields(spec))
    for version in {spec["red"], spec["blue"]} - {"baseline"}:
        versions.verify(root, version)
    folder = local_dir(root)
    log = folder / "reproduce.log"
    result, elapsed = run_isolated(root, spec, 120, replay, log)
    replay_data = result.pop("replay", None)
    output = {"case_id": case_id, "spec": spec, "result": result, "elapsed": elapsed}
    if recorded is not None:
        output["matches_recorded_result"] = result == json.loads(recorded["result"])
        if replay_data is not None and recorded["replay"] is not None:
            output["matches_recorded_replay"] = replay_data == json.loads(gzip.decompress(recorded["replay"]))
    if replay_data is not None:
        path = folder / "replays" / f"{case_id}-reproduced.json"
        write_json(path, replay_data)
        output["replay_path"] = str(path)
    return output


def replay(root, case_id):
    with database(root) as db:
        _, row = _locate(db, case_id)
    if row is None or row["replay"] is None:
        raise ValueError(f"对局 {case_id} 未保存回放，可用 reproduce 生成")
    path = local_dir(root) / "replays" / f"{case_id}.json"
    write_json(path, json.loads(gzip.decompress(row["replay"])))
    return {"case_id": case_id, "replay_path": str(path)}


def logs(root, run_id, *, shard=0, limit=40):
    if type(limit) is not int or limit < 1:
        raise ValueError("limit 必须是正整数")
    with database(root) as db:
        manifest, _ = read_manifest(db, run_id)
        _by_shard(manifest, {}, shard)
    path = _log_path(root, run_id, shard)
    lines = []
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines = [line.rstrip("\r\n") for line in deque(handle, maxlen=limit)]
    return {"run_id": run_id, "shard": shard, "log_path": str(path), "lines": lines}


def export_run(root, run_id, path, *, shard=None):
    path = Path(path)
    with database(root) as db:
        manifest, _ = read_manifest(db, run_id)
        specs = _by_shard(manifest, match_specs(manifest), shard)
        failures = _failure_rows(db, run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            completed = 0
            with temp_path.open("wb") as target, gzip.GzipFile(fileobj=target, filename="", mode="wb", mtime=0) as archive:
                archive.write((json_text({"manifest": manifest, "shard": shard}) + "\n").encode("utf-8"))
                exported = set()
                for case_id, match in iter_matches(db, dict(sorted(specs.items())), include_replay=True):
                    result = match["result"]
                    if match["replay"] is not None:
                        result["replay"] = json.loads(gzip.decompress(match["replay"]))
                    record = {"case_id": case_id, "result": result, "elapsed": match["elapsed"]}
                    if not manifest["replay"] or "replay" in result:
                        exported.add(case_id)
                        completed += 1
                    archive.write((json_text(record) + "\n").encode("utf-8"))
                for case_id in sorted((failures.keys() & specs.keys()) - exported):
                    record = {"case_id": case_id, "failure": failures[case_id]}
                    archive.write((json_text(record) + "\n").encode("utf-8"))
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return {"run_id": run_id, "path": str(path), "shard": shard, "completed": completed}


def import_run(root, path, *, expected_assignment=None, expected_manifest=None):
    with gzip.open(path, "rt", encoding="utf-8") as archive, database(root) as db:
        header = json.loads(archive.readline())
        manifest = required(header, "manifest")
        if expected_manifest is not None and manifest != expected_manifest:
            raise ValueError("结果文件与批次清单不匹配")
        validate_manifest(manifest)
        shard = required(header, "shard")
        if expected_assignment is not None and (manifest["run_id"], shard) != expected_assignment:
            raise ValueError("分片数据与领取记录不符")
        specs = _by_shard(manifest, match_specs(manifest), shard)
        add_manifest(db, manifest)
        received, reported_failures = set(), set()
        for line in archive:
            record = json.loads(line)
            case_id = required(record, "case_id")
            if not isinstance(case_id, str) or case_id not in specs:
                raise ValueError(f"对局 {case_id} 不在批次的指定分片中")
            if "failure" in record:
                failure = record["failure"]
                kind, message = required(failure, "kind"), required(failure, "message")
                if kind not in ("timeout", "error") or not isinstance(message, str):
                    raise ValueError(f"对局 {case_id} 的失败记录格式错误")
                if _actual_matches(manifest, cached_matches(db, {case_id: specs[case_id]}, summary=True)):
                    continue
                db.execute(
                    "INSERT OR IGNORE INTO failures VALUES (?, ?, ?, ?)",
                    (manifest["run_id"], case_id, kind, message),
                )
                reported_failures.add(case_id)
            else:
                result = required(record, "result")
                save_match(db, case_id, specs[case_id], result, required(record, "elapsed"))
                if not manifest["replay"] or result.get("replay") is not None:
                    db.execute("DELETE FROM failures WHERE run_id = ? AND case_id = ?", (manifest["run_id"], case_id))
                received.add(case_id)
        if received or reported_failures:
            db.execute("UPDATE runs SET attempted = 1 WHERE run_id = ?", (manifest["run_id"],))
    return status(root, manifest["run_id"]) | {
        "imported_cases": len(received), "imported_failures": len(reported_failures),
    }


if __name__ == "__main__":
    root_arg, run_id_arg, shard_arg, worker_mode, launch_arg = sys.argv[1:]
    _work(Path(root_arg), run_id_arg, int(shard_arg), worker_mode == "retry", launch_arg)
