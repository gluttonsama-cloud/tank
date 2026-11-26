"""Official ranking challenges, submission activation and replay collection."""

import asyncio
import hashlib
import io
import json
import re
import stat
import tempfile
import time
import unicodedata
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

import websockets
from seedcup.game import replay
from seedcup.server.evaluation_manifest import manifest
from websockets.exceptions import ConnectionClosed, WebSocketException

from . import online, practice_priority, versions
from .eval_store import shard_lock
from .workspace import read_json, write_json

CHALLENGE = "/api/rank-challenge"
OPEN_SERIES = ("pending", "running")
INTENT_UNCONFIRMED = ("sending", "uncertain", "selected")
CHALLENGE_UNCONFIRMED = ("sending", "uncertain", "accepted")
INTENT_REUSABLE = ("paused", "not_sent", "cleared")
SERIES_INTERRUPTED = ("void", "cancelled", "expired")
ACTIVATION_READY = ("pending", *(f"series_{result}" for result in SERIES_INTERRUPTED))
ROTATION = ("climb", "climb", "climb", "coverage")
REPLAYS_PER_TICK = 8
REPLAY_TIMEOUT = 90
REPLAY_IDLE_TIMEOUT = 10
REPLAY_BATCH_SECONDS = 30


def _folder(root):
    return root / ".local/cycle/ladder"


def _state_path(root, team_id):
    return _folder(root) / f"state-{team_id}.json"


def _load(root, team_id):
    path = _state_path(root, team_id)
    if path.is_file():
        return read_json(path)
    return {"rotation": 0, "activation": None, "challenge": None, "series": {}, "replay_retry": {}}


def _save(root, team_id, state):
    write_json(_state_path(root, team_id), state)


def _allowed(permission):
    return permission() if callable(permission) else permission


def _recover_prepared(state):
    for name in ("challenge", "activation"):
        intent = state[name]
        if intent and intent["status"] == "prepared":
            intent["status"] = "not_sent"


def _restore_archive(root, client, item):
    submission_id = item["id"]
    url = f"{online.SUBMISSIONS}/{submission_id}/download"
    digest = item["sha256"]
    if item["download_url"] != url or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise online.OnlineError(f"提交 #{submission_id} 没有可核实的官网下载记录")
    packages = root / ".local/online/packages"
    packages.mkdir(parents=True, exist_ok=True)
    package = packages / f"{digest}.zip"
    payload = package.read_bytes() if package.is_file() else b""
    if hashlib.sha256(payload).hexdigest() != digest:
        payload = client.request("GET", url, compressed=True)
    if len(payload) >= 10_000_000 or hashlib.sha256(payload).hexdigest() != digest:
        raise online.OnlineError(f"提交 #{submission_id} 的 ZIP 大小或 SHA-256 不符合官网记录")
    with tempfile.TemporaryDirectory(dir=packages) as temp:
        source = Path(temp) / "source"
        hashes = {}
        prefixes = {}
        with ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.infolist()
            if not 1 <= len(entries) <= 128 or sum(entry.file_size for entry in entries) > 32 * 1024 * 1024:
                raise online.OnlineError(f"提交 #{submission_id} 的 ZIP 文件数或解压大小不合规")
            all_files = {entry.orig_filename for entry in entries if not entry.is_dir()}
            homes = {name.split("/", 1)[0] for name in all_files}
            wrapper = ""
            if "submission.json" not in all_files and len(homes) == 1:
                candidate = homes.pop() + "/"
                if candidate + "submission.json" in all_files and all(name.startswith(candidate) for name in all_files):
                    wrapper = candidate
            for entry in entries:
                original = entry.orig_filename.removesuffix("/") if entry.is_dir() else entry.orig_filename
                parts = original.split("/")
                mode = stat.S_IFDIR if entry.is_dir() else stat.S_IFREG
                if (len(original) > 240 or re.search(r'[\\:<>"|?*\x00-\x1f\x7f]', original)
                        or any(not part or part in (".", "..") or part.endswith((".", " "))
                               or PureWindowsPath(part).is_reserved() for part in parts)
                        or entry.flag_bits & 1 or stat.S_IFMT(entry.external_attr >> 16) not in (0, mode)):
                    raise online.OnlineError(f"提交 #{submission_id} 的 ZIP 包含不兼容的文件路径")
                if entry.is_dir():
                    continue
                name = original.removeprefix(wrapper)
                parts = name.split("/")
                for index in range(1, len(parts) + 1):
                    prefix = "/".join(parts[:index])
                    key = unicodedata.normalize("NFC", prefix).casefold()
                    node = (prefix, index == len(parts))
                    if prefixes.setdefault(key, node) != node:
                        raise online.OnlineError(f"提交 #{submission_id} 的 ZIP 包含冲突文件路径")
                if name in hashes:
                    raise online.OnlineError(f"提交 #{submission_id} 的 ZIP 包含重复文件")
                content = archive.read(entry)
                hashes[name] = hashlib.sha256(content).hexdigest()
                target = source.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        if "submission.json" not in hashes:
            raise online.OnlineError(f"提交 #{submission_id} 的 ZIP 缺少 submission.json")
        manifest((source / "submission.json").read_bytes(), set(hashes))
        snapshot = versions.create_from_directory(
            root, source, "天梯原版", f"从主号正式提交 {submission_id} 的原始 ZIP 恢复",
        )
        if snapshot["files"] != hashes:
            raise online.OnlineError(f"提交 #{submission_id} 的 ZIP 包含快照未保存的文件")
        draft = Path(temp) / "original.zip"
        draft.write_bytes(payload)
        draft.replace(package)
    return snapshot["id"]


def _read_official(root, client):
    ladder = client.request("GET", CHALLENGE)
    submissions = client.request("GET", online.SUBMISSIONS)
    team_id = client.user["id"]
    active_id = submissions["final_id"]
    pending_id = submissions["pending_final_id"]
    official_active = ladder["strategy"]
    official_pending = ladder["pending_strategy"]
    consistent = ((official_active["id"] if official_active else None) == active_id
                  and (official_pending["id"] if official_pending else None) == pending_id)
    rows = {}
    for row in submissions["submissions"]:
        if (type(row["id"]) is not int or row["id"] < 1 or not isinstance(row["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
            raise online.OnlineError("官网返回了无效的提交记录标识")
        rows[row["id"]] = row
    verified = {}
    archives = {}
    association_errors = {}
    for submission_id in (active_id, pending_id):
        if submission_id in rows:
            item = rows[submission_id]
            version_id = online._associated_version(root, client, item, archives)
            if version_id is None and item["download_url"]:
                try:
                    restored = _restore_archive(root, client, item)
                    archives[item["sha256"]] = (restored, root / ".local/online/packages" / (item["sha256"] + ".zip"))
                    version_id = online._associated_version(root, client, item, archives)
                except (BadZipFile, NotImplementedError, OSError, ValueError) as error:
                    association_errors[submission_id] = str(error)
            verified[submission_id] = version_id
    standings = ladder["rankings"]
    rank = next((row["rank"] for row in standings if row["team_id"] == team_id), None)
    active_score = online._complete_score(rows[active_id]) if active_id in rows else None
    status = {
        "team_id": team_id,
        "active_submission": active_id,
        "active_version": verified.get(active_id) if consistent else None,
        "active_score": active_score["score"] if active_score else None,
        "pending_submission": pending_id,
        "pending_evaluated": pending_id in rows and online._complete_score(rows[pending_id]) is not None,
        "pending_version": verified.get(pending_id) if consistent else None,
        "rank": rank,
        "standings": standings,
        "consistent": consistent,
        "cooldown_remaining": ladder["cooldown_remaining"],
        "association_errors": association_errors,
    }
    return status, ladder, submissions, rows, archives


def current(root: Path) -> dict:
    """Read the official active and pending submissions and their verified local snapshots."""
    return _read_official(root, online.OnlineClient.login(root))[0]


def _settle_activation(state, status, own):
    intent = state["activation"]
    if not intent or not status["consistent"]:
        return
    wanted = intent["submission_id"]
    active, pending = status["active_submission"], status["pending_submission"]
    series_id = intent.get("series_id")
    series = next((row for row in own if row["id"] == series_id), None)
    if series is not None:
        intent["series_status"] = series["status"]
        intent["series_note"] = series.get("note")
    series_status = intent.get("series_status")
    if active == wanted:
        intent["status"] = "active"
    elif series_status in SERIES_INTERRUPTED:
        intent["status"] = f"series_{series_status}"
        intent["can_retry"] = pending == wanted
    elif pending == wanted:
        intent["status"] = ("activating" if series_status in OPEN_SERIES else
                            "settling" if series_status == "finished" else "pending")
    elif intent["status"] in ("pending", "activating", "settling", "active"):
        if pending is not None or active != intent["previous_submission"]:
            intent["status"] = "superseded"
        else:
            intent["status"] = "series_not_activated" if series_status == "finished" else "cancelled"


def request_activation(
    root: Path, version_id: str, submission_id: int, *, allow_activation: bool | Callable[[], bool] = True,
    expected_active_version: str | None = None,
) -> dict:
    """Select an evaluated, verified submission. A later outgoing series activates it."""
    if type(submission_id) is not int or submission_id < 1:
        raise online.OnlineError("官网提交 ID 必须为正整数")
    if not isinstance(version_id, str) or not re.fullmatch(r"v-[0-9a-f]{64}", version_id):
        raise online.OnlineError("换版必须指定本机策略快照 ID")
    with shard_lock(_folder(root), "ladder") as acquired:
        client = online.OnlineClient.login(root)
        status, ladder, submissions, rows, archives = _read_official(root, client)
        if not acquired:
            return status | {"activation": {"status": "busy"}}
        if not status["consistent"]:
            return status | {"activation": {"status": "refresh"}}
        if expected_active_version is not None and status["active_version"] not in (expected_active_version, version_id):
            return status | {"activation": {"status": "active_changed"}}
        item = rows.get(submission_id)
        already_active = status["active_submission"] == submission_id and status["active_version"] == version_id
        failure = None
        if ladder["preview"] or not ladder["qualification"]:
            failure = "官网未开启正式排位换版"
        elif item is None:
            failure = f"官网提交列表中找不到 #{submission_id}"
        elif online._associated_version(root, client, item, archives) != version_id:
            failure = f"提交 #{submission_id} 与本机快照未核实关联"
        elif not already_active and (online._complete_score(item) is None or not item["download_url"]):
            failure = f"提交 #{submission_id} 尚未通过完整人机评测或源码已清理"
        if failure is not None:
            return status | {"activation": {"status": "rejected", "error": failure}}
        team_id = status["team_id"]
        state = _load(root, team_id)
        _recover_prepared(state)
        own = _own_series(ladder, team_id)
        _reconcile_challenge(state, status, own)
        _settle_activation(state, status, own)
        _save(root, team_id, state)
        previous = state["activation"]
        if previous and previous["submission_id"] == submission_id and previous["status"] not in INTENT_REUSABLE:
            return status | {"activation": previous}
        if previous and previous["status"] in INTENT_UNCONFIRMED:
            return status | {"activation": {"status": "other_activation_pending",
                                            "submission_id": previous["submission_id"]}}
        challenge = state["challenge"]
        if challenge and challenge["status"] in CHALLENGE_UNCONFIRMED:
            return status | {"activation": {"status": "challenge_pending"}}
        active, pending = status["active_submission"], status["pending_submission"]
        intent = {
            "submission_id": submission_id, "version_id": version_id,
            "previous_submission": active, "previous_version": status["active_version"],
            "previous_pending": pending, "requested_at": submissions["server_time"], "status": "prepared",
        }
        state["activation"] = intent
        if submission_id == active and pending is None:
            intent["status"] = "active"
        elif submission_id == pending:
            intent["status"] = "pending"
        elif pending is not None and (not previous or previous["submission_id"] != pending):
            return status | {"activation": {"status": "superseded", "error": f"官网已有待生效提交 #{pending}"}}
        elif not _allowed(allow_activation):
            return status | {"activation": {"status": "paused"}}
        else:
            _save(root, team_id, state)
            if not _allowed(allow_activation):
                intent["status"] = "paused"
                _save(root, team_id, state)
                return status | {"activation": intent}
            intent["status"] = "sending"
            _save(root, team_id, state)
            try:
                client.request("POST", f"{online.SUBMISSIONS}/{submission_id}/final")
            except online.RequestUncertain:
                intent["status"] = "uncertain"
            except online.RemoteError as error:
                intent["status"] = "rejected"
                intent["error"] = str(error)
            else:
                intent["status"] = "selected"
            _save(root, team_id, state)
            if intent["status"] != "rejected":
                status, latest, _, _, _ = _read_official(root, client)
                _settle_activation(state, status, _own_series(latest, team_id))
        _save(root, team_id, state)
        return status | {"activation": intent}


def _own_series(data, team_id):
    return [row for row in data["series"] if team_id in (row["challenger"], row["defender"])]


def _coverage_history(root, own, team_id):
    finished = {}
    for row in own:
        if row["status"] == "finished" and row["outcome"] in (-1, 0, 1):
            outgoing = row["challenger"] == team_id
            opponent = row["defender"] if outgoing else row["challenger"]
            finished[row["id"]] = (opponent, row["created_at"], row["outcome"] * (1 if outgoing else -1))
    for path in (_folder(root) / "series").glob("*.json"):
        if int(path.stem) in finished:
            continue
        record = read_json(path)
        if record["team_id"] == team_id and record["status"] == "finished" and record["outcome"] in (-1, 0, 1):
            finished[record["series_id"]] = (record["opponent_id"], record["started_at"], record["outcome"])
    history = {}
    for opponent, started_at, outcome in finished.values():
        history.setdefault(opponent, []).append((started_at, outcome))
    return history


def _coverage_opponent(root, own, options, team_id, rank, bindings):
    history = _coverage_history(root, own, team_id)
    seek_difficult = sum(record["role"] == "coverage" for record in bindings.values()) % 2 == 0
    difficult = []
    for row in options:
        series = history.get(row["team_id"], [])
        differential = sum(outcome for _, outcome in series)
        if differential < 0:
            # Two neutral series keep a single upset from outranking a repeated losing record.
            difficult.append((differential / (len(series) + 2), max(start for start, _ in series), row["team_id"], row))
    if seek_difficult and difficult:
        return min(difficult)[-1], "difficult"

    def priority(row):
        series = history.get(row["team_id"], [])
        return len(series), max((start for start, _ in series), default=0), abs(row["rank"] - rank), row["team_id"]

    return min(options, key=priority), "unfamiliar"


def _candidate(root, data, status, state, own):
    team_id, rank = status["team_id"], status["rank"]
    if data["preview"] or not data["qualification"]:
        return None, "qualification_unavailable"
    if not status["consistent"] or rank is None:
        return None, "strategy_or_rank_unavailable"
    activation = data["pending_strategy"]
    if activation is not None:
        intent = state["activation"]
        if not intent or intent["submission_id"] != activation["id"]:
            return None, "external_pending_strategy"
        if intent["status"] not in ACTIVATION_READY:
            return None, f"activation_{intent['status']}"
        version = status["pending_version"]
        strategy = activation
    else:
        version = status["active_version"]
        strategy = data["strategy"]
    if not version or strategy is None or not (
        status["pending_evaluated"] if activation is not None else strategy["status"] == "completed"
    ):
        return None, "strategy_unverified"
    if data["cooldown_remaining"]:
        return None, "cooldown"
    locked = set(data["locked_teams"])
    can_switch_defending = activation is not None and data["can_activate_while_defending"]
    if ((team_id in locked and not can_switch_defending)
            or any(row["status"] in OPEN_SERIES and (row["challenger"] == team_id or not can_switch_defending)
                   for row in own)):
        return None, "busy"
    try:
        sampling_team = online.sampling_credentials(root)["team_id"]
    except online.OnlineError:
        return None, "sampling_team_unknown"
    excluded = {team_id, sampling_team}
    ready = set(data["ready_teams"])
    desired = ROTATION[state["rotation"]]
    opponent_history = {}
    for series in own:
        opponent = series["defender"] if team_id == series["challenger"] else series["challenger"]
        opponent_history.setdefault(opponent, []).append(series["created_at"])
    pools = {"climb": [], "coverage": []}
    for row in data["rankings"]:
        if row["team_id"] not in excluded and row["rank"] != rank:
            pools["climb" if row["rank"] < rank else "coverage"].append(row)
    role = desired if pools[desired] else ("coverage" if desired == "climb" else "climb")
    options = [
        row for row in pools[role]
        if row["team_id"] in ready and row["team_id"] not in locked
        and not data["opponent_cooldowns"].get(str(row["team_id"]))
    ]
    if not options:
        return None, f"{role}_unavailable"

    coverage_basis = None
    if role == "coverage":
        chosen, coverage_basis = _coverage_opponent(root, own, options, team_id, rank, state["series"])
    else:
        chosen = min(options, key=lambda row: (
            rank - row["rank"], max(opponent_history.get(row["team_id"], []), default=0), row["team_id"],
        ))
    return {
        "opponent_id": chosen["team_id"], "opponent_rank": chosen["rank"], "rank_before": rank,
        "role": role, "rotation_slot": desired, "coverage_basis": coverage_basis, "submission_id": strategy["id"],
        "version_id": version, "activation": activation is not None,
        "previous_submission": status["active_submission"],
    }, None


def _response_series_id(response):
    if not isinstance(response, dict):
        return None
    row = response.get("series", response)
    return row.get("id") if isinstance(row, dict) and type(row.get("id")) is int else None


def _challenge_matches(intent, team_id, own):
    candidates = [
        row for row in own if row["challenger"] == team_id and row["defender"] == intent["opponent_id"]
        and row["id"] > intent["last_series_id"]
        and row["created_at"] >= intent["requested_at"]
        and bool(row["activation"]) == intent["activation"]
    ]
    acknowledged_id = intent.get("acknowledged_series")
    if acknowledged_id is not None:
        candidates = [row for row in candidates if row["id"] == acknowledged_id]
    return candidates


def _reconcile_challenge(state, status, own):
    intent = state["challenge"]
    if not intent:
        return
    if intent["status"] == "confirmed":
        official = next((row for row in own if row["id"] == intent["series_id"]), None)
        if official is not None:
            intent.update(series_status=official["status"], series_note=official.get("note"))
        return
    if intent["status"] not in CHALLENGE_UNCONFIRMED:
        return
    candidates = _challenge_matches(intent, status["team_id"], own)
    if len(candidates) != 1:
        return
    row = candidates[0]
    active, pending = status["active_submission"], status["pending_submission"]
    chosen = intent["submission_id"]
    if intent["activation"]:
        strategy_matches = ((active == intent["previous_submission"] and pending == chosen)
                            or (active == chosen and pending is None))
    else:
        strategy_matches = active == chosen and pending is None
    attributed = intent["status"] == "accepted" and status["consistent"] and strategy_matches
    role = "climb" if row["ranked"] else "coverage"
    state["series"][str(row["id"])] = {
        "role": role, "rank_before": intent["rank_before"],
        "coverage_basis": intent.get("coverage_basis") if role == "coverage" else None,
        "version_id": intent["version_id"] if attributed else None,
        "submission_id": chosen if attributed else None,
        "attribution": "accepted_challenge_with_verified_submission" if attributed else None,
    }
    intent.update(status="confirmed", series_id=row["id"], series_status=row["status"], series_note=row.get("note"))
    activation = state["activation"]
    if intent["activation"] and activation and activation["submission_id"] == chosen:
        activation.update(series_id=row["id"], series_status=row["status"])
        activation.pop("can_retry", None)
    if role == intent["rotation_slot"]:
        state["rotation"] = (state["rotation"] + 1) % len(ROTATION)


def reconcile(root: Path, *, clear: str | None = None) -> dict:
    """Resolve tracked writes from the official read APIs; clear only with explicit operator input."""
    if clear not in (None, "challenge", "activation"):
        raise online.OnlineError("请选择清除 challenge 或 activation 的待确认记录")
    with shard_lock(_folder(root), "ladder") as acquired:
        if not acquired:
            return {"action": "busy"}
        client = online.OnlineClient.login(root)
        status, ladder, submissions, _, _ = _read_official(root, client)
        team_id = status["team_id"]
        state = _load(root, team_id)
        _recover_prepared(state)
        own = _own_series(ladder, team_id)
        _reconcile_challenge(state, status, own)
        _settle_activation(state, status, own)
        challenge, activation = state["challenge"], state["activation"]
        challenge_unconfirmed = challenge and challenge["status"] in CHALLENGE_UNCONFIRMED
        activation_unconfirmed = activation and activation["status"] in INTENT_UNCONFIRMED
        candidates = _challenge_matches(challenge, team_id, own) if challenge_unconfirmed else []
        challenge_clearable = bool(
            challenge_unconfirmed and challenge["status"] != "accepted" and status["consistent"] and not candidates
            and not ladder["cooldown_remaining"] and team_id not in ladder["locked_teams"]
            and not any(row["status"] in OPEN_SERIES for row in own)
        )
        activation_clearable = bool(
            activation_unconfirmed and activation["status"] != "selected" and status["consistent"]
            and status["active_submission"] == activation["previous_submission"]
            and status["pending_submission"] == activation.get("previous_pending")
        )
        clearable = {"challenge": challenge_clearable, "activation": activation_clearable}
        action = "checked"
        if clear is not None:
            action = "clear_blocked"
            if clearable[clear]:
                state[clear].update(status="cleared", cleared_at=submissions["server_time"])
                action = "cleared"
        _save(root, team_id, state)
        return status | {
            "action": action, "challenge": challenge, "activation": activation, "clearable": clearable,
            "matching_series": [row["id"] for row in candidates],
        }


def _series_record(row, status, state, rank_changes):
    team_id = status["team_id"]
    outgoing = team_id == row["challenger"]
    known = state["series"].get(str(row["id"]), {})
    move = rank_changes.get(row["id"])
    return {
        "series_id": row["id"], "team_id": team_id,
        "opponent_id": row["defender"] if outgoing else row["challenger"],
        "role": known.get("role", "external" if outgoing else "incoming"),
        "coverage_basis": known.get("coverage_basis"),
        "version_id": known.get("version_id"), "submission_id": known.get("submission_id"),
        "attribution": known.get("attribution"), "status": row["status"],
        "challenger": row["challenger"], "defender": row["defender"],
        "ranked": bool(row["ranked"]), "activation": bool(row["activation"]), "note": row.get("note"),
        "rank_before": move["previous_rank"] if move else known.get("rank_before"),
        "rank_after": move["current_rank"] if move else None,
        "started_at": row["created_at"], "ended_at": row["ended_at"],
        "outcome": row["outcome"] * (1 if outgoing else -1) if row["outcome"] is not None else None,
        "wins": row["challenger_wins"] if outgoing else row["defender_wins"],
        "losses": row["defender_wins"] if outgoing else row["challenger_wins"],
        "draws": row["draws"], "games": row["games"],
    }


def _series_records(root, own, ladder, status, state):
    rank_changes = {
        row["series_id"]: row for row in ladder["rank_moves"] if row["team_id"] == status["team_id"]
    }
    folder = _folder(root) / "series"
    results = {}
    for row in own:
        record = _series_record(row, status, state, rank_changes)
        path = folder / f"{row['id']}.json"
        previous = read_json(path) if path.is_file() else {}
        for field in ("rank_before", "rank_after"):
            if record[field] is None:
                record[field] = previous.get(field)
        if previous != record:
            write_json(path, record)
        results[row["id"]] = record
    for path in folder.glob("*.json"):
        if int(path.stem) not in results:
            record = read_json(path)
            if record["team_id"] == status["team_id"]:
                results[record["series_id"]] = record
    return list(results.values())


class _ReplayDeferred(online.OnlineError):
    def __init__(self, reason, code=None):
        super().__init__(reason)
        self.code = code


async def _stream_replay(client, game_id):
    server = urlsplit(online.SOCKET).netloc
    try:
        async with asyncio.timeout(REPLAY_TIMEOUT):
            async with websockets.connect(
                f"wss://{server}/ws/replay/{game_id}",
                additional_headers={"Authorization": f"Bearer {client.token}"},
                open_timeout=5, close_timeout=1, max_size=2**22,
            ) as socket:
                start = json.loads(await asyncio.wait_for(socket.recv(), REPLAY_IDLE_TIMEOUT))
                if start["type"] != "start":
                    raise online.OnlineError(f"官网回放 {game_id} 缺少起始数据")
                frames = [start["first"]]
                count = start["frame_count"]
                while len(frames) < count:
                    chunk = json.loads(await asyncio.wait_for(socket.recv(), REPLAY_IDLE_TIMEOUT))
                    if chunk["type"] != "frames" or not chunk["frames"]:
                        raise online.OnlineError(f"官网回放 {game_id} 的帧数据无效")
                    frames.extend(chunk["frames"])
                if len(frames) != count or frames[-1][0] != start["last_tick"]:
                    raise online.OnlineError(f"官网回放 {game_id} 的帧数或末帧不一致")
                return start["data"] | {"frames": frames}
    except ConnectionClosed as error:
        code = error.rcvd.code if error.rcvd is not None else None
        reason = {
            4401: "回放登录失效", 4403: "回放账号需要修改初始密码",
            4404: "官网回放不可访问", 4409: "本账号已有回放在打开",
            4412: "对局尚未结束", 4414: "回放尚未生成",
        }.get(code, "回放传输中断")
        raise _ReplayDeferred(reason, code) from None
    except (TimeoutError, OSError, WebSocketException):
        raise _ReplayDeferred("回放连接失败，请稍后重试") from None
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise online.OnlineError(f"官网回放 {game_id} 的数据格式无效") from None


def _replay_record(data, record, game):
    game_id = game["match_id"]
    try:
        match = data["match"]
        red, blue = match["red"]["id"], match["blue"]["id"]
        ticks = [frame[0] for frame in data["frames"]]
        if (data["format"] != replay.FORMAT_VERSION or not data["result"] or not ticks
                or not all(type(tick) is int for tick in ticks)
                or any(a >= b for a, b in pairwise(ticks))):
            raise online.OnlineError(f"官网回放 {game_id} 未完整结束或帧序无效")
        if (match["id"] != game_id or match["series"] != record["series_id"] or match["game"] != game["game"]
                or not match["rank_challenge"] or {red, blue} != {record["team_id"], record["opponent_id"]}):
            raise online.OnlineError(f"官网回放 {game_id} 的队伍或场次与系列赛不一致")
    except (KeyError, IndexError, TypeError):
        raise online.OnlineError(f"官网回放 {game_id} 缺少对局身份或帧数据") from None
    return {
        "game_id": game_id, "series_id": record["series_id"], "game": game["game"],
        "team_id": record["team_id"], "opponent_id": record["opponent_id"],
        "own_side": "red" if red == record["team_id"] else "blue",
        "version_id": record["version_id"], "submission_id": record["submission_id"],
        "attribution": record["attribution"], "role": record["role"], "status": "archived",
    }


async def _archive_replay(root, client, record, game):
    game_id = game["match_id"]
    path = _folder(root) / "replays" / f"{game_id}.json.gz"
    duration = None
    if path.is_file():
        data = replay.load(path)
        meta = _replay_record(data, record, game)
    else:
        began = time.monotonic()
        data = await _stream_replay(client, game_id)
        duration = round(time.monotonic() - began, 3)
        meta = _replay_record(data, record, game)
        replay.save(path, data)
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    meta.update(replay_path=str(path.resolve()), replay_sha256=digest, frame_count=len(data["frames"]))
    if duration is not None:
        meta["stream_seconds"] = duration
    write_json(_folder(root) / "games" / f"{game_id}.json", meta)
    return game_id


async def _archive_replays(root, client, records, state):
    done, errors = [], []
    now = time.time()
    deadline = time.monotonic() + REPLAY_BATCH_SECONDS
    retries = state["replay_retry"]
    def needs_archive(record, game):
        path = _folder(root) / "games" / f"{game['match_id']}.json"
        if not path.is_file():
            return True
        known = read_json(path)
        return any(known[key] != record[key] for key in ("role", "version_id", "submission_id", "attribution"))

    candidates = (
        (record, game) for record in sorted(records, key=lambda row: (row["version_id"] is None, -row["series_id"]))
        for game in record["games"] if game["status"] == "finished"
        and needs_archive(record, game)
        and retries.get(str(game["match_id"]), 0) <= now
    )
    for _ in range(REPLAYS_PER_TICK):
        if time.monotonic() >= deadline:
            break
        candidate = next(candidates, None)
        if candidate is None:
            break
        record, game = candidate
        game_id = game["match_id"]
        try:
            done.append(await _archive_replay(root, client, record, game))
        except _ReplayDeferred as error:
            errors.append({"game_id": game_id, "code": error.code, "reason": str(error)})
            if error.code in (None, 4401, 4403, 4409):
                break
            retries[str(game_id)] = now + 60
        except (online.OnlineError, OSError, ValueError) as error:
            errors.append({"game_id": game_id, "reason": str(error)})
            retries[str(game_id)] = now + 60
        else:
            retries.pop(str(game_id), None)
    return done, errors


def tick(
    root: Path, allow_challenge: bool | Callable[[], bool] = True, *, pending_only: int | None = None,
) -> dict:
    """Poll the main account, optionally make one official challenge, and archive serially."""
    with shard_lock(_folder(root), "ladder") as acquired:
        if not acquired:
            return current(root) | {"action": "busy"}
        client = online.OnlineClient.login(root)
        status, ladder, submissions, _, _ = _read_official(root, client)
        team_id = status["team_id"]
        state = _load(root, team_id)
        _recover_prepared(state)
        own = _own_series(ladder, team_id)
        _reconcile_challenge(state, status, own)
        _settle_activation(state, status, own)
        _save(root, team_id, state)
        challenge = state["challenge"]
        activation = state["activation"]
        if challenge and challenge["status"] in CHALLENGE_UNCONFIRMED:
            action = "challenge_pending"
        elif activation and activation["status"] in INTENT_UNCONFIRMED:
            action = "activation_pending"
        elif not _allowed(allow_challenge) or practice_priority.foreground_pending(root,team_id):
            action = "archive_only"
        elif pending_only is not None and status["pending_submission"] != pending_only:
            action = "waiting_activation_selection"
        else:
            choice, action = _candidate(root, ladder, status, state, own)
            if choice is not None:
                me = client.request("GET", "/api/me")
                if me["match"] is not None or me["in_event"]:
                    action = "busy"
                elif not _allowed(allow_challenge) or practice_priority.foreground_pending(root,team_id):
                    action = "archive_only"
                else:
                    choice.update(
                        status="prepared", requested_at=submissions["server_time"],
                        last_series_id=max((row["id"] for row in own), default=0),
                    )
                    state["challenge"] = choice
                    _save(root, team_id, state)
                    if not _allowed(allow_challenge) or practice_priority.foreground_pending(root,team_id):
                        choice["status"] = "paused"
                    else:
                        choice["status"] = "sending"
                        _save(root, team_id, state)
                        try:
                            response = client.request("POST", CHALLENGE, fields={"defender": choice["opponent_id"]})
                        except online.RequestUncertain:
                            choice["status"] = "uncertain"
                        except online.RemoteError as error:
                            choice.update(status="rejected", error=str(error))
                        else:
                            choice.update(status="accepted", acknowledged_series=_response_series_id(response))
                    _save(root, team_id, state)
                    action = f"challenge_{choice['status']}"
                    if choice["status"] == "accepted":
                        status, ladder, _, _, _ = _read_official(root, client)
                        own = _own_series(ladder, team_id)
                        _reconcile_challenge(state, status, own)
                        _settle_activation(state, status, own)
                        _save(root, team_id, state)
        archived_series = _series_records(root, own, ladder, status, state)
        games, replay_errors = asyncio.run(_archive_replays(root, client, archived_series, state))
        _save(root, team_id, state)
        return status | {
            "action": action, "challenge": state["challenge"], "activation": state["activation"],
            "series_seen": len(archived_series), "archived_games": games, "replay_errors": replay_errors,
            "next_role": ROTATION[state["rotation"]],
        }
