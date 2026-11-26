import asyncio
import hashlib
import io
import json
from copy import deepcopy
from zipfile import ZipFile

import pytest
from seedcup.game import replay
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

from tank import cycle_ladder, online, versions
from tank.eval_plan import SCORE_ITEMS

MAIN, SAMPLING = 57, 99
BASE, NEW = "v-" + "a" * 64, "v-" + "b" * 64


def submission(submission_id, **fields):
    return {
        "id": submission_id, "version": str(submission_id), "sha256": "e" * 64,
        "download_url": f"{online.SUBMISSIONS}/{submission_id}/download",
        "evaluation": {
            "status": "completed", "score": len(SCORE_ITEMS),
            "breakdown": {name: 1 for name in SCORE_ITEMS},
        },
    } | fields


def strategy(submission_id):
    return {"id": submission_id, "version": str(submission_id), "status": "completed", "error": None}


def series(series_id, defender=42, challenger=MAIN, *, activation=False, games=(), ranked=True):
    return {
        "id": series_id, "challenger": challenger, "defender": defender,
        "created_at": 1400 + series_id, "ended_at": 1500 + series_id, "status": "finished",
        "activation": activation, "ranked": ranked, "outcome": 1,
        "challenger_wins": 6, "defender_wins": 2, "draws": 0, "games": list(games),
    }


class Client:
    token = "test-token"

    def __init__(self):
        self.user = {"id": MAIN}
        self.posts = []
        self.timeout = False
        self.publish_series = True
        self.challenge_status = "finished"
        self.downloads = {}
        self.ladder = {
            "qualification": True, "preview": False, "strategy": strategy(10), "pending_strategy": None,
            "can_activate_while_defending": False, "locked_teams": [],
            "cooldown_remaining": 0, "opponent_cooldowns": {}, "rank_moves": [], "series": [],
            "ready_teams": [MAIN, SAMPLING, 41, 42, 43, 44, 45, 61, 62],
            "rankings": [{"team_id": team, "rank": rank, "name": str(team)} for rank, team in enumerate(
                (41, 42, 43, 44, 45, SAMPLING, MAIN, 61, 62), 1
            )],
        }
        self.listing = {
            "final_id": 10, "pending_final_id": None, "server_time": 1400,
            "submissions": [submission(10), submission(11)],
        }

    def request(self, method, path, *, fields=None, compressed=False):
        if method == "GET":
            if path in self.downloads and compressed:
                return self.downloads[path]
            return deepcopy({
                cycle_ladder.CHALLENGE: self.ladder,
                online.SUBMISSIONS: self.listing,
                "/api/me": {"match": None, "in_event": False},
            }[path])
        self.posts.append((path, fields))
        if path == cycle_ladder.CHALLENGE:
            row = series(
                max((row["id"] for row in self.ladder["series"]), default=0) + 1, fields["defender"],
                activation=self.ladder["pending_strategy"] is not None, ranked=fields["defender"] < MAIN,
            )
            row["status"] = self.challenge_status
            if self.publish_series:
                self.ladder["series"].append(row)
            response = {"series": row}
        else:
            submission_id = int(path.split("/")[-2])
            pending = submission_id if submission_id != self.listing["final_id"] else None
            self.listing["pending_final_id"] = pending
            self.ladder["pending_strategy"] = {"id": pending, "version": str(pending)} if pending else None
            response = {"pending_final_id": pending}
        if self.timeout:
            raise online.RequestUncertain("test")
        return response


@pytest.fixture
def client(monkeypatch):
    fake = Client()
    monkeypatch.setattr(online.OnlineClient, "login", lambda root: fake)
    monkeypatch.setattr(online, "_associated_version", lambda root, client, item, cache: {10: BASE, 11: NEW}[item["id"]])
    monkeypatch.setattr(online, "sampling_credentials", lambda root: {"team_id": SAMPLING})
    return fake


def test_matches_three_ranked_and_one_diverse_unranked_without_self(tmp_path, client):
    client.ladder["locked_teams"] = [45]
    client.ladder["opponent_cooldowns"] = {"44": 32}
    client.ladder["series"] = [series(100, defender=61)]
    results = [cycle_ladder.tick(tmp_path) for _ in range(4)]
    folder = cycle_ladder._folder(tmp_path) / "series"
    archived = [cycle_ladder.read_json(folder / f"{row['challenge']['series_id']}.json") for row in results]
    assert [value["role"] for value in archived] == ["climb", "climb", "climb", "coverage"]
    assert [fields["defender"] for _, fields in client.posts] == [43, 43, 43, 62]
    assert results[-1]["next_role"] == "climb"
    assert results[-1]["active_version"] == BASE
    assert results[-1]["active_score"] == len(SCORE_ITEMS)


def test_coverage_alternates_known_difficult_and_unfamiliar_using_own_finished_series(tmp_path, client):
    history = series(100, defender=MAIN, challenger=61)
    client.ladder["series"] = [history]
    cycle_ladder.tick(tmp_path, allow_challenge=False)
    client.ladder["series"] = []
    folder = cycle_ladder._folder(tmp_path) / "series"
    other_team = cycle_ladder.read_json(folder / "100.json")
    for series_id in (101, 102):
        cycle_ladder.write_json(folder / f"{series_id}.json", other_team | {
            "series_id": series_id, "team_id": SAMPLING, "opponent_id": 62,
        })
    for opponent, basis in ((61, "difficult"), (62, "unfamiliar")):
        state = cycle_ladder._load(tmp_path, MAIN)
        state["rotation"] = 3
        cycle_ladder._save(tmp_path, MAIN, state)
        result = cycle_ladder.tick(tmp_path)
        assert result["challenge"]["opponent_id"] == opponent
        assert result["challenge"]["coverage_basis"] == basis


def test_cold_coverage_selects_nearest_ready_lower_team(tmp_path, client):
    state = cycle_ladder._load(tmp_path, MAIN)
    state["rotation"] = 3
    cycle_ladder._save(tmp_path, MAIN, state)
    result = cycle_ladder.tick(tmp_path)
    assert result["challenge"]["opponent_id"] == 61
    assert result["challenge"]["coverage_basis"] == "unfamiliar"


def test_busy_cooldown_and_archive_only_never_request_challenge(tmp_path, client):
    client.ladder["locked_teams"] = [MAIN]
    assert cycle_ladder.tick(tmp_path)["action"] == "busy"
    client.ladder["locked_teams"] = []
    client.ladder["cooldown_remaining"] = 12
    assert cycle_ladder.tick(tmp_path)["action"] == "cooldown"
    client.ladder["cooldown_remaining"] = 0
    assert cycle_ladder.tick(tmp_path, allow_challenge=False)["action"] == "archive_only"
    assert client.posts == []


def test_pause_after_recording_intent_stops_challenge_before_post(tmp_path, client):
    permission = iter((True, True, False))
    stopped = cycle_ladder.tick(tmp_path, allow_challenge=lambda: next(permission))
    assert stopped["challenge"]["status"] == "paused"
    assert client.posts == []
    assert cycle_ladder.tick(tmp_path)["challenge"]["status"] == "confirmed"
    assert len(client.posts) == 1


def test_selection_guard_uses_fresh_pending_id_before_challenging(tmp_path, client):
    state = cycle_ladder.tick(tmp_path, pending_only=11)
    assert state["action"] == "waiting_activation_selection"
    assert client.posts == []
    cycle_ladder.request_activation(tmp_path, NEW, 11)
    state = cycle_ladder.tick(tmp_path, pending_only=11)
    assert state["challenge"]["submission_id"] == 11
    assert state["challenge"]["activation"] is True


def test_queued_teammate_gets_main_account_before_another_challenge(tmp_path,client):
    with cycle_ladder.practice_priority.queue_pending(tmp_path,MAIN):
        assert cycle_ladder.tick(tmp_path)["action"]=="archive_only"
    assert not client.posts
    assert cycle_ladder.tick(tmp_path)["challenge"]["status"]=="confirmed"


def test_lost_challenge_response_waits_and_reconciles_without_repeat(tmp_path, client):
    client.timeout = True
    client.publish_series = False
    first = cycle_ladder.tick(tmp_path)
    assert first["challenge"]["status"] == "uncertain"
    assert cycle_ladder.tick(tmp_path)["action"] == "challenge_pending"
    client.ladder["series"].append(series(1, defender=45))
    client.ladder["cooldown_remaining"] = 20
    after_restart = cycle_ladder.tick(tmp_path)
    assert after_restart["challenge"]["series_id"] == 1
    assert len(client.posts) == 1
    archived = json.loads((tmp_path / ".local/cycle/ladder/series/1.json").read_text(encoding="utf-8"))
    assert archived["role"] == "climb"
    assert archived["outcome"] == 1
    assert archived["version_id"] is None


@pytest.mark.parametrize("operation", ["challenge", "activation"])
@pytest.mark.parametrize("checkpoint", ["prepared", "sending"])
def test_retry_after_crash_before_post_requires_reconciliation_only_for_sending(
    tmp_path, client, monkeypatch, operation, checkpoint,
):
    normal_save = cycle_ladder._save

    def interrupted_save(root, team_id, state):
        normal_save(root, team_id, state)
        if state[operation] and state[operation]["status"] == checkpoint:
            raise SystemExit("crash after checkpoint")

    def invoke():
        if operation == "challenge":
            return cycle_ladder.tick(tmp_path)
        return cycle_ladder.request_activation(tmp_path, NEW, 11)

    monkeypatch.setattr(cycle_ladder, "_save", interrupted_save)
    with pytest.raises(SystemExit):
        invoke()
    assert client.posts == []
    monkeypatch.setattr(cycle_ladder, "_save", normal_save)
    checked = cycle_ladder.reconcile(tmp_path)
    assert checked[operation]["status"] == ("not_sent" if checkpoint == "prepared" else "sending")
    if checkpoint == "sending":
        assert checked["clearable"][operation]
        invoke()
        assert client.posts == []
        cleared = cycle_ladder.reconcile(tmp_path, clear=operation)
        assert cleared["action"] == "cleared"
        assert cleared[operation]["status"] == "cleared"
        assert client.posts == []
    invoke()
    assert len(client.posts) == 1


def test_reconcile_confirms_later_visible_official_series_without_releasing_it(tmp_path, client):
    client.timeout = True
    client.publish_series = False
    first = cycle_ladder.tick(tmp_path)
    assert first["challenge"]["status"] == "uncertain"
    client.ladder["series"].append(series(1, defender=45))
    result = cycle_ladder.reconcile(tmp_path, clear="challenge")
    assert result["action"] == "clear_blocked"
    assert result["challenge"]["status"] == "confirmed"
    assert result["challenge"]["series_id"] == 1
    assert len(client.posts) == 1


def test_pin_requires_verified_pve_and_only_confirmed_own_series_is_attributed(tmp_path, client):
    mismatch = cycle_ladder.request_activation(tmp_path, BASE, 11)["activation"]
    assert mismatch["status"] == "rejected"
    assert "未核实" in mismatch["error"]
    selection = cycle_ladder.request_activation(tmp_path, NEW, 11)
    assert (selection["active_version"], selection["pending_version"]) == (BASE, NEW)
    assert selection["activation"]["status"] == "pending"
    client.ladder["series"] = [series(1, challenger=61, defender=MAIN, activation=True)]
    result = cycle_ladder.tick(tmp_path)
    assert result["challenge"]["activation"] is True
    challenger = cycle_ladder._folder(tmp_path) / "series/2.json"
    defender = cycle_ladder._folder(tmp_path) / "series/1.json"
    assert cycle_ladder.read_json(challenger)["version_id"] == NEW
    assert cycle_ladder.read_json(defender)["version_id"] is None
    assert cycle_ladder.read_json(defender)["outcome"] == -1
    client.ladder["strategy"] = strategy(11)
    client.ladder["pending_strategy"] = None
    client.listing.update(final_id=11, pending_final_id=None)
    checked = cycle_ladder.tick(tmp_path, allow_challenge=False)
    assert checked["activation"]["status"] == "active"
    assert checked["active_version"] == NEW
    rollback = cycle_ladder.request_activation(tmp_path, BASE, 10)
    assert rollback["pending_version"] == BASE
    assert [path for path, _ in client.posts] == [
        f"{online.SUBMISSIONS}/11/final", cycle_ladder.CHALLENGE, f"{online.SUBMISSIONS}/10/final",
    ]


def test_lost_pin_response_does_not_repeat_when_official_pending_arrives(tmp_path, client):
    client.timeout = True
    selected = cycle_ladder.request_activation(tmp_path, NEW, 11)
    assert selected["activation"]["status"] == "pending"
    assert cycle_ladder.request_activation(tmp_path, NEW, 11)["activation"]["status"] == "pending"
    assert len(client.posts) == 1


def test_pause_after_recording_activation_intent_stops_pin_before_post(tmp_path, client):
    permission = iter((True, False))
    selection = cycle_ladder.request_activation(tmp_path, NEW, 11, allow_activation=lambda: next(permission))
    assert selection["activation"]["status"] == "paused"
    assert client.posts == []
    assert cycle_ladder.request_activation(tmp_path, NEW, 11)["activation"]["status"] == "pending"
    assert len(client.posts) == 1


def test_rechecks_expected_active_version_before_pinning(tmp_path, client):
    changed = cycle_ladder.request_activation(tmp_path, NEW, 11, expected_active_version="v-" + "c" * 64)
    assert changed["activation"]["status"] == "active_changed"
    assert client.posts == []


@pytest.mark.parametrize("removed", [False, True])
def test_rollback_reports_unavailable_previous_submission_without_post(tmp_path, client, removed):
    if removed:
        client.listing["submissions"].pop()
    else:
        client.listing["submissions"][-1]["download_url"] = None
    result = cycle_ladder.request_activation(tmp_path, NEW, 11, expected_active_version=BASE)
    assert result["activation"]["status"] == "rejected"
    assert "#11" in result["activation"]["error"]
    assert client.posts == []


def test_activation_waits_for_official_finish_then_active_id(tmp_path, client):
    cycle_ladder.request_activation(tmp_path, NEW, 11)
    client.challenge_status = "running"
    active = cycle_ladder.tick(tmp_path)
    assert active["activation"]["series_id"] == 1
    assert active["activation"]["status"] == "activating"
    assert cycle_ladder.request_activation(tmp_path, NEW, 11)["activation"]["series_id"] == 1
    client.ladder["series"][0]["status"] = "finished"
    pending = cycle_ladder.tick(tmp_path)
    assert pending["action"] == "activation_settling"
    assert pending["activation"]["status"] == "settling"
    client.ladder["pending_strategy"] = None
    client.listing["pending_final_id"] = None
    finished = cycle_ladder.request_activation(tmp_path, NEW, 11)
    assert finished["activation"]["status"] == "series_not_activated"
    assert len(client.posts) == 2


@pytest.mark.parametrize("terminal", ["void", "cancelled", "expired"])
@pytest.mark.parametrize("remains_pending", [False, True])
def test_interrupted_activation_can_only_reuse_its_same_pending_version(tmp_path, client, terminal, remains_pending):
    cycle_ladder.request_activation(tmp_path, NEW, 11)
    client.challenge_status = "running"
    cycle_ladder.tick(tmp_path)
    client.ladder["series"][0].update(status=terminal, note="测试非正常结算")
    if not remains_pending:
        client.ladder["pending_strategy"] = None
        client.listing["pending_final_id"] = None
    previous = cycle_ladder.tick(tmp_path, allow_challenge=False)["activation"]
    assert previous["status"] == f"series_{terminal}"
    assert previous["can_retry"] is remains_pending
    assert previous["series_note"] == "测试非正常结算"
    assert cycle_ladder.request_activation(tmp_path, NEW, 11)["activation"] == previous
    assert len(client.posts) == 2
    if remains_pending:
        selected = cycle_ladder.tick(tmp_path)["activation"]
        assert selected["status"] == "activating"
        assert selected["series_id"] == 2
        assert len(client.posts) == 3


def test_manual_pending_cancellation_is_terminal_for_repeated_automated_request(tmp_path, client):
    cycle_ladder.request_activation(tmp_path, NEW, 11)
    client.ladder["pending_strategy"] = None
    client.listing["pending_final_id"] = None
    cancelled = cycle_ladder.request_activation(tmp_path, NEW, 11)
    assert cancelled["activation"]["status"] == "cancelled"
    assert len(client.posts) == 1


def _package(extra=(), *, wrapped=False):
    file = io.BytesIO()
    with ZipFile(file, "w") as archive:
        prefix = "upload/" if wrapped else ""
        if wrapped:
            archive.writestr(prefix, "")
        archive.writestr(prefix + "submission.json", '{"runtime":"python","entrypoint":"bot.py"}')
        archive.writestr(prefix + "bot.py", "pass\n")
        for path in extra:
            archive.writestr(path, "test")
    return file.getvalue()


@pytest.mark.parametrize("wrapped", [False, True])
def test_current_recovers_unknown_version_from_byte_verified_official_source(tmp_path, monkeypatch, wrapped):
    client = Client()
    raw = _package(wrapped=wrapped)
    url = f"{online.SUBMISSIONS}/10/download"
    digest = hashlib.sha256(raw).hexdigest()
    client.downloads[url] = raw
    client.listing["submissions"][0]["sha256"] = digest
    monkeypatch.setattr(online.OnlineClient, "login", lambda root: client)
    status = cycle_ladder.current(tmp_path)
    versions.verify(tmp_path, status["active_version"])
    assert (tmp_path / f".local/online/packages/{digest}.zip").read_bytes() == raw
    assert status["association_errors"] == {}
    client.downloads.clear()
    assert cycle_ladder.current(tmp_path)["active_version"] == status["active_version"]


@pytest.mark.parametrize("paths", [("../escape.py",), ("dir/A.py", "Dir/b.py"), ("file:stream",), ("NUL.txt",)])
def test_current_rejects_paths_that_cannot_round_trip_windows(tmp_path, monkeypatch, paths):
    client = Client()
    raw = _package(paths)
    url = f"{online.SUBMISSIONS}/10/download"
    client.downloads[url] = raw
    client.listing["submissions"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(online.OnlineClient, "login", lambda root: client)
    result = cycle_ladder.current(tmp_path)
    assert result["active_version"] is None
    assert "路径" in result["association_errors"][10]
    assert not list(tmp_path.rglob("escape.py"))


class Socket:
    def __init__(self, messages):
        self.messages = iter(messages)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def recv(self):
        value = next(self.messages)
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)


def _messages(game_id, series_id):
    return [
        {"type": "start", "first": [0], "last_tick": 2, "frame_count": 3, "data": {
            "format": replay.FORMAT_VERSION, "rules": {}, "map": {}, "result": {"winner": 0},
            "match": {"id": game_id, "series": series_id, "game": 0, "rank_challenge": True,
                      "kind": "ladder", "red": {"id": 42}, "blue": {"id": MAIN}},
        }},
        {"type": "frames", "frames": [[1], [2]]},
    ]


def test_serial_replay_archives_complete_stream_and_resumes_after_4409(tmp_path, client, monkeypatch):
    two_games = [{"match_id": 99, "game": 0, "status": "finished"}]
    client.ladder["series"] = [series(1, games=two_games)]
    attempts = []
    occupied = True

    def connect(uri, **kwargs):
        attempts.append(uri)
        assert kwargs["additional_headers"] == {"Authorization": "Bearer test-token"}
        if occupied:
            return Socket([ConnectionClosedError(Close(4409, "busy"), None)])
        return Socket(_messages(99, 1))

    monkeypatch.setattr(cycle_ladder.websockets, "connect", connect)
    first = cycle_ladder.tick(tmp_path, allow_challenge=False)
    assert first["replay_errors"][0]["code"] == 4409
    assert first["archived_games"] == []
    client.ladder["series"] = []
    occupied = False
    second = cycle_ladder.tick(tmp_path, allow_challenge=False)
    assert second["archived_games"] == [99]
    meta = cycle_ladder.read_json(cycle_ladder._folder(tmp_path) / "games/99.json")
    assert meta["own_side"] == "blue"
    assert meta["version_id"] is None
    assert meta["frame_count"] == 3
    assert meta["stream_seconds"] >= 0
    assert replay.load(cycle_ladder._folder(tmp_path) / "replays/99.json.gz")["frames"] == [[0], [1], [2]]
    cycle_ladder.tick(tmp_path, allow_challenge=False)
    assert len(attempts) == 2


def test_truncated_and_conflicting_replay_stream_is_not_complete(client, monkeypatch):
    first_chunk = _messages(99, 1)[0]
    monkeypatch.setattr(cycle_ladder.websockets, "connect", lambda *args, **kwargs: Socket([
        first_chunk, ConnectionClosedError(Close(1006, "interrupted"), None),
    ]))
    with pytest.raises(cycle_ladder._ReplayDeferred):
        asyncio.run(cycle_ladder._stream_replay(client, 99))
    monkeypatch.setattr(cycle_ladder.websockets, "connect", lambda *args, **kwargs: Socket([
        first_chunk, {"type": "frames", "frames": [[1], [3]]},
    ]))
    with pytest.raises(online.OnlineError, match="末帧"):
        asyncio.run(cycle_ladder._stream_replay(client, 99))
