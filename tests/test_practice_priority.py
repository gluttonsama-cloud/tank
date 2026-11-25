import asyncio
from pathlib import Path
import subprocess
import sys
import time

import pytest
from seedcup.game import rules
from seedcup.server.config import Settings
from seedcup.server.db import DB
from seedcup.server.hub import Hub

from tank import online, practice_priority
from tank.workspace import write_json


_FOREGROUND = """
from pathlib import Path
import sys
from tank import practice_priority

root = Path(sys.argv[1])
with practice_priority.foreground(root):
    with (root / 'order.txt').open('a', encoding='utf-8') as output:
        output.write(sys.argv[2] + '\\n')
"""


def _wait_for_tickets(root: Path, count: int):
    deadline = time.monotonic() + 10
    folder = root / ".local/online/practice-locks"
    while len(list(folder.glob("*.ticket"))) < count:
        if time.monotonic() >= deadline:
            raise AssertionError(f"等待 {count} 张前台练习票据超时")
        time.sleep(0.02)


def test_waiting_foreground_runs_in_ticket_order_across_processes(tmp_path):
    processes = []
    try:
        with practice_priority.collection(tmp_path) as acquired:
            assert acquired
            for name in ("first", "second"):
                process = subprocess.Popen([sys.executable, "-c", _FOREGROUND, str(tmp_path), name],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                processes.append(process)
                _wait_for_tickets(tmp_path, len(processes))
            assert practice_priority.foreground_pending(tmp_path)
            assert not (tmp_path / "order.txt").exists()
        for process in processes:
            _, error = process.communicate(timeout=10)
            assert process.returncode == 0, error
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()
    assert (tmp_path / "order.txt").read_text(encoding="utf-8").splitlines() == ["first", "second"]
    assert not practice_priority.foreground_pending(tmp_path)


def test_host_queue_publishes_intent_without_holding_the_session(tmp_path):
    with practice_priority.queue_pending(tmp_path):
        assert practice_priority.foreground_pending(tmp_path)
        with practice_priority.foreground(tmp_path):
            with practice_priority.collection(tmp_path) as acquired:
                assert not acquired
        assert practice_priority.foreground_pending(tmp_path)
    assert not practice_priority.foreground_pending(tmp_path)


@pytest.mark.parametrize("when", ["before_first", "between_games", "during_close", "late_start", "official_abort"])
def test_cleanup_waits_for_safe_point_in_sdk_two_game_room(tmp_path, monkeypatch, when):
    monkeypatch.setattr(rules, "GAME_TICKS", 2)

    async def exercise_room():
        settings = Settings(data_dir=tmp_path / "server", between_games=0.1)
        db = DB(settings.db_path)
        hub = Hub(settings, db)
        team_id = db.create_user("room-test", "Room Test", "team", "unused")
        code = "TST234"
        db.run("""INSERT INTO rooms (code,owner_id,opponent,password,games,seed,public,status,created_at)
                  VALUES (?, ?, 'baseline:rookie', '', 2, 42, 0, 'waiting', ?)""", code, team_id, time.time())
        loop = asyncio.get_running_loop()
        close_games = []
        first_query = asyncio.Event()

        def finish(match):
            while not match.game.over:
                match.advance()
            hub._finish([match])

        class Client:
            user = {"id": team_id}

            def request(self, method, path):
                return asyncio.run_coroutine_threadsafe(self.respond(method, path), loop).result(5)

            async def respond(self, method, path):
                match = hub.team_match(team_id)
                if path == "/api/me" and method == "GET":
                    first_query.set()
                    return {"user": self.user, "client": {"online": False}, "in_event": False,
                            "match": match.meta if match else None}
                if path == "/api/team/matches" and method == "GET":
                    return {"matches": db.matches_for_team(team_id)}
                if path == online.ROOMS and method == "GET":
                    return {"rooms": [item | {"matches": []} for item in db.rooms_for_team(team_id)]}
                assert (method, path) == ("POST", f"{online.ROOMS}/{code}/close")
                if when == "during_close" and not close_games:
                    assert match.meta["game"] == 0
                    finish(match)
                    match = None
                if when == "late_start" and not close_games:
                    hub.start_room(db.room(code))
                close_games.append(match.meta["game"] if match else None)
                hub.stop_rooms_for({team_id}, "房主关闭了房间")
                db.run("UPDATE rooms SET status = 'closed' WHERE code = ?", code)
                return {"ok": True}

        try:
            room = online._PracticeRoom(tmp_path, Client())
            write_json(room.pending, {"code": code, "games": 2, "start": "accepted"})
            if when == "before_first":
                closing = asyncio.create_task(room.close(code))
                await first_query.wait()
                assert close_games == []
                hub.start_room(db.room(code))
            else:
                if when != "late_start":
                    hub.start_room(db.room(code))
                    await asyncio.sleep(0)
                    if when == "between_games":
                        finish(hub.team_match(team_id))
                        assert hub.team_match(team_id) is None and db.room(code)["status"] == "playing"
                    elif when == "official_abort":
                        hub.stop_rooms_for({team_id}, "正式赛开始")
                        assert hub.team_match(team_id) is None and db.room(code)["status"] == "waiting"
                closing = asyncio.create_task(room.close(code))

            await asyncio.wait_for(closing, 3)
            assert close_games == {"before_first": [0], "between_games": [1], "during_close": [None, 1],
                                   "late_start": [None, 0], "official_abort": [None]}[when]
            assert not room.pending.exists()
            assert not hub.room_tasks and not hub.matches and db.room(code)["status"] == "closed"
            rows = db.q("SELECT game,status FROM matches WHERE room_code = ? ORDER BY game", code)
            expected = ([{"game": 0, "status": "void"}] if when in ("before_first", "late_start", "official_abort") else
                        [{"game": 0, "status": "finished"}, {"game": 1, "status": "void"}])
            assert rows == expected
        finally:
            await hub.stop()
            db.close()

    asyncio.run(exercise_room())
