import gzip
import hashlib
import json

from seedcup.game import rules
from seedcup.game.maps import generate_map

from tank import cycle_experience as experience
from tank import sampling_data
from tank.online import PRACTICE_NAMES
from tank.workspace import write_json


def _player(team, name):
    return {"id": team, "name": name, "bot": team is None}


def _result(ticks=1):
    return {"ticks": ticks, "reason": "kill", "winner_side": "red",
            "scores": [{"total": 140}, {"total": 12}]}


def _replay(mid, seed, *, ladder=False):
    game_map = generate_map(seed)
    red = [2.5, 5.5, 0, 100, 1, 0, 0, 0, 0, 0, 0, 0]
    blue = [29.5, 5.5, 180, 90, 1, 0, 0, 0, 0, 0, 0, 0]
    return {
        "format": 1, "rules": rules.as_dict(), "map": game_map.to_dict(),
        "match": {"id": mid, "kind": "ladder" if ladder else "room", "room": "ABCDE",
                  "red": _player(57, "Home"), "blue": _player(70, "Visitor") if ladder
                  else _player(None, PRACTICE_NAMES["boss"])},
        "result": _result(),
        "frames": [[0, red, blue, [], []], [1, red, blue, [[4, 5.5, 1, 0, 0]], [["fire", 0, 3.0, 5.5]]]],
    }


def _write_gzip(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = gzip.compress(json.dumps(data).encode(), mtime=0)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _sdk(replay, side, version):
    mid = replay["match"]["id"]
    game_map = generate_map(replay["map"]["seed"])
    rule_set = replay["rules"]
    start = {
        "kind": "room", "side": side, "match_id": mid, "rules": rule_set,
        "opponent": PRACTICE_NAMES["boss"],
        "map": {"grid": game_map.rows(), "seed": game_map.seed, "width": rules.W},
        "spawn": {"x": 2.5, "y": 5.5, "heading": 0},
    }
    sent = {"forward": 0, "turn": 0, "fire": True}
    obs = {
        "frame": 1, "match_id": mid, "zone": {"radius": 20},
        "self": {"x": 2.5, "y": 5.5, "heading": 0, "hp": 100, "alive": True, "reload": 1},
        "enemy": {"x": 29.5, "y": 5.5, "heading": 180, "hp": 90, "alive": True},
        "last_action": sent | {"frame": 0, "source_frame": 0}, "shells": [{"mine": True}],
    }
    practice = {"level": "boss", "room": "ABCDE", "team_id": 57, "version_id": version}
    return {"format": "seedcup-sdk-observations-v1", "start": start,
            "end": {"match_id": mid, "side": side, "result": replay["result"]},
            "frames": [[obs, sent]], "practice": practice}


def test_refresh_deduplicates_provenance_and_limits_verified_stats(tmp_path):
    replay = _replay(11, 31)
    path = ".local/online/replays/11.json.gz"
    digest = _write_gzip(tmp_path / path, replay)
    grid = sampling_data.layout_hash(replay["map"])
    version = "v-" + "a" * 64
    write_json(tmp_path / ".local/versions" / version / "version.json",
               {"name": "lancer", "id": version, "note": "转向预测"})
    write_json(tmp_path / "projects/lancer.json", {"name": "游击规划", "sources": ["lancer"], "purpose": "pve"})
    sdk = _sdk(replay, "red", version)
    observation_file = ".local/sampling/observations/11.json.gz"
    observation_sha256 = _write_gzip(tmp_path / observation_file, sdk)
    observation = {**sdk["practice"], "id": 11, "side": "red", "result": replay["result"],
                   "observation_file": observation_file, "observation_sha256": observation_sha256, "map_seed": 31}
    example = {"id": 11, "path": path, "sha256": digest, "target": "boss", "target_side": "blue",
               "seed": 31, "grid_sha256": grid, "version_id": version}
    with sampling_data.database(tmp_path) as db:
        db.execute("INSERT INTO examples VALUES (?, ?, ?, ?)", (11, "boss", json.dumps(example), "{}"))
        db.execute("INSERT INTO sdk_observations VALUES (?, ?, ?)", (11, "boss", json.dumps(observation)))
        db.execute("INSERT INTO layouts VALUES (?, ?)", (grid, "blind"))

    ladder = _replay(20, 47, ladder=True)
    ladder_grid = sampling_data.layout_hash(ladder["map"])
    with sampling_data.database(tmp_path) as db:
        db.execute("INSERT INTO layouts VALUES (?, 'development')", (ladder_grid,))
    ladder_path = tmp_path / ".local/cycle/ladder/replays/20.json.gz"
    ladder_digest = _write_gzip(ladder_path, ladder)
    ladder_record = {"game_id": 20, "series_id": 80, "team_id": 70, "opponent_id": 57,
                     "own_side": "blue", "version_id": version, "submission_id": None, "role": "coverage",
                     "status": "archived", "replay_path": str(ladder_path), "replay_sha256": ladder_digest}
    ladder_metadata = tmp_path / ".local/cycle/ladder/games/20.json"
    write_json(ladder_metadata, ladder_record)

    assert experience.refresh(tmp_path, limit=1) == {
        "indexed": 1, "errors": [], "games": 1, "failed_games": 0, "remaining": 1,
    }
    assert experience.refresh(tmp_path, limit=1)["games"] == 2
    assert experience.refresh(tmp_path, limit=1)["indexed"] == 0
    assert experience.indexed_grids(tmp_path) == {grid, sampling_data.layout_hash(ladder["map"])}
    overall = experience.summary(tmp_path)
    cursor = overall["cursor"]
    assert overall["sources"] == {"official_replay": 1, "sdk_observations": 1, "ladder_replay": 1}
    assert overall["team_matchups"] == 2
    assert overall["event_counts"]["first_engagement"] == 2
    assert overall["coverage_by_scope"]["pvp"]["games"] == 1

    report = experience.summary(tmp_path, "游击规划")
    assert report["games"] == 1
    assert report["version_stats"][0]["wins"] == 1
    opening = report["representative_events"]["first_engagement"][0]
    assert opening["game_id"] == 11
    assert opening["map"]["split"] == "blind"
    assert opening["map"]["eligible_for_training"] is False
    assert opening["frames"][1]["tanks"]["red"]["actions"]["last_applied"]["source_frame"] == 0
    assert report["previously_tried_changes"][0]["note"] == "转向预测"
    pvp = experience.summary(tmp_path, "pvp")
    assert pvp["version_stats"] == []
    assert any(need["target"] == "team:57" for need in pvp["sampling_needs"])
    assert pvp["top_findings"][0]["event"] == "first_engagement"
    assert pvp["top_findings"][0]["game_ids"] == [20]
    assert pvp["top_findings"][0]["losses"] == 1
    assert not experience.summary(tmp_path, since=cursor)["top_findings"]
    assert not overall["training_focus"]

    with sampling_data.database(tmp_path) as db:
        db.execute("UPDATE layouts SET split='train' WHERE grid=?", (ladder_grid,))
    series_path = tmp_path / ".local/cycle/ladder/series/80.json"
    series = {"series_id": 80, "team_id": 70, "opponent_id": 57, "status": "finished",
              "games": [{"match_id": 20}]}
    write_json(series_path, series | {"games": series["games"] + [{"match_id": 40}]})
    assert experience.refresh(tmp_path)["indexed"] == 1
    incremental = experience.summary(tmp_path, since=cursor)
    assert incremental["games"] == 1
    assert incremental["coverage_by_scope"]["practice"]["games"] == 0
    assert not experience.summary(tmp_path, "pvp")["training_focus"]
    write_json(series_path, series)
    focused = experience.summary(tmp_path, "pvp", training_team_id=70, training_series_ids=[80])
    assert focused["training_focus"][0]["entries"][0]["map_seed"] == 47
    assert focused["training_focus"][0]["entries"][0]["series_id"] == 80
    assert not experience.summary(tmp_path, "pvp", training_team_id=57)["training_focus"]
    cursor = focused["cursor"]

    write_json(ladder_metadata, ladder_record | {"submission_id": 51})
    assert experience.refresh(tmp_path, limit=1)["indexed"] == 1
    pvp = experience.summary(tmp_path, "pvp", since=str(cursor))
    assert pvp["cursor"] > cursor
    assert pvp["games"] == 1
    assert pvp["version_stats"][0]["losses"] == 1
    assert pvp["version_stats"][0]["sides"] == {"blue": 1}
    assert experience.summary(tmp_path, "pvp", since=pvp["cursor"])["games"] == 0
    with sampling_data.database(tmp_path) as db:
        db.execute("INSERT INTO sampling_holdouts VALUES (?)", (ladder_grid,))
    assert not experience.summary(tmp_path, "pvp")["training_focus"]

    write_json(ladder_metadata, ladder_record | {"opponent_id": 88})
    invalid = experience.refresh(tmp_path, limit=1)
    assert [row["game_id"] for row in invalid["errors"]] == [20]
    assert experience.summary(tmp_path, "pvp")["games"] == 0
    assert experience.refresh(tmp_path, limit=1)["indexed"] == 0


def test_pvp_ignores_layout_split_and_isolates_real_pve_overlap(tmp_path):
    with sampling_data.database(tmp_path) as db:
        for mid, seed, split in ((21, 47, "train"), (22, 48, "development")):
            replay = _replay(mid, seed, ladder=True)
            path = tmp_path / f".local/cycle/ladder/replays/{mid}.json.gz"
            digest = _write_gzip(path, replay)
            write_json(tmp_path / f".local/cycle/ladder/games/{mid}.json",
                       {"game_id": mid, "series_id": 81, "team_id": 70, "opponent_id": 57,
                        "own_side": "blue", "version_id": None, "submission_id": None, "role": "external",
                        "status": "archived", "replay_path": str(path), "replay_sha256": digest})
            db.execute("INSERT INTO layouts VALUES (?, ?)", (sampling_data.layout_hash(replay["map"]), split))
    write_json(tmp_path / ".local/cycle/ladder/series/81.json",
               {"series_id": 81, "team_id": 70, "opponent_id": 57, "status": "finished",
                "games": [{"match_id": 21}, {"match_id": 22}]})
    assert experience.refresh(tmp_path)["games"] == 2
    summary = experience.summary(tmp_path, "pvp")
    assert summary["top_findings"][0]["distinct_maps"] == 2
    assert summary["pvp_splits"]["train"]["training_series"] == 1
    assert summary["training_focus"][0]["seen_maps"] == 2
    assert summary["training_focus"][0]["entries"][0]["split_basis"] == "opponent_team"

    _write_gzip(tmp_path / ".local/online/replays/60.json.gz", _replay(60, 48))
    assert experience.refresh(tmp_path)["indexed"] == 1
    summary = experience.summary(tmp_path, "pvp")
    assert summary["pvp_splits"]["train"]["isolated_series"] == 1
    assert summary["pvp_splits"]["train"]["training_series"] == 0
    assert summary["training_focus"] == []


def test_pvp_groups_repeat_opponents_and_keeps_unfinished_series_out(tmp_path):
    teams = {split: next(team for team in range(1, 100) if experience._opponent_split(team) == split)
             for split in ("train", "dev", "test")}
    for mid, split, finished in ((60, "train", True), (61, "train", True), (62, "train", False),
                                 (63, "dev", True), (64, "test", True)):
        team = teams[split]
        replay = _replay(mid, mid + 100, ladder=True)
        replay["match"]["red"]["id"] = team
        path = tmp_path / f".local/cycle/ladder/replays/{mid}.json.gz"
        digest = _write_gzip(path, replay)
        write_json(tmp_path / f".local/cycle/ladder/games/{mid}.json",
                   {"game_id": mid, "series_id": mid, "team_id": 70, "opponent_id": team,
                    "own_side": "blue", "version_id": None, "submission_id": None, "role": "external",
                    "status": "archived", "replay_path": str(path), "replay_sha256": digest})
        write_json(tmp_path / f".local/cycle/ladder/series/{mid}.json",
                   {"series_id": mid, "team_id": 70, "opponent_id": team,
                    "status": "finished" if finished else "running", "games": [{"match_id": mid}]})
    assert experience.refresh(tmp_path)["games"] == 5
    summary = experience.summary(tmp_path, "pvp")
    for split, values in summary["pvp_splits"].items():
        assert values["opponent_ids"] == [teams[split]]
        assert values["training_series"] == (2 if split == "train" else 0)
        assert values["replay_games"] == (3 if split == "train" else 1)
    entries = [entry for focus in summary["training_focus"] for entry in focus["entries"]]
    assert {entry["game_id"] for entry in entries} == {60, 61}
    assert {entry["split"] for entry in entries} == {"train"}
    assert all(need["target"] == f"team:{teams['train']}" for need in summary["sampling_needs"])


def test_extract_five_event_types_and_unknown_discontinuous_streaks():
    rule_set = rules.as_dict()
    rule_set["zone"].update(shrink_start=0, shrink_end=1)
    game_map = generate_map(31)
    frames = []
    for tick in range(14):
        red_x = 10.0 if tick <= 8 else 7.0
        blue_x = 12.0 if tick <= 11 else 8.0
        fired = (0,) if tick in (1, 3, 5) else ()
        damage = (10 if tick >= 7 else 0, 10 if tick >= 8 else 0)
        red = (red_x, 12.0, 0, 100 - damage[1], True, 0)
        blue = (blue_x, 12.0, 180, 100 - damage[0], True, 0)
        frames.append(experience.Frame(tick, (red, blue), (), 20 if tick < 10 else 5, fired, damage))

    examples, counts = experience._extract_events(frames, game_map, rule_set, {})
    assert all(counts[event] for event in experience.EVENTS)
    assert examples["consecutive_misses"][0]["last_three_fire_frames"] == [1, 3, 5]
    assert examples["damage_exchange"][0]["damage_by_side"] == {"red": 10, "blue": 10}
    assert examples["breaking_contact"][0]["main_separating_side"] == "red"
    assert examples["zone_displacement"][0]["transition"] == "left_safe_zone"

    dropped = [frame for frame in frames if frame.tick != 2]
    examples, _ = experience._extract_events(dropped, game_map, rule_set, {})
    assert "consecutive_misses" not in examples
