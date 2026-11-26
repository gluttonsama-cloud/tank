import gzip
import json

import pytest
from seedcup.game import generate_map, rules
from seedcup.game.replay import FORMAT_VERSION

from tank import online
from tank import sampling_data as data
from tank import sampling_opportunities as opportunities


def frame(tick, red_x=10, blue_x=13, *, red_hp=100, blue_hp=100, red_hits=0, blue_hits=0):
    red = [red_x, 10, 0, red_hp, int(red_hp > 0), 0, 0, 0, 0, red_hits, 0, 0]
    blue = [blue_x, 10, 180, blue_hp, int(blue_hp > 0), 0, 0, 0, 0, blue_hits, 0, 0]
    return [tick, red, blue, [], []]


def replay(frames, result=None):
    return {"format": FORMAT_VERSION, "rules": rules.as_dict(), "frames": frames, "result": result}


def test_first_close_limits_trigger_and_separates_projectile_hits_from_zone_damage():
    frames = [frame(0, blue_x=13.01), frame(1, blue_x=13.01, red_hp=90, blue_hits=1),
              frame(2, red_hp=89, blue_hits=1), frame(3, red_hp=79, blue_hits=2),
              frame(4, red_hp=79, blue_hp=90, red_hits=1, blue_hits=2)]
    red = opportunities.extract(replay(frames), "red")
    blue = opportunities.extract(replay(frames), "blue")
    assert red["first_close"]["trigger_frame"] == 2
    assert red["first_close"]["trigger_distance_cells"] == 3
    assert red["first_close_hit"]["trigger_frame"] == 3
    assert red["first_close_hit"]["trigger_target_hp"] == 79
    assert blue["first_close_hit"]["trigger_frame"] == 4
    frames[3] = frame(3, red_hp=0, blue_hits=2)
    assert opportunities.extract(replay(frames), "red")["first_close_hit"] is None


@pytest.mark.parametrize(("gain", "outcome"), [(0, "stayed"), (.99, "stayed"), (1, "withdrawn")])
def test_same_five_cell_distance_can_have_different_movement_attribution(gain, outcome):
    deadline = 3 * rules.TICK_HZ
    frames = [frame(0), *(frame(tick, 10 - gain, 15 - gain) for tick in range(1, deadline + 1))]
    record = opportunities.extract(replay(frames), "red")["first_close"]
    assert record["outcome"] == outcome and record["target_gain_cells"] == gain
    assert record["end_frame"] == (1 if outcome == "withdrawn" else deadline)


def test_first_hit_starts_on_impact_and_is_not_replaced_by_a_later_hit():
    deadline = 1 + 3 * rules.TICK_HZ
    frames = [frame(0, blue_x=12.75), frame(1, 9.75, 12.75, red_hp=90, blue_hits=1)]
    frames.extend(frame(tick, 9, 14, red_hp=90, blue_hits=1) for tick in range(2, deadline + 1))
    frames.extend([frame(deadline + 1, 9, 12, red_hp=80, blue_hits=2),
                   frame(deadline + 2, 8, 13, red_hp=80, blue_hits=2)])
    records = opportunities.extract(replay(frames), "red")
    assert records["first_close"]["outcome"] == "withdrawn"
    first_hit = records["first_close_hit"]
    assert first_hit["target_gain_cells"] == .75
    assert first_hit["outcome"] == "stayed" and first_hit["end_frame"] == deadline


def test_window_has_signed_net_contribution_and_includes_its_deadline():
    deadline = 3 * rules.TICK_HZ
    frames = [frame(0), frame(1, 8.75, 13)]
    frames.extend(frame(tick, 9.25, 13) for tick in range(2, deadline))
    frames.append(frame(deadline, 9.25, 14.25))
    record = opportunities.extract(replay(frames), "red")["first_close"]
    assert record["outcome"] == "stayed" and record["target_gain_cells"] == .75
    frames[-1] = frame(deadline, 9, 14)
    record = opportunities.extract(replay(frames), "red")["first_close"]
    assert record["outcome"] == "withdrawn" and record["end_frame"] == deadline
    frames.insert(-1, frame(deadline))
    frames[-1][0] += 1
    assert opportunities.extract(replay(frames), "red")["first_close"]["outcome"] == "stayed"


@pytest.mark.parametrize(("frames", "result", "end_frame", "reason"), [
    ([frame(0), frame(1, 9, 14, red_hp=0)], {"ticks": 1, "reason": "kill"}, 1, "kill"),
    ([frame(0), frame(1)], {"ticks": 1, "reason": "time"}, 1, "time"),
    ([frame(0), frame(1)], None, 1, "missing_frames"),
    ([frame(0), frame(2, 9, 14)], {"ticks": 2, "reason": "time"}, 0, "missing_frames"),
])
def test_end_frame_and_reason_remain_distinct_from_a_completed_negative(frames, result, end_frame, reason):
    record = opportunities.extract(replay(frames, result), "red")["first_close"]
    assert (record["outcome"], record["end_frame"], record["end_reason"]) == ("ended", end_frame, reason)


def test_comparisons_match_same_side_game_and_layout_with_both_preconditions():
    anchor = {"frame": 5, "distance": 2.5, "probe_heading_error_deg": 1,
              "target_heading_error_deg": -175, "target_retreat_lanes": ["straight"]}

    def timing(fire, source=5):
        return {"source_frame": source, "command_fire": fire, "applied_fire": fire, "confirmed_uses": 1}

    def case(mid, arm, *, map_index=0, game=0, side="red", grid="map", match=True, start=anchor, version="v",
             execution=None):
        kind = "intervention" if arm.startswith("window-") else "repeat"
        return {"id": mid, "probe": arm, "probe_shots": int(arm == "window-fire"),
                "version_id": version, "target_side": side, "game_no": game,
                "grid_sha256": grid, "sdk_timing": {"intervention": execution},
                "collection": {"kind": kind, "run": "run", "map_index": map_index,
                    "role": "intervention" if kind == "intervention" else arm, "layout_match": match,
                    "intervention_anchor": start}}

    entries = [case(1, "window-fire", execution=timing(True)),
        case(2, "window-hold", execution=timing(False), start=anchor | {
            "distance": 2.75, "target_heading_error_deg": 175}),
        case(3, "window-fire", side="blue", execution=timing(True)),
        case(4, "window-hold", side="blue", execution=timing(False), start=anchor | {
            "distance": 2.76, "probe_heading_error_deg": 12, "target_retreat_lanes": []}),
        case(5, "window-hold", game=1, start=None),
        case(6, "reference", map_index=1), case(7, "repeat", map_index=1),
        case(8, "reference", map_index=2), case(9, "repeat", map_index=2, match=False, version="other"),
        case(10, "window-hold", map_index=3, grid="other"), case(11, "window-fire", map_index=3),
        case(12, "window-fire", map_index=4, execution=timing(True, source=6)),
        case(13, "window-hold", map_index=4, execution=timing(False) | {"applied_fire": True})]
    pairs, counts = opportunities.comparisons(entries)
    assert pairs[1] == pairs[2] and pairs[1]["state"] == "confirmed"
    assert pairs[1]["probe_side"] == "blue"
    assert pairs[1]["geometric_match"] and pairs[1]["execution_confirmed"]
    assert pairs[3]["reasons"] == ["distance_cells_mismatch", "probe_heading_deg_mismatch", "retreat_lanes_mismatch"]
    assert not pairs[3]["geometric_match"] and pairs[3]["execution_confirmed"]
    assert pairs[5]["reasons"] == ["missing_window-fire", "missing_anchor_window-hold", "execution_unconfirmed"]
    assert pairs[6]["state"] == "matched"
    assert pairs[8]["reasons"] == ["layout_mismatch", "version_mismatch"]
    assert pairs[10]["reasons"] == ["layout_mismatch", "execution_unconfirmed"]
    assert pairs[12]["geometric_match"] and not pairs[12]["execution_confirmed"]
    assert pairs[12]["state"] == "coverage_only"
    assert pairs[12]["execution"]["window-fire"]["reasons"] == ["source_frame_mismatch"]
    assert pairs[12]["execution"]["window-hold"]["reasons"] == ["applied_fire_mismatch"]
    assert counts["intervention"]["geometry_matched"] == 2 and counts["intervention"]["coverage_only"] == 4
    assert counts["intervention"]["execution_confirmed"] == 2
    assert counts["intervention"]["confirmed_pairs"] == 1
    assert counts["repeat"]["matched"] == counts["repeat"]["coverage_only"] == 1
    for shots in (0, 2):
        cases = [entries[0] | {"probe_shots": shots}, entries[1]]
        pairs, counts = opportunities.comparisons(cases)
        assert counts["intervention"]["confirmed_pairs"] == 0 and pairs[1]["state"] == "coverage_only"
        assert pairs[1]["execution"]["window-fire"]["reasons"] == ["replay_shots_mismatch"]


def test_dataset_backfills_legacy_and_excludes_reservations_from_train(tmp_path, monkeypatch):
    maps = [generate_map(seed).to_dict() for seed in (0, 1, 2, 21)]
    train, reserved, missing, *_ = [m for m in maps if data.default_split(data.layout_hash(m)) == "train"]
    blind = next(m for m in maps if data.default_split(data.layout_hash(m)) == "blind")
    train_grid = data.layout_hash(train)
    anchor = {"frame": 0, "probe_side": "blue"}
    monkeypatch.setattr(data, "intervention_anchor", lambda _, side: anchor | {"probe_side": side})

    def archive(mid, game_map, **kwargs):
        example = replay([frame(0), frame(1, red_hp=90, blue_hits=1)], {"ticks": 1, "reason": "time"})
        example["map"] = game_map
        example["match"] = {"id": mid, "kind": "room", "room": "TEST", "game": 0,
                            "red": {"id": None, "name": online.PRACTICE_NAMES["boss"], "bot": True},
                            "blue": {"id": 100, "name": "采样方", "bot": False}}
        return data.ingest(tmp_path, gzip.compress(json.dumps(example).encode(), mtime=0), "boss", **kwargs)

    collection = {"planned_grid": train_grid, "planned_split": "train", "repeat": 2,
                  "kind": "intervention", "role": "intervention", "map_index": 0, "run": "run"}
    archive(1, train)
    comparison = archive(2, train, probe="window-hold", collection=collection)
    archive(2, train)  # A later import preserves its collection assignment.
    assert comparison["collection"] == collection | {
        "actual_grid": train_grid, "layout_match": True, "intervention_anchor": anchor}
    archive(3, reserved)
    archive(4, blind)
    archive(5, train, collection=collection | {"planned_split": "development", "role": "reference"})
    archive(6, train, collection=collection | {"planned_grid": "a" * 64, "role": "reference"})
    archive(7, missing)
    archive(8, train, probe="holdout-orbit")
    practice = {"id": 9, "side": "blue", "level": "boss", "team_id": 100, "map_seed": train["seed"],
                "room": "TEST", "version_id": "reserved-v1"}
    archive(9, train, probe="another-name", practice=practice)
    with data.database(tmp_path) as db:
        db.execute("INSERT INTO sampling_holdouts VALUES (?)", (data.layout_hash(reserved),))
        db.execute("INSERT INTO sampling_reserved_versions VALUES ('reserved-v1')")
        db.execute("UPDATE examples SET coverage=json_remove(coverage,'$.opportunities_v1') WHERE id IN (1,3,7)")
        db.execute("INSERT INTO runs(id,plan,state,updated) VALUES ('run','{}','running',0)")
        db.execute("INSERT INTO slots VALUES ('run',1,'blue',2)")
    online.practice_replay_path(tmp_path, 7).unlink()

    history = opportunities.history(tmp_path)
    assert len(history) == 2 and all(row[0] == train_grid for row in history)
    assert [row[2]["first_close_hit"]["outcome"] for row in history] == ["ended", "ended"]
    with data.database(tmp_path) as db:
        cached = dict(db.execute("SELECT id,json_type(coverage,'$.opportunities_v1') FROM examples"))
        assert cached[1] == "object" and cached[3] is None and cached[7] is None
    online.practice_replay_path(tmp_path, 1).unlink()
    assert opportunities.history(tmp_path) == history

    output = tmp_path / "train.json"
    summary = data.dataset(tmp_path, "boss", split="train", output=output)
    entries = json.loads(output.read_text(encoding="utf-8"))
    assert [row["id"] for row in entries] == [1, 2, 7]
    assert all(row["eligible_for_training"] for row in entries)
    assert entries[1]["collection"] == comparison["collection"] and entries[1]["repeat"] == 2
    assert entries[2]["opportunities_v1"] is None
    assert summary["opportunities_v1"]["first_close_hit"]["ended"] == {"games": 2, "maps": 1}
    assert summary["opportunities_unavailable_games"] == 1

    data.dataset(tmp_path, "boss", output=output)
    entries = json.loads(output.read_text(encoding="utf-8"))
    assert [row["id"] for row in entries if not row["eligible_for_training"]] == [3, 4, 5, 6, 8, 9]

    comparison_summary = data.dataset(tmp_path, "boss", run="run", output=output)["comparisons"]
    assert comparison_summary["intervention"]["coverage_only"] == 1
    entry, = json.loads(output.read_text(encoding="utf-8"))
    assert entry["comparison"]["reasons"] == ["missing_window-fire", "execution_unconfirmed"]

    scheduled = "sample-" + "1" * 32
    with data.database(tmp_path) as db:
        db.execute("INSERT INTO runs(id,plan,state,updated) VALUES (?,?,'running',0)",
                   (scheduled, json.dumps({"width": 3})))
        db.execute("INSERT INTO sampling_maps VALUES (?,0,'{}'), (?,1,?)",
                   (scheduled, scheduled, json.dumps({"protocol": 2})))
    valid = practice | {"version_id": "sample-v1"}
    duplicate = valid | {"id": 10, "case_tag": f"{scheduled}-4"}
    assert archive(10, train, practice=duplicate)["unassigned_collection"]
    assert archive(10, train)["unassigned_collection"]
    assert opportunities.history(tmp_path) == history
    assigned = archive(10, train, practice=duplicate, collection=collection)
    assert "unassigned_collection" not in assigned
    assert len(opportunities.history(tmp_path)) == 3
    legacy = archive(11, train, practice=valid | {"id": 11, "case_tag": f"{scheduled}-2"})
    assert "unassigned_collection" not in legacy
