import gzip
import hashlib
import json

from tank.cli import execute, parser
from tank.sampling_data import database, sdk_layout_hash
from tank.workspace import read_json


def test_export_sdk_observations_and_filter_training_layouts(tmp_path):
    run = "sample-" + "a" * 32
    other_run = "sample-" + "b" * 32
    plan = {"mode": "active", "target": "boss", "maps": 3, "width": 1}
    start = {"map": {"grid": [[1, 0], [0, 1]]}}
    grid = sdk_layout_hash(start)
    raw = gzip.compress(json.dumps({"start": start}).encode())
    path = tmp_path / "archive.json.gz"
    path.write_bytes(raw)
    other_grid = "f" * 64
    holdout_grid = "a" * 64
    base = {"grid_sha256": grid, "target_side": "blue", "side": "red", "probe": "peek",
            "version_id": "v1", "observation_file": path.name,
            "observation_sha256": hashlib.sha256(raw).hexdigest()}
    entries = [
        {key: value for key, value in base.items() if key not in ("grid_sha256", "target_side")},
        base | {"version_id": "reserved"},
        base | {"grid_sha256": holdout_grid},
        base | {"collection": {"planned_grid": other_grid, "planned_split": "train"}},
        base | {"collection": {"planned_grid": grid, "planned_split": "blind"}},
        base | {"grid_sha256": other_grid},
    ]
    with database(tmp_path) as db:
        db.executemany("INSERT INTO runs(id,plan,state,updated) VALUES (?,?,'completed',0)",
                       ((key, json.dumps(plan)) for key in (run, other_run)))
        db.executemany("INSERT INTO layouts VALUES (?,?)",
                       ((grid, "train"), (other_grid, "blind"), (holdout_grid, "train")))
        db.execute("INSERT INTO sampling_holdouts VALUES (?)", (holdout_grid,))
        db.execute("INSERT INTO sampling_reserved_versions VALUES ('reserved')")
        for mid, entry in enumerate(entries, start=1):
            db.execute("INSERT INTO sdk_observations VALUES (?,'boss',?)", (mid, json.dumps(entry | {"id": mid})))
            db.execute("INSERT INTO slots VALUES (?,?,?,?)", (run, mid, "red", mid))
        db.execute("INSERT INTO slots VALUES (?,0,'red',7)", (run,))
        db.execute("INSERT INTO sdk_observations VALUES (8,'boss',?)", (json.dumps(base | {"id": 8}),))
        db.execute("INSERT INTO slots VALUES (?,0,'red',8)", (other_run,))

    def command(*args):
        return execute(tmp_path, parser().parse_args(["sample", *args]))

    status = command("status", run)
    assert (status["games"], status["sdk_games"]) == (7, 6)
    output = tmp_path / "out.json"
    args = ("export", "--source", "sdk", "--target", "boss", "--run", run)
    summary = command(*args, "--out", str(output))
    assert (summary["source"], summary["games"], summary["maps"]) == ("sdk", 6, 3)
    manifest = read_json(output)
    assert [entry["eligible_for_training"] for entry in manifest] == [True, False, False, False, False, False]
    assert manifest[0]["grid_sha256"] == grid and manifest[0]["target_side"] == "blue"
    assert manifest[0]["observation_file"] == path.name
    assert manifest[3]["collection"]["actual_grid"] == grid
    assert not manifest[3]["collection"]["layout_match"]
    assert command(*args, "--split", "train", "--out", str(output))["games"] == 1
    assert [entry["id"] for entry in read_json(output)] == [1]
    assert command(*args, "--split", "blind")["games"] == 1
