"""Collect one fixed batch of paired practice games into the workbench replay archive."""
import argparse
from pathlib import Path
import shutil
import tempfile

from tank import online, versions
from tank.workspace import project_root, read_json, write_json

LANCER = "v-0081349ee9d189d06bc534ecdac22f731b4908714c90b45644a1962aea0583f5"
PROBES = {"approach": ("Approach", "直视时接近并开火"),
          "silent": ("SilentApproach", "同样接近，全程停火"),
          "retreat": ("Retreat", "直视时倒车并开火")}
BATCHES = {
    "fire": [(seed, name) for seed in (712101, 712117) for name in ("approach", "silent")],
    "distance": [(seed, name) for seed in (713029, 713053) for name in ("approach", "retreat")],
    "roaming": [(seed, name) for seed in (714079, 714091) for name in ("lancer", "retreat")],
    "dev": [(715121, "approach"), (715121, "retreat")],
    "test": [(728311, "lancer"), (728311, "approach"), (728419, "lancer"), (728419, "retreat")],
}


def snapshots(root, lancer=LANCER):
    versions.verify(root, lancer)
    with tempfile.TemporaryDirectory() as folder:
        source = Path(folder)
        shutil.copyfile(versions.snapshot_data(root, lancer) / "bot.py", source / "navigation.py")
        shutil.copyfile(Path(__file__).with_name("boss_practice_probe.py"), source / "bot.py")
        result = {"lancer": lancer}
        for name, (entry, note) in PROBES.items():
            write_json(source / "submission.json", {"runtime": "python", "entrypoint": f"bot.py:{entry}"})
            result[name] = versions.create_from_directory(root, source, "行为采样", note)["id"]
        return result


def collect(root, batch, lancer=LANCER):
    keys = snapshots(root, lancer)
    client = online.OnlineClient.login(root)
    selected = []
    for seed, name in BATCHES[batch]:
        key = keys[name]
        folder = root / ".local/online/practice" / key / "boss"
        records = sorted((r for p in folder.glob("*.json") if (r := read_json(p))["seed"] == seed),
                         key=lambda r: r["id"])
        if {r["side"] for r in records} != {"red", "blue"}:
            print(batch, seed, name, flush=True)
            new = online.practice(root, key, client=client, level="boss", seed=seed, games=2)
            records.extend(new["matches"])
        for side in ("red", "blue"):
            record = next(r for r in records if r["side"] == side)
            if record["map_seed"] != seed or record["version_id"] != key or record["level"] != "boss":
                raise ValueError(f"对局 {record['id']} 的地图、策略或档位与采样计划不符")
            if not online.practice_replay_path(root, record["id"]).is_file():
                online.replay(root, record["id"], client=client)
            selected.append({"id": record["id"], "version_id": key, "probe": name, "seed": seed, "side": side})
        print("archived", [r["id"] for r in selected[-2:]], flush=True)
    write_json(root / ".local/behavior" / f"batch-{batch}.json", selected)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", choices=BATCHES, nargs="+")
    parser.add_argument("--lancer", default=LANCER, help="提供 Lancer 导航类的游击策略快照")
    args = parser.parse_args()
    for batch in args.batch:
        collect(project_root(), batch, args.lancer)
