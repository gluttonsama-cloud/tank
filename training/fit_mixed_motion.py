"""Fit an old-only control and a mixed replay/SDK movement clone from a frozen manifest."""
import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from tank import versions
from tank.workspace import project_root, read_json, write_json
from training.sdk_motion_data import motion_rows


def read_game(row):
    raw = Path(row["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError(f"对局文件已变化 {row['id']}")
    return motion_rows(json.loads(gzip.decompress(raw)), row["source"], row["side"])


def heads(samples, rows):
    from sklearn.ensemble import ExtraTreesRegressor

    x = np.concatenate([s[0] for s in samples])
    y = np.concatenate([s[1] for s in samples])
    risk = np.concatenate([s[2] for s in samples])
    layout_games = Counter(r["grid"] for r in rows)
    w = np.concatenate([np.full(len(s[0]), 1 / (layout_games[r["grid"]] * len(s[0])))
                        for s, r in zip(samples, rows, strict=True)])
    w *= np.where(risk, 2., 1.)
    result = []
    for axis in range(2):
        sign = np.digitize(y[:, axis], [-.12, .12])
        modes = np.eye(3, dtype=np.float32)[sign]
        label = np.column_stack((modes, modes * y[:, axis, None]))
        model = ExtraTreesRegressor(n_estimators=32, max_depth=16, min_samples_leaf=9, max_features=.75,
                                    random_state=215 + axis, n_jobs=4)
        model.fit(x, label, sample_weight=w / w.mean())
        result.append(model)
    return result


def pack(models):
    arrays = {k: [] for k in ("feature", "threshold", "left", "right", "value")}
    roots, offset, depth = [], 0, 0
    for model in models:
        for estimator in model.estimators_:
            tree = estimator.tree_
            roots.append(offset)
            depth = max(depth, tree.max_depth)
            leaf = tree.children_left == -1
            arrays["feature"].append(np.where(leaf, 0, tree.feature).astype(np.int16))
            arrays["threshold"].append(tree.threshold)
            for key, child in (("left", tree.children_left), ("right", tree.children_right)):
                arrays[key].append((offset + np.where(leaf, np.arange(tree.node_count), child)).astype(np.int32))
            arrays["value"].append(tree.value[:, :, 0].astype(np.float32))
            offset += tree.node_count
    return {k: np.concatenate(v) for k, v in arrays.items()} | {"roots": np.array(roots), "depth": depth}


def actions(models, x):
    axes = []
    for head in models:
        votes = head.predict(x)
        mode = votes[:, :3].argmax(axis=1)
        i = np.arange(len(x))
        axes.append(votes[i, mode + 3] / votes[i, mode])
    return np.column_stack(axes).clip(-1, 1)


def evaluate(models, samples, rows):
    grids = {}
    for (x, y, risk), row in zip(samples, rows, strict=True):
        delta = abs(actions(models, x) - y)
        for group, mask in (("all", np.ones(len(x), dtype=bool)), ("threat", risk)):
            if mask.any():
                grouped = grids.setdefault((row["source"], group), {})
                grouped.setdefault(row["grid"], []).append(delta[mask])
    report = {}
    for (source, group), by_grid in grids.items():
        batches = [np.concatenate(v) for v in by_grid.values()]
        forward, turn = np.mean([batch.mean(axis=0) for batch in batches], axis=0)
        report[f"{source}_{group}"] = {"maps": len(batches), "frames": sum(map(len, batches)),
                                       "forward_mae": forward, "turn_mae": turn}
    return report


def run(manifest_path, out):
    root = project_root()
    meta = read_json(manifest_path)
    train, dev = meta["train"], meta["dev"]
    if {r["grid"] for r in train} & {r["grid"] for r in dev}:
        raise ValueError("训练与开发使用了同一布局")
    if len({r["id"] for r in train + dev}) != len(train + dev):
        raise ValueError("混合清单中存在重复对局")
    with ProcessPoolExecutor(max_workers=4) as pool:
        samples = list(pool.map(read_game, train + dev, chunksize=2))
    fit = samples[:len(train)]
    test = samples[len(train):]
    print("Prepared", sum(len(s[0]) for s in fit), "training intervals", flush=True)
    old = [i for i, r in enumerate(train) if r["source"] == "replay"]
    control = heads([fit[i] for i in old], [train[i] for i in old])
    reports = {"old_only": evaluate(control, test, dev)}
    del control
    models = heads(fit, train)
    reports["mixed"] = evaluate(models, test, dev)
    reports["training_frames"] = {source: sum(len(s[0]) for s, r in zip(fit, train, strict=True)
                                               if r["source"] == source) for source in ("sdk", "replay")}
    folder = out / "strategy"
    folder.mkdir(parents=True, exist_ok=True)
    frozen = versions.snapshot_data(root, meta["base"])
    for name, dest in (("weights.npz", "weights.npz"), ("bot.py", "boss_movement.py"),
                       ("boss_sparring_policy.py", "boss_sparring_policy.py")):
        shutil.copyfile(frozen / name, folder / dest)
    for name in ("sdk_motion_policy.py", "mixed_motion_bot.py"):
        shutil.copyfile(Path(__file__).with_name(name), folder / name)
    np.savez_compressed(folder / "mixed_motion.npz", **pack(models))
    write_json(folder / "submission.json", {"runtime": "python", "entrypoint": "mixed_motion_bot.py:MixedMotionSparring"})
    write_json(folder / "training.json", {"base": meta["base"], "policy": "mixed-motion",
                                         "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                                         "train": train, "holdout": dev})
    record = versions.create_from_directory(root, folder, "boss-sparring", "混合官网回放与 SDK 观测，模仿走位及来弹反应")
    reports["candidate"] = record["id"]
    write_json(out / "motion-fit.json", reports)
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.out)
