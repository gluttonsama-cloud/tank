"""Fit a v31 motion forecast from frozen official replay splits."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from seedcup.game import rules
from seedcup.game.engine import SIDES
from seedcup.game.replay import FORMAT_VERSION, load

from tank import versions
from tank.sampling_data import layout_hash
from tank.workspace import project_root

if __package__:
    from .boss_forecast import basis
    from .boss_sparring_policy import ManeuverFeatures, decode, predict
    from .boss_trajectory import HORIZONS, REACH, TURN_STEP, forecast, rank_maneuvers
    from .fit_boss_sparring import (
        SPEED_THRESHOLD,
        boss_side,
        extract,
        map_frame_weights,
        replay_source,
        write_json,
    )
else:
    from boss_forecast import basis
    from boss_sparring_policy import ManeuverFeatures, decode, predict
    from boss_trajectory import HORIZONS, REACH, TURN_STEP, forecast, rank_maneuvers
    from fit_boss_sparring import (
        SPEED_THRESHOLD,
        boss_side,
        extract,
        map_frame_weights,
        replay_source,
        write_json,
    )


FORECAST_WIDTH = 192
RIDGE = 5.
ONSET_CLASSES = ("none", "left_0.3s", "right_0.3s", "left_1s", "right_1s", "left_2s", "right_2s")


def targets(frames, side):
    states = np.asarray([frame[side + 1][:3] for frame in frames])
    xy, heading = states[:, :2], states[:, 2]
    direction = 1 if side == 0 else -1
    angle = np.deg2rad(heading[:-1])
    c, s = np.cos(angle), np.sin(angle)
    positions = np.full((len(frames) - 1, len(HORIZONS), 2), np.nan, dtype=np.float32)
    for index, lag in enumerate(HORIZONS):
        dx, dy = (xy[lag:] - xy[:-lag]).T
        count = len(dx)
        positions[:count, index, 0] = (c[:count] * dx + s[:count] * dy) / REACH[index]
        positions[:count, index, 1] = direction * (c[:count] * dy - s[:count] * dx) / REACH[index]
    # The first 18-degree heading change, starting from the decision frame.
    # Leave incomplete windows out of the seven-way onset classification.
    onset = np.full(len(frames) - 1, -1, dtype=np.int8)
    complete = max(0, len(frames) - HORIZONS[-1])
    known = onset[:complete]
    known[:] = 0
    for lag in range(1, HORIZONS[-1] + 1):
        delta = ((heading[lag:lag + complete] - heading[:complete] + 180) % 360 - 180) * direction
        changed = (known == 0) & (np.abs(delta) >= TURN_STEP - .1)
        bucket = np.searchsorted(HORIZONS, lag)
        known[changed] = 2 * bucket + np.where(delta[changed] > 0, 2, 1)
    return positions, onset


def read_splits(folder, request):
    manifest_hash = request["params"]["manifest_sha256"]
    manifest_input, *inputs = request["inputs"]
    if manifest_input["sha256"] != manifest_hash:
        raise ValueError("冻结清单的 manifest_sha256 不匹配")
    manifest_path = folder / manifest_input["path"]
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_hash:
        raise ValueError("冻结清单内容发生变化")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = {group: manifest[group] for group in ("train", "dev")}
    all_entries = entries["train"] + entries["dev"]
    sources = {Path(source["path"]).name: source for source in inputs}
    if (len({row["id"] for row in all_entries}) != len(all_entries)
            or len(sources) != len(inputs) or len(all_entries) != len(inputs)):
        raise ValueError("训练、开发清单与回放输入必须一一对应")
    for key in ("seed", "grid_sha256"):
        if {row[key] for row in entries["train"]} & {row[key] for row in entries["dev"]}:
            raise ValueError(f"训练与开发地图重复 {key}")

    result = {}
    for group, rows in entries.items():
        games = []
        for row in rows:
            source = sources[Path(row["path"]).name]
            path = folder / source["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != source["sha256"] or digest != row["sha256"]:
                raise ValueError(f"回放 {row['id']} 的内容与冻结清单不符")
            replay = load(path)
            if replay["format"] != FORMAT_VERSION or replay["rules"] != rules.as_dict():
                raise ValueError(f"回放 {row['id']} 格式或规则与当前 SDK 不一致")
            if (replay["match"]["id"] != row["id"] or replay["map"]["seed"] != row["seed"]
                    or layout_hash(replay["map"]) != row["grid_sha256"]
                    or SIDES[boss_side(replay["match"])] != row["target_side"]
                    or replay["match"]["game"] != row["game_no"]):
                raise ValueError(f"回放 {row['id']} 的局号、地图或终极档出生侧与清单不符")
            games.append((replay, digest))
        result[group] = games
    return result


def sample(games, feature_type):
    features, actions, positions, onset = [], [], [], []
    for replay, _ in games:
        x, y = extract(replay, feature_type)
        future, timing = targets(replay["frames"], boss_side(replay["match"]))
        features.append(x)
        actions.append(y)
        positions.append(future)
        onset.append(timing)
    layouts = [replay["map"]["grid"] for replay, _ in games]
    weights = map_frame_weights(layouts, list(map(len, features)))
    return tuple(map(np.concatenate, (features, actions, positions, onset))) + (weights,)


def fit(data, seed):
    features, _, positions, onset, importance = data
    mean = np.average(features, axis=0, weights=importance)
    deviation = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=importance))
    scale = np.maximum(deviation, .08)
    x = np.clip((features - mean) / scale, -5, 5)
    rng = np.random.default_rng(seed % 2**64)
    projection = rng.normal(0, 1 / np.sqrt(x.shape[-1]), (x.shape[-1], FORECAST_WIDTH)).astype(np.float32)
    bias = rng.uniform(-1, 1, FORECAST_WIDTH).astype(np.float32)
    design = basis(x, projection, bias)
    root_weight = np.sqrt(importance)

    def ridge(valid, labels):
        a = design[valid] * root_weight[valid, None]
        b = labels[valid] * root_weight[valid, None]
        return np.linalg.solve(a.T @ a + RIDGE * np.eye(a.shape[-1]), a.T @ b)

    displacement = [ridge(np.isfinite(p[:, 0]), p) for p in positions.transpose(1, 0, 2)]
    timing = ridge(onset >= 0, np.eye(len(ONSET_CLASSES))[onset])
    coef = np.concatenate((*displacement, timing), axis=1)
    return {"mean": mean, "scale": scale, "projection": projection, "bias": bias, "coef": coef}


def evaluate(data, trajectory, base):
    features, actions, positions, onset, weights = data
    aim, timing = forecast(features, trajectory)
    motion = features[:, None, 23:25] / (4 * rules.SPEED * rules.DT)
    errors = {}
    for horizon, reach, expected, forecasted in zip(HORIZONS, REACH, positions.transpose(1, 0, 2),
                                                   aim.transpose(1, 0, 2), strict=True):
        valid = np.isfinite(expected[:, 0])
        errors[str(horizon)] = {"samples": int(valid.sum()), **{
            name: float(np.average(np.linalg.norm((guess - expected[valid]) * reach, axis=1),
                                   weights=weights[valid]))
            for name, guess in (("forecast_mae_cells", forecasted[valid]),
                                ("constant_velocity_mae_cells", motion[valid, 0]))}}
    valid = onset >= 0
    truth = onset[valid]
    guesses = timing[valid].argmax(axis=1)
    reference = np.eye(len(ONSET_CLASSES))[truth]
    onset_metrics = {
        "class_names": ONSET_CLASSES,
        "samples": int(valid.sum()),
        "accuracy": float(np.average(truth == guesses, weights=weights[valid])),
        "always_none_accuracy": float(np.average(truth == 0, weights=weights[valid])),
        "class_counts": np.bincount(truth, minlength=len(ONSET_CLASSES)).tolist(),
        "brier_score": float(np.average(((timing[valid] - reference) ** 2).sum(axis=1), weights=weights[valid])),
    }
    scores = predict(features, base)
    ranked = rank_maneuvers(scores, base, aim, timing)
    nominal = decode(features, scores, base)
    candidate = decode(features, ranked, base)
    short = decode(features, rank_maneuvers(scores, base, aim, timing,
                                            displacement_weights=np.array((1., 0., 0.))), base)
    active = (np.rint(features[:, 0] * rules.GAME_TICKS) >= 2 * rules.TICK_HZ) & ~nominal[2]
    moving = active & np.isfinite(actions[:, 0])
    known_speed = actions[moving, 0]
    speed_sign = np.where(abs(known_speed) > SPEED_THRESHOLD, np.sign(known_speed), 0)

    def difference(before, after):
        return (before[0] != after[0]) | (np.abs(before[1] - after[1]) > .01)

    def action_errors(action):
        return {"turn_mae": float(np.mean(abs(action[1][active] - actions[active, 1]))),
                "speed_sign_accuracy": float(np.mean(np.sign(action[0][moving]) == speed_sign))}

    return {"displacement": errors, "onset": onset_metrics,
            "movement": {"active_frames": int(active.sum()),
                         "changed_frames_vs_v31": int((active & difference(nominal, candidate)).sum()),
                         "changed_frames_due_to_1s_2s_displacement": int((active & difference(short, candidate)).sum()),
                         "v31_nominal": action_errors(nominal),
                         "candidate_nominal": action_errors(candidate)}}


def run(request_path, request):
    folder = request_path.parent
    if set(request["params"]) != {"policy", "manifest_sha256"}:
        raise ValueError("trajectory-imitation 需要 manifest_sha256")
    root = project_root()
    base_id = request["config"]["teacher_snapshot"]
    versions.verify(root, base_id)
    frozen = versions.snapshot_data(root, base_id)
    files = versions.describe(root, base_id)["files"]
    if (files["bot.py"] != request["sources"]["boss_movement.py"]
            or files["boss_sparring_policy.py"] != request["sources"]["boss_sparring_policy.py"]):
        raise ValueError("基座策略的运动代码与训练时的特征实现不同")
    with np.load(frozen / "weights.npz", allow_pickle=False) as weights:
        base = dict(weights)
    if "opening_action" not in base or "zone_action" in base or "evade_incoming" not in base:
        raise ValueError("轨迹策略需要仅带起步恢复和躲弹的 v31 基座")
    splits = read_splits(folder, request)
    for group, rows in splits.items():
        print(f"{group}: {len(rows)} games", flush=True)
    train = sample(splits["train"], ManeuverFeatures)
    print(f"train: {len(train[0])} frames; fitting", flush=True)
    seed = request["config"]["sampling_seed"]
    trajectory = fit(train, seed)
    reports = {"train": evaluate(train, trajectory, base), "dev": evaluate(
        sample(splits["dev"], ManeuverFeatures), trajectory, base)}

    output = folder / request["output"]
    strategy = output / "strategy"
    strategy.mkdir()
    for file, destination in (("weights.npz", "weights.npz"), ("bot.py", "boss_movement.py"),
                              ("boss_sparring_policy.py", "boss_sparring_policy.py")):
        shutil.copyfile(frozen / file, strategy / destination)
    for name in ("boss_forecast.py", "boss_trajectory.py"):
        shutil.copyfile(Path(__file__).with_name(name), strategy / name)
    np.savez(strategy / "trajectory.npz", **trajectory)
    write_json(strategy / "submission.json", {"runtime": "python", "entrypoint": "boss_trajectory.py:TrajectorySparring"})
    write_json(strategy / "training.json", {
        "schema_version": 2, "rules": rules.as_dict()["version"], "seed": seed, "policy": "trajectory-imitation",
        "base_snapshot": base_id, "manifest_sha256": request["params"]["manifest_sha256"],
        "source_sha256": {name: request["sources"][name] for name in (
            "boss_forecast.py", "boss_movement.py", "boss_sparring_policy.py",
            "boss_trajectory.py", "fit_boss_sparring.py", "fit_boss_trajectory.py")},
        "train": [replay_source(*item) for item in splits["train"]],
        "holdout": [replay_source(*item) for item in splits["dev"]],
        "horizon_ticks": HORIZONS, "onset_classes": ONSET_CLASSES, "ridge": RIDGE,
        "metrics": reports,
    })
    write_json(output / "result.json", {"weights": "strategy/trajectory.npz", "step": 1,
                                        "strategy": "strategy", "metrics": reports})
    print(json.dumps(reports, ensure_ascii=False, allow_nan=False), flush=True)
