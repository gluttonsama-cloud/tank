"""Fit a local sparring policy with whole-map holdout from public boss replays."""

import json
from collections import Counter
from math import cos, floor, hypot, radians, sin
from pathlib import Path
import shutil
import sys

import numpy as np
from seedcup.game import GameMap, rules
from seedcup.game.replay import FORMAT_VERSION, load
from seedcup.game.runner import match_info

if __package__:
    from .boss_sparring_policy import TURN_BOUNDS, Features, ManeuverFeatures, angle_delta, decode, predict
else:
    from boss_sparring_policy import TURN_BOUNDS, Features, ManeuverFeatures, angle_delta, decode, predict


BOSS_NAME = "终极练习人机"
MOVE_STEP = rules.SPEED * rules.DT
TURN_STEP = rules.TURN_RATE * rules.DT
POSITION_TOLERANCE = 0.0101  # The difference between two positions rounded to 0.01.
SPEED_THRESHOLD = 0.08


def boss_side(meta):
    sides = [index for index, side in enumerate(("red", "blue"))
             if meta[side]["bot"] is True and meta[side]["name"] == BOSS_NAME]
    if len(sides) != 1:
        raise ValueError(f"回放 {meta['id']} 必须有一方为{BOSS_NAME}")
    return sides[0]


def replay_observation(frame, side):
    tick, red, blue, shells, _ = frame
    own, other = (red, blue) if side == 0 else (blue, red)
    names = ("x", "y", "heading", "hp", "alive")
    return {
        "frame": tick,
        "self": dict(zip(names, own[:5], strict=True)) | {
            "reload": own[5] * rules.DT, "still": own[6] * rules.DT, "repairing": bool(own[7]),
        },
        "enemy": dict(zip(names, other[:5], strict=True)),
        "shells": [{"x": x, "y": y, "dx": dx, "dy": dy, "mine": owner == side}
                   for x, y, dx, dy, owner in shells],
        "zone": {"x": rules.ZONE_CX, "y": rules.ZONE_CY, "radius": round(rules.zone_radius(tick), 3)},
    }


def can_be_blocked(game_map, own, enemy):
    reach = rules.RADIUS + MOVE_STEP + POSITION_TOLERANCE
    x, y = own[:2]
    if enemy[4] and hypot(x - enemy[0], y - enemy[1]) < 2 * reach:
        return True
    for cy in range(floor(y - reach), floor(y + reach) + 1):
        for cx in range(floor(x - reach), floor(x + reach) + 1):
            if game_map.is_wall(cx, cy):
                dx = max(cx - x, 0, x - cx - 1)
                dy = max(cy - y, 0, y - cy - 1)
                if hypot(dx, dy) < reach:
                    return True
    return False


def recover_forward(game_map, own, enemy, after):
    dx, dy = after[0] - own[0], after[1] - own[1]
    h = radians(after[2])
    c, s = cos(h), sin(h)
    if max(abs(dx), abs(dy)) <= POSITION_TOLERANCE:
        if after[6] == own[6] + 1 or not can_be_blocked(game_map, own, enemy):
            return 0.0
        return np.nan

    # The engine can keep one axis fixed when sliding against a wall. Fit the observed axis.
    if abs(dx) <= POSITION_TOLERANCE and abs(s) > .04 and abs(c * dy / s) > 2 * POSITION_TOLERANCE:
        speed = dy / s
    elif abs(dy) <= POSITION_TOLERANCE and abs(c) > .04 and abs(s * dx / c) > 2 * POSITION_TOLERANCE:
        speed = dx / c
    else:
        speed = c*dx + s*dy
    return float(np.clip(speed / MOVE_STEP, -1, 1))


def extract(replay, feature_type=Features):
    game_map = GameMap.from_dict(replay["map"])
    side = boss_side(replay["match"])
    extractor = feature_type(match_info(game_map, side, ""))
    features, targets = [], []
    ticks = replay["result"]["ticks"]
    frames = replay["frames"]
    if len(frames) != ticks + 1 or any(index != frame[0] for index, frame in enumerate(frames)):
        raise ValueError(f"回放 {replay['match']['id']} 帧缺失或不连续")
    source = side + 1
    for frame, after in zip(frames, frames[1:]):
        own, enemy, next_own = frame[source], frame[3 - source], after[source]
        if not own[4]:
            raise ValueError(f"回放 {replay['match']['id']} 在终局前出现已死亡陪练")
        turn = np.clip(angle_delta(next_own[2], own[2]) / TURN_STEP, -1, 1)
        fire = float(next_own[10] > own[10]) if own[5] <= 1 else np.nan
        forward = recover_forward(game_map, own, enemy, next_own)
        features.append(extractor(replay_observation(frame, side)))
        targets.append((forward, -turn if extractor.flip else turn, fire))
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def adam_fit(x, y, seed, epochs):
    rng = np.random.default_rng(seed % (2**64))
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 0.08)
    x = (x - mean) / scale
    forward, turn, fire = y.T
    classes = np.where(forward > SPEED_THRESHOLD, 2, np.where(forward < -SPEED_THRESHOLD, 0, 1))
    reverse, advance = forward[classes == 0], forward[classes == 2]
    if not len(reverse) or not len(advance) or len(np.unique(fire[np.isfinite(fire)])) != 2:
        raise ValueError("训练数据须包含前进、后退、可发射时的发射与未发射样本")
    speeds = np.asarray((np.median(reverse), 0, np.median(advance)), dtype=np.float32)
    valid = np.isfinite(y)
    one_hot = np.eye(3, dtype=np.float32)[classes]
    fire = np.nan_to_num(fire)
    hidden_size = 48
    params = {
        "w1": rng.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], hidden_size)).astype(np.float32),
        "b1": np.zeros(hidden_size, dtype=np.float32),
        "w2": rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, 5)).astype(np.float32),
        "b2": np.zeros(5, dtype=np.float32),
    }
    momentum = {key: np.zeros_like(value) for key, value in params.items()}
    velocity = {key: np.zeros_like(value) for key, value in params.items()}
    batch_size = 256
    step = 0
    for _ in range(epochs):
        shuffled = rng.permutation(len(x))
        for offset in range(0, len(x), batch_size):
            take = shuffled[offset:offset + batch_size]
            batch, mask = x[take], valid[take]
            h = np.tanh(batch @ params["w1"] + params["b1"])
            scores = h @ params["w2"] + params["b2"]
            probabilities = np.exp(scores[:, :3] - scores[:, :3].max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            ds = np.empty_like(scores)
            ds[:, :3] = (probabilities - one_hot[take]) * mask[:, :1] / max(1, mask[:, 0].sum())
            ds[:, 3] = np.clip(scores[:, 3] - turn[take], -1, 1) * 2 / len(take)
            fire_probability = 1 / (1 + np.exp(-np.clip(scores[:, 4], -30, 30)))
            ds[:, 4] = (fire_probability - fire[take]) * mask[:, 2] / max(1, mask[:, 2].sum())
            dh = (ds @ params["w2"].T) * (1 - h*h)
            gradients = {"w1": batch.T @ dh, "b1": dh.sum(axis=0), "w2": h.T @ ds, "b2": ds.sum(axis=0)}
            step += 1
            for key, gradient in gradients.items():
                if key.startswith("w"):
                    gradient += .001 * params[key]
                momentum[key] = .9 * momentum[key] + .1 * gradient
                velocity[key] = .999 * velocity[key] + .001 * gradient**2
                params[key] -= .002 * (momentum[key] / (1 - .9**step)) / (
                    np.sqrt(velocity[key] / (1 - .999**step)) + 1e-8)
    return params | {"mean": mean, "scale": scale, "speeds": speeds}


RANGE_SECONDS = (1, 3)


def future_range_changes(frames):
    """Signed range change as a fraction of the distance both tanks could travel."""
    positions = np.asarray([[*red[:2], *blue[:2]] for _, red, blue, _, _ in frames], dtype=np.float32)
    distance = np.linalg.norm(positions[:, :2] - positions[:, 2:], axis=1)
    targets = np.full((len(frames) - 1, len(RANGE_SECONDS)), np.nan, dtype=np.float32)
    for column, seconds in enumerate(RANGE_SECONDS):
        ticks = seconds * rules.TICK_HZ
        count = len(frames) - ticks
        if count > 0:
            targets[:count, column] = (distance[ticks:] - distance[:-ticks]) / (2 * rules.SPEED * seconds)
    return targets


def map_frame_weights(maps, lengths):
    games = Counter(maps)
    total = sum(lengths)
    return np.concatenate([
        np.full(count, total / (len(games) * games[layout] * count), dtype=np.float32)
        for layout, count in zip(maps, lengths, strict=True)
    ])


def fit_maneuvers(x, y, seed, epochs, *, future=None, range_weight=1.0, frame_weights=None):
    rng = np.random.default_rng(seed % (2**64))
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), .08)
    x = (x - mean) / scale
    forward, turn, fire = y.T
    valid_forward, valid_fire = np.isfinite(forward), np.isfinite(fire)
    speed_class = np.where(forward > SPEED_THRESHOLD, 2, np.where(forward < -SPEED_THRESHOLD, 0, 1))
    turn_class = np.searchsorted(TURN_BOUNDS, turn)
    reverse, advance = forward[speed_class == 0], forward[speed_class == 2]
    if not len(reverse) or not len(advance) or len(np.unique(fire[valid_fire])) != 2:
        raise ValueError("训练数据须包含前进、后退、可发射时的发射与未发射样本")
    speeds = np.asarray((np.median(reverse), 0, np.median(advance)), dtype=np.float32)
    turn_centers = np.array([-1, -.2, 0, .2, 1], dtype=np.float32)
    residual = turn - turn_centers[turn_class]
    fire = np.nan_to_num(fire)

    widths = (x.shape[1], 96, 64, 21)
    params = {}
    for i, (fan_in, fan_out) in enumerate(zip(widths, widths[1:]), start=1):
        params[f"w{i}"] = rng.normal(0, 1 / np.sqrt(fan_in), (fan_in, fan_out)).astype(np.float32)
        params[f"b{i}"] = np.zeros(fan_out, dtype=np.float32)
    if future is not None:
        # Same action initialization and minibatch order as the single-frame control.
        params["w_range"] = np.zeros((widths[-2], future.shape[1]), dtype=np.float32)
        params["b_range"] = np.zeros(future.shape[1], dtype=np.float32)
    momentum = {k: np.zeros_like(v) for k, v in params.items()}
    velocity = {k: np.zeros_like(v) for k, v in params.items()}
    step = 0
    for _ in range(epochs):
        shuffled = rng.permutation(len(x))
        for offset in range(0, len(x), 256):
            take = shuffled[offset:offset + 256]
            batch = x[take]
            h1 = np.tanh(batch @ params["w1"] + params["b1"])
            h2 = np.tanh(h1 @ params["w2"] + params["b2"])
            scores = h2 @ params["w3"] + params["b3"]

            joint = scores[:, :15].reshape(-1, 3, 5)
            probabilities = np.exp(joint - joint.max(axis=(1, 2), keepdims=True))
            probabilities /= probabilities.sum(axis=(1, 2), keepdims=True)
            row = np.arange(len(take))
            turn_at = turn_class[take]
            known = valid_forward[take]
            probabilities[row[known], speed_class[take][known], turn_at[known]] -= 1
            # If wall contact makes speed ambiguous, marginalize it and still learn the turn.
            masked = probabilities[row[~known], :, turn_at[~known]]
            probabilities[row[~known], :, turn_at[~known]] -= masked / masked.sum(axis=1, keepdims=True)
            ds = np.zeros_like(scores)
            ds[:, :15] = probabilities.reshape(-1, 15) / len(take)
            residual_index = 15 + turn_at
            ds[row, residual_index] = (scores[row, residual_index] - residual[take]) / len(take)
            fire_probability = 1 / (1 + np.exp(-np.clip(scores[:, 20], -30, 30)))
            fire_mask = valid_fire[take]
            ds[:, 20] = (fire_probability - fire[take]) * fire_mask / max(1, fire_mask.sum())

            if frame_weights is not None:
                ds *= frame_weights[take, None]
            hidden_gradient = ds @ params["w3"].T
            gradients = {}
            if future is not None:
                target = future[take]
                valid = np.isfinite(target)
                error = h2 @ params["w_range"] + params["b_range"] - target
                dr = np.where(valid, np.clip(error, -1, 1), 0) * (range_weight / max(1, valid.sum()))
                dr = dr.astype(np.float32, copy=False)
                if frame_weights is not None:
                    dr *= frame_weights[take, None]
                gradients.update(w_range=h2.T @ dr, b_range=dr.sum(axis=0))
                hidden_gradient += dr @ params["w_range"].T
            dh2 = hidden_gradient * (1 - h2*h2)
            dh1 = (dh2 @ params["w2"].T) * (1 - h1*h1)
            gradients.update(w1=batch.T @ dh1, b1=dh1.sum(axis=0), w2=h1.T @ dh2,
                             b2=dh2.sum(axis=0), w3=h2.T @ ds, b3=ds.sum(axis=0))
            step += 1
            for key, gradient in gradients.items():
                if key.startswith("w"):
                    gradient += .001 * params[key]
                momentum[key] = .9 * momentum[key] + .1 * gradient
                velocity[key] = .999 * velocity[key] + .001 * gradient**2
                params[key] -= .002 * (momentum[key] / (1 - .9**step)) / (
                    np.sqrt(velocity[key] / (1 - .999**step)) + 1e-8)
    if future is not None:
        del params["w_range"], params["b_range"]
    return params | {"mean": mean, "scale": scale, "speeds": speeds, "turn_centers": turn_centers}


def metrics(x, y, weights):
    """Measure decoded one-frame actions, before any SDK evasion rollout."""
    forward, turn, fire = decode(x, predict(x, weights), weights)
    target_forward, target_turn, target_fire = y.T
    moving = np.isfinite(target_forward)
    ready = np.isfinite(target_fire)
    observed_fire = target_fire[ready] == 1
    fired = fire[ready]
    true_positive = np.count_nonzero(observed_fire & fired)
    return {
        "frames": len(x), "forward_labels": int(moving.sum()), "fire_labels": int(ready.sum()),
        "forward_accuracy": float(np.mean(np.sign(forward[moving]) == np.where(
            abs(target_forward[moving]) > SPEED_THRESHOLD, np.sign(target_forward[moving]), 0))) if moving.any() else None,
        "turn_mae": float(np.mean(abs(turn - target_turn))),
        "observed_full_turn_rate": float(np.mean(abs(target_turn) > .9)),
        "predicted_full_turn_rate": float(np.mean(abs(turn) > .9)),
        "observed_shots": int(observed_fire.sum()), "predicted_shots": int(fired.sum()),
        "fire_precision": float(true_positive / fired.sum()) if fired.any() else None,
        "fire_recall": float(true_positive / observed_fire.sum()) if observed_fire.any() else None,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def selected_ids(params, key):
    values = params[key]
    if not isinstance(values, list) or not values or any(type(value) is not int or value < 1 for value in values):
        raise ValueError(f"{key} 必须包含正整数官网回放 ID")
    if len(values) != len(set(values)):
        raise ValueError(f"{key} 包含重复 ID")
    return sorted(values)


def load_splits(folder, request):
    params = request["params"]
    if unknown := params.keys() - {"train_ids", "holdout_ids", "epochs", "policy", "range_weight", "balance_maps"}:
        raise ValueError(f"未知训练参数 {', '.join(sorted(unknown))}")
    train_ids, holdout_ids = selected_ids(params, "train_ids"), selected_ids(params, "holdout_ids")
    if set(train_ids) & set(holdout_ids):
        raise ValueError("训练与留出不能包含相同回放")
    replays = {}
    for source in request["inputs"]:
        replay = load(folder / source["path"])
        replay_id = replay["match"]["id"]
        if type(replay_id) is not int or replay_id < 1 or replay_id in replays:
            raise ValueError(f"回放 ID 无效或重复 {replay_id}")
        if replay["format"] != FORMAT_VERSION or replay["rules"] != rules.as_dict():
            raise ValueError(f"回放 {replay_id} 格式或规则与当前官方 SDK 不一致")
        replays[replay_id] = (replay, source["sha256"])
    if replays.keys() != set(train_ids + holdout_ids):
        raise ValueError("--input 中的回放 ID 必须与 train_ids、holdout_ids 一致")
    train = [replays[key] for key in train_ids]
    holdout = [replays[key] for key in holdout_ids]
    for key in ("seed", "grid"):
        if {replay["map"][key] for replay, _ in train} & {replay["map"][key] for replay, _ in holdout}:
            raise ValueError("训练与留出必须使用不同地图")
    return train, holdout


def main(request_path):
    folder = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    train, holdout = load_splits(folder, request)
    epochs = request["params"].get("epochs", 36)
    if type(epochs) is not int or not 1 <= epochs <= 200:
        raise ValueError("epochs 必须为 1 到 200 的整数")
    policy = request["params"].get("policy", "maneuver-evasion")
    if policy not in ("legacy", "maneuver", "maneuver-evasion", "range-evasion"):
        raise ValueError("未知训练策略 " + str(policy))
    range_weight = request["params"].get("range_weight", 1.0)
    balance = request["params"].get("balance_maps", False)
    if type(balance) is not bool or (balance and policy != "range-evasion"):
        raise ValueError("balance_maps 是 range-evasion 的布尔参数")
    if ("range_weight" in request["params"] and policy != "range-evasion"
            or type(range_weight) not in (int, float) or not 0 < range_weight <= 10):
        raise ValueError("range_weight 仅用于 range-evasion，范围为 (0, 10]")
    feature_type, fit = (Features, adam_fit) if policy == "legacy" else (ManeuverFeatures, fit_maneuvers)
    seed = request["config"]["sampling_seed"]
    train_xs, train_ys = zip(*(extract(replay, feature_type) for replay, _ in train))
    train_x, train_y = np.concatenate(train_xs), np.concatenate(train_ys)
    if policy == "range-evasion":
        future = np.concatenate([future_range_changes(replay["frames"]) for replay, _ in train])
        frame_weights = (map_frame_weights([rep["map"]["grid"] for rep, _ in train], list(map(len, train_xs)))
                         if balance else None)
        weights = fit_maneuvers(train_x, train_y, seed, epochs, future=future, range_weight=range_weight,
                                frame_weights=frame_weights)
    else:
        weights = fit(train_x, train_y, seed, epochs)
    if policy in ("maneuver-evasion", "range-evasion"):
        weights["evade_incoming"] = np.asarray(True)
    holdout_xs, holdout_ys = zip(*(extract(replay, feature_type) for replay, _ in holdout))
    record = {
        "train": metrics(train_x, train_y, weights),
        "holdout": metrics(np.concatenate(holdout_xs), np.concatenate(holdout_ys), weights),
        "holdout_by_replay": {replay["match"]["id"]: metrics(x, y, weights)
                              for (replay, _), x, y in zip(holdout, holdout_xs, holdout_ys, strict=True)},
    }
    output = folder / request["output"]
    strategy = output / "strategy"
    strategy.mkdir()
    shutil.copyfile(Path(__file__).with_name("boss_sparring_policy.py"), strategy / "bot.py")
    np.savez(strategy / "weights.npz", **weights)
    write_json(strategy / "submission.json", {"runtime": "python", "entrypoint": "bot.py:BossSparring"})
    write_json(strategy / "training.json", {
        "schema_version": 2, "rules": rules.as_dict()["version"], "seed": seed, "epochs": epochs, "policy": policy,
        "source_sha256": {name: request["sources"][name] for name in (
            Path(__file__).name, "boss_sparring_policy.py")},
        "train": [replay_source(replay, digest) for replay, digest in train],
        "holdout": [replay_source(replay, digest) for replay, digest in holdout],
        "metrics_scope": "nominal_action",
        **({"range_seconds": RANGE_SECONDS, "range_weight": range_weight, "balance_maps": balance}
           if policy == "range-evasion" else {}),
        "metrics": record,
    })
    write_json(output / "result.json", {"weights": "strategy/weights.npz", "step": epochs,
                                         "strategy": "strategy", "metrics": record})
    print(json.dumps(record, ensure_ascii=False, allow_nan=False))


def replay_source(replay, digest):
    return {"id": replay["match"]["id"], "seed": replay["map"]["seed"], "sha256": digest}


if __name__ == "__main__":
    main(Path(sys.argv[1]))
