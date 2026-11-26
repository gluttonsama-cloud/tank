"""校准 SDK 策略视角逐帧识别终极档实际发射的精度。"""

from collections import Counter
from math import cos, hypot, radians, sin
from pathlib import Path

from seedcup.game import Game, GameMap, rules
from seedcup.game.engine import SIDES, Shell, Tank
from seedcup.game.replay import FORMAT_VERSION, load

from tank.sampling_data import layout_hash
from tank.workspace import project_root, read_json, write_json

from .fit_boss_replay_forecast import official_boss_side

SPLIT = Path(".local/diagnostics/boss-motion-learning-curve-20261125.json")
REPORT = Path(".local/mixed-boss-20261126/shot-calibration.json")
SHELL_STEP = rules.SHELL_SPEED * rules.DT
FIRST_SHELL_DISTANCE = rules.RADIUS + rules.MUZZLE_EXTENSION + SHELL_STEP
# The archive rounds positions to 0.01 and directions to 0.001. Retain the existing
# replay tracker's 0.025-cell matching window rather than choosing it from development scores.
POSITION_TOLERANCE = 0.025
DIRECTION_TOLERANCE = 0.002
NEAR_DISTANCE = 3.0
DISTANCES = ("all", "near", "far")
VIEWS = ("runtime", "including_terminal", "terminal_only")


def reduced_observation(game_map, frame, side):
    tick, red, blue, projectiles, _ = frame
    tanks = [Tank(i, *state[:4], bool(state[4])) for i, state in enumerate((red, blue))]
    shells = [Shell(i, owner, x, y, dx, dy) for i, (x, y, dx, dy, owner) in enumerate(projectiles)]
    return Game(game_map, tick=tick, tanks=tanks, shells=shells).observe(side)


def new_shells(previous, current):
    """Match continuations one-to-one by direction and one tick of motion."""
    unmatched = list(previous)
    started = []
    for shell in current:
        candidates = []
        for index, old in enumerate(unmatched):
            if hypot(shell["dx"] - old["dx"], shell["dy"] - old["dy"]) > DIRECTION_TOLERANCE:
                continue
            error = hypot(shell["x"] - old["x"] - old["dx"] * SHELL_STEP,
                          shell["y"] - old["y"] - old["dy"] * SHELL_STEP)
            if error <= POSITION_TOLERANCE:
                candidates.append((error, index))
        if candidates:
            _, index = min(candidates)
            unmatched.pop(index)
        else:
            started.append(shell)
    return started


def matches_muzzle(shell, shooter):
    """Independent visibility check for a confirmed shot under the zero-spread rules."""
    angle = radians(shooter["heading"])
    dx, dy = cos(angle), sin(angle)
    return (hypot(shell["dx"] - dx, shell["dy"] - dy) <= DIRECTION_TOLERANCE
            and hypot(shell["x"] - shooter["x"] - dx * FIRST_SHELL_DISTANCE,
                      shell["y"] - shooter["y"] - dy * FIRST_SHELL_DISTANCE) <= POSITION_TOLERANCE)


def count_frames(replay):
    meta = replay["match"]
    boss = official_boss_side(meta)
    if boss is None or replay["format"] != FORMAT_VERSION or replay["rules"] != rules.as_dict() or rules.VARIANT:
        raise ValueError(f"回放 {meta['id']} 的终极档身份、格式或规则不符合校准要求")
    boss_index = SIDES.index(boss)
    observer_index = 1 - boss_index
    game_map = GameMap.from_dict(replay["map"])
    frames = replay["frames"]
    end = replay["result"]["ticks"]
    if (len(frames) != end + 1 or any(frame[0] != tick for tick, frame in enumerate(frames))
            or frames[0][boss_index + 1][10] != 0 or frames[0][3]):
        raise ValueError(f"回放 {meta['id']} 的起始状态或帧序列不完整")

    counts = {view: {distance: Counter() for distance in DISTANCES} for view in VIEWS}
    previous_shells = []
    previous_shots = 0
    for frame in frames[1:]:
        observation = reduced_observation(game_map, frame, observer_index)
        enemy_shells = [shell for shell in observation["shells"] if not shell["mine"]]
        started = new_shells(previous_shells, enemy_shells)
        onset = len(started)
        previous_shells = enemy_shells

        shots = frame[boss_index + 1][10]
        actual = shots - previous_shots
        previous_shots = shots
        if actual not in (0, 1):
            raise ValueError(f"回放 {meta['id']} 第 {frame[0]} 帧的开火计数差为 {actual}")
        own, enemy = observation["self"], observation["enemy"]
        distance = hypot(own["x"] - enemy["x"], own["y"] - enemy["y"])
        visible = actual and any(matches_muzzle(shell, enemy) for shell in enemy_shells)
        visible_detected = actual and any(matches_muzzle(shell, enemy) for shell in started)
        view = "terminal_only" if frame[0] == end else "runtime"
        band = "near" if distance <= NEAR_DISTANCE else "far"

        for scope in ("including_terminal", view):
            for distance_group in ("all", band):
                cell = counts[scope][distance_group]
                cell["transitions"] += 1
                cell["actual"] += actual
                cell["predicted"] += onset
                cell["tp"] += min(actual, onset)
                cell["tn"] += not actual and not onset
                cell["visible_actual"] += visible
                cell["visible_detected"] += visible_detected
    return counts, boss


def summarize(count):
    actual, predicted, tp = count["actual"], count["predicted"], count["tp"]
    visible = count["visible_actual"]
    return {
        "transitions": count["transitions"], "actual": actual, "predicted": predicted,
        "tp": tp, "fp": predicted - tp, "fn": actual - tp, "tn": count["tn"],
        "precision": tp / predicted if predicted else None,
        "recall": tp / actual if actual else None,
        "visible_actual": visible,
        "first_frame_without_projectile": actual - visible,
        "visible_detected": count["visible_detected"],
        "visible_recall": count["visible_detected"] / visible if visible else None,
    }


def summarize_groups(counts):
    return {view: {band: summarize(counts[view][band]) for band in DISTANCES} for view in VIEWS}


def calibrate(root):
    split = read_json(root / SPLIT)
    ids = split["dev_ids"]
    if len(ids) != len(set(ids)) or set(ids).intersection(split["train_ids"]):
        raise ValueError("冻结开发回放与训练回放重叠或开发回放重复")
    totals = {view: {band: Counter() for band in DISTANCES} for view in VIEWS}
    games = []
    for replay_id in ids:
        replay = load(root / ".local/online/replays" / f"{replay_id}.json.gz")
        if replay["match"]["id"] != replay_id:
            raise ValueError(f"回放 {replay_id} 的局号与文件名不一致")
        counts, boss = count_frames(replay)
        games.append({"id": replay_id, "boss_side": boss, "seed": replay["map"]["seed"],
                      "layout": layout_hash(replay["map"]), "metrics": summarize_groups(counts)})
        for view in VIEWS:
            for band in DISTANCES:
                totals[view][band].update(counts[view][band])
    layout_count = len({game["layout"] for game in games})
    if layout_count != split["dev_layouts"]:
        raise ValueError(f"观测到 {layout_count} 张开发地图，与冻结数量 {split['dev_layouts']} 不符")
    return {
        "split": SPLIT.as_posix(), "ids": ids, "layout_count": layout_count,
        "rules": rules.as_dict()["version"],
        "definitions": {
            "observer": "终极练习人机的对手。对原回放的公开位置、方向和场上炮弹调用 SDK Game.observe。",
            "reference": "同一帧终极档累计开火计数相对上一帧的增量，每帧最多一次。",
            "prediction": "逐帧以方向和按弹速外推的位置匹配敌方炮弹，一对一匹配剩余数即预测发射数。",
            "distance": "开火结果帧两车中心距离，近距离 <= 3 格，远距离 > 3 格。",
            "runtime": "计入第 1 帧至终局前一帧。协议会在这些帧调用策略。",
            "terminal_only": "仅计入终局帧。回放有该帧，策略不会收到该帧。",
            "including_terminal": "计入全部结果帧，包含终局，用于对照。",
            "visibility": "独立核对已确认发射的炮弹是否出现在对应帧的炮口加完整一帧飞行距离处。",
            "precision_recall": "按完全相同的结果帧匹配预测与真值，跨局按事件数汇总。",
            "position_tolerance_cells": POSITION_TOLERANCE,
            "direction_vector_tolerance": DIRECTION_TOLERANCE,
            "limitations": "回放位置和方向精度低于线上 SDK。首帧内撞击的炮弹不会出现在观测中。",
        },
        "metrics": summarize_groups(totals), "games": games,
    }


if __name__ == "__main__":
    root = project_root()
    write_json(root / REPORT, calibrate(root))
