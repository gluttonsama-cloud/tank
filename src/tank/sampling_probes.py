"""SeedCup strategy snapshots and observed coverage for behavior sampling."""
import shutil
import tempfile
from math import hypot
from pathlib import Path

from seedcup.game import GameMap, SIDES
from seedcup.game.engine import line_of_sight
from seedcup.game.replay import FORMAT_VERSION

from . import versions
from .workspace import write_json


_PROBES = {
    "approach": ("接近开火", "有射线时接近并开火", "paired.py:Approach"),
    "silent": ("接近停火", "沿同一导航接近，全程停火", "paired.py:SilentApproach"),
    "pulse": ("脉冲开火", "沿同一导航接近，每 3 秒切换开火许可和停火", "pulse.py:Pulse"),
    "retreat": ("交战撤退", "有射线时倒车并开火", "paired.py:Retreat"),
    "peek": ("绕墙探头", "绕墙取得射线，尝试开火后退回掩体", "tactics.py:Peek"),
    "late-zone": ("缩圈回撤", "在外围等待缩圈，出圈后向圈心移动，全程停火", "tactics.py:LateZone"),
    "lancer": ("游击", "使用 Lancer 的预瞄、绕行和躲弹", None),
}

_CLOSE_RADIUS = 4
# Replay positions are rounded to .01 cells. Two positions can differ by
# sqrt(2) * .01 from the actual displacement purely because of rounding.
_MOVE_THRESHOLD = .025
# Headings are rounded to .1 degrees in the replay.
_TURN_THRESHOLD = .2
_PAIR_METRICS = (
    "visible_ticks", "close_ticks", "los_acquired", "los_lost",
    "shrink_ticks", "after_shrink_ticks", "late_visible_ticks", "late_close_ticks",
)
_SIDE_METRICS = (
    "approach_ticks", "retreat_ticks", "visible_approach_ticks", "visible_retreat_ticks",
    "shots", "hits_received", "late_shots", "late_hits_received",
    "turn_ticks", "enemy_shell_ticks", "turn_ticks_with_enemy_shell", "lateral_move_ticks_with_enemy_shell",
    "outside_zone_ticks", "zone_entries", "zone_exits",
)


def catalog() -> dict:
    return {name: {"label": label, "description": description}
            for name, (label, description, _) in _PROBES.items()}


def snapshots(root: Path, names) -> dict:
    """Freeze selected tactics and their navigation code through the normal version store."""
    names = tuple(names)
    unknown = set(names) - _PROBES.keys()
    if unknown:
        raise ValueError(f"未知采样打法：{'、'.join(sorted(unknown))}")
    result = {}
    for name in dict.fromkeys(names):
        label, description, entrypoint = _PROBES[name]
        if name == "lancer":
            result[name] = versions.create(root, name, description)["id"]
            continue
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            navigation = root / "strategies/lancer/bot.py"
            script = entrypoint.split(":")[0]
            shutil.copyfile(navigation, source / "navigation.py")
            if script in ("paired.py", "pulse.py"):
                shutil.copyfile(root / "training/boss_practice_probe.py", source / "paired.py")
            if script != "paired.py":
                shutil.copyfile(root / "strategies/sampling-probes" / script, source / script)
            write_json(source / "submission.json", {"runtime": "python", "entrypoint": entrypoint})
            result[name] = versions.create_from_directory(root, source, label, description)["id"]
    return result


def _counters() -> dict:
    return {"pair": dict.fromkeys(_PAIR_METRICS, 0),
            "target": dict.fromkeys(_SIDE_METRICS, 0),
            "probe": dict.fromkeys(_SIDE_METRICS, 0)}


def coverage(replay: dict, target_side: str) -> dict:
    """Count situations observed after each game tick from an explicit target perspective."""
    if replay["format"] != FORMAT_VERSION:
        raise ValueError(f"不支持的回放格式 {replay['format']}")
    if target_side not in SIDES:
        raise ValueError("target_side 必须为 red 或 blue")
    target_index = 1 + SIDES.index(target_side)
    zone = replay["rules"]["zone"]
    tick_hz = replay["rules"]["tick_hz"]
    game_map = GameMap.from_dict(replay["map"])
    frames = replay["frames"]
    if not frames or frames[0][0] != 0:
        raise ValueError("回放缺少第 0 帧")
    counters = _counters()

    def outside(tank, second):
        if second <= zone["shrink_start"]:
            radius = zone["start_radius"]
        elif second >= zone["shrink_end"]:
            radius = zone["final_radius"]
        else:
            progress = (second - zone["shrink_start"]) / (zone["shrink_end"] - zone["shrink_start"])
            radius = zone["start_radius"] + progress * (zone["final_radius"] - zone["start_radius"])
        return hypot(tank[0] - zone["x"], tank[1] - zone["y"]) > radius

    first = frames[0]
    was_visible = line_of_sight(game_map, *first[1][:2], *first[2][:2])
    was_out = {index: outside(first[index], 0) for index in (1, 2)}
    tracked_sides = (("target", target_index), ("probe", 3 - target_index))
    for previous, current in zip(frames, frames[1:]):
        tick, red, blue, _, _ = current
        if tick != previous[0] + 1:
            raise ValueError(f"回放帧号不连续：{previous[0]}、{tick}")
        second = tick / tick_hz
        late = second > zone["shrink_start"]
        visible = line_of_sight(game_map, *red[:2], *blue[:2])
        close = hypot(red[0] - blue[0], red[1] - blue[1]) <= _CLOSE_RADIUS
        pair = counters["pair"]
        pair["visible_ticks"] += visible
        pair["close_ticks"] += close
        pair["los_acquired"] += visible and not was_visible
        pair["los_lost"] += was_visible and not visible
        pair["shrink_ticks"] += late and second <= zone["shrink_end"]
        pair["after_shrink_ticks"] += second > zone["shrink_end"]
        pair["late_visible_ticks"] += late and visible
        pair["late_close_ticks"] += late and close

        for side, index in tracked_sides:
            other = 3 - index
            me, before, opponent_before = current[index], previous[index], previous[other]
            x, y = opponent_before[:2]
            # Measure each tank's contribution with the opponent held at its previous position.
            gain = hypot(x - before[0], y - before[1]) - hypot(x - me[0], y - me[1])
            approaching = gain > _MOVE_THRESHOLD
            retreating = gain < -_MOVE_THRESHOLD
            is_out = outside(me, second)
            shots = me[10] - before[10]
            hits_received = current[other][9] - opponent_before[9]
            turning = abs((me[2] - before[2] + 180) % 360 - 180) > _TURN_THRESHOLD
            count = counters[side]
            count["approach_ticks"] += approaching
            count["retreat_ticks"] += retreating
            count["visible_approach_ticks"] += visible and approaching
            count["visible_retreat_ticks"] += visible and retreating
            count["shots"] += shots
            count["hits_received"] += hits_received
            count["late_shots"] += shots if late else 0
            count["late_hits_received"] += hits_received if late else 0
            count["turn_ticks"] += turning
            enemy_shells = [shell for shell in previous[3] if shell[4] == other - 1]
            if enemy_shells:
                # Use shells already present when the tank made this frame's decision.
                shell = min(enemy_shells, key=lambda s: (s[0] - before[0]) ** 2 + (s[1] - before[1]) ** 2)
                dx, dy = me[0] - before[0], me[1] - before[1]
                lateral = abs(dx * shell[3] - dy * shell[2])
                count["enemy_shell_ticks"] += 1
                count["turn_ticks_with_enemy_shell"] += turning
                count["lateral_move_ticks_with_enemy_shell"] += lateral > _MOVE_THRESHOLD
            count["outside_zone_ticks"] += is_out
            count["zone_entries"] += was_out[index] and not is_out
            count["zone_exits"] += not was_out[index] and is_out
            was_out[index] = is_out
        was_visible = visible

    return {"target_side": target_side, "ticks": frames[-1][0], **counters}


def aggregate_coverage(items) -> dict:
    totals, games_with = _counters(), _counters()
    games = ticks = 0
    for item in items:
        games += 1
        ticks += item["ticks"]
        for side, metrics in totals.items():
            for name in metrics:
                value = item[side][name]
                metrics[name] += value
                games_with[side][name] += value > 0
    return {"games": games, "ticks": ticks, "totals": totals, "games_with": games_with}
