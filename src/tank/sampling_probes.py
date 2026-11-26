"""SeedCup strategy snapshots and observed coverage for behavior sampling."""
import shutil
import tempfile
from itertools import pairwise
from math import atan2, cos, hypot, radians, sin
from pathlib import Path

from seedcup.game import SIDES, GameMap
from seedcup.game.bots import wall_grid
from seedcup.game.engine import line_of_sight
from seedcup.game.replay import FORMAT_VERSION

from . import versions
from ._window_eligibility import heading_error, shot_opportunity
from .workspace import write_json

_PROBES = {
    "approach": ("接近开火", "有射线时接近并开火", "paired.py:Approach"),
    "silent": ("接近停火", "沿同一导航接近，全程停火", "paired.py:SilentApproach"),
    "pulse": ("脉冲开火", "沿同一导航接近，每 3 秒切换开火许可和停火", "pulse.py:Pulse"),
    "retreat": ("交战撤退", "有射线时倒车并开火", "paired.py:Retreat"),
    "peek": ("绕墙探头", "绕墙取得射线，尝试开火后退回掩体", "tactics.py:Peek"),
    "late-zone": ("缩圈回撤", "在外围等待缩圈，出圈后向圈心移动，全程停火", "tactics.py:LateZone"),
    "recontact": ("拉开再接近", "贴近压迫并尝试开火，拉开距离后再次接近", "recontact.py:Recontact"),
    "window-fire": ("近身开火对照", "同路接近，在首次具备近身开火条件时开火一次", "window.py:WindowFire"),
    "window-hold": ("近身停火对照", "同路接近，在首次具备近身开火条件时继续停火", "window.py:WindowHold"),
    "holdout-orbit": ("换向绕行", "贴近中距离后交替绕行，每次换向前停下瞄准", "holdout_orbit.py:HoldoutOrbit"),
    "lancer": ("游击", "使用 Lancer 的预瞄、绕行和躲弹", None),
}

_CLOSE_RADIUS = 4
# Replay positions are rounded to .01 cells. Two positions can differ by
# sqrt(2) * .01 from the actual displacement purely because of rounding.
_MOVE_THRESHOLD = .025
# Headings are rounded to .1 degrees in the replay.
_TURN_THRESHOLD = .2
_BEHAVIOR_NEAR = 3  # cells
_BEHAVIOR_FAR = 5  # cells
_WITHDRAWAL_GAIN = 1  # cells attributable to the target
_HIT_RESPONSE_SECONDS = 3
_LATE_SECONDS = 60
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
            if script in ("recontact.py", "window.py"):
                shutil.copyfile(root / "strategies/sampling-probes/tactics.py", source / "tactics.py")
            if script == "window.py":
                shutil.copyfile(Path(__file__).with_name("_window_eligibility.py"), source / "window_eligibility.py")
            if script != "paired.py":
                shutil.copyfile(root / "strategies/sampling-probes" / script, source / script)
            write_json(source / "submission.json", {"runtime": "python", "entrypoint": entrypoint})
            result[name] = versions.create_from_directory(root, source, label, description)["id"]
            if name == "holdout-orbit":
                from .sampling_data import database
                with database(root) as db:
                    db.execute("INSERT OR IGNORE INTO sampling_reserved_versions VALUES (?)", (result[name],))
    return result


def _counters() -> dict:
    return {"pair": dict.fromkeys(_PAIR_METRICS, 0),
            "target": dict.fromkeys(_SIDE_METRICS, 0),
            "probe": dict.fromkeys(_SIDE_METRICS, 0)}


def _replay_inputs(replay, target_side):
    if replay["format"] != FORMAT_VERSION:
        raise ValueError(f"不支持的回放格式 {replay['format']}")
    if target_side not in SIDES:
        raise ValueError("target_side 必须为 red 或 blue")
    tick_hz = replay["rules"]["tick_hz"]
    frames = replay["frames"]
    if not frames or frames[0][0] != 0:
        raise ValueError("回放缺少第 0 帧")
    return 1 + SIDES.index(target_side), tick_hz, frames


def _transitions(frames):
    for before, after in pairwise(frames):
        if after[0] != before[0] + 1:
            raise ValueError(f"回放帧号不连续：{before[0]}、{after[0]}")
        yield before, after


def _retreat_lanes(game_map, probe, target, radius):
    """One-cell terrain clearance along and 45 degrees to either side of separation."""
    x, y = target[:2]
    away = atan2(y - probe[1], x - probe[0])
    lanes = []
    for name, angle in (("straight", 0), ("clockwise", 45), ("counterclockwise", -45)):
        dx, dy = cos(away + radians(angle)), sin(away + radians(angle))
        # Test the center and both sides of the tank's straight path through the grid.
        if all(line_of_sight(game_map, x - offset * dy, y + offset * dx,
                             x + dx - offset * dy, y + dy + offset * dx)
               for offset in (0, radius, -radius)):
            lanes.append(name)
    return lanes


def intervention_anchor(replay: dict, probe_side: str) -> dict | None:
    """Return the first replay-observed qualification, using states before action.

    The SDK replay rounds positions and headings and excludes action source frames.
    Online observations and their action execution therefore require separate evidence.
    """
    if probe_side not in SIDES:
        raise ValueError("probe_side 必须为 red 或 blue")
    probe_index, tick_hz, frames = _replay_inputs(replay, probe_side)
    target_index = 3 - probe_index
    game_map = GameMap.from_dict(replay["map"])
    walls = wall_grid({"map": {"grid": game_map.rows()}})
    rule_set = replay["rules"]
    turn_per_tick = rule_set["turn_rate"] / tick_hz
    for current, _ in _transitions(frames):
        probe, target = current[probe_index], current[target_index]
        opportunity = shot_opportunity(walls, probe, target, turn_per_tick=turn_per_tick,
                                      reload_ticks=probe[5])
        if opportunity is not None:
            return {
                "frame": current[0],
                "source": "replay_observation",
                "distance": round(opportunity["distance"], 4),
                "probe_heading_error_deg": round(opportunity["probe_heading_error_deg"], 2),
                "target_heading_error_deg": round(heading_error(target, probe), 2),
                "post_turn_error_deg": round(opportunity["post_turn_error_deg"], 2),
                "probe_pose": probe[:3], "target_pose": target[:3],
                "probe_hp": probe[3], "target_hp": target[3],
                "probe_reload_ticks": probe[5], "target_reload_ticks": target[5],
                "target_retreat_lanes": _retreat_lanes(game_map, probe, target, rule_set["radius"]),
                "shells_by_side": {side: sum(shell[4] == i for shell in current[3])
                                   for i, side in enumerate(SIDES)},
            }
    return None


def _distance(a, b):
    # Snapshot coordinates are rounded to .01 cells. Avoid float artifacts on cutoffs.
    return round(hypot(a[0] - b[0], a[1] - b[1]), 12)


def behavior_signals(replay: dict, target_side: str) -> dict[str, bool]:
    """Tag geometric distance (cells), HP and projectile hits over SDK replay ticks.

    The target's signed contribution to separation averages its movement with
    the probe fixed at its before and after positions. In either withdrawal
    signal, the target must contribute a net cell by the time distance reaches
    five cells. After a near hit this must occur within three seconds.
    """
    target_index, tick_hz, frames = _replay_inputs(replay, target_side)
    probe_index = 3 - target_index
    reengage = boss_withdraw = after_hit_withdraw = False
    disadvantaged_ticks = late_ticks = 0
    distance = _distance(frames[0][target_index], frames[0][probe_index])
    separating = distance <= _BEHAVIOR_NEAR
    separated = False
    target_gain = withdrawal_start = 0.0
    near_hits = []
    for previous, current in _transitions(frames):
        target, probe = current[target_index], current[probe_index]
        last_target, last_probe = previous[target_index], previous[probe_index]
        prior_distance, distance = distance, _distance(target, probe)
        tick = current[0]
        if not (target[4] and probe[4]):
            continue

        near = distance <= _BEHAVIOR_NEAR
        far = distance >= _BEHAVIOR_FAR
        disadvantaged_ticks += near and target[3] < probe[3]
        late_ticks += tick > _LATE_SECONDS * tick_hz and distance <= _BEHAVIOR_FAR

        if separating or near_hits:
            # This symmetric split sums to the actual change across both sides.
            target_gain += (_distance(target, last_probe) - prior_distance
                            + distance - _distance(last_target, probe)) / 2

        if near:
            reengage |= separated
            separating, separated = True, False
            withdrawal_start = target_gain
        elif separating and far:
            boss_withdraw |= round(target_gain - withdrawal_start, 10) >= _WITHDRAWAL_GAIN
            separating, separated = False, True

        near_hits = [(deadline, start) for deadline, start in near_hits if tick <= deadline]
        if far and near_hits:
            after_hit_withdraw |= any(round(target_gain - start, 10) >= _WITHDRAWAL_GAIN
                                      for _, start in near_hits)
            near_hits.clear()
        # Movement precedes impact within an SDK tick. Start counting after impact.
        if near and probe[9] > last_probe[9]:
            near_hits.append((tick + _HIT_RESPONSE_SECONDS * tick_hz, target_gain))

    return {
        "reengage": reengage,
        "boss_withdraw": boss_withdraw,
        "disadvantaged_close": disadvantaged_ticks >= tick_hz,
        "late_close": late_ticks >= tick_hz,
        "after_hit_withdraw": after_hit_withdraw,
    }


def coverage(replay: dict, target_side: str) -> dict:
    """Count situations observed after each game tick from an explicit target perspective."""
    target_index, tick_hz, frames = _replay_inputs(replay, target_side)
    zone = replay["rules"]["zone"]
    game_map = GameMap.from_dict(replay["map"])
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
    for previous, current in _transitions(frames):
        tick, red, blue, _, _ = current
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
