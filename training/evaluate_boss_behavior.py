"""用真实回放做终极档短程行为验证，并在相同地图和出生侧复跑整局。"""

import argparse
from dataclasses import replace
from math import cos, floor, hypot, radians, sin
from pathlib import Path
from statistics import fmean, median

from seedcup.client.loader import load_bot
from seedcup.game import Game, GameMap, parse_action, rules
from seedcup.game.engine import SIDES, Shell, Tank, ang_diff
from seedcup.game.maps import cell_x
from seedcup.game.replay import FORMAT_VERSION, load
from seedcup.game.runner import match_info
from tank import versions
from tank.workspace import project_root, read_json, write_json

if __package__:
    from .boss_behavior_matches import matched_games, rate, verify_unseen
    from .fit_boss_sparring import boss_side, recover_forward
else:
    from boss_behavior_matches import matched_games, rate, verify_unseen
    from fit_boss_sparring import boss_side, recover_forward


HORIZONS = (5, 10, 20)  # 正式规则为 10 Hz。
THREAT_TICKS = HORIZONS[0]
SHELL_STEP = rules.SHELL_SPEED * rules.DT
SHELL_SUBSTEPS = 6  # SDK 0.4.3 Game.step 的炮弹子步数。
SHELL_SUBSTEP = SHELL_STEP / SHELL_SUBSTEPS
SHELL_MATCH_TOLERANCE = 0.025  # 回放中位置为两位小数，方向为三位小数。
MOVE_CLASS_THRESHOLD = 0.08


def replay_action(game_map, before, after, side):
    own, other = before[side + 1], before[2 - side]
    following = after[side + 1]
    forward = float(recover_forward(game_map, own, other, following))
    turn = ang_diff(following[2], own[2]) / (rules.TURN_RATE * rules.DT)
    return {"forward": forward if forward == forward else None,
            "turn": min(1, max(-1, turn)), "fire": following[10] > own[10]}


def restored_tank(state, side, action, previous):
    x, y, heading, hp, alive, reload, still, repairing, damage, hits, shots, _ = state
    if previous is None:
        collided = False
    elif action["forward"] is None:
        collided = True
    else:
        step = action["forward"] * rules.SPEED * rules.DT
        h = radians(heading)
        collided = hypot(x - previous[0] - cos(h) * step,
                         y - previous[1] - sin(h) * step) > 0.025
    return Tank(side, x, y, heading, hp, bool(alive), reload, still, bool(repairing),
                collided=collided, fired=action is not None and action["fire"],
                dmg_dealt=damage, shots=shots, hits=hits)


def engine_action(action):
    return action | {"forward": 0.0} if action["forward"] is None else action


def last_action(action, tick):
    return None if action is None else engine_action(action) | {"frame": tick, "source_frame": tick}


def track_shells(frames):
    """按发射方、方向与下一帧预测位置匹配存活炮弹。"""
    tracks = []
    next_id = 1
    previous = {}
    for _, _, _, recorded, _ in frames:
        current = {}
        unmatched = previous.copy()
        for x, y, dx, dy, owner in recorded:
            candidates = []
            for key, old in unmatched.items():
                if old.owner == owner and hypot(old.dx - dx, old.dy - dy) <= 0.002:
                    error = hypot(old.x + old.dx * SHELL_STEP - x, old.y + old.dy * SHELL_STEP - y)
                    if error <= SHELL_MATCH_TOLERANCE:
                        candidates.append((error, key))
            if candidates:
                _, key = min(candidates)
                del unmatched[key]
            else:
                key = next_id
                next_id += 1
            current[key] = Shell(key, owner, x, y, dx, dy)
        tracks.append(current)
        previous = current
    return tracks, next_id


def projectile_positions(shell, count):
    x, y = shell.x, shell.y
    dx, dy = shell.dx * SHELL_SUBSTEP, shell.dy * SHELL_SUBSTEP
    for _ in range(count):
        x, y = x + dx, y + dy
        yield x, y


def impact_matches(shell, effect):
    kind, victim, x, y = effect
    if kind == "hit" and shell.owner == victim:
        return False
    # SDK records a spark half a substep before the wall cell.
    offset = SHELL_SUBSTEP / 2 if kind == "spark" else 0
    return any(hypot(x - sx + shell.dx * offset, y - sy + shell.dy * offset) <= SHELL_MATCH_TOLERANCE
               for sx, sy in projectile_positions(shell, SHELL_SUBSTEPS))


def replay_impacts(before, after, frame):
    """每颗消失的炮弹对应一个特效，顺序为已有炮弹后接本帧新发射的炮弹。"""
    impacts = [effect for effect in frame[4] if effect[0] in ("spark", "hit")]
    gone = [shell for key, shell in before.items() if key not in after]
    new = [shell for key, shell in after.items() if key not in before]
    live_by_owner = {shell.owner: shell for shell in new}
    if len(live_by_owner) != len(new):
        return None, gone
    ordered = gone.copy()
    for kind, owner, x, y in frame[4]:
        if kind != "fire":
            continue
        heading = radians(frame[owner + 1][2])
        fired = Shell(None, owner, x, y, cos(heading), sin(heading))
        live = live_by_owner.pop(owner, None)
        if live is None:
            ordered.append(fired)
        elif hypot(fired.x + fired.dx * SHELL_STEP - live.x,
                   fired.y + fired.dy * SHELL_STEP - live.y) > SHELL_MATCH_TOLERANCE:
            return None, gone
    if live_by_owner or len(ordered) != len(impacts):
        return None, gone
    pairs = list(zip(ordered, impacts, strict=True))
    if not all(impact_matches(shell, effect) for shell, effect in pairs):
        return None, gone
    return pairs, gone


def static_intercept(game_map, shell, x, y):
    """保持当前坐标，用 SDK 的离散碰撞顺序检测未来半秒。"""
    for i, (sx, sy) in enumerate(projectile_positions(shell, THREAT_TICKS * SHELL_SUBSTEPS), 1):
        if game_map.is_wall(cell_x(sx), floor(sy)):
            return None
        if (sx - x) ** 2 + (sy - y) ** 2 <= rules.HIT_RADIUS ** 2:
            return i * rules.DT / SHELL_SUBSTEPS
    return None


class ReplayTrace:
    def __init__(self, replay):
        self.meta = replay["match"]
        if rules.VARIANT or replay["format"] != FORMAT_VERSION or replay["rules"] != rules.as_dict():
            raise ValueError(f"回放 {self.meta['id']} 格式或规则与正式 SDK 不一致")
        self.game_map = GameMap.from_dict(replay["map"])
        self.side = boss_side(self.meta)
        self.frames = replay["frames"]
        if (len(self.frames) != replay["result"]["ticks"] + 1 or
                any(i != frame[0] for i, frame in enumerate(self.frames))):
            raise ValueError(f"回放 {self.meta['id']} 帧缺失或不连续")
        self.end = self.frames[-1][0]
        self.shells, self.next_shell_id = track_shells(self.frames)
        self.actions = [tuple(replay_action(self.game_map, a, b, side) for side in (0, 1))
                        for a, b in zip(self.frames, self.frames[1:])]

    def tank(self, tick, side):
        previous = self.frames[tick - 1][side + 1] if tick else None
        action = self.actions[tick - 1][side] if tick else None
        return restored_tank(self.frames[tick][side + 1], side, action, previous)

    def state(self, tick):
        return Game(self.game_map, tick=tick,
                    tanks=[self.tank(tick, i) for i in (0, 1)],
                    shells=[replace(shell) for shell in self.shells[tick].values()],
                    last_actions=[last_action(a, tick - 1) for a in self.actions[tick - 1]] if tick else [None, None],
                    _shell_id=self.next_shell_id)

    def info(self):
        return match_info(self.game_map, self.side, self.meta[SIDES[1 - self.side]]["name"],
                          game=self.meta["game"])

    def threats(self):
        hits, uncertain, unassigned, ambiguous_frames = {}, {}, 0, 0
        for tick in range(1, self.end + 1):
            frame = self.frames[tick]
            paired, gone = replay_impacts(self.shells[tick - 1], self.shells[tick], frame)
            if paired is None:
                ambiguous_frames += 1
                uncertain.update({shell.id: tick for shell in gone if shell.owner != self.side})
                unassigned += sum(effect[:2] == ["hit", self.side] for effect in frame[4])
                continue
            for shell, effect in paired:
                if effect[:2] == ["hit", self.side]:
                    if shell.id is None:
                        unassigned += 1
                    else:
                        hits[shell.id] = tick

        events, started = [], set()
        incomplete = 0
        for tick in range(self.end):
            x, y = self.frames[tick][self.side + 1][:2]
            for key, shell in self.shells[tick].items():
                if shell.owner == self.side or key in started:
                    continue
                eta = static_intercept(self.game_map, shell, x, y)
                if eta is None:
                    continue
                started.add(key)
                if tick + THREAT_TICKS > self.end:
                    incomplete += 1
                    continue
                horizon_end = tick + THREAT_TICKS
                reference_hit = None if uncertain.get(key, self.end + 1) <= horizon_end else (
                    hits.get(key, self.end + 1) <= horizon_end)
                events.append({"tick": tick, "shell_id": key, "static_eta_s": eta, "reference_hit": reference_hit})
        return events, {"reference_hits_unassigned_to_observed_shell": unassigned,
                        "reference_ambiguous_impact_frames": ambiguous_frames,
                        "threats_without_full_window": incomplete}


def action_class(forward):
    if forward is None:
        return None
    if abs(forward) <= MOVE_CLASS_THRESHOLD:
        return "stop"
    return "forward" if forward > 0 else "reverse"


def branch(trace, factory, start, horizons, events, history_frames):
    side, opposing = trace.side, 1 - trace.side
    bot = factory() if factory is not None else None
    if bot is not None:
        bot.on_match_start(trace.info())
        for tick in range(max(0, start - history_frames), start):
            bot.decide(trace.state(tick).observe(side))

    game = trace.state(start)
    first = None
    endpoint_errors, result = [], {}
    predicted_hits = {event["shell_id"]: False for event in events}
    for elapsed in range(1, max(horizons) + 1):
        tick = start + elapsed - 1
        if game.tanks[side].alive:
            actions = [engine_action(a) for a in trace.actions[tick]]
            if bot is not None:
                action, invalid = parse_action(bot.decide(game.observe(side)))
                if invalid:
                    raise ValueError(f"回放 {trace.meta['id']} 第 {tick} 帧出现无效预测动作")
                actions[side] = action
            if first is None:
                first = actions[side]
            measure_hits = bool(events) and elapsed <= THREAT_TICKS
            if measure_hits:
                active_ids = [shell.id for shell in game.shells]
                first_new_id = game._shell_id
            fx = game.step(actions)
            if measure_hits:
                remaining = {shell.id for shell in game.shells}
                active_ids.extend(range(first_new_id, game._shell_id))
                gone = [key for key in active_ids if key not in remaining]
                impacts = [effect for effect in fx if effect["type"] in ("spark", "hit")]
                for key, effect in zip(gone, impacts, strict=True):
                    if key in predicted_hits and effect["type"] == "hit" and effect["side"] == side:
                        predicted_hits[key] = True

            opponent = trace.tank(tick + 1, opposing)
            simulated = game.tanks[opposing]
            endpoint_errors.append(hypot(opponent.x - simulated.x, opponent.y - simulated.y))
            # The opponent's path, reload and observed health are exogenous conditioning data.
            game.tanks[opposing] = opponent
            game.over, game.result = False, None

        if elapsed in horizons:
            tank = game.tanks[side]
            observed = trace.frames[start + elapsed][side + 1]
            result[elapsed] = {
                "position_error_cells": hypot(tank.x - observed[0], tank.y - observed[1]),
                "heading_error_deg": abs(ang_diff(tank.h, observed[2])),
                "health_error_hp": tank.hp - observed[3],
                "predicted_dead": not tank.alive,
                "reference_dead": not bool(observed[4]),
                "opponent_endpoint_error_cells_mean": fmean(endpoint_errors),
                "opponent_endpoint_error_cells_max": max(endpoint_errors),
                "prediction_xy": [tank.x, tank.y],
            }

    scored_events = []
    for event in events:
        key = event["shell_id"]
        shell = trace.shells[start][key]
        x, y = trace.frames[start][side + 1][:2]
        ox, oy = trace.frames[start + THREAT_TICKS][side + 1][:2]
        px, py = result[THREAT_TICKS]["prediction_xy"]
        actual = trace.actions[start][side]
        scored_events.append(event | {
            "reference_perpendicular_cells": -(ox - x) * shell.dy + (oy - y) * shell.dx,
            "predicted_perpendicular_cells": -(px - x) * shell.dy + (py - y) * shell.dx,
            "predicted_hit": predicted_hits[key],
            "reference_first_forward_class": action_class(actual["forward"]),
            "predicted_first_forward_class": action_class(first["forward"]),
            "first_turn_error_deg": abs(first["turn"] - actual["turn"]) * rules.TURN_RATE * rules.DT,
            "half_second_position_error_cells": result[THREAT_TICKS]["position_error_cells"],
        })
    return result, scored_events


def distribution(values):
    values = list(values)
    return {"n": len(values), "mean": fmean(values) if values else None,
            "median": median(values) if values else None, "max": max(values) if values else None}


def summarize_trajectory(samples, threats):
    def per_horizon(horizon):
        rows = [row for row in samples if row["horizon_ticks"] == horizon]
        return {
            "position_error_cells": distribution(r["position_error_cells"] for r in rows),
            "heading_error_deg": distribution(r["heading_error_deg"] for r in rows),
            "health_error_hp": distribution(abs(r["health_error_hp"]) for r in rows),
            "opponent_endpoint_error_cells": distribution(r["opponent_endpoint_error_cells_mean"] for r in rows),
            "opponent_endpoint_error_cells_max": max((r["opponent_endpoint_error_cells_max"] for r in rows), default=None),
            "reference_deaths": sum(r["reference_dead"] for r in rows),
            "predicted_deaths": sum(r["predicted_dead"] for r in rows),
        }

    comparable = [e for e in threats if e["reference_hit"] is not None and e["predicted_hit"] is not None]
    labels = [e for e in threats if e["reference_first_forward_class"] is not None]
    missed = sum(not e["reference_hit"] and e["predicted_hit"] for e in comparable)
    extra = sum(e["reference_hit"] and not e["predicted_hit"] for e in comparable)
    return {"trajectory": {str(h * rules.DT): per_horizon(h) for h in HORIZONS}, "threats": {
        "events": len(threats), "events_with_paired_hit_outcome": len(comparable),
        "reference_hits": sum(e["reference_hit"] for e in comparable),
        "predicted_hits": sum(e["predicted_hit"] for e in comparable),
        "reference_evaded_but_predicted_hit": missed, "reference_hit_but_predicted_evaded": extra,
        "hit_outcome_accuracy": rate(len(comparable) - missed - extra, len(comparable)),
        "first_forward_class_accuracy": rate(sum(e["reference_first_forward_class"] == e["predicted_first_forward_class"]
                                                   for e in labels), len(labels)),
        "first_forward_class_labels": len(labels),
        "first_turn_error_deg": distribution(e["first_turn_error_deg"] for e in threats),
        "perpendicular_error_cells": distribution(abs(e["reference_perpendicular_cells"] - e["predicted_perpendicular_cells"])
                                                  for e in threats),
        "half_second_position_error_cells": distribution(e["half_second_position_error_cells"] for e in threats),
    }}


def evaluate_trace(trace, factory, stride_frames, history_frames):
    grid = set(range(0, trace.end - HORIZONS[0] + 1, stride_frames))
    events, coverage = trace.threats()
    by_anchor = {}
    for event in events:
        by_anchor.setdefault(event["tick"], []).append(event)
    samples, scored_events = [], []
    for start in sorted(grid | by_anchor.keys()):
        horizons = [h for h in HORIZONS if start + h <= trace.end] if start in grid else (THREAT_TICKS,)
        points, scored = branch(trace, factory, start, horizons, by_anchor.get(start, ()), history_frames)
        if start in grid:
            samples.extend({"tick": start, "horizon_ticks": h, **value} for h, value in points.items())
        scored_events.extend(scored)
    return samples, scored_events, coverage


def evaluate_replays(traces, factory, stride_frames, history_frames):
    summaries, all_samples, all_events = [], [], []
    for trace in traces:
        samples, events, coverage = evaluate_trace(trace, factory, stride_frames, history_frames)
        summaries.append({"id": trace.meta["id"], "seed": trace.game_map.seed, "side": SIDES[trace.side],
                          **coverage, **summarize_trajectory(samples, events),
                          "samples": samples, "threat_events": events})
        all_samples.extend(samples)
        all_events.extend(events)
    return summarize_trajectory(all_samples, all_events) | {"replays": summaries}


def trajectory_report(root, models, ids, stride_frames, history_frames, require_unseen=False, calibrate=False):
    if len(set(models)) != len(models) or len(set(ids)) != len(ids):
        raise ValueError("模型快照或官网回放 ID 重复")
    if stride_frames <= 0 or history_frames < 0:
        raise ValueError("--stride-frames 须为正整数，--history-frames 须为非负整数")
    replays = []
    for replay_id in ids:
        replay = load(root / ".local/online/replays" / f"{replay_id}.json.gz")
        if replay["match"]["id"] != replay_id:
            raise ValueError(f"回放内的对局 ID 与文件名不一致 {replay_id}")
        replays.append(replay)
    traces = [ReplayTrace(replay) for replay in replays]
    reports = {}
    for model in models:
        if model == "baseline":
            factory = load_bot(model)
        else:
            versions.verify(root, model)
            folder = versions.snapshot_data(root, model)
            if require_unseen:
                verify_unseen(root, model, replays)
            spec = read_json(folder / "submission.json")
            path, _, name = spec["entrypoint"].partition(":")
            if spec["runtime"] != "python" or not name:
                raise ValueError(f"行为验证只支持类形式的 Python 模型快照 {model}")
            factory = load_bot(f"{folder / path}:{name}")
        reports[model] = evaluate_replays(traces, factory, stride_frames, history_frames)

    report = {"rules": rules.as_dict()["version"], "ids": ids, "stride_frames": stride_frames,
            "history_frames": history_frames, "require_unseen": require_unseen,
            "method": {
                "rollout": "每个锚点用回放预热历史，从记录的双方状态和场上炮弹起步；用 SDK 逐帧推进，被评模型闭环，另一方每帧校正至回放轨迹。",
                "limits": "回放坐标和角度有舍入，碰撞、指令与指令来源帧需推断；另一方只在帧间对齐，帧内碰撞与炮口位置可能偏差。记录双方仍在赛中的锚点，模拟死亡后停在死亡位置。",
                "threat_event": "每颗已见来弹只记第一次用 SDK 离散墙体和命中检测发现原地会在 0.5 秒内中弹的时刻；按 SDK 已有弹、同帧新弹的处理顺序关联消失炮弹与命中或撞墙特效。",
                "threat_limits": "同帧出膛即命中的炮弹不在已见来弹内；回放归属顺序或近似位置无法校验时，对应事件不计算命中一致率。事件可能在同一局内相关。",
            }, "models": reports}
    if calibrate:
        report["replay_action_calibration"] = evaluate_replays(traces, None, stride_frames, history_frames)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    trajectories = modes.add_parser("trajectories", help="从官网回放中的相同状态分支")
    trajectories.add_argument("--models", nargs="+", required=True)
    trajectories.add_argument("--ids", nargs="+", type=int, required=True)
    trajectories.add_argument("--stride-frames", type=int, default=10)
    trajectories.add_argument("--history-frames", type=int, default=20)
    trajectories.add_argument("--require-unseen", action="store_true")
    trajectories.add_argument("--calibrate", action="store_true",
                              help="把从相邻回放恢复的终极档动作交给引擎，估计重建误差")
    trajectories.add_argument("--output", type=Path, required=True)
    compare = modes.add_parser("compare", help="官网已有的对照在本地按地图与出生侧复跑")
    compare.add_argument("--models", nargs="+", required=True)
    compare.add_argument("--candidates", nargs="+", required=True)
    compare.add_argument("--seeds", nargs="+", type=int, required=True)
    compare.add_argument("--ids", nargs="+", type=int, help="只用指定回放，每个策略、地图和出生侧选一局")
    compare.add_argument("--require-unseen", action="store_true", help="校验回放与模型训练及选模地图无重叠")
    compare.add_argument("--save-replays", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = project_root()
    if args.mode == "trajectories":
        report = trajectory_report(root, args.models, args.ids, args.stride_frames, args.history_frames,
                                   args.require_unseen, args.calibrate)
    else:
        report = matched_games(root, args.models, args.candidates, args.seeds, args.save_replays,
                               replay_ids=args.ids, require_unseen=args.require_unseen)
    write_json(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
