"""Short, public-observation shot searches for the SeedCup arcade simulator."""

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin
from time import perf_counter

from seedcup.game import rules
from seedcup.game.engine import Shell, ang_diff, ray_wall
from seedcup.game.maps import cell_of

TURN = rules.TURN_RATE * rules.DT
TANK_STEP = rules.SPEED * rules.DT
SHELL_STEP = rules.SHELL_SPEED * rules.DT
SHELL_SUBSTEP = SHELL_STEP / 6
MUZZLE = rules.RADIUS + rules.MUZZLE_EXTENSION


@dataclass(frozen=True)
class AimSettings:
    max_flight: int = 10
    min_hit_margin: float = 0.045
    prepare_weight: float = 0.72
    branches: int = 8
    prepare_frames: int = 2
    budget_ms: float | None = 26


DEFAULT_SETTINGS = AimSettings()


def _turn(x, y, tx, ty, heading):
    return ang_diff(degrees(atan2(ty - y, tx - x)), heading) / TURN


def _clip(turn):
    return max(-1.0, min(1.0, turn))


def _visible(game_map, x, y, ex, ey, clearance):
    radius = hypot(ex - x, ey - y)
    return (
        radius <= clearance
        or ray_wall(game_map, x, y, (ex - x) / radius, (ey - y) / radius, radius - clearance)
        >= radius - clearance
    )


def _fallback(obs, action, direct, distance, visible, settings):
    me, enemy = obs["self"], obs["enemy"]
    zone = obs["zone"]
    safe = hypot(me["x"] - zone["x"], me["y"] - zone["y"]) + 0.75 < zone["radius"]
    x, y = me["x"], me["y"]
    threatened = any(
        not shell["mine"]
        and -0.2 < (x - shell["x"]) * shell["dx"] + (y - shell["y"]) * shell["dy"] < 4
        and abs((x - shell["x"]) * shell["dy"] - (y - shell["y"]) * shell["dx"]) < 0.75
        for shell in obs["shells"]
    )
    if not safe or not visible or abs(direct) > 4 or threatened:
        return action
    turn = action["turn"] + settings.prepare_weight * (_clip(direct) - action["turn"])
    forward = action["forward"]
    if abs(forward) < 0.1 and abs(direct) < 0.9 and distance > 2.2:
        # Move toward a shorter flight while the current firing lane is open.
        advance = radians(me["heading"] + turn * TURN)
        if (enemy["x"] - x) * cos(advance) + (enemy["y"] - y) * sin(advance) > 0:
            forward = 0.8 * settings.prepare_weight
    return action | {"turn": turn, "forward": forward}


def _follow(game, side, action, rollout_policy):
    if rollout_policy is not None:
        return rollout_policy(game.observe(side), action) | {"fire": False}
    me, enemy = game.tanks[side], game.tanks[1 - side]
    return {
        "forward": action["forward"],
        "turn": _clip(_turn(me.x, me.y, enemy.x, enemy.y, me.h)),
        "fire": False,
    }


def _passive_path(world, action, horizon, deadline, rollout_policy):
    branch = world.branch()
    game, side = branch.game, branch.own_side
    points = []
    move = action
    for _ in range(horizon):
        if game.over or perf_counter() >= deadline:
            break
        branch.step(move)
        enemy = game.tanks[1 - side]
        points.append((enemy.x, enemy.y))
        move = _follow(game, side, action, rollout_policy)
    return points


def _candidate_turns(game_map, me, action, path):
    x, y, heading = me["x"], me["y"], me["heading"]
    speed = action["forward"]
    for tick, (ex, ey) in enumerate(path, 1):
        distance = hypot(ex - x, ey - y)
        radius = distance - MUZZLE
        if not (
            (tick - 1) * SHELL_STEP - rules.HIT_RADIUS - TANK_STEP
            <= radius
            <= tick * SHELL_STEP + rules.HIT_RADIUS + TANK_STEP
        ):
            continue
        aim = _turn(x, y, ex, ey, heading)
        if abs(aim) > 1 + degrees(atan2(rules.HIT_RADIUS + TANK_STEP, distance)) / TURN:
            continue
        # Small offsets can lead a dodge induced by a shell in the next public frame.
        width = degrees(atan2(0.32, distance)) / TURN
        offsets = (0.0, width, -width, 2.3 * width, -2.3 * width, 4.0 * width, -4.0 * width)
        for offset in offsets:
            turn = _clip(aim + offset)
            direction = radians(heading + TURN * turn)
            dx, dy = cos(direction), sin(direction)
            # This cheap test only rejects candidates whose line is blocked even before the earliest hit.
            start = MUZZLE + max(0, TANK_STEP * speed)
            if (
                ray_wall(game_map, x + start * dx, y + start * dy, dx, dy, max(0.0, radius - 0.9))
                < radius - 0.9
            ):
                continue
            yield speed, turn
        for alternative in (0.0, 1.0, -1.0):
            if alternative != speed:
                yield alternative, _clip(aim)
    yield speed, action["turn"]


def _shot_points(game_map, shell):
    x, y = shell.x, shell.y
    for _ in range(6):
        x += shell.dx * SHELL_SUBSTEP
        y += shell.dy * SHELL_SUBSTEP
        if game_map.is_wall(*cell_of(x, y)):
            break
        yield x, y


def _hit_margin(game, events, side, shells, shot_id):
    target = game.tanks[1 - side]
    hit_points = iter(
        (event["x"], event["y"])
        for event in events
        if event["type"] == "hit" and event["side"] == target.side
    )
    expected = next(hit_points, None)
    # The SDK records hits in shell order. Assign the event before considering the next shell.
    for shell in shells:
        if expected is None:
            break
        points = tuple(_shot_points(game.map, shell))
        if not any(
            (round(x, 2), round(y, 2)) == expected and hypot(x - target.x, y - target.y) <= rules.HIT_RADIUS
            for x, y in points
        ):
            continue
        if shell.id == shot_id:
            return rules.HIT_RADIUS - min(hypot(x - target.x, y - target.y) for x, y in points)
        expected = next(hit_points, None)
    return None


def _trial(world, action, horizon, deadline, rollout_policy, *, correction=None):
    branch = world.branch()
    game, side = branch.game, branch.own_side
    move = action
    shot_id = None
    nearest = float("inf")
    for _ in range(horizon):
        if game.over or perf_counter() >= deadline:
            break
        # Game.step mutates every surviving Shell in place, so keep the pre-step path origins.
        old = [Shell(s.id, s.owner, s.x, s.y, s.dx, s.dy) for s in game.shells if s.owner == side]
        next_id = game._shell_id
        old_hits = game.tanks[side].hits
        branch.step(move)
        own, enemy = game.tanks[side], game.tanks[1 - side]
        if shot_id is None:
            if not own.fired:
                break
            shot_id = next_id + int(side == 1 and enemy.fired)
            dc, ds = cos(radians(own.h)), sin(radians(own.h))
            origin_x, origin_y = own.x, own.y
            old.append(Shell(shot_id, side, own.x + MUZZLE * dc, own.y + MUZZLE * ds, dc, ds))
        shot = next((shell for shell in game.shells if shell.id == shot_id), None)
        if correction is not None:
            start = next(shell for shell in old if shell.id == shot_id)
            along = (enemy.x - start.x) * start.dx + (enemy.y - start.y) * start.dy
            error = abs(along - SHELL_STEP * 0.5)
            if error < nearest and -0.5 < along < SHELL_STEP + 0.5:
                nearest = error
                source_heading = world.game.tanks[side].h
                correction[:] = [_clip(_turn(origin_x, origin_y, enemy.x, enemy.y, source_heading))]
        if shot is None:
            if own.hits > old_hits and (own.alive or not enemy.alive):
                return _hit_margin(game, branch.events, side, old, shot_id)
            break
        along = (enemy.x - shot.x) * shot.dx + (enemy.y - shot.y) * shot.dy
        if along < -rules.HIT_RADIUS:
            break
        move = _follow(game, side, action, rollout_policy)
    return None


def choose(world, obs, action, settings=None, *, rollout_policy=None):
    """Choose a shot from independent branches of the current public frame."""
    settings = DEFAULT_SETTINGS if settings is None else settings
    if not isinstance(settings, AimSettings):
        settings = AimSettings(**settings)
    deadline = float("inf") if settings.budget_ms is None else perf_counter() + settings.budget_ms / 1000
    me, enemy = obs["self"], obs["enemy"]
    if not me["alive"] or not enemy["alive"] or me["reload"] > (1 + settings.prepare_frames) * rules.DT:
        return action

    no_fire = action | {"fire": False}
    x, y, ex, ey = me["x"], me["y"], enemy["x"], enemy["y"]
    distance = hypot(ex - x, ey - y)
    direct = _turn(x, y, ex, ey, me["heading"])
    game_map = world.game.map
    visible = _visible(game_map, x, y, ex, ey, rules.HIT_RADIUS)
    fallback = _fallback(obs, no_fire, direct, distance, visible, settings)
    if me["reload"] > rules.DT:
        return fallback

    reach = MUZZLE + rules.HIT_RADIUS + TANK_STEP
    earliest = max(1, int((distance - reach) / (SHELL_STEP + TANK_STEP)))
    if earliest > settings.max_flight or abs(direct) > 2.3:
        return fallback
    if not visible and not action["fire"]:
        # Include the edges before giving up on a partially exposed opponent.
        span = rules.HIT_RADIUS + TANK_STEP * min(4, earliest)
        if not any(
            _visible(
                game_map,
                x,
                y,
                ex - (ey - y) * offset / distance,
                ey + (ex - x) * offset / distance,
                rules.HIT_RADIUS,
            )
            for offset in (-span, span)
        ):
            return fallback

    horizon = min(settings.max_flight, int((distance + reach) / (SHELL_STEP - TANK_STEP)) + 2)
    path = _passive_path(world, no_fire, horizon, deadline, rollout_policy)
    tried = set()
    candidates = iter(_candidate_turns(game_map, me, action, path))
    refined = None
    refinements = 0
    while len(tried) < settings.branches and perf_counter() < deadline:
        if refined is None:
            if (candidate := next(candidates, None)) is None:
                break
        else:
            candidate, refined = refined, None
        forward, turn = candidate
        key = round(forward, 3), round(turn, 3)
        if key in tried:
            continue
        tried.add(key)
        firing = action | {"forward": forward, "turn": turn, "fire": True}
        correction = [] if refinements < 2 else None
        margin = _trial(world, firing, horizon, deadline, rollout_policy, correction=correction)
        if margin is not None and margin >= settings.min_hit_margin:
            return firing
        if correction and abs(correction[0] - turn) > 0.025:
            refined = forward, correction[0]
            refinements += 1
    return fallback
