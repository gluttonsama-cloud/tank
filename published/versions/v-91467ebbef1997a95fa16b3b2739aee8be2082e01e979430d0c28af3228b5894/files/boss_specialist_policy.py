"""State-dependent residual controls for the SeedCup arcade simulator."""

import sys
from collections import deque
from collections.abc import Callable, Sequence
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from json import loads
from math import atan2, ceil, cos, degrees, floor, hypot, radians, sin, sqrt
from pathlib import Path

from seedcup import Bot
from seedcup.game import rules
from seedcup.game.bots import clear_shot, ray_wall, wall_grid


PARAM_NAMES = (
    "lead", "curvature", "aim", "trigger", "range_target", "approach",
    "spacing", "orbit", "dodge", "wall", "zone", "repair",
)
DEFAULT_PARAMS = (0.0,) * len(PARAM_NAMES)


def load_strategy(folder: Path) -> type[Bot]:
    folder = folder.resolve()
    submission = loads((folder / "submission.json").read_text(encoding="utf-8"))
    filename, separator, class_name = submission["entrypoint"].rpartition(":")
    entry = (folder / filename).resolve()
    if (submission["runtime"] != "python" or not separator or not class_name.isidentifier()
            or entry.suffix != ".py" or not entry.is_relative_to(folder)):
        raise ValueError(f"Invalid Python entrypoint in {folder / 'submission.json'}")

    module_name = "_seedcup_snapshot_" + sha256(str(entry).encode()).hexdigest()
    if module_name not in sys.modules:
        spec = spec_from_file_location(module_name, entry)
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        old_path = sys.path[:]
        try:
            sys.path[:0] = [str(entry.parent), str(folder)]
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[module_name]
            raise
        finally:
            sys.path[:] = old_path
    cls = getattr(sys.modules[module_name], class_name)
    if not isinstance(cls, type) or not issubclass(cls, Bot):
        raise ValueError(f"The entrypoint {class_name} in {entry} must be a SeedCup Bot class")
    return cls


def clamp(value, low=-1.0, high=1.0):
    return max(low, min(high, value))


def angle_delta(target, heading):
    return (target - heading + 180) % 360 - 180


def bearing(dx, dy):
    return degrees(atan2(dy, dx))


def blend(current, target, weight):
    return current + (target - current) * weight


class Arena:
    def __init__(self, info, route_home):
        self.walls = wall_grid(info)
        self.handedness = -1 if info["side"] == "blue" else 1
        if route_home:
            zone = info["rules"]["zone"]
            self.home = self._home_distances(zone["x"], zone["y"], zone["final_radius"])

    def _is_wall(self, x, y):
        return not (0 <= x < rules.W and 0 <= y < rules.H) or self.walls[y * rules.W + x] == 2

    def _neighbors(self, x, y):
        side = self.handedness
        return ((x + side, y), (x - side, y), (x, y - 1), (x, y + 1))

    def _home_distances(self, cx, cy, radius):
        walkable = [(x, y) for y in range(rules.H) for x in range(rules.W) if not self._is_wall(x, y)]
        home = [(x, y) for x, y in walkable if hypot(x + .5 - cx, y + .5 - cy) <= radius * .55]
        if not home:
            home = [min(walkable, key=lambda p: (p[0] + .5 - cx) ** 2 + (p[1] + .5 - cy) ** 2)]
        distances = dict.fromkeys(home, 0)
        queue = deque(home)
        while queue:
            cell = queue.popleft()
            for next_cell in self._neighbors(*cell):
                if next_cell not in distances and not self._is_wall(*next_cell):
                    queue.append(next_cell)
                    distances[next_cell] = distances[cell] + 1
        return distances

    def waypoint(self, x, y):
        cell = floor(x), floor(y)
        if self.home[cell] == 0:
            return cell[0] + .5, cell[1] + .5
        next_cell = min((p for p in self._neighbors(*cell) if p in self.home), key=self.home.__getitem__)
        return next_cell[0] + .5, next_cell[1] + .5

    def aim_clear(self, x, y, ax, ay):
        return clear_shot(self.walls, x, y, ax, ay)

    def free(self, x, y):
        radius = rules.RADIUS
        for cy in range(floor(y - radius), floor(y + radius) + 1):
            for cx in range(floor(x - radius), floor(x + radius) + 1):
                if self._is_wall(cx, cy):
                    dx, dy = max(cx - x, 0, x - cx - 1), max(cy - y, 0, y - cy - 1)
                    if dx * dx + dy * dy < radius * radius:
                        return False
        return True

    def move(self, x, y, dx, dy, enemy):
        # The SDK checks the old enemy center and allows x- or y-only wall sliding.
        if not (dx or dy):
            return x, y
        ex, ey = enemy
        for nx, ny in ((x + dx, y + dy), (x + dx, y), (x, y + dy)):
            if (nx - ex) ** 2 + (ny - ey) ** 2 >= (2 * rules.RADIUS) ** 2 and self.free(nx, ny):
                return nx, ny
        return x, y

    def shot_distance(self, x, y, dx, dy, extent):
        return ray_wall(self.walls, x, y, dx, dy, extent)


def steer(target, heading):
    diff = angle_delta(target, heading)
    direction = 1.0
    if abs(diff) > 90:
        diff = angle_delta(target + 180, heading)
        direction = -1.0
    return clamp(diff / (rules.TURN_RATE * rules.DT)), direction * max(.15, cos(radians(diff)))


def motion(heading, turn, forward):
    end = radians(heading + turn * rules.TURN_RATE * rules.DT)
    return cos(end) * forward * rules.SPEED * rules.DT, sin(end) * forward * rules.SPEED * rules.DT


def flight_ticks(distance):
    speed = rules.SHELL_SPEED * rules.DT
    muzzle = rules.RADIUS + rules.MUZZLE_EXTENSION
    return clamp(ceil((distance - muzzle - rules.HIT_RADIUS) / speed), 1, 22)


def forecast(arena, own, enemy, displacement, step_turn, ticks, lead, curvature, proposed):
    x, y = own
    ex, ey = enemy
    vx, vy = displacement
    ticks *= max(0, .45 + .75 * lead)
    # Integrate a damped constant-turn arc using consecutive observed headings.
    arc = radians(step_turn * (.5 + .55 * curvature) * ticks * .5)
    chord = ticks * (sin(arc) / arc if arc else 1)
    dx = chord * (vx * cos(arc) - vy * sin(arc))
    dy = chord * (vx * sin(arc) + vy * cos(arc))
    if proposed is not None:
        weight = .55 + .45 * curvature
        dx = blend(dx, proposed[0] - ex, weight)
        dy = blend(dy, proposed[1] - ey, weight)
    ax, ay = ex + dx, ey + dy
    if arena.free(ax, ay) and arena.aim_clear(x, y, ax, ay):
        return ax, ay
    ax, ay = ex + dx * .45, ey + dy * .45
    if arena.free(ax, ay) and arena.aim_clear(x, y, ax, ay):
        return ax, ay
    return enemy


def shot_error(arena, own, heading, turn, forward, enemy, aim):
    px, py = arena.move(*own, *motion(heading, turn, forward), enemy)
    direction = radians(heading + turn * rules.TURN_RATE * rules.DT)
    dx, dy = cos(direction), sin(direction)
    muzzle = rules.RADIUS + rules.MUZZLE_EXTENSION
    sx, sy = px + muzzle * dx, py + muzzle * dy
    ax, ay = aim[0] - sx, aim[1] - sy
    along, across = ax * dx + ay * dy, abs(ax * dy - ay * dx)
    if along <= 0 or across >= rules.HIT_RADIUS:
        return float("inf")
    # The SDK advances shells six substeps per tick and checks the wall first.
    substep = rules.SHELL_SPEED * rules.DT / 6
    contact = sqrt(rules.HIT_RADIUS ** 2 - across ** 2)
    distance = max(substep, ceil((along - contact) / substep) * substep)
    if distance > along + contact or arena.shot_distance(sx, sy, dx, dy, distance) < distance:
        return float("inf")
    return across


def incoming_paths(arena, x, y, shells):
    shell_step = rules.SHELL_SPEED * rules.DT
    horizon = 5
    reach = rules.SPEED * rules.DT * horizon + rules.HIT_RADIUS
    segments = [[] for _ in range(horizon)]
    for shell in shells:
        if shell["mine"]:
            continue
        sx, sy, dx, dy = shell["x"], shell["y"], shell["dx"], shell["dy"]
        rx, ry = x - sx, y - sy
        along = rx * dx + ry * dy
        if not -reach < along < shell_step * horizon + reach or abs(rx * dy - ry * dx) >= reach:
            continue
        extent = shell_step * horizon
        if rules.SHELL_RANGE > 0:
            extent = min(extent, shell["range_left"])
        limit = arena.shot_distance(sx, sy, dx, dy, extent)
        if limit < along - reach:
            continue
        for tick, moves in enumerate(segments):
            start = tick * shell_step
            if start >= limit:
                break
            moves.append((sx + start * dx, sy + start * dy, dx, dy, min(shell_step, limit - start)))
    while segments and not segments[-1]:
        segments.pop()
    return segments


def dodge_maneuver(arena, own, enemy, heading, turn, forward, segments, zone, pressure, strength):
    def cost(steering, speed):
        x, y = own
        h = heading
        risk = 0.0
        for tick, moves in enumerate(segments):
            change = steering if tick < 3 else 0
            x, y = arena.move(x, y, *motion(h, change, speed), enemy)
            h += change * rules.TURN_RATE * rules.DT
            for sx, sy, dx, dy, length in moves:
                offset = clamp((x - sx) * dx + (y - sy) * dy, 0, length)
                separation = hypot(x - sx - offset * dx, y - sy - offset * dy)
                risk = max(risk, max(0, rules.HIT_RADIUS + .12 - separation) * (1 - .07 * tick))
        zone_cost = max(0, hypot(x - zone["x"], y - zone["y"]) - zone["radius"] + .4)
        return risk * (1 + 11 * strength) + .45 * pressure * zone_cost + .17 * abs(speed - forward) + .13 * abs(steering - turn)

    base = cost(turn, forward)
    if base <= .03:
        return turn, forward
    side = arena.handedness
    alternatives = [(steering, speed) for speed in (1.0, -1.0)
                    for steering in (turn, 0.0, side, -side)]
    alternatives.extend(((turn, forward), (turn, 0.0)))
    return min(alternatives, key=lambda action: cost(*action))


def wall_maneuver(arena, own, enemy, heading, turn, forward, strength):
    dx, dy = motion(heading, turn, forward)
    x, y = own
    if arena.free(x + dx, y + dy):
        return turn, forward

    def progress(steering, speed):
        px, py, h = x, y, heading
        for _ in range(3):
            px, py = arena.move(px, py, *motion(h, steering, speed), enemy)
            h += steering * rules.TURN_RATE * rules.DT
        affinity = .32 - .19 * strength
        return (hypot(px - x, py - y) + .45 * ((px - x) * dx + (py - y) * dy)
                - affinity * abs(steering - turn) - affinity * .7 * abs(speed - forward))

    nudge = arena.handedness * strength
    options = [(turn, forward), (clamp(turn + nudge), forward),
               (clamp(turn - nudge), forward), (turn, -forward)]
    return max(options, key=lambda action: progress(*action))


def make_policy(
    base_class: type[Bot],
    params: Sequence[float],
    *,
    predictor: Callable[[dict, float, float, int], tuple[float, float]] | None = None,
) -> type[Bot]:
    """Compose an SDK Bot with residuals and an optional shot target predictor."""
    weights = tuple(float(parameter) for parameter in params)
    if len(weights) != len(DEFAULT_PARAMS):
        raise ValueError(f"Expected {len(DEFAULT_PARAMS)} policy parameters, received {len(weights)}")
    for name, value in zip(PARAM_NAMES, weights, strict=True):
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"Policy parameter {name} must be in [-1, 1]")
    if weights == DEFAULT_PARAMS:
        return base_class

    lead, curvature, aim, trigger, range_target, approach, spacing, orbit, dodge, wall, zone_bias, repair = weights
    target_range = 6.5 + 5 * range_target
    aim_weight = max(abs(aim), .55 * abs(lead), .4 * abs(curvature))

    class ResidualPolicy(base_class):
        def on_match_start(self, info):
            super().on_match_start(info)
            self._residual_arena = Arena(info, route_home=zone_bias > 0)
            self._residual_previous = None
            self._residual_velocity = (0.0, 0.0)

        def decide(self, obs):
            action = super().decide(obs)
            me, enemy = obs["self"], obs["enemy"]
            x, y, heading = me["x"], me["y"], me["heading"]
            ex, ey, enemy_heading = enemy["x"], enemy["y"], enemy["heading"]
            own, target = (x, y), (ex, ey)
            distance = hypot(ex - x, ey - y)
            arena = self._residual_arena
            frame = obs["frame"]
            px, py, past_heading, past_frame = self._residual_previous or (ex, ey, enemy_heading, frame - 1)
            age = max(1, frame - past_frame)
            vx, vy = self._residual_velocity
            dx, dy = (ex - px) / age, (ey - py) / age
            velocity = blend(vx, dx, .75), blend(vy, dy, .75)
            self._residual_velocity = velocity
            self._residual_previous = ex, ey, enemy_heading, frame

            forward, turn, fire = action["forward"], action["turn"], action["fire"]
            visible = enemy["alive"] and arena.aim_clear(x, y, ex, ey)
            ready = me["reload"] <= rules.DT
            near_ready = clamp(1 - (me["reload"] - rules.DT) / ((3.5 + 2.5 * aim) * rules.DT), 0, 1)
            zone = obs["zone"]
            margin = zone["radius"] - hypot(x - zone["x"], y - zone["y"])
            approach_zone = clamp((2.5 - margin) / 2.5, 0, 1)

            if visible and (approach or spacing or orbit):
                radial = distance - target_range
                engage = approach * clamp((radial + 2) / 5, 0, 1)
                keep_range = spacing * clamp((2 - radial) / 5, 0, 1)
                lateral = abs(orbit) * clamp(1 - abs(radial) / 8, 0, 1)
                drive = engage - keep_range
                weight = max(abs(drive), lateral) * (1 - .78 * near_ready) * (1 - approach_zone)
                offset = degrees(atan2(lateral, drive)) * arena.handedness * (-1 if orbit < 0 else 1)
                movement_turn, movement_speed = steer(bearing(ex - x, ey - y) + offset, heading)
                turn = blend(turn, movement_turn, weight)
                forward = blend(forward, movement_speed, weight)

            chosen_aim = target
            if visible and (trigger or aim_weight) and near_ready > 0:
                step_turn = angle_delta(enemy_heading, past_heading) / age
                ticks = flight_ticks(distance)
                proposed = predictor(obs, dx, dy, ticks) if predictor is not None else None
                chosen_aim = forecast(arena, own, target, velocity, step_turn, ticks, lead, curvature, proposed)
                # Movement and rotation both precede firing in the game tick.
                next_x, next_y = arena.move(x, y, *motion(heading, turn, forward), target)
                ideal = angle_delta(bearing(chosen_aim[0] - next_x, chosen_aim[1] - next_y), heading)
                turn = blend(turn, clamp(ideal / (rules.TURN_RATE * rules.DT)), aim_weight * near_ready * (1 - .55 * approach_zone))

            if zone_bias > 0:
                lookahead = round((.6 + 6 * zone_bias) * rules.TICK_HZ)
                future_radius = rules.zone_radius(frame + lookahead)
                pressure = clamp((1.0 + zone["radius"] - future_radius - margin) / 2, 0, 1)
                if pressure:
                    wx, wy = arena.waypoint(x, y)
                    zone_turn, zone_speed = steer(bearing(wx - x, wy - y), heading)
                    weight = pressure * min(1, 1.6 * zone_bias)
                    turn, forward = blend(turn, zone_turn, weight), blend(forward, zone_speed, weight)
            elif zone_bias < 0 and visible:
                # Search less conservative behavior while still inside the circle.
                keep_facing = clamp((margin - .3) / 1.5, 0, 1) * approach_zone * -zone_bias
                turn = blend(turn, clamp(angle_delta(bearing(ex - x, ey - y), heading) / (rules.TURN_RATE * rules.DT)), keep_facing)
                forward *= 1 - keep_facing

            if wall and forward:
                turn, forward = wall_maneuver(arena, own, target, heading, turn, forward, abs(wall))

            can_repair = (repair > 0 and not visible and distance > 4 and margin > 1.5
                          and rules.HP - me["hp"] >= 28 - 23 * repair)
            if (dodge or can_repair) and obs["shells"]:
                incoming = incoming_paths(arena, x, y, obs["shells"])
            else:
                incoming = []
            if can_repair and not incoming:
                return action | {"forward": 0.0, "turn": 0.0, "fire": False}
            if dodge and incoming:
                strength = .18 + .8 * dodge if dodge > 0 else .18 * -dodge
                turn, forward = dodge_maneuver(arena, own, target, heading, turn, forward, incoming, zone, approach_zone, strength)

            if ready and visible and trigger:
                error = shot_error(arena, own, heading, turn, forward, target, chosen_aim)
                if trigger > 0:
                    fire = error <= .19 + .23 * trigger or (fire and trigger < .45)
                else:
                    fire = fire and error <= .42 + .27 * trigger
            return action | {"forward": float(forward), "turn": float(turn), "fire": bool(fire)}

    return ResidualPolicy
