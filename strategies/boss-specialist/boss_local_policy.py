"""SeedCup policy trained against our frozen local sparring model."""
from math import cos, hypot, radians, sin

from seedcup.game import rules

if __package__ and __package__.startswith("training"):
    from .boss_local_aim import choose
    from .boss_local_world import World
    from .boss_specialist_policy import Arena, bearing, blend, clamp, dodge_maneuver, incoming_paths, steer
else:
    from boss_local_aim import choose
    from boss_local_world import World
    from boss_specialist_policy import Arena, bearing, blend, clamp, dodge_maneuver, incoming_paths, steer


NAMES = ("max_flight", "min_hit_margin", "prepare_weight", "engage", "spacing", "anticipate", "dodge")
DEFAULT = (0., -.3, .5, .65, -.35, .6, .55)


def settings(vector):
    flight, margin, prepare, engage, spacing, anticipate, dodge = vector
    return ({"max_flight": round(8.5 + 4.5 * flight), "min_hit_margin": .1 * (1 + margin),
             "prepare_weight": .5 * (1 + prepare), "budget_ms": None},
            .5 * (1 + engage), 3.3 + 1.7 * spacing, .5 * (1 + anticipate), .5 * (1 + dodge))


def prospective_shell(arena, enemy, target, predicted, own):
    # The SDK turns and moves the arcade tank before spawning a new projectile.
    angle = radians(enemy.h + predicted["turn"] * rules.TURN_RATE * rules.DT)
    dx, dy = cos(angle), sin(angle)
    step = predicted["forward"] * rules.SPEED * rules.DT
    x, y = arena.move(*target, dx * step, dy * step, own)
    muzzle = rules.RADIUS + rules.MUZZLE_EXTENSION
    return {"mine": False, "x": x + muzzle * dx, "y": y + muzzle * dy, "dx": dx, "dy": dy}


def future_fire_paths(world, arena, own, target, action, predicted, anticipation):
    foe = world.game.tanks[world.enemy_side]
    x, y = own
    lookahead = round(4 * anticipation)
    if foe.reload > 1 + lookahead or hypot(x-target[0], y-target[1]) > 3 + 6 * anticipation:
        return []
    if foe.reload <= 1 and predicted["fire"]:
        return incoming_paths(arena, x, y, [prospective_shell(arena, foe, target, predicted, own)])
    if not lookahead or not arena.aim_clear(*own, *target):
        return []
    branch = world.branch()
    for delay in range(1, lookahead + 1):
        game = branch.step(action)
        if game.over:
            break
        foe = game.tanks[world.enemy_side]
        if foe.reload <= 1 and branch.enemy_action["fire"]:
            ours = game.tanks[world.own_side]
            shell = prospective_shell(arena, foe, (foe.x, foe.y), branch.enemy_action, (ours.x, ours.y))
            paths = incoming_paths(arena, x, y, [shell])[:5-delay]
            return [[] for _ in range(delay)] + paths if paths else []
    return []


def make_local_policy(base, opponent, vector):
    aiming, engage, separation, anticipate, dodge = settings(vector)

    def mirror(info):
        replica = opponent()
        replica.on_match_start(info)
        return replica

    class LocalSpecialist(base):
        def on_match_start(self, info):
            super().on_match_start(info)
            self._local_world = World(info, mirror)
            self._local_arena = Arena(info, route_home=False)

        def decide(self, obs):
            action = super().decide(obs)
            world, arena = self._local_world, self._local_arena
            _, predicted = world.update(obs)
            me, enemy, zone = obs["self"], obs["enemy"], obs["zone"]
            own, target = (me["x"], me["y"]), (enemy["x"], enemy["y"])
            x, y = own
            dx, dy = target[0] - x, target[1] - y
            distance = hypot(dx, dy)
            in_zone = zone["radius"] - hypot(x - zone["x"], y - zone["y"])
            zone_pressure = clamp((2.5 - in_zone) / 2.5, 0, 1)
            turn, forward = action["turn"], action["forward"]
            if engage and distance > separation and arena.aim_clear(x, y, *target):
                chase_turn, chase_speed = steer(bearing(dx, dy), me["heading"])
                weight = engage * clamp(distance - separation, 0, 1) * (1 - zone_pressure)
                turn = blend(turn, chase_turn, weight * clamp(me["reload"] * 2, 0, 1))
                forward = blend(forward, chase_speed, weight)
            move = {"forward": forward, "turn": turn, "fire": False}
            paths = incoming_paths(arena, x, y, obs["shells"])
            pending = future_fire_paths(world, arena, own, target, move, predicted, anticipate)
            while len(paths) < len(pending):
                paths.append([])
            for existing, future in zip(paths, pending):
                existing.extend(future)
            if paths:
                turn, forward = dodge_maneuver(arena, own, target, me["heading"], turn, forward,
                                               paths, zone, zone_pressure, dodge)
            action = choose(world, obs, {"forward": forward, "turn": turn, "fire": False}, aiming)
            if paths:
                safe_turn, safe_speed = dodge_maneuver(arena, own, target, me["heading"], action["turn"],
                                                       action["forward"], paths, zone, zone_pressure, dodge)
                if (safe_turn, safe_speed) != (action["turn"], action["forward"]):
                    action = {"turn": safe_turn, "forward": safe_speed, "fire": False}
            return {"forward": clamp(action["forward"]), "turn": clamp(action["turn"]), "fire": bool(action["fire"])}

    return LocalSpecialist
