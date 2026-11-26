"""Short-horizon position forecast from public opponent observations."""

from math import cos, hypot, radians, sin

import numpy as np
from seedcup.game import rules


def state_vector(obs, vx, vy, ticks):
    me, enemy = obs["self"], obs["enemy"]
    x, y = enemy["x"], enemy["y"]
    angle = radians(enemy["heading"])
    c, s = cos(angle), sin(angle)
    dx, dy = me["x"] - x, me["y"] - y
    distance = max(.01, hypot(dx, dy))
    step = rules.SPEED * rules.DT
    forward = (vx*c + vy*s) / step
    lateral = (vy*c - vx*s) / step
    # Coordinates in the opponent's body frame give both spawn sides the same representation.
    zone = obs["zone"]
    zx, zy = zone["x"] - x, zone["y"] - y
    incoming = (1., 0., 0.)
    for shell in obs["shells"]:
        if not shell["mine"]:
            continue
        sx, sy = x - shell["x"], y - shell["y"]
        along = sx*shell["dx"] + sy*shell["dy"]
        miss = sx*shell["dy"] - sy*shell["dx"]
        if 0 < along < 14*incoming[0] and abs(miss) < 2:
            incoming = (along / 14, miss / 2, shell["dy"]*c - shell["dx"]*s)
    vector = [forward, lateral, abs(forward), (dx*c + dy*s)/distance, (dy*c - dx*s)/distance,
              min(distance, 30)/15, (zx*c + zy*s)/16, (zy*c - zx*s)/16,
              min(1., max(-1., (zone["radius"] - hypot(zx, zy))/5)), enemy["hp"]/rules.HP,
              me["reload"]/(rules.RELOAD_TICKS*rules.DT), min(ticks, 16)/8, *incoming]
    return vector, c, s


def basis(x, projection, bias):
    return np.concatenate((np.ones((*x.shape[:-1], 1)), x, np.maximum(0., x @ projection + bias)), axis=-1)


def make_predictor(record):
    projection, bias, coef = (np.asarray(record[k], dtype=float) for k in ("projection", "bias", "coef"))

    def forecast(obs, vx, vy, ticks):
        ticks = min(16, max(1, ticks))
        x, c, s = state_vector(obs, vx, vy, ticks)
        f, l = basis(np.asarray(x), projection, bias) @ coef
        length = max(1., hypot(f, l))
        scale = ticks * rules.SPEED * rules.DT / length
        enemy = obs["enemy"]
        return enemy["x"] + (f*c - l*s)*scale, enemy["y"] + (f*s + l*c)*scale

    return forecast
