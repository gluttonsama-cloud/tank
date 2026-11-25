from math import cos, radians, sin

import numpy as np
import pytest
from seedcup.game import rules

from training.boss_forecast import make_predictor, state_vector
from training.fit_boss_forecast import fit


def observed(heading):
    return {"self": {"x": 3., "y": 7., "reload": .2},
            "enemy": {"x": 7., "y": 6., "heading": heading, "hp": 80},
            "zone": {"x": 16., "y": 12., "radius": 18.}, "shells": []}


def test_exported_forecast_rotates_back_into_world_coordinates():
    n = len(state_vector(observed(0), 0., 0., 5)[0])
    coef = np.zeros((n + 2, 2))
    coef[0] = [1., 0.]
    predict = make_predictor({"projection": np.zeros((n, 1)), "bias": [0.], "coef": coef})
    assert predict(observed(90), 0., .25, 4) == pytest.approx((7., 7.))
    assert predict(observed(180), -.25, 0., 4) == pytest.approx((6., 6.))


def test_offline_metric_uses_the_same_predictions_as_the_exported_model():
    rows, observations = [], []
    for tick in range(70):
        heading = (tick * 17) % 360
        obs = observed(heading)
        vx, vy = (.14*cos(radians(heading)), .14*sin(radians(heading)))
        horizon = 1 + tick % 12
        features, c, s = state_vector(obs, vx, vy, horizon)
        reach = horizon * rules.SPEED * rules.DT
        rows.append([*features, .4, -.2, reach])
        target = (7 + reach*(.4*c + .2*s), 6 + reach*(.4*s - .2*c))
        observations.append((obs, vx, vy, horizon, target))
    model, scores = fit(np.asarray(rows[:50]), np.asarray(rows[50:]), seed=73)
    forecast = make_predictor(model)
    errors = [np.linalg.norm(np.asarray(forecast(obs, vx, vy, horizon)) - target)
              for obs, vx, vy, horizon, target in observations[50:]]
    assert scores["forecast"]["mae_cells"] == pytest.approx(np.mean(errors))
    assert scores["forecast"]["mae_cells"] < scores["constant_velocity"]["mae_cells"]
