import numpy as np
from pytest import approx

from training.boss_trajectory import HORIZONS, maneuver_displacements, rank_maneuvers
from training.fit_boss_trajectory import targets


def test_replay_future_positions_and_first_turn_are_mirrored():
    frames = []
    for tick in range(2 * HORIZONS[-1] + 1):
        right_turn = 18 if tick >= 5 else 0
        red = (2 + .2 * tick, 3 + .1 * tick, right_turn)
        blue = (26 - .2 * tick, 3 + .1 * tick, 180 - right_turn)
        frames.append((tick, red, blue, None, None))

    positions, onset = targets(frames, side=0)
    reflected, reflected_onset = targets(frames, side=1)
    np.testing.assert_allclose(positions, reflected)
    np.testing.assert_array_equal(onset, reflected_onset)
    assert positions[0] == approx(np.tile([.8, .4], (3, 1)))
    assert onset[0] == 4  # First right turn is 0.5 seconds away.
    assert onset[4] == 2
    assert onset[5] == 0
    assert onset[-1] == -1
    assert np.isnan(positions[-1]).all()
    assert np.isfinite(positions[-HORIZONS[0], 0]).all()
    assert np.isnan(positions[-HORIZONS[0], 1:]).all()


def test_candidate_paths_turn_before_moving_and_preserve_reverse_sign():
    result = maneuver_displacements(np.array([-1., 0., 1.]), np.array([-1., 0., 1.]))
    np.testing.assert_allclose(result[2, 1], np.tile([1., 0.], (3, 1)))
    np.testing.assert_allclose(result[0], -result[2])
    np.testing.assert_allclose(result[2, 0, :, 1], -result[2, 2, :, 1])
    assert (result[2, 2, :, 1] > 0).all()


def test_future_trajectory_and_turn_onset_both_affect_current_ranking():
    scores = np.zeros(21)
    weights = {"turn_centers": np.array([-1., -.2, 0., .2, 1.]), "speeds": np.array([-1., 0., 1.])}
    onset = np.zeros(7)
    positions = np.array([[0., 0.], [1., 0.], [1., 0.]])
    short = rank_maneuvers(scores, weights, positions, onset, displacement_weights=np.array([1., 0., 0.]))
    full = rank_maneuvers(scores, weights, positions, onset)
    assert short[:15].argmax() == 5  # A zero 0.3-second displacement alone favors a stop.
    assert full[:15].argmax() == 12  # The one- and two-second positions favor driving straight.
    onset[2] = 1  # An imminent right turn favors positive steering for the stopped maneuver.
    turn = rank_maneuvers(scores, weights, np.zeros((3, 2)), onset)
    assert turn[:15].argmax() == 9
    np.testing.assert_array_equal(full[15:], scores[15:])
