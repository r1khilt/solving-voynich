import numpy as np
import pytest

from voynich.workspace.geometry import coordinate_swap, matched_random_delta, normalize_rows


def test_nonorthogonal_coordinate_swap_and_preserved_complement():
    rows = normalize_rows([[1, 0, 0], [1, 2, 0]])
    hidden = np.array([2., 3., 7.])
    delta, report = coordinate_swap(hidden, rows)
    np.testing.assert_allclose(rows @ (hidden + delta), (rows @ hidden)[::-1], atol=1e-6)
    assert delta[2] == 0
    assert report["gram_condition"] < 10


def test_identity_strength_and_degenerate_frame():
    np.testing.assert_array_equal(coordinate_swap([1., 2.], np.eye(2), strength=0)[0], [0., 0.])
    with pytest.raises(ValueError, match="collinear"):
        coordinate_swap([1., 2.], [[1., 0.], [1., 0.]])


def test_random_control_norm_and_orthogonal_complement():
    reference = np.arange(7.)
    delta = matched_random_delta(reference, seed=7, orthogonal_to=np.eye(7)[:2])
    assert np.linalg.norm(delta) == pytest.approx(np.linalg.norm(reference), rel=1e-6)
    np.testing.assert_allclose(delta[:2], 0, atol=1e-7)
    np.testing.assert_array_equal(delta, matched_random_delta(reference, seed=7, orthogonal_to=np.eye(7)[:2]))
