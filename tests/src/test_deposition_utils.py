"""Direct unit tests for the deposition clipped-trapezoid integral helpers.

``_clipped_linear_integral`` and ``_positive_part_integral`` are the load-bearing math of the
deposition banded weight builder (:func:`gwtransport.deposition.compute_deposition_weights`). Each
closed-form integral is checked against an independent fine-grid trapezoid reference, covering the
clip-crossing and full-clip geometries (formerly exercised through ``compute_average_heights``).
"""

import numpy as np
import pytest

from gwtransport.deposition_utils import _clipped_linear_integral, _positive_part_integral

_N = 200_001


def _trapezoid(y: np.ndarray, dx: float) -> float:
    return float(dx * (y.sum() - 0.5 * (y[0] + y[-1])))


def _clip_integral_reference(a: float, b: float, w: float, lo: float, hi: float) -> float:
    """Fine-grid trapezoid integral of ``clip(linear(a->b), lo, hi)`` over ``[0, w]``."""
    if w <= 0.0:
        return 0.0
    t = np.linspace(0.0, w, _N)
    return _trapezoid(np.clip(a + (b - a) * t / w, lo, hi), w / (_N - 1))


def _positive_part_reference(a: float, b: float, w: float) -> float:
    if w <= 0.0:
        return 0.0
    t = np.linspace(0.0, w, _N)
    return _trapezoid(np.maximum(a + (b - a) * t / w, 0.0), w / (_N - 1))


@pytest.mark.parametrize(
    ("a", "b", "lo", "hi"),
    [
        (1.0, 4.0, 0.0, 10.0),  # no clipping
        (3.0, 7.0, 0.0, 5.0),  # upper clip, left-to-right crossing
        (7.0, 3.0, 0.0, 5.0),  # upper clip, right-to-left crossing
        (-2.0, 4.0, 0.0, 10.0),  # lower clip, crossing
        (4.0, -2.0, 0.0, 10.0),  # lower clip, opposite crossing
        (8.0, 9.0, 0.0, 5.0),  # fully clipped above
        (-3.0, -1.0, 0.0, 5.0),  # fully clipped below
        (-2.0, 8.0, 1.0, 5.0),  # crosses both bounds
        (5.0, 5.0, 0.0, 10.0),  # flat, interior
    ],
)
def test_clipped_linear_integral_matches_reference(a, b, lo, hi):
    w = 2.0
    got = _clipped_linear_integral(np.array([a]), np.array([b]), np.array([w]), lo, hi)
    np.testing.assert_allclose(got, _clip_integral_reference(a, b, w, lo, hi), rtol=0, atol=1e-6)


def test_clipped_linear_integral_zero_width():
    got = _clipped_linear_integral(np.array([3.0]), np.array([7.0]), np.array([0.0]), 0.0, 5.0)
    np.testing.assert_allclose(got, 0.0, atol=0)


def test_clipped_linear_integral_vectorized():
    a = np.array([3.0, 7.0, -2.0])
    b = np.array([7.0, 3.0, 4.0])
    w = np.array([2.0, 2.0, 3.0])
    got = _clipped_linear_integral(a, b, w, 0.0, 5.0)
    expected = np.array([_clip_integral_reference(ai, bi, wi, 0.0, 5.0) for ai, bi, wi in zip(a, b, w, strict=True)])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("a", "b"),
    [(2.0, 5.0), (5.0, 2.0), (-3.0, 4.0), (4.0, -3.0), (-2.0, -1.0)],
)
def test_positive_part_integral_matches_reference(a, b):
    w = 2.0
    got = _positive_part_integral(np.array([a]), np.array([b]), np.array([w]))
    np.testing.assert_allclose(got, _positive_part_reference(a, b, w), rtol=0, atol=1e-6)
