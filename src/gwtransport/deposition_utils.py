"""
Utility Functions for the Deposition Module.

This module provides the clipped-trapezoid integral helpers (:func:`_clipped_linear_integral`
and :func:`_positive_part_integral`) used by the deposition module's banded weight builder to
integrate ``clip(y(x), y_lower, y_upper)`` over each cin bin of a streamtube's residence window.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt


def _positive_part_integral(
    a: npt.NDArray[np.floating], b: npt.NDArray[np.floating], w: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Integrate max(f(x), 0) from x=0 to x=w where f is linear from a to b.

    Parameters
    ----------
    a : ndarray
        Function values at x=0.
    b : ndarray
        Function values at x=w.
    w : ndarray
        Integration width.

    Returns
    -------
    ndarray
        Integral values.
    """
    both_pos = (a > 0) & (b > 0)
    only_a_pos = (a > 0) & ~both_pos
    only_b_pos = (b > 0) & ~both_pos

    abs_diff = np.abs(a - b)
    # Sentinel ``1.0`` avoids division by zero in the ``excess**2 / (2*safe_diff)``
    # branch when a == b; the surrounding ``np.where`` discards this branch
    # whenever both endpoints have the same sign (where the trapezoid formula
    # is used instead), so the sentinel value is never observed in the output.
    safe_diff = np.where(abs_diff > 0, abs_diff, 1.0)

    excess = np.where(only_a_pos, a, np.where(only_b_pos, b, 0.0))

    return np.where(
        both_pos,
        w * (a + b) / 2,
        w * excess**2 / (2 * safe_diff),
    )


def _clipped_linear_integral(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    w: npt.NDArray[np.floating],
    lo: float,
    hi: float,
) -> npt.NDArray[np.floating]:
    """
    Integrate clip(f(x), lo, hi) from x=0 to x=w where f is linear from a to b.

    Uses the identity ``clip(f) = f - max(f - hi, 0) + max(lo - f, 0)`` to
    compute the exact integral analytically.

    Parameters
    ----------
    a : ndarray
        Function values at x=0.
    b : ndarray
        Function values at x=w.
    w : ndarray
        Integration width.
    lo : float
        Lower clipping bound.
    hi : float
        Upper clipping bound.

    Returns
    -------
    ndarray
        Integral values.
    """
    raw = w * (a + b) / 2
    excess_above = _positive_part_integral(a - hi, b - hi, w)
    deficit_below = _positive_part_integral(lo - a, lo - b, w)
    return raw - excess_above + deficit_below
