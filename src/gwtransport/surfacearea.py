"""
Surface Area Calculations for Streamline-Based Transport Analysis.

This module provides geometric utilities for computing surface areas and average
heights between streamlines in heterogeneous aquifer systems. These calculations
support direct estimation of pore volume distributions from streamline analysis,
providing an alternative to gamma distribution approximations.

Available functions:

- :func:`compute_average_heights` - Compute average heights of clipped trapezoids formed by
  streamlines. Trapezoids have vertical sides defined by x_edges and top/bottom edges defined
  by y_edges (2D array). Clipping bounds (y_lower, y_upper) restrict the integration domain.
  Returns area/width ratios representing average heights for use in pore volume calculations
  from streamline geometry.

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


def compute_average_heights(
    *, x_edges: npt.ArrayLike, y_edges: npt.ArrayLike, y_lower: float, y_upper: float
) -> npt.NDArray[np.floating]:
    """
    Compute average heights of clipped trapezoids.

    Trapezoids have vertical left and right sides, with corners at:
    - top-left: (y_edges[i, j], x_edges[j])
    - top-right: (y_edges[i, j+1], x_edges[j+1])
    - bottom-left: (y_edges[i+1, j], x_edges[j])
    - bottom-right: (y_edges[i+1, j+1], x_edges[j+1])

    The area is computed as the exact integral of
    ``clip(top(x), y_lower, y_upper) - clip(bottom(x), y_lower, y_upper)``
    where ``top(x)`` and ``bottom(x)`` are linear interpolants of the corner
    values, and ``clip`` restricts values to ``[y_lower, y_upper]``.

    Parameters
    ----------
    x_edges : numpy.ndarray
        1D array of x coordinates, shape (n_x,)
    y_edges : numpy.ndarray
        2D array of y coordinates, shape (n_y, n_x)
    y_lower : float
        Lower horizontal clipping bound
    y_upper : float
        Upper horizontal clipping bound

    Returns
    -------
    avg_heights : numpy.ndarray
        2D array of average heights (area/width) for each clipped trapezoid,
        shape (n_y-1, n_x-1)

    See Also
    --------
    gwtransport.advection.infiltration_to_extraction : Use computed areas in transport calculations

    Examples
    --------
    >>> import numpy as np
    >>> from gwtransport.surfacearea import compute_average_heights
    >>> # Create simple grid
    >>> x_edges = np.array([0.0, 1.0, 2.0])
    >>> y_edges = np.array([[0.0, 0.5, 1.0], [1.0, 1.5, 2.0], [2.0, 2.5, 3.0]])
    >>> # Compute average heights with clipping bounds
    >>> avg_heights = compute_average_heights(
    ...     x_edges=x_edges, y_edges=y_edges, y_lower=0.5, y_upper=2.5
    ... )
    >>> print(f"Shape: {avg_heights.shape}")
    Shape: (2, 2)
    """
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)

    y_tl, y_tr = y_edges[:-1, :-1], y_edges[:-1, 1:]
    y_bl, y_br = y_edges[1:, :-1], y_edges[1:, 1:]
    widths = np.diff(x_edges)

    top_integral = _clipped_linear_integral(y_tl, y_tr, widths, y_lower, y_upper)
    bottom_integral = _clipped_linear_integral(y_bl, y_br, widths, y_lower, y_upper)

    areas = np.maximum(top_integral - bottom_integral, 0.0)

    with np.errstate(invalid="ignore"):
        return areas / widths
