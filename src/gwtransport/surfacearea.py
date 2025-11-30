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
    # Convert inputs to arrays
    x_edges = np.asarray(x_edges)
    y_edges = np.asarray(y_edges)

    y_tl, y_tr = y_edges[:-1, :-1], y_edges[:-1, 1:]
    y_bl, y_br = y_edges[1:, :-1], y_edges[1:, 1:]
    widths = np.diff(x_edges)

    # Handle complete outside cases
    all_corners = np.stack([y_tl, y_tr, y_bl, y_br], axis=-1)
    outside_mask = np.all(all_corners >= y_upper, axis=-1) | np.all(all_corners <= y_lower, axis=-1)

    # Vectorized area calculation using shoelace formula for clipped quadrilaterals
    y_tl_c, y_tr_c = np.clip(y_tl, y_lower, y_upper), np.clip(y_tr, y_lower, y_upper)
    y_bl_c, y_br_c = np.clip(y_bl, y_lower, y_upper), np.clip(y_br, y_lower, y_upper)

    # Use exact shoelace formula for quadrilateral with vertices:
    # (0, y_bl_c), (width, y_br_c), (width, y_tr_c), (0, y_tl_c)
    areas = 0.5 * widths * np.abs(y_bl_c + y_br_c - y_tl_c - y_tr_c)

    # Enhanced correction for cases where edges cross clipping bounds
    # This handles the specific geometry of sloped edge intersections
    top_crosses = ((y_tl - y_upper) * (y_tr - y_upper)) < 0
    correction = np.where(
        top_crosses & (y_tl > y_upper) & (y_tr < y_upper),
        widths * (y_tl - y_upper) ** 2 / (2 * np.maximum(y_tl - y_tr, 1e-15)),
        0,
    )
    areas += correction
    areas = np.where(outside_mask, 0, areas)

    return areas / widths
