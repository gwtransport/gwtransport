"""
Surface area computation utilities for groundwater transport analysis.

This module provides functions for computing surface areas and average heights
of geometric shapes used in groundwater transport calculations.
"""

import numpy as np


def compute_average_heights(x_edges, y_edges, y_lower, y_upper):
    """
    Compute average heights of clipped trapezoids.

    Trapezoids have vertical left and right sides, with corners at:
    - top-left: (y_edges[i, j], x_edges[j])
    - top-right: (y_edges[i, j+1], x_edges[j+1])
    - bottom-left: (y_edges[i+1, j], x_edges[j])
    - bottom-right: (y_edges[i+1, j+1], x_edges[j+1])

    Parameters
    ----------
    x_edges : ndarray
        1D array of x coordinates, shape (n_x,)
    y_edges : ndarray
        2D array of y coordinates, shape (n_y, n_x)
    y_lower : float
        Lower horizontal clipping bound
    y_upper : float
        Upper horizontal clipping bound

    Returns
    -------
    avg_heights : ndarray
        2D array of average heights (area/width) for each clipped trapezoid,
        shape (n_y-1, n_x-1)
    """
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
