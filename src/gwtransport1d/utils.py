"""General utilities for the 1D groundwater transport model."""

import numpy as np
import pandas as pd
from scipy import interpolate


def linear_interpolate(x_ref, y_ref, x_query, left=None, right=None):
    """
    Linear interpolation on monotonically increasing data.

    Parameters
    ----------
    x_ref : array-like
        Reference vector with sorted x-values.
    y_ref : array-like
        Reference vector with y-values.
    x_query : array-like
        Query x-values. Array may have any shape.
    left : float, optional
        Value to return for x_query < x_ref[0].
        - If `left` is set to None, x_query = x_ref[0].
        - If `left` is set to a float, such as np.nan, this value is returned.
    right : float, optional
        Value to return for x_query > x_ref[-1].
        - If `right` is set to None, x_query = x_ref[-1].
        - If `right` is set to a float, such as np.nan, this value is returned.

    Returns
    -------
    array
        Interpolated y-values.
    """
    x_ref = np.asarray(x_ref)
    y_ref = np.asarray(y_ref)
    x_query = np.asarray(x_query)

    # Find indices where x_query would be inserted in x_ref
    idx_no_edges = np.searchsorted(x_ref, x_query)

    idx = np.clip(idx_no_edges, 1, len(x_ref) - 1)

    # Calculate interpolation weights
    x0 = x_ref[idx - 1]
    x1 = x_ref[idx]
    y0 = y_ref[idx - 1]
    y1 = y_ref[idx]

    # Perform linear interpolation
    weights = (x_query - x0) / (x1 - x0)
    y_query = y0 + weights * (y1 - y0)

    # Handle edge cases
    if left is None:
        y_query = np.where(x_query < x_ref[0], y_ref[0], y_query)
    if right is None:
        y_query = np.where(x_query > x_ref[-1], y_ref[-1], y_query)
    if isinstance(left, float):
        y_query = np.where(x_query < x_ref[0], left, y_query)
    if isinstance(right, float):
        y_query = np.where(x_query > x_ref[-1], right, y_query)

    return y_query


def interp_series(series, index_new, **interp1d_kwargs):
    """
    Interpolate a pandas.Series to a new index.

    Parameters
    ----------
    series : pandas.Series
        Series to interpolate.
    index_new : pandas.DatetimeIndex
        New index to interpolate to.
    interp1d_kwargs : dict, optional
        Keyword arguments passed to scipy.interpolate.interp1d. Default is {}.

    Returns
    -------
    pandas.Series
        Interpolated series.
    """
    series = series[series.index.notna() & series.notna()]
    dt = (series.index - series.index[0]) / pd.to_timedelta(1, unit="D")
    dt_interp = (index_new - series.index[0]) / pd.to_timedelta(1, unit="D")
    interp_obj = interpolate.interp1d(dt, series.values, bounds_error=False, **interp1d_kwargs)
    return interp_obj(dt_interp)


def diff(a, alignment="centered"):
    """Compute the cell widths for a given array of cell coordinates.

    If alignment is "centered", the coordinates are assumed to be centered in the cells.
    If alignment is "left", the coordinates are assumed to be at the left edge of the cells.
    If alignment is "right", the coordinates are assumed to be at the right edge of the cells.

    Parameters
    ----------
    a : array-like
        Input array.

    Returns
    -------
    array
        Array with differences between elements.
    """
    if alignment == "centered":
        mid = a[:-1] + (a[1:] - a[:-1]) / 2
        return np.concatenate((a[[1]] - a[[0]], mid[1:] - mid[:-1], a[[-1]] - a[[-2]]))
    if alignment == "left":
        return np.concatenate((a[1:] - a[:-1], a[[-1]] - a[[-2]]))
    if alignment == "right":
        return np.concatenate((a[[1]] - a[[0]], a[1:] - a[:-1]))

    msg = f"Invalid alignment: {alignment}"
    raise ValueError(msg)
