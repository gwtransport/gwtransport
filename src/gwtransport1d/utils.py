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


def find_consecutive_true_ranges(bool_series, *, timestamp_at_end_of_index=True):
    """
    Find consecutive segments in the given boolean series that are True, and return their start and end indices.

    Parameters
    ----------
    bool_series : array-like of bool
        A sequence of boolean values indicating where to identify
        consecutive True segments.
    timestamp_at_end_of_index : bool, optional
        Set to True if the timestamp is at the end of the period. At PWN the case for flow data.

    Returns
    -------
    array
        An array of shape (n_segments, 2) where each row contains the start
        and end index of a consecutive True segment.

    Notes
    -----
    If the series starts with True, this function accounts for the first
    index as a potential start. Likewise, if the series ends with True, that
    last index is considered a valid ending for a segment.
    """
    # Convert to numpy array if it's not already
    arr = np.asarray(bool_series)

    # Find the boundaries where values change
    # This gives us True where the value changes from False to True or True to False
    changes = np.diff(arr.astype(int))

    # Find start indices (where value changes from 0 to 1)
    start_idx = np.where(changes == 1)[0] if timestamp_at_end_of_index else np.where(changes == 1)[0] + 1

    # Find end indices (where value changes from 1 to -1)
    end_idx = np.where(changes == -1)[0]

    # Handle edge cases
    # If series starts with True, add 0 as start index
    if arr[0]:
        start_idx = np.insert(start_idx, 0, 0)

    # If series ends with True, add last index as end index
    if arr[-1]:
        end_idx = np.append(end_idx, len(arr) - 1)

    # Return as list of tuples (start, end)
    return np.stack((start_idx, end_idx), axis=-1)


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
