"""
General Utilities for 1D Groundwater Transport Modeling.

This module provides general-purpose utility functions for time series manipulation,
interpolation, numerical operations, and data processing used throughout the gwtransport
package. Functions include linear interpolation/averaging, bin overlap calculations,
underdetermined system solvers, and external data retrieval.

Available functions:

- :func:`linear_interpolate` - Linear interpolation using numpy's optimized interp function.
  Automatically handles unsorted data with configurable extrapolation (None for clamping,
  float for constant values). Handles multi-dimensional query arrays.

- :func:`interp_series` - Interpolate pandas Series to new DatetimeIndex using
  scipy.interpolate.interp1d. Automatically filters NaN values and converts datetime to
  numerical representation.

- :func:`linear_average` - Compute average values of piecewise linear time series between
  specified x-edges. Supports 1D or 2D edge arrays for batch processing. Handles NaN values
  and offers multiple extrapolation methods ('nan', 'outer', 'raise').

- :func:`diff` - Compute cell widths from cell coordinate arrays with configurable alignment
  ('centered', 'left', 'right'). Returns widths matching input array length.

- :func:`partial_isin` - Calculate fraction of each input bin overlapping with each output bin.
  Returns dense matrix where element (i,j) represents overlap fraction. Uses vectorized
  operations for efficiency.

- :func:`time_bin_overlap` - Calculate fraction of time bins overlapping with specified time
  ranges. Similar to partial_isin but for time-based bin overlaps with list of (start, end)
  tuples.

- :func:`combine_bin_series` - Combine two binned series onto common set of unique edges. Maps
  values from original bins to new combined structure with configurable extrapolation ('nearest'
  or float value).

- :func:`compute_time_edges` - Compute DatetimeIndex of time bin edges from explicit edges,
  start times, or end times. Validates consistency with expected number of bins and handles
  uniform spacing extrapolation.

- :func:`solve_underdetermined_system` - Solve underdetermined linear system (Ax = b, m < n)
  with nullspace regularization. Handles NaN values by row exclusion. Supports built-in
  objectives ('squared_differences', 'summed_differences') or custom callable objectives.

- :func:`get_soil_temperature` - Download soil temperature data from KNMI weather stations with
  automatic caching. Supports stations 260 (De Bilt), 273 (Marknesse), 286 (Nieuw Beerta),
  323 (Wilhelminadorp). Returns DataFrame with columns TB1-TB5, TNB1-TNB2, TXB1-TXB2 at various
  depths. Daily cache prevents redundant downloads.

- :func:`generate_failed_coverage_badge` - Generate SVG badge indicating failed coverage using
  genbadge library. Used in CI/CD workflows.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from scipy import interpolate
from scipy.linalg import null_space
from scipy.optimize import minimize

cache_dir = Path(__file__).parent.parent.parent / "cache"


def linear_interpolate(
    *,
    x_ref: npt.ArrayLike,
    y_ref: npt.ArrayLike,
    x_query: npt.ArrayLike,
    left: float | None = None,
    right: float | None = None,
) -> npt.NDArray[np.floating]:
    """
    Linear interpolation using numpy's optimized interp function.

    Automatically handles unsorted reference data by sorting it first.

    Parameters
    ----------
    x_ref : array-like
        Reference x-values. If unsorted, will be automatically sorted.
    y_ref : array-like
        Reference y-values corresponding to x_ref.
    x_query : array-like
        Query x-values where interpolation is needed. Array may have any shape.
    left : float, optional
        Value to return for x_query < x_ref[0].

        - If ``left=None``: clamp to y_ref[0] (default)
        - If ``left=float``: use specified value (e.g., ``np.nan``)

    right : float, optional
        Value to return for x_query > x_ref[-1].

        - If ``right=None``: clamp to y_ref[-1] (default)
        - If ``right=float``: use specified value (e.g., ``np.nan``)

    Returns
    -------
    numpy.ndarray
        Interpolated y-values with the same shape as x_query.

    Examples
    --------
    Basic interpolation with clamping (default):

    >>> import numpy as np
    >>> from gwtransport.utils import linear_interpolate
    >>> x_ref = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y_ref = np.array([10.0, 20.0, 30.0, 40.0])
    >>> x_query = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    >>> linear_interpolate(x_ref=x_ref, y_ref=y_ref, x_query=x_query)
    array([10., 15., 25., 35., 40.])

    Using NaN for extrapolation:

    >>> linear_interpolate(
    ...     x_ref=x_ref, y_ref=y_ref, x_query=x_query, left=np.nan, right=np.nan
    ... )
    array([nan, 15., 25., 35., nan])

    Handles unsorted reference data automatically:

    >>> x_unsorted = np.array([3.0, 1.0, 4.0, 2.0])
    >>> y_unsorted = np.array([30.0, 10.0, 40.0, 20.0])
    >>> linear_interpolate(x_ref=x_unsorted, y_ref=y_unsorted, x_query=x_query)
    array([10., 15., 25., 35., 40.])

    See Also
    --------
    interp_series : Interpolate pandas Series with datetime index
    """
    # Convert inputs to arrays
    x_ref = np.asarray(x_ref)
    y_ref = np.asarray(y_ref)
    x_query = np.asarray(x_query)

    # Sort reference data to ensure monotonic ordering
    sort_idx = np.argsort(x_ref)
    x_ref_sorted = x_ref[sort_idx]
    y_ref_sorted = y_ref[sort_idx]

    # Default behavior (left=None, right=None) clamps to boundary values
    return np.interp(x_query, x_ref_sorted, y_ref_sorted, left=left, right=right)


def interp_series(*, series: pd.Series, index_new: pd.DatetimeIndex, **interp1d_kwargs: object) -> pd.Series:
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
    return pd.Series(interp_obj(dt_interp), index=index_new)


def diff(*, a: npt.ArrayLike, alignment: str = "centered") -> npt.NDArray[np.floating]:
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
    numpy.ndarray
        Array with differences between elements.
    """
    # Convert input to array
    a = np.asarray(a)

    if alignment == "centered":
        mid = a[:-1] + (a[1:] - a[:-1]) / 2
        return np.concatenate((a[[1]] - a[[0]], mid[1:] - mid[:-1], a[[-1]] - a[[-2]]))
    if alignment == "left":
        return np.concatenate((a[1:] - a[:-1], a[[-1]] - a[[-2]]))
    if alignment == "right":
        return np.concatenate((a[[1]] - a[[0]], a[1:] - a[:-1]))

    msg = f"Invalid alignment: {alignment}"
    raise ValueError(msg)


def linear_average(
    *,
    x_data: npt.ArrayLike,
    y_data: npt.ArrayLike,
    x_edges: npt.ArrayLike,
    extrapolate_method: str = "nan",
) -> npt.NDArray[np.float64]:
    """
    Compute the average value of a piecewise linear time series between specified x-edges.

    Parameters
    ----------
    x_data : array-like
        x-coordinates of the time series data points, must be in ascending order
    y_data : array-like
        y-coordinates of the time series data points
    x_edges : array-like
        x-coordinates of the integration edges. Can be 1D or 2D.
        - If 1D: shape (n_edges,). Can be 1D or 2D.
        - If 1D: shape (n_edges,), must be in ascending order
        - If 2D: shape (n_series, n_edges), each row must be in ascending order
        - If 2D: shape (n_series, n_edges), each row must be in ascending order
    extrapolate_method : str, optional
        Method for handling extrapolation. Default is 'nan'.
        - 'outer': Extrapolate using the outermost data points.
        - 'nan': Extrapolate using np.nan.
        - 'raise': Raise an error for out-of-bounds values.

    Returns
    -------
    numpy.ndarray
        2D array of average values between consecutive pairs of x_edges.
        Shape is (n_series, n_bins) where n_bins = n_edges - 1.
        If x_edges is 1D, n_series = 1.

    Examples
    --------
    >>> import numpy as np
    >>> from gwtransport.utils import linear_average
    >>> x_data = [0, 1, 2, 3]
    >>> y_data = [0, 1, 1, 0]
    >>> x_edges = [0, 1.5, 3]
    >>> linear_average(
    ...     x_data=x_data, y_data=y_data, x_edges=x_edges
    ... )  # doctest: +ELLIPSIS
    array([[0.666..., 0.666...]])

    >>> x_edges_2d = [[0, 1.5, 3], [0.5, 2, 3]]
    >>> linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges_2d)
    array([[0.66666667, 0.66666667],
           [0.91666667, 0.5       ]])
    """
    # Convert inputs to numpy arrays
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    x_edges = np.asarray(x_edges, dtype=float)

    # Ensure x_edges is always 2D
    if x_edges.ndim == 1:
        x_edges = x_edges[np.newaxis, :]
    elif x_edges.ndim != 2:  # noqa: PLR2004
        msg = "x_edges must be 1D or 2D array"
        raise ValueError(msg)

    # Input validation
    if len(x_data) != len(y_data) or len(x_data) == 0:
        msg = "x_data and y_data must have the same length and be non-empty"
        raise ValueError(msg)
    if x_edges.shape[1] < 2:  # noqa: PLR2004
        msg = "x_edges must contain at least 2 values in each row"
        raise ValueError(msg)
    if not np.all(np.diff(x_data) >= 0):
        msg = "x_data must be in ascending order"
        raise ValueError(msg)
    if not np.all(np.diff(x_edges, axis=1) >= 0):
        msg = "x_edges must be in ascending order along each row"
        raise ValueError(msg)

    # Filter out NaN values
    show = ~np.isnan(x_data) & ~np.isnan(y_data)
    if show.sum() < 2:  # noqa: PLR2004
        if show.sum() == 1 and extrapolate_method == "outer":
            # For single data point with outer extrapolation, use constant value
            constant_value = y_data[show][0]
            return np.full(shape=(x_edges.shape[0], x_edges.shape[1] - 1), fill_value=constant_value)
        return np.full(shape=(x_edges.shape[0], x_edges.shape[1] - 1), fill_value=np.nan)

    x_data_clean = x_data[show]
    y_data_clean = y_data[show]

    # Handle extrapolation for all series at once (vectorized)
    if extrapolate_method == "outer":
        edges_processed = np.clip(x_edges, x_data_clean.min(), x_data_clean.max())
    elif extrapolate_method == "raise":
        if np.any(x_edges < x_data_clean.min()) or np.any(x_edges > x_data_clean.max()):
            msg = "x_edges must be within the range of x_data"
            raise ValueError(msg)
        edges_processed = x_edges.copy()
    else:  # nan method
        edges_processed = x_edges.copy()

    # Create a combined grid of all unique x points (data + all edges)
    all_unique_x = np.unique(np.concatenate([x_data_clean, edges_processed.ravel()]))

    # Interpolate y values at all unique x points once
    all_unique_y_result = np.interp(all_unique_x, x_data_clean, y_data_clean, left=np.nan, right=np.nan)
    # Ensure it's an array for type checker
    all_unique_y: npt.NDArray[np.float64] = np.asarray(all_unique_y_result, dtype=np.float64)

    # Compute cumulative integrals once using trapezoidal rule
    dx = np.diff(all_unique_x)
    y_avg = (all_unique_y[:-1] + all_unique_y[1:]) / 2
    segment_integrals = dx * y_avg
    # Replace NaN values with 0 to avoid breaking cumulative sum
    segment_integrals = np.nan_to_num(segment_integrals, nan=0.0)
    cumulative_integral = np.concatenate([[0], np.cumsum(segment_integrals)])

    # Vectorized computation for all series
    # Find indices of all edges in the combined grid
    edge_indices_result = np.searchsorted(all_unique_x, edges_processed)
    # Ensure it's a 2D array for type checker
    edge_indices: npt.NDArray[np.intp] = np.asarray(edge_indices_result, dtype=np.intp).reshape(edges_processed.shape)

    # Compute integral between consecutive edges for all series (vectorized)
    integral_values = cumulative_integral[edge_indices[:, 1:]] - cumulative_integral[edge_indices[:, :-1]]

    # Compute widths between consecutive edges for all series (vectorized)
    edge_widths = np.diff(edges_processed, axis=1)

    # Handle zero-width intervals (vectorized)
    zero_width_mask = edge_widths == 0
    result = np.zeros_like(edge_widths)

    # For non-zero width intervals, compute average = integral / width (vectorized)
    non_zero_mask = ~zero_width_mask
    result[non_zero_mask] = integral_values[non_zero_mask] / edge_widths[non_zero_mask]

    # For zero-width intervals, interpolate y-value directly (vectorized)
    if np.any(zero_width_mask):
        zero_width_positions = edges_processed[:, :-1][zero_width_mask]
        result[zero_width_mask] = np.interp(zero_width_positions, x_data_clean, y_data_clean)

    # Handle extrapolation when 'nan' method is used (vectorized)
    if extrapolate_method == "nan":
        # Set out-of-range bins to NaN
        bins_within_range = (x_edges[:, :-1] >= x_data_clean.min()) & (x_edges[:, 1:] <= x_data_clean.max())
        result[~bins_within_range] = np.nan

    return result


def partial_isin(*, bin_edges_in: npt.ArrayLike, bin_edges_out: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """
    Calculate the fraction of each input bin that overlaps with each output bin.

    This function computes a matrix where element (i, j) represents the fraction
    of input bin i that overlaps with output bin j. The computation uses
    vectorized operations to avoid loops.

    Parameters
    ----------
    bin_edges_in : array-like
        1D array of input bin edges in ascending order. For n_in bins, there
        should be n_in+1 edges.
    bin_edges_out : array-like
        1D array of output bin edges in ascending order. For n_out bins, there
        should be n_out+1 edges.

    Returns
    -------
    overlap_matrix : numpy.ndarray
        Dense matrix of shape (n_in, n_out) where n_in is the number of input
        bins and n_out is the number of output bins. Each element (i, j)
        represents the fraction of input bin i that overlaps with output bin j.
        Values range from 0 (no overlap) to 1 (complete overlap).

    Notes
    -----
    - Both input arrays must be sorted in ascending order
    - The function leverages the sorted nature of both inputs for efficiency
    - Uses vectorized operations to handle large bin arrays efficiently
    - All overlaps sum to 1.0 for each input bin when output bins fully cover input range

    Examples
    --------
    >>> import numpy as np
    >>> from gwtransport.utils import partial_isin
    >>> bin_edges_in = np.array([0, 10, 20, 30])
    >>> bin_edges_out = np.array([5, 15, 25])
    >>> partial_isin(
    ...     bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out
    ... )  # doctest: +NORMALIZE_WHITESPACE
    array([[0.5, 0. ],
           [0.5, 0.5],
           [0. , 0.5]])
    """
    # Convert inputs to numpy arrays
    bin_edges_in = np.asarray(bin_edges_in, dtype=float)
    bin_edges_out = np.asarray(bin_edges_out, dtype=float)

    # Validate inputs
    if bin_edges_in.ndim != 1 or bin_edges_out.ndim != 1:
        msg = "Both bin_edges_in and bin_edges_out must be 1D arrays"
        raise ValueError(msg)
    if len(bin_edges_in) < 2 or len(bin_edges_out) < 2:  # noqa: PLR2004
        msg = "Both edge arrays must have at least 2 elements"
        raise ValueError(msg)

    # Check ascending order, ignoring NaN values
    diffs_in = np.diff(bin_edges_in)
    valid_diffs_in = ~np.isnan(diffs_in)
    if np.any(valid_diffs_in) and not np.all(diffs_in[valid_diffs_in] > 0):
        msg = "bin_edges_in must be in ascending order"
        raise ValueError(msg)

    diffs_out = np.diff(bin_edges_out)
    valid_diffs_out = ~np.isnan(diffs_out)
    if np.any(valid_diffs_out) and not np.all(diffs_out[valid_diffs_out] > 0):
        msg = "bin_edges_out must be in ascending order"
        raise ValueError(msg)

    # Build matrix using fully vectorized approach
    # Create meshgrids for all possible input-output bin combinations
    in_left = bin_edges_in[:-1, None]  # Shape: (n_bins_in, 1)
    in_right = bin_edges_in[1:, None]  # Shape: (n_bins_in, 1)
    in_width = np.diff(bin_edges_in)[:, None]  # Shape: (n_bins_in, 1)

    out_left = bin_edges_out[None, :-1]  # Shape: (1, n_bins_out)
    out_right = bin_edges_out[None, 1:]  # Shape: (1, n_bins_out)

    # Calculate overlaps for all combinations using broadcasting
    overlap_left = np.maximum(in_left, out_left)  # Shape: (n_bins_in, n_bins_out)
    overlap_right = np.minimum(in_right, out_right)  # Shape: (n_bins_in, n_bins_out)

    # Calculate overlap widths (zero where no overlap)
    overlap_widths = np.maximum(0, overlap_right - overlap_left)

    # Calculate fractions (NaN widths will result in NaN fractions)
    return overlap_widths / in_width


def time_bin_overlap(*, tedges: npt.ArrayLike, bin_tedges: list[tuple]) -> npt.NDArray[np.floating]:
    """
    Calculate the fraction of each time bin that overlaps with each time range.

    This function computes an array where element (i, j) represents the fraction
    of time bin j that overlaps with time range i. The computation uses
    vectorized operations to avoid loops.

    Parameters
    ----------
    tedges : array-like
        1D array of time bin edges in ascending order. For n bins, there
        should be n+1 edges.
    bin_tedges : list of tuple
        List of tuples where each tuple contains ``(start_time, end_time)``
        defining a time range.

    Returns
    -------
    overlap_array : numpy.ndarray
        Array of shape (len(bin_tedges), n_bins) where n_bins is the number of
        time bins. Each element (i, j) represents the fraction of time bin j
        that overlaps with time range i. Values range from 0 (no overlap) to
        1 (complete overlap).

    Notes
    -----
    - tedges must be sorted in ascending order
    - Uses vectorized operations to handle large arrays efficiently
    - Time ranges in bin_tedges can be in any order and can overlap

    Examples
    --------
    >>> import numpy as np
    >>> from gwtransport.utils import time_bin_overlap
    >>> tedges = np.array([0, 10, 20, 30])
    >>> bin_tedges = [(5, 15), (25, 35)]
    >>> time_bin_overlap(
    ...     tedges=tedges, bin_tedges=bin_tedges
    ... )  # doctest: +NORMALIZE_WHITESPACE
    array([[0.5, 0.5, 0. ],
           [0. , 0. , 0.5]])
    """
    # Convert inputs to numpy arrays
    tedges = np.asarray(tedges)
    bin_tedges_array = np.asarray(bin_tedges)

    # Validate inputs
    if tedges.ndim != 1:
        msg = "tedges must be a 1D array"
        raise ValueError(msg)
    if len(tedges) < 2:  # noqa: PLR2004
        msg = "tedges must have at least 2 elements"
        raise ValueError(msg)
    if bin_tedges_array.size == 0:
        msg = "bin_tedges must be non-empty"
        raise ValueError(msg)

    # Calculate overlaps for all combinations using broadcasting
    overlap_left = np.maximum(bin_tedges_array[:, [0]], tedges[None, :-1])
    overlap_right = np.minimum(bin_tedges_array[:, [1]], tedges[None, 1:])
    overlap_widths = np.maximum(0, overlap_right - overlap_left)

    # Calculate fractions (handle division by zero for zero-width bins)
    bin_width_bc = np.diff(tedges)[None, :]  # Shape: (1, n_bins)

    return np.divide(
        overlap_widths, bin_width_bc, out=np.zeros_like(overlap_widths, dtype=float), where=bin_width_bc != 0.0
    )


def generate_failed_coverage_badge() -> None:
    """Generate a badge indicating failed coverage."""
    from genbadge import Badge  # type: ignore # noqa: PLC0415

    b = Badge(left_txt="coverage", right_txt="failed", color="red")
    b.write_to("coverage_failed.svg", use_shields=False)


def combine_bin_series(
    *,
    a: npt.ArrayLike,
    a_edges: npt.ArrayLike,
    b: npt.ArrayLike,
    b_edges: npt.ArrayLike,
    extrapolation: str | float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine two binned series onto a common set of unique edges.

    This function takes two binned series (a and b) with their respective bin edges
    and creates new series (c and d) that are defined on a combined set of unique
    edges from both input edge arrays.

    Parameters
    ----------
    a : array-like
        Values for the first binned series.
    a_edges : array-like
        Bin edges for the first series. Must have len(a) + 1 elements.
    b : array-like
        Values for the second binned series.
    b_edges : array-like
        Bin edges for the second series. Must have len(b) + 1 elements.
    extrapolation : str or float, optional
        Method for handling combined bins that fall outside the original series ranges.
        - 'nearest': Use the nearest original bin value
        - float value (e.g., np.nan, 0.0): Fill with the specified value (default: 0.0)

    Returns
    -------
    c : numpy.ndarray
        Values from series a mapped to the combined edge structure.
    c_edges : numpy.ndarray
        Combined unique edges from a_edges and b_edges.
    d : numpy.ndarray
        Values from series b mapped to the combined edge structure.
    d_edges : numpy.ndarray
        Combined unique edges from a_edges and b_edges (same as c_edges).

    Notes
    -----
    The combined edges are created by taking the union of all unique values
    from both a_edges and b_edges, sorted in ascending order. The values
    are then broadcasted/repeated for each combined bin that falls within
    the original bin's range.
    """
    # Convert inputs to numpy arrays
    a = np.asarray(a, dtype=float)
    a_edges = np.asarray(a_edges, dtype=float)
    b = np.asarray(b, dtype=float)
    b_edges = np.asarray(b_edges, dtype=float)

    # Validate inputs
    if len(a_edges) != len(a) + 1:
        msg = "a_edges must have len(a) + 1 elements"
        raise ValueError(msg)
    if len(b_edges) != len(b) + 1:
        msg = "b_edges must have len(b) + 1 elements"
        raise ValueError(msg)

    # Create combined unique edges
    combined_edges = np.unique(np.concatenate([a_edges, b_edges]))

    # Initialize output arrays
    c = np.zeros(len(combined_edges) - 1)
    d = np.zeros(len(combined_edges) - 1)

    # Vectorized mapping using searchsorted - find which original bin each combined bin belongs to
    # For series a: find which original bin each combined bin center falls into
    combined_bin_centers = (combined_edges[:-1] + combined_edges[1:]) / 2
    a_bin_assignment_result = np.searchsorted(a_edges, combined_bin_centers, side="right") - 1
    # Ensure it's an array for type checker
    a_bin_assignment: npt.NDArray[np.intp] = np.asarray(a_bin_assignment_result, dtype=np.intp)
    a_bin_assignment = np.clip(a_bin_assignment, 0, len(a) - 1)

    # Handle extrapolation for series a
    if extrapolation == "nearest":
        # Assign all values using nearest neighbor (already clipped)
        c[:] = a[a_bin_assignment]
    else:
        # Only assign values where the combined bin is completely within the original bin
        a_valid_mask = (combined_edges[:-1] >= a_edges[a_bin_assignment]) & (
            combined_edges[1:] <= a_edges[a_bin_assignment + 1]
        )
        c[a_valid_mask] = a[a_bin_assignment[a_valid_mask]]
        # Fill out-of-range bins with extrapolation value
        c[~a_valid_mask] = extrapolation

    # Handle extrapolation for series b
    b_bin_assignment_result = np.searchsorted(b_edges, combined_bin_centers, side="right") - 1
    # Ensure it's an array for type checker
    b_bin_assignment: npt.NDArray[np.intp] = np.asarray(b_bin_assignment_result, dtype=np.intp)
    b_bin_assignment = np.clip(b_bin_assignment, 0, len(b) - 1)

    if extrapolation == "nearest":
        # Assign all values using nearest neighbor (already clipped)
        d[:] = b[b_bin_assignment]
    else:
        # Only assign values where the combined bin is completely within the original bin
        b_valid_mask = (combined_edges[:-1] >= b_edges[b_bin_assignment]) & (
            combined_edges[1:] <= b_edges[b_bin_assignment + 1]
        )
        d[b_valid_mask] = b[b_bin_assignment[b_valid_mask]]
        # Fill out-of-range bins with extrapolation value
        d[~b_valid_mask] = extrapolation

    # Return the combined series
    c_edges = combined_edges
    d_edges = combined_edges.copy()

    return c, c_edges, d, d_edges


def compute_time_edges(
    *,
    tedges: pd.DatetimeIndex | None,
    tstart: pd.DatetimeIndex | None,
    tend: pd.DatetimeIndex | None,
    number_of_bins: int,
) -> pd.DatetimeIndex:
    """
    Compute time edges for binning data based on provided time parameters.

    This function creates a DatetimeIndex of time bin edges from one of three possible
    input formats: explicit edges, start times, or end times. The resulting edges
    define the boundaries of time intervals for data binning.

    Define either explicit time edges, or start and end times for each bin and leave the others at None.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex or None
        Explicit time edges for the bins. If provided, must have one more element
        than the number of bins (n_bins + 1). Takes precedence over tstart and tend.
    tstart : pandas.DatetimeIndex or None
        Start times for each bin. Must have the same number of elements as the
        number of bins. Used when tedges is None.
    tend : pandas.DatetimeIndex or None
        End times for each bin. Must have the same number of elements as the
        number of bins. Used when both tedges and tstart are None.
    number_of_bins : int
        The expected number of time bins. Used for validation against the provided
        time parameters.

    Returns
    -------
    pandas.DatetimeIndex
        Time edges defining the boundaries of the time bins. Has one more element
        than number_of_bins.

    Raises
    ------
    ValueError
        If tedges has incorrect length (not number_of_bins + 1).
        If tstart has incorrect length (not equal to number_of_bins).
        If tend has incorrect length (not equal to number_of_bins).
        If none of tedges, tstart, or tend are provided.

    Notes
    -----
    - When using tstart, the function assumes uniform spacing and extrapolates
      the final edge based on the spacing between the last two start times.
    - When using tend, the function assumes uniform spacing and extrapolates
      the first edge based on the spacing between the first two end times.
    - All input time data is converted to pandas.DatetimeIndex for consistency.
    """
    if tedges is not None:
        if number_of_bins != len(tedges) - 1:
            msg = "tedges must have one more element than flow"
            raise ValueError(msg)
        return pd.DatetimeIndex(tedges)

    if tstart is not None:
        # Assume the index refers to the time at the start of the measurement interval
        tstart = pd.DatetimeIndex(tstart)
        if number_of_bins != len(tstart):
            msg = "tstart must have the same number of elements as flow"
            raise ValueError(msg)

        return pd.DatetimeIndex(tstart.append(tstart[[-1]] + (tstart[-1] - tstart[-2])))

    if tend is not None:
        # Assume the index refers to the time at the end of the measurement interval
        tend = pd.DatetimeIndex(tend)
        if number_of_bins != len(tend):
            msg = "tend must have the same number of elements as flow"
            raise ValueError(msg)

        return pd.DatetimeIndex((tend[[0]] - (tend[1] - tend[0])).append(tend))

    msg = "Either provide tedges, tstart, and tend"
    raise ValueError(msg)


def get_soil_temperature(*, station_number: int = 260, interpolate_missing_values: bool = True) -> pd.DataFrame:
    """
    Download soil temperature data from the KNMI and return it as a pandas DataFrame.

    The data is available for the following KNMI weather stations:
    - 260: De Bilt, the Netherlands (vanaf 1981)
    - 273: Marknesse, the Netherlands (vanaf 1989)
    - 286: Nieuw Beerta, the Netherlands (vanaf 1990)
    - 323: Wilhelminadorp, the Netherlands (vanaf 1989)

    TB1	 = grondtemperatuur op   5 cm diepte (graden Celsius) tijdens de waarneming
    TB2	 = grondtemperatuur op  10 cm diepte (graden Celsius) tijdens de waarneming
    TB3	 = grondtemperatuur op  20 cm diepte (graden Celsius) tijdens de waarneming
    TB4	 = grondtemperatuur op  50 cm diepte (graden Celsius) tijdens de waarneming
    TB5	 = grondtemperatuur op 100 cm diepte (graden Celsius) tijdens de waarneming
    TNB2 = minimum grondtemperatuur op 10 cm diepte in de afgelopen 6 uur (graden Celsius)
    TNB1 = minimum grondtemperatuur op  5 cm diepte in de afgelopen 6 uur (graden Celsius)
    TXB1 = maximum grondtemperatuur op  5 cm diepte in de afgelopen 6 uur (graden Celsius)
    TXB2 = maximum grondtemperatuur op 10 cm diepte in de afgelopen 6 uur (graden Celsius)

    Parameters
    ----------
    station_number : int, {260, 273, 286, 323}
        The KNMI station number for which to download soil temperature data.
        Default is 260 (De Bilt).
    interpolate_missing_values : bool, optional
        If True, missing values are interpolated and recent NaN values are extrapolated with the previous value.
        If False, missing values remain as NaN. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing soil temperature data in Celsius with a DatetimeIndex.
        Columns include TB1, TB2, TB3, TB4, TB5, TNB1, TNB2, TXB1, TXB2.

    Notes
    -----
    - KNMI: Royal Netherlands Meteorological Institute
    - The timeseries may contain NaN values for missing data.
    """
    # File-based daily cache
    cache_dir.mkdir(exist_ok=True)

    today = date.today().isoformat()  # noqa: DTZ011
    cache_path = cache_dir / f"soil_temp_{station_number}_{interpolate_missing_values}_{today}.pkl"

    # Check if cached file exists and is from today
    if cache_path.exists():
        return pd.read_pickle(cache_path)  # noqa: S301

    # Clean up old cache files to prevent disk bloat
    for old_file in cache_dir.glob(f"soil_temp_{station_number}_{interpolate_missing_values}_*.pkl"):
        old_file.unlink(missing_ok=True)

    url = f"https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/bodemtemps/bodemtemps_{station_number}.zip"

    dtypes = {
        "YYYYMMDD": "int32",
        "HH": "int8",
        "  TB1": "float32",
        "  TB3": "float32",
        "  TB2": "float32",
        "  TB4": "float32",
        "  TB5": "float32",
        " TNB1": "float32",
        " TNB2": "float32",
        " TXB1": "float32",
        " TXB2": "float32",
    }

    # Download the ZIP file
    with requests.get(url, params={"download": "zip"}, timeout=10) as response:
        response.raise_for_status()

    df = pd.read_csv(  # type: ignore[call-overload]
        io.BytesIO(response.content),
        compression="zip",
        dtype=dtypes,
        usecols=list(dtypes.keys()),
        skiprows=16,
        sep=",",
        na_values=["     "],
        engine="c",
        parse_dates=False,
    )

    df.index = pd.to_datetime(df["YYYYMMDD"].values, format=r"%Y%m%d").tz_localize("UTC") + pd.to_timedelta(
        df["HH"].values, unit="h"
    )

    df.drop(columns=["YYYYMMDD", "HH"], inplace=True)
    df.columns = df.columns.str.strip()
    df /= 10.0

    if interpolate_missing_values:
        # Fill NaN values with interpolate linearly and then forward fill
        df.interpolate(method="linear", inplace=True)
        df.ffill(inplace=True)

    # Save to cache for future use
    df.to_pickle(cache_path)
    return df


def solve_underdetermined_system(
    *,
    coefficient_matrix: npt.ArrayLike,
    rhs_vector: npt.ArrayLike,
    nullspace_objective: str | Callable[[np.ndarray, np.ndarray, np.ndarray], float] = "squared_differences",
    optimization_method: str = "BFGS",
) -> npt.NDArray[np.floating]:
    """
    Solve an underdetermined linear system with nullspace regularization.

    For an underdetermined system Ax = b where A has more columns than rows,
    multiple solutions exist. This function computes a least-squares solution
    and then selects a specific solution from the nullspace based on a
    regularization objective.

    Parameters
    ----------
    coefficient_matrix : array-like
        Coefficient matrix of shape (m, n) where m < n (underdetermined).
        May contain NaN values in some rows, which will be excluded from the system.
    rhs_vector : array-like
        Right-hand side vector of length m. May contain NaN values corresponding
        to NaN rows in coefficient_matrix, which will be excluded from the system.
    nullspace_objective : str or callable, optional
        Objective function to minimize in the nullspace. Options:

        * "squared_differences" : Minimize sum of squared differences between
          adjacent elements: ``sum((x[i+1] - x[i])**2)``
        * "summed_differences" : Minimize sum of absolute differences between
          adjacent elements: ``sum(|x[i+1] - x[i]|)``
        * callable : Custom objective function with signature
          ``objective(coeffs, x_ls, nullspace_basis)`` where:

          - coeffs : optimization variables (nullspace coefficients)
          - x_ls : least-squares solution
          - nullspace_basis : nullspace basis matrix

        Default is "squared_differences".
    optimization_method : str, optional
        Optimization method passed to scipy.optimize.minimize.
        Default is "BFGS".

    Returns
    -------
    numpy.ndarray
        Solution vector that minimizes the specified nullspace objective.
        Has length n (number of columns in coefficient_matrix).

    Raises
    ------
    ValueError
        If optimization fails, if coefficient_matrix and rhs_vector have incompatible shapes,
        or if an unknown nullspace objective is specified.

    Notes
    -----
    The algorithm follows these steps:

    1. Remove rows with NaN values from both coefficient_matrix and rhs_vector
    2. Compute least-squares solution: x_ls = pinv(valid_matrix) @ valid_rhs
    3. Compute nullspace basis: N = null_space(valid_matrix)
    4. Find nullspace coefficients: coeffs = argmin objective(x_ls + N @ coeffs)
    5. Return final solution: x = x_ls + N @ coeffs

    For the built-in objectives:

    * "squared_differences" provides smooth solutions, minimizing rapid changes
    * "summed_differences" provides sparse solutions, promoting piecewise constant behavior

    Examples
    --------
    Basic usage with default squared differences objective:

    >>> import numpy as np
    >>> from gwtransport.utils import solve_underdetermined_system
    >>>
    >>> # Create underdetermined system (2 equations, 4 unknowns)
    >>> matrix = np.array([[1, 2, 1, 0], [0, 1, 2, 1]])
    >>> rhs = np.array([3, 4])
    >>>
    >>> # Solve with squared differences regularization
    >>> x = solve_underdetermined_system(coefficient_matrix=matrix, rhs_vector=rhs)
    >>> print(f"Solution: {x}")  # doctest: +SKIP
    >>> print(f"Residual: {np.linalg.norm(matrix @ x - rhs):.2e}")  # doctest: +SKIP

    With summed differences objective:

    >>> x_sparse = solve_underdetermined_system(  # doctest: +SKIP
    ...     coefficient_matrix=matrix,
    ...     rhs_vector=rhs,
    ...     nullspace_objective="summed_differences",
    ... )

    With custom objective function:

    >>> def custom_objective(coeffs, x_ls, nullspace_basis):
    ...     x = x_ls + nullspace_basis @ coeffs
    ...     return np.sum(x**2)  # Minimize L2 norm
    >>>
    >>> x_custom = solve_underdetermined_system(  # doctest: +SKIP
    ...     coefficient_matrix=matrix,
    ...     rhs_vector=rhs,
    ...     nullspace_objective=custom_objective,
    ... )

    Handling NaN values:

    >>> # System with missing data
    >>> matrix_nan = np.array([
    ...     [1, 2, 1, 0],
    ...     [np.nan, np.nan, np.nan, np.nan],
    ...     [0, 1, 2, 1],
    ... ])
    >>> rhs_nan = np.array([3, np.nan, 4])
    >>>
    >>> x_nan = solve_underdetermined_system(
    ...     coefficient_matrix=matrix_nan, rhs_vector=rhs_nan
    ... )  # doctest: +SKIP
    """
    matrix = np.asarray(coefficient_matrix)
    rhs = np.asarray(rhs_vector)

    if matrix.shape[0] != len(rhs):
        msg = f"coefficient_matrix has {matrix.shape[0]} rows but rhs_vector has {len(rhs)} elements"
        raise ValueError(msg)

    # Identify valid rows (no NaN values in either matrix or rhs)
    valid_rows = ~np.isnan(matrix).any(axis=1) & ~np.isnan(rhs)

    if not np.any(valid_rows):
        msg = "No valid rows found (all contain NaN values)"
        raise ValueError(msg)

    valid_matrix = matrix[valid_rows]
    valid_rhs = rhs[valid_rows]

    # Compute least-squares solution
    x_ls, *_ = np.linalg.lstsq(valid_matrix, valid_rhs, rcond=None)

    # Compute nullspace
    nullspace_basis = null_space(valid_matrix, rcond=None)
    nullrank = nullspace_basis.shape[1]

    if nullrank == 0:
        # System is determined, return least-squares solution
        return x_ls

    # Optimize in nullspace
    coeffs = _optimize_nullspace_coefficients(
        x_ls=x_ls,
        nullspace_basis=nullspace_basis,
        nullspace_objective=nullspace_objective,
        optimization_method=optimization_method,
    )

    return x_ls + nullspace_basis @ coeffs


def _optimize_nullspace_coefficients(
    *,
    x_ls: np.ndarray,
    nullspace_basis: np.ndarray,
    nullspace_objective: str | Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    optimization_method: str,
) -> npt.NDArray[np.floating]:
    """Optimize coefficients in the nullspace to minimize the objective."""
    nullrank = nullspace_basis.shape[1]
    objective_func = _get_nullspace_objective_function(nullspace_objective=nullspace_objective)
    coeffs_0 = np.zeros(nullrank)

    # For stability, always start with squared differences if using summed differences
    if nullspace_objective == "summed_differences":
        res_init = minimize(
            _squared_differences_objective,
            x0=coeffs_0,
            args=(x_ls, nullspace_basis),
            method=optimization_method,  # type: ignore[arg-type]
        )
        if not res_init.success:
            msg = f"Initial optimization failed: {res_init.message}"
            raise ValueError(msg)
        coeffs_0 = res_init.x

    # Final optimization with target objective
    res = minimize(
        objective_func,
        x0=coeffs_0,
        args=(x_ls, nullspace_basis),
        method=optimization_method,  # type: ignore[arg-type]
    )

    if not res.success:
        msg = f"Optimization failed: {res.message}"
        raise ValueError(msg)

    return res.x


def _squared_differences_objective(coeffs: np.ndarray, x_ls: np.ndarray, nullspace_basis: np.ndarray) -> float:
    """Minimize sum of squared differences between adjacent elements."""
    x = x_ls + nullspace_basis @ coeffs
    return np.sum(np.square(x[1:] - x[:-1]))


def _summed_differences_objective(coeffs: np.ndarray, x_ls: np.ndarray, nullspace_basis: np.ndarray) -> float:
    """Minimize sum of absolute differences between adjacent elements."""
    x = x_ls + nullspace_basis @ coeffs
    return np.sum(np.abs(x[1:] - x[:-1]))


def _get_nullspace_objective_function(
    *,
    nullspace_objective: str | Callable[[np.ndarray, np.ndarray, np.ndarray], float],
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """Get the objective function for nullspace optimization."""
    if nullspace_objective == "squared_differences":
        return _squared_differences_objective
    if nullspace_objective == "summed_differences":
        return _summed_differences_objective
    if callable(nullspace_objective):
        return nullspace_objective  # type: ignore[return-value]
    msg = f"Unknown nullspace objective: {nullspace_objective}"
    raise ValueError(msg)
