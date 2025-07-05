"""General utilities for the 1D groundwater transport model."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
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


def linear_average(  # noqa: C901
    x_data: Sequence[float] | npt.NDArray[np.float64],
    y_data: Sequence[float] | npt.NDArray[np.float64],
    x_edges: Sequence[float] | npt.NDArray[np.float64],
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
    >>> x_data = [0, 1, 2, 3]
    >>> y_data = [0, 1, 1, 0]
    >>> x_edges = [0, 1.5, 3]
    >>> linear_average(x_data, y_data, x_edges)
    array([[0.667, 0.667]])

    >>> x_edges_2d = [[0, 1.5, 3], [0.5, 2, 3]]
    >>> linear_average(x_data, y_data, x_edges_2d)
    array([[0.667, 0.667], [0.75, 0.5]])
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
    all_unique_y = np.interp(all_unique_x, x_data_clean, y_data_clean, left=np.nan, right=np.nan)

    # Compute cumulative integrals once using trapezoidal rule
    dx = np.diff(all_unique_x)
    y_avg = (all_unique_y[:-1] + all_unique_y[1:]) / 2
    segment_integrals = dx * y_avg
    # Replace NaN values with 0 to avoid breaking cumulative sum
    segment_integrals = np.nan_to_num(segment_integrals, nan=0.0)
    cumulative_integral = np.concatenate([[0], np.cumsum(segment_integrals)])

    # Vectorized computation for all series
    # Find indices of all edges in the combined grid
    edge_indices = np.searchsorted(all_unique_x, edges_processed)

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


def partial_isin(bin_edges_in, bin_edges_out):
    """
    Calculate the fraction of each input bin that overlaps with each output bin.

    This function computes a matrix where element (i, j) represents the fraction
    of input bin i that overlaps with output bin j. The computation uses
    vectorized operations to avoid loops.

    Parameters
    ----------
    bin_edges_in : array_like
        1D array of input bin edges in ascending order. For n_in bins, there
        should be n_in+1 edges.
    bin_edges_out : array_like
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
    >>> bin_edges_in = np.array([0, 10, 20, 30])
    >>> bin_edges_out = np.array([5, 15, 25])
    >>> partial_isin(bin_edges_in, bin_edges_out)
    array([[0.5, 0.0],
           [0.5, 0.5],
           [0.0, 0.5]])
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


def generate_failed_coverage_badge():
    """Generate a badge indicating failed coverage."""
    from genbadge import Badge  # type: ignore # noqa: PLC0415

    b = Badge(left_txt="coverage", right_txt="failed", color="red")
    b.write_to("coverage_failed.svg", use_shields=False)


def combine_bin_series(a, a_edges, b, b_edges, extrapolation=0.0):
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
    a_bin_assignment = np.searchsorted(a_edges, combined_bin_centers, side="right") - 1
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
    b_bin_assignment = np.searchsorted(b_edges, combined_bin_centers, side="right") - 1
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


def compute_time_edges(tedges, tstart, tend, number_of_bins):
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
        tedges = pd.DatetimeIndex(tedges)
        if number_of_bins != len(tedges) - 1:
            msg = "tedges must have one more element than flow"
            raise ValueError(msg)

    elif tstart is not None:
        # Assume the index refers to the time at the start of the measurement interval
        tstart = pd.DatetimeIndex(tstart)
        if number_of_bins != len(tstart):
            msg = "tstart must have the same number of elements as flow"
            raise ValueError(msg)

        tedges = tstart.append(tstart[[-1]] + (tstart[-1] - tstart[-2]))

    elif tend is not None:
        # Assume the index refers to the time at the end of the measurement interval
        tend = pd.DatetimeIndex(tend)
        if number_of_bins != len(tend):
            msg = "tend must have the same number of elements as flow"
            raise ValueError(msg)

        tedges = (tend[[0]] - (tend[1] - tend[0])).append(tend)

    else:
        msg = "Either provide tedges, tstart, and tend"
        raise ValueError(msg)
    return tedges
