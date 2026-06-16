"""
General Utilities for 1D Groundwater Transport Modeling.

This module provides general-purpose utility functions for time series manipulation,
interpolation, numerical operations, and data processing used throughout the gwtransport
package. Functions include linear interpolation/averaging, bin overlap calculations,
underdetermined system solvers, and external data retrieval.

Available functions:

- :func:`step_plot_coords` - Compute step-plot coordinates from bin edges and
  bin-averaged values. Returns paired x/y arrays for plotting piecewise-constant
  functions with ``ax.plot(x, y)``.

- ``_make_strictly_monotone`` (private) - Bump consecutive duplicates in a non-decreasing
  array by ``k * ulp(max)`` so it becomes strictly monotone. Used before V → t inversions to
  prevent ``np.interp`` from silently picking one limit at plateau levels.

- :func:`cumulative_flow_volume` - Cumulative infiltrated/extracted volume from per-bin flow
  rates and bin widths, prepended with a leading zero. Optionally bumped to strict
  monotonicity for V → t inversions.

- :func:`linear_interpolate` - Linear interpolation using numpy's optimized interp function.
  Automatically handles unsorted data with configurable extrapolation (None for clamping,
  float for constant values). Handles multi-dimensional query arrays.

- :func:`linear_average` - Compute average values of piecewise linear time series between
  specified x-edges. Supports 1D or 2D edge arrays for batch processing. Handles NaN values
  and offers multiple extrapolation methods ('nan', 'outer', 'raise').

- :func:`partial_isin` - Calculate fraction of each input bin overlapping with each output bin.
  Returns dense matrix where element (i,j) represents overlap fraction. Uses vectorized
  operations for efficiency.

- :func:`time_bin_overlap` - Calculate fraction of time bins overlapping with specified time
  ranges. Similar to partial_isin but for time-based bin overlaps with list of (start, end)
  tuples.

- :func:`simplify_bins` - Simplify a piecewise-constant time series by merging adjacent bins
  whose values are within a tolerance. Uses volume-weighted (flow x width) averaging when
  flow is provided, otherwise width-weighted. Direction-independent via recursive splitting.

- :func:`combine_bin_series` - Combine two binned series onto common set of unique edges. Maps
  values from original bins to new combined structure with configurable extrapolation ('nearest'
  or float value).

- :func:`compute_time_edges` - Compute DatetimeIndex of time bin edges from explicit edges,
  start times, or end times. Validates consistency with expected number of bins and handles
  uniform spacing extrapolation.

- :func:`solve_tikhonov` - Solve linear system with Tikhonov regularization toward a target.
  Well-determined modes follow the data; poorly-determined modes are pulled toward the target.

- :func:`compute_reverse_target` - Build the regularization target for the inverse problem by
  transposing and row-normalizing the forward coefficient matrix. Consumed by
  :func:`solve_tikhonov` and :func:`solve_inverse_transport`.

- :func:`solve_inverse_transport` - Solve the inverse transport problem (deconvolution) via
  Tikhonov regularization. Shared by advection, diffusion, and diffusion_fast
  ``extraction_to_infiltration`` functions.

- :func:`solve_inverse_transport_banded` - Memory-light banded equivalent of
  :func:`solve_inverse_transport` for a forward operator stored in banded layout. Assembles the
  Tikhonov normal equations directly in banded form and solves them via banded Cholesky.

- :func:`solve_underdetermined_system` - Solve underdetermined linear system (Ax = b, m < n)
  with nullspace regularization. Handles NaN values by row exclusion. Supports built-in
  objectives ('squared_differences', 'summed_differences') or custom callable objectives.
  Used by :mod:`gwtransport.deposition`.

- :func:`get_soil_temperature` - Download soil temperature data from KNMI weather stations with
  automatic caching. Supports stations 260 (De Bilt), 273 (Marknesse), 286 (Nieuw Beerta),
  323 (Wilhelminadorp). Returns DataFrame with columns TB1-TB5, TNB1-TNB2, TXB1-TXB2 at various
  depths. Daily cache prevents redundant downloads.

- ``_generate_failed_coverage_badge`` (private) - Generate SVG badge indicating failed coverage
  using genbadge library. Used in CI/CD workflows.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from scipy.linalg import cho_solve_banded, cholesky_banded, null_space
from scipy.optimize import minimize

cache_dir = Path(__file__).parent.parent.parent / "cache"


def step_plot_coords(edges: npt.ArrayLike, values: npt.ArrayLike) -> tuple[npt.NDArray, npt.NDArray]:
    """Compute step-plot coordinates from bin edges and bin-averaged values.

    Converts bin edges (n+1) and bin values (n) into paired x/y arrays
    suitable for plotting piecewise-constant (step) functions with
    ``ax.plot(x, y)``.

    Parameters
    ----------
    edges : array-like
        Bin edges (n+1 elements for n bins). Can be numeric, datetime, or
        any type accepted by :func:`numpy.repeat`.
    values : array-like
        Bin-averaged values (n elements), one per bin.

    Returns
    -------
    x : ndarray
        Step x-coordinates (2n elements). Same dtype as *edges*.
    y : ndarray
        Step y-coordinates (2n elements). Same dtype as *values*.

    Examples
    --------
    >>> import numpy as np
    >>> edges = np.array([0.0, 1.0, 3.0, 6.0])
    >>> values = np.array([2.0, 5.0, 1.0])
    >>> x, y = step_plot_coords(edges, values)
    >>> x
    array([0., 1., 1., 3., 3., 6.])
    >>> y
    array([2., 2., 5., 5., 1., 1.])
    """
    x = np.repeat(edges, 2)[1:-1]
    y = np.repeat(values, 2)
    return x, y


_DUP_BUMP_ULPS = 16  # safety factor in ulps; see _make_strictly_monotone docstring


def _make_strictly_monotone(arr: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Bump consecutive duplicates so a non-decreasing array becomes strictly monotone.

    Returns the input unchanged if no consecutive duplicates are present. Otherwise returns a
    new array with each duplicate bumped up by ``k * step``, where ``k`` is its 1-based
    position within the consecutive duplicate run and ``step`` is ``16 * ulp(max(arr))``
    capped per run so the largest bump stays strictly below the next genuine value above the
    plateau (``step = min(16 * ulp(max(arr)), gap / (run_len + 1))``). The cap prevents a long
    run from overshooting a closely-spaced successor; a gap narrower than the run length in
    ulps is unrepresentable and cannot be separated.

    The factor of 16 is a safety margin against IEEE 754 rounding noise in ``np.interp``'s
    linear-interpolation arithmetic, which differs subtly between Linux x86_64 (with FMA)
    and ARM macOS. A 1-ulp gap, while strictly monotone, can place a downstream query value
    on the wrong side of a bracket boundary if the intermediate arithmetic rounds 1 ulp away
    from the exact value. 16 ulps ensures the bracket selection is unambiguous on every
    platform we support. The perturbation is relative to the array scale:
    ``bump ≈ 16 * ulp(max(arr)) ≈ 3.5e-15 * max(arr)``, i.e. about 15 significant digits
    below the data scale and well below physical relevance. The absolute size therefore grows
    with the cumulative-volume magnitude (e.g. ``~1e-13`` only for ``max(arr) ~ 30``).

    Parameters
    ----------
    arr : array-like
        1D non-decreasing array (e.g., a cumulative volume sequence ``flow_cum`` that contains
        plateaus from ``Q = 0`` bins).

    Returns
    -------
    ndarray
        Strictly monotone array of the same length.

    Notes
    -----
    Use this before passing ``arr`` as ``x_ref`` to a ``V → t`` inversion via
    :func:`linear_interpolate` or :func:`numpy.interp`. Plateaus in ``arr`` make ``arr⁻¹``
    multi-valued, and ``np.interp`` would silently pick one of the two limits, biasing
    integrals over output bins that span the kink.
    """
    arr = np.asarray(arr, dtype=float)
    diffs = np.diff(arr)
    if not np.any(diffs == 0):
        return arr
    ulp_max = np.nextafter(arr.max(), np.inf) - arr.max()
    n = len(arr)
    idx = np.arange(n)
    is_dup = np.concatenate(([False], diffs == 0))
    # 1-based position of each duplicate within its consecutive run.
    last_nondup = np.maximum.accumulate(np.where(is_dup, -1, idx))
    cumcount = np.where(is_dup, idx - last_nondup, 0)

    # Per-run headroom: each bumped value must stay strictly below the next genuine
    # (non-duplicate) value above the plateau, otherwise a long run can overshoot a
    # closely-spaced next value and break monotonicity. ``next_nondup_idx`` is the first
    # non-duplicate index after each position (``n`` when the run reaches the array end, where
    # there is no successor and hence no overshoot risk). The gap to that successor caps the
    # bump step so the last (largest) bump in a run of length L is at most ``L/(L+1)`` of the
    # gap. A gap narrower than the run length in ulps is unrepresentable and cannot be split.
    next_nondup_idx = np.minimum.accumulate(np.where(is_dup, n, idx)[::-1])[::-1]
    has_successor = next_nondup_idx < n
    gap_to_next = arr[np.clip(next_nondup_idx, 0, n - 1)] - arr[idx]
    run_len = next_nondup_idx - last_nondup - 1
    full_step = _DUP_BUMP_ULPS * ulp_max
    with np.errstate(invalid="ignore", divide="ignore"):
        capped_step = np.where(has_successor, np.minimum(full_step, gap_to_next / (run_len + 1.0)), full_step)
    bump = np.where(is_dup, cumcount * capped_step, 0.0)
    return arr + bump


def cumulative_flow_volume(
    flow: npt.ArrayLike, dt_days: npt.ArrayLike, *, strictly_monotone: bool = False
) -> npt.NDArray[np.floating]:
    """Cumulative infiltrated/extracted volume from per-bin flow rates.

    Multiplies each per-bin flow rate by its bin width and accumulates, with a
    leading zero prepended so the result has one entry per bin edge (n+1 values
    for n bins). The result is the cumulative volume ``V`` at each time edge.

    Parameters
    ----------
    flow : array-like
        Flow rate per bin (m³/day), length n.
    dt_days : array-like
        Bin widths in days, length n (e.g. ``numpy.diff`` of edge days).
    strictly_monotone : bool, optional
        When ``True``, bump consecutive duplicates (plateaus from ``Q = 0``
        bins) via ``_make_strictly_monotone`` so the cumulative volume is
        strictly increasing. Required before a V → t inversion; leave ``False``
        when the plateaus must be preserved. Default is ``False``.

    Returns
    -------
    ndarray
        Cumulative volume at each edge (length ``len(flow) + 1``), starting at
        zero.

    See Also
    --------
    ``_make_strictly_monotone`` : Bump duplicates before V → t inversion.
    """
    flow_cum = np.concatenate(([0.0], np.cumsum(np.asarray(flow) * np.asarray(dt_days))))
    return _make_strictly_monotone(flow_cum) if strictly_monotone else flow_cum


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
    ndarray
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
    """
    x_ref = np.asarray(x_ref)
    y_ref = np.asarray(y_ref)
    x_query = np.asarray(x_query)

    sort_idx = np.argsort(x_ref)
    x_ref_sorted = x_ref[sort_idx]
    y_ref_sorted = y_ref[sort_idx]

    return np.interp(x_query, x_ref_sorted, y_ref_sorted, left=left, right=right)


def linear_average(
    *,
    x_data: npt.ArrayLike,
    y_data: npt.ArrayLike,
    x_edges: npt.ArrayLike,
    extrapolate_method: str = "nan",
) -> npt.NDArray[np.floating]:
    """
    Compute the average value of a piecewise linear time series between specified x-edges.

    Parameters
    ----------
    x_data : array-like
        x-coordinates of the time series data points, must be in ascending order.
    y_data : array-like
        y-coordinates of the time series data points. Can be 1D or 2D.

        - If 1D: shape ``(n_data,)`` -- a single series.
        - If 2D: shape ``(n_series_y, n_data)`` -- multiple series sharing the same
          ``x_data``. The leading axis is averaged independently per row. Cannot be
          combined with 2D ``x_edges`` (each row of ``x_edges`` and each row of
          ``y_data`` would otherwise have to broadcast against each other, which is
          not supported).
    x_edges : array-like
        x-coordinates of the integration edges.

        - If 1D: shape ``(n_edges,)``, must be in ascending order.
        - If 2D: shape ``(n_series_x, n_edges)``, each row must be in ascending order.
    extrapolate_method : str, optional
        Method for handling bin edges that fall outside ``x_data``. Default
        is ``'nan'``.

        - ``'outer'``: average over the **in-range** portion of each bin
          (clip-then-average). The bin width used for normalisation is the
          clipped width, not the original width. For example,
          ``x_data = y_data = [1, 2, 3]`` and ``x_edges = [0, 5]`` returns
          ``2.0`` (integral over ``[1, 3]`` divided by clipped width 2),
          **not** ``2.2`` (which a constant-extension scheme would give).
        - ``'nan'``: bins that extend outside ``x_data`` are returned as ``nan``.
        - ``'raise'``: raise an error if any bin edge falls outside ``x_data``.

    Returns
    -------
    ndarray
        2D array of average values between consecutive pairs of x_edges.
        Shape is ``(n_series, n_bins)`` where ``n_bins = n_edges - 1`` and
        ``n_series = max(n_series_x, n_series_y)``. Both ``x_edges`` and ``y_data``
        being 1D yields ``n_series = 1``.

    Raises
    ------
    ValueError
        If ``x_edges`` is not 1D or 2D. If ``y_data`` is not 1D or 2D. If both
        ``x_edges`` and ``y_data`` are 2D. If ``x_data`` and ``y_data`` have
        incompatible shapes or are empty. If ``x_edges`` has fewer than 2 values per
        row. If ``x_data`` is not in ascending order. If ``x_edges`` rows are not in
        ascending order. If ``extrapolate_method`` is ``'raise'`` and any edge falls
        outside the data range.

    Notes
    -----
    **NaN handling is asymmetric between 1D and 2D ``y_data``.**

    - 1D ``y_data`` is treated as a single series; internal NaN gaps are
      silently bridged by linear interpolation across the gap (via
      ``np.interp`` with ``left=nan, right=nan``).
    - 2D ``y_data`` is treated row-wise; any output bin whose
      ``[edge_left, edge_right]`` touches a NaN segment **in that row** is
      set to NaN, while other rows are unaffected.

    Callers that need NaN-bridging behaviour across multiple series must
    pre-fill (e.g., ``pd.DataFrame.interpolate``) before calling.

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

    Multiple y-series with shared x_data and x_edges:

    >>> y_data_2d = [[0, 1, 1, 0], [0, 2, 2, 0]]
    >>> linear_average(x_data=x_data, y_data=y_data_2d, x_edges=x_edges)
    array([[0.66666667, 0.66666667],
           [1.33333333, 1.33333333]])
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

    # Ensure y_data is always 2D internally with shape (n_series_y, n_data)
    if y_data.ndim == 1:
        y_data = y_data[np.newaxis, :]
    elif y_data.ndim != 2:  # noqa: PLR2004
        msg = "y_data must be 1D or 2D array"
        raise ValueError(msg)

    # 2D y_data requires 1D x_edges (no per-row x_edges allowed). The combination would
    # require an outer product over (n_series_x, n_series_y), which is intentionally
    # not supported -- callers can loop or stack instead.
    n_series_x = x_edges.shape[0]
    n_series_y = y_data.shape[0]
    if n_series_x > 1 and n_series_y > 1:
        msg = "Cannot combine 2D x_edges with 2D y_data"
        raise ValueError(msg)
    n_series = max(n_series_x, n_series_y)

    # Input validation
    if y_data.shape[1] != x_data.shape[0] or x_data.shape[0] == 0:
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

    # Filter out NaN values. With 2D y_data, a column is dropped only when all rows
    # have NaN there; per-row NaNs are handled via segment masking below so that one
    # series' NaNs do not contaminate the others.
    x_nan = np.isnan(x_data)
    y_any_finite = np.any(~np.isnan(y_data), axis=0)
    show = ~x_nan & y_any_finite
    if show.sum() < 2:  # noqa: PLR2004
        if show.sum() == 1 and extrapolate_method == "outer":
            # For a single retained data point with outer extrapolation, use the
            # row-wise value broadcast across all output bins.
            constant_value = y_data[:, show][:, 0]  # shape (n_series_y,)
            return np.broadcast_to(constant_value[:, None], (n_series, x_edges.shape[1] - 1)).astype(
                np.float64, copy=True
            )
        return np.full(shape=(n_series, x_edges.shape[1] - 1), fill_value=np.nan)

    x_data_clean = x_data[show]
    y_data_clean = y_data[:, show]  # shape (n_series_y, n_clean)

    # Handle extrapolation for all series at once (vectorized). The 'raise' and 'nan'
    # branches never mutate edges_processed, so they alias x_edges directly; 'outer'
    # produces a fresh clipped array.
    if extrapolate_method == "outer":
        edges_processed = np.clip(x_edges, x_data_clean[0], x_data_clean[-1])
    elif extrapolate_method == "raise":
        if np.any(x_edges < x_data_clean[0]) or np.any(x_edges > x_data_clean[-1]):
            msg = "x_edges must be within the range of x_data"
            raise ValueError(msg)
        edges_processed = x_edges
    else:  # nan method
        edges_processed = x_edges

    # Create a combined grid of all unique x points (data + all edges)
    all_unique_x = np.unique(np.concatenate([x_data_clean, edges_processed.ravel()]))

    # Interpolate y values at all unique x points once. For 2D y_data we vectorize
    # the linear interpolation manually since np.interp does not accept 2D y.
    if n_series_y == 1:
        all_unique_y_result = np.interp(all_unique_x, x_data_clean, y_data_clean[0], left=np.nan, right=np.nan)
        all_unique_y: npt.NDArray[np.floating] = np.asarray(all_unique_y_result, dtype=np.float64)[np.newaxis, :]
    else:
        # Locate each query x in x_data_clean. For x within the data range, idx is in
        # [1, len(x_data_clean) - 1] so left_idx = idx - 1 is the bracketing left index.
        idx = np.searchsorted(x_data_clean, all_unique_x).clip(1, len(x_data_clean) - 1)
        left_idx = idx - 1
        right_idx = idx
        x_left = x_data_clean[left_idx]
        x_right = x_data_clean[right_idx]
        denom = x_right - x_left
        # Detect query points coincident with an x_data point. Handling them via a
        # direct lookup avoids the IEEE 754 trap where NaN * 0 = NaN, which would
        # otherwise contaminate exact-endpoint queries adjacent to a NaN sample.
        on_left_node = denom == 0  # only happens if x_left == x_right (duplicate)
        weights = np.where(on_left_node, 0.0, (all_unique_x - x_left) / np.where(on_left_node, 1.0, denom))
        all_unique_y = y_data_clean[:, left_idx] * (1.0 - weights) + y_data_clean[:, right_idx] * weights
        # Override at exact x_data positions to avoid NaN * 0 contamination.
        is_left_match = all_unique_x == x_left
        is_right_match = all_unique_x == x_right
        all_unique_y[:, is_left_match] = y_data_clean[:, left_idx[is_left_match]]
        all_unique_y[:, is_right_match] = y_data_clean[:, right_idx[is_right_match]]
        # Mark out-of-range query points as NaN (matches np.interp(left=nan, right=nan)).
        out_of_range = (all_unique_x < x_data_clean[0]) | (all_unique_x > x_data_clean[-1])
        all_unique_y[:, out_of_range] = np.nan

    # Compute cumulative integrals once using trapezoidal rule.
    # Segments outside the data range carry NaN (from the interp step with left/right=NaN);
    # those NaNs will be masked out later via the bin-range check, so we suppress
    # them here only to keep the cumulative sum finite for in-range bins.
    dx = np.diff(all_unique_x)
    y_avg = (all_unique_y[:, :-1] + all_unique_y[:, 1:]) / 2
    segment_integrals = np.where(np.isnan(y_avg), 0.0, dx[np.newaxis, :] * y_avg)
    # Cumulative integral with leading 0 along the x axis.
    cumulative_integral = np.concatenate([np.zeros((y_avg.shape[0], 1)), np.cumsum(segment_integrals, axis=1)], axis=1)

    # Vectorized computation for all series
    # Find indices of all edges in the combined grid
    edge_indices_result = np.searchsorted(all_unique_x, edges_processed)
    # Ensure it's a 2D array for type checker
    edge_indices: npt.NDArray[np.intp] = np.asarray(edge_indices_result, dtype=np.intp).reshape(edges_processed.shape)

    # Compute integral between consecutive edges. Broadcast over n_series via the leading axis
    # of cumulative_integral. edge_indices is (n_series_x, n_bins+1); cumulative_integral is
    # (n_series_y, n_unique_x). We rely on n_series_x == 1 or n_series_y == 1 (enforced above).
    integral_values = cumulative_integral[:, edge_indices[:, 1:]] - cumulative_integral[:, edge_indices[:, :-1]]
    # integral_values has shape (n_series_y, n_series_x, n_bins). Squeeze the singleton.
    integral_values_2d = integral_values[0] if n_series_y == 1 else integral_values[:, 0, :]

    # Compute widths between consecutive edges for all series (vectorized)
    edge_widths = np.diff(edges_processed, axis=1)  # shape (n_series_x, n_bins)
    # Broadcast widths to match (n_series, n_bins)
    edge_widths_b = np.broadcast_to(edge_widths, (n_series, edge_widths.shape[1])) if n_series_y > 1 else edge_widths

    # Handle zero-width intervals (vectorized)
    zero_width_mask = edge_widths_b == 0
    result = np.zeros_like(edge_widths_b, dtype=np.float64)

    # For non-zero width intervals, compute average = integral / width (vectorized)
    non_zero_mask = ~zero_width_mask
    result[non_zero_mask] = integral_values_2d[non_zero_mask] / edge_widths_b[non_zero_mask]

    # For zero-width intervals, interpolate y-value directly (vectorized)
    if np.any(zero_width_mask):
        # Positions where zero width occurs; use the left edge's x position.
        if n_series_y == 1:
            zero_positions = edges_processed[:, :-1][zero_width_mask]  # 1D
            result[zero_width_mask] = np.interp(zero_positions, x_data_clean, y_data_clean[0])
        else:
            # zero_width_mask has shape (n_series_y, n_bins); positions vary per row.
            # edges_processed is (1, n_bins+1) here since n_series_x == 1.
            edges_left = np.broadcast_to(edges_processed[:, :-1], (n_series, edge_widths.shape[1]))
            zero_positions = edges_left[zero_width_mask]
            # Interpolate per series using the same x_data_clean. Find bracketing indices
            # for each zero-width position, then index into the appropriate y row.
            # Get the row index for each zero-width entry.
            row_idx_grid = np.broadcast_to(np.arange(n_series)[:, None], (n_series, edge_widths.shape[1]))
            zero_rows = row_idx_grid[zero_width_mask]
            idx_z = np.searchsorted(x_data_clean, zero_positions).clip(1, len(x_data_clean) - 1)
            xl = x_data_clean[idx_z - 1]
            xr = x_data_clean[idx_z]
            denom_z = np.where(xr == xl, 1.0, xr - xl)
            w_z = (zero_positions - xl) / denom_z
            yl = y_data_clean[zero_rows, idx_z - 1]
            yr = y_data_clean[zero_rows, idx_z]
            result[zero_width_mask] = yl * (1.0 - w_z) + yr * w_z

    # Handle extrapolation when 'nan' method is used (vectorized).
    # Bins must lie entirely within the data range; bins partially outside
    # (straddling) are also set to NaN, since the integral over the missing
    # portion is undefined and dividing by the full bin width would bias the
    # average low. Bins fully outside are likewise NaN.
    if extrapolate_method == "nan":
        bins_within_range = (x_edges[:, :-1] >= x_data_clean[0]) & (x_edges[:, 1:] <= x_data_clean[-1])
        if n_series_y > 1:
            bins_within_range = np.broadcast_to(bins_within_range, (n_series, bins_within_range.shape[1]))
        result[~bins_within_range] = np.nan

        # With 2D y_data, propagate per-row NaNs from the y series itself: any output bin
        # that touches an x_data segment with NaN y in this row must be NaN. Per-row NaN
        # info is preserved in all_unique_y; mark bins whose [edge_left, edge_right]
        # contains a NaN segment for this row.
        if n_series_y > 1:
            # For each unique-x segment, is it NaN in this row?
            seg_nan = np.isnan(y_avg)  # shape (n_series_y, n_unique_x - 1)
            # A bin spans segments [edge_indices[0, b], edge_indices[0, b+1])
            # For each row, count NaN segments per bin via cumulative sums.
            seg_nan_cum = np.concatenate([np.zeros((n_series_y, 1)), np.cumsum(seg_nan, axis=1)], axis=1)
            nan_count_per_bin = seg_nan_cum[:, edge_indices[0, 1:]] - seg_nan_cum[:, edge_indices[0, :-1]]
            row_has_nan_in_bin = nan_count_per_bin > 0
            result[row_has_nan_in_bin] = np.nan

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
    overlap_matrix : ndarray
        Dense matrix of shape (n_in, n_out) where n_in is the number of input
        bins and n_out is the number of output bins. Each element (i, j)
        represents the fraction of input bin i that overlaps with output bin j.
        Values range from 0 (no overlap) to 1 (complete overlap).

    Raises
    ------
    ValueError
        If ``bin_edges_in`` or ``bin_edges_out`` are not 1D arrays. If either edge
        array has fewer than 2 elements. If ``bin_edges_in`` or ``bin_edges_out``
        are not in ascending order.

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

    # Edges must be non-decreasing (ignoring NaN). Zero-width input bins are
    # allowed -- they arise e.g. from cumulative flow with zero-flow intervals --
    # and produce zero overlap fractions (no water passed through).
    diffs_in = np.diff(bin_edges_in)
    valid_diffs_in = ~np.isnan(diffs_in)
    if np.any(valid_diffs_in) and not np.all(diffs_in[valid_diffs_in] >= 0):
        msg = "bin_edges_in must be non-decreasing"
        raise ValueError(msg)

    diffs_out = np.diff(bin_edges_out)
    valid_diffs_out = ~np.isnan(diffs_out)
    if np.any(valid_diffs_out) and not np.all(diffs_out[valid_diffs_out] >= 0):
        msg = "bin_edges_out must be non-decreasing"
        raise ValueError(msg)

    # Build matrix using fully vectorized approach
    # Create meshgrids for all possible input-output bin combinations
    in_left = bin_edges_in[:-1, None]  # Shape: (n_bins_in, 1)
    in_right = bin_edges_in[1:, None]  # Shape: (n_bins_in, 1)
    in_width = diffs_in[:, None]  # Shape: (n_bins_in, 1); reuses the validated np.diff above

    out_left = bin_edges_out[None, :-1]  # Shape: (1, n_bins_out)
    out_right = bin_edges_out[None, 1:]  # Shape: (1, n_bins_out)

    # Calculate overlaps for all combinations using broadcasting
    overlap_left = np.maximum(in_left, out_left)  # Shape: (n_bins_in, n_bins_out)
    overlap_right = np.minimum(in_right, out_right)  # Shape: (n_bins_in, n_bins_out)

    # Calculate overlap widths (zero where no overlap)
    overlap_widths = np.maximum(0, overlap_right - overlap_left)

    # Zero-width input bins contribute no overlap (division-safe). NaN widths still
    # propagate as NaN fractions (preserved for spin-up handling).
    with np.errstate(divide="ignore", invalid="ignore"):
        result = overlap_widths / in_width
    return np.where(in_width == 0, 0.0, result)


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
    overlap_array : ndarray
        Array of shape (len(bin_tedges), n_bins) where n_bins is the number of
        time bins. Each element (i, j) represents the fraction of time bin j
        that overlaps with time range i. Values range from 0 (no overlap) to
        1 (complete overlap).

    Raises
    ------
    ValueError
        If ``tedges`` is not a 1D array, has fewer than 2 elements, or if
        ``bin_tedges`` is empty.

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


def simplify_bins(
    *,
    edges: npt.ArrayLike,
    values: npt.ArrayLike,
    flow: npt.ArrayLike | None = None,
    tol: float = 0.0,
) -> tuple[
    npt.NDArray[np.floating] | pd.DatetimeIndex,
    npt.NDArray[np.floating],
    npt.NDArray[np.floating] | None,
]:
    """Simplify a piecewise-constant time series by merging adjacent bins.

    Recursively splits at the largest value jump until the peak-to-peak
    range within every group does not exceed `tol`. The result is
    independent of scan direction.

    Parameters
    ----------
    edges : array-like
        Bin edges with shape ``(n+1,)``. May be numeric or pandas Timestamps.
    values : array-like
        Bin-averaged values with shape ``(n,)`` (e.g., concentrations).
    flow : array-like, optional
        Flow rate per bin with shape ``(n,)`` (e.g., m³/day). When provided,
        merged-bin values are weighted by volume (flow x bin width) instead of
        bin width alone.
    tol : float, optional
        Maximum peak-to-peak range within a merged group.
        Default is 0.0, which merges only runs of identical values.

    Returns
    -------
    new_edges : ndarray or DatetimeIndex
        Simplified bin edges with shape ``(m+1,)``, preserving the type of
        `edges`.
    new_values : ndarray of float
        Volume-weighted (or width-weighted) average values per simplified
        bin, with shape ``(m,)``.
    new_flow : ndarray of float or None
        Time-weighted (width-weighted) average flow per simplified bin, with
        shape ``(m,)``. None when `flow` is not provided.
    """
    edges = np.asarray(edges) if not isinstance(edges, pd.DatetimeIndex) else edges
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        flow_out = np.asarray(flow, dtype=float) if flow is not None else None
        return edges, values, flow_out

    widths = np.asarray(np.diff(edges), dtype=float)
    if flow is not None:
        flow = np.asarray(flow, dtype=float)
        weights = widths * flow
    else:
        weights = widths

    def _splits(lo: int, hi: int) -> list[int]:
        if np.ptp(values[lo:hi]) <= tol:
            return []
        i = lo + int(np.argmax(np.abs(np.diff(values[lo:hi])))) + 1
        return [*_splits(lo, i), i, *_splits(i, hi)]

    s = np.array([0, *_splits(0, len(values))])
    idx = np.append(s, len(values))
    new_edges = edges[idx]
    new_widths = np.add.reduceat(widths, s)
    weight_sums = np.add.reduceat(weights, s)
    new_values = np.add.reduceat(weights * values, s) / weight_sums
    # When flow is given, weights == flow * widths, so weight_sums == reduceat(flow * widths, s) exactly.
    new_flow = weight_sums / new_widths if flow is not None else None

    return new_edges, new_values, new_flow


def _generate_failed_coverage_badge() -> None:
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
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
    c : ndarray
        Values from series a mapped to the combined edge structure.
    c_edges : ndarray
        Combined unique edges from a_edges and b_edges.
    d : ndarray
        Values from series b mapped to the combined edge structure.
    d_edges : ndarray
        Combined unique edges from a_edges and b_edges (same as c_edges).

    Raises
    ------
    ValueError
        If ``a_edges`` does not have ``len(a) + 1`` elements, or if ``b_edges`` does not
        have ``len(b) + 1`` elements.

    Notes
    -----
    The combined edges are created by taking the union of all unique values
    from both a_edges and b_edges, sorted in ascending order. The values
    are then broadcasted/repeated for each combined bin that falls within
    the original bin's range.
    """
    a = np.asarray(a, dtype=float)
    a_edges = np.asarray(a_edges, dtype=float)
    b = np.asarray(b, dtype=float)
    b_edges = np.asarray(b_edges, dtype=float)

    if len(a_edges) != len(a) + 1:
        msg = "a_edges must have len(a) + 1 elements"
        raise ValueError(msg)
    if len(b_edges) != len(b) + 1:
        msg = "b_edges must have len(b) + 1 elements"
        raise ValueError(msg)

    combined_edges = np.unique(np.concatenate([a_edges, b_edges]))
    combined_bin_centers = (combined_edges[:-1] + combined_edges[1:]) / 2

    def _map(values: npt.NDArray[np.floating], edges: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        # Map one series onto the combined bins, applying the chosen extrapolation.
        assignment = np.clip(np.searchsorted(edges, combined_bin_centers, side="right") - 1, 0, len(values) - 1)
        if extrapolation == "nearest":
            return values[assignment]
        out = np.zeros(len(combined_edges) - 1)
        # A combined bin keeps the original value only when it lies fully inside that bin.
        valid = (combined_edges[:-1] >= edges[assignment]) & (combined_edges[1:] <= edges[assignment + 1])
        out[valid] = values[assignment[valid]]
        out[~valid] = extrapolation
        return out

    c = _map(a, a_edges)
    d = _map(b, b_edges)

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
    - When ``tstart`` or ``tend`` are provided with non-uniformly-spaced bins,
      the extrapolated edge uses only the very first or very last interval and
      may be physically incorrect: the missing edge is implicitly assigned a
      bin width equal to that single neighbouring interval, which is unrelated
      to any other interval in the series. In such cases, supply ``tedges``
      directly so that all bin widths are explicit.
    - All input time data is converted to pandas.DatetimeIndex for consistency.
    """
    if tedges is not None:
        if number_of_bins != len(tedges) - 1:
            msg = "tedges must have one more element than flow"
            raise ValueError(msg)
        tedges = pd.DatetimeIndex(tedges)
        # Ensure nanosecond precision while preserving timezone
        return tedges.as_unit("ns")

    if tstart is not None:
        # Assume the index refers to the time at the start of the measurement interval
        tstart = pd.DatetimeIndex(tstart).as_unit("ns")
        if number_of_bins != len(tstart):
            msg = "tstart must have the same number of elements as flow"
            raise ValueError(msg)
        if len(tstart) < 2:  # noqa: PLR2004
            msg = "tstart must have at least 2 elements to infer the bin width; pass tedges for a single bin"
            raise ValueError(msg)

        # Extrapolate final edge using uniform spacing
        final_edge = tstart[-1] + (tstart[-1] - tstart[-2])
        return pd.DatetimeIndex([*list(tstart), final_edge], dtype=tstart.dtype)

    if tend is not None:
        # Assume the index refers to the time at the end of the measurement interval
        tend = pd.DatetimeIndex(tend).as_unit("ns")
        if number_of_bins != len(tend):
            msg = "tend must have the same number of elements as flow"
            raise ValueError(msg)
        if len(tend) < 2:  # noqa: PLR2004
            msg = "tend must have at least 2 elements to infer the bin width; pass tedges for a single bin"
            raise ValueError(msg)

        # Extrapolate initial edge using uniform spacing
        initial_edge = tend[0] - (tend[1] - tend[0])
        return pd.DatetimeIndex([initial_edge, *list(tend)], dtype=tend.dtype)

    msg = "Either provide tedges, tstart, or tend"
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
        cached = pd.read_pickle(cache_path)  # noqa: S301
        assert isinstance(cached, pd.DataFrame)  # noqa: S101 -- the cache only ever stores DataFrames
        return cached

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

    df = pd.read_csv(  # type: ignore[call-overload]  # ty: ignore[no-matching-overload]
        io.BytesIO(response.content),
        compression="zip",
        dtype=dtypes,  # pyright: ignore[reportArgumentType]
        usecols=list(dtypes.keys()),  # pyright: ignore[reportArgumentType]
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
    nullspace_objective: str
    | Callable[
        [npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]], float
    ] = "squared_differences",
    optimization_method: str = "BFGS",
    rcond: float | None = None,
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
    rcond : float or None, optional
        Cutoff ratio for small singular values in both ``numpy.linalg.lstsq``
        and ``scipy.linalg.null_space``. Singular values smaller than
        ``rcond * largest_singular_value`` are treated as zero.
        Default is None, which uses the default of each function.
        Increasing rcond truncates more modes, expanding the nullspace
        available for smoothness optimization. Useful for noisy data.

    Returns
    -------
    ndarray
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
    x_ls, *_ = np.linalg.lstsq(valid_matrix, valid_rhs, rcond=rcond)

    # Compute nullspace
    nullspace_basis = null_space(valid_matrix, rcond=rcond)
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
    x_ls: npt.NDArray[np.floating],
    nullspace_basis: npt.NDArray[np.floating],
    nullspace_objective: str
    | Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]], float],
    optimization_method: str,
) -> npt.NDArray[np.floating]:
    """Optimize coefficients in the nullspace to minimize the objective.

    Parameters
    ----------
    x_ls : ndarray
        Least-squares solution vector.
    nullspace_basis : ndarray
        Nullspace basis matrix of shape (n, nullrank).
    nullspace_objective : str or callable
        Objective to minimize. Supported string values are
        ``'squared_differences'`` and ``'summed_differences'``. A callable
        with signature ``objective(coeffs, x_ls, nullspace_basis)`` is also
        accepted.
    optimization_method : str
        Optimization method passed to ``scipy.optimize.minimize``.

    Returns
    -------
    ndarray
        Optimal nullspace coefficient vector of length nullrank.

    Raises
    ------
    ValueError
        If iterative optimization fails to converge.
    """
    # For squared_differences, solve the quadratic form analytically:
    # min ||D(x_ls + N c)||^2 => (N'D'DN) c = -N'D'D x_ls
    coeffs_sq = _solve_squared_differences_analytical(x_ls=x_ls, nullspace_basis=nullspace_basis)

    if nullspace_objective == "squared_differences":
        return coeffs_sq

    # For other objectives, use iterative optimization starting from the
    # squared_differences solution for stability
    objective_func = _get_nullspace_objective_function(nullspace_objective=nullspace_objective)
    coeffs_0 = coeffs_sq

    res = minimize(
        objective_func,
        x0=coeffs_0,
        args=(x_ls, nullspace_basis),
        method=optimization_method,
    )

    if not res.success:
        msg = f"Optimization failed: {res.message}"
        raise ValueError(msg)

    return res.x


def _solve_squared_differences_analytical(
    *,
    x_ls: npt.NDArray[np.floating],
    nullspace_basis: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Solve the squared-differences nullspace problem analytically.

    Minimizes ``sum((x[i+1] - x[i])^2)`` where ``x = x_ls + N @ c`` by
    solving the normal equations ``(N^T D^T D N) c = -N^T D^T D x_ls``.

    Parameters
    ----------
    x_ls : ndarray
        Least-squares solution vector of length n.
    nullspace_basis : ndarray
        Nullspace basis matrix of shape (n, nullrank).

    Returns
    -------
    ndarray
        Optimal nullspace coefficient vector of length nullrank.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the normal equations matrix ``(DN)^T(DN)`` is ill-conditioned
        (condition number exceeds 1e12).
    """
    # D is the (n-1, n) first-difference matrix; D @ x = x[1:] - x[:-1]
    # D^T D is the tridiagonal matrix with 2 on diagonal, -1 on off-diagonals
    # (except corners which have 1 on diagonal)
    # Instead of forming D explicitly, compute D @ N and D @ x_ls directly
    dn = nullspace_basis[1:, :] - nullspace_basis[:-1, :]  # (n-1, nullrank)
    dx = x_ls[1:] - x_ls[:-1]  # (n-1,)

    # Normal equations: (DN)^T (DN) c = -(DN)^T (D x_ls)
    dntdn = dn.T @ dn  # (nullrank, nullrank)
    rhs = -(dn.T @ dx)  # (nullrank,)

    cond = np.linalg.cond(dntdn)
    cond_threshold = 1e12
    if cond > cond_threshold:
        msg = (
            f"The normal equations matrix (DN)^T(DN) is ill-conditioned "
            f"(condition number: {cond:.2e}). This typically means the "
            f"nullspace contains a near-constant vector, so the "
            f"squared-differences objective cannot distinguish between "
            f"nullspace directions. Consider using a different "
            f"nullspace_objective (e.g., 'summed_differences'), reducing "
            f"the problem's degrees of freedom, or lowering rcond to "
            f"shrink the nullspace (if the near-constant vector has a "
            f"small but non-zero singular value)."
        )
        raise np.linalg.LinAlgError(msg)

    return np.linalg.solve(dntdn, rhs)


def compute_reverse_target(
    *,
    coeff_matrix: npt.NDArray[np.floating],
    rhs_vector: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute reverse matrix target from forward coefficient matrix.

    Constructs a target solution for the inverse problem by transposing the
    forward coefficient matrix and normalizing rows. For ``W_forward[i,j]``
    representing the fraction of ``cin[j]`` arriving in ``cout[i]``, the
    transpose-and-normalize approach reconstructs ``cin[j]`` as a weighted
    average of ``cout`` bins, weighted by how much ``cin[j]`` contributed
    to each ``cout`` bin.

    Parameters
    ----------
    coeff_matrix : ndarray
        Forward coefficient matrix of shape (n_cout, n_cin).
    rhs_vector : ndarray
        Right-hand side vector of length n_cout (e.g., cout values).

    Returns
    -------
    ndarray
        Target solution vector of length n_cin. Entries with near-zero
        column sums in the forward matrix are set to NaN.

    See Also
    --------
    solve_tikhonov : Consumes this target as the regularization reference.
    """
    min_row_sum = 1e-10
    wt = coeff_matrix.T  # (n_cin, n_cout)
    row_sums = wt.sum(axis=1)
    valid = row_sums > min_row_sum
    w_reverse = np.zeros_like(wt)
    w_reverse[valid] = wt[valid] / row_sums[valid, None]
    x_target = w_reverse @ rhs_vector
    x_target[~valid] = np.nan
    return x_target


def solve_tikhonov(
    *,
    coefficient_matrix: npt.ArrayLike,
    rhs_vector: npt.ArrayLike,
    x_target: npt.NDArray[np.floating],
    regularization_strength: float = 1e-10,
    return_resolution: bool = False,
) -> npt.NDArray[np.floating] | tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Solve a linear system with Tikhonov regularization toward a target.

    Minimizes ``||A x - b||² + λ ||x - x_target||²`` by solving the
    equivalent augmented least-squares problem::

        [A; √λ I_v] x = [b; √λ x_target_v]

    where ``I_v`` selects only entries where ``x_target`` is not NaN.

    Well-determined modes (large singular values relative to √λ) are
    dominated by the data; poorly-determined modes are pulled toward
    ``x_target``. The solution varies continuously with λ, unlike the
    hard singular-value cutoff of ``rcond`` in truncated SVD.

    Parameters
    ----------
    coefficient_matrix : array-like
        Coefficient matrix of shape (m, n). May contain NaN rows, which
        are excluded from the system.
    rhs_vector : array-like
        Right-hand side vector of length m. May contain NaN values
        corresponding to NaN rows in coefficient_matrix.
    x_target : ndarray
        Target solution of length n, typically from
        :func:`compute_reverse_target`. NaN entries are excluded from the
        regularization term.
    regularization_strength : float, optional
        Tikhonov parameter λ. Controls the tradeoff between fitting the
        data and staying close to ``x_target``. Larger values trust the
        target more; smaller values trust the data more. Default is 1e-10.

        A good starting value for noisy data is
        ``λ ≈ (noise_std / signal_amplitude)²``. For noiseless synthetic
        data, the default 1e-10 preserves machine precision.
    return_resolution : bool, optional
        If True, also return the per-element fraction of the solution that
        comes from data (vs from the regularization target). Default is
        False.

    Returns
    -------
    ndarray or tuple of ndarray
        If ``return_resolution`` is False (default), returns the solution
        vector of length n.

        If ``return_resolution`` is True, returns ``(x, fraction_data)``
        where ``fraction_data[j]`` is the diagonal of the model resolution
        matrix ``R = (A^T A + λ D)^{-1} A^T A``:

        - ``fraction_data[j] ≈ 1``: element *j* is data-driven
        - ``fraction_data[j] ≈ 0``: element *j* is target-driven
        - Non-regularized entries (NaN in ``x_target``):
          ``fraction_data[j] = 1.0``

    Raises
    ------
    ValueError
        If ``coefficient_matrix`` and ``rhs_vector`` have incompatible shapes, or if
        all rows contain NaN values.

    See Also
    --------
    compute_reverse_target : Compute the regularization target from the
        forward matrix.
    solve_underdetermined_system : Alternative solver using nullspace
        optimization.
    """
    matrix = np.asarray(coefficient_matrix)
    rhs = np.asarray(rhs_vector)

    if matrix.shape[0] != len(rhs):
        msg = f"coefficient_matrix has {matrix.shape[0]} rows but rhs_vector has {len(rhs)} elements"
        raise ValueError(msg)

    # Filter NaN rows
    valid_rows = ~np.isnan(matrix).any(axis=1) & ~np.isnan(rhs)

    if not np.any(valid_rows):
        msg = "No valid rows found (all contain NaN values)"
        raise ValueError(msg)

    valid_matrix = matrix[valid_rows]
    valid_rhs = rhs[valid_rows]

    n_cin = valid_matrix.shape[1]
    sqrt_lam = np.sqrt(regularization_strength)

    # Only regularize entries where x_target is valid
    valid_target = ~np.isnan(x_target)
    target_indices = np.where(valid_target)[0]

    # Build augmented system: [A; √λ I_v] x = [b; √λ x_target_v]
    n_reg = len(target_indices)
    reg_matrix = np.zeros((n_reg, n_cin))
    reg_matrix[np.arange(n_reg), target_indices] = sqrt_lam
    reg_rhs = sqrt_lam * x_target[target_indices]

    augmented_matrix = np.vstack([valid_matrix, reg_matrix])
    augmented_rhs = np.concatenate([valid_rhs, reg_rhs])

    x, *_ = np.linalg.lstsq(augmented_matrix, augmented_rhs, rcond=None)

    if return_resolution:
        # Compute fraction_data from model resolution matrix diagonal:
        # R = G^{-1} A^T A  where  G = A^T A + λ diag(d)
        # fraction_data[j] = R[j,j] = 1 - λ d[j] G_inv[j,j]
        d_reg = np.zeros(n_cin)
        d_reg[target_indices] = 1.0
        gram = valid_matrix.T @ valid_matrix
        gram[np.arange(n_cin), np.arange(n_cin)] += regularization_strength * d_reg
        gram_inv_diag = np.diag(np.linalg.inv(gram))
        fraction_data = 1.0 - regularization_strength * gram_inv_diag * d_reg
        return x, fraction_data

    return x


# Numerical tolerance for coefficient sum to determine valid output bins
_EPSILON_COEFF_SUM = 1e-10

# Corrected semi-normal-equation refinement steps in solve_inverse_transport_banded. One
# step reaches the QR-accurate solution; a second is a cheap, stable safety margin.
_BANDED_REFINEMENT_STEPS = 2


def solve_inverse_transport(
    *,
    w_forward: npt.NDArray[np.floating],
    observed: npt.NDArray[np.floating],
    n_output: int,
    regularization_strength: float,
    valid_rows: npt.NDArray[np.bool_] | None = None,
    warn_rank_deficient: bool = False,
) -> npt.NDArray[np.floating]:
    """Solve the inverse transport problem via Tikhonov regularization.

    Given the forward model ``w_forward @ x = observed``, recovers ``x`` by
    building the regularization target from the transpose of ``w_forward`` and
    solving the regularized least-squares problem.

    Parameters
    ----------
    w_forward : ndarray
        Forward coefficient matrix with shape ``(n_obs, n_output)``.
    observed : ndarray
        Observed values with shape ``(n_obs,)`` (e.g., extraction
        concentrations).
    n_output : int
        Length of the output vector (e.g., number of cin bins).
    regularization_strength : float
        Tikhonov regularization parameter.
    valid_rows : ndarray of bool, optional
        Which observation rows are valid, with shape ``(n_obs,)``. If None,
        rows with ``row_sum > 1e-10`` are considered valid.
    warn_rank_deficient : bool, optional
        If True, emit a warning when the forward matrix has rank
        deficiency among its active columns. Default is False.

    Returns
    -------
    ndarray
        Recovered signal with shape ``(n_output,)``. NaN for bins with no
        active columns.

    Warns
    -----
    UserWarning
        When ``warn_rank_deficient=True`` and the matrix is rank-deficient.

    See Also
    --------
    solve_inverse_transport_banded : Memory-light banded equivalent.
    """
    row_sums = w_forward.sum(axis=1)
    col_active: npt.NDArray[np.bool_] = w_forward.sum(axis=0) > 0

    if not np.any(col_active):
        return np.full(n_output, np.nan)

    if warn_rank_deficient:
        n_active = int(col_active.sum())
        rank = np.linalg.matrix_rank(w_forward[:, col_active])
        if rank < n_active:
            warnings.warn(
                f"Forward matrix is rank-deficient (rank {rank} < {n_active} active "
                f"columns). This occurs with constant flow when the residence time "
                f"is an integer multiple of the time step width. The "
                f"underdetermined modes will be pulled toward the regularization "
                f"target instead of being determined by data. To achieve full rank, "
                f"adjust aquifer_pore_volumes slightly (e.g., multiply by 1.001).",
                stacklevel=2,
            )

    valid: npt.NDArray[np.bool_] = row_sums > _EPSILON_COEFF_SUM if valid_rows is None else valid_rows

    rhs = np.where(valid, row_sums * observed, np.nan)
    w_solve = w_forward.copy()
    w_solve[~valid, :] = np.nan

    x_target = compute_reverse_target(coeff_matrix=w_forward, rhs_vector=observed)

    x_solved = solve_tikhonov(
        coefficient_matrix=w_solve,
        rhs_vector=rhs,
        x_target=x_target,
        regularization_strength=regularization_strength,
    )

    out = np.full(n_output, np.nan)
    idx = np.flatnonzero(col_active)
    out[idx] = x_solved[idx]
    return out


def solve_inverse_transport_banded(
    *,
    band_vals: npt.NDArray[np.floating],
    col_start: npt.NDArray[np.intp],
    observed: npt.NDArray[np.floating],
    n_output: int,
    regularization_strength: float,
) -> npt.NDArray[np.floating]:
    """Solve the inverse transport problem from a banded forward operator.

    Memory-light equivalent of :func:`solve_inverse_transport` for a forward
    weight matrix stored in banded layout: row ``k`` of the dense operator
    ``W`` is ``band_vals[k]`` placed at columns
    ``[col_start[k], col_start[k] + full_band)``. The Tikhonov normal
    equations ``(WᵀW + λ D) x = Wᵀ observed + λ D x_target`` are assembled
    **directly in banded form** -- ``WᵀW`` is symmetric with half-bandwidth
    ``full_band - 1`` -- and Cholesky-factored with
    :func:`scipy.linalg.cholesky_banded`. Forming ``WᵀW`` squares the condition
    number, so the bare Cholesky solve loses accuracy in the under-determined
    (spin-up nullspace) directions; **corrected semi-normal equations** restore
    it by refining with the residual evaluated through ``W`` itself rather than
    ``WᵀW`` (matching the dense least-squares solution to ~1e-10). Peak memory is
    ``O(n_output * full_band)``, never the dense ``O(n_obs * n_output)``.

    The regularization target ``x_target`` is the transpose-and-normalize of
    ``W`` applied to ``observed`` (the banded form of
    :func:`compute_reverse_target`), matching the dense solver. Columns with no
    forward contribution are decoupled (unit diagonal) so the system stays
    symmetric positive definite, and are returned as NaN.

    Parameters
    ----------
    band_vals : ndarray
        Banded forward weights of shape ``(n_obs, full_band)``. Rows the caller
        considers invalid must already be zeroed (as ``_resolve_spinup_mask``
        does); zero rows contribute nothing to the normal equations.
    col_start : ndarray of int
        First output-column index of each row's band, shape ``(n_obs,)``.
    observed : ndarray
        Observed values of shape ``(n_obs,)`` (e.g. extraction concentrations).
        Must not contain NaN.
    n_output : int
        Length of the output vector (number of cin bins).
    regularization_strength : float
        Tikhonov parameter λ. See :func:`solve_inverse_transport`. Must be
        strictly positive: deconvolution is generically rank-deficient, and λ
        is what makes the banded Cholesky factor positive definite (unlike the
        dense least-squares path, this solver cannot return a λ=0 min-norm
        solution).

    Returns
    -------
    ndarray
        Recovered signal of shape ``(n_output,)``. NaN for output bins with no
        forward contribution (zero column).

    Raises
    ------
    ValueError
        If ``regularization_strength`` is not strictly positive.

    See Also
    --------
    solve_inverse_transport : Dense-matrix equivalent.
    ``gwtransport.advection_utils._infiltration_to_extraction_weights`` : Banded builder.
    """
    if regularization_strength <= 0:
        msg = "regularization_strength must be > 0 for the banded inverse (Tikhonov positive-definiteness)"
        raise ValueError(msg)
    # Precondition: the caller's valid rows sum to 1 (guaranteed by
    # _resolve_spinup_mask), so the data equation is W x ≈ observed and the RHS
    # needs no row_sums scaling -- matching the dense solve_inverse_transport.
    band_vals = np.asarray(band_vals, dtype=float)
    observed = np.asarray(observed, dtype=float)
    full_band = band_vals.shape[1]
    n_cin = n_output
    cols = col_start[:, None] + np.arange(full_band)[None, :]  # (n_obs, full_band) output-column index
    in_range = cols < n_cin
    cols_clipped = np.clip(cols, 0, n_cin - 1)

    # Column sums and Wᵀ observed (the reverse-target numerator) by scattering the band.
    col_sum = np.zeros(n_cin)
    wt_observed = np.zeros(n_cin)
    np.add.at(col_sum, cols_clipped[in_range], band_vals[in_range])
    np.add.at(wt_observed, cols_clipped[in_range], (band_vals * observed[:, None])[in_range])

    col_active = col_sum > 0
    if not np.any(col_active):
        return np.full(n_output, np.nan)

    # Reverse-target: transpose-and-normalize W applied to observed (banded form of
    # compute_reverse_target). The sliver 0 < col_sum <= _EPSILON_COEFF_SUM is left
    # untargeted (filled with 0) as in the dense path.
    with np.errstate(invalid="ignore", divide="ignore"):
        x_target = np.where(col_sum > _EPSILON_COEFF_SUM, wt_observed / col_sum, 0.0)

    # Lower-banded WᵀW: band row d is the d-th sub-diagonal. A row's contribution to
    # (i, j) with i = col_start + b1, j = col_start + b2 lands at offset d = b1 - b2 >= 0
    # and column j. band_vals is zero outside each window, so off-window pairs add 0.
    ab = np.zeros((full_band, n_cin))
    for d in range(full_band):
        prod = band_vals[:, d:] * band_vals[:, : full_band - d]
        c = cols_clipped[:, : full_band - d]
        m = in_range[:, : full_band - d] & in_range[:, d:]
        np.add.at(ab[d], c[m], prod[m])

    lam = regularization_strength
    d_reg = lam * col_active
    ab[0] += d_reg
    # d_reg is zero off the active columns, so x_target needs no masking here or in
    # the refinement loop: the product d_reg * x_target vanishes wherever col_active is False.
    rhs = wt_observed + d_reg * x_target

    # Decouple zero (inactive, unregularized) diagonals so the matrix is SPD.
    dead = ab[0] <= 0.0
    ab[0, dead] = 1.0
    rhs[dead] = 0.0

    factor = cholesky_banded(ab, lower=True)
    x = cho_solve_banded((factor, True), rhs)

    # Forming WᵀW squares the condition number, so the bare Cholesky solution loses
    # accuracy in the under-determined (spin-up nullspace) directions. Corrected
    # semi-normal equations recover it: the residual is evaluated through W itself
    # (in observation space) rather than through WᵀW, avoiding the cancellation that
    # makes plain normal-equation refinement stall. One step reaches the QR-accurate
    # solution; the rest are a safety margin (the iteration's fixed point is stable).
    for _ in range(_BANDED_REFINEMENT_STEPS):
        gathered = x[cols_clipped]
        gathered[~in_range] = 0.0
        residual = observed - (band_vals * gathered).sum(axis=1)  # b - W x  (n_obs,)
        gradient = np.zeros(n_cin)
        np.add.at(gradient, cols_clipped[in_range], (band_vals * residual[:, None])[in_range])  # Wᵀ (b - W x)
        gradient += d_reg * (x_target - x)
        gradient[dead] = 0.0
        x += cho_solve_banded((factor, True), gradient)

    out = np.full(n_output, np.nan)
    out[col_active] = x[col_active]
    return out


def _squared_differences_objective(
    coeffs: npt.NDArray[np.floating], x_ls: npt.NDArray[np.floating], nullspace_basis: npt.NDArray[np.floating]
) -> float:
    """Minimize sum of squared differences between adjacent elements.

    Parameters
    ----------
    coeffs : ndarray
        Nullspace coefficient vector.
    x_ls : ndarray
        Least-squares solution vector.
    nullspace_basis : ndarray
        Nullspace basis matrix.

    Returns
    -------
    float
        Sum of squared differences between adjacent elements of the solution.
    """
    x = x_ls + nullspace_basis @ coeffs
    return np.sum(np.square(x[1:] - x[:-1]))


def _summed_differences_objective(
    coeffs: npt.NDArray[np.floating], x_ls: npt.NDArray[np.floating], nullspace_basis: npt.NDArray[np.floating]
) -> float:
    """Minimize sum of absolute differences between adjacent elements.

    Parameters
    ----------
    coeffs : ndarray
        Nullspace coefficient vector.
    x_ls : ndarray
        Least-squares solution vector.
    nullspace_basis : ndarray
        Nullspace basis matrix.

    Returns
    -------
    float
        Sum of absolute differences between adjacent elements of the solution.
    """
    x = x_ls + nullspace_basis @ coeffs
    return np.sum(np.abs(x[1:] - x[:-1]))


def _get_nullspace_objective_function(
    *,
    nullspace_objective: str
    | Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]], float],
) -> Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]], float]:
    """Get the objective function for nullspace optimization.

    Parameters
    ----------
    nullspace_objective : str or callable
        Objective identifier. Supported string values are
        ``'squared_differences'`` and ``'summed_differences'``. A callable
        with signature ``objective(coeffs, x_ls, nullspace_basis)`` is also
        accepted and returned as-is.

    Returns
    -------
    callable
        Objective function with signature
        ``(coeffs, x_ls, nullspace_basis) -> float``.

    Raises
    ------
    ValueError
        If ``nullspace_objective`` is an unrecognized string.
    """
    if nullspace_objective == "squared_differences":
        return _squared_differences_objective
    if nullspace_objective == "summed_differences":
        return _summed_differences_objective
    if callable(nullspace_objective):
        return nullspace_objective  # type: ignore[return-value]  # ty: ignore[invalid-return-type]
    msg = f"Unknown nullspace objective: {nullspace_objective}"
    raise ValueError(msg)
