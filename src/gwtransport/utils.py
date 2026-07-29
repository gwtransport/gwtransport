"""
General Utilities for 1D Groundwater Transport Modeling.

This module provides general-purpose utility functions for time series manipulation,
interpolation, numerical operations, and data processing used throughout the gwtransport
package. Functions include linear interpolation, cumulative flow volumes, time-edge
construction, linear-system solvers, and external data retrieval.

The inverse solvers below are two intentionally coexisting families: a Tikhonov family (the dense
:func:`solve_inverse_transport` and its banded equivalent :func:`solve_inverse_transport_banded`,
both fed by :func:`compute_reverse_target` and built on :func:`solve_tikhonov`) for the
overdetermined deconvolution in advection/diffusion, and a separate nullspace solver
(:func:`solve_underdetermined_system`) for the underdetermined deposition inverse.

Available functions:

- :func:`step_plot_coords` - Expand bin edges (n+1) and bin-averaged values (n) into paired x/y arrays of 2n
  points each, so that ``ax.plot(x, y)`` draws the piecewise-constant series as a step function. Edges may be
  numeric or datetime; each output keeps the dtype of its input.

- :func:`cumulative_flow_volume` - Accumulate per-bin flow rates times bin widths into the cumulative volume at
  every bin edge (n+1 values, starting at zero). With ``strictly_monotone=True`` the plateaus left by zero-flow
  bins are bumped by a few ulps, which is required before inverting the sequence from volume back to time.

- :func:`linear_interpolate` - Interpolate ``y_ref`` at ``x_query`` linearly; ``x_ref`` must be ascending and the
  result has the shape of ``x_query``. Query points outside the reference range clamp to the end values unless
  ``left`` / ``right`` supply a fill value such as NaN.

- :func:`simplify_bins` - Merge adjacent bins of a piecewise-constant series until the peak-to-peak range within
  each merged group is at most ``tol``, returning the merged edges, values and flow. Values are volume-weighted
  (flow times width) when ``flow`` is given and width-weighted otherwise, while the merged flow is always
  width-weighted. Splitting at the largest value jump makes the result independent of scan direction.

- :func:`compute_time_edges` - Build the n+1 bin edges as a nanosecond-precision DatetimeIndex from exactly one of
  explicit edges, per-bin start times, or per-bin end times, validating the length against ``number_of_bins``.
  From ``tstart`` or ``tend`` the single missing outer edge is extrapolated from the adjacent interval alone, so
  pass ``tedges`` directly when the bins are not uniformly spaced.

- :func:`get_soil_temperature` - Download the KNMI soil-temperature record of one of four Dutch weather stations
  and return it as a DataFrame in degrees Celsius on a UTC DatetimeIndex, with columns for the temperature at 5,
  10, 20, 50 and 100 cm depth and the six-hourly minima and maxima at 5 and 10 cm. The download is cached on disk
  for the calendar day; missing values are interpolated and forward-filled unless disabled. This is the only
  function here that needs ``requests``, which is therefore imported lazily.

- :func:`solve_underdetermined_system` - Solve ``A x = b`` for a wide system (more unknowns than equations) by
  taking the least-squares solution and adding the nullspace component that minimizes a roughness objective;
  rows holding NaN are dropped first. The closed-form squared-differences optimum is always computed and also
  seeds the iterative optimization of the other objectives, so a nullspace containing a near-constant vector
  raises :class:`numpy.linalg.LinAlgError` whichever objective is requested.

- :func:`compute_reverse_target` - Transpose the forward coefficient matrix, normalize its rows and apply it to
  the observations, giving each input bin the contribution-weighted average of the output bins it fed. This is
  the reference solution the Tikhonov solvers pull poorly-determined modes toward; input bins with negligible
  forward weight are returned as NaN.

- :func:`solve_tikhonov` - Solve ``min ||A x - b||² + λ ||x - x_target||²`` as a single augmented least-squares
  problem. Rows of the system holding NaN are excluded, and NaN entries of ``x_target`` are left unregularized.

- :func:`solve_inverse_transport` - Recover the input signal of the dense forward model
  ``w_forward @ x = observed`` by building the target with :func:`compute_reverse_target` and solving it with
  :func:`solve_tikhonov`. Observation rows that are NaN or carry negligible weight drop out (an explicit
  ``valid_rows`` mask overrides the weight test), and output bins left without forward contribution come back
  as NaN.

- :func:`solve_inverse_transport_banded` - Solve the same inverse problem for a forward operator stored in banded
  layout (row ``k`` is ``band_vals[k]`` placed at column ``col_start[k]``), through banded Cholesky normal
  equations plus corrected semi-normal refinement. The factorization, solve and refinement stay at
  ``O(n_output * full_band)``. The regularization strength must be strictly positive, since it is what makes the
  banded factor positive definite.

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

    Parameters
    ----------
    x_ref : array-like
        Reference x-values, in ascending order.
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
    """
    return np.interp(np.asarray(x_query), np.asarray(x_ref), np.asarray(y_ref), left=left, right=right)


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

    Splits at the largest value jump until the peak-to-peak range within
    every group does not exceed `tol`. The result is independent of scan
    direction.

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

    # Iteratively split each segment at its largest value jump until every group's peak-to-peak
    # range is within tol. An explicit LIFO stack replaces the natural recursion, which peels one
    # element per level on smooth monotone data (argmax|diff| sits at a segment edge) and overflows
    # the interpreter stack for a few thousand points. Every split index is interior to its
    # (disjoint) segment, so sorting the collected splits reproduces the recursion's in-order
    # output exactly -- the merged bins are identical.
    splits: list[int] = []
    stack: list[tuple[int, int]] = [(0, len(values))]
    while stack:
        lo, hi = stack.pop()
        if np.ptp(values[lo:hi]) <= tol:
            continue
        i = lo + int(np.argmax(np.abs(np.diff(values[lo:hi])))) + 1
        splits.append(i)
        stack.extend(((lo, i), (i, hi)))
    splits.sort()
    s = np.array([0, *splits])
    idx = np.append(s, len(values))
    new_edges = edges[idx]
    new_widths = np.add.reduceat(widths, s)
    weight_sums = np.add.reduceat(weights, s)
    # A merged group of all-zero-flow bins has zero volume weight (0/0 -> NaN); its average is
    # still well defined, so fall back to width weighting there.
    zero_weight = weight_sums == 0.0
    new_values = np.where(
        zero_weight, np.add.reduceat(widths * values, s), np.add.reduceat(weights * values, s)
    ) / np.where(zero_weight, new_widths, weight_sums)
    # When flow is given, weights == flow * widths, so weight_sums == reduceat(flow * widths, s) exactly.
    new_flow = weight_sums / new_widths if flow is not None else None

    return new_edges, new_values, new_flow


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
            msg = "tedges must have one more element than number_of_bins"
            raise ValueError(msg)
        tedges = pd.DatetimeIndex(tedges)
        # Ensure nanosecond precision while preserving timezone
        return tedges.as_unit("ns")

    if tstart is not None:
        # Assume the index refers to the time at the start of the measurement interval
        tstart = pd.DatetimeIndex(tstart).as_unit("ns")
        if number_of_bins != len(tstart):
            msg = "tstart must have the same number of elements as number_of_bins"
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
            msg = "tend must have the same number of elements as number_of_bins"
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

    # Imported lazily so the rest of the module remains importable in environments
    # without ``requests`` (e.g. Pyodide/JupyterLite, where this KNMI download is the
    # only feature that cannot run client-side).
    import requests  # noqa: PLC0415

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
    numpy.linalg.LinAlgError
        If the squared-differences normal equations ``(DN)^T(DN)`` are ill-conditioned
        (condition number above 1e12), which happens when the nullspace contains a
        near-constant vector.

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
    Rows containing NaN are dropped; the solution satisfies the remaining equations:

    >>> import numpy as np
    >>> from gwtransport.utils import solve_underdetermined_system
    >>> matrix = np.array([
    ...     [1.0, 2.0, 1.0, 0.0],
    ...     [np.nan, np.nan, np.nan, np.nan],
    ...     [0.0, 1.0, 2.0, 1.0],
    ... ])
    >>> rhs = np.array([3.0, np.nan, 4.0])
    >>> x = solve_underdetermined_system(coefficient_matrix=matrix, rhs_vector=rhs)
    >>> bool(np.allclose(matrix[[0, 2]] @ x, [3.0, 4.0]))
    True
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

    if nullspace_basis.shape[1] == 0:
        # System is determined, return least-squares solution
        return x_ls

    # Squared-differences optimum in closed form: minimizing ||D(x_ls + N c)||^2 gives the normal
    # equations (DN)^T(DN) c = -(DN)^T(D x_ls), where D is the (n-1, n) first-difference matrix.
    # D @ N and D @ x_ls are formed directly instead of materializing D.
    dn = nullspace_basis[1:, :] - nullspace_basis[:-1, :]  # (n-1, nullrank)
    dx = x_ls[1:] - x_ls[:-1]  # (n-1,)
    dntdn = dn.T @ dn  # (nullrank, nullrank)

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

    coeffs = np.linalg.solve(dntdn, -(dn.T @ dx))

    if nullspace_objective != "squared_differences":
        # Other objectives are optimized iteratively, started from the squared-differences
        # solution for stability.
        if nullspace_objective == "summed_differences":
            objective_func = _summed_differences_objective
        elif callable(nullspace_objective):
            objective_func = nullspace_objective
        else:
            msg = f"Unknown nullspace objective: {nullspace_objective}"
            raise ValueError(msg)

        res = minimize(objective_func, x0=coeffs, args=(x_ls, nullspace_basis), method=optimization_method)
        if not res.success:
            msg = f"Optimization failed: {res.message}"
            raise ValueError(msg)
        coeffs = res.x

    return x_ls + nullspace_basis @ coeffs


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
) -> npt.NDArray[np.floating]:
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

    Returns
    -------
    ndarray
        Solution vector of length n.

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
        concentrations). NaN entries mark measurement gaps; their rows are
        excluded from the solve and the regularization target.
    n_output : int
        Length of the output vector (e.g., number of cin bins).
    regularization_strength : float
        Tikhonov regularization parameter.
    valid_rows : ndarray of bool, optional
        Which observation rows are valid, with shape ``(n_obs,)``. If None,
        rows with ``row_sum > 1e-10`` are considered valid.

    Returns
    -------
    ndarray
        Recovered signal with shape ``(n_output,)``. NaN for bins with no
        active columns.

    See Also
    --------
    solve_inverse_transport_banded : Memory-light banded equivalent.
    """
    row_sums = w_forward.sum(axis=1)
    nan_obs = np.isnan(observed)
    # Aliases w_forward when there are no gaps; otherwise one masked copy, shared with the
    # regularization target below.
    w_masked = np.where(nan_obs[:, None], 0.0, w_forward) if nan_obs.any() else w_forward
    # A column is active when its weight over the surviving rows exceeds the regularization
    # epsilon; sliver-support and gap-only columns emit NaN instead of a min-norm value.
    col_active: npt.NDArray[np.bool_] = w_masked.sum(axis=0) > _EPSILON_COEFF_SUM

    if not np.any(col_active):
        return np.full(n_output, np.nan)

    # Gapped rows drop out of the data equations and the regularization target.
    valid: npt.NDArray[np.bool_] = (row_sums > _EPSILON_COEFF_SUM if valid_rows is None else valid_rows) & ~nan_obs

    rhs = np.where(valid, row_sums * observed, np.nan)
    w_solve = w_forward.copy()
    w_solve[~valid, :] = np.nan

    x_target = compute_reverse_target(
        coeff_matrix=w_masked,
        rhs_vector=np.where(nan_obs, 0.0, observed),
    )

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
    equations ``(WᵀW + λ D) x = Wᵀ observed + λ D x_target`` are stored **in
    banded form** -- ``WᵀW`` is symmetric with half-bandwidth ``full_band - 1``
    -- and Cholesky-factored with :func:`scipy.linalg.cholesky_banded`. The Gram
    matrix ``WᵀW`` is built with a single dense BLAS matmul (``~24x`` a
    per-diagonal scatter) before its sub-diagonals are read into the banded
    layout. Forming ``WᵀW`` squares the condition number, so the bare Cholesky
    solve loses accuracy in the under-determined (spin-up nullspace) directions;
    **corrected semi-normal equations** restore it by refining with the residual
    evaluated through ``W`` itself rather than ``WᵀW`` (matching the dense
    least-squares solution to ~1e-7 at the default regularization, degrading to
    ~1e-6 only at very small regularization with an ill-conditioned Gram). The
    banded Cholesky factor, solve, and refinement stay at
    ``O(n_output * full_band)``; only the one-shot Gram assembly transiently
    materializes ``W`` and ``WᵀW`` densely.

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
        NaN entries mark measurement gaps; their rows are excluded from the
        normal equations (band row and observed value zeroed).
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
    # Zeroed gapped rows drop out of the normal equations, and a zeroed observed value keeps
    # 0 * NaN out of Wᵀ·observed and the refinement residual.
    nan_obs = np.isnan(observed)
    if nan_obs.any():
        band_vals = np.where(nan_obs[:, None], 0.0, band_vals)
        observed = np.where(nan_obs, 0.0, observed)
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

    # Lower-banded WᵀW via a dense BLAS matmul. Materialize the forward operator W densely
    # (row k is band_vals[k] at columns [col_start[k], col_start[k] + full_band)), form the
    # symmetric Gram matrix WᵀW with a single optimized matmul, then read its lower sub-diagonals
    # into the banded layout (band row d is the d-th sub-diagonal, WᵀW[j + d, j]). Each row's
    # in-range band columns are distinct, so the scatter into W needs no accumulation. This is
    # ~24x the per-diagonal np.add.at scatter; the matmul reorders the summation, so ab matches
    # the scatter to ~1e-13 -- well inside the Tikhonov + refinement tolerance.
    n_obs = band_vals.shape[0]
    w_dense = np.zeros((n_obs, n_cin))
    obs_idx = np.broadcast_to(np.arange(n_obs)[:, None], cols.shape)
    w_dense[obs_idx[in_range], cols_clipped[in_range]] = band_vals[in_range]
    gram = w_dense.T @ w_dense
    ab = np.zeros((full_band, n_cin))
    for d in range(full_band):
        ab[d, : n_cin - d] = np.diagonal(gram, offset=-d)

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
