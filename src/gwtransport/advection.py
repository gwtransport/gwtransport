"""
Advective Transport Modeling for 1D Aquifer Systems.

This module provides functions to model compound transport by advection in one-dimensional
aquifer systems, enabling prediction of solute or temperature concentrations in extracted
water based on infiltration data and aquifer properties. The model assumes one-dimensional
groundwater flow where water infiltrates with concentration ``cin``, flows through the aquifer
with pore volume distribution, compounds are transported with retarded velocity (retardation
factor >= 1.0), and water is extracted with concentration ``cout``.

Available functions:

- :func:`infiltration_to_extraction_series` - Single pore volume, time-shift only. Shifts
  infiltration time edges forward by residence time. Concentration values remain unchanged
  (cout = cin). No support for custom output time edges. Use case: Deterministic transport
  with single flow path.

- :func:`infiltration_to_extraction` - Arbitrary pore volume distribution, convolution.
  Supports explicit distribution of aquifer pore volumes with flow-weighted averaging.
  Flexible output time resolution via cout_tedges. Use case: Known pore volume distribution
  from streamline analysis.

- :func:`gamma_infiltration_to_extraction` - Gamma-distributed pore volumes, convolution.
  Models aquifer heterogeneity with 2-parameter gamma distribution. Parameterizable via
  (alpha, beta) or (mean, std). Discretizes gamma distribution into equal-probability bins.
  Use case: Heterogeneous aquifer with calibrated gamma parameters.

- :func:`extraction_to_infiltration_series` - Single pore volume, time-shift only
  (deconvolution). Shifts extraction time edges backward by residence time. Concentration
  values remain unchanged (cin = cout). Symmetric inverse of infiltration_to_extraction_series.
  Use case: Backward tracing with single flow path.

- :func:`extraction_to_infiltration` - Arbitrary pore volume distribution, deconvolution.
  Inverts forward transport for arbitrary pore volume distributions. Symmetric inverse of
  infiltration_to_extraction. Flow-weighted averaging in reverse direction. Use case:
  Estimating infiltration history from extraction data.

- :func:`gamma_extraction_to_infiltration` - Gamma-distributed pore volumes, deconvolution.
  Inverts forward transport for gamma-distributed pore volumes. Symmetric inverse of
  gamma_infiltration_to_extraction. Use case: Calibrating infiltration conditions from
  extraction measurements.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport import gamma
from gwtransport.residence_time import residence_time
from gwtransport.utils import partial_isin


def infiltration_to_extraction_series(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    retardation_factor: float = 1.0,
) -> pd.DatetimeIndex:
    """
    Compute extraction time edges from infiltration time edges using residence time shifts.

    This function shifts infiltration time edges forward in time based on residence
    times computed from flow rates and aquifer properties. The concentration values remain
    unchanged (cout equals cin), only the time edges are shifted. This assumes a single pore
    volume (no distribution) and deterministic advective transport.

    NOTE: This function is specifically designed for single aquifer pore volumes and does not
    support custom output time edges (cout_tedges). For distributions of aquifer pore volumes
    or custom output time grids, use `infiltration_to_extraction` instead.

    Parameters
    ----------
    flow : array-like
        Flow rate values in the aquifer [m3/day].
        Length must match the number of time bins defined by tedges (len(tedges) - 1).
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cin and flow data. Has length of len(flow) + 1.
    aquifer_pore_volume : float
        Single aquifer pore volume [m3] used to compute residence times.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.

    Returns
    -------
    pandas.DatetimeIndex
        Time edges for the extracted water concentration. Same length as tedges.
        The concentration values in the extracted water (cout) equal cin, but are
        aligned with these shifted time edges.

    Examples
    --------
    Basic usage with constant flow:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import infiltration_to_extraction_series
    >>>
    >>> # Create input data
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Constant concentration and flow
    >>> cin = np.ones(len(dates)) * 10.0
    >>> flow = np.ones(len(dates)) * 100.0  # 100 m3/day
    >>>
    >>> # Run infiltration_to_extraction_series with 500 m3 pore volume
    >>> tedges_out = infiltration_to_extraction_series(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     aquifer_pore_volume=500.0,
    ... )
    >>> len(tedges_out)
    11
    >>> # Time shift should be residence time = pore_volume / flow = 500 / 100 = 5 days
    >>> tedges_out[0] - tedges[0]
    Timedelta('5 days 00:00:00')

    Plotting the input and output concentrations:

    >>> import matplotlib.pyplot as plt
    >>> # Prepare data for step plot (repeat values for visualization)
    >>> xplot_in = np.repeat(tedges, 2)[1:-1]
    >>> yplot_in = np.repeat(cin, 2)
    >>> plt.plot(
    ...     xplot_in, yplot_in, label="Concentration of infiltrated water"
    ... )  # doctest: +SKIP
    >>>
    >>> # cout equals cin, just with shifted time edges
    >>> xplot_out = np.repeat(tedges_out, 2)[1:-1]
    >>> yplot_out = np.repeat(cin, 2)
    >>> plt.plot(
    ...     xplot_out, yplot_out, label="Concentration of extracted water"
    ... )  # doctest: +SKIP
    >>> plt.xlabel("Time")  # doctest: +SKIP
    >>> plt.ylabel("Concentration")  # doctest: +SKIP
    >>> plt.legend()  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    With retardation factor:

    >>> tedges_out = infiltration_to_extraction_series(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     aquifer_pore_volume=500.0,
    ...     retardation_factor=2.0,  # Doubles residence time
    ... )
    >>> # Time shift is doubled: 500 * 2.0 / 100 = 10 days
    >>> tedges_out[0] - tedges[0]
    Timedelta('10 days 00:00:00')
    """
    rt_array = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",
    )
    return tedges + pd.to_timedelta(rt_array[0], unit="D", errors="coerce")


def extraction_to_infiltration_series(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    retardation_factor: float = 1.0,
) -> pd.DatetimeIndex:
    """
    Compute infiltration time edges from extraction time edges (deconvolution).

    This function shifts extraction time edges backward in time based on residence
    times computed from flow rates and aquifer properties. The concentration values remain
    unchanged (cin equals cout), only the time edges are shifted. This assumes a single pore
    volume (no distribution) and deterministic advective transport. This is the inverse
    operation of infiltration_to_extraction_series.

    NOTE: This function is specifically designed for single aquifer pore volumes and does not
    support custom output time edges (cin_tedges). For distributions of aquifer pore volumes
    or custom output time grids, use `extraction_to_infiltration` instead.

    Parameters
    ----------
    flow : array-like
        Flow rate values in the aquifer [m3/day].
        Length must match the number of time bins defined by tedges (len(tedges) - 1).
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cout and flow data. Has length of len(flow) + 1.
    aquifer_pore_volume : float
        Single aquifer pore volume [m3] used to compute residence times.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.

    Returns
    -------
    pandas.DatetimeIndex
        Time edges for the infiltrating water concentration. Same length as tedges.
        The concentration values in the infiltrating water (cin) equal cout, but are
        aligned with these shifted time edges.

    Examples
    --------
    Basic usage with constant flow:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import extraction_to_infiltration_series
    >>>
    >>> # Create input data
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Constant concentration and flow
    >>> cout = np.ones(len(dates)) * 10.0
    >>> flow = np.ones(len(dates)) * 100.0  # 100 m3/day
    >>>
    >>> # Run extraction_to_infiltration_series with 500 m3 pore volume
    >>> tedges_out = extraction_to_infiltration_series(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     aquifer_pore_volume=500.0,
    ... )
    >>> len(tedges_out)
    11
    >>> # Time shift should be residence time = pore_volume / flow = 500 / 100 = 5 days (backward)
    >>> # First few elements are NaT due to insufficient history, check a valid index
    >>> tedges[5] - tedges_out[5]
    Timedelta('5 days 00:00:00')

    Plotting the input and output concentrations:

    >>> import matplotlib.pyplot as plt
    >>> # Prepare data for step plot (repeat values for visualization)
    >>> xplot_in = np.repeat(tedges, 2)[1:-1]
    >>> yplot_in = np.repeat(cout, 2)
    >>> plt.plot(
    ...     xplot_in, yplot_in, label="Concentration of extracted water"
    ... )  # doctest: +SKIP
    >>>
    >>> # cin equals cout, just with shifted time edges
    >>> xplot_out = np.repeat(tedges_out, 2)[1:-1]
    >>> yplot_out = np.repeat(cout, 2)
    >>> plt.plot(
    ...     xplot_out, yplot_out, label="Concentration of infiltrated water"
    ... )  # doctest: +SKIP
    >>> plt.xlabel("Time")  # doctest: +SKIP
    >>> plt.ylabel("Concentration")  # doctest: +SKIP
    >>> plt.legend()  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    With retardation factor:

    >>> tedges_out = extraction_to_infiltration_series(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     aquifer_pore_volume=500.0,
    ...     retardation_factor=2.0,  # Doubles residence time
    ... )
    >>> # Time shift is doubled: 500 * 2.0 / 100 = 10 days (backward)
    >>> # With longer residence time, more elements are NaT, check the last valid index
    >>> tedges[10] - tedges_out[10]
    Timedelta('10 days 00:00:00')
    """
    rt_array = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    return tedges - pd.to_timedelta(rt_array[0], unit="D", errors="coerce")


def gamma_infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer. The aquifer pore volume is approximated by a gamma distribution, with
    parameters alpha and beta.

    This function represents infiltration to extraction modeling (equivalent to convolution).

    Provide either alpha and beta or mean and std.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water or temperature of infiltrating
        water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for both cin and flow data. Used to compute the cumulative concentration.
        Has a length of one more than `cin` and `flow`.
    cout_tedges : pandas.DatetimeIndex
        Time edges for the output data. Used to compute the cumulative concentration.
        Has a length of one more than the desired output length.
    alpha : float, optional
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0)
    beta : float, optional
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0)
    mean : float, optional
        Mean of the gamma distribution.
    std : float, optional
        Standard deviation of the gamma distribution.
    n_bins : int
        Number of bins to discretize the gamma distribution.
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    numpy.ndarray
        Concentration of the compound in the extracted water [ng/m3] or temperature.

    See Also
    --------
    infiltration_to_extraction : Transport with explicit pore volume distribution
    gamma_extraction_to_infiltration : Reverse operation (deconvolution)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.residence_time.residence_time : Compute residence times

    Examples
    --------
    Basic usage with alpha and beta parameters:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import gamma_infiltration_to_extraction
    >>>
    >>> # Create input data with aligned time edges
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Create output time edges (can be different alignment)
    >>> cout_dates = pd.date_range(start="2020-01-05", end="2020-01-15", freq="D")
    >>> cout_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
    ... )
    >>>
    >>> # Input concentration and flow (same length, aligned with tedges)
    >>> cin = pd.Series(np.ones(len(dates)), index=dates)
    >>> flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # 100 m3/day
    >>>
    >>> # Run gamma_infiltration_to_extraction with alpha/beta parameters
    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     n_bins=5,
    ... )
    >>> cout.shape
    (11,)

    Using mean and std parameters instead:

    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=100.0,
    ...     std=20.0,
    ...     n_bins=5,
    ... )

    With retardation factor:

    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     retardation_factor=2.0,  # Doubles residence time
    ... )
    """
    bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        retardation_factor=retardation_factor,
    )


def gamma_extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cin_tedges: pd.DatetimeIndex,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Compute the concentration of the infiltrating water from extracted water (deconvolution).

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer. The aquifer pore volume is approximated by a gamma distribution, with
    parameters alpha and beta.

    This function represents extraction to infiltration modeling (equivalent to deconvolution).
    It is symmetric to gamma_infiltration_to_extraction.

    Provide either alpha and beta or mean and std.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water or temperature of extracted
        water.
    tedges : pandas.DatetimeIndex
        Time edges for the cout and flow data. Used to compute the cumulative concentration.
        Has a length of one more than `cout` and `flow`.
    cin_tedges : pandas.DatetimeIndex
        Time edges for the output (infiltration) data. Used to compute the cumulative concentration.
        Has a length of one more than the desired output length.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    alpha : float, optional
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0)
    beta : float, optional
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0)
    mean : float, optional
        Mean of the gamma distribution.
    std : float, optional
        Standard deviation of the gamma distribution.
    n_bins : int
        Number of bins to discretize the gamma distribution.
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    numpy.ndarray
        Concentration of the compound in the infiltrating water [ng/m3] or temperature.

    Examples
    --------
    Basic usage with alpha and beta parameters:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import gamma_extraction_to_infiltration
    >>>
    >>> # Create input data with aligned time edges
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Create output time edges (can be different alignment)
    >>> cin_dates = pd.date_range(start="2019-12-25", end="2020-01-15", freq="D")
    >>> cin_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates)
    ... )
    >>>
    >>> # Input concentration and flow (same length, aligned with tedges)
    >>> cout = pd.Series(np.ones(len(dates)), index=dates)
    >>> flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # 100 m3/day
    >>>
    >>> # Run gamma_extraction_to_infiltration with alpha/beta parameters
    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     tedges=tedges,
    ...     cin_tedges=cin_tedges,
    ...     flow=flow,
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     n_bins=5,
    ... )
    >>> cin.shape
    (22,)

    Using mean and std parameters instead:

    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     tedges=tedges,
    ...     cin_tedges=cin_tedges,
    ...     flow=flow,
    ...     mean=100.0,
    ...     std=20.0,
    ...     n_bins=5,
    ... )

    With retardation factor:

    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     tedges=tedges,
    ...     cin_tedges=cin_tedges,
    ...     flow=flow,
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     retardation_factor=2.0,  # Doubles residence time
    ... )
    """
    bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        retardation_factor=retardation_factor,
    )


def _infiltration_to_extraction_nonlinear_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute normalized weights for nonlinear infiltration to extraction transformation.

    This helper function computes the weight matrix for concentration-dependent retardation.
    The algorithm tracks infiltration parcels forward in time using concentration-dependent
    retardation factors, computes temporal overlaps with extraction bins, and constructs
    flow-weighted transformation matrix.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for infiltration bins.
    cout_tedges : pandas.DatetimeIndex
        Time edges for extraction bins.
    aquifer_pore_volumes : array-like
        Distribution of pore volumes [m3].
    cin : array-like
        Concentration values (needed for dimensions).
    flow : array-like
        Flow rate values [m3/day].
    retardation_factors : array-like
        Concentration-dependent retardation factors.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix. Shape: (len(cout_tedges) - 1, len(cin))
    """
    # Convert to days
    cin_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # FORWARD PARCEL TRACKING with shock handling
    # Key insight: track infiltration BINS (parcels) not edges
    # Each parcel has concentration, mass, and arrival time
    # When parcels overlap at extraction → mix (captures shocks naturally)

    n_cin_bins = len(cin)
    n_cout_bins = len(cout_tedges) - 1
    n_pv = len(aquifer_pore_volumes)

    # NOTE: For forward tracking, cout_tedges should extend beyond tedges
    # to capture delayed arrivals due to retardation.
    # Rule of thumb: extend cout_tedges by max_residence_time
    # where max_residence_time ≈ max(pore_volumes) * max(retardation_factors) / min(flow)

    # Accumulate weights from all pore volumes
    accumulated_weights = np.zeros((n_cout_bins, n_cin_bins))
    pv_count = np.zeros(n_cout_bins)

    # For each pore volume (aquifer heterogeneity)
    for i_pv in range(n_pv):
        pv = aquifer_pore_volumes[i_pv]

        # STEP 1: Compute residence times for all edges
        # Strategy: Call residence_time once per edge with its retardation
        rt_bin_edges = np.zeros(n_cin_bins + 1)

        # Interpolate retardation to edges (vectorized)
        # NOTE: Edge interpolation for concentration-dependent retardation
        #
        # Physical principle: R = f(C), so we should:
        #   1. Interpolate concentration to edges
        #   2. Compute retardation from edge concentration
        #
        # This is more accurate than directly averaging R values, especially
        # for nonlinear isotherms where R(C_avg) ≠ avg(R(C)).
        #
        # For sharp concentration steps, this still has limitations (O(Δt) error),
        # but is significantly better than arithmetic averaging of R.
        #
        # Compute retardation at edges
        # NOTE: Ideally we would interpolate concentration first, then compute R(C_edge).
        # However, that requires knowing the isotherm parameters, which aren't passed to
        # this function. For now, we use arithmetic averaging of R values.
        #
        # TODO: Consider adding optional isotherm parameters to enable proper C interpolation.
        # This would be more accurate: R_edge = f(0.5*(C[i-1] + C[i]))
        # vs current: R_edge = 0.5*(R[i-1] + R[i])
        #
        # For smooth concentration variations, the error is O(Δt²).
        # For sharp steps, this can introduce significant errors.
        r_at_edges = np.zeros(n_cin_bins + 1)
        r_at_edges[0] = retardation_factors[0]
        r_at_edges[-1] = retardation_factors[-1]
        r_at_edges[1:-1] = 0.5 * (retardation_factors[:-1] + retardation_factors[1:])

        for i_edge in range(n_cin_bins + 1):
            rt_val = residence_time(
                flow=flow,
                flow_tedges=tedges,
                index=tedges[i_edge : i_edge + 1],
                aquifer_pore_volume=np.array([pv]),
                retardation_factor=float(r_at_edges[i_edge]),
                direction="infiltration_to_extraction",
            )
            rt_bin_edges[i_edge] = rt_val[0, 0]

        # Compute extraction times for all edges
        extraction_times = cin_tedges_days + rt_bin_edges

        # STEP 2: VECTORIZED temporal overlap computation
        # Shape: [n_cin_bins, n_cout_bins]
        # Extract parcel boundaries
        t_out_starts = extraction_times[:-1]  # [n_cin_bins]
        t_out_ends = extraction_times[1:]  # [n_cin_bins]

        # Extract cout bin boundaries
        t_cout_starts = cout_tedges_days[:-1]  # [n_cout_bins]
        t_cout_ends = cout_tedges_days[1:]  # [n_cout_bins]

        # Broadcast to compute all overlaps at once
        # overlap_start[i, j] = max(t_out_start[i], t_cout_start[j])
        overlap_starts = np.maximum(t_out_starts[:, None], t_cout_starts[None, :])  # [n_cin, n_cout]
        overlap_ends = np.minimum(t_out_ends[:, None], t_cout_ends[None, :])  # [n_cin, n_cout]
        overlap_durations = np.maximum(0, overlap_ends - overlap_starts)  # [n_cin, n_cout]

        # Compute parcel durations
        parcel_durations = t_out_ends - t_out_starts  # [n_cin_bins]

        # Compute fractions: fraction[i, j] = overlap_duration[i, j] / parcel_duration[i]
        # Handle division by zero
        fractions = np.zeros_like(overlap_durations)
        valid_parcels = parcel_durations > 0
        fractions[valid_parcels, :] = overlap_durations[valid_parcels, :] / parcel_durations[valid_parcels, None]

        # For instantaneous parcels (duration == 0), assign to bin that contains start time
        instantaneous = ~valid_parcels
        if np.any(instantaneous):
            # Check if t_out_start falls within cout bin
            inst_contained = (t_out_starts[instantaneous, None] >= t_cout_starts[None, :]) & (
                t_out_starts[instantaneous, None] < t_cout_ends[None, :]
            )
            fractions[instantaneous, :] = inst_contained.astype(float)

        # Zero out fractions where overlap_duration is 0 (no overlap)
        fractions[overlap_durations == 0] = 0

        # Compute flow-weighted fractions
        # flow_weighted_fractions[i, j] = flow[i] * fraction[i, j]
        # Shape: [n_cin_bins, n_cout_bins]
        flow_weighted_fractions = flow[:, None] * fractions  # [n_cin, n_cout]

        # Transpose to get [n_cout_bins, n_cin_bins] for accumulation
        flow_weighted_fractions_t = flow_weighted_fractions.T  # [n_cout, n_cin]

        # Accumulate weights across pore volumes
        accumulated_weights += flow_weighted_fractions_t

        # Track which cout bins have contributions from this pore volume
        has_contribution = np.sum(fractions, axis=0) > 0  # [n_cout_bins]
        pv_count[has_contribution] += 1

    # Average across valid pore volumes
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cout = pv_count > 0
    averaged_weights[valid_cout, :] = accumulated_weights[valid_cout, :] / pv_count[valid_cout, None]

    # Normalize by total weights per output bin
    total_weights = np.sum(averaged_weights, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(averaged_weights)
    normalized_weights[valid_weights, :] = averaged_weights[valid_weights, :] / total_weights[valid_weights, None]

    return normalized_weights


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    retardation_factor: float | npt.ArrayLike = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Compute the concentration of the extracted water using flow-weighted advection.

    This function implements an infiltration to extraction advection model where cin and flow values
    correspond to the same aligned time bins defined by tedges.

    The algorithm:
    1. Computes residence times for each pore volume at cout time edges
    2. Calculates infiltration time edges by subtracting residence times
    3. Determines temporal overlaps between infiltration and cin time windows
    4. Creates flow-weighted overlap matrices normalized by total weights
    5. Computes weighted contributions and averages across pore volumes

    .. note:: **Nonlinear Sorption and Concentration-Dependent Retardation**

       When ``retardation_factor`` is provided as an array (one value per time bin), this function
       models **nonlinear sorption** where retardation depends on concentration. This is critical
       for contaminants that follow Freundlich or Langmuir isotherms.

       **Physical Behavior:**

       For Freundlich sorption with n < 1 (favorable sorption):

       - High concentrations → Lower retardation → Faster travel
       - Low concentrations → Higher retardation → Slower travel
       - Creates asymmetric breakthrough: **sharp front, long tail**

       **Implementation:**

       Uses forward method of characteristics where each infiltrating parcel travels with its own
       retardation factor R(C). When faster-moving high-C parcels overtake slower low-C parcels,
       they mix at extraction through flow-weighted averaging, naturally capturing shock formation
       without spurious oscillations.

       See Example 6 (Freundlich Sorption) for detailed demonstration.

    Parameters
    ----------
    cin : array-like
        Concentration values of infiltrating water or temperature [concentration units].
        Length must match the number of time bins defined by tedges.
    flow : array-like
        Flow rate values in the aquifer [m3/day].
        Length must match cin and the number of time bins defined by tedges.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cin and flow data. Has length of
        len(cin) + 1 and len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
        Can have different time alignment and resolution than tedges.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution
        of residence times in the aquifer system.
    retardation_factor : float or array-like, optional
        Retardation factor of the compound in the aquifer (default 1.0).

        - **Scalar (float)**: Linear sorption with constant retardation.
          Values > 1.0 indicate slower transport due to sorption/interaction.

        - **Array**: Nonlinear sorption with concentration-dependent retardation.
          Must have length matching cin. Typically computed from Freundlich or
          Langmuir isotherms using :func:`gwtransport.residence_time.freundlich_retardation`.

    Returns
    -------
    numpy.ndarray
        Flow-weighted concentration in the extracted water. Same units as cin.
        Length equals len(cout_tedges) - 1. NaN values indicate time periods
        with no valid contributions from the infiltration data.

    Raises
    ------
    ValueError
        If tedges length doesn't match cin/flow arrays plus one, or if
        infiltration time edges become non-monotonic (invalid input conditions),
        or if retardation_factor array length doesn't match cin.

    See Also
    --------
    gamma_infiltration_to_extraction : Transport with gamma-distributed pore volumes
    extraction_to_infiltration : Reverse operation (deconvolution)
    infiltration_to_extraction_series : Simple time-shift for single pore volume
    gwtransport.residence_time.residence_time : Compute residence times from flow and pore volume
    gwtransport.residence_time.freundlich_retardation : Compute concentration-dependent retardation

    Examples
    --------
    Basic usage with pandas Series:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import infiltration_to_extraction
    >>>
    >>> # Create input data
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Create output time edges (different alignment)
    >>> cout_dates = pd.date_range(start="2020-01-05", end="2020-01-15", freq="D")
    >>> cout_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
    ... )
    >>>
    >>> # Input concentration and flow
    >>> cin = pd.Series(np.ones(len(dates)), index=dates)
    >>> flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # 100 m3/day
    >>>
    >>> # Define distribution of aquifer pore volumes
    >>> aquifer_pore_volumes = np.array([50, 100, 200])  # m3
    >>>
    >>> # Run infiltration_to_extraction
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )
    >>> cout.shape
    (11,)

    Using array inputs instead of pandas Series:

    >>> # Convert to arrays
    >>> cin_values = cin.values
    >>> flow_values = flow.values
    >>>
    >>> cout = infiltration_to_extraction(
    ...     cin=cin_values,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )

    With constant retardation factor (linear sorption):

    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     retardation_factor=2.0,  # Compound moves twice as slowly
    ... )

    With concentration-dependent retardation (nonlinear sorption):

    >>> from gwtransport.residence_time import freundlich_retardation
    >>> # Gaussian concentration pulse
    >>> cin_nonlinear = 100.0 * np.exp(-0.5 * ((np.arange(len(dates)) - 10) / 5) ** 2)
    >>> # Compute Freundlich retardation
    >>> R_freundlich = freundlich_retardation(
    ...     concentration=np.maximum(cin_nonlinear, 0.1),
    ...     freundlich_k=0.02,
    ...     freundlich_n=0.75,
    ...     bulk_density=1600.0,
    ...     porosity=0.35,
    ... )
    >>> cout_nonlinear = infiltration_to_extraction(
    ...     cin=cin_nonlinear,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     retardation_factor=R_freundlich,  # Array enables nonlinear mode
    ... )
    >>> # Result: asymmetric breakthrough (sharp front, long tail)

    Using single pore volume:

    >>> single_volume = np.array([100])  # Single 100 m3 pore volume
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=single_volume,
    ... )
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    # Convert to arrays for vectorized operations
    cin = np.asarray(cin)
    flow = np.asarray(flow)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes)

    if len(tedges) != len(cin) + 1:
        msg = "tedges must have one more element than cin"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    # Validate inputs do not contain NaN values
    if np.any(np.isnan(cin)):
        msg = "cin contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)

    # Check if retardation_factor is array-like (nonlinear case)
    is_linear = isinstance(retardation_factor, (float, int))

    if is_linear:
        # Linear sorption: constant retardation
        # Compute normalized weights (includes all pre-computation)
        normalized_weights = _infiltration_to_extraction_weights(
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            cin=cin,
            flow=flow,
            retardation_factor=retardation_factor,
        )
    else:
        # Nonlinear sorption: concentration-dependent retardation
        retardation_factor = np.asarray(retardation_factor)

        if len(retardation_factor) != len(cin):
            msg = f"retardation_factor array must match cin length ({len(cin)}), got {len(retardation_factor)}"
            raise ValueError(msg)

        # Use nonlinear weights computation
        normalized_weights = _infiltration_to_extraction_nonlinear_weights(
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            cin=cin,
            flow=flow,
            retardation_factors=retardation_factor,
        )

    # Apply to concentrations and handle NaN for periods with no contributions
    out = normalized_weights.dot(cin)
    # Set NaN where no valid pore volumes contributed
    total_weights = np.sum(normalized_weights, axis=1)
    out[total_weights == 0] = np.nan

    return out


def _infiltration_to_extraction_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float | npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    # Convert time edges to days
    cin_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # Pre-compute all residence times and infiltration edges
    rt_edges_2d = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    infiltration_tedges_2d = cout_tedges_days[None, :] - rt_edges_2d

    # Pre-compute valid bins and count
    valid_bins_2d = ~(np.isnan(infiltration_tedges_2d[:, :-1]) | np.isnan(infiltration_tedges_2d[:, 1:]))
    valid_pv_count = np.sum(valid_bins_2d, axis=0)

    accumulated_weights = np.zeros((len(cout_tedges) - 1, len(cin)))

    # Pre-compute cin time range for clip optimization (computed once, used n_bins times)
    cin_time_min = cin_tedges_days[0]
    cin_time_max = cin_tedges_days[-1]

    # Loop over each pore volume
    for i in range(len(aquifer_pore_volumes)):
        if not np.any(valid_bins_2d[i, :]):
            continue

        # Clip optimization: Check for temporal overlap before expensive computation
        # Get the range of infiltration times for this pore volume (only valid bins)
        infiltration_times = infiltration_tedges_2d[i, :]
        valid_infiltration_times = infiltration_times[~np.isnan(infiltration_times)]

        if len(valid_infiltration_times) == 0:
            continue

        infiltration_min = valid_infiltration_times[0]  # Min is first element (monotonic)
        infiltration_max = valid_infiltration_times[-1]  # Max is last element (monotonic)

        # Check if infiltration window overlaps with cin window
        # Two intervals [a1, a2] and [b1, b2] overlap if: max(a1, b1) < min(a2, b2)
        has_overlap = max(infiltration_min, cin_time_min) < min(infiltration_max, cin_time_max)

        if not has_overlap:
            # No temporal overlap - this bin contributes nothing, skip expensive computation
            continue

        # Only compute overlap matrix if there's potential contribution
        overlap_matrix = partial_isin(bin_edges_in=infiltration_tedges_2d[i, :], bin_edges_out=cin_tedges_days)
        accumulated_weights[valid_bins_2d[i, :], :] += overlap_matrix[valid_bins_2d[i, :], :]

    # Average across valid pore volumes and apply flow weighting
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cout = valid_pv_count > 0
    averaged_weights[valid_cout, :] = accumulated_weights[valid_cout, :] / valid_pv_count[valid_cout, None]

    # Apply flow weighting after averaging
    flow_weighted_averaged = averaged_weights * flow[None, :]

    total_weights = np.sum(flow_weighted_averaged, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(flow_weighted_averaged)
    normalized_weights[valid_weights, :] = flow_weighted_averaged[valid_weights, :] / total_weights[valid_weights, None]
    return normalized_weights


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cin_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Compute the concentration of the infiltrating water from extracted water (deconvolution).

    This function implements an extraction to infiltration advection model (inverse of infiltration_to_extraction)
    where cout and flow values correspond to the same aligned time bins defined by tedges.

    SYMMETRIC RELATIONSHIP:
    - infiltration_to_extraction: cin + tedges → cout + cout_tedges
    - extraction_to_infiltration: cout + tedges → cin + cin_tedges

    The algorithm (symmetric to infiltration_to_extraction):
    1. Computes residence times for each pore volume at cint time edges
    2. Calculates extraction time edges by adding residence times (reverse of infiltration_to_extraction)
    3. Determines temporal overlaps between extraction and cout time windows
    4. Creates flow-weighted overlap matrices normalized by total weights
    5. Computes weighted contributions and averages across pore volumes

    Parameters
    ----------
    cout : array-like
        Concentration values of extracted water [concentration units].
        Length must match the number of time bins defined by tedges.
    flow : array-like
        Flow rate values in the aquifer [m3/day].
        Length must match cout and the number of time bins defined by tedges.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cout and flow data. Has length of
        len(cout) + 1 and len(flow) + 1.
    cin_tedges : pandas.DatetimeIndex
        Time edges for output (infiltration) data bins. Has length of desired output + 1.
        Can have different time alignment and resolution than tedges.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution
        of residence times in the aquifer system.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.

    Returns
    -------
    numpy.ndarray
        Flow-weighted concentration in the infiltrating water. Same units as cout.
        Length equals len(cin_tedges) - 1. NaN values indicate time periods
        with no valid contributions from the extraction data.

    Raises
    ------
    ValueError
        If tedges length doesn't match cout/flow arrays plus one, or if
        extraction time edges become non-monotonic (invalid input conditions).

    Examples
    --------
    Basic usage with pandas Series:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import extraction_to_infiltration
    >>>
    >>> # Create input data
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Create output time edges (different alignment)
    >>> cint_dates = pd.date_range(start="2019-12-25", end="2020-01-15", freq="D")
    >>> cin_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates)
    ... )
    >>>
    >>> # Input concentration and flow
    >>> cout = pd.Series(np.ones(len(dates)), index=dates)
    >>> flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # 100 m3/day
    >>>
    >>> # Define distribution of aquifer pore volumes
    >>> aquifer_pore_volumes = np.array([50, 100, 200])  # m3
    >>>
    >>> # Run extraction_to_infiltration
    >>> cin = extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cin_tedges=cin_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )
    >>> cin.shape
    (22,)

    Using array inputs instead of pandas Series:

    >>> # Convert to arrays
    >>> cout = cout.values
    >>> flow = flow.values
    >>>
    >>> cin = extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cin_tedges=cin_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )

    With retardation factor:

    >>> cin = extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cin_tedges=cin_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     retardation_factor=2.0,  # Compound moves twice as slowly
    ... )

    Using single pore volume:

    >>> single_volume = np.array([100])  # Single 100 m3 pore volume
    >>> cin = extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cin_tedges=cin_tedges,
    ...     aquifer_pore_volumes=single_volume,
    ... )
    """
    tedges = pd.DatetimeIndex(tedges)
    cin_tedges = pd.DatetimeIndex(cin_tedges)

    # Convert to arrays for vectorized operations
    cout = np.asarray(cout)
    flow = np.asarray(flow)

    if len(tedges) != len(cout) + 1:
        msg = "tedges must have one more element than cout"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    # Validate inputs do not contain NaN values
    if np.any(np.isnan(cout)):
        msg = "cout contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)

    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes)

    # Compute normalized weights (includes all pre-computation)
    normalized_weights = _extraction_to_infiltration_weights(
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        cout=cout,
        flow=flow,
        retardation_factor=retardation_factor,
    )

    # Apply to concentrations and handle NaN for periods with no contributions
    out = normalized_weights.dot(cout)
    # Set NaN where no valid pore volumes contributed
    total_weights = np.sum(normalized_weights, axis=1)
    out[total_weights == 0] = np.nan

    return out


def _extraction_to_infiltration_weights(
    *,
    tedges: pd.DatetimeIndex,
    cin_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cout: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.floating]:
    """
    Compute extraction to infiltration transformation weights matrix.

    Computes the weight matrix for the extraction to infiltration transformation,
    ensuring mathematical symmetry with the infiltration to extraction operation. The extraction to infiltration
    weights represent the transpose relationship needed for deconvolution.

    SYMMETRIC RELATIONSHIP:
    - Infiltration to extraction weights: W_infiltration_to_extraction maps cin → cout
    - Extraction to infiltration weights: W_extraction_to_infiltration maps cout → cin
    - Mathematical constraint: W_extraction_to_infiltration should be the pseudo-inverse of W_infiltration_to_extraction

    The algorithm mirrors _infiltration_to_extraction_weights but with transposed
    temporal overlap computations to ensure mathematical consistency.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for cout and flow data bins.
    cin_tedges : pandas.DatetimeIndex
        Time edges for output (infiltration) data bins.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3].
    cout : array-like
        Concentration values of extracted water.
    flow : array-like
        Flow rate values in the aquifer [m3/day].
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix for extraction to infiltration transformation.
        Shape: (len(cin_tedges) - 1, len(cout))
    """
    # Convert time edges to days
    cout_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cin_tedges_days = ((cin_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # Pre-compute all residence times and extraction edges (symmetric to infiltration_to_extraction)
    rt_edges_2d = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cin_tedges,
        aquifer_pore_volume=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",  # Computing from infiltration perspective
    )
    extraction_tedges_2d = cin_tedges_days[None, :] + rt_edges_2d

    # Pre-compute valid bins and count
    valid_bins_2d = ~(np.isnan(extraction_tedges_2d[:, :-1]) | np.isnan(extraction_tedges_2d[:, 1:]))
    valid_pv_count = np.sum(valid_bins_2d, axis=0)

    accumulated_weights = np.zeros((len(cin_tedges) - 1, len(cout)))

    # Pre-compute cout time range for clip optimization (computed once, used n_bins times)
    cout_time_min = cout_tedges_days[0]
    cout_time_max = cout_tedges_days[-1]

    # Loop over each pore volume (same structure as infiltration_to_extraction)
    for i in range(len(aquifer_pore_volumes)):
        if not np.any(valid_bins_2d[i, :]):
            continue

        # Clip optimization: Check for temporal overlap before expensive computation
        # Get the range of extraction times for this pore volume (only valid bins)
        extraction_times = extraction_tedges_2d[i, :]
        valid_extraction_times = extraction_times[~np.isnan(extraction_times)]

        if len(valid_extraction_times) == 0:
            continue

        extraction_min = valid_extraction_times[0]  # Min is first element (monotonic)
        extraction_max = valid_extraction_times[-1]  # Max is last element (monotonic)

        # Check if extraction window overlaps with cout window
        # Two intervals [a1, a2] and [b1, b2] overlap if: max(a1, b1) < min(a2, b2)
        has_overlap = max(extraction_min, cout_time_min) < min(extraction_max, cout_time_max)

        if not has_overlap:
            # No temporal overlap - this bin contributes nothing, skip expensive computation
            continue

        # SYMMETRIC temporal overlap computation:
        # Infiltration to extraction: maps infiltration → cout time windows
        # Extraction to infiltration: maps extraction → cout time windows (transposed relationship)
        overlap_matrix = partial_isin(bin_edges_in=extraction_tedges_2d[i, :], bin_edges_out=cout_tedges_days)
        accumulated_weights[valid_bins_2d[i, :], :] += overlap_matrix[valid_bins_2d[i, :], :]

    # Average across valid pore volumes (symmetric to infiltration_to_extraction)
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cout = valid_pv_count > 0
    averaged_weights[valid_cout, :] = accumulated_weights[valid_cout, :] / valid_pv_count[valid_cout, None]

    # Apply flow weighting (symmetric to infiltration_to_extraction)
    flow_weighted_averaged = averaged_weights * flow[None, :]

    # Normalize by total weights (symmetric to infiltration_to_extraction)
    total_weights = np.sum(flow_weighted_averaged, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(flow_weighted_averaged)
    normalized_weights[valid_weights, :] = flow_weighted_averaged[valid_weights, :] / total_weights[valid_weights, None]

    return normalized_weights
