"""
Advection Analysis for 1D Aquifer Systems.

This module provides functions to analyze compound transport by advection
in aquifer systems. It includes tools for computing concentrations of the extracted water
based on the concentration of the infiltrating water, extraction data and aquifer properties.

The model assumes requires the groundwaterflow to be reduced to a 1D system. On one side,
water with a certain concentration infiltrates ('cin'), the water flows through the aquifer and
the compound of interest flows through the aquifer with a retarded velocity. The water is
extracted ('cout').

Main functions:
- infiltration_to_extraction: Compute the concentration of the extracted water by shifting cin with its residence time. This corresponds to a convolution operation.
- gamma_infiltration_to_extraction: Similar to infiltration_to_extraction, but for a gamma distribution of aquifer pore volumes.
- gamma_extraction_to_infiltration: Symmetric to gamma_infiltration_to_extraction, computing infiltration concentration from extraction data (deconvolution).
- infiltration_to_extraction: Similar to infiltration_to_extraction_series, but for an arbitrary distribution of aquifer pore volumes.
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
    >>> plt.plot(xplot_in, yplot_in, label="Concentration of infiltrated water")
    >>>
    >>> # cout equals cin, just with shifted time edges
    >>> xplot_out = np.repeat(tedges_out, 2)[1:-1]
    >>> yplot_out = np.repeat(cin, 2)
    >>> plt.plot(xplot_out, yplot_out, label="Concentration of extracted water")
    >>> plt.xlabel("Time")
    >>> plt.ylabel("Concentration")
    >>> plt.legend()
    >>> plt.show()

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
    >>> tedges[0] - tedges_out[0]
    Timedelta('5 days 00:00:00')

    Plotting the input and output concentrations:

    >>> import matplotlib.pyplot as plt
    >>> # Prepare data for step plot (repeat values for visualization)
    >>> xplot_in = np.repeat(tedges, 2)[1:-1]
    >>> yplot_in = np.repeat(cout, 2)
    >>> plt.plot(xplot_in, yplot_in, label="Concentration of extracted water")
    >>>
    >>> # cin equals cout, just with shifted time edges
    >>> xplot_out = np.repeat(tedges_out, 2)[1:-1]
    >>> yplot_out = np.repeat(cout, 2)
    >>> plt.plot(xplot_out, yplot_out, label="Concentration of infiltrated water")
    >>> plt.xlabel("Time")
    >>> plt.ylabel("Concentration")
    >>> plt.legend()
    >>> plt.show()

    With retardation factor:

    >>> tedges_out = extraction_to_infiltration_series(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     aquifer_pore_volume=500.0,
    ...     retardation_factor=2.0,  # Doubles residence time
    ... )
    >>> # Time shift is doubled: 500 * 2.0 / 100 = 10 days (backward)
    >>> tedges[0] - tedges_out[0]
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
    cin_tedges : pandas.DatetimeIndex
        Time edges for the concentration data. Used to compute the cumulative concentration.
        Has a length of one more than `cin`.
    cout_tedges : pandas.DatetimeIndex
        Time edges for the output data. Used to compute the cumulative concentration.
        Has a length of one more than `flow`.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    flow_tedges : pandas.DatetimeIndex
        Time edges for the flow data. Used to compute the cumulative flow.
        Has a length of one more than `flow`.
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
    ...     cin_tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     flow=flow,
    ...     flow_tedges=tedges,  # Must be identical to cin_tedges
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     n_bins=5,
    ... )
    >>> cout.shape
    (11,)

    Using mean and std parameters instead:

    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     cin_tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     flow=flow,
    ...     flow_tedges=tedges,
    ...     mean=100.0,
    ...     std=20.0,
    ...     n_bins=5,
    ... )

    With retardation factor:

    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     cin_tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     flow=flow,
    ...     flow_tedges=tedges,
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


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    retardation_factor: float = 1.0,
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
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.

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
        infiltration time edges become non-monotonic (invalid input conditions).

    See Also
    --------
    gamma_infiltration_to_extraction : Transport with gamma-distributed pore volumes
    extraction_to_infiltration : Reverse operation (deconvolution)
    infiltration_to_extraction_series : Simple time-shift for single pore volume
    gwtransport.residence_time.residence_time : Compute residence times from flow and pore volume

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
    ...     flow=flow_values,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )

    With retardation factor:

    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     retardation_factor=2.0,  # Compound moves twice as slowly
    ... )

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

    if len(tedges) != len(cin) + 1:
        msg = "tedges must have one more element than cin"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    # Convert to arrays for vectorized operations
    cin_values = np.asarray(cin)
    flow_values = np.asarray(flow)

    # Validate inputs do not contain NaN values
    if np.any(np.isnan(cin_values)):
        msg = "cin contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow_values)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)
    cin_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes)

    # Pre-compute all residence times and infiltration edges
    rt_edges_2d = residence_time(
        flow=flow_values,
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

    # Initialize accumulator
    normalized_weights = _infiltration_to_extraction_weights(
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        cin_values=cin_values,
        flow_values=flow_values,
        cin_tedges_days=cin_tedges_days,
        infiltration_tedges_2d=infiltration_tedges_2d,
        valid_bins_2d=valid_bins_2d,
        valid_pv_count=valid_pv_count,
    )

    # Apply to concentrations and handle NaN for periods with no contributions
    out = normalized_weights.dot(cin_values)
    out[valid_pv_count == 0] = np.nan

    return out


def _infiltration_to_extraction_weights(
    *,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cin_values: npt.NDArray[np.floating],
    flow_values: npt.NDArray[np.floating],
    cin_tedges_days: npt.NDArray[np.floating],
    infiltration_tedges_2d: npt.NDArray[np.floating],
    valid_bins_2d: npt.NDArray[np.bool_],
    valid_pv_count: npt.NDArray[np.intp],
) -> npt.NDArray[np.floating]:
    accumulated_weights = np.zeros((len(cout_tedges) - 1, len(cin_values)))

    # Loop over each pore volume
    for i in range(len(aquifer_pore_volumes)):
        if np.any(valid_bins_2d[i, :]):
            overlap_matrix = partial_isin(bin_edges_in=infiltration_tedges_2d[i, :], bin_edges_out=cin_tedges_days)
            accumulated_weights[valid_bins_2d[i, :], :] += overlap_matrix[valid_bins_2d[i, :], :]

    # Average across valid pore volumes and apply flow weighting
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cout = valid_pv_count > 0
    averaged_weights[valid_cout, :] = accumulated_weights[valid_cout, :] / valid_pv_count[valid_cout, None]

    # Apply flow weighting after averaging
    flow_weighted_averaged = averaged_weights * flow_values[None, :]

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
    >>> cout_values = cout.values
    >>> flow_values = flow.values
    >>>
    >>> cin = extraction_to_infiltration(
    ...     cout=cout_values,
    ...     flow=flow_values,
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

    if len(tedges) != len(cout) + 1:
        msg = "tedges must have one more element than cout"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    # Convert to arrays for vectorized operations
    cout_values = np.asarray(cout)
    flow_values = np.asarray(flow)

    # Validate inputs do not contain NaN values
    if np.any(np.isnan(cout_values)):
        msg = "cout contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow_values)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)
    cout_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cin_tedges_days = ((cin_tedges - tedges[0]) / pd.Timedelta(days=1)).values
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes)

    # Pre-compute all residence times and extraction edges (symmetric to infiltration_to_extraction)
    rt_edges_2d = residence_time(
        flow=flow_values,
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

    # Initialize accumulator
    normalized_weights = _extraction_to_infiltration_weights(
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        cout_values=cout_values,
        flow_values=flow_values,
        cout_tedges_days=cout_tedges_days,
        extraction_tedges_2d=extraction_tedges_2d,
        valid_bins_2d=valid_bins_2d,
        valid_pv_count=valid_pv_count,
    )

    # Apply to concentrations and handle NaN for periods with no contributions
    out = normalized_weights.dot(cout_values)
    out[valid_pv_count == 0] = np.nan

    return out


def _extraction_to_infiltration_weights(
    *,
    cin_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cout_values: npt.NDArray[np.floating],
    flow_values: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    extraction_tedges_2d: npt.NDArray[np.floating],
    valid_bins_2d: npt.NDArray[np.bool_],
    valid_pv_count: npt.NDArray[np.intp],
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
    cin_tedges : pandas.DatetimeIndex
        Time edges for output (infiltration) data bins.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3].
    cout_values : array-like
        Concentration values of extracted water.
    flow_values : array-like
        Flow rate values in the aquifer [m3/day].
    cout_tedges_days : array-like
        Time edges for cout data in days since reference.
    extraction_tedges_2d : array-like
        2D array of extraction time edges for each pore volume.
    valid_bins_2d : array-like
        2D boolean array indicating valid time bins for each pore volume.
    valid_pv_count : array-like
        Count of valid pore volumes for each output time bin.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix for extraction to infiltration transformation.
        Shape: (len(cin_tedges) - 1, len(cout_values))
    """
    accumulated_weights = np.zeros((len(cin_tedges) - 1, len(cout_values)))

    # Loop over each pore volume (same structure as infiltration_to_extraction)
    for i in range(len(aquifer_pore_volumes)):
        if np.any(valid_bins_2d[i, :]):
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
    flow_weighted_averaged = averaged_weights * flow_values[None, :]

    # Normalize by total weights (symmetric to infiltration_to_extraction)
    total_weights = np.sum(flow_weighted_averaged, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(flow_weighted_averaged)
    normalized_weights[valid_weights, :] = flow_weighted_averaged[valid_weights, :] / total_weights[valid_weights, None]

    return normalized_weights
