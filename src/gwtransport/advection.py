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

- :func:`infiltration_to_extraction_front_tracking` - Exact front tracking with Freundlich sorption.
  Event-driven algorithm that solves 1D advective transport with Freundlich isotherm using
  analytical integration of shock and rarefaction waves. Machine-precision physics (no numerical
  dispersion). Returns bin-averaged concentrations. Use case: Sharp concentration fronts with
  exact mass balance required, single deterministic flow path.

- :func:`infiltration_to_extraction_front_tracking_detailed` - Front tracking with piecewise structure.
  Same as infiltration_to_extraction_front_tracking but also returns complete piecewise analytical
  structure including all events, segments, and callable analytical forms C(t). Use case: Detailed
  analysis of shock and rarefaction wave dynamics.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport import gamma
from gwtransport.advection_utils import (
    _extraction_to_infiltration_weights,
    _infiltration_to_extraction_weights,
)
from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.output import compute_bin_averaged_concentration_exact
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave
from gwtransport.residence_time import residence_time

# Numerical tolerance constants
EPSILON_FREUNDLICH_N = 1e-10  # Tolerance for checking if freundlich_n ≈ 1.0


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
        aquifer_pore_volumes=aquifer_pore_volume,
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
        aquifer_pore_volumes=aquifer_pore_volume,
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
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.

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

    Note: For concentration-dependent retardation (nonlinear sorption),
    use `infiltration_to_extraction_front_tracking_detailed` instead, as this
    function only supports constant (float) retardation factors.

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

    # Compute normalized weights (includes all pre-computation)
    normalized_weights = _infiltration_to_extraction_weights(
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        cin=cin,
        flow=flow,
        retardation_factor=retardation_factor,
    )

    # Apply to concentrations and handle NaN for periods with no contributions
    out = normalized_weights.dot(cin)
    # Set NaN where no valid pore volumes contributed
    total_weights = np.sum(normalized_weights, axis=1)
    out[total_weights == 0] = np.nan

    return out


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

    See Also
    --------
    gamma_extraction_to_infiltration : Deconvolution with gamma-distributed pore volumes
    infiltration_to_extraction : Forward operation (convolution)
    extraction_to_infiltration_series : Simple time-shift for single pore volume
    gwtransport.residence_time.residence_time : Compute residence times from flow and pore volume
    gwtransport.residence_time.freundlich_retardation : Compute concentration-dependent retardation

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


def infiltration_to_extraction_front_tracking(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    freundlich_k: float | None = None,
    freundlich_n: float | None = None,
    bulk_density: float | None = None,
    porosity: float | None = None,
    retardation_factor: float | None = None,
    max_iterations: int = 10000,
) -> npt.NDArray[np.floating]:
    """
    Compute extracted concentration using exact front tracking with nonlinear sorption.

    Uses event-driven analytical algorithm that tracks shock waves, rarefaction waves,
    and characteristics with machine precision. No numerical dispersion, exact mass
    balance to floating-point precision.

    Parameters
    ----------
    cin : array-like
        Infiltration concentration [mg/L or any units].
        Length = len(tedges) - 1.
    flow : array-like
        Flow rate [m³/day]. Must be positive.
        Length = len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges. Length = len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Output time bin edges. Can be different from tedges.
        Length determines output array size.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m³] representing the distribution
        of residence times in the aquifer system. Each pore volume must be positive.
    freundlich_k : float, optional
        Freundlich coefficient [(m³/kg)^(1/n)]. Must be positive.
        Used if retardation_factor is None.
    freundlich_n : float, optional
        Freundlich exponent [-]. Must be positive and != 1.
        Used if retardation_factor is None.
    bulk_density : float, optional
        Bulk density [kg/m³]. Must be positive.
        Used if retardation_factor is None.
    porosity : float, optional
        Porosity [-]. Must be in (0, 1).
        Used if retardation_factor is None.
    retardation_factor : float, optional
        Constant retardation factor [-]. If provided, uses linear retardation
        instead of Freundlich sorption. Must be >= 1.0.
    max_iterations : int, optional
        Maximum number of events. Default 10000.

    Returns
    -------
    cout : numpy.ndarray
        Bin-averaged extraction concentration averaged across all pore volumes.
        Length = len(cout_tedges) - 1.

    Notes
    -----
    **Spin-up Period**:
    The function computes the first arrival time t_first. Concentrations
    before t_first are affected by unknown initial conditions and should
    not be used for analysis. Use `infiltration_to_extraction_front_tracking_detailed`
    to access t_first.

    **Machine Precision**:
    All calculations use exact analytical formulas. Mass balance is conserved
    to floating-point precision (~1e-14 relative error). No numerical tolerances
    are used for time/position calculations.

    **Physical Correctness**:
    - All shocks satisfy Lax entropy condition
    - Rarefaction waves use self-similar solutions
    - Causality is strictly enforced

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> # Pulse injection with single pore volume
    >>> tedges = pd.date_range("2020-01-01", periods=4, freq="10D")
    >>> cin = np.array([0.0, 10.0, 0.0])
    >>> flow = np.array([100.0, 100.0, 100.0])
    >>> cout_tedges = pd.date_range("2020-01-01", periods=10, freq="5D")
    >>>
    >>> cout = infiltration_to_extraction_front_tracking(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=np.array([500.0]),
    ...     freundlich_k=0.01,
    ...     freundlich_n=2.0,
    ...     bulk_density=1500.0,
    ...     porosity=0.3,
    ... )

    With multiple pore volumes (distribution):

    >>> aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
    >>> cout = infiltration_to_extraction_front_tracking(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     freundlich_k=0.01,
    ...     freundlich_n=2.0,
    ...     bulk_density=1500.0,
    ...     porosity=0.3,
    ... )

    See Also
    --------
    infiltration_to_extraction_front_tracking_detailed : Returns detailed structure
    infiltration_to_extraction : Convolution-based approach for linear case
    gamma_infiltration_to_extraction : For distributions of pore volumes
    """
    # Input validation
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    if len(tedges) != len(cin) + 1:
        msg = "tedges must have length len(cin) + 1"
        raise ValueError(msg)
    if len(flow) != len(cin):
        msg = "flow must have same length as cin"
        raise ValueError(msg)
    if np.any(cin < 0):
        msg = "cin must be non-negative"
        raise ValueError(msg)
    if np.any(flow <= 0):
        msg = "flow must be positive"
        raise ValueError(msg)
    if np.any(np.isnan(cin)) or np.any(np.isnan(flow)):
        msg = "cin and flow must not contain NaN"
        raise ValueError(msg)
    if np.any(aquifer_pore_volumes <= 0):
        msg = "aquifer_pore_volumes must be positive"
        raise ValueError(msg)

    # Convert cout_tedges to days (relative to tedges[0]) for output computation
    t_ref = tedges[0]
    cout_tedges_days = ((cout_tedges - t_ref) / pd.Timedelta(days=1)).values

    # Create sorption object
    if retardation_factor is not None:
        if retardation_factor < 1.0:
            msg = "retardation_factor must be >= 1.0"
            raise ValueError(msg)

        sorption = ConstantRetardation(retardation_factor=retardation_factor)
    else:
        if freundlich_k is None or freundlich_n is None or bulk_density is None or porosity is None:
            msg = (
                "Must provide either retardation_factor or all Freundlich parameters "
                "(freundlich_k, freundlich_n, bulk_density, porosity)"
            )
            raise ValueError(msg)
        if freundlich_k <= 0 or freundlich_n <= 0:
            msg = "Freundlich parameters must be positive"
            raise ValueError(msg)
        if abs(freundlich_n - 1.0) < EPSILON_FREUNDLICH_N:
            msg = "freundlich_n = 1 not supported (use retardation_factor for linear case)"
            raise ValueError(msg)
        if bulk_density <= 0 or not 0 < porosity < 1:
            msg = "Invalid physical parameters"
            raise ValueError(msg)

        sorption = FreundlichSorption(
            k_f=freundlich_k,
            n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

    # Loop over each pore volume and compute concentration
    cout_all = np.zeros((len(aquifer_pore_volumes), len(cout_tedges) - 1))

    for i, aquifer_pore_volume in enumerate(aquifer_pore_volumes):
        # Create tracker and run simulation for this pore volume
        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            sorption=sorption,
        )

        tracker.run(max_iterations=max_iterations)

        # Extract bin-averaged concentrations at outlet for this pore volume
        cout_all[i, :] = compute_bin_averaged_concentration_exact(
            t_edges=cout_tedges_days,
            v_outlet=aquifer_pore_volume,
            waves=tracker.state.waves,
            sorption=sorption,
        )

    # Return average across all pore volumes
    return np.mean(cout_all, axis=0)


def infiltration_to_extraction_front_tracking_detailed(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    freundlich_k: float | None = None,
    freundlich_n: float | None = None,
    bulk_density: float | None = None,
    porosity: float | None = None,
    retardation_factor: float | None = None,
    max_iterations: int = 10000,
) -> tuple[npt.NDArray[np.floating], list[dict]]:
    """
    Compute extracted concentration with complete diagnostic information.

    Returns both bin-averaged concentrations and detailed simulation structure for each pore volume.

    Parameters
    ----------
    cin : array-like
        Infiltration concentration [mg/L or any units].
        Length = len(tedges) - 1.
    flow : array-like
        Flow rate [m³/day]. Must be positive.
        Length = len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges. Length = len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Output time bin edges. Can be different from tedges.
        Length determines output array size.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m³] representing the distribution
        of residence times in the aquifer system. Each pore volume must be positive.
    freundlich_k : float, optional
        Freundlich coefficient [(m³/kg)^(1/n)]. Must be positive.
        Used if retardation_factor is None.
    freundlich_n : float, optional
        Freundlich exponent [-]. Must be positive and != 1.
        Used if retardation_factor is None.
    bulk_density : float, optional
        Bulk density [kg/m³]. Must be positive.
        Used if retardation_factor is None.
    porosity : float, optional
        Porosity [-]. Must be in (0, 1).
        Used if retardation_factor is None.
    retardation_factor : float, optional
        Constant retardation factor [-]. If provided, uses linear retardation
        instead of Freundlich sorption. Must be >= 1.0.
    max_iterations : int, optional
        Maximum number of events. Default 10000.

    Returns
    -------
    cout : numpy.ndarray
        Bin-averaged concentrations averaged across all pore volumes.

    structures : list of dict
        List of detailed simulation structures, one for each pore volume, with keys:

        - 'waves': List[Wave] - All wave objects created during simulation
        - 'events': List[dict] - All events with times, types, and details
        - 't_first_arrival': float - First arrival time (end of spin-up period)
        - 'n_events': int - Total number of events
        - 'n_shocks': int - Number of shocks created
        - 'n_rarefactions': int - Number of rarefactions created
        - 'n_characteristics': int - Number of characteristics created
        - 'final_time': float - Final simulation time
        - 'sorption': FreundlichSorption | ConstantRetardation - Sorption object
        - 'tracker_state': FrontTrackerState - Complete simulation state
        - 'aquifer_pore_volume': float - Pore volume for this simulation

    Examples
    --------
    ::

        cout, structures = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Access spin-up period for first pore volume
        print(f"First arrival: {structures[0]['t_first_arrival']:.2f} days")

        # Analyze events for first pore volume
        for event in structures[0]["events"]:
            print(f"t={event['time']:.2f}: {event['type']}")
    """
    # Input validation (same as main function)
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    if len(tedges) != len(cin) + 1:
        msg = "tedges must have length len(cin) + 1"
        raise ValueError(msg)
    if len(flow) != len(cin):
        msg = "flow must have same length as cin"
        raise ValueError(msg)
    if np.any(cin < 0):
        msg = "cin must be non-negative"
        raise ValueError(msg)
    if np.any(flow <= 0):
        msg = "flow must be positive"
        raise ValueError(msg)
    if np.any(np.isnan(cin)) or np.any(np.isnan(flow)):
        msg = "cin and flow must not contain NaN"
        raise ValueError(msg)
    if np.any(aquifer_pore_volumes <= 0):
        msg = "aquifer_pore_volumes must be positive"
        raise ValueError(msg)

    # Convert cout_tedges to days (relative to tedges[0]) for output computation
    t_ref = tedges[0]
    cout_tedges_days = ((cout_tedges - t_ref) / pd.Timedelta(days=1)).values

    # Create sorption object
    if retardation_factor is not None:
        if retardation_factor < 1.0:
            msg = "retardation_factor must be >= 1.0"
            raise ValueError(msg)

        sorption = ConstantRetardation(retardation_factor=retardation_factor)
    else:
        if freundlich_k is None or freundlich_n is None or bulk_density is None or porosity is None:
            msg = (
                "Must provide either retardation_factor or all Freundlich parameters "
                "(freundlich_k, freundlich_n, bulk_density, porosity)"
            )
            raise ValueError(msg)
        if freundlich_k <= 0 or freundlich_n <= 0:
            msg = "Freundlich parameters must be positive"
            raise ValueError(msg)
        if abs(freundlich_n - 1.0) < EPSILON_FREUNDLICH_N:
            msg = "freundlich_n = 1 not supported (use retardation_factor for linear case)"
            raise ValueError(msg)
        if bulk_density <= 0 or not 0 < porosity < 1:
            msg = "Invalid physical parameters"
            raise ValueError(msg)

        sorption = FreundlichSorption(
            k_f=freundlich_k,
            n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

    # Loop over each pore volume and compute concentration
    cout_all = np.zeros((len(aquifer_pore_volumes), len(cout_tedges) - 1))
    structures = []

    for i, aquifer_pore_volume in enumerate(aquifer_pore_volumes):
        # Create tracker and run simulation for this pore volume
        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            sorption=sorption,
        )

        tracker.run(max_iterations=max_iterations)

        # Extract bin-averaged concentrations for this pore volume
        cout_all[i, :] = compute_bin_averaged_concentration_exact(
            t_edges=cout_tedges_days,
            v_outlet=aquifer_pore_volume,
            waves=tracker.state.waves,
            sorption=sorption,
        )

        # Build detailed structure dict for this pore volume
        structure = {
            "waves": tracker.state.waves,
            "events": tracker.state.events,
            "t_first_arrival": tracker.t_first_arrival,
            "n_events": len(tracker.state.events),
            "n_shocks": sum(1 for w in tracker.state.waves if isinstance(w, ShockWave)),
            "n_rarefactions": sum(1 for w in tracker.state.waves if isinstance(w, RarefactionWave)),
            "n_characteristics": sum(1 for w in tracker.state.waves if isinstance(w, CharacteristicWave)),
            "final_time": tracker.state.t_current,
            "sorption": sorption,
            "tracker_state": tracker.state,
            "aquifer_pore_volume": aquifer_pore_volume,
        }
        structures.append(structure)

    # Return average concentrations and list of structures
    return np.mean(cout_all, axis=0), structures
