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
- forward: Compute the concentration of the extracted water by shifting cin with its residence time. This corresponds to a convolution operation.
- gamma_forward: Similar to forward, but for a gamma distribution of aquifer pore volumes.
- distribution_forward: Similar to forward, but for an arbitrairy distribution of aquifer pore volumes.
"""

import numpy as np
import pandas as pd

from gwtransport import gamma
from gwtransport.residence_time import residence_time
from gwtransport.utils import compute_time_edges, interp_series, partial_isin


def forward(cin_series, flow_series, aquifer_pore_volume, retardation_factor=1.0, cout_index="cin"):
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer.

    This function represents a forward operation (equivalent to convolution).

    Parameters
    ----------
    cin_series : pandas.Series
        Concentration of the compound in the extracted water [ng/m3]. The cin_series should be the average concentration of a time bin. The index should be a pandas.DatetimeIndex
        and is labeled at the end of the time bin.
    flow_series : pandas.Series
        Flow rate of water in the aquifer [m3/day]. The flow_series should be the average flow of a time bin. The index should be a pandas.DatetimeIndex
        and is labeled at the end of the time bin.
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    cout_index : str, optional
        The index of the output series. Can be 'cin', 'flow', or 'cout'. Default is 'cin'.
        - 'cin': The output series will have the same index as `cin_series`.
        - 'flow': The output series will have the same index as `flow_series`.
        - 'cout': The output series will have the same index as `cin_series + residence_time`.

    Returns
    -------
    numpy.ndarray
        Concentration of the compound in the extracted water [ng/m3].

    Examples
    --------
    Basic usage with single pore volume:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.advection import forward
    >>>
    >>> # Create input data
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    >>> cin = pd.Series(np.ones(len(dates)), index=dates)
    >>> flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # 100 m3/day
    >>>
    >>> # Single aquifer pore volume
    >>> aquifer_pore_volume = 500.0  # m3
    >>>
    >>> # Run forward model (default returns on cin index)
    >>> cout = forward(cin, flow, aquifer_pore_volume)
    >>> len(cout) == len(cin)
    True

    Different output index options:

    >>> # Output on cin index (default)
    >>> cout_cin = forward(cin, flow, aquifer_pore_volume, cout_index="cin")
    >>>
    >>> # Output on flow index
    >>> cout_flow = forward(cin, flow, aquifer_pore_volume, cout_index="flow")
    >>>
    >>> # Output on shifted time index (cin + residence_time)
    >>> cout_shifted = forward(cin, flow, aquifer_pore_volume, cout_index="cout")

    With retardation factor:

    >>> # Compound moves twice as slowly due to sorption
    >>> cout = forward(cin, flow, aquifer_pore_volume, retardation_factor=2.0)
    """
    # Create flow tedges from the flow series index (assuming it's at the end of bins)
    flow_tedges = compute_time_edges(tedges=None, tstart=None, tend=flow_series.index, number_of_bins=len(flow_series))
    rt_array = residence_time(
        flow=flow_series,
        flow_tedges=flow_tedges,
        index=cin_series.index,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="infiltration",
    )

    rt = pd.to_timedelta(rt_array[0], unit="D", errors="coerce")
    index = cin_series.index + rt

    cout = pd.Series(data=cin_series.values, index=index, name="cout")

    if cout_index == "cin":
        return interp_series(cout, cin_series.index)
    if cout_index == "flow":
        # If cout_index is 'flow', we need to resample cout to the flow index
        return interp_series(cout, flow_series.index)
    if cout_index == "cout":
        # If cout_index is 'cout', we return the cout as is
        return cout.values

    msg = f"Invalid cout_index: {cout_index}. Must be 'cin', 'flow', or 'cout'."
    raise ValueError(msg)


def backward(cout, flow, aquifer_pore_volume, retardation_factor=1.0, resample_dates=None):
    """
    Compute the concentration of the infiltrating water by shifting cout with its residence time.

    This function represents a backward operation (equivalent to deconvolution).

    Parameters
    ----------
    cout : pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].

    Returns
    -------
    numpy.ndarray
        Concentration of the compound in the infiltrating water [ng/m3].
    """
    msg = "Backward advection (deconvolution) is not implemented yet"
    raise NotImplementedError(msg)


def gamma_forward(
    *,
    cin,
    flow,
    tedges,
    cout_tedges,
    alpha=None,
    beta=None,
    mean=None,
    std=None,
    n_bins=100,
    retardation_factor=1.0,
):
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer. The aquifer pore volume is approximated by a gamma distribution, with
    parameters alpha and beta.

    This function represents a forward operation (equivalent to convolution).

    Provide either alpha and beta or mean and std.

    Parameters
    ----------
    cin : pandas.Series
        Concentration of the compound in infiltrating water or temperature of infiltrating
        water.
    cin_tedges : pandas.DatetimeIndex
        Time edges for the concentration data. Used to compute the cumulative concentration.
        Has a length of one more than `cin`.
    cout_tedges : pandas.DatetimeIndex
        Time edges for the output data. Used to compute the cumulative concentration.
        Has a length of one more than `flow`.
    flow : pandas.Series
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

    Examples
    --------
    Basic usage with alpha and beta parameters:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport import compute_time_edges
    >>> from gwtransport.advection import gamma_forward
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
    >>> # Run gamma_forward with alpha/beta parameters
    >>> cout = gamma_forward(
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

    >>> cout = gamma_forward(
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

    >>> cout = gamma_forward(
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
    return distribution_forward(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_value"],
        retardation_factor=retardation_factor,
    )


def gamma_backward(cout, flow, alpha, beta, n_bins=100, retardation_factor=1.0):
    """
    Compute the concentration of the infiltrating water by shifting cout with its residence time.

    This function represents a backward operation (equivalent to deconvolution).

    Parameters
    ----------
    cout : pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    alpha : float
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0)
    beta : float
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0)
    n_bins : int
        Number of bins to discretize the gamma distribution.
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    NotImplementedError
        This function is not yet implemented.
    """
    msg = "Backward advection gamma (deconvolution) is not implemented yet"
    raise NotImplementedError(msg)


def distribution_forward(
    *,
    cin,
    flow,
    tedges,
    cout_tedges,
    aquifer_pore_volumes,
    retardation_factor=1.0,
):
    """
    Compute the concentration of the extracted water using flow-weighted advection.

    This function implements a forward advection model where cin and flow values
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

    Examples
    --------
    Basic usage with pandas Series:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport import compute_time_edges
    >>> from gwtransport.advection import distribution_forward
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
    >>> # Run distribution_forward
    >>> cout = distribution_forward(
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
    >>> cout = distribution_forward(
    ...     cin=cin_values,
    ...     flow=flow_values,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )

    With retardation factor:

    >>> cout = distribution_forward(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     retardation_factor=2.0,  # Compound moves twice as slowly
    ... )

    Using single pore volume:

    >>> single_volume = np.array([100])  # Single 100 m3 pore volume
    >>> cout = distribution_forward(
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
        direction="extraction",
    )
    infiltration_tedges_2d = cout_tedges_days[None, :] - rt_edges_2d

    # Pre-compute valid bins and count
    valid_bins_2d = ~(np.isnan(infiltration_tedges_2d[:, :-1]) | np.isnan(infiltration_tedges_2d[:, 1:]))
    valid_pv_count = np.sum(valid_bins_2d, axis=0)

    # Initialize accumulator
    accumulated_weights = np.zeros((len(cout_tedges) - 1, len(cin_values)))

    # Loop over each pore volume
    for i in range(len(aquifer_pore_volumes)):
        if np.any(valid_bins_2d[i, :]):
            overlap_matrix = partial_isin(infiltration_tedges_2d[i, :], cin_tedges_days)
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

    # Apply to concentrations and handle NaN for periods with no contributions
    out = np.sum(normalized_weights * cin_values[None, :], axis=1)
    out[valid_pv_count == 0] = np.nan

    return out


def distribution_backward(cout, flow, aquifer_pore_volume_edges, retardation_factor=1.0):
    """
    Compute the concentration of the infiltrating water from the extracted water concentration considering a distribution of aquifer pore volumes.

    This function represents a backward operation (equivalent to deconvolution).

    Parameters
    ----------
    cout : pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume_edges : array-like
        Edges of the bins that define the distribution of the aquifer pore volume.
        Of size nbins + 1 [m3].
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    NotImplementedError
        This function is not yet implemented.
    """
    msg = "Backward advection distribution (deconvolution) is not implemented yet"
    raise NotImplementedError(msg)
