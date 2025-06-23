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

import warnings

import numpy as np
import pandas as pd

from gwtransport import gamma
from gwtransport.residence_time import residence_time
from gwtransport.utils import compute_time_edges, interp_series, linear_interpolate, partial_isin


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
    cin_tedges,
    cout_tedges,
    flow,
    flow_tedges,
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
    """
    # Ensure cin and flow have aligned time edges
    if not pd.Index(cin_tedges).equals(pd.Index(flow_tedges)):
        msg = "cin_tedges and flow_tedges must be identical for aligned data"
        raise ValueError(msg)

    bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    return distribution_forward(
        cin=cin,
        flow=flow,
        tedges=cin_tedges,
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

    # Convert flow back to Series for residence_time function compatibility
    flow_series = pd.Series(flow_values, index=tedges[:-1])

    # Compute residence times for all pore volumes at cout_tedges
    # rt_edges shape: (n_pore_volumes, n_cout_edges)
    rt_edges = residence_time(
        flow=flow_series,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction",
    ).astype("timedelta64[D]") / np.timedelta64(1, "D")

    # Step 1: Determine infiltration time windows for each pore volume
    # Shape: (n_pore_volumes, n_cout_edges) - each row represents infiltration timing for one pore volume
    # For each cout time, subtract residence time to find when that water infiltrated
    infiltration_tedges_matrix = cout_tedges_days[None, :] - rt_edges

    # Step 2: Calculate temporal overlaps between infiltration windows and cin time bins
    # Shape: (n_pv, n_cout, n_cin) - fraction of each cin bin contributing to each cout bin per pore volume
    # Uses vectorized partial_isin to handle all pore volumes simultaneously
    overlap_matrices = partial_isin(infiltration_tedges_matrix, cin_tedges_days)

    # Step 3: Create flow-weighted overlap matrices
    # Shape: (n_pv, n_cout, n_cin) - combines temporal overlap with flow weighting
    # Each cin time bin is weighted by its corresponding flow rate
    flow_weighted_overlaps = flow_values[None, None, :] * overlap_matrices

    # Step 4: Normalize weights to create proper weighted average coefficients
    # Shape: (n_pv, n_cout, 1) - total weight for normalization per (pore_volume, cout_time) pair
    total_weights = np.sum(flow_weighted_overlaps, axis=2, keepdims=True)

    # Shape: (n_pv, n_cout, n_cin) - normalized weights that sum to 1 along cin dimension
    # Where total_weights is 0 (no overlap), division creates NaN (correct for no-data periods)
    normalized_weights = flow_weighted_overlaps / total_weights

    # Step 5: Apply weights to concentration values
    # Shape: (n_pv, n_cout, n_cin) - weighted concentration contributions
    weighted_contributions = normalized_weights * cin_values[None, None, :]

    # Step 6: Sum weighted contributions to get final concentrations per pore volume
    # Shape: (n_pv, n_cout) - weighted average concentration for each (pore_volume, cout_time) pair
    pore_volume_results = np.sum(weighted_contributions, axis=2)

    # Step 7: Average across all pore volumes to get final result
    # Shape: (n_cout,) - final concentration time series
    # NaN values naturally propagate when no valid contributions exist
    return np.nanmean(pore_volume_results, axis=0)


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
