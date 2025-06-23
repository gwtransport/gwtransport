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
    bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    return distribution_forward(
        cin=cin,
        cin_tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        flow_tedges=flow_tedges,
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
    cin_tedges,
    cout_tedges,
    flow,
    flow_tedges,
    aquifer_pore_volumes,
    retardation_factor=1.0,
):
    """
    Similar to forward_advection, but with a distribution of aquifer pore volumes.

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
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution.
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    numpy.ndarray
        Concentration of the compound in the extracted water or temperature. Same units as cin.
    """
    cin_tedges = pd.DatetimeIndex(cin_tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    if len(cin_tedges) != len(cin) + 1:
        msg = "cin_tedges must have one more element than cin"
        raise ValueError(msg)
    if len(cout_tedges) != len(flow) + 1:
        msg = "cout_tedges must have one more element than flow"
        raise ValueError(msg)

    cout_time = cout_tedges[:-1] + (cout_tedges[1:] - cout_tedges[:-1]) / 2

    # Use residence time at cout_time for all pore volumes
    rt_array = residence_time(
        flow=flow,
        flow_tedges=flow_tedges,
        index=cout_time,
        aquifer_pore_volume=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction",
    ).astype("timedelta64[D]")
    day_of_infiltration_array = cout_time.values[None] - rt_array

    cin_sum = np.concat(([0.0], cin.cumsum()))  # Add a zero at the beginning for cumulative sum
    cin_sum_interpolated = linear_interpolate(cin_tedges, cin_sum, day_of_infiltration_array)
    n_measurements = linear_interpolate(cin_tedges, np.arange(cin_tedges.size), day_of_infiltration_array)

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        warnings.filterwarnings(action="ignore", message="invalid value encountered in divide")
        cout_arr = np.diff(cin_sum_interpolated, axis=0) / np.diff(n_measurements, axis=0)
        return np.nanmean(cout_arr, axis=0)


def distribution_forward_v2(
    *,
    cin,
    tedges,
    flow,
    cout_tedges,
    aquifer_pore_volumes,
    retardation_factor=1.0,
):
    """
    Compute the concentration of the extracted water with aligned time edges.

    This function implements a forward advection model where cin and flow share
    the same time edges (tedges). The cout time edges are provided separately
    and have different time alignment than cin and flow. The output values are
    computed as flow-weighted averages of cin at the time of infiltration using
    the partial_isin function to handle bin overlaps.

    Parameters
    ----------
    cin : pandas.Series
        Concentration of the compound in infiltrating water or temperature of infiltrating
        water.
    tedges : pandas.DatetimeIndex
        Time edges for both cin and flow data. Has a length of one more than both
        cin and flow.
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    cout_tedges : pandas.DatetimeIndex
        Time edges for the output data. Has a length of one more than the desired output.
        These have different time alignment than cin and flow.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution.
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    numpy.ndarray
        Concentration of the compound in the extracted water or temperature.
        Same units as cin.
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
    cin_values = cin.values
    flow_values = flow.values
    cin_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes)

    # Compute residence times for all pore volumes at cout_tedges
    # rt_edges shape: (n_pore_volumes, n_cout_edges)
    rt_edges = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction",
    ).astype("timedelta64[D]") / np.timedelta64(1, "D")

    # Compute infiltration time edges for all pore volumes: shape (n_pore_volumes, n_edges)
    # This is cout_tedges shifted by residence time to get infiltration time
    infiltration_tedges_matrix = cout_tedges_days[None, :] - rt_edges

    # Check for overlap with cin time range
    # Shape: (n_pv,) - boolean mask for pore volumes with overlap
    overlap_mask = (infiltration_tedges_matrix[:, -1] >= cin_tedges_days[0]) & (
        infiltration_tedges_matrix[:, 0] <= cin_tedges_days[-1]
    )

    # Filter to only overlapping pore volumes
    valid_infiltration_edges = infiltration_tedges_matrix[overlap_mask]

    if len(valid_infiltration_edges) == 0:
        # No valid contributions
        return np.full(len(cout_tedges) - 1, np.nan)

    # Compute overlap matrices for ALL valid pore volumes at once using enhanced partial_isin
    # overlap_matrices shape: (n_valid_pv, n_cout, n_cin)
    try:
        overlap_matrices = partial_isin(valid_infiltration_edges, cin_tedges_days)
    except ValueError as e:
        if "ascending order" in str(e):
            # Handle non-monotonic infiltration edges by returning all NaN
            # This can happen when residence time calculations produce non-monotonic results
            # due to extreme flow variations or inappropriate pore volume/flow combinations
            return np.full(len(cout_tedges) - 1, np.nan)
        # Re-raise other ValueError types
        raise

    # Vectorized flow-weighted averaging for all pore volumes simultaneously
    # Get the number of output bins from overlap_matrices
    _, n_cout, _ = overlap_matrices.shape

    # We need to map flow values to cout bins
    # flow_values corresponds to cin/flow bins, but we need weights for cout bins
    # Since cout_tedges might be different from flow tedges, we need to be careful

    # For now, assume we can use flow values directly for cout bins
    # This might need adjustment based on the specific mapping between cout and flow
    if len(flow_values) == n_cout:
        # Direct mapping possible
        flow_broadcast = flow_values[None, :, None]  # Shape: (1, n_cout, 1)
    else:
        # Need to interpolate or map flow values to cout time structure
        # For simplicity, use average flow value
        avg_flow = np.mean(flow_values)
        flow_broadcast = np.full((1, n_cout, 1), avg_flow)

    # Create weight matrices: flow[i] * overlap[pv,i,j] for all pore volumes
    # Shape: (n_valid_pv, n_cout, n_cin)
    weight_matrices = flow_broadcast * overlap_matrices

    # Let NaNs propagate naturally - don't mask cin values
    # Shape: (n_valid_pv, n_cout, n_cin)
    cin_broadcast = cin_values[None, None, :]  # Broadcast cin to all pore volumes and cout bins

    # Compute weighted contributions: weight * cin for each (pv, cout, cin) combination
    weighted_contributions = weight_matrices * cin_broadcast

    # Sum over cin dimension to get total contribution for each (pv, cout) pair
    # Shape: (n_valid_pv, n_cout)
    total_weighted_contributions = np.sum(weighted_contributions, axis=2)
    total_weights = np.sum(weight_matrices, axis=2)

    # Compute contributions with natural NaN propagation
    # Shape: (n_valid_pv, n_cout)
    # Where total_weights is 0 (no overlap), result will be 0/0 = NaN (as desired)
    valid_contributions = total_weighted_contributions / total_weights

    # Create full contributions matrix for all pore volumes (including non-overlapping ones)
    n_pv, n_cout = len(aquifer_pore_volumes), len(cout_tedges) - 1
    final_contributions = np.full((n_pv, n_cout), np.nan)  # Start with NaN
    final_contributions[overlap_mask] = valid_contributions

    # Average contributions across all pore volumes
    # NaNs will propagate naturally in the mean operation
    return np.nanmean(final_contributions, axis=0)


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
