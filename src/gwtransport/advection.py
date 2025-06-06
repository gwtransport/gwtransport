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
from gwtransport.utils import compute_time_edges, interp_series, linear_interpolate


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
    pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    """
    rt_series = residence_time(
        flow=flow_series,
        flow_tedges=None,
        flow_tstart=None,
        flow_tend=flow_series.index,
        index=cin_series.index,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="infiltration",
        return_pandas_series=True,
    )

    rt = pd.to_timedelta(rt_series.values, unit="D", errors="coerce")
    index = cin_series.index + rt

    cout = pd.Series(data=cin_series.values, index=index, name="cout")

    if cout_index == "cin":
        return pd.Series(interp_series(cout, cin_series.index), index=cin_series.index, name="cout")
    if cout_index == "flow":
        # If cout_index is 'flow', we need to resample cout to the flow index
        return pd.Series(interp_series(cout, flow_series.index), index=flow_series.index, name="cout")
    if cout_index == "cout":
        # If cout_index is 'cout', we return the cout as is
        return cout

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
    pandas.Series
        Concentration of the compound in the infiltrating water [ng/m3].
    """
    msg = "Backward advection (deconvolution) is not implemented yet"
    raise NotImplementedError(msg)


def gamma_forward(
    *,
    cin,
    cin_tedges=None,
    cin_tstart=None,
    cin_tend=None,
    cout_tedges=None,
    cout_tstart=None,
    cout_tend=None,
    flow,
    flow_tedges=None,
    flow_tstart=None,
    flow_tend=None,
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

    Provide:
    - either alpha and beta or mean and std.
    - either cin_tedges or cin_tstart or cin_tend.
    - either cout_tedges or cout_tstart or cout_tend.
    - either flow_tedges or flow_tstart or flow_tend.

    Parameters
    ----------
    cin : pandas.Series
        Concentration of the compound in infiltrating water or temperature of infiltrating
        water.
    cin_tedges : pandas.DatetimeIndex, optional
        Time edges for the concentration data. If provided, it is used to compute the cumulative concentration.
    cin_tstart : pandas.DatetimeIndex, optional
        Timestamps aligned to the start of the concentration measurement intervals. Preferably use cin_tedges,
        but if not available this approach can be used for convenience.
    cin_tend : pandas.DatetimeIndex, optional
        Timestamps aligned to the end of the concentration measurement intervals. Preferably use cin_tedges,
        but if not available this approach can be used for convenience.
    cout_tedges : pandas.DatetimeIndex, optional
        Time edges for the output data. If provided, it is used to compute the cumulative concentration.
    cout_tstart : pandas.DatetimeIndex, optional
        Timestamps aligned to the start of the output measurement intervals. Preferably use cout_tedges,
        but if not available this approach can be used for convenience.
    cout_tend : pandas.DatetimeIndex, optional
        Timestamps aligned to the end of the output measurement intervals. Preferably use cout_tedges,
        but if not available this approach can be used for convenience.
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    flow_tedges : pandas.DatetimeIndex, optional
        Time edges for the flow data. If provided, it is used to compute the cumulative flow.
        If left to None, the index of `flow` is used. Has a length of one more than `flow`. Default is None.
    flow_tstart : pandas.DatetimeIndex, optional
        Timestamps aligned to the start of the flow measurement intervals. Preferably use flow_tedges,
        but if not available this approach can be used for convenience. Has the same length as `flow`.
    flow_tend : pandas.DatetimeIndex, optional
        Timestamps aligned to the end of the flow measurement intervals. Preferably use flow_tedges,
        but if not available this approach can be used for convenience. Has the same length as `flow`.
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
    pandas.Series
        Concentration of the compound in the extracted water [ng/m3] or temperature.
    """
    bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    return distribution_forward(
        cin=cin,
        cin_tedges=cin_tedges,
        cin_tstart=cin_tstart,
        cin_tend=cin_tend,
        cout_tedges=cout_tedges,
        cout_tstart=cout_tstart,
        cout_tend=cout_tend,
        flow=flow,
        flow_tedges=flow_tedges,
        flow_tstart=flow_tstart,
        flow_tend=flow_tend,
        aquifer_pore_volume_edges=bins["edges"],
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
    cin_tedges=None,
    cin_tstart=None,
    cin_tend=None,
    cout_tedges=None,
    cout_tstart=None,
    cout_tend=None,
    flow,
    flow_tedges=None,
    flow_tstart=None,
    flow_tend=None,
    aquifer_pore_volume_edges=None,
    retardation_factor=1.0,
):
    """
    Similar to forward_advection, but with a distribution of aquifer pore volumes.

    Provide:
    - either cin_tedges or cin_tstart or cin_tend.
    - either cout_tedges or cout_tstart or cout_tend.
    - either flow_tedges or flow_tstart or flow_tend.

    Parameters
    ----------
    cin : pandas.Series
        Concentration of the compound in infiltrating water or temperature of infiltrating
        water.
    cin_tedges : pandas.DatetimeIndex, optional
        Time edges for the concentration data. If provided, it is used to compute the cumulative concentration.
    cin_tstart : pandas.DatetimeIndex, optional
        Timestamps aligned to the start of the concentration measurement intervals. Preferably use cin_tedges,
        but if not available this approach can be used for convenience.
    cin_tend : pandas.DatetimeIndex, optional
        Timestamps aligned to the end of the concentration measurement intervals. Preferably use cin_tedges,
        but if not available this approach can be used for convenience.
    cout_tedges : pandas.DatetimeIndex, optional
        Time edges for the output data. If provided, it is used to compute the cumulative concentration.
    cout_tstart : pandas.DatetimeIndex, optional
        Timestamps aligned to the start of the output measurement intervals. Preferably use cout_tedges,
        but if not available this approach can be used for convenience.
    cout_tend : pandas.DatetimeIndex, optional
        Timestamps aligned to the end of the output measurement intervals. Preferably use cout_tedges,
        but if not available this approach can be used for convenience.
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    flow_tedges : pandas.DatetimeIndex, optional
        Time edges for the flow data. If provided, it is used to compute the cumulative flow.
        If left to None, the index of `flow` is used. Has a length of one more than `flow`. Default is None.
    flow_tstart : pandas.DatetimeIndex, optional
        Timestamps aligned to the start of the flow measurement intervals. Preferably use flow_tedges,
        but if not available this approach can be used for convenience. Has the same length as `flow`.
    flow_tend : pandas.DatetimeIndex, optional
        Timestamps aligned to the end of the flow measurement intervals. Preferably use flow_tedges,
        but if not available this approach can be used for convenience. Has the same length as `flow`.
    aquifer_pore_volume_edges : array-like
        Edges of the bins that define the distribution of the aquifer pore volume.
        Of size nbins + 1 [m3].
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    pandas.Series
        Concentration of the compound in the extracted water or temperature. Same units as cin.
    """
    cin_tedges = compute_time_edges(tedges=cin_tedges, tstart=cin_tstart, tend=cin_tend, number_of_bins=len(cin))
    cout_tedges = compute_time_edges(tedges=cout_tedges, tstart=cout_tstart, tend=cout_tend, number_of_bins=len(flow))

    if cout_tstart is not None:
        cout_time = cout_tstart
    elif cout_tend is not None:
        cout_time = cout_tend
    elif cout_tedges is not None:
        cout_time = cout_tedges[:-1] + (cout_tedges[1:] - cout_tedges[:-1]) / 2
    else:
        msg = "Either cout_tedges, cout_tstart, or cout_tend must be provided."
        raise ValueError(msg)

    # Use residence time at cout_tedges
    rt_edges = residence_time(
        flow=flow,
        flow_tedges=flow_tedges,
        flow_tstart=flow_tstart,
        flow_tend=flow_tend,
        index=cout_time,
        aquifer_pore_volume=aquifer_pore_volume_edges,
        retardation_factor=retardation_factor,
        direction="extraction",
    ).astype("timedelta64[D]")
    day_of_infiltration_edges = cout_time.values[None] - rt_edges

    cin_sum = np.concat(([0.0], cin.cumsum()))  # Add a zero at the beginning for cumulative sum
    cin_sum_edges = linear_interpolate(cin_tedges, cin_sum, day_of_infiltration_edges)
    n_measurements = linear_interpolate(cin_tedges, np.arange(cin_tedges.size), day_of_infiltration_edges)
    cout_arr = np.diff(cin_sum_edges, axis=0) / np.diff(n_measurements, axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        cout_data = np.nanmean(cout_arr, axis=0)

    return pd.Series(data=cout_data, index=cout_time, name="cout")


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
