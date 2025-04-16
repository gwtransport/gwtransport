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

from gwtransport1d.gamma import bins as gamma_bins
from gwtransport1d.residence_time import residence_time
from gwtransport1d.utils import interp_series, linear_interpolate


def forward(cin, flow, aquifer_pore_volume, retardation_factor=1.0, resample_dates=None):
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer.

    This function represents a forward operation (equivalent to convolution).

    Parameters
    ----------
    cin : pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].

    Returns
    -------
    pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    """
    rt_float = residence_time(
        flow, aquifer_pore_volume, index=cin.index, retardation_factor=retardation_factor, direction="infiltration"
    )
    rt = pd.to_timedelta(rt_float, unit="D")
    cout = pd.Series(data=cin.values, index=cin.index + rt, name="cout")

    if resample_dates is not None:
        cout = pd.Series(interp_series(cout, resample_dates), index=resample_dates, name="cout")

    return cout


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


def gamma_forward(cin, flow, alpha, beta, n_bins=100, retardation_factor=1.0):
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer. The aquifer pore volume is approximated by a gamma distribution, with
    parameters alpha and beta.

    This function represents a forward operation (equivalent to convolution).

    Parameters
    ----------
    cin : pandas.Series
        Concentration of the compound in the extracted water [ng/m3] or temperature in infiltrating water.
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    alpha : float
        Shape parameter of gamma distribution (must be > 0)
    beta : float
        Scale parameter of gamma distribution (must be > 0)
    n_bins : int
        Number of bins to discretize the gamma distribution.
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    pandas.Series
        Concentration of the compound in the extracted water [ng/m3] or temperature.
    """
    bins = gamma_bins(alpha, beta, n_bins)
    return distribution_forward(cin, flow, bins["edges"], retardation_factor=retardation_factor)


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
        Shape parameter of gamma distribution (must be > 0)
    beta : float
        Scale parameter of gamma distribution (must be > 0)
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


def distribution_forward(cin, flow, aquifer_pore_volume_edges, retardation_factor=1.0):
    """
    Similar to forward_advection, but with a distribution of aquifer pore volumes.

    Parameters
    ----------
    cin : pandas.Series
        Concentration of the compound in infiltrating water or temperature of infiltrating
        water.
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
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
    day_of_extraction = np.array(flow.index - flow.index[0]) / np.timedelta64(1, "D")

    # Use temperature at center point of bin
    rt_edges = residence_time(
        flow, aquifer_pore_volume_edges, retardation_factor=retardation_factor, direction="extraction"
    )
    day_of_infiltration_edges = day_of_extraction - rt_edges

    cin_sum = cin.cumsum()
    cin_sum_edges = linear_interpolate(day_of_extraction, cin_sum, day_of_infiltration_edges)
    n_measurements = linear_interpolate(day_of_extraction, np.arange(cin.size), day_of_infiltration_edges)
    cout_arr = np.diff(cin_sum_edges, axis=0) / np.diff(n_measurements, axis=0)

    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="Mean of empty slice")
        cout_data = np.nanmean(cout_arr, axis=0)

    return pd.Series(data=cout_data, index=flow.index, name="cout")


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
