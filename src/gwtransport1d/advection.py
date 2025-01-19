"""
Advection Analysis for 1D Aquifer Systems.

This module provides functions to analyze compound transport by advection
in aquifer systems. It includes tools for computing concentrations of the extracted water
based on the concentration of the infiltrating water, extraction data and aquifer properties.

The model assumes requires the groundwaterflow to be reduced to a 1D system. On one side,
water with a certain concentration infiltrates ('cin'), the water flows through the aquifer and
the compound of intrest flows through the aquifer with a retarded velocity. The water is
extracted ('cout').

Main functions:
- cout_advection: Compute the concentration of the extracted water by shifting cin with its residence time.
- cout_advection_distribution: Similar to cout_advection, but for a distribution of aquifer pore volumes.

The module leverages numpy, pandas, and scipy for efficient numerical computations
and time series handling. It is designed for researchers and engineers working on
groundwater contamination and transport problems.
"""

import warnings

import numpy as np
import pandas as pd

from gwtransport1d.residence_time import residence_time_retarded
from gwtransport1d.utils import interp_series, linear_interpolate


def cout_advection(cin, flow, aquifer_pore_volume, retardation_factor=1.0, resample_dates=None):
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer.

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
    rt_float = residence_time_retarded(
        flow, aquifer_pore_volume, index=cin.index, retardation_factor=retardation_factor, direction="infiltration"
    )
    rt = pd.to_timedelta(rt_float, unit="D")
    cout = pd.Series(data=cin.values, index=cin.index + rt, name="cout")

    if resample_dates is not None:
        cout = pd.Series(interp_series(cout, resample_dates), index=resample_dates, name="cout")

    return cout


def cout_advection_distribution(cin, flow, aquifer_pore_volume_edges, retardation_factor=1.0):
    """
    Similar to cout_advection, but with a distribution of aquifer pore volumes.

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
    rt_edges = residence_time_retarded(
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
