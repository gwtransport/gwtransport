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
- get_cout_advection: Compute the concentration of the extracted water by shifting cin with its residence time.

The module leverages numpy, pandas, and scipy for efficient numerical computations
and time series handling. It is designed for researchers and engineers working on
groundwater contamination and transport problems.
"""

import numpy as np
import pandas as pd

from gwtransport1d.deposition import interp_series
from gwtransport1d.gamma import gamma_equal_mass_bins
from gwtransport1d.residence_time import residence_time_retarded


def get_cout_advection(cin, flow, aquifer_pore_volume, retardation_factor, resample_dates=None):
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
    rt_infiltration = residence_time_retarded(flow, aquifer_pore_volume, retardation_factor, direction="infiltration")
    rt = pd.to_timedelta(interp_series(rt_infiltration, cin.index), unit="D")
    cout = pd.Series(data=cin.values, index=cin.index + rt, name="cout")

    if resample_dates is not None:
        cout = pd.Series(interp_series(cout, resample_dates), index=resample_dates, name="cout")

    return cout


def get_cout_advection_gamma(cin, flow, alpha, beta, n_bins=100, retardation_factor=1.0, min_frac_known=0.75):
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer. The aquifer pore volume is approximated by a gamma distribution, with
    parameters alpha and beta.

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

    Returns
    -------
    pandas.Series
        Concentration of the compound in the extracted water [ng/m3] or temperature.
    """
    # Every apv bin transports the same fraction of flow
    bins = gamma_equal_mass_bins(alpha, beta, n_bins)
    aquifer_pore_volume = bins["expected_value"]

    day_of_extraction = np.array(flow.index - flow.index[0]) / np.timedelta64(1, "D")

    # Use temperature at center point of bin
    rt = residence_time_retarded(flow, aquifer_pore_volume, retardation_factor, direction="extraction")
    day_of_infiltration = day_of_extraction - rt
    iday_of_infiltration = np.searchsorted(day_of_extraction, day_of_infiltration)

    # Setup mask for nan values and only compute temperature for when the infiltration temp of min_frac_known is known
    mask = np.isnan(day_of_infiltration)
    mask[1.0 - (np.count_nonzero(mask, axis=1) / mask.shape[1]) < min_frac_known] = np.nan

    iday_of_infiltration[mask] = 0  # Integers are not allowed to be nan
    tout_arr = cin.values[iday_of_infiltration]
    tout_arr[mask] = np.nan

    return np.nanmean(tout_arr, axis=0)
