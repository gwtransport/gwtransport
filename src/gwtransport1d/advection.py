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

import pandas as pd

from gwtransport1d.deposition import interp_series
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
