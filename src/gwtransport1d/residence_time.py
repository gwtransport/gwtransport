"""
Residence time of a compound in the aquifer.

This module provides functions to compute the residence time of a compound in the aquifer.
The residence time is the time it takes for the compound to travel from the infiltration
point to the extraction point. The compound is retarded in the aquifer with a retardation factor.

Main functions:
- residence_time_retarded: Compute the residence time of a retarded compound in the aquifer.

The module leverages numpy, pandas, and scipy for efficient numerical computations
and time series handling. It is designed for researchers and engineers working on
groundwater contamination and transport problems.
"""

import numpy as np
import pandas as pd

from gwtransport1d.utils import linear_interpolate


def residence_time_retarded(
    flow, aquifer_pore_volume, *, index=None, retardation_factor=1.0, direction="extraction", return_as_series=False
):
    """
    Compute the residence time of retarded compound in the water in the aquifer.

    This function can be used to compute when water was infiltrated that is now extracted and vice versa.

    Parameters
    ----------
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    index : pandas.DatetimeIndex, optional
        Index of the residence time. If left to None, the index of `flow` is used. Default is None.
    retardation_factor : float
        Retardation factor of the compound in the aquifer [dimensionless].
    direction : str, optional
        Direction of the flow. Either 'extraction' or 'infiltration'. Extraction refers to backward modeling: how many days ago did this extracted water infiltrate. Infiltration refers to forward modeling: how many days will it take for this infiltrated water to be extracted. Default is 'extraction'.

    Returns
    -------
    array
        Residence time of the retarded compound in the aquifer [days].
    """
    aquifer_pore_volume = np.atleast_1d(aquifer_pore_volume)
    dates_days_extraction = np.asarray((flow.index - flow.index[0]) / np.timedelta64(1, "D"))
    days_extraction = np.diff(dates_days_extraction, prepend=0.0)
    flow_cum = (flow.values * days_extraction).cumsum()

    if index is None:
        index = flow.index
        index_dates_days_extraction = dates_days_extraction
        flow_cum_at_index = flow_cum
    else:
        index_dates_days_extraction = np.asarray((index - flow.index[0]) / np.timedelta64(1, "D"))
        flow_cum_at_index = linear_interpolate(
            dates_days_extraction, flow_cum, index_dates_days_extraction, left=np.nan, right=np.nan
        )

    if direction == "extraction":
        # How many days ago was the extraced water infiltrated
        a = flow_cum_at_index[None, :] - retardation_factor * aquifer_pore_volume[:, None]
        days = linear_interpolate(flow_cum, dates_days_extraction, a, left=np.nan, right=np.nan)
        data = index_dates_days_extraction - days
    elif direction == "infiltration":
        # In how many days the water that is infiltrated now be extracted
        a = flow_cum_at_index[None, :] + retardation_factor * aquifer_pore_volume[:, None]
        days = linear_interpolate(flow_cum, dates_days_extraction, a, left=np.nan, right=np.nan)
        data = days - index_dates_days_extraction
    else:
        msg = "direction should be 'extraction' or 'infiltration'"
        raise ValueError(msg)

    if return_as_series:
        if len(aquifer_pore_volume) > 1:
            msg = "return_as_series=True is only supported for a single pore volume"
            raise ValueError(msg)
        return pd.Series(data=data[0], index=index, name=f"residence_time_{direction}")
    return data
