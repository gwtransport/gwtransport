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


def residence_time_retarded(flow, aquifer_pore_volume, retardation_factor, direction="extraction"):
    """
    Compute the residence time of retarded compound in the water in the aquifer.

    Parameters
    ----------
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    retardation_factor : float
        Retardation factor of the compound in the aquifer [dimensionless].
    direction : str, optional
        Direction of the flow. Either 'extraction' or 'infiltration'. Extraction refers to backward modeling: how many days ago did this extracted water infiltrate. Infiltration refers to forward modeling: how many days will it take for this infiltrated water to be extracted. Default is 'extraction'.

    Returns
    -------
    pandas.Series
        Residence time of the retarded compound in the aquifer [days].
    """
    flow_cum = flow.cumsum()
    dates_days_extraction = (flow.index - flow.index[0]) / np.timedelta64(1, "D")
    if direction == "extraction":
        # How many days ago was the extraced water infiltrated
        dates_infiltration_retarded = (
            pd.to_timedelta(
                np.interp(
                    flow_cum - retardation_factor * aquifer_pore_volume,
                    flow_cum,
                    dates_days_extraction,
                    left=np.nan,
                    right=np.nan,
                ),
                unit="D",
            )
            + flow.index[0]
        )
        data = (flow.index - dates_infiltration_retarded) / np.timedelta64(1, "D")
    elif direction == "infiltration":
        # In how many days the water that is infiltrated now be extracted
        dates_extraction_retarded = (
            pd.to_timedelta(
                np.interp(
                    flow_cum + retardation_factor * aquifer_pore_volume,
                    flow_cum,
                    dates_days_extraction,
                    left=np.nan,
                    right=np.nan,
                ),
                unit="D",
            )
            + flow.index[0]
        )
        data = (dates_extraction_retarded - flow.index) / np.timedelta64(1, "D")
    else:
        msg = "direction should be 'extraction' or 'infiltration'"
        raise ValueError(msg)

    return pd.Series(data=data, index=flow.index, name="residence_times_retarded")
