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

TODO: Create function residence_time_mean to compute the mean residence time. Making use of tedges argument.
"""

import numpy as np
import pandas as pd

from gwtransport1d.utils import linear_average, linear_interpolate


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


def residence_time_mean(
    flow, flow_tedges, tedges_out, aquifer_pore_volume, *, direction="extraction", retardation_factor=1.0
):
    """
    Compute the mean residence time of a retarded compound in the aquifer between specified time edges.

    This function calculates the average residence time of a retarded compound in the aquifer
    between specified time intervals. It can compute both backward modeling (extraction direction:
    when was extracted water infiltrated) and forward modeling (infiltration direction: when will
    infiltrated water be extracted).

    The function handles time series data by computing the cumulative flow and using linear
    interpolation and averaging to determine mean residence times between the specified time edges.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m3/day]. Should be an array of flow values
        corresponding to the intervals defined by flow_tedges.
    flow_tedges : array-like
        Time edges for the flow data, as datetime64 objects. These define the time
        intervals for which the flow values are provided.
    tedges_out : array-like
        Output time edges as datetime64 objects. These define the intervals for which
        the mean residence times will be calculated.
    aquifer_pore_volume : float or array-like
        Pore volume of the aquifer [m3]. Can be a single value or an array of values
        for multiple pore volume scenarios.
    direction : {'extraction', 'infiltration'}, optional
        Direction of the flow calculation:
        * 'extraction': Backward modeling - how many days ago was the extracted water infiltrated
        * 'infiltration': Forward modeling - how many days until the infiltrated water is extracted
        Default is 'extraction'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless].
        A value greater than 1.0 indicates that the compound moves slower than water.
        Default is 1.0 (no retardation).

    Returns
    -------
    numpy.ndarray
        Mean residence time of the retarded compound in the aquifer [days] for each interval
        defined by tedges_out. If multiple pore volumes are provided, the first dimension
        corresponds to the different pore volumes.

    Notes
    -----
    - The function converts datetime objects to days since the start of the time series.
    - For extraction direction, the function computes how many days ago water was infiltrated.
    - For infiltration direction, the function computes how many days until water will be extracted.
    - The function uses linear interpolation for computing residence times at specific points
      and linear averaging for computing mean values over intervals.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample flow data
    >>> flow_dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    >>> flow_values = np.full(len(flow_dates) - 1, 100.0)  # Constant flow of 100 m³/day
    >>> pore_volume = 200.0  # Aquifer pore volume in m³
    >>> # Define output time edges
    >>> output_dates = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    >>> # Calculate mean residence times
    >>> mean_times = residence_time_mean(
    ...     flow=flow_values,
    ...     flow_tedges=flow_dates,
    ...     tedges_out=output_dates,
    ...     aquifer_pore_volume=pore_volume,
    ...     direction="extraction",
    ... )
    >>> # With constant flow of 100 m³/day and pore volume of 200 m³,
    >>> # mean residence time should be approximately 2 days
    >>> print(mean_times)  # Output: [2.0, 2.0, 2.0, 2.0]
    """
    flow = np.asarray(flow)
    flow_tedges = np.asarray(flow_tedges)
    tedges_out = np.asarray(tedges_out)
    aquifer_pore_volume = np.atleast_1d(aquifer_pore_volume)
    flow_tedges_days = np.asarray((flow_tedges - flow_tedges[0]) / np.timedelta64(1, "D"))

    # compute cumulative flow at flow_tedges and flow_tedges_days
    flow_cum = np.diff(flow_tedges_days, prepend=0.0)
    flow_cum[1:] *= flow
    flow_cum = flow_cum.cumsum()

    if direction == "extraction":
        # How many days ago was the extraced water infiltrated
        a = flow_cum[None, :] - retardation_factor * aquifer_pore_volume[:, None]
        days = linear_interpolate(flow_cum, flow_tedges_days, a, left=np.nan, right=np.nan)
        data_edges = flow_tedges_days - days
        data_avg = linear_average(flow_tedges_days, data_edges, tedges_out)
    elif direction == "infiltration":
        # In how many days the water that is infiltrated now be extracted
        a = flow_cum[None, :] + retardation_factor * aquifer_pore_volume[:, None]
        days = linear_interpolate(flow_cum, flow_tedges_days, a, left=np.nan, right=np.nan)
        data_edges = days - flow_tedges_days
        data_avg = linear_average(flow_tedges_days, data_edges, tedges_out)
    else:
        msg = "direction should be 'extraction' or 'infiltration'"
        raise ValueError(msg)
    return data_avg
