"""
Residence time of a compound in the aquifer.

This module provides functions to compute the residence time of a compound in the aquifer.
The residence time is the time it takes for the compound to travel from the infiltration
point to the extraction point. The compound is retarded in the aquifer with a retardation factor.

Main functions:
- residence_time: Compute the residence time of a retarded compound in the aquifer at indices.
- residence_time_mean: Compute the mean residence time of a retarded compound in the aquifer
  between specified time edges.
"""

import numpy as np
import pandas as pd

from gwtransport.utils import compute_time_edges, linear_average, linear_interpolate


def residence_time(
    flow=None,
    flow_tedges=None,
    flow_tstart=None,
    flow_tend=None,
    aquifer_pore_volume=None,
    index=None,
    retardation_factor=1.0,
    direction="extraction",
    *,
    return_pandas_series=False,
):
    """
    Compute the residence time of retarded compound in the water in the aquifer.

    Parameters
    ----------
    flow : pandas.Series, array-like
        Flow rate of water in the aquifer [m3/day]. The timestamps of the flow data should be aligned with the time edges provided in `flow_tedges`.
        If left to None, the function will raise an error. The length of `flow` should match the length of `flow_tedges` minus one.
    flow_tedges : pandas.DatetimeIndex, optional
        Time edges for the flow data. If provided, it is used to compute the cumulative flow.
        If left to None, the index of `flow` is used. Has a length of one more than `flow`. Default is None.
    flow_tstart : pandas.DatetimeIndex, optional
        Timestamps aligned to the start of the flow measurement intervals. Preferably use flow_tedges,
        but if not available this approach can be used for convenience. Has the same length as `flow`.
    flow_tend : pandas.DatetimeIndex, optional
        Timestamps aligned to the end of the flow measurement intervals. Preferably use flow_tedges,
        but if not available this approach can be used for convenience. Has the same length as `flow`.
    aquifer_pore_volume : float or array-like of float
        Pore volume of the aquifer [m3].
    index : pandas.DatetimeIndex, optional
        Index at which to compute the residence time. If left to None, the index of `flow` is used.
        If Default is None.
    retardation_factor : float
        Retardation factor of the compound in the aquifer [dimensionless].
    direction : str, optional
        Direction of the flow. Either 'extraction' or 'infiltration'. Extraction refers to backward modeling: how many days ago did this extracted water infiltrate. Infiltration refers to forward modeling: how many days will it take for this infiltrated water to be extracted. Default is 'extraction'.
    return_pandas_series : bool, optional
        If True, return a pandas Series with the residence time at the index provided. Only supported for a single aquifer pore volume.

    Returns
    -------
    array
        Residence time of the retarded compound in the aquifer [days].
    """
    aquifer_pore_volume = np.atleast_1d(aquifer_pore_volume)

    flow_tedges = compute_time_edges(flow_tedges, flow_tstart, flow_tend, len(flow))

    flow_tedges_days = np.asarray((flow_tedges - flow_tedges[0]) / np.timedelta64(1, "D"))
    flow_tdelta = np.diff(flow_tedges_days, prepend=0.0)
    flow_values = np.concat(([0.0], np.asarray(flow)))
    flow_cum = (flow_values * flow_tdelta).cumsum()  # at flow_tedges and flow_tedges_days. First value is 0.

    if index is None:
        # If index is not provided return the residence time that matches with the index of the flow; at the center of the flow bin.
        index_dates_days_extraction = (flow_tedges_days[:-1] + flow_tedges_days[1:]) / 2
        flow_cum_at_index = (flow_cum[:-1] + flow_cum[1:]) / 2  # at the center of the flow bin
    else:
        index_dates_days_extraction = np.asarray((index - flow_tedges[0]) / np.timedelta64(1, "D"))
        flow_cum_at_index = linear_interpolate(
            flow_tedges_days, flow_cum, index_dates_days_extraction, left=np.nan, right=np.nan
        )

    if direction == "extraction":
        # How many days ago was the extraced water infiltrated
        a = flow_cum_at_index[None, :] - retardation_factor * aquifer_pore_volume[:, None]
        days = linear_interpolate(flow_cum, flow_tedges_days, a, left=np.nan, right=np.nan)
        data = index_dates_days_extraction - days
    elif direction == "infiltration":
        # In how many days the water that is infiltrated now be extracted
        a = flow_cum_at_index[None, :] + retardation_factor * aquifer_pore_volume[:, None]
        days = linear_interpolate(flow_cum, flow_tedges_days, a, left=np.nan, right=np.nan)
        data = days - index_dates_days_extraction
    else:
        msg = "direction should be 'extraction' or 'infiltration'"
        raise ValueError(msg)

    if return_pandas_series:
        if len(aquifer_pore_volume) > 1:
            msg = "return_pandas_series=True is only supported for a single pore volume"
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
        defined by tedges_out. The first dimension corresponds to the different pore volumes
        and the second to the residence times between tedges_out.

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
    >>> # Calculate mean residence times
    >>> mean_times = residence_time_mean(
    ...     flow=flow_values,
    ...     flow_tedges=flow_dates,
    ...     tedges_out=flow_dates,
    ...     aquifer_pore_volume=pore_volume,
    ...     direction="extraction",
    ... )
    >>> # With constant flow of 100 m³/day and pore volume of 200 m³,
    >>> # mean residence time should be approximately 2 days
    >>> print(mean_times)  # Output: [np.nan, np.nan, 2.0, 2.0, 2.0, ..., 2.0]
    """
    flow = np.asarray(flow)
    flow_tedges = pd.DatetimeIndex(flow_tedges)
    tedges_out = pd.DatetimeIndex(tedges_out)
    aquifer_pore_volume = np.atleast_1d(aquifer_pore_volume)

    flow_tedges_days = np.asarray((flow_tedges - flow_tedges[0]) / np.timedelta64(1, "D"))
    tedges_out_days = np.asarray((tedges_out - flow_tedges[0]) / np.timedelta64(1, "D"))

    # compute cumulative flow at flow_tedges and flow_tedges_days
    flow_cum = np.diff(flow_tedges_days, prepend=0.0)
    flow_cum[1:] *= flow
    flow_cum = flow_cum.cumsum()

    if direction == "extraction":
        # How many days ago was the extraced water infiltrated
        a = flow_cum[None, :] - retardation_factor * aquifer_pore_volume[:, None]
        days = linear_interpolate(flow_cum, flow_tedges_days, a, left=np.nan, right=np.nan)
        data_edges = flow_tedges_days - days
        data_avg = np.array([linear_average(flow_tedges_days, y, tedges_out_days) for y in data_edges])
    elif direction == "infiltration":
        # In how many days the water that is infiltrated now be extracted
        a = flow_cum[None, :] + retardation_factor * aquifer_pore_volume[:, None]
        days = linear_interpolate(flow_cum, flow_tedges_days, a, left=np.nan, right=np.nan)
        data_edges = days - flow_tedges_days
        data_avg = np.array([linear_average(flow_tedges_days, y, tedges_out_days) for y in data_edges])
    else:
        msg = "direction should be 'extraction' or 'infiltration'"
        raise ValueError(msg)
    return data_avg
