"""Deposition analysis for 1D aquifer systems.

Analyze compound transport by deposition in aquifer systems with tools for
computing concentrations and deposition rates based on aquifer properties.

The model assumes 1D groundwater flow where compound deposition occurs along
the flow path, enriching the water. Follows advection module patterns for
consistency.

Functions
---------
extraction_to_deposition : Compute deposition rates from concentration changes
deposition_to_extraction : Compute concentrations from deposition rates
"""

import numpy as np
import pandas as pd

from gwtransport.residence_time import residence_time
from gwtransport.surfacearea import compute_average_heights
from gwtransport.utils import linear_interpolate


def extraction_to_deposition(
    *,
    cout,
    flow,
    tedges,
    dep_tedges,
    aquifer_pore_volume_value,
    porosity,
    thickness,
    retardation_factor=1.0,
):
    """Compute deposition rates from concentration changes (deconvolution).

    Parameters
    ----------
    cout : array_like
        Concentration changes in extracted water [ng/m3].
        Length must equal len(tedges) - 1.
    flow : array_like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges for cout and flow data.
    dep_tedges : pandas.DatetimeIndex
        Time bin edges for output deposition data.
    aquifer_pore_volume_value : float
        Aquifer pore volume [m3].
    porosity : float
        Aquifer porosity [dimensionless].
    thickness : float
        Aquifer thickness [m].
    retardation_factor : float, optional
        Compound retardation factor, by default 1.0.

    Returns
    -------
    ndarray
        Deposition rates [ng/m2/day] with length len(dep_tedges) - 1.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    >>> tedges = pd.date_range("2019-12-31 12:00", "2020-01-10 12:00", freq="D")
    >>> dep_tedges = pd.date_range("2019-12-28 12:00", "2020-01-08 12:00", freq="D")
    >>> cout = np.ones(len(dates))
    >>> flow = np.full(len(dates), 100.0)
    >>> dep = extraction_to_deposition(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     dep_tedges=dep_tedges,
    ...     aquifer_pore_volume_value=500.0,
    ...     porosity=0.3,
    ...     thickness=10.0,
    ... )
    """
    tedges, dep_tedges = pd.DatetimeIndex(tedges), pd.DatetimeIndex(dep_tedges)
    cout_values, flow_values = np.asarray(cout), np.asarray(flow)

    # Validate input dimensions and values
    if len(tedges) != len(cout_values) + 1:
        raise ValueError("tedges must have one more element than cout")
    if len(tedges) != len(flow_values) + 1:
        raise ValueError("tedges must have one more element than flow")
    if np.any(np.isnan(cout_values)) or np.any(np.isnan(flow_values)):
        raise ValueError("Input arrays cannot contain NaN values")

    # Convert to days relative to first time edge
    t0 = tedges[0]
    cout_tedges_days = ((tedges - t0) / pd.Timedelta(days=1)).values
    dep_tedges_days = ((dep_tedges - t0) / pd.Timedelta(days=1)).values

    # Compute residence times and cumulative flow
    rt_edges = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=dep_tedges,
        aquifer_pore_volume=float(aquifer_pore_volume_value),
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",
    )
    dep_tedges_days_extraction = dep_tedges_days + rt_edges[0]

    flow_tdelta = np.diff(cout_tedges_days, prepend=0.0)
    flow_cum = (np.concatenate(([0.0], flow_values)) * flow_tdelta).cumsum()

    # Interpolate volumes at deposition time edges
    start_vol = linear_interpolate(cout_tedges_days, flow_cum, dep_tedges_days)
    end_vol = linear_interpolate(cout_tedges_days, flow_cum, dep_tedges_days_extraction)

    # Compute concentration weights
    flow_cum_dep = flow_cum[None, :] - start_vol[:, None]
    volume_array = compute_average_heights(
        cout_tedges_days, flow_cum_dep, 0.0, retardation_factor * float(aquifer_pore_volume_value)
    )
    area_array = volume_array / (porosity * thickness)
    extracted_volume = np.diff(end_vol)
    concentration_weights = area_array * np.diff(cout_tedges_days)[None, :] / extracted_volume[:, None]

    return concentration_weights.dot(cout_values)


def deposition_to_extraction(
    *,
    dep,
    flow,
    tedges,
    cout_tedges,
    aquifer_pore_volume_value,
    porosity,
    thickness,
    retardation_factor=1.0,
):
    """Compute concentrations from deposition rates (convolution).

    Parameters
    ----------
    dep : array_like
        Deposition rates [ng/m2/day]. Length must equal len(tedges) - 1.
    flow : array_like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges for dep and flow data.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for output concentration data.
    aquifer_pore_volume_value : float
        Aquifer pore volume [m3].
    porosity : float
        Aquifer porosity [dimensionless].
    thickness : float
        Aquifer thickness [m].
    retardation_factor : float, optional
        Compound retardation factor, by default 1.0.

    Returns
    -------
    ndarray
        Concentration changes [ng/m3] with length len(cout_tedges) - 1.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    >>> tedges = pd.date_range("2019-12-31 12:00", "2020-01-10 12:00", freq="D")
    >>> cout_tedges = pd.date_range("2020-01-03 12:00", "2020-01-12 12:00", freq="D")
    >>> dep = np.ones(len(dates))
    >>> flow = np.full(len(dates), 100.0)
    >>> cout = deposition_to_extraction(
    ...     dep=dep,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volume_value=500.0,
    ...     porosity=0.3,
    ...     thickness=10.0,
    ... )
    """
    tedges, cout_tedges = pd.DatetimeIndex(tedges), pd.DatetimeIndex(cout_tedges)
    dep_values, flow_values = np.asarray(dep), np.asarray(flow)

    # Validate input dimensions and values
    if len(tedges) != len(dep_values) + 1:
        raise ValueError("tedges must have one more element than dep")
    if len(tedges) != len(flow_values) + 1:
        raise ValueError("tedges must have one more element than flow")
    if np.any(np.isnan(dep_values)) or np.any(np.isnan(flow_values)):
        raise ValueError("Input arrays cannot contain NaN values")

    # Convert to days relative to first time edge
    t0 = tedges[0]
    tedges_days = ((tedges - t0) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - t0) / pd.Timedelta(days=1)).values

    # Compute residence times and cumulative flow
    rt_edges = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volume=float(aquifer_pore_volume_value),
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    cout_tedges_days_infiltration = cout_tedges_days - rt_edges[0]

    flow_tdelta = np.diff(tedges_days, prepend=0.0)
    flow_cum = (np.concatenate(([0.0], flow_values)) * flow_tdelta).cumsum()

    # Interpolate volumes at concentration time edges
    start_vol = linear_interpolate(tedges_days, flow_cum, cout_tedges_days_infiltration)
    end_vol = linear_interpolate(tedges_days, flow_cum, cout_tedges_days)

    # Compute deposition weights
    flow_cum_cout = flow_cum[None, :] - start_vol[:, None]
    volume_array = compute_average_heights(
        tedges_days, flow_cum_cout, 0.0, retardation_factor * float(aquifer_pore_volume_value)
    )
    area_array = volume_array / (porosity * thickness)
    extracted_volume = np.diff(end_vol)
    deposition_weights = area_array * np.diff(tedges_days)[None, :] / extracted_volume[:, None]

    return deposition_weights.dot(dep_values)
