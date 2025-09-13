"""
Deposition Analysis for 1D Aquifer Systems (Simplified Version).

This module provides functions to analyze compound transport by deposition
in aquifer systems. It includes tools for computing concentrations due to
deposition and deposition rates based on concentration data and aquifer properties.

The model assumes the groundwater flow to be reduced to a 1D system. Water flows
through the aquifer and compound deposition occurs along the flow path, enriching
the water with the compound. This module follows the same patterns as the advection
module for consistency and simplicity.

Main functions:
- extraction_to_deposition: Compute deposition rates from extraction concentration changes (equivalent to deconvolution)
- deposition_to_extraction: Compute concentration changes due to deposition (equivalent to convolution)
- distribution_extraction_to_deposition: Deposition analysis with arbitrary distribution of pore volumes
- distribution_deposition_to_extraction: Concentration analysis with arbitrary distribution of pore volumes
"""

import numpy as np
import pandas as pd

from gwtransport.residence_time import residence_time
from gwtransport.surfacearea import compute_average_heights
from gwtransport.utils import linear_interpolate, partial_isin


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
    """
    Compute deposition rates from concentration changes in extracted water.

    This function represents extraction to deposition modeling (equivalent to deconvolution).
    It computes the deposition rate that would produce the observed concentration changes.

    Parameters
    ----------
    cout : array-like
        Concentration change values of extracted water [ng/m3].
        Length must match the number of time bins defined by tedges.
    flow : array-like
        Flow rate values in the aquifer [m3/day].
        Length must match cout and the number of time bins defined by tedges.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cout and flow data. Has length of
        len(cout) + 1 and len(flow) + 1.
    dep_tedges : pandas.DatetimeIndex
        Time edges for output (deposition) data bins. Has length of desired output + 1.
        Can have different time alignment and resolution than tedges.
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    porosity : float
        Porosity of the aquifer [dimensionless].
    thickness : float
        Thickness of the aquifer [m].
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).

    Returns
    -------
    numpy.ndarray
        Deposition rate of the compound in the aquifer [ng/m2/day].
        Length equals len(dep_tedges) - 1.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.deposition2 import extraction_to_deposition
    >>>
    >>> # Create input data
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Create deposition time edges
    >>> dep_dates = pd.date_range(start="2019-12-28", end="2020-01-08", freq="D")
    >>> dep_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dep_dates, number_of_bins=len(dep_dates)
    ... )
    >>>
    >>> # Input concentration changes and flow
    >>> cout = np.ones(len(dates))
    >>> flow = np.ones(len(dates)) * 100  # 100 m3/day
    >>>
    >>> # Aquifer properties
    >>> aquifer_pore_volume = 500.0  # m3
    >>> porosity = 0.3
    >>> thickness = 10.0  # m
    >>>
    >>> # Run extraction to deposition model
    >>> dep = extraction_to_deposition(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     dep_tedges=dep_tedges,
    ...     aquifer_pore_volume=aquifer_pore_volume,
    ...     porosity=porosity,
    ...     thickness=thickness,
    ... )
    >>> dep.shape
    (12,)
    """
    tedges = pd.DatetimeIndex(tedges)
    dep_tedges = pd.DatetimeIndex(dep_tedges)

    if len(tedges) != len(cout) + 1:
        msg = "tedges must have one more element than cout"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    # Convert to arrays for vectorized operations
    cout_values = np.asarray(cout)
    flow_values = np.asarray(flow)

    # Validate inputs do not contain NaN values
    if np.any(np.isnan(cout_values)):
        msg = "cout contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow_values)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)

    cout_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    dep_tedges_days = ((dep_tedges - tedges[0]) / pd.Timedelta(days=1)).values
    aquifer_pore_volume_value = float(aquifer_pore_volume_value)

    # Pre-compute residence times and deposition edges
    rt_edges = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=dep_tedges,
        aquifer_pore_volume=aquifer_pore_volume_value,
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",
    )
    dep_tedges_days_extraction = dep_tedges_days + rt_edges[0]

    flow_tdelta = np.diff(cout_tedges_days, prepend=0.0)
    flow_values = np.concatenate(([0.0], np.asarray(flow)))
    flow_cum = (flow_values * flow_tdelta).cumsum()

    start_volume_at_dep_tedges_days_infiltration = linear_interpolate(
        x_ref=cout_tedges_days,
        y_ref=flow_cum,
        x_query=dep_tedges_days,
    )
    end_volume_at_dep_tedges_days_extraction = linear_interpolate(
        x_ref=cout_tedges_days,
        y_ref=flow_cum,
        x_query=dep_tedges_days_extraction,
    )

    flow_cum_dep = flow_cum[None, :] - start_volume_at_dep_tedges_days_infiltration[:, None]

    # compute the volumes
    volume_array = compute_average_heights(
        x_edges=cout_tedges_days,
        y_edges=flow_cum_dep,
        y_lower=0.0,
        y_upper=retardation_factor * aquifer_pore_volume_value,
    )  # [m3]
    area_array = volume_array / (porosity * thickness)
    extracted_volume = np.diff(end_volume_at_dep_tedges_days_extraction)
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
    """
    Compute concentration in extracted water due to deposition.

    This function represents deposition to extraction modeling (equivalent to convolution).
    It computes the concentration that would result from given deposition rates.

    Deposition is computed as the residence time times the deposition divided by the porosity and the thickness.

    Parameters
    ----------
    dep : array-like
        Deposition rate values [ng/m2/day].
        Length must match the number of time bins defined by tedges.
    flow : array-like
        Flow rate values in the aquifer [m3/day].
        Length must match dep and the number of time bins defined by tedges.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both dep and flow data. Has length of
        len(dep) + 1 and len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output (concentration change) data bins. Has length of desired output + 1.
        Can have different time alignment and resolution than tedges.
    aquifer_pore_volume_value : float
        Pore volume of the aquifer [m3].
    porosity : float
        Porosity of the aquifer [dimensionless].
    thickness : float
        Thickness of the aquifer [m].
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).

    Returns
    -------
    numpy.ndarray
        Concentration change of the compound in the extracted water [ng/m3].
        Length equals len(cout_tedges) - 1.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.deposition2 import deposition_to_extraction
    >>>
    >>> # Create input data
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Create output time edges
    >>> cout_dates = pd.date_range(start="2020-01-03", end="2020-01-12", freq="D")
    >>> cout_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
    ... )
    >>>
    >>> # Input deposition and flow
    >>> dep = np.ones(len(dates))  # 1 ng/m2/day
    >>> flow = np.ones(len(dates)) * 100  # 100 m3/day
    >>>
    >>> # Aquifer properties
    >>> aquifer_pore_volume = 500.0  # m3
    >>> porosity = 0.3
    >>> thickness = 10.0  # m
    >>>
    >>> # Run deposition to extraction model
    >>> cout = deposition_to_extraction(
    ...     dep=dep,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volume=aquifer_pore_volume,
    ...     porosity=porosity,
    ...     thickness=thickness,
    ... )
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    if len(tedges) != len(dep) + 1:
        msg = "tedges must have one more element than dep"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    # Convert to arrays for vectorized operations
    dep_values = np.asarray(dep)
    flow_values = np.asarray(flow)
    aquifer_pore_volume_value = float(aquifer_pore_volume_value)

    # Validate inputs do not contain NaN values
    if np.any(np.isnan(dep_values)):
        msg = "dep contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow_values)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)

    tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # Pre-compute residence times and deposition edges
    rt_edges = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume_value,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    cout_tedges_days_infiltration = cout_tedges_days - rt_edges[0]

    flow_tdelta = np.diff(tedges_days, prepend=0.0)
    flow_values = np.concatenate(([0.0], np.asarray(flow)))
    flow_cum = (flow_values * flow_tdelta).cumsum()

    start_volume_at_cout_tedges_days_infiltration = linear_interpolate(
        x_ref=tedges_days,
        y_ref=flow_cum,
        x_query=cout_tedges_days_infiltration,
    )
    end_volume_at_cout_tedges_days_extraction = linear_interpolate(
        x_ref=tedges_days,
        y_ref=flow_cum,
        x_query=cout_tedges_days,
    )

    flow_cum_cout = flow_cum[None, :] - start_volume_at_cout_tedges_days_infiltration[:, None]

    # compute the volumes
    volume_array = compute_average_heights(
        x_edges=tedges_days, y_edges=flow_cum_cout, y_lower=0.0, y_upper=retardation_factor * aquifer_pore_volume_value
    )  # [m3]
    area_array = volume_array / (porosity * thickness)
    extracted_volume = np.diff(end_volume_at_cout_tedges_days_extraction)
    deposition_weights = area_array * np.diff(tedges_days)[None, :] / extracted_volume[:, None]

    return deposition_weights.dot(dep_values)
