"""
Deposition Analysis for 1D Aquifer Systems.

This module analyzes compound transport by deposition in aquifer systems with tools for
computing concentrations and deposition rates based on aquifer properties. The model assumes
1D groundwater flow where compound deposition occurs along the flow path, enriching the water.
Deposition processes include pathogen attachment to aquifer matrix, particle filtration, or
chemical precipitation. The module follows advection module patterns for consistency in
forward (deposition to extraction) and reverse (extraction to deposition) calculations.

Available functions:

- :func:`deposition_to_extraction` - Compute concentrations from deposition rates (convolution).
  Given deposition rate time series [ng/m²/day], computes resulting concentration changes in
  extracted water [ng/m³]. Uses flow-weighted integration over contact area between water and
  aquifer matrix. Accounts for aquifer geometry (porosity, thickness) and residence time
  distribution.

- :func:`extraction_to_deposition` - Compute deposition rates from concentration changes
  (deconvolution). Given concentration change time series in extracted water [ng/m³], estimates
  deposition rate history [ng/m²/day] that produced those changes. Solves underdetermined
  inverse problem using nullspace regularization with configurable objectives ('squared_differences'
  for smooth solutions, 'summed_differences' for sparse solutions). Handles NaN values in
  concentration data by excluding corresponding time periods.

- :func:`compute_deposition_weights` - Internal helper function. Compute weight matrix relating
  deposition rates to concentration changes. Used by both deposition_to_extraction (forward) and
  extraction_to_deposition (reverse). Calculates contact area between water parcels and aquifer
  matrix based on streamline geometry and residence times.

- :func:`spinup_duration` - Compute spinup duration for deposition modeling. Returns residence
  time at first time step, representing time needed for system to become fully informed. Before
  this duration, extracted concentration lacks complete deposition history. Useful for determining
  valid analysis period and identifying when boundary effects are negligible.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.residence_time import residence_time
from gwtransport.surfacearea import compute_average_heights
from gwtransport.utils import linear_interpolate, solve_underdetermined_system


def compute_deposition_weights(
    *,
    flow_values: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Compute deposition weights for concentration-deposition convolution.

    Parameters
    ----------
    flow_values : array-like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges for flow data.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for output concentration data.
    aquifer_pore_volume : float
        Aquifer pore volume [m3].
    porosity : float
        Aquifer porosity [dimensionless].
    thickness : float
        Aquifer thickness [m].
    retardation_factor : float, optional
        Compound retardation factor, by default 1.0.

    Returns
    -------
    numpy.ndarray
        Deposition weights matrix with shape (len(cout_tedges) - 1, len(tedges) - 1).
        May contain NaN values where residence time cannot be computed.

    Notes
    -----
    The returned weights matrix may contain NaN values in locations where the
    residence time calculation fails or is undefined. This typically occurs
    when flow conditions result in invalid or non-physical residence times.
    """
    # Convert to days relative to first time edge
    t0 = tedges[0]
    tedges_days = ((tedges - t0) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - t0) / pd.Timedelta(days=1)).values

    # Compute residence times and cumulative flow
    rt_edges = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=float(aquifer_pore_volume),
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    cout_tedges_days_infiltration = cout_tedges_days - rt_edges[0]

    flow_tdelta = np.diff(tedges_days, prepend=0.0)
    flow_cum = (np.concatenate(([0.0], flow_values)) * flow_tdelta).cumsum()

    # Interpolate volumes at concentration time edges
    start_vol = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days_infiltration)
    end_vol = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days)

    # Compute deposition weights
    flow_cum_cout = flow_cum[None, :] - start_vol[:, None]
    volume_array = compute_average_heights(
        x_edges=tedges_days, y_edges=flow_cum_cout, y_lower=0.0, y_upper=retardation_factor * float(aquifer_pore_volume)
    )
    area_array = volume_array / (porosity * thickness)
    extracted_volume = np.diff(end_vol)
    return area_array * np.diff(tedges_days)[None, :] / extracted_volume[:, None]


def deposition_to_extraction(
    *,
    dep: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Compute concentrations from deposition rates (convolution).

    Parameters
    ----------
    dep : array-like
        Deposition rates [ng/m2/day]. Length must equal len(tedges) - 1.
    flow : array-like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges for deposition and flow data.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for output concentration data.
    aquifer_pore_volume : float
        Aquifer pore volume [m3].
    porosity : float
        Aquifer porosity [dimensionless].
    thickness : float
        Aquifer thickness [m].
    retardation_factor : float, optional
        Compound retardation factor, by default 1.0.

    Returns
    -------
    numpy.ndarray
        Concentration changes [ng/m3] with length len(cout_tedges) - 1.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.deposition import deposition_to_extraction
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
    ...     aquifer_pore_volume=500.0,
    ...     porosity=0.3,
    ...     thickness=10.0,
    ... )
    """
    tedges, cout_tedges = pd.DatetimeIndex(tedges), pd.DatetimeIndex(cout_tedges)
    dep_values, flow_values = np.asarray(dep), np.asarray(flow)

    # Validate input dimensions and values
    if len(tedges) != len(dep_values) + 1:
        _msg = "tedges must have one more element than dep"
        raise ValueError(_msg)
    if len(tedges) != len(flow_values) + 1:
        _msg = "tedges must have one more element than flow"
        raise ValueError(_msg)
    if np.any(np.isnan(dep_values)) or np.any(np.isnan(flow_values)):
        _msg = "Input arrays cannot contain NaN values"
        raise ValueError(_msg)

    # Validate physical parameters
    if not 0 < porosity < 1:
        _msg = f"Porosity must be in (0, 1), got {porosity}"
        raise ValueError(_msg)
    if thickness <= 0:
        _msg = f"Thickness must be positive, got {thickness}"
        raise ValueError(_msg)
    if aquifer_pore_volume <= 0:
        _msg = f"Aquifer pore volume must be positive, got {aquifer_pore_volume}"
        raise ValueError(_msg)

    # Compute deposition weights
    deposition_weights = compute_deposition_weights(
        flow_values=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )

    return deposition_weights.dot(dep_values)


def extraction_to_deposition(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout: npt.ArrayLike,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
    nullspace_objective: str = "squared_differences",
) -> npt.NDArray[np.floating]:
    """
    Compute deposition rates from concentration changes (deconvolution).

    The solution for the deposition is fundamentally underdetermined, as multiple
    deposition histories can lead to the same concentration. This function
    computes a least-squares solution and then selects a specific solution from the
    nullspace of the problem based on the provided objective.

    Parameters
    ----------
    flow : array-like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1.
        Must not contain NaN values.
    tedges : pandas.DatetimeIndex
        Time bin edges for deposition and flow data. Length must equal
        len(flow) + 1.
    cout : array-like
        Concentration changes in extracted water [ng/m3]. Length must equal
        len(cout_tedges) - 1. May contain NaN values, which will be excluded
        from the computation along with corresponding rows in the weight matrix.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for output concentration data. Length must equal
        len(cout) + 1.
    aquifer_pore_volume : float
        Aquifer pore volume [m3].
    porosity : float
        Aquifer porosity [dimensionless].
    thickness : float
        Aquifer thickness [m].
    retardation_factor : float, optional
        Compound retardation factor, by default 1.0. Values > 1.0 indicate
        slower transport due to sorption/interaction.
    nullspace_objective : str or callable, optional
        Objective function to minimize in the nullspace. Options:

        * "squared_differences" : Minimize sum of squared differences between
          adjacent deposition rates (default, provides smooth solutions)
        * "summed_differences" : Minimize sum of absolute differences between
          adjacent deposition rates (promotes sparse/piecewise constant solutions)
        * callable : Custom objective function with signature
          ``objective(coeffs, x_ls, nullspace_basis)``

        Default is "squared_differences".

    Returns
    -------
    numpy.ndarray
        Mean deposition rates [ng/m2/day] between tedges. Length equals
        len(tedges) - 1.

    Raises
    ------
    ValueError
        If input dimensions are incompatible, if flow contains NaN values,
        or if the optimization fails.

    Notes
    -----
    This function solves the inverse problem of determining deposition rates
    from observed concentration changes. Since multiple deposition histories
    can produce the same concentration pattern, regularization in the nullspace
    is used to select a physically meaningful solution.

    The algorithm:

    1. Validates input dimensions and checks for NaN values in flow
    2. Computes deposition weight matrix relating deposition to concentration
    3. Identifies valid rows (no NaN in weights or concentration data)
    4. Solves the underdetermined system using nullspace regularization
    5. Returns the regularized deposition rate solution

    Examples
    --------
    Basic usage with default squared differences regularization:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.deposition import extraction_to_deposition
    >>>
    >>> # Create input data
    >>> dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    >>> tedges = pd.date_range("2019-12-31 12:00", "2020-01-10 12:00", freq="D")
    >>> cout_tedges = pd.date_range("2020-01-03 12:00", "2020-01-12 12:00", freq="D")
    >>>
    >>> # Flow and concentration data
    >>> flow = np.full(len(dates), 100.0)  # m3/day
    >>> cout = np.ones(len(cout_tedges) - 1) * 10.0  # ng/m3
    >>>
    >>> # Compute deposition rates
    >>> dep = extraction_to_deposition(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout=cout,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volume=500.0,
    ...     porosity=0.3,
    ...     thickness=10.0,
    ... )
    >>> print(f"Deposition rates shape: {dep.shape}")
    Deposition rates shape: (10,)
    >>> print(f"Mean deposition rate: {np.nanmean(dep):.2f} ng/m2/day")
    Mean deposition rate: 6.00 ng/m2/day

    With summed differences regularization for sparse solutions:

    >>> dep_sparse = extraction_to_deposition(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout=cout,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volume=500.0,
    ...     porosity=0.3,
    ...     thickness=10.0,
    ...     nullspace_objective="summed_differences",
    ... )

    With custom regularization objective:

    >>> def l2_norm_objective(coeffs, x_ls, nullspace_basis):
    ...     x = x_ls + nullspace_basis @ coeffs
    ...     return np.sum(x**2)  # Minimize L2 norm of solution
    >>>
    >>> dep_l2 = extraction_to_deposition(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout=cout,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volume=500.0,
    ...     porosity=0.3,
    ...     thickness=10.0,
    ...     nullspace_objective=l2_norm_objective,
    ... )

    Handling missing concentration data:

    >>> # Concentration with some NaN values
    >>> cout_nan = cout.copy()
    >>> cout_nan[2:4] = np.nan  # Missing data for some time periods
    >>>
    >>> dep_robust = extraction_to_deposition(  # doctest: +SKIP
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout=cout_nan,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volume=500.0,
    ...     porosity=0.3,
    ...     thickness=10.0,
    ... )
    """
    tedges, cout_tedges = pd.DatetimeIndex(tedges), pd.DatetimeIndex(cout_tedges)
    cout_values, flow_values = np.asarray(cout), np.asarray(flow)

    # Validate input dimensions and values
    if len(cout_tedges) != len(cout_values) + 1:
        msg = "cout_tedges must have one more element than cout"
        raise ValueError(msg)
    if len(tedges) != len(flow_values) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if np.any(np.isnan(flow_values)):
        msg = "flow array cannot contain NaN values"
        raise ValueError(msg)

    # Validate physical parameters
    if not 0 < porosity < 1:
        msg = f"Porosity must be in (0, 1), got {porosity}"
        raise ValueError(msg)
    if thickness <= 0:
        msg = f"Thickness must be positive, got {thickness}"
        raise ValueError(msg)
    if aquifer_pore_volume <= 0:
        msg = f"Aquifer pore volume must be positive, got {aquifer_pore_volume}"
        raise ValueError(msg)

    # Compute deposition weights
    deposition_weights = compute_deposition_weights(
        flow_values=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )

    # Solve underdetermined system using utils function
    return solve_underdetermined_system(
        coefficient_matrix=deposition_weights,
        rhs_vector=cout_values,
        nullspace_objective=nullspace_objective,
        optimization_method="Nelder-Mead",
    )


def spinup_duration(
    *,
    flow: np.ndarray,
    flow_tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    retardation_factor: float,
) -> float:
    """
    Compute the spinup duration for deposition modeling.

    The spinup duration is the residence time at the first time step, representing
    the time needed for the system to become fully informed. Before this duration,
    the extracted concentration lacks complete deposition history.

    Parameters
    ----------
    flow : numpy.ndarray
        Flow rate of water in the aquifer [m3/day].
    flow_tedges : pandas.DatetimeIndex
        Time edges for the flow data.
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    retardation_factor : float
        Retardation factor of the compound in the aquifer [dimensionless].

    Returns
    -------
    float
        Spinup duration in days.
    """
    rt = residence_time(
        flow=flow,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",
    )
    rt_value: float = float(np.asarray(rt[0, 0]))
    if np.isnan(rt_value):
        msg = "Residence time at the first time step is NaN. This indicates that the aquifer is not fully informed: flow timeseries too short."
        raise ValueError(msg)

    # Return the first residence time value
    return rt_value
