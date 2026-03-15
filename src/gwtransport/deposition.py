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
  deposition rate history [ng/m²/day] that produced those changes. Uses Tikhonov regularization
  toward a physically motivated target (transpose-and-normalize of the forward matrix). Handles
  NaN values in concentration data by excluding corresponding time periods.

- :func:`extraction_to_deposition_full` - Full-featured inverse solver exposing all options of
  the nullspace-based solver (:func:`~gwtransport.utils.solve_underdetermined_system`). Allows
  choosing between different nullspace objectives (``'squared_differences'``,
  ``'summed_differences'``, or custom callables) and optimization methods.

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

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.residence_time import residence_time
from gwtransport.surfacearea import compute_average_heights
from gwtransport.utils import compute_reverse_target, linear_interpolate, solve_tikhonov, solve_underdetermined_system


def compute_deposition_weights(
    *,
    flow: npt.ArrayLike,
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
    flow : array-like
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
    flow_values = np.asarray(flow)
    rt_edges = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=float(aquifer_pore_volume),
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    cout_tedges_days_infiltration = cout_tedges_days - rt_edges[0]

    flow_cum = np.concatenate(([0.0], np.cumsum(flow_values * np.diff(tedges_days))))

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

    Raises
    ------
    ValueError
        If tedges does not have one more element than dep or flow, if input
        arrays contain NaN values, or if physical parameters are out of
        valid range (porosity not in (0, 1), non-positive thickness or
        aquifer pore volume).

    See Also
    --------
    extraction_to_deposition : Inverse operation (deconvolution)
    gwtransport.advection.infiltration_to_extraction : For concentration transport without deposition
    :ref:`concept-transport-equation` : Flow-weighted averaging approach

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
        msg = "tedges must have one more element than dep"
        raise ValueError(msg)
    if len(tedges) != len(flow_values) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if np.any(np.isnan(dep_values)) or np.any(np.isnan(flow_values)):
        msg = "Input arrays cannot contain NaN values"
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
        flow=flow_values,
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
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
    regularization_strength: float = 1e-10,
) -> npt.NDArray[np.floating]:
    """Compute deposition rates from concentration changes (deconvolution).

    Inverts the forward model by solving ``W @ dep = cout`` where ``W`` is
    the weight matrix from :func:`compute_deposition_weights`. Uses Tikhonov
    regularization to smoothly blend data fitting with a physically motivated
    target (transpose-and-normalize of the forward matrix).

    Well-determined modes (large singular values relative to ``sqrt(λ)``) are
    dominated by the data; poorly-determined modes are pulled toward the
    target.

    Parameters
    ----------
    cout : array-like
        Concentration changes in extracted water [ng/m3]. Length must equal
        len(cout_tedges) - 1. May contain NaN values, which will be excluded
        from the computation along with corresponding rows in the weight matrix.
    flow : array-like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1.
        Must not contain NaN values.
    tedges : pandas.DatetimeIndex
        Time bin edges for deposition and flow data. Length must equal
        len(flow) + 1.
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
    regularization_strength : float, optional
        Tikhonov regularization parameter λ. Controls the tradeoff between
        fitting the data (``||W dep - cout||²``) and staying close to the
        regularization target (``λ ||dep - dep_target||²``). The target is
        the transpose-and-normalize of the forward matrix applied to cout.

        Larger values trust the target more (smoother, more biased); smaller
        values trust the data more (noisier, less biased). Default is 1e-10.

    Returns
    -------
    numpy.ndarray
        Mean deposition rates [ng/m2/day] between tedges. Length equals
        len(tedges) - 1.

    Raises
    ------
    ValueError
        If input dimensions are incompatible or if flow contains NaN values.

    Warns
    -----
    UserWarning
        When the weight matrix is rank-deficient. This occurs with constant
        flow when the residence time is an integer multiple of the time step
        width. The deposition weight matrix then acts as a uniform moving
        average whose transfer function has exact zeros, making certain
        deposition patterns invisible in the concentration signal. To fix,
        adjust ``aquifer_pore_volume`` slightly (e.g., multiply by 1.001).

    See Also
    --------
    deposition_to_extraction : Forward operation (convolution)
    extraction_to_deposition_full : Full solver with nullspace options
    gwtransport.advection.extraction_to_infiltration : For concentration transport without deposition
    gwtransport.utils.solve_tikhonov : Solver used for inversion
    :ref:`concept-transport-equation` : Flow-weighted averaging approach

    Notes
    -----
    The forward model is ``W @ dep = cout``, where the weight matrix ``W``
    encodes the physical relationship between deposition rates and
    concentrations. Unlike advection (where rows sum to ~1), deposition rows
    sum to ``residence_time / (porosity * thickness)``. The system is
    row-normalized before solving so that each observation contributes equally
    and ``compute_reverse_target`` gives the correct scale for the
    regularization target. Rows where the residence time cannot be computed
    (spin-up period) contain NaN and are excluded automatically. NaN values
    in ``cout`` are also excluded.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.deposition import extraction_to_deposition
    >>>
    >>> dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    >>> tedges = pd.date_range("2019-12-31 12:00", "2020-01-10 12:00", freq="D")
    >>> cout_tedges = pd.date_range("2020-01-03 12:00", "2020-01-12 12:00", freq="D")
    >>>
    >>> flow = np.full(len(dates), 100.0)  # m3/day
    >>> cout = np.ones(len(cout_tedges) - 1) * 10.0  # ng/m3
    >>>
    >>> dep = extraction_to_deposition(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volume=500.0,
    ...     porosity=0.3,
    ...     thickness=10.0,
    ... )
    >>> print(f"Deposition rates shape: {dep.shape}")
    Deposition rates shape: (10,)
    >>> print(f"Mean deposition rate: {np.nanmean(dep):.2f} ng/m2/day")
    Mean deposition rate: 6.00 ng/m2/day
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
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )

    n_dep = len(tedges) - 1

    # Normalize weight matrix rows to sum to 1. The deposition weight matrix
    # W has row sums equal to residence_time/(porosity*thickness), not 1 like
    # advection. Normalizing makes each observation equally important and
    # gives compute_reverse_target the correct scale for the target.
    # NaN rows (spin-up) and NaN values in cout are excluded by solve_tikhonov.
    valid_rows = ~np.isnan(deposition_weights).any(axis=1)
    valid_weights = deposition_weights[valid_rows]
    row_sums = valid_weights.sum(axis=1, keepdims=True)
    col_active = np.sum(np.abs(valid_weights), axis=0) > 0

    if not np.any(col_active):
        return np.full(n_dep, np.nan)

    # Build normalized system: W_norm @ dep = cout_norm
    w_norm = valid_weights / row_sums
    cout_norm = cout_values[valid_rows] / row_sums.ravel()

    # Check for rank deficiency. With constant flow and integer RT/dt, the
    # deposition weight matrix acts as a uniform moving average whose transfer
    # function has exact zeros, making certain deposition patterns invisible.
    n_active = int(col_active.sum())
    rank = np.linalg.matrix_rank(w_norm[:, col_active])
    if rank < n_active:
        warnings.warn(
            f"Weight matrix is rank-deficient (rank {rank} < {n_active} active "
            f"columns). This occurs with constant flow when the residence time "
            f"is an integer multiple of the time step width, creating deposition "
            f"patterns that are invisible in the concentration signal. The "
            f"underdetermined modes will be pulled toward the regularization "
            f"target instead of being determined by data. To achieve full rank, "
            f"adjust aquifer_pore_volume slightly (e.g., multiply by 1.001).",
            stacklevel=2,
        )

    # Reconstruct full arrays with NaN for invalid rows
    full_w_norm = np.full_like(deposition_weights, np.nan)
    full_w_norm[valid_rows] = w_norm
    full_cout_norm = np.full(len(cout_values), np.nan)
    full_cout_norm[valid_rows] = cout_norm

    x_target = compute_reverse_target(coeff_matrix=w_norm, rhs_vector=cout_norm)

    dep_solved = solve_tikhonov(
        coefficient_matrix=full_w_norm,
        rhs_vector=full_cout_norm,
        x_target=x_target,
        regularization_strength=regularization_strength,
    )

    out = np.full(n_dep, np.nan)
    out[col_active] = dep_solved[col_active]
    return out


def extraction_to_deposition_full(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
    nullspace_objective: str | Callable = "squared_differences",
    optimization_method: str = "BFGS",
    rcond: float | None = None,
    x_target: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Compute deposition rates from concentration changes using nullspace solver.

    Full-featured inverse solver exposing all options of
    :func:`~gwtransport.utils.solve_underdetermined_system`. For most use
    cases, prefer :func:`extraction_to_deposition` which uses Tikhonov
    regularization.

    Parameters
    ----------
    cout : array-like
        Concentration changes in extracted water [ng/m3]. Length must equal
        len(cout_tedges) - 1. May contain NaN values, which will be excluded
        from the computation along with corresponding rows in the weight matrix.
    flow : array-like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1.
        Must not contain NaN values.
    tedges : pandas.DatetimeIndex
        Time bin edges for deposition and flow data. Length must equal
        len(flow) + 1.
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
        Compound retardation factor, by default 1.0.
    nullspace_objective : str or callable, optional
        Objective function to minimize in the nullspace. Options:

        * ``"squared_differences"`` : Minimize sum of squared differences
          between adjacent deposition rates (default, smooth solutions).
        * ``"summed_differences"`` : Minimize sum of absolute differences
          (sparse/piecewise constant solutions).
        * callable : Custom objective ``f(coeffs, x_ls, nullspace_basis)``.

    optimization_method : str, optional
        Scipy optimization method. Default is ``"BFGS"``.
    rcond : float or None, optional
        Cutoff for small singular values in the least-squares step.
        Default is None (uses numpy default).
    x_target : ndarray or None, optional
        Optional target solution for the nullspace optimization.
        Default is None.

    Returns
    -------
    numpy.ndarray
        Mean deposition rates [ng/m2/day] between tedges. Length equals
        len(tedges) - 1.

    Raises
    ------
    ValueError
        If cout_tedges does not have one more element than cout, if tedges
        does not have one more element than flow, if flow contains NaN
        values, or if physical parameters are out of valid range (porosity
        not in (0, 1), non-positive thickness or aquifer pore volume).

    See Also
    --------
    extraction_to_deposition : Recommended solver using Tikhonov regularization.
    gwtransport.utils.solve_underdetermined_system : Underlying solver.
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
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )

    return solve_underdetermined_system(
        coefficient_matrix=deposition_weights,
        rhs_vector=cout_values,
        nullspace_objective=nullspace_objective,
        optimization_method=optimization_method,
        rcond=rcond,
        x_target=x_target,
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

    Raises
    ------
    ValueError
        If the residence time at the first time step is NaN, indicating the
        flow timeseries is too short to fully characterise the aquifer.
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
