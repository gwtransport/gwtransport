"""
Deposition Analysis for 1D Aquifer Systems.

This module models compound *source enrichment* in 1D groundwater flow: a deposition rate
[ng/m²/day] supplies mass from the aquifer matrix into the water along the flow path,
increasing the extracted concentration. The model is a *source* term (positive deposition
adds mass to the water); it does NOT model removal processes such as pathogen attachment,
particle filtration, or chemical precipitation, which would remove mass from the water and
require an opposite sign convention. Example physical scenarios: dissolution of a sparingly
soluble mineral coating from the matrix, leaching of a stored solute, mass release from a
distributed contaminant source on the matrix surface. The module follows advection module
patterns for consistency in forward (deposition to extraction) and reverse (extraction to
deposition) calculations.

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

- :func:`spinup_duration` - Compute spinup duration for deposition modeling. Returns the
  earliest extraction time at which the extracted water was infiltrated at the start of the
  flow series (equivalently, the time at which cumulative flow first reaches
  ``retardation_factor * aquifer_pore_volume``). Before this duration the extracted
  concentration lacks complete deposition history. Useful for determining the valid analysis
  period and identifying when boundary effects are negligible.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport._time import tedges_to_days
from gwtransport._validation import (
    _validate_no_nan,
    _validate_non_negative_array,
    _validate_positive_scalar,
    _validate_tedges_parity,
)
from gwtransport.advection_utils import _resolve_spinup_inputs
from gwtransport.deposition_utils import compute_average_heights
from gwtransport.residence_time import residence_time
from gwtransport.utils import (
    compute_reverse_target,
    cumulative_flow_volume,
    linear_interpolate,
    solve_tikhonov,
    solve_underdetermined_system,
)


def _validate_deposition_inputs(
    *,
    tedges: pd.DatetimeIndex,
    flow_values: np.ndarray,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
    cout_tedges: pd.DatetimeIndex | None = None,
    cout_values: np.ndarray | None = None,
    dep_values: np.ndarray | None = None,
) -> None:
    """Validate inputs common to deposition forward / reverse / full entry points.

    Activates checks per the kwargs that are not None:

    - ``dep_values`` provided => ``tedges``-parity vs ``dep`` + combined dep+flow
      NaN-check (forward path; preserves the historical "Input arrays cannot
      contain NaN values" message that covered both dep and flow).
    - ``cout_values`` + ``cout_tedges`` provided => ``cout_tedges``-parity check
      (inverse paths). ``cout_values`` itself is intentionally NOT NaN-checked
      -- NaN in ``cout`` is allowed and excluded downstream by ``solve_tikhonov``.
    - ``flow_values`` + ``tedges`` always => parity check + non-negative;
      additionally, in the inverse path (``dep_values is None``), a flow-only
      NaN-check fires with the historical "flow array cannot contain NaN
      values" message.
    - Physical params (``porosity``, ``thickness``, ``aquifer_pore_volume``,
      ``retardation_factor``) always validated.

    Every error message and f-string substitution is preserved verbatim from
    the prior triplicate prologue so that ``match=`` regex tests do not break.

    Raises
    ------
    ValueError
        If any of the activated checks fails. The specific message names which
        invariant was violated (see body for the verbatim strings).
    """
    if dep_values is not None:
        _validate_tedges_parity(tedges, dep_values, tedges_name="tedges", values_name="dep")
    _validate_tedges_parity(tedges, flow_values, tedges_name="tedges", values_name="flow")
    if cout_values is not None and cout_tedges is not None:
        _validate_tedges_parity(cout_tedges, cout_values, tedges_name="cout_tedges", values_name="cout")
    if dep_values is not None:
        # Compound NaN-check covers both ``dep`` and ``flow`` under one message; mapping to
        # two separate ``_validate_no_nan`` calls would change wording and which array name
        # surfaces first.
        if np.any(np.isnan(dep_values)) or np.any(np.isnan(flow_values)):
            msg = "Input arrays cannot contain NaN values"
            raise ValueError(msg)
    else:
        _validate_no_nan(flow_values, name="flow", message="flow array cannot contain NaN values")
    _validate_non_negative_array(
        flow_values, name="flow", message="flow must be non-negative (negative flow not supported)"
    )
    if not 0 < porosity < 1:
        msg = f"Porosity must be in (0, 1), got {porosity}"
        raise ValueError(msg)
    _validate_positive_scalar(thickness, name="thickness", message=f"Thickness must be positive, got {thickness}")
    _validate_positive_scalar(
        aquifer_pore_volume,
        name="aquifer_pore_volume",
        message=f"Aquifer pore volume must be positive, got {aquifer_pore_volume}",
    )
    if retardation_factor < 1.0:
        msg = "retardation_factor must be >= 1.0"
        raise ValueError(msg)


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
    tedges_days = tedges_to_days(tedges, ref=t0)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=t0)

    # Compute residence times and cumulative flow
    flow_values = np.asarray(flow)
    cout_rt_at_edges = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=float(aquifer_pore_volume),
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    cout_tedges_days_infiltration = cout_tedges_days - cout_rt_at_edges.squeeze(axis=0)

    flow_cum = cumulative_flow_volume(flow_values, np.diff(tedges_days))

    # Interpolate volumes at concentration time edges
    start_vol = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days_infiltration)
    end_vol = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days)

    # Compute deposition weights. Zero-flow cout bins have extracted_volume == 0;
    # np.divide with where=mask leaves those rows at the out=zeros sentinel.
    flow_cum_cout = flow_cum[None, :] - start_vol[:, None]
    volume_array = compute_average_heights(
        x_edges=tedges_days, y_edges=flow_cum_cout, y_lower=0.0, y_upper=retardation_factor * float(aquifer_pore_volume)
    )
    area_array = volume_array / (porosity * thickness)
    extracted_volume = np.diff(end_vol)
    numerator = area_array * np.diff(tedges_days)[None, :]
    mask = extracted_volume > 0
    return np.divide(numerator, extracted_volume[:, None], out=np.zeros_like(numerator), where=mask[:, None])


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
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute concentrations from deposition rates (convolution).

    Parameters
    ----------
    dep : array-like
        Deposition rates [ng/m2/day]. Length must equal len(tedges) - 1.
    flow : array-like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1. The model
        assumes this value is constant over each interval ``[tedges[i], tedges[i+1])``.
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
    spinup : {"constant"} | float in [0, 1] | None, optional
        Spin-up policy applied before computing deposition weights.
        Default ``"constant"`` shifts ``tedges[0]`` backward by
        ``retardation_factor * aquifer_pore_volume / flow[0]`` and treats
        ``dep`` and ``flow`` as constant at their first observed values
        over the prepended interval. ``None`` keeps the existing
        strict-validity behavior (NaN cout rows during spin-up). A float
        threshold has no effect with a single pore volume and behaves
        like ``None``.

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

    Notes
    -----
    This is a *source* term -- positive ``dep`` raises ``cout``. Sink
    processes (pathogen attachment, first-order decay, particle filtration)
    require the opposite sign convention and are not modelled here.

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

    _validate_deposition_inputs(
        tedges=tedges,
        flow_values=flow_values,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
        dep_values=dep_values,
    )

    # Apply spinup policy: optionally prepend warm-start bins to tedges/flow/dep.
    weight_tedges, weight_flow, weight_dep, _, _ = _resolve_spinup_inputs(
        spinup,
        tedges=tedges,
        flow=flow_values,
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        retardation_factor=retardation_factor,
        cin=dep_values,
    )
    weight_dep = np.asarray(weight_dep)

    # Compute deposition weights
    deposition_weights = compute_deposition_weights(
        flow=weight_flow,
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )

    return deposition_weights.dot(weight_dep)


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
    spinup: str | float | None = "constant",
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
        The model assumes this value is constant over each interval
        ``[cout_tedges[i], cout_tedges[i+1])``.
    flow : array-like
        Flow rates in aquifer [m3/day]. Length must equal len(tedges) - 1.
        Must not contain NaN values. The model assumes this value is constant
        over each interval ``[tedges[i], tedges[i+1])``.
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
    spinup : {"constant"} | float in [0, 1] | None, optional
        Spin-up policy applied before building the forward weight matrix.
        Default ``"constant"`` shifts ``tedges[0]`` backward by
        ``retardation_factor * aquifer_pore_volume / flow[0]`` and treats
        flow as constant at its first value over the prepended interval;
        the recovered deposition vector is sliced back to the original
        ``tedges`` length so the public output shape is unchanged.
        ``None`` keeps strict-validity behavior. A float threshold has no
        effect with a single pore volume and behaves like ``None``.

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
    This is a *source* term -- positive ``dep`` raises ``cout``. Sink
    processes (pathogen attachment, first-order decay, particle filtration)
    require the opposite sign convention and are not modelled here.

    The forward model is ``W @ dep = cout``, where the weight matrix ``W``
    encodes the physical relationship between deposition rates and
    concentrations. Unlike advection (where rows sum to ~1), deposition rows
    sum to ``r_i = residence_time_i / (porosity * thickness)``. Rows are
    rescaled by ``r_i`` before solving: for systems where ``cout`` lies in
    the column space of ``W`` this preserves the exact ``dep``, while for
    overdetermined systems with noise it is equivalent to weighted least
    squares with weights ``1 / r_i^2`` (shorter residence times get more
    weight; under constant flow all ``r_i`` are equal and this reduces to
    OLS). The rescaling exists to put ``compute_reverse_target`` on the same
    scale as ``dep``, which controls the regularization scale. Rows where
    the residence time cannot be computed (spin-up period) contain NaN and
    are excluded automatically. NaN values in ``cout`` are also excluded.

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

    _validate_deposition_inputs(
        tedges=tedges,
        flow_values=flow_values,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
        cout_tedges=cout_tedges,
        cout_values=cout_values,
    )

    # Apply spinup policy: optionally prepend warm-start bins to tedges/flow.
    weight_tedges, weight_flow, _, _, n_pad = _resolve_spinup_inputs(
        spinup,
        tedges=tedges,
        flow=flow_values,
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        retardation_factor=retardation_factor,
    )

    # Compute deposition weights
    deposition_weights = compute_deposition_weights(
        flow=weight_flow,
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )

    # Rescale rows of W by their sum r_i = RT_i / (porosity*thickness). For
    # systems where cout lies in the column space of W (e.g., noise-free
    # roundtrip), this preserves the exact dep. For overdetermined systems
    # with noise the rescaling is equivalent to weighted least squares with
    # weights 1/r_i^2 -- shorter residence times get more weight; under
    # constant flow all r_i are equal and this reduces to OLS. The rescaling
    # exists to put compute_reverse_target on the same scale as dep, which
    # controls the regularization scale. NaN rows (spin-up), all-zero rows
    # (zero-flow cout bins; carry no info), and NaN values in cout are
    # excluded by solve_tikhonov via its valid_rows = ~isnan(matrix).any(axis=1) & ~isnan(rhs) filter.
    valid_rows = ~np.isnan(deposition_weights).any(axis=1) & (np.abs(deposition_weights).sum(axis=1) > 0)
    valid_weights = deposition_weights[valid_rows]
    row_sums = valid_weights.sum(axis=1, keepdims=True)
    col_active = np.sum(np.abs(valid_weights), axis=0) > 0

    if not np.any(col_active):
        return np.full(deposition_weights.shape[1] - n_pad, np.nan)

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

    x_target = compute_reverse_target(coeff_matrix=w_norm, rhs_vector=cout_norm)

    # solve_tikhonov filters NaN rows in *both* coefficient_matrix and rhs_vector
    # (utils.py: valid_rows = ~isnan(matrix).any(axis=1) & ~isnan(rhs)).  We pass
    # the already-pruned w_norm / cout_norm directly -- a reconstruction back to
    # full size only to be filtered again would be dead work.
    dep_solved = solve_tikhonov(
        coefficient_matrix=w_norm,
        rhs_vector=cout_norm,
        x_target=x_target,
        regularization_strength=regularization_strength,
    )

    out = np.full(deposition_weights.shape[1], np.nan)
    out[col_active] = dep_solved[col_active]
    # Drop warm-start prefix so output aligns with the user-provided tedges.
    return out[n_pad:]


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
    spinup: str | float | None = "constant",
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
    spinup : {"constant"} | float in [0, 1] | None, optional
        Spin-up policy applied before building the forward weight matrix.
        Default ``"constant"`` shifts ``tedges[0]`` backward by
        ``retardation_factor * aquifer_pore_volume / flow[0]``; the
        recovered deposition is sliced back to the original ``tedges``
        length. See :func:`extraction_to_deposition` for full semantics.

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

    Notes
    -----
    This is a *source* term -- positive ``dep`` raises ``cout``. Sink
    processes (pathogen attachment, first-order decay, particle filtration)
    require the opposite sign convention and are not modelled here.
    """
    tedges, cout_tedges = pd.DatetimeIndex(tedges), pd.DatetimeIndex(cout_tedges)
    cout_values, flow_values = np.asarray(cout), np.asarray(flow)

    _validate_deposition_inputs(
        tedges=tedges,
        flow_values=flow_values,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
        cout_tedges=cout_tedges,
        cout_values=cout_values,
    )

    # Apply spinup policy: optionally prepend warm-start bins to tedges/flow.
    weight_tedges, weight_flow, _, _, n_pad = _resolve_spinup_inputs(
        spinup,
        tedges=tedges,
        flow=flow_values,
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        retardation_factor=retardation_factor,
    )

    # Compute deposition weights
    deposition_weights = compute_deposition_weights(
        flow=weight_flow,
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )

    dep_padded = solve_underdetermined_system(
        coefficient_matrix=deposition_weights,
        rhs_vector=cout_values,
        nullspace_objective=nullspace_objective,
        optimization_method=optimization_method,
        rcond=rcond,
    )
    # Drop warm-start prefix so output aligns with the user-provided tedges.
    return dep_padded[n_pad:]


def spinup_duration(
    *,
    flow: np.ndarray,
    flow_tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    retardation_factor: float,
) -> float:
    """
    Compute the spinup duration for deposition modeling.

    The spinup duration is the smallest extraction time ``t*`` (relative to
    ``flow_tedges[0]``) at which the extracted water was infiltrated exactly
    at ``flow_tedges[0]``: equivalently, the time at which the cumulative
    flow first reaches ``retardation_factor * aquifer_pore_volume``. For
    extraction times earlier than ``t*`` the extracted concentration lacks
    complete deposition history. Under constant flow this equals
    ``aquifer_pore_volume * retardation_factor / flow``.

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
        If the cumulative flow over the entire ``flow_tedges`` window does
        not reach ``retardation_factor * aquifer_pore_volume``, indicating
        the flow timeseries is too short to fully characterise the aquifer.
    """
    # Spin-up is the residence time of water *currently being extracted*: how
    # far back in history we must know deposition to fully characterise the
    # extracted concentration. This uses the ``extraction_to_infiltration``
    # direction. Under variable flow this differs from
    # ``infiltration_to_extraction`` (which would describe how long ahead
    # water infiltrated at the first time step will take to be extracted, a
    # forward-in-time question that is not what spin-up means).
    #
    # The smallest extraction time t* at which the extracted water was
    # infiltrated exactly at flow_tedges[0] satisfies
    # ``flow_cum(t*) = R * V_pore``; the spin-up duration is then
    # ``t* - 0 = t*``. Inverting the cumulative flow gives this value
    # exactly (no quantisation to flow_tedges spacing). Under constant flow
    # this matches V*R/Q.
    flow_arr = np.asarray(flow)
    flow_tedges_days = tedges_to_days(flow_tedges)
    dt_days = np.diff(flow_tedges_days)
    target_cum = retardation_factor * float(aquifer_pore_volume)
    # Feasibility guard on the *un-bumped* cumulative total: the request is infeasible iff
    # R*V_pore exceeds the true total infiltrated volume. (The monotone bump below would
    # otherwise lift a trailing Q=0 plateau above target_cum and admit an infeasible request.)
    flow_cum_raw = cumulative_flow_volume(flow_arr, dt_days)
    if not flow_cum_raw[-1] >= target_cum:
        msg = "Residence time at the first time step is NaN. This indicates that the aquifer is not fully informed: flow timeseries too short."
        raise ValueError(msg)
    # Plateaus in flow_cum from Q = 0 bins make V → t inversion multi-valued; bump duplicates
    # by the smallest representable amount so np.interp resolves consistently at plateau levels.
    flow_cum = cumulative_flow_volume(flow_arr, dt_days, strictly_monotone=True)
    rt_value = float(linear_interpolate(x_ref=flow_cum, y_ref=flow_tedges_days, x_query=target_cum))
    if np.isnan(rt_value):
        msg = "Residence time at the first time step is NaN. This indicates that the aquifer is not fully informed: flow timeseries too short."
        raise ValueError(msg)
    return rt_value
