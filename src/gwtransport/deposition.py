"""
Deposition Analysis for 1D Aquifer Systems.

Areal deposition supplies mass to the groundwater, mixed instantaneously over the height of the
aquifer. The aquifer has a constant thickness with a finite pore volume; water with zero
concentration infiltrates at one end and is extracted at the other, whether the flow is radial or
orthogonal. Transport is 1D advection with linear sorption; there is no microdispersion, molecular
diffusion, or macrodispersion. Forward and backward modeling are supported.

The model is a *source* term (positive deposition adds mass to the water); it does NOT model removal
processes such as pathogen attachment, particle filtration, or chemical precipitation, which would
remove mass from the water and require the opposite sign convention.

Available functions:

- :func:`deposition_to_extraction` - Compute concentrations from deposition rates (convolution).
  Given deposition rate time series [g/m²/day], computes resulting concentration changes in
  extracted water [g/m³]. The areal deposition flux is mixed instantaneously over the aquifer
  thickness, so a parcel's concentration gain is proportional to its residence time. Accounts for
  aquifer geometry (porosity, thickness) and residence time distribution.

- :func:`extraction_to_deposition` - Compute deposition rates from concentration changes
  (deconvolution). Given concentration change time series in extracted water [g/m³], estimates
  deposition rate history [g/m²/day] that produced those changes. Uses Tikhonov regularization
  toward a physically motivated target (transpose-and-normalize of the forward matrix). Handles
  NaN values in concentration data by excluding corresponding time periods.

- :func:`extraction_to_deposition_full` - Full-featured inverse solver exposing all options of
  the nullspace-based solver (:func:`~gwtransport.utils.solve_underdetermined_system`). Allows
  choosing between different nullspace objectives (``'squared_differences'``,
  ``'summed_differences'``, or custom callables) and optimization methods.

- :func:`compute_deposition_weights` - Build the banded weight operator relating deposition
  rates to concentration changes in a compact banded layout. Useful for custom inverse solvers.
  Used by deposition_to_extraction (forward), extraction_to_deposition (reverse), and
  extraction_to_deposition_full. Each weight is a water parcel's residence-time contribution to
  its concentration gain under areal deposition mixed over the aquifer thickness, independent of
  whether the flow geometry is radial or orthogonal.

- :func:`spinup_duration` - Compute spinup duration for deposition modeling. Returns the
  earliest extraction time at which the extracted water was infiltrated at the start of the
  flow series (equivalently, the time at which cumulative flow first reaches
  ``retardation_factor * aquifer_pore_volume``). Before this duration the extracted
  concentration lacks complete deposition history. Useful for determining the valid analysis
  period and identifying when boundary effects are negligible.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport._time import tedges_to_days
from gwtransport._validation import (
    _validate_no_nan,
    _validate_non_negative_array,
    _validate_positive_scalar,
    _validate_retardation_factor,
    _validate_tedges_parity,
)
from gwtransport.advection_utils import _densify_weights, _resolve_spinup_inputs
from gwtransport.deposition_utils import _clipped_linear_integral
from gwtransport.utils import (
    _make_strictly_monotone,
    cumulative_flow_volume,
    linear_interpolate,
    solve_inverse_transport_banded,
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
    spinup: str | None = "constant",
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
      -- NaN in ``cout`` is allowed and excluded downstream by the inverse solve.
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
    NotImplementedError
        If ``spinup`` is anything other than ``None`` or ``"constant"``. The
        fraction-threshold mode is not implemented for deposition (matching the
        diffusion family); floats, ints, and bools are all rejected.
    """
    if spinup is not None and spinup != "constant":
        # Accept only None and "constant"; reject everything else (floats, ints,
        # bools, typo'd strings). Checking isinstance(spinup, float) alone let an
        # int threshold (e.g. spinup=0 or 1) slip through and be silently ignored
        # -- the fraction-threshold value from _resolve_spinup_inputs is discarded
        # by the deposition callers, so the request had no effect.
        msg = (
            "deposition's spinup parameter only supports None or 'constant'; "
            f"other values are not yet implemented (got {spinup!r})"
        )
        raise NotImplementedError(msg)
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
    _validate_retardation_factor(retardation_factor)


def compute_deposition_weights(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.intp],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]:
    """Build the deposition weight operator in a compact banded layout.

    Row ``k`` of the dense ``(n_cout, n_cin)`` operator is ``band_vals[k]``
    placed at columns ``[col_start[k], col_start[k] + full_band)``. The operator
    is genuinely banded -- row ``k`` is nonzero only on the cin bins whose
    cumulative through-flow volume lies in the residence-time window
    ``[min(start_vol_k, start_vol_{k+1}), max(start_vol_k, start_vol_{k+1}) +
    R * aquifer_pore_volume]`` -- so each band has at most ``full_band`` slots,
    bounded by ``R * aquifer_pore_volume`` in volume (independent of record
    length ``n_cin``). The window is located by :func:`numpy.searchsorted` on the
    cumulative flow volume ``flow_cum``; the per-cell math reuses
    ``gwtransport.deposition_utils._clipped_linear_integral`` restricted to
    the band columns, so each row sums to
    ``r_k = residence_time_k / (retardation_factor * porosity * thickness)``.
    Reconstruct the dense
    ``(n_cout, n_cin)`` matrix with
    ``gwtransport.advection_utils._densify_weights`` when a dense operator
    is required (the nullspace inverse).

    Parameters
    ----------
    flow : array-like
        Flow rates in aquifer [m³/day]. Length must equal ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Time bin edges for flow data.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for output concentration data.
    aquifer_pore_volume : float
        Aquifer pore volume [m³].
    porosity : float
        Aquifer porosity [dimensionless].
    thickness : float
        Aquifer thickness [m].
    retardation_factor : float, optional
        Compound retardation factor, by default 1.0.

    Returns
    -------
    band_vals : numpy.ndarray
        Banded weights of shape ``(n_cout, full_band)``. Slot ``band_vals[k, b]``
        is the weight on cin bin ``col_start[k] + b``. Row ``k`` sums to
        ``r_k = residence_time_k / (retardation_factor * porosity * thickness)``;
        invalid rows (NaN residence time, zero-flow cout bins) are zero.
    col_start : numpy.ndarray of int
        First cin bin index of each cout row's band, shape ``(n_cout,)``.
    row_valid : numpy.ndarray of bool
        True for cout bins whose residence-time window is fully defined and
        carries flow (the finite, nonzero rows), shape ``(n_cout,)``.
    spinup_row : numpy.ndarray of bool
        True for cout bins whose residence time is undefined (spin-up period),
        shape ``(n_cout,)``. These rows carry an all-zero band; the forward
        path returns NaN for these bins (distinct from zero-flow cout bins,
        which return 0).

    See Also
    --------
    ``gwtransport.advection_utils._densify_weights`` : Reconstruct the dense matrix.
    """
    t0 = tedges[0]
    tedges_days = tedges_to_days(tedges, ref=t0)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=t0)

    flow_values = np.asarray(flow, dtype=float)
    flow_cum = cumulative_flow_volume(flow_values, np.diff(tedges_days))
    end_vol = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days)
    r_apv = retardation_factor * float(aquifer_pore_volume)

    # Infiltration-side cumulative volume of each cout edge is its extraction-side volume minus the
    # retarded pore volume -- the direct cumulative-volume identity, avoiding the residence-time
    # round-trip. NaN where the cout edge is outside the flow record or the look-back precedes the
    # record start (the spin-up NaN the round-trip produced).
    in_record = (cout_tedges_days >= tedges_days[0]) & (cout_tedges_days <= tedges_days[-1])
    start_vol = end_vol - r_apv
    start_vol = np.where(in_record & (start_vol >= flow_cum[0]), start_vol, np.nan)

    n_cin = len(tedges) - 1
    n_cout = len(cout_tedges) - 1
    extracted_volume = np.diff(end_vol)
    dt = np.diff(tedges_days)

    # Row k's clipped trapezoid spans cout edges k (top: start_vol[k]) and k+1
    # (bottom: start_vol[k+1]). It is nonzero only on cin bins j whose cumulative
    # volume window [flow_cum[j], flow_cum[j+1]] overlaps the clip window
    # [lower_k, upper_k] in flow_cum space, located by searchsorted.
    sv_top, sv_bot = start_vol[:-1], start_vol[1:]
    nan_row = np.isnan(sv_top) | np.isnan(sv_bot)
    lower = np.minimum(sv_top, sv_bot)
    upper = np.maximum(sv_top, sv_bot) + r_apv
    j_first = np.clip(np.searchsorted(flow_cum, np.where(nan_row, flow_cum[0], lower), side="right") - 1, 0, n_cin - 1)
    j_last = np.clip(np.searchsorted(flow_cum, np.where(nan_row, flow_cum[0], upper), side="left") - 1, 0, n_cin - 1)
    width = np.where(nan_row, 0, j_last - j_first + 1)
    full_band = int(max(1, width.max(initial=0)))

    col_start = np.where(nan_row, 0, j_first).astype(np.intp)
    row_valid = ~nan_row & (extracted_volume > 0)
    # A row goes NaN in the forward only when its residence time is undefined AND
    # it extracts water (left-edge spin-up). Out-of-range rows have undefined
    # residence but zero extracted volume; they stay at the zeros sentinel (0).
    spinup_row = nan_row & (extracted_volume > 0)

    # Gather each row's band of cin edges (full_band + 1 edges) and bin widths,
    # then evaluate the clipped-trapezoid integral on those columns only. Out-of-
    # range / right-pad slots gather the last edge (zero-width contribution).
    band_edge = col_start[:, None] + np.arange(full_band + 1)[None, :]
    band_edge_clipped = np.clip(band_edge, 0, n_cin)
    y_at_edge = flow_cum[band_edge_clipped]  # (n_cout, full_band + 1)
    y_top = y_at_edge - sv_top[:, None]
    y_bot = y_at_edge - sv_bot[:, None]
    widths = dt[np.clip(band_edge[:, :-1], 0, n_cin - 1)]
    # Zero the width of right-pad slots (band slot index >= width) so they add nothing.
    slot = np.arange(full_band)[None, :]
    widths = np.where(slot < width[:, None], widths, 0.0)

    # _clipped_linear_integral returns the volume*time measure (m³*day) of the
    # parcel's residence-window overlap with each cin bin; dividing by
    # (porosity*thickness) converts that overlap to plan-area * time, the areal
    # footprint (and duration) over which the areal deposition flux is mixed down
    # through the aquifer thickness. The bin width is already folded into the
    # integral, so do NOT multiply by widths again. The R-scaled window (r_apv)
    # stretches this measure by R, but the steady-state areal mass balance
    # Q*cout = dep*A_plan is R-independent (retardation delays breakthrough, it
    # does not raise the outlet concentration), so divide out that same R here.
    top_integral = _clipped_linear_integral(y_top[:, :-1], y_top[:, 1:], widths, 0.0, r_apv)
    bottom_integral = _clipped_linear_integral(y_bot[:, :-1], y_bot[:, 1:], widths, 0.0, r_apv)
    contact_volume_time = np.maximum(top_integral - bottom_integral, 0.0)
    numerator = contact_volume_time / (porosity * thickness * retardation_factor)

    band_vals = np.zeros((n_cout, full_band))
    # row_valid implies extracted_volume > 0, so the masked divide never sees a zero.
    band_vals[row_valid] = numerator[row_valid] / extracted_volume[row_valid, None]
    return band_vals, col_start, row_valid, spinup_row


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
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute concentrations from deposition rates (convolution).

    Parameters
    ----------
    dep : array-like
        Deposition rates [g/m²/day]. Length must equal len(tedges) - 1.
    flow : array-like
        Flow rates in aquifer [m³/day]. Length must equal len(tedges) - 1. The model
        assumes this value is constant over each interval ``[tedges[i], tedges[i+1])``.
    tedges : pandas.DatetimeIndex
        Time bin edges for deposition and flow data.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for output concentration data.
    aquifer_pore_volume : float
        Aquifer pore volume [m³].
    porosity : float
        Aquifer porosity [dimensionless].
    thickness : float
        Aquifer thickness [m].
    retardation_factor : float, optional
        Compound retardation factor, by default 1.0.
    spinup : {"constant"} | None, optional
        Spin-up policy applied before computing deposition weights.
        Default ``"constant"`` shifts ``tedges[0]`` backward by
        ``retardation_factor * aquifer_pore_volume / flow[0]`` and treats
        ``dep`` and ``flow`` as constant at their first observed values
        over the prepended interval. ``None`` keeps the existing
        strict-validity behavior (NaN cout rows during spin-up). A float
        raises ``NotImplementedError`` -- the fraction-threshold mode is
        not implemented for deposition (matching the diffusion family).

    Returns
    -------
    numpy.ndarray
        Concentration changes [g/m³] with length len(cout_tedges) - 1.

        Zero-extraction-flow cout bins (no water leaves the aquifer over the
        bin) return ``0.0``, not NaN. This deliberately differs from advection,
        which returns NaN for its undefined zero-flow output: the deposition
        source term is defined even with no water (an areal flux still supplies
        mass), and a bin that extracts zero volume carries zero mass, so ``0.0``
        is the physically correct value rather than an undefined result. NaN is
        reserved for spin-up bins whose residence time is not yet resolved.

    Raises
    ------
    ValueError
        If tedges does not have one more element than dep or flow, if input
        arrays contain NaN values, or if physical parameters are out of
        valid range (porosity not in (0, 1), non-positive thickness or
        aquifer pore volume).
    NotImplementedError
        If ``spinup`` is anything other than ``None`` or ``"constant"`` (the
        fraction-threshold mode is not implemented for deposition).

    See Also
    --------
    extraction_to_deposition : Inverse operation (deconvolution)
    spinup_duration : Earliest extraction time with a fully resolved deposition history
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
    >>> print(f"First finite cout: {cout[np.isfinite(cout)][0]:.4f} g/m³")
    First finite cout: 1.6667 g/m³
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
        spinup=spinup,
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
    assert weight_dep is not None  # noqa: S101 -- narrowed: cin was passed in

    # Build the banded forward operator and apply it as a banded einsum instead of
    # a dense W.dot(dep). Spin-up rows (NaN residence time) carry an all-zero band
    # and must return NaN. Zero-flow cout bins (extracted_volume == 0) carry a zero
    # band and return 0.
    band_vals, col_start, _, spinup_row = compute_deposition_weights(
        flow=weight_flow,
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )
    n_cin = len(weight_tedges) - 1
    cols = np.clip(col_start[:, None] + np.arange(band_vals.shape[1]), 0, n_cin - 1)
    cout = np.einsum("kb,kb->k", band_vals, weight_dep[cols])
    cout[spinup_row] = np.nan
    return cout


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
    spinup: str | None = "constant",
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
        Concentration changes in extracted water [g/m³]. Length must equal
        len(cout_tedges) - 1. May contain NaN values, which will be excluded
        from the computation along with corresponding rows in the weight matrix.
        The model assumes this value is constant over each interval
        ``[cout_tedges[i], cout_tedges[i+1])``.
    flow : array-like
        Flow rates in aquifer [m³/day]. Length must equal len(tedges) - 1.
        Must not contain NaN values. The model assumes this value is constant
        over each interval ``[tedges[i], tedges[i+1])``.
    tedges : pandas.DatetimeIndex
        Time bin edges for deposition and flow data. Length must equal
        len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for output concentration data. Length must equal
        len(cout) + 1.
    aquifer_pore_volume : float
        Aquifer pore volume [m³].
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
    spinup : {"constant"} | None, optional
        Spin-up policy applied before building the forward weight matrix.
        Default ``"constant"`` shifts ``tedges[0]`` backward by
        ``retardation_factor * aquifer_pore_volume / flow[0]`` and treats
        flow as constant at its first value over the prepended interval;
        the recovered deposition vector is sliced back to the original
        ``tedges`` length so the public output shape is unchanged.
        ``None`` keeps strict-validity behavior. A float raises
        ``NotImplementedError`` -- the fraction-threshold mode is not
        implemented for deposition (matching the diffusion family).

    Returns
    -------
    numpy.ndarray
        Mean deposition rates [g/m²/day] between tedges. Length equals
        len(tedges) - 1.

    Raises
    ------
    ValueError
        If input dimensions are incompatible or if flow contains NaN values.
    NotImplementedError
        If ``spinup`` is anything other than ``None`` or ``"constant"`` (the
        fraction-threshold mode is not implemented for deposition).

    See Also
    --------
    deposition_to_extraction : Forward operation (convolution)
    extraction_to_deposition_full : Full solver with nullspace options
    spinup_duration : Earliest extraction time with a fully resolved deposition history
    gwtransport.advection.extraction_to_infiltration : For concentration transport without deposition
    gwtransport.utils.solve_inverse_transport_banded : Banded Tikhonov solver used for inversion
    :ref:`concept-transport-equation` : Flow-weighted averaging approach

    Notes
    -----
    This is a *source* term -- positive ``dep`` raises ``cout``. Sink
    processes (pathogen attachment, first-order decay, particle filtration)
    require the opposite sign convention and are not modelled here.

    The forward model is ``W @ dep = cout``, where the weight matrix ``W``
    encodes the physical relationship between deposition rates and
    concentrations. ``W`` is genuinely banded -- row ``i`` is nonzero only on
    the cin bins inside its residence-time window -- and is built and solved in
    a compact banded layout (peak memory ``O(n_cin * band)``, never the dense
    ``O(n_cout * n_cin)``). Unlike advection (where rows sum to ~1), deposition
    rows sum to ``r_i = residence_time_i / (retardation_factor * porosity *
    thickness)``. Rows are
    rescaled by ``r_i`` before solving: when ``W`` has full column rank and
    ``cout`` lies in its column space this preserves the exact ``dep`` (with
    the default square ``spinup="constant"`` system the warm-start padding
    makes ``W`` structurally rank-deficient, so even a forward-generated
    ``cout`` is recovered only up to the nullspace, regardless of
    ``regularization_strength``), while for
    overdetermined systems with noise it is equivalent to weighted least
    squares with weights ``1 / r_i^2`` (shorter residence times get more
    weight; under constant flow all ``r_i`` are equal and this reduces to
    OLS). The rescaling puts the regularization target (transpose-and-normalize
    of ``W`` applied to ``cout``) on the same scale as ``dep``, which controls
    the regularization scale. Rows where the residence time cannot be computed
    (spin-up period) and zero-flow cout bins are excluded automatically; NaN
    values in ``cout`` are also excluded. The banded Tikhonov solve stays
    well-defined via ``regularization_strength`` even when ``W`` is
    rank-deficient (constant flow with integer ``RT/dt`` makes it a uniform
    moving average with exact transfer-function zeros), so no rank-deficiency
    warning is emitted.

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
    >>> flow = np.full(len(dates), 100.0)  # m³/day
    >>> cout = np.ones(len(cout_tedges) - 1) * 10.0  # g/m³
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
    >>> print(f"Mean deposition rate: {np.nanmean(dep):.2f} g/m²/day")
    Mean deposition rate: 6.00 g/m²/day
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
        spinup=spinup,
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

    # Build the banded forward operator (rows sum to r_k = RT_k/(porosity*thickness)).
    band_vals, col_start, row_valid, _ = compute_deposition_weights(
        flow=weight_flow,
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )
    n_cin_padded = len(weight_tedges) - 1

    # Per-row rescaling: normalize valid rows to sum 1 (w_norm = W_valid / r_k) and
    # feed the banded solver observed = cout / r_k -- the SAME 1/r_k scaling, REQUIRED
    # since deposition rows sum to r_k != 1 (unlike advection). Excluded rows -- NaN
    # residence time / zero-flow cout bins (~row_valid) and NaN values in cout -- are
    # zeroed in the band and given observed = 0 so they drop out of the normal
    # equations (a zero band contributes nothing).
    row_sums = band_vals.sum(axis=1)
    keep = row_valid & ~np.isnan(cout_values)
    safe_sum = np.where(keep, row_sums, 1.0)
    band_norm = np.where(keep[:, None], band_vals / safe_sum[:, None], 0.0)
    observed = np.where(keep, cout_values / safe_sum, 0.0)

    # The banded Tikhonov solve is well-defined via regularization even when the
    # operator is rank-deficient (constant flow with integer RT/dt makes it a
    # uniform moving average with exact transfer-function zeros).
    dep_padded = solve_inverse_transport_banded(
        band_vals=band_norm,
        col_start=col_start,
        observed=observed,
        n_output=n_cin_padded,
        regularization_strength=regularization_strength,
    )
    # Drop warm-start prefix so output aligns with the user-provided tedges.
    return dep_padded[n_pad:]


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
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute deposition rates from concentration changes using nullspace solver.

    Full-featured inverse solver exposing all options of
    :func:`~gwtransport.utils.solve_underdetermined_system`. For most use
    cases, prefer :func:`extraction_to_deposition` which uses Tikhonov
    regularization.

    Parameters
    ----------
    cout : array-like
        Concentration changes in extracted water [g/m³]. Length must equal
        len(cout_tedges) - 1. May contain NaN values, which will be excluded
        from the computation along with corresponding rows in the weight matrix.
    flow : array-like
        Flow rates in aquifer [m³/day]. Length must equal len(tedges) - 1.
        Must not contain NaN values.
    tedges : pandas.DatetimeIndex
        Time bin edges for deposition and flow data. Length must equal
        len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for output concentration data. Length must equal
        len(cout) + 1.
    aquifer_pore_volume : float
        Aquifer pore volume [m³].
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
    spinup : {"constant"} | None, optional
        Spin-up policy applied before building the forward weight matrix.
        Default ``"constant"`` shifts ``tedges[0]`` backward by
        ``retardation_factor * aquifer_pore_volume / flow[0]``; the
        recovered deposition is sliced back to the original ``tedges``
        length. ``None`` keeps strict-validity behavior. A float raises
        ``NotImplementedError`` -- the fraction-threshold mode is not
        implemented for deposition (matching the diffusion family).
        See :func:`extraction_to_deposition` for full semantics.

    Returns
    -------
    numpy.ndarray
        Mean deposition rates [g/m²/day] between tedges. Length equals
        len(tedges) - 1.

    Raises
    ------
    ValueError
        If cout_tedges does not have one more element than cout, if tedges
        does not have one more element than flow, if flow contains NaN
        values, or if physical parameters are out of valid range (porosity
        not in (0, 1), non-positive thickness or aquifer pore volume).
    NotImplementedError
        If ``spinup`` is anything other than ``None`` or ``"constant"`` (the
        fraction-threshold mode is not implemented for deposition).

    See Also
    --------
    extraction_to_deposition : Recommended solver using Tikhonov regularization.
    spinup_duration : Earliest extraction time with a fully resolved deposition history.
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
        spinup=spinup,
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

    # The nullspace solver (lstsq + null_space SVD) genuinely needs a dense matrix,
    # so build the band and densify it. Spin-up rows are set to NaN to match the
    # behavior of the historical dense build (which left those rows entirely NaN).
    band_vals, col_start, _, spinup_row = compute_deposition_weights(
        flow=weight_flow,
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )
    n_cin_padded = len(weight_tedges) - 1
    deposition_weights = _densify_weights(band_vals, col_start, n_cin_padded)
    deposition_weights[spinup_row] = np.nan

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
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    retardation_factor: float = 1.0,
) -> float:
    """
    Compute the spinup duration for deposition modeling.

    The spinup duration is the smallest extraction time ``t*`` (relative to
    ``tedges[0]``) at which the extracted water was infiltrated exactly at
    ``tedges[0]``: equivalently, the time at which the cumulative flow first
    reaches ``retardation_factor * aquifer_pore_volume``. For extraction times
    earlier than ``t*`` the extracted concentration lacks complete deposition
    history. Under constant flow this equals
    ``aquifer_pore_volume * retardation_factor / flow``.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m³/day].
    tedges : pandas.DatetimeIndex
        Time edges for the flow data.
    aquifer_pore_volume : float
        Pore volume of the aquifer [m³].
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless], by
        default 1.0.

    Returns
    -------
    float
        Spinup duration in days.

    Raises
    ------
    ValueError
        If the cumulative flow over the entire ``tedges`` window does not
        reach ``retardation_factor * aquifer_pore_volume``, indicating the
        flow timeseries is too short to characterise the spin-up duration.

    See Also
    --------
    deposition_to_extraction : Forward solver that uses the spin-up duration to resolve NaN cout rows.
    extraction_to_deposition : Inverse solver.
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
    # infiltrated exactly at tedges[0] satisfies
    # ``flow_cum(t*) = R * V_pore``; the spin-up duration is then
    # ``t* - 0 = t*``. Inverting the cumulative flow gives this value
    # exactly (no quantisation to tedges spacing). Under constant flow
    # this matches V*R/Q.
    flow_arr = np.asarray(flow)
    tedges_days = tedges_to_days(tedges)
    dt_days = np.diff(tedges_days)
    target_cum = retardation_factor * float(aquifer_pore_volume)
    # Feasibility guard on the *un-bumped* cumulative total: the request is infeasible iff
    # R*V_pore exceeds the true total infiltrated volume. (The monotone bump below would
    # otherwise lift a trailing Q=0 plateau above target_cum and admit an infeasible request.)
    flow_cum_raw = cumulative_flow_volume(flow_arr, dt_days)
    if not flow_cum_raw[-1] >= target_cum:
        msg = (
            f"Cumulative flow over the entire tedges window ({flow_cum_raw[-1]:.6g} m³) does not reach "
            f"retardation_factor * aquifer_pore_volume ({target_cum:.6g} m³); the flow timeseries is too "
            "short to characterise the spin-up duration."
        )
        raise ValueError(msg)
    # Plateaus in flow_cum from Q = 0 bins make V → t inversion multi-valued; bump duplicates
    # by the smallest representable amount so np.interp resolves consistently at plateau levels.
    # Reuse the raw cumsum (bit-identical to cumulative_flow_volume(..., strictly_monotone=True),
    # which applies the same _make_strictly_monotone to the same array).
    flow_cum = _make_strictly_monotone(flow_cum_raw)
    return float(linear_interpolate(x_ref=flow_cum, y_ref=tedges_days, x_query=target_cum))
