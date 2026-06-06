"""
Shared closed-form helpers for the Kreft-Zuber flux-concentration transport modules.

This private module holds the pieces common to :mod:`gwtransport.diffusion_fast` and
:mod:`gwtransport.diffusion_fast_fast`: the breakthrough antiderivative, the retardation
flux-coefficient correction, input validation, and the small per-streamtube / spin-up
helpers. Both modules import from here so the closed-form math is defined once and produces
bit-identical floating-point results regardless of which module evaluates it.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import erf, erfcx

from gwtransport._time import dt_to_days
from gwtransport.utils import solve_inverse_transport_banded

# Minimum coefficient sum to consider an output bin valid.
_EPSILON_COEFF_SUM = 1e-10

# sqrt(pi), used in the closed-form breakthrough antiderivative.
_SQRT_PI = np.sqrt(np.pi)

# Floor on the moving-frame dispersion product D_t [m^2] to keep the erf argument finite
# for pre-breakthrough / zero-dispersion edges (where D_t -> 0).
_DT_FLOOR = 1e-30


def _breakthrough_antideriv(
    step_widths: npt.NDArray[np.floating], dt_var: npt.NDArray[np.floating]
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Closed-form antiderivative of the resident concentration, evaluated per edge.

    Returns :math:`I(x) = \tfrac12 x + \tfrac12[x\,\operatorname{erf}(x/s) + (s/\sqrt\pi)e^{-(x/s)^2}]`
    with :math:`s = 2\sqrt{D_t}`, alongside ``s`` and the Gaussian factor ``exp(-(x/s)^2)`` (both
    reused by the retardation correction). Shared by the banded ``C_F`` build
    (:func:`gwtransport.diffusion_fast._pv_band_values`) and the slow quadrature, so both compute
    identical floating-point results for the same inputs.

    Returns
    -------
    antideriv : ndarray
        The antiderivative :math:`I(x)`.
    s : ndarray
        ``2 * sqrt(D_t)``.
    gaussian : ndarray
        ``exp(-(x/s)^2)``.
    """
    s = 2.0 * np.sqrt(dt_var)
    with np.errstate(over="ignore", invalid="ignore"):
        u = step_widths / s
        gaussian = np.exp(-(u * u))
        antideriv = 0.5 * step_widths + 0.5 * (step_widths * erf(u) + (s / _SQRT_PI) * gaussian)
    return antideriv, s, gaussian


def _retardation_excess_density(
    *,
    x_lo: npt.NDArray[np.floating],
    x_hi: npt.NDArray[np.floating],
    dx: npt.NDArray[np.floating],
    d_lo: npt.NDArray[np.floating],
    d_hi: npt.NDArray[np.floating],
    s_lo: npt.NDArray[np.floating],
    s_hi: npt.NDArray[np.floating],
    g_lo: npt.NDArray[np.floating],
    g_hi: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Closed-form bin-average of the Gaussian density ``<g>`` for the retardation correction.

    When ``R != 1`` and ``D_m > 0`` the per-edge antiderivative bakes in an effective flux
    coefficient ``dD_t/dx = R*D_m/v + alpha_L`` (because ``d(tau)/dx = R/v`` under retardation),
    whereas Kreft-Zuber wants ``D_L/v = D_m/v + alpha_L``. The excess ``(R-1)*D_m/v`` is removed
    by subtracting it times ``<g>``, the bin-average of the Gaussian density
    ``g = e^{-x^2/(4 D_t)} / (2 sqrt(pi D_t))``. Within a bin ``D_t`` is linear in ``x``, so ``<g>``
    is closed form: substituting ``w = D_t`` in ``int g dx`` gives
    ``int w^{-1/2} e^{-w/(4 m^2) - c^2/(4 m^2 w)} dw`` -- a pair of error functions -- with slope
    ``m = dD_t/dx`` and intercept ``c = D_t(x=0)``. Reusing the antiderivative's Gaussian factor
    ``G = e^{-x^2/(4 D_t)}``, ``erfcx`` keeps the (always non-negative) ``u+`` branch overflow-free;
    the ``u-`` branch splits: ``erfcx`` where ``u- >= 0``, ``erf`` elsewhere. Each transcendental is
    evaluated on its own subset (fancy indexing) rather than over the full array. The slope->0
    (constant-``D_t``) limit is the exact ``[C_R(x_hi) - C_R(x_lo)] / dx``, applied only on that subset.

    Parameters
    ----------
    x_lo, x_hi : ndarray
        Breakthrough coordinate ``step_widths`` at the lower / upper cout edge of each bin.
    dx : ndarray
        ``x_hi - x_lo``.
    d_lo, d_hi : ndarray
        Moving-frame dispersion product ``D_t`` at the lower / upper edge.
    s_lo, s_hi : ndarray
        ``2 * sqrt(D_t)`` at the lower / upper edge.
    g_lo, g_hi : ndarray
        Gaussian factor ``exp(-(x/s)^2)`` at the lower / upper edge.

    Returns
    -------
    ndarray
        Bin-averaged Gaussian density ``<g>``.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        slope = (d_hi - d_lo) / dx
        intercept = d_lo - slope * x_lo
        abs_int = np.abs(intercept)
        # s = 2*sqrt(D_t) is already computed, so slope*s = 2*slope*sqrt(D_t).
        two_m_sqrt_lo = slope * s_lo
        two_m_sqrt_hi = slope * s_hi
        um_lo = (d_lo - abs_int) / two_m_sqrt_lo
        um_hi = (d_hi - abs_int) / two_m_sqrt_hi
        up_lo = (d_lo + abs_int) / two_m_sqrt_lo
        up_hi = (d_hi + abs_int) / two_m_sqrt_hi
        t_plus = g_lo * erfcx(up_lo) - g_hi * erfcx(up_hi)
        t_minus = np.empty_like(t_plus)
        pos = um_lo >= 0.0
        neg = ~pos
        t_minus[pos] = g_lo[pos] * erfcx(um_lo[pos]) - g_hi[pos] * erfcx(um_hi[pos])
        c_minus = np.exp((intercept[neg] - abs_int[neg]) / (2.0 * slope[neg] ** 2))
        t_minus[neg] = c_minus * (erf(um_hi[neg]) - erf(um_lo[neg]))
        density_binavg = (t_minus + t_plus) / (2.0 * dx)
        flat = np.abs(d_hi - d_lo) <= 1e-9 * np.maximum(d_lo, d_hi)
        d_bar_s = 2.0 * np.sqrt(0.5 * (d_lo[flat] + d_hi[flat]))
        density_binavg[flat] = 0.5 * (erf(x_hi[flat] / d_bar_s) - erf(x_lo[flat] / d_bar_s)) / dx[flat]
        density_binavg[dx <= 0.0] = 0.0
    return density_binavg


def _cout_cumulative_volume(
    *,
    flow_out: npt.NDArray[np.floating] | None,
    cout_tedges: pd.DatetimeIndex,
    cout_tedges_days: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cumulative_volume_at_cin: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Cumulative through-flow volume at each cout edge, on the infiltration volume axis.

    When ``flow_out`` is given (the user-specified extraction-side flow) the cout-edge volumes are
    its cumulative integral, anchored at the first cout edge inside the flow record (so an output
    window starting before the input data stays correctly aligned). Otherwise the cout edges are
    interpolated from the infiltration cumulative-volume curve. Shared by
    :func:`gwtransport.diffusion_fast._closed_form_coeff_matrix` and
    :func:`gwtransport.diffusion_fast_fast.infiltration_to_extraction` so the forward (approximate)
    and reverse (exact) paths place the cout grid on identical volume coordinates.

    Parameters
    ----------
    flow_out : ndarray or None
        Extraction flow rate [m3/day] on the output grid (length ``len(cout_tedges) - 1``), or None
        to interpolate the cout edges from the infiltration curve.
    cout_tedges : DatetimeIndex
        Output time-bin edges (used only for the ``flow_out`` bin widths).
    cout_tedges_days : ndarray
        Output edges as days relative to the (work) infiltration reference.
    tedges_days : ndarray
        Infiltration edges as days relative to the same reference.
    cumulative_volume_at_cin : ndarray
        Cumulative infiltrated volume at each infiltration edge.

    Returns
    -------
    ndarray
        Cumulative volume at each cout edge (length ``len(cout_tedges)``).
    """
    if flow_out is None:
        return np.interp(cout_tedges_days, tedges_days, cumulative_volume_at_cin)
    cumsum_out = np.concatenate(([0.0], np.cumsum(flow_out * dt_to_days(cout_tedges))))
    in_range = (cout_tedges_days >= tedges_days[0]) & (cout_tedges_days <= tedges_days[-1])
    i0 = int(np.argmax(in_range)) if np.any(in_range) else 0
    v_at_i0 = float(np.interp(cout_tedges_days[i0], tedges_days, cumulative_volume_at_cin))
    return v_at_i0 + (cumsum_out - cumsum_out[i0])


def _extend_tedges_flag(spinup: str | float | None) -> bool:
    """Translate the public ``spinup`` parameter to the internal extend flag.

    ``"constant"`` (default) extends ``tedges`` by 100 years on each side so a constant
    warm-start fills the left-edge spin-up region; ``None`` disables the extension (spin-up
    cout becomes NaN). Mirrors :func:`gwtransport.diffusion._diffusion_extend_tedges_flag`.

    Returns
    -------
    bool
        True if ``tedges`` should be extended (warm-start), False otherwise.

    Raises
    ------
    ValueError
        If ``spinup`` is a string other than ``"constant"``.
    NotImplementedError
        If ``spinup`` is a float (fraction-threshold mode is not implemented).
    """
    if spinup is None:
        return False
    if isinstance(spinup, str):
        if spinup != "constant":
            msg = f"spinup string must be 'constant'; got {spinup!r}"
            raise ValueError(msg)
        return True
    msg = f"spinup only supports None or 'constant'; float thresholds are not implemented (got {spinup!r})"
    raise NotImplementedError(msg)


def _broadcast_to_pore_volumes(
    values: npt.NDArray[np.floating] | float, n_pore_volumes: int
) -> npt.NDArray[np.floating]:
    """Return a per-pore-volume array: a scalar broadcasts to all streamtubes, an array passes through.

    Length is validated upstream by :func:`_validate_inputs`, so a non-scalar array is
    returned as-is (assumed length ``n_pore_volumes``).

    Returns
    -------
    ndarray, shape (n_pore_volumes,)
        Per-streamtube values.
    """
    arr = np.atleast_1d(np.asarray(values, dtype=float))
    return np.broadcast_to(arr, (n_pore_volumes,)) if arr.size == 1 else arr


def _validate_inputs(
    *,
    cin_or_cout: np.ndarray,
    flow: np.ndarray,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: np.ndarray,
    streamline_length: npt.NDArray[np.floating] | float,
    molecular_diffusivity: npt.NDArray[np.floating] | float,
    longitudinal_dispersivity: npt.NDArray[np.floating] | float,
    retardation_factor: float,
    is_forward: bool,
    flow_out: np.ndarray | None = None,
) -> None:
    """Validate inputs for infiltration_to_extraction and extraction_to_infiltration.

    ``streamline_length`` / ``molecular_diffusivity`` / ``longitudinal_dispersivity`` may be
    a scalar or an array of length ``len(aquifer_pore_volumes)`` (one value per streamtube).

    Raises
    ------
    ValueError
        If array lengths are inconsistent, molecular_diffusivity or
        longitudinal_dispersivity are negative, cin or cout or flow contain NaN values,
        aquifer_pore_volumes contains non-positive values, streamline_length is
        non-positive, or retardation_factor is below 1 (anti-retardation is not physical for
        the supported sorption isotherms).
    """
    if is_forward:
        if len(tedges) != len(cin_or_cout) + 1:
            msg = "tedges must have one more element than cin"
            raise ValueError(msg)
    elif len(cout_tedges) != len(cin_or_cout) + 1:
        msg = "cout_tedges must have one more element than cout"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    n_pore_volumes = len(aquifer_pore_volumes)
    for name, arr in (
        ("streamline_length", streamline_length),
        ("molecular_diffusivity", molecular_diffusivity),
        ("longitudinal_dispersivity", longitudinal_dispersivity),
    ):
        if np.size(arr) not in {1, n_pore_volumes}:
            msg = f"{name} must be a scalar or have length len(aquifer_pore_volumes) = {n_pore_volumes}"
            raise ValueError(msg)
    if np.any(np.asarray(molecular_diffusivity) < 0):
        msg = "molecular_diffusivity must be non-negative"
        raise ValueError(msg)
    if np.any(np.asarray(longitudinal_dispersivity) < 0):
        msg = "longitudinal_dispersivity must be non-negative"
        raise ValueError(msg)
    if np.any(np.isnan(cin_or_cout)):
        msg = f"{'cin' if is_forward else 'cout'} contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(flow < 0):
        msg = "flow must be non-negative (negative flow not supported)"
        raise ValueError(msg)
    if np.any(aquifer_pore_volumes <= 0):
        msg = "aquifer_pore_volumes must be positive"
        raise ValueError(msg)
    if np.any(np.asarray(streamline_length) <= 0):
        msg = "streamline_length must be positive"
        raise ValueError(msg)
    if retardation_factor < 1.0:
        msg = "retardation_factor must be >= 1.0"
        raise ValueError(msg)
    if flow_out is None:
        # The output-grid extraction flow is only unambiguous when the cout grid matches
        # the flow grid; otherwise it must be supplied (it defines the cout-bin volumes and
        # the outlet velocity used by the retardation correction).
        if not tedges.equals(cout_tedges):
            msg = "flow_out is required when cout_tedges differs from tedges"
            raise ValueError(msg)
    else:
        n_cout = len(cout_tedges) - 1
        if len(flow_out) != n_cout:
            msg = f"flow_out must have length len(cout_tedges) - 1 = {n_cout}, got {len(flow_out)}"
            raise ValueError(msg)
        if np.any(np.isnan(flow_out)):
            msg = "flow_out contains NaN values, which are not allowed"
            raise ValueError(msg)
        if np.any(flow_out < 0):
            msg = "flow_out must be non-negative (negative flow not supported)"
            raise ValueError(msg)


def _solve_reverse_banded(
    *,
    band_vals: npt.NDArray[np.floating],
    col_start: npt.NDArray[np.intp],
    valid_cout_bins: npt.NDArray[np.bool_],
    cout: npt.NDArray[np.floating],
    n_cin: int,
    regularization_strength: float,
) -> npt.NDArray[np.floating]:
    """Normalize, decouple the warm-start tail, and solve the banded Tikhonov inverse.

    Shared by :func:`gwtransport.diffusion_fast.extraction_to_infiltration` and
    :func:`gwtransport.diffusion_fast_fast.extraction_to_infiltration`, which build the same banded
    forward operator and must therefore produce byte-identical inverses. The steps are:

    1. Zero invalid rows (incomplete breakthrough) and normalize the remaining rows to sum to 1 --
       the banded solver's ``W x ~= observed`` precondition.
    2. Decouple the warm-start data-start tail. With a leading zero-flow plateau and ``D_m > 0`` the
       forward operator carries a negative warm-start coefficient at the data-start columns (kept in
       the forward band so the forward ``C_F`` is exact); their net column sum is ``<= 0``, so the
       banded normal-equation solver would leave them unregularized -- a large, unregularized
       ``WᵀW`` diagonal coupled to the spin-up nullspace, which is indefinite and breaks the Cholesky
       factorisation. These columns are the unrecoverable spin-up region (NaN in the dense and the
       slow-module inverses alike), so zeroing their band entries decouples them: the solver's
       zero-diagonal path returns them as NaN and the remaining system is symmetric positive definite.

    Parameters
    ----------
    band_vals : ndarray, shape (n_cout_bins, full_band)
        Banded forward weights from :func:`gwtransport.diffusion_fast._closed_form_coeff_matrix`.
    col_start : ndarray of int, shape (n_cout_bins,)
        First cin-bin column of each cout row's band.
    valid_cout_bins : ndarray of bool, shape (n_cout_bins,)
        Output bins with complete breakthrough information.
    cout : ndarray, shape (n_cout_bins,)
        Observed extraction concentration.
    n_cin : int
        Number of infiltration bins (output length).
    regularization_strength : float
        Tikhonov parameter.

    Returns
    -------
    ndarray, shape (n_cin,)
        Recovered infiltration concentration; NaN for unconstrained / spin-up bins.
    """
    row_sums = band_vals.sum(axis=1)
    valid = valid_cout_bins & (row_sums > _EPSILON_COEFF_SUM)
    bn = band_vals.copy()
    bn[~valid] = 0.0
    bn[valid] /= row_sums[valid, None]

    full_band = bn.shape[1]
    band_cols = col_start[:, None] + np.arange(full_band)
    in_range = band_cols < n_cin
    band_cols_clipped = np.clip(band_cols, 0, n_cin - 1)
    col_sum = np.zeros(n_cin)
    np.add.at(col_sum, band_cols_clipped[in_range], bn[in_range])
    inactive_col = col_sum <= _EPSILON_COEFF_SUM
    bn[inactive_col[band_cols_clipped] & in_range] = 0.0

    return solve_inverse_transport_banded(
        band_vals=bn,
        col_start=col_start,
        observed=cout,
        n_output=n_cin,
        regularization_strength=regularization_strength,
    )
