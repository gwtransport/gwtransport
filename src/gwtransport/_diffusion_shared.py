"""
Shared closed-form helpers for the Kreft-Zuber flux-concentration transport modules.

This private module holds the pieces common to :mod:`gwtransport.diffusion_fast` and
:mod:`gwtransport.diffusion_fast_fast`: the breakthrough antiderivative, input validation, the
banded Tikhonov reverse solve, and the small per-streamtube / spin-up helpers. Both modules import
from here so these primitives are defined once
and evaluate bit-identically in either module (the modules' overall transport is *not* identical:
diffusion_fast is exact, diffusion_fast_fast approximate).

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import erf

from gwtransport._time import dt_to_days
from gwtransport._validation import (
    _validate_no_nan,
    _validate_non_negative_array,
    _validate_positive_array,
    _validate_retardation_factor,
)
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
) -> npt.NDArray[np.floating]:
    r"""Closed-form antiderivative of the resident concentration, evaluated per edge.

    Returns :math:`I(x) = \tfrac12 x + \tfrac12[x\,\operatorname{erf}(x/s) + (s/\sqrt\pi)e^{-(x/s)^2}]`
    with :math:`s = 2\sqrt{D_t}`. Shared by the banded ``C_F`` build
    (:func:`gwtransport.diffusion_fast._pv_band_values`) and the slow quadrature, so both compute
    identical floating-point results for the same inputs. Because ``dD_t/dx = D_s/v_s`` is the
    Kreft-Zuber flux coefficient at the solute-front velocity, differencing ``I`` across a cout bin
    yields the flux concentration ``C_F`` directly.

    Returns
    -------
    antideriv : ndarray
        The antiderivative :math:`I(x)`.
    """
    s = 2.0 * np.sqrt(dt_var)
    with np.errstate(over="ignore", invalid="ignore"):
        u = step_widths / s
        gaussian = np.exp(-(u * u))
        return 0.5 * step_widths + 0.5 * (step_widths * erf(u) + (s / _SQRT_PI) * gaussian)


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
    :func:`gwtransport.diffusion_fast_fast._build_forward_operator` so every path --
    diffusion_fast (exact) and diffusion_fast_fast's forward and reverse (both approximate, and built
    from the same operator) -- places the cout grid on identical volume coordinates.

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
    # np.argmax returns 0 for an all-False mask, the same fallback the guard provided.
    i0 = int(np.argmax(in_range))
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
        longitudinal_dispersivity are negative or non-finite, cin (forward) or flow contain NaN
        values, aquifer_pore_volumes contains non-positive or non-finite values,
        streamline_length is non-positive or non-finite, or retardation_factor is NaN or below 1
        (anti-retardation is not physical for the supported sorption isotherms).
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
    # Delegate the finite+sign invariants to the shared _validation atoms so the NaN/+inf guards
    # (which the bare ``< 0`` / ``<= 0`` / ``< 1.0`` comparisons here would let slip through) are
    # enforced in exactly one place. Order and messages match the historical inline checks.
    _validate_non_negative_array(molecular_diffusivity, name="molecular_diffusivity")
    _validate_non_negative_array(longitudinal_dispersivity, name="longitudinal_dispersivity")
    # Forward cin must be NaN-free; reverse cout may contain NaN (measurement
    # gaps) -- the banded inverse solver excludes those rows, matching deposition (#321).
    if is_forward:
        _validate_no_nan(cin_or_cout, name="cin")
    _validate_no_nan(flow, name="flow")
    _validate_non_negative_array(flow, name="flow", message="flow must be non-negative (negative flow not supported)")
    _validate_positive_array(aquifer_pore_volumes, name="aquifer_pore_volumes")
    _validate_positive_array(streamline_length, name="streamline_length")
    _validate_retardation_factor(retardation_factor)
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

    Shared by :func:`gwtransport.diffusion_fast.extraction_to_infiltration` (which feeds the exact
    closed-form banded operator) and :func:`gwtransport.diffusion_fast_fast.extraction_to_infiltration`
    (which feeds the approximate banded breakthrough operator). The steps are:

    1. Zero invalid rows (incomplete breakthrough) and normalize the remaining rows to sum to 1 --
       the banded solver's ``W x ~= observed`` precondition.
    2. Decouple the warm-start data-start tail. With a leading zero-flow plateau and ``D_m > 0`` the
       forward operator carries a negative warm-start coefficient at the data-start columns (kept in
       the forward band so the forward ``C_F`` is reproduced); their net column sum is ``<= 0``, so the
       banded normal-equation solver would leave them unregularized -- a large, unregularized
       ``WᵀW`` diagonal coupled to the spin-up nullspace, which is indefinite and breaks the Cholesky
       factorisation. These columns are the unrecoverable spin-up region (NaN in the dense and the
       slow-module inverses alike), so zeroing their band entries decouples them: the solver's
       zero-diagonal path returns them as NaN and the remaining system is symmetric positive definite.

    Parameters
    ----------
    band_vals : ndarray, shape (n_cout_bins, full_band)
        Banded forward weights -- from :func:`gwtransport.diffusion_fast._closed_form_coeff_matrix`
        (exact) or :func:`gwtransport.diffusion_fast_fast._banded_forward_matrix` (approximate breakthrough).
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
