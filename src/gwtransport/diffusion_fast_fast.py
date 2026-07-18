"""
Fast *approximate* 1D advection-dispersion transport (Kreft-Zuber flux concentration).

This module shares the conceptual model of :mod:`gwtransport.diffusion` and
:mod:`gwtransport.diffusion_fast` -- advection with microdispersion (``alpha_L``) and molecular
diffusion (``D_m``) along orthogonal (Cartesian) flow paths, one independent streamtube per aquifer
pore volume, the spread across the pore volume distribution providing macrodispersion, and linear
sorption via the retardation factor. It
targets the bin-averaged Kreft-Zuber (1978) flux concentration ``C_F`` on the streamtube bundle, but
trades exactness for a single fast (~1.5 ms) native-grid evaluation that does not depend on the flow
being constant.
It is **approximate**: where :mod:`gwtransport.diffusion_fast` reproduces the quadrature reference to
machine precision, this module is exact only up to a small interpolation floor at constant flow (see
Accuracy below) and degrades under variable flow. When you need machine precision, use
:mod:`gwtransport.diffusion_fast`.

How it works -- one skewed breakthrough on the native volume grid
-----------------------------------------------------------------

The moving-frame dispersion product ``D_t = D_m*tau + alpha_L*xi`` mixes a *time* term (molecular
diffusion ``D_m*tau``) and a *volume* term (microdispersion ``alpha_L*xi``). Under constant flow the
two coincide: elapsed time and breakthrough coordinate are locked, ``tau = r_vpv*xi/(L*Q)``
(``r_vpv = R*V_pore``), so ``D_m*tau = (D_m*r_vpv/(L*Q))*xi`` -- the same ``xi``-proportional form as
``alpha_L*xi``. Molecular diffusion is therefore an **effective dispersivity**
``alpha_eff = alpha_L + D_m*r_vpv/(L*q_mean)`` per streamtube (``q_mean`` the record-mean flow), and
the whole method is a single skewed ``D_t = alpha_eff*xi`` Kreft-Zuber breakthrough applied banded on
the **native cumulative-volume grid**: the whole aquifer pore volume distribution (APVD) is pre-summed
into one 1D antiderivative ``Ibar(dV)`` -- exact for any APVD shape -- finely sampled once and read
back by interpolation.

``tedges`` need **not** be regularly spaced and ``cout_tedges`` need not equal ``tedges`` (supply
``flow_out`` when they differ): the build runs on the native cumulative-volume grid for any spacing.
The **only** approximations are the ``Ibar`` interpolation (~1e-4, below) and -- under *variable* flow
-- freezing ``q_mean`` at the record mean: the ``tau = r_vpv*xi/(L*Q)`` map holds exactly only at
constant flow, so the microdispersion part stays exact but the molecular part picks up a commutator
residual.

Accuracy (vs :mod:`gwtransport.diffusion_fast`)
-----------------------------------------------

- **Constant flow:** exact to the ``Ibar`` interpolation floor -- ~1e-6 for smooth inputs, degrading
  to ~1e-4 for sharp inputs at ``alpha_L = 0`` (the kink limit; ~1e-3 for a single large pore volume).
  This holds across ``D_m`` and ``R`` at realistic Peclet numbers ``Pe = L/alpha_eff >> 1``,
  **including heat transport** (``R > 1``, large ``D_m``): the effective-dispersivity fold reproduces
  the skewed molecular breakthrough exactly, not merely its second moment. Only in the extreme
  low-Peclet corner (``alpha_eff`` approaching ``L`` -- a short streamline with very strong molecular
  diffusion, ``Pe <~ 2``) does the fold's wide dispersive band exceed the fine-grid sample cap
  (``_MAX_KERNEL_SAMPLES``), coarsening the breakthrough shape; use :mod:`gwtransport.diffusion_fast`
  there.
- **Variable flow:** the microdispersion part stays exact; the molecular part carries the frozen-
  ``q_mean`` commutator residual. It is small when molecular diffusion is sub-dominant -- the typical
  ``alpha_L > 0`` regime, <~1e-3 for realistic solute ``D_m`` even through strong flow -- and grows in
  the molecular-diffusion-dominated corner (``alpha_L`` ~ 0) with sharp inputs under strongly variable
  or seasonal flow, reaching the multi-percent range where no fast approximation is reliable.
  **Use** :mod:`gwtransport.diffusion_fast` **there** (in particular for heat transport under strongly
  variable flow).

The inverse (:func:`extraction_to_infiltration`) deconvolves the *same* approximate operator ``W`` the
forward applies. It assembles ``W`` directly in banded form (one ``Ibar`` gather -- no per-pore-volume
closed-form loop, no dense ``(n_cout, n_cin)`` matrix) and solves it with banded Tikhonov
regularisation (banded Cholesky, ``O(n * band**2)``), so it is much faster than
:mod:`gwtransport.diffusion_fast`'s reverse, especially for many streamtubes. Inverting exactly the
forward operator makes a round trip self-consistent (recovering the input up to the deconvolution
conditioning). On *real* extraction data the reverse is only as accurate as the forward: at constant
flow it reproduces :mod:`gwtransport.diffusion_fast`'s reverse to ~1e-6, while in the
molecular-dominated corner under strongly variable flow the deconvolution amplifies the forward's
commutator residual -- use :mod:`gwtransport.diffusion_fast` there.

Available functions:

- :func:`infiltration_to_extraction` -- forward transport (approximate).
- :func:`extraction_to_infiltration` -- inverse via banded Tikhonov regularisation (approximate).
- :func:`gamma_infiltration_to_extraction` -- gamma-distributed APVD (forward).
- :func:`gamma_extraction_to_infiltration` -- same, inverse.

References
----------
Kreft, A., & Zuber, A. (1978). On the physical meaning of the dispersion equation and its
solutions for different initial and boundary conditions. Chemical Engineering Science,
33(11), 1471-1480.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import coo_array

from gwtransport import gamma
from gwtransport._diffusion_shared import (
    _DT_FLOOR,
    _EPSILON_COEFF_SUM,
    _breakthrough_antideriv,
    _broadcast_to_pore_volumes,
    _cout_cumulative_volume,
    _extend_tedges_flag,
    _solve_reverse_banded,
    _validate_inputs,
)
from gwtransport._time import dt_to_days, tedges_to_days
from gwtransport.diffusion_fast import _DEFAULT_SATURATION_THRESHOLD
from gwtransport.residence_time import fraction_explained_full
from gwtransport.utils import cumulative_flow_volume

# Samples per native bin used to discretise the 1D breakthrough antiderivative ``Ibar``. Higher =
# more accurate ``Ibar`` interpolation (breakthrough error ~ O(1/_KERNEL_FINE^2), or O(1/_KERNEL_FINE)
# at the alpha_eff=0 kink) at the cost of a larger one-time precompute; 16 gives ~1e-4 -- the
# constant-flow accuracy floor of the method -- so it is not exposed as a user knob.
_KERNEL_FINE = 16

# Upper bound on the number of fine ``Ibar`` samples. Caps the one-time precompute when the
# breakthrough band is very wide -- either tiny flow (enormous front offset; those bins are masked by
# residence time anyway, so coarsening is benign) or the extreme low-Peclet corner where molecular
# diffusion folds into a large ``alpha_eff`` (a valid, unmasked band). In the latter the cap coarsens
# ``dv_fine`` below one sample per output bin and the breakthrough shape loses accuracy -- use
# :mod:`gwtransport.diffusion_fast` for a short streamline with very strong molecular diffusion
# (Pe = L/alpha_eff <~ 2). No realistic groundwater/heat regime reaches that corner.
_MAX_KERNEL_SAMPLES = 20000


def _summed_antideriv(
    *,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    streamline_length: npt.NDArray[np.floating],
    molecular_diffusivity: npt.NDArray[np.floating],
    longitudinal_dispersivity: npt.NDArray[np.floating],
    retardation_factor: float,
    q_mean: float,
    mean_bin_volume: float,
    saturation_threshold: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float, float, float]:
    r"""Precompute the APVD-summed breakthrough antiderivative ``Ibar(dV)`` on a fine 1D grid.

    For one streamtube the bin-averaged flux fraction over a cout bin equals the second difference of
    the antiderivative ``I(x)`` (:func:`gwtransport._diffusion_shared._breakthrough_antideriv`) of the
    resident concentration in the breakthrough coordinate ``x = (dV - r_vpv)*L/r_vpv``
    (``r_vpv = R*V_pore``). The antiderivative *with respect to cumulative volume* ``dV`` is
    ``(r_vpv/L)*I(x(dV))``; averaging it over the APVD gives a single 1D function

    .. math::

        \bar I(\Delta V) = \operatorname{mean}_{pv}\Bigl[\tfrac{r_{vpv}}{L}\,
        I\bigl((\Delta V - r_{vpv})L/r_{vpv}\bigr)\Bigr],\quad D_t = \alpha_{eff}\,\xi,

    whose edge-differences reproduce the per-streamtube-averaged ``C_F`` **exactly** for any APVD
    (the ``r_vpv/L`` Jacobian and the cout-bin volume normalisation cancel the per-streamtube ``dx``).
    Only the interpolation of ``Ibar`` is approximate.

    Molecular diffusion is folded into ``D_t`` as an **effective dispersivity**. The moving-frame
    variance is ``D_t = D_m*tau + alpha_L*xi``; under constant flow ``tau = r_vpv*xi/(L*Q)``, so
    ``D_m*tau = (D_m*r_vpv/(L*Q))*xi`` -- the same ``xi``-proportional form as ``alpha_L*xi``. Hence
    ``D_t = alpha_eff*xi`` with ``alpha_eff = alpha_L + D_m*r_vpv/(L*q_mean)`` per streamtube, which
    reproduces the skewed Kreft-Zuber molecular breakthrough exactly at constant flow (``q_mean`` is
    the record-mean flow; freezing it is the only approximation, and only under variable flow).

    The grid is **uniform** over the breakthrough band ``[off_lo, off_hi]`` (front ``r_vpv`` plus the
    conservative ``alpha_eff`` dispersion smear, unioned over streamtubes) plus a margin, sampled at
    ``mean_bin_volume / _KERNEL_FINE`` (uniformity lets :func:`_eval_antideriv` interpolate by
    fractional indexing instead of a per-point search). For ``alpha_eff > 0`` ``Ibar`` is smooth and
    the interpolation error is ``O(1/_KERNEL_FINE^2)``; only at ``alpha_L = 0`` *and* ``D_m = 0`` does
    it have a kink at each ``r_vpv`` (error ``O(1/_KERNEL_FINE)``, sub-1e-3).

    Returns
    -------
    grid : ndarray
        Cumulative-volume offsets ``dV`` (uniformly spaced, strictly increasing).
    ibar : ndarray
        ``Ibar`` sampled at ``grid``.
    mean_r_vpv : float
        ``R * mean(V_pore)`` -- the saturated offset (``Ibar -> dV - mean_r_vpv`` above the band).
    off_lo, off_hi : float
        Lower / upper cumulative-volume offset of the breakthrough band.
    """
    r_vpv = retardation_factor * aquifer_pore_volumes
    mean_r_vpv = float(r_vpv.mean())
    u = saturation_threshold
    # Molecular diffusion as an effective dispersivity (see docstring): D_t = alpha_eff*xi. q_mean > 0
    # is guaranteed -- _build_forward_operator returns early when total_volume <= 0, and
    # tedges_days[-1] - tedges_days[0] > 0 for any valid strictly-increasing tedges.
    alpha_eff = longitudinal_dispersivity + molecular_diffusivity * r_vpv / (streamline_length * q_mean)
    # Dispersion half-widths in the breakthrough coordinate x (front D_t = alpha_eff*L): the pre side
    # shrinks (|x| = U*2*sqrt(alpha_eff*L)); the post side grows with slope alpha_eff, giving the
    # quadratic root x = 2U^2*alpha_eff + 2U*sqrt(U^2*alpha_eff^2 + alpha_eff*L). Mapped to volume
    # offsets via r_vpv/L and unioned over streamtubes (conservative, never under-covers the band).
    pre_x = u * 2.0 * np.sqrt(alpha_eff * streamline_length)
    post_x = 2.0 * u * u * alpha_eff + 2.0 * u * np.sqrt(u * u * alpha_eff**2 + alpha_eff * streamline_length)
    off_lo = float(np.min(r_vpv - (r_vpv / streamline_length) * pre_x))
    off_hi = float(np.max(r_vpv + (r_vpv / streamline_length) * post_x))

    margin = 4.0 * mean_bin_volume
    span = (off_hi - off_lo) + 2.0 * margin
    dv_fine = mean_bin_volume / _KERNEL_FINE
    if span / dv_fine > _MAX_KERNEL_SAMPLES:
        dv_fine = span / _MAX_KERNEL_SAMPLES
    grid = np.arange(off_lo - margin, off_hi + margin + dv_fine, dv_fine)

    x = (grid[None, :] - r_vpv[:, None]) * streamline_length[:, None] / r_vpv[:, None]
    dt_var = np.maximum(alpha_eff[:, None] * np.maximum(x + streamline_length[:, None], 0.0), _DT_FLOOR)
    antideriv = _breakthrough_antideriv(x, dt_var)
    ibar = ((r_vpv[:, None] / streamline_length[:, None]) * antideriv).mean(axis=0)
    return grid, ibar, mean_r_vpv, off_lo, off_hi


def _eval_antideriv(
    dv: npt.NDArray[np.floating],
    grid: npt.NDArray[np.floating],
    ibar: npt.NDArray[np.floating],
    mean_r_vpv: float,
) -> npt.NDArray[np.floating]:
    """Evaluate ``Ibar`` at arbitrary cumulative-volume offsets.

    Linear interpolation on the *uniform* precomputed grid by fractional indexing (no per-point
    search -- the gather is evaluated at ``O(N*band)`` points, so this dominates the runtime).
    Below the grid ``Ibar = 0`` (not broken through); above it ``Ibar = dV - mean_r_vpv`` (saturated,
    the exact linear asymptote).

    Returns
    -------
    ndarray
        ``Ibar(dv)`` with the same shape as ``dv``.
    """
    g0 = grid[0]
    dstep = grid[1] - grid[0]
    f = (dv - g0) / dstep
    # astype truncates toward zero; equals floor on f >= 0 (the only regime that survives the
    # dv < g0 override below and the lower clip), so the floor call is redundant. Precompute the
    # segment slopes once so the O(N*band) gather reads a single array instead of differencing two.
    slopes = np.diff(ibar)
    i0 = np.clip(f.astype(np.intp), 0, grid.size - 2)
    out = ibar[i0] + (f - i0) * slopes[i0]
    out = np.where(dv < g0, 0.0, out)
    return np.where(dv > grid[-1], dv - mean_r_vpv, out)


def _breakthrough_band(
    *,
    cumulative_volume_at_cin: npt.NDArray[np.floating],
    cumulative_volume_at_cout: npt.NDArray[np.floating],
    grid: npt.NDArray[np.floating],
    ibar: npt.NDArray[np.floating],
    mean_r_vpv: float,
    off_lo: float,
    off_hi: float,
    extend: bool,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp]]:
    """Banded breakthrough coefficients (advection + macro + micro + molecular) on the volume grid.

    For each cout bin, gathers the band of cin edges whose breakthrough offset falls in
    ``[off_lo, off_hi]`` (a conservative fixed window from ``searchsorted``, mirroring
    :func:`gwtransport.diffusion_fast._closed_form_coeff_matrix`), evaluates ``Ibar`` at the native
    edge offsets, and telescopes the edge-differences into per-cin-bin coefficients. With
    ``extend`` the cin axis is extended by one wide virtual bin each side carrying the constant
    boundary value (``cin[0]`` / ``cin[-1]``), reproducing the 100-year warm-start.

    The coefficients depend only on the volume grid (not on ``cin``), so this band is built once and
    shared: :func:`infiltration_to_extraction` applies it to the (extended) ``cin``, and
    :func:`extraction_to_infiltration` folds it onto the real cin axis to assemble the banded
    operator it deconvolves -- guaranteeing forward and reverse use the same operator.

    Returns
    -------
    coeff : ndarray, shape (n_cout, band)
        Per-(cout bin, band slot) coefficient.
    cin_bin : ndarray of int, shape (n_cout, band)
        Column index of each coefficient on the (warm-start-extended when ``extend``) cin axis:
        ``cin_ext[cin_bin]`` for the forward, ``clip(cin_bin - int(extend), 0, n_cin - 1)`` for the
        real cin axis in the reverse.
    """
    vi = cumulative_volume_at_cin
    vc = cumulative_volume_at_cout
    if extend:
        big = abs(off_hi) + abs(off_lo) + (vc[-1] - vc[0]) + (vi[-1] - vi[0]) + 1.0
        vi_ext = np.concatenate([[vi[0] - big], vi, [vi[-1] + big]])
        n_cin_ext = vi.size + 1  # (vi.size - 1) real bins + 2 virtual boundary bins
    else:
        vi_ext = vi
        n_cin_ext = vi.size - 1

    vc_lo, vc_hi = vc[:-1], vc[1:]
    w = vc_hi - vc_lo
    vc_mid = 0.5 * (vc_lo + vc_hi)
    # Conservative two-sided fixed window: center on the front, widen to the farthest cin edge whose
    # offset is still inside the band on either side. The +1 absorbs the edge consumed by the
    # telescoping difference and searchsorted rounding (an under-sized window silently drops mass).
    center = np.searchsorted(vi_ext, vc_mid - mean_r_vpv)
    j_lo = np.searchsorted(vi_ext, vc_lo - off_hi, side="left")
    j_hi = np.searchsorted(vi_ext, vc_hi - off_lo, side="right")
    hw_lo = int(np.max(center - j_lo)) + 1
    hw_hi = int(np.max(j_hi - center)) + 1

    cols = center[:, None] + np.arange(-hw_lo, hw_hi + 1)[None, :]
    vi_band = vi_ext[np.clip(cols, 0, len(vi_ext) - 1)]
    ibar_hi = _eval_antideriv(vc_hi[:, None] - vi_band, grid, ibar, mean_r_vpv)
    ibar_lo = _eval_antideriv(vc_lo[:, None] - vi_band, grid, ibar, mean_r_vpv)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_edge = np.where(w[:, None] > 0.0, (ibar_hi - ibar_lo) / w[:, None], 0.0)
    coeff = frac_edge[:, :-1] - frac_edge[:, 1:]
    cin_bin = np.clip(cols[:, :-1], 0, n_cin_ext - 1)
    return coeff, cin_bin


def _build_forward_operator(
    *,
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    flow_out: npt.NDArray[np.floating] | None,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    streamline_length: npt.NDArray[np.floating],
    molecular_diffusivity: npt.NDArray[np.floating],
    longitudinal_dispersivity: npt.NDArray[np.floating],
    retardation_factor: float,
    extend: bool,
    saturation_threshold: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp], npt.NDArray[np.bool_]] | None:
    r"""Build the cin-independent pieces of the approximate banded forward operator ``W``.

    Both directions share this build: the banded breakthrough operator ``W`` (``coeff`` / ``cin_bin``,
    :func:`_breakthrough_band` -- advection + macro + micro + molecular, with molecular diffusion
    folded in as an effective dispersivity) and the residence-time ``valid`` mask. Because the band
    depends only on the volume grid (not on ``cin`` / ``cout``), forward transport and reverse
    deconvolution operate on exactly the same operator.

    Returns
    -------
    coeff : ndarray, shape (n_cout, band)
        Banded ``W`` coefficients.
    cin_bin : ndarray of int, shape (n_cout, band)
        Column indices of ``coeff`` on the warm-start-extended cin axis.
    valid : ndarray of bool, shape (n_cout,)
        Output bins with residence time finite at both edges (complete breakthrough).

    None
        Returned instead of the tuple when there is no through-flow (nothing breaks through).
    """
    tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    cumulative_volume_at_cin = cumulative_flow_volume(flow, dt_to_days(tedges))
    total_volume = float(cumulative_volume_at_cin[-1])
    if total_volume <= 0.0:
        return None

    cumulative_volume_at_cout = _cout_cumulative_volume(
        flow_out=flow_out,
        cout_tedges=cout_tedges,
        cout_tedges_days=cout_tedges_days,
        tedges_days=tedges_days,
        cumulative_volume_at_cin=cumulative_volume_at_cin,
    )

    mean_bin_volume = total_volume / len(flow)
    q_mean = total_volume / float(tedges_days[-1] - tedges_days[0])
    grid, ibar, mean_r_vpv, off_lo, off_hi = _summed_antideriv(
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        q_mean=q_mean,
        mean_bin_volume=mean_bin_volume,
        saturation_threshold=saturation_threshold,
    )
    coeff, cin_bin = _breakthrough_band(
        cumulative_volume_at_cin=cumulative_volume_at_cin,
        cumulative_volume_at_cout=cumulative_volume_at_cout,
        grid=grid,
        ibar=ibar,
        mean_r_vpv=mean_r_vpv,
        off_lo=off_lo,
        off_hi=off_hi,
        extend=extend,
    )

    # Mask bins beyond the data range (and, without warm-start, incompletely-broken-through spin-up
    # bins). residence_time uses the extended grid when warm-starting so spin-up bins stay valid.
    work_tedges = tedges
    if extend:
        edge_values = tedges.to_numpy().copy()
        delta = np.timedelta64(36500, "D")
        edge_values[0] -= delta
        edge_values[-1] += delta
        work_tedges = pd.DatetimeIndex(edge_values)
    # Output bin valid where every streamtube's advective look-back is in-record across the whole
    # bin (advective coverage == 1 for all pore volumes; NaN outside the record -> invalid).
    valid = np.all(
        fraction_explained_full(
            flow=flow,
            tedges=work_tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            retardation_factor=retardation_factor,
            direction="extraction_to_infiltration",
        )
        >= 1.0,
        axis=0,
    )
    return coeff, cin_bin, valid


def _banded_forward_matrix(
    *,
    coeff: npt.NDArray[np.floating],
    cin_bin: npt.NDArray[np.intp],
    extend: bool,
    n_cin: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp]]:
    """Assemble the banded forward operator ``W`` as a per-row contiguous band for ``_solve_reverse_banded``.

    The breakthrough coefficients are scattered from the native band onto the real cin axis, folding
    the warm-start virtual columns into the boundary bins (``clip(cin_bin - int(extend), 0, n_cin - 1)``,
    so ``W @ cin`` equals the forward's ``coeff @ cin_ext`` exactly). The returned band carries the
    forward operator verbatim; ``_solve_reverse_banded`` masks the spin-up rows/columns and normalizes,
    so a forward-then-inverse round trip is self-consistent.

    Returns
    -------
    band_vals : ndarray, shape (n_cout, full_band)
        Forward weights in banded layout (explicit zeros outside each row's support).
    col_start : ndarray of int, shape (n_cout,)
        First real-cin column of each row's band.
    """
    n_cout = coeff.shape[0]
    rows = np.broadcast_to(np.arange(n_cout)[:, None], coeff.shape)
    real_col = np.clip(cin_bin - int(extend), 0, n_cin - 1)
    # COO -> CSR sums the warm-start virtual columns folded onto the boundary real columns.
    w_mat = coo_array((coeff.ravel(), (rows.ravel(), real_col.ravel())), shape=(n_cout, n_cin)).tocsr()

    # CSR -> contiguous per-row band. Each row spans one contiguous cin band (the fold only saturates
    # the ends onto the boundary columns), so the banded layout carries no spurious interior gaps.
    w_mat.sort_indices()
    indptr, indices, data = w_mat.indptr, w_mat.indices, w_mat.data
    row_counts = np.diff(indptr)
    nonempty = row_counts > 0
    col_start = np.zeros(n_cout, dtype=np.intp)
    last_col = np.zeros(n_cout, dtype=np.intp)
    col_start[nonempty] = indices[indptr[:-1][nonempty]]
    last_col[nonempty] = indices[indptr[1:][nonempty] - 1]
    full_band = int((last_col[nonempty] - col_start[nonempty] + 1).max()) if nonempty.any() else 1

    band_vals = np.zeros((n_cout, full_band))
    rows_of_nz = np.repeat(np.arange(n_cout), row_counts)
    band_vals[rows_of_nz, indices - col_start[rows_of_nz]] = data
    return band_vals, col_start


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: npt.NDArray[np.floating] | float,
    molecular_diffusivity: npt.NDArray[np.floating] | float,
    longitudinal_dispersivity: npt.NDArray[np.floating] | float,
    retardation_factor: float = 1.0,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | None = "constant",
    saturation_threshold: float = _DEFAULT_SATURATION_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """Compute extracted concentration with advection, microdispersion, and molecular diffusion (approximate).

    Fast *approximate* counterpart of :func:`gwtransport.diffusion_fast.infiltration_to_extraction`.
    Advection + macrodispersion + microdispersion (``alpha_L``) and molecular diffusion (``D_m``,
    folded in as an effective dispersivity ``alpha_L + D_m*r_vpv/(L*q_mean)`` per streamtube) form a
    single exact skewed Kreft-Zuber breakthrough on the native cumulative-volume grid. At **constant
    flow** the result reproduces :func:`gwtransport.diffusion_fast.infiltration_to_extraction` to the
    ``Ibar`` interpolation floor (~1e-6 smooth, ~1e-4 sharp) at realistic Peclet numbers
    (``Pe = L/alpha_eff >> 1``), including heat (``R > 1``, large ``D_m``); only the extreme low-Peclet
    corner (``Pe <~ 2``) loses breakthrough-shape accuracy to the fine-grid sample cap. Under
    **variable flow** the molecular part carries a commutator residual
    from the frozen ``q_mean``: small when molecular diffusion is sub-dominant (``alpha_L > 0``, <~1e-3
    for realistic solute ``D_m``), growing to the multi-percent range for sharp inputs in the
    molecular-diffusion-dominated corner (``alpha_L`` ~ 0) under strongly variable flow. For machine
    precision -- or that corner -- use :mod:`gwtransport.diffusion_fast`.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in the infiltrating water. Length ``len(tedges) - 1``.
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. Length ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Length ``len(cin) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Length ``len(output) + 1``.
    aquifer_pore_volumes : array-like
        Aquifer pore volumes [m³] -- one independent streamtube per entry. Any distribution shape
        (the APVD is pre-summed exactly).
    streamline_length : float or ndarray
        Travel distance L [m]: a scalar (shared by all streamtubes) or an array with one
        value per aquifer pore volume. Must be positive.
    molecular_diffusivity : float or ndarray
        Effective molecular diffusivity D_m [m²/day]: scalar or one value per pore volume.
        Must be non-negative.
    longitudinal_dispersivity : float or ndarray
        Longitudinal dispersivity alpha_L [m] (microdispersion): scalar or one value per pore volume.
        Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    flow_out : array-like or None, optional
        Extraction flow rate [m³/day] on the output grid (aligned to ``cout_tedges``,
        length ``len(cout_tedges) - 1``). Required when ``cout_tedges`` differs from ``tedges``;
        may be omitted only when ``cout_tedges`` equals ``tedges``. Default None.
    spinup : {"constant"} | None, optional
        ``"constant"`` (default) extends ``tedges`` by 100 years on each side so a constant
        warm-start fills the left-edge spin-up region; ``None`` leaves spin-up cout as NaN.
    saturation_threshold : float, optional
        Breakthrough-band cutoff ``U`` (default 7.0). Sets how far into the breakthrough tail the
        banded build reaches; see :func:`gwtransport.diffusion_fast.infiltration_to_extraction`.

    Returns
    -------
    numpy.ndarray
        Bin-averaged Kreft-Zuber flux concentration ``C_F`` in the extracted water. Length
        ``len(cout_tedges) - 1``. NaN where no infiltration data has broken through.

    See Also
    --------
    gwtransport.diffusion_fast.infiltration_to_extraction : Exact (machine-precision) counterpart;
        use it when approximation is unacceptable, especially in the molecular-dominant regime.
    gwtransport.diffusion.infiltration_to_extraction : Quadrature reference.
    extraction_to_infiltration : Inverse operation (deconvolves this same operator).
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion.
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    if flow_out is not None:
        flow_out = np.asarray(flow_out, dtype=float)

    _validate_inputs(
        cin_or_cout=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        is_forward=True,
        flow_out=flow_out,
    )

    n_pv = len(aquifer_pore_volumes)
    streamline_length = _broadcast_to_pore_volumes(streamline_length, n_pv)
    molecular_diffusivity = _broadcast_to_pore_volumes(molecular_diffusivity, n_pv)
    longitudinal_dispersivity = _broadcast_to_pore_volumes(longitudinal_dispersivity, n_pv)

    n_cout = len(cout_tedges) - 1
    extend = _extend_tedges_flag(spinup)
    operator = _build_forward_operator(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow_out=flow_out,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        extend=extend,
        saturation_threshold=saturation_threshold,
    )
    if operator is None:
        # No through-flow: nothing breaks through (matches diffusion_fast's all-NaN result).
        return np.full(n_cout, np.nan)
    coeff, cin_bin, valid = operator

    # Apply the banded breakthrough operator to the (warm-start-extended) cin. Output bins with no
    # through-flow carry coeff.sum() ~= 0, so the support mask NaNs them out; the band adds zero volume
    # across a zero-flow gap, so a constant cin stays constant across it (no smear to leak).
    cin_ext = np.concatenate([[cin[0]], cin, [cin[-1]]]) if extend else cin
    cout = np.einsum("kb,kb->k", coeff, cin_ext[cin_bin])
    support = coeff.sum(axis=1) >= _EPSILON_COEFF_SUM
    return np.where(support & valid, cout, np.nan)


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: npt.NDArray[np.floating] | float,
    molecular_diffusivity: npt.NDArray[np.floating] | float,
    longitudinal_dispersivity: npt.NDArray[np.floating] | float,
    retardation_factor: float = 1.0,
    regularization_strength: float = 1e-10,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | None = "constant",
    saturation_threshold: float = _DEFAULT_SATURATION_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """Reconstruct infiltration concentration from extracted water (fast approximate deconvolution).

    Inverts the **same** approximate operator ``W`` the forward applies: it assembles the banded
    breakthrough operator ``W`` (advection + macro + micro + molecular, molecular folded in as an
    effective dispersivity) directly in banded form and deconvolves it with banded Tikhonov
    regularization (``_solve_reverse_banded`` -- banded Cholesky on the normal equations,
    ``O(n * band**2)``). It builds ``W`` from one ``Ibar`` gather -- no per-pore-volume closed-form
    loop and no dense ``(n_cout, n_cin)`` matrix -- so it is much faster than
    :func:`gwtransport.diffusion_fast.extraction_to_infiltration` (which evaluates the exact
    breakthrough per streamtube), especially for many streamtubes. Because the deconvolved operator is
    exactly the forward operator, a forward-then-inverse round trip recovers ``cin`` up to the
    deconvolution conditioning and regularization. On real extraction data the reverse is as accurate
    as the forward: at constant flow it matches
    :func:`gwtransport.diffusion_fast.extraction_to_infiltration` to ~1e-6, while the
    molecular-diffusion-dominated corner under strongly variable flow amplifies the forward's
    commutator residual (use :mod:`gwtransport.diffusion_fast` there).

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water. Length ``len(cout_tedges) - 1``.
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. Length ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Length ``len(flow) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Length ``len(cout) + 1``.
    aquifer_pore_volumes : array-like
        Aquifer pore volumes [m³] -- one independent streamtube per entry.
    streamline_length : float or ndarray
        Travel distance L [m]: scalar or one value per pore volume. Must be positive.
    molecular_diffusivity : float or ndarray
        Effective molecular diffusivity D_m [m²/day]: scalar or one value per pore volume.
        Must be non-negative.
    longitudinal_dispersivity : float or ndarray
        Longitudinal dispersivity alpha_L [m] (microdispersion): scalar or one value per pore volume.
        Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    regularization_strength : float, optional
        Tikhonov regularization parameter (default 1e-10). Must be strictly positive: the banded
        solver relies on it to make the normal equations positive-definite (it cannot return the
        dense ``lambda = 0`` minimum-norm solution).
    flow_out : array-like or None, optional
        Extraction flow rate [m³/day] on the output grid (aligned to ``cout_tedges``). See
        :func:`infiltration_to_extraction`. Default None.
    spinup : {"constant"} | None, optional
        See :func:`infiltration_to_extraction`. Default ``"constant"``.
    saturation_threshold : float, optional
        See :func:`infiltration_to_extraction`. Default 7.0.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Length ``len(tedges) - 1``.
        NaN where no extraction data constrains the bin.

    See Also
    --------
    infiltration_to_extraction : Forward operation (the operator inverted here).
    gwtransport.diffusion_fast.extraction_to_infiltration : Exact (machine-precision) counterpart;
        use it when the approximation is unacceptable.
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion.
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    cout = np.asarray(cout, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    if flow_out is not None:
        flow_out = np.asarray(flow_out, dtype=float)

    _validate_inputs(
        cin_or_cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        is_forward=False,
        flow_out=flow_out,
    )

    n_pv = len(aquifer_pore_volumes)
    streamline_length = _broadcast_to_pore_volumes(streamline_length, n_pv)
    molecular_diffusivity = _broadcast_to_pore_volumes(molecular_diffusivity, n_pv)
    longitudinal_dispersivity = _broadcast_to_pore_volumes(longitudinal_dispersivity, n_pv)

    n_cin = len(tedges) - 1
    extend = _extend_tedges_flag(spinup)
    operator = _build_forward_operator(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow_out=flow_out,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        extend=extend,
        saturation_threshold=saturation_threshold,
    )
    if operator is None:
        # No through-flow: nothing constrains the infiltration signal.
        return np.full(n_cin, np.nan)
    coeff, cin_bin, valid = operator

    band_vals, col_start = _banded_forward_matrix(coeff=coeff, cin_bin=cin_bin, extend=extend, n_cin=n_cin)
    return _solve_reverse_banded(
        band_vals=band_vals,
        col_start=col_start,
        valid_cout_bins=valid,
        cout=cout,
        n_cin=n_cin,
        regularization_strength=regularization_strength,
    )


def gamma_infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    streamline_length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | None = "constant",
    saturation_threshold: float = _DEFAULT_SATURATION_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """Compute extracted concentration for a gamma-distributed pore volume distribution (approximate).

    Convenience wrapper around :func:`infiltration_to_extraction` that discretizes a (shifted)
    gamma aquifer pore-volume distribution into ``n_bins`` equal-probability streamtubes. Provide
    either (mean, std) or (alpha, beta); ``loc`` defaults to 0. Approximate -- see
    :func:`infiltration_to_extraction`.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m³/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Length ``len(cin) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Length ``len(result) + 1``.
    mean, std : float, optional
        Mean and standard deviation of the gamma pore-volume distribution.
    loc : float, optional
        Location (minimum pore volume), ``0 <= loc < mean``. Default 0.0.
    alpha, beta : float, optional
        Shape and scale parameters of the gamma distribution (alternative to mean/std).
    n_bins : int, optional
        Number of equal-probability streamtubes. Default 100.
    streamline_length : float
        Travel distance L [m], applied to all gamma streamtubes. Must be positive.
    molecular_diffusivity : float
        Effective molecular diffusivity D_m [m²/day], applied to all streamtubes. Must be
        non-negative.
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m] (microdispersion), applied to all streamtubes. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    flow_out : array-like or None, optional
        Extraction flow rate [m³/day] on the output grid. See
        :func:`infiltration_to_extraction`. Default None.
    spinup : {"constant"} | None, optional
        See :func:`infiltration_to_extraction`. Default ``"constant"``.
    saturation_threshold : float, optional
        See :func:`infiltration_to_extraction`. Default 7.0.

    Returns
    -------
    numpy.ndarray
        Bin-averaged Kreft-Zuber flux concentration ``C_F`` in the extracted water.
        Length ``len(cout_tedges) - 1``. NaN where no infiltration data has broken through.

    See Also
    --------
    infiltration_to_extraction : Transport with an explicit pore volume distribution.
    gamma_extraction_to_infiltration : Reverse operation.
    gwtransport.gamma.bins : Create gamma distribution bins.
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model.
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        flow_out=flow_out,
        spinup=spinup,
        saturation_threshold=saturation_threshold,
    )


def gamma_extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    streamline_length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    regularization_strength: float = 1e-10,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | None = "constant",
    saturation_threshold: float = _DEFAULT_SATURATION_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """Reconstruct infiltration concentration for a gamma-distributed pore volume distribution.

    Convenience wrapper around :func:`extraction_to_infiltration` that discretizes a (shifted)
    gamma aquifer pore-volume distribution into ``n_bins`` equal-probability streamtubes. Provide
    either (mean, std) or (alpha, beta); ``loc`` defaults to 0. Fast approximate banded deconvolution
    (see :func:`extraction_to_infiltration`).

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water.
    flow : array-like
        Flow rate of water in the aquifer [m³/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Length ``len(flow) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Length ``len(cout) + 1``.
    mean, std : float, optional
        Mean and standard deviation of the gamma pore-volume distribution.
    loc : float, optional
        Location (minimum pore volume), ``0 <= loc < mean``. Default 0.0.
    alpha, beta : float, optional
        Shape and scale parameters of the gamma distribution (alternative to mean/std).
    n_bins : int, optional
        Number of equal-probability streamtubes. Default 100.
    streamline_length : float
        Travel distance L [m], applied to all gamma streamtubes. Must be positive.
    molecular_diffusivity : float
        Effective molecular diffusivity D_m [m²/day], applied to all streamtubes. Must be
        non-negative.
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m] (microdispersion), applied to all streamtubes. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    regularization_strength : float, optional
        Tikhonov regularization parameter (default 1e-10).
    flow_out : array-like or None, optional
        Extraction flow rate [m³/day] on the output grid. See
        :func:`infiltration_to_extraction`. Default None.
    spinup : {"constant"} | None, optional
        See :func:`infiltration_to_extraction`. Default ``"constant"``.
    saturation_threshold : float, optional
        See :func:`infiltration_to_extraction`. Default 7.0.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Length ``len(tedges) - 1``.

    See Also
    --------
    extraction_to_infiltration : Deconvolution with an explicit pore volume distribution.
    gamma_infiltration_to_extraction : Forward operation.
    gwtransport.gamma.bins : Create gamma distribution bins.
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model.
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        regularization_strength=regularization_strength,
        flow_out=flow_out,
        spinup=spinup,
        saturation_threshold=saturation_threshold,
    )
