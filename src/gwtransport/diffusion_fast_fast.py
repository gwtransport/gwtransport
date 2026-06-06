"""
Fast *approximate* 1D advection-dispersion transport (Kreft-Zuber flux concentration).

This module computes the same physics as :mod:`gwtransport.diffusion_fast` -- the bin-averaged
Kreft-Zuber (1978) flux concentration ``C_F`` on a bundle of 1D streamtubes -- but trades exactness
for a single fast (~1.5 ms) native-grid evaluation that does not depend on the flow being constant.
It is **approximate**: where :mod:`gwtransport.diffusion_fast` reproduces the quadrature reference to
machine precision, this module is accurate to ~3e-4 in the common regime and degrades in a
documented corner (below). When you need machine precision, use :mod:`gwtransport.diffusion_fast`.

How it works -- an operator split in two coordinates
----------------------------------------------------

The moving-frame dispersion product ``D_t = D_m*tau + alpha_L*xi`` mixes a *time* term (molecular
diffusion ``D_m*tau``) and a *volume* term (microdispersion ``alpha_L*xi``). The two are split into
the coordinate each is stationary in, so the dominant part is built once and is flow-independent:

1. **Advection + macrodispersion + microdispersion** are the *exact* skewed ``D_m=0`` Kreft-Zuber
   breakthrough, applied banded on the **native cumulative-volume grid**. The whole aquifer pore
   volume distribution (APVD) is pre-summed into a single 1D antiderivative ``Ibar(dV)`` -- exact
   for any APVD shape -- finely sampled once and read back by interpolation. This part is
   volume-stationary, hence **flow-independent** (constant and strongly variable flow alike).
2. **Molecular diffusion** is a symmetric **time-domain Gaussian** applied to the outlet signal
   (variance ``2*D_m*tau_bt*(R*Vbar/L)^2/Q^2``). This is the only modelling approximation: the true
   Kreft-Zuber molecular breakthrough is skewed, and at realistic (sub-bin) spreading the Gaussian
   is nearly a no-op, so the molecular term is dropped rather than skewed.

``tedges`` need **not** be regularly spaced and ``cout_tedges`` need not equal ``tedges`` (supply
``flow_out`` when they differ): step 1 runs on the native cumulative-volume grid for any spacing.
Only the molecular Gaussian assumes a roughly regular grid -- it convolves in bin-index space using
the mean bin width -- so a strongly irregular grid adds a small extra error to the (usually
sub-dominant) molecular term; use :mod:`gwtransport.diffusion_fast` for the molecular-dominated +
irregular-grid corner.

Accuracy (vs :mod:`gwtransport.diffusion_fast`, flow-independent unless noted)
------------------------------------------------------------------------------

- **~3e-4 whenever mechanical dispersion is present** (``alpha_L > 0`` -- the typical groundwater
  regime, Peclet number >> 1), constant *and* variable flow. Here molecular diffusion is
  sub-dominant, so approximating it barely matters.
- In the **molecular-diffusion-dominated** corner (``alpha_L`` ~ 0): ~1e-4 for smooth inputs, but
  degrading to ~1e-2 for sharp inputs (and ~5e-2 for sharp inputs with a very wide / bimodal APVD or
  a large single pore volume), because the symmetric time-Gaussian cannot reproduce the skewed
  molecular breakthrough. **Use** :mod:`gwtransport.diffusion_fast` **for exact results in this
  regime.**

The inverse (:func:`extraction_to_infiltration`) deconvolves the *same* approximate operator the
forward applies. It assembles ``W = G . M`` directly in banded form (one ``Ibar`` gather plus a
sparse ``G . M`` product -- no per-pore-volume closed-form loop, no dense ``(n_cout, n_cin)`` matrix)
and solves it with banded Tikhonov regularisation (banded Cholesky, ``O(n * band**2)``), so it is
much faster than :mod:`gwtransport.diffusion_fast`'s reverse, especially for many streamtubes.
Inverting exactly the forward operator makes a round trip self-consistent: it recovers the input up
to the deconvolution conditioning, with the only error being the forward operator's approximation of
:mod:`gwtransport.diffusion_fast` (use that module when the approximation is unacceptable).

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
from scipy.ndimage import gaussian_filter1d
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
from gwtransport.residence_time import residence_time
from gwtransport.utils import cumulative_flow_volume

# Samples per native bin used to discretise the 1D breakthrough antiderivative ``Ibar``. Higher =
# more accurate ``Ibar`` interpolation (advection+micro error ~ O(1/_KERNEL_FINE^2)) at the cost of
# a larger one-time precompute; 16 gives ~1e-4 in the advection+micro part, which is below the
# molecular-Gaussian floor, so it is not exposed as a user knob.
_KERNEL_FINE = 16

# Upper bound on the number of fine ``Ibar`` samples. Caps the one-time precompute when the
# breakthrough band spans far more than the record (e.g. tiny flow -> enormous front offset); the
# affected output bins are masked by residence time anyway, so coarsening the band there is benign.
_MAX_KERNEL_SAMPLES = 20000


def _summed_antideriv(
    *,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    streamline_length: npt.NDArray[np.floating],
    longitudinal_dispersivity: npt.NDArray[np.floating],
    retardation_factor: float,
    mean_bin_volume: float,
    saturation_threshold: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float, float, float]:
    r"""Precompute the APVD-summed breakthrough antiderivative ``Ibar(dV)`` on a fine 1D grid.

    For one streamtube the bin-averaged ``D_m=0`` flux fraction over a cout bin equals the second
    difference of the antiderivative ``I(x)`` (:func:`gwtransport._diffusion_shared._breakthrough_antideriv`)
    of the resident concentration in the breakthrough coordinate ``x = (dV - r_vpv)*L/r_vpv``
    (``r_vpv = R*V_pore``). The antiderivative *with respect to cumulative volume* ``dV`` is
    ``(r_vpv/L)*I(x(dV))``; averaging it over the APVD gives a single 1D function

    .. math::

        \bar I(\Delta V) = \operatorname{mean}_{pv}\Bigl[\tfrac{r_{vpv}}{L}\,
        I\bigl((\Delta V - r_{vpv})L/r_{vpv}\bigr)\Bigr],\quad D_t = \alpha_L\,\xi,

    whose edge-differences reproduce the per-streamtube-averaged ``C_F`` **exactly** for any APVD
    (the ``r_vpv/L`` Jacobian and the cout-bin volume normalisation cancel the per-streamtube ``dx``).
    Only the interpolation of ``Ibar`` is approximate.

    The grid is **uniform** over the breakthrough band ``[off_lo, off_hi]`` (front ``r_vpv`` plus the
    conservative ``alpha_L`` dispersion smear, unioned over streamtubes) plus a margin, sampled at
    ``mean_bin_volume / _KERNEL_FINE`` (uniformity lets :func:`_eval_antideriv` interpolate by
    fractional indexing instead of a per-point search). For ``alpha_L > 0`` ``Ibar`` is smooth and
    the interpolation error is ``O(1/_KERNEL_FINE^2)``; at ``alpha_L = 0`` it has a kink at each
    ``r_vpv`` and the error is ``O(1/_KERNEL_FINE)`` (sub-1e-3, well below the molecular floor).

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
    # D_m=0 dispersion half-widths in the breakthrough coordinate x (front D_t = alpha_L*L): the
    # pre side shrinks (|x| = U*2*sqrt(alpha_L*L)); the post side grows with slope alpha_L, giving
    # the quadratic root x = 2U^2*alpha_L + 2U*sqrt(U^2*alpha_L^2 + alpha_L*L). Mapped to volume
    # offsets via r_vpv/L and unioned over streamtubes (conservative, never under-covers the band).
    pre_x = u * 2.0 * np.sqrt(longitudinal_dispersivity * streamline_length)
    post_x = 2.0 * u * u * longitudinal_dispersivity + 2.0 * u * np.sqrt(
        u * u * longitudinal_dispersivity**2 + longitudinal_dispersivity * streamline_length
    )
    off_lo = float(np.min(r_vpv - (r_vpv / streamline_length) * pre_x))
    off_hi = float(np.max(r_vpv + (r_vpv / streamline_length) * post_x))

    margin = 4.0 * mean_bin_volume
    span = (off_hi - off_lo) + 2.0 * margin
    dv_fine = mean_bin_volume / _KERNEL_FINE
    if span / dv_fine > _MAX_KERNEL_SAMPLES:
        dv_fine = span / _MAX_KERNEL_SAMPLES
    grid = np.arange(off_lo - margin, off_hi + margin + dv_fine, dv_fine)

    x = (grid[None, :] - r_vpv[:, None]) * streamline_length[:, None] / r_vpv[:, None]
    dt_var = np.maximum(longitudinal_dispersivity[:, None] * np.maximum(x + streamline_length[:, None], 0.0), _DT_FLOOR)
    antideriv, _, _ = _breakthrough_antideriv(x, dt_var)
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
    i0 = np.clip(np.floor(f).astype(np.intp), 0, grid.size - 2)
    out = ibar[i0] + (f - i0) * (ibar[i0 + 1] - ibar[i0])
    out = np.where(dv < g0, 0.0, out)
    return np.where(dv > grid[-1], dv - mean_r_vpv, out)


def _advection_micro_band(
    *,
    cumulative_volume_at_cin: npt.NDArray[np.floating],
    cumulative_volume_at_cout: npt.NDArray[np.floating],
    grid: npt.NDArray[np.floating],
    ibar: npt.NDArray[np.floating],
    mean_r_vpv: float,
    off_lo: float,
    off_hi: float,
    warmstart: bool,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp]]:
    """Banded ``D_m=0`` advection+macro+micro coefficients on the native cumulative-volume grid.

    For each cout bin, gathers the band of cin edges whose breakthrough offset falls in
    ``[off_lo, off_hi]`` (a conservative fixed window from ``searchsorted``, mirroring
    :func:`gwtransport.diffusion_fast._closed_form_coeff_matrix`), evaluates ``Ibar`` at the native
    edge offsets, and telescopes the edge-differences into per-cin-bin coefficients. With
    ``warmstart`` the cin axis is extended by one wide virtual bin each side carrying the constant
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
        Column index of each coefficient on the (warm-start-extended when ``warmstart``) cin axis:
        ``cin_ext[cin_bin]`` for the forward, ``clip(cin_bin - warmstart, 0, n_cin - 1)`` for the
        real cin axis in the reverse.
    """
    vi = cumulative_volume_at_cin
    vc = cumulative_volume_at_cout
    if warmstart:
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


def _valid_cout_bins(
    *,
    flow: npt.NDArray[np.floating],
    work_tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.bool_]:
    """Output bins with complete breakthrough information for every streamtube.

    Uses the same residence-time validity criterion as
    :func:`gwtransport.diffusion_fast._closed_form_coeff_matrix`: a cout bin is valid when the
    residence time is finite (not beyond the flow record) at both of its edges for all streamtubes.
    ``work_tedges`` is the 100-year-extended grid when warm-starting (so spin-up bins are valid) and
    the raw grid otherwise.

    Returns
    -------
    ndarray of bool, shape (len(cout_tedges) - 1,)
        True where the output bin is fully informed.
    """
    rt = residence_time(
        flow=flow,
        flow_tedges=work_tedges,
        index=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    return ~np.any(np.isnan(rt[:, :-1]) | np.isnan(rt[:, 1:]), axis=0)


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
) -> (
    tuple[npt.NDArray[np.floating], npt.NDArray[np.intp], float, npt.NDArray[np.bool_], npt.NDArray[np.floating]] | None
):
    r"""Build the cin-independent pieces of the approximate banded forward operator ``W = G . M``.

    Both directions share this build: the advection+macro+micro band ``M`` (``coeff`` / ``cin_bin``,
    :func:`_advection_micro_band`), the molecular time-Gaussian width ``sigma_bins`` (the operator
    ``G``), the residence-time ``valid`` mask, and the per-row coefficient sum ``row_sum``. Because
    the band depends only on the volume grid (not on ``cin`` / ``cout``), forward transport and
    reverse deconvolution operate on exactly the same operator.

    Returns
    -------
    coeff : ndarray, shape (n_cout, band)
        Banded ``M`` coefficients.
    cin_bin : ndarray of int, shape (n_cout, band)
        Column indices of ``coeff`` on the warm-start-extended cin axis.
    sigma_bins : float
        Molecular Gaussian width in output-bin units (0 -> ``G`` is the identity).
    valid : ndarray of bool, shape (n_cout,)
        Output bins with residence time finite at both edges (complete breakthrough).
    row_sum : ndarray, shape (n_cout,)
        Per-cout-bin coefficient sum of ``M`` (1 where fully broken through).

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

    n_cout = len(cout_tedges) - 1
    mean_bin_volume = total_volume / len(flow)
    grid, ibar, mean_r_vpv, off_lo, off_hi = _summed_antideriv(
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        mean_bin_volume=mean_bin_volume,
        saturation_threshold=saturation_threshold,
    )
    coeff, cin_bin = _advection_micro_band(
        cumulative_volume_at_cin=cumulative_volume_at_cin,
        cumulative_volume_at_cout=cumulative_volume_at_cout,
        grid=grid,
        ibar=ibar,
        mean_r_vpv=mean_r_vpv,
        off_lo=off_lo,
        off_hi=off_hi,
        warmstart=extend,
    )

    # Molecular diffusion: a single mean-streamtube time-domain Gaussian on the outlet signal.
    # sigma_t^2 = 2*D_m*tau_bt*(r_vpv/L)^2/Q^2, with tau_bt = R*mean(V_pore)/Q the mean breakthrough
    # time and Q = total_volume/total_time the flow-weighted mean throughflow. sigma_t (days) is
    # converted with the OUTPUT-grid mean bin width (not the flow grid -- they differ for a coarse
    # cout grid); a smear wider than the record cannot be resolved, so it is capped at n_cout.
    total_days = float(tedges_days[-1] - tedges_days[0])
    q_mean = total_volume / total_days
    tau_bt = retardation_factor * float(aquifer_pore_volumes.mean()) / q_mean
    sigma_t2 = (
        2.0 * float(molecular_diffusivity.mean()) * tau_bt * (mean_r_vpv / float(streamline_length.mean())) ** 2
    ) / q_mean**2
    mean_cout_dt = (cout_tedges_days[-1] - cout_tedges_days[0]) / n_cout
    sigma_bins = min(float(np.sqrt(sigma_t2)) / mean_cout_dt, float(n_cout))

    # Mask bins beyond the data range (and, without warm-start, incompletely-broken-through spin-up
    # bins). residence_time uses the extended grid when warm-starting so spin-up bins stay valid.
    work_tedges = tedges
    if extend:
        edge_values = tedges.to_numpy().copy()
        delta = np.timedelta64(36500, "D")
        edge_values[0] -= delta
        edge_values[-1] += delta
        work_tedges = pd.DatetimeIndex(edge_values)
    valid = _valid_cout_bins(
        flow=flow,
        work_tedges=work_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
    )
    return coeff, cin_bin, sigma_bins, valid, coeff.sum(axis=1)


def _banded_forward_matrix(
    *,
    coeff: npt.NDArray[np.floating],
    cin_bin: npt.NDArray[np.intp],
    extend: bool,
    n_cin: int,
    sigma_bins: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp]]:
    """Assemble ``W = G . M`` as a per-row contiguous band for :func:`_solve_reverse_banded`.

    ``M`` (advection + macro + microdispersion) is scattered from the native band onto the real cin
    axis, folding the warm-start virtual columns into the boundary bins
    (``clip(cin_bin - warmstart, 0, n_cin - 1)``, so ``M @ cin`` equals the forward's
    ``coeff @ cin_ext`` exactly). ``G`` is the molecular time-Gaussian along the output-bin axis (the
    same ``mode="nearest"`` kernel the forward applies with :func:`scipy.ndimage.gaussian_filter1d`).
    The returned band carries the forward operator verbatim; :func:`_solve_reverse_banded` masks the
    spin-up rows/columns and normalizes, so a forward-then-inverse round trip is self-consistent.

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
    m_mat = coo_array((coeff.ravel(), (rows.ravel(), real_col.ravel())), shape=(n_cout, n_cin)).tocsr()

    lw = min(int(6.0 * sigma_bins + 0.5), n_cout - 1) if sigma_bins > 0.0 else 0
    if lw > 0:
        offsets = np.arange(-lw, lw + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel /= kernel.sum()
        g_rows = np.repeat(np.arange(n_cout), offsets.size)
        g_cols = np.clip(np.arange(n_cout)[:, None] + offsets[None, :], 0, n_cout - 1).ravel()
        g_mat = coo_array((np.tile(kernel, n_cout), (g_rows, g_cols)), shape=(n_cout, n_cout)).tocsr()
        w_mat = (g_mat @ m_mat).tocsr()
    else:
        w_mat = m_mat

    # CSR -> contiguous per-row band. Each W row is the union of overlapping contiguous M bands, so
    # it stays contiguous (a flow spike that briefly opens a gap only adds explicit interior zeros).
    w_mat.sort_indices()
    indptr, indices, data = w_mat.indptr, w_mat.indices, w_mat.data
    nonempty = np.diff(indptr) > 0
    col_start = np.zeros(n_cout, dtype=np.intp)
    last_col = np.zeros(n_cout, dtype=np.intp)
    col_start[nonempty] = indices[indptr[:-1][nonempty]]
    last_col[nonempty] = indices[indptr[1:][nonempty] - 1]
    full_band = int((last_col[nonempty] - col_start[nonempty] + 1).max()) if nonempty.any() else 1

    band_vals = np.zeros((n_cout, full_band))
    rows_of_nz = np.repeat(np.arange(n_cout), np.diff(indptr))
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
    spinup: str | float | None = "constant",
    saturation_threshold: float = _DEFAULT_SATURATION_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """Compute extracted concentration with advection and longitudinal dispersion (approximate).

    Fast *approximate* counterpart of :func:`gwtransport.diffusion_fast.infiltration_to_extraction`.
    The advection + macrodispersion + microdispersion (``alpha_L``) part is the exact skewed
    ``D_m=0`` Kreft-Zuber breakthrough applied on the native cumulative-volume grid; molecular
    diffusion (``D_m``) is a symmetric time-domain Gaussian. The result is flow-independent and
    accurate to ~3e-4 whenever ``alpha_L > 0`` (the typical regime), degrading to ~1e-2 (sharp
    inputs) in the molecular-diffusion-dominated corner (``alpha_L`` ~ 0). For machine precision, use
    :mod:`gwtransport.diffusion_fast`.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in the infiltrating water. Length ``len(tedges) - 1``.
    flow : array-like
        Flow rate of water in the aquifer [m3/day]. Length ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Length ``len(cin) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Length ``len(output) + 1``.
    aquifer_pore_volumes : array-like
        Aquifer pore volumes [m3] -- one independent streamtube per entry. Any distribution shape
        (the APVD is pre-summed exactly).
    streamline_length : float or ndarray
        Travel distance L [m]: a scalar (shared by all streamtubes) or an array with one
        value per aquifer pore volume. Must be positive.
    molecular_diffusivity : float or ndarray
        Effective molecular diffusivity D_m [m2/day]: scalar or one value per pore volume.
        Must be non-negative.
    longitudinal_dispersivity : float or ndarray
        Longitudinal dispersivity alpha_L [m]: scalar or one value per pore volume.
        Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid (aligned to ``cout_tedges``,
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
    coeff, cin_bin, sigma_bins, valid, row_sum = operator

    # Apply the advection+macro+micro band M to the (warm-start-extended) cin, then the molecular
    # time-Gaussian G. The Gaussian runs before masking so it operates on NaN-free values.
    cin_ext = np.concatenate([[cin[0]], cin, [cin[-1]]]) if extend else cin
    cout_micro = (coeff * cin_ext[cin_bin]).sum(axis=1)
    cout = cout_micro if sigma_bins == 0.0 else gaussian_filter1d(cout_micro, sigma_bins, mode="nearest", truncate=6.0)
    return np.where((row_sum >= _EPSILON_COEFF_SUM) & valid, cout, np.nan)


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
    spinup: str | float | None = "constant",
    saturation_threshold: float = _DEFAULT_SATURATION_THRESHOLD,
) -> npt.NDArray[np.floating]:
    """Reconstruct infiltration concentration from extracted water (fast approximate deconvolution).

    Inverts the **same** approximate operator the forward applies: it assembles ``W = G . M`` (the
    advection+macro+micro band ``M`` times the molecular time-Gaussian ``G``) directly in banded form
    and deconvolves it with banded Tikhonov regularization (:func:`_solve_reverse_banded` -- banded
    Cholesky on the normal equations, ``O(n * band**2)``). It builds ``W`` from one ``Ibar`` gather
    plus a sparse ``G . M`` product -- no per-pore-volume closed-form loop and no dense
    ``(n_cout, n_cin)`` matrix -- so it is much faster than
    :func:`gwtransport.diffusion_fast.extraction_to_infiltration` (which evaluates the exact
    breakthrough per streamtube), especially for many streamtubes. Because the deconvolved operator
    is exactly the forward operator, a forward-then-inverse round trip recovers ``cin`` up to the
    deconvolution conditioning and regularization; the approximation lives entirely in the forward
    operator vs :mod:`gwtransport.diffusion_fast`.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water. Length ``len(cout_tedges) - 1``.
    flow : array-like
        Flow rate of water in the aquifer [m3/day]. Length ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Length ``len(flow) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Length ``len(cout) + 1``.
    aquifer_pore_volumes : array-like
        Aquifer pore volumes [m3] -- one independent streamtube per entry.
    streamline_length : float or ndarray
        Travel distance L [m]: scalar or one value per pore volume. Must be positive.
    molecular_diffusivity : float or ndarray
        Effective molecular diffusivity D_m [m2/day]: scalar or one value per pore volume.
        Must be non-negative.
    longitudinal_dispersivity : float or ndarray
        Longitudinal dispersivity alpha_L [m]: scalar or one value per pore volume.
        Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    regularization_strength : float, optional
        Tikhonov regularization parameter (default 1e-10). Must be strictly positive: the banded
        solver relies on it to make the normal equations positive-definite (it cannot return the
        dense ``lambda = 0`` minimum-norm solution).
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid (aligned to ``cout_tedges``). See
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
    coeff, cin_bin, sigma_bins, valid, _row_sum = operator

    band_vals, col_start = _banded_forward_matrix(
        coeff=coeff, cin_bin=cin_bin, extend=extend, n_cin=n_cin, sigma_bins=sigma_bins
    )
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
    spinup: str | float | None = "constant",
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
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Length ``len(cin) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins.
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
        Effective molecular diffusivity D_m [m2/day], applied to all streamtubes. Must be
        non-negative.
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m], applied to all streamtubes. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid. See
        :func:`infiltration_to_extraction`. Default None.
    spinup : {"constant"} | None, optional
        See :func:`infiltration_to_extraction`. Default ``"constant"``.
    saturation_threshold : float, optional
        See :func:`infiltration_to_extraction`. Default 7.0.

    Returns
    -------
    numpy.ndarray
        Bin-averaged Kreft-Zuber flux concentration ``C_F`` in the extracted water.

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
    spinup: str | float | None = "constant",
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
        Flow rate of water in the aquifer [m3/day].
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
        Effective molecular diffusivity D_m [m2/day], applied to all streamtubes. Must be
        non-negative.
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m], applied to all streamtubes. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    regularization_strength : float, optional
        Tikhonov regularization parameter (default 1e-10).
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid. See
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
