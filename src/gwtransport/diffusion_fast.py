"""
Fast closed-form 1D advection-dispersion transport (Kreft-Zuber flux concentration).

This module computes the same physics as :mod:`gwtransport.diffusion` -- the Kreft-Zuber
(1978) flux concentration ``C_F`` at the outlet of a bundle of 1D streamtubes -- but
evaluates the bin-averaged breakthrough in closed form instead of by Gauss-Legendre
quadrature.

For each streamtube (one aquifer pore volume) the resident concentration in moving-frame
cumulative-volume (V) coordinates is the Gaussian CDF
``C_R = 0.5 * erfc((L - xi) / (2 * sqrt(D_t)))``, with ``D_t = D_m * tau + alpha_L * xi``
the moving-frame dispersion product. Its bin-average over a cout bin has the closed-form
antiderivative ``I(x) = 0.5*x + 0.5*[x*erf(x/s) + (s/sqrt(pi))*exp(-(x/s)^2)]``,
``s = 2*sqrt(D_t)``. Evaluating ``I`` once per cout edge with ``D_t`` carried *per edge*
and differencing yields the flux concentration ``C_F`` directly -- not merely ``C_R`` --
because ``dD_t/dx = D_m/v_s + alpha_L = D_s/v_s`` is exactly the Kreft-Zuber flux coefficient
at the solute-front velocity ``v_s = Q*L/(R*V_pore)`` (using ``d(tau)/dx = 1/v_s`` with
``tau = R*V/(L*Q)``). The dispersive boundary-flux correction therefore emerges from the
``D_t`` variation across the bin; no explicit correction term is added.

The elapsed time ``tau`` and travel distance ``xi`` are read directly from the time and
cumulative-volume edges (``tau_ij = t_cout_i - t_cin_j``, ``xi`` geometric), so no per-cell
quadrature and no residence-time inversion is needed. The result reproduces
:mod:`gwtransport.diffusion` to machine precision when the cout grid aligns with the flow
grid (supply ``flow_out`` on the output grid). The coefficient matrix is built only on the
breakthrough band -- the cumulative-volume band where the bin-averaged ``C_F`` is unsaturated,
the only region with non-zero coefficients -- so the build cost scales with the band width
(a few percent of the matrix at realistic dispersion) rather than with the full grid.

Streamtube assumption (no cross-sectional area parameter)
---------------------------------------------------------

Each entry in ``aquifer_pore_volumes`` is an independent 1D streamtube; molecular diffusion
enters the V-space variance through ``D_m * tau`` and mechanical dispersion through
``alpha_L * xi``. ``streamline_length`` / ``molecular_diffusivity`` /
``longitudinal_dispersivity`` may be a scalar (shared by all streamtubes) or an array with
one value per pore volume, exactly as in :mod:`gwtransport.diffusion`.

When to choose this module vs :mod:`gwtransport.diffusion`
----------------------------------------------------------

Both modules implement the *same* physics (Bear resident concentration + Kreft-Zuber flux
concentration on 1D streamtubes, with retardation and the moving-frame variance
``D_t = D_m*tau + alpha_L*xi``), and both accept per-streamtube ``streamline_length`` /
``molecular_diffusivity`` / ``longitudinal_dispersivity`` arrays. Whenever the cout grid is
at or finer than the flow grid, this module reproduces :mod:`gwtransport.diffusion` to
machine precision for *every* parameter regime -- including ``retardation_factor != 1`` with
``molecular_diffusivity > 0``, where the antiderivative's slope ``dD_t/dx = D_s/v_s`` already
carries the solute-front Kreft-Zuber flux coefficient natively -- while being ~80-90x faster
even before banding (closed form, no Gauss-Legendre quadrature, no residence-time inversion),
and the banded build computes only the non-zero breakthrough band -- faster still at the
weak-to-moderate dispersion of realistic problems. So it is the right default. The only case that favours
:mod:`gwtransport.diffusion` is a cout grid *coarser* than the flow detail: this module
treats ``flow_out`` as constant within each cout bin, whereas :mod:`gwtransport.diffusion`
integrates the full ``tedges``-resolution flow within each cout bin -- a ~0.1%-of-peak
difference for a rapidly-varying ``cin`` over wide cout bins under variable flow.

Available functions:

- :func:`infiltration_to_extraction` -- forward transport.
- :func:`extraction_to_infiltration` -- inverse via Tikhonov regularization.
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
from gwtransport.residence_time import residence_time_series
from gwtransport.utils import cumulative_flow_volume

# Default saturation threshold U for the banded build: a cout/cin pair is only evaluated
# while the breakthrough |x|/(2*sqrt(D_t)) <= U; beyond it the bin-averaged C_F is saturated
# to 0 or 1. At U >= ~6 the dense kernel itself already rounds the dropped tail to exactly 0
# or 1 (the Gaussian term underflows below the ulp of x), so the banded matrix is bit-identical
# to the dense one. Smaller U narrows the band -> faster, at the cost of dropping breakthrough
# tails of order exp(-U^2).
_DEFAULT_SATURATION_THRESHOLD = 7.0


def _pv_band_geometry(
    *,
    cumulative_volume_at_cout: npt.NDArray[np.floating],
    cumulative_volume_at_cin: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    r_vpv: float,
    length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    min_cin_flow: float,
    saturation_threshold: float,
    n_cin_bins: int,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    r"""Per-cout-row band bounds (lo, hi) in cin-bin columns for one streamtube (geometry only).

    Locates the narrow cumulative-volume band where the breakthrough transitions between 0 and 1
    -- the only cin bins with a non-zero coefficient. Within a streamtube the moving-frame
    dispersion product is exactly linear in the breakthrough coordinate, ``D_t(x) = A + B*x``,
    with slope ``B = dD_t/dx = R*D_m/v + alpha_L`` and intercept ``A`` the front value, so the
    saturation edge ``|x| = saturation_threshold * 2*sqrt(D_t(x))`` is the root of a quadratic --
    the band half-width is closed form, no iteration. The band is centred per cout bin on the
    front ``V_cin = V_cout - R*V_pore`` and mapped to cin-edge columns with ``searchsorted`` (so
    non-uniform / variable-flow grids need no special handling).

    With a warm-start spin-up and ``D_m > 0`` the breakthrough is *also* unsaturated at the
    data-start edge (cin edge 0): a leading zero-flow plateau holds the cumulative volume flat
    there, so the front search lands the local band one or more columns inside the record while a
    genuine, non-negligible coefficient remains at column 0 (the warm-start tail of a wide bin with
    large ``tau`` -> large ``D_t``). Each row therefore additionally tests ``|x0| < U*2*sqrt(D_t0)``
    at edge 0 and, when non-saturated, drops its band lower bound to 0 so that tail is kept.

    Returns
    -------
    lo : ndarray of int, shape (n_cout_bins,)
        Per-row band lower bound (inclusive), clipped to ``[0, n_cin_bins - 1]``.
    hi : ndarray of int, shape (n_cout_bins,)
        Per-row band upper bound (inclusive), clipped to ``[0, n_cin_bins - 1]``.
    """
    v_cin = cumulative_volume_at_cin
    n_cin_edges = v_cin.size
    v_cout_lo, v_cout_hi = cumulative_volume_at_cout[:-1], cumulative_volume_at_cout[1:]
    t_cout_lo, t_cout_hi = cout_tedges_days[:-1], cout_tedges_days[1:]

    # Front locus per cout bin, then the band half-widths in closed form. In the breakthrough
    # coordinate x (x > 0 broken through, x < 0 not), the moving-frame dispersion product is
    # D_t(x) = D_m*tau(x) + alpha_L*xi(x), with intercept A := front D_t. Both bounds below are
    # conservative -- they never under-cover the unsaturated band, where |x| < U*2*sqrt(D_t):
    #   PRE side  (x < 0): tau and xi both shrink away from the front, so D_t <= a_pre := front D_t
    #     and the band reaches |x| = 2U*sqrt(a_pre). Flow-independent, no cancellation.
    #   POST side (x > 0): D_t grows; bounded by the steepest slope D_t <= A + B*x with
    #     B = alpha_L + D_m*r_vpv/(L*min_cin_flow) (dtau/dx = r_vpv/(L*flow) <= r_vpv/(L*min_cin_flow)),
    #     giving the root x_post = 2U^2*B + 2U*sqrt(U^2*B^2 + A). The flow term only matters for strong
    #     molecular diffusion (already the wide-band regime); the mechanical term is exact.
    # The post extent anchors at the lower cout edge (smallest V_cin), whose front can sit in slower
    # flow with larger tau, so it carries its own intercept a_post; max(a_pre, a_post) bounds the front
    # D_t of both cout edges there. The pre extent anchors at the upper cout edge. a_pre uses the upper
    # edge time / mid front (conservative for the pre side). +1 absorbs searchsorted rounding;
    # min_cin_flow == 0 (no flow) -> the post slope is unbounded; b_max is set to span the whole axis
    # only when D_m > 0, else 0 (no diffusion, no flow term), avoiding a 0/0 NaN.
    u = saturation_threshold
    v_front = 0.5 * (v_cout_lo + v_cout_hi) - r_vpv
    center = np.searchsorted(v_cin, v_front)
    center_post = np.searchsorted(v_cin, v_cout_lo - r_vpv)  # searchsorted >= 0, so only the high clip binds
    disp = longitudinal_dispersivity * length
    a_pre = molecular_diffusivity * np.maximum(t_cout_hi - tedges_days[np.minimum(center, n_cin_edges - 1)], 0.0) + disp
    a_post = (
        molecular_diffusivity * np.maximum(t_cout_lo - tedges_days[np.minimum(center_post, n_cin_edges - 1)], 0.0)
        + disp
    )
    a_post = np.maximum(a_post, a_pre)
    pre_x = 2.0 * u * np.sqrt(a_pre)
    if min_cin_flow > 0.0:
        b_max = longitudinal_dispersivity + molecular_diffusivity * r_vpv / (length * min_cin_flow)
    else:
        b_max = np.inf if molecular_diffusivity > 0.0 else longitudinal_dispersivity
    post_x = 2.0 * u * u * b_max + 2.0 * u * np.sqrt(u * u * b_max * b_max + a_post)
    col_post = np.searchsorted(v_cin, (v_cout_lo - r_vpv) - r_vpv * post_x / length)  # smallest V_cin
    col_pre = np.searchsorted(v_cin, (v_cout_hi - r_vpv) + r_vpv * pre_x / length)  # largest V_cin
    # Scalar (global-max) half-widths, applied to every row via the band centre. Taking the max over
    # all cout rows is conservative: a row whose front sits in a slow / zero-flow plateau (large tau,
    # wide local transition) sets the half-width for all rows, so an interior zero-flow gap straddled
    # by a misaligned cout bin is never under-covered. +1 absorbs the searchsorted rounding.
    hw_post = max(int(np.max(center - col_post)), 1) + 1
    hw_pre = max(int(np.max(col_pre - center)), 1) + 1
    lo = np.clip(center - hw_post, 0, n_cin_bins - 1)
    hi = np.clip(center + hw_pre - 1, 0, n_cin_bins - 1)

    # Warm-start data-start tail: drop the band to column 0 where the breakthrough is still
    # unsaturated at cin edge 0. x0 is the breakthrough coordinate of the lower cout edge measured
    # from V_cin[0] (the smallest x0 over the bin, hence the most-broken-through / largest |D_t0|);
    # D_t0 floors at _DT_FLOOR so the zero-dispersion limit gives x0 != 0 -> saturated -> no spurious
    # widening. The leading zero-flow plateau that triggers this keeps full_band bounded by the
    # plateau length, independent of the record length.
    x0 = (v_cout_lo - v_cin[0] - r_vpv) * length / r_vpv
    tau0 = np.maximum(t_cout_lo - tedges_days[0], 0.0)
    dt0 = np.maximum(molecular_diffusivity * tau0 + longitudinal_dispersivity * np.maximum(x0 + length, 0.0), _DT_FLOOR)
    non_saturated0 = np.abs(x0) < u * 2.0 * np.sqrt(dt0)
    lo[non_saturated0] = 0
    return lo, hi


def _pv_band_values(
    *,
    col_start: npt.NDArray[np.intp],
    full_band: int,
    cumulative_volume_at_cout: npt.NDArray[np.floating],
    cumulative_volume_at_cin: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    r_vpv: float,
    length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
) -> npt.NDArray[np.floating]:
    r"""Bin-averaged ``C_F`` stripe for one streamtube on the shared band (values pass).

    ``C_F`` over a cout bin is ``(I(x_hi) - I(x_lo)) / dx`` with the closed-form antiderivative
    ``I`` evaluated at both cout edges over the band cin edges. Because ``D_t = D_m*tau + alpha_L*xi``
    with ``tau = R*V/(L*Q)``, the antiderivative's slope ``dD_t/dx = R*D_m/v_fluid + alpha_L =
    D_s/v_s`` is exactly the Kreft-Zuber flux coefficient at the solute-front velocity
    ``v_s = Q*L/(R*V_pore)``, so the flux concentration emerges natively -- no correction term is
    added. The stripe is the band itself: each row spans cin edges
    ``col_start[k] .. col_start[k] + full_band`` (``full_band + 1`` edges), so the coefficient for
    band offset ``b`` (cin bin ``col_start[k] + b``) is ``frac[b] - frac[b + 1]``. The
    zero-dispersion limit is exact here too: ``D_t`` floors to ``_DT_FLOOR``, so ``C_F`` is a step
    smoothed by ~1e-15.

    Returns
    -------
    coeff : ndarray, shape (n_cout_bins, full_band)
        Per-row, per-band-offset coefficient ``frac[:, :-1] - frac[:, 1:]`` already aligned to the
        banded buffer (offset 0 -> cin bin ``col_start[k]``).
    """
    v_cin = cumulative_volume_at_cin
    n_cin_edges = v_cin.size
    v_cout_lo, v_cout_hi = cumulative_volume_at_cout[:-1], cumulative_volume_at_cout[1:]
    t_cout_lo, t_cout_hi = cout_tedges_days[:-1], cout_tedges_days[1:]

    # Bin-averaged C_F over the band: I evaluated at both cout edges on the full_band + 1 cin edges
    # of each row's band. Edges are clipped to the real data edges, which the warm-start extension
    # makes saturated, so out-of-range edges carry frac 0/1 and telescope away in the difference.
    cols = col_start[:, None] + np.arange(full_band + 1)[None, :]
    cc = np.clip(cols, 0, n_cin_edges - 1)
    v_c, t_c = v_cin[cc], tedges_days[cc]
    sw_lo = (v_cout_lo[:, None] - v_c - r_vpv) * length / r_vpv
    sw_hi = (v_cout_hi[:, None] - v_c - r_vpv) * length / r_vpv

    # No dispersion: C_R is the step H(x); its exact bin-average is the fraction of the cout bin with
    # x > 0, and at a zero-width (dv_cout = 0) cout bin it is the point value 0.5*(1 + sign(x_lo)).
    # This matches the dense kernel's zero-dispersion branch bit-for-bit (the floored erf form would
    # instead give 0 at dx = 0, dropping the step at a gap-straddling misaligned cout bin).
    if molecular_diffusivity == 0.0 and longitudinal_dispersivity == 0.0:
        dx = sw_hi - sw_lo
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (np.maximum(sw_hi, 0.0) - np.maximum(sw_lo, 0.0)) / dx
        frac = np.where(dx > 0.0, frac, 0.5 + 0.5 * np.sign(sw_lo))
        return frac[:, :-1] - frac[:, 1:]

    dt_lo = np.maximum(
        molecular_diffusivity * np.maximum(t_cout_lo[:, None] - t_c, 0.0)
        + longitudinal_dispersivity * np.maximum(sw_lo + length, 0.0),
        _DT_FLOOR,
    )
    dt_hi = np.maximum(
        molecular_diffusivity * np.maximum(t_cout_hi[:, None] - t_c, 0.0)
        + longitudinal_dispersivity * np.maximum(sw_hi + length, 0.0),
        _DT_FLOOR,
    )
    i_lo = _breakthrough_antideriv(sw_lo, dt_lo)
    i_hi = _breakthrough_antideriv(sw_hi, dt_hi)
    dx = sw_hi - sw_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(dx > 0.0, (i_hi - i_lo) / dx, 0.0)

    return frac[:, :-1] - frac[:, 1:]


def _closed_form_coeff_matrix(
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
    extend_tedges: bool,
    saturation_threshold: float = _DEFAULT_SATURATION_THRESHOLD,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Build the banded forward operator (``cout = W @ cin``) via the closed-form C_F.

    Mirrors :func:`gwtransport.diffusion._infiltration_to_extraction_coeff_matrix`
    (per-streamtube loop over pore volumes, 100-year warm-start extension, residence-time
    validity) but computes the bin-averaged flux concentration in closed form instead of
    16-point Gauss-Legendre quadrature, and stores it in BANDED layout: row ``k`` of the dense
    operator ``W`` is ``band_vals[k]`` placed at columns ``[col_start[k], col_start[k] + full_band)``.
    The build runs in two passes over the pore-volume loop -- a cheap geometry pass that sizes the
    per-row band union, then a values pass that scatters each streamtube's ``C_F`` stripe into the
    banded buffer. The result reproduces the slow module's ``C_F`` to machine precision when the
    cout grid aligns with the flow grid. ``streamline_length``, ``molecular_diffusivity``, and
    ``longitudinal_dispersivity`` are per-pore-volume arrays (length ``len(aquifer_pore_volumes)``).

    Returns
    -------
    band_vals : ndarray, shape (n_cout_bins, full_band)
        Banded forward weights, NaN replaced with zero.
    col_start : ndarray of int, shape (n_cout_bins,)
        First cin-bin column of each cout row's band.
    valid_cout_bins : ndarray of bool, shape (n_cout_bins,)
        Output bins with complete breakthrough information for every streamtube.
    """
    work_tedges = tedges
    if extend_tedges:
        # Extend by 100 years on each side so a constant warm-start fills the spin-up region.
        # Timestamp arithmetic keeps the input timezone (tz-naive stays naive, tz-aware UTC
        # stays tz-aware); going through ``.to_numpy()`` would strip/mix the tz.
        pad = pd.Timedelta(days=36500)
        work_tedges = tedges[:1] - pad
        work_tedges = work_tedges.append(tedges[1:-1]).append(tedges[-1:] + pad)

    tedges_days = tedges_to_days(work_tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=work_tedges[0])

    # Cumulative through-flow volume on a common axis. cout-edge volumes come from flow_out
    # when provided (the user-specified extraction-side flow), placed on the infiltration
    # volume axis by anchoring at the first cout edge inside the flow record (so an output
    # window that starts before the input data stays correctly aligned); otherwise
    # interpolated from the infiltration curve.
    cumulative_volume_at_cin = cumulative_flow_volume(flow, dt_to_days(work_tedges))
    cumulative_volume_at_cout = _cout_cumulative_volume(
        flow_out=flow_out,
        cout_tedges=cout_tedges,
        cout_tedges_days=cout_tedges_days,
        tedges_days=tedges_days,
        cumulative_volume_at_cin=cumulative_volume_at_cin,
    )

    # Residence time identifies cout bins with complete breakthrough (NaN beyond data).
    rt_at_cout_tedges = residence_time_series(
        flow=flow,
        tedges=work_tedges,
        index=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    valid_cout_bins = ~np.any(np.isnan(rt_at_cout_tedges[:, :-1]) | np.isnan(rt_at_cout_tedges[:, 1:]), axis=0)

    # Slowest cin-side flow rate, used to bound the broken-through band width (the slowest flow
    # gives the steepest dD_t/dx). Zero when flow is everywhere zero -> the band widens (capped
    # at n_cin_bins), and the resulting no-flow rows are masked invalid anyway.
    positive_flow = flow[flow > 0.0]
    min_cin_flow = float(positive_flow.min()) if positive_flow.size else 0.0

    n_cout_bins = len(cout_tedges) - 1
    n_cin_bins = len(flow)

    # PASS 1 (geometry): per-streamtube band bounds (lo, hi) in cin-bin columns; accumulate the
    # per-row union over streamtubes. union_lo / union_hi are the min / max bounds per cout row.
    union_lo = np.full(n_cout_bins, n_cin_bins - 1, dtype=np.intp)
    union_hi = np.zeros(n_cout_bins, dtype=np.intp)
    geometry = []
    for i_pv, v_pore in enumerate(aquifer_pore_volumes):
        r_vpv = retardation_factor * v_pore
        length = float(streamline_length[i_pv])
        d_m = float(molecular_diffusivity[i_pv])
        alpha_l = float(longitudinal_dispersivity[i_pv])
        lo, hi = _pv_band_geometry(
            cumulative_volume_at_cout=cumulative_volume_at_cout,
            cumulative_volume_at_cin=cumulative_volume_at_cin,
            cout_tedges_days=cout_tedges_days,
            tedges_days=tedges_days,
            r_vpv=r_vpv,
            length=length,
            molecular_diffusivity=d_m,
            longitudinal_dispersivity=alpha_l,
            min_cin_flow=min_cin_flow,
            saturation_threshold=saturation_threshold,
            n_cin_bins=n_cin_bins,
        )
        np.minimum(union_lo, lo, out=union_lo)
        np.maximum(union_hi, hi, out=union_hi)
        geometry.append((r_vpv, length, d_m, alpha_l))

    col_start = union_lo
    full_band = min(int(np.max(union_hi - union_lo)) + 1, n_cin_bins)
    band_vals = np.zeros((n_cout_bins, full_band))

    # PASS 2 (values): each streamtube's C_F stripe is built on the shared band (col_start,
    # full_band), so its coeff is already offset-aligned and accumulates directly into the buffer.
    for r_vpv, length, d_m, alpha_l in geometry:
        band_vals += _pv_band_values(
            col_start=col_start,
            full_band=full_band,
            cumulative_volume_at_cout=cumulative_volume_at_cout,
            cumulative_volume_at_cin=cumulative_volume_at_cin,
            cout_tedges_days=cout_tedges_days,
            tedges_days=tedges_days,
            r_vpv=r_vpv,
            length=length,
            molecular_diffusivity=d_m,
            longitudinal_dispersivity=alpha_l,
        )

    band_vals /= len(aquifer_pore_volumes)
    return np.nan_to_num(band_vals, nan=0.0), col_start, valid_cout_bins


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
    """Compute extracted concentration with advection and longitudinal dispersion.

    Fast closed-form counterpart of :func:`gwtransport.diffusion.infiltration_to_extraction`.
    Reports the Kreft-Zuber (1978) flux concentration ``C_F`` and reproduces the slow module
    to machine precision when the cout grid aligns with the flow grid (supply ``flow_out``).

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
        Aquifer pore volumes [m³] -- one independent streamtube per entry.
    streamline_length : float or ndarray
        Travel distance L [m]: a scalar (shared by all streamtubes) or an array with one
        value per aquifer pore volume. Must be positive.
    molecular_diffusivity : float or ndarray
        Effective molecular diffusivity D_m [m²/day]: scalar or one value per pore volume.
        Must be non-negative.
    longitudinal_dispersivity : float or ndarray
        Longitudinal dispersivity alpha_L [m]: scalar or one value per pore volume.
        Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    flow_out : array-like or None, optional
        Extraction flow rate [m³/day] on the output grid (aligned to ``cout_tedges``,
        length ``len(cout_tedges) - 1``); constant within each cout bin, like ``flow`` is
        within each ``tedges`` bin. It defines the cout-bin volumes and the outlet velocity.
        **Required when ``cout_tedges`` differs from ``tedges``**; may be omitted only when
        ``cout_tedges`` equals ``tedges`` (then it equals ``flow``). Default None.
    spinup : {"constant"} | None, optional
        ``"constant"`` (default) extends ``tedges`` by 100 years on each side so a constant
        warm-start fills the left-edge spin-up region; ``None`` leaves spin-up cout as NaN.
    saturation_threshold : float, optional
        Breakthrough-band cutoff ``U`` (default 7.0). The coefficient matrix is built only on the
        cumulative-volume band where the breakthrough is unsaturated (``|x| < U * 2*sqrt(D_t)``),
        which is the only region with non-zero coefficients. ``U`` around 7 (any value above ~6)
        reproduces the full dense build to machine precision; a smaller value narrows the band --
        faster -- at the cost of dropping breakthrough tails of order ``exp(-U**2)``.

    Returns
    -------
    numpy.ndarray
        Bin-averaged Kreft-Zuber flux concentration ``C_F`` in the extracted water. Length
        ``len(cout_tedges) - 1``. NaN where no infiltration data has broken through.

    See Also
    --------
    gwtransport.diffusion.infiltration_to_extraction : Quadrature reference; prefer for cout
        grids coarser than the flow detail.
    extraction_to_infiltration : Inverse operation.
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion.
    """
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    tedges = pd.DatetimeIndex(tedges)
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

    n_pore_volumes = len(aquifer_pore_volumes)
    streamline_length = _broadcast_to_pore_volumes(streamline_length, n_pore_volumes)
    molecular_diffusivity = _broadcast_to_pore_volumes(molecular_diffusivity, n_pore_volumes)
    longitudinal_dispersivity = _broadcast_to_pore_volumes(longitudinal_dispersivity, n_pore_volumes)

    band_vals, col_start, valid_cout_bins = _closed_form_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow_out=flow_out,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        extend_tedges=_extend_tedges_flag(spinup),
        saturation_threshold=saturation_threshold,
    )

    n_cin = len(cin)
    full_band = band_vals.shape[1]
    cols = np.clip(col_start[:, None] + np.arange(full_band), 0, n_cin - 1)
    cout = np.einsum("kb,kb->k", band_vals, cin[cols])

    # Mark output bins invalid where no input has broken through (spin-up) or the output
    # bin extends beyond the input data range.
    total_coeff = band_vals.sum(axis=1)
    cout[(total_coeff < _EPSILON_COEFF_SUM) | ~valid_cout_bins] = np.nan
    return cout


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
    """Reconstruct infiltration concentration from extracted water (deconvolution).

    Inverts the forward model by building the same closed-form flux-concentration matrix as
    :func:`infiltration_to_extraction` and solving ``W @ cin = cout`` via Tikhonov
    regularization. Fast closed-form counterpart of
    :func:`gwtransport.diffusion.extraction_to_infiltration`.

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
        Travel distance L [m]: a scalar (shared by all streamtubes) or an array with one
        value per aquifer pore volume. Must be positive.
    molecular_diffusivity : float or ndarray
        Effective molecular diffusivity D_m [m²/day]: scalar or one value per pore volume.
        Must be non-negative.
    longitudinal_dispersivity : float or ndarray
        Longitudinal dispersivity alpha_L [m]: scalar or one value per pore volume.
        Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    regularization_strength : float, optional
        Tikhonov regularization parameter (default 1e-10).
    flow_out : array-like or None, optional
        Extraction flow rate [m³/day] on the output grid (aligned to ``cout_tedges``).
        See :func:`infiltration_to_extraction`. Default None.
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
    infiltration_to_extraction : Forward operation.
    gwtransport.diffusion.extraction_to_infiltration : Quadrature reference.
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

    n_pore_volumes = len(aquifer_pore_volumes)
    streamline_length = _broadcast_to_pore_volumes(streamline_length, n_pore_volumes)
    molecular_diffusivity = _broadcast_to_pore_volumes(molecular_diffusivity, n_pore_volumes)
    longitudinal_dispersivity = _broadcast_to_pore_volumes(longitudinal_dispersivity, n_pore_volumes)

    n_cin = len(tedges) - 1
    band_vals, col_start, valid_cout_bins = _closed_form_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow_out=flow_out,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        extend_tedges=_extend_tedges_flag(spinup),
        saturation_threshold=saturation_threshold,
    )

    return _solve_reverse_banded(
        band_vals=band_vals,
        col_start=col_start,
        valid_cout_bins=valid_cout_bins,
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
    """Compute extracted concentration for a gamma-distributed pore volume distribution.

    Convenience wrapper around :func:`infiltration_to_extraction` that discretizes a
    (shifted) gamma aquifer pore-volume distribution into ``n_bins`` equal-probability
    streamtubes. Provide either (mean, std) or (alpha, beta); ``loc`` defaults to 0.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m³/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Length ``len(cin) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins.
    mean, std : float, optional
        Mean and standard deviation of the gamma pore-volume distribution [m³].
    loc : float, optional
        Location (minimum pore volume) [m³], ``0 <= loc < mean``. Default 0.0.
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
        Longitudinal dispersivity alpha_L [m], applied to all streamtubes. Must be
        non-negative.
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
        Length ``len(cout_tedges) - 1``.

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

    Convenience wrapper around :func:`extraction_to_infiltration` that discretizes a
    (shifted) gamma aquifer pore-volume distribution into ``n_bins`` equal-probability
    streamtubes. Provide either (mean, std) or (alpha, beta); ``loc`` defaults to 0.

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
        Mean and standard deviation of the gamma pore-volume distribution [m³].
    loc : float, optional
        Location (minimum pore volume) [m³], ``0 <= loc < mean``. Default 0.0.
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
        Longitudinal dispersivity alpha_L [m], applied to all streamtubes. Must be
        non-negative.
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
