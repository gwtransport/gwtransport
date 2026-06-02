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
because ``dD_t/dx = D_m/v + alpha_L = D_L/v`` is exactly the Kreft-Zuber flux coefficient
(using ``d(tau)/dx = 1/v``). The dispersive boundary-flux correction therefore emerges from
the ``D_t`` variation across the bin; no explicit correction term is added.

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
``molecular_diffusivity > 0``, whose flux correction is itself evaluated in closed form
(its Gaussian-density bin-average has an exact erf/erfcx form) -- while being ~80-90x faster
even before banding (closed form, no Gauss-Legendre quadrature, no residence-time inversion),
and the banded build computes only the non-zero breakthrough band -- faster still at the
weak-to-moderate dispersion of realistic problems. So it is the right default. The only case that favours
:mod:`gwtransport.diffusion` is a cout grid *coarser* than the flow detail: this module
treats ``flow_out`` as constant within each cout bin, whereas :mod:`gwtransport.diffusion`
integrates the full ``tedges``-resolution flow within each cout bin -- a ~0.1%-of-peak
difference for a rapidly-varying ``cin`` over wide cout bins under variable flow.

Available functions:

- :func:`infiltration_to_extraction` -- forward transport.
- :func:`extraction_to_infiltration` -- inverse via Tikhonov regularisation.
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
    _retardation_excess_density,
    _validate_inputs,
)
from gwtransport._time import dt_to_days, tedges_to_days
from gwtransport.residence_time import residence_time
from gwtransport.utils import cumulative_flow_volume, solve_inverse_transport

# Default saturation threshold U for the banded build: a cout/cin pair is only evaluated
# while the breakthrough |x|/(2*sqrt(D_t)) <= U; beyond it the bin-averaged C_F is saturated
# to 0 or 1. At U >= ~6 the dense kernel itself already rounds the dropped tail to exactly 0
# or 1 (the Gaussian term underflows below the ulp of x), so the banded matrix is bit-identical
# to the dense one. Smaller U narrows the band -> faster, at the cost of dropping breakthrough
# tails of order exp(-U^2).
_DEFAULT_SATURATION_THRESHOLD = 7.0


def _flux_breakthrough_fraction(
    *,
    step_widths: npt.NDArray[np.floating],
    tau: npt.NDArray[np.floating],
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    streamline_len: float,
    retardation_factor: float,
    velocity: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Bin-averaged Kreft-Zuber flux concentration over each cout bin, per cin edge.

    Closed-form evaluation of the bin-averaged outlet breakthrough. For a unit step
    injected at cin edge *j*, the bin-averaged resident concentration over cout bin *i* is

    .. math::

        \frac{1}{\Delta x}\int C_R\,dx,\qquad
        C_R = \tfrac12\bigl(1 + \operatorname{erf}(x/(2\sqrt{D_t}))\bigr),

    with antiderivative
    :math:`I(x) = \tfrac12 x + \tfrac12[x\,\operatorname{erf}(x/s) + (s/\sqrt\pi)e^{-(x/s)^2}]`,
    :math:`s = 2\sqrt{D_t}`. Evaluating :math:`I` once per cout *edge* and differencing,
    with the moving-frame dispersion product :math:`D_t = D_m\,\tau + \alpha_L\,\xi`
    carried *per edge*, yields the **flux** concentration :math:`C_F` (Kreft & Zuber 1978),
    not merely :math:`C_R`. This is exact, not coincidental: the antiderivative difference
    picks up a :math:`\partial I/\partial D_t \cdot (dD_t/dx)` term, and

    .. math::

        \frac{dD_t}{dx} = D_m\frac{d\tau}{dx} + \alpha_L
                        = \frac{D_m}{v} + \alpha_L = \frac{D_L}{v},

    which is precisely the Kreft-Zuber flux coefficient (using :math:`d\tau/dx = 1/v`). The
    dispersive boundary-flux correction therefore emerges from the ``D_t`` variation across
    the bin; no explicit flux term is added.

    Parameters
    ----------
    step_widths : ndarray, shape (n_cout_edges, n_cin_edges)
        ``x = (V_cout - V_cin - R*V_pore) * L / (R*V_pore)`` at each (cout-edge, cin-edge);
        equals :math:`\xi - L`.
    tau : ndarray, shape (n_cout_edges, n_cin_edges)
        Elapsed time ``t_cout - t_cin`` [day] at each (cout-edge, cin-edge), clipped at 0.
    molecular_diffusivity : float
        Effective molecular diffusivity D_m [m^2/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].
    streamline_len : float
        Streamline length L [m].
    retardation_factor : float
        Retardation factor R [-]. Triggers the K-Z flux-coefficient correction when
        ``R != 1`` and ``D_m > 0`` (see notes in the body).
    velocity : ndarray, shape (n_cout_bins,)
        Fluid (unretarded) velocity ``Q*L/V_pore`` [m/day] in each cout bin, used by the
        retardation correction.

    Returns
    -------
    ndarray, shape (n_cout_bins, n_cin_edges)
        Bin-averaged flux concentration for the step injected at each cin edge.

    References
    ----------
    Kreft, A., & Zuber, A. (1978). On the physical meaning of the dispersion equation and
    its solutions for different initial and boundary conditions. Chemical Engineering
    Science, 33(11), 1471-1480.
    """
    x_lo = step_widths[:-1]
    x_hi = step_widths[1:]
    dx = x_hi - x_lo

    # No dispersion: C_R is the step function H(x); its exact bin-average is the fraction
    # of the cout bin with x > 0 (the limit of the erf form as D_t -> 0).
    if molecular_diffusivity == 0.0 and longitudinal_dispersivity == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (np.maximum(x_hi, 0.0) - np.maximum(x_lo, 0.0)) / dx
        return np.where(dx > 0.0, frac, 0.5 + 0.5 * np.sign(x_lo))

    xi = step_widths + streamline_len
    dt_var = np.maximum(molecular_diffusivity * tau + longitudinal_dispersivity * np.maximum(xi, 0.0), _DT_FLOOR)
    antideriv, s, gaussian = _breakthrough_antideriv(step_widths, dt_var)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(dx > 0.0, np.diff(antideriv, axis=0) / dx, 0.0)

    if retardation_factor != 1.0 and molecular_diffusivity > 0.0:
        density_binavg = _retardation_excess_density(
            x_lo=x_lo,
            x_hi=x_hi,
            dx=dx,
            d_lo=dt_var[:-1],
            d_hi=dt_var[1:],
            s_lo=s[:-1],
            s_hi=s[1:],
            g_lo=gaussian[:-1],
            g_hi=gaussian[1:],
        )
        excess = np.where(velocity > 0.0, (retardation_factor - 1.0) * molecular_diffusivity / velocity, 0.0)
        frac -= excess[:, None] * density_binavg
    return frac


def _accumulate_pv_banded(
    *,
    accumulated_coeff: npt.NDArray[np.floating],
    cumulative_volume_at_cout: npt.NDArray[np.floating],
    cumulative_volume_at_cin: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    r_vpv: float,
    length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float,
    velocity: npt.NDArray[np.floating],
    min_cin_flow: float,
    saturation_threshold: float,
) -> bool:
    r"""Scatter one streamtube's banded ``C_F`` contribution into ``accumulated_coeff``.

    Computes the bin-averaged flux concentration only on the narrow cumulative-volume band where
    the breakthrough transitions between 0 and 1 -- the only cin bins with a non-zero coefficient.
    Within a streamtube the moving-frame dispersion product is exactly linear in the breakthrough
    coordinate, ``D_t(x) = A + B*x``, with slope ``B = dD_t/dx = R*D_m/v + alpha_L`` (the per-bin
    flux coefficient) and intercept ``A`` the front value, so the saturation edge
    ``|x| = saturation_threshold * 2*sqrt(D_t(x))`` is the root of a quadratic -- the band
    half-width is closed form, no iteration. The band is centred per cout bin on the front
    ``V_cin = V_cout - R*V_pore`` and mapped to cin-edge columns with ``searchsorted`` (so
    non-uniform / variable-flow grids need no special handling).

    At ``saturation_threshold`` around 7 the dropped tail is ``~exp(-threshold**2)``, far below the
    ulp of the kept entries, so the scattered band reproduces the dense build
    (:func:`_flux_breakthrough_fraction`) to machine precision; a smaller threshold narrows the
    band (faster) at the cost of dropping breakthrough tails of that order. ``C_F`` over a cout bin
    is ``(I(x_hi) - I(x_lo)) / dx`` with the antiderivative ``I`` evaluated at both cout edges over
    the gathered cin stripe, plus the same closed-form retardation correction as the dense kernel.

    Returns
    -------
    bool
        ``True`` if the contribution was scattered. ``False`` (leaving ``accumulated_coeff``
        unchanged) when the band would span more than half the cin axis -- the caller then uses
        the dense kernel for this streamtube, so the cost never exceeds the dense build.
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
    # min_cin_flow == 0 -> B = inf -> the band trips the fallback below.
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
    with np.errstate(divide="ignore", invalid="ignore"):
        b_max = longitudinal_dispersivity + molecular_diffusivity * r_vpv / (length * min_cin_flow)
        post_x = 2.0 * u * u * b_max + 2.0 * u * np.sqrt(u * u * b_max * b_max + a_post)
        col_post = np.searchsorted(v_cin, (v_cout_lo - r_vpv) - r_vpv * post_x / length)  # smallest V_cin
        col_pre = np.searchsorted(v_cin, (v_cout_hi - r_vpv) + r_vpv * pre_x / length)  # largest V_cin
    hw_post = max(int(np.max(center - col_post)), 1) + 1
    hw_pre = max(int(np.max(col_pre - center)), 1) + 1
    if hw_post + hw_pre >= n_cin_edges // 2:
        return False

    # Bin-averaged C_F over the stripe: I evaluated at both cout edges. Columns are clipped to the
    # real data edges, which the warm-start extension makes saturated, so out-of-range columns
    # carry frac 0/1 and telescope away in the cin-edge difference below.
    cols = center[:, None] + np.arange(-hw_post, hw_pre + 1)[None, :]
    cc = np.clip(cols, 0, n_cin_edges - 1)
    v_c, t_c = v_cin[cc], tedges_days[cc]
    sw_lo = (v_cout_lo[:, None] - v_c - r_vpv) * length / r_vpv
    sw_hi = (v_cout_hi[:, None] - v_c - r_vpv) * length / r_vpv
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
    i_lo, s_lo, g_lo = _breakthrough_antideriv(sw_lo, dt_lo)
    i_hi, s_hi, g_hi = _breakthrough_antideriv(sw_hi, dt_hi)
    dx = sw_hi - sw_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(dx > 0.0, (i_hi - i_lo) / dx, 0.0)
    if retardation_factor != 1.0 and molecular_diffusivity > 0.0:
        density_binavg = _retardation_excess_density(
            x_lo=sw_lo, x_hi=sw_hi, dx=dx, d_lo=dt_lo, d_hi=dt_hi, s_lo=s_lo, s_hi=s_hi, g_lo=g_lo, g_hi=g_hi
        )
        excess = np.where(velocity > 0.0, (retardation_factor - 1.0) * molecular_diffusivity / velocity, 0.0)
        frac -= excess[:, None] * density_binavg

    coeff = frac[:, :-1] - frac[:, 1:]
    cin_bin = cols[:, :-1]
    rows = np.broadcast_to(np.arange(center.size)[:, None], coeff.shape)
    valid = (cin_bin >= 0) & (cin_bin < n_cin_edges - 1)
    np.add.at(accumulated_coeff, (rows[valid], cin_bin[valid]), coeff[valid])
    return True


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
    force_dense: bool = False,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Build the forward coefficient matrix W (``cout = W @ cin``) via the closed-form C_F.

    Mirrors :func:`gwtransport.diffusion._infiltration_to_extraction_coeff_matrix`
    (per-streamtube loop over pore volumes, 100-year warm-start extension, residence-time
    validity) but computes the bin-averaged flux concentration in closed form
    (:func:`_flux_breakthrough_fraction`) instead of 16-point Gauss-Legendre quadrature. The
    result reproduces the slow module's C_F to machine precision when the cout grid aligns
    with the flow grid. ``streamline_length``, ``molecular_diffusivity``, and
    ``longitudinal_dispersivity`` are per-pore-volume arrays (length ``len(aquifer_pore_volumes)``).

    Returns
    -------
    coeff_matrix : ndarray, shape (n_cout_bins, n_cin_bins)
        Forward coefficient matrix, NaN replaced with zero.
    valid_cout_bins : ndarray of bool, shape (n_cout_bins,)
        Output bins with complete breakthrough information for every streamtube.
    """
    work_tedges = tedges
    if extend_tedges:
        work_tedges = pd.DatetimeIndex([
            tedges[0] - pd.Timedelta("36500D"),
            *list(tedges[1:-1]),
            tedges[-1] + pd.Timedelta("36500D"),
        ])

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
    rt_at_cout_tedges = residence_time(
        flow=flow,
        flow_tedges=work_tedges,
        index=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    valid_cout_bins = ~np.any(np.isnan(rt_at_cout_tedges[:, :-1]) | np.isnan(rt_at_cout_tedges[:, 1:]), axis=0)

    # Elapsed time tau[i, j] = t_cout_i - t_cin_j (the moving-frame age). Geometric xi and
    # tau are all the closed form needs -- no per-cell residence-time inversion.
    tau = np.maximum(cout_tedges_days[:, None] - tedges_days[None, :], 0.0)

    # Fluid (unretarded) velocity in each cout bin: q_cout * L / V_pore (V_pore folded in
    # per streamtube below). q_cout = dV_cout / dt_cout.
    dv_cout = np.diff(cumulative_volume_at_cout)
    dt_cout = np.diff(cout_tedges_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_cout = np.where(dt_cout > 0, dv_cout / dt_cout, 0.0)

    # Slowest cin-side flow rate, used by the banded build to bound the broken-through band width
    # (the slowest flow gives the steepest dD_t/dx). Zero when flow is everywhere zero -> banding
    # falls back to the dense kernel.
    with np.errstate(divide="ignore", invalid="ignore"):
        cin_flow = np.diff(cumulative_volume_at_cin) / np.diff(tedges_days)
    positive_cin_flow = cin_flow[cin_flow > 0.0]
    min_cin_flow = float(positive_cin_flow.min()) if positive_cin_flow.size else 0.0

    n_cout_bins = len(cout_tedges) - 1
    n_cin_bins = len(flow)
    accumulated_coeff = np.zeros((n_cout_bins, n_cin_bins))
    for i_pv, v_pore in enumerate(aquifer_pore_volumes):
        r_vpv = retardation_factor * v_pore
        length = float(streamline_length[i_pv])
        d_m = float(molecular_diffusivity[i_pv])
        alpha_l = float(longitudinal_dispersivity[i_pv])
        velocity = q_cout * length / v_pore
        # Banded build for the dispersive case. The zero-dispersion step function (no band to
        # find) and a band wider than half the matrix fall back to the dense kernel, so the cost
        # never exceeds -- and is bit-identical to -- the dense build.
        if (
            not force_dense
            and (d_m > 0.0 or alpha_l > 0.0)
            and _accumulate_pv_banded(
                accumulated_coeff=accumulated_coeff,
                cumulative_volume_at_cout=cumulative_volume_at_cout,
                cumulative_volume_at_cin=cumulative_volume_at_cin,
                cout_tedges_days=cout_tedges_days,
                tedges_days=tedges_days,
                r_vpv=r_vpv,
                length=length,
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
                retardation_factor=retardation_factor,
                velocity=velocity,
                min_cin_flow=min_cin_flow,
                saturation_threshold=saturation_threshold,
            )
        ):
            continue
        step_widths = (cumulative_volume_at_cout[:, None] - cumulative_volume_at_cin[None, :] - r_vpv) * length / r_vpv
        frac = _flux_breakthrough_fraction(
            step_widths=step_widths,
            tau=tau,
            molecular_diffusivity=d_m,
            longitudinal_dispersivity=alpha_l,
            streamline_len=length,
            retardation_factor=retardation_factor,
            velocity=velocity,
        )
        accumulated_coeff += frac[:, :-1] - frac[:, 1:]

    coeff_matrix = accumulated_coeff / len(aquifer_pore_volumes)
    return np.nan_to_num(coeff_matrix, nan=0.0), valid_cout_bins


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
    """Compute extracted concentration with advection and longitudinal dispersion.

    Fast closed-form counterpart of :func:`gwtransport.diffusion.infiltration_to_extraction`.
    Reports the Kreft-Zuber (1978) flux concentration ``C_F`` and reproduces the slow module
    to machine precision when the cout grid aligns with the flow grid (supply ``flow_out``).

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
        Aquifer pore volumes [m3] -- one independent streamtube per entry.
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

    coeff_matrix, valid_cout_bins = _closed_form_coeff_matrix(
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

    cout = coeff_matrix @ cin

    # Mark output bins invalid where no input has broken through (spin-up) or the output
    # bin extends beyond the input data range.
    total_coeff = np.sum(coeff_matrix, axis=1)
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
    spinup: str | float | None = "constant",
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
        Flow rate of water in the aquifer [m3/day]. Length ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Length ``len(flow) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Length ``len(cout) + 1``.
    aquifer_pore_volumes : array-like
        Aquifer pore volumes [m3] -- one independent streamtube per entry.
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
        Retardation factor (default 1.0).
    regularization_strength : float, optional
        Tikhonov regularization parameter (default 1e-10).
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid (aligned to ``cout_tedges``).
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

    Warns
    -----
    UserWarning
        When the forward matrix is rank-deficient (constant flow with residence time an
        integer multiple of the time step). Adjust ``aquifer_pore_volumes`` slightly
        (e.g. multiply by 1.001) to fix.

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
    w_forward, valid_cout_bins = _closed_form_coeff_matrix(
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

    return solve_inverse_transport(
        w_forward=w_forward,
        observed=cout,
        n_output=n_cin,
        regularization_strength=regularization_strength,
        valid_rows=valid_cout_bins,
        warn_rank_deficient=True,
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
    """Compute extracted concentration for a gamma-distributed pore volume distribution.

    Convenience wrapper around :func:`infiltration_to_extraction` that discretizes a
    (shifted) gamma aquifer pore-volume distribution into ``n_bins`` equal-probability
    streamtubes. Provide either (mean, std) or (alpha, beta); ``loc`` defaults to 0.

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
        Longitudinal dispersivity alpha_L [m], applied to all streamtubes. Must be
        non-negative.
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

    Convenience wrapper around :func:`extraction_to_infiltration` that discretizes a
    (shifted) gamma aquifer pore-volume distribution into ``n_bins`` equal-probability
    streamtubes. Provide either (mean, std) or (alpha, beta); ``loc`` defaults to 0.

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
        Longitudinal dispersivity alpha_L [m], applied to all streamtubes. Must be
        non-negative.
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
