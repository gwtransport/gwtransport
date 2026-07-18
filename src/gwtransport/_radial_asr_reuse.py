r"""Reused-propagator-matrix acceleration of the multi-cycle radial advection-dispersion engine.

For a signed-flow schedule with ``K`` flow reversals, the per-reversal grid-free composition performs one
de Hoog inversion of the interior resolvent per intermediate reversal to hand the resident field across
the reversal -- the ``O(n_quad^2)`` per-reversal
cost. That hand-off is a *linear* operator on the resident field: the propagated field is
``f_out(r_i) = L^{-1}_s[ sum_j Ghat(r_i, r_j; s) w(r_j) f(r_j) ](tau)``, i.e. ``f_out = P @ f`` with the
propagator matrix

``P_{dir,tau}[i, j] = L^{-1}_s[ Ghat(r_i, r_j; s) w_dir(r_j) ](tau)``.

For a schedule with repeated phase volumes (periodic ASR / SWIW) the same ``(direction, tau)`` recurs every
cycle, so ``P`` is identical across cycles. This module builds each distinct ``P`` *once* (one batched de
Hoog inversion over all ``n_quad^2`` source/output entries, reusing the exact Airy interior Green's
function) and reuses it as a bounded matrix-multiply at every reversal -- the special-function + de Hoog
cost becomes ``O(number of distinct phase volumes)`` instead of ``O(K)``.

The matrix is assembled on the Bromwich contour (``Re s > 0``), where the Airy exponent is growing-real
and tames the divergent injection gauge ``e^{+r/alpha_L}`` (carried in log space by
:func:`gwtransport._radial_asr_kernels.assemble_airy_resolvent`), so ``P`` is well-conditioned at any
Peclet (``|P| ~ O(1)``, the same bounded physical operator the per-reversal engine applies). ``P @ f``
equals the per-reversal engine's ``_propagate(f)`` to the de Hoog floor; the only difference is the de Hoog
Pade acceleration's mild nonlinearity (``invert-then-sum`` vs ``sum-then-invert``), which vanishes as the
de Hoog ``n_terms``/``tol`` tighten.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import ive, kve

from gwtransport._radial_asr_compose import _DEHOOG_TERMS, _SCALE_MARGIN, _fr_step_response
from gwtransport._radial_asr_dehoog import dehoog_inverse
from gwtransport._radial_asr_kernels import (
    _integrate_logderiv,
    _resolvent_airy_pieces,
    assemble_airy_resolvent,
)

_INJECTION = "injection"


def _phase_slices(flow: npt.NDArray[np.floating]) -> list[tuple[int, slice]]:
    """Group the schedule into maximal one-signed phases.

    Returns
    -------
    list of (sign, slice)
        ``sign`` in ``{+1, -1, 0}`` (injection / extraction / rest) and the contiguous bin slice.
    """
    signs = np.sign(flow).astype(int)
    edges = np.flatnonzero(np.diff(signs) != 0) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(flow)]))
    return [(int(signs[a]), slice(a, b)) for a, b in zip(starts, stops, strict=True)]


def _field_grid(
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geo: float,
    r_w: float,
    alpha_l: float,
    molecular_diffusivity: float,
    n_quad: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Radial Gauss-Legendre quadrature grid spanning the plume front plus its dispersive tail.

    The grid is kept **tight**: it covers the advective front radius ``r_front`` (where the plume
    volume ``V(r) = peak net injected volume``) plus a margin of breakthrough widths (radial std
    ``~ sqrt(alpha_L r_front)``) and a molecular reach, and no further. An over-extended grid would
    push the Peclet so high that the divergent (injection) interior Green's function, which grows as
    ``e^{+r/alpha_L}`` before its ``e^{-r/alpha_L}`` Sturm-Liouville weight tames the product, overflows
    double precision; the resident profile is ``~0`` beyond the tail anyway (verified by mass
    conservation). Retardation rescales the clock, not the radius, so it does not enter ``r_max``.

    Returns
    -------
    r_nodes : ndarray
        Radial nodes (m), shape ``(n_quad,)``.
    v_nodes : ndarray
        Volume coordinate ``V(r) = c_geo (r^2 - r_w^2)`` at the nodes.
    dr_weights : ndarray
        Gauss-Legendre weights in ``r`` (so ``int g dr ~ sum g(r_k) dr_weights_k``).
    """
    net_volume = np.concatenate(([0.0], np.cumsum(flow * dt_days)))
    peak_volume = max(float(net_volume.max()), 0.0)
    r_front = np.sqrt(r_w**2 + peak_volume / c_geo)  # advective plume-front radius
    total_time = float(np.sum(dt_days))
    reach = 12.0 * np.sqrt(alpha_l * r_front + alpha_l**2) + 6.0 * np.sqrt(molecular_diffusivity * total_time)
    r_max = r_front + reach + r_w
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    r_nodes = 0.5 * (r_max - r_w) * (nodes + 1.0) + r_w
    dr_weights = 0.5 * (r_max - r_w) * weights
    v_nodes = c_geo * (r_nodes**2 - r_w**2)
    return r_nodes, v_nodes, dr_weights


def _airy_propagator_matrix(
    direction: str,
    tau: float,
    r_nodes: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    *,
    r_w: float,
    alpha_l: float,
    c_geo: float,
    n_terms: int,
    tol: float,
) -> npt.NDArray[np.floating]:
    r"""Field-propagator matrix ``P[i, j]`` for one constant-Q phase (``D_m = 0`` Airy branch).

    ``f_out = P @ f`` with ``P[i, j] = L^{-1}_s[ Ghat(r_i, r_j; s) w(r_j) ](tau)`` -- the same Airy interior
    resolvent and flushed-volume clock as the per-reversal ``_propagate``, assembled for
    every output/source pair and inverted in a single batched de Hoog pass instead of contracting against a
    specific field. The Airy pieces are evaluated once on the grid per de Hoog node (as in ``_propagate``)
    and every output row is assembled by prefix selection (``O(n_quad)`` Airy evaluations).

    The Sturm-Liouville source weight is ``w(r') = (2 c_geo r'/alpha_L) e^{-gauge_sign r'/alpha_L} dr'``
    (``gauge_sign = +1`` injection / ``-1`` extraction); the ``D_m = 0`` Airy kernel depends on the Laplace
    node only through ``beta = 2 c_geo p/alpha_L`` (the autonomy of the flushed-volume S-clock), so it is
    evaluated directly in that flow-free canonical form (``s = 2 c_geo p``, ``a0 = 1``) -- no ``flow_scale``
    round-trip, so ``P`` is exactly independent of the phase flow magnitude.

    Returns
    -------
    ndarray
        Propagator matrix ``P``, shape ``(n_quad, n_quad)``.
    """
    a0 = 1.0
    gauge_sign = 1.0 if direction == _INJECTION else -1.0
    n = r_nodes.size
    # Split the Sturm-Liouville source weight (2 c_geo r'/alpha_L) e^{-gauge_sign r'/alpha_L} dr' into its
    # non-exponential prefactor and its LOG exponent: the exponent is folded into assemble_airy_resolvent
    # so the divergent gauge cancels before np.exp (overflow-safe at any Peclet). Applying the exponential
    # weight afterwards would overflow -- the injection assemble exponent and the extraction weight each blow
    # up past r/alpha_L ~ 700 and meet as Inf * 0 = NaN.
    sl_prefactor = (2.0 * c_geo * r_nodes / alpha_l) * dr_weights
    sl_log = (-gauge_sign * r_nodes / alpha_l)[None, :]
    mu = c_geo * ((float(r_nodes.max()) + alpha_l) ** 2 + alpha_l**2 - r_w**2)
    scaling = _SCALE_MARGIN * max(mu, tau)
    below = r_nodes[None, :] < r_nodes[:, None]  # below[i, j] = r_j < r_i

    def f_hat(p: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        s = (2.0 * c_geo * p).reshape(-1, 1)  # canonical flow-free S-clock: beta = 2 c_geo p / alpha_L
        grid_p = _resolvent_airy_pieces(s, r_nodes.reshape(1, -1), alpha_l, a0, gauge_sign)
        piece_w = _resolvent_airy_pieces(s, np.array([[r_w]]), alpha_l, a0, gauge_sign)
        ghat = np.empty((p.size, n, n), dtype=complex)
        for i in range(n):
            # r_< / r_> selection at output i (the SL kernel is symmetric; radii enter only via r_i + r_j).
            mask = below[i][None, :]
            piece_a = {f: np.where(mask, grid_p[f], grid_p[f][:, i : i + 1]) for f in grid_p}
            piece_b = {f: np.where(mask, grid_p[f][:, i : i + 1], grid_p[f]) for f in grid_p}
            r_sum = (r_nodes[i] + r_nodes)[None, :]
            ghat[:, i, :] = assemble_airy_resolvent(
                piece_a, piece_b, piece_w, r_sum, alpha_l, gauge_sign, source_log_weight=sl_log
            )
        ghat *= sl_prefactor[None, None, :]  # non-exponential SL prefactor (the gauge is already folded in)
        return ghat

    return dehoog_inverse(f_hat=f_hat, t=tau, n_terms=n_terms, scaling=scaling, tol=tol)


def _riccati_propagator_matrix(
    direction: str,
    tau: float,
    r_nodes: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    *,
    r_w: float,
    alpha_l: float,
    c_geo: float,
    flow_scale: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    n_terms: int,
    tol: float,
) -> npt.NDArray[np.floating]:
    r"""Field-propagator matrix ``P[i, j]`` for one constant-Q phase (``D_m > 0`` Riccati branch).

    The matrix form of the per-reversal ``_propagate_diffusive``: instead of contracting
    the interior resolvent against a specific field by the prefix/suffix cumsums of
    :func:`gwtransport._radial_asr_kernels.resolvent_riccati`, the full Sturm-Liouville Green's function is
    assembled as the triangular outer product of the field-independent log-derivative integrals,

    ``Ghat(r_i, r_j) w_j = -[exp(J_-(r_i) + J_+(r_j) + LG_j) if j <= i else exp(J_+(r_i) + J_-(r_j) + LG_j)]
    / (L_-(r_w) - L_+(r_w)) * r_j dr_j``,

    with ``J_-`` / ``J_+`` the inward (decaying) and outward (well-regular) running integrals of the
    log-derivative (:func:`_integrate_logderiv`) and ``LG_j = b ln(G_j/G_w) - ln G_j`` the divergent gauge
    carried in log space (``b = 1 - sigma_A A_0/D_m``). Wall-clock time ``tau`` (the volume clock is not
    autonomous when ``D_m > 0``); retardation rescales ``A_0 -> A_0/R``, ``D_m -> D_m/R``.

    Returns
    -------
    ndarray
        Propagator matrix ``P``, shape ``(n_quad, n_quad)``.
    """
    a0_eff = (flow_scale / (2.0 * c_geo)) / retardation_factor
    d_m_eff = molecular_diffusivity / retardation_factor
    sigma_a = 1 if direction == _INJECTION else -1
    n = r_nodes.size
    mu_t = c_geo * ((float(r_nodes.max()) + alpha_l) ** 2 + alpha_l**2 - r_w**2) / flow_scale
    scaling = _SCALE_MARGIN * max(mu_t, tau)
    b = 1.0 - sigma_a * a0_eff / d_m_eff
    rad = np.concatenate(([r_w], r_nodes))
    g_nodes = alpha_l * a0_eff + d_m_eff * r_nodes
    g_w = alpha_l * a0_eff + d_m_eff * r_w
    lg = b * np.log(g_nodes / g_w) - np.log(g_nodes)  # bounded divergent gauge in log space
    lp_w = a0_eff / (alpha_l * a0_eff + d_m_eff * r_w) if sigma_a > 0 else 0.0
    lower = np.tril(np.ones((n, n)))[None]  # lower[i, j] = (j <= i)
    weight = (r_nodes * dr_weights)[None, None, :]

    def f_hat(s: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        ld_m, jj_m = _integrate_logderiv(s, rad, r_w, alpha_l, a0_eff, d_m_eff, sigma_a, "decaying")
        lm_w = ld_m[:, 0]  # L_-(r_w)
        jm = jj_m[:, 1:]  # int_{r_w}^{r_i} L_-  (n_s, n)
        _, jp = _integrate_logderiv(s, r_nodes, r_w, alpha_l, a0_eff, d_m_eff, sigma_a, "regular")  # int L_+
        denom = (lm_w - lp_w)[:, None, None]
        em_i = np.exp(jm)[:, :, None]
        ep_jl = np.exp(jp + lg[None, :])[:, None, :]
        ep_i = np.exp(jp)[:, :, None]
        em_jl = np.exp(jm + lg[None, :])[:, None, :]
        return -np.where(lower > 0, em_i * ep_jl, ep_i * em_jl) / denom * weight

    return dehoog_inverse(f_hat=f_hat, t=tau, n_terms=n_terms, scaling=scaling, tol=tol)


def _rest_propagator_matrix(
    tau: float,
    r_nodes: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    *,
    r_w: float,
    d_m_eff: float,
    n_terms: int,
    tol: float,
) -> npt.NDArray[np.floating]:
    r"""Field-propagator matrix ``P[i, j]`` for a rest (``Q = 0``) phase -- pure molecular diffusion.

    The matrix form of the per-reversal ``_propagate_rest``: the order-0 modified Bessel
    interior resolvent (:func:`gwtransport._radial_asr_kernels.rest_resolvent`) with the Sturm-Liouville rest
    measure ``w(r') dr' = (r'/D_m) dr'``, on the wall-clock clock. Retardation enters as ``d_m_eff = D_m/R``.

    **Resolution limit.** Molecular diffusion over the rest spreads mass by the diffusion length
    ``sqrt(D_m tau)``. Once that length drops below half the coarsest node gap the Gauss-Legendre grid can no
    longer resolve the near-delta diffusive Green's function: the quadratured resolvent stops conserving mass
    (``dv^T P`` amplifies above ``1`` -- a maximum-principle violation, ``cout > cin``). In that under-resolved
    regime the physical propagator *is* the identity to grid accuracy (diffusion has not crossed a cell), so
    the identity is returned -- mass-exact and equal to the ``D_m = 0`` echo, the correct small-``D_m`` limit.

    Returns
    -------
    ndarray
        Propagator matrix ``P``, shape ``(n_quad, n_quad)``.
    """
    if np.sqrt(d_m_eff * tau) < 0.5 * float(np.diff(r_nodes).max()):
        return np.eye(r_nodes.size)
    scaling = _SCALE_MARGIN * tau
    weight = (r_nodes / d_m_eff * dr_weights)[None, None, :]
    # r_nodes ascend, so r_< = r_nodes[min(i, j)], r_> = r_nodes[max(i, j)]: the order-0 scaled Bessels are
    # evaluated once on the grid per s-node and gathered by these index maps instead of O(n_quad) times.
    idx = np.arange(r_nodes.size)
    lt = np.minimum(idx[:, None], idx[None, :])
    gt = np.maximum(idx[:, None], idx[None, :])

    def f_hat(s: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        kappa = np.sqrt(np.asarray(s, dtype=complex).reshape(-1, 1) / d_m_eff)  # (n_s, 1), Re(kappa) > 0
        zr = kappa * r_nodes[None, :]  # (n_s, n): kappa r' on the grid, Bessels evaluated once here
        iv0, kv0 = ive(0, zr), kve(0, zr)
        z_w = kappa * r_w
        ratio_w = ive(1, z_w) / kve(1, z_w)  # (n_s, 1)
        z_lt, z_gt = zr[:, lt], zr[:, gt]  # (n_s, n, n)
        # Ghat = [I_0(z_<) + (I_1(z_w)/K_1(z_w)) K_0(z_<)] K_0(z_>) with scaled Bessels (see rest_resolvent):
        # the scaling exponents difference to <= 0 so the growing I_0 never overflows at high kappa r.
        ghat = iv0[:, lt] * kv0[:, gt] * np.exp(np.abs(z_lt.real) - z_gt)
        outer = ratio_w[:, :, None] * kv0[:, lt]
        outer *= kv0[:, gt]
        outer *= np.exp((np.abs(z_w.real) + z_w)[:, :, None] - z_lt - z_gt)
        ghat += outer
        ghat *= weight
        return ghat

    # The de Hoog underflow guard zeros far-field cells whose transform decayed to the double-precision floor
    # (physical ~0); a cell whose transform genuinely OVERFLOWED stays NaN so a real breakdown cannot read as
    # a physical zero. No extra masking here (that would zero overflowed cells too, reversing that contract).
    return dehoog_inverse(f_hat=f_hat, t=tau, n_terms=n_terms, scaling=scaling, tol=tol)


def _fr_source_matrix(
    v_nodes: npt.NDArray[np.floating], edges: npt.NDArray[np.floating], **readout: float
) -> npt.NDArray[np.floating]:
    r"""Injection-source operator ``M[k, j]`` such that ``field_k += sum_j M[k, j] cin_dev_j`` over one phase.

    Matrix form of the per-reversal ``_fr_profile``:
    ``M[k, j] = G1_FR(S_inj - sigma_j; V'_k) - G1_FR(S_inj - sigma_{j+1}; V'_k)`` (the same FR step response,
    KB Sec. 10a), built once per distinct injection phase and reused across recurring cycles.

    Returns
    -------
    ndarray
        Source operator, shape ``(n_quad, n_inj_bins)``.
    """
    # edges are phase-relative (edges[0] == 0 by caller construction); corners descend to 0 at the phase end
    inj_corners = edges[-1] - edges  # descending within-phase cumulative volume corners, last is 0
    g = np.array([_fr_step_response(v, inj_corners, **readout) for v in v_nodes])  # (n_quad, n_bins + 1)
    return g[:, :-1] - g[:, 1:]


def _cout_readout_matrix(
    v_nodes: npt.NDArray[np.floating],
    dv_weights: npt.NDArray[np.floating],
    edges: npt.NDArray[np.floating],
    **readout: float,
) -> npt.NDArray[np.floating]:
    r"""Cout-readout operator ``M[i, k]`` such that ``cout_i = sum_k M[i, k] field_k`` over one phase.

    Matrix form of the per-reversal ``_cout_phase``:
    ``M[i, k] = R dv_k [G1_FR(T_{i+1}; V'_k) - G1_FR(T_i; V'_k)] / dT_i`` (the KB Sec. 7 duality arrival
    kernel, same FR step response; the ``R`` amplitude mobilises the sorbed companion), built once per
    distinct extraction phase and reused.

    Returns
    -------
    ndarray
        Readout operator, shape ``(n_ext_bins, n_quad)``.
    """
    # edges are phase-relative (edges[0] == 0 by caller construction)
    dt = np.diff(edges)
    g = np.array([_fr_step_response(v, edges, **readout) for v in v_nodes])  # (n_quad, n_bins + 1)
    diffs = (g[:, 1:] - g[:, :-1]) / dt[None, :]
    return (readout["retardation_factor"] * dv_weights[:, None] * diffs).T


def cout_deviation(
    *,
    cin_deviation: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geo: float,
    r_w: float,
    alpha_l: float,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    n_quad: int = 240,
    n_terms: int = _DEHOOG_TERMS,
    tol: float = 1e-9,
) -> npt.NDArray[np.floating]:
    r"""Multi-cycle extracted-flux deviation via reused per-phase propagator matrices (any ``D_m``).

    Drop-in for the per-reversal ``gridfree_cout_deviation`` reference: every per-phase linear
    operator -- the across-reversal field hand-off (``field = P @ field``), the injection source
    (:func:`_fr_source_matrix`, ``field += M_fr @ cin``), and the cout readout (:func:`_cout_readout_matrix`,
    ``cout = M_cout @ field``) -- is built once per distinct phase and reused as a matrix-multiply whenever
    that phase recurs, instead of recomputing the de Hoog inversion / FR step response each cycle. The
    propagator hand-off uses the Airy branch (:func:`_airy_propagator_matrix`,
    flushed-volume clock) for ``D_m = 0``, the Riccati branch (:func:`_riccati_propagator_matrix`,
    wall-clock) for ``D_m > 0``, and the Bessel rest branch (:func:`_rest_propagator_matrix`) for ``Q = 0``
    phases when ``D_m > 0`` -- so the special-function + de Hoog cost is ``O(number of distinct phases)``
    rather than ``O(K)`` reversals. Equal to the per-reversal grid-free engine to the de Hoog floor at
    matched ``n_terms``/``tol``; tighten both for machine precision (the de Hoog Pade nonlinearity, dominant
    at low Peclet / high retardation).

    Parameters
    ----------
    cin_deviation : ndarray, shape (n,) or (n, k)
        Injected concentration deviation per bin (used on injection bins, ``flow > 0``). A trailing column
        axis is transported through one engine pass (the per-phase matrices are cin-independent), used by
        the reverse operator build to apply all unit-pulse columns at once.
    flow : ndarray, shape (n,)
        Signed flow per bin [m^3/day]: ``> 0`` injection, ``< 0`` extraction, ``0`` rest.
    dt_days : ndarray, shape (n,)
        Bin widths [day].
    c_geo : float
        Geometry constant ``pi b n`` (``V = c_geo (r^2 - r_w^2)``).
    r_w : float
        Well radius [m].
    alpha_l : float
        Longitudinal dispersivity [m].
    molecular_diffusivity : float, optional
        Molecular diffusivity [m^2/day]. ``0`` selects the Airy branch (flushed-volume clock); ``> 0`` the
        Riccati log-derivative pumping branch and the order-0 Bessel rest branch (both wall-clock). Default 0.
    retardation_factor : float, optional
        Linear retardation ``R >= 1`` (rescales the flushed-volume clock). Default 1.
    n_quad : int, optional
        Number of radial Gauss-Legendre nodes. Default 240.
    n_terms : int, optional
        de Hoog series length for the matrix build. Default ``_DEHOOG_TERMS``.
    tol : float, optional
        de Hoog target accuracy for the matrix build. Default ``1e-9``. Tighten (e.g.
        ``tol=1e-13, n_terms=64``) for machine precision at low Peclet / high retardation.

    Returns
    -------
    ndarray, shape (n,) or (n, k)
        Extracted-flux deviation per bin (matching ``cin_deviation``); ``NaN`` on injection / rest bins.
    """
    flow = np.asarray(flow, dtype=float)
    cin_deviation = np.asarray(cin_deviation, dtype=float)
    batch = cin_deviation.shape[1:]  # () for a single series, (k,) for a column batch (reverse operator build)
    flushed = np.abs(flow) * dt_days
    # D_m = 0 keeps the advective grid; D_m > 0 widens it to the molecular reach (the diffusive pumping and
    # rest kernels both spread beyond the advective front), matching gridfree_cout_deviation.
    r_nodes, v_nodes, dr_weights = _field_grid(flow, dt_days, c_geo, r_w, alpha_l, molecular_diffusivity, n_quad)
    dv_weights = 2.0 * c_geo * r_nodes * dr_weights

    phases = _phase_slices(flow)
    matrices: dict[tuple, npt.NDArray[np.floating]] = {}

    def propagate(field, direction, flow_scale, phase_volume, phase_time):
        if molecular_diffusivity == 0.0:  # Airy: flushed-volume clock, retardation rescales the clock
            tau = phase_volume / retardation_factor
            key = ("airy", direction, round(tau, 9))  # matrix is flow-magnitude independent (S-clock autonomy)
            if key not in matrices:
                matrices[key] = _airy_propagator_matrix(
                    direction,
                    tau,
                    r_nodes,
                    dr_weights,
                    r_w=r_w,
                    alpha_l=alpha_l,
                    c_geo=c_geo,
                    n_terms=n_terms,
                    tol=tol,
                )
            return matrices[key] @ field
        key = ("riccati", direction, round(phase_time, 9), round(flow_scale, 9))  # D_m > 0: wall-clock time
        if key not in matrices:
            matrices[key] = _riccati_propagator_matrix(
                direction,
                phase_time,
                r_nodes,
                dr_weights,
                r_w=r_w,
                alpha_l=alpha_l,
                c_geo=c_geo,
                flow_scale=flow_scale,
                molecular_diffusivity=molecular_diffusivity,
                retardation_factor=retardation_factor,
                n_terms=n_terms,
                tol=tol,
            )
        return matrices[key] @ field

    field = np.zeros((n_quad, *batch))
    cout = np.full((len(flow), *batch), np.nan)
    for idx, (sign, sl) in enumerate(phases):
        phase_volume = float(flushed[sl].sum())
        if phase_volume == 0.0:  # rest: pure molecular diffusion on the wall-clock clock (D_m = 0 -> identity)
            if molecular_diffusivity > 0.0:
                tau = float(np.sum(dt_days[sl]))
                key = ("rest", round(tau, 9))
                if key not in matrices:
                    matrices[key] = _rest_propagator_matrix(
                        tau,
                        r_nodes,
                        dr_weights,
                        r_w=r_w,
                        d_m_eff=molecular_diffusivity / retardation_factor,
                        n_terms=n_terms,
                        tol=tol,
                    )
                field = matrices[key] @ field
            continue
        phase_time = float(np.sum(dt_days[sl]))
        flow_scale = phase_volume / phase_time  # dt-weighted mean |Q|: flow_scale * phase_time = flushed volume
        edges = np.concatenate(([0.0], np.cumsum(flushed[sl])))
        readout = {
            "c_geo": c_geo,
            "r_w": r_w,
            "alpha_l": alpha_l,
            "retardation_factor": retardation_factor,
            "flow_scale": flow_scale,
            "molecular_diffusivity": molecular_diffusivity,
        }
        # The injection-source and cout-readout operators are also linear and phase-structure-determined, so
        # they are matrix-cached and reused across recurring cycles too (retardation R and D_m are fixed for
        # the whole call, so the cache key is the within-phase volume edges + the flow scale).
        ekey = (tuple(np.round(edges, 9)), round(flow_scale, 9))
        if sign > 0:  # injection: propagate the buffer, then add the new injected resident profile
            field = propagate(field, _INJECTION, flow_scale, phase_volume, phase_time)
            mkey = ("fr", *ekey)
            if mkey not in matrices:
                matrices[mkey] = _fr_source_matrix(v_nodes, edges, **readout)
            field += matrices[mkey] @ cin_deviation[sl]
        else:  # extraction: read cout, then propagate the residual if more pumping follows
            mkey = ("cout", *ekey)
            if mkey not in matrices:
                matrices[mkey] = _cout_readout_matrix(v_nodes, dv_weights, edges, **readout)
            cout[sl] = matrices[mkey] @ field
            if idx != len(phases) - 1:
                field = propagate(field, "extraction", flow_scale, phase_volume, phase_time)
    return cout
