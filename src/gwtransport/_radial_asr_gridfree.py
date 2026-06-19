r"""Grid-free multi-cycle / general signed-flow radial advection-dispersion engine (any ``D_m``).

This is the exact, mesh-free engine for signed-flow schedules with more than one flow reversal
(multi-cycle ASR / SWIW). It composes the per-phase closed-form pieces of the radial ASR knowledge
base (KB Sec. 4-7, addendum Sec. A1-A4) -- no PDE is discretized, so none of the finite-volume
artefacts (CFL/Courant limit, numerical dispersion, Crank-Nicolson ringing) appear. The only numerical
operations are Gauss-Legendre quadrature over the resident profile and de Hoog Laplace inversion of the
exact special-function kernels: the Airy interior Green's functions for ``D_m = 0`` (flushed-volume
clock, vectorized) and the Whittaker interior Green's functions for appreciable ``D_m > 0`` (wall-clock
time, mpmath -- the molecular-diffusion regime is chosen by :func:`_molecular_regime`, KB addendum
Sec. A6). The spectral-domain acceleration of the composition (KB addendum Sec. A4-A7) is not
implemented here.

State machine (deviation ``c' = c - c_bg``, initial condition 0)
---------------------------------------------------------------
The state is the resident deviation profile carried on a fixed radial Gauss-Legendre grid. Consecutive
same-sign flow bins form a phase; each phase advances the state, and extraction phases additionally
read out ``cout``:

* **Injection** (``Q > 0``, divergent): the pre-existing buffer field is propagated outward by the
  divergent interior Green's function ``G_+`` (Robin/flux well BC), and the new injected signal adds
  its resident profile via the flux-resident (FR) step response (KB Sec. 10a). The two are independent
  by linearity (new water builds from zero, the buffer advects under ``F_well = 0``).
* **Extraction** (``Q < 0``, convergent): ``cout`` is the flow-weighted bin average of the well-face
  flux concentration, read from the current field via the KB Sec. 7 duality kernel (the same FR step
  response); if more pumping follows, the residual field is propagated inward by the convergent
  interior Green's function ``G_-`` (Danckwerts/Neumann well BC) for the next phase.
* **Rest** (``Q = 0``, ``D_m = 0``): identity (no transport).

Everything is carried in the flushed-volume clock, so arbitrary within-phase variable flow is exact
(the S-clock convolution theorem, KB Sec. 3). Retardation ``R`` is a pure clock rescale (propagate
over flushed-volume ``/R``). The across-reversal field hand-off ``G_+ / G_-`` (addendum Sec. A3) is the
``O(n_quad^2)`` part; it runs once per intermediate reversal.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import mpmath as mp
import numpy as np
import numpy.typing as npt

from gwtransport._radial_asr_compose import _fr_step_response
from gwtransport._radial_asr_dehoog import dehoog_inverse
from gwtransport._radial_asr_kernels import (
    _molecular_regime,
    _resolvent_airy_pieces,
    assemble_airy_resolvent,
    whittaker_resolvent_solutions,
)

# Working precision (decimal digits) for the mpmath Whittaker (D_m > 0) field propagation.
_WHITTAKER_DPS = 30

# de Hoog series length for the field-propagation inversion and the front-anchored scaling margin
# (matched to the FR step response in _radial_asr_compose).
_DEHOOG_TERMS = 44
_SCALE_MARGIN = 1.3


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


def _fr_profile(
    v_nodes: npt.NDArray[np.floating],
    cin_dev: npt.NDArray[np.floating],
    phase_volume_edges: npt.NDArray[np.floating],
    *,
    c_geo: float,
    r_w: float,
    alpha_l: float,
    retardation_factor: float,
    flow_scale: float,
    molecular_diffusivity: float,
) -> npt.NDArray[np.floating]:
    """Resident deviation profile built by injecting ``cin_dev`` over one injection phase (KB Sec. 10a).

    ``f(V') = sum_j cin_dev_j [G1_FR(S_inj - sigma_j; V') - G1_FR(S_inj - sigma_{j+1}; V')]`` evaluated
    at every grid node, with ``sigma`` the within-phase cumulative flushed-volume edges.

    Returns
    -------
    ndarray
        Resident deviation at each ``v_node``, shape ``(n_quad,)``.
    """
    sigma = phase_volume_edges - phase_volume_edges[0]
    s_inj = sigma[-1]
    inj_corners = s_inj - sigma  # descending, last is 0
    out = np.zeros(len(v_nodes))
    for k, v in enumerate(v_nodes):
        g_inj = _fr_step_response(
            v,
            inj_corners,
            c_geo=c_geo,
            r_w=r_w,
            alpha_l=alpha_l,
            retardation_factor=retardation_factor,
            flow_scale=flow_scale,
            molecular_diffusivity=molecular_diffusivity,
        )
        out[k] = np.sum((g_inj[:-1] - g_inj[1:]) * cin_dev)
    return out


def _cout_phase(
    field: npt.NDArray[np.floating],
    v_nodes: npt.NDArray[np.floating],
    dv_weights: npt.NDArray[np.floating],
    ext_volume_edges: npt.NDArray[np.floating],
    *,
    c_geo: float,
    r_w: float,
    alpha_l: float,
    retardation_factor: float,
    flow_scale: float,
    molecular_diffusivity: float,
) -> npt.NDArray[np.floating]:
    """Flow-weighted bin-average extracted-flux deviation over one extraction phase (duality readout).

    ``cout_i = sum_k field_k [G1_FR(T_{i+1}; V'_k) - G1_FR(T_i; V'_k)] / dT_i * dV'_k`` -- the KB Sec. 7
    duality arrival kernel (same FR step response) integrated over the output volume bins.

    Returns
    -------
    ndarray
        Extracted-flux deviation per output bin, shape ``(n_ext,)``.
    """
    t_edges = ext_volume_edges - ext_volume_edges[0]
    dt = np.diff(t_edges)
    out = np.zeros(len(dt))
    for v, fk, dv in zip(v_nodes, field, dv_weights, strict=True):
        if fk == 0.0:
            continue
        g_ext = _fr_step_response(
            v,
            t_edges,
            c_geo=c_geo,
            r_w=r_w,
            alpha_l=alpha_l,
            retardation_factor=retardation_factor,
            flow_scale=flow_scale,
            molecular_diffusivity=molecular_diffusivity,
        )
        out += fk * dv * (g_ext[1:] - g_ext[:-1]) / dt
    # Retardation amplitude: the readout's arrival CDF plateaus at 1 but each extracted mobile parcel
    # mobilizes its sorbed companion (total solute = R x mobile), so scale by R (see single_cycle_echo_matrix).
    return retardation_factor * out


def _propagate(
    field: npt.NDArray[np.floating],
    r_nodes: npt.NDArray[np.floating],
    src_measure: npt.NDArray[np.floating],
    direction: str,
    tau: float,
    *,
    r_w: float,
    alpha_l: float,
    flow_scale: float,
    c_geo: float,
) -> npt.NDArray[np.floating]:
    """Propagate a resident field by flushed-volume ``tau`` with the Airy interior Green's function.

    ``f_resid(r_i) = L^{-1}_p[ sum_k Ghat(r_i, r_k; p) field_k src_measure_k ](tau)`` -- one de Hoog
    inversion per output node, with the source superposition folded into the transform (it is linear
    and commutes with the inversion). ``src_measure_k = w(r_k) dr_weights_k`` carries the
    Sturm-Liouville weight; ``flow_scale`` cancels for ``D_m = 0`` (the autonomous S/T-clock). The
    scaled Airy on the grid is evaluated ONCE per de Hoog node set (cached and reused across output
    nodes), and each output node is assembled by prefix selection at ``r_i`` -- ``O(n_quad)`` Airy
    evaluations rather than ``O(n_quad^2)``.

    Returns
    -------
    ndarray
        Propagated resident deviation at each node, shape ``(n_quad,)``.
    """
    a0 = flow_scale / (2.0 * c_geo)
    gauge_sign = 1.0 if direction == "injection" else -1.0
    weighted = field * src_measure
    if not np.any(weighted != 0.0):
        return np.zeros_like(field)
    mu = c_geo * ((float(r_nodes.max()) + alpha_l) ** 2 + alpha_l**2 - r_w**2)
    scaling = _SCALE_MARGIN * max(mu, tau)
    r_grid = r_nodes.reshape(1, -1)
    cache: dict[bytes, tuple] = {}

    def pieces(p: npt.NDArray[np.complexfloating]) -> tuple:
        s = (flow_scale * p).reshape(-1, 1)
        return (
            _resolvent_airy_pieces(s, r_grid, alpha_l, a0, gauge_sign),
            _resolvent_airy_pieces(s, np.array([[r_w]]), alpha_l, a0, gauge_sign),
        )

    out = np.empty(len(r_nodes))
    below = r_nodes[None, :] < r_nodes[:, None]  # below[i, j] = r_j < r_i
    for i in range(len(r_nodes)):

        def f_hat(p: npt.NDArray[np.complexfloating], i: int = i) -> npt.NDArray[np.complexfloating]:
            key = p.tobytes()
            if key not in cache:
                cache[key] = pieces(p)
            grid_p, piece_w = cache[key]
            # r_< / r_> selection at output i: u_0 sits at r_<, u_inf at r_> (the SL kernel is symmetric).
            # The radii enter assembly only through r_< + r_> = r_i + r_j (no per-pair min/max needed).
            mask = below[i][None, :]
            piece_a = {f: np.where(mask, grid_p[f], grid_p[f][:, i : i + 1]) for f in grid_p}
            piece_b = {f: np.where(mask, grid_p[f][:, i : i + 1], grid_p[f]) for f in grid_p}
            r_sum = (r_nodes[i] + r_nodes)[None, :]
            ghat = assemble_airy_resolvent(piece_a, piece_b, piece_w, r_sum, alpha_l, gauge_sign)
            return ghat @ weighted

        out[i] = dehoog_inverse(f_hat=f_hat, t=np.array([tau]), n_terms=_DEHOOG_TERMS, scaling=scaling)[0]
    return out


def _propagate_whittaker(
    field: npt.NDArray[np.floating],
    r_nodes: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    direction: str,
    tau: float,
    *,
    r_w: float,
    alpha_l: float,
    flow_scale: float,
    c_geo: float,
    molecular_diffusivity: float,
    retardation_factor: float,
) -> npt.NDArray[np.floating]:
    r"""Propagate a resident field by wall-clock time ``tau`` with the ``D_m > 0`` Whittaker resolvent.

    Same superposition as :func:`_propagate` but with the Whittaker interior Green's function (KB
    Sec. 4 ``D_m > 0`` branch) and the wall-clock time clock (the volume clock is not autonomous when
    ``D_m > 0`` -- the no-go lemma). No vectorized complex Whittaker exists, so the solutions are
    sampled in mpmath; the cost is contained by (a) evaluating them ONCE per de Hoog node and grid
    radius and (b) assembling every output node from prefix/suffix sums of the source superposition
    ``F(s)_i = -[u_inf(r_i) sum_{j<=i} u_0(r_j) w_j + u_0(r_i) sum_{j>i} u_inf(r_j) w_j]/N(s)`` (using
    the symmetric SL split at ``r_i``), keeping the whole build at ``O(n_nodes * n_quad)`` mpmath
    evaluations. mpmath carries the growing-solution magnitude exactly, and only the bounded ``F(s)_i``
    is cast to double. ``src_measure_k = w(r_k) dr_weights_k`` with the ``D_m > 0`` weight
    ``w_pm(r) = r G(r)^{-/+ A_0/D_m}`` (``G = alpha_L A_0 + D_m r``).

    Returns
    -------
    ndarray
        Propagated resident deviation at each node, shape ``(n_quad,)``.
    """
    a0_eff = (flow_scale / (2.0 * c_geo)) / retardation_factor
    d_m_eff = molecular_diffusivity / retardation_factor
    sigma_a = 1 if direction == "injection" else -1
    # Sturm-Liouville weight w_pm(r) = r G^{b-1}, b-1 = -sigma_A a0/d_m (G = alpha_L a0 + d_m r),
    # matched to the resolvent normalization N = G^b W (the constant in P/w cancels in G_SL * w).
    g_eff = alpha_l * a0_eff + d_m_eff * r_nodes
    sl_weight = r_nodes * g_eff ** (-sigma_a * a0_eff / d_m_eff)
    weighted = (field * sl_weight * dr_weights).astype(complex)
    if not np.any(weighted != 0.0):
        return np.zeros_like(field)
    n = len(r_nodes)
    mu_t = c_geo * ((float(r_nodes.max()) + alpha_l) ** 2 + alpha_l**2 - r_w**2) / flow_scale
    scaling = _SCALE_MARGIN * max(mu_t, tau)
    cache: dict[bytes, npt.NDArray[np.complexfloating]] = {}

    def build(p: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        f_mat = np.empty((len(p), n), dtype=complex)
        with mp.workdps(_WHITTAKER_DPS):
            b_exp = 1 - sigma_a * a0_eff / d_m_eff
            for k, pk in enumerate(p):
                sk = complex(pk)
                uiw, uiwp, urw, urwp = whittaker_resolvent_solutions(sk, r_w, alpha_l, a0_eff, d_m_eff, sigma_a)
                if sigma_a < 0:  # extraction: Danckwerts/Neumann u_0'(r_w) = 0
                    bc_reg, bc_inf = urwp, uiwp
                else:  # injection: Robin F[u]=u-(G/A_0)u', F[u_0](r_w)=0
                    fac = alpha_l + d_m_eff * r_w / a0_eff  # = G(r_w)/A_0
                    bc_reg, bc_inf = urw - fac * urwp, uiw - fac * uiwp
                u0w = bc_reg * uiw - bc_inf * urw
                u0wp = bc_reg * uiwp - bc_inf * urwp
                n_s = (alpha_l * a0_eff + d_m_eff * r_w) ** b_exp * (u0w * uiwp - u0wp * uiw)
                u_inf_g, u0_g = [], []
                for r in r_nodes:
                    ui, _, ur, _ = whittaker_resolvent_solutions(sk, float(r), alpha_l, a0_eff, d_m_eff, sigma_a)
                    u_inf_g.append(ui)
                    u0_g.append(bc_reg * ui - bc_inf * ur)
                wj = [mp.mpc(complex(weighted[m])) for m in range(n)]
                prefix, acc = [mp.mpc(0)] * n, mp.mpc(0)
                for m in range(n):  # inclusive prefix sum_{j<=m} u_0(r_j) w_j
                    acc += u0_g[m] * wj[m]
                    prefix[m] = acc
                suffix, acc = [mp.mpc(0)] * n, mp.mpc(0)
                for m in range(n - 1, -1, -1):  # exclusive suffix sum_{j>m} u_inf(r_j) w_j
                    suffix[m] = acc
                    acc += u_inf_g[m] * wj[m]
                for i in range(n):
                    f_mat[k, i] = complex(-(u_inf_g[i] * prefix[i] + u0_g[i] * suffix[i]) / n_s)
        return f_mat

    out = np.empty(n)
    for i in range(n):

        def f_hat(p: npt.NDArray[np.complexfloating], i: int = i) -> npt.NDArray[np.complexfloating]:
            key = p.tobytes()
            if key not in cache:
                cache[key] = build(p)
            return cache[key][:, i]

        out[i] = dehoog_inverse(f_hat=f_hat, t=np.array([tau]), n_terms=_DEHOOG_TERMS, scaling=scaling)[0]
    return out


def gridfree_cout_deviation(
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
) -> npt.NDArray[np.floating]:
    """Extracted-flux deviation per bin for a general signed-flow schedule, grid-free.

    Composes the exact per-phase kernels through the multi-cycle state machine (see the module
    docstring): the Airy branch for ``D_m = 0`` (vectorized, flushed-volume clock) and the Whittaker
    branch for ``D_m > 0`` (mpmath, wall-clock time -- the volume clock is not autonomous when
    ``D_m > 0``). Returns the flow-weighted bin average of the well-face flux concentration deviation
    on extraction bins, ``NaN`` on injection / rest bins.

    Parameters
    ----------
    cin_deviation : ndarray, shape (n,)
        Injected concentration deviation per bin (used on injection bins, ``flow > 0``).
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
        Molecular diffusivity [m^2/day]. ``0`` selects the Airy branch; ``> 0`` the Whittaker branch.
        Default 0.
    retardation_factor : float, optional
        Linear retardation ``R >= 1``. Default 1.
    n_quad : int, optional
        Number of radial Gauss-Legendre nodes for the resident-profile quadrature. Default 240.

    Returns
    -------
    ndarray, shape (n,)
        Extracted-flux deviation per bin; ``NaN`` on injection / rest bins.
    """
    flushed = np.abs(flow) * dt_days
    # Build the advective (D_m-independent) grid first -- its reach is what the basis-by-regime decision
    # compares against, and the molecular reach is only a grid detail once D_m is known to matter.
    r_nodes, v_nodes, dr_weights = _field_grid(flow, dt_days, c_geo, r_w, alpha_l, 0.0, n_quad)
    # Basis-by-regime (KB addendum Sec. A6): the Airy reduction where molecular diffusion is
    # sub-dominant within the plume (or its exact Whittaker treatment intractable), the exact Whittaker
    # branch where it is appreciable and tractable -- compared against the advective plume reach.
    a0_repr = float(np.mean(np.abs(flow[flow != 0.0]))) / (2.0 * c_geo) if np.any(flow != 0.0) else 0.0
    molecular_diffusivity = _molecular_regime(molecular_diffusivity, a0_repr, alpha_l, float(r_nodes.max()))
    if molecular_diffusivity > 0.0:  # appreciable + tractable: widen the grid to the molecular reach
        r_nodes, v_nodes, dr_weights = _field_grid(flow, dt_days, c_geo, r_w, alpha_l, molecular_diffusivity, n_quad)
    dv_weights = 2.0 * c_geo * r_nodes * dr_weights  # dV' = 2 c_geo r' dr' (cout readout measure)
    # Airy Sturm-Liouville propagation weights w_pm(r') dr' = (2 c_geo r'/alpha_L) e^{-/+ r'/alpha_L} dr'
    # (D_m = 0); the D_m > 0 weight is flow-dependent and built inside _propagate_whittaker.
    sl_inj = (2.0 * c_geo * r_nodes / alpha_l) * np.exp(-r_nodes / alpha_l) * dr_weights
    sl_ext = (2.0 * c_geo * r_nodes / alpha_l) * np.exp(r_nodes / alpha_l) * dr_weights

    def propagate(
        field: npt.NDArray[np.floating], direction: str, flow_scale: float, phase_volume: float, phase_time: float
    ) -> npt.NDArray[np.floating]:
        if molecular_diffusivity == 0.0:  # Airy: flushed-volume clock, retardation rescales the clock
            sl = sl_inj if direction == "injection" else sl_ext
            return _propagate(
                field,
                r_nodes,
                sl,
                direction,
                phase_volume / retardation_factor,
                r_w=r_w,
                alpha_l=alpha_l,
                flow_scale=flow_scale,
                c_geo=c_geo,
            )
        return _propagate_whittaker(  # Whittaker: wall-clock time, retardation rescales A_0, D_m
            field,
            r_nodes,
            dr_weights,
            direction,
            phase_time,
            r_w=r_w,
            alpha_l=alpha_l,
            flow_scale=flow_scale,
            c_geo=c_geo,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
        )

    field = np.zeros(n_quad)
    cout = np.full(len(flow), np.nan)
    phases = _phase_slices(flow)
    for idx, (sign, sl) in enumerate(phases):
        phase_flushed = flushed[sl]
        phase_volume = float(phase_flushed.sum())
        if phase_volume == 0.0:  # rest (D_m = 0): identity
            continue
        flow_scale = float(np.mean(np.abs(flow[sl])))
        phase_time = float(np.sum(dt_days[sl]))
        edges = np.concatenate(([0.0], np.cumsum(phase_flushed)))
        if sign > 0:  # injection: propagate the buffer, then add the new injected resident profile
            field = propagate(field, "injection", flow_scale, phase_volume, phase_time)
            field += _fr_profile(
                v_nodes,
                cin_deviation[sl],
                edges,
                c_geo=c_geo,
                r_w=r_w,
                alpha_l=alpha_l,
                retardation_factor=retardation_factor,
                flow_scale=flow_scale,
                molecular_diffusivity=molecular_diffusivity,
            )
        else:  # extraction: read cout, then propagate the residual if more pumping follows
            cout[sl] = _cout_phase(
                field,
                v_nodes,
                dv_weights,
                edges,
                c_geo=c_geo,
                r_w=r_w,
                alpha_l=alpha_l,
                retardation_factor=retardation_factor,
                flow_scale=flow_scale,
                molecular_diffusivity=molecular_diffusivity,
            )
            if idx != len(phases) - 1:
                field = propagate(field, "extraction", flow_scale, phase_volume, phase_time)
    return cout
