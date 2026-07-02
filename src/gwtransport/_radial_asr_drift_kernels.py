r"""Block (azimuthal-mode) Laplace kernels for single-well ASR with steady regional drift.

Generalizes the radial per-phase kernels (:mod:`gwtransport._radial_asr_kernels`) to a steady uniform
regional drift ``v_d = U/n`` superimposed on the radial well field ``A_0/r``. The 2-D
advection-dispersion generator ``O[c] = v.grad c - div(D grad c)`` (rank-1 Scheidegger tensor
``D = alpha_L (v outer v)/|v| + D_m I``, transverse mechanical dispersion neglected) is expanded in
azimuthal Fourier modes ``c(r, theta) = sum_m c_m(r) e^{i m theta}``. The radial well flow acts within
each mode (``m = 0`` is the radial engine); the drift couples mode ``m`` to its neighbours, giving a
coupled second-order block ODE

    A(r) c'' + B(r) c' + (S0(r) + R s I) c = 0,      c = (c_{-M}, ..., c_M)^T,

with the ``s``-independent coefficient matrices (``M[f]`` the azimuthal Toeplitz coupling matrix whose
``(m, m')`` entry is the ``(m - m')``-th Fourier coefficient of ``f(r, .)``; ``Dm = diag(modes)``)

    A  = -M[D_rr]
    B  =  M[v_r - D_rr/r - d_r D_rr - (1/r) d_theta D_rtheta]  +  M[-2 D_rtheta/r] (i Dm)
    S0 =  M[v_theta/r - (d_r D_rtheta)/r - (1/r^2) d_theta D_thth] (i Dm)  +  M[-D_thth/r^2] (-Dm^2),

derived by collecting ``c''``, ``c'`` and ``c`` (the ``d_r d_theta c`` cross term folds into ``B`` via
``d_theta -> i m``; the ``d_theta^2`` term into ``S0`` via ``-m^2``). At ``v_d = 0`` every ``M[.]`` is
diagonal and the system decouples into the radial per-mode operator, so the ``m = 0`` block is exactly
the radial engine's ODE ``G c'' + (D_m - A_0) c' - R s r c = 0`` (``G = alpha_L A_0 + D_m r``).

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator
from scipy.special import airye

from gwtransport._radial_asr_dehoog import dehoog_inverse
from gwtransport._radial_asr_reuse import _phase_slices  # generic signed-schedule grouping (shared)

# Per-phase orientation. Injection (Q > 0) is the divergent operator (Robin/flux well BC); extraction
# (Q < 0) the convergent operator (Danckwerts/Neumann well BC). The signed well strength A_0 carries the
# sign, so the azimuthal drift coupling uses the signed eps = v_d r / A_0 per phase automatically.
_INJECTION = "injection"

# Riccati integration tolerances (matched to the scalar log-derivative kernel) and the drift-specific
# outer-boundary policy: the recessive initial condition is set at r_far and washed inward; r_far is
# floored well past the field but hard-capped below the stagnation radius r_s = |A_0|/v_d, where the
# coordinate finite-escape of the matrix Riccati would otherwise blow up (the plan's r_far <= 0.6 r_s).
# The slow-drift envelope is honest: the SIGNIFICANT plume (front + _PLUME_WIDTHS breakthrough widths,
# where the resident field has decayed to ~1% of peak) must fit below that cap, else the engine raises
# rather than silently truncate it. Only the negligible far tail beyond r_far is dropped (the recessive
# IC washes it out anyway). At v_d = 0 (r_s = inf) the cap is inactive and this is the scalar grid.
_RICCATI_RTOL = 1e-9
_RICCATI_ATOL = 1e-10
_RFAR_FIELD_MULT = 8.0
_RFAR_DECAY = 44.0
_RS_FRAC = 0.6
_GRID_WIDTHS = 12.0
_PLUME_WIDTHS = 3.0
# Rest-with-drift kernel: Gauss-Hermite quadrature size for the Gaussian spread, and the honest azimuthal
# truncation guard -- the relative spectral tail (harmonics M < |m| <= 2M after reprojection) above which
# the translated plume is declared too eccentric for the kept modes (raise, never silently fold the tail).
_REST_HERMITE = 20
_REST_TAIL_MAX = 1e-2
# Per-call cap on cached per-phase kernel solutions (each entry is O(n_quad n_s (2M+1)^2) complex, ~70 MB
# at the defaults; a periodic schedule needs one entry per pumping direction, so the cap only sheds
# entries on long aperiodic schedules, where nothing recurs anyway).
_SOLUTIONS_CACHE_MAX = 8


def _toeplitz_from_theta(f_theta: npt.NDArray[np.floating], n_modes: int) -> npt.NDArray[np.complexfloating]:
    r"""Azimuthal coupling matrix ``M[f]`` from samples of ``f(theta)`` on a uniform grid.

    ``M[f]_{a,b} = c_{m_a - m_b}`` where ``c_k`` is the ``k``-th Fourier coefficient of ``f`` (the
    multiplication-by-``f`` operator in the ``e^{i m theta}`` basis), ``m`` running ``-n_modes .. n_modes``.
    The grid must satisfy ``len(f_theta) >= 4 n_modes + 1`` so every needed coefficient ``|k| <= 2 n_modes``
    is unaliased.

    Returns
    -------
    ndarray of complex, shape (2 n_modes + 1, 2 n_modes + 1)
        The Toeplitz coupling matrix.
    """
    n = f_theta.shape[-1]
    coeffs = np.fft.fft(f_theta, axis=-1) / n  # coeffs[k] = c_k, with c_{-k} at index n-k
    modes = np.arange(-n_modes, n_modes + 1)
    diff = modes[:, None] - modes[None, :]  # m_a - m_b in [-2M, 2M]
    return coeffs[..., diff % n]


def block_coupling_matrices(
    r: npt.NDArray[np.floating],
    *,
    alpha_l: float,
    a0: float,
    v_d: float,
    d_m: float,
    n_modes: int,
    n_theta: int | None = None,
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    r"""Coefficient matrices ``A(r), B(r), S0(r)`` of the coupled-mode block ODE.

    The block ODE is ``A c'' + B c' + (S0 + R s I) c = 0``; ``A, B, S0`` are ``s``-independent (the Laplace
    node enters only through ``R s I``). ``a0`` is the signed well strength ``A_0 = Q/(2 pi b n)`` (``> 0``
    injection, ``< 0`` extraction); ``v_d = U/n`` the regional drift. Built by FFT-in-theta of the exact
    Scheidegger tensor components and their analytic radial derivatives.

    Parameters
    ----------
    r : ndarray
        Radii (m), shape ``(n_r,)``.
    alpha_l : float
        Longitudinal dispersivity (m).
    a0 : float
        Signed well strength ``A_0 = Q/(2 pi b n)`` (m^2/day).
    v_d : float
        Regional drift seepage velocity ``U/n`` (m/day).
    d_m : float
        Molecular diffusivity (m^2/day).
    n_modes : int
        Azimuthal truncation ``M`` (keeps modes ``-M .. M``).
    n_theta : int, optional
        Azimuthal FFT grid size. The needed Fourier coefficients ``|k| <= 2 n_modes`` of the tensor
        components decay like ``eps^|k|`` and a coefficient ``k`` aliases the ``k +- n_theta`` harmonic
        (``~ eps^(n_theta - 2 n_modes)``); the default ``8 n_modes + 48`` keeps that alias below ~1e-12
        across the slow-drift envelope (``eps <= 0.6`` at the grid edge), where ``8 n_modes + 8`` would
        leave a ~1e-2 alias at the envelope's strong end.

    Returns
    -------
    a_mat, b_mat, s0_mat : ndarray of complex, each shape (n_r, 2 n_modes + 1, 2 n_modes + 1)
        The coefficient matrices at each radius.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    nth = n_theta if n_theta is not None else 8 * n_modes + 48
    theta = np.arange(nth) * (2.0 * np.pi / nth)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    rr = r[:, None]  # (n_r, 1) broadcast against theta
    v_r = a0 / rr + v_d * cos_t[None, :]  # (n_r, nth)
    v_th = -v_d * sin_t[None, :]  # theta-only (r-independent); broadcasts against the (n_r, .) factors
    speed = np.sqrt(v_r**2 + v_th**2)
    speed = np.where(speed == 0.0, 1.0, speed)  # a0=v_d=0 never reached here (alpha_L>0 guard upstream)

    d_rr = alpha_l * v_r**2 / speed + d_m
    d_rt = alpha_l * v_r * v_th / speed
    d_tt = alpha_l * v_th**2 / speed + d_m

    # analytic radial derivatives (v_th is r-independent; d_r v_r = -a0/r^2)
    dv_r = -a0 / rr**2
    dspeed = v_r * dv_r / speed
    d_drr = alpha_l * (2.0 * v_r * dv_r / speed - v_r**2 * dspeed / speed**2)
    d_drt = alpha_l * (dv_r * v_th / speed - v_r * v_th * dspeed / speed**2)

    # angular derivatives via spectral differentiation (exact on the grid)
    kth = np.fft.fftfreq(nth, d=1.0 / nth)  # integer wavenumbers
    dth_drt = np.real(np.fft.ifft(1j * kth * np.fft.fft(d_rt, axis=-1), axis=-1))
    dth_dtt = np.real(np.fft.ifft(1j * kth * np.fft.fft(d_tt, axis=-1), axis=-1))

    modes = np.arange(-n_modes, n_modes + 1)
    i_dm = 1j * modes  # i * diag(m) as a row to scale columns
    m2 = modes**2

    a_mat = -_toeplitz_from_theta(d_rr, n_modes)
    b_coeff = v_r - d_rr / rr - d_drr - dth_drt / rr
    b_mat = (
        _toeplitz_from_theta(b_coeff, n_modes) + _toeplitz_from_theta(-2.0 * d_rt / rr, n_modes) * i_dm[None, None, :]
    )
    s_coeff = v_th / rr - d_drt / rr - dth_dtt / rr**2
    s0_mat = (
        _toeplitz_from_theta(s_coeff, n_modes) * i_dm[None, None, :]
        + _toeplitz_from_theta(-d_tt / rr**2, n_modes) * (-m2)[None, None, :]
    )

    # _toeplitz_from_theta always returns (n_r, nm, nm) and r is atleast_1d, so the shape is uniform.
    return a_mat, b_mat, s0_mat


def _face_matrices(
    r_w: float, *, alpha_l: float, a0_signed: float, v_d: float, d_m: float, n_modes: int
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    r"""Well-face Toeplitz matrices ``M[v_r]``, ``M[D_rr]``, ``M[D_rtheta]`` at ``r = r_w``.

    These carry the exact ``theta``-modulated well-face physics: the drift adds ``v_d cos(theta)`` to the
    face velocity and an ``O(eps_w)`` cross-dispersion ``D_rtheta``, which couple neighbouring modes in
    the face boundary conditions (Robin flux inlet / Danckwerts) and in the injected-flux source. The
    block-diagonal (scalar) face treatment drops these couplings; that drop biases the extraction readout
    by ~``eps^2/2`` of the drift loss (measured against the corrected FV oracle and the exact streamtube
    decomposition), so the face conditions are built exactly here.

    Returns
    -------
    m_vr, m_drr, m_drt : ndarray of complex, each shape (2 n_modes + 1, 2 n_modes + 1)
        Azimuthal coupling matrices of the face velocity and dispersion-tensor components.
    """
    nth = 8 * n_modes + 48
    theta = np.arange(nth) * (2.0 * np.pi / nth)
    v_r = a0_signed / r_w + v_d * np.cos(theta)
    v_th = -v_d * np.sin(theta)
    speed = np.sqrt(v_r**2 + v_th**2)
    m_vr = _toeplitz_from_theta(v_r, n_modes)
    m_drr = _toeplitz_from_theta(alpha_l * v_r**2 / speed + d_m, n_modes)
    m_drt = _toeplitz_from_theta(alpha_l * v_r * v_th / speed, n_modes)
    return m_vr, m_drr, m_drt


def field_grid(
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geo: float,
    r_w: float,
    alpha_l: float,
    v_d: float,
    a0: float,
    n_quad: int,
    d_m: float = 0.0,
    drift_shift: float = 0.0,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float]:
    r"""Radial Gauss-Legendre grid and recessive-IC radius ``r_far`` for the drift block engine.

    The grid spans the advective plume front ``r_front = sqrt(r_w^2 + V_peak/c_geo)`` plus a margin of
    breakthrough widths (radial std ``~ sqrt(alpha_L r_front)``), the molecular reach
    ``~ sqrt(D_m * total_time)`` (as the scalar engine's grid), and the total rest-phase drift
    displacement ``drift_shift`` (the rest kernel translates the plume down-gradient; the grid must
    contain it). For drift, the recessive initial condition at ``r_far`` must stay inside the stagnation
    radius ``r_s = |A_0|/v_d`` (the matrix Riccati has a coordinate finite-escape as
    ``eps(r) = v_d r/A_0 -> 1``), so ``r_far`` is hard-capped at ``_RS_FRAC * r_s``. The envelope is
    enforced **honestly**: the *significant* plume (front + ``_PLUME_WIDTHS`` breakthrough widths + the
    rest displacement, where the resident field has fallen to ~1% of peak) must fit below that cap --
    otherwise a ``ValueError`` is raised rather than the plume being silently truncated. The quadrature
    spans the *significant* plume up to ``r_far`` (a generous ``_GRID_WIDTHS``-width field grid clipped at
    ``r_far``); only the negligible tail beyond ``r_far`` is dropped, which the recessive IC washes out
    anyway. At ``v_d = 0`` (``r_s = inf``) the cap is inactive and the grid is the scalar grid. ``a0``
    should be the **smallest** per-phase ``|A_0|`` (the most restrictive ``r_s``).

    Returns
    -------
    r_nodes : ndarray
        Radial nodes (m), shape ``(n_quad,)``.
    dr_weights : ndarray
        Gauss-Legendre weights in ``r``.
    r_far : float
        Outer radius (m) where the recessive (decaying) block initial condition is imposed.

    Raises
    ------
    ValueError
        If the significant plume reaches the stagnation radius (``r_front + _PLUME_WIDTHS * (width +
        reach_dm) + drift_shift + r_w > _RS_FRAC * r_s``, with ``width`` carrying the rest kernel's
        mechanical spread in quadrature and ``reach_dm = sqrt(D_m * total_time)``) -- the drift is too
        strong for the radial engine's contained-plume assumption.
    """
    net_volume = np.concatenate(([0.0], np.cumsum(flow * dt_days)))
    peak_volume = max(float(net_volume.max()), 0.0)
    total_time = float(np.sum(dt_days))
    r_front = np.sqrt(r_w**2 + peak_volume / c_geo)
    # radial breakthrough variance ~ alpha_L r_front (+ alpha_L^2), plus -- in quadrature -- the rest
    # kernel's own mechanical spread sigma_x^2 = 2 alpha_L v_d t_rest / R = 2 alpha_L drift_shift
    width = np.sqrt(alpha_l * r_front + alpha_l**2 + 2.0 * alpha_l * drift_shift)
    reach_dm = np.sqrt(d_m * total_time)
    r_significant = r_front + _PLUME_WIDTHS * (width + reach_dm) + drift_shift + r_w
    r_grid = r_front + _GRID_WIDTHS * width + 2.0 * _PLUME_WIDTHS * reach_dm + drift_shift + r_w
    r_s = abs(a0) / abs(v_d) if v_d != 0.0 else np.inf
    r_far_cap = _RS_FRAC * r_s
    if r_significant > r_far_cap:
        eps_front = abs(v_d) * r_front / abs(a0)
        msg = (
            f"drift too strong for the radial engine: the plume (front r_front={r_front:.2f} m, "
            f"eps_front={eps_front:.2f}) reaches the stagnation radius r_s={r_s:.2f} m; the slow-drift "
            "envelope requires the plume to stay well inside r_s = |A_0|/|v_d|"
        )
        raise ValueError(msg)
    # r_far >= r_significant (the cap was not exceeded and the washout floor is >= r_grid >= r_significant),
    # so clipping the generous grid at r_far drops only the negligible far tail, not the significant plume.
    r_far = min(max(_RFAR_FIELD_MULT * r_grid, r_grid + _RFAR_DECAY * alpha_l), r_far_cap)
    r_max = min(r_grid, r_far)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    r_nodes = 0.5 * (r_max - r_w) * (nodes + 1.0) + r_w
    dr_weights = 0.5 * (r_max - r_w) * weights
    return r_nodes, dr_weights, r_far


def _interval_transitions(
    l_dense: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.complexfloating]],
    r_from: npt.NDArray[np.floating],
    r_to: npt.NDArray[np.floating],
    n_s: int,
    nm: int,
) -> npt.NDArray[np.complexfloating]:
    r"""Fundamental transitions ``Psi(r_to[i], r_from[i])`` of ``Y' = L(r) Y`` for a batch of intervals.

    All intervals are integrated in lockstep on a common ``tau in [0, 1]`` clock (``r = r_from +
    tau (r_to - r_from)``, signed widths), with the log-derivative field ``L`` evaluated through its dense
    ODE interpolant, chunked to bound memory. Every interval spans adjacent quadrature radii, so each
    transition stays within a few e-foldings of unity -- which is what keeps the prefix/suffix
    Green's-function recursions built from them unconditionally well-conditioned, where a globally
    pivoted fundamental matrix accumulates the full mode-split Sturm-Liouville exponent across the grid
    and overflows / colinearizes.

    Returns
    -------
    ndarray of complex, shape (n_intervals, n_s, nm, nm)
        The per-interval transition matrices.

    Raises
    ------
    RuntimeError
        If an interval-transition integration fails.
    """
    n_int = r_from.size
    eye = np.eye(nm, dtype=complex)
    out = np.empty((n_int, n_s, nm, nm), dtype=complex)
    chunk = 64  # bounds the ODE state (and DOP853's stage copies) to ~chunk * n_s * nm^2 complex
    for lo in range(0, n_int, chunk):
        hi = min(lo + chunk, n_int)
        start, width = r_from[lo:hi], r_to[lo:hi] - r_from[lo:hi]
        nc = hi - lo

        def rhs(tau: float, y: npt.NDArray[np.floating], start=start, width=width, nc=nc) -> npt.NDArray[np.floating]:
            psi = y.view(complex).reshape(nc, n_s, nm, nm)
            ld = l_dense(start + tau * width)  # (nc, n_s, nm, nm)
            return (width[:, None, None, None] * (ld @ psi)).reshape(-1).view(float)

        sol = solve_ivp(
            rhs,
            [0.0, 1.0],
            np.tile(eye.reshape(-1), nc * n_s).view(float),
            rtol=_RICCATI_RTOL,
            atol=_RICCATI_ATOL,
            method="DOP853",
        )
        if not sol.success:
            msg = "block interval-transition integration failed"
            raise RuntimeError(msg)
        out[lo:hi] = np.ascontiguousarray(sol.y[:, -1]).view(complex).reshape(nc, n_s, nm, nm)
    return out


def _block_solutions(
    s: npt.NDArray[np.complexfloating],
    r_nodes: npt.NDArray[np.floating],
    r_w: float,
    *,
    alpha_l: float,
    a0: float,
    v_d: float,
    d_m: float,
    retardation_factor: float,
    n_modes: int,
    direction: str,
    r_far: float,
) -> dict[str, npt.NDArray[np.complexfloating]]:
    r"""Batched block log-derivative + per-interval transition solutions of one constant-Q phase.

    Integrates, for every Laplace node ``s`` at once (the coefficient matrices ``A, B, S0`` are
    ``s``-independent -- only the ``R s A^{-1}`` term carries ``s`` -- so one vectorized ODE pass covers
    all nodes), the matrix Riccati ``L' = -L^2 - A^{-1} B L - A^{-1}(S0 + R s I)`` (``L = c' c^{-1}``) on
    two branches:

    * **decaying** (recessive as ``r -> inf``): inward from ``r_far`` with a block-diagonal recessive IC
      (the scalar per-mode decaying log-derivative on every diagonal -- exact at ``v_d = 0`` and washed in
      by the inward attractor otherwise). Gives ``L_-`` at ``r_nodes`` and ``r_w``.
    * **regular** (well boundary): outward from ``r_w`` with the **exact block well IC** built from the
      face Toeplitz matrices (:func:`_face_matrices`): injection Robin
      ``L_+ = M[D_rr]^{-1}(M[v_r] - M[D_rt](i m)/r_w)``, extraction Danckwerts
      ``L_+ = -M[D_rr]^{-1} M[D_rt](i m)/r_w``. The ``O(eps_w)`` face couplings these carry look small
      but bias the drift loss at ``O(eps^2)`` (~15-20% of the loss) if dropped. Gives ``L_+``.

    The recessive / regular fundamental solutions enter the interior resolvent only through
    **per-interval transitions** ``Psi_-(r_i, r_{i-1})`` (outward hops) and ``Psi_+(r_{i-1}, r_i)``
    (inward hops), with ``r_{-1} = r_w``, integrated through the dense ``L`` interpolants
    (:func:`_interval_transitions`). Each hop spans a few e-foldings at most, so the prefix/suffix
    resolvent recursions (:func:`_resolvent_field_laplace`, :func:`_readout_laplace`,
    :func:`_resident_laplace`) stay bounded and well-conditioned at any Peclet, Laplace frequency, and
    azimuthal truncation. A single fundamental matrix pivoted at ``r_w`` -- even de-scaled by a scalar
    log-amplitude -- accumulates the full mode-split Sturm-Liouville exponent across the grid and
    overflows / colinearizes (observed as singular resolvent blocks for phases with small ``A_0`` or
    short durations), which is why the hops are the stored representation.

    ``a0`` is the unsigned flow scale ``|A_0|``; the phase ``direction`` sets the operator orientation.
    Retardation enters as the explicit ``R`` in the ``R s`` term (the ODE keeps the physical ``A_0``/``D_m``).

    Returns
    -------
    dict of ndarray
        ``Lm_w`` (``L_-`` at ``r_w``, shape ``(n_s, nm, nm)``), ``H`` (the Wronskian blocks
        ``[A (L_- - L_+)]^{-1}`` at ``r_nodes``, ``(n_quad, n_s, nm, nm)``) and ``Tm``/``Tp`` (recessive
        outward / regular inward per-interval transitions, ``(n_quad, n_s, nm, nm)``).

    Raises
    ------
    RuntimeError
        If a matrix Riccati or interval-transition integration does not succeed (the log-derivative hit
        a pole, typically the coordinate finite-escape near the stagnation radius).
    """
    s = np.asarray(s, dtype=complex).reshape(-1)
    n_s = s.size
    nm = 2 * n_modes + 1
    eye = np.eye(nm)
    sigma_a = 1.0 if direction == _INJECTION else -1.0
    a0_signed = sigma_a * abs(a0)
    r_max = float(np.max(r_nodes))

    def riccati_rhs(r: float, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        ld = y.view(complex).reshape(n_s, nm, nm)
        a_m, b_m, s0_m = block_coupling_matrices(
            np.array([r]), alpha_l=alpha_l, a0=a0_signed, v_d=v_d, d_m=d_m, n_modes=n_modes
        )
        a_inv = np.linalg.inv(a_m[0])
        d_ld = (
            -(ld @ ld)
            - ((a_inv @ b_m[0])[None] @ ld)
            - (a_inv @ s0_m[0])[None]
            - (retardation_factor * s)[:, None, None] * a_inv[None]
        )
        return d_ld.reshape(-1).view(float)

    # recessive IC at r_far (block-diagonal; scalar per-mode decaying log-derivative)
    if d_m == 0.0:
        beta = retardation_factor * s / (alpha_l * abs(a0))
        b13 = beta ** (1.0 / 3.0)
        zeta = b13 * r_far + beta ** (-2.0 / 3.0) / (4.0 * alpha_l * alpha_l)
        aie, aipe, _, _ = airye(zeta)
        l0 = 1.0 / (2.0 * alpha_l) + b13 * (aipe / aie)
    else:
        astar = alpha_l * abs(a0) / d_m
        kappa = np.sqrt(retardation_factor * s / d_m)
        a_coef = (1.0 - sigma_a * abs(a0) / d_m) / 2.0 - kappa * astar / 2.0
        l0 = -kappa - a_coef / (r_far + astar)
    lm0 = (l0[:, None, None] * eye[None]).astype(complex)
    sol_m = solve_ivp(
        riccati_rhs,
        [r_far, r_w],
        lm0.reshape(-1).view(float),
        rtol=_RICCATI_RTOL,
        atol=_RICCATI_ATOL,
        dense_output=True,
        method="DOP853",
    )
    # exact block well-face IC for the regular branch: the theta-modulated face velocity and the
    # D_rtheta cross-dispersion couple the modes in the face condition (dropping them, the old scalar
    # diagonal IC, biases the extraction readout by ~eps^2/2 of the drift loss):
    #   injection (Robin flux inlet, homogeneous part):  M[v_r] c - M[D_rr] c' - M[D_rt] (i m / r_w) c = 0
    #   extraction (Danckwerts, zero dispersive flux):   M[D_rr] c' + M[D_rt] (i m / r_w) c = 0
    m_vr, m_drr, m_drt = _face_matrices(r_w, alpha_l=alpha_l, a0_signed=a0_signed, v_d=v_d, d_m=d_m, n_modes=n_modes)
    i_dm_face = 1j * np.arange(-n_modes, n_modes + 1)
    cross = (m_drt * i_dm_face[None, :]) / r_w
    lp_w = np.linalg.solve(m_drr, (m_vr - cross) if sigma_a > 0 else -cross)
    lp0 = np.broadcast_to(lp_w, (n_s, nm, nm)).astype(complex)
    sol_p = solve_ivp(
        riccati_rhs,
        [r_w, r_max],
        lp0.reshape(-1).view(float),
        rtol=_RICCATI_RTOL,
        atol=_RICCATI_ATOL,
        dense_output=True,
        method="DOP853",
    )
    if not (sol_m.success and sol_p.success):
        msg = "block Riccati integration failed (the matrix log-derivative likely hit a pole near stagnation)"
        raise RuntimeError(msg)

    def l_minus_at(r: npt.NDArray[np.floating]) -> npt.NDArray[np.complexfloating]:
        return np.ascontiguousarray(sol_m.sol(r).T).view(complex).reshape(np.size(r), n_s, nm, nm)

    def l_plus_at(r: npt.NDArray[np.floating]) -> npt.NDArray[np.complexfloating]:
        return np.ascontiguousarray(sol_p.sol(r).T).view(complex).reshape(np.size(r), n_s, nm, nm)

    prev = np.concatenate(([r_w], r_nodes[:-1]))
    a_n = block_coupling_matrices(r_nodes, alpha_l=alpha_l, a0=a0_signed, v_d=v_d, d_m=d_m, n_modes=n_modes)[0]
    # The log-derivative branches enter the resolvent only through the Wronskian block
    # H(r') = [A(r')(L_-(r') - L_+(r'))]^{-1} -- bounded wherever the recessive and regular subspaces are
    # transverse -- so H is materialized once here (and cached with the phase) instead of its factors.
    h = np.linalg.inv(a_n[:, None] @ (l_minus_at(r_nodes) - l_plus_at(r_nodes)))
    return {
        "Lm_w": l_minus_at(np.array([r_w]))[0],
        "H": h,
        "Tm": _interval_transitions(l_minus_at, prev, r_nodes, n_s, nm),  # Psi_-(r_i, r_{i-1}), outward hops
        "Tp": _interval_transitions(l_plus_at, r_nodes, prev, n_s, nm),  # Psi_+(r_{i-1}, r_i), inward hops
    }


def _resident_laplace(
    d: dict[str, npt.NDArray[np.complexfloating]],
    *,
    alpha_l: float,
    d_m: float,
    r_w: float,
    a0: float,
    v_d: float,
    n_modes: int,
) -> npt.NDArray[np.complexfloating]:
    r"""Laplace-domain resident mode-field per unit uniform-``cin`` flux injection at the well.

    The injected water enters through the exact Kreft-Zuber flux boundary
    ``v_r c - D_rr d_r c - D_rt (1/r_w) d_th c = v_r cin``: the ``theta``-modulated face velocity gives
    the injected flux an ``O(eps_w)`` ``m = +-1`` modulation and the ``D_rtheta`` cross term couples the
    modes in the face operator, so with the recessive branch (``c' = L_-(r_w) c``) the face field is
    ``c(r_w) = F_w^{-1} M[v_r] e_0`` with ``F_w = M[v_r] - M[D_rr] L_-(r_w) - M[D_rt] (i m)/r_w``, carried
    outward by the stable hops ``c(r_i) = Psi_-(r_i, r_{i-1}) c(r_{i-1})``. Reduces to the scalar FR
    resident transfer ``E / f_w`` at ``v_d = 0``.

    Returns
    -------
    ndarray
        ``c(r_nodes; s)``, shape ``(n_s, n_quad, nm)``.
    """
    nm = 2 * n_modes + 1
    n_s = d["Lm_w"].shape[0]
    e0 = np.zeros(nm, dtype=complex)
    e0[n_modes] = 1.0
    m_vr, m_drr, m_drt = _face_matrices(r_w, alpha_l=alpha_l, a0_signed=abs(a0), v_d=v_d, d_m=d_m, n_modes=n_modes)
    cross = (m_drt * (1j * np.arange(-n_modes, n_modes + 1))[None, :]) / r_w
    fw = (m_vr - cross)[None] - m_drr[None] @ d["Lm_w"]  # (n_s, nm, nm)
    y = np.linalg.solve(fw, np.broadcast_to(m_vr @ e0, (n_s, nm))[..., None])  # (n_s, nm, 1)
    tm = d["Tm"]
    c = np.empty((tm.shape[0], n_s, nm), dtype=complex)
    for i in range(tm.shape[0]):
        y = tm[i] @ y
        c[i] = y[..., 0]
    return np.transpose(c, (1, 0, 2))


def _source_blocks(
    d: dict[str, npt.NDArray[np.complexfloating]],
    field: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.complexfloating]:
    r"""Wronskian-weighted source contributions ``H_j (R dr_j) field_j`` of the interior resolvent.

    ``H(r') = [A(r')(L_-(r') - L_+(r'))]^{-1}`` is the matrix Wronskian block of the interior Green's
    function, materialized with the per-phase solutions (:func:`_block_solutions`).

    Returns
    -------
    ndarray
        ``H_j source_j``, shape ``(n_quad, n_s, nm, k)`` for a ``(n_quad, nm, k)`` mode-field batch.
    """
    src = ((retardation_factor * dr_weights)[:, None, None] * field).astype(complex)  # (n_quad, nm, k)
    return d["H"] @ src[:, None]  # (n_quad, n_s, nm, k)


def _resolvent_field_laplace(
    d: dict[str, npt.NDArray[np.complexfloating]],
    field: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.complexfloating]:
    r"""Laplace-domain resident field after one constant-Q phase propagates ``field`` (all nodes).

    The interior Green's function of the constant-Q block operator, with recessive ``Y_-`` and well-regular
    ``Y_+`` and the matrix Wronskian ``H(r') = [A(r')(L_-(r') - L_+(r'))]^{-1}``, is
    ``Ghat(r, r') = Y_-(r) Y_-(r')^{-1} H(r')`` for ``r >= r'`` and ``Y_+(r) Y_+(r')^{-1} H(r')`` for
    ``r <= r'``. Applied to the source measure ``(R dr_j) field_j`` it separates into a recessive prefix
    (``j <= i``) and a regular suffix (``j > i``), both evaluated as first-order recursions over the
    per-interval transitions (``hs_j = H_j source_j``):

        P_0 = hs_0,        P_i = Psi_-(r_i, r_{i-1}) P_{i-1} + hs_i,
        S_{n-1} = 0,       S_{i-1} = Psi_+(r_{i-1}, r_i) (S_i + hs_i),
        F_i = P_i + S_i.

    Every factor is a short-interval transition (a few e-foldings) or a bounded Wronskian block, so the
    recursion is well-conditioned at any Peclet and Laplace frequency: distant contributions fade by
    repeated bounded multiplication exactly as the physical Green's function does, with no globally
    accumulated exponent to overflow or colinearize.

    Returns
    -------
    ndarray
        Propagated mode-field batch, shape ``(n_s, n_quad, nm, k)`` for a ``(n_quad, nm, k)`` ``field``.
    """
    hs = _source_blocks(d, field, dr_weights, retardation_factor)  # (n_quad, n_s, nm, k)
    tm, tp = d["Tm"], d["Tp"]
    n_quad = hs.shape[0]
    f = np.empty_like(hs)
    p = hs[0]
    f[0] = p
    for i in range(1, n_quad):
        p = tm[i] @ p + hs[i]
        f[i] = p
    s_acc = np.zeros_like(p)
    for i in range(n_quad - 1, 0, -1):
        s_acc = tp[i] @ (s_acc + hs[i])
        f[i - 1] += s_acc
    return np.transpose(f, (1, 0, 2, 3))  # (n_s, n_quad, nm, k)


def _readout_laplace(
    d: dict[str, npt.NDArray[np.complexfloating]],
    field: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    retardation_factor: float,
    n_modes: int,
    eps_w: float,
) -> npt.NDArray[np.complexfloating]:
    r"""Laplace-domain extracted flux concentration from a resident ``field`` under extraction.

    Under the exact Danckwerts boundary (zero dispersive flux across the face, cross term included in the
    regular-branch IC) the extracted flux concentration equals the resident concentration at the face
    pointwise in ``theta``, i.e. the extraction interior resolvent evaluated at ``r_w``. Since ``r_w``
    lies below every node, only the regular (suffix) branch contributes:
    ``c(r_w) = sum_j Y_+(r_w) Y_+(r_j)^{-1} H_j (R dr_j) field_j``, evaluated by running the suffix
    recursion of :func:`_resolvent_field_laplace` one extra inward hop to ``r_w``. The extracted mixture
    is the **inflow-flux-weighted** azimuthal average -- ``|v_r(r_w, theta)| = (|A_0|/r_w)(1 - eps_w
    cos(theta))`` with ``eps_w = v_d r_w / |A_0|`` -- applied explicitly on the face modes:

        cout_hat = c_0(r_w) - (eps_w / 2) (c_{+1}(r_w) + c_{-1}(r_w)).

    (The face flux weighting and the ``D_rtheta`` face coupling are ``O(eps_w)`` individually but bias the
    drift loss at ``O(eps^2)`` -- ~15-20% of the loss -- when dropped; measured against the corrected FV
    oracle and the exact streamtube decomposition.)

    Returns
    -------
    ndarray
        ``cout_hat(s)``, shape ``(n_s, k)`` for a ``(n_quad, nm, k)`` ``field``.
    """
    hs = _source_blocks(d, field, dr_weights, retardation_factor)  # (n_quad, n_s, nm, k)
    tp = d["Tp"]
    s_acc = np.zeros_like(hs[0])
    for i in range(hs.shape[0] - 1, 0, -1):
        s_acc = tp[i] @ (s_acc + hs[i])
    face = tp[0] @ (s_acc + hs[0])  # (n_s, nm, k): the resident face modes
    return face[:, n_modes, :] - 0.5 * eps_w * (face[:, n_modes + 1, :] + face[:, n_modes - 1, :])


def _rest_drift_field(
    field: npt.NDArray[np.floating],
    r_nodes: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    r_w: float,
    *,
    alpha_l: float,
    v_d: float,
    d_m: float,
    retardation_factor: float,
    t_rest: float,
    n_modes: int,
) -> npt.NDArray[np.floating]:
    r"""Advance the resident mode-field through a rest phase (``Q = 0``) under drift.

    With the well shut the velocity field is the bare uniform drift ``v = v_d x_hat``, so the
    advection-dispersion kernel is exact in free space and constant-coefficient: the field translates
    down-gradient by ``delta = v_d t / R`` and spreads by the anisotropic Gaussian with
    ``sigma_x^2 = 2 (alpha_L |v_d| + D_m) t / R`` along the drift and ``sigma_y^2 = 2 D_m t / R`` across
    it (rank-1 Scheidegger tensor, ``alpha_T = 0``). The kernel is applied in real space:

    1. the mode-field is evaluated at the Gauss-Hermite-shifted source points of every polar target node
       (barycentric interpolation on the radial Legendre nodes -- spectrally accurate for the smooth
       resident field -- times the azimuthal phase sum);
    2. Gauss-Hermite quadrature (``_REST_HERMITE`` nodes per Gaussian axis) averages the spread;
    3. the updated real-space samples are reprojected onto the modes by FFT over a uniform theta grid.

    The shut well is closed by a **radial Neumann image**: source points falling inside the well disk are
    folded back across the face (``r' -> 2 r_w - r'``), the leading-order zero-dispersive-flux closure,
    which conserves the near-well mass (a zeroed disk would swallow ``O(sigma r_w / R_b^2)`` of the plume).
    The residual is the circle-vs-line curvature of the image and the neglected ``O(r_w^2/r^2)`` dipole
    distortion of the drift around the cylinder -- at ``v_d = 0``, ``D_m > 0`` the kernel agrees with the
    scalar engine's exact well-respecting Bessel rest kernel to ~4e-4 in ``cout`` for a stored plume (the
    public API dispatches ``v_d = 0`` to the scalar path anyway). Source points outside the radial grid
    carry no mass by the grid's containment guarantee (:func:`field_grid` provisions for the rest
    displacement).

    An **honest truncation guard** protects the azimuthal representation: after reprojection, the energy
    in the harmonics just above the kept band (``M < |m| <= 2M``, available from the FFT grid) measures
    the translated field's spectral tail; if it exceeds ``_REST_TAIL_MAX`` of the field, the translated
    plume is too eccentric for the kept modes and a ``ValueError`` asks for a larger ``n_modes`` rather
    than silently folding the tail.

    Returns
    -------
    ndarray
        The advanced mode-field, shape ``(n_quad, nm, k)`` (matching ``field``).

    Raises
    ------
    ValueError
        If the translated field's azimuthal spectral tail exceeds the truncation guard (increase
        ``n_modes``).
    """
    delta = v_d * t_rest / retardation_factor
    sig_x = np.sqrt(2.0 * (alpha_l * abs(v_d) + d_m) * t_rest / retardation_factor)
    sig_y = np.sqrt(2.0 * d_m * t_rest / retardation_factor)
    if delta == 0.0 and sig_x == 0.0:
        return field  # v_d = 0 and D_m = 0: a rest phase is the identity
    n_quad, nm, k = field.shape
    modes = np.arange(-n_modes, n_modes + 1)
    nth = 8 * n_modes + 48
    theta = np.arange(nth) * (2.0 * np.pi / nth)
    x = r_nodes[:, None] * np.cos(theta)[None, :]  # (n_quad, nth)
    y = r_nodes[:, None] * np.sin(theta)[None, :]
    interp = BarycentricInterpolator(r_nodes, field.reshape(n_quad, nm * k))
    r_hi = float(r_nodes[-1])

    def field_at(xp: npt.NDArray[np.floating], yp: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        rp = np.hypot(xp, yp).ravel()
        thp = np.arctan2(yp, xp).ravel()
        # Source points inside the shut well are folded back across the face (radial Neumann image,
        # r' -> 2 r_w - r'): the leading-order zero-dispersive-flux closure at the well, which conserves
        # the near-well mass that a zeroed disk would silently swallow.
        rp = np.where(rp < r_w, 2.0 * r_w - rp, rp)
        inside = rp <= r_hi  # beyond the grid: no mass (containment guaranteed by field_grid)
        vals = np.zeros((rp.size, nm * k))
        vals[inside] = interp(rp[inside])
        phase = np.exp(1j * thp[:, None] * modes[None, :])  # (n_pts, nm)
        return np.einsum("pmk,pm->pk", vals.reshape(-1, nm, k), phase).real.reshape(*xp.shape, k)

    zh, wh = np.polynomial.hermite.hermgauss(_REST_HERMITE)
    f_new = np.zeros((n_quad, nth, k))
    if sig_y == 0.0:  # D_m = 0: the Gaussian spread is 1-D along the drift
        for za, wa in zip(zh, wh, strict=True):
            f_new += wa * field_at(x - delta - np.sqrt(2.0) * sig_x * za, y)
        f_new /= np.sqrt(np.pi)
    else:
        for za, wa in zip(zh, wh, strict=True):
            x_shift = x - delta - np.sqrt(2.0) * sig_x * za
            for zb, wb in zip(zh, wh, strict=True):
                f_new += (wa * wb) * field_at(x_shift, y - np.sqrt(2.0) * sig_y * zb)
        f_new /= np.pi
    coeffs = np.fft.fft(f_new, axis=1) / nth  # (n_quad, nth, k); c_m at index m mod nth
    measure = (r_nodes * dr_weights)[:, None, None]  # radial area measure for the spectral-tail energy
    tail_idx = np.concatenate([np.arange(n_modes + 1, 2 * n_modes + 1), -np.arange(n_modes + 1, 2 * n_modes + 1)])
    band_idx = np.concatenate([tail_idx, modes])
    # Per COLUMN, not aggregated: a batched build (the reverse operator's unit pulses) must not let one
    # column's excess tail hide in the energy of the others.
    e_tail = np.sum(measure * np.abs(coeffs[:, tail_idx % nth]) ** 2, axis=(0, 1))  # (k,)
    e_band = np.sum(measure * np.abs(coeffs[:, band_idx % nth]) ** 2, axis=(0, 1))
    ratios = np.sqrt(np.divide(e_tail, e_band, out=np.zeros_like(e_tail), where=e_band > 0.0))
    if np.any(ratios > _REST_TAIL_MAX):
        msg = (
            f"rest drift displacement (delta = {delta:.2f} m over {t_rest:.1f} d) makes the plume too "
            f"eccentric for the kept azimuthal modes (spectral tail {float(ratios.max()):.2%} > "
            f"{_REST_TAIL_MAX:.0%}); increase n_modes"
        )
        raise ValueError(msg)
    return coeffs[:, modes % nth].real


def block_cout_deviation(
    *,
    cin_deviation: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geo: float,
    r_w: float,
    alpha_l: float,
    v_d: float,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    n_modes: int = 3,
    n_quad: int = 240,
    n_terms: int = 44,
    tol: float = 1e-9,
) -> npt.NDArray[np.floating]:
    r"""Multi-cycle extracted-flux deviation with steady regional drift, via the azimuthal-mode block engine.

    Generalizes the scalar reused-propagator engine (:func:`gwtransport._radial_asr_reuse.cout_deviation`)
    to the coupled azimuthal modes. The resident state is a mode-field batch ``field[r_node, mode, column]``
    (one column per independent ``cin_deviation`` column); each constant-Q phase advances it with the exact
    per-phase block kernels (:func:`_block_solutions`):

    * **injection** -- the existing field is propagated by the injection-direction interior resolvent
      (:func:`_resolvent_field_laplace`), then the freshly injected resident profile is added via the
      ``m = 0`` Kreft-Zuber flux transfer (:func:`_resident_laplace`) superposed over the injection bins;
    * **extraction** -- the ``m = 0`` well-face flux concentration is read out (:func:`_readout_laplace`)
      and bin-averaged into ``cout``; the residual field is then propagated for any following phase;
    * **rest** (``flow == 0``) -- the field is advanced by the exact free-space drift kernel
      (:func:`_rest_drift_field`): translation by ``v_d t/R`` plus the anisotropic Gaussian spread, with
      a Neumann-image closure at the shut well face and an honest guard on the azimuthal truncation of
      the translated plume.

    Every per-phase operator is grid-free in ``(r, theta)`` (no PDE mesh): the only numerics are the radial
    matrix Riccati ODEs, Gauss-Legendre quadrature and de Hoog Laplace inversion. The interior resolvent is
    applied per reversal (prefix/suffix recursions over the per-interval transitions) rather than as a
    materialized propagator, so memory stays ``O((2M+1)^2 n_quad)`` per phase kernel, and the per-phase
    kernel solutions are cached across recurring phases -- the block analogue of the scalar engine's reused
    propagator matrices, making the ODE cost ``O(distinct phases)`` instead of ``O(reversals)``. At
    ``v_d = 0`` the modes decouple and the ``m = 0`` block reproduces the scalar engine to the de Hoog
    floor *for constant-per-phase flow* (the public API dispatches ``v_d = 0`` to the scalar path for the
    bit-for-bit guarantee).

    Within-phase variable flow is **approximate**: each phase is clocked in wall-clock time at its mean
    magnitude ``a0 = mean(|flow[phase]|)`` (the drift breaks the flushed-volume-clock autonomy the scalar
    ``D_m = 0`` path exploits for exact variable flow, and matches the scalar ``D_m > 0`` path's same mean-
    flow approximation). It is exact for piecewise-constant flow. At ``v_d = 0`` it is additionally exact
    for constant ``cin`` over a phase at any flow profile (the resident profile then depends only on the
    total injected volume), leaving only the *variable cin AND variable flow* cin-placement error (bins at
    time corners rather than exact volume edges). Under drift no such exactness survives for any
    within-phase flow variation: the mode coupling integrates ``eps(r(t))`` on the wall clock, so two flow
    profiles with equal volume and duration end in different fields (an FV differencing of a constant-cin
    ``+-60%`` flow ramp against constant flow shifts the drift recovery loss by ~12% of the loss). The
    error grows with the within-phase variation; use finer phases if needed.

    Parameters
    ----------
    cin_deviation : ndarray, shape (n,) or (n, k)
        Injected concentration deviation per bin (used on injection bins, ``flow > 0``). A 2-D input
        transports ``k`` independent deviation columns through one engine pass sharing the per-phase
        kernels (used to build the reverse operator's column block in a single run).
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
    v_d : float
        Regional drift seepage velocity ``U / n`` [m/day]. ``0`` is the radial-symmetric limit (the modes
        decouple); the public API dispatches that to the scalar engine, but this function handles it too.
    molecular_diffusivity : float, optional
        Molecular diffusivity [m^2/day]. Default 0.
    retardation_factor : float, optional
        Linear retardation ``R >= 1``. Default 1.
    n_modes : int, optional
        Azimuthal truncation ``M`` (keeps modes ``-M .. M``). Default 3.
    n_quad : int, optional
        Number of radial Gauss-Legendre nodes. Default 240.
    n_terms : int, optional
        de Hoog series length. Default 44.
    tol : float, optional
        de Hoog target accuracy. Default ``1e-9``.

    Returns
    -------
    ndarray, shape (n,) or (n, k)
        Extracted-flux deviation per bin (matching ``cin_deviation``); ``NaN`` on injection / rest bins.

    Notes
    -----
    Propagates a ``ValueError`` from :func:`field_grid` when the significant plume (including the
    rest-phase drift displacement) reaches the stagnation radius (drift too strong for the radial
    engine), or from :func:`_rest_drift_field` when a rest translation makes the plume too eccentric for
    the kept azimuthal modes (increase ``n_modes``); and a ``RuntimeError`` from :func:`_block_solutions`
    if a per-phase matrix Riccati or interval-transition integration fails.
    """
    flow = np.asarray(flow, dtype=float)
    dt_days = np.asarray(dt_days, dtype=float)
    cin_deviation = np.asarray(cin_deviation, dtype=float)
    vector_input = cin_deviation.ndim == 1
    cin_cols = cin_deviation[:, None] if vector_input else cin_deviation  # (n, k)
    n_rhs = cin_cols.shape[1]
    phases = _phase_slices(flow)
    pumping = [(sign, sl) for sign, sl in phases if sign != 0]
    if not pumping:  # all-rest schedule: nothing is injected or extracted
        cout_empty = np.full((len(flow), n_rhs), np.nan)
        return cout_empty[:, 0] if vector_input else cout_empty
    while phases[-1][0] == 0:  # trailing rest phases cannot affect any output; don't propagate or guard them
        phases.pop()
    # Each pumping phase is clocked at its mean magnitude, so its A_0 = mean(|flow[phase]|)/(2 c_geo); the
    # stagnation radius r_s = |A_0|/|v_d| is smallest for the weakest phase, so size the grid cap on that
    # (worst-case). Interior rest phases translate the plume by v_d t/R; the grid provisions for their
    # total shift. Leading rests act on an empty field and trailing rests are dropped above, so neither
    # counts -- idle padding must not inflate the envelope guard or dilute the radial resolution.
    a0_min = min(float(np.mean(np.abs(flow[sl]))) for _, sl in pumping) / (2.0 * c_geo)
    nz = np.flatnonzero(flow != 0.0)
    interior = slice(nz[0], nz[-1] + 1)
    rest_time = float(np.sum(dt_days[interior][flow[interior] == 0.0]))
    drift_shift = abs(v_d) * rest_time / retardation_factor
    r_nodes, dr_weights, r_far = field_grid(
        flow, dt_days, c_geo, r_w, alpha_l, v_d, a0_min, n_quad, d_m=molecular_diffusivity, drift_shift=drift_shift
    )
    nm = 2 * n_modes + 1

    # The per-phase block solutions are a pure function of (direction, |A_0|, s) -- every other input is
    # fixed for the call -- and the de Hoog nodes depend only on max(t), so the 2-3 inversions within one
    # phase (propagate / resident / readout all span the same phase duration) and every recurrence of the
    # phase across a periodic schedule share one Riccati + transition solve. Caching them is the block
    # analogue of the scalar engine's reused propagator matrices: the ODE cost becomes O(distinct phases)
    # instead of O(reversals). FIFO-capped: an entry holds O(n_quad n_s (2M+1)^2) complex (~70 MB at the
    # defaults), and a periodic schedule needs only one entry per pumping direction.
    solutions_cache: dict[tuple[str, float, bytes], dict] = {}

    def solutions(s: npt.NDArray[np.complexfloating], a0: float, direction: str) -> dict:
        key = (direction, a0, s.tobytes())
        if key not in solutions_cache:
            if len(solutions_cache) >= _SOLUTIONS_CACHE_MAX:
                solutions_cache.pop(next(iter(solutions_cache)))
            solutions_cache[key] = _block_solutions(
                s,
                r_nodes,
                r_w,
                alpha_l=alpha_l,
                a0=a0,
                v_d=v_d,
                d_m=molecular_diffusivity,
                retardation_factor=retardation_factor,
                n_modes=n_modes,
                direction=direction,
                r_far=r_far,
            )
        return solutions_cache[key]

    def propagate(field: npt.NDArray[np.floating], a0: float, direction: str, t_phase: float) -> npt.NDArray:
        def f_hat(s):
            return _resolvent_field_laplace(solutions(s, a0, direction), field, dr_weights, retardation_factor)

        return dehoog_inverse(f_hat=f_hat, t=t_phase, n_terms=n_terms, tol=tol)

    field = np.zeros((n_quad, nm, n_rhs))
    cout = np.full((len(flow), n_rhs), np.nan)
    for idx, (sign, sl) in enumerate(phases):
        # One cumulative-time base per phase: the propagate / resident / readout inversions then share
        # bitwise-identical de Hoog nodes (max(t) equal), so the solutions cache hits within the phase.
        csum = np.cumsum(dt_days[sl])
        t_phase = float(csum[-1])
        if sign == 0:  # rest: free-space translate + anisotropic-spread kernel (Neumann image at the well)
            if np.any(field):
                field = _rest_drift_field(
                    field,
                    r_nodes,
                    dr_weights,
                    r_w,
                    alpha_l=alpha_l,
                    v_d=v_d,
                    d_m=molecular_diffusivity,
                    retardation_factor=retardation_factor,
                    t_rest=t_phase,
                    n_modes=n_modes,
                )
            continue
        a0 = float(np.mean(np.abs(flow[sl]))) / (2.0 * c_geo)
        if sign > 0:  # injection: propagate the buffer, then add the freshly injected resident profile
            if np.any(field):
                field = propagate(field, a0, _INJECTION, t_phase)
            corners = t_phase - np.concatenate(([0.0], csum))  # descending, last is 0

            def f_hat_resident(s, a0=a0):
                return (
                    _resident_laplace(
                        solutions(s, a0, _INJECTION),
                        alpha_l=alpha_l,
                        d_m=molecular_diffusivity,
                        r_w=r_w,
                        a0=a0,
                        v_d=v_d,
                        n_modes=n_modes,
                    )
                    / s[:, None, None]
                )

            g1 = dehoog_inverse(f_hat=f_hat_resident, t=corners, n_terms=n_terms, tol=tol)  # (n_corner, n_quad, nm)
            field += np.einsum("bk,bqm->qmk", cin_cols[sl], g1[:-1] - g1[1:])
        else:  # extraction: read the m=0 well-face flux, then propagate the residual if more phases follow
            ext_corners = np.concatenate(([0.0], csum))

            def f_hat_readout(s, a0=a0, field=field):
                return (
                    _readout_laplace(
                        solutions(s, a0, "extraction"),
                        field,
                        dr_weights,
                        retardation_factor,
                        n_modes,
                        eps_w=v_d * r_w / a0,
                    )
                    / s[:, None]
                )

            cdf = dehoog_inverse(f_hat=f_hat_readout, t=ext_corners, n_terms=n_terms, tol=tol)  # (n_corner, k)
            cout[sl] = (cdf[1:] - cdf[:-1]) / np.diff(ext_corners)[:, None]
            if idx != len(phases) - 1:
                field = propagate(field, a0, "extraction", t_phase)
    return cout[:, 0] if vector_input else cout
