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

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.special import airye

from gwtransport._radial_asr_dehoog import dehoog_inverse

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
        Azimuthal FFT grid size (defaults to ``8 n_modes + 8``, comfortably unaliased).

    Returns
    -------
    a_mat, b_mat, s0_mat : ndarray of complex, each shape (n_r, 2 n_modes + 1, 2 n_modes + 1)
        The coefficient matrices at each radius.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    nm = 2 * n_modes + 1
    nth = n_theta if n_theta is not None else 8 * n_modes + 8
    theta = np.arange(nth) * (2.0 * np.pi / nth)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    rr = r[:, None]  # (n_r, 1) broadcast against theta
    v_r = a0 / rr + v_d * cos_t[None, :]  # (n_r, nth)
    v_th = -v_d * sin_t[None, :] * np.ones_like(rr)
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

    if r.shape[0] == 1:
        return a_mat, b_mat, s0_mat
    return a_mat.reshape(-1, nm, nm), b_mat.reshape(-1, nm, nm), s0_mat.reshape(-1, nm, nm)


def field_grid(
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geo: float,
    r_w: float,
    alpha_l: float,
    v_d: float,
    a0: float,
    n_quad: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float]:
    r"""Radial Gauss-Legendre grid and recessive-IC radius ``r_far`` for the drift block engine.

    The grid spans the advective plume front ``r_front = sqrt(r_w^2 + V_peak/c_geo)`` plus a margin of
    breakthrough widths (radial std ``~ sqrt(alpha_L r_front)``), exactly as the scalar engine. For drift,
    the recessive initial condition at ``r_far`` must stay inside the stagnation radius ``r_s = |A_0|/v_d``
    (the matrix Riccati has a coordinate finite-escape as ``eps(r) = v_d r/A_0 -> 1``), so ``r_far`` is
    hard-capped at ``_RS_FRAC * r_s``. The envelope is enforced **honestly**: the *significant* plume
    (front + ``_PLUME_WIDTHS`` breakthrough widths, where the resident field has fallen to ~1% of peak) must
    fit below that cap -- otherwise a ``ValueError`` is raised rather than the plume being silently
    truncated. The quadrature spans the *significant* plume up to ``r_far`` (a generous ``_GRID_WIDTHS``-width
    field grid clipped at ``r_far``); only the negligible tail beyond ``r_far`` is dropped, which the
    recessive IC washes out anyway. At ``v_d = 0`` (``r_s = inf``) the cap is inactive and the grid is the
    scalar grid. ``a0`` should be the **smallest** per-phase ``|A_0|`` (the most restrictive ``r_s``).

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
        If the significant plume reaches the stagnation radius (``r_front + _PLUME_WIDTHS * width + r_w >
        _RS_FRAC * r_s``) -- the drift is too strong for the radial engine's contained-plume assumption.
    """
    net_volume = np.concatenate(([0.0], np.cumsum(flow * dt_days)))
    peak_volume = max(float(net_volume.max()), 0.0)
    r_front = np.sqrt(r_w**2 + peak_volume / c_geo)
    width = np.sqrt(alpha_l * r_front + alpha_l**2)
    r_significant = r_front + _PLUME_WIDTHS * width + r_w
    r_grid = r_front + _GRID_WIDTHS * width + r_w
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
    r"""Batched block log-derivative + de-scaled transition solutions of one constant-Q phase.

    Integrates, for every Laplace node ``s`` at once (the coefficient matrices ``A, B, S0`` are
    ``s``-independent -- only the ``R s A^{-1}`` term carries ``s`` -- so one vectorized ODE pass covers
    all nodes), the matrix Riccati ``L' = -L^2 - A^{-1} B L - A^{-1}(S0 + R s I)`` (``L = c' c^{-1}``) on
    two branches and the de-scaled fundamental transitions used to assemble the interior resolvent:

    * **decaying** (recessive as ``r -> inf``): inward from ``r_far`` with a block-diagonal recessive IC
      (the scalar per-mode decaying log-derivative on every diagonal -- exact at ``v_d = 0`` and washed in
      by the inward attractor otherwise). Gives ``L_-`` at ``r_nodes`` and ``r_w``.
    * **regular** (well boundary): outward from ``r_w`` with the block-diagonal well IC
      (``L_+ = A_0/(alpha_L A_0 + D_m r_w) I`` injection Robin / ``0`` extraction Neumann). Gives ``L_+``.

    The recessive / regular fundamental matrices ``Y_-`` , ``Y_+`` (both normalized to ``I`` at ``r_w``) are
    carried as **de-scaled transitions** ``Psi_hat`` with a scalar log-amplitude
    ``phi = int_{r_w}^r L[m0,m0]`` factored out (``Y = Psi_hat e^{phi}``); this keeps ``Psi_hat`` ``O(1)``
    while the divergent Sturm-Liouville exponent lives in ``phi`` (the matrix analogue of the scalar
    kernel's bounded log-difference), so the resolvent assembly never overflows at high Peclet. The
    recessive ``L_-`` is integrated inward from ``r_far`` (its stable direction); its transition is then
    propagated outward from ``r_w`` through the dense ``L_-`` interpolant.

    ``a0`` is the unsigned flow scale ``|A_0|``; the phase ``direction`` sets the operator orientation.
    Retardation enters as the explicit ``R`` in the ``R s`` term (the ODE keeps the physical ``A_0``/``D_m``).

    Returns
    -------
    dict of ndarray
        ``Lm_w`` (``L_-`` at ``r_w``, shape ``(n_s, nm, nm)``), ``Lm_n``/``Lp_n`` (``L_-``/``L_+`` at
        ``r_nodes``, ``(n_quad, n_s, nm, nm)``), ``Psm``/``Psp`` (de-scaled recessive/regular transitions at
        ``r_nodes``), ``phim``/``phip`` (log-amplitudes, ``(n_quad, n_s)``) and ``A_n`` (the ``s``-independent
        principal part at ``r_nodes``, ``(n_quad, nm, nm)``).

    Raises
    ------
    RuntimeError
        If a matrix Riccati integration does not reach every requested radius (the log-derivative hit a
        pole, typically the coordinate finite-escape near the stagnation radius).
    """
    s = np.asarray(s, dtype=complex).reshape(-1)
    n_s = s.size
    nm = 2 * n_modes + 1
    m0 = n_modes
    eye = np.eye(nm)
    sigma_a = 1.0 if direction == _INJECTION else -1.0
    a0_signed = sigma_a * abs(a0)
    r_max = float(np.max(r_nodes))

    def coupling(r: float) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        a_m, b_m, s0_m = block_coupling_matrices(
            np.array([r]), alpha_l=alpha_l, a0=a0_signed, v_d=v_d, d_m=d_m, n_modes=n_modes
        )
        return a_m[0], b_m[0], s0_m[0]

    n_block = n_s * nm * nm

    def riccati_rhs(r: float, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        ld = y.view(complex).reshape(n_s, nm, nm)
        a_m, b_m, s0_m = coupling(r)
        a_inv = np.linalg.inv(a_m)
        d_ld = (
            -(ld @ ld)
            - ((a_inv @ b_m)[None] @ ld)
            - (a_inv @ s0_m)[None]
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
    lp_w = a0_signed / (alpha_l * abs(a0) + d_m * r_w) if sigma_a > 0 else 0.0
    lp0 = (lp_w * eye[None] * np.ones((n_s, 1, 1))).astype(complex)
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

    def l_minus(r: float) -> npt.NDArray[np.complexfloating]:
        return sol_m.sol(r).view(complex).reshape(n_s, nm, nm)

    def l_plus(r: float) -> npt.NDArray[np.complexfloating]:
        return sol_p.sol(r).view(complex).reshape(n_s, nm, nm)

    # De-scaled transitions Psi_hat (Psi_hat' = (L - L[m0,m0] I) Psi_hat) with the log-amplitude
    # phi = int_{r_w}^r L[m0,m0] factored out. Both branches are pivoted at r_w (integrated outward from
    # r_w), which keeps phi small over [r_w, r_max] so the assembly's exp(+-phi) stay bounded -- the
    # recessive L is still integrated inward (its stable direction); only its de-scaled transition is
    # propagated outward through the dense L interpolant.
    def transition_rhs(l_fun):
        def rhs(r: float, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
            yc = y.view(complex)
            psih = yc[:n_block].reshape(n_s, nm, nm)
            ld = l_fun(r)
            ell = ld[:, m0, m0]
            d_psih = (ld - ell[:, None, None] * eye[None]) @ psih
            return np.concatenate([d_psih.reshape(-1), ell]).view(float)

        return rhs

    y0 = np.concatenate([np.tile(eye.reshape(-1), n_s).astype(complex), np.zeros(n_s, complex)]).view(float)
    tr_m = solve_ivp(
        transition_rhs(l_minus),
        [r_w, r_max],
        y0,
        t_eval=r_nodes,
        rtol=_RICCATI_RTOL,
        atol=_RICCATI_ATOL,
        method="DOP853",
    )
    tr_p = solve_ivp(
        transition_rhs(l_plus),
        [r_w, r_max],
        y0,
        t_eval=r_nodes,
        rtol=_RICCATI_RTOL,
        atol=_RICCATI_ATOL,
        method="DOP853",
    )
    if not (tr_m.success and tr_p.success):
        msg = "block transition integration failed"
        raise RuntimeError(msg)

    def transitions(tr) -> tuple[npt.NDArray, npt.NDArray]:
        yc = np.ascontiguousarray(tr.y.T).view(complex)  # (n_quad, n_block + n_s)
        return yc[:, :n_block].reshape(-1, n_s, nm, nm), yc[:, n_block:].reshape(-1, n_s)

    psm, phim = transitions(tr_m)
    psp, phip = transitions(tr_p)
    lm_n = np.stack([l_minus(float(r)) for r in r_nodes])
    lp_n = np.stack([l_plus(float(r)) for r in r_nodes])
    a_n = np.stack([coupling(float(r))[0] for r in r_nodes])
    return {
        "Lm_w": l_minus(r_w),
        "Lm_n": lm_n,
        "Lp_n": lp_n,
        "Psm": psm,
        "Psp": psp,
        "phim": phim,
        "phip": phip,
        "A_n": a_n,
    }


def _resident_laplace(
    d: dict[str, npt.NDArray[np.complexfloating]], *, alpha_l: float, d_m: float, r_w: float, a0: float, n_modes: int
) -> npt.NDArray[np.complexfloating]:
    r"""Laplace-domain resident mode-field per unit ``m = 0`` flux injection at the well.

    The injected water enters through the azimuthally symmetric (``m = 0``) Kreft-Zuber flux boundary;
    the recessive (decaying) block solution carries it outward and drift couples it into the neighbouring
    modes. With the de-scaled recessive transition ``Psi_hat_-`` and the block well-flux operator
    ``F_w = I - (alpha_L + D_m r_w/|A_0|) L_-(r_w)`` the resident field at the nodes is
    ``c(r) = Psi_hat_-(r) e^{phi_-(r)} F_w^{-1} e_0`` (``e_0`` the unit ``m = 0`` source). Reduces to the
    scalar FR resident transfer ``E / f_w`` at ``v_d = 0``.

    Returns
    -------
    ndarray
        ``c(r_nodes; s)``, shape ``(n_s, n_quad, nm)``.
    """
    nm = 2 * n_modes + 1
    n_s = d["Lm_w"].shape[0]
    e0 = np.zeros(nm, dtype=complex)
    e0[n_modes] = 1.0
    fw = np.eye(nm)[None] - (alpha_l + d_m * r_w / abs(a0)) * d["Lm_w"]  # (n_s, nm, nm)
    gamma = np.linalg.solve(fw, np.broadcast_to(e0, (n_s, nm))[..., None])[..., 0]  # (n_s, nm)
    return np.einsum("qsab,sb->sqa", d["Psm"], gamma) * np.exp(d["phim"]).T[:, :, None]


def _resolvent_prefix_suffix(
    d: dict[str, npt.NDArray[np.complexfloating]], source: npt.NDArray[np.complexfloating]
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""Per-node de-scaled left factors and prefix/suffix accumulations of the interior block resolvent.

    The interior Green's function of the constant-Q block operator is, with recessive ``Y_-`` and
    well-regular ``Y_+`` and the matrix Wronskian ``H(r') = [A(r')(L_-(r') - L_+(r'))]^{-1}``,

        Ghat(r, r') = Y_-(r) Y_-(r')^{-1} H(r')   (r >= r'),   Y_+(r) Y_+(r')^{-1} H(r')   (r <= r'),

    so applying it to a source measure ``source_j = (R dr_j) f_j`` separates into a prefix (``j <= i``,
    recessive branch) and a suffix (``j > i``, regular branch):

        F_i = Psi_hat_-(r_i) e^{phi_-(r_i)} sum_{j<=i} Psi_hat_-(r_j)^{-1} H_j e^{-phi_-(r_j)} source_j
            + Psi_hat_+(r_i) e^{phi_+(r_i)} sum_{j>i}  Psi_hat_+(r_j)^{-1} H_j e^{-phi_+(r_j)} source_j .

    The ``e^{+-phi}`` factors are paired so each ``Ghat`` block is bounded (the recessive branch differences
    ``phi_-(r_i) - phi_-(r_j) <= 0`` for ``j <= i``), the matrix analogue of the scalar resolvent's bounded
    log-difference. Returns the building blocks so the caller forms ``F`` for the field hand-off (all nodes)
    or the ``r_w`` well-face trace (suffix only).

    Returns
    -------
    left_m, left_p : ndarray
        De-scaled left factors ``Psi_hat_-(r_i) e^{phi_-(r_i)}`` / ``Psi_hat_+(r_i) e^{phi_+(r_i)}``,
        each ``(n_quad, n_s, nm, nm)``.
    prefix, suffix : ndarray
        Inclusive prefix ``sum_{j<=i}`` and exclusive suffix ``sum_{j>i}`` of the de-scaled source
        contributions, each ``(n_quad, n_s, nm)``.
    """
    term_m, term_p = _resolvent_terms(d, source)
    prefix = np.cumsum(term_m, axis=0)  # sum_{j<=i}
    suffix = np.cumsum(term_p[::-1], axis=0)[::-1] - term_p  # sum_{j>i}
    left_m = d["Psm"] * np.exp(d["phim"])[..., None, None]
    left_p = d["Psp"] * np.exp(d["phip"])[..., None, None]
    return left_m, left_p, prefix, suffix


def _resolvent_terms(
    d: dict[str, npt.NDArray[np.complexfloating]], source: npt.NDArray[np.complexfloating]
) -> tuple[npt.NDArray, npt.NDArray]:
    r"""De-scaled per-node source contributions ``Psi_hat_+-(r_j)^{-1} H_j e^{-phi_+-(r_j)} source_j``.

    Returns
    -------
    term_m, term_p : ndarray
        Recessive- and regular-branch de-scaled contributions, each ``(n_quad, n_s, nm)``.
    """
    h = np.linalg.inv(d["A_n"][:, None] @ (d["Lm_n"] - d["Lp_n"]))  # (n_quad, n_s, nm, nm)
    hg = np.einsum("qsab,qb->qsa", h, source)  # (n_quad, n_s, nm)
    term_m = np.linalg.solve(d["Psm"], hg[..., None])[..., 0] * np.exp(-d["phim"])[..., None]
    term_p = np.linalg.solve(d["Psp"], hg[..., None])[..., 0] * np.exp(-d["phip"])[..., None]
    return term_m, term_p


def _resolvent_field_laplace(
    d: dict[str, npt.NDArray[np.complexfloating]],
    field: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.complexfloating]:
    r"""Laplace-domain resident field after one constant-Q phase propagates ``field`` (all nodes).

    ``F_i = Psi_hat_-(r_i) e^{phi_-(r_i)} prefix_i + Psi_hat_+(r_i) e^{phi_+(r_i)} suffix_i`` -- the interior
    block resolvent applied to the resident profile with source measure ``(R dr_j) field_j``.

    Returns
    -------
    ndarray
        Propagated field ``(n_s, n_quad, nm)``.
    """
    source = (retardation_factor * dr_weights)[:, None] * field  # (n_quad, nm)
    left_m, left_p, prefix, suffix = _resolvent_prefix_suffix(d, source.astype(complex))
    f = np.einsum("qsab,qsb->qsa", left_m, prefix) + np.einsum("qsab,qsb->qsa", left_p, suffix)
    return np.transpose(f, (1, 0, 2))  # (n_s, n_quad, nm)


def _readout_laplace(
    d: dict[str, npt.NDArray[np.complexfloating]],
    field: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    retardation_factor: float,
    n_modes: int,
) -> npt.NDArray[np.complexfloating]:
    r"""Laplace-domain ``m = 0`` well-face flux concentration from a resident ``field`` under extraction.

    The extracted Kreft-Zuber flux concentration under the Danckwerts (zero dispersive flux) well boundary
    equals the resident concentration at the face, i.e. the extraction interior resolvent evaluated at
    ``r_w``. Since ``r_w`` lies below every node, only the regular (suffix) branch contributes and
    ``Psi_hat_+(r_w) = I`` , ``phi_+(r_w) = 0``, so the trace is
    ``cout_hat(s) = [sum_k Psi_hat_+(r_k)^{-1} H_k e^{-phi_+(r_k)} (R dr_k) field_k]_{m0}``.

    Returns
    -------
    ndarray
        ``cout_hat(s)`` (the ``m = 0`` component), shape ``(n_s,)``.
    """
    source = (retardation_factor * dr_weights)[:, None] * field
    # r_w lies below every node, so the well-face trace is the full regular-branch sum (Psi_hat_+(r_w)=I).
    _, term_p = _resolvent_terms(d, source.astype(complex))
    return term_p.sum(axis=0)[:, n_modes]


def _phase_slices(flow: npt.NDArray[np.floating]) -> list[tuple[int, slice]]:
    """Group a signed schedule into maximal one-signed phases ``(sign in {+1,-1,0}, slice)``.

    Returns
    -------
    list of (int, slice)
        Per-phase sign and contiguous bin slice.
    """
    signs = np.sign(flow).astype(int)
    edges = np.flatnonzero(np.diff(signs) != 0) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(flow)]))
    return [(int(signs[a]), slice(a, b)) for a, b in zip(starts, stops, strict=True)]


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
    to the coupled azimuthal modes. The resident state is a mode-field ``field[r_node, mode]``; each
    constant-Q phase advances it with the exact per-phase block kernels (:func:`_block_solutions`):

    * **injection** -- the existing field is propagated by the injection-direction interior resolvent
      (:func:`_resolvent_field_laplace`), then the freshly injected resident profile is added via the
      ``m = 0`` Kreft-Zuber flux transfer (:func:`_resident_laplace`) superposed over the injection bins;
    * **extraction** -- the ``m = 0`` well-face flux concentration is read out (:func:`_readout_laplace`)
      and bin-averaged into ``cout``; the residual field is then propagated for any following phase.

    Every per-phase operator is grid-free in ``(r, theta)`` (no PDE mesh): the only numerics are the radial
    matrix Riccati ODEs, Gauss-Legendre quadrature and de Hoog Laplace inversion. The interior resolvent is
    applied per reversal (prefix/suffix accumulation) rather than as a materialized propagator, so memory
    stays ``O((2M+1)^2 n_quad)``. At ``v_d = 0`` the modes decouple and the ``m = 0`` block reproduces the
    scalar engine to the de Hoog floor (the public API dispatches ``v_d = 0`` to the scalar path for the
    bit-for-bit guarantee).

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
    v_d : float
        Regional drift seepage velocity ``U / n`` [m/day], ``!= 0``.
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
    ndarray, shape (n,)
        Extracted-flux deviation per bin; ``NaN`` on injection / rest bins.

    Raises
    ------
    NotImplementedError
        If the schedule contains a rest phase (``flow == 0``) together with nonzero drift -- the
        rest-with-drift propagator (translation + anisotropic Gaussian spreading) is not yet implemented;
        the field would otherwise be silently frozen across the rest, which is wrong under drift.

    Notes
    -----
    Propagates a ``ValueError`` from :func:`field_grid` when the significant plume reaches the stagnation
    radius (drift too strong for the radial engine), and a ``RuntimeError`` from :func:`_block_solutions`
    if a per-phase matrix Riccati integration fails.
    """
    flow = np.asarray(flow, dtype=float)
    dt_days = np.asarray(dt_days, dtype=float)
    cin_deviation = np.asarray(cin_deviation, dtype=float)
    if np.any(flow == 0.0):
        msg = "rest phases (flow == 0) with regional drift are not yet supported (rest-with-drift kernel)"
        raise NotImplementedError(msg)
    # The stagnation radius r_s = |A_0|/|v_d| is smallest for the weakest pumping phase, so size the grid
    # cap on the minimum per-phase |A_0| (the most restrictive envelope).
    a0_min = float(np.min(np.abs(flow[flow != 0.0]))) / (2.0 * c_geo)
    r_nodes, dr_weights, r_far = field_grid(flow, dt_days, c_geo, r_w, alpha_l, v_d, a0_min, n_quad)
    nm = 2 * n_modes + 1

    def solutions(s: npt.NDArray[np.complexfloating], a0: float, direction: str) -> dict:
        return _block_solutions(
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

    def propagate(field: npt.NDArray[np.floating], a0: float, direction: str, t_phase: float) -> npt.NDArray:
        def f_hat(s):
            return _resolvent_field_laplace(solutions(s, a0, direction), field, dr_weights, retardation_factor)

        return dehoog_inverse(f_hat=f_hat, t=t_phase, n_terms=n_terms, tol=tol)

    field = np.zeros((n_quad, nm))
    cout = np.full(len(flow), np.nan)
    phases = _phase_slices(flow)  # rest phases are excluded above (raised); every phase here pumps
    for idx, (sign, sl) in enumerate(phases):
        a0 = float(np.mean(np.abs(flow[sl]))) / (2.0 * c_geo)
        t_phase = float(np.sum(dt_days[sl]))
        if sign > 0:  # injection: propagate the buffer, then add the freshly injected resident profile
            if np.any(field):
                field = propagate(field, a0, _INJECTION, t_phase)
            corners = t_phase - np.concatenate(([0.0], np.cumsum(dt_days[sl])))  # descending, last is 0

            def f_hat_resident(s, a0=a0):
                return (
                    _resident_laplace(
                        solutions(s, a0, _INJECTION),
                        alpha_l=alpha_l,
                        d_m=molecular_diffusivity,
                        r_w=r_w,
                        a0=a0,
                        n_modes=n_modes,
                    )
                    / s[:, None, None]
                )

            g1 = dehoog_inverse(f_hat=f_hat_resident, t=corners, n_terms=n_terms, tol=tol)  # (n_corner, n_quad, nm)
            field += np.einsum("b,bqm->qm", cin_deviation[sl], g1[:-1] - g1[1:])
        else:  # extraction: read the m=0 well-face flux, then propagate the residual if more phases follow
            ext_corners = np.concatenate(([0.0], np.cumsum(dt_days[sl])))

            def f_hat_readout(s, a0=a0, field=field):
                return (
                    _readout_laplace(solutions(s, a0, "extraction"), field, dr_weights, retardation_factor, n_modes) / s
                )

            cdf = dehoog_inverse(f_hat=f_hat_readout, t=ext_corners, n_terms=n_terms, tol=tol)
            cout[sl] = (cdf[1:] - cdf[:-1]) / np.diff(ext_corners)
            if idx != len(phases) - 1:
                field = propagate(field, a0, "extraction", t_phase)
    return cout
