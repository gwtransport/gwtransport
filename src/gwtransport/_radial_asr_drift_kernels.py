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
