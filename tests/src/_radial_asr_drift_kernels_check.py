"""Independent real-space generator of the drift coupled-mode coefficient matrices (tests only).

Cross-checks :func:`gwtransport._radial_asr_drift_kernels.block_coupling_matrices`. The production build
uses spectral (FFT) angular differentiation and FFT Fourier coefficients; this reference uses **analytic**
angular derivatives of the Scheidegger tensor components and **direct** Fourier integration (rectangle rule
on a uniform periodic grid), so an agreement certifies the production assembly independently of its FFT
machinery and index arithmetic.
"""

import numpy as np
import numpy.typing as npt


def real_space_coupling(
    r: npt.NDArray[np.floating],
    *,
    alpha_l: float,
    a0: float,
    v_d: float,
    d_m: float,
    n_modes: int,
    n_theta: int = 4096,
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    """Coefficient matrices ``A, B, S0`` of ``A c'' + B c' + (S0 + R s I) c = 0`` by direct quadrature.

    Mirrors the production derivation -- ``A = -M[D_rr]``,
    ``B = M[v_r - D_rr/r - d_r D_rr - (1/r) d_theta D_rtheta] + M[-2 D_rtheta/r] (i m)``,
    ``S0 = M[v_theta/r - (d_r D_rtheta)/r - (1/r^2) d_theta D_thth] (i m) + M[-D_thth/r^2] (-m^2)`` -- but
    every angular derivative is taken analytically and every Fourier coefficient ``M[f]`` by the rectangle
    rule, independent of the production FFT path.

    Returns
    -------
    a_mat, b_mat, s0_mat : ndarray of complex, each shape (n_r, 2 n_modes + 1, 2 n_modes + 1)
        The coefficient matrices at each radius.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    nm = 2 * n_modes + 1
    modes = np.arange(-n_modes, n_modes + 1)
    i_m = 1j * modes
    m2 = modes**2
    diff = modes[:, None] - modes[None, :]  # m_a - m_b in [-2M, 2M]
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    def toeplitz(f_theta: npt.NDArray[np.floating]) -> npt.NDArray[np.complexfloating]:
        # c_k = (1/2pi) int f e^{-i k theta} dtheta by the rectangle rule on the uniform periodic grid
        ck = np.array([np.mean(f_theta * np.exp(-1j * k * theta)) for k in range(-2 * n_modes, 2 * n_modes + 1)])
        return ck[diff + 2 * n_modes]

    a_mat = np.empty((r.size, nm, nm), dtype=complex)
    b_mat = np.empty_like(a_mat)
    s0_mat = np.empty_like(a_mat)
    for ir, rv in enumerate(r):
        v_r = a0 / rv + v_d * cos_t
        v_th = -v_d * sin_t
        speed = np.sqrt(v_r**2 + v_th**2)
        d_rr = alpha_l * v_r**2 / speed + d_m
        d_rt = alpha_l * v_r * v_th / speed
        d_tt = alpha_l * v_th**2 / speed + d_m

        # analytic radial derivatives (d_r v_r = -a0/r^2, v_th r-independent)
        dv_r = -a0 / rv**2
        dspeed = v_r * dv_r / speed
        d_drr = alpha_l * (2.0 * v_r * dv_r / speed - v_r**2 * dspeed / speed**2)
        d_drt = alpha_l * (dv_r * v_th / speed - v_r * v_th * dspeed / speed**2)

        # analytic angular derivatives (d_theta v_r = -v_d sin, d_theta v_th = -v_d cos)
        dth_v_r = -v_d * sin_t
        dth_v_th = -v_d * cos_t
        dth_speed = (v_r * dth_v_r + v_th * dth_v_th) / speed
        dth_drt = alpha_l * ((dth_v_r * v_th + v_r * dth_v_th) / speed - v_r * v_th * dth_speed / speed**2)
        dth_dtt = alpha_l * (2.0 * v_th * dth_v_th / speed - v_th**2 * dth_speed / speed**2)

        a_mat[ir] = -toeplitz(d_rr)
        b_mat[ir] = toeplitz(v_r - d_rr / rv - d_drr - dth_drt / rv) + toeplitz(-2.0 * d_rt / rv) * i_m[None, :]
        s0_mat[ir] = (
            toeplitz(v_th / rv - d_drt / rv - dth_dtt / rv**2) * i_m[None, :] + toeplitz(-d_tt / rv**2) * (-m2)[None, :]
        )
    return a_mat, b_mat, s0_mat
