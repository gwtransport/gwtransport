r"""Tests-only flint/Arb Whittaker oracle for the ``D_m > 0`` radial-ASR kernel.

This is an INDEPENDENT, machine-precision ground truth for the production Riccati (log-derivative)
branch in :mod:`gwtransport._radial_asr_kernels`. It evaluates the confluent-hypergeometric
(Tricomi-U / Whittaker) solutions of the constant-Q ODE ``G(r) C'' + (D_m - sigma_A A_0) C' - s r C = 0``
(``G = alpha_L A_0 + D_m r``) exactly with ``flint.acb.hypgeom_u`` / ``hypgeom_1f1`` (Arb compiled
arbitrary-precision ball arithmetic), at a working precision scaled to the divergent-gauge cancellation
(``~ kappa a*`` bits). It was the production kernel before the Riccati rewrite; it lives here so the
production code carries no arbitrary-precision dependency (python-flint is a test-only requirement) while
tests retain a far stronger check than the finite-volume oracle (machine precision vs. ~1%).

Domain of validity (where this oracle is correct, hence where tests use it):

* Injection (``direction='injection'``, ``sigma_A = +1``): any ``A_0/D_m`` -- the Tricomi ``U`` is
  regular through the integer-``b`` degeneracy.
* Extraction (``direction='extraction'``, ``sigma_A = -1``): NON-integer ``A_0/D_m`` only. The growing
  branch uses ``M(a-b+1, 2-b, z)`` with ``2-b = 1 - A_0/D_m`` for extraction, which hits a
  non-positive-integer pole (``NaN``) at integer ratios. The production Riccati path has no such
  degeneracy; tests of the extraction resolvent therefore pick non-integer ratios for the oracle
  comparison and verify integer-ratio finiteness against the Riccati path's own self-consistency.
* Above ``A_0/D_m ~ 1000`` the required precision exceeds the cap and the oracle is impractical/non-finite;
  the Riccati path is validated there by self-consistency and the FV oracle instead.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from flint import acb, ctx  # ty: ignore[unresolved-import]  # python-flint is a test-only dependency

_WHITTAKER_PREC_BASE = 256
_WHITTAKER_PREC_MAX = 12000


def _whittaker_prec(s_abs: float, alpha_l: float, a0_eff: float, d_m_eff: float) -> int:
    """Adaptive Arb working precision (bits): scales with the gauge cancellation ``~ kappa a*``."""
    kappa = (s_abs / d_m_eff) ** 0.5
    astar = alpha_l * a0_eff / d_m_eff
    return min(_WHITTAKER_PREC_MAX, _WHITTAKER_PREC_BASE + int(2.0 * kappa * astar / np.log(2.0)))


def _whittaker_phi_and_flux(
    s: complex, r: float, alpha_l: float, a0_eff: float, d_m_eff: float, sigma_a: int
) -> tuple[acb, acb]:
    r"""``(phi_s(r), F[phi_s](r))`` for the ``D_m > 0`` branch, as flint ``acb`` at scalar ``s``.

    Decaying resident ``phi_s = e^{-kappa x} U(a, b, 2 kappa x)`` (Tricomi U), ``x = r + a*``,
    ``a* = alpha_L A_0/D_m``, ``kappa = sqrt(s/D_m)``, ``b = 1 - sigma_A A_0/D_m``, ``a = b/2 - kappa a*/2``;
    ``phi_s' = -kappa e^{-kappa x}[U + 2a U(a+1,b+1,z)]``; flux ``F[phi] = phi - (alpha_L + D_m r/A_0) phi'``.
    Returned as ``acb`` so the caller's detect/inject ratio cancels the divergent gauge before rounding.
    """
    ctx.prec = _whittaker_prec(abs(s), alpha_l, a0_eff, d_m_eff)
    kappa = (acb(s.real, s.imag) / d_m_eff).sqrt()
    astar = alpha_l * a0_eff / d_m_eff
    b = 1 - sigma_a * a0_eff / d_m_eff
    a = b / 2 - kappa * astar / 2
    x = r + astar
    z = 2 * kappa * x
    u = z.hypgeom_u(a, acb(b))
    u1 = z.hypgeom_u(a + 1, acb(b + 1))  # U(a+1, b+1, z) for dU/dz = -a U(a+1,b+1,z)
    expf = (-(kappa * x)).exp()
    phi = expf * u
    dphi = -kappa * expf * (u + 2 * a * u1)
    flux = phi - (alpha_l + d_m_eff * r / a0_eff) * dphi
    return phi, flux


def whittaker_resolvent_solutions(
    s: complex, r: float, alpha_l: float, a0_eff: float, d_m_eff: float, sigma_a: int
) -> tuple[acb, acb, acb, acb]:
    r"""Two homogeneous solutions ``(u_inf, u_inf', u_reg, u_reg')`` of the ``D_m > 0`` resolvent ODE.

    Decaying ``u_inf = e^{-kappa x} U(a, b, z)`` (Tricomi U) and growing ``u_reg = e^{-kappa x} z^{1-b}
    M(a-b+1, 2-b, z)`` (second Kummer solution), with ``x = r + a*``, ``z = 2 kappa x``,
    ``b = 1 - sigma_A A_0/D_m``, ``a = b/2 - kappa a*/2``. Returned as ``acb`` (the divergent normalization
    cancels in the caller's Green's function). Valid for injection at any ratio; for extraction the ``M``
    branch is undefined at integer ``A_0/D_m`` (a non-positive-integer second parameter).
    """
    ctx.prec = _whittaker_prec(abs(s), alpha_l, a0_eff, d_m_eff)
    kappa = (acb(s.real, s.imag) / d_m_eff).sqrt()
    astar = alpha_l * a0_eff / d_m_eff
    b = 1 - sigma_a * a0_eff / d_m_eff
    a = b / 2 - kappa * astar / 2
    x = r + astar
    z = 2 * kappa * x
    expf = (-(kappa * x)).exp()
    u = z.hypgeom_u(a, acb(b))
    u1 = z.hypgeom_u(a + 1, acb(b + 1))
    u_inf = expf * u
    u_inf_p = -kappa * expf * (u + 2 * a * u1)
    c, ap, bp = 1 - b, a - b + 1, 2 - b
    m = z.hypgeom_1f1(ap, acb(bp))
    m1 = z.hypgeom_1f1(ap + 1, acb(bp + 1))
    w_z = z ** acb(c) * m
    dw_z = c * z ** acb(c - 1) * m + z ** acb(c) * (ap / bp) * m1
    u_reg = expf * w_z
    u_reg_p = -kappa * expf * w_z + expf * dw_z * (2 * kappa)
    return u_inf, u_inf_p, u_reg, u_reg_p


def transfer_function_oracle(
    *,
    s: npt.NDArray[np.complexfloating],
    r: float,
    r_w: float,
    alpha_l: float,
    a0: float,
    d_m: float,
    retardation_factor: float = 1.0,
    inject: str = "flux",
    detect: str = "flux",
) -> npt.NDArray[np.complexfloating]:
    """Whittaker (flint) transfer function for ``D_m > 0`` -- independent oracle for ``_transfer_riccati``.

    Same four Kreft-Zuber modes as :func:`gwtransport._radial_asr_kernels.transfer_function` (divergent
    orientation ``sigma_A = +1``), evaluated per Laplace node with the exact Tricomi-U solutions.
    """
    a0_eff, d_m_eff = a0 / retardation_factor, d_m / retardation_factor
    s = np.asarray(s, dtype=complex)
    out = np.empty(s.shape, dtype=complex)
    flat = out.reshape(-1)
    for i, sv in enumerate(s.reshape(-1)):
        phi_r, flux_r = _whittaker_phi_and_flux(complex(sv), r, alpha_l, a0_eff, d_m_eff, +1)
        phi_w, flux_w = _whittaker_phi_and_flux(complex(sv), r_w, alpha_l, a0_eff, d_m_eff, +1)
        numer = flux_r if detect == "flux" else phi_r
        denom = flux_w if inject == "flux" else phi_w
        flat[i] = complex(numer / denom)
    return out


def resolvent_oracle(
    s: npt.NDArray[np.complexfloating],
    field: npt.NDArray[np.floating],
    r_nodes: npt.NDArray[np.floating],
    dr_weights: npt.NDArray[np.floating],
    r_w: float,
    alpha_l: float,
    a0_eff: float,
    d_m_eff: float,
    direction: str,
) -> npt.NDArray[np.complexfloating]:
    r"""Whittaker (flint) interior resolvent applied to ``field`` -- oracle for :func:`resolvent_riccati`.

    ``F(s)_{k,i} = -[u_inf(r_i) sum_{j<=i} u_0(r_j) w_j + u_0(r_i) sum_{j>i} u_inf(r_j) w_j]/N(s)`` with
    the SL weight ``w_j = r_j G(r_j)^{b-1} dr_j``, ``N = G(r_w)^b W``. Same interface and output as the
    production :func:`gwtransport._radial_asr_kernels.resolvent_riccati`.
    """
    sigma_a = 1 if direction == "injection" else -1
    s = np.asarray(s, dtype=complex).reshape(-1)
    n = len(r_nodes)
    g_eff = alpha_l * a0_eff + d_m_eff * r_nodes
    sl_weight = r_nodes * g_eff ** (-sigma_a * a0_eff / d_m_eff)
    weighted = (np.asarray(field, dtype=float) * sl_weight * dr_weights).astype(complex)
    b_exp = 1 - sigma_a * a0_eff / d_m_eff
    f_mat = np.empty((s.size, n), dtype=complex)
    for k, pk in enumerate(s):
        sk = complex(pk)
        ctx.prec = _whittaker_prec(abs(sk), alpha_l, a0_eff, d_m_eff)
        g_well = acb(alpha_l * a0_eff + d_m_eff * r_w) ** acb(b_exp)  # divergent N gauge; cancels in F(s)_i
        uiw, uiwp, urw, urwp = whittaker_resolvent_solutions(sk, r_w, alpha_l, a0_eff, d_m_eff, sigma_a)
        if sigma_a < 0:  # extraction: Danckwerts/Neumann u_0'(r_w) = 0
            bc_reg, bc_inf = urwp, uiwp
        else:  # injection: Robin F[u]=u-(G/A_0)u', F[u_0](r_w)=0
            fac = alpha_l + d_m_eff * r_w / a0_eff
            bc_reg, bc_inf = urw - fac * urwp, uiw - fac * uiwp
        u0w = bc_reg * uiw - bc_inf * urw
        u0wp = bc_reg * uiwp - bc_inf * urwp
        n_s = g_well * (u0w * uiwp - u0wp * uiw)
        u_inf_g, u0_g = [], []
        for r in r_nodes:
            ui, _, ur, _ = whittaker_resolvent_solutions(sk, float(r), alpha_l, a0_eff, d_m_eff, sigma_a)
            u_inf_g.append(ui)
            u0_g.append(bc_reg * ui - bc_inf * ur)
        wj = [acb(float(weighted[m].real), float(weighted[m].imag)) for m in range(n)]
        prefix, acc = [acb(0)] * n, acb(0)
        for m in range(n):
            acc += u0_g[m] * wj[m]
            prefix[m] = acc
        suffix, acc = [acb(0)] * n, acb(0)
        for m in range(n - 1, -1, -1):
            suffix[m] = acc
            acc += u_inf_g[m] * wj[m]
        for i in range(n):
            f_mat[k, i] = complex(-(u_inf_g[i] * prefix[i] + u0_g[i] * suffix[i]) / n_s)
    return f_mat
