r"""Exact per-phase (constant-Q) Laplace kernels for radial advection-dispersion.

This private module holds the closed-form Laplace-domain transfer functions for a single
fully-penetrating well in an infinite aquifer (the theory of the radial-ADE knowledge base). For a
constant-Q phase the volume-coordinate PDE ``d_t C + Q d_V C = d_V(D_V d_V C)`` has, in the Laplace
domain ``t -> s``, the ODE ``G(r) C'' + (D_m - sigma_A A_0) C' - s r C = 0`` with
``G(r) = alpha_L A_0 + D_m r``. The decaying branch on ``[r_w, inf)`` gives the resident solution
``phi_s`` (Airy when ``D_m = 0``; Tricomi-U / Whittaker when ``D_m > 0``), and the Kreft-Zuber flux
operator ``F[psi] = psi - (G/A_0) psi'`` builds the four injection/detection transfer functions.

Two evaluation regimes
----------------------
* ``D_m = 0`` (mechanical dispersion only): Airy functions of complex argument via
  ``scipy.special.airye`` (exponentially scaled), vectorized over the Laplace nodes. The scaling is
  essential -- the raw ``phi_s = e^{r/2 alpha_L} Ai(zeta)`` overflows/underflows to NaN for
  Peclet ``r/alpha_L`` beyond ~200 (the prefactor overflows while ``Ai`` underflows). All transfer
  functions are *ratios* of ``phi_s`` / ``F[phi_s]`` at ``r`` and ``r_w``; evaluating them with the
  Airy scaling factored into a single bounded log-amplitude keeps the ratio finite at any Peclet.
* ``D_m > 0`` (molecular diffusion present): the confluent-hypergeometric (Whittaker) branch,
  sampled exactly with ``mpmath.whitw`` at the Laplace nodes. No library provides a vectorized
  complex double-precision Whittaker, so this path is slower; it is used only where molecular
  diffusion is appreciable (basis-by-regime -- elsewhere the Airy form is exact to O(D_m/alpha_L|u|)).

Retardation enters by the standard linear-sorption rescaling of the constant-Q operator: dividing the
retarded equation ``R d_t C + ... `` by ``R`` is the unretarded equation with ``A_0 -> A_0/R`` and
``D_m -> D_m/R`` (mechanical dispersivity ``alpha_L`` is geometric and unchanged). Callers pass the
physical ``A_0`` / ``D_m`` and a retardation factor; the rescaling is applied here.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import mpmath as mp
import numpy as np
import numpy.typing as npt
from scipy.special import airye

# Working precision (decimal digits) for the mpmath Whittaker (D_m > 0) branch. 50 digits is ample
# for double-precision output and absorbs the cancellation in the U-function ratios.
_WHITTAKER_DPS = 50

# Injection / detection boundary types (Kreft-Zuber modes). "flux" applies the flux operator
# F[psi] = psi - (G/A_0) psi'; any other value ("resident") uses psi directly.
_FLUX = "flux"


def _airy_amplitudes(
    s: npt.NDArray[np.complexfloating], r: float, alpha_l: float, a0_eff: float
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    r"""Scaled Airy building blocks at radius ``r`` for the ``D_m = 0`` branch.

    Returns ``(log_amp, psi_resident, psi_flux)`` such that, with the Airy scaling factored out,

    * ``phi_s(r)      = exp(log_amp) * psi_resident``
    * ``F[phi_s](r)   = exp(log_amp) * psi_flux``

    where ``psi_resident = Aie(zeta)`` and ``psi_flux = 0.5 Aie(zeta) - alpha_L beta^{1/3} Aipe(zeta)``
    are O(1) (``Aie``/``Aipe`` are the exponentially scaled Airy functions, ``scipy.special.airye``),
    and ``log_amp = r/(2 alpha_L) - (2/3) zeta^{3/2}`` carries the (bounded, once differenced between
    ``r`` and ``r_w``) exponent. Here ``beta = s/(alpha_L a0_eff)`` and
    ``zeta = beta^{1/3} r + beta^{-2/3}/(4 alpha_L^2)``.

    Keeping the amplitude as a log and the Airy parts scaled is what prevents the high-Peclet
    overflow: the raw ``phi_s`` over/under-flows, but every transfer function is a ratio in which the
    ``exp(log_amp)`` factors difference to a bounded exponent.

    Returns
    -------
    log_amp : ndarray of complex
        Bounded log-amplitude ``r/(2 alpha_L) - (2/3) zeta^{3/2}`` per node.
    psi_resident : ndarray of complex
        Scaled resident amplitude ``Aie(zeta)``.
    psi_flux : ndarray of complex
        Scaled flux amplitude ``0.5 Aie(zeta) - alpha_L beta^{1/3} Aipe(zeta)``.
    """
    beta = s / (alpha_l * a0_eff)
    b13 = beta ** (1.0 / 3.0)  # principal cube root; s on the Bromwich contour has Re(s) > 0
    zeta = b13 * r + 1.0 / (4.0 * alpha_l * alpha_l) * beta ** (-2.0 / 3.0)
    eai, eaip, _, _ = airye(zeta)
    psi_resident = eai
    psi_flux = 0.5 * eai - alpha_l * b13 * eaip
    log_amp = r / (2.0 * alpha_l) - (2.0 / 3.0) * zeta**1.5
    return log_amp, psi_resident, psi_flux


def _whittaker_phi_and_flux(
    s: complex, r: float, alpha_l: float, a0_eff: float, d_m_eff: float, sigma_a: int
) -> tuple[mp.mpc, mp.mpc]:
    r"""``(phi_s(r), F[phi_s](r))`` for the ``D_m > 0`` branch, in mpmath at scalar ``s``.

    With ``x = r + a*``, ``a* = alpha_L A_0/D_m``, ``kappa = sqrt(s/D_m)``, the decaying resident
    solution is ``phi_s = e^{-kappa x} U(a, b, 2 kappa x)`` (Tricomi U -- regular through the
    integer-``b`` degeneracy where the Kummer ``M`` is undefined), with ``b = 1 - sigma_A A_0/D_m``,
    ``a = b/2 - kappa a*/2``. Using ``dU/dz = -a U(a+1, b+1, z)``, the derivative is
    ``phi_s'(r) = -kappa e^{-kappa x}[U(a,b,z) + 2 a U(a+1,b+1,z)]`` so the flux operator
    ``F[phi_s] = phi_s - (alpha_L + D_m r/A_0) phi_s'`` is evaluated analytically (no numerical
    differentiation). All quantities use the retardation-effective ``A_0``/``D_m``.

    Returns
    -------
    phi : mpmath.mpc
        Resident solution ``phi_s(r)``.
    flux : mpmath.mpc
        Flux-operator image ``F[phi_s](r)``.
    """
    kappa = mp.sqrt(s / d_m_eff)
    astar = alpha_l * a0_eff / d_m_eff
    b = 1 - sigma_a * a0_eff / d_m_eff
    a = b / 2 - kappa * astar / 2
    x = r + astar
    z = 2 * kappa * x
    u = mp.hyperu(a, b, z)
    u1 = mp.hyperu(a + 1, b + 1, z)  # U(a+1, b+1, z) for dU/dz = -a U(a+1,b+1,z)
    expf = mp.e ** (-kappa * x)
    phi = expf * u
    dphi = -kappa * expf * (u + 2 * a * u1)
    flux = phi - (alpha_l + d_m_eff * r / a0_eff) * dphi
    return phi, flux


def transfer_function(
    *,
    s: npt.NDArray[np.complexfloating],
    r: float,
    r_w: float,
    alpha_l: float,
    a0: float,
    d_m: float = 0.0,
    retardation_factor: float = 1.0,
    inject: str = _FLUX,
    detect: str = _FLUX,
) -> npt.NDArray[np.complexfloating]:
    r"""Laplace-domain transfer function ``g_hat(s)`` for a constant-Q divergent phase.

    Implements the four Kreft-Zuber injection/detection modes (KB Sec. 5) as ratios of ``phi_s``
    (resident) and ``F[phi_s]`` (flux) evaluated at the detection radius ``r`` and the well ``r_w``:

    ===========  ============  ===================================
    inject       detect        ``g_hat``
    ===========  ============  ===================================
    flux (FF)    flux          ``F[phi_s](r) / F[phi_s](r_w)``
    flux (FR)    resident      ``phi_s(r) / F[phi_s](r_w)``
    resident RF  flux          ``F[phi_s](r) / phi_s(r_w)``
    resident RR  resident      ``phi_s(r) / phi_s(r_w)``
    ===========  ============  ===================================

    ``g_hat(0) = 1`` (mass conservation). The flux-flux (FF) mode is the package observable.

    Parameters
    ----------
    s : ndarray of complex
        Laplace nodes (conjugate to time). The Bromwich contour has ``Re(s) > 0``; do not pass
        ``s = 0`` (``g_hat(0) = 1`` is the limit).
    r : float
        Detection radius (m), ``r >= r_w``.
    r_w : float
        Well (screen) radius (m).
    alpha_l : float
        Longitudinal dispersivity (m), ``> 0`` for the Airy branch.
    a0 : float
        Physical flow scale ``A_0 = |Q| / (2 pi b n)`` (m^2/day).
    d_m : float, optional
        Molecular diffusivity (m^2/day). ``0`` selects the Airy branch; ``> 0`` the Whittaker branch.
    retardation_factor : float, optional
        Linear retardation ``R >= 1``. Default 1.
    inject, detect : {'flux', 'resident'}, optional
        Boundary type at the well (injection) and at ``r`` (detection). Default flux/flux (FF).

    Returns
    -------
    ndarray of complex
        ``g_hat(s)``, same shape as ``s``.
    """
    # Linear retardation rescales the constant-Q operator: A_0 -> A_0/R, D_m -> D_m/R (alpha_L is
    # geometric and unchanged); the residence time then scales by R.
    a0_eff, d_m_eff = a0 / retardation_factor, d_m / retardation_factor
    s = np.asarray(s, dtype=complex)

    if d_m_eff == 0.0:
        # Airy branch: vectorized, overflow-safe (scaled-Airy amplitudes, bounded log-difference).
        log_r, res_r, flux_r = _airy_amplitudes(s, r, alpha_l, a0_eff)
        log_w, res_w, flux_w = _airy_amplitudes(s, r_w, alpha_l, a0_eff)
        numer = flux_r if detect == _FLUX else res_r
        denom = flux_w if inject == _FLUX else res_w
        return np.exp(log_r - log_w) * (numer / denom)

    # Whittaker branch (D_m > 0): exact mpmath sampling per node (divergent orientation sigma_A = +1).
    out = np.empty(s.shape, dtype=complex)
    flat = out.reshape(-1)
    with mp.workdps(_WHITTAKER_DPS):
        for i, sv in enumerate(s.reshape(-1)):
            phi_r, flux_r = _whittaker_phi_and_flux(complex(sv), r, alpha_l, a0_eff, d_m_eff, +1)
            phi_w, flux_w = _whittaker_phi_and_flux(complex(sv), r_w, alpha_l, a0_eff, d_m_eff, +1)
            numer = flux_r if detect == _FLUX else phi_r
            denom = flux_w if inject == _FLUX else phi_w
            flat[i] = complex(numer / denom)
    return out
