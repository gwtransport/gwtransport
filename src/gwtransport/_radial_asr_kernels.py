r"""Exact per-phase (constant-Q) Laplace kernels for radial advection-dispersion.

This private module holds the closed-form Laplace-domain transfer functions for a single
fully-penetrating well in an infinite aquifer (the theory of the radial ASR knowledge base). For a
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
* ``D_m > 0`` (molecular diffusion present): the confluent-hypergeometric (Tricomi-U / Whittaker)
  branch, sampled exactly with ``flint.acb.hypgeom_u`` (python-flint / Arb compiled arbitrary-precision
  ball arithmetic) at the Laplace nodes -- exact and ~1e3x faster than mpmath, with a rigorous error
  enclosure. It is sampled per node (no vectorized complex Whittaker exists) but evaluated only where
  molecular diffusion is appreciable within the plume and ``A_0/D_m`` is below the tractability cap
  (basis-by-regime -- elsewhere the Airy form is exact to O(D_m/alpha_L|u|)).

Retardation enters by the standard linear-sorption rescaling of the constant-Q operator: dividing the
retarded equation ``R d_t C + ... `` by ``R`` is the unretarded equation with ``A_0 -> A_0/R`` and
``D_m -> D_m/R`` (mechanical dispersivity ``alpha_L`` is geometric and unchanged). Callers pass the
physical ``A_0`` / ``D_m`` and a retardation factor; the rescaling is applied here.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import warnings

import numpy as np
import numpy.typing as npt
from flint import acb, ctx
from scipy.special import airye, ive, kve

# Working precision (bits) for the flint (python-flint / Arb) Whittaker (D_m > 0) branch. 512 bits keeps
# the confluent-hypergeometric values exact (Arb's error ball stays below the double-precision floor) and
# the divergent-normalization cancellation clean up to A_0/D_m ~ 200. Arb returns a rigorous enclosure, so
# any precision loss is detectable rather than silent.
_WHITTAKER_PREC = 512

# Tractability cap for the Whittaker (D_m > 0) branch. b = 1 +/- A_0/D_m and the argument scales with
# a* = alpha_L A_0/D_m; the Arb confluent-hypergeometric evaluation returns non-finite balls once |b| is
# extreme (A_0/D_m beyond ~250-300, independent of precision). The cap is set safely below that; above it
# the Airy reduction is used (its O(D_m/alpha_L|u|) error is < 1% there and keeps shrinking with A_0/D_m).
_WHITTAKER_MAX_RATIO = 150.0


def _molecular_regime(molecular_diffusivity: float, a0: float, alpha_l: float, plume_reach: float) -> float:
    """Basis-by-regime molecular-diffusion dispatch (KB addendum Sec. A6); returns the D_m to use.

    With the molecular crossover radius ``a* = alpha_L A_0/D_m`` (beyond which molecular diffusion
    dominates the mechanical dispersion ``alpha_L |u|``):

    * ``D_m = 0`` or ``a* >= plume_reach`` -- molecular diffusion is sub-dominant everywhere the tracer
      goes; use the exact Airy reduction (``D_m := 0``). Its error is ``O(D_m/alpha_L|u|)``: measured
      ``< ~4%`` at the boundary ``a* ~ plume_reach``, falling below ``1%`` for ``a* > 4 plume_reach`` and
      to ``~1e-5`` for realistic groundwater ``D_m`` (where ``a*`` is orders of magnitude past the plume).
    * ``a* < plume_reach`` and ``A_0/D_m <= _WHITTAKER_MAX_RATIO`` -- molecular diffusion is appreciable
      within the plume and the flint (Arb) Whittaker branch is exact and tractable; keep ``D_m`` (exact).
    * ``a* < plume_reach`` and ``A_0/D_m > _WHITTAKER_MAX_RATIO`` -- molecular diffusion is appreciable but
      the Arb confluent-hypergeometric evaluation returns non-finite balls (``b = 1 - A_0/D_m`` too extreme
      for the evaluator); fall back to the Airy reduction and warn (relative error grows with
      ``plume_reach/a*``).

    Returns
    -------
    float
        The molecular diffusivity to use: the input value (Whittaker branch) or ``0.0`` (Airy branch).
    """
    if molecular_diffusivity <= 0.0:
        return 0.0
    a_star = alpha_l * a0 / molecular_diffusivity
    if a_star >= plume_reach:
        return 0.0
    if a0 / molecular_diffusivity > _WHITTAKER_MAX_RATIO:
        warnings.warn(
            f"Molecular diffusion is appreciable (crossover a*={a_star:.1f} m is within the plume reach "
            f"{plume_reach:.1f} m) but the exact flint Whittaker is non-finite for A_0/D_m="
            f"{a0 / molecular_diffusivity:.0f} (> {_WHITTAKER_MAX_RATIO:.0f}; b too extreme for Arb). "
            f"Falling back to the Airy reduction, which neglects molecular diffusion; the relative error "
            f"grows with plume_reach/a* = {plume_reach / a_star:.1f}. Reduce D_m if physically justified.",
            stacklevel=2,
        )
        return 0.0
    return molecular_diffusivity


# Injection / detection boundary types (Kreft-Zuber modes). "flux" applies the flux operator
# F[psi] = psi - (G/A_0) psi'; any other value ("resident") uses psi directly.
_FLUX = "flux"

# Phase orientation for the interior two-point resolvent. "injection" is the divergent operator
# (flow pushes outward, Robin/flux well BC); "extraction" is the convergent operator (flow pulls
# inward, Danckwerts/Neumann well BC).
_INJECTION = "injection"


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
) -> tuple[acb, acb]:
    r"""``(phi_s(r), F[phi_s](r))`` for the ``D_m > 0`` branch, as flint ``acb`` at scalar ``s``.

    With ``x = r + a*``, ``a* = alpha_L A_0/D_m``, ``kappa = sqrt(s/D_m)``, the decaying resident
    solution is ``phi_s = e^{-kappa x} U(a, b, 2 kappa x)`` (Tricomi U -- regular through the
    integer-``b`` degeneracy where the Kummer ``M`` is undefined), with ``b = 1 - sigma_A A_0/D_m``,
    ``a = b/2 - kappa a*/2``. Using ``dU/dz = -a U(a+1, b+1, z)``, the derivative is
    ``phi_s'(r) = -kappa e^{-kappa x}[U(a,b,z) + 2 a U(a+1,b+1,z)]`` so the flux operator
    ``F[phi_s] = phi_s - (alpha_L + D_m r/A_0) phi_s'`` is evaluated analytically (no numerical
    differentiation). All quantities use the retardation-effective ``A_0``/``D_m``.

    ``U`` is evaluated with ``flint.acb.hypgeom_u`` (Arb arbitrary-precision ball arithmetic) at
    ``_WHITTAKER_PREC`` bits. ``phi`` and ``flux`` are returned as ``acb`` (not cast to ``complex``)
    so the caller can take their ratio -- where the divergent gauge factors cancel -- before rounding.

    Returns
    -------
    phi : flint.acb
        Resident solution ``phi_s(r)``.
    flux : flint.acb
        Flux-operator image ``F[phi_s](r)``.
    """
    ctx.prec = _WHITTAKER_PREC
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
    r"""Return the two homogeneous solutions and their ``r``-derivatives for the ``D_m > 0`` resolvent.

    The constant-Q ODE ``G C'' + (D_m - sigma_A A_0) C' - s r C = 0`` (KB Sec. 4) has, with
    ``x = r + a*``, ``a* = alpha_L A_0/D_m``, ``kappa = sqrt(s/D_m)``, ``z = 2 kappa x``,
    ``b = 1 - sigma_A A_0/D_m``, ``a = b/2 - kappa a*/2``:

    * decaying solution ``u_inf = e^{-kappa x} U(a, b, z)`` (Tricomi ``U``), with
      ``u_inf' = -kappa e^{-kappa x}[U + 2a U(a+1, b+1, z)]``;
    * growing solution ``u_reg = e^{-kappa x} z^{1-b} M(a-b+1, 2-b, z)``. This second Kummer solution
      is used **uniformly**: ``2-b = 1 + A_0/D_m > 1`` for both orientations, so it is always
      regular (it sidesteps the integer-``b`` degeneracy at ``A_0/D_m in Z+`` where the plain ``M(a,b,z)``
      is undefined) and yields the identical Green's function as ``M(a,b,z)`` where the latter exists.

    Returns
    -------
    tuple of flint.acb
        ``(u_inf, u_inf', u_reg, u_reg')`` -- the decaying and growing solutions and their
        ``r``-derivatives, at scalar ``s`` and ``r`` (as ``acb`` so the caller's divergent-normalization
        cancellation runs in Arb extended precision before rounding).
    """
    ctx.prec = _WHITTAKER_PREC
    kappa = (acb(s.real, s.imag) / d_m_eff).sqrt()
    astar = alpha_l * a0_eff / d_m_eff
    b = 1 - sigma_a * a0_eff / d_m_eff
    a = b / 2 - kappa * astar / 2
    x = r + astar
    z = 2 * kappa * x
    expf = (-(kappa * x)).exp()
    # decaying branch (Tricomi U), dU/dz = -a U(a+1, b+1, z)
    u = z.hypgeom_u(a, acb(b))
    u1 = z.hypgeom_u(a + 1, acb(b + 1))
    u_inf = expf * u
    u_inf_p = -kappa * expf * (u + 2 * a * u1)
    # growing branch: z^{c} M(ap, bp, z), c = 1-b, ap = a-b+1, bp = 2-b;
    # d/dz[z^c M] = c z^{c-1} M + z^c (ap/bp) M(ap+1, bp+1)
    c, ap, bp = 1 - b, a - b + 1, 2 - b
    m = z.hypgeom_1f1(ap, acb(bp))
    m1 = z.hypgeom_1f1(ap + 1, acb(bp + 1))
    w_z = z ** acb(c) * m
    dw_z = c * z ** acb(c - 1) * m + z ** acb(c) * (ap / bp) * m1
    u_reg = expf * w_z
    u_reg_p = -kappa * expf * w_z + expf * dw_z * (2 * kappa)
    return u_inf, u_inf_p, u_reg, u_reg_p


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

    # Whittaker branch (D_m > 0): exact flint (Arb) sampling per node (divergent orientation sigma_A = +1).
    # phi/flux are returned as acb; the detect/inject ratio cancels the divergent gauge before rounding.
    out = np.empty(s.shape, dtype=complex)
    flat = out.reshape(-1)
    for i, sv in enumerate(s.reshape(-1)):
        phi_r, flux_r = _whittaker_phi_and_flux(complex(sv), r, alpha_l, a0_eff, d_m_eff, +1)
        phi_w, flux_w = _whittaker_phi_and_flux(complex(sv), r_w, alpha_l, a0_eff, d_m_eff, +1)
        numer = flux_r if detect == _FLUX else phi_r
        denom = flux_w if inject == _FLUX else phi_w
        flat[i] = complex(numer / denom)
    return out


def _resolvent_airy_pieces(
    s: npt.NDArray[np.complexfloating],
    r: npt.NDArray[np.floating] | float,
    alpha_l: float,
    a0_eff: float,
    gauge_sign: float,
) -> dict[str, npt.NDArray[np.complexfloating]]:
    r"""Scaled-Airy building blocks for the interior two-point resolvent at radius ``r`` (``D_m = 0``).

    The two homogeneous solutions of the constant-Q ODE (KB Sec. 4), in the gauge
    ``e^{gauge_sign * r/(2 alpha_L)}`` (``+1`` divergent/injection, ``-1`` convergent/extraction):

    * ``u_inf = s_inf * exp(gauge_sign r/2alpha_L - xi)`` -- the decaying branch (``Ai``),
    * ``u_reg = s_reg * exp(gauge_sign r/2alpha_L + xiR)`` -- the growing branch (``Bi``),

    with ``zeta = beta^{1/3} r + beta^{-2/3}/(4 alpha_L^2)``, ``beta = s/(alpha_L a0_eff)``,
    ``xi = (2/3) zeta^{3/2}``. The scaled amplitudes ``s_inf = Aie``, ``s_reg = Bie`` and the
    derivative amplitudes are O(1); the (possibly huge) gauge/Airy exponent is carried as the log
    quantities ``xi`` and ``xiR``. **Crucially** ``scipy.special.airye`` scales ``Ai`` by
    ``exp(+xi)`` but ``Bi`` by ``exp(-|Re xi|)``, so ``Ai = Aie exp(-xi)`` while ``Bi = Bie exp(+xiR)``
    with ``xiR = |Re xi|``: the two differ for complex ``s`` (every de Hoog node), so they are tracked
    separately. The caller forms only bounded exponent *differences* (no overflow to Pe ~ 600+).

    Returns
    -------
    dict
        ``s_inf, s_infp, s_reg, s_regp`` (scaled value and r-derivative amplitudes of ``u_inf``,
        ``u_reg``) and the log-exponents ``xi`` (complex, for ``Ai``) and ``xiR`` (``|Re xi|``, ``Bi``).
    """
    beta = s / (alpha_l * a0_eff)
    b13 = beta ** (1.0 / 3.0)
    zeta = b13 * r + beta ** (-2.0 / 3.0) / (4.0 * alpha_l * alpha_l)
    aie, aipe, bie, bipe = airye(zeta)
    xi = (2.0 / 3.0) * zeta**1.5
    g = gauge_sign / (2.0 * alpha_l)
    return {
        "s_inf": aie,
        "s_infp": g * aie + b13 * aipe,
        "s_reg": bie,
        "s_regp": g * bie + b13 * bipe,
        "xi": xi,
        "xiR": np.abs(xi.real),
    }


def interior_resolvent(
    *,
    s: npt.NDArray[np.complexfloating],
    r: float,
    r_prime: npt.ArrayLike,
    r_w: float,
    alpha_l: float,
    a0: float,
    direction: str,
) -> npt.NDArray[np.complexfloating]:
    r"""Interior two-point Laplace resolvent ``Ghat(r, r'; s)`` of a constant-Q phase (``D_m = 0``).

    ``Ghat`` is the kernel of the spatial resolvent ``(s - L)^{-1}`` of the per-phase generator ``L``
    (KB Sec. 7 / addendum Sec. A3): the field after propagating an initial resident profile ``f`` for
    flushed volume ``tau`` is ``f_resid(r) = L^{-1}_s[ int Ghat(r, r'; s) f(r') w(r') dr' ](tau)``, with
    the Sturm-Liouville weight ``w(r') = (2 c_geo r'/alpha_L) e^{-gauge_sign r'/alpha_L} dr'`` supplied
    by the caller. Built from the convergent/divergent Airy solutions with the physical well boundary
    condition (Danckwerts/Neumann for extraction, Robin/flux for injection) and outgoing decay:

    ``Ghat(r, r'; s) = -u_0(r_<) u_inf(r_>) / N(s)``,  ``N(s) = P(r)[u_0 u_inf' - u_0' u_inf]``,

    ``u_inf`` the decaying solution, ``u_0`` the well-BC solution, ``r_< = min(r, r')``,
    ``r_> = max(r, r')``, ``P = e^{-gauge_sign r/alpha_L}`` (``N`` is constant in ``r`` -- the SL Abel
    identity). The leading minus sign and ``N`` are pinned by the KB Sec. 7 duality: the well-face
    trace ``Ghat(r_w, r'; s) w(r')`` equals the extraction arrival kernel. The normalization is
    factored out before exponentiating, so the scaled-Airy form is overflow-safe.

    The Laplace variable enters only through ``beta = s/(alpha_L a0)`` (``D_m = 0``); for the
    flushed-volume clock pass ``s = flow_scale * p``, ``a0 = flow_scale/(2 c_geo)`` so that
    ``beta = 2 c_geo p/alpha_L`` is flow-magnitude independent. Retardation is a pure clock rescale
    handled by the caller (propagate over ``tau/R``); ``Ghat`` itself is retardation-free.

    Parameters
    ----------
    s : ndarray of complex
        Laplace nodes (conjugate to flushed volume). Shape ``(n_s,)``.
    r : float
        Output radius (m), ``>= r_w``.
    r_prime : array-like
        Source radius/radii (m), ``>= r_w``. Scalar or shape ``(n_r',)``.
    r_w : float
        Well radius (m).
    alpha_l : float
        Longitudinal dispersivity (m).
    a0 : float
        Flow scale ``A_0`` setting ``beta = s/(alpha_L a0)``.
    direction : {'injection', 'extraction'}
        Phase orientation: divergent (Robin well BC) or convergent (Neumann well BC).

    Returns
    -------
    ndarray of complex
        ``Ghat(r, r'; s)``, shape ``(n_s, n_r')`` (broadcast of ``s`` and ``r_prime``).
    """
    gauge_sign = 1.0 if direction == _INJECTION else -1.0
    s = np.asarray(s, dtype=complex).reshape(-1, 1)
    rp = np.atleast_1d(np.asarray(r_prime, dtype=float)).reshape(1, -1)
    r_a = np.minimum(r, rp)
    r_b = np.maximum(r, rp)
    piece_a = _resolvent_airy_pieces(s, r_a, alpha_l, a0, gauge_sign)
    piece_b = _resolvent_airy_pieces(s, r_b, alpha_l, a0, gauge_sign)
    piece_w = _resolvent_airy_pieces(s, r_w, alpha_l, a0, gauge_sign)
    return assemble_airy_resolvent(piece_a, piece_b, piece_w, r_a + r_b, alpha_l, gauge_sign)


def assemble_airy_resolvent(
    piece_a: dict[str, npt.NDArray[np.complexfloating]],
    piece_b: dict[str, npt.NDArray[np.complexfloating]],
    piece_w: dict[str, npt.NDArray[np.complexfloating]],
    r_sum: npt.NDArray[np.floating],
    alpha_l: float,
    gauge_sign: float,
) -> npt.NDArray[np.complexfloating]:
    r"""Assemble ``Ghat(r, r'; s) = -(pref_a e^{ea} - pref_b e^{eb})`` from precomputed scaled-Airy pieces.

    ``piece_a``, ``piece_b`` are :func:`_resolvent_airy_pieces` at ``r_< = min(r, r')`` and
    ``r_> = max(r, r')``; ``piece_w`` at ``r_w``; ``r_sum = r_< + r_> = r + r'`` (the radii enter the
    bounded exponents only through their sum). The normalization ``N`` (with its huge exponent) is
    factored into the bounded exponents ``ea, eb``, so the result is overflow-safe (Sec. 1b of the
    plan). Splitting piece computation from assembly lets a caller evaluate the scaled Airy on a grid
    of radii once and assemble every output node from prefix selection -- the ``O(n^2) -> O(n)``
    saving the field propagator relies on.

    Returns
    -------
    ndarray of complex
        ``Ghat(r, r'; s)``, same broadcast shape as the input pieces.
    """
    g = gauge_sign / (2.0 * alpha_l)
    if gauge_sign < 0:  # extraction: Danckwerts -> zero dispersive flux -> Neumann u_0'(r_w) = 0
        bc_inf, bc_reg = piece_w["s_infp"], piece_w["s_regp"]
    else:  # injection: Robin/flux F[u_0](r_w) = 0, F[u] = u - alpha_L u'
        bc_inf = piece_w["s_inf"] - alpha_l * piece_w["s_infp"]
        bc_reg = piece_w["s_reg"] - alpha_l * piece_w["s_regp"]
    # denom0 = scaled Wronskian piece at r_w (= b13/pi up to gauge).
    denom0 = piece_w["s_regp"] * piece_w["s_inf"] - piece_w["s_infp"] * piece_w["s_reg"]
    pref_a = bc_reg * piece_a["s_inf"] * piece_b["s_inf"] / (bc_inf * denom0)
    pref_b = piece_a["s_reg"] * piece_b["s_inf"] / denom0
    exp_a = g * r_sum - (piece_a["xi"] + piece_b["xi"] - 2.0 * piece_w["xi"])
    exp_b = g * r_sum + (piece_a["xiR"] - piece_w["xiR"]) - (piece_b["xi"] - piece_w["xi"])
    return -(pref_a * np.exp(exp_a) - pref_b * np.exp(exp_b))


def rest_resolvent(
    *,
    s: npt.NDArray[np.complexfloating],
    r: float,
    r_prime: npt.ArrayLike,
    r_w: float,
    d_m: float,
) -> npt.NDArray[np.complexfloating]:
    r"""Interior two-point resolvent ``Ghat(r, r'; s)`` of a rest (``Q = 0``) phase -- pure diffusion.

    With no flow the constant-Q ODE (KB Sec. 4) loses its advective and mechanical-dispersion terms and
    collapses to the order-0 modified Bessel equation ``C'' + C'/r - (s/D_m) C = 0`` with
    ``kappa = sqrt(s/D_m)``. The resident solution decaying as ``r -> inf`` is ``u_inf = K_0(kappa r)``;
    the no-dispersive-flux (Danckwerts/Neumann) well solution is
    ``u_0(r) = K_1(kappa r_w) I_0(kappa r) + I_1(kappa r_w) K_0(kappa r)`` (so ``u_0'(r_w) = 0``). The
    Sturm-Liouville Wronskian normalization is ``N(s) = r [u_0 u_inf' - u_0' u_inf] = -K_1(kappa r_w)``
    (constant in ``r``), giving

    ``Ghat(r, r'; s) = -u_0(r_<) u_inf(r_>) / N(s) = u_0(r_<) K_0(kappa r_>) / K_1(kappa r_w)``,

    ``r_< = min(r, r')``, ``r_> = max(r, r')``. It is evaluated overflow-safe with the exponentially
    scaled modified Bessel functions (``scipy.special.ive``/``kve``): each term is a ratio whose scaling
    exponents difference to a bounded value, so the growing ``I_0`` never overflows at high ``kappa r``.
    The clock is wall-clock time (molecular diffusion is autonomous in ``t``); pair with the source
    measure ``w(r') dr' = (r'/D_m) dr'`` (the Sturm-Liouville weight) when superposing a resident field.

    Parameters
    ----------
    s : ndarray of complex
        Laplace nodes (conjugate to wall-clock time). Shape ``(n_s,)``.
    r : float
        Output radius (m), ``>= r_w``.
    r_prime : array-like
        Source radius/radii (m), ``>= r_w``. Scalar or shape ``(n_r',)``.
    r_w : float
        Well radius (m).
    d_m : float
        Molecular diffusivity (m^2/day), ``> 0``.

    Returns
    -------
    ndarray of complex
        ``Ghat(r, r'; s)``, shape ``(n_s, n_r')`` (broadcast of ``s`` and ``r_prime``).
    """
    s = np.asarray(s, dtype=complex).reshape(-1, 1)
    rp = np.atleast_1d(np.asarray(r_prime, dtype=float)).reshape(1, -1)
    kappa = np.sqrt(s / d_m)  # principal root; Re(s) > 0 on the Bromwich contour gives Re(kappa) > 0
    r_lt = np.minimum(r, rp)
    r_gt = np.maximum(r, rp)
    z_lt, z_gt, z_w = kappa * r_lt, kappa * r_gt, kappa * r_w
    # Ghat = [I_0(z_<) + (I_1(z_w)/K_1(z_w)) K_0(z_<)] K_0(z_>); split so the scaled-Bessel scaling
    # exponents (ive scales by e^{-|Re z|}, kve by e^{+z}) difference to bounded values -- no overflow.
    term_inner = ive(0, z_lt) * kve(0, z_gt) * np.exp(np.abs(z_lt.real) - z_gt)
    term_outer = (
        (ive(1, z_w) / kve(1, z_w))
        * np.exp(np.abs(z_w.real) + z_w)
        * kve(0, z_lt)
        * kve(0, z_gt)
        * np.exp(-z_lt - z_gt)
    )
    return term_inner + term_outer
