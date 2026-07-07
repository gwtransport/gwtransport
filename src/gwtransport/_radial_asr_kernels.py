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
* ``D_m > 0`` (molecular diffusion present): the decaying solution is the confluent-hypergeometric
  (Tricomi-U / Whittaker) function, but it is evaluated through its LOG-DERIVATIVE ``L = phi'/phi`` -- a
  vector Riccati ODE ``L' = -L^2 - ((D_m - sigma_A A_0)/G) L + s r/G`` integrated over the Laplace nodes
  (:func:`_integrate_logderiv`). ``L`` is ``O(kappa)`` (bounded -- no 10^900 special-function
  magnitudes), so the transfer functions (:func:`_transfer_riccati`) and the interior resolvent
  (:func:`resolvent_riccati`) are assembled from O(1) quantities, with the divergent Sturm-Liouville
  gauge carried in log space. This is exact to the de Hoog inversion floor at ANY ``A_0/D_m`` -- no
  special-function precision cap, no arbitrary-precision dependency -- and continuously becomes the Airy
  branch as ``D_m -> 0``. The exact flint/Arb Whittaker evaluation it replaced is retained as a
  machine-precision test oracle (``tests/src/_radial_asr_whittaker_oracle.py``).

Retardation enters by the standard linear-sorption rescaling of the constant-Q operator: dividing the
retarded equation ``R d_t C + ... `` by ``R`` is the unretarded equation with ``A_0 -> A_0/R`` and
``D_m -> D_m/R`` (mechanical dispersivity ``alpha_L`` is geometric and unchanged). Callers pass the
physical ``A_0`` / ``D_m`` and a retardation factor; the rescaling is applied here.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from scipy.special import airye, ive, kve

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
        Molecular diffusivity (m^2/day). ``0`` selects the Airy branch; ``> 0`` the Riccati
        log-derivative branch.
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

    # D_m > 0: Riccati log-derivative (numerical ODE on the log-derivative L = phi'/phi; exact to the
    # de Hoog floor at any A_0/D_m, no special-function precision cap). Divergent orientation sigma_A = +1.
    return _transfer_riccati(s.reshape(-1), r, r_w, alpha_l, a0_eff, d_m_eff, inject, detect).reshape(s.shape)


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
    source_log_weight: npt.NDArray[np.floating] | float = 0.0,
) -> npt.NDArray[np.complexfloating]:
    r"""Assemble ``Ghat(r, r'; s) = -(pref_a e^{ea} - pref_b e^{eb})`` from precomputed scaled-Airy pieces.

    ``piece_a``, ``piece_b`` are :func:`_resolvent_airy_pieces` at ``r_< = min(r, r')`` and
    ``r_> = max(r, r')``; ``piece_w`` at ``r_w``; ``r_sum = r_< + r_> = r + r'`` (the radii enter the
    bounded exponents only through their sum). The normalization ``N`` (with its huge exponent) is
    factored into the bounded exponents ``ea, eb``, so the result is overflow-safe (Sec. 1b of the
    plan). Splitting piece computation from assembly lets a caller evaluate the scaled Airy on a grid
    of radii once and assemble every output node from prefix selection -- the ``O(n^2) -> O(n)``
    saving the field propagator relies on.

    The gauge term ``g * r_sum = gauge_sign (r + r')/(2 alpha_L)`` grows with the radii, so for the
    field propagator its ``e^{g r_sum}`` factor is divergent (``+r/alpha_L`` injection) or its Airy
    counterpart is (``-r/alpha_L`` extraction); on its own it overflows/underflows double precision at
    Peclet ``r/alpha_L`` beyond ~700. It is tamed by the caller's Sturm-Liouville source weight
    ``e^{-gauge_sign r'/alpha_L}``, whose LOG (``source_log_weight = -gauge_sign r'/alpha_L``, per
    source node) must therefore be folded into the exponents *before* ``np.exp`` so the divergent parts
    cancel to ``gauge_sign (r - r')/(2 alpha_L)`` (bounded by ``r_max/(2 alpha_L)``, and dominated by the
    Airy decay) -- rather than overflowing to ``Inf`` and then meeting the taming factor as ``Inf * 0``.
    The default ``0.0`` reproduces the bare interior resolvent (no source weight).

    Parameters
    ----------
    piece_a, piece_b, piece_w : dict of ndarray
        :func:`_resolvent_airy_pieces` at ``r_<``, ``r_>`` and ``r_w`` respectively.
    r_sum : ndarray
        ``r_< + r_> = r + r'`` (the radii enter the bounded exponents only through their sum).
    alpha_l : float
        Longitudinal dispersivity (m).
    gauge_sign : float
        ``+1`` divergent (injection, Robin well BC) / ``-1`` convergent (extraction, Neumann well BC).
    source_log_weight : ndarray or float, optional
        Log of the caller's Sturm-Liouville source weight per source node ``r'``
        (``-gauge_sign r'/alpha_L``), broadcast over the source axis and folded into both exponents so
        the divergent gauge cancels before ``np.exp``. Default ``0.0`` (bare resolvent, no weight).

    Returns
    -------
    ndarray of complex
        ``Ghat(r, r'; s)`` (times the source weight when ``source_log_weight`` is given), same broadcast
        shape as the input pieces.
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
    exp_a = g * r_sum + source_log_weight - (piece_a["xi"] + piece_b["xi"] - 2.0 * piece_w["xi"])
    exp_b = g * r_sum + source_log_weight + (piece_a["xiR"] - piece_w["xiR"]) - (piece_b["xi"] - piece_w["xi"])
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
    # Both terms carry the scaling exponents in a SINGLE np.exp of the combined (bounded) sum: the outer
    # term's ``|Re z_w| + z_w`` alone overflows at Re(z_w) > ~354, but ``|Re z_w| + z_w - z_lt - z_gt``
    # <= 0 (since r_w <= r_< <= r_>) so exponentiating the sum is overflow-safe.
    term_inner = ive(0, z_lt) * kve(0, z_gt) * np.exp(np.abs(z_lt.real) - z_gt)
    term_outer = (
        (ive(1, z_w) / kve(1, z_w)) * kve(0, z_lt) * kve(0, z_gt) * np.exp(np.abs(z_w.real) + z_w - z_lt - z_gt)
    )
    return term_inner + term_outer


# ---------------------------------------------------------------------------
# D_m > 0 branch: Riccati log-derivative (numerical ODE, double precision)
# ---------------------------------------------------------------------------
# The decaying resident solution and the well-regular solution are tracked by their log-derivative
# L = C'/C of the constant-Q ODE, integrated as a vector ODE over the Laplace nodes. L is O(kappa)
# (bounded -- no 10^900 special-function magnitudes), so the transfer functions and the
# Sturm-Liouville interior resolvent are assembled from O(1) quantities. This is exact to the de Hoog
# inversion floor at ANY A_0/D_m -- no special-function precision blow-up, no tractability cap -- and
# continuously becomes the Airy branch as D_m -> 0.
_RICCATI_RTOL = 1e-12
_RICCATI_ATOL = 1e-13
# Outer boundary for the inward (decaying) integration. The truncated asymptotic IC at r_far washes out
# because the recessive solution is the inward attractor (damping ~ e^{-2 Re(kappa)(r_far - r)}), but the
# slowest (smallest Re(kappa)) Laplace node also needs r_far far enough that its z = 2 kappa r_far is in
# the large-z asymptotic. So r_far is extended by ~_RICCATI_RFAR_DECAY decay lengths 1/Re(kappa_min)
# beyond the field, floored at _RICCATI_RFAR_MULT * r_max and cost-capped at _RICCATI_RFAR_CAP * r_max
# (the floor on Re(kappa_min) keeps z at the cap ~ 2 * _RICCATI_RFAR_DECAY, large enough for any node).
_RICCATI_RFAR_MULT = 8.0
_RICCATI_RFAR_DECAY = 22.0
_RICCATI_RFAR_CAP = 500.0


def _integrate_logderiv(
    s: npt.NDArray[np.complexfloating],
    radii: npt.ArrayLike,
    r_w: float,
    alpha_l: float,
    a0_eff: float,
    d_m_eff: float,
    sigma_a: int,
    branch: str,
) -> tuple[npt.NDArray[np.complexfloating], npt.NDArray[np.complexfloating]]:
    r"""Vector Riccati integration of the log-derivative ``L = C'/C`` for the ``D_m > 0`` branch.

    Solves ``L' = -L^2 - ((D_m - sigma_A A_0)/G) L + s r/G`` (``G = alpha_L A_0 + D_m r``) over all
    Laplace nodes ``s`` at once, carrying the running integral ``J = int_{r_w}^{r} L dr``.

    * ``branch='decaying'``: inward from ``r_far`` with the recessive asymptotic IC
      ``L(r_far) = -kappa - a/x`` (``kappa = sqrt(s/D_m)``, ``x = r_far + a*``, ``a* = alpha_L A_0/D_m``,
      ``a = b/2 - kappa a*/2``, ``b = 1 - sigma_A A_0/D_m``). The decaying solution is the inward
      attractor, so the result is insensitive to ``r_far``.
    * ``branch='regular'``: outward from ``r_w`` with the well-BC IC ``L(r_w) = A_0/G(r_w)`` (injection,
      Robin ``F[u_0](r_w)=0``) or ``0`` (extraction, Neumann ``u_0'(r_w)=0``) -- both ``s``-independent.

    ``sigma_a`` is ``+1`` divergent (injection) / ``-1`` convergent (extraction); ``radii`` (all ``>= r_w``)
    are where ``L`` and ``J`` are returned.

    Returns
    -------
    ld : ndarray of complex, shape (n_s, n_radii)
        Log-derivative ``L`` at each requested radius.
    jj : ndarray of complex, shape (n_s, n_radii)
        ``int_{r_w}^{r} L dr`` at each requested radius.
    """
    s = np.asarray(s, dtype=complex).reshape(-1)
    n = s.size
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    r_max = max(float(radii.max()), r_w)

    def rhs(r: float, y: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        ld = y[:n]
        g = alpha_l * a0_eff + d_m_eff * r
        d_ld = -(ld * ld) - ((d_m_eff - sigma_a * a0_eff) / g) * ld + s * r / g
        return np.concatenate([d_ld, ld])

    if branch == "decaying":
        astar = alpha_l * a0_eff / d_m_eff
        kappa = np.sqrt(s / d_m_eff)
        # r_far must put the slowest node deep in the large-z asymptotic so the truncated IC washes out:
        # extend by ~_RICCATI_RFAR_DECAY decay lengths 1/Re(kappa_min) (Re(kappa) floored so the extension
        # is cost-capped at _RICCATI_RFAR_CAP * r_max while z = 2 kappa r_far stays large there).
        re_kmin = max(float(kappa.real.min()), _RICCATI_RFAR_DECAY / (_RICCATI_RFAR_CAP * r_max))
        r_far = max(_RICCATI_RFAR_MULT * r_max, r_max + _RICCATI_RFAR_DECAY / re_kmin)
        a = (1.0 - sigma_a * a0_eff / d_m_eff) / 2.0 - kappa * astar / 2.0
        y0 = np.concatenate([-kappa - a / (r_far + astar), np.zeros(n, dtype=complex)])
        sol = solve_ivp(
            rhs, [r_far, r_w], y0, rtol=_RICCATI_RTOL, atol=_RICCATI_ATOL, dense_output=True, method="DOP853"
        )
        y = sol.sol(radii)  # dense output at all radii at once -> shape (2n, n_radii)
        j_w = sol.sol(r_w)[n:, None]  # re-anchor J to int_{r_w}^{r} (the IC put J(r_far) = 0)
        return y[:n], y[n:] - j_w

    # regular branch: outward from r_w to r_max -- the growing solution is the stable outward attractor and
    # the well-BC IC (s-independent) is exact, so no washout is needed; it need only reach the field.
    ld0 = a0_eff / (alpha_l * a0_eff + d_m_eff * r_w) if sigma_a > 0 else 0.0
    y0 = np.concatenate([np.full(n, ld0, dtype=complex), np.zeros(n, dtype=complex)])
    sol = solve_ivp(rhs, [r_w, r_max], y0, rtol=_RICCATI_RTOL, atol=_RICCATI_ATOL, dense_output=True, method="DOP853")
    y = sol.sol(radii)
    return y[:n], y[n:]


def _transfer_riccati(
    s: npt.NDArray[np.complexfloating],
    r: float,
    r_w: float,
    alpha_l: float,
    a0_eff: float,
    d_m_eff: float,
    inject: str,
    detect: str,
) -> npt.NDArray[np.complexfloating]:
    r"""Four Kreft-Zuber transfer modes for the ``D_m > 0`` branch via the decaying log-derivative.

    With ``E = phi(r)/phi(r_w) = exp(int_{r_w}^{r} L)`` and the flux factor ``f(r) = 1 - (alpha_L +
    D_m r/A_0) L(r)`` (so ``F[phi](r) = phi(r) f(r)``), the modes are ``FF = E f(r)/f(r_w)``,
    ``FR = E/f(r_w)``, ``RF = E f(r)``, ``RR = E``. ``sigma_A = +1`` (the divergent operator). ``inject``
    / ``detect`` select the Kreft-Zuber well / detection boundary ('flux' or 'resident').

    Returns
    -------
    ndarray of complex
        ``g_hat(s)`` for the requested ``(inject, detect)`` mode, shape ``(n_s,)``.
    """
    ld, jj = _integrate_logderiv(s, [r, r_w], r_w, alpha_l, a0_eff, d_m_eff, +1, "decaying")
    l_r, l_w = ld[:, 0], ld[:, 1]
    e = np.exp(jj[:, 0])  # phi(r)/phi(r_w)
    f_r = 1.0 - (alpha_l + d_m_eff * r / a0_eff) * l_r
    f_w = 1.0 - (alpha_l + d_m_eff * r_w / a0_eff) * l_w
    numer = e * (f_r if detect == _FLUX else 1.0)
    denom = f_w if inject == _FLUX else 1.0
    return numer / denom


def resolvent_riccati(
    *,
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
    r"""Interior Sturm-Liouville resolvent applied to a source ``field``, ``D_m > 0``, via log-derivatives.

    Returns ``F(s)_{k,i} = sum_j Ghat(r_i, r_j; s_k) field_j w_j`` with the SL measure
    ``w_j = r_j G(r_j)^{b-1} dr_j`` (``b = 1 - sigma_A A_0/D_m``). The Green's function
    ``Ghat = phi_+(r_<) phi_-(r_>)/(-pW)`` is built from the decaying (``phi_-``, inward) and
    well-regular (``phi_+``, outward) solutions normalized at ``r_w``; ``pW = G(r_w)^b (L_-(r_w) -
    L_+(r_w))``. The divergent gauge ``G(r_w)^b`` (``b ~ -A_0/D_m``) is carried in LOG space
    (``LG_j = b ln(G(r_j)/G(r_w)) - ln G(r_j)``, bounded as ``A_0/D_m -> inf``) so it never
    underflows -- the assembly stays in double precision at any ``A_0/D_m``.

    Parameters
    ----------
    s : ndarray of complex
        Laplace nodes (conjugate to wall-clock time).
    field : ndarray
        Source resident-deviation profile on ``r_nodes``.
    r_nodes : ndarray
        Radial quadrature nodes (m), increasing, ``> r_w``.
    dr_weights : ndarray
        Quadrature weights for ``r_nodes`` (the ``dr`` measure).
    r_w : float
        Well radius (m).
    alpha_l, a0_eff, d_m_eff : float
        Dispersivity and the retardation-effective ``A_0`` / ``D_m``.
    direction : {'injection', 'extraction'}
        Phase orientation (divergent Robin / convergent Neumann well BC).

    Returns
    -------
    ndarray of complex
        ``F(s)_{k,i}`` -- the resolvent applied to ``field``, shape ``(n_s, n_nodes)``.
    """
    sigma_a = 1 if direction == _INJECTION else -1
    s = np.asarray(s, dtype=complex).reshape(-1)
    b = 1.0 - sigma_a * a0_eff / d_m_eff
    rad = np.concatenate(([r_w], r_nodes))
    ld_m, jj_m = _integrate_logderiv(s, rad, r_w, alpha_l, a0_eff, d_m_eff, sigma_a, "decaying")
    lm_w = ld_m[:, 0]  # L_-(r_w)
    im = jj_m[:, 1:]  # int_{r_w}^{r_i} L_-  (n_s, n)
    _, ip = _integrate_logderiv(s, r_nodes, r_w, alpha_l, a0_eff, d_m_eff, sigma_a, "regular")  # int L_+
    lp_w = a0_eff / (alpha_l * a0_eff + d_m_eff * r_w) if sigma_a > 0 else 0.0
    g_nodes = alpha_l * a0_eff + d_m_eff * r_nodes
    g_w = alpha_l * a0_eff + d_m_eff * r_w
    lg = b * np.log(g_nodes / g_w) - np.log(g_nodes)  # bounded gauge in log space
    c = field * r_nodes * dr_weights
    pmat = c[None, :] * np.exp(ip + lg[None, :])  # (n_s, n)
    smat = c[None, :] * np.exp(im + lg[None, :])
    prefix = np.cumsum(pmat, axis=1)  # sum_{j<=i}
    suffix = np.cumsum(smat[:, ::-1], axis=1)[:, ::-1] - smat  # sum_{j>i}
    denom = (lm_w - lp_w)[:, None]
    return -(np.exp(im) * prefix + np.exp(ip) * suffix) / denom
