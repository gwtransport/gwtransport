"""
Mathematical Foundation for Front Tracking with Nonlinear Sorption.

This module provides exact analytical computations for:

- Freundlich, Langmuir, and constant retardation models
- Brooks-Corey and van Genuchten-Mualem unsaturated conductivity models
  (for Kinematic-Wave percolation, see :mod:`gwtransport.percolation`)
- Shock velocities via Rankine-Hugoniot condition
- Characteristic velocities and positions
- First arrival time calculations
- Entropy condition verification

All sorption-class computations are exact analytical formulas; the
van Genuchten-Mualem class uses ``scipy.optimize.brentq`` for the two
inversions that have no closed form.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

# Numerical tolerance constants
EPSILON_FREUNDLICH_N = 1e-10  # Tolerance for checking if n ≈ 1.0 (Freundlich constructor rejects this)
EPSILON_DENOMINATOR = 1e-15  # Tolerance for near-zero denominators in shock velocity
_C_MIN = 1e-12  # Shared dry-soil singularity floor for Freundlich n>1, Brooks-Corey, vG-Mualem.
BRENTQ_XTOL = 1e-14  # brentq absolute tolerance for vG-Mualem inversions; matches _invert_freundlich_cr_zero.


class NonlinearSorption(ABC):
    """Abstract base for concentration-dependent sorption models.

    Subclasses must implement `retardation`, `total_concentration`, and
    `concentration_from_retardation`. Shock velocity and entropy checking
    are provided generically via the Rankine-Hugoniot and Lax conditions.

    See Also
    --------
    FreundlichSorption : Freundlich isotherm implementation.
    LangmuirSorption : Langmuir isotherm implementation.
    ConstantRetardation : Linear (constant R) retardation model.
    """

    @abstractmethod
    def retardation(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Compute retardation factor R(C)."""

    @abstractmethod
    def total_concentration(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Compute total concentration (dissolved + sorbed per unit pore volume)."""

    @abstractmethod
    def concentration_from_retardation(self, r: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Invert retardation factor to obtain concentration."""

    def shock_speed(self, c_left: float, c_right: float) -> float:
        """Compute shock speed dV/dθ via Rankine-Hugoniot in (V, θ) coordinates.

        With cumulative-flow coordinate θ = ∫flow(t') dt', the PDE
        ``∂C_T/∂t + flow·∂C/∂V = 0`` becomes ``∂C_T/∂θ + ∂C/∂V = 0``, and
        Rankine-Hugoniot reduces to::

            dV_s/dθ = (C_R - C_L) / (C_T(C_R) - C_T(C_L))

        Flow drops out entirely; the result is a property of the sorption
        isotherm alone.

        Parameters
        ----------
        c_left : float
            Concentration upstream (behind) shock [mass/volume].
        c_right : float
            Concentration downstream (ahead of) shock [mass/volume].

        Returns
        -------
        shock_speed : float
            Shock speed dV/dθ [m³ / m³ flow = dimensionless].
        """
        c_total_left = self.total_concentration(c_left)
        c_total_right = self.total_concentration(c_right)
        denom = c_total_right - c_total_left

        if abs(denom) < EPSILON_DENOMINATOR:
            avg_retardation = 0.5 * float(self.retardation(c_left) + self.retardation(c_right))
            # Degenerate (zero-strength) shock: its speed is the characteristic speed 1/R. A pair of
            # saturated states (R = 0, e.g. Mualem-vG at S_e = 1) gives +∞, matching characteristic_speed.
            return float("inf") if avg_retardation == 0.0 else 1.0 / avg_retardation

        return float((c_right - c_left) / denom)

    def c_and_total_from_retardation(self, r: float) -> tuple[float, float]:
        """Return ``(c, C_T(c))`` at a given retardation ``r``.

        Default implementation calls ``concentration_from_retardation(r)`` then
        ``total_concentration(c)`` — two independent root-finds for sorptions
        where both routes back-solve the same equation (e.g. vG-Mualem with
        ``L ≠ 0``). Subclasses for which both can be computed from a single
        root-find should override this for ~2× speedup of the IBP fan
        integrators.
        """
        c = float(self.concentration_from_retardation(r))
        ct = float(self.total_concentration(c))
        return c, ct

    def fan_converges_at_infinity(self) -> bool:  # noqa: PLR6301
        """Whether a ``c_apex=0`` fan's ``∫ c dθ`` converges as ``θ → +∞``.

        True when ``c → 0`` as ``R → ∞`` (so ``base·c → 0`` faster than ``base → ∞``):
        Brooks-Corey, van Genuchten-Mualem, Langmuir, and Freundlich ``n > 1``. The
        only divergent case is Freundlich ``n < 1`` (``c → ∞`` as ``R → ∞``), which
        overrides this to ``False``. Used by the universal temporal fan integrator to
        reject a ``+∞`` upper bound when the integral diverges.
        """
        return True

    def check_entropy_condition(self, c_left: float, c_right: float, shock_speed: float) -> bool:
        """Verify Lax entropy condition in (V, θ) coordinates.

        In θ-space, characteristic speeds are ``λ_θ(C) = 1 / R(C)``, and the
        Lax condition for a physical shock is::

            λ_θ(C_L) ≥ dV_s/dθ ≥ λ_θ(C_R)

        Parameters
        ----------
        c_left : float
            Concentration upstream of shock [mass/volume].
        c_right : float
            Concentration downstream of shock [mass/volume].
        shock_speed : float
            Shock speed dV/dθ.

        Returns
        -------
        satisfies : bool
            True if shock satisfies entropy condition (is physical).
        """
        r_left = float(self.retardation(c_left))
        r_right = float(self.retardation(c_right))
        lambda_left = float("inf") if r_left == 0.0 else 1.0 / r_left
        lambda_right = float("inf") if r_right == 0.0 else 1.0 / r_right

        # A saturated upstream state (λ_left = +∞, e.g. a Mualem-vG wetting front at S_e = 1) is
        # physical; reject only a non-finite shock speed or downstream characteristic, where the
        # Lax test itself is ill-posed.
        if not np.isfinite(shock_speed) or not np.isfinite(lambda_right):
            return False

        finite_left = abs(lambda_left) if np.isfinite(lambda_left) else 0.0
        tolerance = 1e-14 * max(finite_left, abs(lambda_right), abs(shock_speed))

        return bool((lambda_left > shock_speed - tolerance) and (shock_speed > lambda_right - tolerance))


@dataclass
class FreundlichSorption(NonlinearSorption):
    """
    Freundlich sorption isotherm with exact analytical methods.

    The Freundlich isotherm is: s(C) = k_f * C^(1/n)

    where:
    - s is sorbed concentration [mass/mass of solid]
    - C is dissolved concentration [mass/volume of water]
    - k_f is Freundlich coefficient [(volume/mass)^(1/n)]
    - n is Freundlich exponent (dimensionless)

    For n > 1: Higher C travels faster
    For n < 1: Higher C travels slower
    For n = 1: linear (not supported, use ConstantRetardation instead)

    Parameters
    ----------
    k_f : float
        Freundlich coefficient [(m³/kg)^(1/n)]. Must be positive.
    n : float
        Freundlich exponent [-]. Must be positive and != 1.
    bulk_density : float
        Bulk density of porous medium [kg/m³]. Must be positive.
    porosity : float
        Porosity [-]. Must be in (0, 1).
    c_min : float, optional
        Minimum concentration threshold (the dry-soil singularity floor). For
        n>1, prevents infinite retardation as C→0. Default ``1e-12`` for all n.

    Notes
    -----
    The retardation factor is defined as:
        R(C) = 1 + (rho_b/n_por) * ds/dC
             = 1 + (rho_b*k_f)/(n_por*n) * C^((1/n)-1)

    For Freundlich sorption, R depends on C, which creates nonlinear wave behavior.

    For n>1 (higher C travels faster), R(C)→∞ as C→0, which can cause extremely slow
    wave propagation. The c_min parameter prevents this by enforcing a minimum
    concentration, making R(C) finite for all C≥0.

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> r = sorption.retardation(5.0)
    >>> c_back = sorption.concentration_from_retardation(r)
    >>> bool(np.isclose(c_back, 5.0))
    True
    """

    k_f: float
    """Freundlich coefficient [(m³/kg)^(1/n)]."""
    n: float
    """Freundlich exponent [-]."""
    bulk_density: float
    """Bulk density of porous medium [kg/m³]."""
    porosity: float
    """Porosity [-]."""
    c_min: float = 1e-12
    """Minimum concentration threshold to prevent infinite retardation."""
    _ret_coefficient: float = field(init=False, repr=False, compare=False)
    """Cached ``(rho_b*k_f)/(n_por*n)`` — shared by the scalar and array paths."""
    _ret_exponent: float = field(init=False, repr=False, compare=False)
    """Cached ``(1/n) - 1`` retardation exponent."""
    _ct_coefficient: float = field(init=False, repr=False, compare=False)
    """Cached ``(rho_b/n_por)*k_f`` sorbed-mass coefficient."""
    _cfr_inv_exponent: float = field(init=False, repr=False, compare=False)
    """Cached ``1/((1/n) - 1)`` inversion exponent for concentration_from_retardation."""

    def __post_init__(self):
        """Validate parameters after initialization.

        Raises
        ------
        ValueError
            If any parameter is outside its valid range: ``k_f`` <= 0,
            ``n`` <= 0, ``n`` == 1, ``bulk_density`` <= 0, ``porosity``
            outside (0, 1), or ``c_min`` < 0.
        """
        if self.k_f <= 0:
            msg = f"k_f must be positive, got {self.k_f}"
            raise ValueError(msg)
        if self.n <= 0:
            msg = f"n must be positive, got {self.n}"
            raise ValueError(msg)
        if abs(self.n - 1.0) < EPSILON_FREUNDLICH_N:
            msg = "n = 1 (linear case) not supported, use ConstantRetardation instead"
            raise ValueError(msg)
        if self.bulk_density <= 0:
            msg = f"bulk_density must be positive, got {self.bulk_density}"
            raise ValueError(msg)
        if not 0 < self.porosity < 1:
            msg = f"porosity must be in (0, 1), got {self.porosity}"
            raise ValueError(msg)
        if self.c_min < 0:
            msg = f"c_min must be non-negative, got {self.c_min}"
            raise ValueError(msg)

        self._ret_exponent = (1.0 / self.n) - 1.0
        self._ret_coefficient = (self.bulk_density * self.k_f) / (self.porosity * self.n)
        self._ct_coefficient = (self.bulk_density / self.porosity) * self.k_f
        self._cfr_inv_exponent = 1.0 / self._ret_exponent

    def retardation(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Compute retardation factor R(C).

        The retardation factor relates concentration speed to pore water speed in
        (V, θ) coordinates::

            dV/dθ = 1 / R(C)

        For Freundlich sorption::

            R(C) = 1 + (rho_b*k_f)/(n_por*n) * C^((1/n)-1)

        Parameters
        ----------
        c : float or array-like
            Dissolved concentration [mass/volume]. Non-negative.

        Returns
        -------
        r : float or numpy.ndarray
            Retardation factor [-]. Always >= 1.0.

        Notes
        -----
        - For n > 1: R decreases with increasing C (higher C travels faster)
        - For n < 1: R increases with increasing C (higher C travels slower)
        - n<1 with c_min=0: R(0)=1 (no sorption at zero, physically correct)
          because clamping to ``c_min=0`` leaves ``C^((1/n)-1) = 0^positive = 0``.
        - Otherwise: ``c`` is clamped to ``c_min`` before evaluation. This pairs with
          :meth:`total_concentration`, which also clamps to ``c_min``.

        Clamping with ``np.maximum`` before the power keeps a single general path
        for every ``(n, c_min)`` combination and avoids raising the base to a
        fractional power on negative ``c``.

        A pure-float fast path handles the (dominant) scalar case, bit-identical to
        the array path; it falls through to the array expression only when the
        clamped ``c`` is ``0`` (``c_min == 0`` and ``c <= 0``), where numpy's
        ``0.0**exponent`` yields the documented ``inf`` (n>1) / ``0`` (n<1) that a
        pure ``0.0**neg`` would instead raise on.
        """
        if not isinstance(c, np.ndarray):
            cf = float(c)
            c_eff = max(self.c_min, cf)
            if c_eff > 0.0:
                return 1.0 + self._ret_coefficient * (c_eff**self._ret_exponent)
        c_eff = np.maximum(np.asarray(c), self.c_min)
        result = 1.0 + self._ret_coefficient * (c_eff**self._ret_exponent)
        return result if isinstance(c, np.ndarray) else float(result)

    def total_concentration(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Compute total concentration (dissolved + sorbed per unit pore volume).

        Total concentration includes both dissolved and sorbed mass:
            C_total = C + (rho_b/n_por) * s(C)
                    = C + (rho_b/n_por) * k_f * C^(1/n)

        Parameters
        ----------
        c : float or array-like
            Dissolved concentration [mass/volume]. Non-negative.

        Returns
        -------
        c_total : float or numpy.ndarray
            Total concentration [mass/volume]. Always >= c.

        Notes
        -----
        This is the conserved quantity in the transport equation:
            ∂C_total/∂t + ∂(flow*C)/∂v = 0

        The flux term only includes dissolved concentration because sorbed mass
        is immobile.

        For ``c = 0``, ``c^(1/n) = 0`` exactly (no singularity for any
        ``n > 0``), so ``C_T(0) = 0`` is physically correct and no ``c_min``
        clamp is needed here. ``c_min`` is only required to keep
        :meth:`retardation` finite as ``c -> 0`` for ``n > 1``; clamping
        ``total_concentration`` to ``c_min`` would bias Rankine-Hugoniot
        shock speeds when ``c_R = 0`` (e.g. the canonical 0->c->0 pulse).
        Negative ``c`` is clamped to ``0`` defensively.
        """
        if not isinstance(c, np.ndarray):
            cf = float(c)
            c_eff = max(0.0, cf)
            return c_eff + self._ct_coefficient * (c_eff ** (1.0 / self.n))
        c_arr = np.maximum(np.asarray(c), 0.0)
        return c_arr + self._ct_coefficient * (c_arr ** (1.0 / self.n))

    def concentration_from_retardation(self, r: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Invert retardation factor to obtain concentration analytically.

        Given R, solves R = retardation(C) for C. This is used in rarefaction waves
        where the self-similar solution gives R as a function of position and time.

        Parameters
        ----------
        r : float or array-like
            Retardation factor [-]. Must be >= 1.0.

        Returns
        -------
        c : float or numpy.ndarray
            Dissolved concentration [mass/volume]. Non-negative.

        Notes
        -----
        This inverts the relation:
            R = 1 + (rho_b*k_f)/(n_por*n) * C^((1/n)-1)

        The analytical solution is:
            C = [(R-1) * n_por*n / (rho_b*k_f)]^(n/(1-n))

        For n = 1 (linear sorption), the exponent n/(1-n) is undefined, which is
        why linear sorption must use ConstantRetardation class instead.

        Examples
        --------
        >>> sorption = FreundlichSorption(
        ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
        ... )
        >>> r = sorption.retardation(5.0)
        >>> c = sorption.concentration_from_retardation(r)
        >>> bool(np.isclose(c, 5.0, rtol=1e-14))
        True
        """
        # FreundlichSorption.__post_init__ rejects |n-1| < EPSILON_FREUNDLICH_N,
        # so the previous n≈1 guard here was unreachable.
        if not isinstance(r, np.ndarray):
            base = (float(r) - 1.0) / self._ret_coefficient
            if base > 0.0:
                return max(base**self._cfr_inv_exponent, self.c_min)
            return self.c_min

        base = (np.asarray(r) - 1.0) / self._ret_coefficient
        # Mask base to a safe placeholder before exponentiation; NumPy emits
        # RuntimeWarning otherwise for base <= 0 with a fractional exponent.
        safe_base = np.where(base > 0, base, 1.0)
        c = safe_base**self._cfr_inv_exponent
        return np.where(base > 0, np.maximum(c, self.c_min), self.c_min)

    def fan_converges_at_infinity(self) -> bool:
        """Freundlich ``n > 1``: ``c → 0`` as ``R → ∞`` (converges). ``n < 1``: ``c → ∞`` (diverges)."""
        return self.n > 1.0


@dataclass
class ConstantRetardation:
    """
    Constant (linear) retardation model.

    For linear sorption: s(C) = K_d * C
    This gives constant retardation: R(C) = 1 + (rho_b/n_por) * K_d = constant

    This is a special case where concentration-dependent behavior disappears.
    Used for conservative tracers or as approximation for weak sorption.

    Parameters
    ----------
    retardation_factor : float
        Constant retardation factor [-]. Must be >= 1.0.
        R = 1.0 means no retardation (conservative tracer).

    Notes
    -----
    With constant retardation:
    - All concentrations travel at same speed in (V, θ): dV/dθ = 1/R
    - No rarefaction waves form (all concentrations travel together)
    - Shocks occur only at concentration discontinuities at inlet
    - Solution reduces to simple θ-shifting (and then t-shifting via the θ↔t map)

    This is equivalent to a single-pore-volume advective time-shift (the deterministic limit of
    :func:`gwtransport.advection.infiltration_to_extraction`) in the gwtransport package.

    Examples
    --------
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> sorption.retardation(5.0)
    2.0
    >>> sorption.retardation(10.0)
    2.0
    """

    retardation_factor: float
    """Constant retardation factor [-]. Must be >= 1.0."""

    def __post_init__(self):
        """Validate parameters after initialization.

        Raises
        ------
        ValueError
            If ``retardation_factor`` is less than 1.0.
        """
        if self.retardation_factor < 1.0:
            msg = f"retardation_factor must be >= 1.0, got {self.retardation_factor}"
            raise ValueError(msg)

    def retardation(self, c: float) -> float:  # noqa: ARG002
        """
        Return constant retardation factor (independent of concentration).

        Parameters
        ----------
        c : float
            Dissolved concentration (not used for constant retardation).

        Returns
        -------
        r : float
            Constant retardation factor.
        """
        return self.retardation_factor

    def total_concentration(self, c: float) -> float:
        """
        Compute total concentration for linear sorption.

        For constant retardation:
            C_total = C * R

        Parameters
        ----------
        c : float
            Dissolved concentration [mass/volume].

        Returns
        -------
        c_total : float
            Total concentration [mass/volume].
        """
        return c * self.retardation_factor

    def concentration_from_retardation(self, r: float) -> float:
        """
        Not applicable for constant retardation.

        With constant R, all concentrations have the same retardation, so
        inversion is not meaningful. This method raises an error.

        Raises
        ------
        NotImplementedError
            Always raised for constant retardation.
        """
        msg = "concentration_from_retardation not applicable for ConstantRetardation (R is independent of C)"
        raise NotImplementedError(msg)

    def shock_speed(self, c_left: float, c_right: float) -> float:  # noqa: ARG002
        """Compute shock speed dV/dθ for constant retardation.

        With constant R, ``dV/dθ = 1 / R`` for any concentration pair —
        identical to every characteristic speed.

        Parameters
        ----------
        c_left, c_right : float
            Concentrations (unused — kept for ABC compatibility).

        Returns
        -------
        shock_speed : float
            Shock speed dV/dθ = 1/R.
        """
        return 1.0 / self.retardation_factor

    def check_entropy_condition(self, c_left: float, c_right: float, shock_speed: float) -> bool:  # noqa: PLR6301
        """Entropy condition for constant retardation: trivially satisfied.

        With constant R every characteristic speed equals the shock speed in
        θ-space, so the Lax condition holds as an equality regardless of
        ``c_left``/``c_right``.

        Returns
        -------
        satisfies : bool
            Always True.
        """
        del c_left, c_right, shock_speed
        return True


@dataclass
class LangmuirSorption(NonlinearSorption):
    """
    Langmuir sorption isotherm with exact analytical methods.

    The Langmuir isotherm is: s(C) = s_max * C / (K_L + C)

    where:
    - s is sorbed concentration [mass/mass of solid]
    - C is dissolved concentration [mass/volume of water]
    - s_max is maximum sorption capacity [mass/mass of solid]
    - K_L is half-saturation constant [mass/volume]

    Retardation always decreases with C (favorable isotherm), and R(0) is
    finite — unlike Freundlich with n > 1, no minimum concentration threshold
    is needed.

    Parameters
    ----------
    s_max : float
        Maximum sorption capacity [mass/mass of solid]. Must be positive.
    k_l : float
        Half-saturation constant [mass/volume]. Concentration at which
        s = s_max / 2. Must be positive.
    bulk_density : float
        Bulk density of porous medium [kg/m³]. Must be positive.
    porosity : float
        Porosity [-]. Must be in (0, 1).

    See Also
    --------
    FreundlichSorption : Freundlich isotherm (unbounded sorption).
    ConstantRetardation : Linear (constant R) retardation model.
    :ref:`concept-nonlinear-sorption` : Background on nonlinear sorption.

    Notes
    -----
    The retardation factor is defined as:
        R(C) = 1 + (rho_b * s_max * K_L) / (n_por * (K_L + C)^2)

    Key properties:

    - R(0) = 1 + rho_b * s_max / (n_por * K_L) -- finite for all parameters
    - R -> 1 as C -> infinity (all sorption sites saturated)
    - R always decreases with increasing C (higher C travels faster)
    - Shocks form on concentration increases, rarefaction fans on decreases

    Examples
    --------
    >>> sorption = LangmuirSorption(
    ...     s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> r = sorption.retardation(5.0)
    >>> c_back = sorption.concentration_from_retardation(r)
    >>> bool(np.isclose(c_back, 5.0))
    True
    """

    s_max: float
    """Maximum sorption capacity [mass/mass of solid]."""
    k_l: float
    """Half-saturation constant [mass/volume]."""
    bulk_density: float
    """Bulk density of porous medium [kg/m³]."""
    porosity: float
    """Porosity [-]."""

    def __post_init__(self):
        """Validate parameters after initialization.

        Raises
        ------
        ValueError
            If any parameter is outside its valid range: ``s_max`` <= 0,
            ``k_l`` <= 0, ``bulk_density`` <= 0, or ``porosity``
            outside (0, 1).
        """
        if self.s_max <= 0:
            msg = f"s_max must be positive, got {self.s_max}"
            raise ValueError(msg)
        if self.k_l <= 0:
            msg = f"k_l must be positive, got {self.k_l}"
            raise ValueError(msg)
        if self.bulk_density <= 0:
            msg = f"bulk_density must be positive, got {self.bulk_density}"
            raise ValueError(msg)
        if not 0 < self.porosity < 1:
            msg = f"porosity must be in (0, 1), got {self.porosity}"
            raise ValueError(msg)

        self.a_coeff: float = self.bulk_density * self.s_max * self.k_l / self.porosity
        """Lumped retardation constant rho_b * s_max * K_L / n_por."""
        self._ct_coefficient: float = (self.bulk_density / self.porosity) * self.s_max
        """Cached ``(rho_b/n_por)*s_max`` sorbed-mass coefficient (scalar + array paths)."""

    def retardation(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Compute retardation factor R(C).

        For Langmuir sorption:
            R(C) = 1 + A / (K_L + C)²

        where A = rho_b * s_max * K_L / n_por.

        Parameters
        ----------
        c : float or array-like
            Dissolved concentration [mass/volume]. Non-negative.

        Returns
        -------
        r : float or numpy.ndarray
            Retardation factor [-]. Always >= 1.0.

        Notes
        -----
        - R(0) = 1 + rho_b * s_max / (n_por * K_L) — always finite
        - R decreases with increasing C (higher C travels faster)
        - R → 1 as C → ∞ (all sorption sites saturated)
        """
        if not isinstance(c, np.ndarray):
            cf = float(c)
            c_eff = max(0.0, cf)
            return 1.0 + self.a_coeff / (self.k_l + c_eff) ** 2
        c_eff = np.maximum(np.asarray(c), 0.0)
        return 1.0 + self.a_coeff / (self.k_l + c_eff) ** 2

    def total_concentration(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Compute total concentration (dissolved + sorbed per unit pore volume).

        For Langmuir sorption:
            C_total = C + (rho_b / n_por) * s_max * C / (K_L + C)

        Parameters
        ----------
        c : float or array-like
            Dissolved concentration [mass/volume]. Non-negative.

        Returns
        -------
        c_total : float or numpy.ndarray
            Total concentration [mass/volume]. Always >= c.
        """
        if not isinstance(c, np.ndarray):
            cf = float(c)
            c_eff = max(0.0, cf)
            return cf + self._ct_coefficient * c_eff / (self.k_l + c_eff)
        c_arr = np.asarray(c)
        c_eff = np.maximum(c_arr, 0.0)
        return c_arr + self._ct_coefficient * c_eff / (self.k_l + c_eff)

    def concentration_from_retardation(self, r: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Invert retardation factor to obtain concentration analytically.

        Given R, solves R = 1 + A / (K_L + C)² for C:
            C = sqrt(A / (R - 1)) - K_L

        Parameters
        ----------
        r : float or array-like
            Retardation factor [-]. Must be >= 1.0.

        Returns
        -------
        c : float or numpy.ndarray
            Dissolved concentration [mass/volume]. Non-negative.

        Notes
        -----
        For R <= 1, returns 0.0 (unphysical region).
        For R >= R(0) = 1 + A/K_L², returns 0.0 (at or below zero concentration).

        Examples
        --------
        >>> sorption = LangmuirSorption(
        ...     s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3
        ... )
        >>> r = sorption.retardation(5.0)
        >>> c = sorption.concentration_from_retardation(r)
        >>> bool(np.isclose(c, 5.0, rtol=1e-14))
        True
        """
        if not isinstance(r, np.ndarray):
            r_minus_1 = float(r) - 1.0
            if r_minus_1 > 0.0:
                return max(math.sqrt(self.a_coeff / r_minus_1) - self.k_l, 0.0)
            return 0.0

        r_minus_1 = np.asarray(r) - 1.0
        # Mask r_minus_1 to a safe placeholder before division to avoid the
        # RuntimeWarning emitted by np.where's eager evaluation when r == 1.
        safe_r_minus_1 = np.where(r_minus_1 > 0, r_minus_1, 1.0)
        c = np.where(r_minus_1 > 0, np.sqrt(self.a_coeff / safe_r_minus_1) - self.k_l, 0.0)
        return np.maximum(c, 0.0)


@dataclass
class BrooksCoreyConductivity(NonlinearSorption):
    r"""Brooks-Corey unsaturated conductivity recast as a NonlinearSorption.

    Used by :mod:`gwtransport.percolation` to model gravity-driven percolation
    through a thick unsaturated zone via the Kinematic-Wave method. The
    closed-form conductivity curve

    .. math::
        K(\\theta) = K_s \\cdot \\Theta^a, \\qquad
        \\Theta = (\\theta - \\theta_r)/(\\theta_s - \\theta_r), \\qquad
        a = 3 + 2/\\lambda \\;(\\text{Burdine})

    is recast in the framework's ``(C, C_T)`` variables by identifying
    ``C ≡ K`` (the flux variable) and ``C_T ≡ θ - θ_r`` (the conserved
    storage). All three abstract methods have closed forms; ``shock_speed``
    and ``check_entropy_condition`` are inherited unchanged from
    :class:`NonlinearSorption`.

    Parameters
    ----------
    theta_r : float
        Residual volumetric moisture content [-]. Must satisfy
        ``0 <= theta_r < theta_s``.
    theta_s : float
        Saturated volumetric moisture content [-]. Equal to the porosity
        for typical soils. Must satisfy ``theta_r < theta_s < 1``.
    k_s : float
        Saturated hydraulic conductivity [length/time]. Positive.
    brooks_corey_lambda : float
        Pore-size distribution index ``λ`` [-]. Positive. The exponent
        ``a = 3 + 2/λ`` is the Burdine pore-connectivity result. The Mualem
        variant (``L = 0.5``) gives ``a = 2.5 + 2/λ`` and is not implemented;
        a user wanting it can re-derive ``λ`` so the Burdine ``a`` matches the
        desired Mualem exponent.

    See Also
    --------
    VanGenuchtenMualemConductivity : Van Genuchten variant with brentq inversions.
    FreundlichSorption : Power-law sorption isotherm (closed form, analogous shape).
    gwtransport.percolation.root_zone_to_water_table_kinematic_wave : The public wrapper.

    Notes
    -----
    The retardation factor and total-concentration relation are:

    .. math::
        C_T(C) = \\Delta\\theta \\cdot (C/K_s)^{1/a}, \\qquad
        R(C) = (\\Delta\\theta / (a K_s)) \\cdot (C/K_s)^{1/a - 1},

    with ``Δθ = θ_s − θ_r``. Since ``1/a − 1 < 0`` always (``a > 3``),
    ``R(C) → ∞`` as ``C → 0`` (dry-soil singularity). The class clamps ``C``
    to a small floor in ``retardation`` and ``concentration_from_retardation``
    (the same pattern as :class:`FreundlichSorption` with ``n > 1``);
    ``total_concentration`` and the inherited ``shock_speed`` do **not**
    clamp, so the canonical wetting-front shock ``c_R = 0`` produces the
    correct Rankine-Hugoniot velocity.

    Examples
    --------
    >>> sorption = BrooksCoreyConductivity(
    ...     theta_r=0.01, theta_s=0.337, k_s=0.174, brooks_corey_lambda=0.25
    ... )
    >>> r = sorption.retardation(0.05)
    >>> c = sorption.concentration_from_retardation(r)
    >>> bool(np.isclose(c, 0.05, rtol=1e-13))
    True
    """

    theta_r: float
    """Residual volumetric moisture content [-]."""
    theta_s: float
    """Saturated volumetric moisture content [-]."""
    k_s: float
    """Saturated hydraulic conductivity [length/time]."""
    brooks_corey_lambda: float
    """Pore-size distribution index λ [-]."""
    a: float = field(init=False)
    """Exponent ``a = 3 + 2/λ`` (Burdine); set in ``__post_init__``."""
    delta_theta: float = field(init=False)
    """``θ_s − θ_r``; set in ``__post_init__``."""
    _inv_a: float = field(init=False, repr=False, compare=False)
    """Cached ``1/a`` total-concentration exponent (scalar + array paths)."""
    _ret_coefficient: float = field(init=False, repr=False, compare=False)
    """Cached ``Δθ/(a·K_s)`` retardation coefficient."""
    _ret_exponent: float = field(init=False, repr=False, compare=False)
    """Cached ``1/a − 1`` retardation exponent."""
    _cfr_exponent: float = field(init=False, repr=False, compare=False)
    """Cached ``−a/(a−1)`` inversion exponent for concentration_from_retardation."""

    def __post_init__(self) -> None:
        """Validate parameters and derive ``a``, ``delta_theta``.

        Raises
        ------
        ValueError
            If any parameter is outside its valid range.
        """
        if not 0.0 <= self.theta_r < self.theta_s:
            msg = f"theta_r must satisfy 0 <= theta_r < theta_s, got theta_r={self.theta_r}, theta_s={self.theta_s}"
            raise ValueError(msg)
        if not self.theta_s < 1.0:
            msg = f"theta_s must be < 1, got {self.theta_s}"
            raise ValueError(msg)
        if self.k_s <= 0.0:
            msg = f"k_s must be positive, got {self.k_s}"
            raise ValueError(msg)
        if self.brooks_corey_lambda <= 0.0:
            msg = f"brooks_corey_lambda must be positive, got {self.brooks_corey_lambda}"
            raise ValueError(msg)
        self.a = 3.0 + 2.0 / self.brooks_corey_lambda
        self.delta_theta = self.theta_s - self.theta_r
        self._inv_a = 1.0 / self.a
        self._ret_coefficient = self.delta_theta / (self.a * self.k_s)
        self._ret_exponent = 1.0 / self.a - 1.0
        self._cfr_exponent = -self.a / (self.a - 1.0)

    def total_concentration(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """``C_T(C) = Δθ · (C/K_s)^(1/a)``. Returns 0 at C=0 (no clamp)."""
        if not isinstance(c, np.ndarray):
            cf = float(c)
            c_eff = max(0.0, cf)
            return self.delta_theta * (c_eff / self.k_s) ** self._inv_a
        c_arr = np.maximum(np.asarray(c, dtype=float), 0.0)
        return self.delta_theta * (c_arr / self.k_s) ** self._inv_a

    def retardation(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """``R(C) = (Δθ / (a·K_s)) · (C/K_s)^(1/a − 1)``. Clamped at ``_C_MIN``."""
        if not isinstance(c, np.ndarray):
            cf = float(c)
            c_eff = max(_C_MIN, cf)
            return self._ret_coefficient * (c_eff / self.k_s) ** self._ret_exponent
        c_eff = np.maximum(np.asarray(c, dtype=float), _C_MIN)
        return self._ret_coefficient * (c_eff / self.k_s) ** self._ret_exponent

    def concentration_from_retardation(self, r: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """``C = K_s · (R · a · K_s / Δθ)^{−a/(a−1)}``. Result clamped at ``_C_MIN``."""
        if not isinstance(r, np.ndarray):
            base = float(r) * self.a * self.k_s / self.delta_theta
            if base > 0.0:
                return max(self.k_s * base**self._cfr_exponent, _C_MIN)
            return _C_MIN
        base = np.asarray(r, dtype=float) * self.a * self.k_s / self.delta_theta
        safe_base = np.where(base > 0, base, 1.0)
        ratio = safe_base**self._cfr_exponent
        return np.where(base > 0, np.maximum(self.k_s * ratio, _C_MIN), _C_MIN)


@dataclass
class VanGenuchtenMualemConductivity(NonlinearSorption):
    r"""Mualem prediction for the van Genuchten retention curve, recast as NonlinearSorption.

    Used by :mod:`gwtransport.percolation` for Kinematic-Wave percolation
    with the standard Mualem-van Genuchten conductivity curve

    .. math::
        K(\\theta) = K_s \\cdot S_e^L \\cdot
        \\left[1 - \\left(1 - S_e^{1/m}\\right)^m\\right]^2, \\qquad
        S_e = (\\theta - \\theta_r)/(\\theta_s - \\theta_r), \\qquad
        m = 1 - 1/n_\\text{vG}.

    The retention parameter ``α_vG`` is *not* needed for ``K(θ)`` — the
    Kinematic-Wave approximation drops capillary suction, so only the
    ``K(S_e)`` curve matters. The two inversions ``S_e(C)`` and
    ``S_e(R)`` have no closed form; both use ``scipy.optimize.brentq``
    with ``xtol = BRENTQ_XTOL = 1e-14``.

    Parameters
    ----------
    theta_r : float
        Residual volumetric moisture content [-].
    theta_s : float
        Saturated volumetric moisture content [-].
    k_s : float
        Saturated hydraulic conductivity [length/time].
    van_genuchten_n : float
        Shape parameter ``n_vG > 1``. ``m = 1 − 1/n_vG`` is derived.
    mualem_l : float, optional
        Pore-connectivity parameter ``L``. Default 0.5 (standard Mualem).
        Must satisfy ``L >= 0``. Setting ``L = 0`` (Burdine variant) gives
        a closed-form ``S_e(C)`` inverse; ``L != 0`` requires ``brentq``.

    See Also
    --------
    BrooksCoreyConductivity : Brooks-Corey closed-form variant.
    gwtransport.percolation.root_zone_to_water_table_kinematic_wave : The public wrapper.

    Notes
    -----
    The closed-form derivative is

    .. math::
        \\frac{dK_M}{dS_e} = K_s \\cdot S_e^{L-1} \\cdot U \\cdot
        \\left[L \\cdot U + 2 \\cdot S_e^{1/m} \\cdot T^{m-1}\\right],

    with ``T = 1 - S_e^{1/m}`` and ``U = 1 - T^m``. Used for
    ``retardation(C)`` (after solving ``S_e(C)``) and for the brentq
    objective in ``concentration_from_retardation(R)``. The formula is
    inlined at both call sites, not exposed as a separate method.

    ``dK_M/dS_e`` is strictly increasing for every ``n_vG > 1`` and
    ``L ≥ 0`` (the conductivity flux is convex, ``d²K/dS_e² > 0``; proved at
    200-digit precision in ``docs/theory/front_tracking_interactions.md`` §2),
    so the brentq inversions are always well-posed — no monotonicity guard is
    needed. A convex flux also means the transport admits only shocks and
    rarefactions, never compound waves.

    Examples
    --------
    >>> sorption = VanGenuchtenMualemConductivity(
    ...     theta_r=0.01, theta_s=0.337, k_s=0.174, van_genuchten_n=2.28
    ... )
    >>> r = sorption.retardation(0.05)
    >>> c = sorption.concentration_from_retardation(r)
    >>> bool(np.isclose(c, 0.05, rtol=1e-12))
    True
    """

    theta_r: float
    """Residual volumetric moisture content [-]."""
    theta_s: float
    """Saturated volumetric moisture content [-]."""
    k_s: float
    """Saturated hydraulic conductivity [length/time]."""
    van_genuchten_n: float
    """vG shape parameter ``n_vG > 1``."""
    mualem_l: float = 0.5
    """Mualem pore-connectivity ``L``. Default 0.5."""
    m: float = field(init=False)
    """Derived ``m = 1 − 1/n_vG``; set in ``__post_init__``."""
    delta_theta: float = field(init=False)
    """``θ_s − θ_r``; set in ``__post_init__``."""

    def __post_init__(self) -> None:
        """Validate parameters and derive ``m``, ``delta_theta``.

        No convexity/monotonicity guard is needed: ``dK_M/dS_e`` is strictly
        increasing (the flux ``K(S_e)`` is convex, ``d²K/dS_e² > 0``) for every
        ``n_vG > 1`` and ``L ≥ 0`` — proved at 200-digit precision in
        ``docs/theory/front_tracking_interactions.md`` §2 — so the brentq
        inversions are always well-posed.

        Raises
        ------
        ValueError
            If any parameter is outside its valid range.
        """
        if not 0.0 <= self.theta_r < self.theta_s:
            msg = f"theta_r must satisfy 0 <= theta_r < theta_s, got theta_r={self.theta_r}, theta_s={self.theta_s}"
            raise ValueError(msg)
        if not self.theta_s < 1.0:
            msg = f"theta_s must be < 1, got {self.theta_s}"
            raise ValueError(msg)
        if self.k_s <= 0.0:
            msg = f"k_s must be positive, got {self.k_s}"
            raise ValueError(msg)
        if self.van_genuchten_n <= 1.0:
            msg = f"van_genuchten_n must be > 1, got {self.van_genuchten_n}"
            raise ValueError(msg)
        if self.mualem_l < 0.0:
            msg = f"mualem_l must be >= 0, got {self.mualem_l}"
            raise ValueError(msg)
        self.m = 1.0 - 1.0 / self.van_genuchten_n
        self.delta_theta = self.theta_s - self.theta_r

    def _k_se(self, s: float) -> float:
        """``K_M(S_e)`` evaluated at a scalar ``S_e``. Returns 0 at ``S_e = 0``."""
        if s <= 0.0:
            return 0.0
        if s >= 1.0:
            return self.k_s
        t = 1.0 - s ** (1.0 / self.m)
        u = 1.0 - t**self.m
        return self.k_s * s**self.mualem_l * u * u

    def _dk_dse(self, s: float) -> float:
        """Closed-form ``dK_M/dS_e`` at scalar ``S_e``. Inlined at call sites.

        At ``s → 1`` (saturation), ``dK/dS_e`` diverges because ``t^(m-1) → ∞``
        for ``m < 1``. The function returns ``+∞`` at and above ``s = 1`` so that
        ``brentq`` can use ``s = 1`` as a closed upper bracket endpoint.
        """
        if s <= 0.0:
            # Limit form: K vanishes as S^(L + 2/m), so derivative is 0 at S=0.
            return 0.0
        s_pow_inv_m = s ** (1.0 / self.m)
        t = 1.0 - s_pow_inv_m
        if t <= 0.0:
            # Numerical underflow or s ≥ 1 — the dK/dS_e singularity at saturation.
            return float("inf")
        u = 1.0 - t**self.m
        return self.k_s * s ** (self.mualem_l - 1.0) * u * (self.mualem_l * u + 2.0 * s_pow_inv_m * t ** (self.m - 1.0))

    def _se_from_c(self, c: float) -> float:
        """Invert ``K_M(S_e) = c`` for ``S_e``. Closed form for ``mualem_l = 0``; brentq otherwise.

        For the Burdine variant (``L = 0``), ``K_M(S_e) = K_s · [1 − (1 − S_e^{1/m})^m]^2``
        is invertible as ``S_e = (1 − (1 − √(K/K_s))^{1/m})^m`` — completely closed
        form. For ``L ≠ 0`` (default Mualem ``L = 0.5``), no closed-form inverse
        exists; ``scipy.optimize.brentq`` with ``xtol = BRENTQ_XTOL = 1e-14`` is
        used. The brentq call is unavoidable in the Mualem case because the
        ``K_M(S_e)`` function is transcendental.
        """
        c_eff = max(float(c), _C_MIN)
        if c_eff >= self.k_s:
            return 1.0
        if self.mualem_l == 0.0:
            u = (c_eff / self.k_s) ** 0.5  # U = 1 − (1−S_e^{1/m})^m
            one_minus_u = 1.0 - u
            one_minus_q_to_inv_m = one_minus_u ** (1.0 / self.m)
            q = 1.0 - one_minus_q_to_inv_m
            return float(q**self.m)
        return float(brentq(lambda s: self._k_se(s) - c_eff, _C_MIN, 1.0, xtol=BRENTQ_XTOL))  # type: ignore[arg-type]

    def total_concentration(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """``C_T = Δθ · S_e(C)``. Returns 0 at C=0 (no clamp)."""
        is_array = isinstance(c, np.ndarray)
        c_arr = np.maximum(np.asarray(c, dtype=float), 0.0)
        flat = c_arr.ravel()
        se = np.fromiter(
            (self._se_from_c(ci) if ci > 0.0 else 0.0 for ci in flat), dtype=float, count=flat.size
        ).reshape(c_arr.shape)
        result = self.delta_theta * se
        return result if is_array else float(result)

    def retardation(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """``R = Δθ / (dK_M/dS_e)|_{S_e(C)}``. Uses inlined derivative; clamps C at ``_C_MIN``."""
        is_array = isinstance(c, np.ndarray)
        c_arr = np.maximum(np.asarray(c, dtype=float), _C_MIN)
        flat = c_arr.ravel()
        out = np.empty(flat.size, dtype=float)
        for i, ci in enumerate(flat):
            s = self._se_from_c(ci)
            out[i] = self.delta_theta / self._dk_dse(s)
        result = out.reshape(c_arr.shape)
        return result if is_array else float(result)

    def concentration_from_retardation(self, r: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Invert ``R(C) = r``. Solve ``dK_M/dS_e(S_e) = Δθ/r`` via brentq, then ``C = K_M(S_e)``."""
        is_array = isinstance(r, np.ndarray)
        r_arr = np.asarray(r, dtype=float)
        flat = r_arr.ravel()
        out = np.empty(flat.size, dtype=float)
        for i, ri in enumerate(flat):
            s = self._se_from_retardation(float(ri))
            out[i] = max(self._k_se(s), _C_MIN)
        result = out.reshape(r_arr.shape)
        return result if is_array else float(result)

    def _se_from_retardation(self, r: float) -> float:
        """Invert ``dK_M/dS_e(S_e) = Δθ/r`` for ``S_e`` via brentq.

        Single root-find for vG-Mualem; shared by ``concentration_from_retardation``
        and ``c_and_total_from_retardation`` to avoid duplicate brentq calls.
        """
        if r <= 0.0:
            return _C_MIN
        target = self.delta_theta / r
        try:
            return float(brentq(lambda s, tgt=target: self._dk_dse(s) - tgt, _C_MIN, 1.0, xtol=BRENTQ_XTOL))  # type: ignore[arg-type]
        except ValueError:
            return _C_MIN

    def c_and_total_from_retardation(self, r: float) -> tuple[float, float]:
        """Return ``(c, C_T)`` at retardation ``r`` from a SINGLE brentq call.

        Overrides the default base-class implementation (which calls
        ``concentration_from_retardation`` and ``total_concentration``
        separately and ends up doing two independent brentq solves on the same
        underlying equation). Halves the iterative-solver cost in the IBP fan
        integrators.
        """
        s = self._se_from_retardation(r)
        c = max(self._k_se(s), _C_MIN)
        ct = self.delta_theta * s
        return c, ct


SorptionModel = NonlinearSorption | ConstantRetardation
"""Type alias for all sorption models accepted by the front-tracking solver."""


def characteristic_speed(c: float, sorption: SorptionModel) -> float:
    """Compute characteristic speed dV/dθ = 1/R(C).

    In (V, θ) coordinates, every characteristic propagates at a flow-free
    speed determined solely by the local concentration and the sorption
    isotherm.

    Parameters
    ----------
    c : float
        Dissolved concentration [mass/volume].
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    speed : float
        Characteristic speed dV/dθ.

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> s = characteristic_speed(c=5.0, sorption=sorption)
    >>> s > 0
    True
    """
    r = float(sorption.retardation(c))
    return float("inf") if r == 0.0 else 1.0 / r


def characteristic_position(
    c: float,
    sorption: SorptionModel,
    theta_start: float,
    v_start: float,
    theta: float,
) -> float | None:
    """Compute position of a characteristic at cumulative flow θ.

    Characteristics propagate linearly in θ::

        V(θ) = v_start + characteristic_speed(C) * (θ - θ_start)

    Parameters
    ----------
    c : float
        Concentration carried by characteristic [mass/volume].
    sorption : SorptionModel
        Sorption model.
    theta_start : float
        Cumulative flow at which the characteristic starts [m³].
    v_start : float
        Starting position [m³].
    theta : float
        Cumulative flow at which to evaluate position [m³].

    Returns
    -------
    position : float or None
        Position at θ [m³], or None if θ < θ_start.

    Examples
    --------
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> v = characteristic_position(
    ...     c=5.0, sorption=sorption, theta_start=0.0, v_start=0.0, theta=1000.0
    ... )
    >>> bool(np.isclose(v, 500.0))  # v = (1/2) * 1000 = 500
    True
    """
    if theta < theta_start:
        return None

    return v_start + characteristic_speed(c, sorption) * (theta - theta_start)


def compute_first_front_arrival_theta(
    cin: npt.NDArray[np.floating],
    theta_edges: npt.NDArray[np.floating],
    aquifer_pore_volume: float,
    sorption: SorptionModel,
) -> float:
    """Cumulative-flow θ at which ``c_first`` arrives at the outlet (end of spin-up).

    "Arrival" means the θ at which the ``c_first`` *level* is fully present at
    the outlet, ``θ_emit + V·R(c_first)`` for ``n<1`` and
    ``θ_emit + V·C_T(c_first)/c_first`` for ``n>1``/constant retardation.

    .. warning::

       For ``n<1`` with ``c_min > 0`` (default ``c_min = 1e-12`` in
       :class:`FreundlichSorption`), the actual wave emitted is a
       :class:`~gwtransport.fronttracking.waves.RarefactionWave` whose head (``c = c_min ≈ 0``) reaches the
       outlet at θ ≈ ``V·R(c_min) ≈ V`` — *much* earlier than the value this
       function returns (which is the *tail* arrival ``V·R(c_first)``).
       The function returns "tail arrival" semantics: the returned θ is a
       conservative end-of-spin-up where c ≤ c_first everywhere before it.
       Consult the solver event log for the true rarefaction head crossing.

    Parameters
    ----------
    cin : numpy.ndarray
        Inlet concentration [mass/volume].
    theta_edges : numpy.ndarray
        Cumulative-flow edges; length ``len(cin) + 1``.
    aquifer_pore_volume : float
        Total pore volume [m³]. Must be positive.
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    theta_first_arrival : float
        Cumulative-flow θ at which ``c_first`` is fully present at the outlet
        [m³]. Returns ``np.inf`` only if ``cin`` is identically zero.

    Examples
    --------
    >>> cin = np.array([0.0, 10.0] + [10.0] * 10)
    >>> theta_edges = np.arange(0.0, 1300.0, 100.0)  # constant flow=100, dt=1
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> theta_first = compute_first_front_arrival_theta(
    ...     cin, theta_edges, 500.0, sorption
    ... )
    >>> bool(np.isclose(theta_first, 100.0 + 500.0 * 2.0))  # θ_emit + V·R
    True
    """
    nonzero_indices = np.where(cin > 0)[0]
    if len(nonzero_indices) == 0:
        return float(np.inf)

    idx_first = int(nonzero_indices[0])
    c_first = float(cin[idx_first])

    if isinstance(sorption, FreundlichSorption) and sorption.n < 1.0:
        # n<1: the 0→c_first step emits a rarefaction; its tail (c=c_first)
        # reaches the outlet after V·R(c_first) units of cumulative flow.
        target_volume = aquifer_pore_volume * float(sorption.retardation(c_first))
    else:
        # n>1 or constant: R-H shock with speed = c / (C_T(c) - C_T(0));
        # target volume = V · C_T(c_first) / c_first.
        target_volume = aquifer_pore_volume * float(sorption.total_concentration(c_first)) / c_first

    return float(theta_edges[idx_first]) + target_volume
