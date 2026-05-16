"""
Mathematical Foundation for Front Tracking with Nonlinear Sorption.

This module provides exact analytical computations for:
- Freundlich, Langmuir, and constant retardation models
- Shock velocities via Rankine-Hugoniot condition
- Characteristic velocities and positions
- First arrival time calculations
- Entropy condition verification

All computations are exact analytical formulas with no numerical tolerances.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

# Numerical tolerance constants
EPSILON_FREUNDLICH_N = 1e-10  # Tolerance for checking if n ≈ 1.0 (Freundlich constructor rejects this)
EPSILON_DENOMINATOR = 1e-15  # Tolerance for near-zero denominators in shock velocity


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
        raise NotImplementedError

    @abstractmethod
    def total_concentration(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Compute total concentration (dissolved + sorbed per unit pore volume)."""
        raise NotImplementedError

    @abstractmethod
    def concentration_from_retardation(self, r: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """Invert retardation factor to obtain concentration."""
        raise NotImplementedError

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
            avg_retardation = 0.5 * (self.retardation(c_left) + self.retardation(c_right))
            return float(1.0 / avg_retardation)

        return float((c_right - c_left) / denom)

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
        lambda_left = 1.0 / self.retardation(c_left)
        lambda_right = 1.0 / self.retardation(c_right)

        if not np.isfinite(lambda_left) or not np.isfinite(lambda_right) or not np.isfinite(shock_speed):
            return False

        tolerance = 1e-14 * max(abs(lambda_left), abs(lambda_right), abs(shock_speed))

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
        Minimum concentration threshold. For n>1, prevents infinite retardation
        as C→0. Default: 0.1 for n>1, 0.0 for n<1 (set automatically if not provided).

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

    def retardation(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Compute retardation factor R(C).

        The retardation factor relates concentration speed to pore water speed in
        (V, θ) coordinates:
            dV/dθ = 1 / R(C)

        For Freundlich sorption:
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
        - n<1 with c_min=0: R(0)=1 (no sorption at zero, physically correct).
        - Otherwise: ``c`` is clamped to ``c_min`` before evaluation. This pairs with
          :meth:`total_concentration`, which also clamps to ``c_min``.
        """
        is_array = isinstance(c, np.ndarray)
        c_arr = np.asarray(c)

        if self.c_min == 0 and self.n < 1.0:
            result = np.where(c_arr <= 0, 1.0, self._compute_retardation(c_arr))
        else:
            c_eff = np.maximum(c_arr, self.c_min)
            result = self._compute_retardation(c_eff)

        return result if is_array else float(result)

    def _compute_retardation(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute retardation for positive concentrations.

        Parameters
        ----------
        c : numpy.ndarray
            Dissolved concentration [mass/volume]. Must be positive.

        Returns
        -------
        numpy.ndarray
            Retardation factor [-]. Always >= 1.0.
        """
        exponent = (1.0 / self.n) - 1.0
        coefficient = (self.bulk_density * self.k_f) / (self.porosity * self.n)
        return 1.0 + coefficient * (c**exponent)

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

        ``c`` is clamped to ``c_min`` before evaluating the isotherm so that
        :meth:`retardation` and :meth:`total_concentration` agree on the same
        ``c_eff`` (P1.4 Option A). For ``c < c_min`` both methods yield
        ``c_min``-dependent constants; ``dC_tot/dC`` is 0 there, but no shock
        forms entirely within that regime in practice (default ``c_min`` is
        ``1e-12``).
        """
        is_array = isinstance(c, np.ndarray)
        c_arr = np.asarray(c)

        if self.c_min == 0 and self.n < 1.0:
            sorbed = np.where(
                c_arr <= 0, 0.0, (self.bulk_density / self.porosity) * self.k_f * (c_arr ** (1.0 / self.n))
            )
            result = c_arr + sorbed
        else:
            c_eff = np.maximum(c_arr, self.c_min)
            sorbed = (self.bulk_density / self.porosity) * self.k_f * (c_eff ** (1.0 / self.n))
            result = c_eff + sorbed

        return result if is_array else float(result)

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
        is_array = isinstance(r, np.ndarray)
        r_arr = np.asarray(r)

        # FreundlichSorption.__post_init__ rejects |n-1| < EPSILON_FREUNDLICH_N,
        # so the previous n≈1 guard here was unreachable.
        exponent = (1.0 / self.n) - 1.0
        coefficient = (self.bulk_density * self.k_f) / (self.porosity * self.n)
        base = (r_arr - 1.0) / coefficient
        inversion_exponent = 1.0 / exponent

        # Mask base to a safe placeholder before exponentiation; NumPy emits
        # RuntimeWarning otherwise for base <= 0 with a fractional exponent.
        safe_base = np.where(base > 0, base, 1.0)
        c = safe_base**inversion_exponent
        result = np.where(base > 0, np.maximum(c, self.c_min), self.c_min)

        return result if is_array else float(result)


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

    This is equivalent to using `infiltration_to_extraction_series` in the
    gwtransport package.

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

    def check_entropy_condition(  # noqa: ARG002, PLR6301
        self, c_left: float, c_right: float, shock_speed: float
    ) -> bool:
        """Entropy condition for constant retardation: trivially satisfied.

        With constant R every characteristic speed equals the shock speed in
        θ-space, so the Lax condition holds as an equality regardless of
        ``c_left``/``c_right``.

        Returns
        -------
        satisfies : bool
            Always True.
        """
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
        is_array = isinstance(c, np.ndarray)
        c_arr = np.asarray(c)
        c_eff = np.maximum(c_arr, 0.0)
        result = 1.0 + self.a_coeff / (self.k_l + c_eff) ** 2
        return result if is_array else float(result)

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
        is_array = isinstance(c, np.ndarray)
        c_arr = np.asarray(c)
        c_eff = np.maximum(c_arr, 0.0)
        sorbed = (self.bulk_density / self.porosity) * self.s_max * c_eff / (self.k_l + c_eff)
        result = c_arr + sorbed
        return result if is_array else float(result)

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
        is_array = isinstance(r, np.ndarray)
        r_arr = np.asarray(r)

        r_minus_1 = r_arr - 1.0
        # For R <= 1 or very large R, return 0
        c = np.where(r_minus_1 > 0, np.sqrt(self.a_coeff / r_minus_1) - self.k_l, 0.0)
        result = np.maximum(c, 0.0)

        return result if is_array else float(result)


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
    return float(1.0 / sorption.retardation(c))


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


def compute_first_front_arrival_time(
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    sorption: SorptionModel,
) -> float:
    """
    Compute exact time when first wave reaches outlet (v_max).

    This function returns the precise moment when the first non-zero
    concentration wave from the inlet arrives at the outlet. This marks
    the end of the spin-up period.

    The wave type emitted by a step ``0 → c_first`` depends on the sorption
    regime; this function uses the matching velocity:

    - Freundlich ``n > 1`` and ``ConstantRetardation``: the solver emits a
      Rankine-Hugoniot shock. The arrival uses shock velocity
      ``s = flow · c / (C_tot(c) - C_tot(0))``.
    - Freundlich ``n < 1``: the solver emits a single ``CharacteristicWave``
      with velocity ``flow / R(c)``. The arrival uses that velocity.

    Algorithm:
    1. Find first index where cin > 0
    2. Branch on sorption regime; compute the corresponding target cumulative
       flow volume that the leading wave must accumulate to reach the outlet.
    3. Account for piecewise constant flow during transit.
    4. Return arrival time in days from tedges[0].

    Parameters
    ----------
    cin : numpy.ndarray
        Inlet concentration [mass/volume]. Length = len(tedges) - 1.
    flow : numpy.ndarray
        Flow rate [volume/time]. Length = len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges. Length = len(cin) + 1.
        Expected to be DatetimeIndex.
    aquifer_pore_volume : float
        Total pore volume [volume]. Must be positive.
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    t_first_arrival : float
        Time when first wave reaches outlet, measured in days from tedges[0].
        Returns np.inf if no concentration ever arrives.

    Notes
    -----
    The residence time accounts for retardation:
        residence_time = aquifer_pore_volume * R(C) / flow_avg

    For piecewise constant flow, we integrate:
        ∫₀^residence_time flow(t) dt = aquifer_pore_volume * R(C)

    This function computes the EXACT crossing time in days, not a bin edge.

    Examples
    --------
    >>> import pandas as pd
    >>> cin = np.array([0.0, 10.0] + [10.0] * 10)  # First bin zero, then nonzero
    >>> flow = np.array([100.0] * 12)  # Constant flow
    >>> tedges = pd.date_range("2020-01-01", periods=13, freq="D")
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> t_first = compute_first_front_arrival_time(cin, flow, tedges, 500.0, sorption)
    >>> # Result is in days from tedges[0]
    >>> bool(np.isclose(t_first, 11.0))  # 1 day (offset) + 10 days (travel time)
    True
    """
    # Find first non-zero concentration
    nonzero_indices = np.where(cin > 0)[0]

    if len(nonzero_indices) == 0:
        # No concentration ever arrives
        return np.inf

    idx_first = nonzero_indices[0]
    c_first = float(cin[idx_first])

    # Branch on sorption regime; the solver emits a different wave type for the
    # 0 → c_first inlet step. See `create_inlet_waves_at_time` in handlers.py.
    if isinstance(sorption, FreundlichSorption) and sorption.n < 1.0:
        # n<1: solver emits a CharacteristicWave with θ-speed 1/R(c);
        # target cumulative volume = V * R(c).
        target_volume = aquifer_pore_volume * float(sorption.retardation(c_first))
    else:
        # n>1 (Freundlich) or ConstantRetardation: solver creates a R-H shock.
        # Shock velocity from C_tot(0) ≈ 0 (default c_min=1e-12) to C_tot(c_first):
        #   s = flow · c_first / (C_tot(c_first) - C_tot(0))
        # so target_volume = V · C_tot(c_first) / c_first. For ConstantRetardation
        # this coincides with V · R because C_tot(c)/c = R there.
        target_volume = aquifer_pore_volume * float(sorption.total_concentration(c_first)) / c_first

    # Vectorized integration of piecewise-constant flow starting from idx_first.
    # Convert all bin widths to days at once and accumulate the volume profile.
    tedges_days = np.asarray((tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)
    dt_days = np.diff(tedges_days[idx_first:])
    volumes = np.asarray(flow[idx_first:], dtype=float) * dt_days
    cumulative_volume = np.cumsum(volumes)

    # Locate the first bin whose cumulative volume reaches the target.
    bin_offset = int(np.searchsorted(cumulative_volume, target_volume, side="left"))

    if bin_offset >= len(cumulative_volume):
        # Never reaches outlet with given flow history
        return float(np.inf)

    # Volume already accumulated before entering the bin where arrival occurs.
    volume_before_bin = float(cumulative_volume[bin_offset - 1]) if bin_offset > 0 else 0.0
    remaining_volume = target_volume - volume_before_bin
    dt_partial = remaining_volume / float(flow[idx_first + bin_offset])

    return float(float(tedges_days[idx_first + bin_offset]) + dt_partial)
