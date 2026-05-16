"""
Wave Representation for Front Tracking in (V, θ) coordinates.

This module implements wave classes for representing characteristics, shocks,
and rarefaction waves in the front tracking algorithm. Each wave stores its
formation position in cumulative-flow coordinate ``θ = ∫flow(t') dt'`` and
knows how to compute its position at any later θ.

The change from (V, t) to (V, θ) makes every wave velocity a property of the
sorption isotherm alone — flow no longer enters into wave dynamics. Time-
varying flow is absorbed entirely into the θ(t) mapping at the API boundary;
no wave needs recreation when the flow rate changes.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from gwtransport.fronttracking.math import SorptionModel

# Numerical tolerance constants
EPSILON_POSITION = 1e-15  # Tolerance for checking if two positions are equal


@dataclass
class Wave(ABC):
    """Abstract base class for all wave types in front tracking.

    All waves share common attributes and must implement methods for
    computing position and concentration. Waves can be active or inactive
    (deactivated waves are preserved for history but don't participate in
    future interactions).

    Parameters
    ----------
    theta_start : float
        Cumulative flow at which the wave forms [m³].
    v_start : float
        Position at which the wave forms [m³].
    is_active : bool, optional
        Whether wave is currently active. Default True.
    """

    theta_start: float
    """Cumulative flow at which the wave forms [m³]."""
    v_start: float
    """Position at which the wave forms [m³]."""
    is_active: bool = field(default=True, kw_only=True)
    """Whether wave is currently active."""

    @abstractmethod
    def position_at_theta(self, theta: float) -> float | None:
        """Compute wave position at cumulative flow θ.

        Parameters
        ----------
        theta : float
            Cumulative flow [m³].

        Returns
        -------
        position : float or None
            Position [m³], or None if θ < θ_start or the wave is inactive.
        """

    @abstractmethod
    def concentration_left(self) -> float:
        """Concentration on the left (upstream) side of the wave."""

    @abstractmethod
    def concentration_right(self) -> float:
        """Concentration on the right (downstream) side of the wave."""

    @abstractmethod
    def concentration_at_point(self, v: float, theta: float) -> float | None:
        """Compute concentration at point (v, θ) if the wave controls it.

        Returns
        -------
        concentration : float or None
            Concentration [mass/volume] if the wave controls this point, None
            otherwise.
        """


@dataclass
class CharacteristicWave(Wave):
    """Characteristic line along which concentration is constant.

    In smooth regions, concentration travels at speed ``1/R(C)`` in (V, θ)
    coordinates. Along each characteristic line, the concentration value is
    constant. This is the fundamental solution element for hyperbolic
    conservation laws.

    Parameters
    ----------
    theta_start : float
        Formation cumulative flow [m³].
    v_start : float
        Starting position [m³].
    concentration : float
        Constant concentration carried [mass/volume].
    sorption : SorptionModel
        Sorption model determining the speed.
    is_active : bool, optional
        Activity status. Default True.

    Examples
    --------
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> char = CharacteristicWave(
    ...     theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption
    ... )
    >>> char.speed()
    0.5
    >>> char.position_at_theta(1000.0)
    500.0
    """

    concentration: float
    """Constant concentration carried [mass/volume]."""
    sorption: SorptionModel
    """Sorption model determining the speed."""

    def speed(self) -> float:
        """Characteristic speed dV/dθ = 1/R(C)."""
        return float(1.0 / self.sorption.retardation(self.concentration))

    def position_at_theta(self, theta: float) -> float | None:
        """Position at cumulative flow θ.

        ``V(θ) = v_start + speed * (θ - θ_start)``.
        """
        if theta < self.theta_start or not self.is_active:
            return None
        return self.v_start + self.speed() * (theta - self.theta_start)

    def concentration_left(self) -> float:
        return self.concentration

    def concentration_right(self) -> float:
        return self.concentration

    def concentration_at_point(self, v: float, theta: float) -> float | None:
        """Return the carried concentration if the characteristic has reached ``v`` by θ."""
        v_at_theta = self.position_at_theta(theta)
        if v_at_theta is None:
            return None

        if v_at_theta >= v:
            return self.concentration

        return None


@dataclass
class ShockWave(Wave):
    """Shock wave (discontinuity) with jump in concentration.

    Shocks form when faster water overtakes slower water, creating a sharp
    front. In (V, θ) the shock speed is given by the Rankine-Hugoniot
    condition and is independent of flow::

        dV_s/dθ = (C_R - C_L) / (C_T(C_R) - C_T(C_L))

    Parameters
    ----------
    theta_start : float
        Formation cumulative flow [m³].
    v_start : float
        Formation position [m³].
    c_left : float
        Concentration upstream (behind) shock [mass/volume].
    c_right : float
        Concentration downstream (ahead of) shock [mass/volume].
    sorption : SorptionModel
        Sorption model.
    is_active : bool, optional
        Activity status. Default True.
    speed : float, optional
        Shock speed dV/dθ. Computed from Rankine-Hugoniot in ``__post_init__``.

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> shock = ShockWave(
    ...     theta_start=0.0,
    ...     v_start=0.0,
    ...     c_left=10.0,
    ...     c_right=2.0,
    ...     sorption=sorption,
    ... )
    >>> shock.speed > 0
    True
    >>> shock.satisfies_entropy()
    True
    """

    c_left: float
    """Concentration upstream (behind) shock [mass/volume]."""
    c_right: float
    """Concentration downstream (ahead of) shock [mass/volume]."""
    sorption: SorptionModel
    """Sorption model."""
    speed: float = field(init=False)
    """Shock speed dV/dθ; set in ``__post_init__``."""

    def __post_init__(self) -> None:
        """Compute shock speed from Rankine-Hugoniot in (V, θ)."""
        self.speed = self.sorption.shock_speed(self.c_left, self.c_right)

    def position_at_theta(self, theta: float) -> float | None:
        """Position at cumulative flow θ. Shock propagates linearly in θ."""
        if theta < self.theta_start or not self.is_active:
            return None
        return self.v_start + self.speed * (theta - self.theta_start)

    def concentration_left(self) -> float:
        return self.c_left

    def concentration_right(self) -> float:
        return self.c_right

    def concentration_at_point(self, v: float, theta: float) -> float | None:
        """Return c_left if upstream of the shock at θ, c_right if downstream.

        At the exact shock position the average is returned (convention; the
        shock is infinitesimally thin in practice).
        """
        v_shock = self.position_at_theta(theta)
        if v_shock is None:
            return None

        tol = 1e-15

        if v < v_shock - tol:
            return self.c_left
        if v > v_shock + tol:
            return self.c_right
        return 0.5 * (self.c_left + self.c_right)

    def satisfies_entropy(self) -> bool:
        """Check Lax entropy condition in (V, θ): ``λ_θ(C_L) ≥ s ≥ λ_θ(C_R)``."""
        return self.sorption.check_entropy_condition(self.c_left, self.c_right, self.speed)


@dataclass
class RarefactionWave(Wave):
    """Rarefaction (expansion fan) with smooth concentration gradient.

    Rarefactions form when slower water follows faster water, creating an
    expanding region where concentration varies smoothly. In (V, θ) the
    solution is self-similar in ``(V - v_start)`` vs ``(θ - θ_start)``::

        R(C) = (θ - θ_start) / (V - v_start)

    Head and tail propagate at flow-free speeds ``1/R(C_head)`` and
    ``1/R(C_tail)``.

    Parameters
    ----------
    theta_start : float
        Formation cumulative flow [m³].
    v_start : float
        Formation position [m³].
    c_head : float
        Concentration at leading edge (faster) [mass/volume].
    c_tail : float
        Concentration at trailing edge (slower) [mass/volume].
    sorption : SorptionModel
        Sorption model (must be concentration-dependent).
    is_active : bool, optional
        Activity status. Default True.

    Raises
    ------
    ValueError
        If head speed <= tail speed (would be a compression, not a rarefaction).

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> raref = RarefactionWave(
    ...     theta_start=0.0,
    ...     v_start=0.0,
    ...     c_head=10.0,
    ...     c_tail=2.0,
    ...     sorption=sorption,
    ... )
    >>> raref.head_speed() > raref.tail_speed()
    True
    >>> raref.contains_point(v=150.0, theta=2000.0)
    True
    """

    c_head: float
    """Concentration at leading edge (faster) [mass/volume]."""
    c_tail: float
    """Concentration at trailing edge (slower) [mass/volume]."""
    sorption: SorptionModel
    """Sorption model (must be concentration-dependent)."""

    def __post_init__(self):
        """Verify this is a rarefaction (head faster than tail)."""
        s_head = self.head_speed()
        s_tail = self.tail_speed()

        if s_head <= s_tail:
            msg = (
                f"Not a rarefaction: head_speed={s_head:.6g} <= tail_speed={s_tail:.6g}. "
                f"This would be a compression (shock) instead."
            )
            raise ValueError(msg)

    def head_speed(self) -> float:
        """Speed of rarefaction head dV/dθ = 1/R(C_head)."""
        return float(1.0 / self.sorption.retardation(self.c_head))

    def tail_speed(self) -> float:
        """Speed of rarefaction tail dV/dθ = 1/R(C_tail)."""
        return float(1.0 / self.sorption.retardation(self.c_tail))

    def head_position_at_theta(self, theta: float) -> float | None:
        """Position of rarefaction head at cumulative flow θ."""
        if theta < self.theta_start or not self.is_active:
            return None
        return self.v_start + self.head_speed() * (theta - self.theta_start)

    def tail_position_at_theta(self, theta: float) -> float | None:
        """Position of rarefaction tail at cumulative flow θ."""
        if theta < self.theta_start or not self.is_active:
            return None
        return self.v_start + self.tail_speed() * (theta - self.theta_start)

    def position_at_theta(self, theta: float) -> float | None:
        """Head position (leading edge of rarefaction). Implements abstract Wave method."""
        return self.head_position_at_theta(theta)

    def contains_point(self, v: float, theta: float) -> bool:
        """True if (v, θ) lies between the tail and head of the fan."""
        if theta <= self.theta_start or not self.is_active:
            return False

        v_head = self.head_position_at_theta(theta)
        v_tail = self.tail_position_at_theta(theta)

        if v_head is None or v_tail is None:
            return False

        return v_tail <= v <= v_head

    def concentration_left(self) -> float:
        """Upstream concentration is the trailing-edge value c_tail."""
        return self.c_tail

    def concentration_right(self) -> float:
        """Downstream concentration is the leading-edge value c_head."""
        return self.c_head

    def concentration_at_point(self, v: float, theta: float) -> float | None:
        """Self-similar concentration inside the fan: ``R(C) = (θ - θ_start)/(v - v_start)``.

        Outside the fan returns None. For ``ConstantRetardation``, rarefactions
        don't form (all concentrations travel at the same speed), so this also
        returns None.

        Examples
        --------
        >>> sorption = FreundlichSorption(
        ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
        ... )
        >>> raref = RarefactionWave(0.0, 0.0, 10.0, 2.0, sorption)
        >>> c = raref.concentration_at_point(v=150.0, theta=2000.0)
        >>> c is not None
        True
        >>> 2.0 <= c <= 10.0
        True
        """
        if abs(v - self.v_start) < EPSILON_POSITION and theta >= self.theta_start:
            return self.c_tail

        if not self.contains_point(v, theta):
            return None

        r_target = (theta - self.theta_start) / (v - self.v_start)

        if r_target <= 1.0:
            return None  # Unphysical

        try:
            c = self.sorption.concentration_from_retardation(r_target)
        except NotImplementedError:
            # ConstantRetardation case — rarefactions don't form
            return None

        c_min = min(self.c_tail, self.c_head)
        c_max = max(self.c_tail, self.c_head)

        c_float = float(c)
        if c_min <= c_float <= c_max:
            return c_float

        return None
