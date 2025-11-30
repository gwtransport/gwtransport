"""
Wave Representation for Front Tracking.

This module implements wave classes for representing characteristics, shocks,
and rarefaction waves in the front tracking algorithm. Each wave type knows
how to compute its position and concentration at any point in space-time.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption

# Numerical tolerance constants
EPSILON_POSITION = 1e-15  # Tolerance for checking if two positions are equal


@dataclass
class Wave(ABC):
    """
    Abstract base class for all wave types in front tracking.

    All waves share common attributes and must implement methods for
    computing position and concentration. Waves can be active or inactive
    (deactivated waves are preserved for history but don't participate in
    future interactions).

    Parameters
    ----------
    t_start : float
        Time when wave forms [days].
    v_start : float
        Position where wave forms [m³].
    flow : float
        Flow rate at formation time [m³/day].
    is_active : bool, optional
        Whether wave is currently active. Default True.

    Attributes
    ----------
    t_start : float
        Formation time
    v_start : float
        Formation position
    flow : float
        Flow rate
    is_active : bool
        Activity status
    """

    t_start: float
    v_start: float
    flow: float
    is_active: bool = field(default=True, kw_only=True)

    @abstractmethod
    def position_at_time(self, t: float) -> float | None:
        """
        Compute wave position at time t.

        Parameters
        ----------
        t : float
            Time [days].

        Returns
        -------
        position : float or None
            Position [m³], or None if t < t_start or wave is inactive.
        """

    @abstractmethod
    def concentration_left(self) -> float:
        """
        Get concentration on left (upstream) side of wave.

        Returns
        -------
        c_left : float
            Upstream concentration [mass/volume].
        """

    @abstractmethod
    def concentration_right(self) -> float:
        """
        Get concentration on right (downstream) side of wave.

        Returns
        -------
        c_right : float
            Downstream concentration [mass/volume].
        """

    @abstractmethod
    def concentration_at_point(self, v: float, t: float) -> float | None:
        """
        Compute concentration at point (v, t) if wave controls it.

        Parameters
        ----------
        v : float
            Position [m³].
        t : float
            Time [days].

        Returns
        -------
        concentration : float or None
            Concentration [mass/volume] if wave controls this point, None otherwise.
        """


@dataclass
class CharacteristicWave(Wave):
    """
    Characteristic line along which concentration is constant.

    In smooth regions, concentration travels at velocity flow/R(C). Along
    each characteristic line, the concentration value is constant. This is
    the fundamental solution element for hyperbolic conservation laws.

    Parameters
    ----------
    t_start : float
        Formation time [days].
    v_start : float
        Starting position [m³].
    flow : float
        Flow rate [m³/day].
    concentration : float
        Constant concentration carried [mass/volume].
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model determining velocity.
    is_active : bool, optional
        Activity status. Default True.

    Attributes
    ----------
    concentration : float
        Concentration value
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model

    Methods
    -------
    velocity()
        Compute characteristic velocity
    position_at_time(t)
        Get position at time t
    concentration_at_point(v, t)
        Get concentration if point is on characteristic

    Examples
    --------
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> char = CharacteristicWave(
    ...     t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption
    ... )
    >>> char.velocity()
    50.0
    >>> char.position_at_time(10.0)
    500.0
    """

    concentration: float
    sorption: FreundlichSorption | ConstantRetardation

    def velocity(self) -> float:
        """
        Compute characteristic velocity.

        The velocity is v = flow / R(C), where R is the retardation factor.

        Returns
        -------
        velocity : float
            Characteristic velocity [m³/day].
        """
        return float(self.flow / self.sorption.retardation(self.concentration))

    def position_at_time(self, t: float) -> float | None:
        """
        Compute position at time t.

        Characteristics propagate linearly: V(t) = v_start + velocity*(t - t_start).

        Parameters
        ----------
        t : float
            Time [days].

        Returns
        -------
        position : float or None
            Position [m³], or None if t < t_start or inactive.
        """
        if t < self.t_start or not self.is_active:
            return None
        return self.v_start + self.velocity() * (t - self.t_start)

    def concentration_left(self) -> float:
        """Get upstream concentration (same as concentration for characteristics)."""
        return self.concentration

    def concentration_right(self) -> float:
        """Get downstream concentration (same as concentration for characteristics)."""
        return self.concentration

    def concentration_at_point(self, v: float, t: float) -> float | None:
        """
        Get concentration if point is on this characteristic.

        For practical purposes, we check if the characteristic has reached
        position v by time t.

        Parameters
        ----------
        v : float
            Position [m³].
        t : float
            Time [days].

        Returns
        -------
        concentration : float or None
            Concentration if point is on characteristic, None otherwise.

        Notes
        -----
        In practice, this method is used by higher-level algorithms to
        determine which wave controls a given point. The exact point-on-line
        check is handled by the solver.
        """
        v_at_t = self.position_at_time(t)
        if v_at_t is None:
            return None

        # If characteristic has reached or passed this position
        if v_at_t >= v:
            return self.concentration

        return None


@dataclass
class ShockWave(Wave):
    """
    Shock wave (discontinuity) with jump in concentration.

    Shocks form when faster water overtakes slower water, creating a sharp
    front. The shock velocity is determined by the Rankine-Hugoniot condition
    to ensure mass conservation across the discontinuity.

    Parameters
    ----------
    t_start : float
        Formation time [days].
    v_start : float
        Formation position [m³].
    flow : float
        Flow rate [m³/day].
    c_left : float
        Concentration upstream (behind) shock [mass/volume].
    c_right : float
        Concentration downstream (ahead of) shock [mass/volume].
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.
    is_active : bool, optional
        Activity status. Default True.

    Attributes
    ----------
    c_left : float
        Upstream concentration
    c_right : float
        Downstream concentration
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model
    velocity : float
        Shock velocity computed from Rankine-Hugoniot

    Methods
    -------
    position_at_time(t)
        Get shock position at time t
    satisfies_entropy()
        Check if shock satisfies Lax entropy condition
    concentration_at_point(v, t)
        Get concentration based on which side of shock

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> shock = ShockWave(
    ...     t_start=0.0,
    ...     v_start=0.0,
    ...     flow=100.0,
    ...     c_left=10.0,
    ...     c_right=2.0,
    ...     sorption=sorption,
    ... )
    >>> shock.velocity > 0
    True
    >>> shock.satisfies_entropy()
    True
    """

    c_left: float
    c_right: float
    sorption: FreundlichSorption | ConstantRetardation
    velocity: float | None = None

    def __post_init__(self) -> None:
        """Compute shock velocity from Rankine-Hugoniot condition."""
        if self.velocity is None:
            self.velocity = self.sorption.shock_velocity(self.c_left, self.c_right, self.flow)

    def position_at_time(self, t: float) -> float | None:
        """
        Compute shock position at time t.

        Shock propagates at constant velocity: V(t) = v_start + velocity*(t - t_start).

        Parameters
        ----------
        t : float
            Time [days].

        Returns
        -------
        position : float or None
            Position [m³], or None if t < t_start or inactive.
        """
        if t < self.t_start or not self.is_active:
            return None
        if self.velocity is None:
            msg = "Shock velocity should be set in __post_init__"
            raise RuntimeError(msg)
        return self.v_start + self.velocity * (t - self.t_start)

    def concentration_left(self) -> float:
        """Get upstream concentration."""
        return self.c_left

    def concentration_right(self) -> float:
        """Get downstream concentration."""
        return self.c_right

    def concentration_at_point(self, v: float, t: float) -> float | None:
        """
        Get concentration at point based on which side of shock.

        Parameters
        ----------
        v : float
            Position [m³].
        t : float
            Time [days].

        Returns
        -------
        concentration : float or None
            c_left if upstream of shock, c_right if downstream, None if shock hasn't formed yet.

        Notes
        -----
        At the exact shock position, returns the average of left and right values.
        This is a convention for the singular point; in practice, the shock is
        infinitesimally thin.
        """
        v_shock = self.position_at_time(t)
        if v_shock is None:
            return None

        # Tolerance for exact shock position
        tol = 1e-15

        if v < v_shock - tol:
            # Upstream of shock
            return self.c_left
        if v > v_shock + tol:
            # Downstream of shock
            return self.c_right
        # Exactly at shock (rarely happens in practice)
        return 0.5 * (self.c_left + self.c_right)

    def satisfies_entropy(self) -> bool:
        """
        Check if shock satisfies Lax entropy condition.

        The entropy condition ensures characteristics flow INTO the shock
        from both sides, which is required for physical admissibility.

        Returns
        -------
        satisfies : bool
            True if shock satisfies entropy condition.

        Notes
        -----
        The condition is: lambda(c_left) > shock_velocity > lambda(c_right),
        where lambda(C) = flow / R(C) is the characteristic velocity.

        Shocks that violate this condition are unphysical and should be
        replaced by rarefaction waves.
        """
        if self.velocity is None:
            msg = "Shock velocity should be set in __post_init__"
            raise RuntimeError(msg)
        return self.sorption.check_entropy_condition(self.c_left, self.c_right, self.velocity, self.flow)


@dataclass
class RarefactionWave(Wave):
    """
    Rarefaction (expansion fan) with smooth concentration gradient.

    Rarefactions form when slower water follows faster water, creating an
    expanding region where concentration varies smoothly. The solution is
    self-similar: C = C(V/t).

    Parameters
    ----------
    t_start : float
        Formation time [days].
    v_start : float
        Formation position [m³].
    flow : float
        Flow rate [m³/day].
    c_head : float
        Concentration at leading edge (faster) [mass/volume].
    c_tail : float
        Concentration at trailing edge (slower) [mass/volume].
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model (must be concentration-dependent).
    is_active : bool, optional
        Activity status. Default True.

    Attributes
    ----------
    c_head : float
        Head concentration (faster)
    c_tail : float
        Tail concentration (slower)
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model

    Methods
    -------
    head_velocity()
        Velocity of leading edge
    tail_velocity()
        Velocity of trailing edge
    head_position_at_time(t)
        Position of head at time t
    tail_position_at_time(t)
        Position of tail at time t
    contains_point(v, t)
        Check if point is inside rarefaction fan
    concentration_at_point(v, t)
        Compute concentration via self-similar solution

    Raises
    ------
    ValueError
        If head velocity <= tail velocity (would be compression, not rarefaction).

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> raref = RarefactionWave(
    ...     t_start=0.0,
    ...     v_start=0.0,
    ...     flow=100.0,
    ...     c_head=10.0,
    ...     c_tail=2.0,
    ...     sorption=sorption,
    ... )
    >>> raref.head_velocity() > raref.tail_velocity()
    True
    >>> raref.contains_point(v=150.0, t=20.0)
    True
    """

    c_head: float
    c_tail: float
    sorption: FreundlichSorption | ConstantRetardation

    def __post_init__(self):
        """Verify this is actually a rarefaction (head faster than tail)."""
        v_head = self.head_velocity()
        v_tail = self.tail_velocity()

        if v_head <= v_tail:
            msg = (
                f"Not a rarefaction: head_velocity={v_head:.3f} <= tail_velocity={v_tail:.3f}. "
                f"This would be a compression (shock) instead."
            )
            raise ValueError(msg)

    def head_velocity(self) -> float:
        """
        Compute velocity of rarefaction head (leading edge).

        Returns
        -------
        velocity : float
            Head velocity [m³/day].
        """
        return float(self.flow / self.sorption.retardation(self.c_head))

    def tail_velocity(self) -> float:
        """
        Compute velocity of rarefaction tail (trailing edge).

        Returns
        -------
        velocity : float
            Tail velocity [m³/day].
        """
        return float(self.flow / self.sorption.retardation(self.c_tail))

    def head_position_at_time(self, t: float) -> float | None:
        """
        Compute position of rarefaction head at time t.

        Parameters
        ----------
        t : float
            Time [days].

        Returns
        -------
        position : float or None
            Head position [m³], or None if t < t_start or inactive.
        """
        if t < self.t_start or not self.is_active:
            return None
        return self.v_start + self.head_velocity() * (t - self.t_start)

    def tail_position_at_time(self, t: float) -> float | None:
        """
        Compute position of rarefaction tail at time t.

        Parameters
        ----------
        t : float
            Time [days].

        Returns
        -------
        position : float or None
            Tail position [m³], or None if t < t_start or inactive.
        """
        if t < self.t_start or not self.is_active:
            return None
        return self.v_start + self.tail_velocity() * (t - self.t_start)

    def position_at_time(self, t: float) -> float | None:
        """
        Return head position (leading edge of rarefaction).

        This implements the abstract Wave method.

        Parameters
        ----------
        t : float
            Time [days].

        Returns
        -------
        position : float or None
            Head position [m³].
        """
        return self.head_position_at_time(t)

    def contains_point(self, v: float, t: float) -> bool:
        """
        Check if point (v, t) is inside the rarefaction fan.

        Parameters
        ----------
        v : float
            Position [m³].
        t : float
            Time [days].

        Returns
        -------
        contains : bool
            True if point is between tail and head.
        """
        if t <= self.t_start or not self.is_active:
            return False

        v_head = self.head_position_at_time(t)
        v_tail = self.tail_position_at_time(t)

        if v_head is None or v_tail is None:
            return False

        return v_tail <= v <= v_head

    def concentration_left(self) -> float:
        """Get upstream concentration (tail)."""
        return self.c_tail

    def concentration_right(self) -> float:
        """Get downstream concentration (head)."""
        return self.c_head

    def concentration_at_point(self, v: float, t: float) -> float | None:
        """
        Compute concentration at point (v, t) via self-similar solution.

        Within the rarefaction fan, concentration varies smoothly according to:
            R(C) = flow * (t - t_start) / (v - v_start)

        This is inverted to find C using the sorption model.

        Parameters
        ----------
        v : float
            Position [m³].
        t : float
            Time [days].

        Returns
        -------
        concentration : float or None
            Concentration if point is in rarefaction, None otherwise.

        Notes
        -----
        The self-similar solution automatically maintains mass balance and
        provides the exact analytical form of the concentration profile.

        For ConstantRetardation, rarefactions don't form (all concentrations
        travel at same speed), so this returns None.

        Examples
        --------
        >>> sorption = FreundlichSorption(
        ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
        ... )
        >>> raref = RarefactionWave(0.0, 0.0, 100.0, 10.0, 2.0, sorption)
        >>> c = raref.concentration_at_point(v=150.0, t=20.0)
        >>> c is not None
        True
        >>> 2.0 <= c <= 10.0
        True
        """
        # Special case: at origin of rarefaction (before contains_point check)
        if abs(v - self.v_start) < EPSILON_POSITION and t >= self.t_start:
            return self.c_tail

        if not self.contains_point(v, t):
            return None

        # Self-similar solution: R(C) = flow*(t - t_start)/(v - v_start)
        r_target = self.flow * (t - self.t_start) / (v - self.v_start)

        if r_target <= 1.0:
            return None  # Unphysical

        # Invert R to get C
        # For ConstantRetardation, this would raise NotImplementedError
        try:
            c = self.sorption.concentration_from_retardation(r_target)
        except NotImplementedError:
            # ConstantRetardation case - rarefactions don't form
            return None

        # Verify C is in valid range [c_tail, c_head]
        c_min = min(self.c_tail, self.c_head)
        c_max = max(self.c_tail, self.c_head)

        c_float = float(c)
        if c_min <= c_float <= c_max:
            return c_float

        return None
