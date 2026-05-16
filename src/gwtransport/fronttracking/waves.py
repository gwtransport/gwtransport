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

import numpy as np
from scipy.optimize import brentq

from gwtransport.fronttracking.math import (
    FreundlichSorption,
    LangmuirSorption,
    SorptionModel,
)

# Numerical tolerance constants
EPSILON_POSITION = 1e-15  # Tolerance for checking if two positions are equal
DECAYING_SHOCK_U_FLOOR = 1e-300  # Lower bracket bound for brentq on Freundlich u-invariant


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
        """Concentration on the left (upstream) side; equals the carried value."""
        return self.concentration

    def concentration_right(self) -> float:
        """Concentration on the right (downstream) side; equals the carried value."""
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
        """Upstream concentration of the shock."""
        return self.c_left

    def concentration_right(self) -> float:
        """Downstream concentration of the shock."""
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
        """Return ``True`` if ``(v, θ)`` lies between the fan's tail and head."""
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

        # contains_point(v, theta) was True, so the point is geometrically inside
        # the fan. The inverted c may drift by a few ULPs past [c_tail, c_head]
        # — clamp rather than rejecting so callers at the head/tail boundaries
        # get the correct boundary concentration.
        c_lo = min(self.c_tail, self.c_head)
        c_hi = max(self.c_tail, self.c_head)
        return min(max(float(c), c_lo), c_hi)


@dataclass
class DecayingShockWave(Wave):
    r"""Merging shock with closed-form trajectory in θ-space.

    Formed when a rarefaction fan and a shock collide. The shock then has
    one side fed by the fan's self-similar profile (the "decay" side) and
    the other side at the original outer state (the "fixed" side).

    Two collision regimes are supported via ``decay_side``:

    - ``'left'`` (favorable head-collision): Freundlich ``n > 1`` or Langmuir.
      The rarefaction's head (faster) catches a leading shock. After
      collision, the shock's ``c_left`` decays from the rarefaction head
      value toward ``c_fixed`` (the unchanged downstream c_right).
    - ``'right'`` (unfavorable tail-collision, n<1 mirrored): Freundlich
      ``n < 1``. A trailing shock catches the rarefaction's tail. After
      collision, the shock's ``c_right`` decays from the rarefaction tail
      value toward ``c_fixed`` (the unchanged upstream c_left).

    **Closed forms** (``θ_local := θ - theta_origin`` measured from the
    rarefaction apex, ``α := ρ_b · k_f / n_por`` for Freundlich,
    ``u_d := c_decay^(1/n)``):

    - Freundlich, ``c_fixed = 0`` (general ``n > 0``, ``n ≠ 1``):
        invariant ``θ_local · u_d^n = K · (n · u_d^(n-1) + α)``,
        position ``V_s(θ) = v_origin + n · K / u_d(θ)``.
    - Freundlich, ``c_fixed > 0`` and ``n = 2``:
        invariant ``(u_d - u_R)² · θ_local = K · (2 u_d + α)``
        with ``u_R := c_fixed^(1/2)``,
        position ``V_s(θ) = v_origin + 2 K · u_d(θ) / (u_d - u_R)²``.
    - Langmuir, ``c_fixed = 0``:
        invariant ``θ_local · c_d² = K · ((K_L + c_d)² + a)`` with
        ``a := ρ_b · s_max · K_L / n_por``,
        position ``V_s(θ) = v_origin + K · (K_L + c_d)² / c_d²``.

    The invariant constant ``K`` is set in ``__post_init__`` from the
    collision IC ``(theta_start, c_decay_initial)``.

    Parameters
    ----------
    theta_start : float
        Cumulative flow at which the merged wave forms (collision θ) [m³].
    v_start : float
        Position at which the merged wave forms [m³]. Should equal
        ``v_origin + (V_s) at θ=theta_start`` for a fan-consistent
        construction.
    c_decay_initial : float
        Concentration on the decaying side at θ=theta_start [mass/volume].
        Must be strictly positive.
    c_fixed : float
        Concentration on the non-decaying side [mass/volume]. Constant in θ.
        Non-negative.
    decay_side : str
        ``'left'`` or ``'right'``. See class docstring.
    v_origin : float
        Position of the rarefaction apex [m³].
    theta_origin : float
        Cumulative flow at the rarefaction apex [m³]. Must satisfy
        ``theta_origin < theta_start``.
    sorption : FreundlichSorption or LangmuirSorption
        Sorption model.
    is_active : bool, optional
        Activity flag. Default True.

    Attributes
    ----------
    K : float
        Invariant constant set in ``__post_init__``.

    See Also
    --------
    ShockWave : Linear-θ shock (no decaying side).
    RarefactionWave : Self-similar expansion fan.
    """

    c_decay_initial: float
    """Concentration on the decaying side at θ=theta_start [mass/volume]."""
    c_fixed: float
    """Concentration on the non-decaying side [mass/volume]."""
    decay_side: str
    """``'left'`` (favorable head-collision) or ``'right'`` (n<1 mirrored)."""
    v_origin: float
    """Position of the rarefaction apex [m³]."""
    theta_origin: float
    """Cumulative flow at the rarefaction apex [m³]."""
    sorption: FreundlichSorption | LangmuirSorption
    """Sorption model (Freundlich or Langmuir; nonlinear-only)."""
    K: float = field(init=False)
    """Invariant constant from collision IC; set in ``__post_init__``."""

    def __post_init__(self) -> None:
        """Validate inputs and compute the invariant constant K."""
        if self.decay_side not in {"left", "right"}:
            msg = f"decay_side must be 'left' or 'right', got {self.decay_side!r}"
            raise ValueError(msg)
        if self.c_decay_initial <= 0.0:
            msg = f"c_decay_initial must be positive, got {self.c_decay_initial}"
            raise ValueError(msg)
        if self.c_fixed < 0.0:
            msg = f"c_fixed must be non-negative, got {self.c_fixed}"
            raise ValueError(msg)
        if self.theta_origin >= self.theta_start:
            msg = (
                f"theta_origin ({self.theta_origin}) must be strictly less than "
                f"theta_start ({self.theta_start}); rarefaction apex precedes collision"
            )
            raise ValueError(msg)

        if isinstance(self.sorption, FreundlichSorption):
            # Freundlich c_fixed>0 closed form is currently derived only for n=2.
            if self.c_fixed > 0.0 and not np.isclose(self.sorption.n, 2.0, rtol=1e-12):
                msg = (
                    f"DecayingShockWave with c_fixed > 0 currently supports only Freundlich n=2 "
                    f"and Langmuir; got Freundlich n={self.sorption.n}"
                )
                raise NotImplementedError(msg)
            self.K = _compute_k_freundlich(
                self.sorption,
                self.theta_start - self.theta_origin,
                self.c_decay_initial,
                self.c_fixed,
            )
        elif isinstance(self.sorption, LangmuirSorption):
            self.K = _compute_k_langmuir(
                self.sorption,
                self.theta_start - self.theta_origin,
                self.c_decay_initial,
            )
        else:
            msg = (
                f"DecayingShockWave requires FreundlichSorption or LangmuirSorption, got {type(self.sorption).__name__}"
            )
            raise TypeError(msg)

    def c_decay_at_theta(self, theta: float) -> float | None:
        """Concentration on the decaying side at cumulative flow θ.

        Returns ``None`` for ``θ < theta_start`` or when the wave is inactive.
        """
        if theta < self.theta_start or not self.is_active:
            return None

        theta_local = theta - self.theta_origin

        if isinstance(self.sorption, FreundlichSorption):
            return _c_decay_freundlich(
                self.sorption,
                self.K,
                self.c_decay_initial,
                self.c_fixed,
                self.theta_start - self.theta_origin,
                theta_local,
            )

        if isinstance(self.sorption, LangmuirSorption):
            return _c_decay_langmuir(self.sorption, self.K, theta_local)

        return None

    def position_at_theta(self, theta: float) -> float | None:
        """Shock position ``V_s(θ)`` via closed form.

        Returns ``None`` for ``θ < theta_start`` or when inactive.
        """
        if theta < self.theta_start or not self.is_active:
            return None

        c_d = self.c_decay_at_theta(theta)
        if c_d is None:
            return None

        if isinstance(self.sorption, FreundlichSorption):
            n = self.sorption.n
            u_d = c_d ** (1.0 / n)
            if self.c_fixed == 0.0:
                return float(self.v_origin + n * self.K / u_d)
            u_r = self.c_fixed**0.5
            return float(self.v_origin + 2.0 * self.K * u_d / (u_d - u_r) ** 2)

        if isinstance(self.sorption, LangmuirSorption):
            k_l = self.sorption.k_l
            return float(self.v_origin + self.K * (k_l + c_d) ** 2 / (c_d * c_d))

        return None

    def outlet_crossing_theta(self, v_outlet: float) -> float | None:
        """Cumulative flow at which ``V_s = v_outlet``.

        Returns ``None`` if the outlet is upstream of the wave's birth
        position, the wave is inactive, or no crossing exists in
        ``(theta_start, +∞)``.
        """
        if not self.is_active:
            return None
        if v_outlet <= self.v_start:
            return None

        # V_s is monotonically increasing in θ (positive shock speed); invert
        # via the fan-continuity identity V_s - v_origin = θ_local / R(c_decay)
        # combined with the invariant to eliminate u, then solve for θ.
        if isinstance(self.sorption, FreundlichSorption):
            return _outlet_crossing_freundlich(
                self.sorption,
                self.K,
                self.c_fixed,
                self.v_origin,
                self.theta_origin,
                v_outlet,
            )
        if isinstance(self.sorption, LangmuirSorption):
            return _outlet_crossing_langmuir(
                self.sorption,
                self.K,
                self.v_origin,
                self.theta_origin,
                v_outlet,
            )
        return None

    def mass_after_outlet_arrival(self, v_outlet: float) -> float:  # noqa: PLR6301
        """Mass exiting at ``v_outlet`` from the wave's outlet arrival to ``θ=+∞``.

        After ``V_s`` reaches ``v_outlet``, ``v_outlet`` falls *inside* the fan
        (between the apex and the shock). The c at ``v_outlet`` is then given by
        the fan's self-similar profile ``R(c) = (θ − theta_origin)/(v_outlet −
        v_origin)``, so the mass integral reduces to the fan integral over
        ``[θ_arrival, θ_tail or +∞]``.

        Raises
        ------
        NotImplementedError
            Currently always. The fan integrator lives in ``output.py``
            (``integrate_rarefaction_exact``) and the dispatch site is not
            yet wired; calling this method directly would silently zero out
            the outlet mass contribution.
        """
        del v_outlet
        msg = "mass_after_outlet_arrival is wired via output.py dispatch (not yet implemented)"
        raise NotImplementedError(msg)

    def concentration_left(self) -> float:
        """Concentration on the left (upstream) side at θ=theta_start.

        For ``decay_side='left'`` returns the decaying c at the collision
        moment; for ``decay_side='right'`` returns the fixed side.
        """
        return self.c_decay_initial if self.decay_side == "left" else self.c_fixed

    def concentration_right(self) -> float:
        """Concentration on the right (downstream) side at θ=theta_start.

        For ``decay_side='right'`` returns the decaying c at the collision
        moment; for ``decay_side='left'`` returns the fixed side.
        """
        return self.c_decay_initial if self.decay_side == "right" else self.c_fixed

    def concentration_at_point(self, v: float, theta: float) -> float | None:
        """Concentration at ``(v, θ)`` if controlled by this decaying shock.

        Three regions:

        1. ``v == V_s(θ)`` (within FP): average of decay-side and fixed-side c.
        2. ``v > V_s(θ)`` (downstream): fixed-side c if ``decay_side='left'``;
           decay-side c at θ if ``decay_side='right'``.
        3. ``v < V_s(θ)`` (upstream, inside the fan): the fan's self-similar
           concentration ``R(c) = (θ − theta_origin)/(v − v_origin)``. Outside
           the fan (i.e., the decay-side characteristic from the apex hasn't
           reached v yet at θ), returns ``None``.

        Returns ``None`` for ``θ < theta_start`` or inactive waves.
        """
        if theta < self.theta_start or not self.is_active:
            return None

        v_s = self.position_at_theta(theta)
        if v_s is None:
            return None

        tol = 1e-15 * max(abs(v_s), 1.0)

        if abs(v - v_s) < tol:
            c_decay = self.c_decay_at_theta(theta)
            if c_decay is None:
                return None
            return 0.5 * (c_decay + self.c_fixed)

        if v > v_s + tol:
            if self.decay_side == "left":
                return self.c_fixed
            return self.c_decay_at_theta(theta)

        # v < v_s: inside the fan
        if v <= self.v_origin:
            return None
        r_target = (theta - self.theta_origin) / (v - self.v_origin)
        if r_target <= 1.0:
            return None
        try:
            c_fan = self.sorption.concentration_from_retardation(r_target)
        except NotImplementedError:
            return None
        return float(c_fan)


def _compute_k_freundlich(
    sorption: FreundlichSorption,
    theta_local: float,
    c_decay_initial: float,
    c_fixed: float,
) -> float:
    """Closed-form invariant K for Freundlich DecayingShockWave.

    Derivation: see plan §"Closed-form derivations". For c_fixed=0,
    K = θ_local · u_c^n / (n · u_c^(n-1) + α); for c_fixed>0 (n=2 only),
    K = θ_local · (u_c - u_r)^2 / (2 · u_c + α). Here α = ρ_b · k_f / n_por.

    Parameters
    ----------
    sorption : FreundlichSorption
        Sorption model.
    theta_local : float
        Cumulative flow from rarefaction apex to collision [m³].
    c_decay_initial : float
        Decaying-side concentration at the collision [mass/volume].
    c_fixed : float
        Non-decaying-side concentration [mass/volume].

    Returns
    -------
    float
        Invariant constant K.
    """
    n = sorption.n
    alpha = sorption.bulk_density * sorption.k_f / sorption.porosity
    u_d = c_decay_initial ** (1.0 / n)

    if c_fixed == 0.0:
        return float(theta_local * u_d**n / (n * u_d ** (n - 1.0) + alpha))

    # c_fixed > 0, n=2
    u_r = c_fixed**0.5
    return float(theta_local * (u_d - u_r) ** 2 / (2.0 * u_d + alpha))


def _compute_k_langmuir(
    sorption: LangmuirSorption,
    theta_local: float,
    c_decay_initial: float,
) -> float:
    """Closed-form invariant K for Langmuir DecayingShockWave (c_fixed=0).

    K = θ_local · c_d^2 / ((K_L + c_d)^2 + a) with a = ρ_b · s_max · K_L / n_por.

    Parameters
    ----------
    sorption : LangmuirSorption
        Sorption model.
    theta_local : float
        Cumulative flow from rarefaction apex to collision [m³].
    c_decay_initial : float
        Decaying-side concentration at the collision [mass/volume].

    Returns
    -------
    float
        Invariant constant K.
    """
    return float(theta_local * c_decay_initial**2 / ((sorption.k_l + c_decay_initial) ** 2 + sorption.a_coeff))


def _c_decay_freundlich(
    sorption: FreundlichSorption,
    k_invariant: float,
    c_decay_initial: float,
    c_fixed: float,
    theta_local_collision: float,
    theta_local: float,
) -> float:
    """Invert the Freundlich invariant to get c on the decaying side at θ_local.

    For n=2 c_fixed=0 (quadratic in u): closed form
    ``u = (K + sqrt(K^2 + K·θ_local·α)) / θ_local``. For general n with
    c_fixed=0 (transcendental): brentq on the monotone bracket
    ``(tiny, c_decay_initial^(1/n)]``. For n=2 c_fixed>0 (quadratic in u with u_r):
    closed form with positive sign chosen to give u > u_r as θ_local → ∞.

    Returns
    -------
    float
        Decaying-side concentration c at θ_local.
    """
    n = sorption.n
    alpha = sorption.bulk_density * sorption.k_f / sorption.porosity

    if c_fixed == 0.0:
        if np.isclose(n, 2.0, rtol=1e-12):
            disc = k_invariant * k_invariant + theta_local * k_invariant * alpha
            u = (k_invariant + np.sqrt(disc)) / theta_local
            return float(u * u)
        u_root = _invert_freundlich_cr_zero(k_invariant, c_decay_initial, n, alpha, theta_local_collision, theta_local)
        return float(u_root**n)

    # n=2, c_fixed > 0
    u_r = c_fixed**0.5
    disc = k_invariant * (theta_local * (2.0 * u_r + alpha) + k_invariant)
    u = (u_r * theta_local + k_invariant + np.sqrt(disc)) / theta_local
    return float(u * u)


def _invert_freundlich_cr_zero(
    k_invariant: float,
    c_decay_initial: float,
    n: float,
    alpha: float,
    theta_local_collision: float,
    theta_local: float,
) -> float:
    """Invert ``θ_local · u^n = K · (n·u^(n-1) + α)`` for u via brentq.

    Returns
    -------
    float
        Root u of the invariant at θ_local.
    """
    u_collision = c_decay_initial ** (1.0 / n)

    def f(u: float) -> float:
        return theta_local * u**n - k_invariant * (n * u ** (n - 1.0) + alpha)

    if theta_local <= theta_local_collision:
        # Earlier than (or at) collision: c_decay equals c_decay_initial.
        return u_collision

    u_root = brentq(f, DECAYING_SHOCK_U_FLOOR, u_collision, xtol=1e-15)
    return float(u_root)  # type: ignore[arg-type]


def _c_decay_langmuir(sorption: LangmuirSorption, k_invariant: float, theta_local: float) -> float:
    """Invert the Langmuir invariant ``θ_local · c^2 = K · ((K_L+c)^2 + a)`` for c.

    Expanded: ``(θ_local - K)·c^2 - 2·K·K_L·c - K·(K_L^2 + a) = 0`` (quadratic
    in c). Positive root chosen for c > 0.

    Returns
    -------
    float
        Decaying-side concentration c at θ_local.
    """
    k_l = sorption.k_l
    a_coeff = sorption.a_coeff
    denom = theta_local - k_invariant
    disc = k_invariant * (k_invariant * k_l * k_l + denom * (k_l * k_l + a_coeff))
    return float((k_invariant * k_l + np.sqrt(disc)) / denom)


def _outlet_crossing_freundlich(
    sorption: FreundlichSorption,
    k_invariant: float,
    c_fixed: float,
    v_origin: float,
    theta_origin: float,
    v_outlet: float,
) -> float | None:
    """θ at which a Freundlich DecayingShockWave reaches v_outlet.

    Returns
    -------
    float or None
        Cumulative flow at crossing, or None if no crossing.
    """
    n = sorption.n
    alpha = sorption.bulk_density * sorption.k_f / sorption.porosity
    delta_v = v_outlet - v_origin

    if c_fixed == 0.0:
        u_target = n * k_invariant / delta_v
        if u_target <= 0.0:
            return None
        theta_local = k_invariant * (n * u_target ** (n - 1.0) + alpha) / u_target**n
        return float(theta_origin + theta_local)

    # n=2, c_fixed > 0: V_s - v_origin = 2·K·u / (u - u_r)^2 ⇒ quadratic in u.
    # The plus-sqrt root always satisfies u > u_r for K, alpha, delta_v > 0:
    # the quadratic's roots multiply to u_r² and sum to 2u_r + 2K/delta_v > 2u_r,
    # so exactly one root exceeds u_r and it is the plus-sqrt branch. The
    # minus-sqrt root is the unphysical companion.
    u_r = c_fixed**0.5
    b_coef = -(2.0 * delta_v * u_r + 2.0 * k_invariant)
    c_coef = delta_v * u_r * u_r
    disc = b_coef * b_coef - 4.0 * delta_v * c_coef
    if disc < 0:
        return None
    u_target = (-b_coef + np.sqrt(disc)) / (2.0 * delta_v)
    theta_local = k_invariant * (2.0 * u_target + alpha) / (u_target - u_r) ** 2
    return float(theta_origin + theta_local)


def _outlet_crossing_langmuir(
    sorption: LangmuirSorption,
    k_invariant: float,
    v_origin: float,
    theta_origin: float,
    v_outlet: float,
) -> float | None:
    """θ at which a Langmuir DecayingShockWave reaches v_outlet.

    From V_s - v_origin = K·(K_L + c)^2 / c^2 ⇒ (K_L + c)/c = sqrt(Δv/K) =: ratio,
    so c = K_L/(ratio - 1). Substitute into the invariant for θ_local.

    Returns
    -------
    float or None
        Cumulative flow at crossing, or None if no crossing exists.
    """
    delta_v = v_outlet - v_origin
    ratio = np.sqrt(delta_v / k_invariant)
    if ratio <= 1.0:
        return None
    c_target = sorption.k_l / (ratio - 1.0)
    theta_local = k_invariant * ((sorption.k_l + c_target) ** 2 + sorption.a_coeff) / (c_target * c_target)
    return float(theta_origin + theta_local)
