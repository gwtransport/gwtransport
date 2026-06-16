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

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

from gwtransport.fronttracking.math import (
    _C_MIN,
    BrooksCoreyConductivity,
    FreundlichSorption,
    LangmuirSorption,
    NonlinearSorption,
    SorptionModel,
    characteristic_speed,
)

# Numerical tolerance constants
EPSILON_POSITION = 1e-15  # Tolerance for checking if two positions are equal
DECAYING_SHOCK_U_FLOOR = 1e-300  # Lower bracket bound for brentq on Freundlich u-invariant
DECAYING_SHOCK_BRENTQ_XTOL = (
    1e-14  # brentq absolute tolerance for monotone θ inversions (exhaustion, outlet, numerical)
)


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
    """Whether wave is currently active (in the solver's event-loop sense)."""
    theta_deactivation: float = field(default=float("inf"), kw_only=True)
    """Cumulative flow at which the wave was deactivated (default ``+∞``).

    Historical record set by collision handlers when a wave is replaced
    (e.g., a parent rarefaction superseded by a ``DecayingShockWave``).
    ``is_active = False`` is the "current state" flag the solver uses for
    its event loop; ``theta_deactivation`` is the moment in θ-history when
    the wave stopped contributing. Retrospective queries (any θ in the
    past) must use ``was_active_at(theta)`` instead of ``is_active`` so
    that ``compute_domain_mass`` etc. correctly attribute c at v_outlet
    during the wave's lifetime even after later events have deactivated
    the wave.
    """

    def was_active_at(self, theta: float) -> bool:
        """Whether the wave was active at cumulative flow ``theta`` (geometric truth).

        Use for retrospective queries — ``is_active`` reflects only the
        wave's *current* (post-simulation) state, which is wrong for
        ``compute_domain_mass`` and similar at θ before a deactivation event.

        Parameters
        ----------
        theta : float
            Cumulative flow at which to query historical activity [m³].

        Returns
        -------
        bool
            ``True`` for ``theta_start <= theta < theta_deactivation``.
            A wave constructed with ``is_active=False`` and no recorded
            ``theta_deactivation`` (default ``+∞``) is treated as
            never-active — e.g., synthetic test fixtures that want the
            wave excluded from dispatch entirely.
        """
        if not self.is_active and self.theta_deactivation == float("inf"):
            return False
        return self.theta_start <= theta < self.theta_deactivation

    def deactivate(self, theta: float) -> None:
        """Mark the wave inactive at cumulative flow ``theta`` (collision handler API).

        Sets both ``is_active = False`` (solver event-loop flag) and
        ``theta_deactivation = theta`` (historical record for retrospective
        ``was_active_at`` queries).

        Parameters
        ----------
        theta : float
            Cumulative flow at which the wave is deactivated [m³].
        """
        self.is_active = False
        self.theta_deactivation = theta

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
            Position [m³], or None if θ < θ_start or θ >= theta_deactivation.
            (Past-θ queries respect the wave's historical lifetime; current-state
            queries before deactivation behave identically to the ``is_active``
            check.)
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
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> char = CharacteristicWave(
    ...     theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption
    ... )
    >>> speed = char.speed()
    >>> bool(np.isclose(char.position_at_theta(1000.0), speed * 1000.0))
    True
    """

    concentration: float
    """Constant concentration carried [mass/volume]."""
    sorption: SorptionModel
    """Sorption model determining the speed."""

    def speed(self) -> float:
        """Characteristic speed dV/dθ = 1/R(C) (``+∞`` at a saturated state, R = 0)."""
        return characteristic_speed(self.concentration, self.sorption)

    def position_at_theta(self, theta: float) -> float | None:
        """Position at cumulative flow θ.

        ``V(θ) = v_start + speed * (θ - θ_start)``.
        """
        if not self.was_active_at(theta):
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
        if not self.was_active_at(theta):
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

        # Position-scaled face width (~1 ULP at all positions), matching
        # DecayingShockWave.concentration_at_point; a fixed 1e-15 falls below
        # one ULP for any v_shock > ~1 m³ and degenerates to bit-equality.
        tol = 1e-15 * max(abs(v_shock), 1.0)

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
        """Speed of rarefaction head dV/dθ = 1/R(C_head) (``+∞`` at a saturated state, R = 0)."""
        return characteristic_speed(self.c_head, self.sorption)

    def tail_speed(self) -> float:
        """Speed of rarefaction tail dV/dθ = 1/R(C_tail) (``+∞`` at a saturated state, R = 0)."""
        return characteristic_speed(self.c_tail, self.sorption)

    def head_position_at_theta(self, theta: float) -> float | None:
        """Position of rarefaction head at cumulative flow θ."""
        if not self.was_active_at(theta):
            return None
        return self.v_start + self.head_speed() * (theta - self.theta_start)

    def tail_position_at_theta(self, theta: float) -> float | None:
        """Position of rarefaction tail at cumulative flow θ."""
        if not self.was_active_at(theta):
            return None
        return self.v_start + self.tail_speed() * (theta - self.theta_start)

    def position_at_theta(self, theta: float) -> float | None:
        """Head position (leading edge of rarefaction). Implements abstract Wave method."""
        return self.head_position_at_theta(theta)

    def contains_point(self, v: float, theta: float) -> bool:
        """Return ``True`` if ``(v, θ)`` lies between the fan's tail and head."""
        if theta <= self.theta_start or theta >= self.theta_deactivation:
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
    r"""Merging shock with closed-form (or quadrature) trajectory in θ-space.

    Formed when a rarefaction fan and a shock collide. The shock then has
    one side fed by the fan's self-similar profile (the "decay" side) and
    the other side at the original outer state (the "fixed" side). Valid for
    any :class:`~gwtransport.fronttracking.math.NonlinearSorption`.

    Two collision regimes are supported via ``decay_side``:

    - ``'left'`` (favorable head-collision): the rarefaction's head (faster)
      catches a leading shock. After collision, the shock's ``c_left`` decays
      from the rarefaction head value toward ``c_fan_tail`` (the unchanged
      downstream c_right is ``c_fixed``).
    - ``'right'`` (unfavorable tail-collision, n<1 mirrored): a trailing shock
      catches the rarefaction's tail. After collision, the shock's ``c_right``
      decays from the rarefaction tail value toward ``c_fan_tail`` (the
      unchanged upstream c_left is ``c_fixed``).

    The wave is valid only while ``c_decay ∈ (c_fan_tail, c_decay_initial]``;
    once ``c_decay`` reaches ``c_fan_tail`` the fan is exhausted (see the
    solver's ``DSW_FAN_EXHAUSTED`` event).

    **Dispatch.** ``_c_decay_at_theta_local`` is the single dispatch site
    (position, fan-exhaustion and outlet-crossing all route through it): a
    closed form is used where one exists, otherwise the shared numerical solver
    ``_c_decay_numerical``. No combination raises — any
    :class:`~gwtransport.fronttracking.math.NonlinearSorption` is valid. With
    ``θ_local := θ − theta_origin`` measured from the rarefaction apex,
    ``α := ρ_b · k_f / n_por`` for Freundlich, and ``u_d := c_decay^(1/n)``:

    - Freundlich, ``c_fixed = 0`` (general ``n > 0``, ``n ≠ 1``) — closed form:
      invariant ``θ_local · u_d^n = K · (n · u_d^(n-1) + α)``,
      position ``V_s(θ) = v_origin + n · K / u_d(θ)``.
    - Freundlich, ``c_fixed > 0``, ``n = 2`` and ``c_decay_initial > c_fixed``
      — closed form: invariant ``(u_d - u_R)² · θ_local = K · (2 u_d + α)``
      with ``u_R := c_fixed^(1/2)``,
      position ``V_s(θ) = v_origin + 2 K · u_d(θ) / (u_d - u_R)²``.
      (The ``c_decay_initial < c_fixed`` mirror falls through to numerical.)
    - Langmuir, ``c_fixed = 0`` — closed form:
      invariant ``θ_local · c_d² = K · ((K_L + c_d)² + a)`` with
      ``a := ρ_b · s_max · K_L / n_por``,
      position ``V_s(θ) = v_origin + K · (K_L + c_d)² / c_d²``.
    - Brooks-Corey, ``c_fixed = 0`` — closed form:
      invariant ``θ_local ∝ R(c_decay)^{a/(a−1)}`` (``R·S = 1/a`` constant),
      so ``R(c_d) = R(c0)·(θ_local/θ_local_coll)^{(a−1)/a}``.
    - Every other ``(isotherm, c_fixed)`` combination (Freundlich ``c_fixed>0,
      n≠2``, Langmuir/Brooks-Corey ``c_fixed>0``, any van Genuchten) —
      numerical solver ``_c_decay_numerical``: the decay-agnostic invariant
      ``θ_local(c_d) = θ_local_coll · exp(∫ R'/[(1 − R·S)·R] dc)`` with the
      symmetric secant speed ``S = (c − c_fixed)/(C_T(c) − C_T(c_fixed))`` by
      ``scipy.integrate.quad``, inverted for ``c_d(θ)`` via ``brentq``.

    Every path shares the fan-continuity identity
    ``V_s = v_origin + θ_local / R(c_decay)``, which ``position_at_theta`` and
    ``outlet_crossing_theta`` use uniformly across all isotherms.

    The invariant constant ``K`` (closed-form Freundlich/Langmuir only) is set
    in ``__post_init__`` from the collision IC ``(theta_start, c_decay_initial)``.

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
        Must be non-negative; a fully-drained collision value of ``0`` is
        floored to the shared dry-soil singularity floor ``_C_MIN`` so the
        retardation and secant-speed evaluations stay finite (issue #222).
    c_fixed : float
        Concentration on the non-decaying side [mass/volume]. Constant in θ.
        Non-negative.
    c_fan_tail : float
        Concentration at the fan's far boundary [mass/volume]. The wave is
        valid only while ``c_decay ∈ (c_fan_tail, c_decay_initial]``; at
        ``c_fan_tail`` the fan is exhausted. Non-negative.
    decay_side : str
        ``'left'`` or ``'right'``. See class docstring.
    v_origin : float
        Position of the rarefaction apex [m³].
    theta_origin : float
        Cumulative flow at the rarefaction apex [m³]. Must satisfy
        ``theta_origin < theta_start``.
    sorption : NonlinearSorption
        Sorption model (any concentration-dependent isotherm).
    is_active : bool, optional
        Activity flag. Default True.

    See Also
    --------
    ShockWave : Linear-θ shock (no decaying side).
    RarefactionWave : Self-similar expansion fan.
    """

    c_decay_initial: float
    """Concentration on the decaying side at θ=theta_start [mass/volume]."""
    c_fixed: float
    """Concentration on the non-decaying side [mass/volume]."""
    c_fan_tail: float
    """Concentration at the fan's far boundary [mass/volume]; bounds the decay."""
    decay_side: str
    """``'left'`` (favorable head-collision) or ``'right'`` (n<1 mirrored)."""
    v_origin: float
    """Position of the rarefaction apex [m³]."""
    theta_origin: float
    """Cumulative flow at the rarefaction apex [m³]."""
    sorption: NonlinearSorption
    """Sorption model (any concentration-dependent isotherm)."""
    K: float = field(init=False)
    """Invariant constant set in ``__post_init__`` (closed-form Freundlich ``c_fixed=0``/``n≈2`` and Langmuir
    ``c_fixed=0`` cases; ``nan`` for every numerical case)."""

    def __post_init__(self) -> None:
        """Validate inputs and compute the closed-form invariant K when applicable."""
        if self.decay_side not in {"left", "right"}:
            msg = f"decay_side must be 'left' or 'right', got {self.decay_side!r}"
            raise ValueError(msg)
        if self.c_decay_initial < 0.0:
            msg = f"c_decay_initial must be non-negative, got {self.c_decay_initial}"
            raise ValueError(msg)
        # Floor a fully-drained fan tail (c_decay_initial == 0, #222) to _C_MIN so the
        # retardation and secant-speed evaluations stay finite (package floor convention).
        self.c_decay_initial = max(self.c_decay_initial, _C_MIN)
        if self.c_fixed < 0.0:
            msg = f"c_fixed must be non-negative, got {self.c_fixed}"
            raise ValueError(msg)
        if self.c_fan_tail < 0.0:
            msg = f"c_fan_tail must be non-negative, got {self.c_fan_tail}"
            raise ValueError(msg)
        if self.theta_origin >= self.theta_start:
            msg = (
                f"theta_origin ({self.theta_origin}) must be strictly less than "
                f"theta_start ({self.theta_start}); rarefaction apex precedes collision"
            )
            raise ValueError(msg)

        if not isinstance(self.sorption, NonlinearSorption):
            msg = f"DecayingShockWave requires a NonlinearSorption, got {type(self.sorption).__name__}"
            raise TypeError(msg)

        # Closed where an exact form exists, else the shared numerical solver.
        # K is the closed-form invariant constant; it is set only for the
        # closed-form cases (see ``_uses_freundlich_closed_form`` /
        # ``_uses_langmuir_closed_form``) and left as NaN for every numerical
        # case. The ``_c_decay_at_theta_local`` dispatch mirrors these.
        self.K = float("nan")
        s = self.sorption
        if isinstance(s, FreundlichSorption) and self._uses_freundlich_closed_form():
            self.K = _compute_k_freundlich(
                s,
                self.theta_start - self.theta_origin,
                self.c_decay_initial,
                self.c_fixed,
            )
        elif isinstance(s, LangmuirSorption) and self._uses_langmuir_closed_form():
            self.K = _compute_k_langmuir(
                s,
                self.theta_start - self.theta_origin,
                self.c_decay_initial,
            )

    def _uses_freundlich_closed_form(self) -> bool:
        """Whether the Freundlich closed form (constant ``K``) applies.

        Closed form holds for ``c_fixed = 0`` (general ``n``), and for the
        ``n ≈ 2`` quadratic ONLY when the decaying side starts above the fixed
        side (``c_decay_initial > c_fixed``) — that is the orientation the
        ``+√`` root branch and the ``(u_d − u_R)²`` invariant were derived for.
        The ``c_decay_initial < c_fixed`` mirror (e.g. ``decay_side='right'``)
        has no valid closed form and routes to ``_c_decay_numerical``.
        """
        s = self.sorption
        if not isinstance(s, FreundlichSorption):
            return False
        if self.c_fixed == 0.0:
            return True
        return bool(np.isclose(s.n, 2.0, rtol=1e-12) and self.c_decay_initial > self.c_fixed)

    def _uses_langmuir_closed_form(self) -> bool:
        """Whether the Langmuir closed form (constant ``K``) applies (``c_fixed = 0``)."""
        return isinstance(self.sorption, LangmuirSorption) and self.c_fixed == 0.0

    def c_decay_at_theta(self, theta: float) -> float | None:
        """Concentration on the decaying side at cumulative flow θ.

        Returns ``None`` for ``θ < theta_start`` or when the wave is inactive;
        otherwise delegates to the single per-isotherm dispatch in
        ``_c_decay_at_theta_local``.
        """
        if not self.was_active_at(theta):
            return None
        return self._c_decay_at_theta_local(theta - self.theta_origin)

    def position_at_theta(self, theta: float) -> float | None:
        """Shock position ``V_s(θ)`` via the fan-continuity identity.

        ``V_s = v_origin + θ_local / R(c_decay)`` for every isotherm. Returns
        ``None`` for ``θ < theta_start`` or when inactive.
        """
        if not self.was_active_at(theta):
            return None

        theta_local = theta - self.theta_origin
        c_d = self._c_decay_at_theta_local(theta_local)
        return float(self.v_origin + theta_local / float(self.sorption.retardation(c_d)))

    def theta_at_fan_exhaustion(self) -> float | None:
        """Cumulative flow θ at which ``c_decay`` reaches ``c_fan_tail``.

        ``c_decay(θ)`` is strictly monotone from ``c_decay_initial`` toward
        ``c_fan_tail``, so the exhaustion θ is well-defined. The crossing test is
        orientation-agnostic: it holds for both the shrinking decay
        (``c_decay_initial > c_fan_tail``) and the growing decay
        (``c_decay_initial < c_fan_tail``). Returns ``None`` when ``c_fan_tail``
        is not strictly between ``c_fixed`` and ``c_decay_initial`` — e.g. full drying
        (``c_fan_tail == c_fixed``), where the decay asymptotically merges with
        the fixed state and no finite exhaustion event occurs.

        Returns
        -------
        float or None
            Cumulative flow θ at exhaustion, or ``None`` if not reached.
        """
        # An interior exhaustion needs c_fan_tail strictly between c_fixed and
        # c_decay_initial (orientation-agnostic via min/max). Full drying
        # (c_fan_tail == c_fixed) merges asymptotically with no finite crossing —
        # return None rather than grow the bracket forever (van Genuchten would hang).
        c_lo = min(self.c_fixed, self.c_decay_initial)
        c_hi = max(self.c_fixed, self.c_decay_initial)
        if not (c_lo < self.c_fan_tail < c_hi):
            return None

        theta_local_collision = self.theta_start - self.theta_origin

        def f(theta_local: float) -> float:
            return self._c_decay_at_theta_local(theta_local) - self.c_fan_tail

        # c_decay is monotone in θ_local; find where it crosses c_fan_tail.
        f_lo = f(theta_local_collision)
        if f_lo <= 0.0:
            # Already at/below the tail at collision — exhausted immediately.
            return self.theta_start
        # Seed == theta_local_collision, so reuse f_lo to skip one quad/eval.
        theta_local_exhaust = _invert_monotone_theta_local(f, theta_hi_seed=theta_local_collision, f_seed=f_lo)
        if theta_local_exhaust is None:
            return None
        return self.theta_origin + theta_local_exhaust

    def _c_decay_at_theta_local(self, theta_local: float) -> float:
        """Decaying concentration as a function of ``θ_local`` (apex-relative).

        The SOLE isotherm dispatch site: closed where an exact form exists,
        otherwise the shared numerical solver. The closed forms (Freundlich
        ``c_fixed=0`` or ``n≈2``; Langmuir ``c_fixed=0``; Brooks-Corey
        ``c_fixed=0``) are selected by ``isinstance``; every other
        ``(isotherm, c_fixed)`` combination falls through to
        ``_c_decay_numerical``. Takes ``θ_local`` directly and skips the
        activity check. ``c_decay_at_theta``, ``position_at_theta``,
        ``theta_at_fan_exhaustion`` and ``outlet_crossing_theta`` all route
        through here rather than repeating the dispatch.
        """
        theta_local_collision = self.theta_start - self.theta_origin
        s = self.sorption
        if isinstance(s, FreundlichSorption) and self._uses_freundlich_closed_form():
            return _c_decay_freundlich(
                s, self.K, self.c_decay_initial, self.c_fixed, theta_local_collision, theta_local
            )
        if isinstance(s, LangmuirSorption) and self._uses_langmuir_closed_form():
            return _c_decay_langmuir(s, self.K, theta_local)
        if isinstance(s, BrooksCoreyConductivity) and self.c_fixed == 0.0:
            return _c_decay_brooks_corey(s, self.c_decay_initial, theta_local_collision, theta_local)
        return _c_decay_numerical(
            s, self.c_decay_initial, self.c_fixed, self.c_fan_tail, theta_local_collision, theta_local
        )

    def outlet_crossing_theta(self, v_outlet: float) -> float | None:
        """Cumulative flow at which ``V_s = v_outlet``.

        Returns ``None`` if the outlet is upstream of the wave's birth
        position or no crossing exists in ``(theta_start, +∞)``. The wave's
        current activity flag is not consulted — callers asking
        retrospectively about a historical crossing need the answer regardless
        of subsequent deactivation.

        The closed-form Freundlich/Langmuir cases invert the fan-continuity
        identity ``V_s − v_origin = θ_local / R(c_decay)`` analytically (valid
        only when ``_c_decay_at_theta_local`` itself uses the closed form, so
        the same conditions are mirrored here); every other case inverts the
        monotone ``V_s(θ)`` via ``brentq``.
        """
        if v_outlet <= self.v_start:
            return None

        # V_s is monotonically increasing in θ (positive shock speed); invert
        # via the fan-continuity identity V_s - v_origin = θ_local / R(c_decay)
        # combined with the invariant to eliminate u, then solve for θ.
        s = self.sorption
        if isinstance(s, FreundlichSorption) and self._uses_freundlich_closed_form():
            return _outlet_crossing_freundlich(
                s,
                self.K,
                self.c_fixed,
                self.v_origin,
                self.theta_origin,
                v_outlet,
            )
        if isinstance(s, LangmuirSorption) and self._uses_langmuir_closed_form():
            return _outlet_crossing_langmuir(
                s,
                self.K,
                self.v_origin,
                self.theta_origin,
                v_outlet,
            )
        return self._outlet_crossing_numerical(v_outlet)

    def _outlet_crossing_numerical(self, v_outlet: float) -> float | None:
        """θ at which ``V_s = v_outlet`` for every non-closed-form case.

        ``V_s(θ) = v_origin + θ_local / R(c_decay(θ))`` is monotone increasing;
        invert by ``brentq`` on ``θ_local``.
        """
        theta_local_collision = self.theta_start - self.theta_origin

        def f(theta_local: float) -> float:
            c = self._c_decay_at_theta_local(theta_local)
            return self.v_origin + theta_local / float(self.sorption.retardation(c)) - v_outlet

        f_lo = f(theta_local_collision)
        if f_lo >= 0.0:
            # Already at/past the outlet at collision; the linear-shock guards
            # in the solver handle the duplicate-crossing suppression.
            return self.theta_start
        # Reuse f_lo only when the seed coincides with the collision (≥1.0);
        # otherwise let the helper evaluate f at its own seed.
        theta_hi_seed = max(theta_local_collision, 1.0)
        seed_f = f_lo if theta_local_collision >= 1.0 else None
        theta_local_cross = _invert_monotone_theta_local(f, theta_hi_seed=theta_hi_seed, f_seed=seed_f)
        if theta_local_cross is None:
            return None
        return self.theta_origin + theta_local_cross

    def mass_after_outlet_arrival(self, v_outlet: float) -> float:
        """Mass exiting at ``v_outlet`` from the wave's outlet arrival to ``θ=+∞``.

        Semantics depend on ``decay_side``:

        - ``'left'`` (favorable n>1, Langmuir): after ``V_s`` reaches
          ``v_outlet``, ``v_outlet`` falls *inside* the fan; c at ``v_outlet``
          follows the fan's self-similar profile and asymptotes to
          ``c_fixed``. For ``c_fixed=0`` the +∞ integral converges.
        - ``'right'`` (n<1 mirror): after ``V_s`` reaches ``v_outlet``,
          ``v_outlet`` is *upstream* of the shock; c at ``v_outlet`` jumps to
          ``c_fixed`` and stays there. For ``c_fixed=0`` the +∞ contribution
          is zero.

        Parameters
        ----------
        v_outlet : float
            Outlet position [m³].

        Returns
        -------
        float
            Total mass exiting at v_outlet from theta_arrival to +∞.

        Raises
        ------
        ValueError
            If ``c_fixed > 0`` (mass integral diverges; use segment-based
            mass functions over a finite upper bound instead).
        """
        if self.c_fixed > 0.0:
            msg = (
                f"mass_after_outlet_arrival requires c_fixed=0 for the +∞ integral to converge; "
                f"got c_fixed={self.c_fixed}"
            )
            raise ValueError(msg)
        if self.decay_side == "right":
            return 0.0
        theta_arrival = self.outlet_crossing_theta(v_outlet)
        if theta_arrival is None:
            return 0.0
        from gwtransport.fronttracking.output import integrate_fan_exact  # noqa: PLC0415

        return integrate_fan_exact(
            theta_origin=self.theta_origin,
            v_origin=self.v_origin,
            v_outlet=v_outlet,
            theta_start=theta_arrival,
            theta_end=float("inf"),
            sorption=self.sorption,
        )

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
           the fan — i.e. the decay-side characteristic from the apex hasn't
           reached v yet, OR the point lies beyond the ``c_fan_tail`` boundary
           (the fan's far edge) — returns ``None``.

        Returns ``None`` for ``θ < theta_start`` or inactive waves.
        """
        if not self.was_active_at(theta):
            return None

        # Compute the decaying-side concentration once and derive V_s from it
        # (inlining position_at_theta's body) so the shock-face branch can reuse
        # it instead of re-running the numerical-isotherm root-find.
        theta_local = theta - self.theta_origin
        c_d = self._c_decay_at_theta_local(theta_local)
        v_s = float(self.v_origin + theta_local / float(self.sorption.retardation(c_d)))

        tol = 1e-15 * max(abs(v_s), 1.0)

        if abs(v - v_s) < tol:
            return 0.5 * (c_d + self.c_fixed)

        # Region selection depends on decay_side:
        # 'left'  (favorable n>1, Langmuir): fan extends upstream of V_s
        #         (v < V_s), c_fixed downstream (v > V_s).
        # 'right' (n<1 mirror): fan extends downstream of V_s (v > V_s),
        #         c_fixed upstream (v < V_s).
        if self.decay_side == "left":
            v_fan_side = v < v_s - tol
            v_fixed_side = v > v_s + tol
        else:
            v_fan_side = v > v_s + tol
            v_fixed_side = v < v_s - tol

        if v_fixed_side:
            return self.c_fixed

        if not v_fan_side:
            return None  # within tol of shock face — handled above

        # Fan-interior: self-similar profile with apex at (v_origin, theta_origin).
        if v == self.v_origin:
            return None
        r_target = (theta - self.theta_origin) / (v - self.v_origin)
        if r_target <= 1.0:
            return None
        try:
            c_fan = self.sorption.concentration_from_retardation(r_target)
        except NotImplementedError:
            return None
        c_fan = float(c_fan)

        # The fan the DSW controls spans concentrations between the shock face
        # (c_decay, ≤ c_decay_initial) and the fan's far boundary c_fan_tail.
        # A point past c_fan_tail belongs to whatever lies beyond the fan, not
        # to this wave — reject so the fan is not extended past its extent.
        c_lo = min(self.c_fan_tail, self.c_decay_initial)
        c_hi = max(self.c_fan_tail, self.c_decay_initial)
        if c_fan < c_lo - EPSILON_POSITION or c_fan > c_hi + EPSILON_POSITION:
            return None
        return c_fan


def _invert_monotone_theta_local(f, *, theta_hi_seed: float, f_seed: float | None = None) -> float | None:
    """Bracket-then-brentq a monotone ``f(θ_local)`` with a sign change above the seed.

    Shared by ``theta_at_fan_exhaustion`` and ``_outlet_crossing_numerical``:
    both invert a monotone function of ``θ_local`` whose sign at the collision
    is already known to differ from its sign at large ``θ_local``. Geometrically
    grows ``θ_hi`` (``×2``, ≤200 iters) from ``theta_hi_seed`` until ``f`` flips
    sign, then inverts with ``brentq``. Returns ``None`` if no sign change is
    bracketed within the iteration budget.

    Parameters
    ----------
    f : callable
        Monotone residual ``f(θ_local)``; ``f(theta_hi_seed)`` and the far-field
        value must straddle zero. Caller-specific early sentinels
        (already-exhausted / already-past-outlet) are handled by the caller.
    theta_hi_seed : float
        ``θ_local`` lower bracket; the search grows ``θ_hi`` from here.
    f_seed : float, optional
        Pre-evaluated ``f(theta_hi_seed)``; callers that already computed it
        (the numerical path's ``f`` is a ``scipy.quad`` call) pass it to avoid a
        redundant evaluation. ``None`` recomputes it here.

    Returns
    -------
    float or None
        Root ``θ_local`` of ``f``, or ``None`` if not bracketed.
    """
    if f_seed is None:
        f_seed = f(theta_hi_seed)
    theta_hi = theta_hi_seed
    for _ in range(200):
        theta_hi *= 2.0
        if f(theta_hi) * f_seed < 0.0:
            return float(brentq(f, theta_hi_seed, theta_hi, xtol=DECAYING_SHOCK_BRENTQ_XTOL))
    return None


def _c_decay_numerical(
    sorption: NonlinearSorption,
    c_decay_initial: float,
    c_fixed: float,
    c_fan_tail: float,
    theta_local_collision: float,
    theta_local: float,
) -> float:
    r"""Decaying-side concentration for collisions with no closed form.

    Decay-agnostic: the fan-continuity + Rankine-Hugoniot relations do not
    depend on which side decays. The secant speed
    ``S(c) = (c − c_fixed)/(C_T(c) − C_T(c_fixed))`` is symmetric in
    ``(c_decay, c_fixed)``, so the same invariant
    ``θ_local(c) = θ_local_coll · exp(∫_{c0}^{c} R'/[(1 − R·S)·R] dc)`` (with
    ``R'`` by central finite difference and ``scipy.integrate.quad``) holds for
    Freundlich ``c_fixed>0, n≠2``, Langmuir ``c_fixed>0``, Brooks-Corey
    ``c_fixed>0`` and any van Genuchten case alike. The invariant is monotone
    from ``c_decay_initial`` to the reachable boundary, so the root is unique;
    we bracket strictly inside and invert via brentq. The reachable boundary is
    the fan tail ``c_fan_tail`` UNLESS ``c_fixed`` lies strictly between
    ``c_decay_initial`` and ``c_fan_tail`` — then the secant speed has a pole at
    ``c_fixed`` (the shock asymptotes to the fixed state and cannot cross it),
    so ``c_fixed`` is the limit.

    Parameters
    ----------
    sorption : NonlinearSorption
        Sorption model.
    c_decay_initial : float
        Decaying-side concentration at the collision (``c0``) [mass/volume].
    c_fixed : float
        Non-decaying-side concentration [mass/volume].
    c_fan_tail : float
        Concentration at the fan's far boundary [mass/volume]; bounds the decay.
    theta_local_collision : float
        ``θ_local`` at the collision (``θ_start − theta_origin``) [m³].
    theta_local : float
        ``θ_local`` at which to evaluate the decaying concentration [m³].

    Returns
    -------
    float
        Decaying-side concentration ``c`` at ``theta_local``.
    """
    ct_fixed = float(sorption.total_concentration(c_fixed))

    def r_prime(c: float) -> float:
        h = max(1e-9, 1e-7 * abs(c))
        return (float(sorption.retardation(c + h)) - float(sorption.retardation(c - h))) / (2.0 * h)

    def integrand(c_eval: float) -> float:
        r = float(sorption.retardation(c_eval))
        ct = float(sorption.total_concentration(c_eval))
        s = (c_eval - c_fixed) / (ct - ct_fixed)
        return r_prime(c_eval) / ((1.0 - r * s) * r)

    def theta_local_of_c(c: float) -> float:
        # The integrand can have an integrable endpoint singularity (where
        # R·S → 1); adaptive quadrature may exhaust its subdivision budget
        # there while still converging. Suppress that benign warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            val, _ = quad(integrand, c_decay_initial, c, limit=200)
        return float(theta_local_collision * np.exp(val))

    if theta_local <= theta_local_collision:
        return c_decay_initial

    # The decaying side relaxes from c_decay_initial toward the first boundary
    # it reaches. If c_fixed lies strictly between c_decay_initial and
    # c_fan_tail, the secant speed S has a pole at c_fixed (R·S → 1 there): the
    # shock asymptotes to the fixed state and CANNOT cross it, so c_fixed — not
    # the more distant c_fan_tail — is the reachable limit. Otherwise the fan's
    # far boundary c_fan_tail bounds the decay.
    c_limit = c_fan_tail
    if (c_decay_initial - c_fixed) * (c_fan_tail - c_fixed) < 0.0:
        c_limit = c_fixed
    lo, hi = sorted((c_decay_initial, c_limit))
    eps = (hi - lo) * 1e-9
    # Bracket endpoints: the c_decay_initial side and the asymptotic-limit side,
    # each held strictly off the boundary (the limit may carry a secant-speed
    # pole).
    c_start = c_decay_initial + (eps if c_limit > c_decay_initial else -eps)
    c_bound = c_limit + (-eps if c_limit > c_decay_initial else eps)

    def f(c: float) -> float:
        return theta_local_of_c(c) - theta_local

    # f is monotone: f(c_start) = θ_local_coll − θ_local < 0 and f → +∞ at the
    # asymptotic limit. If even c_bound has not reached the requested θ_local
    # (asymptote not yet numerically exceeded), the root is past machine
    # resolution of the boundary — clamp to it rather than letting brentq fail.
    if f(c_bound) <= 0.0:
        return float(c_bound)
    return float(brentq(f, min(c_start, c_bound), max(c_start, c_bound), xtol=DECAYING_SHOCK_BRENTQ_XTOL))


def _c_decay_brooks_corey(
    sorption: BrooksCoreyConductivity,
    c_decay_initial: float,
    theta_local_collision: float,
    theta_local: float,
) -> float:
    r"""Brooks-Corey ``c_fixed = 0`` closed form for the decaying-side concentration.

    For Brooks-Corey with ``c_fixed = 0`` the product ``R·S = 1/a`` is constant
    (``a = sorption.a``), so the universal invariant integrates to
    ``θ_local ∝ R(c_decay)^{a/(a−1)}``. Inverting,
    ``R(c_d) = R(c0)·(θ_local/θ_local_coll)^{(a−1)/a}`` and
    ``c_d = concentration_from_retardation(R)``.

    Parameters
    ----------
    sorption : BrooksCoreyConductivity
        Sorption model.
    c_decay_initial : float
        Decaying-side concentration at the collision [mass/volume].
    theta_local_collision : float
        ``θ_local`` at the collision [m³].
    theta_local : float
        ``θ_local`` at which to evaluate the decaying concentration [m³].

    Returns
    -------
    float
        Decaying-side concentration ``c`` at ``theta_local``.
    """
    if theta_local <= theta_local_collision:
        return c_decay_initial
    a = sorption.a
    r0 = float(sorption.retardation(c_decay_initial))
    r_target = r0 * (theta_local / theta_local_collision) ** ((a - 1.0) / a)
    return float(sorption.concentration_from_retardation(r_target))


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
