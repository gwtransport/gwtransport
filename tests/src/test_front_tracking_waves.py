"""
Unit tests for front tracking wave classes in (V, θ) coordinates.

Tests verify wave behavior, position calculations, and concentration queries.
All wave dynamics are in cumulative-flow coordinate θ; speeds are flow-free
(``dV/dθ = 1/R(C)`` for characteristics, Rankine-Hugoniot for shocks).

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    LangmuirSorption,
    VanGenuchtenMualemConductivity,
    characteristic_speed,
)
from gwtransport.fronttracking.waves import (
    CharacteristicWave,
    DecayingShockWave,
    DoubleFanShockWave,
    Feeder,
    RarefactionWave,
    ShockWave,
)


def _invariant_theta_local_of_c(sorption, c_decay_initial, c_fixed, theta_local_collision, c_target):
    """Independent reference for the numerical DSW decay invariant.

    Integrates the decay-agnostic invariant
    ``θ_local(c) = θ_local_coll · exp(∫_{c0}^{c} R'/[(1 − R·S)·R] dc)`` with the
    symmetric secant speed ``S = (c − c_fixed)/(C_T(c) − C_T(c_fixed))`` via
    adaptive ``scipy.integrate.quad`` — a method-independent check on the
    module's composite-quadrature cache (which integrates the same integrand).
    """
    ct_fixed = float(sorption.total_concentration(c_fixed))

    def integrand(c):
        h = max(1e-9, 1e-7 * abs(c))
        rp = (float(sorption.retardation(c + h)) - float(sorption.retardation(c - h))) / (2.0 * h)
        r = float(sorption.retardation(c))
        ct = float(sorption.total_concentration(c))
        s = (c - c_fixed) / (ct - ct_fixed)
        return rp / ((1.0 - r * s) * r)

    val, _ = quad(integrand, c_decay_initial, c_target, limit=400)
    return theta_local_collision * np.exp(val)


class TestCharacteristicWave:
    """Test CharacteristicWave class."""

    def test_initialization(self):
        """Test valid initialization."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)

        assert char.theta_start == 0.0
        assert char.v_start == 0.0
        assert char.concentration == 5.0
        assert char.is_active

    def test_velocity_constant_retardation(self):
        """Speed dV/dθ = 1/R for constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)

        speed = char.speed()
        expected = 1.0 / 2.0

        assert np.isclose(speed, expected, rtol=1e-14)

    def test_velocity_freundlich(self):
        """Speed dV/dθ = 1/R(c) for Freundlich sorption."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=10.0, sorption=sorption)

        speed = char.speed()
        r = sorption.retardation(10.0)
        expected = 1.0 / r

        assert np.isclose(speed, expected, rtol=1e-14)

    def test_position_at_theta_linear_propagation(self):
        """V(θ) = (1/R) · θ for theta_start=0, v_start=0."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)

        for theta in [1.0, 5.0, 10.0]:
            v = char.position_at_theta(theta)
            assert v is not None
            expected = (1.0 / 2.0) * theta
            assert np.isclose(v, expected, rtol=1e-14)

    def test_position_at_theta_before_start(self):
        """Position is None for θ < θ_start."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=10.0, v_start=0.0, concentration=5.0, sorption=sorption)

        v = char.position_at_theta(5.0)
        assert v is None

    def test_position_at_theta_inactive(self):
        """Position is None when wave is inactive."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)
        char.is_active = False

        v = char.position_at_theta(10.0)
        assert v is None

    def test_position_at_theta_nonzero_start(self):
        """V(θ) = v_start + (1/R)·(θ - θ_start)."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=5.0, v_start=100.0, concentration=5.0, sorption=sorption)

        v = char.position_at_theta(15.0)
        speed = 1.0 / 2.0
        expected = 100.0 + speed * (15.0 - 5.0)

        assert v is not None
        assert np.isclose(v, expected, rtol=1e-14)

    def test_concentration_left_right_equal(self):
        """Left and right concentrations are equal along a characteristic."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)

        assert char.concentration_left() == 5.0
        assert char.concentration_right() == 5.0
        assert char.concentration_left() == char.concentration_right()

    def test_concentration_at_point_reached(self):
        """Concentration at a point the characteristic has reached."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)

        # At θ=10, char at v = 0.5·10 = 5.0 → query v ≤ 5 returns the carried c.
        c = char.concentration_at_point(v=3.0, theta=10.0)
        assert c == 5.0

    def test_concentration_at_point_not_reached(self):
        """Concentration is None for v ahead of the characteristic's position."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)

        # At θ=10, char at v=5; query v=600 is far ahead → None.
        c = char.concentration_at_point(v=600.0, theta=10.0)
        assert c is None


class TestShockWave:
    """Test ShockWave class."""

    def test_initialization(self):
        """Test valid initialization."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        assert shock.theta_start == 0.0
        assert shock.v_start == 0.0
        assert shock.c_left == 10.0
        assert shock.c_right == 2.0
        assert shock.is_active

    def test_velocity_computed_in_post_init(self):
        """Shock speed is computed automatically in __post_init__."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        assert shock.speed is not None
        assert shock.speed > 0

    def test_velocity_rankine_hugoniot(self):
        """Shock speed satisfies Rankine-Hugoniot in (V, θ)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        # dV_s/dθ = (c_R - c_L) / (C_T(c_R) - C_T(c_L)) — flow-free.
        c_total_left = sorption.total_concentration(10.0)
        c_total_right = sorption.total_concentration(2.0)
        expected = (2.0 - 10.0) / (c_total_right - c_total_left)

        assert shock.speed is not None
        assert np.isclose(shock.speed, expected, rtol=1e-14)

    def test_position_at_theta_linear_propagation(self):
        """Shock propagates linearly in θ."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        speed = shock.speed
        assert speed is not None
        for theta in [1.0, 5.0, 10.0]:
            v = shock.position_at_theta(theta)
            assert v is not None
            expected = speed * theta
            assert np.isclose(v, expected, rtol=1e-14)

    def test_position_at_theta_before_start(self):
        """Position is None for θ < θ_start."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(theta_start=10.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        v = shock.position_at_theta(5.0)
        assert v is None

    def test_concentration_left_right(self):
        """Left and right concentration getters."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        assert shock.concentration_left() == 10.0
        assert shock.concentration_right() == 2.0

    def test_concentration_at_point_upstream(self):
        """Concentration upstream of shock returns c_left."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        v_shock = shock.position_at_theta(10.0)
        assert v_shock is not None
        c = shock.concentration_at_point(v=v_shock - 10.0, theta=10.0)

        assert c == 10.0

    def test_concentration_at_point_downstream(self):
        """Concentration downstream of shock returns c_right."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        v_shock = shock.position_at_theta(10.0)
        assert v_shock is not None
        c = shock.concentration_at_point(v=v_shock + 10.0, theta=10.0)

        assert c == 2.0

    def test_concentration_at_point_exact_shock_position(self):
        """At the exact shock position returns the average."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        v_shock = shock.position_at_theta(10.0)
        assert v_shock is not None
        c = shock.concentration_at_point(v=v_shock, theta=10.0)

        assert c is not None
        assert np.isclose(c, 0.5 * (10.0 + 2.0), rtol=1e-14)

    def test_satisfies_entropy_physical_shock(self):
        """A physical compression shock satisfies entropy."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption)

        assert shock.satisfies_entropy()

    def test_satisfies_entropy_unphysical_shock(self):
        """An unphysical expansion shock violates entropy."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        # For n > 1, c_left < c_right is backwards (expansion)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=2.0, c_right=10.0, sorption=sorption)

        assert not shock.satisfies_entropy()


class TestRarefactionWave:
    """Test RarefactionWave class."""

    def test_initialization_valid(self):
        """Valid initialization (head faster than tail)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        assert raref.c_head == 10.0
        assert raref.c_tail == 2.0
        assert raref.is_active

    def test_initialization_invalid_velocities(self):
        """Rarefaction with head slower than tail raises."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        # For n > 1, lower C is slower — head_speed < tail_speed.
        with pytest.raises(ValueError, match="Not a rarefaction"):
            RarefactionWave(theta_start=0.0, v_start=0.0, c_head=2.0, c_tail=10.0, sorption=sorption)

    def test_head_tail_velocities(self):
        """Head and tail speeds are 1/R(c_head) and 1/R(c_tail)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        s_head = raref.head_speed()
        s_tail = raref.tail_speed()

        assert s_head > s_tail

        r_head = sorption.retardation(10.0)
        r_tail = sorption.retardation(2.0)
        assert np.isclose(s_head, 1.0 / r_head, rtol=1e-14)
        assert np.isclose(s_tail, 1.0 / r_tail, rtol=1e-14)

    def test_head_tail_positions(self):
        """Head and tail position computations."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        theta = 10.0
        v_head = raref.head_position_at_theta(theta)
        v_tail = raref.tail_position_at_theta(theta)

        assert v_head is not None
        assert v_tail is not None
        assert v_head > v_tail

        expected_head = raref.head_speed() * theta
        expected_tail = raref.tail_speed() * theta
        assert np.isclose(v_head, expected_head, rtol=1e-14)
        assert np.isclose(v_tail, expected_tail, rtol=1e-14)

    def test_position_at_theta_returns_head(self):
        """``position_at_theta`` returns head position."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        v = raref.position_at_theta(10.0)
        v_head = raref.head_position_at_theta(10.0)

        assert v == v_head

    def test_contains_point_inside_fan(self):
        """contains_point is True for (v, θ) inside the fan."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        theta = 20.0
        v_head = raref.head_position_at_theta(theta)
        v_tail = raref.tail_position_at_theta(theta)
        assert v_head is not None
        assert v_tail is not None
        v_mid = 0.5 * (v_head + v_tail)

        assert raref.contains_point(v_mid, theta)

    def test_contains_point_outside_fan(self):
        """contains_point is False before tail or beyond head."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        theta = 20.0
        v_head = raref.head_position_at_theta(theta)
        v_tail = raref.tail_position_at_theta(theta)
        assert v_head is not None
        assert v_tail is not None

        assert not raref.contains_point(v_tail - 10.0, theta)
        assert not raref.contains_point(v_head + 10.0, theta)

    def test_concentration_at_point_self_similar(self):
        """Self-similar fan: R(C) = (θ - θ_start)/(v - v_start)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        # Pick a point inside the fan at θ = 20, v that gives R between R(c_tail) and R(c_head).
        theta = 20.0
        r_tail = float(sorption.retardation(2.0))
        r_head = float(sorption.retardation(10.0))
        # v_head_at_theta = theta/r_head; v_tail_at_theta = theta/r_tail. Pick midpoint.
        v_head = theta / r_head
        v_tail = theta / r_tail
        v = 0.5 * (v_head + v_tail)

        c = raref.concentration_at_point(v, theta)

        assert c is not None
        assert 2.0 <= c <= 10.0

        # Verify self-similar solution: R(C) = θ/v.
        r_target = theta / v
        c_from_r = sorption.concentration_from_retardation(r_target)
        assert np.isclose(c, c_from_r, rtol=1e-14)

    def test_concentration_at_point_outside_fan(self):
        """Concentration is None outside the fan."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        theta = 20.0
        v_head = raref.head_position_at_theta(theta)
        assert v_head is not None

        c = raref.concentration_at_point(v_head + 100.0, theta)
        assert c is None

    def test_concentration_at_origin(self):
        """At v = v_start (origin of fan) returns c_tail."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        c = raref.concentration_at_point(v=0.0, theta=10.0)
        assert c is not None
        assert np.isclose(c, 2.0, rtol=1e-14)

    def test_concentration_left_right(self):
        """concentration_left = c_tail (upstream), concentration_right = c_head (downstream)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        assert raref.concentration_left() == 2.0
        assert raref.concentration_right() == 10.0

    def test_concentration_with_constant_retardation_returns_none(self):
        """RarefactionWave with constant R raises (no rarefaction can form)."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        with pytest.raises(ValueError, match="Not a rarefaction"):
            RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)


class TestDecayingShockWave:
    """Phase 2 step 1 analytical acceptance tests for ``DecayingShockWave``.

    Hand-derived against the closed forms for Freundlich n=2 c_R=0 (fast
    quadratic path) and Freundlich n=3 c_R=0 (brentq path) — the latter
    independently exercises the inversion code so the n=2 specialization
    doesn't mask brentq bugs. Math derivations live in the plan document
    `check-out-the-main-wondrous-alpaca.md` §"Closed-form derivations".

    Parametric coverage (Langmuir, n=0.5 mirrored, c_R>0, mass balance,
    pointwise breakthrough, entropy, multi-pulse, flow-change) is deferred
    to Phase 2 steps 5+ per the plan: those tests would all assert
    closed-form numbers that derive from un-reviewed math here, so the
    plan defers them until ``bas-physics-math-reviewer`` signs off on the
    math-interim pass.
    """

    def test_freundlich_n2_cr_zero_closed_form(self):
        """Closed-form values for Freundlich n=2, c_R=0, hand-computed.

        Sorption: k_f=0.01, n=2, bulk_density=1500, porosity=0.3, so
        alpha = bulk_density * k_f / porosity = 50.

        Collision IC: rarefaction apex at (V=0, theta=1000); shock formed at
        (V=1000/27, theta=1500) with decaying-side c_decay_initial=4 (so
        u_c=2) and c_fixed=0.

        Invariant constant: K = (theta_c - theta_R) * u_c^2 / (2*u_c + alpha)
        = 500 * 4 / 54 = 1000/27.

        Hand-derived "future" state at u_target=1 (c_decay=1):
        theta_local = K*(2u + alpha)/u^2 = (1000/27)*52 = 52000/27, so
        theta = theta_origin + theta_local = 79000/27. Position
        V_s - V_origin = n*K/u = 2*(1000/27)/1 = 2000/27.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        alpha_expected = 50.0
        alpha_actual = sorption.bulk_density * sorption.k_f / sorption.porosity
        assert alpha_actual == alpha_expected

        k_expected = 1000.0 / 27.0
        v_collision = 1000.0 / 27.0  # = n*K/u_c = 2*(1000/27)/2

        wave = DecayingShockWave(
            theta_start=1500.0,
            v_start=v_collision,
            c_decay_initial=4.0,
            c_fixed=0.0,
            c_fan_tail=0.0,
            decay_side="left",
            v_origin=0.0,
            theta_origin=1000.0,
            sorption=sorption,
        )

        # 1. K from collision IC.
        assert pytest.approx(k_expected, rel=1e-14) == wave.K

        # 2. Consistency at theta=theta_start: position and c_decay reproduce inputs.
        assert pytest.approx(v_collision, rel=1e-14) == wave.position_at_theta(1500.0)
        assert pytest.approx(4.0, rel=1e-14) == wave.c_decay_at_theta(1500.0)

        # 3. Hand-derived "future" state at u=1, c_decay=1.
        theta_test = 79000.0 / 27.0
        v_test = 2000.0 / 27.0

        assert pytest.approx(1.0, rel=1e-14) == wave.c_decay_at_theta(theta_test)
        assert pytest.approx(v_test, rel=1e-14) == wave.position_at_theta(theta_test)

        # 4. outlet_crossing_theta inverts V_s(theta) -> theta at v_outlet=v_test.
        theta_at_v = wave.outlet_crossing_theta(v_outlet=v_test)
        assert theta_at_v is not None
        assert pytest.approx(theta_test, rel=1e-13) == theta_at_v

        # 5. The invariant theta_local * u^n = K * (n*u^(n-1) + alpha) holds at
        # both checked points.
        for theta, c in [(1500.0, 4.0), (theta_test, 1.0)]:
            theta_local = theta - 1000.0
            u = c**0.5
            lhs = theta_local * u**2
            rhs = wave.K * (2.0 * u + alpha_expected)
            assert pytest.approx(rhs, rel=1e-14) == lhs

        # 6. concentration_left/right reflect decay_side='left' convention.
        assert wave.concentration_left() == 4.0
        assert wave.concentration_right() == 0.0

    def test_freundlich_n3_cr_zero_brentq_path_invariant(self):
        """For n=3 c_R=0, c_decay_at_theta uses brentq; verify the invariant.

        The n=2 closed-form path bypasses brentq entirely, so the previous
        test does NOT exercise the transcendental inverter. This test picks
        n=3 with the same sorption family and asserts that the invariant
        ``theta_local * u^n = K * (n*u^(n-1) + alpha)`` holds at both the
        collision and a non-trivial later point — independent of how the
        inversion is implemented. A silent bug in the brentq bracket or in
        the n != 2 closed-form K formula would surface here.

        Also uses ``v_origin != 0`` so a v_origin-coefficient bug in the
        c_R=0 position formula (``V_s = v_origin + n*K/u``) is detected.
        """
        sorption = FreundlichSorption(k_f=0.01, n=3.0, bulk_density=1500.0, porosity=0.3)
        alpha_expected = sorption.bulk_density * sorption.k_f / sorption.porosity

        # Construct a fan-consistent collision point. With n=3, u_c = c^(1/3).
        c_decay_initial = 8.0  # u_c = 2
        u_c = 2.0
        # theta_local at collision is free; pick 270 for cleaner numbers.
        theta_local_collision = 270.0
        # K = theta_local * u^n / (n*u^(n-1) + alpha) = 270 * 8 / (3*4 + 50) = 2160 / 62
        k_expected = theta_local_collision * u_c**3 / (3.0 * u_c**2 + alpha_expected)
        v_origin = 50.0  # nonzero to detect v_origin-coefficient bugs
        # V_s = v_origin + n*K/u_c
        v_collision = v_origin + 3.0 * k_expected / u_c

        wave = DecayingShockWave(
            theta_start=270.0 + 1000.0,
            v_start=v_collision,
            c_decay_initial=c_decay_initial,
            c_fixed=0.0,
            c_fan_tail=0.0,
            decay_side="left",
            v_origin=v_origin,
            theta_origin=1000.0,
            sorption=sorption,
        )

        assert pytest.approx(k_expected, rel=1e-14) == wave.K

        # Pick a later theta where brentq must run. theta_local = 1080 (4x collision).
        theta_later = 1000.0 + 1080.0
        c_later = wave.c_decay_at_theta(theta_later)
        assert c_later is not None
        # Invariant must hold to machine precision.
        u_later = c_later ** (1.0 / 3.0)
        lhs = 1080.0 * c_later  # theta_local * u^n
        rhs = wave.K * (3.0 * u_later**2 + alpha_expected)
        assert pytest.approx(rhs, rel=1e-12) == lhs

        # And position is v_origin + n*K/u_later.
        v_later = wave.position_at_theta(theta_later)
        assert v_later is not None
        assert pytest.approx(v_origin + 3.0 * wave.K / u_later, rel=1e-12) == v_later

        # Round-trip: outlet_crossing_theta(v_later) returns theta_later.
        theta_recovered = wave.outlet_crossing_theta(v_outlet=v_later)
        assert theta_recovered is not None
        assert pytest.approx(theta_later, rel=1e-12) == theta_recovered

    def test_post_init_rejects_invalid_inputs(self):
        """``__post_init__`` validates decay_side, theta_origin, c_decay_initial."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        common = {
            "theta_start": 1500.0,
            "v_start": 1000.0 / 27.0,
            "c_decay_initial": 4.0,
            "c_fixed": 0.0,
            "c_fan_tail": 0.0,
            "v_origin": 0.0,
            "theta_origin": 1000.0,
            "sorption": sorption,
        }

        with pytest.raises(ValueError, match="decay_side"):
            DecayingShockWave(decay_side="middle", **common)

        with pytest.raises(ValueError, match="c_decay_initial"):
            DecayingShockWave(**{**common, "decay_side": "left", "c_decay_initial": -1.0})

        with pytest.raises(ValueError, match="c_fixed"):
            DecayingShockWave(**{**common, "decay_side": "left", "c_fixed": -1.0})

        with pytest.raises(ValueError, match="c_fan_tail"):
            DecayingShockWave(**{**common, "decay_side": "left", "c_fan_tail": -1.0})

        with pytest.raises(ValueError, match="theta_origin"):
            DecayingShockWave(**{**common, "decay_side": "left", "theta_origin": 1500.0})

        with pytest.raises(TypeError, match="NonlinearSorption"):
            DecayingShockWave(**{
                **common,
                "decay_side": "left",
                "sorption": ConstantRetardation(retardation_factor=2.0),
            })

    def test_freundlich_n_not_2_with_c_fixed_positive_uses_numerical(self):
        """Freundlich c_fixed>0 with n≠2 has no closed form: numerical fallback (no raise).

        ``K`` is left NaN (numerical case), the collision IC is honoured exactly
        (``c_decay`` at the collision equals ``c_decay_initial``), and the
        decaying side relaxes monotonically from ``c_decay_initial`` toward the
        fixed state.
        """
        sorption = FreundlichSorption(k_f=0.01, n=1.5, bulk_density=1500.0, porosity=0.3)
        dsw = DecayingShockWave(
            theta_start=1500.0,
            v_start=20.0,
            c_decay_initial=4.0,
            c_fixed=1.0,
            c_fan_tail=1.0,
            decay_side="left",
            v_origin=0.0,
            theta_origin=1000.0,
            sorption=sorption,
        )
        assert np.isnan(dsw.K)
        # Definitional IC at the collision θ: decaying side starts at c_decay_initial.
        assert dsw.c_decay_at_theta(dsw.theta_start) == pytest.approx(dsw.c_decay_initial)
        # Decaying side relaxes from c_decay_initial toward c_fixed (4 -> 1).
        c_later = dsw.c_decay_at_theta(dsw.theta_start + 2000.0)
        assert c_later is not None
        assert dsw.c_fixed < c_later < dsw.c_decay_initial

    def test_freundlich_n2_cr_positive_closed_form(self):
        """Closed-form values for Freundlich n=2, c_R>0, hand-computed.

        Setup uses ``v_origin != 0`` to also catch v_origin-coefficient bugs
        in the position and outlet-crossing formulas.

        Sorption: k_f=0.01, n=2, bulk_density=1500, porosity=0.3, so
        alpha = bulk_density * k_f / porosity = 50.

        Collision IC: rarefaction apex at (V=20, theta=1000); c_decay_initial=9
        (u_c=3), c_fixed=1 (u_R=1).

        Invariant constant: K = theta_local_collision * (u_c - u_R)^2 / (2*u_c + alpha).
        Pick theta_local_collision = 56 to get K = 56*4/56 = 4. Collision V_s:
        V_s - v_origin = 2*K*u_c/(u_c - u_R)^2 = 2*4*3/4 = 6, so v_start = 26.

        Hand-derived later state at u=2 (c_decay=4):
        theta_local = K*(2u + alpha)/(u - u_R)^2 = 4*54/1 = 216, so theta=1216.
        Position: V_s - v_origin = 2*K*u/(u-u_R)^2 = 2*4*2/1 = 16, so V_s=36.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        alpha_expected = 50.0
        k_expected = 4.0
        v_origin = 20.0
        theta_origin = 1000.0
        v_collision = v_origin + 6.0  # = 26
        theta_collision = theta_origin + 56.0  # = 1056

        wave = DecayingShockWave(
            theta_start=theta_collision,
            v_start=v_collision,
            c_decay_initial=9.0,  # u_c = 3
            c_fixed=1.0,  # u_R = 1
            c_fan_tail=1.0,  # drying toward c_fixed
            decay_side="left",
            v_origin=v_origin,
            theta_origin=theta_origin,
            sorption=sorption,
        )

        # 1. K from collision IC.
        assert pytest.approx(k_expected, rel=1e-14) == wave.K

        # 2. Consistency at theta=theta_start.
        assert pytest.approx(v_collision, rel=1e-14) == wave.position_at_theta(theta_collision)
        assert pytest.approx(9.0, rel=1e-14) == wave.c_decay_at_theta(theta_collision)

        # 3. Hand-derived later state at u=2, c_decay=4.
        theta_test = theta_origin + 216.0  # = 1216
        v_test = v_origin + 16.0  # = 36

        assert pytest.approx(4.0, rel=1e-14) == wave.c_decay_at_theta(theta_test)
        assert pytest.approx(v_test, rel=1e-14) == wave.position_at_theta(theta_test)

        # 4. Outlet-crossing round-trip.
        theta_at_v = wave.outlet_crossing_theta(v_outlet=v_test)
        assert theta_at_v is not None
        assert pytest.approx(theta_test, rel=1e-13) == theta_at_v

        # 5. Invariant (u_d - u_R)^2 * theta_local = K * (2*u_d + alpha) at both
        # checked points.
        u_r = 1.0
        for theta, c in [(theta_collision, 9.0), (theta_test, 4.0)]:
            theta_local = theta - theta_origin
            u_d = c**0.5
            lhs = (u_d - u_r) ** 2 * theta_local
            rhs = wave.K * (2.0 * u_d + alpha_expected)
            assert pytest.approx(rhs, rel=1e-14) == lhs

    def test_langmuir_closed_form_with_later_theta_and_outlet_roundtrip(self):
        """Closed-form values for Langmuir c_R=0, hand-computed.

        Setup uses ``v_origin != 0`` and a later-theta invariant check (the
        previous smoke test only verified self-consistency at theta_start,
        which allowed K-scale mutations to pass silently).

        Sorption: s_max=0.5, k_l=10, bulk_density=1500, porosity=0.3.
        a_coeff = bulk_density * s_max * k_l / porosity = 1500 * 0.5 * 10 / 0.3 = 25000.

        Collision IC: apex at (V=200, theta=1000); c_decay_initial=10
        (so K_L + c = 20).

        K = theta_local_collision * c_d^2 / ((K_L + c_d)^2 + a_coeff)
          = theta_local_collision * 100 / (400 + 25000) = theta_local_collision / 254.
        Pick theta_local_collision = 254 to get K=1. Collision V_s:
        V_s - v_origin = K * (K_L + c)^2 / c^2 = 1 * 400 / 100 = 4, so v_start=204.

        Later state at c=5:
        theta_local = K * ((K_L + c)^2 + a_coeff) / c^2 = 1 * (225 + 25000) / 25 = 1009,
        so theta=2009. Position: V_s - v_origin = 1 * 225 / 25 = 9, so V_s=209.
        """
        sorption = LangmuirSorption(s_max=0.5, k_l=10.0, bulk_density=1500.0, porosity=0.3)
        a_expected = 25000.0
        assert sorption.a_coeff == a_expected

        k_expected = 1.0
        v_origin = 200.0
        theta_origin = 1000.0
        v_collision = v_origin + 4.0  # = 204
        theta_collision = theta_origin + 254.0  # = 1254
        k_l = 10.0

        wave = DecayingShockWave(
            theta_start=theta_collision,
            v_start=v_collision,
            c_decay_initial=10.0,
            c_fixed=0.0,
            c_fan_tail=0.0,
            decay_side="left",
            v_origin=v_origin,
            theta_origin=theta_origin,
            sorption=sorption,
        )

        # 1. K from collision IC.
        assert pytest.approx(k_expected, rel=1e-14) == wave.K

        # 2. Consistency at theta=theta_start.
        assert pytest.approx(v_collision, rel=1e-14) == wave.position_at_theta(theta_collision)
        assert pytest.approx(10.0, rel=1e-14) == wave.c_decay_at_theta(theta_collision)

        # 3. Hand-derived later state at c=5.
        theta_test = theta_origin + 1009.0  # = 2009
        v_test = v_origin + 9.0  # = 209

        assert pytest.approx(5.0, rel=1e-14) == wave.c_decay_at_theta(theta_test)
        assert pytest.approx(v_test, rel=1e-14) == wave.position_at_theta(theta_test)

        # 4. Outlet-crossing round-trip.
        theta_at_v = wave.outlet_crossing_theta(v_outlet=v_test)
        assert theta_at_v is not None
        assert pytest.approx(theta_test, rel=1e-13) == theta_at_v

        # 5. Invariant theta_local * c^2 = K * ((K_L + c)^2 + a_coeff) at both checked points.
        for theta, c in [(theta_collision, 10.0), (theta_test, 5.0)]:
            theta_local = theta - theta_origin
            lhs = theta_local * c * c
            rhs = wave.K * ((k_l + c) ** 2 + a_expected)
            assert pytest.approx(rhs, rel=1e-14) == lhs

    def test_concentration_at_point_three_region_branching(self):
        """concentration_at_point dispatches correctly across the shock, fan, and downstream.

        Reuses the c_R>0 setup (v_origin=20, c_fixed=1) so the test catches
        v_origin-coefficient bugs in the fan-branch r_target and c_fixed-sign
        bugs in the at-shock average. Hand-derived state at theta_test=1216:
        V_s=36, c_decay=4 (u=2), so the fan spans R in [R(4), R(1)] = [13.5, 26].

        Three regions checked:

        - v > V_s(theta) + tol: downstream, returns c_fixed=1.
        - |v - V_s(theta)| within tol: returns 0.5*(c_decay + c_fixed) = 2.5.
        - v_fan = v_origin + 12 = 32 (R = 216/12 = 18, inside [13.5, 26]):
          fan-interior c = concentration_from_retardation(18) = (25/17)^2 =
          625/289 for Freundlich n=2 with alpha=50.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        v_origin = 20.0
        theta_origin = 1000.0
        v_collision = v_origin + 6.0  # = 26

        wave = DecayingShockWave(
            theta_start=theta_origin + 56.0,  # = 1056
            v_start=v_collision,
            c_decay_initial=9.0,  # u_c = 3
            c_fixed=1.0,  # u_R = 1
            c_fan_tail=1.0,  # drying toward c_fixed
            decay_side="left",
            v_origin=v_origin,
            theta_origin=theta_origin,
            sorption=sorption,
        )

        theta_test = theta_origin + 216.0  # = 1216
        v_s_test = v_origin + 16.0  # = 36

        # At the shock face: 0.5 * (c_decay + c_fixed) = 0.5 * (4 + 1) = 2.5.
        c_at_shock = wave.concentration_at_point(v=v_s_test, theta=theta_test)
        assert c_at_shock is not None
        assert pytest.approx(2.5, rel=1e-14) == c_at_shock

        # Downstream of the shock: c_fixed = 1.
        c_downstream = wave.concentration_at_point(v=v_s_test + 1.0, theta=theta_test)
        assert c_downstream == 1.0

        # Inside the fan at v_fan = 32 (R = 18, in [13.5, 26]): c = (25/17)^2.
        v_fan = v_origin + 12.0  # = 32
        c_fan_expected = (25.0 / 17.0) ** 2
        c_fan = wave.concentration_at_point(v=v_fan, theta=theta_test)
        assert c_fan is not None
        assert pytest.approx(c_fan_expected, rel=1e-13) == c_fan


class TestSaturatedVanGenuchtenWaveSpeeds:
    """Wave celerities at a saturated Mualem-vG state (``R = 0``) are the removable limit ``+∞``.

    At ``S_e = 1`` (``C = K_s``) the Mualem-vG retardation is exactly ``0`` (``dK/dS_e → +∞``),
    so every ``1/R`` characteristic celerity is ``+∞`` rather than a ``ZeroDivisionError``. The
    realised wetting front is still bounded by a finite Rankine-Hugoniot shock, so this only
    affects the (infinitely fast) characteristic carriers, never the physical front arrival.
    """

    def _sorption(self):
        return VanGenuchtenMualemConductivity(theta_r=0.01, theta_s=0.337, k_s=0.174, van_genuchten_n=2.28)

    def test_characteristic_wave_speed_saturated_is_inf(self):
        """``CharacteristicWave.speed()`` at ``C = K_s`` is ``+∞`` (was ``ZeroDivisionError``)."""
        sorption = self._sorption()
        wave = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=sorption.k_s, sorption=sorption)
        assert wave.speed() == np.inf

    def test_rarefaction_head_speed_saturated_is_inf(self):
        """A rarefaction whose head is at saturation: head celerity ``+∞``, tail finite."""
        sorption = self._sorption()
        wave = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=sorption.k_s, c_tail=0.3 * sorption.k_s, sorption=sorption
        )
        assert wave.head_speed() == np.inf
        assert np.isfinite(wave.tail_speed())


def _fan_consistent_dsw(sorption, *, decay_side, c0, c_fixed, c_fan_tail, theta_origin, theta_local_collision):
    """Build a fan-continuity-consistent DSW (``v_start = v_origin + θ_local/R(c0)``)."""
    theta_start = theta_origin + theta_local_collision
    v_origin = 0.0
    v_start = v_origin + theta_local_collision / float(sorption.retardation(c0))
    return DecayingShockWave(
        theta_start=theta_start,
        v_start=v_start,
        c_decay_initial=c0,
        c_fixed=c_fixed,
        c_fan_tail=c_fan_tail,
        decay_side=decay_side,
        v_origin=v_origin,
        theta_origin=theta_origin,
        sorption=sorption,
    )


class TestDecayingShockWaveNumericalInversion:
    """Fan-exhaustion and outlet-crossing on the numerical (non-closed-form) decay path.

    These pin the orientation-agnostic inversion of the monotone
    ``c_decay(θ_local)`` map and the outlet-crossing bracket seed. Each numerical
    result is checked against an independent adaptive-``quad`` evaluation of the
    same invariant integral.
    """

    def test_growing_decay_fan_exhaustion_finite(self):
        """Growing decay (``c_decay_initial < c_fan_tail``) exhausts at a finite θ, not θ_start.

        Favorable-Freundlich tail collision ([12,6,3,40]-class geometry): the
        decaying side grows 3 → 6 and reaches ``c_fan_tail`` at a finite
        ``θ_local`` given by the un-clamped invariant. The orientation-blind
        early return previously reported immediate exhaustion at ``θ_start``.
        """
        sorption = FreundlichSorption(k_f=0.02, n=2.0, bulk_density=1500.0, porosity=0.3)
        c0, c_fixed, c_fan_tail = 3.0, 40.0, 6.0
        theta_origin, tlc = 400.0, 363.0
        wave = _fan_consistent_dsw(
            sorption,
            decay_side="right",
            c0=c0,
            c_fixed=c_fixed,
            c_fan_tail=c_fan_tail,
            theta_origin=theta_origin,
            theta_local_collision=tlc,
        )
        # Freundlich n=2 growing mirror (c_decay_initial < c_fixed) now has a closed form
        # (the (u_d−u_R)² invariant, −√ root); K is set, not NaN.
        assert not np.isnan(wave.K)
        assert wave.c_decay_initial < wave.c_fan_tail  # growing orientation

        theta_ref = theta_origin + _invariant_theta_local_of_c(sorption, c0, c_fixed, tlc, c_fan_tail)
        theta_exhaust = wave.theta_at_fan_exhaustion()
        assert theta_exhaust is not None
        assert theta_exhaust > wave.theta_start  # not the θ_start early return
        np.testing.assert_allclose(theta_exhaust, theta_ref, rtol=1e-8)
        # Physical meaning: c_decay has reached c_fan_tail at exhaustion.
        c_at_exhaust = wave.c_decay_at_theta(theta_exhaust)
        np.testing.assert_allclose(c_at_exhaust, c_fan_tail, rtol=1e-8)

    def test_numerical_shrinking_fan_exhaustion_finite(self):
        """Numerical shrinking decay (Langmuir c_fixed>0) exhausts at a finite θ, not None.

        The clamped forward profile saturates AT ``c_fan_tail`` and never crosses
        it, so the previous forward-map bracket reported no exhaustion; the
        un-clamped invariant gives a finite θ.
        """
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c0, c_fixed, c_fan_tail = 10.0, 1.0, 4.0
        theta_origin, tlc = 0.0, 500.0
        wave = _fan_consistent_dsw(
            sorption,
            decay_side="left",
            c0=c0,
            c_fixed=c_fixed,
            c_fan_tail=c_fan_tail,
            theta_origin=theta_origin,
            theta_local_collision=tlc,
        )
        assert np.isnan(wave.K)  # numerical path

        theta_ref = theta_origin + _invariant_theta_local_of_c(sorption, c0, c_fixed, tlc, c_fan_tail)
        theta_exhaust = wave.theta_at_fan_exhaustion()
        assert theta_exhaust is not None
        np.testing.assert_allclose(theta_exhaust, theta_ref, rtol=1e-8)
        c_at_exhaust = wave.c_decay_at_theta(theta_exhaust)
        np.testing.assert_allclose(c_at_exhaust, c_fan_tail, rtol=1e-8)

    def test_numerical_outlet_crossing_roundtrip_sub_unit_theta_local(self):
        """Outlet crossing recovers θ even when the crossing lies within θ_local < 1 of the apex.

        Roundtrip ``position_at_theta → outlet_crossing_theta`` must be an
        identity on the numerical path. The bracket seed was floored at the
        dimensional constant 1.0 m³, so crossings closer than 1.0 to the apex
        returned None.
        """
        sorption = FreundlichSorption(k_f=0.01, n=3.0, bulk_density=1500.0, porosity=0.3)
        c0, c_fixed, c_fan_tail = 8.0, 2.0, 2.5
        theta_origin, tlc = 0.0, 0.3  # tiny θ_local so crossings sit below 1.0 m³
        wave = _fan_consistent_dsw(
            sorption,
            decay_side="left",
            c0=c0,
            c_fixed=c_fixed,
            c_fan_tail=c_fan_tail,
            theta_origin=theta_origin,
            theta_local_collision=tlc,
        )
        assert np.isnan(wave.K)  # numerical path

        for theta_local in (0.35, 0.6, 0.99):
            theta = theta_origin + theta_local
            v = wave.position_at_theta(theta)
            assert v is not None
            recovered = wave.outlet_crossing_theta(v_outlet=v)
            assert recovered is not None
            np.testing.assert_allclose(recovered, theta, rtol=1e-8)

    def test_numerical_c_decay_matches_independent_quad(self):
        """Cached numerical ``c_decay(θ_local)`` agrees with the direct invariant quad to 1e-9.

        Anchors the per-wave cached decay profile (built by composite quadrature)
        against an adaptive ``quad`` inversion of the same invariant.
        """
        sorption = FreundlichSorption(k_f=0.01, n=1.5, bulk_density=1500.0, porosity=0.3)
        c0, c_fixed, c_fan_tail = 8.0, 1.0, 2.0
        theta_origin, tlc = 0.0, 500.0
        wave = _fan_consistent_dsw(
            sorption,
            decay_side="left",
            c0=c0,
            c_fixed=c_fixed,
            c_fan_tail=c_fan_tail,
            theta_origin=theta_origin,
            theta_local_collision=tlc,
        )

        def c_direct(theta_local):
            # Invert θ_local(c) = target via the independent quad reference.
            def f(c):
                return _invariant_theta_local_of_c(sorption, c0, c_fixed, tlc, c) - theta_local

            return brentq(f, c_fan_tail + 1e-12, c0 - 1e-12, xtol=1e-15)

        for theta_local in np.linspace(tlc * 1.01, tlc * 6.0, 25):
            cached = wave.c_decay_at_theta(theta_origin + theta_local)  # theta_origin=0 -> θ = θ_local
            np.testing.assert_allclose(cached, c_direct(theta_local), rtol=1e-9)


class TestWaveSpeedCaching:
    """Cached wave celerities (``__post_init__``) are bit-identical to direct evaluation."""

    def test_characteristic_speed_cached_bit_identical(self):
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        wave = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)
        assert wave.speed() == characteristic_speed(5.0, sorption)

    def test_rarefaction_head_tail_speed_cached_bit_identical(self):
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        wave = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption)
        assert wave.head_speed() == characteristic_speed(10.0, sorption)
        assert wave.tail_speed() == characteristic_speed(2.0, sorption)


class TestDoubleFanShockWave:
    """A shock fed by a fan on both sides (issue #294): closed form (n=2 shared apex) + RK4 fallback."""

    def _fan_value(self, sorption, v_o, th_o, c_lo, c_hi, v, th):
        """Bounded self-similar fan value used to build an independent reference ODE."""
        if th <= th_o or v <= v_o:
            return c_lo
        r = (th - th_o) / (v - v_o)
        r_lo, r_hi = sorted((float(sorption.retardation(c_lo)), float(sorption.retardation(c_hi))))
        if r <= r_lo:
            return c_hi if sorption.retardation(c_hi) < sorption.retardation(c_lo) else c_lo
        if r >= r_hi:
            return c_lo if sorption.retardation(c_lo) > sorption.retardation(c_hi) else c_hi
        return float(sorption.concentration_from_retardation(r))

    def test_shared_apex_closed_form_matches_dop853(self):
        """n=2 shared-apex path uses the first-integral quadratic; matches an independent DOP853."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        left = Feeder.fan(0.0, 800.0, 0.0, 5.0, sorption)  # shared apex position v_o = 0
        right = Feeder.fan(0.0, 300.0, 0.0, 10.0, sorption)
        w = DoubleFanShockWave(
            theta_start=1163.15, v_start=29.8143, left_feeder=left, right_feeder=right, sorption=sorption
        )
        assert w._closed_form  # noqa: SLF001

        def rhs(th, y):
            c_l = self._fan_value(sorption, 0.0, 800.0, 0.0, 5.0, y[0], th)
            c_r = self._fan_value(sorption, 0.0, 300.0, 0.0, 10.0, y[0], th)
            return [float(sorption.shock_speed(c_l, c_r))]

        sol = solve_ivp(rhs, (1163.15, 2500.0), [29.8143], method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True)
        for th in (1300.0, 1600.0, 2000.0, 2400.0):
            np.testing.assert_allclose(w.position_at_theta(th), float(sol.sol(th)[0]), rtol=1e-8)

    def test_rk4_fallback_distinct_apex_self_converges_and_matches_dop853(self):
        """Distinct fan apex positions have no closed form → RK4 spline; verify convergence + DOP853."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        left = Feeder.fan(2.0, 800.0, 0.0, 5.0, sorption)  # distinct apex position v_o = 2 ≠ 0
        right = Feeder.fan(0.0, 300.0, 0.0, 10.0, sorption)
        w = DoubleFanShockWave(
            theta_start=1163.15, v_start=29.8143, left_feeder=left, right_feeder=right, sorption=sorption
        )
        assert not w._closed_form  # noqa: SLF001  # distinct apex → numerical path

        def rhs(th, y):
            c_l = self._fan_value(sorption, 2.0, 800.0, 0.0, 5.0, y[0], th)
            c_r = self._fan_value(sorption, 0.0, 300.0, 0.0, 10.0, y[0], th)
            return [float(sorption.shock_speed(c_l, c_r))]

        sol = solve_ivp(rhs, (1163.15, 2500.0), [29.8143], method="DOP853", rtol=1e-12, atol=1e-13, dense_output=True)
        for th in (1300.0, 1600.0, 2000.0, 2400.0):
            # The RK4+spline trajectory tracks the independent DOP853 integration to <1e-6 relative.
            np.testing.assert_allclose(w.position_at_theta(th), float(sol.sol(th)[0]), rtol=1e-6)
