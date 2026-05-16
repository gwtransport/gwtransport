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

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption, LangmuirSorption
from gwtransport.fronttracking.waves import CharacteristicWave, DecayingShockWave, RarefactionWave, ShockWave


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
        # V_s - V_origin = n*K/u_c = 3*k_expected/2
        v_collision = 3.0 * k_expected / u_c

        wave = DecayingShockWave(
            theta_start=270.0 + 1000.0,
            v_start=v_collision,
            c_decay_initial=c_decay_initial,
            c_fixed=0.0,
            decay_side="left",
            v_origin=0.0,
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

        # And position is V_origin + n*K/u_later.
        v_later = wave.position_at_theta(theta_later)
        assert v_later is not None
        assert pytest.approx(3.0 * wave.K / u_later, rel=1e-12) == v_later

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
            "v_origin": 0.0,
            "theta_origin": 1000.0,
            "sorption": sorption,
        }

        with pytest.raises(ValueError, match="decay_side"):
            DecayingShockWave(decay_side="middle", **common)

        with pytest.raises(ValueError, match="c_decay_initial"):
            DecayingShockWave(**{**common, "decay_side": "left", "c_decay_initial": 0.0})

        with pytest.raises(ValueError, match="c_fixed"):
            DecayingShockWave(**{**common, "decay_side": "left", "c_fixed": -1.0})

        with pytest.raises(ValueError, match="theta_origin"):
            DecayingShockWave(**{**common, "decay_side": "left", "theta_origin": 1500.0})

        with pytest.raises(TypeError, match="FreundlichSorption or LangmuirSorption"):
            DecayingShockWave(**{
                **common,
                "decay_side": "left",
                "sorption": ConstantRetardation(retardation_factor=2.0),
            })

    def test_post_init_rejects_freundlich_n_not_2_with_c_fixed_positive(self):
        """Freundlich with c_fixed>0 currently supports only n=2; other n raises."""
        sorption = FreundlichSorption(k_f=0.01, n=1.5, bulk_density=1500.0, porosity=0.3)
        with pytest.raises(NotImplementedError, match="c_fixed > 0"):
            DecayingShockWave(
                theta_start=1500.0,
                v_start=20.0,
                c_decay_initial=4.0,
                c_fixed=1.0,
                decay_side="left",
                v_origin=0.0,
                theta_origin=1000.0,
                sorption=sorption,
            )

    def test_langmuir_initialization_does_not_raise(self):
        """Smoke test: Langmuir DecayingShockWave constructs and K > 0.

        Math correctness is deferred to the test parametric in Phase 2 step 5;
        this just confirms the class accepts Langmuir and the closed-form
        K is finite and positive.
        """
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        wave = DecayingShockWave(
            theta_start=1500.0,
            v_start=10.0,
            c_decay_initial=4.0,
            c_fixed=0.0,
            decay_side="left",
            v_origin=0.0,
            theta_origin=1000.0,
            sorption=sorption,
        )
        assert wave.K > 0.0
        assert np.isfinite(wave.K)
        # Position consistency: V_s(theta_start) = v_origin + K·(K_L+c_d)²/c_d²
        v_pred = wave.K * (5.0 + 4.0) ** 2 / 16.0
        assert wave.position_at_theta(1500.0) == pytest.approx(v_pred, rel=1e-13)
