"""
Unit tests for front tracking wave classes.

Tests verify wave behavior, position calculations, and concentration queries.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pytest

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave


class TestCharacteristicWave:
    """Test CharacteristicWave class."""

    def test_initialization(self):
        """Test valid initialization."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption)

        assert char.t_start == 0.0
        assert char.v_start == 0.0
        assert char.flow == 100.0
        assert char.concentration == 5.0
        assert char.is_active

    def test_velocity_constant_retardation(self):
        """Test velocity computation with constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption)

        velocity = char.velocity()
        expected = 100.0 / 2.0

        assert np.isclose(velocity, expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_velocity_freundlich(self):
        """Test velocity computation with Freundlich sorption."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=10.0, sorption=sorption)

        velocity = char.velocity()
        r = sorption.retardation(10.0)
        expected = 100.0 / r

        assert np.isclose(velocity, expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_position_at_time_linear_propagation(self):
        """Test that characteristic propagates linearly."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption)

        for t in [1.0, 5.0, 10.0]:
            v = char.position_at_time(t)
            expected = (100.0 / 2.0) * t
            assert np.isclose(v, expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_position_at_time_before_start(self):
        """Test position is None for t < t_start."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=10.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption)

        v = char.position_at_time(5.0)
        assert v is None

    def test_position_at_time_inactive(self):
        """Test position is None when inactive."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption)
        char.is_active = False

        v = char.position_at_time(10.0)
        assert v is None

    def test_position_at_time_nonzero_start(self):
        """Test propagation from non-zero starting position."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=5.0, v_start=100.0, flow=100.0, concentration=5.0, sorption=sorption)

        v = char.position_at_time(15.0)
        velocity = 100.0 / 2.0
        expected = 100.0 + velocity * (15.0 - 5.0)

        assert np.isclose(v, expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_concentration_left_right_equal(self):
        """Test that left and right concentrations are same for characteristic."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption)

        assert char.concentration_left() == 5.0
        assert char.concentration_right() == 5.0
        assert char.concentration_left() == char.concentration_right()

    def test_concentration_at_point_reached(self):
        """Test concentration at point that characteristic has reached."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption)

        # At t=10, characteristic is at v = 50*10 = 500
        # Should return concentration for v < 500
        c = char.concentration_at_point(v=300.0, t=10.0)
        assert c == 5.0

    def test_concentration_at_point_not_reached(self):
        """Test concentration at point that characteristic hasn't reached yet."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption)

        # At t=10, characteristic is at v = 50*10 = 500
        # Should return None for v > 500
        c = char.concentration_at_point(v=600.0, t=10.0)
        assert c is None


class TestShockWave:
    """Test ShockWave class."""

    def test_initialization(self):
        """Test valid initialization."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        assert shock.t_start == 0.0
        assert shock.v_start == 0.0
        assert shock.flow == 100.0
        assert shock.c_left == 10.0
        assert shock.c_right == 2.0
        assert shock.is_active

    def test_velocity_computed_in_post_init(self):
        """Test that shock velocity is computed automatically."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        assert shock.velocity is not None
        assert shock.velocity > 0

    def test_velocity_rankine_hugoniot(self):
        """Test shock velocity satisfies Rankine-Hugoniot."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        # Verify Rankine-Hugoniot
        flux_left = 100.0 * 10.0
        flux_right = 100.0 * 2.0
        c_total_left = sorption.total_concentration(10.0)
        c_total_right = sorption.total_concentration(2.0)

        v_expected = (flux_right - flux_left) / (c_total_right - c_total_left)

        assert np.isclose(shock.velocity, v_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_position_at_time_linear_propagation(self):
        """Test shock propagates linearly."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        v_shock = shock.velocity
        assert v_shock is not None
        for t in [1.0, 5.0, 10.0]:
            v = shock.position_at_time(t)
            assert v is not None
            expected = v_shock * t
            assert np.isclose(v, expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_position_at_time_before_start(self):
        """Test position is None for t < t_start."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(t_start=10.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        v = shock.position_at_time(5.0)
        assert v is None

    def test_concentration_left_right(self):
        """Test left and right concentration getters."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        assert shock.concentration_left() == 10.0
        assert shock.concentration_right() == 2.0

    def test_concentration_at_point_upstream(self):
        """Test concentration upstream of shock."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        # Shock at some position
        v_shock = shock.position_at_time(10.0)
        assert v_shock is not None
        c = shock.concentration_at_point(v=v_shock - 10.0, t=10.0)

        assert c == 10.0  # Upstream concentration

    def test_concentration_at_point_downstream(self):
        """Test concentration downstream of shock."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        # Shock at some position
        v_shock = shock.position_at_time(10.0)
        assert v_shock is not None
        c = shock.concentration_at_point(v=v_shock + 10.0, t=10.0)

        assert c == 2.0  # Downstream concentration

    def test_concentration_at_point_exact_shock_position(self):
        """Test concentration exactly at shock position."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        v_shock = shock.position_at_time(10.0)
        assert v_shock is not None
        c = shock.concentration_at_point(v=v_shock, t=10.0)

        # At exact shock position, returns average
        assert np.isclose(c, 0.5 * (10.0 + 2.0), rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_satisfies_entropy_physical_shock(self):
        """Test that physical compression shock satisfies entropy."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption)

        assert shock.satisfies_entropy()

    def test_satisfies_entropy_unphysical_shock(self):
        """Test that unphysical expansion shock violates entropy."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        # For n > 1, this is backwards (expansion, not compression)
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=2.0, c_right=10.0, sorption=sorption)

        assert not shock.satisfies_entropy()


class TestRarefactionWave:
    """Test RarefactionWave class."""

    def test_initialization_valid(self):
        """Test valid initialization (head faster than tail)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        # For n > 1, higher C is faster
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        assert raref.c_head == 10.0
        assert raref.c_tail == 2.0
        assert raref.is_active

    def test_initialization_invalid_velocities(self):
        """Test that rarefaction with head slower than tail raises error."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        # For n > 1, lower C is slower - this would be backwards
        with pytest.raises(ValueError, match="Not a rarefaction"):
            RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=2.0, c_tail=10.0, sorption=sorption)

    def test_head_tail_velocities(self):
        """Test head and tail velocity computations."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        v_head = raref.head_velocity()
        v_tail = raref.tail_velocity()

        # Head should be faster than tail
        assert v_head > v_tail

        # Verify exact values
        r_head = sorption.retardation(10.0)
        r_tail = sorption.retardation(2.0)
        assert np.isclose(v_head, 100.0 / r_head, rtol=1e-14)  # type: ignore[no-matching-overload]
        assert np.isclose(v_tail, 100.0 / r_tail, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_head_tail_positions(self):
        """Test head and tail position computations."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        t = 10.0
        v_head = raref.head_position_at_time(t)
        v_tail = raref.tail_position_at_time(t)

        # Head should be ahead of tail
        assert v_head is not None
        assert v_tail is not None
        assert v_head > v_tail

        # Verify exact values
        expected_head = raref.head_velocity() * t
        expected_tail = raref.tail_velocity() * t
        assert np.isclose(v_head, expected_head, rtol=1e-14)  # type: ignore[no-matching-overload]
        assert np.isclose(v_tail, expected_tail, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_position_at_time_returns_head(self):
        """Test that position_at_time returns head position."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        v = raref.position_at_time(10.0)
        v_head = raref.head_position_at_time(10.0)

        assert v == v_head

    def test_contains_point_inside_fan(self):
        """Test contains_point for point inside rarefaction fan."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        t = 20.0
        v_head = raref.head_position_at_time(t)
        v_tail = raref.tail_position_at_time(t)
        assert v_head is not None
        assert v_tail is not None
        v_mid = 0.5 * (v_head + v_tail)

        assert raref.contains_point(v_mid, t)

    def test_contains_point_outside_fan(self):
        """Test contains_point for points outside rarefaction fan."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        t = 20.0
        v_head = raref.head_position_at_time(t)
        v_tail = raref.tail_position_at_time(t)
        assert v_head is not None
        assert v_tail is not None

        # Before tail
        assert not raref.contains_point(v_tail - 10.0, t)

        # After head
        assert not raref.contains_point(v_head + 10.0, t)

    def test_concentration_at_point_self_similar(self):
        """Test self-similar solution for concentration in rarefaction."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        t = 20.0
        v = 150.0

        c = raref.concentration_at_point(v, t)

        # Should be between tail and head
        assert c is not None
        assert 2.0 <= c <= 10.0

        # Verify self-similar solution: R(C) = flow*t/v
        r_target = 100.0 * t / v
        c_from_r = sorption.concentration_from_retardation(r_target)
        assert np.isclose(c, c_from_r, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_concentration_at_point_outside_fan(self):
        """Test concentration is None outside rarefaction fan."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        t = 20.0
        v_head = raref.head_position_at_time(t)
        assert v_head is not None

        # After head
        c = raref.concentration_at_point(v_head + 100.0, t)
        assert c is None

    def test_concentration_at_origin(self):
        """Test concentration at origin of rarefaction."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        # At v=0 (origin), concentration should be tail value
        c = raref.concentration_at_point(v=0.0, t=10.0)
        assert np.isclose(c, 2.0, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_concentration_left_right(self):
        """Test concentration_left and concentration_right."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)

        assert raref.concentration_left() == 2.0  # Tail (upstream)
        assert raref.concentration_right() == 10.0  # Head (downstream)

    def test_concentration_with_constant_retardation_returns_none(self):
        """Test that rarefaction with constant R returns None (shouldn't form)."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        # With constant retardation, all concentrations travel at same speed
        # So we can't even create a rarefaction (would raise ValueError)
        with pytest.raises(ValueError, match="Not a rarefaction"):
            RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
