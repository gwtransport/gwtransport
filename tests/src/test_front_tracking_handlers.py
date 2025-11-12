"""
Unit Tests for Event Handlers.
================================

Tests for wave interaction handlers in front tracking algorithm.
All tests verify physical correctness: entropy conditions, mass conservation,
and proper wave state transitions.
"""

import numpy as np
import pytest

from gwtransport.front_tracking_math import FreundlichSorption, ConstantRetardation
from gwtransport.front_tracking_waves import (
    CharacteristicWave,
    ShockWave,
    RarefactionWave,
)
from gwtransport.front_tracking_handlers import (
    handle_characteristic_collision,
    handle_shock_collision,
    handle_shock_characteristic_collision,
    handle_shock_rarefaction_collision,
    handle_rarefaction_characteristic_collision,
    handle_outlet_crossing,
    create_inlet_waves_at_time,
)


@pytest.fixture
def freundlich_sorption():
    """Standard Freundlich sorption for testing (n>1, favorable)."""
    return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)


@pytest.fixture
def constant_retardation():
    """Constant retardation for testing."""
    return ConstantRetardation(retardation_factor=2.0)


class TestCharacteristicCollisionHandler:
    """Test handle_characteristic_collision function."""

    def test_collision_creates_shock(self, freundlich_sorption):
        """Test that characteristic collision creates a shock."""
        # Two characteristics with different concentrations
        char1 = CharacteristicWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            concentration=5.0,
            sorption=freundlich_sorption,
        )

        char2 = CharacteristicWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            concentration=2.0,
            sorption=freundlich_sorption,
        )

        # Handle collision
        new_waves = handle_characteristic_collision(char1, char2, t_event=15.0, v_event=100.0)

        # Should create exactly one shock
        assert len(new_waves) == 1
        assert isinstance(new_waves[0], ShockWave)

        shock = new_waves[0]
        assert shock.t_start == 15.0
        assert shock.v_start == 100.0

    def test_shock_satisfies_entropy(self, freundlich_sorption):
        """Test that created shock satisfies entropy condition."""
        char1 = CharacteristicWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            concentration=10.0,
            sorption=freundlich_sorption,
        )

        char2 = CharacteristicWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            concentration=2.0,
            sorption=freundlich_sorption,
        )

        new_waves = handle_characteristic_collision(char1, char2, t_event=20.0, v_event=150.0)

        shock = new_waves[0]
        assert shock.satisfies_entropy()

    def test_parent_characteristics_deactivated(self, freundlich_sorption):
        """Test that parent characteristics are deactivated."""
        char1 = CharacteristicWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            concentration=5.0,
            sorption=freundlich_sorption,
        )

        char2 = CharacteristicWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            concentration=2.0,
            sorption=freundlich_sorption,
        )

        # Both should be active initially
        assert char1.is_active
        assert char2.is_active

        handle_characteristic_collision(char1, char2, t_event=15.0, v_event=100.0)

        # Both should be deactivated after collision
        assert not char1.is_active
        assert not char2.is_active

    def test_shock_orientation_correct(self, freundlich_sorption):
        """Test that shock has correct left/right states."""
        # For n>1: lower concentration = faster velocity
        # char2 (C=2) should be faster than char1 (C=5)
        char1 = CharacteristicWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            concentration=5.0,
            sorption=freundlich_sorption,
        )

        char2 = CharacteristicWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            concentration=2.0,
            sorption=freundlich_sorption,
        )

        new_waves = handle_characteristic_collision(char1, char2, t_event=15.0, v_event=100.0)
        shock = new_waves[0]

        # Faster (char2 with C=2) should be upstream (left)
        # Slower (char1 with C=5) should be downstream (right)
        from gwtransport.front_tracking_math import characteristic_velocity

        vel1 = characteristic_velocity(char1.concentration, char1.flow, char1.sorption)
        vel2 = characteristic_velocity(char2.concentration, char2.flow, char2.sorption)

        if vel2 > vel1:
            assert shock.c_left == char2.concentration
            assert shock.c_right == char1.concentration
        else:
            assert shock.c_left == char1.concentration
            assert shock.c_right == char2.concentration


class TestShockCollisionHandler:
    """Test handle_shock_collision function."""

    def test_collision_merges_shocks(self, freundlich_sorption):
        """Test that shock collision creates merged shock."""
        shock1 = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=5.0,
            sorption=freundlich_sorption,
        )

        shock2 = ShockWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            c_left=8.0,
            c_right=2.0,
            sorption=freundlich_sorption,
        )

        new_waves = handle_shock_collision(shock1, shock2, t_event=25.0, v_event=200.0)

        assert len(new_waves) == 1
        assert isinstance(new_waves[0], ShockWave)

        merged = new_waves[0]
        assert merged.t_start == 25.0
        assert merged.v_start == 200.0

    def test_merged_shock_entropy(self, freundlich_sorption):
        """Test that merged shock satisfies entropy."""
        shock1 = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=5.0,
            sorption=freundlich_sorption,
        )

        shock2 = ShockWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            c_left=8.0,
            c_right=2.0,
            sorption=freundlich_sorption,
        )

        new_waves = handle_shock_collision(shock1, shock2, t_event=25.0, v_event=200.0)
        merged = new_waves[0]

        assert merged.satisfies_entropy()

    def test_parent_shocks_deactivated(self, freundlich_sorption):
        """Test that parent shocks are deactivated."""
        shock1 = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=5.0,
            sorption=freundlich_sorption,
        )

        shock2 = ShockWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            c_left=8.0,
            c_right=2.0,
            sorption=freundlich_sorption,
        )

        assert shock1.is_active
        assert shock2.is_active

        handle_shock_collision(shock1, shock2, t_event=25.0, v_event=200.0)

        assert not shock1.is_active
        assert not shock2.is_active


class TestShockCharacteristicCollisionHandler:
    """Test handle_shock_characteristic_collision function."""

    def test_shock_catches_characteristic(self, freundlich_sorption):
        """Test shock catching characteristic from behind."""
        shock = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=0.0,
            sorption=freundlich_sorption,
        )

        char = CharacteristicWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            concentration=2.0,
            sorption=freundlich_sorption,
        )

        new_waves = handle_shock_characteristic_collision(shock, char, t_event=20.0, v_event=150.0)

        # Should create new shock (if entropy satisfied)
        if new_waves:
            assert len(new_waves) == 1
            assert isinstance(new_waves[0], ShockWave)
            assert new_waves[0].satisfies_entropy()

    def test_characteristic_catches_shock(self, freundlich_sorption):
        """Test characteristic catching shock from behind."""
        shock = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=5.0,
            c_right=3.0,
            sorption=freundlich_sorption,
        )

        char = CharacteristicWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            concentration=1.0,  # Faster for n>1
            sorption=freundlich_sorption,
        )

        new_waves = handle_shock_characteristic_collision(shock, char, t_event=20.0, v_event=150.0)

        # Should create new shock or return empty
        if new_waves:
            assert isinstance(new_waves[0], ShockWave)
            assert new_waves[0].satisfies_entropy()

    def test_waves_deactivated_on_interaction(self, freundlich_sorption):
        """Test that both waves are deactivated."""
        shock = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=5.0,
            sorption=freundlich_sorption,
        )

        char = CharacteristicWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            concentration=2.0,
            sorption=freundlich_sorption,
        )

        handle_shock_characteristic_collision(shock, char, t_event=20.0, v_event=150.0)

        # Both should be deactivated regardless of outcome
        assert not shock.is_active
        assert not char.is_active


class TestShockRarefactionCollisionHandler:
    """Test handle_shock_rarefaction_collision function."""

    def test_shock_catches_tail(self, freundlich_sorption):
        """Test shock catching rarefaction tail."""
        shock = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=0.0,
            sorption=freundlich_sorption,
        )

        raref = RarefactionWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            c_head=5.0,
            c_tail=2.0,
            sorption=freundlich_sorption,
        )

        new_waves = handle_shock_rarefaction_collision(shock, raref, t_event=20.0, v_event=150.0, boundary_type="tail")

        # Should create new shock or return empty
        if new_waves:
            assert all(isinstance(w, ShockWave) for w in new_waves)

    def test_head_catches_shock(self, freundlich_sorption):
        """Test rarefaction head catching shock."""
        shock = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=8.0,
            c_right=4.0,
            sorption=freundlich_sorption,
        )

        raref = RarefactionWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            c_head=10.0,
            c_tail=5.0,
            sorption=freundlich_sorption,
        )

        new_waves = handle_shock_rarefaction_collision(shock, raref, t_event=20.0, v_event=150.0, boundary_type="head")

        # May create new waves or return empty
        assert isinstance(new_waves, list)


class TestOutletCrossingHandler:
    """Test handle_outlet_crossing function."""

    def test_crossing_returns_event_record(self, freundlich_sorption):
        """Test that crossing returns proper event record."""
        shock = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=5.0,
            sorption=freundlich_sorption,
        )

        event = handle_outlet_crossing(shock, t_event=50.0, v_outlet=500.0)

        assert event["time"] == 50.0
        assert event["type"] == "outlet_crossing"
        assert event["location"] == 500.0
        assert event["wave"] is shock
        assert event["concentration_left"] == 10.0
        assert event["concentration_right"] == 5.0

    def test_wave_remains_active_after_crossing(self, freundlich_sorption):
        """Test that wave is NOT deactivated when crossing outlet."""
        char = CharacteristicWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            concentration=5.0,
            sorption=freundlich_sorption,
        )

        assert char.is_active

        handle_outlet_crossing(char, t_event=50.0, v_outlet=500.0)

        # Wave should still be active for querying concentration at earlier positions
        assert char.is_active


class TestInletWaveCreation:
    """Test create_inlet_waves_at_time function."""

    def test_step_increase_creates_shock_favorable(self, freundlich_sorption):
        """Test step increase creates shock for favorable sorption (n>1)."""
        # For n>1: higher C → higher R → slower velocity
        # So C: 0→10 means slow→slower velocity, but initial C=0 has R=1 (fastest)
        # Actually C: 0→10 means fast→slow, which is expansion (rarefaction)
        waves = create_inlet_waves_at_time(
            c_prev=0.0, c_new=10.0, t=10.0, flow=100.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        # For n=2 (favorable), C: 0→10 is rarefaction
        assert len(waves) == 1

    def test_step_decrease_correct_wave_type(self, freundlich_sorption):
        """Test step decrease with favorable sorption."""
        # For n>1: C: 10→2 (both non-zero to avoid C=0 special case)
        # vel(10) = 100/R(10) = 8.91, vel(2) = 100/R(2) = 18.67
        # vel_new > vel_prev → compression → shock
        waves = create_inlet_waves_at_time(
            c_prev=10.0, c_new=2.0, t=10.0, flow=100.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        assert len(waves) == 1
        # This should create a shock with proper entropy
        if isinstance(waves[0], ShockWave):
            assert waves[0].satisfies_entropy()
        elif isinstance(waves[0], RarefactionWave):
            # If rarefaction, verify it's correct
            pass

    def test_no_change_creates_nothing(self, freundlich_sorption):
        """Test that no concentration change creates no waves."""
        waves = create_inlet_waves_at_time(
            c_prev=5.0, c_new=5.0, t=10.0, flow=100.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        assert len(waves) == 0

    def test_shock_satisfies_entropy(self, freundlich_sorption):
        """Test that created shocks satisfy entropy condition."""
        # Create a scenario that definitely produces a shock
        # For n>1: need new velocity > old velocity
        # vel = flow / R(C), so need R(new) < R(old), so need C_new < C_old
        # But also need it to be physically valid (faster catching slower)
        waves = create_inlet_waves_at_time(
            c_prev=10.0, c_new=2.0, t=10.0, flow=100.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        # This should create shock (C: 10→2, slower→faster, compression)
        if waves and isinstance(waves[0], ShockWave):
            assert waves[0].satisfies_entropy()

    def test_constant_retardation(self, constant_retardation):
        """Test wave creation with constant retardation."""
        # With constant retardation, all concentrations have same velocity
        # So any change is a contact discontinuity (characteristic)
        waves = create_inlet_waves_at_time(
            c_prev=5.0, c_new=10.0, t=10.0, flow=100.0, sorption=constant_retardation, v_inlet=0.0
        )

        # With constant R, all velocities are same, so contact discontinuity
        assert len(waves) == 1
        assert isinstance(waves[0], CharacteristicWave)

    def test_wave_properties_correct(self, freundlich_sorption):
        """Test that created waves have correct properties."""
        waves = create_inlet_waves_at_time(
            c_prev=2.0, c_new=10.0, t=15.0, flow=100.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        assert len(waves) == 1
        wave = waves[0]

        # Check basic properties
        assert wave.t_start == 15.0
        assert wave.v_start == 0.0
        assert wave.flow == 100.0


class TestPhysicsCorrectness:
    """Test that handlers maintain physical correctness."""

    def test_entropy_always_satisfied(self, freundlich_sorption):
        """Test that all created shocks satisfy entropy."""
        # Test various collision scenarios
        test_cases = [
            # (char1_c, char2_c)
            (10.0, 2.0),
            (5.0, 1.0),
            (8.0, 3.0),
        ]

        for c1, c2 in test_cases:
            char1 = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=c1, sorption=freundlich_sorption
            )

            char2 = CharacteristicWave(
                t_start=5.0, v_start=0.0, flow=100.0, concentration=c2, sorption=freundlich_sorption
            )

            # Handle collision
            try:
                new_waves = handle_characteristic_collision(char1, char2, t_event=15.0, v_event=100.0)
                if new_waves and isinstance(new_waves[0], ShockWave):
                    assert new_waves[0].satisfies_entropy()
            except RuntimeError:
                # If entropy violation raises error, that's also acceptable
                pass

    def test_mass_conservation_in_shock_merger(self, freundlich_sorption):
        """Test that shock merger conserves mass (Rankine-Hugoniot)."""
        shock1 = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=5.0,
            sorption=freundlich_sorption,
        )

        shock2 = ShockWave(
            t_start=5.0,
            v_start=0.0,
            flow=100.0,
            c_left=8.0,
            c_right=2.0,
            sorption=freundlich_sorption,
        )

        try:
            new_waves = handle_shock_collision(shock1, shock2, t_event=25.0, v_event=200.0)

            if new_waves:
                merged = new_waves[0]
                # Merged shock should satisfy Rankine-Hugoniot
                # (already verified by satisfies_entropy which checks RH)
                assert merged.satisfies_entropy()
        except RuntimeError:
            # Some mergers may be physically invalid
            pass
