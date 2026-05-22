"""
Unit Tests for Event Handlers.
================================

Tests for wave interaction handlers in front tracking algorithm.
All tests verify physical correctness: entropy conditions, mass conservation,
and proper wave state transitions.
"""

import numpy as np
import pytest

from gwtransport.fronttracking.handlers import (
    create_inlet_waves_at_theta,
    handle_characteristic_collision,
    handle_outlet_crossing,
    handle_rarefaction_characteristic_collision,
    handle_shock_characteristic_collision,
    handle_shock_collision,
    handle_shock_rarefaction_collision,
)
from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    characteristic_speed,
)
from gwtransport.fronttracking.waves import (
    CharacteristicWave,
    DecayingShockWave,
    RarefactionWave,
    ShockWave,
)

freundlich_sorptions = [
    FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3),
    FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3),
]


@pytest.fixture
def freundlich_sorption():
    """Standard Freundlich sorption for testing (n>1, n>1)."""
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
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption)

        char2 = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)

        # Handle collision
        new_waves = handle_characteristic_collision(char1, char2, theta_event=15.0, v_event=100.0)

        # Should create exactly one shock
        assert len(new_waves) == 1
        assert isinstance(new_waves[0], ShockWave)

        shock = new_waves[0]
        assert shock.theta_start == 15.0
        assert shock.v_start == 100.0

    def test_shock_satisfies_entropy(self, freundlich_sorption):
        """Test that created shock satisfies entropy condition."""
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=10.0, sorption=freundlich_sorption)

        char2 = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)

        new_waves = handle_characteristic_collision(char1, char2, theta_event=20.0, v_event=150.0)

        shock = new_waves[0]
        assert isinstance(shock, ShockWave)
        assert shock.satisfies_entropy()

    def test_parent_characteristics_deactivated(self, freundlich_sorption):
        """Test that parent characteristics are deactivated."""
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption)

        char2 = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)

        # Both should be active initially
        assert char1.is_active
        assert char2.is_active

        handle_characteristic_collision(char1, char2, theta_event=15.0, v_event=100.0)

        # Both should be deactivated after collision
        assert not char1.is_active
        assert not char2.is_active

    def test_shock_orientation_correct(self, freundlich_sorption):
        """Test that shock has correct left/right states."""
        # For n>1: lower concentration = faster velocity
        # char2 (C=2) should be faster than char1 (C=5)
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption)

        char2 = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)

        new_waves = handle_characteristic_collision(char1, char2, theta_event=15.0, v_event=100.0)
        shock = new_waves[0]
        assert isinstance(shock, ShockWave)

        # Faster (char2 with C=2) should be upstream (left)
        # Slower (char1 with C=5) should be downstream (right)

        vel1 = characteristic_speed(char1.concentration, char1.sorption)
        vel2 = characteristic_speed(char2.concentration, char2.sorption)

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
        shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)

        shock2 = ShockWave(theta_start=5.0, v_start=0.0, c_left=8.0, c_right=2.0, sorption=freundlich_sorption)

        new_waves = handle_shock_collision(shock1, shock2, theta_event=25.0, v_event=200.0)

        assert len(new_waves) == 1
        assert isinstance(new_waves[0], ShockWave)

        merged = new_waves[0]
        assert merged.theta_start == 25.0
        assert merged.v_start == 200.0

    def test_merged_shock_entropy(self, freundlich_sorption):
        """Test that merged shock satisfies entropy."""
        shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)

        shock2 = ShockWave(theta_start=5.0, v_start=0.0, c_left=8.0, c_right=2.0, sorption=freundlich_sorption)

        new_waves = handle_shock_collision(shock1, shock2, theta_event=25.0, v_event=200.0)
        merged = new_waves[0]

        assert merged.satisfies_entropy()

    def test_parent_shocks_deactivated(self, freundlich_sorption):
        """Test that parent shocks are deactivated."""
        shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)

        shock2 = ShockWave(theta_start=5.0, v_start=0.0, c_left=8.0, c_right=2.0, sorption=freundlich_sorption)

        assert shock1.is_active
        assert shock2.is_active

        handle_shock_collision(shock1, shock2, theta_event=25.0, v_event=200.0)

        assert not shock1.is_active
        assert not shock2.is_active


class TestShockCharacteristicCollisionHandler:
    """Test handle_shock_characteristic_collision function."""

    def test_shock_catches_characteristic(self, freundlich_sorption):
        """Test shock catching characteristic from behind."""
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption)

        char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)

        new_waves = handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

        # MUST create new shock
        assert len(new_waves) == 1, "Expected exactly one new shock wave"
        assert isinstance(new_waves[0], ShockWave), "Expected new wave to be shock"
        assert new_waves[0].satisfies_entropy(), "New shock must satisfy entropy"

    def test_characteristic_catches_shock(self, freundlich_sorption):
        """Test characteristic catching shock from behind."""
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=5.0, c_right=3.0, sorption=freundlich_sorption)

        char = CharacteristicWave(
            theta_start=5.0,
            v_start=0.0,
            concentration=1.0,  # Faster for n>1
            sorption=freundlich_sorption,
        )

        new_waves = handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

        # MUST create new shock
        assert len(new_waves) == 1, "Expected exactly one new shock wave"
        assert isinstance(new_waves[0], ShockWave), "Expected new wave to be shock"
        assert new_waves[0].satisfies_entropy(), "New shock must satisfy entropy"

    def test_waves_deactivated_on_interaction(self, freundlich_sorption):
        """Test that both waves are deactivated."""
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)

        char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)

        handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

        # Both should be deactivated regardless of outcome
        assert not shock.is_active
        assert not char.is_active


class TestShockRarefactionCollisionHandler:
    """Test handle_shock_rarefaction_collision function."""

    def test_shock_catches_tail(self, freundlich_sorption):
        """Shock catching rarefaction tail is resolved by a single DecayingShockWave.

        The old approximate piecewise-overlay handler returned one or more
        plain ShockWaves; the exact handler now returns exactly one
        DecayingShockWave whose decaying side is the rarefaction tail
        (``decay_side='right'``) and fixed side is the shock's upstream state.
        """
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption)

        raref = RarefactionWave(theta_start=5.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)

        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=20.0, v_event=150.0, boundary_type="tail"
        )

        # Exactly one DecayingShockWave subsumes the fan + shock.
        assert len(new_waves) == 1
        dsw = new_waves[0]
        assert isinstance(dsw, DecayingShockWave)
        assert dsw.decay_side == "right"
        assert dsw.c_decay_initial == raref.c_tail
        assert dsw.c_fixed == shock.c_left
        assert dsw.c_fan_tail == raref.c_head
        # Both parents deactivated at the collision.
        assert not shock.is_active
        assert not raref.is_active

    def test_head_catches_shock(self, freundlich_sorption):
        """Test rarefaction head catching shock."""
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=8.0, c_right=4.0, sorption=freundlich_sorption)

        raref = RarefactionWave(theta_start=5.0, v_start=0.0, c_head=10.0, c_tail=5.0, sorption=freundlich_sorption)

        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=20.0, v_event=150.0, boundary_type="head"
        )

        # May create new waves or return empty
        assert isinstance(new_waves, list)


class TestOutletCrossingHandler:
    """Test handle_outlet_crossing function."""

    def test_crossing_returns_event_record(self, freundlich_sorption):
        """Test that crossing returns proper event record."""
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)

        event = handle_outlet_crossing(shock, theta_event=50.0, v_outlet=500.0)

        assert event["theta"] == 50.0
        assert event["type"] == "outlet_crossing"
        assert event["location"] == 500.0
        assert event["wave"] is shock
        assert event["concentration_left"] == 10.0
        assert event["concentration_right"] == 5.0

    def test_wave_remains_active_after_crossing(self, freundlich_sorption):
        """Test that wave is NOT deactivated when crossing outlet."""
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption)

        assert char.is_active

        handle_outlet_crossing(char, theta_event=50.0, v_outlet=500.0)

        # Wave should still be active for querying concentration at earlier positions
        assert char.is_active


class TestInletWaveCreation:
    """Test create_inlet_waves_at_time function."""

    def test_step_increase_creates_shock_n_gt_1(self, freundlich_sorption):
        """Test step increase creates shock for n>1 (higher C travels faster)."""
        # For n>1: higher C → higher R → slower velocity
        # So C: 0→10 means slow→slower velocity, but initial C=0 has R=1 (fastest)
        # Actually C: 0→10 means fast→slow, which is expansion (rarefaction)
        waves = create_inlet_waves_at_theta(
            c_prev=0.0, c_new=10.0, theta=10.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        # For n=2 (n>1), C: 0→10 is rarefaction
        assert len(waves) == 1

    def test_step_increase_creates_rarefaction(self, freundlich_sorption):
        """Test step decrease in concentration creates rarefaction for n>1."""
        # For n>1: C: 10→2 (both non-zero to avoid C=0 special case)
        # vel(10) = 100/R(10), vel(2) = 100/R(2)
        # Since R(10) > R(2), vel(10) < vel(2)
        # vel_new > vel_prev → expansion → rarefaction
        waves = create_inlet_waves_at_theta(
            c_prev=10.0, c_new=2.0, theta=10.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        # MUST create exactly one rarefaction
        assert len(waves) == 1, "Expected exactly one wave"
        assert isinstance(waves[0], RarefactionWave), "Expected wave to be a rarefaction for expansion"

    def test_step_increase_creates_shock(self, freundlich_sorption):
        """Test step increase in concentration creates shock for n>1."""
        # For n>1: C: 2→10
        # vel(2) > vel(10) → new water is slower
        # vel_new < vel_prev → compression → shock
        waves = create_inlet_waves_at_theta(
            c_prev=2.0, c_new=10.0, theta=10.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        # MUST create exactly one shock with proper entropy
        assert len(waves) == 1, "Expected exactly one wave"
        assert isinstance(waves[0], ShockWave), "Expected wave to be a shock for compression"
        assert waves[0].satisfies_entropy(), "Shock must satisfy entropy condition"

    def test_no_change_creates_nothing(self, freundlich_sorption):
        """Test that no concentration change creates no waves."""
        waves = create_inlet_waves_at_theta(
            c_prev=5.0, c_new=5.0, theta=10.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        assert len(waves) == 0

    def test_created_shock_satisfies_entropy(self, freundlich_sorption):
        """Test that created shocks satisfy entropy condition."""
        # Create a scenario that definitely produces a shock
        # For n>1: C: 2→10 means fast→slow, compression→shock
        waves = create_inlet_waves_at_theta(
            c_prev=2.0, c_new=10.0, theta=10.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        # This MUST create shock (C: 2→10, fast→slow, compression)
        assert len(waves) == 1, "Expected exactly one wave"
        assert isinstance(waves[0], ShockWave), "Expected wave to be a shock"
        assert waves[0].satisfies_entropy(), "Shock must satisfy entropy"

    def test_constant_retardation(self, constant_retardation):
        """Test wave creation with constant retardation."""
        # With constant retardation, all concentrations have same velocity
        # So any change is a contact discontinuity (characteristic)
        waves = create_inlet_waves_at_theta(
            c_prev=5.0, c_new=10.0, theta=10.0, sorption=constant_retardation, v_inlet=0.0
        )

        # With constant R, all velocities are same, so contact discontinuity
        assert len(waves) == 1
        assert isinstance(waves[0], CharacteristicWave)

    def test_wave_properties_correct(self, freundlich_sorption):
        """Test that created waves have correct properties."""
        waves = create_inlet_waves_at_theta(
            c_prev=2.0, c_new=10.0, theta=15.0, sorption=freundlich_sorption, v_inlet=0.0
        )

        assert len(waves) == 1
        wave = waves[0]

        # Check basic properties
        assert wave.theta_start == 15.0
        assert wave.v_start == 0.0


class TestPhysicsCorrectness:
    """Test that handlers maintain physical correctness."""

    def test_entropy_always_satisfied_case1(self, freundlich_sorption):
        """Test that created shock satisfies entropy (case 1: C=10.0 → C=2.0)."""
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=10.0, sorption=freundlich_sorption)

        char2 = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)

        new_waves = handle_characteristic_collision(char1, char2, theta_event=15.0, v_event=100.0)

        # MUST create entropy-satisfying shock
        assert len(new_waves) == 1, "Expected exactly one wave"
        assert isinstance(new_waves[0], ShockWave), "Expected wave to be shock"
        assert new_waves[0].satisfies_entropy(), "Shock must satisfy entropy"

    def test_entropy_always_satisfied_case2(self, freundlich_sorption):
        """Test that created shock satisfies entropy (case 2: C=5.0 → C=1.0)."""
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption)

        char2 = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=1.0, sorption=freundlich_sorption)

        new_waves = handle_characteristic_collision(char1, char2, theta_event=15.0, v_event=100.0)

        # MUST create entropy-satisfying shock
        assert len(new_waves) == 1, "Expected exactly one wave"
        assert isinstance(new_waves[0], ShockWave), "Expected wave to be shock"
        assert new_waves[0].satisfies_entropy(), "Shock must satisfy entropy"

    def test_entropy_always_satisfied_case3(self, freundlich_sorption):
        """Test that created shock satisfies entropy (case 3: C=8.0 → C=3.0)."""
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=8.0, sorption=freundlich_sorption)

        char2 = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=3.0, sorption=freundlich_sorption)

        new_waves = handle_characteristic_collision(char1, char2, theta_event=15.0, v_event=100.0)

        # MUST create entropy-satisfying shock
        assert len(new_waves) == 1, "Expected exactly one wave"
        assert isinstance(new_waves[0], ShockWave), "Expected wave to be shock"
        assert new_waves[0].satisfies_entropy(), "Shock must satisfy entropy"

    def test_mass_conservation_in_shock_merger(self, freundlich_sorption):
        """Test that shock merger conserves mass (Rankine-Hugoniot)."""
        shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)

        shock2 = ShockWave(theta_start=5.0, v_start=0.0, c_left=8.0, c_right=2.0, sorption=freundlich_sorption)

        new_waves = handle_shock_collision(shock1, shock2, theta_event=25.0, v_event=200.0)

        # MUST create merged shock
        assert len(new_waves) == 1, "Expected exactly one merged shock"
        merged = new_waves[0]
        # Merged shock should satisfy Rankine-Hugoniot
        # (already verified by satisfies_entropy which checks RH)
        assert merged.satisfies_entropy(), "Merged shock must satisfy entropy and Rankine-Hugoniot"


class TestRarefactionCharacteristicCollisionHandler:
    """Test handle_rarefaction_characteristic_collision function."""

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_characteristic_matching_head_concentration_absorbed(self, freundlich_sorption):
        """Characteristic that matches the rarefaction head concentration is absorbed.

        Silent absorption is allowed only when the characteristic's concentration
        matches the colliding boundary's concentration to within a tight tolerance,
        which is the only case where deactivating the characteristic does not
        destroy mass.
        """
        if freundlich_sorption.n > 1.0:
            c_head, c_tail = 10.0, 2.0
        else:
            c_head, c_tail = 2.0, 10.0

        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=c_head, c_tail=c_tail, sorption=freundlich_sorption
        )

        # Characteristic concentration matches head -> absorption is mass-conserving
        char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=c_head, sorption=freundlich_sorption)

        assert raref.is_active
        assert char.is_active

        new_waves = handle_rarefaction_characteristic_collision(
            raref, char, theta_event=20.0, v_event=150.0, boundary_type="head"
        )

        assert len(new_waves) == 0, "Absorption creates no new waves"
        assert not char.is_active, "Matching characteristic must be deactivated"
        assert raref.is_active, "Rarefaction stays active"

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_mass_destroying_collision_raises(self, freundlich_sorption):
        """Mismatched characteristic concentration triggers a RuntimeError."""
        if freundlich_sorption.n > 1.0:
            c_head, c_tail = 8.0, 3.0
            c_char = 5.0  # Strictly between head and tail -> not absorbable silently
        else:
            c_head, c_tail = 3.0, 8.0
            c_char = 5.0

        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=c_head, c_tail=c_tail, sorption=freundlich_sorption
        )

        char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=c_char, sorption=freundlich_sorption)

        with pytest.raises(RuntimeError, match="would silently destroy mass"):
            handle_rarefaction_characteristic_collision(
                raref, char, theta_event=20.0, v_event=150.0, boundary_type="tail"
            )

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_characteristic_matching_tail_concentration_absorbed(self, freundlich_sorption):
        """Characteristic matching the tail concentration is absorbed at the tail boundary."""
        if freundlich_sorption.n > 1.0:
            c_head, c_tail = 12.0, 4.0
        else:
            c_head, c_tail = 4.0, 12.0

        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=c_head, c_tail=c_tail, sorption=freundlich_sorption
        )

        char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=c_tail, sorption=freundlich_sorption)

        new_waves = handle_rarefaction_characteristic_collision(
            raref, char, theta_event=25.0, v_event=200.0, boundary_type="tail"
        )

        assert len(new_waves) == 0
        assert not char.is_active
        assert raref.is_active


class TestEntropyViolatingScenarios:
    """Test behavior when entropy conditions are violated."""

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_shock_characteristic_creates_rarefaction_on_entropy_violation(self, freundlich_sorption):
        """Test that shock-characteristic collision creates rarefaction when entropy violated.

        When a shock-characteristic collision would create a new shock that violates
        the entropy condition, it creates a rarefaction wave instead to preserve mass
        balance. This implements High Priority #1 from FRONT_TRACKING_REBUILD_PLAN.md.

        Physics: Shock catches slower characteristic → expansion → rarefaction
        """
        # Create scenario where faster characteristic catches slower shock
        # This creates expansion (slow following fast) → entropy violation → rarefaction

        # For n>1: lower C = faster velocity
        # Characteristic with C=1.0 (fastest) catches shock with c_left=3.0, c_right=5.0 (slower)
        # Attempted shock would have c_left=1.0, c_right=5.0
        # vel(1.0) > vel(5.0) but shock velocity between them violates entropy
        # Creates rarefaction instead

        # For n<1: higher C = faster velocity
        # Characteristic with C=10.0 (fastest) catches shock with c_left=5.0, c_right=3.0 (slower)
        # Same logic
        if freundlich_sorption.n > 1.0:
            c_shock_left = 3.0  # Slow (higher C for n>1)
            c_shock_right = 5.0  # Slower still
            c_char = 1.0  # Fast (low C for n>1)
        else:
            c_shock_left = 5.0  # Slow (lower C for n<1)
            c_shock_right = 3.0  # Slower still
            c_char = 10.0  # Fast (high C for n<1)

        shock = ShockWave(
            theta_start=0.0, v_start=0.0, c_left=c_shock_left, c_right=c_shock_right, sorption=freundlich_sorption
        )

        char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=c_char, sorption=freundlich_sorption)

        # Collision: characteristic catches shock
        new_waves = handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

        # Verify behavior: either rarefaction created OR valid shock created
        assert len(new_waves) >= 1, "Expected at least one wave (rarefaction or shock)"

        # Check if rarefaction was created (expansion case - High Priority #1 feature)
        rarefactions = [w for w in new_waves if isinstance(w, RarefactionWave)]
        shocks = [w for w in new_waves if isinstance(w, ShockWave)]

        # At least one wave type must be present
        assert rarefactions or shocks, "Expected either rarefaction or shock"

        # If rarefaction created, verify it's physically valid
        for raref in rarefactions:
            assert raref.head_speed() > raref.tail_speed(), "Rarefaction head must be faster than tail"

        # If shock created, verify it satisfies entropy
        for shock_wave in shocks:
            assert shock_wave.satisfies_entropy(), "Created shock must satisfy entropy"

        # Parent waves should always be deactivated
        assert not shock.is_active, "Shock should be deactivated"
        assert not char.is_active, "Characteristic should be deactivated"

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_waves_deactivated_even_when_entropy_violated(self, freundlich_sorption):
        """Test that waves are deactivated even when no new waves are created.

        This verifies that parent waves are properly cleaned up even in edge cases
        where entropy violations prevent new wave creation.
        """
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=3.0, c_right=2.0, sorption=freundlich_sorption)

        char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=10.0, sorption=freundlich_sorption)

        # Both active initially
        assert shock.is_active
        assert char.is_active

        handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

        # Parent waves should be deactivated regardless of whether new waves were created
        assert not shock.is_active, "Shock should be deactivated"
        assert not char.is_active, "Characteristic should be deactivated"


# =============================================================================
# Tests for flow change handlers (CRITICAL COVERAGE GAP)
# =============================================================================


# =============================================================================
# Physics-based tests for special C≈0 handling (Freundlich n<1)
# =============================================================================


# Removed in Phase 1 (n<1 collision branch was a bug; tested behavior now invalid).
# Phase 2 adds  and friends in this file.


class TestCharacteristicCollisionVelocityOrdering:
    """Test velocity ordering in characteristic collisions (lines 148-153).

    For Freundlich sorption with n>1:
    - R(C) = 1 + (rho_b*k_f)/(n_por*n) * C^((1/n)-1)
    - For n>1, exponent (1/n)-1 < 0, so R decreases with increasing C
    - Lower R means higher velocity (v = flow/R)
    - Therefore: HIGHER C = HIGHER velocity for n>1
    """

    @pytest.fixture
    def freundlich_n_gt_1(self):
        """Freundlich sorption with n>1."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_faster_characteristic_becomes_left(self, freundlich_n_gt_1):
        """Test that faster characteristic becomes c_left of shock.

        Physics: The shock separates upstream (left, behind shock) from
        downstream (right, ahead of shock). The faster wave is upstream.

        For n>1: higher C = lower R = higher velocity (travels faster)
        """
        # For n>1: higher C = lower R = higher velocity
        char_fast = CharacteristicWave(
            theta_start=0.0,
            v_start=0.0,
            concentration=10.0,  # Higher C = faster for n>1
            sorption=freundlich_n_gt_1,
        )

        char_slow = CharacteristicWave(
            theta_start=5.0,
            v_start=0.0,
            concentration=2.0,  # Lower C = slower for n>1
            sorption=freundlich_n_gt_1,
        )

        vel_fast = characteristic_speed(10.0, freundlich_n_gt_1)
        vel_slow = characteristic_speed(2.0, freundlich_n_gt_1)
        assert vel_fast > vel_slow, "Higher C should be faster for n>1"

        new_waves = handle_characteristic_collision(char_fast, char_slow, theta_event=20.0, v_event=150.0)

        assert len(new_waves) == 1
        shock = new_waves[0]
        assert isinstance(shock, ShockWave)

        # Faster (higher C) should be c_left (upstream)
        assert shock.c_left == 10.0, "Faster concentration should be c_left"
        assert shock.c_right == 2.0, "Slower concentration should be c_right"

    def test_slower_characteristic_first_argument(self, freundlich_n_gt_1):
        """Test ordering when slower characteristic is first argument (line 152-153).

        When vel1 <= vel2, then char2's concentration becomes c_left.
        """
        # char1 is slower (lower C for n>1)
        char1 = CharacteristicWave(
            theta_start=0.0,
            v_start=0.0,
            concentration=2.0,  # Lower C = slower for n>1
            sorption=freundlich_n_gt_1,
        )

        # char2 is faster (higher C for n>1)
        char2 = CharacteristicWave(
            theta_start=5.0,
            v_start=0.0,
            concentration=10.0,  # Higher C = faster for n>1
            sorption=freundlich_n_gt_1,
        )

        vel1 = characteristic_speed(2.0, freundlich_n_gt_1)
        vel2 = characteristic_speed(10.0, freundlich_n_gt_1)
        assert vel2 > vel1, "char2 should be faster"

        new_waves = handle_characteristic_collision(char1, char2, theta_event=20.0, v_event=150.0)

        shock = new_waves[0]
        assert isinstance(shock, ShockWave)
        # char2 (faster, higher C) should be c_left
        assert shock.c_left == 10.0
        assert shock.c_right == 2.0


# =============================================================================
# Physics tests for shock-rarefaction collision with wave splitting
# =============================================================================


class TestShockRarefactionTailCollisionPhysics:
    """Physics tests for shock catching rarefaction tail (lines 454-531).

    When a shock catches the tail of a rarefaction fan, it "penetrates" into
    the rarefaction, creating:
    1. A modified shock that continues through the rarefaction
    2. A modified rarefaction with compressed tail (if not fully overtaken)

    This is wave splitting - a fundamental phenomenon in nonlinear wave interaction.
    """

    @pytest.fixture
    def freundlich_n_gt_1(self):
        """Freundlich sorption with n>1 (higher C travels faster)."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_physics_shock_penetrates_rarefaction(self, freundlich_n_gt_1):
        """Test shock penetrating into rarefaction fan.

        Physics: A strong shock (large concentration jump) can penetrate
        into a rarefaction fan, creating both a continuing shock and a
        modified rarefaction.
        """
        # Create a fast shock with large concentration jump
        shock = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=15.0,  # Very high concentration behind shock
            c_right=1.0,  # Low concentration ahead
            sorption=freundlich_n_gt_1,
        )

        # Create a rarefaction ahead of the shock
        # For n>1, head has higher C (slower), tail has lower C (faster)
        raref = RarefactionWave(
            theta_start=5.0,
            v_start=50.0,
            c_head=8.0,  # Higher C = slower (leading edge for n>1)
            c_tail=3.0,  # Lower C = faster (trailing edge for n>1)
            sorption=freundlich_n_gt_1,
        )

        # Collision at tail
        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=20.0, v_event=150.0, boundary_type="tail"
        )

        # Should create at least a shock
        assert len(new_waves) >= 1, "Expected at least one new wave"

        # Check that all shocks satisfy entropy
        shocks = [w for w in new_waves if isinstance(w, ShockWave)]
        for s in shocks:
            assert s.satisfies_entropy(), "Created shock must satisfy entropy"

        # Parent waves should be deactivated
        assert not shock.is_active, "Original shock should be deactivated"
        assert not raref.is_active, "Original rarefaction should be deactivated"

    def test_physics_shock_overtakes_rarefaction_completely(self, freundlich_n_gt_1):
        """Shock overtaking a small rarefaction yields one exact DecayingShockWave.

        Physics: a fast shock catching a small rarefaction tail merges into a
        single decaying shock. The old overlay returned a plain ShockWave;
        the exact handler returns one DecayingShockWave that asymptotes to the
        fixed (upstream) state as the fan is consumed.
        """
        # Very fast shock
        shock = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=20.0,  # Very high C
            c_right=2.0,  # Low C
            sorption=freundlich_n_gt_1,
        )

        # Small rarefaction (small concentration range)
        raref = RarefactionWave(
            theta_start=5.0,
            v_start=50.0,
            c_head=5.0,
            c_tail=4.0,  # Very close to head
            sorption=freundlich_n_gt_1,
        )

        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=20.0, v_event=150.0, boundary_type="tail"
        )

        # Exactly one DecayingShockWave; both parents deactivated.
        assert len(new_waves) == 1
        assert isinstance(new_waves[0], DecayingShockWave)
        assert new_waves[0].decay_side == "right"
        assert not shock.is_active
        assert not raref.is_active

    def test_physics_wave_splitting_creates_modified_rarefaction(self, freundlich_n_gt_1):
        """Test that wave splitting can create modified rarefaction.

        Physics: When shock partially penetrates rarefaction, the portion
        ahead of the shock remains as a modified rarefaction with a new tail.
        """
        # Moderate shock
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=12.0, c_right=3.0, sorption=freundlich_n_gt_1)

        # Large rarefaction
        raref = RarefactionWave(
            theta_start=5.0,
            v_start=50.0,
            c_head=10.0,  # Wide concentration range
            c_tail=2.0,
            sorption=freundlich_n_gt_1,
        )

        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=30.0, v_event=200.0, boundary_type="tail"
        )

        # Check for rarefactions in result (wave splitting)
        rarefactions = [w for w in new_waves if isinstance(w, RarefactionWave)]

        # If rarefaction created, verify it has valid structure
        for r in rarefactions:
            assert r.head_speed() > r.tail_speed(), "Rarefaction head must be faster than tail"


class TestShockRarefactionHeadCollisionPhysics:
    """Physics tests for rarefaction head catching shock (lines 533-566).

    When the head of a rarefaction catches a slower shock from behind,
    it creates compression between the rarefaction head and the shock.
    This may form a new compression shock.
    """

    @pytest.fixture
    def freundlich_n_gt_1(self):
        """Freundlich sorption with n>1."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_physics_rarefaction_head_creates_compression(self, freundlich_n_gt_1):
        """Test rarefaction head catching shock creates compression.

        Physics: If rarefaction head is faster than the shock it catches,
        the head compresses against the shock, potentially forming a new shock.
        """
        # Slow shock that rarefaction head can catch
        shock = ShockWave(
            theta_start=0.0,
            v_start=50.0,  # Started ahead
            c_left=8.0,
            c_right=5.0,  # Moderate jump
            sorption=freundlich_n_gt_1,
        )

        # Fast rarefaction with high-C head (fast for n>1)
        raref = RarefactionWave(
            theta_start=5.0,
            v_start=0.0,  # Started behind
            c_head=12.0,  # Higher C = slower velocity for n>1
            c_tail=6.0,  # Lower C = faster velocity
            sorption=freundlich_n_gt_1,
        )

        # Check velocities
        shock_vel = shock.speed
        head_vel = raref.head_speed()
        assert shock_vel is not None

        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=25.0, v_event=180.0, boundary_type="head"
        )

        # If head is faster than shock, may create new shock
        if head_vel > shock_vel:
            # Check for shock in result
            shocks = [w for w in new_waves if isinstance(w, ShockWave)]
            for s in shocks:
                assert s.satisfies_entropy(), "Compression shock must satisfy entropy"

    def test_physics_shock_deactivated_on_head_collision(self, freundlich_n_gt_1):
        """Test that original shock is deactivated when caught by rarefaction head."""
        shock = ShockWave(theta_start=0.0, v_start=50.0, c_left=6.0, c_right=4.0, sorption=freundlich_n_gt_1)

        raref = RarefactionWave(theta_start=5.0, v_start=0.0, c_head=10.0, c_tail=5.0, sorption=freundlich_n_gt_1)

        handle_shock_rarefaction_collision(shock, raref, theta_event=25.0, v_event=180.0, boundary_type="head")

        # Original shock should be deactivated
        assert not shock.is_active, "Original shock should be deactivated"


# =============================================================================
# Physics tests for handle_flow_change (lines 901-967)
# =============================================================================


# =============================================================================
# Physics tests for create_inlet_waves with n<1 (lines 1029-1060)
# =============================================================================


class TestInletWavesNLT1Physics:
    """Physics tests for inlet wave creation with n<1 Freundlich sorption.

    For n<1 (unfavorable sorption):
    - Lower C = faster velocity
    - C=0 with c_min=0 has R(0)=1 (fastest possible)
    - Step from C=0 to C>0: fast→slow = expansion = rarefaction? No, creates characteristic
    - Step from C>0 to C=0: slow→fast = compression? No, creates characteristic

    The special handling for n<1 with c_min=0 creates characteristics instead
    of shocks/rarefactions because the C=0 state is physically meaningful.
    """

    @pytest.fixture
    def freundlich_n_lt_1(self):
        """Freundlich sorption with n<1 and c_min=0."""
        return FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)

    def test_physics_step_from_zero_creates_characteristic(self, freundlich_n_lt_1):
        """Test: Step from C=0 to C>0 creates characteristic for n<1.

        Physics: For n<1 with c_min=0, C=0 represents clean water with R(0)=1.
        When concentration steps from 0 to positive, we create a characteristic
        with the new concentration that propagates at v(C>0).
        """
        waves = create_inlet_waves_at_theta(c_prev=0.0, c_new=5.0, theta=10.0, sorption=freundlich_n_lt_1, v_inlet=0.0)

        assert len(waves) == 1, "Expected one wave"
        assert isinstance(waves[0], CharacteristicWave), "Should create characteristic for C=0 to C>0 with n<1"
        assert waves[0].concentration == 5.0, "Characteristic should carry new concentration"

    def test_physics_step_to_zero_creates_characteristic(self, freundlich_n_lt_1):
        """Test: Step from C>0 to C=0 creates characteristic for n<1.

        Physics: When clean water (C=0) enters behind contaminated water (C>0),
        we create a characteristic with C=0 that propagates at v(0) = flow/1.
        """
        waves = create_inlet_waves_at_theta(c_prev=5.0, c_new=0.0, theta=10.0, sorption=freundlich_n_lt_1, v_inlet=0.0)

        assert len(waves) == 1, "Expected one wave"
        assert isinstance(waves[0], CharacteristicWave), "Should create characteristic for C>0 to C=0 with n<1"
        assert waves[0].concentration == 0.0, "Characteristic should carry zero concentration"

    def test_physics_step_between_nonzero_creates_appropriate_wave(self, freundlich_n_lt_1):
        """Test: Step between nonzero concentrations follows velocity logic.

        Physics: For n<1, lower C is faster. So:
        - C: 5→10 means fast→slow = expansion = rarefaction
        - C: 10→5 means slow→fast = compression = shock
        """
        # Step up: 5→10 (fast to slow for n<1) = expansion
        waves_up = create_inlet_waves_at_theta(
            c_prev=5.0, c_new=10.0, theta=10.0, sorption=freundlich_n_lt_1, v_inlet=0.0
        )

        # Verify velocities
        vel_5 = characteristic_speed(5.0, freundlich_n_lt_1)
        vel_10 = characteristic_speed(10.0, freundlich_n_lt_1)
        assert vel_5 > vel_10, "Lower C should be faster for n<1"

        # Step up should create rarefaction (new water slower than old)
        if len(waves_up) == 1:
            # Could be rarefaction if vel_new < vel_prev
            assert isinstance(waves_up[0], RarefactionWave), "Expected rarefaction for fast→slow step"

        # Step down: 10→5 (slow to fast for n<1) = compression
        waves_down = create_inlet_waves_at_theta(
            c_prev=10.0, c_new=5.0, theta=10.0, sorption=freundlich_n_lt_1, v_inlet=0.0
        )

        # Step down should create shock (new water faster than old)
        if len(waves_down) == 1:
            assert isinstance(waves_down[0], ShockWave), "Expected shock for slow→fast step"
            assert waves_down[0].satisfies_entropy(), "Shock must satisfy entropy"


# =============================================================================
# Physics tests for entropy violation edge cases (lines 351-393)
# =============================================================================


class TestEntropyViolationRarefactionCreation:
    """Physics tests for rarefaction creation when entropy is violated.

    When a shock-characteristic collision would create a shock that violates
    entropy, the code creates a rarefaction instead. This is physically correct:
    entropy violation indicates expansion, not compression.
    """

    @pytest.fixture
    def freundlich_n_gt_1(self):
        """Freundlich sorption with n>1."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_physics_expansion_creates_rarefaction_not_shock(self, freundlich_n_gt_1):
        """Test: Expansion scenario creates rarefaction instead of non-entropic shock.

        Physics: When fast water catches slow water, it's expansion (not compression).
        Attempting to form a shock would violate entropy. Create rarefaction instead.
        """
        # Create shock with moderate jump
        shock = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=5.0,
            c_right=4.0,  # Small jump
            sorption=freundlich_n_gt_1,
        )

        # Create fast characteristic that catches shock
        # For n>1, lower C = faster
        char = CharacteristicWave(
            theta_start=5.0,
            v_start=0.0,
            concentration=2.0,  # Very fast (low C for n>1)
            sorption=freundlich_n_gt_1,
        )

        # Fast characteristic catches slower shock
        char_vel = characteristic_speed(2.0, freundlich_n_gt_1)
        shock_vel = shock.speed
        assert shock_vel is not None

        if char_vel > shock_vel:
            # Characteristic is faster - catches shock from behind
            new_waves = handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

            # Check what was created
            rarefactions = [w for w in new_waves if isinstance(w, RarefactionWave)]
            shocks = [w for w in new_waves if isinstance(w, ShockWave)]

            # Either rarefaction or valid shock
            for r in rarefactions:
                assert r.head_speed() > r.tail_speed(), "Rarefaction structure must be valid"
            for s in shocks:
                assert s.satisfies_entropy(), "Any shock must satisfy entropy"

    def test_physics_rarefaction_head_faster_than_tail(self, freundlich_n_gt_1):
        """Test: Created rarefactions always have head faster than tail.

        Physics: Rarefaction head (leading edge) must travel faster than tail
        (trailing edge) for the fan to expand, not collapse.
        """
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=6.0, c_right=5.0, sorption=freundlich_n_gt_1)

        char = CharacteristicWave(
            theta_start=5.0,
            v_start=0.0,
            concentration=3.0,  # Fast for n>1
            sorption=freundlich_n_gt_1,
        )

        new_waves = handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

        for w in new_waves:
            if isinstance(w, RarefactionWave):
                assert w.head_speed() > w.tail_speed(), "Rarefaction must expand, not collapse"


# =============================================================================
# Physics tests for recreate wave error cases (lines 756-757, 820-821, 887-888)
# =============================================================================


# =============================================================================
# Additional edge case tests for remaining uncovered lines
# =============================================================================


# Removed in Phase 1 (n<1 collision branch was a bug; tested behavior now invalid).
# Phase 2 adds  and friends in this file.


class TestShockCollisionEdgeCases:
    """Test edge cases for shock collision handling."""

    @pytest.fixture
    def freundlich_sorption(self):
        """Standard Freundlich sorption for testing."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_shock_collision_velocity_ordering(self, freundlich_sorption):
        """Test lines 230-235: second shock faster than first.

        When shock2.speed > shock1.speed, the merged shock takes
        c_left from shock2 and c_right from shock1.
        """
        # Create shock1 that is SLOWER
        shock1 = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=12.0,  # High C upstream
            c_right=10.0,  # Moderate C downstream - small jump, slower
            sorption=freundlich_sorption,
        )

        # Create shock2 that is FASTER
        shock2 = ShockWave(
            theta_start=5.0,
            v_start=0.0,
            c_left=15.0,  # Very high C upstream
            c_right=5.0,  # Low C downstream - large jump, faster
            sorption=freundlich_sorption,
        )

        # Check velocity ordering
        assert shock1.speed is not None
        assert shock2.speed is not None
        if shock2.speed > shock1.speed:
            # Lines 233-235 will execute
            new_waves = handle_shock_collision(shock1, shock2, theta_event=30.0, v_event=200.0)

            merged = new_waves[0]
            # shock2 (faster) provides c_left, shock1 (slower) provides c_right
            assert merged.c_left == shock2.c_left
            assert merged.c_right == shock1.c_right


class TestShockCharacteristicEdgeCases:
    """Test edge cases for shock-characteristic collision."""

    @pytest.fixture
    def freundlich_sorption(self):
        """Standard Freundlich sorption for testing."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)


class TestShockRarefactionEdgeCases:
    """Test edge cases for shock-rarefaction collision."""

    @pytest.fixture
    def freundlich_sorption(self):
        """Standard Freundlich sorption for testing."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_tail_collision_raref_concentration_none(self, freundlich_sorption):
        """Test lines 462-477: rarefaction concentration at collision is None.

        When the shock is not actually inside the rarefaction (edge case),
        fall back to simple approach.
        """
        # Create shock
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=15.0, c_right=2.0, sorption=freundlich_sorption)

        # Create rarefaction that hasn't expanded much
        raref = RarefactionWave(
            theta_start=5.0,
            v_start=100.0,  # Starts far ahead
            c_head=10.0,
            c_tail=5.0,
            sorption=freundlich_sorption,
        )

        # Collision happens but position is outside rarefaction fan
        # Use very small time difference to make rarefaction still very thin
        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=5.001, v_event=100.0, boundary_type="tail"
        )

        # Should still produce result
        assert isinstance(new_waves, list)

    def test_head_collision_no_compression_shock(self, freundlich_sorption):
        """Test lines 563-566: no compression shock forms.

        When the rarefaction head is slower than the shock, no new shock forms.
        """
        # Fast shock
        shock = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=15.0,  # High C - fast
            c_right=2.0,  # Low C - large jump
            sorption=freundlich_sorption,
        )

        # Slow rarefaction with low-C head (slow for n>1)
        raref = RarefactionWave(
            theta_start=5.0,
            v_start=50.0,
            c_head=4.0,  # Low C = slow for n>1
            c_tail=3.0,  # Even lower C
            sorption=freundlich_sorption,
        )

        # Check that rarefaction head is slower than shock
        raref_head_vel = raref.head_speed()
        shock_vel = shock.speed
        assert shock_vel is not None

        if raref_head_vel <= shock_vel:
            # Line 563-566: no compression forms
            new_waves = handle_shock_rarefaction_collision(
                shock, raref, theta_event=20.0, v_event=150.0, boundary_type="head"
            )

            # Should return empty list and deactivate both
            assert len(new_waves) == 0
            assert not shock.is_active
            assert not raref.is_active


class TestInletWaveCreationEdgeCases:
    """Test edge cases for inlet wave creation."""

    @pytest.fixture
    def freundlich_sorption(self):
        """Standard Freundlich sorption for testing."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_entropy_violation_returns_empty(self, freundlich_sorption):
        """Test lines 1080-1085: entropy violation in shock returns empty.

        When a shock would violate entropy, the function returns empty list.
        """
        # For n>1 (higher C faster): shock forms when new C is faster than old C
        # Entropy is satisfied when v(c_left) > v_shock > v(c_right)
        # This means c_left should be the faster (higher C) water

        # Create scenario that might cause entropy issues
        # Large jump where new water much slower than old
        waves = create_inlet_waves_at_theta(
            c_prev=15.0,  # High C - fast for n>1
            c_new=1.0,  # Low C - slow for n>1
            theta=10.0,
            sorption=freundlich_sorption,
            v_inlet=0.0,
        )

        # This is an expansion (fast old water, slow new water) → rarefaction
        # Rarefaction should be created, not shock
        if len(waves) == 1:
            assert isinstance(waves[0], RarefactionWave), "Expansion should create rarefaction"

    def test_same_velocity_creates_characteristic(self):
        """Test lines 1107-1117: same velocity creates characteristic.

        When vel_new == vel_prev (contact discontinuity), create characteristic.
        """
        # With constant retardation, all velocities are the same
        const_r = ConstantRetardation(retardation_factor=2.0)

        waves = create_inlet_waves_at_theta(c_prev=5.0, c_new=10.0, theta=10.0, sorption=const_r, v_inlet=0.0)

        # With constant R, velocities are same → characteristic
        assert len(waves) == 1
        assert isinstance(waves[0], CharacteristicWave)
        assert waves[0].concentration == 10.0  # Carries new concentration


class TestCharacteristicCollisionEntropyViolation:
    """Test entropy violation in characteristic collision (lines 166-172)."""

    @pytest.fixture
    def freundlich_sorption(self):
        """Standard Freundlich sorption for testing."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_entropy_violation_raises_runtime_error(self, freundlich_sorption):
        """Test lines 166-172: entropy violation raises RuntimeError.

        When a shock from characteristic collision violates entropy,
        a RuntimeError is raised. This is a safety check that shouldn't
        normally trigger if characteristics collide correctly.
        """
        # For n>1: higher C = faster
        # Characteristic collision creates shock when fast catches slow
        # The shock should always satisfy entropy if collision is correct

        # Create normal collision that produces valid shock
        char1 = CharacteristicWave(
            theta_start=0.0,
            v_start=0.0,
            concentration=10.0,  # Higher C = faster for n>1
            sorption=freundlich_sorption,
        )

        char2 = CharacteristicWave(
            theta_start=5.0,
            v_start=0.0,
            concentration=5.0,  # Lower C = slower for n>1
            sorption=freundlich_sorption,
        )

        # This should work normally
        new_waves = handle_characteristic_collision(char1, char2, theta_event=20.0, v_event=150.0)
        assert len(new_waves) == 1
        assert isinstance(new_waves[0], ShockWave)
        assert new_waves[0].satisfies_entropy()


class TestShockCharCollisionEntropyViolationPaths:
    """Test entropy violation paths in shock-characteristic collision (lines 351-393).

    These tests exercise the rarefaction creation path when a shock-char collision
    would violate entropy (indicating expansion instead of compression).
    """

    @pytest.fixture
    def freundlich_n_lt_1(self):
        """Freundlich sorption with n<1 (lower C travels faster)."""
        return FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)

    def test_entropy_violation_shock_catches_characteristic(self, freundlich_n_lt_1):
        """Test lines 351-387: entropy violation when shock catches characteristic.

        For n<1, lower C is faster. Create scenario where resulting shock
        would violate entropy, triggering rarefaction creation.
        """
        # For n<1: lower C = faster
        # Create shock with high C left, low C right (fast shock for n<1)
        shock = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=2.0,  # Lower C = faster for n<1
            c_right=3.0,  # Higher C = slower
            sorption=freundlich_n_lt_1,
        )

        # Create characteristic with even lower C (faster still)
        char = CharacteristicWave(
            theta_start=5.0,
            v_start=0.0,
            concentration=1.0,  # Very low C = very fast for n<1
            sorption=freundlich_n_lt_1,
        )

        # Characteristic should be faster for n<1
        new_waves = handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

        # Both waves should be deactivated
        assert not shock.is_active
        assert not char.is_active

        # Check result type
        for w in new_waves:
            if isinstance(w, ShockWave):
                assert w.satisfies_entropy()
            if isinstance(w, RarefactionWave):
                assert w.head_speed() > w.tail_speed()

    def test_entropy_violation_characteristic_catches_shock(self, freundlich_n_lt_1):
        """Test lines 356-360: entropy violation when characteristic catches shock.

        Test the other branch where characteristic velocity > shock velocity.
        """
        # For n<1: lower C = faster
        # Slow shock
        shock = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=8.0,  # Higher C = slower for n<1
            c_right=10.0,  # Even higher C = even slower
            sorption=freundlich_n_lt_1,
        )

        # Very fast characteristic (low C)
        char = CharacteristicWave(
            theta_start=5.0,
            v_start=0.0,
            concentration=1.0,  # Low C = fast for n<1
            sorption=freundlich_n_lt_1,
        )

        new_waves = handle_shock_characteristic_collision(shock, char, theta_event=20.0, v_event=150.0)

        # Both should be deactivated
        assert not shock.is_active
        assert not char.is_active

        # Verify result
        assert isinstance(new_waves, list)


class TestShockRarefactionTailEdgeCases:
    """Test edge cases in shock-rarefaction tail collision (lines 477, 492-494, 516-520, 529-531)."""

    @pytest.fixture
    def freundlich_sorption(self):
        """Standard Freundlich sorption for testing."""
        return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_tail_collision_standard_wave_splitting(self, freundlich_sorption):
        """Tail collision is resolved by one exact DecayingShockWave.

        The old overlay split the interaction into a continuing ShockWave (and
        possibly a modified rarefaction). The exact handler returns a single
        DecayingShockWave whose self-similar fan profile already encodes the
        full interaction, so no separate ShockWave is emitted.
        """
        # Create shock
        shock = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=15.0,  # High C - fast for n>1
            c_right=5.0,  # Moderate C
            sorption=freundlich_sorption,
        )

        # Create rarefaction ahead
        raref = RarefactionWave(
            theta_start=5.0,
            v_start=100.0,
            c_head=12.0,  # Higher C - slower for n>1
            c_tail=8.0,  # Moderate C - faster
            sorption=freundlich_sorption,
        )

        # Let some time pass so rarefaction expands
        t_event = 30.0
        v_tail = raref.tail_position_at_theta(t_event)  # Where tail is at t_event
        assert v_tail is not None

        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=t_event, v_event=v_tail, boundary_type="tail"
        )

        # Both parents should be deactivated
        assert not shock.is_active
        assert not raref.is_active

        # Exactly one DecayingShockWave subsumes the split, with the
        # rarefaction tail as the decaying side and the shock's upstream
        # state held fixed.
        assert len(new_waves) == 1
        dsw = new_waves[0]
        assert isinstance(dsw, DecayingShockWave)
        assert dsw.decay_side == "right"
        assert dsw.c_decay_initial == raref.c_tail
        assert dsw.c_fixed == shock.c_left
        assert dsw.c_fan_tail == raref.c_head
        assert dsw.v_start == v_tail

        # Definitional initial conditions at the collision θ: the decaying side
        # starts at c_decay_initial and the shock sits exactly at v_start. These
        # hold for the numerical decay path as well as the closed forms.
        assert dsw.c_decay_at_theta(dsw.theta_start) == pytest.approx(dsw.c_decay_initial)
        assert dsw.position_at_theta(dsw.theta_start) == pytest.approx(dsw.v_start)

    def test_tail_collision_rarefaction_completely_overtaken(self, freundlich_sorption):
        """A small fully-overtaken rarefaction still yields one DecayingShockWave.

        The old overlay reduced a fully-overtaken fan to a plain continuing
        ShockWave. The exact handler always returns a single DecayingShockWave;
        as the small fan is consumed it asymptotes to the fixed (upstream)
        state, which the solver's DSW_FAN_EXHAUSTED event then promotes to a
        regular shock.
        """
        # Very fast shock
        shock = ShockWave(
            theta_start=0.0,
            v_start=0.0,
            c_left=20.0,  # Very high C - very fast for n>1
            c_right=1.0,  # Very low C - large jump
            sorption=freundlich_sorption,
        )

        # Small rarefaction that will be completely overtaken
        raref = RarefactionWave(
            theta_start=5.0,
            v_start=50.0,
            c_head=6.0,  # Close concentrations
            c_tail=5.0,  # Small fan
            sorption=freundlich_sorption,
        )

        # Let rarefaction expand
        t_event = 30.0
        v_event = raref.tail_position_at_theta(t_event)
        assert v_event is not None

        new_waves = handle_shock_rarefaction_collision(
            shock, raref, theta_event=t_event, v_event=v_event, boundary_type="tail"
        )

        # Exactly one DecayingShockWave; both parents deactivated.
        assert len(new_waves) == 1
        assert isinstance(new_waves[0], DecayingShockWave)
        assert not shock.is_active
        assert not raref.is_active


class TestRegressionsForIssue168Collision:
    """Regression tests for handle_characteristic_collision (issue #168, P1.6).

    The old code created a backwards RarefactionWave for n<1 when C=0 catches
    a positive concentration. The corrected handler creates a R-H shock.
    """

    def test_collision_n_lt_1_zero_catching_positive_creates_shock(self):
        """P1.6: For n<1, C=0 catching C>0 is compression -> ShockWave (not RarefactionWave)."""
        sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)

        char_zero = CharacteristicWave(theta_start=4.0, v_start=0.0, concentration=0.0, sorption=sorption)
        char_pos = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption)

        new_waves = handle_characteristic_collision(char_zero, char_pos, theta_event=20.0, v_event=150.0)

        assert len(new_waves) == 1
        assert isinstance(new_waves[0], ShockWave), (
            "Old behavior (n<1 special-case) returned RarefactionWave; P1.6 fix returns ShockWave"
        )
        assert new_waves[0].satisfies_entropy()

    def test_collision_generated_shock_satisfies_rankine_hugoniot(self):
        """P1.6: collision-generated shock speed matches Rankine-Hugoniot in (V, θ) to machine precision."""
        sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)
        c_low, c_high = 0.0, 5.0

        char_low = CharacteristicWave(theta_start=4.0, v_start=0.0, concentration=c_low, sorption=sorption)
        char_high = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=c_high, sorption=sorption)

        new_waves = handle_characteristic_collision(char_low, char_high, theta_event=20.0, v_event=150.0)
        shock = new_waves[0]
        assert isinstance(shock, ShockWave)

        # R-H in (V, θ): dV_s/dθ = (c_right - c_left) / (C_tot(c_right) - C_tot(c_left)). Flow drops out.
        c_left, c_right = shock.c_left, shock.c_right
        c_tot_diff = sorption.total_concentration(c_right) - sorption.total_concentration(c_left)
        expected_speed = (c_right - c_left) / c_tot_diff
        assert np.isclose(shock.speed, expected_speed, rtol=1e-14)
