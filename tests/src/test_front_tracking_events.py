"""
Unit Tests for Event Detection Module.
=======================================

Tests for exact analytical event detection in front tracking algorithm.
All tests verify machine-precision accuracy (rtol=1e-14) for intersection
calculations.
"""

import numpy as np
import pytest

from gwtransport.fronttracking.events import (
    Event,
    EventType,
    find_characteristic_intersection,
    find_outlet_crossing,
    find_rarefaction_boundary_intersections,
    find_shock_characteristic_intersection,
    find_shock_shock_intersection,
)
from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption, characteristic_position
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave

freundlich_sorptions = [
    FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3),
    FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3),
]


@pytest.fixture
def constant_retardation():
    """Constant retardation for testing."""
    return ConstantRetardation(retardation_factor=2.0)


class TestEventDataStructures:
    """Test Event and EventType classes."""

    def test_event_creation(self):
        """Test Event dataclass creation."""
        event = Event(time=15.5, event_type=EventType.SHOCK_CHAR_COLLISION, waves_involved=[], location=250.0)
        assert event.time == 15.5
        assert event.event_type == EventType.SHOCK_CHAR_COLLISION
        assert event.location == 250.0

    def test_event_ordering(self):
        """Test events are ordered by time."""
        event1 = Event(time=10.0, event_type=EventType.OUTLET_CROSSING, waves_involved=[], location=500.0)
        event2 = Event(time=5.0, event_type=EventType.CHAR_CHAR_COLLISION, waves_involved=[], location=100.0)

        assert event2 < event1
        assert not (event1 < event2)

    def test_event_type_enum(self):
        """Test EventType enum values."""
        assert EventType.CHAR_CHAR_COLLISION.value == "characteristic_collision"
        assert EventType.SHOCK_SHOCK_COLLISION.value == "shock_collision"
        assert EventType.OUTLET_CROSSING.value == "outlet_crossing"


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestCharacteristicIntersection:
    """Test find_characteristic_intersection function."""

    def test_intersection_simple_case(self, freundlich_sorption):
        """Test simple intersection of two characteristics."""
        # Create two characteristics starting at same position and time
        # but with different concentrations (thus different velocities)
        char1 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=freundlich_sorption
        )

        char2 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=2.0, sorption=freundlich_sorption
        )

        # Since they start at same point but have different velocities,
        # they should never intersect again (they diverge) for both n>1 and n<1
        result = find_characteristic_intersection(char1, char2, t_current=0.0)
        assert result is None

    def test_intersection_catching_from_behind(self, freundlich_sorption):
        """Test faster characteristic catching slower one from behind."""
        # char1 starts early with LOWER concentration
        char1 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=10000.0, concentration=1.0, sorption=freundlich_sorption
        )

        # char2 starts later at same position, with HIGHER concentration
        # Higher flow makes waves travel faster, increasing chance of intersection
        char2 = CharacteristicWave(
            t_start=100.0, v_start=0.0, flow=10000.0, concentration=10.0, sorption=freundlich_sorption
        )

        # char2 is faster and will catch char1
        result = find_characteristic_intersection(char1, char2, t_current=100.0)

        if freundlich_sorption.n < 1.0:
            # For n<1, higher concentration = faster velocity, so no intersection occurs
            assert result is None, "Expected intersection for n<1 when faster characteristic starts behind"
        else:
            # For n>1, higher concentration = slower velocity, so intersection
            assert result is not None, "Expected no intersection for n>1 when faster characteristic starts behind"

            t_int, v_int = result

            # Verify both characteristics are at same position at intersection time
            v1 = characteristic_position(
                char1.concentration, char1.flow, char1.sorption, char1.t_start, char1.v_start, t_int
            )
            v2 = characteristic_position(
                char2.concentration, char2.flow, char2.sorption, char2.t_start, char2.v_start, t_int
            )
            assert np.isclose(v1, v2, rtol=1e-14)
            assert np.isclose(v1, v_int, rtol=1e-14)

    def test_parallel_characteristics(self, freundlich_sorption):
        """Test parallel characteristics don't intersect."""
        # Two characteristics with same concentration (same velocity)
        char1 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=freundlich_sorption
        )

        char2 = CharacteristicWave(
            t_start=0.0,
            v_start=10.0,  # Different start position
            flow=100.0,
            concentration=5.0,  # Same concentration
            sorption=freundlich_sorption,
        )

        result = find_characteristic_intersection(char1, char2, t_current=0.0)
        assert result is None

    def test_intersection_in_past(self, freundlich_sorption):
        """Test that intersections in the past are ignored."""
        char1 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=freundlich_sorption
        )

        char2 = CharacteristicWave(
            t_start=0.0, v_start=100.0, flow=100.0, concentration=2.0, sorption=freundlich_sorption
        )

        # If they would have intersected at t=50, asking at t=100 should return None
        result = find_characteristic_intersection(char1, char2, t_current=1000.0)
        assert result is None, "Expected no intersection in the past. Initial conditions shouldn't matter."


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestShockShockIntersection:
    """Test find_shock_shock_intersection function."""

    def test_shock_collision_simple(self, freundlich_sorption):
        """Test two shocks colliding."""
        # Create two shocks with different velocities
        # shock1 starts at t=0, V=0

        if freundlich_sorption.n < 1.0:
            shock1 = ShockWave(
                t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption
            )

            # shock2 starts at t=10, V=0 (after of shock1)
            shock2 = ShockWave(
                t_start=10.0, v_start=0.0, flow=100.0, c_left=5.0, c_right=0.0, sorption=freundlich_sorption
            )

        elif freundlich_sorption.n > 1.0:
            shock1 = ShockWave(
                t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption
            )

            # shock2 starts at t=0, V=200 (ahead of shock1)
            shock2 = ShockWave(
                t_start=0.0, v_start=200.0, flow=100.0, c_left=5.0, c_right=0.0, sorption=freundlich_sorption
            )

        result = find_shock_shock_intersection(shock1, shock2, t_current=0.0)

        # MUST return an intersection (faster shock1 catches slower shock2)
        assert result is not None, "Expected intersection between two shocks"

        t_int, v_int = result
        # Verify both shocks are at same position
        v1 = shock1.v_start + shock1.velocity * (t_int - shock1.t_start)
        v2 = shock2.v_start + shock2.velocity * (t_int - shock2.t_start)
        assert np.isclose(v1, v2, rtol=1e-14)
        assert np.isclose(v1, v_int, rtol=1e-14)

    def test_parallel_shocks(self, freundlich_sorption):
        """Test parallel shocks don't intersect."""
        # Two shocks with same velocity (same concentration jump)
        shock1 = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)

        shock2 = ShockWave(
            t_start=10.0, v_start=50.0, flow=100.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption
        )

        result = find_shock_shock_intersection(shock1, shock2, t_current=10.0)
        assert result is None


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestShockCharacteristicIntersection:
    """Test find_shock_characteristic_intersection function."""

    def test_shock_catches_characteristic(self, freundlich_sorption):
        """Test shock catching a characteristic from behind. Char has lower concentration."""
        if freundlich_sorption.n > 1.0:
            # Characteristic ahead
            char = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=2.0, sorption=freundlich_sorption
            )

            # Shock behind but faster
            shock = ShockWave(
                t_start=10.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption
            )

        elif freundlich_sorption.n < 1.0:
            # Characteristic ahead
            char = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=2.0, sorption=freundlich_sorption
            )

            # Shock behind but faster
            shock = ShockWave(
                t_start=10.0, v_start=0.0, flow=100.0, c_left=1.0, c_right=0.0, sorption=freundlich_sorption
            )

        result = find_shock_characteristic_intersection(shock, char, t_current=10.0)

        # MUST return an intersection
        assert result is not None, "Expected shock to catch characteristic"

        t_int, v_int = result
        assert t_int > 10.0, "Intersection must be in future"
        assert v_int > 0, "Must be at positive position"

        # Verify both are at same position
        v_char = characteristic_position(
            char.concentration, char.flow, char.sorption, char.t_start, char.v_start, t_int
        )
        v_shock = shock.v_start + shock.velocity * (t_int - shock.t_start)
        assert np.isclose(v_char, v_shock, rtol=1e-14)

    def test_shock_not_catches_characteristic(self, freundlich_sorption):
        """Test shock not catching a characteristic from behind."""
        if freundlich_sorption.n > 1.0:
            # Characteristic ahead
            char = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=2.0, sorption=freundlich_sorption
            )

            # Shock behind but faster
            shock = ShockWave(
                t_start=10.0, v_start=0.0, flow=100.0, c_left=1.0, c_right=0.0, sorption=freundlich_sorption
            )

        elif freundlich_sorption.n < 1.0:
            # Characteristic ahead
            char = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=2.0, sorption=freundlich_sorption
            )

            # Shock behind but faster
            shock = ShockWave(
                t_start=10.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption
            )

        result = find_shock_characteristic_intersection(shock, char, t_current=10.0)
        # MUST return no intersection
        assert result is None, "Expected no intersection between shock and characteristic"


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestRarefactionIntersections:
    """Test find_rarefaction_boundary_intersections function."""

    def test_rarefaction_head_characteristic_intersection(self, freundlich_sorption):
        """Test rarefaction head intersecting with characteristic."""
        if freundlich_sorption.n < 1.0:
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                raref = RarefactionWave(
                    t_start=0.0,
                    v_start=0.0,
                    flow=100.0,
                    c_head=5.0,  # Tail concentration (slower)
                    c_tail=2.0,  # Head concentration (faster)
                    sorption=freundlich_sorption,
                )
        elif freundlich_sorption.n > 1.0:
            # For n>1: lower concentration = faster velocity
            # Head (faster) has c=2.0, tail (slower) has c=5.0
            raref = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=100.0,
                c_head=5.0,  # Tail concentration (slower)
                c_tail=2.0,  # Head concentration (faster)
                sorption=freundlich_sorption,
            )

            # Characteristic that might intersect with head or tail
            char = CharacteristicWave(
                t_start=10.0, v_start=0.0, flow=100.0, concentration=3.0, sorption=freundlich_sorption
            )

            results = find_rarefaction_boundary_intersections(raref, char, t_current=10.0)

            # Should return list of intersections
            assert isinstance(results, list)
            for t, v, boundary in results:
                assert boundary in ["head", "tail"]
                assert t >= 10.0  # Must be in future
                assert v >= 0  # Must be at positive position

    def test_rarefaction_shock_intersection(self, freundlich_sorption):
        """Test rarefaction boundary intersecting with shock."""
        # For n=2.0 (unfavorable): higher concentration = faster velocity
        # Head (faster) has c=5.0, tail (slower) has c=2.0
        raref = RarefactionWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_head=5.0,  # Higher concentration (faster for n>1)
            c_tail=2.0,  # Lower concentration (slower for n>1)
            sorption=freundlich_sorption,
        )

        shock = ShockWave(t_start=10.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=1.0, sorption=freundlich_sorption)

        results = find_rarefaction_boundary_intersections(raref, shock, t_current=10.0)

        assert isinstance(results, list)
        # May or may not have intersections depending on velocities


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestOutletCrossing:
    """Test find_outlet_crossing function."""

    def test_characteristic_outlet_crossing(self, freundlich_sorption):
        """Test characteristic crossing outlet."""
        char = CharacteristicWave(t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=freundlich_sorption)

        v_outlet = 500.0
        t_current = 0.0

        t_cross = find_outlet_crossing(char, v_outlet, t_current)

        # MUST return a crossing time
        assert t_cross is not None, "Expected characteristic to cross outlet"

        # Verify characteristic is at outlet at crossing time
        v_at_cross = characteristic_position(
            char.concentration, char.flow, char.sorption, char.t_start, char.v_start, t_cross
        )
        assert np.isclose(v_at_cross, v_outlet, rtol=1e-14)

    def test_shock_outlet_crossing(self, freundlich_sorption):
        """Test shock crossing outlet."""
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption)

        v_outlet = 500.0
        t_current = 0.0

        t_cross = find_outlet_crossing(shock, v_outlet, t_current)

        # MUST return a crossing time
        assert t_cross is not None, "Expected shock to cross outlet"

        # Verify shock is at outlet at crossing time
        v_at_cross = shock.v_start + shock.velocity * (t_cross - shock.t_start)
        assert np.isclose(v_at_cross, v_outlet, rtol=1e-14)

    def test_rarefaction_outlet_crossing(self, freundlich_sorption):
        """Test rarefaction head crossing outlet."""
        # For n=2.0 (unfavorable): higher concentration = faster velocity
        # Head (faster) has c=5.0, tail (slower) has c=2.0
        raref = RarefactionWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_head=5.0,  # Higher concentration (faster for n>1)
            c_tail=2.0,  # Lower concentration (slower for n>1)
            sorption=freundlich_sorption,
        )

        v_outlet = 500.0
        t_current = 0.0

        t_cross = find_outlet_crossing(raref, v_outlet, t_current)

        # MUST return a crossing time
        assert t_cross is not None, "Expected rarefaction head to cross outlet"

        # Verify head is at outlet at crossing time
        v_head = characteristic_position(
            raref.c_head, raref.flow, raref.sorption, raref.t_start, raref.v_start, t_cross
        )
        assert np.isclose(v_head, v_outlet, rtol=1e-14)

    def test_wave_already_past_outlet(self, freundlich_sorption):
        """Test wave that already passed outlet returns None."""
        char = CharacteristicWave(
            t_start=0.0,
            v_start=600.0,  # Start beyond outlet
            flow=100.0,
            concentration=5.0,
            sorption=freundlich_sorption,
        )

        v_outlet = 500.0
        t_current = 0.0

        t_cross = find_outlet_crossing(char, v_outlet, t_current)
        assert t_cross is None

    def test_inactive_wave_returns_none(self, freundlich_sorption):
        """Test inactive wave returns None."""
        char = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=freundlich_sorption, is_active=False
        )

        v_outlet = 500.0
        t_current = 0.0

        t_cross = find_outlet_crossing(char, v_outlet, t_current)
        assert t_cross is None


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestMachinePrecision:
    """Test that all calculations achieve machine precision."""

    def test_roundtrip_precision_characteristic(self, freundlich_sorption):
        """Test characteristic intersection has machine precision."""
        # Create two characteristics that will definitely intersect
        # char1 starts early with LOWER concentration (slower for this sorption)
        char1 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=10000.0, concentration=1.0, sorption=freundlich_sorption
        )

        # char2 starts later at same position, HIGHER concentration (faster for this sorption)
        char2 = CharacteristicWave(
            t_start=100.0, v_start=0.0, flow=10000.0, concentration=10.0, sorption=freundlich_sorption
        )

        result = find_characteristic_intersection(char1, char2, t_current=100.0)

        # MUST return an intersection
        assert result is not None, "Expected intersection for machine precision test"

        t_int, v_int = result

        # Compute positions independently
        v1 = characteristic_position(
            char1.concentration, char1.flow, char1.sorption, char1.t_start, char1.v_start, t_int
        )
        v2 = characteristic_position(
            char2.concentration, char2.flow, char2.sorption, char2.t_start, char2.v_start, t_int
        )

        # Should be identical to machine precision
        assert np.isclose(v1, v2, rtol=1e-14, atol=1e-15)
        assert np.isclose(v1, v_int, rtol=1e-14, atol=1e-15)

    def test_roundtrip_precision_shock(self, freundlich_sorption):
        """Test shock-shock intersection has machine precision."""
        # shock1 starts at t=0, V=0
        shock1 = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=freundlich_sorption)

        # shock2 starts at t=0, V=300 (ahead of shock1)
        shock2 = ShockWave(
            t_start=0.0, v_start=300.0, flow=100.0, c_left=8.0, c_right=1.0, sorption=freundlich_sorption
        )

        result = find_shock_shock_intersection(shock1, shock2, t_current=0.0)

        # MUST return an intersection
        assert result is not None, "Expected shock intersection for machine precision test"

        t_int, v_int = result

        # Compute positions independently
        v1 = shock1.v_start + shock1.velocity * (t_int - shock1.t_start)
        v2 = shock2.v_start + shock2.velocity * (t_int - shock2.t_start)

        # Should be identical to machine precision
        assert np.isclose(v1, v2, rtol=1e-14, atol=1e-15)
        assert np.isclose(v1, v_int, rtol=1e-14, atol=1e-15)
