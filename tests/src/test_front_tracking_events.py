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
        """Test catching-from-behind scenarios for both n>1 and n<1."""
        if freundlich_sorption.n < 1.0:
            # For n<1, lower concentration has higher velocity (faster).
            # char_fast: low concentration, starts early and ahead
            char_fast = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=10000.0, concentration=1.0, sorption=freundlich_sorption
            )

            # char_slow: high concentration, starts later from same position
            # It is slower and therefore can never catch up with char_fast.
            char_slow = CharacteristicWave(
                t_start=100.0, v_start=0.0, flow=10000.0, concentration=10.0, sorption=freundlich_sorption
            )

            result = find_characteristic_intersection(char_fast, char_slow, t_current=100.0)
            assert result is None, "Expected no intersection when slower characteristic starts behind for n<1"
        elif freundlich_sorption.n > 1.0:
            # For n>1, higher concentration has higher velocity.
            # char1: lower c, slower, starts early and ahead
            char1 = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=10000.0, concentration=1.0, sorption=freundlich_sorption
            )

            # char2: higher c, faster, starts later but behind
            char2 = CharacteristicWave(
                t_start=100.0, v_start=0.0, flow=10000.0, concentration=10.0, sorption=freundlich_sorption
            )

            result = find_characteristic_intersection(char1, char2, t_current=100.0)
            assert result is not None, "Expected intersection when faster characteristic starts behind for n>1"

            t_int, v_int = result
            v1 = characteristic_position(
                char1.concentration, char1.flow, char1.sorption, char1.t_start, char1.v_start, t_int
            )
            v2 = characteristic_position(
                char2.concentration, char2.flow, char2.sorption, char2.t_start, char2.v_start, t_int
            )
            assert np.isclose(v1, v2, rtol=1e-14)  # type: ignore[no-matching-overload]
            assert np.isclose(v1, v_int, rtol=1e-14)  # type: ignore[no-matching-overload]
        # For n == 1.0 (if ever used), this test is not defined.

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
        """Test two shock configurations for n>1 and n<1."""
        if freundlich_sorption.n < 1.0:
            shock1 = ShockWave(
                t_start=0.0, v_start=0.0, flow=100.0, c_left=5.0, c_right=0.0, sorption=freundlich_sorption
            )

            shock2 = ShockWave(
                t_start=10.0, v_start=50.0, flow=100.0, c_left=3.0, c_right=0.0, sorption=freundlich_sorption
            )

            result = find_shock_shock_intersection(shock1, shock2, t_current=0.0)
            assert result is None, "Expected no intersection between these shocks for n<1"
        elif freundlich_sorption.n > 1.0:
            shock1 = ShockWave(
                t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption
            )

            shock2 = ShockWave(
                t_start=0.0, v_start=200.0, flow=100.0, c_left=5.0, c_right=0.0, sorption=freundlich_sorption
            )

            result = find_shock_shock_intersection(shock1, shock2, t_current=0.0)
            assert result is not None, "Expected intersection between two shocks for n>1"

            t_int, v_int = result
            assert shock1.velocity is not None
            assert shock2.velocity is not None
            v1 = shock1.v_start + shock1.velocity * (t_int - shock1.t_start)
            v2 = shock2.v_start + shock2.velocity * (t_int - shock2.t_start)
            assert np.isclose(v1, v2, rtol=1e-14)  # type: ignore[no-matching-overload]
            assert np.isclose(v1, v_int, rtol=1e-14)  # type: ignore[no-matching-overload]
        # For n == 1.0, this configuration is not tested.

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
        """Test shock-characteristic interactions for n>1 and n<1."""
        if freundlich_sorption.n < 1.0:
            # Characteristic ahead with low concentration (fastest for n<1)
            char = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=0.5, sorption=freundlich_sorption
            )

            # Shock behind but, for these parameters and n<1, slower than the characteristic
            shock = ShockWave(
                t_start=10.0, v_start=10.0, flow=100.0, c_left=2.0, c_right=0.0, sorption=freundlich_sorption
            )

            result = find_shock_characteristic_intersection(shock, char, t_current=10.0)
            assert result is None, "Expected no intersection between shock and characteristic for n<1"
        elif freundlich_sorption.n > 1.0:
            # Characteristic ahead with lower concentration (slower for n>1)
            char = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=2.0, sorption=freundlich_sorption
            )

            # Shock behind but faster
            shock = ShockWave(
                t_start=10.0, v_start=10.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption
            )

            result = find_shock_characteristic_intersection(shock, char, t_current=10.0)

            assert result is not None, "Expected shock to catch characteristic for n>1"

            t_int, v_int = result
            assert t_int > 10.0, "Intersection must be in future"
            assert v_int > 0, "Must be at positive position"

            v_char = characteristic_position(
                char.concentration, char.flow, char.sorption, char.t_start, char.v_start, t_int
            )
            assert shock.velocity is not None
            v_shock = shock.v_start + shock.velocity * (t_int - shock.t_start)
            assert np.isclose(v_char, v_shock, rtol=1e-14)  # type: ignore[no-matching-overload]
        # For n == 1.0, behaviour is not covered by this test.

    def test_shock_not_catches_characteristic(self, freundlich_sorption):
        """Test configurations where shock does not catch characteristic."""
        if freundlich_sorption.n < 1.0:
            char = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=0.5, sorption=freundlich_sorption
            )

            shock = ShockWave(
                t_start=10.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption
            )

            result = find_shock_characteristic_intersection(shock, char, t_current=10.0)
            assert result is None, "Expected no intersection between shock and characteristic for n<1"
        elif freundlich_sorption.n > 1.0:
            # Characteristic ahead with higher speed than shock
            char = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=100.0, concentration=10.0, sorption=freundlich_sorption
            )

            shock = ShockWave(
                t_start=10.0, v_start=0.0, flow=100.0, c_left=2.0, c_right=0.0, sorption=freundlich_sorption
            )

            result = find_shock_characteristic_intersection(shock, char, t_current=10.0)
            assert result is None, "Expected no intersection between shock and characteristic for n>1"
        # For n == 1.0, behaviour is not covered by this test.


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestRarefactionIntersections:
    """Test find_rarefaction_boundary_intersections function."""

    def test_rarefaction_head_characteristic_intersection_regime_aware(self, freundlich_sorption):
        """Test rarefaction-characteristic head interactions for n<1 and n>1."""
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            # For n<1, higher concentration => slower. A configuration with
            # c_tail < c_head is not a valid rarefaction; ensure it raises.
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(
                    t_start=0.0,
                    v_start=0.0,
                    flow=flow,
                    c_head=5.0,
                    c_tail=2.0,
                    sorption=freundlich_sorption,
                )
            return

        if freundlich_sorption.n > 1.0:
            # For n>1: higher concentration = faster velocity
            # Head (faster) has c=5.0, tail (slower) has c=2.0
            raref = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=5.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )

            # Characteristic that might intersect with head or tail
            char = CharacteristicWave(
                t_start=10.0, v_start=0.0, flow=flow, concentration=3.0, sorption=freundlich_sorption
            )

            results = find_rarefaction_boundary_intersections(raref, char, t_current=10.0)

            assert isinstance(results, list)
            for t, v, boundary in results:
                assert boundary in {"head", "tail"}
                assert t >= 10.0
                assert v >= 0
        else:
            pytest.skip("This test is only defined for n!=1.")

    def test_rarefaction_head_characteristic_invalid_and_valid_regimes(self, freundlich_sorption):
        """Regime-aware validity check: invalid for n<1, valid for n>1."""
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            # For n<1: higher concentration = slower velocity
            # A configuration with c_tail < c_head is not a valid rarefaction
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(
                    t_start=0.0,
                    v_start=0.0,
                    flow=flow,
                    c_head=5.0,
                    c_tail=2.0,
                    sorption=freundlich_sorption,
                )
        elif freundlich_sorption.n > 1.0:
            # For n>1, the same concentration ordering (c_head>c_tail) is now
            # a valid rarefaction. Verify that it does NOT raise.
            RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=5.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )
        else:
            pytest.skip("This test is only defined for n!=1.")

    def test_rarefaction_shock_intersection(self, freundlich_sorption):
        """Test rarefaction boundary intersecting with shock."""
        if freundlich_sorption.n > 1.0:
            # For n>1: higher concentration = faster velocity
            # Head (faster) has c=5.0, tail (slower) has c=2.0
            raref = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=100.0,
                c_head=5.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )

            shock = ShockWave(
                t_start=10.0,
                v_start=0.0,
                flow=100.0,
                c_left=10.0,
                c_right=1.0,
                sorption=freundlich_sorption,
            )

            results = find_rarefaction_boundary_intersections(raref, shock, t_current=10.0)
            assert isinstance(results, list)
        else:
            # For n<1, this (c_tail < c_head) is not a valid rarefaction and must raise.
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(
                    t_start=0.0,
                    v_start=0.0,
                    flow=100.0,
                    c_head=5.0,
                    c_tail=2.0,
                    sorption=freundlich_sorption,
                )

    def test_valid_rarefaction_head_faster_than_tail_regime_aware(self, freundlich_sorption):
        """In both regimes, valid rarefactions must have head faster than tail."""
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            # For n<1: higher concentration = slower; choose c_tail>c_head so
            # velocities increase from tail to head.
            raref = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=2.0,
                c_tail=5.0,
                sorption=freundlich_sorption,
            )

            v_head = raref.head_velocity()
            v_tail = raref.tail_velocity()

            # By construction, head must be faster than tail for a valid rarefaction.
            assert v_head > v_tail
        elif freundlich_sorption.n > 1.0:
            # For n>1, a valid rarefaction has c_head>c_tail and the same
            # requirement that head is faster than tail.
            raref = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=5.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )

            v_head = raref.head_velocity()
            v_tail = raref.tail_velocity()

            assert v_head > v_tail
        else:
            pytest.skip("This check is only defined for n!=1.")

    def test_valid_rarefaction_boundary_intersection_with_characteristic_regime_aware(self, freundlich_sorption):
        """Regime-aware test of rarefaction boundary intersection with a characteristic."""
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            raref = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=1.0,
                c_tail=4.0,
                sorption=freundlich_sorption,
            )

            # Choose a characteristic whose speed lies between tail and head speeds
            # so that one of the rarefaction boundaries intersects it.
            char = CharacteristicWave(
                t_start=5.0,
                v_start=0.0,
                flow=flow,
                concentration=3.0,
                sorption=freundlich_sorption,
            )

            t_current = 5.0
        elif freundlich_sorption.n > 1.0:
            # Mirror configuration for n>1 with c_head>c_tail but a
            # characteristic whose speed lies between boundary speeds.
            raref = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=5.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )

            char = CharacteristicWave(
                t_start=5.0,
                v_start=0.0,
                flow=flow,
                concentration=3.5,
                sorption=freundlich_sorption,
            )

            t_current = 5.0
        else:
            pytest.skip("This check is only defined for n!=1.")

        intersections = find_rarefaction_boundary_intersections(raref, char, t_current=t_current)

        assert isinstance(intersections, list)
        assert intersections

        t_int, v_int, boundary = intersections[0]
        assert t_int >= t_current
        assert v_int >= 0.0

        if boundary == "head":
            v_raref = characteristic_position(
                raref.c_head, raref.flow, raref.sorption, raref.t_start, raref.v_start, t_int
            )
        else:
            v_raref = characteristic_position(
                raref.c_tail, raref.flow, raref.sorption, raref.t_start, raref.v_start, t_int
            )

        v_char = characteristic_position(
            char.concentration, char.flow, char.sorption, char.t_start, char.v_start, t_int
        )

        assert np.isclose(v_raref, v_char, rtol=1e-14)  # type: ignore[no-matching-overload]


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
        assert np.isclose(v_at_cross, v_outlet, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_shock_outlet_crossing(self, freundlich_sorption):
        """Test shock crossing outlet."""
        shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption)

        v_outlet = 500.0
        t_current = 0.0

        t_cross = find_outlet_crossing(shock, v_outlet, t_current)

        # MUST return a crossing time
        assert t_cross is not None, "Expected shock to cross outlet"

        # Verify shock is at outlet at crossing time
        assert shock.velocity is not None
        v_at_cross = shock.v_start + shock.velocity * (t_cross - shock.t_start)
        assert np.isclose(v_at_cross, v_outlet, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_rarefaction_outlet_crossing(self, freundlich_sorption):
        """Test rarefaction head crossing outlet."""
        v_outlet = 500.0
        t_current = 0.0

        if freundlich_sorption.n < 1.0:
            # For n<1, a rarefaction with c_tail < c_head is invalid; ensure it raises.
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(
                    t_start=0.0,
                    v_start=0.0,
                    flow=100.0,
                    c_head=5.0,
                    c_tail=2.0,
                    sorption=freundlich_sorption,
                )
        elif freundlich_sorption.n > 1.0:
            # For n>1: higher concentration = faster velocity
            # Head (faster) has c=5.0, tail (slower) has c=2.0
            raref = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=100.0,
                c_head=5.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )

            t_cross = find_outlet_crossing(raref, v_outlet, t_current)

            # MUST return a crossing time
            assert t_cross is not None, "Expected rarefaction head to cross outlet"

            # Verify head is at outlet at crossing time
            v_head = characteristic_position(
                raref.c_head, raref.flow, raref.sorption, raref.t_start, raref.v_start, t_cross
            )
            assert np.isclose(v_head, v_outlet, rtol=1e-14)  # type: ignore[no-matching-overload]

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
class TestRarefactionRarefactionIntersections:
    """Test rarefaction-rarefaction boundary intersections."""

    def test_head_or_tail_boundary_intersects_other_rarefaction(self, freundlich_sorption):
        """Test that at least one boundary of a rarefaction intersects another rarefaction."""
        flow = 100.0
        if freundlich_sorption.n < 1.0:
            # For n<1, this configuration is not a valid rarefaction; ensure it raises.
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(
                    t_start=0.0,
                    v_start=0.0,
                    flow=flow,
                    c_head=5.0,
                    c_tail=2.0,
                    sorption=freundlich_sorption,
                )
        elif freundlich_sorption.n > 1.0:
            raref1 = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=5.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )

            raref2 = RarefactionWave(
                t_start=10.0,
                v_start=0.0,
                flow=flow,
                c_head=10.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )

            intersections = find_rarefaction_boundary_intersections(raref1, raref2, t_current=10.0)

            assert intersections

            t_int, v_int, boundary = intersections[0]

            assert t_int > 10.0
            assert v_int >= 0.0

            if boundary == "head":
                v_raref1 = characteristic_position(
                    raref1.c_head, raref1.flow, raref1.sorption, raref1.t_start, raref1.v_start, t_int
                )
            else:
                v_raref1 = characteristic_position(
                    raref1.c_tail, raref1.flow, raref1.sorption, raref1.t_start, raref1.v_start, t_int
                )

            assert np.isclose(v_raref1, v_int, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_parallel_rarefaction_boundaries_do_not_intersect(self, freundlich_sorption):
        """Test that parallel rarefaction boundaries do not intersect."""
        flow = 100.0
        if freundlich_sorption.n > 1.0:
            raref1 = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=5.0,
                c_tail=1.0,
                sorption=freundlich_sorption,
            )

            raref2 = RarefactionWave(
                t_start=0.0,
                v_start=10.0,
                flow=flow,
                c_head=5.0,
                c_tail=1.0,
                sorption=freundlich_sorption,
            )

            intersections_1 = find_rarefaction_boundary_intersections(raref1, raref2, t_current=0.0)
            intersections_2 = find_rarefaction_boundary_intersections(raref2, raref1, t_current=0.0)

            # With equal speeds and offset starts, there should be at most a single
            # boundary intersection where the heads coincide.
            assert len(intersections_1) <= 1
            assert len(intersections_2) <= 1
            if intersections_1:
                t_int, v_int, boundary = intersections_1[0]
                assert boundary == "head"
                assert t_int >= 0.0
                assert v_int >= 0.0
        else:
            # For n<1, this configuration is not a valid rarefaction; just
            # verify that construction fails as expected.
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(
                    t_start=0.0,
                    v_start=0.0,
                    flow=flow,
                    c_head=5.0,
                    c_tail=1.0,
                    sorption=freundlich_sorption,
                )

    def test_valid_n_less_than_one_rarefactions_with_boundary_intersection(self, freundlich_sorption):
        """For 0<n<1, two valid rarefactions can have a boundary intersection.

        Both rarefactions satisfy c_tail > c_head (required for n<1), and we
        choose parameters so that a boundary of raref1 intersects a boundary of
        raref2 in the future. We then verify t_int, v_int, the reported
        boundary label, and consistency with characteristic_position.
        """
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            # For n<1: higher concentration => slower characteristics.
            # Valid rarefactions must have c_tail > c_head so that
            # velocity(head) > velocity(tail).
            raref1 = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=1.0,
                c_tail=4.0,
                sorption=freundlich_sorption,
            )

            # Second rarefaction starts later but with a faster head so that
            # one of raref1's boundaries intersects one of raref2's boundaries
            # in the future.
            raref2 = RarefactionWave(
                t_start=10.0,
                v_start=0.0,
                flow=flow,
                c_head=2.0,
                c_tail=5.0,
                sorption=freundlich_sorption,
            )
        elif freundlich_sorption.n > 1.0:
            # For n>1: higher concentration => faster characteristics.
            # Valid rarefactions typically have c_head > c_tail so that
            # velocity(head) > velocity(tail).
            raref1 = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=5.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            )

            # Choose raref2 so that one of raref1's boundaries intersects a
            # boundary of raref2 in the future.
            raref2 = RarefactionWave(
                t_start=5.0,
                v_start=20.0,
                flow=flow,
                c_head=8.0,
                c_tail=3.0,
                sorption=freundlich_sorption,
            )
        else:
            pytest.skip("This test is only defined for n!=1.")

        intersections = find_rarefaction_boundary_intersections(raref1, raref2, t_current=10.0)

        assert isinstance(intersections, list)
        assert intersections, "Expected at least one boundary intersection for valid rarefactions"

        t_int, v_int, boundary = intersections[0]

        assert t_int >= 10.0
        assert v_int >= 0.0
        assert boundary in {"head", "tail"}

        if boundary == "head":
            v_raref1 = characteristic_position(
                raref1.c_head, raref1.flow, raref1.sorption, raref1.t_start, raref1.v_start, t_int
            )
        else:
            v_raref1 = characteristic_position(
                raref1.c_tail, raref1.flow, raref1.sorption, raref1.t_start, raref1.v_start, t_int
            )

        # Position of intersecting boundary of raref1 must match v_int to
        # machine precision.
        assert np.isclose(v_raref1, v_int, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_valid_rarefactions_without_intersection_regime_aware(self, freundlich_sorption):
        """Regime-aware test: separation for n<1, controlled tail intersection for n>1."""
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            # n < 1: higher concentration => slower.
            # raref_front: low concentrations (faster), starting ahead at v=200.
            raref_front = RarefactionWave(
                t_start=0.0,
                v_start=200.0,
                flow=flow,
                c_head=0.5,
                c_tail=2.0,  # c_tail > c_head (valid rarefaction for n<1)
                sorption=freundlich_sorption,
            )

            # raref_back: higher concentrations (slower), starting behind at v=0.
            raref_back = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=3.0,
                c_tail=5.0,  # c_tail > c_head (valid rarefaction for n<1)
                sorption=freundlich_sorption,
            )

            t_current = 0.0

            intersections = find_rarefaction_boundary_intersections(raref_front, raref_back, t_current=t_current)

            assert isinstance(intersections, list)
            # For n<1, the faster front rarefaction starts ahead and remains
            # ahead; no boundary intersections are expected.
            assert intersections == []

        elif freundlich_sorption.n > 1.0:
            # n > 1: higher concentration => faster.
            # Use a configuration that we know produces a tail-tail
            # intersection and check it precisely.
            raref_front = RarefactionWave(
                t_start=0.0,
                v_start=200.0,
                flow=flow,
                c_head=8.0,
                c_tail=3.0,  # c_head > c_tail (valid rarefaction for n>1)
                sorption=freundlich_sorption,
            )

            raref_back = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=5.0,
                c_tail=1.0,  # c_head > c_tail (valid rarefaction for n>1)
                sorption=freundlich_sorption,
            )

            t_current = 0.0

            intersections = find_rarefaction_boundary_intersections(raref_front, raref_back, t_current=t_current)

            assert isinstance(intersections, list)
            assert intersections, "Expected at least one boundary intersection for n>1 configuration"

            # First intersection is reported as a tail boundary of raref_front;
            # verify that the reported position lies on that tail characteristic.
            t_int, v_int, boundary = intersections[0]
            assert boundary == "tail"
            assert t_int > t_current
            assert v_int > 0.0

            v_tail_front = characteristic_position(
                raref_front.c_tail,
                raref_front.flow,
                raref_front.sorption,
                raref_front.t_start,
                raref_front.v_start,
                t_int,
            )

            assert np.isclose(v_tail_front, v_int, rtol=1e-14)  # type: ignore[no-matching-overload]

        else:
            pytest.skip("This test is only defined for n!=1.")

    def test_tail_boundary_intersects_other_rarefaction_for_n_greater_than_one(self, freundlich_sorption):
        """Explicit n>1 case where raref1's tail is the intersecting boundary."""
        flow = 100.0
        if freundlich_sorption.n < 1.0:
            pytest.skip("This test is only defined for n>1.")
        elif freundlich_sorption.n > 1.0:
            # For n>1: higher concentration => faster characteristics.
            # Construct raref1 so that its tail is slower than its head, and
            # choose raref2 so that raref1.tail intersects a boundary of raref2.
            raref1 = RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=flow,
                c_head=10.0,
                c_tail=5.0,
                sorption=freundlich_sorption,
            )

            # Place raref2 so that it starts later and slightly ahead, with
            # concentrations chosen so that the slower tail of raref1 can catch
            # one of its boundaries.
            raref2 = RarefactionWave(
                t_start=5.0,
                v_start=50.0,
                flow=flow,
                c_head=8.0,
                c_tail=3.0,
                sorption=freundlich_sorption,
            )

            intersections = find_rarefaction_boundary_intersections(raref1, raref2, t_current=5.0)

            assert intersections, "Expected at least one boundary intersection for n>1 rarefactions"

            t_int, v_int, boundary = intersections[0]

            assert t_int >= 5.0
            assert v_int >= 0.0
            assert boundary in {"head", "tail"}

            # This test focuses on the case where raref1's tail is the
            # intersecting boundary. Ensure that at least one intersection has
            # boundary == "tail" and that the position matches
            # characteristic_position for the tail.
            tail_intersections = [item for item in intersections if item[2] == "tail"]

            assert tail_intersections, "Expected tail boundary of raref1 to intersect another rarefaction for n>1"

            t_tail, v_tail, boundary_tail = tail_intersections[0]
            assert boundary_tail == "tail"

            v_raref1_tail = characteristic_position(
                raref1.c_tail, raref1.flow, raref1.sorption, raref1.t_start, raref1.v_start, t_tail
            )

            assert np.isclose(v_raref1_tail, v_tail, rtol=1e-14)  # type: ignore[no-matching-overload]


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestShockVelocityAndEntropy:
    """Test shock velocity calculations and entropy conditions."""

    def test_shock_velocity_ordering_for_different_jumps(self, freundlich_sorption):
        """Verify shock velocities follow correct ordering for n>1 and n<1.

        Note: For C>0, the ordering depends on n.
        """
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            # For n<1 and C>0: higher concentration → slower velocity
            # Shock with lower c_left (faster) should have higher velocity
            shock_low = ShockWave(
                t_start=0.0, v_start=0.0, flow=flow, c_left=2.0, c_right=10.0, sorption=freundlich_sorption
            )
            shock_high = ShockWave(
                t_start=0.0, v_start=0.0, flow=flow, c_left=5.0, c_right=10.0, sorption=freundlich_sorption
            )

            # shock_low (c_left=2) should be faster than shock_high (c_left=5)
            # because lower concentrations are faster for n<1
            assert shock_low.velocity is not None
            assert shock_high.velocity is not None
            assert shock_low.velocity > shock_high.velocity, "For n<1, shock with lower c_left should be faster"

        elif freundlich_sorption.n > 1.0:
            # For n>1 and C>0: higher concentration → faster velocity
            # Shock with higher c_left (faster) should have higher velocity
            shock_high = ShockWave(
                t_start=0.0, v_start=0.0, flow=flow, c_left=10.0, c_right=2.0, sorption=freundlich_sorption
            )
            shock_low = ShockWave(
                t_start=0.0, v_start=0.0, flow=flow, c_left=5.0, c_right=2.0, sorption=freundlich_sorption
            )

            # shock_high (c_left=10) should be faster than shock_low (c_left=5)
            # because higher concentrations are faster for n>1 (for C>0)
            assert shock_high.velocity is not None
            assert shock_low.velocity is not None
            assert shock_high.velocity > shock_low.velocity, "For n>1, shock with higher c_left should be faster"

    def test_shock_satisfies_entropy_condition(self, freundlich_sorption):
        """Verify physically valid compression shocks satisfy the Lax entropy condition.

        Note: C=0 is excluded because v(C=0)=flow is faster than any C>0,
        making shocks to/from zero invalid (they would be rarefactions).
        """
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            # For n<1 and C>0: higher c → slower v
            # Valid compression: faster water (low c) catches slower water (high c)
            # So c_left < c_right (low c behind, high c ahead, creates shock)
            test_shocks = [
                (1.0, 10.0),  # Low c catching high c
                (2.0, 5.0),  # Low c catching high c
                (1.0, 5.0),  # Low c catching high c
            ]

            for c_left, c_right in test_shocks:
                shock = ShockWave(
                    t_start=0.0, v_start=0.0, flow=flow, c_left=c_left, c_right=c_right, sorption=freundlich_sorption
                )
                assert shock.satisfies_entropy(), f"Shock {c_left}→{c_right} must satisfy entropy for n<1"

        elif freundlich_sorption.n > 1.0:
            # For n>1 and C>0: higher c → faster v
            # Valid compression: faster water (high c) catches slower water (low c)
            # So c_left > c_right (high c behind, low c ahead, creates shock)
            test_shocks = [
                (10.0, 2.0),  # High c catching low c
                (5.0, 1.0),  # High c catching low c
                (10.0, 5.0),  # High c catching low c
            ]

            for c_left, c_right in test_shocks:
                shock = ShockWave(
                    t_start=0.0, v_start=0.0, flow=flow, c_left=c_left, c_right=c_right, sorption=freundlich_sorption
                )
                assert shock.satisfies_entropy(), f"Shock {c_left}→{c_right} must satisfy entropy for n>1"

    def test_characteristic_velocity_vs_shock_velocity(self, freundlich_sorption):
        """Verify characteristic velocities bracket shock velocity (entropy condition)."""
        flow = 100.0

        if freundlich_sorption.n < 1.0:
            # For n<1: create shock and verify char velocities bracket it
            c_left = 10.0
            c_right = 2.0

            shock = ShockWave(
                t_start=0.0, v_start=0.0, flow=flow, c_left=c_left, c_right=c_right, sorption=freundlich_sorption
            )

            char_left = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=flow, concentration=c_left, sorption=freundlich_sorption
            )

            char_right = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=flow, concentration=c_right, sorption=freundlich_sorption
            )

            # For n<1: higher c → slower v, so c_left (higher) → slower, c_right (lower) → faster
            # Entropy requires: v(c_left) < shock_v < v(c_right)
            assert shock.velocity is not None
            assert char_left.velocity() < shock.velocity < char_right.velocity(), (
                "Entropy condition violated: characteristic velocities must bracket shock velocity for n<1"
            )

        elif freundlich_sorption.n > 1.0:
            # For n>1: create shock and verify char velocities bracket it
            c_left = 10.0
            c_right = 2.0

            shock = ShockWave(
                t_start=0.0, v_start=0.0, flow=flow, c_left=c_left, c_right=c_right, sorption=freundlich_sorption
            )

            char_left = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=flow, concentration=c_left, sorption=freundlich_sorption
            )

            char_right = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=flow, concentration=c_right, sorption=freundlich_sorption
            )

            # For n>1: higher c → faster v, so c_left (higher) → faster, c_right (lower) → slower
            # Entropy requires: v(c_left) > shock_v > v(c_right)
            assert shock.velocity is not None
            assert char_left.velocity() > shock.velocity > char_right.velocity(), (
                "Entropy condition violated: characteristic velocities must bracket shock velocity for n>1"
            )


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestMachinePrecision:
    """Test that all calculations achieve machine precision."""

    def test_roundtrip_precision_characteristic(self, freundlich_sorption):
        """Test characteristic intersection has machine precision."""
        if freundlich_sorption.n < 1.0:
            # For n<1 and these parameters, there is no intersection; just verify this.
            char1 = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=10000.0, concentration=1.0, sorption=freundlich_sorption
            )

            char2 = CharacteristicWave(
                t_start=100.0, v_start=0.0, flow=10000.0, concentration=10.0, sorption=freundlich_sorption
            )

            result = find_characteristic_intersection(char1, char2, t_current=100.0)
            assert result is None
        elif freundlich_sorption.n > 1.0:
            # Create two characteristics that will intersect
            char1 = CharacteristicWave(
                t_start=0.0, v_start=0.0, flow=10000.0, concentration=1.0, sorption=freundlich_sorption
            )

            char2 = CharacteristicWave(
                t_start=100.0, v_start=0.0, flow=10000.0, concentration=10.0, sorption=freundlich_sorption
            )

            result = find_characteristic_intersection(char1, char2, t_current=100.0)

            assert result is not None, "Expected intersection for machine precision test"

            t_int, v_int = result

            v1 = characteristic_position(
                char1.concentration, char1.flow, char1.sorption, char1.t_start, char1.v_start, t_int
            )
            v2 = characteristic_position(
                char2.concentration, char2.flow, char2.sorption, char2.t_start, char2.v_start, t_int
            )

            assert np.isclose(v1, v2, rtol=1e-14, atol=1e-15)  # type: ignore[no-matching-overload]
            assert np.isclose(v1, v_int, rtol=1e-14, atol=1e-15)  # type: ignore[no-matching-overload]

    def test_roundtrip_precision_shock(self, freundlich_sorption):
        """Test shock-shock intersection has machine precision."""
        if freundlich_sorption.n < 1.0:
            shock1 = ShockWave(
                t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=freundlich_sorption
            )

            shock2 = ShockWave(
                t_start=0.0, v_start=300.0, flow=100.0, c_left=8.0, c_right=1.0, sorption=freundlich_sorption
            )

            result = find_shock_shock_intersection(shock1, shock2, t_current=0.0)
            assert result is None
        elif freundlich_sorption.n > 1.0:
            shock1 = ShockWave(
                t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=freundlich_sorption
            )

            shock2 = ShockWave(
                t_start=0.0, v_start=300.0, flow=100.0, c_left=8.0, c_right=1.0, sorption=freundlich_sorption
            )

            result = find_shock_shock_intersection(shock1, shock2, t_current=0.0)

            assert result is not None, "Expected shock intersection for machine precision test"

            t_int, v_int = result

            assert shock1.velocity is not None
            assert shock2.velocity is not None
            v1 = shock1.v_start + shock1.velocity * (t_int - shock1.t_start)
            v2 = shock2.v_start + shock2.velocity * (t_int - shock2.t_start)

            assert np.isclose(v1, v2, rtol=1e-14, atol=1e-15)  # type: ignore[no-matching-overload]
            assert np.isclose(v1, v_int, rtol=1e-14, atol=1e-15)  # type: ignore[no-matching-overload]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
