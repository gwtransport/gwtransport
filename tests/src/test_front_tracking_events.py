"""
Unit Tests for Event Detection Module in (V, θ) coordinates.
============================================================

All intersection math is pure (V, θ) line geometry — no flow appears.
Tests verify machine-precision accuracy (rtol=1e-14) for intersection
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
from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    characteristic_position,
    characteristic_speed,
)
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
        event = Event(theta=15.5, event_type=EventType.SHOCK_CHAR_COLLISION, waves_involved=[], location=250.0)
        assert event.theta == 15.5
        assert event.event_type == EventType.SHOCK_CHAR_COLLISION
        assert event.location == 250.0

    def test_event_ordering(self):
        """Events carry the θ the solver's ``(theta, counter, ...)`` queue orders by.

        ``Event`` defines no ``__lt__``; the solver's priority queue sorts the
        scheduling tuples, keyed on ``theta`` first. This asserts that ordering
        key, not a (removed) ``Event``-level comparison operator.
        """
        event1 = Event(theta=10.0, event_type=EventType.OUTLET_CROSSING, waves_involved=[], location=500.0)
        event2 = Event(theta=5.0, event_type=EventType.CHAR_CHAR_COLLISION, waves_involved=[], location=100.0)

        ordered = sorted([event1, event2], key=lambda e: e.theta)
        assert ordered == [event2, event1]
        assert event2.theta < event1.theta

    def test_event_type_enum(self):
        """Test EventType enum values."""
        assert EventType.CHAR_CHAR_COLLISION.value == "characteristic_collision"
        assert EventType.SHOCK_SHOCK_COLLISION.value == "shock_collision"
        assert EventType.OUTLET_CROSSING.value == "outlet_crossing"


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestCharacteristicIntersection:
    """Test find_characteristic_intersection function."""

    def test_intersection_simple_case(self, freundlich_sorption):
        """Two characteristics from same point with different concentrations diverge."""
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption)
        char2 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)

        result = find_characteristic_intersection(char1, char2, theta_current=0.0)
        assert result is None

    def test_intersection_catching_from_behind(self, freundlich_sorption):
        """Faster characteristic catches slower one when starting later from behind."""
        if freundlich_sorption.n < 1.0:
            # For n<1, lower c is faster. char_fast starts ahead at θ=0.
            char_fast = CharacteristicWave(
                theta_start=0.0, v_start=0.0, concentration=1.0, sorption=freundlich_sorption
            )
            # char_slow (high c, slow) starts later from same v; can never catch.
            char_slow = CharacteristicWave(
                theta_start=100.0, v_start=0.0, concentration=10.0, sorption=freundlich_sorption
            )

            result = find_characteristic_intersection(char_fast, char_slow, theta_current=100.0)
            assert result is None, "Expected no intersection when slower char starts behind for n<1"
        elif freundlich_sorption.n > 1.0:
            # For n>1, higher c is faster.
            char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=1.0, sorption=freundlich_sorption)
            char2 = CharacteristicWave(theta_start=100.0, v_start=0.0, concentration=10.0, sorption=freundlich_sorption)

            result = find_characteristic_intersection(char1, char2, theta_current=100.0)
            assert result is not None, "Expected intersection when faster char starts behind for n>1"

            theta_int, v_int = result
            v1 = characteristic_position(
                char1.concentration, char1.sorption, char1.theta_start, char1.v_start, theta_int
            )
            v2 = characteristic_position(
                char2.concentration, char2.sorption, char2.theta_start, char2.v_start, theta_int
            )
            assert v1 is not None
            assert v2 is not None
            assert np.isclose(v1, v2, rtol=1e-14)
            assert np.isclose(v1, v_int, rtol=1e-14)

    def test_parallel_characteristics(self, freundlich_sorption):
        """Two characteristics with the same concentration (same speed) never intersect."""
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption)
        char2 = CharacteristicWave(
            theta_start=0.0,
            v_start=10.0,  # Different start position
            concentration=5.0,  # Same concentration → same speed
            sorption=freundlich_sorption,
        )

        result = find_characteristic_intersection(char1, char2, theta_current=0.0)
        assert result is None


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestShockShockIntersection:
    """Test find_shock_shock_intersection function."""

    def test_shock_collision_simple(self, freundlich_sorption):
        """Two shock configurations for n>1 and n<1."""
        if freundlich_sorption.n < 1.0:
            shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=5.0, c_right=0.0, sorption=freundlich_sorption)
            shock2 = ShockWave(theta_start=10.0, v_start=50.0, c_left=3.0, c_right=0.0, sorption=freundlich_sorption)

            result = find_shock_shock_intersection(shock1, shock2, theta_current=0.0)
            assert result is None, "Expected no intersection between these shocks for n<1"
        elif freundlich_sorption.n > 1.0:
            shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption)
            shock2 = ShockWave(theta_start=0.0, v_start=200.0, c_left=5.0, c_right=0.0, sorption=freundlich_sorption)

            result = find_shock_shock_intersection(shock1, shock2, theta_current=0.0)
            assert result is not None, "Expected intersection between two shocks for n>1"

            theta_int, v_int = result
            assert shock1.speed is not None
            assert shock2.speed is not None
            v1 = shock1.v_start + shock1.speed * (theta_int - shock1.theta_start)
            v2 = shock2.v_start + shock2.speed * (theta_int - shock2.theta_start)
            assert np.isclose(v1, v2, rtol=1e-14)
            assert np.isclose(v1, v_int, rtol=1e-14)

    def test_parallel_shocks(self, freundlich_sorption):
        """Two shocks with the same speed never intersect."""
        shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)
        shock2 = ShockWave(theta_start=10.0, v_start=50.0, c_left=10.0, c_right=5.0, sorption=freundlich_sorption)

        result = find_shock_shock_intersection(shock1, shock2, theta_current=10.0)
        assert result is None


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestShockCharacteristicIntersection:
    """Test find_shock_characteristic_intersection function."""

    def test_shock_catches_characteristic(self, freundlich_sorption):
        """Shock-characteristic interactions for n>1 and n<1.

        Geometry: shock starts behind a slower characteristic and (for n>1)
        catches it. Selected θ-scales place the shock at v_shock < v_char
        when both are first defined together.
        """
        if freundlich_sorption.n < 1.0:
            # For n<1, low c is fast — char with c=0.5 is fast and far ahead.
            char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=0.5, sorption=freundlich_sorption)
            # Shock between c_left=2 and c_right=0 — slower than the c=0.5 char in n<1.
            shock = ShockWave(theta_start=1000.0, v_start=10.0, c_left=2.0, c_right=0.0, sorption=freundlich_sorption)

            result = find_shock_characteristic_intersection(shock, char, theta_current=1000.0)
            assert result is None, "Expected no intersection between shock and characteristic for n<1"
        elif freundlich_sorption.n > 1.0:
            # For n>1, the c=2 char is slow. shock (c_left=10) is faster and
            # placed behind it at θ=1000, v=10 (v_char at θ=1000 is ~53).
            char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=2.0, sorption=freundlich_sorption)
            shock = ShockWave(theta_start=1000.0, v_start=10.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption)

            result = find_shock_characteristic_intersection(shock, char, theta_current=1000.0)

            assert result is not None, "Expected shock to catch characteristic for n>1"

            theta_int, v_int = result
            assert theta_int > 1000.0, "Intersection must be in future"
            assert v_int > 0, "Must be at positive position"

            v_char = characteristic_position(
                char.concentration, char.sorption, char.theta_start, char.v_start, theta_int
            )
            assert v_char is not None
            assert shock.speed is not None
            v_shock = shock.v_start + shock.speed * (theta_int - shock.theta_start)
            assert np.isclose(v_char, v_shock, rtol=1e-14)
            # The returned v_int must equal both reconstructed positions.
            assert np.isclose(v_int, v_shock, rtol=1e-14)
            assert np.isclose(v_int, v_char, rtol=1e-14)

    def test_shock_not_catches_characteristic(self, freundlich_sorption):
        """Configurations where shock does not catch characteristic."""
        if freundlich_sorption.n < 1.0:
            char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=0.5, sorption=freundlich_sorption)
            shock = ShockWave(theta_start=1000.0, v_start=0.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption)

            result = find_shock_characteristic_intersection(shock, char, theta_current=1000.0)
            assert result is None
        elif freundlich_sorption.n > 1.0:
            # For n>1, char c=10 is fastest; shock c_left=2 is slower → never catches.
            char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=10.0, sorption=freundlich_sorption)
            shock = ShockWave(theta_start=1000.0, v_start=0.0, c_left=2.0, c_right=0.0, sorption=freundlich_sorption)

            result = find_shock_characteristic_intersection(shock, char, theta_current=1000.0)
            assert result is None


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestRarefactionIntersections:
    """Test find_rarefaction_boundary_intersections function."""

    def test_rarefaction_head_characteristic_intersection_regime_aware(self, freundlich_sorption):
        """Rarefaction-characteristic head interactions for n<1 and n>1."""
        if freundlich_sorption.n < 1.0:
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
            return

        if freundlich_sorption.n > 1.0:
            raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)

            char = CharacteristicWave(theta_start=10.0, v_start=0.0, concentration=3.0, sorption=freundlich_sorption)

            results = find_rarefaction_boundary_intersections(raref, char, theta_current=10.0)

            assert results, "Expected a boundary intersection for the n>1 head/tail-characteristic geometry"
            for theta, v, boundary in results:
                assert boundary in {"head", "tail"}
                assert theta >= 10.0
                assert v >= 0
                # The returned position must lie on the named boundary characteristic.
                c_boundary = raref.c_head if boundary == "head" else raref.c_tail
                v_boundary = characteristic_position(
                    c_boundary, raref.sorption, raref.theta_start, raref.v_start, theta
                )
                assert v_boundary is not None
                assert np.isclose(v_boundary, v, rtol=1e-14)

    def test_rarefaction_head_characteristic_invalid_and_valid_regimes(self, freundlich_sorption):
        """Regime-aware validity check: invalid for n<1, valid for n>1."""
        if freundlich_sorption.n < 1.0:
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
        elif freundlich_sorption.n > 1.0:
            RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)

    def test_rarefaction_shock_intersection(self, freundlich_sorption):
        """Rarefaction boundary intersecting with shock."""
        if freundlich_sorption.n > 1.0:
            raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
            shock = ShockWave(theta_start=10.0, v_start=0.0, c_left=10.0, c_right=1.0, sorption=freundlich_sorption)

            results = find_rarefaction_boundary_intersections(raref, shock, theta_current=10.0)
            assert results, "Expected the shock to intersect a rarefaction boundary for n>1"
            for theta, v, boundary in results:
                assert boundary in {"head", "tail"}
                assert theta > 10.0
                # The returned position lies on both the named boundary and the shock line.
                c_boundary = raref.c_head if boundary == "head" else raref.c_tail
                v_boundary = characteristic_position(
                    c_boundary, raref.sorption, raref.theta_start, raref.v_start, theta
                )
                assert shock.speed is not None
                v_shock = shock.v_start + shock.speed * (theta - shock.theta_start)
                assert v_boundary is not None
                assert np.isclose(v_boundary, v, rtol=1e-14)
                assert np.isclose(v_shock, v, rtol=1e-14)
        else:
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)

    def test_valid_rarefaction_head_faster_than_tail_regime_aware(self, freundlich_sorption):
        """Valid rarefactions have head faster than tail."""
        if freundlich_sorption.n < 1.0:
            raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=2.0, c_tail=5.0, sorption=freundlich_sorption)
            assert raref.head_speed() > raref.tail_speed()
        elif freundlich_sorption.n > 1.0:
            raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
            assert raref.head_speed() > raref.tail_speed()

    def test_valid_rarefaction_boundary_intersection_with_characteristic_regime_aware(self, freundlich_sorption):
        """Rarefaction boundary intersects a characteristic whose speed lies between head and tail."""
        if freundlich_sorption.n < 1.0:
            raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=1.0, c_tail=4.0, sorption=freundlich_sorption)
            char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=3.0, sorption=freundlich_sorption)
            theta_current = 5.0
        elif freundlich_sorption.n > 1.0:
            raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
            char = CharacteristicWave(theta_start=5.0, v_start=0.0, concentration=3.5, sorption=freundlich_sorption)
            theta_current = 5.0
        else:
            pytest.skip("This check is only defined for n!=1.")

        intersections = find_rarefaction_boundary_intersections(raref, char, theta_current=theta_current)

        assert isinstance(intersections, list)
        assert intersections

        theta_int, v_int, boundary = intersections[0]
        assert theta_int >= theta_current
        assert v_int >= 0.0

        if boundary == "head":
            v_raref = characteristic_position(raref.c_head, raref.sorption, raref.theta_start, raref.v_start, theta_int)
        else:
            v_raref = characteristic_position(raref.c_tail, raref.sorption, raref.theta_start, raref.v_start, theta_int)

        v_char = characteristic_position(char.concentration, char.sorption, char.theta_start, char.v_start, theta_int)

        assert v_raref is not None
        assert v_char is not None
        assert np.isclose(v_raref, v_char, rtol=1e-14)


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestOutletCrossing:
    """Test find_outlet_crossing function."""

    def test_characteristic_outlet_crossing(self, freundlich_sorption):
        """Characteristic crosses outlet at θ = V·R(c)."""
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption)

        v_outlet = 500.0
        theta_current = 0.0

        theta_cross = find_outlet_crossing(char, v_outlet, theta_current)

        assert theta_cross is not None, "Expected characteristic to cross outlet"

        v_at_cross = characteristic_position(
            char.concentration, char.sorption, char.theta_start, char.v_start, theta_cross
        )
        assert v_at_cross is not None
        assert np.isclose(v_at_cross, v_outlet, rtol=1e-14)

    def test_shock_outlet_crossing(self, freundlich_sorption):
        """Shock crosses outlet at θ = V / shock_speed."""
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=0.0, sorption=freundlich_sorption)

        v_outlet = 500.0
        theta_current = 0.0

        theta_cross = find_outlet_crossing(shock, v_outlet, theta_current)

        assert theta_cross is not None, "Expected shock to cross outlet"

        assert shock.speed is not None
        v_at_cross = shock.v_start + shock.speed * (theta_cross - shock.theta_start)
        assert np.isclose(v_at_cross, v_outlet, rtol=1e-14)

    def test_rarefaction_returns_none(self, freundlich_sorption):
        """Rarefactions are not handled by find_outlet_crossing; it returns None.

        Production routes rarefaction outlet crossings through the solver's
        explicit head/tail boundary logic (``solver.find_next_event``) and
        ``output.py``; ``find_outlet_crossing`` only covers characteristics,
        shocks, and decaying shocks, returning ``None`` for a rarefaction.
        """
        if freundlich_sorption.n < 1.0:
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
        elif freundlich_sorption.n > 1.0:
            raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
            assert find_outlet_crossing(raref, v_outlet=500.0, theta_current=0.0) is None

    def test_wave_already_past_outlet(self, freundlich_sorption):
        """Wave already past outlet returns None."""
        char = CharacteristicWave(
            theta_start=0.0,
            v_start=600.0,  # Start beyond outlet
            concentration=5.0,
            sorption=freundlich_sorption,
        )

        v_outlet = 500.0
        theta_current = 0.0

        theta_cross = find_outlet_crossing(char, v_outlet, theta_current)
        assert theta_cross is None

    def test_inactive_wave_returns_none(self, freundlich_sorption):
        """Inactive wave returns None."""
        char = CharacteristicWave(
            theta_start=0.0, v_start=0.0, concentration=5.0, sorption=freundlich_sorption, is_active=False
        )

        v_outlet = 500.0
        theta_current = 0.0

        theta_cross = find_outlet_crossing(char, v_outlet, theta_current)
        assert theta_cross is None


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestRarefactionRarefactionIntersections:
    """Test rarefaction-rarefaction boundary intersections."""

    def test_head_or_tail_boundary_intersects_other_rarefaction(self, freundlich_sorption):
        """At least one boundary of one rarefaction intersects the other."""
        if freundlich_sorption.n < 1.0:
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
        elif freundlich_sorption.n > 1.0:
            raref1 = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
            raref2 = RarefactionWave(
                theta_start=10.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=freundlich_sorption
            )

            intersections = find_rarefaction_boundary_intersections(raref1, raref2, theta_current=10.0)

            assert intersections

            theta_int, v_int, boundary = intersections[0]

            assert theta_int > 10.0
            assert v_int >= 0.0

            if boundary == "head":
                v_raref1 = characteristic_position(
                    raref1.c_head, raref1.sorption, raref1.theta_start, raref1.v_start, theta_int
                )
            else:
                v_raref1 = characteristic_position(
                    raref1.c_tail, raref1.sorption, raref1.theta_start, raref1.v_start, theta_int
                )

            assert v_raref1 is not None
            assert np.isclose(v_raref1, v_int, rtol=1e-14)

    def test_parallel_rarefaction_boundaries_do_not_intersect(self, freundlich_sorption):
        """Parallel rarefaction boundaries do not intersect (or share head at v offset)."""
        if freundlich_sorption.n > 1.0:
            raref1 = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=1.0, sorption=freundlich_sorption)
            raref2 = RarefactionWave(
                theta_start=0.0, v_start=10.0, c_head=5.0, c_tail=1.0, sorption=freundlich_sorption
            )

            intersections_1 = find_rarefaction_boundary_intersections(raref1, raref2, theta_current=0.0)
            intersections_2 = find_rarefaction_boundary_intersections(raref2, raref1, theta_current=0.0)

            assert len(intersections_1) <= 1
            assert len(intersections_2) <= 1
            if intersections_1:
                theta_int, v_int, boundary = intersections_1[0]
                assert boundary == "head"
                assert theta_int >= 0.0
                assert v_int >= 0.0
        else:
            with pytest.raises(ValueError, match="Not a rarefaction:"):
                RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=1.0, sorption=freundlich_sorption)

    def test_valid_n_less_than_one_rarefactions_with_boundary_intersection(self, freundlich_sorption):
        """For 0<n<1, two valid rarefactions can have a boundary intersection."""
        if freundlich_sorption.n < 1.0:
            raref1 = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=1.0, c_tail=4.0, sorption=freundlich_sorption)
            raref2 = RarefactionWave(
                theta_start=10.0, v_start=0.0, c_head=2.0, c_tail=5.0, sorption=freundlich_sorption
            )
        elif freundlich_sorption.n > 1.0:
            raref1 = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=2.0, sorption=freundlich_sorption)
            raref2 = RarefactionWave(
                theta_start=5.0, v_start=20.0, c_head=8.0, c_tail=3.0, sorption=freundlich_sorption
            )
        else:
            pytest.skip("This test is only defined for n!=1.")

        intersections = find_rarefaction_boundary_intersections(raref1, raref2, theta_current=10.0)

        assert isinstance(intersections, list)
        assert intersections, "Expected at least one boundary intersection for valid rarefactions"

        theta_int, v_int, boundary = intersections[0]

        assert theta_int >= 10.0
        assert v_int >= 0.0
        assert boundary in {"head", "tail"}

        if boundary == "head":
            v_raref1 = characteristic_position(
                raref1.c_head, raref1.sorption, raref1.theta_start, raref1.v_start, theta_int
            )
        else:
            v_raref1 = characteristic_position(
                raref1.c_tail, raref1.sorption, raref1.theta_start, raref1.v_start, theta_int
            )

        assert v_raref1 is not None
        assert np.isclose(v_raref1, v_int, rtol=1e-14)

    def test_valid_rarefactions_without_intersection_regime_aware(self, freundlich_sorption):
        """Separation for n<1, controlled tail intersection for n>1."""
        if freundlich_sorption.n < 1.0:
            # Faster front rarefaction stays ahead; no boundary intersections.
            raref_front = RarefactionWave(
                theta_start=0.0, v_start=200.0, c_head=0.5, c_tail=2.0, sorption=freundlich_sorption
            )
            raref_back = RarefactionWave(
                theta_start=0.0, v_start=0.0, c_head=3.0, c_tail=5.0, sorption=freundlich_sorption
            )
            theta_current = 0.0

            intersections = find_rarefaction_boundary_intersections(
                raref_front, raref_back, theta_current=theta_current
            )

            assert isinstance(intersections, list)
            assert intersections == []

        elif freundlich_sorption.n > 1.0:
            raref_front = RarefactionWave(
                theta_start=0.0, v_start=200.0, c_head=8.0, c_tail=3.0, sorption=freundlich_sorption
            )
            raref_back = RarefactionWave(
                theta_start=0.0, v_start=0.0, c_head=5.0, c_tail=1.0, sorption=freundlich_sorption
            )
            theta_current = 0.0

            intersections = find_rarefaction_boundary_intersections(
                raref_front, raref_back, theta_current=theta_current
            )

            assert isinstance(intersections, list)
            assert intersections, "Expected at least one boundary intersection for n>1 configuration"

            theta_int, v_int, boundary = intersections[0]
            assert boundary == "tail"
            assert theta_int > theta_current
            assert v_int > 0.0

            v_tail_front = characteristic_position(
                raref_front.c_tail,
                raref_front.sorption,
                raref_front.theta_start,
                raref_front.v_start,
                theta_int,
            )

            assert v_tail_front is not None
            assert np.isclose(v_tail_front, v_int, rtol=1e-14)

    def test_tail_boundary_intersects_other_rarefaction_for_n_greater_than_one(self, freundlich_sorption):
        """Explicit n>1 case where raref1's tail is the intersecting boundary."""
        if freundlich_sorption.n < 1.0:
            pytest.skip("This test is only defined for n>1.")
        elif freundlich_sorption.n > 1.0:
            raref1 = RarefactionWave(
                theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=5.0, sorption=freundlich_sorption
            )
            raref2 = RarefactionWave(
                theta_start=5.0, v_start=50.0, c_head=8.0, c_tail=3.0, sorption=freundlich_sorption
            )

            intersections = find_rarefaction_boundary_intersections(raref1, raref2, theta_current=5.0)

            assert intersections, "Expected at least one boundary intersection for n>1 rarefactions"

            theta_int, v_int, boundary = intersections[0]

            assert theta_int >= 5.0
            assert v_int >= 0.0
            assert boundary in {"head", "tail"}

            tail_intersections = [item for item in intersections if item[2] == "tail"]
            assert tail_intersections, "Expected tail boundary of raref1 to intersect another rarefaction for n>1"

            theta_tail, v_tail, boundary_tail = tail_intersections[0]
            assert boundary_tail == "tail"

            v_raref1_tail = characteristic_position(
                raref1.c_tail, raref1.sorption, raref1.theta_start, raref1.v_start, theta_tail
            )
            assert v_raref1_tail is not None
            assert np.isclose(v_raref1_tail, v_tail, rtol=1e-14)


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestShockVelocityAndEntropy:
    """Test shock velocity calculations and entropy conditions."""

    def test_shock_velocity_ordering_for_different_jumps(self, freundlich_sorption):
        """Verify shock velocities follow correct ordering for n>1 and n<1."""
        if freundlich_sorption.n < 1.0:
            shock_low = ShockWave(theta_start=0.0, v_start=0.0, c_left=2.0, c_right=10.0, sorption=freundlich_sorption)
            shock_high = ShockWave(theta_start=0.0, v_start=0.0, c_left=5.0, c_right=10.0, sorption=freundlich_sorption)

            assert shock_low.speed is not None
            assert shock_high.speed is not None
            assert shock_low.speed > shock_high.speed, "For n<1, shock with lower c_left should be faster"

        elif freundlich_sorption.n > 1.0:
            shock_high = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=freundlich_sorption)
            shock_low = ShockWave(theta_start=0.0, v_start=0.0, c_left=5.0, c_right=2.0, sorption=freundlich_sorption)

            assert shock_high.speed is not None
            assert shock_low.speed is not None
            assert shock_high.speed > shock_low.speed, "For n>1, shock with higher c_left should be faster"

    def test_shock_satisfies_entropy_condition(self, freundlich_sorption):
        """Physically valid compression shocks satisfy the Lax entropy condition."""
        if freundlich_sorption.n < 1.0:
            test_shocks = [(1.0, 10.0), (2.0, 5.0), (1.0, 5.0)]
            for c_left, c_right in test_shocks:
                shock = ShockWave(
                    theta_start=0.0, v_start=0.0, c_left=c_left, c_right=c_right, sorption=freundlich_sorption
                )
                assert shock.satisfies_entropy(), f"Shock {c_left}→{c_right} must satisfy entropy for n<1"

        elif freundlich_sorption.n > 1.0:
            test_shocks = [(10.0, 2.0), (5.0, 1.0), (10.0, 5.0)]
            for c_left, c_right in test_shocks:
                shock = ShockWave(
                    theta_start=0.0, v_start=0.0, c_left=c_left, c_right=c_right, sorption=freundlich_sorption
                )
                assert shock.satisfies_entropy(), f"Shock {c_left}→{c_right} must satisfy entropy for n>1"

    def test_characteristic_velocity_vs_shock_velocity(self, freundlich_sorption):
        """Characteristic speeds bracket shock speed (entropy condition)."""
        if freundlich_sorption.n < 1.0:
            c_left = 10.0
            c_right = 2.0

            shock = ShockWave(
                theta_start=0.0, v_start=0.0, c_left=c_left, c_right=c_right, sorption=freundlich_sorption
            )
            char_left = CharacteristicWave(
                theta_start=0.0, v_start=0.0, concentration=c_left, sorption=freundlich_sorption
            )
            char_right = CharacteristicWave(
                theta_start=0.0, v_start=0.0, concentration=c_right, sorption=freundlich_sorption
            )

            # For n<1: higher c → slower. char_left (high c) slower, char_right (low c) faster.
            # Entropy: λ(c_left) < shock_speed < λ(c_right).
            assert shock.speed is not None
            assert char_left.speed() < shock.speed < char_right.speed(), (
                "Entropy condition violated: characteristic speeds must bracket shock speed for n<1"
            )

        elif freundlich_sorption.n > 1.0:
            c_left = 10.0
            c_right = 2.0

            shock = ShockWave(
                theta_start=0.0, v_start=0.0, c_left=c_left, c_right=c_right, sorption=freundlich_sorption
            )
            char_left = CharacteristicWave(
                theta_start=0.0, v_start=0.0, concentration=c_left, sorption=freundlich_sorption
            )
            char_right = CharacteristicWave(
                theta_start=0.0, v_start=0.0, concentration=c_right, sorption=freundlich_sorption
            )

            # For n>1: higher c → faster. char_left (high c) faster, char_right (low c) slower.
            # Entropy: λ(c_left) > shock_speed > λ(c_right).
            assert shock.speed is not None
            assert char_left.speed() > shock.speed > char_right.speed(), (
                "Entropy condition violated: characteristic speeds must bracket shock speed for n>1"
            )


@pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
class TestMachinePrecision:
    """Test that all calculations achieve machine precision."""

    def test_roundtrip_precision_characteristic(self, freundlich_sorption):
        """Characteristic intersection has machine precision."""
        if freundlich_sorption.n < 1.0:
            char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=1.0, sorption=freundlich_sorption)
            char2 = CharacteristicWave(theta_start=100.0, v_start=0.0, concentration=10.0, sorption=freundlich_sorption)

            result = find_characteristic_intersection(char1, char2, theta_current=100.0)
            assert result is None
        elif freundlich_sorption.n > 1.0:
            char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=1.0, sorption=freundlich_sorption)
            char2 = CharacteristicWave(theta_start=100.0, v_start=0.0, concentration=10.0, sorption=freundlich_sorption)

            result = find_characteristic_intersection(char1, char2, theta_current=100.0)

            assert result is not None, "Expected intersection for machine precision test"

            theta_int, v_int = result

            v1 = characteristic_position(
                char1.concentration, char1.sorption, char1.theta_start, char1.v_start, theta_int
            )
            v2 = characteristic_position(
                char2.concentration, char2.sorption, char2.theta_start, char2.v_start, theta_int
            )
            assert v1 is not None
            assert v2 is not None
            assert np.isclose(v1, v2, rtol=1e-14, atol=1e-15)
            assert np.isclose(v1, v_int, rtol=1e-14, atol=1e-15)

    def test_roundtrip_precision_shock(self, freundlich_sorption):
        """Shock-shock intersection has machine precision."""
        if freundlich_sorption.n < 1.0:
            shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=freundlich_sorption)
            shock2 = ShockWave(theta_start=0.0, v_start=300.0, c_left=8.0, c_right=1.0, sorption=freundlich_sorption)

            result = find_shock_shock_intersection(shock1, shock2, theta_current=0.0)
            assert result is None
        elif freundlich_sorption.n > 1.0:
            shock1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=freundlich_sorption)
            shock2 = ShockWave(theta_start=0.0, v_start=300.0, c_left=8.0, c_right=1.0, sorption=freundlich_sorption)

            result = find_shock_shock_intersection(shock1, shock2, theta_current=0.0)

            assert result is not None, "Expected shock intersection for machine precision test"

            theta_int, v_int = result

            assert shock1.speed is not None
            assert shock2.speed is not None
            v1 = shock1.v_start + shock1.speed * (theta_int - shock1.theta_start)
            v2 = shock2.v_start + shock2.speed * (theta_int - shock2.theta_start)

            assert np.isclose(v1, v2, rtol=1e-14, atol=1e-15)
            assert np.isclose(v1, v_int, rtol=1e-14, atol=1e-15)


def _ref_char_char(c1, c2, tc) -> tuple[float, float] | None:
    """Pre-refactor ``find_characteristic_intersection`` body (independent reference)."""
    s1 = characteristic_speed(c1.concentration, c1.sorption)
    s2 = characteristic_speed(c2.concentration, c2.sorption)
    if abs(s1 - s2) < 1e-15:
        return None
    tb = max(c1.theta_start, c2.theta_start, tc)
    v1 = characteristic_position(c1.concentration, c1.sorption, c1.theta_start, c1.v_start, tb)
    v2 = characteristic_position(c2.concentration, c2.sorption, c2.theta_start, c2.v_start, tb)
    if v1 is None or v2 is None:
        return None
    dt = (v2 - v1) / (s1 - s2)
    return None if dt <= 0 else (float(tb + dt), float(v1 + s1 * dt))


def _ref_shock_shock(a, b, tc) -> tuple[float, float] | None:
    """Pre-refactor ``find_shock_shock_intersection`` body."""
    if abs(a.speed - b.speed) < 1e-15:
        return None
    tb = max(a.theta_start, b.theta_start, tc)
    v1 = a.v_start + a.speed * (tb - a.theta_start)
    v2 = b.v_start + b.speed * (tb - b.theta_start)
    dt = (v2 - v1) / (a.speed - b.speed)
    return None if dt <= 0 else (float(tb + dt), float(v1 + a.speed * dt))


def _ref_shock_char(sh, ch, tc) -> tuple[float, float] | None:
    """Pre-refactor ``find_shock_characteristic_intersection`` body (a=shock)."""
    sc = characteristic_speed(ch.concentration, ch.sorption)
    if abs(sh.speed - sc) < 1e-15:
        return None
    tb = max(sh.theta_start, ch.theta_start, tc)
    vs = sh.v_start + sh.speed * (tb - sh.theta_start)
    vc = characteristic_position(ch.concentration, ch.sorption, ch.theta_start, ch.v_start, tb)
    if vc is None:
        return None
    dt = (vc - vs) / (sh.speed - sc)
    return None if dt <= 0 else (float(tb + dt), float(vs + sh.speed * dt))


def _ref_raref_bounds(raref, other, tc) -> list[tuple[float, float, str]]:
    """Pre-refactor ``find_rarefaction_boundary_intersections`` body (temp CharacteristicWaves)."""
    out: list[tuple[float, float, str]] = []
    head = CharacteristicWave(
        theta_start=raref.theta_start, v_start=raref.v_start, concentration=raref.c_head, sorption=raref.sorption
    )
    tail = CharacteristicWave(
        theta_start=raref.theta_start, v_start=raref.v_start, concentration=raref.c_tail, sorption=raref.sorption
    )
    if isinstance(other, CharacteristicWave):
        for bc, tag in ((head, "head"), (tail, "tail")):
            r = _ref_char_char(bc, other, tc)
            if r:
                out.append((r[0], r[1], tag))
    elif isinstance(other, ShockWave):
        for bc, tag in ((head, "head"), (tail, "tail")):
            r = _ref_shock_char(other, bc, tc)
            if r:
                out.append((r[0], r[1], tag))
    elif isinstance(other, RarefactionWave):
        o_head = CharacteristicWave(
            theta_start=other.theta_start, v_start=other.v_start, concentration=other.c_head, sorption=other.sorption
        )
        o_tail = CharacteristicWave(
            theta_start=other.theta_start, v_start=other.v_start, concentration=other.c_tail, sorption=other.sorption
        )
        for bc, tag in ((head, "head"), (tail, "tail")):
            for ob in (o_head, o_tail):
                r = _ref_char_char(bc, ob, tc)
                if r:
                    out.append((r[0], r[1], tag))
    return out


class TestLineIntersectionKernelEquivalence:
    """The ``_line_intersection`` refactor is bit-identical to the pre-refactor helper bodies.

    Each closed-form intersection helper now delegates to one shared kernel using the waves'
    CACHED speeds. The refactor is only equivalence-preserving if the per-branch operand order
    is exact: ``dθ`` is invariant under an a↔b swap but ``V_intersect`` (the successor wave's
    ``v_start``) is NOT — the raref-vs-shock branch in particular must keep ``a = shock``. This
    checks BOTH returned components bit-for-bit against an independent transcription of the
    original bodies over a randomised battery of all four pair kinds and both isotherm regimes.
    """

    @staticmethod
    def _both_bit_equal(got, want):
        if (got is None) != (want is None):
            return False
        if got is None:
            return True
        return got[0].hex() == want[0].hex() and got[1].hex() == want[1].hex()

    def _make_raref(self, rng, sorption):
        for _ in range(20):
            a, b = float(rng.uniform(0.5, 20.0)), float(rng.uniform(0.5, 20.0))
            for c_head, c_tail in ((max(a, b), min(a, b)), (min(a, b), max(a, b))):
                try:
                    return RarefactionWave(
                        theta_start=float(rng.uniform(0, 50)),
                        v_start=float(rng.uniform(0, 50)),
                        c_head=c_head,
                        c_tail=c_tail,
                        sorption=sorption,
                    )
                except ValueError:
                    continue
        return None

    def test_parallel_speeds_return_none_in_kernel_and_reference(self):
        """The kernel's ``|s_a - s_b| < EPSILON_SPEED`` early-return agrees with the reference.

        Randomised concentrations never collide exactly, so the degenerate branch needs an
        explicit equal-speed configuration to be exercised inside the equivalence battery.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=7.0, sorption=sorption)
        c2 = CharacteristicWave(theta_start=3.0, v_start=11.0, concentration=7.0, sorption=sorption)
        assert find_characteristic_intersection(c1, c2, 0.0) is None
        assert _ref_char_char(c1, c2, 0.0) is None

        s1 = ShockWave(theta_start=0.0, v_start=0.0, c_left=8.0, c_right=2.0, sorption=sorption)
        s2 = ShockWave(theta_start=1.0, v_start=9.0, c_left=8.0, c_right=2.0, sorption=sorption)
        assert find_shock_shock_intersection(s1, s2, 0.0) is None
        assert _ref_shock_shock(s1, s2, 0.0) is None

    @pytest.mark.parametrize("n", [2.0, 0.6])
    def test_all_four_helpers_bit_identical_to_reference(self, n):
        rng = np.random.default_rng(20260715)
        sorption = FreundlichSorption(k_f=0.01, n=n, bulk_density=1500.0, porosity=0.3)

        def mk_char():
            return CharacteristicWave(
                theta_start=float(rng.uniform(0, 50)),
                v_start=float(rng.uniform(0, 50)),
                concentration=float(rng.uniform(0.5, 20.0)),
                sorption=sorption,
            )

        def mk_shock():
            return ShockWave(
                theta_start=float(rng.uniform(0, 50)),
                v_start=float(rng.uniform(0, 50)),
                c_left=float(rng.uniform(0.5, 20.0)),
                c_right=float(rng.uniform(0.5, 20.0)),
                sorption=sorption,
            )

        for _ in range(3000):
            tc = float(rng.uniform(0, 50))
            c1, c2 = mk_char(), mk_char()
            assert self._both_bit_equal(find_characteristic_intersection(c1, c2, tc), _ref_char_char(c1, c2, tc))
            s1, s2 = mk_shock(), mk_shock()
            assert self._both_bit_equal(find_shock_shock_intersection(s1, s2, tc), _ref_shock_shock(s1, s2, tc))
            sh, ch = mk_shock(), mk_char()
            assert self._both_bit_equal(find_shock_characteristic_intersection(sh, ch, tc), _ref_shock_char(sh, ch, tc))
            raref = self._make_raref(rng, sorption)
            if raref is None:
                continue
            for other in (mk_char(), mk_shock(), self._make_raref(rng, sorption)):
                if other is None:
                    continue
                got = find_rarefaction_boundary_intersections(raref, other, tc)
                want = _ref_raref_bounds(raref, other, tc)
                assert len(got) == len(want)
                for g, w in zip(got, want, strict=True):
                    # float() is identity here; it narrows the float-annotated element off the
                    # int|float numeric-tower union so .hex() resolves.
                    assert float(g[0]).hex() == float(w[0]).hex()  # θ bit-for-bit
                    assert float(g[1]).hex() == float(w[1]).hex()  # V bit-for-bit
                    assert g[2] == w[2]  # boundary tag
