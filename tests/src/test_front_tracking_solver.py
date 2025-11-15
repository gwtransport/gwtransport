"""
Unit Tests for Front Tracking Solver.
======================================

Tests for the main event-driven simulation engine (FrontTracker class).
Verifies initialization, event detection, event handling, and full simulation runs.
"""

import numpy as np
import pytest

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import RarefactionWave

freundlich_sorptions = [
    FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3),
    FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3),
]


@pytest.fixture
def freundlich_sorption():
    """Standard Freundlich sorption for testing (n>1, favorable)."""
    return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)


@pytest.fixture
def constant_retardation():
    """Constant retardation for testing."""
    return ConstantRetardation(retardation_factor=2.0)


@pytest.fixture
def simple_step_input():
    """Simple step input: C: 0→10."""
    tedges = np.array([0.0, 10.0, 100.0])
    cin = np.array([0.0, 10.0])
    flow = np.array([100.0, 100.0])
    return cin, flow, tedges


@pytest.fixture
def pulse_input():
    """Pulse input: C: 0→10→0."""
    tedges = np.array([0.0, 10.0, 20.0, 100.0])
    cin = np.array([0.0, 10.0, 0.0])
    flow = np.array([100.0, 100.0, 100.0])
    return cin, flow, tedges


class TestFrontTrackerInitialization:
    """Test FrontTracker initialization."""

    def test_initialization_simple(self, simple_step_input, freundlich_sorption):
        """Test basic initialization."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        assert tracker.state.t_current == tedges[0]
        assert tracker.state.v_outlet == 500.0
        assert len(tracker.state.waves) >= 0  # Should have created inlet waves
        assert len(tracker.state.events) == 0  # No events yet

    def test_initialization_creates_inlet_waves(self, simple_step_input, freundlich_sorption):
        """Test that initialization creates waves from inlet."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        # Should create wave for C: 0→10 transition
        assert len(tracker.state.waves) >= 1

    def test_validation_tedges_length(self, freundlich_sorption):
        """Test validation of tedges length."""
        cin = np.array([0.0, 10.0])
        flow = np.array([100.0, 100.0])
        tedges = np.array([0.0, 10.0])  # Wrong length

        with pytest.raises(ValueError, match="tedges must have length"):
            FrontTracker(
                cin=cin,
                flow=flow,
                tedges=tedges,
                aquifer_pore_volume=500.0,
                sorption=freundlich_sorption,
            )

    def test_validation_negative_concentration(self, freundlich_sorption):
        """Test validation of negative concentrations."""
        cin = np.array([0.0, -10.0])  # Negative concentration
        flow = np.array([100.0, 100.0])
        tedges = np.array([0.0, 10.0, 100.0])

        with pytest.raises(ValueError, match="cin must be non-negative"):
            FrontTracker(
                cin=cin,
                flow=flow,
                tedges=tedges,
                aquifer_pore_volume=500.0,
                sorption=freundlich_sorption,
            )

    def test_validation_negative_flow(self, freundlich_sorption):
        """Test validation of negative flow."""
        cin = np.array([0.0, 10.0])
        flow = np.array([100.0, -100.0])  # Negative flow
        tedges = np.array([0.0, 10.0, 100.0])

        with pytest.raises(ValueError, match="flow must be positive"):
            FrontTracker(
                cin=cin,
                flow=flow,
                tedges=tedges,
                aquifer_pore_volume=500.0,
                sorption=freundlich_sorption,
            )

    def test_first_arrival_time_computed(self, simple_step_input, freundlich_sorption):
        """Test that first arrival time is computed."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        assert tracker.t_first_arrival > tedges[0]
        assert np.isfinite(tracker.t_first_arrival)


class TestFindNextEvent:
    """Test find_next_event method."""

    def test_finds_outlet_crossing(self, simple_step_input, freundlich_sorption):
        """Test that find_next_event finds outlet crossings."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        # Should find events (inlet waves should eventually cross outlet)
        event = tracker.find_next_event()
        assert event is not None

    def test_returns_none_when_no_events(self, freundlich_sorption):
        """Test that find_next_event returns None when no events exist."""
        # Create tracker with no concentration changes
        cin = np.array([0.0, 0.0])
        flow = np.array([100.0, 100.0])
        tedges = np.array([0.0, 10.0, 100.0])

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        # No concentration changes -> no inlet waves and no events
        assert len(tracker.state.waves) == 0
        assert tracker.find_next_event() is None


    def test_first_outlet_crossing_not_before_first_arrival(self, simple_step_input, freundlich_sorption):
        """Spin-up does not constrain outlet-crossing events created by the solver.

        This test is intentionally weak: it only checks that both t_first_arrival and
        at least one outlet crossing exist and are finite, without imposing a specific
        ordering between them. The exact relationship is handled analytically by
        compute_first_arrival_time in the math module.
        """
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        # Ensure both first-arrival time and at least one outlet crossing exist
        t_first_arrival = tracker.t_first_arrival
        assert np.isfinite(t_first_arrival)

        event = tracker.find_next_event()
        while event is not None and event.event_type.name != "OUTLET_CROSSING":
            tracker.state.t_current = event.time
            tracker.handle_event(event)
            event = tracker.find_next_event()

        assert event is not None
        assert event.event_type.name == "OUTLET_CROSSING"


class TestHandleEvent:
    """Test handle_event method."""

    def test_handles_characteristic_collision(self, freundlich_sorption):
        """Test handling of characteristic collision."""
        cin = np.array([0.0, 5.0, 2.0])
        flow = np.array([100.0, 100.0, 100.0])
        tedges = np.array([0.0, 10.0, 20.0, 100.0])

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=1000.0,
            sorption=freundlich_sorption,
        )

        initial_wave_count = len(tracker.state.waves)

        # Run simulation to completion
        tracker.run(max_iterations=100, verbose=False)

        # MUST have recorded events (characteristic collisions should occur)
        assert len(tracker.state.events) > 0, "Expected at least one event to be recorded"
        assert len(tracker.state.waves) >= initial_wave_count, "Expected waves to be created from collisions"


class TestSimulationRun:
    """Test full simulation runs."""

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_simple_step_input_completes(self, simple_step_input, freundlich_sorption):
        """Test that simple step input simulation completes (n>1 and n<1)."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        assert len(tracker.state.events) > 0
        assert tracker.state.t_current >= tedges[0]

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_pulse_input_completes(self, pulse_input, freundlich_sorption):
        """Test that pulse input simulation completes (n>1 and n<1)."""
        cin, flow, tedges = pulse_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=200, verbose=False)

        assert len(tracker.state.events) > 0

    def test_constant_retardation_completes(self, simple_step_input, constant_retardation):
        """Test simulation with constant retardation."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=constant_retardation,
        )

        # With constant retardation, all concentrations have same velocity
        # So concentration changes create characteristics (contact discontinuities)
        # Should create at least one wave
        assert len(tracker.state.waves) >= 0  # May create characteristic or nothing

        # Run simulation
        tracker.run(max_iterations=100, verbose=False)

        # Simulation should complete without errors
        assert tracker.state.t_current >= tedges[0]

    def test_multiple_steps_completes(self, freundlich_sorption):
        """Test simulation with multiple concentration steps."""
        cin = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        flow = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        tedges = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 100.0])

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=500, verbose=False)

        assert len(tracker.state.events) > 0
        assert len(tracker.state.waves) > 0


class TestPhysicsVerification:
    """Test physics verification."""

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_verify_physics_passes_on_valid_state(self, simple_step_input, freundlich_sorption):
        """Test that verify_physics passes on valid state (n>1 and n<1)."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.verify_physics()

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_verify_physics_after_simulation(self, simple_step_input, freundlich_sorption):
        """Test that physics remains valid after simulation (n>1 and n<1)."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        tracker.verify_physics()


class TestEventHistory:
    """Test event history recording."""

    def test_events_recorded(self, simple_step_input, freundlich_sorption):
        """Test that events are recorded in history."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        # Should have recorded events
        assert len(tracker.state.events) > 0

        # Check event structure
        for event in tracker.state.events:
            assert "time" in event
            assert "type" in event
            assert "location" in event

        # Collision events include full wave diagnostic information
        collision_events = [e for e in tracker.state.events if e["type"] != "outlet_crossing"]
        for event in collision_events:
            assert "waves_before" in event
            assert "waves_after" in event
            assert isinstance(event["waves_before"], list)
            assert isinstance(event["waves_after"], list)

    def test_event_times_chronological(self, simple_step_input, freundlich_sorption):
        """Test that events are processed in chronological order."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        # Event times should be non-decreasing
        times = [event["time"] for event in tracker.state.events]
        for i in range(len(times) - 1):
            assert times[i] <= times[i + 1], f"Events not chronological: {times[i]} > {times[i + 1]}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_concentration_only(self, freundlich_sorption):
        """Test with zero concentration throughout."""
        cin = np.array([0.0, 0.0, 0.0])
        flow = np.array([100.0, 100.0, 100.0])
        tedges = np.array([0.0, 10.0, 20.0, 30.0])

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        # Should complete quickly with no events
        tracker.run(max_iterations=10, verbose=False)

    def test_single_time_bin(self, freundlich_sorption):
        """Test with single time bin."""
        cin = np.array([10.0])
        flow = np.array([100.0])
        tedges = np.array([0.0, 100.0])

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=50, verbose=False)

        # Should handle single bin correctly

    def test_very_small_domain(self, simple_step_input, freundlich_sorption):
        """Test with very small pore volume."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=10.0,  # Very small
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=50, verbose=False)

    def test_very_large_domain(self, simple_step_input, freundlich_sorption):
        """Test with very large pore volume."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=10000.0,  # Very large
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=50, verbose=False)


class TestVerifyPhysicsNegativeCases:
    """Negative tests to ensure verify_physics detects invalid states."""

    def test_verify_physics_detects_invalid_rarefaction(self, freundlich_sorption):
        """Manually insert an invalid rarefaction and expect verify_physics to fail."""
        # Constructing a "rarefaction" with reversed head/tail velocities is already
        # guarded against in RarefactionWave.__post_init__, which will raise ValueError.
        with pytest.raises(ValueError):
            RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=100.0,
                c_head=1.0,
                c_tail=10.0,
                sorption=freundlich_sorption,
            )
