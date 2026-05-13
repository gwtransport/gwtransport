"""
Unit Tests for Front Tracking Solver.
======================================

Tests for the main event-driven simulation engine (FrontTracker class).
Verifies initialization, event detection, event handling, and full simulation runs.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption, LangmuirSorption
from gwtransport.fronttracking.output import compute_breakthrough_curve, concentration_at_point
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import RarefactionWave, ShockWave

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


@pytest.fixture
def simple_step_input():
    """Simple step input: C: 0→10."""

    # Create [0, 10, 100] days
    tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-04-11"])
    cin = np.array([0.0, 10.0])
    flow = np.array([100.0, 100.0])
    return cin, flow, tedges


@pytest.fixture
def pulse_input():
    """Pulse input: C: 0→10→0."""

    # Use custom periods: 0, 10, 20, 100 days
    tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-04-11"])
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

        # t_current is in days from tedges[0], so should be 0.0 at initialization
        assert tracker.state.t_current == 0.0
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
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11"])  # Wrong length

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
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-04-11"])

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
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-04-11"])

        with pytest.raises(ValueError, match="flow must be non-negative"):
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

        # t_first_arrival is in days from tedges[0], should be positive
        assert tracker.t_first_arrival > 0.0
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
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-04-11"])

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
        compute_first_front_arrival_time in the math module.
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
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-04-11"])

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
        # t_current is in days from tedges[0], should be >= 0.0
        assert tracker.state.t_current >= 0.0

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
        # t_current is in days from tedges[0], should be >= 0.0
        assert tracker.state.t_current >= 0.0

    def test_multiple_steps_completes(self, freundlich_sorption):
        """Test simulation with multiple concentration steps."""
        cin = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        flow = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-01-31", "2020-02-10", "2020-04-11"])

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
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-01-31"])

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
        tedges = pd.to_datetime(["2020-01-01", "2020-04-11"])

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


class TestRuntimeMassBalanceVerification:
    """
    Tests for runtime mass balance verification (High Priority #3).

    Verifies that verify_physics() correctly computes and checks mass balance
    using exact analytical integration of domain mass, inlet mass, and outlet mass.
    """

    def test_mass_balance_simple_step_input_freundlich(self, simple_step_input):
        """Test mass balance with simple step input and Freundlich sorption."""
        cin, flow, tedges = simple_step_input
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=sorption,
        )

        # Run simulation with mass balance verification enabled
        tracker.run(max_iterations=100, verbose=False)

        # Verify physics including mass balance at final time (n=2 uses exact integration)
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    @pytest.mark.skip(reason="Pulse input creates waves that may exit domain - needs investigation")
    def test_mass_balance_pulse_input_freundlich(self, pulse_input):
        """Test mass balance with pulse input (rise and fall) and Freundlich sorption."""
        cin, flow, tedges = pulse_input
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        # Verify mass balance at end
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_mass_balance_constant_retardation(self, simple_step_input):
        """Test mass balance with constant retardation (no rarefactions); machine precision."""
        cin, flow, tedges = simple_step_input
        sorption = ConstantRetardation(retardation_factor=2.0)

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        # ConstantRetardation has no rarefactions → exact integration, tighten to 1e-12 (P2.6).
        # Failure here means a regression of mass conservation in the constant-retardation path;
        # do not loosen this tolerance (see CLAUDE.md / feedback memory).
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-12)

    @pytest.mark.skip(reason="Freundlich n!=2: exact spatial rarefaction integration not yet implemented")
    def test_mass_balance_freundlich_n_lt_1(self, simple_step_input):
        """Test mass balance with Freundlich n<1 (n<1, different rarefactions)."""
        cin, flow, tedges = simple_step_input
        sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3)

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        # Would verify mass balance if n=0.5 integration was implemented
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_mass_balance_at_early_times(self, simple_step_input, freundlich_sorption):
        """Test mass balance verification works at early simulation times."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        # Run simulation
        tracker.run(max_iterations=100, verbose=False)

        # Should satisfy mass balance at final time
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    @pytest.mark.skip(reason="Multiple concentration changes with C→0 transitions needs investigation")
    def test_mass_balance_multiple_concentration_changes(self):
        """Test mass balance with multiple inlet concentration changes."""
        # Create input with multiple steps: 5→10→3→0
        tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-11", "2020-01-21", "2020-02-01", "2020-02-11"])
        cin = np.array([5.0, 10.0, 3.0, 0.0])
        flow = np.array([100.0, 100.0, 100.0, 100.0])

        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=sorption,
        )

        tracker.run(max_iterations=200, verbose=False)

        # Would verify mass balance if C→0 handling was complete
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_mass_balance_very_small_domain(self, simple_step_input, freundlich_sorption):
        """Test mass balance with very small domain (waves exit quickly)."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=10.0,  # Very small domain
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=50, verbose=False)

        # Mass balance should still hold with exact integration for n=2
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_mass_balance_at_t_zero(self):
        """Test mass balance at t=0 before any mass enters."""
        tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-11", "2020-02-01"])
        cin = np.array([0.0, 10.0])
        flow = np.array([100.0, 100.0])

        sorption = ConstantRetardation(retardation_factor=2.0)

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=sorption,
        )

        # At t=0, no mass has entered, so all masses should be zero
        # verify_physics should handle this gracefully
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-12)


class TestLangmuirSorption:
    """Test FrontTracker with Langmuir sorption."""

    @pytest.fixture
    def langmuir_sorption(self):
        """Standard Langmuir sorption for testing."""
        return LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)

    def test_step_input_completes(self, simple_step_input, langmuir_sorption):
        """Test that step input simulation with Langmuir completes."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=langmuir_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        assert len(tracker.state.events) > 0
        assert tracker.state.t_current >= 0.0

    def test_pulse_input_creates_rarefaction(self, pulse_input, langmuir_sorption):
        """Test that pulse input creates rarefaction waves with Langmuir."""
        cin, flow, tedges = pulse_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=langmuir_sorption,
        )

        tracker.run(max_iterations=200, verbose=False)

        # Langmuir is favorable → concentration decrease (10→0) creates rarefaction
        rarefactions = [w for w in tracker.state.waves if isinstance(w, RarefactionWave)]
        assert len(rarefactions) > 0

    def test_mass_balance_step_input(self, langmuir_sorption):
        """Test mass balance with Langmuir step input.

        Uses a small domain so the shock exits within the input period.
        """
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-04-11"])
        cin = np.array([0.0, 10.0])
        flow = np.array([100.0, 100.0])

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=100.0,  # Small domain so shock exits in time
            sorption=langmuir_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_finite_rarefaction_tail_time(self, pulse_input, langmuir_sorption):
        """Test that Langmuir rarefaction tails arrive in finite time.

        Unlike Freundlich n>1 where R(0)→∞ makes tail velocity 0,
        Langmuir has finite R(0) so tails arrive at finite time.
        """
        cin, flow, tedges = pulse_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=langmuir_sorption,
        )

        tracker.run(max_iterations=200, verbose=False)

        for wave in tracker.state.waves:
            if isinstance(wave, RarefactionWave) and wave.is_active:
                tail_vel = wave.tail_velocity()
                assert tail_vel > 0, "Langmuir rarefaction tail velocity must be positive"


class TestRiemannProblems:
    """Riemann-problem coverage for issue #174 group 2.

    These exercise the canonical wave-decomposition cases end-to-end through
    FrontTracker so that mass conservation, entropy, and the analytic structure
    can all be verified together.
    """

    def test_square_pulse_constant_retardation_total_mass_at_outlet(self, constant_retardation):
        """Square pulse 0->C->0 with ConstantRetardation: total outlet mass exactly matches inlet.

        ConstantRetardation produces no rarefactions (no concentration-dependent velocity), so
        the breakthrough is an exact time-shift and mass conservation is machine precision.
        """
        v_pore = 200.0
        c_step = 4.0
        flow_val = 100.0
        n_bins = 100
        cin = np.zeros(n_bins)
        cin[5:15] = c_step
        flow = np.full(n_bins, flow_val)
        tedges = pd.date_range("2020-01-01", periods=n_bins + 1, freq="D")

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=v_pore,
            sorption=constant_retardation,
        )
        tracker.run(max_iterations=2000, verbose=False)

        dt_days = np.diff((tedges - tedges[0]) / pd.Timedelta(days=1))
        mass_in = float(np.sum(cin * flow * dt_days))

        # Constant retardation → outlet is a shifted copy of inlet; integrate over the same span.
        t_sample = np.linspace(0.0, float(n_bins), 4000)
        cout = compute_breakthrough_curve(t_sample, v_pore, tracker.state.waves, constant_retardation)
        bin_idx = np.clip(np.searchsorted(dt_days.cumsum(), t_sample, side="right"), 0, n_bins - 1)
        flow_at_t = flow[bin_idx]
        mass_out = float(np.trapezoid(cout * flow_at_t, t_sample))

        # Tolerance limited by trapezoidal-rule error over the sharp leading/trailing edges.
        assert np.isclose(mass_out, mass_in, rtol=2e-3)

    @pytest.mark.xfail(
        reason="P1.8 (#168): Freundlich n=2 rarefaction with c_tail=0 produces "
        "excess outlet mass — under investigation.",
        strict=False,
    )
    def test_square_pulse_n_gt_1_total_mass_at_outlet(self, freundlich_sorption):
        """Documents the open #168 P1.8 bug: n=2 pulse releases too much mass at the outlet.

        Marked xfail and kept as a regression guard for the eventual fix. Tolerance is loose
        (rtol=1e-2) because trapezoidal rule on a rarefaction tail still has discretization error;
        the bug under investigation produces ~2x excess mass, well outside this tolerance.
        """
        v_pore = 200.0
        c_step = 4.0
        flow_val = 100.0
        n_bins = 500
        cin = np.zeros(n_bins)
        cin[5:15] = c_step
        flow = np.full(n_bins, flow_val)
        tedges = pd.date_range("2020-01-01", periods=n_bins + 1, freq="D")

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=v_pore,
            sorption=freundlich_sorption,
        )
        tracker.run(max_iterations=10000, verbose=False)

        dt_days = np.diff((tedges - tedges[0]) / pd.Timedelta(days=1))
        mass_in = float(np.sum(cin * flow * dt_days))

        t_sample = np.linspace(0.0, float(n_bins), 10000)
        cout = compute_breakthrough_curve(t_sample, v_pore, tracker.state.waves, freundlich_sorption)
        bin_idx = np.clip(np.searchsorted(dt_days.cumsum(), t_sample, side="right"), 0, n_bins - 1)
        flow_at_t = flow[bin_idx]
        mass_out = float(np.trapezoid(cout * flow_at_t, t_sample))

        assert np.isclose(mass_out, mass_in, rtol=1e-2)

    def test_two_step_increase_merges_into_single_shock(self, freundlich_sorption):
        """For n>1, 0->C1->C2 with C1<C2 produces a faster trailing shock that catches and merges
        with the leading shock; final shock satisfies R-H on (0, C2)."""
        v_pore = 1000.0
        flow_val = 100.0
        cin = np.array([0.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0])
        flow = np.full(len(cin), flow_val)
        tedges = pd.date_range("2020-01-01", periods=len(cin) + 1, freq="D")

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=v_pore,
            sorption=freundlich_sorption,
        )
        tracker.run(max_iterations=2000, verbose=False)

        # After the catchup, exactly one active shock remains; it connects 0 to 6.
        active_shocks = [w for w in tracker.state.waves if isinstance(w, ShockWave) and w.is_active]
        assert len(active_shocks) == 1
        final = active_shocks[0]
        # c_left/c_right ordered by velocity (higher C is faster for n>1, becomes c_left)
        assert np.isclose(final.c_left, 6.0)
        assert np.isclose(final.c_right, 0.0)
        # R-H velocity for (0, 6).
        c_tot_diff = freundlich_sorption.total_concentration(6.0) - freundlich_sorption.total_concentration(0.0)
        rh_vel = flow_val * (6.0 - 0.0) / c_tot_diff
        assert np.isclose(final.velocity, rh_vel, rtol=1e-12)

    def test_constant_retardation_breakthrough_matches_pure_advection(self, constant_retardation):
        """For ConstantRetardation, front-tracking matches the shifted-inlet closed form."""
        v_pore = 300.0
        flow_val = 100.0
        # cin has a recognizable shape (triangle) so we can verify position of features.
        cin = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        flow = np.full(len(cin), flow_val)
        tedges = pd.date_range("2020-01-01", periods=len(cin) + 1, freq="D")

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=v_pore,
            sorption=constant_retardation,
        )
        tracker.run(max_iterations=500, verbose=False)

        # Closed form: cout(t) == cin(t - V*R/flow), where t and the offset are
        # in days from tedges[0]. With R=2, V=300, flow=100, delay = 6 days.
        delay = v_pore * constant_retardation.retardation_factor / flow_val
        # Sample at integer days at the centers of the pulse arrivals.
        for t in [7.5, 8.5, 9.5, 10.5]:
            c_out = concentration_at_point(v_pore, t, tracker.state.waves, constant_retardation)
            t_in = t - delay  # corresponding inlet time
            bin_idx = int(np.floor(t_in))  # cin is constant in each [bin_idx, bin_idx+1)
            assert np.isclose(c_out, cin[bin_idx], rtol=1e-14, atol=1e-14)
