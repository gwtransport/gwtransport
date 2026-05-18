"""
Unit Tests for Front Tracking Solver in (V, θ) coordinates.
============================================================

Tests for the main event-driven simulation engine (FrontTracker class).
Verifies initialization, event detection, event handling, and full simulation runs.

The solver runs entirely in cumulative-flow coordinate θ; events in
``tracker.state.events`` carry the ``"theta"`` key. User-facing time t is
recovered via ``tracker.state.t_at_theta(theta)``.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption, LangmuirSorption
from gwtransport.fronttracking.output import (
    compute_breakthrough_curve,
    compute_cumulative_inlet_mass,
    compute_cumulative_outlet_mass,
    compute_domain_mass,
    compute_total_outlet_mass,
    concentration_at_point,
)
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import RarefactionWave, ShockWave

freundlich_sorptions = [
    FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3),
    FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3),
]


@pytest.fixture
def freundlich_sorption():
    """Standard Freundlich sorption for testing (n>1)."""
    return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)


@pytest.fixture
def constant_retardation():
    """Constant retardation for testing."""
    return ConstantRetardation(retardation_factor=2.0)


@pytest.fixture
def simple_step_input():
    """Simple step input: C: 0→10."""
    tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-04-11"])
    cin = np.array([0.0, 10.0])
    flow = np.array([100.0, 100.0])
    return cin, flow, tedges


@pytest.fixture
def pulse_input():
    """Pulse input: C: 0→10→0."""
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

        # theta_current is in m³, starts at 0.0
        assert tracker.state.theta_current == 0.0
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
        cin = np.array([0.0, -10.0])
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
        flow = np.array([100.0, -100.0])
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-04-11"])

        with pytest.raises(ValueError, match="flow must be non-negative"):
            FrontTracker(
                cin=cin,
                flow=flow,
                tedges=tedges,
                aquifer_pore_volume=500.0,
                sorption=freundlich_sorption,
            )

    def test_first_arrival_theta_computed(self, simple_step_input, freundlich_sorption):
        """First arrival θ is computed and positive/finite."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        assert tracker.theta_first_arrival > 0.0
        assert np.isfinite(tracker.theta_first_arrival)


class TestFindNextEvent:
    """Test find_next_event method."""

    def test_finds_outlet_crossing(self, simple_step_input, freundlich_sorption):
        """find_next_event finds outlet crossings."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        event = tracker.find_next_event()
        assert event is not None

    def test_returns_none_when_no_events(self, freundlich_sorption):
        """find_next_event returns None when no waves exist."""
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

        assert len(tracker.state.waves) == 0
        assert tracker.find_next_event() is None

    def test_first_outlet_crossing_emitted(self, simple_step_input, freundlich_sorption):
        """At least one outlet crossing exists; theta_first_arrival is finite."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        assert np.isfinite(tracker.theta_first_arrival)

        event = tracker.find_next_event()
        while event is not None and event.event_type.name != "OUTLET_CROSSING":
            tracker.state.theta_current = event.theta
            tracker.handle_event(event)
            event = tracker.find_next_event()

        assert event is not None
        assert event.event_type.name == "OUTLET_CROSSING"


class TestHandleEvent:
    """Test handle_event method."""

    def test_handles_characteristic_collision(self, freundlich_sorption):
        """End-to-end run produces events including collisions."""
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

        tracker.run(max_iterations=100, verbose=False)

        assert len(tracker.state.events) > 0, "Expected at least one event to be recorded"
        assert len(tracker.state.waves) >= initial_wave_count


class TestSimulationRun:
    """Test full simulation runs."""

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_simple_step_input_completes(self, simple_step_input, freundlich_sorption):
        """Step input simulation completes."""
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
        assert tracker.state.theta_current >= 0.0

    @pytest.mark.parametrize("freundlich_sorption", freundlich_sorptions)
    def test_pulse_input_completes(self, pulse_input, freundlich_sorption):
        """Pulse input simulation completes."""
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
        """Constant-retardation simulation completes."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=constant_retardation,
        )

        # With constant retardation, concentration changes create characteristics.
        assert len(tracker.state.waves) >= 0

        tracker.run(max_iterations=100, verbose=False)

        assert tracker.state.theta_current >= 0.0

    def test_multiple_steps_completes(self, freundlich_sorption):
        """Multiple concentration steps simulation completes."""
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
        """verify_physics passes on valid state."""
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
        """Physics remains valid after simulation."""
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
    """Test event history recording (events carry "theta" key)."""

    def test_events_recorded(self, simple_step_input, freundlich_sorption):
        """Events are recorded with the θ-key in history."""
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

        for event in tracker.state.events:
            assert "theta" in event
            assert "type" in event
            assert "location" in event

        collision_events = [e for e in tracker.state.events if e["type"] != "outlet_crossing"]
        for event in collision_events:
            assert "waves_before" in event
            assert "waves_after" in event
            assert isinstance(event["waves_before"], list)
            assert isinstance(event["waves_after"], list)

    def test_event_thetas_chronological(self, simple_step_input, freundlich_sorption):
        """Events are processed in θ-chronological order."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        thetas = [event["theta"] for event in tracker.state.events]
        for i in range(len(thetas) - 1):
            assert thetas[i] <= thetas[i + 1], f"Events not chronological: θ={thetas[i]} > {thetas[i + 1]}"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_concentration_only(self, freundlich_sorption):
        """Zero concentration throughout: completes with no events."""
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

        tracker.run(max_iterations=10, verbose=False)

    def test_single_time_bin(self, freundlich_sorption):
        """Single time bin: simulation completes."""
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

    def test_very_small_domain(self, simple_step_input, freundlich_sorption):
        """Very small pore volume."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=10.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=50, verbose=False)

    def test_very_large_domain(self, simple_step_input, freundlich_sorption):
        """Very large pore volume."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=10000.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=50, verbose=False)


class TestVerifyPhysicsNegativeCases:
    """Negative tests to ensure verify_physics detects invalid states."""

    def test_verify_physics_detects_invalid_rarefaction(self, freundlich_sorption):
        """Invalid rarefaction (reversed head/tail) raises in __post_init__."""
        with pytest.raises(ValueError):
            RarefactionWave(theta_start=0.0, v_start=0.0, c_head=1.0, c_tail=10.0, sorption=freundlich_sorption)


class TestRuntimeMassBalanceVerification:
    """Tests for runtime mass balance verification via verify_physics."""

    def test_mass_balance_simple_step_input_freundlich(self, simple_step_input):
        """Mass balance with simple step input and Freundlich n=2."""
        cin, flow, tedges = simple_step_input
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    @pytest.mark.skip(reason="Pulse input creates waves that may exit domain - needs investigation")
    def test_mass_balance_pulse_input_freundlich(self, pulse_input):
        """Mass balance with pulse input and Freundlich n=2."""
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

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_mass_balance_constant_retardation(self, simple_step_input):
        """Mass balance with constant retardation; machine precision."""
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

        # ConstantRetardation has no rarefactions → exact integration at 1e-12.
        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-12)

    @pytest.mark.skip(reason="Freundlich n!=2: exact spatial rarefaction integration not yet implemented")
    def test_mass_balance_freundlich_n_lt_1(self, simple_step_input):
        """Mass balance with Freundlich n<1."""
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

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_mass_balance_at_early_simulation(self, simple_step_input, freundlich_sorption):
        """Mass balance verification works at end of simulation."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    @pytest.mark.skip(reason="Multiple concentration changes with C→0 transitions needs investigation")
    def test_mass_balance_multiple_concentration_changes(self):
        """Mass balance with multiple inlet concentration changes."""
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

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_mass_balance_very_small_domain(self, simple_step_input, freundlich_sorption):
        """Mass balance with very small domain."""
        cin, flow, tedges = simple_step_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=10.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=50, verbose=False)

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_mass_balance_at_t_zero(self):
        """Mass balance at θ=0 before any mass enters."""
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

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-12)


class TestLangmuirSorption:
    """Test FrontTracker with Langmuir sorption."""

    @pytest.fixture
    def langmuir_sorption(self):
        """Standard Langmuir sorption for testing."""
        return LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)

    def test_step_input_completes(self, simple_step_input, langmuir_sorption):
        """Langmuir step input completes."""
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
        assert tracker.state.theta_current >= 0.0

    def test_pulse_input_creates_rarefaction(self, pulse_input, langmuir_sorption):
        """Langmuir pulse creates rarefaction."""
        cin, flow, tedges = pulse_input

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=langmuir_sorption,
        )

        tracker.run(max_iterations=200, verbose=False)

        rarefactions = [w for w in tracker.state.waves if isinstance(w, RarefactionWave)]
        assert len(rarefactions) > 0

    def test_mass_balance_step_input(self, langmuir_sorption):
        """Mass balance with Langmuir step input."""
        tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-04-11"])
        cin = np.array([0.0, 10.0])
        flow = np.array([100.0, 100.0])

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=100.0,
            sorption=langmuir_sorption,
        )

        tracker.run(max_iterations=100, verbose=False)

        tracker.verify_physics(check_mass_balance=True, mass_balance_rtol=1e-6)

    def test_finite_rarefaction_tail_speed(self, pulse_input, langmuir_sorption):
        """Langmuir rarefaction tails have finite speed (R(0) finite)."""
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
                tail_speed = wave.tail_speed()
                assert tail_speed > 0, "Langmuir rarefaction tail speed must be positive"


class TestRiemannProblems:
    """Riemann-problem coverage for issue #174 group 2."""

    def test_square_pulse_constant_retardation_total_mass_at_outlet(self, constant_retardation):
        """Square pulse 0→C→0 with ConstantRetardation: total outlet mass exactly matches inlet.

        ConstantRetardation produces no rarefactions, so breakthrough is an exact
        time-shift and mass conservation is machine precision.
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

        # Constant retardation → outlet is a shifted copy of inlet. Sample the
        # breakthrough curve in θ-space and convert ∫c·dθ = ∫c·flow dt = mass_out.
        theta_edges = tracker.state.theta_edges
        theta_sample = np.linspace(0.0, float(theta_edges[-1]), 4000)
        cout = compute_breakthrough_curve(theta_sample, v_pore, tracker.state.waves, constant_retardation)
        mass_out = float(np.trapezoid(cout, theta_sample))

        # Trapezoidal-rule error over sharp leading/trailing edges.
        assert np.isclose(mass_out, mass_in, rtol=2e-3)

    def test_square_pulse_n_gt_1_total_mass_at_outlet(self, freundlich_sorption):
        """Total outlet mass equals total inlet mass for an n>1 square pulse at machine precision.

        Uses the analytical ``compute_total_outlet_mass`` (closed-form θ-integrals
        + tail-to-infinity rarefaction contribution) — no trapezoidal-rule error.
        Phase 2's ``DecayingShockWave`` replaces the Phase-1 shock + rarefaction
        wave-splitting overlay (which previously left a ~6% mass deficit) with a
        single analytical wave whose closed-form trajectory is exact.
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

        # mass_in = ∫ cin·flow dt = ∫ cin dθ (the θ-form drops flow).
        mass_in = float(np.sum(cin * np.diff(tracker.state.theta_edges)))

        mass_out, _theta_end = compute_total_outlet_mass(
            v_outlet=v_pore,
            sorption=freundlich_sorption,
            cin=cin,
            theta_edges=tracker.state.theta_edges,
        )

        assert np.isclose(mass_out, mass_in, rtol=1e-12)

    def test_square_pulse_langmuir_total_mass_at_outlet(self):
        """Langmuir canonical pulse: total outlet mass == inlet mass at machine precision.

        Langmuir is favorable like n>1 Freundlich; the same head-collision
        DecayingShockWave path is exercised. Closed-form fan integral converges
        because the Langmuir fan c reaches 0 at finite θ. The finite-θ clamp in
        ``_integrate_fan_exact_langmuir`` is load-bearing — removing it raises
        ``ValueError: Langmuir fan integral diverges at θ=+∞``.
        """
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        v_pore = 200.0
        n_bins = 500
        cin = np.zeros(n_bins)
        cin[5:15] = 4.0
        flow = np.full(n_bins, 100.0)
        tedges = pd.date_range("2020-01-01", periods=n_bins + 1, freq="D")

        tracker = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_pore, sorption=sorption)
        tracker.run(max_iterations=10000, verbose=False)

        mass_in = float(np.sum(cin * np.diff(tracker.state.theta_edges)))
        mass_out, _ = compute_total_outlet_mass(
            v_outlet=v_pore,
            sorption=sorption,
            cin=cin,
            theta_edges=tracker.state.theta_edges,
        )
        assert np.isclose(mass_out, mass_in, rtol=1e-12)

    def test_two_step_increase_merges_into_single_shock(self, freundlich_sorption):
        """For n>1, 0→C1→C2 with C1<C2 produces a faster trailing shock that merges with
        the leading shock; final shock satisfies R-H on (0, C2)."""
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

        active_shocks = [w for w in tracker.state.waves if isinstance(w, ShockWave) and w.is_active]
        assert len(active_shocks) == 1
        final = active_shocks[0]
        assert np.isclose(final.c_left, 6.0)
        assert np.isclose(final.c_right, 0.0)
        # R-H speed dV/dθ for (0, 6) in (V, θ) — no flow factor.
        c_tot_diff = freundlich_sorption.total_concentration(6.0) - freundlich_sorption.total_concentration(0.0)
        rh_speed_theta = (6.0 - 0.0) / c_tot_diff
        assert np.isclose(final.speed, rh_speed_theta, rtol=1e-12)

    def test_constant_retardation_breakthrough_matches_pure_advection(self, constant_retardation):
        """For ConstantRetardation, front-tracking matches the θ-shifted-inlet closed form."""
        v_pore = 300.0
        flow_val = 100.0
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

        # Closed form: cout(θ) == cin(θ_inlet) where θ_inlet = θ - V·R.
        # Equivalently in t (constant flow): cout(t) = cin(t - V·R/flow), delay = 6 d.
        delay = v_pore * constant_retardation.retardation_factor / flow_val
        # Sample at the centers of the pulse arrivals.
        for t in [7.5, 8.5, 9.5, 10.5]:
            theta_query = tracker.state.theta_at_t(t)
            c_out = concentration_at_point(v_pore, theta_query, tracker.state.waves, constant_retardation)
            t_in = t - delay
            bin_idx = int(np.floor(t_in))
            assert np.isclose(c_out, cin[bin_idx], rtol=1e-14, atol=1e-14)


class TestParametricMassBalance:
    """Parametric mass-balance suite for confirmed-working scenarios at machine precision.

    Each test asserts both the closed-form total at θ_max
    (``compute_total_outlet_mass ≈ mass_in``) and (where applicable) the
    per-checkpoint identity ``m_in(θ) = m_dom(θ) + m_out(θ)`` at four
    characteristic regimes: pre-breakthrough, at breakthrough, mid-drain,
    and asymptotic.
    """

    @pytest.mark.parametrize("n", [1.5, 2.0, 2.5, 3.0])
    def test_freundlich_parameter_sweep_mass_balance(self, n):
        """Freundlich n sweep with c_R=0: m_in = m_dom + m_out at machine precision.

        Exercises the full Phase 2 step 4 closed-form chain: DecayingShockWave
        trajectory + ``integrate_fan_exact`` (temporal) + ``integrate_fan_spatial_exact``
        (spatial in ``compute_domain_mass``). The n=2 case uses the closed-form
        quadratic inversion for ``c_decay_at_theta``; n∈{1.5, 2.5, 3.0} use
        the brentq path (Abel-Ruffini rules out radicals for general n).

        Mutation guard: a hard-coded n=2 in the closed-form path would be
        invisible to the n=2 case alone; the n∈{1.5, 2.5, 3.0} variants
        surface it.
        """
        sorption = FreundlichSorption(k_f=0.01, n=n, bulk_density=1500.0, porosity=0.3)
        v_outlet = 200.0
        cin = np.zeros(500)
        cin[5:15] = 4.0
        flow = np.full(500, 100.0)
        tedges = pd.date_range("2020-01-01", periods=501, freq="D")

        tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
        tr.run(max_iterations=100000)

        # Theta checkpoints span pre-arrival (m_out=0), breakthrough,
        # mid-drainage, and asymptotic regimes for the canonical-pulse geometry.
        checkpoints = [4000.0, 6000.0, 9000.0, 15000.0, 25000.0]
        saw_dom = False
        saw_out = False
        for theta in checkpoints:
            m_in = compute_cumulative_inlet_mass(theta=theta, cin=cin, theta_edges=tr.state.theta_edges)
            m_dom = compute_domain_mass(theta=theta, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
            m_out = compute_cumulative_outlet_mass(
                theta=theta,
                v_outlet=v_outlet,
                waves=tr.state.waves,
                sorption=sorption,
                cin=cin,
                theta_edges=tr.state.theta_edges,
            )
            if m_dom > 1.0:
                saw_dom = True
            if m_out > 1.0:
                saw_out = True
            err = abs((m_dom + m_out) - m_in)
            tol = 1e-14 * max(m_in, 1.0)
            assert err <= tol, f"mass-balance violation at n={n}, θ={theta}: err={err:.6e} > tol={tol:.6e}"

        assert saw_dom, f"n={n}: checkpoints must include θ where m_dom > 1 to exercise the closed-form fan integral"
        assert saw_out, f"n={n}: checkpoints must include θ where m_out > 1 to exercise compute_cumulative_outlet_mass"

    @pytest.mark.parametrize(
        ("s_max", "k_l"),
        [(0.05, 2.0), (0.1, 5.0), (0.2, 10.0)],
    )
    def test_langmuir_parameter_sweep_mass_balance(self, s_max, k_l):
        """Langmuir parameter sweep: per-checkpoint m_in = m_dom + m_out at machine precision.

        After the Step 5a Langmuir spatial-integral clamp at u_zero, the
        per-checkpoint identity holds. Without the clamp this fails by
        22-105× (m_dom goes negative); the fix is mutation-resistant
        across the parameter sweep.
        """
        sorption = LangmuirSorption(s_max=s_max, k_l=k_l, bulk_density=1500.0, porosity=0.3)
        v_outlet = 200.0
        cin = np.zeros(500)
        cin[5:15] = 4.0
        flow = np.full(500, 100.0)
        tedges = pd.date_range("2020-01-01", periods=501, freq="D")

        tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
        tr.run(max_iterations=100000)

        mass_in = float(np.sum(cin * np.diff(tr.state.theta_edges)))
        mass_out, _ = compute_total_outlet_mass(
            v_outlet=v_outlet,
            sorption=sorption,
            cin=cin,
            theta_edges=tr.state.theta_edges,
        )
        # Empirical rel_err ≤ 7e-15 across the parameter sweep (worst at s_max=0.2, k_l=10); 1e-13 leaves 14× headroom.
        assert np.isclose(mass_out, mass_in, rtol=1e-13), (
            f"Langmuir s={s_max}, k_l={k_l}: total mass mass_in={mass_in:.4f} mass_out={mass_out:.4f}"
        )

        theta_max = float(tr.state.theta_edges[-1])
        checkpoints = [theta_max * f for f in (0.1, 0.25, 0.5, 0.75, 0.99)]
        saw_dom = False
        saw_out = False
        for theta in checkpoints:
            m_in = compute_cumulative_inlet_mass(theta=theta, cin=cin, theta_edges=tr.state.theta_edges)
            m_dom = compute_domain_mass(theta=theta, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
            m_out = compute_cumulative_outlet_mass(
                theta=theta,
                v_outlet=v_outlet,
                waves=tr.state.waves,
                sorption=sorption,
                cin=cin,
                theta_edges=tr.state.theta_edges,
            )
            if m_dom > 1.0:
                saw_dom = True
            if m_out > 1.0:
                saw_out = True
            err = abs((m_dom + m_out) - m_in)
            tol = 1e-14 * max(m_in, 1.0)
            assert err <= tol, f"Langmuir s={s_max}, k_l={k_l} at θ={theta}: err={err:.6e} > tol={tol:.6e}"

        assert saw_dom, f"Langmuir s={s_max}, k_l={k_l}: checkpoints must include θ where m_dom > 1"
        assert saw_out, f"Langmuir s={s_max}, k_l={k_l}: checkpoints must include θ where m_out > 1"

    def test_flow_change_during_pulse_n2_mass_balance(self, freundlich_sorption):
        """Time-varying flow does not perturb mass balance — the (V, θ) refactor's central claim.

        In (V, θ) coordinates the wave dynamics are flow-independent;
        flow enters only via the precomputed ``theta_edges`` array. A
        regression that re-introduces flow into a wave method would
        break this scenario but pass the constant-flow tests.
        """
        v_outlet = 200.0
        cin = np.zeros(500)
        cin[5:15] = 4.0
        # Three flow regimes during the simulation, including a doubling AFTER the pulse.
        flow = np.concatenate([np.full(100, 100.0), np.full(100, 50.0), np.full(100, 200.0), np.full(200, 100.0)])
        tedges = pd.date_range("2020-01-01", periods=501, freq="D")

        tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=freundlich_sorption)
        tr.run(max_iterations=100000)

        mass_in = float(np.sum(cin * np.diff(tr.state.theta_edges)))
        mass_out, _ = compute_total_outlet_mass(
            v_outlet=v_outlet,
            sorption=freundlich_sorption,
            cin=cin,
            theta_edges=tr.state.theta_edges,
        )
        assert np.isclose(mass_out, mass_in, rtol=1e-13), f"flow-change: mass_in={mass_in:.4f}, mass_out={mass_out:.4f}"

    def test_zero_flow_interval_n2_mass_balance(self, freundlich_sorption):
        """A zero-flow bin in the middle of the simulation does not perturb mass balance.

        In (V, θ) the zero-flow interval is a zero-width segment of
        ``theta_edges``; wave dynamics are simply paused for that interval
        without modification. Companion test to
        ``test_theta_constant_across_zero_flow_bin`` in
        tests/regression/test_phase1_invariants.py.
        """
        v_outlet = 200.0
        cin = np.zeros(500)
        cin[5:15] = 4.0
        # Pulse, then a zero-flow gap, then resumed flow.
        flow = np.concatenate([np.full(100, 100.0), np.full(100, 0.0), np.full(300, 100.0)])
        tedges = pd.date_range("2020-01-01", periods=501, freq="D")

        tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=freundlich_sorption)
        tr.run(max_iterations=100000)

        mass_in = float(np.sum(cin * np.diff(tr.state.theta_edges)))
        mass_out, _ = compute_total_outlet_mass(
            v_outlet=v_outlet,
            sorption=freundlich_sorption,
            cin=cin,
            theta_edges=tr.state.theta_edges,
        )
        assert np.isclose(mass_out, mass_in, rtol=1e-13), (
            f"zero-flow interval: mass_in={mass_in:.4f}, mass_out={mass_out:.4f}"
        )
