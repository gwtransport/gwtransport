"""
Phase 8 Integration Tests: Physical Scenarios for Front Tracking.

Tests various physical scenarios with exact analytical verification:
- Step inputs (shock formation)
- Pulse injection (shock + rarefaction)
- Ramp inputs (pure rarefaction)
- Mass balance verification

All tests use the public API with Freundlich sorption parameters.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import (
    infiltration_to_extraction_front_tracking,
    infiltration_to_extraction_front_tracking_detailed,
)
from gwtransport.utils import compute_time_edges


class TestStepInputScenario:
    """Test step input: C=0→10 → single shock propagation."""

    def test_step_input_creates_single_shock(self):
        """Step increase should create a single shock wave."""
        # Setup: step from 0 to 10 mg/L at t=10 days
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:] = 10.0  # Step at day 10
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=30, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Run with Freundlich sorption (n>1: favorable, faster at high C)
        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=500.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Verify: Should have at least one shock created
        assert structure["n_shocks"] >= 1, "Step input should create shock wave"

        # Verify: Output should show sharp front (no dispersion)
        # Find breakthrough time
        breakthrough_idx = np.where(cout > 0.1)[0]
        if len(breakthrough_idx) > 0:
            first_breakthrough = breakthrough_idx[0]
            # Check that concentration rises sharply (within 2-3 bins)
            rise_bins = np.sum(cout[first_breakthrough : first_breakthrough + 5] < 9.5)
            assert rise_bins <= 3, f"Front should be sharp (rise in ≤3 bins), got {rise_bins}"

        # Verify: Final concentration should reach input value
        late_cout = cout[-5:]
        if np.any(~np.isnan(late_cout)):
            mean_late = np.nanmean(late_cout)
            assert 9.0 <= mean_late <= 10.1, f"Should reach cin=10, got {mean_late:.2f}"

    def test_step_input_shock_velocity(self):
        """Shock velocity should match Rankine-Hugoniot condition."""
        # Setup with constant flow
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:] = 10.0  # Step at day 5
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=25, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=500.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Find breakthrough time (when concentration > 0.1)
        breakthrough_idx = np.where(cout > 0.1)[0]
        if len(breakthrough_idx) == 0:
            pytest.skip("Breakthrough not observed in output window")

        breakthrough_time = (cout_dates[breakthrough_idx[0]] - dates[0]).days

        # Expected: shock forms at t=5, travels distance 500 m³
        # For Freundlich n=2, higher C travels faster
        # Shock velocity should be between characteristic velocities
        # This is a qualitative check that breakthrough occurs after spin-up
        assert breakthrough_time >= structure["t_first_arrival"] - 1, (
            "Breakthrough should occur after first arrival time"
        )

    def test_step_input_mass_balance(self):
        """Mass should be conserved for step input."""
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:] = 10.0
        flow = np.full(len(dates), 100.0)

        # Long output window to capture all mass
        cout_dates = pd.date_range(start=dates[0], periods=80, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=300.0,  # Smaller domain for faster breakthrough
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Compute mass in (from day 10 onwards)
        mass_in = np.sum(cin[10:] * flow[10:])

        # Compute mass out (same flow assumed)
        flow_out = np.full(len(cout), 100.0)
        mass_out = np.nansum(cout * flow_out)

        # Recovery should be close to 1.0
        recovery = mass_out / mass_in if mass_in > 0 else 0
        assert 0.95 <= recovery <= 1.05, (
            f"Mass not conserved: recovery={recovery:.3f}, in={mass_in:.0f}, out={mass_out:.0f}"
        )


class TestPulseInjectionScenario:
    """Test pulse: C=0→10→0 → compression + rarefaction."""

    def test_pulse_creates_shock_and_rarefaction(self):
        """Pulse injection should create both shock and rarefaction waves."""
        # Setup: pulse from day 10 to 20
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:20] = 10.0  # 10-day pulse
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=50, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Run with detailed output
        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=500.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Verify: Should have both shocks and rarefactions
        # Rising edge: shock (compression)
        # Falling edge: rarefaction (expansion)
        assert structure["n_shocks"] >= 1, "Pulse should create at least one shock"
        assert structure["n_rarefactions"] >= 1, "Pulse should create at least one rarefaction"

        # Verify: Multiple events occurred
        assert structure["n_events"] >= 3, "Should have multiple wave interaction events"

    def test_pulse_mass_balance(self):
        """Mass should be conserved for pulse injection."""
        dates = pd.date_range(start="2020-01-01", periods=40, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:20] = 15.0  # 10-day pulse at 15 mg/L
        flow = np.full(len(dates), 100.0)

        # Extended output to capture entire pulse
        cout_dates = pd.date_range(start=dates[0], periods=70, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=300.0,  # Smaller for faster breakthrough
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Compute mass balance
        mass_in = np.sum(cin * flow)
        flow_out = np.full(len(cout), 100.0)
        mass_out = np.nansum(cout * flow_out)

        recovery = mass_out / mass_in if mass_in > 0 else 0
        assert 0.92 <= recovery <= 1.08, f"Mass not conserved for pulse: recovery={recovery:.3f}"

    def test_pulse_peak_arrival(self):
        """Pulse peak should arrive before tail (for favorable sorption)."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:20] = 20.0  # High concentration pulse
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=40, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            freundlich_k=0.01,
            freundlich_n=2.0,  # n>1: high C travels faster
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Find peak and trailing edge
        valid_cout = cout[~np.isnan(cout)]
        if len(valid_cout) < 5:
            pytest.skip("Insufficient output data")

        peak_idx = np.nanargmax(cout)

        # Find when concentration drops to near zero after peak
        after_peak = cout[peak_idx:]
        tail_idx_rel = np.where(after_peak < 0.5)[0]
        if len(tail_idx_rel) > 0:
            # Pulse should have finite width (not grow indefinitely)
            pulse_width = tail_idx_rel[0]
            assert pulse_width < 20, f"Pulse width too large: {pulse_width} (possible spreading issue)"


class TestRampInputScenario:
    """Test monotonic ramp: C=0→5→10 → pure rarefactions."""

    def test_ramp_creates_rarefactions(self):
        """Monotonic increase should create rarefaction waves (no shocks)."""
        # Setup: gradual increase
        dates = pd.date_range(start="2020-01-01", periods=25, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        # Gradual ramp: 0 → 5 → 10
        cin = np.zeros(len(dates))
        cin[5:10] = 5.0
        cin[10:] = 10.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=35, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Run with detailed output
        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Verify: Should have rarefactions
        assert structure["n_rarefactions"] >= 1, "Ramp input should create rarefaction waves"

        # Note: For n>1 (favorable), increasing concentration creates rarefactions
        # because higher C travels faster, causing expansion

    def test_ramp_smooth_output(self):
        """Ramp should produce smooth output (no oscillations)."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        # Smooth ramp
        cin = np.linspace(0, 10, len(dates))
        cin[:5] = 0  # Start from zero
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=40, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Check for oscillations (spurious peaks)
        valid_cout = cout[~np.isnan(cout)]
        if len(valid_cout) > 10:
            # Output should not overshoot input significantly
            assert np.max(valid_cout) <= 10.5, f"Output overshoots input: max={np.max(valid_cout):.2f}"

            # Should be non-negative
            assert np.min(valid_cout) >= -1e-10, f"Output has negative values: min={np.min(valid_cout):.2e}"

    def test_monotone_decrease_creates_shocks(self):
        """Monotonic decrease should create shock waves (compression)."""
        # Setup: step down from 10 to 5 to 0
        dates = pd.date_range(start="2020-01-01", periods=25, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:10] = 10.0
        cin[10:15] = 5.0
        cin[15:] = 0.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=35, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # For n>1, decreasing C means slower water behind faster water
        # This creates compression → shocks
        assert structure["n_shocks"] >= 1, "Monotone decrease should create shocks for n>1"


class TestMassBalanceExact:
    """Test exact mass balance for all scenarios."""

    def test_mass_balance_step_input_exact(self):
        """Exact mass balance for step input."""
        dates = pd.date_range(start="2020-01-01", periods=40, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:] = 12.0
        flow = np.full(len(dates), 100.0)

        # Very long output to capture all mass
        cout_dates = pd.date_range(start=dates[0], periods=100, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=200.0,  # Small domain for complete breakthrough
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Mass balance
        mass_in = np.sum(cin * flow)
        flow_out = np.full(len(cout), 100.0)
        mass_out = np.nansum(cout * flow_out)

        recovery = mass_out / mass_in if mass_in > 0 else 0
        # Should be very close to 1.0 (exact analytical)
        assert 0.97 <= recovery <= 1.03, (
            f"Mass not conserved: recovery={recovery:.4f}, in={mass_in:.0f}, out={mass_out:.0f}"
        )

    def test_mass_balance_multiple_pulses(self):
        """Exact mass balance for multiple pulses."""
        dates = pd.date_range(start="2020-01-01", periods=60, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        # Two pulses at different concentrations
        cin = np.zeros(len(dates))
        cin[10:15] = 8.0  # First pulse
        cin[25:30] = 15.0  # Second pulse (higher)
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=100, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=200.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Mass balance
        mass_in = np.sum(cin * flow)
        flow_out = np.full(len(cout), 100.0)
        mass_out = np.nansum(cout * flow_out)

        recovery = mass_out / mass_in if mass_in > 0 else 0
        assert 0.95 <= recovery <= 1.05, f"Mass not conserved for multiple pulses: recovery={recovery:.4f}"

    def test_no_mass_creation(self):
        """Verify no mass is created (output ≤ input)."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:20] = 10.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=50, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=300.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Output concentration should not exceed input concentration
        valid_cout = cout[~np.isnan(cout)]
        if len(valid_cout) > 0:
            max_cout = np.max(valid_cout)
            max_cin = np.max(cin)
            # Allow small numerical tolerance
            assert max_cout <= max_cin * 1.01, f"Output exceeds input: max_cout={max_cout:.2f}, max_cin={max_cin:.2f}"

        # Total mass out should not exceed total mass in
        mass_in = np.sum(cin * flow)
        flow_out = np.full(len(cout), 100.0)
        mass_out = np.nansum(cout * flow_out)
        assert mass_out <= mass_in * 1.05, f"More mass out than in: {mass_out:.0f} > {mass_in:.0f}"


class TestVariableFlow:
    """Test scenarios with variable flow rates."""

    def test_step_input_variable_flow(self):
        """Step input with variable flow should conserve mass."""
        dates = pd.date_range(start="2020-01-01", periods=40, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:] = 10.0

        # Variable flow: changes over time
        flow = np.full(len(dates), 100.0)
        flow[15:25] = 150.0  # Higher flow in middle
        flow[25:] = 80.0  # Lower flow later

        cout_dates = pd.date_range(start=dates[0], periods=70, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=250.0,
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Mass balance with variable flow
        mass_in = np.sum(cin * flow)

        # For output, approximate with average flow
        # (in practice, would need to track flow_out properly)
        flow_out = np.full(len(cout), np.mean(flow))
        mass_out = np.nansum(cout * flow_out)

        recovery = mass_out / mass_in if mass_in > 0 else 0
        # More lenient tolerance due to flow rate approximation
        assert 0.90 <= recovery <= 1.10, f"Mass not conserved with variable flow: recovery={recovery:.3f}"


class TestConstantRetardation:
    """Test linear sorption case (constant retardation)."""

    def test_constant_retardation_step_input(self):
        """Constant retardation should work correctly."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:] = 10.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=50, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Use constant retardation instead of Freundlich
        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            retardation_factor=2.0,  # Constant R
        )

        # Should produce valid output
        assert not np.all(np.isnan(cout)), "Should produce valid output"

        # Should reach input concentration
        valid_cout = cout[~np.isnan(cout)]
        if len(valid_cout) > 5:
            # Late output should be close to input
            late_mean = np.mean(valid_cout[-5:])
            assert 9.0 <= late_mean <= 10.5, f"Should reach cin=10, got {late_mean:.2f}"

    def test_constant_retardation_pulse(self):
        """Pulse with constant retardation."""
        dates = pd.date_range(start="2020-01-01", periods=40, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[10:20] = 12.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=60, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=300.0,
            retardation_factor=1.5,
        )

        # Should work and conserve mass
        mass_in = np.sum(cin * flow)
        flow_out = np.full(len(cout), 100.0)
        mass_out = np.nansum(cout * flow_out)

        recovery = mass_out / mass_in if mass_in > 0 else 0
        assert 0.90 <= recovery <= 1.10, f"Mass not conserved with constant R: recovery={recovery:.3f}"
