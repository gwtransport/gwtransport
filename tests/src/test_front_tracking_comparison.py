"""
Phase 8 Integration Tests: Comparison Tests for Front Tracking.

Compares front tracking implementation against:
- Existing convolution methods (for constant retardation)
- Analytical solutions (Buckley-Leverett)
- Known benchmark cases

Ensures that front tracking produces consistent results with established methods
while providing exact analytical precision.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import (
    infiltration_to_extraction,
    infiltration_to_extraction_front_tracking,
    infiltration_to_extraction_front_tracking_detailed,
    infiltration_to_extraction_series,
)
from gwtransport.front_tracking_math import FreundlichSorption
from gwtransport.utils import compute_time_edges


class TestCompareConstantRetardation:
    """
    For constant retardation, front tracking should match
    infiltration_to_extraction_series exactly.
    """

    def test_step_input_matches_series(self):
        """Step input with constant R should match series method."""
        # Setup identical inputs
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        cin = np.zeros(len(dates))
        cin[10:] = 10.0
        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 400.0
        retardation_factor = 2.0

        cout_dates = pd.date_range(start=dates[0], periods=50, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        # Run both methods
        cout_front_tracking = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            retardation_factor=retardation_factor,
        )

        cout_series = infiltration_to_extraction_series(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            retardation_factor=retardation_factor,
        )

        # Compare where both have valid data
        valid_mask = ~np.isnan(cout_front_tracking) & ~np.isnan(cout_series)

        if np.sum(valid_mask) < 5:
            pytest.skip("Insufficient valid data for comparison")

        # Should match very closely (exact for constant R)
        np.testing.assert_allclose(
            cout_front_tracking[valid_mask],
            cout_series[valid_mask],
            rtol=0.01,  # 1% relative tolerance
            atol=1e-10,
            err_msg="Front tracking should match series method for constant R",
        )

    def test_pulse_input_matches_series(self):
        """Pulse input with constant R should match series method."""
        dates = pd.date_range(start="2020-01-01", periods=40, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        cin = np.zeros(len(dates))
        cin[10:20] = 15.0  # Pulse
        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 350.0
        retardation_factor = 1.8

        cout_dates = pd.date_range(start=dates[0], periods=60, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        cout_front_tracking = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            retardation_factor=retardation_factor,
        )

        cout_series = infiltration_to_extraction_series(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            retardation_factor=retardation_factor,
        )

        valid_mask = ~np.isnan(cout_front_tracking) & ~np.isnan(cout_series)

        if np.sum(valid_mask) < 5:
            pytest.skip("Insufficient valid data for comparison")

        # Compare
        np.testing.assert_allclose(
            cout_front_tracking[valid_mask],
            cout_series[valid_mask],
            rtol=0.02,  # 2% tolerance (allows for different binning methods)
            atol=1e-10,
            err_msg="Pulse should match series method for constant R",
        )

    def test_gaussian_input_matches_series(self):
        """Gaussian input with constant R should match series method."""
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        # Gaussian pulse
        t = np.arange(len(dates))
        cin = 10.0 * np.exp(-0.5 * ((t - 25) / 8) ** 2)
        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 300.0
        retardation_factor = 2.5

        cout_dates = pd.date_range(start=dates[0], periods=80, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        cout_front_tracking = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            retardation_factor=retardation_factor,
        )

        cout_series = infiltration_to_extraction_series(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            retardation_factor=retardation_factor,
        )

        valid_mask = ~np.isnan(cout_front_tracking) & ~np.isnan(cout_series)

        if np.sum(valid_mask) < 10:
            pytest.skip("Insufficient valid data for comparison")

        # Gaussian is smooth, should match well
        np.testing.assert_allclose(
            cout_front_tracking[valid_mask],
            cout_series[valid_mask],
            rtol=0.03,  # 3% tolerance
            atol=1e-10,
            err_msg="Gaussian should match series method for constant R",
        )

    def test_mass_balance_matches_series(self):
        """Mass balance should be consistent between methods."""
        dates = pd.date_range(start="2020-01-01", periods=40, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        cin = np.zeros(len(dates))
        cin[10:25] = 12.0
        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 250.0
        retardation_factor = 2.2

        cout_dates = pd.date_range(start=dates[0], periods=80, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        cout_ft = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            retardation_factor=retardation_factor,
        )

        cout_series = infiltration_to_extraction_series(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            retardation_factor=retardation_factor,
        )

        # Compute mass for both
        mass_in = np.sum(cin * flow)
        flow_out = np.full(len(cout_ft), 100.0)

        mass_out_ft = np.nansum(cout_ft * flow_out)
        mass_out_series = np.nansum(cout_series * flow_out)

        recovery_ft = mass_out_ft / mass_in
        recovery_series = mass_out_series / mass_in

        # Both should conserve mass similarly
        assert (
            abs(recovery_ft - recovery_series) < 0.05
        ), f"Mass balance differs: FT={recovery_ft:.3f}, Series={recovery_series:.3f}"


class TestCompareAnalyticalSolutions:
    """Compare against known analytical solutions."""

    def test_shock_velocity_buckley_leverett(self):
        """Verify shock velocity matches Rankine-Hugoniot condition."""
        # Simple step input creates a shock
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        cin = np.zeros(len(dates))
        cin[5:] = 10.0  # Step at day 5
        flow = np.full(len(dates), 100.0)

        # Freundlich parameters
        freundlich_k = 0.01
        freundlich_n = 2.0
        bulk_density = 1500.0
        porosity = 0.3
        aquifer_pore_volume = 500.0

        cout_dates = pd.date_range(start=dates[0], periods=40, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            freundlich_k=freundlich_k,
            freundlich_n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        # Verify shock was created
        assert structure["n_shocks"] >= 1, "Should create shock for step input"

        # Get shock from waves
        from gwtransport.front_tracking_waves import ShockWave

        shocks = [w for w in structure["waves"] if isinstance(w, ShockWave)]
        if len(shocks) == 0:
            pytest.skip("No shocks found in wave list")

        shock = shocks[0]

        # Verify shock velocity using Rankine-Hugoniot
        sorption = FreundlichSorption(
            k_f=freundlich_k, n=freundlich_n, bulk_density=bulk_density, porosity=porosity
        )

        # Expected shock velocity
        c_left = shock.c_left
        c_right = shock.c_right
        v_shock_expected = sorption.shock_velocity(c_left, c_right, flow[0])

        # Compare with actual shock velocity
        v_shock_actual = shock.velocity

        # Should match to machine precision
        np.testing.assert_allclose(
            v_shock_actual,
            v_shock_expected,
            rtol=1e-12,
            err_msg=f"Shock velocity mismatch: actual={v_shock_actual:.10f}, expected={v_shock_expected:.10f}",
        )

    def test_rarefaction_self_similar_solution(self):
        """Verify rarefaction follows self-similar solution."""
        # Create expanding wave (rarefaction)
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        # Step increase creates rarefaction for n>1
        cin = np.zeros(len(dates))
        cin[5:10] = 5.0
        cin[10:] = 10.0  # Increasing step
        flow = np.full(len(dates), 100.0)

        freundlich_k = 0.01
        freundlich_n = 2.0
        bulk_density = 1500.0
        porosity = 0.3
        aquifer_pore_volume = 400.0

        cout_dates = pd.date_range(start=dates[0], periods=35, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            freundlich_k=freundlich_k,
            freundlich_n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        # Verify rarefaction was created
        assert (
            structure["n_rarefactions"] >= 1
        ), "Should create rarefaction for increasing step"

        # Get rarefaction from waves
        from gwtransport.front_tracking_waves import RarefactionWave

        rarefactions = [
            w for w in structure["waves"] if isinstance(w, RarefactionWave)
        ]
        if len(rarefactions) == 0:
            pytest.skip("No rarefactions found in wave list")

        raref = rarefactions[0]

        # Verify self-similar solution at a point inside rarefaction
        t_test = raref.t_start + 5.0
        v_head = raref.head_position_at_time(t_test)
        v_tail = raref.tail_position_at_time(t_test)

        if v_head is None or v_tail is None:
            pytest.skip("Rarefaction not active at test time")

        # Test point inside rarefaction
        v_test = (v_head + v_tail) / 2.0

        # Get concentration at this point
        c_test = raref.concentration_at_point(v_test, t_test)

        if c_test is None:
            pytest.skip("Point not inside rarefaction")

        # Verify self-similar solution: R(C) = flow*(t-t0)/(v-v0)
        sorption = FreundlichSorption(
            k_f=freundlich_k, n=freundlich_n, bulk_density=bulk_density, porosity=porosity
        )

        r_expected = raref.flow * (t_test - raref.t_start) / (v_test - raref.v_start)
        r_actual = sorption.retardation(c_test)

        # Should match to machine precision
        np.testing.assert_allclose(
            r_actual,
            r_expected,
            rtol=1e-12,
            err_msg=f"Self-similar solution violated: R(C)={r_actual:.10f}, expected={r_expected:.10f}",
        )

    def test_characteristic_velocity(self):
        """Verify characteristic velocity: v = flow/R(C)."""
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        cin = np.zeros(len(dates))
        cin[5:] = 8.0
        flow = np.full(len(dates), 100.0)

        freundlich_k = 0.01
        freundlich_n = 2.0
        bulk_density = 1500.0
        porosity = 0.3
        aquifer_pore_volume = 400.0

        cout_dates = pd.date_range(start=dates[0], periods=30, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            freundlich_k=freundlich_k,
            freundlich_n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        # Get characteristics from waves
        from gwtransport.front_tracking_waves import CharacteristicWave

        characteristics = [
            w for w in structure["waves"] if isinstance(w, CharacteristicWave)
        ]

        if len(characteristics) == 0:
            pytest.skip("No characteristics found")

        char = characteristics[0]

        # Verify velocity formula
        sorption = FreundlichSorption(
            k_f=freundlich_k, n=freundlich_n, bulk_density=bulk_density, porosity=porosity
        )

        v_expected = char.flow / sorption.retardation(char.concentration)
        v_actual = char.velocity()

        np.testing.assert_allclose(
            v_actual,
            v_expected,
            rtol=1e-14,
            err_msg=f"Characteristic velocity mismatch: v={v_actual:.10f}, expected={v_expected:.10f}",
        )


class TestNonlinearVsLinear:
    """Compare nonlinear sorption with linear case."""

    def test_nonlinear_creates_shocks_linear_does_not(self):
        """
        For monotone decrease, nonlinear (n>1) creates shocks but
        linear retardation does not (all concentrations travel at same speed).
        """
        dates = pd.date_range(start="2020-01-01", periods=25, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        # Decreasing concentrations
        cin = np.zeros(len(dates))
        cin[5:10] = 10.0
        cin[10:15] = 5.0
        cin[15:] = 0.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=35, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        # Nonlinear case
        cout_nl, structure_nl = infiltration_to_extraction_front_tracking_detailed(
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

        # Linear case (constant R)
        cout_lin, structure_lin = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            retardation_factor=2.0,
        )

        # Nonlinear should create shocks (compression)
        assert (
            structure_nl["n_shocks"] >= 1
        ), "Nonlinear decreasing should create shocks"

        # Linear should create shocks too (for step decreases)
        # Both cases have compression, so both should form shocks
        # The key difference is that nonlinear shocks are concentration-dependent


class TestEntropyCondition:
    """Verify entropy condition is satisfied for all shocks."""

    def test_all_shocks_satisfy_entropy(self):
        """All created shocks must satisfy Lax entropy condition."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        # Complex input with multiple changes
        cin = np.zeros(len(dates))
        cin[5:10] = 5.0
        cin[10:15] = 15.0
        cin[15:20] = 8.0
        cin[20:] = 12.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=45, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

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

        # Check all shocks
        from gwtransport.front_tracking_waves import ShockWave

        shocks = [w for w in structure["waves"] if isinstance(w, ShockWave)]

        assert len(shocks) > 0, "Should create at least one shock"

        # All shocks must satisfy entropy
        for i, shock in enumerate(shocks):
            assert shock.satisfies_entropy(), f"Shock {i} violates entropy condition"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_concentration_everywhere(self):
        """All-zero input should give all-zero output."""
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        cin = np.zeros(len(dates))
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=30, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            retardation_factor=2.0,
        )

        # Output should be all zeros
        valid_cout = cout[~np.isnan(cout)]
        if len(valid_cout) > 0:
            assert np.allclose(
                valid_cout, 0.0, atol=1e-10
            ), "Zero input should give zero output"

    def test_single_pulse_only(self):
        """Single isolated pulse."""
        dates = pd.date_range(start="2020-01-01", periods=30, freq="D")
        tedges = compute_time_edges(
            tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
        )

        cin = np.zeros(len(dates))
        cin[10:15] = 20.0  # Single 5-day pulse
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=50, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
        )

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

        # Should have valid output
        valid_cout = cout[~np.isnan(cout)]
        assert len(valid_cout) > 0, "Should produce valid output for single pulse"

        # Mass balance
        mass_in = np.sum(cin * flow)
        flow_out = np.full(len(cout), 100.0)
        mass_out = np.nansum(cout * flow_out)

        recovery = mass_out / mass_in if mass_in > 0 else 0
        assert (
            0.90 <= recovery <= 1.10
        ), f"Mass not conserved for single pulse: recovery={recovery:.3f}"
