"""
Phase 8 Integration Tests: Scenarios and Validation.

Simplified tests focusing on core integration and validation.
All tests have correct physics expectations for Freundlich n>1.
"""

import numpy as np
import pandas as pd

from gwtransport.advection import (
    infiltration_to_extraction_front_tracking,
    infiltration_to_extraction_front_tracking_detailed,
)
from gwtransport.front_tracking_math import FreundlichSorption
from gwtransport.front_tracking_waves import RarefactionWave, ShockWave
from gwtransport.utils import compute_time_edges


class TestWaveCreation:
    """Test correct wave type creation."""

    def test_step_increase_creates_shock(self):
        """Step increase (0.1→10) creates shock for n>1 (fast catches slow)."""
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        # Note: Use 0.1 instead of 0 to avoid edge case with C=0
        cin = np.full(len(dates), 0.1)
        cin[5:] = 10.0  # Step increase
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=25, freq="D")
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

        # For n>1: high C is faster, so step increase = compression = shock
        assert structure["n_shocks"] >= 1, "Step increase should create shock for n>1"

    def test_step_decrease_creates_rarefaction(self):
        """Step decrease (10→0) creates rarefaction for n>1 (slow follows fast)."""
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.full(len(dates), 10.0)
        cin[8:] = 0.1  # Step decrease (use 0.1 instead of 0 to avoid R→∞)
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=25, freq="D")
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

        # For n>1: low C is slower, so step decrease = expansion = rarefaction
        assert structure["n_rarefactions"] >= 1, "Step decrease should create rarefaction for n>1"


class TestAnalyticalCorrectness:
    """Test analytical solution correctness."""

    def test_shock_velocity_rankine_hugoniot(self):
        """Shock velocity must satisfy Rankine-Hugoniot exactly."""
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:] = 10.0
        flow = np.full(len(dates), 100.0)

        freundlich_k, freundlich_n = 0.01, 2.0
        bulk_density, porosity = 1500.0, 0.3

        cout_dates = pd.date_range(start=dates[0], periods=25, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            freundlich_k=freundlich_k,
            freundlich_n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        # Verify all shocks
        sorption = FreundlichSorption(k_f=freundlich_k, n=freundlich_n, bulk_density=bulk_density, porosity=porosity)

        shocks = [w for w in structure["waves"] if isinstance(w, ShockWave)]
        for shock in shocks:
            v_expected = sorption.shock_velocity(shock.c_left, shock.c_right, shock.flow)
            np.testing.assert_allclose(
                shock.velocity,
                v_expected,
                rtol=1e-14,
                err_msg=f"Shock velocity error: {shock.velocity} != {v_expected}",
            )

    def test_rarefaction_self_similar(self):
        """Rarefaction must follow self-similar solution exactly."""
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.full(len(dates), 10.0)
        cin[8:] = 0.5  # Step down
        flow = np.full(len(dates), 100.0)

        freundlich_k, freundlich_n = 0.01, 2.0
        bulk_density, porosity = 1500.0, 0.3

        cout_dates = pd.date_range(start=dates[0], periods=25, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=400.0,
            freundlich_k=freundlich_k,
            freundlich_n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        sorption = FreundlichSorption(k_f=freundlich_k, n=freundlich_n, bulk_density=bulk_density, porosity=porosity)

        # Test rarefactions
        rarefactions = [w for w in structure["waves"] if isinstance(w, RarefactionWave)]
        for raref in rarefactions:
            t_test = raref.t_start + 3.0
            v_head = raref.head_position_at_time(t_test)
            v_tail = raref.tail_position_at_time(t_test)

            if v_head is None or v_tail is None:
                continue

            v_mid = (v_head + v_tail) / 2
            c_mid = raref.concentration_at_point(v_mid, t_test)

            if c_mid is not None:
                # Verify R(C) = flow*(t-t0)/(v-v0)
                r_expected = raref.flow * (t_test - raref.t_start) / (v_mid - raref.v_start)
                r_actual = sorption.retardation(c_mid)
                np.testing.assert_allclose(r_actual, r_expected, rtol=1e-12, err_msg="Self-similar solution violated")


class TestEntropyAndPhysics:
    """Test physical correctness."""

    def test_all_shocks_satisfy_entropy(self):
        """All shocks must satisfy Lax entropy condition."""
        dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        # Multiple concentration changes
        cin = np.array([0.1] * 4 + [10.0] * 4 + [5.0] * 4 + [15.0] * 4 + [8.0] * 4)
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=30, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
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

        # All shocks must satisfy entropy
        shocks = [w for w in structure["waves"] if isinstance(w, ShockWave)]
        for shock in shocks:
            assert shock.satisfies_entropy(), "Shock violates entropy condition"

    def test_no_negative_concentrations(self):
        """Output should never be negative."""
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:10] = 12.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=25, freq="D")
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

        valid_cout = cout[~np.isnan(cout)]
        assert np.all(valid_cout >= -1e-10), "Negative concentrations found"

    def test_output_does_not_exceed_input(self):
        """Output concentration should not exceed input."""
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:10] = 15.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=25, freq="D")
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

        valid_cout = cout[~np.isnan(cout)]
        if len(valid_cout) > 0:
            assert np.max(valid_cout) <= 15.5, "Output exceeds input concentration"


class TestConstantRetardation:
    """Test constant retardation (linear sorption)."""

    def test_constant_retardation_works(self):
        """Constant retardation should produce valid output."""
        dates = pd.date_range(start="2020-01-01", periods=15, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:10] = 10.0
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=25, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=200.0,
            retardation_factor=2.0,
        )

        valid_cout = cout[~np.isnan(cout)]
        assert len(valid_cout) > 0, "Should produce valid output"
        assert np.max(valid_cout) <= 10.5, "Output should not exceed input"
