"""
Phase 7 Smoke Tests for Front Tracking API Integration.

These tests verify that the new API functions using Phase 1-6 modules work correctly.
"""

import numpy as np
import pandas as pd

from gwtransport.advection import (
    infiltration_to_extraction_front_tracking,
    infiltration_to_extraction_front_tracking_detailed,
)
from gwtransport.utils import compute_time_edges


class TestPhase7APIIntegration:
    """Smoke tests for the new front tracking API."""

    def test_basic_freundlich_sorption(self):
        """Test basic call with Freundlich sorption parameters."""
        # Simple step input
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Call with Freundlich parameters
        cout = infiltration_to_extraction_front_tracking(
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

        # Basic checks
        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)
        assert np.all(cout <= 10.0 * 1.01)  # Allow small numerical tolerance

    def test_constant_retardation(self):
        """Test with constant retardation factor (linear sorption)."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Call with retardation_factor
        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=500.0,
            retardation_factor=2.0,
        )

        # Basic checks
        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)

    def test_detailed_returns_structure(self):
        """Test detailed function returns both concentration and structure."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Call detailed function
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

        # Check concentration output
        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))

        # Check structure contains expected keys
        assert "waves" in structure
        assert "events" in structure
        assert "t_first_arrival" in structure
        assert "n_events" in structure
        assert "n_shocks" in structure
        assert "n_rarefactions" in structure
        assert "n_characteristics" in structure
        assert "final_time" in structure
        assert "sorption" in structure
        assert "tracker_state" in structure

    def test_zero_concentration_input(self):
        """Test with all-zero concentration input."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=500.0,
            retardation_factor=2.0,
        )

        # Should return all zeros
        assert cout.shape == (len(cout_tedges) - 1,)
        assert np.allclose(cout, 0.0)

    def test_parameter_validation(self):
        """Test that parameter validation works."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Should raise error if neither retardation_factor nor Freundlich params provided
        try:
            infiltration_to_extraction_front_tracking(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volume=500.0,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Must provide either retardation_factor" in str(e)
