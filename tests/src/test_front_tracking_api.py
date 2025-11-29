"""Front-tracking public API integration tests.

These tests cover the public front-tracking API functions in ``gwtransport.advection``:

- ``infiltration_to_extraction_front_tracking``
- ``infiltration_to_extraction_front_tracking_detailed``

They were originally developed as Phase 7 of the front-tracking rebuild plan but are
named independently of implementation order.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import (
    infiltration_to_extraction_front_tracking,
    infiltration_to_extraction_front_tracking_detailed,
)
from gwtransport.fronttracking.math import FreundlichSorption, compute_first_front_arrival_time
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave
from gwtransport.utils import compute_time_edges


class TestFrontTrackingAPI:
    """Integration tests for the public front-tracking API."""

    def test_basic_freundlich_sorption(self):
        """Test basic call with Freundlich sorption parameters."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)
        assert np.all(cout <= np.max(cin) * (1.0 + 1e-14))

    def test_constant_retardation(self):
        """Test with constant retardation factor (linear sorption)."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            retardation_factor=2.0,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)

    def test_zero_concentration_input(self):
        """All-zero concentration input should yield zero output."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.zeros(len(dates))
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            retardation_factor=2.0,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert np.allclose(cout, 0.0)

    def test_parameter_validation(self):
        """Error if neither retardation_factor nor Freundlich params provided."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        with pytest.raises(ValueError, match="Must provide either retardation_factor"):
            infiltration_to_extraction_front_tracking(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([500.0]),
            )

    def test_detailed_returns_consistent_structure(self):
        """Detailed API returns structure with consistent counts and sorption type."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 0.0, 10.0, 10.0, 10.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout, structures = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))

        # Check that we get one structure per pore volume
        assert len(structures) == 1
        structure = structures[0]

        waves = structure["waves"]
        n_shocks = sum(isinstance(w, ShockWave) for w in waves)
        n_rarefactions = sum(isinstance(w, RarefactionWave) for w in waves)
        n_characteristics = sum(isinstance(w, CharacteristicWave) for w in waves)

        assert structure["n_shocks"] == n_shocks
        assert structure["n_rarefactions"] == n_rarefactions
        assert structure["n_characteristics"] == n_characteristics

        assert isinstance(structure["sorption"], FreundlichSorption)

    def test_spinup_and_first_arrival_via_api(self):
        """Spin-up: cout is zero before first arrival, t_first_arrival matches math helper."""
        dates = pd.date_range(start="2020-01-01", periods=6, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 500.0

        freundlich_k = 0.01
        freundlich_n = 2.0
        bulk_density = 1500.0
        porosity = 0.3

        sorption = FreundlichSorption(
            k_f=freundlich_k,
            n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        # Compute t_first_arrival (returns days from tedges[0])
        t_first_expected = compute_first_front_arrival_time(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            sorption=sorption,
        )

        cout_dates = pd.date_range(start=dates[0], periods=20, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout, structures = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([aquifer_pore_volume]),
            freundlich_k=freundlich_k,
            freundlich_n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        assert len(structures) == 1
        structure = structures[0]
        assert structure["t_first_arrival"] == pytest.approx(t_first_expected, rel=1e-14)

        # Verify spin-up: cout is zero for bins whose upper edge is before first arrival
        for i in range(len(cout)):
            t_upper = (cout_tedges[i + 1] - tedges[0]) / np.timedelta64(1, "D")
            if t_upper < t_first_expected:
                assert cout[i] == 0.0

    def test_api_freundlich_n_gt_1_n_greater_than_one(self):
        """API works for Freundlich with n>1 sorption (n>1)."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=2.0,  # n>1 (n>1)
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)
        assert np.all(cout <= np.max(cin) * (1.0 + 1e-14))

    def test_api_freundlich_n_lt_1_n_less_than_one(self):
        """API works for Freundlich with n<1 sorption (n<1)."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=dates,
            number_of_bins=len(dates),
        )

        cin = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
        flow = np.full(len(dates), 100.0)

        cout_dates = pd.date_range(start=dates[0], periods=10, freq="D")
        cout_tedges = compute_time_edges(
            tedges=None,
            tstart=None,
            tend=cout_dates,
            number_of_bins=len(cout_dates),
        )

        cout = infiltration_to_extraction_front_tracking(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=0.5,  # n<1 (n<1)
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert cout.shape == (len(cout_tedges) - 1,)
        assert not np.any(np.isnan(cout))
        assert np.all(cout >= 0.0)
        assert np.all(cout <= np.max(cin) * (1.0 + 1e-14))
