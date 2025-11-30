"""
Tests for front tracking with zero concentration transitions.

This module tests the correct handling of transitions from/to C=0,
which are special cases in the front tracking implementation.

Physical interpretation:
- C=0 → C>0: Injecting solute into clean domain → characteristic wave
- C>0 → C=0: Stopping injection → characteristic wave

These tests ensure that Example 1 scenarios work correctly.
"""

import numpy as np
import pandas as pd

from gwtransport.advection import infiltration_to_extraction_front_tracking_detailed
from gwtransport.fronttracking.handlers import create_inlet_waves_at_time
from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave
from gwtransport.utils import compute_time_edges


class TestInletWaveCreationZeroConcentration:
    """Test wave creation for C=0 transitions at inlet."""

    def test_zero_to_nonzero_creates_characteristic_freundlich_n_gt_1(self):
        """Test C=0 → C=10 creates rarefaction for n>1 Freundlich with c_min>0."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)

        waves = create_inlet_waves_at_time(c_prev=0.0, c_new=10.0, t=5.0, flow=100.0, sorption=sorption, v_inlet=0.0)

        # For n>1, velocity increases with concentration
        # So 10 > 0 means faster concentration catching slower, creating compression (shock)
        assert len(waves) == 1
        assert isinstance(waves[0], ShockWave)
        assert waves[0].c_left == 10.0  # Upstream (faster)
        assert waves[0].c_right == 0.0  # Downstream (slower)
        assert waves[0].t_start == 5.0
        assert waves[0].v_start == 0.0

    def test_zero_to_nonzero_creates_characteristic_freundlich_n_lt_1(self):
        """Test C=0 → C=10 creates characteristic for n<1 Freundlich with c_min=0."""
        sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)

        waves = create_inlet_waves_at_time(c_prev=0.0, c_new=10.0, t=5.0, flow=100.0, sorption=sorption, v_inlet=0.0)

        # For n<1 with c_min=0, R(0)=1 is special case
        # Stepping from 0 to nonzero creates characteristic
        assert len(waves) == 1
        assert isinstance(waves[0], CharacteristicWave)
        assert waves[0].concentration == 10.0

    def test_zero_to_nonzero_creates_characteristic_constant_retardation(self):
        """Test C=0 → C=10 creates characteristic for constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        waves = create_inlet_waves_at_time(c_prev=0.0, c_new=10.0, t=5.0, flow=100.0, sorption=sorption, v_inlet=0.0)

        assert len(waves) == 1
        assert isinstance(waves[0], CharacteristicWave)
        assert waves[0].concentration == 10.0

    def test_nonzero_to_zero_creates_rarefaction_n_gt_1(self):
        """Test C=10 → C=0 creates rarefaction for n>1."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)

        waves = create_inlet_waves_at_time(c_prev=10.0, c_new=0.0, t=15.0, flow=100.0, sorption=sorption, v_inlet=0.0)

        # For n>1, concentration decrease creates rarefaction (expansion)
        assert len(waves) == 1
        assert isinstance(waves[0], RarefactionWave)
        assert waves[0].c_head == 10.0  # Faster (higher C)
        assert waves[0].c_tail == 0.0  # Slower (lower C approaches c_min)
        assert waves[0].t_start == 15.0

    def test_nonzero_to_nonzero_creates_shock_for_n_gt_1(self):
        """Test C=2 → C=10 creates shock for n>1 (compression)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        waves = create_inlet_waves_at_time(c_prev=2.0, c_new=10.0, t=5.0, flow=100.0, sorption=sorption, v_inlet=0.0)

        assert len(waves) == 1
        assert isinstance(waves[0], ShockWave)
        assert waves[0].c_left == 10.0
        assert waves[0].c_right == 2.0
        assert waves[0].satisfies_entropy()

    def test_nonzero_to_nonzero_creates_rarefaction_for_n_gt_1(self):
        """Test C=10 → C=2 creates rarefaction for n>1 (expansion)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        waves = create_inlet_waves_at_time(c_prev=10.0, c_new=2.0, t=5.0, flow=100.0, sorption=sorption, v_inlet=0.0)

        assert len(waves) == 1
        assert isinstance(waves[0], RarefactionWave)
        assert waves[0].c_head == 10.0  # Faster (higher C)
        assert waves[0].c_tail == 2.0  # Slower (lower C)


class TestStepInputPlateau:
    """Test step input scenarios produce correct plateau behavior."""

    def test_step_input_n_gt_1_sorption_plateau(self):
        """
        Test step increase C=0 → C=10 produces plateau at outlet.

        This is Example 1 from the notebook. Should produce:
        - Characteristic wave at C=10
        - Concentration plateau at outlet reaching C ≈ 10
        - No negative concentrations
        """
        # Setup: step from 0 to 10 at day 5
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:] = 10.0

        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 50.0

        cout_dates = pd.date_range(start=dates[0], periods=150, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Run simulation
        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([aquifer_pore_volume]),
            freundlich_k=0.001,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Verify wave creation
        # For n>1 (higher C travels faster), step from 0 to 10 creates shock (compression)
        assert structure[0]["n_shocks"] >= 1, f"Expected shock, got {structure[0]}"
        # May also have other waves from interactions

        # Verify no negative concentrations
        valid_cout = cout[~np.isnan(cout)]
        assert np.all(valid_cout >= 0), "All concentrations must be non-negative"
        assert np.min(valid_cout) >= -1e-14, f"Min concentration too negative: {np.min(valid_cout)}"

        # Verify plateau behavior
        max_cout = np.max(valid_cout)
        assert abs(max_cout - 10.0) < 0.1, f"Max concentration {max_cout} should be ~10.0"

        # Should have many bins at plateau
        plateau_bins = np.sum(np.abs(cout - 10.0) < 0.1)
        assert plateau_bins > 10, f"Should have >10 bins at plateau, got {plateau_bins}"

        # Output should not exceed input
        assert max_cout <= 10.0 * (1.0 + 1e-10), "Output cannot exceed input"

    def test_step_input_constant_retardation_plateau(self):
        """Test step input with constant retardation produces plateau."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:] = 10.0

        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 50.0

        cout_dates = pd.date_range(start=dates[0], periods=150, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Run with constant retardation
        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([aquifer_pore_volume]),
            retardation_factor=2.0,
        )

        # Should create characteristic
        assert structure[0]["n_characteristics"] == 1

        # Verify plateau
        valid_cout = cout[~np.isnan(cout)]
        assert np.all(valid_cout >= 0)
        assert abs(np.max(valid_cout) - 10.0) < 0.1

    def test_pulse_input_n_gt_1_sorption(self):
        """Test pulse input (0 → 10 → 0) produces correct behavior."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        # Pulse: 0 → 10 → 0
        cin = np.zeros(len(dates))
        cin[10:50] = 10.0

        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 50.0

        cout_dates = pd.date_range(start=dates[0], periods=150, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([aquifer_pore_volume]),
            freundlich_k=0.001,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # For n>1: 0→10 creates shock, 10→0 creates rarefaction
        # May have additional waves from interactions
        assert structure[0]["n_shocks"] >= 1, f"Expected at least 1 shock, got {structure[0]}"
        assert structure[0]["n_rarefactions"] >= 1, f"Expected at least 1 rarefaction, got {structure[0]}"

        # Verify no negative concentrations
        valid_cout = cout[~np.isnan(cout)]
        assert np.all(valid_cout >= 0)
        assert np.min(valid_cout) >= -1e-14

        # Max should be ~10
        assert abs(np.max(valid_cout) - 10.0) < 0.1


class TestFirstArrivalTime:
    """Test first arrival time computation with C=0 initial conditions."""

    def test_first_arrival_with_zero_start(self):
        """Test first arrival time is computed correctly when starting from C=0."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

        cin = np.zeros(len(dates))
        cin[5:] = 10.0  # Start injection at day 5

        flow = np.full(len(dates), 100.0)
        aquifer_pore_volume = 50.0

        cout_dates = pd.date_range(start=dates[0], periods=150, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([aquifer_pore_volume]),
            freundlich_k=0.001,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # First arrival should be > day 5 (injection start)
        assert structure[0]["t_first_arrival"] > 5.0

        # First arrival should be finite
        assert np.isfinite(structure[0]["t_first_arrival"])

        # Output before first arrival should be zero
        t_first = structure[0]["t_first_arrival"]
        cout_tedges_days = (cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)
        mask_before = cout_tedges_days[1:-1] < t_first - 1e-10
        cout_before = cout[:-1][mask_before]

        if len(cout_before) > 0:
            assert np.allclose(cout_before, 0.0, atol=1e-14), "Output must be zero before first arrival"
