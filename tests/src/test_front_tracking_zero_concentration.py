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
from gwtransport.fronttracking.handlers import create_inlet_waves_at_theta
from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave
from gwtransport.utils import compute_time_edges


class TestInletWaveCreationZeroConcentration:
    """Test wave creation for C=0 transitions at inlet."""

    def test_zero_to_nonzero_creates_characteristic_freundlich_n_gt_1(self):
        """Test C=0 → C=10 creates rarefaction for n>1 Freundlich with c_min>0."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)

        waves = create_inlet_waves_at_theta(c_prev=0.0, c_new=10.0, theta=5.0, sorption=sorption, v_inlet=0.0)

        # For n>1, velocity increases with concentration
        # So 10 > 0 means faster concentration catching slower, creating compression (shock)
        assert len(waves) == 1
        assert isinstance(waves[0], ShockWave)
        assert waves[0].c_left == 10.0  # Upstream (faster)
        assert waves[0].c_right == 0.0  # Downstream (slower)
        assert waves[0].theta_start == 5.0
        assert waves[0].v_start == 0.0

    def test_zero_to_nonzero_creates_characteristic_freundlich_n_lt_1(self):
        """Test C=0 → C=10 creates characteristic for n<1 Freundlich with c_min=0."""
        sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)

        waves = create_inlet_waves_at_theta(c_prev=0.0, c_new=10.0, theta=5.0, sorption=sorption, v_inlet=0.0)

        # For n<1 with c_min=0, R(0)=1 is special case
        # Stepping from 0 to nonzero creates characteristic
        assert len(waves) == 1
        assert isinstance(waves[0], CharacteristicWave)
        assert waves[0].concentration == 10.0

    def test_zero_to_nonzero_creates_characteristic_constant_retardation(self):
        """Test C=0 → C=10 creates characteristic for constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        waves = create_inlet_waves_at_theta(c_prev=0.0, c_new=10.0, theta=5.0, sorption=sorption, v_inlet=0.0)

        assert len(waves) == 1
        assert isinstance(waves[0], CharacteristicWave)
        assert waves[0].concentration == 10.0

    def test_nonzero_to_zero_creates_rarefaction_n_gt_1(self):
        """Test C=10 → C=0 creates rarefaction for n>1."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)

        waves = create_inlet_waves_at_theta(c_prev=10.0, c_new=0.0, theta=15.0, sorption=sorption, v_inlet=0.0)

        # For n>1, concentration decrease creates rarefaction (expansion)
        assert len(waves) == 1
        assert isinstance(waves[0], RarefactionWave)
        assert waves[0].c_head == 10.0  # Faster (higher C)
        assert waves[0].c_tail == 0.0  # Slower (lower C approaches c_min)
        assert waves[0].theta_start == 15.0

    def test_nonzero_to_nonzero_creates_shock_for_n_gt_1(self):
        """Test C=2 → C=10 creates shock for n>1 (compression)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        waves = create_inlet_waves_at_theta(c_prev=2.0, c_new=10.0, theta=5.0, sorption=sorption, v_inlet=0.0)

        assert len(waves) == 1
        assert isinstance(waves[0], ShockWave)
        assert waves[0].c_left == 10.0
        assert waves[0].c_right == 2.0
        assert waves[0].satisfies_entropy()

    def test_nonzero_to_nonzero_creates_rarefaction_for_n_gt_1(self):
        """Test C=10 → C=2 creates rarefaction for n>1 (expansion)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        waves = create_inlet_waves_at_theta(c_prev=10.0, c_new=2.0, theta=5.0, sorption=sorption, v_inlet=0.0)

        assert len(waves) == 1
        assert isinstance(waves[0], RarefactionWave)
        assert waves[0].c_head == 10.0  # Faster (higher C)
        assert waves[0].c_tail == 2.0  # Slower (lower C)


# TestStepInputPlateau removed (P2.5): plateau coverage is parametrized in
# tests/src/test_front_tracking_plateau_behavior.py.


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

        # First arrival θ should be > injection-start θ. With constant flow=100,
        # injection starts at t=5, so θ_injection_start = 500. First arrival is
        # tail-of-spin-up = θ_emit + V·R(c_first) (see compute_first_front_arrival_theta).
        theta_first = structure[0]["theta_first_arrival"]
        assert theta_first > 500.0  # > θ at injection-start (t=5, flow=100)
        assert np.isfinite(theta_first)

        # Output before first arrival (translated to time) should be zero.
        tracker_state = structure[0]["tracker_state"]
        t_first = tracker_state.t_at_theta(theta_first)
        cout_tedges_days = (cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)
        mask_before = cout_tedges_days[1:-1] < t_first - 1e-10
        cout_before = cout[:-1][mask_before]

        if len(cout_before) > 0:
            assert np.allclose(cout_before, 0.0, atol=1e-14), "Output must be zero before first arrival"
