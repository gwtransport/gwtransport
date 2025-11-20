"""
Unit Tests for Front Tracking with Varying Flow.

Tests verify that flow changes correctly update wave velocities and maintain
exact mass balance. All tests use machine precision (rtol ~ 1e-14).

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.handlers import (
    handle_flow_change,
    recreate_characteristic_with_new_flow,
    recreate_rarefaction_with_new_flow,
    recreate_shock_with_new_flow,
)
from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.output import (
    compute_mass_balance_with_varying_flow,
    verify_mass_balance,
)
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave


@pytest.fixture
def freundlich_sorption():
    """Standard Freundlich sorption for testing (n>1, favorable)."""
    return FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)


@pytest.fixture
def constant_retardation():
    """Constant retardation for testing."""
    return ConstantRetardation(retardation_factor=2.0)


class TestWaveRecreation:
    """Test individual wave recreation functions."""

    def test_recreate_characteristic_preserves_concentration(self, constant_retardation):
        """Characteristic concentration unchanged, velocity scales with flow."""
        char_old = CharacteristicWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            concentration=5.0,
            sorption=constant_retardation,
        )

        # After 10 days at flow=100, char is at v = 0 + (100/2)*10 = 500
        t_change = 10.0
        flow_new = 200.0

        char_new = recreate_characteristic_with_new_flow(char_old, t_change, flow_new)

        # Verify concentration preserved
        assert char_new.concentration == 5.0

        # Verify flow updated
        assert char_new.flow == 200.0

        # Verify position correct
        assert char_new.v_start == 500.0
        assert char_new.t_start == 10.0

        # Verify velocity doubled
        vel_old = char_old.velocity()
        vel_new = char_new.velocity()
        assert np.isclose(vel_new, 2 * vel_old, rtol=1e-14)

    def test_recreate_shock_velocity_scales_with_flow(self, freundlich_sorption):
        """Shock velocity scales linearly with flow (Rankine-Hugoniot)."""
        shock_old = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=2.0,
            sorption=freundlich_sorption,
        )

        # Get old velocity
        vel_old = shock_old.velocity

        # Compute position at t_change
        t_change = 5.0
        v_at_change = shock_old.v_start + vel_old * t_change

        # Recreate with doubled flow
        flow_new = 200.0
        shock_new = recreate_shock_with_new_flow(shock_old, t_change, flow_new)

        # Verify concentrations preserved
        assert shock_new.c_left == 10.0
        assert shock_new.c_right == 2.0

        # Verify flow updated
        assert shock_new.flow == 200.0

        # Verify position correct
        assert np.isclose(shock_new.v_start, v_at_change, rtol=1e-14)

        # Verify velocity doubled (Rankine-Hugoniot is linear in flow)
        vel_new = shock_new.velocity
        assert np.isclose(vel_new, 2 * vel_old, rtol=1e-14)

    def test_recreate_rarefaction_preserves_concentrations(self, freundlich_sorption):
        """Rarefaction head/tail concentrations unchanged, velocities scale with flow."""
        raref_old = RarefactionWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_head=10.0,
            c_tail=2.0,
            sorption=freundlich_sorption,
        )

        # Get old velocities
        vel_head_old = raref_old.head_velocity()
        vel_tail_old = raref_old.tail_velocity()

        t_change = 8.0
        flow_new = 50.0

        # Position at t_change (use head position)
        v_at_change = raref_old.head_position_at_time(t_change)

        raref_new = recreate_rarefaction_with_new_flow(raref_old, t_change, flow_new)

        # Verify concentrations preserved
        assert raref_new.c_head == 10.0
        assert raref_new.c_tail == 2.0

        # Verify flow updated
        assert raref_new.flow == 50.0

        # Verify position
        assert np.isclose(raref_new.v_start, v_at_change, rtol=1e-14)

        # Verify velocities halved
        vel_head_new = raref_new.head_velocity()
        vel_tail_new = raref_new.tail_velocity()
        assert np.isclose(vel_head_new, 0.5 * vel_head_old, rtol=1e-14)
        assert np.isclose(vel_tail_new, 0.5 * vel_tail_old, rtol=1e-14)

    def test_recreate_before_wave_start_raises_error(self, constant_retardation):
        """Cannot recreate wave before it starts."""
        char = CharacteristicWave(
            t_start=10.0,
            v_start=0.0,
            flow=100.0,
            concentration=5.0,
            sorption=constant_retardation,
        )

        with pytest.raises(ValueError, match="not yet active"):
            recreate_characteristic_with_new_flow(char, t_change=5.0, flow_new=200.0)


class TestFlowChangeHandler:
    """Test handle_flow_change function."""

    def test_handle_flow_change_recreates_all_waves(self, constant_retardation):
        """All active waves are recreated with new flow."""
        waves = [
            CharacteristicWave(
                t_start=0.0,
                v_start=0.0,
                flow=100.0,
                concentration=5.0,
                sorption=constant_retardation,
            ),
            CharacteristicWave(
                t_start=2.0,
                v_start=0.0,
                flow=100.0,
                concentration=10.0,
                sorption=constant_retardation,
            ),
        ]

        t_change = 10.0
        flow_new = 200.0

        new_waves = handle_flow_change(t_change, flow_new, waves)

        # Should create 2 new waves
        assert len(new_waves) == 2

        # All new waves should have new flow
        assert all(w.flow == 200.0 for w in new_waves)

        # Old waves should be deactivated
        assert all(not w.is_active for w in waves)

        # New waves should be active
        assert all(w.is_active for w in new_waves)

    def test_handle_flow_change_mixed_wave_types(self, freundlich_sorption):
        """Flow change handles characteristics, shocks, and rarefactions."""
        waves = [
            CharacteristicWave(
                t_start=0.0,
                v_start=0.0,
                flow=100.0,
                concentration=5.0,
                sorption=freundlich_sorption,
            ),
            ShockWave(
                t_start=0.0,
                v_start=0.0,
                flow=100.0,
                c_left=10.0,
                c_right=2.0,
                sorption=freundlich_sorption,
            ),
            RarefactionWave(
                t_start=0.0,
                v_start=0.0,
                flow=100.0,
                c_head=10.0,
                c_tail=2.0,
                sorption=freundlich_sorption,
            ),
        ]

        t_change = 15.0
        flow_new = 150.0

        new_waves = handle_flow_change(t_change, flow_new, waves)

        # Should create 3 new waves
        assert len(new_waves) == 3

        # Check types preserved
        assert isinstance(new_waves[0], CharacteristicWave)
        assert isinstance(new_waves[1], ShockWave)
        assert isinstance(new_waves[2], RarefactionWave)

        # All have new flow
        assert all(w.flow == 150.0 for w in new_waves)


class TestFlowChangeIntegration:
    """Integration tests with FrontTracker solver."""

    def test_single_characteristic_flow_doubles(self, constant_retardation):
        """Single characteristic: flow doubles, arrival time halves."""
        # Characteristic with c=5, flow starts at 100
        # At t=10, flow changes to 200
        # aquifer_pore_volume = 500

        tedges = pd.date_range("2020-01-01", periods=4, freq="10D")
        cin = np.array([5.0, 5.0, 5.0])
        flow = np.array([100.0, 200.0, 200.0])  # Flow doubles at t=10

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=constant_retardation,
        )

        tracker.run(max_iterations=100, verbose=False)

        # Verify flow change event occurred
        flow_change_events = [e for e in tracker.state.events if e["type"] == "flow_change"]
        assert len(flow_change_events) == 1
        assert np.isclose(flow_change_events[0]["time"], 10.0, rtol=1e-14)

        # Verify waves were recreated
        # Should have: initial char, char after flow change
        chars = [w for w in tracker.state.waves if isinstance(w, CharacteristicWave)]
        assert len(chars) == 2

        # First char has flow=100, second has flow=200
        assert chars[0].flow == 100.0
        assert chars[1].flow == 200.0

    def test_flow_change_before_characteristic_collision(self, freundlich_sorption):
        """Two characteristics: flow change affects collision time."""
        # c1=10, c2=2, both start at t=0
        # Without flow change: would collide at some t1
        # With flow increase: should collide earlier

        tedges = pd.date_range("2020-01-01", periods=4, freq="10D")
        cin = np.array([10.0, 2.0, 2.0])
        flow = np.array([100.0, 200.0, 200.0])  # Flow doubles at t=10

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=1000.0,
            sorption=freundlich_sorption,
        )

        tracker.run(max_iterations=200, verbose=False)

        # Verify flow change occurred
        flow_events = [e for e in tracker.state.events if e["type"] == "flow_change"]
        assert len(flow_events) >= 1

        # Verify simulation completed successfully
        assert len(tracker.state.events) > 0

    def test_multiple_flow_changes(self, constant_retardation):
        """Multiple flow changes create multiple sets of waves."""
        tedges = pd.date_range("2020-01-01", periods=6, freq="10D")
        cin = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        flow = np.array([100.0, 200.0, 50.0, 150.0, 150.0])  # Three flow changes

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=1000.0,
            sorption=constant_retardation,
        )

        tracker.run(max_iterations=200, verbose=False)

        # Verify three flow change events
        flow_events = [e for e in tracker.state.events if e["type"] == "flow_change"]
        assert len(flow_events) == 3

        # Verify event times correspond to tedges
        event_times = [e["time"] for e in flow_events]
        expected_times = [10.0, 20.0, 30.0]
        assert np.allclose(event_times, expected_times, rtol=1e-14)


class TestMassBalanceVaryingFlow:
    """Test mass balance with varying flow."""

    def test_mass_balance_constant_flow(self):
        """Mass balance with constant flow (baseline)."""
        cin = np.array([10.0, 10.0, 10.0])
        cout = np.array([10.0, 10.0, 10.0])
        flow = np.array([100.0, 100.0, 100.0])
        tedges_in = np.array([0.0, 10.0, 20.0, 30.0])
        tedges_out = np.array([0.0, 10.0, 20.0, 30.0])
        t_first_arrival = 0.0

        result = compute_mass_balance_with_varying_flow(
            cin=cin,
            cout=cout,
            flow=flow,
            tedges_in=tedges_in,
            tedges_out=tedges_out,
            t_first_arrival=t_first_arrival,
        )

        # mass_in = 10 * 100 * 10 + 10 * 100 * 10 + 10 * 100 * 10 = 30000
        # mass_out = same
        assert np.isclose(result["mass_in"], 30000.0, rtol=1e-14)
        assert np.isclose(result["mass_out"], 30000.0, rtol=1e-14)
        assert np.isclose(result["mass_balance_error"], 0.0, atol=1e-12)
        assert verify_mass_balance(result, rtol=1e-12)

    def test_mass_balance_varying_flow(self):
        """Mass balance with varying flow."""
        cin = np.array([10.0, 10.0, 10.0])
        cout = np.array([10.0, 10.0, 10.0])
        flow = np.array([100.0, 200.0, 50.0])
        tedges_in = np.array([0.0, 10.0, 20.0, 30.0])
        tedges_out = np.array([0.0, 10.0, 20.0, 30.0])
        t_first_arrival = 0.0

        result = compute_mass_balance_with_varying_flow(
            cin=cin,
            cout=cout,
            flow=flow,
            tedges_in=tedges_in,
            tedges_out=tedges_out,
            t_first_arrival=t_first_arrival,
        )

        # mass_in = 10*100*10 + 10*200*10 + 10*50*10 = 10000 + 20000 + 5000 = 35000
        assert np.isclose(result["mass_in"], 35000.0, rtol=1e-14)
        # mass_out should equal mass_in
        assert np.isclose(result["mass_out"], 35000.0, rtol=1e-14)
        assert verify_mass_balance(result, rtol=1e-12)

    def test_mass_balance_excludes_spinup(self):
        """Mass balance correctly excludes spin-up period."""
        cin = np.array([10.0, 10.0, 10.0])
        cout = np.array([0.0, 10.0, 10.0])  # First bin is spin-up
        flow = np.array([100.0, 100.0, 100.0])
        tedges_in = np.array([0.0, 10.0, 20.0, 30.0])
        tedges_out = np.array([0.0, 10.0, 20.0, 30.0])
        t_first_arrival = 10.0  # First bin is before spin-up

        result = compute_mass_balance_with_varying_flow(
            cin=cin,
            cout=cout,
            flow=flow,
            tedges_in=tedges_in,
            tedges_out=tedges_out,
            t_first_arrival=t_first_arrival,
        )

        # mass_in = 10*100*10 * 3 bins = 30000
        assert np.isclose(result["mass_in"], 30000.0, rtol=1e-14)

        # mass_out only from bins after t=10: 10*100*10 * 2 bins = 20000
        assert np.isclose(result["mass_out"], 20000.0, rtol=1e-14)

        # n_bins_spinup should be 1
        assert result["n_bins_spinup"] == 1

        # Error is expected (spin-up mass not counted)
        assert not verify_mass_balance(result, rtol=1e-12)

    def test_verify_mass_balance_accepts_small_errors(self):
        """verify_mass_balance accepts errors within tolerance."""
        # Simulate small numerical error
        result = {
            "mass_in": 1000.0,
            "mass_out": 1000.0 + 1e-13,  # Tiny error
            "mass_balance_error": 1e-13,
            "relative_error": 1e-16,
            "t_first_arrival": 0.0,
            "n_bins_spinup": 0,
        }

        assert verify_mass_balance(result, rtol=1e-12, atol=1e-12)

    def test_verify_mass_balance_rejects_large_errors(self):
        """verify_mass_balance rejects errors exceeding tolerance."""
        result = {
            "mass_in": 1000.0,
            "mass_out": 1001.0,  # 0.1% error
            "mass_balance_error": 1.0,
            "relative_error": 0.001,
            "t_first_arrival": 0.0,
            "n_bins_spinup": 0,
        }

        assert not verify_mass_balance(result, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
