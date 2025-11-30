"""
Unit tests for front tracking mathematical foundation.

Tests verify exact analytical computations with machine precision (rtol=1e-14).

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    characteristic_position,
    characteristic_velocity,
    compute_first_front_arrival_time,
)


class TestFreundlichSorption:
    """Test FreundlichSorption class."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        assert sorption.k_f == 0.01
        assert sorption.n == 2.0
        assert sorption.bulk_density == 1500.0
        assert sorption.porosity == 0.3

    def test_initialization_invalid_kf(self):
        """Test that negative k_f raises error."""
        with pytest.raises(ValueError, match="k_f must be positive"):
            FreundlichSorption(k_f=-0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    def test_initialization_invalid_n_zero(self):
        """Test that n=0 raises error."""
        with pytest.raises(ValueError, match="n must be positive"):
            FreundlichSorption(k_f=0.01, n=0.0, bulk_density=1500.0, porosity=0.3)

    def test_initialization_invalid_n_one(self):
        """Test that n=1 raises error."""
        with pytest.raises(ValueError, match="not supported"):
            FreundlichSorption(k_f=0.01, n=1.0, bulk_density=1500.0, porosity=0.3)

    def test_initialization_invalid_bulk_density(self):
        """Test that negative bulk_density raises error."""
        with pytest.raises(ValueError, match="bulk_density must be positive"):
            FreundlichSorption(k_f=0.01, n=2.0, bulk_density=-1500.0, porosity=0.3)

    def test_initialization_invalid_porosity(self):
        """Test that invalid porosity raises error."""
        with pytest.raises(ValueError, match="porosity must be in"):
            FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=1.5)

    def test_retardation_zero_concentration(self):
        """Test R(0) behavior depends on n and c_min."""
        # For n<1 (lower C travels faster) with c_min=0, R(0) = 1
        sorption_unfav = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)
        r = sorption_unfav.retardation(0.0)
        assert r == 1.0

        # For n>1 (higher C travels faster) with c_min>0, R(c_min) is used instead
        sorption_fav = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)
        r = sorption_fav.retardation(0.0)
        # Should return R(c_min), which is > 1 for n>1
        assert r > 1.0

    def test_retardation_positive_concentration_n_greater_1(self):
        """Test R(C) > 1 for C > 0 when n > 1."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        r = sorption.retardation(5.0)
        assert r > 1.0

    def test_retardation_decreases_with_concentration_n_greater_1(self):
        """Test that R decreases with C for n > 1 (n>1)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        r1 = sorption.retardation(1.0)
        r2 = sorption.retardation(10.0)
        assert r1 > r2, "R should decrease with increasing C for n > 1"

    def test_retardation_increases_with_concentration_n_less_1(self):
        """Test that R increases with C for n < 1 (n<1)."""
        sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3)
        r1 = sorption.retardation(1.0)
        r2 = sorption.retardation(10.0)
        assert r1 < r2, "R should increase with increasing C for n < 1"

    def test_total_concentration_zero(self):
        """Test C_total(0) behavior depends on n and c_min."""
        # For n<1 with c_min=0, C_total(0) = 0
        sorption_unfav = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)
        c_total = sorption_unfav.total_concentration(0.0)
        assert c_total == 0.0

        # For n>1 with c_min>0, C_total(c_min) is used
        sorption_fav = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)
        c_total = sorption_fav.total_concentration(0.0)
        # Should be small but positive
        assert c_total > 0.0

    def test_total_concentration_positive(self):
        """Test C_total > C for C > 0."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0
        c_total = sorption.total_concentration(c)
        assert c_total > c

    def test_retardation_roundtrip_machine_precision(self):
        """Test C → R → C roundtrip with machine precision."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        test_concentrations = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

        for c in test_concentrations:
            r = sorption.retardation(c)
            c_back = sorption.concentration_from_retardation(r)
            assert np.isclose(c, c_back, rtol=1e-14), f"Roundtrip failed for C={c}: {c} → {r} → {c_back}"  # type: ignore[no-matching-overload]

    def test_concentration_from_retardation_r_equals_one(self):
        """Test that R=1 gives C=c_min."""
        # For n>1 with c_min>0
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)
        c = sorption.concentration_from_retardation(1.0)
        assert c == sorption.c_min

        # For n<1 with c_min=0
        sorption_unfav = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)
        c = sorption_unfav.concentration_from_retardation(1.0)
        assert c == 0.0

    def test_concentration_from_retardation_r_less_one(self):
        """Test that R<1 gives C=c_min (physical constraint)."""
        # For n>1 with c_min>0
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)
        c = sorption.concentration_from_retardation(0.5)
        assert c == sorption.c_min

        # For n<1 with c_min=0
        sorption_unfav = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)
        c = sorption_unfav.concentration_from_retardation(0.5)
        assert c == 0.0

    def test_shock_velocity_rankine_hugoniot(self):
        """Test shock velocity satisfies Rankine-Hugoniot exactly."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        c_right = 2.0
        flow = 100.0

        v_shock = sorption.shock_velocity(c_left, c_right, flow)

        # Verify Rankine-Hugoniot
        flux_left = flow * c_left
        flux_right = flow * c_right
        c_total_left = sorption.total_concentration(c_left)
        c_total_right = sorption.total_concentration(c_right)

        v_shock_expected = (flux_right - flux_left) / (c_total_right - c_total_left)

        assert np.isclose(v_shock, v_shock_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_shock_velocity_equal_concentrations(self):
        """Test shock velocity when c_left = c_right (degenerate case)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0
        flow = 100.0

        v_shock = sorption.shock_velocity(c, c, flow)

        # Should return characteristic velocity
        v_char = flow / sorption.retardation(c)
        assert np.isclose(v_shock, v_char, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_entropy_condition_physical_shock_n_greater_1(self):
        """Test entropy condition for physical compression shock (n > 1)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        c_right = 2.0
        flow = 100.0

        v_shock = sorption.shock_velocity(c_left, c_right, flow)
        satisfies = sorption.check_entropy_condition(c_left, c_right, v_shock, flow)

        assert satisfies, "Physical compression shock should satisfy entropy"

    def test_entropy_condition_unphysical_shock_n_greater_1(self):
        """Test entropy condition for unphysical expansion shock (n > 1)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 2.0  # Lower concentration on left
        c_right = 10.0  # Higher concentration on right
        flow = 100.0

        v_shock = sorption.shock_velocity(c_left, c_right, flow)
        satisfies = sorption.check_entropy_condition(c_left, c_right, v_shock, flow)

        assert not satisfies, "Unphysical expansion shock should violate entropy"

    def test_entropy_condition_physical_shock_n_less_1(self):
        """Test entropy condition for physical shock with n < 1."""
        sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3)
        # For n < 1, physical shocks have c_left < c_right
        c_left = 2.0
        c_right = 10.0
        flow = 100.0

        v_shock = sorption.shock_velocity(c_left, c_right, flow)
        satisfies = sorption.check_entropy_condition(c_left, c_right, v_shock, flow)

        assert satisfies, "Physical shock for n<1 should satisfy entropy"


class TestConstantRetardation:
    """Test ConstantRetardation class."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        assert sorption.retardation_factor == 2.0

    def test_initialization_conservative_tracer(self):
        """Test R = 1.0 (conservative tracer)."""
        sorption = ConstantRetardation(retardation_factor=1.0)
        assert sorption.retardation_factor == 1.0

    def test_initialization_invalid_retardation(self):
        """Test that R < 1 raises error."""
        with pytest.raises(ValueError, match="retardation_factor must be"):
            ConstantRetardation(retardation_factor=0.5)

    def test_retardation_independent_of_concentration(self):
        """Test that R is constant for all concentrations."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        r1 = sorption.retardation(0.0)
        r2 = sorption.retardation(5.0)
        r3 = sorption.retardation(100.0)
        assert r1 == r2 == r3 == 2.0

    def test_total_concentration_linear(self):
        """Test that C_total = C * R for constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 5.0
        c_total = sorption.total_concentration(c)
        assert np.isclose(c_total, c * 2.0, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_concentration_from_retardation_raises_error(self):
        """Test that inversion is not supported."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        with pytest.raises(NotImplementedError, match="not applicable"):
            sorption.concentration_from_retardation(2.0)

    def test_shock_velocity_constant(self):
        """Test shock velocity equals characteristic velocity."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        flow = 100.0
        v_shock = sorption.shock_velocity(c_left=10.0, c_right=2.0, flow=flow)
        v_expected = flow / 2.0
        assert np.isclose(v_shock, v_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_entropy_condition_always_true(self):
        """Test that entropy condition is always satisfied."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        flow = 100.0
        v_shock = sorption.shock_velocity(10.0, 2.0, flow)
        satisfies = sorption.check_entropy_condition(10.0, 2.0, v_shock, flow)
        assert satisfies


class TestCharacteristicFunctions:
    """Test characteristic velocity and position functions."""

    def test_characteristic_velocity_freundlich(self):
        """Test characteristic velocity computation."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0
        flow = 100.0

        v = characteristic_velocity(c, flow, sorption)
        v_expected = flow / sorption.retardation(c)

        assert np.isclose(v, v_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_characteristic_velocity_constant(self):
        """Test characteristic velocity with constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 5.0
        flow = 100.0

        v = characteristic_velocity(c, flow, sorption)
        v_expected = flow / 2.0

        assert np.isclose(v, v_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_characteristic_position_linear_propagation(self):
        """Test that characteristic propagates linearly."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 5.0
        flow = 100.0
        t_start = 0.0
        v_start = 0.0

        # Test at multiple times
        for t in [1.0, 5.0, 10.0]:
            v_pos = characteristic_position(c, flow, sorption, t_start, v_start, t)
            v_expected = (flow / 2.0) * t
            assert np.isclose(v_pos, v_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_characteristic_position_before_start(self):
        """Test that position is None for t < t_start."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        v_pos = characteristic_position(c=5.0, flow=100.0, sorption=sorption, t_start=10.0, v_start=0.0, t=5.0)
        assert v_pos is None

    def test_characteristic_position_nonzero_start(self):
        """Test propagation from non-zero starting position."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 5.0
        flow = 100.0
        t_start = 5.0
        v_start = 100.0
        t = 15.0

        v_pos = characteristic_position(c, flow, sorption, t_start, v_start, t)
        velocity = flow / 2.0
        v_expected = v_start + velocity * (t - t_start)

        assert np.isclose(v_pos, v_expected, rtol=1e-14)  # type: ignore[no-matching-overload]


class TestFirstArrivalTime:
    """Test first arrival time computation."""

    def test_first_arrival_constant_flow_constant_retardation(self):
        """Test first arrival with constant flow and retardation."""

        cin = np.array([0.0, 10.0, 10.0])
        flow = np.array([100.0, 100.0, 100.0])
        tedges = pd.date_range("2020-01-01", periods=4, freq="10D")  # [0, 10, 20, 30] days
        aquifer_pore_volume = 500.0
        sorption = ConstantRetardation(retardation_factor=2.0)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # Expected: time from tedges[0] when first concentration reaches outlet
        # First non-zero at index 1 (day 10), travels for 500*2/100 = 10 days
        # Arrives at day 10 + 10 = 20 days from tedges[0]
        t_expected = 20.0

        assert np.isclose(t_first, t_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_first_arrival_starts_at_zero(self):
        """Test first arrival when concentration starts at t=0."""

        cin = np.array([10.0, 10.0])
        flow = np.array([100.0, 100.0])
        tedges = pd.date_range("2020-01-01", periods=3, freq="10D")  # [0, 10, 20] days
        aquifer_pore_volume = 500.0
        sorption = ConstantRetardation(retardation_factor=2.0)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # Expected: concentration starts at t=0 (tedges[0]), travels for 500*2/100 = 10 days
        # Arrives at 0 + 10 = 10 days from tedges[0]
        t_expected = 10.0

        assert np.isclose(t_first, t_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_first_arrival_no_concentration(self):
        """Test that all-zero concentration returns infinity."""

        cin = np.array([0.0, 0.0, 0.0])
        flow = np.array([100.0, 100.0, 100.0])
        tedges = pd.date_range("2020-01-01", periods=4, freq="10D")  # [0, 10, 20, 30] days
        aquifer_pore_volume = 500.0
        sorption = ConstantRetardation(retardation_factor=2.0)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        assert t_first == np.inf

    def test_first_arrival_variable_flow(self):
        """Test first arrival with variable flow."""

        cin = np.array([0.0, 10.0, 10.0])
        flow = np.array([100.0, 50.0, 200.0])  # Variable flow
        tedges = pd.date_range("2020-01-01", periods=4, freq="10D")  # [0, 10, 20, 30] days
        aquifer_pore_volume = 500.0
        sorption = ConstantRetardation(retardation_factor=2.0)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # Target volume: 500 * 2 = 1000 m³
        # First non-zero at index 1 (day 10)
        # From day 10 to day 20: flow=50, volume = 50*10 = 500 m³
        # From day 20 onward: flow=200, remaining = 500 m³, time = 500/200 = 2.5 days
        # Total: 20.0 + 2.5 = 22.5 days from tedges[0]
        t_expected = 22.5

        assert np.isclose(t_first, t_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_first_arrival_freundlich_sorption(self):
        """Test first arrival with Freundlich sorption."""

        cin = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        flow = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        tedges = pd.date_range("2020-01-01", periods=8, freq="10D")  # [0, 10, 20, ...] days
        aquifer_pore_volume = 500.0
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # Compute retardation for C=10
        r = sorption.retardation(10.0)
        # First non-zero at index 1 (day 10), travels for 500*r/100 days
        # Expected: 10.0 + 500.0 * r / 100.0 days from tedges[0]
        t_expected = 10.0 + 500.0 * r / 100.0

        assert np.isclose(t_first, t_expected, rtol=1e-14)  # type: ignore[no-matching-overload]

    def test_first_arrival_insufficient_flow_history(self):
        """Test that insufficient flow history returns infinity."""

        cin = np.array([0.0, 10.0])
        flow = np.array([10.0, 10.0])  # Very low flow
        tedges = pd.date_range("2020-01-01", periods=3, freq="10D")  # [0, 10, 20] days
        aquifer_pore_volume = 10000.0  # Very large volume
        sorption = ConstantRetardation(retardation_factor=2.0)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # Target: 10000 * 2 = 20000 m³
        # Available from day 10 to day 20: 10 * 10 = 100 m³
        # Not enough flow history
        assert t_first == np.inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
