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
    LangmuirSorption,
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
        """R(0) behavior depends on n and c_min."""
        # n<1 with c_min=0: R(0)=1 (no sorption at zero, physically correct).
        sorption_unfav = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)
        assert sorption_unfav.retardation(0.0) == 1.0

        # n>1 with c_min>0: c clamps to c_min, R returns R(c_min) > 1.
        sorption_fav = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)
        assert sorption_fav.retardation(0.0) > 1.0

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
        """C_total(0) behavior depends on n and c_min."""
        # n<1 with c_min=0: C_total(0) = 0.
        sorption_unfav = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)
        assert sorption_unfav.total_concentration(0.0) == 0.0

        # n>1 with c_min>0: clamps to C_total(c_min) which is small but positive (P1.4 Option A).
        sorption_fav = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)
        assert sorption_fav.total_concentration(0.0) > 0.0

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
            assert np.isclose(c, c_back, rtol=1e-14), f"Roundtrip failed for C={c}: {c} → {r} → {c_back}"

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

        assert np.isclose(v_shock, v_shock_expected, rtol=1e-14)

    def test_shock_velocity_equal_concentrations(self):
        """Test shock velocity when c_left = c_right (degenerate case)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0
        flow = 100.0

        v_shock = sorption.shock_velocity(c, c, flow)

        # Should return characteristic velocity
        v_char = flow / sorption.retardation(c)
        assert np.isclose(v_shock, v_char, rtol=1e-14)

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


class TestLangmuirSorption:
    """Test LangmuirSorption class."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        assert sorption.s_max == 0.1
        assert sorption.k_l == 5.0
        assert sorption.bulk_density == 1500.0
        assert sorption.porosity == 0.3

    def test_initialization_invalid_s_max(self):
        """Test that non-positive s_max raises error."""
        with pytest.raises(ValueError, match="s_max must be positive"):
            LangmuirSorption(s_max=-0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)

    def test_initialization_invalid_k_l(self):
        """Test that non-positive k_l raises error."""
        with pytest.raises(ValueError, match="k_l must be positive"):
            LangmuirSorption(s_max=0.1, k_l=0.0, bulk_density=1500.0, porosity=0.3)

    def test_initialization_invalid_bulk_density(self):
        """Test that non-positive bulk_density raises error."""
        with pytest.raises(ValueError, match="bulk_density must be positive"):
            LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=-1500.0, porosity=0.3)

    def test_initialization_invalid_porosity(self):
        """Test that invalid porosity raises error."""
        with pytest.raises(ValueError, match="porosity must be in"):
            LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=1.5)

    def test_retardation_zero_concentration_finite(self):
        """Test R(0) is finite — key difference from Freundlich n>1."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        r0 = sorption.retardation(0.0)
        r0_expected = 1.0 + sorption.bulk_density * sorption.s_max / (sorption.porosity * sorption.k_l)
        assert np.isclose(r0, r0_expected, rtol=1e-14)
        assert np.isfinite(r0)

    def test_retardation_positive_concentration(self):
        """Test R(C) > 1 for C > 0."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        assert sorption.retardation(5.0) > 1.0

    def test_retardation_decreases_with_concentration(self):
        """Test R decreases with C (always favorable)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        r1 = sorption.retardation(1.0)
        r2 = sorption.retardation(10.0)
        assert r1 > r2, "R should decrease with increasing C for Langmuir"

    def test_retardation_approaches_one_at_high_concentration(self):
        """Test R → 1 as C → ∞ (all sites saturated)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        r_high = sorption.retardation(1e8)
        assert np.isclose(r_high, 1.0, atol=1e-8)

    def test_retardation_array_input(self):
        """Test retardation with array input."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c = np.array([0.0, 1.0, 5.0, 10.0, 100.0])
        r = sorption.retardation(c)
        assert isinstance(r, np.ndarray)
        assert len(r) == 5
        assert np.all(r >= 1.0)
        # Verify monotonically decreasing
        assert np.all(np.diff(r) < 0)

    def test_total_concentration_zero(self):
        """Test C_total(0) = 0."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        assert sorption.total_concentration(0.0) == 0.0

    def test_total_concentration_positive(self):
        """Test C_total > C for C > 0."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0
        c_total = sorption.total_concentration(c)
        assert c_total > c

    def test_total_concentration_saturation_limit(self):
        """Test C_total → C + rho_b*s_max/n_por as C → ∞."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c_large = 1e10
        c_total = sorption.total_concentration(c_large)
        sorbed_max = sorption.bulk_density * sorption.s_max / sorption.porosity
        assert np.isclose(c_total, c_large + sorbed_max, rtol=1e-8)

    def test_retardation_roundtrip_machine_precision(self):
        """Test C → R → C roundtrip with machine precision."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)

        test_concentrations = [0.01, 0.1, 1.0, 5.0, 10.0, 100.0, 1000.0]

        for c in test_concentrations:
            r = sorption.retardation(c)
            c_back = sorption.concentration_from_retardation(r)
            assert np.isclose(c, c_back, rtol=1e-14), f"Roundtrip failed for C={c}: {c} → {r} → {c_back}"

    def test_concentration_from_retardation_r_equals_one(self):
        """Test that R=1 gives C=0 (sites fully saturated)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c = sorption.concentration_from_retardation(1.0)
        assert c == 0.0

    def test_concentration_from_retardation_r_at_maximum(self):
        """Test that R=R(0) gives C=0."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        r_max = sorption.retardation(0.0)
        c = sorption.concentration_from_retardation(r_max)
        assert np.isclose(c, 0.0, atol=1e-14)

    def test_shock_velocity_rankine_hugoniot(self):
        """Test shock velocity satisfies Rankine-Hugoniot exactly."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
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

        assert np.isclose(v_shock, v_shock_expected, rtol=1e-14)

    def test_shock_velocity_equal_concentrations(self):
        """Test shock velocity when c_left = c_right (degenerate case)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0
        flow = 100.0

        v_shock = sorption.shock_velocity(c, c, flow)
        v_char = flow / sorption.retardation(c)
        assert np.isclose(v_shock, v_char, rtol=1e-14)

    def test_entropy_condition_physical_shock(self):
        """Test entropy for physical compression shock (c_left > c_right)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        c_right = 2.0
        flow = 100.0

        v_shock = sorption.shock_velocity(c_left, c_right, flow)
        assert sorption.check_entropy_condition(c_left, c_right, v_shock, flow)

    def test_entropy_condition_unphysical_shock(self):
        """Test entropy for unphysical expansion shock (c_left < c_right)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c_left = 2.0
        c_right = 10.0
        flow = 100.0

        v_shock = sorption.shock_velocity(c_left, c_right, flow)
        assert not sorption.check_entropy_condition(c_left, c_right, v_shock, flow)

    def test_first_arrival_langmuir_sorption(self):
        """Test first arrival time with Langmuir sorption."""
        cin = np.array([0.0] + [10.0] * 19)
        flow = np.array([100.0] * 20)
        tedges = pd.date_range("2020-01-01", periods=21, freq="10D")
        aquifer_pore_volume = 500.0
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # Langmuir is concave (favorable) like Freundlich n>1: the 0->c step creates
        # a Rankine-Hugoniot shock, so the analytic arrival uses shock velocity
        # s = flow * c / (C_tot(c) - C_tot(0)) = flow * c / C_tot(c).
        c_total = sorption.total_concentration(10.0)
        t_expected = 10.0 + aquifer_pore_volume * c_total / (10.0 * 100.0)

        assert np.isclose(t_first, t_expected, rtol=1e-14)


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
        assert np.isclose(c_total, c * 2.0, rtol=1e-14)

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
        assert np.isclose(v_shock, v_expected, rtol=1e-14)

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

        assert np.isclose(v, v_expected, rtol=1e-14)

    def test_characteristic_velocity_constant(self):
        """Test characteristic velocity with constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 5.0
        flow = 100.0

        v = characteristic_velocity(c, flow, sorption)
        v_expected = flow / 2.0

        assert np.isclose(v, v_expected, rtol=1e-14)

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
            assert v_pos is not None
            v_expected = (flow / 2.0) * t
            assert np.isclose(v_pos, v_expected, rtol=1e-14)

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
        assert v_pos is not None
        velocity = flow / 2.0
        v_expected = v_start + velocity * (t - t_start)

        assert np.isclose(v_pos, v_expected, rtol=1e-14)


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

        assert np.isclose(t_first, t_expected, rtol=1e-14)

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

        assert np.isclose(t_first, t_expected, rtol=1e-14)

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

        assert np.isclose(t_first, t_expected, rtol=1e-14)

    def test_first_arrival_freundlich_sorption(self):
        """First arrival for Freundlich n>1 uses Rankine-Hugoniot shock velocity (P1.3)."""
        n_bins = 20
        cin = np.array([0.0] + [10.0] * (n_bins - 1))
        flow = np.array([100.0] * n_bins)
        tedges = pd.date_range("2020-01-01", periods=n_bins + 1, freq="10D")
        aquifer_pore_volume = 500.0
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # n>1: solver emits a R-H shock for 0->c step. Shock velocity uses C_tot:
        # arrival_from_inlet = V * C_tot(c) / (c * flow).
        c_total = sorption.total_concentration(10.0)
        t_expected = 10.0 + aquifer_pore_volume * c_total / (10.0 * 100.0)

        assert np.isclose(t_first, t_expected, rtol=1e-14)

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
