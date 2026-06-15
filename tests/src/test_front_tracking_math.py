"""
Unit tests for front tracking mathematical foundation.

Tests verify exact analytical computations with machine precision (rtol=1e-14).

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import (
    BrooksCoreyConductivity,
    ConstantRetardation,
    FreundlichSorption,
    LangmuirSorption,
    VanGenuchtenMualemConductivity,
    characteristic_position,
    characteristic_speed,
    compute_first_front_arrival_theta,
)


def _theta_edges_from(flow: np.ndarray, tedges: pd.DatetimeIndex) -> np.ndarray:
    """Build θ-edges from per-bin flow and time-edges (test-only helper)."""
    dt = np.asarray((tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.asarray(flow, dtype=float) * np.diff(dt))))


def _t_at_theta(theta: float, flow: np.ndarray, tedges: pd.DatetimeIndex) -> float:
    """Translate θ→t against a piecewise-constant flow profile (test-only helper)."""
    tedges_days = np.asarray((tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)
    theta_edges = _theta_edges_from(flow, tedges)
    if theta <= theta_edges[0]:
        return float(tedges_days[0])
    if theta >= theta_edges[-1]:
        last_flow = float(flow[-1])
        return (
            float(tedges_days[-1] + (theta - theta_edges[-1]) / last_flow) if last_flow > 0 else float(tedges_days[-1])
        )
    i = int(np.searchsorted(theta_edges, theta, side="right")) - 1
    flow_i = float(flow[i])
    return float(tedges_days[i] + (theta - theta_edges[i]) / flow_i) if flow_i > 0 else float(tedges_days[i])


def compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption):
    """Test shim: θ-native helper + θ→t translation (uses test-local helpers)."""
    theta_edges = _theta_edges_from(np.asarray(flow), tedges)
    theta = compute_first_front_arrival_theta(np.asarray(cin), theta_edges, aquifer_pore_volume, sorption)
    # Past the simulation window with non-positive trailing flow → no finite t,
    # preserve the legacy `np.inf` sentinel.
    if (not np.isfinite(theta) or theta > theta_edges[-1]) and (not np.isfinite(theta) or float(flow[-1]) <= 0):
        return float(np.inf)
    return _t_at_theta(float(theta), np.asarray(flow), tedges)


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
        """C_total(0) = 0 physically: c=0 means no dissolved mass, no sorbed mass (s(0)=0)."""
        # n<1 with c_min=0: C_total(0) = 0.
        sorption_unfav = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)
        assert sorption_unfav.total_concentration(0.0) == 0.0

        # n>1 with c_min>0: c_min only clamps retardation (to keep R finite as c->0);
        # total_concentration uses c^(1/n) which is well-defined at c=0 with no
        # singularity, so C_T(0) = 0 unconditionally. Phase 2 step 4 corrects an
        # earlier design (c_min-clamped C_T) that biased Rankine-Hugoniot shock
        # speeds when c_R=0.
        sorption_fav = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=1e-12)
        assert sorption_fav.total_concentration(0.0) == 0.0

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

    # Math-layer Rankine-Hugoniot check dropped in P2.5: the canonical assertion lives at
    # tests/src/test_front_tracking_waves.py:test_velocity_rankine_hugoniot.

    def test_shock_velocity_equal_concentrations(self):
        """Shock speed when c_left = c_right (degenerate case) returns characteristic speed."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0

        v_shock = sorption.shock_speed(c, c)

        # In (V, θ), characteristic speed = 1/R(c).
        v_char = 1.0 / float(sorption.retardation(c))
        assert np.isclose(v_shock, v_char, rtol=1e-14)

    def test_entropy_condition_physical_shock_n_greater_1(self):
        """Test entropy condition for physical compression shock (n > 1)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        c_right = 2.0

        v_shock = sorption.shock_speed(c_left, c_right)
        satisfies = sorption.check_entropy_condition(c_left, c_right, v_shock)

        assert satisfies, "Physical compression shock should satisfy entropy"

    def test_entropy_condition_unphysical_shock_n_greater_1(self):
        """Test entropy condition for unphysical expansion shock (n > 1)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 2.0  # Lower concentration on left
        c_right = 10.0  # Higher concentration on right

        v_shock = sorption.shock_speed(c_left, c_right)
        satisfies = sorption.check_entropy_condition(c_left, c_right, v_shock)

        assert not satisfies, "Unphysical expansion shock should violate entropy"

    def test_entropy_condition_physical_shock_n_less_1(self):
        """Test entropy condition for physical shock with n < 1."""
        sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3)
        # For n < 1, physical shocks have c_left < c_right
        c_left = 2.0
        c_right = 10.0

        v_shock = sorption.shock_speed(c_left, c_right)
        satisfies = sorption.check_entropy_condition(c_left, c_right, v_shock)

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
        """Shock speed in (V, θ) satisfies flow-free Rankine-Hugoniot."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        c_right = 2.0

        v_shock = sorption.shock_speed(c_left, c_right)

        # In (V, θ): dV_s/dθ = (c_R - c_L) / (C_T(c_R) - C_T(c_L)).
        c_total_left = sorption.total_concentration(c_left)
        c_total_right = sorption.total_concentration(c_right)
        v_shock_expected = (c_right - c_left) / (c_total_right - c_total_left)

        assert np.isclose(v_shock, v_shock_expected, rtol=1e-14)

    def test_shock_velocity_equal_concentrations(self):
        """Shock speed when c_left = c_right (degenerate case)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0

        v_shock = sorption.shock_speed(c, c)
        v_char = 1.0 / float(sorption.retardation(c))
        assert np.isclose(v_shock, v_char, rtol=1e-14)

    def test_entropy_condition_physical_shock(self):
        """Test entropy for physical compression shock (c_left > c_right)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        c_right = 2.0

        v_shock = sorption.shock_speed(c_left, c_right)
        assert sorption.check_entropy_condition(c_left, c_right, v_shock)

    def test_entropy_condition_unphysical_shock(self):
        """Test entropy for unphysical expansion shock (c_left < c_right)."""
        sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        c_left = 2.0
        c_right = 10.0

        v_shock = sorption.shock_speed(c_left, c_right)
        assert not sorption.check_entropy_condition(c_left, c_right, v_shock)

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
        """Shock speed equals characteristic speed 1/R for constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        v_shock = sorption.shock_speed(c_left=10.0, c_right=2.0)
        v_expected = 1.0 / 2.0
        assert np.isclose(v_shock, v_expected, rtol=1e-14)

    def test_entropy_condition_always_true(self):
        """Test that entropy condition is always satisfied."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        v_shock = sorption.shock_speed(10.0, 2.0)
        satisfies = sorption.check_entropy_condition(10.0, 2.0, v_shock)
        assert satisfies


class TestCharacteristicFunctions:
    """Test characteristic velocity and position functions."""

    def test_characteristic_velocity_freundlich(self):
        """Characteristic speed in (V, θ): dV/dθ = 1/R(c) — flow-free."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c = 5.0

        v = characteristic_speed(c, sorption)
        v_expected = 1.0 / float(sorption.retardation(c))

        assert np.isclose(v, v_expected, rtol=1e-14)

    def test_characteristic_velocity_constant(self):
        """Characteristic speed with constant retardation is 1/R."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 5.0

        v = characteristic_speed(c, sorption)
        v_expected = 1.0 / 2.0

        assert np.isclose(v, v_expected, rtol=1e-14)

    def test_characteristic_position_linear_propagation(self):
        """V(θ) = (1/R) · θ — linear in cumulative flow."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 5.0
        theta_start = 0.0
        v_start = 0.0

        for theta in [1.0, 5.0, 10.0]:
            v_pos = characteristic_position(c, sorption, theta_start, v_start, theta)
            assert v_pos is not None
            v_expected = (1.0 / 2.0) * theta
            assert np.isclose(v_pos, v_expected, rtol=1e-14)

    def test_characteristic_position_before_start(self):
        """Position is None for θ < θ_start."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        v_pos = characteristic_position(c=5.0, sorption=sorption, theta_start=10.0, v_start=0.0, theta=5.0)
        assert v_pos is None

    def test_characteristic_position_nonzero_start(self):
        """V(θ) = v_start + (1/R)·(θ - θ_start)."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 5.0
        theta_start = 5.0
        v_start = 100.0
        theta = 15.0

        v_pos = characteristic_position(c, sorption, theta_start, v_start, theta)
        assert v_pos is not None
        speed = 1.0 / 2.0
        v_expected = v_start + speed * (theta - theta_start)

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

    def test_first_arrival_extrapolates_past_simulation_window(self):
        """Past the simulation window, the last bin's flow is extrapolated (θ-native semantics).

        Previously this test asserted ``np.inf`` for "insufficient flow history";
        the (V, θ) refactor moves the time/θ map into the public API where
        out-of-window translation extrapolates the last positive flow.
        """
        cin = np.array([0.0, 10.0])
        flow = np.array([10.0, 10.0])  # Constant 10 m³/day
        tedges = pd.date_range("2020-01-01", periods=3, freq="10D")  # [0, 10, 20] days
        aquifer_pore_volume = 10000.0  # Very large pore volume
        sorption = ConstantRetardation(retardation_factor=2.0)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # V_target = V·R = 20000 m³. cin>0 from t=10 day onward. θ_emit = 100.
        # θ_target = 100 + 20000 = 20100. tedges_days[-1] = 20, theta_edges[-1] = 200.
        # t_target = 20 + (20100 - 200)/10 = 2010 days.
        assert np.isclose(t_first, 2010.0, rtol=1e-14)


class TestRegressionsForIssue168:
    """Regression tests for the physics fixes in Phase 1 (issue #168).

    Each test below is constructed so that the analytic reference value is
    computed independently of the implementation under test (no mirroring of
    the function's own arithmetic). Loosening any tolerance here means the fix
    is incomplete — route the failure back to Phase 1 physics review.
    """

    def test_first_arrival_step_zero_to_c_uses_shock_velocity_n_gt_1(self):
        """P1.3 (n>1 branch): 0 -> C step creates a R-H shock; arrival = V*C_tot(C)/(C*flow)."""
        n_bins = 40
        c_step = 5.0
        flow_val = 100.0
        v_pore = 500.0
        cin = np.array([0.0] + [c_step] * (n_bins - 1))
        flow = np.full(n_bins, flow_val)
        tedges = pd.date_range("2020-01-01", periods=n_bins + 1, freq="10D")
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        # Analytic reference: closed-form shock velocity arrival, independent of
        # compute_first_front_arrival_time's cumulative-sum integration.
        c_total_at_c = sorption.total_concentration(c_step)
        # First nonzero bin starts at tedges[1] = 10 days; transit takes V*C_tot/(C*flow).
        t_analytic = 10.0 + v_pore * c_total_at_c / (c_step * flow_val)

        t_first = compute_first_front_arrival_time(cin, flow, tedges, v_pore, sorption)

        assert np.isclose(t_first, t_analytic, rtol=1e-14)

    def test_first_arrival_step_zero_to_c_n_lt_1_uses_characteristic_speed(self):
        """P1.3 (n<1 branch): solver emits a CharacteristicWave; arrival = V*R(C)/flow."""
        n_bins = 400
        c_step = 5.0
        flow_val = 100.0
        v_pore = 500.0
        cin = np.array([0.0] + [c_step] * (n_bins - 1))
        flow = np.full(n_bins, flow_val)
        tedges = pd.date_range("2020-01-01", periods=n_bins + 1, freq="10D")
        # Use a small k_f to keep R(5) modest so the analytic arrival fits in the time grid.
        sorption = FreundlichSorption(k_f=0.001, n=0.5, bulk_density=1500.0, porosity=0.3, c_min=0.0)

        # Analytic reference: characteristic velocity (n<1 solver path).
        r_at_c = sorption.retardation(c_step)
        t_analytic = 10.0 + v_pore * r_at_c / flow_val

        t_first = compute_first_front_arrival_time(cin, flow, tedges, v_pore, sorption)

        # Sanity: n<1 result must NOT equal the n>1 shock formula here.
        c_total_at_c = sorption.total_concentration(c_step)
        wrong_shock_t = 10.0 + v_pore * c_total_at_c / (c_step * flow_val)
        assert not np.isclose(t_analytic, wrong_shock_t, rtol=1e-3)

        assert np.isclose(t_first, t_analytic, rtol=1e-14)

    def test_freundlich_retardation_clamps_below_c_min_but_total_concentration_does_not(self):
        """Retardation clamps to c_min (avoids R->inf as c->0 for n>1); C_T does not clamp.

        ``c_min`` exists to keep ``R(c)`` finite as ``c -> 0`` for ``n > 1``
        (where ``R(c) = 1 + alpha * c^((1-n)/n)`` diverges). It does NOT apply
        to ``total_concentration``, whose ``c^(1/n)`` factor is well-defined
        at ``c = 0`` for any ``n > 0``. Phase 2 step 4 disentangled these:
        ``C_T(0) = 0`` physically, regardless of ``c_min``.
        """
        c_min = 0.1
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3, c_min=c_min)

        for c in [0.0, c_min / 10.0, c_min / 2.0]:
            # retardation clamps c -> c_min so R(c<c_min) = R(c_min) (no singularity).
            assert sorption.retardation(c) == sorption.retardation(c_min)
            # total_concentration uses c^(1/n) directly; at c=0 it returns 0.
            assert sorption.total_concentration(c) == c + (
                sorption.bulk_density / sorption.porosity * sorption.k_f * (max(c, 0.0) ** (1.0 / sorption.n))
            )

    def test_concentration_from_retardation_no_runtime_warning_at_r_le_1(self):
        """P1.5: masking base before exponentiation removes the warning."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Scalar at r=1 (exactly).
            sorption.concentration_from_retardation(1.0)
            # Scalar at r<1.
            sorption.concentration_from_retardation(0.5)
            # Array spanning r<1, r=1, r>1.
            sorption.concentration_from_retardation(np.array([0.5, 1.0, 5.0]))

        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert runtime_warnings == [], (
            f"Expected no RuntimeWarning, got {[(w.category.__name__, str(w.message)) for w in runtime_warnings]}"
        )


# Staringreeks-like soils used to parametrise tests across exponent regimes.
# Each entry is (theta_r, theta_s, k_s, lam) for BC or (theta_r, theta_s, k_s, n_vG) for vG.
# IDs name the soil class for readable failure reports.
_STARINGREEKS_BC = [
    pytest.param(0.01, 0.337, 0.174, 0.25, id="O05_coarse_sand_a11"),
    pytest.param(0.02, 0.45, 0.05, 0.5, id="B02_loam_a7"),
    pytest.param(0.01, 0.43, 0.005, 5.0, id="C03_clay_a3p4"),
]
_STARINGREEKS_VG = [
    pytest.param(0.01, 0.337, 0.174, 2.28, id="O05_sand_like"),
    pytest.param(0.02, 0.45, 0.05, 1.8, id="B02_loam_like"),
    pytest.param(0.01, 0.43, 0.005, 1.15, id="C03_clay_like"),
]


@pytest.mark.parametrize(("theta_r", "theta_s", "k_s", "lam"), _STARINGREEKS_BC)
class TestBrooksCoreyConductivity:
    """Brooks-Corey closed-form unsaturated-conductivity sorption (BC = Mualem variant)."""

    def test_constructor_derives_a_and_delta_theta(self, theta_r, theta_s, k_s, lam):
        """``a = 3 + 2/λ`` and ``Δθ = θ_s − θ_r`` set in ``__post_init__``."""
        sorption = BrooksCoreyConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, brooks_corey_lambda=lam)
        assert sorption.a == pytest.approx(3.0 + 2.0 / lam)
        assert sorption.delta_theta == pytest.approx(theta_s - theta_r)

    def test_round_trip_concentration_machine_precision(self, theta_r, theta_s, k_s, lam):
        """``C → R → C`` round-trips at machine precision for the closed-form BC."""
        sorption = BrooksCoreyConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, brooks_corey_lambda=lam)
        # Grid avoids the dry-soil singularity (clamped at _C_MIN).
        c_grid = np.geomspace(1e-8 * k_s, 0.99 * k_s, 25)
        r_grid = sorption.retardation(c_grid)
        c_back = sorption.concentration_from_retardation(r_grid)
        np.testing.assert_allclose(c_back, c_grid, rtol=1e-13)

    def test_saturation_limit_exact(self, theta_r, theta_s, k_s, lam):
        """``C_T(K_s) = Δθ`` and ``C_T(0) = 0`` exactly."""
        sorption = BrooksCoreyConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, brooks_corey_lambda=lam)
        assert sorption.total_concentration(k_s) == pytest.approx(theta_s - theta_r, rel=1e-15)
        assert sorption.total_concentration(0.0) == 0.0

    def test_retardation_strictly_positive(self, theta_r, theta_s, k_s, lam):
        """``R(C) > 0`` for ``C ∈ [_C_MIN, K_s]`` (catches sign-flip in inversion exponent)."""
        sorption = BrooksCoreyConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, brooks_corey_lambda=lam)
        c_grid = np.geomspace(1e-10 * k_s, k_s, 20)
        assert np.all(sorption.retardation(c_grid) > 0)

    def test_shock_speed_matches_finite_difference(self, theta_r, theta_s, k_s, lam):
        """Inherited ``shock_speed`` = ``(K_R−K_L)/(θ_R−θ_L)`` (Rankine-Hugoniot)."""
        sorption = BrooksCoreyConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, brooks_corey_lambda=lam)
        a = sorption.a
        theta1, theta2 = theta_r + 0.6 * (theta_s - theta_r), theta_r + 0.1 * (theta_s - theta_r)
        k1 = k_s * ((theta1 - theta_r) / (theta_s - theta_r)) ** a
        k2 = k_s * ((theta2 - theta_r) / (theta_s - theta_r)) ** a
        expected = (k1 - k2) / (theta1 - theta2)
        # Pass K (= C) values as c_left and c_right (framework convention).
        np.testing.assert_allclose(sorption.shock_speed(k1, k2), expected, rtol=1e-13)

    def test_entropy_wetting_vs_drying(self, theta_r, theta_s, k_s, lam):
        """Wetting front (θ_1 > θ_2) is entropy-admissible; drying is not."""
        sorption = BrooksCoreyConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, brooks_corey_lambda=lam)
        a = sorption.a
        theta_high, theta_low = theta_r + 0.7 * (theta_s - theta_r), theta_r + 0.2 * (theta_s - theta_r)
        k_high = k_s * ((theta_high - theta_r) / (theta_s - theta_r)) ** a
        k_low = k_s * ((theta_low - theta_r) / (theta_s - theta_r)) ** a
        # Wetting: c_left = K_high (upstream), c_right = K_low.
        s_wet = sorption.shock_speed(k_high, k_low)
        assert sorption.check_entropy_condition(k_high, k_low, s_wet)
        # Drying: reverse roles; shock speed sign-flips but Lax fails.
        s_dry = sorption.shock_speed(k_low, k_high)
        assert not sorption.check_entropy_condition(k_low, k_high, s_dry)


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"theta_r": -0.1, "theta_s": 0.337, "k_s": 0.174, "brooks_corey_lambda": 0.25},
        {"theta_r": 0.4, "theta_s": 0.337, "k_s": 0.174, "brooks_corey_lambda": 0.25},  # θ_r >= θ_s
        {"theta_r": 0.01, "theta_s": 1.5, "k_s": 0.174, "brooks_corey_lambda": 0.25},
        {"theta_r": 0.01, "theta_s": 0.337, "k_s": -0.1, "brooks_corey_lambda": 0.25},
        {"theta_r": 0.01, "theta_s": 0.337, "k_s": 0.174, "brooks_corey_lambda": -0.5},
    ],
)
def test_brooks_corey_validation_rejects_invalid(invalid_kwargs):
    """Coverage-only parametrised validation. Each invalid input raises ``ValueError``."""
    with pytest.raises(ValueError):
        BrooksCoreyConductivity(**invalid_kwargs)


@pytest.mark.parametrize(("theta_r", "theta_s", "k_s", "n_vg"), _STARINGREEKS_VG)
class TestVanGenuchtenMualemConductivity:
    """Van Genuchten-Mualem K(θ) sorption with brentq inversions."""

    def test_constructor_derives_m_and_delta_theta(self, theta_r, theta_s, k_s, n_vg):
        """``m = 1 − 1/n_vG`` and ``Δθ = θ_s − θ_r`` set in ``__post_init__``."""
        sorption = VanGenuchtenMualemConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, van_genuchten_n=n_vg)
        assert sorption.m == pytest.approx(1.0 - 1.0 / n_vg)
        assert sorption.delta_theta == pytest.approx(theta_s - theta_r)

    def test_round_trip_concentration_brentq_bounded(self, theta_r, theta_s, k_s, n_vg):
        """``C → R → C`` round-trips within brentq xtol (1e-14) headroom.

        Grid stays below ``0.3·k_s`` so the inversion is well-conditioned for
        all Staringreeks soils, including clay (``n_vG ≈ 1.15``). Near
        saturation, the ``K_M(S_e)`` curve becomes nearly vertical and the
        round-trip's ULP precision is dominated by the conditioning of the
        forward function, not by any algebraic error. Well-conditioned regime:
        tolerance is brentq-bounded (``1e-12``), one ULP above ``BRENTQ_XTOL``.
        """
        sorption = VanGenuchtenMualemConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, van_genuchten_n=n_vg)
        c_grid = np.geomspace(1e-6 * k_s, 0.3 * k_s, 12)
        r_grid = sorption.retardation(c_grid)
        c_back = sorption.concentration_from_retardation(r_grid)
        np.testing.assert_allclose(c_back, c_grid, rtol=1e-12)

    def test_saturation_limit_exact(self, theta_r, theta_s, k_s, n_vg):
        """``C_T(K_s) = Δθ`` exactly (the ``_se_from_c`` early-return at saturation)."""
        sorption = VanGenuchtenMualemConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, van_genuchten_n=n_vg)
        assert sorption.total_concentration(k_s) == pytest.approx(theta_s - theta_r, rel=1e-15)
        assert sorption.total_concentration(0.0) == 0.0

    def test_retardation_strictly_positive(self, theta_r, theta_s, k_s, n_vg):
        """``R(C) > 0`` for ``C ∈ [_C_MIN, 0.9·K_s]`` (avoid the s→1 dK/dS singularity edge)."""
        sorption = VanGenuchtenMualemConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, van_genuchten_n=n_vg)
        c_grid = np.geomspace(1e-8 * k_s, 0.9 * k_s, 12)
        assert np.all(sorption.retardation(c_grid) > 0)

    def test_characteristic_speed_at_saturation_is_inf(self, theta_r, theta_s, k_s, n_vg):
        """At ``S_e = 1`` (``C = K_s``) the characteristic celerity is the removable limit ``+∞``.

        ``dK_M/dS_e → +∞`` as ``S_e → 1`` (the ``T^{m-1}`` term, ``m-1 < 0``), so
        ``R = Δθ/(dK/dS_e) = 0`` exactly and ``1/R`` is ``+∞`` — not a ``ZeroDivisionError``.
        Below saturation the speed is finite and equals ``1/R(C)`` to the last bit.
        """
        sorption = VanGenuchtenMualemConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, van_genuchten_n=n_vg)
        assert sorption.retardation(k_s) == 0.0
        assert characteristic_speed(k_s, sorption) == np.inf
        c = 0.3 * k_s
        assert characteristic_speed(c, sorption) == 1.0 / float(sorption.retardation(c))

    def test_check_entropy_condition_saturated_wetting_front(self, theta_r, theta_s, k_s, n_vg):
        """A wetting-front shock from a saturated upstream state (``C_L = K_s``) is admissible.

        The Lax condition ``λ_left = 1/R(K_s) = +∞ > shock_speed > λ_right`` holds, so the
        entropy check returns ``True`` (was a ``ZeroDivisionError`` before the ``R = 0 → +∞`` fix).
        """
        sorption = VanGenuchtenMualemConductivity(theta_r=theta_r, theta_s=theta_s, k_s=k_s, van_genuchten_n=n_vg)
        c_right = 1e-9 * k_s
        shock_speed = sorption.shock_speed(k_s, c_right)
        assert np.isfinite(shock_speed)
        assert sorption.check_entropy_condition(k_s, c_right, shock_speed) is True


@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"theta_r": -0.1, "theta_s": 0.337, "k_s": 0.174, "van_genuchten_n": 2.28},
        {"theta_r": 0.4, "theta_s": 0.337, "k_s": 0.174, "van_genuchten_n": 2.28},  # θ_r >= θ_s
        {"theta_r": 0.01, "theta_s": 1.5, "k_s": 0.174, "van_genuchten_n": 2.28},
        {"theta_r": 0.01, "theta_s": 0.337, "k_s": -0.1, "van_genuchten_n": 2.28},
        {"theta_r": 0.01, "theta_s": 0.337, "k_s": 0.174, "van_genuchten_n": 1.0},  # n_vG <= 1
        {"theta_r": 0.01, "theta_s": 0.337, "k_s": 0.174, "van_genuchten_n": 2.28, "mualem_l": -0.1},
    ],
)
def test_van_genuchten_validation_rejects_invalid(invalid_kwargs):
    """Coverage-only parametrised validation. Each invalid input raises ``ValueError``."""
    with pytest.raises(ValueError):
        VanGenuchtenMualemConductivity(**invalid_kwargs)
