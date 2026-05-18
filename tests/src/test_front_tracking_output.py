"""
Tests for Front Tracking Concentration Extraction in (V, θ) coordinates.

All public output functions in ``gwtransport.fronttracking.output`` take θ
(cumulative flow, m³). These tests follow the same convention.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pytest

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.output import (
    compute_bin_averaged_concentration_exact,
    compute_breakthrough_curve,
    compute_cumulative_inlet_mass,
    compute_cumulative_outlet_mass,
    compute_domain_mass,
    concentration_at_point,
    identify_outlet_segments,
    integrate_rarefaction_exact,
)
from gwtransport.fronttracking.waves import CharacteristicWave, DecayingShockWave, RarefactionWave, ShockWave


class TestConcentrationAtPoint:
    """Test exact point-wise concentration computation."""

    def test_single_characteristic_constant_concentration(self):
        """Concentration from single characteristic with constant retardation."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption, is_active=True)
        waves = [char]

        # At θ=1000, char at v = (1/2)·1000 = 500. char has swept [0, 500].
        # Behind char (already passed): c=5
        c = concentration_at_point(v=400.0, theta=1000.0, waves=waves, sorption=sorption)
        assert c == 5.0

        # At char position: c=5
        c = concentration_at_point(v=500.0, theta=1000.0, waves=waves, sorption=sorption)
        assert c == 5.0

        # Ahead of char (not reached yet): c=0
        c = concentration_at_point(v=600.0, theta=1000.0, waves=waves, sorption=sorption)
        assert c == 0.0

    def test_single_shock_discontinuity(self):
        """Concentration jump across shock."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption, is_active=True)
        waves = [shock]

        theta = 1000.0
        v_shock = shock.position_at_theta(theta)
        assert v_shock is not None

        c = concentration_at_point(v=v_shock - 10.0, theta=theta, waves=waves, sorption=sorption)
        assert c == 10.0

        c = concentration_at_point(v=v_shock + 10.0, theta=theta, waves=waves, sorption=sorption)
        assert c == 2.0

    def test_rarefaction_self_similar_solution(self):
        """Concentration inside rarefaction fan satisfies R(C) = θ/v."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )
        waves = [raref]

        # Pick (v, θ) inside the fan.
        theta = 2000.0
        r_head = float(sorption.retardation(10.0))
        r_tail = float(sorption.retardation(2.0))
        v = 0.5 * (theta / r_head + theta / r_tail)

        c = concentration_at_point(v=v, theta=theta, waves=waves, sorption=sorption)

        r_expected = theta / v
        r_actual = sorption.retardation(c)

        assert np.isclose(r_actual, r_expected, rtol=1e-14)

    def test_multiple_characteristics_most_recent_wins(self):
        """Most recent characteristic determines concentration."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption, is_active=True)
        char2 = CharacteristicWave(
            theta_start=500.0, v_start=0.0, concentration=10.0, sorption=sorption, is_active=True
        )
        waves = [char1, char2]

        # At θ=1500, both have passed v=250.
        # char1 reaches v=250 at θ = 0 + 250/(1/2) = 500.
        # char2 reaches v=250 at θ = 500 + 250/(1/2) = 1000.
        # Most recent is char2, so c=10.
        c = concentration_at_point(v=250.0, theta=1500.0, waves=waves, sorption=sorption)
        assert c == 10.0

    def test_inactive_waves_ignored(self):
        """Inactive waves don't contribute."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption, is_active=False)
        waves = [char]

        c = concentration_at_point(v=500.0, theta=1000.0, waves=waves, sorption=sorption)
        assert c == 0.0

    def test_no_waves_returns_zero(self):
        """Initial condition (no waves) returns zero."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        waves = []

        c = concentration_at_point(v=100.0, theta=500.0, waves=waves, sorption=sorption)
        assert c == 0.0

    def test_dispatch_prefers_decaying_shock_over_rarefaction(self):
        """DecayingShockWave must win over a co-active RarefactionWave at the same point.

        Phase 2 step 4 puts DecayingShockWave first in the wave-priority loop
        in ``concentration_at_point``. End-to-end simulations rarely surface
        this dispatch order because the parent rarefaction is deactivated
        post-collision; this test keeps both active to exercise the priority
        directly.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=8.0, c_tail=1e-12, sorption=sorption)
        v_start_dsw = raref.head_position_at_theta(10.0)
        assert v_start_dsw is not None
        decaying = DecayingShockWave(
            theta_start=10.0,
            v_start=v_start_dsw,
            c_decay_initial=raref.c_head,
            c_fixed=0.0,
            decay_side="left",
            v_origin=0.0,
            theta_origin=0.0,
            sorption=sorption,
        )

        theta_q = 20.0
        v_q = decaying.position_at_theta(theta_q)
        assert v_q is not None

        c_decay_at_q = decaying.c_decay_at_theta(theta_q)
        assert c_decay_at_q is not None
        c_dsw_expected = 0.5 * (c_decay_at_q + decaying.c_fixed)

        # Both orderings must return the DSW value (priority is by type,
        # not list position).
        c1 = concentration_at_point(v=v_q, theta=theta_q, waves=[raref, decaying], sorption=sorption)
        c2 = concentration_at_point(v=v_q, theta=theta_q, waves=[decaying, raref], sorption=sorption)
        assert np.isclose(c1, c_dsw_expected, rtol=1e-12)
        assert np.isclose(c2, c_dsw_expected, rtol=1e-12)


class TestComputeBreakthroughCurve:
    """Test breakthrough curve computation (θ-array based)."""

    def test_constant_concentration_plateau(self):
        """Constant-concentration breakthrough plateaus after head arrival."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption, is_active=True)
        waves = [char]
        v_outlet = 500.0

        # Char reaches outlet at θ = v_outlet · R = 500·2 = 1000.
        theta_array = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
        c_out = compute_breakthrough_curve(theta_array, v_outlet, waves, sorption)

        assert c_out[0] == 0.0
        assert c_out[1] == 0.0
        assert c_out[2] == 5.0  # arrival
        assert c_out[3] == 5.0
        assert c_out[4] == 5.0

    def test_shock_creates_step_breakthrough(self):
        """Breakthrough shows step at shock arrival."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=0.0, sorption=sorption, is_active=True)
        waves = [shock]
        v_outlet = 300.0

        assert shock.speed is not None
        theta_cross = v_outlet / shock.speed

        theta_array = np.linspace(0, theta_cross * 2, 100)
        c_out = compute_breakthrough_curve(theta_array, v_outlet, waves, sorption)

        idx_before = np.where(theta_array < theta_cross - 10.0)[0]
        if len(idx_before) > 0:
            assert np.all(c_out[idx_before] == 0.0)

        idx_after = np.where(theta_array > theta_cross + 10.0)[0]
        if len(idx_after) > 0:
            assert np.all(c_out[idx_after] == 10.0)


class TestIdentifyOutletSegments:
    """Test outlet segment identification (θ-based)."""

    def test_single_characteristic_crossing(self):
        """Segment identification with one characteristic."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption, is_active=True)
        waves = [char]
        v_outlet = 500.0

        # Char crosses at θ = v_outlet · R = 1000.
        segments = identify_outlet_segments(
            theta_start=0.0, theta_end=2000.0, v_outlet=v_outlet, waves=waves, sorption=sorption
        )

        assert len(segments) == 2

        assert segments[0]["theta_start"] == 0.0
        assert np.isclose(segments[0]["theta_end"], 1000.0, rtol=1e-10)
        assert segments[0]["type"] == "constant"
        assert segments[0]["concentration"] == 0.0

        assert np.isclose(segments[1]["theta_start"], 1000.0, rtol=1e-10)
        assert segments[1]["theta_end"] == 2000.0
        assert segments[1]["type"] == "constant"
        assert segments[1]["concentration"] == 5.0

    def test_rarefaction_creates_varying_segment(self):
        """Rarefaction creates θ-varying segment."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )
        waves = [raref]
        v_outlet = 500.0

        # Head crosses at θ = v_outlet / head_speed
        theta_head = v_outlet / raref.head_speed()
        theta_tail = v_outlet / raref.tail_speed()

        segments = identify_outlet_segments(
            theta_start=0.0, theta_end=theta_tail * 1.5, v_outlet=v_outlet, waves=waves, sorption=sorption
        )

        raref_segments = [s for s in segments if s["type"] == "rarefaction"]
        assert len(raref_segments) >= 1

        raref_seg = raref_segments[0]
        assert raref_seg["wave"] is raref
        assert np.isclose(raref_seg["theta_start"], theta_head, rtol=1e-10)

    def test_multiple_crossings_create_multiple_segments(self):
        """Multiple wave crossings create correct segments."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption, is_active=True)
        char2 = CharacteristicWave(
            theta_start=500.0, v_start=0.0, concentration=10.0, sorption=sorption, is_active=True
        )
        waves = [char1, char2]
        v_outlet = 500.0

        # char1 crosses at θ=1000, char2 crosses at θ=1500.
        segments = identify_outlet_segments(
            theta_start=0.0, theta_end=2000.0, v_outlet=v_outlet, waves=waves, sorption=sorption
        )

        # Should have 3 segments: [0, 1000] c=0, [1000, 1500] c=5, [1500, 2000] c=10.
        assert len(segments) == 3


class TestIntegrateRarefactionExact:
    """Test exact analytical integration of rarefaction (θ-based)."""

    def test_integration_positive_definite(self):
        """Integration returns positive value."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )
        v_outlet = 500.0

        # Pick θ range inside the rarefaction at the outlet.
        theta_head = v_outlet / raref.head_speed()
        theta_tail = v_outlet / raref.tail_speed()
        theta_start = theta_head + 100.0
        theta_end = theta_tail - 100.0

        integral = integrate_rarefaction_exact(raref, v_outlet, theta_start, theta_end, sorption)

        assert integral > 0

    def test_integration_additive(self):
        """Additivity: ∫[a,c] = ∫[a,b] + ∫[b,c]."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )
        v_outlet = 500.0

        theta_head = v_outlet / raref.head_speed()
        theta_tail = v_outlet / raref.tail_speed()
        theta_a = theta_head + 100.0
        theta_b = 0.5 * (theta_a + theta_tail)
        theta_c = theta_tail - 100.0

        integral_ac = integrate_rarefaction_exact(raref, v_outlet, theta_a, theta_c, sorption)
        integral_ab = integrate_rarefaction_exact(raref, v_outlet, theta_a, theta_b, sorption)
        integral_bc = integrate_rarefaction_exact(raref, v_outlet, theta_b, theta_c, sorption)

        assert np.isclose(integral_ac, integral_ab + integral_bc, rtol=1e-14)

    def test_integration_matches_numerical_quadrature(self):
        """Exact integral matches high-order numerical quadrature."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )
        v_outlet = 500.0

        theta_head = v_outlet / raref.head_speed()
        theta_tail = v_outlet / raref.tail_speed()
        theta_start = theta_head + 10.0
        theta_end = theta_tail - 10.0

        integral_exact = integrate_rarefaction_exact(raref, v_outlet, theta_start, theta_end, sorption)

        theta_grid = np.linspace(theta_start, theta_end, 10000)
        c_grid = np.array([raref.concentration_at_point(v_outlet, t) or 0.0 for t in theta_grid])
        integral_numerical = float(np.trapezoid(c_grid, theta_grid))

        assert np.isclose(integral_exact, integral_numerical, rtol=1e-4)

    def test_zero_interval_returns_zero(self):
        """Integration over zero-width interval is zero."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )
        v_outlet = 500.0
        theta = 1000.0

        integral = integrate_rarefaction_exact(raref, v_outlet, theta, theta, sorption)

        assert integral == 0.0


class TestComputeBinAveragedConcentrationExact:
    """Test θ-bin-averaged concentration with exact integration."""

    def test_constant_concentration_exact_average(self):
        """Bin averaging with constant concentration."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption, is_active=True)
        waves = [char]
        v_outlet = 500.0

        # Char arrives at θ=1000. Bins: [0,500], [500,1000], [1000,1500], [1500,2000].
        theta_edges = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
        c_avg = compute_bin_averaged_concentration_exact(theta_edges, v_outlet, waves, sorption)

        assert c_avg[0] == 0.0  # [0, 500]
        assert c_avg[1] == 0.0  # [500, 1000] — arrival exactly at right edge
        assert c_avg[2] == 5.0  # [1000, 1500]
        assert c_avg[3] == 5.0  # [1500, 2000]

    def test_bin_averaging_conserves_mass(self):
        """Bin averaging conserves total mass at machine precision."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        # Pulse: c=10 then c=0.
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=10.0, sorption=sorption, is_active=True)
        char2 = CharacteristicWave(theta_start=500.0, v_start=0.0, concentration=0.0, sorption=sorption, is_active=True)
        waves = [char1, char2]
        v_outlet = 500.0

        # char1 reaches outlet at θ=1000, char2 reaches at θ=1500.
        # Total mass extracted via θ-integral = 10 * (1500-1000) = 5000.
        theta_edges = np.linspace(0, 3000, 31)
        c_avg = compute_bin_averaged_concentration_exact(theta_edges, v_outlet, waves, sorption)

        dtheta = np.diff(theta_edges)
        # ∫ c dθ in θ-coordinates equals mass extracted.
        mass_extracted = float(np.sum(c_avg * dtheta))

        expected_mass = 10.0 * 500.0

        assert np.isclose(mass_extracted, expected_mass, rtol=1e-13)

    def test_rarefaction_bin_averaging_exact(self):
        """Bin averaging with rarefaction uses exact integration."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )
        waves = [raref]
        v_outlet = 500.0

        theta_head = v_outlet / raref.head_speed()
        theta_tail = v_outlet / raref.tail_speed()

        theta_edges = np.linspace(theta_head - 500.0, theta_tail + 500.0, 20)
        c_avg = compute_bin_averaged_concentration_exact(theta_edges, v_outlet, waves, sorption)

        assert np.all(c_avg >= 0)

        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        raref_bins = (theta_centers > theta_head) & (theta_centers < theta_tail)
        if np.any(raref_bins):
            c_min = min(raref.c_head, raref.c_tail)
            c_max = max(raref.c_head, raref.c_tail)
            assert np.all(c_avg[raref_bins] >= c_min - 1e-10)
            assert np.all(c_avg[raref_bins] <= c_max + 1e-10)

    def test_invalid_bins_raise_error(self):
        """Invalid θ-bins (decreasing) raise ValueError."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        waves = []
        v_outlet = 500.0

        theta_edges = np.array([1000.0, 500.0, 0.0])

        with pytest.raises(ValueError, match="Invalid θ-bin"):
            compute_bin_averaged_concentration_exact(theta_edges, v_outlet, waves, sorption)

    def test_fine_binning_converges_to_breakthrough_curve(self):
        """Fine binning approximates breakthrough curve."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=2.0, sorption=sorption, is_active=True)
        waves = [shock]
        v_outlet = 300.0

        theta_edges = np.linspace(0, 2000, 1001)
        c_avg = compute_bin_averaged_concentration_exact(theta_edges, v_outlet, waves, sorption)

        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        c_breakthrough = compute_breakthrough_curve(theta_centers, v_outlet, waves, sorption)

        assert np.allclose(c_avg, c_breakthrough, rtol=1e-3)


class TestMachinePrecision:
    """Test that all functions maintain machine precision."""

    def test_characteristic_roundtrip_exact(self):
        """Characteristic position calculation is exact."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=5.0, sorption=sorption, is_active=True)

        # At θ=1000: v = (1/2)·1000 = 500.
        v_at_theta = char.position_at_theta(1000.0)
        assert v_at_theta is not None

        c = concentration_at_point(v=v_at_theta, theta=1000.0, waves=[char], sorption=sorption)

        assert c == 5.0

    def test_shock_velocity_exact_rankine_hugoniot(self):
        """Shock speed satisfies Rankine-Hugoniot exactly in (V, θ)."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        c_left = 10.0
        c_right = 2.0

        shock = ShockWave(
            theta_start=0.0, v_start=0.0, c_left=c_left, c_right=c_right, sorption=sorption, is_active=True
        )

        # R-H in (V, θ): dV_s/dθ = (c_R - c_L)/(C_T(c_R) - C_T(c_L)).
        speed_expected = (c_right - c_left) / (
            sorption.total_concentration(c_right) - sorption.total_concentration(c_left)
        )

        assert shock.speed is not None
        assert np.isclose(shock.speed, speed_expected, rtol=1e-14)

    def test_rarefaction_integral_machine_precision(self):
        """Rarefaction integration is additive to machine precision."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=0.0, v_start=0.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )
        v_outlet = 500.0

        theta_head = v_outlet / raref.head_speed()
        theta_a = theta_head + 100.0
        theta_b = theta_a + 1000.0
        theta_c = theta_a + 2000.0

        integral_ac = integrate_rarefaction_exact(raref, v_outlet, theta_a, theta_c, sorption)
        integral_ab = integrate_rarefaction_exact(raref, v_outlet, theta_a, theta_b, sorption)
        integral_bc = integrate_rarefaction_exact(raref, v_outlet, theta_b, theta_c, sorption)

        assert np.isclose(integral_ac, integral_ab + integral_bc, rtol=1e-14, atol=1e-14)


class TestMassBalanceFunctions:
    """Tests for cumulative inlet/outlet/domain mass in θ-coordinates.

    These tests pin closed-form values to machine precision so that scaling
    mutations (e.g., 0.5*) of the underlying functions are caught.
    """

    def test_compute_domain_mass_constant_concentration_analytic(self):
        """Single characteristic at constant c: domain mass = C_T(c) · v_swept.

        Uses ConstantRetardation so C_T(0)=0 exactly (no c_min clamping); the
        unswept region therefore contributes zero, isolating the swept-region term.
        """
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 10.0
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=c, sorption=sorption, is_active=True)

        v_outlet = 500.0
        theta = 600.0  # speed 0.5 → v_swept = 300, comfortably inside [0, v_outlet]
        waves = [char]

        mass = compute_domain_mass(theta, v_outlet, waves, sorption)

        speed = 1.0 / sorption.retardation_factor
        v_swept = min(speed * theta, v_outlet)
        expected = sorption.total_concentration(c) * v_swept

        assert np.isclose(mass, expected, rtol=1e-12)

    def test_compute_domain_mass_with_shock_analytic(self):
        """Shock at v_shock(θ) separates two constant regions; domain mass adds them.

        Uses ConstantRetardation so C_T(0)=0 (no clamping); the upstream-of-shock
        region (v < v_shock_start) is uncontrolled → C=0 and contributes zero.
        """
        sorption = ConstantRetardation(retardation_factor=2.0)
        c_left, c_right = 15.0, 5.0
        shock = ShockWave(
            theta_start=500.0, v_start=100.0, c_left=c_left, c_right=c_right, sorption=sorption, is_active=True
        )

        v_outlet = 500.0
        theta = 900.0  # speed 0.5 → v_shock = 300, inside [0, 500]
        waves = [shock]

        v_shock = 100.0 + shock.speed * (theta - 500.0)
        assert 0 < v_shock < v_outlet

        mass = compute_domain_mass(theta, v_outlet, waves, sorption)

        # ``concentration_at_point`` reports c_left for ALL v < v_shock (the
        # shock is the only wave, so there is no upstream boundary). Domain
        # mass therefore integrates c_left over [0, v_shock] and c_right over
        # [v_shock, v_outlet].
        ct_left = sorption.total_concentration(c_left)
        ct_right = sorption.total_concentration(c_right)
        expected = ct_left * v_shock + ct_right * (v_outlet - v_shock)

        assert np.isclose(mass, expected, rtol=1e-12)

    def test_compute_domain_mass_with_rarefaction_positive_finite(self):
        """compute_domain_mass with rarefaction is non-zero and finite.

        Analytic value depends on the incomplete-beta closed form; physics-math
        reviewer's `/tmp/check_spatial_freundlich.py` verified it to 1e-16 against
        scipy.quad. Here we only guard against degeneracies (mass==0, NaN, inf).
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        raref = RarefactionWave(
            theta_start=500.0, v_start=100.0, c_head=12.0, c_tail=4.0, sorption=sorption, is_active=True
        )

        v_outlet = 500.0
        theta = 1500.0
        waves = [raref]

        mass = compute_domain_mass(theta, v_outlet, waves, sorption)

        assert mass > 0.0
        assert np.isfinite(mass)

    def test_compute_cumulative_inlet_mass_constant_concentration(self):
        """Constant concentration: ∫ c dθ = c · (θ_query - θ_0)."""
        cin = np.full(100, 10.0)
        theta_edges = np.linspace(0.0, 10000.0, 101)
        theta_query = 3100.0

        mass = compute_cumulative_inlet_mass(theta_query, cin, theta_edges)

        # ∫ 10 dθ from 0 to 3100 = 31000.
        assert np.isclose(mass, 31000.0, rtol=1e-12)

    def test_compute_cumulative_inlet_mass_variable_concentration_analytic(self):
        """Variable cin: analytic integral against piecewise-constant bins."""
        cin = np.linspace(5.0, 15.0, 50)
        theta_edges = np.linspace(0.0, 5000.0, 51)
        theta_query = 3100.0

        mass = compute_cumulative_inlet_mass(theta_query, cin, theta_edges)

        # Closed form: cin[i] is constant on [theta_edges[i], theta_edges[i+1]].
        # Sum c[i]·overlap(bin_i, [0, theta_query]).
        widths = np.clip(theta_query - theta_edges[:-1], 0.0, np.diff(theta_edges))
        expected = float(np.sum(cin * widths))
        assert np.isclose(mass, expected, rtol=1e-14)

    def test_compute_cumulative_outlet_mass_analytic(self):
        """Single characteristic crosses outlet at θ=V·R; cumulative outlet mass after = c · (θ - V·R)."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        c = 10.0
        v_outlet = 500.0
        char = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=c, sorption=sorption, is_active=True)
        waves = [char]

        theta_cross = v_outlet * sorption.retardation_factor  # 1000
        theta_query = 1900.0  # well past crossing

        # Synthetic inlet history matching the wave list: sustained c=10 over
        # the entire theta range.
        cin = np.array([c])
        theta_edges = np.array([0.0, theta_query])
        mass = compute_cumulative_outlet_mass(theta_query, v_outlet, waves, sorption, cin=cin, theta_edges=theta_edges)

        # Closed form: ∫_θ_cross^θ_query c dθ' = c · (θ_query - θ_cross) = 10 · 900 = 9000.
        expected = c * (theta_query - theta_cross)
        assert np.isclose(mass, expected, rtol=1e-12)
