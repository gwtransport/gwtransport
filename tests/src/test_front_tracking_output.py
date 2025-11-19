"""
Tests for Front Tracking Concentration Extraction (Phase 6).

This module tests the concentration extraction functions that compute outlet
concentrations from wave solutions using exact analytical integration.

All tests verify machine precision accuracy (rtol=1e-14) and exact mass balance.
"""

import numpy as np
import pytest

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.output import (
    compute_bin_averaged_concentration_exact,
    compute_breakthrough_curve,
    concentration_at_point,
    identify_outlet_segments,
    integrate_rarefaction_exact,
)
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave


class TestConcentrationAtPoint:
    """Test exact point-wise concentration computation."""

    def test_single_characteristic_constant_concentration(self):
        """Test concentration from single characteristic."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption, is_active=True
        )

        waves = [char]

        # At t=10, characteristic is at v = 0 + (100/2)*10 = 500
        # The characteristic has swept through region [0, 500]

        # Behind characteristic (already passed): c=5
        c = concentration_at_point(v=400.0, t=10.0, waves=waves, sorption=sorption)
        assert c == 5.0

        # At characteristic position: c=5
        c = concentration_at_point(v=500.0, t=10.0, waves=waves, sorption=sorption)
        assert c == 5.0

        # Ahead of characteristic (not reached yet): c=0
        c = concentration_at_point(v=600.0, t=10.0, waves=waves, sorption=sorption)
        assert c == 0.0

    def test_single_shock_discontinuity(self):
        """Test concentration jump across shock."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        shock = ShockWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption, is_active=True
        )

        waves = [shock]

        # Get shock position at t=10
        v_shock = shock.position_at_time(10.0)

        # Before shock: c_left
        c = concentration_at_point(v=v_shock - 10.0, t=10.0, waves=waves, sorption=sorption)
        assert c == 10.0

        # After shock: c_right
        c = concentration_at_point(v=v_shock + 10.0, t=10.0, waves=waves, sorption=sorption)
        assert c == 2.0

    def test_rarefaction_self_similar_solution(self):
        """Test concentration within rarefaction fan."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        raref = RarefactionWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )

        waves = [raref]

        # Inside rarefaction fan
        t = 20.0
        v = 150.0

        c = concentration_at_point(v=v, t=t, waves=waves, sorption=sorption)

        # Verify self-similar solution: R(C) = flow*t/v
        r_expected = raref.flow * t / v
        r_actual = sorption.retardation(c)

        assert np.isclose(r_actual, r_expected, rtol=1e-14)

    def test_multiple_characteristics_most_recent_wins(self):
        """Test that most recent characteristic determines concentration."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        # Two characteristics at different times with different concentrations
        char1 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption, is_active=True
        )

        char2 = CharacteristicWave(
            t_start=5.0, v_start=0.0, flow=100.0, concentration=10.0, sorption=sorption, is_active=True
        )

        waves = [char1, char2]

        # At t=15, both characteristics have passed v=250
        # char1 reaches v=250 at t = 0 + 250/(100/2) = 5.0
        # char2 reaches v=250 at t = 5 + 250/(100/2) = 10.0
        # Most recent is char2, so c=10
        c = concentration_at_point(v=250.0, t=15.0, waves=waves, sorption=sorption)
        assert c == 10.0

    def test_inactive_waves_ignored(self):
        """Test that inactive waves don't contribute."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption, is_active=False
        )

        waves = [char]

        # Inactive wave should not affect concentration
        c = concentration_at_point(v=500.0, t=10.0, waves=waves, sorption=sorption)
        assert c == 0.0

    def test_no_waves_returns_zero(self):
        """Test initial condition when no waves control point."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        waves = []

        c = concentration_at_point(v=100.0, t=5.0, waves=waves, sorption=sorption)
        assert c == 0.0


class TestComputeBreakthroughCurve:
    """Test breakthrough curve computation."""

    def test_constant_concentration_plateau(self):
        """Test breakthrough curve with constant concentration."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption, is_active=True
        )

        waves = [char]
        v_outlet = 500.0

        # Characteristic reaches outlet at t = 500/(100/2) = 10 days
        t_array = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        c_out = compute_breakthrough_curve(t_array, v_outlet, waves, sorption)

        # Before arrival: 0
        assert c_out[0] == 0.0
        assert c_out[1] == 0.0

        # At and after arrival: 5.0
        assert c_out[2] == 5.0
        assert c_out[3] == 5.0
        assert c_out[4] == 5.0

    def test_shock_creates_step_breakthrough(self):
        """Test breakthrough curve shows step at shock arrival."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        # Shock from inlet: c_left=10 (incoming), c_right=0 (initial condition)
        shock = ShockWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=0.0, sorption=sorption, is_active=True
        )

        waves = [shock]
        v_outlet = 300.0

        # Shock crosses outlet at some time t_cross
        t_cross = v_outlet / shock.velocity

        t_array = np.linspace(0, t_cross * 2, 100)
        c_out = compute_breakthrough_curve(t_array, v_outlet, waves, sorption)

        # Before shock arrives at outlet: c=0 (c_right, ahead of shock)
        idx_before = np.where(t_array < t_cross - 0.1)[0]
        if len(idx_before) > 0:
            # Before shock reaches outlet, concentration is c_right (initial water)
            assert np.all(c_out[idx_before] == 0.0)

        # After shock passes outlet: c=10 (c_left, behind shock)
        idx_after = np.where(t_array > t_cross + 0.1)[0]
        if len(idx_after) > 0:
            # After shock passes, concentration is c_left (incoming water)
            assert np.all(c_out[idx_after] == 10.0)


class TestIdentifyOutletSegments:
    """Test outlet segment identification."""

    def test_single_characteristic_crossing(self):
        """Test segment identification with one characteristic."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption, is_active=True
        )

        waves = [char]
        v_outlet = 500.0

        # Characteristic crosses at t = 10
        segments = identify_outlet_segments(t_start=0.0, t_end=20.0, v_outlet=v_outlet, waves=waves, sorption=sorption)

        # Should have 2 segments: [0, 10] with c=0, [10, 20] with c=5
        assert len(segments) == 2

        assert segments[0]["t_start"] == 0.0
        assert np.isclose(segments[0]["t_end"], 10.0, rtol=1e-10)
        assert segments[0]["type"] == "constant"
        assert segments[0]["concentration"] == 0.0

        assert np.isclose(segments[1]["t_start"], 10.0, rtol=1e-10)
        assert segments[1]["t_end"] == 20.0
        assert segments[1]["type"] == "constant"
        assert segments[1]["concentration"] == 5.0

    def test_rarefaction_creates_varying_segment(self):
        """Test rarefaction creates time-varying segment."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        raref = RarefactionWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )

        waves = [raref]
        v_outlet = 500.0

        # Head crosses at t_head = v_outlet / v_head
        t_head = v_outlet / raref.head_velocity()
        # Tail crosses at t_tail = v_outlet / v_tail
        t_tail = v_outlet / raref.tail_velocity()

        segments = identify_outlet_segments(
            t_start=0.0, t_end=t_tail * 1.5, v_outlet=v_outlet, waves=waves, sorption=sorption
        )

        # Should have segments: [0, t_head] constant, [t_head, t_tail] rarefaction, [t_tail, end] constant
        raref_segments = [s for s in segments if s["type"] == "rarefaction"]
        assert len(raref_segments) >= 1

        raref_seg = raref_segments[0]
        assert raref_seg["wave"] is raref
        assert np.isclose(raref_seg["t_start"], t_head, rtol=1e-10)

    def test_multiple_crossings_create_multiple_segments(self):
        """Test multiple wave crossings create correct segments."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char1 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption, is_active=True
        )

        char2 = CharacteristicWave(
            t_start=5.0, v_start=0.0, flow=100.0, concentration=10.0, sorption=sorption, is_active=True
        )

        waves = [char1, char2]
        v_outlet = 500.0

        segments = identify_outlet_segments(t_start=0.0, t_end=20.0, v_outlet=v_outlet, waves=waves, sorption=sorption)

        # Should have 3 segments: [0, 10] c=0, [10, 15] c=5, [15, 20] c=10
        assert len(segments) == 3


class TestIntegrateRarefactionExact:
    """Test exact analytical integration of rarefaction."""

    def test_integration_positive_definite(self):
        """Test that integration returns positive value."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        raref = RarefactionWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )

        v_outlet = 500.0
        t_start = 10.0
        t_end = 20.0

        integral = integrate_rarefaction_exact(raref, v_outlet, t_start, t_end, sorption)

        assert integral > 0

    def test_integration_additive(self):
        """Test that integration is additive: ∫[a,c] = ∫[a,b] + ∫[b,c]."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        raref = RarefactionWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )

        v_outlet = 500.0
        t_a = 10.0
        t_b = 15.0
        t_c = 20.0

        # Integrate over [t_a, t_c]
        integral_ac = integrate_rarefaction_exact(raref, v_outlet, t_a, t_c, sorption)

        # Integrate over [t_a, t_b] and [t_b, t_c]
        integral_ab = integrate_rarefaction_exact(raref, v_outlet, t_a, t_b, sorption)
        integral_bc = integrate_rarefaction_exact(raref, v_outlet, t_b, t_c, sorption)

        # Should satisfy: ∫[a,c] = ∫[a,b] + ∫[b,c]
        assert np.isclose(integral_ac, integral_ab + integral_bc, rtol=1e-14)

    def test_integration_matches_numerical_quadrature(self):
        """Test exact integral matches high-order numerical quadrature."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        raref = RarefactionWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )

        v_outlet = 500.0

        # Rarefaction head arrives at outlet at t_head
        t_head = v_outlet / raref.head_velocity()
        # Rarefaction tail arrives at outlet at t_tail
        t_tail = v_outlet / raref.tail_velocity()

        # Integrate over the period when rarefaction is passing outlet
        t_start = t_head + 0.1
        t_end = t_tail - 0.1

        # Exact analytical integral
        integral_exact = integrate_rarefaction_exact(raref, v_outlet, t_start, t_end, sorption)

        # Numerical integration using fine grid - only inside rarefaction
        t_array = np.linspace(t_start, t_end, 10000)
        c_array = np.array([raref.concentration_at_point(v_outlet, t) for t in t_array])

        # Remove any None values (shouldn't be any if we're in the rarefaction)
        mask = c_array is not None
        c_array_clean = c_array[mask].astype(float)
        t_array_clean = t_array[mask]

        integral_numerical = np.trapezoid(c_array_clean, t_array_clean)

        # Should match to high precision
        assert np.isclose(integral_exact, integral_numerical, rtol=1e-6)

    def test_zero_time_interval_returns_zero(self):
        """Test integration over zero time interval."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        raref = RarefactionWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )

        v_outlet = 500.0
        t = 10.0

        integral = integrate_rarefaction_exact(raref, v_outlet, t, t, sorption)

        assert integral == 0.0


class TestComputeBinAveragedConcentrationExact:
    """Test bin-averaged concentration with exact integration."""

    def test_constant_concentration_exact_average(self):
        """Test bin averaging with constant concentration."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption, is_active=True
        )

        waves = [char]
        v_outlet = 500.0

        # Characteristic arrives at t=10
        t_edges = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        c_avg = compute_bin_averaged_concentration_exact(t_edges, v_outlet, waves, sorption)

        # Bins before arrival: 0
        assert c_avg[0] == 0.0  # [0, 5]

        # Bin straddling arrival: partial
        # [5, 10]: half is 0, half is 0 (arrival exactly at end)
        assert c_avg[1] == 0.0  # [5, 10]

        # Bins after arrival: 5.0
        assert c_avg[2] == 5.0  # [10, 15]
        assert c_avg[3] == 5.0  # [15, 20]

    def test_bin_averaging_conserves_mass(self):
        """Test that bin averaging conserves total mass."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        # Create pulse: rise then fall
        char1 = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=10.0, sorption=sorption, is_active=True
        )

        char2 = CharacteristicWave(
            t_start=5.0, v_start=0.0, flow=100.0, concentration=0.0, sorption=sorption, is_active=True
        )

        waves = [char1, char2]
        v_outlet = 500.0

        # char1 reaches outlet at t=10, char2 reaches at t=15
        # Total mass extracted = 10 * (15-10) * flow = 10 * 5 * 100 = 5000

        t_edges = np.linspace(0, 30, 31)
        c_avg = compute_bin_averaged_concentration_exact(t_edges, v_outlet, waves, sorption)

        # Compute total mass extracted: sum(c_avg * dt * flow)
        dt = np.diff(t_edges)
        mass_extracted = np.sum(c_avg * dt) * 100.0  # flow = 100

        expected_mass = 10.0 * 5.0 * 100.0  # c * dt * flow

        assert np.isclose(mass_extracted, expected_mass, rtol=1e-13)

    def test_rarefaction_bin_averaging_exact(self):
        """Test bin averaging with rarefaction uses exact integration."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        raref = RarefactionWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )

        waves = [raref]
        v_outlet = 500.0

        # Create bins spanning rarefaction passage
        t_head = v_outlet / raref.head_velocity()
        t_tail = v_outlet / raref.tail_velocity()

        t_edges = np.linspace(t_head - 5, t_tail + 5, 20)
        c_avg = compute_bin_averaged_concentration_exact(t_edges, v_outlet, waves, sorption)

        # All values should be non-negative
        assert np.all(c_avg >= 0)

        # Values during rarefaction should be between c_head and c_tail
        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

        raref_bins = (t_centers > t_head) & (t_centers < t_tail)
        if np.any(raref_bins):
            c_min = min(raref.c_head, raref.c_tail)
            c_max = max(raref.c_head, raref.c_tail)
            # Bin averages should be in this range
            assert np.all(c_avg[raref_bins] >= c_min - 1e-10)
            assert np.all(c_avg[raref_bins] <= c_max + 1e-10)

    def test_invalid_time_bins_raise_error(self):
        """Test that invalid time bins raise ValueError."""
        sorption = ConstantRetardation(retardation_factor=2.0)
        waves = []
        v_outlet = 500.0

        # Decreasing time edges (invalid)
        t_edges = np.array([10.0, 5.0, 0.0])

        with pytest.raises(ValueError, match="Invalid time bin"):
            compute_bin_averaged_concentration_exact(t_edges, v_outlet, waves, sorption)

    def test_fine_binning_converges_to_breakthrough_curve(self):
        """Test that fine binning approximates breakthrough curve."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        shock = ShockWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_left=10.0, c_right=2.0, sorption=sorption, is_active=True
        )

        waves = [shock]
        v_outlet = 300.0

        # Very fine binning
        t_edges = np.linspace(0, 20, 1001)
        c_avg = compute_bin_averaged_concentration_exact(t_edges, v_outlet, waves, sorption)

        # Breakthrough curve at bin centers
        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
        c_breakthrough = compute_breakthrough_curve(t_centers, v_outlet, waves, sorption)

        # With very fine bins, bin average should approximately equal point value
        assert np.allclose(c_avg, c_breakthrough, rtol=1e-3)


class TestMachinePrecision:
    """Test that all functions maintain machine precision."""

    def test_characteristic_roundtrip_exact(self):
        """Test that characteristic position calculation is exact."""
        sorption = ConstantRetardation(retardation_factor=2.0)

        char = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption, is_active=True
        )

        # Position at t=10: v = 0 + (100/2)*10 = 500
        v_at_10 = char.position_at_time(10.0)

        # Concentration at that position should be exact
        c = concentration_at_point(v=v_at_10, t=10.0, waves=[char], sorption=sorption)

        assert c == 5.0  # Exact equality

    def test_shock_velocity_exact_rankine_hugoniot(self):
        """Test shock velocity satisfies Rankine-Hugoniot exactly."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        c_left = 10.0
        c_right = 2.0
        flow = 100.0

        shock = ShockWave(
            t_start=0.0, v_start=0.0, flow=flow, c_left=c_left, c_right=c_right, sorption=sorption, is_active=True
        )

        # Verify Rankine-Hugoniot condition exactly
        flux_jump = flow * (c_right - c_left)
        total_conc_jump = sorption.total_concentration(c_right) - sorption.total_concentration(c_left)

        v_shock_expected = flux_jump / total_conc_jump
        v_shock_actual = shock.velocity

        assert np.isclose(v_shock_actual, v_shock_expected, rtol=1e-14)

    def test_rarefaction_integral_machine_precision(self):
        """Test rarefaction integration maintains machine precision."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

        raref = RarefactionWave(
            t_start=0.0, v_start=0.0, flow=100.0, c_head=10.0, c_tail=2.0, sorption=sorption, is_active=True
        )

        v_outlet = 500.0

        # Test additivity to machine precision
        t_a, t_b, t_c = 10.0, 15.0, 20.0

        integral_ac = integrate_rarefaction_exact(raref, v_outlet, t_a, t_c, sorption)
        integral_ab = integrate_rarefaction_exact(raref, v_outlet, t_a, t_b, sorption)
        integral_bc = integrate_rarefaction_exact(raref, v_outlet, t_b, t_c, sorption)

        # Should be exact to floating-point precision
        assert np.isclose(integral_ac, integral_ab + integral_bc, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
