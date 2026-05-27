"""
Tests for Front Tracking Concentration Extraction in (V, θ) coordinates.

All public output functions in ``gwtransport.fronttracking.output`` take θ
(cumulative flow, m³). These tests follow the same convention.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pytest
import scipy.integrate

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
            c_fan_tail=raref.c_tail,
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
        """Exact integral matches high-accuracy adaptive quadrature.

        Validates the closed form itself (not a trapezoid rule's truncation): scipy's adaptive
        Gauss-Kronrod quadrature of the pointwise concentration converges far tighter than a
        fixed grid, so a sign/factor error in the antiderivative would surface here.
        """
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
        integral_quad, _ = scipy.integrate.quad(
            lambda t: raref.concentration_at_point(v_outlet, t), theta_start, theta_end, epsabs=1e-12, epsrel=1e-12
        )

        # The closed form matches adaptive quadrature to ~machine precision (measured ~2e-16);
        # rtol=1e-12 keeps a safe margin while still pinning the antiderivative tightly.
        np.testing.assert_allclose(integral_exact, integral_quad, rtol=1e-12)

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


class TestConservationFormBinAverage:
    """Test the conservation-form branch of ``compute_bin_averaged_concentration_exact``.

    Passing ``cin`` + ``theta_edges_inlet`` switches the function to the
    ``c_avg = (Δm_in − Δm_dom) / Δθ`` identity (the recommended multi-DSW
    path, carrying the FP-cancellation clamp and the post-inlet ``UserWarning``).
    Tests without those kwargs only exercise the legacy segment path.
    """

    def test_conservation_form_bin_average_matches_legacy(self):
        """Conservation form agrees with the legacy segment path for a single shock.

        For a canonical single ShockWave (``c_right = 0``) crossing the outlet,
        both paths are valid. They must agree to the floating-point
        cancellation bound of the ``m_in − m_dom`` subtraction, which is
        ``atol ≈ k · ε_machine · max(m_in) / min(Δθ)`` (the two cumulative-mass
        endpoint evaluations each carry ~``ε·m`` rounding, amplified by
        ``1/Δθ``). This bound is derived from the inputs, not assumed.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=c_left, c_right=0.0, sorption=sorption, is_active=True)
        waves = [shock]
        v_outlet = 300.0

        assert shock.speed is not None
        theta_cross = v_outlet / shock.speed

        # Offset the edges so none lands exactly on the crossing (a coarse edge
        # exactly at the crossing would expose the legacy path's arrival-bin
        # discretization, not an FP effect) and use fine bins.
        theta_edges_out = np.linspace(7.0, theta_cross * 2.0 + 7.0, 200)
        # Inlet history consistent with the wave: sustained c_left, window
        # comfortably covering the output range.
        theta_edges_inlet = np.array([0.0, theta_edges_out[-1] * 3.0])
        cin = np.array([c_left])

        c_legacy = compute_bin_averaged_concentration_exact(theta_edges_out, v_outlet, waves, sorption)
        c_cons = compute_bin_averaged_concentration_exact(
            theta_edges_out, v_outlet, waves, sorption, cin=cin, theta_edges_inlet=theta_edges_inlet
        )

        # Derived FP-cancellation bound (factor 8 covers both endpoint evals
        # plus the final difference).
        eps = np.finfo(float).eps
        m_in_max = compute_cumulative_inlet_mass(float(theta_edges_out[-1]), cin, theta_edges_inlet)
        dtheta_min = float(np.min(np.diff(theta_edges_out)))
        atol = 8.0 * eps * m_in_max / dtheta_min

        assert np.max(np.abs(c_cons - c_legacy)) <= atol

    def test_conservation_form_no_warning_for_in_window_output(self, recwarn):
        """In-window output range emits no UserWarning under the conservation form.

        When every output θ-edge lies inside ``[0, theta_edges_inlet[-1]]`` the
        inlet integral and the wave list share a consistent θ range, so the
        FP-cancellation diagnostic must stay silent.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=c_left, c_right=0.0, sorption=sorption, is_active=True)
        waves = [shock]
        v_outlet = 300.0

        assert shock.speed is not None
        theta_cross = v_outlet / shock.speed

        theta_edges_out = np.linspace(7.0, theta_cross * 2.0 + 7.0, 50)
        # Inlet window strictly past the largest output edge.
        theta_edges_inlet = np.array([0.0, theta_edges_out[-1] * 3.0])
        cin = np.array([c_left])

        result = compute_bin_averaged_concentration_exact(
            theta_edges_out, v_outlet, waves, sorption, cin=cin, theta_edges_inlet=theta_edges_inlet
        )

        assert len(recwarn.list) == 0, "No warning expected for output bins inside the inlet window"
        # In-window bins are fully informed: never NaN, never negative.
        assert not np.any(np.isnan(result)), "in-window output must not be NaN-masked"
        assert np.all(result >= 0.0)

    def test_conservation_form_warns_when_output_exceeds_inlet_window(self):
        """Driving output edges past ``theta_edges_inlet[-1]`` fires the UserWarning.

        Once the output range exceeds the inlet window, ``m_in`` saturates at
        the last injected mass while the wave list keeps evolving, producing
        FP-cancellation residuals on inconsistent θ ranges. The function must
        surface this as a ``UserWarning`` and NaN-mask the un-informed
        out-of-window bins; the in-window bins remain non-negative.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=c_left, c_right=0.0, sorption=sorption, is_active=True)
        waves = [shock]
        v_outlet = 300.0

        assert shock.speed is not None
        theta_cross = v_outlet / shock.speed

        theta_edges_out = np.linspace(7.0, theta_cross * 2.0 + 7.0, 50)
        # Inlet window ends BEFORE the output range — output edges exceed it.
        theta_edges_inlet = np.array([0.0, theta_cross * 0.5])
        cin = np.array([c_left])

        with pytest.warns(UserWarning, match="exceeding"):
            result = compute_bin_averaged_concentration_exact(
                theta_edges_out, v_outlet, waves, sorption, cin=cin, theta_edges_inlet=theta_edges_inlet
            )

        # Out-of-window (un-informed) bins are NaN-masked (#221); the finite
        # in-window bins still satisfy the non-negativity contract.
        assert np.any(np.isnan(result)), "expected un-informed out-of-window bins to be NaN-masked"
        assert np.all(result[~np.isnan(result)] >= 0.0)

    def test_late_breakthrough_past_inlet_window_stays_finite(self):
        """Correct positive late breakthrough past the inlet window is KEPT, not NaN-masked.

        A pulse (cin = 10 then 0) injects all its mass by θ=500; the breakthrough reaches the
        outlet later (θ∈[1000, 1500]). With ``theta_edges_inlet`` ending at 600 those breakthrough
        bins lie past the inlet window yet carry correct positive concentration — ``m_in`` is
        already complete and ``m_dom`` drains as the pulse exits. A θ-position NaN rule would
        wrongly mask them; the raw-residual rule keeps them (#221).
        """
        sorption = ConstantRetardation(retardation_factor=2.0)
        char1 = CharacteristicWave(theta_start=0.0, v_start=0.0, concentration=10.0, sorption=sorption, is_active=True)
        char2 = CharacteristicWave(theta_start=500.0, v_start=0.0, concentration=0.0, sorption=sorption, is_active=True)
        waves = [char1, char2]
        v_outlet = 500.0
        # cin injects 10 over [0, 500] then 0; all mass is in by θ=500, window ends at 600.
        theta_edges_inlet = np.array([0.0, 500.0, 600.0])
        cin = np.array([10.0, 0.0])
        # Output covers the breakthrough pulse (θ∈[1000, 1500]), well past the inlet window.
        theta_edges_out = np.linspace(900.0, 1600.0, 36)

        c_ref = compute_breakthrough_curve(
            0.5 * (theta_edges_out[:-1] + theta_edges_out[1:]), v_outlet, waves, sorption
        )
        c_avg = compute_bin_averaged_concentration_exact(
            theta_edges_out, v_outlet, waves, sorption, cin=cin, theta_edges_inlet=theta_edges_inlet
        )

        assert (c_ref > 1e-9).any(), "test setup invalid: expected a positive out-of-window breakthrough"
        assert not np.any(np.isnan(c_avg)), "correct out-of-window breakthrough was wrongly NaN-masked"
        assert np.all(c_avg >= -1e-9)
        # Mass exiting over the output range equals the injected pulse mass (10 over Δθ=500).
        mass = float(np.sum(c_avg * np.diff(theta_edges_out)))
        assert np.isclose(mass, 10.0 * 500.0, rtol=1e-3)

    def test_retained_mass_regime_no_false_nan(self):
        """Benign sub-ULP residuals in the retained-mass regime stay 0, not NaN (#221).

        With a single sustained shock whose front has not yet reached the outlet, the true
        breakthrough is identically 0 over the output range while ``m_in`` and ``m_dom`` are both
        large; their difference carries sub-ULP cancellation noise. The mass-scaled ``eps_clamp``
        classifies that noise as 0; an output-magnitude-scaled clamp would collapse to its floor
        here and spuriously NaN-mask a benign zero.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        c_left = 10.0
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=c_left, c_right=0.0, sorption=sorption, is_active=True)
        waves = [shock]
        v_outlet = 300.0
        assert shock.speed is not None
        theta_cross = v_outlet / shock.speed
        # Output entirely BEFORE the shock reaches the outlet: true breakthrough is 0.
        theta_edges_out = np.linspace(7.0, theta_cross * 0.8, 60)
        # Inlet window comfortably covers the output range (consistent θ ranges).
        theta_edges_inlet = np.array([0.0, theta_cross * 3.0])
        cin = np.array([c_left])

        result = compute_bin_averaged_concentration_exact(
            theta_edges_out, v_outlet, waves, sorption, cin=cin, theta_edges_inlet=theta_edges_inlet
        )

        assert not np.any(np.isnan(result)), "benign retained-mass residuals must not be NaN-masked"
        assert np.allclose(result, 0.0, atol=1e-9)


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
