"""
Tests for Front Tracking Concentration Extraction in (V, θ) coordinates.

All public output functions in ``gwtransport.fronttracking.output`` take θ
(cumulative flow, m³). These tests follow the same convention.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.integrate

from gwtransport.fronttracking.math import BrooksCoreyConductivity, ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.output import (
    compute_bin_averaged_concentration_exact,
    compute_breakthrough_curve,
    compute_cumulative_inlet_mass,
    compute_cumulative_outlet_mass,
    compute_domain_mass,
    compute_total_outlet_mass,
    concentration_at_point,
    identify_outlet_segments,
    integrate_rarefaction_exact,
)
from gwtransport.fronttracking.solver import FrontTracker
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

    def test_decaying_shock_fan_owns_interior_point(self):
        """The reader attributes a DecayingShockWave's self-similar fan value inside its fan.

        A single Freundlich ``n=2`` pulse ``[0, 10, 10, 0]`` forms a DecayingShockWave once
        the trailing rarefaction catches the leading shock: the parent rarefaction is
        deactivated and the DSW fan (apex at ``(v_origin, θ_origin)``) becomes the sole owner
        of the region between its tail boundary and its curved shock face. The old dispatch-
        priority test kept a rarefaction and a decaying shock co-active over the same region;
        under the nearest-downstream-face sweep reader that geometry is single-owner-ambiguous
        (both fans share the origin apex and range) and is not how the solver produces states.
        This uses an interaction-consistent solver-built state instead: at a point strictly
        inside the DSW fan the reader must return the self-similar fan concentration, which
        discriminates against both the ``c_fixed`` plateau ahead of the shock and the shock's
        two-sided average.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        n = 40
        cin = np.array([0.0, 10.0, 10.0, 0.0] + [0.0] * (n - 4))
        flow = np.full(n, 100.0)
        tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
        tracker = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=300.0, sorption=sorption)
        tracker.run()

        decaying = next(w for w in tracker.state.waves if isinstance(w, DecayingShockWave))
        assert decaying.is_active, "the trailing rarefaction should have formed an active decaying shock"

        # A point strictly inside the fan: upstream of the curved shock face, downstream of the
        # tail boundary (which sits at the apex for c_fan_tail = 0).
        theta_q = 1000.0
        v_shock = decaying.position_at_theta(theta_q)
        assert v_shock is not None
        v_q = 0.5 * v_shock

        # Expected self-similar fan value: R = (θ − θ_apex)/(v − v_apex), inverted to concentration.
        r_fan = (theta_q - decaying.theta_origin) / (v_q - decaying.v_origin)
        c_fan_expected = float(sorption.concentration_from_retardation(r_fan))

        c = concentration_at_point(v=v_q, theta=theta_q, waves=tracker.state.waves, sorption=sorption)
        assert np.isclose(c, c_fan_expected, rtol=1e-12)
        # Discriminate: the reader attributes the fan, not the plateau ahead of the shock.
        assert not np.isclose(c, decaying.c_fixed, atol=1e-6)


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
        """Bin averaging conserves total mass and matches an independent reference.

        Two checks pin the bin-average to a *wave-dependent* truth (a wrong wave
        geometry or a wiped wave list fails both, so neither is tautological):

        1. The total extracted mass ``Σ c_avg·Δθ`` equals the closed-form
           ``c_left·(θ_cross2 − θ_cross1)`` — this depends on both crossing θs,
           hence on both wave speeds.
        2. The same total equals an *independent* adaptive quadrature of the
           pointwise breakthrough curve ``∫ C(v_outlet, θ) dθ`` (scipy
           Gauss-Kronrod, a different integration route than the segment-exact
           bin average). A geometry error that preserves total mass by accident
           would still split differently across these two routes.
        """
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

        # Independent reference: adaptive quadrature of the pointwise curve.
        # The breakthrough is a top-hat in θ (c=10 on [1000, 1500], else 0); the
        # two interior break points are supplied so quad resolves the steps.
        mass_quad, _ = scipy.integrate.quad(
            lambda t: concentration_at_point(v_outlet, t, waves, sorption),
            0.0,
            3000.0,
            points=[1000.0, 1500.0],
            epsabs=1e-12,
            epsrel=1e-12,
        )
        assert np.isclose(mass_extracted, mass_quad, rtol=1e-12)

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

        compute_bin_averaged_concentration_exact(
            theta_edges_out, v_outlet, waves, sorption, cin=cin, theta_edges_inlet=theta_edges_inlet
        )

        assert len(recwarn.list) == 0, "No warning expected for output bins inside the inlet window"

    def test_conservation_form_warns_when_output_exceeds_inlet_window(self):
        """Driving output edges past ``theta_edges_inlet[-1]`` fires the UserWarning.

        Once the output range exceeds the inlet window, ``m_in`` saturates at
        the last injected mass while the wave list keeps evolving, producing
        FP-cancellation residuals on inconsistent θ ranges. The function must
        surface this as a ``UserWarning`` (and still clamp the result to the
        ``cout >= 0`` contract).
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

        # The clamp preserves the non-negativity contract despite the warning.
        assert np.all(result >= 0.0)


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


class TestDecayingShockWaveDomainMassConservation:
    """``compute_domain_mass`` conserves mass once a DecayingShockWave forms (T3 regression).

    A sustained-then-reduced inlet (wet-then-dry) forms a ``decay_side='left'`` DSW whose fan
    tail concentration ``c_fan_tail`` exceeds the downstream ``c_fixed``. Before the first outlet
    crossing no mass has left the domain, so the stored domain mass must equal the cumulative
    injected mass exactly: ``m_dom(theta) == m_in(theta)``. This anchor uses only the closed-form
    ``compute_cumulative_inlet_mass`` and the function under test (never the outlet integral), so
    it is independent -- not tautological. Before the fix (``c_apex=c_fixed`` over-counted the
    abandoned fan tail) the pre-arrival residual reaches ~9 %.
    """

    def test_domain_mass_equals_inlet_mass_before_arrival_with_dsw(self):
        """No-outflow invariant ``m_dom(theta) == m_in(theta)`` to machine precision pre-arrival."""
        sorption = BrooksCoreyConductivity(theta_r=0.01, theta_s=0.337, k_s=0.174, brooks_corey_lambda=0.25)
        cin = np.array([0.003] * 30 + [0.0008] * 170)
        n = len(cin)
        tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
        v_outlet = sorption.theta_s * 0.5
        tracker = FrontTracker(
            cin=cin,
            flow=sorption.theta_s * np.ones(n),
            tedges=tedges,
            aquifer_pore_volume=float(v_outlet),
            sorption=sorption,
        )
        tracker.run(max_iterations=10000)
        waves = tracker.state.waves
        theta_edges = tracker.state.theta_edges

        decaying = [w for w in waves if isinstance(w, DecayingShockWave)]
        assert decaying, "scenario must form a DecayingShockWave"
        dsw = decaying[0]
        assert dsw.c_fan_tail > dsw.c_fixed  # the regime that exposed the over-counting bug

        arrivals = [ev["theta"] for ev in tracker.state.events if ev["type"] == "outlet_crossing"]
        theta_arrival = min(arrivals) if arrivals else float(theta_edges[-1])
        # Sample strictly inside the DSW-active window and before any outlet crossing.
        thetas = np.linspace(dsw.theta_start + 1e-3, min(dsw.theta_deactivation - 1e-3, theta_arrival - 1e-3), 25)
        m_in = np.array([
            compute_cumulative_inlet_mass(theta=float(t), cin=cin, theta_edges=theta_edges) for t in thetas
        ])
        m_dom = np.array([
            compute_domain_mass(theta=float(t), v_outlet=v_outlet, waves=waves, sorption=sorption) for t in thetas
        ])
        np.testing.assert_allclose(m_dom, m_in, rtol=1e-12)


def _godunov_fv_domain_mass(cin, theta_edges, v_outlet, sorption, theta_query, n_cells=1500):
    """Independent upwind (Godunov) FV oracle for the n=2 Freundlich domain mass in [0, v_outlet].

    Marches C_total in (V, θ) with an upwind flux, inverting the closed-form n=2 total→dissolved
    map each step. Stores only the current column (never the space-time history), so it stays well
    under memory limits. Returns (domain_mass, c_outlet) at ``theta_query``.
    """
    a = sorption.bulk_density * sorption.k_f / sorption.porosity  # C_total = c + a*sqrt(c) (n=2)

    def c_from_ct(u):
        # Invert C_total = c + a*sqrt(c): x = sqrt(c) solves x² + a·x − C_total = 0.
        u = np.maximum(u, 0.0)
        x = 0.5 * (-a + np.sqrt(a * a + 4.0 * u))
        return x * x

    te = np.asarray(theta_edges, dtype=float)
    cin = np.asarray(cin, dtype=float)
    dv = v_outlet * 1.25 / n_cells
    dtheta = 0.4 * dv
    out_idx = int(v_outlet / dv)
    c = np.zeros(n_cells)
    ct = sorption.total_concentration(c)
    theta = 0.0
    while theta < theta_query:
        k = min(max(int(np.searchsorted(te, theta, side="right")) - 1, 0), len(cin) - 1)
        c_in = cin[k] if theta < te[-1] else 0.0
        c_left = np.empty(n_cells)
        c_left[0] = c_in
        c_left[1:] = c[:-1]
        ct -= (dtheta / dv) * (c - c_left)
        c = c_from_ct(ct)
        theta += dtheta
    return float(np.sum(ct[:out_idx]) * dv), float(c[out_idx])


class TestSinglePulseDomainMass:
    """``compute_domain_mass`` matches an independent Godunov FV oracle on a single pulse.

    A single pulse produces exactly one fan-bearing wave, so the single-owner
    ``compute_domain_mass`` reader is exact (up to Riemann discretisation) and equals the FV
    resident mass. Multi-pulse / oscillating inputs form OVERLAPPING non-interacting fans that
    the reader cannot compose exactly; that unresolved interaction is now refused at the public
    boundary (:func:`gwtransport.advection.infiltration_to_extraction_nonlinear_sorption` raises
    ``NotImplementedError`` via :func:`gwtransport.fronttracking.solver.find_unresolved_interaction`),
    and its exact resolution is deferred to a solver-level follow-up — so the former multi-peak /
    fan-truncation / oscillating conservation cases (which pinned a since-reverted output-side
    mitigation) are covered by the raise, not by an internal-``compute_domain_mass`` assertion.
    """

    def test_fv_oracle_matches_single_pulse(self):
        """The Godunov FV oracle agrees with ``compute_domain_mass`` on a single pulse (~0.5%)."""
        cin = np.array([0.0, 10.0, 10.0, 0.0, 0.0, 0.0])
        v_outlet = 40.0
        sorption = FreundlichSorption(k_f=0.05, n=2.0, bulk_density=1600.0, porosity=0.35)
        tedges = pd.date_range("2020-01-01", periods=len(cin) + 1, freq="D")
        tr = FrontTracker(
            cin=cin, flow=np.full(len(cin), 100.0), tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption
        )
        tr.run()
        theta_q = 500.0  # before breakthrough: all injected mass resident
        m_dom = compute_domain_mass(theta=theta_q, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
        fv_dom, fv_cout = _godunov_fv_domain_mass(cin, tr.state.theta_edges, v_outlet, sorption, theta_q)
        assert fv_cout < 1e-6  # single pulse has not broken through — resident mass only
        assert abs(m_dom - fv_dom) < 0.01 * fv_dom


class TestConservationFormClampScale:
    """Regression: the FP-noise clamp/warning scales with the m_in−m_dom cancellation (review O2).

    The old band ``1e-12·max(cout)`` keyed off the OUTPUT concentration, which collapses to ~0
    before breakthrough — far too tight, so ~1e-11 cancellation dust on a fully in-range Freundlich
    input tripped a UserWarning carrying a factually false "edges exceed inlet range" message.
    """

    def test_in_range_freundlich_multipeak_emits_no_warning(self, recwarn):
        """The report's in-range multi-peak Freundlich case emits no UserWarning."""
        cin = np.array([0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0])
        v_outlet = 40.0
        sorption = FreundlichSorption(k_f=0.05, n=2.0, bulk_density=1600.0, porosity=0.35)
        tedges = pd.date_range("2020-01-01", periods=len(cin) + 1, freq="D")
        tr = FrontTracker(
            cin=cin, flow=np.full(len(cin), 100.0), tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption
        )
        tr.run()
        theta_edges_out = np.linspace(1.0, 899.0, 80)  # every edge inside [0, 900]
        compute_bin_averaged_concentration_exact(
            theta_edges_out, v_outlet, tr.state.waves, sorption, cin=cin, theta_edges_inlet=tr.state.theta_edges
        )
        assert len(recwarn.list) == 0, "in-range conservation-form input must not warn"

    def test_out_of_window_warning_names_the_true_cause(self):
        """Output edges past the inlet window still warn, and the message says 'exceeding' the window.

        A genuine out-of-window residual needs the wave list to keep growing m_dom while m_in
        saturates — use a sustained boundary (single ShockWave, c_left=10) whose inlet record ends
        before the output range.
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=0.0, sorption=sorption, is_active=True)
        waves = [shock]
        v_outlet = 300.0
        theta_cross = v_outlet / shock.speed
        theta_edges_out = np.linspace(7.0, theta_cross * 2.0 + 7.0, 50)
        theta_edges_inlet = np.array([0.0, theta_cross * 0.5])  # ends before the output range
        cin = np.array([10.0])
        with pytest.warns(UserWarning, match="exceeding theta_edges_inlet"):
            compute_bin_averaged_concentration_exact(
                theta_edges_out, v_outlet, waves, sorption, cin=cin, theta_edges_inlet=theta_edges_inlet
            )


class TestTotalOutletMassSemantics:
    """``compute_total_outlet_mass`` honest c_∞ semantics (review O3).

    Baseline paired the FINITE record integral ``m_in_total`` with the infinite-time steady-state
    fill ``C_T(c_∞)·v_outlet``, returning a NEGATIVE total whenever ``m_in_total < C_T(c_∞)·v_outlet``
    (demonstrated −3405.7). A pulse (``c_∞=0``) keeps ``m_in_total``; a sustained ``c_∞>0`` boundary
    injects forever so the total outlet mass is unbounded → ``+inf``.
    """

    def test_pulse_returns_injected_mass(self):
        """``cin[-1] == 0`` returns the finite injected mass ``Σ cin·Δθ``."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        cin = np.array([0.0, 4.0, 4.0, 0.0])
        theta_edges = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
        m_out = compute_total_outlet_mass(v_outlet=500.0, sorption=sorption, cin=cin, theta_edges=theta_edges)
        assert m_out == float(np.sum(cin * np.diff(theta_edges)))  # = 800.0, exact

    def test_sustained_ambient_returns_inf_not_negative(self):
        """``cin[-1] > 0`` with the −3405.7 layout returns +inf, never a negative mass."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        cin = np.array([0.0, 4.0, 4.0])  # c_∞ = 4 sustained
        theta_edges = np.array([0.0, 100.0, 200.0, 300.0])
        v_outlet = 5000.0  # large enough that C_T(c_∞)·v_outlet > m_in_total (baseline went negative)
        m_in_total = float(np.sum(cin * np.diff(theta_edges)))
        assert float(sorption.total_concentration(4.0)) * v_outlet > m_in_total  # the baseline-negative regime
        m_out = compute_total_outlet_mass(v_outlet=v_outlet, sorption=sorption, cin=cin, theta_edges=theta_edges)
        assert m_out == np.inf


class TestConservationFormInletMassOutOfRange:
    """Cumsum m_in at output edges == per-edge reference, incl. below/above the inlet window (O4).

    The O(N+M) cumsum+searchsorted evaluation of the cumulative inlet integral must match the
    per-edge ``compute_cumulative_inlet_mass`` reference to FP precision even when output edges fall
    below ``theta_edges_inlet[0]`` (mass 0) or above ``theta_edges_inlet[-1]`` (saturated total).
    """

    def test_conservation_form_out_of_range_edges_match_reference(self):
        """End-to-end conservation form with out-of-range edges equals the m_in−m_dom reference."""
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=10.0, c_right=0.0, sorption=sorption, is_active=True)
        waves = [shock]
        v_outlet = 300.0
        # Inlet record starts mid-window (te_in[0] = 50) and ends at 350; output edges straddle both.
        theta_edges_inlet = np.array([50.0, 150.0, 250.0, 350.0])
        cin = np.array([10.0, 10.0, 10.0])
        theta_edges_out = np.array([10.0, 40.0, 90.0, 200.0, 300.0, 380.0, 500.0])

        c_avg = compute_bin_averaged_concentration_exact(
            theta_edges_out, v_outlet, waves, sorption, cin=cin, theta_edges_inlet=theta_edges_inlet
        )

        # Independent reference: per-edge compute_cumulative_inlet_mass (the O(N·M) path the cumsum
        # replaces), minus domain mass, clamped exactly as the conservation branch does.
        m_in_ref = np.array([
            compute_cumulative_inlet_mass(theta=float(e), cin=cin, theta_edges=theta_edges_inlet)
            for e in theta_edges_out
        ])
        m_dom_ref = np.array([
            compute_domain_mass(theta=float(e), v_outlet=v_outlet, waves=waves, sorption=sorption)
            for e in theta_edges_out
        ])
        m_out_ref = np.where(theta_edges_out <= 0.0, 0.0, m_in_ref - m_dom_ref)
        ref = np.maximum(np.diff(m_out_ref) / np.diff(theta_edges_out), 0.0)

        np.testing.assert_allclose(c_avg, ref, rtol=1e-12, atol=1e-12)
