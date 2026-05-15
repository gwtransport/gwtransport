import warnings as warn_module

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special

from gwtransport import gamma as gamma_utils
from gwtransport.advection import (
    extraction_to_infiltration as advection_e2i,
)
from gwtransport.advection import (
    gamma_infiltration_to_extraction as gamma_i2e,
)
from gwtransport.advection import (
    infiltration_to_extraction as advection_i2e,
)
from gwtransport.diffusion import (
    _cfrac_mean_volume,
    _infiltration_to_extraction_coeff_matrix,
    _validate_diffusion_inputs,
    extraction_to_infiltration,
    gamma_extraction_to_infiltration,
    infiltration_to_extraction,
)
from gwtransport.diffusion import (
    gamma_infiltration_to_extraction as diffusion_gamma_i2e,
)
from gwtransport.gamma import mean_std_loc_to_alpha_beta


class TestInfiltrationToExtractionDiffusion:
    """Tests for the infiltration_to_extraction function with diffusion.

    These tests verify that the advection-dispersion transport model
    produces physically correct results.
    """

    @pytest.fixture
    def simple_setup(self):
        """Create a simple test setup with constant flow."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-15", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")

        # Step input: concentration 1 for first 5 days
        cin = np.zeros(len(tedges) - 1)
        cin[0:5] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0  # 100 m3/day

        # Single pore volume: 500 m3 = 5 days residence time at 100 m3/day
        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        return {
            "cin": cin,
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": cout_tedges,
            "aquifer_pore_volumes": aquifer_pore_volumes,
            "streamline_length": streamline_length,
        }

    def test_zero_diffusivity_matches_advection(self, simple_setup):
        """Test that zero diffusivity gives same result as pure advection."""
        cout_advection = advection_i2e(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
        )
        cout_diffusion = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=0.0,
        )
        # Only compare values after spin-up (where advection is not NaN)
        valid_mask = ~np.isnan(cout_advection)
        np.testing.assert_allclose(cout_advection[valid_mask], cout_diffusion[valid_mask])

    def test_small_diffusivity_close_to_advection(self, simple_setup):
        """Test that small diffusivity produces result close to advection."""
        cout_advection = advection_i2e(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
        )
        cout_diffusion = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=0.01,
            longitudinal_dispersivity=0.0,
        )
        # With D_m=0.01 m²/day and tau=5 days, the spatial spreading is
        # sqrt(2*D_m*tau) ≈ 0.32 m on a 100 m streamline (< 0.4% of domain).
        # Use atol since cin contains step transitions where the diffused
        # values cross zero — rtol is meaningless near-zero.
        valid = ~np.isnan(cout_advection) & ~np.isnan(cout_diffusion)
        np.testing.assert_allclose(cout_advection[valid], cout_diffusion[valid], atol=0.01)

    def test_larger_diffusivity_more_spreading(self, simple_setup):
        """Test that larger diffusivity causes more spreading."""
        cout_small_d = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=0.1,
            longitudinal_dispersivity=0.0,
        )
        cout_large_d = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=10.0,
            longitudinal_dispersivity=0.0,
        )
        # With larger diffusivity, the breakthrough curve should be more spread out.
        # Both peaks saturate at the input plateau (1.0), so compare the variance
        # of the breakthrough — wider spreading produces larger variance about
        # the centre of mass of the breakthrough.
        valid = ~np.isnan(cout_small_d) & ~np.isnan(cout_large_d)
        idx = np.arange(len(cout_small_d))[valid]
        small = cout_small_d[valid]
        large = cout_large_d[valid]
        # Centre of mass and second moment of each breakthrough
        com_small = np.sum(idx * small) / np.sum(small)
        com_large = np.sum(idx * large) / np.sum(large)
        var_small = np.sum((idx - com_small) ** 2 * small) / np.sum(small)
        var_large = np.sum((idx - com_large) ** 2 * large) / np.sum(large)
        assert var_large > var_small  # More spreading = larger variance

    def test_output_bounded_by_input(self, simple_setup):
        """Output concentrations lie in the convex hull of the input to machine ULP precision.

        Linear-superposition smoothing of a non-negative signal cannot create excursions
        outside ``[min(cin), max(cin)]``. Implementation-level floating-point composition
        (K-Z flux quadrature + V-coordinate weighting) introduces ULP-scale rounding noise;
        the previous +/- 1e-10 slack was 6 orders of magnitude looser than needed. A
        return value that returned the mean of cin would still satisfy these bounds; the
        next test (constant-input -> constant-output) and the mass-conservation tests
        guard against that mode separately.
        """
        cout = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )
        valid = ~np.isnan(cout)
        assert np.sum(valid) > 0, "expected at least one valid cout bin under simple setup"
        cin_min = float(np.min(simple_setup["cin"]))
        cin_max = float(np.max(simple_setup["cin"]))
        ulp_tol = 16.0 * np.finfo(cout.dtype).eps * max(abs(cin_min), abs(cin_max), 1.0)
        assert np.all(cout[valid] >= cin_min - ulp_tol)
        assert np.all(cout[valid] <= cin_max + ulp_tol)

    def test_constant_input_gives_constant_output(self):
        """Test that constant input concentration gives constant output."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
        cout_tedges = pd.date_range(start="2020-01-06", end="2020-01-15", freq="D")

        cin = np.ones(len(tedges) - 1) * 5.0  # Constant concentration
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        valid = ~np.isnan(cout)
        # With constant input, output should also be constant (after spin-up)
        np.testing.assert_allclose(cout[valid], 5.0, rtol=1e-10)

    def test_multiple_pore_volumes(self):
        """Test with multiple pore volumes (heterogeneous aquifer)."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-15", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")

        cin = np.zeros(len(tedges) - 1)
        cin[0:5] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0

        # Multiple pore volumes with corresponding travel distances
        aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
        streamline_length = np.array([80.0, 100.0, 120.0])

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Should produce valid output
        assert cout.shape == (len(cout_tedges) - 1,)
        valid = ~np.isnan(cout)
        assert np.sum(valid) > 0

        # Output should be bounded
        assert np.all(cout[valid] >= 0.0 - 1e-10)
        assert np.all(cout[valid] <= 1.0 + 1e-10)

    def test_heterogeneous_streamline_length(self):
        """Heterogeneous L (per-streamtube) yields distinct breakthrough vs single mean L.

        Per-streamtube L allows residence time tau = V_p / Q to be paired with
        the corresponding L for diffusion variance sigma^2 = 2*D*tau*L^2 in
        spatial form. A single mean L applied to all streamtubes loses this
        coupling, so the breakthrough should differ when V_p and L vary.
        """
        tedges = pd.date_range(start="2020-01-01", end="2020-02-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-02-15", freq="D")

        cin = np.zeros(len(tedges) - 1)
        cin[0:5] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
        streamline_length_hetero = np.array([80.0, 100.0, 120.0])
        # Single mean L applied to all streamtubes
        streamline_length_single = np.full(3, np.mean(streamline_length_hetero))

        kwargs = {
            "cin": cin,
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": cout_tedges,
            "aquifer_pore_volumes": aquifer_pore_volumes,
            "molecular_diffusivity": 1.0,
            "longitudinal_dispersivity": 0.0,
        }

        cout_hetero = infiltration_to_extraction(streamline_length=streamline_length_hetero, **kwargs)
        cout_single = infiltration_to_extraction(streamline_length=streamline_length_single, **kwargs)

        # Both should be valid and bounded
        valid = ~np.isnan(cout_hetero) & ~np.isnan(cout_single)
        assert np.sum(valid) > 0
        assert np.all(cout_hetero[valid] >= -1e-10)
        assert np.all(cout_single[valid] >= -1e-10)

        # Per-streamtube sigma_v = (R V/L) * sqrt(2 D_m tau + 2 alpha_L L). For the
        # heterogeneous case V/L = 5 for all three streamtubes (so sigma_v is identical
        # across streamtubes); for the single-L case V/L varies (4, 5, 6) so sigma_v
        # spreads across streamtubes. The empirically observed max-bin difference is
        # ~4.6e-3 at this resolution; the threshold below replaces the previous
        # ``not np.allclose(atol=1e-3)`` vacuous form with an explicit lower bound that
        # catches the failure mode the original wording targeted (operator collapses to
        # the single-L form).
        max_abs_diff = float(np.max(np.abs(cout_hetero[valid] - cout_single[valid])))
        assert max_abs_diff > 3e-3, (
            f"hetero and single-L breakthroughs differ by only {max_abs_diff:.4e}; "
            "expected > 3e-3 given the per-streamtube V/L variation"
        )

        # Both outputs lie in the convex hull of the input to machine ULP precision.
        cin_max = float(np.max(cin))
        ulp_tol = 16.0 * np.finfo(cout_hetero.dtype).eps * max(abs(cin_max), 1.0)
        assert np.all(cout_hetero[valid] <= cin_max + ulp_tol)
        assert np.all(cout_single[valid] <= cin_max + ulp_tol)


class TestInfiltrationToExtractionDiffusionPhysics:
    """Physics-based tests for infiltration_to_extraction with diffusion."""

    def test_symmetry_of_pulse(self):
        """Test that a symmetric pulse input produces a symmetric-ish output.

        Note: Perfect symmetry is not expected due to the nature of diffusion
        in a flowing system, but the output should be roughly centered around
        the expected arrival time.
        """
        tedges = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")

        # Narrow pulse in the middle
        cin = np.zeros(len(tedges) - 1)
        cin[10:12] = 1.0  # 2-day pulse starting day 10
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])  # 5 days residence time
        streamline_length = np.array([100.0])

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=5.0,
            longitudinal_dispersivity=0.0,
        )

        valid = ~np.isnan(cout)
        # Hard precondition: spin-up must release enough valid bins to cover the breakthrough.
        # Previously this branch was wrapped in two nested ``if np.sum(...) > 0`` guards, so
        # the assertion never fired when the operator regressed to all-NaN.
        assert np.sum(valid) >= 10, f"expected at least 10 valid bins, got {np.sum(valid)}"
        times = np.arange(len(cout))
        cout_valid = cout.copy()
        cout_valid[~valid] = 0.0
        total_mass = float(np.sum(cout_valid))
        assert total_mass > 0.0, "valid bins exist but carry zero mass"
        # Center of mass should be around day 16 (midpoint of input + 5-day residence).
        center_of_mass = float(np.sum(times * cout_valid) / total_mass)
        assert 14.0 < center_of_mass < 19.0, f"center of mass {center_of_mass:.2f} outside [14, 19]"

    def test_mass_approximately_conserved(self):
        """Test that mass is approximately conserved through transport."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")

        cin = np.zeros(len(tedges) - 1)
        cin[2:7] = 1.0  # 5-day pulse
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Mass in = sum of cin (each bin is 1 day)
        mass_in = np.sum(cin)

        # Mass out = sum of cout (excluding NaN)
        mass_out = np.nansum(cout)

        # Mass is conserved: the forward matrix rows sum to 1 for fully
        # resolved bins, and the cout grid extends beyond all breakthrough.
        np.testing.assert_allclose(mass_out, mass_in, rtol=1e-10)

    def test_retardation_delays_breakthrough(self):
        """Test that retardation factor delays the breakthrough."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-15", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-25", freq="D")

        cin = np.zeros(len(tedges) - 1)
        cin[0:3] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        # Without retardation
        cout_r1 = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
            retardation_factor=1.0,
        )

        # With retardation
        cout_r2 = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
            retardation_factor=2.0,
        )

        # Find where concentration drops below threshold (end of breakthrough)
        def last_significant(cout, threshold=0.9):
            valid = ~np.isnan(cout)
            for i in range(len(cout) - 1, -1, -1):
                if valid[i] and cout[i] > threshold:
                    return i
            return -1

        last_r1 = last_significant(cout_r1)
        last_r2 = last_significant(cout_r2)

        # Retarded breakthrough should persist longer
        assert last_r2 > last_r1

    def test_retardation_scales_mean_breakthrough_time(self):
        """Mean breakthrough time scales linearly with retardation factor.

        The first moment (centroid) of the ADE impulse response at x=L is
        R*V/Q, independent of dispersion parameters. For a pulse input at
        bin ``pulse_bin``, the centroid delay between R=2 and R=1 should
        equal (R2-R1)*V/Q.

        Uses alpha_L=1 m (Pe=L/alpha_L~100) so diffusion spreading is
        non-trivial. Both R values have nearly identical Pe because D_m is
        negligible compared to alpha_L*v_s, so discretization errors in the
        centroid cancel in the difference.
        """
        flow_rate = 100.0  # m3/day
        pore_volume = 500.0  # m3, tau_0 = V/Q = 5 days
        length = 100.0  # m
        d_m = 0.05  # m2/day
        alpha_l = 1.0  # m, Pe ~ L/alpha_L = 100
        r1 = 1.0
        r2 = 2.0
        pulse_bin = 5  # pulse at day 5-6, away from spin-up edge

        n_days = 60
        tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")

        cin = np.zeros(n_days)
        cin[pulse_bin] = 100.0
        flow = np.full(n_days, flow_rate)

        centroids = {}
        for r in [r1, r2]:
            n_out = n_days + int(r * pore_volume / flow_rate) + 20
            cout_tedges = pd.date_range("2020-01-01", periods=n_out + 1, freq="D")
            cout = infiltration_to_extraction(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([pore_volume]),
                streamline_length=np.array([length]),
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
                retardation_factor=r,
            )
            valid = ~np.isnan(cout) & (cout > 0)
            tmid = np.arange(len(cout)) + 0.5
            centroids[r] = np.sum(tmid[valid] * cout[valid]) / np.sum(cout[valid])

        expected_delay = (r2 - r1) * pore_volume / flow_rate  # 5.0 days
        actual_delay = centroids[r2] - centroids[r1]

        np.testing.assert_allclose(actual_delay, expected_delay, rtol=0.01)


class TestDiffusionMatchesApvdCombined:
    """Test diffusion physics with per-bin velocity-dependent dispersivity.

    These tests verify that the new implementation with time-varying diffusivity
    produces physically correct results. The diffusivity is computed as:
    D_L = D_m + alpha_L * v, where v is computed per output bin.
    """

    def test_cout_full_physics_with_dispersivity(self):
        """Test that diffusion produces physically correct results.

        With the new per-bin velocity-dependent dispersivity, the solution
        should:
        1. Conserve mass
        2. Have bounded concentrations
        3. Show appropriate spreading behavior
        """
        np.random.seed(42)

        # System parameters
        streamline_length = 100.0  # L [m]
        mean_apv = 10000.0  # V_mean [m³]
        std_apv = 800.0  # sigma_apv [m³]
        mean_flow = 120.0  # Q [m³/day]
        retardation = 2.0  # R [-]
        diffusivity_molecular = 1e-4  # D_m [m²/day]
        dispersivity = 1.0  # alpha_L [m]

        # Set up time bins
        n_days = 350
        tedges_ = pd.date_range("2019-12-31", periods=n_days + 1, freq="D")

        flow_ = np.full(n_days, mean_flow)
        cin_ = np.zeros(n_days)
        cin_[50] = 100.0

        # Compress bins for efficiency
        cin_itedges = np.flatnonzero(np.diff(cin_, prepend=1.0, append=1.0))
        flow_itedges = np.flatnonzero(np.diff(flow_, prepend=1.0, append=1.0))
        itedges = np.unique(np.concatenate([cin_itedges, flow_itedges]))
        tedges = tedges_[itedges]
        cin = cin_[itedges[:-1]]
        flow = flow_[itedges[:-1]]
        cout_tedges = tedges_.copy()

        # Discretization
        nbins = 5000
        streamline_lengths = np.full(nbins, streamline_length)

        # Full solution: diffusion with APVD + diffusion/dispersion
        gbins = gamma_utils.bins(mean=mean_apv, std=std_apv, n_bins=nbins)
        cout_full = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=gbins["expected_values"],
            streamline_length=streamline_lengths,
            molecular_diffusivity=diffusivity_molecular,
            longitudinal_dispersivity=dispersivity,
            retardation_factor=retardation,
        )

        # Compare in the main breakthrough region
        plot_start, plot_end = 100, 349
        cout_full_slice = cout_full[plot_start : plot_end - 1]

        # Should have valid values
        valid_mask = ~np.isnan(cout_full_slice)
        assert np.sum(valid_mask) > 100, "Should have many valid values"

        # Mass should be conserved
        mass_full = np.nansum(cout_full_slice)
        np.testing.assert_allclose(mass_full, 100.0, rtol=0.01, err_msg="Should conserve mass within 1%")

        # Concentrations should be bounded (with tolerance for numerical precision)
        assert np.all(cout_full_slice[valid_mask] >= -1e-4), "Concentrations should be non-negative"
        assert np.all(cout_full_slice[valid_mask] <= 100.0 + 1e-4), "Concentrations should not exceed input max"

        # Peak should occur at reasonable time (around mean residence time)
        # The mean residence time is mean_apv * retardation / mean_flow
        # Peak occurs around day 50 + residence_time in the full array
        # In our slice (plot_start to plot_end-1), check peak is in a reasonable range
        peak_idx = np.nanargmax(cout_full_slice)
        mean_residence_time = mean_apv * retardation / mean_flow  # ~166.7 days
        # Peak should be somewhere in the expected range of the breakthrough curve
        # Peak idx in slice should correspond to around day 50 + mean_residence_time - plot_start
        expected_peak_relative = 50 + mean_residence_time - plot_start  # ~116.7
        assert abs(peak_idx - expected_peak_relative) <= 25, (
            f"Peak at idx {peak_idx} should be near {expected_peak_relative}"
        )

    def test_zero_dispersivity_matches_molecular_only(self):
        """Test that zero dispersivity gives same result as molecular diffusion only."""
        np.random.seed(42)

        streamline_length = 100.0
        mean_apv = 5000.0
        std_apv = 500.0
        mean_flow = 100.0
        diffusivity_molecular = 0.1

        n_days = 150
        tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
        cout_tedges = tedges.copy()

        cin = np.zeros(n_days)
        cin[20] = 100.0
        flow = np.full(n_days, mean_flow)

        nbins = 1000
        gbins = gamma_utils.bins(mean=mean_apv, std=std_apv, n_bins=nbins)
        streamline_lengths = np.full(nbins, streamline_length)

        # With zero dispersivity
        cout_zero_disp = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=gbins["expected_values"],
            streamline_length=streamline_lengths,
            molecular_diffusivity=diffusivity_molecular,
            longitudinal_dispersivity=0.0,
        )

        # Should conserve mass exactly (forward matrix rows sum to 1)
        mass = np.nansum(cout_zero_disp)
        np.testing.assert_allclose(mass, 100.0, rtol=1e-10, err_msg="Mass should be conserved")

    def test_increased_dispersion_broadens_curve(self):
        """Test that higher dispersion causes broader, lower-peak breakthrough."""
        np.random.seed(42)

        streamline_length = 100.0
        mean_apv = 5000.0
        std_apv = 500.0
        mean_flow = 100.0

        # cin window: 60 days is enough for the pulse, but cout extends much
        # further so that even the longest pore volume in the gamma tail (with
        # D_m=10 dispersion) has its breakthrough fully captured. Without
        # capturing the full tail, the high-dispersion case loses mass at the
        # finite-window boundary and mass-conservation across the two cases
        # would not hold at the 1e-10 level we are testing for.
        n_days_cin = 60
        n_days_cout = 4000
        tedges = pd.date_range("2020-01-01", periods=n_days_cin + 1, freq="D")
        cout_tedges = pd.date_range("2020-01-01", periods=n_days_cout + 1, freq="D")

        cin = np.zeros(n_days_cin)
        cin[20] = 100.0  # Pulse input
        flow = np.full(n_days_cin, mean_flow)

        nbins = 1000
        gbins = gamma_utils.bins(mean=mean_apv, std=std_apv, n_bins=nbins)
        streamline_lengths = np.full(nbins, streamline_length)

        # Low dispersion
        cout_low = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=gbins["expected_values"],
            streamline_length=streamline_lengths,
            molecular_diffusivity=0.1,
            longitudinal_dispersivity=0.0,
        )

        # High dispersion
        cout_high = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=gbins["expected_values"],
            streamline_length=streamline_lengths,
            molecular_diffusivity=10.0,
            longitudinal_dispersivity=0.0,
        )

        # High dispersion should have lower peak
        peak_low = np.nanmax(cout_low)
        peak_high = np.nanmax(cout_high)
        assert peak_high < peak_low, "Higher dispersion should reduce peak concentration"

        # Mass is conserved: both should equal 100.0 (the input pulse)
        mass_low = np.nansum(cout_low)
        mass_high = np.nansum(cout_high)
        np.testing.assert_allclose(mass_low, mass_high, rtol=1e-10, err_msg="Mass should be conserved")

    def test_single_pv_matches_apvd_with_combined_std(self):
        """Test that diffusion with single pore volume matches APVD with combined std.

        This validates the physical equivalence of the spreading formulas.
        The corrected formula for mechanical dispersion has NO retardation factor:
            sigma_disp = V * sqrt(2 * alpha_L / L)
        because D_disp = alpha_L * v = alpha_L * L / tau, and D_disp * tau = alpha_L * L.
        """
        # System parameters
        streamline_length = 100.0  # L [m]
        mean_apv = 10000.0  # V_mean [m³]
        mean_flow = 120.0  # Q [m³/day]
        retardation = 2.0  # R [-]
        diffusivity_molecular = 1e-4  # D_m [m²/day]
        dispersivity = 1.0  # alpha_L [m]

        # CORRECTED formula: R cancels out for mechanical dispersion
        # sigma_diff = (V/L) * sqrt(2 * D_m * R * V / Q)  -- R stays for molecular diffusion
        # sigma_disp = V * sqrt(2 * alpha_L / L)  -- NO R for mechanical dispersion
        sigma_diff_disp = mean_apv * np.sqrt(
            2 * diffusivity_molecular * retardation / (streamline_length * mean_flow)
            + 2 * dispersivity / streamline_length
        )

        # Set up time bins
        n_days = 350
        tedges = pd.date_range("2019-12-31", periods=n_days + 1, freq="D")
        cout_tedges = tedges.copy()

        flow = np.full(n_days, mean_flow)
        cin = np.zeros(n_days)
        cin[50] = 100.0

        # diffusion with single pore volume
        cout_diffusion = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([mean_apv]),
            streamline_length=np.array([streamline_length]),
            molecular_diffusivity=diffusivity_molecular,
            longitudinal_dispersivity=dispersivity,
            retardation_factor=retardation,
        )

        # APVD with combined std (using gamma distribution)
        cout_apvd = gamma_i2e(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            mean=mean_apv,
            std=sigma_diff_disp,
            n_bins=5000,
            retardation_factor=retardation,
        )

        # Peak concentrations should match closely
        peak_diffusion = np.nanmax(cout_diffusion)
        peak_apvd = np.nanmax(cout_apvd)
        np.testing.assert_allclose(
            peak_diffusion, peak_apvd, rtol=0.05, err_msg="Peak concentrations should match with corrected formula"
        )

        # Peak timing should match
        peak_day_diffusion = np.nanargmax(cout_diffusion)
        peak_day_apvd = np.nanargmax(cout_apvd)
        assert abs(peak_day_diffusion - peak_day_apvd) <= 2, (
            f"Peak timing should match: diffusion={peak_day_diffusion}, APVD={peak_day_apvd}"
        )

        # Mass should be conserved during the fully-informed period.
        # During spin-up, the advection module normalizes by the number of
        # contributing bins rather than total bins, which preserves concentration
        # levels but means mass is not conserved until all bins are informed.
        mass_diffusion = np.nansum(cout_diffusion)
        np.testing.assert_allclose(mass_diffusion, 100.0, rtol=0.01)
        # The advection module's gamma_i2e amplifies mass by ~9.14% due to
        # spin-up: it normalizes by the number of contributing bins rather than
        # by total bins, which preserves concentration levels but inflates
        # the integrated mass during the spin-up window.
        mass_apvd = np.nansum(cout_apvd)
        np.testing.assert_allclose(mass_apvd, 100.0, rtol=0.10)


class TestExtractionToInfiltrationDiffusion:
    """Tests for the extraction_to_infiltration function with diffusion.

    These tests verify that the advection-dispersion deconvolution model
    produces physically correct results.
    """

    @pytest.fixture
    def simple_setup(self):
        """Create a simple test setup with constant flow."""
        cin_tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-02-20", freq="D")

        # Step input: concentration 1 for 5 days after residence time offset
        cout = np.zeros(len(cout_tedges) - 1)
        cout[5:10] = 1.0
        # Flow on cin grid
        flow_cin = np.ones(len(cin_tedges) - 1) * 100.0

        # Single pore volume: 500 m3 = 5 days residence time at 100 m3/day
        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        return {
            "cout": cout,
            "flow_cin": flow_cin,
            "cin_tedges": cin_tedges,
            "cout_tedges": cout_tedges,
            "aquifer_pore_volumes": aquifer_pore_volumes,
            "streamline_length": streamline_length,
        }

    def test_zero_diffusivity_matches_advection(self, simple_setup):
        """Test that zero diffusivity gives same result as pure advection.

        Both modules use comparable strict-validity boundary handling here
        (advection: ``spinup=None``; diffusion: ``spinup=None`` disables
        the 100-year tedge extension), so the comparison restricts to bins
        valid in both modules and the algebra matches exactly.
        """
        cin_advection = advection_e2i(
            cout=simple_setup["cout"],
            flow=simple_setup["flow_cin"],
            tedges=simple_setup["cin_tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            spinup=None,
        )
        cin_diffusion = extraction_to_infiltration(
            cout=simple_setup["cout"],
            flow=simple_setup["flow_cin"],
            tedges=simple_setup["cin_tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=0.0,
            spinup=None,
        )
        # Only compare values where advection is not NaN
        valid_mask = ~np.isnan(cin_advection)
        np.testing.assert_allclose(cin_advection[valid_mask], cin_diffusion[valid_mask])

    def test_small_diffusivity_close_to_advection(self, simple_setup):
        """Test that small diffusivity produces result close to advection."""
        cin_advection = advection_e2i(
            cout=simple_setup["cout"],
            flow=simple_setup["flow_cin"],
            tedges=simple_setup["cin_tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
        )
        cin_diffusion = extraction_to_infiltration(
            cout=simple_setup["cout"],
            flow=simple_setup["flow_cin"],
            tedges=simple_setup["cin_tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=0.01,
            longitudinal_dispersivity=0.0,
        )
        # With D_m=0.01 m²/day and tau=5 days, the spatial spreading is
        # sqrt(2*D_m*tau) ≈ 0.32 m on a 100 m streamline (< 0.4% of domain).
        # Use atol since cout contains step transitions where the diffused
        # values cross zero — rtol is meaningless near-zero.
        valid = ~np.isnan(cin_advection) & ~np.isnan(cin_diffusion)
        np.testing.assert_allclose(cin_advection[valid], cin_diffusion[valid], atol=0.01)

    def test_output_bounded_by_input(self, simple_setup):
        """Test that output concentrations are bounded by input range."""
        cin = extraction_to_infiltration(
            cout=simple_setup["cout"],
            flow=simple_setup["flow_cin"],
            tedges=simple_setup["cin_tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )
        valid = ~np.isnan(cin)
        # Pseudoinverse can exceed input range due to deconvolution ringing
        assert np.all(cin[valid] >= np.min(simple_setup["cout"]) - 0.1)
        assert np.all(cin[valid] <= np.max(simple_setup["cout"]) + 0.1)

    def test_constant_input_gives_constant_output(self):
        """Test that constant input concentration gives constant output."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-02-20", freq="D")

        cout = np.ones(len(cout_tedges) - 1) * 5.0  # Constant concentration
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cin = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        valid = ~np.isnan(cin)
        # With constant input, output should also be constant for well-supported bins.
        # Edge bins (where cout window doesn't cover the cin-to-cout transport lag)
        # have near-zero coefficient support, so the solver cannot recover them.
        well_supported = valid & (cin > 1.0)
        np.testing.assert_allclose(cin[well_supported], 5.0, rtol=1e-10, atol=1e-10)
        assert np.sum(well_supported) > 0.85 * np.sum(valid)

    def test_multiple_pore_volumes(self):
        """Test with multiple pore volumes (heterogeneous aquifer)."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-02-20", freq="D")

        cout = np.zeros(len(cout_tedges) - 1)
        cout[5:10] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0

        # Multiple pore volumes with corresponding travel distances
        aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
        streamline_length = np.array([80.0, 100.0, 120.0])

        cin = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Should produce valid output
        assert cin.shape == (len(tedges) - 1,)
        valid = ~np.isnan(cin)
        assert np.sum(valid) > 0

        # Multi-PV deconvolution with underdetermined system can produce
        # significant oscillations beyond input range
        assert np.all(cin[valid] >= 0.0 - 3.0)
        assert np.all(cin[valid] <= 1.0 + 3.0)


class TestExtractionToInfiltrationDiffusionPhysics:
    """Physics-based tests for extraction_to_infiltration with diffusion."""

    def test_mass_approximately_conserved(self):
        """Test that mass is approximately conserved through reconstruction."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-02-20", freq="D")

        cout = np.zeros(len(cout_tedges) - 1)
        cout[5:10] = 1.0  # 5-day pulse
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cin = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Mass in = sum of cout
        mass_in = np.sum(cout)

        # Mass out = sum of cin (excluding NaN)
        mass_out = np.nansum(cin)

        # Mass is approximately conserved through the pseudoinverse
        assert abs(mass_out - mass_in) / mass_in < 1e-6

    def test_retardation_shifts_reconstruction(self):
        """Test that retardation factor shifts the reconstruction timing."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-15", end="2020-02-25", freq="D")

        cout = np.zeros(len(cout_tedges) - 1)
        cout[10:15] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        # Without retardation
        cin_r1 = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
            retardation_factor=1.0,
        )

        # With retardation
        cin_r2 = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
            retardation_factor=2.0,
        )

        # Find the center of mass of the reconstructed signals
        def center_of_mass(c):
            valid = ~np.isnan(c) & (c > 0.01)
            if not np.any(valid):
                return np.nan
            times = np.arange(len(c))
            return np.sum(times[valid] * c[valid]) / np.sum(c[valid])

        com_r1 = center_of_mass(cin_r1)
        com_r2 = center_of_mass(cin_r2)

        # With higher retardation, the infiltration should be reconstructed earlier
        # (because it takes longer to travel through the aquifer)
        assert com_r2 < com_r1


class TestDiffusionRoundTrip:
    """Round-trip tests for diffusion: cin -> cout -> cin_reconstructed."""

    @pytest.fixture
    def roundtrip_setup(self):
        """Create a setup for round-trip testing.

        Long time series (1 year cin, 200 days cout) ensures enough interior
        bins for machine-precision comparison after excluding edge effects.
        """
        tedges = pd.date_range(start="2020-01-01", end="2021-01-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-07-30", freq="D")

        # Sinusoidal input for smooth variation
        cin = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(tedges) - 1) / 30.0)
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        return {
            "cin": cin,
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": cout_tedges,
            "aquifer_pore_volumes": aquifer_pore_volumes,
            "streamline_length": streamline_length,
        }

    def _compute_valid_mask(
        self,
        tedges,
        cout_tedges,
        aquifer_pore_volumes,
        flow,
        retardation_factor=1.0,
        effective_diffusivity=0.0,
        streamline_length=None,
    ):
        """Compute mask for valid comparison region (excluding edge effects).

        The pseudoinverse roundtrip error decays geometrically from edges.
        Two margins are needed:

        - **Advective margin**: ``max(40 bins, 8x residence time)`` for the
          advective spin-up to decay.
        - **Diffusion margin**: higher effective diffusivity D_L means more
          spreading per cin bin, worse conditioning of the forward matrix, and
          slower geometric decay of deconvolution edge errors. Empirically,
          ``15 + 50 * D_L / v`` bins suffices for errors to reach machine
          precision (validated for D_L / v up to ~2.5).
        """
        max_residence_time = np.max(aquifer_pore_volumes) * retardation_factor / np.mean(flow)
        advective_margin = max(40.0, 8.0 * max_residence_time)

        diffusion_margin = 0.0
        if effective_diffusivity > 0 and streamline_length is not None:
            mean_velocity = np.max(streamline_length) / max_residence_time
            diffusion_margin = 15.0 + 50.0 * effective_diffusivity / mean_velocity

        margin_days = max(advective_margin, diffusion_margin)

        # Forward spinup: cout bins before first breakthrough
        forward_spinup_end = tedges[0] + pd.Timedelta(days=margin_days)

        # Backward spinup: cin bins after last extraction data minus residence time
        backward_spinup_start = cout_tedges[-1] - pd.Timedelta(days=margin_days)

        # Valid region: between both spinup periods
        cin_bin_centers = tedges[:-1] + (tedges[1:] - tedges[:-1]) / 2
        return (cin_bin_centers >= forward_spinup_end) & (cin_bin_centers <= backward_spinup_start)

    def test_roundtrip_zero_diffusivity(self, roundtrip_setup):
        """Test that cin -> cout -> cin_reconstructed matches with zero diffusivity."""
        # Forward pass
        cout = infiltration_to_extraction(
            cin=roundtrip_setup["cin"],
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=0.0,
        )

        # Backward pass
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=0.0,
        )

        # Compute valid mask
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
        )

        # Compare in valid region - should be exact for zero diffusivity
        assert np.sum(valid_mask) > 10, "Should have enough valid bins for comparison"
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            roundtrip_setup["cin"][valid_mask],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_roundtrip_molecular_diffusivity(self, roundtrip_setup):
        """Test round-trip with molecular diffusivity only."""
        # Forward pass
        cout = infiltration_to_extraction(
            cin=roundtrip_setup["cin"],
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Backward pass
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Compute valid mask — D_L = D_m = 1.0 m²/day
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
            effective_diffusivity=1.0,
            streamline_length=roundtrip_setup["streamline_length"],
        )

        # Compare in valid region
        assert np.sum(valid_mask) > 10, "Should have enough valid bins for comparison"
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            roundtrip_setup["cin"][valid_mask],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_roundtrip_longitudinal_dispersivity(self, roundtrip_setup):
        """Test round-trip with longitudinal dispersivity only.

        The same forward matrix is used for both directions, so the roundtrip
        is exact up to SVD numerical precision within the valid region.
        Higher D_L requires wider margins because deconvolution edge errors
        decay more slowly.
        """
        # D_L = alpha_L * v = 1.0 * (L / tau) = 1.0 * (100 / 5) = 20 m²/day
        # Forward pass
        cout = infiltration_to_extraction(
            cin=roundtrip_setup["cin"],
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=1.0,
        )

        # Backward pass
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=1.0,
        )

        # Compute valid mask — D_L = alpha_L * v = 1.0 * 20 = 20 m²/day
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
            effective_diffusivity=20.0,
            streamline_length=roundtrip_setup["streamline_length"],
        )

        # Compare in valid region
        assert np.sum(valid_mask) > 10, "Should have enough valid bins for comparison"
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            roundtrip_setup["cin"][valid_mask],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_roundtrip_combined_diffusion_dispersion(self, roundtrip_setup):
        """Test round-trip with both molecular diffusivity and dispersivity."""
        # Forward pass
        cout = infiltration_to_extraction(
            cin=roundtrip_setup["cin"],
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=0.5,
            longitudinal_dispersivity=0.5,
        )

        # Backward pass
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=0.5,
            longitudinal_dispersivity=0.5,
        )

        # Compute valid mask — D_L = D_m + alpha_L * v = 0.5 + 0.5 * 20 = 10.5 m²/day
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
            effective_diffusivity=10.5,
            streamline_length=roundtrip_setup["streamline_length"],
        )

        # Compare in valid region
        assert np.sum(valid_mask) > 10, "Should have enough valid bins for comparison"
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            roundtrip_setup["cin"][valid_mask],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_roundtrip_multiple_pore_volumes(self):
        """Test round-trip with multiple pore volumes.

        With the combined forward-inverse approach, the same forward matrix
        is used for both directions, enabling exact roundtrip recovery even
        with multiple pore volumes.
        """
        # Long time ranges to ensure sufficient valid region after margin exclusion
        tedges = pd.date_range(start="2020-01-01", end="2021-01-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-07-30", freq="D")

        cin = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(tedges) - 1) / 30.0)
        flow = np.ones(len(tedges) - 1) * 100.0

        # Multiple pore volumes
        aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
        streamline_length = np.array([80.0, 100.0, 120.0])

        # Forward pass
        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Backward pass
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Compute valid mask — D_L = D_m = 1.0 m²/day
        valid_mask = self._compute_valid_mask(
            tedges,
            cout_tedges,
            aquifer_pore_volumes,
            flow,
            effective_diffusivity=1.0,
            streamline_length=streamline_length,
        )

        # Compare in valid region
        assert np.sum(valid_mask) > 20, "Should have enough valid bins for comparison"

        # Multiple PVs create a wider mixing pattern (each cout bin averages
        # cin at 3 different offsets), increasing the nullspace dimension.
        # The reverse_matrix target approximates cin in the nullspace but
        # cannot recover it exactly. Error ~0.9% of signal amplitude.
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            cin[valid_mask],
            rtol=5e-3,
            atol=0.01,
        )

    def test_roundtrip_with_retardation(self, roundtrip_setup):
        """Test round-trip with retardation factor > 1."""
        retardation = 2.0

        # Forward pass
        cout = infiltration_to_extraction(
            cin=roundtrip_setup["cin"],
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
            retardation_factor=retardation,
        )

        # Backward pass
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=roundtrip_setup["flow"],
            tedges=roundtrip_setup["tedges"],
            cout_tedges=roundtrip_setup["cout_tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
            retardation_factor=retardation,
        )

        # Compute valid mask with retardation — D_L = D_m = 1.0 m²/day
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
            retardation_factor=retardation,
            effective_diffusivity=1.0,
            streamline_length=roundtrip_setup["streamline_length"],
        )

        # Compare in valid region
        assert np.sum(valid_mask) > 10, "Should have enough valid bins for comparison"
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            roundtrip_setup["cin"][valid_mask],
            rtol=1e-10,
            atol=1e-10,
        )


class TestFlowWeightedDiffusion:
    """Tests that verify flow-weighted output concentrations.

    These tests use varying flow within output bins to expose the difference
    between a 2D (x, τ) rectangle average and the correct 1D volume-space
    trajectory average.  With constant flow both averages coincide, so these
    scenarios are the minimal tests that distinguish the two approaches.
    """

    @pytest.mark.parametrize("retardation_factor", [1.0, 2.7, 5.0])
    def test_zero_diffusivity_varying_flow_matches_advection(self, retardation_factor):
        """D=0 with varying flow: diffusion module must match pure advection across R values."""
        tedges = pd.date_range("2020-01-01", periods=31, freq="D")
        # Coarser output bins so that multiple flow bins fall inside one cout bin
        cout_tedges = pd.date_range("2020-01-01", periods=11, freq="3D")

        cin = np.zeros(len(tedges) - 1)
        cin[0:10] = 1.0
        # Flow varies significantly across consecutive days
        flow = 100.0 + 60.0 * np.sin(np.arange(len(cin)) * 2 * np.pi / 7)

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cout_adv = advection_i2e(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            retardation_factor=retardation_factor,
        )
        cout_diff = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=0.0,
            retardation_factor=retardation_factor,
        )

        valid = ~np.isnan(cout_adv)
        np.testing.assert_allclose(cout_adv[valid], cout_diff[valid])

    def test_mass_conservation_varying_flow(self):
        """Coefficient rows must sum to ~1 for valid bins under varying flow."""
        tedges = pd.date_range("2020-01-01", periods=31, freq="D")
        cout_tedges = pd.date_range("2020-01-01", periods=11, freq="3D")

        flow_arr = 100.0 + 60.0 * np.sin(np.arange(30) * 2 * np.pi / 7)
        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        coeff, valid = _infiltration_to_extraction_coeff_matrix(
            flow=flow_arr,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=np.array([1.0]),
            longitudinal_dispersivity=np.array([0.0]),
            retardation_factor=1.0,
        )

        row_sums = coeff[valid].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    @pytest.mark.parametrize("seed", [1, 3, 7, 42, 999])
    @pytest.mark.parametrize(
        ("d_m", "alpha_l"),
        [
            (0.0, 0.1),  # pure alpha_L (issue #162)
            (0.0, 0.3),  # pure alpha_L
            (1.0, 0.0),  # pure D_m (issue #180)
            (1.0, 0.3),  # mixed
        ],
    )
    def test_column_mass_conservation_variable_flow_dispersion(self, d_m, alpha_l, seed):
        """Volume-weighted column sums must equal infiltrated volume under
        variable flow with dispersion (any combination of D_m and alpha_L).

        Regression for issues #162 and #180. The forward kernel reports the
        Kreft-Zuber (1978) flux concentration at the outlet,

            C_F = C_R - (D_L / v) * dC_R/dx,

        which converts Bear's resident concentration into a flux concentration.
        The invariant ``sum_i W[i,j] * Q_out[i] * dt_out[i] == Q_in[j] * dt_in[j]``
        then holds at machine precision under arbitrary variable Q and for any
        combination of molecular diffusion D_m and longitudinal dispersivity
        alpha_L. Without the flux correction, Bear's leading-order kernel misses
        the dispersive boundary flux and the column sum drifts by O(1/Pe).
        """
        n = 60
        tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
        # Choose cout edges that align with tedges so dV_out is well-defined
        # in terms of the same flow array.
        cout_tedges = tedges

        rng = np.random.default_rng(seed)
        flow = 100.0 * np.exp(rng.normal(0.0, 0.3, n))  # sigma_Q/Q ~ 0.3
        aquifer_pore_volumes = np.array([1000.0])
        streamline_length = np.array([100.0])

        coeff, _ = _infiltration_to_extraction_coeff_matrix(
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=np.array([d_m]),
            longitudinal_dispersivity=np.array([alpha_l]),
            retardation_factor=1.0,
        )

        # Conservation under variable flow:
        #   sum_i W[i,j] * Q_out[i] * dt_out[i] == Q_in[j] * dt_in[j]
        # Restrict to interior cin bins whose parcels fully transit within
        # the cout window. RT ~ V/Q ~ 10 days, plus dispersion-induced tail
        # spread, leaves bins 10..41 fully captured even for the largest
        # combined-dispersion case at n=60.
        dt = np.diff(tedges) / pd.Timedelta("1D")
        v_out = flow * dt  # cout flow == flow for matched edges
        v_in_interior = flow[10:42] * 1.0

        mass_out_per_cin = (v_out[:, None] * coeff).sum(axis=0)
        ratios = mass_out_per_cin[10:42] / v_in_interior
        np.testing.assert_allclose(ratios, 1.0, atol=1e-10, rtol=0)

    def test_constant_cin_varying_flow_gives_constant_cout(self):
        """Constant cin with varying flow must produce constant cout."""
        tedges = pd.date_range("2020-01-01", periods=61, freq="D")
        cout_tedges = pd.date_range("2020-01-20", periods=11, freq="3D")

        cin = np.ones(len(tedges) - 1) * 7.0
        flow = 100.0 + 50.0 * np.sin(np.arange(len(cin)) * 2 * np.pi / 5)

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([400.0]),
            streamline_length=np.array([80.0]),
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        valid = ~np.isnan(cout)
        np.testing.assert_allclose(cout[valid], 7.0, atol=1e-12)

    @pytest.mark.parametrize(
        ("d_m", "alpha_l"),
        [
            (1.5, 0.0),  # pure molecular: D_t = D_m * tau
            (0.0, 0.3),  # pure microdispersion: D_t = alpha_L * xi
            (1.5, 0.3),  # mixed: both terms active
        ],
    )
    def test_cfrac_mean_volume_vs_quad(self, d_m, alpha_l):
        """`_cfrac_mean_volume` must match scipy.integrate.quad for C_F = C_R + flux correction.

        The Kreft-Zuber flux concentration at the outlet is
        ``C_F(L, V; t_j) = C_R(L, V; t_j) + (D_L(t(V))/v(t(V))) * Gaussian density``,
        where ``C_R = 0.5 * erfc((L-xi)/(2*sqrt(D_t)))`` is Bear's resident
        concentration, ``D_t = D_m * tau + alpha_L * xi``, and the flux
        correction uses the instantaneous local dispersion coefficient
        ``D_L = D_m + alpha_L * v(t)``.

        Tests several narrow cout cells covering pre-breakthrough, the
        breakthrough front, and post-breakthrough. Cells are chosen narrow
        enough that 16-point GL quadrature in V-space is precise to ~1e-12.
        """
        cum_cin = np.array([0.0, 50.0, 130.0, 220.0, 330.0, 500.0])
        tedges_days = np.array([0.0, 0.5, 1.4, 2.3, 3.4, 5.0])  # variable Q

        r_vpv = 200.0
        sl = 50.0
        v_pore = 200.0  # = r_vpv when R = 1 (no retardation in this test)

        # Several narrow cout cells: pre, around, and post breakthrough for j=0
        cum_cout = np.array([60.0, 130.0, 200.0, 260.0, 330.0, 420.0, 480.0])

        delta_vol = cum_cout[:, None] - cum_cin[None, :] - r_vpv
        step_widths = delta_vol / r_vpv * sl

        result = _cfrac_mean_volume(
            step_widths=step_widths,
            cumulative_volume_at_cout_tedges=cum_cout,
            cumulative_volume_at_cin_tedges=cum_cin,
            tedges_days=tedges_days,
            molecular_diffusivity=d_m,
            longitudinal_dispersivity=alpha_l,
            r_vpv=r_vpv,
            streamline_len=sl,
            aquifer_pore_volume=v_pore,
        )

        # Fluid velocity per flow bin
        q_per_bin = np.diff(cum_cin) / np.diff(tedges_days)
        v_per_bin = q_per_bin * sl / v_pore

        def _make_integrand(v_j_, t_j_):
            def integrand(v):
                x = (v - v_j_ - r_vpv) * sl / r_vpv
                xi = x + sl
                t_v = float(np.interp(v, cum_cin, tedges_days))
                tau = max(t_v - t_j_, 0.0)
                dt_val = d_m * tau + alpha_l * xi
                if dt_val <= 0:
                    return 0.0
                # Identify which flow bin t_v falls in (left-closed convention)
                k = min(int(np.searchsorted(tedges_days, t_v, side="right") - 1), len(v_per_bin) - 1)
                v_obs = v_per_bin[max(k, 0)]
                cr = 0.5 * special.erfc((sl - xi) / (2.0 * np.sqrt(dt_val)))
                fc = (
                    (d_m + alpha_l * v_obs)
                    / v_obs
                    * np.exp(-((sl - xi) ** 2) / (4.0 * dt_val))
                    / np.sqrt(4.0 * np.pi * dt_val)
                )
                return cr + fc

            return integrand

        # The function only integrates within the flow-data range [V_cin[0], V_cin[-1]];
        # beyond that, t(V) is undefined and there is no flow bin to integrate over,
        # so the reference quad must clip the upper limit to V_cin[-1] as well.
        n_cout_bins = len(cum_cout) - 1
        for i in range(n_cout_bins):
            v_lo, v_hi = cum_cout[i], cum_cout[i + 1]
            for j in range(len(cum_cin)):
                v_j = cum_cin[j]
                t_j = tedges_days[j]
                lower = max(v_lo, v_j, cum_cin[0])
                upper = min(v_hi, cum_cin[-1])
                if lower >= upper:
                    np.testing.assert_allclose(result[i, j], 0.0, atol=1e-13)
                    continue
                ref, _ = integrate.quad(_make_integrand(v_j, t_j), lower, upper, limit=500, epsabs=1e-13)
                ref_mean = ref / (v_hi - v_lo)
                np.testing.assert_allclose(result[i, j], ref_mean, atol=1e-12)


class TestGammaExtractionToInfiltrationDiffusion:
    """Tests for gamma_extraction_to_infiltration in the diffusion module.

    Verifies that the gamma convenience wrapper correctly delegates to
    extraction_to_infiltration and that the combined gamma + dispersion
    deconvolution produces physically correct results.
    """

    @pytest.fixture
    def gamma_setup(self):
        """Create test data with long time series for gamma transport.

        Uses a 1-year cin grid and 200-day cout window. With mean pore
        volume 500 m3 and flow 100 m3/day, the mean residence time is
        5 days (R=1) or 10 days (R=2).
        """
        tedges = pd.date_range(start="2020-01-01", end="2021-01-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-07-30", freq="D")

        cin = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(tedges) - 1) / 30.0)
        flow = np.ones(len(tedges) - 1) * 100.0

        return {
            "cin": cin,
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": cout_tedges,
            "mean": 500.0,
            "std": 100.0,
            "n_bins": 20,
            "streamline_length": 100.0,
        }

    def test_zero_cout_gives_zero_cin(self):
        """Zero extraction concentration must produce zero infiltration."""
        tedges = pd.date_range(start="2020-01-01", end="2020-06-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-05-01", freq="D")
        cout = np.zeros(len(cout_tedges) - 1)
        flow = np.ones(len(tedges) - 1) * 100.0

        cin = gamma_extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            mean=500.0,
            std=100.0,
            n_bins=10,
            streamline_length=100.0,
            molecular_diffusivity=1e-4,
            longitudinal_dispersivity=1.0,
        )

        valid = ~np.isnan(cin)
        assert np.sum(valid) > 20
        np.testing.assert_allclose(cin[valid], 0.0, atol=1e-14)

    def test_constant_cout_gives_constant_cin(self, gamma_setup):
        """Constant extraction concentration must produce constant infiltration."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.full(n_cout, 7.0)

        cin = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            n_bins=gamma_setup["n_bins"],
            streamline_length=gamma_setup["streamline_length"],
            molecular_diffusivity=1e-4,
            longitudinal_dispersivity=0.5,
        )

        valid = ~np.isnan(cin)
        # Well-supported interior bins should recover the constant exactly
        well_supported = valid & (cin > 1.0)
        assert np.sum(well_supported) > 50
        np.testing.assert_allclose(cin[well_supported], 7.0, rtol=1e-10, atol=1e-10)

    def test_roundtrip_single_pore_volume_limit(self, gamma_setup):
        """Roundtrip with very narrow gamma (single-PV-like) recovers cin.

        A gamma distribution with very small std relative to mean behaves
        like a single pore volume. The roundtrip should recover the original
        signal to machine precision in the well-supported interior.
        """
        # Very narrow gamma -> effectively single pore volume
        narrow_std = 1.0  # std/mean = 0.002
        diffusion_kwargs = {
            "mean": gamma_setup["mean"],
            "std": narrow_std,
            "n_bins": 5,
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": 1e-4,
            "longitudinal_dispersivity": 0.0,
        }

        cout = diffusion_gamma_i2e(
            cin=gamma_setup["cin"],
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        # Compute valid interior mask (exclude edges)
        valid = ~np.isnan(cin_recovered) & ~np.isnan(
            np.interp(
                np.arange(len(cin_recovered)),
                np.arange(len(cout)) + 5,  # offset for cout_tedges start
                np.where(np.isnan(cout), 0, 1),
            )
        )
        # Skip first and last 50 bins for edge effects
        interior = np.zeros(len(cin_recovered), dtype=bool)
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 100:
            interior[valid_indices[50:-50]] = True

        assert np.sum(interior) > 20
        np.testing.assert_allclose(
            cin_recovered[interior],
            gamma_setup["cin"][interior],
            rtol=1e-9,
            atol=1e-9,
        )

    def test_roundtrip_moderate_gamma(self, gamma_setup):
        """Roundtrip with moderate gamma spread recovers cin in interior."""
        diffusion_kwargs = {
            "mean": gamma_setup["mean"],
            "std": gamma_setup["std"],
            "n_bins": gamma_setup["n_bins"],
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": 1e-4,
            "longitudinal_dispersivity": 0.0,
        }

        cout = diffusion_gamma_i2e(
            cin=gamma_setup["cin"],
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        valid = ~np.isnan(cin_recovered)
        valid_indices = np.where(valid)[0]
        interior = np.zeros(len(cin_recovered), dtype=bool)
        if len(valid_indices) > 100:
            interior[valid_indices[50:-50]] = True

        assert np.sum(interior) > 20
        np.testing.assert_allclose(
            cin_recovered[interior],
            gamma_setup["cin"][interior],
            rtol=5e-3,
            atol=0.01,
        )

    def test_roundtrip_with_retardation(self, gamma_setup):
        """Roundtrip with retardation factor > 1 recovers cin in interior."""
        retardation = 2.0
        diffusion_kwargs = {
            "mean": gamma_setup["mean"],
            "std": 1.0,  # narrow gamma for near-exact recovery
            "n_bins": 5,
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": 1e-4,
            "longitudinal_dispersivity": 0.0,
            "retardation_factor": retardation,
        }

        cout = diffusion_gamma_i2e(
            cin=gamma_setup["cin"],
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        valid = ~np.isnan(cin_recovered)
        valid_indices = np.where(valid)[0]
        interior = np.zeros(len(cin_recovered), dtype=bool)
        if len(valid_indices) > 100:
            interior[valid_indices[50:-50]] = True

        assert np.sum(interior) > 10
        np.testing.assert_allclose(
            cin_recovered[interior],
            gamma_setup["cin"][interior],
            rtol=1e-9,
            atol=1e-9,
        )

    def test_alpha_beta_matches_mean_std(self, gamma_setup):
        """Alpha/beta parameterization gives identical result to mean/std."""
        alpha, beta = mean_std_loc_to_alpha_beta(mean=gamma_setup["mean"], std=gamma_setup["std"])
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.full(n_cout, 5.0)
        cout[30:50] = 10.0

        common_kwargs = {
            "cout": cout,
            "flow": gamma_setup["flow"],
            "tedges": gamma_setup["tedges"],
            "cout_tedges": gamma_setup["cout_tedges"],
            "n_bins": gamma_setup["n_bins"],
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": 1e-4,
            "longitudinal_dispersivity": 0.5,
        }

        cin_mean_std = gamma_extraction_to_infiltration(
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            **common_kwargs,
        )
        cin_alpha_beta = gamma_extraction_to_infiltration(
            alpha=alpha,
            beta=beta,
            **common_kwargs,
        )

        valid = ~np.isnan(cin_mean_std) & ~np.isnan(cin_alpha_beta)
        assert np.sum(valid) > 50
        np.testing.assert_allclose(cin_mean_std[valid], cin_alpha_beta[valid], atol=0.0)

    def test_matches_explicit_extraction_to_infiltration(self, gamma_setup):
        """Gamma wrapper must produce identical result to explicit call."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.full(n_cout, 5.0)
        cout[20:40] = 8.0

        bins = gamma_utils.bins(
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            n_bins=gamma_setup["n_bins"],
        )

        cin_gamma = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            n_bins=gamma_setup["n_bins"],
            streamline_length=gamma_setup["streamline_length"],
            molecular_diffusivity=1e-4,
            longitudinal_dispersivity=0.5,
        )

        cin_explicit = extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            aquifer_pore_volumes=bins["expected_values"],
            streamline_length=np.full(gamma_setup["n_bins"], gamma_setup["streamline_length"]),
            molecular_diffusivity=1e-4,
            longitudinal_dispersivity=0.5,
        )

        valid = ~np.isnan(cin_gamma) & ~np.isnan(cin_explicit)
        assert np.sum(valid) > 50
        np.testing.assert_allclose(cin_gamma[valid], cin_explicit[valid], atol=0.0)

    def test_dispersion_warning_with_multiple_bins(self, gamma_setup):
        """Multiple pore volumes with dispersivity emits UserWarning."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.ones(n_cout) * 5.0

        with pytest.warns(
            UserWarning,
            match=r"Using multiple aquifer_pore_volumes with non-zero longitudinal_dispersivity",
        ):
            gamma_extraction_to_infiltration(
                cout=cout,
                flow=gamma_setup["flow"],
                tedges=gamma_setup["tedges"],
                cout_tedges=gamma_setup["cout_tedges"],
                mean=gamma_setup["mean"],
                std=gamma_setup["std"],
                n_bins=gamma_setup["n_bins"],
                streamline_length=gamma_setup["streamline_length"],
                molecular_diffusivity=0.0,
                longitudinal_dispersivity=1.0,
            )

    def test_suppress_dispersion_warning(self, gamma_setup):
        """suppress_dispersion_warning=True silences the warning."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.ones(n_cout) * 5.0

        with warn_module.catch_warnings():
            warn_module.simplefilter("error")
            gamma_extraction_to_infiltration(
                cout=cout,
                flow=gamma_setup["flow"],
                tedges=gamma_setup["tedges"],
                cout_tedges=gamma_setup["cout_tedges"],
                mean=gamma_setup["mean"],
                std=gamma_setup["std"],
                n_bins=gamma_setup["n_bins"],
                streamline_length=gamma_setup["streamline_length"],
                molecular_diffusivity=0.0,
                longitudinal_dispersivity=1.0,
                suppress_dispersion_warning=True,
            )

    def test_loc_zero_matches_legacy(self, gamma_setup):
        """With loc=0 the gamma wrapper must exactly match the legacy (mean, std) call."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.full(n_cout, 5.0)
        cout[20:40] = 8.0

        common_kwargs = {
            "cout": cout,
            "flow": gamma_setup["flow"],
            "tedges": gamma_setup["tedges"],
            "cout_tedges": gamma_setup["cout_tedges"],
            "mean": gamma_setup["mean"],
            "std": gamma_setup["std"],
            "n_bins": gamma_setup["n_bins"],
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": 1e-4,
            "longitudinal_dispersivity": 0.5,
        }

        cin_default = gamma_extraction_to_infiltration(**common_kwargs)
        cin_loc_zero = gamma_extraction_to_infiltration(loc=0.0, **common_kwargs)
        np.testing.assert_array_equal(cin_default, cin_loc_zero)

    def test_roundtrip_with_loc(self, gamma_setup):
        """Forward->reverse roundtrip with loc > 0 recovers cin in the interior."""
        diffusion_kwargs = {
            "mean": gamma_setup["mean"] + 200.0,  # mean shifted to keep excess mean reasonable
            "std": 1.0,  # near-delta
            "loc": 200.0,
            "n_bins": 5,
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": 1e-4,
            "longitudinal_dispersivity": 0.0,
        }

        cout = diffusion_gamma_i2e(
            cin=gamma_setup["cin"],
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )
        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        valid = ~np.isnan(cin_recovered)
        valid_indices = np.where(valid)[0]
        interior = np.zeros(len(cin_recovered), dtype=bool)
        if len(valid_indices) > 100:
            interior[valid_indices[50:-50]] = True

        assert np.sum(interior) > 20
        np.testing.assert_allclose(
            cin_recovered[interior],
            gamma_setup["cin"][interior],
            rtol=1e-9,
            atol=1e-9,
        )

    def test_roundtrip_with_dispersivity(self, gamma_setup):
        """Roundtrip with non-zero dispersivity recovers cin."""
        diffusion_kwargs = {
            "mean": gamma_setup["mean"],
            "std": 1.0,  # narrow gamma
            "n_bins": 5,
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": 1e-4,
            "longitudinal_dispersivity": 1.0,
            "suppress_dispersion_warning": True,
        }

        cout = diffusion_gamma_i2e(
            cin=gamma_setup["cin"],
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            **diffusion_kwargs,
        )

        valid = ~np.isnan(cin_recovered)
        valid_indices = np.where(valid)[0]
        interior = np.zeros(len(cin_recovered), dtype=bool)
        if len(valid_indices) > 100:
            interior[valid_indices[50:-50]] = True

        assert np.sum(interior) > 10
        np.testing.assert_allclose(
            cin_recovered[interior],
            gamma_setup["cin"][interior],
            rtol=1e-9,
            atol=1e-9,
        )


# =============================================================================
# Negative-flow rejection and zero-flow invariance
# =============================================================================


def test_infiltration_to_extraction_accepts_zero_flow_without_warnings():
    """flow == 0 is accepted; no runtime warnings emitted."""
    tedges = pd.date_range(start="2020-01-01", periods=201, freq="D")
    cin = np.ones(200)
    flow = np.full(200, 100.0)
    flow[100] = 0.0

    with warn_module.catch_warnings():
        warn_module.simplefilter("error", RuntimeWarning)
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=np.array([80.0]),
            molecular_diffusivity=np.array([0.03]),
            longitudinal_dispersivity=np.array([0.0]),
        )


def test_diffusion_spinup_none_disables_tedge_extension():
    """spinup=None disables the 100-year warm-start extension.

    Under the default (spinup='constant') the diffusion module extends
    tedges by +/-100 years and produces no left-edge NaN. Under spinup=None
    the extension is skipped, so cout bins whose source windows fall
    outside the user-supplied tedges become NaN -- matching the strict-
    validity behavior of the advection module.
    """
    n = 20
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.full(n, 5.0)
    flow = np.full(n, 100.0)
    aquifer_pore_volumes = np.array([500.0])
    streamline_length = np.array([100.0])

    cout_default = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=np.array([0.0]),
        longitudinal_dispersivity=np.array([0.0]),
    )
    cout_strict = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=np.array([0.0]),
        longitudinal_dispersivity=np.array([0.0]),
        spinup=None,
    )

    n_nan_default = int(np.sum(np.isnan(cout_default)))
    n_nan_strict = int(np.sum(np.isnan(cout_strict)))
    assert n_nan_strict > n_nan_default, "spinup=None should produce more NaN than constant warm-start"


def test_diffusion_spinup_float_raises():
    """Diffusion module does not implement the float-threshold mode."""
    n = 10
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    with pytest.raises(NotImplementedError, match="spinup"):
        infiltration_to_extraction(
            cin=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=np.array([100.0]),
            molecular_diffusivity=np.array([0.0]),
            longitudinal_dispersivity=np.array([0.0]),
            spinup=0.5,
        )


# =============================================================================
# Validator helper: parametrized snapshot pinning every ValueError branch of
# _validate_diffusion_inputs. Verbatim messages from the prior duplicated
# prologues; new branches (retardation_factor>=1 in fwd+rev, flow>=0 in rev)
# are the issue #187 omission fixes.
# =============================================================================


def _good_diffusion_inputs():
    """Baseline good-input dict that passes _validate_diffusion_inputs silently."""
    n = 5
    n_pv = 2
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    return {
        "tedges": tedges,
        "flow": np.full(n, 100.0),
        "aquifer_pore_volumes": np.full(n_pv, 500.0),
        "streamline_length": np.full(n_pv, 100.0),
        "molecular_diffusivity": np.full(n_pv, 0.0),
        "longitudinal_dispersivity": np.full(n_pv, 0.0),
        "retardation_factor": 1.0,
        "suppress_dispersion_warning": True,
    }


def test_validate_diffusion_inputs_silent_on_good_input_forward():
    """No exception when the forward (cin) inputs are valid."""
    kwargs = _good_diffusion_inputs()
    n = len(kwargs["flow"])
    _validate_diffusion_inputs(**kwargs, cin_values=np.ones(n))


def test_validate_diffusion_inputs_silent_on_good_input_reverse():
    """No exception when the reverse (cout + cout_tedges) inputs are valid."""
    kwargs = _good_diffusion_inputs()
    n = len(kwargs["flow"])
    _validate_diffusion_inputs(**kwargs, cout_values=np.ones(n), cout_tedges=kwargs["tedges"])


@pytest.mark.parametrize(
    ("path", "mutate", "match_regex"),
    [
        # ---------------- forward path ----------------
        (
            "forward",
            lambda k: {**k, "cin_values": np.ones(len(k["flow"]) + 1)},
            r"tedges must have one more element than cin",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "cin_values": np.ones(len(k["flow"])),
                "flow": np.full(len(k["flow"]) + 1, 100.0),
            },
            r"tedges must have one more element than flow",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "aquifer_pore_volumes": np.full(2, 500.0),
                "streamline_length": np.full(3, 100.0),
                "molecular_diffusivity": np.full(2, 0.0),
                "longitudinal_dispersivity": np.full(2, 0.0),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"aquifer_pore_volumes and streamline_length must have the same length",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "molecular_diffusivity": np.full(3, 0.0),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"molecular_diffusivity must be a scalar or have same length as aquifer_pore_volumes",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "longitudinal_dispersivity": np.full(3, 0.0),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"longitudinal_dispersivity must be a scalar or have same length as aquifer_pore_volumes",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "molecular_diffusivity": np.array([-1.0, 0.0]),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"molecular_diffusivity must be non-negative",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "longitudinal_dispersivity": np.array([-1.0, 0.0]),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"longitudinal_dispersivity must be non-negative",
        ),
        (
            "forward",
            lambda k: {**k, "cin_values": np.array([1.0, np.nan, 1.0, 1.0, 1.0])},
            r"cin contains NaN values, which are not allowed",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "flow": np.array([100.0, np.nan, 100.0, 100.0, 100.0]),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"flow contains NaN values, which are not allowed",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "flow": np.array([100.0, -50.0, 100.0, 100.0, 100.0]),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"flow must be non-negative \(negative flow not supported\)",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "aquifer_pore_volumes": np.array([500.0, 0.0]),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"aquifer_pore_volumes must be positive",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "streamline_length": np.array([100.0, 0.0]),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"streamline_length must be positive",
        ),
        # NEW: retardation_factor < 1 (omission fix)
        (
            "forward",
            lambda k: {**k, "retardation_factor": 0.5, "cin_values": np.ones(len(k["flow"]))},
            r"retardation_factor must be >= 1\.0",
        ),
        # ---------------- reverse path ----------------
        (
            "reverse",
            lambda k: {
                **k,
                "flow": np.full(len(k["flow"]) + 1, 100.0),
                "cout_values": np.ones(len(k["flow"])),
                "cout_tedges": k["tedges"],
            },
            r"tedges must have one more element than flow",
        ),
        (
            "reverse",
            lambda k: {
                **k,
                "cout_values": np.ones(len(k["flow"]) + 1),
                "cout_tedges": k["tedges"],
            },
            r"cout_tedges must have one more element than cout",
        ),
        (
            "reverse",
            lambda k: {**k, "cout_values": np.array([1.0, np.nan, 1.0, 1.0, 1.0]), "cout_tedges": k["tedges"]},
            r"cout contains NaN values, which are not allowed",
        ),
        # NEW: flow >= 0 in reverse (omission fix)
        (
            "reverse",
            lambda k: {
                **k,
                "flow": np.array([100.0, -50.0, 100.0, 100.0, 100.0]),
                "cout_values": np.ones(len(k["flow"])),
                "cout_tedges": k["tedges"],
            },
            r"flow must be non-negative \(negative flow not supported\)",
        ),
        # NEW: retardation_factor < 1 in reverse (omission fix)
        (
            "reverse",
            lambda k: {
                **k,
                "retardation_factor": 0.5,
                "cout_values": np.ones(len(k["flow"])),
                "cout_tedges": k["tedges"],
            },
            r"retardation_factor must be >= 1\.0",
        ),
    ],
)
def test_validate_diffusion_inputs_raises_on_bad_input(path, mutate, match_regex):
    """Each ValueError branch raises with the exact historical message string."""
    del path  # parametrize id only
    bad = mutate(_good_diffusion_inputs())
    with pytest.raises(ValueError, match=match_regex):
        _validate_diffusion_inputs(**bad)
