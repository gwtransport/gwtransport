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
    _erf_integral_space,
    _erf_mean_volume,
    _infiltration_to_extraction_coeff_matrix,
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
        """Test that output concentrations are bounded by input range."""
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
        # Output should be between min and max of input (plus small tolerance for numerics)
        assert np.all(cout[valid] >= np.min(simple_setup["cin"]) - 1e-10)
        assert np.all(cout[valid] <= np.max(simple_setup["cin"]) + 1e-10)

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

        # Heterogeneous and single-L results should be physically distinct,
        # demonstrating the per-streamtube L coupling matters.
        assert not np.allclose(cout_hetero[valid], cout_single[valid], atol=1e-3)

        # Both outputs should be bounded by the input range (convex combination).
        assert np.all(cout_hetero[valid] <= np.max(cin) + 1e-10)
        assert np.all(cout_single[valid] <= np.max(cin) + 1e-10)

    def test_input_validation(self, simple_setup):
        """Test that invalid inputs raise appropriate errors."""
        # Negative molecular_diffusivity
        with pytest.raises(ValueError, match="molecular_diffusivity must be non-negative"):
            infiltration_to_extraction(
                cin=simple_setup["cin"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                molecular_diffusivity=-1.0,
                longitudinal_dispersivity=0.0,
            )

        # Negative longitudinal_dispersivity
        with pytest.raises(ValueError, match="longitudinal_dispersivity must be non-negative"):
            infiltration_to_extraction(
                cin=simple_setup["cin"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                molecular_diffusivity=1.0,
                longitudinal_dispersivity=-1.0,
            )

        # Mismatched pore volumes and travel distances
        with pytest.raises(ValueError, match="same length"):
            infiltration_to_extraction(
                cin=simple_setup["cin"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=np.array([500.0, 600.0]),
                streamline_length=np.array([100.0, 200.0, 300.0]),
                molecular_diffusivity=1.0,
                longitudinal_dispersivity=0.0,
            )

        # NaN in cin
        cin_with_nan = simple_setup["cin"].copy()
        cin_with_nan[2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            infiltration_to_extraction(
                cin=cin_with_nan,
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                molecular_diffusivity=1.0,
                longitudinal_dispersivity=0.0,
            )


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
        if np.sum(valid) > 0:
            # The center of mass should be around day 15-17 (10-12 + 5 days residence)
            times = np.arange(len(cout))
            cout_valid = cout.copy()
            cout_valid[~valid] = 0
            if np.sum(cout_valid) > 0:
                center_of_mass = np.sum(times * cout_valid) / np.sum(cout_valid)
                # Center should be around day 16 (midpoint of input + residence time)
                assert 14 < center_of_mass < 19

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

        n_days = 200
        tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
        cout_tedges = tedges.copy()

        cin = np.zeros(n_days)
        cin[20] = 100.0  # Pulse input
        flow = np.full(n_days, mean_flow)

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
        """Test that zero diffusivity gives same result as pure advection."""
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
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=0.0,
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

    def test_input_validation(self, simple_setup):
        """Test that invalid inputs raise appropriate errors."""
        # Negative molecular_diffusivity
        with pytest.raises(ValueError, match="molecular_diffusivity must be non-negative"):
            extraction_to_infiltration(
                cout=simple_setup["cout"],
                flow=simple_setup["flow_cin"],
                tedges=simple_setup["cin_tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                molecular_diffusivity=-1.0,
                longitudinal_dispersivity=0.0,
            )

        # Negative longitudinal_dispersivity
        with pytest.raises(ValueError, match="longitudinal_dispersivity must be non-negative"):
            extraction_to_infiltration(
                cout=simple_setup["cout"],
                flow=simple_setup["flow_cin"],
                tedges=simple_setup["cin_tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                molecular_diffusivity=1.0,
                longitudinal_dispersivity=-1.0,
            )

        # Mismatched pore volumes and travel distances
        with pytest.raises(ValueError, match="same length"):
            extraction_to_infiltration(
                cout=simple_setup["cout"],
                flow=simple_setup["flow_cin"],
                tedges=simple_setup["cin_tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=np.array([500.0, 600.0]),
                streamline_length=np.array([100.0, 200.0, 300.0]),
                molecular_diffusivity=1.0,
                longitudinal_dispersivity=0.0,
            )

        # NaN in cout
        cout_with_nan = simple_setup["cout"].copy()
        cout_with_nan[2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extraction_to_infiltration(
                cout=cout_with_nan,
                flow=simple_setup["flow_cin"],
                tedges=simple_setup["cin_tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                molecular_diffusivity=1.0,
                longitudinal_dispersivity=0.0,
            )


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

    def test_zero_diffusivity_varying_flow_matches_advection(self):
        """D=0 with varying flow: diffusion module must match pure advection."""
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
            asymptotic_cutoff_sigma=3.0,
        )

        row_sums = coeff[valid].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

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

    def test_volume_mean_capped_matches_space_integral(self):
        """For fully capped cells, _erf_mean_volume must match _erf_integral_space."""
        n_cout_edges = 6
        n_cin_edges = 4
        rng = np.random.default_rng(42)

        # Build monotonically increasing cumulative volumes
        cum_cout = np.cumsum(rng.uniform(80, 120, n_cout_edges))
        cum_cin = np.cumsum(rng.uniform(80, 120, n_cin_edges))
        tedges_days = np.cumsum(np.concatenate([[0], rng.uniform(0.8, 1.2, n_cin_edges - 1)]))

        r_vpv = 500.0
        sl = 100.0

        # Build step_widths: (n_cout_edges, n_cin_edges)
        delta_vol = cum_cout[:, None] - cum_cin[None, :] - r_vpv
        step_widths = delta_vol / r_vpv * sl

        # Use very small RT so all cells are fully capped
        rt_at_cin_edges = np.full(n_cin_edges, 0.1)
        raw_time = np.full((n_cout_edges, n_cin_edges), 1000.0)

        diffusivity = rng.uniform(0.5, 2.0, n_cout_edges - 1)

        result = _erf_mean_volume(
            step_widths=step_widths,
            raw_time=raw_time,
            rt_at_cin_edges=rt_at_cin_edges,
            diffusivity=diffusivity,
            cumulative_volume_at_cout_tedges=cum_cout,
            cumulative_volume_at_cin_tedges=cum_cin,
            tedges_days=tedges_days,
            r_vpv=r_vpv,
            streamline_len=sl,
            asymptotic_cutoff_sigma=3.0,
        )

        # Reference: _erf_integral_space at fixed t=RT for each cell
        x_lo = step_widths[:-1]
        x_hi = step_widths[1:]
        dx = x_hi - x_lo
        n_cout_bins = n_cout_edges - 1

        for i in range(n_cout_bins):
            for j in range(n_cin_edges):
                d = diffusivity[i]
                rt = rt_at_cin_edges[j]
                h_hi = _erf_integral_space(np.array([x_hi[i, j]]), diffusivity=d, t=np.array([rt]))[0]
                h_lo = _erf_integral_space(np.array([x_lo[i, j]]), diffusivity=d, t=np.array([rt]))[0]
                expected = (h_hi - h_lo) / dx[i, j]
                np.testing.assert_allclose(result[i, j], expected, atol=1e-14)

    def test_volume_mean_uncapped_vs_quad(self):
        """Uncapped GL quadrature must match scipy.integrate.quad reference."""
        n_cin_edges = 3  # 3 flow bins
        cum_cin = np.array([0.0, 100.0, 250.0])
        tedges_days = np.array([0.0, 1.0, 2.5])  # varying flow: Q=100, Q=100

        r_vpv = 200.0
        sl = 50.0
        d = 1.5
        # cout edges straddle the boundary: V ∈ [50, 180]
        cum_cout = np.array([50.0, 180.0])

        # RT large enough that the cell is fully uncapped
        rt_at_cin_edges = np.array([100.0, 100.0, 100.0])

        delta_vol = cum_cout[:, None] - cum_cin[None, :] - r_vpv
        step_widths = delta_vol / r_vpv * sl

        raw_time = np.full((len(cum_cout), n_cin_edges), 0.5)  # << RT → uncapped

        result = _erf_mean_volume(
            step_widths=step_widths,
            raw_time=raw_time,
            rt_at_cin_edges=rt_at_cin_edges,
            diffusivity=np.array([d]),
            cumulative_volume_at_cout_tedges=cum_cout,
            cumulative_volume_at_cin_tedges=cum_cin,
            tedges_days=tedges_days,
            r_vpv=r_vpv,
            streamline_len=sl,
            asymptotic_cutoff_sigma=None,
        )

        # Reference via scipy.integrate.quad
        v_lo, v_hi = cum_cout[0], cum_cout[1]

        for j in range(n_cin_edges):
            v_j = cum_cin[j]
            t_j = tedges_days[j]
            rt_j = rt_at_cin_edges[j]

            def _make_integrand(v_j_, t_j_, rt_j_):
                def integrand(v):
                    x = (v - v_j_ - r_vpv) * sl / r_vpv
                    t_v = np.interp(v, cum_cin, tedges_days)
                    tau = min(max(t_v - t_j_, 0.0), rt_j_)
                    if tau <= 0 or d <= 0:
                        return np.sign(x)
                    return special.erf(x / (2.0 * np.sqrt(d * tau)))

                return integrand

            ref, _ = integrate.quad(_make_integrand(v_j, t_j, rt_j), v_lo, v_hi, limit=200)
            ref_mean = ref / (v_hi - v_lo)
            np.testing.assert_allclose(result[0, j], ref_mean, atol=1e-13)

    def test_volume_mean_partially_capped_vs_quad(self):
        """Partially capped cell (uncapped→capped mid-cell) must match quad."""
        n_cin_edges = 3  # 3 flow bins
        cum_cin = np.array([0.0, 100.0, 250.0])
        tedges_days = np.array([0.0, 1.0, 2.5])

        r_vpv = 200.0
        sl = 50.0
        d = 1.5
        # cout bin spans V ∈ [50, 180]
        cum_cout = np.array([50.0, 180.0])

        # RT chosen so that raw_time[lo] < RT < raw_time[hi] for the cell,
        # creating a partially capped cell that transitions mid-bin.
        rt_at_cin_edges = np.array([0.8, 0.8, 0.8])

        delta_vol = cum_cout[:, None] - cum_cin[None, :] - r_vpv
        step_widths = delta_vol / r_vpv * sl

        # raw_time: lo edge uncapped (0.3 < 0.8), hi edge capped (1.2 > 0.8)
        raw_time = np.array([[0.3, 0.3, 0.3], [1.2, 1.2, 1.2]])

        result = _erf_mean_volume(
            step_widths=step_widths,
            raw_time=raw_time,
            rt_at_cin_edges=rt_at_cin_edges,
            diffusivity=np.array([d]),
            cumulative_volume_at_cout_tedges=cum_cout,
            cumulative_volume_at_cin_tedges=cum_cin,
            tedges_days=tedges_days,
            r_vpv=r_vpv,
            streamline_len=sl,
            asymptotic_cutoff_sigma=None,
        )

        # Reference via scipy.integrate.quad
        v_lo, v_hi = cum_cout[0], cum_cout[1]

        for j in range(n_cin_edges):
            v_j = cum_cin[j]
            t_j = tedges_days[j]
            rt_j = rt_at_cin_edges[j]

            def _make_integrand_partial(v_j_, t_j_, rt_j_):
                def integrand(v):
                    x = (v - v_j_ - r_vpv) * sl / r_vpv
                    t_v = np.interp(v, cum_cin, tedges_days)
                    tau = min(max(t_v - t_j_, 0.0), rt_j_)
                    if tau <= 0 or d <= 0:
                        return np.sign(x)
                    return special.erf(x / (2.0 * np.sqrt(d * tau)))

                return integrand

            ref, _ = integrate.quad(_make_integrand_partial(v_j, t_j, rt_j), v_lo, v_hi, limit=200)
            ref_mean = ref / (v_hi - v_lo)
            np.testing.assert_allclose(result[0, j], ref_mean, atol=1e-13)


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

        with pytest.warns(UserWarning, match="multiple.*pore.*volumes|velocity heterogeneity"):
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
