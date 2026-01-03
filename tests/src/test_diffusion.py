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
    _erf_mean_space_time,
    extraction_to_infiltration,
    infiltration_to_extraction,
)


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
        # Should be close but not identical
        # Use atol for near-zero values where rtol would be too strict
        valid = ~np.isnan(cout_advection) & ~np.isnan(cout_diffusion)
        np.testing.assert_allclose(cout_advection[valid], cout_diffusion[valid], rtol=0.1, atol=0.01)

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
            retardation_factor=2.0,
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
        # With larger diffusivity, the breakthrough curve should be more spread out
        # This means the maximum should be lower and the tails should be higher
        valid = ~np.isnan(cout_small_d) & ~np.isnan(cout_large_d)
        max_small = np.max(cout_small_d[valid])
        max_large = np.max(cout_large_d[valid])
        assert max_large <= max_small  # More spreading = lower peak

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
        np.testing.assert_allclose(cout[valid], 5.0, rtol=1e-3)

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
                streamline_length=np.array([100.0]),
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

        # Mass should be approximately conserved (within 20% for this test)
        # Some loss is expected due to boundary effects
        assert abs(mass_out - mass_in) / mass_in < 0.2

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


class TestErfMeanSpaceTimeAnalytical:
    """Tests for _erf_mean_space_time against known analytical solutions.

    The function computes the mean of erf(x/(2√(D*t))) over space-time cells.
    """

    def test_against_numerical_double_integration(self):
        """Compare against scipy.integrate.dblquad."""
        diffusivity = 1.0
        xedges = np.array([0.5, 1.5])
        tedges = np.array([1.0, 2.0])

        def integrand(x, t, diff=diffusivity):
            if t <= 0:
                return 0.0
            return special.erf(x / (2 * np.sqrt(diff * t)))

        integral, _ = integrate.dblquad(
            integrand, tedges[0], tedges[1], xedges[0], xedges[1], epsabs=1e-10, epsrel=1e-10
        )
        dx = xedges[1] - xedges[0]
        dt = tedges[1] - tedges[0]
        expected = integral / (dx * dt)

        result = _erf_mean_space_time(xedges, tedges, diffusivity)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_symmetric_edges_around_zero_in_x(self):
        """Mean over symmetric x interval around 0 should be 0."""
        xedges = np.array([-1.0, 1.0])
        tedges = np.array([1.0, 2.0])
        result = _erf_mean_space_time(xedges, tedges, diffusivity=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_large_positive_x(self):
        """For large positive x, mean erf should approach 1."""
        xedges = np.array([100.0, 200.0])
        tedges = np.array([1.0, 2.0])
        result = _erf_mean_space_time(xedges, tedges, diffusivity=1.0)
        np.testing.assert_allclose(result, 1.0, rtol=1e-4)

    def test_asymptotic_cutoff_matches_full_computation(self):
        """Test that asymptotic cutoff produces same results as full computation."""
        # Create a range of cells: some near the front, some far away
        n_cells = 50
        xedges = np.linspace(-20, 20, n_cells + 1)
        tedges = np.linspace(1, 5, n_cells + 1)
        diffusivity = 1.0

        result_full = _erf_mean_space_time(xedges, tedges, diffusivity)
        result_cutoff = _erf_mean_space_time(xedges, tedges, diffusivity, asymptotic_cutoff_sigma=4.0)

        # Results should be very close (cutoff at 4 sigma gives error < 1e-8)
        np.testing.assert_allclose(result_cutoff, result_full, rtol=1e-6, atol=1e-6)

    def test_asymptotic_cutoff_handles_edge_cases(self):
        """Test asymptotic cutoff with dx=0, dt=0, and mixed cells."""
        # Mix of edge cases - xedges and tedges must be sorted per cell
        xedges = np.array([0.0, 0.0, 10.0, 20.0, 25.0, 30.0])  # dx=0 first cell
        tedges = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 4.0])  # dt=0 third cell
        diffusivity = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        result_full = _erf_mean_space_time(xedges, tedges, diffusivity)
        result_cutoff = _erf_mean_space_time(xedges, tedges, diffusivity, asymptotic_cutoff_sigma=3.0)

        np.testing.assert_allclose(result_cutoff, result_full, rtol=1e-5, atol=1e-5)

    def test_asymptotic_cutoff_cell_straddling_zero_not_optimized(self):
        """Test that cells straddling x=0 are not assigned asymptotic values."""
        # Cell that spans from negative to positive x should not be optimized
        xedges = np.array([-5.0, 5.0])
        tedges = np.array([1.0, 2.0])
        diffusivity = 1.0

        result_full = _erf_mean_space_time(xedges, tedges, diffusivity)
        result_cutoff = _erf_mean_space_time(xedges, tedges, diffusivity, asymptotic_cutoff_sigma=0.1)

        # Should get same result since cell straddles x=0
        np.testing.assert_allclose(result_cutoff, result_full, rtol=1e-10)


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
        _tedges = pd.date_range("2019-12-31", periods=n_days + 1, freq="D")

        _flow = np.full(n_days, mean_flow)
        _cin = np.zeros(n_days)
        _cin[50] = 100.0

        # Compress bins for efficiency
        cin_itedges = np.flatnonzero(np.diff(_cin, prepend=1.0, append=1.0))
        flow_itedges = np.flatnonzero(np.diff(_flow, prepend=1.0, append=1.0))
        itedges = np.unique(np.concatenate([cin_itedges, flow_itedges]))
        tedges = _tedges[itedges]
        cin = _cin[itedges[:-1]]
        flow = _flow[itedges[:-1]]
        cout_tedges = _tedges.copy()

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
        assert np.all(cout_full_slice[valid_mask] >= -1e-9), "Concentrations should be non-negative"
        assert np.all(cout_full_slice[valid_mask] <= 100.0 + 1e-9), "Concentrations should not exceed input max"

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

        # Should conserve mass
        mass = np.nansum(cout_zero_disp)
        np.testing.assert_allclose(mass, 100.0, rtol=0.05, err_msg="Mass should be conserved")

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

        # Mass should be conserved (approximately)
        mass_low = np.nansum(cout_low)
        mass_high = np.nansum(cout_high)
        np.testing.assert_allclose(mass_low, mass_high, rtol=0.1, err_msg="Mass should be approximately conserved")

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

        # Mass should be conserved
        mass_diffusion = np.nansum(cout_diffusion)
        mass_apvd = np.nansum(cout_apvd)
        np.testing.assert_allclose(mass_diffusion, 100.0, rtol=0.01)
        np.testing.assert_allclose(mass_apvd, 100.0, rtol=0.01)


class TestExtractionToInfiltrationDiffusion:
    """Tests for the extraction_to_infiltration function with diffusion.

    These tests verify that the advection-dispersion deconvolution model
    produces physically correct results.
    """

    @pytest.fixture
    def simple_setup(self):
        """Create a simple test setup with constant flow."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-02-20", freq="D")

        # Step input: concentration 1 for 5 days after residence time offset
        cout = np.zeros(len(cout_tedges) - 1)
        cout[5:10] = 1.0
        flow = np.ones(len(cout_tedges) - 1) * 100.0  # 100 m3/day

        # Single pore volume: 500 m3 = 5 days residence time at 100 m3/day
        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        return {
            "cout": cout,
            "flow": flow,
            "tedges": cout_tedges,
            "cin_tedges": tedges,
            "aquifer_pore_volumes": aquifer_pore_volumes,
            "streamline_length": streamline_length,
        }

    def test_zero_diffusivity_matches_advection(self, simple_setup):
        """Test that zero diffusivity gives same result as pure advection."""
        cin_advection = advection_e2i(
            cout=simple_setup["cout"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cin_tedges=simple_setup["cin_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
        )
        cin_diffusion = extraction_to_infiltration(
            cout=simple_setup["cout"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cin_tedges=simple_setup["cin_tedges"],
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
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cin_tedges=simple_setup["cin_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
        )
        cin_diffusion = extraction_to_infiltration(
            cout=simple_setup["cout"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cin_tedges=simple_setup["cin_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=0.01,
            longitudinal_dispersivity=0.0,
        )
        # Should be close but not identical
        valid = ~np.isnan(cin_advection) & ~np.isnan(cin_diffusion)
        np.testing.assert_allclose(cin_advection[valid], cin_diffusion[valid], rtol=0.1, atol=0.01)

    def test_output_bounded_by_input(self, simple_setup):
        """Test that output concentrations are bounded by input range."""
        cin = extraction_to_infiltration(
            cout=simple_setup["cout"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cin_tedges=simple_setup["cin_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )
        valid = ~np.isnan(cin)
        # Output should be between min and max of input (plus small tolerance for numerics)
        assert np.all(cin[valid] >= np.min(simple_setup["cout"]) - 1e-10)
        assert np.all(cin[valid] <= np.max(simple_setup["cout"]) + 1e-10)

    def test_constant_input_gives_constant_output(self):
        """Test that constant input concentration gives constant output."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-02-20", freq="D")

        cout = np.ones(len(cout_tedges) - 1) * 5.0  # Constant concentration
        flow = np.ones(len(cout_tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cin = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=cout_tedges,
            cin_tedges=tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        valid = ~np.isnan(cin)
        # With constant input, output should also be constant
        np.testing.assert_allclose(cin[valid], 5.0, rtol=1e-3)

    def test_multiple_pore_volumes(self):
        """Test with multiple pore volumes (heterogeneous aquifer)."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-02-20", freq="D")

        cout = np.zeros(len(cout_tedges) - 1)
        cout[5:10] = 1.0
        flow = np.ones(len(cout_tedges) - 1) * 100.0

        # Multiple pore volumes with corresponding travel distances
        aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
        streamline_length = np.array([80.0, 100.0, 120.0])

        cin = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=cout_tedges,
            cin_tedges=tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Should produce valid output
        assert cin.shape == (len(tedges) - 1,)
        valid = ~np.isnan(cin)
        assert np.sum(valid) > 0

        # Output should be bounded
        assert np.all(cin[valid] >= 0.0 - 1e-10)
        assert np.all(cin[valid] <= 1.0 + 1e-10)

    def test_input_validation(self, simple_setup):
        """Test that invalid inputs raise appropriate errors."""
        # Negative molecular_diffusivity
        with pytest.raises(ValueError, match="molecular_diffusivity must be non-negative"):
            extraction_to_infiltration(
                cout=simple_setup["cout"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cin_tedges=simple_setup["cin_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                molecular_diffusivity=-1.0,
                longitudinal_dispersivity=0.0,
            )

        # Negative longitudinal_dispersivity
        with pytest.raises(ValueError, match="longitudinal_dispersivity must be non-negative"):
            extraction_to_infiltration(
                cout=simple_setup["cout"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cin_tedges=simple_setup["cin_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                molecular_diffusivity=1.0,
                longitudinal_dispersivity=-1.0,
            )

        # Mismatched pore volumes and travel distances
        with pytest.raises(ValueError, match="same length"):
            extraction_to_infiltration(
                cout=simple_setup["cout"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cin_tedges=simple_setup["cin_tedges"],
                aquifer_pore_volumes=np.array([500.0, 600.0]),
                streamline_length=np.array([100.0]),
                molecular_diffusivity=1.0,
                longitudinal_dispersivity=0.0,
            )

        # NaN in cout
        cout_with_nan = simple_setup["cout"].copy()
        cout_with_nan[2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            extraction_to_infiltration(
                cout=cout_with_nan,
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cin_tedges=simple_setup["cin_tedges"],
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
        flow = np.ones(len(cout_tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cin = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=cout_tedges,
            cin_tedges=tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Mass in = sum of cout
        mass_in = np.sum(cout)

        # Mass out = sum of cin (excluding NaN)
        mass_out = np.nansum(cin)

        # Mass should be approximately conserved (within 20%)
        assert abs(mass_out - mass_in) / mass_in < 0.2

    def test_retardation_shifts_reconstruction(self):
        """Test that retardation factor shifts the reconstruction timing."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-15", end="2020-02-25", freq="D")

        cout = np.zeros(len(cout_tedges) - 1)
        cout[10:15] = 1.0
        flow = np.ones(len(cout_tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        # Without retardation
        cin_r1 = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=cout_tedges,
            cin_tedges=tedges,
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
            tedges=cout_tedges,
            cin_tedges=tedges,
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
        """Create a setup for round-trip testing."""
        tedges = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-02-20", freq="D")

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

    def _compute_valid_mask(self, tedges, cout_tedges, aquifer_pore_volumes, flow, retardation_factor=1.0):
        """Compute mask for valid comparison region (excluding spinup periods)."""
        max_residence_time = np.max(aquifer_pore_volumes) * retardation_factor / np.mean(flow)

        # Forward spinup: cout bins before first breakthrough
        forward_spinup_end = tedges[0] + pd.Timedelta(days=max_residence_time)

        # Backward spinup: cin bins after last extraction data minus residence time
        backward_spinup_start = cout_tedges[-1] - pd.Timedelta(days=max_residence_time)

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

        # Backward pass (use same flow for simplicity)
        flow_backward = np.ones(len(roundtrip_setup["cout_tedges"]) - 1) * 100.0
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_backward,
            tedges=roundtrip_setup["cout_tedges"],
            cin_tedges=roundtrip_setup["tedges"],
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
        flow_backward = np.ones(len(roundtrip_setup["cout_tedges"]) - 1) * 100.0
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_backward,
            tedges=roundtrip_setup["cout_tedges"],
            cin_tedges=roundtrip_setup["tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Compute valid mask
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
        )

        # Compare in valid region
        assert np.sum(valid_mask) > 10, "Should have enough valid bins for comparison"
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            roundtrip_setup["cin"][valid_mask],
            rtol=0.02,
        )

    def test_roundtrip_longitudinal_dispersivity(self, roundtrip_setup):
        """Test round-trip with longitudinal dispersivity only."""
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
        flow_backward = np.ones(len(roundtrip_setup["cout_tedges"]) - 1) * 100.0
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_backward,
            tedges=roundtrip_setup["cout_tedges"],
            cin_tedges=roundtrip_setup["tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=0.0,
            longitudinal_dispersivity=1.0,
        )

        # Compute valid mask
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
        )

        # Compare in valid region
        assert np.sum(valid_mask) > 10, "Should have enough valid bins for comparison"
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            roundtrip_setup["cin"][valid_mask],
            rtol=0.02,
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
        flow_backward = np.ones(len(roundtrip_setup["cout_tedges"]) - 1) * 100.0
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_backward,
            tedges=roundtrip_setup["cout_tedges"],
            cin_tedges=roundtrip_setup["tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=0.5,
            longitudinal_dispersivity=0.5,
        )

        # Compute valid mask
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
        )

        # Compare in valid region
        assert np.sum(valid_mask) > 10, "Should have enough valid bins for comparison"
        np.testing.assert_allclose(
            cin_reconstructed[valid_mask],
            roundtrip_setup["cin"][valid_mask],
            rtol=0.02,
        )

    def test_roundtrip_multiple_pore_volumes(self):
        """Test round-trip with multiple pore volumes.

        With multiple pore volumes, the inverse problem is more complex because
        the forward function mixes signals from different flow paths. The explicit
        weights construction provides an approximate inverse. This test verifies
        that the mean is preserved and the reconstruction is reasonable.
        """
        # Use longer time ranges to ensure sufficient valid region
        tedges = pd.date_range(start="2020-01-01", end="2020-04-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-15", end="2020-03-15", freq="D")

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
        flow_backward = np.ones(len(cout_tedges) - 1) * 100.0
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_backward,
            tedges=cout_tedges,
            cin_tedges=tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
        )

        # Compute valid mask using max pore volume
        max_residence_time = np.max(aquifer_pore_volumes) / np.mean(flow)
        forward_spinup_end = tedges[0] + pd.Timedelta(days=max_residence_time)
        backward_spinup_start = cout_tedges[-1] - pd.Timedelta(days=max_residence_time)
        cin_bin_centers = tedges[:-1] + (tedges[1:] - tedges[:-1]) / 2
        valid_mask = (cin_bin_centers >= forward_spinup_end) & (cin_bin_centers <= backward_spinup_start)

        # Compare in valid region
        assert np.sum(valid_mask) > 20, "Should have enough valid bins for comparison"

        # For multiple pore volumes, the reconstruction is approximate
        # Check that mean is preserved
        np.testing.assert_allclose(
            np.mean(cin_reconstructed[valid_mask]),
            np.mean(cin[valid_mask]),
            rtol=0.15,
        )

        # Check that values are physically reasonable (bounded)
        assert np.all(cin_reconstructed[valid_mask] >= 0)
        assert np.all(cin_reconstructed[valid_mask] <= 10)  # Input ranges from 3 to 7

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
        flow_backward = np.ones(len(roundtrip_setup["cout_tedges"]) - 1) * 100.0
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_backward,
            tedges=roundtrip_setup["cout_tedges"],
            cin_tedges=roundtrip_setup["tedges"],
            aquifer_pore_volumes=roundtrip_setup["aquifer_pore_volumes"],
            streamline_length=roundtrip_setup["streamline_length"],
            molecular_diffusivity=1.0,
            longitudinal_dispersivity=0.0,
            retardation_factor=retardation,
        )

        # Compute valid mask with retardation
        valid_mask = self._compute_valid_mask(
            roundtrip_setup["tedges"],
            roundtrip_setup["cout_tedges"],
            roundtrip_setup["aquifer_pore_volumes"],
            roundtrip_setup["flow"],
            retardation_factor=retardation,
        )

        # Compare in valid region
        if np.sum(valid_mask) > 5:
            np.testing.assert_allclose(
                cin_reconstructed[valid_mask],
                roundtrip_setup["cin"][valid_mask],
                rtol=0.02,
            )
        else:
            # If valid region is too small, just check that reconstruction is reasonable
            valid = ~np.isnan(cin_reconstructed)
            assert np.sum(valid) > 0
            # Mean should be close to input mean
            np.testing.assert_allclose(
                np.mean(cin_reconstructed[valid]),
                np.mean(roundtrip_setup["cin"]),
                rtol=0.1,
            )
