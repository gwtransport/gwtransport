import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction as advection_i2e
from gwtransport.diffusion2 import infiltration_to_extraction


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
            diffusivity=0.0,
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
            diffusivity=0.01,
        )
        # Should be close but not identical
        valid = ~np.isnan(cout_advection) & ~np.isnan(cout_diffusion)
        np.testing.assert_allclose(cout_advection[valid], cout_diffusion[valid], rtol=0.1)

    def test_larger_diffusivity_more_spreading(self, simple_setup):
        """Test that larger diffusivity causes more spreading."""
        cout_small_d = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            diffusivity=0.1,
            retardation_factor=2.0,
        )
        cout_large_d = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            diffusivity=10.0,
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
            diffusivity=1.0,
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
            diffusivity=1.0,
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
            diffusivity=1.0,
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
        # Negative diffusivity
        with pytest.raises(ValueError, match="diffusivity must be non-negative"):
            infiltration_to_extraction(
                cin=simple_setup["cin"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                diffusivity=-1.0,
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
                diffusivity=1.0,
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
                diffusivity=1.0,
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
            diffusivity=5.0,
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
            diffusivity=1.0,
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
            diffusivity=1.0,
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
            diffusivity=1.0,
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
