import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special

from gwtransport import advection
from gwtransport import gamma as gamma_utils
from gwtransport.advection import infiltration_to_extraction as advection_i2e
from gwtransport.diffusion2 import (
    _erf_integral_space,
    _erf_integral_space_time,
    _erf_integral_space_time_pointwise,
    _erf_integral_time,
    _erf_mean_space,
    _erf_mean_space_time,
    _erf_mean_time,
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


class TestErfIntegralSpaceAnalytical:
    """Tests for _erf_integral_space against known analytical solutions.

    The function computes the integral of erf(a*x) from 0 to x:
    ∫₀ˣ erf(a*ξ) dξ = x*erf(a*x) + (exp(-(a*x)²) - 1) / (a*√π)
    """

    def test_at_zero(self):
        """Integral from 0 to 0 should be 0."""
        x = np.array([0.0])
        result = _erf_integral_space(x, a=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_symmetry(self):
        """Test integral behavior for negative x.

        For the integral ∫₀ˣ erf(a*ξ) dξ with x<0:
        ∫₀^{-b} erf(a*ξ) dξ = -∫₀^b erf(-a*ξ) dξ = ∫₀^b erf(a*ξ) dξ (since erf is odd)
        So F(-x) = F(x), the function is symmetric (even) in x.
        """
        x = np.array([1.0, 2.0, 3.0])
        result_pos = _erf_integral_space(x, a=1.0)
        result_neg = _erf_integral_space(-x, a=1.0)
        # Actually F(-x) = -F(x) when we consider the integration direction
        # Let's verify against numerical integration
        for i, x_val in enumerate(x):
            expected_pos, _ = integrate.quad(special.erf, 0, x_val)
            expected_neg, _ = integrate.quad(special.erf, 0, -x_val)
            np.testing.assert_allclose(result_pos[i], expected_pos, rtol=1e-10)
            np.testing.assert_allclose(result_neg[i], expected_neg, rtol=1e-10)

    def test_derivative_is_erf(self):
        """The derivative of the integral should be erf(a*x)."""
        a = 1.0
        x = np.linspace(0.1, 3.0, 50)
        result = _erf_integral_space(x, a=a)

        # Numerical derivative
        dx = x[1] - x[0]
        numerical_derivative = np.gradient(result, dx)

        # Analytical derivative: erf(a*x)
        expected = special.erf(a * x)

        # Should match (allowing for numerical derivative error at boundaries)
        np.testing.assert_allclose(numerical_derivative[5:-5], expected[5:-5], rtol=0.01)

    def test_against_numerical_integration(self):
        """Compare against scipy.integrate.quad for various x values."""
        a = 1.5
        x_values = np.array([0.5, 1.0, 2.0, 3.0])

        for x_val in x_values:
            expected, _ = integrate.quad(lambda xi: special.erf(a * xi), 0, x_val)
            result = _erf_integral_space(np.array([x_val]), a=a)
            np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_a_zero_returns_zero(self):
        """When a=0, erf(0)=0, so integral should be 0."""
        x = np.array([1.0, 2.0, 5.0])
        result = _erf_integral_space(x, a=0.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_large_x_asymptotic(self):
        """For large x, erf(a*x) → 1, so integral ≈ x - 1/(a*√π)."""
        a = 1.0
        x = np.array([10.0, 20.0, 50.0])
        result = _erf_integral_space(x, a=a)
        # Asymptotic: x - 1/(a*√π)
        expected = x - 1.0 / (a * np.sqrt(np.pi))
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_vectorized_a(self):
        """Test with array of a values."""
        x = np.array([0.5, 1.0, 1.5, 2.0])
        a = np.array([0.5, 1.0, 2.0])
        result = _erf_integral_space(x, a=a)

        # Shape should be (len(a), len(x))
        assert result.shape == (3, 4)

        # Verify each row
        for i, a_val in enumerate(a):
            expected = _erf_integral_space(x, a=a_val)
            np.testing.assert_allclose(result[i, :], expected, rtol=1e-10)


class TestErfMeanSpaceAnalytical:
    """Tests for _erf_mean_space against known analytical solutions.

    The function computes the mean of erf(a*x) between edges.
    """

    def test_symmetric_edges_around_zero(self):
        """Mean of erf over symmetric interval around 0 should be 0."""
        edges = np.array([-1.0, 1.0])
        result = _erf_mean_space(edges, a=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_positive_edges(self):
        """Mean over [0, x] should equal integral/x."""
        edges = np.array([0.0, 2.0])
        a = 1.0
        result = _erf_mean_space(edges, a=a)

        # Mean = (∫₀² erf(a*x) dx) / 2
        integral, _ = integrate.quad(lambda x: special.erf(a * x), 0, 2)
        expected = integral / 2.0
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_against_numerical_integration(self):
        """Compare against scipy.integrate.quad for various edge pairs."""
        a = 0.8
        edge_pairs = [
            np.array([0.0, 1.0]),
            np.array([1.0, 3.0]),
            np.array([-1.0, 2.0]),
            np.array([0.5, 0.7]),
        ]

        for edges in edge_pairs:
            integral, _ = integrate.quad(lambda x: special.erf(a * x), edges[0], edges[1])
            dx = edges[1] - edges[0]
            expected = integral / dx
            result = _erf_mean_space(edges, a=a)
            np.testing.assert_allclose(result[0], expected, rtol=1e-10)

    def test_large_positive_edges(self):
        """For large positive edges, mean erf should approach 1."""
        edges = np.array([100.0, 200.0])
        result = _erf_mean_space(edges, a=1.0)
        np.testing.assert_allclose(result, 1.0, rtol=1e-6)

    def test_large_negative_edges(self):
        """For large negative edges, mean erf should approach -1."""
        edges = np.array([-200.0, -100.0])
        result = _erf_mean_space(edges, a=1.0)
        np.testing.assert_allclose(result, -1.0, rtol=1e-6)

    def test_vectorized_a(self):
        """Test with array of a values."""
        edges = np.array([0.0, 1.0, 2.0])
        a = np.array([0.5, 1.0, 2.0])
        result = _erf_mean_space(edges, a=a)

        # Shape should be (len(a), len(edges)-1)
        assert result.shape == (3, 2)

        # Verify each row
        for i, a_val in enumerate(a):
            expected = _erf_mean_space(edges, a=a_val)
            np.testing.assert_allclose(result[i, :], expected, rtol=1e-10)


class TestErfIntegralTimeAnalytical:
    """Tests for _erf_integral_time against known analytical solutions.

    The function computes ∫₀ᵗ erf(x/(2√(D*τ))) dτ.
    """

    def test_at_zero_time(self):
        """Integral at t=0 should be 0."""
        t = np.array([0.0])
        result = _erf_integral_time(t, x=1.0, diffusivity=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_at_zero_x(self):
        """Integral with x=0 should be 0 (erf(0)=0)."""
        t = np.array([1.0, 2.0, 5.0])
        result = _erf_integral_time(t, x=0.0, diffusivity=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_against_numerical_integration(self):
        """Compare against scipy.integrate.quad."""
        x = 2.0
        diffusivity = 1.5
        t_values = np.array([0.5, 1.0, 2.0, 5.0])

        for t_val in t_values:
            # Numerical integration
            def integrand(tau, x_val=x, diff=diffusivity):
                if tau <= 0:
                    return np.sign(x_val) if x_val != 0 else 0
                return special.erf(x_val / (2 * np.sqrt(diff * tau)))

            expected, _ = integrate.quad(integrand, 0, t_val)
            result = _erf_integral_time(np.array([t_val]), x=x, diffusivity=diffusivity)
            np.testing.assert_allclose(result[0], expected, rtol=1e-8)

    def test_positive_x_positive_result(self):
        """For positive x, integral should be positive."""
        t = np.array([1.0, 5.0, 10.0])
        result = _erf_integral_time(t, x=1.0, diffusivity=1.0)
        assert np.all(result > 0)

    def test_negative_x_negative_result(self):
        """For negative x, integral should be negative."""
        t = np.array([1.0, 5.0, 10.0])
        result = _erf_integral_time(t, x=-1.0, diffusivity=1.0)
        assert np.all(result < 0)

    def test_vectorized_x(self):
        """Test with array of x values."""
        t = np.array([0.5, 1.0, 2.0])
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        diffusivity = 1.0
        result = _erf_integral_time(t, x=x, diffusivity=diffusivity)

        # Shape should be (len(t), len(x))
        assert result.shape == (3, 4)

        # Verify each column
        for j, x_val in enumerate(x):
            expected = _erf_integral_time(t, x=x_val, diffusivity=diffusivity)
            np.testing.assert_allclose(result[:, j], expected, rtol=1e-10)


class TestErfMeanTimeAnalytical:
    """Tests for _erf_mean_time against known analytical solutions.

    The function computes the mean of erf(x/(2√(D*t))) between time edges.
    """

    def test_against_numerical_integration(self):
        """Compare against scipy.integrate.quad."""
        x = 1.5
        diffusivity = 2.0
        tedges = np.array([0.5, 1.0, 2.0, 5.0])

        for i in range(len(tedges) - 1):
            t0, t1 = tedges[i], tedges[i + 1]

            def integrand(t, x_val=x, diff=diffusivity):
                return special.erf(x_val / (2 * np.sqrt(diff * t)))

            integral, _ = integrate.quad(integrand, t0, t1)
            expected = integral / (t1 - t0)

            result = _erf_mean_time(np.array([t0, t1]), x=x, diffusivity=diffusivity)
            np.testing.assert_allclose(result[0], expected, rtol=1e-8)

    def test_symmetric_x(self):
        """Mean erf should be antisymmetric in x."""
        tedges = np.array([1.0, 3.0])
        diffusivity = 1.0
        x_val = 2.0

        result_pos = _erf_mean_time(tedges, x=x_val, diffusivity=diffusivity)
        result_neg = _erf_mean_time(tedges, x=-x_val, diffusivity=diffusivity)
        np.testing.assert_allclose(result_neg, -result_pos, rtol=1e-10)

    def test_large_time_approaches_zero(self):
        """For very large times, erf(x/(2√(D*t))) → 0."""
        # Use much larger times to get closer to zero
        tedges = np.array([10000.0, 20000.0])
        result = _erf_mean_time(tedges, x=1.0, diffusivity=1.0)
        np.testing.assert_allclose(result, 0.0, atol=0.01)

    def test_vectorized_x(self):
        """Test with array of x values."""
        tedges = np.array([0.5, 1.0, 2.0])
        x = np.array([0.5, 1.0, 2.0])
        diffusivity = 1.0
        result = _erf_mean_time(tedges, x=x, diffusivity=diffusivity)

        # Shape should be (len(tedges)-1, len(x))
        assert result.shape == (2, 3)

        # Verify each column
        for j, x_val in enumerate(x):
            expected = _erf_mean_time(tedges, x=x_val, diffusivity=diffusivity)
            np.testing.assert_allclose(result[:, j], expected, rtol=1e-10)


class TestErfIntegralSpaceTimeAnalytical:
    """Tests for _erf_integral_space_time against known analytical solutions.

    The function computes the double integral:
    F(x,t) = ∫₀ᵗ ∫₀ˣ erf(ξ/(2√(D*τ))) dξ dτ
    """

    def test_at_zero_x(self):
        """Integral at x=0 should be 0."""
        result = _erf_integral_space_time(np.array([0.0]), np.array([1.0, 5.0]), diffusivity=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_at_zero_t(self):
        """Integral at t=0 should be 0."""
        result = _erf_integral_space_time(np.array([1.0, 2.0]), np.array([0.0]), diffusivity=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-14)

    def test_symmetry_in_x(self):
        """Double integral should be symmetric in x: F(-x,t) = F(x,t)."""
        x = np.array([1.0, 2.0])
        t = np.array([1.0, 2.0])
        diffusivity = 1.0

        result_pos = _erf_integral_space_time(x, t, diffusivity)
        result_neg = _erf_integral_space_time(-x, t, diffusivity)
        np.testing.assert_allclose(result_neg, result_pos, rtol=1e-10)

    def test_against_numerical_double_integration(self):
        """Compare against scipy.integrate.dblquad."""
        diffusivity = 1.0
        x_val = 1.5
        t_val = 2.0

        def integrand(xi, tau, diff=diffusivity):
            if tau <= 0:
                return 0.0
            return special.erf(xi / (2 * np.sqrt(diff * tau)))

        expected, _ = integrate.dblquad(integrand, 0, t_val, 0, x_val, epsabs=1e-10, epsrel=1e-10)

        result = _erf_integral_space_time(np.array([x_val]), np.array([t_val]), diffusivity)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_pointwise_matches_meshgrid(self):
        """Pointwise function should match meshgrid function at same points."""
        x = np.array([1.0, 2.0, 3.0])
        t = np.array([0.5, 1.0, 2.0])
        diffusivity = 1.5

        # Pointwise evaluation
        result_pointwise = _erf_integral_space_time_pointwise(x, t, diffusivity)

        # Meshgrid evaluation - get diagonal
        result_meshgrid = _erf_integral_space_time(x, t, diffusivity)
        # Meshgrid returns shape (len(t), len(x)), diagonal is where t[i] matches x[i]
        result_diag = np.array([result_meshgrid[i, i] for i in range(len(x))])

        np.testing.assert_allclose(result_pointwise, result_diag, rtol=1e-10)


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

    def test_paired_mode_consistency(self):
        """Test that paired mode gives correct results."""
        xedges = np.array([0.0, 1.0, 2.0])
        tedges = np.array([1.0, 2.0, 3.0])
        diffusivity = 1.0

        # Non-paired gives (n_t, n_x) = (2, 2) grid
        result_grid = _erf_mean_space_time(xedges, tedges, diffusivity, paired=False)

        # Paired gives diagonal: 2 cells
        result_paired = _erf_mean_space_time(xedges, tedges, diffusivity, paired=True)

        # Paired[i] should equal grid[i, i]
        assert result_grid.shape == (2, 2)
        assert result_paired.shape == (2,)
        np.testing.assert_allclose(result_paired, np.diag(result_grid), rtol=1e-10)


class TestDiffusionMatchesApvdCombined:
    """Test that diffusion2 output matches APVD with combined std (notebook 5 style).

    This test replicates the comparison from example notebook 5, where
    `cout_full[plot_start:plot_end-1]` should produce similar results to
    `cout_apvd_combined[plot_start:plot_end-1]`.
    """

    def test_cout_full_matches_cout_apvd_combined(self):
        """Test that diffusion2 matches APVD with combined std for constant flow.

        This test reproduces the key comparison from example notebook 5.
        With constant flow, the full diffusion2 solution and the APVD
        approximation with combined standard deviation should produce
        similar breakthrough curves.
        """
        np.random.seed(42)

        # System parameters (from notebook 5)
        streamline_length = 100.0  # L [m]
        mean_apv = 10000.0  # V_mean [m³]
        std_apv = 800.0  # sigma_apv [m³]
        mean_flow = 120.0  # Q [m³/day]
        retardation = 2.0  # R [-]
        diffusivity_molecular = 1e-4  # D_m [m²/day]
        dispersivity = 1.0  # alpha_L [m]

        # Compute equivalent APVD spreading
        sigma_diff_disp = mean_apv * np.sqrt(
            2
            * retardation
            / streamline_length
            * (diffusivity_molecular * mean_apv / (streamline_length * mean_flow) + dispersivity)
        )
        sigma_apv_diff_disp = np.sqrt(std_apv**2 + sigma_diff_disp**2)

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

        # Compute total diffusivity
        pore_velocity = streamline_length * mean_flow / mean_apv
        diffusivity = diffusivity_molecular + dispersivity * pore_velocity

        # 1. Full solution: diffusion2 with APVD + diffusion/dispersion
        gbins = gamma_utils.bins(mean=mean_apv, std=std_apv, n_bins=nbins)
        cout_full = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=gbins["expected_values"],
            streamline_length=streamline_lengths,
            diffusivity=diffusivity,
            retardation_factor=retardation,
        )

        # 2. APVD with combined std (approximation)
        cout_apvd_combined = advection.gamma_infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            mean=mean_apv,
            std=sigma_apv_diff_disp,
            n_bins=nbins,
            retardation_factor=retardation,
        )

        # Compare in the main breakthrough region
        plot_start, plot_end = 100, 349
        cout_full_slice = cout_full[plot_start : plot_end - 1]
        cout_apvd_slice = cout_apvd_combined[plot_start : plot_end - 1]

        # Both should be valid (not NaN) in this region
        valid_mask = ~np.isnan(cout_full_slice) & ~np.isnan(cout_apvd_slice)
        assert np.sum(valid_mask) > 100, "Should have many valid comparison points"

        # Peak concentration should be similar
        peak_full = np.nanmax(cout_full_slice)
        peak_apvd = np.nanmax(cout_apvd_slice)
        np.testing.assert_allclose(
            peak_full, peak_apvd, rtol=0.05, err_msg="Peak concentrations should match within 5%"
        )

        # Peak should occur at similar times
        peak_idx_full = np.nanargmax(cout_full_slice)
        peak_idx_apvd = np.nanargmax(cout_apvd_slice)
        assert abs(peak_idx_full - peak_idx_apvd) <= 2, "Peak times should match within 2 days"

        # Compare in the peak region (within 30 days of peak)
        # This is where both methods should match well
        # The tails differ due to gamma vs normal distribution shapes
        peak_start = max(0, peak_idx_full - 30)
        peak_end = min(len(cout_full_slice), peak_idx_full + 30)
        cout_full_peak = cout_full_slice[peak_start:peak_end]
        cout_apvd_peak = cout_apvd_slice[peak_start:peak_end]

        np.testing.assert_allclose(
            cout_full_peak,
            cout_apvd_peak,
            rtol=0.06,
            err_msg="cout_full and cout_apvd_combined should match within 6% around peak",
        )

        # Total mass should be conserved - this is the key physical requirement
        # Input mass = 100 (single pulse of concentration 100 for 1 day)
        mass_full = np.nansum(cout_full_slice)
        mass_apvd = np.nansum(cout_apvd_slice)
        np.testing.assert_allclose(mass_full, 100.0, rtol=0.01, err_msg="Full solution should conserve mass within 1%")
        np.testing.assert_allclose(mass_apvd, 100.0, rtol=0.01, err_msg="APVD solution should conserve mass within 1%")
        np.testing.assert_allclose(
            mass_full, mass_apvd, rtol=0.01, err_msg="Both solutions should have same mass within 1%"
        )

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
            diffusivity=0.1,
        )

        # High dispersion
        cout_high = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=gbins["expected_values"],
            streamline_length=streamline_lengths,
            diffusivity=10.0,
        )

        # High dispersion should have lower peak
        peak_low = np.nanmax(cout_low)
        peak_high = np.nanmax(cout_high)
        assert peak_high < peak_low, "Higher dispersion should reduce peak concentration"

        # Mass should be conserved (approximately)
        mass_low = np.nansum(cout_low)
        mass_high = np.nansum(cout_high)
        np.testing.assert_allclose(mass_low, mass_high, rtol=0.1, err_msg="Mass should be approximately conserved")
