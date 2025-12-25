import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage, special

from gwtransport.diffusion import (
    compute_scaled_sigma_array,
    convolve_diffusion,
    create_example_data,
    deconvolve_diffusion,
    extraction_to_infiltration,
    infiltration_to_extraction,
)


class AnalyticalSolutions:
    """Collection of analytical solutions for diffusion problems.

    This class provides analytical solutions for various initial conditions
    of the diffusion equation. Each solution is derived from the fundamental
    solution of the heat equation.

    Notes
    -----
    The diffusion equation:
        ∂c/∂t = diffusivity ∂²c/∂x²

    Has a fundamental solution (Green's function):
        c(x,t) = 1/sqrt(4πDt) * exp(-x²/(4Dt))

    For variable diffusivitys or time steps, we can use the
    relationship:
        sigma = sqrt(2Dt)/dx
    """

    @staticmethod
    def gaussian_pulse(x, t, diffusivity, x0, amplitude, width):
        """Analytical solution for initial Gaussian pulse.

        Parameters
        ----------
        x : ndarray
            Spatial coordinates
        t : float or ndarray
            Time or array of times
        diffusivity : float
            diffusivity
        x0 : float
            Initial center position
        amplitude : float
            Initial pulse amplitude
        width : float
            Initial pulse width (standard deviation)

        Returns
        -------
        ndarray
            Solution at time t

        Notes
        -----
        For an initial condition:
            c(x,0) = A * exp(-(x-x0)²/(2w²))

        The solution is:
            c(x,t) = A * w/sqrt(w² + 2Dt) *
                     exp(-(x-x0)²/(2(w² + 2Dt)))
        """
        if np.isscalar(t):
            t = np.full_like(x, t)

        new_width = np.sqrt(width**2 + 2 * diffusivity * t)
        return amplitude * width / new_width * np.exp(-((x - x0) ** 2) / (2 * new_width**2))

    @staticmethod
    def step_function(x, t, diffusivity, x0):
        """Analytical solution for initial step function.

        Parameters
        ----------
        x : ndarray
            Spatial coordinates
        t : float or ndarray
            Time or array of times
        diffusivity : float
            diffusivity
        x0 : float
            Position of step

        Returns
        -------
        ndarray
            Solution at time t

        Notes
        -----
        For an initial condition:
            c(x,0) = 1 for x > x0, 0 otherwise

        The solution is:
            c(x,t) = 1/2 * (1 + erf((x-x0)/sqrt(4Dt)))
        """
        if np.isscalar(t):
            t = np.full_like(x, t)

        return 0.5 * (1 + special.erf((x - x0) / np.sqrt(4 * diffusivity * t)))

    @staticmethod
    def delta_function(x, t, diffusivity, x0, amplitude=1.0):
        """Analytical solution for initial delta function.

        Parameters
        ----------
        x : ndarray
            Spatial coordinates
        t : float or ndarray
            Time or array of times
        diffusivity : float
            diffusivity
        x0 : float
            Position of delta function
        amplitude : float, optional
            Strength of delta function

        Returns
        -------
        ndarray
            Solution at time t

        Notes
        -----
        For an initial condition:
            c(x,0) = A * δ(x-x0)

        The solution is:
            c(x,t) = A/sqrt(4πDt) * exp(-(x-x0)²/(4Dt))
        """
        if np.isscalar(t):
            t = np.full_like(x, t)

        return amplitude / np.sqrt(4 * np.pi * diffusivity * t) * np.exp(-((x - x0) ** 2) / (4 * diffusivity * t))


def test_gaussian_pulse_constant_time():
    """Test diffusion of Gaussian pulse with constant time step."""
    # Setup grid
    domain_length = 10.0
    nx = 1000
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    # Diffusion parameters
    diffusivity = 0.1
    dt = 0.01
    t = dt

    # Initial condition parameters
    x0 = 0.0
    amplitude = 1.0
    width = 0.5

    # Create initial condition
    initial = AnalyticalSolutions.gaussian_pulse(x, 0, diffusivity, x0, amplitude, width)

    # Calculate sigma for our filter
    sigma = np.sqrt(2 * diffusivity * dt) / dx
    sigma_array = np.full_like(x, sigma)

    # Apply our filter
    numerical = convolve_diffusion(input_signal=initial, sigma_array=sigma_array)

    # Calculate analytical solution
    analytical = AnalyticalSolutions.gaussian_pulse(x, t, diffusivity, x0, amplitude, width)

    # Compare solutions
    assert_allclose(numerical, analytical, rtol=1e-3, atol=1e-3)


def test_gaussian_pulse_variable_time():
    """Test diffusion of Gaussian pulse with variable time steps."""
    # Setup grid
    domain_length = 10.0
    nx = 1000
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    # Diffusion parameters
    diffusivity = 0.1
    # Time steps vary sinusoidally
    dt = 0.01 * (1 + 0.5 * np.sin(2 * np.pi * x / domain_length))

    # Initial condition parameters
    x0 = 0.0
    amplitude = 1.0
    width = 0.5

    # Create initial condition
    initial = AnalyticalSolutions.gaussian_pulse(x, 0, diffusivity, x0, amplitude, width)

    # Calculate variable sigma values
    sigma_array = np.sqrt(2 * diffusivity * dt) / dx

    # Apply our filter
    numerical = convolve_diffusion(input_signal=initial, sigma_array=sigma_array)

    # Calculate analytical solution at each point with its local time
    analytical = AnalyticalSolutions.gaussian_pulse(x, dt, diffusivity, x0, amplitude, width)

    # Compare solutions
    assert_allclose(numerical, analytical, rtol=1e-3, atol=1e-3)


def test_step_function():
    """Test diffusion of step function."""
    # Setup grid
    domain_length = 10.0
    nx = 1000
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    # Diffusion parameters
    diffusivity = 0.1
    dt = 20.0
    t = dt

    # Create initial condition (step at x=0)
    initial = np.heaviside(x, 1.0)

    # Calculate sigma for our filter
    sigma = np.sqrt(2 * diffusivity * dt) / dx
    sigma_array = np.full_like(x, sigma)

    # Apply our filter
    numerical = convolve_diffusion(input_signal=initial, sigma_array=sigma_array, truncate=10.0)

    # Calculate analytical solution
    analytical = AnalyticalSolutions.step_function(x, t, diffusivity, 0.0)

    # Compare solutions
    assert_allclose(numerical, analytical, rtol=1e-6, atol=1e-6)


def test_delta_function():
    """Test diffusion of delta function."""
    # Setup grid
    domain_length = 10.0
    nx = 1000
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    # Diffusion parameters
    diffusivity = 0.2
    dt = 0.01
    t = dt

    # Create initial condition (approximate delta function)
    x0 = dx / 2
    initial = np.zeros_like(x)
    center_idx = np.argmin(np.abs(x - x0))
    initial[center_idx] = 1.0 / dx  # Normalize by dx

    # Calculate sigma for our filter
    sigma = np.sqrt(2 * diffusivity * dt) / dx
    sigma_array = np.full_like(x, sigma)

    # Apply our filter
    numerical = convolve_diffusion(input_signal=initial, sigma_array=sigma_array)

    # Calculate analytical solution
    analytical = AnalyticalSolutions.delta_function(x, t, diffusivity, x0)

    # Compare solutions
    # Use larger tolerance due to discrete approximation of delta function
    assert_allclose(numerical, analytical, rtol=1e-2, atol=1e-2)


def test_zero_sigma():
    """Test behavior when sigma is zero."""
    # Setup
    x = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * x)
    sigma_array = np.zeros_like(x)

    # Filter should return identical signal when sigma is zero
    result = convolve_diffusion(input_signal=signal, sigma_array=sigma_array)
    assert_allclose(result, signal)


def test_input_validation():
    """Test input validation."""
    # Setup
    x = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * x)

    # Test mismatched lengths
    sigma_array = np.zeros(len(x) + 1)
    with pytest.raises(ValueError):
        convolve_diffusion(input_signal=signal, sigma_array=sigma_array)


class TestGaussianComparison:
    """Test suite comparing variable sigma implementation with scipy.ndimage.

    These tests validate our implementation against scipy's gaussian_filter1d
    in cases where they should produce equivalent results. We also examine
    cases where the implementations differ due to the variable sigma capability.

    The test suite includes:
    1. Comparison with constant sigma values
    3. Boundary condition verification
    4. Error analysis
    """

    @staticmethod
    def generate_test_signals():
        """Generate a variety of test signals for comprehensive testing.

        Returns
        -------
        dict
            Dictionary containing different test signals and their descriptions.
        """
        # Create a range of x values
        x = np.linspace(-10, 10, 1000)

        # Create impulse signal
        impulse_signal = np.zeros_like(x)
        impulse_signal[len(x) // 2] = 1

        return {
            "gaussian": {"signal": np.exp(-(x**2) / 2), "description": "Single Gaussian peak"},
            "two_gaussians": {
                "signal": np.exp(-((x - 2) ** 2) / 2) + np.exp(-((x + 2) ** 2) / 2),
                "description": "Two Gaussian peaks",
            },
            "step": {"signal": np.heaviside(x, 1), "description": "Step function"},
            "sine": {"signal": np.sin(2 * np.pi * x / 5), "description": "Sinusoidal wave"},
            "impulse": {"signal": impulse_signal, "description": "Delta function approximation"},
        }

    def test_constant_sigma_equivalence(self):
        """Test that our implementation matches scipy for constant sigma."""
        signals = self.generate_test_signals()
        sigma = 2.0  # Constant sigma value
        truncate = 4.0

        for name, signal_dict in signals.items():
            signal = signal_dict["signal"]

            # Create constant sigma array
            sigma_array = np.full_like(signal, sigma)

            # Apply both filters
            result_variable = convolve_diffusion(input_signal=signal, sigma_array=sigma_array, truncate=truncate)
            result_scipy = ndimage.gaussian_filter1d(signal, sigma, mode="nearest", truncate=truncate)

            # Compare results
            assert_allclose(
                result_variable, result_scipy, rtol=1e-10, atol=1e-10, err_msg=f"Failed for signal type: {name}"
            )

    def test_variable_vs_constant_sigma(self):
        """Compare behavior with varying vs constant sigma."""
        # Create test signal with sharp features
        x = np.linspace(-10, 10, 1000)
        signal = np.exp(-(x**2) / 2) + np.heaviside(x, 1)

        # Create varying sigma array
        sigma_base = 1.0
        sigma_variable = sigma_base * (1 + 0.5 * np.sin(2 * np.pi * x / 20))

        # Compare with equivalent constant sigma
        sigma_constant = np.mean(sigma_variable)

        # Apply both filters
        result_variable = convolve_diffusion(input_signal=signal, sigma_array=sigma_variable)
        result_constant = ndimage.gaussian_filter1d(signal, sigma_constant)

        # Calculate difference
        difference = result_variable - result_constant

        # Store results for analysis
        self.variable_constant_comparison = {
            "x": x,
            "signal": signal,
            "result_variable": result_variable,
            "result_constant": result_constant,
            "difference": difference,
            "sigma_variable": sigma_variable,
            "sigma_constant": sigma_constant,
        }

        # Verify that the results are different (as they should be)
        assert not np.allclose(result_variable, result_constant, rtol=1e-3), (
            "Variable and constant sigma gave identical results"
        )

    @staticmethod
    def test_compare_with_scipy():
        """Test behavior with scipy.ndimage.gaussian_filter1d()."""
        # Create small signal for edge cases
        signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        # Test zero sigma
        sigma_zero = np.zeros_like(signal)
        result_zero = convolve_diffusion(input_signal=signal, sigma_array=sigma_zero)
        assert_allclose(result_zero, signal, err_msg="Failed for zero sigma")

        # Test various sigmas
        sigmas = [1e-10, 1e-5, 1.0, 1e-3]
        for sigma in sigmas:
            sigma_array = np.full_like(signal, sigma)
            result_small_variable = convolve_diffusion(input_signal=signal, sigma_array=sigma_array, truncate=4.0)
            result_small_scipy = ndimage.gaussian_filter1d(signal, sigma, mode="nearest", truncate=4.0)
            assert_allclose(result_small_variable, result_small_scipy, rtol=1e-6, err_msg=f"Failed for sigma={sigma}")


# =============================================================================
# Tests for infiltration_to_extraction function
# =============================================================================


@pytest.mark.parametrize(
    "diffusivity",
    [
        0.01,  # fine_sand
        0.03,  # typical_sand
        0.08,  # coarse_gravel
    ],
)
def test_infiltration_to_extraction_with_various_diffusivity(diffusivity):
    """Test infiltration_to_extraction with different diffusivity values."""
    # Setup
    n_days = 100
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    cin = np.ones(n_days) * 10.0
    flow = np.ones(n_days) * 100.0

    # Test
    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=5000.0,
        diffusivity=diffusivity,
        retardation_factor=1.0,
        aquifer_length=80.0,
    )

    # Verify output
    assert len(cout) == n_days
    assert np.all(np.isfinite(cout))
    # With constant input, output should converge to input value
    assert np.mean(cout[-20:]) == pytest.approx(10.0, rel=0.1)


@pytest.mark.parametrize(
    "retardation_factor",
    [
        1.0,  # conservative_solute
        2.0,  # temperature
        3.5,  # reactive_solute
    ],
)
def test_infiltration_to_extraction_with_retardation(retardation_factor):
    """Test infiltration_to_extraction with different retardation factors."""
    # Setup - use smaller pore volume to ensure valid residence time
    # Max residence time: 1000 * 3.5 / 100 = 35 days (well within 150 days)
    n_days = 150
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    # Step change in concentration
    cin = np.zeros(n_days)
    cin[50:] = 10.0
    flow = np.ones(n_days) * 100.0

    # Test
    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=1000.0,  # Smaller pore volume for valid residence time
        diffusivity=0.03,
        retardation_factor=retardation_factor,
        aquifer_length=80.0,
    )

    # Verify output
    assert len(cout) == n_days
    assert np.all(np.isfinite(cout))
    # Higher retardation should delay and spread the breakthrough
    assert np.mean(cout[:50]) < 1.0  # Low before step
    assert np.mean(cout[-20:]) > 8.0  # Converges to input after step


def test_infiltration_to_extraction_with_variable_flow():
    """Test infiltration_to_extraction with variable flow pattern."""
    # Setup with variable flow
    n_days = 100
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    cin = np.ones(n_days) * 10.0
    # Seasonal flow variation
    t = np.arange(n_days)
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 30))

    # Test
    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=5000.0,
        diffusivity=0.03,
        retardation_factor=1.0,
        aquifer_length=80.0,
    )

    # Verify output
    assert len(cout) == n_days
    assert np.all(np.isfinite(cout))
    assert np.mean(cout) > 0.0


# =============================================================================
# Tests for compute_scaled_sigma_array function
# =============================================================================


def test_compute_scaled_sigma_array_constant_flow():
    """Test compute_scaled_sigma_array with constant flow."""
    # Setup - use smaller pore volume to ensure we have enough flow data
    # With 50 days at 100 m3/day, we have 5000 m3 total
    # Use pore_volume = 1000 m3 for 10 day residence time, leaving plenty of margin
    n_days = 50
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    flow = np.ones(n_days) * 100.0

    # Test
    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=1000.0,  # Smaller pore volume for valid residence time calculation
        diffusivity=0.03,
        retardation_factor=1.0,
        aquifer_length=80.0,
    )

    # Verify
    assert len(sigma_array) == n_days
    assert np.all(np.isfinite(sigma_array))
    assert np.all(sigma_array >= 0.0)
    # With constant flow, sigma should be fairly uniform
    assert np.std(sigma_array) < np.mean(sigma_array) * 0.1


def test_compute_scaled_sigma_array_variable_flow():
    """Test compute_scaled_sigma_array with variable flow."""
    # Setup with variable flow
    n_days = 50
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    t = np.arange(n_days)
    flow = 100.0 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 20))

    # Test - with retardation_factor=2.0, residence time doubles
    # So use pore_volume = 500 to get ~10 day residence time with safety margin
    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,  # Smaller pore volume accounting for retardation
        diffusivity=0.03,
        retardation_factor=2.0,
        aquifer_length=80.0,
    )

    # Verify
    assert len(sigma_array) == n_days
    assert np.all(np.isfinite(sigma_array))
    assert np.all(sigma_array >= 0.0)
    # Sigma should vary with flow
    assert np.std(sigma_array) > 0.0


def test_compute_scaled_sigma_array_with_nan_in_residence_time():
    """Test compute_scaled_sigma_array handles NaN in residence time correctly."""
    # Setup - flow pattern that might cause NaN
    n_days = 50
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    flow = np.ones(n_days) * 100.0
    # Create scenario with zero flow (could cause NaN)
    flow[10:15] = 1e-10  # Very small flow

    # Test - use smaller pore volume
    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=1000.0,  # Smaller pore volume
        diffusivity=0.03,
        retardation_factor=1.0,
        aquifer_length=80.0,
    )

    # Verify - should interpolate NaN values
    assert len(sigma_array) == n_days
    assert np.all(np.isfinite(sigma_array))
    assert np.all(sigma_array >= 0.0)


def test_compute_scaled_sigma_array_clipping():
    """Test that compute_scaled_sigma_array clips extreme values."""
    # Setup with extreme parameters to trigger clipping
    # Use enough days and appropriate pore volume to avoid NaN
    n_days = 100
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    flow = np.ones(n_days) * 10.0  # Higher flow to keep residence time manageable

    # Test with high diffusivity to trigger clipping
    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=100.0,  # Smaller pore volume: residence time = 100 * 5 / 10 = 50 days
        diffusivity=10.0,  # Very high diffusivity
        retardation_factor=5.0,
        aquifer_length=80.0,
    )

    # Verify clipping at 100.0
    assert len(sigma_array) == n_days
    assert np.all(sigma_array <= 100.0)
    assert np.all(sigma_array >= 0.0)


@pytest.mark.parametrize(
    ("diffusivity", "retardation"),
    [
        (0.01, 1.0),
        (0.05, 2.0),
        (0.10, 1.5),
    ],
)
def test_compute_scaled_sigma_array_parametrized(diffusivity, retardation):
    """Test compute_scaled_sigma_array with various parameter combinations."""
    # Setup - use smaller pore volume to ensure valid residence time
    # Max residence time: 1000 * 2.0 / 100 = 20 days (well within 50 days)
    n_days = 50
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    flow = np.ones(n_days) * 100.0

    # Test
    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=1000.0,  # Smaller pore volume
        diffusivity=diffusivity,
        retardation_factor=retardation,
        aquifer_length=100.0,
    )

    # Verify
    assert len(sigma_array) == n_days
    assert np.all(np.isfinite(sigma_array))
    # Higher diffusivity and retardation should increase sigma
    assert np.mean(sigma_array) > 0.0


# =============================================================================
# Tests for create_example_data function
# =============================================================================


def test_create_example_data_basic():
    """Test basic functionality of create_example_data."""
    # Test with default parameters
    x, signal, sigma_array, dt = create_example_data()

    # Verify outputs
    assert len(x) == 1000  # Default nx
    assert len(signal) == 1000
    assert len(sigma_array) == 1000
    assert len(dt) == 1000

    # Verify all are finite
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(signal))
    assert np.all(np.isfinite(sigma_array))
    assert np.all(np.isfinite(dt))

    # Verify reasonable values
    assert np.all(x >= 0.0)
    assert np.all(sigma_array >= 0.0)
    assert np.all(dt > 0.0)


def test_create_example_data_custom_parameters():
    """Test create_example_data with custom parameters."""
    # Test with custom parameters
    nx = 500
    domain_length = 20.0
    diffusivity = 0.5

    x, signal, sigma_array, dt = create_example_data(nx=nx, domain_length=domain_length, diffusivity=diffusivity)

    # Verify outputs
    assert len(x) == nx
    assert len(signal) == nx
    assert len(sigma_array) == nx
    assert len(dt) == nx

    # Verify domain length
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(domain_length, rel=0.01)


def test_create_example_data_different_sizes():
    """Test create_example_data with different grid sizes."""
    for nx in [100, 500, 2000]:
        x, signal, sigma_array, dt = create_example_data(nx=nx)

        assert len(x) == nx
        assert len(signal) == nx
        assert len(sigma_array) == nx
        assert len(dt) == nx


# =============================================================================
# Tests for NotImplementedError paths
# =============================================================================


def test_extraction_to_infiltration_not_implemented():
    """Test that extraction_to_infiltration raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        extraction_to_infiltration(
            cout=np.array([1.0, 2.0, 3.0]),
            flow=np.array([100.0, 100.0, 100.0]),
            aquifer_pore_volume=5000.0,
            diffusivity=0.03,
            retardation_factor=1.0,
            aquifer_length=80.0,
            porosity=0.35,
        )


def test_deconvolve_diffusion_not_implemented():
    """Test that deconvolve_diffusion raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        deconvolve_diffusion(
            output_signal=np.array([1.0, 2.0, 3.0]),
            sigma_array=np.array([0.1, 0.1, 0.1]),
            truncate=4.0,
        )


# =============================================================================
# Edge case tests
# =============================================================================


def test_infiltration_to_extraction_zero_diffusivity():
    """Test infiltration_to_extraction with zero diffusivity (no diffusion)."""
    # Setup - use smaller pore volume for valid residence time
    n_days = 50
    tedges = pd.date_range(start="2020-01-01", periods=n_days + 1, freq="D")
    cin = np.ones(n_days) * 10.0
    flow = np.ones(n_days) * 100.0

    # Test with zero diffusivity
    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=1000.0,  # Smaller pore volume for valid residence time
        diffusivity=0.0,  # No diffusion
        retardation_factor=1.0,
        aquifer_length=80.0,
    )

    # Verify - should return input unchanged (pure advection)
    assert len(cout) == n_days
    assert_allclose(cout, cin, rtol=0.01)


def test_convolve_diffusion_mixed_zero_and_nonzero_sigma():
    """Test convolve_diffusion with mixture of zero and non-zero sigma values."""
    # Create signal with some points having zero sigma
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    sigma_array = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0])

    result = convolve_diffusion(input_signal=signal, sigma_array=sigma_array)

    # Verify
    assert len(result) == len(signal)
    assert np.all(np.isfinite(result))
    # Points with zero sigma should be unchanged
    assert result[0] == pytest.approx(signal[0], abs=0.1)
    assert result[3] == pytest.approx(signal[3], abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__])
