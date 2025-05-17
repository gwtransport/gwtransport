import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import ndimage, special

from gwtransport.diffusion import convolve_diffusion


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
    numerical = convolve_diffusion(initial, sigma_array)

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
    numerical = convolve_diffusion(initial, sigma_array)

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
    numerical = convolve_diffusion(initial, sigma_array, truncate=10.0)

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
    numerical = convolve_diffusion(initial, sigma_array)

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
    result = convolve_diffusion(signal, sigma_array)
    assert_allclose(result, signal)


def test_input_validation():
    """Test input validation."""
    # Setup
    x = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * x)

    # Test mismatched lengths
    sigma_array = np.zeros(len(x) + 1)
    with pytest.raises(ValueError):
        convolve_diffusion(signal, sigma_array)


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

        signals = {
            "gaussian": {"signal": np.exp(-(x**2) / 2), "description": "Single Gaussian peak"},
            "two_gaussians": {
                "signal": np.exp(-((x - 2) ** 2) / 2) + np.exp(-((x + 2) ** 2) / 2),
                "description": "Two Gaussian peaks",
            },
            "step": {"signal": np.heaviside(x, 1), "description": "Step function"},
            "sine": {"signal": np.sin(2 * np.pi * x / 5), "description": "Sinusoidal wave"},
            "impulse": {"signal": np.zeros_like(x), "description": "Delta function approximation"},
        }

        # Add impulse in the middle
        signals["impulse"]["signal"][len(x) // 2] = 1

        return signals

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
            result_variable = convolve_diffusion(signal, sigma_array, truncate=truncate)
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
        result_variable = convolve_diffusion(signal, sigma_variable)
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
        result_zero = convolve_diffusion(signal, sigma_zero)
        assert_allclose(result_zero, signal, err_msg="Failed for zero sigma")

        # Test various sigmas
        sigmas = [1e-10, 1e-5, 1.0, 1e-3]
        for sigma in sigmas:
            sigma_array = np.full_like(signal, sigma)
            result_small_variable = convolve_diffusion(signal, sigma_array, truncate=4.0)
            result_small_scipy = ndimage.gaussian_filter1d(signal, sigma, mode="nearest", truncate=4.0)
            assert_allclose(result_small_variable, result_small_scipy, rtol=1e-6, err_msg=f"Failed for sigma={sigma}")


if __name__ == "__main__":
    pytest.main([__file__])
