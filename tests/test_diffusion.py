import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import special

# Import our function to test
from gwtransport1d.diffusion import gaussian_filter_variable_sigma


class AnalyticalSolutions:
    """Collection of analytical solutions for diffusion problems.

    This class provides analytical solutions for various initial conditions
    of the diffusion equation. Each solution is derived from the fundamental
    solution of the heat equation.

    Notes
    -----
    The diffusion equation:
        ∂c/∂t = diffusion_coefficient ∂²c/∂x²

    Has a fundamental solution (Green's function):
        c(x,t) = 1/sqrt(4πDt) * exp(-x²/(4Dt))

    For variable diffusion coefficients or time steps, we can use the
    relationship:
        sigma = sqrt(2Dt)/dx
    """

    @staticmethod
    def gaussian_pulse(x, t, diffusion_coefficient, x0, amplitude, width):
        """Analytical solution for initial Gaussian pulse.

        Parameters
        ----------
        x : ndarray
            Spatial coordinates
        t : float or ndarray
            Time or array of times
        diffusion_coefficient : float
            Diffusion coefficient
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

        new_width = np.sqrt(width**2 + 2 * diffusion_coefficient * t)
        return amplitude * width / new_width * np.exp(-((x - x0) ** 2) / (2 * new_width**2))

    @staticmethod
    def step_function(x, t, diffusion_coefficient, x0):
        """Analytical solution for initial step function.

        Parameters
        ----------
        x : ndarray
            Spatial coordinates
        t : float or ndarray
            Time or array of times
        diffusion_coefficient : float
            Diffusion coefficient
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

        return 0.5 * (1 + special.erf((x - x0) / np.sqrt(4 * diffusion_coefficient * t)))

    @staticmethod
    def delta_function(x, t, diffusion_coefficient, x0, amplitude=1.0):
        """Analytical solution for initial delta function.

        Parameters
        ----------
        x : ndarray
            Spatial coordinates
        t : float or ndarray
            Time or array of times
        diffusion_coefficient : float
            Diffusion coefficient
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

        return amplitude / np.sqrt(4 * np.pi * diffusion_coefficient * t) * np.exp(-((x - x0) ** 2) / (4 * diffusion_coefficient * t))


def test_gaussian_pulse_constant_time():
    """Test diffusion of Gaussian pulse with constant time step."""
    # Setup grid
    domain_length = 10.0
    nx = 1000
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    # Diffusion parameters
    diffusion_coefficient = 0.1
    dt = 0.01
    t = dt

    # Initial condition parameters
    x0 = 0.0
    amplitude = 1.0
    width = 0.5

    # Create initial condition
    initial = AnalyticalSolutions.gaussian_pulse(x, 0, diffusion_coefficient, x0, amplitude, width)

    # Calculate sigma for our filter
    sigma = np.sqrt(2 * diffusion_coefficient * dt) / dx
    sigma_array = np.full_like(x, sigma)

    # Apply our filter
    numerical = gaussian_filter_variable_sigma(initial, sigma_array)

    # Calculate analytical solution
    analytical = AnalyticalSolutions.gaussian_pulse(x, t, diffusion_coefficient, x0, amplitude, width)

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
    diffusion_coefficient = 0.1
    # Time steps vary sinusoidally
    dt = 0.01 * (1 + 0.5 * np.sin(2 * np.pi * x / domain_length))

    # Initial condition parameters
    x0 = 0.0
    amplitude = 1.0
    width = 0.5

    # Create initial condition
    initial = AnalyticalSolutions.gaussian_pulse(x, 0, diffusion_coefficient, x0, amplitude, width)

    # Calculate variable sigma values
    sigma_array = np.sqrt(2 * diffusion_coefficient * dt) / dx

    # Apply our filter
    numerical = gaussian_filter_variable_sigma(initial, sigma_array)

    # Calculate analytical solution at each point with its local time
    analytical = AnalyticalSolutions.gaussian_pulse(x, dt, diffusion_coefficient, x0, amplitude, width)

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
    diffusion_coefficient = 0.1
    dt = 0.01
    t = dt

    # Create initial condition (step at x=0)
    initial = np.heaviside(x, 1.0)

    # Calculate sigma for our filter
    sigma = np.sqrt(2 * diffusion_coefficient * dt) / dx
    sigma_array = np.full_like(x, sigma)

    # Apply our filter
    numerical = gaussian_filter_variable_sigma(initial, sigma_array)

    # Calculate analytical solution
    analytical = AnalyticalSolutions.step_function(x, t, diffusion_coefficient, 0.0)

    # Compare solutions
    assert_allclose(numerical, analytical, rtol=1e-3, atol=1e-3)


def test_delta_function():
    """Test diffusion of delta function."""
    # Setup grid
    domain_length = 10.0
    nx = 1000
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    # Diffusion parameters
    diffusion_coefficient = 0.2
    dt = 0.01
    t = dt

    # Create initial condition (approximate delta function)
    x0 = dx / 2
    initial = np.zeros_like(x)
    center_idx = np.argmin(np.abs(x - x0))
    initial[center_idx] = 1.0 / dx  # Normalize by dx

    # Calculate sigma for our filter
    sigma = np.sqrt(2 * diffusion_coefficient * dt) / dx
    sigma_array = np.full_like(x, sigma)

    # Apply our filter
    numerical = gaussian_filter_variable_sigma(initial, sigma_array)

    # Calculate analytical solution
    analytical = AnalyticalSolutions.delta_function(x, t, diffusion_coefficient, x0)

    # Compare solutions
    # Use larger tolerance due to discrete approximation of delta function
    assert_allclose(numerical, analytical, rtol=1e-2, atol=1e-2)


def test_boundary_conditions():
    """Test different boundary conditions."""
    # Setup grid
    domain_length = 10.0
    nx = 100
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    # Diffusion parameters
    diffusion_coefficient = 0.1
    dt = 0.01

    # Create Gaussian pulse near boundary
    initial = np.exp(-((x - domain_length / 2.5) ** 2) / 0.1)

    # Calculate sigma
    sigma = np.sqrt(2 * diffusion_coefficient * dt) / dx
    sigma_array = np.full_like(x, sigma)

    # Test different boundary conditions
    modes = ["reflect", "constant", "nearest", "mirror", "wrap"]

    for mode in modes:
        result = gaussian_filter_variable_sigma(initial, sigma_array, mode=mode)
        # Check conservation of mass for appropriate boundary conditions
        if mode in {"reflect", "mirror", "wrap"}:
            assert_allclose(result.sum() * dx, initial.sum() * dx, rtol=1e-3)


def test_zero_sigma():
    """Test behavior when sigma is zero."""
    # Setup
    x = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * x)
    sigma_array = np.zeros_like(x)

    # Filter should return identical signal when sigma is zero
    result = gaussian_filter_variable_sigma(signal, sigma_array)
    assert_allclose(result, signal)


def test_input_validation():
    """Test input validation."""
    # Setup
    x = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * x)

    # Test mismatched lengths
    sigma_array = np.zeros(len(x) + 1)
    with pytest.raises(ValueError):  # noqa: PT011
        gaussian_filter_variable_sigma(signal, sigma_array)

    # Test invalid mode
    sigma_array = np.zeros_like(x)
    with pytest.raises(ValueError):  # noqa: PT011
        gaussian_filter_variable_sigma(signal, sigma_array, mode="invalid_mode")


if __name__ == "__main__":
    pytest.main([__file__])
