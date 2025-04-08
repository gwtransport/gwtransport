import numpy as np
import pandas as pd
import pytest
from scipy import special

from gwtransport1d.advection import cout_advection
from gwtransport1d.diffusion import compute_diffusion


class AnalyticalSolutions:
    """Analytical solutions for advection-diffusion problems."""

    @staticmethod
    def temperature_step(x, t, v, D, x0=0, T0=0, T1=1):
        """
        Analytical solution for temperature step under advection-diffusion.

        Parameters
        ----------
        x : array-like
            Spatial coordinates
        t : float
            Time
        v : float
            Advection velocity
        D : float
            Diffusion coefficient
        x0 : float, optional
            Initial position of step
        T0 : float, optional
            Initial temperature before step
        T1 : float, optional
            Initial temperature after step

        Returns
        -------
        array-like
            Temperature distribution at time t

        Notes
        -----
        For initial condition: T(x,0) = T0 for x < x0, T1 for x ≥ x0
        Solution: T(x,t) = T0 + (T1-T0)/2 * (1 + erf((x - x0 - v*t)/(2*sqrt(D*t))))
        """
        return T0 + (T1 - T0) * 0.5 * (1 + special.erf((x - x0 - v * t) / (2 * np.sqrt(D * t))))

    @staticmethod
    def gaussian_pulse(x, t, v, D, x0, amplitude, width):
        """
        Analytical solution for Gaussian temperature pulse under advection-diffusion.

        Parameters
        ----------
        x : array-like
            Spatial coordinates
        t : float
            Time
        v : float
            Advection velocity
        D : float
            Diffusion coefficient
        x0 : float
            Initial center position
        amplitude : float
            Initial pulse amplitude
        width : float
            Initial pulse width (standard deviation)

        Returns
        -------
        array-like
            Temperature distribution at time t

        Notes
        -----
        For initial condition: T(x,0) = A * exp(-(x-x0)²/(2w²))
        Solution: T(x,t) = A * w/sqrt(w² + 2Dt) * exp(-(x-x0-vt)²/(2(w² + 2Dt)))
        """
        new_width = np.sqrt(width**2 + 2 * D * t)
        shifted_x = x - x0 - v * t
        return amplitude * width / new_width * np.exp(-(shifted_x**2) / (2 * new_width**2))

    @staticmethod
    def periodic_solution(x, t, v, D, L, n_terms=10):
        """
        Analytical solution for periodic initial condition under advection-diffusion.

        Parameters
        ----------
        x : array-like
            Spatial coordinates
        t : float
            Time
        v : float
            Advection velocity
        D : float
            Diffusion coefficient
        L : float
            Domain length (period)
        n_terms : int, optional
            Number of Fourier terms to include

        Returns
        -------
        array-like
            Temperature distribution at time t

        Notes
        -----
        For initial condition: T(x,0) = sin(2πx/L)
        Solution uses Fourier series decomposition with decay terms
        """
        k = 2 * np.pi / L
        solution = np.zeros_like(x, dtype=float)

        for n in range(1, n_terms + 1):
            kn = n * k
            solution += np.exp(-D * kn**2 * t) * np.sin(kn * (x - v * t))

        return solution


@pytest.fixture
def basic_flow_setup():
    """Create basic flow setup for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-31", freq="D")
    flow = pd.Series(100.0, index=dates)  # Constant flow of 100 m³/day
    aquifer_pore_volume = 1000.0  # m³
    porosity = 0.35
    aquifer_length = 80.0
    return dates, flow, aquifer_pore_volume, porosity, aquifer_length


def test_step_solution_analytical(basic_flow_setup):
    """Compare numerical solution with analytical solution for temperature step."""
    dates, flow, aquifer_pore_volume, porosity, aquifer_length = basic_flow_setup

    # Create temperature step
    T0, T1 = 10.0, 20.0
    temperature = pd.Series(T0, index=dates)
    step_date = dates[10]
    temperature[step_date:] = T1

    # Calculate velocity and diffusivity
    v = flow.iloc[0] / (porosity * aquifer_length)  # m/day
    D = 0.1  # m²/day

    # Apply numerical solution (advection then diffusion)
    advected = cout_advection(cin=temperature, flow=flow, aquifer_pore_volume=aquifer_pore_volume)

    numerical = compute_diffusion(
        cin=advected,
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        diffusivity=D,
        aquifer_length=aquifer_length,
        porosity=porosity,
    )

    # Calculate analytical solution
    t = 5.0  # days after step
    x = np.linspace(0, aquifer_length, 100)
    analytical = AnalyticalSolutions.temperature_step(x=x, t=t, v=v, D=D, x0=0, T0=T0, T1=T1)

    # Compare solutions at specific points
    # Note: Need to account for boundary conditions and time shifting
    mid_idx = len(dates) // 2
    numerical_profile = numerical[mid_idx : mid_idx + 100]
    non_nan = ~np.isnan(numerical_profile)

    if non_nan.any():
        # Compare shapes (normalized to account for boundary effects)
        num_norm = (numerical_profile[non_nan] - T0) / (T1 - T0)
        ana_norm = (analytical - T0) / (T1 - T0)
        assert np.allclose(num_norm, ana_norm, atol=0.1)


def test_gaussian_pulse_analytical(basic_flow_setup):
    """Compare numerical solution with analytical solution for Gaussian pulse."""
    dates, flow, aquifer_pore_volume, porosity, aquifer_length = basic_flow_setup

    # Create Gaussian pulse input
    x0 = aquifer_length / 4
    width = aquifer_length / 20
    amplitude = 10.0
    x = np.linspace(0, aquifer_length, len(dates))
    temperature = pd.Series(amplitude * np.exp(-((x - x0) ** 2) / (2 * width**2)), index=dates)

    # Calculate velocity and diffusivity
    v = flow.iloc[0] / (porosity * aquifer_length)  # m/day
    D = 0.1  # m²/day

    # Apply numerical solution
    advected = cout_advection(cin=temperature, flow=flow, aquifer_pore_volume=aquifer_pore_volume)

    numerical = compute_diffusion(
        cin=advected,
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        diffusivity=D,
        aquifer_length=aquifer_length,
        porosity=porosity,
    )

    # Calculate analytical solution
    t = 5.0  # days
    analytical = AnalyticalSolutions.gaussian_pulse(x=x, t=t, v=v, D=D, x0=x0, amplitude=amplitude, width=width)

    # Compare solutions
    mid_idx = len(dates) // 2
    numerical_profile = numerical[mid_idx : mid_idx + 100]
    non_nan = ~np.isnan(numerical_profile)

    if non_nan.any():
        # Compare peak position and width
        num_peak = x[np.argmax(numerical_profile[non_nan])]
        ana_peak = x[np.argmax(analytical)]
        assert np.isclose(num_peak, ana_peak, rtol=0.1)

        # Compare peak amplitude (accounting for numerical diffusion)
        assert np.isclose(np.max(numerical_profile[non_nan]) / amplitude, np.max(analytical) / amplitude, rtol=0.2)


def test_periodic_solution_analytical(basic_flow_setup):
    """Compare numerical solution with analytical solution for periodic input."""
    dates, flow, aquifer_pore_volume, porosity, aquifer_length = basic_flow_setup

    # Create periodic input
    x = np.linspace(0, aquifer_length, len(dates))
    temperature = pd.Series(10 + 5 * np.sin(2 * np.pi * x / aquifer_length), index=dates)

    # Calculate velocity and diffusivity
    v = flow.iloc[0] / (porosity * aquifer_length)  # m/day
    D = 0.1  # m²/day

    # Apply numerical solution
    advected = cout_advection(cin=temperature, flow=flow, aquifer_pore_volume=aquifer_pore_volume)

    numerical = compute_diffusion(
        cin=advected,
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        diffusivity=D,
        aquifer_length=aquifer_length,
        porosity=porosity,
    )

    # Calculate analytical solution
    t = 5.0  # days
    analytical = AnalyticalSolutions.periodic_solution(x=x, t=t, v=v, D=D, L=aquifer_length)

    # Compare solutions
    mid_idx = len(dates) // 2
    numerical_profile = numerical[mid_idx : mid_idx + 100]
    non_nan = ~np.isnan(numerical_profile)

    if non_nan.any():
        # Compare phase and amplitude
        num_fft = np.fft.fft(numerical_profile[non_nan])
        ana_fft = np.fft.fft(analytical)

        # Compare dominant frequencies
        num_freq = np.abs(num_fft[1])
        ana_freq = np.abs(ana_fft[1])
        assert np.isclose(num_freq / ana_freq, 1.0, rtol=0.2)


def test_conservation_properties(basic_flow_setup):
    """Test conservation properties against analytical predictions."""
    dates, flow, aquifer_pore_volume, porosity, aquifer_length = basic_flow_setup

    # Create Gaussian input
    x0 = aquifer_length / 4
    width = aquifer_length / 20
    amplitude = 10.0
    x = np.linspace(0, aquifer_length, len(dates))
    temperature = pd.Series(amplitude * np.exp(-((x - x0) ** 2) / (2 * width**2)), index=dates)

    # Apply numerical solution
    D = 0.1  # m²/day
    advected = cout_advection(cin=temperature, flow=flow, aquifer_pore_volume=aquifer_pore_volume)

    numerical = compute_diffusion(
        cin=advected,
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        diffusivity=D,
        aquifer_length=aquifer_length,
        porosity=porosity,
    )

    # Check conservation of total heat content
    # (excluding boundary regions)
    mid_idx = len(dates) // 2
    initial_heat = np.trapz(temperature[mid_idx : mid_idx + 100], x[mid_idx : mid_idx + 100])
    final_heat = np.trapz(
        numerical[mid_idx : mid_idx + 100][~np.isnan(numerical[mid_idx : mid_idx + 100])],
        x[mid_idx : mid_idx + 100][~np.isnan(numerical[mid_idx : mid_idx + 100])],
    )
    assert np.isclose(initial_heat, final_heat, rtol=0.1)


def test_boundary_conditions(basic_flow_setup):
    """Test boundary condition handling against analytical solutions."""
    dates, flow, aquifer_pore_volume, porosity, aquifer_length = basic_flow_setup

    # Create step function near boundary
    T0, T1 = 10.0, 20.0
    temperature = pd.Series(T0, index=dates)
    step_idx = 5
    temperature[dates[step_idx] :] = T1

    # Apply numerical solution
    D = 0.1  # m²/day
    advected = cout_advection(cin=temperature, flow=flow, aquifer_pore_volume=aquifer_pore_volume)

    numerical = compute_diffusion(
        cin=advected,
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        diffusivity=D,
        aquifer_length=aquifer_length,
        porosity=porosity,
    )

    # Check that boundary conditions are handled consistently
    assert not np.any(np.isnan(numerical[:step_idx]))  # No NaN at start
    assert np.allclose(numerical[:step_idx], T0, atol=0.1)  # Maintains initial temperature


if __name__ == "__main__":
    pytest.main([__file__])
