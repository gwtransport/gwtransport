"""Module that implements diffusion."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from gwtransport1d.residence_time import residence_time_retarded
from gwtransport1d.utils import diff


def compute_diffusion(
    cin,
    flow,
    aquifer_pore_volume,
    diffusivity=0.1,
    retardation_factor=1.0,
    aquifer_length=80.0,
    porosity=0.35,
):
    """Compute the diffusion of a compound during 1D transport in the aquifer.

    Parameters
    ----------
    cin : pandas.Series
        Concentration or temperature of the compound in the infiltrating water [ng/m3].
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    diffusivity : float, optional
        diffusivity of the compound in the aquifer [m2/day]. Default is 0.1.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.
    aquifer_length : float, optional
        Length of the aquifer [m]. Default is 80.0.
    porosity : float, optional
        Porosity of the aquifer [dimensionless]. Default is 0.35.

    Returns
    -------
    pandas.Series
        Concentration of the compound in the extracted water [ng/m3].
    """
    sigma_array = compute_sigma_array(
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        diffusivity=diffusivity,
        retardation_factor=retardation_factor,
        aquifer_length=aquifer_length,
        porosity=porosity,
    )
    return gaussian_filter_variable_sigma(cin.values, sigma_array, truncate=30.0)


def gaussian_filter_variable_sigma(input_signal, sigma_array, truncate=4.0):
    """
    Apply analytical diffusion solution with position-dependent sigma values.

    This is a wrapper function that maintains the original interface while using
    the new analytical solution implementation.

    Parameters remain the same as the original function for compatibility.
    """
    return analytical_diffusion_filter(input_signal, sigma_array, truncate)


def diffusion_filter(
    input_signal,
    flow,
    aquifer_pore_volume,
    diffusivity=0.1,
    retardation_factor=1.0,
    aquifer_length=80.0,
    porosity=0.35,
    flow_alignment="right",
):
    """Compute sigma values for diffusion based on flow and aquifer properties.

    Parameters
    ----------
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    diffusivity : float, optional
        diffusivity of the compound in the aquifer [m2/day]. Default is 0.1.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.
    aquifer_length : float, optional
        Length of the aquifer [m]. Default is 80.0.
    porosity : float, optional
        Porosity of the aquifer [dimensionless]. Default is 0.35.

    Returns
    -------
    array
        Array of sigma values for diffusion.
    """
    residence_time = residence_time_retarded(
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="infiltration",
        return_as_series=True,
    )
    residence_time = residence_time.interpolate(method="nearest").ffill().bfill()
    timedelta_at_departure = diff(flow.index, alignment=flow_alignment) / pd.to_timedelta(1, unit="D")
    volume_infiltrated_at_departure = flow * timedelta_at_departure
    cross_sectional_area = aquifer_pore_volume / aquifer_length
    dx = volume_infiltrated_at_departure / cross_sectional_area / porosity
    xedges = np.concatenate(([0.0], np.cumsum(dx)))
    return analytical_diffusion_filter(input_signal=input_signal, xedges=xedges, diffusivity=diffusivity, time=1.0)


def compute_sigma_array(
    flow, aquifer_pore_volume, diffusivity=0.1, retardation_factor=1.0, aquifer_length=80.0, porosity=0.35
):
    """Compute sigma values for diffusion based on flow and aquifer properties.

    Parameters
    ----------
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    diffusivity : float, optional
        diffusivity of the compound in the aquifer [m2/day]. Default is 0.1.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.
    aquifer_length : float, optional
        Length of the aquifer [m]. Default is 80.0.
    porosity : float, optional
        Porosity of the aquifer [dimensionless]. Default is 0.35.

    Returns
    -------
    array
        Array of sigma values for diffusion.
    """
    residence_time = residence_time_retarded(
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="infiltration",
        return_as_series=True,
    )
    residence_time = residence_time.interpolate(method="nearest").ffill().bfill()
    timedelta_at_departure = diff(flow.index, alignment="right") / pd.to_timedelta(1, unit="D")
    volume_infiltrated_at_departure = flow * timedelta_at_departure
    cross_sectional_area = aquifer_pore_volume / aquifer_length
    dx = volume_infiltrated_at_departure / cross_sectional_area / porosity
    sigma_array = np.sqrt(2 * diffusivity * residence_time) / dx
    return np.clip(a=sigma_array.values, a_min=0.0, a_max=100)


def create_example_diffusion_data(nx=1000, domain_length=10.0, diffusivity=0.1):
    """Create example data for demonstrating variable-sigma diffusion.

    Parameters
    ----------
    nx : int, optional
        Number of spatial points. Default is 1000.
    domain_length : float, optional
        Domain length. Default is 10.0.
    diffusivity : float, optional
        diffusivity. Default is 0.1.

    Returns
    -------
    x : ndarray
        Spatial coordinates.
    signal : ndarray
        Initial signal (sum of two Gaussians).
    sigma_array : ndarray
        Array of sigma values varying in space.
    dt : ndarray
        Array of time steps varying in space.

    Notes
    -----
    This function creates a test case with:
    - A signal composed of two Gaussian peaks
    - Sinusoidally varying time steps
    - Corresponding sigma values for diffusion
    """
    # Create spatial grid
    x = np.linspace(0, domain_length, nx)
    dx = x[1] - x[0]

    # Create initial signal (two Gaussians)
    signal = np.exp(-((x - 3) ** 2)) + 0.5 * np.exp(-((x - 7) ** 2) / 0.5) + 0.1 * np.random.randn(nx)

    # Create varying time steps
    dt = 0.001 * (1 + np.sin(2 * np.pi * x / domain_length))

    # Calculate corresponding sigma values
    sigma_array = np.sqrt(2 * diffusivity * dt) / dx

    return x, signal, sigma_array, dt


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # Generate example data
    x, signal, sigma_array, dt = create_example_diffusion_data()

    # Apply variable-sigma filtering
    filtered = gaussian_filter_variable_sigma(signal, sigma_array * 5)

    # Compare with regular Gaussian filter
    avg_sigma = np.mean(sigma_array)
    regular_filtered = ndimage.gaussian_filter1d(signal, avg_sigma)
    plt.figure(figsize=(10, 6))
    plt.plot(x, signal, label="Original signal", lw=0.8)
    plt.plot(x, filtered, label="Variable-sigma filtered", lw=1.0)

    plt.plot(x, regular_filtered, label="Regular Gaussian filter", lw=0.8, ls="--")
