"""Module that implements diffusion."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, sparse

from gwtransport.residence_time import residence_time
from gwtransport.utils import diff


def forward(
    cin,
    flow,
    aquifer_pore_volume,
    diffusivity=0.1,
    retardation_factor=1.0,
    aquifer_length=80.0,
    porosity=0.35,
):
    """Compute the diffusion of a compound during 1D transport in the aquifer.

    This function represents a forward operation (equivalent to convolution).

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
    return convolve_diffusion(cin.values, sigma_array, truncate=30.0)


def backward(
    cout,
    flow,
    aquifer_pore_volume,
    diffusivity=0.1,
    retardation_factor=1.0,
    aquifer_length=80.0,
    porosity=0.35,
):
    """Compute the reverse diffusion of a compound during 1D transport in the aquifer.

    This function represents a backward operation (equivalent to deconvolution).

    Parameters
    ----------
    cout : pandas.Series
        Concentration or temperature of the compound in the extracted water [ng/m3].
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
        Concentration of the compound in the infiltrating water [ng/m3].

    Notes
    -----
    Backward diffusion (deconvolution) is mathematically ill-posed and requires
    regularization to obtain a stable solution.
    """
    msg = "Backward diffusion (deconvolution) is not implemented yet"
    raise NotImplementedError(msg)


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
    residence_time = residence_time(
        flow=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="infiltration",
        return_pandas_series=True,
    )
    residence_time = residence_time.interpolate(method="nearest").ffill().bfill()
    timedelta_at_departure = diff(flow.index, alignment="right") / pd.to_timedelta(1, unit="D")
    volume_infiltrated_at_departure = flow * timedelta_at_departure
    cross_sectional_area = aquifer_pore_volume / aquifer_length
    dx = volume_infiltrated_at_departure / cross_sectional_area / porosity
    sigma_array = np.sqrt(2 * diffusivity * residence_time) / dx
    return np.clip(a=sigma_array.values, a_min=0.0, a_max=100)


def convolve_diffusion(input_signal, sigma_array, truncate=4.0):
    """Apply Gaussian filter with position-dependent sigma values.

    This function extends scipy.ndimage.gaussian_filter1d by allowing the standard
    deviation (sigma) of the Gaussian kernel to vary at each point in the signal.
    It implements the filter using a sparse convolution matrix where each row
    represents a Gaussian kernel with a locally-appropriate standard deviation.

    Parameters
    ----------
    input_signal : ndarray
        One-dimensional input array to be filtered.
    sigma_array : ndarray
        One-dimensional array of standard deviation values, must have same length
        as input_signal. Each value specifies the Gaussian kernel width at the
        corresponding position.
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    ndarray
        The filtered input signal. Has the same shape as input_signal.

    Notes
    -----
    At the boundaries, the outer values are repeated to avoid edge effects. Equal to mode=`nearest`
    in `scipy.ndimage.gaussian_filter1d`.

    The function constructs a sparse convolution matrix where each row represents
    a position-specific Gaussian kernel. The kernel width adapts to local sigma
    values, making it suitable for problems with varying diffusivitys
    or time steps.

    For diffusion problems, the local sigma values can be calculated as:
    sigma = sqrt(2 * diffusivity * dt) / dx
    where diffusivity is the diffusivity, dt is the time step, and dx is the
    spatial step size.

    The implementation uses sparse matrices for memory efficiency when dealing
    with large signals or when sigma values vary significantly.

    See Also
    --------
    scipy.ndimage.gaussian_filter1d : Fixed-sigma Gaussian filtering
    scipy.sparse : Sparse matrix implementations

    Examples
    --------
    >>> # Create a sample signal
    >>> x = np.linspace(0, 10, 1000)
    >>> signal = np.exp(-((x - 3) ** 2)) + 0.5 * np.exp(-((x - 7) ** 2) / 0.5)

    >>> # Create position-dependent sigma values
    >>> diffusivity = 0.1  # diffusivity
    >>> dt = 0.001 * (1 + np.sin(2 * np.pi * x / 10))  # Varying time steps
    >>> dx = x[1] - x[0]
    >>> sigma_array = np.sqrt(2 * diffusivity * dt) / dx

    >>> # Apply the filter
    >>> filtered = convolve_diffusion(signal, sigma_array)
    """
    if len(input_signal) != len(sigma_array):
        msg = "Input signal and sigma array must have the same length"
        raise ValueError(msg)

    n = len(input_signal)

    # Handle zero sigma values
    zero_mask = sigma_array == 0
    if np.all(zero_mask):
        return input_signal.copy()

    # Get maximum kernel size and create position arrays
    max_sigma = np.max(sigma_array)
    max_radius = int(truncate * max_sigma + 0.5)

    # Create arrays for all possible kernel positions
    positions = np.arange(-max_radius, max_radius + 1)

    # Create a mask for valid sigma values
    valid_sigma = ~zero_mask
    valid_indices = np.where(valid_sigma)[0]

    # Create position matrices for broadcasting
    # Shape: (n_valid_points, 1)
    center_positions = valid_indices[:, np.newaxis]
    # Shape: (1, max_kernel_size)
    kernel_positions = positions[np.newaxis, :]

    # Calculate the relative positions for each point
    # This creates a matrix of shape (n_valid_points, max_kernel_size)
    relative_positions = kernel_positions

    # Calculate Gaussian weights for all positions at once
    # Using broadcasting to create a matrix of shape (n_valid_points, max_kernel_size)
    sigmas = sigma_array[valid_sigma][:, np.newaxis]
    weights = np.exp(-0.5 * (relative_positions / sigmas) ** 2)

    # Normalize each kernel
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Calculate absolute positions in the signal
    absolute_positions = center_positions + relative_positions

    # Handle boundary conditions
    absolute_positions = np.clip(absolute_positions, 0, n - 1)

    # Create coordinate arrays for sparse matrix
    rows = np.repeat(center_positions, weights.shape[1])
    cols = absolute_positions.ravel()
    data = weights.ravel()

    # Remove zero weights to save memory
    nonzero_mask = data != 0
    rows = rows[nonzero_mask]
    cols = cols[nonzero_mask]
    data = data[nonzero_mask]

    # Add identity matrix elements for zero-sigma positions
    if np.any(zero_mask):
        zero_indices = np.where(zero_mask)[0]
        rows = np.concatenate([rows, zero_indices])
        cols = np.concatenate([cols, zero_indices])
        data = np.concatenate([data, np.ones(len(zero_indices))])

    # Create the sparse matrix
    conv_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

    # remove diffusion from signal with inverse of the convolution matrix
    # conv_matrix_inv = np.linalg.lstsq(conv_matrix.todense(), np.eye(n), rcond=None)[0]

    # Apply the filter
    return conv_matrix.dot(input_signal)


def deconvolve_diffusion(output_signal, sigma_array, truncate=4.0):
    """Apply Gaussian deconvolution with position-dependent sigma values.

    This function extends scipy.ndimage.gaussian_filter1d by allowing the standard
    deviation (sigma) of the Gaussian kernel to vary at each point in the signal.
    It implements the filter using a sparse convolution matrix where each row
    represents a Gaussian kernel with a locally-appropriate standard deviation.

    Parameters
    ----------
    output_signal : ndarray
        One-dimensional input array to be filtered.
    sigma_array : ndarray
        One-dimensional array of standard deviation values, must have same length
        as output_signal. Each value specifies the Gaussian kernel width at the
        corresponding position.
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    ndarray
        The filtered output signal. Has the same shape as output_signal.
    """
    msg = "Deconvolution is not implemented yet"
    raise NotImplementedError(msg)


def create_example_data(nx=1000, domain_length=10.0, diffusivity=0.1):
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
    x, signal, sigma_array, dt = create_example_data()

    # Apply variable-sigma filtering
    filtered = convolve_diffusion(signal, sigma_array * 5)

    # Compare with regular Gaussian filter
    avg_sigma = np.mean(sigma_array)
    regular_filtered = ndimage.gaussian_filter1d(signal, avg_sigma)
    plt.figure(figsize=(10, 6))
    plt.plot(x, signal, label="Original signal", lw=0.8)
    plt.plot(x, filtered, label="Variable-sigma filtered", lw=1.0)

    plt.plot(x, regular_filtered, label="Regular Gaussian filter", lw=0.8, ls="--")
