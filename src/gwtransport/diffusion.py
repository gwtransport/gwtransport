"""
Diffusive Transport Corrections for Aquifer Systems.

This module implements diffusion/dispersion processes that modify advective transport
in aquifer systems. Diffusion causes spreading and smoothing of concentration or
temperature fronts as they travel through the aquifer. While advection moves compounds
with water flow, diffusion causes spreading due to molecular diffusion, mechanical
dispersion, and thermal diffusion (for temperature).

Limitation: It works well for when flow and tedges are relatively constant.
dx in compute_scaled_sigma_array is assumed to be the same for neighboring cells.

Available functions:

- :func:`infiltration_to_extraction` - Apply diffusion during infiltration to extraction
  transport. Combines advection (via residence time) with diffusion (via Gaussian smoothing).
  Computes position-dependent diffusion based on local residence time and returns concentration
  or temperature in extracted water.

- :func:`extraction_to_infiltration` - NOT YET IMPLEMENTED. Inverse diffusion is numerically
  unstable and requires regularization techniques. Placeholder for future implementation.

- :func:`compute_scaled_sigma_array` - Calculate position-dependent diffusion parameters. Computes
  standard deviation (sigma) for Gaussian smoothing at each time step based on residence time,
  diffusivity, and spatial discretization: sigma = sqrt(2 * diffusivity * residence_time) / dx.

- :func:`convolve_diffusion` - Apply variable-sigma Gaussian filtering. Extends
  scipy.ndimage.gaussian_filter1d to position-dependent sigma using sparse matrix representation
  for efficiency. Handles boundary conditions via nearest-neighbor extrapolation.

- :func:`deconvolve_diffusion` - NOT YET IMPLEMENTED. Inverse filtering placeholder for future
  diffusion deconvolution with required regularization for stability.

- :func:`create_example_data` - Generate test data for demonstrating diffusion effects with
  signals having varying time steps and corresponding sigma arrays. Useful for testing and
  validation.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import ndimage, sparse

from gwtransport.residence_time import residence_time


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    diffusivity: float = 0.1,
    retardation_factor: float = 1.0,
    aquifer_length: float = 80.0,
) -> npt.NDArray[np.floating]:
    """Compute the diffusion of a compound during 1D transport in the aquifer.

    This function represents infiltration to extraction modeling (equivalent to convolution). The function
    is exact if flow and tedges are constant. Errors are made otherwise, but the mass balance is kept.
    If diffusion behavior is important during times of varying flow and tedges, the diffusion2 module provides
    an alternative approach that works better in those conditions but takes much longer to compute.

    Parameters
    ----------
    cin : array-like
        Concentration or temperature of the compound in the infiltrating water [ng/m3].
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges corresponding to the flow values.
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
    numpy.ndarray
        Concentration of the compound in the extracted water [ng/m3].

    Notes
    -----
    Common values for heat in saturated sand in m²/day:

    * Lower end (finer sand/silt): ~0.007-0.01 m²/day
    * Typical saturated sand: ~0.01-0.05 m²/day
    * Upper end (coarse sand/gravel): ~0.05-0.10 m²/day
    """
    cin = np.asarray(cin)
    flow = np.asarray(flow)

    # Validate physical parameters
    if aquifer_pore_volume <= 0:
        msg = f"Aquifer pore volume must be positive, got {aquifer_pore_volume}"
        raise ValueError(msg)
    if aquifer_length <= 0:
        msg = f"Aquifer length must be positive, got {aquifer_length}"
        raise ValueError(msg)
    if diffusivity < 0:
        msg = f"Diffusivity must be non-negative, got {diffusivity}"
        raise ValueError(msg)

    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        diffusivity=diffusivity,
        retardation_factor=retardation_factor,
        aquifer_length=aquifer_length,
    )
    return convolve_diffusion(input_signal=cin, sigma_array=sigma_array, truncate=30.0)


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    aquifer_pore_volume: float,
    diffusivity: float = 0.1,
    retardation_factor: float = 1.0,
    aquifer_length: float = 80.0,
    porosity: float = 0.35,
) -> npt.NDArray[np.floating]:
    """Compute the reverse diffusion of a compound during 1D transport in the aquifer.

    This function represents extraction to infiltration modeling (equivalent to deconvolution).

    Parameters
    ----------
    cout : array-like
        Concentration or temperature of the compound in the extracted water [ng/m3].
    flow : array-like
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
    numpy.ndarray
        Concentration of the compound in the infiltrating water [ng/m3].

    Notes
    -----
    Extraction to infiltration diffusion (deconvolution) is mathematically ill-posed and requires
    regularization to obtain a stable solution.
    """
    msg = "Extraction to infiltration diffusion (deconvolution) is not implemented yet"
    raise NotImplementedError(msg)


def compute_sigma_array(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    diffusivity: float = 0.1,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Compute scaled sigma values for diffusion based on flow and aquifer properties.

    Sigma represents the dimensionless spreading parameter for Gaussian filtering,
    expressed in units of array indices (time steps). It determines how many
    neighboring time steps are blended together when applying diffusive smoothing.

    The computation follows these steps:
    1. Calculate residence time (rt) for water parcels traveling through the aquifer
    2. Compute the diffusive spreading length: L_diff = sqrt(2 * D * rt) [m]
       This is the physical distance over which concentrations spread due to diffusion
    3. Compute the advective step size: dx = (Q * dt / V_pore) * L_aquifer [m]
       This is the physical distance the water moves during one time step
    4. Sigma = L_diff / dx converts the physical spreading into array index units

    Why divide by dx?
    The Gaussian filter operates on array indices, not physical distances. If the
    diffusive spreading is 10 meters and each time step moves water 2 meters, then
    sigma = 10/2 = 5 means the filter should blend across ~5 time steps. This
    normalization accounts for variable flow rates: faster flow means larger dx,
    so fewer time steps are blended (smaller sigma), even though the physical
    spreading remains the same.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges corresponding to the flow values.
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    diffusivity : float, optional
        Diffusivity of the compound in the aquifer [m2/day]. Default is 0.1.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Array of sigma values (in units of array indices), clipped to range [0, 100].
        Each value corresponds to a time step in the input flow series.
    """
    rt_array = residence_time(
        flow=flow,
        flow_tedges=tedges,
        aquifer_pore_volumes=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",
    )[0]  # Extract first pore volume

    # Interpolate NaN values using linear interpolation with nearest extrapolation
    valid_mask = ~np.isnan(rt_array)
    if np.any(valid_mask):
        rt_array = np.interp(np.arange(len(rt_array)), np.where(valid_mask)[0], rt_array[valid_mask])

    # Diffusive spreading length [m]: how far concentrations spread physically
    return np.sqrt(2 * diffusivity * rt_array)


def compute_scaled_sigma_array(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    diffusivity: float = 0.1,
    retardation_factor: float = 1.0,
    aquifer_length: float = 80.0,
) -> npt.NDArray[np.floating]:
    """Compute scaled sigma values for diffusion based on flow and aquifer properties.

    Sigma represents the dimensionless spreading parameter for Gaussian filtering,
    expressed in units of array indices (time steps). It determines how many
    neighboring time steps are blended together when applying diffusive smoothing.

    The computation follows these steps:
    1. Calculate residence time (rt) for water parcels traveling through the aquifer
    2. Compute the diffusive spreading length: L_diff = sqrt(2 * D * rt) [m]
       This is the physical distance over which concentrations spread due to diffusion
    3. Compute the advective step size: dx = (Q * dt / V_pore) * L_aquifer [m]
       This is the physical distance the water moves during one time step
    4. Sigma = L_diff / dx converts the physical spreading into array index units

    Why divide by dx?
    The Gaussian filter operates on array indices, not physical distances. If the
    diffusive spreading is 10 meters and each time step moves water 2 meters, then
    sigma = 10/2 = 5 means the filter should blend across ~5 time steps. This
    normalization accounts for variable flow rates: faster flow means larger dx,
    so fewer time steps are blended (smaller sigma), even though the physical
    spreading remains the same.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges corresponding to the flow values.
    aquifer_pore_volume : float
        Pore volume of the aquifer [m3].
    diffusivity : float, optional
        Diffusivity of the compound in the aquifer [m2/day]. Default is 0.1.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.
    aquifer_length : float, optional
        Length of the aquifer [m]. Default is 80.0.

    Returns
    -------
    numpy.ndarray
        Array of sigma values (in units of array indices), clipped to range [0, 100].
        Each value corresponds to a time step in the input flow series.
    """
    # Diffusive spreading length [m]: how far concentrations spread physically
    diffusive_spreading_length = compute_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        diffusivity=diffusivity,
        retardation_factor=retardation_factor,
    )

    # Advective step size [m]: how far water moves during one time step
    timedelta_at_departure = np.diff(tedges) / pd.to_timedelta(1, unit="D")
    volume_infiltrated_at_departure = flow * timedelta_at_departure
    dx = volume_infiltrated_at_departure / aquifer_pore_volume * aquifer_length

    # Sigma in array index units: number of time steps to blend
    sigma_array = diffusive_spreading_length / dx
    return np.clip(sigma_array, 0.0, 100.0)


def convolve_diffusion(
    *, input_signal: npt.ArrayLike, sigma_array: npt.ArrayLike, truncate: float = 4.0
) -> npt.NDArray[np.floating]:
    """Apply Gaussian filter with position-dependent sigma values.

    This function extends scipy.ndimage.gaussian_filter1d by allowing the standard
    deviation (sigma) of the Gaussian kernel to vary at each point in the signal.
    It implements the filter using a sparse convolution matrix where each row
    represents a Gaussian kernel with a locally-appropriate standard deviation.

    Parameters
    ----------
    input_signal : numpy.ndarray
        One-dimensional input array to be filtered.
    sigma_array : numpy.ndarray
        One-dimensional array of standard deviation values, must have same length
        as input_signal. Each value specifies the Gaussian kernel width at the
        corresponding position.
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    numpy.ndarray
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
    >>> import numpy as np
    >>> from gwtransport.diffusion import convolve_diffusion
    >>> # Create a sample signal
    >>> x = np.linspace(0, 10, 1000)
    >>> signal = np.exp(-((x - 3) ** 2)) + 0.5 * np.exp(-((x - 7) ** 2) / 0.5)

    >>> # Create position-dependent sigma values
    >>> diffusivity = 0.1  # diffusivity
    >>> dt = 0.001 * (1 + np.sin(2 * np.pi * x / 10))  # Varying time steps
    >>> dx = x[1] - x[0]
    >>> sigma_array = np.sqrt(2 * diffusivity * dt) / dx

    >>> # Apply the filter
    >>> filtered = convolve_diffusion(input_signal=signal, sigma_array=sigma_array)
    """
    # Convert to arrays for type safety
    input_signal = np.asarray(input_signal)
    sigma_array = np.asarray(sigma_array)

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


def deconvolve_diffusion(
    *, output_signal: npt.ArrayLike, sigma_array: npt.ArrayLike, truncate: float = 4.0
) -> npt.NDArray[np.floating]:
    """Apply Gaussian deconvolution with position-dependent sigma values.

    This function extends scipy.ndimage.gaussian_filter1d by allowing the standard
    deviation (sigma) of the Gaussian kernel to vary at each point in the signal.
    It implements the filter using a sparse convolution matrix where each row
    represents a Gaussian kernel with a locally-appropriate standard deviation.

    Parameters
    ----------
    output_signal : numpy.ndarray
        One-dimensional input array to be filtered.
    sigma_array : numpy.ndarray
        One-dimensional array of standard deviation values, must have same length
        as output_signal. Each value specifies the Gaussian kernel width at the
        corresponding position.
    truncate : float, optional
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    numpy.ndarray
        The filtered output signal. Has the same shape as output_signal.
    """
    msg = "Deconvolution is not implemented yet"
    raise NotImplementedError(msg)


def create_example_data(
    *, nx: int = 1000, domain_length: float = 10.0, diffusivity: float = 0.1
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
    x : numpy.ndarray
        Spatial coordinates.
    signal : numpy.ndarray
        Initial signal (sum of two Gaussians).
    sigma_array : numpy.ndarray
        Array of sigma values varying in space.
    dt : numpy.ndarray
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
    filtered = convolve_diffusion(input_signal=signal, sigma_array=sigma_array * 5)

    # Compare with regular Gaussian filter
    avg_sigma = np.mean(sigma_array)
    regular_filtered = ndimage.gaussian_filter1d(signal, avg_sigma)
    plt.figure(figsize=(10, 6))
    plt.plot(x, signal, label="Original signal", lw=0.8)
    plt.plot(x, filtered, label="Variable-sigma filtered", lw=1.0)

    plt.plot(x, regular_filtered, label="Regular Gaussian filter", lw=0.8, ls="--")
