"""
Fast Diffusive Transport Corrections via Gaussian Smoothing.

This module provides a computationally efficient approximation of diffusion/dispersion
using Gaussian smoothing. It is much faster than :mod:`gwtransport.diffusion` but
less physically accurate, especially under variable flow conditions.

Both ``diffusion_fast`` and :mod:`gwtransport.diffusion` add microdispersion and
molecular diffusion on top of macrodispersion captured by the APVD.

**When to use diffusion_fast vs diffusion:**

- Use ``diffusion_fast`` when: Speed is critical, flow and time steps are relatively
  constant, or you need real-time processing
- Use ``diffusion`` when: Physical accuracy is critical, flow varies significantly,
  or you're analyzing periods with changing conditions

See :ref:`concept-dispersion` for background on macrodispersion and microdispersion.

This module implements diffusion/dispersion processes that modify advective transport
in aquifer systems. Diffusion causes spreading and smoothing of concentration or
temperature fronts as they travel through the aquifer. While advection moves compounds
with water flow, diffusion causes spreading due to molecular diffusion, mechanical
dispersion, and thermal diffusion (for temperature).

Limitation: When ``flow_out`` is not provided, this fast approximation works best when
flow and tedges are relatively constant. The underlying assumption is that dx (spatial
step between cells) remains approximately constant, which holds for steady flow but
breaks down under highly variable conditions. When input time bins are non-uniform
(e.g., from signal compression), provide ``flow_out`` to apply Gaussian smoothing on
the (typically uniform) output grid instead. For scenarios with significant flow
variability, consider using :mod:`gwtransport.diffusion` instead.

Available functions:

- :func:`infiltration_to_extraction` - Apply diffusion during infiltration to extraction
  transport. Combines advection (via residence time) with diffusion (via Gaussian smoothing).
  Computes position-dependent diffusion based on local residence time and returns concentration
  or temperature in extracted water on the extraction time grid.

- :func:`extraction_to_infiltration` - Inverse diffusion via Tikhonov regularization. Builds
  the combined forward matrix (advection + Gaussian diffusion) and solves the inverse problem
  to reconstruct infiltration concentrations from extraction data.

- :func:`gamma_infiltration_to_extraction` - Gamma-distributed pore volumes with fast
  Gaussian diffusion. Convenience wrapper that parameterizes the pore volume distribution
  as a gamma distribution.

- :func:`gamma_extraction_to_infiltration` - Gamma-distributed pore volumes, inverse
  fast Gaussian diffusion. Symmetric inverse of gamma_infiltration_to_extraction.

- :func:`convolve_diffusion` - Apply variable-sigma Gaussian filtering. Extends
  scipy.ndimage.gaussian_filter1d to position-dependent sigma using sparse matrix representation
  for efficiency. Handles boundary conditions via nearest-neighbor extrapolation.

- :func:`create_example_data` - Generate test data for demonstrating diffusion effects with
  signals having varying time steps and corresponding sigma arrays. Useful for testing and
  validation.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from scipy.special import erf

from gwtransport import gamma
from gwtransport.advection_utils import _infiltration_to_extraction_weights
from gwtransport.residence_time import residence_time
from gwtransport.utils import solve_inverse_transport

# Minimum coefficient sum to consider a row valid
_EPSILON_COEFF_SUM = 1e-10


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    flow_out: npt.ArrayLike | None = None,
    suppress_uniform_tedges_check: bool = False,
) -> npt.NDArray[np.floating]:
    """Compute diffusion during 1D transport using fast Gaussian smoothing.

    Combines advection (via pore volume distribution) with diffusion (via Gaussian
    smoothing) to produce extraction concentrations on the ``cout_tedges`` time grid.
    This is the fast approximate counterpart of
    :func:`gwtransport.diffusion.infiltration_to_extraction`.

    When ``flow_out`` is provided, diffusion is applied on the output grid
    (advect-then-smooth), which correctly handles non-uniform ``tedges``
    (e.g., from signal compression). Without ``flow_out``, diffusion is applied
    on the input grid and ``tedges`` must have constant time steps.

    Parameters
    ----------
    cin : array-like
        Concentration or temperature of the compound in the infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Has length len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution of
        flow paths.
    mean_streamline_length : float
        Mean travel distance [m]. Must be positive.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity [m2/day]. Must be non-negative.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations.
        Default is 3.0.
    flow_out : array-like or None, optional
        Flow rate [m3/day] on the output grid (aligned to ``cout_tedges``).
        Required when ``tedges`` has non-uniform time steps. Length must
        equal ``len(cout_tedges) - 1``. Default is None.
    suppress_uniform_tedges_check : bool, optional
        Skip the constant-dt check on ``tedges`` when ``flow_out`` is None.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in extracted water. Length equals
        len(cout_tedges) - 1. NaN values indicate time periods with no valid
        contributions from the infiltration data.

    See Also
    --------
    gwtransport.diffusion.infiltration_to_extraction : Physically rigorous analytical solution
        that supports per-pore-volume arrays for streamline_length, molecular_diffusivity,
        and longitudinal_dispersivity.
    extraction_to_infiltration : Inverse operation
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    A single ``mean_streamline_length`` is shared across all pore volumes. This
    assumes streamline-length heterogeneity is captured solely through the
    pore-volume distribution: the effective cross-sectional area
    ``A_eff = V_p / L_mean`` varies with V_p while the path length L is held
    fixed. This is appropriate for many systems but breaks down for
    partially-penetrating wells or wedge-shaped capture zones, where path
    length itself varies between streamtubes. In those cases, use
    :func:`gwtransport.diffusion.infiltration_to_extraction` with a per-streamtube
    ``streamline_length`` array instead.
    """
    # Convert inputs
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    tedges = pd.DatetimeIndex(tedges)
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    if flow_out is not None:
        flow_out = np.asarray(flow_out, dtype=float)

    n_pore_volumes = len(aquifer_pore_volumes)

    # Input validation
    _validate_inputs(
        cin_or_cout=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        is_forward=True,
        flow_out=flow_out,
    )

    # Dispersion warning
    if n_pore_volumes > 1 and mean_longitudinal_dispersivity > 0 and not suppress_dispersion_warning:
        _warn_dispersion()

    # Build advection weight matrix (needed in both branches)
    w_adv = _infiltration_to_extraction_weights(
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=flow,
        retardation_factor=retardation_factor,
    )

    if flow_out is None:
        # Original algorithm: smooth-then-advect (requires uniform tedges)
        _check_uniform_tedges(tedges=tedges, suppress=suppress_uniform_tedges_check)

        sigma_array = _compute_sigma(
            flow=flow,
            tedges=tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=mean_streamline_length,
            molecular_diffusivity=mean_molecular_diffusivity,
            longitudinal_dispersivity=mean_longitudinal_dispersivity,
            retardation_factor=retardation_factor,
            direction="infiltration_to_extraction",
        )
        cin_diffused = convolve_diffusion(
            input_signal=cin, sigma_array=sigma_array, asymptotic_cutoff_sigma=asymptotic_cutoff_sigma
        )
        cout = w_adv @ cin_diffused
    else:
        # New algorithm: advect-then-smooth (works with any tedges spacing)
        cout_unsmoothed = w_adv @ cin

        # Invalid rows (no breakthrough yet) have near-zero values that would
        # bleed into valid neighbors during smoothing. Use normalized convolution:
        # smooth(signal * mask) / smooth(mask) to properly renormalize the kernel.
        total_coeff = np.sum(w_adv, axis=1)
        invalid_mask = total_coeff < _EPSILON_COEFF_SUM
        cout_unsmoothed[invalid_mask] = 0.0
        validity = np.where(invalid_mask, 0.0, 1.0)

        sigma_out = _compute_sigma(
            flow=flow_out,
            tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=mean_streamline_length,
            molecular_diffusivity=mean_molecular_diffusivity,
            longitudinal_dispersivity=mean_longitudinal_dispersivity,
            retardation_factor=retardation_factor,
            direction="extraction_to_infiltration",
        )
        smoothed = convolve_diffusion(
            input_signal=cout_unsmoothed, sigma_array=sigma_out, asymptotic_cutoff_sigma=asymptotic_cutoff_sigma
        )
        smoothed_validity = convolve_diffusion(
            input_signal=validity, sigma_array=sigma_out, asymptotic_cutoff_sigma=asymptotic_cutoff_sigma
        )
        with np.errstate(invalid="ignore"):
            cout = np.where(smoothed_validity > _EPSILON_COEFF_SUM, smoothed / smoothed_validity, np.nan)

    # Mark invalid rows as NaN (where no cin has broken through)
    total_coeff = np.sum(w_adv, axis=1)
    cout[total_coeff < _EPSILON_COEFF_SUM] = np.nan

    return cout


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    regularization_strength: float = 1e-10,
    flow_out: npt.ArrayLike | None = None,
    suppress_uniform_tedges_check: bool = False,
) -> npt.NDArray[np.floating]:
    """Reconstruct infiltration concentration from extracted water via Tikhonov inversion.

    Inverts the forward transport model (Gaussian diffusion + advection) by building
    the combined forward matrix and solving via Tikhonov regularization. This is the
    fast approximate counterpart of
    :func:`gwtransport.diffusion.extraction_to_infiltration`.

    When ``flow_out`` is provided, diffusion is applied on the output grid,
    which correctly handles non-uniform ``tedges``. Without ``flow_out``,
    diffusion is applied on the input grid and ``tedges`` must have constant
    time steps.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Has length of len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Has length of len(cout) + 1.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3].
    mean_streamline_length : float
        Mean travel distance [m]. Must be positive.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity [m2/day]. Must be non-negative.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations.
        Default is 3.0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-10.
    flow_out : array-like or None, optional
        Flow rate [m3/day] on the output grid (aligned to ``cout_tedges``).
        Required when ``tedges`` has non-uniform time steps. Length must
        equal ``len(cout_tedges) - 1``. Default is None.
    suppress_uniform_tedges_check : bool, optional
        Skip the constant-dt check on ``tedges`` when ``flow_out`` is None.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in infiltrating water. Length equals
        len(tedges) - 1. NaN values indicate time periods with no valid
        contributions from the extraction data.

    Warns
    -----
    UserWarning
        When the forward matrix is rank-deficient. This occurs with constant
        flow when the residence time is an integer multiple of the time step
        width. To fix, adjust ``aquifer_pore_volumes`` slightly (e.g.,
        multiply by 1.001).

    See Also
    --------
    infiltration_to_extraction : Forward operation
    gwtransport.diffusion.extraction_to_infiltration : Analytically correct deconvolution
        that supports per-pore-volume arrays for streamline_length, molecular_diffusivity,
        and longitudinal_dispersivity.
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    A single ``mean_streamline_length`` is shared across all pore volumes. This
    assumes streamline-length heterogeneity is captured solely through the
    pore-volume distribution: the effective cross-sectional area
    ``A_eff = V_p / L_mean`` varies with V_p while the path length L is held
    fixed. This is appropriate for many systems but breaks down for
    partially-penetrating wells or wedge-shaped capture zones, where path
    length itself varies between streamtubes. In those cases, use
    :func:`gwtransport.diffusion.extraction_to_infiltration` with a per-streamtube
    ``streamline_length`` array instead.
    """
    # Convert inputs
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    cout = np.asarray(cout, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    if flow_out is not None:
        flow_out = np.asarray(flow_out, dtype=float)

    n_pore_volumes = len(aquifer_pore_volumes)

    # Input validation
    _validate_inputs(
        cin_or_cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        is_forward=False,
        flow_out=flow_out,
    )

    # Dispersion warning
    if n_pore_volumes > 1 and mean_longitudinal_dispersivity > 0 and not suppress_dispersion_warning:
        _warn_dispersion()

    n_cin = len(tedges) - 1
    n_cout = len(cout_tedges) - 1

    # Build advection weight matrix
    w_adv = _infiltration_to_extraction_weights(
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=flow,
        retardation_factor=retardation_factor,
    )

    if flow_out is None:
        # Original algorithm: G on input grid, W_forward = W_adv @ G
        _check_uniform_tedges(tedges=tedges, suppress=suppress_uniform_tedges_check)

        sigma_array = _compute_sigma(
            flow=flow,
            tedges=tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=mean_streamline_length,
            molecular_diffusivity=mean_molecular_diffusivity,
            longitudinal_dispersivity=mean_longitudinal_dispersivity,
            retardation_factor=retardation_factor,
            direction="infiltration_to_extraction",
        )
        g_matrix = _build_gaussian_matrix(
            n=n_cin, sigma_array=sigma_array, asymptotic_cutoff_sigma=asymptotic_cutoff_sigma
        )
        # Combined forward matrix: W_adv @ G via sparse @ dense (avoids dense G allocation)
        w_forward = (g_matrix.T @ w_adv.T).T
    else:
        # New algorithm: G on output grid, W_forward = G @ W_adv
        sigma_out = _compute_sigma(
            flow=flow_out,
            tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=mean_streamline_length,
            molecular_diffusivity=mean_molecular_diffusivity,
            longitudinal_dispersivity=mean_longitudinal_dispersivity,
            retardation_factor=retardation_factor,
            direction="extraction_to_infiltration",
        )
        g_matrix = _build_gaussian_matrix(
            n=n_cout, sigma_array=sigma_out, asymptotic_cutoff_sigma=asymptotic_cutoff_sigma
        )
        w_forward = g_matrix @ w_adv  # sparse @ dense

    return solve_inverse_transport(
        w_forward=w_forward,
        observed=cout,
        n_output=n_cin,
        regularization_strength=regularization_strength,
        warn_rank_deficient=True,
    )


def gamma_infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    flow_out: npt.ArrayLike | None = None,
    suppress_uniform_tedges_check: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Compute extracted concentration with fast Gaussian diffusion for gamma-distributed pore volumes.

    Combines advective transport (based on gamma-distributed pore volumes) with
    fast Gaussian diffusion. This is a convenience wrapper around
    :func:`infiltration_to_extraction` that parameterizes the aquifer pore volume
    distribution as a (shifted) gamma distribution.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Has length len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
    mean : float, optional
        Mean of the gamma distribution of the aquifer pore volume. Must be strictly
        greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma distribution of the aquifer pore volume
        (invariant under the ``loc`` shift).
    loc : float, optional
        Location (minimum pore volume) of the gamma distribution. Must satisfy
        ``0 <= loc < mean``. Default is ``0.0``.
    alpha : float, optional
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0).
    beta : float, optional
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0).
    n_bins : int, optional
        Number of bins to discretize the gamma distribution. Default is 100.
    mean_streamline_length : float
        Mean travel distance through the aquifer [m] averaged over all aquifer
        pore volumes. Must be positive. For per-pore-volume arrays, use
        :func:`gwtransport.diffusion.gamma_infiltration_to_extraction`.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity [m2/day] averaged over all
        aquifer pore volumes. Must be non-negative. For per-pore-volume
        arrays, use :func:`gwtransport.diffusion.gamma_infiltration_to_extraction`.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity [m] averaged over all aquifer pore
        volumes. Must be non-negative. For per-pore-volume arrays, use
        :func:`gwtransport.diffusion.gamma_infiltration_to_extraction`.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations.
        Default is 3.0.
    flow_out : array-like or None, optional
        Flow rate [m3/day] on the output grid (aligned to ``cout_tedges``).
        Required when ``tedges`` has non-uniform time steps. Length must
        equal ``len(cout_tedges) - 1``. Default is None.
    suppress_uniform_tedges_check : bool, optional
        Skip the constant-dt check on ``tedges`` when ``flow_out`` is None.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in extracted water. Length equals
        len(cout_tedges) - 1. NaN values indicate time periods with no valid
        contributions from the infiltration data.

    See Also
    --------
    infiltration_to_extraction : Transport with explicit pore volume distribution
    gamma_extraction_to_infiltration : Reverse operation (deconvolution)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.diffusion.gamma_infiltration_to_extraction : Physically rigorous analytical
        solution that supports per-pore-volume arrays for streamline_length,
        molecular_diffusivity, and longitudinal_dispersivity.
    gwtransport.advection.gamma_infiltration_to_extraction : Pure advection (no dispersion)
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The APVD is only time-invariant under the steady-streamlines assumption
    (see :ref:`assumption-steady-streamlines`).

    The spreading from the gamma-distributed pore volumes represents macrodispersion
    (aquifer-scale heterogeneity). When ``std`` comes from calibration on measurements,
    it absorbs all mixing: macrodispersion, microdispersion, and an average molecular
    diffusion contribution. When ``std`` comes from streamline analysis, it represents
    macrodispersion only; microdispersion and molecular diffusion can be added via the
    dispersion parameters.
    See :ref:`concept-dispersion-scales` for guidance on when to add microdispersion.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion_fast import gamma_infiltration_to_extraction
    >>>
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>> cin = np.zeros(len(tedges) - 1)
    >>> cin[5:10] = 1.0
    >>> flow = np.ones(len(tedges) - 1) * 100.0
    >>>
    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=500.0,
    ...     std=100.0,
    ...     n_bins=5,
    ...     mean_streamline_length=100.0,
    ...     mean_molecular_diffusivity=1e-4,
    ...     mean_longitudinal_dispersivity=1.0,
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        suppress_dispersion_warning=suppress_dispersion_warning,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
        flow_out=flow_out,
        suppress_uniform_tedges_check=suppress_uniform_tedges_check,
    )


def gamma_extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    regularization_strength: float = 1e-10,
    flow_out: npt.ArrayLike | None = None,
    suppress_uniform_tedges_check: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Reconstruct infiltration concentration from extracted water for gamma-distributed pore volumes.

    Inverts the forward transport model (fast Gaussian diffusion + advection with
    gamma-distributed pore volumes) via Tikhonov regularization. This is a convenience
    wrapper around :func:`extraction_to_infiltration` that parameterizes the aquifer
    pore volume distribution as a (shifted) gamma distribution.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Has length of len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Has length of len(cout) + 1.
    mean : float, optional
        Mean of the gamma distribution of the aquifer pore volume. Must be strictly
        greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma distribution of the aquifer pore volume
        (invariant under the ``loc`` shift).
    loc : float, optional
        Location (minimum pore volume) of the gamma distribution. Must satisfy
        ``0 <= loc < mean``. Default is ``0.0``.
    alpha : float, optional
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0).
    beta : float, optional
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0).
    n_bins : int, optional
        Number of bins to discretize the gamma distribution. Default is 100.
    mean_streamline_length : float
        Mean travel distance through the aquifer [m] averaged over all aquifer
        pore volumes. Must be positive. For per-pore-volume arrays, use
        :func:`gwtransport.diffusion.gamma_extraction_to_infiltration`.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity [m2/day] averaged over all
        aquifer pore volumes. Must be non-negative. For per-pore-volume
        arrays, use :func:`gwtransport.diffusion.gamma_extraction_to_infiltration`.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity [m] averaged over all aquifer pore
        volumes. Must be non-negative. For per-pore-volume arrays, use
        :func:`gwtransport.diffusion.gamma_extraction_to_infiltration`.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations.
        Default is 3.0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-10.
    flow_out : array-like or None, optional
        Flow rate [m3/day] aligned to ``cout_tedges``. When provided, the
        Gaussian matrix operates on the output grid. Length must equal
        ``len(cout_tedges) - 1``. Default is None.
    suppress_uniform_tedges_check : bool, optional
        When True, skip the check that ``tedges`` has constant time steps
        (only relevant when ``flow_out`` is None). Default is False.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in infiltrating water. Length equals
        len(tedges) - 1. NaN values indicate time periods with no valid
        contributions from the extraction data.

    See Also
    --------
    extraction_to_infiltration : Deconvolution with explicit pore volume distribution
    gamma_infiltration_to_extraction : Forward operation (convolution)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.diffusion.gamma_extraction_to_infiltration : Physically rigorous analytical
        solution that supports per-pore-volume arrays for streamline_length,
        molecular_diffusivity, and longitudinal_dispersivity.
    gwtransport.advection.gamma_extraction_to_infiltration : Pure advection (no dispersion)
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The APVD is only time-invariant under the steady-streamlines assumption
    (see :ref:`assumption-steady-streamlines`).

    The spreading from the gamma-distributed pore volumes represents macrodispersion
    (aquifer-scale heterogeneity). When ``std`` comes from calibration on measurements,
    it absorbs all mixing: macrodispersion, microdispersion, and an average molecular
    diffusion contribution. When ``std`` comes from streamline analysis, it represents
    macrodispersion only; microdispersion and molecular diffusion can be added via the
    dispersion parameters.
    See :ref:`concept-dispersion-scales` for guidance on when to add microdispersion.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion_fast import gamma_extraction_to_infiltration
    >>>
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>> cout = np.zeros(len(cout_tedges) - 1)
    >>> cout[5:10] = 1.0
    >>> flow = np.ones(len(tedges) - 1) * 100.0
    >>>
    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=500.0,
    ...     std=100.0,
    ...     n_bins=5,
    ...     mean_streamline_length=100.0,
    ...     mean_molecular_diffusivity=1e-4,
    ...     mean_longitudinal_dispersivity=1.0,
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        suppress_dispersion_warning=suppress_dispersion_warning,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
        regularization_strength=regularization_strength,
        flow_out=flow_out,
        suppress_uniform_tedges_check=suppress_uniform_tedges_check,
    )


def _compute_sigma(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float,
    direction: str,
) -> npt.NDArray[np.floating]:
    """Compute sigma in array-index units with moment-based averaging.

    The per-streamtube variance of the Gaussian kernel (in index units)
    is sigma_idx^2(V) = L_diff^2(V) * V^2 * R^2 / (Q * dt * L)^2.
    L_diff^2 = 2*D_m*tau + 2*alpha_L*L is linear in V (via tau ~ V),
    so sigma_idx^2 is cubic + quadratic in V.  The averaged variance
    over the pore volume distribution is (law of total variance):

        E[sigma_idx^2] = R^2 / (Q*dt*L)^2
            * (2*D_m*tau_bar/Vbar * E[V^3] + 2*alpha_L*L * E[V^2])

    When a single pore volume is given this reduces to the classical formula.

    See :ref:`concept-variance-components` for the derivation.

    Parameters
    ----------
    flow : array-like
        Flow rate [m3/day].
    tedges : DatetimeIndex
        Time edges for flow data.
    aquifer_pore_volumes : array-like
        Pore volumes [m3]. Moments E[V^2] and E[V^3] are computed
        from this array.
    streamline_length : float
        Streamline length [m].
    molecular_diffusivity : float
        Effective molecular diffusivity [m2/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity [m].
    retardation_factor : float
        Retardation factor [-].
    direction : {'infiltration_to_extraction', 'extraction_to_infiltration'}
        Direction for residence time computation.

    Returns
    -------
    ndarray
        Sigma values in array-index units, clipped to [0, 100].
    """
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    mean_pv = float(np.mean(aquifer_pore_volumes))
    ev2 = float(np.mean(aquifer_pore_volumes**2))
    ev3 = float(np.mean(aquifer_pore_volumes**3))

    # Residence time at the mean pore volume [days]
    rt_array = residence_time(
        flow=flow,
        flow_tedges=tedges,
        aquifer_pore_volumes=mean_pv,
        retardation_factor=retardation_factor,
        direction=direction,
    )[0]

    # Interpolate NaN values
    valid_mask = ~np.isnan(rt_array)
    if np.any(valid_mask):
        rt_array = np.interp(np.arange(len(rt_array)), np.where(valid_mask)[0], rt_array[valid_mask])

    # Two independent spatial variance components [m^2]:
    #   molecular:    2 * D_m * tau          (linear in V via tau)
    #   dispersive:   2 * alpha_L * L        (constant in V)
    mol_var_x = 2.0 * molecular_diffusivity * rt_array
    disp_var_x = 2.0 * longitudinal_dispersivity * streamline_length

    # Averaged variance in index units (see docstring):
    var_numerator = mol_var_x / mean_pv * ev3 + disp_var_x * ev2

    timedelta = np.diff(tedges) / pd.to_timedelta(1, unit="D")
    # q_dt_l_over_r = Q * dt * L / R has units m^4 (it is the squared scaling
    # factor that converts spatial variance [m^2] into index-space variance).
    q_dt_l_over_r = flow * timedelta * streamline_length / retardation_factor

    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_sq = np.where(q_dt_l_over_r > 0.0, var_numerator / q_dt_l_over_r**2, 0.0)

    return np.clip(np.sqrt(np.maximum(sigma_sq, 0.0)), 0.0, 100.0)


def _check_uniform_tedges(*, tedges: pd.DatetimeIndex, suppress: bool) -> None:
    """Raise ValueError if tedges has non-constant time steps.

    Raises
    ------
    ValueError
        If tedges has non-constant time steps and suppress is False.
    """
    if suppress:
        return
    dt = np.diff(tedges)
    if not np.all(dt == dt[0]):
        msg = (
            "tedges must have constant time steps when flow_out is not provided. "
            "Either provide flow_out or set suppress_uniform_tedges_check=True."
        )
        raise ValueError(msg)


def _build_gaussian_matrix(
    *, n: int, sigma_array: npt.ArrayLike, asymptotic_cutoff_sigma: float | None = 3.0
) -> sparse.csr_matrix:
    """Build sparse Gaussian convolution matrix with CDF-integrated bin weights.

    Each row represents a position-specific Gaussian kernel where weights are
    computed by integrating the Gaussian CDF over each bin, rather than
    point-sampling the PDF. This is more accurate for small sigma values
    (< ~2 bins) where the PDF peak can fall between bin centers.

    Parameters
    ----------
    n : int
        Length of the signal (number of bins).
    sigma_array : array-like
        Array of sigma values of length n.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the kernel at this many standard deviations. Set to None to
        disable truncation (use full signal length). Default is 3.0.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse convolution matrix of shape (n, n).
    """
    sigma_array = np.asarray(sigma_array)

    # Handle zero sigma values
    zero_mask = sigma_array == 0
    if np.all(zero_mask):
        return sparse.eye(n, format="csr", dtype=float)  # pyright: ignore[reportReturnType]

    # Get maximum kernel size and create position arrays
    max_sigma = np.max(sigma_array)
    truncate = asymptotic_cutoff_sigma if asymptotic_cutoff_sigma is not None else n / max(max_sigma, 1.0)
    max_radius = int(truncate * max_sigma + 0.5)

    # Create arrays for all possible kernel positions
    positions = np.arange(-max_radius, max_radius + 1)

    # Create a mask for valid sigma values
    valid_sigma = ~zero_mask
    valid_indices = np.where(valid_sigma)[0]

    # Shape: (n_valid_points, 1)
    center_positions = valid_indices[:, np.newaxis]
    # Shape: (1, max_kernel_size)
    kernel_positions = positions[np.newaxis, :]

    # Calculate CDF-integrated weights for proper bin averaging.
    # Compute erf at unique bin edges, then take differences. Adjacent bins
    # share an edge, so this halves the number of erf evaluations.
    sigmas = sigma_array[valid_sigma][:, np.newaxis]
    sqrt2 = np.sqrt(2)
    edges = np.arange(-max_radius - 0.5, max_radius + 1.5)  # (2*max_radius + 2,)
    erf_at_edges = erf(edges[np.newaxis, :] / (sigmas * sqrt2))  # (n_valid, 2*max_radius+2)
    weights = 0.5 * np.diff(erf_at_edges, axis=1)  # (n_valid, 2*max_radius+1)

    # Normalize each kernel
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Calculate absolute positions in the signal
    absolute_positions = center_positions + kernel_positions

    # Handle boundary conditions (nearest-neighbor extrapolation)
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

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def convolve_diffusion(
    *, input_signal: npt.ArrayLike, sigma_array: npt.ArrayLike, asymptotic_cutoff_sigma: float | None = 3.0
) -> npt.NDArray[np.floating]:
    """Apply Gaussian filter with position-dependent sigma values.

    This function extends scipy.ndimage.gaussian_filter1d by allowing the standard
    deviation (sigma) of the Gaussian kernel to vary at each point in the signal.
    It implements the filter using a sparse convolution matrix where each row
    represents a Gaussian kernel with a locally-appropriate standard deviation.

    Kernel weights are computed by integrating the Gaussian CDF over each bin,
    which is more accurate than point-sampling the PDF for small sigma values.

    Parameters
    ----------
    input_signal : array-like
        One-dimensional input array to be filtered.
    sigma_array : array-like
        One-dimensional array of standard deviation values, must have same length
        as input_signal.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the filter at this many standard deviations. Set to None to
        disable truncation. Default is 3.0.

    Returns
    -------
    numpy.ndarray
        The filtered input signal. Has the same shape as input_signal.

    Raises
    ------
    ValueError
        If input_signal and sigma_array do not have the same length.

    See Also
    --------
    scipy.ndimage.gaussian_filter1d : Fixed-sigma Gaussian filtering

    Notes
    -----
    At the boundaries, the outer values are repeated to avoid edge effects
    (equivalent to mode='nearest' in `scipy.ndimage.gaussian_filter1d`).

    Examples
    --------
    >>> import numpy as np
    >>> from gwtransport.diffusion_fast import convolve_diffusion
    >>> # Create a sample signal
    >>> x = np.linspace(0, 10, 1000)
    >>> signal = np.exp(-((x - 3) ** 2)) + 0.5 * np.exp(-((x - 7) ** 2) / 0.5)

    >>> # Create position-dependent sigma values
    >>> diffusivity = 0.1
    >>> dt = 0.001 * (1 + np.sin(2 * np.pi * x / 10))
    >>> dx = x[1] - x[0]
    >>> sigma_array = np.sqrt(2 * diffusivity * dt) / dx

    >>> # Apply the filter
    >>> filtered = convolve_diffusion(input_signal=signal, sigma_array=sigma_array)
    """
    input_signal = np.asarray(input_signal)
    sigma_array = np.asarray(sigma_array)

    if len(input_signal) != len(sigma_array):
        msg = "Input signal and sigma array must have the same length"
        raise ValueError(msg)

    n = len(input_signal)

    # Handle zero sigma values
    if np.all(sigma_array == 0):
        return input_signal.copy()

    g_matrix = _build_gaussian_matrix(n=n, sigma_array=sigma_array, asymptotic_cutoff_sigma=asymptotic_cutoff_sigma)
    return g_matrix.dot(input_signal)


def create_example_data(
    *,
    nx: int = 1000,
    domain_length: float = 10.0,
    diffusivity: float = 0.1,
    seed: int | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Create example data for demonstrating variable-sigma diffusion.

    Parameters
    ----------
    nx : int, optional
        Number of spatial points. Default is 1000.
    domain_length : float, optional
        Domain length. Default is 10.0.
    diffusivity : float, optional
        Diffusivity. Default is 0.1.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    x : numpy.ndarray
        Spatial coordinates.
    signal : numpy.ndarray
        Initial signal (sum of two Gaussians with noise).
    sigma_array : numpy.ndarray
        Array of sigma values varying in space.
    dt : numpy.ndarray
        Array of time steps varying in space.
    """
    rng = np.random.default_rng(seed)

    # Create spatial grid
    x = np.linspace(0, domain_length, nx)
    dx = x[1] - x[0]

    # Create initial signal (two Gaussians)
    signal = np.exp(-((x - 3) ** 2)) + 0.5 * np.exp(-((x - 7) ** 2) / 0.5) + 0.1 * rng.standard_normal(nx)

    # Create varying time steps
    dt = 0.001 * (1 + np.sin(2 * np.pi * x / domain_length))

    # Calculate corresponding sigma values
    sigma_array = np.sqrt(2 * diffusivity * dt) / dx

    return x, signal, sigma_array, dt


def _validate_inputs(
    *,
    cin_or_cout: np.ndarray,
    flow: np.ndarray,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: np.ndarray,
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    is_forward: bool,
    flow_out: np.ndarray | None = None,
) -> None:
    """Validate inputs for infiltration_to_extraction and extraction_to_infiltration.

    Raises
    ------
    ValueError
        If array lengths are inconsistent, mean_molecular_diffusivity or
        mean_longitudinal_dispersivity are negative, cin or cout or flow
        contain NaN values, aquifer_pore_volumes contains non-positive
        values, or mean_streamline_length is non-positive.
    """
    if is_forward:
        if len(tedges) != len(cin_or_cout) + 1:
            msg = "tedges must have one more element than cin"
            raise ValueError(msg)
    elif len(cout_tedges) != len(cin_or_cout) + 1:
        msg = "cout_tedges must have one more element than cout"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if mean_molecular_diffusivity < 0:
        msg = "mean_molecular_diffusivity must be non-negative"
        raise ValueError(msg)
    if mean_longitudinal_dispersivity < 0:
        msg = "mean_longitudinal_dispersivity must be non-negative"
        raise ValueError(msg)
    if np.any(np.isnan(cin_or_cout)):
        msg = f"{'cin' if is_forward else 'cout'} contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(flow < 0):
        msg = "flow must be non-negative (negative flow not supported)"
        raise ValueError(msg)
    if np.any(aquifer_pore_volumes <= 0):
        msg = "aquifer_pore_volumes must be positive"
        raise ValueError(msg)
    if mean_streamline_length <= 0:
        msg = "mean_streamline_length must be positive"
        raise ValueError(msg)
    if flow_out is not None:
        n_cout = len(cout_tedges) - 1
        if len(flow_out) != n_cout:
            msg = f"flow_out must have length len(cout_tedges) - 1 = {n_cout}, got {len(flow_out)}"
            raise ValueError(msg)
        if np.any(np.isnan(flow_out)):
            msg = "flow_out contains NaN values, which are not allowed"
            raise ValueError(msg)
        if np.any(flow_out <= 0):
            msg = "flow_out must be positive"
            raise ValueError(msg)


def _warn_dispersion() -> None:
    """Emit warning about combining multiple pore volumes with dispersivity."""
    msg = (
        "Using multiple aquifer_pore_volumes with non-zero mean_longitudinal_dispersivity. "
        "Both represent spreading from velocity heterogeneity at different scales.\n\n"
        "This is appropriate when:\n"
        "  - APVD comes from streamline analysis (explicit geometry)\n"
        "  - You want to add unresolved microdispersion and molecular diffusion\n\n"
        "This may double-count spreading when:\n"
        "  - APVD was calibrated from measurements (microdispersion already included)\n\n"
        "For gamma-parameterized APVD, consider using the 'equivalent APVD std' approach\n"
        "from 05_Diffusion_Dispersion.ipynb to combine both effects in the faster advection\n"
        "module. Note: this variance combination is only valid for continuous (gamma)\n"
        "distributions, not for discrete streamline volumes.\n"
        "Suppress this warning with suppress_dispersion_warning=True."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)
