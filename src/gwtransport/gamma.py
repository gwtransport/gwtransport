"""
Gamma Distribution Utilities for Aquifer Pore Volume Heterogeneity.

This module provides utilities for working with gamma distributions to model heterogeneous
aquifer pore volumes in groundwater transport analysis. The gamma distribution offers a
flexible two-parameter model for representing the natural variability in flow path lengths
and residence times within aquifer systems. In heterogeneous aquifers, water travels through
multiple flow paths with different pore volumes, and the gamma distribution provides a
realistic representation of this heterogeneity.

Available functions:

- :func:`parse_parameters` - Parse and validate gamma distribution parameters from either
  (alpha, beta) or (mean, std). Ensures exactly one parameter pair is provided and validates
  positivity constraints.

- :func:`mean_std_to_alpha_beta` - Convert physically intuitive (mean, std) parameters to
  gamma shape/scale parameters. Uses formulas: alpha = mean^2 / std^2 and beta = std^2 / mean.

- :func:`alpha_beta_to_mean_std` - Convert gamma (alpha, beta) parameters back to (mean, std)
  for physical interpretation. Uses formulas: mean = alpha * beta and std = sqrt(alpha) * beta.

- :func:`bins` - Primary function for transport modeling. Creates discrete probability bins from
  continuous gamma distribution with equal-probability bins (default) or custom quantile edges.
  Returns bin edges, expected values (mean pore volume within each bin), and probability masses
  (weight in transport calculations).

- :func:`bin_masses` - Calculate probability mass for custom bin edges using incomplete gamma
  function. Lower-level function used internally by bins().

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import gammainc
from scipy.stats import gamma as gamma_dist


def parse_parameters(
    *,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[float, float]:
    """
    Parse parameters for gamma distribution.

    Either alpha and beta or mean and std must be provided.

    Parameters
    ----------
    alpha : float, optional
        Shape parameter of gamma distribution (must be > 0)
    beta : float, optional
        Scale parameter of gamma distribution (must be > 0)
    mean : float, optional
        Mean of the gamma distribution.
    std : float, optional
        Standard deviation of the gamma distribution.

    Returns
    -------
    alpha : float
        Shape parameter of gamma distribution
    beta : float
        Scale parameter of gamma distribution

    Raises
    ------
    ValueError
        If both alpha and beta are None or if both mean and std are None.
        If alpha or beta are not positive.
    """
    if alpha is None or beta is None:
        if mean is None or std is None:
            msg = "Either alpha and beta or mean and std must be provided."
            raise ValueError(msg)

        alpha, beta = mean_std_to_alpha_beta(mean=mean, std=std)

    if alpha <= 0 or beta <= 0:
        msg = "Alpha and beta must be positive"
        raise ValueError(msg)

    return alpha, beta


def mean_std_to_alpha_beta(*, mean: float, std: float) -> tuple[float, float]:
    """
    Convert mean and standard deviation of gamma distribution to shape and scale parameters.

    Parameters
    ----------
    mean : float
        Mean of the gamma distribution.
    std : float
        Standard deviation of the gamma distribution.

    Returns
    -------
    alpha : float
        Shape parameter of gamma distribution
    beta : float
        Scale parameter of gamma distribution

    See Also
    --------
    alpha_beta_to_mean_std : Convert shape and scale parameters to mean and std
    parse_parameters : Parse and validate gamma distribution parameters

    Examples
    --------
    >>> from gwtransport.gamma import mean_std_to_alpha_beta
    >>> mean_pore_volume = 30000.0  # m³
    >>> std_pore_volume = 8100.0  # m³
    >>> alpha, beta = mean_std_to_alpha_beta(mean=mean_pore_volume, std=std_pore_volume)
    >>> print(f"Shape parameter (alpha): {alpha:.2f}")
    Shape parameter (alpha): 13.72
    >>> print(f"Scale parameter (beta): {beta:.2f}")
    Scale parameter (beta): 2187.00
    """
    alpha = mean**2 / std**2
    beta = std**2 / mean
    return alpha, beta


def alpha_beta_to_mean_std(*, alpha: float, beta: float) -> tuple[float, float]:
    """
    Convert shape and scale parameters of gamma distribution to mean and standard deviation.

    Parameters
    ----------
    alpha : float
        Shape parameter of the gamma distribution.
    beta : float
        Scale parameter of the gamma distribution.

    Returns
    -------
    mean : float
        Mean of the gamma distribution.
    std : float
        Standard deviation of the gamma distribution.

    See Also
    --------
    mean_std_to_alpha_beta : Convert mean and std to shape and scale parameters
    parse_parameters : Parse and validate gamma distribution parameters

    Examples
    --------
    >>> from gwtransport.gamma import alpha_beta_to_mean_std
    >>> alpha = 13.72  # shape parameter
    >>> beta = 2187.0  # scale parameter
    >>> mean, std = alpha_beta_to_mean_std(alpha=alpha, beta=beta)
    >>> print(f"Mean pore volume: {mean:.0f} m³")  # doctest: +ELLIPSIS
    Mean pore volume: 3000... m³
    >>> print(f"Std pore volume: {std:.0f} m³")  # doctest: +ELLIPSIS
    Std pore volume: 810... m³
    """
    mean = alpha * beta
    std = np.sqrt(alpha) * beta
    return mean, std


def bins(
    *,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    quantile_edges: np.ndarray | None = None,
) -> dict[str, npt.NDArray[np.floating]]:
    """
    Divide gamma distribution into bins and compute various bin properties.

    If n_bins is provided, the gamma distribution is divided into n_bins equal-mass bins.
    If quantile_edges is provided, the gamma distribution is divided into bins defined by
    the quantile edges. The quantile edges must be in the range [0, 1] and of size n_bins + 1.
    The first and last quantile edges must be 0 and 1, respectively.

    Parameters
    ----------
    alpha : float, optional
        Shape parameter of gamma distribution (must be > 0)
    beta : float, optional
        Scale parameter of gamma distribution (must be > 0)
    mean : float, optional
        Mean of the gamma distribution.
    std : float, optional
        Standard deviation of the gamma distribution.
    n_bins : int, optional
        Number of bins to divide the gamma distribution (must be > 1). Default is 100.
    quantile_edges : array-like, optional
        Quantile edges for binning. Must be in the range [0, 1] and of size n_bins + 1.
        The first and last quantile edges must be 0 and 1, respectively.
        If provided, n_bins is ignored.

    Returns
    -------
    dict
        Dictionary with keys of type str and values of type numpy.ndarray:

        - ``lower_bound``: lower bounds of bins (first one is 0)
        - ``upper_bound``: upper bounds of bins (last one is inf)
        - ``edges``: bin edges (lower_bound[0], upper_bound[0], ..., upper_bound[-1])
        - ``expected_values``: expected values in bins. Is what you would expect to observe if you repeatedly sampled from the probability distribution, but only considered samples that fall within that particular bin
        - ``probability_mass``: probability mass in bins

    See Also
    --------
    bin_masses : Calculate probability mass for bins
    mean_std_to_alpha_beta : Convert mean/std to alpha/beta parameters

    Examples
    --------
    Create equal-mass bins for a gamma distribution:

    >>> from gwtransport.gamma import bins
    >>> # Define gamma distribution using mean and std
    >>> result = bins(mean=30000.0, std=8100.0, n_bins=5)

    Create bins with custom quantile edges:

    >>> import numpy as np
    >>> quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> result = bins(mean=30000.0, std=8100.0, quantile_edges=quantiles)
    >>> print(f"Number of bins: {len(result['probability_mass'])}")
    Number of bins: 4
    """
    alpha, beta = parse_parameters(alpha=alpha, beta=beta, mean=mean, std=std)

    # Calculate boundaries for equal mass bins
    # If quantile_edges is provided, use it (n_bins is ignored)
    # Otherwise, use n_bins (which defaults to 100)
    if quantile_edges is not None:
        n_bins = len(quantile_edges) - 1
    else:
        quantile_edges = np.linspace(0, 1, n_bins + 1)  # includes 0 and 1

    if n_bins <= 1:
        msg = "Number of bins must be greater than 1"
        raise ValueError(msg)

    bin_edges = gamma_dist.ppf(quantile_edges, alpha, scale=beta)
    probability_mass = np.diff(quantile_edges)  # probability mass for each bin

    # Calculate expected value for each bin
    diff_alpha_plus_1 = bin_masses(alpha=alpha + 1, beta=beta, bin_edges=bin_edges)
    expected_values = beta * alpha * diff_alpha_plus_1 / probability_mass

    return {
        "lower_bound": bin_edges[:-1],
        "upper_bound": bin_edges[1:],
        "edges": bin_edges,
        "expected_values": expected_values,
        "probability_mass": probability_mass,
    }


def bin_masses(*, alpha: float, beta: float, bin_edges: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """
    Calculate probability mass for each bin in gamma distribution.

    Is the area under the gamma distribution PDF between the bin edges.

    Parameters
    ----------
    alpha : float
        Shape parameter of gamma distribution (must be > 0)
    beta : float
        Scale parameter of gamma distribution (must be > 0)
    bin_edges : array-like
        Bin edges. Array of increasing values of size len(bins) + 1.
        Must be > 0.

    Returns
    -------
    numpy.ndarray
        Probability mass for each bin
    """
    # Convert inputs to numpy arrays
    bin_edges = np.asarray(bin_edges)

    # Validate inputs
    if alpha <= 0 or beta <= 0:
        msg = "Alpha and beta must be positive"
        raise ValueError(msg)
    if len(bin_edges) < 2:  # noqa: PLR2004
        msg = "Bin edges must contain at least two values"
        raise ValueError(msg)
    if np.any(np.diff(bin_edges) < 0):
        msg = "Bin edges must be increasing"
        raise ValueError(msg)
    if np.any(bin_edges < 0):
        msg = "Bin edges must be positive"
        raise ValueError(msg)
    val = gammainc(alpha, bin_edges / beta)
    return val[1:] - val[:-1]
