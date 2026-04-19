"""
Gamma Distribution Utilities for Aquifer Pore Volume Heterogeneity.

This module provides utilities for working with gamma distributions to model heterogeneous
aquifer pore volumes in groundwater transport analysis. The gamma distribution offers a
flexible three-parameter model (shape, scale, location) for representing the natural
variability in flow path lengths and residence times within aquifer systems. In
heterogeneous aquifers, water travels through multiple flow paths with different pore
volumes; the location parameter additionally represents a guaranteed minimum pore volume
(for example, immobile porosity or a geometric minimum travel distance).

Parameterizations
-----------------
Two equivalent parameterizations are supported, each optionally with a location shift:

- **(mean, std, loc)** — physically intuitive. ``mean`` is the total expected value,
  ``std`` is the spread (invariant under shift), and ``loc`` is the lower bound of
  support. Constraint: ``0 <= loc < mean``.
- **(alpha, beta, loc)** — scipy-style. ``alpha`` is shape, ``beta`` is scale, and
  ``loc`` is the lower bound of support. Constraint: ``alpha > 0``, ``beta > 0``,
  ``loc >= 0``.

Conversion formulas (with constraint ``mean > loc``):

    alpha = ((mean - loc) / std) ** 2
    beta  = std ** 2 / (mean - loc)
    mean  = alpha * beta + loc
    std   = sqrt(alpha) * beta

When ``loc == 0`` the three-parameter model reduces to the standard two-parameter
gamma distribution.

Available functions:

- :func:`parse_parameters` - Parse and validate gamma distribution parameters from either
  (mean, std, loc) or (alpha, beta, loc). Ensures exactly one parameter pair is provided
  and validates positivity and ordering constraints.

- :func:`mean_std_loc_to_alpha_beta` - Convert physically intuitive (mean, std, loc) parameters
  to gamma shape/scale parameters.

- :func:`alpha_beta_loc_to_mean_std` - Convert gamma (alpha, beta, loc) parameters back to
  (mean, std) for physical interpretation.

- :func:`bins` - Primary function for transport modeling. Creates discrete probability bins
  from the (optionally shifted) gamma distribution with equal-probability bins (default) or
  custom quantile edges. Returns bin edges, expected values (mean pore volume within each
  bin), and probability masses (weight in transport calculations).

- :func:`bin_masses` - Calculate probability mass for custom bin edges using the incomplete
  gamma function. Lower-level function used internally by bins().

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import gammainc
from scipy.stats import gamma as gamma_dist


def parse_parameters(
    *,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
) -> tuple[float, float, float]:
    """
    Parse parameters for gamma distribution.

    Either ``(mean, std)`` or ``(alpha, beta)`` must be provided. ``loc`` is optional
    and defaults to 0, which recovers the standard two-parameter gamma distribution.

    Parameters
    ----------
    mean : float, optional
        Mean of the gamma distribution. Must be strictly greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma distribution. Must be positive. See
        :ref:`concept-dispersion-scales` for what std represents depending
        on APVD source. ``std`` is invariant under the ``loc`` shift.
    loc : float, optional
        Location (horizontal shift) of the gamma distribution; the lower bound of
        support. Must satisfy ``loc >= 0`` and, when ``mean`` is supplied,
        ``loc < mean``. Default is ``0.0``.
    alpha : float, optional
        Shape parameter of gamma distribution (must be > 0).
    beta : float, optional
        Scale parameter of gamma distribution (must be > 0).

    Returns
    -------
    alpha : float
        Shape parameter of gamma distribution.
    beta : float
        Scale parameter of gamma distribution.
    loc : float
        Location parameter of gamma distribution.

    Raises
    ------
    ValueError
        If neither ``(mean, std)`` nor ``(alpha, beta)`` is provided, if only one
        of a pair is provided, if ``alpha`` or ``beta`` are not positive, if
        ``loc`` is negative, or if ``mean <= loc``.
    """
    if loc < 0:
        msg = "loc must be non-negative"
        raise ValueError(msg)

    if (alpha is None) != (beta is None):
        msg = "alpha and beta must both be provided or both be None."
        raise ValueError(msg)

    if alpha is None or beta is None:
        if mean is None or std is None:
            msg = "Either (alpha, beta) or (mean, std) must be provided."
            raise ValueError(msg)

        alpha, beta = mean_std_loc_to_alpha_beta(mean=mean, std=std, loc=loc)

    if alpha <= 0 or beta <= 0:
        msg = "Alpha and beta must be positive"
        raise ValueError(msg)

    return alpha, beta, loc


def mean_std_loc_to_alpha_beta(*, mean: float, std: float, loc: float = 0.0) -> tuple[float, float]:
    """
    Convert mean, standard deviation, and location of gamma distribution to shape/scale.

    The two-parameter shape/scale representation (``alpha``, ``beta``) is derived from
    the excess-over-``loc`` moments: ``mean_excess = mean - loc``, ``std_excess = std``.

    Parameters
    ----------
    mean : float
        Mean of the gamma distribution. Must be strictly greater than ``loc``.
    std : float
        Standard deviation of the gamma distribution. Must be positive. See
        :ref:`concept-dispersion-scales` for what std represents depending
        on APVD source. ``std`` is invariant under the ``loc`` shift.
    loc : float, optional
        Location (horizontal shift) of the gamma distribution. Must satisfy
        ``0 <= loc < mean``. Default is ``0.0``.

    Returns
    -------
    alpha : float
        Shape parameter of gamma distribution.
    beta : float
        Scale parameter of gamma distribution.

    Raises
    ------
    ValueError
        If ``std`` is not positive, if ``loc`` is negative, or if ``mean <= loc``.

    See Also
    --------
    alpha_beta_loc_to_mean_std : Convert shape/scale/loc parameters to mean and std.
    parse_parameters : Parse and validate gamma distribution parameters.

    Examples
    --------
    >>> from gwtransport.gamma import mean_std_loc_to_alpha_beta
    >>> mean_pore_volume = 30000.0  # m³
    >>> std_pore_volume = 8100.0  # m³
    >>> alpha, beta = mean_std_loc_to_alpha_beta(
    ...     mean=mean_pore_volume, std=std_pore_volume
    ... )
    >>> print(f"Shape parameter (alpha): {alpha:.2f}")
    Shape parameter (alpha): 13.72
    >>> print(f"Scale parameter (beta): {beta:.2f}")
    Scale parameter (beta): 2187.00

    With a 5000 m³ minimum pore volume:

    >>> alpha, beta = mean_std_loc_to_alpha_beta(mean=30000.0, std=8100.0, loc=5000.0)
    >>> print(f"Shape parameter (alpha): {alpha:.2f}")
    Shape parameter (alpha): 9.45
    >>> print(f"Scale parameter (beta): {beta:.2f}")
    Scale parameter (beta): 2646.00
    """
    if std <= 0:
        msg = "std must be positive"
        raise ValueError(msg)
    if loc < 0:
        msg = "loc must be non-negative"
        raise ValueError(msg)
    if mean <= loc:
        msg = "mean must be strictly greater than loc"
        raise ValueError(msg)

    mean_excess = mean - loc
    alpha = mean_excess**2 / std**2
    beta = std**2 / mean_excess
    return alpha, beta


def alpha_beta_loc_to_mean_std(*, alpha: float, beta: float, loc: float = 0.0) -> tuple[float, float]:
    """
    Convert shape, scale, and location of gamma distribution to mean and standard deviation.

    Parameters
    ----------
    alpha : float
        Shape parameter of the gamma distribution. Must be positive.
    beta : float
        Scale parameter of the gamma distribution. Must be positive.
    loc : float, optional
        Location (horizontal shift) of the gamma distribution. Must be non-negative.
        Default is ``0.0``.

    Returns
    -------
    mean : float
        Mean of the gamma distribution, equal to ``alpha * beta + loc``.
    std : float
        Standard deviation of the gamma distribution, equal to ``sqrt(alpha) * beta``.
        ``std`` is invariant under the ``loc`` shift.

    Raises
    ------
    ValueError
        If ``loc`` is negative.

    See Also
    --------
    mean_std_loc_to_alpha_beta : Convert mean/std/loc to shape and scale parameters.
    parse_parameters : Parse and validate gamma distribution parameters.

    Examples
    --------
    >>> from gwtransport.gamma import alpha_beta_loc_to_mean_std
    >>> alpha = 13.72  # shape parameter
    >>> beta = 2187.0  # scale parameter
    >>> mean, std = alpha_beta_loc_to_mean_std(alpha=alpha, beta=beta)
    >>> print(f"Mean pore volume: {mean:.0f} m³")  # doctest: +ELLIPSIS
    Mean pore volume: 3000... m³
    >>> print(f"Std pore volume: {std:.0f} m³")  # doctest: +ELLIPSIS
    Std pore volume: 810... m³
    """
    if loc < 0:
        msg = "loc must be non-negative"
        raise ValueError(msg)
    mean = alpha * beta + loc
    std = np.sqrt(alpha) * beta
    return mean, std


def bins(
    *,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    quantile_edges: np.ndarray | None = None,
) -> dict[str, npt.NDArray[np.floating]]:
    """
    Divide a (shifted) gamma distribution into bins and compute bin properties.

    If ``n_bins`` is provided, the gamma distribution is divided into ``n_bins``
    equal-mass bins. If ``quantile_edges`` is provided, the distribution is divided
    into bins defined by those quantile edges. The quantile edges must lie in
    ``[0, 1]`` with size ``n_bins + 1``; the first and last entries must be 0 and 1.

    Parameters
    ----------
    mean : float, optional
        Mean of the gamma distribution. Must be strictly greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma distribution. Must be positive.
    loc : float, optional
        Location (horizontal shift) of the gamma distribution; the lower bound of
        support. Must satisfy ``0 <= loc < mean`` (or ``loc >= 0`` when using
        alpha/beta). Default is ``0.0``.
    alpha : float, optional
        Shape parameter of gamma distribution (must be > 0).
    beta : float, optional
        Scale parameter of gamma distribution (must be > 0).
    n_bins : int, optional
        Number of bins to divide the gamma distribution (must be > 1). Default is 100.
    quantile_edges : array-like, optional
        Quantile edges for binning. Must be in the range [0, 1] and of size
        ``n_bins + 1``. The first and last quantile edges must be 0 and 1, respectively.
        If provided, ``n_bins`` is ignored.

    Returns
    -------
    dict
        Dictionary with keys of type str and values of type numpy.ndarray:

        - ``lower_bound``: lower bounds of bins (first one equals ``loc``)
        - ``upper_bound``: upper bounds of bins (last one is inf)
        - ``edges``: bin edges (lower_bound[0], upper_bound[0], ..., upper_bound[-1])
        - ``expected_values``: expected values in bins. Is what you would expect to
          observe if you repeatedly sampled from the probability distribution, but only
          considered samples that fall within that particular bin.
        - ``probability_mass``: probability mass in bins (invariant under ``loc`` shift).

    Raises
    ------
    ValueError
        If ``n_bins`` is not greater than 1, or if parameter validation in
        :func:`parse_parameters` fails.

    See Also
    --------
    bin_masses : Calculate probability mass for bins.
    mean_std_loc_to_alpha_beta : Convert mean/std/loc to alpha/beta parameters.
    gwtransport.advection.gamma_infiltration_to_extraction : Use bins for transport modeling.
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model.
    :ref:`concept-gamma-loc` : Shifted gamma with minimum pore volume.
    :ref:`concept-dispersion-scales` : What ``std`` represents (macrodispersion vs total spreading).
    :ref:`assumption-gamma-distribution` : When gamma distribution is adequate.

    Examples
    --------
    Create equal-mass bins for a gamma distribution:

    >>> from gwtransport.gamma import bins
    >>> result = bins(mean=30000.0, std=8100.0, n_bins=5)

    With a location parameter representing a minimum pore volume:

    >>> result = bins(mean=30000.0, std=8100.0, loc=5000.0, n_bins=5)
    >>> float(result["edges"][0])
    5000.0

    Create bins with custom quantile edges:

    >>> import numpy as np
    >>> quantiles = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> result = bins(mean=30000.0, std=8100.0, quantile_edges=quantiles)
    >>> print(f"Number of bins: {len(result['probability_mass'])}")
    Number of bins: 4
    """
    alpha, beta, loc = parse_parameters(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta)

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

    # Unshifted bin edges for the standard Gamma(alpha, beta) distribution, then shift
    unshifted_edges = gamma_dist.ppf(quantile_edges, alpha, scale=beta)
    bin_edges = unshifted_edges + loc
    probability_mass = np.diff(quantile_edges)  # probability mass for each bin

    # Conditional mean within each bin for the unshifted distribution, then shift by loc.
    # E[X | a <= X < b] for X ~ Gamma(alpha, beta) uses the identity
    #     E[X * 1_{a<=X<b}] = alpha * beta * (F_{alpha+1}(b/beta) - F_{alpha+1}(a/beta))
    # where F_{alpha+1} is the CDF of Gamma(alpha+1, beta). The bin_masses helper returns
    # exactly those differences.
    diff_alpha_plus_1 = bin_masses(alpha=alpha + 1, beta=beta, bin_edges=unshifted_edges)
    expected_values = beta * alpha * diff_alpha_plus_1 / probability_mass + loc

    return {
        "lower_bound": bin_edges[:-1],
        "upper_bound": bin_edges[1:],
        "edges": bin_edges,
        "expected_values": expected_values,
        "probability_mass": probability_mass,
    }


def bin_masses(*, alpha: float, beta: float, bin_edges: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """
    Calculate probability mass for each bin in a standard (unshifted) gamma distribution.

    Is the area under the Gamma(alpha, beta) PDF between the bin edges. This lower-level
    function operates on the unshifted gamma distribution; if a location shift is needed,
    callers should subtract ``loc`` from their physical bin edges before passing them in.
    Because probability mass is invariant under a location shift, the result is identical
    to that of the shifted distribution.

    Parameters
    ----------
    alpha : float
        Shape parameter of gamma distribution (must be > 0).
    beta : float
        Scale parameter of gamma distribution (must be > 0).
    bin_edges : array-like
        Bin edges of the unshifted distribution. Array of increasing values of size
        ``len(bins) + 1``. Must be non-negative.

    Returns
    -------
    numpy.ndarray
        Probability mass for each bin.

    Raises
    ------
    ValueError
        If ``alpha`` or ``beta`` are not positive, if ``bin_edges`` contains fewer than two
        values, if ``bin_edges`` are not monotonically increasing, or if any ``bin_edges``
        are negative.
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
        msg = "Bin edges must be non-negative"
        raise ValueError(msg)
    val = gammainc(alpha, bin_edges / beta)
    return val[1:] - val[:-1]
