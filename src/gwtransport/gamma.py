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

Streamtube discretization by :func:`bins` is always into bins of **equal probability mass**.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.stats import gamma as gamma_dist

# Numerical-envelope guard threshold for bins(): when alpha*eps exceeds this fraction of the
# equal-mass quantile gap 1/n_bins, alpha + 1 == alpha to machine precision and the per-bin
# expected values degrade to noise.
_HUGE_ALPHA_GAP_FRACTION = 0.01


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
        If neither ``(mean, std)`` nor ``(alpha, beta)`` is provided, if both pairs
        are provided, if only one of a pair is provided, if ``alpha`` or ``beta`` are
        not positive, if ``loc`` is negative, if the resolved ``alpha``, ``beta``, or
        ``loc`` is not finite, or if ``mean <= loc``.
    """
    if loc < 0:
        msg = "loc must be non-negative"
        raise ValueError(msg)

    if (alpha is None) != (beta is None):
        msg = "alpha and beta must both be provided or both be None."
        raise ValueError(msg)

    if alpha is not None and (mean is not None or std is not None):
        msg = "Provide either (alpha, beta) or (mean, std), not both."
        raise ValueError(msg)

    if (mean is None) != (std is None):
        msg = "mean and std must both be provided or both be None."
        raise ValueError(msg)

    # The ``or beta is None`` is redundant at runtime (the check above pairs them) but lets the
    # type checker narrow ``beta`` to a float on the fall-through return.
    if alpha is None or beta is None:
        if mean is None or std is None:
            msg = "Either (alpha, beta) or (mean, std) must be provided."
            raise ValueError(msg)
        # mean_std_loc_to_alpha_beta enforces std>0 and mean>loc, which together with
        # loc>=0 guarantee alpha=(mean-loc)**2/std**2 > 0 and beta=std**2/(mean-loc) > 0.
        alpha, beta = mean_std_loc_to_alpha_beta(mean=mean, std=std, loc=loc)
    elif alpha <= 0 or beta <= 0:
        msg = "Alpha and beta must be positive"
        raise ValueError(msg)

    # A non-finite alpha/beta/loc slips past the comparisons above (``nan <= 0`` and ``nan < 0`` are
    # both False), producing an all-NaN distribution instead of a clear error. Reject it loudly.
    if not (np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(loc)):
        msg = "alpha, beta, and loc must be finite."
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
    Shape parameter (alpha): 9.53
    >>> print(f"Scale parameter (beta): {beta:.2f}")
    Scale parameter (beta): 2624.40
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


def bins(
    *,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
) -> dict[str, npt.NDArray[np.floating]]:
    """
    Divide a (shifted) gamma distribution into equal-probability-mass bins and compute bin properties.

    The distribution is split at the ``n_bins + 1`` uniform quantile edges, so every bin
    (streamtube) carries probability mass ``1 / n_bins``.

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
        Number of bins to divide the gamma distribution (must be >= 2). Default is 100.

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
        :func:`parse_parameters` fails. Also raised for numerically-degenerate requests that
        would otherwise return silently-wrong structure: an ``alpha`` so large that
        ``alpha + 1 == alpha`` in float64 relative to the ``1 / n_bins`` quantile gap (the
        distribution is numerically a point mass), or a bin whose expected value underflows
        to ``loc``.

    See Also
    --------
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
    >>> print(f"Number of bins: {len(result['probability_mass'])}")
    Number of bins: 5

    With a location parameter representing a minimum pore volume:

    >>> result = bins(mean=30000.0, std=8100.0, loc=5000.0, n_bins=5)
    >>> float(result["edges"][0])
    5000.0
    """
    alpha, beta, loc = parse_parameters(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta)

    if n_bins <= 1:
        # Validate before np.linspace: a negative n_bins would otherwise surface as
        # numpy's opaque "Number of samples ... must be non-negative" error.
        msg = "Number of bins must be greater than 1"
        raise ValueError(msg)

    quantile_edges = np.linspace(0, 1, n_bins + 1)

    # Guard the numerical cliff of the closed-form conditional mean below: an alpha so large that
    # alpha*eps exceeds ~1% of the equal-mass quantile gap 1/n_bins makes alpha+1 == alpha to machine
    # precision, so the conditional means degrade to noise. Only reachable with a near-degenerate
    # (delta-like) distribution.
    if alpha * np.finfo(float).eps * n_bins > _HUGE_ALPHA_GAP_FRACTION:
        msg = (
            f"alpha ({alpha:.3g}) is too large for float64 bin resolution: alpha*eps exceeds 1% of "
            f"the equal-mass quantile gap (1/{n_bins}), so alpha+1 == alpha to machine precision and "
            "the per-bin expected values are numerical noise. The distribution is effectively a point "
            "mass at alpha*beta + loc; use a larger std/(mean-loc) or fewer bins."
        )
        raise ValueError(msg)

    # Unshifted bin edges for the standard Gamma(alpha, beta) distribution, then shift
    unshifted_edges = gamma_dist.ppf(quantile_edges, alpha, scale=beta)
    bin_edges = unshifted_edges + loc
    probability_mass = np.diff(quantile_edges)  # probability mass for each bin

    # Conditional mean within each bin for the unshifted distribution, then shift by loc.
    # E[X | a <= X < b] for X ~ Gamma(alpha, beta) uses the identity
    #     E[X * 1_{a<=X<b}] = alpha * beta * (F_{alpha+1}(b) - F_{alpha+1}(a))
    # where F_{alpha+1} is the CDF of Gamma(alpha+1, scale=beta) (equivalently the regularized
    # lower incomplete gamma P(alpha+1, b/beta) - P(alpha+1, a/beta)).
    cdf_alpha_plus_1 = gamma_dist.cdf(unshifted_edges, alpha + 1, scale=beta)
    diff_alpha_plus_1 = np.diff(cdf_alpha_plus_1)

    # Pre-shift conditional mean of the excess over loc. Every positive-mass bin of a Gamma(alpha, beta)
    # has a strictly positive conditional mean, so this must be > 0; a value <= 0 signals an underflow /
    # cancellation of the CDF difference (very small alpha or extremely fine bins) that would emit a
    # numerically-zero or negative pore volume. Test this pre-shift quantity rather than the shifted
    # expected value: for loc > 0 a benign ``loc + tiny_positive_excess == loc`` rounding would otherwise
    # be misread as underflow and reject correct, usable output for shifted heterogeneous APVDs.
    cond_mean_excess = beta * alpha * diff_alpha_plus_1 / probability_mass
    if np.any(cond_mean_excess <= 0.0):
        msg = (
            "A bin's conditional expected value underflowed to loc (its excess conditional mean over loc "
            "is not strictly positive). This happens for very small alpha or extremely fine bins where "
            "the CDF difference underflows. Use fewer bins or a larger alpha."
        )
        raise ValueError(msg)

    expected_values = cond_mean_excess + loc

    return {
        "lower_bound": bin_edges[:-1],
        "upper_bound": bin_edges[1:],
        "edges": bin_edges,
        "expected_values": expected_values,
        "probability_mass": probability_mass,
    }
