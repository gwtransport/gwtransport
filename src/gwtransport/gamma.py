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
  (mean, std, loc) or (alpha, beta, loc). Requires exactly one parameter pair and raises
  ``ValueError`` if both are supplied; validates positivity and ordering constraints.

- :func:`mean_std_loc_to_alpha_beta` - Convert physically intuitive (mean, std, loc) parameters
  to gamma shape/scale parameters.

- :func:`alpha_beta_loc_to_mean_std` - Convert gamma (alpha, beta, loc) parameters back to
  (mean, std) for physical interpretation.

- :func:`bins` - Primary function for transport modeling. Creates discrete probability bins
  from the (optionally shifted) gamma distribution with equal-probability bins (default) or
  custom quantile edges. Returns bin edges, expected values (mean pore volume within each
  bin), and probability masses (weight in transport calculations).

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.stats import gamma as gamma_dist

# Numerical-envelope guard thresholds for bins() (issue #331). Below _MIN_QUANTILE_GAP the
# conditional-mean CDF difference cancels catastrophically; when alpha*eps exceeds
# _HUGE_ALPHA_GAP_FRACTION of the smallest quantile gap, alpha + 1 == alpha to machine precision
# and the per-bin expected values degrade to noise.
_MIN_QUANTILE_GAP = 1e-8
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


def alpha_beta_loc_to_mean_std(*, alpha: float, beta: float, loc: float = 0.0) -> tuple[float, float]:
    """
    Convert shape, scale, and location of gamma distribution to mean and standard deviation.

    Parameters are validated via :func:`parse_parameters`, which raises ``ValueError`` if
    ``alpha`` or ``beta`` are non-positive or ``loc`` is negative.

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
    >>> print(f"Mean pore volume: {mean:.0f} m³")
    Mean pore volume: 30006 m³
    >>> print(f"Std pore volume: {std:.0f} m³")
    Std pore volume: 8101 m³
    """
    parse_parameters(alpha=alpha, beta=beta, loc=loc)
    return alpha * beta + loc, np.sqrt(alpha) * beta


def bins(
    *,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    quantile_edges: npt.ArrayLike | None = None,
) -> dict[str, npt.NDArray[np.floating]]:
    """
    Divide a (shifted) gamma distribution into bins and compute bin properties.

    If ``n_bins`` is provided, the gamma distribution is divided into ``n_bins``
    equal-mass bins. If ``quantile_edges`` is provided, the distribution is divided
    into bins defined by those quantile edges. The quantile edges must be a strictly
    increasing 1-D array of at least 3 entries (>= 2 bins) in ``[0, 1]``, with the
    first and last entries exactly 0 and 1; ``n_bins`` is then ignored.

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
    quantile_edges : array-like, optional
        Quantile edges for binning. Must be a strictly increasing 1-D array of at least
        3 entries (>= 2 bins), all in ``[0, 1]``, with the first and last entries exactly
        0 and 1. If provided, ``n_bins`` is ignored.

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
        If ``n_bins`` is not greater than 1, if ``quantile_edges`` is not a strictly
        increasing 1-D array in ``[0, 1]`` with endpoints exactly 0 and 1, or if
        parameter validation in :func:`parse_parameters` fails. Also raised for
        numerically-degenerate requests that would otherwise return silently-wrong
        structure: a smallest quantile-edge gap below ``1e-8`` (catastrophic
        cancellation of the conditional-mean CDF difference), an ``alpha`` so large that
        ``alpha + 1 == alpha`` in float64 relative to that gap (the distribution is
        numerically a point mass), or a bin whose expected value underflows to ``loc``.

    See Also
    --------
    mean_std_loc_to_alpha_beta : Convert mean/std/loc to alpha/beta parameters.
    gwtransport.advection.gamma_infiltration_to_extraction : Use bins for transport modeling.
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model.
    :ref:`concept-gamma-loc` : Shifted gamma with minimum pore volume.
    :ref:`concept-dispersion-scales` : What ``std`` represents (macrodispersion vs total spreading).
    :ref:`assumption-gamma-distribution` : When gamma distribution is adequate.

    Notes
    -----
    For a very large ``alpha`` (``>= ~1e6``) combined with custom ``quantile_edges`` deep
    in the left tail, ``scipy``'s ``ppf`` loses relative accuracy; prefer equal-mass bins
    or a coarser grid in that regime.

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

    if quantile_edges is not None:
        quantile_edges = np.asarray(quantile_edges, dtype=float)
        if quantile_edges.ndim != 1:
            msg = "quantile_edges must be a 1-D array."
            raise ValueError(msg)
        if quantile_edges.size == 0:
            msg = "quantile_edges must not be empty."
            raise ValueError(msg)
        if not np.all(np.diff(quantile_edges) > 0):
            msg = "quantile_edges must be strictly increasing."
            raise ValueError(msg)
        if quantile_edges[0] != 0.0 or quantile_edges[-1] != 1.0:
            msg = "quantile_edges must start at 0 and end at 1."
            raise ValueError(msg)
        n_bins = len(quantile_edges) - 1
    else:
        if n_bins <= 1:
            # Validate before np.linspace: a negative n_bins would otherwise surface as
            # numpy's opaque "Number of samples ... must be non-negative" error.
            msg = "Number of bins must be greater than 1"
            raise ValueError(msg)
        quantile_edges = np.linspace(0, 1, n_bins + 1)

    if n_bins <= 1:
        msg = "Number of bins must be greater than 1"
        raise ValueError(msg)

    # Guard the two numerical cliffs of the closed-form conditional mean below. A quantile gap
    # narrower than ~1e-8 makes the CDF difference cancel catastrophically (expected values leave
    # their own bins); an alpha so large that alpha*eps exceeds ~1% of that gap makes alpha+1 == alpha
    # to machine precision, so the conditional means degrade to noise. Both are only reachable with
    # extreme custom quantile_edges or a near-degenerate (delta-like) distribution.
    min_quantile_gap = float(np.min(np.diff(quantile_edges)))
    if min_quantile_gap < _MIN_QUANTILE_GAP:
        msg = (
            f"The smallest quantile-edge gap ({min_quantile_gap:.3g}) is below {_MIN_QUANTILE_GAP:g}; the "
            "conditional-mean CDF difference cancels catastrophically and expected values may fall "
            "outside their own bins. Use wider quantile bins."
        )
        raise ValueError(msg)
    if alpha * np.finfo(float).eps > _HUGE_ALPHA_GAP_FRACTION * min_quantile_gap:
        msg = (
            f"alpha ({alpha:.3g}) is too large for float64 bin resolution: alpha*eps exceeds 1% of "
            f"the smallest quantile gap ({min_quantile_gap:.3g}), so alpha+1 == alpha to machine "
            "precision and the per-bin expected values are numerical noise. The distribution is "
            "effectively a point mass at alpha*beta + loc; use a larger std/(mean-loc) or fewer bins."
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
