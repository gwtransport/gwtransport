"""
Log Removal Efficiency Calculations for Water Treatment Systems.

This module provides utilities to calculate log removal values for different configurations
of water treatment systems, including basic log removal calculations and parallel flow
arrangements where multiple treatment processes operate simultaneously on different fractions
of the total flow. Log removal is a standard measure in water treatment that represents the
reduction of pathogen concentration on a logarithmic scale (e.g., log removal of 3 represents
99.9% reduction). For systems in series, log removals are typically summed directly, while
for parallel systems, a weighted average based on flow distribution is required.

The log removal model uses standard first-order decay:

    Log Removal = log10_removal_rate * residence_time

where ``log10_removal_rate`` has units [log10/day] and ``residence_time`` has units [days].
This is equivalent to exponential decay ``C_out/C_in = 10^(-mu * t)``, where mu is the
log10 removal rate and t is residence time. The first-order decay rate constant lambda [1/day]
is related to mu by ``lambda = mu * ln(10)``.

Available functions:

- :func:`residence_time_to_log_removal` - Calculate log removal from residence times and
  removal rate coefficient. Uses formula: Log Removal = log10_removal_rate * residence_time.
  Handles single values, 1D arrays, or multi-dimensional arrays of residence times. Returns
  log removal values with same shape as input.

- :func:`decay_rate_to_log10_removal_rate` - Convert a natural-log decay rate constant
  lambda [1/day] to a log10 removal rate mu [log10/day].

- :func:`log10_removal_rate_to_decay_rate` - Convert a log10 removal rate mu [log10/day]
  to a natural-log decay rate constant lambda [1/day].

- :func:`parallel_mean` - Calculate weighted average log removal for parallel flow systems.
  Computes overall efficiency when multiple treatment paths operate in parallel with different
  log removal values and flow fractions. Uses formula: Total Log Removal = -log10(sum(F_i * 10^(-LR_i)))
  where F_i is flow fraction and LR_i is log removal for path i. Supports multi-dimensional
  arrays via axis parameter for batch processing. Assumes equal flow distribution if flow_fractions
  not provided.

- :func:`gamma_pdf` - Compute probability density function (PDF) of log removal given
  gamma-distributed residence time. Since R = mu*T and T ~ Gamma(alpha, beta), R follows a
  Gamma(alpha, mu*beta) distribution.

- :func:`gamma_cdf` - Compute cumulative distribution function (CDF) of log removal given
  gamma-distributed residence time. Returns probability that log removal is less than or equal
  to specified values.

- :func:`gamma_mean` - Compute mean log removal for gamma-distributed residence time. Returns
  expected value E[R] = mu * alpha * beta, where mu is the log10 removal rate.

- :func:`gamma_find_flow_for_target_mean` - Find flow rate that produces specified target mean
  log removal given gamma-distributed aquifer pore volume. Solves inverse problem analytically:
  flow = mu * alpha * apv_beta / target_mean.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy import stats


def residence_time_to_log_removal(
    *, residence_times: npt.ArrayLike, log10_removal_rate: float
) -> npt.NDArray[np.floating]:
    """
    Compute log removal given residence times and a log10 removal rate.

    This function calculates the log removal efficiency based on the
    residence times of water in a treatment system and the log10 removal
    rate coefficient using first-order decay:

    Log Removal = log10_removal_rate * residence_time

    This corresponds to exponential decay of pathogen concentration:
    C_out/C_in = 10^(-log10_removal_rate * residence_time).

    Parameters
    ----------
    residence_times : array-like
        Array of residence times in days. Must be positive values.
    log10_removal_rate : float
        Log10 removal rate coefficient (log10/day). Relates residence time
        to log removal efficiency via first-order decay.

    Returns
    -------
    log_removals : numpy.ndarray
        Array of log removal values corresponding to the input residence times.
        Same shape as input residence_times.

    See Also
    --------
    decay_rate_to_log10_removal_rate : Convert natural-log decay rate to log10 removal rate
    log10_removal_rate_to_decay_rate : Convert log10 removal rate to natural-log decay rate
    gamma_mean : Compute mean log removal for gamma-distributed residence times
    gamma_find_flow_for_target_mean : Find flow rate to achieve target log removal
    parallel_mean : Calculate weighted average for parallel flow systems
    gwtransport.residence_time.residence_time : Compute residence times from flow and pore volume
    :ref:`concept-residence-time` : Time in aquifer determines pathogen contact time

    Notes
    -----
    Log removal is a logarithmic measure of pathogen reduction:
    - Log 1 = 90% reduction
    - Log 2 = 99% reduction
    - Log 3 = 99.9% reduction

    The first-order decay model is mathematically identical to radioactive
    decay used in tracer dating. To convert a published natural-log decay
    rate lambda [1/day] to log10_removal_rate mu [log10/day], use
    :func:`decay_rate_to_log10_removal_rate`.

    Examples
    --------
    >>> import numpy as np
    >>> from gwtransport.logremoval import residence_time_to_log_removal
    >>> residence_times = np.array([10.0, 20.0, 50.0])
    >>> log10_removal_rate = 0.2
    >>> residence_time_to_log_removal(
    ...     residence_times=residence_times, log10_removal_rate=log10_removal_rate
    ... )  # doctest: +NORMALIZE_WHITESPACE
    array([ 2.,  4., 10.])

    >>> # Single residence time
    >>> residence_time_to_log_removal(residence_times=5.0, log10_removal_rate=0.3)
    np.float64(1.5)

    >>> # 2D array of residence times
    >>> residence_times_2d = np.array([[10.0, 20.0], [30.0, 40.0]])
    >>> residence_time_to_log_removal(
    ...     residence_times=residence_times_2d, log10_removal_rate=0.1
    ... )
    array([[1., 2.],
           [3., 4.]])
    """
    return log10_removal_rate * np.asarray(residence_times, dtype=float)


def decay_rate_to_log10_removal_rate(decay_rate: float) -> float:
    """
    Convert a natural-log decay rate constant to a log10 removal rate.

    Converts lambda [1/day] to mu [log10/day] using the relationship
    mu = lambda / ln(10).

    Parameters
    ----------
    decay_rate : float
        Natural-log first-order decay rate constant lambda (1/day).
        For example, from tracer dating: lambda = ln(2) / half_life.

    Returns
    -------
    log10_removal_rate : float
        Log10 removal rate mu (log10/day).

    See Also
    --------
    log10_removal_rate_to_decay_rate : Inverse conversion
    residence_time_to_log_removal : Apply the log10 removal rate

    Examples
    --------
    >>> from gwtransport.logremoval import decay_rate_to_log10_removal_rate
    >>> import numpy as np
    >>> # Convert a decay rate of ln(2)/30 (half-life of 30 days)
    >>> decay_rate = np.log(2) / 30
    >>> decay_rate_to_log10_removal_rate(decay_rate)  # doctest: +SKIP
    0.01003...
    """
    return decay_rate / np.log(10)


def log10_removal_rate_to_decay_rate(log10_removal_rate: float) -> float:
    """
    Convert a log10 removal rate to a natural-log decay rate constant.

    Converts mu [log10/day] to lambda [1/day] using the relationship
    lambda = mu * ln(10).

    Parameters
    ----------
    log10_removal_rate : float
        Log10 removal rate mu (log10/day).

    Returns
    -------
    decay_rate : float
        Natural-log first-order decay rate constant lambda (1/day).

    See Also
    --------
    decay_rate_to_log10_removal_rate : Inverse conversion

    Examples
    --------
    >>> from gwtransport.logremoval import log10_removal_rate_to_decay_rate
    >>> log10_removal_rate_to_decay_rate(0.2)  # doctest: +SKIP
    0.4605...
    """
    return log10_removal_rate * np.log(10)


def parallel_mean(
    *, log_removals: npt.ArrayLike, flow_fractions: npt.ArrayLike | None = None, axis: int | None = None
) -> npt.NDArray[np.floating]:
    """
    Calculate the weighted average log removal for a system with parallel flows.

    This function computes the overall log removal efficiency of a parallel
    filtration system. If flow_fractions is not provided, it assumes equal
    distribution of flow across all paths.

    The calculation uses the formula:

    Total Log Removal = -log10(sum(F_i * 10^(-LR_i)))

    Where:
    - F_i = fraction of flow through system i (decimal, sum to 1.0)
    - LR_i = log removal of system i

    Parameters
    ----------
    log_removals : array-like
        Array of log removal values for each parallel flow.
        Each value represents the log10 reduction of pathogens.
        For multi-dimensional arrays, the parallel mean is computed along
        the specified axis.

    flow_fractions : array-like, optional
        Array of flow fractions for each parallel flow.
        Must sum to 1.0 along the specified axis and have compatible shape
        with log_removals. If None, equal flow distribution is assumed
        (default is None).

    axis : int, optional
        Axis along which to compute the parallel mean for multi-dimensional
        arrays. If None, the array is treated as 1-dimensional
        (default is None).

    Returns
    -------
    float or array-like
        The combined log removal value for the parallel system.
        If log_removals is multi-dimensional and axis is specified,
        returns an array with the specified axis removed.

    Notes
    -----
    This function performs minimal input validation to reduce complexity.
    NumPy will handle most error cases naturally through broadcasting
    and array operations.

    Notes
    -----
    Log removal is a logarithmic measure of pathogen reduction:
    - Log 1 = 90% reduction
    - Log 2 = 99% reduction
    - Log 3 = 99.9% reduction

    For parallel flows, the combined removal is typically less effective
    than the best individual removal but better than the worst.

    Examples
    --------
    >>> import numpy as np
    >>> from gwtransport.logremoval import parallel_mean
    >>> # Three parallel streams with equal flow and log removals of 3, 4, and 5
    >>> log_removals = np.array([3, 4, 5])
    >>> parallel_mean(log_removals=log_removals)
    np.float64(3.431798275933005)

    >>> # Two parallel streams with weighted flow
    >>> log_removals = np.array([3, 5])
    >>> flow_fractions = np.array([0.7, 0.3])
    >>> parallel_mean(log_removals=log_removals, flow_fractions=flow_fractions)
    np.float64(3.153044674980176)

    >>> # Multi-dimensional array: parallel mean along axis 1
    >>> log_removals_2d = np.array([[3, 4, 5], [2, 3, 4]])
    >>> parallel_mean(log_removals=log_removals_2d, axis=1)
    array([3.43179828, 2.43179828])

    See Also
    --------
    residence_time_to_log_removal : Compute log removal from residence times

    Notes
    -----
    For systems in series, log removals would be summed directly.
    """
    # Convert log_removals to numpy array if it isn't already
    log_removals = np.asarray(log_removals, dtype=float)

    # If flow_fractions is not provided, assume equal distribution
    if flow_fractions is None:
        if axis is None:
            # 1D case: calculate the number of parallel flows
            n = len(log_removals)
            # Create equal flow fractions (avoid division by zero)
            flow_fractions = np.full(n, 1.0 / n) if n > 0 else np.array([])
        else:
            # Multi-dimensional case: create equal flow fractions along the specified axis
            n = log_removals.shape[axis]
            shape = [1] * log_removals.ndim
            shape[axis] = n
            flow_fractions = np.full(shape, 1.0 / n)
    else:
        # Convert flow_fractions to numpy array
        flow_fractions = np.asarray(flow_fractions, dtype=float)

        # Note: Shape compatibility and sum validation removed to reduce complexity
        # NumPy will handle incompatible shapes through broadcasting or errors

    # Convert log removal to decimal reduction values
    decimal_reductions = 10 ** (-log_removals)

    # Calculate weighted average decimal reduction
    weighted_decimal_reduction = np.sum(flow_fractions * decimal_reductions, axis=axis)

    # Convert back to log scale
    return -np.log10(weighted_decimal_reduction)


def gamma_pdf(
    *, r: npt.ArrayLike, rt_alpha: float, rt_beta: float, log10_removal_rate: float
) -> npt.NDArray[np.floating]:
    """
    Compute the PDF of log removal given gamma-distributed residence time.

    Since log removal R = mu*T and T ~ Gamma(alpha, beta), the log removal R
    follows a Gamma(alpha, mu*beta) distribution.

    Parameters
    ----------
    r : array-like
        Log removal values at which to compute the PDF.
    rt_alpha : float
        Shape parameter of the gamma distribution for residence time.
    rt_beta : float
        Scale parameter of the gamma distribution for residence time (days).
    log10_removal_rate : float
        Log10 removal rate mu (log10/day). Relates residence time to
        log removal via R = mu * T.

    Returns
    -------
    pdf : numpy.ndarray
        PDF values corresponding to the input r values.

    See Also
    --------
    gamma_cdf : Cumulative distribution function of log removal
    gamma_mean : Mean of the log removal distribution
    """
    r = np.asarray(r)
    return stats.gamma.pdf(r, a=rt_alpha, scale=log10_removal_rate * rt_beta)


def gamma_cdf(
    *, r: npt.ArrayLike, rt_alpha: float, rt_beta: float, log10_removal_rate: float
) -> npt.NDArray[np.floating]:
    """
    Compute the CDF of log removal given gamma-distributed residence time.

    Since log removal R = mu*T and T ~ Gamma(alpha, beta), the CDF is
    P(R <= r) = P(T <= r/mu) = Gamma_CDF(r/mu; alpha, beta).

    Parameters
    ----------
    r : array-like
        Log removal values at which to compute the CDF.
    rt_alpha : float
        Shape parameter of the gamma distribution for residence time.
    rt_beta : float
        Scale parameter of the gamma distribution for residence time (days).
    log10_removal_rate : float
        Log10 removal rate mu (log10/day). Relates residence time to
        log removal via R = mu * T.

    Returns
    -------
    cdf : numpy.ndarray
        CDF values corresponding to the input r values.

    See Also
    --------
    gamma_pdf : Probability density function of log removal
    gamma_mean : Mean of the log removal distribution
    """
    r = np.asarray(r)
    return stats.gamma.cdf(r, a=rt_alpha, scale=log10_removal_rate * rt_beta)


def gamma_mean(*, rt_alpha: float, rt_beta: float, log10_removal_rate: float) -> float:
    """
    Compute the mean log removal for gamma-distributed residence time.

    Since R = mu*T and T ~ Gamma(alpha, beta), the mean is
    E[R] = mu * alpha * beta.

    Parameters
    ----------
    rt_alpha : float
        Shape parameter of the gamma distribution for residence time.
    rt_beta : float
        Scale parameter of the gamma distribution for residence time (days).
    log10_removal_rate : float
        Log10 removal rate mu (log10/day).

    Returns
    -------
    mean : float
        Mean value of the log removal distribution.

    See Also
    --------
    gamma_find_flow_for_target_mean : Find flow for target mean log removal
    gamma_pdf : PDF of the log removal distribution
    gamma_cdf : CDF of the log removal distribution
    """
    return log10_removal_rate * rt_alpha * rt_beta


def gamma_find_flow_for_target_mean(
    *, target_mean: float, apv_alpha: float, apv_beta: float, log10_removal_rate: float
) -> float:
    """
    Find the flow rate that produces a target mean log removal.

    Given a gamma-distributed aquifer pore volume with parameters (apv_alpha, apv_beta),
    the residence time distribution is Gamma(apv_alpha, apv_beta/flow). The mean log
    removal is mu * apv_alpha * apv_beta / flow. Solving for flow:

    flow = mu * apv_alpha * apv_beta / target_mean

    Rearranging the mean formula to solve for Q:
    target_mean = (log_removal_rate/ln(10)) * (digamma(alpha) + ln(beta) - ln(Q))
    ln(Q) = digamma(alpha) + ln(beta) - (ln(10)*target_mean/log_removal_rate)
    Q = beta * exp(-(ln(10)*target_mean/log_removal_rate - digamma(alpha)))

    Parameters
    ----------
    target_mean : float
        Target mean log removal value.
    apv_alpha : float
        Shape parameter of the gamma distribution for aquifer pore volume.
    apv_beta : float
        Scale parameter of the gamma distribution for aquifer pore volume.
    log10_removal_rate : float
        Log10 removal rate mu (log10/day).

    Returns
    -------
    flow : float
        Flow rate (same units as apv_beta per day) that produces the
        target mean log removal.

    See Also
    --------
    gamma_mean : Compute mean log removal for given parameters
    """
    return log10_removal_rate * apv_alpha * apv_beta / target_mean
