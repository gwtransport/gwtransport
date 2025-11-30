"""
Log Removal Efficiency Calculations for Water Treatment Systems.

This module provides utilities to calculate log removal values for different configurations
of water treatment systems, including basic log removal calculations and parallel flow
arrangements where multiple treatment processes operate simultaneously on different fractions
of the total flow. Log removal is a standard measure in water treatment that represents the
reduction of pathogen concentration on a logarithmic scale (e.g., log removal of 3 represents
99.9% reduction). For systems in series, log removals are typically summed directly, while
for parallel systems, a weighted average based on flow distribution is required.

Available functions:

- :func:`residence_time_to_log_removal` - Calculate log removal from residence times and
  removal rate coefficient. Uses formula: Log Removal = log_removal_rate * log10(residence_time).
  Handles single values, 1D arrays, or multi-dimensional arrays of residence times. Returns
  log removal values with same shape as input.

- :func:`parallel_mean` - Calculate weighted average log removal for parallel flow systems.
  Computes overall efficiency when multiple treatment paths operate in parallel with different
  log removal values and flow fractions. Uses formula: Total Log Removal = -log10(sum(F_i * 10^(-LR_i)))
  where F_i is flow fraction and LR_i is log removal for path i. Supports multi-dimensional
  arrays via axis parameter for batch processing. Assumes equal flow distribution if flow_fractions
  not provided.

- :func:`gamma_pdf` - Compute probability density function (PDF) of log removal given
  gamma-distributed residence time. Transforms gamma distribution gamma(rt_alpha, rt_beta) to
  log removal distribution using log_removal_rate coefficient. Useful for uncertainty quantification
  and probabilistic risk assessment.

- :func:`gamma_cdf` - Compute cumulative distribution function (CDF) of log removal given
  gamma-distributed residence time. Returns probability that log removal is less than or equal
  to specified values. Uses scipy.stats.gamma.cdf internally with transformed residence time
  values t = 10^(r/log_removal_rate).

- :func:`gamma_mean` - Compute mean log removal for gamma-distributed residence time. Returns
  expected value E[R] = (log_removal_rate/ln(10)) * (digamma(rt_alpha) + ln(rt_beta)). More
  efficient than numerical integration of PDF. Useful for design calculations and performance
  prediction.

- :func:`gamma_find_flow_for_target_mean` - Find flow rate that produces specified target mean
  log removal given gamma-distributed aquifer pore volume. Solves inverse problem analytically
  using relationship: flow = apv_beta * exp(digamma(apv_alpha) - (ln(10)*target_mean)/log_removal_rate).
  Useful for designing systems to meet regulatory requirements.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.special import digamma, gamma


def residence_time_to_log_removal(
    *, residence_times: npt.ArrayLike, log_removal_rate: float
) -> npt.NDArray[np.floating]:
    """
    Compute log removal given residence times and a log removal rate.

    This function calculates the log removal efficiency based on the
    residence times of water in a treatment system and the log removal
    rate coefficient.

    The calculation uses the formula:
    Log Removal = log_removal_rate * log10(residence_time)

    Parameters
    ----------
    residence_times : array-like
        Array of residence times (in consistent units, e.g., hours, days).
        Must be positive values.
    log_removal_rate : float
        Log removal rate coefficient that relates residence time to
        log removal efficiency. Units should be consistent with
        residence_times.

    Returns
    -------
    log_removals : numpy.ndarray
        Array of log removal values corresponding to the input residence times.
        Same shape as input residence_times.

    See Also
    --------
    gamma_mean : Compute mean log removal for gamma-distributed residence times
    gamma_find_flow_for_target_mean : Find flow rate to achieve target log removal
    parallel_mean : Calculate weighted average for parallel flow systems
    gwtransport.residence_time.residence_time : Compute residence times from flow and pore volume

    Notes
    -----
    Log removal is a logarithmic measure of pathogen reduction:
    - Log 1 = 90% reduction
    - Log 2 = 99% reduction
    - Log 3 = 99.9% reduction

    The log removal rate coefficient determines how effectively the
    treatment system removes pathogens per unit log time.

    Examples
    --------
    >>> import numpy as np
    >>> from gwtransport.logremoval import residence_time_to_log_removal
    >>> residence_times = np.array([1.0, 10.0, 100.0])
    >>> log_removal_rate = 2.0
    >>> residence_time_to_log_removal(
    ...     residence_times=residence_times, log_removal_rate=log_removal_rate
    ... )  # doctest: +NORMALIZE_WHITESPACE
    array([0., 2., 4.])

    >>> # Single residence time
    >>> residence_time_to_log_removal(residence_times=5.0, log_removal_rate=1.5)
    np.float64(1.0484550065040283)

    >>> # 2D array of residence times
    >>> residence_times_2d = np.array([[1.0, 10.0], [100.0, 1000.0]])
    >>> residence_time_to_log_removal(
    ...     residence_times=residence_times_2d, log_removal_rate=1.0
    ... )
    array([[0., 1.],
           [2., 3.]])
    """
    # Convert to numpy array for consistent handling
    residence_times = np.asarray(residence_times, dtype=float)

    # Calculate log removal using the formula
    return log_removal_rate * np.log10(residence_times)


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
    *, r: npt.ArrayLike, rt_alpha: float, rt_beta: float, log_removal_rate: float
) -> npt.NDArray[np.floating]:
    """
    Compute the probability density function (PDF) of log removal given a gamma distribution for the residence time.

    gamma(rt_alpha, rt_beta) = gamma(apv_alpha, apv_beta / flow)

    Parameters
    ----------
    r : array-like
        Log removal values at which to compute the PDF.
    rt_alpha : float
        Shape parameter of the gamma distribution for residence time.
    rt_beta : float
        Scale parameter of the gamma distribution for residence time.
    log_removal_rate : float
        Coefficient for log removal calculation (R = log_removal_rate * log10(T)).

    Returns
    -------
    pdf : numpy.ndarray
        PDF values corresponding to the input r values.
    """
    # Convert input to array
    r = np.asarray(r)

    # Compute the transformed PDF
    t_values = 10 ** (r / log_removal_rate)

    return (
        (np.log(10) / (log_removal_rate * gamma(rt_alpha) * (rt_beta**rt_alpha)))
        * (t_values**rt_alpha)
        * np.exp(-t_values / rt_beta)
    )


def gamma_cdf(
    *, r: npt.ArrayLike, rt_alpha: float, rt_beta: float, log_removal_rate: float
) -> npt.NDArray[np.floating]:
    """
    Compute the cumulative distribution function (CDF) of log removal given a gamma distribution for the residence time.

    gamma(rt_alpha, rt_beta) = gamma(apv_alpha, apv_beta / flow)

    Parameters
    ----------
    r : array-like
        Log removal values at which to compute the CDF.
    alpha : float
        Shape parameter of the gamma distribution for residence time.
    beta : float
        Scale parameter of the gamma distribution for residence time.
    log_removal_rate : float
        Coefficient for log removal calculation (R = log_removal_rate * log10(T)).

    Returns
    -------
    cdf : numpy.ndarray
        CDF values corresponding to the input r values.
    """
    # Convert input to array
    r = np.asarray(r)

    # Compute t values corresponding to r values
    t_values = 10 ** (r / log_removal_rate)

    # Use the gamma CDF directly
    return stats.gamma.cdf(t_values, a=rt_alpha, scale=rt_beta)


def gamma_mean(*, rt_alpha: float, rt_beta: float, log_removal_rate: float) -> float:
    """
    Compute the mean of the log removal distribution given a gamma distribution for the residence time.

    gamma(rt_alpha, rt_beta) = gamma(apv_alpha, apv_beta / flow)

    Parameters
    ----------
    rt_alpha : float
        Shape parameter of the gamma distribution for residence time.
    rt_beta : float
        Scale parameter of the gamma distribution for residence time.
    log_removal_rate : float
        Coefficient for log removal calculation (R = log_removal_rate * log10(T)).

    Returns
    -------
    mean : float
        Mean value of the log removal distribution.
    """
    # Calculate E[R] = log_removal_rate * E[log10(T)]
    # For gamma distribution: E[ln(T)] = digamma(alpha) + ln(beta_adjusted)
    # Convert to log10: E[log10(T)] = E[ln(T)] / ln(10)

    return (log_removal_rate / np.log(10)) * (digamma(rt_alpha) + np.log(rt_beta))


def gamma_find_flow_for_target_mean(
    *, target_mean: float, apv_alpha: float, apv_beta: float, log_removal_rate: float
) -> float:
    """
    Find the flow rate flow that produces a specified target mean log removal given a gamma distribution for the residence time.

    gamma(rt_alpha, rt_beta) = gamma(apv_alpha, apv_beta / flow)

    Parameters
    ----------
    target_mean : float
        Target mean log removal value.
    apv_alpha : float
        Shape parameter of the gamma distribution for residence time.
    apv_beta : float
        Scale parameter of the gamma distribution for pore volume.
    log_removal_rate : float
        Coefficient for log removal calculation (R = log_removal_rate * log10(T)).

    Returns
    -------
    flow : float
        Flow rate that produces the target mean log removal.

    Notes
    -----
    This function uses the analytical solution derived from the mean formula.
    From E[R] = (log_removal_rate/ln(10)) * (digamma(alpha) + ln(beta) - ln(Q)),
    we can solve for Q to get:
    flow = beta * exp(ln(10)*target_mean/log_removal_rate - digamma(alpha))
    """
    # Rearranging the mean formula to solve for Q:
    # target_mean = (log_removal_rate/ln(10)) * (digamma(alpha) + ln(beta) - ln(Q))
    # ln(Q) = digamma(alpha) + ln(beta) - (ln(10)*target_mean/log_removal_rate)
    # Q = beta * exp(-(ln(10)*target_mean/log_removal_rate - digamma(alpha)))
    return apv_beta * np.exp(digamma(apv_alpha) - (np.log(10) * target_mean) / log_removal_rate)
