import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma, gammainc
from scipy.stats import gamma as gamma_dist

# Create a logger instance
logger = logging.getLogger(__name__)

def gamma_equal_mass_bins(alpha, beta, n_bins):
    """
    Divide gamma distribution into n bins with equal probability mass using array operations.

    Parameters
    ----------
    alpha : float
        Shape parameter of gamma distribution (must be > 0)
    beta : float
        Scale parameter of gamma distribution (must be > 0)
    n_bins : int
        Number of bins

    Returns
    -------
    dict of arrays with keys:
        - lower_bound: lower bounds of bins
        - upper_bound: upper bounds of bins (last one is inf)
        - expected_value: expected values in bins
        - probability_mass: probability mass in bins (1/n_bins for all)
    """
    # Parameter validation
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    if n_bins < 1:
        raise ValueError("Number of bins must be positive")

    # Calculate boundaries for equal mass bins
    prob_per_bin = 1.0 / n_bins
    quantiles = np.linspace(0, 1, n_bins + 1)  # includes 0 and 1
    bin_boundaries = gamma_dist.ppf(quantiles, alpha, scale=beta)

    # Create arrays for bounds
    lower_bounds = bin_boundaries[:-1]
    upper_bounds = bin_boundaries[1:]

    # Calculate expected values using vectorized operations
    # For finite bounds
    gamma_diff_alpha = gamma(alpha) * (
        gammainc(alpha, upper_bounds[:-1] / beta) - gammainc(alpha, lower_bounds[:-1] / beta)
    )
    gamma_diff_alpha_plus_1 = gamma(alpha + 1) * (
        gammainc(alpha + 1, upper_bounds[:-1] / beta) - gammainc(alpha + 1, lower_bounds[:-1] / beta)
    )
    finite_expectations = beta * gamma_diff_alpha_plus_1 / gamma_diff_alpha

    # For the last bin (to infinity)
    gamma_ratio_alpha = 1 - gammainc(alpha, lower_bounds[-1] / beta)  # Upper tail
    gamma_ratio_alpha_plus_1 = 1 - gammainc(alpha + 1, lower_bounds[-1] / beta)  # Upper tail
    infinite_expectation = beta * alpha * gamma_ratio_alpha_plus_1 / gamma_ratio_alpha

    # Combine finite and infinite expectations
    expected_values = np.append(finite_expectations, infinite_expectation)

    # Create uniform probability mass array
    probability_mass = np.full(n_bins, prob_per_bin)

    return {
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "expected_value": expected_values,
        "probability_mass": probability_mass,
    }


# Example usage
if __name__ == "__main__":
    # Example parameters
    alpha = 10.0
    beta = 1.0
    n_bins = 8

    bins = gamma_equal_mass_bins(alpha, beta, n_bins)

    logger.info(f"Gamma distribution (α={alpha}, β={beta}) divided into {n_bins} equal-mass bins:")
    logger.info("-" * 80)
    logger.info(f"{'Bin':>3} {'Lower':>10} {'Upper':>10} {'E[X|bin]':>10} {'P(bin)':>10}")
    logger.info("-" * 80)

    for i in range(n_bins):
        upper = f"{bins['upper_bound'][i]:.3f}" if not np.isinf(bins["upper_bound"][i]) else "∞"
        lower = f"{bins['lower_bound'][i]:.3f}"
        expected = f"{bins['expected_value'][i]:.3f}"
        prob = f"{bins['probability_mass'][i]:.3f}"
        logger.info(f"{i:3d} {lower:>10} {upper:>10} {expected:>10} {prob:>10}")

    # Verify total probability is exactly 1
    logger.info(f"\nTotal probability mass: {bins['probability_mass'].sum():.6f}")

    # Verify expected value is close to the mean of the distribution
    mean = alpha * beta
    expected_value = np.sum(bins["expected_value"] * bins["probability_mass"])
    logger.info(f"Mean of distribution: {mean:.3f}")
    logger.info(f"Expected value of bins: {expected_value:.3f}")

    # plot the gamma distribution and the bins
    x = np.linspace(0, 30, 1000)
    y = gamma_dist.pdf(x, alpha, scale=beta)
    plt.plot(x, y, label="Gamma PDF")
    for i in range(n_bins):
        plt.axvline(bins["lower_bound"][i], color="black", linestyle="--", alpha=0.5)
        plt.axvline(bins["upper_bound"][i], color="black", linestyle="--", alpha=0.5)
        plt.axvline(bins["expected_value"][i], color="red", linestyle="--", alpha=0.5)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.show()
