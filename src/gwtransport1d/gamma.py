"""Functions for working with gamma distributions."""
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc
from scipy.stats import gamma as gamma_dist

# Create a logger instance
logger = logging.getLogger(__name__)


def gamma_equal_mass_bins(alpha, beta, n_bins):
    """
    Divide gamma distribution into n bins with equal probability mass.

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
        msg = "Alpha and beta must be positive"
        raise ValueError(msg)
    if n_bins < 1:
        msg = "Number of bins must be positive"
        raise ValueError(msg)

    # Calculate boundaries for equal mass bins
    prob_per_bin = 1.0 / n_bins
    quantiles = np.linspace(0, 1, n_bins + 1)  # includes 0 and 1
    bin_boundaries = gamma_dist.ppf(quantiles, alpha, scale=beta)

    # Create arrays for bounds
    lower_bounds = bin_boundaries[:-1]
    upper_bounds = bin_boundaries[1:]

    diff_alpha = gammainc(alpha, upper_bounds[:-1] / beta) - gammainc(alpha, lower_bounds[:-1] / beta)
    diff_alpha_plus_1 = gammainc(alpha + 1, upper_bounds[:-1] / beta) - gammainc(alpha + 1, lower_bounds[:-1] / beta)
    finite_expectations = beta * alpha * diff_alpha_plus_1 / diff_alpha

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

def bin_masses(alpha, beta, lower_bounds, upper_bounds):
   """
   Calculate prbability mass for each bin in gamma distribution.

   Parameters
   ----------
   alpha : float
       Shape parameter of gamma distribution (must be > 0)
   beta : float
       Scale parameter of gamma distribution (must be > 0)
   lower_bounds : array-like
       Lower bounds of bins
   upper_bounds : array-like
       Upper bounds of bins (can include inf)

   Returns
   -------
   array
       Probability mass for each bin
   """
   # Convert inputs to numpy arrays
   lower_bounds = np.asarray(lower_bounds)
   upper_bounds = np.asarray(upper_bounds)

   # Parameter validation
   if alpha <= 0 or beta <= 0:
       msg = "Alpha and beta must be positive"
       raise ValueError(msg)
   if len(lower_bounds) != len(upper_bounds):
       msg = "Lower and upper bounds must have same length"
       raise ValueError(msg)
   if np.any(lower_bounds > upper_bounds):
       msg = "Lower bounds must be less than upper bounds"
       raise ValueError(msg)

   # Handle infinite upper bounds
   is_infinite = np.isinf(upper_bounds)

   # Initialize probability mass array
   masses = np.zeros(len(lower_bounds))

   # Calculate for finite bounds
   finite_mask = ~is_infinite
   if np.any(finite_mask):
       masses[finite_mask] = (
           gammainc(alpha, upper_bounds[finite_mask] / beta) -
           gammainc(alpha, lower_bounds[finite_mask] / beta)
       )

   # Calculate for infinite bounds
   if np.any(is_infinite):
       masses[is_infinite] = 1 - gammainc(alpha, lower_bounds[is_infinite] / beta)

   return masses

# Example usage
if __name__ == "__main__":
    # Example parameters
    alpha = 500.0
    beta = 1.0
    n_bins = 8

    bins = gamma_equal_mass_bins(alpha, beta, n_bins)

    logger.info("Gamma distribution (α=%s, β=%s) divided into %d equal-mass bins:", alpha, beta, n_bins)
    logger.info("-" * 80)
    logger.info("%3s %10s %10s %10s %10s", "Bin", "Lower", "Upper", "E[X|bin]", "P(bin)")
    logger.info("-" * 80)

    for i in range(n_bins):
        upper = f"{bins['upper_bound'][i]:.3f}" if not np.isinf(bins["upper_bound"][i]) else "∞"
        lower = f"{bins['lower_bound'][i]:.3f}"
        expected = f"{bins['expected_value'][i]:.3f}"
        prob = f"{bins['probability_mass'][i]:.3f}"
        logger.info("%3d %10s %10s %10s %10s", i, lower, upper, expected, prob)

    # Verify total probability is exactly 1
    logger.info("\nTotal probability mass: %.6f", bins['probability_mass'].sum())

    # Verify expected value is close to the mean of the distribution
    mean = alpha * beta
    expected_value = np.sum(bins["expected_value"] * bins["probability_mass"])
    logger.info("Mean of distribution: %.3f", mean)
    logger.info("Expected value of bins: %.3f", expected_value)

    mass_per_bin = bin_masses(alpha, beta, bins['lower_bound'], bins['upper_bound'])
    logger.info("Total probability mass: %.6f", mass_per_bin.sum())
    logger.info("Probability mass per bin:")
    logger.info(mass_per_bin)

    # plot the gamma distribution and the bins
    x = np.linspace(0, 530, 1000)
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
