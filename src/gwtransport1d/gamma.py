"""Functions for working with gamma distributions."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gammainc
from scipy.stats import gamma as gamma_dist

from gwtransport1d.advection import cout_advection_distribution

# Create a logger instance
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True


def gamma_mean_std_to_alpha_beta(mean, std):
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
    tuple
        Shape and scale parameters of the gamma distribution.
    """
    alpha = mean**2 / std**2
    beta = std**2 / mean
    return alpha, beta


def gamma_alpha_beta_to_mean_std(alpha, beta):
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
    tuple
        Mean and standard deviation of the gamma distribution.
    """
    mean = alpha * beta
    std = np.sqrt(alpha) * beta
    return mean, std


def cout_advection_gamma(cin, flow, alpha, beta, n_bins=100, retardation_factor=1.0):
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer. The aquifer pore volume is approximated by a gamma distribution, with
    parameters alpha and beta.

    Parameters
    ----------
    cin : pandas.Series
        Concentration of the compound in the extracted water [ng/m3] or temperature in infiltrating water.
    flow : pandas.Series
        Flow rate of water in the aquifer [m3/day].
    alpha : float
        Shape parameter of gamma distribution (must be > 0)
    beta : float
        Scale parameter of gamma distribution (must be > 0)
    n_bins : int
        Number of bins to discretize the gamma distribution.
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    pandas.Series
        Concentration of the compound in the extracted water [ng/m3] or temperature.
    """
    bins = gamma_equal_mass_bins(alpha, beta, n_bins)
    return cout_advection_distribution(cin, flow, bins["edges"], retardation_factor=retardation_factor)


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
        Number of bins to divide the gamma distribution.

    Returns
    -------
    dict of arrays with keys:
        - lower_bound: lower bounds of bins (first one is 0)
        - upper_bound: upper bounds of bins (last one is inf)
        - edges: bin edges (lower_bound[0], upper_bound[0], ..., upper_bound[-1])
        - expected_value: expected values in bins
        - probability_mass: probability mass in bins (1/n_bins for all)
    """
    # Calculate boundaries for equal mass bins
    probability_mass = np.full(n_bins, 1.0 / n_bins)
    quantiles = np.linspace(0, 1, n_bins + 1)  # includes 0 and 1
    bin_edges = gamma_dist.ppf(quantiles, alpha, scale=beta)

    # Calculate expected value for each bin
    diff_alpha_plus_1 = bin_masses(alpha + 1, beta, bin_edges)
    expected_values = beta * alpha * diff_alpha_plus_1 / probability_mass

    return {
        "lower_bound": bin_edges[:-1],
        "upper_bound": bin_edges[1:],
        "edges": bin_edges,
        "expected_value": expected_values,
        "probability_mass": probability_mass,
    }


def bin_masses(alpha, beta, bin_edges):
    """
    Calculate probability mass for each bin in gamma distribution.

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
    array
        Probability mass for each bin
    """
    # Convert inputs to numpy arrays
    bin_edges = np.asarray(bin_edges)

    # Parameter validation
    if alpha <= 0 or beta <= 0:
        msg = "Alpha and beta must be positive"
        raise ValueError(msg)

    val = gammainc(alpha, bin_edges / beta)
    return val[1:] - val[:-1]


# Example usage
if __name__ == "__main__":
    # Example parameters
    alpha = 300.0
    beta = 15.0
    n_bins = 12

    bins = gamma_equal_mass_bins(alpha, beta, n_bins)

    logger.info("Gamma distribution (alpha=%s, beta=%s) divided into %d equal-mass bins:", alpha, beta, n_bins)
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
    logger.info("\nTotal probability mass: %.6f", bins["probability_mass"].sum())

    # Verify expected value is close to the mean of the distribution
    mean = alpha * beta
    expected_value = np.sum(bins["expected_value"] * bins["probability_mass"])
    logger.info("Mean of distribution: %.3f", mean)
    logger.info("Expected value of bins: %.3f", expected_value)

    mass_per_bin = bin_masses(alpha, beta, bins["edges"])
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
