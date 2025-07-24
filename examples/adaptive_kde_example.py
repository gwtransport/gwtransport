"""
Example of adaptive kernel density estimation with varying bandwidth.

This example demonstrates how to create a KDE where each point has a
variance that adapts to the local density of neighboring points.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gwtransport.gamma import bins, mean_std_to_alpha_beta


def estimate_adaptive_bandwidth_vectorized(points, k=2):
    """Vectorized version for better performance."""
    points = np.asarray(points)

    # Create distance matrix
    distance_matrix = np.abs(points[:, np.newaxis] - points[np.newaxis, :])

    # Sort each row and take k+1 smallest (includes diagonal)
    sorted_distances = np.sort(distance_matrix, axis=1)

    # Take k nearest neighbors (excluding self at index 0)
    k_nearest = sorted_distances[:, 1 : k + 1]

    # Average distance for each point
    avg_distances = np.mean(k_nearest, axis=1)

    # Convert to variances
    return avg_distances**2


def create_adaptive_normal_kde(points, variances, x_eval):
    """Create adaptive KDE given points, variances, and evaluation points."""
    kde_result = np.zeros_like(x_eval)

    for point, var in zip(points, variances, strict=False):
        kde_result += (1 / len(points)) * norm.pdf(x_eval, point, np.sqrt(var))

    return kde_result


def create_adaptive_gamma_kde(points, variances, x_eval):
    """Create adaptive KDE given points, variances, and evaluation points."""
    kde_result = np.zeros_like(x_eval)

    for point, var in zip(points, variances, strict=False):
        alpha, beta = mean_std_to_alpha_beta(point, np.sqrt(var))
        kde_result += (1 / len(points)) * gamma_dist.pdf(x_eval, alpha, scale=beta)

    return kde_result


create_adaptive_kde = create_adaptive_normal_kde


def main():
    """Demonstrate adaptive KDE with sample data."""
    # Create sample data with clusters and isolated points
    np.random.seed(42)

    # Clustered points
    cluster1 = np.random.normal(2, 0.3, 4)
    cluster2 = np.random.normal(8, 0.4, 3)

    # Isolated points
    isolated = np.array([5.5, 11.2])

    # Combine all points
    points = np.concatenate([cluster1, cluster2, isolated])
    points = np.sort(points)

    print(f"Sample points: {points}")

    # Estimate adaptive bandwidths
    variances = estimate_adaptive_bandwidth_vectorized(points, k=2)
    std_devs = np.sqrt(variances)

    print(f"Adaptive standard deviations: {std_devs}")

    # Create evaluation grid
    x_eval = np.linspace(points.min() - 2, points.max() + 2, 200)

    # Compute adaptive KDE
    adaptive_kde = create_adaptive_kde(points, variances, x_eval)

    # For comparison, create fixed bandwidth KDE
    fixed_std = np.std(points) * len(points) ** (-1 / 5)  # Silverman's rule
    fixed_variances = np.full_like(variances, fixed_std**2)
    fixed_kde = create_adaptive_kde(points, fixed_variances, x_eval)

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Points and individual kernels
    ax1.plot(x_eval, adaptive_kde, "b-", label="Adaptive KDE", linewidth=2)

    # Plot individual kernels
    for point, std in zip(points, std_devs, strict=False):
        individual_kernel = (1 / len(points)) * norm.pdf(x_eval, point, std)
        ax1.plot(x_eval, individual_kernel, "r--", alpha=0.5, linewidth=1)

    # Mark the data points
    ax1.scatter(points, np.zeros_like(points), c="red", s=50, zorder=5, label="Data points")

    ax1.set_xlabel("x")
    ax1.set_ylabel("Density")
    ax1.set_title("Adaptive KDE with Individual Kernels")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Comparison between adaptive and fixed bandwidth
    ax2.plot(x_eval, adaptive_kde, "b-", label="Adaptive bandwidth", linewidth=2)
    ax2.plot(x_eval, fixed_kde, "g-", label="Fixed bandwidth", linewidth=2)
    ax2.scatter(points, np.zeros_like(points), c="red", s=50, zorder=5, label="Data points")

    ax2.set_xlabel("x")
    ax2.set_ylabel("Density")
    ax2.set_title("Adaptive vs Fixed Bandwidth KDE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Gamma distribution adaptive KDE
    # Create two gamma distributions with different parameters
    gamma1_params = {"alpha": 2.0, "beta": 1.5}
    gamma2_params = {"alpha": 4.0, "beta": 0.8}

    # Get bins and expected values for each gamma distribution
    for params in [gamma1_params, gamma2_params]:
        gbins = bins(alpha=params["alpha"], beta=params["beta"], n_bins=10)
        # gamma_points = bins["expected_value"]

        gamma_variances = estimate_adaptive_bandwidth_vectorized(gbins["expected_value"], k=1)
        x_gamma = np.linspace(0, max(gbins["expected_value"]) + 3, 300)
        gamma_kde = create_adaptive_kde(gbins["expected_value"], gamma_variances, x_gamma)

        pdf = gamma_dist.pdf(x_gamma, params["alpha"], scale=params["beta"])

        ax3.plot(
            x_gamma, gamma_kde, label=f"Adaptive KDE (alpha={params['alpha']}, beta={params['beta']})", linewidth=2
        )
        ax3.plot(x_gamma, pdf, "--", label=f"True PDF (alpha={params['alpha']}, beta={params['beta']})", alpha=0.7)
        ax3.scatter(
            gbins["expected_value"],
            np.zeros_like(gbins["expected_value"]),
            s=30,
            zorder=5,
            label="Expected values",
        )

    ax3.set_xlabel("x")
    ax3.set_ylabel("Density")
    ax3.set_title("Gamma Mixture: Adaptive KDE from Bin Expected Values")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(gbins["expected_value"]) + 2)

    plt.tight_layout()
    plt.savefig("adaptive_kde_demo.png", dpi=150, bbox_inches="tight")
    print("Plot saved as 'adaptive_kde_demo.png'")

    # Print bandwidth information
    print(f"\nFixed bandwidth (Silverman's rule): {fixed_std:.3f}")
    print("Adaptive bandwidths:")
    for point, std in zip(points, std_devs, strict=False):
        print(f"  Point {point:.2f}: std = {std:.3f}")

    print("\nGamma distribution parameters:")
    print(f"Gamma 1: alpha={gamma1_params['alpha']}, beta={gamma1_params['beta']}")
    print(f"Gamma 2: alpha={gamma2_params['alpha']}, beta={gamma2_params['beta']}")
    print(f"Number of expected values used: {len(gbins['expected_value'])}")
    print(f"Gamma expected values: {gbins['expected_value']}")
    print(f"Gamma adaptive std devs: {np.sqrt(gamma_variances)}")


if __name__ == "__main__":
    main()
