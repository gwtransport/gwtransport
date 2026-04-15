import numpy as np
import pandas as pd
import pytest
from scipy.stats import gamma as gamma_dist

from gwtransport.gamma import (
    alpha_beta_to_mean_std,
    bin_masses,
    mean_std_to_alpha_beta,
    parse_parameters,
)
from gwtransport.gamma import (
    bins as gamma_bins,
)


# Fixtures
@pytest.fixture
def sample_time_series():
    """Create sample time series data for testing."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    concentration = pd.Series(np.sin(np.linspace(0, 4 * np.pi, len(dates))) + 2, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # Constant flow of 100 m3/day
    return concentration, flow


@pytest.fixture
def gamma_params():
    """Sample gamma distribution parameters."""
    return {
        "alpha": 200.0,  # Shape parameter
        "beta": 5.0,  # Scale parameter
        "n_bins": 10,  # Number of bins
    }


# Test bin_masses function
def test_bin_masses_basic():
    """Test basic functionality of bin_masses."""
    edges = np.array([0, 1, 2, np.inf])
    masses = bin_masses(alpha=2.0, beta=1.0, bin_edges=edges)

    assert len(masses) == len(edges) - 1
    assert np.all(masses >= 0)
    assert np.isclose(np.sum(masses), 1.0, rtol=1e-10)


def test_bin_masses_invalid_params():
    """Test bin_masses with invalid parameters."""
    edges = np.array([0, 1, 2])

    with pytest.raises(ValueError):
        bin_masses(alpha=-1, beta=1.0, bin_edges=edges)

    with pytest.raises(ValueError):
        bin_masses(alpha=1.0, beta=-1, bin_edges=edges)


def test_bin_masses_single_bin():
    """Test bin_masses with a single bin."""
    edges = np.array([0, np.inf])
    masses = bin_masses(alpha=2.0, beta=1.0, bin_edges=edges)

    assert len(masses) == 1
    assert np.isclose(masses[0], 1.0, rtol=1e-10)


# Test bins function
def test_bins_basic(gamma_params):
    """Test basic functionality of bins."""
    result = gamma_bins(**gamma_params)

    # Check all required keys are present
    expected_keys = {"lower_bound", "upper_bound", "edges", "expected_values", "probability_mass"}
    assert set(result.keys()) == expected_keys

    # Check array lengths
    n_bins = gamma_params["n_bins"]
    assert len(result["lower_bound"]) == n_bins
    assert len(result["upper_bound"]) == n_bins
    assert len(result["edges"]) == n_bins + 1
    assert len(result["expected_values"]) == n_bins
    assert len(result["probability_mass"]) == n_bins

    # Check probability masses sum to 1
    assert np.isclose(np.sum(result["probability_mass"]), 1.0, rtol=1e-10)

    # Check bin edges are monotonically increasing
    assert np.all(np.diff(result["edges"]) > 0)

    # Check if the sum of the expected value of each bin is equal to the expected value of the distribution (alpha * beta)
    expected_value_bins = np.sum(result["expected_values"] * result["probability_mass"])
    expected_value_gamma = gamma_params["alpha"] * gamma_params["beta"]
    assert expected_value_gamma == expected_value_bins


def test_bins_expected_values(gamma_params):
    """Test that expected values are within their respective bins."""
    result = gamma_bins(**gamma_params)

    for i in range(len(result["expected_values"])):
        assert result["lower_bound"][i] <= result["expected_values"][i] <= result["upper_bound"][i]


# Edge cases and error handling
def test_invalid_parameters():
    """Test error handling for invalid parameters."""
    with pytest.raises(ValueError):
        gamma_bins(alpha=-1, beta=1, n_bins=10)

    with pytest.raises(ValueError):
        gamma_bins(alpha=1, beta=-1, n_bins=10)


def test_numerical_stability():
    """Test numerical stability with extreme parameters."""
    # Test with very small alpha and beta
    result_small = gamma_bins(alpha=1e-5, beta=1e-5, n_bins=10)
    assert not np.any(np.isnan(result_small["expected_values"]))

    # Test with very large alpha and beta
    result_large = gamma_bins(alpha=1e5, beta=1e5, n_bins=10)
    assert not np.any(np.isnan(result_large["expected_values"]))


def test_gamma_mean_std_to_alpha_beta_basic():
    """Test gamma_mean_std_to_alpha_beta with typical input values."""
    mean, std = 10.0, 2.0
    alpha, beta = mean_std_to_alpha_beta(mean=mean, std=std)
    assert alpha > 0
    assert beta > 0

    # Convert back and check if we get approximately the same mean/std
    mean_back, std_back = alpha_beta_to_mean_std(alpha=alpha, beta=beta)
    assert np.isclose(mean, mean_back, rtol=1e-7), f"Expected mean ~ {mean}, got {mean_back}"
    assert np.isclose(std, std_back, rtol=1e-7), f"Expected std ~ {std}, got {std_back}"


def test_gamma_mean_std_to_alpha_beta_with_loc():
    """Round-trip with a positive location parameter."""
    mean, std, loc = 10.0, 2.0, 3.0
    alpha, beta = mean_std_to_alpha_beta(mean=mean, std=std, loc=loc)

    # Excess-over-loc moments: mean_excess = 7.0, std = 2.0
    np.testing.assert_allclose(alpha, (7.0 / 2.0) ** 2, rtol=1e-15)
    np.testing.assert_allclose(beta, 2.0**2 / 7.0, rtol=1e-15)

    mean_back, std_back = alpha_beta_to_mean_std(alpha=alpha, beta=beta, loc=loc)
    np.testing.assert_allclose(mean, mean_back, rtol=1e-15)
    np.testing.assert_allclose(std, std_back, rtol=1e-15)


def test_gamma_mean_std_to_alpha_beta_zero_std():
    """Test gamma_mean_std_to_alpha_beta when std is zero."""
    mean, std = 10.0, 0.0
    with pytest.raises(ValueError, match="std must be positive"):
        mean_std_to_alpha_beta(mean=mean, std=std)


def test_gamma_mean_std_to_alpha_beta_loc_too_large():
    """loc must be strictly less than mean when using (mean, std)."""
    with pytest.raises(ValueError, match="mean must be strictly greater than loc"):
        mean_std_to_alpha_beta(mean=5.0, std=1.0, loc=5.0)

    with pytest.raises(ValueError, match="mean must be strictly greater than loc"):
        mean_std_to_alpha_beta(mean=5.0, std=1.0, loc=6.0)


def test_gamma_mean_std_to_alpha_beta_negative_loc():
    """loc must be non-negative."""
    with pytest.raises(ValueError, match="loc must be non-negative"):
        mean_std_to_alpha_beta(mean=5.0, std=1.0, loc=-0.1)


def test_gamma_alpha_beta_to_mean_std_basic():
    """Test gamma_alpha_beta_to_mean_std with typical alpha/beta."""
    alpha, beta = 4.0, 2.0
    mean_expected = alpha * beta
    mean, std = alpha_beta_to_mean_std(alpha=alpha, beta=beta)
    assert mean == mean_expected, f"Expected mean = {mean_expected}, got {mean}"
    assert np.isclose(std, 4.0, rtol=1e-7), f"Expected std ~ 4.0, got {std}"


def test_gamma_alpha_beta_to_mean_std_with_loc():
    """mean shifts by loc; std is invariant under the shift."""
    alpha, beta, loc = 4.0, 2.0, 3.5
    mean, std = alpha_beta_to_mean_std(alpha=alpha, beta=beta, loc=loc)
    np.testing.assert_allclose(mean, alpha * beta + loc, rtol=1e-15)
    np.testing.assert_allclose(std, np.sqrt(alpha) * beta, rtol=1e-15)


def test_expected_bin_values_monte_carlo():
    """Test expected bin values using Monte Carlo sampling with strong curvature gamma distributions."""
    np.random.seed(42)

    # Test parameters for gamma distributions with strong curvature (low alpha values)
    test_cases = [
        {"alpha": 0.5, "beta": 2.0, "n_bins": 5},  # Strong curvature, low alpha
        {"alpha": 1.0, "beta": 1.5, "n_bins": 4},  # Exponential-like
        {"alpha": 2.0, "beta": 0.5, "n_bins": 6},  # Moderate curvature
    ]

    n_samples = 100000
    tolerance = 0.005  # 0.5% tolerance for convergence

    for params in test_cases:
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        n_bins = int(params["n_bins"])

        # Get theoretical bin properties
        bin_result = gamma_bins(alpha=alpha, beta=beta, n_bins=n_bins)
        theoretical_expected = bin_result["expected_values"]
        lower_bounds = bin_result["lower_bound"]
        upper_bounds = bin_result["upper_bound"]

        # Generate samples from gamma distribution
        samples = np.array(gamma_dist.rvs(alpha, scale=beta, size=n_samples, random_state=42))

        # Calculate empirical expected values for each bin
        empirical_expected = np.zeros(n_bins)

        for i in range(n_bins):
            # Find samples that fall within this bin
            if i == n_bins - 1:  # Last bin goes to infinity
                bin_mask = samples >= lower_bounds[i]
            else:
                bin_mask = (samples >= lower_bounds[i]) & (samples < upper_bounds[i])

            bin_samples = samples[bin_mask]

            if len(bin_samples) > 0:
                empirical_expected[i] = np.mean(bin_samples)
            else:
                empirical_expected[i] = np.nan

        # Compare theoretical and empirical expected values
        for i in range(n_bins):
            if not np.isnan(empirical_expected[i]):
                relative_error = abs(empirical_expected[i] - theoretical_expected[i]) / theoretical_expected[i]
                assert relative_error < tolerance, (
                    f"Bin {i} for alpha={alpha}, beta={beta}: "
                    f"Theoretical={theoretical_expected[i]:.6f}, "
                    f"Empirical={empirical_expected[i]:.6f}, "
                    f"Relative error={relative_error:.6f} > {tolerance}"
                )


def test_expected_bin_values_convergence():
    """Test convergence of expected values to theoretical values with increasing sample sizes."""
    np.random.seed(123)

    # Strong curvature gamma distribution
    alpha, beta = 0.8, 1.2
    n_bins = 4

    # Get theoretical values
    bin_result = gamma_bins(alpha=alpha, beta=beta, n_bins=n_bins)
    theoretical_expected = bin_result["expected_values"]
    lower_bounds = bin_result["lower_bound"]
    upper_bounds = bin_result["upper_bound"]

    # Test with increasing sample sizes
    sample_sizes = [1000, 5000, 25000, 100000]

    for n_samples in sample_sizes:
        samples = np.array(gamma_dist.rvs(alpha, scale=beta, size=n_samples, random_state=123))

        empirical_expected = np.zeros(n_bins)

        for i in range(n_bins):
            if i == n_bins - 1:
                bin_mask = samples >= lower_bounds[i]
            else:
                bin_mask = (samples >= lower_bounds[i]) & (samples < upper_bounds[i])

            bin_samples = samples[bin_mask]

            if len(bin_samples) > 0:
                empirical_expected[i] = np.mean(bin_samples)
            else:
                empirical_expected[i] = np.nan

        # Check that empirical values are getting closer to theoretical values
        # Use relaxed tolerance for smaller sample sizes
        tolerance = 0.04 if n_samples <= 5000 else 0.02

        for i in range(n_bins):
            if not np.isnan(empirical_expected[i]):
                relative_error = abs(empirical_expected[i] - theoretical_expected[i]) / theoretical_expected[i]
                assert relative_error < tolerance, (
                    f"Sample size {n_samples}, bin {i}: "
                    f"Theoretical={theoretical_expected[i]:.6f}, "
                    f"Empirical={empirical_expected[i]:.6f}, "
                    f"Relative error={relative_error:.6f} > {tolerance}"
                )


def test_multiple_gamma_distributions_expected_values():
    """Test expected bin values for multiple gamma distributions with different parameters."""
    np.random.seed(456)

    # Various gamma distributions with different characteristics
    distributions = [
        {"alpha": 0.3, "beta": 3.0, "description": "Very strong curvature"},
        {"alpha": 1.0, "beta": 2.0, "description": "Exponential"},
        {"alpha": 2.5, "beta": 1.5, "description": "Moderate shape"},
        {"alpha": 5.0, "beta": 0.8, "description": "Bell-shaped"},
    ]

    n_samples = 50000
    n_bins = 5
    tolerance = 0.01

    for dist_params in distributions:
        alpha = float(dist_params["alpha"])
        beta = float(dist_params["beta"])

        # Get theoretical bin properties
        bin_result = gamma_bins(alpha=alpha, beta=beta, n_bins=n_bins)
        theoretical_expected = bin_result["expected_values"]
        lower_bounds = bin_result["lower_bound"]
        upper_bounds = bin_result["upper_bound"]

        # Generate samples
        samples = np.array(gamma_dist.rvs(alpha, scale=beta, size=n_samples, random_state=456))

        # Calculate empirical expected values
        empirical_expected = np.zeros(n_bins)

        for i in range(n_bins):
            if i == n_bins - 1:
                bin_mask = samples >= lower_bounds[i]
            else:
                bin_mask = (samples >= lower_bounds[i]) & (samples < upper_bounds[i])

            bin_samples = samples[bin_mask]

            if len(bin_samples) > 0:
                empirical_expected[i] = np.mean(bin_samples)
            else:
                empirical_expected[i] = np.nan

        # Validate convergence for each bin
        for i in range(n_bins):
            if not np.isnan(empirical_expected[i]):
                relative_error = abs(empirical_expected[i] - theoretical_expected[i]) / theoretical_expected[i]
                assert relative_error < tolerance, (
                    f"{dist_params['description']} (alpha={alpha}, beta={beta}), bin {i}: "
                    f"Theoretical={theoretical_expected[i]:.6f}, "
                    f"Empirical={empirical_expected[i]:.6f}, "
                    f"Relative error={relative_error:.6f} > {tolerance}"
                )


# =============================================================================
# Tests for parse_parameters function
# =============================================================================


def test_parse_parameters_with_alpha_beta():
    """Test parse_parameters with alpha and beta provided."""
    alpha_in, beta_in = 5.0, 2.0
    alpha_out, beta_out, loc_out = parse_parameters(alpha=alpha_in, beta=beta_in)

    assert alpha_out == alpha_in
    assert beta_out == beta_in
    assert loc_out == 0.0


def test_parse_parameters_with_mean_std():
    """Test parse_parameters with mean and std provided."""
    mean, std = 10.0, 3.0
    alpha, beta, loc = parse_parameters(mean=mean, std=std)

    # Verify alpha and beta are positive
    assert alpha > 0
    assert beta > 0
    assert loc == 0.0

    # Verify conversion back gives same mean/std
    mean_back, std_back = alpha_beta_to_mean_std(alpha=alpha, beta=beta, loc=loc)
    assert np.isclose(mean, mean_back)
    assert np.isclose(std, std_back)


def test_parse_parameters_with_loc():
    """parse_parameters returns loc alongside (alpha, beta) for both parameterizations."""
    alpha, beta, loc = parse_parameters(alpha=5.0, beta=2.0, loc=3.0)
    assert (alpha, beta, loc) == (5.0, 2.0, 3.0)

    alpha, beta, loc = parse_parameters(mean=10.0, std=3.0, loc=2.0)
    np.testing.assert_allclose(alpha, (8.0 / 3.0) ** 2, rtol=1e-15)
    np.testing.assert_allclose(beta, 9.0 / 8.0, rtol=1e-15)
    assert loc == 2.0


def test_parse_parameters_negative_loc():
    """loc must be non-negative."""
    with pytest.raises(ValueError, match="loc must be non-negative"):
        parse_parameters(alpha=1.0, beta=1.0, loc=-0.5)


def test_parse_parameters_loc_exceeds_mean():
    """In mean/std form, loc must be strictly less than mean."""
    with pytest.raises(ValueError, match="mean must be strictly greater than loc"):
        parse_parameters(mean=5.0, std=1.0, loc=5.0)


def test_parse_parameters_missing_both():
    """Test parse_parameters raises error when both parameter sets are missing."""
    with pytest.raises(ValueError, match=r"Either \(alpha, beta\) or \(mean, std\) must be provided"):
        parse_parameters()


def test_parse_parameters_partial_alpha_beta():
    """Test parse_parameters raises error with partial alpha/beta."""
    with pytest.raises(ValueError, match="alpha and beta must both be provided"):
        parse_parameters(alpha=5.0)

    with pytest.raises(ValueError, match="alpha and beta must both be provided"):
        parse_parameters(beta=2.0)


def test_parse_parameters_partial_mean_std():
    """Test parse_parameters raises error with partial mean/std."""
    with pytest.raises(ValueError, match=r"Either \(alpha, beta\) or \(mean, std\) must be provided"):
        parse_parameters(mean=10.0)

    with pytest.raises(ValueError, match=r"Either \(alpha, beta\) or \(mean, std\) must be provided"):
        parse_parameters(std=3.0)


def test_parse_parameters_negative_alpha():
    """Test parse_parameters raises error with negative alpha."""
    with pytest.raises(ValueError, match="Alpha and beta must be positive"):
        parse_parameters(alpha=-1.0, beta=2.0)


def test_parse_parameters_negative_beta():
    """Test parse_parameters raises error with negative beta."""
    with pytest.raises(ValueError, match="Alpha and beta must be positive"):
        parse_parameters(alpha=5.0, beta=-2.0)


def test_parse_parameters_zero_alpha():
    """Test parse_parameters raises error with zero alpha."""
    with pytest.raises(ValueError, match="Alpha and beta must be positive"):
        parse_parameters(alpha=0.0, beta=2.0)


def test_parse_parameters_zero_beta():
    """Test parse_parameters raises error with zero beta."""
    with pytest.raises(ValueError, match="Alpha and beta must be positive"):
        parse_parameters(alpha=5.0, beta=0.0)


# =============================================================================
# Tests for bins function with quantile_edges
# =============================================================================


def test_bins_with_quantile_edges_basic():
    """Test bins with custom quantile edges."""
    quantile_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = gamma_bins(alpha=10.0, beta=2.0, quantile_edges=quantile_edges)

    # Verify correct number of bins
    assert len(result["probability_mass"]) == 4
    assert len(result["expected_values"]) == 4
    assert len(result["edges"]) == 5

    # Verify probability masses sum to 1
    assert np.isclose(np.sum(result["probability_mass"]), 1.0)

    # Verify probability masses match quantile differences
    expected_masses = np.diff(quantile_edges)
    np.testing.assert_allclose(result["probability_mass"], expected_masses, rtol=1e-15)


def test_bins_with_quantile_edges_unequal():
    """Test bins with unequal quantile edges."""
    quantile_edges = np.array([0.0, 0.1, 0.3, 0.7, 1.0])
    result = gamma_bins(alpha=5.0, beta=3.0, quantile_edges=quantile_edges)

    # Verify correct number of bins
    assert len(result["probability_mass"]) == 4

    # Verify probability masses
    expected_masses = np.diff(quantile_edges)
    np.testing.assert_allclose(result["probability_mass"], expected_masses, rtol=1e-15)

    # Verify expected values are within bins
    for i in range(len(result["expected_values"])):
        assert result["lower_bound"][i] <= result["expected_values"][i] <= result["upper_bound"][i]


def test_bins_quantile_edges_vs_n_bins():
    """Test that quantile_edges produces same result as n_bins for uniform quantiles."""
    alpha, beta = 8.0, 1.5
    n_bins = 5

    # Using n_bins
    result_n_bins = gamma_bins(alpha=alpha, beta=beta, n_bins=n_bins)

    # Using quantile_edges (uniform)
    quantile_edges = np.linspace(0, 1, n_bins + 1)
    result_quantiles = gamma_bins(alpha=alpha, beta=beta, quantile_edges=quantile_edges)

    # Results should be identical to machine precision
    np.testing.assert_allclose(result_n_bins["edges"], result_quantiles["edges"], rtol=1e-15)
    np.testing.assert_allclose(result_n_bins["probability_mass"], result_quantiles["probability_mass"], rtol=1e-15)
    np.testing.assert_allclose(result_n_bins["expected_values"], result_quantiles["expected_values"], rtol=1e-15)


def test_bins_neither_n_bins_nor_quantiles():
    """Test bins uses default n_bins=100 when neither n_bins nor quantile_edges provided."""
    result = gamma_bins(alpha=5.0, beta=2.0)
    # Should use default n_bins=100
    assert len(result["probability_mass"]) == 100
    assert len(result["expected_values"]) == 100
    assert len(result["edges"]) == 101


def test_bins_both_n_bins_and_quantiles():
    """Test bins uses quantile_edges when both n_bins and quantile_edges provided (quantile_edges takes precedence)."""
    quantile_edges = np.array([0.0, 0.5, 1.0])

    # When both are provided, quantile_edges takes precedence
    result = gamma_bins(alpha=5.0, beta=2.0, n_bins=10, quantile_edges=quantile_edges)
    # Should use quantile_edges (2 bins), not n_bins (10)
    assert len(result["probability_mass"]) == 2
    assert len(result["expected_values"]) == 2
    assert len(result["edges"]) == 3


def test_bins_n_bins_too_small():
    """Test bins raises error with n_bins <= 1."""
    with pytest.raises(ValueError, match="Number of bins must be greater than 1"):
        gamma_bins(alpha=5.0, beta=2.0, n_bins=1)

    with pytest.raises(ValueError, match="Number of bins must be greater than 1"):
        gamma_bins(alpha=5.0, beta=2.0, n_bins=0)


def test_bins_loc_zero_matches_legacy():
    """With loc=0, bins() reproduces the two-parameter result bit-for-bit."""
    r_default = gamma_bins(alpha=5.0, beta=2.0, n_bins=20)
    r_explicit = gamma_bins(alpha=5.0, beta=2.0, loc=0.0, n_bins=20)
    np.testing.assert_array_equal(r_default["edges"], r_explicit["edges"])
    np.testing.assert_array_equal(r_default["expected_values"], r_explicit["expected_values"])
    np.testing.assert_array_equal(r_default["probability_mass"], r_explicit["probability_mass"])


def test_bins_loc_shifts_edges_and_values():
    """A positive loc shifts bin edges and expected values by exactly loc, leaves masses unchanged."""
    r0 = gamma_bins(alpha=5.0, beta=2.0, n_bins=20)
    loc = 7.5
    r_shift = gamma_bins(alpha=5.0, beta=2.0, loc=loc, n_bins=20)

    # Finite edges shift by loc (last edge is inf in both)
    np.testing.assert_allclose(r_shift["edges"][:-1], r0["edges"][:-1] + loc, rtol=1e-15)
    assert np.isinf(r0["edges"][-1])
    assert np.isinf(r_shift["edges"][-1])

    np.testing.assert_allclose(r_shift["expected_values"], r0["expected_values"] + loc, rtol=1e-15)
    np.testing.assert_array_equal(r_shift["probability_mass"], r0["probability_mass"])

    # Total mean equals alpha*beta + loc
    total_mean = float(np.sum(r_shift["expected_values"] * r_shift["probability_mass"]))
    np.testing.assert_allclose(total_mean, 5.0 * 2.0 + loc, rtol=1e-15)


def test_bins_mean_std_loc_conservation():
    """With (mean, std, loc) the discretized total mean equals mean to machine precision."""
    mean, std, loc = 30000.0, 8100.0, 5000.0
    r = gamma_bins(mean=mean, std=std, loc=loc, n_bins=100)
    total_mean = float(np.sum(r["expected_values"] * r["probability_mass"]))
    np.testing.assert_allclose(total_mean, mean, rtol=1e-12)


def test_bins_loc_monte_carlo_expected_values():
    """Monte Carlo validation of expected_values under a shifted gamma distribution."""
    np.random.seed(789)
    alpha, beta, loc = 2.0, 0.5, 1.3
    n_bins = 6
    n_samples = 100_000

    r = gamma_bins(alpha=alpha, beta=beta, loc=loc, n_bins=n_bins)

    # Samples from the shifted gamma distribution
    samples = np.array(gamma_dist.rvs(alpha, loc=loc, scale=beta, size=n_samples, random_state=789))

    # Assign each sample to a bin via searchsorted on the finite interior edges.
    # ``r["edges"]`` has n_bins+1 entries from ``loc`` to +inf; the interior edges
    # (indices 1..n_bins-1) partition the samples into bins 0..n_bins-1.
    interior_edges = r["edges"][1:-1]
    bin_idx = np.searchsorted(interior_edges, samples, side="right")
    sums = np.bincount(bin_idx, weights=samples, minlength=n_bins)
    counts = np.bincount(bin_idx, minlength=n_bins)
    # Bins may be empty in rare cases; only compare populated bins.
    populated = counts > 0
    empirical = np.full(n_bins, np.nan)
    empirical[populated] = sums[populated] / counts[populated]

    tolerance = 0.005
    rel_err = np.abs(empirical - r["expected_values"]) / r["expected_values"]
    failures = populated & (rel_err >= tolerance)
    assert not np.any(failures), (
        f"bins with rel_err >= {tolerance}: {np.where(failures)[0].tolist()}; "
        f"theoretical={r['expected_values']}, empirical={empirical}"
    )


def test_bins_loc_lower_bound_equals_loc():
    """With loc>0, the first lower_bound equals loc exactly."""
    loc = 42.0
    r = gamma_bins(alpha=3.0, beta=1.5, loc=loc, n_bins=10)
    assert r["lower_bound"][0] == loc
    assert r["edges"][0] == loc


def test_bins_alpha_less_than_one_large_nbins():
    """Test numerical stability of gamma.bins() with alpha < 1 and large n_bins.

    Alpha < 1 produces a distribution with a singularity at 0, which can
    cause numerical issues with quantile computation.
    """
    result = gamma_bins(alpha=0.5, beta=10.0, n_bins=100)

    # No NaN values in expected values
    assert not np.any(np.isnan(result["expected_values"]))

    # All expected values should be positive
    assert np.all(result["expected_values"] > 0)

    # Probability masses should sum to 1
    np.testing.assert_allclose(np.sum(result["probability_mass"]), 1.0, rtol=1e-10)

    # Edges should be monotonically increasing
    assert np.all(np.diff(result["edges"]) > 0)

    # Expected values should be within their bin bounds
    for i in range(len(result["expected_values"])):
        assert result["lower_bound"][i] <= result["expected_values"][i] <= result["upper_bound"][i]
