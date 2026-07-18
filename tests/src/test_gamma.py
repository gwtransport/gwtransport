import numpy as np
import pytest
from scipy.stats import gamma as gamma_dist

from gwtransport.gamma import (
    alpha_beta_loc_to_mean_std,
    mean_std_loc_to_alpha_beta,
    parse_parameters,
)
from gwtransport.gamma import (
    bins as gamma_bins,
)


# Fixtures
@pytest.fixture
def gamma_params():
    """Sample gamma distribution parameters."""
    return {
        "alpha": 200.0,  # Shape parameter
        "beta": 5.0,  # Scale parameter
        "n_bins": 10,  # Number of bins
    }


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

    # Check probability masses sum to 1 (the sum is exactly representable; honor rtol with atol=0)
    np.testing.assert_allclose(np.sum(result["probability_mass"]), 1.0, rtol=1e-14)

    # Check bin edges are monotonically increasing
    assert np.all(np.diff(result["edges"]) > 0)

    # Each conditional mean must lie within its own bin. This per-bin oracle is
    # immune to permutations that a sum-only conservation check would not catch.
    assert np.all(result["lower_bound"] <= result["expected_values"])
    assert np.all(result["expected_values"] <= result["upper_bound"])

    # Check if the sum of the expected value of each bin is equal to the expected value of the distribution (alpha * beta)
    expected_value_bins = np.sum(result["expected_values"] * result["probability_mass"])
    expected_value_gamma = gamma_params["alpha"] * gamma_params["beta"]
    np.testing.assert_allclose(expected_value_gamma, expected_value_bins, rtol=1e-12)


# Edge cases and error handling
def test_invalid_parameters():
    """Test error handling for invalid parameters."""
    with pytest.raises(ValueError):
        gamma_bins(alpha=-1, beta=1, n_bins=10)

    with pytest.raises(ValueError):
        gamma_bins(alpha=1, beta=-1, n_bins=10)


@pytest.mark.parametrize(
    ("alpha", "beta", "n_bins"),
    [
        (0.05, 1.0, 10),  # alpha << 1: PDF singularity at 0
        (0.5, 1.0, 10),  # alpha < 1
        (0.5, 1e-3, 10),  # alpha < 1, beta tiny
        (1.0, 1.0, 10),  # exponential
        (10.0, 1.0, 10),  # moderate
        (1e4, 1.0, 10),  # very large alpha
        (1.0, 1e3, 10),  # beta large
        (0.5, 10.0, 100),  # alpha < 1 with many bins (PDF singularity + fine discretization)
    ],
)
def test_bins_extreme_regimes_finite_and_conservative(alpha, beta, n_bins):
    """Across extreme regimes, ``bins`` must produce finite, in-bin values and conserve mass + first moment.

    Replaces the original ``test_numerical_stability`` (only asserted non-NaN).
    """
    r = gamma_bins(alpha=alpha, beta=beta, n_bins=n_bins)

    assert np.all(np.isfinite(r["expected_values"])), f"non-finite expected_values at alpha={alpha}, beta={beta}"
    assert np.all(r["expected_values"] > 0)

    np.testing.assert_allclose(np.sum(r["probability_mass"]), 1.0, rtol=1e-14)

    # sum(p_i * E[X|bin_i]) == alpha * beta to machine precision
    total_mean = float(np.sum(r["expected_values"] * r["probability_mass"]))
    np.testing.assert_allclose(total_mean, alpha * beta, rtol=1e-14)

    # Conditional means lie strictly within their bins
    assert np.all(r["lower_bound"] <= r["expected_values"])
    assert np.all(r["expected_values"] <= r["upper_bound"])


def test_gamma_mean_std_loc_to_alpha_beta_basic():
    """Test gamma_mean_std_loc_to_alpha_beta with typical input values."""
    mean, std = 10.0, 2.0
    alpha, beta = mean_std_loc_to_alpha_beta(mean=mean, std=std)
    assert alpha > 0
    assert beta > 0

    # Convert back: the conversion is bit-exact algebra, so the round-trip is exact.
    mean_back, std_back = alpha_beta_loc_to_mean_std(alpha=alpha, beta=beta)
    np.testing.assert_allclose(mean, mean_back, rtol=1e-15)
    np.testing.assert_allclose(std, std_back, rtol=1e-15)


def test_gamma_mean_std_loc_to_alpha_beta_with_loc():
    """Round-trip with a positive location parameter."""
    mean, std, loc = 10.0, 2.0, 3.0
    alpha, beta = mean_std_loc_to_alpha_beta(mean=mean, std=std, loc=loc)

    # Excess-over-loc moments: mean_excess = 7.0, std = 2.0
    np.testing.assert_allclose(alpha, (7.0 / 2.0) ** 2, rtol=1e-15)
    np.testing.assert_allclose(beta, 2.0**2 / 7.0, rtol=1e-15)

    mean_back, std_back = alpha_beta_loc_to_mean_std(alpha=alpha, beta=beta, loc=loc)
    np.testing.assert_allclose(mean, mean_back, rtol=1e-15)
    np.testing.assert_allclose(std, std_back, rtol=1e-15)


def test_gamma_mean_std_loc_to_alpha_beta_zero_std():
    """Test gamma_mean_std_loc_to_alpha_beta when std is zero."""
    mean, std = 10.0, 0.0
    with pytest.raises(ValueError, match="std must be positive"):
        mean_std_loc_to_alpha_beta(mean=mean, std=std)


def test_gamma_mean_std_loc_to_alpha_beta_loc_too_large():
    """loc must be strictly less than mean when using (mean, std)."""
    with pytest.raises(ValueError, match="mean must be strictly greater than loc"):
        mean_std_loc_to_alpha_beta(mean=5.0, std=1.0, loc=5.0)

    with pytest.raises(ValueError, match="mean must be strictly greater than loc"):
        mean_std_loc_to_alpha_beta(mean=5.0, std=1.0, loc=6.0)


def test_gamma_mean_std_loc_to_alpha_beta_negative_loc():
    """loc must be non-negative."""
    with pytest.raises(ValueError, match="loc must be non-negative"):
        mean_std_loc_to_alpha_beta(mean=5.0, std=1.0, loc=-0.1)


def test_gamma_alpha_beta_loc_to_mean_std_basic():
    """Test gamma_alpha_beta_loc_to_mean_std with typical alpha/beta."""
    alpha, beta = 4.0, 2.0
    mean_expected = alpha * beta
    mean, std = alpha_beta_loc_to_mean_std(alpha=alpha, beta=beta)
    assert mean == mean_expected, f"Expected mean = {mean_expected}, got {mean}"
    # std = sqrt(4) * 2 = 4.0 exactly
    np.testing.assert_allclose(std, 4.0, rtol=0, atol=0)


def test_gamma_alpha_beta_loc_to_mean_std_with_loc():
    """mean shifts by loc; std is invariant under the shift."""
    alpha, beta, loc = 4.0, 2.0, 3.5
    mean, std = alpha_beta_loc_to_mean_std(alpha=alpha, beta=beta, loc=loc)
    np.testing.assert_allclose(mean, alpha * beta + loc, rtol=1e-15)
    np.testing.assert_allclose(std, np.sqrt(alpha) * beta, rtol=1e-15)


# The unshifted Monte Carlo accuracy tests have been removed: their bug-catching is a
# strict subset of test_bins_expected_values_against_scipy_quadrature (exact, deterministic)
# and the single retained sampling sanity check test_bins_loc_monte_carlo_expected_values
# (shifted case). See the per-module review for the deduplication rationale.


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

    # Verify conversion back gives same mean/std (bit-exact algebra)
    mean_back, std_back = alpha_beta_loc_to_mean_std(alpha=alpha, beta=beta, loc=loc)
    np.testing.assert_allclose(mean, mean_back, rtol=1e-15)
    np.testing.assert_allclose(std, std_back, rtol=1e-15)


def test_parse_parameters_with_loc():
    """parse_parameters returns loc alongside (alpha, beta) for both parameterizations."""
    alpha, beta, loc = parse_parameters(alpha=5.0, beta=2.0, loc=3.0)
    assert (alpha, beta, loc) == (5.0, 2.0, 3.0)

    alpha, beta, loc = parse_parameters(mean=10.0, std=3.0, loc=2.0)
    np.testing.assert_allclose(alpha, (8.0 / 3.0) ** 2, rtol=1e-15)
    np.testing.assert_allclose(beta, 9.0 / 8.0, rtol=1e-15)
    assert loc == 2.0


def test_parse_parameters_both_pairs_provided():
    """Regression: providing both (alpha, beta) and (mean, std) must raise, not silently ignore mean/std.

    Before the fix, (alpha, beta) silently won and (mean, std) were dropped without error.
    """
    with pytest.raises(ValueError, match="Provide either"):
        parse_parameters(alpha=5.0, beta=2.0, mean=10.0, std=3.0)

    # A single stray mean/std alongside a full (alpha, beta) pair must also raise.
    with pytest.raises(ValueError, match="Provide either"):
        parse_parameters(alpha=5.0, beta=2.0, mean=10.0)

    with pytest.raises(ValueError, match="Provide either"):
        parse_parameters(alpha=5.0, beta=2.0, std=3.0)


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

    # Verify probability masses sum to 1 (the sum is exactly representable)
    np.testing.assert_allclose(np.sum(result["probability_mass"]), 1.0, rtol=1e-14)

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


def test_bins_quantile_edges_must_span_unit_interval():
    """Regression: quantile_edges not spanning [0, 1] must raise, not silently lose probability mass.

    Before the fix, ``quantile_edges=[0.1, 0.5, 0.9]`` ran and produced bins covering
    only 0.8 of the total probability mass, silently violating the conservation contract.
    """
    with pytest.raises(ValueError, match="must start at 0 and end at 1"):
        gamma_bins(alpha=5.0, beta=2.0, quantile_edges=np.array([0.1, 0.5, 0.9]))

    # First edge not 0
    with pytest.raises(ValueError, match="must start at 0 and end at 1"):
        gamma_bins(alpha=5.0, beta=2.0, quantile_edges=np.array([0.1, 0.5, 1.0]))

    # Last edge not 1
    with pytest.raises(ValueError, match="must start at 0 and end at 1"):
        gamma_bins(alpha=5.0, beta=2.0, quantile_edges=np.array([0.0, 0.5, 0.9]))


def test_bins_quantile_edges_must_be_strictly_increasing():
    """Non-monotone or duplicated quantile_edges must raise."""
    with pytest.raises(ValueError, match="strictly increasing"):
        gamma_bins(alpha=5.0, beta=2.0, quantile_edges=np.array([0.0, 0.5, 0.5, 1.0]))

    with pytest.raises(ValueError, match="strictly increasing"):
        gamma_bins(alpha=5.0, beta=2.0, quantile_edges=np.array([0.0, 0.7, 0.3, 1.0]))


def test_bins_quantile_edges_too_few():
    """quantile_edges with only 2 entries (n_bins=1) must raise via the n_bins guard."""
    with pytest.raises(ValueError, match="Number of bins must be greater than 1"):
        gamma_bins(alpha=5.0, beta=2.0, quantile_edges=np.array([0.0, 1.0]))


def test_bins_empty_quantile_edges_raises():
    """Regression (G1): an empty quantile_edges must raise ValueError, not leak IndexError.

    Before the fix the empty array passed the (vacuously true) strictly-increasing check
    and then indexed ``quantile_edges[0]`` on a size-0 array, leaking an IndexError instead
    of the documented ValueError family.
    """
    with pytest.raises(ValueError):
        gamma_bins(alpha=2.0, beta=3.0, quantile_edges=np.array([]))

    # A single edge cannot define even one bin either.
    with pytest.raises(ValueError):
        gamma_bins(alpha=2.0, beta=3.0, quantile_edges=np.array([0.0]))


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


@pytest.mark.parametrize(
    ("mean", "std", "loc"),
    [
        (10.0, 2.0, 0.0),  # alpha=25, loc=0
        (30000.0, 8100.0, 5000.0),  # large with shift
        (5.0, 1.5, 1.0),  # moderate
        (2.0, 1.5, 0.0),  # alpha=(2/1.5)**2 ~ 1.78
        (0.5, 0.5, 0.0),  # alpha=1 (exponential)
        (0.6, 1.0, 0.0),  # alpha < 1 (PDF singularity at 0)
        (1e6, 1.0, 0.0),  # alpha very large (1e12)
    ],
)
def test_bins_parameterization_equivalence(mean, std, loc):
    """``bins(mean=m, std=s, loc=l)`` must equal ``bins(alpha, beta, loc=l)`` bit-for-bit.

    Both arms share the same converter, so the conversion *formula* (caught by
    ``test_bins_second_moment_matches_parameterization``) is not what this test
    targets. What it pins is the dispatcher: ``parse_parameters`` correctly forwarding
    ``loc`` through both branches, neither branch silently dropping a default, and the
    downstream pipeline being deterministic in ``(alpha, beta, loc)``.
    """
    n_bins = 20
    r_msl = gamma_bins(mean=mean, std=std, loc=loc, n_bins=n_bins)
    alpha, beta = mean_std_loc_to_alpha_beta(mean=mean, std=std, loc=loc)
    r_ab = gamma_bins(alpha=alpha, beta=beta, loc=loc, n_bins=n_bins)

    np.testing.assert_array_equal(r_msl["edges"], r_ab["edges"])
    np.testing.assert_array_equal(r_msl["expected_values"], r_ab["expected_values"])
    np.testing.assert_array_equal(r_msl["probability_mass"], r_ab["probability_mass"])
    np.testing.assert_array_equal(r_msl["lower_bound"], r_ab["lower_bound"])
    np.testing.assert_array_equal(r_msl["upper_bound"], r_ab["upper_bound"])


@pytest.mark.parametrize(
    ("mean", "std", "loc"),
    [
        (10.0, 2.0, 0.0),
        (30000.0, 8100.0, 5000.0),
        (5.0, 1.5, 1.0),
        (2.0, 0.5, 0.5),
    ],
)
def test_bins_second_moment_matches_parameterization(mean, std, loc):
    """``std`` is the parameter that controls the variance: ``Var(Gamma) = alpha * beta**2 == std**2`` exactly.

    The (mean, std, loc) parameterization sets ``alpha = (mean-loc)**2 / std**2`` and
    ``beta = std**2 / (mean-loc)``, so ``alpha * beta**2 == std**2`` is an algebraic
    identity. A coefficient swap (e.g. ``alpha = (mean-loc) / std**2``) would silently
    lose this identity at machine-precision tolerance.
    """
    alpha, beta, loc_out = parse_parameters(mean=mean, std=std, loc=loc)
    np.testing.assert_allclose(alpha * beta**2, std**2, rtol=1e-15)
    np.testing.assert_allclose(gamma_dist(alpha, loc=loc_out, scale=beta).var(), std**2, rtol=1e-12)
    np.testing.assert_allclose(gamma_dist(alpha, loc=loc_out, scale=beta).mean(), mean, rtol=1e-12)


def test_bins_quantile_edge_construction_inverts_to_quantile_diff():
    """The bin edges produced by ``gamma.bins`` recover the input quantile diff exactly.

    The construction
        edges = gamma_dist.ppf(q, alpha, scale=beta) + loc
        diff(gamma_dist.cdf(edges - loc, alpha, scale=beta)) = diff(q)
    holds by the CDF-PPF round-trip identity ``F(F^{-1}(q)) == q``. Catches any
    inconsistency between the edge-construction path and the mass-computation path.
    """
    quantile_edges = np.array([0.0, 0.1, 0.3, 0.55, 0.85, 1.0])
    alpha, beta, loc = 3.0, 2.0, 1.5
    r = gamma_bins(alpha=alpha, beta=beta, loc=loc, quantile_edges=quantile_edges)

    # probability_mass is constructed directly as np.diff(q) inside bins
    np.testing.assert_array_equal(r["probability_mass"], np.diff(quantile_edges))

    # Re-deriving the masses from the constructed (unshifted) edges must reproduce diff(q)
    unshifted_edges = r["edges"] - loc
    cdf = gamma_dist.cdf(unshifted_edges, alpha, scale=beta)
    np.testing.assert_allclose(np.diff(cdf), np.diff(quantile_edges), rtol=1e-13)


def test_bins_loc_monte_carlo_expected_values():
    """Monte Carlo validation of expected_values under a shifted gamma distribution.

    Determinism comes from the ``random_state=`` argument on ``rvs`` below.
    """
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


def test_bins_expected_values_against_scipy_quadrature():
    """Validate ``gamma.bins()['expected_values']`` against scipy.stats.gamma.expect to machine precision.

    ``gamma.bins`` uses the closed-form identity
        E[X * 1_{a <= X < b}] = alpha * beta * (F_{alpha+1}(b/beta) - F_{alpha+1}(a/beta))
    so the conditional mean within a bin is
        E[X | a <= X < b] = alpha * beta * (CDF diff of Gamma(alpha+1, beta)) / P(a <= X < b)

    Compared against scipy adaptive quadrature (``gamma.expect``), which is accurate to roughly
    1e-8 absolute error by default. A coefficient swap such as ``alpha * beta`` -> ``alpha + beta``
    would produce errors many orders of magnitude larger than the quadrature tolerance.
    """
    test_cases = [
        # (alpha, beta, loc, n_bins / quantile_edges)
        (0.5, 2.0, 0.0, 5),  # strong curvature
        (1.0, 1.5, 0.0, 4),  # exponential
        (2.5, 1.5, 0.0, 6),  # moderate
        (5.0, 0.8, 0.0, 8),  # bell-shaped
        (3.0, 2.0, 4.5, 7),  # shifted
        (0.7, 3.0, 1.2, 5),  # shifted strong-curvature
    ]
    custom_quantile_edges = np.array([0.0, 0.1, 0.35, 0.6, 0.85, 1.0])

    for alpha, beta, loc, n_bins in test_cases:
        for edges_spec in (n_bins, custom_quantile_edges):
            kwargs: dict = {"alpha": alpha, "beta": beta, "loc": loc}
            if isinstance(edges_spec, int):
                kwargs["n_bins"] = edges_spec
            else:
                kwargs["quantile_edges"] = edges_spec

            result = gamma_bins(**kwargs)
            edges = result["edges"]
            expected_values = result["expected_values"]
            probability_mass = result["probability_mass"]

            dist = gamma_dist(alpha, scale=beta, loc=loc)
            n = len(expected_values)
            expected_quad = np.empty(n)
            for i in range(n):
                lo = edges[i]
                hi = edges[i + 1] if np.isfinite(edges[i + 1]) else np.inf
                # scipy.gamma.expect returns int_{lo}^{hi} x * f(x) dx (an unconditional integral).
                unconditional_mean = dist.expect(lambda x: x, lb=lo, ub=hi, conditional=False)
                expected_quad[i] = unconditional_mean / probability_mass[i]

            # scipy.integrate.quad default tolerance is ~1.49e-8 absolute, which scales with the
            # magnitude of the expected values. Use a comparable absolute tolerance.
            atol = 1e-7 * max(1.0, np.abs(expected_values).max())
            np.testing.assert_allclose(
                expected_values,
                expected_quad,
                atol=atol,
                rtol=1e-10,
                err_msg=f"alpha={alpha}, beta={beta}, loc={loc}, edges_spec={edges_spec}",
            )


# The former test_bins_alpha_less_than_one_large_nbins (alpha=0.5, beta=10.0, n_bins=100)
# is now covered, at tighter tolerance and with the first-moment check, by the
# (0.5, 10.0, 100) case in test_bins_extreme_regimes_finite_and_conservative.


# =============================================================================
# Numerical-envelope guards for bins() / parse_parameters (issue #331).
#
# Each degenerate input below previously returned silently-wrong structure -- expected
# values outside their own bins, E == loc, or an all-NaN distribution -- with no error.
# The guards now raise a clear ValueError. Every "must raise" test fails on the pre-guard
# baseline (no exception was raised there), which is what makes it a regression test.
# =============================================================================


def test_parse_parameters_rejects_nonfinite():
    """Non-finite alpha/beta/loc must raise, not slip through to an all-NaN distribution (GAM-F4).

    ``nan <= 0`` and ``nan < 0`` are both False, so a non-finite value passes the positivity checks.
    """
    with pytest.raises(ValueError, match="must be finite"):
        parse_parameters(alpha=np.nan, beta=2.0)
    with pytest.raises(ValueError, match="must be finite"):
        parse_parameters(alpha=2.0, beta=np.inf)
    with pytest.raises(ValueError, match="must be finite"):
        parse_parameters(alpha=2.0, beta=3.0, loc=np.nan)


def test_bins_rejects_nonfinite_parameters():
    """bins() surfaces the non-finite-parameter guard instead of returning an all-NaN dict (GAM-F4)."""
    with pytest.raises(ValueError, match="must be finite"):
        gamma_bins(alpha=np.nan, beta=2.0, n_bins=5)


def test_bins_rejects_huge_alpha():
    """alpha beyond float64 bin resolution (alpha + 1 == alpha) must raise (GAM-F1).

    Reached through the documented (mean, std) parameterization: std/(mean - loc) = 1e-8 gives
    alpha = 1e16 >= 2**53. On the baseline this returned noisy, non-monotone expected values with
    8 of 10 lying outside their own bins, and nothing was raised.
    """
    with pytest.raises(ValueError, match="too large for float64 bin resolution"):
        gamma_bins(mean=1.0, std=1e-8, n_bins=10)


def test_bins_rejects_tiny_quantile_gap():
    """A quantile gap below 1e-8 makes the conditional-mean CDF difference cancel catastrophically,
    placing an expected value outside its own (infinitesimal) bin; it must raise (GAM-F2)."""
    q0 = 0.5
    edges = np.array([0.0, q0, np.nextafter(q0, 1.0), 1.0])
    with pytest.raises(ValueError, match="quantile-edge gap"):
        gamma_bins(alpha=10.0, beta=1.0, loc=5.0, quantile_edges=edges)


def test_bins_rejects_expected_value_underflow():
    """A very small alpha underflows the leftmost equal-mass bin's expected value to loc (a
    numerically-zero pore volume); it must raise rather than propagate a bad value downstream (ADV-F2)."""
    with pytest.raises(ValueError, match="underflowed to loc"):
        gamma_bins(alpha=5e-3, beta=100.0, n_bins=100)


def test_bins_guards_leave_realistic_configs_untouched():
    """The #331 guards must not reject any realistic APVD: default and shifted configs, a fine
    250-bin grid, and custom quantile edges all still succeed with strictly in-bin expected values."""
    for kwargs in (
        {"mean": 30000.0, "std": 8100.0, "n_bins": 10},
        {"mean": 30000.0, "std": 8100.0, "loc": 5000.0, "n_bins": 250},
        {"alpha": 13.7, "beta": 2000.0, "n_bins": 20},
        {"mean": 30000.0, "std": 8100.0, "quantile_edges": np.array([0.0, 0.25, 0.5, 0.75, 1.0])},
    ):
        result = gamma_bins(**kwargs)
        assert result["probability_mass"].sum() == 1.0
        assert np.all(result["expected_values"] >= result["lower_bound"])
        assert np.all(result["expected_values"] <= result["upper_bound"])


def test_bins_accepts_shifted_heterogeneous_apvd():
    """The underflow guard must test the pre-shift excess conditional mean, not the shifted expected
    value. For a shifted, highly-heterogeneous APVD (loc > 0, small alpha) the excess conditional mean
    is tiny but strictly positive; the shifted ``loc + tiny_excess`` can round to exactly ``loc`` in
    float64 without any underflow, and that correctly-rounded, usable output must NOT be rejected.
    Both parameterizations reach the same distribution and both must be accepted."""
    for kwargs in (
        {"alpha": 0.15, "beta": 100.0, "loc": 5000.0, "n_bins": 100},
        {"mean": 6000.0, "std": 4472.0, "loc": 5000.0, "n_bins": 100},
    ):
        result = gamma_bins(**kwargs)
        # Expected values are at least loc (the excess is non-negative) and inside their own bins,
        # and the flow-weighted mean recovers the distribution mean.
        assert np.all(result["expected_values"] >= result["lower_bound"])
        assert np.all(result["expected_values"] <= result["upper_bound"])
        assert np.all(result["expected_values"] >= 5000.0)


def test_bins_rejects_expected_value_underflow_shifted():
    """loc > 0: a genuine underflow (excess conditional mean == 0) must still raise. This pins that the
    guard compares the excess against 0, not the shifted value against a hardcoded 0 -- a ``<= 0.0``
    regression would silently pass at loc=0 but return E == loc bins here (the dominant shifted case)."""
    with pytest.raises(ValueError, match="underflowed to loc"):
        gamma_bins(alpha=5e-3, beta=100.0, n_bins=100, loc=5.0)


def test_bins_rejects_huge_alpha_direct_alpha():
    """The huge-alpha guard fires regardless of parameterization (direct alpha=, not only mean/std)."""
    with pytest.raises(ValueError, match="too large for float64 bin resolution"):
        gamma_bins(alpha=1e16, beta=1.0, n_bins=10)


def test_parse_parameters_rejects_nonfinite_from_mean_std():
    """A non-finite mean/std (e.g. an upstream division by zero) is rejected after conversion, so the
    finite guard sits downstream of the (mean, std) -> (alpha, beta) converter, not only on the direct
    alpha/beta arm."""
    with pytest.raises(ValueError, match="must be finite"):
        parse_parameters(mean=np.inf, std=1.0)
