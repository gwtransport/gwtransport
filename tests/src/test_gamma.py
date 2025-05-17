import numpy as np
import pandas as pd
import pytest

from gwtransport.gamma import (
    alpha_beta_to_mean_std,
    bin_masses,
    mean_std_to_alpha_beta,
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
    expected_keys = {"lower_bound", "upper_bound", "edges", "expected_value", "probability_mass"}
    assert set(result.keys()) == expected_keys

    # Check array lengths
    n_bins = gamma_params["n_bins"]
    assert len(result["lower_bound"]) == n_bins
    assert len(result["upper_bound"]) == n_bins
    assert len(result["edges"]) == n_bins + 1
    assert len(result["expected_value"]) == n_bins
    assert len(result["probability_mass"]) == n_bins

    # Check probability masses sum to 1
    assert np.isclose(np.sum(result["probability_mass"]), 1.0, rtol=1e-10)

    # Check bin edges are monotonically increasing
    assert np.all(np.diff(result["edges"]) > 0)

    # Check if the sum of the expected value of each bin is equal to the expected value of the distribution (alpha * beta)
    expected_value_bins = np.sum(result["expected_value"] * result["probability_mass"])
    expected_value_gamma = gamma_params["alpha"] * gamma_params["beta"]
    assert expected_value_gamma == expected_value_bins


def test_bins_expected_values(gamma_params):
    """Test that expected values are within their respective bins."""
    result = gamma_bins(**gamma_params)

    for i in range(len(result["expected_value"])):
        assert result["lower_bound"][i] <= result["expected_value"][i] <= result["upper_bound"][i]


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
    assert not np.any(np.isnan(result_small["expected_value"]))

    # Test with very large alpha and beta
    result_large = gamma_bins(alpha=1e5, beta=1e5, n_bins=10)
    assert not np.any(np.isnan(result_large["expected_value"]))


def test_gamma_mean_std_to_alpha_beta_basic():
    """Test gamma_mean_std_to_alpha_beta with typical input values."""
    mean, std = 10.0, 2.0
    alpha, beta = mean_std_to_alpha_beta(mean, std)
    assert alpha > 0
    assert beta > 0

    # Convert back and check if we get approximately the same mean/std
    mean_back, std_back = alpha_beta_to_mean_std(alpha, beta)
    assert np.isclose(mean, mean_back, rtol=1e-7), f"Expected mean ~ {mean}, got {mean_back}"
    assert np.isclose(std, std_back, rtol=1e-7), f"Expected std ~ {std}, got {std_back}"


def test_gamma_mean_std_to_alpha_beta_zero_std():
    """Test gamma_mean_std_to_alpha_beta when std is zero."""
    mean, std = 10.0, 0.0
    with pytest.raises(ZeroDivisionError):
        mean_std_to_alpha_beta(mean, std)


def test_gamma_alpha_beta_to_mean_std_basic():
    """Test gamma_alpha_beta_to_mean_std with typical alpha/beta."""
    alpha, beta = 4.0, 2.0
    mean_expected = alpha * beta
    mean, std = alpha_beta_to_mean_std(alpha, beta)
    assert mean == mean_expected, f"Expected mean = {mean_expected}, got {mean}"
    assert np.isclose(std, 4.0, rtol=1e-7), f"Expected std ~ 4.0, got {std}"
