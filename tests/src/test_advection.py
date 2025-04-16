import numpy as np
import pandas as pd
import pytest

from gwtransport1d.advection import gamma_forward


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


# Test gamma_forward function
def test_gamma_forward_basic(sample_time_series, gamma_params):
    """Test basic functionality of gamma_forward."""
    cin, flow = sample_time_series

    cout = gamma_forward(
        cin=cin, flow=flow, alpha=gamma_params["alpha"], beta=gamma_params["beta"], n_bins=gamma_params["n_bins"]
    )

    # Check output type and length
    assert isinstance(cout, pd.Series)
    assert len(cout) == len(cin)

    # Check output values are non-negative
    assert np.all(cout[~np.isnan(cout)] >= 0)

    # Check conservation of mass (approximately)
    # Note: This might not hold exactly due to boundary effects
    assert np.isclose(cout.mean(), cin.mean(), rtol=0.01)


def test_gamma_forward_retardation(sample_time_series, gamma_params):
    """Test gamma_forward with different retardation factors."""
    cin, flow = sample_time_series

    # Compare results with different retardation factors
    cout1 = gamma_forward(
        cin=cin, flow=flow, alpha=gamma_params["alpha"], beta=gamma_params["beta"], retardation_factor=1.0
    )

    cout2 = gamma_forward(
        cin=cin, flow=flow, alpha=gamma_params["alpha"], beta=gamma_params["beta"], retardation_factor=2.0
    )

    # The signal with higher retardation should be more delayed
    assert not np.allclose(cout1, cout2)


def test_gamma_forward_constant_input(gamma_params):
    """Test gamma_forward with constant input concentration."""
    # Create constant input concentration
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    cout = gamma_forward(cin=cin, flow=flow, alpha=gamma_params["alpha"], beta=gamma_params["beta"])

    # Output should also be constant and equal to input
    assert np.allclose(cout[~np.isnan(cout)], 1.0, rtol=1e-10)
