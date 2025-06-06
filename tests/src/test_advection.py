import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import distribution_forward, forward, gamma_forward


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


# Test forward function
def test_forward_basic(sample_time_series):
    """Test basic functionality of forward."""
    cin, flow = sample_time_series
    aquifer_pore_volume = 1000.0

    cout = forward(
        cin_series=cin,
        flow_series=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=1.0,
        cout_index="cin",
    )

    # Check output type and length
    assert isinstance(cout, pd.Series)
    assert len(cout) == len(cin)
    assert cout.name == "cout"

    # Check output values are non-negative (ignoring NaN values)
    valid_values = cout[~np.isnan(cout)]
    assert np.all(valid_values >= 0)


def test_forward_cout_index_options(sample_time_series):
    """Test forward with different cout_index options."""
    cin, flow = sample_time_series
    aquifer_pore_volume = 1000.0

    # Test cout_index="cin"
    cout_cin = forward(cin_series=cin, flow_series=flow, aquifer_pore_volume=aquifer_pore_volume, cout_index="cin")
    assert len(cout_cin) == len(cin)
    assert cout_cin.index.equals(cin.index)

    # Test cout_index="flow"
    cout_flow = forward(cin_series=cin, flow_series=flow, aquifer_pore_volume=aquifer_pore_volume, cout_index="flow")
    assert len(cout_flow) == len(flow)
    assert cout_flow.index.equals(flow.index)

    # Test cout_index="cout"
    cout_cout = forward(cin_series=cin, flow_series=flow, aquifer_pore_volume=aquifer_pore_volume, cout_index="cout")
    # This should have a shifted index
    assert len(cout_cout) == len(cin)
    assert not cout_cout.index.equals(cin.index)


def test_forward_invalid_cout_index(sample_time_series):
    """Test forward with invalid cout_index raises ValueError."""
    cin, flow = sample_time_series
    aquifer_pore_volume = 1000.0

    with pytest.raises(ValueError, match="Invalid cout_index"):
        forward(cin_series=cin, flow_series=flow, aquifer_pore_volume=aquifer_pore_volume, cout_index="invalid")


def test_forward_retardation(sample_time_series):
    """Test forward with different retardation factors."""
    cin, flow = sample_time_series
    aquifer_pore_volume = 1000.0

    # Compare results with different retardation factors
    cout1 = forward(
        cin_series=cin,
        flow_series=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=1.0,
        cout_index="cin",
    )

    cout2 = forward(
        cin_series=cin,
        flow_series=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=2.0,
        cout_index="cin",
    )

    # The signal with higher retardation should be different
    # We need to check where both have valid values
    valid_mask = ~np.isnan(cout1) & ~np.isnan(cout2)
    if np.any(valid_mask):
        assert not np.allclose(cout1[valid_mask], cout2[valid_mask])


# Test gamma_forward function
def test_gamma_forward_basic(sample_time_series, gamma_params):
    """Test basic functionality of gamma_forward."""
    cin, flow = sample_time_series

    cout = gamma_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        alpha=gamma_params["alpha"],
        beta=gamma_params["beta"],
        n_bins=gamma_params["n_bins"],
    )

    # Check output type and length
    assert isinstance(cout, pd.Series)
    assert len(cout) == len(flow)
    assert cout.name == "cout"

    # Check output values are non-negative (ignoring NaN values)
    valid_values = cout[~np.isnan(cout)]
    assert np.all(valid_values >= 0)


def test_gamma_forward_with_mean_std(sample_time_series):
    """Test gamma_forward using mean and std parameters."""
    cin, flow = sample_time_series
    mean = 1000.0
    std = 200.0

    cout = gamma_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        mean=mean,
        std=std,
        n_bins=10,
    )

    # Check output type and length
    assert isinstance(cout, pd.Series)
    assert len(cout) == len(flow)


def test_gamma_forward_retardation(sample_time_series, gamma_params):
    """Test gamma_forward with different retardation factors."""
    cin, flow = sample_time_series

    # Compare results with different retardation factors
    cout1 = gamma_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        alpha=gamma_params["alpha"],
        beta=gamma_params["beta"],
        retardation_factor=1.0,
    )

    cout2 = gamma_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        alpha=gamma_params["alpha"],
        beta=gamma_params["beta"],
        retardation_factor=2.0,
    )

    # The signal with higher retardation should be different
    valid_mask = ~np.isnan(cout1) & ~np.isnan(cout2)
    if np.any(valid_mask):
        assert not np.allclose(cout1[valid_mask], cout2[valid_mask])


def test_gamma_forward_constant_input():
    """Test gamma_forward with constant input concentration."""
    # Create constant input concentration
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    cout = gamma_forward(
        cin=cin, cin_tend=cin.index, cout_tend=flow.index, flow=flow, flow_tend=flow.index, alpha=200.0, beta=5.0
    )

    # Output should also be constant where valid (ignoring NaN values)
    valid_values = cout[~np.isnan(cout)]
    if len(valid_values) > 0:
        assert np.allclose(valid_values, 1.0, rtol=1e-2)


# Test distribution_forward function
def test_distribution_forward_basic(sample_time_series):
    """Test basic functionality of distribution_forward."""
    cin, flow = sample_time_series
    # Create simple pore volume distribution edges
    aquifer_pore_volume_edges = np.array([500, 1000, 1500, 2000])

    cout = distribution_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        aquifer_pore_volume_edges=aquifer_pore_volume_edges,
        retardation_factor=1.0,
    )

    # Check output type and length
    assert isinstance(cout, pd.Series)
    assert len(cout) == len(flow)
    assert cout.name == "cout"

    # Check output values are non-negative (ignoring NaN values)
    valid_values = cout[~np.isnan(cout)]
    assert np.all(valid_values >= 0)


def test_distribution_forward_different_time_edges(sample_time_series):
    """Test distribution_forward with different time edge specifications."""
    cin, flow = sample_time_series
    aquifer_pore_volume_edges = np.array([500, 1000, 1500])

    # Test with tedges
    cin_tedges = pd.date_range(start=cin.index[0] - pd.Timedelta(days=1), end=cin.index[-1], freq="D")
    flow_tedges = pd.date_range(start=flow.index[0] - pd.Timedelta(days=1), end=flow.index[-1], freq="D")
    cout_tedges = flow_tedges

    cout1 = distribution_forward(
        cin=cin,
        cin_tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        flow_tedges=flow_tedges,
        aquifer_pore_volume_edges=aquifer_pore_volume_edges,
    )

    # Test with tend
    cout2 = distribution_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        aquifer_pore_volume_edges=aquifer_pore_volume_edges,
    )

    # Both should produce valid outputs
    assert isinstance(cout1, pd.Series)
    assert isinstance(cout2, pd.Series)
    assert len(cout1) == len(flow)
    assert len(cout2) == len(flow)


def test_distribution_forward_single_bin(sample_time_series):
    """Test distribution_forward with a single bin (two edges)."""
    cin, flow = sample_time_series
    aquifer_pore_volume_edges = np.array([1000, 2000])

    cout = distribution_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        aquifer_pore_volume_edges=aquifer_pore_volume_edges,
    )

    # Check output type and length
    assert isinstance(cout, pd.Series)
    assert len(cout) == len(flow)


# Test error conditions
def test_gamma_forward_missing_parameters(sample_time_series):
    """Test that gamma_forward raises appropriate errors for missing parameters."""
    cin, flow = sample_time_series

    # Test missing both alpha/beta and mean/std
    with pytest.raises(ValueError):
        gamma_forward(cin=cin, cin_tend=cin.index, cout_tend=flow.index, flow=flow, flow_tend=flow.index)


def test_time_edge_consistency():
    """Test that time edges are handled consistently."""
    # Create small test data
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Test with consistent time edges
    cout = gamma_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        alpha=10.0,
        beta=100.0,
        n_bins=5,
    )

    assert isinstance(cout, pd.Series)
    assert len(cout) == len(flow)


def test_conservation_properties():
    """Test mass conservation properties where applicable."""
    # Create test data with longer time series for better conservation
    dates = pd.date_range(start="2020-01-01", end="2021-12-31", freq="D")
    cin = pd.Series(np.ones(len(dates)), index=dates)  # Constant input
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # Constant flow

    cout = gamma_forward(
        cin=cin,
        cin_tend=cin.index,
        cout_tend=flow.index,
        flow=flow,
        flow_tend=flow.index,
        alpha=10.0,
        beta=100.0,
        n_bins=20,
    )

    # For constant input and flow, output should eventually stabilize
    # Check the latter part of the series where it should be stable
    valid_mask = ~np.isnan(cout)
    if np.sum(valid_mask) > 100:  # If we have enough valid values
        stable_region = cout[valid_mask][-100:]  # Last 100 valid values
        assert np.std(stable_region) < 0.1  # Should be relatively stable


# Test edge cases
def test_empty_series():
    """Test handling of empty series."""
    empty_cin = pd.Series([], dtype=float)
    empty_flow = pd.Series([], dtype=float)

    # This should handle gracefully or raise appropriate error
    with pytest.raises((ValueError, IndexError)):
        gamma_forward(
            cin=empty_cin,
            cin_tend=empty_cin.index,
            cout_tend=empty_flow.index,
            flow=empty_flow,
            flow_tend=empty_flow.index,
            alpha=10.0,
            beta=100.0,
        )


def test_mismatched_series_lengths():
    """Test handling of mismatched series lengths."""
    dates_cin = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    dates_flow = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")

    cin = pd.Series(np.ones(len(dates_cin)), index=dates_cin)
    flow = pd.Series(np.ones(len(dates_flow)) * 100, index=dates_flow)

    # This should work - the function should handle different lengths
    cout = gamma_forward(
        cin=cin, cin_tend=cin.index, cout_tend=flow.index, flow=flow, flow_tend=flow.index, alpha=10.0, beta=100.0
    )

    assert isinstance(cout, pd.Series)
    assert len(cout) == len(flow)
