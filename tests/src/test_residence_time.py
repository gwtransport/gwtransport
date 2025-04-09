import numpy as np
import pandas as pd
import pytest

from gwtransport1d.residence_time import residence_time_retarded


@pytest.fixture
def sample_flow_data():
    """Create sample flow data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.array([100.0, 110.0, 105.0, 95.0, 98.0, 102.0, 107.0, 103.0, 96.0, 99.0])
    return pd.Series(data=flow_values, index=dates)


@pytest.fixture
def constant_flow_data():
    """Create constant flow data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(dates), 100.0)
    return pd.Series(data=flow_values, index=dates)


def test_basic_extraction():
    """Test basic extraction scenario with constant flow."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow = pd.Series(data=[100.0] * 5, index=dates)
    pore_volume = 200.0

    result = residence_time_retarded(flow=flow, aquifer_pore_volume=pore_volume, direction="extraction")

    # With constant flow of 100 m³/day and pore volume of 200 m³,
    # residence time should be approximately 2 days
    assert np.isclose(result[0, -1], 2.0, rtol=0.1)


def test_basic_infiltration():
    """Test basic infiltration scenario with constant flow."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow = pd.Series(data=[100.0] * 5, index=dates)
    pore_volume = 200.0

    result = residence_time_retarded(flow=flow, aquifer_pore_volume=pore_volume, direction="infiltration")

    # With constant flow of 100 m³/day and pore volume of 200 m³,
    # residence time should be approximately 2 days
    assert np.isclose(result[0, 0], 2.0, rtol=0.1)


def test_retardation_factor():
    """Test the effect of retardation factor."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow = pd.Series(data=[100.0] * 5, index=dates)
    pore_volume = 200.0

    result_no_retardation = residence_time_retarded(
        flow=flow, aquifer_pore_volume=pore_volume, retardation_factor=1.0, direction="extraction"
    )

    result_with_retardation = residence_time_retarded(
        flow=flow, aquifer_pore_volume=pore_volume, retardation_factor=2.0, direction="extraction"
    )

    # Residence time should double with retardation factor of 2
    assert np.isclose(result_with_retardation[0, -1], 2 * result_no_retardation[0, -1], rtol=0.1)


def test_custom_index():
    """Test using custom index for results."""
    flow_dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow = pd.Series(data=[100.0] * 5, index=flow_dates)
    custom_dates = pd.date_range(start="2023-01-02", end="2023-01-04", freq="D")
    pore_volume = 200.0

    result = residence_time_retarded(
        flow=flow, aquifer_pore_volume=pore_volume, index=custom_dates, direction="extraction"
    )

    assert result.shape[1] == len(custom_dates)


def test_return_as_series():
    """Test returning results as pandas Series."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow = pd.Series(data=[100.0] * 5, index=dates)
    pore_volume = 200.0

    result = residence_time_retarded(
        flow=flow, aquifer_pore_volume=pore_volume, direction="extraction", return_as_series=True
    )

    assert isinstance(result, pd.Series)
    assert result.name == "residence_time_extraction"
    assert len(result) == len(dates)


def test_multiple_pore_volumes():
    """Test handling of multiple pore volumes."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow = pd.Series(data=[100.0] * 5, index=dates)
    pore_volumes = np.array([200.0, 300.0, 400.0])

    result = residence_time_retarded(flow=flow, aquifer_pore_volume=pore_volumes, direction="extraction")

    assert result.shape[0] == len(pore_volumes)
    assert result.shape[1] == len(dates)
    # Residence times should increase with increasing pore volumes
    assert np.all(np.diff(result[:, -1]) > 0)


def test_invalid_direction():
    """Test that invalid direction raises ValueError."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow = pd.Series(data=[100.0] * 5, index=dates)
    pore_volume = 200.0

    with pytest.raises(ValueError, match="direction should be 'extraction' or 'infiltration'"):
        residence_time_retarded(flow=flow, aquifer_pore_volume=pore_volume, direction="invalid")


def test_edge_cases(sample_flow_data):
    """Test edge cases such as zero flow and very large pore volumes."""
    # Test zero flow
    zero_flow = pd.Series(data=[0.0] * len(sample_flow_data), index=sample_flow_data.index)
    result_zero = residence_time_retarded(flow=zero_flow, aquifer_pore_volume=100.0, direction="extraction")
    assert np.all(np.isnan(result_zero))

    # Test very large pore volume
    result_large = residence_time_retarded(flow=sample_flow_data, aquifer_pore_volume=1e6, direction="extraction")
    assert np.all(np.isnan(result_large))


def test_negative_flow():
    """Test handling of negative flow values."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow = pd.Series(data=[-100.0] * 5, index=dates)
    pore_volume = 200.0

    result = residence_time_retarded(flow=flow, aquifer_pore_volume=pore_volume, direction="extraction")

    # Negative flow should result in NaN values
    assert np.all(np.isnan(result))


def test_flow_variations(sample_flow_data):
    """Test that residence times respond appropriately to flow variations."""
    pore_volume = 100.0

    result1 = residence_time_retarded(flow=sample_flow_data, aquifer_pore_volume=pore_volume, direction="extraction")
    result2 = residence_time_retarded(
        flow=sample_flow_data * 2, aquifer_pore_volume=pore_volume, direction="extraction"
    )

    # Residence times should half with double flow
    np.testing.assert_array_almost_equal(result1[1:], result2[1:] * 2)
