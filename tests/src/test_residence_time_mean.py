import numpy as np
import pandas as pd
import pytest

from gwtransport.residence_time import residence_time_mean


@pytest.fixture
def sample_flow_data():
    """Create sample flow data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.array([100.0, 110.0, 105.0, 95.0, 98.0, 102.0, 107.0, 103.0, 96.0])
    return flow_values, dates


@pytest.fixture
def constant_flow_data():
    """Create constant flow data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(dates) - 1, 100.0)
    return flow_values, dates


def test_basic_extraction(constant_flow_data):
    """Test basic extraction scenario with constant flow."""
    flow_values, flow_tedges = constant_flow_data
    tedges_out = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1D")
    pore_volume = 200.0

    result = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³,
    # residence time should be approximately 2 days
    # The first results should be NaN as not enough water has passed
    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])
    # Later values should be close to 2 days
    assert np.isclose(result[0, 2], 2.0, rtol=0.1)
    assert np.isclose(result[0, 3], 2.0, rtol=0.1)


def test_basic_infiltration(constant_flow_data):
    """Test basic infiltration scenario with constant flow."""
    flow_values, flow_tedges = constant_flow_data
    pore_volume = 200.0

    result = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="infiltration_to_extraction",
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³,
    # residence time should be approximately 2 days
    # The first values should be valid
    assert np.isclose(result[0, 0], 2.0, rtol=0.1)
    assert np.isclose(result[0, -3], 2.0, rtol=0.1)
    # Later values should be NaN as water hasn't been extracted yet
    assert np.isnan(result[0, -2])
    assert np.isnan(result[0, -1])


def test_varying_extraction(constant_flow_data):
    """Test basic extraction scenario with constant flow."""
    flow_values, flow_tedges = constant_flow_data
    flow_values[5:] *= 2.0  # Double the flow after the 5th day
    tedges_out_highres = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1h")
    tedges_out_lowres = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1D")
    pore_volume = 200.0

    result_highres = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out_highres,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    df_highres = pd.Series(result_highres[0], index=tedges_out_highres[:-1])
    df_lowres = df_highres.resample("1D").mean()
    result_lowres = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out_lowres,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Check that the mean values are consistent
    assert np.allclose(df_lowres.values, result_lowres[0], equal_nan=True)


def test_retardation_factor(constant_flow_data):
    """Test the effect of retardation factor."""
    flow_values, flow_tedges = constant_flow_data
    pore_volume = 100.0

    result_no_retardation = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=1.0,
        direction="extraction_to_infiltration",
    )

    result_with_retardation = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=2.0,
        direction="extraction_to_infiltration",
    )

    # Residence time should double with retardation factor of 2
    # We need to check in positions where both have valid values
    assert np.isnan(result_no_retardation[0, 0])
    assert np.isnan(result_with_retardation[0, 0])
    assert np.isnan(result_with_retardation[0, 1])

    # Later values should be valid
    assert np.isclose(result_with_retardation[0, 2], 2 * result_no_retardation[0, 1])
    assert np.isclose(result_with_retardation[0, 3], 2 * result_no_retardation[0, 2])


def test_multiple_pore_volumes(constant_flow_data):
    """Test handling of multiple pore volumes."""
    flow_values, flow_tedges = constant_flow_data
    pore_volumes = np.array([100.0, 200.0, 300.0])

    result = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    assert result.shape[0] == len(pore_volumes)
    assert result.shape[1] == len(flow_tedges) - 1

    # Check that NaN pattern follows pore volumes
    # More NaNs at the beginning for larger pore volumes
    valid_counts = np.sum(~np.isnan(result), axis=1)
    assert np.all(np.diff(valid_counts) <= 0)

    # Check values for the smallest pore volume where we should have valid results
    # Residence times should increase with increasing pore volumes
    assert np.all(np.diff(result[:, -1]) > 0)


def test_invalid_direction(constant_flow_data):
    """Test that invalid direction raises ValueError."""
    flow_values, flow_tedges = constant_flow_data
    tedges_out = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    pore_volume = 200.0

    with pytest.raises(
        ValueError, match="direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
    ):
        residence_time_mean(
            flow=flow_values,
            flow_tedges=flow_tedges,
            tedges_out=tedges_out,
            aquifer_pore_volumes=pore_volume,
            direction="invalid",
        )


def test_edge_cases(sample_flow_data):
    """Test edge cases such as zero flow and very large pore volumes."""
    flow_values, flow_tedges = sample_flow_data
    tedges_out = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")

    # Test zero flow
    zero_flow = np.zeros_like(flow_values)
    result_zero = residence_time_mean(
        flow=zero_flow,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=100.0,
        direction="extraction_to_infiltration",
    )
    assert np.all(np.isnan(result_zero))

    # Test very large pore volume
    result_large = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=1e6,
        direction="extraction_to_infiltration",
    )
    assert np.all(np.isnan(result_large))


def test_negative_flow(constant_flow_data):
    """Test handling of negative flow values."""
    _, flow_tedges = constant_flow_data
    flow_values = np.full(len(flow_tedges) - 1, -100.0)
    tedges_out = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    pore_volume = 200.0

    result = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Negative flow should result in NaN values
    assert np.all(np.isnan(result))


def test_flow_variations(sample_flow_data):
    """Test that residence times respond appropriately to flow variations."""
    flow_values, flow_tedges = sample_flow_data
    double_flow = flow_values * 2
    tedges_out = pd.date_range(start="2023-01-04", end="2023-01-09", freq="1D")
    pore_volume = 100.0

    result1 = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result2 = residence_time_mean(
        flow=double_flow,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Residence times should approximately halve with double flow
    # We need to find positions where both results have valid values
    valid_mask = ~np.isnan(result1[0]) & ~np.isnan(result2[0])
    if np.any(valid_mask):
        ratio = result1[0, valid_mask] / result2[0, valid_mask]
        # Allow for some numerical imprecision
        assert np.all(np.isclose(ratio, 2.0, rtol=0.2))


def test_output_tedges_alignment():
    """Test that results align correctly with output time edges."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)

    # Test with different output time edges
    tedges_out1 = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    tedges_out2 = pd.date_range(start="2023-01-02", end="2023-01-09", freq="1D")
    pore_volume = 100.0

    result1 = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out1,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result2 = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out2,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Check output shapes match the expected dimensions
    assert result1.shape[1] == len(tedges_out1) - 1
    assert result2.shape[1] == len(tedges_out2) - 1

    # With constant flow, residence time should be constant after initial NaNs
    valid_mask1 = ~np.isnan(result1[0])
    valid_mask2 = ~np.isnan(result2[0])

    if np.any(valid_mask1):
        assert np.allclose(result1[0, valid_mask1], result1[0, valid_mask1][0], rtol=0.01)

    if np.any(valid_mask2):
        assert np.allclose(result2[0, valid_mask2], result2[0, valid_mask2][0], rtol=0.01)


def test_example_from_docstring():
    """Test the example provided in the function's docstring."""
    flow_dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(flow_dates) - 1, 100.0)  # Constant flow of 100 m³/day
    pore_volume = 200.0

    mean_times = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_dates,
        tedges_out=flow_dates,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # The first values should be NaN (not enough water has passed)
    assert np.isnan(mean_times[0, 0])
    assert np.isnan(mean_times[0, 1])

    # Later values should be approximately 2 days
    assert np.isclose(mean_times[0, 2], 2.0, rtol=0.1)
    assert np.isclose(mean_times[0, 3], 2.0, rtol=0.1)
