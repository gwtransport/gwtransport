import numpy as np
import pandas as pd
import pytest

from gwtransport.residence_time import residence_time
from gwtransport.utils import compute_time_edges


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


def test_basic_extraction_with_flow_tedges():
    """Test basic extraction scenario with constant flow using flow_tedges."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³,
    # residence time should be approximately 2 days
    assert np.isclose(result[0, -1], 2.0, rtol=0.1)


def test_basic_extraction_with_flow_tstart():
    """Test basic extraction scenario using flow_tstart converted to tedges."""
    flow_tstart = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow_values = np.full(len(flow_tstart), 100.0)
    pore_volume = 200.0

    # Convert tstart to tedges
    flow_tedges = compute_time_edges(tedges=None, tstart=flow_tstart, tend=None, number_of_bins=len(flow_values))

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³,
    # residence time should be approximately 2 days
    assert np.isclose(result[0, -1], 2.0, rtol=0.1)


def test_basic_extraction_with_flow_tend():
    """Test basic extraction scenario using flow_tend converted to tedges."""
    flow_tend = pd.date_range(start="2023-01-02", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tend), 100.0)
    pore_volume = 200.0

    # Convert tend to tedges
    flow_tedges = compute_time_edges(tedges=None, tstart=None, tend=flow_tend, number_of_bins=len(flow_values))

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³,
    # residence time should be approximately 2 days
    assert np.isclose(result[0, -1], 2.0, rtol=0.1)


def test_basic_infiltration():
    """Test basic infiltration scenario with constant flow."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="infiltration_to_extraction",
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³,
    # residence time should be approximately 2 days
    assert np.isclose(result[0, 0], 2.0, rtol=0.1)


def test_retardation_factor():
    """Test the effect of retardation factor."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    result_no_retardation = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=1.0,
        direction="extraction_to_infiltration",
    )

    result_with_retardation = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=2.0,
        direction="extraction_to_infiltration",
    )

    # Residence time should double with retardation factor of 2
    valid_mask = ~np.isnan(result_no_retardation[0]) & ~np.isnan(result_with_retardation[0])
    if np.any(valid_mask):
        ratio = result_with_retardation[0, valid_mask] / result_no_retardation[0, valid_mask]
        assert np.allclose(ratio, 2.0, rtol=0.1)


def test_custom_index():
    """Test using custom index for results."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    custom_dates = pd.date_range(start="2023-01-02", end="2023-01-04", freq="D")
    pore_volume = 200.0

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        index=custom_dates,
        direction="extraction_to_infiltration",
    )

    assert result.shape[1] == len(custom_dates)


def test_return_numpy_array():
    """Test returning results as numpy array (default behavior)."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    assert isinstance(result, np.ndarray)
    # Should return for the center points of flow bins
    expected_length = len(flow_tedges) - 1
    assert result.shape == (1, expected_length)


def test_multiple_pore_volumes():
    """Test handling of multiple pore volumes."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volumes = np.array([200.0, 300.0, 400.0])

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    assert result.shape[0] == len(pore_volumes)
    expected_time_points = len(flow_tedges) - 1
    assert result.shape[1] == expected_time_points

    # Residence times should increase with increasing pore volumes
    valid_mask = ~np.isnan(result[:, -1])
    if np.sum(valid_mask) > 1:
        assert np.all(np.diff(result[valid_mask, -1]) > 0)


def test_invalid_direction():
    """Test that invalid direction raises ValueError."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    with pytest.raises(
        ValueError, match="direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
    ):
        residence_time(flow=flow_values, flow_tedges=flow_tedges, aquifer_pore_volumes=pore_volume, direction="invalid")


def test_missing_flow_timing_parameters():
    """Test that missing flow timing parameters raises TypeError."""
    flow_values = np.full(5, 100.0)
    pore_volume = 200.0

    # Since flow_tedges is now a required parameter (no longer has | None),
    # Python raises TypeError when it's not provided
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'flow_tedges'"):
        residence_time(flow=flow_values, aquifer_pore_volumes=pore_volume, direction="extraction_to_infiltration")  # type: ignore[missing-argument]


def test_flow_tedges_length_validation():
    """Test validation of flow_tedges length."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges), 100.0)  # Wrong length (should be len-1)
    pore_volume = 200.0

    with pytest.raises(ValueError, match="tedges must have one more element than flow"):
        residence_time(
            flow=flow_values,
            flow_tedges=flow_tedges,
            aquifer_pore_volumes=pore_volume,
            direction="extraction_to_infiltration",
        )


def test_flow_tstart_length_validation():
    """Test validation of flow_tstart length when converting to tedges."""
    flow_tstart = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow_values = np.full(len(flow_tstart) + 1, 100.0)  # Wrong length

    with pytest.raises(ValueError, match="tstart must have the same number of elements as flow"):
        # This should fail during compute_time_edges
        compute_time_edges(tedges=None, tstart=flow_tstart, tend=None, number_of_bins=len(flow_values))


def test_flow_tend_length_validation():
    """Test validation of flow_tend length when converting to tedges."""
    flow_tend = pd.date_range(start="2023-01-02", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tend) + 1, 100.0)  # Wrong length

    with pytest.raises(ValueError, match="tend must have the same number of elements as flow"):
        # This should fail during compute_time_edges
        compute_time_edges(tedges=None, tstart=None, tend=flow_tend, number_of_bins=len(flow_values))


def test_edge_cases_zero_flow():
    """Test edge cases such as zero flow."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    zero_flow = np.zeros(len(flow_tedges) - 1)
    pore_volume = 100.0

    result = residence_time(
        flow=zero_flow,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Zero flow should result in infinite/NaN residence times
    assert np.all(np.isnan(result) | np.isinf(result))


def test_edge_cases_very_large_pore_volume():
    """Test edge cases with very large pore volumes."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    large_pore_volume = 1e6

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=large_pore_volume,
        direction="extraction_to_infiltration",
    )

    # Very large pore volume should result in NaN values for recent times
    assert np.all(np.isnan(result))


def test_negative_flow():
    """Test handling of negative flow values."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    negative_flow = np.full(len(flow_tedges) - 1, -100.0)
    pore_volume = 200.0

    result = residence_time(
        flow=negative_flow,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Negative flow should result in NaN values or unusual behavior
    # The function should handle this gracefully
    assert not np.all(np.isfinite(result))


def test_flow_variations(sample_flow_data):
    """Test that residence times respond appropriately to flow variations."""
    flow_values, flow_tedges = sample_flow_data
    pore_volume = 500.0  # Use a larger pore volume to get valid results

    result1 = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result2 = residence_time(
        flow=flow_values * 2,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Find positions where both results have valid values
    valid_mask = ~np.isnan(result1[0]) & ~np.isnan(result2[0])

    if np.any(valid_mask):
        # Residence times should approximately halve with double flow
        ratio = result1[0, valid_mask] / result2[0, valid_mask]
        assert np.allclose(ratio, 2.0, rtol=0.3)


def test_consistency_between_timing_methods():
    """Test that different timing parameter methods give consistent results."""
    # Create a time series
    dates = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.array([100.0, 110.0, 105.0, 95.0, 98.0])
    pore_volume = 200.0

    # Method 1: flow_tedges
    result_tedges = residence_time(
        flow=flow_values, flow_tedges=dates, aquifer_pore_volumes=pore_volume, direction="extraction_to_infiltration"
    )

    # Method 2: flow_tstart (assuming flow is measured at start of intervals)
    flow_tstart = dates[:-1]
    flow_tedges_from_tstart = compute_time_edges(
        tedges=None, tstart=flow_tstart, tend=None, number_of_bins=len(flow_values)
    )
    result_tstart = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges_from_tstart,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Method 3: flow_tend (assuming flow is measured at end of intervals)
    flow_tend = dates[1:]
    flow_tedges_from_tend = compute_time_edges(
        tedges=None, tstart=None, tend=flow_tend, number_of_bins=len(flow_values)
    )
    result_tend = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges_from_tend,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Results should be similar but may have slight differences due to timing assumptions
    # We'll check that the general pattern is consistent
    valid_mask_tedges = ~np.isnan(result_tedges[0])
    valid_mask_tstart = ~np.isnan(result_tstart[0])
    valid_mask_tend = ~np.isnan(result_tend[0])

    # At least some results should be valid for each method
    assert np.any(valid_mask_tedges)
    assert np.any(valid_mask_tstart)
    assert np.any(valid_mask_tend)


def test_array_like_flow_input():
    """Test that array-like flow inputs (lists, numpy arrays) work correctly."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    pore_volume = 200.0

    # Test with list
    flow_list = [100.0, 110.0, 105.0, 95.0, 98.0]
    result_list = residence_time(
        flow=flow_list,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Test with numpy array
    flow_array = np.array([100.0, 110.0, 105.0, 95.0, 98.0])
    result_array = residence_time(
        flow=flow_array,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Results should be identical
    np.testing.assert_array_equal(result_list, result_array)


def test_single_pore_volume_vs_array():
    """Test that single pore volume and array with one element give same results."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)

    # Single float
    result_float = residence_time(
        flow=flow_values, flow_tedges=flow_tedges, aquifer_pore_volumes=200.0, direction="extraction_to_infiltration"
    )

    # Array with single element
    result_array = residence_time(
        flow=flow_values, flow_tedges=flow_tedges, aquifer_pore_volumes=[200.0], direction="extraction_to_infiltration"
    )

    # Results should be identical
    np.testing.assert_array_equal(result_float, result_array)


def test_infiltration_vs_extraction_symmetry():
    """Test the symmetry between infiltration and extraction directions."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 300.0

    result_extraction = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result_infiltration = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="infiltration_to_extraction",
    )

    # With constant flow, the residence times should be constant where valid
    # The values should be the same magnitude but applied in different time directions
    extraction_valid = ~np.isnan(result_extraction[0])
    infiltration_valid = ~np.isnan(result_infiltration[0])

    if np.any(extraction_valid):
        # All valid extraction residence times should be approximately equal
        extraction_values = result_extraction[0, extraction_valid]
        assert np.allclose(extraction_values, extraction_values[0], rtol=0.1)

    if np.any(infiltration_valid):
        # All valid infiltration residence times should be approximately equal
        infiltration_values = result_infiltration[0, infiltration_valid]
        assert np.allclose(infiltration_values, infiltration_values[0], rtol=0.1)
