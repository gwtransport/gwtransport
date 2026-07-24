import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests.exceptions
from _oracles import partial_isin  # ty: ignore[unresolved-import]  # tests/src on path via conftest
from scipy.linalg import null_space

from gwtransport._time import dt_to_days, tedges_to_days
from gwtransport.examples import generate_example_data, generate_example_deposition_timeseries
from gwtransport.utils import (
    _make_strictly_monotone,
    compute_reverse_target,
    compute_time_edges,
    cumulative_flow_volume,
    get_soil_temperature,
    linear_average,
    linear_interpolate,
    simplify_bins,
    solve_inverse_transport,
    solve_inverse_transport_banded,
    solve_tikhonov,
    solve_underdetermined_system,
    step_plot_coords,
    time_bin_overlap,
)


def test_linear_interpolate():
    # Test 1: Basic linear interpolation
    x_ref = np.array([0, 2, 4, 6, 8, 10])
    y_ref = np.array([0, 4, 8, 12, 16, 20])  # y = 2x
    x_query = np.array([1, 3, 5, 7, 9])
    expected = np.array([2, 6, 10, 14, 18])

    result = linear_interpolate(x_ref=x_ref, y_ref=y_ref, x_query=x_query)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-06)

    # Test 2: Single value interpolation
    x_ref = np.array([0, 1])
    y_ref = np.array([0, 1])
    x_query = np.array([0.5])
    expected = np.array([0.5])

    result = linear_interpolate(x_ref=x_ref, y_ref=y_ref, x_query=x_query)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-06)

    # Test 3: Edge cases - query points outside range
    x_ref = np.array([0, 1, 2])
    y_ref = np.array([0, 1, 2])
    x_query = np.array([-1, 3])  # Outside the range
    expected = np.array([0, 2])  # Should clip to nearest values

    result = linear_interpolate(x_ref=x_ref, y_ref=y_ref, x_query=x_query)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-06)

    # Test 4: Non-uniform spacing
    x_ref = np.array([0, 1, 10])
    y_ref = np.array([0, 2, 20])
    x_query = np.array([0.5, 5.5])
    expected = np.array([1, 11])

    result = linear_interpolate(x_ref=x_ref, y_ref=y_ref, x_query=x_query)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-06)

    # Test 5: Exact matches with reference points
    x_ref = np.array([0, 1, 2])
    y_ref = np.array([0, 10, 20])
    x_query = np.array([0, 1, 2])
    expected = np.array([0, 10, 20])

    result = linear_interpolate(x_ref=x_ref, y_ref=y_ref, x_query=x_query)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-06)


def test_make_strictly_monotone_no_duplicates_returns_input():
    """Already strictly monotone arrays are returned unchanged (no allocation needed)."""
    arr = np.array([0.0, 1.0, 2.5, 7.0])
    result = _make_strictly_monotone(arr)
    np.testing.assert_array_equal(result, arr)


def test_make_strictly_monotone_breaks_plateau_with_safety_factor_per_duplicate():
    """A run of k duplicates is bumped by ``k * BUMP * ulp(max(arr))`` where BUMP=16.

    The factor of 16 is a platform-robust margin against ``np.interp``'s FMA-related
    rounding noise. The exact value is an internal implementation detail; this test pins
    it down so any future change is intentional.
    """
    arr = np.array([0.0, 100.0, 100.0, 100.0, 200.0])
    ulp_max = np.nextafter(200.0, np.inf) - 200.0
    bump = 16 * ulp_max
    expected = np.array([0.0, 100.0, 100.0 + bump, 100.0 + 2 * bump, 200.0])
    result = _make_strictly_monotone(arr)
    np.testing.assert_array_equal(result, expected)
    assert np.all(np.diff(result) > 0)  # strictly monotone -- the whole point


def test_make_strictly_monotone_multiple_runs_reset_cumcount():
    """A non-duplicate breaks the run; the next duplicate starts again at 1 * bump."""
    arr = np.array([0.0, 5.0, 5.0, 5.0, 7.0, 7.0, 9.0])
    ulp_max = np.nextafter(9.0, np.inf) - 9.0
    bump = 16 * ulp_max
    expected = np.array([0.0, 5.0, 5.0 + bump, 5.0 + 2 * bump, 7.0, 7.0 + bump, 9.0])
    result = _make_strictly_monotone(arr)
    np.testing.assert_array_equal(result, expected)


def test_constant_function():
    """Test average of constant function y=2."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([2, 2, 2, 2, 2])
    x_edges = np.array([0, 2, 4])

    expected = np.array([[2, 2]])  # Average is constant, now 2D
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected)


def test_linear_function():
    """Test average of linear function y=x."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 2, 3, 4])
    x_edges = np.array([0, 2, 4])

    # Average of y=x from 0 to 2 = 1
    # Average of y=x from 2 to 4 = 3
    expected = np.array([[1, 3]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected)


def test_piecewise_linear():
    """Test average of piecewise linear function."""
    x_data = np.array([0, 1, 2, 3])
    y_data = np.array([0, 1, 1, 0])
    x_edges = np.array([0, 1.5, 3])

    # Integral from 0 to 1.5 = 1, width = 1.5 → average = 2/3
    # Integral from 1.5 to 3 = 1, width = 1.5 → average = 2/3
    expected = np.array([[1.0 / 1.5, 1.0 / 1.5]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_edges_beyond_data():
    """Test averages with edges outside the data range."""
    x_data = np.array([1, 2, 3])
    y_data = np.array([1, 2, 3])
    x_edges = np.array([0, 4])

    # Extrapolation should extend the first and last segments
    # Average of y=x from 0 to 4 = 2
    expected = np.array([[2]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges, extrapolate_method="outer")

    np.testing.assert_allclose(result, expected)


def test_edges_matching_data():
    """Test when edges exactly match data points."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 4, 9, 16])
    x_edges = np.array([1, 3])

    # Average under the curve from 1 to 3 = 4.5
    expected = np.array([[4.5]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected)


def test_multiple_edge_intervals():
    """Test with multiple averaging intervals."""
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0, 1, 4, 9, 16, 25])
    x_edges = np.array([0, 1, 2, 3, 4, 5])

    # Average of each segment
    expected = np.array([[0.5, 2.5, 6.5, 12.5, 20.5]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected)


def test_empty_interval():
    """Test averaging over an empty interval (edges are the same)."""
    x_data = np.array([0, 1, 2, 3])
    y_data = np.array([0, 1, 4, 9])
    x_edges = np.array([0, 1, 1, 2])

    # Second interval has zero width at x=1, so average should be y=1
    expected = np.array([[0.5, 1.0, 2.5]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected)


def test_input_validation():
    """Test input validation."""
    # Test unequal lengths of x_data and y_data
    with pytest.raises(ValueError, match="x_data and y_data must have the same length and be non-empty"):
        linear_average(x_data=[0, 1], y_data=[0], x_edges=[0, 1])

    # Test x_edges too short
    with pytest.raises(ValueError, match="x_edges must contain at least 2 values in each row"):
        linear_average(x_data=[0, 1], y_data=[0, 1], x_edges=[0])

    # Test x_data not in ascending order
    with pytest.raises(ValueError, match="x_data must be in ascending order"):
        linear_average(x_data=[1, 0], y_data=[0, 1], x_edges=[0, 1])

    # Test x_edges not in ascending order
    with pytest.raises(ValueError, match="x_edges must be in ascending order"):
        linear_average(x_data=[0, 1], y_data=[0, 1], x_edges=[1, 0])


def test_complex_piecewise_function():
    """Test a more complex piecewise linear function."""
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0, 2, 1, 3, 0, 2])
    x_edges = np.array([0.5, 2.5, 4.5])

    # First interval: integral = 3.0, width = 2.0 → average = 1.5
    # Second interval: integral = 3.0, width = 2.0 → average = 1.5
    expected = np.array([[1.5, 1.5]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected)


def test_edge_case_numerical_precision():
    """Test numerical precision for very close x values."""
    x_data = np.array([0, 1e-10, 1])
    y_data = np.array([0, 1e-10, 1])
    x_edges = np.array([0, 0.5, 1])

    # For a linear function y=x, the average from 0 to 0.5 is 0.25
    # and from 0.5 to 1 is 0.75
    expected = np.array([[0.25, 0.75]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_single_point_data():
    """Test with a single data point - should extrapolate as constant."""
    x_data = np.array([1])
    y_data = np.array([5])
    x_edges = np.array([0, 2])

    # Single point should be treated as constant value
    expected = np.array([[5]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges, extrapolate_method="outer")

    np.testing.assert_allclose(result, expected)


def test_zero_width_interval_edge_case():
    """Test handling of a zero-width interval at the edge."""
    x_data = np.array([0, 1, 2])
    y_data = np.array([0, 1, 2])
    x_edges = np.array([0, 0, 1])

    # First interval has zero width at x=0, so average should be y=0
    # Second interval is 0 to 1, average is 0.5
    expected = np.array([[0.0, 0.5]])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges)

    np.testing.assert_allclose(result, expected)


def test_2d_x_edges():
    """Test 2D x_edges functionality."""
    x_data = np.array([0, 1, 2, 3])
    y_data = np.array([0, 1, 1, 0])

    # 2D x_edges: two different edge sets
    x_edges_2d = np.array([
        [0, 1.5, 3],  # First set of edges
        [0.5, 2, 3],  # Second set of edges
    ])

    # Expected results for each row
    expected = np.array([
        [2 / 3, 2 / 3],  # First row results
        [11 / 12, 0.5],  # Second row results (0.916667, 0.5)
    ])

    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges_2d)

    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_linear_average_2d_y_data_basic():
    """2D y_data computes the per-row average independently and matches looping."""
    x_data = np.array([0.0, 1.0, 2.0, 3.0])
    y_data_2d = np.array([
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 2.0, 2.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    x_edges = np.array([0.0, 1.5, 3.0])

    result = linear_average(x_data=x_data, y_data=y_data_2d, x_edges=x_edges)

    # Compare to a loop over rows -- each row averaged independently.
    expected = np.vstack([
        linear_average(x_data=x_data, y_data=y_data_2d[i], x_edges=x_edges)[0] for i in range(y_data_2d.shape[0])
    ])
    np.testing.assert_allclose(result, expected, rtol=1e-12)
    # Sanity check: row 2 is constant 1, so its average over any interval is 1.
    np.testing.assert_allclose(result[2], 1.0, rtol=1e-12)


def test_linear_average_2d_y_data_per_row_nan():
    """2D y_data: a NaN in one row only marks bins of that row, not others."""
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data_2d = np.array([
        [0.0, 1.0, 2.0, 3.0, 4.0],  # ramp y = x
        [0.0, 1.0, np.nan, 3.0, 4.0],  # NaN at x=2 in row 1 only
    ])
    x_edges = np.array([0.0, 1.0, 3.0, 4.0])

    result = linear_average(x_data=x_data, y_data=y_data_2d, x_edges=x_edges)

    # Row 0: y = x clean. Means over [0,1], [1,3], [3,4] are 0.5, 2.0, 3.5.
    np.testing.assert_allclose(result[0], [0.5, 2.0, 3.5], rtol=1e-12)
    # Row 1: bin [0, 1] does not touch any NaN segment -> finite. Bin [1, 3] does -> NaN.
    # Bin [3, 4] does not -> finite (the NaN segment is [1,2] U [2,3]).
    np.testing.assert_allclose(result[1, 0], 0.5, rtol=1e-12)
    assert np.isnan(result[1, 1])
    np.testing.assert_allclose(result[1, 2], 3.5, rtol=1e-12)


def test_linear_average_2d_y_data_with_2d_x_edges_raises():
    """Combining 2D y_data with 2D x_edges is intentionally unsupported."""
    x_data = np.array([0.0, 1.0, 2.0])
    y_data_2d = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    x_edges_2d = np.array([[0.0, 1.0, 2.0], [0.5, 1.0, 1.5]])
    with pytest.raises(ValueError, match="Cannot combine 2D x_edges with 2D y_data"):
        linear_average(x_data=x_data, y_data=y_data_2d, x_edges=x_edges_2d)


def test_linear_average_straddling_bin_is_nan():
    """Bins partially outside the data range must be NaN, not biased low.

    With the previous implementation, a straddling bin's integral covered only
    the in-range portion while being divided by the full bin width, biasing the
    result low. The fix sets such bins to NaN, the same as bins fully outside
    the range.
    """
    # Linear ramp y = x on [0, 4]; mean over [-1, 4] would equal 4/2 = 2 but
    # only the in-range portion (over [0, 4]) is integrable; the result must
    # therefore be NaN, not the buggy 8/5 = 1.6.
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    x_edges = np.array([-1.0, 4.0])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges, extrapolate_method="nan")
    assert np.isnan(result).all()

    # Same on the right side.
    x_edges = np.array([0.0, 5.0])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges, extrapolate_method="nan")
    assert np.isnan(result).all()

    # Mixed: one fully-inside bin, one straddling-left bin, one straddling-right bin.
    x_edges = np.array([-1.0, 1.0, 3.0, 5.0])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges, extrapolate_method="nan")
    assert np.isnan(result[0, 0])
    np.testing.assert_allclose(result[0, 1], 2.0, rtol=1e-12)  # mean of y=x over [1, 3] = 2
    assert np.isnan(result[0, 2])


def test_basic_case():
    """Test the basic case with new interface."""
    bin_edges_in = np.array([0, 10, 20, 30])
    bin_edges_out = np.array([5, 15, 25])
    expected = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_no_overlap():
    """Test when there is no overlap between input and output bins."""
    bin_edges_in = np.array([0, 10, 20])
    bin_edges_out = np.array([30, 40, 50])
    expected = np.array([[0.0, 0.0], [0.0, 0.0]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_complete_overlap():
    """Test when input bins completely overlap with output bins."""
    bin_edges_in = np.array([0, 10, 20])
    bin_edges_out = np.array([0, 10, 20])
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_exact_bin_match():
    """Test when input and output bins exactly match."""
    bin_edges_in = np.array([10, 20, 30])
    bin_edges_out = np.array([10, 20, 30])
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_multiple_bins():
    """Test with multiple bins of different sizes."""
    bin_edges_in = np.array([0, 10, 20, 30])
    bin_edges_out = np.array([5, 12, 18, 25])
    expected = np.array([[0.5, 0.0, 0.0], [0.2, 0.6, 0.2], [0.0, 0.0, 0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_partial_overlaps():
    """Test various partial overlaps."""
    bin_edges_in = np.array([0, 20])  # One large input bin
    bin_edges_out = np.array([5, 10, 15])  # Two smaller output bins
    expected = np.array([[0.25, 0.25]])  # 25% overlap with each output bin

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_list_inputs():
    """Test with list inputs instead of numpy arrays."""
    bin_edges_in = [0, 10, 20, 30]
    bin_edges_out = [5, 15, 25]
    expected = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_empty_inputs():
    """Test with minimal valid inputs."""
    bin_edges_in = np.array([0, 10])
    bin_edges_out = np.array([5, 15])
    expected = np.array([[0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_single_bin():
    """Test with single input and output bins."""
    bin_edges_in = np.array([0, 10])
    bin_edges_out = np.array([5, 15])
    expected = np.array([[0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_edge_alignment():
    """Test when edges are perfectly aligned."""
    bin_edges_in = np.array([0, 10, 20, 30])
    bin_edges_out = np.array([0, 15, 30])
    expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_floating_point_precision():
    """Test with floating point values."""
    bin_edges_in = np.array([0.1, 0.3, 0.5])
    bin_edges_out = np.array([0.2, 0.4])
    expected = np.array([[0.5], [0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_negative_values():
    """Test with negative values."""
    bin_edges_in = np.array([-30, -20, -10])
    bin_edges_out = np.array([-25, -15, -5])
    expected = np.array([[0.5, 0.0], [0.5, 0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_array_equal(result, expected)


def test_invalid_inputs():
    """Test with invalid inputs for new interface."""
    # Test with unsorted input edges
    bin_edges_in = np.array([30, 20, 10])  # Descending order
    bin_edges_out = np.array([5, 15, 25])
    with pytest.raises(ValueError, match="bin_edges_in must be non-decreasing"):
        partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)

    # Test with unsorted output edges
    bin_edges_in = np.array([0, 10, 20])
    bin_edges_out = np.array([25, 15, 5])  # Descending order
    with pytest.raises(ValueError, match="bin_edges_out must be non-decreasing"):
        partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)


def test_get_soil_temperature_basic():
    """Test basic functionality of get_soil_temperature."""
    df = get_soil_temperature(station_number=260)

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) > 0
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing

    # Check expected columns
    expected_columns = {"TB1", "TB2", "TB3", "TB4", "TB5", "TNB1", "TNB2", "TXB1", "TXB2"}
    assert expected_columns.issubset(set(df.columns))

    # Check data quality
    for col in expected_columns:
        assert pd.api.types.is_numeric_dtype(df[col])
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            assert -50 <= valid_data.min() <= valid_data.max() <= 50


@pytest.mark.parametrize("station", [260, 273, 286, 323])
def test_get_soil_temperature_valid_stations(station):
    """Test get_soil_temperature works for all valid stations."""
    df = get_soil_temperature(station_number=station)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_get_soil_temperature_interpolation():
    """Test interpolation parameter affects results."""
    df_interp = get_soil_temperature(station_number=260, interpolate_missing_values=True)
    df_no_interp = get_soil_temperature(station_number=260, interpolate_missing_values=False)

    assert not df_interp.equals(df_no_interp)
    # Interpolated version should have fewer NaN values
    assert df_interp.isna().sum().sum() <= df_no_interp.isna().sum().sum()


def test_get_soil_temperature_invalid_station():
    """Test invalid station numbers raise errors."""
    with pytest.raises((requests.exceptions.HTTPError, ValueError)):
        get_soil_temperature(station_number=999)


def test_get_soil_temperature_cache():
    """Test caching functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_date = date(2023, 10, 15)

        with (
            patch("gwtransport.utils.cache_dir", Path(temp_dir) / "cache"),
            patch("gwtransport.utils.date") as mock_date_module,
        ):
            mock_date_module.today.return_value = mock_date

            # First call creates cache
            df1 = get_soil_temperature(station_number=260)
            cache_path = Path(temp_dir) / "cache" / f"soil_temp_260_True_{mock_date.isoformat()}.pkl"
            assert cache_path.exists()

            # Second call uses cache
            df2 = get_soil_temperature(station_number=260)
            pd.testing.assert_frame_equal(df1, df2)

            # Different parameters create separate cache
            df3 = get_soil_temperature(station_number=260, interpolate_missing_values=False)
            cache_path_false = Path(temp_dir) / "cache" / f"soil_temp_260_False_{mock_date.isoformat()}.pkl"
            assert cache_path_false.exists()
            assert not df3.equals(df1)


def test_time_bin_overlap_basic():
    """Test basic functionality of time_bin_overlap."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(5, 15), (25, 35)]
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_no_overlap():
    """Test when time ranges don't overlap with any bins."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(40, 50), (60, 70)]
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_complete_overlap():
    """Test when time ranges completely overlap with bins."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(0, 10), (10, 20), (20, 30)]
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_partial_overlap():
    """Test partial overlaps with multiple bins."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(5, 25)]
    expected = np.array([[0.5, 1.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_single_bin():
    """Test with single time bin."""
    tedges = np.array([0, 10])
    bin_tedges = [(5, 15), (-5, 5)]
    expected = np.array([[0.5], [0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_edge_alignment():
    """Test when time ranges align perfectly with bin edges."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(0, 20), (10, 30)]
    expected = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_floating_point():
    """Test with floating point values."""
    tedges = np.array([0.1, 0.3, 0.5, 0.7])
    bin_tedges = [(0.2, 0.4), (0.6, 0.8)]
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_negative_values():
    """Test with negative time values."""
    tedges = np.array([-30, -20, -10, 0])
    bin_tedges = [(-25, -15), (-5, 5)]
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_overlapping_ranges():
    """Test with overlapping time ranges."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(5, 15), (10, 25)]
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_zero_width_bins():
    """Test with zero-width time bins."""
    tedges = np.array([0, 10, 10, 20])
    bin_tedges = [(5, 15)]
    expected = np.array([[0.5, 0.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_large_range():
    """Test with a large time range covering all bins."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(-10, 40)]
    expected = np.array([[1.0, 1.0, 1.0]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_list_inputs():
    """Test with list inputs instead of numpy arrays."""
    tedges = [0, 10, 20, 30]
    bin_tedges = [(5, 15), (25, 35)]
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_array_equal(result, expected)


def test_time_bin_overlap_input_validation():
    """Test input validation for time_bin_overlap."""
    # Test with non-1D tedges
    with pytest.raises(ValueError, match="tedges must be a 1D array"):
        time_bin_overlap(tedges=np.array([[0, 10], [20, 30]]), bin_tedges=[(5, 15)])

    # Test with too few tedges
    with pytest.raises(ValueError, match="tedges must have at least 2 elements"):
        time_bin_overlap(tedges=np.array([0]), bin_tedges=[(5, 15)])

    # Test with empty bin_tedges
    with pytest.raises(ValueError, match="bin_tedges must be non-empty"):
        time_bin_overlap(tedges=np.array([0, 10, 20]), bin_tedges=[])


def test_time_bin_overlap_precision():
    """Test numerical precision with very small overlaps."""
    tedges = np.array([0, 1e-10, 1])
    bin_tedges = [(0.5, 1.5)]
    expected = np.array([[0.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-10)


def test_time_bin_overlap_many_ranges():
    """Test with many time ranges and bins for performance."""
    tedges = np.linspace(0, 100, 101)  # 100 bins
    bin_tedges = [(i, i + 10) for i in range(0, 90, 5)]  # 18 overlapping ranges

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)

    # Check output shape
    assert result.shape == (18, 100)

    # Check that each range has correct total overlap
    for i in range(len(bin_tedges)):
        # Each range spans 10 units, so total overlap should be 10
        total_overlap = np.sum(result[i, :])
        np.testing.assert_array_equal([total_overlap], [10.0])


def test_time_bin_overlap_boundary_cases():
    """Test boundary cases and edge conditions."""
    tedges = np.array([0, 5, 10, 15, 20])

    # Range that touches bin boundary but doesn't overlap
    bin_tedges = [(10, 10)]
    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    expected = np.array([[0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_array_equal(result, expected)

    # Range that exactly matches a bin
    bin_tedges = [(5, 10)]
    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    expected = np.array([[0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Tests for solve_tikhonov resolution diagnostics
# =============================================================================


def test_solve_tikhonov_resolution_well_determined():
    """Well-determined system: fraction_data should be close to 1 everywhere."""
    # Overdetermined 4x3 system with good conditioning
    coeff = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    x_true = np.array([1.0, 2.0, 3.0])
    rhs = coeff @ x_true
    x_target = np.zeros(3)

    x, fraction_data = solve_tikhonov(
        coefficient_matrix=coeff,
        rhs_vector=rhs,
        x_target=x_target,
        regularization_strength=1e-10,
        return_resolution=True,
    )

    np.testing.assert_allclose(x, x_true, atol=1e-6)
    np.testing.assert_allclose(fraction_data, 1.0, rtol=0, atol=1e-9)


def test_solve_tikhonov_resolution_underdetermined():
    """Underdetermined system: some fraction_data values should be < 1."""
    # 2 equations, 4 unknowns
    coeff = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    rhs = np.array([3.0, 7.0])
    x_target = np.array([1.0, 2.0, 3.0, 4.0])

    _x, fraction_data = solve_tikhonov(
        coefficient_matrix=coeff,
        rhs_vector=rhs,
        x_target=x_target,
        regularization_strength=0.1,
        return_resolution=True,
    )

    # fraction_data should be in [0, 1]
    assert np.all(fraction_data >= -1e-10), f"fraction_data has values < 0: {fraction_data}"
    assert np.all(fraction_data <= 1.0 + 1e-10), f"fraction_data has values > 1: {fraction_data}"
    # Some elements should be pulled toward target (fraction_data < 1)
    assert np.any(fraction_data < 0.99), "Expected some target-driven elements in underdetermined system"
    # Pin the resolution scale. For this symmetric block system the model resolution
    # diagonal is analytically 10/21: each block has inverse-gram diagonal
    # (G^T G + 0.1 I)^-1[j, j] = 1.1/0.21, so
    # R[j, j] = 1 - 0.1 * (1.1/0.21) = 0.10/0.21 = 10/21. A 0.5 or 0.3 rescaling of the
    # regularization coefficient (R = 1 - c*lambda*d*G_inv) would shift this value.
    np.testing.assert_allclose(fraction_data, 10 / 21, rtol=1e-12)


def test_solve_tikhonov_resolution_bounds():
    """fraction_data should always be in [0, 1]."""
    rng = np.random.default_rng(42)
    coeff = rng.standard_normal((3, 6))
    x_true = rng.standard_normal(6)
    rhs = coeff @ x_true
    x_target = rng.standard_normal(6)

    for lam in [1e-12, 1e-6, 1e-2, 1.0, 100.0]:
        _, fraction_data = solve_tikhonov(
            coefficient_matrix=coeff,
            rhs_vector=rhs,
            x_target=x_target,
            regularization_strength=lam,
            return_resolution=True,
        )
        assert np.all(fraction_data >= -1e-10), f"λ={lam}: fraction_data has values < 0"
        assert np.all(fraction_data <= 1.0 + 1e-10), f"λ={lam}: fraction_data has values > 1"


def test_solve_tikhonov_resolution_nan_target():
    """NaN entries in x_target should have fraction_data = 1.0."""
    coeff = np.eye(3, dtype=float)
    rhs = np.array([1.0, 2.0, 3.0])
    x_target = np.array([0.0, np.nan, 0.0])

    _x, fraction_data = solve_tikhonov(
        coefficient_matrix=coeff,
        rhs_vector=rhs,
        x_target=x_target,
        regularization_strength=0.1,
        return_resolution=True,
    )

    # Entry with NaN target is not regularized → fully data-driven
    assert fraction_data[1] == pytest.approx(1.0), (
        f"NaN target entry should have fraction_data=1.0, got {fraction_data[1]}"
    )
    # Other entries are regularized
    assert fraction_data[0] < 1.0
    assert fraction_data[2] < 1.0


def test_solve_tikhonov_resolution_not_returned_by_default():
    """Without return_resolution, only x is returned (not a tuple)."""
    coeff = np.eye(3, dtype=float)
    rhs = np.array([1.0, 2.0, 3.0])
    x_target = np.zeros(3)

    result = solve_tikhonov(
        coefficient_matrix=coeff,
        rhs_vector=rhs,
        x_target=x_target,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


# =============================================================================
# Tests for examples.generate_example_data rng reproducibility (U3)
# =============================================================================


def test_generate_example_data_seed_reproducibility():
    """Two calls with the same integer seed must produce identical data.

    Covers the rng plumbing through generate_example_data: flow noise,
    spill placement, and measurement noise all must be sourced from the
    user-supplied generator.
    """
    df_a, tedges_a = generate_example_data(
        date_start="2020-01-01",
        date_end="2020-06-30",
        cin_method="constant",
        rng=12345,
    )
    df_b, tedges_b = generate_example_data(
        date_start="2020-01-01",
        date_end="2020-06-30",
        cin_method="constant",
        rng=12345,
    )
    np.testing.assert_array_equal(df_a["flow"].to_numpy(), df_b["flow"].to_numpy())
    np.testing.assert_array_equal(df_a["cin"].to_numpy(), df_b["cin"].to_numpy())
    np.testing.assert_array_equal(df_a["cout"].to_numpy(), df_b["cout"].to_numpy())
    np.testing.assert_array_equal(np.asarray(tedges_a), np.asarray(tedges_b))


def test_generate_example_data_seed_changes_output():
    """Different seeds must produce different stochastic series."""
    df_a, _ = generate_example_data(
        date_start="2020-01-01",
        date_end="2020-06-30",
        cin_method="constant",
        rng=1,
    )
    df_b, _ = generate_example_data(
        date_start="2020-01-01",
        date_end="2020-06-30",
        cin_method="constant",
        rng=2,
    )
    assert not np.array_equal(df_a["flow"].to_numpy(), df_b["flow"].to_numpy())


@pytest.mark.parametrize("seed", list(range(20)))
def test_generate_example_data_flow_floor_enforced(seed):
    """Flow floor of 5 m³/day must hold even after the spill loop divides flow.

    Regression test for issue 167: the floor was applied before spills, so
    spill divisions could push flow well below 5 m³/day (observed min ~0.3).
    """
    df, _ = generate_example_data(rng=seed)
    assert df["flow"].min() >= 5.0, f"flow min {df['flow'].min()} below 5.0 m³/day for seed={seed}"


# =============================================================================
# Tests for examples.generate_example_deposition_timeseries
# =============================================================================


def test_generate_example_deposition_timeseries_default_call():
    """Calling with no arguments must work end-to-end.

    Regression test for issue 167: the default event_dates list was tz-naive
    while the dates index was tz-aware (UTC), causing
    ``TypeError: Cannot compare dtypes datetime64[us, UTC] and datetime64[us]``
    inside ``get_indexer``.
    """
    series, tedges = generate_example_deposition_timeseries()
    assert len(series) == len(tedges) - 1
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.tz is not None
    assert series.notna().all()


def test_generate_example_deposition_timeseries_naive_event_dates():
    """Tz-naive user event_dates must be accepted and aligned to the UTC index."""
    series, _ = generate_example_deposition_timeseries(
        date_start="2020-01-01",
        date_end="2020-12-31",
        event_dates=["2020-06-15", "2020-09-01"],
        rng=0,
    )
    assert isinstance(series.index, pd.DatetimeIndex)
    assert series.index.tz is not None


def test_generate_example_deposition_timeseries_tz_aware_event_dates():
    """Tz-aware event_dates in a non-UTC zone must be converted, not rejected."""
    event_dates = pd.DatetimeIndex(["2020-06-15 02:00"]).tz_localize("Europe/Amsterdam")
    series, _ = generate_example_deposition_timeseries(
        date_start="2020-01-01",
        date_end="2020-12-31",
        event_dates=event_dates,
        rng=0,
    )
    assert series.notna().all()


def test_generate_example_deposition_timeseries_seed_reproducibility():
    """Two calls with the same integer seed must produce identical series."""
    s_a, _ = generate_example_deposition_timeseries(rng=12345)
    s_b, _ = generate_example_deposition_timeseries(rng=12345)
    np.testing.assert_array_equal(s_a.to_numpy(), s_b.to_numpy())


def test_generate_example_deposition_timeseries_seed_changes_output():
    """Different seeds must produce different stochastic series."""
    s_a, _ = generate_example_deposition_timeseries(rng=1)
    s_b, _ = generate_example_deposition_timeseries(rng=2)
    assert not np.array_equal(s_a.to_numpy(), s_b.to_numpy())


# =============================================================================
# Tests for solve_underdetermined_system nullspace regularization
# =============================================================================


def test_solve_underdetermined_system_minimizes_squared_differences():
    """The ``squared_differences`` solution must be the true constrained minimizer.

    For an underdetermined system the solution is the affine set
    ``{x_ls + N c}`` where ``N`` spans the nullspace. The
    ``squared_differences`` objective ``Σ (x[i+1] - x[i])^2`` is convex, so its
    unique minimizer over that affine set is characterized by the first-order
    optimality condition ``N^T D^T D x = 0`` (the objective gradient projected
    onto the nullspace vanishes). A solver that merely returned *some* feasible
    point — the prior "finite and reasonable" smoke check — would not satisfy
    this. We assert (1) feasibility ``A x = b``, (2) nullspace-projected
    stationarity, and (3) that no nullspace perturbation lowers the objective.
    """
    matrix = np.array([[1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 1.0]])
    rhs = np.array([5.0, 4.0])

    result = solve_underdetermined_system(
        coefficient_matrix=matrix, rhs_vector=rhs, nullspace_objective="squared_differences"
    )

    # (1) Feasibility: the solution must satisfy the original equations.
    np.testing.assert_allclose(matrix @ result, rhs, atol=1e-12)

    # (2) First-order optimality: D maps x to its adjacent differences, so the
    # objective is ||D x||^2 and its gradient is 2 D^T D x. At the constrained
    # minimum this gradient is orthogonal to the feasible directions (the
    # nullspace of the matrix), i.e. its projection onto N is zero.
    diff_op = np.diff(np.eye(result.size), axis=0)
    nullspace = null_space(matrix)
    projected_gradient = nullspace.T @ (diff_op.T @ diff_op @ result)
    np.testing.assert_allclose(projected_gradient, 0.0, atol=1e-12)

    def objective(x):
        return float(np.sum(np.diff(x) ** 2))

    obj_result = objective(result)

    # (3) Global check on the convex objective: every nullspace perturbation
    # must have an objective at least as large as the returned solution, and
    # the bare least-squares point (no smoothing) must be strictly worse.
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((nullspace.shape[1], 500))
    candidates = result[:, None] + nullspace @ coeffs
    candidate_objectives = np.sum(np.diff(candidates, axis=0) ** 2, axis=0)
    assert np.all(candidate_objectives >= obj_result - 1e-9)

    x_ls = np.linalg.pinv(matrix) @ rhs
    assert obj_result < objective(x_ls)


# ---------------------------------------------------------------------------
# #186 guard: cumulative_flow_volume reproduces the consolidated inline idiom
# ---------------------------------------------------------------------------


def test_cumulative_flow_volume_matches_inline_form():
    """The helper must reproduce ``concatenate([0], cumsum(flow*dt))`` bit-for-bit.

    Uses non-uniform bin widths so the Δt-weighting (not just a bare cumsum) is
    exercised: dropping the ``dt_days`` factor would change every value.
    """
    flow = np.array([2.0, 0.0, 5.0, 3.5, 1.25])
    dt_days = np.array([1.0, 2.0, 0.5, 4.0, 1.0])
    expected = np.concatenate(([0.0], np.cumsum(flow * dt_days)))
    result = cumulative_flow_volume(flow, dt_days)
    np.testing.assert_array_equal(result, expected)
    # Leading zero and one value per edge (n+1 for n bins).
    assert result[0] == 0.0
    assert result.shape == (flow.size + 1,)


def test_cumulative_flow_volume_strictly_monotone_matches_helper():
    """``strictly_monotone=True`` must equal ``_make_strictly_monotone`` of the inline form.

    The flow contains a ``Q = 0`` bin, so the inline cumulative volume has a
    plateau; the helper must bump it identically to the standalone routine.
    """
    flow = np.array([3.0, 0.0, 0.0, 4.0, 1.0])
    dt_days = np.array([1.0, 2.0, 1.5, 0.5, 3.0])
    inline = np.concatenate(([0.0], np.cumsum(flow * dt_days)))
    expected = _make_strictly_monotone(inline)
    result = cumulative_flow_volume(flow, dt_days, strictly_monotone=True)
    np.testing.assert_array_equal(result, expected)
    # The plateau-bumping must actually have changed the array (otherwise the
    # monotone branch is a no-op and the guard is vacuous).
    assert not np.array_equal(inline, expected)


# ---------------------------------------------------------------------------
# #185 guard: tedges_to_days / dt_to_days are the single conversion idiom
# ---------------------------------------------------------------------------


def test_tedges_to_days_matches_reference_days():
    """Calling the helper must yield the exact float64 days-since-ref array.

    This is the #185 contract: the helper (not a re-spelling of pandas) maps
    bin edges to days relative to the first edge.
    """
    tedges = pd.DatetimeIndex([
        "2020-01-01",
        "2020-01-02",
        "2020-01-04",
        "2020-02-01",
        "2021-01-01",
    ])
    # Days from 2020-01-01: 0, 1, 3, 31, 366 (2020 is a leap year).
    expected = np.array([0.0, 1.0, 3.0, 31.0, 366.0])
    result = tedges_to_days(tedges)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64


def test_tedges_to_days_ref_kwarg_shifts_origin():
    """A shared ``ref`` re-bases the conversion; default ref is ``tedges[0]``."""
    tedges = pd.DatetimeIndex(["2020-01-10", "2020-01-12", "2020-01-15"])
    ref = pd.Timestamp("2020-01-01")
    np.testing.assert_array_equal(tedges_to_days(tedges, ref=ref), np.array([9.0, 11.0, 14.0]))
    # ref=None must equal ref=tedges[0].
    np.testing.assert_array_equal(tedges_to_days(tedges), tedges_to_days(tedges, ref=tedges[0]))


def test_tedges_to_days_cross_array_ref_aligns_origins():
    """A second edge array converted with ``ref=A[0]`` shares A's origin (the cin/cout pairing)."""
    a = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
    b = pd.DatetimeIndex(["2020-01-05", "2020-01-09"])
    # b measured against a's origin: 4 and 8 days after a[0].
    np.testing.assert_array_equal(tedges_to_days(b, ref=a[0]), np.array([4.0, 8.0]))


def test_tedges_to_days_dtype_contract_snapshot():
    """The five pre-change spellings must all equal the helper bit-for-bit.

    A divisor/origin regression in the helper diverges from these independent spellings (a
    days=1 -> days=2 swap is caught). The float64 cast in the helper is a no-op on modern
    pandas but is a forward guard for the minimum-deps CI leg, where the timedelta quotient
    could surface as object dtype (#185).
    """
    tedges = pd.DatetimeIndex(["2019-06-01", "2019-06-02", "2019-06-05", "2019-07-01"])
    ref = tedges[0]
    delta = (tedges - ref) / pd.Timedelta(days=1)
    spellings = {
        "values": delta.values,
        "to_numpy(dtype=float)": delta.to_numpy(dtype=float),
        "astype(float)": delta.astype(float),
        "values.astype(float)": delta.values.astype(float),
        "asarray/timedelta64": np.asarray((tedges - ref) / np.timedelta64(1, "D")),
    }
    helper = tedges_to_days(tedges)
    for name, arr in spellings.items():
        np.testing.assert_allclose(helper, arr, rtol=0, atol=0, err_msg=name)


def test_tedges_to_days_timezone_invariance():
    """tz-naive, UTC, and non-UTC edges give bit-identical day arrays on a no-DST span.

    The helper measures *absolute elapsed* days (instant differences over Timedelta), so it is
    origin-relative but NOT wall-clock/tz-invariant across a DST boundary. This guard locks the
    narrower realistic property: for a span with no DST transition (November) the local offset is
    constant, so tz-naive, UTC, and the package's UTC-aware indices give identical elapsed days.
    (It does not catch a ``tz_localize(None)`` slip, which preserves the elapsed-day spacing.)
    """
    naive = pd.DatetimeIndex(["2020-11-02", "2020-11-03", "2020-11-06", "2020-12-01"])
    utc = naive.tz_localize("UTC")
    other = naive.tz_localize("Europe/Amsterdam")
    base = tedges_to_days(naive)
    np.testing.assert_array_equal(tedges_to_days(utc), base)
    np.testing.assert_array_equal(tedges_to_days(other), base)


def test_dt_to_days_matches_reference_widths():
    """``dt_to_days`` must return the exact float64 bin widths in days."""
    tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04", "2020-02-01"])
    expected = np.array([1.0, 2.0, 28.0])  # widths in days (Jan -> Feb 1 = 28 days)
    result = dt_to_days(tedges)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64
    # Consistency with tedges_to_days: widths are the successive differences.
    np.testing.assert_array_equal(result, np.diff(tedges_to_days(tedges)))


# ---------------------------------------------------------------------------
# step_plot_coords
# ---------------------------------------------------------------------------


def test_step_plot_coords_single_bin():
    """n=1: one bin doubles each edge-pair and repeats the single value."""
    edges = np.array([0.0, 2.0])
    values = np.array([7.0])
    x, y = step_plot_coords(edges, values)
    # repeat(edges,2)[1:-1] = [0,0,2,2][1:-1] = [0,2]; repeat(values,2) = [7,7].
    np.testing.assert_array_equal(x, [0.0, 2.0])
    np.testing.assert_array_equal(y, [7.0, 7.0])


def test_step_plot_coords_three_bins():
    """n=3: hand-computed paired step coordinates for a piecewise-constant series."""
    edges = np.array([0.0, 1.0, 3.0, 6.0])
    values = np.array([2.0, 5.0, 1.0])
    x, y = step_plot_coords(edges, values)
    np.testing.assert_array_equal(x, [0.0, 1.0, 1.0, 3.0, 3.0, 6.0])
    np.testing.assert_array_equal(y, [2.0, 2.0, 5.0, 5.0, 1.0, 1.0])
    # The step coordinates have 2n points for n bins.
    assert x.shape == (2 * values.size,)
    assert y.shape == (2 * values.size,)


def test_step_plot_coords_datetime_edges_preserve_dtype():
    """Datetime edges round-trip through the step expansion without losing dtype."""
    edges = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04"])
    values = np.array([1.0, 2.0])
    x, y = step_plot_coords(np.asarray(edges), values)
    expected_x = np.asarray(edges)[[0, 1, 1, 2]]
    np.testing.assert_array_equal(x, expected_x)
    np.testing.assert_array_equal(y, [1.0, 1.0, 2.0, 2.0])


# ---------------------------------------------------------------------------
# compute_time_edges
# ---------------------------------------------------------------------------


def test_compute_time_edges_explicit_tedges_roundtrip():
    """Explicit tedges pass through unchanged (ns unit) when the length matches."""
    tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    result = compute_time_edges(tedges=tedges, tstart=None, tend=None, number_of_bins=2)
    assert isinstance(result, pd.DatetimeIndex)
    assert result.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(result.to_numpy(), tedges.as_unit("ns").to_numpy())


def test_compute_time_edges_explicit_tedges_wrong_length():
    """tedges with the wrong length (not number_of_bins + 1) is rejected."""
    tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    with pytest.raises(ValueError, match="tedges must have one more element than number_of_bins"):
        compute_time_edges(tedges=tedges, tstart=None, tend=None, number_of_bins=5)


def test_compute_time_edges_tstart_extrapolates_final_edge():
    """tstart: the FINAL edge is extrapolated as tstart[-1] + (tstart[-1] - tstart[-2])."""
    tstart = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    result = compute_time_edges(tedges=None, tstart=tstart, tend=None, number_of_bins=3)
    expected = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    np.testing.assert_array_equal(result.to_numpy(), expected.as_unit("ns").to_numpy())


def test_compute_time_edges_tstart_wrong_length():
    """tstart length must equal number_of_bins."""
    tstart = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    with pytest.raises(ValueError, match="tstart must have the same number of elements as number_of_bins"):
        compute_time_edges(tedges=None, tstart=tstart, tend=None, number_of_bins=2)


def test_compute_time_edges_tend_extrapolates_initial_edge():
    """tend: the INITIAL edge is extrapolated as tend[0] - (tend[1] - tend[0])."""
    tend = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-04"])
    result = compute_time_edges(tedges=None, tstart=None, tend=tend, number_of_bins=3)
    expected = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
    np.testing.assert_array_equal(result.to_numpy(), expected.as_unit("ns").to_numpy())


def test_compute_time_edges_tend_wrong_length():
    """tend length must equal number_of_bins."""
    tend = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    with pytest.raises(ValueError, match="tend must have the same number of elements as number_of_bins"):
        compute_time_edges(tedges=None, tstart=None, tend=tend, number_of_bins=3)


def test_compute_time_edges_none_provided_raises():
    """With none of tedges/tstart/tend provided, a clear 'or' message is raised."""
    with pytest.raises(ValueError, match="Either provide tedges, tstart, or tend"):
        compute_time_edges(tedges=None, tstart=None, tend=None, number_of_bins=3)


def test_compute_time_edges_tstart_single_bin_raises():
    """A single-bin series from tstart leaves the bin width undefined: a clear ValueError."""
    tstart = pd.DatetimeIndex(["2020-01-01"])
    with pytest.raises(ValueError, match="tstart must have at least 2 elements"):
        compute_time_edges(tedges=None, tstart=tstart, tend=None, number_of_bins=1)


def test_compute_time_edges_tend_single_bin_raises():
    """A single-bin series from tend leaves the bin width undefined: a clear ValueError."""
    tend = pd.DatetimeIndex(["2020-01-01"])
    with pytest.raises(ValueError, match="tend must have at least 2 elements"):
        compute_time_edges(tedges=None, tstart=None, tend=tend, number_of_bins=1)


# ---------------------------------------------------------------------------
# compute_reverse_target
# ---------------------------------------------------------------------------


def test_compute_reverse_target_transpose_normalize():
    """The target is the row-normalized transpose of the forward matrix applied to observed.

    For a forward matrix whose columns map cin to cout, transpose-and-normalize reconstructs
    each cin bin as the column-sum-weighted average of the cout values it fed. The expected
    values here are hand-derived rational numbers, so the check is exact.
    """
    # Column 0 feeds rows 0 and 1; column 1 feeds only row 1.
    coeff = np.array([[1.0, 0.0], [1.0, 2.0]])
    observed = np.array([4.0, 10.0])
    # wt = coeff.T = [[1,1],[0,2]]; row_sums = [2, 2].
    # x_target[0] = (1*4 + 1*10) / 2 = 7; x_target[1] = (0*4 + 2*10) / 2 = 10.
    result = compute_reverse_target(coeff_matrix=coeff, rhs_vector=observed)
    np.testing.assert_array_equal(result, [7.0, 10.0])


def test_compute_reverse_target_zero_column_is_nan():
    """A cin bin with a near-zero forward column sum is returned as NaN (undetermined)."""
    coeff = np.array([[1.0, 0.0], [1.0, 0.0]])  # column 1 has zero sum
    observed = np.array([3.0, 5.0])
    result = compute_reverse_target(coeff_matrix=coeff, rhs_vector=observed)
    # column 0 sum = 2; x_target[0] = (3 + 5) / 2 = 4. column 1 -> NaN.
    np.testing.assert_array_equal(result[0], 4.0)
    assert np.isnan(result[1])


# ---------------------------------------------------------------------------
# _make_strictly_monotone overshoot regression
# ---------------------------------------------------------------------------


def test_make_strictly_monotone_long_run_does_not_overshoot_close_successor():
    """A long plateau followed by a closely-spaced genuine value must stay strictly monotone.

    Regression: the unconditional ``cumcount * 16 * ulp`` bump could push the last duplicate of
    a long run past a successor only a few ulps above the plateau, producing a NON-monotone
    array. The per-run cap (step <= gap / (run_len + 1)) prevents the overshoot.
    """
    base = 1.0
    ulp = np.nextafter(base, np.inf) - base
    # Run of 4 duplicates at ``base`` followed by a genuine value only 10 ulps above; the
    # uncapped bump (up to 4 * 16 = 64 ulp) would overshoot the 10-ulp gap.
    arr = np.array([0.0, base, base, base, base, base + 10 * ulp])
    result = _make_strictly_monotone(arr)
    assert np.all(np.diff(result) > 0), f"non-monotone result: {result}"
    # The successor value is preserved (never exceeded).
    assert result[-1] == base + 10 * ulp


def test_make_strictly_monotone_trailing_plateau_uses_full_bump():
    """A plateau at the array end has no successor, so the uncapped bump applies."""
    arr = np.array([0.0, 1.0, 1.0, 1.0])
    ulp_max = np.nextafter(1.0, np.inf) - 1.0
    bump = 16 * ulp_max
    expected = np.array([0.0, 1.0, 1.0 + bump, 1.0 + 2 * bump])
    result = _make_strictly_monotone(arr)
    np.testing.assert_array_equal(result, expected)
    assert np.all(np.diff(result) > 0)


# ---------------------------------------------------------------------------
# partial_isin zero-width and NaN-edge handling
# ---------------------------------------------------------------------------


def test_partial_isin_zero_width_input_bin_is_zero():
    """A zero-width (degenerate) input bin yields 0.0 overlap (no water passed)."""
    bin_edges_in = np.array([0.0, 10.0, 10.0, 20.0])  # middle bin has zero width
    bin_edges_out = np.array([0.0, 20.0])
    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    # First and third bins fully overlap the single output bin; the zero-width bin is 0.
    np.testing.assert_array_equal(result, [[1.0], [0.0], [1.0]])


def test_partial_isin_nan_input_edge_propagates_nan():
    """A NaN input edge propagates as NaN fractions for the adjacent bins (spin-up masking)."""
    bin_edges_in = np.array([0.0, np.nan, 20.0])
    bin_edges_out = np.array([0.0, 10.0, 20.0])
    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    # Both bins are adjacent to the NaN edge, so every fraction is NaN.
    assert np.isnan(result).all()


# ---------------------------------------------------------------------------
# linear_average extrapolate_method='raise' and 1D NaN bridging
# ---------------------------------------------------------------------------


def test_linear_average_raise_out_of_range():
    """extrapolate_method='raise' rejects any edge outside the data range."""
    x_data = np.array([0.0, 1.0, 2.0])
    y_data = np.array([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="x_edges must be within the range of x_data"):
        linear_average(x_data=x_data, y_data=y_data, x_edges=np.array([-1.0, 2.0]), extrapolate_method="raise")


def test_linear_average_raise_in_range_passes():
    """extrapolate_method='raise' returns the in-range average when all edges are in range."""
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # y = x
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=np.array([1.0, 3.0]), extrapolate_method="raise")
    # mean of y = x over [1, 3] = 2.
    np.testing.assert_array_equal(result, [[2.0]])


def test_linear_average_1d_nan_bridging():
    """1D y_data silently bridges an interior NaN by linear interpolation across the gap.

    This pins the documented 1D-vs-2D asymmetry: a 1D series treats an interior NaN as a gap
    to interpolate across, so a bin spanning it returns the bridged linear mean.
    """
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([0.0, 1.0, np.nan, 3.0, 4.0])  # y = x with an interior hole at x=2
    # The NaN sample is dropped; the remaining points still lie on y = x, so the bridged
    # value at x=2 is 2. Mean of y = x over [1, 3] is therefore 2 (no NaN leaks through).
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=np.array([1.0, 3.0]))
    np.testing.assert_array_equal(result, [[2.0]])


# ---------------------------------------------------------------------------
# solve_underdetermined_system: summed_differences and NaN-row paths
# ---------------------------------------------------------------------------


def test_solve_underdetermined_system_summed_differences_is_feasible_and_sparse():
    """The summed_differences objective stays feasible and yields a sparser difference profile.

    summed_differences (L1 on adjacent differences) promotes piecewise-constant solutions, so
    its difference vector must have strictly more (near-)zero entries than the smooth
    squared_differences (L2) solution while still satisfying A x = b exactly. The L1 objective
    is non-smooth at its piecewise-constant optimum, so a derivative-free method is used.
    """
    # Block-structured system: each equation constrains a separate pair of unknowns.
    matrix = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    rhs = np.array([4.0, 6.0])

    x_summed = solve_underdetermined_system(
        coefficient_matrix=matrix,
        rhs_vector=rhs,
        nullspace_objective="summed_differences",
        optimization_method="Nelder-Mead",
    )
    x_squared = solve_underdetermined_system(
        coefficient_matrix=matrix, rhs_vector=rhs, nullspace_objective="squared_differences"
    )

    # (1) Feasibility of the L1 solution.
    np.testing.assert_allclose(matrix @ x_summed, rhs, atol=1e-8)

    # (2) The L1 minimizer is piecewise-constant within each block: [2, 2, 3, 3].
    np.testing.assert_allclose(x_summed, [2.0, 2.0, 3.0, 3.0], atol=1e-6)

    # (3) Strictly sparser difference profile than the smooth L2 solution.
    n_zero_summed = int(np.sum(np.abs(np.diff(x_summed)) < 1e-6))
    n_zero_squared = int(np.sum(np.abs(np.diff(x_squared)) < 1e-6))
    assert n_zero_summed > n_zero_squared


def test_solve_underdetermined_system_excludes_nan_row():
    """A NaN row is dropped from the system; the solution still satisfies the finite equations."""
    matrix = np.array([
        [1.0, 2.0, 1.0, 0.0],
        [np.nan, np.nan, np.nan, np.nan],
        [0.0, 1.0, 2.0, 1.0],
    ])
    rhs = np.array([5.0, np.nan, 4.0])
    result = solve_underdetermined_system(coefficient_matrix=matrix, rhs_vector=rhs)
    finite = np.array([[1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 1.0]])
    np.testing.assert_allclose(finite @ result, [5.0, 4.0], atol=1e-8)


def test_solve_underdetermined_system_all_nan_rows_raises():
    """When every row contains NaN, no valid rows remain and a ValueError is raised."""
    matrix = np.full((2, 3), np.nan)
    rhs = np.array([np.nan, np.nan])
    with pytest.raises(ValueError, match="No valid rows found"):
        solve_underdetermined_system(coefficient_matrix=matrix, rhs_vector=rhs)


# ---------------------------------------------------------------------------
# simplify_bins early-exit must echo the flow array when flow is provided
# ---------------------------------------------------------------------------


def test_simplify_bins_empty_with_flow_returns_empty_flow_array():
    """Empty input with flow provided returns an empty float array, not None."""
    edges = np.array([0.0])
    values = np.array([])
    flow = np.array([])
    new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
    assert len(new_edges) == 1
    assert len(new_values) == 0
    assert new_flow is not None
    assert new_flow.dtype == np.float64
    assert len(new_flow) == 0


# ---------------------------------------------------------------------------
# U1: linear_average 2-D NaN contract is method-independent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["outer", "raise", "nan"])
def test_linear_average_2d_nan_propagates_for_all_methods(method):
    """The documented 2-D per-row NaN contract must hold for every extrapolate_method.

    Regression: the per-row NaN-propagation block was nested under the ``'nan'`` branch, so
    ``'outer'``/``'raise'`` zeroed the NaN trapezoids and returned a silently wrong finite
    average (1.0) for a row containing an interior NaN. A row whose bin touches a NaN segment
    must be NaN regardless of the extrapolation method; a clean row is unaffected.
    """
    x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y_data = np.array([[2.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, np.nan, 2.0, 2.0]])
    x_edges = np.array([0.0, 4.0])
    result = linear_average(x_data=x_data, y_data=y_data, x_edges=x_edges, extrapolate_method=method)
    np.testing.assert_array_equal(result[0], [2.0])  # clean row: unaffected
    assert np.isnan(result[1, 0])  # NaN-touching row: NaN, not the buggy 1.0


# ---------------------------------------------------------------------------
# U2: time_bin_overlap accepts pandas Timestamp / datetime64 bin_tedges
# ---------------------------------------------------------------------------


def test_time_bin_overlap_timestamp_bin_tedges():
    """Timestamp tedges and Timestamp (start, end) tuples must work like the numeric case.

    Regression: object arrays of Timestamps drove ``np.maximum(0, Timedelta)`` and raised
    ``TypeError``. Ten-day bins mirror the numeric ``test_time_bin_overlap_basic`` case, so the
    expected fractions are identical.
    """
    tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-11", "2020-01-21", "2020-01-31"])
    bin_tedges = [
        (pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-16")),
        (pd.Timestamp("2020-01-26"), pd.Timestamp("2020-02-05")),
    ]
    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, [[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]])


def test_time_bin_overlap_timestamp_matches_datetime64_and_numeric():
    """Timestamp, datetime64, and numeric inputs give bit-identical overlap fractions."""
    numeric = time_bin_overlap(tedges=np.array([0, 10, 20, 30]), bin_tedges=[(5, 15), (25, 35)])
    tedges_dt = pd.DatetimeIndex(["2020-01-01", "2020-01-11", "2020-01-21", "2020-01-31"])
    bins_ts = [
        (pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-16")),
        (pd.Timestamp("2020-01-26"), pd.Timestamp("2020-02-05")),
    ]
    bins_dt64 = [(a.to_datetime64(), b.to_datetime64()) for a, b in bins_ts]
    np.testing.assert_array_equal(time_bin_overlap(tedges=tedges_dt, bin_tedges=bins_ts), numeric)
    np.testing.assert_array_equal(time_bin_overlap(tedges=np.asarray(tedges_dt), bin_tedges=bins_dt64), numeric)


# ---------------------------------------------------------------------------
# U3: simplify_bins uses an iterative stack (no RecursionError) with identical splits
# ---------------------------------------------------------------------------


def _simplify_bins_recursive_reference(*, edges, values, flow=None, tol=0.0):
    """Reference implementation using the original recursive _splits (pre-U3).

    Reproduces the exact merged-bin outputs so the iterative-stack rewrite can be pinned
    bit-for-bit on inputs shallow enough not to overflow the interpreter stack.
    """
    edges = np.asarray(edges) if not isinstance(edges, pd.DatetimeIndex) else edges
    values = np.asarray(values, dtype=float)
    widths = np.asarray(np.diff(edges), dtype=float)
    weights = widths * np.asarray(flow, dtype=float) if flow is not None else widths

    def _splits(lo, hi):
        if np.ptp(values[lo:hi]) <= tol:
            return []
        i = lo + int(np.argmax(np.abs(np.diff(values[lo:hi])))) + 1
        return [*_splits(lo, i), i, *_splits(i, hi)]

    s = np.array([0, *_splits(0, len(values))])
    idx = np.append(s, len(values))
    new_edges = edges[idx]
    new_widths = np.add.reduceat(widths, s)
    weight_sums = np.add.reduceat(weights, s)
    new_values = np.add.reduceat(weights * values, s) / weight_sums
    new_flow = weight_sums / new_widths if flow is not None else None
    return new_edges, new_values, new_flow


def test_simplify_bins_no_recursion_error_on_smooth_monotone():
    """A smooth monotone breakthrough (logistic, n=2500) must not overflow the stack.

    Regression: the recursive splitter peeled one element per level on monotone data whose
    largest |diff| sits at a segment edge, raising RecursionError. The iterative stack removes
    the depth limit; the result is a valid, non-trivial simplification.
    """
    n = 2500
    values = 1.0 / (1.0 + np.exp(-np.linspace(-6.0, 6.0, n)))
    edges = np.arange(n + 1, dtype=float)
    new_edges, new_values, _ = simplify_bins(edges=edges, values=values, tol=0.0)
    # tol=0 merges only runs of identical values; the strictly increasing logistic has none,
    # so no bins merge and the series is returned intact.
    assert len(new_values) == n
    np.testing.assert_array_equal(new_edges, edges)


def test_simplify_bins_iterative_matches_recursive_reference():
    """The iterative stack reproduces the recursive splitter's merged bins bit-for-bit.

    Uses a non-monotone series with a non-trivial merge structure and volume weighting so the
    split indices, merged values, and merged flow are all exercised on a case shallow enough
    for the recursive reference to run.
    """
    edges = np.array([0.0, 1.0, 3.0, 6.0, 6.5, 8.0, 11.0, 12.0])
    values = np.array([1.0, 1.0, 1.0, 5.0, 5.2, 2.0, 2.0])
    flow = np.array([2.0, 4.0, 1.0, 3.0, 0.5, 2.5, 1.5])
    for tol in (0.0, 0.5):
        e_new, v_new, f_new = simplify_bins(edges=edges, values=values, flow=flow, tol=tol)
        e_ref, v_ref, f_ref = _simplify_bins_recursive_reference(edges=edges, values=values, flow=flow, tol=tol)
        np.testing.assert_array_equal(e_new, e_ref)
        np.testing.assert_array_equal(v_new, v_ref)
        np.testing.assert_array_equal(f_new, f_ref)


# ---------------------------------------------------------------------------
# U4: banded WᵀW assembly (dense BLAS build) equivalence
# ---------------------------------------------------------------------------


def test_solve_inverse_transport_banded_wtw_assembly_matches_add_at_reference():
    """The dense-BLAS WᵀW build equals the per-diagonal np.add.at scatter (atol 1e-12).

    Forming WᵀW with a matmul reorders the summation relative to np.add.at, so the two agree
    to ~1e-13 rather than bit-for-bit; pin the equivalence at atol 1e-12.
    """
    rng = np.random.default_rng(0)
    n_obs, full_band, n_output = 60, 9, 45
    band_vals = rng.random((n_obs, full_band))
    col_start = rng.integers(0, n_output - full_band, size=n_obs).astype(np.intp)
    cols = col_start[:, None] + np.arange(full_band)[None, :]
    in_range = cols < n_output
    cols_clipped = np.clip(cols, 0, n_output - 1)

    # Reference: original per-diagonal add.at assembly of the lower banded WᵀW.
    ab_ref = np.zeros((full_band, n_output))
    for d in range(full_band):
        prod = band_vals[:, d:] * band_vals[:, : full_band - d]
        c = cols_clipped[:, : full_band - d]
        m = in_range[:, : full_band - d] & in_range[:, d:]
        np.add.at(ab_ref[d], c[m], prod[m])

    # New path: dense W build, one BLAS matmul, sub-diagonal extraction.
    w_dense = np.zeros((n_obs, n_output))
    obs_idx = np.broadcast_to(np.arange(n_obs)[:, None], cols.shape)
    w_dense[obs_idx[in_range], cols_clipped[in_range]] = band_vals[in_range]
    gram = w_dense.T @ w_dense
    ab_new = np.zeros((full_band, n_output))
    for d in range(full_band):
        ab_new[d, : n_output - d] = np.diagonal(gram, offset=-d)

    np.testing.assert_allclose(ab_new, ab_ref, atol=1e-12)


def test_solve_inverse_transport_banded_matches_dense_solver():
    """The banded solver reproduces the dense solve_inverse_transport on a random band.

    End-to-end guard that the dense-BLAS assembly leaves the recovered signal unchanged: the
    banded Cholesky + semi-normal refinement matches the dense least-squares solution to within
    the asserted atol=1e-9 on the shared active columns (and the NaN pattern is identical).
    """
    rng = np.random.default_rng(3)
    n_obs, full_band, n_output = 80, 11, 60
    band_vals = rng.random((n_obs, full_band))
    band_vals /= band_vals.sum(axis=1, keepdims=True)  # precondition: valid rows sum to 1
    col_start = rng.integers(0, n_output - full_band, size=n_obs).astype(np.intp)
    observed = rng.random(n_obs)
    lam = 1e-3

    cols = col_start[:, None] + np.arange(full_band)[None, :]
    in_range = cols < n_output
    cols_clipped = np.clip(cols, 0, n_output - 1)
    obs_idx = np.broadcast_to(np.arange(n_obs)[:, None], cols.shape)
    w_dense = np.zeros((n_obs, n_output))
    w_dense[obs_idx[in_range], cols_clipped[in_range]] = band_vals[in_range]

    x_banded = solve_inverse_transport_banded(
        band_vals=band_vals, col_start=col_start, observed=observed, n_output=n_output, regularization_strength=lam
    )
    x_dense = solve_inverse_transport(
        w_forward=w_dense, observed=observed, n_output=n_output, regularization_strength=lam
    )
    np.testing.assert_array_equal(np.isnan(x_banded), np.isnan(x_dense))
    active = ~np.isnan(x_banded)
    np.testing.assert_allclose(x_banded[active], x_dense[active], atol=1e-9)


def test_solve_inverse_transport_banded_rejects_nonpositive_lambda():
    """λ <= 0 cannot make the banded Cholesky positive definite, so it is rejected."""
    band_vals = np.ones((3, 2))
    col_start = np.array([0, 1, 2], dtype=np.intp)
    observed = np.ones(3)
    with pytest.raises(ValueError, match="regularization_strength must be > 0"):
        solve_inverse_transport_banded(
            band_vals=band_vals, col_start=col_start, observed=observed, n_output=4, regularization_strength=0.0
        )


def test_simplify_bins_all_zero_flow_group_falls_back_to_width_weights():
    """#313 UTI-F2: a merged group whose bins all have zero flow has zero volume weight.

    The volume-weighted average is then 0/0 -> NaN (with a RuntimeWarning) even though
    the group's value is perfectly well defined; fall back to width weighting there.
    """
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([2.0, 2.0, 5.0, 5.0])
    flow = np.array([0.0, 0.0, 10.0, 10.0])
    new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
    np.testing.assert_allclose(new_edges, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(new_values, [2.0, 5.0])
    assert new_flow is not None  # narrow the flow=None overload for the type checker
    np.testing.assert_allclose(new_flow, [0.0, 10.0])


def test_time_bin_overlap_nanosecond_precision_far_epoch():
    """#313 UTI-F3: datetime edges must be differenced in int64 nanoseconds.

    Converting asi8 to float64 BEFORE differencing rounds a year-2200 epoch value
    (~7.3e18 ns, float64 ulp = 1024 ns) to the nearest 1024 ns, corrupting
    sub-microsecond bins. Differences of int64 nanoseconds are exact.
    """
    base = pd.Timestamp("2200-01-01")
    tedges = pd.DatetimeIndex([base + pd.Timedelta(n, "ns") for n in (0, 1000, 2000, 3000)])
    bin_tedges = [(base + pd.Timedelta(500, "ns"), base + pd.Timedelta(1500, "ns"))]
    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, [[0.5, 0.5, 0.0]], rtol=0, atol=0)


def test_solve_tikhonov_resolution_dead_column_nan_target():
    """#313 UTI-F4: a dead (all-zero) column whose x_target is NaN is unregularized.

    Its gram row/column is then exactly zero and np.linalg.inv raised LinAlgError.
    The pinned diagonal leaves every other resolution entry unchanged and reports
    fraction_data = 1.0, the documented convention for non-regularized entries.
    """
    coeff = np.array([[1.0, 0.0], [2.0, 0.0]])
    rhs = np.array([1.0, 2.0])
    x_target = np.array([1.0, np.nan])
    lam = 1e-10
    x, fraction_data = solve_tikhonov(
        coefficient_matrix=coeff,
        rhs_vector=rhs,
        x_target=x_target,
        regularization_strength=lam,
        return_resolution=True,
    )
    # lstsq returns the minimum-norm solution: dead column -> 0, live column exact.
    np.testing.assert_allclose(x, [1.0, 0.0])
    # Live column: R = 1 - lam / (||col||^2 + lam); dead unregularized column: 1.0 by convention.
    np.testing.assert_allclose(fraction_data, [1.0 - lam / (5.0 + lam), 1.0], rtol=1e-12)
