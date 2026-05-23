import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests.exceptions
from scipy.linalg import null_space

from gwtransport.examples import generate_example_data, generate_example_deposition_timeseries
from gwtransport.utils import (
    _make_strictly_monotone,
    combine_bin_series,
    get_soil_temperature,
    linear_average,
    linear_interpolate,
    partial_isin,
    solve_tikhonov,
    solve_underdetermined_system,
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
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_no_overlap():
    """Test when there is no overlap between input and output bins."""
    bin_edges_in = np.array([0, 10, 20])
    bin_edges_out = np.array([30, 40, 50])
    expected = np.array([[0.0, 0.0], [0.0, 0.0]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_complete_overlap():
    """Test when input bins completely overlap with output bins."""
    bin_edges_in = np.array([0, 10, 20])
    bin_edges_out = np.array([0, 10, 20])
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_exact_bin_match():
    """Test when input and output bins exactly match."""
    bin_edges_in = np.array([10, 20, 30])
    bin_edges_out = np.array([10, 20, 30])
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_multiple_bins():
    """Test with multiple bins of different sizes."""
    bin_edges_in = np.array([0, 10, 20, 30])
    bin_edges_out = np.array([5, 12, 18, 25])
    expected = np.array([[0.5, 0.0, 0.0], [0.2, 0.6, 0.2], [0.0, 0.0, 0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_partial_overlaps():
    """Test various partial overlaps."""
    bin_edges_in = np.array([0, 20])  # One large input bin
    bin_edges_out = np.array([5, 10, 15])  # Two smaller output bins
    expected = np.array([[0.25, 0.25]])  # 25% overlap with each output bin

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_list_inputs():
    """Test with list inputs instead of numpy arrays."""
    bin_edges_in = [0, 10, 20, 30]
    bin_edges_out = [5, 15, 25]
    expected = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_empty_inputs():
    """Test with minimal valid inputs."""
    bin_edges_in = np.array([0, 10])
    bin_edges_out = np.array([5, 15])
    expected = np.array([[0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_single_bin():
    """Test with single input and output bins."""
    bin_edges_in = np.array([0, 10])
    bin_edges_out = np.array([5, 15])
    expected = np.array([[0.5]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_edge_alignment():
    """Test when edges are perfectly aligned."""
    bin_edges_in = np.array([0, 10, 20, 30])
    bin_edges_out = np.array([0, 15, 30])
    expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    result = partial_isin(bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


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
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


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


def test_combine_bin_series_basic():
    """Test basic functionality of combine_bin_series."""
    # Simple case: non-overlapping bins
    a = np.array([1.0, 2.0])
    a_edges = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 4.0])
    b_edges = np.array([1.0, 1.5, 2.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [0, 1, 1.5, 2]
    expected_edges = np.array([0.0, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # Expected values for c: [1, 2, 2] (a[0] broadcasts to first bin, a[1] broadcasts to bins 2&3)
    # Expected values for d: [0, 3, 4] (b[0] broadcasts to second bin, b[1] broadcasts to third bin)
    expected_c = np.array([1.0, 2.0, 2.0])
    expected_d = np.array([0.0, 3.0, 4.0])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_identical_edges():
    """Test combine_bin_series when both series have identical edges."""
    a = np.array([1.0, 2.0, 3.0])
    a_edges = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    b_edges = np.array([0.0, 1.0, 2.0, 3.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Edges should remain the same
    expected_edges = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # Values should be preserved
    np.testing.assert_allclose(c, a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, b, rtol=0, atol=1e-6)


def test_combine_bin_series_overlapping_bins():
    """Test combine_bin_series with overlapping bin structures."""
    a = np.array([10.0, 20.0])
    a_edges = np.array([0.0, 5.0, 10.0])
    b = np.array([30.0, 40.0])
    b_edges = np.array([2.0, 7.0, 12.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [0, 2, 5, 7, 10, 12]
    expected_edges = np.array([0.0, 2.0, 5.0, 7.0, 10.0, 12.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # Test that the values are broadcasted/repeated correctly
    # a[0]=10 covers [0,5]: broadcasts to bins [0,2] and [2,5]
    # a[1]=20 covers [5,10]: broadcasts to bins [5,7] and [7,10]
    # b[0]=30 covers [2,7]: broadcasts to bins [2,5] and [5,7]
    # b[1]=40 covers [7,12]: broadcasts to bins [7,10] and [10,12]
    expected_c = np.array([10.0, 10.0, 20.0, 20.0, 0.0])
    expected_d = np.array([0.0, 30.0, 30.0, 40.0, 40.0])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_single_bins():
    """Test combine_bin_series with single bins."""
    a = np.array([5.0])
    a_edges = np.array([0.0, 2.0])
    b = np.array([10.0])
    b_edges = np.array([1.0, 3.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [0, 1, 2, 3]
    expected_edges = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # a=5 covers [0,2]: broadcasts to [0,1] and [1,2]
    # b=10 covers [1,3]: broadcasts to [1,2] and [2,3]
    expected_c = np.array([5.0, 5.0, 0.0])
    expected_d = np.array([0.0, 10.0, 10.0])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_nested_bins():
    """Test combine_bin_series where one series is nested within another."""
    a = np.array([100.0])
    a_edges = np.array([0.0, 10.0])
    b = np.array([20.0, 30.0])
    b_edges = np.array([2.0, 5.0, 8.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [0, 2, 5, 8, 10]
    expected_edges = np.array([0.0, 2.0, 5.0, 8.0, 10.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # a=100 covers [0,10]: broadcasts to all combined bins within its range
    # b[0]=20 covers [2,5] and b[1]=30 covers [5,8]
    expected_c = np.array([100.0, 100.0, 100.0, 100.0])
    expected_d = np.array([0.0, 20.0, 30.0, 0.0])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_non_overlapping():
    """Test combine_bin_series with completely non-overlapping bins."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 4.0])
    b_edges = np.array([3.0, 4.0, 5.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [0, 1, 2, 3, 4, 5]
    expected_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # a maps to first two bins, b maps to last two bins
    expected_c = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
    expected_d = np.array([0.0, 0.0, 0.0, 3.0, 4.0])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_zero_values():
    """Test combine_bin_series with zero values."""
    a = np.array([0.0, 5.0])
    a_edges = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 0.0])
    b_edges = np.array([0.5, 1.5, 2.5])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [0, 0.5, 1, 1.5, 2, 2.5]
    expected_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # Check that zero values are preserved and broadcasted correctly
    expected_c = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    expected_d = np.array([0.0, 3.0, 3.0, 0.0, 0.0])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_floating_point():
    """Test combine_bin_series with floating point precision."""
    a = np.array([1.1, 2.2])
    a_edges = np.array([0.1, 1.1, 2.1])
    b = np.array([3.3, 4.4])
    b_edges = np.array([0.6, 1.6, 2.6])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [0.1, 0.6, 1.1, 1.6, 2.1, 2.6]
    expected_edges = np.array([0.1, 0.6, 1.1, 1.6, 2.1, 2.6])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # Test with appropriate precision and broadcasting
    expected_c = np.array([1.1, 1.1, 2.2, 2.2, 0.0])
    expected_d = np.array([0.0, 3.3, 3.3, 4.4, 4.4])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-10)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-10)


def test_combine_bin_series_input_validation():
    """Test input validation for combine_bin_series."""
    # Test mismatched array lengths
    a = np.array([1.0, 2.0])
    a_edges = np.array([0.0, 1.0])  # Should have 3 elements for 2 bins
    b = np.array([3.0])
    b_edges = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="a_edges must have len\\(a\\) \\+ 1 elements"):
        combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Test mismatched b array lengths
    a = np.array([1.0])
    a_edges = np.array([0.0, 1.0])
    b = np.array([3.0, 4.0])
    b_edges = np.array([1.0, 2.0])  # Should have 3 elements for 2 bins

    with pytest.raises(ValueError, match="b_edges must have len\\(b\\) \\+ 1 elements"):
        combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)


def test_combine_bin_series_list_inputs():
    """Test combine_bin_series with list inputs."""
    a = [1.0, 2.0]
    a_edges = [0.0, 1.0, 2.0]
    b = [3.0, 4.0]
    b_edges = [1.5, 2.5, 3.5]

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Should work with list inputs and return numpy arrays
    assert isinstance(c, np.ndarray)
    assert isinstance(c_edges, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(d_edges, np.ndarray)

    # Expected combined edges: [0, 1, 1.5, 2, 2.5, 3.5]
    expected_edges = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.5])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)


def test_combine_bin_series_negative_values():
    """Test combine_bin_series with negative values."""
    a = np.array([-5.0, -2.0])
    a_edges = np.array([-10.0, -3.0, 0.0])
    b = np.array([1.0, 4.0])
    b_edges = np.array([-1.0, 2.0, 5.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [-10, -3, -1, 0, 2, 5]
    expected_edges = np.array([-10.0, -3.0, -1.0, 0.0, 2.0, 5.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # Test correct mapping with negative values and broadcasting
    expected_c = np.array([-5.0, -2.0, -2.0, 0.0, 0.0])
    expected_d = np.array([0.0, 0.0, 1.0, 1.0, 4.0])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_empty_arrays():
    """Test combine_bin_series with minimal valid inputs."""
    a = np.array([42.0])
    a_edges = np.array([0.0, 1.0])
    b = np.array([24.0])
    b_edges = np.array([0.5, 1.5])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)

    # Expected combined edges: [0, 0.5, 1, 1.5]
    expected_edges = np.array([0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    expected_c = np.array([42.0, 42.0, 0.0])
    expected_d = np.array([0.0, 24.0, 24.0])
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_extrapolation_nearest():
    """Test combine_bin_series with nearest extrapolation."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    b_edges = np.array([2.0, 3.0, 4.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges, extrapolation="nearest")

    # Expected combined edges: [1, 2, 3, 4]
    expected_edges = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # With nearest extrapolation:
    # c extends to all bins using nearest values
    # d extends to all bins using nearest values
    expected_c = np.array([1.0, 2.0, 2.0])  # nearest for [3,4] is a[1]=2.0
    expected_d = np.array([10.0, 10.0, 20.0])  # nearest for [1,2] is b[0]=10.0
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_extrapolation_nan():
    """Test combine_bin_series with nan extrapolation."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    b_edges = np.array([2.0, 3.0, 4.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges, extrapolation=np.nan)

    # Expected combined edges: [1, 2, 3, 4]
    expected_edges = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # With nan extrapolation:
    # Out-of-range bins get nan values
    expected_c = np.array([1.0, 2.0, np.nan])  # [3,4] is out of range for a
    expected_d = np.array([np.nan, 10.0, 20.0])  # [1,2] is out of range for b
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_extrapolation_custom_value():
    """Test combine_bin_series with custom extrapolation value."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    b_edges = np.array([2.0, 3.0, 4.0])

    c, c_edges, d, d_edges = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges, extrapolation=-999.0)

    # Expected combined edges: [1, 2, 3, 4]
    expected_edges = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(c_edges, expected_edges, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges, expected_edges, rtol=0, atol=1e-6)

    # With custom extrapolation value:
    # Out-of-range bins get the custom value
    expected_c = np.array([1.0, 2.0, -999.0])  # [3,4] is out of range for a
    expected_d = np.array([-999.0, 10.0, 20.0])  # [1,2] is out of range for b
    np.testing.assert_allclose(c, expected_c, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d, expected_d, rtol=0, atol=1e-6)


def test_combine_bin_series_extrapolation_default_behavior():
    """Test that default extrapolation preserves backwards compatibility."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    b_edges = np.array([2.0, 3.0, 4.0])

    # Default behavior should be equivalent to extrapolation=0.0
    c1, c_edges1, d1, d_edges1 = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges)
    c2, c_edges2, d2, d_edges2 = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges, extrapolation=0.0)

    np.testing.assert_allclose(c1, c2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d1, d2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(c_edges1, c_edges2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d_edges1, d_edges2, rtol=0, atol=1e-6)


def test_combine_bin_series_extrapolation_no_out_of_range():
    """Test extrapolation when there are no out-of-range bins."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 4.0])
    b_edges = np.array([0.0, 1.0, 2.0])

    # When series have identical ranges, extrapolation method shouldn't matter
    c1, _, d1, _ = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges, extrapolation="nearest")
    c2, _, d2, _ = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges, extrapolation=np.nan)
    c3, _, d3, _ = combine_bin_series(a=a, a_edges=a_edges, b=b, b_edges=b_edges, extrapolation=0.0)

    np.testing.assert_allclose(c1, c2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(c1, c3, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d1, d2, rtol=0, atol=1e-6)
    np.testing.assert_allclose(d1, d3, rtol=0, atol=1e-6)


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
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_no_overlap():
    """Test when time ranges don't overlap with any bins."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(40, 50), (60, 70)]
    expected = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_complete_overlap():
    """Test when time ranges completely overlap with bins."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(0, 10), (10, 20), (20, 30)]
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_partial_overlap():
    """Test partial overlaps with multiple bins."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(5, 25)]
    expected = np.array([[0.5, 1.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_single_bin():
    """Test with single time bin."""
    tedges = np.array([0, 10])
    bin_tedges = [(5, 15), (-5, 5)]
    expected = np.array([[0.5], [0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_edge_alignment():
    """Test when time ranges align perfectly with bin edges."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(0, 20), (10, 30)]
    expected = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


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
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_overlapping_ranges():
    """Test with overlapping time ranges."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(5, 15), (10, 25)]
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 1.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_zero_width_bins():
    """Test with zero-width time bins."""
    tedges = np.array([0, 10, 10, 20])
    bin_tedges = [(5, 15)]
    expected = np.array([[0.5, 0.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_large_range():
    """Test with a large time range covering all bins."""
    tedges = np.array([0, 10, 20, 30])
    bin_tedges = [(-10, 40)]
    expected = np.array([[1.0, 1.0, 1.0]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


def test_time_bin_overlap_list_inputs():
    """Test with list inputs instead of numpy arrays."""
    tedges = [0, 10, 20, 30]
    bin_tedges = [(5, 15), (25, 35)]
    expected = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.5]])

    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


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
        np.testing.assert_allclose([total_overlap], [10.0], rtol=0, atol=1e-06)


def test_time_bin_overlap_boundary_cases():
    """Test boundary cases and edge conditions."""
    tedges = np.array([0, 5, 10, 15, 20])

    # Range that touches bin boundary but doesn't overlap
    bin_tedges = [(10, 10)]
    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    expected = np.array([[0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)

    # Range that exactly matches a bin
    bin_tedges = [(5, 10)]
    result = time_bin_overlap(tedges=tedges, bin_tedges=bin_tedges)
    expected = np.array([[0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)


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
    np.testing.assert_allclose(fraction_data, 1.0, atol=1e-4)


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
    """Flow floor of 5 m3/day must hold even after the spill loop divides flow.

    Regression test for issue 167: the floor was applied before spills, so
    spill divisions could push flow well below 5 m3/day (observed min ~0.3).
    """
    df, _ = generate_example_data(rng=seed)
    assert df["flow"].min() >= 5.0, f"flow min {df['flow'].min()} below 5.0 m3/day for seed={seed}"


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
