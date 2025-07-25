import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal

from gwtransport.utils import (
    combine_bin_series,
    diff,
    get_soil_temperature,
    linear_average,
    linear_interpolate,
    partial_isin,
)


def test_linear_interpolate():
    # Test 1: Basic linear interpolation
    x_ref = np.array([0, 2, 4, 6, 8, 10])
    y_ref = np.array([0, 4, 8, 12, 16, 20])  # y = 2x
    x_query = np.array([1, 3, 5, 7, 9])
    expected = np.array([2, 6, 10, 14, 18])

    result = linear_interpolate(x_ref, y_ref, x_query)
    assert_array_almost_equal(result, expected, decimal=6)

    # Test 2: Single value interpolation
    x_ref = np.array([0, 1])
    y_ref = np.array([0, 1])
    x_query = np.array([0.5])
    expected = np.array([0.5])

    result = linear_interpolate(x_ref, y_ref, x_query)
    assert_array_almost_equal(result, expected, decimal=6)

    # Test 3: Edge cases - query points outside range
    x_ref = np.array([0, 1, 2])
    y_ref = np.array([0, 1, 2])
    x_query = np.array([-1, 3])  # Outside the range
    expected = np.array([0, 2])  # Should clip to nearest values

    result = linear_interpolate(x_ref, y_ref, x_query)
    assert_array_almost_equal(result, expected, decimal=6)

    # Test 4: Non-uniform spacing
    x_ref = np.array([0, 1, 10])
    y_ref = np.array([0, 2, 20])
    x_query = np.array([0.5, 5.5])
    expected = np.array([1, 11])

    result = linear_interpolate(x_ref, y_ref, x_query)
    assert_array_almost_equal(result, expected, decimal=6)

    # Test 5: Exact matches with reference points
    x_ref = np.array([0, 1, 2])
    y_ref = np.array([0, 10, 20])
    x_query = np.array([0, 1, 2])
    expected = np.array([0, 10, 20])

    result = linear_interpolate(x_ref, y_ref, x_query)
    assert_array_almost_equal(result, expected, decimal=6)


def test_diff():
    # Test 1: Basic difference
    x = np.array([0, 1, 2, 3, 4, 6])
    expected = np.array([1, 1, 1, 1, 1.5, 2])

    result = diff(x, alignment="centered")
    assert_array_almost_equal(result, expected, decimal=6)


def test_diff_centered_two_points():
    x = np.array([10, 20])
    expected = np.array([10, 10])
    result = diff(x, alignment="centered")
    assert_array_almost_equal(result, expected, decimal=6)


def test_diff_left():
    x = np.array([0, 1, 2, 3, 4, 6])
    expected = np.array([1, 1, 1, 1, 2, 2])
    result = diff(x, alignment="left")
    assert_array_almost_equal(result, expected, decimal=6)


def test_diff_right():
    x = np.array([0, 1, 2, 3, 4, 6])
    expected = np.array([1, 1, 1, 1, 1, 2])
    result = diff(x, alignment="right")
    assert_array_almost_equal(result, expected, decimal=6)


def test_constant_function():
    """Test average of constant function y=2."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([2, 2, 2, 2, 2])
    x_edges = np.array([0, 2, 4])

    expected = np.array([[2, 2]])  # Average is constant, now 2D
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_linear_function():
    """Test average of linear function y=x."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 2, 3, 4])
    x_edges = np.array([0, 2, 4])

    # Average of y=x from 0 to 2 = 1
    # Average of y=x from 2 to 4 = 3
    expected = np.array([[1, 3]])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_piecewise_linear():
    """Test average of piecewise linear function."""
    x_data = np.array([0, 1, 2, 3])
    y_data = np.array([0, 1, 1, 0])
    x_edges = np.array([0, 1.5, 3])

    # Integral from 0 to 1.5 = 1, width = 1.5 → average = 2/3
    # Integral from 1.5 to 3 = 1, width = 1.5 → average = 2/3
    expected = np.array([[1.0 / 1.5, 1.0 / 1.5]])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_edges_beyond_data():
    """Test averages with edges outside the data range."""
    x_data = np.array([1, 2, 3])
    y_data = np.array([1, 2, 3])
    x_edges = np.array([0, 4])

    # Extrapolation should extend the first and last segments
    # Average of y=x from 0 to 4 = 2
    expected = np.array([[2]])
    result = linear_average(x_data, y_data, x_edges, extrapolate_method="outer")

    np.testing.assert_allclose(result, expected)


def test_edges_matching_data():
    """Test when edges exactly match data points."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 4, 9, 16])
    x_edges = np.array([1, 3])

    # Average under the curve from 1 to 3 = 4.5
    expected = np.array([[4.5]])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_multiple_edge_intervals():
    """Test with multiple averaging intervals."""
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0, 1, 4, 9, 16, 25])
    x_edges = np.array([0, 1, 2, 3, 4, 5])

    # Average of each segment
    expected = np.array([[0.5, 2.5, 6.5, 12.5, 20.5]])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_empty_interval():
    """Test averaging over an empty interval (edges are the same)."""
    x_data = np.array([0, 1, 2, 3])
    y_data = np.array([0, 1, 4, 9])
    x_edges = np.array([0, 1, 1, 2])

    # Second interval has zero width at x=1, so average should be y=1
    expected = np.array([[0.5, 1.0, 2.5]])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_input_validation():
    """Test input validation."""
    # Test unequal lengths of x_data and y_data
    with pytest.raises(ValueError, match="x_data and y_data must have the same length and be non-empty"):
        linear_average([0, 1], [0], [0, 1])

    # Test x_edges too short
    with pytest.raises(ValueError, match="x_edges must contain at least 2 values in each row"):
        linear_average([0, 1], [0, 1], [0])

    # Test x_data not in ascending order
    with pytest.raises(ValueError, match="x_data must be in ascending order"):
        linear_average([1, 0], [0, 1], [0, 1])

    # Test x_edges not in ascending order
    with pytest.raises(ValueError, match="x_edges must be in ascending order"):
        linear_average([0, 1], [0, 1], [1, 0])


def test_complex_piecewise_function():
    """Test a more complex piecewise linear function."""
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0, 2, 1, 3, 0, 2])
    x_edges = np.array([0.5, 2.5, 4.5])

    # First interval: integral = 3.0, width = 2.0 → average = 1.5
    # Second interval: integral = 3.0, width = 2.0 → average = 1.5
    expected = np.array([[1.5, 1.5]])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_edge_case_numerical_precision():
    """Test numerical precision for very close x values."""
    x_data = np.array([0, 1e-10, 1])
    y_data = np.array([0, 1e-10, 1])
    x_edges = np.array([0, 0.5, 1])

    # For a linear function y=x, the average from 0 to 0.5 is 0.25
    # and from 0.5 to 1 is 0.75
    expected = np.array([[0.25, 0.75]])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_single_point_data():
    """Test with a single data point - should extrapolate as constant."""
    x_data = np.array([1])
    y_data = np.array([5])
    x_edges = np.array([0, 2])

    # Single point should be treated as constant value
    expected = np.array([[5]])
    result = linear_average(x_data, y_data, x_edges, extrapolate_method="outer")

    np.testing.assert_allclose(result, expected)


def test_zero_width_interval_edge_case():
    """Test handling of a zero-width interval at the edge."""
    x_data = np.array([0, 1, 2])
    y_data = np.array([0, 1, 2])
    x_edges = np.array([0, 0, 1])

    # First interval has zero width at x=0, so average should be y=0
    # Second interval is 0 to 1, average is 0.5
    expected = np.array([[0.0, 0.5]])
    result = linear_average(x_data, y_data, x_edges)

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

    result = linear_average(x_data, y_data, x_edges_2d)

    np.testing.assert_allclose(result, expected, rtol=1e-2)


def test_basic_case():
    """Test the basic case with new interface."""
    bin_edges_in = np.array([0, 10, 20, 30])
    bin_edges_out = np.array([5, 15, 25])
    expected = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_no_overlap():
    """Test when there is no overlap between input and output bins."""
    bin_edges_in = np.array([0, 10, 20])
    bin_edges_out = np.array([30, 40, 50])
    expected = np.array([[0.0, 0.0], [0.0, 0.0]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_complete_overlap():
    """Test when input bins completely overlap with output bins."""
    bin_edges_in = np.array([0, 10, 20])
    bin_edges_out = np.array([0, 10, 20])
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_exact_bin_match():
    """Test when input and output bins exactly match."""
    bin_edges_in = np.array([10, 20, 30])
    bin_edges_out = np.array([10, 20, 30])
    expected = np.array([[1.0, 0.0], [0.0, 1.0]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_multiple_bins():
    """Test with multiple bins of different sizes."""
    bin_edges_in = np.array([0, 10, 20, 30])
    bin_edges_out = np.array([5, 12, 18, 25])
    expected = np.array([[0.5, 0.0, 0.0], [0.2, 0.6, 0.2], [0.0, 0.0, 0.5]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_partial_overlaps():
    """Test various partial overlaps."""
    bin_edges_in = np.array([0, 20])  # One large input bin
    bin_edges_out = np.array([5, 10, 15])  # Two smaller output bins
    expected = np.array([[0.25, 0.25]])  # 25% overlap with each output bin

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_list_inputs():
    """Test with list inputs instead of numpy arrays."""
    bin_edges_in = [0, 10, 20, 30]
    bin_edges_out = [5, 15, 25]
    expected = np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_empty_inputs():
    """Test with minimal valid inputs."""
    bin_edges_in = np.array([0, 10])
    bin_edges_out = np.array([5, 15])
    expected = np.array([[0.5]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_single_bin():
    """Test with single input and output bins."""
    bin_edges_in = np.array([0, 10])
    bin_edges_out = np.array([5, 15])
    expected = np.array([[0.5]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_edge_alignment():
    """Test when edges are perfectly aligned."""
    bin_edges_in = np.array([0, 10, 20, 30])
    bin_edges_out = np.array([0, 15, 30])
    expected = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_floating_point_precision():
    """Test with floating point values."""
    bin_edges_in = np.array([0.1, 0.3, 0.5])
    bin_edges_out = np.array([0.2, 0.4])
    expected = np.array([[0.5], [0.5]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_negative_values():
    """Test with negative values."""
    bin_edges_in = np.array([-30, -20, -10])
    bin_edges_out = np.array([-25, -15, -5])
    expected = np.array([[0.5, 0.0], [0.5, 0.5]])

    result = partial_isin(bin_edges_in, bin_edges_out)
    # Convert sparse matrix to dense for comparison
    if hasattr(result, "todense"):
        result = result.todense()
    assert_array_almost_equal(result, expected)


def test_invalid_inputs():
    """Test with invalid inputs for new interface."""
    # Test with unsorted input edges
    bin_edges_in = np.array([30, 20, 10])  # Descending order
    bin_edges_out = np.array([5, 15, 25])
    with pytest.raises(ValueError, match="bin_edges_in must be in ascending order"):
        partial_isin(bin_edges_in, bin_edges_out)

    # Test with unsorted output edges
    bin_edges_in = np.array([0, 10, 20])
    bin_edges_out = np.array([25, 15, 5])  # Descending order
    with pytest.raises(ValueError, match="bin_edges_out must be in ascending order"):
        partial_isin(bin_edges_in, bin_edges_out)


def test_combine_bin_series_basic():
    """Test basic functionality of combine_bin_series."""
    # Simple case: non-overlapping bins
    a = np.array([1.0, 2.0])
    a_edges = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 4.0])
    b_edges = np.array([1.0, 1.5, 2.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [0, 1, 1.5, 2]
    expected_edges = np.array([0.0, 1.0, 1.5, 2.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # Expected values for c: [1, 2, 2] (a[0] broadcasts to first bin, a[1] broadcasts to bins 2&3)
    # Expected values for d: [0, 3, 4] (b[0] broadcasts to second bin, b[1] broadcasts to third bin)
    expected_c = np.array([1.0, 2.0, 2.0])
    expected_d = np.array([0.0, 3.0, 4.0])
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_identical_edges():
    """Test combine_bin_series when both series have identical edges."""
    a = np.array([1.0, 2.0, 3.0])
    a_edges = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    b_edges = np.array([0.0, 1.0, 2.0, 3.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Edges should remain the same
    expected_edges = np.array([0.0, 1.0, 2.0, 3.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # Values should be preserved
    assert_array_almost_equal(c, a)
    assert_array_almost_equal(d, b)


def test_combine_bin_series_overlapping_bins():
    """Test combine_bin_series with overlapping bin structures."""
    a = np.array([10.0, 20.0])
    a_edges = np.array([0.0, 5.0, 10.0])
    b = np.array([30.0, 40.0])
    b_edges = np.array([2.0, 7.0, 12.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [0, 2, 5, 7, 10, 12]
    expected_edges = np.array([0.0, 2.0, 5.0, 7.0, 10.0, 12.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # Test that the values are broadcasted/repeated correctly
    # a[0]=10 covers [0,5]: broadcasts to bins [0,2] and [2,5]
    # a[1]=20 covers [5,10]: broadcasts to bins [5,7] and [7,10]
    # b[0]=30 covers [2,7]: broadcasts to bins [2,5] and [5,7]
    # b[1]=40 covers [7,12]: broadcasts to bins [7,10] and [10,12]
    expected_c = np.array([10.0, 10.0, 20.0, 20.0, 0.0])
    expected_d = np.array([0.0, 30.0, 30.0, 40.0, 40.0])
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_single_bins():
    """Test combine_bin_series with single bins."""
    a = np.array([5.0])
    a_edges = np.array([0.0, 2.0])
    b = np.array([10.0])
    b_edges = np.array([1.0, 3.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [0, 1, 2, 3]
    expected_edges = np.array([0.0, 1.0, 2.0, 3.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # a=5 covers [0,2]: broadcasts to [0,1] and [1,2]
    # b=10 covers [1,3]: broadcasts to [1,2] and [2,3]
    expected_c = np.array([5.0, 5.0, 0.0])
    expected_d = np.array([0.0, 10.0, 10.0])
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_nested_bins():
    """Test combine_bin_series where one series is nested within another."""
    a = np.array([100.0])
    a_edges = np.array([0.0, 10.0])
    b = np.array([20.0, 30.0])
    b_edges = np.array([2.0, 5.0, 8.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [0, 2, 5, 8, 10]
    expected_edges = np.array([0.0, 2.0, 5.0, 8.0, 10.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # a=100 covers [0,10]: broadcasts to all combined bins within its range
    # b[0]=20 covers [2,5] and b[1]=30 covers [5,8]
    expected_c = np.array([100.0, 100.0, 100.0, 100.0])
    expected_d = np.array([0.0, 20.0, 30.0, 0.0])
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_non_overlapping():
    """Test combine_bin_series with completely non-overlapping bins."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 4.0])
    b_edges = np.array([3.0, 4.0, 5.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [0, 1, 2, 3, 4, 5]
    expected_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # a maps to first two bins, b maps to last two bins
    expected_c = np.array([1.0, 2.0, 0.0, 0.0, 0.0])
    expected_d = np.array([0.0, 0.0, 0.0, 3.0, 4.0])
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_zero_values():
    """Test combine_bin_series with zero values."""
    a = np.array([0.0, 5.0])
    a_edges = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 0.0])
    b_edges = np.array([0.5, 1.5, 2.5])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [0, 0.5, 1, 1.5, 2, 2.5]
    expected_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # Check that zero values are preserved and broadcasted correctly
    expected_c = np.array([0.0, 0.0, 5.0, 5.0, 0.0])
    expected_d = np.array([0.0, 3.0, 3.0, 0.0, 0.0])
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_floating_point():
    """Test combine_bin_series with floating point precision."""
    a = np.array([1.1, 2.2])
    a_edges = np.array([0.1, 1.1, 2.1])
    b = np.array([3.3, 4.4])
    b_edges = np.array([0.6, 1.6, 2.6])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [0.1, 0.6, 1.1, 1.6, 2.1, 2.6]
    expected_edges = np.array([0.1, 0.6, 1.1, 1.6, 2.1, 2.6])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # Test with appropriate precision and broadcasting
    expected_c = np.array([1.1, 1.1, 2.2, 2.2, 0.0])
    expected_d = np.array([0.0, 3.3, 3.3, 4.4, 4.4])
    assert_array_almost_equal(c, expected_c, decimal=10)
    assert_array_almost_equal(d, expected_d, decimal=10)


def test_combine_bin_series_input_validation():
    """Test input validation for combine_bin_series."""
    # Test mismatched array lengths
    a = np.array([1.0, 2.0])
    a_edges = np.array([0.0, 1.0])  # Should have 3 elements for 2 bins
    b = np.array([3.0])
    b_edges = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="a_edges must have len\\(a\\) \\+ 1 elements"):
        combine_bin_series(a, a_edges, b, b_edges)

    # Test mismatched b array lengths
    a = np.array([1.0])
    a_edges = np.array([0.0, 1.0])
    b = np.array([3.0, 4.0])
    b_edges = np.array([1.0, 2.0])  # Should have 3 elements for 2 bins

    with pytest.raises(ValueError, match="b_edges must have len\\(b\\) \\+ 1 elements"):
        combine_bin_series(a, a_edges, b, b_edges)


def test_combine_bin_series_list_inputs():
    """Test combine_bin_series with list inputs."""
    a = [1.0, 2.0]
    a_edges = [0.0, 1.0, 2.0]
    b = [3.0, 4.0]
    b_edges = [1.5, 2.5, 3.5]

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Should work with list inputs and return numpy arrays
    assert isinstance(c, np.ndarray)
    assert isinstance(c_edges, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(d_edges, np.ndarray)

    # Expected combined edges: [0, 1, 1.5, 2, 2.5, 3.5]
    expected_edges = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.5])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)


def test_combine_bin_series_negative_values():
    """Test combine_bin_series with negative values."""
    a = np.array([-5.0, -2.0])
    a_edges = np.array([-10.0, -3.0, 0.0])
    b = np.array([1.0, 4.0])
    b_edges = np.array([-1.0, 2.0, 5.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [-10, -3, -1, 0, 2, 5]
    expected_edges = np.array([-10.0, -3.0, -1.0, 0.0, 2.0, 5.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # Test correct mapping with negative values and broadcasting
    expected_c = np.array([-5.0, -2.0, -2.0, 0.0, 0.0])
    expected_d = np.array([0.0, 0.0, 1.0, 1.0, 4.0])
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_empty_arrays():
    """Test combine_bin_series with minimal valid inputs."""
    a = np.array([42.0])
    a_edges = np.array([0.0, 1.0])
    b = np.array([24.0])
    b_edges = np.array([0.5, 1.5])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges)

    # Expected combined edges: [0, 0.5, 1, 1.5]
    expected_edges = np.array([0.0, 0.5, 1.0, 1.5])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    expected_c = np.array([42.0, 42.0, 0.0])
    expected_d = np.array([0.0, 24.0, 24.0])
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_extrapolation_nearest():
    """Test combine_bin_series with nearest extrapolation."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    b_edges = np.array([2.0, 3.0, 4.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges, extrapolation="nearest")

    # Expected combined edges: [1, 2, 3, 4]
    expected_edges = np.array([1.0, 2.0, 3.0, 4.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # With nearest extrapolation:
    # c extends to all bins using nearest values
    # d extends to all bins using nearest values
    expected_c = np.array([1.0, 2.0, 2.0])  # nearest for [3,4] is a[1]=2.0
    expected_d = np.array([10.0, 10.0, 20.0])  # nearest for [1,2] is b[0]=10.0
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_extrapolation_nan():
    """Test combine_bin_series with nan extrapolation."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    b_edges = np.array([2.0, 3.0, 4.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges, extrapolation=np.nan)

    # Expected combined edges: [1, 2, 3, 4]
    expected_edges = np.array([1.0, 2.0, 3.0, 4.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # With nan extrapolation:
    # Out-of-range bins get nan values
    expected_c = np.array([1.0, 2.0, np.nan])  # [3,4] is out of range for a
    expected_d = np.array([np.nan, 10.0, 20.0])  # [1,2] is out of range for b
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_extrapolation_custom_value():
    """Test combine_bin_series with custom extrapolation value."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    b_edges = np.array([2.0, 3.0, 4.0])

    c, c_edges, d, d_edges = combine_bin_series(a, a_edges, b, b_edges, extrapolation=-999.0)

    # Expected combined edges: [1, 2, 3, 4]
    expected_edges = np.array([1.0, 2.0, 3.0, 4.0])
    assert_array_almost_equal(c_edges, expected_edges)
    assert_array_almost_equal(d_edges, expected_edges)

    # With custom extrapolation value:
    # Out-of-range bins get the custom value
    expected_c = np.array([1.0, 2.0, -999.0])  # [3,4] is out of range for a
    expected_d = np.array([-999.0, 10.0, 20.0])  # [1,2] is out of range for b
    assert_array_almost_equal(c, expected_c)
    assert_array_almost_equal(d, expected_d)


def test_combine_bin_series_extrapolation_default_behavior():
    """Test that default extrapolation preserves backwards compatibility."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    b_edges = np.array([2.0, 3.0, 4.0])

    # Default behavior should be equivalent to extrapolation=0.0
    c1, c_edges1, d1, d_edges1 = combine_bin_series(a, a_edges, b, b_edges)
    c2, c_edges2, d2, d_edges2 = combine_bin_series(a, a_edges, b, b_edges, extrapolation=0.0)

    assert_array_almost_equal(c1, c2)
    assert_array_almost_equal(d1, d2)
    assert_array_almost_equal(c_edges1, c_edges2)
    assert_array_almost_equal(d_edges1, d_edges2)


def test_combine_bin_series_extrapolation_no_out_of_range():
    """Test extrapolation when there are no out-of-range bins."""
    a = np.array([1.0, 2.0])
    a_edges = np.array([0.0, 1.0, 2.0])
    b = np.array([3.0, 4.0])
    b_edges = np.array([0.0, 1.0, 2.0])

    # When series have identical ranges, extrapolation method shouldn't matter
    c1, _, d1, _ = combine_bin_series(a, a_edges, b, b_edges, extrapolation="nearest")
    c2, _, d2, _ = combine_bin_series(a, a_edges, b, b_edges, extrapolation=np.nan)
    c3, _, d3, _ = combine_bin_series(a, a_edges, b, b_edges, extrapolation=0.0)

    assert_array_almost_equal(c1, c2)
    assert_array_almost_equal(c1, c3)
    assert_array_almost_equal(d1, d2)
    assert_array_almost_equal(d1, d3)


@pytest.mark.parametrize("station_number", [260, 273, 286, 323])
def test_get_soil_temperature_valid_stations(station_number):
    """Test get_soil_temperature for all valid station numbers."""
    df = get_soil_temperature(station_number)

    # Check that result is a pandas DataFrame
    assert isinstance(df, pd.DataFrame)

    # Check that index is a DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)

    # Check that DataFrame is not empty
    assert len(df) > 0

    # Expected columns based on the docstring
    expected_columns = {"TB1", "TB2", "TB3", "TB4", "TB5", "TNB1", "TNB2", "TXB1", "TXB2"}

    # Check that all expected columns are present
    assert expected_columns.issubset(set(df.columns))


def test_get_soil_temperature_column_data_quality():
    """Test that each column has at least some non-NaN values."""
    df = get_soil_temperature(260)  # Use default station

    expected_columns = ["TB1", "TB2", "TB3", "TB4", "TB5", "TNB1", "TNB2", "TXB1", "TXB2"]

    # Check that each column has at least some non-NaN values
    for col in expected_columns:
        assert col in df.columns, f"Column {col} not found in DataFrame"
        non_nan_count = df[col].notna().sum()
        assert non_nan_count > 0, f"Column {col} has no non-NaN values"
        # Also check that we have a reasonable amount of data (at least 10 data points)
        assert non_nan_count >= 10, f"Column {col} has fewer than 10 non-NaN values ({non_nan_count})"


def test_get_soil_temperature_default_station():
    """Test that default station parameter works."""
    df = get_soil_temperature()

    # Should return data (default is station 260)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_get_soil_temperature_data_types():
    """Test that soil temperature data has correct data types and reasonable values."""
    df = get_soil_temperature(260)

    # All temperature columns should be numeric
    temp_columns = ["TB1", "TB2", "TB3", "TB4", "TB5", "TNB1", "TNB2", "TXB1", "TXB2"]
    for col in temp_columns:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} should be numeric"

        # Check for reasonable temperature ranges (in Celsius, should be between -50 and +50)
        valid_data = df[col].dropna()
        if len(valid_data) > 0:
            assert valid_data.min() >= -50, f"Column {col} has unreasonably low temperature: {valid_data.min()}"
            assert valid_data.max() <= 50, f"Column {col} has unreasonably high temperature: {valid_data.max()}"


def test_get_soil_temperature_index_properties():
    """Test that the DataFrame index has correct timezone and is sorted."""
    df = get_soil_temperature(260)

    # Index should be timezone-aware (UTC)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"

    # Index should be sorted
    assert df.index.is_monotonic_increasing


def test_get_soil_temperature_invalid_station():
    """Test that invalid station numbers raise appropriate errors."""
    # Test with an invalid station number
    with pytest.raises((ValueError, Exception)):
        get_soil_temperature(999)  # Invalid station number

    # Test with a station that might not have data or causes HTTP errors
    with pytest.raises((ValueError, Exception)):
        get_soil_temperature(123)  # Invalid station number
