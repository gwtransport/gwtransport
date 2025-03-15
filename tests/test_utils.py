import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from gwtransport1d.utils import diff, linear_average, linear_interpolate


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

    expected = np.array([2, 2])  # Average is constant
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_linear_function():
    """Test average of linear function y=x."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 2, 3, 4])
    x_edges = np.array([0, 2, 4])

    # Average of y=x from 0 to 2 = 1
    # Average of y=x from 2 to 4 = 3
    expected = np.array([1, 3])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_piecewise_linear():
    """Test average of piecewise linear function."""
    x_data = np.array([0, 1, 2, 3])
    y_data = np.array([0, 1, 1, 0])
    x_edges = np.array([0, 1.5, 3])

    # Integral from 0 to 1.5 = 1, width = 1.5 → average = 2/3
    # Integral from 1.5 to 3 = 1, width = 1.5 → average = 2/3
    expected = np.array([1.0 / 1.5, 1.0 / 1.5])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_edges_beyond_data():
    """Test averages with edges outside the data range."""
    x_data = np.array([1, 2, 3])
    y_data = np.array([1, 2, 3])
    x_edges = np.array([0, 4])

    # Extrapolation should extend the first and last segments
    # Average of y=x from 0 to 4 = 2
    expected = np.array([2])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_edges_matching_data():
    """Test when edges exactly match data points."""
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 1, 4, 9, 16])
    x_edges = np.array([1, 3])

    # Average under the curve from 1 to 3 = 4.5
    expected = np.array([4.5])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_multiple_edge_intervals():
    """Test with multiple averaging intervals."""
    x_data = np.array([0, 1, 2, 3, 4, 5])
    y_data = np.array([0, 1, 4, 9, 16, 25])
    x_edges = np.array([0, 1, 2, 3, 4, 5])

    # Average of each segment
    expected = np.array([0.5, 2.5, 6.5, 12.5, 20.5])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_empty_interval():
    """Test averaging over an empty interval (edges are the same)."""
    x_data = np.array([0, 1, 2, 3])
    y_data = np.array([0, 1, 4, 9])
    x_edges = np.array([0, 1, 1, 2])

    # Second interval has zero width at x=1, so average should be y=1
    expected = np.array([0.5, 1.0, 2.5])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_input_validation():
    """Test input validation."""
    # Test unequal lengths of x_data and y_data
    with pytest.raises(ValueError, match="x_data and y_data must have the same length"):
        linear_average([0, 1], [0], [0, 1])

    # Test x_edges too short
    with pytest.raises(ValueError, match="x_edges must contain at least 2 values"):
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
    expected = np.array([1.5, 1.5])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_edge_case_numerical_precision():
    """Test numerical precision for very close x values."""
    x_data = np.array([0, 1e-10, 1])
    y_data = np.array([0, 1e-10, 1])
    x_edges = np.array([0, 0.5, 1])

    # For a linear function y=x, the average from 0 to 0.5 is 0.25
    # and from 0.5 to 1 is 0.75
    expected = np.array([0.25, 0.75])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_single_point_data():
    """Test with a single data point - should extrapolate as constant."""
    x_data = np.array([1])
    y_data = np.array([5])
    x_edges = np.array([0, 2])

    # Single point should be treated as constant value
    expected = np.array([5])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)


def test_zero_width_interval_edge_case():
    """Test handling of a zero-width interval at the edge."""
    x_data = np.array([0, 1, 2])
    y_data = np.array([0, 1, 2])
    x_edges = np.array([0, 0, 1])

    # First interval has zero width at x=0, so average should be y=0
    # Second interval is 0 to 1, average is 0.5
    expected = np.array([0.0, 0.5])
    result = linear_average(x_data, y_data, x_edges)

    np.testing.assert_allclose(result, expected)
