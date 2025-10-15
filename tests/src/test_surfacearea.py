import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from gwtransport.surfacearea import compute_average_heights


def test_rectangle_no_clipping():
    """Test rectangle with no clipping - analytical solution."""
    # Rectangle: width=2, height=3
    x_edges = np.array([0, 2])
    y_edges = np.array([[3, 3], [0, 0]])

    # No clipping bounds
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=-1, y_upper=4)

    # Analytical: area = width * height = 2 * 3 = 6, avg_height = area/width = 3
    assert_almost_equal(avg_heights[0, 0], 3.0)


def test_rectangle_full_clipping():
    """Test rectangle completely clipped out."""
    x_edges = np.array([0, 2])
    y_edges = np.array([[3, 3], [0, 0]])

    # Clip completely above
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=4, y_upper=5)
    assert_almost_equal(avg_heights[0, 0], 0.0)

    # Clip completely below
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=-2, y_upper=-1)
    assert_almost_equal(avg_heights[0, 0], 0.0)


def test_rectangle_partial_clipping():
    """Test rectangle with partial clipping - analytical solution."""
    # Rectangle: width=2, height=4 (y from 0 to 4)
    x_edges = np.array([0, 2])
    y_edges = np.array([[4, 4], [0, 0]])

    # Clip bottom half: keep y from 2 to 4
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=2, y_upper=5)

    # Analytical: clipped area = width * clipped_height = 2 * 2 = 4
    # avg_height = area/width = 4/2 = 2
    assert_almost_equal(avg_heights[0, 0], 2.0)

    # Clip top half: keep y from 0 to 2
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=-1, y_upper=2)
    assert_almost_equal(avg_heights[0, 0], 2.0)


def test_trapezoid_no_clipping():
    """Test trapezoid with analytical solution."""
    # Trapezoid: left height=4, right height=2, width=3
    x_edges = np.array([0, 3])
    y_edges = np.array([[4, 2], [0, 0]])

    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=-1, y_upper=5)

    # Analytical: trapezoid area = 0.5 * (left + right) * width = 0.5 * (4 + 2) * 3 = 9
    # avg_height = area/width = 9/3 = 3
    assert_almost_equal(avg_heights[0, 0], 3.0)


def test_trapezoid_with_clipping():
    """Test trapezoid with clipping - analytical solution."""
    # Trapezoid: left height=6, right height=4, width=2
    x_edges = np.array([0, 2])
    y_edges = np.array([[6, 4], [0, 0]])

    # Clip to keep y from 1 to 5
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=1, y_upper=5)

    # Analytical: The original trapezoid has a sloped top edge from (0,6) to (2,4)
    # When clipped at y=5, this creates a pentagonal region with area = 7.5
    # Average height = 7.5 / 2 = 3.75
    assert_almost_equal(avg_heights[0, 0], 3.75)


def test_triangle_cases():
    """Test edge crossing cases that form triangles."""
    # Trapezoid where top edge crosses clipping bound
    x_edges = np.array([0, 2])
    y_edges = np.array([[3, 1], [0, 0]])  # Top slopes down from 3 to 1

    # Clip at y=2 - top edge crosses this bound
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=-1, y_upper=2)

    # This should form a pentagon/triangle after clipping
    # Exact analytical solution is complex, but should be > 0 and < full height
    assert avg_heights[0, 0] > 0
    assert avg_heights[0, 0] < 2.0  # Less than average of full trapezoid


def test_multiple_quads():
    """Test grid with multiple quadrilaterals."""
    x_edges = np.array([0, 1, 2])
    y_edges = np.array([[2, 2, 2], [1, 1, 1], [0, 0, 0]])

    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=0.5, y_upper=1.5)

    # Top row: quads from y=1 to y=2, clipped to y=1 to y=1.5 (height=0.5)
    # Bottom row: quads from y=0 to y=1, clipped to y=0.5 to y=1 (height=0.5)
    expected = 0.5 * np.ones((2, 2))
    assert_array_almost_equal(avg_heights, expected)


def test_zero_width_handling():
    """Test handling of zero-width quadrilaterals."""
    x_edges = np.array([0, 0, 2])  # Zero width, then width=2
    y_edges = np.array([[2, 2, 2], [0, 0, 0]])

    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=-1, y_upper=3)

    # First column: zero width produces NaN (area/0), check if properly handled
    assert np.isnan(avg_heights[0, 0]) or avg_heights[0, 0] == 0
    # Second column: normal rectangle
    assert_almost_equal(avg_heights[0, 1], 2.0)


def test_edge_case_bounds():
    """Test when clipping bounds exactly match quad boundaries."""
    x_edges = np.array([0, 1])
    y_edges = np.array([[2, 2], [0, 0]])

    # Bounds exactly match quad
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=0, y_upper=2)
    assert_almost_equal(avg_heights[0, 0], 2.0)

    # Lower bound exactly at bottom
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=0, y_upper=3)
    assert_almost_equal(avg_heights[0, 0], 2.0)

    # Upper bound exactly at top
    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=-1, y_upper=2)
    assert_almost_equal(avg_heights[0, 0], 2.0)


def test_sloped_quad():
    """Test quadrilateral with all corners at different heights."""
    x_edges = np.array([0, 2])
    y_edges = np.array([[3, 4], [1, 2]])  # All corners different

    avg_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=-1, y_upper=5)

    # Analytical: corners at (0,3), (2,4), (0,1), (2,2)
    # This is a general quadrilateral, area = 0.5 * |shoelace formula|
    # Using shoelace: area = 0.5 * |x1(y2-y4) + x2(y3-y1) + x3(y4-y2) + x4(y1-y3)|
    # With corners: (0,1), (2,2), (2,4), (0,3)
    # area = 0.5 * |0*(2-3) + 2*(4-1) + 2*(3-2) + 0*(1-4)| = 0.5 * |0 + 6 + 2 + 0| = 4
    # avg_height = area/width = 4/2 = 2
    assert_almost_equal(avg_heights[0, 0], 2.0)


def test_conservation_property():
    """Test that clipping conserves total area when bounds are expanded."""
    x_edges = np.array([0, 1])
    y_edges = np.array([[3, 3], [1, 1]])

    # Full area
    full_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=0, y_upper=4)

    # Split into two parts
    lower_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=0, y_upper=2)
    upper_heights = compute_average_heights(x_edges=x_edges, y_edges=y_edges, y_lower=2, y_upper=4)

    # Conservation: lower + upper should equal full
    assert_almost_equal(lower_heights[0, 0] + upper_heights[0, 0], full_heights[0, 0])
