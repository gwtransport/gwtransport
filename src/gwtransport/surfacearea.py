import numpy as np


def compute_average_heights(x_edges, y_edges, y_lower, y_upper):
    """
    Compute average heights of clipped trapezoids.

    Trapezoids have vertical left and right sides, with corners at:
    - top-left: (y_edges[i, j], x_edges[j])
    - top-right: (y_edges[i, j+1], x_edges[j+1])
    - bottom-left: (y_edges[i+1, j], x_edges[j])
    - bottom-right: (y_edges[i+1, j+1], x_edges[j+1])

    Parameters
    ----------
    x_edges : ndarray
        1D array of x coordinates, shape (n_x,)
    y_edges : ndarray
        2D array of y coordinates, shape (n_y, n_x)
    y_lower : float
        Lower horizontal clipping bound
    y_upper : float
        Upper horizontal clipping bound

    Returns
    -------
    avg_heights : ndarray
        2D array of average heights (area/width) for each clipped trapezoid,
        shape (n_y-1, n_x-1)
    """
    # Clip y_edges before extracting corners
    y_edges_clip = np.clip(y_edges, y_lower, y_upper)

    # Extract corners (y-coordinates) - both original and clipped
    y_tl = y_edges[:-1, :-1]  # top-left
    y_tr = y_edges[:-1, 1:]  # top-right
    y_bl = y_edges[1:, :-1]  # bottom-left
    y_br = y_edges[1:, 1:]  # bottom-right

    y_tl_clip = y_edges_clip[:-1, :-1]  # top-left clipped
    y_tr_clip = y_edges_clip[:-1, 1:]  # top-right clipped
    y_bl_clip = y_edges_clip[1:, :-1]  # bottom-left clipped
    y_br_clip = y_edges_clip[1:, 1:]  # bottom-right clipped

    # Calculate widths
    widths = (x_edges[1:] - x_edges[:-1])[np.newaxis, :]

    # Default: standard trapezoid formula
    left_height = y_tl_clip - y_bl_clip
    right_height = y_tr_clip - y_br_clip
    areas = 0.5 * (left_height + right_height) * widths

    # Detect edge crossings vectorized
    edge_data = [
        # (left_y, right_y, threshold, heights_for_interpolation)
        (y_tl, y_tr, y_upper, (right_height, left_height)),
        (y_bl, y_br, y_lower, (right_height, left_height)),
        (y_tl, y_tr, y_lower, (y_br_clip - y_lower, y_bl_clip - y_lower)),
        (y_bl, y_br, y_upper, (y_upper - y_tr_clip, y_upper - y_tl_clip)),
    ]

    for left_y, right_y, threshold, (h_right, h_left) in edge_data:
        mask = (left_y > threshold) != (right_y > threshold)
        t = (threshold - left_y) / (right_y - left_y)
        areas = np.where(mask, 0.5 * widths * (t * h_right + (1 - t) * h_left), areas)

    # Calculate average heights
    avg_heights = areas / widths

    # Set to 0 where trapezoid is completely outside bounds
    all_above = np.all(np.stack([y_tl, y_tr, y_bl, y_br], axis=-1) >= y_upper, axis=-1)
    all_below = np.all(np.stack([y_tl, y_tr, y_bl, y_br], axis=-1) <= y_lower, axis=-1)
    avg_heights[all_above | all_below] = 0

    return avg_heights
