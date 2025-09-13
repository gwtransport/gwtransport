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
    # Extract corners (y-coordinates)
    y_tl = y_edges[:-1, :-1]  # top-left
    y_tr = y_edges[:-1, 1:]  # top-right
    y_bl = y_edges[1:, :-1]  # bottom-left
    y_br = y_edges[1:, 1:]  # bottom-right

    # Calculate widths
    widths = (x_edges[1:] - x_edges[:-1])[np.newaxis, :]

    # Clip y-coordinates
    y_tl_clip = np.clip(y_tl, y_lower, y_upper)
    y_tr_clip = np.clip(y_tr, y_lower, y_upper)
    y_bl_clip = np.clip(y_bl, y_lower, y_upper)
    y_br_clip = np.clip(y_br, y_lower, y_upper)

    # Default: standard trapezoid formula
    left_height = y_tl_clip - y_bl_clip
    right_height = y_tr_clip - y_br_clip
    areas = 0.5 * (left_height + right_height) * widths

    # Detect crossing conditions and calculate interpolation parameters
    # Top edge crosses y_upper
    mask = (y_tl > y_upper) != (y_tr > y_upper)
    t = (y_upper - y_tl) / (y_tr - y_tl)
    areas = np.where(mask, 0.5 * widths * (t * right_height + (1 - t) * left_height), areas)

    # Bottom edge crosses y_lower
    mask = (y_bl < y_lower) != (y_br < y_lower)
    t = (y_lower - y_bl) / (y_br - y_bl)
    areas = np.where(mask, 0.5 * widths * (t * right_height + (1 - t) * left_height), areas)

    # Top edge crosses y_lower (small triangle at bottom)
    mask = (y_tl < y_lower) != (y_tr < y_lower)
    t = (y_lower - y_tl) / (y_tr - y_tl)
    areas = np.where(mask, 0.5 * widths * (t * (y_br_clip - y_lower) + (1 - t) * (y_bl_clip - y_lower)), areas)

    # Bottom edge crosses y_upper (small triangle at top)
    mask = (y_bl > y_upper) != (y_br > y_upper)
    t = (y_upper - y_bl) / (y_br - y_bl)
    areas = np.where(mask, 0.5 * widths * (t * (y_upper - y_tr_clip) + (1 - t) * (y_upper - y_tl_clip)), areas)

    # Calculate average heights
    avg_heights = areas / widths

    # Set to 0 where trapezoid is completely outside bounds
    completely_outside = ((y_tl >= y_upper) & (y_tr >= y_upper) & (y_bl >= y_upper) & (y_br >= y_upper)) | (
        (y_tl <= y_lower) & (y_tr <= y_lower) & (y_bl <= y_lower) & (y_br <= y_lower)
    )
    avg_heights[completely_outside] = 0

    return avg_heights
