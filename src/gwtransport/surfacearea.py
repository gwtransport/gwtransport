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
    # Extract corners (y-coordinates) for bounds checking
    y_tl = y_edges[:-1, :-1]  # top-left
    y_tr = y_edges[:-1, 1:]  # top-right
    y_bl = y_edges[1:, :-1]  # bottom-left
    y_br = y_edges[1:, 1:]  # bottom-right

    # Calculate widths
    widths = (x_edges[1:] - x_edges[:-1])[np.newaxis, :]

    # Use Sutherland-Hodgman polygon clipping algorithm
    # This is the minimal fix: replace the flawed edge crossing logic with proper clipping
    areas = np.zeros((y_edges.shape[0] - 1, y_edges.shape[1] - 1), dtype=float)

    for i in range(areas.shape[0]):
        for j in range(areas.shape[1]):
            # Original trapezoid corners
            corners = np.array([
                [x_edges[j], y_edges[i + 1, j]],  # bottom-left
                [x_edges[j + 1], y_edges[i + 1, j + 1]],  # bottom-right
                [x_edges[j + 1], y_edges[i, j + 1]],  # top-right
                [x_edges[j], y_edges[i, j]],  # top-left
            ])

            # Clip against y_lower
            clipped = []
            for k in range(len(corners)):
                curr = corners[k]
                prev = corners[k - 1]

                if curr[1] >= y_lower:  # current point is inside
                    if prev[1] < y_lower:  # previous was outside, add intersection
                        t = (y_lower - prev[1]) / (curr[1] - prev[1])
                        clipped.append([prev[0] + t * (curr[0] - prev[0]), y_lower])
                    clipped.append(curr)
                elif prev[1] >= y_lower:  # current is outside, previous inside, add intersection
                    t = (y_lower - prev[1]) / (curr[1] - prev[1])
                    clipped.append([prev[0] + t * (curr[0] - prev[0]), y_lower])

            if not clipped:
                areas[i, j] = 0
                continue

            corners = np.array(clipped)

            # Clip against y_upper
            clipped = []
            for k in range(len(corners)):
                curr = corners[k]
                prev = corners[k - 1]

                if curr[1] <= y_upper:  # current point is inside
                    if prev[1] > y_upper:  # previous was outside, add intersection
                        t = (y_upper - prev[1]) / (curr[1] - prev[1])
                        clipped.append([prev[0] + t * (curr[0] - prev[0]), y_upper])
                    clipped.append(curr)
                elif prev[1] <= y_upper:  # current is outside, previous inside, add intersection
                    t = (y_upper - prev[1]) / (curr[1] - prev[1])
                    clipped.append([prev[0] + t * (curr[0] - prev[0]), y_upper])

            # Need at least 3 vertices to form a polygon
            min_vertices = 3
            if len(clipped) < min_vertices:
                areas[i, j] = 0
                continue

            # Calculate area using shoelace formula
            vertices = np.array(clipped)
            n = len(vertices)
            area = 0.5 * abs(
                sum(
                    vertices[k, 0] * vertices[(k + 1) % n, 1] - vertices[(k + 1) % n, 0] * vertices[k, 1]
                    for k in range(n)
                )
            )
            areas[i, j] = area

    # Calculate average heights
    avg_heights = areas / widths

    # Set to 0 where trapezoid is completely outside bounds
    all_above = np.all(np.stack([y_tl, y_tr, y_bl, y_br], axis=-1) >= y_upper, axis=-1)
    all_below = np.all(np.stack([y_tl, y_tr, y_bl, y_br], axis=-1) <= y_lower, axis=-1)
    avg_heights[all_above | all_below] = 0

    return avg_heights
