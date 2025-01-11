"""General utilities for the 1D groundwater transport model."""
import numpy as np


def linear_interpolate(x_ref, y_ref, x_query):
    """
    Linear interpolation on monotonically increasing data.

    Parameters
    ----------
    x_ref : array-like
        Reference x-values.
    y_ref : array-like
        Reference y-values.
    x_query : array-like
        Query x-values.

    Returns
    -------
    array
        Interpolated y-values.
    """
    x_ref = np.asarray(x_ref)
    y_ref = np.asarray(y_ref)
    x_query = np.asarray(x_query)

    # Find indices where x_query would be inserted in x_ref
    idx = np.searchsorted(x_ref, x_query)

    # Handle edge cases
    idx = np.clip(idx, 1, len(x_ref) - 1)

    # Calculate interpolation weights
    x0 = x_ref[idx - 1]
    x1 = x_ref[idx]
    y0 = y_ref[idx - 1]
    y1 = y_ref[idx]

    # Perform linear interpolation
    weights = (x_query - x0) / (x1 - x0)
    return y0 + weights * (y1 - y0)
