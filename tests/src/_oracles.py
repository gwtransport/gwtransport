"""Independent test-support oracles for gwtransport.

This module holds reference implementations that the test suite uses to cross-check the
package's production code, but that are not themselves part of the public ``gwtransport`` API.
Keeping them here (rather than in ``src/gwtransport``) means the shipped package carries no
runtime code that only the tests exercise, while the tests retain an independent oracle to
validate against.

The underscore prefix keeps pytest from collecting this module as a test file. It is importable
by bare name from any test because ``tests/src`` is placed on ``sys.path`` by
``tests/src/conftest.py`` (mirroring ``_radial_asr_fv_oracle``).

Oracles
-------
- :func:`partial_isin` -- fraction of each input bin overlapping each output bin. An independent
  bin-overlap reference for the advection weight builder and the diffusion_fast band checks.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def partial_isin(*, bin_edges_in: npt.ArrayLike, bin_edges_out: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """
    Calculate the fraction of each input bin that overlaps with each output bin.

    This function computes a matrix where element (i, j) represents the fraction
    of input bin i that overlaps with output bin j. The computation uses
    vectorized operations to avoid loops.

    Parameters
    ----------
    bin_edges_in : array-like
        1D array of input bin edges in ascending order. For n_in bins, there
        should be n_in+1 edges.
    bin_edges_out : array-like
        1D array of output bin edges in ascending order. For n_out bins, there
        should be n_out+1 edges.

    Returns
    -------
    overlap_matrix : ndarray
        Dense matrix of shape (n_in, n_out) where n_in is the number of input
        bins and n_out is the number of output bins. Each element (i, j)
        represents the fraction of input bin i that overlaps with output bin j.
        Values range from 0 (no overlap) to 1 (complete overlap).

    Raises
    ------
    ValueError
        If ``bin_edges_in`` or ``bin_edges_out`` are not 1D arrays. If either edge
        array has fewer than 2 elements. If ``bin_edges_in`` or ``bin_edges_out``
        are not in ascending order.

    Notes
    -----
    - Both input arrays must be sorted in ascending order
    - The function leverages the sorted nature of both inputs for efficiency
    - Uses vectorized operations to handle large bin arrays efficiently
    - All overlaps sum to 1.0 for each input bin when output bins fully cover input range

    Examples
    --------
    >>> import numpy as np
    >>> from _oracles import partial_isin
    >>> bin_edges_in = np.array([0, 10, 20, 30])
    >>> bin_edges_out = np.array([5, 15, 25])
    >>> partial_isin(
    ...     bin_edges_in=bin_edges_in, bin_edges_out=bin_edges_out
    ... )  # doctest: +NORMALIZE_WHITESPACE
    array([[0.5, 0. ],
           [0.5, 0.5],
           [0. , 0.5]])
    """
    # Convert inputs to numpy arrays
    bin_edges_in = np.asarray(bin_edges_in, dtype=float)
    bin_edges_out = np.asarray(bin_edges_out, dtype=float)

    # Validate inputs
    if bin_edges_in.ndim != 1 or bin_edges_out.ndim != 1:
        msg = "Both bin_edges_in and bin_edges_out must be 1D arrays"
        raise ValueError(msg)
    if len(bin_edges_in) < 2 or len(bin_edges_out) < 2:  # noqa: PLR2004
        msg = "Both edge arrays must have at least 2 elements"
        raise ValueError(msg)

    # Edges must be non-decreasing (ignoring NaN). Zero-width input bins are
    # allowed -- they arise e.g. from cumulative flow with zero-flow intervals --
    # and produce zero overlap fractions (no water passed through).
    diffs_in = np.diff(bin_edges_in)
    valid_diffs_in = ~np.isnan(diffs_in)
    if np.any(valid_diffs_in) and not np.all(diffs_in[valid_diffs_in] >= 0):
        msg = "bin_edges_in must be non-decreasing"
        raise ValueError(msg)

    diffs_out = np.diff(bin_edges_out)
    valid_diffs_out = ~np.isnan(diffs_out)
    if np.any(valid_diffs_out) and not np.all(diffs_out[valid_diffs_out] >= 0):
        msg = "bin_edges_out must be non-decreasing"
        raise ValueError(msg)

    # Build matrix using fully vectorized approach
    # Create meshgrids for all possible input-output bin combinations
    in_left = bin_edges_in[:-1, None]  # Shape: (n_bins_in, 1)
    in_right = bin_edges_in[1:, None]  # Shape: (n_bins_in, 1)
    in_width = diffs_in[:, None]  # Shape: (n_bins_in, 1); reuses the validated np.diff above

    out_left = bin_edges_out[None, :-1]  # Shape: (1, n_bins_out)
    out_right = bin_edges_out[None, 1:]  # Shape: (1, n_bins_out)

    # Calculate overlaps for all combinations using broadcasting
    overlap_left = np.maximum(in_left, out_left)  # Shape: (n_bins_in, n_bins_out)
    overlap_right = np.minimum(in_right, out_right)  # Shape: (n_bins_in, n_bins_out)

    # Calculate overlap widths (zero where no overlap)
    overlap_widths = np.maximum(0, overlap_right - overlap_left)

    # Zero-width input bins contribute no overlap (division-safe). NaN widths still
    # propagate as NaN fractions (preserved for spin-up handling).
    with np.errstate(divide="ignore", invalid="ignore"):
        result = overlap_widths / in_width
    return np.where(in_width == 0, 0.0, result)
