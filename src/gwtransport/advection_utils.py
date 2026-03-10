"""
Private helper functions for advective transport modeling.

This module contains internal helper functions used by the advection module.
These functions implement various algorithms for computing transport weights
and handling nonlinear sorption.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.residence_time import residence_time
from gwtransport.utils import partial_isin


def _infiltration_to_extraction_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.floating]:
    """
    Compute normalized weights for linear infiltration to extraction transformation.

    This helper function computes the weight matrix for constant retardation factor.
    It handles the main advective transport calculation with flow-weighted averaging.

    The resulting cout values represent volume-weighted (flow-weighted) bin averages,
    where periods with higher infiltration flow rates contribute more to the output concentration.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for infiltration bins.
    cout_tedges : pandas.DatetimeIndex
        Time edges for extraction bins.
    aquifer_pore_volumes : array-like
        Distribution of pore volumes [m3].
    flow : array-like
        Flow rate values [m3/day].
    retardation_factor : float
        Constant retardation factor.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix. Shape: (len(cout_tedges) - 1, len(tedges) - 1)
    """
    # Convert time edges to days
    cin_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # Pre-compute all residence times and infiltration edges
    rt_edges_2d = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    infiltration_tedges_2d = cout_tedges_days[None, :] - rt_edges_2d

    # Pre-compute valid bins
    valid_bins_2d = ~(np.isnan(infiltration_tedges_2d[:, :-1]) | np.isnan(infiltration_tedges_2d[:, 1:]))

    # Pre-compute cin time range for clip optimization (computed once, used n_bins times)
    cin_time_min = cin_tedges_days[0]
    cin_time_max = cin_tedges_days[-1]

    # Accumulate flow-weighted overlap matrices from all pore volumes
    # Each pore volume has equal probability (equal-mass bins from gamma distribution)
    accumulated_weights = np.zeros((len(cout_tedges) - 1, len(tedges) - 1))

    # Loop over each pore volume
    for i in range(len(aquifer_pore_volumes)):
        if not np.any(valid_bins_2d[i, :]):
            continue

        # Clip optimization: Check for temporal overlap before expensive computation
        # Get the range of infiltration times for this pore volume (only valid bins)
        infiltration_times = infiltration_tedges_2d[i, :]
        valid_infiltration_times = infiltration_times[~np.isnan(infiltration_times)]

        if len(valid_infiltration_times) == 0:
            continue

        infiltration_min = valid_infiltration_times[0]  # Min is first element (monotonic)
        infiltration_max = valid_infiltration_times[-1]  # Max is last element (monotonic)

        # Check if infiltration window overlaps with cin window
        # Two intervals [a1, a2] and [b1, b2] overlap if: max(a1, b1) < min(a2, b2)
        has_overlap = max(infiltration_min, cin_time_min) < min(infiltration_max, cin_time_max)

        if not has_overlap:
            # No temporal overlap - this bin contributes nothing, skip expensive computation
            continue

        # Compute overlap matrix for this pore volume
        overlap_matrix = partial_isin(bin_edges_in=infiltration_tedges_2d[i, :], bin_edges_out=cin_tedges_days)

        # Apply flow weighting to this pore volume's overlap matrix
        flow_weighted_overlap = overlap_matrix * flow[None, :]

        # Normalize this pore volume's contribution (each row sums to 1 after flow weighting)
        row_sums = np.sum(flow_weighted_overlap, axis=1)
        valid_rows_pv = row_sums > 0
        normalized_overlap = np.zeros_like(flow_weighted_overlap)
        normalized_overlap[valid_rows_pv, :] = flow_weighted_overlap[valid_rows_pv, :] / row_sums[valid_rows_pv, None]

        # Accumulate only the valid bins from this pore volume
        accumulated_weights[valid_bins_2d[i, :], :] += normalized_overlap[valid_bins_2d[i, :], :]

    # Average across all pore volumes assuming equal probability per bin.
    # This is correct when aquifer_pore_volumes comes from gamma.bins() which
    # produces equal-probability bins. For user-supplied pore volumes with
    # unequal probability mass, a weights parameter would be needed.
    return accumulated_weights / len(aquifer_pore_volumes)


def _extraction_to_infiltration_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float,
    regularization: float = 0.0,
) -> npt.NDArray[np.floating]:
    """
    Compute deconvolution weights by inverting the forward weight matrix.

    Builds the forward weight matrix W (mapping cin_on_tedges → cout_on_cout_tedges)
    using ``_infiltration_to_extraction_weights``, then returns the (regularized)
    inverse. When ``regularization > 0``, uses Tikhonov regularization with a
    first-difference smoothness penalty to stabilize the inversion for
    ill-conditioned systems.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data bins.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3].
    flow : array-like
        Flow rate values in the aquifer [m3/day].
    retardation_factor : float
        Retardation factor of the compound in the aquifer.
    regularization : float, optional
        Tikhonov regularization strength (default 0.0). When > 0, solves
        ``min ||W x - b||^2 + a^2 ||D x||^2`` where D is the first-difference
        operator and ``a`` is the regularization parameter. Larger values produce
        smoother solutions at the cost of reduced fidelity. Typical values
        range from 0.01 to 1.0 depending on the noise level in cout.

    Returns
    -------
    numpy.ndarray
        Deconvolution weight matrix. Shape: (len(tedges) - 1, len(cout_tedges) - 1).
        Rows for unconstrained cin bins (zero columns in W_forward) are all zeros.
    """
    n_cin = len(tedges) - 1
    n_cout = len(cout_tedges) - 1

    # Build forward weight matrix: cin_on_tedges → cout_on_cout_tedges
    w_forward = _infiltration_to_extraction_weights(
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=flow,
        retardation_factor=retardation_factor,
    )

    # Identify active rows (cout bins with valid contributions) and
    # active columns (cin bins that influence at least one cout bin)
    row_valid = w_forward.sum(axis=1) > 0
    col_active = w_forward.sum(axis=0) > 0

    if not np.any(row_valid) or not np.any(col_active):
        return np.zeros((n_cin, n_cout))

    w_active = w_forward[np.ix_(row_valid, col_active)]

    if regularization > 0:
        # Tikhonov regularization with first-difference smoothness penalty:
        # weights = (W'W + α² D'D)^{-1} W'
        n_active = w_active.shape[1]
        wtw = w_active.T @ w_active
        # D'D is the tridiagonal matrix from first differences
        dtd = np.zeros((n_active, n_active))
        diag_idx = np.arange(n_active)
        dtd[diag_idx, diag_idx] = 2.0
        dtd[0, 0] = 1.0
        dtd[-1, -1] = 1.0
        off_diag = np.arange(n_active - 1)
        dtd[off_diag, off_diag + 1] = -1.0
        dtd[off_diag + 1, off_diag] = -1.0
        inv_active = np.linalg.solve(wtw + regularization**2 * dtd, w_active.T)
    else:
        inv_active = np.linalg.pinv(w_active)

    # Place into full weight matrix
    weights = np.zeros((n_cin, n_cout))
    active_cin_idx = np.where(col_active)[0]
    valid_cout_idx = np.where(row_valid)[0]
    weights[np.ix_(active_cin_idx, valid_cout_idx)] = inv_active

    return weights
