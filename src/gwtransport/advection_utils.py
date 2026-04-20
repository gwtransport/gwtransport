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
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
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
    weights : numpy.ndarray
        Normalized weight matrix. Shape: (len(cout_tedges) - 1, len(tedges) - 1).
        Rows for spin-up and zero-flow cout bins are all zero; use ``spinup_mask``
        to distinguish spin-up (no contributing streamtube) from zero-flow
        (contributing streamtubes all trace back into a zero-flow cin window).
    spinup_mask : numpy.ndarray of bool
        Shape: (len(cout_tedges) - 1,). True for cout bins where no streamtube
        traced back into the cin time range (signal has not broken through).
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

    # Per-streamtube derivation: for streamtube i and outlet bin k, mass balance gives
    #     c_out[k|i] = sum_j (Q_j * overlap[i,k,j] * c_j) / sum_j' (Q_j' * overlap[i,k,j'])
    # which is the literal definition of mass-flux divided by water-flux for the bin.
    # Each streamtube carries equal flow at the outlet (equal-mass pore-volume bins from
    # the gamma distribution), so the bundle outlet concentration is the simple arithmetic
    # average over streamtubes that contributed to the bin:
    #     c_out[k] = (1 / N_valid_k) * sum_{i valid} c_out[k|i]
    # During spin-up, only the shorter pore-volume streamtubes have a valid source window
    # within the cin time range; we average over the contributing streamtubes only.
    n_cout = len(cout_tedges) - 1
    n_cin = len(tedges) - 1
    accumulated_weights = np.zeros((n_cout, n_cin))
    contributing_bins = np.zeros(n_cout)

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

        # Compute overlap matrix for this pore volume.
        # partial_isin returns shape (n_in, n_out) = (n_cout, n_cin) here, because
        # bin_edges_in has length n_cout+1 (one infiltration-time edge per cout edge)
        # and bin_edges_out has length n_cin+1 (the cin time edges).
        overlap_matrix = partial_isin(bin_edges_in=infiltration_tedges_2d[i, :], bin_edges_out=cin_tedges_days)

        # Per-streamtube weights: row k gives c_out[k|i] = sum_j weight[k,j] * c_in[j].
        # The row-normalization is the direct mass-flux/water-flux ratio, NOT an
        # error-hiding renormalization.
        flow_weighted_overlap = overlap_matrix * flow[None, :]
        row_sums = np.sum(flow_weighted_overlap, axis=1)
        valid_rows_pv = row_sums > 0
        normalized_overlap = np.zeros_like(flow_weighted_overlap)
        normalized_overlap[valid_rows_pv, :] = flow_weighted_overlap[valid_rows_pv, :] / row_sums[valid_rows_pv, None]

        # Accumulate only the valid bins from this pore volume for a simple count-average
        # over contributing streamtubes.
        accumulated_weights[valid_bins_2d[i, :], :] += normalized_overlap[valid_bins_2d[i, :], :]
        contributing_bins[valid_bins_2d[i, :]] += 1

    # Simple arithmetic average across contributing streamtubes (equal flow per
    # streamtube at the outlet means equal weight). This is correct under variable
    # flow; an end-of-loop global normalization would silently bias streamtubes by
    # source-window length.
    valid_rows = contributing_bins > 0
    result = np.zeros_like(accumulated_weights)
    result[valid_rows, :] = accumulated_weights[valid_rows, :] / contributing_bins[valid_rows, None]
    spinup_mask = ~valid_rows
    return result, spinup_mask
