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

    # Accumulate flow-weighted overlap matrices from all pore volumes, then perform
    # a single end-of-loop flow-weighted normalization. This preserves mass conservation
    # under variable flow: each output row's contribution from cin bin j is proportional
    # to the volume of water (flow * overlap fraction) that left bin j and arrived in
    # the output bin via any pore volume path.
    n_cout = len(cout_tedges) - 1
    n_cin = len(tedges) - 1
    accumulated_flow_weighted = np.zeros((n_cout, n_cin))
    accumulated_denominators = np.zeros(n_cout)

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

        # Apply flow weighting (cin bins carry their own flow rate)
        fw = overlap_matrix * flow[None, :]
        row_sums = fw.sum(axis=1)

        # Accumulate only the valid bins from this pore volume
        accumulated_flow_weighted[valid_bins_2d[i], :] += fw[valid_bins_2d[i], :]
        accumulated_denominators[valid_bins_2d[i]] += row_sums[valid_bins_2d[i]]

    # Single flow-weighted normalization across all contributing pore volumes
    valid_rows = accumulated_denominators > 0
    result = np.zeros_like(accumulated_flow_weighted)
    result[valid_rows] = accumulated_flow_weighted[valid_rows] / accumulated_denominators[valid_rows, None]
    return result
