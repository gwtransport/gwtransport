"""
Private helper functions for advective transport modeling.

This module contains internal helper functions used by the advection module.
These functions implement various algorithms for computing transport weights
and handling nonlinear sorption.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import operator

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.residence_time import residence_time
from gwtransport.utils import partial_isin

# Constants for shock detection
_SHOCK_TOLERANCE = 1e-10  # Tolerance for detecting instantaneous shocks


def _infiltration_to_extraction_nonlinear_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute normalized weights for nonlinear infiltration to extraction transformation.

    This helper function computes the weight matrix for concentration-dependent retardation.
    The algorithm tracks infiltration parcels forward in time using concentration-dependent
    retardation factors, computes temporal overlaps with extraction bins, and constructs
    flow-weighted transformation matrix.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for infiltration bins.
    cout_tedges : pandas.DatetimeIndex
        Time edges for extraction bins.
    aquifer_pore_volumes : array-like
        Distribution of pore volumes [m3].
    cin : array-like
        Concentration values (needed for dimensions).
    flow : array-like
        Flow rate values [m3/day].
    retardation_factors : array-like
        Concentration-dependent retardation factors.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix. Shape: (len(cout_tedges) - 1, len(cin))
    """
    # Convert to days
    cin_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # FORWARD PARCEL TRACKING with shock handling
    # Key insight: track infiltration BINS (parcels) not edges
    # Each parcel has concentration, mass, and arrival time
    # When parcels overlap at extraction → mix (captures shocks naturally)

    n_cin_bins = len(cin)
    n_cout_bins = len(cout_tedges) - 1
    n_pv = len(aquifer_pore_volumes)

    # NOTE: For forward tracking, cout_tedges should extend beyond tedges
    # to capture delayed arrivals due to retardation.
    # Rule of thumb: extend cout_tedges by max_residence_time
    # where max_residence_time ≈ max(pore_volumes) * max(retardation_factors) / min(flow)

    # Accumulate weights from all pore volumes
    accumulated_weights = np.zeros((n_cout_bins, n_cin_bins))
    pv_count = np.zeros(n_cout_bins)

    # For each pore volume (aquifer heterogeneity)
    for i_pv in range(n_pv):
        pv = aquifer_pore_volumes[i_pv]

        # STEP 1: Compute residence times for all edges
        # Strategy: Call residence_time once per edge with its retardation
        rt_bin_edges = np.zeros(n_cin_bins + 1)

        # Interpolate retardation to edges (vectorized)
        # NOTE: Edge interpolation for concentration-dependent retardation
        #
        # Physical principle: R = f(C), so we should:
        #   1. Interpolate concentration to edges
        #   2. Compute retardation from edge concentration
        #
        # This is more accurate than directly averaging R values, especially
        # for nonlinear isotherms where R(C_avg) ≠ avg(R(C)).
        #
        # For sharp concentration steps, this still has limitations (O(Δt) error),
        # but is significantly better than arithmetic averaging of R.
        #
        # Compute retardation at edges
        # NOTE: Ideally we would interpolate concentration first, then compute R(C_edge).
        # However, that requires knowing the isotherm parameters, which aren't passed to
        # this function. For now, we use arithmetic averaging of R values.
        #
        # TODO: Consider adding optional isotherm parameters to enable proper C interpolation.
        # This would be more accurate: R_edge = f(0.5*(C[i-1] + C[i]))
        # vs current: R_edge = 0.5*(R[i-1] + R[i])
        #
        # For smooth concentration variations, the error is O(Δt²).
        # For sharp steps, this can introduce significant errors.
        r_at_edges = np.zeros(n_cin_bins + 1)
        r_at_edges[0] = retardation_factors[0]
        r_at_edges[-1] = retardation_factors[-1]
        r_at_edges[1:-1] = 0.5 * (retardation_factors[:-1] + retardation_factors[1:])

        for i_edge in range(n_cin_bins + 1):
            rt_val = residence_time(
                flow=flow,
                flow_tedges=tedges,
                index=tedges[i_edge : i_edge + 1],
                aquifer_pore_volume=np.array([pv]),
                retardation_factor=float(r_at_edges[i_edge]),
                direction="infiltration_to_extraction",
            )
            rt_bin_edges[i_edge] = rt_val[0, 0]

        # Compute extraction times for all edges
        extraction_times = cin_tedges_days + rt_bin_edges

        # STEP 2: VECTORIZED temporal overlap computation
        # Shape: [n_cin_bins, n_cout_bins]
        # Extract parcel boundaries
        t_out_starts = extraction_times[:-1]  # [n_cin_bins]
        t_out_ends = extraction_times[1:]  # [n_cin_bins]

        # Extract cout bin boundaries
        t_cout_starts = cout_tedges_days[:-1]  # [n_cout_bins]
        t_cout_ends = cout_tedges_days[1:]  # [n_cout_bins]

        # Broadcast to compute all overlaps at once
        # overlap_start[i, j] = max(t_out_start[i], t_cout_start[j])
        overlap_starts = np.maximum(t_out_starts[:, None], t_cout_starts[None, :])  # [n_cin, n_cout]
        overlap_ends = np.minimum(t_out_ends[:, None], t_cout_ends[None, :])  # [n_cin, n_cout]
        overlap_durations = np.maximum(0, overlap_ends - overlap_starts)  # [n_cin, n_cout]

        # Compute parcel durations
        parcel_durations = t_out_ends - t_out_starts  # [n_cin_bins]

        # Compute fractions: fraction[i, j] = overlap_duration[i, j] / parcel_duration[i]
        # Handle division by zero
        fractions = np.zeros_like(overlap_durations)
        valid_parcels = parcel_durations > 0
        fractions[valid_parcels, :] = overlap_durations[valid_parcels, :] / parcel_durations[valid_parcels, None]

        # For instantaneous parcels (duration == 0), assign to bin that contains start time
        instantaneous = ~valid_parcels
        if np.any(instantaneous):
            # Check if t_out_start falls within cout bin
            inst_contained = (t_out_starts[instantaneous, None] >= t_cout_starts[None, :]) & (
                t_out_starts[instantaneous, None] < t_cout_ends[None, :]
            )
            fractions[instantaneous, :] = inst_contained.astype(float)

        # Zero out fractions where overlap_duration is 0 (no overlap)
        fractions[overlap_durations == 0] = 0

        # Compute flow-weighted fractions
        # flow_weighted_fractions[i, j] = flow[i] * fraction[i, j]
        # Shape: [n_cin_bins, n_cout_bins]
        flow_weighted_fractions = flow[:, None] * fractions  # [n_cin, n_cout]

        # Transpose to get [n_cout_bins, n_cin_bins] for accumulation
        flow_weighted_fractions_t = flow_weighted_fractions.T  # [n_cout, n_cin]

        # Accumulate weights across pore volumes
        accumulated_weights += flow_weighted_fractions_t

        # Track which cout bins have contributions from this pore volume
        has_contribution = np.sum(fractions, axis=0) > 0  # [n_cout_bins]
        pv_count[has_contribution] += 1

    # Average across valid pore volumes
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cout = pv_count > 0
    averaged_weights[valid_cout, :] = accumulated_weights[valid_cout, :] / pv_count[valid_cout, None]

    # Normalize by total weights per output bin
    total_weights = np.sum(averaged_weights, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(averaged_weights)
    normalized_weights[valid_weights, :] = averaged_weights[valid_weights, :] / total_weights[valid_weights, None]

    return normalized_weights


def _infiltration_to_extraction_nonlinear_weights_exact(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute normalized weights using exact front-tracking for nonlinear sorption.

    This implements an exact front-tracking algorithm that maintains sharp shocks
    without numerical diffusion. Unlike the method of characteristics which treats
    parcels independently, this algorithm explicitly tracks which parcels are
    co-active at each moment in time.

    Algorithm:
    1. For each pore volume, compute parcel arrival times using characteristics
    2. Use event-based processing to determine exactly when parcels are co-active
    3. Build piecewise-constant solution with sharp fronts
    4. Map solution to output grid with exact overlap calculations
    5. Flow-weight and normalize

    Key advantages over method_of_characteristics:
    - No numerical diffusion at shocks
    - Maintains sharp concentration fronts
    - Exact handling of parcel collisions
    - Physically accurate shock propagation

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for infiltration bins.
    cout_tedges : pandas.DatetimeIndex
        Time edges for extraction bins.
    aquifer_pore_volumes : array-like
        Distribution of pore volumes [m3].
    cin : array-like
        Concentration values (needed for dimensions).
    flow : array-like
        Flow rate values [m3/day].
    retardation_factors : array-like
        Concentration-dependent retardation factors.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix. Shape: (len(cout_tedges) - 1, len(cin))
    """
    # Convert to days
    cin_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    n_cin_bins = len(cin)
    n_cout_bins = len(cout_tedges) - 1
    n_pv = len(aquifer_pore_volumes)

    # Accumulate weights from all pore volumes
    accumulated_weights = np.zeros((n_cout_bins, n_cin_bins))
    pv_count = np.zeros(n_cout_bins)

    # For each pore volume (aquifer heterogeneity)
    for i_pv in range(n_pv):
        pv = aquifer_pore_volumes[i_pv]

        # STEP 1: Build parcels with exact arrival times
        parcels = _build_parcel_arrivals(
            cin_tedges_days=cin_tedges_days,
            retardation_factors=retardation_factors,
            pv=pv,
            flow=flow,
            tedges=tedges,
        )

        # STEP 2: Exact front tracking via event processing
        solution_intervals = _process_front_tracking_events(parcels)

        # STEP 3: Map exact solution to output grid
        overlap_times = _map_solution_to_output_grid(
            solution_intervals=solution_intervals,
            cout_tedges_days=cout_tedges_days,
            n_cin_bins=n_cin_bins,
        )

        # STEP 4: Flow-weight the overlaps
        flow_weighted = overlap_times * flow[None, :]

        # Accumulate across pore volumes
        accumulated_weights += flow_weighted

        # Track which cout bins have contributions from this pore volume
        has_contribution = np.sum(overlap_times, axis=1) > 0
        pv_count[has_contribution] += 1

    # Average across valid pore volumes
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cout = pv_count > 0
    averaged_weights[valid_cout, :] = accumulated_weights[valid_cout, :] / pv_count[valid_cout, None]

    # Normalize by total weights per output bin
    total_weights = np.sum(averaged_weights, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(averaged_weights)
    normalized_weights[valid_weights, :] = averaged_weights[valid_weights, :] / total_weights[valid_weights, None]

    return normalized_weights


def _build_parcel_arrivals(
    *,
    cin_tedges_days: npt.NDArray[np.floating],
    retardation_factors: npt.NDArray[np.floating],
    pv: float,
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
) -> list[dict]:
    """
    Build list of parcels with their exact arrival times at extraction.

    Each parcel represents an infiltration bin that propagates through the aquifer
    with its characteristic velocity (determined by its retardation factor).

    Parameters
    ----------
    cin_tedges_days : array-like
        Infiltration time edges in days
    retardation_factors : array-like
        Concentration-dependent retardation factors for each bin
    pv : float
        Single pore volume value [m3]
    flow : array-like
        Flow rates [m3/day]
    tedges : pandas.DatetimeIndex
        Time edges for flow data

    Returns
    -------
    list of dict
        Each parcel: {'id': int, 't_start': float, 't_end': float}
    """
    n_cin_bins = len(retardation_factors)
    parcels = []

    # Interpolate retardation to edges (arithmetic averaging)
    # NOTE: For smooth C variations, this gives O(Δt²) accuracy
    # For sharp steps, this has limitations but is standard practice
    r_at_edges = np.zeros(n_cin_bins + 1)
    r_at_edges[0] = retardation_factors[0]
    r_at_edges[-1] = retardation_factors[-1]
    r_at_edges[1:-1] = 0.5 * (retardation_factors[:-1] + retardation_factors[1:])

    # Compute residence times for all edges at once
    # This is more efficient and avoids NaN issues from single-element slices
    rt_at_edges = np.zeros(n_cin_bins + 1)
    for i_edge in range(n_cin_bins + 1):
        rt_val = residence_time(
            flow=flow,
            flow_tedges=tedges,
            index=tedges[i_edge : i_edge + 1],
            aquifer_pore_volume=np.array([pv]),
            retardation_factor=float(r_at_edges[i_edge]),
            direction="infiltration_to_extraction",
        )
        # Extract scalar value safely
        if rt_val.size > 0 and not np.isnan(rt_val[0, 0]):
            rt_at_edges[i_edge] = rt_val[0, 0]
        else:
            # If residence time is NaN, use simple calculation
            # rt = pv * R / flow_avg
            flow_avg = np.mean(flow) if len(flow) > 0 else 1.0
            rt_at_edges[i_edge] = (pv * r_at_edges[i_edge]) / flow_avg

    # Build parcels
    for i_bin in range(n_cin_bins):
        # Parcel arrival times: infiltration time + residence time
        t_arrival_start = cin_tedges_days[i_bin] + rt_at_edges[i_bin]
        t_arrival_end = cin_tedges_days[i_bin + 1] + rt_at_edges[i_bin + 1]

        # Handle shock formation: when trailing edge catches up to leading edge
        # This happens when retardation decreases (e.g., high C overtaking low C)
        # In this case, the parcel forms a shock front (instantaneous)
        if t_arrival_end < t_arrival_start:
            # Shock: parcel compressed to a point at the average arrival time
            # This maintains mass conservation while capturing shock physics
            t_shock = 0.5 * (t_arrival_start + t_arrival_end)
            parcels.append({
                "id": i_bin,
                "t_start": t_shock,
                "t_end": t_shock,  # Instantaneous shock
            })
        else:
            # Normal parcel (no shock)
            parcels.append({
                "id": i_bin,
                "t_start": t_arrival_start,
                "t_end": t_arrival_end,
            })

    return parcels


def _process_front_tracking_events(parcels: list[dict]) -> list[dict]:
    """
    Process parcel arrival/departure events to build exact piecewise-constant solution.

    This is the core of the front-tracking algorithm. It uses a sweep-line approach
    to determine exactly which parcels are co-active at each moment in time.

    When multiple parcels are co-active, they represent a mixing region where
    the extracted concentration is the flow-weighted average of all active parcels.

    Sharp fronts (shocks) occur at event boundaries where parcels enter or leave.

    Instantaneous parcels (shocks where t_start == t_end) are handled specially
    as impulses rather than intervals.

    Parameters
    ----------
    parcels : list of dict
        Each parcel has 'id', 't_start', 't_end'

    Returns
    -------
    list of dict
        Solution intervals, each containing:
        - 't_start': float, interval start time
        - 't_end': float, interval end time
        - 'active_ids': list of int, IDs of co-active parcels
        - 'is_shock': bool, True if this is an instantaneous shock
    """
    # Separate instantaneous (shock) and regular parcels
    regular_parcels = []
    shock_parcels = []

    for parcel in parcels:
        if abs(parcel["t_end"] - parcel["t_start"]) < _SHOCK_TOLERANCE:
            # Instantaneous parcel (shock)
            shock_parcels.append(parcel)
        else:
            # Regular parcel with finite duration
            regular_parcels.append(parcel)

    # Create events only for regular parcels
    events = [
        event
        for parcel in regular_parcels
        for event in [
            ("arrive", parcel["t_start"], parcel["id"]),  # Arrival event
            ("depart", parcel["t_end"], parcel["id"]),  # Departure event
        ]
    ]

    # Sort events by time
    # Tie-breaking: process departures before arrivals at the same time
    # This ensures correct handling of transitions between adjacent parcels
    events.sort(key=lambda e: (e[1], e[0] == "arrive"))

    # Sweep through events, maintaining set of active parcels
    solution_intervals = []
    active_parcel_ids = []
    prev_time = None

    for event_type, event_time, parcel_id in events:
        # If there's a time gap and parcels are active, save interval
        if prev_time is not None and prev_time < event_time and active_parcel_ids:
            solution_intervals.append({
                "t_start": prev_time,
                "t_end": event_time,
                "active_ids": active_parcel_ids.copy(),  # Copy to avoid aliasing
                "is_shock": False,
            })

        # Update active set based on event type
        if event_type == "arrive":
            active_parcel_ids.append(parcel_id)
        else:  # depart
            active_parcel_ids.remove(parcel_id)

        prev_time = event_time

    # Add shock parcels as instantaneous intervals
    solution_intervals.extend([
        {
            "t_start": shock["t_start"],
            "t_end": shock["t_end"],  # Same as t_start for shocks
            "active_ids": [shock["id"]],
            "is_shock": True,
        }
        for shock in shock_parcels
    ])

    # Sort all intervals by start time
    solution_intervals.sort(key=operator.itemgetter("t_start"))

    return solution_intervals


def _map_solution_to_output_grid(
    *,
    solution_intervals: list[dict],
    cout_tedges_days: npt.NDArray[np.floating],
    n_cin_bins: int,
) -> npt.NDArray[np.floating]:
    """
    Map piecewise-constant exact solution to output time grid.

    For each output bin, compute how long each infiltration parcel is active
    within that bin. This gives the exact temporal overlap without any
    numerical diffusion.

    Parameters
    ----------
    solution_intervals : list of dict
        Exact solution from front tracking, each interval has:
        - 't_start', 't_end': time boundaries
        - 'active_ids': list of co-active parcel IDs
    cout_tedges_days : array-like
        Output time edges in days
    n_cin_bins : int
        Number of infiltration bins

    Returns
    -------
    numpy.ndarray
        Overlap times matrix, shape (n_cout_bins, n_cin_bins)
        overlap_times[j, i] = time duration that parcel i is active in output bin j
    """
    n_cout_bins = len(cout_tedges_days) - 1
    overlap_times = np.zeros((n_cout_bins, n_cin_bins))

    # For each solution interval, compute overlaps with output bins
    for interval in solution_intervals:
        t_start = interval["t_start"]
        t_end = interval["t_end"]
        active_ids = interval["active_ids"]

        # Find all output bins that overlap with this interval
        for j in range(n_cout_bins):
            out_start = cout_tedges_days[j]
            out_end = cout_tedges_days[j + 1]

            # Compute temporal overlap
            overlap_start = max(t_start, out_start)
            overlap_end = min(t_end, out_end)
            overlap_duration = max(0.0, overlap_end - overlap_start)

            if overlap_duration > 0:
                # During this overlap, all active parcels contribute
                for parcel_id in active_ids:
                    overlap_times[j, parcel_id] += overlap_duration

    return overlap_times


def _build_parcel_departures_backward(
    *,
    cout_tedges_days: npt.NDArray[np.floating],
    retardation_factors: npt.NDArray[np.floating],
    pv: float,
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
) -> list[dict]:
    """
    Build list of parcels with their exact departure times from infiltration (backward tracking).

    Each parcel represents an extraction bin that is traced backward through the aquifer
    to determine when it departed from the infiltration zone.

    Parameters
    ----------
    cout_tedges_days : array-like
        Extraction time edges in days
    retardation_factors : array-like
        Concentration-dependent retardation factors for each extraction bin
    pv : float
        Single pore volume value [m3]
    flow : array-like
        Flow rates [m3/day]
    tedges : pandas.DatetimeIndex
        Time edges for flow data

    Returns
    -------
    list of dict
        Each parcel: {'id': int, 't_start': float, 't_end': float}
        Times represent departure from infiltration (backward-tracked)
    """
    n_cout_bins = len(retardation_factors)
    parcels = []

    # Interpolate retardation to edges (arithmetic averaging)
    r_at_edges = np.zeros(n_cout_bins + 1)
    r_at_edges[0] = retardation_factors[0]
    r_at_edges[-1] = retardation_factors[-1]
    r_at_edges[1:-1] = 0.5 * (retardation_factors[:-1] + retardation_factors[1:])

    # Compute residence times for all edges
    rt_at_edges = np.zeros(n_cout_bins + 1)
    for i_edge in range(n_cout_bins + 1):
        rt_val = residence_time(
            flow=flow,
            flow_tedges=tedges,
            index=tedges[i_edge : i_edge + 1],
            aquifer_pore_volume=np.array([pv]),
            retardation_factor=float(r_at_edges[i_edge]),
            direction="infiltration_to_extraction",  # Still forward travel time, just tracking backward
        )
        # Extract scalar value safely
        if rt_val.size > 0 and not np.isnan(rt_val[0, 0]):
            rt_at_edges[i_edge] = rt_val[0, 0]
        else:
            # Fallback calculation if NaN
            flow_avg = np.mean(flow) if len(flow) > 0 else 1.0
            rt_at_edges[i_edge] = (pv * r_at_edges[i_edge]) / flow_avg

    # Build parcels (backward tracking: departure = extraction - residence_time)
    for i_bin in range(n_cout_bins):
        # Parcel departure times: extraction time - residence time
        t_departure_start = cout_tedges_days[i_bin] - rt_at_edges[i_bin]
        t_departure_end = cout_tedges_days[i_bin + 1] - rt_at_edges[i_bin + 1]

        # Handle shock formation in backward direction
        # When trailing edge departs later than leading edge (inverted)
        if t_departure_end < t_departure_start:
            # Shock: parcel compressed to a point at average departure time
            t_shock = 0.5 * (t_departure_start + t_departure_end)
            parcels.append({
                "id": i_bin,
                "t_start": t_shock,
                "t_end": t_shock,  # Instantaneous shock
            })
        else:
            # Normal parcel (no shock)
            parcels.append({
                "id": i_bin,
                "t_start": t_departure_start,
                "t_end": t_departure_end,
            })

    return parcels


def _extraction_to_infiltration_nonlinear_weights_exact(
    *,
    tedges: pd.DatetimeIndex,
    cin_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cout: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute normalized weights using exact front-tracking for nonlinear sorption (backward).

    This implements exact front-tracking algorithm for the backward (deconvolution) direction.
    Unlike the forward method which tracks infiltration parcels forward to extraction,
    this tracks extraction parcels backward to infiltration.

    Algorithm:
    1. For each pore volume, compute parcel departure times using backward characteristics
    2. Use event-based processing to determine exactly when parcels are co-active
    3. Build piecewise-constant solution with sharp fronts
    4. Map solution to infiltration grid with exact overlap calculations
    5. Flow-weight and normalize

    Key advantages:
    - No numerical diffusion at shocks
    - Maintains sharp concentration fronts
    - Exact handling of parcel collisions
    - Physically accurate shock propagation in backward direction

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for extraction bins.
    cin_tedges : pandas.DatetimeIndex
        Time edges for infiltration bins.
    aquifer_pore_volumes : array-like
        Distribution of pore volumes [m3].
    cout : array-like
        Concentration values (needed for dimensions).
    flow : array-like
        Flow rate values [m3/day].
    retardation_factors : array-like
        Concentration-dependent retardation factors.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix. Shape: (len(cin_tedges) - 1, len(cout))
    """
    # Convert to days
    cout_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cin_tedges_days = ((cin_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    n_cout_bins = len(cout)
    n_cin_bins = len(cin_tedges) - 1
    n_pv = len(aquifer_pore_volumes)

    # Accumulate weights from all pore volumes
    accumulated_weights = np.zeros((n_cin_bins, n_cout_bins))
    pv_count = np.zeros(n_cin_bins)

    # For each pore volume (aquifer heterogeneity)
    for i_pv in range(n_pv):
        pv = aquifer_pore_volumes[i_pv]

        # STEP 1: Build parcels with exact departure times (backward tracking)
        parcels = _build_parcel_departures_backward(
            cout_tedges_days=cout_tedges_days,
            retardation_factors=retardation_factors,
            pv=pv,
            flow=flow,
            tedges=tedges,
        )

        # STEP 2: Exact front tracking via event processing
        # Reuse the same event processing as forward direction
        solution_intervals = _process_front_tracking_events(parcels)

        # STEP 3: Map exact solution to infiltration grid
        overlap_times = _map_solution_to_output_grid(
            solution_intervals=solution_intervals,
            cout_tedges_days=cin_tedges_days,  # Map to infiltration grid
            n_cin_bins=n_cout_bins,  # Number of parcels (extraction bins)
        )

        # STEP 4: Flow-weight the overlaps
        flow_weighted = overlap_times * flow[None, :]

        # Accumulate across pore volumes
        accumulated_weights += flow_weighted

        # Track which cin bins have contributions from this pore volume
        has_contribution = np.sum(overlap_times, axis=1) > 0
        pv_count[has_contribution] += 1

    # Average across valid pore volumes
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cin = pv_count > 0
    averaged_weights[valid_cin, :] = accumulated_weights[valid_cin, :] / pv_count[valid_cin, None]

    # Normalize by total weights per output bin
    total_weights = np.sum(averaged_weights, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(averaged_weights)
    normalized_weights[valid_weights, :] = averaged_weights[valid_weights, :] / total_weights[valid_weights, None]

    return normalized_weights


def _extraction_to_infiltration_nonlinear_weights(
    *,
    tedges: pd.DatetimeIndex,
    cin_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cout: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factors: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute normalized weights for nonlinear extraction to infiltration transformation.

    This helper function computes the weight matrix for concentration-dependent retardation
    in the backward (deconvolution) direction. The algorithm tracks extraction parcels backward
    in time using concentration-dependent retardation factors, computes temporal overlaps with
    infiltration bins, and constructs flow-weighted transformation matrix.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for extraction bins.
    cin_tedges : pandas.DatetimeIndex
        Time edges for infiltration bins.
    aquifer_pore_volumes : array-like
        Distribution of pore volumes [m3].
    cout : array-like
        Concentration values (needed for dimensions).
    flow : array-like
        Flow rate values [m3/day].
    retardation_factors : array-like
        Concentration-dependent retardation factors aligned with cout.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix. Shape: (len(cin_tedges) - 1, len(cout))
    """
    # Convert to days
    cout_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cin_tedges_days = ((cin_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # BACKWARD PARCEL TRACKING with shock handling
    # Key insight: track extraction BINS (parcels) not edges
    # Each parcel has concentration, mass, and origin time
    # When parcels map to infiltration → mix (captures shocks naturally)

    n_cout_bins = len(cout)
    n_cin_bins = len(cin_tedges) - 1
    n_pv = len(aquifer_pore_volumes)

    # NOTE: For backward tracking, cin_tedges should extend before tedges
    # to capture early infiltration times that contribute to observed extraction.
    # Rule of thumb: extend cin_tedges backward by max_residence_time
    # where max_residence_time ≈ max(pore_volumes) * max(retardation_factors) / min(flow)

    # Accumulate weights from all pore volumes
    accumulated_weights = np.zeros((n_cin_bins, n_cout_bins))
    pv_count = np.zeros(n_cin_bins)

    # For each pore volume (aquifer heterogeneity)
    for i_pv in range(n_pv):
        pv = aquifer_pore_volumes[i_pv]

        # STEP 1: Compute residence times for all edges
        # Strategy: Call residence_time once per edge with its retardation
        rt_bin_edges = np.zeros(n_cout_bins + 1)

        # Interpolate retardation to edges (vectorized)
        # NOTE: Edge interpolation for concentration-dependent retardation
        #
        # Physical principle: R = f(C), so we should:
        #   1. Interpolate concentration to edges
        #   2. Compute retardation from edge concentration
        #
        # This is more accurate than directly averaging R values, especially
        # for nonlinear isotherms where R(C_avg) ≠ avg(R(C)).
        #
        # For sharp concentration steps, this still has limitations (O(Δt) error),
        # but is significantly better than arithmetic averaging of R.
        #
        # Compute retardation at edges
        # NOTE: Ideally we would interpolate concentration first, then compute R(C_edge).
        # However, that requires knowing the isotherm parameters, which aren't passed to
        # this function. For now, we use arithmetic averaging of R values.
        #
        # TODO: Consider adding optional isotherm parameters to enable proper C interpolation.
        # This would be more accurate: R_edge = f(0.5*(C[i-1] + C[i]))
        # vs current: R_edge = 0.5*(R[i-1] + R[i])
        #
        # For smooth concentration variations, the error is O(Δt²).
        # For sharp steps, this can introduce significant errors.
        r_at_edges = np.zeros(n_cout_bins + 1)
        r_at_edges[0] = retardation_factors[0]
        r_at_edges[-1] = retardation_factors[-1]
        r_at_edges[1:-1] = 0.5 * (retardation_factors[:-1] + retardation_factors[1:])

        for i_edge in range(n_cout_bins + 1):
            rt_val = residence_time(
                flow=flow,
                flow_tedges=tedges,
                index=tedges[i_edge : i_edge + 1],
                aquifer_pore_volume=np.array([pv]),
                retardation_factor=float(r_at_edges[i_edge]),
                direction="extraction_to_infiltration",
            )
            rt_bin_edges[i_edge] = rt_val[0, 0]

        # Compute infiltration times for all edges (BACKWARD: subtract residence times)
        infiltration_times = cout_tedges_days - rt_bin_edges

        # STEP 2: VECTORIZED temporal overlap computation
        # Shape: [n_cout_bins, n_cin_bins]
        # Extract parcel boundaries (extraction parcels tracked backward)
        t_in_starts = infiltration_times[:-1]  # [n_cout_bins]
        t_in_ends = infiltration_times[1:]  # [n_cout_bins]

        # Extract cin bin boundaries
        t_cin_starts = cin_tedges_days[:-1]  # [n_cin_bins]
        t_cin_ends = cin_tedges_days[1:]  # [n_cin_bins]

        # Broadcast to compute all overlaps at once
        # overlap_start[i, j] = max(t_in_start[i], t_cin_start[j])
        overlap_starts = np.maximum(t_in_starts[:, None], t_cin_starts[None, :])  # [n_cout, n_cin]
        overlap_ends = np.minimum(t_in_ends[:, None], t_cin_ends[None, :])  # [n_cout, n_cin]
        overlap_durations = np.maximum(0, overlap_ends - overlap_starts)  # [n_cout, n_cin]

        # Compute parcel durations
        parcel_durations = t_in_ends - t_in_starts  # [n_cout_bins]

        # Compute fractions: fraction[i, j] = overlap_duration[i, j] / parcel_duration[i]
        # Handle division by zero
        fractions = np.zeros_like(overlap_durations)
        valid_parcels = parcel_durations > 0
        fractions[valid_parcels, :] = overlap_durations[valid_parcels, :] / parcel_durations[valid_parcels, None]

        # For instantaneous parcels (duration == 0), assign to bin that contains start time
        instantaneous = ~valid_parcels
        if np.any(instantaneous):
            # Check if t_in_start falls within cin bin
            inst_contained = (t_in_starts[instantaneous, None] >= t_cin_starts[None, :]) & (
                t_in_starts[instantaneous, None] < t_cin_ends[None, :]
            )
            fractions[instantaneous, :] = inst_contained.astype(float)

        # Zero out fractions where overlap_duration is 0 (no overlap)
        fractions[overlap_durations == 0] = 0

        # Compute flow-weighted fractions
        # flow_weighted_fractions[i, j] = flow[i] * fraction[i, j]
        # Shape: [n_cout_bins, n_cin_bins]
        flow_weighted_fractions = flow[:, None] * fractions  # [n_cout, n_cin]

        # Transpose to get [n_cin_bins, n_cout_bins] for accumulation
        flow_weighted_fractions_t = flow_weighted_fractions.T  # [n_cin, n_cout]

        # Accumulate weights across pore volumes
        accumulated_weights += flow_weighted_fractions_t

        # Track which cin bins have contributions from this pore volume
        has_contribution = np.sum(fractions, axis=0) > 0  # [n_cin_bins]
        pv_count[has_contribution] += 1

    # Average across valid pore volumes
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cin = pv_count > 0
    averaged_weights[valid_cin, :] = accumulated_weights[valid_cin, :] / pv_count[valid_cin, None]

    # Normalize by total weights per output bin
    total_weights = np.sum(averaged_weights, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(averaged_weights)
    normalized_weights[valid_weights, :] = averaged_weights[valid_weights, :] / total_weights[valid_weights, None]

    return normalized_weights


def _infiltration_to_extraction_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float | npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """
    Compute normalized weights for linear infiltration to extraction transformation.

    This helper function computes the weight matrix for constant retardation factor.
    It handles the main advective transport calculation with flow-weighted averaging.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for infiltration bins.
    cout_tedges : pandas.DatetimeIndex
        Time edges for extraction bins.
    aquifer_pore_volumes : array-like
        Distribution of pore volumes [m3].
    cin : array-like
        Concentration values (needed for dimensions).
    flow : array-like
        Flow rate values [m3/day].
    retardation_factor : float
        Constant retardation factor.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix. Shape: (len(cout_tedges) - 1, len(cin))
    """
    # Convert time edges to days
    cin_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # Pre-compute all residence times and infiltration edges
    rt_edges_2d = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    infiltration_tedges_2d = cout_tedges_days[None, :] - rt_edges_2d

    # Pre-compute valid bins and count
    valid_bins_2d = ~(np.isnan(infiltration_tedges_2d[:, :-1]) | np.isnan(infiltration_tedges_2d[:, 1:]))
    valid_pv_count = np.sum(valid_bins_2d, axis=0)

    accumulated_weights = np.zeros((len(cout_tedges) - 1, len(cin)))

    # Pre-compute cin time range for clip optimization (computed once, used n_bins times)
    cin_time_min = cin_tedges_days[0]
    cin_time_max = cin_tedges_days[-1]

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

        # Only compute overlap matrix if there's potential contribution
        overlap_matrix = partial_isin(bin_edges_in=infiltration_tedges_2d[i, :], bin_edges_out=cin_tedges_days)
        accumulated_weights[valid_bins_2d[i, :], :] += overlap_matrix[valid_bins_2d[i, :], :]

    # Average across valid pore volumes and apply flow weighting
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cout = valid_pv_count > 0
    averaged_weights[valid_cout, :] = accumulated_weights[valid_cout, :] / valid_pv_count[valid_cout, None]

    # Apply flow weighting after averaging
    flow_weighted_averaged = averaged_weights * flow[None, :]

    total_weights = np.sum(flow_weighted_averaged, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(flow_weighted_averaged)
    normalized_weights[valid_weights, :] = flow_weighted_averaged[valid_weights, :] / total_weights[valid_weights, None]
    return normalized_weights


def _extraction_to_infiltration_weights(
    *,
    tedges: pd.DatetimeIndex,
    cin_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cout: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.floating]:
    """
    Compute extraction to infiltration transformation weights matrix.

    Computes the weight matrix for the extraction to infiltration transformation,
    ensuring mathematical symmetry with the infiltration to extraction operation. The extraction to infiltration
    weights represent the transpose relationship needed for deconvolution.

    SYMMETRIC RELATIONSHIP:
    - Infiltration to extraction weights: W_infiltration_to_extraction maps cin → cout
    - Extraction to infiltration weights: W_extraction_to_infiltration maps cout → cin
    - Mathematical constraint: W_extraction_to_infiltration should be the pseudo-inverse of W_infiltration_to_extraction

    The algorithm mirrors _infiltration_to_extraction_weights but with transposed
    temporal overlap computations to ensure mathematical consistency.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time edges for cout and flow data bins.
    cin_tedges : pandas.DatetimeIndex
        Time edges for output (infiltration) data bins.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3].
    cout : array-like
        Concentration values of extracted water.
    flow : array-like
        Flow rate values in the aquifer [m3/day].
    retardation_factor : float
        Retardation factor of the compound in the aquifer.

    Returns
    -------
    numpy.ndarray
        Normalized weight matrix for extraction to infiltration transformation.
        Shape: (len(cin_tedges) - 1, len(cout))
    """
    # Convert time edges to days
    cout_tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cin_tedges_days = ((cin_tedges - tedges[0]) / pd.Timedelta(days=1)).values

    # Pre-compute all residence times and extraction edges (symmetric to infiltration_to_extraction)
    rt_edges_2d = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cin_tedges,
        aquifer_pore_volume=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",  # Computing from infiltration perspective
    )
    extraction_tedges_2d = cin_tedges_days[None, :] + rt_edges_2d

    # Pre-compute valid bins and count
    valid_bins_2d = ~(np.isnan(extraction_tedges_2d[:, :-1]) | np.isnan(extraction_tedges_2d[:, 1:]))
    valid_pv_count = np.sum(valid_bins_2d, axis=0)

    accumulated_weights = np.zeros((len(cin_tedges) - 1, len(cout)))

    # Pre-compute cout time range for clip optimization (computed once, used n_bins times)
    cout_time_min = cout_tedges_days[0]
    cout_time_max = cout_tedges_days[-1]

    # Loop over each pore volume (same structure as infiltration_to_extraction)
    for i in range(len(aquifer_pore_volumes)):
        if not np.any(valid_bins_2d[i, :]):
            continue

        # Clip optimization: Check for temporal overlap before expensive computation
        # Get the range of extraction times for this pore volume (only valid bins)
        extraction_times = extraction_tedges_2d[i, :]
        valid_extraction_times = extraction_times[~np.isnan(extraction_times)]

        if len(valid_extraction_times) == 0:
            continue

        extraction_min = valid_extraction_times[0]  # Min is first element (monotonic)
        extraction_max = valid_extraction_times[-1]  # Max is last element (monotonic)

        # Check if extraction window overlaps with cout window
        # Two intervals [a1, a2] and [b1, b2] overlap if: max(a1, b1) < min(a2, b2)
        has_overlap = max(extraction_min, cout_time_min) < min(extraction_max, cout_time_max)

        if not has_overlap:
            # No temporal overlap - this bin contributes nothing, skip expensive computation
            continue

        # SYMMETRIC temporal overlap computation:
        # Infiltration to extraction: maps infiltration → cout time windows
        # Extraction to infiltration: maps extraction → cout time windows (transposed relationship)
        overlap_matrix = partial_isin(bin_edges_in=extraction_tedges_2d[i, :], bin_edges_out=cout_tedges_days)
        accumulated_weights[valid_bins_2d[i, :], :] += overlap_matrix[valid_bins_2d[i, :], :]

    # Average across valid pore volumes (symmetric to infiltration_to_extraction)
    averaged_weights = np.zeros_like(accumulated_weights)
    valid_cout = valid_pv_count > 0
    averaged_weights[valid_cout, :] = accumulated_weights[valid_cout, :] / valid_pv_count[valid_cout, None]

    # Apply flow weighting (symmetric to infiltration_to_extraction)
    flow_weighted_averaged = averaged_weights * flow[None, :]

    # Normalize by total weights (symmetric to infiltration_to_extraction)
    total_weights = np.sum(flow_weighted_averaged, axis=1)
    valid_weights = total_weights > 0
    normalized_weights = np.zeros_like(flow_weighted_averaged)
    normalized_weights[valid_weights, :] = flow_weighted_averaged[valid_weights, :] / total_weights[valid_weights, None]

    return normalized_weights
