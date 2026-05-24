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

from gwtransport._time import tedges_to_days
from gwtransport.residence_time import residence_time
from gwtransport.utils import partial_isin


def _infiltration_to_extraction_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """
    Compute raw streamtube-bundle weights for infiltration to extraction transformation.

    Builds the per-cout-bin sum of streamtube-normalized overlap matrices,
    plus the count of contributing streamtubes per cout bin and the
    zero-flow cout-bin mask. The caller decides how to convert these into
    final weights (strict-validity, count-mean, or padded "constant"
    spin-up); see :func:`_resolve_spinup_mask` and
    :func:`_pad_tedges_for_spinup`.

    The per-streamtube weight is the literal mass-flux / water-flux ratio
    for that streamtube and cout bin: each row sums to 1 to ULP. Equal-mass
    pore-volume bins from a gamma APVD discretization carry equal flow at
    the outlet (steady-streamline assumption), so the bundle output is the
    arithmetic mean over contributing streamtubes.

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
    accumulated_weights : numpy.ndarray
        Sum over streamtubes of per-streamtube normalized overlap rows.
        Shape: (len(cout_tedges) - 1, len(tedges) - 1). For a cout bin
        ``k`` with ``c`` contributing streamtubes, the row already sums
        to ``c`` (not ``n_pv`` and not 1).
    contributing_bins : numpy.ndarray of int
        Shape: (len(cout_tedges) - 1,). Number of streamtubes that
        actually contributed to each cout bin (had nonzero flow-weighted
        overlap with the cin range).
    zero_flow_cout : numpy.ndarray of bool
        Shape: (len(cout_tedges) - 1,). True for cout bins with zero
        time-averaged extraction flow over their interval.
    """
    # Convert time edges to days
    cin_tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])

    # Detect zero-flow cout bins: a cout bin is "zero-flow" when the time-
    # averaged extraction flow over its interval is zero. Compute via the
    # cout-vs-cin time-overlap matrix.
    cout_in_cin_overlap = partial_isin(bin_edges_in=cout_tedges_days, bin_edges_out=cin_tedges_days)
    flow_during_cout = cout_in_cin_overlap @ flow
    zero_flow_cout = flow_during_cout == 0

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

    n_pv = len(aquifer_pore_volumes)
    n_cout = len(cout_tedges) - 1
    n_cin = len(tedges) - 1
    accumulated_weights = np.zeros((n_cout, n_cin))
    contributing_bins = np.zeros(n_cout, dtype=int)

    # Loop over each pore volume
    for i in range(n_pv):
        if not np.any(valid_bins_2d[i, :]):
            continue

        # Clip optimization: check for temporal overlap before expensive computation.
        infiltration_times = infiltration_tedges_2d[i, :]
        valid_infiltration_times = infiltration_times[~np.isnan(infiltration_times)]

        if len(valid_infiltration_times) == 0:
            continue

        infiltration_min = valid_infiltration_times[0]  # monotonic, first element is min
        infiltration_max = valid_infiltration_times[-1]  # monotonic, last element is max

        # Two intervals overlap iff max(starts) < min(ends).
        has_overlap = max(infiltration_min, cin_time_min) < min(infiltration_max, cin_time_max)
        if not has_overlap:
            continue

        # partial_isin returns shape (n_cout, n_cin) here: bin_edges_in has length
        # n_cout+1 (one infiltration-time edge per cout edge), bin_edges_out has
        # length n_cin+1 (the cin time edges).
        overlap_matrix = partial_isin(bin_edges_in=infiltration_tedges_2d[i, :], bin_edges_out=cin_tedges_days)

        # Per-streamtube row sums to 1 by construction (mass-flux / water-flux).
        # Rows where row_sums == 0 (zero-flow / zero-overlap) are not normalized
        # and therefore not counted as "contributing" — those bins are deferred
        # to the policy resolver instead.
        flow_weighted_overlap = overlap_matrix * flow[None, :]
        row_sums = np.sum(flow_weighted_overlap, axis=1)
        valid_rows_pv = row_sums > 0
        normalized_overlap = np.zeros_like(flow_weighted_overlap)
        normalized_overlap[valid_rows_pv, :] = flow_weighted_overlap[valid_rows_pv, :] / row_sums[valid_rows_pv, None]

        accumulated_weights[valid_rows_pv, :] += normalized_overlap[valid_rows_pv, :]
        contributing_bins[valid_rows_pv] += 1

    return accumulated_weights, contributing_bins, zero_flow_cout


def _resolve_spinup_mask(
    *,
    accumulated_weights: npt.NDArray[np.floating],
    contributing_bins: npt.NDArray[np.intp],
    zero_flow_cout: npt.NDArray[np.bool_],
    n_pv: int,
    spinup: float | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Convert raw bundle outputs into final weights + invalid mask.

    Parameters
    ----------
    accumulated_weights : numpy.ndarray
        Per-cout-bin sum of streamtube-normalized rows from
        :func:`_infiltration_to_extraction_weights`.
    contributing_bins : numpy.ndarray of int
        Number of streamtubes that contributed to each cout bin.
    zero_flow_cout : numpy.ndarray of bool
        Mask of zero-extraction-flow cout bins.
    n_pv : int
        Total number of streamtubes (length of aquifer_pore_volumes).
    spinup : float in [0, 1] or None
        Spin-up policy. ``None`` requires every streamtube to have
        contributed (strict-validity, mass-conserving across cin → cout
        per row). A float in [0, 1] is the minimum fraction of
        contributing streamtubes for the row to be emitted; the bundle
        is then a count-mean over the contributing subset (NOT
        mass-conserving across the bundle when contributing < n_pv —
        this is the pre-fix behavior of issue #161 made explicit).

    Returns
    -------
    weights : numpy.ndarray
        Final weight matrix of the same shape as ``accumulated_weights``.
        Rows where the policy is not satisfied are zero.
    invalid_mask : numpy.ndarray of bool
        True for cout bins where the policy is not satisfied.

    Raises
    ------
    ValueError
        If ``spinup`` is not ``None`` or a float in ``[0, 1]``. The string
        ``"constant"`` is resolved into a threshold by
        :func:`_resolve_spinup_inputs` before reaching this function.
    """
    if n_pv == 0:
        # No streamtubes means no transport: every cout bin is invalid.
        return np.zeros_like(accumulated_weights), np.ones(accumulated_weights.shape[0], dtype=bool)

    if spinup is None:
        valid = (contributing_bins == n_pv) & ~zero_flow_cout
        weights = np.zeros_like(accumulated_weights)
        weights[valid, :] = accumulated_weights[valid, :] / n_pv
        return weights, ~valid

    if not (isinstance(spinup, (int, float)) and not isinstance(spinup, bool) and 0.0 <= spinup <= 1.0):
        msg = f"spinup must be None, 'constant', or float in [0, 1]; got {spinup!r}"
        raise ValueError(msg)

    valid = (contributing_bins >= spinup * n_pv) & ~zero_flow_cout & (contributing_bins > 0)
    weights = np.zeros_like(accumulated_weights)
    safe_div = np.where(valid, contributing_bins, 1)
    weights[valid, :] = accumulated_weights[valid, :] / safe_div[valid, None]
    return weights, ~valid


# Pad duration used by the wide-edges spinup mode. 100 years is overkill for
# any reasonable groundwater residence time + Gaussian-kernel tail, but it
# matches the analytical diffusion module's historical extension and keeps
# the wide-bin numerics stable across pathological inputs.
_DIFFUSION_WIDE_PAD = pd.Timedelta(days=36500)


def _resolve_spinup_inputs_wide_edges(
    spinup: object,
    *,
    tedges: pd.DatetimeIndex,
) -> tuple[pd.DatetimeIndex, float | None]:
    """Validate ``spinup`` and apply diffusion-style wide-edge padding.

    Unlike :func:`_resolve_spinup_inputs` (which prepends ``n_pad`` uniform
    bins for advection-only physics-precise warm-start), this helper
    extends only the first and last edges by 100 years each. Single wide
    boundary bins are used so the diffusion module's Gaussian kernel tail
    is captured without the cost of many extra bins. The length of
    ``tedges`` is preserved; cin and flow lengths are therefore also
    preserved, and inverse-direction outputs need no slicing.

    For the smooth-then-advect path in ``diffusion_fast``, the per-bin
    sigma scales inversely with bin width, so smoothing on the (very wide)
    warm-start bin is automatically negligible and the algorithm remains
    self-consistent.

    Parameters
    ----------
    spinup : None, "constant", or float in [0, 1]
        Public spin-up policy. Same semantics as the parameter on the
        public i2e/e2i functions.
    tedges : pandas.DatetimeIndex
        Original cin/flow time edges (length n_cin + 1).

    Returns
    -------
    weight_tedges : pandas.DatetimeIndex
        Tedges to pass to the weight-matrix computation. Same length as
        ``tedges``; under ``"constant"``, the first and last edges are
        shifted outward by 100 years each.
    threshold : float in [0, 1] or None
        Threshold for :func:`_resolve_spinup_mask`. ``None`` under
        ``spinup`` is ``None`` or ``"constant"``.

    Raises
    ------
    TypeError
        If ``spinup`` has an unsupported type (notably ``bool``).
    ValueError
        If ``spinup`` is a string other than ``"constant"`` or a float
        outside ``[0, 1]``, or if ``len(tedges) < 2``.
    """
    if spinup is None:
        return tedges, None
    if isinstance(spinup, str):
        if spinup != "constant":
            msg = f"spinup string must be 'constant'; got {spinup!r}"
            raise ValueError(msg)
        if len(tedges) < 2:  # noqa: PLR2004
            msg = "spinup='constant' requires len(tedges) >= 2"
            raise ValueError(msg)
        new_tedges = pd.DatetimeIndex([
            tedges[0] - _DIFFUSION_WIDE_PAD,
            *list(tedges[1:-1]),
            tedges[-1] + _DIFFUSION_WIDE_PAD,
        ])
        return new_tedges, None
    if isinstance(spinup, bool) or not isinstance(spinup, (int, float)):
        msg = f"spinup must be None, 'constant', or float in [0, 1]; got {spinup!r}"
        raise TypeError(msg)
    if not (0.0 <= spinup <= 1.0):
        msg = f"spinup float must be in [0, 1]; got {spinup!r}"
        raise ValueError(msg)
    return tedges, float(spinup)


def _resolve_spinup_inputs(
    spinup: object,
    *,
    tedges: pd.DatetimeIndex,
    flow: npt.NDArray[np.floating],
    aquifer_pore_volumes: npt.ArrayLike,
    retardation_factor: float,
    cin: npt.NDArray[np.floating] | None = None,
) -> tuple[pd.DatetimeIndex, npt.NDArray[np.floating], npt.NDArray[np.floating] | None, float | None, int]:
    """Validate ``spinup`` and apply its input-side effects.

    Returns the (possibly padded) tedges, flow, and cin to use for
    weight computation, plus the threshold for the output-side policy
    and the number of bins prepended.

    Modes:

    - ``spinup is None`` — strict-validity. Returns inputs unchanged
      and ``threshold = None``. Mass-conserving but NaN where any
      streamtube has not broken through.
    - ``spinup == "constant"`` — warm-start. Prepends ``n_pad`` bins
      (each of width ``tedges[1] - tedges[0]``) so the total prepended
      duration covers ``retardation_factor * max(aquifer_pore_volumes)
      / flow[0]``. cin and flow are extended with their first observed
      value (constant warm start). Strict-validity on the padded system
      yields no spin-up NaN for cout bins at or after the original
      ``tedges[0]``. Right-edge spin-up (cout extending past
      ``tedges[-1]``) is not addressed.
    - ``spinup`` is a float in ``[0, 1]`` — fraction threshold. Returns
      inputs unchanged and ``threshold = float(spinup)``. *Warning:*
      with ``spinup < 1.0`` the bundle is a count-mean over contributing
      streamtubes; this conserves mass per row but NOT cin → cout (it
      is exactly the issue #161 over-attribution made explicit).

    The first bin width ``tedges[1] - tedges[0]`` is used as the unit
    for prepended bins; this preserves uniformity if the input
    ``tedges`` are uniform (required by the smooth-then-advect path of
    ``diffusion_fast``).

    Parameters
    ----------
    spinup : None, "constant", or float in [0, 1]
        Public spin-up policy.
    tedges : pandas.DatetimeIndex
        Original cin/flow time edges (length n_cin + 1).
    flow : numpy.ndarray
        Flow values; ``flow[0]`` sets the warm-start flow.
    aquifer_pore_volumes : array-like
        Pore volumes; the maximum sets the warm-start duration.
    retardation_factor : float
        Retardation factor.
    cin : numpy.ndarray, optional
        Concentration values to prepend with ``cin[0]``. Pass ``None``
        for the inverse direction (cin is the unknown to recover); the
        returned ``new_cin`` is then ``None``.

    Returns
    -------
    new_tedges : pandas.DatetimeIndex
        Tedges to pass to :func:`_infiltration_to_extraction_weights`.
        Length is ``len(tedges) + n_pad``.
    new_flow : numpy.ndarray
        Flow values aligned with ``new_tedges``; length ``n_cin + n_pad``.
    new_cin : numpy.ndarray or None
        cin values aligned with ``new_tedges`` if ``cin`` was provided;
        otherwise ``None``.
    threshold : float in [0, 1] or None
        Threshold for :func:`_resolve_spinup_mask`. ``None`` (strict)
        for ``spinup is None`` and ``spinup="constant"``.
    n_pad : int
        Number of bins prepended. ``0`` for non-padding modes. Callers
        of the inverse direction must drop the first ``n_pad`` entries
        from the recovered cin to align with the original ``tedges``.

    Raises
    ------
    TypeError
        If ``spinup`` has an unsupported type.
    ValueError
        If ``spinup`` is a string other than ``"constant"`` or a float
        outside ``[0, 1]``.
    """
    if spinup is None:
        return tedges, np.asarray(flow, dtype=float), cin, None, 0
    flow_arr = np.asarray(flow, dtype=float)
    if isinstance(spinup, str):
        if spinup != "constant":
            msg = f"spinup string must be 'constant'; got {spinup!r}"
            raise ValueError(msg)
        # Determine whether padding is feasible. We fall back to strict-validity
        # (no padding) silently when the warm-start is undefined (zero or NaN
        # initial flow) or when the implied padding would be unreasonably large
        # (extreme pore volumes). This keeps the default usable for edge cases
        # while still triggering the strict-validity NaN when the warm-start
        # assumption cannot meaningfully be applied.
        if len(tedges) < 2:  # noqa: PLR2004
            return tedges, flow_arr, cin, None, 0
        q0 = float(flow_arr[0])
        apvs = np.asarray(aquifer_pore_volumes, dtype=float)
        if apvs.size == 0:
            return tedges, flow_arr, cin, None, 0
        v_max = float(np.max(apvs))
        if not (q0 > 0 and v_max > 0):
            return tedges, flow_arr, cin, None, 0
        bin_width = tedges[1] - tedges[0]
        bin_width_days = bin_width / pd.Timedelta(days=1)
        if not bin_width_days > 0:
            return tedges, flow_arr, cin, None, 0
        pad_days = retardation_factor * v_max / q0
        # Add 1 extra bin so the longest streamtube's source window for the
        # earliest original cout bin lies strictly inside the padded range
        # (avoids strict-validity NaN due to floating-point edge alignment).
        n_pad_float = np.ceil(pad_days / bin_width_days) + 1
        # Cap to keep memory bounded; beyond this, "constant" is no longer a
        # meaningful warm-start (the user probably has unphysical pore volumes
        # or extreme retardation), so fall through to strict-validity.
        max_n_pad = max(10_000, 10 * len(flow_arr))
        if not np.isfinite(n_pad_float) or n_pad_float > max_n_pad:
            return tedges, flow_arr, cin, None, 0
        n_pad = int(n_pad_float)
        offsets = pd.TimedeltaIndex(bin_width * np.arange(n_pad, 0, -1))
        new_tedges = (tedges[0] - offsets).append(tedges)
        new_flow = np.concatenate([np.full(n_pad, flow_arr[0]), flow_arr])
        new_cin = np.concatenate([np.full(n_pad, cin[0]), cin]) if cin is not None else None
        return new_tedges, new_flow, new_cin, None, n_pad
    if isinstance(spinup, bool) or not isinstance(spinup, (int, float)):
        msg = f"spinup must be None, 'constant', or float in [0, 1]; got {spinup!r}"
        raise TypeError(msg)
    if not (0.0 <= spinup <= 1.0):
        msg = f"spinup float must be in [0, 1]; got {spinup!r}"
        raise ValueError(msg)
    return tedges, flow_arr, cin, float(spinup), 0
