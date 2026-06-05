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
from gwtransport.utils import cumulative_flow_volume

# Target number of (streamtube x cout-bin) pairs per tile in the banded weight build. The
# per-tile working set is O(_WEIGHT_BUILD_BLOCK x band), so peak memory is bounded
# independent of record length; chosen to keep the build under ~30 MB while spanning most
# records in one or a few tiles. See _infiltration_to_extraction_weights.
_WEIGHT_BUILD_BLOCK = 100_000


def _infiltration_to_extraction_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp], npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """
    Compute raw streamtube-bundle weights for infiltration to extraction transformation.

    Builds the per-cout-bin sum of streamtube-normalized overlap rows in a
    **compact banded layout**, plus the count of contributing streamtubes per
    cout bin and the zero-flow cout-bin mask. The caller decides how to convert
    these into final weights (strict-validity, count-mean, or padded "constant"
    spin-up); see :func:`_resolve_spinup_mask` and :func:`_resolve_spinup_inputs`.
    Reconstruct the dense ``(n_cout, n_cin)`` matrix with :func:`_densify_weights`.

    The per-streamtube weight is the literal mass-flux / water-flux ratio for
    that streamtube and cout bin: each contributing row sums to 1 to ULP.
    Equal-mass pore-volume bins from a gamma APVD discretization carry equal
    flow at the outlet (steady-streamline assumption), so the bundle output is
    the arithmetic mean over contributing streamtubes.

    Pure advection (``D_m = 0``, ``alpha_L = 0``) is volume-stationary. Let
    ``Vi`` be the cumulative throughflow volume at the cin edges and ``Vc`` the
    same cumulative volume sampled at the cout edges. A streamtube of retarded
    pore volume ``r = R * V_pore`` carries each cout edge back to the
    infiltration time at cumulative volume ``Vc_edge - r``. The cout bin's
    source window in infiltration time spans one cout bin's worth of volume, so
    it overlaps only a few cin bins. The nonzeros of cout row ``k`` therefore
    span only the residence-time spread of the APVD, ``[col_start[k],
    col_start[k] + full_band)``, and are accumulated into one banded buffer over
    a **pore-volume loop** -- ``O(n_cout * full_band)`` memory regardless of
    record length or ``n_pv``. The overlap is the flow-weighted time overlap
    normalized by the window's total in-range flow, so each contributing row
    sums to 1 exactly: the exact flow-weighted overlap, matching the dense
    ``residence_time`` + ``partial_isin`` reference to machine precision, not an
    approximation.

    A streamtube whose source volume leaves ``[Vi[0], Vi[-1]]`` -- or a cout
    edge outside the cin time range -- maps to NaN and is **dropped, not
    clipped**: its whole window for that cout bin is discarded (mirroring the
    dense build, where a single NaN residence-time edge poisons the row).

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
    band_vals : numpy.ndarray
        Sum over streamtubes of per-streamtube normalized overlap rows in
        banded layout. Shape: (len(cout_tedges) - 1, full_band). Slot
        ``band_vals[k, b]`` is the weight on cin bin ``col_start[k] + b``;
        for a cout bin ``k`` with ``c`` contributing streamtubes the row
        sums to ``c`` (not ``n_pv`` and not 1).
    col_start : numpy.ndarray of int
        Shape: (len(cout_tedges) - 1,). First cin bin index of each cout
        row's band. Defaults to 0 for rows with no contributing streamtube.
    contributing_bins : numpy.ndarray of int
        Shape: (len(cout_tedges) - 1,). Number of streamtubes that
        actually contributed to each cout bin (had a source window fully
        inside the cin volume range).
    zero_flow_cout : numpy.ndarray of bool
        Shape: (len(cout_tedges) - 1,). True for cout bins with zero
        time-averaged extraction flow over their interval.

    See Also
    --------
    _densify_weights : Reconstruct the dense (n_cout, n_cin) matrix.
    """
    cin_tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    flow = np.asarray(flow, dtype=float)

    n_cout = len(cout_tedges) - 1
    n_cin = len(tedges) - 1
    n_pv = len(aquifer_pore_volumes)

    # Cumulative throughflow volume at cin edges (Vi) and sampled at cout edges (Vc).
    vi = cumulative_flow_volume(flow, np.diff(cin_tedges_days))
    vc = np.interp(cout_tedges_days, cin_tedges_days, vi)

    # A cout bin's time-averaged extraction flow is zero exactly when no throughflow
    # volume passes during the bin, i.e. its cumulative-volume width is zero. This is
    # bit-identical to the dense flow-overlap-matrix product but is O(n_cout), not O(N^2).
    zero_flow_cout = np.diff(vc) == 0

    if n_pv == 0:
        return np.zeros((n_cout, 1)), np.zeros(n_cout, dtype=np.intp), np.zeros(n_cout, dtype=np.intp), zero_flow_cout

    r = np.sort(np.asarray(aquifer_pore_volumes, dtype=float) * retardation_factor)

    # Cumulative source volume at each cout edge. Out-of-range cout edges have no source
    # volume; NaN there propagates so the adjacent windows are dropped (not clipped),
    # matching the dense build.
    edge_in_range = (cout_tedges_days >= cin_tedges_days[0]) & (cout_tedges_days <= cin_tedges_days[-1])
    src_edge = np.where(edge_in_range, vc, np.nan)

    # Tile the cout axis so the (n_pv, tile, band) overlap tensor never materialises for the
    # whole record: peak memory stays O(block) regardless of record length, while each tile
    # stays fully vectorised over its streamtubes. Most records span one or a few tiles. Each
    # tile sizes its own band independently and is right-padded to the global width at the end.
    block = max(1, _WEIGHT_BUILD_BLOCK // n_pv)
    col_start = np.zeros(n_cout, dtype=np.intp)
    contributing_bins = np.zeros(n_cout, dtype=np.intp)
    tiles: list[tuple[int, npt.NDArray[np.floating]]] = []
    full_band = 1
    for start in range(0, n_cout, block):
        stop = min(start + block, n_cout)
        m = stop - start

        # Back-project this tile's cout edges by every streamtube to infiltration time.
        infil = np.interp(
            (src_edge[start : stop + 1] - r[:, None]).ravel(), vi, cin_tedges_days, left=np.nan, right=np.nan
        ).reshape(n_pv, m + 1)
        win_lo, win_hi = infil[:, :-1], infil[:, 1:]  # (n_pv, m) infiltration-time window per cout bin
        contained = np.isfinite(win_lo) & np.isfinite(win_hi)
        j_lo = np.clip(np.searchsorted(cin_tedges_days, win_lo, side="right") - 1, 0, n_cin - 1)
        j_hi = np.clip(np.searchsorted(cin_tedges_days, win_hi, side="left"), 0, n_cin)

        # Tile band = union of contained streamtube windows; per_band = widest single window.
        any_contained = contained.any(axis=0)
        tile_start = np.where(any_contained, np.where(contained, j_lo, n_cin).min(axis=0), 0)
        row_hi = np.where(contained, j_hi, 0).max(axis=0)
        tile_band = min(int(np.max(np.where(any_contained, row_hi - tile_start, 0))) + 1, n_cin)
        per_band = min(int(np.max(np.where(contained, j_hi - j_lo, 0))) + 1, n_cin)

        # Per-streamtube flow-weighted overlap on its own narrow window, normalised so each
        # contributing row sums to 1, then scatter-summed over streamtubes into the tile's
        # banded buffer (offset by each row's tile_start).
        cols = j_lo[:, :, None] + np.arange(per_band)[None, None, :]  # (n_pv, m, per_band)
        in_cin = contained[:, :, None] & (cols < n_cin)
        cols_clipped = np.clip(cols, 0, n_cin - 1)
        overlap = np.maximum(
            0.0,
            np.minimum(win_hi[:, :, None], cin_tedges_days[cols_clipped + 1])
            - np.maximum(win_lo[:, :, None], cin_tedges_days[cols_clipped]),
        )
        flux = np.where(in_cin, flow[cols_clipped] * overlap, 0.0)
        window_flux = flux.sum(axis=2)  # (n_pv, m) total in-range flow per window
        contributes = contained & (window_flux > 0)
        np.divide(flux, window_flux[:, :, None], out=flux, where=contributes[:, :, None])

        # Only nonzero in-window slots scatter; their banded offset is < tile_band by
        # construction (a contributing slot lies in [tile_start, row_hi)).
        keep = flux > 0
        offset = cols_clipped - tile_start[None, :, None]
        row_idx = np.broadcast_to(np.arange(m)[None, :, None], cols.shape)
        tile_vals = (
            np
            .bincount((row_idx * tile_band + offset)[keep], weights=flux[keep], minlength=m * tile_band)
            .astype(float, copy=False)
            .reshape(m, tile_band)
        )

        col_start[start:stop] = tile_start
        contributing_bins[start:stop] = contributes.sum(axis=0)
        tiles.append((start, tile_vals))
        full_band = max(full_band, tile_band)

    band_vals = np.zeros((n_cout, full_band))
    for start, tile_vals in tiles:
        band_vals[start : start + tile_vals.shape[0], : tile_vals.shape[1]] = tile_vals

    return band_vals, col_start, contributing_bins, zero_flow_cout


def _resolve_spinup_mask(
    *,
    band_vals: npt.NDArray[np.floating],
    col_start: npt.NDArray[np.intp],
    contributing_bins: npt.NDArray[np.intp],
    zero_flow_cout: npt.NDArray[np.bool_],
    n_pv: int,
    spinup: float | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.intp], npt.NDArray[np.bool_]]:
    """Convert raw banded bundle outputs into final banded weights + invalid mask.

    Parameters
    ----------
    band_vals : numpy.ndarray
        Per-cout-bin sum of streamtube-normalized rows in banded layout from
        :func:`_infiltration_to_extraction_weights`. Shape (n_cout, full_band).
    col_start : numpy.ndarray of int
        First cin bin index of each cout row's band. Passed through unchanged.
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
        Final banded weight matrix of the same shape as ``band_vals``.
        Rows where the policy is not satisfied are zero.
    col_start : numpy.ndarray of int
        The input ``col_start``, returned unchanged for caller convenience.
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
        return np.zeros_like(band_vals), col_start, np.ones(band_vals.shape[0], dtype=bool)

    if spinup is None:
        valid = (contributing_bins == n_pv) & ~zero_flow_cout
        weights = np.zeros_like(band_vals)
        weights[valid, :] = band_vals[valid, :] / n_pv
        return weights, col_start, ~valid

    if not (isinstance(spinup, (int, float)) and not isinstance(spinup, bool) and 0.0 <= spinup <= 1.0):
        msg = f"spinup must be None, 'constant', or float in [0, 1]; got {spinup!r}"
        raise ValueError(msg)

    valid = (contributing_bins >= spinup * n_pv) & ~zero_flow_cout & (contributing_bins > 0)
    weights = np.zeros_like(band_vals)
    safe_div = np.where(valid, contributing_bins, 1)
    weights[valid, :] = band_vals[valid, :] / safe_div[valid, None]
    return weights, col_start, ~valid


def _densify_weights(
    band_vals: npt.NDArray[np.floating], col_start: npt.NDArray[np.intp], n_cin: int
) -> npt.NDArray[np.floating]:
    """Reconstruct the dense (n_cout, n_cin) weight matrix from the banded layout.

    Inverse of the banded packing produced by
    :func:`_infiltration_to_extraction_weights` and
    :func:`_resolve_spinup_mask`: row ``k`` places ``band_vals[k]`` at cin
    columns ``[col_start[k], col_start[k] + full_band)``, dropping band slots
    that fall past ``n_cin`` (the right-edge padding).

    Parameters
    ----------
    band_vals : numpy.ndarray
        Banded weights of shape (n_cout, full_band).
    col_start : numpy.ndarray of int
        First cin bin index of each row's band, shape (n_cout,).
    n_cin : int
        Number of cin bins (dense column count).

    Returns
    -------
    numpy.ndarray
        Dense weight matrix of shape (n_cout, n_cin).
    """
    n_cout, full_band = band_vals.shape
    dense = np.zeros((n_cout, n_cin))
    cols = col_start[:, None] + np.arange(full_band)[None, :]
    in_range = cols < n_cin
    rows = np.broadcast_to(np.arange(n_cout)[:, None], cols.shape)
    dense[rows[in_range], cols[in_range]] = band_vals[in_range]
    return dense


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
