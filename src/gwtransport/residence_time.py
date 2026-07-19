"""
Residence Time Calculations for Retarded Compound Transport.

This module provides functions to compute residence times for compounds traveling through
aquifer systems, accounting for flow variability, pore volume, and retardation due to
physical or chemical interactions with the aquifer matrix. Residence time represents the
duration a compound spends traveling from infiltration to extraction points, depending on
flow rate (higher flow yields shorter residence time), pore volume (larger volume yields
longer residence time), and retardation factor (interaction with matrix yields longer
residence time).

Available functions:

- :func:`full` - Compute the flow-weighted mean residence time over
  output bins, per pore volume (full ``(n_pore_volumes, n_bins)`` array). Follows the package's
  bin-edge convention and is the form consumed elsewhere in the package. Supports both forward
  (infiltration to extraction) and reverse (extraction to infiltration) directions.

- :func:`mean` - Compute the mean residence time over output bins for a discrete
  aquifer pore-volume distribution (an array of equally-weighted pore volumes). Collapses the
  pore-volume axis to a single per-bin series. The ``spinup`` policy (default ``"constant"``)
  warm-starts the spin-up by extrapolating the boundary flow.

- :func:`gamma` - Compute the closed-form mean residence time over output bins for a
  (shifted) gamma aquifer pore-volume distribution, with no pore-volume discretization. The
  ``spinup`` policy (default ``"constant"``) warm-starts the spin-up; ``spinup=0.0`` instead
  renormalizes over the covered sub-mass exactly.

- :func:`fraction_explained_full`, :func:`fraction_explained_mean`,
  :func:`fraction_explained_gamma` - Compute the **advective** fraction of each output bin that is
  explained by the flow record: the flow-weighted share of the bin whose retarded advective parcel
  was infiltrated/extracted inside the record. ``full`` returns one row per pore volume, ``mean``
  the equal-weight discrete-APVD mean, and ``gamma`` the closed-form (shifted) gamma-APVD value,
  mirroring :func:`full` / :func:`mean` / :func:`gamma`. These are **purely advective** -- molecular
  diffusion and microdispersion spread each bin over a range of infiltration times that is
  not captured here, so no bin is fully informed once dispersion is present (for that dispersive
  informed fraction use the captured kernel mass of the diffusion coefficient matrix).

- :func:`freundlich_retardation` - Compute concentration-dependent retardation factors from a
  Freundlich isotherm, for use as the ``retardation_factor`` input to the transport functions.

Spin-up period
--------------
The spin-up **region** is determined entirely by the supplied flow record (``tedges``, which
fixes the cumulative throughflow volume ``V`` from ``0`` at the record start to ``V_end`` at the
record end) together with the retarded pore volume ``retardation_factor * V_p`` -- it is not a
length you set. A residence time for an output time needs the corresponding parcel to stay inside
the flow record:

* ``direction='extraction_to_infiltration'`` looks **back** to the infiltration event, so the
  spin-up sits at the **start** of the output record: the residence time of a pore volume ``V_p``
  needs ``V(t) >= retardation_factor * V_p`` (the extracted water was infiltrated before the record
  began otherwise).
* ``direction='infiltration_to_extraction'`` looks **forward** to the extraction event, so the
  spin-up sits at the **end** of the output record: it needs ``V_end - V(t) >= retardation_factor *
  V_p`` (the infiltrated water is extracted after the record ends otherwise).

The spin-up therefore lengthens with both the pore volume and the retardation factor, and is
longest for the largest pore volumes of a distribution.

What happens in that region is governed by a ``spinup`` policy, following the package convention
(see :mod:`gwtransport.advection`); :func:`full`, :func:`mean` and
:func:`gamma` all share the contract ``spinup={'constant'} | None | float in
[0, 1]`` and the default is ``"constant"`` everywhere:

* ``"constant"`` (default) **warm-starts** by extrapolating the boundary flow (flow held constant
  at its first/last value), so no in-record output is ``NaN``.
* ``None`` is strict (no extrapolation), marking a pore volume ``NaN`` for any output bin its parcel
  leaves the record within. Where the pore-volume axis is collapsed -- :func:`mean` over a
  discrete set, :func:`gamma` over the continuum -- the bin mean then **renormalizes**
  over the covered streamtubes / sub-mass, emitted wherever any coverage remains.
* a ``float`` covered-fraction threshold is the strict mode with a minimum coverage gate: the
  renormalized mean is emitted only where the covered streamtube fraction / sub-mass fraction is at
  least ``spinup`` (``0.0`` matches ``None``; larger values demand more coverage). For the
  per-pore-volume :func:`full` there is no axis to collapse, so the ``float`` behaves
  exactly like ``None``.

Output bins lying wholly outside ``tedges`` are ``NaN`` under every policy.

The :func:`fraction_explained_full` / :func:`fraction_explained_mean` /
:func:`fraction_explained_gamma` diagnostics report, per output bin, the advective fraction of the
pore-volume distribution that is out of spin-up (``1.0`` = advectively fully informed, ``0.0`` =
entirely in spin-up) and are the way to locate the spin-up region when the means warm-start over it.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import gamma as gamma_dist

from gwtransport._time import tedges_to_days
from gwtransport.gamma import parse_parameters
from gwtransport.utils import cumulative_flow_volume, linear_interpolate

# Relative slack on the covered-fraction spin-up gate. The covered sub-mass ``den`` equals its
# fully-covered reference only up to summation-reassociation ulps, so an exact ``den >= threshold *
# reference`` comparison spuriously rejects fully-covered bins at the strictest threshold (1.0). The
# slack is far above that float noise (band widths reach ~1e4 pieces, ~1e-12 relative) yet negligible
# as a physical coverage tolerance.
_SPINUP_GATE_RELTOL = 1e-9


def _boundary_extrapolated_map(
    flow: npt.NDArray[np.floating],
    flow_cum: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    pad: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Extend the cumulative-volume -> time map past the record for the ``"constant"`` warm-start.

    The boundary extrapolation slope is ``1/Q``, anchored on the nearest strictly-positive flow so a
    zero-flow boundary bin (whose ``flow_cum`` step is only the strictly-monotone ulp bump) does not
    give a ``1/0`` extrapolation. Bit-identical to ``1/Q_boundary`` when the boundary already carries
    flow.

    Returns
    -------
    volume_map : ndarray
        ``flow_cum`` padded by ``pad`` on each end.
    time_map : ndarray
        ``tedges_days`` padded by ``pad * (1/Q)`` on each end, aligned with ``volume_map``.
        Shared by :func:`full` and :func:`gamma`.
    """
    positive_flow = flow[flow > 0.0]
    inv_q_first = 1.0 / positive_flow[0]
    inv_q_last = 1.0 / positive_flow[-1]
    volume_map = np.concatenate([[flow_cum[0] - pad], flow_cum, [flow_cum[-1] + pad]])
    time_map = np.concatenate([
        [tedges_days[0] - pad * inv_q_first],
        tedges_days,
        [tedges_days[-1] + pad * inv_q_last],
    ])
    return volume_map, time_map


def _resolve_spinup(spinup: str | float | None) -> tuple[bool, float]:
    """Normalize the residence-time ``spinup`` policy to ``(extrapolate, threshold)``.

    The three sibling residence-time functions share one spin-up contract,
    ``{'constant'} | None | float in [0, 1]``, matching the package convention (see
    :mod:`gwtransport.advection`):

    * ``'constant'`` -> ``(True, 0.0)`` -- warm-start by extrapolating the boundary flow.
    * ``None`` -> ``(False, 0.0)`` -- strict map (no extrapolation); a collapsed mean is emitted
      wherever any covered sub-mass / valid streamtube remains.
    * ``float`` in ``[0, 1]`` -> ``(False, float)`` -- strict map, with the covered-fraction
      threshold gating a collapsed mean. ``0.0`` matches ``None``; larger values require a larger
      covered fraction before a bin is emitted, and ``1.0`` is strictest (full coverage required).

    Parameters
    ----------
    spinup : {'constant'}, None, or float in [0, 1]
        Public spin-up policy.

    Returns
    -------
    extrapolate : bool
        Whether to warm-start by extrapolating the boundary flow.
    threshold : float
        Covered-fraction gate applied where the pore-volume axis is collapsed (ignored by the
        per-pore-volume :func:`full`, which is strict per bin).

    Raises
    ------
    ValueError
        If ``spinup`` is not ``'constant'``, ``None``, or a float in ``[0, 1]``.
    """
    if spinup == "constant":
        return True, 0.0
    if spinup is None:
        return False, 0.0
    if isinstance(spinup, int | float) and not isinstance(spinup, bool) and 0.0 <= spinup <= 1.0:
        return False, float(spinup)
    msg = "spinup should be 'constant', None, or a float in [0, 1]"
    raise ValueError(msg)


def _phi_setup(
    flow: npt.NDArray[np.floating],
    flow_cum: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    *,
    extrapolate: bool,
    pad: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Build the antiderivative ``phi(x) = int_0^x T(w) dw`` of the cumulative-volume -> time map ``T``.

    ``T`` is piecewise-linear (knots at the cumulative-volume edges), so ``phi`` is piecewise-quadratic
    with the same knots. With ``extrapolate`` the map is extended one anchor past each end of the record
    at the boundary flow rate (padded by ``pad``) for the ``"constant"`` warm-start; otherwise it is the
    raw record. Shared by :func:`full` and :func:`gamma`.

    Returns
    -------
    phi_v : ndarray
        Cumulative-volume knots of the map (extended by ``pad`` at each end when ``extrapolate``).
    phi_t : ndarray
        Time knots aligned with ``phi_v``.
    phi_knot : ndarray
        ``phi`` evaluated at each volume knot.
    phi_rate : ndarray
        Per-segment ``dV/dt`` (flow) rate between consecutive knots.
    """
    # pad == 0 (no spin-up reach, e.g. retardation_factor or all pore volumes 0) carries no
    # extrapolation and would only add zero-width boundary segments (0/0 -> NaN phi_rate), so skip it.
    if extrapolate and pad > 0.0 and np.any(flow > 0.0):
        phi_v, phi_t = _boundary_extrapolated_map(flow, flow_cum, tedges_days, pad)
    else:
        phi_v, phi_t = flow_cum, tedges_days
    phi_dv = phi_v[1:] - phi_v[:-1]
    phi_rate = phi_dv / (phi_t[1:] - phi_t[:-1])
    phi_knot = np.concatenate([[0.0], np.cumsum(phi_t[:-1] * phi_dv + phi_dv**2 / (2 * phi_rate))])
    return phi_v, phi_t, phi_knot, phi_rate


def _eval_phi(
    x: npt.NDArray[np.floating],
    phi_v: npt.NDArray[np.floating],
    phi_t: npt.NDArray[np.floating],
    phi_knot: npt.NDArray[np.floating],
    phi_rate: npt.NDArray[np.floating],
    *,
    strict_nan: bool = False,
) -> npt.NDArray[np.floating]:
    """Evaluate the piecewise-quadratic antiderivative ``phi`` from :func:`_phi_setup` at ``x``.

    ``x`` is clipped to the map range ``[phi_v[0], phi_v[-1]]`` (the warm-start extrapolation lives in
    that range when padded). With ``strict_nan`` any ``x`` outside the range returns ``NaN`` instead --
    used by :func:`full` so an output bin whose parcel leaves the record is ``NaN``.

    Returns
    -------
    ndarray
        ``phi`` evaluated at ``x``, with the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    xc = np.clip(x, phi_v[0], phi_v[-1])
    j = np.clip(np.searchsorted(phi_v, xc, side="right") - 1, 0, len(phi_rate) - 1)
    dv = xc - phi_v[j]
    out = phi_knot[j] + phi_t[j] * dv + dv * dv / (2 * phi_rate[j])
    if strict_nan:
        out = np.where((x < phi_v[0]) | (x > phi_v[-1]), np.nan, out)
    return out


def full(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volumes: npt.ArrayLike,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    r"""
    Compute the mean residence time over output bins, per pore volume.

    The flow-weighted mean residence time is computed over each output interval
    ``[cout_tedges[i], cout_tedges[i + 1])`` and returned as the full
    ``(n_pore_volumes, n_output_bins)`` array -- one row per entry in
    ``aquifer_pore_volumes``, without collapsing the pore-volume axis. The average is uniform in
    cumulative throughflow volume, matching the package's bin-edge convention (and what the diffusion
    modules consume to compute a per-bin retarded velocity).

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. Length matches ``tedges`` minus one.
    tedges : pandas.DatetimeIndex
        Time edges for the flow data, as datetime64 objects, defining the flow intervals.
    cout_tedges : pandas.DatetimeIndex
        Output time edges as datetime64 objects; ``n + 1`` edges define ``n`` output bins.
    aquifer_pore_volumes : float or array-like
        Pore volume(s) of the aquifer [m³]. A single value or an array of pore volumes
        representing different flow paths.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation:

        * 'extraction_to_infiltration':
          Extraction to infiltration modeling - how many days ago was the extracted water infiltrated.
        * 'infiltration_to_extraction':
          Infiltration to extraction modeling - how many days until the infiltrated water is extracted.

        Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. A value greater
        than 1.0 indicates the compound moves slower than water. Default is 1.0.
    spinup : {'constant'}, None, or float in [0, 1], optional
        How to treat the spin-up zone, where a pore volume's retarded look-back/forward parcel
        leaves the flow record. Matches the package convention (see :mod:`gwtransport.advection`).

        * ``'constant'`` (default): warm-start -- extrapolate the cumulative-volume-to-time map past
          the record at the boundary flow rates (flow held constant at its first/last value), so
          the residence time stays finite. No left-edge (extraction) or right-edge (infiltration)
          spin-up ``NaN``.
        * ``None`` or a ``float`` in ``[0, 1]``: strict -- a pore volume whose parcel leaves the
          record at any point within an output bin is ``NaN`` for that bin (all-or-nothing per bin),
          with no extrapolation. This function returns the full per-pore-volume array, so there is no
          pore-volume axis to collapse; the ``float`` covered-fraction threshold therefore behaves
          identically to ``None`` here and only takes effect once the axis is collapsed in
          :func:`mean` / :func:`gamma`.

        Output bins lying wholly outside ``tedges`` are ``NaN`` under either policy.

    Returns
    -------
    numpy.ndarray
        Mean residence time [days], shape ``(n_pore_volumes, n_output_bins)``. The first
        dimension corresponds to the pore volumes and the second to the ``cout_tedges`` bins.
        Negative or ``NaN`` ``flow`` makes the cumulative-volume map non-monotone or undefined; the
        whole array is returned as ``NaN`` (the function refuses rather than raising).

    Raises
    ------
    ValueError
        If ``tedges`` does not have exactly one more element than ``flow``. If
        ``direction`` is not ``'extraction_to_infiltration'`` or
        ``'infiltration_to_extraction'``. If ``spinup`` is not ``'constant'``, ``None``, or a float
        in ``[0, 1]``.

    See Also
    --------
    fraction_explained_full : Advective fraction of each output bin explained, per pore volume
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction
    :ref:`concept-transport-equation` : Flow-weighted averaging convention

    Notes
    -----
    With the default ``spinup='constant'`` the spin-up zone is warm-started by extrapolating the
    boundary flow, so no in-record bin is ``NaN``; use :func:`fraction_explained_mean` (or
    ``spinup=None``) to locate the spin-up region. See the module docstring (``Spin-up period``)
    for the full rule.

    The single-streamtube residence time :math:`\tau(V) = \mathrm{sign}\,[T(V + \mathrm{sign}\,R V_p)
    - T(V)]` is piecewise-linear in cumulative throughflow volume :math:`V` (:math:`T` is the
    volume :math:`\to` time map, :math:`\mathrm{sign} = -1` for ``extraction_to_infiltration`` and
    :math:`+1` for ``infiltration_to_extraction``). Its flow-weighted bin average is therefore a
    closed-form difference of the antiderivative :math:`\Phi(x) = \int_0^x T(w)\,dw` (piecewise-
    quadratic), evaluated at four points per pore volume and output bin:

    .. math::

        \bar\tau
        = \frac{1}{\Delta V}\int_{V_\mathrm{lo}}^{V_\mathrm{hi}} \tau(V)\,dV
        = \frac{\mathrm{sign}}{\Delta V}\bigl[
            \Phi(V_\mathrm{hi} + \mathrm{sign}\,R V_p) - \Phi(V_\mathrm{lo} + \mathrm{sign}\,R V_p)
            - \Phi(V_\mathrm{hi}) + \Phi(V_\mathrm{lo})\bigr],

    where :math:`V` is cumulative throughflow volume (:math:`dV = Q\,dt`). This avoids materialising a
    per-streamtube integration grid, so memory and time scale as the output size
    :math:`O(n_\mathrm{pore\ volumes}\cdot n_\mathrm{bins})`. A zero-throughflow output bin
    (:math:`\Delta V \to 0`) has a fixed volume while output time advances, so it degenerates to the
    pointwise residence time at the bin's time midpoint.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.residence_time import full
    >>> flow_dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    >>> flow_values = np.full(len(flow_dates) - 1, 100.0)  # Constant flow of 100 m³/day
    >>> mean_times = full(
    ...     flow=flow_values,
    ...     tedges=flow_dates,
    ...     cout_tedges=flow_dates,
    ...     aquifer_pore_volumes=200.0,
    ...     direction="extraction_to_infiltration",
    ... )
    >>> # 200 m³ / 100 m³/day = 2 days residence time; the default constant warm-start
    >>> # extrapolates the boundary flow, so the left-edge spin-up bins are also 2 days
    >>> print(mean_times)  # doctest: +NORMALIZE_WHITESPACE
    [[2. 2. 2. 2. 2. 2. 2. 2. 2.]]
    """
    if direction not in {"extraction_to_infiltration", "infiltration_to_extraction"}:
        msg = "direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
        raise ValueError(msg)
    extrapolate, _ = _resolve_spinup(spinup)

    aquifer_pore_volumes = np.atleast_1d(aquifer_pore_volumes)
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    flow = np.asarray(flow, dtype=float)
    n_pv = len(aquifer_pore_volumes)
    n_out = len(cout_tedges) - 1

    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if np.any(flow < 0) or np.any(np.isnan(flow)):
        return np.full((n_pv, n_out), np.nan)

    tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    # Plateaus in flow_cum from Q = 0 bins make the V -> t inversion multi-valued; bump duplicates by
    # the smallest representable amount so the inverse map is single-valued.
    flow_cum = cumulative_flow_volume(flow, np.diff(tedges_days), strictly_monotone=True)

    # Sign convention: sign = -1 for extraction_to_infiltration, +1 for infiltration_to_extraction;
    # the look-back/forward parcel sits at volume V + shift and tau(V) = sign * (T(V + shift) - T(V)) is
    # piecewise-linear in V (T is the volume -> time map). Its flow-weighted bin average is a closed-
    # form difference of an antiderivative phi with phi' = T (any additive constant cancels in the
    # difference below; over the extrapolated map T is the extended map), so no per-streamtube
    # integration grid is built (memory/time O(n_pore_volumes * n_bins), not O(n_pore_volumes^2 * n_flow)).
    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0
    shift = sign * retardation_factor * aquifer_pore_volumes  # (n_pv,)

    # phi over the cumulative-volume -> time map. With spinup="constant" the map is extrapolated past
    # the record at the boundary flow (padded by the largest reach R * max(V_p)) so phi warm-starts the
    # spin-up; otherwise phi is NaN outside the record so a bin whose parcel leaves it becomes NaN.
    pad = retardation_factor * float(aquifer_pore_volumes.max()) if aquifer_pore_volumes.size else 0.0
    phi_v, phi_t, phi_knot, phi_rate = _phi_setup(flow, flow_cum, tedges_days, extrapolate=extrapolate, pad=pad)
    # The map is only actually extended when there is a positive boundary flow to extrapolate; with
    # all-zero flow (or spinup=None) it stays the raw record, so out-of-record look-backs are NaN.
    strict = not (extrapolate and bool(np.any(flow > 0.0)))

    vol_out = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days, left=np.nan, right=np.nan)
    v_lo = vol_out[:-1]
    v_hi = vol_out[1:]
    dvol = v_hi - v_lo
    bins_within = np.isfinite(v_lo) & np.isfinite(v_hi)

    phi_base = _eval_phi(vol_out, phi_v, phi_t, phi_knot, phi_rate, strict_nan=strict)  # (n_out + 1,)
    phi_shift = _eval_phi(vol_out[None, :] + shift[:, None], phi_v, phi_t, phi_knot, phi_rate, strict_nan=strict)

    tol = 1e6 * np.spacing(float(np.max(np.abs(flow_cum)))) if flow_cum.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (
            sign * ((phi_shift[:, 1:] - phi_shift[:, :-1]) - (phi_base[1:] - phi_base[:-1])[None, :]) / dvol[None, :]
        )
    # Zero-throughflow output bin (dvol -> 0): the volume is fixed while output time advances, so the
    # flow-weighted average degenerates to the pointwise tau at the bin time midpoint, sign*(T(V_lo +
    # shift) - t_mid). A direct ratio there would catastrophically cancel. Only in-record zero-flow
    # bins reach it, so skip the interp entirely when none are present.
    degenerate = bins_within & (dvol <= tol)
    if np.any(degenerate):
        t_mid = 0.5 * (cout_tedges_days[:-1] + cout_tedges_days[1:])
        nan_outside = np.nan if strict else None
        with np.errstate(divide="ignore", invalid="ignore"):
            t_lookback = linear_interpolate(
                x_ref=phi_v, y_ref=phi_t, x_query=v_lo[None, :] + shift[:, None], left=nan_outside, right=nan_outside
            )
            point = sign * (t_lookback - t_mid[None, :])
        result = np.where(dvol[None, :] > tol, result, point)
    return np.where(bins_within[None, :], result, np.nan)


def mean(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volumes: npt.ArrayLike,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    r"""
    Compute the mean residence time over output bins for a discrete APVD.

    The mean is taken over a **discrete** set of equally-weighted aquifer pore volumes -- one
    streamtube per entry in ``aquifer_pore_volumes``. Each streamtube's flow-weighted bin average
    is computed with :func:`full` and the pore-volume axis is then collapsed to a
    single per-output-bin series by averaging over the streamtubes that are valid in each bin. For
    a continuous (shifted) gamma pore-volume distribution evaluated in closed form, use
    :func:`gamma`.

    The mean is over the valid streamtubes,

    .. math::

        \bar\tau_b = \frac{1}{|V_b|}\sum_{i \in V_b} \tau_{i,b},
        \qquad V_b = \{\, i : \tau_{i,b}\ \mathrm{finite} \,\}.

    With the default ``spinup='constant'`` every streamtube is finite within the flow record
    (the boundary flow is extrapolated), so this is simply the mean over all pore volumes; with
    ``spinup=None`` it renormalizes over the streamtubes that have broken through.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. Length matches ``tedges`` minus one.
    tedges : pandas.DatetimeIndex
        Time edges for the flow data, as datetime64 objects, defining the flow intervals.
    cout_tedges : pandas.DatetimeIndex
        Output time edges as datetime64 objects; ``n + 1`` edges define ``n`` output bins.
    aquifer_pore_volumes : array-like
        Discrete pore volumes [m³], one per (equally-weighted) streamtube. A single value
        collapses to the per-streamtube mean of :func:`full`.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation:
        * 'extraction_to_infiltration': how many days ago was the extracted water infiltrated
        * 'infiltration_to_extraction': how many days until the infiltrated water is extracted
        Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.
    spinup : {'constant'}, None, or float in [0, 1], optional
        Spin-up policy, sharing the contract of :func:`gamma`. ``'constant'``
        (default) warm-starts by extrapolating the boundary flow so no in-record bin is ``NaN``;
        ``None`` leaves spin-up streamtubes ``NaN`` and the mean renormalizes over those that have
        broken through (emitted wherever at least one streamtube is valid). A ``float`` in
        ``[0, 1]`` is the covered-fraction threshold: the renormalized mean is emitted only where
        the fraction of valid streamtubes is at least ``spinup`` (``0.0`` matches ``None``; ``1.0``
        demands every streamtube; larger values demand more streamtubes to have broken through). Use
        :func:`fraction_explained_mean` to
        locate the spin-up region.

    Returns
    -------
    numpy.ndarray
        Mean residence time [days], shape ``(n_output_bins,)``. Output bins with no valid
        streamtube (outside the flow record, or -- with ``spinup=None`` -- fully in the spin-up
        zone) are NaN; with a ``float`` ``spinup`` so are bins whose valid-streamtube fraction is
        below the threshold. Negative or ``NaN`` ``flow`` makes the cumulative-volume map non-monotone
        or undefined; the whole series is returned as ``NaN`` (the function refuses rather than raising).

    See Also
    --------
    gamma : Exact closed-form mean for a continuous (shifted) gamma APVD
    full : Per-pore-volume mean residence time over output bins
    fraction_explained_mean : Advective fraction of each output bin explained by the record
    gwtransport.gamma.bins : Discretize a gamma APVD into pore-volume bins
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction

    Notes
    -----
    With ``spinup=None`` the spin-up is **all-or-nothing per streamtube**: a streamtube whose
    look-back/forward parcel leaves the flow record part-way through an output bin has a ``NaN`` bin
    average (inherited from :func:`full`) and is dropped from that bin's mean
    entirely, rather than contributing its partially-covered share; the bin is ``NaN`` only once
    every streamtube is in spin-up. In that mode the discrete mean differs from
    :func:`gamma`, which renormalizes over the covered sub-mass exactly. See the
    module docstring (``Spin-up period``) for the full rule.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.residence_time import mean
    >>> flow_dates = pd.date_range(start="2023-01-01", end="2023-02-10", freq="D")
    >>> flow_values = np.full(len(flow_dates) - 1, 100.0)  # 100 m³/day
    >>> tau_bar = mean(
    ...     flow=flow_values,
    ...     tedges=flow_dates,
    ...     cout_tedges=flow_dates,
    ...     aquifer_pore_volumes=[400.0, 600.0],  # two equally-weighted streamtubes
    ... )
    >>> # Deep in the record: mean pore volume 500 / 100 m³/day = 5 days
    >>> float(np.round(tau_bar[-1], 6))
    5.0
    """
    _, threshold = _resolve_spinup(spinup)
    rt = full(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        direction=direction,
        retardation_factor=retardation_factor,
        spinup=spinup,
    )

    # Mean over the streamtubes that are valid (non-NaN) in each output bin; bins with no valid
    # streamtube reduce to 0/0 and are NaN. With spinup='constant' every in-record streamtube is
    # finite, so this is the plain mean; otherwise it renormalizes over the broken-through set and
    # a float covered-fraction threshold further NaNs bins where too few streamtubes have arrived.
    n_streamtubes = rt.shape[0]
    valid_count = np.isfinite(rt).sum(axis=0)
    with np.errstate(invalid="ignore"):
        bin_mean = np.nansum(rt, axis=0) / valid_count
    if threshold > 0.0:
        bin_mean = np.where(valid_count >= threshold * n_streamtubes, bin_mean, np.nan)
    return bin_mean


def gamma(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
    spinup: str | float | None = "constant",
    _max_tile_elements: int = 1_000_000,
) -> npt.NDArray[np.floating]:
    r"""
    Compute the mean residence time over output bins for a (shifted) gamma APVD.

    The expectation over a (shifted) gamma aquifer pore-volume distribution (APVD),
    parameterized by either ``(mean, std, loc)`` or ``(alpha, beta, loc)``, is taken in closed
    form -- no pore-volume binning and no ``n_bins`` accuracy/cost knob. The bin mean is
    flow-weighted (uniform in cumulative volume), matching the bin-edge convention of the
    package, and a single per-output-bin series is returned.

    The single-streamtube residence time is piecewise-linear in the pore volume :math:`V_p`, so
    its per-bin time integral :math:`G_b(V_p) = \int_{\mathrm{bin}} \tau\,dV` is piecewise-
    quadratic in :math:`V_p` and the covered length :math:`L_b(V_p)` piecewise-linear. The bin
    mean is the ratio of two closed-form integrals against the gamma density -- its zeroth,
    first and second partial moments (regularized incomplete gamma) -- formed once after
    integrating. The ``spinup`` policy sets what happens where part of the APVD lacks flow
    history: ``'constant'`` (default) extrapolates the boundary flow over the full distribution
    (the package default warm-start), while a ``float`` threshold renormalizes the mean over the
    covered sub-mass (``0.0`` reproduces the exact covered-sub-mass conditional mean).

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. Length matches ``tedges`` minus one.
    tedges : pandas.DatetimeIndex
        Time edges for the flow data, as datetime64 objects, defining the flow intervals.
    cout_tedges : pandas.DatetimeIndex
        Output time edges as datetime64 objects; ``n + 1`` edges define ``n`` output bins.
    mean : float, optional
        Mean of the gamma APVD [m³]. Must be strictly greater than ``loc``. Provide either
        ``(mean, std)`` or ``(alpha, beta)``.
    std : float, optional
        Standard deviation of the gamma APVD [m³]. Must be positive.
    loc : float, optional
        Location (lower bound of support) of the gamma APVD [m³]; a guaranteed minimum pore
        volume. Must satisfy ``0 <= loc < mean``. Default is 0.0.
    alpha : float, optional
        Shape parameter of the gamma APVD (must be > 0).
    beta : float, optional
        Scale parameter of the gamma APVD (must be > 0).
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation:
        * 'extraction_to_infiltration': how many days ago was the extracted water infiltrated
        * 'infiltration_to_extraction': how many days until the infiltrated water is extracted
        Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.
    spinup : {'constant'}, None, or float in [0, 1], optional
        How to treat the spin-up zone, where part of the gamma APVD lacks flow history. Matches
        the package convention (see :mod:`gwtransport.advection`).

        * ``'constant'`` (default): warm-start -- extrapolate the cumulative-volume-to-time map past
          the record at the boundary flow rates (flow held constant at its first/last value) and
          integrate the full distribution, so no in-record bin is ``NaN``.
        * ``None`` or a ``float`` in ``[0, 1]``: renormalize the mean over the covered sub-mass,
          emitting a bin only where the covered fraction of the distribution is at least the
          threshold. ``None`` and ``0.0`` both give the exact covered-sub-mass conditional mean
          (emit whenever any sub-mass is covered); larger values demand a larger covered fraction,
          and ``1.0`` requires the full distribution to be covered.

        Output bins lying wholly outside ``tedges`` are ``NaN`` under either policy.

    Returns
    -------
    numpy.ndarray
        APVD-mean residence time [days], shape ``(n_output_bins,)``. Output bins outside the flow
        record are NaN; with a ``float`` ``spinup`` so are bins whose covered fraction is below the
        threshold. Negative or ``NaN`` ``flow`` makes the cumulative-volume map non-monotone or
        undefined; the whole series is returned as ``NaN`` (the function refuses rather than raising).

    Raises
    ------
    ValueError
        If ``tedges`` does not have exactly one more element than ``flow``. If ``direction``
        is not ``'extraction_to_infiltration'`` or ``'infiltration_to_extraction'``. If ``spinup``
        is not ``'constant'``, ``None``, or a float in ``[0, 1]``. Gamma parameter validation is
        delegated to :func:`gwtransport.gamma.parse_parameters`.

    See Also
    --------
    mean : Equally-weighted mean for a discrete set of pore volumes
    full : Per-pore-volume mean residence time over output bins
    fraction_explained_mean : Advective fraction of each output bin explained by the record
    gwtransport.gamma.bins : Discretize a gamma APVD into pore-volume bins
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model

    Notes
    -----
    With the default ``spinup='constant'`` the spin-up is warm-started exactly as in
    :func:`mean` (constant-boundary-flow extrapolation) -- the same warm-start policy applies across
    the whole distribution (``mean`` discretizes it, ``gamma`` integrates it in closed form). With
    ``spinup=0.0`` the spin-up is instead handled by exact covered-sub-mass renormalization: each
    output bin integrates over only the pore-volume sub-range with sufficient flow history. See the
    module docstring (``Spin-up period``) for the full rule.

    The closed form uses regularized incomplete-gamma CDFs. For an extremely narrow APVD
    (``alpha = (mean / std) ** 2`` above ~1e7, i.e. ``std / mean`` below ~3e-4) SciPy's incomplete
    gamma loses precision in its far tail and the result degrades to ~1e-5 relative; such a
    distribution is effectively a single pore volume, so :func:`mean` with one bin is the exact
    alternative there.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.residence_time import gamma
    >>> flow_dates = pd.date_range(start="2023-01-01", end="2023-02-10", freq="D")
    >>> flow_values = np.full(len(flow_dates) - 1, 100.0)  # 100 m³/day
    >>> tau_bar = gamma(
    ...     flow=flow_values,
    ...     tedges=flow_dates,
    ...     cout_tedges=flow_dates,
    ...     mean=500.0,
    ...     std=100.0,
    ...     direction="extraction_to_infiltration",
    ... )
    >>> # Deep in the record the mean residence time approaches mean / flow = 5 days
    >>> float(np.round(tau_bar[-1], 6))
    5.0
    """
    if direction not in {"extraction_to_infiltration", "infiltration_to_extraction"}:
        msg = "direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
        raise ValueError(msg)
    extrapolate, spinup_threshold = _resolve_spinup(spinup)

    alpha, beta, loc = parse_parameters(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta)

    flow = np.asarray(flow, dtype=float)
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    n_out = len(cout_tedges) - 1

    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if np.any(flow < 0) or np.any(np.isnan(flow)):
        return np.full(n_out, np.nan)

    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0
    r = retardation_factor

    tedges_days = tedges_to_days(tedges)
    flow_cum = cumulative_flow_volume(flow, np.diff(tedges_days), strictly_monotone=True)
    v_end = flow_cum[-1]
    n_edges = len(flow_cum)

    # Finite support: drop the gamma tails (mass ~1e-13, far below the discretization error this
    # closed form replaces). Restricting the integral to [support_lo, support_hi] is what keeps
    # the per-bin flow-edge band -- and the gamma CDF evaluations -- bounded.
    tail = 1e-13
    support_lo = float(loc + gamma_dist.ppf(tail, alpha, scale=beta))
    support_hi = float(loc + gamma_dist.ppf(1.0 - tail, alpha, scale=beta))

    # phi(v) = int_0^v T(w) dw with T the inverse cumulative-volume map (piecewise-linear); phi is
    # piecewise-quadratic with knots at the cumulative-volume edges, so the per-bin time integral of
    # tau is a difference of phi at the look-back/forward limits. With spinup="constant" the map is
    # extended past the record at the boundary flow rates (one anchor each end, padded by the largest
    # reach r*support_hi) so phi extrapolates the spin-up; with a float spinup it stays clipped to
    # [0, v_end] (the covered sub-mass only).
    phi_v, phi_t, phi_knot, phi_rate = _phi_setup(
        flow, flow_cum, tedges_days, extrapolate=extrapolate, pad=r * support_hi
    )

    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    vol_out = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days, left=np.nan, right=np.nan)
    good_all = np.isfinite(vol_out[:-1]) & np.isfinite(vol_out[1:]) & (vol_out[1:] > vol_out[:-1])
    if not np.any(good_all):
        return np.full(n_out, np.nan)
    v_lo_all = np.where(good_all, vol_out[:-1], 0.0)
    v_hi_all = np.where(good_all, vol_out[1:], 1.0)

    # Fixed band of flow edges each bin's look-back/forward sweep can cross over the supported
    # pore volumes. band_width is the global maximum so every tile shares one column layout; a
    # spurious column clips to the support and merely splits a quadratic piece without changing
    # its integral, so the band is an exact superset.
    #
    # The support-shifted sweep alone leaves a gap: for sign > 0 the low edge v_lo + r*support_lo
    # sits above the bin's own v_lo, and for sign < 0 the high edge v_hi - r*support_lo sits below
    # v_hi. In strict mode (spinup other than 'constant') the coverage clamp puts breakpoints on the
    # flow edges inside the bin's own [v_lo, v_hi] window, so those must be columns too or a
    # quadratic piece straddles a flow-edge kink and is mis-integrated. Widen the band to always
    # cover [v_lo, v_hi]; the extra columns clip to the support and split without changing the
    # integral where they are not needed.
    a_min = (
        np.minimum(v_lo_all - r * support_hi, v_lo_all) if sign < 0 else np.minimum(v_lo_all + r * support_lo, v_lo_all)
    )
    a_max = (
        np.maximum(v_hi_all - r * support_lo, v_hi_all) if sign < 0 else np.maximum(v_hi_all + r * support_hi, v_hi_all)
    )
    jlo_all = np.clip(np.searchsorted(flow_cum, a_min, "left") - 1, 0, n_edges - 1)
    jhi_all = np.clip(np.searchsorted(flow_cum, a_max, "right"), 0, n_edges - 1)
    band_width = int((jhi_all - jlo_all).max()) + 1
    band = np.arange(band_width)

    # Tile over output bins to bound peak memory. Each bin carries ~3*band_width pieces through a
    # (3 nodes, 3 phi-args) stack of 9 * n_pieces elements; size the tile to that element budget.
    n_pieces = 3 * band_width + 3
    tile = max(1, _max_tile_elements // (9 * n_pieces))

    result = np.full(n_out, np.nan)
    for t0 in range(0, n_out, tile):
        t1 = min(t0 + tile, n_out)
        nt = t1 - t0
        good = good_all[t0:t1]
        v_lo = v_lo_all[t0:t1]
        v_hi = v_hi_all[t0:t1]
        cols = np.clip(jlo_all[t0:t1, None] + band[None, :], 0, n_edges - 1)
        fcb = flow_cum[cols]
        vlo = v_lo[:, None]
        vhi = v_hi[:, None]

        # Direction-specific breakpoint families: the V_p values where a phi argument (a clipped
        # look-back/forward limit) crosses a flow edge or a validity bound. Clip to the support
        # and sort to form the integration pieces (zero-width pieces contribute nothing).
        if sign < 0:
            cand = np.concatenate([(vhi - fcb) / r, (vlo - fcb) / r, fcb / r, vlo / r, vhi / r], axis=1)
        else:
            cand = np.concatenate(
                [(fcb - vlo) / r, (fcb - vhi) / r, (v_end - fcb) / r, (v_end - vlo) / r, (v_end - vhi) / r], axis=1
            )
        np.clip(cand, support_lo, support_hi, out=cand)
        cand.sort(axis=1)
        edges = np.concatenate([np.full((nt, 1), support_lo), cand, np.full((nt, 1), support_hi)], axis=1)
        lo = edges[:, :-1]
        hi = edges[:, 1:]
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        nodes = np.stack([lo, mid, hi], axis=0)  # (3, nt, n_pieces)

        # G(V_p) at the three quadrature nodes (piece lo/mid/hi) as a difference of phi. The constant
        # term phi(v_hi)/phi(v_lo) does not depend on V_p, so evaluate it once per bin and batch only
        # the three V_p-dependent phi arguments. With spinup="constant" the full bin is integrated
        # against the extrapolated phi (length is the full bin width); with a float spinup the limits
        # clamp to the streamtube's covered sub-interval and the covered length renormalizes.
        if sign < 0:
            a_hi = vhi[None] - r * nodes
            if extrapolate:
                v_start = np.broadcast_to(vlo[None], a_hi.shape)
                a_lo = vlo[None] - r * nodes
                length = np.broadcast_to(vhi[None] - vlo[None], a_hi.shape)
            else:
                v_start = np.maximum(vlo[None], r * nodes)
                a_lo = np.maximum(vlo[None] - r * nodes, 0.0)
                length = np.maximum(vhi[None] - v_start, 0.0)
            phi_const = _eval_phi(v_hi, phi_v, phi_t, phi_knot, phi_rate)
            phi_stack = _eval_phi(np.stack([v_start, a_hi, a_lo]), phi_v, phi_t, phi_knot, phi_rate)
            g = phi_const[None, :, None] - phi_stack[0] - phi_stack[1] + phi_stack[2]
        else:
            a_lo = vlo[None] + r * nodes
            if extrapolate:
                v_stop = np.broadcast_to(vhi[None], a_lo.shape)
                a_hi = vhi[None] + r * nodes
                length = np.broadcast_to(vhi[None] - vlo[None], a_lo.shape)
            else:
                v_stop = np.minimum(vhi[None], v_end - r * nodes)
                a_hi = np.minimum(vhi[None] + r * nodes, v_end)
                length = np.maximum(v_stop - vlo[None], 0.0)
            phi_const = _eval_phi(v_lo, phi_v, phi_t, phi_knot, phi_rate)
            phi_stack = _eval_phi(np.stack([a_hi, a_lo, v_stop]), phi_v, phi_t, phi_knot, phi_rate)
            g = phi_stack[0] - phi_stack[1] - phi_stack[2] + phi_const[None, :, None]
        g = np.where(length > 0, g, 0.0)
        length = np.where(length > 0, length, 0.0)
        g_lo, g_mid, g_hi = g
        l_lo, l_mid, l_hi = length

        # Gamma partial moments over each piece: one CDF per shape on the piece edges, then diff.
        # M1/M2 follow from the shifted-gamma partial-moment identities.
        cdf_edges = edges - loc
        f0 = gamma_dist.cdf(cdf_edges, alpha, scale=beta)
        f1 = gamma_dist.cdf(cdf_edges, alpha + 1, scale=beta)
        f2 = gamma_dist.cdf(cdf_edges, alpha + 2, scale=beta)
        m0 = np.diff(f0, axis=1)
        d1 = np.diff(f1, axis=1)
        m1 = alpha * beta * d1 + loc * m0
        d2 = np.diff(f2, axis=1)
        m2 = alpha * (alpha + 1) * beta**2 * d2 + 2 * loc * alpha * beta * d1 + loc * loc * m0

        # Piece-centred gamma moments mu_k = int (x - mid)^k f over each piece. They are formed by
        # shifting the raw partial moments (m1 - mid m0, m2 - 2 mid m1 + mid^2 m0), which for R > 1
        # subtracts large near-equal terms (|mid| up to support_hi, with m1/m2 carrying the matching
        # mid powers) and catastrophically cancels on near-degenerate pieces. They are, however,
        # bounded purely by the piece geometry -- |x - mid| <= half over the piece -- so clip each to
        # its exact range. This is not a strict no-op on well-conditioned pieces (a raw value may sit
        # a rounding step outside the tight bound; clamping it back is a negligible correction). It
        # earns its place on the near-degenerate pieces, where the cancellation noise -- which
        # otherwise pairs with the 1/half^2 second-difference below to manufacture a spurious
        # contribution -- is forced back into the physically valid band.
        b0 = np.maximum(m0, 0.0)
        mu1 = np.clip(m1 - mid * m0, -half * b0, half * b0)
        mu2 = np.clip(m2 - 2 * mid * m1 + mid**2 * m0, 0.0, half**2 * b0)

        # G is quadratic per piece, L linear; reconstruct each from its three nodes (Lagrange-
        # exact) centred at mid and contract with the moments: int (a(x-mid)^2 + b(x-mid) + c) f
        # = a mu2 + b mu1 + c mu0. Zero-width pieces (duplicate breakpoints) divide by half == 0;
        # their masked result is discarded by np.where.
        safe = half > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            g_a = np.where(safe, (g_lo - 2 * g_mid + g_hi) / (2 * half**2), 0.0)
            g_b = np.where(safe, (g_hi - g_lo) / (2 * half), 0.0)
            l_b = np.where(safe, (l_hi - l_lo) / (2 * half), 0.0)
        int_g = g_a * mu2 + g_b * mu1 + g_mid * m0
        int_l = l_b * mu1 + l_mid * m0

        num = int_g.sum(axis=1)
        den = int_l.sum(axis=1)
        covered = good & (den > 0)
        if spinup_threshold > 0.0:
            # float spinup: emit only where the covered fraction of the APVD reaches the threshold.
            # den is the covered sub-mass; (v_hi - v_lo) * m0.sum is the fully-covered reference (equal
            # up to reassociation ulps when fully covered). The relative slack keeps the strictest
            # threshold (1.0) from rejecting fully-covered bins on that float noise.
            covered &= den >= (spinup_threshold - _SPINUP_GATE_RELTOL) * (v_hi - v_lo) * m0.sum(axis=1)
        tile_result = np.full(nt, np.nan)
        tile_result[covered] = num[covered] / den[covered]
        result[t0:t1] = tile_result

    # Zero-throughflow output bins (Q = 0 over the bin) have a cumulative-volume window only as wide
    # as the strictly-monotone ulp bump, so the bin-average num/den ratio above catastrophically
    # cancels. There the bin-average degenerates to its well-defined zero-width-bin limit: the
    # pointwise gamma-mean residence time at the bin's cumulative volume (matching full).
    dvol = vol_out[1:] - vol_out[:-1]
    tol = 1e6 * np.spacing(float(np.max(np.abs(flow_cum)))) if flow_cum.size else 0.0
    degenerate = good_all & (dvol <= tol)
    if np.any(degenerate):
        v = vol_out[:-1][degenerate]  # (k,) bin cumulative volume (constant over a zero-flow bin)
        # Over a zero-flow bin the volume is fixed but output time advances, so tau ramps linearly with
        # output time; its bin-average is the value at the bin's time midpoint, not at T(v).
        t_v = (0.5 * (cout_tedges_days[:-1] + cout_tedges_days[1:]))[degenerate]
        # Upper V_p integration bound, mirroring the main loop's spin-up handling. With
        # spinup="constant" the extrapolated phi map warm-starts the full support; otherwise a V_p
        # whose look-back/forward parcel leaves the record (e2i: v - r*V_p < 0, i2e: v + r*V_p >
        # v_end) is in spin-up, so integrate only the covered sub-range [support_lo, vp_hi] and let the
        # covered sub-mass renormalize the mean (den_pt below), keeping the parcel inside the raw phi map.
        # Warm-start the full support only when the phi map was actually extended -- i.e. there is a
        # positive boundary flow to extrapolate from (matching full's `strict`). With all-zero flow the
        # map stays the raw record (see _phi_setup), so an out-of-record parcel is in spin-up (-> NaN);
        # the raw `extrapolate` flag alone would instead clamp it to a finite pointwise value (RT-P2).
        if extrapolate and bool(np.any(flow > 0.0)):
            vp_hi = np.full(v.shape, support_hi)
        else:
            vp_hi = np.clip((v if sign < 0 else v_end - v) / r, support_lo, support_hi)
        # tau(v, V_p) = sign * (T(v + sign*r*V_p) - t_mid) is piecewise-linear in V_p with knots where
        # v + sign*r*V_p crosses a phi_v edge; integrate it against the gamma density over the sub-range.
        bp = np.clip(sign * (phi_v[None, :] - v[:, None]) / r, support_lo, vp_hi[:, None])
        bp.sort(axis=1)
        edges_pt = np.concatenate([np.full((v.size, 1), support_lo), bp, vp_hi[:, None]], axis=1)
        lo_pt, hi_pt = edges_pt[:, :-1], edges_pt[:, 1:]
        nodes_pt = np.stack([lo_pt, 0.5 * (lo_pt + hi_pt), hi_pt])  # (3, k, n_pieces)
        a_pt = v[None, :, None] + sign * r * nodes_pt
        tau_lo, tau_mid, tau_hi = sign * (
            np.interp(a_pt.ravel(), phi_v, phi_t).reshape(a_pt.shape) - t_v[None, :, None]
        )
        width = np.where(hi_pt > lo_pt, hi_pt - lo_pt, 1.0)
        slope = np.where(hi_pt > lo_pt, (tau_hi - tau_lo) / width, 0.0)
        mid_pt = 0.5 * (lo_pt + hi_pt)
        cdf_pt = edges_pt - loc
        m0_pt = np.diff(gamma_dist.cdf(cdf_pt, alpha, scale=beta), axis=1)
        m1_pt = alpha * beta * np.diff(gamma_dist.cdf(cdf_pt, alpha + 1, scale=beta), axis=1) + loc * m0_pt
        # Clip the piece-centred first moment mu1 = int (V_p - mid) f to its exact geometric bound
        # |mu1| <= half * m0, mirroring the main loop's mu1 clip above. On a piece whose V_p sweep
        # crosses another zero-flow plateau, the raw (m1 - mid m0) subtracts large near-equal terms and
        # catastrophically cancels; the large piecewise slope (~ r / ulp-bump) then amplifies the noise
        # into a spurious residence-time contribution. Only mu1 is needed here: tau is piecewise-linear
        # in V_p (no quadratic mu2 term).
        b0_pt = np.maximum(m0_pt, 0.0)
        half_pt = 0.5 * (hi_pt - lo_pt)
        mu1_pt = np.clip(m1_pt - mid_pt * m0_pt, -half_pt * b0_pt, half_pt * b0_pt)
        num_pt = (tau_mid * m0_pt + slope * mu1_pt).sum(axis=1)
        den_pt = m0_pt.sum(axis=1)  # covered sub-mass over [support_lo, vp_hi]
        covered_pt = den_pt > 0
        if spinup_threshold > 0.0:
            # float spinup: emit only where the covered fraction reaches the threshold. den_pt is the
            # covered sub-mass; the full-support mass is the fully-covered reference. The same relative
            # slack as the main loop keeps threshold 1.0 robust to reassociation ulps in den_pt.
            full_mass = gamma_dist.cdf(support_hi - loc, alpha, scale=beta) - gamma_dist.cdf(
                support_lo - loc, alpha, scale=beta
            )
            covered_pt &= den_pt >= (spinup_threshold - _SPINUP_GATE_RELTOL) * full_mass
        with np.errstate(divide="ignore", invalid="ignore"):
            result[degenerate] = np.where(covered_pt, num_pt / den_pt, np.nan)
    return result


def fraction_explained_full(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volumes: npt.ArrayLike,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    r"""
    Advective coverage per pore volume: the fraction of each output bin explained by the record.

    For each streamtube (entry in ``aquifer_pore_volumes``) and each output bin
    ``[cout_tedges[i], cout_tedges[i + 1])`` this returns the flow-weighted fraction of the bin whose
    retarded **advective** parcel lies inside the supplied flow record -- the share of the bin's
    throughflow volume for which the look-back infiltration (``extraction_to_infiltration``) or
    look-forward extraction (``infiltration_to_extraction``) event is covered by ``cin``. ``1.0``
    means the whole bin is explained for that pore volume, ``0.0`` that none of it is. The full
    ``(n_pore_volumes, n_output_bins)`` array is returned -- one row per pore volume, mirroring
    :func:`full`.

    .. warning::

        This is a **purely advective** diagnostic: it uses only the cumulative-volume look-back
        ``V(t) - retardation_factor * V_p`` and ignores molecular diffusion and longitudinal
        dispersion. Those spread each output bin over a *range* of infiltration times whose kernel
        tails extend outside any finite record, so a bin that is advectively "fully explained"
        (``1.0``) is not fully informed once dispersion is present. For the dispersive informed
        fraction of an advection-dispersion model use the captured kernel mass (the column sum of the
        diffusion coefficient matrix), not this function.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. Length matches ``tedges`` minus one.
    tedges : pandas.DatetimeIndex
        Time edges for the flow data; ``n + 1`` edges for ``n`` flow values.
    cout_tedges : pandas.DatetimeIndex
        Output time edges; ``n + 1`` edges define ``n`` output bins.
    aquifer_pore_volumes : float or array-like
        Pore volume(s) of the aquifer [m³], one per streamtube.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation. Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Advective coverage [dimensionless], shape ``(n_pore_volumes, n_output_bins)``, values in
        ``[0, 1]``. Output bins lying wholly outside ``tedges`` are ``NaN``. Negative or ``NaN``
        ``flow`` makes the cumulative-volume map non-monotone or undefined; the whole array is
        returned as ``NaN`` (the function refuses rather than raising).

    Raises
    ------
    ValueError
        If ``tedges`` does not have exactly one more element than ``flow``, or if ``direction`` is not
        ``'extraction_to_infiltration'`` or ``'infiltration_to_extraction'``.

    See Also
    --------
    fraction_explained_mean : Equal-weight mean of this over a discrete APVD
    fraction_explained_gamma : Closed-form coverage for a (shifted) gamma APVD
    full : Per-pore-volume mean residence time over output bins
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction
    """
    if direction not in {"extraction_to_infiltration", "infiltration_to_extraction"}:
        msg = "direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
        raise ValueError(msg)

    aquifer_pore_volumes = np.atleast_1d(aquifer_pore_volumes)
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    flow = np.asarray(flow, dtype=float)

    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    n_out = len(cout_tedges) - 1
    # Negative or non-finite flow makes V(t) non-monotone or undefined; refuse to answer (match siblings).
    if np.any(flow < 0) or np.any(np.isnan(flow)):
        return np.full((len(aquifer_pore_volumes), n_out), np.nan)

    tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    flow_cum = cumulative_flow_volume(flow, np.diff(tedges_days))
    v_total = flow_cum[-1]

    vol_out = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days, left=np.nan, right=np.nan)
    v_lo = vol_out[:-1]
    v_hi = vol_out[1:]
    dvol = v_hi - v_lo
    r_vp = retardation_factor * aquifer_pore_volumes

    # The flow-weighted (uniform-in-volume) coverage of [v_lo, v_hi] is a clipped ramp of the
    # in-record indicator: e2i needs V >= R*V_p, i2e needs V <= v_total - R*V_p. A zero-throughflow
    # bin (dvol <= tol) has no volume to average over, so use the pointwise indicator at its volume.
    tol = 1e6 * np.spacing(float(np.max(np.abs(flow_cum)))) if flow_cum.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if direction == "extraction_to_infiltration":
            frac = np.clip((v_hi[None, :] - r_vp[:, None]) / dvol[None, :], 0.0, 1.0)
            point = (r_vp[:, None] <= v_lo[None, :]).astype(float)
        else:
            frac = np.clip(((v_total - r_vp[:, None]) - v_lo[None, :]) / dvol[None, :], 0.0, 1.0)
            point = (r_vp[:, None] <= v_total - v_lo[None, :]).astype(float)
    out = np.where(dvol[None, :] > tol, frac, point)
    out[:, ~(np.isfinite(v_lo) & np.isfinite(v_hi))] = np.nan
    return out


def fraction_explained_mean(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volumes: npt.ArrayLike,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Advective coverage for a discrete APVD: equal-weight mean of :func:`fraction_explained_full`.

    Collapses the pore-volume axis of :func:`fraction_explained_full` to a single per-output-bin
    series by averaging over the equally-weighted streamtubes in ``aquifer_pore_volumes`` -- the
    coverage analogue of :func:`mean`. ``1.0`` means every streamtube fully explains the
    bin, ``0.0`` that none do.

    .. warning::

        Purely advective -- see :func:`fraction_explained_full`. Molecular diffusion and longitudinal
        dispersion spreading are not captured, so a value of ``1.0`` is advective coverage, not full
        dispersive information.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. Length matches ``tedges`` minus one.
    tedges : pandas.DatetimeIndex
        Time edges for the flow data; ``n + 1`` edges for ``n`` flow values.
    cout_tedges : pandas.DatetimeIndex
        Output time edges; ``n + 1`` edges define ``n`` output bins.
    aquifer_pore_volumes : array-like
        Discrete pore volumes [m³], one per (equally-weighted) streamtube.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation. Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Advective coverage [dimensionless], shape ``(n_output_bins,)``, values in ``[0, 1]``.
        Output bins lying wholly outside ``tedges`` are ``NaN``. Negative or ``NaN`` ``flow`` makes
        the cumulative-volume map non-monotone or undefined; the whole series is returned as ``NaN``
        (the function refuses rather than raising).

    See Also
    --------
    fraction_explained_full : Per-pore-volume coverage (the array this averages)
    fraction_explained_gamma : Closed-form coverage for a (shifted) gamma APVD
    mean : Equally-weighted mean residence time for a discrete APVD

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gwtransport.residence_time import fraction_explained_mean
    >>> tedges = pd.date_range("2020-01-01", periods=11, freq="D")
    >>> flow = np.full(10, 100.0)
    >>> fraction_explained_mean(
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     aquifer_pore_volumes=[200.0, 1500.0],
    ... ).tolist()
    [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    """
    return fraction_explained_full(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        direction=direction,
        retardation_factor=retardation_factor,
    ).mean(axis=0)


def fraction_explained_gamma(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    r"""
    Closed-form advective coverage for a (shifted) gamma APVD.

    The expectation of the advective in-record indicator over a (shifted) gamma aquifer pore-volume
    distribution (APVD), parameterized by either ``(mean, std, loc)`` or ``(alpha, beta, loc)``, is
    taken in closed form -- the continuum analogue of :func:`fraction_explained_mean`, with no
    pore-volume binning. For each output bin it returns the flow-weighted fraction of the bin whose
    advective parcel lies inside the flow record.

    The flow-weighted bin average :math:`\frac{1}{\Delta V}\int_{V_\mathrm{lo}}^{V_\mathrm{hi}}
    F_{V_p}(\mathrm{threshold}(V))\,dV` (with :math:`\mathrm{threshold}(V) = V / R` for
    ``extraction_to_infiltration`` and :math:`(V_\mathrm{end} - V) / R` for
    ``infiltration_to_extraction``) is evaluated from the antiderivative of the shifted-gamma CDF,

    .. math::

        \Phi(x) = \int_\mathrm{loc}^{x} F_{V_p}(s)\,ds
                = y\,P(\alpha, y/\beta) - \alpha\beta\,P(\alpha + 1, y/\beta),
        \qquad y = \max(x - \mathrm{loc},\, 0),

    with :math:`P` the regularized lower incomplete gamma function -- two CDF evaluations per output
    edge, no quadrature and no pore-volume binning.

    .. warning::

        Purely advective -- see :func:`fraction_explained_full`. Molecular diffusion and longitudinal
        dispersion are not captured; a value of ``1.0`` is advective coverage, not full dispersive
        information.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. Length matches ``tedges`` minus one.
    tedges : pandas.DatetimeIndex
        Time edges for the flow data; ``n + 1`` edges for ``n`` flow values.
    cout_tedges : pandas.DatetimeIndex
        Output time edges; ``n + 1`` edges define ``n`` output bins.
    mean : float, optional
        Mean of the gamma APVD [m³]. Must be strictly greater than ``loc``. Provide either
        ``(mean, std)`` or ``(alpha, beta)``.
    std : float, optional
        Standard deviation of the gamma APVD [m³]. Must be positive.
    loc : float, optional
        Location (lower bound of support) of the gamma APVD [m³]. Must satisfy ``0 <= loc < mean``.
        Default is 0.0.
    alpha : float, optional
        Shape parameter of the gamma APVD (must be > 0).
    beta : float, optional
        Scale parameter of the gamma APVD (must be > 0).
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation. Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Advective coverage [dimensionless], shape ``(n_output_bins,)``, values in ``[0, 1]``.
        Output bins lying wholly outside ``tedges`` are ``NaN``. Negative or ``NaN`` ``flow`` makes
        the cumulative-volume map non-monotone or undefined; the whole series is returned as ``NaN``
        (the function refuses rather than raising).

    Raises
    ------
    ValueError
        If ``tedges`` does not have exactly one more element than ``flow``, or if ``direction`` is
        not ``'extraction_to_infiltration'`` or ``'infiltration_to_extraction'``. Gamma parameter
        validation is delegated to :func:`gwtransport.gamma.parse_parameters`.

    See Also
    --------
    fraction_explained_mean : Discrete equal-weight APVD coverage
    fraction_explained_full : Per-pore-volume coverage
    gamma : Closed-form mean residence time for a (shifted) gamma APVD
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    """
    if direction not in {"extraction_to_infiltration", "infiltration_to_extraction"}:
        msg = "direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
        raise ValueError(msg)
    alpha, beta, loc = parse_parameters(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta)

    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    flow = np.asarray(flow, dtype=float)

    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    n_out = len(cout_tedges) - 1
    if np.any(flow < 0) or np.any(np.isnan(flow)):
        return np.full(n_out, np.nan)

    r = retardation_factor
    tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    flow_cum = cumulative_flow_volume(flow, np.diff(tedges_days))
    v_total = flow_cum[-1]

    vol_out = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days, left=np.nan, right=np.nan)
    v_lo = vol_out[:-1]
    v_hi = vol_out[1:]
    dvol = v_hi - v_lo

    # threshold(V) per output edge, evaluated once on the full edge array so each shared interior
    # edge's CDF is computed a single time (the per-bin lo/hi are then slices of these):
    #   Phi(x) = int_loc^x F_Vp(s) ds = y P(alpha, y/beta) - alpha beta P(alpha+1, y/beta), y = max(x-loc, 0)
    #   cdf(x) = F_Vp(x) = P(alpha, y/beta)
    threshold = (vol_out if direction == "extraction_to_infiltration" else v_total - vol_out) / r
    y = np.maximum(threshold - loc, 0.0)
    cdf_edge = gamma_dist.cdf(y, alpha, scale=beta)
    phi_edge = y * cdf_edge - alpha * beta * gamma_dist.cdf(y, alpha + 1, scale=beta)

    # Flow-weighted bin average (1/dvol) int F_Vp(threshold(V)) dV = (R/dvol) [Phi(hi) - Phi(lo)]; the
    # i2e threshold decreases in V so its edge order flips. A zero-throughflow bin (dvol <= tol)
    # degenerates to the pointwise CDF at the bin's lower-edge volume.
    tol = 1e6 * np.spacing(float(np.max(np.abs(flow_cum)))) if flow_cum.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        if direction == "extraction_to_infiltration":
            ratio = (r / dvol) * (phi_edge[1:] - phi_edge[:-1])
        else:
            ratio = (r / dvol) * (phi_edge[:-1] - phi_edge[1:])
        point = cdf_edge[:-1]
    out = np.where(dvol > tol, ratio, point)
    out[~(np.isfinite(v_lo) & np.isfinite(v_hi))] = np.nan
    return out


def freundlich_retardation(
    *,
    concentration: npt.ArrayLike,
    freundlich_k: float,
    freundlich_n: float,
    bulk_density: float,
    porosity: float,
) -> npt.NDArray[np.floating]:
    """
    Compute concentration-dependent retardation factors using Freundlich isotherm.

    The Freundlich isotherm relates sorbed concentration s to aqueous concentration C using the
    heterogeneity-index convention (matching :class:`gwtransport.fronttracking.math.FreundlichSorption`
    and :func:`gwtransport.advection.infiltration_to_extraction_nonlinear_sorption`, so a fitted
    ``freundlich_n`` is portable across the package)::

        s = k_f * C ^ (1 / n)

    The retardation factor is computed as::

        R = 1 + (rho_b/θ) * ds/dC = 1 + (rho_b/θ) * k_f * (1/n) * C^(1/n - 1)

    Parameters
    ----------
    concentration : array-like
        Concentration of compound in water [mass/volume]. One value per time bin, consistent
        with the ``flow`` array passed to the transport function.
    freundlich_k : float
        Freundlich coefficient [(m³/kg)^(1/n)] (under s = k_f * C^(1/n) with s dimensionless
        and C in [kg/m³]).
    freundlich_n : float
        Freundlich sorption exponent [dimensionless] (heterogeneity index; ``n = 1`` recovers a
        linear isotherm).
    bulk_density : float
        Bulk density of aquifer material [mass/volume].
    porosity : float
        Porosity of aquifer [dimensionless, 0-1].

    Returns
    -------
    numpy.ndarray
        Retardation factors for each flow interval.
        Length equals len(concentration) for use as retardation_factor in the transport functions.

    Raises
    ------
    ValueError
        If ``porosity`` is not in ``(0, 1)``, if ``bulk_density`` is not positive, if
        ``freundlich_k`` is negative, or if any ``concentration`` is non-positive while
        ``freundlich_n > 1`` (the retardation factor diverges as ``C -> 0``).

    See Also
    --------
    full : Compute residence times from flow and pore volume
    gwtransport.advection.infiltration_to_extraction_nonlinear_sorption : Transport with nonlinear sorption
    :ref:`concept-nonlinear-sorption` : Freundlich isotherm and concentration-dependent retardation

    Examples
    --------
    >>> concentration = np.array([0.1, 0.2, 0.3])  # same length as flow
    >>> R = freundlich_retardation(
    ...     concentration=concentration,
    ...     freundlich_k=0.5,
    ...     freundlich_n=2.0,
    ...     bulk_density=1600,  # kg/m³
    ...     porosity=0.35,
    ... )
    >>> # Use R as retardation_factor in the transport functions
    """
    concentration = np.asarray(concentration)

    if not 0 < porosity < 1:
        msg = f"Porosity must be in (0, 1), got {porosity}"
        raise ValueError(msg)
    if bulk_density <= 0:
        msg = f"Bulk density must be positive, got {bulk_density}"
        raise ValueError(msg)
    if freundlich_k < 0:
        msg = f"Freundlich K must be non-negative, got {freundlich_k}"
        raise ValueError(msg)

    # For n > 1 the Freundlich retardation factor 1 + (rho_b/theta) * k_f * (1/n) * C^(1/n-1)
    # diverges as C -> 0 (the exponent 1/n - 1 < 0). Silently clamping concentration would produce
    # a very large but finite value that depends on an arbitrary regularization constant; instead,
    # refuse the call so the user can decide how to handle non-positive concentrations.
    if freundlich_n > 1.0 and np.any(concentration <= 0):
        msg = "concentration must be strictly positive when freundlich_n > 1 (retardation diverges as C -> 0)"
        raise ValueError(msg)

    inv_n = 1.0 / freundlich_n
    return 1.0 + (bulk_density / porosity) * freundlich_k * inv_n * concentration ** (inv_n - 1.0)
