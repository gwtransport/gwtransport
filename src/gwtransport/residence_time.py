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

- :func:`residence_time_full` - Compute the flow- or time-weighted mean residence time over
  output bins, per pore volume (full ``(n_pore_volumes, n_bins)`` array). Follows the package's
  bin-edge convention and is the form consumed elsewhere in the package. Supports both forward
  (infiltration to extraction) and reverse (extraction to infiltration) directions.

- :func:`residence_time_series` - Compute residence times at specific time instants, per pore
  volume. Sampling at arbitrary instants departs from the bin-edge convention, so it is kept
  separate from :func:`residence_time_full`. Same directional options.

- :func:`residence_time` - Compute the mean residence time over output bins for a discrete
  aquifer pore-volume distribution (an array of equally-weighted pore volumes). Collapses the
  pore-volume axis to a single per-bin series. The ``spinup`` policy (default ``"constant"``)
  warm-starts the spin-up by extrapolating the boundary flow.

- :func:`gamma_residence_time` - Compute the closed-form mean residence time over output bins for a
  (shifted) gamma aquifer pore-volume distribution, with no pore-volume discretization. The
  ``spinup`` policy (default ``"constant"``) warm-starts the spin-up; ``spinup=0.0`` instead
  renormalizes over the covered sub-mass exactly.

- :func:`fraction_explained` - Compute fraction of aquifer pore volumes with valid residence
  times. Indicates how many pore volumes have sufficient flow history to compute residence time.
  Returns values in [0, 1] where 1.0 means all volumes are fully informed. Useful for assessing
  spin-up periods and data coverage. NaN residence times indicate insufficient flow history.

Spin-up period
--------------
The spin-up **region** is determined entirely by the supplied flow record (``flow_tedges``, which
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
(see :mod:`gwtransport.advection`); the default is ``"constant"`` everywhere:

* :func:`residence_time_full` and :func:`residence_time` take ``spinup={'constant', None}``.
  ``"constant"`` (default) **warm-starts** by extrapolating the boundary flow (flow held constant at
  its first/last value), so no in-record output is ``NaN``; ``None`` is strict, marking a pore
  volume ``NaN`` for any output bin its parcel leaves the record within (and :func:`residence_time`
  then averages over the remaining informed streamtubes).
* :func:`gamma_residence_time` takes ``spinup={'constant'} | float in [0, 1)``. ``"constant"``
  (default) warm-starts identically; a ``float`` instead **renormalizes** over the covered
  sub-mass, with ``0.0`` giving the exact covered-sub-mass conditional mean.
* :func:`residence_time_series` (point samples) has no ``spinup`` policy: it always returns ``NaN``
  in the spin-up. It is the primitive behind :func:`fraction_explained`.

Output bins lying wholly outside ``flow_tedges`` are ``NaN`` under every policy.

:func:`fraction_explained` reports, per output instant, the fraction of the pore-volume
distribution that is out of spin-up (``1.0`` = fully informed, ``0.0`` = entirely in spin-up) and
is the way to locate the spin-up region when the means warm-start over it.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import gamma as gamma_dist

from gwtransport._time import tedges_to_days
from gwtransport.gamma import parse_parameters
from gwtransport.utils import cumulative_flow_volume, linear_average, linear_interpolate


def residence_time_series(
    *,
    flow: npt.ArrayLike,
    flow_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volumes: npt.ArrayLike,
    index: pd.DatetimeIndex | np.ndarray | None = None,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Compute the residence time of a retarded compound at specific time instants, per pore volume.

    This evaluates the residence time at the individual instants in ``index`` (point samples),
    rather than as a bin average over an output grid. Sampling at arbitrary instants departs
    from the package's bin-edge convention, so this is kept separate from
    :func:`residence_time_full`, which returns the flow- or time-weighted bin average over
    ``tedges_out`` and is the form consumed elsewhere in the package.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m3/day]. The length of `flow` should match the length of `flow_tedges` minus one.
    flow_tedges : pandas.DatetimeIndex
        Time edges for the flow data. Used to compute the cumulative flow.
        Has a length of one more than `flow`.
    aquifer_pore_volumes : float or array-like of float
        Pore volume(s) of the aquifer [m3]. Can be a single value or an array
        of pore volumes representing different flow paths.
    index : pandas.DatetimeIndex, optional
        Instants at which to evaluate the residence time. If left to None, flow-bin centres
        are used. Default is None.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation:
        * 'extraction_to_infiltration': Extraction to infiltration modeling - how many days ago was the extracted water infiltrated
        * 'infiltration_to_extraction': Infiltration to extraction modeling - how many days until the infiltrated water is extracted
        Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Residence time of the retarded compound in the aquifer [days], shape
        ``(n_pore_volumes, n_index)``.

    Raises
    ------
    ValueError
        If ``flow_tedges`` does not have exactly one more element than ``flow``.
        If ``direction`` is not ``'extraction_to_infiltration'`` or
        ``'infiltration_to_extraction'``.

    See Also
    --------
    residence_time_full : Flow- or time-weighted mean residence time over output bins (per pore volume)
    gwtransport.advection.gamma_infiltration_to_extraction : Use residence times for transport
    gwtransport.logremoval.residence_time_to_log_removal : Convert residence time to log removal
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction
    :ref:`concept-retardation-factor` : Slower movement due to sorption

    Notes
    -----
    Instants whose retarded look-back/forward parcel falls outside the supplied flow record -- the
    spin-up period -- are returned as ``NaN`` rather than extrapolated. The spin-up is set by the
    flow record and the retarded pore volume, not by any argument: for ``extraction_to_infiltration``
    it occupies the start of the record, for ``infiltration_to_extraction`` the end. See the module
    docstring (``Spin-up period``) for the full rule.
    """
    aquifer_pore_volumes = np.atleast_1d(aquifer_pore_volumes)
    flow_tedges = pd.DatetimeIndex(flow_tedges)
    flow = np.asarray(flow)

    if len(flow_tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    # Negative or non-finite flow makes V(t) non-monotone or undefined; refuse to answer
    # rather than fail noisily downstream where the cumulative volume must be strictly ascending.
    if np.any(flow < 0) or np.any(np.isnan(flow)):
        n_output = len(flow_tedges) - 1 if index is None else len(index)
        return np.full((len(aquifer_pore_volumes), n_output), np.nan)

    if direction not in {"extraction_to_infiltration", "infiltration_to_extraction"}:
        msg = "direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
        raise ValueError(msg)

    flow_tedges_days = tedges_to_days(flow_tedges)
    # Plateaus in flow_cum from Q = 0 bins make V → t inversion multi-valued; bump duplicates
    # by the smallest representable amount so downstream np.interp resolves consistently.
    flow_cum = cumulative_flow_volume(flow, np.diff(flow_tedges_days), strictly_monotone=True)

    if index is None:
        # Bin-center evaluation; for piecewise-linear V the midpoint of cumulative values
        # equals V at the midpoint time.
        index_days = (flow_tedges_days[:-1] + flow_tedges_days[1:]) / 2
        flow_cum_at_index = (flow_cum[:-1] + flow_cum[1:]) / 2
    else:
        index_days = tedges_to_days(pd.DatetimeIndex(index), ref=flow_tedges[0])
        flow_cum_at_index = linear_interpolate(
            x_ref=flow_tedges_days, y_ref=flow_cum, x_query=index_days, left=np.nan, right=np.nan
        )

    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0
    a = flow_cum_at_index[None, :] + sign * retardation_factor * aquifer_pore_volumes[:, None]
    days = linear_interpolate(x_ref=flow_cum, y_ref=flow_tedges_days, x_query=a, left=np.nan, right=np.nan)
    return sign * (days - index_days)


def residence_time_full(
    *,
    flow: npt.ArrayLike,
    flow_tedges: pd.DatetimeIndex | np.ndarray,
    tedges_out: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volumes: npt.ArrayLike,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
    weighting: str = "flow",
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    r"""
    Compute the mean residence time over output bins, per pore volume.

    The flow- or time-weighted mean residence time is computed over each output interval
    ``[tedges_out[i], tedges_out[i + 1])`` and returned as the full
    ``(n_pore_volumes, n_output_bins)`` array -- one row per entry in
    ``aquifer_pore_volumes``, without collapsing the pore-volume axis. This bin-average form
    follows the package's bin-edge convention; for point samples at arbitrary instants use
    :func:`residence_time_series`.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m3/day]. Length matches ``flow_tedges`` minus one.
    flow_tedges : array-like
        Time edges for the flow data, as datetime64 objects, defining the flow intervals.
    tedges_out : array-like
        Output time edges as datetime64 objects; ``n + 1`` edges define ``n`` output bins.
    aquifer_pore_volumes : float or array-like
        Pore volume(s) of the aquifer [m3]. A single value or an array of pore volumes
        representing different flow paths.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation:
        * 'extraction_to_infiltration': Extraction to infiltration modeling - how many days ago was the extracted water infiltrated
        * 'infiltration_to_extraction': Infiltration to extraction modeling - how many days until the infiltrated water is extracted
        Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. A value greater
        than 1.0 indicates the compound moves slower than water. Default is 1.0.
    weighting : {'flow', 'time'}, optional
        How the per-instant residence time is averaged over each output bin:

        * ``'flow'`` (default): flow-weighted average -- uniform in cumulative volume,
          matching the bin-edge convention of the package, and what the diffusion modules
          consume to compute a per-bin retarded velocity.
        * ``'time'``: time-weighted average -- uniform in clock time. Coincides with
          ``'flow'`` when flow is constant within an output bin.
    spinup : {'constant'} or None, optional
        How to treat the spin-up zone, where a pore volume's retarded look-back/forward parcel
        leaves the flow record. Matches the package convention (see :mod:`gwtransport.advection`).

        * ``'constant'`` (default): warm-start -- extrapolate the cumulative-volume-to-time map past
          the record at the boundary flow rates (flow held constant at its first/last value), so
          the residence time stays finite. No left-edge (extraction) or right-edge (infiltration)
          spin-up ``NaN``.
        * ``None``: strict -- a pore volume whose parcel leaves the record at any point within an
          output bin is ``NaN`` for that bin (all-or-nothing per bin), with no extrapolation.

        Output bins lying wholly outside ``flow_tedges`` are ``NaN`` under either policy.

    Returns
    -------
    numpy.ndarray
        Mean residence time [days], shape ``(n_pore_volumes, n_output_bins)``. The first
        dimension corresponds to the pore volumes and the second to the ``tedges_out`` bins.

    Raises
    ------
    ValueError
        If ``flow_tedges`` does not have exactly one more element than ``flow``. If
        ``direction`` is not ``'extraction_to_infiltration'`` or
        ``'infiltration_to_extraction'``. If ``weighting`` is not ``'flow'`` or ``'time'``. If
        ``spinup`` is not ``'constant'`` or ``None``.

    See Also
    --------
    residence_time_series : Residence time at specific time instants (per pore volume)
    fraction_explained : Fraction of pore volumes out of spin-up at each instant
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction
    :ref:`concept-transport-equation` : Flow-weighted averaging convention

    Notes
    -----
    With the default ``spinup='constant'`` the spin-up zone is warm-started by extrapolating the
    boundary flow, so no in-record bin is ``NaN``; use :func:`fraction_explained` (or
    ``spinup=None``) to locate the spin-up region. See the module docstring (``Spin-up period``)
    for the full rule.

    Exact-zero flow bins produce a plateau in cumulative volume that is bumped up by one
    float64 ulp per duplicate so the cumulative volume is strictly monotone; trapezoidal
    integration over the resulting one-ulp-wide ramp recovers the underlying step
    discontinuity in the residence time exactly. The flow vs time weighting choices are

    .. math::

        \bar\tau^{\mathrm{time}}
        = \frac{1}{\Delta t}\int_{t_\mathrm{lo}}^{t_\mathrm{hi}} \tau(t)\,dt,
        \qquad
        \bar\tau^{\mathrm{flow}}
        = \frac{1}{\Delta V}\int_{V_\mathrm{lo}}^{V_\mathrm{hi}} \tau(V)\,dV,

    where :math:`V` is cumulative throughflow volume (:math:`dV = Q\,dt`); they coincide
    whenever :math:`Q` is constant within :math:`[t_\mathrm{lo}, t_\mathrm{hi}]`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.residence_time import residence_time_full
    >>> flow_dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    >>> flow_values = np.full(len(flow_dates) - 1, 100.0)  # Constant flow of 100 m³/day
    >>> mean_times = residence_time_full(
    ...     flow=flow_values,
    ...     flow_tedges=flow_dates,
    ...     tedges_out=flow_dates,
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
    if weighting not in {"flow", "time"}:
        msg = "weighting should be 'flow' or 'time'"
        raise ValueError(msg)
    if spinup not in {"constant", None}:
        msg = "spinup should be 'constant' or None"
        raise ValueError(msg)

    aquifer_pore_volumes = np.atleast_1d(aquifer_pore_volumes)
    flow_tedges = pd.DatetimeIndex(flow_tedges)
    tedges_out = pd.DatetimeIndex(tedges_out)
    flow = np.asarray(flow, dtype=float)

    if len(flow_tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)

    if np.any(flow < 0) or np.any(np.isnan(flow)):
        return np.full((len(aquifer_pore_volumes), len(tedges_out) - 1), np.nan)

    flow_tedges_days = tedges_to_days(flow_tedges)
    tedges_out_days = tedges_to_days(tedges_out, ref=flow_tedges[0])
    # Plateaus in flow_cum from Q = 0 bins make V → t inversion multi-valued; bump duplicates
    # by the smallest representable amount so downstream np.interp resolves consistently.
    flow_cum = cumulative_flow_volume(flow, np.diff(flow_tedges_days), strictly_monotone=True)

    # Sign convention: with sign = -1 for extraction_to_infiltration and +1 for
    # infiltration_to_extraction, the look-back/forward target volume is
    # ``a = flow_cum + sign * R * V_p`` and the residence time is ``tau = sign * (days(a) - t)``.
    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0

    # tau(t) is piecewise-linear in t (equivalently in cumulative volume V), but its
    # breakpoints are not only at flow_tedges: within a flow bin Q is constant, so as t advances
    # the look-back parcel sweeps through V at a constant rate and crosses each interior flow_cum
    # edge at a definite time. Sampling tau only at flow_tedges would miss those kinks and bias
    # the bin mean under regime changes; augment the integration grid by exactly those crossings.
    target_volumes_at_kinks = flow_cum[None, :] - sign * retardation_factor * aquifer_pore_volumes[:, None]
    kink_times = linear_interpolate(
        x_ref=flow_cum, y_ref=flow_tedges_days, x_query=target_volumes_at_kinks, left=np.nan, right=np.nan
    )
    augmented_grid = np.unique(np.concatenate([flow_tedges_days, kink_times[np.isfinite(kink_times)]]))

    flow_cum_at_grid = linear_interpolate(x_ref=flow_tedges_days, y_ref=flow_cum, x_query=augmented_grid)
    a_grid = flow_cum_at_grid[None, :] + sign * retardation_factor * aquifer_pore_volumes[:, None]

    # Spin-up handling. A parcel whose infiltration/extraction falls outside the flow record has an
    # out-of-range look-back/forward target a_grid. With spinup="constant" the cumulative-volume ->
    # time map is extrapolated past the record at the boundary flow rates (flow held constant at its
    # first/last value), so the residence time stays finite (the package default warm-start). One
    # anchor each end, padded by the largest reach R * max(V_p), is enough that a_grid never exceeds
    # the extended map, so the interpolation is an exact linear extrapolation. With spinup=None the
    # map is not extended and the out-of-record target yields NaN (strict, no extrapolation).
    if spinup == "constant":
        pad = retardation_factor * float(aquifer_pore_volumes.max())
        inv_q_first = (flow_tedges_days[1] - flow_tedges_days[0]) / (flow_cum[1] - flow_cum[0])
        inv_q_last = (flow_tedges_days[-1] - flow_tedges_days[-2]) / (flow_cum[-1] - flow_cum[-2])
        flow_cum_map = np.concatenate([[flow_cum[0] - pad], flow_cum, [flow_cum[-1] + pad]])
        days_map = np.concatenate([
            [flow_tedges_days[0] - pad * inv_q_first],
            flow_tedges_days,
            [flow_tedges_days[-1] + pad * inv_q_last],
        ])
        days_grid = linear_interpolate(x_ref=flow_cum_map, y_ref=days_map, x_query=a_grid)
    else:
        days_grid = linear_interpolate(
            x_ref=flow_cum, y_ref=flow_tedges_days, x_query=a_grid, left=np.nan, right=np.nan
        )
    data_grid = sign * (days_grid - augmented_grid[None, :])

    if weighting == "time":
        return linear_average(x_data=augmented_grid, y_data=data_grid, x_edges=tedges_out_days)

    flow_cum_at_tedges_out = linear_interpolate(x_ref=flow_tedges_days, y_ref=flow_cum, x_query=tedges_out_days)
    result = linear_average(x_data=flow_cum_at_grid, y_data=data_grid, x_edges=flow_cum_at_tedges_out)
    bins_within = (tedges_out_days[:-1] >= flow_tedges_days[0]) & (tedges_out_days[1:] <= flow_tedges_days[-1])
    if not np.all(bins_within):
        result = np.where(bins_within[None, :], result, np.nan)
    return result


def residence_time(
    *,
    flow: npt.ArrayLike,
    flow_tedges: pd.DatetimeIndex | np.ndarray,
    tedges_out: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volumes: npt.ArrayLike,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    r"""
    Compute the mean residence time over output bins for a discrete APVD.

    The mean is taken over a **discrete** set of equally-weighted aquifer pore volumes -- one
    streamtube per entry in ``aquifer_pore_volumes``. Each streamtube's flow-weighted bin average
    is computed with :func:`residence_time_full` and the pore-volume axis is then collapsed to a
    single per-output-bin series by averaging over the streamtubes that are valid in each bin. For
    a continuous (shifted) gamma pore-volume distribution evaluated in closed form, use
    :func:`gamma_residence_time`.

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
        Flow rate of water in the aquifer [m3/day]. Length matches ``flow_tedges`` minus one.
    flow_tedges : array-like
        Time edges for the flow data, as datetime64 objects, defining the flow intervals.
    tedges_out : array-like
        Output time edges as datetime64 objects; ``n + 1`` edges define ``n`` output bins.
    aquifer_pore_volumes : array-like
        Discrete pore volumes [m3], one per (equally-weighted) streamtube. A single value
        collapses to the per-streamtube mean of :func:`residence_time_full`.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation:
        * 'extraction_to_infiltration': how many days ago was the extracted water infiltrated
        * 'infiltration_to_extraction': how many days until the infiltrated water is extracted
        Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.
    spinup : {'constant'} or None, optional
        Spin-up policy, forwarded to :func:`residence_time_full`. ``'constant'`` (default)
        warm-starts by extrapolating the boundary flow so no in-record bin is ``NaN``; ``None``
        leaves spin-up streamtubes ``NaN`` and the mean renormalizes over those that have broken
        through. Use :func:`fraction_explained` to locate the spin-up region.

    Returns
    -------
    numpy.ndarray
        Mean residence time [days], shape ``(n_output_bins,)``. Output bins with no valid
        streamtube (outside the flow record, or -- with ``spinup=None`` -- fully in the spin-up
        zone) are NaN.

    See Also
    --------
    gamma_residence_time : Exact closed-form mean for a continuous (shifted) gamma APVD
    residence_time_full : Per-pore-volume mean residence time over output bins
    fraction_explained : Fraction of pore volumes out of spin-up at each instant
    gwtransport.gamma.bins : Discretize a gamma APVD into pore-volume bins
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction

    Notes
    -----
    With ``spinup=None`` the spin-up is **all-or-nothing per streamtube**: a streamtube whose
    look-back/forward parcel leaves the flow record part-way through an output bin has a ``NaN`` bin
    average (inherited from :func:`residence_time_full`) and is dropped from that bin's mean
    entirely, rather than contributing its partially-covered share; the bin is ``NaN`` only once
    every streamtube is in spin-up. In that mode the discrete mean differs from
    :func:`gamma_residence_time`, which renormalizes over the covered sub-mass exactly. See the
    module docstring (``Spin-up period``) for the full rule.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.residence_time import residence_time
    >>> flow_dates = pd.date_range(start="2023-01-01", end="2023-02-10", freq="D")
    >>> flow_values = np.full(len(flow_dates) - 1, 100.0)  # 100 m³/day
    >>> tau_bar = residence_time(
    ...     flow=flow_values,
    ...     flow_tedges=flow_dates,
    ...     tedges_out=flow_dates,
    ...     aquifer_pore_volumes=[400.0, 600.0],  # two equally-weighted streamtubes
    ... )
    >>> # Deep in the record: mean pore volume 500 / 100 m³/day = 5 days
    >>> float(np.round(tau_bar[-1], 6))
    5.0
    """
    rt = residence_time_full(
        flow=flow,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=aquifer_pore_volumes,
        direction=direction,
        retardation_factor=retardation_factor,
        weighting="flow",
        spinup=spinup,
    )

    # Mean over the streamtubes that are valid (non-NaN) in each output bin; bins with no valid
    # streamtube reduce to 0/0 and are NaN. With spinup='constant' every in-record streamtube is
    # finite, so this is the plain mean; with spinup=None it renormalizes over the broken-through set.
    valid_count = np.isfinite(rt).sum(axis=0)
    with np.errstate(invalid="ignore"):
        return np.nansum(rt, axis=0) / valid_count


def gamma_residence_time(
    *,
    flow: npt.ArrayLike,
    flow_tedges: pd.DatetimeIndex | np.ndarray,
    tedges_out: pd.DatetimeIndex | np.ndarray,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
    spinup: str | float = "constant",
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
        Flow rate of water in the aquifer [m3/day]. Length matches ``flow_tedges`` minus one.
    flow_tedges : array-like
        Time edges for the flow data, as datetime64 objects, defining the flow intervals.
    tedges_out : array-like
        Output time edges as datetime64 objects; ``n + 1`` edges define ``n`` output bins.
    mean : float, optional
        Mean of the gamma APVD [m3]. Must be strictly greater than ``loc``. Provide either
        ``(mean, std)`` or ``(alpha, beta)``.
    std : float, optional
        Standard deviation of the gamma APVD [m3]. Must be positive.
    loc : float, optional
        Location (lower bound of support) of the gamma APVD [m3]; a guaranteed minimum pore
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
    spinup : {'constant'} or float in [0, 1), optional
        How to treat the spin-up zone, where part of the gamma APVD lacks flow history. Matches
        the package convention (see :mod:`gwtransport.advection`); ``None`` is not offered because
        a continuous distribution always has some covered sub-mass.

        * ``'constant'`` (default): warm-start -- extrapolate the cumulative-volume-to-time map past
          the record at the boundary flow rates (flow held constant at its first/last value) and
          integrate the full distribution, so no in-record bin is ``NaN``.
        * ``float`` in ``[0, 1)``: renormalize the mean over the covered sub-mass, emitting a bin
          only where the covered fraction of the distribution is at least ``spinup``. ``0.0`` gives
          the exact covered-sub-mass conditional mean (emit whenever any sub-mass is covered).

        Output bins lying wholly outside ``flow_tedges`` are ``NaN`` under either policy.

    Returns
    -------
    numpy.ndarray
        APVD-mean residence time [days], shape ``(n_output_bins,)``. Output bins outside the flow
        record are NaN; with a ``float`` ``spinup`` so are bins whose covered fraction is below the
        threshold.

    Raises
    ------
    ValueError
        If ``flow_tedges`` does not have exactly one more element than ``flow``. If ``direction``
        is not ``'extraction_to_infiltration'`` or ``'infiltration_to_extraction'``. If ``spinup``
        is not ``'constant'`` or a float in ``[0, 1)``. Gamma parameter validation is delegated to
        :func:`gwtransport.gamma.parse_parameters`.

    See Also
    --------
    residence_time : Equally-weighted mean for a discrete set of pore volumes
    residence_time_full : Per-pore-volume mean residence time over output bins
    fraction_explained : Fraction of pore volumes out of spin-up at each instant
    gwtransport.gamma.bins : Discretize a gamma APVD into pore-volume bins
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model

    Notes
    -----
    With the default ``spinup='constant'`` the spin-up is warm-started exactly as in
    :func:`residence_time` (constant-boundary-flow extrapolation), so the two agree everywhere. With
    ``spinup=0.0`` the spin-up is instead handled by exact covered-sub-mass renormalization: each
    output bin integrates over only the pore-volume sub-range with sufficient flow history. See the
    module docstring (``Spin-up period``) for the full rule.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.residence_time import gamma_residence_time
    >>> flow_dates = pd.date_range(start="2023-01-01", end="2023-02-10", freq="D")
    >>> flow_values = np.full(len(flow_dates) - 1, 100.0)  # 100 m³/day
    >>> tau_bar = gamma_residence_time(
    ...     flow=flow_values,
    ...     flow_tedges=flow_dates,
    ...     tedges_out=flow_dates,
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
    extrapolate = spinup == "constant"
    if extrapolate:
        spinup_threshold = 0.0
    elif isinstance(spinup, int | float) and not isinstance(spinup, bool) and 0.0 <= spinup < 1.0:
        spinup_threshold = float(spinup)
    else:
        msg = "spinup should be 'constant' or a float in [0, 1)"
        raise ValueError(msg)

    alpha, beta, loc = parse_parameters(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta)

    flow = np.asarray(flow, dtype=float)
    flow_tedges = pd.DatetimeIndex(flow_tedges)
    tedges_out = pd.DatetimeIndex(tedges_out)
    n_out = len(tedges_out) - 1

    if len(flow_tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if np.any(flow < 0) or np.any(np.isnan(flow)):
        return np.full(n_out, np.nan)

    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0
    r = retardation_factor

    flow_tedges_days = tedges_to_days(flow_tedges)
    flow_cum = cumulative_flow_volume(flow, np.diff(flow_tedges_days), strictly_monotone=True)
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
    if extrapolate:
        pad = r * support_hi
        inv_q_first = (flow_tedges_days[1] - flow_tedges_days[0]) / (flow_cum[1] - flow_cum[0])
        inv_q_last = (flow_tedges_days[-1] - flow_tedges_days[-2]) / (flow_cum[-1] - flow_cum[-2])
        phi_v = np.concatenate([[flow_cum[0] - pad], flow_cum, [flow_cum[-1] + pad]])
        phi_t = np.concatenate([
            [flow_tedges_days[0] - pad * inv_q_first],
            flow_tedges_days,
            [flow_tedges_days[-1] + pad * inv_q_last],
        ])
    else:
        phi_v = flow_cum
        phi_t = flow_tedges_days
    phi_dv = phi_v[1:] - phi_v[:-1]
    phi_rate = phi_dv / (phi_t[1:] - phi_t[:-1])
    phi_knot = np.concatenate([[0.0], np.cumsum(phi_t[:-1] * phi_dv + phi_dv**2 / (2 * phi_rate))])

    def phi(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        v = np.clip(v, phi_v[0], phi_v[-1])
        j = np.clip(np.searchsorted(phi_v, v, side="right") - 1, 0, len(phi_rate) - 1)
        dv = v - phi_v[j]
        return phi_knot[j] + phi_t[j] * dv + dv * dv / (2 * phi_rate[j])

    tedges_out_days = tedges_to_days(tedges_out, ref=flow_tedges[0])
    vol_out = linear_interpolate(
        x_ref=flow_tedges_days, y_ref=flow_cum, x_query=tedges_out_days, left=np.nan, right=np.nan
    )
    good_all = np.isfinite(vol_out[:-1]) & np.isfinite(vol_out[1:]) & (vol_out[1:] > vol_out[:-1])
    if not np.any(good_all):
        return np.full(n_out, np.nan)
    v_lo_all = np.where(good_all, vol_out[:-1], 0.0)
    v_hi_all = np.where(good_all, vol_out[1:], 1.0)

    # Fixed band of flow edges each bin's look-back/forward sweep can cross over the supported
    # pore volumes. band_width is the global maximum so every tile shares one column layout; a
    # spurious column clips to the support and merely splits a quadratic piece without changing
    # its integral, so the band is an exact superset.
    a_min = v_lo_all - r * support_hi if sign < 0 else v_lo_all + r * support_lo
    a_max = v_hi_all - r * support_lo if sign < 0 else v_hi_all + r * support_hi
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
            phi_const = phi(v_hi)
            phi_stack = phi(np.stack([v_start, a_hi, a_lo]))
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
            phi_const = phi(v_lo)
            phi_stack = phi(np.stack([a_hi, a_lo, v_stop]))
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

        # G is quadratic per piece, L linear; reconstruct each from its three nodes (Lagrange-
        # exact) centred at mid and contract with the moments: int (a(x-mid)^2 + b(x-mid) + c) f
        # = a (m2 - 2 mid m1 + mid^2 m0) + b (m1 - mid m0) + c m0. Zero-width pieces (duplicate
        # breakpoints) divide by half == 0; their masked result is discarded by np.where.
        safe = half > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            g_a = np.where(safe, (g_lo - 2 * g_mid + g_hi) / (2 * half**2), 0.0)
            g_b = np.where(safe, (g_hi - g_lo) / (2 * half), 0.0)
            l_b = np.where(safe, (l_hi - l_lo) / (2 * half), 0.0)
        int_g = g_a * (m2 - 2 * mid * m1 + mid**2 * m0) + g_b * (m1 - mid * m0) + g_mid * m0
        int_l = l_b * (m1 - mid * m0) + l_mid * m0

        num = int_g.sum(axis=1)
        den = int_l.sum(axis=1)
        covered = good & (den > 0)
        if spinup_threshold > 0.0:
            # float spinup: emit only where the covered fraction of the APVD reaches the threshold.
            # den is the covered sub-mass; (v_hi - v_lo) * m0.sum is the fully-covered sub-mass.
            covered &= den >= spinup_threshold * (v_hi - v_lo) * m0.sum(axis=1)
        tile_result = np.full(nt, np.nan)
        tile_result[covered] = num[covered] / den[covered]
        result[t0:t1] = tile_result
    return result


def fraction_explained(
    *,
    rt: npt.NDArray[np.floating] | None = None,
    flow: npt.ArrayLike | None = None,
    flow_tedges: pd.DatetimeIndex | np.ndarray | None = None,
    aquifer_pore_volumes: npt.ArrayLike | None = None,
    index: pd.DatetimeIndex | np.ndarray | None = None,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Compute the fraction of the aquifer that is informed with respect to the retarded flow.

    For each output instant this is the fraction of the supplied pore volumes whose residence time
    is finite, i.e. the complement of the spin-up coverage: ``1.0`` means every pore volume is out
    of spin-up, ``0.0`` means all are still in it. The spin-up itself is set by the flow record and
    the retarded pore volumes (see the module docstring, ``Spin-up period``), not by any argument.

    Parameters
    ----------
    rt : numpy.ndarray, optional
        Pre-computed residence time array [days]. If not provided, it will be computed.
    flow : array-like, optional
        Flow rate of water in the aquifer [m3/day]. The length of `flow` should match the length of `flow_tedges` minus one.
    flow_tedges : pandas.DatetimeIndex, optional
        Time edges for the flow data. Used to compute the cumulative flow.
        Has a length of one more than `flow`. Inbetween neighboring time edges, the flow is assumed constant.
    aquifer_pore_volumes : float or array-like of float, optional
        Pore volume(s) of the aquifer [m3]. Can be a single value or an array
        of pore volumes representing different flow paths.
    index : pandas.DatetimeIndex, optional
        Index at which to compute the fraction. If left to None, the index of `flow` is used.
        Default is None.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation:
        * 'extraction_to_infiltration': Extraction to infiltration modeling - how many days ago was the extracted water infiltrated
        * 'infiltration_to_extraction': Infiltration to extraction modeling - how many days until the infiltrated water is extracted
        Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless]. Default is 1.0.

    Returns
    -------
    numpy.ndarray
        Fraction of the aquifer that is informed with respect to the retarded flow.

    Raises
    ------
    ValueError
        If ``rt`` is not provided and any of ``flow``, ``flow_tedges``, or
        ``aquifer_pore_volumes`` are missing. If ``rt`` is provided but is not 2D.
    """
    if rt is None:
        if flow is None:
            msg = "Either rt or flow must be provided"
            raise ValueError(msg)
        if flow_tedges is None:
            msg = "Either rt or flow_tedges must be provided"
            raise ValueError(msg)
        if aquifer_pore_volumes is None:
            msg = "Either rt or aquifer_pore_volumes must be provided"
            raise ValueError(msg)

        rt = residence_time_series(
            flow=flow,
            flow_tedges=flow_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            index=index,
            direction=direction,
            retardation_factor=retardation_factor,
        )

    expected_ndim = 2
    if rt.ndim != expected_ndim:
        msg = f"rt must be 2D with shape (n_pore_volumes, n_times), got {rt.ndim}D"
        raise ValueError(msg)

    n_aquifer_pore_volume = rt.shape[0]
    return (n_aquifer_pore_volume - np.isnan(rt).sum(axis=0)) / n_aquifer_pore_volume


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

    The Freundlich isotherm relates sorbed concentration S to aqueous concentration C:
        S = rho_f * C^n

    The retardation factor is computed as:
        R = 1 + (rho_b/θ) * dS/dC = 1 + (rho_b/θ) * rho_f * n * C^(n-1)

    Parameters
    ----------
    concentration : array-like
        Concentration of compound in water [mass/volume].
        Length should match flow (i.e., len(flow_tedges) - 1).
    freundlich_k : float
        Freundlich coefficient [(m³/kg)^n] (under S = k_f * C^n with S dimensionless
        and C in [kg/m³]).
    freundlich_n : float
        Freundlich sorption exponent [dimensionless].
    bulk_density : float
        Bulk density of aquifer material [mass/volume].
    porosity : float
        Porosity of aquifer [dimensionless, 0-1].

    Returns
    -------
    numpy.ndarray
        Retardation factors for each flow interval.
        Length equals len(concentration) for use as retardation_factor in residence_time_series.

    Raises
    ------
    ValueError
        If ``porosity`` is not in ``(0, 1)``, if ``bulk_density`` is not positive, if
        ``freundlich_k`` is negative, or if any ``concentration`` is non-positive while
        ``freundlich_n < 1`` (the retardation factor diverges as ``C -> 0``).

    See Also
    --------
    residence_time_series : Compute residence times with retardation
    gwtransport.advection.infiltration_to_extraction_nonlinear_sorption : Transport with nonlinear sorption
    :ref:`concept-nonlinear-sorption` : Freundlich isotherm and concentration-dependent retardation

    Examples
    --------
    >>> concentration = np.array([0.1, 0.2, 0.3])  # same length as flow
    >>> R = freundlich_retardation(
    ...     concentration=concentration,
    ...     freundlich_k=0.5,
    ...     freundlich_n=0.9,
    ...     bulk_density=1600,  # kg/m3
    ...     porosity=0.35,
    ... )
    >>> # Use R in residence_time_series as retardation_factor
    """
    concentration = np.asarray(concentration)

    # Validate physical parameters
    if not 0 < porosity < 1:
        msg = f"Porosity must be in (0, 1), got {porosity}"
        raise ValueError(msg)
    if bulk_density <= 0:
        msg = f"Bulk density must be positive, got {bulk_density}"
        raise ValueError(msg)
    if freundlich_k < 0:
        msg = f"Freundlich K must be non-negative, got {freundlich_k}"
        raise ValueError(msg)

    # For n < 1 the Freundlich retardation factor 1 + (rho_b/theta) * k_f * n * C^(n-1)
    # diverges as C -> 0. Silently clamping concentration would produce a very large but
    # finite value that depends on an arbitrary regularization constant; instead, refuse
    # the call so the user can decide how to handle non-positive concentrations.
    if freundlich_n < 1.0 and np.any(concentration <= 0):
        msg = "concentration must be strictly positive when freundlich_n < 1 (retardation diverges as C -> 0)"
        raise ValueError(msg)

    return 1.0 + (bulk_density / porosity) * freundlich_k * freundlich_n * concentration ** (freundlich_n - 1)
