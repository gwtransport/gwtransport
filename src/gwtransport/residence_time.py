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

- :func:`fraction_explained` - Compute fraction of aquifer pore volumes with valid residence
  times. Indicates how many pore volumes have sufficient flow history to compute residence time.
  Returns values in [0, 1] where 1.0 means all volumes are fully informed. Useful for assessing
  spin-up periods and data coverage. NaN residence times indicate insufficient flow history.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport._time import tedges_to_days
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
        ``'infiltration_to_extraction'``. If ``weighting`` is not ``'flow'`` or ``'time'``.

    See Also
    --------
    residence_time_series : Residence time at specific time instants (per pore volume)
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction
    :ref:`concept-transport-equation` : Flow-weighted averaging convention

    Notes
    -----
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
    >>> # 200 m³ / 100 m³/day = 2 days residence time
    >>> print(mean_times)  # doctest: +NORMALIZE_WHITESPACE
    [[nan nan  2.  2.  2.  2.  2.  2.  2.]]
    """
    if direction not in {"extraction_to_infiltration", "infiltration_to_extraction"}:
        msg = "direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
        raise ValueError(msg)
    if weighting not in {"flow", "time"}:
        msg = "weighting should be 'flow' or 'time'"
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
    days_grid = linear_interpolate(x_ref=flow_cum, y_ref=flow_tedges_days, x_query=a_grid, left=np.nan, right=np.nan)
    data_grid = sign * (days_grid - augmented_grid[None, :])

    if weighting == "time":
        return linear_average(x_data=augmented_grid, y_data=data_grid, x_edges=tedges_out_days)

    flow_cum_at_tedges_out = linear_interpolate(x_ref=flow_tedges_days, y_ref=flow_cum, x_query=tedges_out_days)
    result = linear_average(x_data=flow_cum_at_grid, y_data=data_grid, x_edges=flow_cum_at_tedges_out)
    bins_within = (tedges_out_days[:-1] >= flow_tedges_days[0]) & (tedges_out_days[1:] <= flow_tedges_days[-1])
    if not np.all(bins_within):
        result = np.where(bins_within[None, :], result, np.nan)
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
