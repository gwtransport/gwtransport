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

- :func:`residence_time` - Compute residence times at specific time indices. Supports both
  forward (infiltration to extraction) and reverse (extraction to infiltration) directions.
  Handles single or multiple pore volumes (2D output for multiple volumes). Returns residence
  times in days using cumulative flow integration for accurate time-varying flow handling.

- :func:`residence_time_mean` - Compute mean residence times over time intervals. Defaults to a
  flow-weighted average (uniform in cumulative volume), matching the bin-edge convention used
  elsewhere in the package; pass ``weighting='time'`` for the legacy uniform-in-time average.
  Supports same directional options as :func:`residence_time`. Particularly useful for time-binned
  analysis.

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

from gwtransport.utils import _make_strictly_monotone, linear_average, linear_interpolate


def residence_time(
    *,
    flow: npt.ArrayLike,
    flow_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volumes: npt.ArrayLike,
    index: pd.DatetimeIndex | np.ndarray | None = None,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Compute the residence time of retarded compound in the water in the aquifer.

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
        Index at which to compute the residence time. If left to None, flow_tedges is used.
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
        Residence time of the retarded compound in the aquifer [days].

    Raises
    ------
    ValueError
        If ``flow_tedges`` does not have exactly one more element than ``flow``.
        If ``direction`` is not ``'extraction_to_infiltration'`` or
        ``'infiltration_to_extraction'``.

    See Also
    --------
    residence_time_mean : Compute mean residence time over time intervals
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

    # Negative or non-finite flow makes V(t) non-monotone or undefined; refuse to answer.
    # The NaN gate mirrors residence_time_mean, which otherwise fails noisily inside
    # linear_average where x_data must be strictly ascending.
    if np.any(flow < 0) or np.any(np.isnan(flow)):
        n_output = len(flow_tedges) - 1 if index is None else len(index)
        n_pore_volumes = len(aquifer_pore_volumes)
        return np.full((n_pore_volumes, n_output), np.nan)

    if direction not in {"extraction_to_infiltration", "infiltration_to_extraction"}:
        msg = "direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
        raise ValueError(msg)

    flow_tedges_days = np.asarray((flow_tedges - flow_tedges[0]) / np.timedelta64(1, "D"))
    flow_cum = np.concatenate(([0.0], np.cumsum(flow * np.diff(flow_tedges_days))))
    # Plateaus in flow_cum from Q = 0 bins make V → t inversion multi-valued; bump duplicates
    # by the smallest representable amount so downstream np.interp resolves consistently.
    flow_cum = _make_strictly_monotone(flow_cum)

    if index is None:
        # Bin-center evaluation; for piecewise-linear V the midpoint of cumulative values
        # equals V at the midpoint time.
        index_days = (flow_tedges_days[:-1] + flow_tedges_days[1:]) / 2
        flow_cum_at_index = (flow_cum[:-1] + flow_cum[1:]) / 2
    else:
        index_days = np.asarray((index - flow_tedges[0]) / np.timedelta64(1, "D"))
        flow_cum_at_index = linear_interpolate(
            x_ref=flow_tedges_days, y_ref=flow_cum, x_query=index_days, left=np.nan, right=np.nan
        )

    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0
    a = flow_cum_at_index[None, :] + sign * retardation_factor * aquifer_pore_volumes[:, None]
    days = linear_interpolate(x_ref=flow_cum, y_ref=flow_tedges_days, x_query=a, left=np.nan, right=np.nan)
    return sign * (days - index_days)


def residence_time_mean(
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
    Compute the mean residence time of a retarded compound in the aquifer between specified time edges.

    This function calculates the average residence time of a retarded compound in the aquifer
    between specified time intervals. It can compute both extraction to infiltration modeling (extraction direction:
    when was extracted water infiltrated) and infiltration to extraction modeling (infiltration direction: when will
    infiltrated water be extracted).

    The function handles time series data by computing the cumulative flow and using linear
    interpolation and averaging to determine mean residence times between the specified time edges.

    Parameters
    ----------
    flow : array-like
        Flow rate of water in the aquifer [m3/day]. Should be an array of flow values
        corresponding to the intervals defined by flow_tedges.
    flow_tedges : array-like
        Time edges for the flow data, as datetime64 objects. These define the time
        intervals for which the flow values are provided.
    tedges_out : array-like
        Output time edges as datetime64 objects. These define the intervals for which
        the mean residence times will be calculated.
    aquifer_pore_volumes : float or array-like
        Pore volume(s) of the aquifer [m3]. Can be a single value or an array of values
        for multiple pore volume scenarios.
    direction : {'extraction_to_infiltration', 'infiltration_to_extraction'}, optional
        Direction of the flow calculation:
        * 'extraction_to_infiltration': Extraction to infiltration modeling - how many days ago was the extracted water infiltrated
        * 'infiltration_to_extraction': Infiltration to extraction modeling - how many days until the infiltrated water is extracted
        Default is 'extraction_to_infiltration'.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer [dimensionless].
        A value greater than 1.0 indicates that the compound moves slower than water.
        Default is 1.0 (no retardation).
    weighting : {'flow', 'time'}, optional
        How the per-instant residence time is averaged over each output bin:

        * ``'flow'`` (default): flow-weighted average -- uniform in cumulative
          volume. This matches the bin-edge convention of the package, where
          per-bin output quantities are flow-weighted averages of the underlying
          continuous-time signal, and is what the diffusion modules consume to
          compute a per-bin retarded velocity.
        * ``'time'``: time-weighted average -- uniform in clock time. Coincides
          with ``'flow'`` when flow is constant within an output bin and differs
          by :math:`\mathcal{O}(\delta Q \cdot \delta\tau)` under variable flow.

    Returns
    -------
    numpy.ndarray
        Mean residence time of the retarded compound in the aquifer [days] for each interval
        defined by tedges_out. The first dimension corresponds to the different pore volumes
        and the second to the residence times between tedges_out.

    Raises
    ------
    ValueError
        If ``direction`` is not ``'extraction_to_infiltration'`` or
        ``'infiltration_to_extraction'``. If ``weighting`` is not ``'flow'`` or
        ``'time'``.

    See Also
    --------
    residence_time : Compute residence time at specific time indices
    :ref:`concept-residence-time` : Time in aquifer between infiltration and extraction
    :ref:`concept-transport-equation` : Flow-weighted averaging convention

    Notes
    -----
    - The function converts datetime objects to days since the start of the time series.
    - For extraction_to_infiltration direction, the function computes how many days ago water was infiltrated.
    - For infiltration_to_extraction direction, the function computes how many days until water will be extracted.
    - The function uses linear interpolation for computing residence times at specific points
      and linear averaging for computing mean values over intervals.
    - Exact-zero flow bins produce a plateau in cumulative volume ``V(t)`` and a step
      discontinuity in residence time at the kink time. The implementation silently bumps
      each duplicate ``flow_cum`` value up by one float64 ulp per duplicate so the cumulative
      volume is strictly monotone, the smallest representable perturbation. Trapezoidal
      integration over the resulting one-ulp-wide ramp recovers the underlying step exactly
      to machine precision (left/right average over the ramp width equals the step's
      contribution at zero width).

    The two weighting choices are

    .. math::

        \bar\tau^{\mathrm{time}}
        = \frac{1}{\Delta t}\int_{t_\mathrm{lo}}^{t_\mathrm{hi}} \tau(t)\,dt,
        \qquad
        \bar\tau^{\mathrm{flow}}
        = \frac{\int_{t_\mathrm{lo}}^{t_\mathrm{hi}} \tau(t)\,Q(t)\,dt}
               {\int_{t_\mathrm{lo}}^{t_\mathrm{hi}} Q(t)\,dt}
        = \frac{1}{\Delta V}\int_{V_\mathrm{lo}}^{V_\mathrm{hi}} \tau(V)\,dV,

    where :math:`V` is cumulative throughflow volume (:math:`dV = Q\,dt`). They
    coincide whenever :math:`Q` is constant within
    :math:`[t_\mathrm{lo}, t_\mathrm{hi}]`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.residence_time import residence_time_mean
    >>> # Create sample flow data
    >>> flow_dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    >>> flow_values = np.full(len(flow_dates) - 1, 100.0)  # Constant flow of 100 m³/day
    >>> pore_volume = 200.0  # Aquifer pore volume in m³
    >>> mean_times = residence_time_mean(
    ...     flow=flow_values,
    ...     flow_tedges=flow_dates,
    ...     tedges_out=flow_dates,
    ...     aquifer_pore_volumes=pore_volume,
    ...     direction="extraction_to_infiltration",
    ... )
    >>> # With constant flow of 100 m³/day and pore volume of 200 m³,
    >>> # mean residence time should be approximately 2 days
    >>> print(mean_times)  # doctest: +NORMALIZE_WHITESPACE
    [[nan nan  2.  2.  2.  2.  2.  2.  2.]]
    """
    if weighting not in {"flow", "time"}:
        msg = "weighting should be 'flow' or 'time'"
        raise ValueError(msg)
    if direction not in {"extraction_to_infiltration", "infiltration_to_extraction"}:
        msg = "direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
        raise ValueError(msg)

    flow = np.asarray(flow, dtype=float)
    flow_tedges = pd.DatetimeIndex(flow_tedges)
    tedges_out = pd.DatetimeIndex(tedges_out)
    aquifer_pore_volumes = np.atleast_1d(aquifer_pore_volumes)

    if np.any(flow < 0) or np.any(np.isnan(flow)):
        n_pore_volumes = len(aquifer_pore_volumes)
        n_output_bins = len(tedges_out) - 1
        return np.full((n_pore_volumes, n_output_bins), np.nan)

    flow_tedges_days = np.asarray((flow_tedges - flow_tedges[0]) / np.timedelta64(1, "D"))
    tedges_out_days = np.asarray((tedges_out - flow_tedges[0]) / np.timedelta64(1, "D"))

    flow_cum = np.concatenate(([0.0], np.cumsum(flow * np.diff(flow_tedges_days))))

    # Q = 0 produces plateaus in flow_cum that np.unique would collapse to a single grid point
    # in the augmented trapezoidal grid, smearing tau's step discontinuity at the kink. The
    # ulp-scale bump restores strict monotonicity; trapezoidal integration over the resulting
    # steep ramp recovers the underlying step exactly.
    flow_cum = _make_strictly_monotone(flow_cum)

    # Sign convention: with sign = -1 for extraction_to_infiltration and +1 for
    # infiltration_to_extraction, the look-back/forward target volume is
    # ``a = flow_cum + sign * R * V_p`` and the residence time is
    # ``tau = sign * (days(a) - t)``.
    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0

    # tau(t) is piecewise-linear in t (and equivalently in cumulative volume V), but
    # the breakpoints are not only at flow_tedges: within a flow bin Q is constant,
    # so as t advances the look-back parcel sweeps through V at constant rate and
    # crosses each interior flow_cum edge at a definite time t*. Sampling tau only
    # at flow_tedges and linear-interpolating between samples can miss those interior
    # kinks and, under regime changes with widely different Q, can overestimate the
    # bin-mean residence time substantially.  The cure is to augment the integration
    # grid by exactly those crossing times.
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

        rt = residence_time(
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
        Length equals len(concentration) for use as retardation_factor in residence_time.

    Raises
    ------
    ValueError
        If ``porosity`` is not in ``(0, 1)``, if ``bulk_density`` is not positive, if
        ``freundlich_k`` is negative, or if any ``concentration`` is non-positive while
        ``freundlich_n < 1`` (the retardation factor diverges as ``C -> 0``).

    See Also
    --------
    residence_time : Compute residence times with retardation
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
    >>> # Use R in residence_time as retardation_factor
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
