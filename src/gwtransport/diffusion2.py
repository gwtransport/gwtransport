"""
Analytical solutions for 1D advection-dispersion transport.

This module implements analytical solutions for solute transport in 1D aquifer
systems, combining advection with longitudinal dispersion. The solutions are
based on the error function (erf) and its integrals.

Key function:
- infiltration_to_extraction: Main transport function combining advection and dispersion

The dispersion is characterized by the longitudinal dispersion coefficient D_L,
which the user should compute as:

    D_L = D_m + alpha_L * v

where:
- D_m is the molecular diffusion coefficient [m^2/day]
- alpha_L is the longitudinal dispersivity [m]
- v is the pore velocity [m/day] (v = q/n, where q is specific discharge and n is porosity)
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import NDArray
from scipy import special

from gwtransport.residence_time import residence_time

# Numerical tolerance for coefficient sum to determine valid output bins
EPSILON_COEFF_SUM = 1e-10


def _erf_integral_time_paired(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    diffusivity: float,
) -> NDArray[np.float64]:
    r"""Compute the integral of the error function over time at each (t[i], x[i]) point.

    This function computes the integral of erf(x/(2*sqrt(D*tau))) from 0 to t,
    where t and x are broadcastable arrays.

    The analytical solution is:

    .. math::
        \int_0^t \text{erf}\left(\frac{x}{2\sqrt{D \tau}}\right) d\tau
        = t \cdot \text{erf}\left(\frac{x}{2\sqrt{D t}}\right)
        + \frac{x \sqrt{t}}{\sqrt{\pi D}} \exp\left(-\frac{x^2}{4 D t}\right)
        - \frac{x^2}{2D} \text{erfc}\left(\frac{x}{2\sqrt{D t}}\right)

    Parameters
    ----------
    t : ndarray
        Input time values. Broadcastable with x. Must be non-negative.
    x : ndarray
        Position values. Broadcastable with t.
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.

    Returns
    -------
    ndarray
        Integral values at each (t, x) point. Shape is broadcast shape of t and x.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    diffusivity = float(diffusivity)

    # Broadcast t and x to common shape
    t, x = np.broadcast_arrays(t, x)
    out = np.zeros_like(t, dtype=float)

    if diffusivity <= 0.0:
        return out

    # Mask for valid computation: t > 0 and x != 0
    mask_valid = (t > 0.0) & (x != 0.0)

    if np.any(mask_valid):
        t_v = t[mask_valid]
        x_v = x[mask_valid]

        sqrt_t = np.sqrt(t_v)
        sqrt_d = np.sqrt(diffusivity)
        sqrt_pi = np.sqrt(np.pi)

        arg = x_v / (2 * sqrt_d * sqrt_t)
        exp_term = np.exp(-(x_v**2) / (4 * diffusivity * t_v))
        erf_term = special.erf(arg)
        erfc_term = special.erfc(arg)

        term1 = t_v * erf_term
        term2 = (x_v * sqrt_t / (sqrt_pi * sqrt_d)) * exp_term
        term3 = -(x_v**2 / (2 * diffusivity)) * erfc_term

        out[mask_valid] = term1 + term2 + term3

    # Handle infinity: as t -> inf, integral -> inf * sign(x)
    mask_t_inf = np.isinf(t)
    if np.any(mask_t_inf):
        out[mask_t_inf] = np.inf * np.sign(x[mask_t_inf])

    return out


def _erf_mean_time_paired(
    tedges: NDArray[np.float64],
    x: NDArray[np.float64],
    diffusivity: float,
) -> NDArray[np.float64]:
    """Compute the mean of the error function over time intervals with paired x values.

    This function computes the mean of erf(x/(2*sqrt(D*t))) between consecutive
    time edges, where each cell has its own x value.

    Parameters
    ----------
    tedges : ndarray, shape (n,)
        Time edges of size n.
    x : ndarray
        Position values. Broadcastable to shape (n,) for edge evaluation.
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.

    Returns
    -------
    ndarray, shape (n-1,)
        Mean of the error function for each time interval.
    """
    tedges = np.asarray(tedges, dtype=float)
    x = np.asarray(x, dtype=float)
    diffusivity = float(diffusivity)

    # Broadcast x to match tedges length
    x_edges = np.broadcast_to(x, tedges.shape)

    # Compute integral at all edge points
    erfint = _erf_integral_time_paired(tedges, x=x_edges, diffusivity=diffusivity)

    # Compute mean using shifted views
    dt = tedges[1:] - tedges[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(dt != 0.0, (erfint[1:] - erfint[:-1]) / dt, np.nan)

    # Handle dt == 0 (point evaluation): erf at that point
    mask_dt_zero = dt == 0.0
    if np.any(mask_dt_zero):
        t_at_zero = tedges[:-1][mask_dt_zero]
        x_at_zero = x_edges[:-1][mask_dt_zero]

        with np.errstate(divide="ignore", invalid="ignore"):
            arg = np.where(
                t_at_zero > 0,
                x_at_zero / (2 * np.sqrt(diffusivity * t_at_zero)),
                np.where(x_at_zero != 0, np.inf * np.sign(x_at_zero), 0.0),
            )
        out[mask_dt_zero] = special.erf(arg)

    # Handle infinite time edges
    lt, ut = tedges[:-1], tedges[1:]
    out[~np.isinf(lt) & np.isposinf(ut)] = 0.0
    out[np.isposinf(lt) & np.isposinf(ut)] = 0.0

    return out


def _erf_integral_space_time_pointwise(x, t, diffusivity):
    """
    Compute the integral of the error function in space and time at (x, t) points.

    Unlike erf_integral_space_time which uses meshgrid, this function evaluates
    F(x[i], t[i]) for each i, where x and t have the same shape. This is useful
    for batched computations where we need F at arbitrary (x, t) pairs.

    The double integral F(x,t) = ∫₀ᵗ ∫₀ˣ erf(ξ/(2√(Dτ))) dξ dτ is symmetric in x:
    F(-x, t) = F(x, t). The analytical formula is only valid for x >= 0, so we
    compute using |x| and the symmetry property.

    Parameters
    ----------
    x : ndarray
        Input values in space. Same shape as t.
    t : ndarray
        Input values in time. Same shape as x.
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.

    Returns
    -------
    ndarray
        Integral F(x[i], t[i]) for each i. Same shape as x and t.
    """
    x = np.asarray(x)
    t = np.asarray(t)

    # The double integral is symmetric in x: F(-x, t) = F(x, t)
    # Use |x| for the computation
    x = np.abs(x)

    isnan = np.isnan(x) | np.isnan(t)

    sqrt_diffusivity = np.sqrt(diffusivity)
    sqrt_pi = np.sqrt(np.pi)

    # Handle t <= 0 to avoid sqrt of negative
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_t = np.sqrt(np.maximum(t, 1e-30))
        exp_term = np.exp(-(x**2) / (4 * diffusivity * np.maximum(t, 1e-30)))
        erf_term = special.erf(x / (2 * sqrt_diffusivity * sqrt_t))

        term1 = -4 * sqrt_diffusivity * t ** (3 / 2) / (3 * sqrt_pi)
        term2 = (2 * sqrt_diffusivity / sqrt_pi) * (
            (2 * t ** (3 / 2) * exp_term / 3)
            - (sqrt_t * x**2 * exp_term / (3 * diffusivity))
            - (sqrt_pi * x**3 * erf_term / (6 * diffusivity ** (3 / 2)))
        )
        term3 = x * (
            t * erf_term
            + (x**2 * erf_term / (2 * diffusivity))
            + (sqrt_t * x * exp_term / (sqrt_pi * sqrt_diffusivity))
        )
        term4 = -(x**3) / (6 * diffusivity)
        out = term1 + term2 + term3 + term4

    out = np.where(isnan, np.nan, out)
    out = np.where(t <= 0.0, 0.0, out)
    return np.where(np.isinf(x) | np.isinf(t), np.inf, out)


def _erf_integral_space_paired(
    x: NDArray[np.float64],
    diffusivity: float,
    t: NDArray[np.float64],
    clip_to_inf: float = 6.0,
) -> NDArray[np.float64]:
    """Compute the integral of the error function at each (x[i], t[i]) point.

    This function computes the integral of erf from 0 to x[i] at time t[i],
    where x and t are broadcastable arrays.

    Parameters
    ----------
    x : ndarray
        Input x values. Broadcastable with t.
    diffusivity : float
        Diffusivity [m²/day]. Must be non-negative.
    t : ndarray
        Time values [day]. Broadcastable with x. Must be non-negative.
    clip_to_inf : float, optional
        Clip ax values beyond this to avoid numerical issues. Default is 6.

    Returns
    -------
    ndarray
        Integral values at each (x, t) point. Shape is broadcast shape of x and t.
    """
    x = np.asarray(x)
    t = np.asarray(t)

    # Broadcast x and t to common shape
    x, t = np.broadcast_arrays(x, t)
    out = np.zeros_like(x, dtype=float)

    # Compute a = 1/(2*sqrt(diffusivity*t)) from diffusivity and t
    # Handle edge cases: diffusivity=0 or t=0 means a=inf (step function)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where((diffusivity == 0.0) | (t == 0.0), np.inf, 1.0 / (2.0 * np.sqrt(diffusivity * t)))

    # Handle a=inf case: integral of sign(x) from 0 to x is |x|
    mask_a_inf = np.isinf(a)
    if np.any(mask_a_inf):
        out[mask_a_inf] = np.abs(x[mask_a_inf])

    # Handle a=0 case: erf(0) = 0, integral is 0
    mask_a_zero = a == 0.0
    # out[mask_a_zero] already 0

    # Handle finite non-zero a
    mask_valid = ~mask_a_inf & ~mask_a_zero
    if np.any(mask_valid):
        x_v = x[mask_valid]
        a_v = a[mask_valid]
        ax = a_v * x_v

        # Initialize result for valid entries
        result = np.zeros_like(x_v)

        # Handle clipped regions
        maskl = ax <= -clip_to_inf
        masku = ax >= clip_to_inf
        mask_mid = ~maskl & ~masku

        result[maskl] = -x_v[maskl] - 1 / (a_v[maskl] * np.sqrt(np.pi))
        result[masku] = x_v[masku] - 1 / (a_v[masku] * np.sqrt(np.pi))
        result[mask_mid] = x_v[mask_mid] * special.erf(ax[mask_mid]) + (np.exp(-(ax[mask_mid] ** 2)) - 1) / (
            a_v[mask_mid] * np.sqrt(np.pi)
        )

        out[mask_valid] = result

    return out


def _erf_mean_space_paired(
    edges: NDArray[np.float64],
    diffusivity: float,
    t: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the mean of the error function between edges with paired t values.

    This function computes the mean of erf(x/(2*sqrt(D*t))) between consecutive
    edges, where each cell has its own t value.

    Parameters
    ----------
    edges : ndarray, shape (n,)
        Cell edges of size n.
    diffusivity : float
        Diffusivity [m²/day]. Must be non-negative.
    t : ndarray
        Time values [day]. Broadcastable to shape (n,) for edge evaluation,
        or shape (n-1,) for cell evaluation. Must be non-negative.

    Returns
    -------
    ndarray, shape (n-1,)
        Mean of the error function for each cell.
    """
    edges = np.asarray(edges)
    t = np.asarray(t)

    # Clip edges to avoid numerical issues with very large values
    _edges = np.clip(edges, -1e6, 1e6)

    # Broadcast t to match edges length if needed
    t_edges = np.broadcast_to(t, edges.shape)

    # Compute integral at all edge points
    erfint = _erf_integral_space_paired(_edges, diffusivity=diffusivity, t=t_edges)

    # Compute mean using shifted views
    dx = _edges[1:] - _edges[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(dx != 0.0, (erfint[1:] - erfint[:-1]) / dx, np.nan)

    # Handle dx == 0 (point evaluation): erf at that point
    mask_dx_zero = dx == 0.0
    if np.any(mask_dx_zero):
        x_at_zero = edges[:-1][mask_dx_zero]
        t_at_zero = t_edges[:-1][mask_dx_zero]

        # Compute a = 1/(2*sqrt(diffusivity*t))
        with np.errstate(divide="ignore", invalid="ignore"):
            a_at_zero = np.where(
                (diffusivity == 0.0) | (t_at_zero == 0.0),
                np.inf,
                1.0 / (2.0 * np.sqrt(diffusivity * t_at_zero)),
            )

        # For infinite a, erf(inf*x) = sign(x)
        mask_a_inf = np.isinf(a_at_zero)
        result = np.zeros(np.sum(mask_dx_zero))
        if np.any(mask_a_inf):
            result[mask_a_inf] = np.sign(x_at_zero[mask_a_inf])
        if np.any(~mask_a_inf):
            result[~mask_a_inf] = special.erf(a_at_zero[~mask_a_inf] * x_at_zero[~mask_a_inf])
        out[mask_dx_zero] = result

    # Handle infinite edges (known asymptotic values)
    le, ue = edges[:-1], edges[1:]
    out[np.isinf(le) & ~np.isinf(ue)] = -1.0
    out[np.isneginf(le) & np.isneginf(ue)] = -1.0
    out[~np.isinf(le) & np.isinf(ue)] = 1.0
    out[np.isposinf(le) & np.isposinf(ue)] = 1.0
    out[np.isneginf(le) & np.isposinf(ue)] = 0.0
    out[np.isposinf(le) & np.isneginf(ue)] = 0.0

    return out


def _erf_mean_space_time(xedges, tedges, diffusivity):
    """
    Compute the mean of the error function over paired space-time cells.

    Computes the average value of erf(x/(2*sqrt(D*t))) over cells where
    cell i spans [xedges[i], xedges[i+1]] x [tedges[i], tedges[i+1]].

    The mean is computed using the inclusion-exclusion principle:
        (F(x₁,t₁) - F(x₀,t₁) - F(x₁,t₀) + F(x₀,t₀)) / ((x₁-x₀)(t₁-t₀))

    where F(x,t) = ∫₀ᵗ ∫₀ˣ erf(ξ/(2√(D·τ))) dξ dτ

    Parameters
    ----------
    xedges : ndarray
        Cell edges in space of size n.
    tedges : ndarray
        Cell edges in time of size n (same length as xedges).
    diffusivity : float
        Diffusivity [m²/day]. Must be non-negative.

    Returns
    -------
    ndarray
        Mean of the error function over each cell.
        Returns 1D array of length n-1, or scalar if n=2.
    """
    xedges = np.asarray(xedges)
    tedges = np.asarray(tedges)

    if len(xedges) != len(tedges):
        msg = "xedges and tedges must have the same length"
        raise ValueError(msg)

    n_cells = len(xedges) - 1
    out = np.full(n_cells, np.nan, dtype=float)

    dx = xedges[1:] - xedges[:-1]
    dt = tedges[1:] - tedges[:-1]

    mask_dx_zero = dx == 0.0
    mask_dt_zero = dt == 0.0

    # Handle cells where both dx=0 AND dt=0 (point evaluation of erf)
    mask_both_zero = mask_dx_zero & mask_dt_zero
    if np.any(mask_both_zero):
        x_pts = xedges[:-1][mask_both_zero]
        t_pts = tedges[:-1][mask_both_zero]
        with np.errstate(divide="ignore", invalid="ignore"):
            a_pts = np.where(
                (diffusivity == 0.0) | (t_pts == 0.0),
                np.inf,
                1.0 / (2.0 * np.sqrt(diffusivity * t_pts)),
            )
        mask_a_inf = np.isinf(a_pts)
        result = np.zeros(np.sum(mask_both_zero))
        if np.any(mask_a_inf):
            result[mask_a_inf] = np.sign(x_pts[mask_a_inf])
        if np.any(~mask_a_inf):
            result[~mask_a_inf] = special.erf(a_pts[~mask_a_inf] * x_pts[~mask_a_inf])
        out[mask_both_zero] = result

    # Handle remaining dt=0 cells (mean over space at fixed time)
    mask_dt_zero_only = mask_dt_zero & ~mask_dx_zero
    if np.any(mask_dt_zero_only):
        out[mask_dt_zero_only] = _erf_mean_space_paired(xedges, diffusivity=diffusivity, t=tedges)[mask_dt_zero_only]

    # Handle remaining dx=0 cells (mean over time at fixed x)
    mask_dx_zero_only = mask_dx_zero & ~mask_dt_zero
    if np.any(mask_dx_zero_only):
        out[mask_dx_zero_only] = _erf_mean_time_paired(tedges, x=xedges, diffusivity=diffusivity)[mask_dx_zero_only]

    # Handle remaining cells with full double integral
    mask_remainder = ~mask_dx_zero & ~mask_dt_zero
    if np.any(mask_remainder):
        if diffusivity == 0.0:
            x_mid = (xedges[:-1] + xedges[1:]) / 2
            out[mask_remainder] = np.sign(x_mid[mask_remainder])
        else:
            idx_remainder = np.where(mask_remainder)[0]
            x_corners = np.concatenate([
                xedges[idx_remainder],
                xedges[idx_remainder + 1],
                xedges[idx_remainder],
                xedges[idx_remainder + 1],
            ])
            t_corners = np.concatenate([
                tedges[idx_remainder],
                tedges[idx_remainder + 1],
                tedges[idx_remainder + 1],
                tedges[idx_remainder],
            ])
            f = _erf_integral_space_time_pointwise(x_corners, t_corners, diffusivity)
            n_rem = len(idx_remainder)
            f_00 = f[:n_rem]
            f_11 = f[n_rem : 2 * n_rem]
            f_01 = f[2 * n_rem : 3 * n_rem]
            f_10 = f[3 * n_rem :]

            double_integrals = f_11 - f_10 - f_01 + f_00
            cell_areas = dx[mask_remainder] * dt[mask_remainder]
            out[mask_remainder] = double_integrals / cell_areas

    # Handle infinite x edges
    le, ue = xedges[:-1], xedges[1:]
    out[np.isinf(le) & ~np.isinf(ue)] = -1.0
    out[np.isneginf(le) & np.isneginf(ue)] = -1.0
    out[~np.isinf(le) & np.isinf(ue)] = 1.0
    out[np.isposinf(le) & np.isposinf(ue)] = 1.0
    out[np.isneginf(le) & np.isposinf(ue)] = 0.0
    out[np.isposinf(le) & np.isneginf(ue)] = 0.0

    # Handle NaN in edges
    out[np.isnan(xedges[:-1]) | np.isnan(xedges[1:])] = np.nan
    out[np.isnan(tedges[:-1]) | np.isnan(tedges[1:])] = np.nan

    if len(out) == 1:
        return out[0]
    return out


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: npt.ArrayLike,
    diffusivity: float,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """
    Compute extracted concentration with advection and longitudinal dispersion.

    This function models 1D solute transport through an aquifer system,
    combining advective transport (based on residence times) with longitudinal
    dispersion (diffusive spreading during transport).

    The physical model assumes:
    1. Water infiltrates with concentration cin at time t_in
    2. Water travels distance L through aquifer with residence time tau = V_pore / Q
    3. During transport, longitudinal dispersion causes spreading
    4. At extraction, the concentration is a diffused breakthrough curve

    The dispersion is characterized by the longitudinal dispersion coefficient,
    which should be computed by the user as:

        D_L = D_m + alpha_L * v

    where D_m is molecular diffusion [m^2/day], alpha_L is dispersivity [m],
    and v is pore velocity [m/day] (v = q/n, where q is specific discharge
    and n is porosity).

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water [concentration units].
        Length must match the number of time bins defined by tedges.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
        Length must match cin and the number of time bins defined by tedges.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cin and flow data. Has length of
        len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
        The output concentration is averaged over each bin.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution
        of flow paths. Each pore volume determines the residence time for
        that flow path: tau = V_pore / Q.
    streamline_length : array-like
        Array of travel distances [m] corresponding to each pore volume.
        Must have the same length as aquifer_pore_volumes. The travel distance
        determines the dispersion length scale: sqrt(D_L * tau).
    diffusivity : float
        Longitudinal dispersion coefficient [m2/day]. Must be non-negative.
        Compute as D_L = D_m + alpha_L * v where:
        - D_m: molecular diffusion coefficient [m2/day]
        - alpha_L: longitudinal dispersivity [m]
        - v: pore velocity [m/day] (v = q/n, where q is specific discharge
          and n is porosity)
        Set to 0 for pure advection (no dispersion).
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the extracted water. Same units as cin.
        Length equals len(cout_tedges) - 1. NaN values indicate time periods
        with no valid contributions from the infiltration data.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, if diffusivity is negative,
        or if aquifer_pore_volumes and streamline_length have different lengths.

    See Also
    --------
    gwtransport.advection.infiltration_to_extraction : Pure advection (no dispersion)

    Notes
    -----
    The algorithm works as follows:

    1. For each output time bin [t_out_start, t_out_end]:
       - Compute the residence time for each pore volume
       - Determine which infiltration times contribute to this output bin

    2. For each input concentration step (change in cin):
       - The step diffuses as it travels through the aquifer
       - The diffused contribution is computed using the error function
       - Time-averaging over the output bin uses analytical space-time averaging

    3. The final output is a flow-weighted average across all pore volumes

    The error function solution assumes an initial step function that diffuses
    over time. The position coordinate x represents the distance from the
    concentration front to the observation point.

    Examples
    --------
    Basic usage with constant flow:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion2 import infiltration_to_extraction
    >>>
    >>> # Create time edges
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>>
    >>> # Input concentration (step function) and constant flow
    >>> cin = np.zeros(len(tedges) - 1)
    >>> cin[5:10] = 1.0  # Pulse of concentration
    >>> flow = np.ones(len(tedges) - 1) * 100.0  # 100 m3/day
    >>>
    >>> # Single pore volume of 500 m3, travel distance 100 m
    >>> aquifer_pore_volumes = np.array([500.0])
    >>> streamline_length = np.array([100.0])
    >>>
    >>> # Compute with dispersion
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     diffusivity=1.0,  # m2/day
    ... )

    With multiple pore volumes (heterogeneous aquifer):

    >>> # Distribution of pore volumes and corresponding travel distances
    >>> aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
    >>> streamline_length = np.array([80.0, 100.0, 120.0])
    >>>
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     diffusivity=1.0,
    ... )
    """
    # Convert to pandas DatetimeIndex if needed
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    tedges = pd.DatetimeIndex(tedges)

    # Convert to arrays
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    streamline_length = np.asarray(streamline_length, dtype=float)

    # Input validation
    if len(tedges) != len(cin) + 1:
        msg = "tedges must have one more element than cin"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if len(aquifer_pore_volumes) != len(streamline_length):
        msg = "aquifer_pore_volumes and streamline_length must have the same length"
        raise ValueError(msg)
    if diffusivity < 0:
        msg = "diffusivity must be non-negative"
        raise ValueError(msg)
    if np.any(np.isnan(cin)):
        msg = "cin contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(aquifer_pore_volumes <= 0):
        msg = "aquifer_pore_volumes must be positive"
        raise ValueError(msg)
    if np.any(streamline_length <= 0):
        msg = "streamline_length must be positive"
        raise ValueError(msg)

    # Extend tedges for spin up
    tedges = pd.DatetimeIndex([
        tedges[0] - pd.Timedelta("36500D"),
        *list(tedges[1:-1]),
        tedges[-1] + pd.Timedelta("36500D"),
    ])

    # Compute the cumulative flow at tedges
    infiltration_volume = flow * (np.diff(tedges) / pd.Timedelta("1D"))  # m3
    cumulative_volume_at_cin_tedges = np.concatenate(([0], np.cumsum(infiltration_volume)))

    # Compute the cumulative flow at cout_tedges
    cumulative_volume_at_cout_tedges = np.interp(cout_tedges, tedges, cumulative_volume_at_cin_tedges)

    rt_edges_2d = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",
    )

    # Compute residence time at cout_tedges to identify valid output bins
    # RT is NaN for cout_tedges beyond the input data range
    rt_at_cout_tedges = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    # Output bin i is valid if both cout_tedges[i] and cout_tedges[i+1] have valid RT for all pore volumes
    # Check if any pore volume has NaN RT at the bin edges
    valid_cout_bins = ~np.any(np.isnan(rt_at_cout_tedges[:, :-1]) | np.isnan(rt_at_cout_tedges[:, 1:]), axis=0)

    # Initialize coefficient matrix accumulator
    n_cout_bins = len(cout_tedges) - 1
    n_cin_bins = len(cin)
    n_cin_edges = len(tedges)
    accumulated_coeff = np.zeros((n_cout_bins, n_cin_bins))

    # At cout_tedges < tedges the concentration has not entered the aquifer yet.
    isactive = cout_tedges.to_numpy()[:, None] >= tedges.to_numpy()[None, :]

    # Loop over each pore volume
    for i_pv in range(len(aquifer_pore_volumes)):
        # The amount of apv between a change in concentration (tedges) and the point of extraction.
        # Positive in the flow direction.
        delta_volume_after_extraction = (
            cumulative_volume_at_cout_tedges[:, None]
            - cumulative_volume_at_cin_tedges[None, :]
            - (retardation_factor * aquifer_pore_volumes[i_pv])
        )
        delta_volume_after_extraction[~isactive] = np.nan

        # Convert volume to distances (x-coordinate for erf)
        step_widths_cin = (
            delta_volume_after_extraction / (retardation_factor * aquifer_pore_volumes[i_pv]) * streamline_length[i_pv]
        )

        # Compute the time a concentration jump is active, limited by the residence time in days
        time_active = (cout_tedges.to_numpy()[:, None] - tedges.to_numpy()[None, :]) / pd.to_timedelta(1, unit="D")
        time_active[~isactive] = np.nan
        time_active = np.minimum(time_active, rt_edges_2d[[i_pv]])

        # Compute erf response for each step at tedges[j]
        # response[i_cout_bin, j_step] = mean erf for step j at output bin i
        response = np.zeros((n_cout_bins, n_cin_edges))

        for j in range(n_cin_edges):
            # Extract edges for this step across all cout edges
            xedges_j = step_widths_cin[:, j]  # shape (n_cout_edges,)
            tedges_j = time_active[:, j]  # shape (n_cout_edges,)

            response[:, j] = _erf_mean_space_time(xedges_j, tedges_j, diffusivity)

        # Convert erf response [-1, 1] to breakthrough fraction [0, 1]
        frac = 0.5 * (1 + response)  # shape (n_cout_bins, n_cin_edges)

        # Coefficient matrix: coeff[i, j] = frac[i, j] - frac[i, j+1]
        # This represents the fraction of cin[j] that arrives in output bin i
        # frac[:, j] is the breakthrough of step at tedges[j]
        # For cin[j] (between tedges[j] and tedges[j+1]), contribution is frac at start minus frac at end
        # Handle NaN: if frac[j+1] is NaN but frac[j] is valid, use frac[j]
        frac_start = frac[:, :-1]  # frac at tedges[j] for j=0..n-1
        frac_end = frac[:, 1:]  # frac at tedges[j+1] for j=0..n-1
        # Where frac_end is NaN but frac_start is valid, use frac_start
        frac_end_filled = np.where(np.isnan(frac_end) & ~np.isnan(frac_start), 0.0, frac_end)
        coeff = frac_start - frac_end_filled  # shape (n_cout_bins, n_cin_bins)

        accumulated_coeff += coeff

    # Average across pore volumes
    coeff_matrix = accumulated_coeff / len(aquifer_pore_volumes)

    # Handle NaN in coefficient matrix: replace with 0 for multiplication
    # NaN means that cin bin hasn't entered the aquifer yet for that cout bin
    coeff_matrix_filled = np.nan_to_num(coeff_matrix, nan=0.0)

    # Matrix multiply: cout = coeff_matrix @ cin
    cout = coeff_matrix_filled @ cin

    # Handle invalid outputs where no valid contributions exist
    # A cout bin is invalid when:
    # 1. The sum of coefficients is near zero (no cin has broken through yet - early bins)
    # 2. The output bin extends beyond the input data range (late bins - from valid_cout_bins)
    total_coeff = np.sum(coeff_matrix_filled, axis=1)
    no_valid_contribution = (total_coeff < EPSILON_COEFF_SUM) | ~valid_cout_bins
    cout[no_valid_contribution] = np.nan

    return cout
