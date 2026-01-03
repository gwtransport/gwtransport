"""
Analytical solutions for 1D advection-dispersion transport.

This module implements analytical solutions for solute transport in 1D aquifer
systems, combining advection with longitudinal dispersion. The solutions are
based on the error function (erf) and its integrals.

Key function:
- infiltration_to_extraction: Main transport function combining advection and dispersion

The dispersion is characterized by the longitudinal dispersion coefficient D_L,
which is computed internally from:

    D_L = D_m + alpha_L * v

where:
- D_m is the molecular diffusion coefficient [m^2/day]
- alpha_L is the longitudinal dispersivity [m]
- v is the pore velocity [m/day], computed as v = L / tau_mean

The velocity v is computed per (pore_volume, output_bin) using the mean residence
time, which correctly accounts for time-varying flow. This formulation ensures that:
- Molecular diffusion spreading scales with residence time: sqrt(D_m * tau)
- Mechanical dispersion spreading scales with travel distance: sqrt(alpha_L * L)
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import NDArray
from scipy import special

from gwtransport.residence_time import residence_time, residence_time_mean

# Numerical tolerance for coefficient sum to determine valid output bins
EPSILON_COEFF_SUM = 1e-10


def _erf_integral_time(
    t: NDArray[np.float64],
    x: NDArray[np.float64],
    diffusivity: npt.ArrayLike,
) -> NDArray[np.float64]:
    r"""Compute the integral of the error function over time at each (t[i], x[i]) point.

    This function computes the integral of erf(x/(2*sqrt(D*tau))) from 0 to t,
    where t, x, and diffusivity are broadcastable arrays.

    The analytical solution is:

    .. math::
        \int_0^t \text{erf}\left(\frac{x}{2\sqrt{D \tau}}\right) d\tau
        = t \cdot \text{erf}\left(\frac{x}{2\sqrt{D t}}\right)
        + \frac{x \sqrt{t}}{\sqrt{\pi D}} \exp\left(-\frac{x^2}{4 D t}\right)
        - \frac{x^2}{2D} \text{erfc}\left(\frac{x}{2\sqrt{D t}}\right)

    Parameters
    ----------
    t : ndarray
        Input time values. Broadcastable with x and diffusivity. Must be non-negative.
    x : ndarray
        Position values. Broadcastable with t and diffusivity.
    diffusivity : float or ndarray
        Diffusivity [m²/day]. Must be positive. Broadcastable with t and x.

    Returns
    -------
    ndarray
        Integral values at each (t, x, D) point. Shape is broadcast shape of inputs.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    diffusivity = np.asarray(diffusivity, dtype=float)

    # Broadcast all inputs to common shape
    t, x, diffusivity = np.broadcast_arrays(t, x, diffusivity)
    out = np.zeros_like(t, dtype=float)

    # Mask for valid computation: t > 0, x != 0, and diffusivity > 0
    mask_valid = (t > 0.0) & (x != 0.0) & (diffusivity > 0.0)

    if np.any(mask_valid):
        t_v = t[mask_valid]
        x_v = x[mask_valid]
        d_v = diffusivity[mask_valid]

        sqrt_t = np.sqrt(t_v)
        sqrt_d = np.sqrt(d_v)
        sqrt_pi = np.sqrt(np.pi)

        arg = x_v / (2 * sqrt_d * sqrt_t)
        exp_term = np.exp(-(x_v**2) / (4 * d_v * t_v))
        erf_term = special.erf(arg)
        erfc_term = special.erfc(arg)

        term1 = t_v * erf_term
        term2 = (x_v * sqrt_t / (sqrt_pi * sqrt_d)) * exp_term
        term3 = -(x_v**2 / (2 * d_v)) * erfc_term

        out[mask_valid] = term1 + term2 + term3

    # Handle infinity: as t -> inf, integral -> inf * sign(x)
    mask_t_inf = np.isinf(t)
    if np.any(mask_t_inf):
        out[mask_t_inf] = np.inf * np.sign(x[mask_t_inf])

    return out


def _erf_integral_space_time(x, t, diffusivity):
    """
    Compute the integral of the error function in space and time at (x, t) points.

    This function evaluates
    F(x[i], t[i], D[i]) for each i, where x, t, and diffusivity are broadcastable.
    This is useful for batched computations where we need F at arbitrary
    (x, t, D) triplets.

    The double integral F(x,t,D) = ∫₀ᵗ ∫₀ˣ erf(ξ/(2√(Dτ))) dξ dτ is symmetric in x:
    F(-x, t, D) = F(x, t, D). The analytical formula is only valid for x >= 0, so we
    compute using |x| and the symmetry property.

    Parameters
    ----------
    x : ndarray
        Input values in space. Broadcastable with t and diffusivity.
    t : ndarray
        Input values in time. Broadcastable with x and diffusivity.
    diffusivity : float or ndarray
        Diffusivity [m²/day]. Must be positive. Can be a scalar or array
        broadcastable with x and t.

    Returns
    -------
    ndarray
        Integral F(x[i], t[i], D[i]) for each i. Shape is broadcast shape of inputs.
    """
    x = np.asarray(x)
    t = np.asarray(t)
    diffusivity = np.asarray(diffusivity)

    # The double integral is symmetric in x: F(-x, t, D) = F(x, t, D)
    # Use |x| for the computation
    x = np.abs(x)

    # Broadcast all inputs to common shape
    x, t, diffusivity = np.broadcast_arrays(x, t, diffusivity)

    isnan = np.isnan(x) | np.isnan(t) | np.isnan(diffusivity)

    sqrt_diffusivity = np.sqrt(diffusivity)
    sqrt_pi = np.sqrt(np.pi)

    # Handle t <= 0 or diffusivity <= 0 to avoid sqrt of negative / division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_t = np.maximum(t, 1e-30)
        safe_d = np.maximum(diffusivity, 1e-30)
        sqrt_t = np.sqrt(safe_t)
        exp_term = np.exp(-(x**2) / (4 * safe_d * safe_t))
        erf_term = special.erf(x / (2 * np.sqrt(safe_d) * sqrt_t))

        term1 = -4 * sqrt_diffusivity * t ** (3 / 2) / (3 * sqrt_pi)
        term2 = (2 * sqrt_diffusivity / sqrt_pi) * (
            (2 * t ** (3 / 2) * exp_term / 3)
            - (sqrt_t * x**2 * exp_term / (3 * safe_d))
            - (sqrt_pi * x**3 * erf_term / (6 * safe_d ** (3 / 2)))
        )
        term3 = x * (
            t * erf_term + (x**2 * erf_term / (2 * safe_d)) + (sqrt_t * x * exp_term / (sqrt_pi * np.sqrt(safe_d)))
        )
        term4 = -(x**3) / (6 * safe_d)
        out = term1 + term2 + term3 + term4

    out = np.where(isnan, np.nan, out)
    out = np.where(t <= 0.0, 0.0, out)
    out = np.where(diffusivity <= 0.0, 0.0, out)
    return np.where(np.isinf(x) | np.isinf(t), np.inf, out)


def _erf_integral_space(
    x: NDArray[np.float64],
    diffusivity: npt.ArrayLike,
    t: NDArray[np.float64],
    clip_to_inf: float = 6.0,
) -> NDArray[np.float64]:
    """Compute the integral of the error function at each (x[i], t[i], D[i]) point.

    This function computes the integral of erf from 0 to x[i] at time t[i],
    where x, t, and diffusivity are broadcastable arrays.

    Parameters
    ----------
    x : ndarray
        Input x values. Broadcastable with t and diffusivity.
    diffusivity : float or ndarray
        Diffusivity [m²/day]. Must be non-negative. Broadcastable with x and t.
    t : ndarray
        Time values [day]. Broadcastable with x and diffusivity. Must be non-negative.
    clip_to_inf : float, optional
        Clip ax values beyond this to avoid numerical issues. Default is 6.

    Returns
    -------
    ndarray
        Integral values at each (x, t, D) point. Shape is broadcast shape of inputs.
    """
    x = np.asarray(x)
    t = np.asarray(t)
    diffusivity = np.asarray(diffusivity)

    # Broadcast all inputs to common shape
    x, t, diffusivity = np.broadcast_arrays(x, t, diffusivity)
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


def _erf_mean_space_time(xedges, tedges, diffusivity):
    """
    Compute the mean of the error function over paired space-time cells.

    Computes the average value of erf(x/(2*sqrt(D*t))) over cells where
    cell i spans [xedges[i], xedges[i+1]] x [tedges[i], tedges[i+1]].

    The mean is computed using the inclusion-exclusion principle:
        (F(x₁,t₁,D) - F(x₀,t₁,D) - F(x₁,t₀,D) + F(x₀,t₀,D)) / ((x₁-x₀)(t₁-t₀))

    where F(x,t,D) = ∫₀ᵗ ∫₀ˣ erf(ξ/(2√(D·τ))) dξ dτ

    Parameters
    ----------
    xedges : ndarray
        Cell edges in space of size n.
    tedges : ndarray
        Cell edges in time of size n (same length as xedges).
    diffusivity : float or ndarray
        Diffusivity [m²/day]. Must be non-negative. Can be a scalar (same for
        all cells) or an array of size n-1 (one value per cell).

    Returns
    -------
    ndarray
        Mean of the error function over each cell.
        Returns 1D array of length n-1, or scalar if n=2.
    """
    xedges = np.asarray(xedges)
    tedges = np.asarray(tedges)
    diffusivity = np.atleast_1d(np.asarray(diffusivity))

    if len(xedges) != len(tedges):
        msg = "xedges and tedges must have the same length"
        raise ValueError(msg)

    n_cells = len(xedges) - 1

    # Broadcast scalar diffusivity to all cells
    if diffusivity.size == 1:
        diffusivity = np.broadcast_to(diffusivity, (n_cells,))

    if len(diffusivity) != n_cells:
        msg = "diffusivity must be a scalar or have length n_cells (len(xedges) - 1)"
        raise ValueError(msg)

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
        d_pts = diffusivity[mask_both_zero]
        with np.errstate(divide="ignore", invalid="ignore"):
            a_pts = np.where(
                (d_pts == 0.0) | (t_pts == 0.0),
                np.inf,
                1.0 / (2.0 * np.sqrt(d_pts * t_pts)),
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
        idx_dt_zero = np.where(mask_dt_zero_only)[0]
        n_dt_zero = len(idx_dt_zero)

        # Build edge arrays for all dt=0 cells
        x_edges_flat = np.concatenate([xedges[idx_dt_zero], xedges[idx_dt_zero + 1]])
        t_edges_flat = np.concatenate([tedges[idx_dt_zero], tedges[idx_dt_zero + 1]])
        d_cells = diffusivity[idx_dt_zero]

        # Replicate diffusivity for both edges of each cell
        d_edges = np.concatenate([d_cells, d_cells])

        # Compute integral at all edge points
        erfint_flat = _erf_integral_space(x_edges_flat, diffusivity=d_edges, t=t_edges_flat)
        erfint_lower = erfint_flat[:n_dt_zero]
        erfint_upper = erfint_flat[n_dt_zero:]

        dx_cells = xedges[idx_dt_zero + 1] - xedges[idx_dt_zero]
        out[idx_dt_zero] = (erfint_upper - erfint_lower) / dx_cells

    # Handle remaining dx=0 cells (mean over time at fixed x)
    mask_dx_zero_only = mask_dx_zero & ~mask_dt_zero
    if np.any(mask_dx_zero_only):
        idx_dx_zero = np.where(mask_dx_zero_only)[0]
        n_dx_zero = len(idx_dx_zero)

        # Build edge arrays for all dx=0 cells
        t_edges_flat = np.concatenate([tedges[idx_dx_zero], tedges[idx_dx_zero + 1]])
        x_edges_flat = np.concatenate([xedges[idx_dx_zero], xedges[idx_dx_zero + 1]])
        d_cells = diffusivity[idx_dx_zero]

        # Replicate diffusivity for both edges of each cell
        d_edges = np.concatenate([d_cells, d_cells])

        # Compute integral at all edge points
        erfint_flat = _erf_integral_time(t_edges_flat, x=x_edges_flat, diffusivity=d_edges)
        erfint_lower = erfint_flat[:n_dx_zero]
        erfint_upper = erfint_flat[n_dx_zero:]

        dt_cells = tedges[idx_dx_zero + 1] - tedges[idx_dx_zero]
        out[idx_dx_zero] = (erfint_upper - erfint_lower) / dt_cells

    # Handle remaining cells with full double integral
    mask_remainder = ~mask_dx_zero & ~mask_dt_zero
    if np.any(mask_remainder):
        # Check for zero diffusivity cells
        d_remainder = diffusivity[mask_remainder]
        mask_d_zero = d_remainder == 0.0

        if np.any(mask_d_zero):
            # For zero diffusivity, erf becomes sign function
            idx_d_zero = np.where(mask_remainder)[0][mask_d_zero]
            x_mid = (xedges[idx_d_zero] + xedges[idx_d_zero + 1]) / 2
            out[idx_d_zero] = np.sign(x_mid)

        # Handle non-zero diffusivity cells
        mask_d_nonzero = ~mask_d_zero
        if np.any(mask_d_nonzero):
            idx_remainder = np.where(mask_remainder)[0][mask_d_nonzero]
            d_nonzero = d_remainder[mask_d_nonzero]

            # Build corner arrays: each cell has 4 corners, all using same diffusivity
            x_corners = np.concatenate([
                xedges[idx_remainder],  # x0
                xedges[idx_remainder + 1],  # x1
                xedges[idx_remainder],  # x0
                xedges[idx_remainder + 1],  # x1
            ])
            t_corners = np.concatenate([
                tedges[idx_remainder],  # t0
                tedges[idx_remainder + 1],  # t1
                tedges[idx_remainder + 1],  # t1
                tedges[idx_remainder],  # t0
            ])
            # Replicate diffusivity for all 4 corners of each cell
            d_corners = np.concatenate([d_nonzero, d_nonzero, d_nonzero, d_nonzero])

            f = _erf_integral_space_time(x_corners, t_corners, d_corners)
            n_rem = len(idx_remainder)
            f_00 = f[:n_rem]
            f_11 = f[n_rem : 2 * n_rem]
            f_01 = f[2 * n_rem : 3 * n_rem]
            f_10 = f[3 * n_rem :]

            double_integrals = f_11 - f_10 - f_01 + f_00
            cell_areas = dx[idx_remainder] * dt[idx_remainder]
            out[idx_remainder] = double_integrals / cell_areas

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
    molecular_diffusivity: npt.ArrayLike,
    longitudinal_dispersivity: npt.ArrayLike,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
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

    The longitudinal dispersion coefficient D_L is computed internally as:

        D_L = D_m + alpha_L * v

    where v = L / tau_mean is the mean pore velocity computed from the mean
    residence time for each (pore_volume, output_bin) combination. This
    formulation correctly captures that:
    - Molecular diffusion spreading scales with residence time: sqrt(D_m * tau)
    - Mechanical dispersion spreading scales with travel distance: sqrt(alpha_L * L)

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
        Must have the same length as aquifer_pore_volumes.
    molecular_diffusivity : float or array-like
        Molecular diffusion coefficient [m2/day]. Can be a scalar (same for all
        pore volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative. Represents Brownian motion of solute molecules,
        independent of velocity.
    longitudinal_dispersivity : float or array-like
        Longitudinal dispersivity [m]. Can be a scalar (same for all pore
        volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative. Represents mechanical dispersion due to pore-scale
        velocity variations. Set to 0 for pure molecular diffusion.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption.
    suppress_dispersion_warning : bool, optional
        If True, suppress the warning when using multiple pore volumes with
        non-zero longitudinal_dispersivity. Default is False.

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

    Warns
    -----
    UserWarning
        If multiple pore volumes are used with non-zero longitudinal_dispersivity.
        This may lead to double-counting of spreading effects. Suppress with
        ``suppress_dispersion_warning=True`` if this is intentional.

    See Also``````````````````````````````````
    --------
    gwtransport.advection.infiltration_to_extraction : Pure advection (n``````````````````````````````````o dispersion)
    examples/05_Diffusion_Dispersion.ipynb : Guidance on choosing modeling approaches

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

    The error function solution assumes an initial step function that difwq`1       qw  q`fuses
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
    >>> # Compute with dispersion (molecular diffusion + dispersivity)
    >>> # Scalar values broadcast to all pore volumes
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     molecular_diffusivity=1e-4,  # m2/day, same for all pore volumes
    ...     longitudinal_dispersivity=1.0,  # m, same for all pore volumes
    ... )

    With multiple pore volumes (heterogeneous aquifer):

    >>> # Distribution of pore volumes and corresponding travel distances
    >>> aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
    >>> streamline_length = np.array([80.0, 100.0, 120.0])
    >>>
    >>> # Scalar diffusion parameters broadcast to all pore volumes
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     molecular_diffusivity=1e-4,  # m2/day
    ...     longitudinal_dispersivity=1.0,  # m
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

    # Convert diffusion parameters to arrays and broadcast to pore volumes
    n_pore_volumes = len(aquifer_pore_volumes)
    molecular_diffusivity = np.atleast_1d(np.asarray(molecular_diffusivity, dtype=float))
    longitudinal_dispersivity = np.atleast_1d(np.asarray(longitudinal_dispersivity, dtype=float))

    # Broadcast scalar values to match pore volumes
    if molecular_diffusivity.size == 1:
        molecular_diffusivity = np.broadcast_to(molecular_diffusivity, (n_pore_volumes,)).copy()
    if longitudinal_dispersivity.size == 1:
        longitudinal_dispersivity = np.broadcast_to(longitudinal_dispersivity, (n_pore_volumes,)).copy()

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
    if len(molecular_diffusivity) != n_pore_volumes:
        msg = "molecular_diffusivity must be a scalar or have same length as aquifer_pore_volumes"
        raise ValueError(msg)
    if len(longitudinal_dispersivity) != n_pore_volumes:
        msg = "longitudinal_dispersivity must be a scalar or have same length as aquifer_pore_volumes"
        raise ValueError(msg)
    if np.any(molecular_diffusivity < 0):
        msg = "molecular_diffusivity must be non-negative"
        raise ValueError(msg)
    if np.any(longitudinal_dispersivity < 0):
        msg = "longitudinal_dispersivity must be non-negative"
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

    # Check for conflicting approaches: multiple pore volumes with longitudinal dispersivity
    # Both represent spreading mechanisms but at different scales:
    # - Multiple pore volumes: macro-scale aquifer heterogeneity
    # - Longitudinal dispersivity: pore-scale mechanical dispersion
    # Using both simultaneously may lead to double-counting of spreading effects.
    # See notebook 05_Diffusion_Dispersion.ipynb for guidance on choosing the right approach.
    if n_pore_volumes > 1 and np.any(longitudinal_dispersivity > 0) and not suppress_dispersion_warning:
        msg = (
            "Using multiple aquifer_pore_volumes with non-zero longitudinal_dispersivity. "
            "Multiple pore volumes represent macro-scale aquifer heterogeneity, while "
            "longitudinal_dispersivity represents pore-scale mechanical dispersion. "
            "Using both may double-count spreading effects. Consider:\n"
            "  1. Use multiple pore volumes with longitudinal_dispersivity=0 for macro-scale heterogeneity\n"
            "  2. Use a single pore volume with longitudinal_dispersivity>0 for pore-scale dispersion\n"
            "See examples/05_Diffusion_Dispersion.ipynb for guidance on selecting the appropriate method.\n"
            "Suppress this warning with suppress_dispersion_warning=True."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

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

    # Compute mean residence time per (pore_volume, cout_bin) for velocity calculation
    # Shape: (n_pore_volumes, n_cout_bins)
    rt_mean_2d = residence_time_mean(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        direction="extraction_to_infiltration",
        retardation_factor=retardation_factor,
    )

    # Compute velocity: v = L / tau_mean
    # Shape: (n_pore_volumes, n_cout_bins)
    # Use explicit validity handling instead of error suppression
    valid_rt = np.isfinite(rt_mean_2d) & (rt_mean_2d > 0)
    velocity_2d = np.where(valid_rt, streamline_length[:, None] / rt_mean_2d, 0.0)

    # Compute D_L = D_m + alpha_L * v
    # Shape: (n_pore_volumes, n_cout_bins)
    # Fall back to molecular diffusivity only where RT is invalid (spinup period)
    diffusivity_2d = np.where(
        valid_rt,
        molecular_diffusivity[:, None] + longitudinal_dispersivity[:, None] * velocity_2d,
        molecular_diffusivity[:, None],
    )

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

        # Get diffusivity for this pore volume across all cout bins
        diff_pv = diffusivity_2d[i_pv, :]  # shape (n_cout_bins,)

        for j in range(n_cin_edges):
            # Extract edges for this step across all cout edges
            xedges_j = step_widths_cin[:, j]  # shape (n_cout_edges,)
            tedges_j = time_active[:, j]  # shape (n_cout_edges,)

            # Use per-cell diffusivity for varying D_L across output bins
            response[:, j] = _erf_mean_space_time(xedges_j, tedges_j, diff_pv)

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


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cin_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: npt.ArrayLike,
    molecular_diffusivity: npt.ArrayLike,
    longitudinal_dispersivity: npt.ArrayLike,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
) -> npt.NDArray[np.floating]:
    """
    Compute infiltration concentration from extracted water (deconvolution with dispersion).

    This function implements the inverse of infiltration_to_extraction, reconstructing
    the original infiltration concentration from the extracted water concentration.
    It explicitly constructs the weights matrix (rather than inverting the forward matrix)
    to ensure numerical stability and proper handling of dispersion effects.

    The physical model assumes:
    1. Water is extracted with concentration cout at time t_out
    2. Water traveled distance L through aquifer with residence time tau = V_pore / Q
    3. During transport, longitudinal dispersion caused spreading
    4. At infiltration, the concentration is reconstructed from the diffused signal

    The longitudinal dispersion coefficient D_L is computed internally as:

        D_L = D_m + alpha_L * v

    where v = L / tau_mean is the mean pore velocity computed from the mean
    residence time for each (pore_volume, cin_bin) combination.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water [concentration units].
        Length must match the number of time bins defined by tedges.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
        Length must match cout and the number of time bins defined by tedges.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cout and flow data (extraction times).
        Has length of len(cout) + 1.
    cin_tedges : pandas.DatetimeIndex
        Time edges for output infiltration data bins. Has length of desired output + 1.
        The output concentration is averaged over each bin.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution
        of flow paths. Each pore volume determines the residence time for
        that flow path: tau = V_pore / Q.
    streamline_length : array-like
        Array of travel distances [m] corresponding to each pore volume.
        Must have the same length as aquifer_pore_volumes.
    molecular_diffusivity : float or array-like
        Molecular diffusion coefficient [m2/day]. Can be a scalar (same for all
        pore volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative.
    longitudinal_dispersivity : float or array-like
        Longitudinal dispersivity [m]. Can be a scalar (same for all pore
        volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption.
    suppress_dispersion_warning : bool, optional
        If True, suppress the warning when using multiple pore volumes with
        non-zero longitudinal_dispersivity. Default is False.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Same units as cout.
        Length equals len(cin_tedges) - 1. NaN values indicate time periods
        with no valid contributions from the extraction data.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, if diffusivity is negative,
        or if aquifer_pore_volumes and streamline_length have different lengths.

    Warns
    -----
    UserWarning
        If multiple pore volumes are used with non-zero longitudinal_dispersivity.

    See Also
    --------
    infiltration_to_extraction : Forward operation (convolution)
    gwtransport.advection.extraction_to_infiltration : Pure advection (no dispersion)

    Examples
    --------
    Basic usage with constant flow:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion2 import extraction_to_infiltration
    >>>
    >>> # Create time edges for extraction data
    >>> tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>> cin_tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>>
    >>> # Extracted concentration and constant flow
    >>> cout = np.zeros(len(tedges) - 1)
    >>> cout[5:10] = 1.0  # Observed pulse at extraction
    >>> flow = np.ones(len(tedges) - 1) * 100.0  # 100 m3/day
    >>>
    >>> # Single pore volume of 500 m3, travel distance 100 m
    >>> aquifer_pore_volumes = np.array([500.0])
    >>> streamline_length = np.array([100.0])
    >>>
    >>> # Reconstruct infiltration concentration
    >>> cin = extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cin_tedges=cin_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     molecular_diffusivity=1e-4,
    ...     longitudinal_dispersivity=1.0,
    ... )
    """
    # Convert to pandas DatetimeIndex if needed
    cin_tedges = pd.DatetimeIndex(cin_tedges)
    tedges = pd.DatetimeIndex(tedges)

    # Convert to arrays
    cout = np.asarray(cout, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    streamline_length = np.asarray(streamline_length, dtype=float)

    # Convert diffusion parameters to arrays and broadcast to pore volumes
    n_pore_volumes = len(aquifer_pore_volumes)
    molecular_diffusivity = np.atleast_1d(np.asarray(molecular_diffusivity, dtype=float))
    longitudinal_dispersivity = np.atleast_1d(np.asarray(longitudinal_dispersivity, dtype=float))

    # Broadcast scalar values to match pore volumes
    if molecular_diffusivity.size == 1:
        molecular_diffusivity = np.broadcast_to(molecular_diffusivity, (n_pore_volumes,)).copy()
    if longitudinal_dispersivity.size == 1:
        longitudinal_dispersivity = np.broadcast_to(longitudinal_dispersivity, (n_pore_volumes,)).copy()

    # Input validation
    if len(tedges) != len(cout) + 1:
        msg = "tedges must have one more element than cout"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if len(aquifer_pore_volumes) != len(streamline_length):
        msg = "aquifer_pore_volumes and streamline_length must have the same length"
        raise ValueError(msg)
    if len(molecular_diffusivity) != n_pore_volumes:
        msg = "molecular_diffusivity must be a scalar or have same length as aquifer_pore_volumes"
        raise ValueError(msg)
    if len(longitudinal_dispersivity) != n_pore_volumes:
        msg = "longitudinal_dispersivity must be a scalar or have same length as aquifer_pore_volumes"
        raise ValueError(msg)
    if np.any(molecular_diffusivity < 0):
        msg = "molecular_diffusivity must be non-negative"
        raise ValueError(msg)
    if np.any(longitudinal_dispersivity < 0):
        msg = "longitudinal_dispersivity must be non-negative"
        raise ValueError(msg)
    if np.any(np.isnan(cout)):
        msg = "cout contains NaN values, which are not allowed"
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

    # Check for conflicting approaches: multiple pore volumes with longitudinal dispersivity
    if n_pore_volumes > 1 and np.any(longitudinal_dispersivity > 0) and not suppress_dispersion_warning:
        msg = (
            "Using multiple aquifer_pore_volumes with non-zero longitudinal_dispersivity. "
            "Multiple pore volumes represent macro-scale aquifer heterogeneity, while "
            "longitudinal_dispersivity represents pore-scale mechanical dispersion. "
            "Using both may double-count spreading effects. Consider:\n"
            "  1. Use multiple pore volumes with longitudinal_dispersivity=0 for macro-scale heterogeneity\n"
            "  2. Use a single pore volume with longitudinal_dispersivity>0 for pore-scale dispersion\n"
            "See examples/05_Diffusion_Dispersion.ipynb for guidance on selecting the appropriate method.\n"
            "Suppress this warning with suppress_dispersion_warning=True."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    # Extend tedges for spin up (same as forward function)
    tedges = pd.DatetimeIndex([
        tedges[0] - pd.Timedelta("36500D"),
        *list(tedges[1:-1]),
        tedges[-1] + pd.Timedelta("36500D"),
    ])

    # Compute the cumulative flow at tedges (extraction times)
    extraction_volume = flow * (np.diff(tedges) / pd.Timedelta("1D"))  # m3
    cumulative_volume_at_cout_tedges = np.concatenate(([0], np.cumsum(extraction_volume)))

    # Compute the cumulative flow at cin_tedges (interpolated)
    cumulative_volume_at_cin_tedges = np.interp(cin_tedges, tedges, cumulative_volume_at_cout_tedges)

    # Compute residence times at cin_tedges using forward direction
    # (from infiltration perspective: when will water infiltrated at cin_tedges be extracted?)
    rt_edges_2d = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="infiltration_to_extraction",
    )

    # Determine valid cin bins immediately (reuse rt_edges_2d, no duplicate computation)
    # A cin bin is valid if both its edges have finite residence times across all pore volumes
    valid_cin_bins = ~np.any(np.isnan(rt_edges_2d[:, :-1]) | np.isnan(rt_edges_2d[:, 1:]), axis=0)

    # Compute mean residence time per (pore_volume, cin_bin) for velocity calculation
    # Shape: (n_pore_volumes, n_cin_bins)
    rt_mean_2d = residence_time_mean(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        direction="infiltration_to_extraction",
        retardation_factor=retardation_factor,
    )

    # Compute velocity: v = L / tau_mean
    # Shape: (n_pore_volumes, n_cin_bins)
    # Use explicit validity handling instead of error suppression
    valid_rt = np.isfinite(rt_mean_2d) & (rt_mean_2d > 0)
    velocity_2d = np.where(valid_rt, streamline_length[:, None] / rt_mean_2d, 0.0)

    # Compute D_L = D_m + alpha_L * v
    # Shape: (n_pore_volumes, n_cin_bins)
    # Fall back to molecular diffusivity only where RT is invalid (spinup period)
    diffusivity_2d = np.where(
        valid_rt,
        molecular_diffusivity[:, None] + longitudinal_dispersivity[:, None] * velocity_2d,
        molecular_diffusivity[:, None],
    )

    # Initialize coefficient matrix accumulator
    n_cin_bins = len(cin_tedges) - 1
    n_cout_bins = len(cout)
    n_cout_edges = len(tedges)
    accumulated_coeff = np.zeros((n_cin_bins, n_cout_bins))

    # At cin_tedges, check if water has been extracted yet (looking forward in time)
    # Water infiltrated at cin_tedges will be extracted at cin_tedges + rt
    # It can be observed if that extraction time falls within tedges

    # Loop over each pore volume
    for i_pv in range(len(aquifer_pore_volumes)):
        # Compute extraction times at cin_tedges for this pore volume
        # t_extraction = t_infiltration + residence_time
        extraction_times_at_cin_tedges = cin_tedges + pd.to_timedelta(rt_edges_2d[i_pv, :], unit="D")

        # Convert to days relative to tedges[0] for computation
        cout_tedges_days = ((tedges - tedges[0]) / pd.Timedelta("1D")).values
        extraction_times_days = ((extraction_times_at_cin_tedges - tedges[0]) / pd.Timedelta("1D")).values

        # The spatial coordinate x represents the distance from the concentration front
        # to the observation point. For the inverse problem:
        # We need to find what fraction of cout[j] originated from cin[i]
        #
        # For each (cin_edge, cout_edge) pair, compute the volume relationship:
        # delta_volume = volume at cin_tedge + R*APV - volume at cout_tedge
        # This represents how far the concentration front has traveled
        delta_volume_for_extraction = (
            cumulative_volume_at_cin_tedges[:, None]
            + (retardation_factor * aquifer_pore_volumes[i_pv])
            - cumulative_volume_at_cout_tedges[None, :]
        )

        # Mask positions where cin_tedge + RT < cout_tedge (extraction hasn't happened yet)
        # The extraction time for water at cin_tedge[i] is extraction_times_at_cin_tedges[i]
        # This should be >= cout_tedge[j] for cout[j] to contain this water
        isactive = extraction_times_days[:, None] >= cout_tedges_days[None, :]
        delta_volume_for_extraction[~isactive] = np.nan

        # Convert volume to distances (x-coordinate for erf)
        step_widths_cout = (
            delta_volume_for_extraction / (retardation_factor * aquifer_pore_volumes[i_pv]) * streamline_length[i_pv]
        )

        # Compute time_active: time from infiltration until the cout_edge is observed
        # This is limited by the residence time (can't observe before extraction)
        time_active = extraction_times_days[:, None] - cout_tedges_days[None, :]
        time_active[~isactive] = np.nan
        time_active = np.maximum(time_active, 0)  # No negative times
        time_active = np.minimum(time_active, rt_edges_2d[[i_pv]].T)  # Limited by residence time

        # Compute erf response for each cout_edge
        # We need to compute response[i_cin_bin, j_cout_edge] for each cout edge
        # This is similar to the forward function but with transposed structure
        #
        # For each cout_edge j, we compute the erf mean over cin bins
        # xedges are at cin_tedges (n_cin_edges), giving n_cin_bins cells
        # tedges are at cin_tedges (n_cin_edges), giving n_cin_bins cells

        # Get diffusivity for this pore volume across all cin bins
        diff_pv = diffusivity_2d[i_pv, :]  # shape (n_cin_bins,)

        # response[i_cin_bin, j_cout_edge] = mean erf for cin bin i at cout edge j
        response = np.zeros((n_cin_bins, n_cout_edges))

        for j in range(n_cout_edges):
            # For cout_edge j, extract the x and t values at each cin_edge
            xedges_j = step_widths_cout[:, j]  # shape (n_cin_edges,)
            tedges_j = time_active[:, j]  # shape (n_cin_edges,)

            # _erf_mean_space_time takes edge arrays and returns cell averages
            # Input: n edges, Output: n-1 cell averages
            response[:, j] = _erf_mean_space_time(xedges_j, tedges_j, diff_pv)

        # Convert erf response [-1, 1] to breakthrough fraction [0, 1]
        frac = 0.5 * (1 + response)  # shape (n_cin_bins, n_cout_edges)

        # Coefficient matrix: coeff[i, j] = frac[i, j] - frac[i, j+1]
        # This represents the fraction of cout[j] that originated from cin[i]
        # frac[:, j] is the breakthrough at cout_tedge[j]
        # For cout[j] (between tedges[j] and tedges[j+1]), contribution is frac at start minus frac at end
        frac_start = frac[:, :-1]  # frac at tedges[j] for j=0..n-1
        frac_end = frac[:, 1:]  # frac at tedges[j+1] for j=0..n-1

        # Handle NaN: if frac_end is NaN but frac_start is valid, use 0.0 for frac_end
        frac_end_filled = np.where(np.isnan(frac_end) & ~np.isnan(frac_start), 0.0, frac_end)
        coeff = frac_start - frac_end_filled  # shape (n_cin_bins, n_cout_bins)

        # Replace NaN with 0 BEFORE accumulating to avoid NaN propagation
        # (NaN + valid = NaN, which would lose valid contributions from other pore volumes)
        accumulated_coeff += np.nan_to_num(coeff, nan=0.0)

    # Average across pore volumes
    coeff_matrix = accumulated_coeff / len(aquifer_pore_volumes)

    # coeff_matrix is already NaN-free due to nan_to_num in the loop
    coeff_matrix_filled = coeff_matrix

    # Matrix multiply: cin = coeff_matrix @ cout
    cin = coeff_matrix_filled @ cout

    # Handle invalid outputs where no valid contributions exist
    # A cin bin is invalid when:
    # 1. The sum of coefficients is near zero (no cout data covers this infiltration time)
    # 2. The infiltration bin extends beyond the extraction data range
    total_coeff = np.sum(coeff_matrix_filled, axis=1)

    # Use pre-computed valid_cin_bins (computed early from rt_edges_2d)
    no_valid_contribution = (total_coeff < EPSILON_COEFF_SUM) | ~valid_cin_bins
    cin[no_valid_contribution] = np.nan

    return cin
