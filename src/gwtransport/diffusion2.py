"""
Analytical solutions for 1D advection-dispersion transport.

This module implements analytical solutions for solute transport in 1D aquifer
systems, combining advection with longitudinal dispersion. The solutions are
based on the error function (erf) and its integrals.

Key functions:
- infiltration_to_extraction: Main transport function combining advection and dispersion
- erf_mean_space_time: Analytical space-time averaging of error function
- erf_integral_space_time: Double integral of error function

The dispersion is characterized by the longitudinal dispersion coefficient D_L,
which the user should compute as:

    D_L = D_m + alpha_L * v

where:
- D_m is the molecular diffusion coefficient [m^2/day]
- alpha_L is the longitudinal dispersivity [m]
- v is the pore velocity [m/day]
"""

import itertools

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import NDArray
from scipy import special
from scipy.integrate import dblquad, quad_vec

from gwtransport.residence_time import residence_time


def analytical_diffusion_filter(
    input_signal: NDArray[np.float64],
    xedges: NDArray[np.float64],
    diffusivity: float,
    *,
    times: NDArray[np.float64] = None,
    times_in: pd.DatetimeIndex = None,
    times_out: pd.DatetimeIndex = None,
) -> NDArray[np.float64]:
    """Apply analytical diffusion solution with position-dependent diffusivity.

    This function implements the analytical solution to the diffusion equation using
    error functions. The solution is computed by decomposing the initial signal into
    a series of steps and summing their individual diffusion solutions.

    Parameters
    ----------
    input_signal : ndarray, shape (n,)
        Initial concentration/temperature profile.
    xedges : ndarray, shape (n+1,)
        Cell edge positions [m]. Must be monotonically increasing.
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.
    times : ndarray, shape (n+1,)
        Residence times of the xedges during which they endure diffusion [days].

    Returns
    -------
    ndarray, shape (n,)
        Diffused signal at the query points.

    Notes
    -----
    The analytical solution for diffusion of a step function is:
    T(x,t) = T₀ * (erf((x-x₀)/(2√(Dt))) - erf((x-x₁)/(2√(Dt))))
    where T₀ is the initial temperature, x₀ and x₁ are the step boundaries

    The solution is computed by decomposing the initial signal into a series of steps
    and summing their individual diffusion solutions.

    The error function mean between two edges is computed using the erf_mean function.

    Raises
    ------
    ValueError
        If the xedges are not monotonically increasing, if the diffusivity is not positive,
        or if the input signal has a length different from n.
    """
    # Check inputs
    if not np.all(np.diff(xedges) > 0):
        msg = "xedges must be monotonically increasing."
        raise ValueError(msg)
    if diffusivity < 0:
        msg = "Diffusivity must be positive."
        raise ValueError(msg)
    if len(input_signal) != len(xedges) - 1:
        msg = "Input signal must have length n."
        raise ValueError(msg)

    # Check times
    if times is not None:
        times = np.full_like(xedges, times)[:, None]
    else:
        if times_in is None or times_out is None:
            msg = "Either times or times_in and times_out must be provided."
            raise ValueError(msg)
        if len(times_in) != len(xedges) - 1:
            msg = "times_in must have length n."
            raise ValueError(msg)
        if len(times_out) != len(xedges) - 1:
            msg = "times_out must have length n."
            raise ValueError(msg)

        times_in = np.full_like(xedges, times_in)
        times_out = np.full_like(xedges, times_out)

        times = times_out[:, None] - times_in[None, 1:-1]
        if not np.all(times >= 0):
            msg = "times_out must be greater than times_in, or clip to zero"
            raise ValueError(msg)

    # Op het moment dat ik een xedge (axis=0) onttrek, wil ik weten welke steps(axis=1) hebben er hoelang invloed gehad op het punt van onttrekking
    delta_input_signal = input_signal[1:] - input_signal[:-1]

    # Locations with respect to the cell edges, shape (n_edges, n_steps)
    xloc = xedges[:, None] - xedges[None, 1:-1]

    # Compute the diffusion between the edges
    with np.errstate(divide="ignore", invalid="ignore"):
        # division by zero should result in -inf and +inf without warnings
        arg = np.where(np.isclose(xloc, 0.0), 0, xloc / (2 * np.sqrt(diffusivity * times)))

    translate_array = delta_input_signal[None, :] + delta_input_signal[None, :] * erf_mean_space(arg) / 2
    return np.sum(translate_array, axis=1)


def erf_integral_space(x: NDArray[np.float64], a: float = 1.0, clip_to_inf=6.0) -> NDArray[np.float64]:
    """Compute the integral of the error function.

    This function computes the integral of the error function from 0 to x.

    Parameters
    ----------
    x : ndarray
        Input values.
    a : float, optional
        a = 1/(2*sqrt(diffusivity*t)). Default is 1.
    clip_to_inf : float, optional
        Clip the input values to -clip_to_inf and clip_to_inf to avoid numerical issues.
        Default is 6.

    Returns
    -------
    ndarray
        Integral of the error function from 0 to x.

    TODO: a may by a vector
    """
    a = float(a)
    ax = a * x

    out = np.zeros_like(x, dtype=float)

    if a == 0.0:
        return out

    # Fill in the limits of the error function
    maskl, masku = ax <= -clip_to_inf, ax >= clip_to_inf
    out[maskl] = -x[maskl] - 1 / (a * np.sqrt(np.pi))
    out[masku] = x[masku] - 1 / (a * np.sqrt(np.pi))

    # Fill in the rest of the values
    mask = np.logical_and(~maskl, ~masku)
    out[mask] = x[mask] * special.erf(ax[mask]) + (np.exp(-(ax[mask] ** 2)) - 1) / (a * np.sqrt(np.pi))
    return out


def erf_integral_numerical_space(x: NDArray[np.float64], a: float = 1.0) -> NDArray[np.float64]:
    r"""Compute the integral of the error function numerically.

    y = \int_0^x erf(a * t) dt

    This function computes the integral of the error function from 0 to x numerically and
    is slower than the erf_integral function.

    Parameters
    ----------
    x : ndarray
        Input values.
    a : float, optional
        Scaling factor for the error function. Default is
        1.

    Returns
    -------
    ndarray
        Integral of the error function from 0 to x.
    """
    shape = np.shape(x)
    isnan = np.isnan(x)
    x = x.copy()
    x[isnan] = 0.0

    a = float(a)

    out = np.array([quad_vec(lambda d, a=a: special.erf(a * d), 0, xi)[0] for xi in x.ravel()]).reshape(shape)

    # The infinity limits are not handled well by quad
    out[np.isinf(x)] = np.inf
    out[isnan] = np.nan
    return out


def erf_mean_space(edges: NDArray[np.float64], a: float = 1.0) -> NDArray[np.float64]:
    """Compute the mean of the error function between edges.

    This function computes the mean of the error function between two bounds. Provides an
    alternative to computing directly the error function at the cell node. This alternative
    conserves the mass of the signal.

    Parameters
    ----------
    edges : ndarray
        Cell edges of size n.
    a : float, optional
        Scaling factor for the error function. Default is
        1.

    Returns
    -------
    ndarray
        Mean of the error function between the bounds of size n - 1.
    """
    _edges = np.clip(edges, -1e6, 1e6)
    _erfint = erf_integral_space(_edges, a=a)
    dx = _edges[1:] - _edges[:-1]

    out = np.where(dx != 0.0, (_erfint[1:] - _erfint[:-1]) / dx, np.nan)
    out[dx == 0.0] = special.erf(edges[:-1][dx == 0.0])

    # Handle the case where the edges are far from the origin and have a known outcome
    ue, le = edges[1:], edges[:-1]  # upper and lower cell edges
    out[np.isinf(le) & ~np.isinf(ue)] = -1.0
    out[np.isneginf(le) & np.isneginf(ue)] = -1.0
    out[~np.isinf(le) & np.isinf(ue)] = 1.0
    out[np.isposinf(le) & np.isposinf(ue)] = 1.0
    out[np.isneginf(le) & np.isposinf(ue)] = 0.0
    out[np.isposinf(le) & np.isneginf(ue)] = 0.0
    return out


def erf_mean_numerical_space(edges: NDArray[np.float64], a: float = 1.0) -> NDArray[np.float64]:
    """Compute the mean of the error function between edges numerically.

    This function computes the mean of the error function between two bounds numerically and
    is slower than the erf_mean function.

    Parameters
    ----------
    edges : ndarray
        Cell edges of size n.
    a : float, optional
        Scaling factor for the error function. Default is
        1.

    Returns
    -------
    ndarray
        Mean of the error function between the bounds of size n - 1.
    """
    shape = edges[:-1].shape
    out = np.array([
        quad_vec(lambda d, a=a: special.erf(a * d), el, er)[0] / (er - el)
        for el, er in itertools.pairwise(edges.ravel())
    ]).reshape(shape)

    # Handle the case where the edges are far from the origin and have a known outcome
    ue, le = edges[1:], edges[:-1]  # upper and lower cell edges
    out[np.isinf(le) & ~np.isinf(ue)] = -1.0
    out[np.isneginf(le) & np.isneginf(ue)] = -1.0
    out[~np.isinf(le) & np.isinf(ue)] = 1.0
    out[np.isposinf(le) & np.isposinf(ue)] = 1.0
    out[np.isneginf(le) & np.isposinf(ue)] = 0.0
    out[np.isposinf(le) & np.isneginf(ue)] = 0.0
    return out


def _erf_integral_space_time_pointwise(x, t, diffusivity):
    """
    Compute the integral of the error function in space and time at (x, t) points.

    Unlike erf_integral_space_time which uses meshgrid, this function evaluates
    F(x[i], t[i]) for each i, where x and t have the same shape. This is useful
    for batched computations where we need F at arbitrary (x, t) pairs.

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


def erf_integral_space_time(x, t, diffusivity):
    """
    Compute the integral of the error function in space and time.

    This function computes the integral of the error function in space and time.

    Parameters
    ----------
    x : ndarray
        Input values in space.
    t : ndarray
        Input values in time.
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.

    Returns
    -------
    ndarray
        Integral of the error function in space and time.
    """
    xarray, tarray = np.meshgrid(x, t, sparse=True)
    isnan = np.isnan(xarray) | np.isnan(tarray)

    sqrt_diffusivity = np.sqrt(diffusivity)
    sqrt_t = np.sqrt(tarray)
    sqrt_pi = np.sqrt(np.pi)
    exp_term = np.exp(-(xarray**2) / (4 * diffusivity * tarray))
    erf_term = special.erf(xarray / (2 * sqrt_diffusivity * sqrt_t))

    term1 = -4 * sqrt_diffusivity * tarray ** (3 / 2) / (3 * sqrt_pi)
    term2 = (2 * sqrt_diffusivity / sqrt_pi) * (
        (2 * tarray ** (3 / 2) * exp_term / 3)
        - (sqrt_t * xarray**2 * exp_term / (3 * diffusivity))
        - (sqrt_pi * xarray**3 * erf_term / (6 * diffusivity ** (3 / 2)))
    )
    term3 = xarray * (
        tarray * erf_term
        + (xarray**2 * erf_term / (2 * diffusivity))
        + (sqrt_t * xarray * exp_term / (sqrt_pi * sqrt_diffusivity))
    )
    term4 = -(xarray**3) / (6 * diffusivity)
    out = term1 + term2 + term3 + term4

    out = np.where(isnan, np.nan, out)
    # out = np.where(tarray <= 0.0 and x > 0.0, x, out)
    # out = np.where(tarray <= 0.0 and x < 0.0, -1.0, out)

    # out[maskl] = -x[maskl] - 1 / (a * np.sqrt(np.pi))
    # out[masku] = x[masku] - 1 / (a * np.sqrt(np.pi))

    # if x == 0.0:
    #     return 0.0
    # if np.isposinf(x) or (x > 0.0 and t <= 0.0):
    #     return 1.0
    # if np.isneginf(x) or (x < 0.0 and t <= 0.0):
    #     return -1.0

    out = np.where(tarray <= 0.0, 0.0, out)
    result = np.where(np.isinf(xarray) | np.isinf(tarray), np.inf, out)

    if np.size(x) == 1 and np.size(t) == 1:
        return result[0, 0]
    if np.size(x) == 1 and np.size(t) != 1:
        return result[:, 0]
    if np.size(x) != 1 and np.size(t) == 1:
        return result[0, :]
    return result


# def erf_integral_space_time2(x, t, diffusivity):
#     """
#     Calculates the analytical solution to the double integral.

#     ∫_0^t ∫_0^x erf(ξ/(2*sqrt(diffusivity*ti))) dξ dti.

#     Parameters
#     ----------
#     x : float or numpy.ndarray
#         Upper limit of the inner integral
#     t : float or numpy.ndarray
#         Upper limit of the outer integral
#     diffusivity : float
#         Diffusivity constant (must be positive)

#     Returns
#     -------
#     float or numpy.ndarray
#         The value of the double integral
#     """
#     # Input validation
#     if diffusivity <= 0:
#         raise ValueError("diffusivity must be positive")

#     # For x=0, the inner integral is always 0, so the double integral is 0
#     if np.isscalar(x) and x == 0:
#         return np.zeros_like(t) if isinstance(t, np.ndarray) else 0.0

#     # For t=0, the outer integral bounds are the same, so the result is 0
#     if np.isscalar(t) and t == 0:
#         return np.zeros_like(x) if isinstance(x, np.ndarray) else 0.0

#     # Calculate each term of the analytical solution
#     term1 = x * t

#     term2 = -4 * np.sqrt(diffusivity) * t ** (3 / 2) / (3 * np.sqrt(np.pi))

#     term3 = (x * np.sqrt(t)) / (np.sqrt(np.pi * diffusivity)) * (np.exp(-(x**2) / (4 * diffusivity * t)) - 1)

#     term4 = (np.sqrt(t) - x / np.sqrt(np.pi)) * special.erf(x / (2 * np.sqrt(diffusivity * t)))

#     return term1 + term2 + term3 + term4


def erf_integral_numerical_space_time2(x, t, diffusivity):
    """
    Numerical solution of the double integral using scipy's dblquad.

    Parameters
    ----------
    x : float or array-like
        Upper limit of the inner integral
    t : float or array-like
        Upper limit of the outer integral
    diffusivity : float
        Diffusivity constant (D)

    Returns
    -------
    result : float or ndarray
        Result of the double integral
    """

    # Define the integrand function for numerical integration
    def integrand(t, x):
        if x == 0.0:
            return 0.0
        if np.isposinf(x) or (x > 0.0 and t <= 0.0):
            return 1.0
        if np.isneginf(x) or (x < 0.0 and t <= 0.0):
            return -1.0
        return special.erf(x / (2 * np.sqrt(diffusivity * t)))

    # Convert to arrays
    x_vals = np.atleast_1d(x)
    t_vals = np.atleast_1d(t)

    # Initialize result array
    result = np.zeros((len(t_vals), len(x_vals)))

    # Calculate for each x,t pair
    for i, t_val in enumerate(t_vals):
        for j, x_val in enumerate(x_vals):
            result[i, j], _ = dblquad(
                integrand,
                0.0,
                x_val,
                0.0,
                t_val,
            )

    result = np.where(np.isinf(x_vals[None, :]), np.inf, result)
    result = np.where(np.isnan(x_vals[None, :]) | np.isnan(t_vals[:, None]), np.nan, result)

    # Return scalar if both inputs were scalar
    if len(x_vals) == 1 and len(t_vals) == 1:
        return result[0, 0]
    if len(x_vals) == 1 and len(t_vals) != 1:
        return result[:, 0]
    if len(x_vals) != 1 and len(t_vals) == 1:
        return result[0, :]
    return result


def erf_mean_space_time(xedges, tedges, diffusivity):
    """
    Compute the mean of the error function over space-time cells.

    Computes the average value of erf(x/(2*sqrt(D*t))) over rectangular cells
    defined by xedges and tedges using the analytical double integral solution.

    The mean is computed as:
        (F(x₁,t₁) - F(x₀,t₁) - F(x₁,t₀) + F(x₀,t₀)) / ((x₁-x₀)(t₁-t₀))

    where F(x,t) = ∫₀ᵗ ∫₀ˣ erf(ξ/(2√(D·τ))) dξ dτ

    Parameters
    ----------
    xedges : ndarray
        Cell edges in space of size (n_x + 1,).
    tedges : ndarray
        Cell edges in time of size (n_t + 1,).
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.

    Returns
    -------
    ndarray
        Mean of the error function over each cell. Shape is (n_t, n_x).
        Returns scalar if both n_t=1 and n_x=1.
        Returns 1D array if either n_t=1 or n_x=1.
    """
    # Ensure arrays
    xedges = np.asarray(xedges)
    tedges = np.asarray(tedges)

    # Clip extreme values to avoid numerical issues
    _xedges = np.clip(xedges, -1e6, 1e6)

    # Compute the double integral at all edge combinations
    # erf_integral_space_time returns shape (n_t, n_x) for inputs of size n_t and n_x
    _erfint = erf_integral_space_time(_xedges, tedges, diffusivity)

    # Use inclusion-exclusion to get integral over each cell
    # I[i,j] = F(x_{j+1}, t_{i+1}) - F(x_j, t_{i+1}) - F(x_{j+1}, t_i) + F(x_j, t_i)
    # _erfint has shape (n_tedges, n_xedges), we want (n_t_cells, n_x_cells)
    double_integrals = _erfint[1:, 1:] - _erfint[:-1, 1:] - _erfint[1:, :-1] + _erfint[:-1, :-1]

    # Calculate cell areas
    dt = np.diff(tedges)[:, np.newaxis]
    dx = np.diff(_xedges)[np.newaxis, :]
    cell_areas = dx * dt

    # Compute averages
    with np.errstate(divide="ignore", invalid="ignore"):
        out = double_integrals / cell_areas

    # Handle zero-area cells (dt=0 or dx=0) - these are instantaneous or point evaluations
    # For dt=0: need to evaluate mean over space at fixed time
    # For dx=0: need to evaluate mean over time at fixed space
    # For now, leave as NaN (consistent with erf_mean_space_time2 behavior)

    # Handle the case where x edges are at infinity (known asymptotic values)
    ue, le = xedges[1:], xedges[:-1]  # upper and lower cell edges in x
    # Broadcasting: ue and le have shape (n_x_cells,), out has shape (n_t_cells, n_x_cells)
    out[:, np.isinf(le) & ~np.isinf(ue)] = -1.0
    out[:, np.isneginf(le) & np.isneginf(ue)] = -1.0
    out[:, ~np.isinf(le) & np.isinf(ue)] = 1.0
    out[:, np.isposinf(le) & np.isposinf(ue)] = 1.0
    out[:, np.isneginf(le) & np.isposinf(ue)] = 0.0
    out[:, np.isposinf(le) & np.isneginf(ue)] = 0.0

    # Return appropriate shape
    n_t_cells, n_x_cells = out.shape
    if n_x_cells == 1 and n_t_cells == 1:
        return out[0, 0]
    if n_x_cells == 1 and n_t_cells != 1:
        return out[:, 0]
    if n_x_cells != 1 and n_t_cells == 1:
        return out[0, :]
    return out


def erf_mean_space_time2(xedges, tedges, diffusivity):
    """Vectorized computation of cell averages using a single analytical_solution call."""
    # Ensure arrays
    xedges = np.asarray(xedges)
    tedges = np.asarray(tedges)

    # Calculate integral for all edge combinations at once
    integral = erf_integral_numerical_space_time2(xedges, tedges, diffusivity)

    # Use inclusion-exclusion to calculate integrals for all cells
    # I[i,j] - I[i-1,j] - I[i,j-1] + I[i-1,j-1]
    double_integrals = integral[1:, 1:] - integral[:-1, 1:] - integral[1:, :-1] + integral[:-1, :-1]

    # Calculate cell areas
    dt = np.diff(tedges)[:, np.newaxis]
    dx = np.diff(xedges)[np.newaxis, :]
    cell_areas = dx * dt

    # Return averages
    averages = double_integrals / cell_areas

    # Handle dt == 0 case
    if (dt == 0.0).any():
        # ts = tedges[:-1][dt == 0.0]
        # aa = [1 / (2 * np.sqrt(diffusivity * t)) for t in ts]
        # averages[dt == 0.0] = [erf_mean_space(xedges, a) for a in aa]
        msg = "dt == 0.0 case not implemented yet"
        raise NotImplementedError(msg)

    # Handle dx == 0 case
    if (dx == 0.0).any():
        # xs = xedges[:-1][dx == 0.0]
        # bb = []
        # averages[dx == 0.0] = [erf_mean_time(tedges, b) for b in bb]
        msg = "dx == 0.0 case not implemented yet"
        raise NotImplementedError(msg)

    n_t_cells, n_x_cells = averages.shape

    if n_x_cells == 1 and n_t_cells == 1:
        return averages[0, 0]
    if n_x_cells == 1 and n_t_cells != 1:
        return averages[:, 0]
    if n_x_cells != 1 and n_t_cells == 1:
        return averages[0, :]
    return averages


def erf_mean_numerical_space_time2(xedges, tedges, diffusivity):
    """
    Compute average of erf(x/(2*sqrt(diffusivity*t))) for grid cells defined by edge coordinates.

    Parameters
    ----------
    x_edges : array-like
        Monotonically increasing x-coordinate edges (n+1 values for n cells)
        e.g., [x₀, x₁, x₂, ..., xₙ] where each cell i spans [xᵢ, xᵢ₊₁]
    t_edges : array-like
        Monotonically increasing t-coordinate edges (m+1 values for m cells)
        e.g., [t₀, t₁, t₂, ..., tₘ] where each cell j spans [tⱼ, tⱼ₊₁]
    diffusivity : float
        Diffusivity constant

    Returns
    -------
    averages : ndarray
        2D array (nxm) containing average values for each cell
    """
    # Ensure arrays
    xedges = np.asarray(xedges)
    tedges = np.asarray(tedges)

    # Get cell counts
    n_x_cells = len(xedges) - 1
    n_t_cells = len(tedges) - 1

    # Initialize output array
    # averages = np.zeros((n_x_cells, n_t_cells))
    averages = np.zeros((n_t_cells, n_x_cells))

    # Define the integrand function for numerical integration
    def integrand(t, x):
        if x == 0.0:
            return 0.0
        if np.isposinf(x) or (x > 0.0 and t <= 0.0):
            return 1.0
        if np.isneginf(x) or (x < 0.0 and t <= 0.0):
            return -1.0
        return special.erf(x / (2 * np.sqrt(diffusivity * t)))

    # Compute average for each cell
    for i in range(n_t_cells):
        for j in range(n_x_cells):
            t1, t2 = tedges[i], tedges[i + 1]
            x1, x2 = xedges[j], xedges[j + 1]

            # Skip empty cells
            if x1 == x2 and t1 != t2:
                averages[i, j] = quad_vec(integrand, t1, t2, args=(x1,))[0] / (t2 - t1)
                continue
            if t1 == t2 and x1 != x2:
                averages[i, j] = quad_vec(lambda x, t: integrand(t, x), x1, x2, args=(t1,))[0] / (x2 - x1)
                continue
            if x1 == x2 and t1 == t2:
                averages[i, j] = integrand(t1, x1)
                continue

            # Direct numerical integration with dblquad
            result, _ = dblquad(
                integrand,
                x1,
                x2,
                t1,
                t2,
            )

            # Calculate average
            domain_area = (x2 - x1) * (t2 - t1)
            averages[i, j] = result / domain_area

    # Return scalar if both inputs were scalar
    if n_x_cells == 1 and n_t_cells == 1:
        return averages[0, 0]
    if n_x_cells == 1 and n_t_cells != 1:
        return averages[:, 0]
    if n_x_cells != 1 and n_t_cells == 1:
        return averages[0, :]
    return averages


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    travel_distances: npt.ArrayLike,
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
    and v is pore velocity [m/day].

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
    travel_distances : array-like
        Array of travel distances [m] corresponding to each pore volume.
        Must have the same length as aquifer_pore_volumes. The travel distance
        determines the dispersion length scale: sqrt(D_L * tau).
    diffusivity : float
        Longitudinal dispersion coefficient [m2/day]. Must be non-negative.
        Compute as D_L = D_m + alpha_L * v where:
        - D_m: molecular diffusion coefficient [m2/day]
        - alpha_L: longitudinal dispersivity [m]
        - v: pore velocity [m/day]
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
        or if aquifer_pore_volumes and travel_distances have different lengths.

    See Also
    --------
    gwtransport.advection.infiltration_to_extraction : Pure advection (no dispersion)
    erf_mean_space_time : Analytical space-time averaging of error function

    Notes
    -----
    The algorithm works as follows:

    1. For each output time bin [t_out_start, t_out_end]:
       - Compute the residence time for each pore volume
       - Determine which infiltration times contribute to this output bin

    2. For each input concentration step (change in cin):
       - The step diffuses as it travels through the aquifer
       - The diffused contribution is computed using the error function
       - Time-averaging over the output bin uses erf_mean_space_time

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
    >>> travel_distances = np.array([100.0])
    >>>
    >>> # Compute with dispersion
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     travel_distances=travel_distances,
    ...     diffusivity=1.0,  # m2/day
    ... )

    With multiple pore volumes (heterogeneous aquifer):

    >>> # Distribution of pore volumes and corresponding travel distances
    >>> aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
    >>> travel_distances = np.array([80.0, 100.0, 120.0])
    >>>
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     travel_distances=travel_distances,
    ...     diffusivity=1.0,
    ... )
    """
    # Convert to pandas DatetimeIndex if needed
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    # Convert to arrays
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    travel_distances = np.asarray(travel_distances, dtype=float)

    # Input validation
    if len(tedges) != len(cin) + 1:
        msg = "tedges must have one more element than cin"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if len(aquifer_pore_volumes) != len(travel_distances):
        msg = "aquifer_pore_volumes and travel_distances must have the same length"
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
    if np.any(travel_distances <= 0):
        msg = "travel_distances must be positive"
        raise ValueError(msg)

    # Handle zero diffusivity case (pure advection)
    if diffusivity == 0.0:
        # Import here to avoid circular dependency at module load time
        from gwtransport.advection import (  # noqa: PLC0415
            infiltration_to_extraction as advection_i2e,
        )

        return advection_i2e(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            retardation_factor=retardation_factor,
        )

    # Convert time edges to days (relative to tedges[0])
    t_ref = tedges[0]
    cin_tedges_days = ((tedges - t_ref) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - t_ref) / pd.Timedelta(days=1)).values

    n_cin = len(cin)
    n_cout = len(cout_tedges) - 1
    n_pv = len(aquifer_pore_volumes)

    # Compute residence times for all pore volumes at all cout_tedges
    # Shape: (n_pv, n_cout_edges)
    rt_edges_2d = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )

    # Initialize output accumulator for each pore volume
    # We'll average across pore volumes at the end
    cout_per_pv = np.full((n_pv, n_cout), np.nan)

    # Decompose input signal into steps at bin edges
    # Step at cin_tedges_days[0]: from 0 to cin[0]  (magnitude = cin[0])
    # Step at cin_tedges_days[j]: from cin[j-1] to cin[j]  (magnitude = cin[j] - cin[j-1])
    # Step at cin_tedges_days[n_cin]: from cin[n_cin-1] to 0  (magnitude = -cin[n_cin-1])
    #
    # Total number of steps = n_cin + 1
    step_times = cin_tedges_days  # Times at which steps occur
    step_magnitudes = np.zeros(n_cin + 1)
    step_magnitudes[0] = cin[0]  # Initial step from 0 to cin[0]
    step_magnitudes[1:n_cin] = cin[1:] - cin[:-1]  # Interior steps
    step_magnitudes[n_cin] = -cin[n_cin - 1]  # Final step back to 0

    # Loop over each pore volume
    for i_pv in range(n_pv):
        travel_distance = travel_distances[i_pv]

        # Get residence times for this pore volume at cout_tedges
        rt_at_cout_edges = rt_edges_2d[i_pv, :]  # Shape: (n_cout_edges,)

        # Check which output bins have valid residence times
        valid_cout_bins = ~(np.isnan(rt_at_cout_edges[:-1]) | np.isnan(rt_at_cout_edges[1:]))

        if not np.any(valid_cout_bins):
            continue

        # ===== VECTORIZED COMPUTATION =====
        #
        # For each output bin i and each step j, we compute:
        #   contribution[i, j] = delta_c[j] * (1 + erf_avg[i, j]) / 2
        #
        # Each step j has its own cell [x_lo, x_hi] x [t_lo, t_hi] in (x, tau) space.
        # The cells are offset diagonally (not a checkerboard grid) because:
        # - tau = t_out - step_time varies per step
        # - x = v * tau - L is linear in tau
        #
        # We cannot use erf_mean_space_time directly because the erf solution
        # erf(x / (2*sqrt(D*t))) is NOT translation-invariant in t.
        # Instead, we compute F at all 4 corners in a single batched call.

        # Output time bin edges
        t_out_start = cout_tedges_days[:-1]  # Shape: (n_cout,)
        t_out_end = cout_tedges_days[1:]  # Shape: (n_cout,)

        # Average residence time for each output bin
        rt_avg = (rt_at_cout_edges[:-1] + rt_at_cout_edges[1:]) / 2.0  # Shape: (n_cout,)

        # Velocity for this pore volume (at average conditions)
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = travel_distance / rt_avg  # Shape: (n_cout,)

        # Initialize contributions array
        contributions = np.zeros((n_cout, n_cin + 1))

        # Process each output bin
        for i_cout in range(n_cout):
            if not valid_cout_bins[i_cout]:
                continue

            v = velocity[i_cout]
            if np.isnan(v):
                continue

            # Tau at output bin edges for each step: tau = t_out - step_time
            tau_at_out_start = t_out_start[i_cout] - step_times  # Shape: (n_steps,)
            tau_at_out_end = t_out_end[i_cout] - step_times  # Shape: (n_steps,)

            # Clip tau values to positive (required for erf solution)
            tau_start_clipped = np.maximum(tau_at_out_start, 1e-10)

            # x values at tau edges: x = v * tau - L
            x_start = v * tau_start_clipped - travel_distance
            x_end = v * tau_at_out_end - travel_distance

            # Find valid steps: entered system, non-zero magnitude, positive time span
            step_entered = tau_at_out_end > 0
            valid_steps = step_entered & (step_magnitudes != 0) & (tau_at_out_end > tau_start_clipped)

            if not np.any(valid_steps):
                continue

            # Ensure x_lo < x_hi
            x_lo = np.minimum(x_start, x_end)
            x_hi = np.maximum(x_start, x_end)

            # Compute F at all four corners for all steps in one call
            n_steps_total = len(step_magnitudes)
            x_all = np.concatenate([x_lo, x_hi, x_lo, x_hi])
            t_all = np.concatenate([tau_start_clipped, tau_start_clipped, tau_at_out_end, tau_at_out_end])

            f_all = _erf_integral_space_time_pointwise(x_all, t_all, diffusivity)

            f00 = f_all[:n_steps_total]
            f10 = f_all[n_steps_total : 2 * n_steps_total]
            f01 = f_all[2 * n_steps_total : 3 * n_steps_total]
            f11 = f_all[3 * n_steps_total :]

            # Inclusion-exclusion for double integral
            double_integral = f11 - f01 - f10 + f00

            # Cell areas
            dx = x_hi - x_lo
            dt = tau_at_out_end - tau_start_clipped
            cell_area = dx * dt

            # Compute erf_avg
            with np.errstate(divide="ignore", invalid="ignore"):
                erf_avg = np.where(cell_area > 0, double_integral / cell_area, 0.0)

            # Contributions: delta_c * (1 + erf_avg) / 2
            step_contrib = step_magnitudes * (1.0 + erf_avg) / 2.0

            # Zero out invalid steps
            step_contrib = np.where(valid_steps, step_contrib, 0.0)

            contributions[i_cout, :] = step_contrib

        # Sum contributions from all steps for each output bin
        cout_per_pv[i_pv, :] = np.sum(contributions, axis=1)

        # Set invalid output bins to NaN
        cout_per_pv[i_pv, ~valid_cout_bins] = np.nan

    # Average across pore volumes (only where we have valid values)
    # Vectorized: compute mean ignoring NaN
    with np.errstate(all="ignore"):
        cout = np.nanmean(cout_per_pv, axis=0)

    # Where all pore volumes are NaN, the result should be NaN (not zero from nanmean)
    all_nan = np.all(np.isnan(cout_per_pv), axis=0)
    cout[all_nan] = np.nan

    return cout
