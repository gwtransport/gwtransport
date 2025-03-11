"""
Implementation of the double integral: ∫₀ᵗ∫₀ˣ erf(ξ/(2√(D·τ))) dξ dτ .

Provides both numerical and analytical solutions.
"""

import itertools

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import special
from scipy.integrate import dblquad, quad_vec


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
        assert np.all(times >= 0), "times_out must be greater than times_in. or clip to zero"

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
        Scaling factor for the error function. Default is 1.
    clip_to_inf : float, optional
        Clip the input values to -clip_to_inf and clip_to_inf to avoid numerical issues.
        Default is 6.

    Returns
    -------
    ndarray
        Integral of the error function from 0 to x.
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

    with np.errstate(divide="ignore", invalid="ignore"):
        out = (_erfint[1:] - _erfint[:-1]) / (_edges[1:] - _edges[:-1])

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
    if np.ndim(t) == 1:
        t = t[:, None]
    out_shape = np.broadcast_shapes(np.shape(x), np.shape(t))
    xarray = np.broadcast_to(x, out_shape)
    tarray = np.broadcast_to(t, out_shape)
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
    out = np.where(tarray <= 0.0, 0.0, out)
    return np.where(np.isinf(xarray) | np.isinf(tarray), np.inf, out)


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
    Compute the mean of the error function in space and time.

    This function computes the mean of the error function in space and time.

    Parameters
    ----------
    xedges : ndarray
        Cell edges in space.
    tedges : ndarray
        Cell edges in time.
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.

    Returns
    -------
    ndarray
        Mean of the error function in space and time.
    """
    _xedges = np.clip(xedges, -1e6, 1e6)
    _erfint = erf_integral_space_time(xedges=_xedges, tedges=tedges, diffusivity=diffusivity)

    with np.errstate(divide="ignore", invalid="ignore"):
        _out = (_erfint[1:] - _erfint[:-1]) / (_xedges[1:] - _xedges[:-1])
        out = (_out[:, 1:] - _out[:, :-1]) / (tedges[1:] - tedges[:-1])

    # Handle the case where the edges are far from the origin and have a known outcome
    ue, le = xedges[1:], xedges[:-1]  # upper and lower cell edges
    out[np.isinf(le) & ~np.isinf(ue)] = -1.0
    out[np.isneginf(le) & np.isneginf(ue)] = -1.0
    out[~np.isinf(le) & np.isinf(ue)] = 1.0
    out[np.isposinf(le) & np.isposinf(ue)] = 1.0
    out[np.isneginf(le) & np.isposinf(ue)] = 0.0
    out[np.isposinf(le) & np.isneginf(ue)] = 0.0
    return out


def erf_mean_space_time2(xedges, tedges, diffusivity):
    """
    Vectorized computation of cell averages using a single analytical_solution call.
    Most efficient for large grids.

    Parameters same as compute_cell_averages.
    """
    from erf_double_integral import analytical_solution

    # Ensure arrays
    xedges = np.asarray(xedges)
    tedges = np.asarray(tedges)

    # Calculate integral for all edge combinations at once
    X, T = np.meshgrid(xedges, tedges, indexing="ij")
    I_all = analytical_solution(X, T, diffusivity)

    # Use inclusion-exclusion to calculate integrals for all cells
    # I[i,j] - I[i-1,j] - I[i,j-1] + I[i-1,j-1]
    double_integrals = I_all[1:, 1:] - I_all[:-1, 1:] - I_all[1:, :-1] + I_all[:-1, :-1]

    # Calculate cell areas
    dx = np.diff(xedges)[:, np.newaxis]
    dt = np.diff(tedges)[np.newaxis, :]
    cell_areas = dx * dt

    # Return averages
    return double_integrals / cell_areas


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
        2D array (n×m) containing average values for each cell
    """
    # Ensure arrays
    xedges = np.asarray(xedges)
    tedges = np.asarray(tedges)

    # Get cell counts
    n_x_cells = len(xedges) - 1
    n_t_cells = len(tedges) - 1

    # Initialize output array
    averages = np.zeros((n_x_cells, n_t_cells))

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
    for i in range(n_x_cells):
        for j in range(n_t_cells):
            x1, x2 = xedges[i], xedges[i + 1]
            t1, t2 = tedges[j], tedges[j + 1]

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
