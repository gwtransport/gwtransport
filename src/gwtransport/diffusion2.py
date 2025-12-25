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
- v is the pore velocity [m/day]
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import NDArray
from scipy import special

from gwtransport.residence_time import residence_time

# Numerical tolerance for coefficient sum to determine valid output bins
EPSILON_COEFF_SUM = 1e-10


def _erf_integral_space(
    x: NDArray[np.float64],
    a: float | NDArray[np.float64] = 1.0,
    clip_to_inf: float = 6.0,
) -> NDArray[np.float64]:
    """Compute the integral of the error function.

    This function computes the integral of the error function from 0 to x.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Input values.
    a : float or ndarray, shape (m,), optional
        a = 1/(2*sqrt(diffusivity*t)). Default is 1.
        If array, output will have shape (m, n).
    clip_to_inf : float, optional
        Clip the input values to -clip_to_inf and clip_to_inf to avoid numerical issues.
        Default is 6.

    Returns
    -------
    ndarray
        Integral of the error function from 0 to x.
        Shape is (n,) if a is scalar, (m, n) if a is array of shape (m,).
    """
    x = np.asarray(x)
    a = np.asarray(a)
    a_is_scalar = a.ndim == 0

    if a_is_scalar:
        a_val = float(a)
        ax = a_val * x
        out = np.zeros_like(x, dtype=float)

        if a_val == 0.0:
            return out

        # Fill in the limits of the error function
        maskl, masku = ax <= -clip_to_inf, ax >= clip_to_inf
        out[maskl] = -x[maskl] - 1 / (a_val * np.sqrt(np.pi))
        out[masku] = x[masku] - 1 / (a_val * np.sqrt(np.pi))

        # Fill in the rest of the values
        mask = ~maskl & ~masku
        out[mask] = x[mask] * special.erf(ax[mask]) + (np.exp(-(ax[mask] ** 2)) - 1) / (a_val * np.sqrt(np.pi))
        return out

    # Vectorized case: a is array of shape (m,), x is shape (n,)
    # Output shape: (m, n)
    a_2d = a[:, np.newaxis]  # Shape: (m, 1)
    x_2d = x[np.newaxis, :]  # Shape: (1, n)
    ax = a_2d * x_2d  # Shape: (m, n)

    out = np.zeros_like(ax, dtype=float)

    # Handle a == 0 rows
    mask_a_zero = a == 0.0
    # Rows where a == 0 remain zero

    # For non-zero a values
    mask_a_nonzero = ~mask_a_zero
    if np.any(mask_a_nonzero):
        ax_valid = ax[mask_a_nonzero, :]  # Shape: (m_valid, n)
        a_valid = a[mask_a_nonzero, np.newaxis]  # Shape: (m_valid, 1)
        x_broadcast = x_2d  # Shape: (1, n), broadcasts to (m_valid, n)

        maskl = ax_valid <= -clip_to_inf
        masku = ax_valid >= clip_to_inf
        mask_mid = ~maskl & ~masku

        out_valid = np.zeros_like(ax_valid)
        out_valid[maskl] = -np.broadcast_to(x_broadcast, ax_valid.shape)[maskl] - 1 / (
            np.broadcast_to(a_valid, ax_valid.shape)[maskl] * np.sqrt(np.pi)
        )
        out_valid[masku] = np.broadcast_to(x_broadcast, ax_valid.shape)[masku] - 1 / (
            np.broadcast_to(a_valid, ax_valid.shape)[masku] * np.sqrt(np.pi)
        )
        x_mid = np.broadcast_to(x_broadcast, ax_valid.shape)[mask_mid]
        a_mid = np.broadcast_to(a_valid, ax_valid.shape)[mask_mid]
        out_valid[mask_mid] = x_mid * special.erf(ax_valid[mask_mid]) + (np.exp(-(ax_valid[mask_mid] ** 2)) - 1) / (
            a_mid * np.sqrt(np.pi)
        )
        out[mask_a_nonzero, :] = out_valid

    return out


def _erf_mean_space(
    edges: NDArray[np.float64],
    a: float | NDArray[np.float64] = 1.0,
) -> NDArray[np.float64]:
    """Compute the mean of the error function between edges.

    This function computes the mean of the error function between two bounds. Provides an
    alternative to computing directly the error function at the cell node. This alternative
    conserves the mass of the signal.

    Parameters
    ----------
    edges : ndarray, shape (n,)
        Cell edges of size n.
    a : float or ndarray, shape (m,), optional
        Scaling factor for the error function a = 1/(2*sqrt(diffusivity*t)).
        Default is 1. If array, output will have shape (m, n-1).

    Returns
    -------
    ndarray
        Mean of the error function between the bounds.
        Shape is (n-1,) if a is scalar, (m, n-1) if a is array of shape (m,).
    """
    edges = np.asarray(edges)
    a = np.asarray(a)
    a_is_scalar = a.ndim == 0

    _edges = np.clip(edges, -1e6, 1e6)
    _erfint = _erf_integral_space(_edges, a=a)
    dx = _edges[1:] - _edges[:-1]

    if a_is_scalar:
        # Original scalar logic
        out = np.where(dx != 0.0, (_erfint[1:] - _erfint[:-1]) / dx, np.nan)
        out[dx == 0.0] = special.erf(a * edges[:-1][dx == 0.0])

        # Handle the case where the edges are far from the origin and have a known outcome
        ue, le = edges[1:], edges[:-1]  # upper and lower cell edges
        out[np.isinf(le) & ~np.isinf(ue)] = -1.0
        out[np.isneginf(le) & np.isneginf(ue)] = -1.0
        out[~np.isinf(le) & np.isinf(ue)] = 1.0
        out[np.isposinf(le) & np.isposinf(ue)] = 1.0
        out[np.isneginf(le) & np.isposinf(ue)] = 0.0
        out[np.isposinf(le) & np.isneginf(ue)] = 0.0
        return out

    # Vectorized case: a is array of shape (m,), edges is shape (n,)
    # _erfint has shape (m, n), output shape: (m, n-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (_erfint[:, 1:] - _erfint[:, :-1]) / dx[np.newaxis, :]

    # Handle dx == 0 (point evaluation): erf(a * x) at that point
    mask_dx_zero = dx == 0.0
    if np.any(mask_dx_zero):
        # a[:, np.newaxis] * edges[:-1][mask_dx_zero] gives shape (m, n_zero)
        ax_at_zero = a[:, np.newaxis] * edges[:-1][mask_dx_zero][np.newaxis, :]
        out[:, mask_dx_zero] = special.erf(ax_at_zero)

    # Handle the case where the edges are at infinity (known asymptotic values)
    ue, le = edges[1:], edges[:-1]  # upper and lower cell edges
    out[:, np.isinf(le) & ~np.isinf(ue)] = -1.0
    out[:, np.isneginf(le) & np.isneginf(ue)] = -1.0
    out[:, ~np.isinf(le) & np.isinf(ue)] = 1.0
    out[:, np.isposinf(le) & np.isposinf(ue)] = 1.0
    out[:, np.isneginf(le) & np.isposinf(ue)] = 0.0
    out[:, np.isposinf(le) & np.isneginf(ue)] = 0.0
    return out


def _erf_integral_time(
    t: NDArray[np.float64],
    x: float | NDArray[np.float64],
    diffusivity: float,
) -> NDArray[np.float64]:
    r"""Compute the integral of the error function over time.

    This function computes the integral of erf(x/(2*sqrt(D*t))) from 0 to t.

    The analytical solution is derived using substitution u = x/(2*sqrt(D*t)):

    .. math::
        \int_0^t \text{erf}\left(\frac{x}{2\sqrt{D \tau}}\right) d\tau
        = t \cdot \text{erf}\left(\frac{x}{2\sqrt{D t}}\right)
        + \frac{x \sqrt{t}}{\sqrt{\pi D}} \exp\left(-\frac{x^2}{4 D t}\right)
        - \frac{x^2}{2D} \text{erfc}\left(\frac{x}{2\sqrt{D t}}\right)

    Parameters
    ----------
    t : ndarray, shape (n,)
        Input time values. Must be non-negative.
    x : float or ndarray, shape (m,)
        Position value (distance from concentration front).
        If array, output will have shape (n, m).
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.

    Returns
    -------
    ndarray
        Integral of the error function from 0 to t.
        Shape is (n,) if x is scalar, (n, m) if x is array of shape (m,).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    x_is_scalar = x.ndim == 0
    diffusivity = float(diffusivity)

    if diffusivity <= 0.0:
        if x_is_scalar:
            return np.zeros_like(t, dtype=float)
        return np.zeros((len(t), len(x)), dtype=float)

    if x_is_scalar:
        # Original scalar logic
        x_val = float(x)
        out = np.zeros_like(t, dtype=float)

        if x_val == 0.0:
            return out

        mask_positive = t > 0.0
        if not np.any(mask_positive):
            return out

        t_pos = t[mask_positive]
        sqrt_t = np.sqrt(t_pos)
        sqrt_d = np.sqrt(diffusivity)
        sqrt_pi = np.sqrt(np.pi)

        arg = x_val / (2 * sqrt_d * sqrt_t)
        exp_term = np.exp(-(x_val**2) / (4 * diffusivity * t_pos))
        erf_term = special.erf(arg)
        erfc_term = special.erfc(arg)

        term1 = t_pos * erf_term
        term2 = (x_val * sqrt_t / (sqrt_pi * sqrt_d)) * exp_term
        term3 = -(x_val**2 / (2 * diffusivity)) * erfc_term

        out[mask_positive] = term1 + term2 + term3
        out[np.isinf(t)] = np.inf * np.sign(x_val)
        return out

    # Vectorized case: t is shape (n,), x is shape (m,)
    # Output shape: (n, m)
    t_2d = t[:, np.newaxis]  # Shape: (n, 1)
    x_2d = x[np.newaxis, :]  # Shape: (1, m)

    out = np.zeros((len(t), len(x)), dtype=float)

    # Mask for valid computation: t > 0 and x != 0
    mask_t_pos = t > 0.0
    mask_x_nonzero = x != 0.0

    # Compute for all valid (t, x) pairs
    if np.any(mask_t_pos) and np.any(mask_x_nonzero):
        t_valid = t_2d[mask_t_pos]  # Shape: (n_valid, 1)
        x_valid = x_2d[:, mask_x_nonzero]  # Shape: (1, m_valid)

        # Broadcast to full grid
        t_grid = np.broadcast_to(t_valid, (np.sum(mask_t_pos), np.sum(mask_x_nonzero)))
        x_grid = np.broadcast_to(x_valid, (np.sum(mask_t_pos), np.sum(mask_x_nonzero)))

        sqrt_t = np.sqrt(t_grid)
        sqrt_d = np.sqrt(diffusivity)
        sqrt_pi = np.sqrt(np.pi)

        arg = x_grid / (2 * sqrt_d * sqrt_t)
        exp_term = np.exp(-(x_grid**2) / (4 * diffusivity * t_grid))
        erf_term = special.erf(arg)
        erfc_term = special.erfc(arg)

        term1 = t_grid * erf_term
        term2 = (x_grid * sqrt_t / (sqrt_pi * sqrt_d)) * exp_term
        term3 = -(x_grid**2 / (2 * diffusivity)) * erfc_term

        out[np.ix_(mask_t_pos, mask_x_nonzero)] = term1 + term2 + term3

    # Handle infinity: as t -> inf, integral -> inf * sign(x)
    mask_t_inf = np.isinf(t)
    if np.any(mask_t_inf):
        out[mask_t_inf, :] = np.inf * np.sign(x)[np.newaxis, :]

    return out


def _erf_mean_time(
    tedges: NDArray[np.float64],
    x: float | NDArray[np.float64],
    diffusivity: float,
) -> NDArray[np.float64]:
    """Compute the mean of the error function over time intervals.

    This function computes the mean of erf(x/(2*sqrt(D*t))) between time edges.

    Parameters
    ----------
    tedges : ndarray, shape (n,)
        Time edges of size n.
    x : float or ndarray, shape (m,)
        Position value (distance from concentration front).
        If array, output will have shape (n-1, m).
    diffusivity : float
        Diffusivity [m²/day]. Must be positive.

    Returns
    -------
    ndarray
        Mean of the error function between the time bounds.
        Shape is (n-1,) if x is scalar, (n-1, m) if x is array of shape (m,).
    """
    tedges = np.asarray(tedges, dtype=float)
    x = np.asarray(x, dtype=float)
    x_is_scalar = x.ndim == 0
    diffusivity = float(diffusivity)

    # Compute integral at all edges
    erfint = _erf_integral_time(tedges, x, diffusivity)

    # Compute mean as difference of integrals divided by time interval
    dt = tedges[1:] - tedges[:-1]

    if x_is_scalar:
        # Original scalar logic
        x_val = float(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(dt != 0.0, (erfint[1:] - erfint[:-1]) / dt, np.nan)

        # For dt == 0, evaluate erf at that point
        mask_zero_dt = dt == 0.0
        if np.any(mask_zero_dt):
            t_at_zero = tedges[:-1][mask_zero_dt]
            with np.errstate(divide="ignore", invalid="ignore"):
                arg = np.where(
                    t_at_zero > 0,
                    x_val / (2 * np.sqrt(diffusivity * t_at_zero)),
                    np.inf * np.sign(x_val) if x_val != 0 else 0.0,
                )
            out[mask_zero_dt] = special.erf(arg)

        # Handle infinite time edges
        ut, lt = tedges[1:], tedges[:-1]
        out[~np.isinf(lt) & np.isposinf(ut)] = 0.0
        out[np.isposinf(lt) & np.isposinf(ut)] = 0.0
        return out

    # Vectorized case: tedges is shape (n,), x is shape (m,)
    # erfint has shape (n, m), output shape: (n-1, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (erfint[1:, :] - erfint[:-1, :]) / dt[:, np.newaxis]

    # Handle dt == 0 (point evaluation): erf(x / (2*sqrt(D*t))) at that point
    mask_zero_dt = dt == 0.0
    if np.any(mask_zero_dt):
        t_at_zero = tedges[:-1][mask_zero_dt, np.newaxis]  # Shape: (n_zero, 1)
        x_2d = x[np.newaxis, :]  # Shape: (1, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = np.where(
                t_at_zero > 0,
                x_2d / (2 * np.sqrt(diffusivity * t_at_zero)),
                np.where(x_2d != 0, np.inf * np.sign(x_2d), 0.0),
            )
        out[mask_zero_dt, :] = special.erf(arg)

    # Handle infinite time edges
    ut, lt = tedges[1:], tedges[:-1]
    out[~np.isinf(lt) & np.isposinf(ut), :] = 0.0
    out[np.isposinf(lt) & np.isposinf(ut), :] = 0.0

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


def _erf_integral_space_time(x, t, diffusivity):
    """
    Compute the integral of the error function in space and time.

    This function computes the double integral:
    F(x,t) = ∫₀ᵗ ∫₀ˣ erf(ξ/(2√(Dτ))) dξ dτ

    The double integral is symmetric in x: F(-x, t) = F(x, t).

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
    # The double integral is symmetric in x: F(-x, t) = F(x, t)
    x = np.abs(np.asarray(x))
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


def _erf_mean_space_time(xedges, tedges, diffusivity, *, paired=False):
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
    paired : bool, optional
        If True and xedges/tedges have same length n, compute n-1 paired cells
        where cell i spans [xedges[i], xedges[i+1]] x [tedges[i], tedges[i+1]].
        Returns 1D array of length n-1. Default False.

    Returns
    -------
    ndarray
        Mean of the error function over each cell. Shape is (n_t, n_x).
        Returns scalar if both n_t=1 and n_x=1.
        Returns 1D array if either n_t=1 or n_x=1.
        If paired=True, returns 1D array of length n-1.
    """
    # Ensure arrays
    xedges = np.asarray(xedges)
    tedges = np.asarray(tedges)

    # Handle paired mode: compute diagonal cells where x and t edges move together
    if paired:
        if len(xedges) != len(tedges):
            msg = "paired=True requires xedges and tedges to have same length"
            raise ValueError(msg)

        # Handle zero diffusivity case: erf(x/(2*sqrt(D*t))) -> sign(x) as D->0
        if diffusivity == 0.0:
            # For zero diffusivity, the mean erf over a cell is simply sign(x_midpoint)
            x_mid = (xedges[:-1] + xedges[1:]) / 2
            out = np.sign(x_mid)
            # Handle NaN in xedges
            out[np.isnan(xedges[:-1]) | np.isnan(xedges[1:])] = np.nan
            if len(out) == 1:
                return out[0]
            return out

        # Use pointwise function for F(x,t) at all 4 corners of each cell
        # Cell i has corners: (x[i],t[i]), (x[i+1],t[i]), (x[i],t[i+1]), (x[i+1],t[i+1])
        f_00 = _erf_integral_space_time_pointwise(xedges[:-1], tedges[:-1], diffusivity)
        f_10 = _erf_integral_space_time_pointwise(xedges[1:], tedges[:-1], diffusivity)
        f_01 = _erf_integral_space_time_pointwise(xedges[:-1], tedges[1:], diffusivity)
        f_11 = _erf_integral_space_time_pointwise(xedges[1:], tedges[1:], diffusivity)

        # Inclusion-exclusion: integral over each cell
        double_integrals = f_11 - f_10 - f_01 + f_00

        # Cell areas
        dx = xedges[1:] - xedges[:-1]
        dt = tedges[1:] - tedges[:-1]
        cell_areas = dx * dt

        # Compute averages
        with np.errstate(divide="ignore", invalid="ignore"):
            out = double_integrals / cell_areas

        # Handle zero-area cells (dt=0 or dx=0)
        mask_dt_zero = dt == 0.0
        if np.any(mask_dt_zero):
            dt_zero_indices = np.where(mask_dt_zero)[0]
            for idx in dt_zero_indices:
                t_val = tedges[idx]
                x_mid = (xedges[idx] + xedges[idx + 1]) / 2
                if t_val > 0:
                    # a = 1/(2*sqrt(D*t)) for point evaluation
                    a_val = 1.0 / (2.0 * np.sqrt(diffusivity * t_val))
                    out[idx] = special.erf(a_val * x_mid)
                else:
                    # At t=0, erf(x/(2*sqrt(D*0))) = sign(x)
                    out[idx] = np.sign(x_mid)

        mask_dx_zero = dx == 0.0
        if np.any(mask_dx_zero):
            # For cells with dx=0, compute mean erf over time at fixed x
            zero_indices = np.where(mask_dx_zero)[0]
            for idx in zero_indices:
                x_val = xedges[idx]
                t_edges = np.array([tedges[idx], tedges[idx + 1]])
                out[idx] = _erf_mean_time(t_edges, x_val, diffusivity)[0]

        # Handle infinite x edges
        le, ue = xedges[:-1], xedges[1:]
        out[np.isinf(le) & ~np.isinf(ue)] = -1.0
        out[np.isneginf(le) & np.isneginf(ue)] = -1.0
        out[~np.isinf(le) & np.isinf(ue)] = 1.0
        out[np.isposinf(le) & np.isposinf(ue)] = 1.0
        out[np.isneginf(le) & np.isposinf(ue)] = 0.0
        out[np.isposinf(le) & np.isneginf(ue)] = 0.0

        # Return 1D array or scalar
        if len(out) == 1:
            return out[0]
        return out

    # Compute the double integral at all edge combinations
    # _erf_integral_space_time returns shape (n_t, n_x) for inputs of size n_t and n_x
    _erfint = _erf_integral_space_time(xedges, tedges, diffusivity)

    # Use inclusion-exclusion to get integral over each cell
    # I[i,j] = F(x_{j+1}, t_{i+1}) - F(x_j, t_{i+1}) - F(x_{j+1}, t_i) + F(x_j, t_i)
    # _erfint has shape (n_tedges, n_xedges), we want (n_t_cells, n_x_cells)
    double_integrals = _erfint[1:, 1:] - _erfint[:-1, 1:] - _erfint[1:, :-1] + _erfint[:-1, :-1]

    # Calculate cell areas
    dt = np.diff(tedges)[:, np.newaxis]
    dx = np.diff(xedges)[np.newaxis, :]
    cell_areas = dx * dt

    # Compute averages
    with np.errstate(divide="ignore", invalid="ignore"):
        out = double_integrals / cell_areas

    # Handle zero-area cells (dt=0 or dx=0) - these are instantaneous or point evaluations
    # For dt=0: need to evaluate mean over space at fixed time using erf_mean_space
    dt_flat = np.diff(tedges)
    mask_dt_zero = dt_flat == 0.0
    if np.any(mask_dt_zero):
        idx_dt_zero = np.where(mask_dt_zero)[0]
        t_fixed_values = tedges[idx_dt_zero]
        mask_t_positive = t_fixed_values > 0

        if np.any(mask_t_positive):
            # a = 1/(2*sqrt(D*t)) for each t value
            t_pos = t_fixed_values[mask_t_positive]
            a_values = 1.0 / (2.0 * np.sqrt(diffusivity * t_pos))
            # Use vectorized _erf_mean_space: returns shape (len(a_values), n_x_cells)
            erf_means_space = _erf_mean_space(xedges, a=a_values)
            out[idx_dt_zero[mask_t_positive], :] = erf_means_space

        if np.any(~mask_t_positive):
            # At t=0, erf(x/(2*sqrt(D*0))) = erf(+-inf) = sign(x)
            cell_centers = (xedges[:-1] + xedges[1:]) / 2
            out[idx_dt_zero[~mask_t_positive], :] = np.sign(cell_centers)

    # For dx=0: need to evaluate mean over time at fixed space using _erf_mean_time
    dx_flat = np.diff(xedges)
    mask_dx_zero = dx_flat == 0.0
    if np.any(mask_dx_zero):
        idx_dx_zero = np.where(mask_dx_zero)[0]
        x_fixed_values = xedges[idx_dx_zero]
        # Use vectorized _erf_mean_time: returns shape (n_t_cells, len(x_fixed_values))
        erf_means_time = _erf_mean_time(tedges, x=x_fixed_values, diffusivity=diffusivity)
        out[:, idx_dx_zero] = erf_means_time

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
    streamline_length : array-like
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

            # With paired=True, returns (n_cout_bins,) - one value per output bin
            response[:, j] = _erf_mean_space_time(xedges_j, tedges_j, diffusivity, paired=True)

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
