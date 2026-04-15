"""
Analytical solutions for 1D advection-dispersion transport.

This module implements analytical solutions for solute transport in 1D aquifer
systems, combining advection with longitudinal dispersion. The solutions are
based on the error function (erf) and its integrals.

Key functions:

- :func:`infiltration_to_extraction` - Main transport function combining advection and dispersion
  with explicit pore volume distribution and streamline lengths.

- :func:`extraction_to_infiltration` - Inverse operation (deconvolution with dispersion).

- :func:`gamma_infiltration_to_extraction` - Gamma-distributed pore volumes with dispersion.
  Models aquifer heterogeneity with 2-parameter gamma distribution. Parameterizable via
  (alpha, beta) or (mean, std). Discretizes gamma distribution into equal-probability bins.

- :func:`gamma_extraction_to_infiltration` - Gamma-distributed pore volumes, deconvolution
  with dispersion. Symmetric inverse of gamma_infiltration_to_extraction.

The dispersion is characterized by the longitudinal dispersion coefficient D_L,
which is computed internally from:

    D_L = D_m + alpha_L * v_s

where:
- D_m is the molecular diffusion coefficient [m^2/day]
- alpha_L is the longitudinal dispersivity [m]
- v_s is the retarded velocity [m/day], computed as v_s = L / (R * tau_mean)

The velocity v_s is computed per (pore_volume, output_bin) using the mean residence
time (which includes retardation), correctly accounting for time-varying flow.
This formulation ensures that:
- Molecular diffusion spreading scales with residence time: sqrt(D_m * tau)
- Mechanical dispersion spreading scales with travel distance: sqrt(alpha_L * L)

This module adds microdispersion (alpha_L) and molecular diffusion (D_m) on top of
macrodispersion captured by the pore volume distribution (APVD). Both represent velocity
heterogeneity at different scales. Microdispersion is an aquifer property; macrodispersion
depends additionally on hydrological boundary conditions. See :ref:`concept-dispersion-scales`
for guidance on when to use each approach and how to avoid double-counting spreading effects.
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.typing import NDArray
from scipy import special

from gwtransport import gamma
from gwtransport.residence_time import residence_time, residence_time_mean
from gwtransport.utils import solve_inverse_transport

# Numerical tolerance for coefficient sum to determine valid output bins
EPSILON_COEFF_SUM = 1e-10

# Gauss-Legendre quadrature nodes and weights for volume-space integration
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


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


def _erf_mean_volume(
    *,
    step_widths: NDArray[np.float64],
    raw_time: NDArray[np.float64],
    rt_at_cin_edges: NDArray[np.float64],
    diffusivity: NDArray[np.float64],
    cumulative_volume_at_cout_tedges: NDArray[np.float64],
    cumulative_volume_at_cin_tedges: NDArray[np.float64],
    tedges_days: NDArray[np.float64],
    r_vpv: float,
    streamline_len: float,
    asymptotic_cutoff_sigma: float | None,
) -> NDArray[np.float64]:
    r"""Compute mean erf along the physical trajectory in cumulative volume space.

    For each cell (cout_bin *i*, cin_edge *j*), computes the flow-weighted
    average of the error function along the 1D extraction trajectory:

    .. math::

        \text{mean\_erf}_{i,j} = \frac{1}{\Delta V_i}
        \int_{V_i}^{V_{i+1}} \text{erf}\!\left(
            \frac{x(V)}{2\sqrt{D \cdot \tau(V)}}
        \right) dV

    where *x(V)* is the normalized distance (linear in *V*) and *τ(V)* is the
    elapsed time since infiltration (capped at the residence time *RT*).
    Averaging over cumulative volume gives a flow-weighted average because
    dV = Q(t) dt.

    **Fully capped cells** (both cout edges have elapsed time ≥ RT):
    τ = RT (constant), so the integral reduces to ``_erf_integral_space``
    at fixed *t* = RT. This path gives machine precision.

    **Uncapped cells** (at least one cout edge has elapsed time < RT):
    Vectorized 16-point Gauss-Legendre quadrature in volume space.  The
    integration is split at flow-bin boundaries (where *Q* changes) so
    that within each sub-interval the integrand is smooth, and at the
    capping transition (where *τ* switches from linear growth to the
    constant *RT*) where the exact ``_erf_integral_space`` is used for
    the capped portion.  The outer loop runs over flow bins; within each
    iteration all overlapping uncapped cells are evaluated simultaneously.

    Parameters
    ----------
    step_widths : ndarray, shape (n_cout_edges, n_cin_edges)
        Normalized x-position at each (cout_edge, cin_edge) point.
        NaN for inactive cells.
    raw_time : ndarray, shape (n_cout_edges, n_cin_edges)
        Raw elapsed time in days (before capping at RT).
        NaN for inactive cells.
    rt_at_cin_edges : ndarray, shape (n_cin_edges,)
        Residence time at each cin edge for this pore volume [days].
    diffusivity : ndarray, shape (n_cout_bins,)
        Longitudinal dispersion coefficient per cout bin [m²/day].
    cumulative_volume_at_cout_tedges : ndarray, shape (n_cout_edges,)
        Cumulative extracted volume at each cout time edge [m³].
    cumulative_volume_at_cin_tedges : ndarray, shape (n_cin_edges,)
        Cumulative volume at each cin (flow) time edge [m³].
    tedges_days : ndarray, shape (n_cin_edges,)
        Flow time edges in days (same reference as raw_time).
    r_vpv : float
        Retardation factor times pore volume [m³].
    streamline_len : float
        Streamline length [m].
    asymptotic_cutoff_sigma : float or None
        Erf cutoff threshold for ``_erf_integral_space``.

    Returns
    -------
    ndarray, shape (n_cout_bins, n_cin_edges)
        Mean erf value for each cell. NaN for inactive cells.
    """
    n_cout_edges, n_cin_edges = step_widths.shape
    n_cout_bins = n_cout_edges - 1

    x_lo = step_widths[:-1]
    x_hi = step_widths[1:]
    dx = x_hi - x_lo

    v_lo_arr = cumulative_volume_at_cout_tedges[:-1]
    v_hi_arr = cumulative_volume_at_cout_tedges[1:]

    is_valid = ~np.isnan(x_lo) & ~np.isnan(x_hi)

    # Determine capping status at each cell edge
    with np.errstate(invalid="ignore"):
        is_capped_lo = raw_time[:-1] >= rt_at_cin_edges[np.newaxis, :]
        is_capped_hi = raw_time[1:] >= rt_at_cin_edges[np.newaxis, :]
    is_fully_capped = is_capped_lo & is_capped_hi

    response = np.full((n_cout_bins, n_cin_edges), np.nan)

    # --- D = 0: erf = sign(x), exact for any τ ---
    mask_d_zero = diffusivity == 0.0
    if np.any(mask_d_zero):
        mask_d_zero_2d = is_valid & np.broadcast_to(mask_d_zero[:, np.newaxis], (n_cout_bins, n_cin_edges))
        with np.errstate(divide="ignore", invalid="ignore"):
            d_zero_vals = (np.abs(x_hi) - np.abs(x_lo)) / dx
        d_zero_vals = np.where(dx == 0.0, np.sign(x_lo), d_zero_vals)
        response = np.where(mask_d_zero_2d, d_zero_vals, response)
        is_valid &= ~mask_d_zero_2d

    # --- Fully capped cells: exact via _erf_integral_space ---
    mask_capped = is_valid & is_fully_capped
    if np.any(mask_capped):
        t_const = np.broadcast_to(rt_at_cin_edges[np.newaxis, :], (n_cout_bins, n_cin_edges))[mask_capped]
        d_bc = np.broadcast_to(diffusivity[:, np.newaxis], (n_cout_bins, n_cin_edges))[mask_capped]

        clip_kw = {"clip_to_inf": asymptotic_cutoff_sigma} if asymptotic_cutoff_sigma is not None else {}
        h_hi = _erf_integral_space(x_hi[mask_capped], diffusivity=d_bc, t=t_const, **clip_kw)
        h_lo = _erf_integral_space(x_lo[mask_capped], diffusivity=d_bc, t=t_const, **clip_kw)
        dx_c = dx[mask_capped]

        with np.errstate(divide="ignore", invalid="ignore"):
            a_vals = np.where(
                (t_const <= 0) | (d_bc <= 0),
                np.inf,
                1.0 / (2.0 * np.sqrt(d_bc * t_const)),
            )
        point_val = np.where(np.isinf(a_vals), np.sign(x_lo[mask_capped]), special.erf(a_vals * x_lo[mask_capped]))

        with np.errstate(divide="ignore", invalid="ignore"):
            response[mask_capped] = np.where(dx_c == 0.0, point_val, (h_hi - h_lo) / dx_c)

    # --- Uncapped cells: vectorized Gauss-Legendre quadrature in volume space ---
    mask_uncapped = is_valid & ~is_fully_capped
    if np.any(mask_uncapped):
        idx_i, idx_j = np.nonzero(mask_uncapped)
        n_cells = len(idx_i)

        # Gather cell parameters (all shape (n_cells,))
        v_lo_cells = v_lo_arr[idx_i]
        v_hi_cells = v_hi_arr[idx_i]
        v_cin_cells = cumulative_volume_at_cin_tedges[idx_j]
        rt_cells = rt_at_cin_edges[idx_j]
        t_j_cells = tedges_days[idx_j]
        d_cells = diffusivity[idx_i]
        total_dv = v_hi_cells - v_lo_cells

        valid_cells = np.isfinite(rt_cells) & (total_dv > 0)

        # Partially capped: start uncapped, end capped
        has_capped = is_capped_hi[idx_i, idx_j] & ~is_capped_lo[idx_i, idx_j]
        t_kink_all = t_j_cells + rt_cells
        v_kink_all = np.interp(t_kink_all, tedges_days, cumulative_volume_at_cin_tedges)
        v_kink_all = np.clip(v_kink_all, v_lo_cells, v_hi_cells)
        v_end = np.where(has_capped, v_kink_all, v_hi_cells)

        # --- GL quadrature over [v_lo, v_end] for all cells simultaneously ---
        # Split at flow bin boundaries for smoothness within each sub-interval.
        # Loop over consecutive pairs of flow-bin volume edges; vectorize
        # across all cells whose uncapped interval overlaps that sub-interval.
        uncapped_integral = np.zeros(n_cells)
        vol_edges = cumulative_volume_at_cin_tedges  # monotonic volume grid

        for k in range(len(vol_edges) - 1):
            ve_lo, ve_hi = vol_edges[k], vol_edges[k + 1]
            # Clip to each cell's uncapped interval
            sub_lo = np.maximum(v_lo_cells, ve_lo)
            sub_hi = np.minimum(v_end, ve_hi)
            overlap = (sub_hi > sub_lo) & valid_cells
            if not np.any(overlap):
                continue

            dv_sub = sub_hi[overlap] - sub_lo[overlap]  # (n_overlap,)
            v_mid_sub = (sub_hi[overlap] + sub_lo[overlap]) / 2
            v_half_sub = dv_sub / 2

            # GL nodes: (n_overlap, n_gl)
            v_nodes = v_mid_sub[:, np.newaxis] + v_half_sub[:, np.newaxis] * _GL_NODES[np.newaxis, :]

            # x(V): (n_overlap, n_gl)
            x_nodes = (v_nodes - v_cin_cells[overlap, np.newaxis] - r_vpv) * streamline_len / r_vpv

            # t(V): within flow sub-interval k, Q is constant so t is linear in V
            dt_sub = tedges_days[k + 1] - tedges_days[k]
            dv_sub_edge = ve_hi - ve_lo
            t_nodes = tedges_days[k] + (v_nodes - ve_lo) * (dt_sub / dv_sub_edge)

            # τ(V) = clip(t(V) - t_j, 0, RT_j): (n_overlap, n_gl)
            tau_nodes = np.clip(t_nodes - t_j_cells[overlap, np.newaxis], 0.0, rt_cells[overlap, np.newaxis])

            # erf(x / (2√(Dτ))): (n_overlap, n_gl)
            with np.errstate(divide="ignore", invalid="ignore"):
                arg = x_nodes / (2.0 * np.sqrt(d_cells[overlap, np.newaxis] * tau_nodes))
            erf_vals = np.where(np.isfinite(arg), special.erf(arg), np.sign(x_nodes))

            # Accumulate weighted integral for overlapping cells
            uncapped_integral[overlap] += (dv_sub / 2) * (erf_vals @ _GL_WEIGHTS)

        # --- Capped sub-interval [v_kink, v_hi]: exact via _erf_integral_space ---
        capped_integral = np.zeros(n_cells)
        mask_cap = has_capped & (v_kink_all < v_hi_cells) & valid_cells
        if np.any(mask_cap):
            x_at_kink = (v_kink_all[mask_cap] - v_cin_cells[mask_cap] - r_vpv) * streamline_len / r_vpv
            x_at_hi_cap = x_hi[idx_i[mask_cap], idx_j[mask_cap]]
            clip_kw = {"clip_to_inf": asymptotic_cutoff_sigma} if asymptotic_cutoff_sigma is not None else {}
            h_hi_val = _erf_integral_space(x_at_hi_cap, diffusivity=d_cells[mask_cap], t=rt_cells[mask_cap], **clip_kw)
            h_lo_val = _erf_integral_space(x_at_kink, diffusivity=d_cells[mask_cap], t=rt_cells[mask_cap], **clip_kw)
            capped_integral[mask_cap] = (h_hi_val - h_lo_val) * (r_vpv / streamline_len)

        # Combine: mean erf = (uncapped + capped) / total_dv
        with np.errstate(divide="ignore", invalid="ignore"):
            response_cells = np.where(valid_cells, (uncapped_integral + capped_integral) / total_dv, np.nan)
        response[idx_i, idx_j] = response_cells

    return response


def _infiltration_to_extraction_coeff_matrix(
    *,
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    streamline_length: npt.NDArray[np.floating],
    molecular_diffusivity: npt.NDArray[np.floating],
    longitudinal_dispersivity: npt.NDArray[np.floating],
    retardation_factor: float,
    asymptotic_cutoff_sigma: float | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Build the forward coefficient matrix for diffusion transport.

    Constructs the matrix W such that ``cout = W @ cin``, accounting for
    advection and longitudinal dispersion. NaN entries in the raw coefficient
    matrix are replaced with zero.

    Parameters
    ----------
    flow : ndarray
        Flow rate of water [m3/day]. Already validated.
    tedges : DatetimeIndex
        Cin/flow time edges (not yet extended for spin-up).
    cout_tedges : DatetimeIndex
        Cout time edges.
    aquifer_pore_volumes : ndarray
        Pore volumes [m3]. Already validated.
    streamline_length : ndarray
        Travel distances [m]. Already validated.
    molecular_diffusivity : ndarray
        Effective molecular diffusivities [m2/day]. Already broadcasted.
        See :func:`infiltration_to_extraction` for physical interpretation.
    longitudinal_dispersivity : ndarray
        Longitudinal dispersivities [m]. Already broadcasted.
    retardation_factor : float
        Retardation factor.
    asymptotic_cutoff_sigma : float or None
        Erf cutoff threshold.

    Returns
    -------
    coeff_matrix : ndarray
        Filled coefficient matrix of shape (n_cout, n_cin). NaN replaced
        with zero.
    valid_cout_bins : ndarray
        Boolean mask of shape (n_cout,) indicating valid output bins.
    """
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
    cumulative_volume_at_cout_tedges = np.interp(cout_tedges, tedges, cumulative_volume_at_cin_tedges).astype(float)

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
    valid_cout_bins = ~np.any(np.isnan(rt_at_cout_tedges[:, :-1]) | np.isnan(rt_at_cout_tedges[:, 1:]), axis=0)

    # Compute mean residence time per (pore_volume, cout_bin) for velocity calculation
    rt_mean_2d = residence_time_mean(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        direction="extraction_to_infiltration",
        retardation_factor=retardation_factor,
    )

    # Compute retarded velocity: v_s = L / (R * tau) = v / R.
    # Using R*tau (from residence_time with retardation) gives the retarded
    # velocity, not the pore velocity. This is intentional:
    # - Dispersivity term: alpha_L * v_s * R * tau = alpha_L * L — R cancels,
    #   giving correct distance-dependent mechanical dispersion.
    # - Molecular term: D_m * R * tau — exact when D_m represents D_eff = D_m/R.
    #   For solutes D_m ~ 1e-5 m²/day (negligible); for heat, users pass
    #   D_th = lambda/(rho*c)_eff which is already the effective diffusivity.
    valid_rt = np.isfinite(rt_mean_2d) & (rt_mean_2d > 0)
    velocity_2d = np.where(valid_rt, streamline_length[:, None] / rt_mean_2d, 0.0)

    # Compute D_L = D_m + alpha_L * v_s
    diffusivity_2d = np.where(
        valid_rt,
        molecular_diffusivity[:, None] + longitudinal_dispersivity[:, None] * velocity_2d,
        molecular_diffusivity[:, None],
    )

    # Initialize coefficient matrix accumulator
    n_cout_bins = len(cout_tedges) - 1
    n_cin_bins = len(flow)
    accumulated_coeff = np.zeros((n_cout_bins, n_cin_bins))

    # Determine when infiltration has occurred: cout_tedge must be >= tedge (infiltration time)
    isactive = cout_tedges.to_numpy()[:, None] >= tedges.to_numpy()[None, :]

    # Compute raw elapsed time (before capping at RT) once for all pore volumes
    raw_time = (cout_tedges.to_numpy()[:, None] - tedges.to_numpy()[None, :]) / pd.to_timedelta(1, unit="D")
    raw_time[~isactive] = np.nan

    # Convert tedges to days for volume→time interpolation
    tedges_days_arr = ((tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)

    # Loop over each pore volume
    for i_pv in range(len(aquifer_pore_volumes)):
        r_vpv = retardation_factor * aquifer_pore_volumes[i_pv]

        delta_volume = cumulative_volume_at_cout_tedges[:, None] - cumulative_volume_at_cin_tedges[None, :] - r_vpv
        delta_volume[~isactive] = np.nan

        step_widths = delta_volume / r_vpv * streamline_length[i_pv]

        diff_pv = diffusivity_2d[i_pv, :]
        response = _erf_mean_volume(
            step_widths=step_widths,
            raw_time=raw_time,
            rt_at_cin_edges=rt_edges_2d[i_pv],
            diffusivity=diff_pv,
            cumulative_volume_at_cout_tedges=cumulative_volume_at_cout_tedges,
            cumulative_volume_at_cin_tedges=cumulative_volume_at_cin_tedges,
            tedges_days=tedges_days_arr,
            r_vpv=r_vpv,
            streamline_len=streamline_length[i_pv],
            asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
        )

        frac = 0.5 * (1 + response)
        frac_start = frac[:, :-1]
        frac_end = frac[:, 1:]
        frac_end_filled = np.where(np.isnan(frac_end) & ~np.isnan(frac_start), 0.0, frac_end)
        coeff = frac_start - frac_end_filled

        accumulated_coeff += coeff

    coeff_matrix = accumulated_coeff / len(aquifer_pore_volumes)
    coeff_matrix_filled = np.nan_to_num(coeff_matrix, nan=0.0)

    return coeff_matrix_filled, valid_cout_bins


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
    asymptotic_cutoff_sigma: float | None = 3.0,
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

        D_L = D_m + alpha_L * v_s

    where v_s = L / (R * tau_mean) is the retarded velocity computed from the
    mean residence time (which includes retardation) for each (pore_volume,
    output_bin) combination. For the dispersivity term, the R factors cancel
    (alpha_L * v_s * R * tau = alpha_L * L), giving correct distance-dependent
    mechanical dispersion. See the ``molecular_diffusivity`` parameter for how
    the molecular term interacts with retardation.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water [concentration units].
        Length must match the number of time bins defined by tedges. The model assumes
        this value is constant over each interval ``[tedges[i], tedges[i+1])``.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
        Length must match cin and the number of time bins defined by tedges. The model
        assumes this value is constant over each interval ``[tedges[i], tedges[i+1])``.
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
        Effective molecular diffusivity [m2/day]. Can be a scalar (same for all
        pore volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative. For solute transport, this is the molecular
        diffusion coefficient D_m [m2/day] — typically ~1e-5 m2/day, negligible
        compared to mechanical dispersion. For heat transport, pass the thermal
        diffusivity D_th = lambda / (rho*c)_eff [m2/day], typically 0.01-0.1
        m2/day.

        Internally, this value enters the dispersion formula as
        D_L = molecular_diffusivity + alpha_L * v_s, where v_s = L / (R * tau)
        is the retarded velocity. For the dispersivity term, the R factors cancel
        (alpha_L * L / (R * tau) * R * tau = alpha_L * L), giving correct
        distance-dependent spreading. For the molecular term, the spreading
        scales as molecular_diffusivity * R * tau. This is exact when
        molecular_diffusivity equals D_m/R — which is negligible for solutes
        (D_m ~ 1e-5) and correct for heat (where thermal diffusivity already
        represents D_eff, not D_m * R).
    longitudinal_dispersivity : float or array-like
        Longitudinal dispersivity [m]. Can be a scalar (same for all pore
        volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative. Represents microdispersion (mechanical dispersion
        from pore-scale velocity variations). Set to 0 for pure molecular diffusion.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption.
    suppress_dispersion_warning : bool, optional
        If True, suppress the warning when using multiple pore volumes with
        non-zero longitudinal_dispersivity. Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Performance optimization. Cells where the erf argument magnitude exceeds
        this threshold are assigned asymptotic values (±1) instead of computing
        the expensive integral. Since erf(3) ≈ 0.99998, the default of 3.0
        provides excellent accuracy with significant speedup. Set to None to
        disable the optimization. Default is 3.0.

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

    See Also
    --------
    extraction_to_infiltration : Inverse operation (deconvolution)
    gwtransport.advection.infiltration_to_extraction : Pure advection (no dispersion)
    gwtransport.diffusion_fast.infiltration_to_extraction : Fast Gaussian approximation
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The algorithm constructs a coefficient matrix W where cout = W @ cin:

    1. For each pore volume, compute the breakthrough curve contribution:
       - delta_volume: volume between infiltration event and extraction point
       - step_widths: convert volume to spatial distance x (erf coordinate)
       - time_active: diffusion time, limited by residence time

    2. For each infiltration time edge, compute the erf response at all
       extraction time edges using analytical space-time averaging.

    3. Convert erf response to breakthrough fraction: frac = 0.5 * (1 + erf)

    4. Coefficient for bin: coeff[i,j] = frac_start - frac_end
       This represents the fraction of cin[j] that arrives in cout[i].

    5. Average coefficients across all pore volumes.

    The error function solution assumes an initial step function that diffuses
    over time. The position coordinate x represents the distance from the
    concentration front to the observation point.

    Examples
    --------
    Basic usage with constant flow:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion import infiltration_to_extraction
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
    streamline_length = np.atleast_1d(np.asarray(streamline_length, dtype=float))

    # Convert diffusion parameters to arrays and broadcast to pore volumes
    n_pore_volumes = len(aquifer_pore_volumes)
    molecular_diffusivity = np.atleast_1d(np.asarray(molecular_diffusivity, dtype=float))
    longitudinal_dispersivity = np.atleast_1d(np.asarray(longitudinal_dispersivity, dtype=float))

    # Broadcast scalar values to match pore volumes
    if streamline_length.size == 1:
        streamline_length = np.broadcast_to(streamline_length, (n_pore_volumes,)).copy()
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
    if np.any(flow <= 0):
        msg = "flow must be positive"
        raise ValueError(msg)
    if np.any(aquifer_pore_volumes <= 0):
        msg = "aquifer_pore_volumes must be positive"
        raise ValueError(msg)
    if np.any(streamline_length <= 0):
        msg = "streamline_length must be positive"
        raise ValueError(msg)

    # Check for conflicting approaches: multiple pore volumes with longitudinal dispersivity
    # Both represent the same physical phenomenon (velocity heterogeneity) at different scales.
    # See concept-dispersion-scales in the documentation for details.
    if n_pore_volumes > 1 and np.any(longitudinal_dispersivity > 0) and not suppress_dispersion_warning:
        msg = (
            "Using multiple aquifer_pore_volumes with non-zero longitudinal_dispersivity. "
            "Both represent spreading from velocity heterogeneity at different scales.\n\n"
            "This is appropriate when:\n"
            "  - APVD comes from streamline analysis (explicit geometry)\n"
            "  - You want to add unresolved microdispersion and molecular diffusion\n\n"
            "This may double-count spreading when:\n"
            "  - APVD was calibrated from measurements (microdispersion already included)\n\n"
            "For gamma-parameterized APVD, consider using the 'equivalent APVD std' approach\n"
            "from 05_Diffusion_Dispersion.ipynb to combine both effects in the faster advection\n"
            "module. Note: this variance combination is only valid for continuous (gamma)\n"
            "distributions, not for discrete streamline volumes.\n"
            "Suppress this warning with suppress_dispersion_warning=True."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    coeff_matrix, valid_cout_bins = _infiltration_to_extraction_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
    )

    # Matrix multiply: cout = W @ cin
    cout = coeff_matrix @ cin

    # Mark output bins as invalid where no valid contributions exist:
    # 1. Sum of coefficients near zero (no cin has broken through yet - spinup)
    # 2. Output bin extends beyond input data range (from valid_cout_bins)
    total_coeff = np.sum(coeff_matrix, axis=1)
    no_valid_contribution = (total_coeff < EPSILON_COEFF_SUM) | ~valid_cout_bins
    cout[no_valid_contribution] = np.nan

    return cout


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: npt.ArrayLike,
    molecular_diffusivity: npt.ArrayLike,
    longitudinal_dispersivity: npt.ArrayLike,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    regularization_strength: float = 1e-10,
) -> npt.NDArray[np.floating]:
    """
    Compute infiltration concentration from extracted water (deconvolution with dispersion).

    Inverts the forward transport model by building the forward coefficient
    matrix ``W_forward`` from :func:`infiltration_to_extraction` and solving
    ``W_forward @ cin = cout`` via Tikhonov regularization. Well-determined
    modes are dominated by the data; poorly-determined modes are pulled
    toward the physically motivated target (transpose-and-normalize of the
    forward matrix).

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water [concentration units].
        Length must match the number of time bins defined by cout_tedges.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
        Length must match the number of time bins defined by tedges.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for cin (output) and flow data.
        Has length of len(flow) + 1. Output cin has length len(tedges) - 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Has length of len(cout) + 1.
        Can have different time alignment and resolution than tedges.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution
        of flow paths. Each pore volume determines the residence time for
        that flow path: tau = V_pore / Q.
    streamline_length : array-like
        Array of travel distances [m] corresponding to each pore volume.
        Must have the same length as aquifer_pore_volumes.
    molecular_diffusivity : float or array-like
        Effective molecular diffusivity [m2/day]. Can be a scalar (same for all
        pore volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative. See :func:`infiltration_to_extraction` for
        details on the physical interpretation and the interaction with
        retardation_factor.
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
    asymptotic_cutoff_sigma : float or None, optional
        Performance optimization for the forward matrix construction.
        Default is 3.0.
    regularization_strength : float, optional
        Tikhonov regularization parameter λ. See
        :func:`gwtransport.advection.extraction_to_infiltration` for details.
        Default is 1e-10.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Same units as cout.
        Length equals len(tedges) - 1. NaN values indicate time periods
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
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The algorithm builds the forward coefficient matrix ``W_forward`` (same as
    used by :func:`infiltration_to_extraction`) and solves ``W_forward @ cin = cout``
    using :func:`gwtransport.utils.solve_tikhonov`. This ensures mathematical
    consistency between forward and inverse operations.

    Examples
    --------
    Basic usage with constant flow:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion import extraction_to_infiltration
    >>>
    >>> # Create time edges: tedges for cin/flow, cout_tedges for cout
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>>
    >>> # Extracted concentration and constant flow
    >>> cout = np.zeros(len(cout_tedges) - 1)
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
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     molecular_diffusivity=1e-4,
    ...     longitudinal_dispersivity=1.0,
    ... )
    """
    # Convert to pandas DatetimeIndex if needed
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    # Convert to arrays
    cout = np.asarray(cout, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    streamline_length = np.atleast_1d(np.asarray(streamline_length, dtype=float))

    # Convert diffusion parameters to arrays and broadcast to pore volumes
    n_pore_volumes = len(aquifer_pore_volumes)
    molecular_diffusivity = np.atleast_1d(np.asarray(molecular_diffusivity, dtype=float))
    longitudinal_dispersivity = np.atleast_1d(np.asarray(longitudinal_dispersivity, dtype=float))

    # Broadcast scalar values to match pore volumes
    if streamline_length.size == 1:
        streamline_length = np.broadcast_to(streamline_length, (n_pore_volumes,)).copy()
    if molecular_diffusivity.size == 1:
        molecular_diffusivity = np.broadcast_to(molecular_diffusivity, (n_pore_volumes,)).copy()
    if longitudinal_dispersivity.size == 1:
        longitudinal_dispersivity = np.broadcast_to(longitudinal_dispersivity, (n_pore_volumes,)).copy()

    # Input validation
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if len(cout_tedges) != len(cout) + 1:
        msg = "cout_tedges must have one more element than cout"
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

    # Check for conflicting approaches
    if n_pore_volumes > 1 and np.any(longitudinal_dispersivity > 0) and not suppress_dispersion_warning:
        msg = (
            "Using multiple aquifer_pore_volumes with non-zero longitudinal_dispersivity. "
            "Both represent spreading from velocity heterogeneity at different scales.\n\n"
            "This is appropriate when:\n"
            "  - APVD comes from streamline analysis (explicit geometry)\n"
            "  - You want to add unresolved microdispersion and molecular diffusion\n\n"
            "This may double-count spreading when:\n"
            "  - APVD was calibrated from measurements (microdispersion already included)\n\n"
            "For gamma-parameterized APVD, consider using the 'equivalent APVD std' approach\n"
            "from 05_Diffusion_Dispersion.ipynb to combine both effects in the faster advection\n"
            "module. Note: this variance combination is only valid for continuous (gamma)\n"
            "distributions, not for discrete streamline volumes.\n"
            "Suppress this warning with suppress_dispersion_warning=True."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    n_cin = len(tedges) - 1

    # Build forward weight matrix: W_forward @ cin = cout
    w_forward, valid_cout_bins = _infiltration_to_extraction_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
    )

    return solve_inverse_transport(
        w_forward=w_forward,
        observed=cout,
        n_output=n_cin,
        regularization_strength=regularization_strength,
        valid_rows=valid_cout_bins,
    )


def gamma_infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    streamline_length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
) -> npt.NDArray[np.floating]:
    """
    Compute extracted concentration with advection-dispersion for gamma-distributed pore volumes.

    Combines advective transport (based on gamma-distributed pore volumes) with
    longitudinal dispersion (diffusive spreading during transport). This is a
    convenience wrapper around :func:`infiltration_to_extraction` that parameterizes
    the aquifer pore volume distribution as a (shifted) gamma distribution.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Has length len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
    mean : float, optional
        Mean of the gamma distribution of the aquifer pore volume. Must be strictly
        greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma distribution of the aquifer pore volume
        (invariant under the ``loc`` shift).
    loc : float, optional
        Location (minimum pore volume) of the gamma distribution. Must satisfy
        ``0 <= loc < mean``. Default is ``0.0``.
    alpha : float, optional
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0).
    beta : float, optional
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0).
    n_bins : int, optional
        Number of bins to discretize the gamma distribution. Default is 100.
    streamline_length : float
        Travel distance through the aquifer [m]. Applied uniformly to all
        gamma-discretized pore volumes.
    molecular_diffusivity : float
        Effective molecular diffusivity [m2/day]. Must be non-negative.
        See :func:`infiltration_to_extraction` for details on the interaction
        with retardation_factor.
    longitudinal_dispersivity : float
        Longitudinal dispersivity [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Performance optimization. Cells where the erf argument magnitude exceeds
        this threshold are assigned asymptotic values. Default is 3.0.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the extracted water. Length equals
        len(cout_tedges) - 1. NaN values indicate time periods with no valid
        contributions from the infiltration data.

    See Also
    --------
    infiltration_to_extraction : Transport with explicit pore volume distribution
    gamma_extraction_to_infiltration : Reverse operation (deconvolution)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.advection.gamma_infiltration_to_extraction : Pure advection (no dispersion)
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The APVD is only time-invariant under the steady-streamlines assumption
    (see :ref:`assumption-steady-streamlines`).

    The spreading from the gamma-distributed pore volumes represents macrodispersion
    (aquifer-scale heterogeneity). When ``std`` comes from calibration on measurements,
    it absorbs all mixing: macrodispersion, microdispersion, and an average molecular
    diffusion contribution. When ``std`` comes from streamline analysis, it represents
    macrodispersion only; microdispersion and molecular diffusion can be added via the
    dispersion parameters.
    See :ref:`concept-dispersion-scales` for guidance on when to add microdispersion.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion import gamma_infiltration_to_extraction
    >>>
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>> cin = np.zeros(len(tedges) - 1)
    >>> cin[5:10] = 1.0
    >>> flow = np.ones(len(tedges) - 1) * 100.0
    >>>
    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=500.0,
    ...     std=100.0,
    ...     n_bins=5,
    ...     streamline_length=100.0,
    ...     molecular_diffusivity=1e-4,
    ...     longitudinal_dispersivity=1.0,
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        suppress_dispersion_warning=suppress_dispersion_warning,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
    )


def gamma_extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    streamline_length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    regularization_strength: float = 1e-10,
) -> npt.NDArray[np.floating]:
    """
    Compute infiltration concentration from extracted water for gamma-distributed pore volumes.

    Inverts the forward transport model (advection + dispersion with gamma-distributed
    pore volumes) via Tikhonov regularization. This is a convenience wrapper around
    :func:`extraction_to_infiltration` that parameterizes the aquifer pore volume
    distribution as a (shifted) gamma distribution.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Has length of len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Has length of len(cout) + 1.
    mean : float, optional
        Mean of the gamma distribution of the aquifer pore volume. Must be strictly
        greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma distribution of the aquifer pore volume
        (invariant under the ``loc`` shift).
    loc : float, optional
        Location (minimum pore volume) of the gamma distribution. Must satisfy
        ``0 <= loc < mean``. Default is ``0.0``.
    alpha : float, optional
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0).
    beta : float, optional
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0).
    n_bins : int, optional
        Number of bins to discretize the gamma distribution. Default is 100.
    streamline_length : float
        Travel distance through the aquifer [m]. Applied uniformly to all
        gamma-discretized pore volumes.
    molecular_diffusivity : float
        Effective molecular diffusivity [m2/day]. Must be non-negative.
        See :func:`infiltration_to_extraction` for details on the interaction
        with retardation_factor.
    longitudinal_dispersivity : float
        Longitudinal dispersivity [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Performance optimization for the forward matrix construction.
        Default is 3.0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-10.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Length equals
        len(tedges) - 1. NaN values indicate time periods with no valid
        contributions from the extraction data.

    See Also
    --------
    extraction_to_infiltration : Deconvolution with explicit pore volume distribution
    gamma_infiltration_to_extraction : Forward operation (convolution)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.advection.gamma_extraction_to_infiltration : Pure advection (no dispersion)
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The APVD is only time-invariant under the steady-streamlines assumption
    (see :ref:`assumption-steady-streamlines`).

    The spreading from the gamma-distributed pore volumes represents macrodispersion
    (aquifer-scale heterogeneity). When ``std`` comes from calibration on measurements,
    it absorbs all mixing: macrodispersion, microdispersion, and an average molecular
    diffusion contribution. When ``std`` comes from streamline analysis, it represents
    macrodispersion only; microdispersion and molecular diffusion can be added via the
    dispersion parameters.
    See :ref:`concept-dispersion-scales` for guidance on when to add microdispersion.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion import gamma_extraction_to_infiltration
    >>>
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>> cout = np.zeros(len(cout_tedges) - 1)
    >>> cout[5:10] = 1.0
    >>> flow = np.ones(len(tedges) - 1) * 100.0
    >>>
    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=500.0,
    ...     std=100.0,
    ...     n_bins=5,
    ...     streamline_length=100.0,
    ...     molecular_diffusivity=1e-4,
    ...     longitudinal_dispersivity=1.0,
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        suppress_dispersion_warning=suppress_dispersion_warning,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
        regularization_strength=regularization_strength,
    )
