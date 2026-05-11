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

The breakthrough kernel implemented here is the variable-flow Ogata-Banks
solution from Bear (1972), Dynamics of Fluids in Porous Media, eq. 10.6.4:

    epsilon(x, t) = (C(x, t) - C0) / (C1 - C0)
                  = 0.5 * erfc{ -[x - integral_0^t (q(t)/n) dt]
                                / (2 * sqrt[integral_0^t (alpha_I |q|/n + D_d T*) dt]) }

with the dispersion variance accumulated in the moving (Lagrangian) frame:

    sigma^2(V) = 2 * D_m * tau(V) + 2 * alpha_L * xi(V)

where:

- D_m is the molecular (or thermal) effective diffusivity D_d * T* [m^2/day]
- alpha_L is the longitudinal dispersivity alpha_I [m]
- tau(V) is the elapsed time since infiltration [day], with V the cumulative
  extracted volume; equivalent to t in Bear's notation
- xi(V) is the distance the parcel has actually travelled [m], equivalent to
  ``integral_0^t q/n dt`` in Bear's notation

This formulation ensures that:

- Molecular diffusion spreading scales with residence time: sqrt(D_m * tau)
- Mechanical dispersion spreading scales with travel distance: sqrt(alpha_L * xi)

Evaluating sigma^2 directly at each volume node — rather than via a per-bin
scalar D_L = D_m + alpha_L * v_s built from a mean velocity — keeps the
breakthrough kernel a function only of V (and the infiltration edge), which
is what mass conservation requires when Q varies in time.

This module adds microdispersion (alpha_L) and molecular diffusion (D_m) on top of
macrodispersion captured by the pore volume distribution (APVD). Both represent velocity
heterogeneity at different scales. Microdispersion is an aquifer property; macrodispersion
depends additionally on hydrological boundary conditions. See :ref:`concept-dispersion-scales`
for guidance on when to use each approach and how to avoid double-counting spreading effects.

References
----------
Bear, J. (1972). Dynamics of Fluids in Porous Media. American Elsevier
Publishing Company. Equation 10.6.4 (variable-flow Ogata-Banks form).
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special

from gwtransport import gamma
from gwtransport.residence_time import residence_time
from gwtransport.utils import solve_inverse_transport

# Numerical tolerance for coefficient sum to determine valid output bins
EPSILON_COEFF_SUM = 1e-10

# Gauss-Legendre quadrature nodes and weights for volume-space integration
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _erf_integral_space(
    x: npt.NDArray[np.float64],
    *,
    dispersion: npt.ArrayLike,
    clip_to_inf: float = 6.0,
) -> npt.NDArray[np.float64]:
    r"""Compute the integral of the error function in space at each (x[i], dispersion[i]) point.

    Evaluates

    .. math:: \int_0^{x_i} \text{erf}\!\left(\frac{s}{2\sqrt{K_i}}\right) ds

    where ``K = dispersion = D \cdot t = \sigma^2 / 2`` collapses the time
    and diffusivity dependence into a single dispersion product. This form
    lets the caller assemble the variance from arbitrary contributions
    (e.g. ``D_m \cdot \tau + \alpha_L \cdot \xi``) without forcing the
    kernel to expose separate ``D`` and ``t`` parameters.

    Parameters
    ----------
    x : ndarray
        Upper limits of integration. Broadcastable with ``dispersion``.
    dispersion : float or ndarray
        Dispersion product ``D * t`` [m²], equivalently ``sigma^2 / 2``.
        Must be non-negative. Broadcastable with ``x``.
    clip_to_inf : float, optional
        Clip ``x / (2*sqrt(K))`` values beyond this magnitude to the
        asymptotic form. Default is 6.

    Returns
    -------
    ndarray
        Integral values. Shape is the broadcast of inputs.
    """
    x = np.asarray(x)
    dispersion = np.asarray(dispersion)

    x, dispersion = np.broadcast_arrays(x, dispersion)
    out = np.zeros_like(x, dtype=float)

    # a = 1 / (2*sqrt(K)) — equivalent to a = 1/(sqrt(2)*sigma)
    # K = 0 → a = inf (step function limit)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(dispersion == 0.0, np.inf, 1.0 / (2.0 * np.sqrt(dispersion)))

    # K = 0: integral of sign(s) from 0 to x equals |x|
    mask_a_inf = np.isinf(a)
    if np.any(mask_a_inf):
        out[mask_a_inf] = np.abs(x[mask_a_inf])

    mask_valid = ~mask_a_inf
    if np.any(mask_valid):
        x_v = x[mask_valid]
        a_v = a[mask_valid]
        ax = a_v * x_v

        result = np.zeros_like(x_v)

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
    step_widths: npt.NDArray[np.float64],
    raw_time: npt.NDArray[np.float64],
    rt_at_cin_edges: npt.NDArray[np.float64],
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    cumulative_volume_at_cout_tedges: npt.NDArray[np.float64],
    cumulative_volume_at_cin_tedges: npt.NDArray[np.float64],
    tedges_days: npt.NDArray[np.float64],
    r_vpv: float,
    streamline_len: float,
    asymptotic_cutoff_sigma: float | None,
) -> npt.NDArray[np.float64]:
    r"""Compute mean erf along the physical trajectory in cumulative volume space.

    For each cell (cout_bin *i*, cin_edge *j*), computes the flow-weighted
    average of the error function along the 1D extraction trajectory:

    .. math::

        \text{mean\_erf}_{i,j} = \frac{1}{\Delta V_i}
        \int_{V_i}^{V_{i+1}} \text{erf}\!\left(
            \frac{x(V)}{\sqrt{2\,\sigma^2(V)}}
        \right) dV

    where *x(V)* is the normalized distance (linear in *V*), *τ(V)* is the
    elapsed time since infiltration (capped at the residence time *RT*),
    *ξ(V) = x(V) + L* is the distance the parcel has actually travelled,
    and the dispersion variance is

    .. math::

        \sigma^2(V) = 2\,D_m\,\tau(V) + 2\,\alpha_L\,\xi(V).

    Averaging over cumulative volume gives a flow-weighted average because
    dV = Q(t) dt. Evaluating ``sigma^2(V)`` directly at each quadrature node —
    rather than via a per-cell scalar ``D_L = D_m + alpha_L * L / (R tau_mean)``
    — keeps the kernel a function only of *V* and the infiltration edge *j*,
    which is what mass conservation requires when ``Q`` varies.

    **Fully capped cells** (both cout edges have elapsed time ≥ RT):
    tau = RT and xi = L, so sigma^2 is constant and the integral reduces to
    ``_erf_integral_space`` at fixed dispersion product
    ``Dt = D_m * RT + alpha_L * L``. This path gives machine precision.

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
    molecular_diffusivity : float
        Molecular (or thermal) diffusivity for this pore volume [m²/day].
        Contributes ``2 D_m τ`` to the variance.
    longitudinal_dispersivity : float
        Longitudinal dispersivity for this pore volume [m]. Contributes
        ``2 * alpha_L * xi`` to the variance, where ``xi`` is the distance
        the parcel has actually travelled.
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

    # --- No dispersion: erf = sign(x), exact for any tau, xi ---
    if molecular_diffusivity == 0.0 and longitudinal_dispersivity == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            d_zero_vals = (np.abs(x_hi) - np.abs(x_lo)) / dx
        d_zero_vals = np.where(dx == 0.0, np.sign(x_lo), d_zero_vals)
        return np.where(is_valid, d_zero_vals, response)

    # --- Fully capped cells: sigma^2 = 2 D_m RT_j + 2 alpha_L L, constant over the cell ---
    mask_capped = is_valid & is_fully_capped
    if np.any(mask_capped):
        rt_capped = np.broadcast_to(rt_at_cin_edges[np.newaxis, :], (n_cout_bins, n_cin_edges))[mask_capped]
        # Dispersion product Dt = σ²/2 at the fully-transited parcel:
        # tau = RT_j (fully capped) and xi = streamline_len.
        dt_capped = molecular_diffusivity * rt_capped + longitudinal_dispersivity * streamline_len

        clip_kw = {"clip_to_inf": asymptotic_cutoff_sigma} if asymptotic_cutoff_sigma is not None else {}
        h_hi = _erf_integral_space(x_hi[mask_capped], dispersion=dt_capped, **clip_kw)
        h_lo = _erf_integral_space(x_lo[mask_capped], dispersion=dt_capped, **clip_kw)
        dx_c = dx[mask_capped]

        with np.errstate(divide="ignore", invalid="ignore"):
            a_vals = np.where(dt_capped <= 0.0, np.inf, 1.0 / (2.0 * np.sqrt(dt_capped)))
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

            # x(V): (n_overlap, n_gl), position relative to the outlet
            x_nodes = (v_nodes - v_cin_cells[overlap, np.newaxis] - r_vpv) * streamline_len / r_vpv

            # ξ(V) = x(V) + L: distance the parcel has actually travelled
            xi_nodes = x_nodes + streamline_len

            # t(V): within flow sub-interval k, Q is constant so t is linear in V
            dt_sub = tedges_days[k + 1] - tedges_days[k]
            dv_sub_edge = ve_hi - ve_lo
            t_nodes = tedges_days[k] + (v_nodes - ve_lo) * (dt_sub / dv_sub_edge)

            # τ(V) = clip(t(V) - t_j, 0, RT_j): (n_overlap, n_gl)
            tau_nodes = np.clip(t_nodes - t_j_cells[overlap, np.newaxis], 0.0, rt_cells[overlap, np.newaxis])

            # Dispersion product Dt(V) = sigma^2(V)/2 = D_m * tau(V) + alpha_L * xi(V).
            # Evaluating sigma^2 directly at each node — rather than via a scalar
            # D_L = D_m + alpha_L * L / (R * tau_mean) — makes the kernel a
            # function only of V (and the cin edge j), restoring exact mass
            # conservation under variable Q.
            dt_nodes = molecular_diffusivity * tau_nodes + longitudinal_dispersivity * xi_nodes

            # erf(x / (2*sqrt(Dt))) = erf(x / sqrt(2 σ²)): (n_overlap, n_gl)
            with np.errstate(divide="ignore", invalid="ignore"):
                arg = x_nodes / (2.0 * np.sqrt(dt_nodes))
            erf_vals = np.where(np.isfinite(arg), special.erf(arg), np.sign(x_nodes))

            # Accumulate weighted integral for overlapping cells
            uncapped_integral[overlap] += (dv_sub / 2) * (erf_vals @ _GL_WEIGHTS)

        # --- Capped sub-interval [v_kink, v_hi]: exact via _erf_integral_space.
        # Beyond the kink the parcel has fully exited, so sigma^2 is constant at
        # 2 * D_m * RT_j + 2 * alpha_L * L for cells in this branch.
        capped_integral = np.zeros(n_cells)
        mask_cap = has_capped & (v_kink_all < v_hi_cells) & valid_cells
        if np.any(mask_cap):
            x_at_kink = (v_kink_all[mask_cap] - v_cin_cells[mask_cap] - r_vpv) * streamline_len / r_vpv
            x_at_hi_cap = x_hi[idx_i[mask_cap], idx_j[mask_cap]]
            dt_cap = molecular_diffusivity * rt_cells[mask_cap] + longitudinal_dispersivity * streamline_len
            clip_kw = {"clip_to_inf": asymptotic_cutoff_sigma} if asymptotic_cutoff_sigma is not None else {}
            h_hi_val = _erf_integral_space(x_at_hi_cap, dispersion=dt_cap, **clip_kw)
            h_lo_val = _erf_integral_space(x_at_kink, dispersion=dt_cap, **clip_kw)
            capped_integral[mask_cap] = (h_hi_val - h_lo_val) * (r_vpv / streamline_len)

        # Combine: mean erf = (uncapped + capped) / total_dv
        with np.errstate(divide="ignore", invalid="ignore"):
            response_cells = np.where(valid_cells, (uncapped_integral + capped_integral) / total_dv, np.nan)
        response[idx_i, idx_j] = response_cells

    return response


def _diffusion_extend_tedges_flag(spinup: object) -> bool:
    """Translate the public ``spinup`` parameter to the internal extend flag.

    The diffusion module's existing warm-start behavior is to extend
    ``tedges`` by 100 years on each side. The public ``spinup`` parameter
    maps onto this binary toggle: ``"constant"`` enables the extension
    (default; preserves legacy behavior), ``None`` disables it (cout in
    spin-up region becomes NaN). The float fraction-threshold mode of
    other modules is not implemented here.

    Returns
    -------
    bool
        True if tedges should be extended (warm-start), False if not.

    Raises
    ------
    ValueError
        If ``spinup`` is a string other than ``"constant"``.
    NotImplementedError
        If ``spinup`` is a float (fraction-threshold mode is not
        implemented for the diffusion module).
    """
    if spinup is None:
        return False
    if isinstance(spinup, str):
        if spinup != "constant":
            msg = f"spinup string must be 'constant'; got {spinup!r}"
            raise ValueError(msg)
        return True
    msg = (
        "diffusion's spinup parameter only supports None or 'constant'; "
        f"float thresholds are not yet implemented (got {spinup!r})"
    )
    raise NotImplementedError(msg)


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
    extend_tedges: bool = True,
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
    if extend_tedges:
        # Extend tedges by 100 years on each side to provide warm-start cin
        # and post-data flow for cout bins near the boundaries. Equivalent to
        # the public ``spinup="constant"`` policy in other modules.
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

        response = _erf_mean_volume(
            step_widths=step_widths,
            raw_time=raw_time,
            rt_at_cin_edges=rt_edges_2d[i_pv],
            molecular_diffusivity=float(molecular_diffusivity[i_pv]),
            longitudinal_dispersivity=float(longitudinal_dispersivity[i_pv]),
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
    spinup: str | None = "constant",
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

    Longitudinal dispersion enters as the moving-frame variance

        sigma^2(V) = 2 * D_m * tau(V) + 2 * alpha_L * xi(V),

    where ``tau(V)`` is the elapsed time since infiltration, ``xi(V)`` is the
    distance the parcel has actually travelled, and V is the cumulative
    extracted volume. The breakthrough kernel is the Gaussian CDF in V-space
    using this variance. Evaluating sigma^2 directly at each volume node —
    rather than via a per-cout-bin scalar D_L = D_m + alpha_L * L / (R * tau_mean)
    — makes the kernel a function only of V (and the cin edge), restoring
    exact mass conservation under variable flow.

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

        Internally, this contributes ``2 * molecular_diffusivity * tau`` to the
        variance, where ``tau`` is the elapsed time in days (no extra factor of
        R). For heat transport, the thermal diffusivity already represents the
        effective diffusivity D_eff in the porous matrix; for solutes the
        contribution is typically negligible.
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
    if np.any(flow < 0):
        msg = "flow must be non-negative (negative flow not supported)"
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

    extend_tedges = _diffusion_extend_tedges_flag(spinup)
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
        extend_tedges=extend_tedges,
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
    spinup: str | None = "constant",
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
    extend_tedges = _diffusion_extend_tedges_flag(spinup)
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
        extend_tedges=extend_tedges,
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
    spinup: str | None = "constant",
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
        spinup=spinup,
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
    spinup: str | None = "constant",
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
        spinup=spinup,
    )
