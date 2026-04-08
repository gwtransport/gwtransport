"""
Utility functions for the Radial Push-Pull Well Transport Model.

Internal helper functions used by :mod:`gwtransport.radial`. These functions
build the coefficient matrices for pure advection and diffusion transport
in radial geometry.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy import special

from gwtransport.utils import partial_isin


def _push_pull_advection_matrix(
    *,
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool]]:
    """Build LIFO stack coefficient matrix for pure advection.

    Processes bins chronologically: injection bins push volume onto a stack,
    extraction bins pop from the stack top. The resulting weight matrix
    includes flow-weighted resampling onto the ``cout_tedges`` grid.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day]. Positive = injection, negative = extraction.
    dt : ndarray, shape (n,)
        Time bin widths [days].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days from reference.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days from reference.

    Returns
    -------
    weights : ndarray, shape (n_cout, n)
        Normalized weight matrix. Rows sum to 1 for output bins that fully
        recover injected volume, < 1 for over-extraction, and 0 for
        bins without extraction.
    has_extraction : ndarray, shape (n_cout,)
        Boolean mask indicating which output bins have extraction volume.
    """
    n = len(flow)
    n_cout = len(cout_tedges_days) - 1

    # Temporal overlap fractions: overlap[i, k] = fraction of fine bin i
    # that falls in cout bin k. Since flow is constant within a fine bin,
    # temporal fraction = volume fraction.
    overlap = partial_isin(bin_edges_in=tedges_days, bin_edges_out=cout_tedges_days)  # (n, n_cout)

    # Total extraction volume per cout bin (for normalization)
    is_extraction = flow < 0
    extraction_volume = np.abs(flow) * dt * is_extraction  # (n,)
    total_ext = (extraction_volume[:, np.newaxis] * overlap).sum(axis=0)  # (n_cout,)
    has_extraction = total_ext > 0

    # Accumulate raw consumed volumes directly into (n_cout, n) matrix.
    # For each consumed volume from the LIFO stack, distribute it across
    # cout bins proportional to how much of the fine extraction bin
    # overlaps with each cout bin.
    w_raw = np.zeros((n_cout, n))
    stack: list[list[float]] = []

    for i in range(n):
        vol = flow[i] * dt[i]
        if vol > 0:
            stack.append([float(i), vol])
        elif vol < 0:
            vol_needed = -vol
            overlap_i = overlap[i, :]  # (n_cout,)
            while vol_needed > 0 and stack:
                j_idx, vol_avail = stack[-1]
                consumed = min(vol_avail, vol_needed)
                w_raw[:, int(j_idx)] += consumed * overlap_i
                vol_needed -= consumed
                stack[-1][1] -= consumed
                if stack[-1][1] <= 0:
                    stack.pop()

    # Normalize per cout bin
    w = np.zeros((n_cout, n))
    w[has_extraction] = w_raw[has_extraction] / total_ext[has_extraction, np.newaxis]

    return w, has_extraction


def _signed_radial_distance(
    *,
    delta_volume: npt.NDArray[np.floating],
    layer_height: float,
    porosity: float,
    retardation_factor: float,
    n_layers: int,
) -> npt.NDArray[np.floating]:
    r"""Compute signed radial distance from volume difference.

    .. math::

        x = \text{sign}(\Delta V) \cdot
            \sqrt{\frac{|\Delta V|}{N \cdot \pi \cdot h \cdot n \cdot R}}

    Parameters
    ----------
    delta_volume : ndarray
        Volume difference [m³]. Positive means front has been extracted past
        the well screen.
    layer_height : float
        Height of a single horizontal streamtube flowing to the well
        screen [m].
    porosity : float
        Porosity n [-].
    retardation_factor : float
        Retardation factor R [-].
    n_layers : int
        Number of layers N.

    Returns
    -------
    ndarray
        Signed radial distance [m]. Same shape as delta_volume.
    """
    dv = np.asarray(delta_volume, dtype=float)
    scale = n_layers * np.pi * layer_height * porosity * retardation_factor
    return np.sign(dv) * np.sqrt(np.abs(dv) / scale)


# Gauss-Legendre quadrature nodes and weights for volume-space integration
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _erf_mean_volume_radial(
    *,
    v_cum: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    layer_height: float,
    porosity: float,
    retardation_factor: float,
    n_layers: int,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    v_max_after: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Compute mean erf in volume space for radial geometry.

    For each (extraction bin *i*, injection edge *j*), computes:

    .. math::

        \text{mean\_erf}_{i,j} = \frac{1}{|\Delta V_i|}
        \int_{V_i}^{V_{i+1}} \text{erf}\!\left(
            \frac{x(V)}{2\sqrt{\sigma^2/2}}
        \right) dV

    where *x(V)* is the signed radial distance from edge *j*'s front
    (positive = front already extracted past well, negative = front still
    in aquifer), and the cumulative dispersion variance is:

    .. math::

        \sigma^2/2 = D_m \, \tau(V) / R + \alpha_L \, L(V, j)

    - *D_m · τ / R* — molecular diffusion, proportional to time, divided
      by retardation (effective diffusivity = D_m / R in retarded ADE).
    - *alpha_L * L* -- mechanical dispersion, proportional to the total path
      length of the retarded front (out to r_max and partially back).

    The integral is evaluated by 16-point Gauss-Legendre quadrature.

    Parameters
    ----------
    v_cum : ndarray, shape (n+1,)
        Cumulative volume at time edges [m³].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days from reference.
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day].
    layer_height : float
        Height of a single horizontal streamtube flowing to the well
        screen [m].
    porosity : float
        Porosity n [-].
    retardation_factor : float
        Retardation factor R [-].
    n_layers : int
        Number of layers N.
    molecular_diffusivity : float
        Molecular diffusivity D_m [m²/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].
    v_max_after : ndarray, shape (n+1,)
        Maximum cumulative volume from each edge onward.

    Returns
    -------
    ndarray, shape (n, n_edges)
        Mean erf value for each (extraction bin, injection edge) pair.
        NaN for inactive or injection bins.
    """
    n = len(flow)
    n_edges = n + 1
    scale = n_layers * np.pi * layer_height * porosity * retardation_factor

    is_extraction = flow < 0
    mean_erf = np.full((n, n_edges), np.nan)

    if not np.any(is_extraction):
        return mean_erf

    v_lo = v_cum[:-1]  # shape (n,)
    v_hi = v_cum[1:]  # shape (n,)
    dv_bin = v_lo - v_hi  # positive for extraction
    active = is_extraction & (dv_bin > 0)

    if not np.any(active):
        return mean_erf

    no_diffusion = (molecular_diffusivity == 0.0) and (longitudinal_dispersivity == 0.0)

    if no_diffusion:
        v_lo_2d = v_lo[:, np.newaxis]
        v_hi_2d = v_hi[:, np.newaxis]
        v_j_2d = v_cum[np.newaxis, :]
        dv_2d = dv_bin[:, np.newaxis]

        with np.errstate(divide="ignore", invalid="ignore"):
            frac_pos = (v_lo_2d - v_j_2d) / dv_2d
        result = 1.0 - 2.0 * frac_pos
        result = np.where(v_j_2d <= v_hi_2d, -1.0, result)
        result = np.where(v_j_2d >= v_lo_2d, 1.0, result)
        mean_erf[active] = result[active]
        return mean_erf

    # --- With diffusion: Gauss-Legendre quadrature with position-dependent D_L ---

    # Maximum radial distance for each injection edge j
    dv_max_j = np.maximum(v_max_after - v_cum, 0.0)  # (n_edges,)
    r_max_j = np.sqrt(dv_max_j / scale)  # (n_edges,)

    # Active extraction bin indices
    idx_active = np.flatnonzero(active)

    # Map GL nodes from [-1, 1] to [V_hi, V_lo] for each active extraction bin.
    # ξ = -1 → V_hi, ξ = +1 → V_lo.
    v_lo_a = v_lo[idx_active]  # (n_active,)
    v_hi_a = v_hi[idx_active]  # (n_active,)
    dv_a = dv_bin[idx_active]  # (n_active,)

    v_mid = 0.5 * (v_lo_a + v_hi_a)
    v_half = 0.5 * dv_a
    v_gl = v_mid[:, np.newaxis] + v_half[:, np.newaxis] * _GL_NODES[np.newaxis, :]  # (n_active, n_gl)

    # --- Signed radial distance x(V, j) ---
    # Convention: x > 0 when front has been extracted past the well (V < V_j),
    # x < 0 when front is still in the aquifer (V > V_j).
    # This gives erf > 0 for extracted water → frac > 0.5.
    # Shape: (n_active, n_gl, n_edges)
    vj_minus_v = v_cum[np.newaxis, np.newaxis, :] - v_gl[:, :, np.newaxis]
    x_gl = np.sign(vj_minus_v) * np.sqrt(np.abs(vj_minus_v) / scale)

    # --- Elapsed time since injection at edge j: τ(V) = t(V) - t_j ---
    # t(V) = t_i + (V - V_i) / Q_i  within extraction bin i
    t_i = tedges_days[idx_active]  # (n_active,)
    q_i = flow[idx_active]  # (n_active,), negative for extraction
    t_gl = t_i[:, np.newaxis] + (v_gl - v_lo_a[:, np.newaxis]) / q_i[:, np.newaxis]  # (n_active, n_gl)
    tau_gl = t_gl[:, :, np.newaxis] - tedges_days[np.newaxis, np.newaxis, :]  # (n_active, n_gl, n_edges)

    # --- Path length of retarded front for edge j at quadrature point V ---
    # The front goes from r=0 to r_max_j during injection, then returns.
    # r_front(V) = sqrt(max(V - V_j, 0) / scale) is the current radial position.
    # L_path = r_max_j + (r_max_j - r_front) = 2·r_max_j - r_front
    v_minus_vj = v_gl[:, :, np.newaxis] - v_cum[np.newaxis, np.newaxis, :]  # (n_active, n_gl, n_edges)
    r_front = np.sqrt(np.maximum(v_minus_vj, 0.0) / scale)
    l_path = 2.0 * r_max_j[np.newaxis, np.newaxis, :] - r_front

    # --- Cumulative dispersion: sigma^2/2 = D_m*tau/R + alpha_L*L_path ---
    # From the retarded ADE (R*dC/dt + v*dC/dx = D_L*d2C/dx2):
    #   D_eff = D_L/R, and sigma^2 = 2*int(D_eff dt).
    # Molecular part: D_m*tau/R (constant D_m, integrates over time).
    # Mechanical part: alpha_L*L_path (int(alpha_L*v_pore/R dt)
    #   = alpha_L*int(|v_s|dt) = alpha_L*L_path,
    #   because int(v_pore dt) = R*L_path and dividing by R cancels).
    half_sigma_sq = molecular_diffusivity * tau_gl / retardation_factor + longitudinal_dispersivity * l_path

    # --- Evaluate erf(x / (2·√(σ²/2))) ---
    with np.errstate(divide="ignore", invalid="ignore"):
        safe_denom = np.where(half_sigma_sq > 0, half_sigma_sq, np.inf)
        arg = x_gl / (2.0 * np.sqrt(safe_denom))

    # Edge cases:
    # - τ ≤ 0: injection edge j is in the future → no contribution (erf = -1, frac = 0)
    # - half_sigma_sq = 0 with τ > 0: no diffusion for this edge → step function
    erf_val = np.where(
        tau_gl <= 0,
        -1.0,
        np.where(half_sigma_sq > 0, special.erf(arg), np.sign(x_gl)),
    )

    # GL quadrature: mean = Σ w_k · f(ξ_k) / 2
    mean_erf[idx_active] = np.einsum("ijk,j->ik", erf_val, _GL_WEIGHTS) / 2.0

    return mean_erf


def _push_pull_diffusion_matrix(
    *,
    flow: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    v_cum: npt.NDArray[np.floating],
    layer_height: float,
    porosity: float,
    retardation_factor: float,
    n_layers: int,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    v_max_after: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool]]:
    """Build coefficient matrix with radial diffusion for one layer.

    For each (cout_bin, cin_edge), computes flow-weighted mean erf via
    :func:`_erf_mean_volume_radial`. Converts to coefficients and applies
    flow-weighted resampling onto the ``cout_tedges`` grid.

    In the push-pull model, frac increases with edge index j (later injection
    = front closer to well = extracted sooner), so the coefficient is
    ``frac_end - frac_start`` (reversed from the linear model). Only injection
    bin columns are populated; extraction/rest columns remain zero.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days.
    v_cum : ndarray, shape (n+1,)
        Cumulative volume at time edges [m³].
    layer_height : float
        Height of a single horizontal streamtube flowing to the well
        screen [m].
    porosity : float
        Porosity n [-].
    retardation_factor : float
        Retardation factor R [-].
    n_layers : int
        Number of layers N.
    molecular_diffusivity : float
        Molecular diffusivity D_m [m²/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].
    v_max_after : ndarray, shape (n+1,)
        Maximum cumulative volume from each edge onward.

    Returns
    -------
    weights : ndarray, shape (n_cout, n)
        Normalized weight matrix on the cout_tedges grid.
    has_extraction : ndarray, shape (n_cout,)
        Boolean mask indicating which output bins have extraction volume.
    """
    n = len(flow)
    dt = np.diff(tedges_days)
    is_injection = flow > 0

    response = _erf_mean_volume_radial(
        v_cum=v_cum,
        tedges_days=tedges_days,
        flow=flow,
        layer_height=layer_height,
        porosity=porosity,
        retardation_factor=retardation_factor,
        n_layers=n_layers,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        v_max_after=v_max_after,
    )

    # Convert mean_erf (n, n+1) -> coefficients (n, n)
    # frac[i,j] = probability that water injected before edge j has been
    # extracted in bin i. In the push-pull LIFO model, frac increases with j
    # (later injection edges = closer to well = extracted sooner).
    frac = 0.5 * (1.0 + response)
    frac_filled = np.nan_to_num(frac, nan=0.0)

    # Reflecting boundary at the well screen (r = 0, V_cum = 0 at
    # edge 0): the erf model allows diffusion to r < 0, which is
    # unphysical.  Clamping frac at edge 0 to zero enforces the
    # reflecting boundary so that all tracer mass remains in the
    # domain and is attributed to injection bins.
    frac_filled[:, 0] = 0.0

    # Enforce monotonicity: frac must be non-decreasing along the
    # edge axis because later injection edges are closer to the well
    # and are extracted sooner.  Strong diffusion can cause local
    # non-monotonicity in the mean-erf values; the cumulative maximum
    # corrects this without introducing artificial mass.
    frac_filled = np.maximum.accumulate(frac_filled, axis=1)

    coeff = frac_filled[:, 1:] - frac_filled[:, :-1]
    coeff[:, ~is_injection] = 0.0

    # Flow-weighted resampling onto cout_tedges grid
    overlap = partial_isin(bin_edges_in=tedges_days, bin_edges_out=cout_tedges_days)  # (n, n_cout)
    is_extraction = flow < 0
    extraction_volume = np.abs(flow) * dt * is_extraction  # (n,)
    ext_vol_overlap = extraction_volume[:, np.newaxis] * overlap  # (n, n_cout)
    total_ext = ext_vol_overlap.sum(axis=0)  # (n_cout,)
    has_extraction = total_ext > 0

    n_cout = len(cout_tedges_days) - 1
    w = np.zeros((n_cout, n))
    # w[k,j] = Σ_i ext_vol_overlap[i,k] * coeff[i,j] / total_ext[k]
    w[has_extraction] = (ext_vol_overlap[:, has_extraction].T @ coeff) / total_ext[has_extraction, np.newaxis]

    return w, has_extraction
