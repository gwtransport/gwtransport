"""
Radial Push-Pull Well Transport Model.

This module provides functions to model solute transport in a single well with radial
flow in a vertically heterogeneous aquifer. The model handles bidirectional flow
(injection and extraction) through a LIFO (last-in-first-out) transport mechanism,
where the most recently injected water is extracted first.

The key difference from the linear infiltration-to-extraction models in
:mod:`gwtransport.advection` and :mod:`gwtransport.diffusion` is:

- **Bidirectional flow**: Water is injected and then extracted from the same well.
  Transport follows LIFO ordering — the last parcel injected is the first extracted.
- **Radial geometry**: The radial distance to the advective front scales as
  r = √(V / (N · π · h · n · R)), where V is cumulative volume, N is the
  number of layers, h is the layer height, n is porosity, and R is retardation.

**Heterogeneity model**: The aquifer is divided into N horizontal streamtubes
flowing to the well screen, each carrying equal flow but having a different
height h_i. Thinner streamtubes push the tracer front further radially for
the same injected volume, leading to more diffusive spreading and irreversible
mixing. Without diffusion, streamtube heterogeneity is invisible — all
streamtubes produce identical breakthrough (perfect LIFO recovery).

Available functions:

- :func:`push_pull_well` - Forward model: compute extraction concentration for a
  radial push-pull well given injection concentration and flow history.

- :func:`push_pull_well_inverse` - Inverse model: estimate injection concentration
  from measured extraction concentration using Tikhonov regularization.

- :func:`gamma_push_pull_well` - Convenience wrapper using gamma-distributed layer
  heights. Parameterizable via (alpha, beta) or (mean, std).

- :func:`gamma_push_pull_well_inverse` - Convenience wrapper for the inverse model
  with gamma-distributed layer heights.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special

from gwtransport import gamma
from gwtransport.utils import solve_inverse_transport


def _push_pull_advection_matrix(
    *,
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Build LIFO stack coefficient matrix for pure advection.

    Processes bins chronologically: injection bins push volume onto a stack,
    extraction bins pop from the stack top. The resulting matrix W satisfies
    ``cout = W @ cin``.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day]. Positive = injection, negative = extraction.
    dt : ndarray, shape (n,)
        Time bin widths [days].

    Returns
    -------
    ndarray, shape (n, n)
        Coefficient matrix. Rows sum to 1 for extraction bins that fully
        recover injected volume, < 1 for over-extraction, and 0 for
        injection/rest bins.
    """
    n = len(flow)
    w = np.zeros((n, n))
    # Stack entries: [bin_index, remaining_volume]
    stack: list[list[float]] = []

    for i in range(n):
        vol = flow[i] * dt[i]
        if vol > 0:
            # Injection: push onto stack
            stack.append([float(i), vol])
        elif vol < 0:
            # Extraction: pop from stack (LIFO)
            vol_needed = -vol
            extract_vol = vol_needed  # total extraction volume for normalization
            while vol_needed > 0 and stack:
                j_idx, vol_avail = stack[-1]
                consumed = min(vol_avail, vol_needed)
                w[i, int(j_idx)] += consumed / extract_vol
                vol_needed -= consumed
                stack[-1][1] -= consumed
                if stack[-1][1] <= 0:
                    stack.pop()
        # vol == 0: rest period, no action

    return w


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


def _compute_residence_times(
    *,
    v_cum: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute residence time for each injection edge.

    For each edge j, RT_j is the time from injection at t_j until extraction
    brings the cumulative volume back to v_cum[j].

    Parameters
    ----------
    v_cum : ndarray, shape (n+1,)
        Cumulative volume at time edges [m³].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days from reference.

    Returns
    -------
    ndarray, shape (n+1,)
        Residence time for each edge [days]. NaN if extraction never
        reaches v_cum[j].
    """
    n_edges = len(v_cum)
    rt = np.full(n_edges, np.nan)

    # v_max_after[j] = max(v_cum[j:])
    v_max_after = np.maximum.accumulate(v_cum[::-1])[::-1]

    for j in range(n_edges):
        v_j = v_cum[j]
        v_max_j = v_max_after[j]

        if v_max_j <= v_j:
            # Never pushed out past edge j
            rt[j] = 0.0
            continue

        # Find peak position (first time v_cum reaches v_max_j after j)
        peak_pos = j + np.argmax(v_cum[j:])

        # After peak, find where v_cum crosses back to v_j
        found = False
        for k in range(peak_pos + 1, n_edges):
            if v_cum[k] <= v_j:
                # Interpolate between k-1 and k
                dv_seg = v_cum[k] - v_cum[k - 1]
                if dv_seg == 0.0:
                    t_cross = tedges_days[k]
                else:
                    t_cross = tedges_days[k - 1] + (v_j - v_cum[k - 1]) / dv_seg * (tedges_days[k] - tedges_days[k - 1])
                rt[j] = t_cross - tedges_days[j]
                found = True
                break

        if not found:
            # Extraction never reaches v_j: use total elapsed time
            rt[j] = tedges_days[-1] - tedges_days[j]

    return rt


_N_SERIES_TERMS = 50
_J_ERFCX_PA2_THRESHOLD = 40.0
_J_OWENS_T_P_THRESHOLD = 500.0


def _jk_incomplete_gamma(
    p: npt.NDArray[np.floating],
    a: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    r"""Compute J(p, a) and K(p, a).

    Both are integrals of ``exp(-p t²)`` against rational weights:

    .. math::

        J(p, a) = \int_0^a \frac{e^{-p\,t^2}}{1 + t^2}\,dt \qquad
        K(p, a) = \int_0^a \frac{e^{-p\,t^2}}{1 - t^2}\,dt

    K (with 0 ≤ a < 1) is computed via the incomplete gamma series (all
    positive terms, numerically stable).

    J uses the incomplete gamma series as a starting point, then overrides
    elements where the alternating signs cause catastrophic cancellation:

    - **p·a² > 40**: The tail beyond *a* is < 10⁻¹⁷, so
      ``J ≈ (π/2)·erfcx(√p)`` (the closed-form full integral).
    - **p ≤ 500**: ``J = 2π·exp(p)·T(√(2p), a)`` via Owen's T function,
      which is numerically stable for all *a*.
    - **p > 500 and p·a² ≤ 40**: The series itself converges rapidly
      (terms decay as ``(k/p)^k``), so no override is needed.

    Parameters
    ----------
    p : ndarray
        Parameter p > 0.
    a : ndarray
        Parameter a >= 0. Broadcastable with p.

    Returns
    -------
    J : ndarray
        J(p, a) values.
    K : ndarray
        K(p, a) values.
    """
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    p, a = np.broadcast_arrays(p, a)

    pa2 = p * a**2
    n_terms = _N_SERIES_TERMS

    # Build coefficients: coeff[k] = Gamma(k+0.5) / (2 * p^{k+0.5})
    coeff_0 = np.sqrt(np.pi) / (2.0 * np.sqrt(p))
    half_ints = np.arange(1, n_terms) - 0.5  # [0.5, 1.5, 2.5, ...]
    ratios = half_ints.reshape((-1,) + (1,) * p.ndim) / p[np.newaxis, ...]
    coeffs = np.empty((n_terms, *p.shape))
    coeffs[0] = coeff_0
    coeffs[1:] = coeff_0 * np.cumprod(ratios, axis=0)

    signs = np.empty(n_terms)
    signs[0::2] = 1.0
    signs[1::2] = -1.0

    # Batch all gammainc calls: shape (n_terms,) + p.shape
    s_vals = np.arange(n_terms) + 0.5
    s_bc = s_vals.reshape((n_terms,) + (1,) * p.ndim)
    pa2_bc = pa2[np.newaxis, ...]
    gi = special.gammainc(s_bc, pa2_bc)

    terms = coeffs * gi

    # K: always stable (all positive terms)
    k_result = terms.sum(axis=0)

    # J: series as default, then override where unstable
    signs_bc = signs.reshape((n_terms,) + (1,) * p.ndim)
    j_result = (signs_bc * terms).sum(axis=0)

    # Override 1: erfcx for large pa² (tail integral negligible)
    m1 = pa2 > _J_ERFCX_PA2_THRESHOLD
    if np.any(m1):
        j_result[m1] = (np.pi / 2.0) * special.erfcx(np.sqrt(p[m1]))

    # Override 2: Owen's T for moderate p (avoids alternating-sign cancellation)
    m2 = ~m1 & (p <= _J_OWENS_T_P_THRESHOLD)
    if np.any(m2):
        h = np.sqrt(2.0 * p[m2])
        j_result[m2] = 2.0 * np.pi * np.exp(p[m2]) * special.owens_t(h, a[m2])

    return j_result, k_result


def _erf_antideriv_approaching(
    u: npt.NDArray[np.floating],
    p: npt.NDArray[np.floating],
    q: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Antiderivative F(u) for the V > V_j regime (erf argument has q - u).

    .. math::

        F(u) = -(q - u) \, \text{erf}\!\left(\sqrt{\frac{p \, u}{q - u}}\right)
        + 2 q \sqrt{\frac{p}{\pi}} \, J\!\left(p, \sqrt{\frac{u}{q - u}}\right)

    Valid for 0 <= u < q.

    Parameters
    ----------
    u : ndarray
        Integration variable, 0 <= u < q.
    p : ndarray
        Parameter p > 0. Broadcastable with u and q.
    q : ndarray
        Parameter q > 0. Broadcastable with u and p.

    Returns
    -------
    ndarray
        F(u) values.
    """
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    u, p, q = np.broadcast_arrays(u, p, q)

    result = np.zeros_like(u)
    mask = u > 0

    if np.any(mask):
        um, pm, qm = u[mask], p[mask], q[mask]
        diff = qm - um
        valid = diff > 0
        ratio = np.where(valid, um / diff, 0.0)
        sqrt_ratio = np.sqrt(ratio)

        erf_term = np.where(valid, -diff * special.erf(np.sqrt(pm * ratio)), -diff)
        j_val, _ = _jk_incomplete_gamma(np.maximum(pm, 1e-300), sqrt_ratio)
        j_term = 2.0 * qm * np.sqrt(pm / np.pi) * j_val
        result[mask] = erf_term + j_term

    return result


def _erf_antideriv_past(
    u: npt.NDArray[np.floating],
    p: npt.NDArray[np.floating],
    q: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Antiderivative G(u) for the V < V_j regime (erf argument has q + u).

    .. math::

        G(u) = (q + u) \, \text{erf}\!\left(\sqrt{\frac{p \, u}{q + u}}\right)
        - 2 q \sqrt{\frac{p}{\pi}} \, K\!\left(p, \sqrt{\frac{u}{q + u}}\right)

    Valid for u >= 0.

    Parameters
    ----------
    u : ndarray
        Integration variable, u >= 0.
    p : ndarray
        Parameter p > 0. Broadcastable with u and q.
    q : ndarray
        Parameter q > 0. Broadcastable with u and p.

    Returns
    -------
    ndarray
        G(u) values.
    """
    u = np.asarray(u, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    u, p, q = np.broadcast_arrays(u, p, q)

    result = np.zeros_like(u)
    mask = u > 0

    if np.any(mask):
        um, pm, qm = u[mask], p[mask], q[mask]
        ratio = um / (qm + um)
        sqrt_ratio = np.sqrt(ratio)

        erf_term = (qm + um) * special.erf(np.sqrt(pm * ratio))
        _, k_val = _jk_incomplete_gamma(pm, sqrt_ratio)
        k_term = 2.0 * qm * np.sqrt(pm / np.pi) * k_val
        result[mask] = erf_term - k_term

    return result


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
    rt_j: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Compute mean erf in volume space for radial geometry.

    For each (cout_bin i, cin_edge j), computes:

    .. math::

        \text{mean\_erf}_{i,j} = \frac{1}{|\Delta V_i|}
        \int_{V_i}^{V_{i+1}} \text{erf}\!\left(
            \frac{x(V)}{2\sqrt{D_L \cdot \tau(V)}}
        \right) dV

    where x(V) is the signed radial distance and D_L is a constant effective
    dispersion coefficient per injection edge. Uses exact antiderivatives
    based on incomplete gamma series for machine-precision evaluation.

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
    rt_j : ndarray, shape (n+1,)
        Residence time for each injection edge [days].

    Returns
    -------
    ndarray, shape (n_cout_bins, n_cin_edges)
        Mean erf value for each (extraction bin, injection edge) pair.
        NaN for inactive or injection bins.
    """
    n = len(flow)
    n_edges = n + 1
    scale = n_layers * np.pi * layer_height * porosity * retardation_factor

    # Identify extraction bins (flow < 0)
    is_extraction = flow < 0

    # Mean erf: shape (n, n_edges)
    mean_erf = np.full((n, n_edges), np.nan)

    if not np.any(is_extraction):
        return mean_erf

    # 2D grids: v_lo[i], v_hi[i], v_j[j]  →  broadcast to (n, n_edges)
    v_lo = v_cum[:-1]  # shape (n,)
    v_hi = v_cum[1:]  # shape (n,)
    dv_bin = v_lo - v_hi  # positive for extraction
    v_j_all = v_cum  # shape (n_edges,)

    # Masks for active extraction bins with nonzero volume
    active = is_extraction & (dv_bin > 0)  # shape (n,)

    # --- No diffusion: vectorized step function ---
    no_diffusion = (molecular_diffusivity == 0.0) and (longitudinal_dispersivity == 0.0)

    if no_diffusion:
        # Broadcast to (n, n_edges)
        v_lo_2d = v_lo[:, np.newaxis]
        v_hi_2d = v_hi[:, np.newaxis]
        v_j_2d = v_j_all[np.newaxis, :]
        dv_2d = dv_bin[:, np.newaxis]

        with np.errstate(divide="ignore", invalid="ignore"):
            frac_pos = (v_lo_2d - v_j_2d) / dv_2d
        result = 1.0 - 2.0 * frac_pos
        result = np.where(v_j_2d <= v_hi_2d, -1.0, result)
        result = np.where(v_j_2d >= v_lo_2d, 1.0, result)
        mean_erf[active] = result[active]
        return mean_erf

    # --- With diffusion: exact antiderivatives via incomplete gamma series ---

    # Pre-compute per injection edge j: dispersion coefficient
    dv_max = np.maximum(v_max_after - v_cum, 0.0)
    r_max_j = np.sqrt(dv_max / scale)
    path_len_j = 2.0 * r_max_j

    with np.errstate(divide="ignore", invalid="ignore"):
        velocity_j = np.where(rt_j > 0, path_len_j / rt_j, 0.0)
    d_l_j = molecular_diffusivity + longitudinal_dispersivity * velocity_j  # (n_edges,)

    # Build 2D arrays: (n, n_edges)
    v_lo_2d = v_lo[:, np.newaxis]
    v_hi_2d = v_hi[:, np.newaxis]
    v_j_2d = v_j_all[np.newaxis, :]
    dv_2d = dv_bin[:, np.newaxis]
    d_l_2d = d_l_j[np.newaxis, :]  # (1, n_edges)
    active_2d = active[:, np.newaxis] & np.ones(n_edges, dtype=bool)[np.newaxis, :]

    # Step function for D_L = 0 cells
    mask_step = active_2d & (d_l_2d == 0.0)
    if np.any(mask_step):
        with np.errstate(divide="ignore", invalid="ignore"):
            frac_pos = (v_lo_2d - v_j_2d) / dv_2d
        step_result = 1.0 - 2.0 * frac_pos
        step_result = np.where(v_j_2d <= v_hi_2d, -1.0, step_result)
        step_result = np.where(v_j_2d >= v_lo_2d, 1.0, step_result)
        mean_erf[mask_step] = step_result[mask_step]

    # Cells with D_L > 0
    mask_diff = active_2d & (d_l_2d > 0.0)
    if not np.any(mask_diff):
        return mean_erf

    # Flatten to 1D for vectorized antiderivative calls
    idx_i, idx_j = np.nonzero(mask_diff)
    n_cells = len(idx_i)

    p_flat = np.abs(flow[idx_i]) / (4.0 * scale * d_l_j[idx_j])
    # tau_0 = tedges_days[i] + (v_j - v_lo) / Q_i - tedges_days[j]
    tau_0 = tedges_days[idx_i] + (v_cum[idx_j] - v_cum[idx_i]) / flow[idx_i] - tedges_days[idx_j]
    q_flat = tau_0 * np.abs(flow[idx_i])

    vlo = v_cum[idx_i]
    vhi = v_cum[idx_i + 1]
    vj = v_cum[idx_j]
    dv = vlo - vhi

    # Regime masks (1D, length n_cells)
    mask_neg_q = q_flat < 0.0
    mask_past = (vj >= vlo) & ~mask_neg_q  # entire bin V <= V_j → G
    mask_appr = (vj <= vhi) & ~mask_neg_q & ~mask_past  # entire bin V >= V_j → -F
    mask_straddle = ~mask_neg_q & ~mask_past & ~mask_appr

    result_flat = np.full(n_cells, -1.0)  # default: -1 (handles mask_neg_q)

    # --- Batch all edge evaluations into single antiderivative calls ---
    # G antiderivative (past, V < V_j): needed at hi+lo edges for mask_past,
    # and at hi edge for mask_straddle.
    # F antiderivative (approaching, V > V_j): needed at hi+lo edges for
    # mask_appr (q>0), and at lo edge for mask_straddle (q>0).

    m_appr_q = mask_appr & (q_flat > 0.0)
    m_straddle_q = mask_straddle & (q_flat > 0.0)
    n_past = int(mask_past.sum())
    n_straddle = int(mask_straddle.sum())
    n_appr_q = int(m_appr_q.sum())
    n_straddle_q = int(m_straddle_q.sum())

    # Collect G evaluation points: [past_hi, past_lo, straddle_hi]
    g_u_parts = []
    g_p_parts = []
    g_q_parts = []
    if n_past > 0:
        m = mask_past
        g_u_parts.extend([vj[m] - vhi[m], vj[m] - vlo[m]])
        g_p_parts.extend([p_flat[m], p_flat[m]])
        g_q_parts.extend([q_flat[m], q_flat[m]])
    if n_straddle > 0:
        m = mask_straddle
        g_u_parts.append(vj[m] - vhi[m])
        g_p_parts.append(p_flat[m])
        g_q_parts.append(q_flat[m])

    if g_u_parts:
        g_all = _erf_antideriv_past(
            np.concatenate(g_u_parts),
            np.concatenate(g_p_parts),
            np.concatenate(g_q_parts),
        )

    # Collect F evaluation points: [appr_hi, appr_lo, straddle_lo]
    f_u_parts = []
    f_p_parts = []
    f_q_parts = []
    if n_appr_q > 0:
        f_u_parts.extend([vlo[m_appr_q] - vj[m_appr_q], vhi[m_appr_q] - vj[m_appr_q]])
        f_p_parts.extend([p_flat[m_appr_q], p_flat[m_appr_q]])
        f_q_parts.extend([q_flat[m_appr_q], q_flat[m_appr_q]])
    if n_straddle_q > 0:
        f_u_parts.append(vlo[m_straddle_q] - vj[m_straddle_q])
        f_p_parts.append(p_flat[m_straddle_q])
        f_q_parts.append(q_flat[m_straddle_q])

    if f_u_parts:
        f_all = _erf_antideriv_approaching(
            np.concatenate(f_u_parts),
            np.concatenate(f_p_parts),
            np.concatenate(f_q_parts),
        )

    # --- Distribute results ---
    if n_past > 0:
        result_flat[mask_past] = (g_all[:n_past] - g_all[n_past : 2 * n_past]) / dv[mask_past]

    if n_appr_q > 0:
        result_flat[m_appr_q] = -(f_all[:n_appr_q] - f_all[n_appr_q : 2 * n_appr_q]) / dv[m_appr_q]

    if n_straddle > 0:
        g_straddle = g_all[2 * n_past : 2 * n_past + n_straddle]

        has_q = q_flat[mask_straddle] > 0.0
        int_f = np.zeros(n_straddle)
        if n_straddle_q > 0:
            int_f[has_q] = -f_all[2 * n_appr_q : 2 * n_appr_q + n_straddle_q]
        if np.any(~has_q):
            int_f[~has_q] = -(vlo[mask_straddle][~has_q] - vj[mask_straddle][~has_q])

        result_flat[mask_straddle] = (g_straddle + int_f) / dv[mask_straddle]

    # Write back to 2D array
    mean_erf[idx_i, idx_j] = result_flat

    return mean_erf


def _push_pull_diffusion_matrix(
    *,
    flow: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    v_cum: npt.NDArray[np.floating],
    layer_height: float,
    porosity: float,
    retardation_factor: float,
    n_layers: int,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    v_max_after: npt.NDArray[np.floating],
    rt_j: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Build coefficient matrix with radial diffusion for one layer.

    For each (cout_bin, cin_edge), computes flow-weighted mean erf via
    :func:`_erf_mean_volume_radial`. Converts to coefficients.

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
    rt_j : ndarray, shape (n+1,)
        Residence time for each injection edge [days].

    Returns
    -------
    ndarray, shape (n, n)
        Coefficient matrix W such that ``cout = W @ cin``.
    """
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
        rt_j=rt_j,
    )

    # Convert mean_erf (n, n+1) -> coefficients (n, n)
    # frac[i,j] = probability that water injected before edge j has been
    # extracted in bin i. In the push-pull LIFO model, frac increases with j
    # (later injection edges = closer to well = extracted sooner).
    frac = 0.5 * (1.0 + response)
    frac_filled = np.nan_to_num(frac, nan=0.0)

    coeff = frac_filled[:, 1:] - frac_filled[:, :-1]
    coeff[:, ~is_injection] = 0.0

    return np.maximum(coeff, 0.0)


def push_pull_well(
    *,
    flow: npt.ArrayLike,
    cin: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    layer_heights: npt.ArrayLike,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute extraction concentration for a radial push-pull well.

    Models solute transport in a single well with radial flow in a vertically
    heterogeneous aquifer. Injection (flow > 0) pushes tracer outward; extraction
    (flow < 0) pulls it back in LIFO order.

    Without diffusion, all layers produce identical breakthrough (layer
    heterogeneity is invisible). With diffusion, layers with smaller height
    push the front further radially, causing more irreversible spreading.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cin : array-like, shape (n,)
        Injection concentration per time bin [concentration units].
        Values during extraction/rest bins are ignored.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    layer_heights : array-like, shape (N,)
        Height of each horizontal streamtube flowing to the well screen [m].
        The well screen is divided into N streamtubes of equal flow but
        different heights. Thinner streamtubes push the advective front
        further radially for the same injected volume, causing more
        diffusive spreading along the flow path.
    porosity : float
        Porosity n [-].
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water [concentration
        units]. When extraction pulls water from beyond the injection zone,
        the unaccounted fraction is assigned this concentration. Default is 0.

    Returns
    -------
    ndarray, shape (n,)
        Extraction concentration. NaN for injection and rest bins.

    Raises
    ------
    ValueError
        If flow and cin have different lengths, or tedges has wrong length.

    See Also
    --------
    push_pull_well_inverse : Estimate injection concentration from extraction data.
    gamma_push_pull_well : Convenience wrapper with gamma-distributed layer heights.
    """
    flow = np.asarray(flow, dtype=float)
    cin = np.asarray(cin, dtype=float)
    layer_heights = np.asarray(layer_heights, dtype=float)
    n = len(flow)

    if len(cin) != n:
        msg = f"flow and cin must have the same length, got {n} and {len(cin)}"
        raise ValueError(msg)
    if len(tedges) != n + 1:
        msg = f"tedges must have length len(flow) + 1, got {len(tedges)} and {n}"
        raise ValueError(msg)

    dt = np.diff(tedges) / pd.Timedelta("1D")

    # Prepend a large background injection to establish ambient aquifer
    # concentration. This ensures extraction beyond the injection zone pulls
    # water at c_background rather than implicitly zero.
    prepended = c_background != 0.0
    if prepended:
        v_cum_test = np.cumsum(flow * dt)
        max_abs_v = max(np.max(np.abs(np.concatenate(([0.0], v_cum_test)))), 1.0)
        prepend_volume = 1000.0 * max_abs_v
        flow = np.concatenate(([prepend_volume], flow))
        cin = np.concatenate(([c_background], cin))
        dt = np.concatenate(([1.0], dt))
        tedges = tedges.insert(0, tedges[0] - pd.Timedelta("1D"))  # type: ignore[assignment]
        n += 1

    n_layers = len(layer_heights)
    has_diffusion = molecular_diffusivity > 0 or longitudinal_dispersivity > 0

    if not has_diffusion:
        # Pure advection: LIFO stack, layer heterogeneity invisible
        w = _push_pull_advection_matrix(flow=flow, dt=dt)
    else:
        # With diffusion: average coefficient matrices across layers
        tedges_days = ((tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)
        v_cum = np.concatenate(([0.0], np.cumsum(flow * dt)))

        # Compute once: shared across all layers
        v_max_after = np.maximum.accumulate(v_cum[::-1])[::-1]
        rt_j = _compute_residence_times(v_cum=v_cum, tedges_days=tedges_days)

        w = np.zeros((n, n))
        for h in layer_heights:
            w_layer = _push_pull_diffusion_matrix(
                flow=flow,
                tedges_days=tedges_days,
                v_cum=v_cum,
                layer_height=h,
                porosity=porosity,
                retardation_factor=retardation_factor,
                n_layers=n_layers,
                molecular_diffusivity=molecular_diffusivity,
                longitudinal_dispersivity=longitudinal_dispersivity,
                v_max_after=v_max_after,
                rt_j=rt_j,
            )
            w += w_layer
        w /= n_layers

    # Compute output concentration
    cout = w @ cin
    is_extraction = flow < 0
    cout[~is_extraction] = np.nan

    if prepended:
        cout = cout[1:]

    return cout


def push_pull_well_inverse(
    *,
    flow: npt.ArrayLike,
    cout: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    layer_heights: npt.ArrayLike,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
    regularization_strength: float = 1e-3,
) -> npt.NDArray[np.floating]:
    """Estimate injection concentration from extraction measurements.

    Solves the inverse transport problem for the radial push-pull well model
    using Tikhonov regularization via :func:`gwtransport.utils.solve_inverse_transport`.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cout : array-like, shape (n,)
        Measured extraction concentration [concentration units].
        Values during injection/rest bins are ignored.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    layer_heights : array-like, shape (N,)
        Height of each horizontal streamtube flowing to the well screen [m].
        The well screen is divided into N streamtubes of equal flow but
        different heights. Thinner streamtubes push the advective front
        further radially for the same injected volume, causing more
        diffusive spreading along the flow path.
    porosity : float
        Porosity n [-].
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water [concentration
        units]. Default is 0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-3.

    Returns
    -------
    ndarray, shape (n,)
        Estimated injection concentration. NaN for extraction and rest bins.

    See Also
    --------
    push_pull_well : Forward model.
    gwtransport.utils.solve_inverse_transport : Tikhonov solver.
    """
    flow = np.asarray(flow, dtype=float)
    cout = np.asarray(cout, dtype=float)
    layer_heights = np.asarray(layer_heights, dtype=float)
    n = len(flow)

    dt = np.diff(tedges) / pd.Timedelta("1D")

    # Prepend background injection (same as forward model)
    prepended = c_background != 0.0
    if prepended:
        v_cum_test = np.cumsum(flow * dt)
        max_abs_v = max(np.max(np.abs(np.concatenate(([0.0], v_cum_test)))), 1.0)
        prepend_volume = 1000.0 * max_abs_v
        flow = np.concatenate(([prepend_volume], flow))
        cout = np.concatenate(([np.nan], cout))
        dt = np.concatenate(([1.0], dt))
        tedges = tedges.insert(0, tedges[0] - pd.Timedelta("1D"))  # type: ignore[assignment]
        n += 1

    n_layers = len(layer_heights)
    has_diffusion = molecular_diffusivity > 0 or longitudinal_dispersivity > 0

    if not has_diffusion:
        w = _push_pull_advection_matrix(flow=flow, dt=dt)
    else:
        tedges_days = ((tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)
        v_cum = np.concatenate(([0.0], np.cumsum(flow * dt)))

        # Compute once: shared across all layers
        v_max_after = np.maximum.accumulate(v_cum[::-1])[::-1]
        rt_j = _compute_residence_times(v_cum=v_cum, tedges_days=tedges_days)

        w = np.zeros((n, n))
        for h in layer_heights:
            w_layer = _push_pull_diffusion_matrix(
                flow=flow,
                tedges_days=tedges_days,
                v_cum=v_cum,
                layer_height=h,
                porosity=porosity,
                retardation_factor=retardation_factor,
                n_layers=n_layers,
                molecular_diffusivity=molecular_diffusivity,
                longitudinal_dispersivity=longitudinal_dispersivity,
                v_max_after=v_max_after,
                rt_j=rt_j,
            )
            w += w_layer
        w /= n_layers

    is_extraction = flow < 0
    is_injection = flow > 0

    if prepended:
        # Subtract known background column, solve reduced system
        bg_contribution = w[1:, 0] * c_background
        observed = np.where(is_extraction[1:], cout[1:] - bg_contribution, 0.0)
        w_reduced = w[1:, 1:]

        cin_recovered = solve_inverse_transport(
            w_forward=w_reduced,
            observed=observed,
            n_output=n - 1,
            regularization_strength=regularization_strength,
            valid_rows=is_extraction[1:],
        )

        out = np.full(n - 1, np.nan)
        idx = np.flatnonzero(is_injection[1:])
        out[idx] = cin_recovered[idx]
    else:
        cin_recovered = solve_inverse_transport(
            w_forward=w,
            observed=np.where(is_extraction, cout, 0.0),
            n_output=n,
            regularization_strength=regularization_strength,
            valid_rows=is_extraction,
        )

        out = np.full(n, np.nan)
        idx = np.flatnonzero(is_injection)
        out[idx] = cin_recovered[idx]

    return out


def gamma_push_pull_well(
    *,
    flow: npt.ArrayLike,
    cin: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute extraction concentration with gamma-distributed layer heights.

    Convenience wrapper around :func:`push_pull_well` that discretizes a gamma
    distribution into equal-probability bins for layer heights.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cin : array-like, shape (n,)
        Injection concentration per time bin [concentration units].
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    alpha : float, optional
        Shape parameter of gamma distribution for layer heights.
    beta : float, optional
        Scale parameter of gamma distribution for layer heights.
    mean : float, optional
        Mean layer height [m].
    std : float, optional
        Standard deviation of layer heights [m].
    n_bins : int, optional
        Number of bins for gamma discretization. Default is 100.
    porosity : float
        Porosity n [-].
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water [concentration
        units]. Default is 0.

    Returns
    -------
    ndarray, shape (n,)
        Extraction concentration. NaN for injection and rest bins.

    See Also
    --------
    push_pull_well : Base function with explicit layer heights.
    gwtransport.gamma.bins : Gamma distribution discretization.
    """
    gamma_bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    layer_heights = gamma_bins["expected_values"]

    return push_pull_well(
        flow=flow,
        cin=cin,
        tedges=tedges,
        layer_heights=layer_heights,
        porosity=porosity,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        c_background=c_background,
    )


def gamma_push_pull_well_inverse(
    *,
    flow: npt.ArrayLike,
    cout: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
    regularization_strength: float = 1e-3,
) -> npt.NDArray[np.floating]:
    """Estimate injection concentration with gamma-distributed layer heights.

    Convenience wrapper around :func:`push_pull_well_inverse` that discretizes
    a gamma distribution into equal-probability bins for layer heights.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cout : array-like, shape (n,)
        Measured extraction concentration [concentration units].
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    alpha : float, optional
        Shape parameter of gamma distribution for layer heights.
    beta : float, optional
        Scale parameter of gamma distribution for layer heights.
    mean : float, optional
        Mean layer height [m].
    std : float, optional
        Standard deviation of layer heights [m].
    n_bins : int, optional
        Number of bins for gamma discretization. Default is 100.
    porosity : float
        Porosity n [-].
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water [concentration
        units]. Default is 0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-3.

    Returns
    -------
    ndarray, shape (n,)
        Estimated injection concentration. NaN for extraction and rest bins.

    See Also
    --------
    push_pull_well_inverse : Base function with explicit layer heights.
    gwtransport.gamma.bins : Gamma distribution discretization.
    """
    gamma_bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    layer_heights = gamma_bins["expected_values"]

    return push_pull_well_inverse(
        flow=flow,
        cout=cout,
        tedges=tedges,
        layer_heights=layer_heights,
        porosity=porosity,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        c_background=c_background,
        regularization_strength=regularization_strength,
    )
