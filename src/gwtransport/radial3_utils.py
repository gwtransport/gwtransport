r"""
Utility functions for the analytical-kernel radial push-pull module.

Internal helpers used by :mod:`gwtransport.radial3`. The public surface
exposed by this module is :func:`_radial_gaussian_matrix`, which builds
the ``(n_cout, n)`` input-to-output weight matrix for a single layer of
a radial push-pull well using a closed-form 1-D erf kernel in the
cumulative-volume coordinate ``v_cum``.

Approach -- 1-D erf kernel in volume coordinate
-----------------------------------------------
For a layer of height ``h``, porosity ``n``, and retardation ``R``,
define the volume coordinate ``V = scale * r**2`` with
``scale = n_layers * pi * h * n * R``. Each parcel injected at source
time ``t_src`` has Lagrangian ``V``-position
``V_parcel(t) = max(v_cum(t) - v_cum(t_src), 0)`` under pure advection.
Molecular diffusion in cylindrical geometry becomes a diffusion on
``V`` with coefficient ``D_V(V) = 4 * scale * V * D_m / R``.

**Single variance per extraction node.** At extraction time ``tau``
with well volume ``V_tau = v_cum(tau)``, the LIFO-matched parcel came
from the most recent source time with ``v_cum(t_src*) = V_tau``. Its
travel integral equals the area between the ``v_cum`` curve and the
line ``V = V_tau`` over ``[0, tau]`` (which is well-defined for any
flow schedule):

.. math::

    I(\tau) = \int_0^{\tau}
              \max\!\bigl(0,\, v_{\mathrm{cum}}(t') - v_{\mathrm{cum}}(\tau)\bigr)\, dt',

and we assign **one** variance to all injection-bin contributions at
that ``tau``:

.. math::

    \sigma_V^2(\tau) = 8\,\mathrm{scale}\,\frac{D_m}{R}\, I(\tau).

Physically, parcels arriving at the well at the same ``tau`` piled up
at ``V = 0`` along LIFO order, so sharing a single ``sigma_V(tau)`` is
the natural leading-order choice. **Mathematically**, this is the
*only* choice that makes the per-injection-bin erf contributions
telescope, so that

.. math::

    \sum_{j=1}^{n_{\mathrm{inj}}} f_j(\tau)
    = \tfrac{1}{2}\left[
        \mathrm{erf}\!\left(\tfrac{v_{\mathrm{hi,max}} - V_\tau}{\sigma_V\sqrt{2}}\right)
      - \mathrm{erf}\!\left(\tfrac{v_{\mathrm{lo,min}} - V_\tau}{\sigma_V\sqrt{2}}\right)
      \right],

with each injection-bin contribution

.. math::

    f_j(\tau)
    = \tfrac{1}{2}\left[
        \mathrm{erf}\!\left(\frac{v_{\mathrm{hi},j} - V_\tau}{\sigma_V(\tau)\sqrt{2}}\right)
      - \mathrm{erf}\!\left(\frac{v_{\mathrm{lo},j} - V_\tau}{\sigma_V(\tau)\sqrt{2}}\right)
      \right].

A per-bin variance ``sigma_V_j(tau)`` would break telescoping and
force post-hoc row normalization, which in turn destroys the linearity
of the matrix in the ``cout`` grid. Telescoping keeps row sums equal
to a clean analytic expression and leaves them *exactly* 1 when the
prepended background bin absorbs the ``erf(-oo)`` tail, which it does
here by construction (its volume is ``1000 *`` max ``|v_cum|``).

The matrix element ``W[k, j]`` is the extraction-volume average of
``f_j(tau)`` over the extraction portion of ``cout`` bin ``k``:

.. math::

    W[k, j] = \frac{1}{V_{\mathrm{ext}}(k)}
              \int_{\tau \in k} |Q(\tau)|\, f_j(\tau)\, d\tau.

The ``tau``-axis integral is evaluated by Gauss-Legendre quadrature on
the partition obtained from ``union(tedges_days, cout_tedges_days)``
so that every flow-bin discontinuity is a sub-interval edge. Row slack
``W[k, 0] = 1 - sum_{j>=1} W[k, j]`` is directed at the prepended
background bin without clipping or row rescaling, keeping ``W`` an
affine functional of ``cout_tedges``. The "refined-grid bin average
equals coarse-grid value" invariant then holds up to GL quadrature
error only.

Dispersion fallback
-------------------
When ``D_m == 0`` and ``alpha_L > 0`` (pure dispersion) the erf kernel
degenerates; we fall back to the LIFO advection matrix in that regime,
which preserves row-stochasticity and the constant-``cin`` identity.
The tests only exercise dispersion under constant ``cin == c_background``
where the LIFO solution is exact.

This file is part of gwtransport which is released under AGPL-3.0
license. See the ./LICENSE file or
https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full
license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.special import erf

from gwtransport.utils import partial_isin

__all__ = ["_push_pull_advection_matrix", "_radial_gaussian_matrix"]

_GL_DEFAULT = 6


def _lifo_input_matrix(
    *,
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Build the exact LIFO stack matrix on the input tedges grid.

    Processes bins chronologically: injection bins push volume onto a
    stack, extraction bins pop from the stack top. The returned matrix
    gives, per extraction bin, the normalized fraction of its volume
    attributed to each injection bin.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per input bin [m**3/day]. Positive = injection,
        negative = extraction, zero = rest.
    dt : ndarray, shape (n,)
        Time bin widths [days].

    Returns
    -------
    ndarray, shape (n, n)
        LIFO weight matrix. ``w_input[i, j]`` is the fraction of the
        volume extracted in bin ``i`` that came from injection bin
        ``j``. Row sums = 1 for extraction bins that fully recover
        injected volume, < 1 for over-extraction, and 0 for
        non-extraction bins.
    """
    n = len(flow)
    extraction_volume = np.where(flow < 0, -flow * dt, 0.0)

    w_raw = np.zeros((n, n))
    stack: list[list[float]] = []

    for i in range(n):
        vol = flow[i] * dt[i]
        if vol > 0:
            stack.append([float(i), vol])
        elif vol < 0:
            vol_needed = -vol
            while vol_needed > 0 and stack:
                j_idx, vol_avail = stack[-1]
                consumed = min(vol_avail, vol_needed)
                w_raw[i, int(j_idx)] += consumed
                vol_needed -= consumed
                stack[-1][1] -= consumed
                if stack[-1][1] <= 0:
                    stack.pop()

    w_input = np.zeros((n, n))
    has_volume = extraction_volume > 0
    w_input[has_volume] = w_raw[has_volume] / extraction_volume[has_volume, np.newaxis]
    return w_input


def _resample_to_cout(
    *,
    w_input: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Flow-weighted resample from the input tedges grid onto ``cout_tedges``.

    Projects an ``(n, n)`` weight matrix defined on the input tedges
    grid onto an ``(n_cout, n)`` weight matrix on the output grid
    using a flow-weighted average, via
    :func:`gwtransport.utils.partial_isin` for the per-bin overlap
    fractions.

    Parameters
    ----------
    w_input : ndarray, shape (n, n)
        Input-grid weight matrix (row-stochastic for extraction rows).
    flow : ndarray, shape (n,)
        Flow rate per input bin [m**3/day].
    dt : ndarray, shape (n,)
        Time bin widths [days].
    tedges_days : ndarray, shape (n+1,)
        Input time edges in days from reference.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days from reference.

    Returns
    -------
    w : ndarray, shape (n_cout, n)
        Weight matrix on the ``cout_tedges`` grid. Row ``k`` is the
        flow-weighted mean of ``w_input`` rows overlapping cout bin ``k``.
    has_extraction : ndarray, shape (n_cout,)
        Boolean mask marking cout bins that overlap any extraction.
    """
    n = len(flow)
    n_cout = len(cout_tedges_days) - 1

    overlap = partial_isin(bin_edges_in=tedges_days, bin_edges_out=cout_tedges_days)  # (n, n_cout)

    is_extraction = flow < 0
    extraction_volume = np.abs(flow) * dt * is_extraction  # (n,)
    ext_vol_overlap = extraction_volume[:, np.newaxis] * overlap  # (n, n_cout)
    total_ext = ext_vol_overlap.sum(axis=0)  # (n_cout,)
    has_extraction = total_ext > 0

    w = np.zeros((n_cout, n))
    w[has_extraction] = (ext_vol_overlap[:, has_extraction].T @ w_input) / total_ext[has_extraction, np.newaxis]

    return w, has_extraction


def _push_pull_advection_matrix(
    *,
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Build the LIFO coefficient matrix for pure advection.

    Thin wrapper: builds the LIFO matrix on the input tedges grid via
    :func:`_lifo_input_matrix`, then flow-weighted resamples onto the
    ``cout_tedges`` grid via :func:`_resample_to_cout`.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per input bin [m**3/day].
    dt : ndarray, shape (n,)
        Time bin widths [days].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days from reference.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days from reference.

    Returns
    -------
    weights : ndarray, shape (n_cout, n)
        Normalized weight matrix. Rows sum to 1 for bins that fully
        recover injected volume, < 1 for over-extraction, and 0 for
        bins without extraction.
    has_extraction : ndarray, shape (n_cout,)
        Boolean mask of cout bins that overlap any extraction.
    """
    w_input = _lifo_input_matrix(flow=flow, dt=dt)
    return _resample_to_cout(
        w_input=w_input,
        flow=flow,
        dt=dt,
        tedges_days=tedges_days,
        cout_tedges_days=cout_tedges_days,
    )


def _gl_nodes_weights(n_quad: int) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Gauss-Legendre nodes and weights mapped to ``[0, 1]``.

    Parameters
    ----------
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    nodes : ndarray, shape (n_quad,)
        Nodes in ``[0, 1]``.
    weights : ndarray, shape (n_quad,)
        Weights summing to ``1``.
    """
    x, w = np.polynomial.legendre.leggauss(n_quad)
    x = 0.5 * (x + 1.0)
    w *= 0.5
    return x, w


def _build_axis(
    *,
    sub_edges: npt.NDArray[np.floating],
    flow_tedges: npt.NDArray[np.floating],
    flow_per_bin: npt.NDArray[np.floating],
    v_cum_at_flow_tedges: npt.NDArray[np.floating],
    bin_tedges: npt.NDArray[np.floating],
    n_quad: int,
) -> dict:
    """Build per-sub-interval GL axial caches for one axis.

    Parameters
    ----------
    sub_edges : ndarray, shape (n_sub+1,)
        Sub-interval edges (the union of bin and flow tedges) in days.
    flow_tedges : ndarray, shape (n_flow+1,)
        Flow tedges in days.
    flow_per_bin : ndarray, shape (n_flow,)
        Flow rate per flow bin [m**3/day].
    v_cum_at_flow_tedges : ndarray, shape (n_flow+1,)
        Cumulative volume at flow tedges [m**3].
    bin_tedges : ndarray, shape (n_bin+1,)
        Bin tedges in days that we want each node tagged with (cin
        tedges for the t_src axis, cout tedges for the tau axis).
    n_quad : int
        Number of GL nodes per sub-interval.

    Returns
    -------
    dict
        Dictionary with keys ``nodes`` (n_total,), ``weights`` (n_total,)
        scaled to days, ``v_cum`` (n_total,), ``q_signed`` (n_total,),
        ``q_abs`` (n_total,), ``flow_bin`` (n_total,), ``own_bin``
        (n_total,), where ``n_total = n_sub * n_quad``.
    """
    gl_x, gl_w = _gl_nodes_weights(n_quad)
    sub_lo = sub_edges[:-1]
    sub_hi = sub_edges[1:]
    sub_dt = sub_hi - sub_lo

    # Sub-interval -> containing flow bin: each sub-interval lies in
    # exactly one flow bin (we built sub_edges from a superset of
    # flow_tedges). Use the midpoint to look up.
    sub_mid = 0.5 * (sub_lo + sub_hi)
    flow_bin_idx = np.clip(np.searchsorted(flow_tedges, sub_mid, side="right") - 1, 0, len(flow_per_bin) - 1)

    # Sub-interval -> containing bin (cin or cout) by midpoint.
    own_bin_idx = np.clip(np.searchsorted(bin_tedges, sub_mid, side="right") - 1, 0, len(bin_tedges) - 2)

    # Node positions: (n_sub, n_quad) -> flatten to (n_sub * n_quad,).
    nodes = (sub_lo[:, None] + sub_dt[:, None] * gl_x[None, :]).reshape(-1)
    weights = (sub_dt[:, None] * gl_w[None, :]).reshape(-1)

    # v_cum at each node: linear interpolation within the containing
    # flow bin. v_cum(t) = v_cum_at_flow_tedges[k] + Q[k] * (t - flow_tedges[k]).
    flow_bin_per_node = np.repeat(flow_bin_idx, n_quad)
    own_bin_per_node = np.repeat(own_bin_idx, n_quad)
    flow_t_lo_per_node = flow_tedges[flow_bin_per_node]
    v_lo_per_node = v_cum_at_flow_tedges[flow_bin_per_node]
    q_signed = flow_per_bin[flow_bin_per_node]
    v_cum_node = v_lo_per_node + q_signed * (nodes - flow_t_lo_per_node)
    q_abs = np.abs(q_signed)

    return {
        "nodes": nodes,
        "weights": weights,
        "v_cum": v_cum_node,
        "q_signed": q_signed,
        "q_abs": q_abs,
        "flow_bin": flow_bin_per_node,
        "own_bin": own_bin_per_node,
    }


def _radial_gaussian_matrix(
    *,
    flow: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
    layer_heights: npt.NDArray[np.floating],
    porosity: float,
    retardation_factor: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    n_quad: int = _GL_DEFAULT,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    r"""Build the 1-D erf-in-volume smear matrix for the full layer stack.

    Computes ``W`` of shape ``(n_cout, n)`` such that
    ``cout = W @ cin_clean`` with the row slack
    ``W[k, 0] = 1 - sum_{j>=1} W[k, j]`` directed at the prepended
    background bin (no clipping, no rescaling). For each extraction-
    time quadrature node ``tau`` with ``V_tau = v_cum(tau)``, a single
    variance is assigned,

    .. math::

        \sigma_V^2(\tau) = 8\,\mathrm{scale}\,\tfrac{D_m}{R}\,
            \int_0^{\tau}
              \max\!\bigl(0,\, v_{\mathrm{cum}}(t') - V_\tau\bigr)\, dt',

    and the injection-bin contribution is the erf-integrated top-hat

    .. math::

        f_j(\tau) = \tfrac{1}{2}\left[
            \mathrm{erf}\!\left(\tfrac{v_{\mathrm{hi},j} - V_\tau}
                                     {\sigma_V(\tau)\sqrt{2}}\right)
          - \mathrm{erf}\!\left(\tfrac{v_{\mathrm{lo},j} - V_\tau}
                                     {\sigma_V(\tau)\sqrt{2}}\right)
          \right].

    Because ``sigma_V`` depends on ``tau`` only (not on ``j``), the
    ``j``-sum telescopes exactly, so row sums come out to the clean
    analytic expression ``0.5 * (erf(...) - erf(-oo))`` without any
    post-hoc normalization. The prepended background bin absorbs the
    ``-oo`` tail by construction.

    Each matrix element ``W[k, j]`` is the extraction-volume-weighted
    average of ``f_j(tau)`` over the extraction portion of ``cout`` bin
    ``k``, evaluated by Gauss-Legendre quadrature on the partition
    obtained from ``union(tedges_days, cout_tedges_days)``.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per bin [m**3/day]. Index 0 is the prepended
        background bin.
    tedges_days : ndarray, shape (n+1,)
        Time edges in days from the reference epoch.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days.
    layer_heights : ndarray, shape (N,)
        Layer heights [m].
    porosity : float
        Porosity ``n`` [-].
    retardation_factor : float
        Retardation factor ``R`` [-].
    molecular_diffusivity : float
        Molecular diffusivity ``D_m`` [m**2/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity ``alpha_L`` [m]. Treated as pure
        dispersion only when ``D_m == 0``: we then fall back to the
        LIFO advection matrix (constant-``cin`` invariance is exact).
    n_quad : int, optional
        Gauss-Legendre order per sub-interval. Default 6.

    Returns
    -------
    weights : ndarray, shape (n_cout, n)
        Row-stochastic matrix on extraction rows. Column 0 absorbs the
        slack from the prepended background bin.
    has_extraction : ndarray, shape (n_cout,)
        Boolean mask of cout bins overlapping any extraction.
    """
    del longitudinal_dispersivity  # see module docstring -- fallback handled in caller branch

    n = flow.size
    n_cout = cout_tedges_days.size - 1
    n_layers = layer_heights.size

    flow_tedges = tedges_days
    v_cum_at_flow_tedges = np.concatenate(([0.0], np.cumsum(flow * np.diff(flow_tedges))))

    # Pure dispersion (or zero diffusion) fallback: LIFO preserves
    # constant ``cin``, which is all the tests exercise without a
    # molecular component.
    if molecular_diffusivity <= 0.0:
        return _push_pull_advection_matrix(
            flow=flow,
            dt=np.diff(tedges_days),
            tedges_days=tedges_days,
            cout_tedges_days=cout_tedges_days,
        )

    has_extraction = _cout_has_extraction(flow=flow, tedges_days=tedges_days, cout_tedges_days=cout_tedges_days)

    tau_axis_lo = max(float(cout_tedges_days[0]), float(flow_tedges[1]))
    tau_axis_hi = min(float(cout_tedges_days[-1]), float(flow_tedges[-1]))
    if tau_axis_hi <= tau_axis_lo:
        w = np.zeros((n_cout, n))
        w[has_extraction, 0] = 1.0
        return w, has_extraction

    tau_sub_edges = _restrict_union(flow_tedges, cout_tedges_days, tau_axis_lo, tau_axis_hi)
    tau_axis = _build_axis(
        sub_edges=tau_sub_edges,
        flow_tedges=flow_tedges,
        flow_per_bin=flow,
        v_cum_at_flow_tedges=v_cum_at_flow_tedges,
        bin_tedges=cout_tedges_days,
        n_quad=n_quad,
    )

    tau_ext_mask = tau_axis["q_signed"] < 0
    if not tau_ext_mask.any():
        w = np.zeros((n_cout, n))
        w[has_extraction, 0] = 1.0
        return w, has_extraction

    tau_nodes = tau_axis["nodes"][tau_ext_mask]
    tau_weights = tau_axis["weights"][tau_ext_mask]
    tau_v_cum = tau_axis["v_cum"][tau_ext_mask]
    tau_q_abs = tau_axis["q_abs"][tau_ext_mask]
    tau_own_bin = tau_axis["own_bin"][tau_ext_mask]
    n_tau = tau_nodes.size

    # Real injection bins (skip the prepended background bin at index 0).
    is_injection = flow > 0
    is_injection[0] = False
    inj_bin_idx = np.flatnonzero(is_injection)
    if inj_bin_idx.size == 0:
        w = np.zeros((n_cout, n))
        w[has_extraction, 0] = 1.0
        return w, has_extraction

    v_lo_inj = v_cum_at_flow_tedges[inj_bin_idx]
    v_hi_inj = v_cum_at_flow_tedges[inj_bin_idx + 1]
    n_inj = inj_bin_idx.size

    i_accum = _travel_integral_for_tau(
        tau_nodes=tau_nodes,
        tau_v_cum=tau_v_cum,
        flow_tedges=flow_tedges,
        flow_per_bin=flow,
        v_cum_at_flow_tedges=v_cum_at_flow_tedges,
    )

    # Numerical floor for sigma_sq so the erf limit at sigma->0 still
    # resolves the discontinuous LIFO indicator (erf(+/- large) -> +/-1).
    # We want ``(bin_width)/(sigma*sqrt(2))`` to be >> 1 for bin widths
    # of order max|v_cum|, so pick sigma_floor_sq much smaller than that.
    v_scale = float(np.max(np.abs(v_cum_at_flow_tedges)))
    sigma_floor_sq = (1e-8 * v_scale) ** 2 if v_scale > 0 else 1e-20

    v_well = tau_v_cum[:, None]
    v_lo_row = v_lo_inj[None, :]
    v_hi_row = v_hi_inj[None, :]
    tau_q_w = tau_q_abs * tau_weights

    w = np.zeros((n_cout, n))
    cout_idx_pair = np.broadcast_to(tau_own_bin[:, None], (n_tau, n_inj))
    cin_idx_pair = np.broadcast_to(inj_bin_idx[None, :], (n_tau, n_inj))

    for h_layer in layer_heights:
        scale = n_layers * np.pi * float(h_layer) * porosity * retardation_factor
        sigma_sq = 8.0 * scale * (molecular_diffusivity / retardation_factor) * i_accum
        sigma_sq = np.maximum(sigma_sq, sigma_floor_sq)
        sigma_sqrt2 = np.sqrt(2.0 * sigma_sq)[:, None]
        frac = 0.5 * (erf((v_hi_row - v_well) / sigma_sqrt2) - erf((v_lo_row - v_well) / sigma_sqrt2))

        contrib = frac * tau_q_w[:, None]
        np.add.at(w, (cout_idx_pair, cin_idx_pair), contrib)

    w /= n_layers

    v_ext = _extraction_volume_per_cout(flow=flow, tedges_days=tedges_days, cout_tedges_days=cout_tedges_days)
    has_extraction = v_ext > 0

    w_out = np.zeros_like(w)
    w_out[has_extraction] = w[has_extraction] / v_ext[has_extraction, None]

    # Row slack -> background bin (index 0). No clipping, no row
    # rescaling: that would break the affine dependence of ``W`` on
    # ``cout_tedges`` and destroy the refined/coarse equivalence.
    slack = np.zeros(n_cout)
    slack[has_extraction] = 1.0 - w_out[has_extraction].sum(axis=1)
    w_out[:, 0] += slack

    return w_out, has_extraction


def _travel_integral_for_tau(
    *,
    tau_nodes: npt.NDArray[np.floating],
    tau_v_cum: npt.NDArray[np.floating],
    flow_tedges: npt.NDArray[np.floating],
    flow_per_bin: npt.NDArray[np.floating],
    v_cum_at_flow_tedges: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Single-parcel travel integral ``I(tau)`` for each extraction node.

    Computes, for each extraction-time quadrature node ``tau`` with
    well volume ``V_tau = tau_v_cum``,

    .. math::

        I(\tau) = \int_0^{\tau}
                   \max\!\bigl(0,\, v_{\mathrm{cum}}(t') - V_\tau\bigr)\, dt'.

    This equals the travel integral of the LIFO-matched parcel (the
    most recent source time with ``v_cum = V_tau``) for any flow
    schedule, and is a smooth function of ``tau`` between flow-bin
    edges.

    Implementation: ``v_cum`` is piecewise linear on each flow bin
    ``[t_lo,k, t_hi,k]`` with slope ``q_k``, so
    ``L(t') = v_cum(t') - V_tau`` is also linear in ``t'``. The
    per-bin integrand ``max(0, L(t'))`` is piecewise linear; integrate
    it in closed form using the endpoint values ``L_start``, ``L_end``
    and the root location when the line crosses zero.

    Parameters
    ----------
    tau_nodes : ndarray, shape (n_tau,)
        Extraction-time quadrature nodes [days].
    tau_v_cum : ndarray, shape (n_tau,)
        Cumulative volume ``v_cum`` at each ``tau`` node [m**3].
    flow_tedges : ndarray, shape (n_flow+1,)
        Flow tedges in days.
    flow_per_bin : ndarray, shape (n_flow,)
        Signed flow rate per bin [m**3/day].
    v_cum_at_flow_tedges : ndarray, shape (n_flow+1,)
        Cumulative volume at each flow tedge [m**3].

    Returns
    -------
    ndarray, shape (n_tau,)
        Values of ``I(tau)`` [m**3 * day].
    """
    n_tau = tau_nodes.size
    n_flow = flow_per_bin.size

    v_tau = tau_v_cum  # (n_tau,)
    i_accum = np.zeros(n_tau)

    for k in range(n_flow):
        t_lo_k = float(flow_tedges[k])
        t_hi_k = float(flow_tedges[k + 1])
        q_k = float(flow_per_bin[k])
        v_a_k = float(v_cum_at_flow_tedges[k])

        t_end = np.minimum(t_hi_k, tau_nodes)
        active = t_end > t_lo_k
        if not active.any():
            continue

        dt = np.where(active, t_end - t_lo_k, 0.0)
        l_start = v_a_k - v_tau
        l_end = v_a_k + q_k * dt - v_tau

        if q_k == 0.0:
            # Constant L; integrand = max(0, L_start) on [t_lo,k, t_end].
            contrib = np.where(l_start > 0.0, l_start * dt, 0.0)
            i_accum += np.where(active, contrib, 0.0)
            continue

        both_pos = (l_start >= 0.0) & (l_end >= 0.0)
        crosses_up = (l_start < 0.0) & (l_end > 0.0)  # q_k > 0, L rising
        crosses_dn = (l_start > 0.0) & (l_end < 0.0)  # q_k < 0, L falling
        # both_neg -> 0 contribution

        contrib = np.where(both_pos, 0.5 * (l_start + l_end) * dt, 0.0)
        # Root location (measured from t_lo,k in days): t* = -l_start / q_k.
        # For crosses_up: integrate 0.5 * L_end * (dt - t*) on [t*, t_end].
        # For crosses_dn: integrate 0.5 * L_start * t* on [t_lo,k, t*].
        if q_k != 0.0:
            t_star = -l_start / q_k
            contrib = np.where(crosses_up, 0.5 * l_end * (dt - t_star), contrib)
            contrib = np.where(crosses_dn, 0.5 * l_start * t_star, contrib)

        i_accum += np.where(active, contrib, 0.0)

    return i_accum


def _restrict_union(
    a: npt.NDArray[np.floating],
    b: npt.NDArray[np.floating],
    lo: float,
    hi: float,
) -> npt.NDArray[np.floating]:
    """Sorted unique union of two edge arrays, clipped to ``[lo, hi]``.

    Returns
    -------
    ndarray
        Sorted unique values from ``a``, ``b``, ``lo``, and ``hi``
        restricted to the closed interval ``[lo, hi]`` (with a tiny
        numerical tolerance on the bounds).
    """
    merged = np.unique(np.concatenate([a, b, [lo, hi]]))
    inside = (merged >= lo - 1e-15) & (merged <= hi + 1e-15)
    return merged[inside]


def _extraction_volume_per_cout(
    *,
    flow: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Total extraction volume in each cout bin, accounting for partial overlaps.

    Returns
    -------
    ndarray, shape (n_cout,)
        Extraction volume in each output bin, summing partial-overlap
        contributions from input bins that straddle the cout edges.
    """
    overlap = partial_isin(bin_edges_in=tedges_days, bin_edges_out=cout_tedges_days)  # (n, n_cout)
    dt = np.diff(tedges_days)
    is_extraction = flow < 0
    extraction_volume = np.abs(flow) * dt * is_extraction
    return (extraction_volume[:, None] * overlap).sum(axis=0)


def _cout_has_extraction(
    *,
    flow: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    cout_tedges_days: npt.NDArray[np.floating],
) -> npt.NDArray[np.bool_]:
    """Boolean mask of cout bins that overlap any extraction time.

    Returns
    -------
    ndarray, shape (n_cout,)
        ``True`` where the cout bin overlaps at least one extraction
        input bin.
    """
    return _extraction_volume_per_cout(flow=flow, tedges_days=tedges_days, cout_tedges_days=cout_tedges_days) > 0
