r"""
Utility functions for the analytical-kernel radial push-pull module.

Internal helpers used by :mod:`gwtransport.radial3`. The public surface
exposed by this module is :func:`_radial_bessel_matrix`, which builds
the ``(n_cout, n)`` input-to-output weight matrix for a single layer of
a radial push-pull well using the 2-D radial heat-equation Green's
function evaluated at the well screen.

Approach -- 2-D radial Bessel kernel at the well
------------------------------------------------
For a layer of height ``h``, porosity ``n``, and retardation ``R``,
define the volume coordinate ``V = scale * r**2`` with
``scale = n_layers * pi * h * n * R``. Each parcel injected at source
time ``t_src`` has Lagrangian ``V``-position
``V_parcel(t) = max(v_cum(t) - v_cum(t_src), 0)`` under pure advection.
The 2-D radial heat equation in ``V`` coordinates has a V-dependent
diffusion coefficient ``D_V(V) = 4 * scale * (D_m / R) * V`` that
vanishes at ``V = 0`` -- the physical absorbing behaviour of the well
face where the cylindrical area element ``r dr`` collapses.

**Green's function at the well.** The 2-D radial heat equation with
diffusivity ``D = D_m / R`` has the classical Green's function

.. math::

    G(r, r'; t) = \frac{1}{4\pi D t}\,
                  \exp\!\left(-\frac{r^2 + r'^2}{4 D t}\right)
                  I_0\!\left(\frac{r\,r'}{2 D t}\right).

Evaluated at the well screen ``r = 0`` (with ``I_0(0) = 1``) and
transformed to ``V`` coordinates with the area element
``2\pi r' dr' = (\pi / \mathrm{scale})\, dV``, the concentration at
``V = 0`` becomes a linear functional of the V-space initial
condition with an **exponential** kernel

.. math::

    K(V_{\mathrm{src}}; t_{\mathrm{eff}})
        = \lambda\,\exp(-\lambda\,V_{\mathrm{src}}),
    \qquad
    \lambda = \frac{1}{4\,\mathrm{scale}\,(D_m/R)\,t_{\mathrm{eff}}}.

Place this kernel in the Lagrangian frame of the LIFO-matched parcel
at ``tau`` (the most recent source time with
``v_cum(t_src*) = V_\tau := v_cum(\tau)``). The displacement from the
well face for injection bin ``j`` is
``\xi_j = \max(V_\tau - V_\mathrm{src}, 0)``, and the injection-bin
contribution is a difference of exponentials:

.. math::

    f_j(\tau) = \exp(-\lambda\,\xi_{\mathrm{near},j})
              - \exp(-\lambda\,\xi_{\mathrm{far},j}),
    \quad
    \xi_{\mathrm{near},j} = \max(V_\tau - v_{\mathrm{hi},j}, 0),
    \quad
    \xi_{\mathrm{far},j}  = \max(V_\tau - v_{\mathrm{lo},j}, 0).

The effective time ``t_eff(\tau)`` is the LIFO parcel age, i.e., the
elapsed time between the extraction moment ``\tau`` and the source
time ``t_src^*(V_\tau)`` at which the currently-exiting parcel was
originally injected:

.. math::

    t_{\mathrm{eff}}(\tau) = \tau - t_{\mathrm{src}}^*(V_\tau),

where ``t_src^*(V_\tau)`` is the most recent time at which
``v_{\mathrm{cum}}`` rose through ``V_\tau``, taking prior extractions
into account via LIFO bookkeeping. This is the physical elapsed time
the matched parcel has been diffusing -- rest phases between push and
pull are counted correctly (the plume sits and diffuses in place
during rest). The parcel age is supplied by :func:`_lifo_age_for_tau`
via a chronological push-stack simulation and is well-defined for any
flow schedule.

**Row sums telescope.** For injection bins consecutive in ``V_src``
(``V_{\mathrm{lo},j+1} = V_{\mathrm{hi},j}``) the per-bin
contributions telescope:

.. math::

    \sum_{j=1}^{n_{\mathrm{inj}}} f_j(\tau)
        = \exp(-\lambda\,\xi_{\mathrm{near},n_{\mathrm{inj}}})
        - \exp(-\lambda\,\xi_{\mathrm{far},1}).

At the start of extraction after a push-then-rest,
``V_\tau = V_{\mathrm{hi,max}}`` so ``\xi_{\mathrm{near}} = 0``, the
first term is ``1``, and nearly all mass lands in the most recently
injected bin -- giving ``c_{\mathrm{out}} \approx c_{\mathrm{in}}[\mathrm{last}]``
as required. As extraction proceeds, ``t_{\mathrm{eff}}`` grows, the
exponential broadens, and the tail
``\exp(-\lambda \cdot (V_\tau - v_{\mathrm{lo},1}))`` bleeds out to
the background column via the row-slack step, producing the physical
relaxation from ``c_{\mathrm{in}}[\mathrm{last}]`` toward
``c_{\mathrm{background}}``.

**Boundary at ``V = 0``.** Because the kernel is evaluated at ``r=0``
and derived from the 2-D radial Green's function, the V-dependent
``D_V`` that vanishes at the well face is baked into the kernel
geometry, so there is no leakage of kernel mass beyond the
LIFO-matched parcel (i.e., into the unphysical region ``V > V_\tau``).
The symmetric Gaussian kernel that this module previously used did
leak half its mass into that region, producing a ``0.5 * c_in[last]``
artifact at the first extraction node after a rest.

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
When ``D_m == 0`` and ``alpha_L > 0`` (pure dispersion) the Bessel
kernel degenerates; we fall back to the LIFO advection matrix in that
regime, which preserves row-stochasticity and the constant-``cin``
identity. The tests only exercise dispersion under constant
``cin == c_background`` where the LIFO solution is exact.

This file is part of gwtransport which is released under AGPL-3.0
license. See the ./LICENSE file or
https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full
license details.
"""

import numpy as np
import numpy.typing as npt

from gwtransport.utils import partial_isin

__all__ = ["_push_pull_advection_matrix", "_radial_bessel_matrix"]

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


def _radial_bessel_matrix(
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
    r"""Build the 2-D radial Bessel smear matrix for the full layer stack.

    Computes ``W`` of shape ``(n_cout, n)`` such that
    ``cout = W @ cin_clean`` with the row slack
    ``W[k, 0] = 1 - sum_{j>=1} W[k, j]`` directed at the prepended
    background bin (no clipping, no rescaling). The kernel is the
    2-D radial heat-equation Green's function evaluated at the well
    screen ``r = 0`` and transformed to the ``V = scale * r**2``
    coordinate -- an exponential in ``V_src`` (see the module
    docstring for the derivation).

    For each extraction-time quadrature node ``tau`` with
    ``V_tau = v_cum(tau)``, the effective diffusion time is the LIFO
    parcel age

    .. math::

        t_{\mathrm{eff}}(\tau) = \tau - t_{\mathrm{src}}^*(V_\tau),

    where ``t_src^*(V_tau)`` is the most recent time at which
    ``v_cum`` rose through ``V_tau`` after accounting for prior
    extraction (supplied by :func:`_lifo_age_for_tau`). The kernel
    rate is ``lambda = 1 / (4 * scale * (D_m/R) * t_eff)``, and the
    contribution of injection bin ``j`` with source-volume edges
    ``[v_lo_j, v_hi_j]`` is the difference of exponentials

    .. math::

        f_j(\tau)
        = \exp\!\left(-\lambda\,\xi_{\mathrm{near},j}\right)
        - \exp\!\left(-\lambda\,\xi_{\mathrm{far},j}\right),

    with ``xi_near = max(V_tau - v_hi_j, 0)`` (the close-to-well edge
    of bin ``j``) and ``xi_far = max(V_tau - v_lo_j, 0)``. Injection
    bins that lie entirely above ``V_tau`` contribute zero, bins that
    straddle ``V_tau`` contribute only their physical portion, and
    fully-inside bins contribute the clean exponential difference.

    Because the rate ``lambda`` depends on ``tau`` only (not on
    ``j``), the ``j``-sum telescopes exactly for bins consecutive in
    ``V_src``, so row sums come out to a clean analytic expression
    without any post-hoc normalization. The prepended background bin
    absorbs the remaining slack (mass in ``V_src`` below the first
    real injection bin and any tail beyond the Lagrangian reach).

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

    # LIFO parcel age: t_eff(tau) = tau - t_src*(V_tau), where t_src*
    # is the most recent time v_cum rose through V_tau. This is the
    # correct elapsed diffusion time for the 2-D radial Green's
    # function -- it properly includes any rest phase between push
    # and pull, which the alternative I(tau)/V_tau formula loses (its
    # integrand vanishes identically whenever v_cum == V_tau).
    t_eff = _lifo_age_for_tau(
        tau_nodes=tau_nodes,
        tau_v_cum=tau_v_cum,
        flow_tedges=flow_tedges,
        flow_per_bin=flow,
        v_cum_at_flow_tedges=v_cum_at_flow_tedges,
    )

    # Numerical floor on the kernel denominator D := 4*scale*(D_m/R)*t_eff
    # so lam = 1/D stays finite when t_eff -> 0 (start of extraction with
    # V_tau = V_hi_max). With D very small, exp(-xi/D) underflows to 0
    # for any xi > 0 and equals 1 at xi = 0, giving the LIFO-delta limit.
    v_scale = float(np.max(np.abs(v_cum_at_flow_tedges)))
    denom_floor = 1e-12 * v_scale if v_scale > 0.0 else 1e-20

    v_well = tau_v_cum[:, None]
    v_lo_row = v_lo_inj[None, :]
    v_hi_row = v_hi_inj[None, :]
    tau_q_w = tau_q_abs * tau_weights

    # xi = V_tau - V_src, measured inward from the LIFO-matched parcel
    # at V_tau toward the well face at V = 0, clipped to [0, inf). A
    # straddling bin (V_lo_j < V_tau < V_hi_j) collapses xi_near to 0
    # and keeps only the physical portion of the bin; a fully-above-
    # V_tau bin collapses both to 0 and contributes zero.
    xi_near = np.maximum(v_well - v_hi_row, 0.0)
    xi_far = np.maximum(v_well - v_lo_row, 0.0)

    w = np.zeros((n_cout, n))
    cout_idx_pair = np.broadcast_to(tau_own_bin[:, None], (n_tau, n_inj))
    cin_idx_pair = np.broadcast_to(inj_bin_idx[None, :], (n_tau, n_inj))

    for h_layer in layer_heights:
        scale = n_layers * np.pi * float(h_layer) * porosity * retardation_factor
        denom = 4.0 * scale * (molecular_diffusivity / retardation_factor) * t_eff
        denom = np.maximum(denom, denom_floor)
        lam = (1.0 / denom)[:, None]
        frac = np.exp(-lam * xi_near) - np.exp(-lam * xi_far)

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


def _lifo_age_for_tau(
    *,
    tau_nodes: npt.NDArray[np.floating],
    tau_v_cum: npt.NDArray[np.floating],
    flow_tedges: npt.NDArray[np.floating],
    flow_per_bin: npt.NDArray[np.floating],
    v_cum_at_flow_tedges: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""LIFO parcel age ``tau - t_src*(V_tau)`` per extraction node.

    Simulates the LIFO push-stack chronologically and, for each
    extraction quadrature node ``tau``, returns the elapsed time since
    the matched parcel was injected. This is the correct "effective
    diffusion time" for the 2-D radial Green's function at the well
    screen: it counts any rest phase between push and pull (when the
    plume sits and diffuses in place), which the alternative
    travel-integral formula ``I(tau) / V_tau`` silently misses because
    its integrand vanishes whenever ``v_cum == V_tau``.

    Stack structure. Each push bin appends one entry
    ``(t_start, v_bot, q_push)``. During extraction we pop entries
    whose lower volume edge lies above the current ``V_tau``, and
    within the surviving top segment we recover the matched source
    time via ``t_src = t_start + (V_tau - v_bot) / q_push``. For the
    push-then-rest-then-pull reference schedule this reduces to
    ``age = tau - V_tau / q_push`` with the rest duration absorbed
    into ``tau`` directly -- exactly the physical elapsed time between
    injection and extraction of a given parcel.

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
        Parcel age per ``tau`` node [days]. Zero for over-extraction
        (stack empty), where the kernel contribution collapses to the
        background-bin slack.
    """
    n_tau = tau_nodes.size
    n_flow = flow_per_bin.size

    # Chronological sort for sequential stack walk.
    order = np.argsort(tau_nodes, kind="stable")
    tau_sorted = tau_nodes[order]
    v_sorted = tau_v_cum[order]
    age_sorted = np.zeros(n_tau)

    # Stack entries: (t_start, v_bot, q_push).
    stack: list[tuple[float, float, float]] = []

    tol = 1e-12
    tau_i = 0
    for k in range(n_flow):
        t_lo_k = float(flow_tedges[k])
        t_hi_k = float(flow_tedges[k + 1])
        q_k = float(flow_per_bin[k])

        # Advance past any tau nodes belonging to an earlier (already-
        # processed) flow bin. In practice tau nodes are filtered to
        # extraction sub-intervals upstream, so this branch is a
        # defensive no-op.
        while tau_i < n_tau and tau_sorted[tau_i] < t_lo_k - tol:
            age_sorted[tau_i] = 0.0
            tau_i += 1

        if q_k > 0.0:
            stack.append((t_lo_k, float(v_cum_at_flow_tedges[k]), q_k))
            # Skip any tau nodes that landed in a push bin (shouldn't
            # happen -- the caller filters on ``q_signed < 0``).
            while tau_i < n_tau and tau_sorted[tau_i] <= t_hi_k + tol:
                age_sorted[tau_i] = 0.0
                tau_i += 1
            continue

        if q_k < 0.0:
            # Process each tau node inside this extract bin in time order.
            while tau_i < n_tau and tau_sorted[tau_i] <= t_hi_k + tol:
                tau = float(tau_sorted[tau_i])
                v_tau = float(v_sorted[tau_i])
                # Pop segments consumed below ``v_tau``: if the top's
                # ``v_bot`` is above the current well level, the extract
                # has already pierced through it.
                while stack and stack[-1][1] > v_tau + tol:
                    stack.pop()
                if stack:
                    t_start, v_bot, q_push = stack[-1]
                    t_src = t_start + (v_tau - v_bot) / q_push
                    age_sorted[tau_i] = max(tau - t_src, 0.0)
                else:
                    age_sorted[tau_i] = 0.0
                tau_i += 1
            continue

        # Rest bin: no stack change; skip any tau nodes that land here
        # (should also not happen under the upstream filter).
        while tau_i < n_tau and tau_sorted[tau_i] <= t_hi_k + tol:
            age_sorted[tau_i] = 0.0
            tau_i += 1

    # Any remaining tau nodes past the last flow bin: background.
    while tau_i < n_tau:
        age_sorted[tau_i] = 0.0
        tau_i += 1

    age = np.zeros(n_tau)
    age[order] = age_sorted
    return age


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
