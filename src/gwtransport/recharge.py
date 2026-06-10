"""
Recharge-Driven Transport for Aquifers with Areal Recharge.

This module models solute (or heat) transport to a single extraction well in an
aquifer that receives uniform areal recharge. The recharge infiltrates with
concentration ``cin_recharge`` and mixes instantaneously over the saturated
thickness; transport is purely advective (no microdispersion or molecular
diffusion). Two conceptual models share one entry point:

- **Unbounded aquifer** (``aquifer_pore_volume=None``): all extracted water
  originates as recharge. The residence-time distribution is exponential with
  mean ``retardation_factor * aquifer_pore_depth / N`` — independent of the
  pumping rate, hydraulic conductivity, capture-zone size, and planform shape
  (Haitjema, 1995). In the cumulative-recharge clock
  ``u(t) = ∫ N dt / (retardation_factor * aquifer_pore_depth)`` (pore volumes
  flushed) the model is the stationary unit filter ``dC/du = cin_recharge - C``,
  which this module integrates in closed form per bin. No flow rate is needed.

- **Bounded aquifer** (``aquifer_pore_volume`` set): the aquifer extent is
  capped at pore volume ``aquifer_pore_volume`` (strip area
  ``aquifer_pore_volume / aquifer_pore_depth``). Water with concentration
  ``cin`` enters at the upstream side at rate ``q_b = flow - N * area``
  whenever extraction exceeds the rainfall on the strip. When rainfall exceeds
  extraction (``q_b < 0``) the surplus flows out across the upstream boundary
  and is lost; the outside has no memory, so when extraction later dominates
  again the inflow carries the current ``cin``. The exact solution is the
  unbounded exponential kernel acting on ``cin_recharge``, truncated at the
  boundary-entry time of the extracted water, with the residual tail weight
  placed as an atom on ``cin`` at the entry time. With zero recharge this
  reduces exactly to single-pore-volume piston flow
  (:func:`gwtransport.advection.infiltration_to_extraction`); with the
  boundary never feeding the well it reduces exactly to the unbounded model.

Available functions:

- :func:`recharge_to_extraction` - Compute extracted concentration from
  recharge concentration (and, in the bounded model, upstream-boundary
  concentration). Exact closed-form solution; output is a flow-weighted
  (bounded) or recharge-weighted (unbounded) bin average.

References
----------
Haitjema, H.M. (1995). On the residence time distribution in idealized
groundwatersheds. Journal of Hydrology, 172(1-4), 127-146.
https://doi.org/10.1016/0022-1694(95)02732-5

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport._time import tedges_to_days
from gwtransport._validation import (
    _validate_no_nan,
    _validate_non_negative_array,
    _validate_positive_scalar,
    _validate_tedges_parity,
)

# Kernel weights older than this many pore-volume flushes are below one ulp of the
# row sum (e^-60 ~ 9e-27); truncating them keeps the gathers banded without
# changing any double-precision result.
_KERNEL_CUTOFF = 60.0


def recharge_to_extraction(
    *,
    cin_recharge: npt.ArrayLike,
    recharge: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_depth: float,
    cin: npt.ArrayLike | None = None,
    flow: npt.ArrayLike | None = None,
    aquifer_pore_volume: float | None = None,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Compute the concentration of extracted water under uniform areal recharge.

    Unbounded model (``aquifer_pore_volume=None``): exponential residence-time
    distribution with mean ``retardation_factor * aquifer_pore_depth / N``
    (Haitjema, 1995), exact for bin-constant inputs. Bounded model
    (``aquifer_pore_volume`` set, together with ``cin`` and ``flow``): the
    exponential kernel is truncated at the upstream-boundary entry time and the
    residual weight is an atom on ``cin``; water pushed out across the boundary
    during rainfall surplus is lost.

    Parameters
    ----------
    cin_recharge : array-like
        Concentration of the recharge water entering via the surface
        [concentration units]. Length must equal ``len(tedges) - 1``; constant
        over each interval ``[tedges[i], tedges[i+1])``.
    recharge : array-like
        Areal recharge rate N [m/day; same length unit as
        ``aquifer_pore_depth``]. Length must equal ``len(tedges) - 1``.
        Must be non-negative and NaN-free.
    tedges : pandas.DatetimeIndex
        Time bin edges for the input series.
    cout_tedges : pandas.DatetimeIndex
        Time bin edges for the output series. Bins not fully inside the
        ``tedges`` range return NaN.
    aquifer_pore_depth : float
        Pore volume per unit surface area: porosity times saturated thickness
        [m]. The only static aquifer parameter of the unbounded model.
    cin : array-like, optional
        Concentration of the water entering at the upstream side of the
        bounded aquifer [concentration units]. Required when
        ``aquifer_pore_volume`` is set; must be None otherwise.
    flow : array-like, optional
        Extraction rate [m3/day]. Required when ``aquifer_pore_volume`` is
        set; must be None otherwise, because the unbounded model is
        independent of the pumping rate (see Notes). Must be non-negative and
        NaN-free.
    aquifer_pore_volume : float, optional
        Pore volume of the bounded aquifer [m3]. The strip area between the
        upstream boundary and the well is
        ``aquifer_pore_volume / aquifer_pore_depth``. Default None (unbounded).
    retardation_factor : float, optional
        Compound retardation factor (>= 1.0), by default 1.0. Dilates the
        solute clock; mixing fractions are unaffected.

    Returns
    -------
    numpy.ndarray
        Extracted concentration per ``cout_tedges`` bin, length
        ``len(cout_tedges) - 1``. Flow-weighted bin average (bounded model) or
        recharge-weighted bin average (unbounded model). NaN for bins outside
        the input time range, for zero-recharge bins (unbounded), and for
        zero-extraction bins (bounded).

    Raises
    ------
    ValueError
        If array lengths do not match the bin-edge pattern, inputs contain NaN
        or negative values, physical parameters are out of range, or only part
        of the bounded-model triple (``cin``, ``flow``,
        ``aquifer_pore_volume``) is provided.

    See Also
    --------
    gwtransport.advection.infiltration_to_extraction : Zero-recharge limit of the bounded model.
    gwtransport.deposition.deposition_to_extraction : Distributed source along the flow path.
    :ref:`concept-residence-time` : Background on residence times.
    :ref:`concept-transport-equation` : Flow-weighted averaging approach.

    Notes
    -----
    The unbounded model needs no flow rate because the capture zone
    self-adjusts: the well always draws exactly its pumping rate from
    recharge, over a capture area ``flow / N``. Pumping harder widens the
    capture area proportionally, leaving the age composition of the extracted
    water -- set by the ratio of pore storage per unit area
    (``aquifer_pore_depth``) to recharge per unit area (``N``) -- unchanged,
    so the flow rate cancels exactly (Haitjema, 1995). In the bounded model
    the area is fixed by ``aquifer_pore_volume`` instead of adjusting to the
    well, so the flow rate no longer cancels and must be given.

    Spin-up follows the ``"constant"`` policy: all inputs are treated as
    constant at their first values before ``tedges[0]``. For the bounded model
    this is the steady concentration profile
    ``C(V) = cr0 + (cin0 - cr0) * (V_R - apv) / (V_R - V)`` when the boundary
    feeds the well (``q_b(0) > 0``, ``V_R = flow[0] * aquifer_pore_depth /
    recharge[0]``), and the uniform profile ``cin_recharge[0]`` otherwise.

    Under constant inputs with ``flow > N * area`` the extracted water is the
    mass-balance mixture ``cin_recharge + (cin - cin_recharge) * q_b / flow``:
    an exponential residence-time density carrying the recharge fraction plus
    a piston atom of mass ``q_b / flow`` at the boundary-to-well travel time.

    The exponential kernel lives on the dimensionless clock ``u`` and is
    parameter-free; the pumping rate enters the bounded model only through the
    boundary-entry times. All formulas are closed-form (exp/log of bin-local
    quantities), exact to machine precision for bin-constant inputs.

    References
    ----------
    Haitjema, H.M. (1995). On the residence time distribution in idealized
    groundwatersheds. Journal of Hydrology, 172(1-4), 127-146.
    https://doi.org/10.1016/0022-1694(95)02732-5

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gwtransport.recharge import recharge_to_extraction
    >>> tedges = pd.date_range("2020-01-01", periods=11, freq="D")
    >>> cout = recharge_to_extraction(
    ...     cin_recharge=np.full(10, 2.5),
    ...     recharge=np.full(10, 0.002),
    ...     tedges=tedges,
    ...     cout_tedges=tedges[3:],
    ...     aquifer_pore_depth=3.0,
    ... )
    >>> np.allclose(cout, 2.5)
    True
    """
    tedges, cout_tedges = pd.DatetimeIndex(tedges), pd.DatetimeIndex(cout_tedges)
    cr = np.asarray(cin_recharge, dtype=float)
    rech = np.asarray(recharge, dtype=float)
    bounded = aquifer_pore_volume is not None

    if (cin is None) != (flow is None) or (cin is None) == bounded:
        msg = "cin, flow, and aquifer_pore_volume must be provided together (bounded model) or all be None"
        raise ValueError(msg)
    _validate_tedges_parity(tedges, cr, tedges_name="tedges", values_name="cin_recharge")
    _validate_tedges_parity(tedges, rech, tedges_name="tedges", values_name="recharge")
    _validate_no_nan(cr, name="cin_recharge")
    _validate_no_nan(rech, name="recharge")
    _validate_non_negative_array(rech, name="recharge")
    _validate_positive_scalar(aquifer_pore_depth, name="aquifer_pore_depth")
    if retardation_factor < 1.0:
        msg = "retardation_factor must be >= 1.0"
        raise ValueError(msg)

    t = tedges_to_days(tedges)
    tq = tedges_to_days(cout_tedges, ref=tedges[0])
    dt = np.diff(t)
    k = rech / (retardation_factor * aquifer_pore_depth)
    u = np.concatenate([[0.0], np.cumsum(k * dt)])
    covered = (tq[:-1] >= t[0]) & (tq[1:] <= t[-1])

    if not bounded:
        # The unbounded aquifer is the no-boundary special case: an infinite
        # pore volume puts the boundary beyond reach (pure kernel and spin-up
        # terms), and a synthetic flow proportional to recharge makes the
        # flow-weighted bin average the recharge-weighted one.
        return _bounded_average(t=t, dt=dt, u=u, k=k, q=k, cr=cr, cb=cr, apv=np.inf, tq=tq, covered=covered)

    cb = np.asarray(cin, dtype=float)
    q = np.asarray(flow, dtype=float)
    _validate_tedges_parity(tedges, cb, tedges_name="tedges", values_name="cin")
    _validate_tedges_parity(tedges, q, tedges_name="tedges", values_name="flow")
    _validate_no_nan(cb, name="cin")
    _validate_no_nan(q, name="flow")
    _validate_non_negative_array(q, name="flow")
    _validate_positive_scalar(aquifer_pore_volume, name="aquifer_pore_volume")
    # The solute clock divides flow by the retardation factor; weighting ratios are unaffected.
    return _bounded_average(
        t=t,
        dt=dt,
        u=u,
        k=k,
        q=q / retardation_factor,
        cr=cr,
        cb=cb,
        apv=float(aquifer_pore_volume),
        tq=tq,
        covered=covered,
    )


def _arrival_times(*, t, dt, k, q, apv):
    """Forward arrival time at the well of parcels released at ``(t[j], V=apv)``.

    Trajectories obey ``dV/dt = -(q - k V)`` with the per-bin closed form
    ``V(s) = V_R + (V - V_R) e^{k s}``. Returns NaN for parcels that are
    expelled across the boundary (lost) or have not arrived by ``t[-1]``.
    These arrivals are exactly the output times where the boundary-entry bin
    of the extracted water changes, including the pre-record transition. The
    bin loop over the live front is O(n^2) in the number of input bins.

    Returns
    -------
    ndarray
        Arrival time in days per release edge ``t[0..n-1]``; NaN if the parcel
        never reaches the well within the record.
    """
    n = len(dt)
    arrivals = np.full(n, np.nan)
    pos = np.full(n, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        for i in range(n):
            pos[i] = apv
            idx = np.nonzero(np.isfinite(pos[: i + 1]))[0]
            vi = pos[idx]
            if k[i] > 0:
                v_r = q[i] / k[i]
                v_end = v_r + (vi - v_r) * np.exp(k[i] * dt[i])
                hit = v_end <= 0.0
                s_hit = np.log(v_r / (v_r - vi[hit])) / k[i]
            else:
                v_end = vi - q[i] * dt[i]
                hit = v_end <= 0.0
                s_hit = vi[hit] / q[i] if q[i] > 0 else vi[hit][:0]  # q == 0 cannot hit the well
            arrivals[idx[hit]] = t[i] + s_hit
            # Resolved (arrived) parcels and parcels at or beyond the boundary
            # (expelled, or parked exactly on a stagnant boundary -- the next
            # edge's release duplicates a parked parcel) drop out of the front.
            pos[idx] = np.where(hit | (v_end >= apv), np.nan, v_end)
    return arrivals


def _backward_entries(*, queries, t, dt, k, q, apv, v_start=0.0, edge_side="right"):
    """Trace backward characteristics from ``(query, V=v_start)`` to the boundary or to ``t[0]``.

    Returns ``(s, v0)``: the boundary-entry time ``s`` (NaN if the water
    predates the record) and the landing position ``v0`` at ``t[0]`` for
    pre-record water (NaN otherwise). Uses the numerically local per-bin form
    ``V_a = V_R + (V_b - V_R) e^{-k seg}``; differencing global accumulators
    instead loses precision catastrophically once ``u >> 1``.

    ``v_start=apv`` with ``edge_side="left"`` traces the grazing trajectory
    that touches the boundary exactly at an (on-edge) query time backward into
    the preceding bin, yielding the left-branch limit of the entry-time map at
    that arrival (its release time); for queries not preceded by outflow the
    walk exits immediately and returns the query time itself.

    Returns
    -------
    tuple of ndarray
        ``(s, v0)`` per query: boundary-entry time in days (NaN for pre-record
        water) and landing position at ``t[0]`` (NaN for entered water).
    """
    n = len(dt)
    nq = len(queries)
    m = np.clip(np.searchsorted(t, queries, side="left" if edge_side == "left" else "right") - 1, 0, n - 1)
    s_out = np.full(nq, np.nan)
    v0 = np.full(nq, np.nan)
    pos = np.full(nq, float(v_start))
    open_ = np.ones(nq, dtype=bool)
    for i in range(n - 1, -1, -1):
        sel = np.nonzero(open_ & (m >= i))[0]
        if sel.size == 0:
            continue
        starts_here = m[sel] == i
        seg = np.where(starts_here, queries[sel] - t[i], dt[i])
        t_hi = np.where(starts_here, queries[sel], t[i + 1])
        vi = pos[sel]
        if k[i] > 0:
            v_r = q[i] / k[i]
            va = v_r + (vi - v_r) * np.exp(-k[i] * seg)
            ent = va >= apv
            with np.errstate(invalid="ignore", divide="ignore"):
                back = -np.log((apv - v_r) / (vi[ent] - v_r)) / k[i]
            back = np.where(np.isfinite(back), back, 0.0)  # stagnant boundary (vi == v_r == apv)
        else:
            va = vi + q[i] * seg
            # The q > 0 guard keeps a parcel parked exactly on the boundary
            # through a fully stagnant bin (no flow, no recharge) walking into
            # earlier bins: nothing moves and nothing is lost there.
            ent = (va >= apv) & (q[i] > 0.0)
            back = (apv - vi[ent]) / q[i] if q[i] > 0 else vi[ent][:0]
        s_ent = np.clip(t_hi[ent] - back, t[i], t_hi[ent])
        s_out[sel[ent]] = s_ent
        open_[sel[ent]] = False
        pos[sel[~ent]] = va[~ent]
    v0[open_] = pos[open_]
    return s_out, v0


def _bounded_average(*, t, dt, u, k, q, cr, cb, apv, tq, covered):
    """Flow-weighted bin averages of the bounded model, exact via piece integration.

    The output interval is split at: input edges, output edges, and the
    arrival times of boundary parcels released at the input edges. Between
    consecutive breakpoints the entry time stays within one source bin and
    every term integrates in closed form. The boundary atom integrates to
    ``cin_js * q_b,js * (s2 - s1)`` via the change of variables
    ``q e^{-u(t)} dt = e^{-u(s)} q_b(s) ds`` along the entry map; the kernel
    terms reduce to bin-local exponentials times
    ``I2 = ∫ q e^{-(u(t)-u(t2))} dt``; pre-record water carries the steady
    spin-up profile evaluated at the landing position ``v0 = G(t)``, whose
    flow-weighted integral is ``cr0``-linear plus a closed-form logarithm.

    Returns
    -------
    ndarray
        Flow-weighted average per output bin; NaN where undefined.
    """
    n = len(dt)
    n_out = len(tq) - 1
    qb = q - k * np.where(k > 0, apv, 0.0)  # the where avoids 0 * inf in the unbounded (apv = inf) routing

    arrivals = _arrival_times(t=t, dt=dt, k=k, q=q, apv=apv)
    bp = np.unique(np.concatenate([t, tq[(tq >= t[0]) & (tq <= t[-1])], arrivals[np.isfinite(arrivals)]]))
    mids = 0.5 * (bp[:-1] + bp[1:])
    s_bp, v0_bp = _backward_entries(queries=bp, t=t, dt=dt, k=k, q=q, apv=apv)
    s_mid, _ = _backward_entries(queries=mids, t=t, dt=dt, k=k, q=q, apv=apv)

    # The entry-time map s*(t) is discontinuous at the arrival of a parcel
    # released at an edge where boundary inflow resumes after an expulsion or
    # stagnation episode (the skipped span is the lost window), and the walk at
    # exactly that arrival resolves to one branch by floating-point luck. Both
    # one-sided limits are computed robustly instead: the right limit is the
    # release edge t[g] itself; the left limit is the release time of the
    # grazing trajectory, traced backward from (t[g], V=apv).
    g_idx = np.nonzero(np.isfinite(arrivals))[0]
    s_left = np.full(len(arrivals), np.nan)
    g_pos = g_idx[g_idx >= 1]  # the g = 0 arrival is the pre-record transition; its left side needs no entry
    if g_pos.size:
        s_left[g_pos], _ = _backward_entries(
            queries=t[g_pos], t=t, dt=dt, k=k, q=q, apv=apv, v_start=apv, edge_side="left"
        )
    av = arrivals[g_idx]
    order = np.argsort(av)
    av, ae = av[order], g_idx[order]

    def arrival_edge(x):
        if av.size == 0:
            return np.full(len(x), -1)
        pos = np.minimum(np.searchsorted(av, x), av.size - 1)
        return np.where(av[pos] == x, ae[pos], -1)

    t1, t2 = bp[:-1], bp[1:]
    span = t2 - t1
    m = np.clip(np.searchsorted(t, mids, side="right") - 1, 0, n - 1)
    kc = np.searchsorted(tq, mids, side="right") - 1
    in_out = (kc >= 0) & (kc < n_out)
    kc = np.clip(kc, 0, n_out - 1)

    pre = np.isnan(s_mid)
    js = np.clip(np.searchsorted(t, np.where(pre, t[0], s_mid), side="right") - 1, 0, n - 1)
    # Entry times at piece endpoints: one-sided limits at arrival breakpoints,
    # the walked values elsewhere, NaN (grazing/pre-record endpoints) falling
    # back to the entry-bin edge; the final clip into the piece's own entry bin
    # [t[js], t[js+1]] is a roundoff guard.
    e1, e2 = arrival_edge(t1), arrival_edge(t2)
    s1_raw = np.where(e1 >= 0, t[np.maximum(e1, 0)], s_bp[:-1])
    s2_raw = np.where(e2 >= 0, s_left[np.maximum(e2, 0)], s_bp[1:])
    s_lo, s_hi = t[js], t[js + 1]
    s1 = np.clip(np.where(np.isnan(s1_raw), s_lo, s1_raw), s_lo, s_hi)
    s2 = np.clip(np.where(np.isnan(s2_raw), s_hi, s2_raw), s_lo, s_hi)
    ds = np.maximum(s2 - s1, 0.0)

    u1t = u[m] + k[m] * (t1 - t[m])
    u2t = u[m] + k[m] * (t2 - t[m])
    kpos = k[m] > 0
    vol = q[m] * span
    fac = np.where(kpos, q[m] / np.where(kpos, k[m], 1.0), vol)

    def piece_integral(x):
        """Integrate ``q e^{x - u(t)}`` over each piece in closed form (callers keep x <= u(t1) bounded).

        Returns
        -------
        ndarray
            One integral per piece.
        """
        return np.where(kpos, fac * (np.exp(x - u1t) - np.exp(x - u2t)), fac * np.exp(x - u2t))

    # Kernel mass: full-bin sum for j in [lo, m), then the current-bin and
    # entry-bin partial corrections (telescopes to vol when cr == cb == 1).
    lo = np.where(pre, 0, js)
    lo_eff = np.maximum(lo, np.clip(np.searchsorted(u, u2t - _KERNEL_CUTOFF, side="right") - 1, 0, None))
    w_max = int((m - lo_eff).max(initial=0))
    ker = np.zeros(len(mids))
    if w_max > 0:
        cols = lo_eff[:, None] + np.arange(w_max)[None, :]
        valid = cols < m[:, None]
        colsc = np.clip(cols, 0, n - 1)
        w_lo = np.exp(u[colsc + 1] - u1t[:, None]) - np.exp(u[colsc] - u1t[:, None])
        w_hi = np.exp(u[colsc + 1] - u2t[:, None]) - np.exp(u[colsc] - u2t[:, None])
        wgt = np.where(kpos[:, None], w_lo - w_hi, w_hi)
        ker = np.einsum("pw,pw->p", np.where(valid, wgt, 0.0), cr[colsc])
    mass = cr[m] * vol - cr[m] * piece_integral(u[m]) + ker * fac

    # Boundary atom (entered) or steady-profile spin-up atom (pre-record). The
    # where keeps -inf * 0 (unbounded routing, pre-record pieces) out of the
    # discarded branch.
    entered_atom = (cb[js] - cr[js]) * np.where(pre, 0.0, qb[js]) * ds + cr[js] * piece_integral(u[js])
    qb0 = qb[0]
    if qb0 > 0 and k[0] > 0:
        # Landing positions v0 = G(t) at the piece endpoints; at the pre-record
        # transition the walk may resolve the grazing endpoint as entered (v0
        # NaN) -- that path lands exactly on the boundary.
        v0_1 = np.where(pre & np.isnan(v0_bp[:-1]), apv, np.where(pre, v0_bp[:-1], 0.0))
        v0_2 = np.where(pre & np.isnan(v0_bp[1:]), apv, np.where(pre, v0_bp[1:], 0.0))
        v_r0 = q[0] / k[0]
        ic_log = np.log((v_r0 - v0_1) / (v_r0 - v0_2))
        ic_atom = cr[0] * piece_integral(0.0) + (cb[0] - cr[0]) * (v_r0 - apv) * ic_log
    elif qb0 > 0:
        ic_atom = cb[0] * piece_integral(0.0)  # piston pre-record: domain full of boundary water
    else:
        ic_atom = cr[0] * piece_integral(0.0)  # boundary never fed the domain before t0
    mass += np.where(pre, ic_atom, entered_atom)

    take = in_out & (span > 0)
    masses = np.zeros(n_out)
    vols = np.zeros(n_out)
    np.add.at(masses, kc[take], mass[take])
    np.add.at(vols, kc[take], vol[take])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(covered & (vols > 0), masses / vols, np.nan)
