"""Tests for gwtransport.recharge.

The closed-form solution was verified during design against an independent
finite-volume PDE solve, forward trajectory integration, and an independent
re-derivation, all at machine precision. These tests pin that behavior:
analytic limits (steady state, piston flow against the advection module,
unbounded exponential reservoir), exact invariances (grid refinement, warm-up
equivalence of the constant spin-up, volume additivity, retardation-clock
scaling), conservation (weights sum to one), an independent pointwise oracle,
and the NaN/validation contract.
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction
from gwtransport.recharge import recharge_to_extraction

NH = 0.8
APV = 2.0  # strip area APV/NH = 2.5


def to_tedges(t_days):
    return pd.Timestamp("2020-01-01") + pd.to_timedelta(np.asarray(t_days, dtype=float), unit="D")


def nasty_scenario():
    """Entry, expulsion (rainfall surplus), zero-extraction, and zero-recharge piston episodes."""
    dt = np.array([
        1.0,
        1.5,
        0.8,
        1.2,
        1.0,
        0.7,
        1.3,
        1.0,  # entry: N*A = 0.75 < Q = 1.2
        1.1,
        0.9,
        1.4,
        1.0,
        1.2,  # expulsion: N*A = 3.0 > Q = 1.0
        1.0,  # no extraction
        0.9,
        1.1,
        1.3,  # zero recharge, piston entry
        1.0,
        0.8,
        1.2,
        1.0,
        1.1,
        0.9,
        1.0,  # strong-extraction entry
    ])
    n = len(dt)
    tedges = to_tedges(np.concatenate([[0.0], np.cumsum(dt)]))
    flow = np.empty(n)
    rech = np.empty(n)
    flow[0:8], rech[0:8] = 1.2, 0.3
    flow[8:13], rech[8:13] = 1.0, 1.2
    flow[13], rech[13] = 0.0, 0.5
    flow[14:17], rech[14:17] = 1.5, 0.0
    flow[17:24], rech[17:24] = 2.0, 0.4
    rng = np.random.default_rng(42)
    cin = rng.uniform(0.5, 4.0, n)
    cin_recharge = rng.uniform(0.5, 4.0, n)
    return tedges, flow, rech, cin, cin_recharge


def refine(tedges, arrays, p):
    """Split every bin into p sub-bins with unchanged values."""
    t = (tedges - tedges[0]) / pd.Timedelta(days=1)
    sub = np.concatenate([np.linspace(t[j], t[j + 1], p + 1)[:-1] for j in range(len(t) - 1)])
    sub = np.concatenate([sub, [t[-1]]])
    return tedges[0] + pd.to_timedelta(sub, unit="D"), [np.repeat(a, p) for a in arrays]


def pointwise_oracle(t_days, tedges, flow, rech, cin, cin_recharge, nh, apv, retardation_factor=1.0):
    """Independent pointwise C(0, t): scalar backward characteristic walk plus direct kernel sum.

    Shares no code path with the module (no piece integration, no arrival
    breakpoints); mirrors the design-phase reference verified against an
    independent PDE solve and re-derivation.
    """
    t = (tedges - tedges[0]) / pd.Timedelta(days=1)
    t = t.to_numpy(dtype=float)
    dt = np.diff(t)
    k = rech / (retardation_factor * nh)
    q = flow / retardation_factor
    u = np.concatenate([[0.0], np.cumsum(k * dt)])
    n = len(dt)
    out = np.empty(len(t_days))
    for iq, tval in enumerate(t_days):
        m = int(np.clip(np.searchsorted(t, tval, side="right") - 1, 0, n - 1))
        u_t = u[m] + k[m] * (tval - t[m])
        v_b, t_b, entry = 0.0, tval, None
        for j in range(m, -1, -1):
            seg = (t_b - t[j]) if j == m else dt[j]
            if k[j] > 0:
                v_r = q[j] / k[j]
                v_a = v_r + (v_b - v_r) * np.exp(-k[j] * seg)
            else:
                v_a = v_b + q[j] * seg
            if v_a >= apv:
                back = -np.log((apv - v_r) / (v_b - v_r)) / k[j] if k[j] > 0 else (apv - v_b) / q[j]
                s = t_b - back
                entry = (j, u[j] + k[j] * (s - t[j]))
                break
            v_b, t_b = v_a, t[j]
        if entry is None:
            qb0 = q[0] - k[0] * apv
            if qb0 > 0 and k[0] > 0:
                v_r0 = q[0] / k[0]
                c0 = cin_recharge[0] + (cin[0] - cin_recharge[0]) * (v_r0 - apv) / (v_r0 - v_b)
            elif qb0 > 0:
                c0 = cin[0]
            else:
                c0 = cin_recharge[0]
            js, u_s, c_atom = 0, 0.0, c0
        else:
            js, u_s = entry
            c_atom = cin[js]
        val = c_atom * np.exp(u_s - u_t)
        for j in range(js, m + 1):
            u_lo, u_hi = max(u[j], u_s), min(u[j + 1], u_t)
            if u_hi > u_lo:
                val += cin_recharge[j] * (np.exp(u_hi - u_t) - np.exp(u_lo - u_t))
        out[iq] = val
    return out


class TestUnbounded:
    def test_constant_inputs_pass_through(self):
        tedges = to_tedges(np.arange(11.0))
        cout = recharge_to_extraction(
            cin_recharge=np.full(10, 3.7),
            recharge=np.full(10, 0.4),
            tedges=tedges,
            cout_tedges=tedges[2:8],
            aquifer_pore_depth=NH,
        )
        np.testing.assert_allclose(cout, 3.7, rtol=1e-14)

    def test_step_response_analytic(self):
        """Step in cin_recharge at a bin edge: exact bin-averaged 1 - e^{-(u-u0)} response."""
        n, n0, rech = 20, 5, 0.4
        ku = rech / NH
        cr = np.zeros(n)
        cr[n0:] = 1.0
        tedges = to_tedges(np.arange(n + 1.0))
        cout = recharge_to_extraction(
            cin_recharge=cr, recharge=np.full(n, rech), tedges=tedges, cout_tedges=tedges, aquifer_pore_depth=NH
        )
        edges = np.arange(n + 1.0)
        lo, hi = np.maximum(edges[:-1] - n0, 0.0), np.maximum(edges[1:] - n0, 0.0)
        expected = np.where(
            hi > lo, (hi - lo + (np.exp(-ku * hi) - np.exp(-ku * lo)) / ku) / (edges[1:] - edges[:-1]), 0.0
        )
        np.testing.assert_allclose(cout, expected, atol=1e-14)

    def test_weight_sum_and_zero_recharge_nan(self):
        n = 12
        rech = np.full(n, 0.5)
        rech[4] = 0.0  # zero-recharge bin -> NaN output bin
        tedges = to_tedges(np.cumsum(np.concatenate([[0.0], np.linspace(0.4, 1.6, n)])))
        cout = recharge_to_extraction(
            cin_recharge=np.ones(n), recharge=rech, tedges=tedges, cout_tedges=tedges, aquifer_pore_depth=NH
        )
        assert np.isnan(cout[4])
        np.testing.assert_allclose(np.delete(cout, 4), 1.0, rtol=1e-13)

    def test_refinement_invariance(self):
        tedges, _, rech, _, cr = nasty_scenario()
        rech += 0.05  # strictly positive recharge so every output bin is defined
        cout_tedges = to_tedges(np.linspace(2.0, 23.0, 9))
        base = recharge_to_extraction(
            cin_recharge=cr, recharge=rech, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_depth=NH
        )
        rt, (rr, rc) = refine(tedges, [rech, cr], 7)
        fine = recharge_to_extraction(
            cin_recharge=rc, recharge=rr, tedges=rt, cout_tedges=cout_tedges, aquifer_pore_depth=NH
        )
        np.testing.assert_allclose(base, fine, rtol=1e-12)

    def test_retardation_dilates_clock(self):
        """R = 2 with recharge N equals R = 1 with recharge N/2 exactly."""
        tedges, _, rech, _, cr = nasty_scenario()
        cout_tedges = to_tedges(np.linspace(1.0, 24.0, 7))
        kwargs = {"cin_recharge": cr, "tedges": tedges, "cout_tedges": cout_tedges, "aquifer_pore_depth": NH}
        retarded = recharge_to_extraction(recharge=rech, retardation_factor=2.0, **kwargs)
        slowed = recharge_to_extraction(recharge=rech / 2.0, **kwargs)
        np.testing.assert_allclose(retarded, slowed, rtol=1e-14)

    def test_pointwise_and_weighting_against_oracle(self):
        """Direct independent anchor for the unbounded mode with varying recharge.

        Narrow midpoint-centered bins pin the pointwise values against the
        in-file oracle (run with apv=inf, where the boundary is unreachable):
        their width error is second order (~4e-12 at half-width 5e-6 days, no
        bin straddles an input edge) and the roundoff amplification eps/du of
        the tiny-bin average stays ~2e-9, so atol 1e-7 has ample headroom. One
        wide bin pins the recharge-weighted averaging against midpoint
        quadrature weighted by N (cout(t) is continuous in the unbounded
        model, so composite midpoint converges as the square of the step). The
        wide-bin check catches weighting mutations (e.g. weights proportional
        to the square of the recharge) that narrow bins cannot see.
        """
        tedges, _, rech, _, cr = nasty_scenario()
        rech += 0.05
        n = len(rech)
        mids = np.array([2.3, 7.9, 9.4, 12.7, 14.6, 16.1, 18.55, 21.0, 24.9])
        eps = 5e-6
        nb = to_tedges(np.sort(np.concatenate([mids - eps, mids + eps])))
        cout = recharge_to_extraction(
            cin_recharge=cr, recharge=rech, tedges=tedges, cout_tedges=nb, aquifer_pore_depth=NH
        )[::2]
        expected = pointwise_oracle(mids, tedges, np.zeros(n), rech, cr, cr, NH, np.inf)
        np.testing.assert_allclose(cout, expected, atol=1e-7)

        a, b = 3.0, 21.0
        wide = recharge_to_extraction(
            cin_recharge=cr, recharge=rech, tedges=tedges, cout_tedges=to_tedges([a, b]), aquifer_pore_depth=NH
        )
        t = ((tedges - tedges[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)
        cuts = np.unique(np.concatenate([[a, b], t[(t > a) & (t < b)]]))
        mass = vol = 0.0
        for g1, g2 in pairwise(cuts):
            ts = np.linspace(g1, g2, 4001)
            mid = 0.5 * (ts[:-1] + ts[1:])
            w = rech[np.clip(np.searchsorted(t, mid, side="right") - 1, 0, n - 1)] * np.diff(ts)
            mass += np.sum(pointwise_oracle(mid, tedges, np.zeros(n), rech, cr, cr, NH, np.inf) * w)
            vol += np.sum(w)
        np.testing.assert_allclose(wide[0], mass / vol, atol=2e-8)

    def test_outside_record_nan(self):
        tedges = to_tedges(np.arange(6.0))
        cout = recharge_to_extraction(
            cin_recharge=np.full(5, 2.0),
            recharge=np.full(5, 0.4),
            tedges=tedges,
            cout_tedges=to_tedges([-1.0, 1.0, 4.0, 7.0]),
            aquifer_pore_depth=NH,
        )
        assert np.isnan(cout[0])
        assert np.isnan(cout[2])
        np.testing.assert_allclose(cout[1], 2.0, rtol=1e-14)


class TestBounded:
    def test_zero_recharge_is_piston_advection(self):
        """N = 0 reduces exactly to single-pore-volume advection (existing module as oracle)."""
        tedges, flow, _, cin, cr = nasty_scenario()
        flow += 0.3  # strictly positive so every cout bin carries water
        n = len(flow)
        cout_tedges = to_tedges(np.linspace(0.0, 25.4, 12))
        ours = recharge_to_extraction(
            cin_recharge=cr,
            recharge=np.zeros(n),
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_depth=NH,
            cin=cin,
            flow=flow,
            aquifer_pore_volume=APV,
        )
        oracle = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([APV]),
            spinup="constant",
        )
        np.testing.assert_allclose(ours, oracle, rtol=1e-12)
        # cin_recharge carries no water when N = 0: its algebraic coefficient is
        # q*ds - vol == 0, realized through floating-point cancellation, so a
        # 100x scaling may shift cout by O(eps) but no more.
        scaled = recharge_to_extraction(
            cin_recharge=100.0 * cr,
            recharge=np.zeros(n),
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_depth=NH,
            cin=cin,
            flow=flow,
            aquifer_pore_volume=APV,
        )
        np.testing.assert_allclose(ours, scaled, rtol=1e-12)

    @pytest.mark.parametrize(("q0", "expected_kind"), [(2.0, "mixture"), (0.5, "recharge_only")])
    def test_steady_state_mixture(self, q0, expected_kind):
        """Constants: cout = cr + (cb - cr) q_b/Q when the boundary feeds the well, else cr."""
        n, rech, cb, cr = 30, 0.4, 4.2, 1.3
        tedges = to_tedges(np.arange(n + 1.0))
        cout = recharge_to_extraction(
            cin_recharge=np.full(n, cr),
            recharge=np.full(n, rech),
            tedges=tedges,
            cout_tedges=tedges[10:20],
            aquifer_pore_depth=NH,
            cin=np.full(n, cb),
            flow=np.full(n, q0),
            aquifer_pore_volume=APV,
        )
        qb_over_q = max(q0 - rech * APV / NH, 0.0) / q0
        expected = cr + (cb - cr) * qb_over_q if expected_kind == "mixture" else cr
        np.testing.assert_allclose(cout, expected, rtol=1e-14)

    def test_weight_sum_nasty_scenario(self):
        """cin == cin_recharge == 1 -> cout == 1 through entry, expulsion, and piston episodes."""
        tedges, flow, rech, _, _ = nasty_scenario()
        n = len(flow)
        cout = recharge_to_extraction(
            cin_recharge=np.ones(n),
            recharge=rech,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_depth=NH,
            cin=np.ones(n),
            flow=flow,
            aquifer_pore_volume=APV,
        )
        assert np.isnan(cout[13])  # zero-extraction bin carries no water
        np.testing.assert_allclose(np.delete(cout, 13), 1.0, rtol=1e-13)

    def test_pointwise_oracle_narrow_bins(self):
        """Bin averages over 1e-7-day bins pin the pointwise values of the independent oracle.

        The bin average deviates from the midpoint value by at most
        slope * width/2 ~ 1e-7, which sets the comparison tolerance.
        """
        tedges, flow, rech, cin, cr = nasty_scenario()
        mids = np.array([2.3, 7.9, 9.4, 12.7, 13.5, 16.1, 18.55, 21.0, 24.9])  # 13.5: expulsion episode; the
        # zero-extraction bin [14.1, 15.1] itself is NaN by contract and is covered elsewhere
        eps = 5e-8
        cout_tedges = to_tedges(np.sort(np.concatenate([mids - eps, mids + eps])))
        cout = recharge_to_extraction(
            cin_recharge=cr,
            recharge=rech,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_depth=NH,
            cin=cin,
            flow=flow,
            aquifer_pore_volume=APV,
        )[::2]
        expected = pointwise_oracle(mids, tedges, flow, rech, cin, cr, NH, APV)
        np.testing.assert_allclose(cout, expected, atol=1e-7)  # measured floor 1.1e-8 (breakpoint-straddling bin)

    def test_refinement_invariance(self):
        tedges, flow, rech, cin, cr = nasty_scenario()
        cout_tedges = to_tedges(np.linspace(0.0, 25.4, 12))
        kwargs = {"aquifer_pore_depth": NH, "aquifer_pore_volume": APV, "cout_tedges": cout_tedges}
        base = recharge_to_extraction(cin_recharge=cr, recharge=rech, tedges=tedges, cin=cin, flow=flow, **kwargs)
        rt, (rf, rr, rc, rcr) = refine(tedges, [flow, rech, cin, cr], 7)
        fine = recharge_to_extraction(cin_recharge=rcr, recharge=rr, tedges=rt, cin=rc, flow=rf, **kwargs)
        np.testing.assert_allclose(base, fine, rtol=1e-12)

    @pytest.mark.parametrize("lead_with_surplus", [False, True])
    def test_constant_spinup_equals_explicit_warmup(self, lead_with_surplus):
        """The steady-profile IC equals an explicit 120-day constant warm-up, in both q_b(0) regimes."""
        tedges, flow, rech, cin, cr = nasty_scenario()
        if lead_with_surplus:  # reorder so the rainfall-surplus (q_b < 0) period leads
            order = np.r_[8:13, 0:8, 13:24]
            dt = np.diff((tedges - tedges[0]) / pd.Timedelta(days=1))[order]
            tedges = to_tedges(np.concatenate([[0.0], np.cumsum(dt)]))
            flow, rech, cin, cr = flow[order], rech[order], cin[order], cr[order]
        cout_tedges = to_tedges(np.linspace(0.0, 25.4, 12))
        kwargs = {"aquifer_pore_depth": NH, "aquifer_pore_volume": APV, "cout_tedges": cout_tedges}
        base = recharge_to_extraction(cin_recharge=cr, recharge=rech, tedges=tedges, cin=cin, flow=flow, **kwargs)
        n_pre = 40
        t = (tedges - tedges[0]) / pd.Timedelta(days=1)
        wt = to_tedges(np.concatenate([-3.0 * np.arange(n_pre, 0, -1.0), t]))
        wcr, wre, wci, wfl = (np.concatenate([np.full(n_pre, a[0]), a]) for a in (cr, rech, cin, flow))
        warm = recharge_to_extraction(cin_recharge=wcr, recharge=wre, tedges=wt, cin=wci, flow=wfl, **kwargs)
        np.testing.assert_allclose(base, warm, rtol=1e-12)

    def test_boundary_never_feeding_equals_unbounded(self):
        """With q_b < 0 throughout and flow proportional to recharge, both modes agree exactly.

        Flow proportional to recharge makes the flow-weighted and
        recharge-weighted bin averages identical, isolating the transport
        equivalence.
        """
        tedges, _, rech, cin, cr = nasty_scenario()
        rech += 0.1
        flow = 0.5 * rech * APV / NH  # q_b = -0.5 N A < 0 everywhere
        cout_tedges = to_tedges(np.linspace(1.0, 24.0, 9))
        bounded = recharge_to_extraction(
            cin_recharge=cr,
            recharge=rech,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_depth=NH,
            cin=cin,
            flow=flow,
            aquifer_pore_volume=APV,
        )
        unbounded = recharge_to_extraction(
            cin_recharge=cr, recharge=rech, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_depth=NH
        )
        np.testing.assert_allclose(bounded, unbounded, rtol=1e-12)

    def test_source_fractions_sum_to_one(self):
        """Source apportionment: boundary fraction plus recharge fraction equals one.

        Running the full scenario with (cin=1, cin_recharge=0) yields the
        fraction of extracted water originating at the upstream boundary
        (including its pre-record share via the constant spin-up profile);
        swapping the ones yields the recharge-borne fraction. By linearity and
        conservation the two must sum to exactly one in every defined output
        bin, and each must lie in [0, 1].
        """
        tedges, flow, rech, _, _ = nasty_scenario()
        n = len(flow)
        cout_tedges = to_tedges(np.linspace(0.0, 25.4, 14))
        kwargs = {
            "recharge": rech,
            "tedges": tedges,
            "cout_tedges": cout_tedges,
            "aquifer_pore_depth": NH,
            "flow": flow,
            "aquifer_pore_volume": APV,
        }
        frac_boundary = recharge_to_extraction(cin=np.ones(n), cin_recharge=np.zeros(n), **kwargs)
        frac_recharge = recharge_to_extraction(cin=np.zeros(n), cin_recharge=np.ones(n), **kwargs)
        assert np.all((frac_boundary >= -1e-15) & (frac_boundary <= 1 + 1e-15))
        assert np.all((frac_recharge >= -1e-15) & (frac_recharge <= 1 + 1e-15))
        np.testing.assert_allclose(frac_boundary + frac_recharge, 1.0, rtol=1e-13)

    def test_volume_additivity(self):
        """A coarse output bin equals the volume-weighted combination of its sub-bins exactly."""
        tedges, flow, rech, cin, cr = nasty_scenario()
        fine_edges = np.linspace(4.0, 22.0, 19)
        kwargs = {
            "cin_recharge": cr,
            "recharge": rech,
            "tedges": tedges,
            "aquifer_pore_depth": NH,
            "cin": cin,
            "flow": flow,
            "aquifer_pore_volume": APV,
        }
        coarse = recharge_to_extraction(cout_tedges=to_tedges(fine_edges[[0, -1]]), **kwargs)
        fine = recharge_to_extraction(cout_tedges=to_tedges(fine_edges), **kwargs)
        t = (tedges - tedges[0]) / pd.Timedelta(days=1)
        cum = np.concatenate([[0.0], np.cumsum(flow * np.diff(t))])
        vols = np.diff(np.interp(fine_edges, t, cum))
        np.testing.assert_allclose(coarse[0], np.sum(fine * vols) / np.sum(vols), rtol=1e-14)

    def test_retardation_dilates_clock(self):
        """R = 2 with (Q, N) equals R = 1 with (Q/2, N/2) exactly."""
        tedges, flow, rech, cin, cr = nasty_scenario()
        cout_tedges = to_tedges(np.linspace(0.0, 25.4, 10))
        kwargs = {
            "cin_recharge": cr,
            "tedges": tedges,
            "cout_tedges": cout_tedges,
            "aquifer_pore_depth": NH,
            "cin": cin,
            "aquifer_pore_volume": APV,
        }
        retarded = recharge_to_extraction(recharge=rech, flow=flow, retardation_factor=2.0, **kwargs)
        slowed = recharge_to_extraction(recharge=rech / 2.0, flow=flow / 2.0, **kwargs)
        np.testing.assert_allclose(retarded, slowed, rtol=1e-13)

    @staticmethod
    def assert_matches_oracle_quadrature(tedges, flow, rech, cin, cr, nh, apv, a, b):
        """Compare one wide output bin against flow-weighted quadrature of the pointwise oracle.

        Pins the integral of the solution, not just its values. cout(t) jumps
        where the entry bin of the atom changes (the arrival times of boundary
        parcels released at the input edges), so the quadrature panels are
        aligned to those independently computed arrival times; on the smooth
        segments composite midpoint converges as the square of the step,
        giving an error bound ~1e-8 at this resolution.
        """
        cout = recharge_to_extraction(
            cin_recharge=cr,
            recharge=rech,
            tedges=tedges,
            cout_tedges=to_tedges([a, b]),
            aquifer_pore_depth=nh,
            cin=cin,
            flow=flow,
            aquifer_pore_volume=apv,
        )
        t = ((tedges - tedges[0]) / pd.Timedelta(days=1)).to_numpy(dtype=float)
        dt, k, q = np.diff(t), rech / nh, flow
        arrivals = []  # forward trajectory walk per release edge, independent of the module internals
        for j in range(len(dt)):
            v, tj = apv, t[j]
            for i in range(j, len(dt)):
                if k[i] > 0:
                    v_r = q[i] / k[i]
                    v_end = v_r + (v - v_r) * np.exp(min(k[i] * (t[i + 1] - tj), 700.0))
                else:
                    v_r, v_end = np.inf, v - q[i] * (t[i + 1] - tj)
                if v_end <= 0.0:
                    s = np.log(v_r / (v_r - v)) / k[i] if k[i] > 0 else v / q[i]
                    arrivals.append(tj + s)
                    break
                if v_end >= apv:
                    break
                v, tj = v_end, t[i + 1]
        cuts = np.unique(np.concatenate([[a, b], [x for x in [*arrivals, *t] if a < x < b]]))
        total_mass, total_vol = 0.0, 0.0
        for g1, g2 in pairwise(cuts):
            ts = np.linspace(g1, g2, max(int((g2 - g1) * 4000), 50) + 1)
            mid = 0.5 * (ts[:-1] + ts[1:])
            c_mid = pointwise_oracle(mid, tedges, flow, rech, cin, cr, nh, apv)
            w = flow[np.clip(np.searchsorted(t, mid, side="right") - 1, 0, len(flow) - 1)] * np.diff(ts)
            total_mass += np.sum(c_mid * w)
            total_vol += np.sum(w)
        np.testing.assert_allclose(cout[0], total_mass / total_vol, atol=2e-8)

    @pytest.mark.parametrize(("a", "b"), [(18.0, 23.0), (1.0, 3.875)])
    def test_bin_average_matches_oracle_quadrature(self, a, b):
        """Entered regime (18-23) and pre-record/expulsion regime (1-3.875) of the nasty scenario.

        A kernel term referenced to the wrong piece endpoint preserves weight
        sums, steady states, refinement invariance, and additivity, but fails
        here.
        """
        tedges, flow, rech, cin, cr = nasty_scenario()
        self.assert_matches_oracle_quadrature(tedges, flow, rech, cin, cr, NH, APV, a, b)

    def test_bin_average_across_entry_map_jump(self):
        """Output bin straddling a post-expulsion arrival, with cin != cin_recharge.

        The entry-time map s*(t) jumps discontinuously at the arrival of the
        parcel released when boundary inflow resumes after a rainfall-surplus
        episode; a piece endpoint evaluated on the wrong branch of s* inflates
        the boundary atom by the span of the lost window (regression: the
        original implementation returned 0.995 for this bin, outside the
        attainable range [1, 3], where the true average is 2.81).
        """
        n = 10
        tedges = to_tedges(np.arange(n + 1.0))
        flow = np.array([2.0, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 2.0])
        rech = np.full(n, 0.5)
        self.assert_matches_oracle_quadrature(tedges, flow, rech, np.ones(n), np.full(n, 3.0), 1.0, 3.0, 7.0, 8.0)

    def test_stagnant_bin_zero_flow_and_zero_recharge(self):
        """A bin with zero extraction AND zero recharge: nothing moves, nothing is lost.

        The grazing backward walk parks a parcel exactly on the boundary
        through the stagnant bin and must continue into earlier bins
        (regression: the original implementation crashed on a shape mismatch
        resolving the parked parcel as an entry with no flow to enter by).
        The stagnant output bin carries no water (NaN); conservation holds
        elsewhere, and values with distinct series match the oracle.
        """
        n = 8
        tedges = to_tedges(np.arange(n + 1.0))
        flow = np.array([2.0, 2.0, 0.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        rech = np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
        ones = recharge_to_extraction(
            cin_recharge=np.ones(n),
            recharge=rech,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_depth=1.0,
            cin=np.ones(n),
            flow=flow,
            aquifer_pore_volume=3.0,
        )
        assert np.isnan(ones[2])
        np.testing.assert_allclose(np.delete(ones, 2), 1.0, rtol=1e-14)
        rng = np.random.default_rng(5)
        cin, cr = rng.uniform(0.5, 4.0, n), rng.uniform(0.5, 4.0, n)
        self.assert_matches_oracle_quadrature(tedges, flow, rech, cin, cr, 1.0, 3.0, 3.0, 8.0)

    def test_single_bin_recharge_exceeding_pore_volume(self):
        """One bin receives six pore volumes of rainfall (N * area * dt = 6 * apv).

        The closed forms carry no stability constraint: recharge mixes in and
        the surplus exits continuously, so the bin relaxes toward cin_recharge
        with kernel weight 1 - e^-6 while the pre-existing water (including
        freshly entered boundary water) is flushed out and lost. Verified by
        conservation and by quadrature of the independent oracle across and
        after the extreme bin.
        """
        n = 8
        dt = np.array([1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        tedges = to_tedges(np.concatenate([[0.0], np.cumsum(dt)]))
        rech = np.full(n, 0.4)
        rech[2] = 2.4  # N * A * dt = 2.4 * 2.5 * 2 = 12 = 6 * APV in this bin
        flow = np.full(n, 1.5)  # q_b = +0.5 elsewhere, -4.5 during the extreme bin
        rng = np.random.default_rng(3)
        cin, cr = rng.uniform(0.5, 4.0, n), rng.uniform(0.5, 4.0, n)
        ones = recharge_to_extraction(
            cin_recharge=np.ones(n),
            recharge=rech,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_depth=NH,
            cin=np.ones(n),
            flow=flow,
            aquifer_pore_volume=APV,
        )
        np.testing.assert_allclose(ones, 1.0, rtol=1e-13)
        self.assert_matches_oracle_quadrature(tedges, flow, rech, cin, cr, NH, APV, 1.5, 8.0)

    def test_hand_derived_piston_breakthrough(self):
        """Q = 100, N = 0, apv = 250: the [3, 4] day bin averages cin bins 0 and 1 half-half."""
        n = 10
        tedges = to_tedges(np.arange(n + 1.0))
        cin = np.arange(1.0, n + 1.0)
        cout = recharge_to_extraction(
            cin_recharge=np.zeros(n),
            recharge=np.zeros(n),
            tedges=tedges,
            cout_tedges=to_tedges([3.0, 4.0]),
            aquifer_pore_depth=NH,
            cin=cin,
            flow=np.full(n, 100.0),
            aquifer_pore_volume=250.0,
        )
        np.testing.assert_allclose(cout, (cin[0] + cin[1]) / 2.0, rtol=1e-14)


class TestValidation:
    def setup_method(self):
        self.tedges = to_tedges(np.arange(6.0))
        self.ok = {
            "cin_recharge": np.ones(5),
            "recharge": np.full(5, 0.4),
            "tedges": self.tedges,
            "cout_tedges": self.tedges,
            "aquifer_pore_depth": NH,
        }

    def test_partial_bounded_triple_raises(self):
        with pytest.raises(ValueError, match="provided together"):
            recharge_to_extraction(**self.ok, cin=np.ones(5))
        with pytest.raises(ValueError, match="provided together"):
            recharge_to_extraction(**self.ok, aquifer_pore_volume=APV)
        with pytest.raises(ValueError, match="provided together"):
            recharge_to_extraction(**self.ok, cin=np.ones(5), flow=np.ones(5))

    def test_parity_and_nan_and_sign(self):
        bad = dict(self.ok)
        bad["recharge"] = np.full(4, 0.4)
        with pytest.raises(ValueError, match="one more element"):
            recharge_to_extraction(**bad)
        bad = dict(self.ok)
        bad["cin_recharge"] = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="NaN"):
            recharge_to_extraction(**bad)
        bad = dict(self.ok)
        bad["recharge"] = np.array([0.4, -0.1, 0.4, 0.4, 0.4])
        with pytest.raises(ValueError, match="non-negative"):
            recharge_to_extraction(**bad)

    def test_bounded_inputs_validated(self):
        nan5 = np.array([1.0, np.nan, 1.0, 1.0, 1.0])
        for bad in (
            {"cin": nan5, "flow": np.ones(5)},
            {"cin": np.ones(5), "flow": nan5},
            {"cin": np.ones(5), "flow": np.array([1.0, -1.0, 1.0, 1.0, 1.0])},
            {"cin": np.ones(4), "flow": np.ones(5)},
        ):
            with pytest.raises(ValueError, match=r"NaN|non-negative|one more element"):
                recharge_to_extraction(**self.ok, aquifer_pore_volume=APV, **bad)

    def test_scalar_ranges(self):
        bad = dict(self.ok)
        bad["aquifer_pore_depth"] = 0.0
        with pytest.raises(ValueError, match="positive"):
            recharge_to_extraction(**bad)
        with pytest.raises(ValueError, match="retardation_factor"):
            recharge_to_extraction(**self.ok, retardation_factor=0.5)
        with pytest.raises(ValueError, match="positive"):
            recharge_to_extraction(**self.ok, cin=np.ones(5), flow=np.ones(5), aquifer_pore_volume=-1.0)
