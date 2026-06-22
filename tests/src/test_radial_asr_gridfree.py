"""Tests for the grid-free multi-cycle radial advection-dispersion engine.

Precision discipline (see each test): closed-form / mpmath gates are checked to machine precision
(~1e-12); quantities that pass through the de Hoog numerical Laplace inversion carry its floor
(~1e-7..1e-9, tightened by ``n_terms``); the finite-volume oracle is FIRST-ORDER (``O(1/n_cells)``),
so grid-free-vs-finite-volume agreement is ~1% and the finite-volume solve is shown to *converge to*
the grid-free reference -- the grid-free engine is the reference, not finite-volume.
"""

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from _radial_asr_fv_oracle import fv_cout_deviation  # ty: ignore[unresolved-import]  # tests/src on path via conftest
from _radial_asr_whittaker_oracle import (  # ty: ignore[unresolved-import]  # flint oracle, tests/src on path
    resolvent_oracle,
    transfer_function_oracle,
    whittaker_resolvent_solutions,
)
from flint import acb  # ty: ignore[unresolved-import]  # python-flint is a test-only dependency

from gwtransport._radial_asr_compose import single_cycle_echo_matrix
from gwtransport._radial_asr_gridfree import gridfree_cout_deviation
from gwtransport._radial_asr_kernels import (
    _transfer_riccati,
    interior_resolvent,
    resolvent_riccati,
    rest_resolvent,
)
from gwtransport.radial_asr import infiltration_to_extraction


@pytest.fixture(autouse=True)
def _mp_precision():
    # Scope the mpmath working precision to this module's tests via a context manager -- a module-level
    # mp.mp.dps assignment would leak globally and under-resolve other modules' high-precision fits.
    with mp.workdps(40):
        yield


# --- mpmath ground-truth references (independent of the production assembly) ---------------------
def _mp_airy_greens(s, r, rp, r_w, alpha_l, a0, gauge_sign):
    """Interior Airy resolvent Ghat(r, rp; s) in mpmath (gauge_sign=+1 injection / -1 extraction)."""
    s = mp.mpc(s)
    beta = s / (alpha_l * a0)
    b13 = beta ** (mp.mpf(1) / 3)

    def sol(rr):
        zeta = b13 * rr + beta ** (-mp.mpf(2) / 3) / (4 * alpha_l * alpha_l)
        g = mp.e ** (gauge_sign * rr / (2 * alpha_l))
        gp = gauge_sign / (2 * alpha_l) * g
        ai, aip = mp.airyai(zeta), mp.airyai(zeta, 1)
        bi, bip = mp.airybi(zeta), mp.airybi(zeta, 1)
        return g * ai, gp * ai + g * b13 * aip, g * bi, gp * bi + g * b13 * bip

    u_inf, u_inf_p, u_reg, u_reg_p = sol(r)
    uiw, uiwp, urw, urwp = sol(r_w)
    if gauge_sign < 0:  # Neumann
        bc_reg, bc_inf = urwp, uiwp
    else:  # Robin F[u] = u - alpha_l u'
        bc_reg, bc_inf = urw - alpha_l * urwp, uiw - alpha_l * uiwp
    u0 = bc_reg * u_inf - bc_inf * u_reg
    u0p = bc_reg * u_inf_p - bc_inf * u_reg_p
    ui_rp = sol(rp)[0]
    u0_rp = bc_reg * sol(rp)[0] - bc_inf * sol(rp)[2]
    big_p = mp.e ** (-gauge_sign * r / alpha_l)
    n_s = big_p * (u0 * u_inf_p - u0p * u_inf)
    u0_lt = u0 if r <= rp else u0_rp
    ui_gt = ui_rp if r <= rp else u_inf
    return -u0_lt * ui_gt / n_s


class TestInteriorResolventAiry:
    """The overflow-safe numpy resolvent must match the mpmath construction to double precision."""

    @pytest.mark.parametrize("direction", ["extraction", "injection"])
    @pytest.mark.parametrize(("r", "rp"), [(1.2, 4.0), (4.0, 1.2), (2.0, 50.0), (180.0, 2.0)])  # incl high Pe
    def test_matches_mpmath_no_overflow(self, direction, r, rp):
        r_w, alpha_l, a0 = 0.5, 0.5, 2.0
        gauge = 1.0 if direction == "injection" else -1.0
        s = np.array([0.004 + 0j, 0.03 - 0.02j, 0.4 + 0.1j])
        got = np.array([
            interior_resolvent(s=np.array([sv]), r=r, r_prime=rp, r_w=r_w, alpha_l=alpha_l, a0=a0, direction=direction)[
                0, 0
            ]
            for sv in s
        ])
        assert np.all(np.isfinite(got))
        for i, sv in enumerate(s):
            ref = complex(_mp_airy_greens(sv, mp.mpf(r), mp.mpf(rp), r_w, alpha_l, a0, gauge))
            if abs(ref) > 1e-200:  # skip pure underflow-to-zero (both agree it is ~0)
                np.testing.assert_allclose(got[i], ref, rtol=1e-10)


class TestLaplaceMassBalance:
    """Per-source Laplace conservation identity -- machine precision, NO inversion (the cleanest gate).

    For a unit spectral source at r0 the extraction conservation law dm/dT = -cout transforms to the
    exact operator identity  p*mbar(r0;p) - m0(r0) + Ghat_-(r_w, r0; p) = 0, with
    mbar = int Ghat_-(r, r0; p) 2 c_geo r dr and m0 = 2 c_geo r0 / w_-(r0). Splitting the r-quadrature
    at the kink r = r0 gives geometric convergence, so the only error is Gauss-Legendre quadrature
    (NOT the de Hoog floor): the identity holds to ~1e-11.
    """

    def _defect_airy(self, p, r0, c_geo, r_w, alpha_l, a0, nq=160, r_max=200.0):
        def mbar_panel(lo, hi):
            nodes, wts = np.polynomial.legendre.leggauss(nq)
            r = 0.5 * (hi - lo) * (nodes + 1) + lo
            dr = 0.5 * (hi - lo) * wts
            g = interior_resolvent(
                s=np.array([p]), r=r0, r_prime=r, r_w=r_w, alpha_l=alpha_l, a0=a0, direction="extraction"
            )[0]
            return np.sum(g * 2 * c_geo * r * dr)

        mbar = mbar_panel(r_w, r0) + mbar_panel(r0, r_max)
        w_minus = (2 * c_geo * r0 / alpha_l) * np.exp(r0 / alpha_l)
        m0 = 2 * c_geo * r0 / w_minus
        coutbar = interior_resolvent(
            s=np.array([p]), r=r_w, r_prime=r0, r_w=r_w, alpha_l=alpha_l, a0=a0, direction="extraction"
        )[0, 0]
        return abs(p * mbar - m0 + coutbar) / abs(m0)

    @pytest.mark.parametrize("p", [0.002, 0.05, 0.05 + 0.1j])
    @pytest.mark.parametrize("r0", [3.0, 20.0])
    def test_extraction_conservation(self, p, r0):
        # a0 = 1/(2 c_geo) makes beta = s/(alpha_L a0) = 2 c_geo s/alpha_L the flushed-volume clock, so
        # the conservation measure w_-=(2 c_geo r/alpha_L)e^{r/alpha_L} and m0 = alpha_L e^{-r0/alpha_L} apply.
        c_geo, r_w, alpha_l = np.pi * 10 * 0.3, 0.5, 0.5
        a0 = 1.0 / (2.0 * c_geo)
        assert self._defect_airy(complex(p), r0, c_geo, r_w, alpha_l, a0) < 1e-9


class TestWhittakerResolvent:
    """D_m > 0 interior Green's function: SL constancy and the KB Sec. 7 duality of the flint ORACLE.

    These validate the tests-only flint/Arb Whittaker oracle (``_radial_asr_whittaker_oracle``), the
    machine-precision ground truth the production Riccati path is checked against (in
    ``TestRiccatiVsOracle``). ``whittaker_resolvent_solutions`` returns flint ``acb`` (Arb) values; the
    divergent normalization ``N`` is carried in ``acb`` and only the bounded ratios are cast to double, so
    these identities hold to the double floor (~1e-13, achieved ~1e-16).
    """

    def _greens(self, s, r, rp, r_w, alpha_l, a0, d_m, sigma_a):
        u_inf, u_inf_p, u_reg, u_reg_p = whittaker_resolvent_solutions(s, r, alpha_l, a0, d_m, sigma_a)
        uiw, uiwp, urw, urwp = whittaker_resolvent_solutions(s, r_w, alpha_l, a0, d_m, sigma_a)
        if sigma_a < 0:
            bc_reg, bc_inf = urwp, uiwp
        else:
            fac = alpha_l + d_m * r_w / a0
            bc_reg, bc_inf = urw - fac * urwp, uiw - fac * uiwp
        u0 = bc_reg * u_inf - bc_inf * u_reg
        u0p = bc_reg * u_inf_p - bc_inf * u_reg_p
        b = 1 - sigma_a * a0 / d_m
        # N carries the divergent G^b gauge -- keep it in acb (cast only the bounded ratios below).
        n_s = acb(alpha_l * a0 + d_m * r) ** acb(b) * (u0 * u_inf_p - u0p * u_inf)
        ui_rp = whittaker_resolvent_solutions(s, rp, alpha_l, a0, d_m, sigma_a)[0]
        u0_rp = bc_reg * ui_rp - bc_inf * whittaker_resolvent_solutions(s, rp, alpha_l, a0, d_m, sigma_a)[2]
        u0_lt = u0 if r <= rp else u0_rp
        ui_gt = ui_rp if r <= rp else u_inf
        return -u0_lt * ui_gt / n_s, n_s

    @pytest.mark.parametrize("sigma_a", [-1, 1])
    def test_sl_constancy(self, sigma_a):
        # N(s) = P W is constant in r (SL Abel identity); the divergent N is compared via bounded ratios.
        a0, d_m, alpha_l, r_w = 2.0, 0.6, 0.5, 0.5
        ns = [self._greens(0.1 + 0.05j, r, 3.0, r_w, alpha_l, a0, d_m, sigma_a)[1] for r in (1.0, 8.0, 25.0)]
        spread = max(abs(complex((n - ns[0]) / ns[0])) for n in ns)
        assert spread < 1e-13  # double-precision floor of the float-parameter resolvent (achieved ~1e-16)

    def test_duality_normalization(self):
        # Well-face trace matches the KB Sec. 7 duality cout-readout kernel (pins the normalization).
        a0, d_m, alpha_l, r_w, c_geo = 2.0, 0.6, 0.5, 0.5, np.pi * 10 * 0.3
        for rp in (2.0, 12.0):
            s = 0.08 + 0.04j
            ghat_wf = self._greens(s, r_w, rp, r_w, alpha_l, a0, d_m, -1)[0]
            w_minus = rp * (alpha_l * a0 + d_m * rp) ** (a0 / d_m)
            phi = whittaker_resolvent_solutions(s, rp, alpha_l, a0, d_m, 1)[0]
            phi_w, phi_w_p = whittaker_resolvent_solutions(s, r_w, alpha_l, a0, d_m, 1)[:2]
            flux_w = phi_w - (alpha_l + d_m * r_w / a0) * phi_w_p
            rhs = phi * (2 * c_geo * rp) / (2 * c_geo * a0 * flux_w)
            assert float(abs(complex(ghat_wf * w_minus - rhs)) / abs(complex(rhs))) < 1e-13


class TestRiccatiVsOracle:
    """The production D_m > 0 Riccati path matches the independent flint Whittaker oracle to ~machine precision.

    The Riccati log-derivative kernel (production, pure double precision) is checked against the flint/Arb
    Whittaker oracle (tests-only ground truth) for the transfer function (the four Kreft-Zuber modes) and
    the interior resolvent (both flow orientations). Extraction uses NON-integer A_0/D_m, where the
    oracle's growing-M branch is defined; the production Riccati has no such degeneracy, covered separately
    by :meth:`test_extraction_integer_ratio_is_finite`.
    """

    @pytest.mark.parametrize(("inject", "detect"), [("flux", "flux"), ("flux", "resident")])
    @pytest.mark.parametrize("a0", [5.0, 20.0])  # A_0/D_m = 50, 200
    def test_transfer_matches_oracle(self, inject, detect, a0):
        alpha_l, d_m, r_w, r = 0.5, 0.1, 0.5, 4.0
        s = np.array([0.02 + 0.05j, 0.02 + 0.4j, 0.05 + 1.5j, 0.1 + 6.0j])
        ric = _transfer_riccati(s, r, r_w, alpha_l, a0, d_m, inject, detect)
        ref = transfer_function_oracle(s=s, r=r, r_w=r_w, alpha_l=alpha_l, a0=a0, d_m=d_m, inject=inject, detect=detect)
        np.testing.assert_allclose(ric, ref, rtol=1e-11)  # double-precision agreement vs the Arb oracle

    @pytest.mark.parametrize("direction", ["injection", "extraction"])
    @pytest.mark.parametrize(
        ("a0", "s"),
        [
            # ratio 50: the oracle's growing-M branch is robust across the whole |s| range (incl. small |s|)
            (5.03, np.array([0.02 + 0.05j, 0.05 + 0.4j, 0.1 + 2.0j, 0.2 + 6.0j])),
            # ratio ~120: the oracle is valid only for moderate |s| (its M branch loses precision at large
            # |s| and high ratio -- a production bug the Riccati removes; high-ratio finiteness across all
            # |s| is covered by test_high_ratio_no_cap and the FV oracle).
            (12.03, np.array([0.03 + 0.1j, 0.05 + 0.4j, 0.1 + 2.0j])),
        ],
    )
    def test_resolvent_matches_oracle(self, direction, a0, s):
        alpha_l, d_m, r_w = 0.5, 0.1, 0.5
        rng = np.random.default_rng(0)
        r_nodes = np.linspace(r_w + 0.1, 5.5, 16)
        dr = np.gradient(r_nodes)
        field = rng.standard_normal(16)
        ric = resolvent_riccati(
            s=s,
            field=field,
            r_nodes=r_nodes,
            dr_weights=dr,
            r_w=r_w,
            alpha_l=alpha_l,
            a0_eff=a0,
            d_m_eff=d_m,
            direction=direction,
        )
        ref = resolvent_oracle(s, field, r_nodes, dr, r_w, alpha_l, a0, d_m, direction)
        np.testing.assert_allclose(ric, ref, rtol=1e-11)  # log-space gauge agreement vs the Arb oracle

    @pytest.mark.parametrize(("d_m", "r"), [(20.0, 2.05), (5.0, 2.05), (1.0, 0.65)])
    def test_transfer_near_well_high_dm_vs_oracle(self, d_m, r):
        # Regression for the r_far IC washout: at high D_m, near the well, and small |s| the decaying-branch
        # asymptotic IC must wash out before r reaches r_w. r_far is decay-length-aware (extended by ~22
        # decay lengths of the slowest node), so these match the oracle to ~machine precision -- with the
        # old fixed r_far = 8 r_max the worst case was ~6e-3.
        r_w = 0.5
        s = np.array([0.01 + 0.0j, 0.03 + 0.01j, 0.05 + 0.4j, 0.1 + 2.0j])  # incl tiny-|s| / real nodes
        ric = _transfer_riccati(s, r, r_w, 0.5, 2.0, d_m, "flux", "resident")
        ref = transfer_function_oracle(
            s=s, r=r, r_w=r_w, alpha_l=0.5, a0=2.0, d_m=d_m, inject="flux", detect="resident"
        )
        np.testing.assert_allclose(ric, ref, rtol=1e-9)

    @pytest.mark.slow
    def test_extraction_integer_ratio_is_finite(self):
        # Regression: the flint extraction resolvent returns NaN at integer A_0/D_m (the growing-M branch
        # hits a non-positive-integer pole); the Riccati path has no degeneracy. Compare the engine cout at
        # an exactly-integer ratio to a slightly-perturbed ratio -- finite and continuous through the pole.
        geom = {"c_geo": np.pi * 10 * 0.3, "r_w": 0.5, "alpha_l": 0.5}
        flow = np.array([100.0] * 4 + [-100.0] * 6 + [100.0] * 4 + [-100.0] * 8)  # multi-cycle exercises the resolvent
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        a0 = 100.0 / (2.0 * geom["c_geo"])
        d_m_int = a0 / 3.0  # A_0/D_m = 3 exactly (oracle NaN); d_m_off perturbs off the pole
        on = gridfree_cout_deviation(
            cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=d_m_int, n_quad=8, **geom
        )
        off = gridfree_cout_deviation(
            cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=a0 / 3.02, n_quad=8, **geom
        )
        assert np.all(np.isfinite(on[ext]))
        np.testing.assert_allclose(on[ext], off[ext], atol=5e-3)  # continuous through the integer-ratio pole


# --- engine-level gates -------------------------------------------------------------------------
def _scenario(n_inj, n_ext, rate=100.0):
    flow = np.array([rate] * n_inj + [-rate] * n_ext)
    return flow, np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)


CG = np.pi * 10.0 * 0.3
GEOM = {"c_geo": CG, "r_w": 0.5, "alpha_l": 0.5}


class TestGridfreeEngine:
    def test_single_cycle_equals_echo(self):
        # Grid-free single cycle reduces to the closed-form echo (same machinery) -- de Hoog floor.
        flow, dt, cin = _scenario(8, 24)
        inj_vol = np.concatenate(([0.0], np.cumsum((100.0 * dt)[:8])))
        ext_vol = np.concatenate(([0.0], np.cumsum((100.0 * dt)[8:])))
        w = single_cycle_echo_matrix(
            inj_volume_edges=inj_vol,
            ext_volume_edges=ext_vol,
            inj_flow_scale=100.0,
            ext_flow_scale=100.0,
            n_quad=200,
            **GEOM,
        )
        echo = w @ cin[flow > 0]
        gf = gridfree_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_quad=200, **GEOM)
        np.testing.assert_allclose(gf[flow < 0], echo, atol=1e-8)  # de Hoog floor (achieved ~2e-10)

    def test_multi_cycle_self_convergence(self):
        # The grid-free engine is n_quad-INSENSITIVE once converged: Gauss-Legendre on the smooth
        # resident-profile integrand reaches the de Hoog inversion floor (~1e-9) by a modest node count,
        # so coarser grids already agree with a fine reference to that floor (the remaining ~1% gap vs
        # finite-volume below is the oracle's discretization, not the engine's). The errors at this floor are
        # de-Hoog noise (non-monotone), so the meaningful assertion is that every grid sits at the floor.
        flow = np.array([100.0] * 6 + [-100.0] * 10 + [100.0] * 6 + [-100.0] * 14)
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        ref = gridfree_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_quad=256, **GEOM)
        for nq in (100, 200):
            err = np.max(
                np.abs(
                    gridfree_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_quad=nq, **GEOM)[ext] - ref[ext]
                )
            )
            assert err < 1e-8  # at the de Hoog floor regardless of n_quad (engine is converged)

    def test_fv_converges_to_gridfree_first_order(self):
        # finite-volume is a FIRST-ORDER oracle: its error vs the grid-free reference halves as n_cells doubles,
        # proving the ~1% gap is finite-volume's discretization, not the grid-free engine's.
        flow = np.array([100.0] * 6 + [-100.0] * 10 + [100.0] * 6 + [-100.0] * 14)
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        gf = gridfree_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_quad=64, **GEOM)

        def fv_err(nc, ns):
            fv = fv_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_cells=nc, n_sub=ns, **GEOM)
            return np.max(np.abs(fv[ext] - gf[ext]))

        e_coarse, e_fine = fv_err(100, 4), fv_err(400, 16)
        assert e_fine < 0.4 * e_coarse  # first-order: ~4x cells -> ~4x smaller error

    def test_retardation_recovers_mass_and_matches_echo(self):
        # Retardation R>1: the readout must restore the desorbing sorbed mass (factor R), so total
        # recovery stays ~1. Anchored on the closed-form echo (machine precision); the finite-volume oracle is
        # unvalidated under R, so it is NOT used here. Exercises the xR in both single_cycle_echo_matrix
        # and _cout_phase (the grid-free engine run directly on a single cycle).
        flow, dt, cin = _scenario(8, 24)
        inj_vol = np.concatenate(([0.0], np.cumsum((100.0 * dt)[:8])))
        ext_vol = np.concatenate(([0.0], np.cumsum((100.0 * dt)[8:])))
        r_fac = 2.5
        w = single_cycle_echo_matrix(
            inj_volume_edges=inj_vol,
            ext_volume_edges=ext_vol,
            inj_flow_scale=100.0,
            ext_flow_scale=100.0,
            n_quad=200,
            retardation_factor=r_fac,
            **GEOM,
        )
        echo = w @ cin[flow > 0]
        gf = gridfree_cout_deviation(
            cin_deviation=cin, flow=flow, dt_days=dt, retardation_factor=r_fac, n_quad=200, **GEOM
        )
        np.testing.assert_allclose(gf[flow < 0], echo, atol=2e-6)  # grid-free == echo under R (de Hoog floor)
        # Recovery ~0.99 (a little of the R-stretched tail spills past the finite window); the xR fix is
        # decisive vs the bug, which would give 1/R = 0.4. > 0.95 cleanly separates the two.
        assert np.sum(echo * 100.0) / np.sum(cin[flow > 0] * 100.0) > 0.95

    def test_ensemble_equals_weighted_mean_of_disks(self):
        # Multi-cycle ensemble over disk heights = weighted mean of per-disk grid-free solves (the
        # _gridfree_ensemble averaging), to machine precision (pure averaging, no new inversion).
        flow = np.array([100.0] * 4 + [-100.0] * 8 + [100.0] * 4 + [-100.0] * 10)
        tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
        cin = np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        porosity, heights = 0.3, np.array([6.0, 14.0])
        geom = {"porosity": porosity, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
        ens = infiltration_to_extraction(
            cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, pore_heights=heights, n_quad=80, **geom
        )
        manual = np.mean(
            [
                gridfree_cout_deviation(
                    cin_deviation=cin,
                    flow=flow,
                    dt_days=np.ones(len(flow)),
                    c_geo=np.pi * b * porosity,
                    r_w=0.5,
                    alpha_l=0.5,
                    n_quad=80,
                )[ext]
                for b in heights
            ],
            axis=0,
        )
        np.testing.assert_allclose(ens[ext], manual, rtol=1e-12)


class TestMolecularDiffusion:
    @pytest.mark.slow
    def test_appreciable_dm_vs_fv(self):
        # Appreciable molecular diffusion routes to the D_m > 0 Riccati branch; cross-check the
        # de-Hoog-inverted breakthrough vs the finite-volume oracle (~1% first-order floor). Single cycle:
        # this exercises the Riccati kernel + FR profile + duality readout end-to-end. The multi-cycle
        # resolvent hand-off (_propagate_diffusive) is gated separately by TestRiccatiVsOracle (machine
        # precision vs the flint oracle). Still @slow -- the per-node ODE solve at appreciable A_0/D_m.
        flow = np.array([100.0] * 3 + [-100.0] * 7)
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        d_m = 1.5  # A0/D_m = 3.5 -> a* = 1.77 inside the plume (Whittaker needed and tractable)
        gf = gridfree_cout_deviation(
            cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=d_m, n_quad=16, **GEOM
        )
        fv = fv_cout_deviation(
            cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=d_m, n_cells=600, n_sub=12, **GEOM
        )
        assert np.max(np.abs(gf[ext] - fv[ext])) < 3e-2  # finite-volume first-order floor

    def test_high_ratio_no_cap(self):
        # No A_0/D_m cap any more, and no small-|s| NaN with it: the Riccati kernel is finite and
        # non-trivial at heat / seasonal-storage ratios (A_0/D_m = 1000) where the old flint kernel hit its
        # precision wall and NaN'd at the small-|s| de Hoog nodes. Both orientations, small AND large |s|.
        alpha_l, d_m, r_w = 0.5, 0.1, 0.5
        rng = np.random.default_rng(1)
        r_nodes = np.linspace(r_w + 0.1, 6.0, 20)
        dr, field = np.gradient(r_nodes), rng.standard_normal(20)
        s = np.array([0.02 + 0.01j, 0.05 + 0.4j, 0.2 + 6.0j])  # incl small |s| (the old flint NaN regime)
        a0 = 1000.0 * d_m  # A_0/D_m = 1000, far beyond the old cap (150, then 1000)
        for direction in ("injection", "extraction"):
            f = resolvent_riccati(
                s=s,
                field=field,
                r_nodes=r_nodes,
                dr_weights=dr,
                r_w=r_w,
                alpha_l=alpha_l,
                a0_eff=a0,
                d_m_eff=d_m,
                direction=direction,
            )
            assert np.all(np.isfinite(f))
            assert np.max(np.abs(f)) > 0.0  # non-trivial (not silently zeroed)
        g = _transfer_riccati(s, 5.0, r_w, alpha_l, a0, d_m, "flux", "resident")
        assert np.all(np.isfinite(g))
        assert np.all(np.abs(g) > 0.0)
        # value-pin at ratio 1000: the transfer oracle (Tricomi-U) IS finite/correct for moderate |s|
        # (only its resolvent M-branch overflows there), so the transfer can be checked exactly, not just
        # for finiteness.
        s_mod = np.array([0.05 + 0.4j, 0.1 + 1.0j, 0.2 + 6.0j])
        gp = _transfer_riccati(s_mod, 5.0, r_w, alpha_l, a0, d_m, "flux", "flux")
        ref = transfer_function_oracle(s=s_mod, r=5.0, r_w=r_w, alpha_l=alpha_l, a0=a0, d_m=d_m)
        np.testing.assert_allclose(gp, ref, rtol=1e-11)


class TestRestDiffusion:
    """Molecular / thermal diffusion during a rest (``Q = 0``) -- the seasonal-storage / ATES regime.

    Rest diffusion runs on the wall-clock clock (KB Sec. 3): advection and mechanical dispersion pause
    but molecular diffusion does not, so a long rest smears the plume. It is carried by the order-0
    modified Bessel pure-diffusion kernel (``rest_resolvent``). These gate the new kernel (machine
    precision), the engine's rest-sensitivity (rest was a no-op before this fix), and an exact
    end-to-end cross-check vs the FV oracle (which integrates diffusion through the rest).
    """

    def test_rest_resolvent_matches_mpmath(self):
        # Overflow-safe Bessel rest resolvent vs an independent mpmath construction (incl. high kappa*r,
        # where the unscaled I_0 would overflow) -- the new-kernel correctness gate, machine precision.
        d_m, r_w = 0.5, 0.5

        def ref(s, r, rp):
            s = mp.mpc(s)
            k = mp.sqrt(s / d_m)
            rl, rg = (r, rp) if r <= rp else (rp, r)
            u0 = mp.besselk(1, k * r_w) * mp.besseli(0, k * rl) + mp.besseli(1, k * r_w) * mp.besselk(0, k * rl)
            return u0 * mp.besselk(0, k * rg) / mp.besselk(1, k * r_w)

        for s in (0.01 + 0j, 0.4 + 0.2j, 2.0 + 1.0j):
            for r, rp in ((1.2, 4.0), (4.0, 1.2), (2.0, 30.0)):
                got = rest_resolvent(s=np.array([s]), r=r, r_prime=rp, r_w=r_w, d_m=d_m)[0, 0]
                assert abs(got - complex(ref(s, r, rp))) / abs(complex(ref(s, r, rp))) < 1e-12

    @pytest.mark.slow
    def test_seasonal_rest_matches_fv_oracle(self):
        # Exact end-to-end: inject -> long (200 d) rest -> extract. The rest diffusion (Bessel, exact)
        # dominates; gridfree must match the FV oracle (which integrates D_m through the rest). A broken
        # rest measure/wiring would give a ~0.5 gap, so this cleanly gates the end-to-end magnitude.
        # @slow: the D_m>0 pumping phases now go through the per-node Riccati ODE (no more Airy reduction).
        flow = np.array([100.0] * 10 + [0.0] * 200 + [-50.0] * 40)
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        gf = gridfree_cout_deviation(
            cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=0.05, n_quad=24, **GEOM
        )
        fv = fv_cout_deviation(
            cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=0.05, n_cells=600, n_sub=8, **GEOM
        )
        assert np.max(np.abs(gf[ext] - fv[ext])) < 3e-2  # FV first-order floor (the gridfree side is exact)
