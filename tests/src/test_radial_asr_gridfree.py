"""Tests for the grid-free multi-cycle radial advection-dispersion engine.

Precision discipline (see each test): closed-form / mpmath gates are checked to machine precision
(~1e-12); quantities that pass through the de Hoog numerical Laplace inversion carry its floor
(~1e-7..1e-9, tightened by ``n_terms``); the finite-volume oracle is FIRST-ORDER (``O(1/n_cells)``),
so grid-free-vs-finite-volume agreement is ~1% and the finite-volume solve is shown to *converge to*
the grid-free reference -- the grid-free engine is the reference, not finite-volume.
"""

import contextlib

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from _radial_asr_fv_oracle import fv_cout_deviation  # ty: ignore[unresolved-import]  # tests/src on path via conftest
from _radial_asr_gridfree_oracle import (  # ty: ignore[unresolved-import]  # per-reversal reference, tests/src on path
    _cout_phase,
    _fr_profile,
    _propagate,
    _propagate_diffusive,
    _propagate_rest,
    gridfree_cout_deviation,
)
from _radial_asr_whittaker_oracle import (  # ty: ignore[unresolved-import]  # flint oracle, tests/src on path
    resolvent_oracle,
    transfer_function_oracle,
    whittaker_resolvent_solutions,
)
from flint import acb  # ty: ignore[unresolved-import]  # python-flint is a test-only dependency

from gwtransport._radial_asr_compose import single_cycle_echo_matrix
from gwtransport._radial_asr_kernels import (
    _transfer_riccati,
    interior_resolvent,
    resolvent_riccati,
    rest_resolvent,
)
from gwtransport._radial_asr_reuse import (
    _airy_propagator_matrix,
    _cout_readout_matrix,
    _field_grid,
    _fr_source_matrix,
    _rest_propagator_matrix,
    _riccati_propagator_matrix,
    cout_deviation,
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
                cout_deviation(
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
        # The ensemble is a pure weighted average of the per-disk engine (no new inversion), so it equals
        # the manual mean of the same production engine (cout_deviation) to machine precision.
        np.testing.assert_allclose(ens[ext], manual, rtol=1e-12)

    @pytest.mark.parametrize("alpha_l", [8.0, 3.0])
    def test_echo_v_window_conserves_mass_at_low_peclet(self, alpha_l):
        # The resident-profile V' quadrature window must be Peclet-aware. At low Peclet (large alpha_L,
        # r_front/alpha_L << 1) the profile spreads over many breakthrough widths; a flat 4 S_inj/R margin
        # truncates that tail and loses mass. Isolate the window from any finite-extraction loss with ONE
        # injection bin and ONE huge extraction bin (every parcel arrives, so G1(T_end; V') -> 1): then
        # recovered = W[0, 0] * T_end = int f(V') dV' = S_inj/R exactly, and any deficit is pure window
        # truncation. Before the Peclet-aware window this lost ~1.5e-2 at alpha_L = 8 / ~3e-4 at alpha_L = 3.
        c_geo, r_w, s_inj, t_end = 25.0, 0.5, 5000.0, 5.0e7
        w = single_cycle_echo_matrix(
            inj_volume_edges=np.array([0.0, s_inj]),
            ext_volume_edges=np.array([0.0, t_end]),
            c_geo=c_geo,
            r_w=r_w,
            alpha_l=alpha_l,
            inj_flow_scale=100.0,
            ext_flow_scale=100.0,
            n_quad=240,
        )
        np.testing.assert_allclose(w[0, 0] * t_end, s_inj, rtol=1e-4)


@contextlib.contextmanager
def _tight_dehoog():
    """Force the per-reversal field-propagation de Hoog (oracle namespace) to (64, 1e-13).

    Patches only the public ``dehoog_inverse`` the per-reversal ``_propagate*`` use, forcing tight terms
    regardless of the passed ``n_terms``/``tol``, so the per-reversal engine matches a matrix builder /
    ``cout_deviation`` run at ``n_terms=64, tol=1e-13`` -- the two then agree to the genuine de Hoog floor
    (no Pade-nonlinearity gap), the matched-settings machine-precision contract.
    """
    import _radial_asr_gridfree_oracle as _ga  # ty: ignore[unresolved-import]  # noqa: PLC0415 -- tests/src on path

    import gwtransport._radial_asr_dehoog as _dh  # noqa: PLC0415

    orig_inv = _ga.dehoog_inverse

    def tight_inv(*, f_hat, t, n_terms=64, scaling=None, alpha=0.0, tol=1e-13):  # noqa: ARG001 -- force tight terms
        return _dh.dehoog_inverse(f_hat=f_hat, t=t, n_terms=64, scaling=scaling, alpha=alpha, tol=1e-13)

    _ga.dehoog_inverse = tight_inv
    try:
        yield
    finally:
        _ga.dehoog_inverse = orig_inv


def _gridfree_tight(**kw):
    """``gridfree_cout_deviation`` with its field-propagation de Hoog forced to (64, 1e-13)."""
    with _tight_dehoog():
        return gridfree_cout_deviation(**kw)


class TestReuseEngine:
    """The reused-propagator-matrix engine (``cout_deviation``) reproduces the per-reversal grid-free engine
    (``gridfree_cout_deviation``) by caching each phase's field-propagator as a matrix and reusing it. The
    load-bearing identity is that each matrix COLUMN is exactly the per-reversal ``_propagate`` of a unit
    source -- the same single de Hoog inversion of the same kernel -- so the matrix *is* the per-reversal
    operator (bit-exact, independent of Peclet). End-to-end, ``P @ field`` then equals the per-reversal
    ``_propagate(field)`` to the de Hoog floor (the only gap is the Pade acceleration's mild nonlinearity,
    ``invert-then-sum`` vs ``sum-then-invert``, which tightens with ``n_terms``/``tol`` and is never an
    accuracy regression)."""

    def test_single_cycle_is_exact(self):
        # K=1 never invokes the propagator (inject builds the field, extract reads it -- no inter-phase
        # hand-off), so the reuse engine equals the per-reversal engine to machine precision: the readout is
        # the same FR step response applied as M_cout @ field (a matrix-multiply) rather than a loop sum, so
        # it agrees up to the BLAS-vs-loop summation order (~1e-15), not via a fresh de Hoog inversion.
        flow, dt, cin = _scenario(8, 24)
        gf = gridfree_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_quad=120, **GEOM)
        re = cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_quad=120, **GEOM)
        np.testing.assert_allclose(re, gf, rtol=1e-13, atol=1e-14, equal_nan=True)

    def test_readout_matrices_match_loops(self):
        # The injection-source M_fr (cin->field) and cout-readout M_cout (field->cout) operators reproduce the
        # per-phase _fr_profile / _cout_phase loops they replace, to machine precision -- the same FR step
        # response applied as a matrix-multiply rather than a loop sum. This pins the readout-matrix reuse
        # directly, independent of how a schedule composes phases.
        flow, dt = np.array([100.0] * 4 + [-100.0] * 4), np.ones(8)
        r_nodes, v_nodes, dr = _field_grid(flow, dt, CG, 0.5, 0.5, 0.0, 60)
        dv = 2.0 * CG * r_nodes * dr
        ro = {
            "c_geo": CG,
            "r_w": 0.5,
            "alpha_l": 0.5,
            "retardation_factor": 1.0,
            "flow_scale": 100.0,
            "molecular_diffusivity": 0.0,
        }
        edges = np.array([0.0, 100.0, 200.0, 300.0, 400.0])  # 4 within-phase volume bins
        cin = np.array([1.0, -0.5, 0.3, 0.8])
        field = np.random.default_rng(0).standard_normal(60)
        m_fr = _fr_source_matrix(v_nodes, edges, **ro)
        np.testing.assert_allclose(m_fr @ cin, _fr_profile(v_nodes, cin, edges, **ro), atol=1e-14, rtol=1e-12)
        m_cout = _cout_readout_matrix(v_nodes, dv, edges, **ro)
        np.testing.assert_allclose(m_cout @ field, _cout_phase(field, v_nodes, dv, edges, **ro), atol=1e-13, rtol=1e-11)

    @pytest.mark.parametrize("direction", ["injection", "extraction"])
    def test_airy_matrix_is_per_reversal_operator(self, direction):
        # THE correctness identity (D_m=0): each cached-matrix column equals the per-reversal _propagate of a
        # unit source at that node -- the same single de Hoog inversion of the same Airy kernel. The reuse
        # matrix folds the Sturm-Liouville source weight e^{-gauge r'/alpha_L} into assemble_airy_resolvent's
        # log-exponents (overflow-safe at any Peclet -- the divergent injection branch and the extraction
        # weight would otherwise blow up past r/alpha_L ~ 700 and meet as Inf * 0 = NaN), while _propagate
        # applies it as a separate post-multiply. The two therefore differ only by exp(a + b) vs exp(a) e^b
        # reorder noise (~1e-14 in the transform, amplified by the de Hoog condition to ~1e-8 in the column) --
        # the same benign effect the Riccati sibling below tolerates. Measured worst |Δcol| = 7e-8 across the
        # three probed columns/directions, so atol=5e-7 keeps ~7× headroom over the real reorder noise while a
        # sign / structural error is O(|col|) ~ 1e-2..1e-1, five orders of magnitude above the tolerance.
        flow, dt = np.array([100.0] * 4 + [-100.0] * 4), np.ones(8)
        r_nodes, _, dr = _field_grid(flow, dt, CG, 0.5, 0.5, 0.0, 60)
        tau = 400.0  # flushed-volume tau = phase_volume (the D_m=0 Airy kernel is flow-magnitude free)
        gauge = -1.0 if direction == "injection" else 1.0
        sl = (2.0 * CG * r_nodes / 0.5) * np.exp(gauge * r_nodes / 0.5) * dr
        pmat = _airy_propagator_matrix(
            direction, tau, r_nodes, dr, r_w=0.5, alpha_l=0.5, c_geo=CG, n_terms=44, tol=1e-9
        )
        for j in (5, 30, 55):
            e = np.zeros(60)
            e[j] = 1.0
            col = _propagate(e, r_nodes, sl, direction, tau, r_w=0.5, alpha_l=0.5, c_geo=CG)
            np.testing.assert_allclose(pmat[:, j], col, rtol=1e-6, atol=5e-7)

    @pytest.mark.parametrize("direction", ["injection", "extraction"])
    def test_airy_matrix_finite_at_high_peclet(self, direction):
        # At Peclet r_max/alpha_L > ~700 the growing-Airy injection branch (assemble's e^{+g r_sum}) and the
        # extraction Sturm-Liouville weight e^{+r'/alpha_L} each overflow double precision; only folding the
        # weight into the assembly's log-exponents keeps their bounded product finite. Before the fold the
        # propagator was all-NaN, which the ensemble's nan_to_num silently mapped to a physical zero (cout=0).
        alpha_l, r_w, c_geo = 0.18, 0.5, np.pi * 30.0 * 0.3
        flow = np.array([5000.0] * 60 + [-10000.0] * 15 + [5000.0] * 30 + [-10000.0] * 30)
        dt = np.ones(len(flow))
        r_nodes, _, dr = _field_grid(flow, dt, c_geo, r_w, alpha_l, 0.0, 120)
        assert r_nodes.max() / alpha_l > 700.0  # the regime that overflowed before the fold
        signed = flow[flow > 0] if direction == "injection" else -flow[flow < 0]
        tau = float(np.sum(signed))  # flushed-volume tau = phase_volume
        pmat = _airy_propagator_matrix(
            direction,
            tau,
            r_nodes,
            dr,
            r_w=r_w,
            alpha_l=alpha_l,
            c_geo=c_geo,
            n_terms=44,
            tol=1e-9,
        )
        assert np.all(np.isfinite(pmat))
        assert np.max(np.abs(pmat)) < 1.0  # the bounded physical field hand-off (|P| ~ O(1), not amplifying)

    @pytest.mark.slow
    @pytest.mark.parametrize(("direction", "r_fac"), [("injection", 1.0), ("extraction", 1.0), ("extraction", 2.0)])
    def test_riccati_matrix_is_per_reversal_operator(self, direction, r_fac):
        # D_m>0 correctness identity: each Riccati matrix column equals the per-reversal _propagate_diffusive
        # of a unit source. This per-column anchor is REQUIRED -- the end-to-end differential is sign-blind
        # (injection/extraction propagations pair per cycle, so an overall sign flip cancels). The outer-product
        # vs resolvent_riccati cumsum reorder differs at ~1e-15, amplified by the de Hoog condition (~1e8) to
        # ~1e-7 at matched-tight settings, so a sign or structural error (col_err ~ |col|) is caught with a
        # ~1e5 margin below.
        flow, dt = np.array([100.0] * 4 + [-100.0] * 4), np.ones(8)
        r_nodes, _, dr = _field_grid(flow, dt, CG, 0.5, 0.5, 1.0, 24)
        tau, flow_scale = 4.0, 100.0  # wall-clock tau = phase_time
        pmat = _riccati_propagator_matrix(
            direction,
            tau,
            r_nodes,
            dr,
            r_w=0.5,
            alpha_l=0.5,
            c_geo=CG,
            flow_scale=flow_scale,
            molecular_diffusivity=1.0,
            retardation_factor=r_fac,
            n_terms=64,
            tol=1e-13,
        )
        for j in (3, 12, 21):
            e = np.zeros(24)
            e[j] = 1.0
            with _tight_dehoog():  # match _propagate_diffusive's de Hoog to the matrix's (64, 1e-13)
                col = _propagate_diffusive(
                    e,
                    r_nodes,
                    dr,
                    direction,
                    tau,
                    r_w=0.5,
                    alpha_l=0.5,
                    flow_scale=flow_scale,
                    c_geo=CG,
                    molecular_diffusivity=1.0,
                    retardation_factor=r_fac,
                )
            np.testing.assert_allclose(pmat[:, j], col, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize(
        ("alpha_l", "r_fac"),
        [(0.25, 1.0), (0.5, 1.0), (1.0, 1.0), (0.5, 2.0)],  # front-Peclet ~45 / 22 / 11; last adds D_m=0 R>1
    )
    def test_matches_per_reversal(self, alpha_l, r_fac):
        # D_m=0 multi-cycle end-to-end correctness anchor. At MATCHED-TIGHT de Hoog the reuse engine equals the
        # per-reversal engine to the shared-code floor; the only residual is the Pade nonlinearity of the
        # intermediate field hand-offs (invert-then-sum P@field vs sum-then-invert _propagate), which closes as
        # n_terms tightens and is Peclet-dependent (~1e-11 at Pe~45 down to ~5e-8 at Pe~11). R>1 exercises the
        # Airy tau=V/R clock rescale and the M_cout xR readout. (The column-identity test pins each matrix
        # bit-exactly; this gate adds caching + composition. The shipped-settings gap is the same Pade residual,
        # below the de Hoog floor itself, so comparing at matched-tight is both stricter and non-fragile.)
        flow = np.array(([100.0] * 4 + [-100.0] * 4) * 3)
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        geom = {**GEOM, "alpha_l": alpha_l}
        gf = _gridfree_tight(cin_deviation=cin, flow=flow, dt_days=dt, retardation_factor=r_fac, n_quad=120, **geom)
        re = cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            retardation_factor=r_fac,
            n_quad=120,
            n_terms=64,
            tol=1e-13,
            **geom,
        )
        assert np.max(np.abs(re[ext] - gf[ext])) < 1e-6

    @pytest.mark.slow
    @pytest.mark.parametrize("r_fac", [1.0, 2.0])
    def test_dm_positive_matches_per_reversal(self, r_fac):
        # D_m>0 (Riccati branch) multi-cycle end-to-end: the reused matrix reproduces the per-reversal
        # _propagate_diffusive hand-off to the matched-tight de Hoog floor (the column-identity test above
        # pins the matrix bit-exactly; this gate adds caching + composition). R=1 and R>1 via the wall-clock
        # rescale. n_quad small -- the cost is the per-de-Hoog-node Riccati ODE.
        flow = np.array(([100.0] * 4 + [-100.0] * 4) * 2)
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        gf = _gridfree_tight(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            molecular_diffusivity=1.0,
            retardation_factor=r_fac,
            n_quad=8,
            **GEOM,
        )
        re = cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            molecular_diffusivity=1.0,
            retardation_factor=r_fac,
            n_quad=8,
            n_terms=64,
            tol=1e-13,
            **GEOM,
        )
        assert np.max(np.abs(re[ext] - gf[ext])) < 1e-5  # diffusive (high-ratio D_m) de Hoog floor

    @pytest.mark.slow
    def test_seasonal_matches_per_reversal(self):
        # D_m>0 with Q=0 rest phases (seasonal storage / ATES, inject -> rest -> extract): the Riccati and
        # Bessel-rest matrices are both reused across cycles and reproduce the per-reversal engine end-to-end
        # to the matched-tight de Hoog floor (the per-branch matrices are pinned bit-exactly above).
        flow = np.array(([100.0] * 4 + [0.0] * 3 + [-100.0] * 4) * 2)
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        gf = _gridfree_tight(cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=1.0, n_quad=8, **GEOM)
        re = cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            molecular_diffusivity=1.0,
            n_quad=8,
            n_terms=64,
            tol=1e-13,
            **GEOM,
        )
        assert np.max(np.abs(re[ext] - gf[ext])) < 1e-5  # diffusive + rest de Hoog floor

    @pytest.mark.slow
    @pytest.mark.parametrize("d_m_eff", [1.0, 2.0])  # d_m_eff != 1 exercises the 1/d_m_eff source measure
    def test_rest_matrix_is_per_reversal_operator(self, d_m_eff):
        # Rest (Q=0) Bessel branch: each matrix column equals _propagate_rest of a unit source (seasonal
        # storage / ATES wall-clock molecular diffusion). Shares rest_resolvent with _propagate_rest, so the
        # column is bit-exact (the SL measure r'/d_m_eff is identity at d_m_eff=1, so d_m_eff=2 pins it).
        flow, dt = np.array([100.0] * 4 + [0.0] * 3 + [-100.0] * 4), np.ones(11)
        r_nodes, _, dr = _field_grid(flow, dt, CG, 0.5, 0.5, 1.0, 40)
        tau = 3.0  # wall-clock rest duration
        pmat = _rest_propagator_matrix(tau, r_nodes, dr, r_w=0.5, d_m_eff=d_m_eff, n_terms=44, tol=1e-9)
        for j in (5, 20, 35):
            e = np.zeros(40)
            e[j] = 1.0
            col = _propagate_rest(e, r_nodes, dr, tau, r_w=0.5, d_m_eff=d_m_eff)
            np.testing.assert_allclose(pmat[:, j], col, atol=1e-9, rtol=1e-8)

    def test_multi_cycle_converges_to_finite_volume(self):
        # Fully independent end-to-end anchor for the accelerated path: a finite-volume solve of the SAME PDE
        # (shares NO Laplace-kernel / de Hoog / FR-step-response code) CONVERGES first-order to the reuse engine
        # on a multi-cycle schedule that actually exercises the propagator-matrix reuse. Every other reuse test
        # compares against the kernel-sharing per-reversal engine; this closes the validation chain through the
        # accelerated path itself. The reuse engine is the reference; FV is first-order O(1/n_cells), so refining
        # the grid drives the error toward zero (a wrong engine would leave a non-vanishing offset, ratio -> 1).
        flow = np.array(([100.0] * 4 + [-100.0] * 4) * 2)
        dt, cin = np.ones(len(flow)), np.where(flow > 0, 1.0, 0.0)
        ext = flow < 0
        re = cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_quad=120, **GEOM)

        def err(n_cells, n_sub):
            fv = fv_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, n_cells=n_cells, n_sub=n_sub, **GEOM)
            return np.max(np.abs(fv[ext] - re[ext]))

        assert err(800, 16) < 0.6 * err(200, 8)  # refining 4x reduces the FV error toward the reuse engine


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

    def test_rest_resolvent_high_frequency_no_overflow(self):
        # At a high Laplace frequency Re(kappa r_w) exceeds ~354, where splitting the outer term's bounded
        # combined exponent into exp(|Re z_w| + z_w) * exp(-z_lt - z_gt) overflows the first factor to Inf and
        # forms Inf * 0 = NaN, defeating the scaled Bessels. The single-combined-exponent form stays finite and
        # matches mpmath (arbitrary precision, which represents the huge I_1). r ~ r_w keeps the true (tiny)
        # value above the double-precision underflow floor so the match is a real relative comparison.
        d_m, r_w = 0.5, 0.5

        def ref(s, r, rp):
            s = mp.mpc(s)
            k = mp.sqrt(s / d_m)
            rl, rg = (r, rp) if r <= rp else (rp, r)
            u0 = mp.besselk(1, k * r_w) * mp.besseli(0, k * rl) + mp.besseli(1, k * r_w) * mp.besselk(0, k * rl)
            return u0 * mp.besselk(0, k * rg) / mp.besselk(1, k * r_w)

        s = 720.0**2 * d_m  # Re(kappa r_w) = 360 > 354 -> the split outer factor overflows to Inf
        for r, rp in ((0.6, 0.6), (0.55, 0.7)):
            got = rest_resolvent(s=np.array([s + 0j]), r=r, r_prime=rp, r_w=r_w, d_m=d_m)[0, 0]
            expected = complex(ref(s, r, rp))
            assert np.isfinite(got)
            assert abs(got - expected) / abs(expected) < 1e-10

    def test_rest_propagator_conserves_mass_across_regimes(self):
        # RA2: the rest (Q=0) propagator must be finite AND must never AMPLIFY resident mass -- the diffusion
        # invariant dv^T P <= dv per source column (amplification is a maximum-principle violation that drives
        # cout above cin end-to-end). Two regimes are exercised on the same grid:
        #   sub-grid (small D_m): the diffusion length sqrt(D_m tau) drops below the grid spacing, so the
        #     Bessel resolvent's near-delta Green's function is unresolved. The quadratured matrix AMPLIFIED
        #     mass here (dv^T P reached ~2.4x -> cout > cin, recovered mass > injected) and, before the de Hoog
        #     underflow guard, its far-field cells were NaN and collapsed cout to background. The
        #     resolution-limit guard returns the identity (diffusion has not crossed a cell), so mass is
        #     EXACTLY conserved and the rest reduces to the D_m=0 echo (the correct small-D_m limit).
        #   resolved (larger D_m): the de Hoog matrix is used; it conserves interior mass and only loses a
        #     little to the far field at the outer boundary -- never amplifies.
        r_w, alpha_l, c_geo = 0.5, 0.5, np.pi * 10.0 * 0.35
        flow = np.array([5.0] * 40 + [0.0] + [-5.0] * 40)
        dt = np.ones(len(flow))
        for d_m, sub_grid in ((1e-3, True), (0.05, False)):
            r_nodes, _, dr = _field_grid(flow, dt, c_geo, r_w, alpha_l, d_m, 120)
            dv = 2.0 * c_geo * r_nodes * dr  # radial mass measure: mass = dv @ field
            pmat = _rest_propagator_matrix(1.0, r_nodes, dr, r_w=r_w, d_m_eff=d_m, n_terms=44, tol=1e-9)
            assert np.all(np.isfinite(pmat))
            col_mass = dv @ pmat  # resident mass produced per unit source; must never exceed the source mass
            assert np.max(col_mass / dv) <= 1.0 + 1e-6  # no amplification (pre-fix reached ~2.4x at small D_m)
            if sub_grid:  # sub-grid rest is the identity to grid accuracy -> mass EXACTLY conserved
                np.testing.assert_allclose(col_mass, dv, rtol=1e-12)

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


def _tedges_from_dt(dt):
    """DatetimeIndex bin edges (n+1) whose day-widths equal ``dt`` -- for the public ``radial_asr`` API."""
    return pd.Timestamp("2020-01-01") + pd.to_timedelta(np.concatenate(([0.0], np.cumsum(dt))), unit="D")


def _assemble(segments):
    """Concatenate a list of ``(flow, dt, cin)`` phase segments into flat ``(flow, dt, cin)`` arrays."""
    return tuple(np.concatenate(col) for col in zip(*segments, strict=True))


def _i2e(segments, n_quad):
    """Run the public ``infiltration_to_extraction`` on assembled phase segments (fixed ASR geometry)."""
    flow, dt, cin = _assemble(segments)
    tedges = _tedges_from_dt(dt)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        pore_heights=10.0,
        porosity=0.3,
        well_radius=0.5,
        longitudinal_dispersivity=0.5,
        molecular_diffusivity=0.05,
        n_quad=n_quad,
    )


def _rebinning_schedules(direction):
    r"""Two binnings (A, B) of one physical Q(t): a tracer push, then a clean probe phase binned coarsely
    (A: 2 bins where ``|Q|`` and ``dt`` covary, so ``mean(|Q|) != phase_volume/phase_time``) vs. finely
    (B: the identical physical flow in uniform 1-day bins), then a final extraction read on 6 shared bins.

    The probe (``injection`` or ``extraction``) is an intermediate phase, so its across-reversal propagator
    hand-off carries the ``flow_scale`` defect. Returns ``(segments_A, segments_B)``; the last 6 bins are the
    identical extraction readout in both.
    """
    read = [(np.full(6, -100.0), np.full(6, 2.0), np.zeros(6))]  # final extraction: read cout on 6 shared bins
    if direction == "injection":
        pre = [
            (np.full(4, 100.0), np.ones(4), np.ones(4)),  # tracer push (cin deviation = 1)
            (np.full(2, -100.0), np.ones(2), np.zeros(2)),  # partial extract -> a resident plume remains
        ]
        # probe: clean injection of 380 m^3 over 20 d. mean|Q| = 100 (A) / 19 (B); phase_volume/time = 19 both.
        probe_a = [(np.array([190.0, 10.0]), np.array([1.0, 19.0]), np.zeros(2))]
        probe_b = [(np.concatenate([[190.0], np.full(19, 10.0)]), np.ones(20), np.zeros(20))]
        return pre + probe_a + read, pre + probe_b + read
    # extraction probe: tracer push, then a clean extraction binned two ways (its residual is propagated
    # inward by the convergent hand-off), then a clean injection separates it from the readout extraction.
    pre = [(np.full(6, 100.0), np.ones(6), np.ones(6))]  # tracer push (cin deviation = 1)
    # probe: clean extraction of 500 m^3 over 4 d. mean|Q| = 100 (A) / 125 (B); phase_volume/time = 125 both.
    probe_a = [(np.array([-150.0, -50.0]), np.array([3.0, 1.0]), np.zeros(2))]
    probe_b = [(np.array([-150.0, -150.0, -150.0, -50.0]), np.ones(4), np.zeros(4))]
    mid = [(np.full(2, 100.0), np.ones(2), np.zeros(2))]  # clean injection separates the two extraction phases
    return pre + probe_a + mid + read, pre + probe_b + mid + read


class TestRebinningInvariance:
    r"""Issue #333: two valid binnings of the *same physical* Q(t) must give identical ``cout``.

    The per-phase advective clock must advance the plume by the flushed volume ``sum|Q| dt`` (the
    dt-weighted mean ``flow_scale = phase_volume/phase_time``), not by the unweighted ``mean(|Q|) * sum(dt)``.
    Any phase where ``|Q|`` and ``dt`` covary makes the two differ, so the two binnings expose the bug
    (issue #333 confirmed baselines: ~0.95 rel for ``D_m > 0``, ~7e-5 for ``D_m = 0``; the per-test baselines
    below vary with the schedule). Every uniform-bin schedule -- i.e. every other radial test -- is a
    bit-exact no-op for the fix, which is why it went uncaught. All comparisons are on physically identical
    extraction bins.
    """

    @pytest.mark.parametrize("direction", ["injection", "extraction"])
    def test_rebinning_invariance_dm0(self, direction):
        # D_m = 0 (Airy S-clock, vectorized -> fast). After the fix flow_scale is binning-invariant AND the
        # kernel is evaluated flow-free, so the two binnings share bit-identical grids/matrices: A == B to the
        # last bit (measured 0.0). Baseline round-trips a different flow_scale per binning (RAR-F2), leaving
        # de-Hoog-amplified bit-noise ~2.6e-8 on this schedule.
        seg_a, seg_b = _rebinning_schedules(direction)
        fa, da, ca = _assemble(seg_a)
        fb, db, cb = _assemble(seg_b)
        cout_a = cout_deviation(cin_deviation=ca, flow=fa, dt_days=da, molecular_diffusivity=0.0, n_quad=64, **GEOM)
        cout_b = cout_deviation(cin_deviation=cb, flow=fb, dt_days=db, molecular_diffusivity=0.0, n_quad=64, **GEOM)
        # atol near machine precision: integer volumes give bit-identical tau/flow_scale, so A and B build the
        # identical matrices; the ~2.6e-8 baseline fails this by ~5 orders.
        np.testing.assert_allclose(cout_a[-6:], cout_b[-6:], rtol=0, atol=1e-13)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("direction", "r_fac"),
        [("injection", 1.0), ("extraction", 1.0), ("injection", 2.0)],  # both hand-off directions + retardation
    )
    def test_rebinning_invariance_dm_positive(self, direction, r_fac):
        # D_m > 0 (Riccati wall-clock) -- THE major regression (RAR-F1). Baseline mis-advects the probe phase
        # by mean(|Q|)*sum(dt) instead of the flushed volume: ~0.95 rel between the two binnings. After the fix
        # flow_scale = phase_volume/phase_time is binning-invariant, so both build the identical cached Riccati
        # propagator and A == B bit-for-bit (measured 0.0). R>1 exercises the a0_eff = flow_scale/(2c_geo)/R
        # coupling; the extraction probe exercises the convergent (Danckwerts) hand-off.
        seg_a, seg_b = _rebinning_schedules(direction)
        kw = {"molecular_diffusivity": 0.05, "retardation_factor": r_fac, "n_quad": 40, **GEOM}
        fa, da, ca = _assemble(seg_a)
        fb, db, cb = _assemble(seg_b)
        cout_a = cout_deviation(cin_deviation=ca, flow=fa, dt_days=da, **kw)
        cout_b = cout_deviation(cin_deviation=cb, flow=fb, dt_days=db, **kw)
        np.testing.assert_allclose(cout_a[-6:], cout_b[-6:], rtol=0, atol=1e-13)

    def test_dm0_flow_magnitude_invariance(self):
        # RAR-F2 in isolation: for D_m = 0 the S-clock kernel is flow-magnitude free, so scaling all |Q| by a
        # constant (with inverse-scaled dt -> fixed volume schedule) must leave cout unchanged, and with a
        # different flow magnitude on each run this is independent of the RAR-F1 (dt-weighting) fix. Multi-cycle
        # so the across-reversal Airy propagator hand-off (which carries the round-trip noise) is invoked -- a
        # single inject/extract cycle never propagates. The scale factor is NON-power-of-two on purpose: a power
        # of two makes the baseline round-trip cancel exactly (s=c*s, a0=c*a0 -> beta=s/a0 bit-identical),
        # hiding the bug. Baseline: ~1e-8 de-Hoog-amplified flow_scale bit-noise; post-fix the kernel is
        # evaluated flow-free so only ~1e-14 volume-ulp noise remains.
        flow = np.array([100.0] * 4 + [-100.0] * 2 + [50.0] * 4 + [-100.0] * 6)  # 3 reversals -> propagator used
        dt = np.ones(len(flow))
        cin = np.where(flow > 0, 1.0, 0.0)  # tracer on injections
        c = 1.7  # NON-power-of-two
        ext = flow < 0
        cout1 = cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=0.0, n_quad=80, **GEOM)
        cout2 = cout_deviation(
            cin_deviation=cin, flow=flow * c, dt_days=dt / c, molecular_diffusivity=0.0, n_quad=80, **GEOM
        )
        np.testing.assert_allclose(cout1[ext], cout2[ext], rtol=0, atol=1e-11)

    @pytest.mark.slow
    def test_dm_engine_reduces_to_airy_limit(self):
        # Physical-correctness ANCHOR that pins the flow_scale VALUE (not just self-consistency, which the
        # rebinning tests already give): as D_m -> 0 the wall-clock Riccati engine must reduce to the
        # volume-exact D_m=0 Airy engine on the non-uniform schedule. This holds ONLY if the Riccati clock
        # advects exactly the flushed volume, i.e. flow_scale = phase_volume/phase_time. Baseline (mean|Q|)
        # mis-advects -> ~0.97 gap; any wrong-but-binning-invariant scale (e.g. max|Q|, = 190 in both binnings)
        # would also fail here while passing the A==B tests. Post-fix the gap is the first-order D_m truncation
        # (~2e-3 at D_m=1e-3).
        flow, dt, cin = _assemble(_rebinning_schedules("injection")[0])
        ext = flow < 0
        ref = cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=0.0, n_quad=48, **GEOM)
        approx = cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=1e-3, n_quad=48, **GEOM)
        scale = np.max(np.abs(ref[ext]))
        assert np.max(np.abs(approx[ext] - ref[ext])) / scale < 8e-3  # first-order D_m limit; baseline ~0.97

    @pytest.mark.slow
    def test_nonuniform_dm_matches_fv_oracle(self):
        # Independent cross-check: the fixed non-uniform D_m>0 engine matches the finite-volume oracle, which
        # shares NO kernel code with the engine and is itself rebinning-invariant. This is the definitive guard
        # against "engine and gridfree oracle silently agree on the wrong value" (both were fixed together).
        # The floor is the constant-|Q| per-phase kernel approximation (~7%, ratio-independent) + slow-FV
        # first-order convergence -- NOT "a few %". Baseline engine-vs-FV = 0.87 (sides with the wrong value);
        # post-fix ~0.08.
        flow, dt, cin = _assemble(_rebinning_schedules("injection")[0])
        ext = flow < 0
        eng = cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=0.05, n_quad=64, **GEOM)
        fv = fv_cout_deviation(
            cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=0.05, n_cells=600, n_sub=8, **GEOM
        )
        scale = np.max(np.abs(fv[ext]))
        assert np.max(np.abs(eng[ext] - fv[ext])) / scale < 0.15  # constant-|Q| kernel (~7%) + slow-FV floor

    @pytest.mark.slow
    def test_echo_operator_rebinning_invariance(self):
        # The single-cycle D_m>0 public path uses the echo operator (radial_asr._echo_operator), whose
        # inj/ext_flow_scale are the milder RAR-F1 site (volume corners exact, only the kernel A_0 shape is
        # off). Non-uniform vs uniform injection of the same tracer -> identical cout after the fix (baseline
        # ~2.6%). Exercises the public infiltration_to_extraction dispatch + tedges->dt conversion.
        inj_a = (np.array([190.0, 10.0]), np.array([1.0, 19.0]), np.array([1.0, 1.0]))
        inj_b = (np.concatenate([[190.0], np.full(19, 10.0)]), np.ones(20), np.ones(20))
        ext = (np.full(6, -100.0), np.full(6, 2.0), np.zeros(6))
        np.testing.assert_allclose(_i2e([inj_a, ext], 48)[-6:], _i2e([inj_b, ext], 48)[-6:], rtol=0, atol=1e-13)

    @pytest.mark.slow
    def test_public_multicycle_rebinning(self):
        # The multi-cycle public path (infiltration_to_extraction -> _reuse_ensemble -> cout_deviation, plus
        # the tedges->dt conversion and streamtube averaging) is rebinning-invariant. Complements the direct
        # cout_deviation tests, which bypass the public wrapper. Baseline ~0.95; post-fix bit-identical.
        seg_a, seg_b = _rebinning_schedules("injection")
        np.testing.assert_allclose(_i2e(seg_a, 40)[-6:], _i2e(seg_b, 40)[-6:], rtol=0, atol=1e-13)
