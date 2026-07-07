"""Tests for the exact radial advection-dispersion module.

The physics anchors are the closed-form temporal moments (KB Sec. 6) and the extraction arrival
moments (KB Sec. 7), checked to machine precision against an independent mpmath construction of the
transfer functions -- these are derivation-independent of the production code. The de-Hoog-inverted
``cout`` is intrinsically a ~1e-3..1e-7 quantity (the inversion floor), so its conservation /
round-trip tolerances are set accordingly, while quantities that bypass the inversion (the
deviation-formulation constant-input invariant, the ensemble averaging) are exact.
"""

import mpmath as mp
import numpy as np
import pandas as pd
import pytest
from _radial_asr_fv_oracle import fv_cout_deviation  # ty: ignore[unresolved-import]  # tests/src on path via conftest
from scipy.special import erfc

from gwtransport._radial_asr_dehoog import dehoog_inverse
from gwtransport._radial_asr_kernels import transfer_function
from gwtransport._radial_asr_reuse import cout_deviation
from gwtransport._time import dt_to_days
from gwtransport.radial_asr import (
    extraction_to_infiltration,
    gamma_extraction_to_infiltration,
    gamma_infiltration_to_extraction,
    infiltration_to_extraction,
)

mp.mp.dps = 60


def _ff_moments(*, r, r_w, alpha_l, a0, d_m=0.0, retardation_factor=1.0):
    """Closed-form FF temporal moments (KB Sec. 6) -- the analytic anchor for the inverted kernel."""
    a0e, dme = a0 / retardation_factor, d_m / retardation_factor
    d2, d3, d4 = r**2 - r_w**2, r**3 - r_w**3, r**4 - r_w**4
    mech = (2.0 / 3.0) * alpha_l * d3 + alpha_l**2 * d2
    var = (
        mech / a0e**2
        if dme == 0.0
        else mech / ((a0e - dme) * (a0e - 2 * dme)) + dme * d4 / (2 * a0e**2 * (a0e - 2 * dme))
    )
    mu3 = (3 * alpha_l**2 * d4 + 12 * alpha_l**3 * d3 + 18 * alpha_l**4 * d2) / a0e**3
    return {"mean": d2 / (2 * a0e), "variance": var, "mu3": mu3, "skewness": mu3 / var**1.5}


# --- independent mpmath anchors (Airy branch) ---------------------------------------------------
def _ff_mp(s, r, r_w, a_l, a0, *, detect="flux"):
    """FF (or FR if detect='resident') transfer function via mpmath Airy, high precision."""
    s = mp.mpc(s)
    beta = s / (a_l * a0)
    b13 = beta ** (mp.mpf(1) / 3)

    def parts(rr):
        zeta = b13 * rr + beta ** (-mp.mpf(2) / 3) / (4 * a_l * a_l)
        ai, aip = mp.airyai(zeta), mp.airyai(zeta, 1)
        amp = mp.e ** (rr / (2 * a_l))
        return amp * ai, amp * (0.5 * ai - a_l * b13 * aip)  # phi_s, F[phi_s]

    num = parts(r)[1] if detect == "flux" else parts(r)[0]
    return num / parts(r_w)[1]


def _cumulants_mp(ghat, mean, n=9):
    """First three cumulants of a transfer function via a Vandermonde fit of ln ghat near s=0."""
    h = mp.mpf("0.03") / mean
    s = [j * h for j in range(1, n + 1)]
    rhs = mp.matrix([mp.log(ghat(sj)) for sj in s])
    vmat = mp.matrix(n, n)
    for j in range(n):
        for k in range(n):
            vmat[j, k] = s[j] ** (k + 1)
    c = mp.lu_solve(vmat, rhs)
    return float((-c[0]).real), float((2 * c[1]).real), float((-6 * c[2]).real)


# --- de Hoog inversion --------------------------------------------------------------------------
class TestDeHoog:
    @pytest.mark.parametrize(
        ("f_hat", "f_true"),
        [
            (lambda s: 1.0 / (s + 1.0), lambda t: np.exp(-t)),
            (lambda s: 1.0 / s**2, lambda t: t),
            (lambda s: 1.0 / (s * (s + 1.0)), lambda t: 1.0 - np.exp(-t)),
        ],
    )
    def test_known_inverses(self, f_hat, f_true):
        t = np.linspace(0.1, 5.0, 40)
        got = dehoog_inverse(f_hat=f_hat, t=t, n_terms=32, scaling=0.8 * t.max())
        np.testing.assert_allclose(got, f_true(t), atol=1e-5)

    def test_step_like_accurate_and_bounded(self):
        # erfc-front transform exp(-sqrt s)/s -> erfc(1/(2 sqrt t)): Talbot produces negative swings;
        # de Hoog must stay in [0, 1] AND be accurate (the KB rationale for de-Hoog-over-Talbot).
        t = np.linspace(0.1, 5.0, 60)
        got = dehoog_inverse(f_hat=lambda s: np.exp(-np.sqrt(s)) / s, t=t, n_terms=32, scaling=t.max())
        assert np.all(got >= -1e-6)
        assert np.all(got <= 1.0 + 1e-6)
        np.testing.assert_allclose(got, erfc(1.0 / (2.0 * np.sqrt(t))), atol=1e-5)

    def test_batch_axis_equals_scalar_loop(self):
        # The batch generalization (f_hat may return (n_nodes, *batch)) must invert every batch entry in one
        # QD/continued-fraction pass identically to a per-entry scalar inversion -- the core contract of the
        # batched de Hoog the propagator-matrix builders rely on. Pin it both ways: vs the scalar loop
        # (bit-exact) and vs the analytic sum-of-exponentials.
        coeffs = np.array([[1.0, 2.0], [0.5, 1.5], [3.0, 0.25]])  # (3, 2)
        poles = np.array([[0.5, 1.0], [2.0, 0.3], [1.2, 0.8]])
        t = np.array([0.4, 1.1, 2.7])
        batched = dehoog_inverse(f_hat=lambda s: coeffs / (s[:, None, None] + poles), t=t, n_terms=32, scaling=6.0)
        scalar = np.empty((t.size, 3, 2))
        for a in range(3):
            for b in range(2):
                scalar[:, a, b] = dehoog_inverse(
                    f_hat=lambda s, a=a, b=b: coeffs[a, b] / (s + poles[a, b]), t=t, n_terms=32, scaling=6.0
                )
        np.testing.assert_allclose(batched, scalar, rtol=0, atol=0)  # bit-exact vs the per-entry loop
        analytic = coeffs[None] * np.exp(-poles[None] * t[:, None, None])
        np.testing.assert_allclose(batched, analytic, atol=1e-5)

    def test_nonpositive_t_returns_zero(self):
        # de Hoog is defined only for t > 0; the final t>0 mask returns exactly 0 there (the big_t clamp keeps
        # the internal scaling finite) while still inverting the positive entries -- one general path, no
        # early return. Unreachable from the reuse engine (every phase has tau > 0), but the primitive must
        # stay safe.
        t = np.array([-1.0, 0.0, 2.0])
        got = dehoog_inverse(f_hat=lambda s: 1.0 / (s + 1.0), t=t, n_terms=32, scaling=4.0)
        np.testing.assert_array_equal(got[:2], 0.0)
        np.testing.assert_allclose(got[2], np.exp(-2.0), atol=1e-5)

    def test_underflow_columns_invert_to_zero_overflow_surfaces(self):
        # RA2 (guard): a batch column whose transform decays to the double-precision underflow floor (exact
        # or denormal zeros at the high-frequency nodes -- the far-field rest / Airy / Riccati propagator
        # entries) drives the quotient-difference recurrence into 0/0 and x/0. Its physical inverse is ~0, so
        # it must invert to exactly 0, NOT NaN -- a NaN there propagated through P @ field and silently
        # collapsed the whole cout to background. A column whose transform OVERFLOWED (Inf/NaN already present)
        # is a genuine breakdown and must still surface as NaN so a real failure cannot read as a physical
        # zero. (Before the guard only *identically*-zero columns were parked; a partial-underflow column
        # returned NaN.)
        m = 24

        def f_hat(s):
            base = 1.0 / (s + 1.0)  # a well-resolved transform -> exp(-t)
            underflow = base.copy()
            underflow[3:] = 0.0  # decayed to the floor at the high-frequency nodes -> QD 0/0
            overflow = base.copy()
            overflow[5] = np.inf  # a genuinely broken (overflowed) transform
            return np.stack([base, underflow, overflow], axis=1)

        out = dehoog_inverse(f_hat=f_hat, t=np.array([1.0]), n_terms=m)
        np.testing.assert_allclose(out[0, 0], np.exp(-1.0), atol=1e-6)  # normal column unaffected
        assert out[0, 1] == 0.0  # underflowed column -> physical zero (was NaN before the guard)
        assert not np.isfinite(out[0, 2])  # overflowed column -> NaN surfaces (never masked to 0)


# --- Airy kernel + KB Sec. 6 moments ------------------------------------------------------------
class TestAiryKernel:
    @pytest.mark.parametrize(("r", "a_l"), [(5.0, 0.5), (200.0, 0.5)])  # Pe = 10 and Pe = 400
    def test_ff_matches_mpmath_no_overflow(self, r, a_l):
        # The scaled-Airy evaluation must match mpmath even at high Peclet (the raw form overflows).
        r_w, a0 = 0.5, 2.0
        s = np.array([0.05 + 0.1j, 0.3 - 0.2j, 1.7 + 2.3j, 0.011 + 0.0j])
        got = transfer_function(s=s, r=r, r_w=r_w, alpha_l=a_l, a0=a0)
        ref = np.array([complex(_ff_mp(sv, r, r_w, a_l, a0)) for sv in s])
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, ref, rtol=1e-10)

    def test_mass_conservation_ghat0(self):
        g = transfer_function(s=np.array([1e-6 + 0j]), r=5.0, r_w=0.5, alpha_l=0.5, a0=2.0)[0]
        np.testing.assert_allclose(g.real, 1.0, atol=1e-5)

    @pytest.mark.parametrize(("r", "a_l", "a0", "r_w"), [(50.0, 0.5, 2.0, 0.5), (400.0, 1.0, 3.0, 0.2)])
    def test_moments_match_cumulants(self, r, a_l, a0, r_w):
        m = _ff_moments(r=r, r_w=r_w, alpha_l=a_l, a0=a0)
        k1, k2, k3 = _cumulants_mp(lambda s: _ff_mp(s, r, r_w, a_l, a0), mp.mpf(m["mean"]))
        np.testing.assert_allclose(k1, m["mean"], rtol=1e-9)
        np.testing.assert_allclose(k2, m["variance"], rtol=1e-7)
        np.testing.assert_allclose(k3, m["mu3"], rtol=1e-5)

    def test_skewness_asymptote(self):
        # skewness -> (9 sqrt 6 / 4) Pe^{-1/2}; check it converges to the constant as Pe grows.
        consts = [
            _ff_moments(r=r, r_w=0.5, alpha_l=1.0, a0=2.0)["skewness"] * np.sqrt(r / 1.0)
            for r in (200.0, 2000.0, 20000.0)
        ]
        target = 9.0 * np.sqrt(6.0) / 4.0
        assert abs(consts[-1] - target) < abs(consts[0] - target)
        np.testing.assert_allclose(consts[-1], target, rtol=2e-2)

    def test_retardation_scales_moments(self):
        m1 = _ff_moments(r=10.0, r_w=0.5, alpha_l=0.5, a0=2.0, retardation_factor=1.0)
        m3 = _ff_moments(r=10.0, r_w=0.5, alpha_l=0.5, a0=2.0, retardation_factor=3.0)
        np.testing.assert_allclose(m3["mean"], 3.0 * m1["mean"], rtol=1e-14)
        np.testing.assert_allclose(m3["variance"], 9.0 * m1["variance"], rtol=1e-14)


# --- KB Sec. 7 duality (extraction arrival moments) ---------------------------------------------
class TestDuality:
    def test_extraction_arrival_moments(self):
        # The divergent FR transfer function's cumulants equal the Sec. 7 extraction arrival moments
        # divided by |Q|, |Q|^2 -- an independent analytic anchor for the readout kernel.
        rp, r_w, a_l, a0 = 8.0, 0.5, 0.5, 2.0
        mu = ((rp + a_l) ** 2 + a_l**2 - r_w**2) / (2 * a0)
        var = (
            mp.mpf(8) / 3 * a_l * (rp**3 - r_w**3) + a_l**2 * (16 * rp**2 - 4 * r_w**2) + 40 * a_l**3 * rp + 44 * a_l**4
        ) / (4 * a0**2)
        k1, k2, _ = _cumulants_mp(lambda s: _ff_mp(s, rp, r_w, a_l, a0, detect="resident"), mp.mpf(mu))
        np.testing.assert_allclose(k1, float(mu), rtol=1e-11)
        np.testing.assert_allclose(k2, float(var), rtol=1e-8)


# --- Whittaker (D_m > 0) ------------------------------------------------------------------------
class TestWhittakerKernel:
    def test_mass_conservation(self):
        for d_m in (0.4, 2.0):
            g = transfer_function(s=np.array([1e-5 + 0j]), r=10.0, r_w=0.5, alpha_l=0.5, a0=2.0, d_m=d_m)[0]
            np.testing.assert_allclose(g.real, 1.0, atol=1e-3)

    def test_mean_is_v_over_q(self):
        # Whittaker FF mean = V/Q (KB Sec. 6). Fit the cumulants of an INDEPENDENT mpmath Whittaker
        # reference, then confirm the production Riccati (log-derivative) kernel matches that reference to
        # ~machine precision -- so the production mean is V/Q too.
        r, r_w, a_l, a0, d_m = 10.0, 0.5, 0.5, 2.0, 0.2  # A0/Dm = 10 (variance exists, fit well-conditioned)
        v_over_q = (r**2 - r_w**2) / (2 * a0)

        def ghat_mp(s):  # independent mpmath Whittaker flux-flux transfer function
            def flux(rr):
                kappa = mp.sqrt(mp.mpc(s) / d_m)
                astar = a_l * a0 / d_m
                b = 1 - a0 / d_m
                a = b / 2 - kappa * astar / 2
                z = 2 * kappa * (rr + astar)
                u, u1 = mp.hyperu(a, b, z), mp.hyperu(a + 1, b + 1, z)
                expf = mp.e ** (-kappa * (rr + astar))
                return expf * u - (a_l + d_m * rr / a0) * (-kappa * expf * (u + 2 * a * u1))

            return flux(r) / flux(r_w)

        k1, _, _ = _cumulants_mp(ghat_mp, mp.mpf(v_over_q))
        np.testing.assert_allclose(k1, v_over_q, rtol=1e-5)
        g_prod = transfer_function(s=np.array([0.05 + 0.1j]), r=r, r_w=r_w, alpha_l=a_l, a0=a0, d_m=d_m)[0]
        np.testing.assert_allclose(g_prod, complex(ghat_mp(0.05 + 0.1j)), rtol=1e-12)

    def test_converges_to_airy_first_order(self):
        # |g_Whittaker(D_m) - g_Airy| should fall ~ linearly in D_m (ratio -> 4 under D_m/4). a0 is kept
        # modest so A_0/D_m stays small and the mpmath confluent-hypergeometric evals are fast.
        s = np.array([0.05 + 0.1j])
        kw = {"r": 10.0, "r_w": 0.5, "alpha_l": 0.5, "a0": 0.5}
        g_airy = transfer_function(s=s, d_m=0.0, **kw)[0]
        e = [abs(transfer_function(s=s, d_m=d_m, **kw)[0] - g_airy) for d_m in (0.05, 0.0125)]
        assert 3.5 < e[0] / e[1] < 4.5

    @pytest.mark.parametrize("ratio", [1.0, 2.0, 3.0])  # b = 1 - A0/Dm hits 0, -1, -2 (integer degeneracy)
    def test_integer_b_regular(self, ratio):
        a0 = 2.0
        g = transfer_function(s=np.array([0.1 + 0.05j]), r=10.0, r_w=0.5, alpha_l=0.5, a0=a0, d_m=a0 / ratio)[0]
        assert np.isfinite(g)

    def test_retardation_rescale_dm(self):
        # Retardation on the D_m > 0 branch is the exact operator rescale A_0 -> A_0/R, D_m -> D_m/R
        # (alpha_L geometric, unchanged): g_hat(a0, d_m, R) must equal g_hat(a0/R, d_m/R, R=1) to machine
        # precision. Catches a one-sided rescale (e.g. A_0 scaled but not D_m) that the advective-only
        # moment tests miss. Every other retardation test is D_m = 0.
        s = np.array([0.05 + 0.1j, 0.1 + 0.5j, 0.2 + 2.0j])
        kw = {"s": s, "r": 10.0, "r_w": 0.5, "alpha_l": 0.5}
        for r_fac in (2.0, 3.5):
            with_r = transfer_function(a0=2.0, d_m=0.2, retardation_factor=r_fac, **kw)
            rescaled = transfer_function(a0=2.0 / r_fac, d_m=0.2 / r_fac, **kw)
            np.testing.assert_allclose(with_r, rescaled, rtol=0, atol=1e-13)


# --- forward / reverse / ensemble public API ----------------------------------------------------
def _scenario(n_inj=10, n_ext=30, inj_rate=100.0):
    tedges = pd.date_range("2024-01-01", periods=n_inj + n_ext + 1, freq="D")
    flow = np.array([inj_rate] * n_inj + [-inj_rate] * n_ext)
    geom = {"pore_heights": 10.0, "porosity": 0.3, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
    return tedges, flow, geom


class TestForward:
    def test_nan_pattern_and_bounds(self):
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        cout = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=160)
        np.testing.assert_array_equal(np.isnan(cout), flow >= 0.0)
        ext = flow < 0
        assert np.all(cout[ext] >= -1e-6)
        assert np.all(cout[ext] <= 1.0 + 1e-3)

    def test_mass_conservation(self):
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        dt = np.ones(len(flow))
        cout = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=240)
        injected = np.sum(cin[flow > 0] * flow[flow > 0] * dt[flow > 0])
        recovered = np.sum(cout[flow < 0] * (-flow[flow < 0]) * dt[flow < 0])
        np.testing.assert_allclose(recovered, injected, rtol=4e-3)  # de Hoog + quadrature floor

    def test_constant_input_equals_background_exactly(self):
        # Deviation formulation: cin == background is annihilated -> cout == background, bit-exact.
        tedges, flow, geom = _scenario()
        bg = 5.0
        cout = infiltration_to_extraction(
            cin=np.full(len(flow), bg), flow=flow, tedges=tedges, cout_tedges=tedges, **geom, background=bg, n_quad=80
        )
        np.testing.assert_allclose(cout[flow < 0], bg, atol=1e-12)

    def test_pulse_arrival_matches_section7(self):
        # Non-constant injection (single-bin pulse) -- the test a constant input cannot do. The mean
        # arrival volume of the composed cout must match the KB Sec. 7 mu_T at the shell where that
        # bin's water ends up, and an earlier-injected (deeper) pulse must return LATER than a
        # late-injected (near-well) one (LIFO ordering). Catches mass-conserving SHAPE errors in W.
        n_inj, n_ext, rate = 6, 60, 100.0
        tedges = pd.date_range("2024-01-01", periods=n_inj + n_ext + 1, freq="D")
        flow = np.array([rate] * n_inj + [-rate] * n_ext)
        geom = {"pore_heights": 10.0, "porosity": 0.3, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
        c_geo = np.pi * geom["pore_heights"] * geom["porosity"]
        s_inj, a_l, r_w = rate * n_inj, geom["longitudinal_dispersivity"], geom["well_radius"]
        t_centers = (np.arange(n_ext) + 0.5) * rate  # extracted-volume bin centers

        def echo_mean(jbin):
            cin = np.zeros(len(flow))
            cin[jbin] = 1.0
            cout = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=200)
            c = np.clip(cout[flow < 0], 0.0, None)
            return np.sum(c * t_centers) / np.sum(c)

        def mu_t_section7(jbin):  # bin j's water ends at shell V' ~ S_inj - sigma_mid (later inj = nearer well)
            v_prime = s_inj - rate * (jbin + 0.5)
            r_p = np.sqrt(r_w**2 + v_prime / c_geo)
            return c_geo * ((r_p + a_l) ** 2 + a_l**2 - r_w**2)

        mt_deep, mt_near = echo_mean(0), echo_mean(5)
        # LIFO ordering: early-injected (deep) water returns LATER than late-injected (near-well)
        # water. The sigma-vs-(S_inj-sigma) mirror bug reverses this, so this assertion catches the
        # mass-conserving shape-error class a constant input cannot see.
        assert mt_deep > mt_near
        # Absolute Sec. 7 anchor for the deep pulse (narrow shell, mildly-curved mu_T). The near-well
        # shell is wide and mu_T is strongly convex there, so its mean exceeds the centroid mu_T --
        # bracket it between the well arrival and the shell's outer-edge arrival instead.
        np.testing.assert_allclose(mt_deep, mu_t_section7(0), rtol=0.15)
        assert mt_near > c_geo * ((r_w + a_l) ** 2 + a_l**2 - r_w**2)  # exceeds the well-face arrival


class TestEnsemble:
    def test_single_disk_array_equals_scalar(self):
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        kw = {"flow": flow, "tedges": tedges, "cout_tedges": tedges, "n_quad": 160}
        scalar = infiltration_to_extraction(cin=cin, **geom, **kw)
        arr = infiltration_to_extraction(cin=cin, **{**geom, "pore_heights": [10.0]}, **kw)
        np.testing.assert_array_equal(scalar, arr)

    def test_height_spread_increases_macrodispersion(self):
        # Spreading the disk heights lowers the recovery peak (macrodispersion smears the breakthrough).
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        kw = {"cin": cin, "flow": flow, "tedges": tedges, "cout_tedges": tedges, "n_quad": 160}
        sharp = infiltration_to_extraction(**geom, **kw)
        spread = infiltration_to_extraction(**{**geom, "pore_heights": [6.0, 10.0, 14.0]}, **kw)
        assert np.nanmax(spread) < np.nanmax(sharp)


class TestReverse:
    def test_round_trip(self):
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        cout = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=200)
        rec = extraction_to_infiltration(
            cout=cout, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, regularization_strength=1e-10, n_quad=200
        )
        np.testing.assert_array_equal(np.isnan(rec), flow <= 0.0)
        np.testing.assert_allclose(rec[flow > 0], cin[flow > 0], atol=1e-3)


class TestGammaWrapper:
    def test_runs_and_conserves(self):
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        cout = gamma_infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            porosity=geom["porosity"],
            well_radius=geom["well_radius"],
            longitudinal_dispersivity=geom["longitudinal_dispersivity"],
            screen_height=10.0,
            velocity_cv=0.3,
            n_bins=8,
            n_quad=120,
        )
        recovered = np.sum(cout[flow < 0] * (-flow[flow < 0]))
        np.testing.assert_allclose(recovered, np.sum(cin[flow > 0] * flow[flow > 0]), rtol=1e-2)

    def test_velocity_cv_zero_is_homogeneous_screen(self):
        # velocity_cv = 0 is a homogeneous screen -- a single streamtube at the mean velocity (pore height
        # screen_height), matching the docstring. It must run (a degenerate std-0 gamma is not a valid
        # distribution) and reproduce the plain engine at pore_heights = screen_height, bit-for-bit.
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        common = {
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": tedges,
            "porosity": geom["porosity"],
            "well_radius": geom["well_radius"],
            "longitudinal_dispersivity": geom["longitudinal_dispersivity"],
            "n_quad": 120,
        }
        gam = gamma_infiltration_to_extraction(cin=cin, screen_height=10.0, velocity_cv=0.0, n_bins=8, **common)
        homogeneous = infiltration_to_extraction(cin=cin, pore_heights=10.0, **common)
        np.testing.assert_array_equal(gam, homogeneous)

    @pytest.mark.slow
    def test_gamma_multi_cycle_round_trip(self):
        # Both gamma (screen-velocity ensemble) wrappers on a MULTI-CYCLE schedule, so they route through the
        # reuse engine: the forward as a weighted streamtube ensemble, the reverse as the dense ensemble
        # round-trip. (Single-cycle gamma uses the echo operator; only multi-cycle exercises the reuse path.)
        flow = np.array([100.0] * 4 + [-100.0] * 6 + [100.0] * 4 + [-100.0] * 8)
        tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
        cin = np.where(flow > 0, 1.0, 0.0)
        gargs = {
            "porosity": 0.3,
            "well_radius": 0.5,
            "longitudinal_dispersivity": 0.5,
            "screen_height": 10.0,
            "velocity_cv": 0.3,
            "n_bins": 6,
            "n_quad": 60,
        }
        cout = gamma_infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **gargs)
        recovered = np.sum(cout[flow < 0] * (-flow[flow < 0])) / np.sum(cin[flow > 0] * flow[flow > 0])
        assert 0.9 < recovered <= 1.0 + 1e-3
        rec = gamma_extraction_to_infiltration(
            cout=cout, flow=flow, tedges=tedges, cout_tedges=tedges, regularization_strength=1e-6, **gargs
        )
        np.testing.assert_array_equal(np.isnan(rec), flow <= 0.0)
        np.testing.assert_allclose(rec[flow > 0], cin[flow > 0], atol=5e-3)


class TestValidation:
    def test_bad_lengths_raise(self):
        tedges = pd.date_range("2024-01-01", periods=4, freq="D")
        with pytest.raises(ValueError, match="one more element"):
            infiltration_to_extraction(
                cin=np.ones(5),
                flow=np.ones(5),
                tedges=tedges,
                cout_tedges=tedges,
                pore_heights=10.0,
                porosity=0.3,
                well_radius=0.5,
                longitudinal_dispersivity=0.5,
            )

    def test_zero_dispersivity_raises(self):
        tedges, flow, geom = _scenario()
        with pytest.raises(ValueError, match="alpha_L"):
            infiltration_to_extraction(
                cin=np.where(flow > 0, 1.0, 0.0),
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                **{**geom, "longitudinal_dispersivity": 0.0},
            )

    def test_reverse_nan_cout_on_extraction_raises(self):
        # A single NaN measurement on an extraction bin would poison the whole least-squares reverse into
        # an all-NaN cin; the inverse must raise (like the advection / diffusion inverses), not silently
        # return NaN. Structural NaN on injection / rest bins (ignored by the inverse) stays allowed.
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        cout = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=80)
        cout[np.flatnonzero(flow < 0)[3]] = np.nan  # poison one extraction measurement
        with pytest.raises(ValueError, match="cout contains NaN"):
            extraction_to_infiltration(cout=cout, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=80)


class TestGeneralEngine:
    """Multi-cycle schedules route to the reused-propagator-matrix engine; an independent finite-volume
    solve of the exact PDE (KB Sec. 9) is the cross-check oracle."""

    def test_fv_matches_analytic_single_cycle(self):
        # Independent engines: finite-volume discretizes the PDE, the analytic path inverts the Laplace kernel.
        # They must agree to the finite-volume first-order discretization floor (~1%).
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        ana = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=120)
        c_geo = np.pi * geom["pore_heights"] * geom["porosity"]
        fv = fv_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=np.ones(len(flow)),
            c_geo=c_geo,
            r_w=geom["well_radius"],
            alpha_l=geom["longitudinal_dispersivity"],
            n_cells=800,
            n_sub=16,
        )
        np.testing.assert_allclose(fv[flow < 0], ana[flow < 0], atol=2e-2)

    def test_fv_converges_to_analytic(self):
        # First-order convergence of the finite-volume to the exact analytic confirms the finite-volume is correct.
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        ana = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=120)
        c_geo = np.pi * geom["pore_heights"] * geom["porosity"]
        ext = flow < 0

        def err(n_cells, n_sub):
            fv = fv_cout_deviation(
                cin_deviation=cin,
                flow=flow,
                dt_days=np.ones(len(flow)),
                c_geo=c_geo,
                r_w=geom["well_radius"],
                alpha_l=geom["longitudinal_dispersivity"],
                n_cells=n_cells,
                n_sub=n_sub,
            )
            return np.max(np.abs(fv[ext] - ana[ext]))

        assert err(400, 16) < 0.6 * err(100, 4)  # refining the grid reduces the error

    def test_multi_cycle_conserves_and_bounded(self):
        flow = np.array([100.0] * 6 + [-100.0] * 10 + [100.0] * 6 + [-100.0] * 14)
        tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
        cin = np.where(flow > 0, 1.0, 0.0)
        geom = {"pore_heights": 10.0, "porosity": 0.3, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
        cout = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom)
        ext = flow < 0
        np.testing.assert_array_equal(np.isnan(cout), flow >= 0.0)
        assert np.all(cout[ext] >= -1e-3)
        assert np.all(cout[ext] <= 1.0 + 1e-2)
        recovered = np.sum(cout[ext] * (-flow[ext])) / np.sum(cin[flow > 0] * flow[flow > 0])
        assert 0.9 < recovered <= 1.0 + 1e-3  # near-full recovery, up to the buffer + finite-volume floor

    @pytest.mark.slow
    def test_high_peclet_multi_cycle_recovery_monotone(self):
        # RA1: a large-plume multi-cycle push-pull whose grid reaches Peclet r_max/alpha_L > 700, where the
        # reuse engine's Airy interior Green's function overflowed to Inf, the propagator went all-NaN, and the
        # blanket nan_to_num silently mapped it to exactly 0 (recovery collapsed to 0). With the overflow-safe
        # fold the recovery stays finite and PHYSICAL: sharpening the front (decreasing alpha_L) recovers a
        # monotone-non-decreasing fraction of the injected mass -- the collapse produced a non-monotone drop to
        # 0 at alpha_L = 0.22.
        c_geo_heights = 30.0  # pore height; c_geo = pi * b * porosity
        flow = np.array([5000.0] * 60 + [-10000.0] * 15 + [5000.0] * 30 + [-10000.0] * 30)
        cin = np.array([100.0] * 60 + [0.0] * 15 + [50.0] * 30 + [0.0] * 30)
        dt = np.ones(len(flow))
        tedges = pd.date_range("2020-01-01", periods=len(flow) + 1, freq="D")
        geom = {"pore_heights": c_geo_heights, "porosity": 0.3, "well_radius": 0.5}
        injected = np.sum((cin * flow * dt)[flow > 0])
        recovered = []
        for alpha_l in (0.50, 0.30, 0.22, 0.18):
            cout = infiltration_to_extraction(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                longitudinal_dispersivity=alpha_l,
                **geom,
                n_quad=120,
            )
            ext = flow < 0
            assert not np.any(np.isnan(cout[ext]))  # the overflow no longer produces NaN -> silent-zero
            recovered.append(np.sum((cout * (-flow) * dt)[ext]) / injected)
        recovered = np.array(recovered)
        assert np.all(recovered > 0.9)  # no collapse (baseline dropped to ~0 at alpha_L <= 0.22)
        assert np.all(np.diff(recovered) > -1e-9)  # monotone non-decreasing as the front sharpens

    @pytest.mark.slow
    def test_dm_positive_forward_conserves(self):
        # Single inject->extract cycle (no rest) with D_m>0 routes to the closed-form echo operator; this
        # checks that path conserves mass under molecular diffusion. The multi-cycle reuse-engine D_m>0 path
        # is covered by test_dm_positive_multi_cycle_round_trip. n_quad-insensitive (verified 8/16/120).
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        cout = infiltration_to_extraction(
            cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, molecular_diffusivity=0.05, n_quad=16
        )
        recovered = np.sum(cout[flow < 0] * (-flow[flow < 0]))
        np.testing.assert_allclose(recovered, np.sum(cin[flow > 0] * flow[flow > 0]), rtol=2e-2)

    def test_multi_cycle_reverse_round_trip(self):
        flow = np.array([100.0] * 3 + [-100.0] * 7 + [100.0] * 3 + [-100.0] * 9)
        tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
        cin = np.where(flow > 0, 1.0, 0.0)
        geom = {"pore_heights": 10.0, "porosity": 0.3, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
        cout = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, n_quad=140)
        rec = extraction_to_infiltration(
            cout=cout, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, regularization_strength=1e-6, n_quad=140
        )
        np.testing.assert_array_equal(np.isnan(rec), flow <= 0.0)
        np.testing.assert_allclose(rec[flow > 0], cin[flow > 0], atol=5e-3)

    @pytest.mark.slow
    def test_dm_positive_multi_cycle_round_trip(self):
        # D_m>0 forward AND reverse through the reuse engine's Riccati branch on a multi-cycle schedule. The
        # public single-cycle D_m>0 path routes to the echo operator, so only a multi-cycle schedule drives
        # the Riccati propagator matrices (forward) and the dense-column ensemble reverse. @slow: the per-node
        # Riccati ODE dominates; the reverse builds the propagator matrices once and applies them to the whole
        # unit-pulse batch, so the schedule and n_quad are kept minimal (two single-injection cycles -> a
        # 2-column dense reverse).
        flow = np.array([100.0] * 1 + [-100.0] * 2 + [100.0] * 1 + [-100.0] * 2)
        tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
        cin = np.where(flow > 0, 1.0, 0.0)
        geom = {"pore_heights": 10.0, "porosity": 0.3, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
        cout = infiltration_to_extraction(
            cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, molecular_diffusivity=0.05, n_quad=10
        )
        rec = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            **geom,
            molecular_diffusivity=0.05,
            regularization_strength=1e-6,
            n_quad=10,
        )
        np.testing.assert_array_equal(np.isnan(rec), flow <= 0.0)
        np.testing.assert_allclose(rec[flow > 0], cin[flow > 0], atol=2e-3)

    @pytest.mark.slow
    def test_single_cycle_rest_dm_routes_to_reuse_engine(self):
        # A single inject->rest->extract cycle under D_m>0 must NOT use the rest-blind echo operator: the
        # dispatch carve-out routes it to the reuse engine (Bessel rest branch). Pin the routing by matching
        # the public result to a direct single-streamtube engine call -- they agree to machine precision iff
        # the public dispatch routed there (the echo operator would drop the rest-phase diffusion entirely).
        flow = np.array([100.0] * 4 + [0.0] * 3 + [-100.0] * 4)
        tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
        cin = np.where(flow > 0, 1.0, 0.0)
        geom = {"pore_heights": 10.0, "porosity": 0.3, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
        pub = infiltration_to_extraction(
            cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, **geom, molecular_diffusivity=1.0, n_quad=24
        )
        eng = cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt_to_days(tedges),
            c_geo=np.pi * geom["pore_heights"] * geom["porosity"],
            r_w=geom["well_radius"],
            alpha_l=geom["longitudinal_dispersivity"],
            molecular_diffusivity=1.0,
            n_quad=24,
        )
        ext = flow < 0
        np.testing.assert_allclose(pub[ext], eng[ext], rtol=1e-12, atol=1e-12)
