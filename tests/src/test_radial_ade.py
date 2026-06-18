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
from scipy.special import erfc

from gwtransport._radial_dehoog import dehoog_inverse
from gwtransport._radial_kernels import _whittaker_phi_and_flux, transfer_function
from gwtransport.radial_ade import (
    extraction_to_infiltration,
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
        r, r_w, a_l, a0, d_m = 10.0, 0.5, 0.5, 2.0, 0.2  # A0/Dm = 10 (variance exists, fit well-conditioned)
        v_over_q = (r**2 - r_w**2) / (2 * a0)

        def ghat(s):
            a0e, dme = a0, d_m
            return (
                _whittaker_phi_and_flux(s, r, a_l, a0e, dme, +1)[1]
                / _whittaker_phi_and_flux(s, r_w, a_l, a0e, dme, +1)[1]
            )

        k1, _, _ = _cumulants_mp(ghat, mp.mpf(v_over_q))
        np.testing.assert_allclose(k1, v_over_q, rtol=1e-5)

    def test_converges_to_airy_first_order(self):
        # |g_Whittaker(D_m) - g_Airy| should fall ~ linearly in D_m (ratio -> 4 under D_m/4).
        s = np.array([0.05 + 0.1j])
        kw = {"r": 10.0, "r_w": 0.5, "alpha_l": 0.5, "a0": 2.0}
        g_airy = transfer_function(s=s, d_m=0.0, **kw)[0]
        e = [abs(transfer_function(s=s, d_m=d_m, **kw)[0] - g_airy) for d_m in (0.05, 0.0125)]
        assert 3.5 < e[0] / e[1] < 4.5

    @pytest.mark.parametrize("ratio", [1.0, 2.0, 3.0])  # b = 1 - A0/Dm hits 0, -1, -2 (integer degeneracy)
    def test_integer_b_regular(self, ratio):
        a0 = 2.0
        g = transfer_function(s=np.array([0.1 + 0.05j]), r=10.0, r_w=0.5, alpha_l=0.5, a0=a0, d_m=a0 / ratio)[0]
        assert np.isfinite(g)


# --- forward / reverse / ensemble public API ----------------------------------------------------
def _scenario(n_inj=10, n_ext=30, inj_rate=100.0):
    tedges = pd.date_range("2024-01-01", periods=n_inj + n_ext + 1, freq="D")
    flow = np.array([inj_rate] * n_inj + [-inj_rate] * n_ext)
    geom = {"pore_height": 10.0, "porosity": 0.3, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
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
        geom = {"pore_height": 10.0, "porosity": 0.3, "well_radius": 0.5, "longitudinal_dispersivity": 0.5}
        c_geo = np.pi * geom["pore_height"] * geom["porosity"]
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
        arr = infiltration_to_extraction(cin=cin, **{**geom, "pore_height": [10.0]}, **kw)
        np.testing.assert_array_equal(scalar, arr)

    def test_height_spread_increases_macrodispersion(self):
        # Spreading the disk heights lowers the recovery peak (macrodispersion smears the breakthrough).
        tedges, flow, geom = _scenario()
        cin = np.where(flow > 0, 1.0, 0.0)
        kw = {"cin": cin, "flow": flow, "tedges": tedges, "cout_tedges": tedges, "n_quad": 160}
        sharp = infiltration_to_extraction(**geom, **kw)
        spread = infiltration_to_extraction(**{**geom, "pore_height": [6.0, 10.0, 14.0]}, **kw)
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
            mean=10.0,
            std=3.0,
            n_bins=8,
            n_quad=120,
        )
        recovered = np.sum(cout[flow < 0] * (-flow[flow < 0]))
        np.testing.assert_allclose(recovered, np.sum(cin[flow > 0] * flow[flow > 0]), rtol=1e-2)


class TestValidation:
    def test_multi_cycle_raises(self):
        tedges = pd.date_range("2024-01-01", periods=5, freq="D")
        flow = np.array([100.0, -100.0, 100.0, -100.0])  # two reversals
        with pytest.raises(NotImplementedError, match="multi-cycle"):
            infiltration_to_extraction(
                cin=np.ones(4),
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                pore_height=10.0,
                porosity=0.3,
                well_radius=0.5,
                longitudinal_dispersivity=0.5,
            )

    def test_bad_lengths_raise(self):
        tedges = pd.date_range("2024-01-01", periods=4, freq="D")
        with pytest.raises(ValueError, match="one more element"):
            infiltration_to_extraction(
                cin=np.ones(5),
                flow=np.ones(5),
                tedges=tedges,
                cout_tedges=tedges,
                pore_height=10.0,
                porosity=0.3,
                well_radius=0.5,
                longitudinal_dispersivity=0.5,
            )

    def test_start_with_extraction_raises(self):
        tedges = pd.date_range("2024-01-01", periods=5, freq="D")
        flow = np.array([-100.0, -100.0, 100.0, 100.0])  # extraction before injection
        with pytest.raises(NotImplementedError, match="extraction"):
            infiltration_to_extraction(
                cin=np.ones(4),
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                pore_height=10.0,
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

    def test_molecular_diffusivity_positive_raises(self):
        # The composed D_m>0 forward is deferred (the Whittaker kernel itself is validated separately).
        tedges, flow, geom = _scenario()
        with pytest.raises(ValueError, match="molecular_diffusivity must be 0"):
            infiltration_to_extraction(
                cin=np.where(flow > 0, 1.0, 0.0),
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                **geom,
                molecular_diffusivity=0.05,
            )
