"""Tests for the azimuthal-mode block engine: single-well ASR with steady regional drift.

Precision discipline mirrors the radial engine: U=0 reductions and exact symmetries are machine /
de Hoog precision (~1e-8, the matrix-Riccati + de Hoog floor); finite-volume comparisons are
first-order (``O(1/n_cells)``), so engine-vs-FV agreement is judged on the *drift recovery loss* (which
cancels the common FV bias) at ~1-2%, with the finite-volume solve shown to converge toward the engine.

Memory note: the block solutions scale as ``n_quad * n_s * (2 n_modes + 1)^2``; these tests deliberately
keep ``n_quad`` small and ``n_modes`` low so the suite stays light.
"""

import numpy as np
import pandas as pd
import pytest
from _radial_asr_drift_fv_oracle import fv2d_cout_deviation  # ty: ignore[unresolved-import]  # tests/src on path
from _radial_asr_drift_kernels_check import real_space_coupling  # ty: ignore[unresolved-import]

from gwtransport._radial_asr_drift_kernels import block_coupling_matrices, block_cout_deviation
from gwtransport._radial_asr_reuse import cout_deviation
from gwtransport.radial_asr import (
    extraction_to_infiltration,
    gamma_extraction_to_infiltration,
    gamma_infiltration_to_extraction,
    infiltration_to_extraction,
)

# --- a small shared scenario -------------------------------------------------------------------
_POROSITY, _B, _R_W, _ALPHA_L, _Q = 0.3, 10.0, 0.5, 0.5, 100.0
_C_GEO = np.pi * _B * _POROSITY
_A0 = _Q / (2.0 * _C_GEO)


def _single_cycle(n_inj=6, n_ext=10):
    flow = np.array([_Q] * n_inj + [-_Q] * n_ext)
    dt = np.ones(len(flow))
    cin = np.array([1.0] * n_inj + [0.0] * n_ext)
    return flow, dt, cin


def _eps_to_vd(eps, r_b):
    return eps * _A0 / r_b


# --- U=0 reductions: the block engine collapses to the radial engine -----------------------------
def test_block_path_reduces_to_scalar_single_cycle():
    """At v_d=0 the modes decouple; the block engine's m=0 cout == the scalar engine (de Hoog floor).

    This exercises the FULL block machinery (Riccati, transitions, resolvent, readout) -- not the public
    U=0 dispatch -- so it certifies the drift kernels themselves reduce correctly, not just the bypass.
    """
    flow, dt, cin = _single_cycle()
    ext = flow < 0
    scalar = cout_deviation(
        cin_deviation=cin,
        flow=flow,
        dt_days=dt,
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        n_quad=80,
        n_terms=40,
        tol=1e-11,
    )
    block = block_cout_deviation(
        cin_deviation=cin,
        flow=flow,
        dt_days=dt,
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        v_d=0.0,
        n_modes=2,
        n_quad=80,
        n_terms=40,
        tol=1e-11,
    )
    np.testing.assert_allclose(block[ext], scalar[ext], atol=1e-7)


def test_block_path_reduces_to_scalar_multicycle():
    """The resolvent field hand-off across reversals also reduces to the scalar reuse engine at v_d=0."""
    flow = np.array([_Q] * 4 + [-_Q] * 4 + [_Q] * 4 + [-_Q] * 6)
    dt = np.ones(len(flow))
    rng = np.random.default_rng(1)
    cin = np.where(flow > 0, rng.uniform(0.5, 2.0, len(flow)), 0.0)
    ext = flow < 0
    scalar = cout_deviation(
        cin_deviation=cin,
        flow=flow,
        dt_days=dt,
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        n_quad=80,
        n_terms=40,
        tol=1e-11,
    )
    block = block_cout_deviation(
        cin_deviation=cin,
        flow=flow,
        dt_days=dt,
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        v_d=0.0,
        n_modes=2,
        n_quad=80,
        n_terms=40,
        tol=1e-11,
    )
    np.testing.assert_allclose(block[ext], scalar[ext], atol=1e-7)


def test_block_path_reduces_to_scalar_with_diffusion():
    """With molecular diffusion (D_m>0) the v_d=0 block engine still reduces to the scalar reuse engine."""
    flow, dt, cin = _single_cycle()
    ext = flow < 0
    scalar = cout_deviation(
        cin_deviation=cin,
        flow=flow,
        dt_days=dt,
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        molecular_diffusivity=0.5,
        n_quad=80,
        n_terms=40,
        tol=1e-11,
    )
    block = block_cout_deviation(
        cin_deviation=cin,
        flow=flow,
        dt_days=dt,
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        v_d=0.0,
        molecular_diffusivity=0.5,
        n_modes=2,
        n_quad=80,
        n_terms=40,
        tol=1e-11,
    )
    np.testing.assert_allclose(block[ext], scalar[ext], atol=1e-7)


# --- exact U != 0 anchors that exercise the drift coupling ---------------------------------------
def test_drift_reversal_symmetry():
    """cout(+U) == cout(-U): a 180 deg rotation maps the +x drift to -x with a symmetric IC and an m=0
    observable, so the extracted concentration is exactly even in the drift -- a machine-precision structural
    anchor. NB this evenness is covariant under (v_d, theta) -> (-v_d, theta + pi) for any operator built
    from this velocity field, so it pins the theta-grid parity and the m=0 readout, NOT the coupling
    magnitude (the coupling-vs-generator and eps-scaling tests do that)."""
    flow, dt, cin = _single_cycle()
    ext = flow < 0
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    v_d = _eps_to_vd(0.25, r_b)
    kw = {
        "cin_deviation": cin,
        "flow": flow,
        "dt_days": dt,
        "c_geo": _C_GEO,
        "r_w": _R_W,
        "alpha_l": _ALPHA_L,
        "n_modes": 3,
        "n_quad": 80,
        "n_terms": 40,
        "tol": 1e-11,
    }
    cp = block_cout_deviation(v_d=v_d, **kw)
    cm = block_cout_deviation(v_d=-v_d, **kw)
    np.testing.assert_allclose(cp[ext], cm[ext], atol=1e-8)


def test_drift_loss_quadratic_in_eps():
    """Within the slow-drift envelope the recovery loss is analytic O(eps^2): loss(2 eps)/loss(eps) -> 4.

    This pins the coupling MAGNITUDE end-to-end (the reversal test only pins parity), and documents that the
    non-analytic O(|eps|) straggler loss -- which needs the plume to reach the stagnation radius -- is
    exponentially suppressed below r_s, so it does NOT appear here (ratio 4, not the strong-drift ratio 2)."""
    flow, dt, cin = _single_cycle()
    ext = flow < 0
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)

    def loss(eps):
        v_d = _eps_to_vd(eps, r_b)
        re0 = np.nansum(
            block_cout_deviation(
                cin_deviation=cin,
                flow=flow,
                dt_days=dt,
                c_geo=_C_GEO,
                r_w=_R_W,
                alpha_l=_ALPHA_L,
                v_d=1e-9,
                n_modes=4,
                n_quad=90,
                n_terms=40,
                tol=1e-10,
            )[ext]
        )
        re = np.nansum(
            block_cout_deviation(
                cin_deviation=cin,
                flow=flow,
                dt_days=dt,
                c_geo=_C_GEO,
                r_w=_R_W,
                alpha_l=_ALPHA_L,
                v_d=v_d,
                n_modes=4,
                n_quad=90,
                n_terms=40,
                tol=1e-10,
            )[ext]
        )
        return float(re0 - re)

    ratio = loss(0.2) / loss(0.1)
    assert 3.7 < ratio < 4.3, f"loss(2 eps)/loss(eps) = {ratio:.3f}, expected ~4 (quadratic)"


def test_retardation_rescale_in_drift():
    """Retardation is a joint degree-1 rescale of the operator: the block coupling obeys
    ``A,B,S0 -> A,B,S0 / R`` under ``(A_0, v_d, D_m) -> (A_0/R, v_d/R, D_m/R)``, so the ``R s`` Laplace
    kernel ``(A_0, v_d, D_m, R)`` equals the unretarded kernel ``(A_0/R, v_d/R, D_m/R, 1)``. With the same
    bin durations (retardation rescales the clock, not the radius) the extracted concentration matches.
    """
    flow, dt, cin = _single_cycle()
    ext = flow < 0
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    v_d = _eps_to_vd(0.2, r_b)
    r = 2.5
    retarded = block_cout_deviation(
        cin_deviation=cin,
        flow=flow,
        dt_days=dt,
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        v_d=v_d,
        molecular_diffusivity=0.4,
        retardation_factor=r,
        n_modes=3,
        n_quad=100,
        n_terms=40,
        tol=1e-11,
    )
    # A_0 ~ |Q| ~ flow, so flow/R gives A_0/R; v_d/R, D_m/R, R=1; bin durations unchanged.
    plain = block_cout_deviation(
        cin_deviation=cin,
        flow=flow / r,
        dt_days=dt,
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        v_d=v_d / r,
        molecular_diffusivity=0.4 / r,
        retardation_factor=1.0,
        n_modes=3,
        n_quad=100,
        n_terms=40,
        tol=1e-11,
    )
    np.testing.assert_allclose(retarded[ext], plain[ext], atol=1e-6)


def test_coupling_matrices_match_real_space_generator():
    """The FFT-in-theta coupling matrices A, B, S0 equal an independent real-space mode generator.

    Evaluated within the slow-drift envelope (``eps = v_d r / A_0 <= 0.5`` at the outermost radius), where
    the production FFT grid resolves the eps-banded harmonics; both methods then agree to the FFT/quadrature
    floor.
    """
    r = np.array([1.5, 4.0, 7.0])  # eps(r=7) = 0.46 at v_d=0.35, within envelope
    for v_d, d_m in [(0.2, 0.0), (0.35, 0.4)]:
        a_fft, b_fft, s0_fft = block_coupling_matrices(r, alpha_l=_ALPHA_L, a0=_A0, v_d=v_d, d_m=d_m, n_modes=3)
        a_rs, b_rs, s0_rs = real_space_coupling(r, alpha_l=_ALPHA_L, a0=_A0, v_d=v_d, d_m=d_m, n_modes=3)
        # the two methods agree to FFT round-off (~1e-16); the 1e-12 floor guards the production n_theta
        # default against an aliasing regression (8*M+8 would leave a ~1e-2 alias at the envelope edge).
        np.testing.assert_allclose(a_fft, a_rs, atol=1e-12)
        np.testing.assert_allclose(b_fft, b_rs, atol=1e-12)
        np.testing.assert_allclose(s0_fft, s0_rs, atol=1e-12)


# --- honest envelope guards ----------------------------------------------------------------------
def test_strong_drift_raises():
    """Beyond the slow-drift envelope (plume reaches the stagnation radius) the engine raises rather than
    silently truncating the plume and returning a wrong cout."""
    flow, dt, cin = _single_cycle()
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    v_d = _eps_to_vd(0.8, r_b)
    with pytest.raises(ValueError, match="drift too strong"):
        block_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            c_geo=_C_GEO,
            r_w=_R_W,
            alpha_l=_ALPHA_L,
            v_d=v_d,
            n_modes=6,
            n_quad=60,
        )


def test_rest_with_drift_raises():
    """A rest phase under drift is not silently frozen (the plume translates and disperses during rest)."""
    flow = np.array([_Q] * 4 + [0.0] * 3 + [-_Q] * 6)
    dt = np.ones(len(flow))
    cin = np.where(flow > 0, 1.0, 0.0)
    r_b = np.sqrt(_R_W**2 + _Q * 4 / _C_GEO)
    v_d = _eps_to_vd(0.2, r_b)
    with pytest.raises(NotImplementedError, match="rest"):
        block_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            c_geo=_C_GEO,
            r_w=_R_W,
            alpha_l=_ALPHA_L,
            v_d=v_d,
            n_modes=2,
            n_quad=60,
        )


# --- public API dispatch -------------------------------------------------------------------------
def test_public_u0_dispatch_bit_for_bit():
    """regional_flux=0 (default) returns the untouched scalar path bit-for-bit (the U=0 guarantee)."""
    flow, _, cin = _single_cycle()
    ext = flow < 0
    tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
    kw = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pore_heights": _B,
        "porosity": _POROSITY,
        "well_radius": _R_W,
        "longitudinal_dispersivity": _ALPHA_L,
        "n_quad": 80,
    }
    base = infiltration_to_extraction(**kw)
    drift0 = infiltration_to_extraction(regional_flux=0.0, **kw)
    np.testing.assert_array_equal(base[ext], drift0[ext])


def test_public_u0_dispatch_all_entry_points():
    """regional_flux=0.0 is bit-for-bit identical to the default for ALL four entry points (forward,
    reverse, and both gamma wrappers) -- the U=0 guarantee is wired everywhere, not just the forward."""
    flow, _, cin = _single_cycle()
    tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
    geom = {
        "tedges": tedges,
        "cout_tedges": tedges,
        "well_radius": _R_W,
        "longitudinal_dispersivity": _ALPHA_L,
        "porosity": _POROSITY,
        "n_quad": 50,
    }
    direct = {**geom, "flow": flow, "pore_heights": _B}
    np.testing.assert_array_equal(
        infiltration_to_extraction(cin=cin, **direct),
        infiltration_to_extraction(cin=cin, regional_flux=0.0, **direct),
    )
    cout = np.nan_to_num(infiltration_to_extraction(cin=cin, **direct))
    np.testing.assert_array_equal(
        extraction_to_infiltration(cout=cout, **direct),
        extraction_to_infiltration(cout=cout, regional_flux=0.0, **direct),
    )
    gkw = {**geom, "flow": flow, "screen_height": _B, "velocity_cv": 0.3, "n_bins": 4}
    np.testing.assert_array_equal(
        gamma_infiltration_to_extraction(cin=cin, **gkw),
        gamma_infiltration_to_extraction(cin=cin, regional_flux=0.0, **gkw),
    )
    cout_g = np.nan_to_num(gamma_infiltration_to_extraction(cin=cin, **gkw))
    np.testing.assert_array_equal(
        gamma_extraction_to_infiltration(cout=cout_g, **gkw),
        gamma_extraction_to_infiltration(cout=cout_g, regional_flux=0.0, **gkw),
    )


def test_public_drift_background_linearity():
    """Drift transport is linear in the deviation: cout(bg) == bg + cout_dev(bg=0) on extraction bins."""
    flow, _, cin = _single_cycle()
    ext = flow < 0
    tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
    kw = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pore_heights": _B,
        "porosity": _POROSITY,
        "well_radius": _R_W,
        "longitudinal_dispersivity": _ALPHA_L,
        "regional_flux": 0.04,
        "n_modes": 3,
        "n_quad": 70,
    }
    bg = 5.0
    base = infiltration_to_extraction(cin=cin, background=0.0, **kw)
    shifted = infiltration_to_extraction(cin=cin + bg, background=bg, **kw)
    np.testing.assert_allclose(shifted[ext], bg + base[ext], atol=1e-9)


def test_reverse_round_trip_under_drift():
    """The paired reverse (extraction_to_infiltration) under drift inverts the forward operator: feeding it
    the forward drift cout recovers the injected cin on the injection bins (Tikhonov, near-exact at the tiny
    default regularization). This pins the drift forward operator's INVERTIBILITY and the reverse plumbing /
    conditioning -- not the forward physics (the reverse rebuilds the operator from the same engine, so a
    forward physics error cancels in F^-1 F); the FV-loss and coupling tests guard the forward physics."""
    flow, _, cin = _single_cycle()
    inj = flow > 0
    tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
    kw = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pore_heights": _B,
        "porosity": _POROSITY,
        "well_radius": _R_W,
        "longitudinal_dispersivity": _ALPHA_L,
        "regional_flux": 0.05,
        "n_modes": 3,
        "n_quad": 70,
    }
    cout = infiltration_to_extraction(cin=cin, **kw)
    recovered = extraction_to_infiltration(cout=np.where(flow < 0, cout, 0.0), **kw)
    np.testing.assert_allclose(recovered[inj], cin[inj], atol=1e-4)


def test_public_single_streamtube_ensemble_equals_engine():
    """The public drift path with one streamtube is exactly the block engine on that streamtube's c_geo
    (the velocity-CV ensemble is a weighted average that reduces to a single engine call here)."""
    flow, dt, cin = _single_cycle()
    ext = flow < 0
    tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
    u = 0.05
    public = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        pore_heights=_B,
        porosity=_POROSITY,
        well_radius=_R_W,
        longitudinal_dispersivity=_ALPHA_L,
        regional_flux=u,
        n_modes=3,
        n_quad=70,
    )
    engine = block_cout_deviation(
        cin_deviation=cin,
        flow=flow,
        dt_days=dt,
        c_geo=np.pi * _B * _POROSITY,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        v_d=u / _POROSITY,
        n_modes=3,
        n_quad=70,
    )
    np.testing.assert_allclose(public[ext], engine[ext], atol=1e-12)


def test_public_multi_streamtube_ensemble_averaging():
    """The velocity-heterogeneity macrodispersion ensemble under drift is the weight-averaged engine over
    streamtubes: cout = sum_i w_i engine(c_geo_i; shared v_d) / sum_i w_i, with v_d = U/n shared and
    A_0 ~ 1/c_geo per tube. Uses unequal weights summing != 1, which the single-tube test cannot probe."""
    flow, dt, cin = _single_cycle()
    ext = flow < 0
    tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
    u, heights, weights = 0.05, np.array([10.0, 5.0]), np.array([2.0, 1.0])
    public = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        pore_heights=heights,
        weights=weights,
        porosity=_POROSITY,
        well_radius=_R_W,
        longitudinal_dispersivity=_ALPHA_L,
        regional_flux=u,
        n_modes=3,
        n_quad=60,
    )
    per_tube = [
        block_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            c_geo=np.pi * h * _POROSITY,
            r_w=_R_W,
            alpha_l=_ALPHA_L,
            v_d=u / _POROSITY,
            n_modes=3,
            n_quad=60,
        )
        for h in heights
    ]
    manual = (weights[0] * per_tube[0] + weights[1] * per_tube[1]) / weights.sum()
    np.testing.assert_allclose(public[ext], manual[ext], atol=1e-12)


def test_variable_within_phase_flow_is_bounded_mean_flow_approximation():
    """Within-phase variable flow is the mean-flow approximation: the drift engine clocks each phase at its
    mean magnitude (placing the cin bins at time corners, not exact volume edges), unlike the scalar D_m=0
    path which is volume-exact. Measured cleanly at v_d=0 vs the scalar engine:
    constant cin is exact to the de Hoog floor for any flow profile (the resident profile depends only on
    total injected volume); only variable cin AND variable flow expose the misplacement (~7% of peak for a
    +-20% flow ramp with ramped cin) -- nonzero but bounded. Finer phases or constant flow recover exactness.
    """
    n_inj, n_ext = 6, 10
    ramp_flow = 100.0 * np.linspace(0.8, 1.2, n_inj)  # +-20% flow ramp within the injection phase
    flow = np.concatenate([ramp_flow, np.full(n_ext, -100.0)])
    flow_const = np.concatenate([np.full(n_inj, 100.0), np.full(n_ext, -100.0)])
    ramp_cin = np.concatenate([np.linspace(0.2, 1.8, n_inj), np.zeros(n_ext)])  # cin varies within the phase
    const_cin = np.concatenate([np.ones(n_inj), np.zeros(n_ext)])
    dt = np.ones(n_inj + n_ext)
    ext = flow < 0
    kw = {"dt_days": dt, "c_geo": _C_GEO, "r_w": _R_W, "alpha_l": _ALPHA_L, "n_quad": 80, "n_terms": 40, "tol": 1e-11}

    def gap(flow, cin):
        sc = cout_deviation(cin_deviation=cin, flow=flow, **kw)
        bk = block_cout_deviation(cin_deviation=cin, flow=flow, v_d=1e-9, n_modes=2, **kw)
        return float(np.nanmax(np.abs(bk[ext] - sc[ext])))

    assert gap(flow_const, ramp_cin) < 1e-6  # constant flow: exact for any cin (de Hoog floor)
    assert gap(flow, const_cin) < 1e-6  # constant cin: exact for any flow (profile = total volume only)
    assert 1e-3 < gap(flow, ramp_cin) < 0.12  # variable flow AND cin: the mean-flow approximation appears


def test_degenerate_schedules():
    """All-injection -> all-NaN cout; a single extraction phase (no prior field) -> ~0; no crashes."""
    dt = np.ones(4)
    u = 0.05
    common = {
        "dt_days": dt,
        "c_geo": _C_GEO,
        "r_w": _R_W,
        "alpha_l": _ALPHA_L,
        "v_d": u / _POROSITY,
        "n_modes": 2,
        "n_quad": 50,
    }
    all_inj = block_cout_deviation(cin_deviation=np.ones(4), flow=np.full(4, 100.0), **common)
    assert np.all(np.isnan(all_inj))
    only_ext = block_cout_deviation(cin_deviation=np.zeros(4), flow=np.full(4, -100.0), **common)
    np.testing.assert_allclose(only_ext, 0.0, atol=1e-9)


# --- finite-volume oracle: the drift recovery loss --------------------------------------------------
def test_drift_loss_matches_fv_oracle():
    """The engine's drift-induced recovery loss matches the independent 2-D FV oracle. The loss
    (RE(U=0) - RE(U)) cancels the FV's first-order bias, so this is a meaningful ~few-percent check that
    the engine is non-perturbative (a Taylor-in-eps engine would mis-scale the loss)."""
    flow, dt, cin = _single_cycle(6, 10)
    ext = flow < 0
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    eps = 0.25
    v_d = _eps_to_vd(eps, r_b)
    u = v_d * _POROSITY
    n_inj = 6

    def block_re(vd):
        c = block_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            c_geo=_C_GEO,
            r_w=_R_W,
            alpha_l=_ALPHA_L,
            v_d=vd,
            n_modes=3,
            n_quad=100,
            n_terms=44,
            tol=1e-10,
        )
        return float(np.nansum(c[ext]) / n_inj)

    block_loss = block_re(1e-9) - block_re(v_d)

    def fv_re(n_r):
        c = fv2d_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            b=_B,
            porosity=_POROSITY,
            r_w=_R_W,
            alpha_l=_ALPHA_L,
            regional_flux=u,
            n_r=n_r,
            n_theta=48,
            n_sub=8,
        )
        c0 = fv2d_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            b=_B,
            porosity=_POROSITY,
            r_w=_R_W,
            alpha_l=_ALPHA_L,
            regional_flux=0.0,
            n_r=n_r,
            n_theta=48,
            n_sub=8,
        )
        return float(np.sum(c0[ext]) / n_inj) - float(np.sum(c[ext]) / n_inj)

    # Richardson extrapolation of the first-order FV loss
    fv_loss = (fv_re(220) * (1 / 140) - fv_re(140) * (1 / 220)) / (1 / 140 - 1 / 220)
    assert abs(block_loss - fv_loss) < 0.1 * abs(fv_loss) + 2e-3
    assert block_loss > 0.0  # drift always reduces recovery
