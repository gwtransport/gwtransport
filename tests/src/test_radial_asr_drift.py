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
from gwtransport.radial_asr import extraction_to_infiltration, infiltration_to_extraction

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
    observable, so the extracted concentration is exactly even in the drift -- a machine-precision anchor
    that the slow-drift FV comparison cannot provide and that genuinely exercises the mode coupling."""
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
        np.testing.assert_allclose(a_fft, a_rs, atol=1e-9)
        np.testing.assert_allclose(b_fft, b_rs, atol=1e-9)
        np.testing.assert_allclose(s0_fft, s0_rs, atol=1e-9)


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
    default regularization). This exercises the drift forward operator's invertibility, not just the forward."""
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
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, pore_heights=_B, porosity=_POROSITY,
        well_radius=_R_W, longitudinal_dispersivity=_ALPHA_L, regional_flux=u, n_modes=3, n_quad=70,
    )
    engine = block_cout_deviation(
        cin_deviation=cin, flow=flow, dt_days=dt, c_geo=np.pi * _B * _POROSITY, r_w=_R_W, alpha_l=_ALPHA_L,
        v_d=u / _POROSITY, n_modes=3, n_quad=70,
    )
    np.testing.assert_allclose(public[ext], engine[ext], atol=1e-12)


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
