"""Tests for the azimuthal-mode block engine: single-well ASR with steady regional drift.

Precision discipline mirrors the radial engine: U=0 reductions and exact symmetries are machine /
de Hoog precision (~1e-8, the matrix-Riccati + de Hoog floor); finite-volume comparisons are
first-order (``O(1/n_cells)``), so engine-vs-FV agreement is judged on the *drift recovery loss* (which
cancels the common FV bias) at the ~0.3-2% level. The FV-loss tolerances double as physics guards: the
oracle's cross-dispersion distribution sign and the engine's O(eps_w) well-face couplings (flux
modulation, D_rtheta face terms, flux-weighted readout) each move the loss by ~15-20% -- an order of
magnitude beyond the tolerance -- so neither can regress silently.
The rest-with-drift kernel has its own anchors: the t -> 0 identity, drift-reversal evenness through the
rest, the v_d = 0 reduction to the scalar Bessel rest kernel (to the Neumann-image closure residual), an
FV drift-loss cross-check of an inject-rest-extract cycle, and the honest spectral-tail guard.

Cost note: the block solutions scale as ``n_quad * n_s * (2 n_modes + 1)^2`` in memory, and the block
Riccati integration cost grows steeply with ``n_modes`` at the production ODE tolerances. Each test keeps
``n_quad``/``n_modes`` as small as its physics anchor allows, but the suite is compute-bound (~25 min on
four workers) -- the price of exercising the full engine rather than mocks.
"""

import numpy as np
import pandas as pd
import pytest
from _radial_asr_drift_fv_oracle import fv2d_cout_deviation  # ty: ignore[unresolved-import]  # tests/src on path
from _radial_asr_drift_kernels_check import real_space_coupling, real_space_face  # ty: ignore[unresolved-import]

from gwtransport._radial_asr_dehoog import dehoog_inverse
from gwtransport._radial_asr_drift_kernels import _face_matrices, block_coupling_matrices, block_cout_deviation
from gwtransport._radial_asr_reuse import cout_deviation
from gwtransport.radial_asr import (
    _auto_n_modes,
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


def test_retardation_rescale_selects_same_n_modes():
    """_auto_n_modes picks the same M for the (R, flow, v_d) run and its (1, flow/R, v_d/R) rescale.

    Retardation is a degree-1 rescale of the operator ``(A_0, v_d, D_m, R) -> (A_0/R, v_d/R, D_m/R,
    1)`` that leaves the physics -- and therefore the correct azimuthal truncation -- unchanged (see
    :func:`test_retardation_rescale_in_drift`). Sizing the azimuthal radius ``R_b`` and the rest
    displacement ``delta`` from the retarded solute front (the ``/R`` factors, #297) makes the auto
    heuristic invariant, so a retardation-invariance check under drift no longer sees a spurious
    truncation-level ``M`` mismatch. The scenario exercises both bounds: an interior rest phase
    drives the ``m_shift`` (delta) term, and moderate drift drives the ``m_eps`` term.
    """
    n_inj, n_rest, n_ext = 6, 8, 10
    flow = np.array([_Q] * n_inj + [0.0] * n_rest + [-_Q] * n_ext)
    dt = np.ones(len(flow))
    r_b = np.sqrt(_R_W**2 + _Q * n_inj / _C_GEO)
    v_d = _eps_to_vd(0.35, r_b)
    r = 2.5
    m_retarded = _auto_n_modes(flow, dt, _C_GEO, _R_W, _ALPHA_L, v_d, r)
    m_plain = _auto_n_modes(flow / r, dt, _C_GEO, _R_W, _ALPHA_L, v_d / r, 1.0)
    assert m_retarded == m_plain
    # The shared value must be a genuine interior choice, not both pinned to the same [2, 8] rail
    # (which would pass even without the fix and prove nothing).
    assert 2 < m_retarded < 8


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
        # the two methods agree to FFT round-off (~1e-16); the 1e-12 floor guards the production
        # _theta_grid sizing against an aliasing regression (8*M+8 would leave a ~1e-2 alias at the
        # envelope edge).
        np.testing.assert_allclose(a_fft, a_rs, atol=1e-12)
        np.testing.assert_allclose(b_fft, b_rs, atol=1e-12)
        np.testing.assert_allclose(s0_fft, s0_rs, atol=1e-12)


def test_face_matrices_match_real_space_generator():
    """The face operators M[v_r], M[D_rr], M[D_rtheta] at r_w equal an independent real-space generator.

    These matrices carry the flux-modulated Robin/Danckwerts face conditions, the injected-flux source, and
    the flux-weighted readout; the reference builds them by analytic tensor components and rectangle-rule
    Fourier integration, sharing no FFT machinery with production. The signed a0 covers both pumping signs.
    """
    for a0_signed, v_d, d_m in [(_A0, 0.2, 0.0), (-_A0, 0.35, 0.4)]:
        prod = _face_matrices(_R_W, alpha_l=_ALPHA_L, a0_signed=a0_signed, v_d=v_d, d_m=d_m, n_modes=3)
        ref = real_space_face(_R_W, alpha_l=_ALPHA_L, a0=a0_signed, v_d=v_d, d_m=d_m, n_modes=3)
        for m_prod, m_ref in zip(prod, ref, strict=True):
            np.testing.assert_allclose(m_prod, m_ref, atol=1e-12)


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


def test_rest_long_translation_raises():
    """A rest translation that outruns the kept azimuthal modes raises (honest spectral-tail guard) rather
    than silently folding the eccentric plume's harmonics back into the truncated band."""
    flow = np.array([_Q] * 6 + [0.0] * 40 + [-_Q] * 10)
    dt = np.ones(len(flow))
    cin = np.where(flow > 0, 1.0, 0.0)
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    v_d = _eps_to_vd(0.2, r_b)
    with pytest.raises(ValueError, match="n_modes"):
        block_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            c_geo=_C_GEO,
            r_w=_R_W,
            alpha_l=_ALPHA_L,
            v_d=v_d,
            n_modes=3,
            n_quad=60,
        )


# --- rest phases under drift: the free-space translate + spread kernel ----------------------------
def test_rest_tiny_duration_reduces_to_no_rest():
    """A vanishing rest phase is a no-op: the translate+spread kernel reduces to the identity (t -> 0),
    exercising the full real-space evaluate / Gauss-Hermite / FFT-reproject round trip at delta ~ 0."""
    kw = {"c_geo": _C_GEO, "r_w": _R_W, "alpha_l": _ALPHA_L, "n_modes": 3, "n_quad": 80, "n_terms": 40, "tol": 1e-10}
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    v_d = _eps_to_vd(0.2, r_b)
    flow_r = np.array([_Q] * 6 + [0.0] + [-_Q] * 10)
    dt_r = np.ones(len(flow_r))
    dt_r[6] = 1e-9
    cin_r = np.where(flow_r > 0, 1.0, 0.0)
    flow_n = np.array([_Q] * 6 + [-_Q] * 10)
    c_rest = block_cout_deviation(cin_deviation=cin_r, flow=flow_r, dt_days=dt_r, v_d=v_d, **kw)
    c_none = block_cout_deviation(
        cin_deviation=np.where(flow_n > 0, 1.0, 0.0), flow=flow_n, dt_days=np.ones(len(flow_n)), v_d=v_d, **kw
    )
    np.testing.assert_allclose(c_rest[flow_r < 0], c_none[flow_n < 0], atol=1e-8)


def test_rest_drift_reversal_symmetry():
    """cout(+U) == cout(-U) with a rest phase: the rest kernel's translation direction covaries with the
    drift sign under the 180-degree rotation, so the evenness anchor extends through the rest kernel."""
    flow = np.array([_Q] * 6 + [0.0] * 3 + [-_Q] * 10)
    dt = np.ones(len(flow))
    cin = np.where(flow > 0, 1.0, 0.0)
    ext = flow < 0
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    v_d = _eps_to_vd(0.2, r_b)
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
        "tol": 1e-10,
    }
    cp = block_cout_deviation(v_d=v_d, **kw)
    cm = block_cout_deviation(v_d=-v_d, **kw)
    np.testing.assert_allclose(cp[ext], cm[ext], atol=1e-8)
    # the rest translation strictly degrades recovery relative to the no-rest schedule (drift keeps acting)
    flow_n = np.array([_Q] * 6 + [-_Q] * 10)
    c_none = block_cout_deviation(
        cin_deviation=np.where(flow_n > 0, 1.0, 0.0),
        flow=flow_n,
        dt_days=np.ones(len(flow_n)),
        c_geo=_C_GEO,
        r_w=_R_W,
        alpha_l=_ALPHA_L,
        v_d=v_d,
        n_modes=3,
        n_quad=80,
        n_terms=40,
        tol=1e-10,
    )
    assert np.nansum(cp[ext]) < np.nansum(c_none[flow_n < 0])


def test_rest_dm_reduces_to_scalar_bessel():
    """At v_d=0 with D_m>0 the rest kernel (isotropic free-space Gaussian + Neumann image at the well)
    matches the scalar engine's exact well-respecting Bessel rest kernel to the image-closure residual
    (~4e-4, the circle-vs-line curvature of the image at the r_w scale) -- an independent exact anchor for
    the Gaussian-spread half of the kernel."""
    flow = np.array([_Q] * 6 + [0.0] * 3 + [-_Q] * 10)
    dt = np.ones(len(flow))
    cin = np.where(flow > 0, 1.0, 0.0)
    ext = flow < 0
    kw = {"c_geo": _C_GEO, "r_w": _R_W, "alpha_l": _ALPHA_L, "n_quad": 90, "n_terms": 40, "tol": 1e-10}
    blk = block_cout_deviation(
        cin_deviation=cin, flow=flow, dt_days=dt, v_d=0.0, molecular_diffusivity=0.5, n_modes=3, **kw
    )
    sca = cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, molecular_diffusivity=0.5, **kw)
    np.testing.assert_allclose(blk[ext], sca[ext], atol=1e-3)


def test_rest_drift_loss_matches_fv_oracle():
    """The rest-phase drift loss matches the independent 2-D FV oracle: an inject-rest-extract cycle at
    eps=0.25 with a 4-day rest, judged on the drift recovery loss (RE(U~0) - RE(U)), which the rest phase
    roughly doubles relative to the no-rest schedule -- the seasonal-storage effect the kernel exists for.
    The agreement is ~2% of the loss (the residual is the Neumann-image closure of the free-space rest
    kernel plus the FV floor)."""
    n_inj, n_rest, n_ext = 6, 4, 10
    flow = np.concatenate([np.full(n_inj, _Q), np.zeros(n_rest), np.full(n_ext, -_Q)])
    dt = np.ones(len(flow))
    cin = np.where(flow > 0, 1.0, 0.0)
    ext = flow < 0
    r_b = np.sqrt(_R_W**2 + _Q * n_inj / _C_GEO)
    v_d = _eps_to_vd(0.25, r_b)
    u = v_d * _POROSITY

    def block_re(vd):
        c = block_cout_deviation(
            cin_deviation=cin,
            flow=flow,
            dt_days=dt,
            c_geo=_C_GEO,
            r_w=_R_W,
            alpha_l=_ALPHA_L,
            v_d=vd,
            n_modes=4,
            n_quad=90,
            n_terms=40,
            tol=1e-10,
        )
        return float(np.nansum(c[ext]) / n_inj)

    block_loss = block_re(1e-9) - block_re(v_d)

    def fv_loss(n_r):
        re = {}
        for u_flux in (0.0, u):
            c = fv2d_cout_deviation(
                cin_deviation=cin,
                flow=flow,
                dt_days=dt,
                b=_B,
                porosity=_POROSITY,
                r_w=_R_W,
                alpha_l=_ALPHA_L,
                regional_flux=u_flux,
                n_r=n_r,
                n_theta=48,
                n_sub=8,
            )
            re[u_flux] = float(np.sum(c[ext]) / n_inj)
        return re[0.0] - re[u]

    fv = (fv_loss(220) * (1 / 140) - fv_loss(140) * (1 / 220)) / (1 / 140 - 1 / 220)  # Richardson
    assert abs(block_loss - fv) < 0.05 * abs(fv) + 1e-3
    assert block_loss > 0.0


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


def test_auto_n_modes_sizing():
    """_auto_n_modes: the eps^{M+1} tail bound (eps = 0.3 -> M = 5), the [2, .] floor at vanishing drift,
    the rest-displacement growth (a rest phase raises M via the displaced-radius eps and the delta/width
    harmonic bound), the [., 8] cap for long rests, and the all-rest floor (nothing pumps -> 2, letting
    the engine's graceful all-NaN branch run instead of an opaque empty-reduction crash)."""
    flow, dt, _ = _single_cycle()
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    assert _auto_n_modes(flow, dt, _C_GEO, _R_W, _ALPHA_L, 0.3 * _A0 / r_b, 1.0) == 5  # ceil(ln 5e-3 / ln 0.3)
    assert _auto_n_modes(flow, dt, _C_GEO, _R_W, _ALPHA_L, 1e-9 * _A0 / r_b, 1.0) == 2  # floor
    v_d = _eps_to_vd(0.2, r_b)
    flow_rest = np.concatenate([flow[:6], np.zeros(30), flow[6:]])
    m_rest = _auto_n_modes(flow_rest, np.ones(len(flow_rest)), _C_GEO, _R_W, _ALPHA_L, v_d, 1.0)
    assert m_rest > _auto_n_modes(flow, dt, _C_GEO, _R_W, _ALPHA_L, v_d, 1.0)
    flow_long = np.concatenate([flow[:6], np.zeros(300), flow[6:]])
    assert _auto_n_modes(flow_long, np.ones(len(flow_long)), _C_GEO, _R_W, _ALPHA_L, v_d, 1.0) == 8  # cap
    assert _auto_n_modes(np.zeros(5), np.ones(5), _C_GEO, _R_W, _ALPHA_L, v_d, 1.0) == 2  # all-rest


def test_public_auto_n_modes_dispatch():
    """n_modes=None (the default) auto-sizes the truncation: the public result is bit-for-bit the explicit
    call at the _auto_n_modes value -- the default drift path is exercised end to end."""
    flow, dt, cin = _single_cycle()
    tedges = pd.date_range("2024-01-01", periods=len(flow) + 1, freq="D")
    u = 0.05
    m = _auto_n_modes(flow, dt, _C_GEO, _R_W, _ALPHA_L, u / _POROSITY, 1.0)
    kw = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pore_heights": _B,
        "porosity": _POROSITY,
        "well_radius": _R_W,
        "longitudinal_dispersivity": _ALPHA_L,
        "regional_flux": u,
        "n_quad": 60,
    }
    np.testing.assert_array_equal(
        infiltration_to_extraction(**kw),
        infiltration_to_extraction(n_modes=m, **kw),
    )


def test_public_drift_validation_errors():
    """The drift validations reject a non-finite regional_flux and a non-positive n_modes before any
    engine work starts."""
    flow, _dt, cin = _single_cycle()
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
    }
    with pytest.raises(ValueError, match="regional_flux must be finite"):
        infiltration_to_extraction(regional_flux=np.inf, **kw)
    with pytest.raises(ValueError, match="n_modes must be >= 1"):
        infiltration_to_extraction(regional_flux=0.05, n_modes=0, **kw)


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
    (the velocity-CV ensemble is a weighted average that reduces to a single engine call here). The
    schedule includes a rest phase, so the public dispatch of rest-with-drift is exercised too."""
    flow = np.array([_Q] * 6 + [0.0] * 2 + [-_Q] * 10)
    dt = np.ones(len(flow))
    cin = np.where(flow > 0, 1.0, 0.0)
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

    NB the constant-cin exactness is a v_d = 0 statement only: under drift the mode coupling integrates
    eps(r(t)) on the wall clock, so equal-volume equal-duration flow profiles end in different fields (an
    FV differencing of a constant-cin +-60% ramp against constant flow shifts the drift recovery loss by
    ~12% of the loss) while the mean-flow engine returns identical results for both. Under drift the
    mean-flow approximation therefore applies to ANY within-phase variation, constant cin included.
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


def test_nonuniform_phase_schedule():
    """Schedules with unequal per-phase durations AND magnitudes work and reduce to the scalar engine at
    v_d ~ 0. This shape (small A_0, short phases) pushes the mode-split Sturm-Liouville exponent past
    double precision for any globally pivoted fundamental-matrix resolvent, so it pins that the
    per-interval transition recursions stay conditioned."""
    flow = np.concatenate([np.full(4, 80.0), np.full(5, -60.0), np.full(3, 120.0), np.full(6, -90.0)])
    dt = np.ones(len(flow))
    cin = np.where(flow > 0, 1.0, 0.0)
    ext = flow < 0
    kw = {"c_geo": _C_GEO, "r_w": _R_W, "alpha_l": _ALPHA_L, "n_quad": 80, "n_terms": 40, "tol": 1e-11}
    scalar = cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, **kw)
    block0 = block_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, v_d=1e-9, n_modes=2, **kw)
    # 1e-6, not the uniform-schedule 1e-7: the mixed durations/magnitudes give the two engines different
    # de Hoog scalings per phase, so the comparison floor is the Pade-acceleration nonlinearity (~2e-7).
    np.testing.assert_allclose(block0[ext], scalar[ext], atol=1e-6)
    r_b = np.sqrt(_R_W**2 + 80.0 * 4 / _C_GEO)
    v_d = 0.15 * (60.0 / (2.0 * _C_GEO)) / r_b  # eps relative to the WEAKEST phase (the binding envelope)
    drift = block_cout_deviation(cin_deviation=cin, flow=flow, dt_days=dt, v_d=v_d, n_modes=3, **kw)
    assert np.isfinite(drift[ext]).all()
    assert np.all(drift[ext] > -1e-9)
    assert np.nansum(drift[ext]) < np.nansum(block0[ext])  # drift strictly reduces recovery


def test_batched_columns_match_vector_runs():
    """A (n, k) cin batch equals the k separate vector runs to the de Hoog floor (the per-phase kernels are
    bit-shared; only BLAS blocking differs), a zero column returns exactly zero (no cross-column leakage),
    and a power-of-two scaled column is exactly scaled (the whole pipeline is linear per column)."""
    flow, dt, cin = _single_cycle()
    ext = flow < 0
    r_b = np.sqrt(_R_W**2 + _Q * 6 / _C_GEO)
    kw = {
        "flow": flow,
        "dt_days": dt,
        "c_geo": _C_GEO,
        "r_w": _R_W,
        "alpha_l": _ALPHA_L,
        "v_d": _eps_to_vd(0.2, r_b),
        "n_modes": 2,
        "n_quad": 60,
        "n_terms": 40,
        "tol": 1e-10,
    }
    other = np.roll(cin, 1) * 0.7
    batched = block_cout_deviation(cin_deviation=np.stack([cin, other, np.zeros_like(cin), 2.0 * cin], axis=1), **kw)
    np.testing.assert_allclose(batched[ext, 0], block_cout_deviation(cin_deviation=cin, **kw)[ext], atol=1e-9)
    np.testing.assert_allclose(batched[ext, 1], block_cout_deviation(cin_deviation=other, **kw)[ext], atol=1e-9)
    np.testing.assert_array_equal(batched[ext, 2], 0.0)  # exact: no cross-column contamination
    np.testing.assert_array_equal(batched[ext, 3], 2.0 * batched[ext, 0])  # exact: linearity per column
    # the rest kernel carries the batch axis too (the reverse build feeds pulse columns through rests)
    flow_r = np.array([_Q] * 6 + [0.0] * 2 + [-_Q] * 10)
    cin_r = np.where(flow_r > 0, 1.0, 0.0)
    kw_r = {**kw, "flow": flow_r, "dt_days": np.ones(len(flow_r))}
    ext_r = flow_r < 0
    batched_r = block_cout_deviation(cin_deviation=np.stack([cin_r, np.zeros_like(cin_r)], axis=1), **kw_r)
    np.testing.assert_allclose(batched_r[ext_r, 0], block_cout_deviation(cin_deviation=cin_r, **kw_r)[ext_r], atol=1e-9)
    np.testing.assert_array_equal(batched_r[ext_r, 1], 0.0)


def test_solutions_cache_eviction_is_pure(monkeypatch):
    """The per-phase solutions cache is a pure memo: capping it at one entry (forcing FIFO eviction and
    recomputation on every phase-signature revisit) leaves cout bit-identical to the default cap."""
    flow = np.array([_Q] * 2 + [-_Q] * 2 + [_Q] * 2 + [-_Q] * 3)
    dt = np.ones(len(flow))
    cin = np.where(flow > 0, 1.0, 0.0)
    r_b = np.sqrt(_R_W**2 + _Q * 2 / _C_GEO)
    kw = {
        "cin_deviation": cin,
        "flow": flow,
        "dt_days": dt,
        "c_geo": _C_GEO,
        "r_w": _R_W,
        "alpha_l": _ALPHA_L,
        "v_d": _eps_to_vd(0.15, r_b),
        "n_modes": 2,
        "n_quad": 40,
        "n_terms": 24,
        "tol": 1e-8,
    }
    base = block_cout_deviation(**kw)
    monkeypatch.setattr("gwtransport._radial_asr_drift_kernels._SOLUTIONS_CACHE_MAX", 1)
    capped = block_cout_deviation(**kw)
    np.testing.assert_array_equal(base, capped)  # NaN injection bins compare equal under array_equal


def test_dehoog_zero_and_mixed_batch_columns():
    """An identically-zero transform inverts to exactly zero without poisoning its batch neighbours: the
    quotient-difference recurrence would form 0/0 on the zero column (a decoupled azimuthal mode with no
    source), so dehoog_inverse parks such columns and restores exact zeros -- while the nonzero columns
    still invert to their analytic values."""
    t = np.array([0.5, 1.0, 2.0])

    def f_hat(s):
        return np.stack([1.0 / (s + 1.0), np.zeros_like(s), 1.0 / s], axis=-1)  # e^{-t}, 0, 1

    out = dehoog_inverse(f_hat=f_hat, t=t, n_terms=24, tol=1e-10)
    np.testing.assert_allclose(out[:, 0], np.exp(-t), atol=1e-8)
    np.testing.assert_array_equal(out[:, 1], 0.0)
    np.testing.assert_allclose(out[:, 2], 1.0, atol=1e-8)
    # The no-batch (1-D transform) case: zero_cols is then 0-d and indexes as a broadcast mask, so an
    # identically-zero scalar transform parks and inverts to exact 0 with the input's shape preserved.
    np.testing.assert_array_equal(dehoog_inverse(f_hat=np.zeros_like, t=t, n_terms=24, tol=1e-10), 0.0)
    scalar_out = dehoog_inverse(f_hat=np.zeros_like, t=0.7, n_terms=24, tol=1e-10)
    assert np.shape(scalar_out) == ()
    np.testing.assert_array_equal(scalar_out, 0.0)


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
    (RE(U=0) - RE(U)) cancels the FV's first-order bias, so this is a meaningful check that the engine is
    non-perturbative (a Taylor-in-eps engine would mis-scale the loss). The agreement is ~0.3% of the
    loss; the 5% + 1e-3 tolerance doubles as a physics guard, since the oracle's cross-dispersion sign
    and the engine's well-face couplings each move the loss by ~18%."""
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
    assert abs(block_loss - fv_loss) < 0.05 * abs(fv_loss) + 1e-3
    assert block_loss > 0.0  # drift always reduces recovery
