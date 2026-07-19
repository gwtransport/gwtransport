"""Tests for gwtransport.diffusion_fast_fast.

diffusion_fast_fast is a fast *approximate* forward transport: advection + macrodispersion +
microdispersion AND molecular diffusion (folded in as an effective dispersivity
alpha_eff = alpha_L + D_m*r_vpv/(L*q_mean) per streamtube) form a single skewed Kreft-Zuber
breakthrough on the native cumulative-volume grid. Its reference is gwtransport.diffusion_fast
(machine-precision exact), which is itself anchored to the gwtransport.diffusion Gauss-Legendre
quadrature. At constant flow the method reproduces diffusion_fast to the Ibar-interpolation floor
(~1e-4) for every regime including heat; under variable flow the molecular part carries a
frozen-q_mean commutator residual (small when molecular diffusion is sub-dominant).

Tolerances here are deliberately approximate and pinned per regime to the measured worst case
(comments record the measurement) -- never blanket-loose. The reverse path deconvolves the SAME
approximate operator the forward applies (the banded breakthrough operator W solved with banded
Tikhonov), so a round trip is self-consistent: machine-precision at constant flow R=1 and
conditioning-limited otherwise.
"""

import time
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import gwtransport.diffusion_fast_fast as fff_module
from gwtransport import gamma
from gwtransport.advection import infiltration_to_extraction as advection_i2e
from gwtransport.diffusion_fast import extraction_to_infiltration as df_e2i
from gwtransport.diffusion_fast import gamma_infiltration_to_extraction as df_gamma_i2e
from gwtransport.diffusion_fast import infiltration_to_extraction as df_i2e
from gwtransport.diffusion_fast_fast import (
    extraction_to_infiltration,
    gamma_extraction_to_infiltration,
    gamma_infiltration_to_extraction,
    infiltration_to_extraction,
)
from gwtransport.gamma import mean_std_loc_to_alpha_beta
from gwtransport.utils import solve_inverse_transport

SL = 80.0


def _make_transport_data(*, n_days=200, flow_rate=100.0):
    """Aligned daily grid with constant flow."""
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    return tedges, tedges.copy(), np.full(n_days, flow_rate)


def _apv(n_bins):
    return np.array([500.0]) if n_bins == 1 else gamma.bins(mean=500.0, std=120.0, n_bins=n_bins)["expected_values"]


def _build_w_via_probes(call_fn, n):
    """Reconstruct the (n_out, n) coefficient matrix by feeding canonical basis vectors.

    Independent of the module internals (probes the public API), so column-mass checks against it
    are not tautological.
    """
    cout0 = call_fn(np.zeros(n))
    w = np.zeros((len(cout0), n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        cout = call_fn(ej)
        w[:, j] = np.where(np.isnan(cout), 0.0, cout)
    return w


def _peak_rel(actual, ref, *, skip=8):
    """Peak-relative error on the jointly-finite interior (excludes the boundary smear region)."""
    mask = ~np.isnan(actual) & ~np.isnan(ref)
    mask[:skip] = False
    mask[-skip:] = False
    assert mask.sum() > 10
    return np.nanmax(np.abs(actual[mask] - ref[mask])) / np.nanmax(np.abs(ref[mask]))


def _pulse(n, start=None):
    c = np.zeros(n)
    s = n // 4 if start is None else start
    c[s : s + 4] = 1.0
    return c


def _step(n):
    c = np.full(n, 2.0)
    c[n // 3 :] = 8.0
    return c


def _smooth(n, freq=8.0):
    return np.sin(np.linspace(0.0, freq * np.pi, n)) * 3.0 + 6.0


def _variable_flow(n, cv, seed):
    return 100.0 * np.exp(np.random.default_rng(seed).normal(0.0, cv, n))


def _timed(fn):
    """Wall-clock seconds for a single call to ``fn``."""
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


# =============================================================================
# Step 1 in isolation: advection + microdispersion (D_m = 0) vs diffusion_fast(D_m = 0).
# This is the dominant, skew-carrying part and the delicate banded-index code; validate it first.
# =============================================================================


@pytest.mark.parametrize("spinup", ["constant", None])
@pytest.mark.parametrize("retardation", [1.0, 2.7])
@pytest.mark.parametrize("flow_kind", ["const", "var"])
@pytest.mark.parametrize("inp", ["pulse", "step"])
def test_step1_advection_micro_parity(inp, flow_kind, retardation, spinup):
    """D_m=0 advection+micro matches diffusion_fast to ~1e-4 (the exact skewed kernel), incl. sharp
    pulse AND step, constant AND variable flow, with retardation. Tight because step 1 is the
    near-exact part (only the Ibar interpolation is approximate at the default fineness)."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0) if flow_kind == "const" else _variable_flow(n, 0.4, seed=3)
    cin = _pulse(n) if inp == "pulse" else _step(n)
    kw = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(25),
        "streamline_length": SL,
        "molecular_diffusivity": 0.0,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": retardation,
        "spinup": spinup,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    # measured worst case ~2.9e-4 (variable-flow pulse); bound at 5e-4.
    assert _peak_rel(cout_ff, cout_df) < 5e-4


def test_step1_zero_dispersion_single_pv_is_exact():
    """alpha_L=0, D_m=0, single streamtube is pure advection: the antiderivative is the exact step
    ramp and the aligned grid evaluates it off the lone kink cell, so parity is machine precision
    (not just ~1e-4). Robust to the pore volume (kink alignment)."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    cin = _pulse(n)
    for v_pore in (500.0, 517.3, 623.1):
        kw = {
            "cin": cin,
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": tedges.copy(),
            "aquifer_pore_volumes": np.array([v_pore]),
            "streamline_length": SL,
            "molecular_diffusivity": 0.0,
            "longitudinal_dispersivity": 0.0,
        }
        assert _peak_rel(infiltration_to_extraction(**kw), df_i2e(**kw)) < 1e-12


def test_step1_kernel_fineness_convergence(monkeypatch):
    """The only step-1 approximation is the Ibar interpolation; its error must DECREASE monotonically
    as _KERNEL_FINE increases (guards against a nearest-neighbour / constant-offset interp bug)."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    kw = {
        "cin": _pulse(n),
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(25),
        "streamline_length": SL,
        "molecular_diffusivity": 0.0,
        "longitudinal_dispersivity": 1.0,
    }
    cout_df = df_i2e(**kw)
    errs = []
    for fine in (16, 64, 256):
        monkeypatch.setattr(fff_module, "_KERNEL_FINE", fine)
        errs.append(_peak_rel(infiltration_to_extraction(**kw), cout_df))
    assert errs[0] > errs[1] > errs[2]  # strictly decreasing
    assert errs[2] < 2e-6  # measured 4.6e-7 at fine=256


# =============================================================================
# Full method (advection+micro + molecular Gaussian) vs diffusion_fast, per-regime bounds.
# =============================================================================


@pytest.mark.parametrize("flow_kind", ["const", "cv01", "cv03", "cv06"])
@pytest.mark.parametrize("inp", ["pulse", "smooth"])
def test_full_typical_alpha_present_flow_independent(inp, flow_kind):
    """With mechanical dispersion present (alpha_L>0, the typical regime) the method stays within the
    ~1e-3 bound for constant AND strongly variable flow: the advection+micro part is exact and
    volume-stationary, and the sub-dominant molecular commutator residual stays small under flow
    variation."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = {
        "const": np.full(n, 100.0),
        "cv01": _variable_flow(n, 0.1, 1),
        "cv03": _variable_flow(n, 0.3, 2),
        "cv06": _variable_flow(n, 0.6, 5),
    }[flow_kind]
    cin = _pulse(n) if inp == "pulse" else _smooth(n)
    kw = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(25),
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.0,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    assert _peak_rel(cout_ff, cout_df) < 1e-3  # measured worst ~6.8e-4 (R=2.0 sharp pulse)


def test_flow_independence_with_microdispersion():
    """The advection+micro part (D_m=0) is volume-stationary, so its error vs diffusion_fast stays at
    the Ibar-interpolation floor under strongly variable flow -- NOT a ratio test: with the
    effective-dispersivity fold the constant-flow error is near-zero (the operator is exact there),
    so err_cv06/err_const blows up while both stay tiny; the meaningful guard is an ABSOLUTE bound.
    With realistic sub-dominant molecular diffusion (D_m>0) the molecular part carries a small
    flow-DEPENDENT commutator residual (that is the honest behaviour, not a regression) -- bounded,
    and strictly larger than the D_m=0 part."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    common = {
        "cin": _smooth(n),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(25),
        "streamline_length": SL,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.0,
    }

    def worst_cv06(d_m):
        return max(
            _peak_rel(
                infiltration_to_extraction(flow=_variable_flow(n, 0.6, seed=s), molecular_diffusivity=d_m, **common),
                df_i2e(flow=_variable_flow(n, 0.6, seed=s), molecular_diffusivity=d_m, **common),
            )
            for s in range(8)
        )

    err_micro = worst_cv06(0.0)  # pure advection+micro, volume-stationary
    err_molecular = worst_cv06(0.05)  # + sub-dominant molecular commutator
    assert err_micro < 1e-4  # measured worst-of-8-seeds ~3.9e-5 (Ibar interpolation floor)
    assert err_molecular < 5e-4  # measured worst-of-8-seeds ~1.3e-4 (the flow-dependent commutator)
    assert err_molecular > err_micro  # the molecular term is what carries the flow dependence


@pytest.mark.parametrize("retardation", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("d_m", [0.01, 0.5, 1.0])
@pytest.mark.parametrize("alpha_l", [0.0, 1.0])
@pytest.mark.parametrize("inp", ["pulse", "smooth"])
def test_constant_flow_exact_all_regimes(inp, alpha_l, d_m, retardation):
    """At CONSTANT flow the effective-dispersivity fold reproduces diffusion_fast to the
    Ibar-interpolation floor across every realistic regime (Peclet Pe = L/alpha_eff >> 1) -- including
    heat (R>1, large D_m) and the molecular-dominant sharp corner. (The extreme low-Peclet corner
    Pe <~ 2, where the fine-grid sample cap coarsens the wide band, is out of scope -- see the module
    docstring.) This is the headline property: molecular diffusion D_m*tau maps
    exactly to an added dispersivity D_m*r_vpv/(L*Q) at constant flow, so the SKEWED molecular
    breakthrough is reproduced, not merely its second moment. The large-D_m buckets (0.5, 1.0) are the
    load-bearing guard: the old global time-domain Gaussian leaves up to ~3.6e-2 here (caught in the
    high-R / sharp-pulse buckets), and dropping or mis-scaling the molecular term blows up further.
    Replaces the old per-regime bound sweep (its buckets are subsumed; D_m=0 flow-independence is
    covered by test_step1_advection_micro_parity)."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = _pulse(n) if inp == "pulse" else _smooth(n)
    kw = {
        "cin": cin,
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(25),
        "streamline_length": SL,
        "molecular_diffusivity": d_m,
        "longitudinal_dispersivity": alpha_l,
        "retardation_factor": retardation,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    assert _peak_rel(cout_ff, cout_df) < 5e-4  # measured worst ~1.7e-4 (alpha_L=0 sharp kink, Ibar floor)


def test_full_bimodal_apvd():
    """A bimodal APVD (no gamma assumption) at constant flow: tight everywhere with the eff-disp fold,
    including the molecular-dominant sharp corner that the old symmetric time-Gaussian left at ~5e-2."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    bimodal = np.array([300.0, 320.0, 340.0, 900.0, 920.0, 940.0])
    base = {
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": bimodal,
        "streamline_length": SL,
    }
    # alpha_L present -> tight
    ff = infiltration_to_extraction(cin=_smooth(n), molecular_diffusivity=0.1, longitudinal_dispersivity=1.0, **base)
    df = df_i2e(cin=_smooth(n), molecular_diffusivity=0.1, longitudinal_dispersivity=1.0, **base)
    assert _peak_rel(ff, df) < 5e-4
    # molecular-dominant sharp -> exact at constant flow (skew captured), measured ~1.4e-4
    ffp = infiltration_to_extraction(cin=_pulse(n), molecular_diffusivity=0.3, longitudinal_dispersivity=0.0, **base)
    dfp = df_i2e(cin=_pulse(n), molecular_diffusivity=0.3, longitudinal_dispersivity=0.0, **base)
    assert _peak_rel(ffp, dfp) < 1e-3


def test_full_per_pore_volume_arrays_mixed_diffusivity_retardation():
    """Per-streamtube L / D_m / alpha_L arrays with D_m mixed zero/non-zero and R != 1. Tight bound is
    load-bearing: the effective dispersivity D_m*r_vpv/(L*q_mean) is PER STREAMTUBE, so a regression
    that used the bundle-mean r_vpv (mean_r_vpv) instead of the per-tube r_vpv errs ~1.5e-5 -- caught
    by the 5e-6 bound but not by a loose one."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    kw = {
        "cin": _smooth(n),
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([350.0, 450.0, 500.0, 600.0, 700.0, 850.0]),
        "streamline_length": np.array([70.0, 75.0, 80.0, 85.0, 90.0, 100.0]),
        "molecular_diffusivity": np.array([0.0, 0.05, 0.0, 0.08, 0.0, 0.03]),
        "longitudinal_dispersivity": np.array([0.5, 1.0, 1.5, 1.0, 2.0, 0.8]),
        "retardation_factor": 2.7,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    assert _peak_rel(cout_ff, cout_df) < 5e-6  # measured ~1.8e-6; per-tube skew fold (mean_r_vpv mutant ~1.5e-5)


@pytest.mark.parametrize("retardation", [1.0, 3.0])
def test_molecular_term_is_load_bearing(retardation):
    """Prove the molecular effective-dispersivity term is doing real work AND that its retardation
    dependence is correct. At a large pore volume with strong D_m the molecular breakthrough is a big
    part of the signal: including it (the eff-disp fold) matches diffusion_fast to the Ibar floor,
    while dropping it (D_m=0) errs ~0.8. V_pore is held fixed and R varied, so r_vpv = R*V_pore -- and
    hence alpha_eff = alpha_L + D_m*r_vpv/(L*q_mean) -- genuinely changes with R (unlike a fixed-r_vpv
    setup, where the R parametrization would run identical math); a wrong R / r_vpv dependence in the
    fold breaks the match."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    base = {
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([2000.0]),  # fixed V_pore -> r_vpv = R*2000 varies with R
        "streamline_length": SL,
        "longitudinal_dispersivity": 0.0,
        "retardation_factor": retardation,
    }
    ref = df_i2e(cin=_pulse(n), molecular_diffusivity=0.3, **base)
    with_molecular = infiltration_to_extraction(cin=_pulse(n), molecular_diffusivity=0.3, **base)
    without_molecular = infiltration_to_extraction(cin=_pulse(n), molecular_diffusivity=0.0, **base)
    err_with = _peak_rel(with_molecular, ref)
    err_without = _peak_rel(without_molecular, ref)
    assert err_with < 1e-3  # measured ~1.6e-5 (R=3) .. 7.4e-5 (R=1) -- the Ibar floor
    assert err_with < err_without / 3.0  # molecular term is load-bearing (drop it -> err ~0.8)


@pytest.mark.parametrize(
    ("n", "pv", "retardation", "d_m", "length", "start", "k_tail", "bound"),
    [
        (150, 6000.0, 2.0, 0.05, 80.0, 40, 10, 1e-4),  # config A: front lands at the record end
        (400, 10000.0, 3.0, 0.01, 100.0, 100, 20, 1e-5),  # config B: molecular tail into the last ~20 bins
    ],
)
def test_record_end_molecular_tail_matches_df(n, pv, retardation, d_m, length, start, k_tail, bound):
    """DFF-F1 regression. A molecular early-arrival tail that lands at/beyond the last output edge is
    built directly from in-record cin by the volume-domain fold (the old time-domain Gaussian could
    only redistribute in-record mass, so it lost this tail entirely -- peak-rel 1.0 at the record end).
    Assert VALUE parity on the raw trailing slice (NOT via _peak_rel, which skips the last 8 bins --
    exactly the tail), plus NaN-pattern parity (the tail bins are valid in both, just wrong-valued in
    the old code, so the NaN check alone would not have caught this)."""
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.zeros(n)
    cin[start : start + 4] = 1.0
    kw = {
        "cin": cin,
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([pv]),
        "streamline_length": length,
        "molecular_diffusivity": d_m,
        "longitudinal_dispersivity": 0.0,
        "retardation_factor": retardation,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    peak = np.nanmax(np.abs(cout_df[~np.isnan(cout_df)]))
    tail_err = np.nanmax(np.abs(cout_ff[-k_tail:] - cout_df[-k_tail:])) / peak
    assert tail_err < bound  # old time-Gaussian lost the tail (1.0); measured A 1.1e-5 / B 1.9e-6


def test_subbin_molecular_no_aliasing():
    """DFF-F2 regression. The old time-domain Gaussian was the integer-SAMPLED kernel; for sub-bin
    molecular width it aliased badly on an edge-aligned sharp feature (peak-rel 0.327) and on a
    discrete-APVD comb (0.249). The volume-domain fold has no bin-index kernel at all, so both collapse
    to the Ibar floor."""
    # edge-aligned 1-bin pulse, single tube, sub-bin molecular width
    n = 150
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.zeros(n)
    cin[40] = 1.0
    single = {
        "cin": cin,
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([4000.0]),
        "streamline_length": SL,
        "molecular_diffusivity": 0.01,
        "longitudinal_dispersivity": 0.0,
    }
    assert _peak_rel(infiltration_to_extraction(**single), df_i2e(**single)) < 5e-3  # old 0.327; measured ~6.8e-4
    # discrete-APVD comb: many streamtubes each smearing sub-bin
    n = 320
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.zeros(n)
    cin[40] = 1.0
    comb = {
        "cin": cin,
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": gamma.bins(mean=4000.0, std=1000.0, n_bins=25)["expected_values"],
        "streamline_length": SL,
        "molecular_diffusivity": 0.005,
        "longitudinal_dispersivity": 0.0,
    }
    assert _peak_rel(infiltration_to_extraction(**comb), df_i2e(**comb)) < 5e-3  # old comb 0.249; measured ~1.2e-3


def test_variable_flow_molecular_commutator_bounded():
    """DFF-F3 regression. Under variable flow the molecular width is local, but the fold freezes q_mean
    -> a commutator residual. The old global time-Gaussian obeyed err ~ |sigma_loc/sigma_bar - 1| and
    reached ~0.9 here; the volume-domain fold carries the dominant 1/Q(t*) factor per streamtube and
    stays bounded. The sinusoidal flow period is pinned to the breakthrough time tau_bt = r_vpv/q_mean
    (resonance is the worst case). A seasonal (annual) period would push the residual higher -- the
    documented 'use diffusion_fast' corner -- so this bound must NOT be read as a universal
    variable-flow guarantee; it guards against a regression back to the global-sigma behaviour."""
    n = 480
    t = np.arange(n)
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    pv, retardation, length = 6000.0, 2.0, 100.0
    # q_mean = mean(flow) = 100 (the sinusoid averages to 0); tau_bt = R*pv/q_mean = 120 -> pin period.
    period = retardation * pv / 100.0
    flow = 100.0 * (1.0 + 0.5 * np.sin(2.0 * np.pi * t / period))
    cin = np.zeros(n)
    cin[90:94] = 1.0  # arrival in a rising-flow phase (worst placement)
    kw = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([pv]),
        "streamline_length": length,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 0.0,
        "retardation_factor": retardation,
    }
    assert _peak_rel(infiltration_to_extraction(**kw), df_i2e(**kw)) < 0.08  # old global-sigma ~0.9; measured ~2.4e-2


def test_variable_flow_bundle_molecular_commutator_bounded():
    """Companion to test_variable_flow_molecular_commutator_bounded (single tube) and
    test_full_typical_alpha_present_flow_independent (molecular sub-dominant): a gamma BUNDLE under
    variable flow with molecular diffusion a non-trivial fraction of the dispersion. The effective
    dispersivity D_m*r_vpv/(L*q_mean) is per streamtube, so this guards that the frozen-q_mean
    commutator stays bounded across the bundle -- a regression that froze q_mean at a bundle-wrong
    value, or used mean_r_vpv under variable flow, is invisible to the single-tube F3 test and to the
    sub-dominant alpha_L=1 typical test."""
    n = 240
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    common = {
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(25),
        "streamline_length": SL,
        "molecular_diffusivity": 0.1,
        "longitudinal_dispersivity": 0.2,
        "retardation_factor": 2.0,
    }
    worst = 0.0
    for cin in (_pulse(n), _smooth(n)):
        for seed in range(6):
            flow = _variable_flow(n, 0.4, seed)
            worst = max(
                worst,
                _peak_rel(
                    infiltration_to_extraction(cin=cin, flow=flow, **common),
                    df_i2e(cin=cin, flow=flow, **common),
                ),
            )
    assert worst < 3e-3  # measured worst-of-6-seeds x {pulse,smooth} ~7.0e-4 (per-tube commutator, bundle)


# =============================================================================
# flow_out / non-aligned output grid (exercises the shared _cout_cumulative_volume anchoring).
# =============================================================================


def test_flow_out_equals_flow_matches_omitted():
    """Passing flow_out == flow on an aligned grid is identical to omitting it (same Vc)."""
    n = 180
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    common = {
        "cin": _smooth(n),
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(10),
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
    }
    a = infiltration_to_extraction(**common)
    b = infiltration_to_extraction(**common, flow_out=flow.copy())
    assert_allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True)


def test_coarse_cout_grid_with_flow_out():
    """A coarser cout grid + flow_out (the anchored-cumsum Vc branch) matches diffusion_fast to the
    forward accuracy."""
    n = 180
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    cout_tedges = pd.date_range("2020-01-01", periods=n // 3 + 1, freq="3D")
    flow_out = np.full(len(cout_tedges) - 1, 100.0)
    kw = {
        "cin": _smooth(n),
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": _apv(10),
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
        "flow_out": flow_out,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    assert _peak_rel(cout_ff, cout_df) < 1e-3  # measured ~5e-6


def test_coarse_cout_grid_strong_molecular():
    """Strong molecular diffusion (D_m=0.5) on a COARSE cout grid + flow_out (the anchored-cumsum Vc
    branch). The molecular term is now folded into the volume-domain Ibar (widening alpha_eff and the
    band), so it is independent of the output bin width -- unlike the old time-domain Gaussian, whose
    width had to be converted with the cout-grid spacing. This exercises the wide-band eff-disp fold
    through the coarse-cout Vc path; it matches diffusion_fast to the Ibar floor at constant flow."""
    n = 360
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    cout_tedges = pd.date_range("2020-01-01", periods=n // 4 + 1, freq="4D")
    flow_out = np.full(len(cout_tedges) - 1, 100.0)
    kw = {
        "cin": _smooth(n, freq=12.0),
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([1000.0]),
        "streamline_length": SL,
        "molecular_diffusivity": 0.5,
        "longitudinal_dispersivity": 1.0,
        "flow_out": flow_out,
    }
    assert _peak_rel(infiltration_to_extraction(**kw), df_i2e(**kw)) < 1e-5  # measured ~8.9e-7


# =============================================================================
# Physical invariants (tight -- these stay tight even though the module is approximate).
# =============================================================================


@pytest.mark.parametrize("retardation", [1.0, 2.5])
def test_constant_input_constant_output(retardation):
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout = infiltration_to_extraction(
        cin=np.full(n, 10.0),
        flow=np.full(n, 100.0),
        tedges=tedges,
        cout_tedges=tedges.copy(),
        aquifer_pore_volumes=_apv(25),
        streamline_length=SL,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
        retardation_factor=retardation,
    )
    valid = ~np.isnan(cout)
    assert valid.sum() > 50
    assert_allclose(cout[valid], 10.0, atol=1e-9, rtol=0)  # measured ~1.8e-15


def test_output_bounded_by_input():
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = _smooth(n, freq=4.0)
    cout = infiltration_to_extraction(
        cin=cin,
        flow=np.full(n, 100.0),
        tedges=tedges,
        cout_tedges=tedges.copy(),
        aquifer_pore_volumes=_apv(25),
        streamline_length=SL,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
    )
    valid = ~np.isnan(cout)
    assert np.all(cout[valid] >= cin.min() - 1e-12)
    assert np.all(cout[valid] <= cin.max() + 1e-12)


def test_zero_diffusion_matches_advection():
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = _smooth(n, freq=4.0)
    cout_ff = infiltration_to_extraction(
        cin=cin,
        flow=np.full(n, 100.0),
        tedges=tedges,
        cout_tedges=tedges.copy(),
        aquifer_pore_volumes=np.array([500.0]),
        streamline_length=SL,
        molecular_diffusivity=0.0,
        longitudinal_dispersivity=0.0,
    )
    cout_adv = advection_i2e(
        cin=cin,
        flow=np.full(n, 100.0),
        tedges=tedges,
        cout_tedges=tedges.copy(),
        aquifer_pore_volumes=np.array([500.0]),
    )
    assert _peak_rel(cout_ff, cout_adv) < 1e-12


def test_forward_zero_flow_gap_no_smear_into_neighbours():
    """A pump-off gap (zero through-flow) must not corrupt the valid neighbours. Because molecular
    diffusion lives in the volume-domain band (a zero-flow gap adds zero cumulative volume, so its
    coefficients are zero), there is no time-domain smear that could leak the gap's missing data into
    the neighbours: a constant input is preserved EXACTLY across the gap (every valid bin == cin) while
    the gap-interior bins stay NaN, matching diffusion_fast's NaN pattern."""
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[40:44] = 0.0  # 4-bin pump-off gap
    kw = {
        "cin": np.full(n, 7.3),
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": gamma.bins(mean=500.0, std=200.0, n_bins=25)["expected_values"],
        "streamline_length": 10.0,
        "molecular_diffusivity": 0.1,
        "longitudinal_dispersivity": 0.5,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))  # gap-interior bins NaN in both
    valid = ~np.isnan(cout_ff)
    assert valid.sum() > 100
    assert_allclose(cout_ff[valid], 7.3, atol=1e-12, rtol=0)  # constant preserved across the gap


# =============================================================================
# Mass conservation: tight at constant flow; characterised (NOT conserved) at variable flow.
# =============================================================================


def test_column_mass_exact_at_constant_flow():
    """At constant flow the breakthrough operator conserves volumetric column mass exactly, so the
    probe-built operator's volume-weighted column sums equal the infiltrated volume per cin bin to
    machine precision -- a genuine conservation check (holds for D_m>0)."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)

    def call(cin_arr):
        return infiltration_to_extraction(
            cin=cin_arr,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges.copy(),
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=SL,
            molecular_diffusivity=0.3,
            longitudinal_dispersivity=1.0,
        )

    w = _build_w_via_probes(call, n)
    v_in = flow * 1.0
    ratios = (v_in[:, None] * w).sum(axis=0)[20:180] / v_in[20:180]
    assert_allclose(ratios, 1.0, atol=1e-12, rtol=0)


def test_column_mass_conserved_at_variable_flow_matches_diffusion_fast():
    """Under variable flow, on fully-broken-through interior bins the approximate operator conserves
    volumetric column mass just as the exact diffusion_fast does -- the method does not introduce
    extra mass error. (The molecular effective-dispersivity term is mass-conserving in the volume
    domain; it is guarded by test_molecular_term_is_load_bearing.) Probe-built W, so non-tautological."""
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = _variable_flow(n, 0.5, seed=5)
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([500.0]),
        "streamline_length": SL,
        "molecular_diffusivity": 0.3,
        "longitudinal_dispersivity": 1.0,
    }
    w_ff = _build_w_via_probes(lambda c: infiltration_to_extraction(cin=c, **common), n)
    w_df = _build_w_via_probes(lambda c: df_i2e(cin=c, **common), n)
    v_in = flow * 1.0
    # Fully-broken-through interior bins (both operators have full column support there).
    full = (w_ff.sum(axis=0) > 1.0 - 1e-3) & (w_df.sum(axis=0) > 1.0 - 1e-3)
    full[:8] = False
    full[-8:] = False
    assert full.sum() > 10
    cm_ff = (v_in[:, None] * w_ff).sum(axis=0)[full] / v_in[full]
    cm_df = (v_in[:, None] * w_df).sum(axis=0)[full] / v_in[full]
    assert np.max(np.abs(cm_ff - 1.0)) < 1e-8  # conserves like the exact operator (measured ~3e-10)
    assert np.max(np.abs(cm_ff - cm_df)) < 1e-8  # introduces no extra mass error (measured ~9e-11)


# =============================================================================
# spinup=None (partial breakthrough) and edge cases.
# =============================================================================


def test_spinup_none_partial_breakthrough_masks_like_diffusion_fast():
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.linspace(2.0, 9.0, n)
    kw = {
        "cin": cin,
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([900.0]),  # long spin-up region of partial breakthrough
        "streamline_length": 60.0,
        "molecular_diffusivity": 0.1,
        "longitudinal_dispersivity": 1.0,
        "spinup": None,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    # Confirm partial-breakthrough bins exist (row sums strictly between 0 and 1).
    w = _build_w_via_probes(lambda c: df_i2e(**{**kw, "cin": c}), n)
    rowsum = w.sum(axis=1)
    assert ((rowsum > 1e-6) & (rowsum < 1.0 - 1e-6)).sum() >= 3
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    assert _peak_rel(cout_ff, cout_df) < 1e-3  # measured ~3.1e-4


@pytest.mark.parametrize("flow_rate", [0.0, 1e-6])
def test_degenerate_flow_all_nan_like_diffusion_fast(flow_rate):
    """Zero / tiny flow: nothing breaks through within the (extended) record -> all NaN, matching
    diffusion_fast, without crashing or exploding the kernel grid / molecular sigma."""
    n = 50
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    kw = {
        "cin": np.linspace(1.0, 5.0, n),
        "flow": np.full(n, flow_rate),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([500.0]),
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    assert np.all(np.isnan(cout_ff))


def test_band_wider_than_series():
    """Large PV so the breakthrough band (front offset ~50 bins) spans more than the cin record, yet
    the record is long enough that PART of the real (non-warm-start) cin signal breaks through. The
    native build must reproduce diffusion_fast both on the flat warm-start region AND on the varying
    region carried by the real signal -- so the assertion is pinned on the bins where cout deviates
    from the constant warm-start value, not only on the preserved flat baseline. D_m=0 isolates the
    wide native banded build (the molecular Gaussian would otherwise dominate this sharp corner)."""
    n = 70
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    warm = 8.0  # warm-start value = cin[0]
    cin = np.full(n, 2.0)
    cin[:20] = warm  # a long high block, then a drop to 2.0 that breaks through near the tail
    kw = {
        "cin": cin,
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([5000.0]),  # residence ~50 bins < n=70, band > record
        "streamline_length": SL,
        "molecular_diffusivity": 0.0,
        "longitudinal_dispersivity": 1.0,
    }
    cout_ff = infiltration_to_extraction(**kw)
    cout_df = df_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    # The real (varying) part of the signal must break through, not just the flat warm-start.
    varying = ~np.isnan(cout_df) & (np.abs(cout_df - warm) > 1e-3)
    assert varying.sum() > 10
    peak_rel = np.nanmax(np.abs(cout_ff[varying] - cout_df[varying])) / np.nanmax(np.abs(cout_df[varying]))
    assert peak_rel < 1e-4  # measured ~1.1e-6 (D_m=0 wide native band is near-exact)
    assert_allclose(cout_ff, cout_df, atol=1e-3, rtol=0, equal_nan=True)


# =============================================================================
# Reverse: fast approximate banded deconvolution of the SAME operator as the forward.
# =============================================================================


@pytest.mark.parametrize(
    ("cv", "retardation", "n_seeds", "bound", "note"),
    [
        (0.0, 1.0, 1, 1e-9, "constant flow R=1: self-consistent to machine precision, measured ~3e-11"),
        (0.0, 2.7, 1, 5e-4, "constant flow + retardation, measured ~1e-4"),
        (0.3, 1.0, 8, 1e-3, "variable flow R=1, worst of 8 seeds ~3.9e-4"),
        (0.6, 2.0, 8, 1e-2, "strong variable flow + retardation, worst of 8 seeds ~5.2e-3"),
    ],
)
def test_reverse_round_trip_recovers_signal(cv, retardation, n_seeds, bound, note):
    """Forward (approximate) then reverse recovers cin. Because the reverse deconvolves the SAME
    operator the forward applies, the round trip is self-consistent -- machine precision at constant
    flow R=1, limited only by the deconvolution conditioning otherwise (NOT by the forward's
    approximation of diffusion_fast). For variable flow the bound is the WORST over a fixed seed set:
    the conditioning of the ill-posed deconvolution varies with the flow realization, so a single
    seed under-reports the spread (one lucky seed can sit ~50x below the worst). The reverse is
    warning-clean, so no warning suppression is needed."""
    n = 240
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.sin(np.linspace(0.0, 2.0 * np.pi, n)) + 5.0
    worst = 0.0
    for seed in range(n_seeds):
        flow = np.full(n, 100.0) if cv == 0.0 else _variable_flow(n, cv, seed)
        common = {
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": tedges.copy(),
            "aquifer_pore_volumes": _apv(25),
            "streamline_length": SL,
            "molecular_diffusivity": 0.05,
            "longitudinal_dispersivity": 1.0,
            "retardation_factor": retardation,
        }
        cout = infiltration_to_extraction(cin=cin, **common)
        cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)
        cin_rec = extraction_to_infiltration(cout=cout_clean, **common)
        valid = ~np.isnan(cin_rec)
        valid[:35] = False
        valid[-35:] = False
        assert valid.sum() > 100
        worst = max(worst, _peak_rel(cin_rec, cin, skip=35))
    assert worst < bound, note


def test_reverse_molecular_dominant_round_trip():
    """Exercise the reverse at a WIDE molecular band. A large pore volume + strong D_m makes the
    effective-dispersivity molecular smear a big part of the operator, and a sharp input makes it
    load-bearing. The reverse deconvolves exactly the operator the forward built, so the round trip is
    self-consistent to ~1e-5 -- a structural check that the wide molecular band is assembled and solved
    consistently in the reverse (molecular correctness itself is owned by the forward tests)."""
    n = 300
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.sin(np.linspace(0.0, 16.0 * np.pi, n)) + 5.0  # sharp -> the smear matters
    common = {
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([2000.0]),  # large PV -> wide molecular band
        "streamline_length": SL,
        "molecular_diffusivity": 0.5,
        "longitudinal_dispersivity": 1.0,
    }
    cout = infiltration_to_extraction(cin=cin, **common)
    cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)
    cin_rec = extraction_to_infiltration(cout=cout_clean, **common)
    assert _peak_rel(cin_rec, cin, skip=60) < 1e-3  # measured ~1.2e-6 (self-consistent at the wide band)


def test_reverse_on_true_operator_data_matches_df_reverse():
    """DFF-F4 regression. On cout generated by the TRUE operator (diffusion_fast), the reverse is only
    as accurate as the forward. With the effective-dispersivity fold the forward is near-exact at
    constant flow, so the fast reverse REPRODUCES diffusion_fast's own reverse -- there is no forward
    mismatch for the deconvolution to amplify. (The old symmetric-Gaussian forward mismatch made the
    fast reverse diverge from df's reverse by ~2e-2 on the same data.)"""
    n = 300
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    common = {
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": np.array([6000.0]),
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 0.0,  # molecular-only -> the forward mismatch the reverse would amplify
        "retardation_factor": 2.0,
    }
    cout_true = df_i2e(cin=_smooth(n), **common)
    cout_clean = np.where(np.isnan(cout_true), np.nanmean(cout_true), cout_true)
    fff_rev = extraction_to_infiltration(cout=cout_clean, **common)
    df_rev = df_e2i(cout=cout_clean, **common)
    assert _peak_rel(fff_rev, df_rev, skip=20) < 1e-5  # old code diverges from df-reverse by ~2.4; measured ~2.4e-6


def test_forward_and_reverse_warning_clean():
    """Both directions must be warning-clean on valid inputs (variable flow, R != 1, D_m > 0), even
    though the internals use np.errstate-guarded divisions and a banded Cholesky. Encodes the
    documented invariant that no warning suppression is needed around a normal call."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.sin(np.linspace(0.0, 6.0 * np.pi, n)) + 5.0
    common = {
        "flow": _variable_flow(n, 0.3, seed=2),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(25),
        "streamline_length": SL,
        "molecular_diffusivity": 0.1,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.0,
    }
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        cout = infiltration_to_extraction(cin=cin, **common)
        cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)
        extraction_to_infiltration(cout=cout_clean, **common)
    assert recorded == [], [str(w.message) for w in recorded]


def test_reverse_banded_matches_dense_solve_of_same_operator():
    """Structural correctness: the banded assembly of the breakthrough operator W + banded Tikhonov
    solve must match a DENSE solve of the SAME approximate operator. The dense operator is
    reconstructed from public forward probes (independent of the reverse internals) and solved with the
    dense Tikhonov path; agreement to ~1e-10 proves the banded assembly and banded solver are
    equivalent to the dense reference -- the check a self-consistent round trip cannot give."""
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    common = {
        "flow": _variable_flow(n, 0.3, seed=3),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(15),
        "streamline_length": SL,
        "molecular_diffusivity": 0.2,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.0,
    }
    cout = np.sin(np.linspace(0.0, 6.0 * np.pi, n)) + 5.0
    w = _build_w_via_probes(lambda c: infiltration_to_extraction(cin=c, **common), n)
    good = ~np.isnan(infiltration_to_extraction(cin=np.zeros(n), **common))
    dense = solve_inverse_transport(
        w_forward=w, observed=cout, n_output=n, regularization_strength=1e-10, valid_rows=good
    )
    banded = extraction_to_infiltration(cout=cout, **common)
    both = ~np.isnan(dense) & ~np.isnan(banded)
    both[:15] = False
    both[-15:] = False
    assert both.sum() > 50
    assert np.nanmax(np.abs(dense[both] - banded[both])) / np.nanmax(np.abs(dense[both])) < 1e-8  # measured ~1e-10


def test_reverse_much_faster_than_diffusion_fast():
    """The point of the fast reverse: it assembles the banded breakthrough operator W with one Ibar
    gather (no per-pore-volume closed-form loop) and banded-solves it, whereas diffusion_fast
    evaluates the exact breakthrough per streamtube. The speedup grows with the streamtube count
    (measured ~14x at gamma-25, ~47x at gamma-100, n=2920). Assert the relative speedup (robust to CI
    load, since both scale together) plus a generous absolute ceiling (a quadratic-regression guard).

    The fast path here is only ~30 ms, so a single OS scheduling hiccup can inflate its measurement
    several-fold; the min-of-N over N samples rejects those transients (a clean sample dominates the
    min). N=7 and a 3x floor keep this reliably green on shared CI while still proving the fast
    reverse is several-fold faster (the diffusion_fast own-band optimization, issue #299, sped up the
    dense reference, so the constant-factor margin is measured conservatively)."""
    n = 1000
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    kw = {
        "cout": np.sin(np.linspace(0.0, 30.0 * np.pi, n)) + 5.0,
        "flow": _variable_flow(n, 0.3, seed=1),
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(50),
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.0,
    }
    extraction_to_infiltration(**kw)  # warm-up / import jit
    df_e2i(**kw)  # warm-up the dense reference too

    def best_of(fn, repeats=7):
        return min(_timed(fn) for _ in range(repeats))

    t_fast = best_of(lambda: extraction_to_infiltration(**kw))
    t_dense = best_of(lambda: df_e2i(**kw))
    assert t_fast < 2.0  # generous: measured ~30ms; guards a quadratic regression
    assert t_dense > 3.0 * t_fast  # nominal ~8-25x (grows with streamtube count); min-of-7 is CI-load-robust


def test_reverse_constant_cout_gives_constant_cin():
    """Constant extraction -> constant infiltration (row sums = 1, target = constant)."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = extraction_to_infiltration(
        cout=np.full(n, 7.0),
        flow=np.full(n, 100.0),
        tedges=tedges,
        cout_tedges=tedges.copy(),
        aquifer_pore_volumes=_apv(25),
        streamline_length=SL,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
        retardation_factor=2.0,
    )
    valid = ~np.isnan(cin)
    valid[:35] = False
    valid[-35:] = False
    assert valid.sum() > 50
    assert_allclose(cin[valid], 7.0, atol=1e-9, rtol=0)  # measured ~6e-13


def test_reverse_zero_cout_gives_zero_cin():
    """Zero extraction -> zero infiltration exactly (homogeneous solve)."""
    n = 150
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = extraction_to_infiltration(
        cout=np.zeros(n),
        flow=np.full(n, 100.0),
        tedges=tedges,
        cout_tedges=tedges.copy(),
        aquifer_pore_volumes=_apv(10),
        streamline_length=SL,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
    )
    assert_allclose(cin[~np.isnan(cin)], 0.0, atol=1e-12)


def test_reverse_zero_flow_gap_preserves_constant():
    """Reverse analogue of test_forward_zero_flow_gap_no_smear_into_neighbours: a constant extraction
    with a pump-off gap must deconvolve to a constant infiltration on the valid bins, without the
    gap's zero-mass rows corrupting the neighbours. The banded solver row-normalizes W (dividing each
    row by its band mass, so the gap rows' zero mass drops out), so constants are preserved to solver
    precision across the gap."""
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[40:44] = 0.0  # 4-bin pump-off gap
    cin = extraction_to_infiltration(
        cout=np.full(n, 7.3),
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges.copy(),
        aquifer_pore_volumes=gamma.bins(mean=500.0, std=200.0, n_bins=25)["expected_values"],
        streamline_length=10.0,
        molecular_diffusivity=0.1,
        longitudinal_dispersivity=0.5,
    )
    valid = ~np.isnan(cin)
    assert valid.sum() > 100
    assert_allclose(cin[valid], 7.3, atol=1e-9, rtol=0)  # measured ~3.5e-13 on the row-normalized W


def test_reverse_forwards_retardation_and_flow_out():
    """The reverse must forward EVERY transport parameter into the operator, not just the aligned-grid
    R=1 defaults: a round trip with retardation AND a VARIABLE flow_out on a coarse cout grid recovers
    cin. flow_out must be variable -- a constant flow_out on a constant-flow record places the cout
    edges identically to the omitted-flow_out interpolation, so it would be a no-op and not guard the
    flow_out path (dropping a variable flow_out blows the round trip up ~400x)."""
    n = 180
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = pd.date_range("2020-01-01", periods=n // 3 + 1, freq="3D")
    cin = np.sin(np.linspace(0.0, 4.0 * np.pi, n)) + 5.0
    flow_out = 100.0 * np.exp(np.random.default_rng(2).normal(0.0, 0.3, len(cout_tedges) - 1))
    common = {
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": _apv(15),
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.7,
        "flow_out": flow_out,
    }
    cout = infiltration_to_extraction(cin=cin, **common)
    cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)
    cin_rec = extraction_to_infiltration(cout=cout_clean, **common)
    assert _peak_rel(cin_rec, cin, skip=20) < 5e-3  # measured ~2.5e-3 (coarse grid + R + variable flow_out)


def test_reverse_regularization_must_be_positive():
    """The banded inverse needs lambda > 0 for Cholesky positive-definiteness (it cannot return the
    dense lambda=0 minimum-norm solution); lambda <= 0 must raise rather than silently misbehave."""
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    with pytest.raises(ValueError, match="regularization_strength must be"):
        extraction_to_infiltration(
            cout=np.full(n, 5.0),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=tedges.copy(),
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=SL,
            molecular_diffusivity=0.05,
            longitudinal_dispersivity=1.0,
            regularization_strength=0.0,
        )


def test_reverse_degenerate_flow_all_nan():
    """Zero through-flow: nothing constrains the infiltration signal -> all NaN, no crash."""
    n = 50
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = extraction_to_infiltration(
        cout=np.linspace(1.0, 5.0, n),
        flow=np.zeros(n),
        tedges=tedges,
        cout_tedges=tedges.copy(),
        aquifer_pore_volumes=np.array([500.0]),
        streamline_length=SL,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
    )
    assert cin.shape == (n,)
    assert np.all(np.isnan(cin))


def test_gamma_forward_matches_diffusion_fast():
    tedges, cout_tedges, flow = _make_transport_data(n_days=150)
    n = len(flow)
    kw = {
        "cin": np.sin(np.linspace(0.0, 6.0 * np.pi, n)) + 4.0,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "mean": 500.0,
        "std": 120.0,
        "n_bins": 25,
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
    }
    cout_ff = gamma_infiltration_to_extraction(**kw)
    cout_df = df_gamma_i2e(**kw)
    assert np.array_equal(np.isnan(cout_ff), np.isnan(cout_df))
    assert _peak_rel(cout_ff, cout_df) < 5e-4  # measured ~4.5e-6


def test_gamma_forward_matches_explicit():
    """The gamma forward wrapper must equal an explicit infiltration_to_extraction call on the same
    discretized bins, exactly. A coarse cout grid + variable flow_out + non-default R exercises every
    transport pass-through, so dropping any of retardation_factor / flow_out / saturation_threshold
    in the wrapper breaks the exact equality."""
    n = 180
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = pd.date_range("2020-01-01", periods=n // 3 + 1, freq="3D")
    flow_out = 100.0 * np.exp(np.random.default_rng(7).normal(0.0, 0.3, len(cout_tedges) - 1))
    common = {
        "cin": np.sin(np.linspace(0.0, 6.0 * np.pi, n)) + 4.0,
        "flow": np.full(n, 100.0),
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.7,  # non-default -> a dropped R pass-through breaks exact equality
        "flow_out": flow_out,  # variable -> a dropped flow_out pass-through breaks exact equality
    }
    bins = gamma.bins(mean=501.3, std=100.0, n_bins=20)["expected_values"]
    cout_gamma = gamma_infiltration_to_extraction(mean=501.3, std=100.0, n_bins=20, **common)
    cout_explicit = infiltration_to_extraction(aquifer_pore_volumes=bins, **common)
    assert_allclose(cout_gamma, cout_explicit, atol=0.0, rtol=0.0, equal_nan=True)


def test_gamma_reverse_round_trip():
    """gamma forward then gamma reverse recovers cin (constant flow R=1 -> machine precision)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n = len(flow)
    cin = np.sin(np.linspace(0.0, 2.0 * np.pi, n)) + 5.0
    kw = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "mean": 500.0,
        "std": 120.0,
        "n_bins": 25,
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
    }
    cout = gamma_infiltration_to_extraction(cin=cin, **kw)
    cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)
    cin_rec = gamma_extraction_to_infiltration(cout=cout_clean, **kw)
    assert _peak_rel(cin_rec, cin, skip=35) < 1e-6  # measured ~4e-11


def test_gamma_reverse_matches_explicit():
    """The gamma reverse wrapper must equal an explicit extraction_to_infiltration call on the same
    discretized bins, exactly (guards a dropped parameter in the wrapper)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=140)
    n = len(flow)
    cout = np.sin(np.linspace(0.0, 6.0 * np.pi, n)) + 5.0
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.7,  # non-default -> a dropped R pass-through breaks exact equality
    }
    bins = gamma.bins(mean=501.3, std=100.0, n_bins=20)["expected_values"]
    cin_gamma = gamma_extraction_to_infiltration(cout=cout, mean=501.3, std=100.0, n_bins=20, **common)
    cin_explicit = extraction_to_infiltration(cout=cout, aquifer_pore_volumes=bins, **common)
    assert_allclose(cin_gamma, cin_explicit, atol=0.0, rtol=0.0, equal_nan=True)


def test_gamma_wrappers_alpha_beta_equals_mean_std():
    """The (alpha, beta) parameterization of both gamma wrappers must reproduce the equivalent
    (mean, std) call (alpha/beta derived with the canonical mean_std_loc_to_alpha_beta converter)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=150)
    n = len(flow)
    mean, std = 500.0, 120.0
    alpha, beta = mean_std_loc_to_alpha_beta(mean=mean, std=std)
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "n_bins": 25,
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.0,
    }
    cin = np.sin(np.linspace(0.0, 6.0 * np.pi, n)) + 4.0
    fwd_ms = gamma_infiltration_to_extraction(cin=cin, mean=mean, std=std, **common)
    fwd_ab = gamma_infiltration_to_extraction(cin=cin, alpha=alpha, beta=beta, **common)
    assert_allclose(fwd_ab, fwd_ms, atol=0.0, equal_nan=True)

    cout = np.sin(np.linspace(0.0, 6.0 * np.pi, n)) + 5.0
    rev_ms = gamma_extraction_to_infiltration(cout=cout, mean=mean, std=std, **common)
    rev_ab = gamma_extraction_to_infiltration(cout=cout, alpha=alpha, beta=beta, **common)
    assert_allclose(rev_ab, rev_ms, atol=0.0, equal_nan=True)


# =============================================================================
# Performance smoke + input validation.
# =============================================================================


def test_large_n_is_fast():
    """A large series must stay well under O(N^2): n=20000 single call in a fraction of a second
    (catches an accidental quadratic regression without asserting a brittle absolute runtime)."""
    n = 20000
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = _variable_flow(n, 0.3, seed=1)
    cin = np.sin(np.linspace(0.0, 200.0 * np.pi, n)) + 5.0
    kw = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges.copy(),
        "aquifer_pore_volumes": _apv(50),
        "streamline_length": SL,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
    }
    infiltration_to_extraction(**kw)  # warm-up / import jit
    t0 = time.perf_counter()
    infiltration_to_extraction(**kw)
    assert time.perf_counter() - t0 < 1.0


def test_flow_out_required_when_grids_differ():
    n = 30
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = pd.date_range("2020-01-02", periods=n + 1, freq="D")
    with pytest.raises(ValueError, match="flow_out is required"):
        infiltration_to_extraction(
            cin=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=SL,
            molecular_diffusivity=0.05,
            longitudinal_dispersivity=1.0,
        )


def test_negative_diffusivity_rejected():
    tedges, cout_tedges, flow = _make_transport_data(n_days=30)
    with pytest.raises(ValueError, match="molecular_diffusivity must be non-negative"):
        infiltration_to_extraction(
            cin=np.ones(len(flow)),
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=SL,
            molecular_diffusivity=-1.0,
            longitudinal_dispersivity=1.0,
        )


def test_extraction_to_infiltration_gapped_cout_masked():
    """NaN gaps in cout are masked out of the banded inverse instead of raising (#321).

    Sparse lab samples leave NaN cout bins; the reverse operator must exclude
    those rows from the banded Tikhonov normal equations -- matching
    :func:`gwtransport.deposition.extraction_to_deposition` -- rather than
    reject the whole series. The reverse inverts the same approximate operator
    the forward applies, so with cout at 12h resolution (via ``flow_out``) the
    overdetermined surviving rows recover cin at the no-gap round-trip floor.
    """
    n_days = 120
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = pd.date_range("2020-01-01", periods=2 * n_days + 1, freq="12h")
    flow = np.full(n_days, 100.0)
    cin_true = 5.0 + 3.0 * np.sin(2 * np.pi * np.arange(n_days) / 30.0)
    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([517.0]),
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.01,
        "longitudinal_dispersivity": 0.1,
        "flow_out": np.full(2 * n_days, 100.0),
    }

    cout = infiltration_to_extraction(cin=cin_true, **kwargs)
    cout[[0, 90, 120, 121, 150, 239]] = np.nan  # boundary + scattered gaps

    cin_rec = extraction_to_infiltration(cout=cout, **kwargs)

    interior = slice(30, 95)
    assert_allclose(cin_rec[interior], cin_true[interior], atol=1e-10)
