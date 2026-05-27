import warnings as warn_module

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from gwtransport import gamma as gamma_utils
from gwtransport._time import tedges_to_days
from gwtransport.advection import infiltration_to_extraction as advection_i2e
from gwtransport.diffusion import gamma_infiltration_to_extraction as diffusion_gamma_i2e
from gwtransport.diffusion import infiltration_to_extraction as diffusion_exact
from gwtransport.diffusion_fast import (
    _validate_inputs,
    extraction_to_infiltration,
    gamma_extraction_to_infiltration,
    gamma_infiltration_to_extraction,
    infiltration_to_extraction,
)
from gwtransport.gamma import mean_std_loc_to_alpha_beta
from gwtransport.utils import partial_isin

# =============================================================================
# Helper: create common test data for transport functions
# =============================================================================


def _make_transport_data(*, n_days=200, flow_rate=100.0):
    """Create common test data for transport tests."""
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = tedges.copy()
    flow = np.full(n_days, flow_rate)
    return tedges, cout_tedges, flow


# =============================================================================
# Machine-precision tests for infiltration_to_extraction
# =============================================================================


def test_infiltration_to_extraction_zero_diffusion_matches_advection():
    """With zero diffusion, G = I, so result must match pure advection exactly."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin = np.sin(np.linspace(0, 4 * np.pi, n_days)) + 2.0

    aquifer_pore_volumes = np.array([500.0])

    cout_fast = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.0,
        mean_longitudinal_dispersivity=0.0,
    )

    cout_adv = advection_i2e(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
    )

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_adv)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_adv[valid], atol=1e-13)


def test_infiltration_to_extraction_constant_input():
    """Constant input yields constant output (exact, from row-sum = 1)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin = np.full(n_days, 10.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.03,
        mean_longitudinal_dispersivity=0.0,
    )

    valid = ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert_allclose(cout[valid], 10.0, atol=1e-12)


def test_infiltration_to_extraction_output_bounded_by_input():
    """Output concentration bounded by input range (convex combination)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin = np.sin(np.linspace(0, 4 * np.pi, n_days)) * 3.0 + 5.0

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.05,
        mean_longitudinal_dispersivity=1.0,
    )

    valid = ~np.isnan(cout)
    # Forward matrix is non-negative with row sums ≤ 1, so output is a
    # convex combination of input values — strictly within [min, max].
    assert np.all(cout[valid] >= np.min(cin) - 1e-14)
    assert np.all(cout[valid] <= np.max(cin) + 1e-14)


# =============================================================================
# Physical correctness tests for infiltration_to_extraction
# =============================================================================


def test_infiltration_to_extraction_multiple_pore_volumes():
    """Test with multiple pore volumes."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=300)
    n_days = len(flow)
    cin = np.zeros(n_days)
    cin[30:40] = 1.0

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([400.0, 500.0, 600.0]),
        mean_streamline_length=100.0,
        mean_molecular_diffusivity=0.03,
        mean_longitudinal_dispersivity=0.0,
    )

    assert len(cout) == n_days
    valid = ~np.isnan(cout)
    assert np.any(valid)
    assert np.all(cout[valid] >= -1e-10)


def test_infiltration_to_extraction_cout_tedges_different_resolution():
    """Test with coarser output grid."""
    n_days = 200
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = pd.date_range("2020-01-01", periods=n_days // 7 + 1, freq="7D")
    flow = np.full(n_days, 100.0)
    cin = np.full(n_days, 5.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.03,
        mean_longitudinal_dispersivity=0.0,
    )

    assert len(cout) == len(cout_tedges) - 1
    valid = ~np.isnan(cout)
    assert np.sum(valid) > 0
    assert_allclose(cout[valid], 5.0, atol=1e-12)


@pytest.mark.parametrize("retardation_factor", [1.0, 2.0, 3.5])
def test_infiltration_to_extraction_with_retardation(retardation_factor):
    """Test with different retardation factors."""
    n_days = 300
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = tedges.copy()
    cin = np.zeros(n_days)
    cin[50:] = 10.0
    flow = np.full(n_days, 100.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([1000.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.03,
        mean_longitudinal_dispersivity=0.0,
        retardation_factor=retardation_factor,
    )

    assert len(cout) == n_days
    valid = ~np.isnan(cout)
    assert np.any(valid)


@pytest.mark.parametrize("molecular_diffusivity", [0.01, 0.03, 0.08])
def test_infiltration_to_extraction_with_various_diffusivity(molecular_diffusivity):
    """Constant input preserved regardless of diffusivity."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin = np.full(n_days, 10.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=molecular_diffusivity,
        mean_longitudinal_dispersivity=0.0,
    )

    valid = ~np.isnan(cout)
    assert np.any(valid)
    assert_allclose(cout[valid], 10.0, atol=1e-12)


def test_infiltration_to_extraction_with_variable_flow():
    """Variable flow with constant input preserves concentration."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    t = np.arange(n_days)
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 30))
    cin = np.full(n_days, 10.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.03,
        mean_longitudinal_dispersivity=0.0,
    )

    valid = ~np.isnan(cout)
    assert np.any(valid)
    # Under variable flow, constant preservation holds because the combined
    # matrix (W_adv @ G) has row sums ≤ 1 for valid rows, and cin is constant.
    assert_allclose(cout[valid], 10.0, atol=1e-13)


# =============================================================================
# Machine-precision tests for retardation (R != 1)
#
# With R=1, the retardation factor cancels from the dx computation. These tests
# use non-integer R=2.7 to verify that the retardation factor is correctly
# included in the sigma calculation. Each test uses an invariant that holds
# exactly regardless of R.
# =============================================================================


@pytest.mark.parametrize("retardation_factor", [1.0, 2.7])
def test_retardation_constant_input(retardation_factor):
    """Constant cin with retardation produces constant output (exact, row sums)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=400)
    n_days = len(flow)
    cin = np.full(n_days, 10.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.05,
        mean_longitudinal_dispersivity=1.0,
        retardation_factor=retardation_factor,
    )

    valid = ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert_allclose(cout[valid], 10.0, atol=1e-12)


@pytest.mark.parametrize("retardation_factor", [1.0, 2.7, 5.0])
def test_retardation_zero_diffusion_matches_advection(retardation_factor):
    """Zero diffusion with retardation matches pure advection (exact, G=I), constant Q.

    Companion ``test_retardation_zero_diffusion_matches_advection_variable_flow`` extends
    to a variable-flow regime.
    """
    tedges, cout_tedges, flow = _make_transport_data(n_days=400)
    n_days = len(flow)
    cin = np.sin(np.linspace(0, 4 * np.pi, n_days)) + 2.0

    kwargs = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
        "retardation_factor": retardation_factor,
    }

    cout_fast = infiltration_to_extraction(
        **kwargs,
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.0,
        mean_longitudinal_dispersivity=0.0,
    )

    cout_adv = advection_i2e(**kwargs)

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_adv)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_adv[valid], atol=1e-13)


@pytest.mark.parametrize("retardation_factor", [1.0, 2.7, 5.0])
def test_retardation_zero_diffusion_matches_advection_variable_flow(retardation_factor):
    """Zero-diffusion match must hold under variable Q, too (extends the constant-flow sibling)."""
    n_days = 400
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    # Variable flow with both a weekly oscillation and a slow trend.
    flow = 100.0 + 60.0 * np.sin(np.arange(n_days) * 2 * np.pi / 7) + 0.05 * np.arange(n_days)
    cin = np.sin(np.linspace(0, 4 * np.pi, n_days)) + 2.0

    kwargs = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
        "retardation_factor": retardation_factor,
    }

    cout_fast = infiltration_to_extraction(
        **kwargs,
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.0,
        mean_longitudinal_dispersivity=0.0,
    )
    cout_adv = advection_i2e(**kwargs)

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_adv)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_adv[valid], atol=1e-12)


@pytest.mark.parametrize("retardation_factor", [1.0, 2.7])
def test_retardation_output_bounded_by_input(retardation_factor):
    """Output bounded by input range with retardation (convex combination)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=400)
    n_days = len(flow)
    cin = np.sin(np.linspace(0, 4 * np.pi, n_days)) * 3.0 + 5.0

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.05,
        mean_longitudinal_dispersivity=1.0,
        retardation_factor=retardation_factor,
    )

    valid = ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert np.all(cout[valid] >= np.min(cin) - 1e-14)
    assert np.all(cout[valid] <= np.max(cin) + 1e-14)


@pytest.mark.parametrize("retardation_factor", [1.0, 2.7])
def test_retardation_compressed_matches_uncompressed(retardation_factor):
    """Compressed non-uniform bins match uncompressed uniform bins with retardation (exact)."""
    n_days = 500
    tedges_full = pd.date_range("2019-12-31", periods=n_days + 1, freq="D")
    flow_full = np.full(n_days, 100.0)
    cin_full = np.zeros(n_days)
    cin_full[50] = 100.0

    # Compress to non-uniform bins
    cin_itedges = np.flatnonzero(np.diff(cin_full, prepend=1.0, append=1.0))
    flow_itedges = np.flatnonzero(np.diff(flow_full, prepend=1.0, append=1.0))
    itedges = np.unique(np.concatenate([cin_itedges, flow_itedges]))
    tedges_compressed = tedges_full[itedges]
    cin_compressed = cin_full[itedges[:-1]]
    flow_compressed = flow_full[itedges[:-1]]

    cout_tedges = tedges_full.copy()
    flow_out = np.full(n_days, 100.0)

    kwargs = {
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([10000.0]),
        "mean_streamline_length": 100.0,
        "mean_molecular_diffusivity": 1e-4,
        "mean_longitudinal_dispersivity": 1.0,
        "retardation_factor": retardation_factor,
        "flow_out": flow_out,
    }

    cout_compressed = infiltration_to_extraction(
        cin=cin_compressed, flow=flow_compressed, tedges=tedges_compressed, **kwargs
    )
    cout_uncompressed = infiltration_to_extraction(cin=cin_full, flow=flow_full, tedges=tedges_full, **kwargs)

    both_valid = ~np.isnan(cout_compressed) & ~np.isnan(cout_uncompressed)
    assert np.sum(both_valid) > 50
    assert_allclose(cout_compressed[both_valid], cout_uncompressed[both_valid], atol=0.0)


@pytest.mark.parametrize("retardation_factor", [1.0, 2.7])
def test_retardation_nonuniform_cout_tedges_constant_input(retardation_factor):
    """Constant cin with non-uniform cout_tedges produces constant output (exact)."""
    tedges = pd.date_range("2020-01-01", periods=401, freq="D")
    flow = np.full(400, 100.0)
    cin = np.full(400, 10.0)

    # Non-uniform output grid: mix of 1-day, 3-day, and 7-day bins
    cout_dates = [pd.Timestamp("2020-01-01")]
    t = cout_dates[0]
    end = tedges[-1]
    widths = [1, 3, 7]
    i = 0
    while t < end:
        t += pd.Timedelta(days=widths[i % len(widths)])
        if t <= end:
            cout_dates.append(t)
        i += 1
    cout_tedges = pd.DatetimeIndex(cout_dates)
    n_cout = len(cout_tedges) - 1

    # flow_out must align with cout_tedges
    flow_out = np.full(n_cout, 100.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.05,
        mean_longitudinal_dispersivity=1.0,
        retardation_factor=retardation_factor,
        flow_out=flow_out,
    )

    valid = ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert_allclose(cout[valid], 10.0, atol=1e-12)


# =============================================================================
# Tests for extraction_to_infiltration
# =============================================================================


def test_extraction_to_infiltration_constant_input():
    """Constant cout yields constant cin (Tikhonov with exact solution)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cout = np.full(n_days, 7.0)

    cin = extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.03,
        mean_longitudinal_dispersivity=0.0,
    )

    valid = ~np.isnan(cin)
    assert np.sum(valid) > 50
    assert_allclose(cin[valid], 7.0, atol=1e-13)


def test_round_trip():
    """Forward then inverse recovers the original signal at machine precision.

    The same closed-form forward matrix W is built in both directions, so the
    Tikhonov solve recovers the original cin exactly up to the floating-point
    floor of the dense solve (~1e-12).

    The pore volume is 517 (not 500) m^3: with constant flow the forward matrix
    is rank-deficient for a band of residence times (the closed-form
    breakthrough is a near-shift operator whose discretisation drops rank for
    those shifts, and the module warns to perturb the pore volume). 517 m^3 / Q
    = 5.17 days sits in a full-rank plateau (neighbours all recover cleanly), so
    the inverse is fully data-determined rather than pulled to the
    regularisation target.
    """
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin_original = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0

    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([517.0]),
        "mean_streamline_length": 80.0,
        "mean_molecular_diffusivity": 0.01,
        "mean_longitudinal_dispersivity": 0.0,
    }

    cout = infiltration_to_extraction(cin=cin_original, **kwargs)

    # Replace NaN with mean for inverse (NaN not allowed)
    cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)

    cin_recovered = extraction_to_infiltration(cout=cout_clean, **kwargs)

    # Compare where both forward and inverse give valid results
    valid = ~np.isnan(cin_recovered) & ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert_allclose(cin_recovered[valid], cin_original[valid], atol=1e-11)


# =============================================================================
# Integration test: diffusion_fast vs diffusion on same extraction grid
# =============================================================================


def test_diffusion_fast_vs_diffusion_same_grid():
    """Equivalence to ``gwtransport.diffusion`` on the ``flow_out=None`` path.

    Both modules report the Kreft-Zuber flux concentration C_F. This is the only
    equivalence-to-slow test that omits ``flow_out``, so the cout-bin volumes are
    interpolated from the flow grid rather than taken from ``flow_out`` directly;
    agreement is then to the interpolation floor (~1e-7), not the machine precision
    of the aligned ``flow_out`` path. Covers a smooth and a step input.
    """
    n_days = 200
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = tedges.copy()
    flow = np.full(n_days, 100.0)

    common_kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
    }
    fast_kwargs = {
        **common_kwargs,
        "mean_streamline_length": 80.0,
        "mean_molecular_diffusivity": 0.05,
        "mean_longitudinal_dispersivity": 0.0,
    }
    exact_kwargs = {
        **common_kwargs,
        "streamline_length": np.array([80.0]),
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 0.0,
    }

    # Test with smooth signal (sinusoidal) — better agreement
    cin_smooth = np.sin(np.linspace(0, 4 * np.pi, n_days)) + 2.0
    cout_fast = infiltration_to_extraction(cin=cin_smooth, **fast_kwargs)
    cout_exact = diffusion_exact(cin=cin_smooth, **exact_kwargs)

    assert len(cout_fast) == len(cout_exact) == n_days

    both_valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    assert np.sum(both_valid) > 50

    assert np.all(cout_fast[~np.isnan(cout_fast)] >= -1e-10)
    assert np.all(cout_exact[~np.isnan(cout_exact)] >= -1e-10)

    # Smooth signal: agreement to the interpolation floor of the flow_out=None path.
    assert_allclose(cout_fast[both_valid], cout_exact[both_valid], atol=1e-6)

    # Step function: sharpest front, still within the interpolation floor.
    cin_step = np.zeros(n_days)
    cin_step[30:] = 1.0
    cout_fast_step = infiltration_to_extraction(cin=cin_step, **fast_kwargs)
    cout_exact_step = diffusion_exact(cin=cin_step, **exact_kwargs)

    both_valid_step = ~np.isnan(cout_fast_step) & ~np.isnan(cout_exact_step)
    assert_allclose(cout_fast_step[both_valid_step], cout_exact_step[both_valid_step], atol=1e-6)


def _breakthrough_centre_idx(*, flow, dt, j_in, retarded_volume):
    """Index of the breakthrough front for a pulse injected at bin ``j_in``.

    The parcel reaches the outlet when the cumulative through-flow volume
    since injection equals ``retarded_volume = R * V_pore``. Using cumulative
    flow (rather than ``R*V/Q``) locates the front correctly even when ``flow``
    contains zero-flow plateaus, where wall-clock travel time stretches.
    """
    cum = np.cumsum(np.asarray(flow, dtype=float) * np.asarray(dt, dtype=float))
    return int(np.searchsorted(cum - cum[j_in], retarded_volume))


@pytest.mark.parametrize("zero_flow", [False, True], ids=["constant_flow", "zero_flow"])
@pytest.mark.parametrize("retardation", [1.0, 2.7])
@pytest.mark.parametrize("n_bins", [1, 5, 25], ids=["single_pv", "gamma_5bin", "gamma_25bin"])
def test_diffusion_fast_vs_diffusion_parity_sweep(zero_flow, retardation, n_bins):
    """Cross-module parity at a well-resolved breakthrough front, including zero-flow.

    Both modules report the Kreft-Zuber flux concentration, so they agree to machine
    precision when the cout grid equals the flow grid. This sweep exercises that
    equivalence over the dimensions the single-PV equivalence test does not: explicit
    zero-flow plateaus (``flow=[100,100,0,0,100,100]`` tiled), ``R in {1.0, 2.7}``, and
    single / 5-bin / 25-bin gamma pore-volume distributions. ``D_m=0`` (pure mechanical
    dispersion, ``Pe = L/alpha_L = 25``) keeps it in the machine-precision regime while
    the front spans several output bins. The comparison is windowed from the breakthrough
    onset to ``+4*R*sigma`` to exclude the left-edge warm-start region, where the two
    modules' boundary conventions differ harmlessly.
    """
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    flow_active = 100.0
    flow = np.tile([100.0, 100.0, 0.0, 0.0, 100.0, 100.0], n // 6 + 1)[:n] if zero_flow else np.full(n, flow_active)

    streamline_length = 50.0
    d_m = 0.0
    alpha_l = 2.0  # Pe = L / alpha_L = 25 (D_m = 0)
    mean_apv = 1800.0
    std_apv = 240.0

    # Both modules report C_F here (D_m=0), so the front agrees to machine precision;
    # gate at a tight peak-relative tolerance. sigma_V in output-bin-index units carries R
    # (the front spreads as R*sigma_x), so the +4*sigma window includes R to keep the full
    # front in-window at R=2.7.
    peak_rel_tol = 1e-11
    sigma_idx = retardation * (mean_apv / streamline_length) * np.sqrt(2.0 * alpha_l * streamline_length) / flow_active

    j_in = 12
    cin = np.zeros(n)
    cin[10:15] = 1.0  # cin[0] = 0 so the warm-start constant is 0

    with warn_module.catch_warnings():
        warn_module.simplefilter("ignore")
        if n_bins == 1:
            cout_fast = infiltration_to_extraction(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([mean_apv]),
                mean_streamline_length=streamline_length,
                mean_molecular_diffusivity=d_m,
                mean_longitudinal_dispersivity=alpha_l,
                retardation_factor=retardation,
            )
            cout_exact = diffusion_exact(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([mean_apv]),
                streamline_length=np.array([streamline_length]),
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
                retardation_factor=retardation,
            )
        else:
            cout_fast = gamma_infiltration_to_extraction(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                mean=mean_apv,
                std=std_apv,
                n_bins=n_bins,
                mean_streamline_length=streamline_length,
                mean_molecular_diffusivity=d_m,
                mean_longitudinal_dispersivity=alpha_l,
                retardation_factor=retardation,
            )
            cout_exact = diffusion_gamma_i2e(
                cin=cin,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                mean=mean_apv,
                std=std_apv,
                n_bins=n_bins,
                streamline_length=streamline_length,
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
                retardation_factor=retardation,
            )

    dt = np.diff(tedges) / pd.Timedelta("1D")
    centre = _breakthrough_centre_idx(flow=flow, dt=dt, j_in=j_in, retarded_volume=retardation * mean_apv)
    i_arr = np.arange(n)
    both_valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    window = both_valid & (i_arr >= centre - 1) & (i_arr <= centre + int(4 * sigma_idx))
    assert window.sum() >= 10
    # Machine-precision equivalence across the resolved front (both modules compute C_F).
    peak = np.nanmax(cout_exact[window])
    assert_allclose(cout_fast[window], cout_exact[window], atol=float(peak_rel_tol * peak), rtol=0)


# =============================================================================
# Tests for flow_out (extraction flow defining the output-grid volumes)
# =============================================================================
def test_flow_out_constant_input():
    """Constant cin with non-uniform bins + flow_out produces constant output (exact)."""
    tedges = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-20"),
        pd.Timestamp("2020-02-21"),
        pd.Timestamp("2020-12-16"),
    ])
    cin = np.array([10.0, 10.0, 10.0])
    flow = np.array([100.0, 100.0, 100.0])
    cout_tedges = pd.date_range("2020-01-01", periods=351, freq="D")
    flow_out = np.full(len(cout_tedges) - 1, 100.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.05,
        mean_longitudinal_dispersivity=1.0,
        flow_out=flow_out,
    )

    valid = ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert_allclose(cout[valid], 10.0, atol=1e-13)


def test_flow_out_constant_input_variable_flow():
    """Constant cin with non-uniform bins + variable flow_out produces constant output (exact)."""
    tedges = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-20"),
        pd.Timestamp("2020-02-21"),
        pd.Timestamp("2020-12-16"),
    ])
    cin = np.array([10.0, 10.0, 10.0])
    flow = np.array([80.0, 150.0, 120.0])
    cout_tedges = pd.date_range("2020-01-01", periods=351, freq="D")
    t = np.arange(350)
    flow_out = 100.0 + 30.0 * np.sin(2 * np.pi * t / 365)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.05,
        mean_longitudinal_dispersivity=1.0,
        flow_out=flow_out,
    )

    valid = ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert_allclose(cout[valid], 10.0, atol=1e-13)


def test_flow_out_zero_diffusion_matches_advection():
    """Zero diffusion with a consistent flow_out on a non-uniform grid matches advection.

    ``flow_out`` legitimately defines the extraction-side cout volumes, so it
    must be the infiltration ``flow`` rebinned onto the daily cout grid. The
    rebin matrix ``ov`` maps each cout bin to the fraction of each flow bin it
    overlaps, so ``ov @ flow`` is the cout-grid through-flow that is mass-
    consistent with the infiltration record. With zero diffusion the closed-form
    operator then reduces to pure advection. The residual (~1e-12) is the
    floating-point gap between the closed-form volume bookkeeping and the
    advection module's, not a physical difference.
    """
    tedges = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-20"),
        pd.Timestamp("2020-02-21"),
        pd.Timestamp("2020-12-16"),
    ])
    cin = np.array([0.0, 100.0, 0.0])
    flow = np.array([80.0, 150.0, 120.0])
    cout_tedges = pd.date_range("2020-01-01", periods=351, freq="D")
    # flow_out = flow rebinned onto the daily cout grid (mass-consistent).
    ov = partial_isin(bin_edges_in=tedges_to_days(cout_tedges), bin_edges_out=tedges_to_days(tedges))
    flow_out = ov @ flow

    cout_fast = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.0,
        mean_longitudinal_dispersivity=0.0,
        flow_out=flow_out,
    )

    cout_adv = advection_i2e(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
    )

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_adv)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_adv[valid], atol=1e-11)


def test_flow_out_uniform_bins_matches_default():
    """With uniform bins + constant flow, flow_out produces same result as default (exact)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin = np.sin(np.linspace(0, 4 * np.pi, n_days)) + 2.0
    flow_out = flow.copy()

    common_kwargs = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
        "mean_streamline_length": 80.0,
        "mean_molecular_diffusivity": 0.03,
        "mean_longitudinal_dispersivity": 0.0,
    }

    cout_default = infiltration_to_extraction(**common_kwargs)
    cout_flow_out = infiltration_to_extraction(**common_kwargs, flow_out=flow_out)

    valid = ~np.isnan(cout_default) & ~np.isnan(cout_flow_out)
    assert np.sum(valid) > 50
    assert_allclose(cout_flow_out[valid], cout_default[valid], atol=1e-13)


def test_flow_out_output_bounded_by_input():
    """Output concentration bounded by input range on non-uniform grid (convex combination)."""
    tedges = pd.DatetimeIndex([
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-02-20"),
        pd.Timestamp("2020-02-21"),
        pd.Timestamp("2020-12-16"),
    ])
    cin = np.array([0.0, 100.0, 0.0])
    flow = np.array([80.0, 150.0, 120.0])
    cout_tedges = pd.date_range("2020-01-01", periods=351, freq="D")
    t = np.arange(len(cout_tedges) - 1)
    flow_out = 100.0 + 30.0 * np.sin(2 * np.pi * t / 365)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        mean_streamline_length=80.0,
        mean_molecular_diffusivity=0.05,
        mean_longitudinal_dispersivity=1.0,
        flow_out=flow_out,
    )

    valid = ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert np.all(cout[valid] >= np.min(cin) - 1e-14)
    assert np.all(cout[valid] <= np.max(cin) + 1e-14)


def test_flow_out_nonuniform_bins_pulse():
    """Compressed non-uniform bins with flow_out gives identical result to uncompressed uniform bins."""
    n_days = 350
    tedges_full = pd.date_range("2019-12-31", periods=n_days + 1, freq="D")
    flow_full = np.full(n_days, 100.0)
    cin_full = np.zeros(n_days)
    cin_full[50] = 100.0

    # Compress to non-uniform bins (same as notebook 05)
    cin_itedges = np.flatnonzero(np.diff(cin_full, prepend=1.0, append=1.0))
    flow_itedges = np.flatnonzero(np.diff(flow_full, prepend=1.0, append=1.0))
    itedges = np.unique(np.concatenate([cin_itedges, flow_itedges]))
    tedges_compressed = tedges_full[itedges]
    cin_compressed = cin_full[itedges[:-1]]
    flow_compressed = flow_full[itedges[:-1]]

    cout_tedges = tedges_full.copy()
    flow_out = np.full(len(cout_tedges) - 1, 100.0)

    kwargs = {
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([10000.0]),
        "mean_streamline_length": 100.0,
        "mean_molecular_diffusivity": 1e-4,
        "mean_longitudinal_dispersivity": 1.0,
        "retardation_factor": 2.0,
        "flow_out": flow_out,
    }

    # Compressed (non-uniform bins)
    cout_compressed = infiltration_to_extraction(
        cin=cin_compressed, flow=flow_compressed, tedges=tedges_compressed, **kwargs
    )

    # Uncompressed (uniform daily bins)
    cout_uncompressed = infiltration_to_extraction(cin=cin_full, flow=flow_full, tedges=tedges_full, **kwargs)

    both_valid = ~np.isnan(cout_compressed) & ~np.isnan(cout_uncompressed)
    assert np.sum(both_valid) > 50
    assert_allclose(cout_compressed[both_valid], cout_uncompressed[both_valid], atol=0.0)


def test_flow_out_validation_wrong_length():
    """flow_out with wrong length raises ValueError."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    cin = np.full(200, 10.0)
    flow_out = np.full(100, 100.0)  # Wrong length

    with pytest.raises(ValueError, match="flow_out must have length"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            mean_streamline_length=80.0,
            mean_molecular_diffusivity=0.03,
            mean_longitudinal_dispersivity=0.0,
            flow_out=flow_out,
        )


def test_flow_out_validation_nan():
    """flow_out with NaN raises ValueError."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    cin = np.full(200, 10.0)
    flow_out = np.full(200, 100.0)
    flow_out[50] = np.nan

    with pytest.raises(ValueError, match="flow_out contains NaN"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            mean_streamline_length=80.0,
            mean_molecular_diffusivity=0.03,
            mean_longitudinal_dispersivity=0.0,
            flow_out=flow_out,
        )


def test_flow_out_validation_negative():
    """flow_out with negative values raises ValueError."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    cin = np.full(200, 10.0)
    flow_out = np.full(200, 100.0)
    flow_out[50] = -1.0

    with pytest.raises(ValueError, match="flow_out must be non-negative"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            mean_streamline_length=80.0,
            mean_molecular_diffusivity=0.03,
            mean_longitudinal_dispersivity=0.0,
            flow_out=flow_out,
        )


def test_extraction_to_infiltration_flow_out_round_trip():
    """Round-trip with flow_out recovers original signal (machine precision).

    Uses pore volume 517 m^3 for the same full-rank reason as ``test_round_trip``:
    at Q=100 the closed-form forward matrix is rank-deficient for a band of
    residence times, and 5.17 days sits in a clean full-rank plateau so the
    Tikhonov inverse is fully data-determined.
    """
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin_original = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0
    flow_out = flow.copy()

    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([517.0]),
        "mean_streamline_length": 80.0,
        "mean_molecular_diffusivity": 0.01,
        "mean_longitudinal_dispersivity": 0.0,
        "flow_out": flow_out,
    }

    cout = infiltration_to_extraction(cin=cin_original, **kwargs)
    cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)
    cin_recovered = extraction_to_infiltration(cout=cout_clean, **kwargs)

    valid = ~np.isnan(cin_recovered) & ~np.isnan(cout)
    assert np.sum(valid) > 50
    assert_allclose(cin_recovered[valid], cin_original[valid], atol=1e-11)


# =============================================================================
# Tests for gamma_extraction_to_infiltration
# =============================================================================


class TestGammaExtractionToInfiltrationFast:
    """Tests for gamma_extraction_to_infiltration in the diffusion_fast module.

    Verifies that the gamma convenience wrapper correctly delegates to
    extraction_to_infiltration and that the combined gamma + fast Gaussian
    diffusion deconvolution produces physically correct results.
    """

    @pytest.fixture
    def gamma_setup(self):
        """Create test data with long time series for gamma transport."""
        tedges = pd.date_range(start="2020-01-01", end="2020-08-01", freq="D")
        cout_tedges = pd.date_range(start="2020-01-10", end="2020-07-01", freq="D")

        cin = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(tedges) - 1) / 30.0)
        flow = np.full(len(tedges) - 1, 100.0)

        return {
            "cin": cin,
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": cout_tedges,
            "mean": 500.0,
            "std": 100.0,
            "n_bins": 20,
            "mean_streamline_length": 80.0,
            "mean_molecular_diffusivity": 0.03,
            "mean_longitudinal_dispersivity": 0.0,
        }

    def test_zero_cout_gives_zero_cin(self):
        """Zero extraction concentration must produce zero infiltration."""
        tedges, cout_tedges, flow = _make_transport_data(n_days=200)
        cout = np.zeros(len(cout_tedges) - 1)

        cin = gamma_extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            mean=500.0,
            std=100.0,
            n_bins=10,
            mean_streamline_length=80.0,
            mean_molecular_diffusivity=0.03,
            mean_longitudinal_dispersivity=0.0,
        )

        valid = ~np.isnan(cin)
        assert np.sum(valid) > 20
        assert_allclose(cin[valid], 0.0, atol=1e-14)

    def test_constant_cout_gives_constant_cin(self, gamma_setup):
        """Constant extraction concentration must produce constant infiltration."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.full(n_cout, 7.0)

        cin = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            n_bins=gamma_setup["n_bins"],
            mean_streamline_length=gamma_setup["mean_streamline_length"],
            mean_molecular_diffusivity=gamma_setup["mean_molecular_diffusivity"],
            mean_longitudinal_dispersivity=gamma_setup["mean_longitudinal_dispersivity"],
        )

        valid = ~np.isnan(cin)
        well_supported = valid & (cin > 1.0)
        assert np.sum(well_supported) > 30
        assert_allclose(cin[well_supported], 7.0, atol=1e-12)

    def test_roundtrip_recovers_signal(self):
        """Forward then inverse roundtrip recovers the original signal.

        Machine precision is achieved in the interior. Boundary bins are
        excluded because the forward matrix has incomplete column coverage
        there: the first/last input bins lack enough output observations for
        the Tikhonov inverse to fully constrain them.
        """
        n_days = 500
        tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
        cout_tedges = tedges.copy()
        cin = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0
        flow = np.full(n_days, 100.0)

        diffusion_kwargs = {
            "mean": 501.3,
            "std": 100.0,
            "n_bins": 20,
            "mean_streamline_length": 80.0,
            "mean_molecular_diffusivity": 0.03,
            "mean_longitudinal_dispersivity": 0.0,
        }

        cout = gamma_infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout_clean,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        margin = 50
        valid = ~np.isnan(cin_recovered) & ~np.isnan(cout)
        valid[:margin] = False
        valid[-margin:] = False
        assert np.sum(valid) > 300
        assert_allclose(cin_recovered[valid], cin[valid], atol=1e-12)

    def test_roundtrip_with_retardation(self):
        """Roundtrip with retardation factor recovers signal.

        Machine precision is NOT achievable here. Retardation stretches the
        effective residence times (V*R/flow), spreading the 20 gamma bins
        over a wider range of shifts. Adjacent bins then produce nearly
        collinear advection columns, making the forward matrix ill-conditioned
        (smallest significant singular value ~3e-4 for R=2.7, giving ~3000x
        noise amplification). The Tikhonov regularization adds bias
        proportional to regularization_strength / s_i^2 for each mode,
        which is O(1e-7) for the ill-conditioned modes with the default
        regularization_strength=1e-10.

        The strict-validity advection (NaN until every streamtube has broken
        through, NaN on zero-flow cout bins) makes the boundary spin-up region
        slightly wider than the longest single-streamtube residence time, so
        the margin must skip this region for a clean inverse comparison.
        """
        retardation = 2.7

        n_days = 800
        tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
        cout_tedges = tedges.copy()
        cin = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(n_days) / 30.0)
        flow = np.full(n_days, 100.0)

        # mean=489.2 minimizes near-integer residence times for R=2.7
        diffusion_kwargs = {
            "mean": 489.2,
            "std": 100.0,
            "n_bins": 20,
            "mean_streamline_length": 80.0,
            "mean_molecular_diffusivity": 0.03,
            "mean_longitudinal_dispersivity": 0.0,
            "retardation_factor": retardation,
        }

        cout = gamma_infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout_clean,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        # Skip the inverse-boundary regions where the regularization bias bleeds
        # in from the mean-filled cout cells; the well-conditioned interior
        # recovers within atol=1e-3.
        margin = 150
        valid = ~np.isnan(cin_recovered) & ~np.isnan(cout)
        valid[:margin] = False
        valid[-margin:] = False
        assert np.sum(valid) > 400
        assert_allclose(cin_recovered[valid], cin[valid], atol=1e-3)

    def test_alpha_beta_matches_mean_std(self, gamma_setup):
        """Alpha/beta parameterization gives identical result to mean/std."""
        alpha, beta = mean_std_loc_to_alpha_beta(mean=gamma_setup["mean"], std=gamma_setup["std"])
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.full(n_cout, 5.0)
        cout[30:50] = 10.0

        common_kwargs = {
            "cout": cout,
            "flow": gamma_setup["flow"],
            "tedges": gamma_setup["tedges"],
            "cout_tedges": gamma_setup["cout_tedges"],
            "n_bins": gamma_setup["n_bins"],
            "mean_streamline_length": gamma_setup["mean_streamline_length"],
            "mean_molecular_diffusivity": gamma_setup["mean_molecular_diffusivity"],
            "mean_longitudinal_dispersivity": gamma_setup["mean_longitudinal_dispersivity"],
        }

        cin_mean_std = gamma_extraction_to_infiltration(
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            **common_kwargs,
        )
        cin_alpha_beta = gamma_extraction_to_infiltration(
            alpha=alpha,
            beta=beta,
            **common_kwargs,
        )

        valid = ~np.isnan(cin_mean_std) & ~np.isnan(cin_alpha_beta)
        assert np.sum(valid) > 50
        assert_allclose(cin_mean_std[valid], cin_alpha_beta[valid], atol=0.0)

    def test_matches_explicit_extraction_to_infiltration(self, gamma_setup):
        """Gamma wrapper must produce identical result to explicit call."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.full(n_cout, 5.0)
        cout[20:40] = 8.0

        bins = gamma_utils.bins(
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            n_bins=gamma_setup["n_bins"],
        )

        cin_gamma = gamma_extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            n_bins=gamma_setup["n_bins"],
            mean_streamline_length=gamma_setup["mean_streamline_length"],
            mean_molecular_diffusivity=gamma_setup["mean_molecular_diffusivity"],
            mean_longitudinal_dispersivity=gamma_setup["mean_longitudinal_dispersivity"],
        )

        cin_explicit = extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            aquifer_pore_volumes=bins["expected_values"],
            mean_streamline_length=gamma_setup["mean_streamline_length"],
            mean_molecular_diffusivity=gamma_setup["mean_molecular_diffusivity"],
            mean_longitudinal_dispersivity=gamma_setup["mean_longitudinal_dispersivity"],
        )

        valid = ~np.isnan(cin_gamma) & ~np.isnan(cin_explicit)
        assert np.sum(valid) > 50
        assert_allclose(cin_gamma[valid], cin_explicit[valid], atol=0.0)

    def test_loc_zero_matches_legacy(self, gamma_setup):
        """With loc=0 the gamma wrapper must exactly match the legacy (mean, std) call."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.full(n_cout, 5.0)
        cout[20:40] = 8.0

        common_kwargs = {
            "cout": cout,
            "flow": gamma_setup["flow"],
            "tedges": gamma_setup["tedges"],
            "cout_tedges": gamma_setup["cout_tedges"],
            "mean": gamma_setup["mean"],
            "std": gamma_setup["std"],
            "n_bins": gamma_setup["n_bins"],
            "mean_streamline_length": gamma_setup["mean_streamline_length"],
            "mean_molecular_diffusivity": gamma_setup["mean_molecular_diffusivity"],
            "mean_longitudinal_dispersivity": gamma_setup["mean_longitudinal_dispersivity"],
        }

        cin_default = gamma_extraction_to_infiltration(**common_kwargs)
        cin_loc_zero = gamma_extraction_to_infiltration(loc=0.0, **common_kwargs)
        np.testing.assert_array_equal(cin_default, cin_loc_zero)

    def test_roundtrip_with_loc(self):
        """Forward->reverse roundtrip with loc > 0 recovers signal in interior."""
        n_days = 500
        tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
        cout_tedges = tedges.copy()
        cin = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0
        flow = np.full(n_days, 100.0)

        # loc shifts the entire gamma by 200 m³. Keep the excess (mean - loc)
        # comparable to the legacy roundtrip test: excess mean ~ 300.
        diffusion_kwargs = {
            "mean": 501.3,
            "std": 100.0,
            "loc": 200.0,
            "n_bins": 20,
            "mean_streamline_length": 80.0,
            "mean_molecular_diffusivity": 0.03,
            "mean_longitudinal_dispersivity": 0.0,
        }

        cout = gamma_infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )
        cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout_clean,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        margin = 50
        valid = ~np.isnan(cin_recovered) & ~np.isnan(cout)
        valid[:margin] = False
        valid[-margin:] = False
        assert np.sum(valid) > 300
        assert_allclose(cin_recovered[valid], cin[valid], atol=1e-10)

    def test_roundtrip_with_dispersivity(self):
        """Roundtrip with non-zero dispersivity recovers signal.

        Near-machine precision is achieved in the interior. The Gaussian
        kernel from dispersivity adds slight numerical diffusion that lifts
        the error floor to ~1e-11 (vs ~1e-13 without dispersivity).
        """
        n_days = 500
        tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
        cout_tedges = tedges.copy()
        cin = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0
        flow = np.full(n_days, 100.0)

        diffusion_kwargs = {
            "mean": 501.3,
            "std": 100.0,
            "n_bins": 20,
            "mean_streamline_length": 80.0,
            "mean_molecular_diffusivity": 0.03,
            "mean_longitudinal_dispersivity": 1.0,
        }

        cout = gamma_infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout_clean,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        margin = 50
        valid = ~np.isnan(cin_recovered) & ~np.isnan(cout)
        valid[:margin] = False
        valid[-margin:] = False
        assert np.sum(valid) > 300
        assert_allclose(cin_recovered[valid], cin[valid], atol=1e-10)

    def test_roundtrip_with_flow_out(self):
        """Roundtrip with flow_out parameter recovers signal.

        Machine precision is achieved in the interior, same as the basic
        roundtrip. The flow_out path applies diffusion on the output grid
        but uses the same Tikhonov inversion.
        """
        n_days = 500
        tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
        cout_tedges = tedges.copy()
        cin = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0
        flow = np.full(n_days, 100.0)
        flow_out = flow.copy()

        diffusion_kwargs = {
            "mean": 501.3,
            "std": 100.0,
            "n_bins": 20,
            "mean_streamline_length": 80.0,
            "mean_molecular_diffusivity": 0.03,
            "mean_longitudinal_dispersivity": 0.0,
            "flow_out": flow_out,
        }

        cout = gamma_infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        cout_clean = np.where(np.isnan(cout), np.nanmean(cout), cout)

        cin_recovered = gamma_extraction_to_infiltration(
            cout=cout_clean,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **diffusion_kwargs,
        )

        margin = 50
        valid = ~np.isnan(cin_recovered) & ~np.isnan(cout)
        valid[:margin] = False
        valid[-margin:] = False
        assert np.sum(valid) > 300
        assert_allclose(cin_recovered[valid], cin[valid], atol=1e-12)


# =============================================================================
# Negative-flow rejection and zero-flow invariance
# =============================================================================


def test_infiltration_to_extraction_rejects_negative_flow():
    """Negative flow must raise ValueError with the standard message."""
    tedges = pd.date_range(start="2020-01-01", periods=11, freq="D")
    cin = np.ones(10)
    flow = np.full(10, 100.0)
    flow[5] = -1.0

    with pytest.raises(ValueError, match="flow must be non-negative"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([500.0]),
            mean_streamline_length=80.0,
            mean_molecular_diffusivity=0.03,
            mean_longitudinal_dispersivity=0.0,
        )


def test_infiltration_to_extraction_accepts_zero_flow_without_warnings():
    """flow == 0 is accepted; no runtime warnings emitted."""
    tedges = pd.date_range(start="2020-01-01", periods=201, freq="D")
    cin = np.ones(200)
    flow = np.full(200, 100.0)
    flow[100] = 0.0

    with warn_module.catch_warnings():
        warn_module.simplefilter("error", RuntimeWarning)
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([500.0]),
            mean_streamline_length=80.0,
            mean_molecular_diffusivity=0.03,
            mean_longitudinal_dispersivity=0.0,
        )


def test_infiltration_to_extraction_zero_flow_insertion_invariance():
    """Insert a zero-flow bin mid-series: preserves NaN count and value pattern."""
    n = 200
    k = 100
    tedges_base = pd.date_range(start="2020-01-01", periods=n + 1, freq="D")
    cin_base = 10.0 + np.sin(np.linspace(0, 4 * np.pi, n))
    flow_base = np.full(n, 100.0)

    tedges_mod = pd.date_range(start="2020-01-01", periods=n + 2, freq="D")
    cin_mod = np.insert(cin_base, k, cin_base[k - 1])
    flow_mod = np.insert(flow_base, k, 0.0)

    common_kwargs = {
        "aquifer_pore_volumes": np.array([500.0]),
        "mean_streamline_length": 80.0,
        "mean_molecular_diffusivity": 0.03,
        "mean_longitudinal_dispersivity": 5.0,
    }

    cout_base = infiltration_to_extraction(
        cin=cin_base, flow=flow_base, tedges=tedges_base, cout_tedges=tedges_base, **common_kwargs
    )
    cout_mod = infiltration_to_extraction(
        cin=cin_mod, flow=flow_mod, tedges=tedges_mod, cout_tedges=tedges_mod, **common_kwargs
    )

    cout_mod_stripped = np.delete(cout_mod, k)
    assert np.sum(np.isnan(cout_mod_stripped)) == np.sum(np.isnan(cout_base))

    valid = ~np.isnan(cout_base) & ~np.isnan(cout_mod_stripped)
    # Near the inserted bin the diffusion kernel reaches across the gap, so
    # values drift by up to a few percent of the input amplitude; farther out
    # the signal converges back to the baseline.
    assert_allclose(cout_mod_stripped[valid], cout_base[valid], atol=5e-2)


# =============================================================================
# Column mass conservation under variable flow + dispersion
#
# The forward operator reports the Kreft-Zuber flux concentration C_F in closed
# form. Bear's formula with the K-Z boundary-flux correction satisfies the
# variable-coefficient advection-dispersion equation exactly, so the
# volume-weighted column-sum invariant
#
#     sum_i W[i, j] * Q_out[i] * dt_out[i] = Q_in[j] * dt_in[j]
#
# holds at machine precision in the interior for *every* dispersion mix (pure
# D_m, pure alpha_L, or mixed) -- there is no variable-sigma discretisation
# residual. The only departure from unity is physical tail-exit near the domain
# boundary: a column's breakthrough is an erf curve whose far tail runs past the
# last cout bin, so the last few columns lose the mass carried by that
# un-captured tail. That residual grows sharply in the final columns and is
# excluded by restricting the check to an interior window.
# =============================================================================


def _build_w_via_probes(call_fn, n):
    """Build the (n_out, n) coefficient matrix by feeding canonical basis vectors."""
    cout0 = call_fn(np.zeros(n))
    n_out = len(cout0)
    w = np.zeros((n_out, n))
    for j in range(n):
        ej = np.zeros(n)
        ej[j] = 1.0
        cout = call_fn(ej)
        w[:, j] = np.where(np.isnan(cout), 0.0, cout)
    return w


@pytest.mark.parametrize("seed", [1, 3, 42])
@pytest.mark.parametrize(
    ("d_m", "alpha_l"),
    [(1.0, 0.0), (0.0, 0.3), (1.0, 0.3)],
)
def test_column_mass_conservation_variable_flow_dispersion_flow_out(d_m, alpha_l, seed):
    """Volume-weighted column sums of W match infiltrated volume to machine
    precision under variable Q, with ``flow_out`` provided (extraction flow on the
    output grid).

    The closed-form C_F satisfies the variable-coefficient ADE exactly (Bear +
    Kreft-Zuber), so the invariant
    ``sum_i W[i, j] * V_out[i] = V_in[j]`` holds to the floating-point floor for
    *every* dispersion mix -- there is no variable-sigma_V discretisation error.
    The interior window ``[15:35]`` excludes the final columns where the
    breakthrough's erf tail exits the domain past the last cout bin (a physical
    boundary loss, not a numerical one).
    """
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    rng = np.random.default_rng(seed)
    flow = 100.0 * np.exp(rng.normal(0.0, 0.3, n))
    flow_out = flow.copy()

    def call(cin_arr):
        with warn_module.catch_warnings():
            warn_module.simplefilter("ignore")
            return infiltration_to_extraction(
                cin=cin_arr,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([1000.0]),
                mean_streamline_length=100.0,
                mean_molecular_diffusivity=d_m,
                mean_longitudinal_dispersivity=alpha_l,
                flow_out=flow_out,
            )

    w = _build_w_via_probes(call, n=n)

    dt = (np.diff(tedges) / pd.Timedelta("1D")).astype(float)
    v_out = flow_out * dt
    v_in_interior = (flow * dt)[15:35]
    mass_out_per_cin = (v_out[:, None] * w).sum(axis=0)
    ratios = mass_out_per_cin[15:35] / v_in_interior
    assert_allclose(ratios, 1.0, atol=1e-11, rtol=0)


@pytest.mark.parametrize("seed", [1, 3, 42])
@pytest.mark.parametrize(
    ("d_m", "alpha_l"),
    [(1.0, 0.0), (0.0, 0.3), (1.0, 0.3)],
)
def test_column_mass_conservation_variable_flow_dispersion_default(d_m, alpha_l, seed):
    """Same machine-precision invariant as the ``flow_out`` variant, but with
    ``flow_out`` omitted (cout volumes interpolated from the flow grid).

    Holds to the floating-point floor for the same reason: the closed-form C_F
    is mass-conservative for the variable-coefficient ADE regardless of the
    dispersion mix. The interior window ``[15:35]`` excludes the boundary
    columns whose breakthrough erf tail exits the domain.
    """
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    rng = np.random.default_rng(seed)
    flow = 100.0 * np.exp(rng.normal(0.0, 0.3, n))

    def call(cin_arr):
        with warn_module.catch_warnings():
            warn_module.simplefilter("ignore")
            return infiltration_to_extraction(
                cin=cin_arr,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([1000.0]),
                mean_streamline_length=100.0,
                mean_molecular_diffusivity=d_m,
                mean_longitudinal_dispersivity=alpha_l,
            )

    w = _build_w_via_probes(call, n=n)

    dt = (np.diff(tedges) / pd.Timedelta("1D")).astype(float)
    # For matched cout/tedges and no flow_out, cout flow == flow.
    v_out = flow * dt
    v_in_interior = (flow * dt)[15:35]
    mass_out_per_cin = (v_out[:, None] * w).sum(axis=0)
    ratios = mass_out_per_cin[15:35] / v_in_interior
    assert_allclose(ratios, 1.0, atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    ("d_m", "alpha_l"),
    [(1.0, 0.0), (0.0, 0.3), (1.0, 0.3)],
)
def test_column_mass_conservation_constant_flow_control(d_m, alpha_l):
    """Constant Q is the machine-precision control for column mass conservation.

    With ``flow`` constant the through-flow volume is uniform, so the only
    residual the volume-weighted column sum can carry is the closed-form C_F's
    own floating-point arithmetic. Any residual above ~1e-13 in the interior is
    a structural bug in the operator. The window stays at ``[15:40]`` because
    the constant-flow tail-exit is itself at machine epsilon.
    """
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    flow = np.full(n, 100.0)

    def call(cin_arr):
        with warn_module.catch_warnings():
            warn_module.simplefilter("ignore")
            return infiltration_to_extraction(
                cin=cin_arr,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([1000.0]),
                mean_streamline_length=100.0,
                mean_molecular_diffusivity=d_m,
                mean_longitudinal_dispersivity=alpha_l,
                flow_out=flow,
            )

    w = _build_w_via_probes(call, n=n)
    dt = (np.diff(tedges) / pd.Timedelta("1D")).astype(float)
    v_out = flow * dt
    v_in_interior = (flow * dt)[15:40]
    ratios = (v_out[:, None] * w).sum(axis=0)[15:40] / v_in_interior
    assert_allclose(ratios, 1.0, atol=1e-13, rtol=0)


@pytest.mark.parametrize("seed", [3, 42])
@pytest.mark.parametrize(
    ("d_m", "alpha_l"),
    [(1.0, 0.0), (0.0, 0.3), (1.0, 0.3)],
)
def test_column_mass_conservation_multipv(d_m, alpha_l, seed):
    """Mass conservation under variable Q for a multi-pore-volume APVD, to
    machine precision.

    A spread of pore volumes ``[600, 1000, 1400]`` superposes three C_F
    breakthroughs of different residence times. Each is individually
    mass-conservative for the variable-coefficient ADE, so their flow-weighted
    superposition conserves the column mass exactly. The window is narrowed to
    ``[20:35]``: the widest streamtube (1400 m^3 ~ 14-day residence) pushes its
    erf tail past the last cout bin sooner than the single-PV cases, so the
    tail-exit boundary loss reaches a few columns further in.
    """
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    rng = np.random.default_rng(seed)
    flow = 100.0 * np.exp(rng.normal(0.0, 0.3, n))

    def call(cin_arr):
        with warn_module.catch_warnings():
            warn_module.simplefilter("ignore")
            return infiltration_to_extraction(
                cin=cin_arr,
                flow=flow,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=np.array([600.0, 1000.0, 1400.0]),
                mean_streamline_length=100.0,
                mean_molecular_diffusivity=d_m,
                mean_longitudinal_dispersivity=alpha_l,
                flow_out=flow,
            )

    w = _build_w_via_probes(call, n=n)
    dt = (np.diff(tedges) / pd.Timedelta("1D")).astype(float)
    v_out = flow * dt
    v_in_interior = (flow * dt)[20:35]
    ratios = (v_out[:, None] * w).sum(axis=0)[20:35] / v_in_interior
    assert_allclose(ratios, 1.0, atol=1e-11, rtol=0)


@pytest.mark.parametrize(
    "mechanism", ["zero_flow_plateau", "cout_precedes_cin"], ids=["zero_flow_plateau", "cout_precedes_cin"]
)
def test_edge_pulse_with_uninformed_region_conserves_mass_without_warning(mechanism):
    """A pulse with an un-informed output region conserves mass and emits no warning.

    Two grid configurations leave some output bins un-informed -- no input bin breaks
    through them: an early zero-flow plateau, and an output window that pre-dates the
    earliest input (``spinup=None``). The closed-form coefficient rows for those bins sum
    to ~0, so they are marked NaN via the ``(total_coeff < _EPSILON_COEFF_SUM) |
    ~valid_cout_bins`` mask in :func:`infiltration_to_extraction`. This is a regression
    gate on two properties:

    - The un-informed (NaN) bins must not corrupt the informed breakthrough. The captured
      pulse conserves mass exactly under ``nansum``, and the breakthrough peak and its
      rising limb stay finite (no NaN bleeds into the informed front).
    - The validity handling emits no spurious ``RuntimeWarning`` -- the test runs under
      ``simplefilter("error")``.

    The pulse sits two bins past the start of the informed region so its full breakthrough
    is captured (matched cout/tedges resolution, ``flow_out == flow`` so V_out == V_in).
    ``D_m=0.5`` widens the breakthrough enough that a regression letting NaN leak from the
    un-informed bins would reach the breakthrough rising limb.
    """
    n = 120
    v_pore = 500.0  # breakthrough centre ~ 5 bins past injection at Q=100
    streamline_length = 80.0
    d_m = 0.5

    if mechanism == "zero_flow_plateau":
        # Early zero-flow plateau leaves the leading output bins un-informed.
        tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
        cout_tedges = tedges
        flow = np.full(n, 100.0)
        flow[:20] = 0.0
        j_pulse = 22  # two bins after flow resumes
        spinup = "constant"
    else:
        # Output window starts 30 days before the input window; spinup=None so
        # there is no warm-start echo to fill the leading output bins.
        tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D") + pd.Timedelta(days=30)
        cout_tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
        flow = np.full(n, 100.0)
        j_pulse = 2  # two bins into the input window
        spinup = None

    flow_out = flow.copy()
    cin = np.zeros(n)
    cin[j_pulse] = 1.0

    with warn_module.catch_warnings():
        warn_module.simplefilter("error")
        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([v_pore]),
            mean_streamline_length=streamline_length,
            mean_molecular_diffusivity=d_m,
            mean_longitudinal_dispersivity=0.0,
            flow_out=flow_out,
            spinup=spinup,
        )

    # The un-informed core is genuinely present (some bins are NaN).
    assert np.any(np.isnan(cout)), "test input must exercise the un-informed (NaN-masked) region"
    # Exact mass conservation: the full pulse breakthrough lies in the informed
    # region (matched cout/tedges resolution, flow_out == flow -> V_out == V_in).
    assert_allclose(np.nansum(cout), cin.sum(), atol=1e-12, rtol=0)
    # The breakthrough peak and its rising limb must be finite: no NaN from the
    # un-informed bins bleeds into the informed front.
    peak_bin = int(np.nanargmax(cout))
    assert not np.any(np.isnan(cout[peak_bin - 2 : peak_bin + 1]))


# =============================================================================
# Closed-form equivalence to the slow ``gwtransport.diffusion`` module
#
# diffusion_fast computes the Kreft-Zuber flux concentration C_F in closed form
# -- the *same* physical quantity as gwtransport.diffusion, not an
# approximation. When the cout grid aligns with the flow grid (flow_out=flow on
# a matched daily grid) the two modules agree to the floating-point floor of the
# closed-form evaluation for a sharp delta/step input -- the most demanding case,
# because all spectral content is present.
#
# The agreement is machine precision (~1e-13) for R=1 (any D_m/alpha_L mix) and for pure
# mechanical dispersion (D_m=0) at any R, where the closed form equals the slow module's
# C_F exactly. For R != 1 AND D_m > 0 the closed form applies a retardation correction to
# the Kreft-Zuber flux coefficient (the per-edge mechanism would otherwise weight the
# molecular-diffusion flux by R); the residual is then the trapezoidal O(dx^2)
# discretisation of that correction, ~1e-5 of the breakthrough peak for a sharp delta.
# =============================================================================


@pytest.mark.parametrize(
    ("retardation", "d_m", "alpha_l", "atol"),
    [
        # R=1: the flux-vs-resident advective correction is identical in both
        # discretisations, so molecular diffusion and dispersion both reproduce
        # the slow module exactly (measured residual ~1e-14; atol set 100x over).
        (1.0, 2.0, 0.0, 1e-12),
        (1.0, 0.0, 2.0, 1e-12),
        (1.0, 1.0, 2.0, 1e-12),
        # Pure mechanical dispersion (D_m=0): no tau-dependent term, so R does
        # not perturb the agreement -- exact at R=2.7 too.
        (2.7, 0.0, 2.0, 1e-12),
        # R != 1 with D_m > 0: the retardation correction holds the agreement to the
        # trapezoidal discretisation floor (~2e-5 of peak here), not machine precision.
        (2.7, 2.0, 0.0, 1e-4),
        (2.7, 1.0, 2.0, 1e-4),
        (5.0, 1.0, 0.0, 1e-4),
    ],
)
def test_delta_input_single_pv_matches_diffusion_exact_constant_flow(retardation, d_m, alpha_l, atol):
    """Single delta pulse reproduces ``gwtransport.diffusion`` to machine precision.

    A delta input is the hardest equivalence case: its breakthrough is the full
    closed-form kernel, so any difference in the C_F formula -- the sigma_V
    magnitude, the retardation coupling in ``tau = R*V/Q``, or the K-Z
    flux-vs-resident correction -- shows up directly. Passing ``flow_out=flow``
    on a matched daily grid aligns the cout volumes with the flow grid, which is
    the condition under which the closed form equals the slow module's
    16-point Gauss-Legendre quadrature of the same C_F.

    The parametrisation spans both variance terms (``2*D_m*tau`` and ``2*alpha_L*L``)
    and the retardation factor. The exact legs (R=1, or D_m=0) hold to machine
    precision; the ``R != 1 with D_m > 0`` legs hold to the trapezoidal discretisation
    floor of the retardation correction (``atol`` set per case).
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    flow_rate = 100.0
    flow = np.full(n, flow_rate)
    v_pore = 1000.0
    streamline_length = 30.0

    j_in = 60
    cin = np.zeros(n)
    cin[j_in] = 1.0

    cout_fast = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([v_pore]),
        mean_streamline_length=streamline_length,
        mean_molecular_diffusivity=d_m,
        mean_longitudinal_dispersivity=alpha_l,
        retardation_factor=retardation,
        flow_out=flow,
    )
    cout_exact = diffusion_exact(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([v_pore]),
        streamline_length=np.array([streamline_length]),
        molecular_diffusivity=d_m,
        longitudinal_dispersivity=alpha_l,
        retardation_factor=retardation,
    )

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    assert np.sum(valid) > 50
    # Both modules report C_F; the residual is the floating-point floor of the closed form
    # vs the 16-point quadrature (exact legs), or the trapezoidal retardation-correction
    # floor for the R != 1 with D_m > 0 legs.
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=atol, rtol=0)


def test_step_input_single_pv_variable_flow_matches_diffusion_exact():
    """Step input under variable flow reproduces ``gwtransport.diffusion`` exactly.

    Extends the constant-flow delta equivalence to a time-varying flow record
    with a sharp step input. ``flow_out`` is the infiltration ``flow`` itself
    (the cout grid equals the flow grid here, so ``flow`` is already the
    cout-grid through-flow). The closed form and the slow quadrature evaluate
    the same C_F at the same variable-velocity geometry, so they agree to the
    floating-point floor across the whole breakthrough.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    t = np.arange(n)
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 30))
    cin = np.zeros(n)
    cin[50:] = 1.0

    common = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([1000.0]),
    }
    cout_fast = infiltration_to_extraction(
        **common,
        mean_streamline_length=30.0,
        mean_molecular_diffusivity=0.5,
        mean_longitudinal_dispersivity=1.0,
        flow_out=flow,
    )
    cout_exact = diffusion_exact(
        **common,
        streamline_length=np.array([30.0]),
        molecular_diffusivity=0.5,
        longitudinal_dispersivity=1.0,
    )

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-11, rtol=0)


def test_step_input_retardation_diffusion_variable_flow_matches_diffusion_exact():
    """Retardation correction reproduces ``gwtransport.diffusion`` under VARIABLE flow.

    The R != 1 with D_m > 0 regime triggers the per-cout-bin ``velocity`` retardation
    correction in :func:`gwtransport.diffusion_fast._flux_breakthrough_fraction` (the
    per-edge mechanism would otherwise weight the molecular-diffusion flux by R). Under
    constant flow the correction's velocity is a single value, so a bug that used a mean
    velocity instead of the per-bin ``q_cout`` would be invisible. Variable flow makes
    ``q_cout`` vary across the front, so this is the regression guard for that term: a
    mean-velocity (or dropped) correction drives the front error to >1e-2, far above the
    1e-3 gate (the correct trapezoidal discretisation floor here is ~7e-5 of peak).
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    t = np.arange(n)
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 30))
    cin = np.zeros(n)
    cin[50:] = 1.0

    common = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([1000.0]),
    }
    cout_fast = infiltration_to_extraction(
        **common,
        mean_streamline_length=30.0,
        mean_molecular_diffusivity=2.0,
        mean_longitudinal_dispersivity=0.0,
        retardation_factor=2.7,
        flow_out=flow,
    )
    cout_exact = diffusion_exact(
        **common,
        streamline_length=np.array([30.0]),
        molecular_diffusivity=2.0,
        longitudinal_dispersivity=0.0,
        retardation_factor=2.7,
    )

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-3, rtol=0)


def test_step_input_gamma_multipv_matches_diffusion_gamma():
    """Gamma multi-PV step input reproduces ``gwtransport.diffusion`` gamma exactly.

    The gamma wrappers discretise the aquifer-pore-volume distribution into
    ``n_bins`` streamtubes and superpose their C_F breakthroughs. Because each
    streamtube's closed form equals the slow module's quadrature, the
    superposition matches too -- this pins the bin placement, the per-bin
    weights, and the C_F kernel jointly. ``R=1`` keeps every streamtube in the
    machine-precision regime (the R != 1 with D_m > 0 carve-out would otherwise
    apply per streamtube).
    """
    n = 300
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    flow = np.full(n, 100.0)
    cin = np.zeros(n)
    cin[50:] = 1.0

    common = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "mean": 1000.0,
        "std": 200.0,
        "n_bins": 10,
    }
    cout_fast = gamma_infiltration_to_extraction(
        **common,
        mean_streamline_length=30.0,
        mean_molecular_diffusivity=1.0,
        mean_longitudinal_dispersivity=1.0,
    )
    cout_exact = diffusion_gamma_i2e(
        **common,
        streamline_length=30.0,
        molecular_diffusivity=1.0,
        longitudinal_dispersivity=1.0,
    )

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-11, rtol=0)


def _good_diffusion_fast_inputs():
    """Baseline known-valid inputs for _validate_inputs (mirrors the diffusion.py snapshot)."""
    return {
        "cin_or_cout": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "flow": np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
        "tedges": pd.date_range("2023-01-01", periods=6, freq="D"),
        "cout_tedges": pd.date_range("2023-01-01", periods=6, freq="D"),
        "aquifer_pore_volumes": np.array([300.0]),
        "mean_streamline_length": 80.0,
        "mean_molecular_diffusivity": 0.01,
        "mean_longitudinal_dispersivity": 0.5,
        "retardation_factor": 1.0,
        "is_forward": True,
    }


@pytest.mark.parametrize(
    ("mutate", "match_regex"),
    [
        (
            lambda k: {**k, "tedges": k["tedges"][:-1]},
            r"tedges must have one more element than cin",
        ),
        (
            lambda k: {**k, "flow": k["flow"][:-1]},
            r"tedges must have one more element than flow",
        ),
        (
            lambda k: {**k, "mean_molecular_diffusivity": -0.1},
            r"mean_molecular_diffusivity must be non-negative",
        ),
        (
            lambda k: {**k, "mean_longitudinal_dispersivity": -0.1},
            r"mean_longitudinal_dispersivity must be non-negative",
        ),
        (
            lambda k: {**k, "cin_or_cout": np.array([1.0, np.nan, 3.0, 4.0, 5.0])},
            r"cin contains NaN values",
        ),
        (
            lambda k: {**k, "flow": np.array([100.0, np.nan, 100.0, 100.0, 100.0])},
            r"flow contains NaN values",
        ),
        (
            lambda k: {**k, "flow": np.array([100.0, -50.0, 100.0, 100.0, 100.0])},
            r"flow must be non-negative \(negative flow not supported\)",
        ),
        (
            lambda k: {**k, "aquifer_pore_volumes": np.array([0.0])},
            r"aquifer_pore_volumes must be positive",
        ),
        (
            lambda k: {**k, "mean_streamline_length": 0.0},
            r"mean_streamline_length must be positive",
        ),
        (
            lambda k: {**k, "retardation_factor": 0.5},
            r"retardation_factor must be >= 1\.0",
        ),
    ],
)
def test_validate_diffusion_fast_inputs_raises_on_bad_input(mutate, match_regex):
    """Each ValueError branch of _validate_inputs raises with the historical message.

    Mirrors the snapshot pattern in test_diffusion.py:test_validate_diffusion_inputs_raises_on_bad_input.
    Pins the diffusion_fast validation surface (with the new R>=1 contract from Group 8).
    """
    bad = mutate(_good_diffusion_fast_inputs())
    with pytest.raises(ValueError, match=match_regex):
        _validate_inputs(**bad)


def _call_diffusion_fast_entry(entry, *, retardation_factor, tedges, cout_tedges, flow):
    """Invoke one of the four public entry points with R < 1 inputs.

    Centralises the forward/reverse + single-PV/gamma parametrization so the
    R<1 rejection (and other shared-validator) tests cover every public
    surface, mirroring the paired forward/reverse convention.
    """
    n = len(flow)
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "mean_streamline_length": 80.0,
        "mean_molecular_diffusivity": 0.01,
        "mean_longitudinal_dispersivity": 0.5,
        "retardation_factor": retardation_factor,
    }
    if entry == "infiltration_to_extraction":
        return infiltration_to_extraction(cin=np.ones(n), aquifer_pore_volumes=np.array([500.0]), **common)
    if entry == "extraction_to_infiltration":
        return extraction_to_infiltration(cout=np.ones(n), aquifer_pore_volumes=np.array([500.0]), **common)
    if entry == "gamma_infiltration_to_extraction":
        return gamma_infiltration_to_extraction(cin=np.ones(n), mean=500.0, std=100.0, n_bins=5, **common)
    if entry == "gamma_extraction_to_infiltration":
        return gamma_extraction_to_infiltration(cout=np.ones(n), mean=500.0, std=100.0, n_bins=5, **common)
    msg = f"unknown entry {entry}"
    raise AssertionError(msg)


@pytest.mark.parametrize(
    "entry",
    [
        "infiltration_to_extraction",
        "extraction_to_infiltration",
        "gamma_infiltration_to_extraction",
        "gamma_extraction_to_infiltration",
    ],
)
@pytest.mark.parametrize("bad_r", [0.0, 0.5, 0.99])
def test_diffusion_fast_rejects_retardation_below_one(entry, bad_r):
    """Every public surface (forward + reverse, single-PV + gamma) raises on R < 1.

    The forward and reverse entry points share ``_validate_inputs``; this pins
    that the R>=1 contract fires identically on both directions (mirrors
    diffusion.py's paired forward/reverse validator coverage), not only on the
    forward ``infiltration_to_extraction`` path that was previously tested.
    """
    tedges, cout_tedges, flow = _make_transport_data(n_days=50)
    with pytest.raises(ValueError, match=r"retardation_factor must be >= 1\.0"):
        _call_diffusion_fast_entry(entry, retardation_factor=bad_r, tedges=tedges, cout_tedges=cout_tedges, flow=flow)


@pytest.mark.parametrize(
    "entry",
    ["extraction_to_infiltration", "gamma_extraction_to_infiltration"],
)
def test_diffusion_fast_reverse_rejects_mismatched_cout_tedges(entry):
    """Reverse surfaces raise when ``cout_tedges`` length does not match ``cout``.

    Exercises the reverse-only ``len(cout_tedges) != len(cout) + 1`` branch of
    :func:`gwtransport.diffusion_fast._validate_inputs`, which the forward
    direction never reaches. ``cout`` has ``len(flow)`` entries, so a
    ``cout_tedges`` of length ``len(flow)`` (one too short) is invalid.
    """
    tedges, _cout_tedges, flow = _make_transport_data(n_days=50)
    # cout_tedges one element too short for the supplied cout (len == len(flow)).
    bad_cout_tedges = tedges[:-1]

    with pytest.raises(ValueError, match=r"cout_tedges must have one more element than cout"):
        _call_diffusion_fast_entry(entry, retardation_factor=1.0, tedges=tedges, cout_tedges=bad_cout_tedges, flow=flow)


def test_streamline_length_zero_rejected():
    """L=0 has no streamtube; raise rather than risk div-by-zero downstream."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=50)
    cin = np.ones(len(flow))

    with pytest.raises(ValueError, match=r"mean_streamline_length must be positive"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            mean_streamline_length=0.0,
            mean_molecular_diffusivity=0.01,
            mean_longitudinal_dispersivity=0.5,
        )
