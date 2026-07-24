import warnings as warn_module

import numpy as np
import pandas as pd
import pytest
from _oracles import partial_isin  # ty: ignore[unresolved-import]  # tests/src on path via conftest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import erf, erfc

from gwtransport import gamma as gamma_utils
from gwtransport._diffusion_shared import (
    _DT_FLOOR,
    _breakthrough_antideriv,
    _cout_cumulative_volume,
    _extend_tedges_flag,
    _validate_inputs,
)
from gwtransport._time import dt_to_days, tedges_to_days
from gwtransport.advection import infiltration_to_extraction as advection_i2e
from gwtransport.diffusion import extraction_to_infiltration as diffusion_exact_reverse
from gwtransport.diffusion import gamma_infiltration_to_extraction as diffusion_gamma_i2e
from gwtransport.diffusion import infiltration_to_extraction as diffusion_exact
from gwtransport.diffusion_fast import (
    _closed_form_coeff_matrix,
    extraction_to_infiltration,
    gamma_extraction_to_infiltration,
    gamma_infiltration_to_extraction,
    infiltration_to_extraction,
)
from gwtransport.gamma import mean_std_loc_to_alpha_beta
from gwtransport.utils import cumulative_flow_volume

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
        streamline_length=80.0,
        molecular_diffusivity=0.0,
        longitudinal_dispersivity=0.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
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
    """A pulse through a 3-streamtube APVD conserves mass exactly and stays non-negative.

    On a matched cout/flow grid with ``flow_out == flow`` the volume-weighted superposition of the
    three streamtubes' C_F breakthroughs is mass-conservative, so the full pulse breakthrough
    (which lies inside the informed region here) recovers the infiltrated mass to machine precision
    under ``nansum`` -- a real invariant, not just a sign check.
    """
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
        streamline_length=100.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
        flow_out=flow,
    )

    assert len(cout) == n_days
    valid = ~np.isnan(cout)
    assert np.any(valid)
    assert np.all(cout[valid] >= -1e-10)
    # Exact mass conservation: flow_out == flow on a matched grid -> V_out == V_in, and the full
    # pulse breakthrough lies in the informed region, so nansum(cout) recovers the infiltrated mass.
    assert_allclose(np.nansum(cout), cin.sum(), atol=1e-12, rtol=0.0)


def test_infiltration_to_extraction_cout_tedges_different_resolution():
    """Test with coarser output grid (flow_out required since cout_tedges != tedges)."""
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
        flow_out=np.full(len(cout_tedges) - 1, 100.0),
        aquifer_pore_volumes=np.array([500.0]),
        streamline_length=80.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
    )

    assert len(cout) == len(cout_tedges) - 1
    valid = ~np.isnan(cout)
    assert np.sum(valid) > 0
    assert_allclose(cout[valid], 5.0, atol=1e-12)


@pytest.mark.parametrize("retardation_factor", [1.0, 2.0, 3.5])
def test_infiltration_to_extraction_with_retardation(retardation_factor):
    """Retardation shifts the breakthrough centre by exactly ``R * V / Q``.

    A step ``cin`` injected at bin 50 breaks through at the bin where the cumulative through-flow
    volume since injection equals ``R * V_pore`` -- i.e. ``50 + R * V / Q`` bins under constant
    flow. Asserting the half-rise crossing lands there pins that the retardation factor enters the
    travel-volume ``tau = R * V / Q`` correctly, an R-dependent invariant a sign check would miss.
    """
    n_days = 300
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = tedges.copy()
    cin = np.zeros(n_days)
    step_bin = 50
    cin[step_bin:] = 10.0
    flow_rate = 100.0
    flow = np.full(n_days, flow_rate)
    v_pore = 1000.0

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([v_pore]),
        streamline_length=80.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
        retardation_factor=retardation_factor,
        flow_out=flow,
    )

    assert len(cout) == n_days
    valid = ~np.isnan(cout)
    assert np.any(valid)
    # Half-rise crossing of the step (final level 10.0 -> half 5.0) lands at 50 + R*V/Q.
    crossing = int(np.argmax(valid & (cout >= 5.0)))
    expected = step_bin + retardation_factor * v_pore / flow_rate
    assert abs(crossing - expected) <= 1.0


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
        streamline_length=80.0,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=0.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
    )

    valid = ~np.isnan(cout)
    assert np.any(valid)
    # Under variable flow, constant preservation holds because the combined
    # matrix (W_adv @ G) has row sums ≤ 1 for valid rows, and cin is constant.
    assert_allclose(cout[valid], 10.0, atol=1e-13)


def test_infiltration_to_extraction_tz_aware_matches_naive():
    """tz-aware (UTC) tedges run without error and match the tz-naive result exactly.

    The example data is tz-aware UTC by design, and the ``spinup="constant"`` warm-start
    extends ``tedges`` by 100 years on each side. That extension must preserve the input
    timezone -- a tz-stripping/mixing extension raises "Mixed timezones detected". With the
    same wall-clock edges, the tz-aware run is identical to the tz-naive run to machine
    precision (NaN spin-up mask included).
    """
    tedges_naive, _, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin = np.sin(np.linspace(0, 4 * np.pi, n_days)) + 2.0
    tedges_aware = tedges_naive.tz_localize("UTC")
    assert tedges_aware.tz is not None

    kwargs = {
        "cin": cin,
        "flow": flow,
        "aquifer_pore_volumes": np.array([400.0, 500.0]),
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.03,
        "longitudinal_dispersivity": 1.0,
        "spinup": "constant",
    }
    cout_naive = infiltration_to_extraction(tedges=tedges_naive, cout_tedges=tedges_naive, **kwargs)
    cout_aware = infiltration_to_extraction(tedges=tedges_aware, cout_tedges=tedges_aware, **kwargs)

    assert np.any(~np.isnan(cout_naive))
    assert_allclose(np.isnan(cout_aware), np.isnan(cout_naive))
    valid = ~np.isnan(cout_naive)
    assert_allclose(cout_aware[valid], cout_naive[valid], rtol=0.0, atol=0.0)


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
        streamline_length=80.0,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.0,
        longitudinal_dispersivity=0.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.0,
        longitudinal_dispersivity=0.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
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
        "streamline_length": 100.0,
        "molecular_diffusivity": 1e-4,
        "longitudinal_dispersivity": 1.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
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
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.01,
        "longitudinal_dispersivity": 0.0,
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
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 0.0,
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

    # Smooth signal: agreement to the interpolation floor of the flow_out=None path
    # (measured ~1.3e-8; gate at 1e-7 to keep margin while catching a regression).
    assert_allclose(cout_fast[both_valid], cout_exact[both_valid], atol=1e-7)

    # Step function: sharpest front, still within the interpolation floor (measured ~2.1e-7).
    cin_step = np.zeros(n_days)
    cin_step[30:] = 1.0
    cout_fast_step = infiltration_to_extraction(cin=cin_step, **fast_kwargs)
    cout_exact_step = diffusion_exact(cin=cin_step, **exact_kwargs)

    both_valid_step = ~np.isnan(cout_fast_step) & ~np.isnan(cout_exact_step)
    assert_allclose(cout_fast_step[both_valid_step], cout_exact_step[both_valid_step], atol=5e-7)


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
                streamline_length=streamline_length,
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
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
                streamline_length=streamline_length,
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
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
        streamline_length=80.0,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.0,
        longitudinal_dispersivity=0.0,
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
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.03,
        "longitudinal_dispersivity": 0.0,
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
        streamline_length=80.0,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
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
        "streamline_length": 100.0,
        "molecular_diffusivity": 1e-4,
        "longitudinal_dispersivity": 1.0,
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
            streamline_length=80.0,
            molecular_diffusivity=0.03,
            longitudinal_dispersivity=0.0,
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
            streamline_length=80.0,
            molecular_diffusivity=0.03,
            longitudinal_dispersivity=0.0,
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
            streamline_length=80.0,
            molecular_diffusivity=0.03,
            longitudinal_dispersivity=0.0,
            flow_out=flow_out,
        )


def test_flow_out_required_when_cout_tedges_differ():
    """Omitting flow_out is rejected when cout_tedges differs from tedges.

    The extraction flow on a distinct output grid is not implied by the input ``flow``
    (it would have to be guessed by interpolation), so the module requires it. When
    cout_tedges equals tedges, flow_out may be omitted (it equals ``flow``).
    """
    n_days = 200
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = pd.date_range("2020-01-01", periods=n_days // 7 + 1, freq="7D")
    flow = np.full(n_days, 100.0)
    cin = np.full(n_days, 5.0)
    common = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "aquifer_pore_volumes": np.array([500.0]),
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.03,
        "longitudinal_dispersivity": 0.0,
    }
    with pytest.raises(ValueError, match="flow_out is required when cout_tedges differs from tedges"):
        infiltration_to_extraction(cout_tedges=cout_tedges, **common)
    # cout_tedges == tedges: flow_out may be omitted.
    out = infiltration_to_extraction(cout_tedges=tedges, **common)
    assert len(out) == n_days


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
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.01,
        "longitudinal_dispersivity": 0.0,
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
            # cout grid differs from tedges, so the extraction flow on the cout grid is
            # required (constant 100 here, matching the input flow).
            "flow_out": np.full(len(cout_tedges) - 1, 100.0),
            "mean": 500.0,
            "std": 100.0,
            "n_bins": 20,
            "streamline_length": 80.0,
            "molecular_diffusivity": 0.03,
            "longitudinal_dispersivity": 0.0,
        }

    def test_zero_cout_gives_zero_cin(self):
        """Zero extraction concentration deconvolves to zero -- no additive bias, zeros preserved.

        For a linear Tikhonov solve a zero right-hand side trivially yields a zero solution, so this
        is not a deconvolution-accuracy check (the constant-nonzero-cout test carries that signal).
        Its purpose is narrower: the gamma wrapper's normalization and NaN-masking path must not
        inject any additive offset or leak a NaN into a defined bin -- an all-zero ``cout`` stays
        exactly zero wherever the inverse is defined.
        """
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
            streamline_length=80.0,
            molecular_diffusivity=0.03,
            longitudinal_dispersivity=0.0,
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
            flow_out=gamma_setup["flow_out"],
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            n_bins=gamma_setup["n_bins"],
            streamline_length=gamma_setup["streamline_length"],
            molecular_diffusivity=gamma_setup["molecular_diffusivity"],
            longitudinal_dispersivity=gamma_setup["longitudinal_dispersivity"],
        )

        # Use a column-support index margin (as the roundtrip tests do) rather than a
        # value-based ``cin > 1.0`` filter, so an erroneously-small interior recovery fails
        # instead of being silently masked out.
        margin = 50
        interior = ~np.isnan(cin)
        interior[:margin] = False
        interior[-margin:] = False
        assert np.sum(interior) > 30
        assert_allclose(cin[interior], 7.0, atol=1e-12)

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
            "streamline_length": 80.0,
            "molecular_diffusivity": 0.03,
            "longitudinal_dispersivity": 0.0,
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
            "streamline_length": 80.0,
            "molecular_diffusivity": 0.03,
            "longitudinal_dispersivity": 0.0,
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
            "flow_out": gamma_setup["flow_out"],
            "n_bins": gamma_setup["n_bins"],
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": gamma_setup["molecular_diffusivity"],
            "longitudinal_dispersivity": gamma_setup["longitudinal_dispersivity"],
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
            flow_out=gamma_setup["flow_out"],
            mean=gamma_setup["mean"],
            std=gamma_setup["std"],
            n_bins=gamma_setup["n_bins"],
            streamline_length=gamma_setup["streamline_length"],
            molecular_diffusivity=gamma_setup["molecular_diffusivity"],
            longitudinal_dispersivity=gamma_setup["longitudinal_dispersivity"],
        )

        cin_explicit = extraction_to_infiltration(
            cout=cout,
            flow=gamma_setup["flow"],
            tedges=gamma_setup["tedges"],
            cout_tedges=gamma_setup["cout_tedges"],
            flow_out=gamma_setup["flow_out"],
            aquifer_pore_volumes=bins["expected_values"],
            streamline_length=gamma_setup["streamline_length"],
            molecular_diffusivity=gamma_setup["molecular_diffusivity"],
            longitudinal_dispersivity=gamma_setup["longitudinal_dispersivity"],
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
            "flow_out": gamma_setup["flow_out"],
            "mean": gamma_setup["mean"],
            "std": gamma_setup["std"],
            "n_bins": gamma_setup["n_bins"],
            "streamline_length": gamma_setup["streamline_length"],
            "molecular_diffusivity": gamma_setup["molecular_diffusivity"],
            "longitudinal_dispersivity": gamma_setup["longitudinal_dispersivity"],
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
            "streamline_length": 80.0,
            "molecular_diffusivity": 0.03,
            "longitudinal_dispersivity": 0.0,
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
            "streamline_length": 80.0,
            "molecular_diffusivity": 0.03,
            "longitudinal_dispersivity": 1.0,
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
            "streamline_length": 80.0,
            "molecular_diffusivity": 0.03,
            "longitudinal_dispersivity": 0.0,
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
            streamline_length=80.0,
            molecular_diffusivity=0.03,
            longitudinal_dispersivity=0.0,
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
            streamline_length=80.0,
            molecular_diffusivity=0.03,
            longitudinal_dispersivity=0.0,
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
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.03,
        "longitudinal_dispersivity": 5.0,
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
                streamline_length=100.0,
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
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
                streamline_length=100.0,
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
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
                streamline_length=100.0,
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
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
                streamline_length=100.0,
                molecular_diffusivity=d_m,
                longitudinal_dispersivity=alpha_l,
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
            streamline_length=streamline_length,
            molecular_diffusivity=d_m,
            longitudinal_dispersivity=0.0,
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
# molecular-diffusion flux by R); the correction's bin-average of the Gaussian density is
# itself evaluated in closed form (erf/erfcx), so this regime is machine precision too.
# =============================================================================


@pytest.mark.parametrize(
    ("retardation", "d_m", "alpha_l"),
    [
        # R=1: the flux-vs-resident advective correction is identical in both
        # discretisations, so molecular diffusion and dispersion both reproduce the slow module.
        (1.0, 2.0, 0.0),
        (1.0, 0.0, 2.0),
        (1.0, 1.0, 2.0),
        # Pure mechanical dispersion (D_m=0): no tau-dependent term, so R does not perturb it.
        (2.7, 0.0, 2.0),
        # R != 1 with D_m > 0: the closed-form retardation correction (exact bin-average of
        # the Gaussian density via erf/erfcx) restores machine precision here too.
        (2.7, 2.0, 0.0),
        (2.7, 1.0, 2.0),
        (5.0, 1.0, 0.0),
    ],
)
def test_delta_input_single_pv_matches_diffusion_exact_constant_flow(retardation, d_m, alpha_l):
    """Single delta pulse reproduces ``gwtransport.diffusion`` to machine precision.

    A delta input is the hardest equivalence case: its breakthrough is the full
    closed-form kernel, so any difference in the C_F formula -- the sigma_V
    magnitude, the retardation coupling in ``tau = R*V/Q``, or the K-Z
    flux-vs-resident correction -- shows up directly. Passing ``flow_out=flow``
    on a matched daily grid aligns the cout volumes with the flow grid, which is
    the condition under which the closed form equals the slow module's
    16-point Gauss-Legendre quadrature of the same C_F.

    The parametrisation spans both variance terms (``2*D_m*tau`` and ``2*alpha_L*L``)
    and the retardation factor. Every leg holds to machine precision: the R=1 and D_m=0
    legs need no retardation correction, and the ``R != 1 with D_m > 0`` legs evaluate the
    correction's Gaussian-density bin-average in closed form (measured residual ~3e-14).
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
        streamline_length=streamline_length,
        molecular_diffusivity=d_m,
        longitudinal_dispersivity=alpha_l,
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
    # vs the 16-point quadrature -- machine precision for every leg, including R != 1 with
    # D_m > 0 now that the retardation correction is evaluated in closed form.
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-12, rtol=0)


def test_delta_response_matches_hand_coded_erfc_known_answer():
    r"""The forward delta-response column equals a hand-coded ``0.5*erfc`` breakthrough.

    Independent known-answer anchor (F086): every other equivalence test compares
    ``diffusion_fast`` against ``gwtransport.diffusion``, which shares the *same* physical
    formula (``C_R = 0.5*erfc((L - xi) / (2*sqrt(D_t)))`` with ``D_t = D_m*tau + alpha_L*xi``)
    and the *same* per-edge antiderivative -- only the evaluation differs (closed form vs
    16-point quadrature). A conceptual error in that shared formula would be invisible to a
    fast-vs-slow comparison. This test reconstructs the breakthrough from scratch using
    :func:`scipy.special.erfc` / :func:`scipy.special.erf` (not the module's
    ``_breakthrough_antideriv``, not the slow module), so it pins the formula itself.

    Construction (constant flow, single PV, ``R=1``, matched grid). A unit delta injected at
    ``cin`` bin ``j0`` produces the forward operator column ``W[:, j0]``. For a single
    streamtube the bin-averaged flux concentration over cout bin ``i`` is the per-edge
    antiderivative difference ``(J(x_hi; D_t_hi) - J(x_lo; D_t_lo)) / dx`` evaluated at cin
    edges ``j0`` and ``j0+1`` and differenced, where

        ``J(x; D_t) = 0.5*x + 0.5*x*erf(x / (2*sqrt(D_t))) + sqrt(D_t/pi)*exp(-x**2/(4*D_t))``

    is the antiderivative of the resident concentration ``0.5*erfc(-x / (2*sqrt(D_t)))``
    (here ``x = xi - L`` is the breakthrough coordinate, so ``L - xi = -x``). Freezing ``D_t``
    per cout edge is what turns the resident antiderivative into the Kreft-Zuber *flux*
    concentration. The breakthrough coordinate, ``tau``, ``xi`` and ``D_t`` are rebuilt from
    the same 100-year warm-start extension and cumulative-volume geometry the module uses.

    Two independent checks: (1) the hand-coded ``J`` antiderivative matches a direct
    :func:`scipy.integrate.quad` of ``0.5*erfc(-x/(2*sqrt(D_t)))`` at frozen ``D_t`` (anchors
    that ``J`` integrates the claimed resident concentration), and (2) the full delta-response
    column ``W[:, j0]`` reproduces the hand-coded breakthrough to machine precision (~1e-14),
    spanning the partial-breakthrough front rather than only the saturated 0/1 tails.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow_rate = 100.0
    flow = np.full(n, flow_rate)
    v_pore = 1000.0
    length = 30.0
    d_m = 0.5
    alpha_l = 1.0
    j0 = 60

    def antideriv_erfc(x, d_t):
        """Hand-coded antiderivative of the resident concentration 0.5*erfc(-x/(2*sqrt(D_t)))."""
        two_s = 2.0 * np.sqrt(d_t)
        return 0.5 * x + 0.5 * x * erf(x / two_s) + np.sqrt(d_t / np.pi) * np.exp(-((x / two_s) ** 2))

    # Check (1): J integrates 0.5*erfc(-x/(2 sqrt D_t)) at frozen D_t (independent of the module).
    d_t_frozen, x_lo_chk, x_hi_chk = 7.3, -2.0, 5.0
    quad_binavg, _ = quad(lambda xx: 0.5 * erfc(-xx / (2.0 * np.sqrt(d_t_frozen))), x_lo_chk, x_hi_chk)
    hand_binavg = (antideriv_erfc(x_hi_chk, d_t_frozen) - antideriv_erfc(x_lo_chk, d_t_frozen)) / (x_hi_chk - x_lo_chk)
    assert_allclose(hand_binavg, quad_binavg / (x_hi_chk - x_lo_chk), atol=1e-12, rtol=0.0)

    cin = np.zeros(n)
    cin[j0] = 1.0
    w_col = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=np.array([v_pore]),
        streamline_length=length,
        molecular_diffusivity=d_m,
        longitudinal_dispersivity=alpha_l,
        retardation_factor=1.0,
        flow_out=flow,
    )

    # Rebuild the module's geometry from scratch: 100-year warm-start extension, cumulative
    # through-flow volume, and per-(cout edge, cin edge) breakthrough coordinate / tau / xi / D_t.
    pad = pd.Timedelta(days=36500)
    work_tedges = tedges[:1] - pad
    work_tedges = work_tedges.append(tedges[1:-1]).append(tedges[-1:] + pad)
    work_days = tedges_to_days(work_tedges)
    cout_days = tedges_to_days(tedges, ref=work_tedges[0])
    v_edges = cumulative_flow_volume(flow, dt_to_days(work_tedges))  # at work edges == cout edges (matched grid)
    r_vpv = 1.0 * v_pore

    def frac_for_cin_edge(j):
        """Bin-averaged resident-antiderivative fraction over every cout bin for cin edge j."""
        x = (v_edges - v_edges[j] - r_vpv) * length / r_vpv  # breakthrough coord at each cout edge
        tau = np.maximum(cout_days - work_days[j], 0.0)
        d_t = np.maximum(d_m * tau + alpha_l * np.maximum(x + length, 0.0), _DT_FLOOR)
        j_anti = antideriv_erfc(x, d_t)
        x_lo, x_hi = x[:-1], x[1:]
        dx = x_hi - x_lo
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(dx > 0.0, (j_anti[1:] - j_anti[:-1]) / dx, 0.0)

    w_col_expected = frac_for_cin_edge(j0) - frac_for_cin_edge(j0 + 1)

    valid = ~np.isnan(w_col)
    # The partial-breakthrough front (strictly between the 0 and 1 saturation plateaus) must be
    # present, so the test genuinely exercises the erfc breakthrough rather than just saturation.
    front = valid & (w_col > 1e-6)
    assert front.sum() > 10
    assert_allclose(w_col[valid], w_col_expected[valid], atol=1e-13, rtol=0.0)


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
        streamline_length=30.0,
        molecular_diffusivity=0.5,
        longitudinal_dispersivity=1.0,
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
    correction in :func:`gwtransport.diffusion_fast._pv_band_values` (the
    per-edge mechanism would otherwise weight the molecular-diffusion flux by R). Under
    constant flow the correction's velocity is a single value, so a bug that used a mean
    velocity instead of the per-bin ``q_cout`` would be invisible. Variable flow makes
    ``q_cout`` vary across the front, so this is the regression guard for that term: a
    mean-velocity (or dropped) correction drives the front error to >1e-2, far above the
    1e-11 gate. The correction's Gaussian-density bin-average is evaluated in closed form
    (erf/erfcx), so the agreement is machine precision (~6e-13) under variable flow too.
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
        streamline_length=30.0,
        molecular_diffusivity=2.0,
        longitudinal_dispersivity=0.0,
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
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-11, rtol=0)


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
        streamline_length=30.0,
        molecular_diffusivity=1.0,
        longitudinal_dispersivity=1.0,
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


def test_per_pv_arrays_match_diffusion_exact():
    """Per-streamtube ``streamline_length`` / ``molecular_diffusivity`` /
    ``longitudinal_dispersivity`` arrays reproduce ``gwtransport.diffusion`` exactly.

    The closed form indexes each of the three dispersion parameters per pore
    volume (``..._length[i_pv]`` etc.); the slow module does the same in its
    quadrature. Here every streamtube carries a *distinct* L, D_m and alpha_L,
    so a bug that collapsed any of them to a single value (first entry, or a
    mean) would shift that tube's breakthrough and break the match. ``R=1`` keeps
    all tubes in the machine-precision regime, and ``flow_out=flow`` aligns the
    cout grid to the flow grid, so the two implementations agree to the
    floating-point floor across the whole superposed breakthrough.
    """
    n = 250
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    t = np.arange(n)
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 30))
    cin = np.zeros(n)
    cin[50:] = 1.0

    # Three genuinely heterogeneous streamtubes (longer paths drain the larger pores).
    common = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([400.0, 700.0, 1100.0]),
        "streamline_length": np.array([60.0, 90.0, 130.0]),
        "molecular_diffusivity": np.array([0.2, 0.5, 1.0]),
        "longitudinal_dispersivity": np.array([0.5, 1.5, 3.0]),
    }
    cout_fast = infiltration_to_extraction(**common, flow_out=flow)
    cout_exact = diffusion_exact(**common)

    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-11, rtol=0)


def test_per_pv_arrays_reverse_matches_diffusion_exact():
    """Per-streamtube arrays through the REVERSE direction match ``gwtransport.diffusion``.

    The reverse function carries its own ``_broadcast_to_pore_volumes`` call-sites
    (distinct from the forward function's), so a regression that collapsed a per-PV
    parameter to its first entry on only the reverse path would slip past
    :func:`test_per_pv_arrays_match_diffusion_exact`. Every other reverse test uses scalar
    / single-element parameters, where such a collapse is a no-op. This drives the same
    three heterogeneous tubes through both inverse solvers.

    The inverse is ill-conditioned (a step ``cout`` leaves a few late ``cin`` modes
    underdetermined), so comparing recovered ``cin`` directly would be dominated by
    conditioning, not by the per-PV physics. Two choices keep the comparison about the
    physics: pore volumes off integer-residence multiples, and a moderate
    ``regularization_strength=1e-6``. The fast closed form and the slow quadrature then
    build forward matrices agreeing to ~1e-12, the shared Tikhonov solve amplifies that to
    ~7e-11, and a 1e-9 gate clears that floor with margin while still catching a per-PV
    collapse (which shifts the recovered ``cin`` by 1e-2 to 4 -- 7+ orders above the gate).
    """
    n = 250
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    t = np.arange(n)
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 30))
    cout = np.zeros(n)
    cout[50:] = 1.0

    common = {
        "cout": cout,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        # Off integer-residence multiples so the inverse is not needlessly degenerate.
        "aquifer_pore_volumes": np.array([437.3, 712.9, 1063.1]),
        "streamline_length": np.array([60.0, 90.0, 130.0]),
        "molecular_diffusivity": np.array([0.2, 0.5, 1.0]),
        "longitudinal_dispersivity": np.array([0.5, 1.5, 3.0]),
        "regularization_strength": 1e-6,
    }
    with warn_module.catch_warnings():
        warn_module.simplefilter("ignore")  # underdetermined late-cin modes (expected for a step inverse)
        cin_fast = extraction_to_infiltration(**common)
        cin_exact = diffusion_exact_reverse(**common)

    valid = ~np.isnan(cin_fast) & ~np.isnan(cin_exact)
    assert np.sum(valid) > 50
    assert_allclose(cin_fast[valid], cin_exact[valid], atol=1e-9, rtol=0)


def _good_diffusion_fast_inputs():
    """Baseline known-valid inputs for _validate_inputs (mirrors the diffusion.py snapshot)."""
    return {
        "cin_or_cout": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "flow": np.array([100.0, 100.0, 100.0, 100.0, 100.0]),
        "tedges": pd.date_range("2023-01-01", periods=6, freq="D"),
        "cout_tedges": pd.date_range("2023-01-01", periods=6, freq="D"),
        "aquifer_pore_volumes": np.array([300.0]),
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.01,
        "longitudinal_dispersivity": 0.5,
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
            lambda k: {**k, "molecular_diffusivity": -0.1},
            r"molecular_diffusivity must be non-negative",
        ),
        # Non-finite molecular_diffusivity slips past the ``< 0`` comparison (shared validator).
        (
            lambda k: {**k, "molecular_diffusivity": np.inf},
            r"molecular_diffusivity must be non-negative",
        ),
        (
            lambda k: {**k, "molecular_diffusivity": np.nan},
            r"molecular_diffusivity must be non-negative",
        ),
        (
            lambda k: {**k, "longitudinal_dispersivity": -0.1},
            r"longitudinal_dispersivity must be non-negative",
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
        # Non-finite aquifer_pore_volumes slips past the ``<= 0`` comparison (shared validator).
        (
            lambda k: {**k, "aquifer_pore_volumes": np.array([np.inf])},
            r"aquifer_pore_volumes must be positive",
        ),
        (
            lambda k: {**k, "aquifer_pore_volumes": np.array([np.nan])},
            r"aquifer_pore_volumes must be positive",
        ),
        (
            lambda k: {**k, "streamline_length": 0.0},
            r"streamline_length must be positive",
        ),
        (
            lambda k: {**k, "streamline_length": np.array([80.0, 90.0])},
            r"streamline_length must be a scalar or have length len\(aquifer_pore_volumes\)",
        ),
        (
            lambda k: {**k, "molecular_diffusivity": np.array([0.01, 0.02])},
            r"molecular_diffusivity must be a scalar or have length len\(aquifer_pore_volumes\)",
        ),
        (
            lambda k: {**k, "longitudinal_dispersivity": np.array([0.5, 0.6])},
            r"longitudinal_dispersivity must be a scalar or have length len\(aquifer_pore_volumes\)",
        ),
        (
            lambda k: {**k, "retardation_factor": 0.5},
            r"retardation_factor must be >= 1\.0",
        ),
        # NaN retardation slips past the ``< 1.0`` comparison (shared validator).
        (
            lambda k: {**k, "retardation_factor": np.nan},
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
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.01,
        "longitudinal_dispersivity": 0.5,
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

    with pytest.raises(ValueError, match=r"streamline_length must be positive"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=0.0,
            molecular_diffusivity=0.01,
            longitudinal_dispersivity=0.5,
        )


def test_spinup_invalid_string_rejected(tedges_short, constant_flow_short):
    """An unsupported ``spinup`` string raises ValueError with the helper's message."""
    flow = constant_flow_short
    cin = np.ones(len(flow))
    with pytest.raises(ValueError, match=r"spinup string must be 'constant'; got 'bad'"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges_short,
            cout_tedges=tedges_short,
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=80.0,
            molecular_diffusivity=0.01,
            longitudinal_dispersivity=0.5,
            spinup="bad",
        )


def test_spinup_float_not_implemented(tedges_short, constant_flow_short):
    """A float ``spinup`` (fraction-threshold mode) raises NotImplementedError.

    The public type hint advertises only ``str | None``; a float still reaches the shared
    ``_extend_tedges_flag`` and is rejected there rather than silently accepted.
    """
    flow = constant_flow_short
    cin = np.ones(len(flow))
    with pytest.raises(NotImplementedError, match=r"float thresholds are not implemented"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges_short,
            cout_tedges=tedges_short,
            aquifer_pore_volumes=np.array([500.0]),
            streamline_length=80.0,
            molecular_diffusivity=0.01,
            longitudinal_dispersivity=0.5,
            spinup=0.5,
        )


# =============================================================================
# Banded-build regression tests
#
# The forward operator is stored in banded layout (band_vals, col_start): row k of the dense W is
# band_vals[k] placed at columns [col_start[k], col_start[k] + full_band). The band spans only the
# breakthrough transition -- the cin bins with a non-zero coefficient -- because the bin-averaged
# C_F saturates to 0 or 1 outside it. These tests densify the banded production build and pin it
# against an EXPLICIT full-width dense reference (`_dense_closed_form_w`) that evaluates the same
# closed-form C_F on EVERY cin column, so no genuine coefficient is dropped: an under-sized band
# drops O(1e-2..1e-4) and would fail at once. The reference shares the antiderivative kernel, so
# it is bit-identical to the band where both are non-saturated. They complement the public-API
# equivalence tests above, which anchor against the independent `gwtransport.diffusion` quadrature.
# =============================================================================


def _densify_banded(band_vals, col_start, n_cin):
    """Place each row's band at columns [col_start[k], col_start[k] + full_band) -> dense (n_cout, n_cin)."""
    n_cout, full_band = band_vals.shape
    dense = np.zeros((n_cout, n_cin))
    cols = col_start[:, None] + np.arange(full_band)[None, :]
    rows = np.broadcast_to(np.arange(n_cout)[:, None], cols.shape)
    in_range = (cols >= 0) & (cols < n_cin)
    dense[rows[in_range], cols[in_range]] = band_vals[in_range]
    return dense


def _banded_dense(*, flow, tedges, cout_tedges, pv, length, d_m, alpha_l, retardation, flow_out):
    """Densified banded production build of the forward operator."""
    nb = len(pv)
    band_vals, col_start, _ = _closed_form_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow_out=flow_out,
        aquifer_pore_volumes=pv,
        streamline_length=np.full(nb, length),
        molecular_diffusivity=np.full(nb, d_m),
        longitudinal_dispersivity=np.full(nb, alpha_l),
        retardation_factor=retardation,
        extend_tedges=_extend_tedges_flag("constant"),
    )
    return _densify_banded(band_vals, col_start, len(flow))


def _dense_closed_form_w(*, flow, tedges, cout_tedges, pv, length, d_m, alpha_l, retardation, flow_out):
    """Explicit full-width dense closed-form C_F: the same kernel evaluated on EVERY cin column.

    Reference for the banded build -- it shares the antiderivative kernel and the volume/tau setup,
    so on the breakthrough band the two are computed from identical floating-point inputs (bit-exact
    for any regime). Outside the band the closed form saturates to exactly 0 or 1 in the cin-edge
    difference, which the banded build drops. The antiderivative's slope ``dD_t/dx = D_s/v_s`` is the
    Kreft-Zuber flux coefficient at the solute-front velocity, so differencing it gives ``C_F``
    natively for R > 1 with D_m > 0 -- no explicit retardation correction term.
    """
    nb = len(pv)
    streamline_length = np.full(nb, length)
    molecular_diffusivity = np.full(nb, d_m)
    longitudinal_dispersivity = np.full(nb, alpha_l)

    work_tedges = pd.DatetimeIndex([
        tedges[0] - pd.Timedelta("36500D"),
        *list(tedges[1:-1]),
        tedges[-1] + pd.Timedelta("36500D"),
    ])
    tedges_days = tedges_to_days(work_tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=work_tedges[0])
    v_cin = cumulative_flow_volume(flow, dt_to_days(work_tedges))
    v_cout = _cout_cumulative_volume(
        flow_out=flow_out,
        cout_tedges=cout_tedges,
        cout_tedges_days=cout_tedges_days,
        tedges_days=tedges_days,
        cumulative_volume_at_cin=v_cin,
    )
    tau = np.maximum(cout_tedges_days[:, None] - tedges_days[None, :], 0.0)

    n_cout = len(cout_tedges) - 1
    n_cin = len(flow)
    acc = np.zeros((n_cout, n_cin))
    for i_pv, v_pore in enumerate(pv):
        r_vpv = retardation * v_pore
        ell = float(streamline_length[i_pv])
        dm = float(molecular_diffusivity[i_pv])
        al = float(longitudinal_dispersivity[i_pv])
        step_widths = (v_cout[:, None] - v_cin[None, :] - r_vpv) * ell / r_vpv
        x_lo, x_hi = step_widths[:-1], step_widths[1:]
        dx = x_hi - x_lo
        if dm == 0.0 and al == 0.0:
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = (np.maximum(x_hi, 0.0) - np.maximum(x_lo, 0.0)) / dx
            frac = np.where(dx > 0.0, frac, 0.5 + 0.5 * np.sign(x_lo))
        else:
            xi = step_widths + ell
            dt_var = np.maximum(dm * tau + al * np.maximum(xi, 0.0), _DT_FLOOR)
            antideriv = _breakthrough_antideriv(step_widths, dt_var)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(dx > 0.0, np.diff(antideriv, axis=0) / dx, 0.0)
        acc += frac[:, :-1] - frac[:, 1:]
    return np.nan_to_num(acc / len(pv), nan=0.0)


def _assert_banded_matches_dense(w_banded, w_dense, retardation):  # noqa: ARG001
    """Banded build is bit-identical to the dense reference in every regime.

    The flux concentration emerges natively from the antiderivative's ``D_s/v_s`` slope, so the
    banded and dense paths compute the same ``frac`` from identical inputs for any R (no separate
    retardation-correction term that could diverge in the last ulps).
    """
    np.testing.assert_array_equal(w_banded, w_dense)


@pytest.mark.parametrize(
    ("d_m", "alpha_l", "retardation"),
    [
        (0.05, 0.0, 1.0),  # weak molecular diffusion
        (2.0, 0.0, 1.0),  # strong molecular diffusion (wide band)
        (0.0, 2.0, 1.0),  # pure mechanical dispersion
        (2.0, 0.0, 2.7),  # strong diffusion + retardation correction
        (0.0, 2.0, 2.7),  # mechanical dispersion + retardation
    ],
)
def test_banded_equals_dense_single_pv(d_m, alpha_l, retardation):
    """Densified banded build reproduces the full-width dense closed form across regimes (single PV).

    Geometry (pv=200, L=60) keeps every dispersive row's band narrow, so the banded path -- not a
    near-full band -- is exercised, including the R != 1 retardation correction.
    """
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pv": np.array([200.0]),
        "length": 60.0,
        "d_m": d_m,
        "alpha_l": alpha_l,
        "retardation": retardation,
    }
    w_banded = _banded_dense(**common, flow_out=flow)
    w_dense = _dense_closed_form_w(**common, flow_out=flow)
    _assert_banded_matches_dense(w_banded, w_dense, retardation)


def _banded_dense_threshold(*, flow, tedges, pv, length, d_m, alpha_l, saturation_threshold):
    """Densified banded production build at an explicit ``saturation_threshold``."""
    nb = len(pv)
    band_vals, col_start, _ = _closed_form_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        flow_out=flow,
        aquifer_pore_volumes=pv,
        streamline_length=np.full(nb, length),
        molecular_diffusivity=np.full(nb, d_m),
        longitudinal_dispersivity=np.full(nb, alpha_l),
        retardation_factor=1.0,
        extend_tedges=_extend_tedges_flag("constant"),
        saturation_threshold=saturation_threshold,
    )
    return _densify_banded(band_vals, col_start, len(flow))


@pytest.mark.parametrize("u_high", [12.0, 15.0])
def test_saturation_threshold_high_reproduces_default_build(u_high):
    """A larger ``U`` than the default (7) reproduces the default build bit-for-bit.

    The dropped breakthrough tail is of order ``exp(-U**2)``; at the default ``U=7`` it is
    already below the ulp of the saturated 0/1 values, so widening the band to ``U>=12`` adds
    no new non-zero coefficient. This pins that the default sits in the saturated/exact regime.
    """
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    common = {"flow": flow, "tedges": tedges, "pv": np.array([200.0]), "length": 60.0, "d_m": 0.5, "alpha_l": 1.0}
    w_default = _banded_dense_threshold(**common, saturation_threshold=7.0)
    w_high = _banded_dense_threshold(**common, saturation_threshold=u_high)
    np.testing.assert_array_equal(w_high, w_default)


@pytest.mark.parametrize("u_small", [2.0, 3.0])
def test_saturation_threshold_small_narrows_by_bounded_tail(u_small):
    """A small ``U`` narrows the band, dropping only a breakthrough tail of order ``exp(-U**2)``.

    The narrower band drops the saturated tail rather than any interior coefficient, so the
    densified forward operator departs from the default build by at most a bound proportional to
    ``exp(-U**2)`` (a per-edge ``frac`` difference), well above the ``U>=7`` machine-epsilon floor.
    """
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    common = {"flow": flow, "tedges": tedges, "pv": np.array([200.0]), "length": 60.0, "d_m": 0.5, "alpha_l": 1.0}
    w_default = _banded_dense_threshold(**common, saturation_threshold=7.0)
    w_small = _banded_dense_threshold(**common, saturation_threshold=u_small)
    max_diff = np.max(np.abs(w_small - w_default))
    # Non-trivially narrower: a genuine tail is dropped (not bit-identical to the default).
    assert max_diff > 0.0
    # ...but bounded by the saturated-tail magnitude exp(-U**2).
    assert max_diff < 10.0 * np.exp(-u_small * u_small)


def test_banded_both_zero_equals_advection_step():
    """Both-zero (D_m=0, alpha_L=0) forward equals a pure-advection step to ~1e-14.

    The zero-dispersion case routes through the same banded path; D_t floors to ~1e-30 so C_F is a
    step smoothed by ~1e-15. The densified band must reproduce the advection breakthrough operator,
    whose entries are 0 or 1 on aligned grids.
    """
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    w_banded = _banded_dense(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        pv=np.array([200.0]),
        length=60.0,
        d_m=0.0,
        alpha_l=0.0,
        retardation=1.0,
        flow_out=flow,
    )
    # Pure-advection step reference: dense closed form's exact 0/1 breakthrough operator.
    w_step = _dense_closed_form_w(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        pv=np.array([200.0]),
        length=60.0,
        d_m=0.0,
        alpha_l=0.0,
        retardation=1.0,
        flow_out=flow,
    )
    np.testing.assert_allclose(w_banded, w_step, atol=1e-15, rtol=0.0)
    # Honor the "advection" claim: on this aligned grid the operator is a pure shift, so every
    # non-zero coefficient is exactly 1.0 (no dispersive splitting across neighbouring columns).
    nonzero = w_banded[np.abs(w_banded) > 1e-12]
    assert nonzero.size > 50
    np.testing.assert_array_equal(nonzero, 1.0)


def test_banded_variable_flow_wandering_center():
    """Under variable flow the band centre drifts non-linearly across cout rows; the band tracks it."""
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 30.0))
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pv": np.array([500.0]),
        "length": 30.0,
        "d_m": 0.5,
        "alpha_l": 1.0,
        "retardation": 1.0,
    }
    w_banded = _banded_dense(**common, flow_out=flow)
    w_dense = _dense_closed_form_w(**common, flow_out=flow)
    np.testing.assert_array_equal(w_banded, w_dense)


@pytest.mark.parametrize("retardation", [1.0, 2.7])
def test_banded_multipv_union_of_bands(retardation):
    """Each streamtube's band sits at a different cin offset; the accumulated band is their union.

    The R != 1 case also guards the per-tube accumulation of the (natively flux-corrected) bands
    into the union buffer -- a per-tube offset error would corrupt the overlap.
    """
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pv": np.array([120.0, 200.0, 280.0]),
        "length": 100.0,
        "d_m": 1.0,
        "alpha_l": 0.3,
        "retardation": retardation,
    }
    w_banded = _banded_dense(**common, flow_out=flow)
    w_dense = _dense_closed_form_w(**common, flow_out=flow)
    _assert_banded_matches_dense(w_banded, w_dense, retardation)


def test_banded_nonuniform_cin_grid_equals_dense():
    """Non-uniform cin bins must not mis-size the band.

    The band edges are located by ``searchsorted`` on the cumulative-volume axis, not by a
    column count, so mixed bin widths (here 1/1/2/5-day) are handled directly -- the regime a
    fixed-width column band would break.
    """
    widths = np.tile([1, 1, 2, 5], 35)
    days = np.concatenate(([0], np.cumsum(widths)))
    tedges = pd.DatetimeIndex(pd.Timestamp("2020-01-01") + pd.to_timedelta(days, unit="D"))
    n = len(tedges) - 1
    flow = np.full(n, 100.0)
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pv": np.array([800.0]),
        "length": 60.0,
        "d_m": 0.5,
        "alpha_l": 1.0,
        "retardation": 1.0,
    }
    w_banded = _banded_dense(**common, flow_out=flow)
    w_dense = _dense_closed_form_w(**common, flow_out=flow)
    np.testing.assert_array_equal(w_banded, w_dense)


@pytest.mark.parametrize(
    ("d_m", "pv", "length", "flow_lo", "flow_hi"),
    [
        # Strong diffusion: the broken-through band reaches injections from a faster era, so the post
        # slope must use the SLOWEST flow rate, not the front's. Guards the min_cin_flow bound.
        (50.0, 300.0, 80.0, 100.0, 300.0),
        # Weak diffusion + sharp contrast: the post band anchors at the lower cout edge, whose front
        # sits in slow flow with large tau, so it needs its own intercept. Guards a_post (lower edge).
        (0.001, 20.0, 50.0, 0.5, 100.0),
    ],
)
def test_banded_variable_flow_step_equals_dense(d_m, pv, length, flow_lo, flow_hi):
    """Sharp flow step under-covers the band unless both conservative post bounds hold.

    The breakthrough at a cout bin in one flow era reaches injections from another era (different
    tau, hence different D_t), so a front-anchored band width would drop real coefficients. Both
    parameter rows would fail against an under-sized band (~1e-2..1e-4 error); the conservative
    closed form must reproduce the full-width dense build.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.concatenate([np.full(n // 2, flow_lo), np.full(n - n // 2, flow_hi)])
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pv": np.array([pv]),
        "length": length,
        "d_m": d_m,
        "alpha_l": 0.0,
        "retardation": 2.0,
    }
    w_banded = _banded_dense(**common, flow_out=flow)
    w_dense = _dense_closed_form_w(**common, flow_out=flow)
    _assert_banded_matches_dense(w_banded, w_dense, 2.0)


def test_banded_retardation_matches_diffusion_exact():
    """Banded R != 1 build (native solute-front flux coefficient) matches the independent
    ``gwtransport.diffusion`` quadrature -- an anchor that does not share the closed-form code.

    Geometry (pv=200, L=60) keeps the band narrow enough that banding engages rather than falling
    back, so the banded retardation branch is genuinely exercised.
    """
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    cin = np.zeros(n)
    cin[40] = 1.0
    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "aquifer_pore_volumes": np.array([200.0]),
        "molecular_diffusivity": 2.0,
        "longitudinal_dispersivity": 0.0,
        "retardation_factor": 2.7,
    }
    cout_fast = infiltration_to_extraction(cin=cin, streamline_length=60.0, **kwargs)
    cout_exact = diffusion_exact(cin=cin, streamline_length=np.array([60.0]), **kwargs)
    valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-12, rtol=0.0)


def test_banded_forward_masking_matches_diffusion():
    """Banded forward output values AND NaN (spin-up / out-of-range) mask match the slow quadrature.

    Guards that the banded einsum and the total-coefficient masking reproduce the slow module's
    finite/NaN pattern, not just the in-band values.
    """
    n = 150
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 100.0 * (1.0 + 0.2 * np.sin(2 * np.pi * np.arange(n) / 40.0))
    cin = np.sin(np.linspace(0, 8, n)) + 1.0
    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "aquifer_pore_volumes": np.array([300.0]),
        "molecular_diffusivity": 0.5,
        "longitudinal_dispersivity": 1.0,
        "retardation_factor": 1.0,
        "flow_out": flow,
    }
    cout_fast = infiltration_to_extraction(cin=cin, streamline_length=40.0, **kwargs)
    kwargs_exact = {k: v for k, v in kwargs.items() if k != "flow_out"}
    cout_exact = diffusion_exact(cin=cin, streamline_length=np.array([40.0]), **kwargs_exact)
    np.testing.assert_array_equal(np.isnan(cout_fast), np.isnan(cout_exact))
    valid = ~np.isnan(cout_fast)
    assert np.sum(valid) > 50
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-12, rtol=0.0)


def test_banded_inverse_roundtrip_nonuniform_cin_and_cout():
    """Inverse round-trip on NON-uniform cin AND non-uniform (misaligned, sub-bin) cout.

    Builds cout via the forward banded operator and deconvolves with the banded inverse on a
    non-uniform cout grid that REFINES the non-uniform cin grid (each cin bin split into halves,
    plus a sub-bin first edge), so the system is well-determined. Exercises col_start / full_band
    on grids where a fixed-width band would mis-align; the smooth interior recovers tightly.
    """
    cin_widths = np.tile([1, 1, 2, 3], 40)
    cin_days = np.concatenate(([0], np.cumsum(cin_widths)))
    tedges = pd.DatetimeIndex(pd.Timestamp("2020-01-01") + pd.to_timedelta(cin_days, unit="D"))
    n = len(tedges) - 1
    flow = np.full(n, 100.0)
    cin_true = np.sin(np.linspace(0, 10, n)) + 2.0

    # Non-uniform cout that refines cin (each cin edge plus its midpoint), so cout is finer than cin
    # and the inverse is well-posed despite the misaligned, non-uniform layout.
    mids = 0.5 * (cin_days[:-1] + cin_days[1:])
    cout_days = np.unique(np.concatenate([cin_days.astype(float), mids]))
    cout_tedges = pd.DatetimeIndex(pd.Timestamp("2020-01-01") + pd.to_timedelta(cout_days, unit="D"))
    flow_out = np.full(len(cout_tedges) - 1, 100.0)

    fwd_kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
        "streamline_length": 50.0,
        "molecular_diffusivity": 0.5,
        "longitudinal_dispersivity": 1.0,
        "flow_out": flow_out,
    }
    cout = infiltration_to_extraction(cin=cin_true, **fwd_kwargs)
    cout_filled = np.nan_to_num(cout, nan=0.0)
    cin_rec = extraction_to_infiltration(cout=cout_filled, regularization_strength=1e-10, **fwd_kwargs)

    # The interior (away from spin-up edges) is well-constrained; check it is recovered.
    finite = ~np.isnan(cin_rec)
    interior = finite.copy()
    interior[: n // 4] = False
    interior[-n // 4 :] = False
    assert np.sum(interior) > 20
    assert_allclose(cin_rec[interior], cin_true[interior], atol=5e-3, rtol=0.0)


@pytest.mark.parametrize("n_lead", [1, 5, 12, 30], ids=["lead1", "lead5", "lead12", "lead30"])
@pytest.mark.parametrize(("d_m", "alpha_l"), [(0.1, 0.0), (2.0, 0.0), (0.05, 0.5)], ids=["dm0.1", "dm2", "dm_al"])
def test_banded_leading_zero_flow_warm_start_equals_dense(n_lead, d_m, alpha_l):
    """Leading zero-flow plateau + dispersion: the band keeps the warm-start data-start tail.

    Regression for the banded under-coverage bug. With ``flow[:n_lead] == 0`` the cumulative volume
    is flat over the leading edges, so the front search lands the local band one or more columns
    inside the record while a genuine warm-start coefficient (large ``tau`` -> large ``D_t`` at cin
    edge 0, magnitude up to ~0.47 per row) remains at column 0 and spreads across the leading
    plateau. A band that did not reach column 0 dropped it (~0.3..0.5 error). The densified band must
    reproduce the full-width dense closed form bit-for-bit.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[:n_lead] = 0.0
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "pv": np.array([300.0]),
        "length": 50.0,
        "d_m": d_m,
        "alpha_l": alpha_l,
        "retardation": 1.0,
    }
    w_banded = _banded_dense(**common, flow_out=flow)
    w_dense = _dense_closed_form_w(**common, flow_out=flow)
    np.testing.assert_array_equal(w_banded, w_dense)


@pytest.mark.parametrize("retardation", [1.0, 2.0], ids=["R1", "R2"])
def test_banded_band_sizing_misaligned_subbin_multipv(retardation):
    """Band sizing on a misaligned, sub-bin cout grid (multi-PV, variable flow) equals the dense ref.

    The cout grid is finer than -- and offset from -- the variable-flow cin grid, so each cout bin
    straddles cin bins and the per-row band must track a non-integer front across multiple
    streamtubes. Bit-identical to the full-width dense closed form for any R (the flux coefficient
    emerges natively). An under-sized band would drop O(1e-2..1e-4).
    """
    n = 120
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * np.arange(n) / 25.0))
    # Misaligned sub-bin cout: refine each cin bin into halves and offset the first edge.
    cout_tedges = pd.date_range(tedges[0], tedges[-1], periods=2 * n + 1)
    flow_out = np.repeat(flow, 2)
    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "pv": np.array([200.0, 350.0, 500.0]),
        "length": 50.0,
        "d_m": 0.1,
        "alpha_l": 0.5,
        "retardation": retardation,
    }
    w_banded = _banded_dense(**common, flow_out=flow_out)
    w_dense = _dense_closed_form_w(**common, flow_out=flow_out)
    np.testing.assert_array_equal(w_banded, w_dense)


def test_leading_zero_flow_forward_matches_diffusion_exact():
    """Forward with a leading zero-flow plateau + molecular diffusion matches the slow quadrature.

    The leading ``n_lead`` zero-flow bins hold the cumulative volume flat; with ``D_m > 0`` the
    warm-start constant breaks through the data-start edge with a large ``D_t``, contributing a
    non-saturated coefficient at cin column 0. The banded build keeps it (regression for the
    dropped-warm-start bug), so a constant ``cin`` -- whose flux concentration is itself a clean
    Kreft-Zuber breakthrough -- reproduces ``gwtransport.diffusion`` to machine precision.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[:12] = 0.0
    cin = np.ones(n)
    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "aquifer_pore_volumes": np.array([300.0]),
        "molecular_diffusivity": 0.1,
        "longitudinal_dispersivity": 0.0,
        "retardation_factor": 1.0,
    }
    cout_fast = infiltration_to_extraction(cin=cin, streamline_length=50.0, **kwargs)
    cout_exact = diffusion_exact(cin=cin, streamline_length=np.array([50.0]), **kwargs)
    # NaN (spin-up / out-of-range) mask must match, and the finite breakthrough is bit-exact.
    np.testing.assert_array_equal(np.isnan(cout_fast), np.isnan(cout_exact))
    valid = ~np.isnan(cout_fast)
    assert np.sum(valid) > 100
    assert_allclose(cout_fast[valid], cout_exact[valid], atol=1e-12, rtol=0.0)


def test_leading_zero_flow_inverse_finite_and_matches_dense_reverse():
    """Inverse with a leading zero-flow plateau + molecular diffusion returns finite (no LinAlgError).

    The warm-start data-start tail makes the forward operator carry negative coefficients at the
    leading columns; their non-positive column sum left the banded normal equations indefinite,
    crashing ``cholesky_banded``. The warm-start tail is decoupled before the solve, so the inverse
    now returns a finite result with no exception, and the well-determined interior (away from the
    unrecoverable spin-up nullspace) matches the slow-module reverse.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[:12] = 0.0
    cin_true = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 40.0)
    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": tedges,
        "aquifer_pore_volumes": np.array([300.0]),
        "streamline_length": 50.0,
        "molecular_diffusivity": 0.1,
        "longitudinal_dispersivity": 0.0,
        "retardation_factor": 1.0,
    }
    cout = infiltration_to_extraction(cin=cin_true, **kwargs)
    cout_filled = np.nan_to_num(cout, nan=0.0)

    with warn_module.catch_warnings():
        warn_module.simplefilter("ignore")
        cin_rec = extraction_to_infiltration(cout=cout_filled, regularization_strength=1e-10, **kwargs)
        cin_ref = diffusion_exact_reverse(
            cout=cout_filled,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([300.0]),
            streamline_length=np.array([50.0]),
            molecular_diffusivity=0.1,
            longitudinal_dispersivity=0.0,
            retardation_factor=1.0,
        )

    # No exception was raised, and the recovered signal is finite where defined (not all-NaN).
    assert np.isfinite(cin_rec).any()
    # The well-determined interior (away from the spin-up nullspace) matches the slow reverse.
    interior = np.zeros(n, dtype=bool)
    interior[n // 2 : n - 10] = True
    both = interior & np.isfinite(cin_rec) & np.isfinite(cin_ref)
    assert np.sum(both) > 50
    assert_allclose(cin_rec[both], cin_ref[both], atol=1e-6, rtol=0.0)


# =============================================================================
# Interior zero-flow gap: band coverage and stagnation-straddle aggregation (#306)
# =============================================================================


def test_interior_zero_flow_gap_band_covers_gap_accumulated_diffusion():
    """Interior zero-flow gap: the default band (U=7) reproduces the wide band (U=25).

    During a pumped-off (zero-flow) gap, molecular diffusion keeps growing the moving-frame
    variance ``D_t = D_m*tau + alpha_L*xi`` (``tau`` grows, ``xi`` frozen), so post-restart
    cout bins carry non-saturated coefficients at pre-gap cin columns whose ``D_t`` includes
    the full stagnation time. A post-side band extent sized from the dispersion slope and the
    front intercept alone truncates those columns (errors up to ~5e-4 here, ~4.9e-3 in wider
    scans); the record is long enough that the warm-start data-start tail check is saturated
    and cannot mask the truncation. At any ``U >= ~6`` the dropped coefficients are exactly
    zero, so the U=7 and U=25 builds must agree to machine precision.
    """
    n = 1200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[1000:1100] = 0.0
    # Misaligned cout grid: same daily spacing, offset 6 h, so bins straddle the gap edges.
    cout_tedges = tedges[:-1] + pd.Timedelta(hours=6)
    tedges_days = tedges_to_days(tedges)
    cout_days = tedges_to_days(cout_tedges, ref=tedges[0])
    cumvol = cumulative_flow_volume(flow, dt_to_days(tedges))
    flow_out = np.diff(np.interp(cout_days, tedges_days, cumvol)) / np.diff(cout_days)
    cin = np.random.default_rng(42).random(n)
    kwargs = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([300.0]),
        "streamline_length": 10.0,
        "molecular_diffusivity": 0.1,
        "longitudinal_dispersivity": 0.0,
        "flow_out": flow_out,
    }
    cout_default = infiltration_to_extraction(**kwargs, saturation_threshold=7.0)
    cout_wide = infiltration_to_extraction(**kwargs, saturation_threshold=25.0)
    np.testing.assert_array_equal(np.isnan(cout_default), np.isnan(cout_wide))
    valid = ~np.isnan(cout_default)
    assert np.sum(valid) > 1000
    assert_allclose(cout_default[valid], cout_wide[valid], atol=1e-13, rtol=0.0)


@pytest.mark.parametrize(
    ("d_m", "alpha_l", "pv"),
    # (0.2, 0.1, 1500) is excluded: at that dispersion strength the 16-point quadrature reference
    # itself carries a ~6e-12 resolution floor on this coarse grid even without a gap.
    [(0.05, 0.1, 1000.0), (0.2, 0.0, 1500.0)],
)
def test_coarse_cout_straddling_flow_restart_matches_diffusion_quadrature(d_m, alpha_l, pv):
    """Coarse cout bins straddling a zero-flow gap match the quadrature volume-weighted aggregate.

    The banded ``C_F`` is a two-endpoint antiderivative difference ``(I(x_hi) - I(x_lo))/dx``,
    exact whenever ``D_t`` follows the flow path (``dD_t/dx = D_s/v_s``). Across a stagnation
    (zero-flow) interval, ``x`` is frozen while ``D_t`` keeps growing, so a coarse cout bin
    straddling the gap must exclude the vertical ``int (dI/dD_t) dD_t`` jump -- no volume is
    extracted during the gap, so it carries no weight in the flux-weighted bin average.
    ``gwtransport.diffusion`` integrates in volume space and skips zero-``dV`` flow bins, so it
    defines the aggregate exactly (a naive endpoint difference deviates by up to ~7e-3 here).
    """
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[20:30] = 0.0
    # Coarse 4-day cout bins; bin [day 28, day 32) straddles the flow restart at day 30.
    cout_tedges = tedges[::4]
    tedges_days = tedges_to_days(tedges)
    cout_days = tedges_to_days(cout_tedges, ref=tedges[0])
    cumvol = cumulative_flow_volume(flow, dt_to_days(tedges))
    flow_out = np.diff(np.interp(cout_days, tedges_days, cumvol)) / np.diff(cout_days)
    cin = np.random.default_rng(7).random(n)
    kwargs = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([pv]),
        "molecular_diffusivity": d_m,
        "longitudinal_dispersivity": alpha_l,
    }
    cout_fast = infiltration_to_extraction(**kwargs, streamline_length=10.0, flow_out=flow_out)
    cout_quad = diffusion_exact(**kwargs, streamline_length=np.array([10.0]))
    both = np.isfinite(cout_fast) & np.isfinite(cout_quad)
    assert np.sum(both) > 5
    assert_allclose(cout_fast[both], cout_quad[both], atol=1e-12, rtol=0.0)


def test_extraction_to_infiltration_gapped_cout_masked():
    """NaN gaps in cout are masked out of the banded inverse instead of raising (#321).

    Sparse lab samples leave NaN cout bins; the reverse operator must exclude
    those rows from the banded Tikhonov normal equations -- matching
    :func:`gwtransport.deposition.extraction_to_deposition` -- rather than
    reject the whole series. cout at 12h resolution (via ``flow_out``) vs daily
    cin keeps the system overdetermined, so the surviving rows still pin every
    interior cin bin and recovery stays at the no-gap round-trip floor.
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
