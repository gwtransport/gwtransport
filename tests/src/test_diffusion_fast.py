import warnings as warn_module

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import erf

from gwtransport import gamma as gamma_utils
from gwtransport.advection import infiltration_to_extraction as advection_i2e
from gwtransport.diffusion import infiltration_to_extraction as diffusion_exact
from gwtransport.diffusion_fast import (
    _build_gaussian_matrix,
    _build_rebin_matrix,
    _build_v_smooth_matrix,
    extraction_to_infiltration,
    gamma_extraction_to_infiltration,
    gamma_infiltration_to_extraction,
    infiltration_to_extraction,
)
from gwtransport.gamma import mean_std_loc_to_alpha_beta

# =============================================================================
# Tests for _build_gaussian_matrix
# =============================================================================


def test_build_gaussian_matrix_identity_for_zero_sigma():
    """Zero sigma produces exact identity matrix."""
    n = 10
    sigma_array = np.zeros(n)
    g = _build_gaussian_matrix(n=n, sigma_array=sigma_array)
    assert_allclose(g.toarray(), np.eye(n), atol=0.0)


def test_build_gaussian_matrix_row_sums_to_one():
    """Every row of the Gaussian matrix sums to exactly 1."""
    n = 50
    sigma_array = np.random.default_rng(42).uniform(0.5, 5.0, n)
    g = _build_gaussian_matrix(n=n, sigma_array=sigma_array, asymptotic_cutoff_sigma=10.0)
    row_sums = np.array(g.sum(axis=1)).flatten()
    assert_allclose(row_sums, np.ones(n), atol=1e-14)


def test_build_gaussian_matrix_symmetry():
    """With constant sigma, interior rows of G are symmetric about diagonal."""
    n = 100
    sigma = 3.0
    sigma_array = np.full(n, sigma)
    g = _build_gaussian_matrix(n=n, sigma_array=sigma_array, asymptotic_cutoff_sigma=10.0)
    g_dense = g.toarray()

    # Check that G[i,j] = G[i, 2i-j] for interior rows (kernel symmetry)
    margin = int(10 * sigma)
    for i in range(margin, n - margin):
        radius = min(i, n - 1 - i, int(10 * sigma))
        for k in range(1, radius):
            assert_allclose(
                g_dense[i, i - k],
                g_dense[i, i + k],
                atol=1e-15,
                err_msg=f"Asymmetry at row {i}, offset {k}",
            )


# =============================================================================
# Convergence tests: analytical solutions
#
# These compare discrete CDF-kernel output against continuous analytical
# solutions. The error is O(dx^2) from discretizing the initial condition,
# NOT from the kernel. The CDF kernel is exact for piecewise-constant input.
# =============================================================================
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
        suppress_dispersion_warning=True,
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


@pytest.mark.parametrize("retardation_factor", [1.0, 2.7])
def test_retardation_zero_diffusion_matches_advection(retardation_factor):
    """Zero diffusion with retardation matches pure advection (exact, G=I)."""
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

    The forward matrix W = W_adv @ G is built identically in both functions,
    so the Tikhonov solve recovers the original cin exactly (up to floating-point
    arithmetic).
    """
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin_original = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0

    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
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
    assert_allclose(cin_recovered[valid], cin_original[valid], atol=1e-13)


# =============================================================================
# Integration test: diffusion_fast vs diffusion on same extraction grid
# =============================================================================


def test_diffusion_fast_vs_diffusion_same_grid():
    """Both modules return results on the same extraction time grid.

    Under constant flow, the fast (Gaussian smoothing) and exact (analytical
    erf) methods agree within the inherent approximation error. The Gaussian
    kernel is an approximation of the erf-based analytical solution, so
    exact agreement is not expected. The error is largest near sharp fronts
    (step function) and smallest for smooth signals.
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

    # Smooth signal: Gaussian approximation agrees within 0.2% of signal range
    assert_allclose(cout_fast[both_valid], cout_exact[both_valid], atol=2e-3)

    # Test with step function — larger approximation gap
    cin_step = np.zeros(n_days)
    cin_step[30:] = 1.0
    cout_fast_step = infiltration_to_extraction(cin=cin_step, **fast_kwargs)
    cout_exact_step = diffusion_exact(cin=cin_step, **exact_kwargs)

    both_valid_step = ~np.isnan(cout_fast_step) & ~np.isnan(cout_exact_step)
    assert_allclose(cout_fast_step[both_valid_step], cout_exact_step[both_valid_step], atol=0.02)


# =============================================================================
# Tests for flow_out (advect-then-smooth) algorithm
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
    """Zero diffusion with flow_out on non-uniform grid matches pure advection (exact)."""
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
    assert_allclose(cout_fast[valid], cout_adv[valid], atol=0.0)


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
    """Round-trip with flow_out recovers original signal (machine precision)."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin_original = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0
    flow_out = flow.copy()

    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
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
    assert_allclose(cin_recovered[valid], cin_original[valid], atol=1e-14)


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
            "suppress_dispersion_warning": True,
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

    def test_dispersion_warning_emitted(self, gamma_setup):
        """Multiple pore volumes with dispersivity emits UserWarning."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.ones(n_cout) * 5.0

        with pytest.warns(UserWarning, match="multiple.*pore.*volumes|velocity heterogeneity"):
            gamma_extraction_to_infiltration(
                cout=cout,
                flow=gamma_setup["flow"],
                tedges=gamma_setup["tedges"],
                cout_tedges=gamma_setup["cout_tedges"],
                mean=gamma_setup["mean"],
                std=gamma_setup["std"],
                n_bins=gamma_setup["n_bins"],
                mean_streamline_length=gamma_setup["mean_streamline_length"],
                mean_molecular_diffusivity=0.0,
                mean_longitudinal_dispersivity=1.0,
            )

    def test_suppress_dispersion_warning(self, gamma_setup):
        """suppress_dispersion_warning=True silences the dispersion warning."""
        n_cout = len(gamma_setup["cout_tedges"]) - 1
        cout = np.ones(n_cout) * 5.0

        with warn_module.catch_warnings():
            warn_module.simplefilter("error")
            # Allow the rank-deficient warning (separate concern)
            warn_module.filterwarnings("ignore", message="Forward matrix is rank-deficient")
            gamma_extraction_to_infiltration(
                cout=cout,
                flow=gamma_setup["flow"],
                tedges=gamma_setup["tedges"],
                cout_tedges=gamma_setup["cout_tedges"],
                mean=gamma_setup["mean"],
                std=gamma_setup["std"],
                n_bins=gamma_setup["n_bins"],
                mean_streamline_length=gamma_setup["mean_streamline_length"],
                mean_molecular_diffusivity=0.0,
                mean_longitudinal_dispersivity=1.0,
                suppress_dispersion_warning=True,
            )


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
# The forward operator reports an approximate Bear resident concentration
# (no Kreft-Zuber boundary-flux correction); see the module docstring for
# the trade-off. After the V-coordinate smoothing rewrite, the column-sum
# invariant
#
#     sum_i W[i, j] * Q_out[i] * dt_out[i] = Q_in[j] * dt_in[j]
#
# holds at machine precision when the breakthrough sigma is V-independent
# (pure alpha_L), and to O(sigma_V slope) when sigma_V varies with V
# (pure D_m or mixed). The remaining residual is the discretisation error
# of the variable-sigma row-normalised Gaussian on the uniform-V grid; it
# matches the asymptotic bound predicted by Sinkhorn-Knopp theory for a
# non-doubly-stochastic operator. Adding a Kreft-Zuber additive correction
# does NOT close this residual — see ``audit_pe_residual.py`` in the
# Step-1/2 prototypes for the diagnostics.
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
    ("d_m", "alpha_l", "atol"),
    [
        # Observed worst-case residuals across the three seeds: 4.05e-3
        # (pure D_m) and 2.74e-3 (mixed). atol set ~1.5x over the worst
        # case so a future seed slightly outside this range does not flake.
        (1.0, 0.0, 6e-3),
        (0.0, 0.3, 1e-10),  # sigma_V V-independent; exact at machine precision
        (1.0, 0.3, 5e-3),
    ],
)
def test_column_mass_conservation_variable_flow_dispersion_flow_out(d_m, alpha_l, seed, atol):
    """Volume-weighted column sums of W approximately match infiltrated volume
    under variable Q for the advect-then-smooth (``flow_out`` provided) branch.

    Slow ``gwtransport.diffusion`` holds this invariant at ``atol=1e-10``
    (issue #180). The fast V-coordinate smoothing path here is an
    approximation; the residual is dominated by the variable-sigma_V
    discretisation on the uniform-V grid and stays below ~1e-2 for typical
    aquifer parameters.
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
    v_in_interior = (flow * dt)[15:40]
    mass_out_per_cin = (v_out[:, None] * w).sum(axis=0)
    ratios = mass_out_per_cin[15:40] / v_in_interior
    assert_allclose(ratios, 1.0, atol=atol, rtol=0)


@pytest.mark.parametrize("seed", [1, 3, 42])
@pytest.mark.parametrize(
    ("d_m", "alpha_l", "atol"),
    [
        # smooth-then-advect on the input grid carries a slightly larger
        # variable-sigma_V residual than the advect-then-smooth branch
        # because the input V-grid contains the wide spin-up bins. Worst
        # observed residual is 5.63e-3 (D_m=1); atol set ~1.5x over.
        (1.0, 0.0, 8e-3),
        (0.0, 0.3, 1e-10),
        (1.0, 0.3, 6e-3),
    ],
)
def test_column_mass_conservation_variable_flow_dispersion_default(d_m, alpha_l, seed, atol):
    """Same invariant as the ``flow_out`` variant, but for the smooth-then-advect
    (``flow_out is None``) branch.

    Bounded by ~1e-2 for the same physical reason. Original ``diffusion_fast``
    (pre-V-coord rewrite) had ~1e-1 here; the V-coord smoothing reduces it
    by 30-100x depending on the dispersion mix.
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
    v_in_interior = (flow * dt)[15:40]
    mass_out_per_cin = (v_out[:, None] * w).sum(axis=0)
    ratios = mass_out_per_cin[15:40] / v_in_interior
    assert_allclose(ratios, 1.0, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("d_m", "alpha_l"),
    [(1.0, 0.0), (0.0, 0.3), (1.0, 0.3)],
)
def test_column_mass_conservation_constant_flow_control(d_m, alpha_l):
    """Constant Q is the machine-precision control for column mass conservation.

    With ``flow`` constant, ``v_edges`` is uniform and the V-smoothing
    operator's row-stochastic-plus-mass-preserving rebins compose exactly
    (no variable-sigma-on-uniform-grid drift). Any residual above ~1e-12
    is a structural bug in the operator stack.
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
    assert_allclose(ratios, 1.0, atol=1e-12, rtol=0)


@pytest.mark.parametrize("seed", [3, 42])
@pytest.mark.parametrize(
    ("d_m", "alpha_l", "atol"),
    [
        (1.0, 0.0, 5e-3),
        (0.0, 0.3, 1e-10),
        (1.0, 0.3, 5e-3),
    ],
)
def test_column_mass_conservation_multipv(d_m, alpha_l, seed, atol):
    """Mass conservation under variable Q for a multi-pore-volume APVD.

    With a single pore volume the moment formula
    ``var_numerator = (2*D_m*tau_bar/V_bar)*E[V^3] + 2*alpha_L*L*E[V^2]``
    collapses to a scalar because ``E[V^2] = V^2`` and ``E[V^3]/V_bar = V^2``.
    Using ``aquifer_pore_volumes = [600, 1000, 1400]`` makes
    ``E[V^2]`` and ``E[V^3]/V_bar`` differ by ~19%, which exercises the
    moment placement in :func:`gwtransport.diffusion_fast._compute_sigma_v`.
    The mass-conservation invariant is V-bin-local and remains valid
    independent of the moment formula.
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
                suppress_dispersion_warning=True,
                flow_out=flow,
            )

    w = _build_w_via_probes(call, n=n)
    dt = (np.diff(tedges) / pd.Timedelta("1D")).astype(float)
    v_out = flow * dt
    v_in_interior = (flow * dt)[20:40]
    ratios = (v_out[:, None] * w).sum(axis=0)[20:40] / v_in_interior
    assert_allclose(ratios, 1.0, atol=atol, rtol=0)


# =============================================================================
# Sigma-sensitivity: tests that fail if sigma_V is wrong by a constant factor
#
# The mass-conservation and round-trip tests above are blind to sigma magnitude
# (the former tests row-stochasticity, the latter cancels sigma in forward and
# inverse). The tests below probe sigma directly by comparing the discrete
# Gaussian breakthrough to its analytical limit.
# =============================================================================


@pytest.mark.parametrize("retardation", [1.0, 2.7])
def test_delta_input_matches_analytical_gaussian_constant_flow(retardation):
    """Single delta cin pulse under constant Q matches the analytical Gaussian.

    For a single pore volume and constant flow, the V-coordinate breakthrough
    kernel for a delta input at bin j is approximately

        cout[i] proportional to (erf((tau_idx + 0.5 - i)/(sigma_idx*sqrt(2)))
                                 - erf((tau_idx - 0.5 - i)/(sigma_idx*sqrt(2)))) / 2

    where ``tau_idx = j + R*V_pore / (Q*dt)`` is the breakthrough centre and
    ``sigma_idx = (R*V_pore/L) * sqrt(2*D_m*tau + 2*alpha_L*L) / (Q*dt)`` is
    the V-coordinate sigma converted to bin-index units. Mutating sigma_V
    by any constant factor (or zeroing it) breaks this match — the
    mass-conservation tests do not catch this because they probe row
    sums, not the kernel shape.

    Parametrising on ``retardation`` exercises the R factor in both
    ``tau = R*V_pore/Q`` and ``sigma_V = (R*V_pore/L)*sigma_x``: an
    accidentally dropped or mispowered R in
    :func:`gwtransport.diffusion_fast._compute_sigma_v` fails the R=2.7
    case where it would slip past R=1.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    flow_rate = 100.0
    flow = np.full(n, flow_rate)
    d_m = 2.0
    alpha_l = 0.0
    v_pore = 1000.0
    streamline_length = 30.0
    # Pick a regime where sigma is well-resolved by the grid:
    #   tau = R*V_pore/Q  (10 days at R=1; 27 days at R=2.7)
    #   sigma_x = sqrt(2*D_m*tau)
    #   sigma_V = (R*V_pore/L)*sigma_x
    #   sigma_t = sigma_V/(Q*1d)   -> several bins, resolved

    j_in = 60
    cin = np.zeros(n)
    cin[j_in] = 1.0

    cout = infiltration_to_extraction(
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

    tau = retardation * v_pore / flow_rate
    sigma_x = np.sqrt(2.0 * d_m * tau)
    sigma_v = (retardation * v_pore / streamline_length) * sigma_x
    sigma_idx = sigma_v / (flow_rate * 1.0)
    centre = j_in + tau / 1.0

    i_arr = np.arange(n)
    sqrt2 = np.sqrt(2.0)
    expected = 0.5 * (
        erf((i_arr - centre + 0.5) / (sigma_idx * sqrt2)) - erf((i_arr - centre - 0.5) / (sigma_idx * sqrt2))
    )

    valid = ~np.isnan(cout)
    # Compare in the kernel-active window. Outside ±8 sigma the analytical
    # Gaussian is numerically zero; boundary effects in the V-smoothing add
    # a tiny drift there that is not load-bearing.
    window = (i_arr > centre - 8 * sigma_idx) & (i_arr < centre + 8 * sigma_idx) & valid
    # The Gaussian-vs-exact-kernel agreement is ~1e-3, set by the per-
    # streamtube sigma_V variation across bins j_in..j_in+8*sigma.
    assert_allclose(cout[window], expected[window], atol=2e-3, rtol=0)
    # Stronger: peak amplitude matches within 1%, location to nearest bin.
    peak_idx = int(np.argmax(cout[valid]))
    expected_peak_idx = int(np.argmax(expected[valid]))
    assert abs(peak_idx - expected_peak_idx) <= 1
    assert abs(cout[valid][peak_idx] - expected[valid][expected_peak_idx]) < 1e-2


def test_delta_input_multipv_matches_moment_averaged_gaussian():
    """Multi-PV delta pulse matches the moment-averaged-sigma Gaussian.

    For ``aquifer_pore_volumes = [V_1, V_2, V_3]`` and constant Q, the
    fast operator builds:

      1. A per-PV-averaged advection matrix ``w_adv`` that maps the
         single delta cin to three sub-spikes at j_in + R*V_k/(Q*dt).
      2. A V-smoothing operator with **moment-averaged** sigma_V

             sigma_V_eff = (R/L) * sqrt(2*D_m*tau_bar/V_bar * E[V^3]
                                         + 2*alpha_L*L * E[V^2])

         applied uniformly to the sub-spike pattern.

    The expected cout is therefore the sum of three Gaussians of width
    sigma_V_eff, each weighted 1/3, centred at the three breakthrough
    positions. Mutating the moment placement in
    :func:`gwtransport.diffusion_fast._compute_sigma_v` (e.g.
    ``ev3/V_bar → ev2``) shifts sigma_V_eff by ~9% on
    ``[600, 1000, 1400]`` and the per-bin discrepancy exceeds atol.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    flow_rate = 100.0
    flow = np.full(n, flow_rate)
    d_m = 2.0
    alpha_l = 0.0
    apvs = np.array([600.0, 1000.0, 1400.0])
    streamline_length = 30.0
    retardation = 1.0

    j_in = 60
    cin = np.zeros(n)
    cin[j_in] = 1.0

    with warn_module.catch_warnings():
        warn_module.simplefilter("ignore")
        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=apvs,
            mean_streamline_length=streamline_length,
            mean_molecular_diffusivity=d_m,
            mean_longitudinal_dispersivity=alpha_l,
            retardation_factor=retardation,
            suppress_dispersion_warning=True,
            flow_out=flow,
        )

    v_bar = float(np.mean(apvs))
    ev2 = float(np.mean(apvs**2))
    ev3 = float(np.mean(apvs**3))
    tau_bar = retardation * v_bar / flow_rate
    var_numerator = 2.0 * d_m * tau_bar / v_bar * ev3 + 2.0 * alpha_l * streamline_length * ev2
    sigma_v_eff = (retardation / streamline_length) * np.sqrt(var_numerator)
    sigma_idx = sigma_v_eff / (flow_rate * 1.0)

    # Three breakthrough centres
    centres = j_in + retardation * apvs / flow_rate

    i_arr = np.arange(n)
    sqrt2 = np.sqrt(2.0)
    expected = np.zeros(n)
    for centre in centres:
        expected += (
            (1.0 / len(apvs))
            * 0.5
            * (erf((i_arr - centre + 0.5) / (sigma_idx * sqrt2)) - erf((i_arr - centre - 0.5) / (sigma_idx * sqrt2)))
        )

    valid = ~np.isnan(cout)
    # Window spans all three breakthroughs with ample margin
    window = (i_arr > centres.min() - 6 * sigma_idx) & (i_arr < centres.max() + 6 * sigma_idx) & valid
    # Mutating ev3/V_bar -> ev2 shifts sigma_v_eff by ~9% on this PV array.
    # The resulting per-bin discrepancy is bounded above 5e-3 across the
    # window; atol=3e-3 keeps headroom under correct moments.
    assert_allclose(cout[window], expected[window], atol=3e-3, rtol=0)


# =============================================================================
# Direct unit tests for V-coordinate smoothing helpers
# =============================================================================


def test_build_rebin_matrix_v_mass_preserving():
    """``sum_i dV_dst[i] * R[i, k] == dV_src[k]`` to machine precision.

    This is the rebin-matrix V-mass-preservation invariant documented in
    :func:`gwtransport.diffusion_fast._build_rebin_matrix`. Without this,
    the V-coordinate smoothing operator drifts mass at every resampling step.
    """
    rng = np.random.default_rng(7)
    # Non-uniform source: 30 bins of varying width
    dv_src = rng.uniform(0.5, 3.0, 30)
    v_src = np.concatenate(([0.0], np.cumsum(dv_src)))
    # Destination: different non-uniform spacing covering same range
    dv_dst = rng.uniform(0.5, 3.0, 50)
    v_dst = np.concatenate(([0.0], np.cumsum(dv_dst)))
    v_dst *= v_src[-1] / v_dst[-1]
    dv_dst = np.diff(v_dst)

    r = _build_rebin_matrix(v_src_edges=v_src, v_dst_edges=v_dst).toarray()
    # V-weighted column sum should equal dV_src
    col_v_sums = (dv_dst[:, None] * r).sum(axis=0)
    assert_allclose(col_v_sums, np.diff(v_src), atol=1e-14)
    # Row sums should be exactly 1 (every dst bin fully covered)
    assert_allclose(r.sum(axis=1), 1.0, atol=1e-14)


def test_build_v_smooth_matrix_row_sums_to_one_for_random_grid():
    """``M @ ones == ones`` for a random non-uniform V-grid with random sigma_V.

    The V-smoothing matrix ``M = R_from @ G_uniform @ R_to`` should be
    row-stochastic by construction: every operator in the composition has
    row sums = 1, so a constant signal in is a constant signal out.
    """
    rng = np.random.default_rng(11)
    n = 80
    # Non-uniform V-edges: small base spacing perturbed by 30%
    dv = rng.uniform(0.8, 1.4, n)
    v_edges = np.concatenate(([0.0], np.cumsum(dv)))
    # Random per-bin sigma_V comparable to the bin width
    sigma_v = rng.uniform(0.3, 1.5, n) * np.mean(dv)

    m = _build_v_smooth_matrix(
        v_edges=v_edges,
        sigma_v_per_bin=sigma_v,
        refinement=4,
        asymptotic_cutoff_sigma=4.0,
    ).toarray()
    row_sums = m.sum(axis=1)
    assert_allclose(row_sums, 1.0, atol=1e-12)


def test_build_v_smooth_matrix_identity_for_zero_sigma():
    """All-zero sigma -> identity matrix to machine precision."""
    n = 25
    v_edges = np.linspace(0.0, 100.0, n + 1)
    sigma_v = np.zeros(n)
    m = _build_v_smooth_matrix(v_edges=v_edges, sigma_v_per_bin=sigma_v).toarray()
    assert_allclose(m, np.eye(n), atol=0.0)


def test_build_v_smooth_matrix_v_weighted_col_sums_alpha_l():
    """For sigma_V independent of V (pure alpha_L regime), the V-smoothing
    matrix is V-weighted-column-stochastic to machine precision.

    This is the structural invariant that makes the pure-alpha_L
    mass-conservation test land at ~1e-12 in the i2e function.
    """
    n = 60
    rng = np.random.default_rng(13)
    dv = rng.uniform(0.8, 1.2, n)
    v_edges = np.concatenate(([0.0], np.cumsum(dv)))
    # Constant sigma_V (alpha_L only contribution)
    sigma_v = np.full(n, 0.5 * np.mean(dv))

    m = _build_v_smooth_matrix(v_edges=v_edges, sigma_v_per_bin=sigma_v, refinement=4).toarray()
    dv_arr = np.diff(v_edges)
    # V-weighted column sum
    col_v_sums = (dv_arr[:, None] * m).sum(axis=0)
    # The interior is exact; the boundary picks up edge truncation when the
    # uniform-V grid spans the original v_edges range tightly. Restrict to
    # interior columns where the kernel is fully contained.
    interior = slice(10, n - 10)
    assert_allclose(col_v_sums[interior], dv_arr[interior], atol=1e-12)
