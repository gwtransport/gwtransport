import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import erf

from gwtransport.advection import infiltration_to_extraction as advection_i2e
from gwtransport.diffusion import infiltration_to_extraction as diffusion_exact
from gwtransport.diffusion_fast import (
    _build_gaussian_matrix,
    compute_scaled_sigma_array,
    convolve_diffusion,
    create_example_data,
    extraction_to_infiltration,
    infiltration_to_extraction,
)

# =============================================================================
# Machine-precision tests for convolve_diffusion (CDF-integrated kernel)
#
# The CDF-integrated kernel computes the exact bin-averaged convolution of
# piecewise-constant input with a Gaussian. This means:
# - Step function + CDF kernel = analytical erf (machine precision)
# - Linear signal + CDF kernel = same linear signal (machine precision)
# - Constant signal + CDF kernel = same constant (machine precision)
# =============================================================================


def test_convolve_constant_signal_preservation():
    """Constant input must be preserved exactly (row sums = 1)."""
    n = 200
    signal = np.full(n, 42.0)
    rng = np.random.default_rng(0)
    sigma_array = rng.uniform(0.5, 10.0, n)

    result = convolve_diffusion(input_signal=signal, sigma_array=sigma_array, asymptotic_cutoff_sigma=10.0)
    assert_allclose(result, 42.0, atol=1e-13)


def test_convolve_linear_signal_preservation():
    """Symmetric kernel preserves linear signals exactly (interior points).

    For a symmetric Gaussian kernel centered at position i, the weighted average
    of a linear function f(j) = a*j + b is exactly f(i), because:
    sum_j G[i,j] * (a*j + b) = a * sum_j G[i,j] * j + b * sum_j G[i,j]
    = a * i + b (by symmetry of G and normalization).
    """
    n = 200
    signal = 3.0 * np.arange(n, dtype=float) + 7.0
    sigma = 5.0
    sigma_array = np.full(n, sigma)

    result = convolve_diffusion(input_signal=signal, sigma_array=sigma_array, asymptotic_cutoff_sigma=10.0)

    # Interior points (avoid boundary clipping effects)
    margin = int(10 * sigma)
    interior = slice(margin, n - margin)
    assert_allclose(result[interior], signal[interior], atol=1e-12)


def test_convolve_step_function_matches_erf():
    """CDF kernel applied to step function gives erf at machine precision.

    For input[j] = 1 if j >= j0 else 0, the CDF kernel output telescopes to:
        out[i] = 0.5 * (1 + erf((i - j0 + 0.5) / (sqrt(2) * sigma)))
    The +0.5 arises because the step edge is at the LEFT boundary of bin j0.
    """
    n = 500
    j0 = n // 2
    x = np.arange(n, dtype=float)
    initial = np.where(x >= j0, 1.0, 0.0)

    for sigma in [0.5, 1.0, 2.0, 5.0, 10.0]:
        sigma_array = np.full(n, sigma)
        result = convolve_diffusion(input_signal=initial, sigma_array=sigma_array, asymptotic_cutoff_sigma=10.0)

        # Analytical CDF result (step edge at j0 - 0.5)
        expected = 0.5 * (1 + erf((x - j0 + 0.5) / (np.sqrt(2) * sigma)))

        # Test interior (avoid boundary clipping effects)
        margin = max(1, int(10 * sigma))
        interior = slice(margin, n - margin)
        assert_allclose(
            result[interior],
            expected[interior],
            atol=1e-14,
            err_msg=f"Failed for sigma={sigma}",
        )


def test_convolve_zero_sigma_identity():
    """Zero sigma returns input unchanged."""
    x = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * x)
    sigma_array = np.zeros_like(x)

    result = convolve_diffusion(input_signal=signal, sigma_array=sigma_array)
    assert_allclose(result, signal, atol=0.0)


def test_convolve_mixed_zero_and_nonzero_sigma():
    """Points with sigma=0 are unchanged; others are smoothed."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    sigma_array = np.array([0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0])

    result = convolve_diffusion(input_signal=signal, sigma_array=sigma_array)

    # Zero-sigma positions must be exactly preserved
    zero_mask = sigma_array == 0
    assert_allclose(result[zero_mask], signal[zero_mask], atol=0.0)

    # All values must be finite
    assert np.all(np.isfinite(result))


def test_convolve_input_validation():
    """Mismatched input lengths raise ValueError."""
    signal = np.linspace(0, 1, 100)
    sigma_array = np.zeros(101)
    with pytest.raises(ValueError, match="same length"):
        convolve_diffusion(input_signal=signal, sigma_array=sigma_array)


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


@pytest.mark.parametrize("nx", [500, 2000, 5000])
def test_gaussian_pulse_convergence(nx):
    """CDF kernel of discretized Gaussian converges to analytical solution.

    The error is O(dx^2) from bin-averaging the smooth initial condition,
    not from the kernel itself.
    """
    domain_length = 10.0
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    diffusivity = 0.1
    dt = 0.01
    sigma = np.sqrt(2 * diffusivity * dt) / dx

    x0, amplitude, width = 0.0, 1.0, 0.5
    initial = amplitude * np.exp(-((x - x0) ** 2) / (2 * width**2))
    sigma_array = np.full_like(x, sigma)

    numerical = convolve_diffusion(input_signal=initial, sigma_array=sigma_array, asymptotic_cutoff_sigma=10.0)

    new_width = np.sqrt(width**2 + 2 * diffusivity * dt)
    analytical = amplitude * width / new_width * np.exp(-((x - x0) ** 2) / (2 * new_width**2))

    # Error should scale as O(dx^2)
    interior = slice(nx // 10, 9 * nx // 10)
    max_err = np.max(np.abs(numerical[interior] - analytical[interior]))

    # Tolerance scales with dx^2: at nx=500, dx=0.02, tol≈7e-5
    expected_tol = 2.0 * dx**2
    assert max_err < expected_tol, f"nx={nx}: max_err={max_err:.2e} > tol={expected_tol:.2e}"


@pytest.mark.parametrize("nx", [500, 2000, 5000])
def test_delta_function_convergence(nx):
    """CDF kernel of discretized delta converges to analytical solution.

    Delta function is poorly resolved at any finite resolution.
    Error is O(dx) because the delta input is a single bin.
    """
    domain_length = 10.0
    x = np.linspace(-domain_length / 2, domain_length / 2, nx)
    dx = x[1] - x[0]

    diffusivity = 0.2
    dt = 0.01
    sigma = np.sqrt(2 * diffusivity * dt) / dx

    x0 = dx / 2
    initial = np.zeros_like(x)
    center_idx = np.argmin(np.abs(x - x0))
    initial[center_idx] = 1.0 / dx

    sigma_array = np.full_like(x, sigma)
    numerical = convolve_diffusion(input_signal=initial, sigma_array=sigma_array, asymptotic_cutoff_sigma=10.0)
    analytical = 1.0 / np.sqrt(4 * np.pi * diffusivity * dt) * np.exp(-((x - x0) ** 2) / (4 * diffusivity * dt))

    interior = slice(nx // 10, 9 * nx // 10)
    max_err = np.max(np.abs(numerical[interior] - analytical[interior]))

    # Tolerance scales with dx: at nx=500, dx=0.02, tol≈0.5
    expected_tol = 25.0 * dx
    assert max_err < expected_tol, f"nx={nx}: max_err={max_err:.2e} > tol={expected_tol:.2e}"


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
        streamline_length=np.array([80.0]),
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
        streamline_length=np.array([80.0]),
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
        streamline_length=np.array([80.0]),
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
    )

    valid = ~np.isnan(cout)
    # Forward matrix is non-negative with row sums = 1, so output is a
    # convex combination of input values.
    assert np.all(cout[valid] >= np.min(cin) - 1e-10)
    assert np.all(cout[valid] <= np.max(cin) + 1e-10)


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
        streamline_length=np.array([80.0, 100.0, 120.0]),
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
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
        streamline_length=np.array([80.0]),
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
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
        streamline_length=np.array([80.0]),
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
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
        streamline_length=np.array([80.0]),
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
        streamline_length=np.array([80.0]),
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
    )

    valid = ~np.isnan(cout)
    assert np.any(valid)
    # Under variable flow, constant preservation holds because the combined
    # matrix (W_adv @ G) has row sums ≤ 1 for valid rows, and cin is constant.
    assert_allclose(cout[valid], 10.0, atol=1e-10)


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
        streamline_length=np.array([80.0]),
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
    )

    valid = ~np.isnan(cin)
    assert np.sum(valid) > 50
    # Tikhonov with regularization_strength=1e-10 limits precision
    assert_allclose(cin[valid], 7.0, atol=1e-5)


def test_round_trip():
    """Forward then inverse recovers the original signal."""
    tedges, cout_tedges, flow = _make_transport_data(n_days=200)
    n_days = len(flow)
    cin_original = np.sin(np.linspace(0, 2 * np.pi, n_days)) + 5.0

    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
        "streamline_length": np.array([80.0]),
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
    assert_allclose(cin_recovered[valid], cin_original[valid], atol=0.05)


# =============================================================================
# Tests for compute_scaled_sigma_array
# =============================================================================


def test_compute_scaled_sigma_array_constant_flow():
    """With constant flow, sigma should be uniform."""
    n_days = 50
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    flow = np.full(n_days, 100.0)

    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=1000.0,
        streamline_length=80.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
    )

    assert len(sigma_array) == n_days
    assert np.all(np.isfinite(sigma_array))
    assert np.all(sigma_array >= 0.0)
    # Constant flow → uniform sigma (slight variation from residence time edge effects)
    assert np.std(sigma_array) < np.mean(sigma_array) * 0.1


def test_compute_scaled_sigma_array_variable_flow():
    """Variable flow produces variable sigma."""
    n_days = 50
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    t = np.arange(n_days)
    flow = 100.0 * (1.0 + 0.5 * np.sin(2 * np.pi * t / 20))

    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
        streamline_length=80.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
        retardation_factor=2.0,
    )

    assert len(sigma_array) == n_days
    assert np.all(np.isfinite(sigma_array))
    assert np.all(sigma_array >= 0.0)
    assert np.std(sigma_array) > 0.0


def test_compute_scaled_sigma_array_with_nan_in_residence_time():
    """NaN residence times are interpolated, producing finite sigma."""
    n_days = 50
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    flow = np.full(n_days, 100.0)
    flow[10:15] = 1e-10

    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=1000.0,
        streamline_length=80.0,
        molecular_diffusivity=0.03,
        longitudinal_dispersivity=0.0,
    )

    assert len(sigma_array) == n_days
    assert np.all(np.isfinite(sigma_array))
    assert np.all(sigma_array >= 0.0)


def test_compute_scaled_sigma_array_clipping():
    """Extreme sigma values are clipped to [0, 100]."""
    n_days = 100
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    flow = np.full(n_days, 10.0)

    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=100.0,
        streamline_length=80.0,
        molecular_diffusivity=10.0,
        longitudinal_dispersivity=0.0,
        retardation_factor=5.0,
    )

    assert np.all(sigma_array <= 100.0)
    assert np.all(sigma_array >= 0.0)


@pytest.mark.parametrize(
    ("molecular_diffusivity", "retardation"),
    [(0.01, 1.0), (0.05, 2.0), (0.10, 1.5)],
)
def test_compute_scaled_sigma_array_parametrized(molecular_diffusivity, retardation):
    """Various parameter combinations produce positive finite sigma."""
    n_days = 50
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    flow = np.full(n_days, 100.0)

    sigma_array = compute_scaled_sigma_array(
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=1000.0,
        streamline_length=100.0,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=0.0,
        retardation_factor=retardation,
    )

    assert len(sigma_array) == n_days
    assert np.all(np.isfinite(sigma_array))
    assert np.mean(sigma_array) > 0.0


def test_compute_scaled_sigma_array_dispersivity_increases_sigma():
    """Adding longitudinal dispersivity increases sigma."""
    n_days = 50
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    flow = np.full(n_days, 100.0)

    kwargs = {
        "flow": flow,
        "tedges": tedges,
        "aquifer_pore_volume": 1000.0,
        "streamline_length": 80.0,
        "molecular_diffusivity": 0.03,
    }

    sigma_no_disp = compute_scaled_sigma_array(**kwargs, longitudinal_dispersivity=0.0)
    sigma_with_disp = compute_scaled_sigma_array(**kwargs, longitudinal_dispersivity=1.0)

    assert np.all(sigma_with_disp >= sigma_no_disp - 1e-14)
    assert np.mean(sigma_with_disp) > np.mean(sigma_no_disp)


# =============================================================================
# Tests for create_example_data
# =============================================================================


def test_create_example_data_basic():
    """Output shapes and value ranges are correct."""
    x, signal, sigma_array, dt = create_example_data(seed=42)

    assert len(x) == 1000
    assert len(signal) == 1000
    assert len(sigma_array) == 1000
    assert len(dt) == 1000

    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(signal))
    assert np.all(np.isfinite(sigma_array))
    assert np.all(sigma_array >= 0.0)
    assert np.all(dt > 0.0)


def test_create_example_data_reproducible():
    """Same seed gives identical output."""
    _, signal1, _, _ = create_example_data(seed=42)
    _, signal2, _, _ = create_example_data(seed=42)
    assert_allclose(signal1, signal2, atol=0.0)


def test_create_example_data_custom_parameters():
    """Custom parameters are respected."""
    nx = 500
    domain_length = 20.0
    x, _, _, _ = create_example_data(nx=nx, domain_length=domain_length, diffusivity=0.5, seed=0)

    assert len(x) == nx
    assert_allclose(x[0], 0.0, atol=0.0)
    assert_allclose(x[-1], domain_length, rtol=1e-14)


def test_create_example_data_different_sizes():
    """Works for various grid sizes."""
    for nx in [100, 500, 2000]:
        x, signal, sigma_array, dt = create_example_data(nx=nx, seed=0)
        assert len(x) == nx


# =============================================================================
# Integration test: diffusion_fast vs diffusion on same extraction grid
# =============================================================================


def test_diffusion_fast_produces_extraction_grid_output():
    """Both modules return results on the same extraction time grid.

    Under constant flow, the fast (Gaussian) and exact (erf) methods should
    agree reasonably well, since the Gaussian approximation is accurate when
    flow and time steps are constant.
    """
    n_days = 200
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    cout_tedges = tedges.copy()

    cin = np.zeros(n_days)
    cin[30:] = 1.0
    flow = np.full(n_days, 100.0)

    kwargs = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volumes": np.array([500.0]),
        "streamline_length": np.array([80.0]),
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 0.0,
    }

    cout_fast = infiltration_to_extraction(**kwargs)
    cout_exact = diffusion_exact(**kwargs)

    # Same length (same extraction grid)
    assert len(cout_fast) == len(cout_exact) == n_days

    both_valid = ~np.isnan(cout_fast) & ~np.isnan(cout_exact)
    assert np.sum(both_valid) > 50

    assert np.all(cout_fast[~np.isnan(cout_fast)] >= -1e-10)
    assert np.all(cout_exact[~np.isnan(cout_exact)] >= -1e-10)

    # Approximate agreement under constant flow
    assert_allclose(cout_fast[both_valid], cout_exact[both_valid], rtol=0.2, atol=0.05)


if __name__ == "__main__":
    pytest.main([__file__])
