import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import integrate, stats

from gwtransport.gamma import bins as gamma_bins
from gwtransport.logremoval import (
    decay_rate_to_log10_decay_rate,
    gamma_cdf,
    gamma_find_flow_for_target_mean,
    gamma_mean,
    gamma_pdf,
    log10_decay_rate_to_decay_rate,
    parallel_mean,
    residence_time_to_log_removal,
)
from gwtransport.residence_time import residence_time_series as compute_residence_time


def test_single_flow():
    """Test with a single flow path - result should be the same as input."""
    assert parallel_mean(log_removals=[4.0]) == 4.0
    assert parallel_mean(log_removals=np.array([3.5])) == 3.5
    # With explicit flow fraction
    assert parallel_mean(log_removals=[4.0], flow_fractions=[1.0]) == 4.0


def test_identical_flows_equal_distribution():
    """Test with multiple identical flows - result should match the inputs."""
    assert parallel_mean(log_removals=[3.0, 3.0]) == 3.0
    assert parallel_mean(log_removals=[2.0, 2.0, 2.0, 2.0]) == 2.0


def test_different_flows_equal_distribution():
    """Test with flows having different removal values with equal distribution."""
    # Test case: Two flows with log removals 3 and 5
    result = parallel_mean(log_removals=[3, 4, 5])
    expected = -np.log10((10 ** (-3.0) + 10 ** (-4.0) + 10 ** (-5.0)) / 3)
    assert_allclose(result, expected, rtol=1e-15)
    assert_allclose(result, 3.431798275933005, rtol=1e-15)  # example in docstring


def test_array_inputs_equal_distribution():
    """Test with numpy array inputs for equal distribution."""
    # NumPy arrays as input
    result = parallel_mean(log_removals=np.array([3.0, 4.0, 5.0]))
    expected = -np.log10((10 ** (-3.0) + 10 ** (-4.0) + 10 ** (-5.0)) / 3)
    assert_allclose(result, expected, rtol=1e-15)


def test_special_values_equal_distribution():
    """Test with special values like zero and large numbers with equal distribution."""
    # With log removal of 0 (no removal)
    result = parallel_mean(log_removals=[0.0, 4.0])
    expected = -np.log10((1.0 + 10 ** (-4.0)) / 2)
    assert_allclose(result, expected, rtol=1e-15)

    # With very large log removal (effectively complete removal)
    # Using a large number instead of infinity to avoid numerical issues
    result = parallel_mean(log_removals=[20.0, 3.0])
    # The 10^-20 term is effectively zero
    expected = -np.log10((10 ** (-20.0) + 10 ** (-3.0)) / 2)
    assert_allclose(result, expected, rtol=1e-15)


def test_float_precision_equal_distribution():
    """Test handling of floating point precision with equal distribution."""
    # Testing with values that require good floating point handling
    result = parallel_mean(log_removals=[9.999, 9.998])
    expected = -np.log10((10 ** (-9.999) + 10 ** (-9.998)) / 2)
    assert_allclose(result, expected, rtol=1e-15)


def test_equal_weights_explicit():
    """Test with explicitly provided equal weights - should match the implicit equal weights."""
    log_removals = [3.0, 4.0, 5.0]
    weights = [1 / 3, 1 / 3, 1 / 3]

    weighted_result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    unweighted_result = parallel_mean(log_removals=log_removals)

    assert_allclose(weighted_result, unweighted_result, rtol=1e-15)


def test_weighted_flows():
    """Test with different weights for each flow."""
    # Test case: Two flows with different weights
    log_removals = [3.0, 5.0]
    weights = [0.7, 0.3]

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    expected = -np.log10(0.7 * 10 ** (-3.0) + 0.3 * 10 ** (-5.0))
    assert_allclose(result, expected, rtol=1e-15)
    assert_allclose(result, 3.153044674980176, rtol=1e-15)  # example in docstring

    # Test case: Three flows with different weights
    log_removals = [2.0, 4.0, 6.0]
    weights = [0.5, 0.3, 0.2]

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    expected = -np.log10(0.5 * 10 ** (-2.0) + 0.3 * 10 ** (-4.0) + 0.2 * 10 ** (-6.0))
    assert_allclose(result, expected, rtol=1e-15)


def test_weight_sum_behavior():
    """Test behavior when weights don't sum to 1.0 raises ValueError."""
    log_removals = [3.0, 4.0]

    with pytest.raises(ValueError, match=r"flow_fractions must sum to 1\.0"):
        parallel_mean(log_removals=log_removals, flow_fractions=[0.7, 0.4])  # Sum > 1

    with pytest.raises(ValueError, match=r"flow_fractions must sum to 1\.0"):
        parallel_mean(log_removals=log_removals, flow_fractions=[0.7, 0.2])  # Sum < 1


def test_weighted_array_inputs():
    """Test with numpy array inputs for weights."""
    log_removals = np.array([3.0, 5.0])
    weights = np.array([0.6, 0.4])

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    expected = -np.log10(0.6 * 10 ** (-3.0) + 0.4 * 10 ** (-5.0))
    assert_allclose(result, expected, rtol=1e-15)


def test_extreme_weights():
    """Test with extreme weight distributions."""
    # One weight is almost 1.0, others are tiny
    log_removals = [3.0, 5.0, 6.0]
    weights = [0.999, 0.0005, 0.0005]

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    # Compare to the analytical value rather than just "close to 3.0".
    expected = -np.log10(0.999 * 10 ** (-3.0) + 0.0005 * 10 ** (-5.0) + 0.0005 * 10 ** (-6.0))
    assert_allclose(result, expected, rtol=1e-15)

    # One weight is exactly 1.0, others are exactly 0.0
    log_removals = [4.0, 5.0, 6.0]
    weights = [1.0, 0.0, 0.0]

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    assert result == 4.0


@pytest.mark.parametrize(
    ("apv_alpha", "apv_beta", "log10_decay_rate", "target_mean"),
    [
        (1.5, 5.0, 0.1, 0.5),  # low alpha, small system
        (3.0, 10.0, 0.2, 1.5),  # moderate parameters
        (10.0, 5.0, 0.1, 3.0),  # high alpha, narrow distribution
        (2.0, 20.0, 0.5, 2.0),  # high removal rate
        (0.8, 100.0, 0.01, 0.3),  # alpha < 1, very heterogeneous
    ],
)
def test_gamma_find_flow_for_target_mean(apv_alpha, apv_beta, log10_decay_rate, target_mean):
    """Test that gamma_find_flow_for_target_mean inverts gamma_mean correctly."""
    required_flow = gamma_find_flow_for_target_mean(
        target_mean=target_mean, apv_alpha=apv_alpha, apv_beta=apv_beta, log10_decay_rate=log10_decay_rate
    )

    # Verify the round-trip: flow -> residence time params -> gamma_mean == target.
    # apv_loc=0 takes the closed-form branch, so the round-trip is bit-exact.
    rt_alpha = apv_alpha
    rt_beta = apv_beta / required_flow
    verification_mean = gamma_mean(rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)
    assert_allclose(verification_mean, target_mean, rtol=1e-15)


def test_axis_parameter_2d_arrays():
    """Test axis parameter with 2D arrays."""
    # Create a 2D array with known values
    log_removals_2d = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])

    # Test axis=1 (along columns)
    result_axis1 = parallel_mean(log_removals=log_removals_2d, axis=1)

    # Expected results: parallel_mean for each row
    expected_row0 = parallel_mean(log_removals=[3.0, 4.0, 5.0])
    expected_row1 = parallel_mean(log_removals=[2.0, 3.0, 4.0])
    expected_axis1 = np.array([expected_row0, expected_row1])

    assert_allclose(result_axis1, expected_axis1, rtol=1e-15)

    # Test axis=0 (along rows)
    result_axis0 = parallel_mean(log_removals=log_removals_2d, axis=0)

    # Expected results: parallel_mean for each column
    expected_col0 = parallel_mean(log_removals=[3.0, 2.0])
    expected_col1 = parallel_mean(log_removals=[4.0, 3.0])
    expected_col2 = parallel_mean(log_removals=[5.0, 4.0])
    expected_axis0 = np.array([expected_col0, expected_col1, expected_col2])

    assert_allclose(result_axis0, expected_axis0, rtol=1e-15)


def test_axis_parameter_3d_arrays():
    """Test axis parameter with 3D arrays."""
    # Create a 3D array
    log_removals_3d = np.array([[[3.0, 4.0], [5.0, 2.0]], [[1.0, 6.0], [2.0, 3.0]]])

    # Test axis=2 (innermost dimension)
    result_axis2 = parallel_mean(log_removals=log_removals_3d, axis=2)

    # Expected results: parallel_mean for each pair along axis 2
    expected_00 = parallel_mean(log_removals=[3.0, 4.0])
    expected_01 = parallel_mean(log_removals=[5.0, 2.0])
    expected_10 = parallel_mean(log_removals=[1.0, 6.0])
    expected_11 = parallel_mean(log_removals=[2.0, 3.0])
    expected_axis2 = np.array([[expected_00, expected_01], [expected_10, expected_11]])

    assert_allclose(result_axis2, expected_axis2, rtol=1e-15)


def test_axis_parameter_with_flow_fractions():
    """Test axis parameter with explicit flow fractions."""
    log_removals_2d = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])
    flow_fractions_2d = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2]])

    # Test with axis=1
    result = parallel_mean(log_removals=log_removals_2d, flow_fractions=flow_fractions_2d, axis=1)

    # Expected results: weighted parallel_mean for each row
    expected_row0 = parallel_mean(log_removals=[3.0, 4.0, 5.0], flow_fractions=[0.5, 0.3, 0.2])
    expected_row1 = parallel_mean(log_removals=[2.0, 3.0, 4.0], flow_fractions=[0.6, 0.2, 0.2])
    expected = np.array([expected_row0, expected_row1])

    assert_allclose(result, expected, rtol=1e-15)


def test_negative_axis():
    """Test negative axis indexing."""
    log_removals_2d = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])

    # axis=-1 should be equivalent to axis=1 for 2D array
    result_neg1 = parallel_mean(log_removals=log_removals_2d, axis=-1)
    result_pos1 = parallel_mean(log_removals=log_removals_2d, axis=1)

    assert_allclose(result_neg1, result_pos1, rtol=1e-15)

    # axis=-2 should be equivalent to axis=0 for 2D array
    result_neg2 = parallel_mean(log_removals=log_removals_2d, axis=-2)
    result_pos0 = parallel_mean(log_removals=log_removals_2d, axis=0)

    assert_allclose(result_neg2, result_pos0, rtol=1e-15)


@pytest.mark.parametrize(
    ("residence_times", "log10_decay_rate", "expected"),
    [
        # Scalars covering zero/positive/large t and zero/varied mu.
        (0.0, 2.0, 0.0),
        (10.0, 0.2, 2.0),
        (20.0, 0.15, 3.0),
        (10.0, 0.1, 1.0),
        (10.0, 0.3, 3.0),
        (5.0, 0.3, 1.5),
        (1.0e6, 0.1, 1.0e5),
        (100.0, 0.0, 0.0),
        # 1D arrays (numpy and list inputs).
        (np.array([10.0, 20.0, 50.0]), 0.2, np.array([2.0, 4.0, 10.0])),
        ([10.0, 20.0, 50.0], 0.2, np.array([2.0, 4.0, 10.0])),
        (np.array([5.0, 25.0, 50.0]), 0.15, np.array([0.75, 3.75, 7.5])),
        (np.array([2.5, 7.3, 15.8, 33.2]), 0.23, 0.23 * np.array([2.5, 7.3, 15.8, 33.2])),
        # 2D / 3D arrays.
        (np.array([[10.0, 20.0], [30.0, 40.0]]), 0.1, np.array([[1.0, 2.0], [3.0, 4.0]])),
        (
            np.array([[[10.0, 20.0]], [[30.0, 40.0]]]),
            0.1,
            np.array([[[1.0, 2.0]], [[3.0, 4.0]]]),
        ),
    ],
)
def test_residence_time_to_log_removal_formula(residence_times, log10_decay_rate, expected):
    """LR = mu * t across scalar / list / 1D / 2D / 3D inputs at machine precision."""
    result = residence_time_to_log_removal(residence_times=residence_times, log10_decay_rate=log10_decay_rate)
    assert_allclose(result, expected, rtol=1e-15)


def test_residence_time_to_log_removal_shape_preservation():
    """Test that residence_time_to_log_removal preserves input array shape."""
    # Test 1D
    residence_times_1d = np.array([10.0, 20.0, 50.0])
    result_1d = residence_time_to_log_removal(residence_times=residence_times_1d, log10_decay_rate=0.1)
    assert result_1d.shape == residence_times_1d.shape

    # Test 2D
    residence_times_2d = np.array([[10.0, 20.0], [30.0, 40.0]])
    result_2d = residence_time_to_log_removal(residence_times=residence_times_2d, log10_decay_rate=0.1)
    assert result_2d.shape == residence_times_2d.shape

    # Test 3D
    residence_times_3d = np.array([[[10.0, 20.0]], [[30.0, 40.0]]])
    result_3d = residence_time_to_log_removal(residence_times=residence_times_3d, log10_decay_rate=0.1)
    assert result_3d.shape == residence_times_3d.shape


@pytest.mark.parametrize(
    ("mu", "lambda_"),
    [
        (0.0, 0.0),
        (1.0, np.log(10.0)),
        (0.2, 0.2 * np.log(10.0)),
        (0.5 / np.log(10.0), 0.5),
    ],
)
def test_decay_rate_conversion_round_trip(mu, lambda_):
    """``mu`` <-> ``lambda`` conversions match the ln(10) factor and round-trip exactly."""
    assert_allclose(log10_decay_rate_to_decay_rate(mu), lambda_, rtol=1e-15)
    assert_allclose(decay_rate_to_log10_decay_rate(lambda_), mu, rtol=1e-15)
    assert_allclose(decay_rate_to_log10_decay_rate(log10_decay_rate_to_decay_rate(mu)), mu, rtol=1e-15)


def test_gamma_pdf_integrates_to_one():
    """Test that the gamma PDF integrates to 1."""
    rt_alpha = 3.0
    rt_beta = 10.0
    log10_decay_rate = 0.2

    result, _ = integrate.quad(
        lambda r: gamma_pdf(r=r, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate),
        0,
        np.inf,
    )
    # scipy.integrate.quad's adaptive Gauss-Kronrod quadrature only achieves
    # finite accuracy on an infinite interval; tighter than ~1e-10 is unreliable.
    assert_allclose(result, 1.0, rtol=1e-10)  # quad-limited, do not tighten


def test_gamma_cdf_approaches_one():
    """Test that the gamma CDF approaches 1 for large r values."""
    rt_alpha = 3.0
    rt_beta = 10.0
    log10_decay_rate = 0.2

    # CDF at a very large r should be close to 1
    result = gamma_cdf(r=1000.0, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)
    assert_allclose(result, 1.0, atol=1e-10)

    # CDF at 0 should be 0
    result = gamma_cdf(r=0.0, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)
    assert_allclose(result, 0.0, atol=1e-10)


def test_gamma_cdf_is_scaled_gamma():
    """Test that the log removal CDF matches a scaled gamma distribution.

    R = mu * T, so R ~ Gamma(rt_alpha, scale=mu*rt_beta). This pins the scale factor
    mu * rt_beta: dropping the decay-rate factor (scale=mu*beta -> beta) would shift
    the CDF and fail this assertion. Mirrors ``test_gamma_pdf_is_scaled_gamma``.
    """
    rt_alpha = 4.0
    rt_beta = 5.0
    log10_decay_rate = 0.3

    r_values = np.linspace(0.1, 20.0, 50)
    cdf_values = gamma_cdf(r=r_values, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)

    expected = stats.gamma.cdf(r_values, a=rt_alpha, scale=log10_decay_rate * rt_beta)
    assert_allclose(cdf_values, expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("rt_alpha", "rt_beta", "log10_decay_rate"),
    [
        (1.5, 5.0, 0.1),  # low alpha
        (3.0, 10.0, 0.2),  # moderate
        (10.0, 5.0, 0.1),  # high alpha, narrow distribution
        (2.0, 20.0, 0.5),  # high removal rate
        (0.8, 100.0, 0.01),  # alpha < 1
    ],
)
def test_gamma_mean_matches_numerical_integration(rt_alpha, rt_beta, log10_decay_rate):
    """Test that gamma_mean matches -log10(E[10^(-R)]) via numerical integration."""
    analytical_mean = gamma_mean(rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)

    # Integrate E[10^(-R)] = integral of 10^(-r) * pdf(r) dr.
    # scipy.integrate.quad on a half-infinite interval reports relative error
    # around 1e-8 for these integrands; the log10 step amplifies that, so an
    # atol of 1e-8 is an honest reflection of quad's accuracy here.
    expected_decimal_reduction, _ = integrate.quad(
        lambda r: 10 ** (-r) * gamma_pdf(r=r, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate),
        0,
        np.inf,
    )
    numerical_mean = -np.log10(expected_decimal_reduction)

    assert_allclose(analytical_mean, numerical_mean, atol=1e-8)


@pytest.mark.parametrize(
    ("apv_alpha", "apv_beta", "flow", "log10_decay_rate"),
    [
        (1.5, 50.0, 10.0, 0.1),  # low alpha
        (3.0, 100.0, 10.0, 0.2),  # moderate
        (10.0, 50.0, 10.0, 0.1),  # high alpha, narrow distribution
        (2.0, 200.0, 10.0, 0.5),  # high removal rate
        (0.8, 1000.0, 10.0, 0.01),  # alpha < 1
    ],
)
def test_gamma_mean_matches_discretized_parallel_mean(apv_alpha, apv_beta, flow, log10_decay_rate):
    """Test gamma_mean matches the full pipeline: bins, residence_time, log_removal, parallel_mean.

    Uses the full pipeline:
    1. gamma.bins() to discretize aquifer pore volumes
    2. residence_time.residence_time_series() to compute residence times from pore volumes and flow
    3. residence_time_to_log_removal() to compute log removals
    4. parallel_mean() to compute effective log removal

    The result should match gamma_mean() with rt_beta = apv_beta / flow.
    """
    # Analytical effective mean
    rt_alpha = apv_alpha
    rt_beta = apv_beta / flow
    analytical = gamma_mean(rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)

    # Step 1: Discretize aquifer pore volume distribution
    b = gamma_bins(alpha=apv_alpha, beta=apv_beta, n_bins=10000)
    pore_volumes = b["expected_values"]
    flow_fractions = b["probability_mass"]

    # Step 2: Compute residence times using residence_time module
    # Create a constant flow time series long enough for the largest pore volume
    max_residence_days = pore_volumes.max() / flow * 2
    n_days = int(np.ceil(max_residence_days)) + 10
    tedges = pd.date_range("2020-01-01", periods=n_days + 1, freq="D")
    constant_flow = np.full(n_days, flow)

    # Compute residence time at a single point (midpoint, far enough from edges)
    index = pd.DatetimeIndex([tedges[0] + (tedges[-1] - tedges[0]) / 2])
    rt_array = compute_residence_time(
        flow=constant_flow,
        tedges=tedges,
        aquifer_pore_volumes=pore_volumes,
        index=index,
        direction="extraction_to_infiltration",
    )
    residence_times = rt_array[:, 0]  # shape (n_bins,)

    # Step 3: Compute log removals
    log_removals = residence_time_to_log_removal(residence_times=residence_times, log10_decay_rate=log10_decay_rate)

    # Step 4: Compute parallel mean
    discretized = parallel_mean(log_removals=log_removals, flow_fractions=flow_fractions)

    # Discretizing a gamma distribution into 10000 expected-value bins and combining
    # via parallel_mean is an O(1/n_bins) approximation of the exact MGF integral.
    # The empirical error across the parameter sweep is up to ~4e-4, with the worst
    # case at high alpha (narrow distribution), so atol=1e-3 is a tight bound.
    assert_allclose(analytical, discretized, atol=1e-3)


@pytest.mark.parametrize(
    ("rt_alpha", "rt_beta", "log10_decay_rate"),
    [
        (1.5, 5.0, 0.1),
        (3.0, 10.0, 0.2),
        (10.0, 5.0, 0.1),
        (2.0, 20.0, 0.5),
        (0.8, 100.0, 0.01),
    ],
)
def test_gamma_mean_less_than_arithmetic_mean(rt_alpha, rt_beta, log10_decay_rate):
    """Test that effective parallel mean is less than arithmetic mean.

    The parallel mean is always less than the arithmetic mean because
    short residence time paths contribute disproportionately to output
    concentration.
    """
    effective_mean = gamma_mean(rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)
    arithmetic_mean = log10_decay_rate * rt_alpha * rt_beta

    assert effective_mean < arithmetic_mean


def test_gamma_pdf_is_scaled_gamma():
    """Test that the log removal PDF matches a scaled gamma distribution."""
    rt_alpha = 4.0
    rt_beta = 5.0
    log10_decay_rate = 0.3

    r_values = np.linspace(0.1, 20.0, 50)
    pdf_values = gamma_pdf(r=r_values, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)

    # R = mu*T, so R ~ Gamma(alpha, mu*beta)
    expected = stats.gamma.pdf(r_values, a=rt_alpha, scale=log10_decay_rate * rt_beta)
    assert_allclose(pdf_values, expected, rtol=1e-12)


@pytest.mark.parametrize(
    ("alpha", "beta", "mu"),
    [
        (2.0, 5.0, 0.2),
        (10.0, 2.0, 0.1),
        (0.5, 10.0, 0.3),
        (50.0, 1.0, 0.05),
    ],
)
def test_gamma_mean_leq_arithmetic_mean(alpha, beta, mu):
    """Test that gamma_mean (parallel mean) <= arithmetic mean (Jensen's inequality).

    The arithmetic mean of log removals is mu * alpha * beta (= mu * E[T]).
    The parallel (mixed-effluent) mean is always less due to Jensen's inequality
    applied to the convex function 10^(-x).
    """
    parallel = gamma_mean(rt_alpha=alpha, rt_beta=beta, log10_decay_rate=mu)
    arithmetic = mu * alpha * beta

    assert parallel <= arithmetic + 1e-12  # Allow tiny numerical tolerance


@pytest.mark.parametrize(
    ("rt_alpha", "rt_beta", "rt_loc", "log10_decay_rate"),
    [
        (3.0, 10.0, 5.0, 0.2),
        (10.0, 5.0, 20.0, 0.1),
        (0.8, 100.0, 50.0, 0.01),
    ],
)
def test_gamma_pdf_shifted_by_mu_loc(rt_alpha, rt_beta, rt_loc, log10_decay_rate):
    """PDF with rt_loc > 0 is a pure horizontal shift of the loc=0 PDF by mu*rt_loc."""
    r_base = np.linspace(0.01, 20.0, 30)
    pdf_loc_zero = gamma_pdf(r=r_base, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)
    pdf_shifted = gamma_pdf(
        r=r_base + log10_decay_rate * rt_loc,
        rt_alpha=rt_alpha,
        rt_beta=rt_beta,
        rt_loc=rt_loc,
        log10_decay_rate=log10_decay_rate,
    )
    assert_allclose(pdf_shifted, pdf_loc_zero, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    ("rt_alpha", "rt_beta", "rt_loc", "log10_decay_rate"),
    [
        (3.0, 10.0, 5.0, 0.2),
        (10.0, 5.0, 20.0, 0.1),
        (0.8, 100.0, 50.0, 0.01),
    ],
)
def test_gamma_cdf_shifted_by_mu_loc(rt_alpha, rt_beta, rt_loc, log10_decay_rate):
    """CDF with rt_loc > 0 is a pure horizontal shift of the loc=0 CDF by mu*rt_loc."""
    r_base = np.linspace(0.01, 20.0, 30)
    cdf_loc_zero = gamma_cdf(r=r_base, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)
    cdf_shifted = gamma_cdf(
        r=r_base + log10_decay_rate * rt_loc,
        rt_alpha=rt_alpha,
        rt_beta=rt_beta,
        rt_loc=rt_loc,
        log10_decay_rate=log10_decay_rate,
    )
    assert_allclose(cdf_shifted, cdf_loc_zero, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize(
    ("rt_alpha", "rt_beta", "rt_loc", "log10_decay_rate"),
    [
        (3.0, 10.0, 5.0, 0.2),
        (10.0, 5.0, 20.0, 0.1),
        (2.0, 20.0, 0.1, 0.5),
        (0.8, 100.0, 50.0, 0.01),
    ],
)
def test_gamma_mean_loc_adds_mu_loc(rt_alpha, rt_beta, rt_loc, log10_decay_rate):
    """gamma_mean with rt_loc > 0 equals loc=0 case plus the constant mu*rt_loc.

    Compares ``mean_loc`` directly against ``mean_zero + mu*rt_loc`` instead
    of the subtraction form ``mean_loc - mean_zero``. Both sides sum the
    same two floats (``mu*rt_loc`` and ``alpha*log10(...)``) in commutative
    order, so the comparison is bit-identical at ``rtol=1e-15``. The
    subtraction form suffered catastrophic cancellation when ``mu*rt_loc``
    was much smaller than the alpha/beta term.
    """
    mean_loc_zero = gamma_mean(rt_alpha=rt_alpha, rt_beta=rt_beta, log10_decay_rate=log10_decay_rate)
    mean_loc = gamma_mean(
        rt_alpha=rt_alpha,
        rt_beta=rt_beta,
        rt_loc=rt_loc,
        log10_decay_rate=log10_decay_rate,
    )
    assert_allclose(mean_loc, mean_loc_zero + log10_decay_rate * rt_loc, rtol=1e-15)


@pytest.mark.parametrize(
    ("rt_alpha", "rt_beta", "rt_loc", "log10_decay_rate"),
    [
        (3.0, 10.0, 5.0, 0.2),
        (10.0, 5.0, 20.0, 0.1),
        (0.8, 100.0, 50.0, 0.01),
    ],
)
def test_gamma_mean_loc_matches_numerical_integration(rt_alpha, rt_beta, rt_loc, log10_decay_rate):
    """gamma_mean with rt_loc > 0 matches -log10(E[10^(-R)]) via numerical integration."""
    analytical_mean = gamma_mean(
        rt_alpha=rt_alpha,
        rt_beta=rt_beta,
        rt_loc=rt_loc,
        log10_decay_rate=log10_decay_rate,
    )
    # E[10^(-R)] = integral of 10^(-r) * pdf(r) dr, where pdf has location mu*rt_loc
    lower = log10_decay_rate * rt_loc
    expected_decimal_reduction, _ = integrate.quad(
        lambda r: (
            10 ** (-r)
            * gamma_pdf(
                r=r,
                rt_alpha=rt_alpha,
                rt_beta=rt_beta,
                rt_loc=rt_loc,
                log10_decay_rate=log10_decay_rate,
            )
        ),
        lower,
        np.inf,
    )
    numerical_mean = -np.log10(expected_decimal_reduction)
    # scipy.integrate.quad on a half-infinite interval; same accuracy ceiling as
    # test_gamma_mean_matches_numerical_integration above.
    assert_allclose(analytical_mean, numerical_mean, atol=1e-8)


@pytest.mark.parametrize(
    ("apv_alpha", "apv_beta", "apv_loc", "log10_decay_rate", "target_mean"),
    [
        (1.5, 5.0, 2.0, 0.1, 0.5),
        (3.0, 10.0, 5.0, 0.2, 1.8),
        (10.0, 5.0, 20.0, 0.1, 3.5),
        (0.8, 100.0, 50.0, 0.01, 0.4),
    ],
)
def test_gamma_find_flow_for_target_mean_with_loc(apv_alpha, apv_beta, apv_loc, log10_decay_rate, target_mean):
    """gamma_find_flow_for_target_mean with apv_loc > 0 inverts gamma_mean round-trip."""
    required_flow = gamma_find_flow_for_target_mean(
        target_mean=target_mean,
        apv_alpha=apv_alpha,
        apv_beta=apv_beta,
        apv_loc=apv_loc,
        log10_decay_rate=log10_decay_rate,
    )
    # The residence-time distribution at flow Q is a shifted gamma with
    # shape=apv_alpha, scale=apv_beta/Q, location=apv_loc/Q.
    rt_alpha = apv_alpha
    rt_beta = apv_beta / required_flow
    rt_loc = apv_loc / required_flow
    verification_mean = gamma_mean(
        rt_alpha=rt_alpha,
        rt_beta=rt_beta,
        rt_loc=rt_loc,
        log10_decay_rate=log10_decay_rate,
    )
    # apv_loc>0 requires brentq, whose default tolerance limits the round-trip.
    assert_allclose(verification_mean, target_mean, rtol=1e-10)


def test_gamma_find_flow_for_target_mean_loc_zero_matches_legacy():
    """With apv_loc=0 the implementation must use the closed-form branch and match."""
    apv_alpha = 3.0
    apv_beta = 10.0
    log10_decay_rate = 0.2
    target_mean = 1.5
    flow_default = gamma_find_flow_for_target_mean(
        target_mean=target_mean,
        apv_alpha=apv_alpha,
        apv_beta=apv_beta,
        log10_decay_rate=log10_decay_rate,
    )
    flow_loc_zero = gamma_find_flow_for_target_mean(
        target_mean=target_mean,
        apv_alpha=apv_alpha,
        apv_beta=apv_beta,
        apv_loc=0.0,
        log10_decay_rate=log10_decay_rate,
    )
    assert flow_default == flow_loc_zero


def test_gamma_pdf_negative_loc_raises():
    """gamma_pdf must reject negative rt_loc."""
    with pytest.raises(ValueError, match="rt_loc must be non-negative"):
        gamma_pdf(r=np.array([1.0]), rt_alpha=3.0, rt_beta=10.0, rt_loc=-1.0, log10_decay_rate=0.2)


def test_gamma_cdf_negative_loc_raises():
    """gamma_cdf must reject negative rt_loc."""
    with pytest.raises(ValueError, match="rt_loc must be non-negative"):
        gamma_cdf(r=np.array([1.0]), rt_alpha=3.0, rt_beta=10.0, rt_loc=-1.0, log10_decay_rate=0.2)


def test_gamma_mean_negative_loc_raises():
    """gamma_mean must reject negative rt_loc."""
    with pytest.raises(ValueError, match="rt_loc must be non-negative"):
        gamma_mean(rt_alpha=3.0, rt_beta=10.0, rt_loc=-1.0, log10_decay_rate=0.2)


def test_gamma_find_flow_for_target_mean_negative_loc_raises():
    """gamma_find_flow_for_target_mean must reject negative apv_loc."""
    with pytest.raises(ValueError, match="apv_loc must be non-negative"):
        gamma_find_flow_for_target_mean(
            target_mean=1.0,
            apv_alpha=3.0,
            apv_beta=10.0,
            apv_loc=-5.0,
            log10_decay_rate=0.2,
        )


def test_gamma_find_flow_for_target_mean_zero_decay_raises():
    """log10_decay_rate=0 with apv_loc>0 used to call brentq on [0, inf]; now raises.

    Without decay the effective mean log removal is identically zero regardless of flow,
    so no finite flow can reach a strictly positive target_mean; ``flow_closed_form``
    becomes 0, ``u_upper`` becomes ``inf``, and brentq would receive an invalid bracket.
    The guard converts that into an informative ValueError.
    """
    with pytest.raises(ValueError, match="log10_decay_rate must be positive"):
        gamma_find_flow_for_target_mean(
            target_mean=1.0,
            apv_alpha=3.0,
            apv_beta=10.0,
            apv_loc=5.0,
            log10_decay_rate=0.0,
        )
    # Same guard fires when apv_loc=0, since the closed-form would otherwise
    # divide by 10**(target_mean/apv_alpha) - 1 with a zero numerator.
    with pytest.raises(ValueError, match="log10_decay_rate must be positive"):
        gamma_find_flow_for_target_mean(
            target_mean=1.0,
            apv_alpha=3.0,
            apv_beta=10.0,
            apv_loc=0.0,
            log10_decay_rate=0.0,
        )


def test_gamma_find_flow_for_target_mean_negative_decay_raises():
    """Negative decay rates are physically meaningless and must raise."""
    with pytest.raises(ValueError, match="log10_decay_rate must be positive"):
        gamma_find_flow_for_target_mean(
            target_mean=1.0,
            apv_alpha=3.0,
            apv_beta=10.0,
            apv_loc=5.0,
            log10_decay_rate=-0.1,
        )


@pytest.mark.parametrize(("apv_alpha", "apv_beta"), [(0.0, 10.0), (-1.0, 10.0), (3.0, 0.0), (3.0, -5.0)])
def test_gamma_find_flow_for_target_mean_invalid_alpha_beta_raises(apv_alpha, apv_beta):
    """apv_alpha and apv_beta must be strictly positive."""
    with pytest.raises(ValueError, match="must be positive"):
        gamma_find_flow_for_target_mean(
            target_mean=1.0,
            apv_alpha=apv_alpha,
            apv_beta=apv_beta,
            apv_loc=5.0,
            log10_decay_rate=0.1,
        )


def test_gamma_mean_matches_parallel_mean_discretized():
    """Test that gamma_mean matches parallel_mean computed from discretized bins."""
    alpha = 5.0
    beta = 3.0
    mu = 0.15
    n_bins = 500  # Large number for good approximation

    # Compute gamma_mean (analytical via MGF)
    result_analytical = gamma_mean(rt_alpha=alpha, rt_beta=beta, log10_decay_rate=mu)

    # Compute via discretized bins
    bins = gamma_bins(alpha=alpha, beta=beta, n_bins=n_bins)
    log_removals = residence_time_to_log_removal(
        residence_times=bins["expected_values"],
        log10_decay_rate=mu,
    )
    result_discretized = parallel_mean(log_removals=log_removals)

    # 500 equiprobable bins discretizing the gamma distribution have empirical
    # relative error of ~1e-4 here; rtol=1e-3 is a tight bound.
    assert_allclose(result_analytical, result_discretized, rtol=1e-3)


# ---------------------------------------------------------------------------
# New tests added for #35 (parallel_mean default-axis on multi-D) and #173
# groups C and D (log<->linear identity, series composition, NaN/inf, sign
# contracts).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2),  # Square 2D: returns silently wrong answer in pre-fix code (#35)
        (2, 3),  # Non-square 2D: raises broadcast ValueError in pre-fix code (#35)
    ],
)
def test_parallel_mean_2d_default_axis_flat(shape):
    """parallel_mean(arr) with axis=None reduces over the flat array (matches np.mean)."""
    rng = np.random.default_rng(42)
    arr = rng.uniform(0.5, 6.0, size=shape)
    result_2d = parallel_mean(log_removals=arr)
    result_flat = parallel_mean(log_removals=arr.ravel())
    assert_allclose(result_2d, result_flat, rtol=1e-15)


def test_parallel_mean_3d_default_axis_flat():
    """parallel_mean on a 3D array with axis=None is identical to the flattened call."""
    arr = np.array([[[3.0, 4.0], [5.0, 2.0]], [[1.0, 6.0], [2.0, 3.0]]])
    result_3d = parallel_mean(log_removals=arr)
    result_flat = parallel_mean(log_removals=arr.ravel())
    assert_allclose(result_3d, result_flat, rtol=1e-15)


def test_parallel_mean_two_path_analytic():
    """Equal-flow mix of LR=BIG and LR=0 has effective LR = log10(2) for BIG -> inf."""
    result = parallel_mean(log_removals=[20.0, 0.0])
    # 10^-20 is below double-precision relevance for the additive comparison;
    # the analytic limit is exactly log10(2).
    assert_allclose(result, np.log10(2.0), rtol=1e-15)


def test_parallel_mean_zero_concentration_handling():
    """LR=inf (perfect removal) reduces the flow contribution to zero in the mix."""
    result = parallel_mean(log_removals=[np.inf, 3.0])
    # ``(0 + 10^-3) / 2`` ⇒ ``-log10`` ⇒ ``3 + log10(2)``.
    assert_allclose(result, 3.0 + np.log10(2.0), rtol=1e-15)


def test_parallel_mean_negative_log_removal():
    """Negative log removal (amplification) is mathematically well-defined and accepted."""
    result = parallel_mean(log_removals=[-1.0, 3.0])
    expected = -np.log10((10.0**1.0 + 10.0**-3.0) / 2.0)
    assert_allclose(result, expected, rtol=1e-15)


def test_parallel_mean_nan_input_propagates():
    """NaN in log_removals propagates to a NaN result (no silent dropping)."""
    result = parallel_mean(log_removals=[np.nan, 3.0])
    assert np.isnan(result)


def test_parallel_mean_empty_array():
    """Empty input returns NaN; the RuntimeWarning from np.mean is suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = parallel_mean(log_removals=[])
    assert np.isnan(result)


@pytest.mark.parametrize(
    "log_removals",
    [
        [3.0, 4.0, 5.0],
        [0.0, 2.0, 4.0],
        [-1.0, 3.0],  # amplification path
        [1.5, 1.5, 1.5],  # uniform: parallel == both bounds
        [9.999, 9.998],  # near-equal large LR
    ],
)
def test_parallel_mean_bounded_by_min_and_max(log_removals):
    """``min(LR) <= parallel_mean(LR) <= max(LR)`` for any input.

    The weighted decimal reductions are a convex combination of ``10^{-LR_i}``,
    so the result lies between ``10^{-max(LR)}`` and ``10^{-min(LR)}``; taking
    ``-log10`` reverses the bounds. A strictly stronger invariant than the
    existing ``parallel_mean <= arithmetic_mean`` Jensen check used in the
    gamma tests.
    """
    arr = np.asarray(log_removals, dtype=float)
    result = parallel_mean(log_removals=arr)
    assert arr.min() - 1e-12 <= result <= arr.max() + 1e-12


@pytest.mark.parametrize(
    ("residence_time", "log10_decay_rate"),
    [
        (0.0, 0.2),
        (1.0, 0.2),
        (10.0, 0.05),
        (1.0e3, 0.5),
    ],
)
def test_residence_time_to_log_removal_concentration_ratio_round_trip(residence_time, log10_decay_rate):
    """``10**(-LR) == exp(-mu * ln10 * t)`` to machine precision.

    Pins both the sign of the exponent and the ``ln(10)`` factor in the
    conversion between the decimal and natural-log decay-rate spellings.
    """
    log_removal = residence_time_to_log_removal(residence_times=residence_time, log10_decay_rate=log10_decay_rate)
    decimal_form = 10.0 ** (-log_removal)
    natural_form = np.exp(-log10_decay_rate * np.log(10.0) * residence_time)
    assert_allclose(decimal_form, natural_form, rtol=1e-15)


def test_residence_time_to_log_removal_series_composition_with_pinned_value():
    """Series composition is additive AND lands on the expected pinned value.

    Linearity alone is satisfied by any ``c * t`` formula (including the sign-
    flipped one); pinning a specific numeric value rules those mutations out.
    """
    mu = 0.2
    lr_full = residence_time_to_log_removal(residence_times=10.0, log10_decay_rate=mu)
    lr_first = residence_time_to_log_removal(residence_times=7.0, log10_decay_rate=mu)
    lr_second = residence_time_to_log_removal(residence_times=3.0, log10_decay_rate=mu)
    assert_allclose(lr_full, 2.0, rtol=1e-15)
    assert_allclose(lr_first + lr_second, lr_full, rtol=1e-15)


def test_residence_time_to_log_removal_negative_residence_time():
    """Negative residence time returns negative log removal; no validation is performed."""
    result = residence_time_to_log_removal(residence_times=-5.0, log10_decay_rate=0.2)
    assert_allclose(result, -1.0, rtol=1e-15)
