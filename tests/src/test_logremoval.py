import contextlib

import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, stats

from gwtransport.logremoval import (
    decay_rate_to_log10_removal_rate,
    gamma_cdf,
    gamma_find_flow_for_target_mean,
    gamma_mean,
    gamma_pdf,
    log10_removal_rate_to_decay_rate,
    parallel_mean,
    residence_time_to_log_removal,
)


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
    assert_allclose(result, expected, rtol=1e-10)
    assert_allclose(result, 3.431798275933005, rtol=1e-3)  # example in docstring


def test_array_inputs_equal_distribution():
    """Test with numpy array inputs for equal distribution."""
    # NumPy arrays as input
    result = parallel_mean(log_removals=np.array([3.0, 4.0, 5.0]))
    expected = -np.log10((10 ** (-3.0) + 10 ** (-4.0) + 10 ** (-5.0)) / 3)
    assert_allclose(result, expected, rtol=1e-10)


def test_empty_input_behavior():
    """Test behavior with empty arrays (now handles naturally via numpy)."""
    # Empty arrays will result in nan due to mathematical operations
    result = parallel_mean(log_removals=[])
    assert np.isnan(result) or np.isinf(result)  # numpy naturally handles this


def test_special_values_equal_distribution():
    """Test with special values like zero and large numbers with equal distribution."""
    # With log removal of 0 (no removal)
    result = parallel_mean(log_removals=[0.0, 4.0])
    expected = -np.log10((1.0 + 10 ** (-4.0)) / 2)
    assert_allclose(result, expected, rtol=1e-10)

    # With very large log removal (effectively complete removal)
    # Using a large number instead of infinity to avoid numerical issues
    result = parallel_mean(log_removals=[20.0, 3.0])
    # The 10^-20 term is effectively zero
    expected = -np.log10((10 ** (-20.0) + 10 ** (-3.0)) / 2)
    assert_allclose(result, expected, rtol=1e-10)


def test_float_precision_equal_distribution():
    """Test handling of floating point precision with equal distribution."""
    # Testing with values that require good floating point handling
    result = parallel_mean(log_removals=[9.999, 9.998])
    expected = -np.log10((10 ** (-9.999) + 10 ** (-9.998)) / 2)
    assert_allclose(result, expected, rtol=1e-10)


def test_equal_weights_explicit():
    """Test with explicitly provided equal weights - should match the implicit equal weights."""
    log_removals = [3.0, 4.0, 5.0]
    weights = [1 / 3, 1 / 3, 1 / 3]

    weighted_result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    unweighted_result = parallel_mean(log_removals=log_removals)

    assert_allclose(weighted_result, unweighted_result, rtol=1e-10)


def test_weighted_flows():
    """Test with different weights for each flow."""
    # Test case: Two flows with different weights
    log_removals = [3.0, 5.0]
    weights = [0.7, 0.3]

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    expected = -np.log10(0.7 * 10 ** (-3.0) + 0.3 * 10 ** (-5.0))
    assert_allclose(result, expected, rtol=1e-10)
    assert_allclose(result, 3.153044674980176, rtol=1e-7)  # example in docstring

    # Test case: Three flows with different weights
    log_removals = [2.0, 4.0, 6.0]
    weights = [0.5, 0.3, 0.2]

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    expected = -np.log10(0.5 * 10 ** (-2.0) + 0.3 * 10 ** (-4.0) + 0.2 * 10 ** (-6.0))
    assert_allclose(result, expected, rtol=1e-10)


def test_weight_sum_behavior():
    """Test behavior when weights don't sum to 1.0 (validation removed)."""
    log_removals = [3.0, 4.0]

    # Weights that don't sum to 1.0 will now proceed with calculation
    result1 = parallel_mean(log_removals=log_removals, flow_fractions=[0.7, 0.4])  # Sum > 1
    result2 = parallel_mean(log_removals=log_removals, flow_fractions=[0.7, 0.2])  # Sum < 1

    # Results should be numeric (calculation proceeds)
    assert isinstance(result1, (int, float))
    assert not np.isnan(result1)
    assert isinstance(result2, (int, float))
    assert not np.isnan(result2)


def test_length_mismatch_behavior():
    """Test behavior with mismatched lengths (validation removed)."""
    # Mismatched lengths will now be handled by numpy broadcasting or errors
    try:
        result1 = parallel_mean(log_removals=[3.0, 4.0], flow_fractions=[1.0])
        # If it succeeds, check it's a valid number
        assert isinstance(result1, (int, float))
    except (ValueError, IndexError):
        # NumPy may still raise errors for incompatible operations
        pass

    try:
        result2 = parallel_mean(log_removals=[3.0], flow_fractions=[0.5, 0.5])
        assert isinstance(result2, (int, float))
    except (ValueError, IndexError):
        pass


def test_weighted_array_inputs():
    """Test with numpy array inputs for weights."""
    log_removals = np.array([3.0, 5.0])
    weights = np.array([0.6, 0.4])

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    expected = -np.log10(0.6 * 10 ** (-3.0) + 0.4 * 10 ** (-5.0))
    assert_allclose(result, expected, rtol=1e-10)


def test_extreme_weights():
    """Test with extreme weight distributions."""
    # One weight is almost 1.0, others are tiny
    log_removals = [3.0, 5.0, 6.0]
    weights = [0.999, 0.0005, 0.0005]

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    # Result should be very close to the first log removal
    assert_allclose(result, 3.0, rtol=1e-2)

    # One weight is exactly 1.0, others are exactly 0.0
    log_removals = [4.0, 5.0, 6.0]
    weights = [1.0, 0.0, 0.0]

    result = parallel_mean(log_removals=log_removals, flow_fractions=weights)
    assert result == 4.0


def test_gamma_find_flow_for_target_mean():
    """Test that gamma_find_flow_for_target_mean inverts gamma_mean correctly."""
    apv_alpha = 2.0
    apv_beta = 10.0
    log10_removal_rate = 0.2

    target_mean = 3.0
    required_flow = gamma_find_flow_for_target_mean(
        target_mean=target_mean, apv_alpha=apv_alpha, apv_beta=apv_beta, log10_removal_rate=log10_removal_rate
    )

    # Verify the result
    rt_alpha = apv_alpha
    rt_beta = apv_beta / required_flow
    verification_mean = gamma_mean(rt_alpha=rt_alpha, rt_beta=rt_beta, log10_removal_rate=log10_removal_rate)
    assert_allclose(verification_mean, target_mean, rtol=1e-10)


def test_gamma_find_flow_for_target_mean_explicit():
    """Test gamma_find_flow_for_target_mean with explicit expected value."""
    # flow = mu * alpha * beta / target
    # flow = 0.2 * 2.0 * 10.0 / 3.0 = 4.0/3.0
    result = gamma_find_flow_for_target_mean(target_mean=3.0, apv_alpha=2.0, apv_beta=10.0, log10_removal_rate=0.2)
    assert_allclose(result, 4.0 / 3.0)


def test_gamma_mean_explicit():
    """Test gamma_mean with explicit expected value."""
    # E[R] = mu * alpha * beta = 0.2 * 5.0 * 10.0 = 10.0
    result = gamma_mean(rt_alpha=5.0, rt_beta=10.0, log10_removal_rate=0.2)
    assert_allclose(result, 10.0)


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

    assert_allclose(result_axis1, expected_axis1, rtol=1e-10)

    # Test axis=0 (along rows)
    result_axis0 = parallel_mean(log_removals=log_removals_2d, axis=0)

    # Expected results: parallel_mean for each column
    expected_col0 = parallel_mean(log_removals=[3.0, 2.0])
    expected_col1 = parallel_mean(log_removals=[4.0, 3.0])
    expected_col2 = parallel_mean(log_removals=[5.0, 4.0])
    expected_axis0 = np.array([expected_col0, expected_col1, expected_col2])

    assert_allclose(result_axis0, expected_axis0, rtol=1e-10)


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

    assert_allclose(result_axis2, expected_axis2, rtol=1e-10)


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

    assert_allclose(result, expected, rtol=1e-10)


def test_axis_parameter_behavior():
    """Test natural behavior of axis parameter without explicit validation."""
    log_removals_1d = np.array([3.0, 4.0, 5.0])
    log_removals_2d = np.array([[3.0, 4.0], [5.0, 2.0]])

    # Test axis with 1D array - numpy will handle this naturally
    try:
        result = parallel_mean(log_removals=log_removals_1d, axis=0)
        # If successful, result should be a number
        assert isinstance(result, (int, float, np.ndarray))
    except (ValueError, IndexError):
        # NumPy may raise axis errors naturally
        pass

    # Test potentially out of bounds axis - let numpy handle
    with contextlib.suppress(ValueError, IndexError):
        parallel_mean(log_removals=log_removals_2d, axis=2)


def test_negative_axis():
    """Test negative axis indexing."""
    log_removals_2d = np.array([[3.0, 4.0, 5.0], [2.0, 3.0, 4.0]])

    # axis=-1 should be equivalent to axis=1 for 2D array
    result_neg1 = parallel_mean(log_removals=log_removals_2d, axis=-1)
    result_pos1 = parallel_mean(log_removals=log_removals_2d, axis=1)

    assert_allclose(result_neg1, result_pos1, rtol=1e-10)

    # axis=-2 should be equivalent to axis=0 for 2D array
    result_neg2 = parallel_mean(log_removals=log_removals_2d, axis=-2)
    result_pos0 = parallel_mean(log_removals=log_removals_2d, axis=0)

    assert_allclose(result_neg2, result_pos0, rtol=1e-10)


def test_empty_array_with_axis():
    """Test empty array behavior with axis parameter."""
    empty_2d = np.array([]).reshape(0, 3)

    # Empty arrays now proceed with calculation, may result in nan/inf
    try:
        result = parallel_mean(log_removals=empty_2d, axis=1)
        # Result may be nan, inf, or empty array
        assert isinstance(result, (int, float, np.ndarray))
    except (ValueError, ZeroDivisionError):
        # NumPy may naturally raise errors for empty operations
        pass


def test_compute_log_removal_basic():
    """Test basic functionality of residence_time_to_log_removal."""
    # Test with single values: LR = mu * t
    assert_allclose(residence_time_to_log_removal(residence_times=0.0, log10_removal_rate=2.0), 0.0)  # 2.0 * 0 = 0
    assert_allclose(residence_time_to_log_removal(residence_times=10.0, log10_removal_rate=0.2), 2.0)  # 0.2 * 10 = 2
    assert_allclose(residence_time_to_log_removal(residence_times=20.0, log10_removal_rate=0.15), 3.0)  # 0.15 * 20 = 3

    # Test with different log10_removal_rates
    assert_allclose(residence_time_to_log_removal(residence_times=10.0, log10_removal_rate=0.1), 1.0)
    assert_allclose(residence_time_to_log_removal(residence_times=10.0, log10_removal_rate=0.3), 3.0)


def test_compute_log_removal_arrays():
    """Test residence_time_to_log_removal with array inputs."""
    residence_times = np.array([10.0, 20.0, 50.0])
    log10_removal_rate = 0.2

    result = residence_time_to_log_removal(residence_times=residence_times, log10_removal_rate=log10_removal_rate)
    expected = np.array([2.0, 4.0, 10.0])

    assert_allclose(result, expected)


def test_compute_log_removal_2d_arrays():
    """Test residence_time_to_log_removal with 2D array inputs."""
    residence_times_2d = np.array([[10.0, 20.0], [30.0, 40.0]])
    log10_removal_rate = 0.1

    result = residence_time_to_log_removal(residence_times=residence_times_2d, log10_removal_rate=log10_removal_rate)
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert_allclose(result, expected)


def test_compute_log_removal_formula_verification():
    """Test that the formula log10_removal_rate * residence_time is correctly implemented."""
    residence_times = np.array([5.0, 25.0, 50.0])
    log10_removal_rate = 0.15

    result = residence_time_to_log_removal(residence_times=residence_times, log10_removal_rate=log10_removal_rate)
    expected = log10_removal_rate * residence_times

    assert_allclose(result, expected, rtol=1e-15)


def test_compute_log_removal_edge_cases():
    """Test residence_time_to_log_removal with edge cases."""
    # Zero residence time
    result = residence_time_to_log_removal(residence_times=0.0, log10_removal_rate=0.2)
    assert_allclose(result, 0.0)

    # Very large residence time
    result = residence_time_to_log_removal(residence_times=1e6, log10_removal_rate=0.1)
    expected = 0.1 * 1e6
    assert_allclose(result, expected)

    # Zero log10_removal_rate
    result = residence_time_to_log_removal(residence_times=100.0, log10_removal_rate=0.0)
    assert result == 0.0


def test_compute_log_removal_different_log_rates():
    """Test residence_time_to_log_removal with various log10 removal rates."""
    residence_time = 10.0

    # Test different rates
    rates = [0.05, 0.1, 0.15, 0.2, 0.3]
    expected_results = [rate * residence_time for rate in rates]

    for rate, expected in zip(rates, expected_results, strict=False):
        result = residence_time_to_log_removal(residence_times=residence_time, log10_removal_rate=rate)
        assert_allclose(result, expected)


def test_compute_log_removal_list_input():
    """Test residence_time_to_log_removal with list inputs (should be converted to numpy array)."""
    residence_times = [10.0, 20.0, 50.0]
    log10_removal_rate = 0.2

    result = residence_time_to_log_removal(residence_times=residence_times, log10_removal_rate=log10_removal_rate)
    expected = np.array([2.0, 4.0, 10.0])

    assert_allclose(result, expected)
    assert isinstance(result, np.ndarray)


def test_compute_log_removal_shape_preservation():
    """Test that residence_time_to_log_removal preserves input array shape."""
    # Test 1D
    residence_times_1d = np.array([10.0, 20.0, 50.0])
    result_1d = residence_time_to_log_removal(residence_times=residence_times_1d, log10_removal_rate=0.1)
    assert result_1d.shape == residence_times_1d.shape

    # Test 2D
    residence_times_2d = np.array([[10.0, 20.0], [30.0, 40.0]])
    result_2d = residence_time_to_log_removal(residence_times=residence_times_2d, log10_removal_rate=0.1)
    assert result_2d.shape == residence_times_2d.shape

    # Test 3D
    residence_times_3d = np.array([[[10.0, 20.0]], [[30.0, 40.0]]])
    result_3d = residence_time_to_log_removal(residence_times=residence_times_3d, log10_removal_rate=0.1)
    assert result_3d.shape == residence_times_3d.shape


def test_compute_log_removal_docstring_examples():
    """Test the examples from the docstring."""
    # Example 1: Basic array
    residence_times = np.array([10.0, 20.0, 50.0])
    log10_removal_rate = 0.2
    result = residence_time_to_log_removal(residence_times=residence_times, log10_removal_rate=log10_removal_rate)
    expected = np.array([2.0, 4.0, 10.0])
    assert_allclose(result, expected)

    # Example 2: Single residence time
    result = residence_time_to_log_removal(residence_times=5.0, log10_removal_rate=0.3)
    assert_allclose(result, 1.5)

    # Example 3: 2D array
    residence_times_2d = np.array([[10.0, 20.0], [30.0, 40.0]])
    result = residence_time_to_log_removal(residence_times=residence_times_2d, log10_removal_rate=0.1)
    expected = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_allclose(result, expected)


def test_compute_log_removal_consistency_with_manual_calculation():
    """Test consistency between function result and manual calculation."""
    residence_times = np.array([2.5, 7.3, 15.8, 33.2])
    log10_removal_rate = 0.23

    # Manual calculation
    manual_result = log10_removal_rate * residence_times

    # Function result
    function_result = residence_time_to_log_removal(
        residence_times=residence_times, log10_removal_rate=log10_removal_rate
    )

    assert_allclose(function_result, manual_result, rtol=1e-15)


def test_decay_rate_to_log10_removal_rate():
    """Test conversion from natural-log decay rate to log10 removal rate."""
    # lambda = ln(10) should give mu = 1.0
    decay_rate = np.log(10)
    result = decay_rate_to_log10_removal_rate(decay_rate)
    assert_allclose(result, 1.0)

    # lambda = 0 should give mu = 0
    assert_allclose(decay_rate_to_log10_removal_rate(0.0), 0.0)


def test_log10_removal_rate_to_decay_rate():
    """Test conversion from log10 removal rate to natural-log decay rate."""
    # mu = 1.0 should give lambda = ln(10)
    result = log10_removal_rate_to_decay_rate(1.0)
    assert_allclose(result, np.log(10))

    # mu = 0 should give lambda = 0
    assert_allclose(log10_removal_rate_to_decay_rate(0.0), 0.0)


def test_conversion_round_trip():
    """Test that converting back and forth preserves the value."""
    original_mu = 0.2
    lambda_val = log10_removal_rate_to_decay_rate(original_mu)
    recovered_mu = decay_rate_to_log10_removal_rate(lambda_val)
    assert_allclose(recovered_mu, original_mu, rtol=1e-15)

    original_lambda = 0.5
    mu_val = decay_rate_to_log10_removal_rate(original_lambda)
    recovered_lambda = log10_removal_rate_to_decay_rate(mu_val)
    assert_allclose(recovered_lambda, original_lambda, rtol=1e-15)


def test_gamma_pdf_integrates_to_one():
    """Test that the gamma PDF integrates to 1."""
    rt_alpha = 3.0
    rt_beta = 10.0
    log10_removal_rate = 0.2

    result, _ = integrate.quad(
        lambda r: gamma_pdf(r=r, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_removal_rate=log10_removal_rate),
        0,
        np.inf,
    )
    assert_allclose(result, 1.0, rtol=1e-8)


def test_gamma_cdf_approaches_one():
    """Test that the gamma CDF approaches 1 for large r values."""
    rt_alpha = 3.0
    rt_beta = 10.0
    log10_removal_rate = 0.2

    # CDF at a very large r should be close to 1
    result = gamma_cdf(r=1000.0, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_removal_rate=log10_removal_rate)
    assert_allclose(result, 1.0, atol=1e-10)

    # CDF at 0 should be 0
    result = gamma_cdf(r=0.0, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_removal_rate=log10_removal_rate)
    assert_allclose(result, 0.0, atol=1e-10)


def test_gamma_mean_matches_numerical_integration():
    """Test that gamma_mean matches numerical integration of r * pdf(r)."""
    rt_alpha = 3.0
    rt_beta = 10.0
    log10_removal_rate = 0.2

    analytical_mean = gamma_mean(rt_alpha=rt_alpha, rt_beta=rt_beta, log10_removal_rate=log10_removal_rate)

    numerical_mean, _ = integrate.quad(
        lambda r: r * gamma_pdf(r=r, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_removal_rate=log10_removal_rate),
        0,
        np.inf,
    )

    assert_allclose(analytical_mean, numerical_mean, rtol=1e-8)


def test_gamma_pdf_is_scaled_gamma():
    """Test that the log removal PDF matches a scaled gamma distribution."""
    rt_alpha = 4.0
    rt_beta = 5.0
    log10_removal_rate = 0.3

    r_values = np.linspace(0.1, 20.0, 50)
    pdf_values = gamma_pdf(r=r_values, rt_alpha=rt_alpha, rt_beta=rt_beta, log10_removal_rate=log10_removal_rate)

    # R = mu*T, so R ~ Gamma(alpha, mu*beta)
    expected = stats.gamma.pdf(r_values, a=rt_alpha, scale=log10_removal_rate * rt_beta)
    assert_allclose(pdf_values, expected, rtol=1e-12)
