import numpy as np
import pytest
from numpy.testing import assert_allclose

from gwtransport.logremoval import gamma_find_flow_for_target_mean, gamma_mean, parallel_mean


def test_single_flow():
    """Test with a single flow path - result should be the same as input."""
    assert parallel_mean([4.0]) == 4.0
    assert parallel_mean(np.array([3.5])) == 3.5
    # With explicit flow fraction
    assert parallel_mean([4.0], [1.0]) == 4.0


def test_identical_flows_equal_distribution():
    """Test with multiple identical flows - result should match the inputs."""
    assert parallel_mean([3.0, 3.0]) == 3.0
    assert parallel_mean([2.0, 2.0, 2.0, 2.0]) == 2.0


def test_different_flows_equal_distribution():
    """Test with flows having different removal values with equal distribution."""
    # Test case: Two flows with log removals 3 and 5
    result = parallel_mean([3, 4, 5])
    expected = -np.log10((10 ** (-3.0) + 10 ** (-4.0) + 10 ** (-5.0)) / 3)
    assert_allclose(result, expected, rtol=1e-10)
    assert_allclose(result, 3.431798275933005, rtol=1e-3)  # example in docstring


def test_array_inputs_equal_distribution():
    """Test with numpy array inputs for equal distribution."""
    # NumPy arrays as input
    result = parallel_mean(np.array([3.0, 4.0, 5.0]))
    expected = -np.log10((10 ** (-3.0) + 10 ** (-4.0) + 10 ** (-5.0)) / 3)
    assert_allclose(result, expected, rtol=1e-10)


def test_empty_input_raises_error():
    """Test that an empty array input raises ValueError."""
    with pytest.raises(ValueError):
        parallel_mean([])
    with pytest.raises(ValueError):
        parallel_mean(np.array([]))
    with pytest.raises(ValueError):
        parallel_mean([], [])


def test_special_values_equal_distribution():
    """Test with special values like zero and large numbers with equal distribution."""
    # With log removal of 0 (no removal)
    result = parallel_mean([0.0, 4.0])
    expected = -np.log10((1.0 + 10 ** (-4.0)) / 2)
    assert_allclose(result, expected, rtol=1e-10)

    # With very large log removal (effectively complete removal)
    # Using a large number instead of infinity to avoid numerical issues
    result = parallel_mean([20.0, 3.0])
    # The 10^-20 term is effectively zero
    expected = -np.log10((10 ** (-20.0) + 10 ** (-3.0)) / 2)
    assert_allclose(result, expected, rtol=1e-10)


def test_float_precision_equal_distribution():
    """Test handling of floating point precision with equal distribution."""
    # Testing with values that require good floating point handling
    result = parallel_mean([9.999, 9.998])
    expected = -np.log10((10 ** (-9.999) + 10 ** (-9.998)) / 2)
    assert_allclose(result, expected, rtol=1e-10)


def test_equal_weights_explicit():
    """Test with explicitly provided equal weights - should match the implicit equal weights."""
    log_removals = [3.0, 4.0, 5.0]
    weights = [1 / 3, 1 / 3, 1 / 3]

    weighted_result = parallel_mean(log_removals, weights)
    unweighted_result = parallel_mean(log_removals)

    assert_allclose(weighted_result, unweighted_result, rtol=1e-10)


def test_weighted_flows():
    """Test with different weights for each flow."""
    # Test case: Two flows with different weights
    log_removals = [3.0, 5.0]
    weights = [0.7, 0.3]

    result = parallel_mean(log_removals, weights)
    expected = -np.log10(0.7 * 10 ** (-3.0) + 0.3 * 10 ** (-5.0))
    assert_allclose(result, expected, rtol=1e-10)
    assert_allclose(result, 3.153044674980176, rtol=1e-7)  # example in docstring

    # Test case: Three flows with different weights
    log_removals = [2.0, 4.0, 6.0]
    weights = [0.5, 0.3, 0.2]

    result = parallel_mean(log_removals, weights)
    expected = -np.log10(0.5 * 10 ** (-2.0) + 0.3 * 10 ** (-4.0) + 0.2 * 10 ** (-6.0))
    assert_allclose(result, expected, rtol=1e-10)


def test_weight_sum_validation():
    """Test that weights must sum to 1.0."""
    log_removals = [3.0, 4.0]

    # Weights that don't sum to 1.0 should raise an error
    with pytest.raises(ValueError):
        parallel_mean(log_removals, [0.7, 0.4])  # Sum > 1

    with pytest.raises(ValueError):
        parallel_mean(log_removals, [0.7, 0.2])  # Sum < 1

    # Weights that sum to almost 1.0 (within floating point tolerance) should be ok
    result = parallel_mean(log_removals, [0.7, 0.3 - 1e-10])
    assert result is not None


def test_length_validation():
    """Test that log_removals and weights must have the same length."""
    with pytest.raises(ValueError):
        parallel_mean([3.0, 4.0], [1.0])

    with pytest.raises(ValueError):
        parallel_mean([3.0], [0.5, 0.5])


def test_weighted_array_inputs():
    """Test with numpy array inputs for weights."""
    log_removals = np.array([3.0, 5.0])
    weights = np.array([0.6, 0.4])

    result = parallel_mean(log_removals, weights)
    expected = -np.log10(0.6 * 10 ** (-3.0) + 0.4 * 10 ** (-5.0))
    assert_allclose(result, expected, rtol=1e-10)


def test_extreme_weights():
    """Test with extreme weight distributions."""
    # One weight is almost 1.0, others are tiny
    log_removals = [3.0, 5.0, 6.0]
    weights = [0.999, 0.0005, 0.0005]

    result = parallel_mean(log_removals, weights)
    # Result should be very close to the first log removal
    assert_allclose(result, 3.0, rtol=1e-2)

    # One weight is exactly 1.0, others are exactly 0.0
    log_removals = [4.0, 5.0, 6.0]
    weights = [1.0, 0.0, 0.0]

    result = parallel_mean(log_removals, weights)
    assert result == 4.0


def test_gamma_find_flow_for_target_mean():
    # Example parameters
    apv_alpha = 2.0  # Shape parameter for pore volume
    apv_beta = 10.0  # Scale parameter for pore volume
    log_removal_rate = 2.0  # Coefficient for log removal

    # Example of finding Q for a target mean log removal
    target_mean = 3.0  # Example target mean
    required_flow = gamma_find_flow_for_target_mean(target_mean, apv_alpha, apv_beta, log_removal_rate)

    # Verify the result
    rt_alpha = 2.0  # Shape parameter for residence time
    rt_beta = apv_beta / required_flow  # Scale parameter for residence time
    verification_mean = gamma_mean(rt_alpha, rt_beta, log_removal_rate)
    assert_allclose(verification_mean, target_mean, rtol=1e-10)
