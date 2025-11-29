import numpy as np
import pandas as pd
import pytest

from gwtransport.residence_time import fraction_explained


@pytest.fixture
def constant_flow_setup():
    """Create constant flow setup for testing."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-11", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)  # 100 m³/day constant flow
    return flow_values, flow_tedges


@pytest.fixture
def multiple_pore_volumes():
    """Create multiple pore volumes for testing."""
    # Create pore volumes that will have residence times: 1, 2, 3, 4, 5 days
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0])


def test_single_pore_volume_constant_flow(constant_flow_setup):
    """
    Test fraction_explained with single pore volume and constant flow.

    Analytical solution: With constant flow of 100 m³/day and pore volume of 200 m³,
    residence time = 2 days. For times >= 2 days from start, fraction should be 1.0,
    for earlier times it should be 0.0 (NaN residence times).
    """
    flow_values, flow_tedges = constant_flow_setup
    pore_volume = 200.0  # Should give 2-day residence time

    result = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Should be 1D array with same length as flow values
    assert result.shape == (len(flow_values),)

    # All values should be either 0.0 or 1.0 for single pore volume
    assert np.all((result == 0.0) | (result == 1.0))

    # Later time points should have fraction = 1.0 (valid residence times)
    # Earlier time points should have fraction = 0.0 (NaN residence times)
    assert np.sum(result == 1.0) > 0  # At least some valid points
    assert np.sum(result == 0.0) > 0  # At least some invalid points


def test_multiple_pore_volumes_gradual_increase(constant_flow_setup, multiple_pore_volumes):
    """
    Test fraction_explained with multiple pore volumes showing gradual increase.

    Analytical solution: With pore volumes [100, 200, 300, 400, 500] m³ and
    constant flow of 100 m³/day, residence times are [1, 2, 3, 4, 5] days.
    Fraction should increase stepwise: 0.0 → 0.2 → 0.4 → 0.6 → 0.8 → 1.0
    """
    flow_values, flow_tedges = constant_flow_setup

    result = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=multiple_pore_volumes,
        direction="extraction_to_infiltration",
    )

    # Should be 1D array with same length as flow values
    assert result.shape == (len(flow_values),)

    # Values should be between 0.0 and 1.0
    assert np.all((result >= 0.0) & (result <= 1.0))

    # Should have increasing fractions over time (monotonic increase)
    # Allow for some plateau regions where fraction doesn't change
    assert np.all(np.diff(result) >= 0)

    # Should start at 0.0 and eventually reach 1.0
    assert result[0] == 0.0
    assert result[-1] == 1.0

    # Check expected fraction values (0.2, 0.4, 0.6, 0.8, 1.0)
    expected_fractions = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    unique_fractions = np.unique(result)

    # All unique fractions should be in expected set
    for frac in unique_fractions:
        assert np.any(np.isclose(frac, expected_fractions, atol=1e-10))


def test_zero_flow_all_invalid():
    """
    Test fraction_explained with zero flow.

    Analytical solution: Zero flow results in infinite/NaN residence times
    for all pore volumes, so fraction should be 0.0 everywhere.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.zeros(len(flow_tedges) - 1)
    pore_volumes = np.array([100.0, 200.0, 300.0])

    result = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    # All fractions should be 0.0 (no valid residence times)
    assert np.allclose(result, 0.0)


def test_very_large_pore_volumes_all_invalid():
    """
    Test fraction_explained with very large pore volumes.

    Analytical solution: Very large pore volumes require more flow history
    than available, resulting in NaN residence times and 0.0 fraction.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    # Very large pore volumes that exceed total flow
    large_pore_volumes = np.array([1e6, 2e6, 3e6])

    result = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=large_pore_volumes,
        direction="extraction_to_infiltration",
    )

    # All fractions should be 0.0 (no valid residence times)
    assert np.allclose(result, 0.0)


def test_retardation_factor_effect(constant_flow_setup):
    """
    Test effect of retardation factor on fraction_explained.

    Analytical solution: Higher retardation factors increase residence times,
    making more residence time calculations result in NaN for early times,
    thus reducing the fraction explained.
    """
    flow_values, flow_tedges = constant_flow_setup
    pore_volume = 200.0

    result_no_retardation = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=1.0,
        direction="extraction_to_infiltration",
    )

    result_with_retardation = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=3.0,
        direction="extraction_to_infiltration",
    )

    # With higher retardation, fewer time points should be valid (lower or equal fractions)
    assert np.all(result_with_retardation <= result_no_retardation)

    # At least some difference should be observed
    assert not np.allclose(result_no_retardation, result_with_retardation)


def test_direction_consistency(constant_flow_setup, multiple_pore_volumes):
    """
    Test that both directions give reasonable results.

    The fraction calculation logic is the same regardless of direction,
    but residence time patterns will be different.
    """
    flow_values, flow_tedges = constant_flow_setup

    result_extraction = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=multiple_pore_volumes,
        direction="extraction_to_infiltration",
    )

    result_infiltration = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=multiple_pore_volumes,
        direction="infiltration_to_extraction",
    )

    # Both should be valid fractions
    assert np.all((result_extraction >= 0.0) & (result_extraction <= 1.0))
    assert np.all((result_infiltration >= 0.0) & (result_infiltration <= 1.0))

    # Both should have some variation (not all zeros or all ones)
    assert len(np.unique(result_extraction)) > 1
    assert len(np.unique(result_infiltration)) > 1


def test_precomputed_residence_time():
    """
    Test using pre-computed residence time array.

    Create a known residence time pattern and verify fraction calculation.
    """
    n_times = 10
    n_pore_volumes = 4

    # Create residence time array with known NaN pattern
    # First 2 time points: all NaN (fraction = 0.0)
    # Next 2 time points: half valid (fraction = 0.5)
    # Last 6 time points: all valid (fraction = 1.0)
    rt = np.full((n_pore_volumes, n_times), np.nan)
    rt[0, 2:] = 1.0  # First pore volume valid from index 2
    rt[1, 2:] = 2.0  # Second pore volume valid from index 2
    rt[2, 4:] = 3.0  # Third pore volume valid from index 4
    rt[3, 4:] = 4.0  # Fourth pore volume valid from index 4

    result = fraction_explained(rt=rt)

    expected = np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    assert np.allclose(result, expected)


def test_edge_case_single_time_point():
    """Test fraction_explained with minimal data (single time point)."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-03", freq="D")
    flow_values = np.array([100.0, 100.0])
    pore_volume = 50.0  # Small enough to give valid residence time

    result = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    assert result.shape == (2,)
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_mixed_valid_invalid_residence_times():
    """
    Test with a mix of valid and invalid pore volumes creating known pattern.

    Use pore volumes where some will give valid residence times and others won't.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-08", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)

    # Mix of small (valid) and large (invalid) pore volumes
    pore_volumes = np.array([100.0, 1e6, 200.0, 2e6, 300.0])  # 3 valid, 2 invalid

    result = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    # Maximum fraction should be 3/5 = 0.6 (only 3 out of 5 pore volumes can be valid)
    assert np.max(result) <= 0.6 + 1e-10  # Small tolerance for floating point

    # Should have fractions that are multiples of 1/5 = 0.2
    unique_fractions = np.unique(result)
    expected_multiples = np.array([0.0, 0.2, 0.4, 0.6])

    for frac in unique_fractions:
        assert np.any(np.isclose(frac, expected_multiples, atol=1e-10))


def test_input_validation_missing_parameters():
    """Test that appropriate errors are raised for missing parameters."""

    # Missing both rt and flow parameters
    with pytest.raises((ValueError, TypeError)):
        fraction_explained()

    # Missing flow_tedges when rt is None
    with pytest.raises(ValueError):
        fraction_explained(
            flow=np.array([100.0, 100.0]),
            aquifer_pore_volumes=200.0,
        )


def test_invalid_direction():
    """Test that invalid direction parameter raises error."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-04", freq="D")
    flow_values = np.array([100.0, 100.0, 100.0])

    with pytest.raises(ValueError, match="direction should be"):
        fraction_explained(
            flow=flow_values,
            flow_tedges=flow_tedges,
            aquifer_pore_volumes=200.0,
            direction="invalid_direction",
        )


def test_array_consistency():
    """Test that array inputs give consistent results."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.array([100.0, 110.0, 105.0, 95.0, 98.0])

    # Test with list vs numpy array pore volumes
    pore_volumes_list = [100.0, 200.0, 300.0]
    pore_volumes_array = np.array([100.0, 200.0, 300.0])

    result_list = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volumes_list,
        direction="extraction_to_infiltration",
    )

    result_array = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volumes_array,
        direction="extraction_to_infiltration",
    )

    np.testing.assert_array_equal(result_list, result_array)


def test_numerical_precision():
    """
    Test numerical precision in fraction calculation.

    Ensure that fractions are calculated precisely, especially for
    cases with many pore volumes.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-11", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)

    # Use 7 pore volumes for non-trivial fraction arithmetic
    n_volumes = 7
    pore_volumes = np.linspace(50, 350, n_volumes)  # Range to get mix of valid/invalid

    result = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    # All fractions should be exact multiples of 1/n_volumes
    expected_increment = 1.0 / n_volumes

    for frac in np.unique(result):
        # Check if frac is close to k/n_volumes for some integer k
        k = np.round(frac / expected_increment)
        expected_frac = k * expected_increment
        assert np.isclose(frac, expected_frac, atol=1e-14)


def test_monotonic_behavior_across_pore_volumes():
    """
    Test that fraction increases monotonically when pore volumes are ordered.

    With ordered pore volumes and constant flow, smaller pore volumes should
    become valid before larger ones, creating monotonic increase pattern.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-15", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)

    # Well-separated pore volumes to ensure clear ordering
    pore_volumes = np.array([100, 300, 500, 700, 900])

    result = fraction_explained(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    # Fraction should be non-decreasing over time
    assert np.all(np.diff(result) >= -1e-10)  # Allow tiny numerical errors

    # Should start at 0.0 and increase
    assert result[0] == 0.0
    assert result[-1] > result[0]
