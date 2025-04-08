"""Tests for advection module functions."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from gwtransport1d.advection import cout_advection, cout_advection_distribution


@pytest.fixture
def sample_data() -> tuple[pd.Series, pd.Series]:
    """Create sample input data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    # Create a sinusoidal concentration pattern
    cin = pd.Series(2 + np.sin(np.linspace(0, 4 * np.pi, len(dates))), index=dates, name="concentration")
    # Create a constant flow rate
    flow = pd.Series(100.0, index=dates, name="flow")
    return cin, flow


@pytest.fixture
def sample_distribution_data() -> tuple[pd.Series, pd.Series, np.ndarray]:
    """Create sample data for distribution-based tests."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    cin = pd.Series(
        np.ones(len(dates)),  # Constant concentration
        index=dates,
        name="concentration",
    )
    flow = pd.Series(100.0, index=dates, name="flow")
    # Create edges for a simple uniform distribution
    edges = np.array([0, 50, 100, 150, 200])
    return cin, flow, edges


def test_cout_advection_conservation(sample_data: tuple[pd.Series, pd.Series]) -> None:
    """Test mass conservation in the advection process."""
    cin, flow = sample_data
    aquifer_pore_volume = 1000.0  # m³

    cout = cout_advection(cin, flow, aquifer_pore_volume)

    # After initial transient period, mean concentration should be conserved
    # Calculate mean after first residence time period
    rt_days = aquifer_pore_volume / flow[0]  # days for one residence time
    steady_state_cin = cin[int(rt_days * 2) :]  # Use data after 2 residence times
    steady_state_cout = cout[int(rt_days * 2) :]

    assert_allclose(
        steady_state_cin.mean(),
        steady_state_cout.mean(),
        rtol=1e-2,
        err_msg="Mean concentration not conserved in steady state",
    )


def test_cout_advection_zero_flow(sample_data: tuple[pd.Series, pd.Series]) -> None:
    """Test behavior with zero flow rate."""
    cin, flow = sample_data
    flow = pd.Series(0.0, index=flow.index)
    aquifer_pore_volume = 1000.0

    cout = cout_advection(cin, flow, aquifer_pore_volume)
    assert np.all(np.isnan(cout)), "Zero flow should result in NaN concentrations"


def test_cout_advection_zero_volume(sample_data: tuple[pd.Series, pd.Series]) -> None:
    """Test behavior with zero aquifer volume."""
    cin, flow = sample_data
    aquifer_pore_volume = 0.0

    cout = cout_advection(cin, flow, aquifer_pore_volume)
    # With zero volume, output should match input immediately
    assert_allclose(cout[1:], cin[1:], rtol=1e-10)


def test_cout_advection_retardation(sample_data: tuple[pd.Series, pd.Series]) -> None:
    """Test effect of retardation factor on transport time."""
    cin, flow = sample_data
    aquifer_pore_volume = 1000.0

    # Compare transport times for different retardation factors
    cout1 = cout_advection(cin, flow, aquifer_pore_volume, retardation_factor=1.0)
    cout2 = cout_advection(cin, flow, aquifer_pore_volume, retardation_factor=2.0)

    # Cross-correlation to find time shift
    corr1 = np.correlate(cin[:-50], cout1[50:], mode="valid")
    corr2 = np.correlate(cin[:-50], cout2[50:], mode="valid")
    shift1 = np.argmax(corr1)
    shift2 = np.argmax(corr2)

    # Time shift should be approximately doubled
    assert abs(shift2 / shift1 - 2.0) < 0.1, "Retardation factor not properly affecting transport time"


def test_cout_advection_distribution_conservation(
    sample_distribution_data: tuple[pd.Series, pd.Series, np.ndarray],
) -> None:
    """Test mass conservation in distribution-based advection."""
    cin, flow, edges = sample_distribution_data

    cout = cout_advection_distribution(cin, flow, edges)

    # Allow for startup transient by skipping first portion
    skip_days = int(edges[-1] / flow[0] * 2)  # Use 2x maximum residence time
    steady_state_cin = cin[skip_days:]
    steady_state_cout = cout[skip_days:]

    assert_allclose(
        steady_state_cin.mean(),
        steady_state_cout.mean(),
        rtol=1e-2,
        err_msg="Mean concentration not conserved in distribution-based advection",
    )


def test_cout_advection_distribution_uniform_response(
    sample_distribution_data: tuple[pd.Series, pd.Series, np.ndarray],
) -> None:
    """Test response to uniform input in distribution-based advection."""
    cin, flow, edges = sample_distribution_data

    # For constant input, output should eventually reach same constant value
    cout = cout_advection_distribution(cin, flow, edges)

    # Check after startup transient
    skip_days = int(edges[-1] / flow[0] * 2)
    steady_state = cout[skip_days:]

    assert_allclose(
        steady_state,
        cin[0],  # Should match input concentration
        rtol=1e-2,
        err_msg="Distribution-based advection not converging to uniform input",
    )


def test_cout_advection_distribution_retardation(
    sample_distribution_data: tuple[pd.Series, pd.Series, np.ndarray],
) -> None:
    """Test retardation effect in distribution-based advection."""
    cin, flow, edges = sample_distribution_data

    # Create step change in input concentration
    cin_step = cin.copy()
    step_index = len(cin) // 2
    cin_step[step_index:] = 2.0

    # Compare breakthrough times with different retardation factors
    cout1 = cout_advection_distribution(cin_step, flow, edges, retardation_factor=1.0)
    cout2 = cout_advection_distribution(cin_step, flow, edges, retardation_factor=2.0)

    # Find when concentration reaches 50% of step change
    threshold = 1.5  # Midway between 1.0 and 2.0
    breakthrough1 = np.where(cout1 > threshold)[0][0]
    breakthrough2 = np.where(cout2 > threshold)[0][0]

    # Breakthrough time should be approximately doubled
    assert abs(breakthrough2 / breakthrough1 - 2.0) < 0.2, (
        "Retardation not properly affecting breakthrough time in distribution"
    )


def test_cout_advection_resampling(sample_data: tuple[pd.Series, pd.Series]) -> None:
    """Test resampling functionality in advection."""
    cin, flow = sample_data
    aquifer_pore_volume = 1000.0

    # Create custom dates for resampling
    resample_dates = pd.date_range(
        start=cin.index[0],
        end=cin.index[-1],
        freq="2D",  # Resample to every other day
    )

    cout = cout_advection(cin, flow, aquifer_pore_volume, resample_dates=resample_dates)

    assert len(cout) == len(resample_dates), "Resampled output length doesn't match requested dates"
    assert (cout.index == resample_dates).all(), "Resampled output index doesn't match requested dates"


def test_invalid_inputs() -> None:
    """Test handling of invalid inputs."""
    # Create minimal valid inputs
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    cin = pd.Series(1.0, index=dates)
    flow = pd.Series(100.0, index=dates)

    # Test negative flow
    with pytest.raises(ValueError):
        cout_advection(cin, pd.Series(-100.0, index=dates), 1000.0)

    # Test negative volume
    with pytest.raises(ValueError):
        cout_advection(cin, flow, -1000.0)

    # Test negative retardation
    with pytest.raises(ValueError):
        cout_advection(cin, flow, 1000.0, retardation_factor=-1.0)

    # Test mismatched indices
    wrong_dates = pd.date_range(start="2023-02-01", end="2023-02-10", freq="D")
    wrong_flow = pd.Series(100.0, index=wrong_dates)
    with pytest.raises(ValueError):
        cout_advection(cin, wrong_flow, 1000.0)


def test_edge_cases(sample_data: tuple[pd.Series, pd.Series]) -> None:
    """Test edge cases and boundary conditions."""
    cin, flow = sample_data
    aquifer_pore_volume = 1000.0

    # Test very small pore volume (should approach direct throughput)
    cout_small = cout_advection(cin, flow, 1e-6)
    assert_allclose(
        cout_small[1:],  # Skip first point due to potential initialization effects
        cin[1:],
        rtol=1e-3,
        err_msg="Very small volume not approaching direct throughput",
    )

    # Test very large pore volume (should show very delayed response)
    cout_large = cout_advection(cin, flow, 1e6)
    # First portion should be NaN due to the long residence time
    assert np.all(np.isnan(cout_large[:100])), "Large volume not showing appropriate initialization period"
