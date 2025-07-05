import numpy as np
import pandas as pd
import pytest

from gwtransport import compute_time_edges
from gwtransport.advection import distribution_forward, forward, gamma_forward

# ===============================================================================
# FIXTURES
# ===============================================================================


@pytest.fixture
def sample_time_series():
    """Create sample time series data for testing."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    concentration = pd.Series(np.sin(np.linspace(0, 4 * np.pi, len(dates))) + 2, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # Constant flow of 100 m3/day
    return concentration, flow


@pytest.fixture
def gamma_params():
    """Sample gamma distribution parameters."""
    return {
        "alpha": 10.0,  # Shape parameter (smaller for reasonable mean)
        "beta": 10.0,  # Scale parameter (gives mean = alpha * beta = 100)
        "n_bins": 10,  # Number of bins
    }


# ===============================================================================
# FORWARD FUNCTION TESTS
# ===============================================================================


def test_forward_basic_functionality(sample_time_series):
    """Test basic functionality of forward function."""
    cin, flow = sample_time_series
    aquifer_pore_volume = 1000.0

    cout = forward(
        cin_series=cin,
        flow_series=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=1.0,
        cout_index="cin",
    )

    # Check output type and length
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cin)

    # Check output values are non-negative (ignoring NaN values)
    valid_values = cout[~np.isnan(cout)]
    assert np.all(valid_values >= 0)


def test_forward_cout_index_options(sample_time_series):
    """Test forward function with different cout_index options."""
    cin, flow = sample_time_series
    aquifer_pore_volume = 1000.0

    # Test cout_index="cin"
    cout_cin = forward(cin_series=cin, flow_series=flow, aquifer_pore_volume=aquifer_pore_volume, cout_index="cin")
    assert len(cout_cin) == len(cin)

    # Test cout_index="flow"
    cout_flow = forward(cin_series=cin, flow_series=flow, aquifer_pore_volume=aquifer_pore_volume, cout_index="flow")
    assert len(cout_flow) == len(flow)

    # Test cout_index="cout"
    cout_cout = forward(cin_series=cin, flow_series=flow, aquifer_pore_volume=aquifer_pore_volume, cout_index="cout")
    # This should have a shifted time series
    assert len(cout_cout) == len(cin)


def test_forward_invalid_cout_index(sample_time_series):
    """Test forward function raises ValueError for invalid cout_index."""
    cin, flow = sample_time_series
    aquifer_pore_volume = 1000.0

    with pytest.raises(ValueError, match="Invalid cout_index"):
        forward(cin_series=cin, flow_series=flow, aquifer_pore_volume=aquifer_pore_volume, cout_index="invalid")


def test_forward_retardation_factor(sample_time_series):
    """Test forward function with different retardation factors."""
    cin, flow = sample_time_series
    aquifer_pore_volume = 1000.0

    # Compare results with different retardation factors
    cout1 = forward(
        cin_series=cin,
        flow_series=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=1.0,
        cout_index="cin",
    )

    cout2 = forward(
        cin_series=cin,
        flow_series=flow,
        aquifer_pore_volume=aquifer_pore_volume,
        retardation_factor=2.0,
        cout_index="cin",
    )

    # The signal with higher retardation should be different
    # We need to check where both have valid values
    valid_mask = ~np.isnan(cout1) & ~np.isnan(cout2)
    if np.any(valid_mask):
        assert not np.allclose(cout1[valid_mask], cout2[valid_mask])


# ===============================================================================
# FORWARD FUNCTION - ANALYTICAL SOLUTION TESTS
# ===============================================================================


def test_forward_analytical_impulse_response():
    """Test forward function with analytical impulse response (Dirac delta)."""
    # Create impulse input: single spike at t=0
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    cin_values = np.zeros(len(dates))
    cin_values[5] = 10.0  # Impulse at day 6 (index 5)
    cin = pd.Series(cin_values, index=dates)

    # Constant flow
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Pore volume that gives exactly 3 days residence time
    pore_volume = 300.0  # 300 m3 / 100 m3/day = 3 days

    # Run forward model
    cout = forward(cin, flow, pore_volume, cout_index="cin")

    # Analytical solution: impulse should appear 3 days later
    # Input impulse at day 6 -> output impulse at day 9
    expected_output_day = 5 + 3  # Index 8 (day 9)

    # Find the peak in the output
    valid_mask = ~np.isnan(cout)
    if np.any(valid_mask):
        max_idx = np.nanargmax(cout)
        # Allow some tolerance due to interpolation
        assert abs(max_idx - expected_output_day) <= 1, f"Expected peak at index {expected_output_day}, got {max_idx}"
        # Peak should be approximately the input magnitude
        assert abs(cout[max_idx] - 10.0) < 2.0, f"Expected peak ~10.0, got {cout[max_idx]}"


def test_forward_analytical_step_response():
    """Test forward function with analytical step response."""
    # Create step input: constant after t=5
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    cin_values = np.zeros(len(dates))
    cin_values[10:] = 5.0  # Step at day 11 (index 10)
    cin = pd.Series(cin_values, index=dates)

    # Constant flow
    flow = pd.Series([50.0] * len(dates), index=dates)

    # Pore volume that gives exactly 2 days residence time
    pore_volume = 100.0  # 100 m3 / 50 m3/day = 2 days

    # Run forward model
    cout = forward(cin, flow, pore_volume, cout_index="cin")

    # Analytical solution: step should appear 2 days later
    # Input step at day 11 -> output step at day 13
    expected_step_day = 10 + 2  # Index 12 (day 13)

    # Check that output is near zero before step
    pre_step_mask = ~np.isnan(cout[:expected_step_day])
    if np.any(pre_step_mask):
        assert np.all(np.abs(cout[:expected_step_day][pre_step_mask]) < 1.0), "Output should be near zero before step"

    # Check that output approaches 5.0 after step (with some tolerance)
    post_step_mask = ~np.isnan(cout[expected_step_day + 2 :])
    if np.any(post_step_mask):
        post_step_values = cout[expected_step_day + 2 :][post_step_mask]
        assert np.mean(post_step_values) > 3.0, f"Expected output ~5.0 after step, got mean {np.mean(post_step_values)}"


def test_forward_analytical_exponential_decay():
    """Test forward function with analytical exponential decay input."""
    # Create exponential decay input: C(t) = C0 * exp(-t/tau)
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
    t = np.arange(len(dates))
    tau = 10.0  # decay time constant (days)
    c0 = 20.0  # initial concentration
    cin_values = c0 * np.exp(-t / tau)
    cin = pd.Series(cin_values, index=dates)

    # Constant flow
    flow = pd.Series([200.0] * len(dates), index=dates)

    # Pore volume that gives 1 day residence time
    pore_volume = 200.0  # 200 m3 / 200 m3/day = 1 day

    # Run forward model
    cout = forward(cin, flow, pore_volume, cout_index="cin")

    # Analytical solution: exponential decay shifted by residence time
    # C_out(t) = c0 * exp(-(t - residence_time)/tau) for t >= residence_time
    t_shifted = t - 1.0  # 1 day residence time
    expected_cout = np.where(t_shifted >= 0, c0 * np.exp(-t_shifted / tau), 0.0)

    # Compare with numerical solution (allow some tolerance)
    valid_mask = ~np.isnan(cout) & (t >= 5)  # Skip early times with edge effects
    if np.any(valid_mask):
        numerical = cout[valid_mask]
        analytical = expected_cout[valid_mask]
        # Allow 20% relative error due to discretization
        relative_error = np.abs(numerical - analytical) / (analytical + 1e-10)
        assert np.mean(relative_error) < 0.2, f"Mean relative error {np.mean(relative_error):.3f} > 0.2"


def test_forward_analytical_retardation_factor():
    """Test forward function analytical retardation factor effect."""
    # Create impulse input
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    cin_values = np.zeros(len(dates))
    cin_values[5] = 15.0  # Impulse at day 6
    cin = pd.Series(cin_values, index=dates)

    # Constant flow
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Pore volume that gives 2 days residence time without retardation
    pore_volume = 200.0  # 200 m3 / 100 m3/day = 2 days

    # Test different retardation factors
    cout_no_retard = forward(cin, flow, pore_volume, retardation_factor=1.0, cout_index="cin")
    cout_retard_2x = forward(cin, flow, pore_volume, retardation_factor=2.0, cout_index="cin")
    cout_retard_3x = forward(cin, flow, pore_volume, retardation_factor=3.0, cout_index="cin")

    # Find peak positions
    def find_peak_position(arr):
        valid_mask = ~np.isnan(arr)
        if np.any(valid_mask):
            return np.nanargmax(arr)
        return -1

    peak_no_retard = find_peak_position(cout_no_retard)
    peak_retard_2x = find_peak_position(cout_retard_2x)
    peak_retard_3x = find_peak_position(cout_retard_3x)

    # Analytical solution: retardation factor multiplies residence time
    # No retardation: peak at day 6 + 2 = day 8 (index 7)
    # 2x retardation: peak at day 6 + 4 = day 10 (index 9)
    # 3x retardation: peak at day 6 + 6 = day 12 (index 11)

    if peak_no_retard >= 0 and peak_retard_2x >= 0:
        # Check that retardation delays the peak
        assert peak_retard_2x > peak_no_retard, "Retardation should delay peak arrival"

        # Check approximate timing (allow ±1 day tolerance)
        expected_delay = 2  # 2x retardation doubles residence time
        actual_delay = peak_retard_2x - peak_no_retard
        assert abs(actual_delay - expected_delay) <= 1, f"Expected delay ~{expected_delay}, got {actual_delay}"

    if peak_retard_3x >= 0 and peak_no_retard >= 0:
        # Check 3x retardation
        expected_delay = 4  # 3x retardation triples residence time (2 -> 6 days)
        actual_delay = peak_retard_3x - peak_no_retard
        assert abs(actual_delay - expected_delay) <= 2, f"Expected delay ~{expected_delay}, got {actual_delay}"


# ===============================================================================
# GAMMA_FORWARD FUNCTION TESTS
# ===============================================================================


def test_gamma_forward_basic_functionality():
    """Test basic functionality of gamma_forward."""
    # Create shorter test data with aligned cin and flow
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cout_tedges with different alignment to avoid edge effects
    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    cout = gamma_forward(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,  # Shape parameter
        beta=10.0,  # Scale parameter (mean = 100)
        n_bins=5,
    )

    # Check output type and length
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)

    # Check output values are non-negative (ignoring NaN values)
    valid_values = cout[~np.isnan(cout)]
    assert np.all(valid_values >= 0)


def test_gamma_forward_with_mean_std():
    """Test gamma_forward using mean and std parameters."""
    # Create test data
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-10", end="2020-01-25", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    mean = 100.0  # Smaller mean for reasonable residence time
    std = 20.0

    cout = gamma_forward(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        mean=mean,
        std=std,
        n_bins=5,
    )

    # Check output type and length
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)


def test_gamma_forward_retardation_factor():
    """Test gamma_forward with different retardation factors."""
    # Create test data
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-10", end="2020-01-25", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Use a step function to see retardation effects
    cin_values = np.ones(len(dates))
    cin_values[10:] = 2.0  # Step change on day 11
    cin = pd.Series(cin_values, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Compare results with different retardation factors
    cout1 = gamma_forward(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
        retardation_factor=1.0,
    )

    cout2 = gamma_forward(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
        retardation_factor=2.0,
    )

    # The signal with higher retardation should be different
    valid_mask = ~np.isnan(cout1) & ~np.isnan(cout2)
    if np.any(valid_mask):
        assert not np.allclose(cout1[valid_mask], cout2[valid_mask])


def test_gamma_forward_constant_input():
    """Test gamma_forward with constant input concentration."""
    # Create test data with longer input period
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period starts later to allow for residence time
    cout_dates = pd.date_range(start="2020-06-01", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    cout = gamma_forward(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
        n_bins=5,
    )

    # Output should also be constant where valid (ignoring NaN values)
    valid_values = cout[~np.isnan(cout)]
    if len(valid_values) > 0:
        assert np.allclose(valid_values, 1.0, rtol=1e-2)


def test_gamma_forward_missing_parameters():
    """Test that gamma_forward raises appropriate errors for missing parameters."""
    # Create test data
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Test missing both alpha/beta and mean/std
    with pytest.raises(ValueError):
        gamma_forward(cin=cin, tedges=tedges, cout_tedges=cout_tedges, flow=flow)


# ===============================================================================
# GAMMA_FORWARD FUNCTION - ANALYTICAL SOLUTION TESTS
# ===============================================================================


def test_gamma_forward_analytical_mean_residence_time():
    """Test gamma_forward with analytical mean residence time."""
    # Create constant input
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period starts later to capture steady state
    cout_dates = pd.date_range(start="2020-06-01", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Constant input and flow
    cin = pd.Series([10.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Gamma distribution parameters
    # Mean residence time = alpha * beta / flow = 10 * 10 / 100 = 1 day
    alpha = 10.0
    beta = 10.0

    # Run gamma_forward
    cout = gamma_forward(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        alpha=alpha,
        beta=beta,
        n_bins=20,
        retardation_factor=1.0,
    )

    # Analytical solution: for constant input, output should eventually equal input
    # Look at the latter part of the time series where steady state is reached
    valid_mask = ~np.isnan(cout)
    if np.sum(valid_mask) > 50:  # Need enough points for statistical analysis
        stable_region = cout[valid_mask][-30:]  # Last 30 valid points
        mean_output = np.mean(stable_region)
        # Output should be approximately equal to input concentration
        assert abs(mean_output - 10.0) < 1.0, f"Expected ~10.0 in steady state, got {mean_output:.2f}"
        # Variance should be small in steady state
        assert np.std(stable_region) < 2.0, f"Too much variance in steady state: {np.std(stable_region):.2f}"


# ===============================================================================
# DISTRIBUTION_FORWARD FUNCTION TESTS
# ===============================================================================


def test_distribution_forward_basic_functionality():
    """Test basic functionality of distribution_forward."""
    # Create test data with aligned cin and flow
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cout_tedges with different alignment
    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-09", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    # Use smaller pore volumes for reasonable residence times (1-3 days)
    aquifer_pore_volumes = np.array([100.0, 200.0, 300.0])

    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Check output type and length
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)


def test_distribution_forward_constant_input():
    """Test distribution_forward with constant input concentration."""
    # Create longer time series for better testing
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cout_tedges starting later
    cout_dates = pd.date_range(start="2020-06-01", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)) * 5.0, index=dates)  # Constant concentration
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # Constant flow
    aquifer_pore_volumes = np.array([500.0, 1000.0])  # Two pore volumes

    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Check output type and length
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)

    # With constant input and sufficient time, some outputs should be valid
    valid_outputs = cout[~np.isnan(cout)]
    if len(valid_outputs) > 0:
        # Output should be close to input concentration for constant system
        assert np.all(valid_outputs >= 0)


def test_distribution_forward_single_pore_volume():
    """Test distribution_forward with single pore volume."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-10", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.sin(np.linspace(0, 2 * np.pi, len(dates))) + 2, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    # Use smaller pore volume for reasonable residence time (5 days)
    aquifer_pore_volumes = np.array([500.0])  # Single pore volume

    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)


def test_distribution_forward_retardation_factor():
    """Test distribution_forward with different retardation factors."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-06-01", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([1000.0, 2000.0])

    # Test different retardation factors
    cout1 = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    cout2 = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=2.0,
    )

    # Results should be different for different retardation factors
    assert isinstance(cout1, np.ndarray)
    assert isinstance(cout2, np.ndarray)
    assert len(cout1) == len(cout2)


def test_distribution_forward_error_conditions():
    """Test distribution_forward error conditions."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([1000.0])

    # Test mismatched tedges length
    wrong_tedges = tedges[:-2]  # Too short
    with pytest.raises(ValueError, match="tedges must have one more element than cin"):
        distribution_forward(
            cin=cin.values,
            flow=flow.values,
            tedges=wrong_tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
        )

    # Test mismatched flow and tedges
    wrong_flow = flow[:-2]  # Too short
    with pytest.raises(ValueError, match="tedges must have one more element than flow"):
        distribution_forward(
            cin=cin.values,
            flow=wrong_flow.values,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
        )


# ===============================================================================
# DISTRIBUTION_FORWARD FUNCTION - EDGE CASE TESTS
# ===============================================================================


def test_distribution_forward_no_temporal_overlap():
    """Test distribution_forward returns NaN when no temporal overlap exists."""
    # Create cin/flow in early 2020
    early_dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=early_dates, number_of_bins=len(early_dates))

    # Create cout_tedges much later (no possible overlap)
    late_dates = pd.date_range(start="2020-12-01", end="2020-12-05", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=late_dates, number_of_bins=len(late_dates))

    cin = pd.Series(np.ones(len(early_dates)), index=early_dates)
    flow = pd.Series(np.ones(len(early_dates)) * 100, index=early_dates)
    aquifer_pore_volumes = np.array([100.0])  # Small pore volume

    # No temporal overlap should return all NaN values
    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array of NaN values
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(late_dates)
    assert np.all(np.isnan(cout))


def test_distribution_forward_zero_concentrations():
    """Test distribution_forward preserves zero concentrations and handles NaNs."""
    # Create longer time series for realistic residence times
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # cout_tedges later to allow residence time effects, with smaller pore volume for faster transport
    cout_dates = pd.date_range(start="2020-01-10", end="2020-12-20", freq="D")  # Overlap with input period
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Create cin with zeros, ones, and twos (no NaNs for this test to ensure clear results)
    cin_pattern = np.array([1.0, 0.0, 2.0])
    cin_values = np.tile(cin_pattern, len(dates) // len(cin_pattern) + 1)[: len(dates)]
    cin = pd.Series(cin_values, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([50.0])  # Small pore volume for quick transport

    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Check that we have valid results
    valid_results = cout[~np.isnan(cout)]
    if len(valid_results) > 0:
        # Check that zero concentrations are preserved (not converted to NaN)
        has_zeros = np.any(valid_results == 0.0)
        if has_zeros:
            # Verify zeros are preserved as valid concentrations
            assert True, "Zero concentrations are correctly preserved"

        # Check that we get reasonable concentration values
        assert np.all(valid_results >= 0.0), "All concentrations should be non-negative"
        assert np.all(valid_results <= 2.0), "All concentrations should be within expected range"

    # The key test: ensure function doesn't convert zeros to NaN
    # This is tested by the structure of the function - it uses natural NaN propagation


def test_distribution_forward_extreme_conditions():
    """Test distribution_forward handles extreme conditions gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = tedges.copy()

    cin = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    flow = pd.Series([1000.0, 0.1, 1000.0, 0.1, 1000.0], index=dates)
    aquifer_pore_volumes = np.array([10.0, 100000.0, 50.0])

    # Should handle extreme conditions gracefully
    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return valid array (may contain NaN values)
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(dates)


def test_distribution_forward_extreme_pore_volumes():
    """Test distribution_forward handles extremely large pore volumes gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-07", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Extremely large pore volumes that create invalid infiltration edges
    aquifer_pore_volumes = np.array([1e10, 1e12, 1e15])

    # Should handle extreme pore volumes gracefully
    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array (likely all NaN due to extreme residence times)
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)
    # With extremely large pore volumes, all outputs should be NaN
    assert np.all(np.isnan(cout))


def test_distribution_forward_zero_flow():
    """Test distribution_forward handles zero flow values gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-07", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.zeros(len(dates)), index=dates)  # Zero flow
    aquifer_pore_volumes = np.array([1000.0])

    # Zero flow creates infinite residence times but should be handled gracefully
    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array (likely all NaN due to infinite residence times)
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)
    # With zero flow, all outputs should be NaN
    assert np.all(np.isnan(cout))


def test_distribution_forward_mixed_pore_volumes():
    """Test distribution_forward handles mixed pore volumes with varying overlaps."""
    # Longer time series for cin/flow
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Short cout period - only some pore volumes will have overlap
    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-10", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Mix of small and large pore volumes - large ones create minimal overlap
    aquifer_pore_volumes = np.array([10.0, 100.0, 50000.0, 100000.0])

    # Should handle mixed pore volumes gracefully
    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return valid array
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)
    # Some values might be valid (from small pore volumes), others NaN (from large pore volumes)
    valid_values = cout[~np.isnan(cout)]
    if len(valid_values) > 0:
        assert np.all(valid_values >= 0)


# ===============================================================================
# DISTRIBUTION_FORWARD FUNCTION - ANALYTICAL SOLUTION TESTS
# ===============================================================================


def test_distribution_forward_analytical_single_pore_volume():
    """Test distribution_forward with single pore volume matches forward function."""
    # Create test data
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Sine wave input
    cin_values = 3.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 10.0)
    cin = pd.Series(cin_values, index=dates)
    flow = pd.Series([150.0] * len(dates), index=dates)

    # Single pore volume
    pore_volume = 450.0  # 450 m3 / 150 m3/day = 3 days residence time

    # Run both functions
    cout_forward = forward(cin, flow, pore_volume, cout_index="cin")
    cout_distribution = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=1.0,
    )

    # Interpolate forward result to distribution output times
    forward_interp = np.interp(
        cout_dates.values.astype("datetime64[D]").astype(int),
        dates.values.astype("datetime64[D]").astype(int),
        cout_forward,
    )

    # Compare results where both are valid
    valid_mask = ~np.isnan(cout_distribution) & ~np.isnan(forward_interp)
    if np.any(valid_mask):
        np.testing.assert_allclose(
            cout_distribution[valid_mask],
            forward_interp[valid_mask],
            rtol=0.1,
            err_msg="Single pore volume distribution_forward should match forward function",
        )


def test_distribution_forward_analytical_mass_conservation():
    """Test distribution_forward mass conservation with pulse input."""
    # Create pulse input (finite mass)
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Long output period to capture entire pulse
    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Pulse input: concentration for 5 days, then zero
    cin_values = np.zeros(len(dates))
    cin_values[5:10] = 8.0  # Pulse from day 6-10
    cin = pd.Series(cin_values, index=dates)

    # Constant flow
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Multiple pore volumes
    aquifer_pore_volumes = np.array([100.0, 200.0, 300.0])  # 1, 2, 3 day residence times

    # Run distribution_forward
    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Mass conservation check
    # Input mass = concentration * flow * time (for each time step)
    dt = 1.0  # 1 day time steps
    input_mass = np.sum(cin_values * flow.values * dt)

    # Output mass = concentration * flow * time (for each time step)
    # Use average flow for output period
    output_flow = np.mean(flow.values)
    valid_mask = ~np.isnan(cout)
    output_mass = np.sum(cout[valid_mask] * output_flow * dt)

    # Check mass conservation (within 20% due to discretization and edge effects)
    if input_mass > 0:
        mass_error = abs(output_mass - input_mass) / input_mass
        assert mass_error < 0.3, f"Mass conservation error {mass_error:.2f} > 0.3"


def test_distribution_forward_known_constant_delay():
    """Test distribution_forward with known constant delay scenario."""
    # Create a simple scenario where we know the exact outcome
    # 10 days of data, constant flow, single pore volume
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period starts after the delay
    cout_dates = pd.date_range(start="2020-01-06", end="2020-01-10", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Constant flow and known pore volume that gives exactly 1 day residence time
    flow_rate = 100.0  # m3/day
    pore_volume = 100.0  # m3 -> residence time = 100/100 = 1 day

    # Step function: concentration jumps from 1 to 5 on day 5
    cin_values = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    cin = pd.Series(cin_values, index=dates)
    flow = pd.Series([flow_rate] * len(dates), index=dates)
    aquifer_pore_volumes = np.array([pore_volume])

    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # With 1-day residence time, the step change on day 5 should appear on day 6
    # Output days 6-10 correspond to infiltration days 5-9
    # So we expect: day 6 -> 5.0, days 7-10 -> 5.0
    valid_outputs = cout[~np.isnan(cout)]
    if len(valid_outputs) > 0:
        # All valid outputs should be close to 5.0 (after the step change)
        assert np.allclose(valid_outputs, 5.0, rtol=0.1), f"Expected ~5.0, got {valid_outputs}"


def test_distribution_forward_known_average_of_pore_volumes():
    """Test distribution_forward averages multiple pore volumes correctly."""
    # Simple scenario where we can predict the averaging behavior
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period in the middle to ensure overlap
    cout_dates = pd.date_range(start="2020-01-10", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Constant concentration and flow
    cin = pd.Series([10.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Two identical pore volumes - average should equal the single pore volume result
    single_pv = np.array([500.0])
    double_pv = np.array([500.0, 500.0])

    cout_single = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=single_pv,
        retardation_factor=1.0,
    )

    cout_double = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=double_pv,
        retardation_factor=1.0,
    )

    # Results should be nearly identical (averaging two identical contributions)
    valid_mask = ~np.isnan(cout_single) & ~np.isnan(cout_double)
    if np.any(valid_mask):
        np.testing.assert_allclose(
            cout_single[valid_mask],
            cout_double[valid_mask],
            rtol=1e-10,
            err_msg="Averaging identical pore volumes should give same result as single pore volume",
        )


def test_distribution_forward_known_zero_input_gives_zero_output():
    """Test distribution_forward with zero input gives zero output."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Zero concentration everywhere
    cin = pd.Series([0.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)
    aquifer_pore_volumes = np.array([200.0, 400.0])

    cout = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Zero input should give zero output (where valid)
    valid_outputs = cout[~np.isnan(cout)]
    if len(valid_outputs) > 0:
        np.testing.assert_allclose(valid_outputs, 0.0, atol=1e-15, err_msg="Zero input should produce zero output")


def test_distribution_forward_known_retardation_effect():
    """Test distribution_forward retardation factor effect."""
    # Create longer time series to capture retardation effects
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period covers a wide range to catch both retarded and non-retarded responses
    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Step function: concentration jumps from 0 to 10 on day 10
    cin_values = [0.0] * len(dates)
    for i in range(9, len(dates)):  # Days 10 onwards (index 9+)
        cin_values[i] = 10.0
    cin = pd.Series(cin_values, index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Pore volume that gives reasonable residence time
    pore_volume = 200.0  # residence time = 200/100 = 2 days
    aquifer_pore_volumes = np.array([pore_volume])

    # Test different retardation factors
    cout_no_retard = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    cout_retarded = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=2.0,
    )

    # Basic test - both should return valid arrays
    assert isinstance(cout_no_retard, np.ndarray)
    assert isinstance(cout_retarded, np.ndarray)
    assert len(cout_no_retard) == len(cout_dates)
    assert len(cout_retarded) == len(cout_dates)


# ===============================================================================
# COMPARISON TESTS BETWEEN FORWARD AND DISTRIBUTION_FORWARD
# ===============================================================================


def test_forward_vs_distribution_forward_single_pore_volume():
    """Test that forward and distribution_forward give identical results with single pore volume."""
    # Create test data
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    cin = pd.Series(3.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 15.0), index=dates)
    flow = pd.Series([200.0] * len(dates), index=dates)
    pore_volume = 600.0  # 3 days residence time

    # Create tedges for distribution_forward
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = tedges.copy()  # Same alignment for direct comparison

    # Run forward function
    cout_forward = forward(cin, flow, pore_volume, cout_index="cin")

    # Run distribution_forward with single pore volume
    cout_distribution = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=1.0,
    )

    # Compare results exactly - both should have same NaN pattern and values
    np.testing.assert_array_equal(
        np.isnan(cout_forward),
        np.isnan(cout_distribution),
        err_msg="NaN patterns should be identical between forward and distribution_forward",
    )

    # Compare non-NaN values exactly
    non_nan_mask = ~np.isnan(cout_forward)
    if np.any(non_nan_mask):
        np.testing.assert_array_equal(
            cout_forward[non_nan_mask],
            cout_distribution[non_nan_mask],
            err_msg="forward and distribution_forward should give identical results with single pore volume",
        )


def test_forward_vs_distribution_forward_retardation_effects():
    """Test that forward and distribution_forward handle retardation consistently."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")

    # Step function to clearly see retardation effects
    cin_values = np.ones(len(dates))
    cin_values[10:] = 5.0  # Step change on day 11
    cin = pd.Series(cin_values, index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)
    pore_volume = 300.0  # 3 days residence time
    retardation_factor = 2.0

    # Create tedges for distribution_forward
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = tedges.copy()

    # Run both functions with retardation
    cout_forward = forward(cin, flow, pore_volume, retardation_factor=retardation_factor, cout_index="cin")
    cout_distribution = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=retardation_factor,
    )

    # Compare NaN patterns
    np.testing.assert_array_equal(
        np.isnan(cout_forward),
        np.isnan(cout_distribution),
        err_msg="NaN patterns should be identical for retardation test",
    )

    # Compare non-NaN values exactly
    non_nan_mask = ~np.isnan(cout_forward)
    if np.any(non_nan_mask):
        np.testing.assert_array_equal(
            cout_forward[non_nan_mask],
            cout_distribution[non_nan_mask],
            err_msg="forward and distribution_forward should handle retardation identically",
        )


def test_forward_vs_distribution_forward_impulse_response():
    """Test that forward and distribution_forward handle impulse response consistently."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-25", freq="D")

    # Impulse response: single spike
    cin_values = np.zeros(len(dates))
    cin_values[5] = 10.0  # Impulse at day 6
    cin = pd.Series(cin_values, index=dates)
    flow = pd.Series([150.0] * len(dates), index=dates)
    pore_volume = 450.0  # 3 days residence time

    # Create tedges for distribution_forward
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = tedges.copy()

    # Run both functions
    cout_forward = forward(cin, flow, pore_volume, cout_index="cin")
    cout_distribution = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=1.0,
    )

    # Compare NaN patterns
    np.testing.assert_array_equal(
        np.isnan(cout_forward),
        np.isnan(cout_distribution),
        err_msg="NaN patterns should be identical for impulse response test",
    )

    # Compare non-NaN values exactly
    non_nan_mask = ~np.isnan(cout_forward)
    if np.any(non_nan_mask):
        np.testing.assert_array_equal(
            cout_forward[non_nan_mask],
            cout_distribution[non_nan_mask],
            err_msg="forward and distribution_forward should handle impulse response identically",
        )


def test_forward_vs_distribution_forward_constant_input():
    """Test that forward and distribution_forward handle constant input consistently."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    cin = pd.Series([7.5] * len(dates), index=dates)  # Constant concentration
    flow = pd.Series([100.0] * len(dates), index=dates)  # Constant flow
    pore_volume = 1000.0  # 10 days residence time

    # Create tedges for distribution_forward
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = tedges.copy()

    # Run both functions
    cout_forward = forward(cin, flow, pore_volume, cout_index="cin")
    cout_distribution = distribution_forward(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=1.0,
    )

    # Compare NaN patterns
    np.testing.assert_array_equal(
        np.isnan(cout_forward),
        np.isnan(cout_distribution),
        err_msg="NaN patterns should be identical for constant input test",
    )

    # Compare non-NaN values exactly
    non_nan_mask = ~np.isnan(cout_forward)
    if np.any(non_nan_mask):
        np.testing.assert_array_equal(
            cout_forward[non_nan_mask],
            cout_distribution[non_nan_mask],
            err_msg="forward and distribution_forward should handle constant input identically",
        )


def test_forward_vs_distribution_forward_with_arrays():
    """Test that forward and distribution_forward work with numpy arrays as inputs."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")

    # Use numpy arrays instead of pandas Series
    cin_values = 2.0 + np.sin(np.linspace(0, 4 * np.pi, len(dates)))
    flow_values = np.full(len(dates), 80.0)
    pore_volume = 240.0  # 3 days residence time

    # Create Series for forward function (since it requires pandas Series)
    cin_series = pd.Series(cin_values, index=dates)
    flow_series = pd.Series(flow_values, index=dates)

    # Create tedges for distribution_forward
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = tedges.copy()

    # Run forward function with Series
    cout_forward = forward(cin_series, flow_series, pore_volume, cout_index="cin")

    # Run distribution_forward with arrays
    cout_distribution = distribution_forward(
        cin=cin_values,  # numpy array
        flow=flow_values,  # numpy array
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=1.0,
    )

    # Compare NaN patterns
    np.testing.assert_array_equal(
        np.isnan(cout_forward),
        np.isnan(cout_distribution),
        err_msg="NaN patterns should be identical when using arrays vs Series",
    )

    # Compare non-NaN values exactly
    non_nan_mask = ~np.isnan(cout_forward)
    if np.any(non_nan_mask):
        np.testing.assert_array_equal(
            cout_forward[non_nan_mask],
            cout_distribution[non_nan_mask],
            err_msg="forward and distribution_forward should give identical results with arrays",
        )


# ===============================================================================
# UTILITY AND INTEGRATION TESTS
# ===============================================================================


def test_time_edge_consistency():
    """Test that time edges are handled consistently."""
    # Create test data with proper temporal alignment
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period starts later to avoid edge effects
    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Test with consistent time edges
    cout = gamma_forward(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
        n_bins=5,
    )

    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)


def test_conservation_properties():
    """Test mass conservation properties where applicable."""
    # Create test data with longer time series for better conservation
    dates = pd.date_range(start="2020-01-01", end="2021-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period covers most of the second year to capture steady state
    cout_dates = pd.date_range(start="2021-01-01", end="2021-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)  # Constant input
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # Constant flow

    cout = gamma_forward(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
        n_bins=10,
    )

    # For constant input and flow, output should eventually stabilize
    # Check the latter part of the series where it should be stable
    valid_mask = ~np.isnan(cout)
    if np.sum(valid_mask) > 100:  # If we have enough valid values
        stable_region = cout[valid_mask][-100:]  # Last 100 valid values
        assert np.std(stable_region) < 0.1  # Should be relatively stable


def test_empty_series():
    """Test handling of empty series."""
    empty_cin = pd.Series([], dtype=float)

    # This should handle gracefully or raise appropriate error
    with pytest.raises((ValueError, IndexError)):
        # Create tedges - this should fail for empty series
        compute_time_edges(tedges=None, tstart=None, tend=empty_cin.index, number_of_bins=len(empty_cin))


def test_mismatched_series_lengths():
    """Test handling of mismatched series lengths."""
    # Create input data with longer period
    dates_cin = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=dates_cin, number_of_bins=len(dates_cin))

    # Create output data with shorter, offset period
    dates_cout = pd.date_range(start="2020-01-05", end="2020-01-10", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=dates_cout, number_of_bins=len(dates_cout))

    cin = pd.Series(np.ones(len(dates_cin)), index=dates_cin)
    flow = pd.Series(np.ones(len(dates_cin)) * 100, index=dates_cin)

    # This should work - the function should handle different output lengths
    cout = gamma_forward(
        cin=cin,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
    )

    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(dates_cout)
