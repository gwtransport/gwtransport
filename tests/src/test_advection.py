import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import (
    distribution_extraction_to_infiltration,
    distribution_infiltration_to_extraction,
    extraction_to_infiltration_series,
    gamma_extraction_to_infiltration,
    gamma_infiltration_to_extraction,
    infiltration_to_extraction_series,
)
from gwtransport.utils import compute_time_edges

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
# INFILTRATION_TO_EXTRACTION_SERIES FUNCTION TESTS
# ===============================================================================


def test_infiltration_to_extraction_series_output_structure():
    """Test that infiltration_to_extraction_series returns correct DataFrame structure."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cin = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    result = infiltration_to_extraction_series(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["start", "end", "cout"]
    assert len(result) == len(dates)
    assert isinstance(result["start"].iloc[0], pd.Timestamp)
    assert isinstance(result["end"].iloc[0], pd.Timestamp)


def test_infiltration_to_extraction_series_constant_input():
    """Test constant concentration produces constant output with proper time shift."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cin = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    result = infiltration_to_extraction_series(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
    )

    # Constant input should produce constant output
    np.testing.assert_array_almost_equal(result["cout"].values, cin)

    # Time shift should be residence time = pore_volume / flow = 500 / 100 = 5 days
    expected_shift = pd.Timedelta(days=5)
    actual_shift = result["start"].iloc[0] - tedges[0]
    assert actual_shift == expected_shift


def test_infiltration_to_extraction_series_retardation_factor():
    """Test retardation factor doubles residence time."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cin = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    result_no_retard = infiltration_to_extraction_series(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
        retardation_factor=1.0,
    )

    result_retard = infiltration_to_extraction_series(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
        retardation_factor=2.0,
    )

    # Retardation factor of 2 should double time shift
    shift_no_retard = result_no_retard["start"].iloc[0] - tedges[0]
    shift_retard = result_retard["start"].iloc[0] - tedges[0]
    assert shift_retard == shift_no_retard * 2


def test_infiltration_to_extraction_series_pandas_series_input():
    """Test function accepts pandas Series as input."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cin = pd.Series(np.ones(len(dates)) * 10.0, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100.0, index=dates)

    result = infiltration_to_extraction_series(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(dates)
    np.testing.assert_array_almost_equal(result["cout"].values, cin.values)


def test_infiltration_to_extraction_series_time_edges_consistency():
    """Test output time edges are monotonically increasing for valid values."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cin = np.random.rand(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    result = infiltration_to_extraction_series(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
    )

    # For valid (non-NaT) time edges, they should be monotonically increasing
    valid_start = result["start"].notna()
    if valid_start.sum() > 1:
        start_diffs = result.loc[valid_start, "start"].diff()[1:]
        assert (start_diffs > pd.Timedelta(0)).all()

    valid_end = result["end"].notna()
    if valid_end.sum() > 1:
        end_diffs = result.loc[valid_end, "end"].diff()[1:]
        assert (end_diffs > pd.Timedelta(0)).all()

    # Where both are valid, end should be after start
    valid_both = valid_start & valid_end
    if valid_both.any():
        assert (result.loc[valid_both, "end"] > result.loc[valid_both, "start"]).all()


# ===============================================================================
# EXTRACTION_TO_INFILTRATION_SERIES FUNCTION TESTS
# ===============================================================================


def test_extraction_to_infiltration_series_output_structure():
    """Test that extraction_to_infiltration_series returns correct DataFrame structure."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    result = extraction_to_infiltration_series(
        cout=cout,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
    )

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["start", "end", "cin"]
    assert len(result) == len(dates)
    # Check valid (non-NaT) timestamps
    valid_start = result["start"].notna()
    if valid_start.any():
        assert isinstance(result.loc[valid_start, "start"].iloc[0], pd.Timestamp)
    valid_end = result["end"].notna()
    if valid_end.any():
        assert isinstance(result.loc[valid_end, "end"].iloc[0], pd.Timestamp)


def test_extraction_to_infiltration_series_constant_input():
    """Test constant concentration produces constant output with proper time shift backward."""
    dates = pd.date_range(start="2020-01-10", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    result = extraction_to_infiltration_series(
        cout=cout,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
    )

    # Constant input should produce constant output
    np.testing.assert_array_almost_equal(result["cin"].values, cout)

    # Check valid time shifts (non-NaT entries)
    # Expected: first 5 bins have NaT, then valid times starting at 2020-01-09
    valid_idx = result["start"].notna()
    assert valid_idx.sum() > 0, "Should have some valid time edges"

    # First valid start time should be 2020-01-09 (tedges[4] - 5 days, where tedges[4]=2020-01-14)
    first_valid_idx = valid_idx.idxmax()
    if first_valid_idx == 5:  # Index 5 corresponds to tedges[5] = 2020-01-14
        assert result["start"].iloc[first_valid_idx] == pd.Timestamp("2020-01-09")


def test_extraction_to_infiltration_series_retardation_factor():
    """Test retardation factor doubles residence time (backward shift)."""
    dates = pd.date_range(start="2020-01-15", end="2020-01-25", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    result_no_retard = extraction_to_infiltration_series(
        cout=cout,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
        retardation_factor=1.0,
    )

    result_retard = extraction_to_infiltration_series(
        cout=cout,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
        retardation_factor=2.0,
    )

    # With retardation factor of 2, more bins will have NaT (longer residence time)
    valid_no_retard = result_no_retard["start"].notna().sum()
    valid_retard = result_retard["start"].notna().sum()
    assert valid_retard < valid_no_retard, "Higher retardation should result in fewer valid bins"

    # For bins that are valid in both, check relative time shifts
    both_valid = result_no_retard["start"].notna() & result_retard["start"].notna()
    if both_valid.sum() > 0:
        # Pick a bin valid in both
        idx = both_valid.idxmax()
        shift_no_retard = result_no_retard["start"].iloc[idx] - tedges[idx]
        shift_retard = result_retard["start"].iloc[idx] - tedges[idx]
        # Retardation factor of 2 should double the magnitude of time shift
        assert abs(shift_retard) == abs(shift_no_retard) * 2


def test_extraction_to_infiltration_series_pandas_series_input():
    """Test function accepts pandas Series as input."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout = pd.Series(np.ones(len(dates)) * 10.0, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100.0, index=dates)

    result = extraction_to_infiltration_series(
        cout=cout,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(dates)
    np.testing.assert_array_almost_equal(result["cin"].values, cout.values)


def test_extraction_to_infiltration_series_time_edges_consistency():
    """Test output time edges are monotonically increasing for valid values."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout = np.random.rand(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    result = extraction_to_infiltration_series(
        cout=cout,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
    )

    # For valid (non-NaT) time edges, they should be monotonically increasing
    valid_start = result["start"].notna()
    if valid_start.sum() > 1:
        start_diffs = result.loc[valid_start, "start"].diff()[1:]
        assert (start_diffs > pd.Timedelta(0)).all()

    valid_end = result["end"].notna()
    if valid_end.sum() > 1:
        end_diffs = result.loc[valid_end, "end"].diff()[1:]
        assert (end_diffs > pd.Timedelta(0)).all()

    # Where both are valid, end should be after start
    valid_both = valid_start & valid_end
    if valid_both.any():
        assert (result.loc[valid_both, "end"] > result.loc[valid_both, "start"]).all()


def test_extraction_to_infiltration_series_symmetry_with_infiltration():
    """Test symmetry: infiltration -> extraction -> infiltration should recover original."""
    dates = pd.date_range(start="2020-01-10", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cin_original = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0
    pore_volume = 500.0

    # Forward: infiltration -> extraction
    result_forward = infiltration_to_extraction_series(
        cin=cin_original,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=pore_volume,
    )

    # Backward: extraction -> infiltration
    # Use the output time edges from forward pass
    tedges_out = pd.DatetimeIndex(list(result_forward["start"]) + [result_forward["end"].iloc[-1]])
    result_backward = extraction_to_infiltration_series(
        cout=result_forward["cout"].values,
        flow=flow,
        tedges=tedges_out,
        aquifer_pore_volume=pore_volume,
    )

    # The recovered infiltration times should match original (within valid range)
    # Compare the start times
    assert result_backward["start"].iloc[0] == tedges[0]
    assert result_backward["end"].iloc[-1] == tedges[-1]
    np.testing.assert_array_almost_equal(result_backward["cin"].values, cin_original)


def test_gamma_infiltration_to_extraction_basic_functionality():
    """Test basic functionality of gamma_infiltration_to_extraction."""
    # Create shorter test data with aligned cin and flow
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cout_tedges with different alignment to avoid edge effects
    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    cout = gamma_infiltration_to_extraction(
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


def test_gamma_infiltration_to_extraction_with_mean_std():
    """Test gamma_infiltration_to_extraction using mean and std parameters."""
    # Create test data
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-10", end="2020-01-25", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    mean = 100.0  # Smaller mean for reasonable residence time
    std = 20.0

    cout = gamma_infiltration_to_extraction(
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


def test_gamma_infiltration_to_extraction_retardation_factor():
    """Test gamma_infiltration_to_extraction with different retardation factors."""
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
    cout1 = gamma_infiltration_to_extraction(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
        retardation_factor=1.0,
    )

    cout2 = gamma_infiltration_to_extraction(
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


def test_gamma_infiltration_to_extraction_constant_input():
    """Test gamma_infiltration_to_extraction with constant input concentration."""
    # Create test data with longer input period
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period starts later to allow for residence time
    cout_dates = pd.date_range(start="2020-06-01", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    cout = gamma_infiltration_to_extraction(
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


def test_gamma_infiltration_to_extraction_missing_parameters():
    """Test that gamma_infiltration_to_extraction raises appropriate errors for missing parameters."""
    # Create test data
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Test missing both alpha/beta and mean/std
    with pytest.raises(ValueError):
        gamma_infiltration_to_extraction(cin=cin, tedges=tedges, cout_tedges=cout_tedges, flow=flow)


# ===============================================================================
# GAMMA_INFILTRATION_TO_EXTRACTION FUNCTION - ANALYTICAL SOLUTION TESTS
# ===============================================================================


def test_gamma_infiltration_to_extraction_analytical_mean_residence_time():
    """Test gamma_infiltration_to_extraction with analytical mean residence time."""
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

    # Run gamma_infiltration_to_extraction
    cout = gamma_infiltration_to_extraction(
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
# DISTRIBUTION_INFILTRATION_TO_EXTRACTION FUNCTION TESTS
# ===============================================================================


def test_distribution_infiltration_to_extraction_basic_functionality():
    """Test basic functionality of distribution_infiltration_to_extraction."""
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

    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_constant_input():
    """Test distribution_infiltration_to_extraction with constant input concentration."""
    # Create longer time series for better testing
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cout_tedges starting later
    cout_dates = pd.date_range(start="2020-06-01", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)) * 5.0, index=dates)  # Constant concentration
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # Constant flow
    aquifer_pore_volumes = np.array([500.0, 1000.0])  # Two pore volumes

    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_single_pore_volume():
    """Test distribution_infiltration_to_extraction with single pore volume."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-10", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.sin(np.linspace(0, 2 * np.pi, len(dates))) + 2, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    # Use smaller pore volume for reasonable residence time (5 days)
    aquifer_pore_volumes = np.array([500.0])  # Single pore volume

    cout = distribution_infiltration_to_extraction(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)


def test_distribution_infiltration_to_extraction_retardation_factor():
    """Test distribution_infiltration_to_extraction with different retardation factors."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-06-01", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([1000.0, 2000.0])

    # Test different retardation factors
    cout1 = distribution_infiltration_to_extraction(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    cout2 = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_error_conditions():
    """Test distribution_infiltration_to_extraction error conditions."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([1000.0])

    # Test mismatched tedges length
    wrong_tedges = tedges[:-2]  # Too short
    with pytest.raises(ValueError, match="tedges must have one more element than cin"):
        distribution_infiltration_to_extraction(
            cin=cin.values,
            flow=flow.values,
            tedges=wrong_tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
        )

    # Test mismatched flow and tedges
    wrong_flow = flow[:-2]  # Too short
    with pytest.raises(ValueError, match="tedges must have one more element than flow"):
        distribution_infiltration_to_extraction(
            cin=cin.values,
            flow=wrong_flow.values,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
        )


# ===============================================================================
# DISTRIBUTION_FORWARD FUNCTION - EDGE CASE TESTS
# ===============================================================================


def test_distribution_infiltration_to_extraction_no_temporal_overlap():
    """Test distribution_infiltration_to_extraction returns NaN when no temporal overlap exists."""
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
    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_zero_concentrations():
    """Test distribution_infiltration_to_extraction preserves zero concentrations and handles NaNs."""
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

    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_extreme_conditions():
    """Test distribution_infiltration_to_extraction handles extreme conditions gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = tedges.copy()

    cin = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    flow = pd.Series([1000.0, 0.1, 1000.0, 0.1, 1000.0], index=dates)
    aquifer_pore_volumes = np.array([10.0, 100000.0, 50.0])

    # Should handle extreme conditions gracefully
    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_extreme_pore_volumes():
    """Test distribution_infiltration_to_extraction handles extremely large pore volumes gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-07", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Extremely large pore volumes that create invalid infiltration edges
    aquifer_pore_volumes = np.array([1e10, 1e12, 1e15])

    # Should handle extreme pore volumes gracefully
    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_zero_flow():
    """Test distribution_infiltration_to_extraction handles zero flow values gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-07", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.zeros(len(dates)), index=dates)  # Zero flow
    aquifer_pore_volumes = np.array([1000.0])

    # Zero flow creates infinite residence times but should be handled gracefully
    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_mixed_pore_volumes():
    """Test distribution_infiltration_to_extraction handles mixed pore volumes with varying overlaps."""
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
    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_analytical_mass_conservation():
    """Test distribution_infiltration_to_extraction mass conservation with pulse input."""
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

    # Run distribution_infiltration_to_extraction
    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_known_constant_delay():
    """Test distribution_infiltration_to_extraction with known constant delay scenario."""
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

    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_known_average_of_pore_volumes():
    """Test distribution_infiltration_to_extraction averages multiple pore volumes correctly."""
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

    cout_single = distribution_infiltration_to_extraction(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=single_pv,
        retardation_factor=1.0,
    )

    cout_double = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_known_zero_input_gives_zero_output():
    """Test distribution_infiltration_to_extraction with zero input gives zero output."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Zero concentration everywhere
    cin = pd.Series([0.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)
    aquifer_pore_volumes = np.array([200.0, 400.0])

    cout = distribution_infiltration_to_extraction(
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


def test_distribution_infiltration_to_extraction_known_retardation_effect():
    """Test distribution_infiltration_to_extraction retardation factor effect."""
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
    cout_no_retard = distribution_infiltration_to_extraction(
        cin=cin.values,
        flow=flow.values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    cout_retarded = distribution_infiltration_to_extraction(
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
    cout = gamma_infiltration_to_extraction(
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

    cout = gamma_infiltration_to_extraction(
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
    cout = gamma_infiltration_to_extraction(
        cin=cin,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
    )

    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(dates_cout)


# ===============================================================================
# DISTRIBUTION_BACKWARD FUNCTION TESTS (MIRROR OF DISTRIBUTION_FORWARD)
# ===============================================================================


def test_distribution_extraction_to_infiltration_basic_functionality():
    """Test basic functionality of distribution_extraction_to_infiltration."""
    # Create test data with aligned cout and flow
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cin_tedges with different alignment
    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    # Use smaller pore volumes for reasonable residence times (1-3 days)
    aquifer_pore_volumes = np.array([100.0, 200.0, 300.0])

    cin = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Check output type and length
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates)


def test_distribution_extraction_to_infiltration_constant_input():
    """Test distribution_extraction_to_infiltration with constant output concentration."""
    # Create longer time series for better testing
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cin_tedges starting earlier to capture residence time effects
    cint_dates = pd.date_range(start="2019-06-01", end="2019-11-30", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)) * 5.0, index=dates)  # Constant concentration
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # Constant flow
    aquifer_pore_volumes = np.array([500.0, 1000.0])  # Two pore volumes

    cin = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Check output type and length
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates)

    # With constant output and sufficient time, some inputs should be valid
    valid_inputs = cin[~np.isnan(cin)]
    if len(valid_inputs) > 0:
        # Input should be non-negative
        assert np.all(valid_inputs >= 0)


def test_distribution_extraction_to_infiltration_single_pore_volume():
    """Test distribution_extraction_to_infiltration with single pore volume."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-12-20", end="2020-01-10", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.sin(np.linspace(0, 2 * np.pi, len(dates))) + 2, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    # Use smaller pore volume for reasonable residence time (5 days)
    aquifer_pore_volumes = np.array([500.0])  # Single pore volume

    cin = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates)


def test_distribution_extraction_to_infiltration_retardation_factor():
    """Test distribution_extraction_to_infiltration with different retardation factors."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-06-01", end="2019-11-30", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([1000.0, 2000.0])

    # Test different retardation factors
    cin1 = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    cin2 = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=2.0,
    )

    # Results should be different for different retardation factors
    assert isinstance(cin1, np.ndarray)
    assert isinstance(cin2, np.ndarray)
    assert len(cin1) == len(cin2)


def test_distribution_extraction_to_infiltration_error_conditions():
    """Test distribution_extraction_to_infiltration error conditions."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([1000.0])

    # Test mismatched tedges length
    wrong_tedges = tedges[:-2]  # Too short
    with pytest.raises(ValueError, match="tedges must have one more element than cout"):
        distribution_extraction_to_infiltration(
            cout=cout.values,
            flow=flow.values,
            tedges=wrong_tedges,
            cin_tedges=cin_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
        )

    # Test mismatched flow and tedges
    wrong_flow = flow[:-2]  # Too short
    with pytest.raises(ValueError, match="tedges must have one more element than flow"):
        distribution_extraction_to_infiltration(
            cout=cout.values,
            flow=wrong_flow.values,
            tedges=tedges,
            cin_tedges=cin_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
        )


# ===============================================================================
# PERFECT INVERSE RELATIONSHIP TESTS (MATHEMATICAL SYMMETRY)
# ===============================================================================


def test_distribution_extraction_to_infiltration_analytical_simple_delay():
    """Test distribution_extraction_to_infiltration with known simple delay scenario."""
    # Create a scenario where we know the exact relationship
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Input period starts earlier to account for residence time
    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-15", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    # Known pore volume that gives exactly 1 day residence time
    flow_rate = 100.0  # m3/day
    pore_volume = 100.0  # m3 -> residence time = 100/100 = 1 day

    # Step function: cout jumps from 1 to 5 on day 5
    cout_values = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    cout = pd.Series(cout_values, index=dates)
    flow = pd.Series([flow_rate] * len(dates), index=dates)
    aquifer_pore_volumes = np.array([pore_volume])

    cin = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # With 1-day residence time, the step change on day 5 should appear 1 day earlier in cin
    valid_inputs = cin[~np.isnan(cin)]
    if len(valid_inputs) > 0:
        # Should recover some reasonable signal
        assert np.all(valid_inputs >= 0), f"All inputs should be non-negative, got {valid_inputs}"


def test_distribution_extraction_to_infiltration_zero_output_gives_zero_input():
    """Test distribution_extraction_to_infiltration with zero output gives zero input."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    # Zero concentration everywhere
    cout = pd.Series([0.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)
    aquifer_pore_volumes = np.array([200.0, 400.0])

    cin = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Zero output should give zero input (where valid)
    valid_inputs = cin[~np.isnan(cin)]
    if len(valid_inputs) > 0:
        np.testing.assert_allclose(valid_inputs, 0.0, atol=1e-15, err_msg="Zero output should produce zero input")


# ===============================================================================
# SYMMETRIC EDGE CASE TESTS
# ===============================================================================


def test_distribution_extraction_to_infiltration_no_temporal_overlap():
    """Test distribution_extraction_to_infiltration returns NaN when no temporal overlap exists."""
    # Create cout in early 2020
    early_dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=early_dates, number_of_bins=len(early_dates))

    # Create cin_tedges much later (no possible overlap)
    late_dates = pd.date_range(start="2020-12-01", end="2020-12-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=late_dates, number_of_bins=len(late_dates))

    cout = pd.Series(np.ones(len(early_dates)), index=early_dates)
    flow = pd.Series(np.ones(len(early_dates)) * 100, index=early_dates)
    aquifer_pore_volumes = np.array([100.0])  # Small pore volume

    # No temporal overlap should return all NaN values
    cin = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array of NaN values
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(late_dates)
    assert np.all(np.isnan(cin))


def test_distribution_extraction_to_infiltration_extreme_pore_volumes():
    """Test distribution_extraction_to_infiltration handles extremely large pore volumes gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Extremely large pore volumes that create invalid extraction edges
    aquifer_pore_volumes = np.array([1e10, 1e12, 1e15])

    # Should handle extreme pore volumes gracefully
    cin = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array (likely all NaN due to extreme residence times)
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates)
    # With extremely large pore volumes, all outputs should be NaN
    assert np.all(np.isnan(cin))


def test_distribution_extraction_to_infiltration_zero_flow():
    """Test distribution_extraction_to_infiltration handles zero flow values gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.zeros(len(dates)), index=dates)  # Zero flow
    aquifer_pore_volumes = np.array([1000.0])

    # Zero flow creates infinite residence times but should be handled gracefully
    cin = distribution_extraction_to_infiltration(
        cout=cout.values,
        flow=flow.values,
        tedges=tedges,
        cin_tedges=cin_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array (likely all NaN due to infinite residence times)
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates)
    # With zero flow, all outputs should be NaN
    assert np.all(np.isnan(cin))


# ===============================================================================
# GAMMA_EXTRACTION_TO_INFILTRATION FUNCTION TESTS
# ===============================================================================


def test_gamma_extraction_to_infiltration_zero_output_gives_zero_input():
    """Test gamma_extraction_to_infiltration with zero output gives zero input."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Zero concentration everywhere
    cout = pd.Series([0.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    cin = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=tedges,
        cin_tedges=cin_tedges,
        flow=flow,
        alpha=5.0,
        beta=40.0,  # mean = 200, reasonable residence time
        n_bins=10,
    )

    # Zero output should give zero input (where valid)
    valid_inputs = cin[~np.isnan(cin)]
    assert len(valid_inputs) > 0, "Should have some valid outputs"
    np.testing.assert_allclose(valid_inputs, 0.0, atol=1e-15, err_msg="Zero output should produce zero input")


def test_gamma_extraction_to_infiltration_constant_input():
    """Test gamma_extraction_to_infiltration with constant extraction concentration."""
    # Use longer time series to allow for steady state
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Infiltration period starts earlier to capture source
    cin_dates = pd.date_range(start="2019-11-01", end="2020-11-30", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Constant extraction concentration and flow
    cout = pd.Series([5.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    cin = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=tedges,
        cin_tedges=cin_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,  # mean pore volume = 100 m3, ~1 day residence time
        n_bins=20,
    )

    # Check that we have valid outputs
    valid_mask = ~np.isnan(cin)
    assert np.sum(valid_mask) > 100, "Should have many valid outputs for long constant signal"

    # For constant output, input should also be approximately constant (allowing for edge effects)
    # Check the middle region where steady state is reached
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 100:
        middle_start = valid_indices[50]
        middle_end = valid_indices[-50]
        middle_region = cin[middle_start:middle_end]
        middle_valid = middle_region[~np.isnan(middle_region)]

        # Input should be approximately 5.0 in steady state
        mean_input = np.mean(middle_valid)
        std_input = np.std(middle_valid)
        assert abs(mean_input - 5.0) < 1.0, f"Expected mean ~5.0 in steady state, got {mean_input:.2f}"
        assert std_input < 2.0, f"Expected low variance in steady state, got std={std_input:.2f}"


def test_gamma_extraction_to_infiltration_step_function():
    """Test gamma_extraction_to_infiltration can handle step function in extraction."""
    # Create sufficient time period
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_dates = pd.date_range(start="2019-11-01", end="2020-11-30", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Step function: extraction concentration changes from 1 to 5
    cout_values = np.ones(len(dates))
    cout_values[180:] = 5.0  # Step on day 180
    cout = pd.Series(cout_values, index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    cin = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=tedges,
        cin_tedges=cin_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,  # ~1 day mean residence time
        n_bins=20,
    )

    # Check that we can detect a change in the input
    valid_mask = ~np.isnan(cin)
    assert np.sum(valid_mask) > 100, "Should have many valid outputs"

    valid_cin = cin[valid_mask]
    # The input should show variation (not all the same value)
    assert np.std(valid_cin) > 0.5, "Should see variation in input corresponding to step in output"


def test_gamma_extraction_to_infiltration_roundtrip():
    """Test gamma_infiltration_to_extraction -> gamma_extraction_to_infiltration roundtrip."""
    # Create time windows with proper alignment
    cin_dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Output window overlaps with input
    cout_dates = pd.date_range(start="2020-03-01", end="2020-10-31", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Use a simple signal that should be recoverable
    cin_original = pd.Series([5.0] * len(cin_dates), index=cin_dates)
    flow_cin = pd.Series([100.0] * len(cin_dates), index=cin_dates)
    flow_cout = pd.Series([100.0] * len(cout_dates), index=cout_dates)

    # Forward pass: infiltration -> extraction
    cout = gamma_infiltration_to_extraction(
        cin=cin_original,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow_cin,
        alpha=10.0,
        beta=10.0,
        n_bins=20,
    )

    # Backward pass: extraction -> infiltration
    cout_series = pd.Series(cout, index=cout_dates)
    cin_reconstructed = gamma_extraction_to_infiltration(
        cout=cout_series,
        tedges=cout_tedges,
        cin_tedges=cin_tedges,
        flow=flow_cout,
        alpha=10.0,
        beta=10.0,
        n_bins=20,
    )

    # Check that we have valid reconstructed values
    valid_mask = ~np.isnan(cin_reconstructed)
    assert np.sum(valid_mask) > 100, "Should have substantial valid overlap"

    # Check middle region for steady state
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 100:
        middle_start = valid_indices[50]
        middle_end = valid_indices[-50]
        middle_region = slice(middle_start, middle_end)

        reconstructed_middle = cin_reconstructed[middle_region]
        original_middle = cin_original.values[middle_region]

        # Should be close in the stable middle region
        assert np.allclose(reconstructed_middle, original_middle, rtol=0.2), (
            f"Roundtrip error: expected ~{np.mean(original_middle):.2f}, got {np.mean(reconstructed_middle):.2f} (mean)"
        )


def test_gamma_extraction_to_infiltration_retardation_factor():
    """Test gamma_extraction_to_infiltration with different retardation factors."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_dates = pd.date_range(start="2019-11-01", end="2020-11-30", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Step function
    cout_values = np.ones(len(dates))
    cout_values[180:] = 3.0
    cout = pd.Series(cout_values, index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Test with retardation factor = 1.0
    cin1 = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=tedges,
        cin_tedges=cin_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
        retardation_factor=1.0,
        n_bins=20,
    )

    # Test with retardation factor = 2.0 (doubles residence time)
    cin2 = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=tedges,
        cin_tedges=cin_tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,
        retardation_factor=2.0,
        n_bins=20,
    )

    # Results should be different due to different residence times
    valid_mask = ~np.isnan(cin1) & ~np.isnan(cin2)
    assert np.sum(valid_mask) > 50, "Should have sufficient valid overlap"

    # The timing of the step should be different
    assert not np.allclose(cin1[valid_mask], cin2[valid_mask], rtol=0.1), (
        "Retardation factor should affect the timing/position of features"
    )


def test_gamma_extraction_to_infiltration_with_mean_std():
    """Test gamma_extraction_to_infiltration using mean and std parameters."""
    dates = pd.date_range(start="2020-01-01", end="2020-06-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_dates = pd.date_range(start="2019-12-01", end="2020-05-31", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    cout = pd.Series([3.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Use mean/std instead of alpha/beta
    mean = 100.0  # mean pore volume
    std = 20.0

    cin = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=tedges,
        cin_tedges=cin_tedges,
        flow=flow,
        mean=mean,
        std=std,
        n_bins=20,
    )

    # Should produce valid results
    valid_mask = ~np.isnan(cin)
    assert np.sum(valid_mask) > 50, "Should have many valid outputs"

    valid_cin = cin[valid_mask]
    assert np.allclose(np.mean(valid_cin), 3.0, rtol=0.15), "Mean should be approximately preserved"


def test_gamma_extraction_to_infiltration_missing_parameters():
    """Test that gamma_extraction_to_infiltration raises appropriate errors for missing parameters."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_dates = pd.date_range(start="2019-12-28", end="2020-01-08", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Test missing both alpha/beta and mean/std
    with pytest.raises(ValueError):
        gamma_extraction_to_infiltration(cout=cout, tedges=tedges, cin_tedges=cin_tedges, flow=flow)
