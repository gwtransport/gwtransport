import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest
from _oracles import partial_isin  # ty: ignore[unresolved-import]  # tests/src on path via conftest

from gwtransport import advection as adv_mod
from gwtransport import advection_utils, gamma
from gwtransport._time import tedges_to_days
from gwtransport.advection import (
    _validate_advection_inputs,
    extraction_to_infiltration,
    gamma_extraction_to_infiltration,
    gamma_infiltration_to_extraction,
    infiltration_to_extraction,
    infiltration_to_extraction_nonlinear_sorption,
)
from gwtransport.advection_utils import (
    _densify_weights,
    _resolve_spinup_inputs,
)
from gwtransport.advection_utils import (
    _infiltration_to_extraction_weights as _banded_weights,
)
from gwtransport.advection_utils import (
    _resolve_spinup_mask as _banded_mask,
)
from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption, LangmuirSorption
from gwtransport.fronttracking.output import compute_domain_mass
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.utils import (
    compute_time_edges,
    cumulative_flow_volume,
    linear_interpolate,
    solve_inverse_transport,
    solve_inverse_transport_banded,
)


# The package builder/mask are banded (see advection_utils). These thin adapters present
# the historical dense (n_cout, n_cin) signatures so the dense-oracle and Fraction-truth
# comparisons below stay unchanged: the build is densified, and the row-wise mask is applied
# to the dense matrix (its col_start passthrough is irrelevant when the input is full-width).
def _infiltration_to_extraction_weights(**kwargs):
    band_vals, col_start, contributing_bins, zero_flow_cout = _banded_weights(**kwargs)
    dense = _densify_weights(band_vals, col_start, len(kwargs["tedges"]) - 1)
    return dense, contributing_bins, zero_flow_cout


def _resolve_spinup_mask(*, accumulated_weights, contributing_bins, zero_flow_cout, n_pv, spinup):
    weights, _, invalid = _banded_mask(
        band_vals=accumulated_weights,
        col_start=np.zeros(len(accumulated_weights), dtype=np.intp),
        contributing_bins=contributing_bins,
        zero_flow_cout=zero_flow_cout,
        n_pv=n_pv,
        spinup=spinup,
    )
    return weights, invalid


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
    """Retardation shifts the gamma step arrival by exactly (R-1)*mean_pv/flow days.

    With a narrow gamma (std << mean) the APVD collapses toward a single pore
    volume, so the mean residence time ``R*mean_pv/flow`` dominates and the step
    in cout lands at output index ``offset + R*mean_pv/flow``. Comparing R=1 and
    R=2 gives a deterministic shift of ``mean_pv/flow`` bins. This replaces a
    loose diff/std comparison that passed even under wrong-direction retardation.
    """
    # Create test data - use year-long series for robust results
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-03-01", end="2020-10-31", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Step function: jump from 0 to 1 on cin day 90 (index 90).
    cin_values = np.zeros(len(dates))
    cin_values[90:] = 1.0
    cin = pd.Series(cin_values, index=dates)
    flow_rate = 100.0
    flow = pd.Series(np.full(len(dates), flow_rate), index=dates)

    # Narrow gamma: mean_pv = 300 m³ -> mean RT = 3 days; std = 10 m³ -> 0.1 day.
    mean_pv = 300.0
    std_pv = 10.0

    cout_r1 = gamma_infiltration_to_extraction(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        mean=mean_pv,
        std=std_pv,
        retardation_factor=1.0,
        n_bins=50,
    )
    cout_r2 = gamma_infiltration_to_extraction(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        mean=mean_pv,
        std=std_pv,
        retardation_factor=2.0,
        n_bins=50,
    )

    # Find the step midpoint (first bin where cout exceeds 0.5) on each path.
    valid = ~np.isnan(cout_r1) & ~np.isnan(cout_r2)
    idx_r1 = np.argmax((cout_r1 > 0.5) & valid)
    idx_r2 = np.argmax((cout_r2 > 0.5) & valid)

    # The R=2 step lags R=1 by exactly (R-1)*mean_pv/flow = 3 bins.
    expected_delay = round((2.0 - 1.0) * mean_pv / flow_rate)
    assert idx_r2 - idx_r1 == expected_delay, (
        f"Expected R=2 step to lag R=1 by {expected_delay} bins, got {idx_r2 - idx_r1}"
    )


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
        n_bins=20,
    )

    # Explicit validation
    assert len(cout) == len(cout_dates), f"Expected {len(cout_dates)} output bins, got {len(cout)}"
    valid_count = np.sum(~np.isnan(cout))
    assert valid_count >= 150, f"Expected at least 150 valid bins for 6-month extraction, got {valid_count}"

    # For constant input and constant flow, the post-spin-up steady state is exact.
    # Skip the spin-up region (first valid bins where some pore-volume paths have not
    # yet contributed) and assert exact equality on the steady-state interior.
    valid_indices = np.where(~np.isnan(cout))[0]
    # Drop the first 90 valid bins as spin-up margin (mean RT ~1 day; with n_bins=20
    # the longest gamma bin's RT is well under 90 days).
    steady_indices = valid_indices[90:]
    np.testing.assert_allclose(
        cout[steady_indices],
        1.0,
        rtol=1e-12,
        err_msg="Constant input with constant flow should reproduce input exactly in steady state",
    )


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
    """Step crossing lags by the analytical gamma mean residence time.

    For a symmetric-enough gamma APVD the cout 50%-crossing of a unit step lags
    the cin step by the mean residence time ``mean_pv/flow``. With mean_pv = 500
    and flow = 100 the expected lag is 5 days; a wider gamma broadens the
    breakthrough but leaves the 50%-crossing at the mean. This is a genuine
    residence-time assertion, unlike the previous constant-input steady-state
    check (covered by test_gamma_infiltration_to_extraction_constant_input).
    """
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-15", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Unit step in cin on day 100 (index 100).
    step_idx = 100
    cin_values = np.zeros(len(dates))
    cin_values[step_idx:] = 1.0
    cin = pd.Series(cin_values, index=dates)
    flow_rate = 100.0
    flow = pd.Series([flow_rate] * len(dates), index=dates)

    # Mean residence time = mean_pv / flow = 500 / 100 = 5 days.
    mean_pv = 500.0
    std_pv = 100.0
    cout = gamma_infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        mean=mean_pv,
        std=std_pv,
        n_bins=50,
        retardation_factor=1.0,
    )

    # cout index of the step day, accounting for the cout window offset.
    offset_days = (cout_tedges[0] - tedges[0]) / pd.Timedelta(days=1)
    cin_step_in_cout = step_idx - round(offset_days)

    # The 50%-crossing of cout must lag the cin step by the mean RT (5 days).
    valid = ~np.isnan(cout)
    crossing = np.argmax((cout > 0.5) & valid)
    measured_lag = crossing - cin_step_in_cout
    expected_lag = round(mean_pv / flow_rate)
    assert abs(measured_lag - expected_lag) <= 1, (
        f"Expected step 50%-crossing to lag by ~{expected_lag} days, got {measured_lag}"
    )


# ===============================================================================
# DISTRIBUTION_INFILTRATION_TO_EXTRACTION FUNCTION TESTS
# ===============================================================================


def test_infiltration_to_extraction_basic_functionality():
    """Test basic functionality of infiltration_to_extraction."""
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

    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Check output type and length
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)


def test_infiltration_to_extraction_constant_input():
    """Test infiltration_to_extraction with constant input concentration."""
    # Create longer time series for better testing
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cout_tedges starting later
    cout_dates = pd.date_range(start="2020-06-01", end="2020-11-30", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)) * 5.0, index=dates)  # Constant concentration
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # Constant flow
    aquifer_pore_volumes = np.array([500.0, 1000.0])  # Two pore volumes

    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Explicit validation
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates), f"Expected {len(cout_dates)} output bins, got {len(cout)}"

    valid_count = np.sum(~np.isnan(cout))
    assert valid_count >= 150, f"Expected at least 150 valid bins for 6-month extraction, got {valid_count}"

    # For constant input and constant flow, the post-spin-up steady state is exact.
    # Skip the first valid bins (spin-up margin: longest RT = 1000/100 = 10 days) and
    # assert exact equality to the input on the steady-state interior.
    valid_indices = np.where(~np.isnan(cout))[0]
    steady_indices = valid_indices[30:]
    np.testing.assert_allclose(
        cout[steady_indices],
        5.0,
        rtol=1e-12,
        err_msg="Constant input with constant flow should reproduce input exactly in steady state",
    )


def test_infiltration_to_extraction_single_pore_volume():
    """Test infiltration_to_extraction with single pore volume."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-10", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.sin(np.linspace(0, 2 * np.pi, len(dates))) + 2, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    # Use smaller pore volume for reasonable residence time (5 days)
    aquifer_pore_volumes = np.array([500.0])  # Single pore volume

    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)


def test_infiltration_to_extraction_retardation_factor():
    """Retardation factor delays the step arrival in cout by (R-1)*PV/flow bins.

    With a single pore volume and a step input, residence time = PV*R/flow. The
    output reaches the post-step value at cout index ``offset + R*PV/flow``,
    where ``offset = cout_tedges[0] - tedges[0]`` in days. Comparing R=1 and
    R=2 gives a deterministic shift of ``PV/flow`` bins. This replaces a
    structurally-only test that passed even when retardation was ignored.
    """
    dates = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin_values = np.zeros(len(dates))
    cin_values[9:] = 10.0  # step on cin day 10 (cin index 9)
    flow = np.full(len(dates), 100.0)
    pore_volume = 200.0  # residence time R*PV/flow = 2 d (R=1) vs 4 d (R=2)
    aquifer_pore_volumes = np.array([pore_volume])

    cout_r1 = infiltration_to_extraction(
        cin=cin_values,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )
    cout_r2 = infiltration_to_extraction(
        cin=cin_values,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=2.0,
    )

    # cout starts on cin day 5; first fully-stepped cout index satisfies 5 + i - RT >= 10.
    expected_step_idx_r1 = 5 + 2  # 7
    expected_step_idx_r2 = 5 + 4  # 9
    np.testing.assert_allclose(cout_r1[expected_step_idx_r1:], 10.0, rtol=1e-12)
    np.testing.assert_allclose(cout_r2[expected_step_idx_r2:], 10.0, rtol=1e-12)
    # Pre-step bins are zero (single PV → row weight on a zero cin bin)
    assert cout_r1[expected_step_idx_r1 - 3] == 0.0
    assert cout_r2[expected_step_idx_r2 - 3] == 0.0
    # The R=2 step lags by exactly PV/flow = 2 bins
    assert expected_step_idx_r2 - expected_step_idx_r1 == int(pore_volume / 100.0)


# ===============================================================================
# DISTRIBUTION_FORWARD FUNCTION - EDGE CASE TESTS
# ===============================================================================


def test_infiltration_to_extraction_no_temporal_overlap():
    """Test infiltration_to_extraction returns NaN when no temporal overlap exists."""
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
    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array of NaN values
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(late_dates)
    assert np.all(np.isnan(cout))


def test_infiltration_to_extraction_zero_concentrations():
    """Test infiltration_to_extraction preserves zero concentrations with aligned residence time."""
    # Create longer time series for realistic residence times
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # cout_tedges later to allow residence time effects
    cout_dates = pd.date_range(start="2020-01-10", end="2020-12-20", freq="D")  # Overlap with input period
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Create cin with repeating pattern: [1.0, 0.0, 2.0] (3-day period)
    cin_pattern = np.array([1.0, 0.0, 2.0])
    cin_values = np.tile(cin_pattern, len(dates) // len(cin_pattern) + 1)[: len(dates)]
    cin = pd.Series(cin_values, index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Use pore volume that gives residence time matching pattern period (3 days)
    # This ensures zeros align and are preserved in output
    aquifer_pore_volumes = np.array([300.0])  # 300/100 = 3 day residence time

    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Explicit validation
    valid_count = np.sum(~np.isnan(cout))
    assert valid_count == len(cout_dates), f"Expected {len(cout_dates)} valid bins, got {valid_count}"

    # Check that zero concentrations are preserved (not converted to NaN)
    valid_results = cout[~np.isnan(cout)]
    has_zeros = np.sum(valid_results == 0.0)
    assert has_zeros >= 80, f"Expected at least 80 zero values preserved from input pattern, got {has_zeros}"

    # Check range of concentration values matches input pattern
    assert np.all(valid_results >= 0.0), "All concentrations should be non-negative"
    assert np.all(valid_results <= 2.0), "All concentrations should be within input range [0, 2]"

    # Verify output pattern matches input pattern (exact preservation with aligned residence time)
    unique_values = np.unique(valid_results)
    expected_values = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        unique_values, expected_values, rtol=1e-10, err_msg="Output should preserve exact input pattern values"
    )


def test_infiltration_to_extraction_extreme_conditions():
    """Test infiltration_to_extraction handles extreme conditions gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-05", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_tedges = tedges.copy()

    cin = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    flow = pd.Series([1000.0, 0.1, 1000.0, 0.1, 1000.0], index=dates)
    aquifer_pore_volumes = np.array([10.0, 100000.0, 50.0])

    # Should handle extreme conditions gracefully
    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return valid array (may contain NaN values)
    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(dates)


def test_infiltration_to_extraction_extreme_pore_volumes():
    """Test infiltration_to_extraction handles extremely large pore volumes gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-07", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)

    # Extremely large pore volumes that create invalid infiltration edges
    aquifer_pore_volumes = np.array([1e10, 1e12, 1e15])

    # Should handle extreme pore volumes gracefully
    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
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


def test_infiltration_to_extraction_zero_flow():
    """Test infiltration_to_extraction handles zero flow values gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-07", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.zeros(len(dates)), index=dates)  # Zero flow
    aquifer_pore_volumes = np.array([1000.0])

    # Zero flow creates infinite residence times but should be handled gracefully
    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
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


def test_infiltration_to_extraction_mixed_pore_volumes_all_nan_when_any_unresolved():
    """Mixed pore volumes with one residence time exceeding cin window → all cout bins NaN.

    Strict-validity semantics: a cout bin is valid only when *every* streamtube has
    a fully resolved source window inside the cin range. With pore volumes spanning
    1 day and 1000+ days residence times against a 365-day cin window, the longest
    streamtube never breaks through, so the bundle output is NaN everywhere.
    Returning a value (averaged over the contributing subset only) would
    over-attribute mass to the cin bins seen by the short-PV streamtubes.
    """
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-10", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([10.0, 100.0, 50000.0, 100000.0])

    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
        spinup=None,  # strict-validity: this is the property under test
    )

    assert isinstance(cout, np.ndarray)
    assert len(cout) == len(cout_dates)
    assert np.all(np.isnan(cout)), "Expected all NaN: longest pore volume's streamtube has not broken through"


def test_infiltration_to_extraction_mixed_pore_volumes_valid_post_breakthrough():
    """Mixed pore volumes produce valid output once the longest streamtube has broken through."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # cout window starts well past max residence time (300 m3 / 100 m3/day = 3 days)
    cout_dates = pd.date_range(start="2020-01-15", end="2020-12-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(dates)) * 100, index=dates)
    aquifer_pore_volumes = np.array([10.0, 100.0, 200.0, 300.0])

    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    assert not np.any(np.isnan(cout)), "All cout bins should be valid (cout window past max residence time)"
    np.testing.assert_allclose(cout, 1.0, atol=1e-13, err_msg="Constant cin must yield constant cout")


# ===============================================================================
# DISTRIBUTION_FORWARD FUNCTION - ANALYTICAL SOLUTION TESTS
# ===============================================================================


def test_infiltration_to_extraction_analytical_mass_conservation():
    """Test infiltration_to_extraction mass conservation with pulse input."""
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

    # Run infiltration_to_extraction
    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Mass conservation check
    # Input mass = concentration * flow * time (for each time step)
    dt = 1.0  # 1 day time steps
    input_mass = np.sum(cin_values * flow.to_numpy() * dt)

    # Output mass = concentration * flow * time, using the actual output-period flow per bin.
    # cin/flow/cout are aligned to daily bins of equal duration; cout_tedges starts on
    # 2020-01-05 (index 4 of cin/flow), so the matching flow slice is flow[4:4+len(cout)].
    cout_flow = flow.to_numpy()[4 : 4 + len(cout)]
    valid_mask = ~np.isnan(cout)
    output_mass = np.sum(cout[valid_mask] * cout_flow[valid_mask] * dt)

    # Mass is conserved to machine precision: per-streamtube row-normalization
    # gives the exact mass-flux/water-flux ratio per streamtube, and the simple
    # arithmetic average over equal-flow streamtubes preserves total mass when
    # the output window captures the entire pulse from every pore-volume path.
    assert input_mass > 0
    mass_error = abs(output_mass - input_mass) / input_mass
    assert mass_error < 1e-12, f"Mass conservation error {mass_error:.2e} >= 1e-12"


def test_infiltration_to_extraction_known_constant_delay():
    """Test infiltration_to_extraction with known constant delay scenario."""
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

    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # With 1-day residence time, the step change on day 5 appears on day 6.
    # Output days 6-10 correspond entirely to post-step infiltration days 5-9,
    # so every valid output bin must equal the post-step value 5.0 exactly.
    valid_count = np.sum(~np.isnan(cout))
    assert valid_count >= 4, f"Expected at least 4 valid bins for 1-day delay, got {valid_count}"

    valid_outputs = cout[~np.isnan(cout)]
    np.testing.assert_allclose(
        valid_outputs,
        5.0,
        rtol=1e-12,
        err_msg="Post-step output window should equal the post-step input exactly",
    )


def test_infiltration_to_extraction_known_average_of_pore_volumes():
    """Test infiltration_to_extraction averages multiple pore volumes correctly."""
    # Simple scenario where we can predict the averaging behavior
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Output period in the middle to ensure overlap
    cout_dates = pd.date_range(start="2020-01-10", end="2020-01-15", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Non-constant input so that averaging two identical pore volumes is a
    # non-trivial equality (a constant input would make this hold trivially).
    cin = pd.Series(np.sin(np.linspace(0, 2 * np.pi, len(dates))) + 2.0, index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)

    # Two identical pore volumes - average should equal the single pore volume result
    single_pv = np.array([500.0])
    double_pv = np.array([500.0, 500.0])

    cout_single = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=single_pv,
        retardation_factor=1.0,
    )

    cout_double = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=double_pv,
        retardation_factor=1.0,
    )

    # Results must be identical (averaging two identical contributions).
    valid_mask = ~np.isnan(cout_single) & ~np.isnan(cout_double)
    assert np.any(valid_mask), "Expected at least one valid overlapping output bin"
    # Verify the input is genuinely non-constant so the equality is non-trivial.
    assert np.ptp(cout_single[valid_mask]) > 0.5, "Output should vary (non-trivial test)"
    np.testing.assert_allclose(
        cout_single[valid_mask],
        cout_double[valid_mask],
        rtol=1e-10,
        err_msg="Averaging identical pore volumes should give same result as single pore volume",
    )


def test_infiltration_to_extraction_known_zero_input_gives_zero_output():
    """Test infiltration_to_extraction with zero input gives zero output."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Zero concentration everywhere
    cin = pd.Series([0.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(dates), index=dates)
    aquifer_pore_volumes = np.array([200.0, 400.0])

    cout = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Explicit validation - zero input should give zero output
    valid_count = np.sum(~np.isnan(cout))
    assert valid_count == len(cout_dates), f"Expected {len(cout_dates)} valid bins, got {valid_count}"

    valid_outputs = cout[~np.isnan(cout)]
    np.testing.assert_allclose(valid_outputs, 0.0, atol=1e-15, err_msg="Zero input should produce zero output")


def test_infiltration_to_extraction_known_retardation_effect():
    """Test infiltration_to_extraction retardation factor effect."""
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
    cout_no_retard = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    cout_retarded = infiltration_to_extraction(
        cin=cin.to_numpy(),
        flow=flow.to_numpy(),
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=2.0,
    )

    # Basic structural checks
    assert isinstance(cout_no_retard, np.ndarray)
    assert isinstance(cout_retarded, np.ndarray)
    assert len(cout_no_retard) == len(cout_dates)
    assert len(cout_retarded) == len(cout_dates)

    # Physics check: with PV=200 m3 and flow=100 m3/day, R=1 -> RT=2 d, R=2 -> RT=4 d.
    # cin steps from 0 to 10 on cin day 10 (cin index 9). cout(day k) = cin(day k - RT),
    # so cout reaches the step value when cout_day - RT >= 10.
    # cout starts on cin day 5, so the first fully-retarded cout index satisfies
    # 5 + i - RT >= 10  =>  i >= 5 + RT.
    expected_step_idx_r1 = 5 + 2  # = 7
    expected_step_idx_r2 = 5 + 4  # = 9
    np.testing.assert_allclose(cout_no_retard[expected_step_idx_r1:], 10.0, rtol=1e-12)
    np.testing.assert_allclose(cout_no_retard[: expected_step_idx_r1 - 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(cout_retarded[expected_step_idx_r2:], 10.0, rtol=1e-12)
    np.testing.assert_allclose(cout_retarded[: expected_step_idx_r2 - 1], 0.0, atol=1e-12)
    # Retardation by factor 2 doubles the residence time, shifting the step by exactly
    # (R-1)*PV/flow = 2 days.
    assert expected_step_idx_r2 - expected_step_idx_r1 == 2


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


def test_gamma_infiltration_to_extraction_pulse_mass_conservation():
    """Gamma forward transport conserves pulse mass under constant flow.

    A finite cin pulse carries total mass ``Σ cin·flow·dt``. When the cout
    window captures the entire pulse from every gamma pore-volume path
    (strict ``spinup=None``), the flow-weighted output mass must equal the
    input mass to machine precision -- a genuine property of the gamma path
    not covered by the constant-input steady-state tests.
    """
    dates = pd.date_range(start="2020-01-01", end="2020-03-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # cin record starts ~20 days before the cout window so the pulse and the full
    # gamma residence-time spread lie inside the cin range for every cout bin.
    cout_dates = pd.date_range(start="2020-01-25", end="2020-03-31", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Finite pulse: nonzero for 5 days, breaking through inside the cout window.
    cin_values = np.zeros(len(dates))
    cin_values[30:35] = 8.0
    cin = pd.Series(cin_values, index=dates)
    flow_rate = 100.0
    flow = pd.Series([flow_rate] * len(dates), index=dates)

    cout = gamma_infiltration_to_extraction(
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        mean=200.0,  # mean RT = 2 days
        std=50.0,
        n_bins=10,
        spinup=None,
    )

    # Input mass over the (1-day) cin bins.
    input_mass = np.sum(cin_values * flow_rate * 1.0)
    assert input_mass > 0

    # Output mass: constant flow, so Q_cout = flow_rate; sum over valid 1-day cout bins.
    valid = ~np.isnan(cout)
    output_mass = np.sum(cout[valid] * flow_rate * 1.0)

    mass_error = abs(output_mass - input_mass) / input_mass
    assert mass_error < 1e-12, f"Gamma pulse mass conservation error {mass_error:.2e} >= 1e-12"


def test_empty_series():
    """Test handling of empty series."""
    empty_cin = pd.Series([], dtype=float)

    # This should handle gracefully or raise appropriate error
    with pytest.raises((ValueError, IndexError)):
        # Create tedges - this should fail for empty series
        compute_time_edges(
            tedges=None, tstart=None, tend=pd.DatetimeIndex(empty_cin.index), number_of_bins=len(empty_cin)
        )


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


def test_extraction_to_infiltration_basic_functionality():
    """Test basic functionality of extraction_to_infiltration."""
    # Create test data with aligned cout and flow
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cin_tedges with different alignment
    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(cint_dates)) * 100, index=cint_dates)
    # Use smaller pore volumes for reasonable residence times (1-3 days)
    aquifer_pore_volumes = np.array([100.0, 200.0, 300.0])

    cin = extraction_to_infiltration(
        cout=cout.to_numpy(),
        flow=flow.to_numpy(),
        tedges=cin_tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Check output type and length
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates)


def test_extraction_to_infiltration_constant_input():
    """Test extraction_to_infiltration with constant output concentration."""
    # Create longer time series for better testing
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Create cin_tedges that properly overlaps with required infiltration times
    # With residence times of 5-10 days, we need cin dates ending around Dec 22-27 to catch Jan 1 cout
    cint_dates = pd.date_range(start="2019-12-15", end="2020-12-25", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)) * 5.0, index=dates)  # Constant concentration
    flow = pd.Series(np.ones(len(cint_dates)) * 100, index=cint_dates)  # Constant flow
    aquifer_pore_volumes = np.array([500.0, 1000.0])  # 5 and 10 day residence times

    cin = extraction_to_infiltration(
        cout=cout.to_numpy(),
        flow=flow.to_numpy(),
        tedges=cin_tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Explicit validation
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates), f"Expected {len(cint_dates)} output bins, got {len(cin)}"

    valid_count = np.sum(~np.isnan(cin))
    assert valid_count >= 300, f"Expected at least 300 valid bins with proper overlap, got {valid_count}"

    # Non-negativity over all valid bins.
    valid_inputs = cin[~np.isnan(cin)]
    assert np.all(valid_inputs >= 0), "All inputs should be non-negative"

    # For constant cout and constant flow the deconvolution recovers the exact
    # constant in the data-dominated interior; clip spin-up/boundary bins (which
    # absorb the Tikhonov bias of the ill-posed inverse) and assert exact equality.
    valid_indices = np.where(~np.isnan(cin))[0]
    interior_indices = valid_indices[20:-20]
    np.testing.assert_allclose(
        cin[interior_indices],
        5.0,
        atol=1e-9,
        err_msg="Constant extraction with constant flow should recover the exact constant infiltration",
    )


def test_extraction_to_infiltration_single_pore_volume():
    """Test extraction_to_infiltration with single pore volume."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-12-20", end="2020-01-10", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.sin(np.linspace(0, 2 * np.pi, len(dates))) + 2, index=dates)
    flow = pd.Series(np.ones(len(cint_dates)) * 100, index=cint_dates)
    # Use smaller pore volume for reasonable residence time (5 days)
    aquifer_pore_volumes = np.array([500.0])  # Single pore volume

    cin = extraction_to_infiltration(
        cout=cout.to_numpy(),
        flow=flow.to_numpy(),
        tedges=cin_tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates)


def test_extraction_to_infiltration_retardation_factor():
    """Reverse step arrival shifts back by (R-1)*PV/flow bins under deconvolution.

    Forward: cin step at day k_in produces cout step at day k_in + R*PV/flow.
    Reverse: cout step at day k_out implies cin step at day k_out - R*PV/flow.
    Use a cout window that lies fully past the spin-up region so the forward
    output is NaN-free, then deconvolve and compare to the original cin.
    This replaces a structurally-only test that passed even when retardation
    was ignored.
    """
    n_cin = 80
    cin_dates = pd.date_range(start="2020-01-01", periods=n_cin, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=n_cin)

    # cout window starts at day 10 (past R*PV/flow = 4 d for R=2, PV=200, flow=100).
    n_cout = 60
    cout_dates = pd.date_range(start="2020-01-11", periods=n_cout, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=n_cout)

    flow = np.full(n_cin, 100.0)
    pore_volume = 200.0
    cin_values = np.zeros(n_cin)
    cin_values[30:] = 7.0  # step on cin day 30

    cout_r1 = infiltration_to_extraction(
        cin=cin_values,
        flow=flow,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=1.0,
    )
    cout_r2 = infiltration_to_extraction(
        cin=cin_values,
        flow=flow,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=2.0,
    )
    assert not np.any(np.isnan(cout_r1)), "cout_r1 should be free of NaN in chosen window"
    assert not np.any(np.isnan(cout_r2)), "cout_r2 should be free of NaN in chosen window"

    cin_back_r1 = extraction_to_infiltration(
        cout=cout_r1,
        flow=flow,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=1.0,
    )
    cin_back_r2 = extraction_to_infiltration(
        cout=cout_r2,
        flow=flow,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=2.0,
    )

    # Both reverse passes must recover the cin signal to machine precision in the
    # well-determined interior. Tikhonov bias on edges is below 1e-9 for this
    # well-conditioned single-PV problem.
    valid = ~np.isnan(cin_back_r1) & ~np.isnan(cin_back_r2)
    assert np.sum(valid) > 30
    np.testing.assert_allclose(cin_back_r1[valid], cin_values[valid], atol=1e-9)
    np.testing.assert_allclose(cin_back_r2[valid], cin_values[valid], atol=1e-9)


# ===============================================================================
# PERFECT INVERSE RELATIONSHIP TESTS (MATHEMATICAL SYMMETRY)
# ===============================================================================


def test_extraction_to_infiltration_analytical_simple_delay():
    """Forward-then-reverse round-trip on a step input recovers the step
    to machine precision in the interior.

    Uses a non-integer-day residence time (RT = 1.3 days) so the forward
    operator is well-conditioned (no integer-shift rank deficiency). A
    generous spin-up buffer (30 days on each end) and a buffer around the
    step discontinuity (±10 bins, where Tikhonov smoothing dominates)
    isolates the data-dominated interior where the round-trip recovers cin
    to ~6e-14. ``atol=1e-12`` gives ample margin over that.
    """
    n = 400
    tedges = pd.date_range(start="2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges

    flow_rate = 100.0
    pore_volume = 130.0  # RT = 1.3 days (non-integer to avoid rank deficiency)
    aquifer_pore_volumes = np.array([pore_volume])

    cin_original = np.where(np.arange(n) < n // 2, 1.0, 5.0).astype(float)
    flow = np.full(n, flow_rate)

    cout = infiltration_to_extraction(
        cin=cin_original,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
        spinup="constant",
    )
    cin_recovered = extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
        spinup="constant",
    )

    step_idx = int(np.argmax(np.diff(cin_original) != 0)) + 1
    buffer = 30  # spin-up on either end (≫ residence time)
    step_clip = 10  # bins around the step where Tikhonov bias dominates
    mask = np.ones(n, dtype=bool)
    mask[:buffer] = False
    mask[-buffer:] = False
    mask[step_idx - step_clip : step_idx + step_clip] = False

    np.testing.assert_allclose(cin_recovered[mask], cin_original[mask], atol=1e-12)


def test_extraction_to_infiltration_zero_output_gives_zero_input():
    """Test extraction_to_infiltration with zero output gives zero input."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    # Zero concentration everywhere
    cout = pd.Series([0.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(cint_dates), index=cint_dates)
    aquifer_pore_volumes = np.array([200.0, 400.0])

    cin = extraction_to_infiltration(
        cout=cout.to_numpy(),
        flow=flow.to_numpy(),
        tedges=cin_tedges,
        cout_tedges=tedges,
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


def test_extraction_to_infiltration_no_temporal_overlap():
    """Test extraction_to_infiltration returns NaN when no temporal overlap exists."""
    # Create cout in early 2020
    early_dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=early_dates, number_of_bins=len(early_dates))

    # Create cin_tedges much later (no possible overlap)
    late_dates = pd.date_range(start="2020-12-01", end="2020-12-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=late_dates, number_of_bins=len(late_dates))

    cout = pd.Series(np.ones(len(early_dates)), index=early_dates)
    flow = pd.Series(np.ones(len(late_dates)) * 100, index=late_dates)
    aquifer_pore_volumes = np.array([100.0])  # Small pore volume

    # No temporal overlap should return all NaN values
    cin = extraction_to_infiltration(
        cout=cout.to_numpy(),
        flow=flow.to_numpy(),
        tedges=cin_tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array of NaN values
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(late_dates)
    assert np.all(np.isnan(cin))


def test_extraction_to_infiltration_extreme_pore_volumes():
    """Test extraction_to_infiltration handles extremely large pore volumes gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(cint_dates)) * 100, index=cint_dates)

    # Extremely large pore volumes that create invalid extraction edges
    aquifer_pore_volumes = np.array([1e10, 1e12, 1e15])

    # Should handle extreme pore volumes gracefully
    cin = extraction_to_infiltration(
        cout=cout.to_numpy(),
        flow=flow.to_numpy(),
        tedges=cin_tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )

    # Should return array (likely all NaN due to extreme residence times)
    assert isinstance(cin, np.ndarray)
    assert len(cin) == len(cint_dates)
    # With extremely large pore volumes, all outputs should be NaN
    assert np.all(np.isnan(cin))


def test_extraction_to_infiltration_zero_flow():
    """Test extraction_to_infiltration handles zero flow values gracefully."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cint_dates = pd.date_range(start="2019-12-25", end="2020-01-05", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cint_dates, number_of_bins=len(cint_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.zeros(len(cint_dates)), index=cint_dates)  # Zero flow
    aquifer_pore_volumes = np.array([1000.0])

    # Zero flow creates infinite residence times but should be handled gracefully
    cin = extraction_to_infiltration(
        cout=cout.to_numpy(),
        flow=flow.to_numpy(),
        tedges=cin_tedges,
        cout_tedges=tedges,
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
    # Use sufficient time span: 60 days extraction, 90 days infiltration window
    dates = pd.date_range(start="2020-01-01", end="2020-03-01", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Infiltration window starts earlier to capture source
    cin_dates = pd.date_range(start="2019-12-01", end="2020-02-28", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Zero concentration everywhere
    cout = pd.Series([0.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(cin_dates), index=cin_dates)

    cin = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=tedges,
        flow=flow,
        alpha=5.0,
        beta=40.0,  # mean pore volume = 200 m3, ~2 day residence time with 100 m3/day flow
        n_bins=10,
    )

    # Zero extraction should give zero infiltration (where valid)
    assert len(cin) == len(cin_dates), f"Expected {len(cin_dates)} output bins, got {len(cin)}"
    valid_count = np.sum(~np.isnan(cin))
    assert valid_count >= 50, f"Expected at least 50 valid bins, got {valid_count}"

    valid_inputs = cin[~np.isnan(cin)]
    np.testing.assert_allclose(valid_inputs, 0.0, atol=1e-15, err_msg="Zero extraction must produce zero infiltration")


def test_gamma_extraction_to_infiltration_constant_input():
    """Test constant extraction recovers constant infiltration in fully informed region."""
    # Use long time series: 365 days extraction, extended infiltration window
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Infiltration window extends earlier to capture all source contributions
    cin_dates = pd.date_range(start="2019-11-01", end="2020-11-30", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Constant extraction concentration
    cout = pd.Series([5.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(cin_dates), index=cin_dates)

    cin = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,  # mean pore volume = 100 m3, ~1 day residence time
        n_bins=20,
    )

    # Verify output structure
    assert len(cin) == len(cin_dates), f"Expected {len(cin_dates)} output bins"

    # Count valid bins - should have substantial overlap
    valid_count = np.sum(~np.isnan(cin))
    assert valid_count >= 300, f"Expected at least 300 valid bins for constant signal, got {valid_count}"

    # For constant extraction, infiltration should also be constant in fully informed region
    # Check the middle 200 bins where steady state is guaranteed
    valid_indices = np.where(~np.isnan(cin))[0]
    assert len(valid_indices) >= 300, f"Need at least 300 valid bins, got {len(valid_indices)}"

    # Take middle 200 bins (skip 50 from each end)
    middle_indices = valid_indices[50:-50]
    assert len(middle_indices) >= 200, f"Need at least 200 middle bins, got {len(middle_indices)}"

    middle_values = cin[middle_indices]
    assert not np.any(np.isnan(middle_values)), "Middle region must have no NaN values"

    # Constant extraction with constant flow recovers the exact constant
    # infiltration in the data-dominated interior (boundary bins absorb the
    # Tikhonov bias and are clipped above).
    np.testing.assert_allclose(
        middle_values,
        5.0,
        atol=1e-9,
        err_msg="Constant extraction with constant flow should recover the exact constant infiltration",
    )


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
    flow = pd.Series([100.0] * len(cin_dates), index=cin_dates)

    cin = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=tedges,
        flow=flow,
        alpha=10.0,
        beta=10.0,  # ~1 day mean residence time
        n_bins=20,
    )

    # Explicit validation of output quantity
    assert len(cin) == len(cin_dates), f"Expected {len(cin_dates)} output bins, got {len(cin)}"
    valid_count = np.sum(~np.isnan(cin))
    assert valid_count >= 300, f"Expected at least 300 valid bins for year-long data, got {valid_count}"

    # Extract valid values and test variation
    valid_mask = ~np.isnan(cin)
    valid_cin = cin[valid_mask]
    cin_std = np.std(valid_cin)
    assert cin_std > 0.5, f"Expected std > 0.5 to see step variation, got {cin_std:.3f}"

    # Test that both low and high values are present (step function was recovered)
    cin_min = np.min(valid_cin)
    cin_max = np.max(valid_cin)
    assert cin_min < 2.0, f"Expected values below 2.0 (before step), got min={cin_min:.3f}"
    assert cin_max > 3.5, f"Expected values above 3.5 (after step), got max={cin_max:.3f}"


def test_gamma_extraction_to_infiltration_roundtrip():
    """Test gamma_infiltration_to_extraction -> gamma_extraction_to_infiltration roundtrip."""
    # Create time windows with proper alignment - use full year for sufficient overlap
    cin_dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Output window overlaps with input
    cout_dates = pd.date_range(start="2020-03-01", end="2020-10-31", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Use varying signal (sine wave) to test actual transport, not just constant recovery
    cin_original_values = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(cin_dates)) / 30.0)
    cin_original = pd.Series(cin_original_values, index=cin_dates)
    flow_cin = pd.Series([100.0] * len(cin_dates), index=cin_dates)

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
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow_cin,
        alpha=10.0,
        beta=10.0,
        n_bins=20,
    )

    # Explicit validation of overlap
    valid_mask = ~np.isnan(cin_reconstructed)
    valid_count = np.sum(valid_mask)
    assert valid_count >= 200, f"Expected at least 200 valid bins for substantial overlap, got {valid_count}"

    # Extract middle region explicitly (skip 50 bins on each end)
    valid_indices = np.where(valid_mask)[0]
    assert len(valid_indices) >= 200, f"Need at least 200 valid bins, got {len(valid_indices)}"

    middle_indices = valid_indices[50:-50]
    assert len(middle_indices) >= 100, (
        f"Need at least 100 middle bins for stable region test, got {len(middle_indices)}"
    )

    middle_start = middle_indices[0]
    middle_end = middle_indices[-1] + 1
    middle_region = slice(middle_start, middle_end)

    reconstructed_middle = cin_reconstructed[middle_region]
    original_middle = cin_original.to_numpy()[middle_region]

    # The Tikhonov inversion recovers the smooth sine input in the stable interior
    # to ~1e-13 (the measured regularization bias on this well-conditioned gamma
    # roundtrip); rtol=1e-10 is a generous margin that still fails any percent-level bias.
    np.testing.assert_allclose(
        reconstructed_middle,
        original_middle,
        rtol=1e-10,
        err_msg=f"Roundtrip error: expected mean ~{np.mean(original_middle):.2f}, got {np.mean(reconstructed_middle):.2f}",
    )


def test_gamma_infiltration_to_extraction_loc_zero_matches_legacy():
    """With loc=0 the loc-aware call must match the legacy (mean, std) call bit-for-bit."""
    dates = pd.date_range(start="2020-01-01", end="2020-06-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-02-01", end="2020-06-01", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin = pd.Series(5.0 + 2.0 * np.sin(np.linspace(0, 4 * np.pi, len(dates))), index=dates)
    flow = pd.Series(np.full(len(dates), 100.0), index=dates)

    common = {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "mean": 500.0,
        "std": 150.0,
        "n_bins": 20,
    }
    cout_default = gamma_infiltration_to_extraction(**common)
    cout_loc_zero = gamma_infiltration_to_extraction(loc=0.0, **common)
    np.testing.assert_array_equal(cout_default, cout_loc_zero)


def test_gamma_infiltration_to_extraction_loc_shifts_arrival_time():
    """A positive loc shifts the step arrival by loc/flow days.

    Comparing (mean_excess, std) fixed: case A is (mean=300, std=10, loc=0) and
    case B is (mean=500, std=10, loc=200). The underlying gamma over
    ``T - loc`` is identical in both cases, so case B is a pure time-shift of
    case A by loc/flow = 2 days.
    """
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout_dates = pd.date_range(start="2020-01-15", end="2020-10-31", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Step function: jump from 0 to 1 on day 60 (March 1)
    cin_values = np.zeros(len(dates))
    cin_values[60:] = 1.0
    cin = pd.Series(cin_values, index=dates)
    flow_rate = 100.0
    flow = pd.Series(np.full(len(dates), flow_rate), index=dates)

    std_pv = 10.0

    # Case A: loc=0, mean=300 -> mean rt = 3 days
    cout_loc0 = gamma_infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        mean=300.0,
        std=std_pv,
        loc=0.0,
        n_bins=50,
    )

    # Case B: loc=200, mean=500 -> same excess gamma, mean rt = 5 days
    loc_pv = 200.0
    cout_loc = gamma_infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        mean=300.0 + loc_pv,
        std=std_pv,
        loc=loc_pv,
        n_bins=50,
    )

    # Find the day where cout first exceeds 0.5 (step midpoint)
    valid0 = ~np.isnan(cout_loc0)
    valid_loc = ~np.isnan(cout_loc)
    common = valid0 & valid_loc
    idx0 = np.argmax((cout_loc0 > 0.5) & common)
    idx_loc = np.argmax((cout_loc > 0.5) & common)
    delay = idx_loc - idx0
    expected_delay_days = round(loc_pv / flow_rate)
    assert delay == expected_delay_days, f"Expected {expected_delay_days}-day delay for loc>0, got {delay}"


def test_gamma_roundtrip_with_loc():
    """Forward->reverse roundtrip with loc>0 must recover the input in the stable region."""
    cin_dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    cout_dates = pd.date_range(start="2020-03-01", end="2020-10-31", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cin_values = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(cin_dates)) / 30.0)
    cin_original = pd.Series(cin_values, index=cin_dates)
    flow = pd.Series(np.full(len(cin_dates), 100.0), index=cin_dates)

    mean_pv = 800.0
    std_pv = 100.0
    loc_pv = 300.0

    cout = gamma_infiltration_to_extraction(
        cin=cin_original,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        mean=mean_pv,
        std=std_pv,
        loc=loc_pv,
        n_bins=20,
    )

    cout_series = pd.Series(cout, index=cout_dates)
    cin_reconstructed = gamma_extraction_to_infiltration(
        cout=cout_series,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        mean=mean_pv,
        std=std_pv,
        loc=loc_pv,
        n_bins=20,
    )

    valid_mask = ~np.isnan(cin_reconstructed)
    valid_indices = np.where(valid_mask)[0]
    assert len(valid_indices) >= 200, f"Need at least 200 valid bins, got {len(valid_indices)}"

    middle_indices = valid_indices[50:-50]
    middle_region = slice(middle_indices[0], middle_indices[-1] + 1)
    # Well-conditioned gamma roundtrip with loc>0: the interior recovers to ~1e-13;
    # rtol=1e-10 is a generous margin that still fails any percent-level bias.
    np.testing.assert_allclose(
        cin_reconstructed[middle_region],
        cin_original.to_numpy()[middle_region],
        rtol=1e-10,
    )


def test_gamma_extraction_to_infiltration_retardation_factor():
    """Retardation shifts the deconvolved cin step earlier by (R-1)*mean_pv/flow days.

    Deconvolution maps a step in cout back to a step in cin that leads it by the
    residence time ``R*mean_pv/flow``. A larger R means a longer lead, so the
    recovered cin step lands earlier. With a narrow gamma (std << mean) the
    arrival is sharp and the R=2 step leads the R=1 step by exactly
    ``(R-1)*mean_pv/flow`` bins. This replaces a loose diff/std comparison that
    passed even under wrong-direction retardation.
    """
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_dates = pd.date_range(start="2019-11-01", end="2020-11-30", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Step function: jump from 0 to 3 on cout day 180.
    cout_values = np.zeros(len(dates))
    cout_values[180:] = 3.0
    cout = pd.Series(cout_values, index=dates)
    flow_rate = 100.0
    flow = pd.Series([flow_rate] * len(cin_dates), index=cin_dates)

    # Narrow gamma: mean_pv = 300 m³ -> mean RT = 3 days; std = 10 m³ -> 0.1 day.
    mean_pv = 300.0
    std_pv = 10.0

    cin1 = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=tedges,
        flow=flow,
        mean=mean_pv,
        std=std_pv,
        retardation_factor=1.0,
        n_bins=50,
    )
    cin2 = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=tedges,
        flow=flow,
        mean=mean_pv,
        std=std_pv,
        retardation_factor=2.0,
        n_bins=50,
    )

    # Find the step midpoint (first bin where recovered cin exceeds 1.5) on each path.
    valid = ~np.isnan(cin1) & ~np.isnan(cin2)
    idx_r1 = np.argmax((cin1 > 1.5) & valid)
    idx_r2 = np.argmax((cin2 > 1.5) & valid)

    # The R=2 cin step leads R=1 by exactly (R-1)*mean_pv/flow = 3 bins.
    expected_lead = round((2.0 - 1.0) * mean_pv / flow_rate)
    assert idx_r1 - idx_r2 == expected_lead, (
        f"Expected R=2 step to lead R=1 by {expected_lead} bins, got {idx_r1 - idx_r2}"
    )


def test_gamma_extraction_to_infiltration_with_mean_std():
    """Test gamma_extraction_to_infiltration using mean and std parameters."""
    dates = pd.date_range(start="2020-01-01", end="2020-06-30", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_dates = pd.date_range(start="2019-12-01", end="2020-05-31", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    cout = pd.Series([3.0] * len(dates), index=dates)
    flow = pd.Series([100.0] * len(cin_dates), index=cin_dates)

    # Use mean/std instead of alpha/beta
    mean = 100.0  # mean pore volume
    std = 20.0

    cin = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=tedges,
        flow=flow,
        mean=mean,
        std=std,
        n_bins=20,
    )

    # Explicit validation
    assert len(cin) == len(cin_dates), f"Expected {len(cin_dates)} output bins, got {len(cin)}"
    valid_count = np.sum(~np.isnan(cin))
    assert valid_count >= 100, f"Expected at least 100 valid bins for 6-month data, got {valid_count}"

    # For constant extraction with constant flow the deconvolution recovers the
    # exact constant in the data-dominated interior; clip spin-up/boundary bins
    # (which absorb the Tikhonov bias) and assert exact equality.
    valid_indices = np.where(~np.isnan(cin))[0]
    interior_indices = valid_indices[20:-20]
    np.testing.assert_allclose(
        cin[interior_indices],
        3.0,
        atol=1e-9,
        err_msg="Constant extraction with constant flow should recover the exact constant infiltration",
    )


def test_gamma_extraction_to_infiltration_missing_parameters():
    """Test that gamma_extraction_to_infiltration raises appropriate errors for missing parameters."""
    dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_dates = pd.date_range(start="2019-12-28", end="2020-01-08", freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    cout = pd.Series(np.ones(len(dates)), index=dates)
    flow = pd.Series(np.ones(len(cin_dates)) * 100, index=cin_dates)

    # Test missing both alpha/beta and mean/std
    with pytest.raises(ValueError):
        gamma_extraction_to_infiltration(cout=cout, tedges=cin_tedges, cout_tedges=tedges, flow=flow)


# =============================================================================
# Comprehensive tests for inverse operations (CRITICAL COVERAGE GAPS)
# =============================================================================


@pytest.mark.roundtrip
def test_gamma_roundtrip_constant_concentration():
    """Test roundtrip: infiltration->extraction->infiltration with constant concentration."""
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Constant input concentration
    cin_original = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0

    # Define cout_tedges for forward operation
    # Mean residence time = 5000/100 = 50 days, std = 10 days
    # Start output after mean + 5*std = 100 days to avoid NaN values
    cout_dates = pd.date_range(start="2020-04-20", periods=90, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Forward: infiltration to extraction
    cout = gamma_infiltration_to_extraction(
        cin=cin_original,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        mean=5000.0,
        std=1000.0,
        retardation_factor=1.0,
    )

    # Backward: extraction to infiltration
    cin_dates = pd.date_range(start="2019-12-01", periods=250, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Create flow array matching tedges (cin_tedges) length
    flow_backward = np.ones(len(cin_dates)) * 100.0

    cin_recovered = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow_backward,
        mean=5000.0,
        std=1000.0,
        retardation_factor=1.0,
    )

    # For a constant input the roundtrip must recover the exact value in the
    # data-dominated interior, not merely stay within loose bounds. The
    # boundary bins absorb the Tikhonov bias of the ill-posed inverse, so they
    # are clipped with the same valid_indices[50:-50] interior slice used by
    # the loc>0 roundtrip (test_gamma_roundtrip_with_loc). The measured
    # interior recovery floor on this loc=0 path is ~4e-12, so atol=1e-6 is a
    # ~2e5x margin while still failing any percent-level reverse-solve bias.
    valid_indices = np.where(~np.isnan(cin_recovered))[0]
    assert len(valid_indices) >= 101, f"need >=101 valid bins for the [50:-50] interior slice, got {len(valid_indices)}"

    interior_indices = valid_indices[50:-50]
    interior = slice(interior_indices[0], interior_indices[-1] + 1)
    np.testing.assert_allclose(cin_recovered[interior], 10.0, atol=1e-6)


@pytest.mark.roundtrip
def test_gamma_roundtrip_step_function():
    """Test roundtrip with step function input."""
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Step function
    cin_original = np.zeros(len(dates))
    cin_original[100:] = 15.0
    flow = np.ones(len(dates)) * 100.0

    # Define cout_tedges for forward operation
    # alpha=20, beta=250: mean = 5000, std = sqrt(20)*250 = 1118
    # Mean residence time = 5000/100 = 50 days, std = 11.18 days
    # Step at day 100 will appear in output around day 150 (100 + 50)
    # Start after day 106 to avoid NaN, run long enough to capture transition
    cout_dates = pd.date_range(start="2020-04-16", periods=90, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Forward
    cout = gamma_infiltration_to_extraction(
        cin=cin_original,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        alpha=20.0,
        beta=250.0,
        retardation_factor=1.0,
    )

    # Backward
    cin_dates = pd.date_range(start="2019-12-01", periods=250, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Create flow array matching tedges (cin_tedges) length
    flow_backward = np.ones(len(cin_dates)) * 100.0

    cin_recovered = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow_backward,
        alpha=20.0,
        beta=250.0,
        retardation_factor=1.0,
    )

    # Build the analytic step on the recovered cin grid: 0 before the step day,
    # 15.0 from the step day onward. The original step is on cin index 100 of the
    # 2020-01-01 record, i.e. at the cin_tedges day equal to dates[100].
    step_day = dates[100]
    cin_grid_days = cin_tedges[:-1]  # left edge of each recovered bin
    expected = np.where(cin_grid_days >= step_day, 15.0, 0.0)

    # The forward gamma here has a wide APVD (mean RT 50 d, std ~11 d), so the
    # inverse is genuinely ill-posed: even away from the discontinuity the
    # Tikhonov-regularized roundtrip carries a residual oscillation of ~1.6 on a
    # step height of 15 (~11%). Machine-precision recovery is therefore NOT
    # attainable for this scenario (unlike the narrow single-PV roundtrips). We
    # assert the flat data-dominated runs stay close to the original step within
    # that physical bias, which still fails any gross (>~17%) inversion regression
    # -- far stronger than the previous "anything in [-2, 30]" bound.
    valid_mask = ~np.isnan(cin_recovered)
    valid_indices = np.where(valid_mask)[0]
    assert len(valid_indices) >= 100, f"Need a long valid region, got {len(valid_indices)}"

    step_idx = int(np.argmax(cin_grid_days >= step_day))
    buffer = 15  # exclude bins within ~1.3 std of the breakthrough spread
    interior = valid_indices[20:-20]
    flat = interior[np.abs(interior - step_idx) > buffer]
    assert len(flat) > 30, f"Need enough flat interior bins, got {len(flat)}"
    np.testing.assert_allclose(
        cin_recovered[flat],
        expected[flat],
        atol=2.5,
        err_msg="Step roundtrip should recover the original step away from the discontinuity",
    )


@pytest.mark.roundtrip
def test_extraction_to_infiltration_single_pore_volume_roundtrip():
    """Test extraction_to_infiltration with single pore volume (square system) roundtrip."""
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=150, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_original = 5.0 + 3.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
    flow = np.ones(len(dates)) * 100.0
    pore_volumes = np.array([1000.0])  # Smaller pore volume for valid residence time

    # Define cout_tedges for output
    cout_dates = pd.date_range(start="2020-01-15", periods=120, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Forward
    cout = infiltration_to_extraction(
        cin=cin_original,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
    )

    # Backward
    cin_dates = pd.date_range(start="2019-12-01", periods=180, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Create flow array matching tedges (cin_tedges) length
    flow_backward = np.ones(len(cin_dates)) * 100.0

    cin_recovered = extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow_backward,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
    )

    # Single-PV roundtrip is a well-conditioned (square-shift) inverse, so the
    # recovered cin must match the original pointwise in the interior, not merely
    # in the mean. Align the recovered cin (on cin_tedges) to the original cin
    # (on tedges) by matching bin-left-edge days; clip spin-up/boundary bins.
    valid_indices = np.where(~np.isnan(cin_recovered))[0]
    assert len(valid_indices) >= 40, f"Expected a substantial valid region, got {len(valid_indices)}"

    rec_days = cin_tedges[:-1]
    orig_days = tedges[:-1]
    orig_lookup = dict(zip(orig_days, cin_original, strict=True))
    interior = valid_indices[20:-20]
    matched = [(i, orig_lookup[rec_days[i]]) for i in interior if rec_days[i] in orig_lookup]
    assert len(matched) > 20, f"Expected overlapping interior bins to compare, got {len(matched)}"
    idx = np.array([m[0] for m in matched])
    expected = np.array([m[1] for m in matched])
    np.testing.assert_allclose(cin_recovered[idx], expected, atol=1e-8)


@pytest.mark.roundtrip
def test_extraction_to_infiltration_multiple_pore_volumes():
    """Test extraction_to_infiltration with multiple pore volumes."""
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=120, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_original = np.ones(len(dates)) * 12.0
    flow = np.ones(len(dates)) * 100.0
    # Use smaller pore volumes for valid residence times
    pore_volumes = np.array([600.0, 1000.0, 1400.0])

    # Define cout_tedges
    cout_dates = pd.date_range(start="2020-01-15", periods=90, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Forward
    cout = infiltration_to_extraction(
        cin=cin_original,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
    )

    # Backward
    cin_dates = pd.date_range(start="2019-12-10", periods=150, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    # Create flow array matching tedges (cin_tedges) length
    flow_backward = np.ones(len(cin_dates)) * 100.0

    cin_recovered = extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow_backward,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
    )

    # Verify roundtrip
    valid_mask = ~np.isnan(cin_recovered)
    assert np.any(valid_mask), "Expected at least one valid recovered bin"
    valid_recovered = cin_recovered[valid_mask]

    # lstsq inversion should recover constant value to machine precision
    np.testing.assert_allclose(valid_recovered, 12.0, atol=1e-8)


def test_extraction_to_infiltration_nan_handling():
    """Test that extraction_to_infiltration properly handles periods with no valid contribution."""
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout = np.ones(len(dates)) * 10.0
    pore_volumes = np.array([1000.0])  # Smaller pore volume

    # Very short cin_tedges (before system has stabilized)
    cin_dates = pd.date_range(start="2019-12-25", periods=20, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))
    flow = np.ones(len(cin_dates)) * 100.0

    cin_recovered = extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=tedges,
        flow=flow,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
    )

    # Early indices should be NaN (no sufficient history)
    nan_count = np.sum(np.isnan(cin_recovered))
    assert nan_count > 0  # Some early values should be NaN


@pytest.mark.parametrize(
    ("mean", "std"),
    [
        (600.0, 100.0),  # Smaller pore volumes for valid residence time
        (1000.0, 200.0),
        (1500.0, 300.0),
    ],
)
def test_gamma_extraction_to_infiltration_parameter_sensitivity(mean, std):
    """Test gamma_extraction_to_infiltration with various distribution parameters."""
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=150, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout = 8.0 + 2.0 * np.sin(2 * np.pi * np.arange(len(dates)) / 40)

    # Backward
    cin_dates = pd.date_range(start="2019-12-15", periods=180, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))
    flow = np.ones(len(cin_dates)) * 100.0

    cin_recovered = gamma_extraction_to_infiltration(
        cout=cout, tedges=cin_tedges, cout_tedges=tedges, flow=flow, mean=mean, std=std, retardation_factor=1.0
    )

    # Verify outputs
    valid_mask = ~np.isnan(cin_recovered)
    if not np.any(valid_mask):
        pytest.skip("No valid recovered values")
    valid_recovered = cin_recovered[valid_mask]

    assert len(valid_recovered) > 0
    assert np.all(np.isfinite(valid_recovered))
    # Mean preservation under gamma deconvolution: the recovered cin mean
    # equals the driving cout mean (8.0) up to boundary truncation of the
    # sinusoidal period. atol=0.1 catches any gross bias; the previous
    # abs=3.0 was a 30x looser slop that masked the underlying invariant.
    np.testing.assert_allclose(np.mean(valid_recovered), 8.0, atol=0.1)


def test_extraction_to_infiltration_with_retardation():
    """Constant cin survives forward-then-reverse with retardation to machine precision.

    Forward and reverse use the same forward weight matrix (Tikhonov inversion targets
    the transpose-and-normalize of the forward). For constant cin and constant flow,
    the well-determined modes are dominated by the data and the small regularization
    bias is below 1e-10 relative.
    """
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_original = np.ones(len(dates)) * 10.0
    flow = np.ones(len(dates)) * 100.0
    pore_volumes = np.array([5000.0])
    retardation_factor = 2.0

    cout_dates = pd.date_range(start="2020-04-21", periods=40, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cout = infiltration_to_extraction(
        cin=cin_original,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=retardation_factor,
    )

    cin_dates = pd.date_range(start="2019-11-01", periods=220, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))
    flow_backward = np.ones(len(cin_dates)) * 100.0

    cin_recovered = extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow_backward,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=retardation_factor,
    )

    # Constant cout uniquely determines constant cin in the post-spinup window.
    # Restrict to indices where the source window is fully inside the cin range.
    valid_mask = ~np.isnan(cin_recovered)
    np.testing.assert_allclose(cin_recovered[valid_mask], 10.0, atol=1e-10)


def test_extraction_to_infiltration_variable_flow():
    """Constant cin survives a forward-then-reverse roundtrip under variable flow.

    The previous version used ``rel=0.4`` for a deterministic roundtrip, which lets
    a 40 % systematic bias slip through unnoticed. Tighten to ``atol=1e-10`` in the
    valid (post-spinup) region.
    """
    dates = pd.date_range(start="2020-01-01", periods=150, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin_original = np.ones(len(dates)) * 10.0
    t = np.arange(len(dates))
    flow = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 40))
    pore_volumes = np.array([5000.0])

    cout_dates = pd.date_range(start="2020-03-22", periods=40, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cout = infiltration_to_extraction(
        cin=cin_original,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow=flow,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
    )

    cin_dates = pd.date_range(start="2019-11-15", periods=180, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))
    t_backward = np.arange(len(cin_dates))
    flow_backward = 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t_backward / 40))

    cin_recovered = extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        flow=flow_backward,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
    )

    valid_mask = ~np.isnan(cin_recovered)
    np.testing.assert_allclose(cin_recovered[valid_mask], 10.0, atol=1e-10)


def test_gamma_extraction_to_infiltration_mean_preservation():
    """Constant cout uniquely recovers constant cin under gamma APVD deconvolution.

    A constant signal is in the well-determined null-space-orthogonal subspace of
    the forward operator: every streamtube returns the same value. The recovered
    cin in the valid region must therefore equal the input constant to machine
    precision (after the very small Tikhonov regularization bias).
    """
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cout = np.ones(len(dates)) * 25.0

    cin_dates = pd.date_range(start="2019-11-01", periods=250, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))
    flow = np.ones(len(cin_dates)) * 100.0

    cin_recovered = gamma_extraction_to_infiltration(
        cout=cout,
        tedges=cin_tedges,
        cout_tedges=tedges,
        flow=flow,
        mean=5000.0,
        std=1000.0,
        retardation_factor=1.0,
    )

    valid_mask = ~np.isnan(cin_recovered)
    np.testing.assert_allclose(cin_recovered[valid_mask], 25.0, atol=1e-9)


# ===============================================================================
# FLOW-WEIGHTED FRONT TRACKING TESTS
# ===============================================================================


class TestFlowWeightedFrontTracking:
    """Tests that verify flow-weighted output for front-tracking transport."""

    def test_constant_flow_unchanged(self):
        """With constant flow, front-tracking must match pure advection."""

        tedges = pd.date_range("2020-01-01", periods=31, freq="D")
        cout_tedges = pd.date_range("2020-01-01", periods=11, freq="3D")

        cin = np.zeros(30)
        cin[0:5] = 10.0
        flow = np.full(30, 100.0)
        aquifer_pore_volumes = np.array([500.0])

        cout_ft, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            retardation_factor=1.0,
        )

        cout_adv = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            retardation_factor=1.0,
            spinup=None,  # match front-tracking's spin-up handling for the comparison
        )

        valid = ~np.isnan(cout_adv) & ~np.isnan(cout_ft)
        np.testing.assert_allclose(cout_ft[valid], cout_adv[valid])

    def test_constant_cin_varying_flow_gives_constant_cout(self):
        """Constant cin with varying flow must produce constant cout."""

        tedges = pd.date_range("2020-01-01", periods=61, freq="D")
        cout_tedges = pd.date_range("2020-01-20", periods=11, freq="3D")

        cin = np.ones(60) * 7.0
        flow = 100.0 + 50.0 * np.sin(np.arange(60) * 2 * np.pi / 5)

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([400.0]),
            retardation_factor=1.0,
        )

        valid = cout > 0
        assert np.any(valid), "Expected at least one valid (post-breakthrough) output bin"
        np.testing.assert_allclose(cout[valid], 7.0, atol=1e-13)

    def test_constant_flow_mass_conservation(self):
        """Mass must be conserved under constant flow."""

        tedges = pd.date_range("2020-01-01", periods=61, freq="D")
        # Use coarser cout so multiple flow bins fall in one cout bin
        cout_tedges = pd.date_range("2020-01-01", periods=21, freq="3D")

        cin = np.zeros(60)
        cin[5:10] = 10.0
        flow = np.full(60, 100.0)

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([300.0]),
            retardation_factor=1.0,
        )

        # Mass in = Σ cin_i * Q_i * dt_i
        dt_in = np.ones(60)
        mass_in = np.sum(cin * flow * dt_in)
        # Mass out = Σ cout_i * Q_cout_i * dt_cout_i
        # With constant flow, Q_cout = Q = 100
        dt_out = np.diff(((cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)).values)
        mass_out = np.sum(cout * flow[0] * dt_out)
        np.testing.assert_allclose(mass_out, mass_in, rtol=1e-13)

    def test_varying_flow_mass_conservation_is_flow_weighted(self):
        """Flow-weighted output mass balance under genuinely varying flow.

        This is the only test that combines a *varying* flow with a *varying*
        cin and a cout grid coarser than the flow grid (several flow values per
        cout bin). The bin-averaging in :func:`infiltration_to_extraction_nonlinear_sorption`
        is flow-weighted: ``cout[k] = Σ_i Q_i c_i Δt_i / Σ_i Q_i Δt_i`` over the
        fine sub-bins ``i`` falling in cout bin ``k`` (advection.py). Multiplying
        each ``cout[k]`` by its true flow integral ``∫_bin Q dt`` and summing must
        recover the total injected mass ``Σ_i cin_i Q_i Δt_i`` exactly.

        Because flow varies *within* each cout bin, the flow-weighted average
        differs from the plain time-average: dropping the ``q_fine`` factor from
        the weight (so ``cout`` becomes a time-average) breaks this identity at
        the percent level even though every NaN/shape check still passes.

        The ``∫_bin Q dt`` weight is computed from the *same* piecewise-constant
        flow the advection uses: the integral of flow over a cout bin equals the
        flow-bin-overlap sum, identical to the internal fine-grid ``Σ q_fine·dt``
        since the fine grid is built from the union of the cout and flow edges.
        """

        n = 60
        tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
        t = np.arange(n)
        flow = 100.0 + 40.0 * np.sin(2 * np.pi * t / 7.0)

        # Finite cin pulse so the injected mass is finite and fully captured.
        cin = np.zeros(n)
        cin[5:12] = 10.0

        # cout grid coarser than the daily flow grid: 3-day bins put several
        # distinct flow values inside each cout bin (the regime that
        # distinguishes flow-weighting from time-averaging).
        cout_tedges = pd.date_range("2020-01-01", periods=21, freq="3D")

        cout, _ = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([300.0]),
            retardation_factor=1.0,
        )

        # Mass in = Σ_i cin_i Q_i Δt_i.
        dt_in = np.diff(((tedges - tedges[0]) / pd.Timedelta(days=1)).values)
        mass_in = np.sum(cin * flow * dt_in)

        # ∫_bin Q dt for each cout bin: overlap of the cout bin with each
        # (piecewise-constant) flow bin, summed. Vectorized, no Python loop.
        flow_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
        cout_days = ((cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)).values
        lo = np.maximum(cout_days[:-1, None], flow_days[None, :-1])
        hi = np.minimum(cout_days[1:, None], flow_days[None, 1:])
        overlap = np.clip(hi - lo, 0.0, None)
        int_q_dt = (overlap * flow[None, :]).sum(axis=1)

        valid = ~np.isnan(cout)
        mass_out = np.sum(cout[valid] * int_q_dt[valid])
        np.testing.assert_allclose(mass_out, mass_in, rtol=1e-13)


# =============================================================================
# Negative-flow rejection and zero-flow invariance
# =============================================================================


def _zero_flow_invariance_setup():
    """Baseline and zero-flow-inserted scenarios sharing the same cout grid."""
    n = 200
    tedges_base = pd.date_range(start="2020-01-01", periods=n + 1, freq="D")
    cin_base = 10.0 + np.sin(np.linspace(0, 4 * np.pi, n))
    flow_base = np.full(n, 100.0)

    k = 100  # insertion index
    tedges_mod = pd.date_range(start="2020-01-01", periods=n + 2, freq="D")
    cin_mod = np.insert(cin_base, k, cin_base[k - 1])
    flow_mod = np.insert(flow_base, k, 0.0)

    return {
        "n": n,
        "k": k,
        "tedges_base": tedges_base,
        "cin_base": cin_base,
        "flow_base": flow_base,
        "tedges_mod": tedges_mod,
        "cin_mod": cin_mod,
        "flow_mod": flow_mod,
    }


def test_infiltration_to_extraction_accepts_zero_flow_without_warnings():
    """flow == 0 is accepted; no division-by-zero warnings emitted."""
    tedges = pd.date_range(start="2020-01-01", periods=201, freq="D")
    cin = np.ones(200)
    flow = np.full(200, 100.0)
    flow[100] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([500.0]),
        )


def test_infiltration_to_extraction_zero_flow_insertion_invariance():
    """Insert a zero-flow bin: cout at that bin is NaN; other bins are unchanged.

    The pre-fix version masked the zero-flow output cell with ``np.delete`` before
    asserting, hiding bug #161 (zero-flow cout bins returning fabricated values).
    Under strict-validity, a zero-flow cout bin must be NaN; after stripping it,
    the remaining cout matches the baseline to machine precision.
    """
    s = _zero_flow_invariance_setup()
    apv = np.array([500.0])

    cout_base = infiltration_to_extraction(
        cin=s["cin_base"],
        flow=s["flow_base"],
        tedges=s["tedges_base"],
        cout_tedges=s["tedges_base"],
        aquifer_pore_volumes=apv,
    )
    cout_mod = infiltration_to_extraction(
        cin=s["cin_mod"],
        flow=s["flow_mod"],
        tedges=s["tedges_mod"],
        cout_tedges=s["tedges_mod"],
        aquifer_pore_volumes=apv,
    )

    # The inserted zero-flow cout bin MUST be NaN, not a fabricated mean of nearby bins.
    assert np.isnan(cout_mod[s["k"]])

    # After stripping the inserted zero-flow position, cout_mod matches cout_base.
    cout_mod_stripped = np.delete(cout_mod, s["k"])
    assert np.sum(np.isnan(cout_mod_stripped)) == np.sum(np.isnan(cout_base))
    valid = ~np.isnan(cout_base) & ~np.isnan(cout_mod_stripped)
    np.testing.assert_allclose(cout_mod_stripped[valid], cout_base[valid], atol=1e-10)


# =============================================================================
# Regression tests for issue #161 (strict-validity / zero-flow NaN)
# and issue #169 (variable-flow correctness, linearity, time-translation, mass)
# =============================================================================


def test_infiltration_to_extraction_strict_validity_left_boundary_nan():
    """All cout bins are NaN until every streamtube has broken through.

    Regression test for issue #161 (mass over-attribution at the cin left-time
    boundary). With ``apv = [100, 500, 1500]`` and constant flow=100 m³/d,
    residence times are 1, 5, and 15 days. The cout window starts at the same
    instant as cin (day 0), so the longest-residence streamtube cannot have a
    valid source window until day 15. Pre-fix the function returned a partial
    count-mean over the contributing subset and over-attributed mass to the
    earliest cin bins (Σcout = 1.833 vs cin pulse mass 1.0); under strict
    validity the spin-up region is NaN.
    """
    n = 50
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    apv = np.array([100.0, 500.0, 1500.0])

    cin = np.zeros(n)
    cin[0] = 1.0  # left-edge pulse

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
        spinup=None,  # strict-validity: this is the property under test
    )

    # First valid cout bin is at index 15 (longest residence time = 15 d).
    assert np.all(np.isnan(cout[:15]))
    assert not np.isnan(cout[15])
    # cout[15] captures cin[0]'s mass via the longest streamtube only: 1/N = 1/3.
    # The bundle row sums to N/N = 1 to ULP, so cout[15] = 1/3 to a few eps.
    np.testing.assert_allclose(cout[15], 1.0 / 3.0, atol=1e-15)


def test_infiltration_to_extraction_mass_conservation_full_breakthrough():
    """Mid-domain pulse: full mass conserved when cout window covers all break-through times.

    With cin in [0, 100) and a pulse at days [40, 41), all streamtubes
    (PVs 100, 500, 1500 → RTs 1, 5, 15 days) deliver their share of the
    pulse mass to bins inside [41, 56], well before the right edge of cout.
    Mass conservation must hold to machine precision.
    """
    n = 100
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    apv = np.array([100.0, 500.0, 1500.0])

    cin = np.zeros(n)
    cin[40:42] = 5.0

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
    )

    cin_mass = np.sum(cin * flow)
    cout_mass = np.nansum(cout * flow)
    np.testing.assert_allclose(cout_mass, cin_mass, atol=1e-10)


def test_infiltration_to_extraction_zero_flow_bin_returns_nan_single_pv():
    """A zero-flow cout bin must be NaN, not a fabricated value.

    With one streamtube the source window for the zero-flow bin collapses to a
    near-zero width; floating-point jitter of the residence-time computation
    produces a spurious "valid" weight of ~1.0 on whichever cin bin contains
    the collapse point. Pre-fix this gave a value resembling a real measurement
    (e.g., ~10 for a sin-around-10 cin signal). Under strict validity, the
    integrated flow during the cout bin is exactly zero, so the bin is NaN.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[100] = 0.0
    cin = 10.0 + np.sin(np.linspace(0.0, 4.0 * np.pi, n))

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=np.array([500.0]),
        retardation_factor=1.0,
    )

    assert np.isnan(cout[100]), "zero-flow cout bin must be NaN"


def test_infiltration_to_extraction_zero_flow_bin_returns_nan_multi_pv():
    """Zero-flow cout bin is NaN even with multiple streamtubes (no count-mean fabrication)."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[100] = 0.0
    cin = 10.0 + np.sin(np.linspace(0.0, 4.0 * np.pi, n))

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=np.array([100.0, 500.0, 1500.0]),
        retardation_factor=1.0,
    )

    assert np.isnan(cout[100])


def test_infiltration_to_extraction_variable_flow_constant_cin_post_spinup():
    """Constant cin x variable flow x multi-PV -> constant cout post-spinup (issue #169 group 1).

    The pre-fix mass over-attribution combined with variable flow could mask
    incorrect flow weighting. Under strict-validity the flow-weighted bundle row
    sums to 1, so a constant cin must produce constant cout to machine precision.
    """
    n = 600
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 100.0 + 50.0 * np.sin(np.arange(n) * 2 * np.pi / 30.0)
    cin = np.full(n, 7.5)
    apv = np.array([100.0, 500.0, 1500.0])

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
    )

    valid = ~np.isnan(cout)
    assert np.sum(valid) > 500
    np.testing.assert_allclose(cout[valid], 7.5, atol=1e-13)


def test_infiltration_to_extraction_variable_flow_mass_conservation():
    """Mass conservation holds under variable flow with full breakthrough (issue #169 group 9).

    Mutation testing flagged that flow-weighting could be silently dropped
    (replacing ``flow[None, :]`` with ``np.ones_like(flow)[None, :]``) without
    any existing test failing. A pulse propagated under variable flow with all
    streamtubes inside the cin range must conserve total mass.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    rng = np.random.default_rng(42)
    flow = 100.0 + 30.0 * rng.uniform(-1.0, 1.0, n)
    apv = np.array([200.0, 600.0])

    cin = np.zeros(n)
    cin[80:85] = 3.0  # mid-domain pulse

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
    )

    cin_mass = np.sum(cin * flow)
    cout_mass = np.nansum(cout * flow)
    np.testing.assert_allclose(cout_mass, cin_mass, atol=1e-10)


def test_infiltration_to_extraction_linearity():
    """Forward operator is linear: ``f(alpha*a + beta*b) == alpha*f(a) + beta*f(b)`` (issue #169 group 2)."""
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 100.0 + 20.0 * np.sin(np.arange(n) * 2 * np.pi / 23.0)
    apv = np.array([200.0, 700.0])

    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 5.0, n)
    b = rng.uniform(0.0, 5.0, n)
    alpha, beta = 2.5, -1.3

    fa = infiltration_to_extraction(
        cin=a,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
    )
    fb = infiltration_to_extraction(
        cin=b,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
    )
    f_combined = infiltration_to_extraction(
        cin=alpha * a + beta * b,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
    )

    expected = alpha * fa + beta * fb
    valid = ~np.isnan(f_combined) & ~np.isnan(expected)
    assert np.sum(valid) > 100
    np.testing.assert_allclose(f_combined[valid], expected[valid], atol=1e-13)


def test_infiltration_to_extraction_time_translation_invariance():
    """Shifting all time edges by Δt shifts cout by Δt (issue #169 group 5).

    An indexing bug that referenced absolute time would survive the existing
    suite. Shifting tedges, cout_tedges, cin, and flow together must produce
    bit-identical outputs (the algorithm only uses time differences).
    """
    n = 150
    tedges_a = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    tedges_b = pd.date_range("2025-08-15", periods=n + 1, freq="D")
    flow = 100.0 + 25.0 * np.sin(np.arange(n) * 2 * np.pi / 17.0)
    apv = np.array([300.0, 800.0])

    rng = np.random.default_rng(1)
    cin = rng.uniform(0.0, 4.0, n)

    cout_a = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges_a,
        cout_tedges=tedges_a,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
    )
    cout_b = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges_b,
        cout_tedges=tedges_b,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
    )

    # NaN positions must coincide
    np.testing.assert_array_equal(np.isnan(cout_a), np.isnan(cout_b))
    valid = ~np.isnan(cout_a)
    np.testing.assert_array_equal(cout_a[valid], cout_b[valid])


def test_infiltration_to_extraction_integer_residence_time_exact_shift():
    """Single PV with integer residence time: cout[k] == cin[k - n] exactly (issue #169 group 3)."""
    n = 80
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    pv = 300.0  # PV / flow = 3 days exactly
    rng = np.random.default_rng(2)
    cin = rng.uniform(0.0, 5.0, n)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=np.array([pv]),
        retardation_factor=1.0,
        spinup=None,  # strict-validity: tests the exact-shift property where cout has NaN spin-up
    )

    shift = int(pv / 100.0)
    valid = ~np.isnan(cout)
    expected = np.full(n, np.nan)
    expected[shift:] = cin[: n - shift]
    np.testing.assert_array_equal(cout[valid], expected[valid])


# ---------------------------------------------------------------------------
# Tests for the public ``spinup`` parameter
# ---------------------------------------------------------------------------


def test_spinup_constant_eliminates_left_edge_nan():
    """spinup='constant' warm-starts the system so left-edge cout has no NaN.

    With ``apv = [100, 500, 1500]``, flow=100, the strict-validity left-edge
    spin-up extends to day 15 (longest RT). Under ``spinup="constant"``
    the warm-start prepends bins so every cout bin at or after tedges[0]
    is valid. On the bins where the strict computation IS valid, both
    modes must agree exactly.
    """
    n = 50
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    apv = np.array([100.0, 500.0, 1500.0])
    cin = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 10.0)

    cout_strict = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=None
    )
    cout_warm = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup="constant"
    )

    assert np.all(np.isnan(cout_strict[:15]))
    assert not np.any(np.isnan(cout_warm)), "constant warm-start should leave no left-edge NaN"
    valid = ~np.isnan(cout_strict)
    np.testing.assert_array_equal(cout_warm[valid], cout_strict[valid])


@pytest.mark.parametrize("retardation_factor", [2.0, 3.0])
def test_spinup_constant_eliminates_left_edge_nan_with_retardation(retardation_factor):
    """warm-start pad must scale with the retardation factor, not just pore volume.

    The warm-start pad length is ``R · v_max / q0`` (advection_utils.py): the
    longest streamtube's *retarded* residence time. With ``apv=[100,500,1500]``
    and ``flow=100`` the longest unretarded RT is 15 days, so at ``R=2``/``R=3``
    the strict-validity left edge extends to 30/45 days. ``spinup="constant"``
    must still warm-start every cout bin at or after ``tedges[0]``.

    Mirrors :func:`test_spinup_constant_eliminates_left_edge_nan` at ``R>1``.
    Dropping the ``R·`` factor from the pad (so it degenerates to the ``R=1``
    length of 15 days) silently re-introduces left-edge NaN bins in
    ``[15, R·15)`` — this test fails on that mutation while the ``R=1`` test
    still passes.
    """
    n = 80
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    apv = np.array([100.0, 500.0, 1500.0])
    cin = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(n) / 10.0)

    cout_strict = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=retardation_factor,
        spinup=None,
    )
    cout_warm = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=retardation_factor,
        spinup="constant",
    )

    # Strict validity blanks the left edge up to the longest retarded RT.
    longest_rt_days = int(retardation_factor * 1500.0 / 100.0)
    assert np.all(np.isnan(cout_strict[:longest_rt_days]))
    assert not np.any(np.isnan(cout_warm)), "constant warm-start should leave no left-edge NaN at R>1"
    valid = ~np.isnan(cout_strict)
    np.testing.assert_array_equal(cout_warm[valid], cout_strict[valid])


def test_spinup_constant_uses_first_cin_value():
    """The warm-start pads cin with cin[0], not zero or cin[-1].

    With a single PV giving integer residence time of 3 days and
    ``cin[0]=7, cin[1:]=0``, the warm-start interpretation says cin was 7
    for all time before tedges[0]. The first 4 cout bins therefore see
    cin=7 in their source windows (the warm-start plus cin[0]); from
    cout[4] onward the source windows fall on cin[1:]=0.
    """
    n = 50
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    cin = np.zeros(n)
    cin[0] = 7.0

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=np.array([300.0]),  # RT = 3 days
        spinup="constant",
    )

    np.testing.assert_array_equal(cout[:4], 7.0)
    np.testing.assert_array_equal(cout[4:], 0.0)


def test_spinup_constant_does_not_pad_right_edge():
    """cout extending past tedges[-1] is still NaN under spinup='constant'.

    Warm-start only handles the left edge. Right-edge cout bins whose
    source windows extend past tedges[-1] remain NaN regardless of mode.
    """
    cin_dates = pd.date_range("2020-01-01", end="2020-01-15", freq="D")
    cout_dates = pd.date_range("2020-01-01", end="2020-01-20", freq="D")
    tedges = pd.DatetimeIndex([*cin_dates, cin_dates[-1] + pd.Timedelta(days=1)])
    cout_tedges = pd.DatetimeIndex([*cout_dates, cout_dates[-1] + pd.Timedelta(days=1)])

    cin = np.full(len(cin_dates), 5.0)
    flow = np.full(len(cin_dates), 100.0)
    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),  # RT = 5 days
        spinup="constant",
    )

    # cin ends at index 15 of cout_tedges (2020-01-16). cout bins past
    # 2020-01-15 with RT=5 source windows extending past tedges[-1] are NaN.
    assert np.any(np.isnan(cout)), "right-edge bins should be NaN under constant warm-start"
    # Left-edge bins are populated (cin[0]=5 warm-started)
    assert not np.isnan(cout[0])


def test_spinup_zero_threshold_reproduces_issue_161_overattribution():
    """spinup=0.0 reproduces the count-mean issue #161 over-attribution.

    With a left-edge cin pulse cin[0]=1.0, apv=[100, 500, 1500], flow=100,
    the count-mean over contributing streamtubes gives:
    - cout[1] (only 1d streamtube contributes): cin[0]/1 = 1.0
    - cout[5] (1d + 5d contribute): cin[0]/2 = 0.5
    - cout[15] (all 3 contribute): cin[0]/3 ≈ 0.333

    Sum = 1 + 0.5 + 1/3 = 11/6 ≈ 1.833 — the documented pre-fix
    over-attribution. This is the deliberately opt-in behavior of the
    fraction-threshold mode at ``spinup=0.0``.
    """
    n = 50
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    apv = np.array([100.0, 500.0, 1500.0])
    cin = np.zeros(n)
    cin[0] = 1.0

    cout = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=0.0
    )

    np.testing.assert_allclose(cout[1], 1.0, atol=1e-15)
    np.testing.assert_allclose(cout[5], 0.5, atol=1e-15)
    np.testing.assert_allclose(cout[15], 1.0 / 3.0, atol=1e-15)
    np.testing.assert_allclose(np.nansum(cout), 11.0 / 6.0, atol=1e-13)


def test_spinup_constant_round_trip_recovers_cin():
    """Forward+reverse with spinup='constant' on both ends recovers cin.

    The warm-start padding extends both directions identically, so the
    forward output has no spin-up NaN to feed into the reverse problem.
    With a single pore volume (well-conditioned inverse) recovery is
    tight to machine precision in the interior.
    """
    n = 200
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / 30.0)
    flow = np.full(n, 100.0)
    apv = np.array([301.0])  # avoid integer-RT rank deficiency

    cout = infiltration_to_extraction(cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv)
    assert not np.any(np.isnan(cout))
    cin_rec = extraction_to_infiltration(
        cout=cout, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv
    )

    # Interior recovery should be tight to machine precision once the
    # boundary bins (which absorb Tikhonov bias) are clipped on both ends.
    interior = slice(int(0.15 * n), int(0.85 * n))
    np.testing.assert_allclose(cin_rec[interior], cin[interior], atol=1e-12)


def test_spinup_constant_falls_back_when_flow_zero():
    """spinup='constant' silently falls back to strict when flow[0]=0.

    The warm-start needs flow[0] > 0 to derive a finite warm-start
    residence time. When the first flow bin is zero, the helper falls
    back to strict-validity (left-edge NaN), so the output matches
    spinup=None.
    """
    n = 20
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[0] = 0.0
    cin = np.ones(n)
    cout_warm = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=np.array([500.0]), spinup="constant"
    )
    cout_strict = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=np.array([500.0]), spinup=None
    )
    np.testing.assert_array_equal(np.isnan(cout_warm), np.isnan(cout_strict))


@pytest.mark.parametrize("bad", ["constants", "none", "", "default"])
def test_spinup_invalid_string_raises(bad):
    """spinup must be 'constant', None, or a float in [0, 1]."""
    n = 10
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    with pytest.raises(ValueError, match="spinup"):
        infiltration_to_extraction(
            cin=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([100.0]),
            spinup=bad,
        )


@pytest.mark.parametrize("bad", [True, False])
def test_spinup_bool_raises_type_error(bad):
    """Booleans are rejected (Python bool is a numeric subtype, easy to mis-pass)."""
    n = 10
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    with pytest.raises(TypeError, match="spinup"):
        infiltration_to_extraction(
            cin=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([100.0]),
            spinup=bad,
        )


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0, -1.0])
def test_spinup_float_out_of_range_raises(bad):
    """Float spinup must be in [0, 1]."""
    n = 10
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    with pytest.raises(ValueError, match="spinup"):
        infiltration_to_extraction(
            cin=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([100.0]),
            spinup=bad,
        )


# =============================================================================
# Front-tracking output binning — defensive guard against off-by-one routing
# =============================================================================


def test_flow_weighted_front_tracking_output_edge_routing(monkeypatch):
    """Routing of fine sub-bins to flow / cout bins must follow the half-open
    ``[t_k, t_{k+1})`` convention.

    In (V, θ) the helper translates fine ``t``-edges to ``θ``-edges via the
    ``(flow_tedges_days, theta_edges)`` map and queries the patched
    ``compute_bin_averaged_concentration_exact`` in θ-space. The flow-weighting
    that follows is still in (t, flow) — this test pins the routing invariant.
    """
    cout_tedges_days = np.array([0.0, 6.0, 12.0])
    flow_tedges_days = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    flow = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    dt_flow = np.diff(flow_tedges_days)
    theta_edges = np.concatenate(([0.0], np.cumsum(flow * dt_flow)))

    # Patch C(θ) to be the θ-midpoint of each fine sub-bin so the flow-weighted
    # average is strictly routing-dependent.
    def _linear_c(*, theta_bin_edges, v_outlet, waves, sorption, cin=None, theta_edges_inlet=None):
        del v_outlet, waves, sorption, cin, theta_edges_inlet
        return (theta_bin_edges[:-1] + theta_bin_edges[1:]) / 2

    monkeypatch.setattr(adv_mod, "compute_bin_averaged_concentration_exact", _linear_c)

    out = adv_mod._flow_weighted_front_tracking_output(  # noqa: SLF001
        cout_tedges_days=cout_tedges_days,
        flow_tedges_days=flow_tedges_days,
        flow=flow,
        v_outlet=100.0,
        waves=[],
        sorption=ConstantRetardation(retardation_factor=1.0),
        theta_edges=theta_edges,
        cin=np.zeros(len(flow)),
    )

    # Reference: explicit half-open bin assignment, with c_fine computed in θ
    # to match the patched function.
    fine_edges = np.unique(np.concatenate([cout_tedges_days, flow_tedges_days[1:-1]]))
    fine_mids = (fine_edges[:-1] + fine_edges[1:]) / 2
    dt_fine = np.diff(fine_edges)
    fine_theta_edges = np.interp(fine_edges, flow_tedges_days, theta_edges)
    c_fine = (fine_theta_edges[:-1] + fine_theta_edges[1:]) / 2
    # For each fine_mid, find the flow bin k such that t_k <= mid < t_{k+1}.
    flow_idx = np.array([np.searchsorted(flow_tedges_days, m, side="right") - 1 for m in fine_mids])
    cout_idx = np.array([np.searchsorted(cout_tedges_days, m, side="right") - 1 for m in fine_mids])
    expected = np.zeros(len(cout_tedges_days) - 1)
    for k in range(len(expected)):
        mask = cout_idx == k
        q_dt = flow[flow_idx[mask]] * dt_fine[mask]
        expected[k] = np.sum(c_fine[mask] * q_dt) / np.sum(q_dt)

    np.testing.assert_allclose(out, expected, atol=1e-12)


# =============================================================================
# Variable-flow + sinusoidal cin: analytic-reference correctness (Group 1)
# =============================================================================


def _reference_single_pv_variable_flow(cin, flow, tedges_days, cout_tedges_days, pore_volume, retardation):
    """Compute reference cout for a single streamtube under variable flow.

    Uses cumulative-pumped-volume inverse-interpolation followed by
    flow-weighted bin-overlap aggregation. Independent of the package's
    internal ``_infiltration_to_extraction_weights`` so the comparison
    cross-validates both code paths.
    """
    dt_cin = np.diff(tedges_days)
    flow_cum = np.concatenate([[0.0], np.cumsum(flow * dt_cin)])  # at tedges
    flow_cum_at_cout = np.interp(cout_tedges_days, tedges_days, flow_cum)
    v_source_left = flow_cum_at_cout[:-1] - retardation * pore_volume
    v_source_right = flow_cum_at_cout[1:] - retardation * pore_volume
    t_source_left = np.interp(v_source_left, flow_cum, tedges_days, left=np.nan, right=np.nan)
    t_source_right = np.interp(v_source_right, flow_cum, tedges_days, left=np.nan, right=np.nan)

    n_cout = len(cout_tedges_days) - 1
    reference = np.full(n_cout, np.nan)
    for k in range(n_cout):
        if np.isnan(t_source_left[k]) or np.isnan(t_source_right[k]):
            continue
        a, b = t_source_left[k], t_source_right[k]
        overlap = np.maximum(0.0, np.minimum(b, tedges_days[1:]) - np.maximum(a, tedges_days[:-1]))
        weights = flow * overlap
        total = np.sum(weights)
        if total > 0:
            reference[k] = np.sum(cin * weights) / total
    return reference


@pytest.mark.parametrize(
    ("flow_profile", "pore_volume", "retardation"),
    [
        ("sine7", 200.0, 1.0),
        ("sine7", 200.0, 2.0),
        ("step", 300.0, 1.0),
        ("cosine5", 450.0, 1.5),
    ],
)
def test_infiltration_to_extraction_sinusoidal_variable_flow_analytic(flow_profile, pore_volume, retardation):
    """Sinusoidal cin under variable flow matches an analytic flow-weighted
    bin-convolution reference to machine precision."""
    n_cin = 80
    tedges = pd.date_range("2020-01-01", periods=n_cin + 1, freq="D")
    cout_tedges = pd.date_range("2020-01-15", periods=40, freq="D")

    t_days = np.arange(n_cin)
    cin = 5.0 + 2.0 * np.sin(2 * np.pi * t_days / 11.0)
    if flow_profile == "sine7":
        flow = 100.0 + 30.0 * np.sin(2 * np.pi * t_days / 7.0)
    elif flow_profile == "cosine5":
        flow = 120.0 + 40.0 * np.cos(2 * np.pi * t_days / 5.0)
    elif flow_profile == "step":
        flow = np.where(t_days < n_cin // 2, 90.0, 160.0)
    else:
        raise AssertionError(flow_profile)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([pore_volume]),
        retardation_factor=retardation,
        spinup=None,
    )

    tedges_days = ((tedges - tedges[0]) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta(days=1)).values
    reference = _reference_single_pv_variable_flow(
        cin=cin,
        flow=flow,
        tedges_days=tedges_days,
        cout_tedges_days=cout_tedges_days,
        pore_volume=pore_volume,
        retardation=retardation,
    )

    valid = ~np.isnan(reference) & ~np.isnan(cout)
    assert np.any(valid), "Test setup gives no valid bins"
    np.testing.assert_allclose(cout[valid], reference[valid], atol=1e-12)


# =============================================================================
# Impulse response vs discrete RTD kernel (Group 4)
# =============================================================================


def _reference_impulse_constant_flow(pore_volumes, flow_rate, pulse_left, pulse_right, n_cout, cout_bin_width):
    """Discrete RTD reference for a single-bin unit pulse with constant flow.

    Returns the streamtube-arithmetic-mean ``cout`` for each cout bin under
    ``retardation_factor=1.0``: per streamtube, the source window for cout
    bin k is ``[k*Δ - V/Q, (k+1)*Δ - V/Q]`` (in days from ``tedges[0]``), and
    its flow-weighted contribution to that cout bin is ``overlap_with_pulse / Δ``
    (constant flow cancels). The bundle mean averages over streamtubes.
    """
    n_pv = len(pore_volumes)
    cout = np.zeros(n_cout)
    for pv in pore_volumes:
        rt = pv / flow_rate
        for k in range(n_cout):
            src_left = k * cout_bin_width - rt
            src_right = (k + 1) * cout_bin_width - rt
            overlap = max(0.0, min(src_right, pulse_right) - max(src_left, pulse_left))
            cout[k] += (overlap / cout_bin_width) / n_pv
    return cout


@pytest.mark.parametrize(
    "pv_distribution",
    [
        "uniform",
        "two_cluster",
        "single",
    ],
)
def test_infiltration_to_extraction_impulse_response_discrete_rtd(pv_distribution):
    """Single-bin pulse cin under constant flow + multi-PV: bundle output
    equals the discrete sum-of-deltas RTD kernel to machine precision."""
    n_cin = 60
    tedges = pd.date_range("2020-01-01", periods=n_cin + 1, freq="D")
    cout_tedges = tedges

    pulse_idx = 8
    cin = np.zeros(n_cin)
    cin[pulse_idx] = 1.0
    flow_rate = 100.0
    flow = np.full(n_cin, flow_rate)

    if pv_distribution == "uniform":
        pore_volumes = np.array([200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0])
    elif pv_distribution == "two_cluster":
        pore_volumes = np.array([200.0, 210.0, 220.0, 600.0, 610.0, 620.0])
    elif pv_distribution == "single":
        pore_volumes = np.array([350.0])
    else:
        raise AssertionError(pv_distribution)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
        spinup=None,
    )

    reference = _reference_impulse_constant_flow(
        pore_volumes=pore_volumes,
        flow_rate=flow_rate,
        pulse_left=float(pulse_idx),
        pulse_right=float(pulse_idx + 1),
        n_cout=n_cin,
        cout_bin_width=1.0,
    )

    valid = ~np.isnan(cout)
    assert np.any(valid)
    np.testing.assert_allclose(cout[valid], reference[valid], atol=1e-12)


# =============================================================================
# Forward x reverse weight-matrix identity (Group 7)
# =============================================================================


@pytest.mark.parametrize("spinup", [None, "constant"])
def test_weight_matrix_identity_forward_reverse(spinup):
    """The W matrix used by the reverse path is bit-identical to the W matrix
    built by the forward path on the same inputs.

    The reverse path's Tikhonov regularization is applied inside
    ``solve_inverse_transport`` *after* W is constructed, so the underlying
    operator must match the forward one exactly. The test rebuilds W twice
    in the test by replicating the same private pipeline both paths use.
    """
    n = 40
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    flow = 100.0 + 25.0 * np.sin(np.arange(n) * 2 * np.pi / 9)
    cin = 3.0 + np.sin(np.arange(n) * 2 * np.pi / 13)
    aquifer_pore_volumes = np.array([200.0, 350.0, 500.0])
    retardation_factor = 1.0

    def _build_w(cin_arg):
        weight_tedges, weight_flow, _, threshold, _ = _resolve_spinup_inputs(
            spinup,
            tedges=tedges,
            flow=flow,
            aquifer_pore_volumes=aquifer_pore_volumes,
            retardation_factor=retardation_factor,
            cin=cin_arg,
        )
        acc, contrib, zero_flow = _infiltration_to_extraction_weights(
            tedges=weight_tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            flow=weight_flow,
            retardation_factor=retardation_factor,
        )
        w, _ = _resolve_spinup_mask(
            accumulated_weights=acc,
            contributing_bins=contrib,
            zero_flow_cout=zero_flow,
            n_pv=len(aquifer_pore_volumes),
            spinup=threshold,
        )
        return w

    w_forward = _build_w(cin)
    w_reverse = _build_w(None)
    np.testing.assert_array_equal(w_forward, w_reverse)


# =============================================================================
# Multi-PV boundary partial-coverage contract (Group 8)
# =============================================================================


def _per_streamtube_constant_flow_reference(*, cin, flow_rate, pore_volume, n_cout, cin_offset_days=0.0):
    """Per-streamtube cout reference under constant flow, retardation=1.

    For a streamtube with residence time RT = pore_volume / flow_rate, the
    cout for bin k spans cin time ``[k - RT, k+1 - RT]``. Computes the
    flow-weighted average (constant flow makes this an overlap-weighted
    mean) over the cin bins that intersect this source window. Returns NaN
    when the source window is not fully covered by cin.

    ``cin_offset_days`` is the start of cin in days relative to cout's
    start (negative if cin is padded earlier than cout).
    """
    rt = pore_volume / flow_rate
    n_cin = len(cin)
    cin_left = cin_offset_days
    cin_right = cin_offset_days + n_cin
    out = np.full(n_cout, np.nan)
    for k in range(n_cout):
        src_left = k - rt
        src_right = (k + 1) - rt
        if src_left < cin_left or src_right > cin_right:
            continue
        total_overlap = 0.0
        weighted_sum = 0.0
        for i in range(n_cin):
            bin_left = cin_left + i
            bin_right = cin_left + i + 1
            overlap = max(0.0, min(src_right, bin_right) - max(src_left, bin_left))
            if overlap > 0:
                weighted_sum += cin[i] * overlap
                total_overlap += overlap
        if total_overlap > 0:
            out[k] = weighted_sum / total_overlap
    return out


@pytest.mark.parametrize("spinup", [None, "constant"])
def test_multi_pv_boundary_partial_coverage_contract(spinup):
    """Pin the contract for cout bins with partial multi-PV streamtube
    coverage at the cin boundary.

    Setup: two streamtubes with very different residence times (RT = 1d
    short, RT = 8d long), constant flow. Early cout bins have a fully-valid
    source window for the short PV but a source window past ``tedges[0]``
    for the long PV. This is the multi-PV partial-coverage scenario the
    spinup parameter was introduced to handle (issue #161 + PR #178).

    Contract pinned here:

    - ``spinup=None`` (strict): bins where ANY streamtube has an invalid
      source window become NaN. The exact NaN indices are pinned.
    - ``spinup="constant"``: left-edge padding restores validity for all
      streamtubes; the cout values match the bundle mean of per-streamtube
      flow-weighted overlap references built independently of the package.
    """
    n_cin = 30
    tedges = pd.date_range("2020-01-01", periods=n_cin + 1, freq="D")
    cout_tedges = tedges

    flow_rate = 100.0
    flow = np.full(n_cin, flow_rate)
    cin = 2.0 + np.cos(np.arange(n_cin) * 2 * np.pi / 11)
    pore_volumes = np.array([100.0, 800.0])  # RT = 1 d (short) and 8 d (long)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volumes,
        retardation_factor=1.0,
        spinup=spinup,
    )

    if spinup is None:
        # The long streamtube needs ≥ 8 days of cin history before its
        # source window is fully inside cin; cout bins 0..7 are invalid.
        expected_nan = np.zeros(n_cin, dtype=bool)
        expected_nan[:8] = True
        np.testing.assert_array_equal(np.isnan(cout), expected_nan)

        # Independent reference for valid bins: arithmetic mean of per-
        # streamtube outputs against the original cin (no padding).
        ref_short = _per_streamtube_constant_flow_reference(
            cin=cin, flow_rate=flow_rate, pore_volume=pore_volumes[0], n_cout=n_cin
        )
        ref_long = _per_streamtube_constant_flow_reference(
            cin=cin, flow_rate=flow_rate, pore_volume=pore_volumes[1], n_cout=n_cin
        )
        reference = 0.5 * (ref_short + ref_long)
        valid = ~np.isnan(cout)
        np.testing.assert_allclose(cout[valid], reference[valid], atol=1e-12)
    else:
        # spinup="constant" pads the left edge with cin[0] for the warm-
        # start duration ceil(R · V_max / flow[0]) + 1 = 9 days. Reference:
        # extend cin and flow on the left by that many days with the
        # warm-start values; per-streamtube overlap reference becomes
        # valid for every cout bin.
        assert not np.any(np.isnan(cout))
        n_pad = int(np.ceil(1.0 * pore_volumes[-1] / flow_rate)) + 1
        cin_padded = np.concatenate([np.full(n_pad, cin[0]), cin])
        ref_short = _per_streamtube_constant_flow_reference(
            cin=cin_padded,
            flow_rate=flow_rate,
            pore_volume=pore_volumes[0],
            n_cout=n_cin,
            cin_offset_days=-float(n_pad),
        )
        ref_long = _per_streamtube_constant_flow_reference(
            cin=cin_padded,
            flow_rate=flow_rate,
            pore_volume=pore_volumes[1],
            n_cout=n_cin,
            cin_offset_days=-float(n_pad),
        )
        reference = 0.5 * (ref_short + ref_long)
        np.testing.assert_allclose(cout, reference, atol=1e-12)


def test_weight_matrix_rows_sum_to_one_or_zero():
    """Each row of W either sums to 1 (valid output bin) or to 0 (invalid).

    Mass-conservation invariant of the bundle weight matrix under strict
    validity. Catches mutations that double or halve row normalization
    (mutations that the W*cin cross-check would mask)."""
    n_cin = 40
    tedges = pd.date_range("2020-01-01", periods=n_cin + 1, freq="D")
    cout_tedges = tedges
    flow = 100.0 + 25.0 * np.sin(np.arange(n_cin) * 2 * np.pi / 9)
    aquifer_pore_volumes = np.array([150.0, 350.0, 600.0])

    weight_tedges, weight_flow, _, threshold, _ = _resolve_spinup_inputs(
        None,
        tedges=tedges,
        flow=flow,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=1.0,
    )
    acc, contrib, zero_flow = _infiltration_to_extraction_weights(
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=weight_flow,
        retardation_factor=1.0,
    )
    w, invalid = _resolve_spinup_mask(
        accumulated_weights=acc,
        contributing_bins=contrib,
        zero_flow_cout=zero_flow,
        n_pv=len(aquifer_pore_volumes),
        spinup=threshold,
    )

    row_sums = w.sum(axis=1)
    valid_rows = ~invalid
    np.testing.assert_allclose(row_sums[valid_rows], 1.0, atol=1e-13)
    np.testing.assert_allclose(row_sums[invalid], 0.0, atol=1e-13)


# =============================================================================
# Validator helper: parametrized snapshot pinning every ValueError branch of
# _validate_advection_inputs. Verbatim messages from the prior duplicated
# prologues; new branches (flow>=0 in rev) are the issue #187 omission fix.
# =============================================================================


def _good_advection_inputs():
    """Baseline good-input dict that passes _validate_advection_inputs silently."""
    n = 5
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    return {
        "tedges": tedges,
        "flow": np.full(n, 100.0),
        "retardation_factor": 1.0,
    }


def test_advection_public_rejects_nan_retardation_factor():
    """NaN retardation slipped past ``< 1.0`` and silently produced all-NaN output; must raise."""
    n = 5
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    with pytest.raises(ValueError, match=r"retardation_factor must be >= 1\.0"):
        infiltration_to_extraction(
            cin=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([500.0]),
            retardation_factor=np.nan,
        )


def test_validate_advection_inputs_silent_on_good_input_forward():
    """No exception when the forward (cin) inputs are valid."""
    kwargs = _good_advection_inputs()
    n = len(kwargs["flow"])
    _validate_advection_inputs(**kwargs, cin_values=np.ones(n))


def test_validate_advection_inputs_silent_on_good_input_reverse():
    """No exception when the reverse (cout + cout_tedges) inputs are valid."""
    kwargs = _good_advection_inputs()
    n = len(kwargs["flow"])
    _validate_advection_inputs(**kwargs, cout_values=np.ones(n), cout_tedges=kwargs["tedges"])


@pytest.mark.parametrize(
    ("path", "mutate", "match_regex"),
    [
        # ---------------- forward path ----------------
        (
            "forward",
            lambda k: {**k, "cin_values": np.ones(len(k["flow"]) + 1)},
            r"tedges must have one more element than cin",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "cin_values": np.ones(len(k["flow"])),
                "flow": np.full(len(k["flow"]) + 1, 100.0),
            },
            r"tedges must have one more element than flow",
        ),
        (
            "forward",
            lambda k: {**k, "cin_values": np.array([1.0, np.nan, 1.0, 1.0, 1.0])},
            r"cin contains NaN values, which are not allowed",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "flow": np.array([100.0, np.nan, 100.0, 100.0, 100.0]),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"flow contains NaN values, which are not allowed",
        ),
        (
            "forward",
            lambda k: {
                **k,
                "flow": np.array([100.0, -50.0, 100.0, 100.0, 100.0]),
                "cin_values": np.ones(len(k["flow"])),
            },
            r"flow must be non-negative \(negative flow not supported\)",
        ),
        (
            "forward",
            lambda k: {**k, "retardation_factor": 0.5, "cin_values": np.ones(len(k["flow"]))},
            r"retardation_factor must be >= 1\.0",
        ),
        # ---------------- reverse path ----------------
        (
            "reverse",
            lambda k: {
                **k,
                "flow": np.full(len(k["flow"]) + 1, 100.0),
                "cout_values": np.ones(len(k["flow"])),
                "cout_tedges": k["tedges"],
            },
            r"tedges must have one more element than flow",
        ),
        (
            "reverse",
            lambda k: {
                **k,
                "cout_values": np.ones(len(k["flow"]) + 1),
                "cout_tedges": k["tedges"],
            },
            r"cout_tedges must have one more element than cout",
        ),
        (
            "reverse",
            lambda k: {
                **k,
                "cout_values": np.array([1.0, np.nan, 1.0, 1.0, 1.0]),
                "cout_tedges": k["tedges"],
            },
            r"cout contains NaN values, which are not allowed",
        ),
        # NEW: flow >= 0 in reverse (omission fix; symmetric with diffusion reverse)
        (
            "reverse",
            lambda k: {
                **k,
                "flow": np.array([100.0, -50.0, 100.0, 100.0, 100.0]),
                "cout_values": np.ones(len(k["flow"])),
                "cout_tedges": k["tedges"],
            },
            r"flow must be non-negative \(negative flow not supported\)",
        ),
        (
            "reverse",
            lambda k: {
                **k,
                "retardation_factor": 0.5,
                "cout_values": np.ones(len(k["flow"])),
                "cout_tedges": k["tedges"],
            },
            r"retardation_factor must be >= 1\.0",
        ),
    ],
)
def test_validate_advection_inputs_raises_on_bad_input(path, mutate, match_regex):
    """Each ValueError branch raises with the exact historical message string."""
    del path
    bad = mutate(_good_advection_inputs())
    with pytest.raises(ValueError, match=match_regex):
        _validate_advection_inputs(**bad)


# =============================================================================
# Fast banded builder: dense + rational oracles and the exactness suite.
#
# The package builder (advection_utils._infiltration_to_extraction_weights)
# performs the dense build's exact per-streamtube time-domain arithmetic but
# only on the nonzero band (the source window overlaps a few cin bins), so it is
# machine-precision identical to the dense build at O(n_pv * n_cout * band)
# instead of O(n_pv * n_cout * n_cin). Two independent oracles pin it:
#
#   _dense_weights_reference -- the relocated O(n_pv * N^2) dense full-overlap
#       build (inverse cumulative-volume map + partial_isin) the banded builder replaced. The
#       builder reproduces it to float reordering noise (~1e-14); accumulated
#       parity uses atol=1e-12 (two independent float paths), structure exactly.
#   _exact_rational_advection -- fractions.Fraction overlap arithmetic on the
#       cumulative-volume axis, the absolute truth; the builder matches it to
#       ~1e-15 (machine precision) on every grid including sub-bin cout.
#
# Both encode the drop-not-clip contract: a streamtube contributes to a cout bin
# only if its source window is fully inside [Vi[0], Vi[-1]].
# =============================================================================


def _dense_weights_reference(*, tedges, cout_tedges, aquifer_pore_volumes, flow, retardation_factor):
    """Relocated copy of the original dense full-overlap-matrix builder.

    Loops over pore volumes, back-projects the cout edges via the inverse cumulative-volume map
    (no residence-time function), and accumulates per-streamtube flow-normalized ``partial_isin``
    overlap rows. A streamtube whose source window leaves the cin range yields a NaN look-back edge
    -> NaN overlap row -> dropped (not counted, not clipped).
    Returns the same ``(accumulated_weights, contributing_bins, zero_flow_cout)``
    triple as the package builder.
    """
    cin_tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    flow = np.asarray(flow, dtype=float)

    cout_in_cin_overlap = partial_isin(bin_edges_in=cout_tedges_days, bin_edges_out=cin_tedges_days)
    zero_flow_cout = (cout_in_cin_overlap @ flow) == 0

    # Back-project cout edges to infiltration times via the inverse cumulative-volume map, inline and
    # independent of any residence-time function: extraction_to_infiltration look-back is
    # T(V(cout_edge) - R*V_p), NaN where the look-back leaves the record.
    flow_cum = cumulative_flow_volume(flow, np.diff(cin_tedges_days), strictly_monotone=True)
    v_at_cout = linear_interpolate(
        x_ref=cin_tedges_days, y_ref=flow_cum, x_query=cout_tedges_days, left=np.nan, right=np.nan
    )
    a = v_at_cout[None, :] - retardation_factor * np.atleast_1d(aquifer_pore_volumes)[:, None]
    infiltration_tedges_2d = linear_interpolate(
        x_ref=flow_cum, y_ref=cin_tedges_days, x_query=a, left=np.nan, right=np.nan
    )
    valid_bins_2d = ~(np.isnan(infiltration_tedges_2d[:, :-1]) | np.isnan(infiltration_tedges_2d[:, 1:]))
    cin_min, cin_max = cin_tedges_days[0], cin_tedges_days[-1]

    n_pv = len(aquifer_pore_volumes)
    n_cout = len(cout_tedges) - 1
    n_cin = len(tedges) - 1
    accumulated = np.zeros((n_cout, n_cin))
    contributing = np.zeros(n_cout, dtype=np.intp)

    for i in range(n_pv):
        if not np.any(valid_bins_2d[i, :]):
            continue
        infil = infiltration_tedges_2d[i, :]
        valid_times = infil[~np.isnan(infil)]
        if len(valid_times) == 0 or not (max(valid_times[0], cin_min) < min(valid_times[-1], cin_max)):
            continue
        overlap_matrix = partial_isin(bin_edges_in=infil, bin_edges_out=cin_tedges_days)
        flow_weighted = overlap_matrix * flow[None, :]
        row_sums = flow_weighted.sum(axis=1)
        valid = row_sums > 0
        normalized = np.zeros_like(flow_weighted)
        normalized[valid, :] = flow_weighted[valid, :] / row_sums[valid, None]
        accumulated[valid, :] += normalized[valid, :]
        contributing[valid] += 1

    return accumulated, contributing, zero_flow_cout


def _exact_rational_advection(*, tedges, cout_tedges, aquifer_pore_volumes, flow, retardation_factor):
    """Exact Fraction overlap arithmetic on the cumulative-volume axis.

    Independent of both the package and the dense reference. Encodes the
    drop-not-clip contract: only streamtubes whose source window lies fully
    inside ``[Vi[0], Vi[-1]]`` contribute. Returns ``(accumulated_weights,
    contributing_bins)`` as float64 arrays converted from exact Fractions.
    """
    cin_days = [Fraction(x) for x in tedges_to_days(tedges)]
    cout_days = [Fraction(x) for x in tedges_to_days(cout_tedges, ref=tedges[0])]
    flow_f = [Fraction(x) for x in np.asarray(flow, dtype=float)]
    n_cin = len(cin_days) - 1
    n_cout = len(cout_days) - 1
    r = sorted(Fraction(retardation_factor) * Fraction(v) for v in np.asarray(aquifer_pore_volumes, dtype=float))

    vi = [Fraction(0)]
    for j in range(n_cin):
        vi.append(vi[-1] + flow_f[j] * (cin_days[j + 1] - cin_days[j]))
    vi_lo, vi_hi = vi[0], vi[-1]

    def interp_v(t):
        if t <= cin_days[0]:
            return vi[0]
        if t >= cin_days[-1]:
            return vi[-1]
        j = 0
        while cin_days[j + 1] < t:
            j += 1
        width = cin_days[j + 1] - cin_days[j]
        if width == 0:
            return vi[j]
        return vi[j] + (t - cin_days[j]) / width * (vi[j + 1] - vi[j])

    vc = [interp_v(t) for t in cout_days]
    edge_in_range = [cin_days[0] <= t <= cin_days[-1] for t in cout_days]

    accumulated = np.zeros((n_cout, n_cin))
    contributing = np.zeros(n_cout, dtype=np.intp)
    for i in range(n_cout):
        lo_v, hi_v = vc[i], vc[i + 1]
        w = hi_v - lo_v
        if w == 0 or not (edge_in_range[i] and edge_in_range[i + 1]):
            continue
        row = [Fraction(0)] * n_cin
        ncontrib = 0
        for rp in r:
            a, b = lo_v - rp, hi_v - rp
            if a < vi_lo or b > vi_hi:  # window not fully contained -> dropped
                continue
            ncontrib += 1
            for m in range(n_cin):
                overlap = min(b, vi[m + 1]) - max(a, vi[m])
                if overlap > 0:
                    row[m] += overlap / w
        contributing[i] = ncontrib
        accumulated[i, :] = [float(x) for x in row]
    return accumulated, contributing


def _forward_via(builder, *, cin, flow, tedges, cout_tedges, aquifer_pore_volumes, retardation_factor, spinup):
    """Drive ``builder`` through the public forward spin-up pipeline.

    Mirrors :func:`infiltration_to_extraction` exactly but lets the weight
    builder be swapped (package builder vs dense oracle) so the two can be
    compared through identical spin-up resolution.
    """
    weight_tedges, weight_flow, weight_cin, threshold, _ = _resolve_spinup_inputs(
        spinup,
        tedges=tedges,
        flow=flow,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        cin=cin,
    )
    acc, contrib, zero_flow = builder(
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=weight_flow,
        retardation_factor=retardation_factor,
    )
    w, invalid = _resolve_spinup_mask(
        accumulated_weights=acc,
        contributing_bins=contrib,
        zero_flow_cout=zero_flow,
        n_pv=len(aquifer_pore_volumes),
        spinup=threshold,
    )
    assert weight_cin is not None  # cin is always supplied in these tests
    out = w.dot(weight_cin)
    out[invalid] = np.nan
    return out, contrib


_BANDED_APVD = {
    "single": np.array([500.0]),
    "bimodal": np.array([120.0, 130.0, 900.0, 950.0]),
    "uniform_wide": np.linspace(50.0, 800.0, 12),
    "heavy_tail": np.array([50.0, 60.0, 70.0, 2000.0]),
    "gamma": gamma.bins(mean=300.0, std=120.0, n_bins=24)["expected_values"],
}


def _banded_tedges(n, kind, seed):
    """Return tedges for a regular or irregular cin grid of n bins."""
    if kind == "regular":
        return pd.date_range("2020-01-01", periods=n + 1, freq="D")
    rng = np.random.default_rng(seed)
    widths = 0.3 + rng.uniform(0.0, 1.7, n)
    return pd.Timestamp("2020-01-01") + pd.to_timedelta(np.concatenate([[0.0], np.cumsum(widths)]), unit="D")


def _banded_cout(tedges, kind):
    """Return cout_tedges: aligned, coarse (every 3rd edge), or fine (each cin
    bin split into 3 equal sub-bins -- irregular when ``tedges`` are irregular).

    The fine grid is a clean equal subdivision (no random near-coincident edges)
    so its smallest bin volume is bounded; pathologically tiny cout bins are
    exercised separately by the rational-oracle tests at small cumulative volume.
    """
    if kind == "same":
        return tedges
    if kind == "coarse":
        return tedges[::3]
    d = tedges_to_days(tedges)
    pts = np.unique(np.concatenate([np.linspace(d[i], d[i + 1], 4) for i in range(len(d) - 1)]))
    return tedges[0] + pd.to_timedelta(pts, unit="D")


@pytest.mark.parametrize("apv_name", list(_BANDED_APVD))
def test_banded_builder_triple_parity_vs_dense(apv_name):
    """Banded builder reproduces the dense oracle's full triple across
    {APVD shape} x {const, CV 0.1/0.3/0.6 flow} x {regular, irregular tedges}
    x {cout=tedges, coarse, fine} x {R = 1, 2.7}.

    contributing_bins and zero_flow_cout match *exactly* (integer / bool);
    accumulated_weights matches to atol=1e-12 -- the builder runs the dense
    build's arithmetic on the nonzero band, so the only difference is float
    reordering between the two paths (np.interp vs the inverse cumulative-volume
    map + partial_isin). Sub-bin cout exactness is pinned tighter against the rational
    oracle in test_banded_builder_exact_vs_rational."""
    apv = _BANDED_APVD[apv_name]
    n = 60
    seed0 = list(_BANDED_APVD).index(apv_name) * 100
    flows = {
        "const": np.full(n, 100.0),
        "cv0.1": 100.0 * (1 + 0.1 * np.sin(np.arange(n) * 2 * np.pi / 17)),
        "cv0.3": 100.0 * (1 + 0.3 * np.sin(np.arange(n) * 2 * np.pi / 17)),
        "cv0.6": 100.0 * (1 + 0.6 * np.sin(np.arange(n) * 2 * np.pi / 13)),
    }
    for tk in ("regular", "irregular"):
        tedges = _banded_tedges(n, tk, seed0)
        for fname, flow in flows.items():
            for ck in ("same", "coarse", "fine"):
                cout = _banded_cout(tedges, ck)
                for r_factor in (1.0, 2.7):
                    kw = {
                        "tedges": tedges,
                        "cout_tedges": cout,
                        "aquifer_pore_volumes": apv,
                        "flow": flow,
                        "retardation_factor": r_factor,
                    }
                    acc, contrib, zf = _infiltration_to_extraction_weights(**kw)
                    acc_d, contrib_d, zf_d = _dense_weights_reference(**kw)
                    tag = f"{apv_name}/{tk}/{fname}/{ck}/R{r_factor}"
                    np.testing.assert_array_equal(contrib, contrib_d, err_msg=f"contributing {tag}")
                    np.testing.assert_array_equal(zf, zf_d, err_msg=f"zero_flow {tag}")
                    # atol=1e-13: observed worst is ~3e-14 (two independent float paths);
                    # tighter exactness is pinned by the rational oracle (atol=1e-14).
                    np.testing.assert_allclose(acc, acc_d, atol=1e-13, rtol=0, err_msg=f"accumulated {tag}")


def test_banded_builder_boundary_touch_contributing():
    """A back-projected window edge landing exactly on Vi[0]/Vi[-1] counts as
    contained (inclusive), pinning the searchsorted '<' vs '<=' sides.

    Constant flow 100 m3/d, daily bins => Vi = [0, 100, 200, ...]. With apv
    chosen so r = R*PV is an exact multiple of 100, several cout bins have a
    source-window edge exactly on a data boundary. The banded contributing
    count must match the dense oracle bit-for-bit at those touches."""
    n = 30
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    # r in {100, 500, 1500} m3 -> window edges land exactly on Vi multiples.
    apv = np.array([100.0, 500.0, 1500.0])
    for r_factor in (1.0, 2.0):
        kw = {
            "tedges": tedges,
            "cout_tedges": tedges,
            "aquifer_pore_volumes": apv,
            "flow": flow,
            "retardation_factor": r_factor,
        }
        _, contrib, _ = _infiltration_to_extraction_weights(**kw)
        _, contrib_d, _ = _dense_weights_reference(**kw)
        _, contrib_r = _exact_rational_advection(**kw)
        np.testing.assert_array_equal(contrib, contrib_d)
        np.testing.assert_array_equal(contrib, contrib_r)


# Small-volume cases where float64 reproduces the exact Fraction truth to a few
# ULP. atol=1e-14 is the machine-precision floor at these volume scales
# (ulp of a ~10 m3 cumulative volume is ~1.8e-15); it is NOT a loosened
# tolerance -- a real overlap/drop bug produces O(0.01-1) errors.
_RATIONAL_CASES = {
    "bimodal_const": {"flow": np.full(8, 1.0), "apv": np.array([2.0, 3.0, 5.0]), "R": 1.0, "cout": "same"},
    "single_noninteger_rt": {"flow": np.full(10, 1.0), "apv": np.array([2.7]), "R": 1.0, "cout": "same"},
    # general fringe: variable flow, irregular cout, non-integer residence time
    "variable_flow_fringe": {
        "flow": 1.0 + 0.4 * np.sin(np.arange(10)),
        "apv": np.array([1.3, 2.7, 4.1]),
        "R": 1.4,
        "cout": "fine",
    },
    "retardation": {"flow": np.full(9, 2.0), "apv": np.array([1.5, 4.0]), "R": 2.3, "cout": "coarse"},
}


@pytest.mark.parametrize("case_name", list(_RATIONAL_CASES))
def test_banded_builder_exact_vs_rational(case_name):
    """Banded accumulated_weights equals the exact Fraction truth to machine
    precision (incl. a general fringe case: variable flow + irregular cout +
    non-integer residence time). Weights are cin-independent, so exactness of
    the operator implies exactness of every cout = W @ cin."""
    case = _RATIONAL_CASES[case_name]
    flow = np.asarray(case["flow"], dtype=float)
    n = len(flow)
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    if case["cout"] == "same":
        cout = tedges
    elif case["cout"] == "coarse":
        cout = tedges[::2]
    else:
        days = np.sort(np.unique(np.concatenate([np.linspace(0.0, n, 3 * n), [0.0, float(n)]])))
        cout = tedges[0] + pd.to_timedelta(days, unit="D")
    kw = {
        "tedges": tedges,
        "cout_tedges": cout,
        "aquifer_pore_volumes": case["apv"],
        "flow": flow,
        "retardation_factor": case["R"],
    }
    acc, contrib, _ = _infiltration_to_extraction_weights(**kw)
    acc_r, contrib_r = _exact_rational_advection(**kw)
    np.testing.assert_array_equal(contrib, contrib_r)
    np.testing.assert_allclose(acc, acc_r, atol=1e-14, rtol=0)


def test_banded_builder_degenerate_zero_width_broken_through_bin():
    """A near-zero-width (w ~ 0.01 m3) but fully broken-through cout bin is exact.

    The weight is overlap/w with overlap ~ w, so the division is well
    conditioned and the row still sums to the contributing count. Exercises the
    w != 0 path with a pathological width against the exact rational oracle."""
    n = 20
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)  # 100 m3 per daily bin
    apv = np.array([300.0, 700.0])  # RTs 3 d and 7 d -> broken through by day 7
    # cout: one ordinary bin then a 0.0001-day (=0.01 m3) sliver, deep in the
    # broken-through interior (day ~12).
    cout_days = np.array([10.0, 12.0, 12.0001, 15.0])
    cout = tedges[0] + pd.to_timedelta(cout_days, unit="D")
    kw = {
        "tedges": tedges,
        "cout_tedges": cout,
        "aquifer_pore_volumes": apv,
        "flow": flow,
        "retardation_factor": 1.0,
    }
    acc, contrib, zf = _infiltration_to_extraction_weights(**kw)
    acc_r, contrib_r = _exact_rational_advection(**kw)
    assert contrib[1] == len(apv)  # sliver bin fully broken through
    assert not zf[1]
    np.testing.assert_array_equal(contrib, contrib_r)
    np.testing.assert_allclose(acc, acc_r, atol=1e-14, rtol=0)
    # The sliver row, normalized by its contributing count, sums to 1.
    np.testing.assert_allclose(acc[1].sum() / contrib[1], 1.0, atol=1e-13)


@pytest.mark.parametrize("spinup", [None, "constant", 0.5])
def test_banded_builder_public_forward_parity_spinup_modes(spinup):
    """Public forward output matches the dense-oracle-driven forward for all
    three spin-up modes, including bit-identical NaN masks. For spinup=0.5 the
    fringe (0 < contributing < n_pv on surviving bins) is asserted non-empty, so
    a broken fringe path would change the float-mode output and fail here."""
    n = 50
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 100.0 + 20.0 * np.sin(np.arange(n) * 2 * np.pi / 11)
    cin = 2.0 + np.cos(np.arange(n) * 2 * np.pi / 9)
    apv = np.array([100.0, 500.0, 1500.0])  # widely separated RTs => real fringe

    real_out = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=spinup
    )
    dense_out, contrib = _forward_via(
        _dense_weights_reference,
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        retardation_factor=1.0,
        spinup=spinup,
    )
    np.testing.assert_array_equal(np.isnan(real_out), np.isnan(dense_out))
    valid = ~np.isnan(real_out)
    np.testing.assert_allclose(real_out[valid], dense_out[valid], atol=1e-12, rtol=0)
    if spinup == 0.5:
        assert np.any(valid & (contrib < len(apv))), "fringe path not exercised"


def test_banded_builder_fringe_decircularized_vs_rational():
    """De-circularize the fringe: a surviving fringe bin's spinup=0.5 public
    value equals the exact rational count-mean (W_rational @ cin / contributing),
    not just the relocated dense build."""
    n = 24
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 1.0 + 0.3 * np.sin(np.arange(n) * 2 * np.pi / 7)  # small-volume, variable
    cin = 3.0 + np.cos(np.arange(n) * 2 * np.pi / 5)
    apv = np.array([1.3, 2.7, 4.1])  # non-integer residence times

    out = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=0.5
    )
    acc_r, contrib_r = _exact_rational_advection(
        tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, flow=flow, retardation_factor=1.0
    )
    # Exact count-mean reference for the threshold mode (contributing >= 0.5*n_pv).
    n_pv = len(apv)
    fringe = (contrib_r > 0) & (contrib_r < n_pv) & (contrib_r >= 0.5 * n_pv)
    assert np.any(fringe), "no surviving fringe bin in this setup"
    expected = (acc_r[fringe] @ cin) / contrib_r[fringe]
    np.testing.assert_allclose(out[fringe], expected, atol=1e-13, rtol=0)


def test_banded_builder_sharp_step_and_pulse_exact_vs_rational():
    """Sharp step and pulse cin produce the exact W @ cin in the broken-through
    region (W is cin-independent; this checks the cin-contraction too)."""
    n = 40
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    apv = np.array([200.0, 400.0])  # RTs 2 d, 4 d
    acc_r, contrib_r = _exact_rational_advection(
        tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, flow=flow, retardation_factor=1.0
    )
    n_pv = len(apv)
    broken = contrib_r == n_pv
    step = np.where(np.arange(n) >= 15, 7.0, 1.0)
    pulse = np.zeros(n)
    pulse[20:22] = 5.0
    for cin in (step, pulse):
        out = infiltration_to_extraction(
            cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=None
        )
        expected = acc_r[broken] @ cin / n_pv
        np.testing.assert_allclose(out[broken], expected, atol=1e-13, rtol=0)


def test_banded_builder_cout_outside_range_masked_like_dense():
    """cout bins straddling or outside the cin time range are dropped exactly as
    the dense build drops them (NaN residence-time edge), not clipped."""
    n = 20
    tedges = pd.date_range("2020-01-10", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    apv = np.array([150.0, 350.0])
    # cout extends 5 days before and after the cin range -> straddling + outside.
    cout = pd.date_range("2020-01-05", periods=n + 11, freq="D")
    kw = {
        "tedges": tedges,
        "cout_tedges": cout,
        "aquifer_pore_volumes": apv,
        "flow": flow,
        "retardation_factor": 1.0,
    }
    acc, contrib, zf = _infiltration_to_extraction_weights(**kw)
    acc_d, contrib_d, zf_d = _dense_weights_reference(**kw)
    np.testing.assert_array_equal(contrib, contrib_d)
    np.testing.assert_array_equal(zf, zf_d)
    np.testing.assert_allclose(acc, acc_d, atol=1e-12, rtol=0)


def test_banded_builder_zero_and_tiny_flow_match_dense():
    """Zero-flow series gives an all-zero operator; tiny flow (band spanning the
    whole series) still matches the dense oracle."""
    n = 25
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    apv = np.array([100.0, 300.0])

    zero_kw = {
        "tedges": tedges,
        "cout_tedges": tedges,
        "aquifer_pore_volumes": apv,
        "flow": np.zeros(n),
        "retardation_factor": 1.0,
    }
    acc0, contrib0, zf0 = _infiltration_to_extraction_weights(**zero_kw)
    assert np.count_nonzero(acc0) == 0
    assert np.all(contrib0 == 0)
    assert np.all(zf0)

    # Tiny flow: residence volumes dwarf the per-bin volume, so the band spans
    # (nearly) the whole series -- the builder degrades gracefully to dense.
    tiny_kw = {**zero_kw, "flow": np.full(n, 0.5)}
    acc_t, contrib_t, zf_t = _infiltration_to_extraction_weights(**tiny_kw)
    acc_d, contrib_d, zf_d = _dense_weights_reference(**tiny_kw)
    np.testing.assert_array_equal(contrib_t, contrib_d)
    np.testing.assert_array_equal(zf_t, zf_d)
    np.testing.assert_allclose(acc_t, acc_d, atol=1e-12, rtol=0)


def test_banded_builder_structurally_banded_large_n():
    """The weight operator is structurally banded at large N: nonzeros per cout
    row are bounded by the APVD volume span / per-bin volume, independent of N.
    This is an invariant of the advection operator -- each cout bin draws from
    one narrow source window -- so it guards against a band/gather bug that
    smears weights across too many cin bins, not against the build cost itself."""
    n = 1500
    tedges = pd.date_range("2018-01-01", periods=n + 1, freq="D")
    flow = 100.0 + 30.0 * np.sin(np.arange(n) * 2 * np.pi / 365)
    apv = gamma.bins(mean=250.0, std=90.0, n_bins=60)["expected_values"]
    acc, _, _ = _infiltration_to_extraction_weights(
        tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, flow=flow, retardation_factor=1.0
    )
    # Per-row band bound: source-volume span / smallest per-bin volume + margin
    # for the cout bin width and searchsorted rounding.
    band_bound = int(np.ceil((apv.max() - apv.min()) / flow.min())) + 5
    nnz = np.count_nonzero(acc)
    assert nnz <= n * band_bound, f"nnz={nnz} exceeds banded bound {n * band_bound}"
    assert nnz < 0.05 * n * n, "operator is not sub-dense -- banding failed"


def test_banded_builder_reverse_and_gamma_match_dense_driven():
    """Reverse (extraction_to_infiltration) and gamma wrappers inherit the banded
    builder: their outputs match the same pipeline driven by the dense oracle."""
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 100.0 + 15.0 * np.sin(np.arange(n) * 2 * np.pi / 13)
    apv = np.array([150.0, 400.0, 700.0])
    cin_true = 5.0 + 2.0 * np.sin(2 * np.pi * np.arange(n) / 21.0)

    # Forward with the package builder, then invert with both pipelines.
    cout = infiltration_to_extraction(
        cin=cin_true, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv
    )
    cin_real = extraction_to_infiltration(
        cout=cout, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv
    )

    # Dense-driven reverse: rebuild W_forward with the dense oracle and solve the
    # same Tikhonov system the package uses.
    weight_tedges, weight_flow, _, threshold, n_pad = _resolve_spinup_inputs(
        "constant", tedges=tedges, flow=flow, aquifer_pore_volumes=apv, retardation_factor=1.0
    )
    acc_d, contrib_d, zf_d = _dense_weights_reference(
        tedges=weight_tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, flow=weight_flow, retardation_factor=1.0
    )
    w_d, _ = _resolve_spinup_mask(
        accumulated_weights=acc_d, contributing_bins=contrib_d, zero_flow_cout=zf_d, n_pv=len(apv), spinup=threshold
    )
    cin_dense = solve_inverse_transport(
        w_forward=w_d, observed=cout, n_output=len(weight_tedges) - 1, regularization_strength=1e-10
    )[n_pad:]
    np.testing.assert_allclose(cin_real, cin_dense, atol=1e-9, rtol=0)

    # Gamma wrapper parity: gamma_* delegates to the same forward builder.
    cout_gamma_real = gamma_infiltration_to_extraction(
        cin=cin_true, flow=flow, tedges=tedges, cout_tedges=tedges, mean=300.0, std=120.0, n_bins=12
    )
    apv_gamma = gamma.bins(mean=300.0, std=120.0, n_bins=12)["expected_values"]
    cout_gamma_dense, _ = _forward_via(
        _dense_weights_reference,
        cin=cin_true,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv_gamma,
        retardation_factor=1.0,
        spinup="constant",
    )
    np.testing.assert_array_equal(np.isnan(cout_gamma_real), np.isnan(cout_gamma_dense))
    g_valid = ~np.isnan(cout_gamma_real)
    np.testing.assert_allclose(cout_gamma_real[g_valid], cout_gamma_dense[g_valid], atol=1e-12, rtol=0)


def test_banded_builder_multi_tile_matches_single_tile(monkeypatch):
    """Tiling the cout axis must not change the operator: a many-tile build is
    bit-identical to a single-tile build.

    Variable flow plus a wide APVD make the per-tile band widths differ, so this
    also guards the global ``full_band`` tracking and per-tile right-padding --
    a ``full_band = tile_band`` bug (dropping the cross-tile ``max``) is invisible
    to every single-tile test but corrupts or fails to assemble a multi-tile build.
    """
    n = 300
    tedges = pd.date_range("2018-01-01", periods=n + 1, freq="D")
    flow = np.linspace(20.0, 400.0, n)
    apv = np.array([50.0, 2000.0])
    kw = {
        "tedges": tedges,
        "cout_tedges": tedges,
        "aquifer_pore_volumes": apv,
        "flow": flow,
        "retardation_factor": 1.0,
    }

    monkeypatch.setattr(advection_utils, "_WEIGHT_BUILD_BLOCK", 10**9)  # one tile
    bv1, cs1, cb1, zf1 = _banded_weights(**kw)
    monkeypatch.setattr(advection_utils, "_WEIGHT_BUILD_BLOCK", 21)  # block = 21 // 2 = 10 -> ~30 tiles
    bv_n, cs_n, cb_n, zf_n = _banded_weights(**kw)

    np.testing.assert_array_equal(cs1, cs_n)
    np.testing.assert_array_equal(cb1, cb_n)
    np.testing.assert_array_equal(zf1, zf_n)
    np.testing.assert_array_equal(bv1, bv_n)  # same shape (global full_band) and values


def test_solve_inverse_transport_banded_matches_dense():
    """The banded CSNE solver reproduces the dense least-squares solver to the
    documented ~1e-9, isolated from the forward build.

    The corrected-semi-normal refinement is load-bearing: with
    ``_BANDED_REFINEMENT_STEPS = 0`` the under-determined (spin-up) directions
    lose ~1e-6 to the squared condition number, so this comparison pins it.
    """
    n = 80
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 100.0 + 20.0 * np.sin(np.arange(n) * 2 * np.pi / 11)
    apv = np.array([200.0, 450.0, 700.0])
    weight_tedges, weight_flow, _, threshold, _ = _resolve_spinup_inputs(
        "constant", tedges=tedges, flow=flow, aquifer_pore_volumes=apv, retardation_factor=1.0
    )
    n_out = len(weight_tedges) - 1
    band_vals, col_start, contrib, zero_flow = _banded_weights(
        tedges=weight_tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, flow=weight_flow, retardation_factor=1.0
    )
    band_vals, col_start, _ = _banded_mask(
        band_vals=band_vals,
        col_start=col_start,
        contributing_bins=contrib,
        zero_flow_cout=zero_flow,
        n_pv=len(apv),
        spinup=threshold,
    )
    w_dense = _densify_weights(band_vals, col_start, n_out)
    observed = 3.0 + np.sin(np.arange(n) * 2 * np.pi / 17)
    lam = 1e-10
    x_banded = solve_inverse_transport_banded(
        band_vals=band_vals, col_start=col_start, observed=observed, n_output=n_out, regularization_strength=lam
    )
    x_dense = solve_inverse_transport(w_forward=w_dense, observed=observed, n_output=n_out, regularization_strength=lam)
    np.testing.assert_array_equal(np.isnan(x_banded), np.isnan(x_dense))
    valid = ~np.isnan(x_banded)
    np.testing.assert_allclose(x_banded[valid], x_dense[valid], atol=1e-9, rtol=0)


def test_solve_inverse_transport_banded_degenerate_cases():
    """Inactive output columns return NaN, an all-zero operator returns all-NaN
    without raising, and non-positive regularization is rejected."""
    band_vals = np.array([[0.6, 0.4], [0.5, 0.5], [0.0, 0.0]])
    col_start = np.array([0, 2, 0], dtype=np.intp)  # contributes to cols {0,1,2,3}; cols 4,5 inactive
    observed = np.array([1.0, 2.0, 0.0])
    n_output = 6
    out = solve_inverse_transport_banded(
        band_vals=band_vals, col_start=col_start, observed=observed, n_output=n_output, regularization_strength=1e-8
    )
    col_sum = _densify_weights(band_vals, col_start, n_output).sum(axis=0)
    np.testing.assert_array_equal(np.isnan(out), col_sum == 0)

    out_zero = solve_inverse_transport_banded(
        band_vals=np.zeros((3, 2)),
        col_start=np.zeros(3, dtype=np.intp),
        observed=np.ones(3),
        n_output=4,
        regularization_strength=1e-8,
    )
    assert np.all(np.isnan(out_zero))

    with pytest.raises(ValueError, match="must be > 0"):
        solve_inverse_transport_banded(
            band_vals=band_vals, col_start=col_start, observed=observed, n_output=n_output, regularization_strength=0.0
        )


def test_advection_empty_pore_volumes_all_nan():
    """An empty pore-volume distribution means no transport: both directions
    return all-NaN through the public API without raising."""
    n = 20
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    empty = np.array([])
    cout = infiltration_to_extraction(
        cin=np.ones(n), flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=empty
    )
    assert np.all(np.isnan(cout))
    cin_rec = extraction_to_infiltration(
        cout=np.ones(n), flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=empty
    )
    assert np.all(np.isnan(cin_rec))


# =============================================================================
# Regression: nonlinear-sorption output wrapper zero-flow / left-edge handling
# (review item A1 / M3). Baseline (pre-fix) raised ValueError "Invalid θ-bin"
# because a zero-flow input span collapses θ-edges and a cout edge left of the
# flow record clamps to θ=0; both produce a zero-width θ sub-bin. The linear
# sibling handles both gracefully, so it is the physics oracle here (front
# tracking with ConstantRetardation == linear advection at the same R).
# =============================================================================


def test_nonlinear_sorption_interior_zero_flow_coarse_cout_matches_linear():
    """A1(a): an interior zero-flow input bin must not crash the front-tracking
    output wrapper. With a coarse (3-day) cout grid the zero-flow day is absorbed
    into a bin with positive throughflow, so every output bin is finite and equals
    the linear sibling to machine precision."""
    n = 20
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.zeros(n)
    cin[2:5] = 10.0
    flow = np.full(n, 100.0)
    flow[10] = 0.0  # single interior zero-flow bin (validator allows flow == 0)
    cout_tedges = pd.date_range("2020-01-01", periods=7, freq="3D")  # fully inside the flow record
    apv = np.array([300.0])

    cout_nl, _ = infiltration_to_extraction_nonlinear_sorption(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_volumes=apv, retardation_factor=2.0
    )
    cout_lin = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_volumes=apv, retardation_factor=2.0
    )

    np.testing.assert_array_equal(np.isnan(cout_nl), np.isnan(cout_lin))
    assert not np.any(np.isnan(cout_nl))  # zero-flow day absorbed -> no undefined bin
    np.testing.assert_allclose(cout_nl, cout_lin, atol=1e-13)


def test_nonlinear_sorption_zero_throughflow_output_bin_is_nan():
    """A1(a): when the cout grid isolates the zero-flow input bin (daily cout
    aligned with the flow grid), that output bin has zero throughflow and an
    undefined flow-weighted average. It must be returned as NaN -- exactly where
    the linear sibling is NaN -- while every other bin matches to machine
    precision."""
    n = 20
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cin = np.zeros(n)
    cin[2:5] = 10.0
    flow = np.full(n, 100.0)
    flow[10] = 0.0
    apv = np.array([300.0])

    cout_nl, _ = infiltration_to_extraction_nonlinear_sorption(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, retardation_factor=2.0
    )
    cout_lin = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, retardation_factor=2.0
    )

    assert np.isnan(cout_nl[10])  # the isolated zero-throughflow output bin
    np.testing.assert_array_equal(np.isnan(cout_nl), np.isnan(cout_lin))
    finite = ~np.isnan(cout_lin)
    np.testing.assert_allclose(cout_nl[finite], cout_lin[finite], atol=1e-13)


def test_nonlinear_sorption_cout_before_record_returns_zero_left_bins():
    """A1(b): a cout grid starting before tedges[0] must not crash. The left
    underflow fixup extrapolates the θ map at flow[0] (mirroring the right-edge
    rule), so pre-record output bins read θ <= 0 -> mass 0 -> exactly 0.0 (the
    documented out-of-range contract), and the in-record window matches the linear
    sibling to machine precision."""
    n = 20
    tedges = pd.date_range("2020-01-10", periods=n + 1, freq="D")
    cin = np.zeros(n)
    cin[2:5] = 10.0
    flow = np.full(n, 100.0)
    apv = np.array([300.0])
    cout_tedges = pd.date_range("2020-01-07", periods=27, freq="D")  # starts 3 days before tedges[0]

    cout_nl, _ = infiltration_to_extraction_nonlinear_sorption(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_volumes=apv, retardation_factor=2.0
    )
    cout_lin = infiltration_to_extraction(
        cin=cin, flow=flow, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_volumes=apv, retardation_factor=2.0
    )

    assert not np.any(np.isnan(cout_nl))  # positive flow throughout -> out-of-range reads as 0
    # Bins wholly before the flow record (2020-01-07..10) are exactly 0.0.
    np.testing.assert_array_equal(cout_nl[:3], np.zeros(3))
    # Where the linear sibling is defined, the breakthrough pulse matches exactly.
    finite = ~np.isnan(cout_lin)
    np.testing.assert_allclose(cout_nl[finite], cout_lin[finite], atol=1e-13)


def test_infiltration_to_extraction_negative_pore_volume_raises():
    """A2: a negative aquifer pore volume back-projects a cout bin to future
    infiltration (anti-causal). The linear forward path must reject it, matching
    the nonlinear and diffusion paths, rather than returning a silent wrong value."""
    n = 10
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    with pytest.raises(ValueError, match="aquifer_pore_volumes must be positive"):
        infiltration_to_extraction(
            cin=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([-300.0]),
        )


def test_extraction_to_infiltration_negative_pore_volume_raises():
    """A2: the reverse (deconvolution) path must reject a negative pore volume too."""
    n = 10
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    with pytest.raises(ValueError, match="aquifer_pore_volumes must be positive"):
        extraction_to_infiltration(
            cout=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([-300.0]),
        )


# Issue #309: an interior zero-flow (pump-off) bin must be transport-invisible.
# Oracle: physically deleting the zero-flow bins from the record leaves the
# θ-history (cumulative flow) of every water-carrying bin unchanged, and the
# front-tracking transport depends on θ only — so cout of the water-carrying
# bins must match the gap-free run bin-for-bin.
_PUMP_OFF_SORPTION_KWARGS = [
    pytest.param({"retardation_factor": 2.0}, id="constant-retardation"),
    pytest.param(
        {"freundlich_k": 0.01, "freundlich_n": 2.0, "bulk_density": 1500.0, "porosity": 0.3},
        id="freundlich",
    ),
    pytest.param(
        {"langmuir_s_max": 1.0, "langmuir_k_l": 1.0, "bulk_density": 1500.0, "porosity": 0.3},
        id="langmuir",
    ),
]


@pytest.mark.parametrize("c_gap", [5.0, 0.0], ids=["nonzero-gap-cin", "zero-gap-cin"])
@pytest.mark.parametrize("sorption_kwargs", _PUMP_OFF_SORPTION_KWARGS)
def test_nonlinear_sorption_interior_zero_flow_gap_matches_deleted_gap(sorption_kwargs, c_gap):
    """Interior pump-off bins (flow=0) must not alter cout of the water-carrying bins."""
    n = 40
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    gap = slice(20, 24)
    flow[gap] = 0.0
    cin = np.full(n, 8.0)
    cin[gap] = c_gap
    cin[24:] = 3.0

    keep = np.ones(n, dtype=bool)
    keep[gap] = False
    tedges_nogap = pd.date_range("2020-01-01", periods=int(keep.sum()) + 1, freq="D")

    cout_gap, _ = infiltration_to_extraction_nonlinear_sorption(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=[200.0],
        **sorption_kwargs,
    )
    cout_nogap, _ = infiltration_to_extraction_nonlinear_sorption(
        cin=cin[keep],
        flow=flow[keep],
        tedges=tedges_nogap,
        cout_tedges=tedges_nogap,
        aquifer_pore_volumes=[200.0],
        **sorption_kwargs,
    )

    # The kept bins of the gap run cover exactly the same θ-intervals as the
    # gap-free run, so the flow-weighted bin averages must agree to roundoff.
    np.testing.assert_allclose(cout_gap[keep], cout_nogap, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("c_gap", [5.0, 0.0], ids=["nonzero-gap-cin", "zero-gap-cin"])
@pytest.mark.parametrize(
    "sorption",
    [
        pytest.param(ConstantRetardation(retardation_factor=2.0), id="constant-retardation"),
        pytest.param(FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3), id="freundlich"),
        pytest.param(LangmuirSorption(s_max=1.0, k_l=1.0, bulk_density=1500.0, porosity=0.3), id="langmuir"),
    ],
)
def test_fronttracking_domain_mass_interior_zero_flow_gap_matches_deleted_gap(sorption, c_gap):
    """Stored mass in the domain must be identical with and without the pump-off gap."""
    n = 40
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    gap = slice(20, 24)
    flow[gap] = 0.0
    cin = np.full(n, 8.0)
    cin[gap] = c_gap
    cin[24:] = 3.0

    keep = np.ones(n, dtype=bool)
    keep[gap] = False
    tedges_nogap = pd.date_range("2020-01-01", periods=int(keep.sum()) + 1, freq="D")

    aquifer_pore_volume = 200.0
    tracker_gap = FrontTracker(
        cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=aquifer_pore_volume, sorption=sorption
    )
    tracker_gap.run()
    tracker_nogap = FrontTracker(
        cin=cin[keep],
        flow=flow[keep],
        tedges=tedges_nogap,
        aquifer_pore_volume=aquifer_pore_volume,
        sorption=sorption,
    )
    tracker_nogap.run()

    for theta in [500.0, 1500.0, 2100.0, 3000.0, 3600.0]:
        mass_gap = compute_domain_mass(
            theta=theta, v_outlet=aquifer_pore_volume, waves=tracker_gap.state.waves, sorption=sorption
        )
        mass_nogap = compute_domain_mass(
            theta=theta, v_outlet=aquifer_pore_volume, waves=tracker_nogap.state.waves, sorption=sorption
        )
        np.testing.assert_allclose(mass_gap, mass_nogap, rtol=0.0, atol=1e-9)
def test_infiltration_to_extraction_non_monotonic_tedges_raises():
    """#313 ADV-P2: the docstring promises a ValueError for non-monotonic time edges.

    Without the check, non-monotonic tedges silently corrupt the cumulative-volume
    mapping and produce wrong output.
    """
    n = 10
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D").to_numpy().copy()
    tedges[3], tedges[4] = tedges[4], tedges[3]  # swap two interior edges
    with pytest.raises(ValueError, match="strictly increasing"):
        infiltration_to_extraction(
            cin=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=pd.DatetimeIndex(tedges),
            cout_tedges=pd.date_range("2020-01-01", periods=n + 1, freq="D"),
            aquifer_pore_volumes=np.array([300.0]),
        )


def test_extraction_to_infiltration_non_monotonic_tedges_raises():
    """#313 ADV-P2: the reverse (deconvolution) path must reject non-monotonic tedges too."""
    n = 10
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D").to_numpy().copy()
    tedges[3], tedges[4] = tedges[4], tedges[3]
    with pytest.raises(ValueError, match="strictly increasing"):
        extraction_to_infiltration(
            cout=np.ones(n),
            flow=np.full(n, 100.0),
            tedges=pd.DatetimeIndex(tedges),
            cout_tedges=pd.date_range("2020-01-01", periods=n + 1, freq="D"),
            aquifer_pore_volumes=np.array([300.0]),
        )
