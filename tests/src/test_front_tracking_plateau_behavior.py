"""
Integration tests for front-tracking plateau behavior.

This module tests that outlet concentrations ultimately plateau at the final
inlet concentration for various wave types and sorption conditions.

Tests cover:
- Favorable sorption (n>1): shocks from increases, rarefactions from decreases
- Unfavorable sorption (n<1): rarefactions from increases, shocks from decreases
- Various inlet patterns: steps, pulses, multiple changes
- Plateau at C=0: Important limiting case for remediation and tracer tests

All tests now pass for both favorable and unfavorable sorption regimes.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ../LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction_front_tracking_detailed
from gwtransport.utils import compute_time_edges


@pytest.mark.parametrize(
    "c_initial,c_final,freundlich_n,expected_wave_type",
    [
        # Favorable sorption (n > 1)
        (2.0, 10.0, 2.0, "shock"),  # Increase creates shock
        (10.0, 2.0, 2.0, "rarefaction"),  # Decrease creates rarefaction
        # Unfavorable sorption (n < 1)
        (2.0, 10.0, 0.5, "rarefaction"),  # Increase creates rarefaction
        (10.0, 2.0, 0.5, "shock"),  # Decrease creates shock
    ],
)
def test_step_change_plateau(c_initial, c_final, freundlich_n, expected_wave_type):
    """
    Test that outlet plateaus at final concentration after a single step change.

    Parameters
    ----------
    c_initial : float
        Initial inlet concentration
    c_final : float
        Final inlet concentration
    freundlich_n : float
        Freundlich exponent (n>1: favorable, n<1: unfavorable)
    expected_wave_type : str
        Expected wave type created ("shock" or "rarefaction")
    """
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=300, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Step change at day 60 (late enough for initial plateau to form)
    cin = np.full(len(dates), c_initial)
    cin[60:] = c_final

    flow = np.full(len(dates), 100.0)
    aquifer_pore_volume = 200.0

    # Freundlich parameters
    # For unfavorable sorption (n<1), use much smaller k_f to get reasonable residence times
    # n=2.0, k=0.01 gives ~18-37 day residence times
    # n=0.5, k=0.0001 gives ~6-22 day residence times (comparable)
    # Previous value of 0.1 for n<1 gave 4000-20000 day residence times!
    freundlich_k = 0.01 if freundlich_n > 1.0 else 0.0001
    bulk_density = 1500.0
    porosity = 0.3

    # Extended output to ensure final plateau is reached
    cout_dates = pd.date_range(start=dates[0], periods=300, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Run front tracking
    cout, structure = infiltration_to_extraction_front_tracking_detailed(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Verify wave type
    if expected_wave_type == "shock":
        assert structure["n_shocks"] >= 1, f"Expected at least one shock, got {structure['n_shocks']}"
    elif expected_wave_type == "rarefaction":
        assert structure["n_rarefactions"] >= 1, f"Expected at least one rarefaction, got {structure['n_rarefactions']}"

    # Check final plateau (last 20% of output)
    n_check = len(cout) // 5  # Last 20%
    final_concentrations = cout[-n_check:]

    # Remove any NaN values
    final_concentrations = final_concentrations[~np.isnan(final_concentrations)]

    # Verify plateau at final inlet concentration
    assert len(final_concentrations) > 0, "No valid concentrations in final period"

    # Check that final concentrations are close to c_final
    mean_final = np.mean(final_concentrations)
    std_final = np.std(final_concentrations)

    # Tolerance: allow 1% relative error or 0.01 absolute error (whichever is larger)
    rtol = 0.01
    atol = 0.01
    tolerance = max(abs(c_final) * rtol, atol)

    assert abs(mean_final - c_final) < tolerance, (
        f"Final plateau mean ({mean_final:.4f}) not close to final inlet "
        f"concentration ({c_final:.4f}). Difference: {abs(mean_final - c_final):.4f}, "
        f"tolerance: {tolerance:.4f}. Wave type: {expected_wave_type}, n={freundlich_n:.1f}"
    )

    # Check that plateau is relatively stable (std < 5% of mean or 0.1 absolute)
    max_std = max(abs(c_final) * 0.05, 0.1)
    assert std_final < max_std, (
        f"Final plateau unstable (std={std_final:.4f}, max allowed={max_std:.4f}). "
        f"Wave type: {expected_wave_type}, n={freundlich_n:.1f}"
    )


@pytest.mark.parametrize(
    "freundlich_n",
    [
        2.0,  # Favorable sorption
        0.5,  # Unfavorable sorption
    ],
)
def test_pulse_returns_to_baseline(freundlich_n):
    """
    Test that outlet returns to baseline after a concentration pulse.

    A pulse creates both rising and falling edges, exercising both wave types
    for each sorption condition.

    Parameters
    ----------
    freundlich_n : float
        Freundlich exponent (n>1: favorable, n<1: unfavorable)
    """
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=300, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Pulse: baseline → high → baseline
    c_baseline = 2.0
    c_pulse = 10.0
    cin = np.full(len(dates), c_baseline)
    cin[60:120] = c_pulse  # Pulse from day 60-120

    flow = np.full(len(dates), 100.0)
    aquifer_pore_volume = 200.0

    # Freundlich parameters
    # For unfavorable sorption (n<1), use much smaller k_f to get reasonable residence times
    # n=2.0, k=0.01 gives ~18-37 day residence times
    # n=0.5, k=0.0001 gives ~6-22 day residence times (comparable)
    # Previous value of 0.1 for n<1 gave 4000-20000 day residence times!
    freundlich_k = 0.01 if freundlich_n > 1.0 else 0.0001
    bulk_density = 1500.0
    porosity = 0.3

    # Extended output
    cout_dates = pd.date_range(start=dates[0], periods=300, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Run front tracking
    cout, structure = infiltration_to_extraction_front_tracking_detailed(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Verify both wave types were created
    if freundlich_n > 1.0:
        # Favorable: shock on rise, rarefaction on fall
        assert structure["n_shocks"] >= 1, "Expected shock from pulse rising edge"
        assert structure["n_rarefactions"] >= 1, "Expected rarefaction from pulse falling edge"
    else:
        # Unfavorable: rarefaction on rise, shock on fall
        assert structure["n_rarefactions"] >= 1, "Expected rarefaction from pulse rising edge"
        assert structure["n_shocks"] >= 1, "Expected shock from pulse falling edge"

    # Check final plateau returns to baseline (last 20% of output)
    n_check = len(cout) // 5
    final_concentrations = cout[-n_check:]
    final_concentrations = final_concentrations[~np.isnan(final_concentrations)]

    assert len(final_concentrations) > 0, "No valid concentrations in final period"

    mean_final = np.mean(final_concentrations)
    rtol = 0.01
    atol = 0.01
    tolerance = max(abs(c_baseline) * rtol, atol)

    assert abs(mean_final - c_baseline) < tolerance, (
        f"Final plateau mean ({mean_final:.4f}) did not return to baseline "
        f"({c_baseline:.4f}) after pulse. Difference: {abs(mean_final - c_baseline):.4f}, "
        f"tolerance: {tolerance:.4f}. n={freundlich_n:.1f}"
    )


@pytest.mark.parametrize(
    "freundlich_n",
    [
        2.0,  # Favorable sorption
        0.5,  # Unfavorable sorption
    ],
)
def test_multiple_steps_final_plateau(freundlich_n):
    """
    Test plateau behavior after multiple concentration changes.

    Tests that after a series of concentration changes, the outlet ultimately
    plateaus at the final inlet concentration.

    Parameters
    ----------
    freundlich_n : float
        Freundlich exponent (n>1: favorable, n<1: unfavorable)
    """
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=400, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Multiple steps: 2 → 5 → 10 → 3 → 7
    cin = np.full(len(dates), 2.0)
    cin[50:100] = 5.0
    cin[100:150] = 10.0
    cin[150:200] = 3.0
    cin[200:] = 7.0  # Final concentration

    flow = np.full(len(dates), 100.0)
    aquifer_pore_volume = 200.0

    # Freundlich parameters
    # For unfavorable sorption (n<1), use much smaller k_f to get reasonable residence times
    # n=2.0, k=0.01 gives ~18-37 day residence times
    # n=0.5, k=0.0001 gives ~6-22 day residence times (comparable)
    # Previous value of 0.1 for n<1 gave 4000-20000 day residence times!
    freundlich_k = 0.01 if freundlich_n > 1.0 else 0.0001
    bulk_density = 1500.0
    porosity = 0.3

    # Extended output
    cout_dates = pd.date_range(start=dates[0], periods=400, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Run front tracking
    cout, structure = infiltration_to_extraction_front_tracking_detailed(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Verify multiple waves were created
    total_waves = structure["n_shocks"] + structure["n_rarefactions"]
    assert total_waves >= 4, f"Expected at least 4 waves from multiple steps, got {total_waves}"

    # Check final plateau (last 15% of output)
    n_check = len(cout) * 15 // 100
    final_concentrations = cout[-n_check:]
    final_concentrations = final_concentrations[~np.isnan(final_concentrations)]

    assert len(final_concentrations) > 0, "No valid concentrations in final period"

    c_final = 7.0
    mean_final = np.mean(final_concentrations)
    rtol = 0.01
    atol = 0.01
    tolerance = max(abs(c_final) * rtol, atol)

    assert abs(mean_final - c_final) < tolerance, (
        f"Final plateau mean ({mean_final:.4f}) not close to final inlet "
        f"concentration ({c_final:.4f}) after multiple steps. "
        f"Difference: {abs(mean_final - c_final):.4f}, tolerance: {tolerance:.4f}. "
        f"n={freundlich_n:.1f}"
    )


@pytest.mark.parametrize(
    "c_initial,freundlich_n",
    [
        (10.0, 2.0),  # Favorable sorption: decrease to zero
        (10.0, 0.5),  # Unfavorable sorption: decrease to zero
    ],
)
def test_step_down_to_zero_plateau(c_initial, freundlich_n):
    """
    Test that outlet correctly plateaus at C=0 after step decrease to zero.

    This tests an important limiting case: rarefaction waves (or shocks) that
    asymptotically approach C=0. This is physically relevant for:
    - Aquifer remediation (contaminated → clean)
    - End of tracer injection tests
    - Return to baseline conditions

    At C=0, R(0) = 1.0 exactly (no sorption), so zero concentration travels
    at maximum velocity (pore water velocity).

    Parameters
    ----------
    c_initial : float
        Initial inlet concentration (> 0)
    freundlich_n : float
        Freundlich exponent (n>1: favorable, n<1: unfavorable)
    """
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=300, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Step change to zero at day 60
    cin = np.full(len(dates), c_initial)
    cin[60:] = 0.0

    flow = np.full(len(dates), 100.0)
    aquifer_pore_volume = 200.0

    # Freundlich parameters
    freundlich_k = 0.01 if freundlich_n > 1.0 else 0.0001
    bulk_density = 1500.0
    porosity = 0.3

    # Extended output
    cout_dates = pd.date_range(start=dates[0], periods=300, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Run front tracking
    cout, structure = infiltration_to_extraction_front_tracking_detailed(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Check final plateau (last 20% of output)
    n_check = len(cout) // 5
    final_concentrations = cout[-n_check:]

    # Remove any NaN values
    final_concentrations = final_concentrations[~np.isnan(final_concentrations)]

    # Verify plateau at zero
    assert len(final_concentrations) > 0, "No valid concentrations in final period"

    mean_final = np.mean(final_concentrations)
    max_final = np.max(final_concentrations)

    # For zero, use absolute tolerance only
    atol = 1e-10

    assert abs(mean_final) < atol, f"Final plateau mean ({mean_final:.10f}) not close to zero. n={freundlich_n:.1f}"

    assert abs(max_final) < atol, f"Final plateau max ({max_final:.10f}) not close to zero. n={freundlich_n:.1f}"


@pytest.mark.parametrize(
    "freundlich_n",
    [
        2.0,  # Favorable sorption
        0.5,  # Unfavorable sorption
    ],
)
def test_pulse_from_zero_returns_to_zero(freundlich_n):
    """
    Test that outlet returns to C=0 after a pulse starting from zero.

    Tests the important case of injection pulses that start and end at
    zero concentration (e.g., tracer tests, remediation pulses).

    Parameters
    ----------
    freundlich_n : float
        Freundlich exponent (n>1: favorable, n<1: unfavorable)
    """
    # Setup
    dates = pd.date_range(start="2020-01-01", periods=300, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Pulse from zero: 0 → 10 → 0
    c_pulse = 10.0
    cin = np.full(len(dates), 0.0)
    cin[60:120] = c_pulse

    flow = np.full(len(dates), 100.0)
    aquifer_pore_volume = 200.0

    # Freundlich parameters
    freundlich_k = 0.01 if freundlich_n > 1.0 else 0.0001
    bulk_density = 1500.0
    porosity = 0.3

    # Extended output
    cout_dates = pd.date_range(start=dates[0], periods=300, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Run front tracking
    cout, structure = infiltration_to_extraction_front_tracking_detailed(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Check that pulse was detected (non-zero concentrations)
    max_cout = np.max(cout)
    assert max_cout > 0.5 * c_pulse, f"Pulse not detected at outlet (max={max_cout:.2f})"

    # Check final plateau returns to zero (last 20% of output)
    n_check = len(cout) // 5
    final_concentrations = cout[-n_check:]
    final_concentrations = final_concentrations[~np.isnan(final_concentrations)]

    assert len(final_concentrations) > 0, "No valid concentrations in final period"

    mean_final = np.mean(final_concentrations)
    max_final = np.max(final_concentrations)

    # For zero, use absolute tolerance
    atol = 1e-10

    assert abs(mean_final) < atol, (
        f"Final plateau mean ({mean_final:.10f}) did not return to zero after pulse. n={freundlich_n:.1f}"
    )

    assert abs(max_final) < atol, (
        f"Final plateau max ({max_final:.10f}) did not return to zero after pulse. n={freundlich_n:.1f}"
    )
