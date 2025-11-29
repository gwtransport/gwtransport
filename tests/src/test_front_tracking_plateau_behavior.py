"""
Integration tests for front-tracking plateau behavior.

This module tests that outlet concentrations ultimately plateau at the final
inlet concentration for various wave types and sorption conditions.

Tests cover:
- n>1 sorption (n>1): shocks from increases, rarefactions from decreases
- n<1 sorption (n<1): rarefactions from increases, shocks from decreases
- Various inlet patterns: steps, pulses, multiple changes
- Plateau at C=0: Important limiting case for remediation and tracer tests

All tests now pass for both n>1 and n<1 regimes.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ../LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction_front_tracking_detailed
from gwtransport.utils import compute_time_edges


@pytest.mark.parametrize(
    ("c_initial", "c_final", "freundlich_n", "expected_wave_type"),
    [
        # n>1 sorption (n > 1)
        (2.0, 10.0, 2.0, "shock"),  # Increase creates shock
        (10.0, 2.0, 2.0, "rarefaction"),  # Decrease creates rarefaction
        # n<1 sorption (n < 1)
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
        Freundlich exponent (n>1: n>1, n<1: n<1)
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
    # For n<1 (lower C travels faster), use much smaller k_f to get reasonable residence times
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
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Verify wave type
    if expected_wave_type == "shock":
        assert structure[0]["n_shocks"] >= 1, f"Expected at least one shock, got {structure[0]['n_shocks']}"
    elif expected_wave_type == "rarefaction":
        assert structure[0]["n_rarefactions"] >= 1, (
            f"Expected at least one rarefaction, got {structure[0]['n_rarefactions']}"
        )

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
        2.0,  # n>1 sorption
        0.5,  # n<1 sorption
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
        Freundlich exponent (n>1: n>1, n<1: n<1)
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
    # For n<1 (lower C travels faster), use much smaller k_f to get reasonable residence times
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
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Verify both wave types were created
    if freundlich_n > 1.0:
        # n>1: shock on rise, rarefaction on fall
        assert structure[0]["n_shocks"] >= 1, "Expected shock from pulse rising edge"
        assert structure[0]["n_rarefactions"] >= 1, "Expected rarefaction from pulse falling edge"
    else:
        # n<1: rarefaction on rise, shock on fall
        assert structure[0]["n_rarefactions"] >= 1, "Expected rarefaction from pulse rising edge"
        assert structure[0]["n_shocks"] >= 1, "Expected shock from pulse falling edge"

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
        2.0,  # n>1 sorption
        0.5,  # n<1 sorption
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
        Freundlich exponent (n>1: n>1, n<1: n<1)
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
    # For n<1 (lower C travels faster), use much smaller k_f to get reasonable residence times
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
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Verify multiple waves were created
    total_waves = structure[0]["n_shocks"] + structure[0]["n_rarefactions"]
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
    ("c_initial", "freundlich_n"),
    [
        (10.0, 2.0),  # n>1 sorption: decrease to zero
        (10.0, 0.5),  # n<1 sorption: decrease to zero
    ],
)
def test_step_down_to_zero_plateau(c_initial, freundlich_n):
    """
    Test wave structure for step decrease to C≈0.

    For n>1 (higher C travels faster), stepping down to C≈0 creates a rarefaction wave
    whose tail moves extremely slowly (R→∞ as C→c_min). The rarefaction may take
    thousands of years to fully propagate, but the wave structure should be correct.

    For n<1 (lower C travels faster), the wave completes quickly and outlet reaches zero.

    Parameters
    ----------
    c_initial : float
        Initial inlet concentration (> 0)
    freundlich_n : float
        Freundlich exponent (n>1: n>1, n<1: n<1)
    """
    # Setup - use longer simulation time for n>1
    n_days = 3000 if freundlich_n > 1.0 else 300
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
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
    cout_dates = pd.date_range(start=dates[0], periods=n_days, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Run front tracking
    cout, structure = infiltration_to_extraction_front_tracking_detailed(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Verify correct wave structure
    if freundlich_n > 1.0:
        # n>1: step down creates rarefaction
        assert structure[0]["n_rarefactions"] >= 1, (
            f"Expected rarefaction for n>1 step down, "
            f"got {structure[0]['n_rarefactions']} rarefactions, {structure[0]['n_shocks']} shocks"
        )
    else:
        # n<1: step down creates shock
        assert structure[0]["n_shocks"] >= 1, (
            f"Expected shock for n<1 step down, "
            f"got {structure[0]['n_shocks']} shocks, {structure[0]['n_rarefactions']} rarefactions"
        )

    # Verify concentration behavior
    # For n>1, the rarefaction wave is created but may take very long to complete
    # We just need to verify the wave structure is correct (already checked above)

    # For n<1, verify it reaches near zero
    if freundlich_n < 1.0:
        # For n<1, waves complete quickly
        # Check that final concentration is near zero
        final_outlet = cout[-60:]  # Last 60 days
        final_outlet = final_outlet[~np.isnan(final_outlet)]

        assert len(final_outlet) > 0, "No valid final concentrations"
        mean_final = np.mean(final_outlet)

        # Should reach near zero for n<1
        atol = 0.1
        assert abs(mean_final) < atol, f"n<1 sorption should reach zero: final={mean_final:.3e}"
    # For n>1 (higher C travels faster), the rarefaction structure is verified above
    # The wave may take thousands of days to complete, so we don't check final concentration


@pytest.mark.parametrize(
    "freundlich_n",
    [
        2.0,  # n>1 sorption
        0.5,  # n<1 sorption
    ],
)
def test_pulse_from_zero_returns_to_zero(freundlich_n):
    """
    Test wave structure for pulse from C≈0 that returns to C≈0.

    For n>1 (higher C travels faster), the falling edge creates a rarefaction whose tail
    moves extremely slowly. The wave structure should be correct even if the rarefaction
    doesn't complete in finite time.

    For n<1 (lower C travels faster), waves complete quickly and outlet returns to zero.

    Parameters
    ----------
    freundlich_n : float
        Freundlich exponent (n>1: n>1, n<1: n<1)
    """
    # Setup - use longer simulation time for n>1
    n_days = 3000 if freundlich_n > 1.0 else 300
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
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
    cout_dates = pd.date_range(start=dates[0], periods=n_days, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Run front tracking
    cout, structure = infiltration_to_extraction_front_tracking_detailed(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Check that pulse was detected
    max_cout = np.max(cout)
    assert max_cout > 0.5 * c_pulse, f"Pulse not detected at outlet (max={max_cout:.2f})"

    # Verify correct wave structure
    if freundlich_n > 1.0:
        # n>1: rising edge creates shock, falling edge creates rarefaction
        assert structure[0]["n_shocks"] >= 1, (
            f"Expected shock from pulse rising edge for n>1, got {structure[0]['n_shocks']} shocks"
        )
        assert structure[0]["n_rarefactions"] >= 1, (
            f"Expected rarefaction from pulse falling edge for n>1, got {structure[0]['n_rarefactions']} rarefactions"
        )
    else:
        # n<1: rising edge creates rarefaction, falling edge creates shock
        assert structure[0]["n_rarefactions"] >= 1, (
            f"Expected rarefaction from pulse rising edge for n<1, got {structure[0]['n_rarefactions']} rarefactions"
        )
        assert structure[0]["n_shocks"] >= 1, (
            f"Expected shock from pulse falling edge for n<1, got {structure[0]['n_shocks']} shocks"
        )

    # Verify concentration is decreasing after pulse
    mid_outlet = cout[len(cout) // 3 : len(cout) // 2]  # Middle third
    final_outlet = cout[-60:]  # Last 60 days
    mid_outlet = mid_outlet[~np.isnan(mid_outlet)]
    final_outlet = final_outlet[~np.isnan(final_outlet)]

    assert len(mid_outlet) > 0, "No valid mid-period concentrations"
    assert len(final_outlet) > 0, "No valid final concentrations"

    mean_final = np.mean(final_outlet)

    # Should be decreasing after pulse passes
    assert mean_final < c_pulse / 2, (
        f"Final concentration should decrease from pulse: final={mean_final:.3e}, pulse={c_pulse}"
    )

    # For n<1 (lower C travels faster), should actually return to near zero
    if freundlich_n < 1.0:
        atol = 0.1
        assert abs(mean_final) < atol, f"n<1 sorption should return to zero: final={mean_final:.3e}"
    # For n>1 (higher C travels faster), just verify it's decreasing (may not reach c_min in finite time)
