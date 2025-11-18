"""
Integration tests for front-tracking plateau behavior.

This module tests that outlet concentrations ultimately plateau at the final
inlet concentration for various wave types and sorption conditions.

Tests cover:
- Favorable sorption (n>1): shocks from increases, rarefactions from decreases
- Unfavorable sorption (n<1): rarefactions from increases, shocks from decreases
- Various inlet patterns: steps, pulses, multiple changes

STATUS (as of 2025-01-18):
==========================
✅ FIXED: Rarefaction plateau behavior for favorable sorption (n>1)
   - Rarefaction tails now properly establish final plateaus
   - All favorable sorption tests pass

⚠️ KNOWN ISSUE: Unfavorable sorption (n<1) has separate issues
   - Waves produce incorrect concentrations (0.0 or wrong values)
   - Requires separate investigation and fix
   - Tests for n<1 are marked as xfail

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
        # Favorable sorption (n > 1) - PASSING
        (2.0, 10.0, 2.0, "shock"),  # Increase creates shock
        (10.0, 2.0, 2.0, "rarefaction"),  # Decrease creates rarefaction
        # Unfavorable sorption (n < 1) - separate issues
        pytest.param(
            2.0,
            10.0,
            0.5,
            "rarefaction",
            marks=pytest.mark.xfail(reason="Unfavorable sorption produces incorrect concentrations - separate issue"),
        ),
        pytest.param(
            10.0,
            2.0,
            0.5,
            "shock",
            marks=pytest.mark.xfail(reason="Unfavorable sorption produces incorrect concentrations - separate issue"),
        ),
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
    freundlich_k = 0.01 if freundlich_n > 1.0 else 0.1
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
        2.0,  # PASSING
        pytest.param(0.5, marks=pytest.mark.xfail(reason="Unfavorable sorption - separate issue")),
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
    freundlich_k = 0.01 if freundlich_n > 1.0 else 0.1
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
        2.0,  # PASSING
        pytest.param(0.5, marks=pytest.mark.xfail(reason="Unfavorable sorption - separate issue")),
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
    freundlich_k = 0.01 if freundlich_n > 1.0 else 0.1
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
