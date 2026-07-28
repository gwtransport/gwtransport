"""
Integration tests for front-tracking plateau behavior.

This module tests that outlet concentrations ultimately plateau at the final
inlet concentration for various wave types and sorption conditions.

Tests cover:
- Freundlich n>1: shocks from increases, rarefactions from decreases
- Freundlich n<1: rarefactions from increases, shocks from decreases
- Various inlet patterns: steps, pulses, multiple changes
- Plateau at C=0: important limiting case for remediation and tracer tests

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ../LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction_nonlinear_sorption
from gwtransport.utils import compute_time_edges

FLOW = 100.0
PORE_VOLUME = 200.0
BULK_DENSITY = 1500.0
POROSITY = 0.3


def _run(cin, freundlich_n):
    """Run the public nonlinear-sorption transport on a daily grid spanning ``cin``.

    Output bins coincide with the inlet bins. ``k_f`` is picked per regime so both keep
    comparable residence times (~6-40 days): 0.01 for n>1, 1e-4 for n<1.

    Parameters
    ----------
    cin : numpy.ndarray
        Inlet concentration per daily bin.
    freundlich_n : float
        Freundlich exponent.

    Returns
    -------
    tuple
        ``(cout, structure)`` as returned by the public entry point.
    """
    n_days = len(cin)
    dates = pd.date_range(start="2020-01-01", periods=n_days, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n_days)
    return infiltration_to_extraction_nonlinear_sorption(
        cin=cin,
        flow=np.full(n_days, FLOW),
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=np.array([PORE_VOLUME]),
        freundlich_k=0.01 if freundlich_n > 1.0 else 0.0001,
        freundlich_n=freundlich_n,
        bulk_density=BULK_DENSITY,
        porosity=POROSITY,
    )


def _valid_tail(cout, n_bins):
    """The last ``n_bins`` outlet values with NaNs stripped."""
    tail = cout[-n_bins:]
    tail = tail[~np.isnan(tail)]
    assert len(tail) > 0, "No valid concentrations in final period"
    return tail


@pytest.mark.parametrize(
    ("c_initial", "c_final", "freundlich_n", "expected_wave_type"),
    [
        (2.0, 10.0, 2.0, "shock"),  # n>1: increase creates shock
        (10.0, 2.0, 2.0, "rarefaction"),  # n>1: decrease creates rarefaction
        (2.0, 10.0, 0.5, "rarefaction"),  # n<1: increase creates rarefaction
        (10.0, 2.0, 0.5, "shock"),  # n<1: decrease creates shock
    ],
)
def test_step_change_plateau(c_initial, c_final, freundlich_n, expected_wave_type):
    """The outlet plateaus at the final inlet concentration after a single step change."""
    cin = np.full(300, c_initial)
    cin[60:] = c_final  # Step change late enough for the initial plateau to form
    cout, structure = _run(cin, freundlich_n)

    key = "n_shocks" if expected_wave_type == "shock" else "n_rarefactions"
    assert structure[0][key] >= 1, f"Expected at least one {expected_wave_type}, got {structure[0][key]}"

    final_concentrations = _valid_tail(cout, len(cout) // 5)
    mean_final = np.mean(final_concentrations)
    std_final = np.std(final_concentrations)

    # Tolerance: 1% relative or 0.01 absolute, whichever is larger.
    tolerance = max(abs(c_final) * 0.01, 0.01)
    assert abs(mean_final - c_final) < tolerance, (
        f"Final plateau mean ({mean_final:.4f}) not close to final inlet "
        f"concentration ({c_final:.4f}). Difference: {abs(mean_final - c_final):.4f}, "
        f"tolerance: {tolerance:.4f}. Wave type: {expected_wave_type}, n={freundlich_n:.1f}"
    )

    max_std = max(abs(c_final) * 0.05, 0.1)
    assert std_final < max_std, (
        f"Final plateau unstable (std={std_final:.4f}, max allowed={max_std:.4f}). "
        f"Wave type: {expected_wave_type}, n={freundlich_n:.1f}"
    )


@pytest.mark.parametrize("freundlich_n", [2.0, 0.5])
def test_pulse_returns_to_baseline(freundlich_n):
    """The outlet returns to baseline after a pulse, which exercises both wave types."""
    c_baseline, c_pulse = 2.0, 10.0
    cin = np.full(300, c_baseline)
    cin[60:120] = c_pulse
    cout, structure = _run(cin, freundlich_n)

    # n>1: shock on the rise, rarefaction on the fall. n<1: the mirror image.
    assert structure[0]["n_shocks"] >= 1, "Expected a shock from one pulse edge"
    assert structure[0]["n_rarefactions"] >= 1, "Expected a rarefaction from the other pulse edge"

    mean_final = np.mean(_valid_tail(cout, len(cout) // 5))
    tolerance = max(abs(c_baseline) * 0.01, 0.01)
    assert abs(mean_final - c_baseline) < tolerance, (
        f"Final plateau mean ({mean_final:.4f}) did not return to baseline "
        f"({c_baseline:.4f}) after pulse. Difference: {abs(mean_final - c_baseline):.4f}, "
        f"tolerance: {tolerance:.4f}. n={freundlich_n:.1f}"
    )


@pytest.mark.parametrize("freundlich_n", [2.0, 0.5])
def test_multiple_steps_final_plateau(freundlich_n):
    """After a series of steps (2→5→10→3→7) the outlet plateaus at the final inlet level."""
    cin = np.full(400, 2.0)
    cin[50:100] = 5.0
    cin[100:150] = 10.0
    cin[150:200] = 3.0
    cin[200:] = 7.0
    cout, structure = _run(cin, freundlich_n)

    total_waves = structure[0]["n_shocks"] + structure[0]["n_rarefactions"]
    assert total_waves >= 4, f"Expected at least 4 waves from multiple steps, got {total_waves}"

    c_final = 7.0
    mean_final = np.mean(_valid_tail(cout, len(cout) * 15 // 100))
    tolerance = max(abs(c_final) * 0.01, 0.01)
    assert abs(mean_final - c_final) < tolerance, (
        f"Final plateau mean ({mean_final:.4f}) not close to final inlet "
        f"concentration ({c_final:.4f}) after multiple steps. "
        f"Difference: {abs(mean_final - c_final):.4f}, tolerance: {tolerance:.4f}. "
        f"n={freundlich_n:.1f}"
    )


@pytest.mark.parametrize("freundlich_n", [2.0, 0.5])
def test_step_down_to_zero_plateau(freundlich_n):
    """Step decrease to C=0 builds the correct wave; for n<1 the outlet also reaches zero.

    For n>1 the falling edge is a rarefaction whose tail moves ever more slowly
    (``R → ∞`` as ``C → c_min``), so only the wave structure is asserted — the outlet
    would need thousands of years to drain.
    """
    n_days = 3000 if freundlich_n > 1.0 else 300
    cin = np.full(n_days, 10.0)
    cin[60:] = 0.0
    cout, structure = _run(cin, freundlich_n)

    if freundlich_n > 1.0:
        assert structure[0]["n_rarefactions"] >= 1, (
            f"Expected rarefaction for n>1 step down, "
            f"got {structure[0]['n_rarefactions']} rarefactions, {structure[0]['n_shocks']} shocks"
        )
    else:
        assert structure[0]["n_shocks"] >= 1, (
            f"Expected shock for n<1 step down, "
            f"got {structure[0]['n_shocks']} shocks, {structure[0]['n_rarefactions']} rarefactions"
        )
        mean_final = np.mean(_valid_tail(cout, 60))
        assert abs(mean_final) < 0.1, f"n<1 sorption should reach zero: final={mean_final:.3e}"


@pytest.mark.parametrize("freundlich_n", [2.0, 0.5])
def test_pulse_from_zero_returns_to_zero(freundlich_n):
    """Pulse from C=0 builds both wave types and decays; for n<1 the outlet returns to zero.

    For n>1 the falling edge is a rarefaction whose tail moves ever more slowly, so only
    the decay (not a return to zero) is asserted there.
    """
    n_days = 3000 if freundlich_n > 1.0 else 300
    c_pulse = 10.0
    cin = np.zeros(n_days)
    cin[60:120] = c_pulse
    cout, structure = _run(cin, freundlich_n)

    assert np.max(cout) > 0.5 * c_pulse, f"Pulse not detected at outlet (max={np.max(cout):.2f})"
    # n>1: shock on the rise, rarefaction on the fall. n<1: the mirror image.
    assert structure[0]["n_shocks"] >= 1, "Expected a shock from one pulse edge"
    assert structure[0]["n_rarefactions"] >= 1, "Expected a rarefaction from the other pulse edge"

    mean_final = np.mean(_valid_tail(cout, 60))
    assert mean_final < c_pulse / 2, (
        f"Final concentration should decrease from pulse: final={mean_final:.3e}, pulse={c_pulse}"
    )
    if freundlich_n < 1.0:
        assert abs(mean_final) < 0.1, f"n<1 sorption should return to zero: final={mean_final:.3e}"
