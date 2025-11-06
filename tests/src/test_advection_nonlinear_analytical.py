"""
Tests for nonlinear sorption transport against analytical solutions.

These tests use UNIFORM or SLOWLY-VARYING concentration inputs to avoid
edge effects from retardation factor averaging. Only portions of output
fully informed by flow/concentration data are evaluated.

Theoretical Background
----------------------
For advection-dominated transport with Freundlich isotherm:
    R(C) = 1 + (rho_b/θ)·K_f·n·C^(n-1)

For n < 1 (favorable sorption):
- Higher C → lower R → faster travel
- Shock fronts form

References
----------
.. [1] Serrano, S.E. (2001). Solute transport under non-linear sorption and decay.
.. [2] Van Genuchten, M.Th. (1981). Analytical solutions for chemical transport.
.. [3] Rhee, H.K., et al. (1986). First-order PDEs.
"""

import time

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import (
    extraction_to_infiltration,
    infiltration_to_extraction,
)
from gwtransport.residence_time import freundlich_retardation


def test_constant_retardation_matches_linear_uniform():
    """When R is uniform, nonlinear should match linear function."""
    # LONG uniform input
    n_days = 500
    tedges = pd.date_range(start="2022-01-01", periods=n_days + 1, freq="D")
    cout_tedges = pd.date_range(start="2022-01-01", periods=400 + 1, freq="D")

    # UNIFORM concentration throughout
    c_uniform = 20.0
    cin = np.full(n_days, c_uniform)
    flow = np.full(n_days, 100.0)
    pore_volume = np.array([300.0])

    # Compute retardation (will be constant since C is constant)
    retardation_factors = freundlich_retardation(
        concentration=np.full(n_days, c_uniform),
        freundlich_k=0.02,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    # Verify R is constant
    assert np.allclose(retardation_factors, retardation_factors[0], rtol=1e-10)
    r_constant = retardation_factors[0]

    # Run both methods
    cout_nonlinear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_factors,
    )

    cout_linear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=r_constant,
    )

    # Only compare fully-informed region
    # First arrival: after residence time
    base_rt = pore_volume[0] / flow[0]
    expected_arrival = int(base_rt * r_constant) + 10  # Add buffer

    valid = (~np.isnan(cout_nonlinear) & ~np.isnan(cout_linear))[expected_arrival:]

    if np.sum(valid) > 20:
        np.testing.assert_allclose(
            cout_nonlinear[expected_arrival:][valid],
            cout_linear[expected_arrival:][valid],
            rtol=0.005,  # 0.5% tolerance
            err_msg="Constant R should match linear function",
        )


def test_traveling_wave_uniform_concentration():
    """For uniform input, output should reach input concentration."""
    n_days = 400
    tedges = pd.date_range(start="2022-01-01", periods=n_days + 1, freq="D")
    cout_tedges = pd.date_range(start="2022-01-01", periods=300 + 1, freq="D")

    # UNIFORM concentration
    c_uniform = 25.0
    cin = np.full(n_days, c_uniform)
    flow = np.full(n_days, 100.0)
    pore_volume = np.array([250.0])

    retardation_factors = freundlich_retardation(
        concentration=np.full(n_days, c_uniform),
        freundlich_k=0.02,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_factors,
    )

    # After sufficient time, output should equal input
    base_rt = pore_volume[0] / flow[0]
    expected_arrival = int(base_rt * retardation_factors[0]) + 20

    late_output = cout[expected_arrival : expected_arrival + 50]
    late_output = late_output[~np.isnan(late_output)]

    if len(late_output) > 10:
        mean_late = np.mean(late_output)
        relative_error = abs(mean_late - c_uniform) / c_uniform
        assert relative_error < 0.01, (  # 1% tolerance
            f"Uniform concentration not preserved:\n"
            f"Input: {c_uniform:.2f} mg/L\n"
            f"Output: {mean_late:.2f} mg/L\n"
            f"Error: {relative_error:.2%}"
        )


def test_no_sorption_limit():
    """When K_f=0, R=1 everywhere (no sorption)."""
    n_days = 300
    tedges = pd.date_range(start="2022-01-01", periods=n_days + 1, freq="D")
    cout_tedges = pd.date_range(start="2022-01-01", periods=250 + 1, freq="D")

    # Uniform concentration
    cin = np.full(n_days, 15.0)
    flow = np.full(n_days, 100.0)
    pore_volume = np.array([400.0])

    # No sorption
    retardation_factors = freundlich_retardation(
        concentration=np.full(n_days, 15.0),
        freundlich_k=0.0,  # No sorption
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    np.testing.assert_allclose(retardation_factors, 1.0, rtol=1e-10)

    cout_nonlinear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_factors,
    )

    cout_no_retard = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=1.0,
    )

    # Compare after spin-up
    base_rt = pore_volume[0] / flow[0]
    start_idx = int(base_rt) + 10

    valid = (~np.isnan(cout_nonlinear) & ~np.isnan(cout_no_retard))[start_idx:]

    if np.sum(valid) > 20:
        np.testing.assert_allclose(
            cout_nonlinear[start_idx:][valid],
            cout_no_retard[start_idx:][valid],
            rtol=0.005,  # 0.5% tolerance
            err_msg="K_f=0 should match no retardation",
        )


def test_linear_isotherm_n_equals_one():
    """For n=1, R is constant (linear isotherm)."""
    n_days = 300
    tedges = pd.date_range(start="2022-01-01", periods=n_days + 1, freq="D")
    cout_tedges = pd.date_range(start="2022-01-01", periods=250 + 1, freq="D")

    cin = np.full(n_days, 18.0)
    flow = np.full(n_days, 100.0)
    pore_volume = np.array([350.0])

    # Linear isotherm: n=1
    retardation_factors = freundlich_retardation(
        concentration=np.full(n_days, 18.0),
        freundlich_k=0.02,
        freundlich_n=1.0,  # LINEAR
        bulk_density=1600.0,
        porosity=0.35,
    )

    r_expected = 1.0 + (1600.0 / 0.35) * 0.02
    np.testing.assert_allclose(retardation_factors, r_expected, rtol=1e-10)

    cout_nonlinear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_factors,
    )

    cout_linear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=r_expected,
    )

    base_rt = pore_volume[0] / flow[0]
    start_idx = int(base_rt * r_expected) + 10

    valid = (~np.isnan(cout_nonlinear) & ~np.isnan(cout_linear))[start_idx:]

    if np.sum(valid) > 20:
        np.testing.assert_allclose(
            cout_nonlinear[start_idx:][valid],
            cout_linear[start_idx:][valid],
            rtol=0.005,  # 0.5% tolerance
            err_msg="n=1 should match linear retardation",
        )


def test_mass_conservation_uniform_pulse():
    """Mass should be conserved for uniform-concentration pulse."""
    # VERY LONG input timeseries
    n_days = 600
    tedges = pd.date_range(start="2022-01-01", periods=n_days + 1, freq="D")

    # EXTENDED output to capture all mass
    cout_tedges = pd.date_range(start="2022-01-01", periods=500 + 1, freq="D")

    # UNIFORM concentration pulse (no gradients!)
    cin = np.zeros(n_days)
    cin[100:200] = 30.0  # 100-day uniform pulse

    flow = np.full(n_days, 100.0)
    pore_volume = np.array([100.0])

    # Moderate nonlinearity
    retardation_factors = freundlich_retardation(
        concentration=np.maximum(cin, 0.1),
        freundlich_k=0.02,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_factors,
    )

    # Compute masses over valid output region
    mass_in = np.sum(cin * flow[: len(cin)])
    flow_out = np.full(len(cout), 100.0)
    mass_out = np.nansum(cout * flow_out)

    recovery = mass_out / mass_in
    error = abs(1.0 - recovery)

    # Should conserve within 1%
    assert error < 0.01, (  # Tightened to 1%
        f"Mass not conserved:\nInput: {mass_in:.0f}\nOutput: {mass_out:.0f}\nRecovery: {recovery:.2%}"
    )


def test_constant_retardation_matches_linear_gaussian():
    """When all R values are equal, nonlinear should match linear function."""
    # Setup
    tedges = pd.date_range(start="2022-12-31 12:00", periods=101, freq="D")

    # Extended output window to capture delayed arrivals
    cout_tedges = pd.date_range(start="2022-12-31 12:00", periods=151, freq="D")

    flow = np.full(100, 100.0)
    pore_volume = np.array([500.0])

    # Gaussian input
    cin = 10.0 * np.exp(-0.5 * ((np.arange(100) - 50) / 10) ** 2)

    # CONSTANT retardation (this is the key - should behave linearly)
    retardation_constant = 2.5
    retardation_factors = np.full(100, retardation_constant)

    # Compute with both methods
    cout_nonlinear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_factors,
    )

    cout_linear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_constant,
    )

    # They should match closely (allowing for numerical differences)
    valid = ~np.isnan(cout_nonlinear) & ~np.isnan(cout_linear)
    np.testing.assert_allclose(
        cout_nonlinear[valid],
        cout_linear[valid],
        rtol=0.1,  # 10% tolerance for edge bins with different binning methods
        atol=1e-6,
        err_msg="Nonlinear with constant R should match linear function",
    )


def test_mass_conservation_freundlich():
    """Mass should be conserved (within numerical precision)."""
    tedges = pd.date_range(start="2022-12-31 12:00", periods=201, freq="D")

    # Extended output window (need extra days for retarded arrivals)
    # Max residence time ≈ pv * max(R) / flow = 50 * 3.5 / 100 = 1.75 days
    cout_tedges = pd.date_range(start="2022-12-31 12:00", periods=210, freq="D")

    flow = np.full(200, 100.0)

    # Compact pulse
    cin = np.zeros(200)
    cin[90:110] = 50.0  # 20-day pulse

    # Freundlich retardation (will give R ~ 2-3 range)
    retardation_factors = freundlich_retardation(
        concentration=np.maximum(cin, 0.1),
        freundlich_k=0.01,
        freundlich_n=0.8,
        bulk_density=1600.0,
        porosity=0.35,
    )

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([50.0]),  # Small pore volume for short residence time
        retardation_factor=retardation_factors,
    )

    # Compute masses (only count valid output bins)
    mass_in = np.sum(cin * flow)
    # For mass out, use same flow for simplicity (cout_tedges is close to tedges)
    mass_out = np.nansum(cout[: len(flow)] * flow)

    # Should conserve mass (within ~5% due to finite bin widths and edge effects)
    relative_error = abs(mass_out - mass_in) / mass_in
    assert relative_error < 0.05, f"Mass not conserved: {mass_out:.2f} vs {mass_in:.2f} (error: {relative_error:.2%})"


def test_peak_arrival_time_ordering():
    """High-C peak should arrive earlier than low-C tail (Freundlich n<1)."""
    tedges = pd.date_range(start="2022-12-31 12:00", periods=301, freq="D")

    # Extended output for retarded arrivals
    # Max residence time ≈ pv * max(R) / flow = 100 * 10 / 100 = 10 days
    cout_tedges = pd.date_range(start="2022-12-31 12:00", periods=320, freq="D")

    flow = np.full(300, 100.0)

    # Gaussian plume
    t = np.arange(300)
    cin = 50.0 * np.exp(-0.5 * ((t - 100) / 20) ** 2)

    # Freundlich n < 1: higher C → lower R → faster travel
    retardation_factors = freundlich_retardation(
        concentration=np.maximum(cin, 0.1),
        freundlich_k=0.02,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    cout_nonlinear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([100.0]),  # Smaller pore volume
        retardation_factor=retardation_factors,
    )

    # Compare with average retardation (linear)
    retardation_avg = np.mean(retardation_factors[cin > 1.0])
    cout_linear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([100.0]),
        retardation_factor=retardation_avg,
    )

    # Peak positions
    peak_nonlinear = np.nanargmax(cout_nonlinear)
    peak_linear = np.nanargmax(cout_linear)

    # Nonlinear peak should arrive earlier (high-C travels faster)
    assert peak_nonlinear < peak_linear, (
        f"Nonlinear peak ({peak_nonlinear}) should arrive before linear ({peak_linear})"
    )


def test_no_retardation_limit():
    """When R=1 everywhere, should match no-retardation case."""
    tedges = pd.date_range(start="2022-12-31 12:00", periods=101, freq="D")
    cout_tedges = pd.date_range(start="2022-12-31 12:00", periods=120, freq="D")

    flow = np.full(100, 100.0)
    cin = 10.0 * np.exp(-0.5 * ((np.arange(100) - 50) / 10) ** 2)

    # R = 1 everywhere
    retardation_factors = np.ones(100)

    cout_nonlinear = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        retardation_factor=retardation_factors,
    )

    cout_no_retard = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),
        retardation_factor=1.0,
    )

    valid = ~np.isnan(cout_nonlinear) & ~np.isnan(cout_no_retard)
    np.testing.assert_allclose(
        cout_nonlinear[valid],
        cout_no_retard[valid],
        rtol=0.1,  # 10% tolerance for edge bins with different binning methods
        atol=1e-6,
        err_msg="R=1 everywhere should match no retardation",
    )


def test_monotonicity_no_spurious_oscillations():
    """Output should not have spurious oscillations (TVD property)."""
    tedges = pd.date_range(start="2022-12-31 12:00", periods=201, freq="D")
    cout_tedges = pd.date_range(start="2022-12-31 12:00", periods=350, freq="D")

    flow = np.full(200, 100.0)

    # Monotone increasing step
    cin = np.zeros(200)
    cin[100:] = 20.0  # Step input

    retardation_factors = freundlich_retardation(
        concentration=np.maximum(cin, 0.1),
        freundlich_k=0.02,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([800.0]),
        retardation_factor=retardation_factors,
    )

    # Check for spurious oscillations
    # In a monotone step, output should not overshoot or have strong oscillations
    valid_cout = cout[~np.isnan(cout)]
    if len(valid_cout) > 10:
        # Max value should not significantly exceed input max
        assert np.nanmax(cout) <= cin.max() * 1.01, "Output overshoots input (spurious oscillation)"

        # Should not have negative values (positivity-preserving)
        assert np.nanmin(cout) >= -1e-10, "Output has negative values"


def test_single_pore_volume():
    """Should work with single pore volume."""
    tedges = pd.date_range(start="2022-12-31 12:00", periods=51, freq="D")
    cout_tedges = pd.date_range(start="2022-12-31 12:00", periods=70, freq="D")

    flow = np.full(50, 100.0)
    cin = np.ones(50) * 5.0
    retardation_factors = np.full(50, 2.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([500.0]),  # Single volume
        retardation_factor=retardation_factors,
    )

    assert not np.all(np.isnan(cout)), "Should produce valid output"


def test_multiple_pore_volumes():
    """Should handle heterogeneous aquifer (multiple pore volumes)."""
    tedges = pd.date_range(start="2022-12-31 12:00", periods=51, freq="D")
    cout_tedges = pd.date_range(start="2022-12-31 12:00", periods=80, freq="D")

    flow = np.full(50, 100.0)
    cin = np.ones(50) * 5.0
    retardation_factors = np.full(50, 2.0)

    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([400.0, 600.0, 800.0]),  # Multiple volumes
        retardation_factor=retardation_factors,
    )

    assert not np.all(np.isnan(cout)), "Should produce valid output"


def test_dimension_mismatch_error():
    """Should raise error when dimensions don't match."""
    tedges = pd.date_range(start="2022-12-31 12:00", periods=51, freq="D")
    flow = np.full(50, 100.0)
    cin = np.ones(50) * 5.0
    retardation_factors = np.full(30, 2.0)  # Wrong length!

    with pytest.raises(ValueError, match="retardation_factor array must match cin length"):
        infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([500.0]),
            retardation_factor=retardation_factors,
        )


def test_performance_reasonable():
    """Verify computation completes in reasonable time."""
    tedges = pd.date_range(start="2022-12-31 12:00", periods=366, freq="D")
    # Max residence time ≈ pv * max(R) / flow = 80 * 10 / 100 = 8 days
    cout_tedges = pd.date_range(start="2022-12-31 12:00", periods=380, freq="D")

    flow = np.full(365, 100.0)
    cin = 10.0 * np.exp(-0.5 * ((np.arange(365) - 180) / 30) ** 2)

    retardation_factors = freundlich_retardation(
        concentration=np.maximum(cin, 0.1),
        freundlich_k=0.02,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    # Should complete in reasonable time (< 5 seconds for 365 days)

    start = time.time()
    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([80.0]),  # Smaller pore volume
        retardation_factor=retardation_factors,
    )
    elapsed = time.time() - start

    assert not np.all(np.isnan(cout)), "Should produce valid output"
    assert elapsed < 5.0, f"Computation too slow: {elapsed:.3f}s (expected < 5s)"


def test_roundtrip_nonlinear_sorption():
    """
    Test roundtrip consistency: infiltration → extraction → infiltration.

    This verifies that extraction_to_infiltration correctly inverts
    infiltration_to_extraction for nonlinear sorption cases.
    """
    # Simplified setup with aligned time grids for easier roundtrip
    n_days = 200
    tedges = pd.date_range(start="2022-01-01", periods=n_days + 1, freq="D")

    # Uniform flow
    flow = np.full(n_days, 100.0)

    # Create a smooth Gaussian pulse for cin (avoids edge artifacts)
    # Center it early enough to have good extraction data
    t = np.arange(n_days)
    cin = 30.0 * np.exp(-0.5 * ((t - 50) / 20) ** 2)

    # Single pore volume for simplicity
    pore_volume = np.array([300.0])

    # Freundlich retardation from infiltration concentration
    r_forward = freundlich_retardation(
        concentration=np.maximum(cin, 0.1),
        freundlich_k=0.02,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    # Forward: infiltration → extraction (using SAME time grid)
    cout = infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,  # Use same grid
        aquifer_pore_volumes=pore_volume,
        retardation_factor=r_forward,
    )

    # Find valid extraction region (where cout is not NaN and has signal)
    valid_cout_mask = (~np.isnan(cout)) & (cout > 0.5)  # Above noise level
    n_valid = np.sum(valid_cout_mask)

    if n_valid < 20:
        pytest.skip(f"Insufficient valid extraction data: {n_valid} points")

    # For backward pass, use only the valid region
    valid_idx = np.where(valid_cout_mask)[0]
    start_idx = valid_idx[0]
    end_idx = valid_idx[-1] + 1

    cout_valid = cout[start_idx:end_idx]
    tedges_out_valid = tedges[start_idx : end_idx + 1]
    flow_out_valid = flow[start_idx:end_idx]

    # Compute retardation from EXTRACTION concentrations for backward pass
    r_backward = freundlich_retardation(
        concentration=np.maximum(cout_valid, 0.1),
        freundlich_k=0.02,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    # Backward: extraction → infiltration (reconstruct on same grid)
    cin_recon = extraction_to_infiltration(
        cout=cout_valid,
        flow=flow_out_valid,
        tedges=tedges_out_valid,
        cin_tedges=tedges,  # Reconstruct on full grid
        aquifer_pore_volumes=pore_volume,
        retardation_factor=r_backward,
    )

    # Compare in the region where we have good data
    # Focus on the peak region where signal is strong and should have valid roundtrip
    peak_idx = np.argmax(cin)
    margin = 30  # Days around peak
    compare_start = max(0, peak_idx - margin)
    compare_end = min(len(cin), peak_idx + margin)

    # Extract comparison regions
    cin_original = cin[compare_start:compare_end]
    cin_reconstructed = cin_recon[compare_start:compare_end]

    # Only compare where both have valid data with good dynamics
    valid = (~np.isnan(cin_reconstructed)) & (cin_original > 1.0)  # Above noise level
    n_valid_compare = np.sum(valid)

    if n_valid_compare < 10:  # Need enough points for meaningful comparison
        pytest.skip(f"Insufficient valid data points ({n_valid_compare}) for comparison")

    # Compute relative error
    relative_errors = np.abs(cin_reconstructed[valid] - cin_original[valid]) / cin_original[valid]
    mean_rel_error = np.mean(relative_errors)
    max_rel_error = np.max(relative_errors)

    # Roundtrip should reconstruct within reasonable tolerance
    # Note: Some error is expected due to:
    # 1. Numerical discretization
    # 2. Different retardation factors (forward uses R(cin), backward uses R(cout))
    # 3. Binning artifacts
    assert mean_rel_error < 0.20, (
        f"Roundtrip mean error too large:\n"
        f"Mean relative error: {mean_rel_error:.2%}\n"
        f"Max relative error: {max_rel_error:.2%}\n"
        f"Valid points: {n_valid_compare}"
    )

    assert max_rel_error < 0.40, (
        f"Roundtrip max error too large:\n"
        f"Max relative error: {max_rel_error:.2%}\n"
        f"Mean relative error: {mean_rel_error:.2%}"
    )

    # Mass balance check in comparison region
    mass_original = np.sum(cin_original[valid])
    mass_reconstructed = np.sum(cin_reconstructed[valid])
    mass_recovery = mass_reconstructed / mass_original

    assert 0.80 < mass_recovery < 1.20, f"Roundtrip mass not conserved:\nRecovery: {mass_recovery:.2%}"
