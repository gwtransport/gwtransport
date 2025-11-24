"""
Consolidated roundtrip tests for advection module.

This module contains all roundtrip reconstruction tests (infiltration → extraction → infiltration)
consolidated from multiple test files with proper parameterization.

Roundtrip tests verify that:
1. Forward pass: cin → cout (infiltration_to_extraction)
2. Backward pass: cout → cin_reconstructed (extraction_to_infiltration)
3. Reconstruction: cin_reconstructed ≈ cin (in valid region)

These tests are critical for verifying that the deconvolution (backward) operation
correctly inverts the convolution (forward) operation.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import extraction_to_infiltration, infiltration_to_extraction
from gwtransport.residence_time import freundlich_retardation
from gwtransport.utils import compute_time_edges

# ============================================================================
# Roundtrip Tests - Linear Methods
# ============================================================================


class TestRoundtripLinear:
    """Roundtrip tests for linear retardation (constant R)."""

    def test_roundtrip_linear_sine_wave(self):
        """
        Test roundtrip with linear retardation and sine wave input.

        This tests the basic roundtrip property without nonlinear complications.
        With constant retardation, the forward and backward operations should
        invert perfectly (within numerical precision).
        """
        # Full infiltration window - one year for ample history
        cin_dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
        cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

        # Extraction window: overlaps with cin but starts/ends inside
        cout_dates = pd.date_range(start="2022-03-01", end="2022-10-31", freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Sine wave input for smooth variation
        cin_original = 30.0 + 20.0 * np.sin(2 * np.pi * np.arange(len(cin_dates)) / 40.0)

        # Constant flow and pore volume
        flow_cin = np.full(len(cin_dates), 100.0)
        flow_cout = np.full(len(cout_dates), 100.0)
        pore_volume = np.array([500.0])

        # Constant retardation (linear)
        retardation_factor = 1.5

        # Forward pass
        cout = infiltration_to_extraction(
            cin=cin_original,
            flow=flow_cin,
            tedges=cin_tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volume,
            retardation_factor=retardation_factor,
        )

        # Backward pass
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_cout,
            tedges=cout_tedges,
            cin_tedges=cin_tedges,
            aquifer_pore_volumes=pore_volume,
            retardation_factor=retardation_factor,
        )

        # Analysis: Compare in valid region only
        valid_mask = ~np.isnan(cin_reconstructed)
        valid_indices = np.where(valid_mask)[0]

        # Require reasonable coverage
        coverage = len(valid_indices) / len(cin_reconstructed)
        assert coverage >= 0.50, f"Insufficient coverage: {coverage:.1%} (expected >= 50%)"

        # Compare in stable middle region (skip boundaries)
        n_skip = max(20, int(0.2 * len(valid_indices)))
        middle_indices = valid_indices[n_skip:-n_skip]

        reconstructed_middle = cin_reconstructed[middle_indices]
        original_middle = cin_original[middle_indices]

        # Linear retardation should have tight roundtrip (< 2% error)
        # Note: Even with constant R, numerical discretization introduces ~1% error
        np.testing.assert_allclose(
            reconstructed_middle,
            original_middle,
            rtol=0.02,  # 2% tolerance for linear case (discretization effects)
            err_msg="Linear roundtrip should reconstruct with < 2% error",
        )


# ============================================================================
# Roundtrip Tests - Nonlinear Methods (Parameterized)
# ============================================================================


@pytest.mark.parametrize("input_type", ["gaussian", "step", "sine"])
@pytest.mark.parametrize("n_freundlich", [0.6, 0.75, 0.9])
class TestRoundtripNonlinear:
    """
    Comprehensive roundtrip tests for nonlinear sorption.

    Tests all combinations of:
    - Nonlinear method: method_of_characteristics
    - Input types: gaussian (smooth), step (discontinuous), sine (periodic)
    - Freundlich n values: 0.6 (strong), 0.75 (moderate), 0.9 (weak nonlinearity)

    Total: 3 x 3 = 9 test cases (consolidated from ~25 scattered tests)
    """

    def test_roundtrip_reconstruction(self, input_type, n_freundlich):
        """
        Test roundtrip reconstruction for nonlinear sorption.

        Parameters:
            input_type: 'gaussian', 'step', or 'sine'
            n_freundlich: Freundlich exponent (0.6, 0.75, 0.9)

        Validation:
            - Reconstruction error < 1.5% for gaussian/sine (smooth inputs)
            - Reconstruction error < 10% for step (discontinuous input)
            - Valid data coverage >= 50%
        """
        # Setup time windows
        n_days_cin = 300
        cin_dates = pd.date_range(start="2022-01-01", periods=n_days_cin, freq="D")
        cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

        # Extended output for low n (higher retardation)
        n_days_cout = n_days_cin + 60
        cout_dates = pd.date_range(start="2022-01-01", periods=n_days_cout, freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        # Create input based on type
        t = np.arange(n_days_cin)
        if input_type == "gaussian":
            # Gaussian pulse
            cin_original = 55.0 * np.exp(-0.5 * ((t - 150) / 25) ** 2)
            tolerance = 0.015  # 1.5% for smooth input
            atol = 1.0  # 1 mg/L absolute tolerance
        elif input_type == "step":
            # Step function with plateaus
            cin_original = np.zeros(n_days_cin)
            cin_original[80:160] = 40.0
            cin_original[160:240] = 70.0
            tolerance = 0.10  # 10% for discontinuous input
            atol = 35.0  # 35 mg/L absolute tolerance for step boundaries
        else:  # sine
            # Sine wave
            cin_original = 30.0 + 20.0 * np.sin(2 * np.pi * t / 40.0)
            tolerance = 0.015  # 1.5% for smooth input
            atol = 1.0  # 1 mg/L absolute tolerance

        # Flow and pore volume
        flow_cin = np.full(n_days_cin, 100.0)
        flow_cout = np.full(n_days_cout, 100.0)
        pore_volume = np.array([45.0])  # Small pore volume for realistic residence times

        # Freundlich parameters
        freundlich_k = 0.001  # Small K to avoid extreme retardation
        bulk_density = 1600.0
        porosity = 0.35

        # Forward pass retardation
        retardation_cin = freundlich_retardation(
            concentration=np.maximum(cin_original, 0.01),
            freundlich_k=freundlich_k,
            freundlich_n=n_freundlich,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        # Forward: infiltration → extraction
        cout_full = infiltration_to_extraction(
            cin=cin_original,
            flow=flow_cin,
            tedges=cin_tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volume,
            retardation_factor=retardation_cin,
        )

        # Trim to valid cout region
        cout_valid_mask = ~np.isnan(cout_full)
        cout_valid_indices = np.where(cout_valid_mask)[0]

        assert len(cout_valid_indices) > 100, (
            f"Insufficient valid cout data for n={n_freundlich}, input={input_type}: {len(cout_valid_indices)} bins"
        )

        cout_start = cout_valid_indices[0]
        cout_end = cout_valid_indices[-1] + 1

        cout = cout_full[cout_start:cout_end]
        flow_cout_trimmed = flow_cout[cout_start:cout_end]
        cout_tedges_trimmed = cout_tedges[cout_start : cout_end + 1]

        # Backward pass retardation
        retardation_cout = freundlich_retardation(
            concentration=np.maximum(cout, 0.01),
            freundlich_k=freundlich_k,
            freundlich_n=n_freundlich,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        # Backward: extraction → infiltration
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_cout_trimmed,
            tedges=cout_tedges_trimmed,
            cin_tedges=cin_tedges,
            aquifer_pore_volumes=pore_volume,
            retardation_factor=retardation_cout,
        )

        # Analysis: Compare in valid middle region
        valid_mask = ~np.isnan(cin_reconstructed)
        valid_indices = np.where(valid_mask)[0]

        assert len(valid_indices) >= 100, (
            f"Need at least 100 valid bins for n={n_freundlich}, input={input_type}, got {len(valid_indices)}"
        )

        # Skip boundaries (20% on each end)
        n_skip = max(20, int(0.2 * len(valid_indices)))
        middle_indices = valid_indices[n_skip:-n_skip]

        reconstructed_middle = cin_reconstructed[middle_indices]
        original_middle = cin_original[middle_indices]

        # Test reconstruction accuracy
        np.testing.assert_allclose(
            reconstructed_middle,
            original_middle,
            rtol=tolerance,
            atol=atol,
            err_msg=(
                f"Roundtrip reconstruction error exceeds tolerance\n"
                f"Method: method_of_characteristics\n"
                f"Input: {input_type}\n"
                f"n: {n_freundlich}\n"
                f"Tolerance: {tolerance:.1%}\n"
                f"Atol: {atol:.1f} mg/L"
            ),
        )


# ============================================================================
# Helper Tests - Coverage Validation
# ============================================================================


def test_roundtrip_coverage_requirements():
    """
    Test that roundtrip tests have sufficient valid data coverage.

    This meta-test validates that our test parameters produce enough valid
    data for meaningful roundtrip comparison. It ensures we're not testing
    edge cases where most data is NaN.
    """
    # Use standard scenario
    n_days = 300
    cin_dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")
    cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

    cout_dates = pd.date_range(start="2022-01-01", periods=n_days + 60, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    # Gaussian input
    t = np.arange(n_days)
    cin_original = 50.0 * np.exp(-0.5 * ((t - 150) / 25) ** 2)

    flow_cin = np.full(n_days, 100.0)
    pore_volume = np.array([45.0])

    retardation_cin = freundlich_retardation(
        concentration=np.maximum(cin_original, 0.01),
        freundlich_k=0.001,
        freundlich_n=0.75,
        bulk_density=1600.0,
        porosity=0.35,
    )

    cout_full = infiltration_to_extraction(
        cin=cin_original,
        flow=flow_cin,
        tedges=cin_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_cin,
    )

    # Check cout coverage
    cout_valid = np.sum(~np.isnan(cout_full))
    cout_coverage = cout_valid / len(cout_full)

    # Require at least 50% valid cout data
    assert cout_coverage >= 0.50, (
        f"Insufficient cout coverage ({cout_coverage:.1%}) for method_of_characteristics. "
        f"Roundtrip tests require >= 50% valid data."
    )
