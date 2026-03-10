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

from gwtransport.advection import extraction_to_infiltration, infiltration_to_extraction
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

        # Backward pass: tedges = cin/flow grid, cout_tedges = cout grid
        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_cin,
            tedges=cin_tedges,
            cout_tedges=cout_tedges,
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


class TestRoundtripSameGrid:
    """
    Roundtrip tests where cin/flow grid and cout grid have the same resolution.

    The cin/tedges range starts earlier to ensure all cout bins have valid cin
    history (no NaN in cout). This tests that the lstsq inversion gives
    machine-precision reconstruction when the forward system is well-conditioned.
    """

    def test_roundtrip_same_grid_single_pore_volume(self):
        """
        Roundtrip with single pore volume should be near machine precision.

        The forward weight matrix is effectively a square shift matrix for a
        single pore volume on the same daily grid. Inverting via lstsq gives
        back cin to high accuracy in the fully-constrained interior region.
        """
        # cin/flow grid starts earlier to provide history (pore_volume/flow = 10 days)
        cin_dates = pd.date_range(start="2021-12-22", end="2022-12-31", freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

        # cout grid: same daily resolution, starts after spin-up
        cout_dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cin_original = 30.0 + 20.0 * np.sin(2 * np.pi * np.arange(len(cin_dates)) / 40.0)
        flow = np.full(len(cin_dates), 100.0)
        pore_volume = np.array([1000.0])  # 10-day residence time at 100 m3/day

        cout = infiltration_to_extraction(
            cin=cin_original,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volume,
        )

        # cout should be fully valid (no NaN) because tedges starts 10 days before cout_tedges
        assert not np.any(np.isnan(cout)), "Expected no NaN in cout with sufficient spin-up"

        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volume,
        )

        valid_mask = ~np.isnan(cin_reconstructed)
        valid_indices = np.where(valid_mask)[0]

        assert len(valid_indices) >= 10, f"Too few valid bins: {len(valid_indices)}"

        n_skip = max(5, int(0.1 * len(valid_indices)))
        middle_indices = valid_indices[n_skip:-n_skip]

        np.testing.assert_allclose(
            cin_reconstructed[middle_indices],
            cin_original[middle_indices],
            rtol=1e-6,
            err_msg="Same-resolution single pore volume roundtrip should reconstruct with < 1e-6 relative error",
        )

    def test_roundtrip_same_grid_multiple_pore_volumes(self):
        """
        Roundtrip with multiple pore volumes on same-resolution grids.

        With multiple pore volumes, W_forward is a square averaging of shift
        matrices. The lstsq inversion should still give tight reconstruction
        in the fully-constrained interior region.
        """
        # cin/flow grid starts earlier (max pore_volume/flow = 700/100 = 7 days)
        cin_dates = pd.date_range(start="2021-12-25", end="2022-12-31", freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

        # cout grid: same daily resolution, starts after spin-up
        cout_dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cin_original = 30.0 + 20.0 * np.sin(2 * np.pi * np.arange(len(cin_dates)) / 40.0)
        flow = np.full(len(cin_dates), 100.0)
        pore_volumes = np.array([300.0, 500.0, 700.0])  # 3-7 day residence times

        cout = infiltration_to_extraction(
            cin=cin_original,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volumes,
        )

        assert not np.any(np.isnan(cout)), "Expected no NaN in cout with sufficient spin-up"

        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volumes,
        )

        valid_mask = ~np.isnan(cin_reconstructed)
        valid_indices = np.where(valid_mask)[0]

        assert len(valid_indices) >= 10, f"Too few valid bins: {len(valid_indices)}"

        n_skip = max(10, int(0.15 * len(valid_indices)))
        middle_indices = valid_indices[n_skip:-n_skip]

        np.testing.assert_allclose(
            cin_reconstructed[middle_indices],
            cin_original[middle_indices],
            rtol=0.005,
            err_msg="Same-resolution multi pore volume roundtrip should reconstruct with < 0.5% relative error",
        )


# ============================================================================
