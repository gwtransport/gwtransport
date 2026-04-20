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
from gwtransport.advection_utils import _infiltration_to_extraction_weights
from gwtransport.utils import compute_time_edges

# ============================================================================
# Roundtrip Tests - Linear Methods
# ============================================================================


class TestRoundtripLinear:
    """Roundtrip tests for linear retardation (constant R)."""

    def test_roundtrip_linear_sine_wave(self):
        """
        Test roundtrip with linear retardation and sine wave input.

        Uses a fully-determined setup (single PV, n_cout = n_cin, sufficient
        spin-up) so the forward operator is a square shift matrix and the
        roundtrip should reconstruct to near machine precision.
        """
        # Effective residence time = pore_volume * retardation_factor / flow
        # = 400 * 1.5 / 100 = 6 days (integer days so the forward map is a pure
        # bin shift and the roundtrip is exact apart from solver round-off).
        # Pad cin grid by 6 days for spin-up.
        cin_dates = pd.date_range(start="2021-12-26", end="2022-12-31", freq="D")
        cin_tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

        # Same daily resolution; cout starts after spin-up so the system is
        # fully determined in the interior.
        cout_dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
        cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

        cin_original = 30.0 + 20.0 * np.sin(2 * np.pi * np.arange(len(cin_dates)) / 40.0)

        flow_cin = np.full(len(cin_dates), 100.0)
        pore_volume = np.array([400.0])
        retardation_factor = 1.5

        cout = infiltration_to_extraction(
            cin=cin_original,
            flow=flow_cin,
            tedges=cin_tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volume,
            retardation_factor=retardation_factor,
        )

        # cout should be fully valid (no NaN) because cin grid starts 8 days early.
        assert not np.any(np.isnan(cout)), "Expected no NaN in cout with sufficient spin-up"

        cin_reconstructed = extraction_to_infiltration(
            cout=cout,
            flow=flow_cin,
            tedges=cin_tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volume,
            retardation_factor=retardation_factor,
        )

        valid_mask = ~np.isnan(cin_reconstructed)
        valid_indices = np.where(valid_mask)[0]

        n_skip = max(20, int(0.1 * len(valid_indices)))
        middle_indices = valid_indices[n_skip:-n_skip]

        np.testing.assert_allclose(
            cin_reconstructed[middle_indices],
            cin_original[middle_indices],
            rtol=1e-12,
            err_msg="Linear roundtrip should reconstruct to machine precision for fully-determined setup",
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

        The forward operator W has shape (n_cout, n_cin) with
        ``n_cin - n_cout = max(rt_in_bins)`` extra unknowns from cin spin-up
        bins that map only to early cout bins. The system is therefore
        under-determined with a nullspace of dim = (n_cin - n_cout).
        Reconstruction error is bounded by the projection of cin_original
        onto null(W), which we compute explicitly below and use as the
        pointwise tolerance. Tikhonov regularization (default 1e-10) does
        not contribute meaningfully to the error in the well-conditioned
        directions: ``lambda / s_min**2 ~ 1e-10 / 0.01**2 = 1e-6`` << nullspace bound.
        """
        # cin/flow grid starts earlier (max pore_volume/flow = 700/100 = 7 days)
        cin_dates = pd.date_range(start="2021-12-25", end="2022-12-31", freq="D")
        tedges = compute_time_edges(tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates))

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

        # Compute the pointwise nullspace projection of cin_original. The
        # reconstruction error |cin_recovered - cin_original| at each bin is
        # bounded by |P_null(cin_original)| at that bin, since cin_recovered
        # and cin_original agree modulo null(W).
        w_forward = _infiltration_to_extraction_weights(
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volumes,
            flow=flow,
            retardation_factor=1.0,
        )
        _, sing_vals, vt = np.linalg.svd(w_forward, full_matrices=True)
        null_basis = vt[len(sing_vals) :]  # rows of Vt past the singular values
        nullspace_proj = null_basis.T @ (null_basis @ cin_original)

        valid_mask = ~np.isnan(cin_reconstructed)
        valid_indices = np.where(valid_mask)[0]
        n_skip = max(10, int(0.15 * len(valid_indices)))
        middle_indices = valid_indices[n_skip:-n_skip]

        # Per-bin tolerance: a small numerical safety factor times the
        # theoretical nullspace bound (plus a tiny absolute floor for bins
        # with negligible nullspace energy where round-off dominates).
        bound = 2.0 * np.abs(nullspace_proj[middle_indices]) + 1e-12
        actual_err = np.abs(cin_reconstructed[middle_indices] - cin_original[middle_indices])
        assert np.all(actual_err <= bound), (
            f"Reconstruction error exceeds nullspace bound: max ratio = {(actual_err / bound).max():.3f}"
        )


# ============================================================================
