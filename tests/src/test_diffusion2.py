import itertools

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction as advection_i2e
from gwtransport.diffusion2 import (
    analytical_diffusion_filter,
    erf_integral_numerical_space,
    erf_integral_space,
    erf_integral_space_time,
    erf_mean_numerical_space,
    erf_mean_numerical_space_time2,
    erf_mean_space,
    erf_mean_space_time,
    erf_mean_space_time2,
    infiltration_to_extraction,
)
from gwtransport.diffusion2 import erf_integral_numerical_space_time2 as erf_integral_numerical_space_time


def test_create_test_temperature_mass_conservation():
    def create_test_temperature(n, xlim=30):
        temperature = np.zeros(n)
        temperature[n // 2 - 1 : n // 2 + 1] = 1.0
        xedges = np.linspace(-xlim, xlim, n + 1)
        dxedges = np.diff(xedges, prepend=0)
        dxedges[n // 2] *= 3
        xedges = np.cumsum(dxedges)
        return xedges, temperature

    # Test zero diffusivity case
    xedges, temperature = create_test_temperature(10)
    yout = analytical_diffusion_filter(input_signal=temperature, xedges=xedges, diffusivity=0.0, times=5.0)
    np.testing.assert_allclose(temperature, yout, rtol=1e-16)

    for n in [10, 100, 1000]:
        xedges, temperature = create_test_temperature(n)
        # Compute cell widths
        dx = np.diff(xedges)
        # Compute the initial mass as the sum over each cell: temperature * cell width.
        mass_in = np.sum(temperature * dx)
        # Apply diffusion with a fixed diffusivity and time.
        yout = analytical_diffusion_filter(input_signal=temperature, xedges=xedges, diffusivity=0.1, times=5.0)
        # Compute the mass after diffusion.
        mass_out = np.sum(yout * dx)
        np.testing.assert_allclose(mass_in, mass_out, rtol=1e-15)


def test_erf_integral_space():
    # Test comparison between analytical and numerical solutions
    x_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    a_values = np.array([0.5, 1.0, 2.0])

    for a in a_values:
        analytical = erf_integral_space(x_values, a)
        numerical = erf_integral_numerical_space(x_values, a)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-14)

    # Test array input for scaling factor a
    x = np.array([0.0, 1.0, 2.0])
    a = 3.15
    analytical = erf_integral_space(x, a)
    numerical = erf_integral_numerical_space(x, a)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-14)

    # Test clipping for large values
    x_large = np.array([-20000.0, -10000.0, 10000.0, 20000.0])
    analytical = erf_integral_space(x_large)
    # For x -> -inf: -x - 1/(a*sqrt(pi))
    # For x -> +inf: x - 1/(a*sqrt(pi))
    expected = np.where(x_large < 0, -x_large - 1 / np.sqrt(np.pi), x_large - 1 / np.sqrt(np.pi))
    np.testing.assert_allclose(analytical, expected, rtol=1e-14)

    # test limits
    x_limits = np.array([-np.inf, -np.inf, -1.0, 0.0, 1.0, np.inf, np.inf])
    analytical = erf_integral_space(x_limits)
    numerical = erf_integral_numerical_space(x_limits)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-14)
    assert np.all(np.isposinf(analytical[[0, -1]])), "Limits should be positive infinite"

    # test NaN values
    x_nan = np.array([-1.0, 0.0, np.nan, 1.0])
    analytical = erf_integral_space(x_nan)
    numerical = erf_integral_numerical_space(x_nan)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-14)
    assert np.isnan(analytical[2]), "NaN value should be preserved"


def test_erf_mean_space():
    # Test comparison between analytical and numerical solutions
    edges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    a_values = np.array([-0.5, 0.5, 1.0, 2.0])

    for a in a_values:
        analytical = erf_mean_space(edges, a=a)
        numerical = erf_mean_numerical_space(edges, a=a)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-14)

    # Test array input for scaling factor a
    edges = np.array([0.0, 1.0, 2.0])
    a = 3.5
    analytical = erf_mean_space(edges, a)
    numerical = erf_mean_numerical_space(edges, a)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-14)

    # Test edge cases with infinite values
    edges_inf = np.array([-np.inf, -np.inf, -1.0, 0.0, 1.0, np.inf, np.inf])
    a = 1.0
    analytical = erf_mean_space(edges_inf, a=a)
    numerical = erf_mean_numerical_space(edges_inf, a=a)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-14)

    # Test very large but finite values
    edges_large = np.array([-1e10, -1.0, 0.0, 1.0, 1e10])
    analytical = erf_mean_space(edges_large)
    numerical = erf_mean_numerical_space(edges_large, a)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-6)


def test_erf_integral_space_time():
    # Test comparison between analytical and numerical solutions for different parameter combinations
    x_values = np.array([0.01, 1.0, 2.0, 3.0])
    t_values = np.array([0.5, 1.0, 2.0])
    diffusivity_values = np.array([0.1, 0.5, 1.0])

    for x, t, diffusivity in itertools.product(x_values, t_values, diffusivity_values):
        analytical = erf_integral_space_time(x, t, diffusivity)
        numerical = erf_integral_numerical_space_time(x, t, diffusivity)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    # Test broadcasting
    x = np.array([1.0, 2.0, 3.0])
    t = np.array([0.5, 1.0])
    diffusivity = 0.5
    analytical = erf_integral_space_time(x, t, diffusivity)
    numerical = erf_integral_numerical_space_time(x, t, diffusivity)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-6)

    # Test behavior at x=0
    x = np.array([0.0, 1.0, 1.0])
    t = np.array([1.0, 0.0, 1.0])
    analytical = erf_integral_space_time(x=x, t=t, diffusivity=0.5)
    numerical = erf_integral_numerical_space_time(x=x, t=t, diffusivity=0.5)
    np.testing.assert_allclose(analytical, numerical, atol=1e-10)

    # Test behavior for small time values
    x = np.array([1.0, 2.0])
    t = 1e-4
    diffusivity = 0.5
    analytical = erf_integral_space_time(x, t, diffusivity)
    numerical = erf_integral_numerical_space_time(x, t, diffusivity)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-3)

    # Test behavior for small time values
    x = np.array([1.0, 2.0])
    t = 0.0
    diffusivity = 0.5
    analytical = erf_integral_space_time(x, t, diffusivity)
    numerical = erf_integral_numerical_space_time(x, t, diffusivity)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4, atol=1e-10)

    # Test behavior for large x values
    x = np.array([1000.0, 2000.0])
    t = 1.0
    diffusivity = 0.5
    analytical = erf_integral_space_time(x, t, diffusivity)
    numerical = erf_integral_numerical_space_time(x, t, diffusivity)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4)

    # Test infinite values
    x_inf = np.array([np.inf, -np.inf])
    t = 1.0
    diffusivity = 0.5
    analytical = erf_integral_space_time(x_inf, t, diffusivity)
    numerical = erf_integral_numerical_space_time(x_inf, t, diffusivity)
    assert np.all(np.isinf(analytical))
    assert np.all(np.isinf(numerical))

    # Test NaN handling
    x_nan = np.array([1.0, np.nan, 3.0])
    t = np.array([0.1, 0.5, 1.0, np.nan])
    diffusivity = 0.5
    analytical = erf_integral_space_time(x_nan, t, diffusivity)
    numerical = erf_integral_numerical_space_time(x_nan, t, diffusivity)
    assert np.all(np.isnan(analytical) == np.isnan(numerical))
    assert np.all(np.isnan(analytical[np.isnan(x_nan) | np.isnan(t[:, None])]))
    assert ~np.any(np.isnan(analytical[~(np.isnan(x_nan) | np.isnan(t[:, None]))]))


def test_erf_mean_space_time2_basic():
    """Test basic functionality of erf_mean_space_time2 vs numerical implementation."""
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([0.5, 1.0, 1.5])
    diffusivity = 0.5

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Both should return a 2x2 array
    assert analytical.shape == (2, 2)
    assert numerical.shape == (2, 2)

    # Results should be close
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5)


def test_erf_mean_space_time2_various_parameters():
    """Test with various combinations of parameters."""
    xedges_list = [
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([-1.0, 0.0, 1.0]),
    ]

    tedges_list = [
        np.array([0.1, 0.5, 1.0]),
        np.array([0.5, 1.0, 1.5, 2.0]),
        np.array([1.0, 2.0, 3.0]),
    ]

    diffusivity_values = [0.1, 0.5, 1.0]

    for xedges, tedges, diffusivity in itertools.product(xedges_list, tedges_list, diffusivity_values):
        analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
        numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

        # Results should be close
        np.testing.assert_allclose(
            analytical,
            numerical,
            rtol=1e-4,
            err_msg=f"Failed with xedges={xedges}, tedges={tedges}, diffusivity={diffusivity}",
        )


def test_erf_mean_space_time2_scalar_outputs():
    """Test scalar output cases."""
    # Single cell case (1x1 output)
    xedges = np.array([0.0, 1.0])
    tedges = np.array([0.5, 1.0])
    diffusivity = 0.5

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Should be scalar or 1x1
    assert np.isscalar(analytical) or analytical.shape == (1, 1)
    assert np.isscalar(numerical) or numerical.shape == (1, 1)

    # Results should be close
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5)

    # 1D array cases
    # Single x cell, multiple t cells
    xedges = np.array([0.0, 1.0])
    tedges = np.array([0.5, 1.0, 1.5])

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Should be 1D arrays
    assert analytical.shape == (2,)
    assert numerical.shape == (2,)

    # Results should be close
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5)

    # Multiple x cells, single t cell
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([0.5, 1.0])

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Should be 1D arrays
    assert analytical.shape == (2,)
    assert numerical.shape == (2,)

    # Results should be close
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5)


@pytest.mark.skip(reason="Zero width cell averaging not implemented yet")
def test_erf_mean_space_time2_zero_width_cells():
    """Test edge cases including zero width cells and cells at time/space extremes."""
    # Zero width cells in space
    xedges = np.array([0.0, 0.0, 1.0, 2.0])
    tedges = np.array([0.5, 1.0, 1.5])
    diffusivity = 0.5

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Results should be close
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5)

    # Zero width cells in time
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([0.5, 0.5, 1.0, 1.5])

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Results should be close
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5)


def test_erf_mean_space_time2_edge_cases2():
    diffusivity = 0.5

    # Zero time limit
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([0.0, 0.5, 1.0])

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Results should be close
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5)

    # Very small time differences
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([0.0001, 0.0002, 0.0003])

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Results should be close (with higher tolerance for numerical precision issues)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-3)


def test_erf_mean_space_time2_large_values():
    """Test with large values for space and time."""
    # Large space values
    xedges = np.array([1000.0, 2000.0, 3000.0])
    tedges = np.array([0.5, 1.0, 1.5])
    diffusivity = 0.5

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Results should be close (with higher tolerance)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3)

    # Large time values
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([100.0, 200.0, 300.0])

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Results should be close (with higher tolerance)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


def test_erf_mean_space_time2_negative_values():
    """Test with negative x values."""
    xedges = np.array([-2.0, -1.0, 0.0, 1.0])
    tedges = np.array([0.5, 1.0, 1.5])
    diffusivity = 0.5

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Results should be close
    np.testing.assert_allclose(analytical, numerical, rtol=1e-4)


def test_erf_mean_space_time2_nan_handling():
    """Test handling of NaN values in inputs."""
    xedges = np.array([0.0, 1.0, np.nan, 3.0])
    tedges = np.array([0.5, np.nan, 1.5])
    diffusivity = 0.5

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Check that NaN pattern is consistent
    assert np.all(np.isnan(analytical) == np.isnan(numerical))

    # Check non-NaN values are close
    valid_mask = ~np.isnan(analytical)
    if np.any(valid_mask):
        np.testing.assert_allclose(analytical[valid_mask], numerical[valid_mask], rtol=1e-4)


def test_erf_mean_space_time2_different_diffusivities():
    """Test with various diffusivity values, including very small and large."""
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([0.5, 1.0, 1.5])

    # Test with very small diffusivity
    diffusivity = 1e-4
    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-3)

    # Test with very large diffusivity
    diffusivity = 1e4
    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-3)


# =============================================================================
# Tests comparing erf_mean_space_time (analytical) vs erf_mean_space_time2 (numerical)
# =============================================================================


class TestErfMeanSpaceTimeConsistency:
    """Test that erf_mean_space_time and erf_mean_space_time2 produce consistent results.

    erf_mean_space_time uses the analytical double integral solution via erf_integral_space_time.
    erf_mean_space_time2 uses numerical integration via erf_integral_numerical_space_time2.
    Both should produce the same results within numerical tolerance.
    """

    def test_basic_case(self):
        """Test basic functionality with simple inputs."""
        xedges = np.array([0.0, 1.0, 2.0])
        tedges = np.array([0.5, 1.0, 1.5])
        diffusivity = 0.5

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        assert analytical.shape == numerical.shape == (2, 2)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5)

    def test_single_cell(self):
        """Test scalar output for single space-time cell."""
        xedges = np.array([0.0, 1.0])
        tedges = np.array([0.5, 1.0])
        diffusivity = 0.5

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        # Both should return scalar
        assert np.isscalar(analytical)
        assert np.isscalar(numerical)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5)

    def test_single_time_multiple_space(self):
        """Test 1D output for single time cell, multiple space cells."""
        xedges = np.array([0.0, 1.0, 2.0, 3.0])
        tedges = np.array([0.5, 1.0])
        diffusivity = 0.5

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        assert analytical.shape == numerical.shape == (3,)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5)

    def test_single_space_multiple_time(self):
        """Test 1D output for multiple time cells, single space cell."""
        xedges = np.array([0.0, 1.0])
        tedges = np.array([0.5, 1.0, 1.5, 2.0])
        diffusivity = 0.5

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        assert analytical.shape == numerical.shape == (3,)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5)

    def test_negative_x_values(self):
        """Test with negative x values (concentration front behind observation point)."""
        xedges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        tedges = np.array([0.5, 1.0, 2.0])
        diffusivity = 0.5

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-4)

    def test_various_diffusivities(self):
        """Test consistency across a range of diffusivity values."""
        xedges = np.array([0.0, 1.0, 2.0])
        tedges = np.array([0.5, 1.0, 1.5])

        for diffusivity in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            analytical = erf_mean_space_time(xedges, tedges, diffusivity)
            numerical = erf_mean_space_time2(xedges, tedges, diffusivity)
            np.testing.assert_allclose(
                analytical,
                numerical,
                rtol=1e-4,
                err_msg=f"Mismatch for diffusivity={diffusivity}",
            )

    def test_various_grid_sizes(self):
        """Test consistency across different grid resolutions."""
        diffusivity = 0.5

        for n_x in [2, 5, 10]:
            for n_t in [2, 5, 10]:
                xedges = np.linspace(0.0, 3.0, n_x + 1)
                tedges = np.linspace(0.5, 2.0, n_t + 1)

                analytical = erf_mean_space_time(xedges, tedges, diffusivity)
                numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

                np.testing.assert_allclose(
                    analytical,
                    numerical,
                    rtol=1e-4,
                    err_msg=f"Mismatch for n_x={n_x}, n_t={n_t}",
                )

    def test_asymmetric_grid(self):
        """Test with non-uniform grid spacing."""
        xedges = np.array([0.0, 0.1, 0.5, 2.0, 5.0])
        tedges = np.array([0.1, 0.2, 1.0, 5.0])
        diffusivity = 0.3

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-4)

    def test_large_time_values(self):
        """Test with large time values (more diffusion spreading)."""
        xedges = np.array([0.0, 1.0, 2.0])
        tedges = np.array([10.0, 50.0, 100.0])
        diffusivity = 0.5

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-4)

    def test_small_time_values(self):
        """Test with small time values (less diffusion spreading)."""
        xedges = np.array([0.0, 1.0, 2.0])
        tedges = np.array([0.01, 0.05, 0.1])
        diffusivity = 0.5

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-3, atol=1e-3)

    def test_time_starting_at_zero(self):
        """Test when time edges start at zero."""
        xedges = np.array([0.0, 1.0, 2.0])
        tedges = np.array([0.0, 0.5, 1.0])
        diffusivity = 0.5

        analytical = erf_mean_space_time(xedges, tedges, diffusivity)
        numerical = erf_mean_space_time2(xedges, tedges, diffusivity)

        np.testing.assert_allclose(analytical, numerical, rtol=1e-4)


class TestErfMeanSpaceTimePhysics:
    """Test physical properties of erf_mean_space_time.

    These tests verify that the function behaves correctly from a physical
    perspective, independent of the numerical implementation.
    """

    def test_symmetry_around_zero(self):
        """Test that erf is antisymmetric: erf(-x) = -erf(x).

        For symmetric x edges around zero, the average over [-a, 0] should be
        the negative of the average over [0, a].
        """
        xedges = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        tedges = np.array([0.5, 1.0])
        diffusivity = 0.5

        result = erf_mean_space_time(xedges, tedges, diffusivity)
        # result has shape (1, 4) -> squeezed to (4,)
        # Cells: [-2,-1], [-1,0], [0,1], [1,2]
        # Due to antisymmetry: avg[-2,-1] ≈ -avg[1,2] and avg[-1,0] ≈ -avg[0,1]
        np.testing.assert_allclose(result[0], -result[3], rtol=1e-10)
        np.testing.assert_allclose(result[1], -result[2], rtol=1e-10)

    def test_bounded_by_minus_one_and_one(self):
        """Test that erf mean values are bounded by [-1, 1]."""
        xedges = np.linspace(-5.0, 5.0, 11)
        tedges = np.linspace(0.1, 2.0, 5)
        diffusivity = 0.5

        result = erf_mean_space_time(xedges, tedges, diffusivity)

        assert np.all(result >= -1.0 - 1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    def test_monotonicity_in_space(self):
        """Test that erf mean increases monotonically with x position.

        For fixed time, cells at larger x should have larger mean erf values.
        """
        xedges = np.linspace(0.0, 5.0, 6)
        tedges = np.array([0.5, 1.0])
        diffusivity = 0.5

        result = erf_mean_space_time(xedges, tedges, diffusivity)
        # result should be monotonically increasing along x axis
        assert np.all(np.diff(result) > 0)

    def test_convergence_to_one_for_large_x(self):
        """Test that mean erf approaches 1 for large positive x."""
        xedges = np.array([100.0, 101.0])
        tedges = np.array([0.5, 1.0])
        diffusivity = 0.5

        result = erf_mean_space_time(xedges, tedges, diffusivity)
        np.testing.assert_allclose(result, 1.0, atol=1e-6)

    def test_convergence_to_minus_one_for_large_negative_x(self):
        """Test that mean erf approaches -1 for large negative x."""
        xedges = np.array([-101.0, -100.0])
        tedges = np.array([0.5, 1.0])
        diffusivity = 0.5

        result = erf_mean_space_time(xedges, tedges, diffusivity)
        np.testing.assert_allclose(result, -1.0, atol=1e-6)

    def test_larger_diffusivity_means_more_spreading(self):
        """Test that larger diffusivity leads to values closer to zero.

        With more diffusion, the concentration profile spreads out more,
        so the erf values become more moderate (closer to 0).
        """
        xedges = np.array([1.0, 2.0])
        tedges = np.array([0.5, 1.0])

        result_low_d = erf_mean_space_time(xedges, tedges, diffusivity=0.1)
        result_high_d = erf_mean_space_time(xedges, tedges, diffusivity=10.0)

        # With high diffusivity, the erf values should be closer to 0
        # (more spreading = more mixing = concentration closer to average)
        assert abs(result_high_d) < abs(result_low_d)

    def test_longer_time_means_more_spreading(self):
        """Test that longer time leads to values closer to zero.

        Similar to diffusivity, longer time allows more diffusion.
        """
        xedges = np.array([1.0, 2.0])
        diffusivity = 0.5

        result_short_t = erf_mean_space_time(xedges, np.array([0.1, 0.2]), diffusivity)
        result_long_t = erf_mean_space_time(xedges, np.array([10.0, 20.0]), diffusivity)

        # With longer time, the erf values should be closer to 0
        assert abs(result_long_t) < abs(result_short_t)


# =============================================================================
# Tests for infiltration_to_extraction with diffusion
# =============================================================================


class TestInfiltrationToExtractionDiffusion:
    """Tests for the infiltration_to_extraction function with diffusion.

    These tests verify that the advection-dispersion transport model
    produces physically correct results.
    """

    @pytest.fixture
    def simple_setup(self):
        """Create a simple test setup with constant flow."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-15", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")

        # Step input: concentration 1 for first 5 days
        cin = np.zeros(len(tedges) - 1)
        cin[0:5] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0  # 100 m3/day

        # Single pore volume: 500 m3 = 5 days residence time at 100 m3/day
        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        return {
            "cin": cin,
            "flow": flow,
            "tedges": tedges,
            "cout_tedges": cout_tedges,
            "aquifer_pore_volumes": aquifer_pore_volumes,
            "streamline_length": streamline_length,
        }

    def test_zero_diffusivity_matches_advection(self, simple_setup):
        """Test that zero diffusivity gives same result as pure advection."""
        cout_advection = advection_i2e(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
        )
        cout_diffusion = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            diffusivity=0.0,
        )
        np.testing.assert_allclose(cout_advection, cout_diffusion, equal_nan=True)

    def test_small_diffusivity_close_to_advection(self, simple_setup):
        """Test that small diffusivity produces result close to advection."""
        cout_advection = advection_i2e(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
        )
        cout_diffusion = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            diffusivity=0.01,
        )
        # Should be close but not identical
        valid = ~np.isnan(cout_advection) & ~np.isnan(cout_diffusion)
        np.testing.assert_allclose(cout_advection[valid], cout_diffusion[valid], rtol=0.1)

    def test_larger_diffusivity_more_spreading(self, simple_setup):
        """Test that larger diffusivity causes more spreading."""
        cout_small_d = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            diffusivity=0.1,
            retardation_factor=2.0
        )
        cout_large_d = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            diffusivity=10.0,
        )
        # With larger diffusivity, the breakthrough curve should be more spread out
        # This means the maximum should be lower and the tails should be higher
        valid = ~np.isnan(cout_small_d) & ~np.isnan(cout_large_d)
        max_small = np.max(cout_small_d[valid])
        max_large = np.max(cout_large_d[valid])
        assert max_large <= max_small  # More spreading = lower peak

    def test_output_bounded_by_input(self, simple_setup):
        """Test that output concentrations are bounded by input range."""
        cout = infiltration_to_extraction(
            cin=simple_setup["cin"],
            flow=simple_setup["flow"],
            tedges=simple_setup["tedges"],
            cout_tedges=simple_setup["cout_tedges"],
            aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
            streamline_length=simple_setup["streamline_length"],
            diffusivity=1.0,
        )
        valid = ~np.isnan(cout)
        # Output should be between min and max of input (plus small tolerance for numerics)
        assert np.all(cout[valid] >= np.min(simple_setup["cin"]) - 1e-10)
        assert np.all(cout[valid] <= np.max(simple_setup["cin"]) + 1e-10)

    def test_constant_input_gives_constant_output(self):
        """Test that constant input concentration gives constant output."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
        cout_tedges = pd.date_range(start="2020-01-06", end="2020-01-15", freq="D")

        cin = np.ones(len(tedges) - 1) * 5.0  # Constant concentration
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            diffusivity=1.0,
        )

        valid = ~np.isnan(cout)
        # With constant input, output should also be constant (after spin-up)
        np.testing.assert_allclose(cout[valid], 5.0, rtol=1e-3)

    def test_multiple_pore_volumes(self):
        """Test with multiple pore volumes (heterogeneous aquifer)."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-15", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")

        cin = np.zeros(len(tedges) - 1)
        cin[0:5] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0

        # Multiple pore volumes with corresponding travel distances
        aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
        streamline_length = np.array([80.0, 100.0, 120.0])

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            diffusivity=1.0,
        )

        # Should produce valid output
        assert cout.shape == (len(cout_tedges) - 1,)
        valid = ~np.isnan(cout)
        assert np.sum(valid) > 0

        # Output should be bounded
        assert np.all(cout[valid] >= 0.0 - 1e-10)
        assert np.all(cout[valid] <= 1.0 + 1e-10)

    def test_input_validation(self, simple_setup):
        """Test that invalid inputs raise appropriate errors."""
        # Negative diffusivity
        with pytest.raises(ValueError, match="diffusivity must be non-negative"):
            infiltration_to_extraction(
                cin=simple_setup["cin"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                diffusivity=-1.0,
            )

        # Mismatched pore volumes and travel distances
        with pytest.raises(ValueError, match="same length"):
            infiltration_to_extraction(
                cin=simple_setup["cin"],
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=np.array([500.0, 600.0]),
                streamline_length=np.array([100.0]),
                diffusivity=1.0,
            )

        # NaN in cin
        cin_with_nan = simple_setup["cin"].copy()
        cin_with_nan[2] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            infiltration_to_extraction(
                cin=cin_with_nan,
                flow=simple_setup["flow"],
                tedges=simple_setup["tedges"],
                cout_tedges=simple_setup["cout_tedges"],
                aquifer_pore_volumes=simple_setup["aquifer_pore_volumes"],
                streamline_length=simple_setup["streamline_length"],
                diffusivity=1.0,
            )


class TestInfiltrationToExtractionDiffusionPhysics:
    """Physics-based tests for infiltration_to_extraction with diffusion."""

    def test_symmetry_of_pulse(self):
        """Test that a symmetric pulse input produces a symmetric-ish output.

        Note: Perfect symmetry is not expected due to the nature of diffusion
        in a flowing system, but the output should be roughly centered around
        the expected arrival time.
        """
        tedges = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")

        # Narrow pulse in the middle
        cin = np.zeros(len(tedges) - 1)
        cin[10:12] = 1.0  # 2-day pulse starting day 10
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])  # 5 days residence time
        streamline_length = np.array([100.0])

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            diffusivity=5.0,
        )

        valid = ~np.isnan(cout)
        if np.sum(valid) > 0:
            # The center of mass should be around day 15-17 (10-12 + 5 days residence)
            times = np.arange(len(cout))
            cout_valid = cout.copy()
            cout_valid[~valid] = 0
            if np.sum(cout_valid) > 0:
                center_of_mass = np.sum(times * cout_valid) / np.sum(cout_valid)
                # Center should be around day 16 (midpoint of input + residence time)
                assert 14 < center_of_mass < 19

    def test_mass_approximately_conserved(self):
        """Test that mass is approximately conserved through transport."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-30", freq="D")

        cin = np.zeros(len(tedges) - 1)
        cin[2:7] = 1.0  # 5-day pulse
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        cout = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            diffusivity=1.0,
        )

        # Mass in = sum of cin (each bin is 1 day)
        mass_in = np.sum(cin)

        # Mass out = sum of cout (excluding NaN)
        mass_out = np.nansum(cout)

        # Mass should be approximately conserved (within 20% for this test)
        # Some loss is expected due to boundary effects
        assert abs(mass_out - mass_in) / mass_in < 0.2

    def test_retardation_delays_breakthrough(self):
        """Test that retardation factor delays the breakthrough."""
        tedges = pd.date_range(start="2020-01-01", end="2020-01-15", freq="D")
        cout_tedges = pd.date_range(start="2020-01-01", end="2020-01-25", freq="D")

        cin = np.zeros(len(tedges) - 1)
        cin[0:3] = 1.0
        flow = np.ones(len(tedges) - 1) * 100.0

        aquifer_pore_volumes = np.array([500.0])
        streamline_length = np.array([100.0])

        # Without retardation
        cout_r1 = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            diffusivity=1.0,
            retardation_factor=1.0,
        )

        # With retardation
        cout_r2 = infiltration_to_extraction(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            diffusivity=1.0,
            retardation_factor=2.0,
        )

        # Find first significant arrival (>0.1)
        def first_arrival(cout):
            valid = ~np.isnan(cout)
            for i, c in enumerate(cout):
                if valid[i] and c > 0.1:
                    return i
            return len(cout)

        arrival_r1 = first_arrival(cout_r1)
        arrival_r2 = first_arrival(cout_r2)

        # Retarded breakthrough should be later
        assert arrival_r2 > arrival_r1
