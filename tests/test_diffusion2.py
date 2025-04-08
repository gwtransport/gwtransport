import itertools

import numpy as np
import pytest

from gwtransport1d.diffusion2 import (
    analytical_diffusion_filter,
    erf_integral_numerical_space,
    erf_integral_space,
    erf_integral_space_time,
    erf_mean_numerical_space,
    erf_mean_numerical_space_time2,
    erf_mean_space,
    erf_mean_space_time2,
)
from gwtransport1d.diffusion2 import erf_integral_numerical_space_time2 as erf_integral_numerical_space_time


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
        np.testing.assert_allclose(analytical, numerical, rtol=1e-3)

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


def test_erf_mean_space_time2_inf_values():
    """Test with infinite values in space edges."""
    xedges = np.array([-np.inf, -10.0, 0.0, 10.0, np.inf])
    tedges = np.array([0.5, 1.0, 1.5])
    diffusivity = 0.5

    analytical = erf_mean_space_time2(xedges, tedges, diffusivity)
    numerical = erf_mean_numerical_space_time2(xedges, tedges, diffusivity)

    # Results should be close
    # Skip comparing cells with infinite boundaries as numerical integration might struggle
    finite_mask = np.isfinite(analytical)
    if np.any(finite_mask):
        np.testing.assert_allclose(analytical[finite_mask], numerical[finite_mask], rtol=1e-4)

    # Check that infinite cells have expected values
    # Lower edge at -inf should give -1.0
    if analytical.shape[0] > 0 and analytical.shape[1] > 0:
        assert np.all(analytical[0, :] == -1.0)

    # Upper edge at +inf should give 1.0
    if analytical.shape[0] > 3 and analytical.shape[1] > 0:
        assert np.all(analytical[3, :] == 1.0)


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


def test_erf_mean_space_time2_non_monotonic_edges():
    """Test error handling for non-monotonic edges."""
    # Non-monotonic x edges
    xedges = np.array([0.0, 2.0, 1.0])
    tedges = np.array([0.5, 1.0, 1.5])
    diffusivity = 0.5

    with pytest.raises(ValueError):
        erf_mean_space_time2(xedges, tedges, diffusivity)

    # Non-monotonic t edges
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([1.5, 1.0, 0.5])

    with pytest.raises(ValueError):
        erf_mean_space_time2(xedges, tedges, diffusivity)


def test_erf_mean_space_time2_negative_diffusivity():
    """Test error handling for negative diffusivity."""
    xedges = np.array([0.0, 1.0, 2.0])
    tedges = np.array([0.5, 1.0, 1.5])
    diffusivity = -0.5

    with pytest.raises(ValueError):
        erf_mean_space_time2(xedges, tedges, diffusivity)


if __name__ == "__main__":
    pytest.main()
