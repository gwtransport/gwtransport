import itertools

import numpy as np
import pytest

from gwtransport1d.diffusion2 import (
    analytical_diffusion_filter,
    erf_integral_numerical_space,
    erf_integral_space,
    erf_integral_space_time,
    erf_mean_numerical_space,
    erf_mean_space,
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


if __name__ == "__main__":
    pytest.main()
