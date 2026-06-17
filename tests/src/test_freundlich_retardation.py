"""Tests for ``freundlich_retardation`` -- concentration-dependent retardation from a Freundlich isotherm.

Relocated from the deleted ``test_residence_time_series.py`` (the point-sampler that file tested was
removed; ``freundlich_retardation`` itself is unchanged).
"""

import numpy as np
import pytest

from gwtransport.residence_time import freundlich_retardation


def test_freundlich_retardation_analytical():
    """Test freundlich_retardation against hand-computed analytical values.

    R = 1 + (rho_b / theta) * k_f * (1/n) * C^(1/n - 1)
    """
    concentration = np.array([1.0])
    k_f = 0.5
    n = 0.8
    rho_b = 1500.0
    theta = 0.3

    result = freundlich_retardation(
        concentration=concentration,
        freundlich_k=k_f,
        freundlich_n=n,
        bulk_density=rho_b,
        porosity=theta,
    )

    # R = 1 + (1500/0.3) * 0.5 * (1/0.8) * 1.0^(1/0.8 - 1) = 1 + 3125
    expected = 1.0 + (rho_b / theta) * k_f * (1.0 / n) * np.power(1.0, 1.0 / n - 1.0)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_freundlich_retardation_concentration_dependence():
    """Test that freundlich_retardation varies correctly with concentration."""
    concentrations = np.array([0.1, 1.0, 10.0])
    k_f = 0.01
    n = 0.7
    rho_b = 1600.0
    theta = 0.35

    result = freundlich_retardation(
        concentration=concentrations,
        freundlich_k=k_f,
        freundlich_n=n,
        bulk_density=rho_b,
        porosity=theta,
    )

    # For n < 1 the exponent 1/n - 1 > 0, so retardation increases with concentration
    assert result[0] < result[1] < result[2]

    # Check exact values
    expected = 1.0 + (rho_b / theta) * k_f * (1.0 / n) * np.power(concentrations, 1.0 / n - 1.0)
    np.testing.assert_allclose(result, expected, rtol=1e-13)


def test_freundlich_retardation_zero_concentration_n_gt_one_raises():
    """For n > 1 the exponent 1/n - 1 < 0, so R diverges as C -> 0; non-positive C must raise."""
    # Zero concentration with n > 1: must raise
    with pytest.raises(ValueError, match="concentration must be strictly positive when freundlich_n > 1"):
        freundlich_retardation(
            concentration=np.array([0.0, 1.0]),
            freundlich_k=0.5,
            freundlich_n=1.5,
            bulk_density=1500.0,
            porosity=0.3,
        )

    # Negative concentration with n > 1: must raise
    with pytest.raises(ValueError, match="concentration must be strictly positive when freundlich_n > 1"):
        freundlich_retardation(
            concentration=np.array([-0.1, 1.0]),
            freundlich_k=0.5,
            freundlich_n=1.5,
            bulk_density=1500.0,
            porosity=0.3,
        )


def test_freundlich_retardation_zero_concentration_n_leq_one_allowed():
    """For n <= 1 the retardation factor is finite at C = 0 (or constant for n = 1); must not raise."""
    # n = 1 -> retardation factor independent of C
    result = freundlich_retardation(
        concentration=np.array([0.0, 1.0, 2.0]),
        freundlich_k=0.5,
        freundlich_n=1.0,
        bulk_density=1500.0,
        porosity=0.3,
    )
    expected_constant = 1.0 + (1500.0 / 0.3) * 0.5 * 1.0
    np.testing.assert_allclose(result, expected_constant, rtol=1e-12)

    # n < 1 -> exponent 1/n - 1 > 0, so R equals 1 at C = 0
    result = freundlich_retardation(
        concentration=np.array([0.0, 1.0]),
        freundlich_k=0.5,
        freundlich_n=0.7,
        bulk_density=1500.0,
        porosity=0.3,
    )
    np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)
