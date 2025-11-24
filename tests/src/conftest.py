"""
Shared pytest fixtures for advection tests.

This module provides common fixtures used across multiple test files to reduce
code duplication and improve test maintainability.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.utils import compute_time_edges

# ============================================================================
# Time Series Fixtures
# ============================================================================


@pytest.fixture
def standard_dates():
    """Return standard date range for testing (1 year)."""
    return pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")


@pytest.fixture
def short_dates():
    """Short date range for quick tests (30 days)."""
    return pd.date_range(start="2020-01-01", periods=30, freq="D")


@pytest.fixture
def tedges_standard(standard_dates):
    """Time edges for standard date range."""
    return compute_time_edges(tedges=None, tstart=None, tend=standard_dates, number_of_bins=len(standard_dates))


@pytest.fixture
def tedges_short(short_dates):
    """Time edges for short date range."""
    return compute_time_edges(tedges=None, tstart=None, tend=short_dates, number_of_bins=len(short_dates))


# ============================================================================
# Flow Fixtures
# ============================================================================


@pytest.fixture
def constant_flow_standard(standard_dates):
    """Constant flow of 100 m³/day for standard period."""
    return np.full(len(standard_dates), 100.0)


@pytest.fixture
def constant_flow_short(short_dates):
    """Constant flow of 100 m³/day for short period."""
    return np.full(len(short_dates), 100.0)


# ============================================================================
# Concentration Input Fixtures
# ============================================================================


@pytest.fixture
def constant_concentration():
    """Create factory for constant concentration inputs."""

    def _make_concentration(n_days, value=10.0):
        return np.full(n_days, value)

    return _make_concentration


@pytest.fixture
def gaussian_pulse():
    """Create factory for Gaussian pulse concentration inputs."""

    def _make_pulse(n_days, peak=50.0, center=None, sigma=10.0):
        if center is None:
            center = n_days / 2
        t = np.arange(n_days)
        return peak * np.exp(-0.5 * ((t - center) / sigma) ** 2)

    return _make_pulse


@pytest.fixture
def step_function():
    """Create factory for step function concentration inputs."""

    def _make_step(n_days, level_1=0.0, level_2=50.0, step_day=None):
        if step_day is None:
            step_day = n_days // 2
        cin = np.full(n_days, level_1)
        cin[step_day:] = level_2
        return cin

    return _make_step


@pytest.fixture
def sine_wave():
    """Create factory for sinusoidal concentration inputs."""

    def _make_sine(n_days, mean=30.0, amplitude=20.0, period=40.0):
        t = np.arange(n_days)
        return mean + amplitude * np.sin(2 * np.pi * t / period)

    return _make_sine


# ============================================================================
# Sorption Parameter Fixtures
# ============================================================================


@pytest.fixture
def standard_freundlich_params():
    """Return standard Freundlich parameters for testing.

    These parameters give moderate nonlinearity suitable for most tests.
    """
    return {
        "freundlich_k": 0.001,  # Reduced from typical to avoid extreme retardation
        "freundlich_n": 0.75,  # Favorable sorption (n < 1)
        "bulk_density": 1600.0,  # kg/m³
        "porosity": 0.35,
    }


@pytest.fixture
def linear_sorption_params():
    """Linear sorption parameters (n = 1)."""
    return {
        "freundlich_k": 0.02,
        "freundlich_n": 1.0,  # Linear isotherm
        "bulk_density": 1600.0,
        "porosity": 0.35,
    }


@pytest.fixture
def no_sorption_params():
    """No sorption parameters (K_f = 0, R = 1)."""
    return {
        "freundlich_k": 0.0,  # No sorption
        "freundlich_n": 0.75,  # n doesn't matter when K_f = 0
        "bulk_density": 1600.0,
        "porosity": 0.35,
    }


# ============================================================================
# Pore Volume Fixtures
# ============================================================================


@pytest.fixture
def small_pore_volume():
    """Small pore volume for quick breakthrough (~1 day residence time)."""
    return np.array([100.0])  # 100 m³ / 100 m³/day = 1 day


@pytest.fixture
def medium_pore_volume():
    """Medium pore volume for moderate residence time (~5 days)."""
    return np.array([500.0])  # 500 m³ / 100 m³/day = 5 days


@pytest.fixture
def large_pore_volume():
    """Large pore volume for delayed breakthrough (~10 days)."""
    return np.array([1000.0])  # 1000 m³ / 100 m³/day = 10 days


@pytest.fixture
def multiple_pore_volumes():
    """Multiple pore volumes for heterogeneous aquifer."""
    return np.array([300.0, 500.0, 700.0])  # 3, 5, 7 day residence times


# ============================================================================
# Composite Fixtures for Common Test Scenarios
# ============================================================================


@pytest.fixture
def simple_transport_scenario(tedges_short, constant_flow_short, small_pore_volume):
    """Complete scenario for simple transport test.

    Returns
    -------
        dict: Contains tedges, flow, pore_volumes for quick test
    """
    return {
        "tedges": tedges_short,
        "flow": constant_flow_short,
        "pore_volumes": small_pore_volume,
    }


# ============================================================================
# Parametrize Helpers
# ============================================================================


def pytest_generate_tests(metafunc):
    """
    Dynamic test generation based on fixtures.

    This allows for parameterizing tests across different sorption types,
    input types, etc. without explicit parametrize decorators in every test.
    """
    # Example: If test function has 'all_sorption_params' parameter,
    # automatically parametrize it across all sorption types
    if "all_sorption_params" in metafunc.fixturenames:
        metafunc.parametrize(
            "all_sorption_params",
            [
                pytest.param(
                    {"freundlich_k": 0.0, "freundlich_n": 0.75, "bulk_density": 1600.0, "porosity": 0.35},
                    id="no_sorption",
                ),
                pytest.param(
                    {"freundlich_k": 0.02, "freundlich_n": 1.0, "bulk_density": 1600.0, "porosity": 0.35},
                    id="linear_sorption",
                ),
                pytest.param(
                    {"freundlich_k": 0.001, "freundlich_n": 0.75, "bulk_density": 1600.0, "porosity": 0.35},
                    id="nonlinear_favorable",
                ),
            ],
        )


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Fast unit tests (< 0.1 seconds)")
    config.addinivalue_line("markers", "integration: Integration tests (< 1 second)")
    config.addinivalue_line("markers", "slow: Slow tests (> 1 second)")
    config.addinivalue_line("markers", "exact: Tests using exact solvers")
    config.addinivalue_line("markers", "numerical: Tests using numerical methods")
    config.addinivalue_line("markers", "analytical: Tests against analytical solutions")
    config.addinivalue_line("markers", "roundtrip: Roundtrip reconstruction tests")
