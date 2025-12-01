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
        "freundlich_n": 0.75,  # n<1 (lower C travels faster)
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
# Aquifer Configuration Fixtures
# ============================================================================


@pytest.fixture
def aquifer_scenarios():
    """Various realistic aquifer configurations for parametrized testing.

    Returns
    -------
    list of dict
        Each dict contains pore_volume, length, porosity, and descriptive name.
        Covers typical, small, and large aquifer systems.
    """
    return [
        {"pore_volume": 30000.0, "length": 80.0, "porosity": 0.35, "name": "typical"},
        {"pore_volume": 10000.0, "length": 50.0, "porosity": 0.25, "name": "small"},
        {"pore_volume": 100000.0, "length": 150.0, "porosity": 0.40, "name": "large"},
    ]


# ============================================================================
# Diffusivity Fixtures
# ============================================================================


@pytest.fixture
def diffusivity_values():
    """Different diffusivity values for various aquifer materials.

    Returns
    -------
    dict
        Maps material type to thermal diffusivity in m²/day.
        Based on typical values for heat transport in saturated porous media.
    """
    return {
        "fine_sand": 0.01,  # Lower end - finer material
        "typical_sand": 0.03,  # Representative value
        "coarse_gravel": 0.08,  # Upper end - coarser material
    }


# ============================================================================
# Retardation Factor Fixtures
# ============================================================================


@pytest.fixture
def retardation_scenarios():
    """Different retardation factors for various transport scenarios.

    Returns
    -------
    dict
        Maps scenario name to retardation factor.
        - Conservative solute: R = 1.0 (no retardation)
        - Temperature: R ~ 2.0 (typical for heat transport)
        - Reactive solute: R > 1.0 (sorbing compound)
    """
    return {
        "conservative_solute": 1.0,
        "temperature": 2.0,
        "reactive_solute": 3.5,
    }


# ============================================================================
# Enhanced Sorption Parameter Fixtures
# ============================================================================


@pytest.fixture
def sorption_parameters():
    """Freundlich sorption parameters for various scenarios.

    Returns
    -------
    list of dict
        Each dict contains n, kf, and descriptive name.
        Covers unfavorable (n<1), linear (n=1), and favorable (n>1) isotherms.
    """
    return [
        {"n": 0.5, "kf": 1.0, "name": "unfavorable"},
        {"n": 1.0, "kf": 1.0, "name": "linear"},
        {"n": 1.5, "kf": 1.0, "name": "favorable"},
        {"n": 2.0, "kf": 1.0, "name": "strongly_favorable"},
    ]


# ============================================================================
# Flow Pattern Fixtures
# ============================================================================


@pytest.fixture
def flow_patterns():
    """Create various flow patterns for testing.

    Returns
    -------
    callable
        Function that generates different flow patterns:
        - constant: Uniform flow
        - seasonal: Sinusoidal variation
        - with_zeros: Includes zero-flow periods
        - step_change: Abrupt flow changes
    """

    def _make_flow_pattern(pattern_type, n_days, base_flow=100.0):
        """Generate flow patterns.

        Parameters
        ----------
        pattern_type : str
            One of: 'constant', 'seasonal', 'with_zeros', 'step_change'
        n_days : int
            Number of days
        base_flow : float
            Base flow rate in m³/day

        Returns
        -------
        np.ndarray
            Flow values for each day
        """
        if pattern_type == "constant":
            return np.full(n_days, base_flow)

        if pattern_type == "seasonal":
            t = np.arange(n_days)
            return base_flow * (1.0 + 0.5 * np.sin(2 * np.pi * t / 365))

        if pattern_type == "with_zeros":
            flow = np.full(n_days, base_flow)
            # Add some zero-flow periods
            flow[10:15] = 0.0
            flow[25:28] = 0.0
            return flow

        if pattern_type == "step_change":
            flow = np.full(n_days, base_flow)
            flow[n_days // 3 :] = base_flow * 1.5
            return flow

        msg = f"Unknown pattern_type: {pattern_type}"
        raise ValueError(msg)

    return _make_flow_pattern


# ============================================================================
# Variable Flow Fixtures
# ============================================================================


@pytest.fixture
def variable_flow_standard(standard_dates):
    """Variable flow with seasonal pattern for standard period."""
    n_days = len(standard_dates)
    t = np.arange(n_days)
    return 100.0 * (1.0 + 0.3 * np.sin(2 * np.pi * t / 365))


@pytest.fixture
def variable_flow_short(short_dates):
    """Variable flow with weekly pattern for short period."""
    n_days = len(short_dates)
    t = np.arange(n_days)
    return 100.0 * (1.0 + 0.2 * np.sin(2 * np.pi * t / 7))


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
                    id="nonlinear_n_lt_1",
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
