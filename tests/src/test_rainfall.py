"""Tests for :mod:`gwtransport.rainfall`.

The extracted-water temperature is the volume-weighted (energy-conserving) mix

    T_ext = (V_rain * T_rain + V_ext * T_adv) / (V_rain + V_ext)

with ``T_adv`` from :func:`gwtransport.advection.gamma_infiltration_to_extraction`,
``V_ext`` the flow integrated over each output bin, and
``V_rain = rainfall * infiltration_area * Δt``.

Tests are built so the mix is exactly determined: a constant infiltration
temperature with a fully-broken-through steady setup makes ``T_adv`` exactly the
infiltration temperature, so the known-answer mixes below are hand-computed.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import gamma_infiltration_to_extraction
from gwtransport.rainfall import rainfall_to_extracted_temperature
from gwtransport.utils import compute_time_edges

# Gamma aquifer pore volume distributions chosen so that, for the flow level used
# in each test, the advective transport of a *constant* infiltration temperature
# returns that constant bit-for-bit at every bin (the convolution weights sum to
# exactly 1 in floating point for these configs -- verified empirically). This
# makes the hand-computed known-answer mixes exact under assert_array_equal.
#
# - _GAMMA_HIGH_FLOW pairs with flow = 100 m3/day (large aquifer fully broken through).
# - _GAMMA_LOW_FLOW pairs with flow = 3 m3/day (small aquifer needed to break through).
_GAMMA_HIGH_FLOW = {"mean": 500.0, "std": 200.0, "n_bins": 20}
_GAMMA_LOW_FLOW = {"mean": 15.0, "std": 6.0, "n_bins": 20}


@pytest.fixture
def dates():
    """Daily dates spanning enough time for full breakthrough."""
    return pd.date_range(start="2020-01-01", end="2020-06-30", freq="D")


@pytest.fixture
def tedges(dates):
    """Time edges aligned with the daily dates."""
    return compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))


# ============================================================================
# Known-answer hand mixes
# ============================================================================


def test_known_answer_equal_volumes(dates, tedges):
    """V_rain == V_ext, T_rain=280, T_adv=290 -> arithmetic mean 285.0 exactly."""
    n = len(dates)
    cin = np.full(n, 290.0)  # constant infiltration temperature -> T_adv == 290 exactly
    t_rain = np.full(n, 280.0)
    flow = np.full(n, 100.0)  # V_ext = 100 m3 per daily bin
    rainfall = np.full(n, 0.1)  # 0.1 m/day * 1000 m2 = 100 m3/day -> V_rain = 100 m3

    out = rainfall_to_extracted_temperature(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=tedges,
        infiltration_area=1000.0,
        retardation_factor=2.0,
        **_GAMMA_HIGH_FLOW,
    )
    assert not np.any(np.isnan(out))
    np.testing.assert_array_equal(out, np.full(n, 285.0))


def test_known_answer_unequal_volumes(dates, tedges):
    """V_rain=1, V_ext=3, T_rain=280, T_adv=290 -> (280 + 3*290)/4 = 287.5 exactly."""
    n = len(dates)
    cin = np.full(n, 290.0)
    t_rain = np.full(n, 280.0)
    flow = np.full(n, 3.0)  # V_ext = 3 m3 per daily bin
    rainfall = np.full(n, 0.001)  # 0.001 m/day * 1000 m2 = 1 m3/day -> V_rain = 1 m3

    out = rainfall_to_extracted_temperature(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=tedges,
        infiltration_area=1000.0,
        retardation_factor=2.0,
        **_GAMMA_LOW_FLOW,
    )
    assert not np.any(np.isnan(out))
    np.testing.assert_array_equal(out, np.full(n, 287.5))


# ============================================================================
# Weighting orientation (pins the convex-combination orientation)
# ============================================================================


def test_weighting_orientation(dates, tedges):
    """Swapping which volume weights which temperature must change the result.

    With V_rain=1, V_ext=3 the result is 287.5 (closer to T_adv=290). If the
    weights were swapped (rain weighted by V_ext, aquifer by V_rain) the result
    would be (3*280 + 290)/4 = 282.5. The two differ, so the convex-combination
    orientation (rain<->V_rain, advection<->V_ext) is pinned.
    """
    n = len(dates)
    cin = np.full(n, 290.0)
    t_rain = np.full(n, 280.0)
    flow = np.full(n, 3.0)
    rainfall = np.full(n, 0.001)

    out = rainfall_to_extracted_temperature(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=tedges,
        infiltration_area=1000.0,
        retardation_factor=2.0,
        **_GAMMA_LOW_FLOW,
    )
    swapped = 282.5  # (V_ext*T_rain + V_rain*T_adv)/(V_rain+V_ext) = (3*280 + 1*290)/4
    assert not np.any(np.isclose(out, swapped))
    np.testing.assert_array_equal(out, np.full(n, 287.5))


# ============================================================================
# Independent advective-conservation (T_adv not hidden behind the mix)
# ============================================================================


def test_independent_advective_conservation(dates, tedges):
    """In the zero-rain limit the result equals gamma_infiltration_to_extraction.

    A non-constant infiltration temperature exercises the advective transport so
    a T_adv bug would surface; the rainfall result must match the standalone
    advection call bit-for-bit (zero rain -> exact T_adv branch).
    """
    n = len(dates)
    t = np.arange(n)
    cin = 285.0 + 8.0 * np.sin(2 * np.pi * t / 90.0)  # seasonal infiltration temperature
    t_rain = np.full(n, 280.0)
    flow = 100.0 * (1.0 + 0.2 * np.sin(2 * np.pi * t / 30.0))  # variable flow
    rainfall = np.zeros(n)  # zero rain

    t_adv = gamma_infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        retardation_factor=2.0,
        **_GAMMA_HIGH_FLOW,
    )
    out = rainfall_to_extracted_temperature(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=tedges,
        infiltration_area=1000.0,
        retardation_factor=2.0,
        **_GAMMA_HIGH_FLOW,
    )
    # T_adv varies in time here (not a flat constant), so this is a real check.
    assert np.nanstd(t_adv) > 1.0
    np.testing.assert_array_equal(out, t_adv)


# ============================================================================
# Limiting cases (bit-exact)
# ============================================================================


def test_zero_rain_is_exactly_advection(dates, tedges):
    """rainfall == 0 -> result is bit-exact T_adv (V_rain == 0 branch)."""
    n = len(dates)
    cin = np.full(n, 290.0)
    t_rain = np.full(n, 280.0)
    flow = np.full(n, 100.0)
    rainfall = np.zeros(n)

    t_adv = gamma_infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        retardation_factor=2.0,
        **_GAMMA_HIGH_FLOW,
    )
    out = rainfall_to_extracted_temperature(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=tedges,
        infiltration_area=1000.0,
        retardation_factor=2.0,
        **_GAMMA_HIGH_FLOW,
    )
    np.testing.assert_array_equal(out, t_adv)


def test_zero_flow_is_exactly_rain(dates, tedges):
    """flow == 0 -> result is bit-exact t_rain (V_ext == 0 branch)."""
    n = len(dates)
    cin = np.full(n, 290.0)
    t_rain = 280.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 50.0)  # varying rain temperature
    flow = np.zeros(n)  # no extraction -> V_ext == 0
    rainfall = np.full(n, 0.1)

    out = rainfall_to_extracted_temperature(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=tedges,
        infiltration_area=1000.0,
        retardation_factor=2.0,
        **_GAMMA_HIGH_FLOW,
    )
    np.testing.assert_array_equal(out, t_rain)


def test_both_zero_is_nan(dates, tedges):
    """V_rain == 0 and V_ext == 0 -> NaN (0/0, documented)."""
    n = len(dates)
    cin = np.full(n, 290.0)
    t_rain = np.full(n, 280.0)
    flow = np.zeros(n)
    rainfall = np.zeros(n)

    out = rainfall_to_extracted_temperature(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=tedges,
        infiltration_area=1000.0,
        retardation_factor=2.0,
        **_GAMMA_HIGH_FLOW,
    )
    assert np.all(np.isnan(out))


# ============================================================================
# Bin alignment: cout_tedges coarser than tedges
# ============================================================================


def test_misaligned_output_bins(dates, tedges):
    """V_ext integrates flow over wider output bins (cout_tedges != tedges)."""
    n = len(dates)
    cin = np.full(n, 290.0)
    flow = np.full(n, 100.0)  # 100 m3/day
    rainfall = np.full(n, 0.1)  # 0.1 m/day * 1000 m2 = 100 m3/day

    # 2-day output bins, inside the input window.
    cout_dates = pd.date_range(start="2020-03-01", end="2020-05-31", freq="2D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))
    t_rain = np.full(len(cout_dates), 280.0)

    out = rainfall_to_extracted_temperature(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=cout_tedges,
        infiltration_area=1000.0,
        retardation_factor=2.0,
        **_GAMMA_HIGH_FLOW,
    )
    # Over each 2-day bin: V_ext = 100*2 = 200, V_rain = 100*2 = 200 -> equal -> 285.0.
    assert out.shape == (len(cout_dates),)
    assert not np.any(np.isnan(out))
    np.testing.assert_array_equal(out, np.full(len(cout_dates), 285.0))


# ============================================================================
# Fixtures from conftest
# ============================================================================


def test_uses_temperature_retardation_default(dates, tedges, retardation_scenarios):
    """retardation_factor flows through and defaults to the temperature scenario (2.0).

    Uses a time-varying ``cin`` so the retardation factor actually shifts the advective
    breakthrough: a constant ``cin`` would make every retardation give the same steady
    ``T_adv``, so the test could not distinguish R=1 from the R=2 default.
    """
    n = len(dates)
    cin = np.linspace(280.0, 300.0, n)  # time-varying so retardation changes T_adv
    t_rain = np.full(n, 280.0)
    flow = np.full(n, 100.0)
    rainfall = np.full(n, 0.1)
    common = {
        "cin": cin,
        "t_rain": t_rain,
        "flow": flow,
        "rainfall": rainfall,
        "tedges": tedges,
        "cout_tedges": tedges,
        "infiltration_area": 1000.0,
        **_GAMMA_HIGH_FLOW,
    }

    default = rainfall_to_extracted_temperature(**common)
    explicit_two = rainfall_to_extracted_temperature(**common, retardation_factor=retardation_scenarios["temperature"])
    explicit_one = rainfall_to_extracted_temperature(**common, retardation_factor=1.0)

    # The default equals the temperature scenario (2.0), bit-for-bit.
    np.testing.assert_array_equal(default, explicit_two)
    # Retardation genuinely affects the result: R=1 differs from the R=2 default.
    finite = ~np.isnan(default) & ~np.isnan(explicit_one)
    assert finite.any(), "test setup invalid: no overlapping finite bins to compare"
    assert not np.array_equal(default[finite], explicit_one[finite])


# ============================================================================
# Input validation
# ============================================================================


def test_negative_rainfall_raises(dates, tedges):
    """Negative rainfall is rejected."""
    n = len(dates)
    rainfall = np.full(n, 0.1)
    rainfall[10] = -0.01
    with pytest.raises(ValueError, match="rainfall must be non-negative"):
        rainfall_to_extracted_temperature(
            cin=np.full(n, 290.0),
            t_rain=np.full(n, 280.0),
            flow=np.full(n, 100.0),
            rainfall=rainfall,
            tedges=tedges,
            cout_tedges=tedges,
            infiltration_area=1000.0,
            **_GAMMA_HIGH_FLOW,
        )


def test_negative_flow_raises(dates, tedges):
    """Negative flow is rejected."""
    n = len(dates)
    flow = np.full(n, 100.0)
    flow[5] = -1.0
    with pytest.raises(ValueError, match="flow must be non-negative"):
        rainfall_to_extracted_temperature(
            cin=np.full(n, 290.0),
            t_rain=np.full(n, 280.0),
            flow=flow,
            rainfall=np.full(n, 0.1),
            tedges=tedges,
            cout_tedges=tedges,
            infiltration_area=1000.0,
            **_GAMMA_HIGH_FLOW,
        )


def test_nonpositive_infiltration_area_raises(dates, tedges):
    """infiltration_area must be strictly positive."""
    n = len(dates)
    with pytest.raises(ValueError, match="infiltration_area must be positive"):
        rainfall_to_extracted_temperature(
            cin=np.full(n, 290.0),
            t_rain=np.full(n, 280.0),
            flow=np.full(n, 100.0),
            rainfall=np.full(n, 0.1),
            tedges=tedges,
            cout_tedges=tedges,
            infiltration_area=0.0,
            **_GAMMA_HIGH_FLOW,
        )


def test_tedges_parity_raises(dates, tedges):
    """Mismatched flow length is rejected by bin-edge parity."""
    n = len(dates)
    with pytest.raises(ValueError, match="tedges must have one more element than flow"):
        rainfall_to_extracted_temperature(
            cin=np.full(n, 290.0),
            t_rain=np.full(n, 280.0),
            flow=np.full(n - 1, 100.0),  # wrong length
            rainfall=np.full(n, 0.1),
            tedges=tedges,
            cout_tedges=tedges,
            infiltration_area=1000.0,
            **_GAMMA_HIGH_FLOW,
        )


def test_nan_input_raises(dates, tedges):
    """NaN in cin is rejected."""
    n = len(dates)
    cin = np.full(n, 290.0)
    cin[3] = np.nan
    with pytest.raises(ValueError, match="cin contains NaN"):
        rainfall_to_extracted_temperature(
            cin=cin,
            t_rain=np.full(n, 280.0),
            flow=np.full(n, 100.0),
            rainfall=np.full(n, 0.1),
            tedges=tedges,
            cout_tedges=tedges,
            infiltration_area=1000.0,
            **_GAMMA_HIGH_FLOW,
        )
