import numpy as np
import pandas as pd
import pytest

from gwtransport.residence_time import residence_time, residence_time_mean


@pytest.fixture
def sample_flow_data():
    """Create sample flow data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.array([100.0, 110.0, 105.0, 95.0, 98.0, 102.0, 107.0, 103.0, 96.0])
    return flow_values, dates


@pytest.fixture
def constant_flow_data():
    """Create constant flow data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(dates) - 1, 100.0)
    return flow_values, dates


def test_basic_extraction(constant_flow_data):
    """Test basic extraction scenario with constant flow."""
    flow_values, flow_tedges = constant_flow_data
    tedges_out = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1D")
    pore_volume = 200.0

    result = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³, residence time is
    # exactly 2 days for piecewise-constant flow -- the underlying linear
    # interpolation and bin average are exact under constant Q.
    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])
    np.testing.assert_allclose(result[0, 2:], 2.0, atol=0, rtol=1e-13)


def test_basic_infiltration(constant_flow_data):
    """Test basic infiltration scenario with constant flow."""
    flow_values, flow_tedges = constant_flow_data
    pore_volume = 200.0

    result = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="infiltration_to_extraction",
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³, residence time
    # is exactly 2 days everywhere the parcel exits the record (machine precision).
    np.testing.assert_allclose(result[0, :-2], 2.0, atol=0, rtol=1e-13)
    # Later values should be NaN as water hasn't been extracted yet
    assert np.isnan(result[0, -2])
    assert np.isnan(result[0, -1])


def test_varying_extraction(constant_flow_data):
    """Test basic extraction scenario with constant flow."""
    flow_values, flow_tedges = constant_flow_data
    flow_values[5:] *= 2.0  # Double the flow after the 5th day
    tedges_out_highres = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1h")
    tedges_out_lowres = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1D")
    pore_volume = 200.0

    result_highres = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out_highres,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    df_highres = pd.Series(result_highres[0], index=tedges_out_highres[:-1])
    df_lowres = df_highres.resample("1D").mean()
    result_lowres = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out_lowres,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Check that the mean values are consistent
    assert np.allclose(df_lowres.values, result_lowres[0], equal_nan=True)


def test_retardation_factor(constant_flow_data):
    """Test the effect of retardation factor."""
    flow_values, flow_tedges = constant_flow_data
    pore_volume = 100.0

    result_no_retardation = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=1.0,
        direction="extraction_to_infiltration",
    )

    result_with_retardation = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=2.0,
        direction="extraction_to_infiltration",
    )

    # Residence time should double with retardation factor of 2
    # We need to check in positions where both have valid values
    assert np.isnan(result_no_retardation[0, 0])
    assert np.isnan(result_with_retardation[0, 0])
    assert np.isnan(result_with_retardation[0, 1])

    # Later values should be valid. With constant flow, retardation R = 2 must scale
    # the mean residence time by exactly 2, since the underlying linear interpolation
    # (and the linear average over output bins) is exact for piecewise-constant flow.
    np.testing.assert_allclose(result_with_retardation[0, 2], 2 * result_no_retardation[0, 1], rtol=1e-12)
    np.testing.assert_allclose(result_with_retardation[0, 3], 2 * result_no_retardation[0, 2], rtol=1e-12)


def test_multiple_pore_volumes(constant_flow_data):
    """Test handling of multiple pore volumes."""
    flow_values, flow_tedges = constant_flow_data
    pore_volumes = np.array([100.0, 200.0, 300.0])

    result = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    assert result.shape[0] == len(pore_volumes)
    assert result.shape[1] == len(flow_tedges) - 1

    # Check that NaN pattern follows pore volumes
    # More NaNs at the beginning for larger pore volumes
    valid_counts = np.sum(~np.isnan(result), axis=1)
    assert np.all(np.diff(valid_counts) <= 0)

    # Check values for the smallest pore volume where we should have valid results
    # Residence times should increase with increasing pore volumes
    assert np.all(np.diff(result[:, -1]) > 0)


def test_invalid_direction(constant_flow_data):
    """Test that invalid direction raises ValueError."""
    flow_values, flow_tedges = constant_flow_data
    tedges_out = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    pore_volume = 200.0

    with pytest.raises(
        ValueError, match="direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
    ):
        residence_time_mean(
            flow=flow_values,
            flow_tedges=flow_tedges,
            tedges_out=tedges_out,
            aquifer_pore_volumes=pore_volume,
            direction="invalid",
        )


def test_edge_cases(sample_flow_data):
    """Test edge cases such as zero flow and very large pore volumes."""
    flow_values, flow_tedges = sample_flow_data
    tedges_out = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")

    # Test zero flow
    zero_flow = np.zeros_like(flow_values)
    result_zero = residence_time_mean(
        flow=zero_flow,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=100.0,
        direction="extraction_to_infiltration",
    )
    assert np.all(np.isnan(result_zero))

    # Test very large pore volume
    result_large = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=1e6,
        direction="extraction_to_infiltration",
    )
    assert np.all(np.isnan(result_large))


def test_negative_flow(constant_flow_data):
    """Test handling of negative flow values."""
    _, flow_tedges = constant_flow_data
    flow_values = np.full(len(flow_tedges) - 1, -100.0)
    tedges_out = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    pore_volume = 200.0

    result = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Negative flow should result in NaN values
    assert np.all(np.isnan(result))


def test_flow_variations(constant_flow_data):
    """Doubling constant flow exactly halves the mean residence time.

    Doubling a *variable* flow does not halve the residence time exactly (the inversion
    V_p = integral Q ds rescales differently for non-constant Q). With *constant* flow,
    doubling Q halves both the pointwise residence time and its bin average, so the
    ratio must be 2.0 to machine precision wherever both are valid.
    """
    flow_values, flow_tedges = constant_flow_data
    double_flow = flow_values * 2
    tedges_out = pd.date_range(start="2023-01-04", end="2023-01-09", freq="1D")
    pore_volume = 100.0

    result1 = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result2 = residence_time_mean(
        flow=double_flow,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    valid_mask = ~np.isnan(result1[0]) & ~np.isnan(result2[0])
    assert np.any(valid_mask)
    ratio = result1[0, valid_mask] / result2[0, valid_mask]
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-12)


def test_output_tedges_alignment():
    """Test that results align correctly with output time edges."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)

    # Test with different output time edges
    tedges_out1 = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    tedges_out2 = pd.date_range(start="2023-01-02", end="2023-01-09", freq="1D")
    pore_volume = 100.0

    result1 = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out1,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result2 = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out2,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Check output shapes match the expected dimensions
    assert result1.shape[1] == len(tedges_out1) - 1
    assert result2.shape[1] == len(tedges_out2) - 1

    # With constant flow, residence time should be constant after initial NaNs
    valid_mask1 = ~np.isnan(result1[0])
    valid_mask2 = ~np.isnan(result2[0])

    if np.any(valid_mask1):
        np.testing.assert_allclose(result1[0, valid_mask1], result1[0, valid_mask1][0], atol=0, rtol=1e-13)

    if np.any(valid_mask2):
        np.testing.assert_allclose(result2[0, valid_mask2], result2[0, valid_mask2][0], atol=0, rtol=1e-13)


def test_example_from_docstring():
    """Test the example provided in the function's docstring."""
    flow_dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(flow_dates) - 1, 100.0)  # Constant flow of 100 m³/day
    pore_volume = 200.0

    mean_times = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_dates,
        tedges_out=flow_dates,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # The first values should be NaN (not enough water has passed)
    assert np.isnan(mean_times[0, 0])
    assert np.isnan(mean_times[0, 1])

    # Later values are exactly 2 days under constant flow.
    np.testing.assert_allclose(mean_times[0, 2:], 2.0, atol=0, rtol=1e-13)


# ---------------------------------------------------------------------------
# weighting={"flow","time"} parameter (issue #160)
# ---------------------------------------------------------------------------


def test_constant_flow_weighting_equivalence(constant_flow_data):
    """Flow- and time-weighting agree to machine precision when Q is constant.

    Under constant flow, integrating uniformly in cumulative volume and
    integrating uniformly in time produce the same per-bin average. Any drift
    here would indicate that the volume-coordinate path has lost an exact
    invariant, e.g. via off-by-one indexing of ``flow_cum_at_tedges_out``.
    """
    flow_values, flow_tedges = constant_flow_data
    tedges_out = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1D")
    pore_volumes = np.array([100.0, 200.0, 300.0])

    common = {
        "flow": flow_values,
        "flow_tedges": flow_tedges,
        "tedges_out": tedges_out,
        "aquifer_pore_volumes": pore_volumes,
        "direction": "extraction_to_infiltration",
    }
    rt_flow = residence_time_mean(**common, weighting="flow")
    rt_time = residence_time_mean(**common, weighting="time")
    np.testing.assert_array_equal(rt_flow, rt_time)  # exact, NaN-aware


def test_default_weighting_is_flow():
    """Calling without the ``weighting`` kwarg must behave like ``weighting='flow'``.

    Uses the same Q = [1, 1, 2] scenario as ``test_analytical_variable_flow_weighting``
    where the output bin spans a flow-step boundary -- without that, the two
    weightings would coincide and the default-vs-explicit comparison would be
    vacuous.
    """
    flow_tedges = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    flow_values = np.array([1.0, 1.0, 2.0])
    pore_volume = 0.5
    tedges_out = pd.DatetimeIndex(["2023-01-02", "2023-01-04"])

    common = {
        "flow": flow_values,
        "flow_tedges": flow_tedges,
        "tedges_out": tedges_out,
        "aquifer_pore_volumes": pore_volume,
        "direction": "extraction_to_infiltration",
    }
    rt_default = residence_time_mean(**common)
    rt_flow = residence_time_mean(**common, weighting="flow")
    rt_time = residence_time_mean(**common, weighting="time")
    np.testing.assert_array_equal(rt_default, rt_flow)
    # Sanity: the two weightings actually differ here, otherwise the assertion above
    # would also be satisfied by the legacy time-weighted code path.
    assert not np.isclose(rt_default[0, 0], rt_time[0, 0])


def test_invalid_weighting(constant_flow_data):
    """An unrecognised ``weighting`` value must raise ValueError."""
    flow_values, flow_tedges = constant_flow_data
    tedges_out = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")

    with pytest.raises(ValueError, match=r"weighting should be 'flow' or 'time'"):
        residence_time_mean(
            flow=flow_values,
            flow_tedges=flow_tedges,
            tedges_out=tedges_out,
            aquifer_pore_volumes=200.0,
            weighting="invalid",
        )


def test_analytical_variable_flow_weighting():
    """Closed-form check on a hand-computed three-bin variable-flow case.

    Setup: piecewise-constant Q = [1, 1, 2] m^3/day over three 1-day bins
    (total 3 days), V_p = 0.5 m^3, R = 1, direction = extraction_to_infiltration.
    The look-back parcel crosses an internal flow edge at t* = 2.25 (kink), so
    on the output bin [day 1, day 3] the residence-time signal is

        tau(t) = 0.5,                   t in [1, 2]
        tau(t) = 2.5 - t,               t in [2, 2.25]
        tau(t) = 0.25,                  t in [2.25, 3].

    Hand calculation gives:

    - time-weighted mean = (1/2) [int_1^2 0.5 dt + int_2^{2.25}(2.5 - t) dt
                                   + int_{2.25}^3 0.25 dt]
                         = (1/2) [0.5 + 0.09375 + 0.1875] = 25/64 day.
    - flow-weighted mean = (1/3) [int_{V=1}^{V=2} 0.5 dV + int_{V=2}^{V=2.5} linear(0.5, 0.25) dV
                                   + int_{V=2.5}^{V=4} 0.25 dV]
                         = (1/3) [0.5 + 0.1875 + 0.375] = 17/48 day.
    """
    flow_tedges = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    flow_values = np.array([1.0, 1.0, 2.0])
    pore_volume = 0.5
    tedges_out = pd.DatetimeIndex(["2023-01-02", "2023-01-04"])

    common = {
        "flow": flow_values,
        "flow_tedges": flow_tedges,
        "tedges_out": tedges_out,
        "aquifer_pore_volumes": pore_volume,
        "direction": "extraction_to_infiltration",
    }
    rt_time = residence_time_mean(**common, weighting="time")
    rt_flow = residence_time_mean(**common, weighting="flow")

    np.testing.assert_allclose(rt_time[0, 0], 25.0 / 64.0, atol=0, rtol=1e-13)
    np.testing.assert_allclose(rt_flow[0, 0], 17.0 / 48.0, atol=0, rtol=1e-13)


def test_kink_handling_against_fine_grid():
    """Recover the residence-time integral against a fine-grid reference.

    Issue #165: ``residence_time_mean`` previously sampled tau only at
    ``flow_tedges`` and missed kinks within a flow bin where the look-back
    parcel crosses an internal flow edge. Under the regime change
    ``Q = [100, 100, 100, 100, 100, 10, 200]`` from the issue body, the legacy
    edge-sampled estimate over the last bin overshoots the truth by ~70 %.

    This test pins the function against an independent fine-grid trapezoidal
    average of ``residence_time`` (which is pointwise correct).
    """
    flow_tedges = pd.date_range(start="2023-01-01", periods=8, freq="D")
    flow_values = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 10.0, 200.0])
    pore_volume = 50.0
    tedges_out = flow_tedges  # daily output bins, same as flow tedges
    n_fine = 20001

    rt_mean = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
        weighting="time",
    )

    # Independent reference: dense pointwise tau via residence_time, trapezoidal
    # average per output bin.
    flow_tedges_days_arr = (flow_tedges - flow_tedges[0]) / np.timedelta64(1, "D")
    tedges_days_arr = np.asarray(flow_tedges_days_arr, dtype=float)
    expected = np.full((1, len(tedges_out) - 1), np.nan)
    for k in range(len(tedges_out) - 1):
        t_dense = np.linspace(tedges_days_arr[k], tedges_days_arr[k + 1], n_fine)
        index = flow_tedges[0] + pd.to_timedelta(t_dense, unit="D")
        tau_dense = residence_time(
            flow=flow_values,
            flow_tedges=flow_tedges,
            aquifer_pore_volumes=pore_volume,
            index=index,
            direction="extraction_to_infiltration",
        )[0]
        if np.any(np.isnan(tau_dense)):
            expected[0, k] = np.nan
        else:
            expected[0, k] = np.trapezoid(tau_dense, t_dense) / (t_dense[-1] - t_dense[0])

    np.testing.assert_allclose(rt_mean, expected, atol=0, rtol=1e-13, equal_nan=True)


def test_weighting_extrapolation_nan_consistency():
    """Output bins extending beyond the flow record are NaN under both weightings.

    The volume path computes ``flow_cum_at_tedges_out`` by clamping; without the
    explicit bin-bounds mask in the implementation, those bins would yield a
    finite (but meaningless) value instead of NaN.
    """
    flow_tedges = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-03"])
    flow_values = np.array([1.0, 2.0])
    # Output bin extends past the last flow edge.
    tedges_out = pd.DatetimeIndex(["2023-01-02", "2023-01-04"])

    rt_flow = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=0.25,
        direction="extraction_to_infiltration",
        weighting="flow",
    )
    rt_time = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=0.25,
        direction="extraction_to_infiltration",
        weighting="time",
    )
    assert np.isnan(rt_flow[0, 0])
    assert np.isnan(rt_time[0, 0])
