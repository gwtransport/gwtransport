import numpy as np
import pandas as pd
import pytest

from gwtransport._time import tedges_to_days
from gwtransport.residence_time import full
from gwtransport.utils import cumulative_flow_volume, linear_interpolate


def test_basic_extraction(constant_flow_data):
    """Test basic extraction scenario with constant flow."""
    flow_values, tedges = constant_flow_data
    cout_tedges = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1D")
    pore_volume = 200.0

    # spinup=None keeps the strict spin-up NaN at the leading extraction bins.
    result = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
        spinup=None,
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³, residence time is
    # exactly 2 days for piecewise-constant flow -- the underlying linear
    # interpolation and bin average are exact under constant Q.
    assert np.isnan(result[0, 0])
    assert np.isnan(result[0, 1])
    np.testing.assert_allclose(result[0, 2:], 2.0, atol=0, rtol=1e-13)

    # With the default spinup='constant' the spin-up zone is warm-started, so there is
    # no in-record NaN: residence time is exactly 2 days at every output bin.
    result_constant = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    np.testing.assert_allclose(result_constant[0], 2.0, atol=0, rtol=1e-13)


def test_basic_infiltration(constant_flow_data):
    """Test basic infiltration scenario with constant flow."""
    flow_values, tedges = constant_flow_data
    pore_volume = 200.0

    # spinup=None keeps the strict spin-up NaN at the trailing infiltration bins.
    result = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=pore_volume,
        direction="infiltration_to_extraction",
        spinup=None,
    )

    # With constant flow of 100 m³/day and pore volume of 200 m³, residence time
    # is exactly 2 days everywhere the parcel exits the record (machine precision).
    np.testing.assert_allclose(result[0, :-2], 2.0, atol=0, rtol=1e-13)
    # Later values should be NaN as water hasn't been extracted yet
    assert np.isnan(result[0, -2])
    assert np.isnan(result[0, -1])

    # With the default spinup='constant' the trailing spin-up is warm-started, so there
    # is no in-record NaN: residence time is exactly 2 days at every output bin.
    result_constant = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=pore_volume,
        direction="infiltration_to_extraction",
    )
    np.testing.assert_allclose(result_constant[0], 2.0, atol=0, rtol=1e-13)


def test_varying_extraction(constant_flow_data):
    """Test basic extraction scenario with constant flow."""
    flow_values, tedges = constant_flow_data
    flow_values[5:] *= 2.0  # Double the flow after the 5th day
    cout_tedges_highres = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1h")
    cout_tedges_lowres = pd.date_range(start="2023-01-01", end="2023-01-09", freq="1D")
    pore_volume = 200.0

    result_highres = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges_highres,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    df_highres = pd.Series(result_highres[0], index=cout_tedges_highres[:-1])
    df_lowres = df_highres.resample("1D").mean()
    result_lowres = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges_lowres,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Check that the mean values are consistent. Resampling the high-res result and
    # computing the low-res result directly are bit-identical here, so pin to machine
    # precision rather than the loose np.allclose default (rtol=1e-5).
    np.testing.assert_allclose(df_lowres.values, result_lowres[0], atol=0, rtol=1e-12, equal_nan=True)


def test_retardation_factor(constant_flow_data):
    """Test the effect of retardation factor."""
    flow_values, tedges = constant_flow_data
    pore_volume = 100.0

    # spinup=None keeps the strict spin-up NaN so the leading-NaN assertions stay meaningful.
    result_no_retardation = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=1.0,
        direction="extraction_to_infiltration",
        spinup=None,
    )

    result_with_retardation = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=2.0,
        direction="extraction_to_infiltration",
        spinup=None,
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
    flow_values, tedges = constant_flow_data
    pore_volumes = np.array([100.0, 200.0, 300.0])

    result = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    assert result.shape[0] == len(pore_volumes)
    assert result.shape[1] == len(tedges) - 1

    # Check that NaN pattern follows pore volumes
    # More NaNs at the beginning for larger pore volumes
    valid_counts = np.sum(~np.isnan(result), axis=1)
    assert np.all(np.diff(valid_counts) <= 0)

    # Check values for the smallest pore volume where we should have valid results
    # Residence times should increase with increasing pore volumes
    assert np.all(np.diff(result[:, -1]) > 0)


def test_invalid_direction(constant_flow_data):
    """Test that invalid direction raises ValueError."""
    flow_values, tedges = constant_flow_data
    cout_tedges = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    pore_volume = 200.0

    with pytest.raises(
        ValueError, match="direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
    ):
        full(
            flow=flow_values,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volume,
            direction="invalid",
        )


def test_edge_cases(sample_flow_data):
    """Test edge cases such as zero flow and very large pore volumes."""
    flow_values, tedges = sample_flow_data
    cout_tedges = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")

    # Test zero flow
    zero_flow = np.zeros_like(flow_values)
    result_zero = full(
        flow=zero_flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=100.0,
        direction="extraction_to_infiltration",
    )
    assert np.all(np.isnan(result_zero))

    # Test very large pore volume. The whole output record lies inside the spin-up zone, so
    # the strict spinup=None path marks every bin NaN. (Under the default spinup='constant'
    # these bins are warm-started to finite, very large residence times instead.)
    result_large = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=1e6,
        direction="extraction_to_infiltration",
        spinup=None,
    )
    assert np.all(np.isnan(result_large))


def test_negative_flow(constant_flow_data):
    """Test handling of negative flow values."""
    _, tedges = constant_flow_data
    flow_values = np.full(len(tedges) - 1, -100.0)
    cout_tedges = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    pore_volume = 200.0

    result = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
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
    flow_values, tedges = constant_flow_data
    double_flow = flow_values * 2
    cout_tedges = pd.date_range(start="2023-01-04", end="2023-01-09", freq="1D")
    pore_volume = 100.0

    result1 = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result2 = full(
        flow=double_flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    valid_mask = ~np.isnan(result1[0]) & ~np.isnan(result2[0])
    assert np.any(valid_mask)
    ratio = result1[0, valid_mask] / result2[0, valid_mask]
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-12)


def test_output_tedges_alignment():
    """Test that results align correctly with output time edges."""
    tedges = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(tedges) - 1, 100.0)

    # Test with different output time edges
    cout_tedges1 = pd.date_range(start="2023-01-02", end="2023-01-09", freq="2D")
    cout_tedges2 = pd.date_range(start="2023-01-02", end="2023-01-09", freq="1D")
    pore_volume = 100.0

    result1 = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges1,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result2 = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges2,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Check output shapes match the expected dimensions
    assert result1.shape[1] == len(cout_tedges1) - 1
    assert result2.shape[1] == len(cout_tedges2) - 1

    # With constant flow, residence time should be constant after initial NaNs
    valid_mask1 = ~np.isnan(result1[0])
    valid_mask2 = ~np.isnan(result2[0])

    if np.any(valid_mask1):
        np.testing.assert_allclose(result1[0, valid_mask1], result1[0, valid_mask1][0], atol=0, rtol=1e-13)

    if np.any(valid_mask2):
        np.testing.assert_allclose(result2[0, valid_mask2], result2[0, valid_mask2][0], atol=0, rtol=1e-13)


def test_spinup_policy_constant_vs_none():
    """Pin the spinup policy contract under constant flow with a spin-up region.

    With constant flow Q and pore volume V_p the residence time is exactly R*V_p/Q at
    every in-record output bin. A pore volume large enough that the look-back parcel
    leaves the record over the first bins creates a spin-up region:

    * ``spinup='constant'`` (default) warm-starts the spin-up by extrapolating the
      boundary flow, so there is no in-record NaN and every bin equals R*V_p/Q.
    * ``spinup=None`` is strict: the leading extraction bins are NaN, then R*V_p/Q.

    An unrecognised ``spinup`` value must raise ``ValueError``.
    """
    tedges = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow = 100.0
    flow_values = np.full(len(tedges) - 1, flow)
    retardation_factor = 1.0
    # V_p = 300 with Q = 100 needs 3 days of look-back, so the first two daily output bins
    # fall in the spin-up region under the strict policy.
    pore_volume = 300.0
    expected = retardation_factor * pore_volume / flow  # = 3.0 days

    common = {
        "flow": flow_values,
        "tedges": tedges,
        "cout_tedges": tedges,
        "aquifer_pore_volumes": pore_volume,
        "retardation_factor": retardation_factor,
        "direction": "extraction_to_infiltration",
    }

    rt_constant = full(**common, spinup="constant")
    rt_none = full(**common, spinup=None)

    # Default warm-start: no NaN anywhere in-record, exactly R*V_p/Q at every bin.
    assert not np.any(np.isnan(rt_constant))
    np.testing.assert_allclose(rt_constant[0], expected, atol=0, rtol=1e-13)

    # Strict: there is a genuine leading spin-up region (the first bin is NaN), it is a
    # contiguous leading block, and every finite bin afterwards is exactly R*V_p/Q.
    nan_mask = np.isnan(rt_none[0])
    assert nan_mask[0]
    n_nan = int(np.argmin(nan_mask)) if not nan_mask.all() else nan_mask.size
    assert n_nan >= 1
    assert not np.any(nan_mask[n_nan:])  # NaNs form a contiguous leading block
    np.testing.assert_allclose(rt_none[0, n_nan:], expected, atol=0, rtol=1e-13)

    # Wherever the strict path is finite it agrees exactly with the warm-started default.
    np.testing.assert_allclose(rt_none[0, n_nan:], rt_constant[0, n_nan:], atol=0, rtol=1e-13)

    with pytest.raises(ValueError, match="spinup"):
        full(**common, spinup="bad")


# ---------------------------------------------------------------------------
# Flow-weighted bin average -- closed-form phi antiderivative (issues #160, #165)
# ---------------------------------------------------------------------------


def _flow_weighted_reference(flow, tedges, cout_tedges, pore_volume, retardation_factor, direction):
    """Independent exact flow-weighted bin average, reconstructed from the inverse volume->time map.

    ``tau(V) = sign * (T(V + shift) - T(V))`` is piecewise-linear in cumulative volume ``V`` (``T`` is
    the strictly-monotone inverse cumulative-volume map, ``shift = sign * R * V_p``), so its exact
    flow-weighted average over ``[V_lo, V_hi]`` is the trapezoidal integral taken at the ``tau``
    breakpoints (where ``V`` or ``V + shift`` crosses a flow edge). This calls neither ``full`` nor any
    residence-time function -- only ``cumulative_flow_volume`` / ``linear_interpolate`` -- so it is a
    genuinely independent oracle. A zero-throughflow output bin (fixed volume, advancing time)
    degenerates to the pointwise ``tau`` at the bin time midpoint. Uses the strict (no warm-start) map,
    so it matches ``full(..., spinup=None)``.
    """
    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0
    shift = sign * retardation_factor * pore_volume
    tdays = tedges_to_days(pd.DatetimeIndex(tedges))
    flow_cum = cumulative_flow_volume(np.asarray(flow, dtype=float), np.diff(tdays), strictly_monotone=True)
    cout_days = tedges_to_days(pd.DatetimeIndex(cout_tedges), ref=pd.DatetimeIndex(tedges)[0])
    vol_out = linear_interpolate(x_ref=tdays, y_ref=flow_cum, x_query=cout_days, left=np.nan, right=np.nan)

    def time_at(v):
        return linear_interpolate(x_ref=flow_cum, y_ref=tdays, x_query=v, left=np.nan, right=np.nan)

    out = np.full(len(cout_days) - 1, np.nan)
    width_tol = 1e-9 * max(1.0, abs(float(flow_cum[-1])))
    for k in range(len(cout_days) - 1):
        v_lo, v_hi = vol_out[k], vol_out[k + 1]
        if not (np.isfinite(v_lo) and np.isfinite(v_hi)):
            continue
        if v_hi - v_lo <= width_tol:  # zero-throughflow bin: fixed volume, so the time midpoint
            t_lb = time_at(np.array([v_lo + shift]))[0]
            out[k] = sign * (t_lb - 0.5 * (cout_days[k] + cout_days[k + 1])) if np.isfinite(t_lb) else np.nan
            continue
        kinks_v = flow_cum[(flow_cum > v_lo) & (flow_cum < v_hi)]  # tau(V) kinks (interior flow edges)
        kinks_shift = (flow_cum - shift)[(flow_cum - shift > v_lo) & (flow_cum - shift < v_hi)]  # tau(V+shift) kinks
        bps = np.unique(np.concatenate([[v_lo, v_hi], kinks_v, kinks_shift]))
        tau = sign * (time_at(bps + shift) - time_at(bps))
        if np.any(np.isnan(tau)):
            continue  # strict spin-up: the look-back/forward parcel leaves the record within the bin
        out[k] = np.trapezoid(tau, bps) / (v_hi - v_lo)
    return out


def test_analytical_variable_flow():
    """Closed-form check on a hand-computed three-bin variable-flow case.

    Setup: piecewise-constant Q = [1, 1, 2] m^3/day over three 1-day bins (total 3 days),
    V_p = 0.5 m^3, R = 1. Both directions are pinned to a from-first-principles literal (not the
    oracle) so the direction-specific sign/shift is independently anchored.

    extraction_to_infiltration, output bin [day 1, day 3]: the look-back parcel crosses an internal
    flow edge at V = 2.5 (a kink in tau(V)), so

        flow-weighted mean = (1/3) [int_{V=1}^{V=2} 0.5 dV + int_{V=2}^{V=2.5} linear(0.5, 0.25) dV
                                     + int_{V=2.5}^{V=4} 0.25 dV]
                           = (1/3) [0.5 + 0.1875 + 0.375] = 17/48 day.

    infiltration_to_extraction, output bin [day 1, day 2] (V in [1, 2]): the look-forward parcel
    crosses the flow edge at V = 1.5, so

        flow-weighted mean = int_{V=1}^{V=1.5} 0.5 dV + int_{V=1.5}^{V=2} linear(0.5, 0.25) dV
                           = 0.25 + 0.1875 = 7/16 day.
    """
    tedges = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    flow_values = np.array([1.0, 1.0, 2.0])
    rt_flow = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=pd.DatetimeIndex(["2023-01-02", "2023-01-04"]),
        aquifer_pore_volumes=0.5,
        direction="extraction_to_infiltration",
    )
    np.testing.assert_allclose(rt_flow[0, 0], 17.0 / 48.0, atol=0, rtol=1e-13)

    rt_i2e = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=pd.DatetimeIndex(["2023-01-02", "2023-01-03"]),
        aquifer_pore_volumes=0.5,
        direction="infiltration_to_extraction",
        spinup=None,
    )
    np.testing.assert_allclose(rt_i2e[0, 0], 7.0 / 16.0, atol=0, rtol=1e-13)


def test_kink_handling_against_independent_reference():
    """Recover the flow-weighted residence time against an independent breakpoint-exact reference.

    Issue #165: ``full`` previously sampled tau only at ``tedges`` and missed kinks within a flow bin
    where the look-back parcel crosses an internal flow edge; under ``Q = [100, 100, 100, 100, 100,
    10, 200]`` the legacy edge-sampled estimate over the last bin overshot the truth by ~70 %. The
    reference reconstructs the piecewise-linear tau(V) from the inverse cumulative-volume map and
    integrates it exactly, independently of ``full``.
    """
    tedges = pd.date_range(start="2023-01-01", periods=8, freq="D")
    flow_values = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 10.0, 200.0])
    pore_volume = 50.0
    cout_tedges = tedges
    rt = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
        spinup=None,
    )
    expected = _flow_weighted_reference(
        flow_values, tedges, cout_tedges, pore_volume, 1.0, "extraction_to_infiltration"
    )
    np.testing.assert_allclose(rt[0], expected, atol=0, rtol=1e-13, equal_nan=True)


def test_zero_flow_plateau_in_lookback_matches_reference():
    """Zero-flow plateaus crossed by the look-back/forward parcel: exact vs the independent reference.

    The closed-form phi antiderivative integrates the step in tau exactly. The legacy augmented-grid
    path smeared multi-plateau crossings (and at a zero-throughflow output bin returned the bin
    endpoint rather than the time midpoint), drifting ~1e-2; this pins the fix against the
    breakpoint-exact reference over both directions and several pore volumes.
    """
    tedges = pd.date_range(start="2023-01-01", periods=12, freq="D")
    flow_values = np.array([40.0, 0.0, 25.0, 60.0, 0.0, 0.0, 80.0, 10.0, 0.0, 50.0, 30.0])
    cout_tedges = tedges
    for direction in ("extraction_to_infiltration", "infiltration_to_extraction"):
        for pore_volume in (30.0, 120.0, 240.0):
            rt = full(
                flow=flow_values,
                tedges=tedges,
                cout_tedges=cout_tedges,
                aquifer_pore_volumes=pore_volume,
                direction=direction,
                retardation_factor=1.3,
                spinup=None,
            )
            expected = _flow_weighted_reference(flow_values, tedges, cout_tedges, pore_volume, 1.3, direction)
            np.testing.assert_allclose(rt[0], expected, atol=0, rtol=1e-12, equal_nan=True)


def test_zero_flow_plateau_extraction():
    """Q = 0 over an upstream interval: the strictly-monotone regularization keeps the step exact.

    Setup: ``flow = [100, 0, 0, 100]`` (m^3/day) over four 1-day bins, ``V_p = 50`` m^3, ``R = 1``,
    extraction direction. On the output bin [3, 4] the look-back crosses the upstream plateau at the
    bin volume midpoint, so the flow-weighted mean is 0.5 * 2.5 + 0.5 * 0.5 = 1.5 day. Without the
    strictly-monotone (ulp-bump) regularization the duplicate ``flow_cum`` values would collapse the
    step into a ramp and return ~1.0. (The step here is symmetric, so the Phi quadratic term cancels;
    Phi correctness itself is pinned by ``test_analytical_variable_flow`` and the kink test.)
    """
    tedges = pd.date_range(start="2023-01-01", periods=5, freq="D")
    flow_values = np.array([100.0, 0.0, 0.0, 100.0])
    cout_tedges = tedges
    rt_flow = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=50.0,
        direction="extraction_to_infiltration",
    )
    np.testing.assert_allclose(rt_flow[0, 3], 1.5, atol=0, rtol=1e-13)


def test_zero_throughflow_output_bin_time_midpoint():
    """A zero-throughflow output bin returns the pointwise tau at the bin time midpoint.

    With ``flow = [200, 0, 100]`` (V_p = 50, R = 1, extraction, spinup=None) the cumulative-volume
    row is ``[0, 200, 200, 300]``. Output bin 0 ([day 0, day 1]) is in strict spin-up (look-back
    leaves the record) -> NaN. Output bin 1 ([day 1, day 2]) has Q = 0, so its volume is fixed at 200
    while output time advances: tau ramps with output time and its flow-weighted average is the value
    at the bin TIME midpoint, sign*(T(200 - 50) - 1.5) = -(0.75 - 1.5) = 0.75 day -- distinct from the
    left edge (0.25) and right edge (1.25), so the midpoint rule is pinned independently. Output bin 2
    ([day 2, day 3], Q = 100) gives 0.9375 day. These literals are hand-derived, not from the oracle.
    """
    tedges = pd.date_range(start="2023-01-01", periods=4, freq="D")
    flow_values = np.array([200.0, 0.0, 100.0])
    rt = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=50.0,
        direction="extraction_to_infiltration",
        spinup=None,
    )
    np.testing.assert_allclose(rt[0], [np.nan, 0.75, 0.9375], atol=0, rtol=1e-13, equal_nan=True)


def test_extrapolation_nan_consistency():
    """Output bins extending beyond the flow record are NaN.

    The volume path computes ``vol_out`` by interpolating with NaN fill; without the bin-bounds mask
    those bins would yield a finite (but meaningless) value instead of NaN.
    """
    tedges = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-03"])
    flow_values = np.array([1.0, 2.0])
    cout_tedges = pd.DatetimeIndex(["2023-01-02", "2023-01-04"])  # extends past the last flow edge
    rt_flow = full(
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=0.25,
        direction="extraction_to_infiltration",
    )
    assert np.isnan(rt_flow[0, 0])


def _warmstart_reference(flow, tedges, cout_tedges, pore_volume, retardation_factor, direction):
    """Independent warm-start ('constant' spin-up) reference: breakpoint-exact over the extended map.

    Extends the cumulative-volume -> time map one anchor past each end at the FIRST and LAST
    positive-flow rates (start slope 1/Q_first, end slope 1/Q_last -- distinct under variable flow),
    then integrates the piecewise-linear tau(V) = sign*(T(V+shift) - T(V)) EXACTLY at its breakpoints
    (where V or V+shift crosses a map knot). Calls no residence-time function, so a first/last
    extrapolation-slope swap in ``full`` is caught at machine precision. Assumes positive throughflow
    over each output bin (no zero-throughflow degeneracy), as the warm-start test uses.
    """
    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0
    shift = sign * retardation_factor * pore_volume
    tdays = tedges_to_days(pd.DatetimeIndex(tedges))
    flow = np.asarray(flow, dtype=float)
    flow_cum = cumulative_flow_volume(flow, np.diff(tdays), strictly_monotone=True)
    pad = retardation_factor * pore_volume
    pos = flow[flow > 0.0]
    map_v = np.concatenate([[flow_cum[0] - pad], flow_cum, [flow_cum[-1] + pad]])
    map_t = np.concatenate([[tdays[0] - pad / pos[0]], tdays, [tdays[-1] + pad / pos[-1]]])
    cout_days = tedges_to_days(pd.DatetimeIndex(cout_tedges), ref=pd.DatetimeIndex(tedges)[0])
    vol_out = linear_interpolate(x_ref=tdays, y_ref=flow_cum, x_query=cout_days, left=np.nan, right=np.nan)

    def time_at(v):
        return linear_interpolate(x_ref=map_v, y_ref=map_t, x_query=v)  # clamps within the padded range

    out = np.full(len(cout_days) - 1, np.nan)
    for k in range(len(cout_days) - 1):
        v_lo, v_hi = vol_out[k], vol_out[k + 1]
        if not (np.isfinite(v_lo) and np.isfinite(v_hi)):
            continue
        kinks_v = map_v[(map_v > v_lo) & (map_v < v_hi)]  # tau(V) kinks
        kinks_shift = (map_v - shift)[(map_v - shift > v_lo) & (map_v - shift < v_hi)]  # tau(V+shift) kinks
        bps = np.unique(np.concatenate([[v_lo, v_hi], kinks_v, kinks_shift]))
        tau = sign * (time_at(bps + shift) - time_at(bps))
        out[k] = np.trapezoid(tau, bps) / (v_hi - v_lo)
    return out


def test_variable_flow_warmstart_against_extended_map():
    """Warm-start ('constant' spin-up) under variable flow uses the boundary flow rate at each end.

    With Q_first != Q_last the start (extraction) and end (infiltration) extrapolation slopes are
    distinct numbers (1/Q_first vs 1/Q_last), so a first/last slope swap is invisible under constant
    flow (the regime every other warm-start test uses). A pore volume large enough to push the parcel
    out of the record at the boundary creates a genuine warm-started spin-up region; the result is
    pinned against an independent boundary-extended fine-grid reference in both directions.
    """
    tedges = pd.date_range("2020-01-01", periods=17, freq="D")
    flow_values = np.linspace(50.0, 300.0, 16)  # Q_first = 50 != Q_last = 300
    pore_volume = 600.0  # large enough for a warm-started spin-up region at each record end
    for direction in ("extraction_to_infiltration", "infiltration_to_extraction"):
        rt = full(
            flow=flow_values,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=pore_volume,
            direction=direction,
        )  # default spinup="constant"
        assert not np.isnan(rt[0]).any(), "warm-start must leave no in-record NaN"
        expected = _warmstart_reference(flow_values, tedges, tedges, pore_volume, 1.0, direction)
        np.testing.assert_allclose(rt[0], expected, atol=0, rtol=1e-12)


def test_zero_shift_returns_zero():
    """Zero shift (retardation_factor = 0, or all pore volumes 0) gives exactly-zero residence time.

    The look-back/forward parcel coincides with the parcel itself, so tau is 0 in every in-record bin.
    Regression guard: a zero extrapolation pad (= R * max(V_p)) must not build degenerate boundary
    knots whose 0/0 rate would poison the antiderivative to NaN (it returned all-NaN before the fix).
    """
    tedges = pd.date_range("2023-01-01", periods=9, freq="D")
    flow_values = np.array([100.0, 110.0, 105.0, 95.0, 98.0, 102.0, 107.0, 103.0])
    for kwargs in (
        {"retardation_factor": 0.0, "aquifer_pore_volumes": 200.0},
        {"retardation_factor": 1.0, "aquifer_pore_volumes": 0.0},
    ):
        for direction in ("extraction_to_infiltration", "infiltration_to_extraction"):
            rt = full(flow=flow_values, tedges=tedges, cout_tedges=tedges, direction=direction, **kwargs)
            np.testing.assert_array_equal(rt, np.zeros_like(rt))
