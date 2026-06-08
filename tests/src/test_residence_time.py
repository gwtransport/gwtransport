import numpy as np
import pandas as pd
import pytest

from gwtransport.residence_time import (
    fraction_explained,
    residence_time,
    residence_time_full,
    residence_time_series,
)

DIRECTIONS = ["extraction_to_infiltration", "infiltration_to_extraction"]


def _variable_flow(n=40, seed=1):
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    rng = np.random.default_rng(seed)
    flow = np.clip(100.0 + 30.0 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 6, n), 5.0, None)
    return flow, tedges


def _constant_flow(n=40, q=100.0):
    return np.full(n, q), pd.date_range("2020-01-01", periods=n + 1, freq="D")


def _mean_over_valid(rt):
    """Per-bin arithmetic mean over the finite per-pore-volume entries, via an explicit loop.

    Structurally different from the vectorized ``nansum/count`` in the implementation, so it
    catches the realistic mutations (renormalizing over the full ``n`` instead of the valid
    subset, or treating NaN as 0 while keeping the full count).
    """
    out = np.full(rt.shape[1], np.nan)
    for b in range(rt.shape[1]):
        column = rt[:, b]
        finite = column[np.isfinite(column)]
        if finite.size:
            out[b] = finite.sum() / finite.size
    return out


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("r", [1.0, 2.5])
@pytest.mark.parametrize("spinup", ["constant", None])
def test_discrete_equals_mean_over_valid_streamtubes(direction, r, spinup):
    """residence_time is exactly the mean over the valid streamtubes of residence_time_full.

    Holds to machine precision for both spin-up policies. Under ``spinup='constant'`` every
    in-record streamtube is finite (warm-started), so it is a plain mean; under ``spinup=None`` it
    renormalizes over the streamtubes that have broken through, dropping the rest all-or-nothing.
    """
    flow, tedges = _variable_flow()
    apv = np.array([250.0, 600.0, 900.0, 1200.0])
    got = residence_time(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=apv,
        direction=direction,
        retardation_factor=r,
        spinup=spinup,
    )
    rt = residence_time_full(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=apv,
        direction=direction,
        retardation_factor=r,
        weighting="flow",
        spinup=spinup,
    )
    expected = _mean_over_valid(rt)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got, expected, rtol=0, atol=0, equal_nan=True)
    if spinup == "constant":
        assert not np.isnan(got).any(), "constant warm-start must leave no in-record NaN"
    else:
        assert np.isnan(got).any(), "fixture must include a fully-uncovered (all-streamtube spin-up) bin"


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("r", [1.0, 2.5])
def test_constant_spinup_constant_flow_exact(direction, r):
    """Constant flow + constant warm-start: tau == R*mean(V_p)/Q at every bin, no NaN.

    The extrapolated boundary flow equals the in-record flow, so the spin-up bins take the same
    value as the deep bins. This pins the warm-start to an independent analytic oracle.
    """
    q = 100.0
    flow, tedges = _constant_flow(q=q)
    apv = np.array([200.0, 500.0, 900.0])
    got = residence_time(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=apv,
        direction=direction,
        retardation_factor=r,
    )
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, r * apv.mean() / q, rtol=1e-11)


def test_single_pore_volume_reduces_to_residence_time_full():
    """A one-element APVD collapses to that streamtube's residence_time_full row."""
    flow, tedges = _variable_flow()
    got = residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=750.0)
    ref = residence_time_full(
        flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=750.0, weighting="flow"
    )[0]
    np.testing.assert_array_equal(got, ref)


def test_spinup_none_drops_genuinely_partial_streamtube():
    """With spinup=None a streamtube covered only part-way through a bin is dropped whole.

    The output grid is coarser than the flow grid (5-day bins over a daily record), so the large
    streamtube is genuinely ~50% covered inside the transition bin while its ``residence_time_full``
    row is NaN. All-or-nothing must drop it: the bin mean equals the small streamtube alone, not a
    coverage-weighted blend.
    """
    flow = np.full(40, 100.0)
    flow_tedges = pd.date_range("2020-01-01", periods=41, freq="D")
    tedges_out = pd.date_range("2020-01-01", periods=9, freq="5D")  # 8 output bins of 5 days
    apv = np.array([200.0, 2750.0])  # large needs 27.5 days history -> transition inside bin 5
    rt = residence_time_full(
        flow=flow,
        flow_tedges=flow_tedges,
        tedges_out=tedges_out,
        aquifer_pore_volumes=apv,
        weighting="flow",
        spinup=None,
    )
    assert np.isfinite(rt[0][5]), "test setup: small streamtube must be informed in bin 5"
    assert np.isnan(rt[1][5]), "test setup: large streamtube must straddle the record edge inside bin 5"
    got = residence_time(
        flow=flow, flow_tedges=flow_tedges, tedges_out=tedges_out, aquifer_pore_volumes=apv, spinup=None
    )
    # If the large tube's partial coverage leaked in, the mean would be pulled toward 27.5 days.
    np.testing.assert_allclose(got[5], rt[0][5], rtol=0, atol=0)


def test_spinup_none_fully_uncovered_bin_is_nan():
    """With spinup=None, a bin with no valid streamtube (all in spin-up) is NaN."""
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([3500.0, 3800.0])  # both need > 35 days of history; the first bins have none
    got = residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=apv, spinup=None)
    assert np.isnan(got[0])


@pytest.mark.parametrize("spinup", ["constant", None])
def test_out_of_window_bin_is_nan(spinup):
    """Output bins beyond the flow record are NaN under either spin-up policy."""
    flow, tedges = _constant_flow(n=20, q=100.0)
    tedges_out = pd.date_range("2020-01-01", periods=31, freq="D")  # extends 10 days past the record
    got = residence_time(
        flow=flow, flow_tedges=tedges, tedges_out=tedges_out, aquifer_pore_volumes=300.0, spinup=spinup
    )
    assert np.isnan(got[-1])


def test_series_still_marks_spinup_while_full_warm_starts():
    """residence_time_series keeps its spin-up NaN even though residence_time_full warm-starts.

    The point series is the primitive behind :func:`fraction_explained`, which therefore still
    locates the spin-up region (fraction < 1) where the warm-started means are finite.
    """
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([200.0, 1500.0])
    series = residence_time_series(flow=flow, flow_tedges=tedges, aquifer_pore_volumes=apv)
    full = residence_time_full(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=apv)
    assert np.isnan(series).any(), "series must keep its spin-up NaN"
    assert not np.isnan(full).any(), "full default warm-start must be finite in-record"
    frac = fraction_explained(rt=series)
    assert (frac < 1.0).any(), "fraction_explained must flag the spin-up region"
    assert (frac == 1.0).any(), "fraction_explained must reach 1 outside the spin-up"


def test_invalid_direction_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="direction"):
        residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=300.0, direction="bad")


def test_invalid_spinup_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="spinup"):
        residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=300.0, spinup="bad")
