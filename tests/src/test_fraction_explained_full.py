"""Tests for ``fraction_explained_full`` -- per-streamtube advective coverage of each output bin.

The oracle is an independent interval-intersection in cumulative-volume space (min/max overlap),
derived without the implementation's clipped-ramp formula, so a wrong ``/R``, a swapped direction, or
a per-row permutation cannot slip through.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport._time import tedges_to_days
from gwtransport.residence_time import fraction_explained_full
from gwtransport.utils import cumulative_flow_volume

DIRECTIONS = ["extraction_to_infiltration", "infiltration_to_extraction"]


def coverage_interval_oracle(*, flow, tedges, cout_tedges, aquifer_pore_volumes, direction, retardation_factor):
    """Independent per-(streamtube, bin) coverage via interval intersection in cumulative volume.

    A parcel at cumulative volume ``V`` is in-record iff ``V >= R*Vp`` (extraction_to_infiltration)
    or ``V <= V_total - R*Vp`` (infiltration_to_extraction). The flow-weighted coverage of a bin is
    the length of the in-record sub-interval over the bin width (pointwise indicator if zero-width).
    """
    tdays = tedges_to_days(pd.DatetimeIndex(tedges))
    flow_cum = cumulative_flow_volume(np.asarray(flow, dtype=float), np.diff(tdays))
    v_total = flow_cum[-1]
    cdays = tedges_to_days(pd.DatetimeIndex(cout_tedges), ref=pd.DatetimeIndex(tedges)[0])
    vol = np.interp(cdays, tdays, flow_cum, left=np.nan, right=np.nan)
    pore_volumes = np.atleast_1d(aquifer_pore_volumes).astype(float)
    out = np.full((len(pore_volumes), len(cdays) - 1), np.nan)
    for i, vp in enumerate(pore_volumes):
        if direction == "extraction_to_infiltration":
            in_lo, in_hi = retardation_factor * vp, np.inf
        else:
            in_lo, in_hi = -np.inf, v_total - retardation_factor * vp
        for b in range(out.shape[1]):
            lo, hi = vol[b], vol[b + 1]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            if hi <= lo:  # zero-throughflow bin -> pointwise indicator at its volume
                out[i, b] = 1.0 if in_lo <= lo <= in_hi else 0.0
            else:
                out[i, b] = max(0.0, min(hi, in_hi) - max(lo, in_lo)) / (hi - lo)
    return out


def variable_flow(seed=0, n=40, *, zero_gap=True):
    rng = np.random.default_rng(seed)
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 50.0 + 40.0 * np.sin(np.arange(n) / 4.0) + rng.uniform(0.0, 15.0, n)
    if zero_gap:
        flow[8:11] = 0.0  # a zero-throughflow plateau inside the record
    return flow, tedges


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("retardation_factor", [1.0, 2.3])
def test_matches_interval_oracle_variable_flow(direction, retardation_factor):
    flow, tedges = variable_flow()
    cout_tedges = pd.date_range("2019-12-25", periods=50, freq="D")  # over/underhangs both record ends
    pore_volumes = np.array([30.0, 80.0, 150.0, 400.0])
    got = fraction_explained_full(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction=direction,
        retardation_factor=retardation_factor,
    )
    exp = coverage_interval_oracle(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction=direction,
        retardation_factor=retardation_factor,
    )
    assert got.shape == (len(pore_volumes), len(cout_tedges) - 1)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(exp))
    np.testing.assert_allclose(got, exp, equal_nan=True, atol=1e-12)
    finite = got[~np.isnan(got)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_per_row_correctness_not_just_column_mean():
    """A row permutation would preserve the column mean; assert every row against the oracle."""
    flow, tedges = variable_flow(seed=3, zero_gap=False)
    pore_volumes = np.array([50.0, 250.0, 700.0])
    got = fraction_explained_full(
        flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=pore_volumes, retardation_factor=1.5
    )
    exp = coverage_interval_oracle(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
        retardation_factor=1.5,
    )
    np.testing.assert_allclose(got, exp, equal_nan=True, atol=1e-12)
    # The rows are genuinely distinct (larger pore volume -> later breakthrough), so per-row matters.
    assert not np.allclose(got[0], got[-1], equal_nan=True)


def test_partial_full_empty_staircase_constant_flow():
    """Constant flow, single pore volume: coverage is a clean 0 / partial / 1 staircase."""
    tedges = pd.date_range("2020-01-01", periods=11, freq="D")
    flow = np.full(10, 100.0)  # V_hi[b] = 100*(b+1)
    got = fraction_explained_full(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=[250.0])[0]
    # coverage[b] = clip((100*(b+1) - 250)/100, 0, 1)
    assert got[0] == 0.0
    assert got[1] == 0.0
    assert got[2] == pytest.approx(0.5)
    assert np.all(got[3:] == 1.0)


def test_out_of_record_bins_are_nan():
    flow, tedges = variable_flow(zero_gap=False)
    cout_tedges = pd.date_range("2019-12-01", periods=15, freq="D")  # starts well before the record
    got = fraction_explained_full(flow=flow, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_volumes=[100.0])
    assert np.isnan(got[0, 0])  # bin entirely before the flow record


def test_zero_flow_bin_finite_and_in_unit_interval():
    flow, tedges = variable_flow()  # includes a zero-throughflow gap
    got = fraction_explained_full(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=[80.0, 300.0])
    finite = got[~np.isnan(got)]
    assert finite.size > 0
    assert np.all((finite >= 0.0) & (finite <= 1.0))


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_negative_or_nan_flow_returns_all_nan(direction):
    flow, tedges = variable_flow(zero_gap=False)
    neg = flow.copy()
    neg[0] = -1.0
    nan = flow.copy()
    nan[2] = np.nan
    for bad in (neg, nan):
        got = fraction_explained_full(
            flow=bad, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=[100.0, 200.0], direction=direction
        )
        assert got.shape == (2, len(tedges) - 1)
        assert np.all(np.isnan(got))


def test_invalid_direction_raises():
    flow, tedges = variable_flow(zero_gap=False)
    with pytest.raises(ValueError, match="direction"):
        fraction_explained_full(
            flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=[100.0], direction="bogus"
        )
