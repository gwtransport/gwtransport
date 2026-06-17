"""Tests for ``fraction_explained_mean`` -- equal-weight discrete-APVD advective coverage.

Independent oracle: the equal-weight mean over streamtubes of the interval-intersection coverage
(derived without the implementation's clipped-ramp formula). Also checks discrete -> gamma
convergence and that a swapped direction (which still looks like a valid staircase) is rejected.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport._time import tedges_to_days
from gwtransport.gamma import bins as gamma_bins
from gwtransport.residence_time import fraction_explained_gamma, fraction_explained_mean
from gwtransport.utils import cumulative_flow_volume

DIRECTIONS = ["extraction_to_infiltration", "infiltration_to_extraction"]


def mean_coverage_oracle(*, flow, tedges, cout_tedges, aquifer_pore_volumes, direction, retardation_factor):
    """Equal-weight mean over streamtubes of the interval-intersection coverage (independent oracle)."""
    tdays = tedges_to_days(pd.DatetimeIndex(tedges))
    flow_cum = cumulative_flow_volume(np.asarray(flow, dtype=float), np.diff(tdays))
    v_total = flow_cum[-1]
    cdays = tedges_to_days(pd.DatetimeIndex(cout_tedges), ref=pd.DatetimeIndex(tedges)[0])
    vol = np.interp(cdays, tdays, flow_cum, left=np.nan, right=np.nan)
    pore_volumes = np.atleast_1d(aquifer_pore_volumes).astype(float)
    rows = np.full((len(pore_volumes), len(cdays) - 1), np.nan)
    for i, vp in enumerate(pore_volumes):
        if direction == "extraction_to_infiltration":
            in_lo, in_hi = retardation_factor * vp, np.inf
        else:
            in_lo, in_hi = -np.inf, v_total - retardation_factor * vp
        for b in range(rows.shape[1]):
            lo, hi = vol[b], vol[b + 1]
            if not (np.isfinite(lo) and np.isfinite(hi)):
                continue
            rows[i, b] = (
                (1.0 if in_lo <= lo <= in_hi else 0.0)
                if hi <= lo
                else max(0.0, min(hi, in_hi) - max(lo, in_lo)) / (hi - lo)
            )
    return rows.mean(axis=0)


def variable_flow(seed=0, n=40, *, zero_gap=True):
    rng = np.random.default_rng(seed)
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 50.0 + 40.0 * np.sin(np.arange(n) / 4.0) + rng.uniform(0.0, 15.0, n)
    if zero_gap:
        flow[8:11] = 0.0
    return flow, tedges


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("retardation_factor", [1.0, 2.3])
def test_matches_mean_interval_oracle(direction, retardation_factor):
    flow, tedges = variable_flow()
    cout_tedges = pd.date_range("2019-12-25", periods=50, freq="D")
    pore_volumes = np.array([30.0, 80.0, 150.0, 400.0])
    got = fraction_explained_mean(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction=direction,
        retardation_factor=retardation_factor,
    )
    exp = mean_coverage_oracle(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction=direction,
        retardation_factor=retardation_factor,
    )
    assert got.shape == (len(cout_tedges) - 1,)
    np.testing.assert_allclose(got, exp, equal_nan=True, atol=1e-12)


def test_swapped_direction_is_rejected():
    """e2i and i2e are distinct staircases; the e2i result must match the e2i oracle, not the i2e one."""
    flow, tedges = variable_flow(seed=1, zero_gap=False)
    pore_volumes = np.array([60.0, 200.0])
    got = fraction_explained_mean(
        flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=pore_volumes, retardation_factor=1.4
    )
    e2i = mean_coverage_oracle(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
        retardation_factor=1.4,
    )
    i2e = mean_coverage_oracle(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="infiltration_to_extraction",
        retardation_factor=1.4,
    )
    np.testing.assert_allclose(got, e2i, equal_nan=True, atol=1e-12)
    assert not np.allclose(got, i2e, equal_nan=True)  # the two directions genuinely differ here


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_converges_to_gamma(direction):
    """Discretizing a gamma APVD with more bins drives the discrete mean to the closed-form gamma value."""
    flow, tedges = variable_flow(seed=2)
    cout_tedges = tedges
    alpha, beta, loc = 4.0, 50.0, 30.0
    target = fraction_explained_gamma(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        alpha=alpha,
        beta=beta,
        loc=loc,
        direction=direction,
        retardation_factor=2.0,
    )
    errs = []
    for n_bins in (50, 300, 2000):
        pore_volumes = gamma_bins(alpha=alpha, beta=beta, loc=loc, n_bins=n_bins)["expected_values"]
        got = fraction_explained_mean(
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=pore_volumes,
            direction=direction,
            retardation_factor=2.0,
        )
        errs.append(np.nanmax(np.abs(got - target)))
    # monotone convergence toward the closed form
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 5e-4


def test_single_pore_volume_is_indicator():
    """One pore volume reduces to a 0/partial/1 coverage (the streamtube indicator, bin-averaged)."""
    tedges = pd.date_range("2020-01-01", periods=11, freq="D")
    flow = np.full(10, 100.0)
    got = fraction_explained_mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=[250.0])
    assert got[0] == 0.0
    assert got[2] == pytest.approx(0.5)
    assert np.all(got[3:] == 1.0)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_negative_or_nan_flow_all_nan(direction):
    flow, tedges = variable_flow(zero_gap=False)
    neg = flow.copy()
    neg[0] = -1.0
    nan = flow.copy()
    nan[2] = np.nan
    for bad in (neg, nan):
        got = fraction_explained_mean(
            flow=bad, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=[100.0], direction=direction
        )
        assert np.all(np.isnan(got))
