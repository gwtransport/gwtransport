"""Tests for ``fraction_explained_gamma`` -- closed-form (shifted) gamma-APVD advective coverage.

Primary anchor: a brute-force ``scipy.integrate.quad`` of the in-record indicator over each bin in
cumulative-volume space, whose integrand calls only ``scipy.stats.gamma.cdf`` -- it never reuses the
implementation's ``Phi`` antiderivative, so a wrong ``/R``, dropped ``loc``, or swapped direction is
caught.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import gamma as gamma_dist

from gwtransport._time import tedges_to_days
from gwtransport.residence_time import fraction_explained_gamma
from gwtransport.utils import cumulative_flow_volume

DIRECTIONS = ["extraction_to_infiltration", "infiltration_to_extraction"]


def quad_oracle(*, flow, tedges, cout_tedges, alpha, beta, loc, direction, retardation_factor):
    """Flow-weighted bin average of F_Vp(threshold(V)) via quad over cumulative volume (cdf-only)."""
    tdays = tedges_to_days(pd.DatetimeIndex(tedges))
    flow_cum = cumulative_flow_volume(np.asarray(flow, dtype=float), np.diff(tdays))
    v_total = flow_cum[-1]
    cdays = tedges_to_days(pd.DatetimeIndex(cout_tedges), ref=pd.DatetimeIndex(tedges)[0])
    vol = np.interp(cdays, tdays, flow_cum, left=np.nan, right=np.nan)

    def cdf_shifted(threshold):
        return gamma_dist.cdf(np.maximum(threshold - loc, 0.0), alpha, scale=beta)

    out = np.full(len(cdays) - 1, np.nan)
    for b in range(len(out)):
        lo, hi = vol[b], vol[b + 1]
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            continue
        if direction == "extraction_to_infiltration":
            val, _ = quad(lambda v: cdf_shifted(v / retardation_factor), lo, hi, limit=200)
        else:
            val, _ = quad(lambda v: cdf_shifted((v_total - v) / retardation_factor), lo, hi, limit=200)
        out[b] = val / (hi - lo)
    return out


def variable_flow(seed=0, n=40, *, zero_gap=True):
    rng = np.random.default_rng(seed)
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = 50.0 + 40.0 * np.sin(np.arange(n) / 4.0) + rng.uniform(0.0, 15.0, n)
    if zero_gap:
        flow[8:11] = 0.0
    return flow, tedges


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("retardation_factor", [1.0, 2.3])
@pytest.mark.parametrize("loc", [0.0, 30.0])
def test_matches_quad_oracle(direction, retardation_factor, loc):
    flow, tedges = variable_flow()
    cout_tedges = pd.date_range("2019-12-25", periods=50, freq="D")
    alpha, beta = 4.0, 50.0
    got = fraction_explained_gamma(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        alpha=alpha,
        beta=beta,
        loc=loc,
        direction=direction,
        retardation_factor=retardation_factor,
    )
    exp = quad_oracle(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        alpha=alpha,
        beta=beta,
        loc=loc,
        direction=direction,
        retardation_factor=retardation_factor,
    )
    # Compare on the bins the lazy oracle evaluates (it skips zero-width bins, which the impl fills).
    both = ~np.isnan(got) & ~np.isnan(exp)
    assert both.sum() > 5
    np.testing.assert_allclose(got[both], exp[both], rtol=1e-8, atol=1e-9)
    finite = got[~np.isnan(got)]
    assert np.all((finite >= -1e-12) & (finite <= 1.0 + 1e-12))


def test_mean_std_matches_alpha_beta():
    flow, tedges = variable_flow(seed=4)
    mean_val, std_val, loc = 250.0, 90.0, 20.0
    alpha = (mean_val - loc) ** 2 / std_val**2
    beta = std_val**2 / (mean_val - loc)
    a = fraction_explained_gamma(
        flow=flow, tedges=tedges, cout_tedges=tedges, mean=mean_val, std=std_val, loc=loc, retardation_factor=1.5
    )
    b = fraction_explained_gamma(
        flow=flow, tedges=tedges, cout_tedges=tedges, alpha=alpha, beta=beta, loc=loc, retardation_factor=1.5
    )
    np.testing.assert_allclose(a, b, equal_nan=True, rtol=1e-12)


def test_fully_explained_bin_is_exactly_one():
    """Deep in a long record every pore volume has broken through: coverage saturates at exactly 1.0."""
    tedges = pd.date_range("2020-01-01", periods=400, freq="D")
    flow = np.full(399, 100.0)  # v_total = 39900, far beyond the gamma support
    got = fraction_explained_gamma(flow=flow, tedges=tedges, cout_tedges=tedges, mean=500.0, std=100.0)
    assert got[-1] == 1.0  # exact, because the integral divides by dvol only (not dvol * m0sum)


def test_out_of_record_bins_nan():
    flow, tedges = variable_flow(zero_gap=False)
    cout_tedges = pd.date_range("2019-11-15", periods=12, freq="D")  # before the record
    got = fraction_explained_gamma(flow=flow, tedges=tedges, cout_tedges=cout_tedges, mean=300.0, std=80.0)
    assert np.isnan(got[0])


def test_zero_flow_bin_finite_and_in_unit_interval():
    flow, tedges = variable_flow()  # includes a zero-throughflow gap
    got = fraction_explained_gamma(flow=flow, tedges=tedges, cout_tedges=tedges, mean=300.0, std=80.0, loc=20.0)
    finite = got[~np.isnan(got)]
    assert finite.size > 0
    assert np.all((finite >= -1e-12) & (finite <= 1.0 + 1e-12))


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_zero_flow_bin_value_matches_pointwise_cdf(direction):
    """A zero-throughflow output bin degenerates to the pointwise shifted-gamma CDF at the bin volume.

    Pins the degenerate-branch *value* against an independently hand-computed CDF (not via the
    closed-form Phi path the quad oracle skips), so a dropped loc, a wrong edge, or a missing /R is caught.
    """
    flow, tedges = variable_flow()  # flow[8:11] = 0 -> zero-throughflow output bins
    alpha, beta, loc, r = 4.0, 50.0, 30.0, 1.8
    flow_cum = cumulative_flow_volume(np.asarray(flow, dtype=float), np.diff(tedges_to_days(tedges)))
    v_lo = flow_cum[:-1]
    zero_bins = np.flatnonzero(flow_cum[1:] == v_lo)
    assert zero_bins.size >= 1
    got = fraction_explained_gamma(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        alpha=alpha,
        beta=beta,
        loc=loc,
        direction=direction,
        retardation_factor=r,
    )
    for b in zero_bins:
        threshold = v_lo[b] / r if direction == "extraction_to_infiltration" else (flow_cum[-1] - v_lo[b]) / r
        expected = gamma_dist.cdf(max(threshold - loc, 0.0), alpha, scale=beta)
        np.testing.assert_allclose(got[b], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_negative_or_nan_flow_all_nan(direction):
    flow, tedges = variable_flow(zero_gap=False)
    neg = flow.copy()
    neg[0] = -1.0
    nan = flow.copy()
    nan[2] = np.nan
    for bad in (neg, nan):
        got = fraction_explained_gamma(
            flow=bad, tedges=tedges, cout_tedges=tedges, mean=300.0, std=80.0, direction=direction
        )
        assert np.all(np.isnan(got))


def test_invalid_direction_raises():
    flow, tedges = variable_flow(zero_gap=False)
    with pytest.raises(ValueError, match="direction"):
        fraction_explained_gamma(flow=flow, tedges=tedges, cout_tedges=tedges, mean=300.0, std=80.0, direction="bogus")
