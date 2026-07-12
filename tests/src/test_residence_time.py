import numpy as np
import pandas as pd
import pytest

from gwtransport.residence_time import (
    fraction_explained_full,
    fraction_explained_gamma,
    fraction_explained_mean,
    full,
    gamma,
    mean,
)

DIRECTIONS = ["extraction_to_infiltration", "infiltration_to_extraction"]

# The six public residence-time functions, each wrapped so it can be called with a common
# (flow, tedges, cout_tedges, direction) signature for the all-NaN refusal contract test.
_APV = np.array([1000.0, 2500.0])
_GAMMA_KW = {"mean": 2000.0, "std": 600.0}
_RESIDENCE_FUNCS = {
    "full": lambda flow, te, cte, d: full(
        flow=flow, tedges=te, cout_tedges=cte, aquifer_pore_volumes=_APV, direction=d
    ),
    "mean": lambda flow, te, cte, d: mean(
        flow=flow, tedges=te, cout_tedges=cte, aquifer_pore_volumes=_APV, direction=d
    ),
    "gamma": lambda flow, te, cte, d: gamma(flow=flow, tedges=te, cout_tedges=cte, direction=d, **_GAMMA_KW),
    "fraction_explained_full": lambda flow, te, cte, d: fraction_explained_full(
        flow=flow, tedges=te, cout_tedges=cte, aquifer_pore_volumes=_APV, direction=d
    ),
    "fraction_explained_mean": lambda flow, te, cte, d: fraction_explained_mean(
        flow=flow, tedges=te, cout_tedges=cte, aquifer_pore_volumes=_APV, direction=d
    ),
    "fraction_explained_gamma": lambda flow, te, cte, d: fraction_explained_gamma(
        flow=flow, tedges=te, cout_tedges=cte, direction=d, **_GAMMA_KW
    ),
}


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
    """mean is exactly the mean over the valid streamtubes of full.

    Holds to machine precision for both spin-up policies. Under ``spinup='constant'`` every
    in-record streamtube is finite (warm-started), so it is a plain mean; under ``spinup=None`` it
    renormalizes over the streamtubes that have broken through, dropping the rest all-or-nothing.
    """
    flow, tedges = _variable_flow()
    apv = np.array([250.0, 600.0, 900.0, 1200.0])
    got = mean(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        direction=direction,
        retardation_factor=r,
        spinup=spinup,
    )
    rt = full(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        direction=direction,
        retardation_factor=r,
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
    got = mean(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=apv,
        direction=direction,
        retardation_factor=r,
    )
    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, r * apv.mean() / q, rtol=1e-13)


def test_single_pore_volume_reduces_to_full():
    """A one-element APVD collapses to that streamtube's full row."""
    flow, tedges = _variable_flow()
    got = mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=750.0)
    ref = full(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=750.0)[0]
    np.testing.assert_array_equal(got, ref)


def test_spinup_none_drops_genuinely_partial_streamtube():
    """With spinup=None a streamtube covered only part-way through a bin is dropped whole.

    The output grid is coarser than the flow grid (5-day bins over a daily record), so the large
    streamtube is genuinely ~50% covered inside the transition bin while its ``full``
    row is NaN. All-or-nothing must drop it: the bin mean equals the small streamtube alone, not a
    coverage-weighted blend.
    """
    flow = np.full(40, 100.0)
    tedges = pd.date_range("2020-01-01", periods=41, freq="D")
    cout_tedges = pd.date_range("2020-01-01", periods=9, freq="5D")  # 8 output bins of 5 days
    apv = np.array([200.0, 2750.0])  # large needs 27.5 days history -> transition inside bin 5
    rt = full(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=apv,
        spinup=None,
    )
    assert np.isfinite(rt[0][5]), "test setup: small streamtube must be informed in bin 5"
    assert np.isnan(rt[1][5]), "test setup: large streamtube must straddle the record edge inside bin 5"
    got = mean(flow=flow, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_volumes=apv, spinup=None)
    # If the large tube's partial coverage leaked in, the mean would be pulled toward 27.5 days.
    np.testing.assert_allclose(got[5], rt[0][5], rtol=0, atol=0)


def test_spinup_none_fully_uncovered_bin_is_nan():
    """With spinup=None, a bin with no valid streamtube (all in spin-up) is NaN."""
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([3500.0, 3800.0])  # both need > 35 days of history; the first bins have none
    got = mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=None)
    assert np.isnan(got[0])


@pytest.mark.parametrize("spinup", ["constant", None])
def test_out_of_window_bin_is_nan(spinup):
    """Output bins beyond the flow record are NaN under either spin-up policy."""
    flow, tedges = _constant_flow(n=20, q=100.0)
    cout_tedges = pd.date_range("2020-01-01", periods=31, freq="D")  # extends 10 days past the record
    got = mean(flow=flow, tedges=tedges, cout_tedges=cout_tedges, aquifer_pore_volumes=300.0, spinup=spinup)
    assert np.isnan(got[-1])


def test_fraction_explained_marks_spinup_while_full_warm_starts():
    """fraction_explained_mean flags the spin-up region where full's warm-start is finite.

    The default ``spinup='constant'`` makes :func:`full` finite at every in-record bin, so the
    means alone no longer reveal the spin-up. :func:`fraction_explained_mean` is the diagnostic
    that locates it: its advective coverage is below 1 in the spin-up region (where the larger
    streamtube's look-back leaves the record) and reaches exactly 1 once both streamtubes are
    informed. This replaces the old ``residence_time_series`` NaN probe.
    """
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([200.0, 1500.0])
    warm = full(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv)
    frac = fraction_explained_mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv)
    assert not np.isnan(warm).any(), "full default warm-start must be finite in-record"
    assert ((frac < 1.0) & (frac >= 0.0)).any(), "fraction_explained_mean must flag the spin-up region (< 1)"
    assert (frac == 1.0).any(), "fraction_explained_mean must reach 1 outside the spin-up"
    # Coverage is non-decreasing with time under constant flow and rises into the fully-informed
    # plateau, so the spin-up bins are a contiguous leading block of below-1 coverage.
    assert np.all(np.diff(frac) >= 0.0)
    first_full = int(np.argmax(frac == 1.0))
    assert np.all(frac[:first_full] < 1.0)
    assert np.all(frac[first_full:] == 1.0)


def test_invalid_direction_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="direction"):
        mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=300.0, direction="bad")


def test_invalid_spinup_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="spinup"):
        mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=300.0, spinup="bad")
    # The unified contract is {'constant'} | None | float in [0, 1]: only floats outside [0, 1] raise.
    with pytest.raises(ValueError, match="spinup"):
        mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=300.0, spinup=1.5)
    with pytest.raises(ValueError, match="spinup"):
        mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=300.0, spinup=-0.1)


@pytest.mark.parametrize("spinup", ["constant", None, 0.0, 0.5, 1.0])
def test_residence_time_accepts_unified_spinup_set(spinup):
    """mean accepts each member of the shared {'constant', None, float in [0, 1]} set."""
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([200.0, 400.0, 600.0, 800.0])
    got = mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=spinup)
    assert got.shape == (40,)
    # Deep in the record every streamtube has broken through, so the bin mean is the plain APV mean.
    np.testing.assert_allclose(got[-1], apv.mean() / 100.0, rtol=1e-12)


def test_residence_time_spinup_none_equals_zero_float():
    """spinup=None and spinup=0.0 are the same lenient policy (emit wherever any streamtube valid)."""
    flow, tedges = _variable_flow(n=60)
    apv = np.array([200.0, 1500.0, 3000.0])
    got_none = mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=None)
    got_zero = mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=0.0)
    np.testing.assert_array_equal(got_none, got_zero)  # bit-identical, including NaN placement


def test_residence_time_float_threshold_gates_covered_fraction():
    """A higher float threshold NaNs more spin-up bins (monotone in the covered-streamtube fraction).

    Where a bin's mean is finite it is independent of the threshold (renormalization is unchanged);
    the threshold only decides whether the bin is emitted at all.
    """
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([200.0, 1500.0, 2800.0])  # staggered break-through inside the spin-up
    got_zero = mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=0.0)
    got_two_thirds = mean(flow=flow, tedges=tedges, cout_tedges=tedges, aquifer_pore_volumes=apv, spinup=0.7)
    nan_zero = np.isnan(got_zero)
    nan_high = np.isnan(got_two_thirds)
    assert np.all(nan_zero <= nan_high)  # the stricter threshold is a NaN superset
    assert nan_high.sum() > nan_zero.sum()  # and strictly NaNs more in the staggered spin-up
    both_finite = ~nan_zero & ~nan_high
    np.testing.assert_array_equal(got_zero[both_finite], got_two_thirds[both_finite])


def test_spinup_one_is_strictest_gate():
    """spinup=1.0 is accepted and requires full coverage (regression for the [0, 1) -> [0, 1] widening).

    Previously ``spinup=1.0`` raised (the residence-time contract was ``[0, 1)`` while advection uses
    ``[0, 1]``). It must now be the strictest threshold: a bin is emitted only where every streamtube
    has broken through, and there the renormalized mean equals the warm-started default.
    """
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([200.0, 1500.0, 2800.0])  # staggered break-through: an early partial-coverage region
    common = {"flow": flow, "tedges": tedges, "cout_tedges": tedges, "aquifer_pore_volumes": apv}
    got_one = mean(**common, spinup=1.0)  # must not raise
    got_const = mean(**common, spinup="constant")
    # 1.0 emits iff every streamtube is valid under the strict per-pore-volume map.
    all_valid = np.isfinite(full(**common, spinup=None)).all(axis=0)
    # a genuine partial-coverage case: some fully-covered bins, some not
    assert all_valid.any()
    assert not all_valid.all()
    np.testing.assert_array_equal(np.isfinite(got_one), all_valid)
    # Where emitted it is the full-coverage mean, bit-identical to the warm-started default there.
    np.testing.assert_array_equal(got_one[all_valid], got_const[all_valid])
    # gamma also accepts the widened endpoint without raising.
    assert gamma(flow=flow, tedges=tedges, cout_tedges=tedges, mean=1500.0, std=400.0, spinup=1.0).shape == (40,)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_zero_flow_boundary_warmstart_is_finite(direction):
    """A Q=0 boundary bin must not corrupt the constant warm-start extrapolation.

    Regression: ``inv_q = dt / dV`` from a zero-flow boundary bin used the strictly-monotone ulp
    bump as ``dV``, giving ~1e13-day spin-up residence times. The slope must instead come from the
    nearest strictly-positive flow. The fix only touches the out-of-record warm-start, so the
    deep-interior bins (look-back fully inside the record) stay at the steady value Vp/Q exactly.
    """
    tedges = pd.date_range("2020-01-01", periods=10, freq="D")
    flow = np.array([0.0, 100, 100, 100, 100, 100, 100, 100, 0.0])  # zero first AND last bin
    pv, q = 200.0, 100.0
    got_full = full(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=np.array([pv]),
        direction=direction,
        spinup="constant",
    )[0]
    got_mean = mean(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        aquifer_pore_volumes=np.array([pv]),
        direction=direction,
        spinup="constant",
    )
    got_gamma = gamma(
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        mean=pv,
        std=0.5,
        direction=direction,
        spinup="constant",
    )
    for name, got in (("full", got_full), ("mean", got_mean), ("gamma", got_gamma)):
        assert np.all(np.nan_to_num(np.abs(got)) < 1e4), f"{name} {direction} blew up: max={np.nanmax(np.abs(got))}"
    # Single pore volume: deep-interior bins (look-back fully inside the record) == steady Vp/Q exactly.
    np.testing.assert_allclose(got_full[3:6], pv / q, atol=1e-12)
    np.testing.assert_allclose(got_mean[3:6], pv / q, atol=1e-12)


def test_gamma_zero_flow_output_bins_match_full_oracle():
    """Zero-throughflow output bins must return the pointwise gamma-mean, not cancellation garbage.

    Regression: a Q=0 output bin has a cumulative-volume window only as wide as the ulp bump, so the
    closed-form bin-average ``num/den`` cancelled to ~1e10 days. The result must instead equal the
    well-defined zero-width-bin limit, which ``full`` computes correctly.
    """
    tedges = pd.date_range("2020-01-01", periods=10, freq="D")
    flow = np.array([100.0, 100, 100, 100, 0, 0, 100, 100, 100])  # mid-record shutdown
    for direction in DIRECTIONS:
        got = gamma(
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            mean=300.0,
            std=0.1,
            direction=direction,
            spinup="constant",
        )
        ref = full(
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=np.array([300.0]),
            direction=direction,
            spinup="constant",
        )[0]
        assert np.nanmax(np.abs(got)) < 1e4, f"{direction}: zero-flow bins blew up to {np.nanmax(np.abs(got))}"
        # Narrow gamma (std=0.1) ~ single pore volume at the mean, so it tracks the full oracle.
        valid = np.isfinite(got) & np.isfinite(ref)
        np.testing.assert_allclose(got[valid], ref[valid], atol=0.05)


@pytest.mark.parametrize("func_name", list(_RESIDENCE_FUNCS))
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("bad_kind", ["negative", "nan"])
def test_negative_or_nan_flow_returns_all_nan(func_name, direction, bad_kind):
    """Negative or NaN flow makes every public residence_time function refuse (all-NaN), not raise.

    Regression test for the documented refusal convention (issue #301, PR #295): a negative or
    NaN flow bin makes the cumulative-volume map V(t) non-monotone or undefined, so the whole
    array/series is returned as NaN rather than raising or returning partial finite values. The
    contract was documentation-only until now. Guarding all six public functions in both
    directions keeps a future refactor from silently starting to return partial values or to
    raise.
    """
    func = _RESIDENCE_FUNCS[func_name]
    n = 30
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges

    # Sanity guard against a trivial pass: valid flow must yield at least one finite value, so the
    # all-NaN result below is attributable to the bad flow rather than to a degenerate fixture.
    valid_flow = np.full(n, 100.0)
    valid_out = func(valid_flow, tedges, cout_tedges, direction)
    assert np.isfinite(valid_out).any(), "fixture sanity: valid flow must produce some finite output"

    bad_flow = np.full(n, 100.0)
    bad_flow[10] = -50.0 if bad_kind == "negative" else np.nan

    out = func(bad_flow, tedges, cout_tedges, direction)
    assert np.isnan(out).all(), f"{func_name}/{direction}/{bad_kind}: expected all-NaN, got finite entries"
