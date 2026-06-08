import numpy as np
import pandas as pd
import pytest

from gwtransport.residence_time import residence_time, residence_time_full

DIRECTIONS = ["extraction_to_infiltration", "infiltration_to_extraction"]


def _variable_flow(n=40, seed=1):
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    rng = np.random.default_rng(seed)
    flow = np.clip(100.0 + 30.0 * np.sin(np.arange(n) / 7.0) + rng.normal(0, 6, n), 5.0, None)
    return flow, tedges


def _constant_flow(n=40, q=100.0):
    return np.full(n, q), pd.date_range("2020-01-01", periods=n + 1, freq="D")


def _manual_weighted_mean(rt, weights):
    """Mass-weighted mean over the valid (non-NaN) streamtubes per output bin."""
    w = np.asarray(weights, dtype=float)[:, None]
    num = np.nansum(rt * w, axis=0)
    den = np.sum(np.where(np.isfinite(rt), w, 0.0), axis=0)
    out = np.full(rt.shape[1], np.nan)
    nonzero = den > 0
    out[nonzero] = num[nonzero] / den[nonzero]
    return out


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("r", [1.0, 2.5])
def test_discrete_equals_manual_weighted_mean(direction, r):
    """residence_time is exactly the valid-renormalized, mass-weighted mean of residence_time_full.

    This is the defining identity of the discrete APVD mean and must hold to machine precision
    under variable flow, retardation, and a non-uniform mass.
    """
    flow, tedges = _variable_flow()
    apv = np.array([300.0, 800.0, 1500.0, 2600.0])
    mass = np.array([0.4, 0.3, 0.2, 0.1])
    got = residence_time(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=apv,
        probability_mass=mass,
        direction=direction,
        retardation_factor=r,
    )
    rt = residence_time_full(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=apv,
        direction=direction,
        retardation_factor=r,
        weighting="flow",
    )
    expected = _manual_weighted_mean(rt, mass)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got, expected, rtol=0, atol=0, equal_nan=True)


def test_uniform_default_matches_explicit_uniform_mass():
    """probability_mass=None is exactly the uniform 1/n distribution."""
    flow, tedges = _variable_flow()
    apv = np.array([250.0, 900.0, 2000.0])
    default = residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=apv)
    explicit = residence_time(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=apv,
        probability_mass=np.full(len(apv), 1.0 / len(apv)),
    )
    np.testing.assert_array_equal(default, explicit)


def test_single_pore_volume_reduces_to_residence_time_full():
    """A one-element APVD collapses to that streamtube's residence_time_full row."""
    flow, tedges = _variable_flow()
    got = residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=750.0)
    ref = residence_time_full(
        flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=750.0, weighting="flow"
    )[0]
    np.testing.assert_array_equal(got, ref)


def test_spinup_all_or_nothing_drops_straddle_streamtube():
    """A streamtube only partially covering a bin is dropped entirely (all-or-nothing spin-up).

    With constant flow the per-streamtube bin average is NaN wherever the look-back parcel leaves
    the record part-way through the bin. residence_time renormalizes over the remaining valid
    streamtubes rather than crediting the dropped one's partial sub-mass -- unlike
    gamma_residence_time, which integrates the partial coverage exactly.
    """
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([200.0, 3000.0])  # the large streamtube is still in spin-up where the small one is informed
    rt = residence_time_full(
        flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=apv, weighting="flow"
    )
    straddle = np.isfinite(rt[0]) & np.isnan(rt[1])
    assert straddle.any(), "test setup must include a bin where only the small streamtube is informed"

    got = residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=apv)
    # Where only the small streamtube is valid, the mean equals its value exactly (the large one,
    # though partially covered, contributes nothing).
    np.testing.assert_allclose(got[straddle], rt[0][straddle], rtol=0, atol=0)


def test_spinup_all_or_nothing_drops_genuinely_partial_streamtube():
    """A streamtube covered only part-way *through a single output bin* is still dropped whole.

    The existing straddle test uses ``tedges_out == flow_tedges``, where a streamtube's bin
    average flips from whole-bin NaN to whole-bin finite at a bin edge -- it never spans the
    record edge mid-bin, so it cannot distinguish all-or-nothing from sub-mass crediting. Here
    the output grid is coarser than the flow grid (5-day bins over a daily record), so the large
    streamtube is genuinely ~50% covered inside the transition bin while its
    ``residence_time_full`` row is NaN. All-or-nothing must drop it: the bin mean equals the
    small streamtube alone, not a coverage-weighted blend with the large one.
    """
    flow = np.full(40, 100.0)
    flow_tedges = pd.date_range("2020-01-01", periods=41, freq="D")
    tedges_out = pd.date_range("2020-01-01", periods=9, freq="5D")  # 8 output bins of 5 days
    apv = np.array([200.0, 2750.0])  # large needs 27.5 days history -> transition inside bin 5 (days 25..30)
    rt = residence_time_full(
        flow=flow, flow_tedges=flow_tedges, tedges_out=tedges_out, aquifer_pore_volumes=apv, weighting="flow"
    )
    # Bin 5 must be exactly the configuration we rely on: small informed, large row NaN.
    assert np.isfinite(rt[0][5]), "test setup: small streamtube must be informed in bin 5"
    assert np.isnan(rt[1][5]), "test setup: large streamtube must straddle the record edge inside bin 5"
    got = residence_time(flow=flow, flow_tedges=flow_tedges, tedges_out=tedges_out, aquifer_pore_volumes=apv)
    # If the large tube's partial sub-mass leaked in, the mean would be pulled toward 27.5; it must
    # stay exactly the small streamtube's 2 days.
    np.testing.assert_allclose(got[5], rt[0][5], rtol=0, atol=0)


def test_loc_shift_matches_manual_weighted_mean():
    """A non-zero gamma ``loc`` is irrelevant to the discrete mean, but the path must still hold.

    The discrete API takes explicit pore volumes, so ``loc`` enters only through the user's chosen
    ``aquifer_pore_volumes``; this exercises large shifted pore volumes (relative to the flow
    record) under variable flow and a non-uniform mass, confirming the valid-renormalized identity
    survives the deep-spin-up regime those large volumes induce.
    """
    flow, tedges = _variable_flow()
    apv = np.array([1200.0, 1850.0, 2900.0])  # loc-shifted regime: all well into spin-up early on
    mass = np.array([0.5, 0.3, 0.2])
    got = residence_time(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=apv,
        probability_mass=mass,
        direction="infiltration_to_extraction",
    )
    rt = residence_time_full(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=apv,
        direction="infiltration_to_extraction",
        weighting="flow",
    )
    expected = _manual_weighted_mean(rt, mass)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    np.testing.assert_allclose(got, expected, rtol=0, atol=0, equal_nan=True)


def test_fully_uncovered_bin_is_nan():
    """A bin with no valid streamtube (entirely in spin-up) is NaN."""
    flow, tedges = _constant_flow(n=40, q=100.0)
    apv = np.array([3500.0, 3800.0])  # both need > 35 days of history; early bins have none
    got = residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=apv)
    assert np.isnan(got[0])


def test_probability_mass_negative_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="non-negative"):
        residence_time(
            flow=flow,
            flow_tedges=tedges,
            tedges_out=tedges,
            aquifer_pore_volumes=[300.0, 600.0],
            probability_mass=[-0.1, 1.1],
        )


def test_probability_mass_not_summing_to_one_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="sum to 1"):
        residence_time(
            flow=flow,
            flow_tedges=tedges,
            tedges_out=tedges,
            aquifer_pore_volumes=[300.0, 600.0],
            probability_mass=[0.4, 0.4],
        )


def test_probability_mass_wrong_shape_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="same shape"):
        residence_time(
            flow=flow,
            flow_tedges=tedges,
            tedges_out=tedges,
            aquifer_pore_volumes=[300.0, 600.0],
            probability_mass=[0.2, 0.3, 0.5],
        )


def test_invalid_direction_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="direction"):
        residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=300.0, direction="bad")
