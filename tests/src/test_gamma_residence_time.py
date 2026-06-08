import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import gamma as gamma_dist

from gwtransport._time import tedges_to_days
from gwtransport.residence_time import gamma_residence_time, residence_time_full
from gwtransport.utils import cumulative_flow_volume

DIRECTIONS = ["extraction_to_infiltration", "infiltration_to_extraction"]


def _constant_flow(n_days=40, q=100.0):
    tedges = pd.date_range("2023-01-01", periods=n_days + 1, freq="D")
    return np.full(n_days, q), tedges


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize(("mean", "std", "loc", "r"), [(300.0, 80.0, 0.0, 1.0), (250.0, 60.0, 90.0, 2.0)])
def test_constant_flow_deep_bin_equals_analytic(direction, mean, std, loc, r):
    """Fully-informed bins under constant flow: tau_bar = R * mean / Q to machine precision.

    With constant flow the residence time of every streamtube is ``R * V_p / Q`` regardless of
    time, so the APVD mean collapses to ``R * E[V_p] / Q = R * mean / Q``. The tiny residual is
    the gamma tail truncated at a high quantile, far below any discretization error.
    """
    q = 100.0
    flow, tedges = _constant_flow(q=q)
    tau = gamma_residence_time(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        mean=mean,
        std=std,
        loc=loc,
        direction=direction,
        retardation_factor=r,
    )
    deep = tau[-1] if direction == "extraction_to_infiltration" else tau[0]
    np.testing.assert_allclose(deep, r * mean / q, atol=0, rtol=1e-11)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_spinup_partial_bin_matches_direct_integral(direction):
    """In the spin-up zone the covered-sub-mass renormalization equals a direct 2-D integral.

    Only streamtubes with enough flow history contribute; the mean is the flow-and-coverage
    weighted ratio of two integrals, not an average of the per-instant ratio. The reference
    integrates ``tau(V_p) = R * V_p / Q`` (constant flow) over the covered length per V_p.
    """
    q, mean, std, r, n = 100.0, 600.0, 200.0, 2.0, 40  # r != 1 exercises the coverage clamp
    flow, tedges = _constant_flow(n_days=n, q=q)
    tau = gamma_residence_time(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        mean=mean,
        std=std,
        direction=direction,
        retardation_factor=r,
    )
    alpha, beta = (mean / std) ** 2, std**2 / mean
    v_end = q * n
    b = 4 if direction == "extraction_to_infiltration" else n - 5  # a partially-covered bin
    v_lo, v_hi = q * b, q * (b + 1)

    def covered_length(vp):
        if direction == "extraction_to_infiltration":
            return max(v_hi - max(v_lo, r * vp), 0.0)
        return max(min(v_hi, v_end - r * vp) - v_lo, 0.0)

    hi = gamma_dist.ppf(1 - 1e-12, alpha, scale=beta)
    num, _ = quad(
        lambda vp: (r * vp / q) * covered_length(vp) * gamma_dist.pdf(vp, alpha, scale=beta), 0, hi, limit=200
    )
    den, _ = quad(lambda vp: covered_length(vp) * gamma_dist.pdf(vp, alpha, scale=beta), 0, hi, limit=200)
    np.testing.assert_allclose(tau[b], num / den, rtol=1e-9)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_narrow_gamma_reduces_to_single_pore_volume(direction):
    """A near-degenerate gamma (std -> 0) reduces to the single-pore-volume bin mean.

    Because the residence time is piecewise-linear in V_p, the mean over a narrow distribution
    centred at ``mean`` equals the residence time at ``mean`` -- i.e. ``residence_time_full`` for
    a single pore volume. This cross-checks the closed form against the per-pore-volume path.
    """
    rng = np.random.default_rng(0)
    n = 40
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.clip(100 * (1 + 0.3 * np.sin(np.arange(n) / 9)) + rng.normal(0, 4, n), 5, None)
    mean = 433.0  # generic value, not on a flow-edge breakpoint
    # std small enough that the std -> 0 limit is reached to ~3e-9 (residual scales with std^2);
    # loc != 0 exercises the loc shift in the partial moments.
    tau = gamma_residence_time(
        flow=flow, flow_tedges=tedges, tedges_out=tedges, mean=mean, std=0.05, loc=120.0, direction=direction
    )
    ref = residence_time_full(
        flow=flow, flow_tedges=tedges, tedges_out=tedges, aquifer_pore_volumes=mean, direction=direction
    )[0]
    mask = np.isfinite(tau) & np.isfinite(ref)
    assert mask.sum() > n // 2
    np.testing.assert_allclose(tau[mask], ref[mask], atol=1e-7, rtol=1e-7)


def test_alpha_beta_matches_mean_std():
    """The (alpha, beta) parameterization matches the equivalent (mean, std) one exactly."""
    flow, tedges = _constant_flow()
    mean, std = 350.0, 90.0
    alpha, beta = (mean / std) ** 2, std**2 / mean
    by_moments = gamma_residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, mean=mean, std=std)
    by_shape = gamma_residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, alpha=alpha, beta=beta)
    np.testing.assert_allclose(by_moments, by_shape, atol=0, rtol=1e-12, equal_nan=True)


def test_invalid_direction_raises():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError, match="direction"):
        gamma_residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges, mean=300.0, std=80.0, direction="bad")


def test_requires_gamma_parameters():
    flow, tedges = _constant_flow()
    with pytest.raises(ValueError):
        gamma_residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges)


def test_bins_outside_record_are_nan():
    """Output bins extending beyond the flow record are NaN, like residence_time_full."""
    flow, tedges = _constant_flow(n_days=20)
    tedges_out = pd.date_range("2023-01-01", periods=31, freq="D")  # 10 days past the record
    tau = gamma_residence_time(flow=flow, flow_tedges=tedges, tedges_out=tedges_out, mean=300.0, std=80.0)
    assert np.isnan(tau[-1])


def _quad_bin_moments(flow, tedges, v_lo, v_hi, mean, std, loc, r, direction):
    """Independent reference: numerator and denominator of the flow-weighted bin mean by quadrature.

    The bin mean is ``num / den`` with ``num = int f(Vp) G_b(Vp) dVp`` and ``den = int f(Vp)
    L_b(Vp) dVp``: ``G_b`` the per-streamtube time integral and ``L_b`` the covered length over the
    bin's cumulative-volume window. The inverse cumulative-volume map ``T`` is built directly with
    ``np.interp`` and both integrals are taken with ``scipy.quad`` -- independent of this module's
    phi antiderivative and fixed-band machinery. The kink times of ``T`` are passed to the inner
    quadrature as break points so the piecewise-linear integrand is integrated cleanly.
    """
    alpha, beta = (mean - loc) ** 2 / std**2, std**2 / (mean - loc)
    td = tedges_to_days(tedges)
    flow_cum = cumulative_flow_volume(np.asarray(flow, float), np.diff(td), strictly_monotone=True)
    v_end = flow_cum[-1]
    sign = -1.0 if direction == "extraction_to_infiltration" else 1.0

    inner_limit = 2 * len(flow_cum) + 50

    def inner(vp):  # (time integral over the covered part of the bin, covered length)
        if sign < 0:
            lo, hi = max(v_lo, r * vp), v_hi
            if hi <= lo:
                return 0.0, 0.0
            kinks = sorted(v for v in np.concatenate([flow_cum, flow_cum + r * vp]) if lo < v < hi)
            g, _ = quad(
                lambda v: np.interp(v, flow_cum, td) - np.interp(v - r * vp, flow_cum, td),
                lo,
                hi,
                points=kinks,
                limit=inner_limit,
            )
            return g, hi - lo
        lo, hi = v_lo, min(v_hi, v_end - r * vp)
        if hi <= lo:
            return 0.0, 0.0
        kinks = sorted(v for v in np.concatenate([flow_cum, flow_cum - r * vp]) if lo < v < hi)
        g, _ = quad(
            lambda v: np.interp(v + r * vp, flow_cum, td) - np.interp(v, flow_cum, td),
            lo,
            hi,
            points=kinks,
            limit=inner_limit,
        )
        return g, hi - lo

    hi_vp = loc + gamma_dist.ppf(1 - 1e-12, alpha, scale=beta)
    num, _ = quad(lambda vp: gamma_dist.pdf(vp - loc, alpha, scale=beta) * inner(vp)[0], loc, hi_vp, limit=200)
    den, _ = quad(lambda vp: gamma_dist.pdf(vp - loc, alpha, scale=beta) * inner(vp)[1], loc, hi_vp, limit=200)
    return num, den


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_wide_gamma_matches_double_quad(direction):
    """Wide gamma + variable flow: output bins match an independent double-quadrature reference.

    A wide distribution makes the look-back/forward band span many flow bins, exercising the full
    piecewise-quadratic G machinery -- the regime where a per-bin looped engine accrues FP noise.
    Tested against ``scipy.quad`` over both cumulative volume and the gamma density (rtol=1e-7, the
    quadrature floor), on bins ranging from spin-up to fully informed. Bins whose covered sub-mass
    is a tiny tail sliver are skipped, since there the quadrature reference -- not the closed form
    -- is the inaccurate one.
    """
    n = 14
    tedges = pd.date_range("2021-01-01", periods=n + 1, freq="D")
    rng = np.random.default_rng(3)
    flow = np.clip(100.0 + 60.0 * np.sin(np.arange(n) / 3.0) + rng.normal(0, 10, n), 10.0, None)
    mean, std, loc, r = 700.0, 450.0, 50.0, 1.6
    tau = gamma_residence_time(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        mean=mean,
        std=std,
        loc=loc,
        direction=direction,
        retardation_factor=r,
    )
    flow_cum = cumulative_flow_volume(flow, np.diff(tedges_to_days(tedges)), strictly_monotone=True)
    tested = 0
    for b in (3, 5, 7, 9, 11):
        num, den = _quad_bin_moments(flow, tedges, flow_cum[b], flow_cum[b + 1], mean, std, loc, r, direction)
        if den < 1e-6 or not np.isfinite(tau[b]):  # negligible covered sub-mass -> quad-fragile, skip
            continue
        np.testing.assert_allclose(tau[b], num / den, rtol=1e-7)
        tested += 1
    assert tested >= 3, "wide-gamma reference should validate several bins from spin-up to deep"


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_tiling_boundary_invariance(direction):
    """Tiling the output bins does not change the result: 1-bin-per-tile equals a single tile.

    band_width is computed globally, so each bin's integration pieces are independent of the tile
    it lands in; the loop boundaries must not couple bins. Compared NaN-pattern and value.
    """
    n = 60
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    rng = np.random.default_rng(5)
    flow = np.clip(110.0 + 40.0 * np.sin(np.arange(n) / 9.0) + rng.normal(0, 7, n), 8.0, None)
    common = {
        "flow": flow,
        "flow_tedges": tedges,
        "tedges_out": tedges,
        "mean": 900.0,
        "std": 600.0,
        "loc": 80.0,
        "direction": direction,
        "retardation_factor": 1.3,
    }
    one_tile = gamma_residence_time(**common, _max_tile_elements=10**18)
    tiny_tile = gamma_residence_time(**common, _max_tile_elements=1)
    np.testing.assert_array_equal(np.isnan(one_tile), np.isnan(tiny_tile))
    np.testing.assert_allclose(one_tile, tiny_tile, rtol=1e-12, equal_nan=True)
