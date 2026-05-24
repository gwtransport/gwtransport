import numpy as np
import pandas as pd
import pytest

from gwtransport.residence_time import freundlich_retardation, residence_time, residence_time_mean
from gwtransport.utils import compute_time_edges


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


@pytest.mark.parametrize("direction", ["extraction_to_infiltration", "infiltration_to_extraction"])
def test_basic_constant_flow_residence_time(direction):
    """Constant flow Q with pore volume V_p gives tau = V_p / Q exactly in both directions.

    Under piecewise-constant Q the cumulative-flow inversion is exact, so the residence time
    at every informed index equals V_p / Q to machine precision. This replaces three
    near-identical legacy tests that re-stated the same physics through different timing-edge
    construction paths -- the timing-equivalence is now asserted separately by
    ``test_timing_methods_yield_identical_results``.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction=direction,
    )

    valid = ~np.isnan(result[0])
    assert np.any(valid)
    np.testing.assert_allclose(result[0, valid], 2.0, atol=0, rtol=1e-13)


def test_retardation_factor():
    """Test the effect of retardation factor."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    result_no_retardation = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=1.0,
        direction="extraction_to_infiltration",
    )

    result_with_retardation = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=2.0,
        direction="extraction_to_infiltration",
    )

    # With constant flow, retardation factor R must scale residence time by exactly R.
    # The linear-interpolation step on a piecewise-linear cumulative-flow curve is exact for
    # constant flow, so the ratio is 2.0 to machine precision wherever both are valid.
    valid_mask = ~np.isnan(result_no_retardation[0]) & ~np.isnan(result_with_retardation[0])
    assert np.any(valid_mask)
    ratio = result_with_retardation[0, valid_mask] / result_no_retardation[0, valid_mask]
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-12)


def test_custom_index():
    """Test using custom index for results."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    custom_dates = pd.date_range(start="2023-01-02", end="2023-01-04", freq="D")
    pore_volume = 200.0

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        index=custom_dates,
        direction="extraction_to_infiltration",
    )

    assert result.shape[1] == len(custom_dates)


def test_return_numpy_array():
    """Test returning results as numpy array (default behavior)."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    assert isinstance(result, np.ndarray)
    # Should return for the center points of flow bins
    expected_length = len(flow_tedges) - 1
    assert result.shape == (1, expected_length)


def test_multiple_pore_volumes():
    """Test handling of multiple pore volumes."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volumes = np.array([200.0, 300.0, 400.0])

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volumes,
        direction="extraction_to_infiltration",
    )

    assert result.shape[0] == len(pore_volumes)
    expected_time_points = len(flow_tedges) - 1
    assert result.shape[1] == expected_time_points

    # Residence times should increase with increasing pore volumes
    valid_mask = ~np.isnan(result[:, -1])
    if np.sum(valid_mask) > 1:
        assert np.all(np.diff(result[valid_mask, -1]) > 0)


def test_invalid_direction():
    """Test that invalid direction raises ValueError."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 200.0

    with pytest.raises(
        ValueError, match="direction should be 'extraction_to_infiltration' or 'infiltration_to_extraction'"
    ):
        residence_time(flow=flow_values, flow_tedges=flow_tedges, aquifer_pore_volumes=pore_volume, direction="invalid")


def test_missing_flow_timing_parameters():
    """Test that missing flow timing parameters raises TypeError."""
    flow_values = np.full(5, 100.0)
    pore_volume = 200.0

    # Since flow_tedges is now a required parameter (no longer has | None),
    # Python raises TypeError when it's not provided
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'flow_tedges'"):
        residence_time(flow=flow_values, aquifer_pore_volumes=pore_volume, direction="extraction_to_infiltration")  # type: ignore[missing-argument]


def test_flow_tedges_length_validation():
    """Test validation of flow_tedges length."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges), 100.0)  # Wrong length (should be len-1)
    pore_volume = 200.0

    with pytest.raises(ValueError, match="tedges must have one more element than flow"):
        residence_time(
            flow=flow_values,
            flow_tedges=flow_tedges,
            aquifer_pore_volumes=pore_volume,
            direction="extraction_to_infiltration",
        )


def test_flow_tstart_length_validation():
    """Test validation of flow_tstart length when converting to tedges."""
    flow_tstart = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
    flow_values = np.full(len(flow_tstart) + 1, 100.0)  # Wrong length

    with pytest.raises(ValueError, match="tstart must have the same number of elements as flow"):
        # This should fail during compute_time_edges
        compute_time_edges(tedges=None, tstart=flow_tstart, tend=None, number_of_bins=len(flow_values))


def test_flow_tend_length_validation():
    """Test validation of flow_tend length when converting to tedges."""
    flow_tend = pd.date_range(start="2023-01-02", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tend) + 1, 100.0)  # Wrong length

    with pytest.raises(ValueError, match="tend must have the same number of elements as flow"):
        # This should fail during compute_time_edges
        compute_time_edges(tedges=None, tstart=None, tend=flow_tend, number_of_bins=len(flow_values))


def test_edge_cases_zero_flow():
    """All-zero flow gives all-NaN residence times.

    With Q identically zero, ``flow_cum`` is all zeros and the ulp-bump regularization makes
    it strictly increasing in the bottom ~16 ulps. Every target volume ``flow_cum_at_index +/-
    R * V_p`` (with V_p > 0) sits outside the data range, so ``np.interp(..., left=nan,
    right=nan)`` returns NaN. The previous union assertion ``isnan | isinf`` accepted two
    physically distinct outcomes; tighten to the actual all-NaN behavior.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    zero_flow = np.zeros(len(flow_tedges) - 1)
    pore_volume = 100.0

    result = residence_time(
        flow=zero_flow,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    assert np.all(np.isnan(result))


def test_zero_flow_plateau_kink_consistency():
    """At the kink time t* where the source crosses a Q = 0 plateau, residence_time returns
    the midpoint of the two valid limits.

    For ``flow = [100, 0, 0, 100]`` (1-day bins), ``V_p = 50``, ``R = 1``, ``V(t)`` is flat at
    100 over ``t in [1, 3]``. At ``t* = 3.5`` (where ``V(t*) - 50 = 100``), the source jumps
    from ``t_in = 1`` (left limit, ``tau = 2.5``) to ``t_in = 3`` (right limit, ``tau = 0.5``).
    The ulp-bump regularization in :func:`residence_time` resolves the multi-valued ``V_inv``
    deterministically to the midpoint ``t_in = 2``, giving ``tau = 1.5``. This is the
    consistent counterpart to :func:`residence_time_mean`'s exact step-integral over the
    same kink.
    """
    flow_tedges = pd.date_range(start="2023-01-01", periods=5, freq="D")
    flow_values = np.array([100.0, 0.0, 0.0, 100.0])
    pore_volume = 50.0
    index = pd.DatetimeIndex(["2023-01-04 12:00:00"])  # t = 3.5 d

    rt = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        index=index,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    np.testing.assert_allclose(rt[0, 0], 1.5, atol=0, rtol=1e-13)


def test_edge_cases_very_large_pore_volume():
    """Test edge cases with very large pore volumes."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    large_pore_volume = 1e6

    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=large_pore_volume,
        direction="extraction_to_infiltration",
    )

    # Very large pore volume should result in NaN values for recent times
    assert np.all(np.isnan(result))


def test_negative_flow():
    """Negative flow yields all-NaN, not just some-NaN.

    The legacy assertion ``not np.all(np.isfinite(result))`` admitted partial-NaN returns;
    the function spec is ``np.full(..., np.nan)``, so tighten to exact all-NaN like
    ``test_edge_cases_zero_flow`` and ``test_nan_flow_returns_nan``.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    negative_flow = np.full(len(flow_tedges) - 1, -100.0)
    pore_volume = 200.0

    result = residence_time(
        flow=negative_flow,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    assert np.all(np.isnan(result))


def test_flow_variations(constant_flow_data):
    """Doubling constant flow exactly halves the residence time.

    With variable flow, doubling flow does *not* halve residence time exactly: the
    inverse-cum-flow inversion satisfies V_p = integral_{t-tau}^{t} Q(s) ds, and
    rescaling Q -> 2Q gives V_p = integral_{t-tau'}^{t} 2 * Q(s) ds, i.e. the new tau'
    is the time over which the *original* Q integrates to V_p / 2 -- generally not
    tau / 2. Only for constant Q is the relationship exact. This test therefore uses a
    constant-flow fixture to enforce the exact halving to machine precision.
    """
    flow_values, flow_tedges = constant_flow_data
    pore_volume = 200.0  # m³, gives valid results across the whole window

    result1 = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    result2 = residence_time(
        flow=flow_values * 2,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    valid_mask = ~np.isnan(result1[0]) & ~np.isnan(result2[0])
    assert np.any(valid_mask)
    ratio = result1[0, valid_mask] / result2[0, valid_mask]
    np.testing.assert_allclose(ratio, 2.0, rtol=1e-12)


def test_timing_methods_yield_identical_results():
    """The three timing inputs (tedges, tstart, tend) produce bit-identical residence times.

    ``compute_time_edges`` reconstructs the same ``flow_tedges`` regardless of which timing
    parameter is supplied, so the resulting ``residence_time`` outputs must agree to bit
    precision. The previous assertion only checked ``np.any(valid_mask)`` -- a mutation that
    skews indices by one would not have failed it.
    """
    dates = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.array([100.0, 110.0, 105.0, 95.0, 98.0])
    pore_volume = 200.0

    common = {
        "flow": flow_values,
        "aquifer_pore_volumes": pore_volume,
        "direction": "extraction_to_infiltration",
    }
    result_tedges = residence_time(flow_tedges=dates, **common)
    flow_tedges_from_tstart = compute_time_edges(
        tedges=None, tstart=dates[:-1], tend=None, number_of_bins=len(flow_values)
    )
    result_tstart = residence_time(flow_tedges=flow_tedges_from_tstart, **common)
    flow_tedges_from_tend = compute_time_edges(
        tedges=None, tstart=None, tend=dates[1:], number_of_bins=len(flow_values)
    )
    result_tend = residence_time(flow_tedges=flow_tedges_from_tend, **common)

    np.testing.assert_array_equal(result_tedges, result_tstart)
    np.testing.assert_array_equal(result_tedges, result_tend)


def test_array_like_flow_input():
    """Test that array-like flow inputs (lists, numpy arrays) work correctly."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    pore_volume = 200.0

    # Test with list
    flow_list = [100.0, 110.0, 105.0, 95.0, 98.0]
    result_list = residence_time(
        flow=flow_list,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Test with numpy array
    flow_array = np.array([100.0, 110.0, 105.0, 95.0, 98.0])
    result_array = residence_time(
        flow=flow_array,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )

    # Results should be identical
    np.testing.assert_array_equal(result_list, result_array)


def test_single_pore_volume_vs_array():
    """Test that single pore volume and array with one element give same results."""
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)

    # Single float
    result_float = residence_time(
        flow=flow_values, flow_tedges=flow_tedges, aquifer_pore_volumes=200.0, direction="extraction_to_infiltration"
    )

    # Array with single element
    result_array = residence_time(
        flow=flow_values, flow_tedges=flow_tedges, aquifer_pore_volumes=[200.0], direction="extraction_to_infiltration"
    )

    # Results should be identical
    np.testing.assert_array_equal(result_float, result_array)


def test_infiltration_vs_extraction_symmetry():
    """Under constant Q, both directions give the same constant tau = V_p / Q at informed indices.

    The legacy test of the same name only asserted that each direction was internally constant,
    never compared the two directions to each other. Under constant Q with R = 1 the residence
    time is V_p / Q regardless of direction (the inverse cumulative-flow equation
    ``V_p = integral_{t-tau}^{t} Q ds`` reduces to V_p = Q * tau identically in both directions),
    so the *values* at every informed index must agree to machine precision -- which is what
    this test now asserts.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    flow_values = np.full(len(flow_tedges) - 1, 100.0)
    pore_volume = 300.0
    expected_tau = pore_volume / flow_values[0]  # 3.0 days

    result_extraction = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    result_infiltration = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="infiltration_to_extraction",
    )

    extraction_valid = ~np.isnan(result_extraction[0])
    infiltration_valid = ~np.isnan(result_infiltration[0])
    assert np.any(extraction_valid)
    assert np.any(infiltration_valid)

    # Each direction is internally constant at exactly V_p / Q ...
    np.testing.assert_allclose(result_extraction[0, extraction_valid], expected_tau, atol=0, rtol=1e-13)
    np.testing.assert_allclose(result_infiltration[0, infiltration_valid], expected_tau, atol=0, rtol=1e-13)

    # ... and the two directions agree on the informed value.
    np.testing.assert_allclose(
        result_extraction[0, extraction_valid],
        result_infiltration[0, infiltration_valid],
        atol=0,
        rtol=1e-13,
    )


def test_freundlich_retardation_analytical():
    """Test freundlich_retardation against hand-computed analytical values.

    R = 1 + (rho_b / theta) * k_f * n * C^(n-1)
    """
    concentration = np.array([1.0])
    k_f = 0.5
    n = 0.8
    rho_b = 1500.0
    theta = 0.3

    result = freundlich_retardation(
        concentration=concentration,
        freundlich_k=k_f,
        freundlich_n=n,
        bulk_density=rho_b,
        porosity=theta,
    )

    # R = 1 + (1500/0.3) * 0.5 * 0.8 * 1.0^(-0.2) = 1 + 2000
    expected = 1.0 + (rho_b / theta) * k_f * n * np.power(1.0, n - 1)
    np.testing.assert_allclose(result, expected, rtol=1e-12)


def test_freundlich_retardation_concentration_dependence():
    """Test that freundlich_retardation varies correctly with concentration."""
    concentrations = np.array([0.1, 1.0, 10.0])
    k_f = 0.01
    n = 0.7
    rho_b = 1600.0
    theta = 0.35

    result = freundlich_retardation(
        concentration=concentrations,
        freundlich_k=k_f,
        freundlich_n=n,
        bulk_density=rho_b,
        porosity=theta,
    )

    # For n < 1, retardation decreases with increasing concentration
    assert result[0] > result[1] > result[2]

    # Check exact values
    expected = 1.0 + (rho_b / theta) * k_f * n * np.power(concentrations, n - 1)
    np.testing.assert_allclose(result, expected, rtol=1e-13)


def test_freundlich_retardation_zero_concentration_n_lt_one_raises():
    """For n < 1 the retardation factor diverges as C -> 0; non-positive C must raise."""
    # Zero concentration with n < 1: must raise
    with pytest.raises(ValueError, match="concentration must be strictly positive when freundlich_n < 1"):
        freundlich_retardation(
            concentration=np.array([0.0, 1.0]),
            freundlich_k=0.5,
            freundlich_n=0.7,
            bulk_density=1500.0,
            porosity=0.3,
        )

    # Negative concentration with n < 1: must raise
    with pytest.raises(ValueError, match="concentration must be strictly positive when freundlich_n < 1"):
        freundlich_retardation(
            concentration=np.array([-0.1, 1.0]),
            freundlich_k=0.5,
            freundlich_n=0.7,
            bulk_density=1500.0,
            porosity=0.3,
        )


def test_freundlich_retardation_zero_concentration_n_geq_one_allowed():
    """For n >= 1 the retardation factor is finite at C = 0 (or constant for n = 1); must not raise."""
    # n = 1 -> retardation factor independent of C
    result = freundlich_retardation(
        concentration=np.array([0.0, 1.0, 2.0]),
        freundlich_k=0.5,
        freundlich_n=1.0,
        bulk_density=1500.0,
        porosity=0.3,
    )
    expected_constant = 1.0 + (1500.0 / 0.3) * 0.5 * 1.0
    np.testing.assert_allclose(result, expected_constant, rtol=1e-12)

    # n > 1 -> retardation factor equals 1 at C = 0
    result = freundlich_retardation(
        concentration=np.array([0.0, 1.0]),
        freundlich_k=0.5,
        freundlich_n=1.5,
        bulk_density=1500.0,
        porosity=0.3,
    )
    np.testing.assert_allclose(result[0], 1.0, rtol=1e-12)


def test_variable_flow_residence_time_analytical():
    """Test ``residence_time`` against an exact analytical residence time for variable flow.

    For piecewise-constant flow ``flow[i]`` over time bins of width ``dt``, sample the rate
    ``Q(t) = q0 + a * t`` at bin midpoints (``flow[i] = q0 + a * (i + 0.5) * dt``). Then the
    function's cumulative-flow curve, evaluated at ``index = flow_tedges`` (i.e. integer
    edge times), is exact for the piecewise-constant flow profile -- there is no bin-center
    interpolation error. The analytical residence time ``tau`` at extraction time
    ``t_index = N * dt`` is recovered exactly by accumulating bin volumes backward in time
    until ``V_p`` is exhausted, then taking the partial bin contribution. This matches the
    function's internal inversion to machine precision.

    Note: comparing against the *continuous* analytical solution
    ``V(t) - V(t - tau) = V_p`` with ``V(t) = q0 * t + a / 2 * t^2`` would only be accurate
    to ``O(a * dt^2 / Q)`` because the function's piecewise-linear V differs from the
    continuous quadratic V on the open intervals. Since the goal of this test is to validate
    the function rather than the discretization, we compare to the function's exact
    residence time given its piecewise-constant flow, computed via direct backward bin
    accumulation.
    """
    q0 = 100.0  # m³/day
    a = 2.0  # m³/day²
    n_days = 200
    pore_volume = 500.0  # m³
    dt = 1.0  # day

    flow_tedges = pd.date_range(start="2023-01-01", periods=n_days + 1, freq="D")
    t_days = np.arange(n_days, dtype=float)
    flow_values = q0 + a * (t_days + 0.5) * dt  # midpoint sampling of the linear ramp

    # Evaluate at flow_tedges (integer edge times) -- avoids the bin-center interpolation
    # path and keeps the test exact w.r.t. the function's piecewise-constant flow.
    result = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        index=flow_tedges,
        direction="extraction_to_infiltration",
    )

    # Closed-form continuous-Q residence time, used only as a sanity check for monotonicity
    # and ordering, not for the precision comparison.
    t_ext = np.arange(n_days + 1, dtype=float)
    disc = (q0 + a * t_ext) ** 2 - 2.0 * a * pore_volume
    tau_continuous = np.full_like(t_ext, np.nan)
    valid_disc = disc >= 0
    tau_continuous[valid_disc] = ((q0 + a * t_ext[valid_disc]) - np.sqrt(disc[valid_disc])) / a

    # Exact analytical residence time for the piecewise-constant flow profile and
    # integer-endpoint extraction times. Walk backward bin-by-bin until V_p is exhausted.
    cum_flow = np.concatenate(([0.0], np.cumsum(flow_values * dt)))  # V at edges 0, 1, ..., N

    tau_exact = np.full(n_days + 1, np.nan)
    for i in range(n_days + 1):
        target = cum_flow[i] - pore_volume
        if target < 0:
            continue  # not enough cumulative flow yet -> NaN
        # Find the bin index k such that cum_flow[k] <= target < cum_flow[k+1].
        k = int(np.searchsorted(cum_flow, target, side="right") - 1)
        # Partial position within bin k (in days from start of bin k).
        partial = (target - cum_flow[k]) / flow_values[k]
        s_back = k + partial  # days since simulation start
        tau_exact[i] = i - s_back  # both in days

    valid = ~np.isnan(result[0]) & ~np.isnan(tau_exact)
    assert np.any(valid)

    # Function output must equal the bin-by-bin exact solution to machine precision.
    np.testing.assert_allclose(result[0, valid], tau_exact[valid], rtol=1e-12)

    # Sanity check: the discrete solution agrees with the continuous one to within the
    # expected discretization error a * dt^2 / (8 * Q_min).
    q_min = float(np.min(flow_values))
    bias_bound = a * dt**2 / (8.0 * q_min)
    valid_cont = valid & ~np.isnan(tau_continuous)
    np.testing.assert_allclose(tau_exact[valid_cont], tau_continuous[valid_cont], atol=bias_bound, rtol=1e-3)


# ---------------------------------------------------------------------------
# Property-based invariants (issue #172): mass conservation, forward-then-reverse
# identity, monotonicity, NaN flow propagation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flow_values",
    [
        np.full(20, 100.0),  # constant
        np.array([50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0] * 2),  # monotone
        np.array([100.0, 10.0, 100.0, 200.0, 50.0, 100.0, 100.0, 300.0, 80.0, 100.0] * 2),  # kinky
    ],
)
@pytest.mark.parametrize("pore_volume", [50.0, 200.0, 500.0])
@pytest.mark.parametrize("retardation_factor", [1.0, 1.5, 3.0])
def test_mass_conservation_under_cumulative_flow(flow_values, pore_volume, retardation_factor):
    """The integral of Q over the residence window equals R * V_p exactly.

    By construction ``residence_time(direction='extraction_to_infiltration')`` inverts the
    equation ``R * V_p = integral_{t-tau}^{t} Q(s) ds``. The check below recomputes the
    cumulative flow independently and asserts that ``flow_cum(t) - flow_cum(t - tau) == R *
    V_p`` to machine precision at every informed index. Off-by-one in the ``flow_cum``
    concatenation (residence_time.py:114) breaks all R; applying R outside the integration
    breaks all R != 1 under variable Q.
    """
    flow_tedges = pd.date_range(start="2023-01-01", periods=len(flow_values) + 1, freq="D")

    tau = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )[0]
    valid = ~np.isnan(tau)
    assert np.any(valid)

    # Independent cumulative-flow integral; ``residence_time`` evaluates at bin centers
    # when ``index`` is None, so the query times are the bin centers.
    flow_tedges_days = np.asarray((flow_tedges - flow_tedges[0]) / np.timedelta64(1, "D"))
    bin_centers = (flow_tedges_days[:-1] + flow_tedges_days[1:]) / 2
    flow_cum_at_edges = np.concatenate(([0.0], np.cumsum(flow_values * np.diff(flow_tedges_days))))
    flow_cum_at_t = np.interp(bin_centers, flow_tedges_days, flow_cum_at_edges)
    flow_cum_at_t_minus_tau = np.interp(bin_centers - tau, flow_tedges_days, flow_cum_at_edges)

    np.testing.assert_allclose(
        flow_cum_at_t[valid] - flow_cum_at_t_minus_tau[valid],
        retardation_factor * pore_volume,
        atol=0,
        rtol=1e-13,
    )


@pytest.mark.parametrize("pore_volume", [50.0, 200.0, 500.0])
def test_forward_then_reverse_identity(pore_volume):
    """``residence_time(i2e)`` at t and ``residence_time(e2i)`` at t + tau_fwd recover tau_fwd.

    The two directions invert the same equation ``V_p = integral_{t_inf}^{t_ext} Q ds`` from
    opposite ends, so they must agree to bit precision when wired in series. This is a
    different proof than the constant-flow symmetry test -- it works under variable flow with
    interior kinks.
    """
    flow_tedges = pd.date_range(start="2023-01-01", periods=21, freq="D")
    flow_values = np.array([100.0, 50.0, 200.0, 80.0, 150.0, 60.0, 120.0, 90.0, 110.0, 70.0] * 2)

    # Forward tau at flow_tedges (avoid bin-center timing for clean evaluation).
    tau_fwd = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        index=flow_tedges,
        direction="infiltration_to_extraction",
    )[0]
    valid = ~np.isnan(tau_fwd)
    assert np.any(valid)

    # Reverse evaluation at the extraction times ``t + tau_fwd``.
    extraction_times = flow_tedges + pd.to_timedelta(np.where(valid, tau_fwd, 0.0), unit="D")
    tau_rev = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        index=extraction_times[valid],
        direction="extraction_to_infiltration",
    )[0]

    np.testing.assert_allclose(tau_rev, tau_fwd[valid], atol=0, rtol=1e-13)


def test_monotonicity_under_increasing_flow():
    """Strictly increasing Q yields strictly decreasing residence time at fixed V_p.

    Under increasing Q the look-back parcel takes less time to accumulate V_p, so tau(t) must
    be strictly decreasing across informed indices. The legacy ``test_multiple_pore_volumes``
    asserted monotonicity in V_p only; this asserts the dual monotonicity in Q.
    """
    flow_tedges = pd.date_range(start="2023-01-01", periods=51, freq="D")
    # Linear ramp Q(t) sampled at bin midpoints; strictly increasing.
    t_mid = (np.arange(50) + 0.5) * 1.0
    flow_values = 50.0 + 5.0 * t_mid
    pore_volume = 500.0

    tau = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )[0]
    valid = ~np.isnan(tau)
    assert np.sum(valid) > 5
    assert np.all(np.diff(tau[valid]) < 0)


@pytest.mark.parametrize("nan_index", [0, 1, -1])
def test_nan_flow_returns_nan(nan_index):
    """A single NaN in flow makes the result all-NaN with the right shape.

    ``np.any(flow < 0)`` evaluates ``False`` for NaN under IEEE 754; the gate at
    residence_time.py:105 must therefore also catch NaN explicitly, otherwise NaN propagates
    through ``cumsum`` into ``flow_cum`` and ``residence_time_mean`` (which calls
    ``linear_average``, requiring strictly-ascending ``x_data``) fails noisily. This test pins
    the symmetric NaN-return behavior across both functions.

    The NaN position is parametrized: an early NaN (index 0 or 1) poisons everything
    downstream of ``cumsum`` and would pass even without the gate, but a late NaN (index -1)
    leaves the leading entries finite, exposing a partial-NaN result if the gate is removed.
    """
    flow_tedges = pd.date_range(start="2023-01-01", end="2023-01-06", freq="D")
    flow_values = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
    flow_values[nan_index] = np.nan
    pore_volume = 100.0

    rt = residence_time(
        flow=flow_values,
        flow_tedges=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    assert rt.shape == (1, len(flow_tedges) - 1)
    assert np.all(np.isnan(rt))

    rt_mean = residence_time_mean(
        flow=flow_values,
        flow_tedges=flow_tedges,
        tedges_out=flow_tedges,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    assert rt_mean.shape == (1, len(flow_tedges) - 1)
    assert np.all(np.isnan(rt_mean))


@pytest.mark.parametrize("tz", ["UTC", "America/New_York"])
def test_timezone_invariance(tz):
    """A tz-aware ``flow_tedges`` (no DST in-window) matches the tz-naive result.

    Both functions reduce time edges to days relative to ``flow_tedges[0]``, so the result
    is origin-invariant: any constant shift -- including a tz strip or UTC-normalize --
    cancels by construction. This guard therefore does NOT lock origin-preservation. It
    locks the narrower, still-useful contract that a tz-aware index with the same
    wall-clock spacing as its naive counterpart (no DST transition in the window) yields
    bit-identical days, which is the realistic path for the package's UTC-aware
    ``get_soil_temperature``/``generate_example_data`` indices. (Across a DST boundary the
    localized day-spacing is not constant, so the equality holds only for DST-free windows
    such as the January one used here.)
    """
    flow_values = np.array([100.0, 110.0, 105.0, 95.0, 98.0, 102.0, 107.0, 103.0, 96.0])
    naive = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
    aware = naive.tz_localize(tz)
    pore_volume = 200.0

    rt_naive = residence_time(
        flow=flow_values,
        flow_tedges=naive,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    rt_aware = residence_time(
        flow=flow_values,
        flow_tedges=aware,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    np.testing.assert_array_equal(rt_aware, rt_naive)

    rt_mean_naive = residence_time_mean(
        flow=flow_values,
        flow_tedges=naive,
        tedges_out=naive,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    rt_mean_aware = residence_time_mean(
        flow=flow_values,
        flow_tedges=aware,
        tedges_out=aware,
        aquifer_pore_volumes=pore_volume,
        direction="extraction_to_infiltration",
    )
    np.testing.assert_array_equal(rt_mean_aware, rt_mean_naive)
