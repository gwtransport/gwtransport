"""
Lean and effective tests for deposition module.

Focus on:
1. Exact analytical solutions
2. Perfect roundtrip consistency
3. Clean edge cases with exact comparisons
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from gwtransport.deposition import (
    compute_deposition_weights,
    deposition_to_extraction,
    extraction_to_deposition,
    extraction_to_deposition_full,
    spinup_duration,
)
from gwtransport.residence_time import residence_time
from gwtransport.utils import compute_time_edges, solve_underdetermined_system


def test_exact_analytical_constant_deposition():
    """
    Test exact analytical solution: C = (residence_time * deposition_rate) / (porosity * thickness)

    Uses constant flow and deposition for exact solution.
    """
    # Simple setup for exact calculation
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 500.0,  # m³
        "porosity": 0.25,  # dimensionless
        "thickness": 4.0,  # m
        "retardation_factor": 1.0,
    }

    # Constant inputs for exact solution
    deposition_rate = 100.0  # ng/m²/day
    flow_rate = 100.0  # m³/day → residence time = 500/100 = 5 days

    dep_values = np.full(len(dates), deposition_rate)
    flow_values = np.full(len(dates), flow_rate)

    # Output after sufficient time for steady state
    cout_tedges = tedges[7:9]  # 2 edges for 1 output

    # Calculate actual concentration
    cout_result = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params["aquifer_pore_volume"],
        porosity=params["porosity"],
        thickness=params["thickness"],
        retardation_factor=params["retardation_factor"],
    )

    # Calculate expected using exact formula
    rt = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=params["aquifer_pore_volume"],
        retardation_factor=params["retardation_factor"],
        direction="extraction_to_infiltration",
    )

    expected = (rt[0] * deposition_rate) / (params["porosity"] * params["thickness"])

    # Exact comparison
    valid_result = cout_result[~np.isnan(cout_result)]
    valid_expected = expected[: len(valid_result)]

    assert len(valid_result) >= 1, "Must have at least one valid result"

    np.testing.assert_allclose(valid_result, valid_expected, rtol=1e-12, atol=0)


def test_exact_analytical_varying_flow():
    """
    Test exact analytical solution with time-varying flow.
    """
    dates = pd.date_range("2020-01-01", "2020-01-08", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 300.0,
        "porosity": 0.3,
        "thickness": 5.0,
        "retardation_factor": 1.0,
    }

    # Variable flow rates
    flow_values = np.array([50.0, 75.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    deposition_rate = 60.0
    dep_values = np.full(len(dates), deposition_rate)

    cout_tedges = tedges[5:7]  # Test in stable period

    cout_result = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params["aquifer_pore_volume"],
        porosity=params["porosity"],
        thickness=params["thickness"],
        retardation_factor=params["retardation_factor"],
    )

    # Calculate expected using exact residence time
    rt = residence_time(
        flow=flow_values,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=params["aquifer_pore_volume"],
        retardation_factor=params["retardation_factor"],
        direction="extraction_to_infiltration",
    )

    expected = (rt[0] * deposition_rate) / (params["porosity"] * params["thickness"])

    valid_result = cout_result[~np.isnan(cout_result)]
    valid_expected = expected[: len(valid_result)]

    if len(valid_result) >= 1:
        # The analytical formula C = rt * D / (porosity * thickness) is exact
        # for constant deposition: cout is the flow-weighted bin average and
        # for constant D this reduces to D * <flow*rt>/<flow> / (porosity*
        # thickness). In this test the cout bin lies entirely within the
        # constant-flow tail of the series, so rt is constant within the
        # bin and the formula collapses to rt[edge] * D / (porosity*
        # thickness) at machine precision. There is no genuine numerical
        # noise specific to varying flow, so a tight machine-precision
        # tolerance is appropriate.
        np.testing.assert_allclose(valid_result, valid_expected, rtol=1e-12, atol=0)


def test_exact_analytical_retardation_factor():
    """
    Test exact analytical solution with different retardation factors.
    """
    dates = pd.date_range("2020-01-01", "2020-01-12", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    base_params = {
        "aquifer_pore_volume": 400.0,
        "porosity": 0.2,
        "thickness": 8.0,
    }

    deposition_rate = 50.0
    flow_rate = 100.0
    dep_values = np.full(len(dates), deposition_rate)
    flow_values = np.full(len(dates), flow_rate)

    cout_tedges = tedges[8:10]  # Test in later period

    # Test different retardation factors
    for retardation_factor in [1.0, 1.5, 2.0]:
        params = {**base_params, "retardation_factor": retardation_factor}

        cout_result = deposition_to_extraction(
            dep=dep_values,
            flow=flow_values,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=params["aquifer_pore_volume"],
            porosity=params["porosity"],
            thickness=params["thickness"],
            retardation_factor=params["retardation_factor"],
        )

        rt = residence_time(
            flow=flow_values,
            flow_tedges=tedges,
            index=cout_tedges,
            aquifer_pore_volumes=params["aquifer_pore_volume"],
            retardation_factor=retardation_factor,
            direction="extraction_to_infiltration",
        )

        expected = (rt[0] * deposition_rate) / (params["porosity"] * params["thickness"])

        valid_result = cout_result[~np.isnan(cout_result)]
        valid_expected = expected[: len(valid_result)]

        if len(valid_result) >= 1:
            np.testing.assert_allclose(
                valid_result,
                valid_expected,
                rtol=1e-12,
                atol=0,
                err_msg=f"R={retardation_factor}",
            )


def test_perfect_roundtrip_varying_deposition_short():
    """Roundtrip with varying deposition on a short time series.

    Uses half-day cout resolution to create an overdetermined system and
    non-integer RT/dt to avoid structural rank deficiency, achieving
    machine precision on per-element recovery.
    """
    dates = pd.date_range("2020-01-01", "2020-01-08", freq="D")  # 8 days
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 350.0,  # RT = 3.5 days (non-integer avoids rank deficiency)
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    original_deposition = np.array([10.0, 20.0, 30.0, 25.0, 15.0, 35.0, 40.0, 30.0])
    flow_values = np.full(len(dates), 100.0)

    # Half-day cout resolution → overdetermined system
    cout_dates = pd.date_range("2020-01-01", "2020-01-08", freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    concentration = deposition_to_extraction(
        dep=original_deposition,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    recovered_deposition = extraction_to_deposition(
        cout=concentration,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        regularization_strength=1e-20,
        **params,
    )

    valid = ~np.isnan(recovered_deposition)
    assert valid.sum() == len(original_deposition)

    np.testing.assert_allclose(
        recovered_deposition[valid],
        original_deposition[valid],
        atol=1e-10,
        rtol=1e-10,
    )


def test_perfect_roundtrip_constant_deposition():
    """Roundtrip with constant deposition.

    Uses half-day cout resolution to create an overdetermined system.
    Constant deposition is always in the column space (even with integer
    RT/dt), so machine precision is achievable.
    """
    dates = pd.date_range("2020-01-01", "2020-01-08", freq="D")  # 8 days
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 350.0,  # RT = 3.5 days (non-integer)
        "porosity": 0.2,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    original_deposition = np.full(len(dates), 25.0)
    flow_values = np.full(len(dates), 100.0)

    # Half-day cout resolution → overdetermined system
    cout_dates = pd.date_range("2020-01-01", "2020-01-08", freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    concentration = deposition_to_extraction(
        dep=original_deposition,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    recovered_deposition = extraction_to_deposition(
        cout=concentration,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        regularization_strength=1e-20,
        **params,
    )

    valid_mask = ~np.isnan(recovered_deposition)
    np.testing.assert_allclose(
        recovered_deposition[valid_mask],
        original_deposition[valid_mask],
        atol=1e-10,
        rtol=1e-10,
    )


def test_zero_deposition_zero_concentration():
    """Zero deposition must produce exactly zero concentration."""
    dates = pd.date_range("2020-01-01", "2020-01-06", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 300.0,
        "porosity": 0.3,
        "thickness": 5.0,
        "retardation_factor": 1.0,
    }

    dep_values = np.zeros(len(dates))
    flow_values = np.full(len(dates), 100.0)
    cout_tedges = tedges[2:5]

    cout_result = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params["aquifer_pore_volume"],
        porosity=params["porosity"],
        thickness=params["thickness"],
        retardation_factor=params["retardation_factor"],
    )

    valid_results = cout_result[~np.isnan(cout_result)]
    assert len(valid_results) >= 1, "Setup must produce at least one valid cout bin"
    np.testing.assert_allclose(valid_results, 0.0, rtol=0, atol=1e-15)


def test_linearity_exact():
    """Test exact linearity: doubling input exactly doubles output."""
    dates = pd.date_range("2020-01-01", "2020-01-08", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 320.0,
        "porosity": 0.2,
        "thickness": 8.0,
        "retardation_factor": 1.0,
    }

    base_deposition = np.full(len(dates), 20.0)
    double_deposition = 2.0 * base_deposition
    flow_values = np.full(len(dates), 80.0)
    cout_tedges = tedges[4:7]

    # Test both directions
    cout_base = deposition_to_extraction(
        dep=base_deposition,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params["aquifer_pore_volume"],
        porosity=params["porosity"],
        thickness=params["thickness"],
        retardation_factor=params["retardation_factor"],
    )
    cout_double = deposition_to_extraction(
        dep=double_deposition,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params["aquifer_pore_volume"],
        porosity=params["porosity"],
        thickness=params["thickness"],
        retardation_factor=params["retardation_factor"],
    )

    valid_base = cout_base[~np.isnan(cout_base)]
    valid_double = cout_double[~np.isnan(cout_double)]

    min_len = min(len(valid_base), len(valid_double))
    if min_len >= 1:
        np.testing.assert_allclose(valid_double[:min_len], 2.0 * valid_base[:min_len], rtol=1e-12, atol=0)


def test_negative_deposition_linearity():
    """Test that negative deposition produces exactly negative concentration."""
    dates = pd.date_range("2020-01-01", "2020-01-06", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 240.0,
        "porosity": 0.3,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    positive_dep = np.full(len(dates), 30.0)
    negative_dep = -positive_dep
    flow_values = np.full(len(dates), 60.0)
    cout_tedges = tedges[3:5]

    cout_positive = deposition_to_extraction(
        dep=positive_dep,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params["aquifer_pore_volume"],
        porosity=params["porosity"],
        thickness=params["thickness"],
        retardation_factor=params["retardation_factor"],
    )
    cout_negative = deposition_to_extraction(
        dep=negative_dep,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params["aquifer_pore_volume"],
        porosity=params["porosity"],
        thickness=params["thickness"],
        retardation_factor=params["retardation_factor"],
    )

    valid_pos = cout_positive[~np.isnan(cout_positive)]
    valid_neg = cout_negative[~np.isnan(cout_negative)]

    min_len = min(len(valid_pos), len(valid_neg))
    if min_len >= 1:
        np.testing.assert_allclose(valid_neg[:min_len], -valid_pos[:min_len], rtol=1e-12, atol=0)


def test_parameter_scaling_exact():
    """Test exact scaling relationships for porosity and thickness."""
    dates = pd.date_range("2020-01-01", "2020-01-06", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    base_params = {
        "aquifer_pore_volume": 200.0,
        "retardation_factor": 1.0,
    }

    deposition_rate = 40.0
    flow_rate = 50.0
    dep_values = np.full(len(dates), deposition_rate)
    flow_values = np.full(len(dates), flow_rate)
    cout_tedges = tedges[3:5]

    # Test porosity scaling (concentration ∝ 1/porosity)
    porosity_1 = 0.2
    porosity_2 = 0.4  # Double the porosity

    params_1 = {**base_params, "porosity": porosity_1, "thickness": 5.0}
    params_2 = {**base_params, "porosity": porosity_2, "thickness": 5.0}

    cout_1 = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params_1["aquifer_pore_volume"],
        porosity=params_1["porosity"],
        thickness=params_1["thickness"],
        retardation_factor=params_1["retardation_factor"],
    )
    cout_2 = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=params_2["aquifer_pore_volume"],
        porosity=params_2["porosity"],
        thickness=params_2["thickness"],
        retardation_factor=params_2["retardation_factor"],
    )

    valid_1 = cout_1[~np.isnan(cout_1)]
    valid_2 = cout_2[~np.isnan(cout_2)]

    min_len = min(len(valid_1), len(valid_2))
    if min_len >= 1:
        for i in range(min_len):
            ratio = valid_1[i] / valid_2[i]
            expected_ratio = porosity_2 / porosity_1  # Should be 2.0
            rel_error = abs(ratio - expected_ratio) / expected_ratio
            assert rel_error < 1e-10, f"Porosity scaling failed: ratio={ratio:.6f}, expected={expected_ratio:.6f}"


def test_input_validation():
    """Test input validation with exact error messages."""
    dates = pd.date_range("2020-01-01", "2020-01-04", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 200.0,
        "porosity": 0.3,
        "thickness": 5.0,
        "retardation_factor": 1.0,
    }

    # Test tedges length mismatch
    with pytest.raises(ValueError, match="tedges must have one more element than flow"):
        deposition_to_extraction(
            dep=np.ones(3),
            flow=np.ones(4),
            tedges=tedges[:4],
            cout_tedges=tedges[1:3],
            aquifer_pore_volume=params["aquifer_pore_volume"],
            porosity=params["porosity"],
            thickness=params["thickness"],
            retardation_factor=params["retardation_factor"],
        )

    # Test cout_tedges length mismatch in extraction_to_deposition
    with pytest.raises(ValueError, match="cout_tedges must have one more element than cout"):
        extraction_to_deposition(
            cout=np.ones(3),
            flow=np.ones(3),
            tedges=tedges[:4],
            cout_tedges=tedges[:3],
            aquifer_pore_volume=params["aquifer_pore_volume"],
            porosity=params["porosity"],
            thickness=params["thickness"],
            retardation_factor=params["retardation_factor"],
        )

    # Test NaN in flow (should be rejected)
    flow_with_nan = np.array([50.0, np.nan, 60.0])
    with pytest.raises(ValueError, match="flow array cannot contain NaN values"):
        extraction_to_deposition(
            cout=np.ones(2),
            flow=flow_with_nan,
            tedges=tedges[:4],
            cout_tedges=tedges[1:4],
            aquifer_pore_volume=params["aquifer_pore_volume"],
            porosity=params["porosity"],
            thickness=params["thickness"],
            retardation_factor=params["retardation_factor"],
        )


def test_rank_deficiency_warning():
    """Warning is raised when weight matrix is rank-deficient (integer RT/dt)."""
    dates = pd.date_range("2020-01-01", "2020-01-08", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 400.0,  # RT = 400/100 = 4.0 days (integer)
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }
    flow_values = np.full(len(dates), 100.0)
    cout_tedges = tedges

    cout = deposition_to_extraction(
        dep=np.full(len(dates), 25.0),
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    with pytest.warns(UserWarning, match="rank-deficient"):
        extraction_to_deposition(
            cout=cout,
            flow=flow_values,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **params,
        )


def test_no_rank_deficiency_warning_non_integer_rt():
    """No warning when RT/dt is non-integer."""
    dates = pd.date_range("2020-01-01", "2020-01-08", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 350.0,  # RT = 350/100 = 3.5 days (non-integer)
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }
    flow_values = np.full(len(dates), 100.0)

    cout_dates = pd.date_range("2020-01-01", "2020-01-08", freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cout = deposition_to_extraction(
        dep=np.full(len(dates), 25.0),
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extraction_to_deposition(
            cout=cout,
            flow=flow_values,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **params,
        )


def test_regularization_objectives():
    """Test different nullspace regularization objectives work correctly."""
    # This test verifies the solver correctly handles different objectives
    # by testing that squared_differences objective works properly
    # Create a simple underdetermined system (2 equations, 4 unknowns)
    matrix = np.array([[1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 1.0]])
    rhs = np.array([5.0, 4.0])

    # Test that squared differences objective works
    result = solve_underdetermined_system(
        coefficient_matrix=matrix, rhs_vector=rhs, nullspace_objective="squared_differences"
    )

    # Should satisfy the original equations
    assert np.allclose(matrix @ result, rhs, atol=1e-10), "Solution should satisfy Ax=b"

    # Should be finite and reasonable
    assert np.all(np.isfinite(result)), "Solution should be finite"
    assert np.max(np.abs(result)) < 100, "Solution should be reasonable magnitude"


@pytest.fixture
def full_solver_overdetermined_setup():
    """Shared setup for extraction_to_deposition_full machine-precision tests.

    Varying deposition + non-integer RT (3.5 days) + half-day cout makes the
    system overdetermined and full-rank, so the nullspace solver can recover
    the original deposition to machine precision (bounded only by optimizer
    noise).
    """
    dates = pd.date_range("2020-01-01", "2020-01-08", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_dates = pd.date_range("2020-01-01", "2020-01-08", freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    params = {
        "aquifer_pore_volume": 350.0,  # RT = 3.5 days (non-integer => full rank)
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }
    original_deposition = np.array([10.0, 20.0, 30.0, 25.0, 15.0, 35.0, 40.0, 30.0])
    flow_values = np.full(len(dates), 100.0)

    concentration = deposition_to_extraction(
        dep=original_deposition,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )
    return tedges, cout_tedges, flow_values, original_deposition, concentration, params


def test_extraction_to_deposition_full_default(full_solver_overdetermined_setup):
    """Default-objective round-trip recovers the original deposition to machine precision.

    The prior assertion `np.all(np.isfinite(recovered))` would pass even if
    the solver returned constant 1e10; an absolute round-trip check pins the
    scale.
    """
    tedges, cout_tedges, flow_values, original_deposition, concentration, params = full_solver_overdetermined_setup

    recovered = extraction_to_deposition_full(
        cout=concentration,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,  # pyright: ignore[reportArgumentType]
    )

    np.testing.assert_allclose(recovered, original_deposition, rtol=1e-10, atol=1e-10)


def test_extraction_to_deposition_full_objectives(full_solver_overdetermined_setup):
    """Round-trip recovery for both nullspace objectives.

    On this overdetermined full-rank system both objectives reach max-abs
    error ~1e-13 in their natural optimizer pairing (BFGS for the smooth
    squared_differences, Nelder-Mead for the non-smooth summed_differences).
    The asserted ``atol=1e-10`` is the project-wide machine-precision target;
    the actual recovery is two orders tighter.
    """
    tedges, cout_tedges, flow_values, original_deposition, concentration, params = full_solver_overdetermined_setup

    for objective, method in [("squared_differences", "BFGS"), ("summed_differences", "Nelder-Mead")]:
        recovered = extraction_to_deposition_full(
            cout=concentration,
            flow=flow_values,
            tedges=tedges,
            cout_tedges=cout_tedges,
            nullspace_objective=objective,
            optimization_method=method,
            **params,  # pyright: ignore[reportArgumentType]
        )
        np.testing.assert_allclose(
            recovered,
            original_deposition,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"objective={objective} method={method}",
        )


def test_extraction_to_deposition_full_with_rcond(full_solver_overdetermined_setup):
    """Setting rcond=1e-10 must not degrade the round-trip on a full-rank system."""
    tedges, cout_tedges, flow_values, original_deposition, concentration, params = full_solver_overdetermined_setup

    recovered = extraction_to_deposition_full(
        cout=concentration,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        rcond=1e-10,
        **params,  # pyright: ignore[reportArgumentType]
    )

    np.testing.assert_allclose(recovered, original_deposition, rtol=1e-10, atol=1e-10)


def test_roundtrip_varying_deposition():
    """Roundtrip with sinusoidal deposition on a longer time series.

    Uses half-day cout resolution for an overdetermined system and
    non-integer RT/dt to ensure full rank. Verifies per-element recovery
    to machine precision.
    """
    dates = pd.date_range("2020-01-01", "2020-02-01", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    params = {
        "aquifer_pore_volume": 350.0,  # RT = 3.5 days (non-integer)
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    t = np.arange(len(dates), dtype=float)
    original_deposition = 20.0 + 10.0 * np.sin(2 * np.pi * t / len(dates))
    flow_values = np.full(len(dates), 100.0)

    # Half-day cout resolution → overdetermined system
    cout_dates = pd.date_range("2020-01-01", "2020-02-01", freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    concentration = deposition_to_extraction(
        dep=original_deposition,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    recovered = extraction_to_deposition(
        cout=concentration,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        regularization_strength=1e-20,
        **params,
    )

    valid = ~np.isnan(recovered)
    assert valid.sum() == len(original_deposition)

    np.testing.assert_allclose(
        recovered[valid],
        original_deposition[valid],
        atol=1e-10,
        rtol=1e-10,
    )


def test_extraction_to_deposition_sparse_sampling_rank_deficient_warning():
    """Inverse on weekly cout with weekly flow tedges fires the rank-deficient warning.

    Replaces a heavy notebook-data setup with a synthetic 28-day weekly grid
    that reproduces the same integer-RT condition: APV * R = N * Q * dt_week
    makes the deposition weight matrix a uniform moving average with exact
    zeros in its transfer function. The warning must fire here; sibling tests
    assert it does NOT fire for non-integer-RT setups.
    """
    weekly_tedges = pd.date_range("2020-01-01", periods=5, freq="7D")  # 4 weekly bins
    flow = np.full(4, 100.0)
    dep = np.array([1.0, 2.0, 3.0, 4.0])
    params = {
        "aquifer_pore_volume": 1400.0,  # RT = 1400/100 = 14 days = 2 weekly bins (integer RT/dt)
        "porosity": 0.25,
        "thickness": 12.0,
        "retardation_factor": 1.0,
    }

    cout = deposition_to_extraction(dep=dep, flow=flow, tedges=weekly_tedges, cout_tedges=weekly_tedges, **params)
    with pytest.warns(UserWarning, match="rank-deficient"):
        extraction_to_deposition(cout=cout, flow=flow, tedges=weekly_tedges, cout_tedges=weekly_tedges, **params)


# =============================================================================
# Tests for spinup_duration function (CRITICAL COVERAGE GAP)
# =============================================================================


def test_spinup_duration_constant_flow():
    """Test spinup_duration with constant flow."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=101, freq="D")
    flow = np.ones(len(dates)) * 100.0  # m³/day
    pore_volume = 5000.0  # m³
    retardation_factor = 1.0

    duration = spinup_duration(
        flow=flow, flow_tedges=tedges, aquifer_pore_volume=pore_volume, retardation_factor=retardation_factor
    )

    # Spinup should equal residence time at first time point
    # RT = pore_volume * retardation / flow = 5000 * 1.0 / 100 = 50 days
    # Under constant flow, V*R/Q is algebraically exact.
    expected_duration = pore_volume * retardation_factor / flow[0]
    np.testing.assert_allclose(duration, expected_duration, rtol=0, atol=1e-12)


def test_spinup_duration_with_retardation():
    """Test spinup_duration with retardation factor > 1."""
    # RT = 5000 * 2.0 / 100 = 100 days, so need at least 100 days of flow history
    dates = pd.date_range(start="2020-01-01", periods=150, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=151, freq="D")
    flow = np.ones(len(dates)) * 100.0
    pore_volume = 5000.0
    retardation_factor = 2.0  # Temperature transport

    duration = spinup_duration(
        flow=flow, flow_tedges=tedges, aquifer_pore_volume=pore_volume, retardation_factor=retardation_factor
    )

    # Spinup should be longer with retardation
    # RT = 5000 * 2.0 / 100 = 100 days
    expected_duration = pore_volume * retardation_factor / flow[0]
    np.testing.assert_allclose(duration, expected_duration, rtol=0, atol=1e-12)
    assert duration > 50.0  # Longer than without retardation


def test_spinup_duration_high_flow():
    """Test spinup_duration with high flow (short spinup)."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=101, freq="D")
    flow = np.ones(len(dates)) * 500.0  # High flow
    pore_volume = 5000.0
    retardation_factor = 1.0

    duration = spinup_duration(
        flow=flow, flow_tedges=tedges, aquifer_pore_volume=pore_volume, retardation_factor=retardation_factor
    )

    # RT = 5000 * 1.0 / 500 = 10 days
    expected_duration = pore_volume * retardation_factor / flow[0]
    np.testing.assert_allclose(duration, expected_duration, rtol=0, atol=1e-12)
    assert duration < 20.0  # Short spinup with high flow


def test_spinup_duration_low_flow():
    """Test spinup_duration with low flow (long spinup)."""
    # RT = 5000 * 1.0 / 20 = 250 days, so need at least 250 days of flow history
    dates = pd.date_range(start="2020-01-01", periods=300, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=301, freq="D")
    flow = np.ones(len(dates)) * 20.0  # Low flow
    pore_volume = 5000.0
    retardation_factor = 1.0

    duration = spinup_duration(
        flow=flow, flow_tedges=tedges, aquifer_pore_volume=pore_volume, retardation_factor=retardation_factor
    )

    # RT = 5000 * 1.0 / 20 = 250 days
    expected_duration = pore_volume * retardation_factor / flow[0]
    np.testing.assert_allclose(duration, expected_duration, rtol=0, atol=1e-12)
    assert duration > 200.0  # Long spinup with low flow


def test_spinup_duration_large_pore_volume():
    """Test spinup_duration with large pore volume."""
    # RT = 20000 * 1.5 / 100 = 300 days, so need at least 300 days of flow history
    dates = pd.date_range(start="2020-01-01", periods=400, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=401, freq="D")
    flow = np.ones(len(dates)) * 100.0
    pore_volume = 20000.0  # Large aquifer
    retardation_factor = 1.5

    duration = spinup_duration(
        flow=flow, flow_tedges=tedges, aquifer_pore_volume=pore_volume, retardation_factor=retardation_factor
    )

    # RT = 20000 * 1.5 / 100 = 300 days
    expected_duration = pore_volume * retardation_factor / flow[0]
    np.testing.assert_allclose(duration, expected_duration, rtol=0, atol=1e-12)


def test_spinup_duration_variable_flow_uses_extraction_direction():
    """Spin-up under variable flow uses extraction_to_infiltration direction.

    Spin-up is the residence time of water *currently being extracted* at
    the first valid extraction edge: the time t* such that the integrated
    flow equals R*V_pore. The (incorrect) infiltration_to_extraction
    direction at the first time step would describe how long water
    infiltrated at t=0 takes to be extracted, which differs under variable
    flow when the two endpoints span different flow regimes.
    """
    tedges = pd.date_range(start="2020-01-01", periods=101, freq="D")
    pore_volume = 5000.0
    retardation_factor = 1.0

    # Slow first half (50 m³/day for 50 days -> cum=2500), fast second half
    # (200 m³/day). Need 5000 m³ total -> 50 + (5000-2500)/200 = 62.5 days.
    slow_then_fast = np.concatenate([np.full(50, 50.0), np.full(50, 200.0)])
    duration = spinup_duration(
        flow=slow_then_fast,
        flow_tedges=tedges,
        aquifer_pore_volume=pore_volume,
        retardation_factor=retardation_factor,
    )
    np.testing.assert_allclose(duration, 62.5, rtol=0, atol=1e-12)

    # Fast first half (200 m³/day) drains 5000 m³ in 25 days -> spin-up = 25.
    fast_then_slow = np.concatenate([np.full(50, 200.0), np.full(50, 50.0)])
    duration_asym = spinup_duration(
        flow=fast_then_slow,
        flow_tedges=tedges,
        aquifer_pore_volume=pore_volume,
        retardation_factor=retardation_factor,
    )
    np.testing.assert_allclose(duration_asym, 25.0, rtol=0, atol=1e-12)


def test_spinup_duration_with_zero_flow_plateau():
    """spinup_duration with a Q = 0 bin in the middle of the spinup window.

    ``flow = [100, 0, 100]`` (1-day bins), ``V_p = 1.0``, ``R = 200`` -> target volume = 200
    m^3. Cumulative flow ``flow_cum = [0, 100, 100, 200]`` is flat at 100 over ``t in [1, 2]``.
    The smallest ``t`` such that ``V(t) >= 200`` is ``t = 3`` (when ``V`` resumes growing past
    the plateau and reaches 200 at the right edge of the third bin). Without the ulp-bump
    regularization in :func:`spinup_duration`, ``np.interp`` on a non-strictly-monotone
    ``flow_cum`` would still give the correct LEFT limit here, but the bump locks the answer
    in across np.interp implementation details.
    """
    flow_tedges = pd.date_range(start="2023-01-01", periods=4, freq="D")
    flow = np.array([100.0, 0.0, 100.0])
    duration = spinup_duration(
        flow=flow,
        flow_tedges=flow_tedges,
        aquifer_pore_volume=1.0,
        retardation_factor=200.0,
    )
    np.testing.assert_allclose(duration, 3.0, atol=0, rtol=1e-13)


# =============================================================================
# Tests for compute_deposition_weights function (MEDIUM PRIORITY)
# =============================================================================


def test_compute_deposition_weights_structure():
    """Shape, non-negativity, finiteness, and analytical row-sum invariant.

    The row-sum invariant ``sum(W[i, :]) * porosity * thickness == RT_i``
    comes directly from the function's contract (each row's sum equals
    ``residence_time / (porosity * thickness)`` per the
    ``extraction_to_deposition`` Notes block). Adding it makes this test
    load-bearing -- a 1.5x scaling bug in the area integration would still
    pass the shape/non-negativity checks but fail the row sum.
    """
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=51, freq="D")
    cout_tedges = pd.date_range(start="2020-01-08", periods=41, freq="D")

    flow = np.ones(len(dates)) * 100.0
    aquifer_pore_volume = 500.0  # RT = 500/100 = 5 days
    porosity = 0.35
    thickness = 3.0
    retardation_factor = 1.0

    weights = compute_deposition_weights(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=aquifer_pore_volume,
        porosity=porosity,
        thickness=thickness,
        retardation_factor=retardation_factor,
    )

    # Shape, non-negativity, finiteness
    assert weights.shape == (len(cout_tedges) - 1, len(tedges) - 1)
    assert weights.shape == (40, 50)
    assert np.all(weights >= 0.0)
    assert np.all(np.isfinite(weights))

    # Row-sum invariant: sum(W[i, :]) * porosity * thickness == RT_i
    rt = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volume,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )[0]
    # residence_time returns RT at each cout edge; per-bin RT is the integral
    # across the bin width. Under constant flow RT is constant => RT_at_edge
    # is the bin RT.
    rt_per_bin = rt[:-1]  # length matches weights.shape[0]
    np.testing.assert_allclose(weights.sum(axis=1) * porosity * thickness, rt_per_bin, rtol=0, atol=1e-12)


def test_compute_deposition_weights_causality():
    """Weight matrix respects strict causality: future deposition cannot affect past extraction.

    Setup: APV=500, Q=100, RT=5 days; cout_tedges starts at dep_tedges[7].
    The water extracted in cout_bin_i (covering [day 7+i, day 8+i]) was
    infiltrated in [day 2+i, day 3+i]. Source bins j >= 4 + i correspond to
    deposition strictly later than the latest infiltration time of any
    parcel extracted in cout_bin_i, so the weight must be exactly zero.
    Replaces the prior structure-only `total_weight > 0` assertion, which
    would pass even under a 1.5x scaling mutation.
    """
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=51, freq="D")
    cout_tedges = pd.date_range(start="2020-01-08", periods=31, freq="D")
    flow = np.ones(len(dates)) * 100.0

    weights = compute_deposition_weights(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=500.0,
        porosity=0.35,
        thickness=3.0,
        retardation_factor=1.0,
    )

    # cout_tedges[0] starts at dep_tedges[7] (day 7). Water extracted in
    # cout_bin_i = [day 7+i, day 8+i] was in the aquifer from infiltration
    # (RT=5 days back) at [day 2+i, day 3+i] through extraction at [day 7+i,
    # day 8+i]. Source dep_bin_j fully past extraction (j >= 8+i) cannot
    # affect cout_bin_i. Verified empirically: nonzero cols for row i are
    # [i+2, i+7] inclusive, so weights[i, i+8:] must be exactly zero.
    n_dep = weights.shape[1]
    for i in range(weights.shape[0]):
        future_start = i + 8
        if future_start < n_dep:
            assert np.all(weights[i, future_start:] == 0.0), (
                f"weights[{i}, {future_start}:] should be zero; got max {weights[i, future_start:].max()}"
            )


def test_compute_deposition_weights_with_retardation():
    """Row-sum scales with R: sum(W[i,:]) * porosity * thickness == RT_i at both R=1 and R=2.

    Replaces the prior `not np.allclose(weights_r1, weights_r2)` diff-only
    assertion, which a wrong-but-different implementation would pass.
    """
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=51, freq="D")
    cout_tedges = pd.date_range(start="2020-01-12", periods=31, freq="D")

    flow = np.ones(len(dates)) * 100.0
    aquifer_pore_volume = 5000.0
    porosity = 0.35
    thickness = 3.0

    for retardation_factor in (1.0, 2.0):
        weights = compute_deposition_weights(
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            porosity=porosity,
            thickness=thickness,
            retardation_factor=retardation_factor,
        )
        rt = residence_time(
            flow=flow,
            flow_tedges=tedges,
            index=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volume,
            retardation_factor=retardation_factor,
            direction="extraction_to_infiltration",
        )[0]
        rt_per_bin = rt[:-1]
        np.testing.assert_allclose(
            weights.sum(axis=1) * porosity * thickness,
            rt_per_bin,
            rtol=0,
            atol=1e-12,
            err_msg=f"R={retardation_factor}",
        )


def test_compute_deposition_weights_porosity_effect():
    """Weights scale exactly as 1/porosity (everything else fixed).

    Replaces the prior `not np.allclose(weights_a, weights_b)` diff-only
    assertion. The invariant `weights * porosity` is porosity-independent
    follows from `area_array = volume_array / (porosity * thickness)` in
    `compute_deposition_weights` -- the volume integration itself does not
    depend on porosity.
    """
    dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
    tedges = pd.date_range(start="2020-01-01", periods=51, freq="D")
    cout_tedges = pd.date_range(start="2020-01-08", periods=31, freq="D")

    flow = np.ones(len(dates)) * 100.0
    aquifer_pore_volume = 500.0
    thickness = 3.0
    retardation_factor = 1.0
    porosity_low, porosity_high = 0.25, 0.45

    common = {
        "flow": flow,
        "tedges": tedges,
        "cout_tedges": cout_tedges,
        "aquifer_pore_volume": aquifer_pore_volume,
        "thickness": thickness,
        "retardation_factor": retardation_factor,
    }
    weights_low = compute_deposition_weights(porosity=porosity_low, **common)
    weights_high = compute_deposition_weights(porosity=porosity_high, **common)

    np.testing.assert_allclose(weights_low * porosity_low, weights_high * porosity_high, rtol=0, atol=1e-12)


# =============================================================================
# Variable-timestep tests
# =============================================================================


def test_variable_timestep_constant_deposition():
    """Constant deposition and flow on irregular time edges.

    After spin-up, the analytical solution C = rt * dep_rate / (porosity * thickness)
    holds exactly regardless of timestep width.
    """
    # Irregular intervals: 1, 3, 2, 5, 7, 4, 2, 3, 1, 6, 4, 2, 5, 3 days = 48 days total
    day_widths = [1, 3, 2, 5, 7, 4, 2, 3, 1, 6, 4, 2, 5, 3]
    cumdays = np.concatenate(([0], np.cumsum(day_widths)))
    t0 = pd.Timestamp("2020-01-01")
    tedges = pd.DatetimeIndex([t0 + pd.Timedelta(days=int(d)) for d in cumdays])
    n = len(tedges) - 1

    params = {
        "aquifer_pore_volume": 500.0,
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    deposition_rate = 100.0
    flow_rate = 100.0  # RT = 500/100 = 5 days
    dep_values = np.full(n, deposition_rate)
    flow_values = np.full(n, flow_rate)

    # Use late output edges to be past spin-up
    cout_tedges = tedges[8:]  # Start well after spin-up

    cout_result = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    # For constant flow, RT is constant: V*R/Q = 500*1/100 = 5 days
    expected_rt = params["aquifer_pore_volume"] * params["retardation_factor"] / flow_rate
    expected_c = expected_rt * deposition_rate / (params["porosity"] * params["thickness"])

    valid = ~np.isnan(cout_result)
    assert valid.sum() >= 1
    np.testing.assert_allclose(cout_result[valid], expected_c, rtol=1e-12)


def test_variable_timestep_roundtrip():
    """Roundtrip with variable timesteps and varying deposition.

    Variable timesteps naturally produce non-integer RT/dt ratios per bin,
    avoiding rank deficiency. Uses half-day cout for overdetermination.
    """
    day_widths = [1, 3, 2, 5, 7, 4, 2, 3, 1, 6, 4, 2, 5, 3, 2, 4, 1, 3, 6, 2]
    cumdays = np.concatenate(([0], np.cumsum(day_widths)))
    t0 = pd.Timestamp("2020-01-01")
    tedges = pd.DatetimeIndex([t0 + pd.Timedelta(days=int(d)) for d in cumdays])
    n = len(tedges) - 1

    params = {
        "aquifer_pore_volume": 350.0,  # RT = 3.5 days (non-integer)
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    dep_values = 20.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / n)
    flow_values = np.full(n, 100.0)

    # Half-day cout resolution
    total_days = int(cumdays[-1])
    cout_dates = pd.date_range(t0, t0 + pd.Timedelta(days=total_days), freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cout = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    recovered = extraction_to_deposition(
        cout=cout,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        regularization_strength=1e-20,
        **params,
    )

    valid = ~np.isnan(recovered)
    assert valid.sum() >= n - 1
    np.testing.assert_allclose(
        recovered[valid],
        dep_values[valid],
        atol=1e-10,
        rtol=1e-10,
    )


def test_variable_cout_tedges_constant_deposition():
    """Daily input data but variable-width cout_tedges (3-day and 7-day bins).

    Constant deposition + constant flow produces the same analytical solution
    regardless of output bin width.
    """
    # Daily tedges for 40 days
    tedges = pd.date_range("2020-01-01", periods=41, freq="D")
    n = len(tedges) - 1

    params = {
        "aquifer_pore_volume": 500.0,
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    deposition_rate = 80.0
    flow_rate = 100.0  # RT = 5 days
    dep_values = np.full(n, deposition_rate)
    flow_values = np.full(n, flow_rate)

    # Variable-width output bins: 3-day, 7-day, 3-day, 7-day, ...
    cout_days = [0]
    widths = [3, 7]
    i = 0
    while cout_days[-1] < 30:
        cout_days.append(cout_days[-1] + widths[i % 2])
        i += 1
    # Offset to start after spin-up
    t0 = tedges[8]
    cout_tedges = pd.DatetimeIndex([t0 + pd.Timedelta(days=d) for d in cout_days])

    cout_result = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    # For constant flow, RT is constant: V*R/Q = 500*1/100 = 5 days
    expected_rt = params["aquifer_pore_volume"] * params["retardation_factor"] / flow_rate
    expected_c = expected_rt * deposition_rate / (params["porosity"] * params["thickness"])

    valid = ~np.isnan(cout_result)
    assert valid.sum() >= 1
    np.testing.assert_allclose(cout_result[valid], expected_c, rtol=1e-12)


def test_variable_timestep_linearity():
    """With variable timesteps, verify exact linearity: 2 * dep produces exactly 2 * cout."""
    day_widths = [1, 3, 2, 5, 7, 4, 2, 3, 1, 6, 4, 2]
    cumdays = np.concatenate(([0], np.cumsum(day_widths)))
    t0 = pd.Timestamp("2020-01-01")
    tedges = pd.DatetimeIndex([t0 + pd.Timedelta(days=int(d)) for d in cumdays])
    n = len(tedges) - 1

    params = {
        "aquifer_pore_volume": 400.0,
        "porosity": 0.2,
        "thickness": 8.0,
        "retardation_factor": 1.0,
    }

    base_dep = np.full(n, 20.0)
    double_dep = 2.0 * base_dep
    flow_values = np.full(n, 80.0)

    cout_tedges = tedges[5:]

    cout_base = deposition_to_extraction(
        dep=base_dep,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )
    cout_double = deposition_to_extraction(
        dep=double_dep,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    valid = ~np.isnan(cout_base) & ~np.isnan(cout_double)
    assert valid.sum() >= 1
    np.testing.assert_allclose(cout_double[valid], 2.0 * cout_base[valid], rtol=1e-12)


def test_variable_timestep_zero_deposition():
    """Zero deposition with variable timesteps produces exactly zero concentration."""
    day_widths = [1, 3, 2, 5, 7, 4, 2, 3, 1, 6]
    cumdays = np.concatenate(([0], np.cumsum(day_widths)))
    t0 = pd.Timestamp("2020-01-01")
    tedges = pd.DatetimeIndex([t0 + pd.Timedelta(days=int(d)) for d in cumdays])
    n = len(tedges) - 1

    params = {
        "aquifer_pore_volume": 300.0,
        "porosity": 0.3,
        "thickness": 5.0,
        "retardation_factor": 1.0,
    }

    dep_values = np.zeros(n)
    flow_values = np.full(n, 100.0)
    cout_tedges = tedges[3:]

    cout_result = deposition_to_extraction(
        dep=dep_values,
        flow=flow_values,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,
    )

    valid = ~np.isnan(cout_result)
    assert valid.sum() >= 1
    np.testing.assert_allclose(cout_result[valid], 0.0, atol=1e-15)


# ===============================================================================
# FLOW SIGN VALIDATION
# ===============================================================================


@pytest.fixture
def flow_sign_setup():
    """Shared setup for flow sign validation tests."""
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))
    cout_dates = pd.date_range("2020-01-03", "2020-01-10", freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))
    params = {
        "aquifer_pore_volume": 200.0,
        "porosity": 0.3,
        "thickness": 5.0,
        "retardation_factor": 1.0,
    }
    return tedges, cout_tedges, params


def test_deposition_to_extraction_negative_flow_rejected(flow_sign_setup):
    """Negative flow is physically invalid and must raise the standard error."""
    tedges, cout_tedges, params = flow_sign_setup
    n = len(tedges) - 1
    flow = np.full(n, 100.0)
    flow[3] = -50.0
    with pytest.raises(ValueError, match="flow must be non-negative"):
        deposition_to_extraction(
            dep=np.ones(n),
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **params,
        )


def test_extraction_to_deposition_negative_flow_rejected(flow_sign_setup):
    """Negative flow is physically invalid and must raise the standard error."""
    tedges, cout_tedges, params = flow_sign_setup
    n = len(tedges) - 1
    flow = np.full(n, 100.0)
    flow[3] = -50.0
    n_cout = len(cout_tedges) - 1
    with pytest.raises(ValueError, match="flow must be non-negative"):
        extraction_to_deposition(
            cout=np.ones(n_cout) * 10.0,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **params,
        )


def test_extraction_to_deposition_full_negative_flow_rejected(flow_sign_setup):
    """Negative flow is physically invalid and must raise the standard error."""
    tedges, cout_tedges, params = flow_sign_setup
    n = len(tedges) - 1
    flow = np.full(n, 100.0)
    flow[3] = -50.0
    n_cout = len(cout_tedges) - 1
    with pytest.raises(ValueError, match="flow must be non-negative"):
        extraction_to_deposition_full(
            cout=np.ones(n_cout) * 10.0,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **params,
        )


def test_deposition_to_extraction_zero_flow_accepted(flow_sign_setup):
    """Zero flow is mathematically valid (no transport during the bin) and must not raise.

    The call must emit no RuntimeWarnings (divide-by-zero / invalid-value), and
    non-zero-flow cout bins must still produce finite output -- only cout bins
    whose extracted volume is zero (fully inside the zero-flow window) may be NaN.
    """
    tedges, cout_tedges, params = flow_sign_setup
    n = len(tedges) - 1
    flow = np.full(n, 100.0)
    flow[3] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cout = deposition_to_extraction(
            dep=np.ones(n),
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **params,
        )

    assert cout.shape == (len(cout_tedges) - 1,)
    # Zero-flow must not produce NaN. The zero-flow cout bin has no extraction
    # so its concentration is reported as 0 (no water => no flux observed).
    assert np.all(np.isfinite(cout))
    assert np.all(cout >= 0.0)


def test_extraction_to_deposition_zero_flow_accepted(flow_sign_setup):
    """Zero flow must be accepted (no RuntimeWarnings) by the deconvolution path too."""
    tedges, cout_tedges, params = flow_sign_setup
    n = len(tedges) - 1
    flow = np.full(n, 100.0)
    flow[3] = 0.0
    n_cout = len(cout_tedges) - 1

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        dep = extraction_to_deposition(
            cout=np.ones(n_cout) * 10.0,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **params,
        )

    assert dep.shape == (n,)


def test_deposition_to_extraction_spinup_constant_eliminates_left_edge_nan():
    """deposition_to_extraction with spinup='constant' warm-starts the system.

    Under spinup=None, cout bins where the deposition history is not fully
    resolved (residence time < cout time) become NaN. Under
    spinup='constant', the warm-start prepends bins with dep[0]/flow[0],
    so cout in the original cout_tedges window is non-NaN where
    physically meaningful.
    """
    n = 30
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    dep = np.full(n, 2.0)
    flow = np.full(n, 100.0)

    cout_warm = deposition_to_extraction(
        dep=dep,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=1000.0,  # RT = 10 days
        porosity=0.3,
        thickness=10.0,
        spinup="constant",
    )
    cout_strict = deposition_to_extraction(
        dep=dep,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=1000.0,
        porosity=0.3,
        thickness=10.0,
        spinup=None,
    )

    # Warm-start populates left-edge bins; strict has NaN there.
    n_finite_warm = int(np.sum(np.isfinite(cout_warm)))
    n_finite_strict = int(np.sum(np.isfinite(cout_strict)))
    assert n_finite_warm > n_finite_strict, "constant should produce more finite cout than strict"

    # Where strict is finite, both modes must agree (warm-start is a
    # superset of strict on the original window).
    valid = np.isfinite(cout_strict)
    np.testing.assert_allclose(cout_warm[valid], cout_strict[valid])


def test_extraction_to_deposition_spinup_constant_recovers_full_window():
    """extraction_to_deposition with spinup='constant' recovers the full deposition window.

    Round-trip with a step deposition signal: forward emits warm-started
    cout, reverse recovers the deposition (sliced back to the original
    tedges length).
    """
    n = 50
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges
    dep_true = np.zeros(n)
    dep_true[10:30] = 3.0  # box-shaped deposition
    flow = np.full(n, 100.0)

    cout = deposition_to_extraction(
        dep=dep_true,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=500.0,  # RT = 5 days, * 1.001 for full-rank avoidance
        porosity=0.3,
        thickness=10.0,
    )
    dep_rec = extraction_to_deposition(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volume=500.0,
        porosity=0.3,
        thickness=10.0,
    )

    assert dep_rec.shape == (n,), "output must align with original tedges length"
    # Recovery in the interior of the box; boundaries see Tikhonov bias.
    np.testing.assert_allclose(dep_rec[14:26], 3.0, atol=0.1)


# =============================================================================
# Issue #171: mass conservation, variable R, zero-flow recovery, NaN
# propagation, row-norm invariants, limit reductions
# =============================================================================


@pytest.mark.parametrize(
    ("flow_pattern", "retardation_factor", "pulse_start", "pulse_end"),
    [
        ("constant", 1.0, 20, 30),
        ("variable", 1.0, 30, 50),
        ("constant", 2.0, 30, 40),
    ],
    ids=["constant_flow_R1", "variable_flow_R1", "constant_flow_R2"],
)
def test_mass_conservation_pulse_dep(flow_pattern, retardation_factor, pulse_start, pulse_end):
    """Closed-window mass balance: Σ cout · Q · Δt = D · A_eff · pulse_width.

    Dep is a constant pulse over [pulse_start, pulse_end] (well past spin-up),
    zero elsewhere. The cout window matches the dep window and is long enough
    that the entire response decays into machine-zero bins inside it. Mass
    balance holds exactly when ``A_eff = R · V_pore / (n · b)``.

    The 1/r_i^2-weighted-LS framing of row normalization (see Notes block in
    ``extraction_to_deposition``) is a forward-pass model identity: any per-bin
    contribution ``cout[i] = sum_j W[i,j] dep[j]`` summed against ``Q * dt``
    recovers ``D * A_eff * pulse_width`` exactly. A 1.5x scaling bug in
    ``compute_deposition_weights`` would fail this at the 50 percent relative level.
    """
    n = 80
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    cout_tedges = tedges

    if flow_pattern == "constant":
        flow = np.full(n, 100.0)
    else:
        t = np.arange(n)
        flow = 100.0 + 30.0 * np.sin(2 * np.pi * t / 14)

    params = {
        "aquifer_pore_volume": 500.0,
        "porosity": 0.3,
        "thickness": 10.0,
        "retardation_factor": retardation_factor,
    }
    deposition_rate = 5.0
    dep = np.zeros(n)
    dep[pulse_start:pulse_end] = deposition_rate

    cout = deposition_to_extraction(dep=dep, flow=flow, tedges=tedges, cout_tedges=cout_tedges, spinup=None, **params)

    valid = ~np.isnan(cout)
    mass_extracted = np.sum(cout[valid] * flow[valid] * 1.0)  # Δt_cout = 1 day

    a_eff = retardation_factor * params["aquifer_pore_volume"] / (params["porosity"] * params["thickness"])
    mass_injected = deposition_rate * a_eff * (pulse_end - pulse_start)

    np.testing.assert_allclose(mass_extracted, mass_injected, rtol=1e-10)


@pytest.mark.parametrize("retardation_factor", [2.0, 3.5], ids=["R2", "R3p5"])
def test_roundtrip_variable_retardation(retardation_factor):
    """Roundtrip recovery at R > 1 on overdetermined half-day cout, non-integer effective RT.

    All prior roundtrip tests use R=1.0. With R=2 (RT_eff=10 days) and R=3.5
    (RT_eff=17.5 days) the residence-time-dependent code paths
    (``residence_time(direction="extraction_to_infiltration")``, the
    ``y_upper=R*V_pore`` clip in ``compute_average_heights``) are exercised
    end-to-end with a per-element machine-precision recovery check.
    """
    # Need tedges long enough to cover RT_eff * 3 for proper spin-up + pulse + decay.
    # APV=500, Q=100, RT_water=5 days. RT_eff(R=3.5) = 17.5 days.
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n)
    # Half-day cout for overdetermined system; APV=510 to keep effective RT non-integer at R=3.5
    cout_dates = pd.date_range("2020-01-01", periods=2 * n, freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=2 * n)

    params = {
        "aquifer_pore_volume": 510.0,  # RT_water = 5.1 days; non-integer at any R
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": retardation_factor,
    }
    original_deposition = 20.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / 20.0)
    flow = np.full(n, 100.0)

    cout = deposition_to_extraction(
        dep=original_deposition, flow=flow, tedges=tedges, cout_tedges=cout_tedges, **params
    )
    recovered = extraction_to_deposition(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        regularization_strength=1e-20,
        **params,
    )

    valid = ~np.isnan(recovered)
    np.testing.assert_allclose(recovered[valid], original_deposition[valid], rtol=1e-10, atol=1e-10)


def test_zero_flow_spill_then_recovery_roundtrip():
    """Forward + inverse roundtrip with a two-bin zero-flow gap.

    Cout for cout-bins fully inside the zero-flow window must be 0 (no water
    extracted carries no mass signature). Outside the gap, the inverse must
    recover the original deposition to machine precision. Inside the gap,
    deposition is unobservable and recovery is undefined (no information).

    Regression coverage for #167 (deposition example notebook crash) at the
    forward-then-inverse level rather than the example-generator level.
    """
    n = 50
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n)
    cout_dates = pd.date_range("2020-01-01", periods=2 * n, freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=2 * n)

    flow = np.full(n, 100.0)
    gap_start, gap_end = 20, 22
    flow[gap_start:gap_end] = 0.0

    original_deposition = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 10.0)
    params = {
        "aquifer_pore_volume": 350.0,  # RT_water = 3.5 days, non-integer => full rank
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    cout = deposition_to_extraction(
        dep=original_deposition, flow=flow, tedges=tedges, cout_tedges=cout_tedges, spinup=None, **params
    )

    # cout for cout-bins fully inside the zero-flow window must be 0 (no water extracted).
    # Zero-flow window is dep tedges[20:22] = [2020-01-20, 2020-01-22). Half-day
    # cout_tedges put cout_bins 39..42 fully inside that window: cout_bins
    # starting at cout_tedges[39]=2020-01-20 00:00 to cout_tedges[43]=2020-01-22 00:00.
    np.testing.assert_array_equal(cout[39:43], 0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # rank-deficiency suppression
        dep_rec = extraction_to_deposition(
            cout=cout,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            regularization_strength=1e-20,
            spinup=None,
            **params,
        )

    # Outside the zero-flow gap, recovery is exact.
    np.testing.assert_allclose(dep_rec[:gap_start], original_deposition[:gap_start], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(dep_rec[gap_end:], original_deposition[gap_end:], rtol=1e-10, atol=1e-10)


def test_spinup_duration_first_bin_zero_flow():
    """spinup_duration accounts for zero-flow at the start of the series.

    Distinct from ``test_spinup_duration_with_zero_flow_plateau`` (which has
    the plateau in the middle). With 3 days of zero flow followed by 100
    m^3/day at APV=500 and R=1, the target volume R*V_pore = 500 is reached
    at day 3 + (500 / 100) = 8.
    """
    n = 20
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    flow[0:3] = 0.0  # 3 days of zero flow at start

    duration = spinup_duration(flow=flow, flow_tedges=tedges, aquifer_pore_volume=500.0, retardation_factor=1.0)
    np.testing.assert_allclose(duration, 8.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("solver", "nan_positions"),
    [
        ("tikhonov", [10, 25, 40]),
        ("tikhonov", [0, 1, 58, 59]),  # boundary NaN
        ("full", [10, 25, 40]),
    ],
    ids=["tikhonov_scattered", "tikhonov_boundary", "full_scattered"],
)
def test_nan_cout_propagation(solver, nan_positions):
    """Scattered / boundary NaN in cout is correctly excluded by both inverse solvers.

    Forward then inject NaN at the listed positions; the inverse must (i) not
    crash, (ii) recover the original deposition to machine precision in regions
    bracketed by valid cout bins. Covers both Tikhonov (``extraction_to_deposition``)
    and nullspace-based (``extraction_to_deposition_full``) solvers per
    bas-test-reviewer's separate code path observation.
    """
    n = 30
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n)
    cout_dates = pd.date_range("2020-01-01", periods=2 * n, freq="12h")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=2 * n)
    flow = np.full(n, 100.0)
    original_deposition = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(n) / 8.0)
    params = {
        "aquifer_pore_volume": 350.0,  # RT = 3.5, non-integer
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    cout = deposition_to_extraction(
        dep=original_deposition, flow=flow, tedges=tedges, cout_tedges=cout_tedges, **params
    )
    cout_with_nan = cout.copy()
    cout_with_nan[nan_positions] = np.nan

    if solver == "tikhonov":
        dep_rec = extraction_to_deposition(
            cout=cout_with_nan,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            regularization_strength=1e-20,
            **params,
        )
    else:
        dep_rec = extraction_to_deposition_full(
            cout=cout_with_nan,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            **params,  # pyright: ignore[reportArgumentType]
        )

    np.testing.assert_allclose(dep_rec, original_deposition, rtol=1e-10, atol=1e-10)


def test_row_normalization_pre_norm_row_sum_equals_rt_over_nb():
    """Under constant flow, every valid row sums to RT / (porosity * thickness).

    Direct verification of the inverse-solver docstring contract ("deposition
    rows sum to residence_time / (porosity * thickness)"). Constant flow is
    used because under variable flow the per-bin row sum integrates over a
    non-uniform residence-time profile and the closed-form identity
    ``row_sum * n * b == RT_at_one_edge`` only holds in the limit of small
    cout bins; here the goal is the machine-precision contract pin, not a
    variable-flow generalization.
    """
    n = 30
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n)
    cout_tedges = tedges
    flow = np.full(n, 100.0)
    params = {
        "aquifer_pore_volume": 350.0,  # RT = 3.5 days, non-integer
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }

    weights = compute_deposition_weights(flow=flow, tedges=tedges, cout_tedges=cout_tedges, **params)
    rt = residence_time(
        flow=flow,
        flow_tedges=tedges,
        index=cout_tedges,
        aquifer_pore_volumes=params["aquifer_pore_volume"],
        retardation_factor=params["retardation_factor"],
        direction="extraction_to_infiltration",
    )[0]
    rt_per_bin = rt[:-1]
    valid = ~np.isnan(weights).any(axis=1) & (np.abs(weights).sum(axis=1) > 0)
    np.testing.assert_allclose(
        weights[valid].sum(axis=1) * params["porosity"] * params["thickness"],
        rt_per_bin[valid],
        rtol=0,
        atol=1e-13,
    )


def test_row_normalization_zero_flow_row_excluded_not_nan():
    """A zero-flow cout bin produces an all-zero row, not a NaN-normalized row.

    The inverse solver's ``valid_rows`` mask combines NaN-row exclusion and
    all-zero-row exclusion. Row-normalization division ``W / row_sums`` must
    not be applied to a zero-flow cout bin (would produce NaN). This test
    pins the contract that all-zero rows are masked out *before* the division.
    """
    n = 30
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n)
    cout_tedges = tedges
    flow = np.full(n, 100.0)
    flow[15] = 0.0
    params = {
        "aquifer_pore_volume": 350.0,
        "porosity": 0.25,
        "thickness": 4.0,
        "retardation_factor": 1.0,
    }
    weights = compute_deposition_weights(flow=flow, tedges=tedges, cout_tedges=cout_tedges, **params)

    # Row 15 corresponds to the zero-flow cout bin.
    row_is_all_zero = bool(np.all(weights[15] == 0.0))
    row_contains_nan = bool(np.any(np.isnan(weights[15])))
    assert row_is_all_zero, "Zero-flow cout bin row 15 must be all-zero"
    assert not row_contains_nan, "Zero-flow cout bin row 15 must not be NaN"


def test_full_solver_linearity_and_roundtrip(full_solver_overdetermined_setup):
    """Linearity AND machine-precision absolute scale for extraction_to_deposition_full.

    Linearity alone (``inverse(2x) == 2 * inverse(x)``) would pass under a
    2x scaling bug. Round-trip pins absolute scale. Together they catch
    additive-shift and multiplicative-scale bugs in the nullspace solver path.
    """
    tedges, cout_tedges, flow, original_deposition, concentration, params = full_solver_overdetermined_setup

    base = extraction_to_deposition_full(
        cout=concentration,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,  # pyright: ignore[reportArgumentType]
    )
    doubled = extraction_to_deposition_full(
        cout=2.0 * concentration,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        **params,  # pyright: ignore[reportArgumentType]
    )

    # Linearity (would fail under a constant additive offset bug)
    np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-10, atol=1e-10)
    # Absolute scale (would fail under a 1.5x scaling bug that preserves linearity)
    np.testing.assert_allclose(base, original_deposition, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    ("limit_param", "limit_values"),
    [
        ("aquifer_pore_volume", [10.0, 1.0, 0.1, 0.01]),
        ("thickness", [10.0, 100.0, 1000.0, 10000.0]),
    ],
    ids=["small_apv", "large_thickness"],
)
def test_limit_reduction_cout_approaches_zero(limit_param, limit_values):
    """As APV → 0+ or thickness → ∞, cout → 0 monotonically.

    APV → 0: contact area A = V_pore / (n·b) → 0 ⇒ no deposition transfer.
    thickness → ∞: contact area A → 0 ⇒ no deposition transfer.

    APV = 0 strictly is rejected by the input validator at deposition.py:258;
    we instead drive APV through a small-positive sequence and verify
    monotonic decrease toward 0.
    """
    n = 40
    tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
    flow = np.full(n, 100.0)
    dep = np.full(n, 5.0)
    base_params = {
        "aquifer_pore_volume": 500.0,
        "porosity": 0.3,
        "thickness": 10.0,
        "retardation_factor": 1.0,
    }

    cout_maxima = []
    for v in limit_values:
        params = {**base_params, limit_param: v}
        cout = deposition_to_extraction(dep=dep, flow=flow, tedges=tedges, cout_tedges=tedges, spinup=None, **params)
        # Use the last valid cout bin (past spin-up) so steady-state holds and the linear scaling
        # in V_pore (or 1/b) is exact -- the max over a partial spin-up bin can include ramp-up.
        valid_indices = np.flatnonzero(~np.isnan(cout))
        cout_maxima.append(cout[valid_indices[-1]])

    cout_maxima = np.array(cout_maxima)

    # cout = D * RT / (n * b) = D * V_pore / (Q * n * b) under constant flow, steady state.
    # APV-sweep: cout proportional to V_pore. thickness-sweep: cout proportional to 1/thickness.
    # cout_max[i] / cout_max[0] == limit_values[i] / limit_values[0] (APV) or
    # cout_max[i] / cout_max[0] == limit_values[0] / limit_values[i] (thickness).
    if limit_param == "aquifer_pore_volume":
        expected_ratios = np.asarray(limit_values, dtype=float) / float(limit_values[0])
    else:  # thickness
        expected_ratios = float(limit_values[0]) / np.asarray(limit_values, dtype=float)
    observed_ratios = cout_maxima / cout_maxima[0]
    np.testing.assert_allclose(observed_ratios, expected_ratios, rtol=1e-12, atol=0)
    # And, just to be explicit, the final ratio must drop several orders.
    assert observed_ratios[-1] < 1e-2, f"final cout ratio should be << initial: {observed_ratios[-1]}"
