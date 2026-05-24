"""
Tests for verify_physics function from fronttracking.validation.

These tests are based on Example 1 from notebook 09_Front_Tracking_Rarefaction_Waves.ipynb
which demonstrates a concentration pulse with favorable sorption (n>1).
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction_nonlinear_sorption
from gwtransport.fronttracking.math import FreundlichSorption
from gwtransport.fronttracking.validation import verify_physics
from gwtransport.fronttracking.waves import ShockWave


def _make_check7_structure(*, target_rel: float):
    """Build a synthetic structure whose mass-balance check (7) has a chosen relative error.

    Check 7 is tautological in its physical inputs: ``compute_total_outlet_mass`` returns
    ``m_in - C_T(c_inf)*V`` and validation re-adds ``C_T(c_inf)*V``, so the residual is
    algebraically zero for any real ``(cin, theta_edges, v_outlet, sorption)``. To exercise the
    gate's *comparison* we decouple the two ``C_T(c_inf)`` evaluations with a sequenced stub:
    validation calls ``sorption.total_concentration`` twice for c_inf -- first inside
    ``compute_total_outlet_mass`` (line ~178), then in the ``m_dom_asymptotic`` recompute
    (line ~189). Returning the correct value first and an inflated value second yields a
    residual ``|C_T_inflated - C_T_correct| * V / m_in`` equal to ``target_rel``. This is the
    inconsistency check 7 exists to catch (e.g. a stale c_inf or a mismatched sorption object).

    Parameters
    ----------
    target_rel : float
        Desired relative mass-balance error for check 7.

    Returns
    -------
    tuple
        ``(structure, cout, cout_tedges, cin)`` ready to pass to ``verify_physics``.
    """
    real = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    theta_edges = np.array([0.0, 100.0, 200.0, 300.0])
    cin = np.array([5.0, 5.0, 5.0])  # c_inf = 5 > 0 so m_dom_asymptotic > 0
    v_outlet = 200.0
    c_inf = float(cin[-1])
    ct_correct = float(real.total_concentration(c_inf))
    m_in = float(np.sum(cin * np.diff(theta_edges)))
    # residual = |C_T_inflated - C_T_correct| * V; rel = residual / m_in
    ct_inflated = ct_correct + target_rel * m_in / v_outlet

    sorption_stub = MagicMock()
    sorption_stub.total_concentration = MagicMock(side_effect=[ct_correct, ct_inflated])

    tracker_state = MagicMock()
    tracker_state.theta_edges = theta_edges
    tracker_state.v_outlet = v_outlet
    tracker_state.sorption = sorption_stub
    # Spin-up mask: every output edge maps after theta_first_arrival (no NaN to flag).
    tracker_state.theta_at_t = MagicMock(return_value=0.0)

    structure = {
        "waves": [],
        "theta_first_arrival": 50.0,
        "events": [],
        "tracker_state": tracker_state,
    }
    cout = np.array([5.0, 5.0, 5.0])
    cout_tedges = pd.date_range("2020-01-01", periods=4, freq="D")
    return structure, cout, cout_tedges, cin


class TestVerifyPhysicsPassingChecks:
    """Tests for verify_physics with valid physics that should pass all checks."""

    @pytest.fixture(scope="module")
    def pulse_simulation_data(self):
        """
        Run a concentration pulse simulation (Example 1 from notebook 9).

        This setup creates a concentration pulse (0 → 10 → 0) with favorable
        sorption (n=2.0) that should pass all physics checks.

        Returns
        -------
        tuple
            (cin, cout, cout_tedges, structure) where structure is from
            infiltration_to_extraction_nonlinear_sorption.
        """
        # Setup from Example 1 in notebook 9
        tedges = pd.date_range(start="2020-01-01", periods=100, freq="D")

        # Pulse: 0 → 10 → 0 (shock on rise, rarefaction on fall)
        cin = np.zeros(len(tedges) - 1)
        cin[10:40] = 10.0  # Pulse from day 10 to 40

        # Aquifer properties
        flow = np.full(len(tedges) - 1, 100.0)  # m³/day
        aquifer_pore_volume = 200.0  # m³

        # Freundlich sorption (n > 1)
        freundlich_k = 0.01  # (m³/kg)^(1/n)
        freundlich_n = 2.0  # n>1
        bulk_density = 1500.0  # kg/m³
        porosity = 0.3

        # Output grid
        cout_tedges = pd.date_range(start=tedges[0], periods=1350, freq="D")

        # Run simulation
        cout, structure = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=[aquifer_pore_volume],
            freundlich_k=freundlich_k,
            freundlich_n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        return cin, cout, cout_tedges, structure[0]

    def test_verify_physics_all_checks_pass(self, pulse_simulation_data):
        """Test that verify_physics passes all checks on valid simulation."""
        cin, cout, cout_tedges, structure = pulse_simulation_data

        # Run physics verification
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)

        # All checks should pass
        assert results["all_passed"], f"Expected all checks to pass. Failures: {results['failures']}"
        assert results["n_passed"] == results["n_checks"]
        assert len(results["failures"]) == 0
        assert "passed" in results["summary"]

    def test_verify_physics_check_count(self, pulse_simulation_data):
        """Guard the check count so a new check cannot be added without a matching violation test.

        Not a physics test — a tripwire. Check 7 (rarefaction ordering) was removed
        because ``RarefactionWave.__post_init__`` already rejects ``head_speed <= tail_speed``,
        leaving the verify_physics gate unreachable; the count dropped from 8 to 7.
        """
        cin, cout, cout_tedges, structure = pulse_simulation_data

        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)

        # 7 checks after the dead rarefaction-ordering check was removed.
        assert results["n_checks"] == 7
        assert len(results["checks"]) == 7

    def test_verify_physics_check_names(self, pulse_simulation_data):
        """Guard the exact set of check names (tripwire for adding a check without a violation test)."""
        cin, cout, cout_tedges, structure = pulse_simulation_data

        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)

        # "Rarefaction wave ordering" removed: the constructor guard makes the gate unreachable.
        expected_checks = {
            "Shock entropy condition",
            "Non-negative concentrations",
            "Output <= input maximum",
            "Finite first arrival θ",
            "No NaN after spin-up",
            "Events θ-ordered",
            "Total integrated outlet mass",
        }

        actual_checks = {check["name"] for check in results["checks"]}
        assert actual_checks == expected_checks

    def test_verify_physics_verbose_mode(self, pulse_simulation_data):
        """Test that verbose mode runs without errors."""
        cin, cout, cout_tedges, structure = pulse_simulation_data

        # Run with verbose=True
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=True)

        # Should still pass all checks
        assert results["all_passed"]

    def test_verify_physics_returns_correct_structure(self, pulse_simulation_data):
        """Test that verify_physics returns the expected dictionary structure."""
        cin, cout, cout_tedges, structure = pulse_simulation_data

        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)

        # Check required keys
        required_keys = {"all_passed", "n_checks", "n_passed", "failures", "checks", "summary"}
        assert set(results.keys()) == required_keys

        # Check types (allow both int and numpy int types)
        assert isinstance(results["all_passed"], bool)
        assert isinstance(results["n_checks"], (int, np.integer))
        assert isinstance(results["n_passed"], (int, np.integer))
        assert isinstance(results["failures"], list)
        assert isinstance(results["checks"], list)
        assert isinstance(results["summary"], str)


class TestVerifyPhysicsFailingChecks:
    """Tests for verify_physics with intentionally violated physics."""

    @pytest.fixture(scope="module")
    def pulse_simulation_base(self):
        """
        Create base simulation for modification tests.

        Returns the same setup as pulse_simulation_data but returns all
        intermediate values for easy modification.
        """
        tedges = pd.date_range(start="2020-01-01", periods=100, freq="D")
        cin = np.zeros(len(tedges) - 1)
        cin[10:40] = 10.0
        flow = np.full(len(tedges) - 1, 100.0)
        aquifer_pore_volume = 200.0
        freundlich_k = 0.01
        freundlich_n = 2.0
        bulk_density = 1500.0
        porosity = 0.3
        cout_tedges = pd.date_range(start=tedges[0], periods=1350, freq="D")

        cout, structure = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=[aquifer_pore_volume],
            freundlich_k=freundlich_k,
            freundlich_n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )

        return cin, cout, cout_tedges, structure[0]

    def test_negative_concentration_violation(self, pulse_simulation_base):
        """Test that verify_physics detects negative concentrations."""
        cin, cout, cout_tedges, structure = pulse_simulation_base

        # Introduce negative concentration
        cout_modified = cout.copy()
        cout_modified[100] = -0.5  # Introduce significant negative value

        results = verify_physics(structure, cout_modified, cout_tedges, cin, verbose=False)

        # Should fail the non-negative concentration check
        assert not results["all_passed"]
        assert results["n_passed"] < results["n_checks"]
        assert len(results["failures"]) > 0
        assert any("Negative concentrations" in f for f in results["failures"])

    def test_output_exceeds_input_violation(self, pulse_simulation_base):
        """Test that verify_physics detects output exceeding input maximum."""
        cin, cout, cout_tedges, structure = pulse_simulation_base

        # Make output exceed input maximum
        cout_modified = cout.copy()
        max_cin = np.max(cin)
        cout_modified[100:200] = max_cin * 1.5  # Exceed input by 50%

        results = verify_physics(structure, cout_modified, cout_tedges, cin, verbose=False)

        # Should fail the output ≤ input check
        assert not results["all_passed"]
        assert len(results["failures"]) > 0
        assert any("exceeds input" in f for f in results["failures"])

    def test_nan_values_after_spinup(self, pulse_simulation_base):
        """Test that verify_physics detects NaN values after spin-up period."""
        cin, cout, cout_tedges, structure = pulse_simulation_base

        # Introduce NaN values after spin-up
        cout_modified = cout.copy()
        # Translate θ_first_arrival → t for time-based indexing
        t_first = structure["tracker_state"].t_at_theta(structure["theta_first_arrival"])

        # Find indices after spin-up
        cout_tedges_days = ((cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)).values
        mask_after_spinup = cout_tedges_days[:-1] >= t_first

        # Set some values to NaN after spin-up
        indices_after_spinup = np.where(mask_after_spinup)[0]
        if len(indices_after_spinup) > 10:
            cout_modified[indices_after_spinup[5:10]] = np.nan

        results = verify_physics(structure, cout_modified, cout_tedges, cin, verbose=False)

        # Should fail the NaN check
        assert not results["all_passed"]
        assert len(results["failures"]) > 0
        assert any("NaN" in f for f in results["failures"])

    def test_entropy_violation_detected(self, pulse_simulation_base):
        """Check 1 fails when a shock violates the Lax entropy condition.

        For Freundlich n=2 (favorable sorption) a shock with ``c_left < c_right``
        (here 2 < 10) is a rarefaction-direction discontinuity whose
        ``satisfies_entropy()`` returns False. Injecting such a shock into the
        wave list must trip only the entropy check.
        """
        cin, cout, cout_tedges, structure = pulse_simulation_base

        sorption = structure["tracker_state"].sorption
        bad_shock = ShockWave(theta_start=0.0, v_start=0.0, c_left=2.0, c_right=10.0, sorption=sorption)
        assert not bad_shock.satisfies_entropy(), "test setup invalid: shock unexpectedly satisfies entropy"

        structure_modified = dict(structure)
        structure_modified["waves"] = [*structure["waves"], bad_shock]

        results = verify_physics(structure_modified, cout, cout_tedges, cin, verbose=False)

        assert not results["all_passed"]
        assert any("Entropy violations" in f for f in results["failures"])
        check1 = next(c for c in results["checks"] if c["name"] == "Shock entropy condition")
        assert not check1["passed"]

    def test_infinite_first_arrival_detected(self, pulse_simulation_base):
        """Check 4 fails when ``theta_first_arrival`` is non-finite (here +inf)."""
        cin, cout, cout_tedges, structure = pulse_simulation_base

        structure_modified = dict(structure)
        structure_modified["theta_first_arrival"] = np.inf

        results = verify_physics(structure_modified, cout, cout_tedges, cin, verbose=False)

        assert not results["all_passed"]
        assert any("First arrival θ is not finite" in f for f in results["failures"])
        check4 = next(c for c in results["checks"] if c["name"] == "Finite first arrival θ")
        assert not check4["passed"]

    def test_event_ordering_violation_detected(self, pulse_simulation_base):
        """Check 6 fails when event ``"theta"`` keys are not monotone non-decreasing.

        Hand-built with three events (a length-2 list would hit the singleton/"N/A"
        message split but still report ordered). Post-#208 the event key is ``"theta"``.
        No solver run is needed.
        """
        cin, cout, cout_tedges, structure = pulse_simulation_base

        structure_modified = dict(structure)
        structure_modified["events"] = [
            {"theta": 100.0, "type": "outlet_crossing"},
            {"theta": 300.0, "type": "outlet_crossing"},
            {"theta": 200.0, "type": "outlet_crossing"},  # out of order
        ]

        results = verify_physics(structure_modified, cout, cout_tedges, cin, verbose=False)

        assert not results["all_passed"]
        assert any("Events are not θ-ordered" in f for f in results["failures"])
        check6 = next(c for c in results["checks"] if c["name"] == "Events θ-ordered")
        assert not check6["passed"]

    # NOTE: check 7 (mass balance) has no positive-violation test here. It is tautological in
    # its physical inputs (compute_total_outlet_mass = m_in - C_T(c_inf)*V, which validation
    # re-adds), so no real input can violate it; only a stubbed-sorption mock could, which
    # tests mock arithmetic rather than reachable behaviour. test_custom_rtol_gates_mass_balance_check
    # below pins the rtol wiring against a real total_concentration evaluation. Redesigning check 7
    # (and solver.py FrontTracker.verify_physics) to compare against an independent outlet-mass
    # integral is tracked as a follow-up.
    def test_mass_balance_skipped_without_tracker_state(self):
        """Check 7 takes the skip path (passes with the exact 'Skipped' message) when tracker_state is None.

        Pins the vacuous pass as deliberate so a refactor that always routes into the skip
        branch cannot silently hide a real mass-balance failure.
        """
        structure = {
            "waves": [],
            "theta_first_arrival": 50.0,
            "events": [],
            "tracker_state": None,
        }
        cin = np.array([5.0, 5.0, 5.0])
        cout = np.array([5.0, 5.0, 5.0])
        cout_tedges = pd.date_range("2020-01-01", periods=4, freq="D")

        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)

        check7 = next(c for c in results["checks"] if c["name"] == "Total integrated outlet mass")
        assert check7["passed"] is True
        assert check7["message"] == "Skipped (tracker state not available)"

    def test_multiple_violations(self, pulse_simulation_base):
        """Test verify_physics with multiple simultaneous violations."""
        cin, cout, cout_tedges, structure = pulse_simulation_base

        # Introduce multiple violations
        cout_modified = cout.copy()
        max_cin = np.max(cin)

        # Violation 1: Negative concentration
        cout_modified[50] = -1.0

        # Violation 2: Exceeds input
        cout_modified[100:150] = max_cin * 2.0

        # Violation 3: NaN after spin-up
        t_first = structure["tracker_state"].t_at_theta(structure["theta_first_arrival"])
        cout_tedges_days = ((cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)).values
        mask_after_spinup = cout_tedges_days[:-1] >= t_first
        indices_after_spinup = np.where(mask_after_spinup)[0]
        if len(indices_after_spinup) > 5:
            cout_modified[indices_after_spinup[:5]] = np.nan

        results = verify_physics(structure, cout_modified, cout_tedges, cin, verbose=False)

        # All three injected violations must co-occur (localization, not just an aggregate count).
        assert not results["all_passed"]
        failure_text = " | ".join(results["failures"])
        assert "Negative concentrations" in failure_text
        assert "exceeds input" in failure_text
        assert "NaN" in failure_text

        # Cross-talk guard: corrupting cout must NOT trip the four cout-independent checks.
        check_by_name = {c["name"]: c for c in results["checks"]}
        for unaffected in (
            "Shock entropy condition",
            "Finite first arrival θ",
            "Events θ-ordered",
            "Total integrated outlet mass",
        ):
            assert check_by_name[unaffected]["passed"], f"{unaffected} spuriously failed"

        # Exactly the three cout-based checks failed.
        assert results["n_passed"] == results["n_checks"] - 3

    def test_custom_rtol_gates_mass_balance_check(self):
        """rtol must gate the mass-balance check (7): a 5e-4 error fails at rtol<=1e-6 but passes at 1e-3.

        Check 7's threshold is ``max(rtol, 1e-6)``, so a residual sized at 5e-4 (between the
        1e-6 floor and 1e-3) straddles the two tolerances. Confirms rtol is wired into the gate
        rather than merely accepted.
        """
        target_rel = 5e-4

        # rtol below the 1e-6 floor -> threshold is 1e-6 -> 5e-4 residual fails check 7.
        structure, cout, cout_tedges, cin = _make_check7_structure(target_rel=target_rel)
        results_strict = verify_physics(structure, cout, cout_tedges, cin, verbose=False, rtol=1e-7)
        check7_strict = next(c for c in results_strict["checks"] if c["name"] == "Total integrated outlet mass")
        assert not check7_strict["passed"]
        assert any("Total outlet mass mismatch" in f for f in results_strict["failures"])

        # rtol above the residual -> threshold is 1e-3 -> 5e-4 residual passes check 7.
        structure, cout, cout_tedges, cin = _make_check7_structure(target_rel=target_rel)
        results_relaxed = verify_physics(structure, cout, cout_tedges, cin, verbose=False, rtol=1e-3)
        check7_relaxed = next(c for c in results_relaxed["checks"] if c["name"] == "Total integrated outlet mass")
        assert check7_relaxed["passed"]

        assert results_strict["n_checks"] == results_relaxed["n_checks"]


class TestVerifyPhysicsEdgeCases:
    """Test verify_physics with edge cases and special scenarios."""

    def test_nonfinite_first_arrival_on_zero_input(self):
        """All-zero cin drives ``theta_first_arrival`` non-finite: exactly check 4 fails.

        Documented outcome (verified empirically): nothing ever arrives at the outlet, so
        ``theta_first_arrival = +inf`` and the finite-first-arrival check (4) fails. The
        no-NaN check (5) takes the non-finite-θ masking branch (every output row counts as
        "after spin-up"); with an all-zero cout it still passes. Mass balance (7) is trivially
        satisfied (m_in = m_out = 0). Replaces the previous vacuous
        ``isinstance(all_passed, bool)`` assertion with a definite outcome.
        """
        tedges = pd.date_range(start="2020-01-01", periods=50, freq="D")
        cin = np.zeros(len(tedges) - 1)  # All zeros
        flow = np.full(len(tedges) - 1, 100.0)
        aquifer_pore_volume = 200.0
        cout_tedges = pd.date_range(start=tedges[0], periods=100, freq="D")

        cout, structure = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=[aquifer_pore_volume],
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        assert not np.isfinite(structure[0]["theta_first_arrival"]), "test setup invalid: expected non-finite θ_first"

        results = verify_physics(structure[0], cout, cout_tedges, cin, verbose=False)

        assert results["n_checks"] == 7
        # Exactly the finite-first-arrival check fails.
        assert not results["all_passed"]
        assert results["n_passed"] == 6
        check_by_name = {c["name"]: c for c in results["checks"]}
        assert not check_by_name["Finite first arrival θ"]["passed"]
        assert any("First arrival θ is not finite" in f for f in results["failures"])
        # Check 5 exercises the non-finite-θ masking branch and still passes (no NaN in cout).
        assert check_by_name["No NaN after spin-up"]["passed"]
        # Mass balance is trivially satisfied: m_in = m_out = 0.
        assert check_by_name["Total integrated outlet mass"]["passed"]

    def test_constant_concentration_input(self):
        """Test verify_physics with constant non-zero concentration ending in zero."""
        tedges = pd.date_range(start="2020-01-01", periods=50, freq="D")
        cin = np.full(len(tedges) - 1, 5.0)  # Constant 5.0
        cin[-1] = 0.0  # End with explicit zero for mass balance
        flow = np.full(len(tedges) - 1, 100.0)
        aquifer_pore_volume = 200.0
        cout_tedges = pd.date_range(start=tedges[0], periods=100, freq="D")

        cout, structure = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=[aquifer_pore_volume],
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        results = verify_physics(structure[0], cout, cout_tedges, cin, verbose=False)

        # Should pass all checks with constant input ending in zero
        assert results["all_passed"]
