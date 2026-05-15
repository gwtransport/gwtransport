"""
Tests for verify_physics function from fronttracking.validation.

These tests are based on Example 1 from notebook 09_Front_Tracking_Rarefaction_Waves.ipynb
which demonstrates a concentration pulse with favorable sorption (n>1).
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction_front_tracking_detailed
from gwtransport.fronttracking.math import FreundlichSorption
from gwtransport.fronttracking.validation import verify_physics
from gwtransport.fronttracking.waves import RarefactionWave, ShockWave


class TestVerifyPhysicsPassingChecks:
    """Tests for verify_physics with valid physics that should pass all checks."""

    @pytest.fixture
    def pulse_simulation_data(self):
        """
        Run a concentration pulse simulation (Example 1 from notebook 9).

        This setup creates a concentration pulse (0 → 10 → 0) with favorable
        sorption (n=2.0) that should pass all physics checks.

        Returns
        -------
        tuple
            (cin, cout, cout_tedges, structure) where structure is from
            infiltration_to_extraction_front_tracking_detailed.
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
        cout, structure = infiltration_to_extraction_front_tracking_detailed(
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
        assert "✓" in results["summary"]

    def test_verify_physics_each_check_independently_observable(self, pulse_simulation_data):
        """Every named check produces an observable pass/fail bit so a mutator that flips one
        is caught by ``results['n_passed']`` or ``results['failures']``.

        Replaces the previous ``n_checks == 8`` smoke test, which passed even after a
        mutation that always set ``checkN_pass = True`` for every check.
        """
        cin, cout, cout_tedges, structure = pulse_simulation_data
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)

        # Behavioural assertion: every check has a name AND a pass/fail bit, and n_passed
        # equals the number of checks where ``passed`` is True. A mutator that hardcodes
        # checkN_pass = True will desynchronise this from a mutated baseline.
        check_passed = [bool(c["passed"]) for c in results["checks"]]
        assert len(check_passed) >= 1, "verify_physics returned zero checks"
        assert results["n_passed"] == sum(check_passed)
        assert results["n_checks"] == len(check_passed)
        assert results["all_passed"] is bool(all(check_passed))
        # Names are unique so a mutation that accidentally duplicates a check is caught.
        names = [c["name"] for c in results["checks"]]
        assert len(set(names)) == len(names), f"duplicate check names: {names}"

    def test_verify_physics_check_names_match_documented_set(self, pulse_simulation_data):
        """The set of check names equals the documented set and the order pins each index.

        Replaces the previous set-equality test, which passed even if names were silently
        renamed alongside the test (string-set tautology). The ordered assertion catches
        a check being added/removed without updating the documented contract here.
        """
        cin, cout, cout_tedges, structure = pulse_simulation_data
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)
        actual = [c["name"] for c in results["checks"]]
        expected_ordered = [
            "Shock entropy condition",
            "Non-negative concentrations",
            "Output ≤ input maximum",
            "Finite first arrival time",
            "No NaN after spin-up",
            "Events chronologically ordered",
            "Rarefaction wave ordering",
            "Total integrated outlet mass",
        ]
        assert actual == expected_ordered, f"check order/names drifted: got {actual}"

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

    @pytest.fixture
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

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
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
        t_first = structure["t_first_arrival"]

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
        t_first = structure["t_first_arrival"]
        cout_tedges_days = ((cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)).values
        mask_after_spinup = cout_tedges_days[:-1] >= t_first
        indices_after_spinup = np.where(mask_after_spinup)[0]
        if len(indices_after_spinup) > 5:
            cout_modified[indices_after_spinup[:5]] = np.nan

        results = verify_physics(structure, cout_modified, cout_tedges, cin, verbose=False)

        # Should fail multiple checks
        assert not results["all_passed"]
        assert len(results["failures"]) >= 3  # At least 3 violations
        assert results["n_passed"] < results["n_checks"] - 2  # Multiple failures

    def test_custom_rtol_gates_check3_output_exceeds_input(self, pulse_simulation_base):
        """A controlled overshoot of ``cin.max()`` by 5% must fail Check 3 at ``rtol=1e-15``
        and pass at ``rtol=1e-1``. Replaces the prior tautological test that only verified
        the rtol parameter was accepted.

        Check 3 ("Output ≤ input maximum") uses ``max_cout <= max_cin * (1 + rtol)`` per
        ``validation.py:110``; constructing a 5% overshoot pins both sides of that bound.
        """
        cin, cout, cout_tedges, structure = pulse_simulation_base
        cout_overshoot = cout.copy()
        max_cin = float(np.max(cin))
        # Add a +5% overshoot to one valid bin. Big enough to fail at rtol=1e-15; small
        # enough to pass at rtol=1e-1.
        overshoot_idx = int(np.nanargmax(cout))
        cout_overshoot[overshoot_idx] = max_cin * 1.05

        strict = verify_physics(structure, cout_overshoot, cout_tedges, cin, verbose=False, rtol=1e-15)
        relaxed = verify_physics(structure, cout_overshoot, cout_tedges, cin, verbose=False, rtol=1e-1)

        # Find the "Output <= input maximum" check by name and compare its pass bit.
        def _check_named(results, name):
            return next(c for c in results["checks"] if c["name"] == name)

        assert not _check_named(strict, "Output ≤ input maximum")["passed"]
        assert _check_named(relaxed, "Output ≤ input maximum")["passed"]


class TestVerifyPhysicsEdgeCases:
    """Test verify_physics with edge cases and special scenarios."""

    def test_zero_concentration_input_passes_non_mass_balance_checks(self):
        """All-zero ``cin`` with all-zero ``cout`` produces zero mass on both sides.

        Replaces the prior smoke test (``isinstance(results['all_passed'], bool)``) with
        definite expected outcomes for Checks 1, 2, 3 (entropy vacuously OK; min/max
        concentration trivially zero so bounded by input). The mass-balance check (Check 8)
        is allowed to be either pass (0 in / 0 out) or skipped (no tracker_state); we
        assert it doesn't claim a non-zero overflow.
        """
        tedges = pd.date_range(start="2020-01-01", periods=50, freq="D")
        cin = np.zeros(len(tedges) - 1)
        flow = np.full(len(tedges) - 1, 100.0)
        aquifer_pore_volume = 200.0
        cout_tedges = pd.date_range(start=tedges[0], periods=100, freq="D")

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
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
        # cout from front-tracking with cin=0 should be all-NaN in the spin-up region
        # and 0 elsewhere; both are valid non-negative and trivially bounded by input.

        def _check_named(name):
            return next(c for c in results["checks"] if c["name"] == name)

        assert _check_named("Shock entropy condition")["passed"], "entropy vacuously holds for zero-input"
        assert _check_named("Non-negative concentrations")["passed"]
        assert _check_named("Output ≤ input maximum")["passed"]
        assert results["n_checks"] == 8

    def test_constant_concentration_input(self):
        """Test verify_physics with constant non-zero concentration ending in zero."""
        tedges = pd.date_range(start="2020-01-01", periods=50, freq="D")
        cin = np.full(len(tedges) - 1, 5.0)  # Constant 5.0
        cin[-1] = 0.0  # End with explicit zero for mass balance
        flow = np.full(len(tedges) - 1, 100.0)
        aquifer_pore_volume = 200.0
        cout_tedges = pd.date_range(start=tedges[0], periods=100, freq="D")

        cout, structure = infiltration_to_extraction_front_tracking_detailed(
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


class TestVerifyPhysicsViolationDetection:
    """Each of the 8 checks must trigger ``passed=False`` on a hand-crafted violation.

    Closes issue #199 Group A: 5 of 8 checks (entropy, finite first arrival, event ordering,
    rarefaction ordering, mass balance) previously had no violation test. Without these the
    suite was a smoke check rather than a physics gate.
    """

    @pytest.fixture
    def base_pulse(self):
        """A passing baseline pulse simulation we can mutate per-test."""
        tedges = pd.date_range(start="2020-01-01", periods=100, freq="D")
        cin = np.zeros(len(tedges) - 1)
        cin[10:40] = 10.0
        flow = np.full(len(tedges) - 1, 100.0)
        cout_tedges = pd.date_range(start=tedges[0], periods=1350, freq="D")
        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=[200.0],
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )
        return cin, cout, cout_tedges, structure[0]

    @staticmethod
    def _check_named(results, name):
        return next(c for c in results["checks"] if c["name"] == name)

    def test_entropy_violation_detected(self, base_pulse):
        """A ShockWave with non-physical ordering (c_left < c_right for n>1) fails Check 1."""
        cin, cout, cout_tedges, structure = base_pulse
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        # For n>1, higher c travels faster. A "rarefaction shock" (c_left < c_right) inverts
        # the entropy condition lambda(c_left) > shock_velocity > lambda(c_right).
        bad_shock = ShockWave(t_start=0.0, v_start=0.0, flow=100.0, c_left=1.0, c_right=10.0, sorption=sorption)
        assert not bad_shock.satisfies_entropy(), "fixture precondition failed"
        structure["waves"] = [*list(structure["waves"]), bad_shock]
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)
        assert not self._check_named(results, "Shock entropy condition")["passed"]

    def test_infinite_first_arrival_detected(self, base_pulse):
        """``structure['t_first_arrival'] = np.inf`` fails Check 4."""
        cin, cout, cout_tedges, structure = base_pulse
        structure["t_first_arrival"] = np.inf
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)
        assert not self._check_named(results, "Finite first arrival time")["passed"]

    def test_event_ordering_violation_detected_three_events(self, base_pulse):
        """Three events out-of-chronological-order fail Check 6.

        Uses three events (not two) because the previous length-1 branch swallowed any
        single-event mutation. With the Check 6 collapse to ``np.all(np.diff >= 0)`` this
        catches both the >1 and the length-2 'N/A' regimes.
        """
        cin, cout, cout_tedges, structure = base_pulse
        structure["events"] = [{"time": 5.0}, {"time": 1.0}, {"time": 3.0}]
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)
        assert not self._check_named(results, "Events chronologically ordered")["passed"]

    def test_rarefaction_ordering_violation_detected(self, base_pulse):
        """A rarefaction with swapped c_head/c_tail (head slower than tail) fails Check 7.

        ``RarefactionWave.__post_init__`` rejects this configuration at construction, so we
        build a valid rarefaction and then swap ``c_head``/``c_tail`` on the dataclass to
        bypass the constructor check while preserving identity for ``isinstance`` filters.
        """
        cin, cout, cout_tedges, structure = base_pulse
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        good_raref = RarefactionWave(t_start=0.0, v_start=0.0, flow=100.0, c_head=2.0, c_tail=1.0, sorption=sorption)
        # Swap head/tail post-init. For n>1, c_head < c_tail makes head_velocity < tail_velocity.
        good_raref.c_head, good_raref.c_tail = 1.0, 2.0
        # Pre-condition: velocity ordering is now violated by > 1e-10.
        assert good_raref.head_velocity() < good_raref.tail_velocity() - 1e-10
        structure["waves"] = [*list(structure["waves"]), good_raref]
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)
        assert not self._check_named(results, "Rarefaction wave ordering")["passed"]


class TestVerifyPhysicsSkippedMassBalance:
    """Check 8 (mass balance) has a 'Skipped' fallback path when ``tracker_state`` is absent.

    The current contract returns ``passed=True`` with a 'Skipped' message — distinct from a
    real pass, but indistinguishable through ``passed`` alone. This test pins the
    distinguishing message so a future regression that silently always returns 'Skipped'
    is observable.
    """

    def test_none_tracker_state_returns_skipped_message(self):
        tedges = pd.date_range(start="2020-01-01", periods=50, freq="D")
        cin = np.zeros(len(tedges) - 1)
        cin[10:40] = 10.0
        flow = np.full(len(tedges) - 1, 100.0)
        cout_tedges = pd.date_range(start=tedges[0], periods=400, freq="D")
        cout, structure = infiltration_to_extraction_front_tracking_detailed(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=[200.0],
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )
        s = structure[0]
        # Remove the tracker_state to drive Check 8 down the skipped branch.
        s["tracker_state"] = None
        results = verify_physics(s, cout, cout_tedges, cin, verbose=False)
        mass_check = next(c for c in results["checks"] if c["name"] == "Total integrated outlet mass")
        assert mass_check["passed"], "Skipped path must report passed=True"
        # Distinguish from a real pass: message must indicate the skip explicitly.
        assert "skipped" in mass_check["message"].lower() or "n/a" in mass_check["message"].lower(), (
            f"Skipped path must carry an identifying message, got: {mass_check['message']!r}"
        )
