"""
Tests for verify_physics function from fronttracking.validation.

These tests are based on Example 1 from notebook 09_Front_Tracking_Rarefaction_Waves.ipynb
which demonstrates a concentration pulse with favorable sorption (n>1).
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.advection import infiltration_to_extraction_front_tracking_detailed
from gwtransport.fronttracking.validation import verify_physics


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

    def test_verify_physics_check_count(self, pulse_simulation_data):
        """Test that verify_physics runs the expected number of checks."""
        cin, cout, cout_tedges, structure = pulse_simulation_data

        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)

        # Should run 8 checks (as documented in function docstring)
        assert results["n_checks"] == 8
        assert len(results["checks"]) == 8

    def test_verify_physics_check_names(self, pulse_simulation_data):
        """Test that all expected checks are present."""
        cin, cout, cout_tedges, structure = pulse_simulation_data

        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)

        expected_checks = {
            "Shock entropy condition",
            "Non-negative concentrations",
            "Output ≤ input maximum",
            "Finite first arrival time",
            "No NaN after spin-up",
            "Events chronologically ordered",
            "Rarefaction wave ordering",
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

    def test_custom_rtol(self, pulse_simulation_base):
        """Test verify_physics with custom relative tolerance."""
        cin, cout, cout_tedges, structure = pulse_simulation_base

        # Run with very strict tolerance
        results_strict = verify_physics(structure, cout, cout_tedges, cin, verbose=False, rtol=1e-15)

        # Run with relaxed tolerance
        results_relaxed = verify_physics(structure, cout, cout_tedges, cin, verbose=False, rtol=1e-6)

        # Both should have same number of checks
        assert results_strict["n_checks"] == results_relaxed["n_checks"]

        # With valid data, both should pass (unless numerical precision is an issue)
        # But we're mainly testing that rtol parameter is accepted
        assert isinstance(results_strict["all_passed"], bool)
        assert isinstance(results_relaxed["all_passed"], bool)


class TestVerifyPhysicsEdgeCases:
    """Test verify_physics with edge cases and special scenarios."""

    def test_zero_concentration_input(self):
        """Test verify_physics with all-zero concentration input."""
        tedges = pd.date_range(start="2020-01-01", periods=50, freq="D")
        cin = np.zeros(len(tedges) - 1)  # All zeros
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

        # Zero concentration may cause first arrival time to be inf or cause mass balance issues
        # Just check that verify_physics runs and returns valid structure
        assert isinstance(results["all_passed"], bool)
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
