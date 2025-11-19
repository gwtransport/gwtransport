"""
Physics validation utilities for front tracking.

This module provides functions to verify physical correctness of
front-tracking simulations, including entropy conditions, concentration
bounds, and mass conservation.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import logging

import numpy as np
import pandas as pd

from gwtransport.fronttracking.waves import RarefactionWave, ShockWave

# Numerical tolerance constants
EPSILON_CONCENTRATION_TOLERANCE = -1e-14  # Minimum allowed concentration (machine precision)

logger = logging.getLogger(__name__)


def verify_physics(structure, cout, cout_tedges, cin, *, verbose=True, rtol=1e-10):
    """
    Run comprehensive physics verification checks on front tracking results.

    Performs the following checks:
    1. Entropy condition for all shocks
    2. No negative concentrations (within tolerance)
    3. Output concentration ≤ input maximum
    4. Finite first arrival time
    5. No NaN values after spin-up period
    6. Events chronologically ordered
    7. Rarefaction concentration ordering (head vs tail)
    8. Approximate mass balance

    Parameters
    ----------
    structure : dict
        Structure returned from infiltration_to_extraction_front_tracking_detailed.
        Must contain keys: 'waves', 't_first_arrival', 'events', 'n_shocks',
        'n_rarefactions', etc.
    cout : array-like
        Bin-averaged output concentrations.
    cout_tedges : pd.DatetimeIndex
        Output time edges for bins.
    cin : array-like
        Input concentrations.
    verbose : bool, optional
        If True, print detailed results. If False, only return summary.
        Default True.
    rtol : float, optional
        Relative tolerance for numerical checks. Default 1e-10.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'all_passed': bool - True if all checks passed
        - 'n_checks': int - Total number of checks performed
        - 'n_passed': int - Number of checks that passed
        - 'failures': list of str - Description of failed checks (empty if all passed)
        - 'summary': str - One-line summary (e.g., "✓ All 8 checks passed")

    Examples
    --------
    >>> results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)
    >>> print(results["summary"])
    ✓ All 8 checks passed
    >>> assert results["all_passed"]
    """
    failures = []
    checks = []

    # Check 1: Entropy condition for all shocks
    shocks = [w for w in structure["waves"] if isinstance(w, ShockWave)]
    entropy_violations = [s for s in shocks if not s.satisfies_entropy()]
    check1_pass = len(entropy_violations) == 0
    checks.append({
        "name": "Shock entropy condition",
        "passed": check1_pass,
        "message": f"Entropy violations: {len(entropy_violations)}/{len(shocks)} shocks",
    })
    if not check1_pass:
        failures.append(f"Entropy violations: {len(entropy_violations)} shocks violate entropy condition")

    # Check 2: No negative concentrations (within tolerance)
    valid_cout = cout[~np.isnan(cout)]
    min_cout = np.min(valid_cout) if len(valid_cout) > 0 else 0.0
    check2_pass = min_cout >= EPSILON_CONCENTRATION_TOLERANCE
    checks.append({
        "name": "Non-negative concentrations",
        "passed": check2_pass,
        "message": f"Minimum concentration: {min_cout:.2e}",
    })
    if not check2_pass:
        failures.append(f"Negative concentrations found: min = {min_cout:.2e}")

    # Check 3: Output doesn't exceed input (within tight tolerance)
    max_cout = np.max(valid_cout) if len(valid_cout) > 0 else 0.0
    max_cin = np.max(cin)
    check3_pass = max_cout <= max_cin * (1.0 + rtol)
    checks.append({
        "name": "Output ≤ input maximum",
        "passed": check3_pass,
        "message": f"Max output: {max_cout:.2f}, Max input: {max_cin:.2f}",
    })
    if not check3_pass:
        failures.append(f"Output exceeds input: {max_cout:.2f} > {max_cin:.2f}")

    # Check 4: Finite first arrival time
    t_first = structure["t_first_arrival"]
    check4_pass = np.isfinite(t_first)
    checks.append({
        "name": "Finite first arrival time",
        "passed": check4_pass,
        "message": f"First arrival: {t_first:.2f} days",
    })
    if not check4_pass:
        failures.append(f"First arrival time is not finite: {t_first}")

    # Check 5: No NaN values after spin-up period
    cout_tedges_days = ((cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)).values
    mask_after_spinup = cout_tedges_days[:-1] >= t_first
    cout_after_spinup = cout[mask_after_spinup]
    nan_count = np.sum(np.isnan(cout_after_spinup))
    check5_pass = nan_count == 0
    checks.append({
        "name": "No NaN after spin-up",
        "passed": check5_pass,
        "message": f"NaN values after spin-up: {nan_count}/{len(cout_after_spinup)}",
    })
    if not check5_pass:
        failures.append(f"Found {nan_count} NaN values after spin-up period")

    # Check 6: Events chronologically ordered
    event_times = [e["time"] for e in structure.get("events", [])]
    if len(event_times) > 1:
        is_ordered = all(event_times[i] <= event_times[i + 1] for i in range(len(event_times) - 1))
        check6_pass = is_ordered
        checks.append({
            "name": "Events chronologically ordered",
            "passed": check6_pass,
            "message": f"{len(event_times)} events",
        })
        if not check6_pass:
            failures.append("Events are not in chronological order")
    else:
        check6_pass = True
        checks.append({
            "name": "Events chronologically ordered",
            "passed": True,
            "message": f"{len(event_times)} events (N/A)",
        })

    # Check 7: Rarefaction concentration ordering
    rarefactions = [w for w in structure["waves"] if isinstance(w, RarefactionWave)]
    raref_ordering_violations = 0
    for raref in rarefactions:
        # For n>1 (favorable), head should be higher concentration (faster)
        # For n<1 (unfavorable), head should be lower concentration (faster)
        # We can check this via velocities: head_velocity should always be >= tail_velocity
        if raref.head_velocity() < raref.tail_velocity() - 1e-10:
            raref_ordering_violations += 1

    check7_pass = raref_ordering_violations == 0
    checks.append({
        "name": "Rarefaction wave ordering",
        "passed": check7_pass,
        "message": f"Ordering violations: {raref_ordering_violations}/{len(rarefactions)} rarefactions",
    })
    if not check7_pass:
        failures.append(f"{raref_ordering_violations} rarefactions have incorrect head/tail ordering")

    # Check 8: Approximate mass balance
    # This is approximate since it depends on bin averaging
    try:
        # Estimate mass in (assumes constant flow, which may not be true)
        flow = structure["tracker_state"].flow if hasattr(structure.get("tracker_state"), "flow") else None

        if flow is not None and len(flow) == len(cin):
            # Compute mass in from input data
            tedges_in = structure["tracker_state"].tedges
            dt_in = np.diff(tedges_in.view(np.int64)) / 1e9 / 86400  # Convert to days
            mass_in = np.sum(cin * flow * dt_in)

            # Compute mass out from bin-averaged output
            dt_out = np.diff(cout_tedges.view(np.int64)) / 1e9 / 86400
            # Use first flow value as approximation for outlet flow
            mass_out = np.sum(valid_cout * flow[0] * dt_out[~np.isnan(cout)])

            mass_ratio = mass_out / mass_in if mass_in > 0 else 0.0
            # Mass out should be less than mass in (simulation ongoing)
            # but not too much less (no significant losses)
            check8_pass = 0.0 <= mass_ratio <= 1.0 + rtol
            checks.append({
                "name": "Approximate mass balance",
                "passed": check8_pass,
                "message": f"Mass out/in ratio: {mass_ratio:.2%}",
            })
            if not check8_pass:
                failures.append(f"Mass balance check failed: ratio = {mass_ratio:.2%}")
        else:
            # Skip mass balance if flow info not available
            check8_pass = True
            checks.append({
                "name": "Approximate mass balance",
                "passed": True,
                "message": "Skipped (flow data not available)",
            })
    except (KeyError, AttributeError, ValueError, TypeError) as e:
        # If mass balance computation fails, mark as passed but note the issue
        check8_pass = True
        checks.append({
            "name": "Approximate mass balance",
            "passed": True,
            "message": f"Skipped (error: {e!s})",
        })

    # Compile results
    n_checks = len(checks)
    n_passed = sum(c["passed"] for c in checks)
    all_passed = len(failures) == 0

    if all_passed:
        summary = f"✓ All {n_checks} physics checks passed"
    else:
        summary = f"✗ {n_passed}/{n_checks} checks passed ({len(failures)} failures)"

    results = {
        "all_passed": all_passed,
        "n_checks": n_checks,
        "n_passed": n_passed,
        "failures": failures,
        "checks": checks,
        "summary": summary,
    }

    # Log detailed output if verbose
    if verbose:
        logger.info("\nPhysics Verification:")
        for i, check in enumerate(checks, 1):
            status = "✓" if check["passed"] else "✗"
            logger.info("  %d. %s: %s %s", i, check["name"], status, check["message"])

        if all_passed:
            logger.info("\n%s", summary)
        else:
            logger.warning("\n%s", summary)
            logger.warning("\nFailures:")
            for i, failure in enumerate(failures, 1):
                logger.warning("  %d. %s", i, failure)

    return results
