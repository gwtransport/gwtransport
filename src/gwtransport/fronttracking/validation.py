"""
Physics validation utilities for front tracking in (V, θ) coordinates.

This module provides functions to verify physical correctness of front-tracking
simulations, including entropy conditions, concentration bounds, mass conservation,
and event ordering. The solver runs in cumulative-flow coordinate
``θ = ∫flow(t') dt'``; events on ``state.events`` carry ``"theta"`` (m³). Because
``flow ≥ 0`` is enforced, θ is monotone non-decreasing in t, so θ-ordering and
chronological ordering are equivalent.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import logging

import numpy as np

from gwtransport.fronttracking.output import (
    compute_cumulative_inlet_mass,
    compute_total_outlet_mass,
)
from gwtransport.fronttracking.waves import RarefactionWave, ShockWave

# Numerical tolerance constants
EPSILON_CONCENTRATION_TOLERANCE = -1e-14  # Minimum allowed concentration (machine precision)
EPSILON_RAREFACTION_ORDERING = 1e-10  # Slack for c_head/c_tail θ-speed ordering

logger = logging.getLogger(__name__)


def verify_physics(structure, cout, cout_tedges, cin, *, verbose=True, rtol=1e-10):
    """
    Run comprehensive physics verification checks on front tracking results.

    Performs the following checks:

    1. Entropy condition for all shocks
    2. No negative concentrations (within tolerance)
    3. Output concentration <= input maximum
    4. Finite first arrival θ
    5. No NaN values after spin-up period
    6. Events θ-ordered (equivalent to chronological under non-negative flow)
    7. Rarefaction head/tail θ-speed ordering
    8. Total integrated outlet mass (until all mass exits)

    Parameters
    ----------
    structure : dict
        Structure returned from ``infiltration_to_extraction_nonlinear_sorption``.
        Must contain keys: ``'waves'``, ``'theta_first_arrival'``, ``'events'``,
        ``'n_shocks'``, ``'n_rarefactions'``, and optionally ``'tracker_state'``.
    cout : array-like
        Bin-averaged output concentrations.
    cout_tedges : pandas.DatetimeIndex
        Output time edges for bins (only used for the spin-up mask).
    cin : array-like
        Input concentrations.
    verbose : bool, optional
        If True, print detailed results. If False, only return summary. Default True.
    rtol : float, optional
        Relative tolerance for numerical checks. Default 1e-10.

    Returns
    -------
    results : dict
        Dictionary containing:

        - ``'all_passed'``: bool - True if all checks passed
        - ``'n_checks'``: int - Total number of checks performed
        - ``'n_passed'``: int - Number of checks that passed
        - ``'failures'``: list of str - Description of failed checks (empty if all passed)
        - ``'summary'``: str - One-line summary

    Examples
    --------
    ::

        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)
        print(results["summary"])
        assert results["all_passed"]
    """
    failures: list[str] = []
    checks: list[dict] = []

    # Check 1: Entropy condition for all shocks (Lax in (V, θ): λ_θ(C_L) >= s >= λ_θ(C_R)).
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
        "name": "Output <= input maximum",
        "passed": check3_pass,
        "message": f"Max output: {max_cout:.2f}, Max input: {max_cin:.2f}",
    })
    if not check3_pass:
        failures.append(f"Output exceeds input: {max_cout:.2f} > {max_cin:.2f}")

    # Check 4: Finite first arrival θ
    theta_first = structure["theta_first_arrival"]
    check4_pass = np.isfinite(theta_first)
    checks.append({
        "name": "Finite first arrival θ",
        "passed": check4_pass,
        "message": f"First arrival: θ={theta_first:.2f}",
    })
    if not check4_pass:
        failures.append(f"First arrival θ is not finite: {theta_first}")

    # Check 5: No NaN values after spin-up
    tracker_state = structure.get("tracker_state")
    if tracker_state is not None and np.isfinite(theta_first):
        theta_at_edge = np.asarray([
            tracker_state.theta_at_t(float(t)) for t in (cout_tedges[:-1] - cout_tedges[0]).total_seconds() / 86400.0
        ])
        mask_after_spinup = theta_at_edge >= theta_first
    elif not np.isfinite(theta_first):
        # No spin-up bound — every output row counts as "after spin-up".
        mask_after_spinup = np.ones(len(cout), dtype=bool)
    else:
        # No tracker state to translate — nothing to check.
        mask_after_spinup = np.zeros(len(cout), dtype=bool)
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

    # Check 6: Events θ-ordered.
    event_thetas = [e["theta"] for e in structure.get("events", [])]
    if len(event_thetas) > 1:
        is_ordered = all(event_thetas[i] <= event_thetas[i + 1] for i in range(len(event_thetas) - 1))
        check6_pass = is_ordered
        checks.append({
            "name": "Events θ-ordered",
            "passed": check6_pass,
            "message": f"{len(event_thetas)} events",
        })
        if not check6_pass:
            failures.append("Events are not θ-ordered")
    else:
        check6_pass = True
        checks.append({
            "name": "Events θ-ordered",
            "passed": True,
            "message": f"{len(event_thetas)} events (N/A)",
        })

    # Check 7: Rarefaction head/tail θ-speed ordering.
    # In (V, θ) every speed is dV/dθ = 1/R(C); a valid rarefaction has the head
    # propagating faster than (or equal to) the tail.
    rarefactions = [w for w in structure["waves"] if isinstance(w, RarefactionWave)]
    raref_ordering_violations = 0
    for raref in rarefactions:
        if raref.head_speed() < raref.tail_speed() - EPSILON_RAREFACTION_ORDERING:
            raref_ordering_violations += 1

    check7_pass = raref_ordering_violations == 0
    checks.append({
        "name": "Rarefaction wave ordering",
        "passed": check7_pass,
        "message": f"Ordering violations: {raref_ordering_violations}/{len(rarefactions)} rarefactions",
    })
    if not check7_pass:
        failures.append(f"{raref_ordering_violations} rarefactions have incorrect head/tail ordering")

    # Check 8: Total integrated outlet mass vs total inlet mass (in θ-space).
    if tracker_state is not None and hasattr(tracker_state, "theta_edges"):
        v_outlet = tracker_state.v_outlet
        sorption = tracker_state.sorption
        theta_edges_arr = np.asarray(tracker_state.theta_edges, dtype=float)

        total_mass_in = compute_cumulative_inlet_mass(
            theta=float(theta_edges_arr[-1]), cin=cin, theta_edges=theta_edges_arr
        )

        total_mass_out = compute_total_outlet_mass(
            v_outlet=v_outlet, sorption=sorption, cin=cin, theta_edges=theta_edges_arr
        )
        theta_integration_end = float(theta_edges_arr[-1])

        # ``compute_total_outlet_mass`` returns the asymptotic m_out = m_in − C_T(c_∞)·V_outlet
        # where c_∞ = cin[-1]. For the conservation identity at θ→∞ to hold, the
        # validation check must add back the steady-state aquifer mass that stays
        # in the domain for c_∞ > 0. Equivalent invariant: m_in_total ≈ m_out_total +
        # m_dom_asymptotic, where m_dom_asymptotic = C_T(c_∞)·V_outlet.
        c_inf = float(cin[-1]) if len(cin) > 0 else 0.0
        m_dom_asymptotic = float(sorption.total_concentration(c_inf)) * v_outlet

        if total_mass_in > 0:
            relative_error_total = abs(total_mass_out + m_dom_asymptotic - total_mass_in) / total_mass_in
        else:
            relative_error_total = abs(total_mass_out + m_dom_asymptotic - total_mass_in)

        check8_pass = relative_error_total <= max(rtol, 1e-6)
        checks.append({
            "name": "Total integrated outlet mass",
            "passed": check8_pass,
            "message": (
                f"Relative error: {relative_error_total:.2e} (integrated to θ={theta_integration_end:.1f}; "
                f"m_dom_asymptotic={m_dom_asymptotic:.2e} for c_∞={c_inf:.3f})"
            ),
        })
        if not check8_pass:
            failures.append(
                f"Total outlet mass mismatch: relative_error={relative_error_total:.2e} > {rtol:.2e} "
                f"(total_mass_out={total_mass_out:.6e}, total_mass_in={total_mass_in:.6e}, "
                f"m_dom_asymptotic={m_dom_asymptotic:.6e}, "
                f"θ_integration_end={theta_integration_end:.1f})"
            )
    else:
        check8_pass = True
        checks.append({
            "name": "Total integrated outlet mass",
            "passed": True,
            "message": "Skipped (tracker state not available)",
        })

    # Compile results
    n_checks = len(checks)
    n_passed = sum(c["passed"] for c in checks)
    all_passed = len(failures) == 0

    if all_passed:
        summary = f"All {n_checks} physics checks passed"
    else:
        summary = f"{n_passed}/{n_checks} checks passed ({len(failures)} failures)"

    results = {
        "all_passed": all_passed,
        "n_checks": n_checks,
        "n_passed": n_passed,
        "failures": failures,
        "checks": checks,
        "summary": summary,
    }

    if verbose:
        logger.info("\nPhysics Verification:")
        for i, check in enumerate(checks, 1):
            status = "PASS" if check["passed"] else "FAIL"
            logger.info("  %d. %s: %s %s", i, check["name"], status, check["message"])

        if all_passed:
            logger.info("\n%s", summary)
        else:
            logger.warning("\n%s", summary)
            logger.warning("\nFailures:")
            for i, failure in enumerate(failures, 1):
                logger.warning("  %d. %s", i, failure)

    return results
