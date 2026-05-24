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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from gwtransport.fronttracking.output import (
    compute_breakthrough_curve,
    compute_cumulative_inlet_mass,
    compute_domain_mass,
)
from gwtransport.fronttracking.waves import ShockWave

if TYPE_CHECKING:
    from gwtransport.fronttracking.solver import FrontTrackerState

# Numerical tolerance constants
EPSILON_CONCENTRATION_TOLERANCE = -1e-14  # Minimum allowed concentration (machine precision)

# Mass-balance check (7) tolerance and grid.
#
# The independent outlet mass integrates the breakthrough curve with the trapezoid
# rule (see ``_independent_outlet_mass``). For a shock-bearing, sharply-curved
# breakthrough this is only first-order accurate: the measured relative error for the
# canonical favorable-sorption pulse oscillates at ~5e-4 across a modest grid band
# (2000-4000 points; it only falls reliably below 1e-4 above ~20000 points). The grid is
# deliberately kept modest because a numerical DecayingShockWave makes
# ``compute_breakthrough_curve`` slow (seconds to minutes for large grids), so we cannot
# refine the integral to machine precision. ``_MASS_BALANCE_RTOL`` therefore bounds that
# ~5e-4 grid noise with ~16x margin; a physical 30% inlet-mass error yields a relative
# error of ~0.23 (= 1 - 1/1.3) to ~0.43 (= 1/0.7 - 1), i.e. ~20-40x this floor, so the
# check still has strong teeth against a genuine conservation failure.
_MASS_BALANCE_RTOL = 1e-2
_MASS_BALANCE_GRID_POINTS = 3000

logger = logging.getLogger(__name__)


def _independent_outlet_mass(tracker_state: FrontTrackerState, *, n_grid: int = _MASS_BALANCE_GRID_POINTS) -> float:
    """Outlet-side mass total computed independently of the ``m_in - m_dom`` identity.

    Integrated to θ_max (the last θ-bin edge). Sums the mass that has already left through
    the outlet, ``∫₀^θ_max c_out(τ) dτ``, and the mass still in the domain, ``m_dom(θ_max)``.
    The breakthrough integral uses :func:`compute_breakthrough_curve`, which dispatches
    :func:`concentration_at_point` directly (pure wave evaluation), so this total never
    references the conservation identity ``m_out = m_in − m_dom`` that the mass-balance
    check is meant to test. Comparing it to :func:`compute_cumulative_inlet_mass` at θ_max
    is therefore a genuine, non-tautological conservation check: for a pulse that has not
    fully broken through by θ_max, the partial breakthrough integral plus the residual
    domain mass still equals the cumulative inlet mass.

    Parameters
    ----------
    tracker_state : FrontTrackerState
        Solver state; must expose ``v_outlet``, ``sorption``, ``waves`` and
        ``theta_edges``.
    n_grid : int, optional
        Number of trapezoid nodes for the breakthrough integral over ``[0, θ_max]``.

    Returns
    -------
    float
        Independent outlet-side mass total [mass].
    """
    v_outlet = tracker_state.v_outlet
    sorption = tracker_state.sorption
    waves = tracker_state.waves
    theta_max = float(np.asarray(tracker_state.theta_edges, dtype=float)[-1])

    if theta_max <= 0.0:
        return compute_domain_mass(theta=theta_max, v_outlet=v_outlet, waves=waves, sorption=sorption)

    theta_grid: npt.NDArray[np.floating] = np.linspace(0.0, theta_max, n_grid)
    breakthrough = compute_breakthrough_curve(theta_grid, v_outlet, waves, sorption)
    mass_out = float(np.trapezoid(breakthrough, theta_grid))
    mass_dom = compute_domain_mass(theta=theta_max, v_outlet=v_outlet, waves=waves, sorption=sorption)
    return mass_out + mass_dom


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
    7. Mass conservation: independent outlet integral + domain mass == inlet mass at θ_max

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
        Relative tolerance for numerical checks. Default 1e-10. For the mass-balance
        check (7) the effective tolerance is ``max(rtol, _MASS_BALANCE_RTOL)`` because
        that check integrates a shock-bearing breakthrough curve and is only first-order
        accurate (see ``_MASS_BALANCE_RTOL``).

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

    # Check 6: Events θ-ordered. ``np.all(np.diff(...) >= 0)`` is vacuously True
    # for an empty/singleton sequence, so the same expression covers the N/A case;
    # only the message differs.
    event_thetas = [e["theta"] for e in structure.get("events", [])]
    check6_pass = bool(np.all(np.diff(event_thetas) >= 0))
    checks.append({
        "name": "Events θ-ordered",
        "passed": check6_pass,
        "message": f"{len(event_thetas)} events" if len(event_thetas) > 1 else f"{len(event_thetas)} events (N/A)",
    })
    if not check6_pass:
        failures.append("Events are not θ-ordered")

    # Check 7: Total integrated outlet mass vs total inlet mass (in θ-space).
    #
    # The outlet-side total is computed *independently* of the conservation identity
    # ``m_out = m_in − m_dom`` (which the old check used on both sides, making it an
    # algebraic tautology that passed for any input). ``_independent_outlet_mass``
    # integrates the breakthrough curve and adds the spatial domain mass; both come from
    # direct wave evaluation, so a mismatch with the cumulative inlet mass signals a real
    # conservation failure. Integrated to θ_max (the last θ-bin edge); for pulses that
    # have not fully broken through there, the partial breakthrough integral plus the
    # residual domain mass still equals the cumulative inlet mass.
    if tracker_state is not None and hasattr(tracker_state, "theta_edges"):
        theta_edges_arr = np.asarray(tracker_state.theta_edges, dtype=float)
        theta_integration_end = float(theta_edges_arr[-1])

        total_mass_in = compute_cumulative_inlet_mass(theta=theta_integration_end, cin=cin, theta_edges=theta_edges_arr)
        independent_mass_out = _independent_outlet_mass(tracker_state)

        if total_mass_in > 0:
            relative_error_total = abs(independent_mass_out - total_mass_in) / total_mass_in
        else:
            relative_error_total = abs(independent_mass_out - total_mass_in)

        mass_balance_threshold = max(rtol, _MASS_BALANCE_RTOL)
        check7_pass = relative_error_total <= mass_balance_threshold
        checks.append({
            "name": "Total integrated outlet mass",
            "passed": check7_pass,
            "message": (
                f"Relative error: {relative_error_total:.2e} (independent outlet integral to "
                f"θ={theta_integration_end:.1f}; threshold {mass_balance_threshold:.2e})"
            ),
        })
        if not check7_pass:
            failures.append(
                f"Total outlet mass mismatch: relative_error={relative_error_total:.2e} > "
                f"{mass_balance_threshold:.2e} (independent_mass_out={independent_mass_out:.6e}, "
                f"total_mass_in={total_mass_in:.6e}, θ_integration_end={theta_integration_end:.1f})"
            )
    else:
        check7_pass = True
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
