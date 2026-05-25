"""
Regression and conservation tests for issue #222.

A wetting shock that collides with a *fully drained* (c=0) drying rarefaction
fan used to raise ``ValueError: c_decay_initial must be positive`` deep in the
solver (``handlers.py`` → ``DecayingShockWave.__post_init__``). The collision
hands the fan-tail concentration to the merged
:class:`~gwtransport.fronttracking.waves.DecayingShockWave` as
``c_decay_initial``; for a fully drained fan that value is exactly ``0``.

The fix floors ``c_decay_initial`` to the shared dry-soil singularity floor
``_C_MIN`` (matching the package's retardation-floor convention) rather than
rejecting the collision, and generalises the fan-exhaustion orientation guard
so the *growing* decay (``c_decay_initial < c_fan_tail``) is handled. For the
issue's inputs the decaying shock asymptotically merges with the fan head, so
there is no finite exhaustion event -- the solver terminates cleanly.

These end-to-end tests drive the public
:func:`gwtransport.percolation.root_zone_to_water_table_kinematic_wave` entry
point. Mass conservation is checked via an *independent* route -- a fine-grid
trapezoidal integral of :func:`compute_breakthrough_curve` over θ compared to
``Σ cin·Δθ − compute_domain_mass(θ_end)`` -- following the
``test_percolation.py::test_b5a`` pattern. This deliberately avoids the
``compute_bin_averaged_concentration_exact`` route (which is itself
``m_in − m_dom`` and would make the check tautological).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import _C_MIN, BrooksCoreyConductivity
from gwtransport.fronttracking.output import compute_breakthrough_curve, compute_domain_mass
from gwtransport.fronttracking.waves import DecayingShockWave
from gwtransport.percolation import root_zone_to_water_table_kinematic_wave

# Brooks-Corey soil and short column from issue #222.
_BC_KWARGS = {"theta_r": 0.05, "theta_s": 0.40, "k_s": 0.01, "brooks_corey_lambda": 0.5}
_V_OUTLET = 0.05  # cumulative pore volume of the (short) column = theta_s * z_wt


def _run_percolation(q_root: np.ndarray):
    """Drive the percolation solver and return ``(q_wt, tracker_state)``.

    Parameters
    ----------
    q_root : numpy.ndarray
        Root-zone flux per daily bin (m/day).

    Returns
    -------
    q_wt : numpy.ndarray
        Water-table flux (the bin-averaged public output).
    state : FrontTrackerState
        Complete solver state (wave list, θ-edges, sorption) for the single
        column at ``_V_OUTLET``.
    """
    tedges = pd.date_range("2024-01-01", periods=len(q_root) + 1, freq="D")
    with warnings.catch_warnings():
        # The wet-zero-wet forcing pushes a few output θ-bins past the inlet
        # window; that back-transform warning is incidental here -- conservation
        # is read from the wave list / compute_domain_mass directly, not q_wt.
        warnings.simplefilter("ignore")
        q_wt, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=[_V_OUTLET],
            **_BC_KWARGS,
        )
    return q_wt, structures[0]["tracker_state"]


def _independent_conservation_rel_err(q_root: np.ndarray, state, n_grid: int = 600) -> float:
    """Relative mass-balance residual via the independent breakthrough integral.

    Computes ``|∫ breakthrough dθ + M_domain(θ_end) − M_in| / M_in`` where the
    breakthrough integral is a trapezoidal rule over
    :func:`compute_breakthrough_curve` (independent of the ``m_in − m_dom``
    back-transform), ``M_domain`` is the spatial domain-mass integral, and
    ``M_in = Σ cin·Δθ`` in the solver's native θ/V frame. With no K-scaling the
    solver-frame inlet concentration equals ``q_root`` and ``Δθ = θ_s·Δt`` is
    carried in ``state.theta_edges``.

    Parameters
    ----------
    q_root : numpy.ndarray
        Root-zone flux per bin; equals the solver-frame inlet concentration.
    state : FrontTrackerState
        Solver state holding the wave list, θ-edges, and sorption model.
    n_grid : int, optional
        Number of θ-grid points for the trapezoidal breakthrough integral.
        Default 600. The integrand is first-order across shock fronts, so the
        grid truncation dominates the residual.

    Returns
    -------
    float
        Relative conservation error.
    """
    theta_edges = np.asarray(state.theta_edges, dtype=float)
    theta_end = float(theta_edges[-1])

    theta_grid = np.linspace(0.0, theta_end, n_grid)
    breakthrough = compute_breakthrough_curve(theta_grid, _V_OUTLET, state.waves, state.sorption)
    mass_out = float(np.trapezoid(breakthrough, theta_grid))

    mass_domain = compute_domain_mass(theta=theta_end, v_outlet=_V_OUTLET, waves=state.waves, sorption=state.sorption)
    mass_in = float(np.sum(q_root * np.diff(theta_edges)))

    return abs(mass_out + mass_domain - mass_in) / mass_in


class TestWettingShockIntoDrainedFan:
    """Issue #222: wetting shock colliding with a fully-drained (c=0) fan."""

    def test_wetting_shock_into_drained_fan_runs(self):
        """The issue's exact wet→zero→wet forcing runs without raising.

        On baseline this raises ``ValueError: c_decay_initial must be positive``
        from ``DecayingShockWave.__post_init__`` when the wetting shock collides
        with the fully drained fan tail (c_decay_initial == 0). The floor fixes
        it; this is the regression guard.
        """
        q_root = np.array([0.003] * 30 + [0.0] * 30 + [0.003] * 60)
        q_wt, _ = _run_percolation(q_root)
        # The solver completed and produced a finite, physically bounded output.
        assert np.all(np.isfinite(q_wt))
        assert np.nanmin(q_wt) >= 0.0
        np.testing.assert_array_less(q_wt, 0.003 + 1e-9)

    def test_wetting_shock_into_drained_fan_conserves_mass(self):
        """Equal-level (#222 exact input) conserves mass to a first-order grid tolerance.

        Independent breakthrough integral vs ``Σ cin·Δθ − M_domain``; NOT the
        ``compute_bin_averaged_concentration_exact`` (``m_in − m_dom``) route.
        """
        q_root = np.array([0.003] * 30 + [0.0] * 30 + [0.003] * 60)
        _, state = _run_percolation(q_root)
        rel_err = _independent_conservation_rel_err(q_root, state)
        # ~1.5e-3: first-order trapezoidal truncation across the breakthrough shock
        # fronts plus the numerical DecayingShockWave (quad+brentq) floor. An
        # independent RK4 reference confirms the underlying solution conserves to
        # ~0.17%; refining the grid further is dominated by the numerical-DSW cost.
        assert rel_err < 2e-3, f"mass not conserved: relative error {rel_err:.3e}"

    def test_unequal_wetting_shock_into_drained_fan_runs_and_conserves(self):
        """Unequal-wet variant (post-gap flux differs) runs and conserves mass.

        ``[0.003]*30 + [0.0]*30 + [0.005]*60``: the trailing wet plateau is at a
        different level from the leading one, so the collision orientation and
        fan-exhaustion guard are exercised in the growing decay regime.
        """
        q_root = np.array([0.003] * 30 + [0.0] * 30 + [0.005] * 60)
        q_wt, state = _run_percolation(q_root)
        assert np.all(np.isfinite(q_wt))
        assert np.nanmin(q_wt) >= 0.0
        rel_err = _independent_conservation_rel_err(q_root, state)
        assert rel_err < 1e-3, f"mass not conserved: relative error {rel_err:.3e}"


@pytest.mark.parametrize(
    "c_decay_initial",
    [0.0, -1e-30],
)
def test_decaying_shock_wave_floors_zero_decay_initial(c_decay_initial):
    """A drained collision value of ``0`` is floored to ``_C_MIN``, not rejected.

    Directly constructs the wave that the collision handler builds for the
    #222 case (Brooks-Corey, ``c_fixed > 0``, fully drained tail) and asserts
    the floor leaves ``c_decay_initial`` finite and positive and routes through
    the numerical DSW path (closed-form ``K`` stays NaN).
    """
    sorption = BrooksCoreyConductivity(**_BC_KWARGS)
    if c_decay_initial < 0.0:
        with pytest.raises(ValueError, match="c_decay_initial must be non-negative"):
            DecayingShockWave(
                theta_start=24.0,
                v_start=0.01,
                c_decay_initial=c_decay_initial,
                c_fixed=0.003,
                c_fan_tail=0.003,
                decay_side="right",
                v_origin=0.0,
                theta_origin=12.0,
                sorption=sorption,
            )
        return

    dsw = DecayingShockWave(
        theta_start=24.0,
        v_start=0.01,
        c_decay_initial=c_decay_initial,
        c_fixed=0.003,
        c_fan_tail=0.003,
        decay_side="right",
        v_origin=0.0,
        theta_origin=12.0,
        sorption=sorption,
    )
    assert dsw.c_decay_initial == _C_MIN
    # c_fixed > 0 with Brooks-Corey routes to the numerical solver, not a closed form.
    assert np.isnan(dsw.K)
    # Full drying (c_fan_tail == c_fixed) merges asymptotically: no finite exhaustion.
    assert dsw.theta_at_fan_exhaustion() is None
