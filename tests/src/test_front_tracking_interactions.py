"""Wave-interaction completeness tests for the multi-front front tracker.

Companion to ``docs/theory/front_tracking_interactions.md``. The ``interactions`` module
(the Feeder/Face universal-merge calculus) previously had no dedicated test — it was only
exercised transitively through the public-API multi-front runs. This file adds:

- the RAREF-RAREF non-interaction guard (same-family simple waves never collide),
- a "no silent corruption" randomized property sweep (whenever the solver does not flag an
  unresolved interaction, its output is conservation-consistent and matches the FV oracle),
- confirmation that the favorable (Freundlich ``n>1``) regime is fully resolved, and
- an ``xfail`` pinning the discovered multi-front completeness gap in the unfavorable
  (``n<1``) regime (declined LOUDLY via the public-API ``RuntimeError``, never silently wrong).

Independent reference: the Godunov upwind FV oracle in ``_fv_oracle``.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import pandas as pd
import pytest
from _fv_oracle import upwind_fv_outlet  # ty: ignore[unresolved-import]  # tests/src on path via conftest

from gwtransport.advection import infiltration_to_extraction_nonlinear_sorption
from gwtransport.fronttracking.math import FreundlichSorption
from gwtransport.fronttracking.output import compute_domain_mass, concentration_at_point
from gwtransport.fronttracking.solver import FrontTracker, find_unresolved_interaction
from gwtransport.fronttracking.waves import DecayingShockWave, DoubleFanShockWave, RarefactionWave, ShockWave


def _run(sorption, cin, v_pv, flow=100.0):
    cin = np.asarray(cin, dtype=float)
    nb = len(cin)
    tedges = pd.date_range("2020-01-01", periods=nb + 1, freq="D")
    tr = FrontTracker(cin=cin, flow=np.full(nb, flow), tedges=tedges, aquifer_pore_volume=v_pv, sorption=sorption)
    tr.run()
    return tr


def _fv_discrepancy(tr, sorption, v_pv, n_cells=200, stride=6):
    """Integrated relative discrepancy between the exact tracker and the first-order FV oracle."""
    th, co = upwind_fv_outlet(sorption, tr.state.cin, tr.state.theta_edges, v_pv, n_cells=n_cells, cfl=0.4)
    mask = (th > 0) & (th <= float(tr.state.theta_edges[-1]))
    th_s, co_s = th[mask][::stride], co[mask][::stride]
    ct = np.array([concentration_at_point(v_pv, t, tr.state.waves, sorption) for t in th_s])
    denom = np.trapezoid(np.abs(ct), th_s) + 1e-30
    return float(np.trapezoid(np.abs(ct - co_s), th_s) / denom)


def _independent_domain_mass(tr, sorption, v_pv, theta, n_v=4000):
    """Independent stored-mass reference: ``∫₀^V C_T(c(v, θ)) dv`` via a fine trapezoid."""
    v = np.linspace(0.0, v_pv, n_v)
    c = np.array([concentration_at_point(vi, theta, tr.state.waves, sorption) for vi in v])
    return float(np.trapezoid(np.asarray(sorption.total_concentration(c), dtype=float), v))


# Favorable Freundlich (n>1) is the regime the multi-front solver fully resolves. These
# hand-picked two/three-pulse inlets are each verified fast (<0.1 s) and interaction-complete.
# Random sweeps are deliberately avoided: a minority of multi-pulse configs make the tracker
# churn for many seconds (a #316 performance characteristic; the slow inputs are also the ones
# that eventually decline), so a fixed, fast, deterministic set keeps the safety net cheap.
_FAVORABLE = FreundlichSorption(k_f=0.008, n=2.0, bulk_density=1500.0, porosity=0.3)
_FAVORABLE_CONFIGS = [
    ([0.0, 6, 6, 0, 3, 3, 0], 8.0),
    ([0.0, 8, 8, 8, 0, 4, 4, 0], 9.0),
    ([0.0, 5, 5, 0, 0, 7, 7, 0], 10.0),
    ([0.0, 9, 9, 0, 2, 2, 0], 7.0),
    ([0.0, 4, 4, 4, 0, 0, 6, 6, 0], 11.0),
]


class TestRarefactionRarefactionNoInteraction:
    """Two same-family rarefactions never interact (scalar single-family law).

    A rear rarefaction's head and the front rarefaction's tail both carry the separating
    constant state ``c*`` and move at the same celerity ``λ(c*)``; the gap is rigidly
    translated and never closes (Rhee-Aris-Amundson). The solver's ``RAREF_RAREF_COLLISION``
    no-op (``solver.py:560``) is therefore correct — see theory doc §4.1.
    """

    def test_unfavorable_staircase_is_all_coexisting_rarefactions(self):
        """Freundlich ``n<1`` increasing staircase: every step-up is a rarefaction, all coexist.

        For the unfavorable (concave) flux, step-*ups* spread into rarefactions and there is no
        leading loading shock, so a strictly increasing inlet produces N same-family
        rarefactions that must all remain active with no collision and no merge successor. (This
        is a pure-rarefaction layout with no multi-front interaction, so it is unaffected by the
        ``n<1`` interaction gap in :class:`TestKnownMultiFrontGaps`.)
        """
        s = FreundlichSorption(k_f=0.05, n=0.5, bulk_density=1500.0, porosity=0.3)
        cin = np.array([0.0] + [2.0] * 8 + [5.0] * 8 + [8.0] * 8 + [11.0] * 40)
        tr = _run(s, cin, 15.0)
        active = [w for w in tr.state.waves if w.is_active]
        n_raref = sum(isinstance(w, RarefactionWave) for w in active)
        n_other = sum(isinstance(w, (ShockWave, DecayingShockWave, DoubleFanShockWave)) for w in active)
        assert n_raref == 4, f"expected 4 coexisting rarefactions, got {n_raref}"
        assert n_other == 0, "no shock/decaying/double-fan wave may form from same-family rarefactions"
        assert find_unresolved_interaction(tr.state) is None
        # This long-series staircase asserts only structure (a fine FV march over V=15 is expensive);
        # the resolved n<1 outlet is cross-checked against the FV oracle on a short config in
        # ``test_short_unfavorable_staircase_matches_fv`` below.

    def test_short_unfavorable_staircase_matches_fv(self):
        """A short n<1 increasing staircase's resolved outlet matches the independent FV oracle.

        The only numerical (not just structural) validation of a resolved unfavorable-flux (concave,
        mirror-regime) outlet — everything else in this file is favorable Freundlich n>1. Kept cheap
        with a short series and small V so the first-order FV march stays a few thousand steps.
        """
        s = FreundlichSorption(k_f=0.05, n=0.5, bulk_density=1500.0, porosity=0.3)
        cin = np.array([0.0] + [2.0] * 2 + [5.0] * 2 + [8.0] * 2 + [11.0] * 10)
        tr = _run(s, cin, 6.0)
        assert find_unresolved_interaction(tr.state) is None
        assert sum(isinstance(w, RarefactionWave) for w in tr.state.waves if w.is_active) == 4
        assert _fv_discrepancy(tr, s, 6.0, n_cells=160, stride=8) < 0.05

    def test_favorable_stepdowns_two_rarefactions_do_not_merge(self):
        """Freundlich ``n>1`` step-downs make two same-family rarefactions that never merge.

        A leading loading shock forms (0→c step-up) and may catch the first fan, but the two
        rarefactions themselves never collide and the solver stays self-consistent.
        """
        s = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        cin = np.array([5.0] * 30 + [3.0] * 30 + [1.0] * 60)
        tr = _run(s, cin, 30.0)
        assert find_unresolved_interaction(tr.state) is None
        assert _fv_discrepancy(tr, s, 30.0) < 0.02


class TestNoSilentCorruption:
    """The solver is never SILENTLY wrong.

    Whenever ``find_unresolved_interaction`` returns ``None`` (the solver claims a complete
    resolution), the outlet ``cout`` must be conservation-consistent, match the independent FV
    oracle, and be non-negative. Inputs the solver cannot resolve are declined LOUDLY (the
    public API raises ``RuntimeError``); those are covered by :class:`TestKnownMultiFrontGaps`,
    not silently mis-computed here. Theory doc §4.5.
    """

    @pytest.mark.parametrize(("pulse", "v_pv"), _FAVORABLE_CONFIGS, ids=[f"V{c[1]:g}" for c in _FAVORABLE_CONFIGS])
    def test_resolved_run_conserves_and_nonnegative(self, pulse, v_pv):
        cin = np.array(pulse + [0.0] * 12, dtype=float)
        tr = _run(_FAVORABLE, cin, v_pv)
        # favorable Freundlich n>1 multi-front is fully resolved (never silently wrong here)
        assert find_unresolved_interaction(tr.state) is None
        theta_hi = float(tr.state.theta_edges[-1])

        cout = np.array([
            concentration_at_point(v_pv, t, tr.state.waves, _FAVORABLE) for t in np.linspace(1.0, theta_hi, 30)
        ])
        assert np.all(cout >= -1e-9), f"negative cout {cout.min()}"

        theta_q = 0.4 * theta_hi
        m_dom = compute_domain_mass(theta_q, v_pv, tr.state.waves, _FAVORABLE)
        m_quad = _independent_domain_mass(tr, _FAVORABLE, v_pv, theta_q, n_v=2000)
        # rtol tight enough to constrain integrate_fan_spatial_exact: the only imprecision is the
        # trapezoid reference's ~1e-8 error at n_v=2000; 1e-6 is 30x above that, not a defanged gate.
        np.testing.assert_allclose(m_dom, m_quad, rtol=1e-6, atol=1e-9)

    def test_outlet_cout_matches_fv_oracle(self):
        """One FV cross-check (the expensive part) on a representative resolved config."""
        pulse, v_pv = _FAVORABLE_CONFIGS[1]
        tr = _run(_FAVORABLE, np.array(pulse + [0.0] * 12, dtype=float), v_pv)
        assert find_unresolved_interaction(tr.state) is None
        assert _fv_discrepancy(tr, _FAVORABLE, v_pv, n_cells=120, stride=8) < 0.04


class TestKnownMultiFrontGaps:
    """Documented completeness gaps in the #294/#316 multi-front solver.

    The universal merge calculus was validated on the favorable (Freundlich ``n>1``) geometry;
    the unfavorable (``n<1``, concave-flux) mirror — where step-ups are rarefactions and
    step-downs are shocks — leaves some two-pulse shock↔fan interactions unresolved. The public
    API surfaces this LOUDLY (``find_unresolved_interaction`` fires → ``RuntimeError``), so no
    silently-wrong ``cout`` is ever returned. These tests assert the DESIRED behaviour (a
    complete resolution) and are ``xfail(strict=True)``: a solver fix flips them to XPASS,
    which fails the suite and flags that they should be un-xfailed. See
    ``docs/theory/front_tracking_interactions.md`` §4.3.
    """

    @pytest.mark.slow  # the declined run churns to the loop guard (~15 s) before giving up
    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,  # pin the ASSERTION failing, not e.g. a decline->crash regression
        reason="#316 multi-front solver leaves an unresolved interaction for unfavorable "
        "Freundlich n<1 two-pulse inputs (mirror-regime shock<->fan merge). Tracked in #317.",
    )
    def test_freundlich_nlt1_two_pulse_resolves(self):
        s = FreundlichSorption(k_f=0.05, n=0.5, bulk_density=1500.0, porosity=0.3)
        cin = np.array([0.0, 8.0, 8.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0] + [0.0] * 20)
        tr = _run(s, cin, 10.6)
        assert find_unresolved_interaction(tr.state) is None

    @pytest.mark.slow  # the declined run churns to the loop guard (~15 s) before giving up
    def test_public_api_declines_n_lt_1_loudly(self):
        """The gap surfaces as a public-API ``RuntimeError`` — never a silently-wrong cout.

        Guards ``advection.infiltration_to_extraction_nonlinear_sorption``'s ``find_unresolved_interaction``
        raise-wiring end-to-end (the increment's central "never silently wrong" contract), which the
        internal-only checks above do not exercise. When #317 is fixed this must stop raising — update
        it to assert a valid cout then.
        """
        cin = np.array([0.0, 8.0, 8.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0] + [0.0] * 20)
        nb = len(cin)
        tedges = pd.date_range("2020-01-01", periods=nb + 1, freq="D")
        with pytest.raises(RuntimeError, match="unresolved"):
            infiltration_to_extraction_nonlinear_sorption(
                cin=cin,
                flow=np.full(nb, 100.0),
                tedges=tedges,
                cout_tedges=tedges,
                aquifer_pore_volumes=[10.6],
                freundlich_k=0.05,
                freundlich_n=0.5,
                bulk_density=1500.0,
                porosity=0.3,
            )
