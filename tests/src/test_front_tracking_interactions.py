"""Wave-interaction completeness tests for the multi-front front tracker.

Companion to ``docs/theory/front_tracking_interactions.md``. The ``interactions`` module
(the Feeder/Face universal-merge calculus) previously had no dedicated test — it was only
exercised transitively through the public-API multi-front runs. This file adds:

- the RAREF-RAREF non-interaction *outcome* (same-family simple waves coexist without merging),
- a "no silent corruption" property check over a fixed set of favorable multi-front inputs
  (whenever the solver does not flag an unresolved interaction, its output is
  conservation-consistent and matches the FV oracle),
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
from gwtransport.fronttracking.math import (
    BrooksCoreyConductivity,
    ConstantRetardation,
    FreundlichSorption,
    LangmuirSorption,
)
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


def test_fv_oracle_matches_analytic_contact():
    """Known-answer self-check for the FV oracle: a constant-retardation contact.

    Validates ``_fv_oracle.upwind_fv_outlet`` itself — the reference every tracker-vs-oracle
    cross-check in this file leans on. For ``ConstantRetardation(R)`` a single ``0→c0`` inlet is a
    contact discontinuity: it breaks through at ``θ = R·V`` and the outlet plateaus at ``c0``.
    Catches a silent oracle regression (CFL/step-count, the ``U→c`` inversion, the ``max(·, 0)``
    clamp) that would otherwise degrade all FV cross-checks without any localized failure.
    """
    r, c0, v_out = 2.0, 3.0, 5.0
    sorption = ConstantRetardation(retardation_factor=r)
    theta_edges = np.linspace(0.0, 25.0, 6)  # θ_end = 25 > R·V = 10 so breakthrough is captured
    th, co = upwind_fv_outlet(sorption, np.full(5, c0), theta_edges, v_out, n_cells=200, cfl=0.4)
    half_rise = float(th[np.searchsorted(co, c0 / 2.0)])
    # first-order front smearing sets the breakthrough resolution to ~ΔV/V = 1/n_cells
    np.testing.assert_allclose(half_rise, r * v_out, rtol=0.02)
    np.testing.assert_allclose(co[-1], c0, rtol=1e-9)  # plateau to the inlet value


class TestRarefactionRarefactionNoInteraction:
    """Two same-family rarefactions never interact (scalar single-family law).

    A rear rarefaction's head and the front rarefaction's tail both carry the separating
    constant state ``c*`` and move at the same celerity ``λ(c*)``; the gap is rigidly
    translated and never closes (Rhee-Aris-Amundson). These tests verify that *outcome* —
    same-family rarefactions coexist and never spawn a merge/shock successor between them —
    not the solver's ``RAREF_RAREF_COLLISION`` handler itself, which for physical inputs is
    never reached (no ``rarefaction_rarefaction`` event is ever emitted), consistent with its
    documented no-op status (see theory doc §4.1).
    """

    def test_unfavorable_staircase_is_all_coexisting_rarefactions(self):
        """Freundlich ``n<1`` increasing staircase: every step-up is a rarefaction, all coexist.

        For the unfavorable (concave) flux, step-*ups* spread into rarefactions and there is no
        leading loading shock, so a strictly increasing inlet produces N same-family
        rarefactions that must all remain active with no collision and no merge successor. (This
        is a pure-rarefaction layout with no multi-front interaction, distinct from the resolved
        ``n<1`` shock↔fan side exhaustion in :class:`TestSideExhaustion`.)
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

    @pytest.mark.slow  # a fine FV-oracle cross-check: ~1e5 first-order upwind steps (~40 s serial)
    def test_short_unfavorable_staircase_matches_fv(self):
        """A short n<1 increasing staircase's resolved outlet matches the independent FV oracle.

        The only numerical (not just structural) validation of a resolved unfavorable-flux (concave,
        mirror-regime) outlet — everything else in this file is favorable Freundlich n>1. Even with a
        short series and small V, the first-order FV march is ~1e5 steps (the CFL step scales with
        ``V/n_cells`` while ``θ_end`` is fixed by the flow), so this is a slow independent cross-check.
        """
        s = FreundlichSorption(k_f=0.05, n=0.5, bulk_density=1500.0, porosity=0.3)
        cin = np.array([0.0] + [2.0] * 2 + [5.0] * 2 + [8.0] * 2 + [11.0] * 10)
        tr = _run(s, cin, 6.0)
        assert find_unresolved_interaction(tr.state) is None
        assert sum(isinstance(w, RarefactionWave) for w in tr.state.waves if w.is_active) == 4
        assert _fv_discrepancy(tr, s, 6.0, n_cells=160, stride=8) < 0.05

    def test_favorable_stepdowns_two_rarefactions_do_not_merge(self):
        """Freundlich ``n>1`` step-downs make two same-family rarefactions that never merge.

        The inlet ``0→5→3→1`` forms a leading loading shock (``0→5`` step-up) followed by two
        step-down rarefactions (fan1 ``5→3``, fan2 ``3→1``). The leading shock catches and
        absorbs fan1 (a shock↔rarefaction interaction, not a rarefaction↔rarefaction one), so
        exactly one rarefaction (fan2) survives — the two fans never merge *with each other*.
        The surviving-rarefaction count and the FV cross-check pin that structure.
        """
        s = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        cin = np.array([5.0] * 30 + [3.0] * 30 + [1.0] * 60)
        tr = _run(s, cin, 30.0)
        assert find_unresolved_interaction(tr.state) is None
        active = [w for w in tr.state.waves if w.is_active]
        # fan2 survives; fan1 is absorbed by the leading loading shock (never merged into fan2).
        assert sum(isinstance(w, RarefactionWave) for w in active) == 1
        assert not any(isinstance(w, DoubleFanShockWave) for w in active)
        assert _fv_discrepancy(tr, s, 30.0) < 0.02


class TestNoSilentCorruption:
    """The solver is never SILENTLY wrong.

    Whenever ``find_unresolved_interaction`` returns ``None`` (the solver claims a complete
    resolution), the outlet ``cout`` must be conservation-consistent, match the independent FV
    oracle, and be non-negative. Any input the solver still cannot resolve is declined LOUDLY
    (the public API raises ``RuntimeError``) rather than silently mis-computed here; the
    formerly-declining ``n<1``/Langmuir side-exhaustion inputs are now resolved and covered by
    :class:`TestSideExhaustion`. Theory doc §4.
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

    @pytest.mark.slow  # Langmuir multi-front resolution (~7 s) plus a ~5e3-step FV cross-check
    def test_langmuir_resolved_run_conserves_and_matches_fv(self):
        """Extend the "never silently wrong" net to a second isotherm (Langmuir).

        The universal Feeder/Face merge calculus is claimed (theory doc §5) to resolve *every*
        ``NonlinearSorption``, but every other resolved-run check here is Freundlich. A favorable
        Langmuir two-pulse input is either fully resolved — mass-conserving and FV-consistent — or
        declined loudly; this pins the resolved branch for Langmuir specifically.
        """
        s = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        v_pv = 8.0
        tr = _run(s, np.array([0.0, 6, 6, 0, 3, 3, 0] + [0.0] * 12, dtype=float), v_pv)
        assert find_unresolved_interaction(tr.state) is None
        theta_hi = float(tr.state.theta_edges[-1])

        cout = np.array([concentration_at_point(v_pv, t, tr.state.waves, s) for t in np.linspace(1.0, theta_hi, 30)])
        assert np.all(cout >= -1e-9), f"negative cout {cout.min()}"

        theta_q = 0.4 * theta_hi
        m_dom = compute_domain_mass(theta_q, v_pv, tr.state.waves, s)
        # Langmuir's steeper fan makes the trapezoid reference converge more slowly than Freundlich's
        # (rel ~9e-5 at n_v=6000, shrinking toward m_dom as n_v grows), so this gate is reference-error
        # limited — not a defanged solver gate. m_dom (integrate_fan_spatial_exact) is exact.
        m_quad = _independent_domain_mass(tr, s, v_pv, theta_q, n_v=6000)
        np.testing.assert_allclose(m_dom, m_quad, rtol=3e-4, atol=1e-9)
        # First-order FV diffusion runs higher for Langmuir's steeper fan than for Freundlich n>1.
        assert _fv_discrepancy(tr, s, v_pv, n_cells=200, stride=8) < 0.06

    def test_brooks_corey_resolved_run_conserves_and_nonnegative(self):
        """Extend the resolved-branch conservation net to a third isotherm (Brooks-Corey).

        A favorable Brooks-Corey two-pulse input is fully resolved (``find_unresolved`` None) and its
        stored mass from ``integrate_fan_spatial_exact`` matches an independent pointwise trapezoid,
        exercising Brooks-Corey's fan quadrature / merge calculus (theory doc §5's "every
        ``NonlinearSorption``"). The FV-independent cross-check for a nonlinear isotherm is carried by
        the Langmuir test above; Brooks-Corey's FV march is ~1e5 steps, too slow to repeat here.
        """
        s = BrooksCoreyConductivity(theta_r=0.045, theta_s=0.43, k_s=10.0, brooks_corey_lambda=0.5)
        v_pv = 8.0
        tr = _run(s, np.array([0.0, 6, 6, 0, 3, 3, 0] + [0.0] * 12, dtype=float), v_pv)
        assert find_unresolved_interaction(tr.state) is None
        theta_hi = float(tr.state.theta_edges[-1])

        cout = np.array([concentration_at_point(v_pv, t, tr.state.waves, s) for t in np.linspace(1.0, theta_hi, 30)])
        assert np.all(cout >= -1e-9), f"negative cout {cout.min()}"

        theta_q = 0.4 * theta_hi
        m_dom = compute_domain_mass(theta_q, v_pv, tr.state.waves, s)
        m_quad = _independent_domain_mass(tr, s, v_pv, theta_q, n_v=6000)
        np.testing.assert_allclose(m_dom, m_quad, rtol=1e-4, atol=1e-9)


class TestSideExhaustion:
    """Regression guards for the resolved #317 multi-front gap: doubly-fed side exhaustion.

    A :class:`DoubleFanShockWave` side ends in finite θ exactly when its shock face crosses
    one of its own fan boundary lines — the left fan's slow upstream characteristic catching
    the shock from behind (the ``n<1`` mirror, where fan characteristics outrun the shock),
    or the shock catching the right fan's downstream characteristic. Lax entropy
    (``λ(c_L) ≥ σ ≥ λ(c_R)``) forbids any other finite-θ side ending. The solver resolves
    these same-wave face crossings through the universal ``WAVE_MERGE`` machinery: the merge
    of the boundary line with the shock face builds the degraded successor
    (``const far-bound | surviving fan``) and retires the boundary.

    Before the fix, a special-case detector missed these crossings (it targeted the fan edge
    farther from the current side value — wrong whenever the wave is born mid-fan — and the
    feeder clamp flattened its residual to exactly zero, defeating the bracket), so the
    exhausted fan's boundary line stayed exposed BEYOND the shock: a ghost face the reader
    sweep used to misattribute the region between shock and line, breaking outlet-mass
    monotonicity — the loud decline of issue #317 (Freundlich ``n<1`` two-pulse and several
    favorable Langmuir multi-pulse inputs). See
    ``docs/theory/front_tracking_interactions.md`` §4.3.
    """

    @pytest.mark.slow  # a fine FV-oracle cross-check over θ_end=2900 (~6 s)
    def test_freundlich_nlt1_two_pulse_resolves(self):
        """The #317 minimal reproducer resolves, with the exhaustion pinned to its closed form.

        For Freundlich exponent 2 (``n=0.5``) the doubly-fed secant collapses to the harmonic
        mean of the two shared-apex fan rays, ``σ = 2V/((θ−θ_L)+(θ−θ_R))``, which integrates
        to the linear-fractional trajectory ``V = (θ − (θ_L+θ_R)/2)/2401`` through the birth
        point ``(725.15625, 5/32)`` (``θ_L=600``, ``θ_R=100``; ``2401 = 375.15625/(5/32)``).
        Crossing the left fan's ``c=2`` characteristic ``V = (θ−600)/1001`` (``R(2)=1001``)
        gives the side exhaustion at exactly ``θ = 778.75``, ``V = 5/28``. The solver's
        RK4-spline trajectory plus the face-march/brentq reproduce the anchor to a relative
        error of ~1e-9 (θ) and ~4e-9 (V) — both deterministic; the asserted rtols sit ~10-20×
        above those floors.
        """
        s = FreundlichSorption(k_f=0.05, n=0.5, bulk_density=1500.0, porosity=0.3)
        cin = np.array([0.0, 8.0, 8.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0] + [0.0] * 20)
        tr = _run(s, cin, 10.6)
        assert find_unresolved_interaction(tr.state) is None

        # The side-exhaustion event: a same-wave merge (the DFSW's own boundary line × shock).
        self_merges = [
            ev
            for ev in tr.state.events
            if ev["type"] == "wave_merge" and ev["waves_before"][0] is ev["waves_before"][1]
        ]
        assert len(self_merges) == 1
        ev = self_merges[0]
        np.testing.assert_allclose(ev["theta"], 778.75, rtol=1e-8)
        np.testing.assert_allclose(ev["location"], 5.0 / 28.0, rtol=1e-7)
        (successor,) = ev["waves_after"]
        assert isinstance(successor, DecayingShockWave)
        assert successor.c_fixed == 2.0  # the exhausted left side froze at the fan's far bound
        assert successor.decay_side == "right"
        dfsw = ev["waves_before"][0]
        assert not dfsw.is_active
        np.testing.assert_allclose(dfsw.theta_deactivation, ev["theta"], rtol=0, atol=0)
        np.testing.assert_allclose(dfsw.theta_left_boundary_consumed, ev["theta"], rtol=0, atol=0)

        # Independent numerical gate: the resolved outlet matches the first-order FV oracle
        # (0.05 = the oracle's own smearing floor at n_cells=160, cf. the staircase test above).
        assert _fv_discrepancy(tr, s, 10.6, n_cells=160, stride=8) < 0.05

    def test_public_api_resolves_n_lt_1(self):
        """The public API returns a physical ``cout`` for the former #317 decline input.

        Guards ``advection.infiltration_to_extraction_nonlinear_sorption`` end-to-end on the
        input that used to raise ``RuntimeError`` (the loud decline): the returned bin averages
        must be finite, non-negative, and the implied stored mass must be correct.

        The stored-mass check must NOT be the closed-system balance ``m_out + m_dom == m_in``:
        the public ``cout`` is *defined* per bin as ``Δ(m_in − m_dom)/Δθ`` (see
        ``compute_bin_averaged_concentration_exact``), so that balance collapses to
        ``m_in == m_in`` and passes even with a badly wrong ``compute_domain_mass`` (a uniform
        50% ``m_dom`` error still balances to the FP floor — mutation-confirmed). Instead validate
        ``compute_domain_mass`` against a genuinely *independent* reference — a fine trapezoid of
        ``C_T(c(v, θ_end))`` sampled over ``[0, V]`` (a different integration than the analytic
        fan IBP), which then pins ``m_out = m_in − m_dom``. (The resolved outlet field itself is
        FV-oracle-gated in ``test_freundlich_nlt1_two_pulse_resolves``.)
        """
        cin = np.array([0.0, 8.0, 8.0, 8.0, 0.0, 0.0, 2.0, 2.0, 0.0] + [0.0] * 20)
        nb = len(cin)
        tedges = pd.date_range("2020-01-01", periods=nb + 1, freq="D")
        cout, structures = infiltration_to_extraction_nonlinear_sorption(
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
        cout = np.asarray(cout, dtype=float).ravel()
        # Out-of-window bins are returned as 0.0; genuinely zero-throughflow bins as NaN.
        # This all-positive-flow run has neither, so every bin is finite and non-negative.
        assert np.all(np.isfinite(cout))
        assert np.all(cout >= -1e-9)
        state, s = structures[0]["tracker_state"], structures[0]["sorption"]
        theta_end = float(state.theta_edges[-1])
        m_in = float(np.sum(cin * 100.0))
        m_out = float(np.sum(cout * 100.0))
        assert m_out > 0.0
        # Independent stored-mass reference: trapezoid of C_T over [0, V] (samples the field via
        # concentration_at_point, then integrates numerically) vs the analytic fan integral in
        # compute_domain_mass. The two integrations agree and CONVERGE (rel 3.4e-4 → 7.3e-5 → 1.9e-5
        # at n_v 4e3 → 2e4 → 1e5), so compute_domain_mass is the trapezoid's limit; rtol 1e-3 sits
        # ~3× above the n_v=4000 discretization floor, yet is ~400× tighter than the ~40% gap a
        # uniform compute_domain_mass error opens — so this is NOT the m_in−m_dom tautology
        # (mutation-confirmed). This pins the public cout, since cout ≡ (m_in − m_dom)/Δθ.
        m_domain = compute_domain_mass(theta_end, 10.6, state.waves, s)
        v = np.linspace(0.0, 10.6, 4000)
        c_field = np.array([concentration_at_point(vi, theta_end, state.waves, s) for vi in v])
        m_domain_indep = float(np.trapezoid(np.asarray(s.total_concentration(c_field), dtype=float), v))
        np.testing.assert_allclose(m_domain, m_domain_indep, rtol=1e-3)
        np.testing.assert_allclose(m_out, m_in - m_domain, rtol=1e-9)

    @pytest.mark.parametrize(
        ("cin", "v_pv"),
        [
            # Both former decline configs (found by a seeded random sweep on the unfixed
            # solver; each declined loudly there). The first exercises the born-mid-fan
            # geometry (the wrong-target defect); the second is the clamp-flatline-only
            # fingerprint (its detector targeted the correct edge and still never fired).
            ([0.0, 9, 9, 0, 0, 2.6, 2.6, 0, 8.8, 0, 0, 8.3], 8.8),
            ([0.0, 3.3, 0.5, 0.5, 8.3, 6.1, 6.1, 5, 5, 0.3, 0.3], 11.6),
        ],
        ids=["multi-pulse", "staircase"],
    )
    def test_langmuir_multipulse_resolves(self, cin, v_pv):
        """Favorable Langmuir multi-pulse inputs that formerly declined now resolve.

        Side exhaustion is geometry-dependent, not regime-dependent: these favorable-regime
        configs hit the same missed self-crossing. Each run must resolve completely and must
        actually exercise the same-wave merge path (at least one such event fires).
        """
        s = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
        tr = _run(s, np.array(cin + [0.0] * 15, dtype=float), v_pv)
        assert find_unresolved_interaction(tr.state) is None
        assert any(
            ev["type"] == "wave_merge" and ev["waves_before"][0] is ev["waves_before"][1] for ev in tr.state.events
        )
