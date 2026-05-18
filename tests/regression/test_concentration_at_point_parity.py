"""Regression: ``concentration_at_point`` honors ``was_active_at`` retrospectively
and dispatches stacked shocks by V_shock, not by theta_start.

Round 2 introduced a structural asymmetry: ``compute_domain_mass`` switched to
``was_active_at(theta)`` for retrospective queries, but ``concentration_at_point``
still filtered by ``wave.is_active``. After the canonical n=2 pulse, the leading
shock is deactivated at θ=2580 (collision with the trailing rarefaction tail).
Retrospective queries at θ ∈ [θ_shock_start, θ_deactivation] for any v left of
the shock should return ``c_left=4`` — but pre-fix returned 0 because the shock
was skipped by the ``is_active`` filter.

The bug surfaces through ``compute_domain_mass`` (which calls
``concentration_at_point`` in its constant-region fallback): m_dom drops to 0
for θ < θ_first_outlet_crossing, and the conservation form
``m_out = m_in - m_dom`` then echoes the inlet directly through
``compute_bin_averaged_concentration_exact`` in the production advection path.

Round 4 fix: ``concentration_at_point``'s elif-branch tiebreaker for ``v > v_shock``
(shock not yet passed v) switched from ``wave.theta_start`` to a separate
``rightmost_passed_v_shock`` tracker. The pre-fix tiebreaker mis-ranked stacked
shocks (the youngest shock by theta_start = innermost in V, ≠ closest to v from
the left). A c=[0,3,6,9,12] ramp through advection produced a 777 spike in
cout[4] pre-fix.
"""

import numpy as np
import pandas as pd

from gwtransport.fronttracking.math import FreundlichSorption
from gwtransport.fronttracking.output import (
    compute_cumulative_inlet_mass,
    compute_domain_mass,
    concentration_at_point,
)
from gwtransport.fronttracking.solver import FrontTracker


def _build_canonical_n2_pulse_tracker():
    """Canonical Freundlich n=2 pulse: c=4 over 10 days, v_outlet=200, flow=100."""
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    cin = np.zeros(500)
    cin[5:15] = 4.0
    flow = np.full(500, 100.0)
    tedges = pd.date_range("2020-01-01", periods=501, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)
    return tr, cin, v_outlet, sorption


def test_concentration_at_point_picks_up_deactivated_shock_retrospectively():
    """At θ=550 (pre-DSW, shock still progressing toward v_outlet), the c=4
    plateau left of the shock must be returned for v < V_shock(550)."""
    tr, _cin, _v_outlet, sorption = _build_canonical_n2_pulse_tracker()
    shock = tr.state.waves[0]
    assert not shock.is_active, "canonical-pulse leading shock is deactivated post-simulation"
    assert shock.was_active_at(550.0), "shock is_active=False but was historically active at θ=550"

    v_shock_at_550 = shock.position_at_theta(550.0)
    assert v_shock_at_550 is not None

    # v left of shock → c_left = 4
    c_left = concentration_at_point(0.5 * v_shock_at_550, 550.0, tr.state.waves, sorption)
    assert c_left == 4.0, f"expected c=4 at v=0.5*V_shock (left of shock); got {c_left}"

    # v right of shock → c_right = 0 (initial condition)
    c_right = concentration_at_point(v_shock_at_550 * 2.0 + 1.0, 550.0, tr.state.waves, sorption)
    assert c_right == 0.0, f"expected c=0 right of shock; got {c_right}"


def test_compute_domain_mass_matches_inlet_pre_outlet_arrival():
    """For the canonical n=2 pulse, no mass has reached v_outlet=200 by θ=750
    (shock + rarefaction collide at θ=2580, then DSW propagates to v_outlet only
    much later). At intermediate θ, m_dom(θ) ≈ m_in(θ) to machine precision.
    """
    tr, cin, v_outlet, sorption = _build_canonical_n2_pulse_tracker()
    theta_edges = np.asarray(tr.state.theta_edges, dtype=float)

    # rtol=1e-12 ≈ 10 ULPs of the sum-of-integrals path; tighter values trip on
    # FP noise (observed rel_err ≈ 1.25e-13 at θ=2000.0).
    rtol = 1e-12
    for theta in (550.0, 700.0, 1000.0, 1500.0, 2000.0):
        m_in = compute_cumulative_inlet_mass(theta=theta, cin=cin, theta_edges=theta_edges)
        m_dom = compute_domain_mass(theta=theta, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
        rel_err = abs(m_dom - m_in) / m_in
        assert rel_err < rtol, f"θ={theta}: m_in={m_in:.6f}, m_dom={m_dom:.6f}, rel_err={rel_err:.3e}"


def test_stacked_shocks_right_side_uses_v_shock_tiebreaker():
    """Round-4 fix: among stacked shocks all left of v, the c at v is
    c_right of the shock with LARGEST V_shock (closest from the left), not
    the largest theta_start (which mis-ranks innermost = deepest in stack).

    Reproduces the c=[0,3,6,9,12] ramp scenario through Freundlich n=2 at
    θ_query=250.0 — a time where multiple cascade shocks coexist before
    fully merging. Pre-fix, the elif-branch tiebreaker used ``theta_start``,
    causing the youngest (innermost in V, leftmost from the query) shock
    to win for any v right of all shocks. The ``test_api_freundlich_n_gt_1_
    n_greater_than_one`` API test surfaced this as a 777 spike in cout[4].
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    v_outlet = 500.0
    cin = np.array([0.0, 3.0, 6.0, 9.0, 12.0])
    flow = np.full(5, 100.0)
    tedges = pd.date_range("2020-01-01", periods=6, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=10000)

    # At θ_query=250 (during the second inlet bin, before pulse end):
    # the leading cascade shock (3→0) is at V ≈ 0.0335*150 = 5.0,
    # and the trailing cascade shock (6→3) is at V ≈ 0.0772*50 = 3.86.
    # So V_oldest=5.0 > V_newest=3.86: shock layout V_new=3.86 < V_old=5.0.
    # Pre-fix mutation: youngest shock (theta_start=200) wins → c_right of
    # newest = 3. Post-fix: rightmost-V shock (theta_start=100) wins →
    # c_right of oldest = 0.
    theta_query = 250.0
    v_query_right = 100.0  # well right of both shocks

    c = concentration_at_point(v=v_query_right, theta=theta_query, waves=tr.state.waves, sorption=sorption)
    # At v=100 right of all shocks at θ=250: c must be the IC value =
    # c_right of the leading (rightmost-V) shock = 0.
    # Pre-fix would have returned 3 (c_right of the second-cascade shock).
    assert c == 0.0, f"v=100 right of all shocks at θ=250: expected c=0 (IC), got {c}"


def test_multi_rarefaction_overlap_no_overcount_in_domain_mass():
    """Round-5c fix: ``concentration_at_point`` shock c_R only applies in the
    immediate-right zone, not past intervening rarefactions.

    For an n<1 ramp ``cin=[0,3,6,9,12,0]`` (Freundlich n=0.5), the simulator
    produces 4 stacked rarefactions from the upramp PLUS a closing shock
    from the trailing zero. At θ=500 (just after the closing shock forms at
    V=0), 4 active rarefactions sit in the V-range [0.083, 400] with c
    plateaus between them (c=9, 6, 3 at the gaps). Pre-fix, the shock's
    c_R=12 propagated through ALL constant regions — including [400, 500]
    where c should be 0 (IC, past the outermost rarefaction's head). The
    overcount made m_dom=727694 vs the physically-injected 3000 (240×
    overcount). Post-fix, the _intervening_wave_between obstruction check
    correctly stops the shock c_R's reach at the next downstream wave.

    This test asserts the c profile is geometrically correct across all
    intermediate-V positions of the padded n=0.5 ramp at θ=500.
    """
    sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3)
    v_outlet = 500.0
    cin = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 0.0])
    flow = np.full(6, 100.0)
    tedges = pd.DatetimeIndex(
        pd.date_range("2020-01-01", periods=6, freq="D").append(pd.DatetimeIndex([pd.Timestamp("2020-01-11")]))
    )
    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=200000)

    theta_query = 500.0

    # Probe v at the midpoints of each constant region. Expected c at each
    # midpoint corresponds to the plateau value between rarefactions.
    expectations = [
        (0.0417, 12.0),  # just past closing shock at V=0, in c=12 plateau
        (0.1665, 9.0),  # between r#3 head and r#2 tail (c=9 plateau)
        (0.4160, 6.0),  # between r#2 head and r#1 tail (c=6 plateau)
        (1.1628, 3.0),  # between r#1 head and r#0 tail (c=3 plateau)
        (450.0, 0.0),  # past all rarefactions (c=0 IC)
    ]
    for v, c_expected in expectations:
        c = concentration_at_point(v=v, theta=theta_query, waves=tr.state.waves, sorption=sorption)
        assert c == c_expected, (
            f"n=0.5 ramp at θ=500, v={v}: expected c={c_expected}, got {c} "
            f"(pre-round-5c shock c_R=12 would propagate to all constant regions)"
        )

    # Sanity: m_dom should match m_in (no mass has reached v_outlet=500 yet).
    from gwtransport.fronttracking.output import compute_cumulative_inlet_mass, compute_domain_mass

    m_in = compute_cumulative_inlet_mass(theta=theta_query, cin=cin, theta_edges=tr.state.theta_edges)
    m_dom = compute_domain_mass(theta=theta_query, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
    rel_err = abs(m_dom - m_in) / max(m_in, 1.0)
    # Pre-round-5c: m_dom=727694, rel_err ≈ 240. Post-fix: rel_err ≤ 1e-12.
    assert rel_err < 1e-12, f"m_dom={m_dom}, m_in={m_in}, rel_err={rel_err:.3e}"
