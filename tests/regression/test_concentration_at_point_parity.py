"""Dispatch contracts of ``concentration_at_point``, and their parity with ``compute_domain_mass``.

``compute_domain_mass`` reconstructs the spatial profile through
``concentration_at_point``, so the two must resolve the wave list by the same
rules. A dispatch error is not local: m_dom drops out of the conservation form
``m_out = m_in − m_dom``, and the production advection path then echoes the
inlet straight through ``compute_bin_averaged_concentration_exact``. Each test
below pairs the pointwise contract with the m_dom consequence.

Contracts pinned here:

- **Retrospective activity.** A query at θ selects faces by
  ``wave.was_active_at(θ)``, never by the current ``wave.is_active`` flag. The
  canonical n=2 pulse deactivates its leading shock at θ=2580 (collision with
  the trailing rarefaction tail); a query at θ=550 still sees it, and the c=4
  plateau it carries.
- **Nearest downstream face.** Among stacked shocks all upstream of the query
  point, the state at ``v`` comes from the nearest face downstream of ``v`` —
  the one with the largest ``V_shock`` — not from the youngest by
  ``theta_start``, which in a cascade is the innermost in V.
- **Obstruction.** A shock's downstream state reaches only as far as the next
  face; it does not propagate past an intervening rarefaction into plateaus that
  rarefaction owns.
- **Overlapping fans.** Where two DecayingShockWave fans cover the same ``v``,
  the value comes from the newer wave, whose fan has physically swept the region.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import FreundlichSorption
from gwtransport.fronttracking.output import (
    compute_cumulative_inlet_mass,
    compute_domain_mass,
    concentration_at_point,
)
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import DecayingShockWave


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
    """Among stacked shocks all left of v, c at v is the right state of the
    shock with the LARGEST V_shock — the one closest to v from the left — not
    of the one with the largest theta_start.

    A c=[0,3,6,9,12] ramp through Freundlich n=2 produces a cascade of shocks
    that coexist at θ_query=250 before merging. There the youngest shock by
    ``theta_start`` is the innermost in V, so an age-based tiebreaker returns
    the wrong plateau for every v right of the stack; through the advection API
    that surfaces as a spike in ``cout``.
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
    # So the layout is V_new=3.86 < V_old=5.0. An age tiebreaker would pick the
    # youngest shock (theta_start=200) → c_right = 3; the rightmost-V shock
    # (theta_start=100) is the correct owner → c_right = 0.
    theta_query = 250.0
    v_query_right = 100.0  # well right of both shocks

    c = concentration_at_point(v=v_query_right, theta=theta_query, waves=tr.state.waves, sorption=sorption)
    # At v=100 right of all shocks at θ=250: c must be the IC value =
    # c_right of the leading (rightmost-V) shock = 0.
    assert c == 0.0, f"v=100 right of all shocks at θ=250: expected c=0 (IC), got {c}"


def test_multi_rarefaction_overlap_no_overcount_in_domain_mass():
    """A shock's right state reaches only to the next face, not past intervening rarefactions.

    For an n<1 ramp ``cin=[0,3,6,9,12,0]`` (Freundlich n=0.5), the simulator
    produces 4 stacked rarefactions from the upramp PLUS a closing shock from
    the trailing zero. At θ=500 (just after the closing shock forms at V=0), 4
    active rarefactions sit in the V-range [0.083, 400] with c plateaus between
    them (c=9, 6, 3 at the gaps), and c=0 past the outermost head.

    The test asserts the c profile is geometrically correct at a midpoint of
    every constant region. Letting the closing shock's c_R=12 reach through all
    of them — including [400, 500], which the outermost rarefaction owns — sends
    m_dom to 727694 against the 3000 actually injected, a 240× overcount.
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
            f"(an unobstructed shock c_R=12 would fill every constant region)"
        )

    # m_dom must match m_in: no mass has reached v_outlet=500 yet.
    m_in = compute_cumulative_inlet_mass(theta=theta_query, cin=cin, theta_edges=tr.state.theta_edges)
    m_dom = compute_domain_mass(theta=theta_query, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
    rel_err = abs(m_dom - m_in) / max(m_in, 1.0)
    assert rel_err < 1e-12, f"m_dom={m_dom}, m_in={m_in}, rel_err={rel_err:.3e}"


def test_multi_dsw_concentration_at_point_uses_newest():
    """Two-pulse Freundlich n=2 with two coexisting DSWs: the newer fan owns the overlap.

    Where both fans cover ``v``, the newer DecayingShockWave has physically
    swept through the region, so its value (~0.300 here) is the state at ``v``,
    not the older one's (~0.102). Returning the first in-fan match found by
    iterating the wave list chronologically would take the older value and
    disagree with ``compute_domain_mass``.
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    cin = np.zeros(80)
    cin[5:15] = 4.0
    cin[40:50] = 6.0
    flow = np.full(80, 100.0)
    tedges = pd.date_range("2020-01-01", periods=81, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    dsws = [w for w in tr.state.waves if isinstance(w, DecayingShockWave)]
    theta_query = 10000.0
    active = [w for w in dsws if w.was_active_at(theta_query)]
    assert len(active) == 2, f"need two coexisting DSWs to exercise the dispatch; got {len(active)}"

    v_s_values = [w.position_at_theta(theta_query) for w in active]
    assert all(v is not None for v in v_s_values)
    v_query = min(v for v in v_s_values if v is not None) * 0.5  # left of both shocks → in both fans

    newest = max(active, key=lambda w: w.theta_start)
    c_newest = newest.concentration_at_point(v_query, theta_query)
    c_dispatch = concentration_at_point(v_query, theta_query, tr.state.waves, sorption)
    assert c_newest is not None
    assert c_dispatch is not None
    assert c_dispatch == pytest.approx(c_newest, rel=1e-12)

    older_in_fan = min(active, key=lambda w: w.theta_start).concentration_at_point(v_query, theta_query)
    assert older_in_fan is not None
    assert abs(c_dispatch - older_in_fan) > 0.1, (
        f"dispatch={c_dispatch} suspiciously close to older DSW's fan value {older_in_fan}; "
        "regression to chronological iteration?"
    )
