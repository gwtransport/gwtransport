"""Regression: ``concentration_at_point`` honors ``was_active_at`` retrospectively.

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
