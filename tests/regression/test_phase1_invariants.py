"""Phase-1 invariants for the fronttracking (V, θ) refactor.

These tests target *specific* defects fixed by the refactor — independent of
the snapshot baselines, so they catch regressions even if a scenario is
removed or replaced.

Covered:

- ``RarefactionWave.concentration_at_point`` FP-clamp at the tail boundary.
- One ``outlet_crossing`` event per (wave, outlet) pair, even at FP-edge
  collisions.
- Hard-coded ``compute_first_front_arrival_theta`` value for a
  ``ConstantRetardation`` case (independent of the function's own output).
- Mass balance to machine precision for the conservative-tracer case (the
  only scenario where Phase-1 wave splitting fully conserves; nonlinear
  sorption has a known ~few-thousand-mass deficit traced to the
  shock-rarefaction overlay, fixed in Phase 2 by ``DecayingShockWave``).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd

from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    compute_first_front_arrival_theta,
)
from gwtransport.fronttracking.output import (
    compute_cumulative_inlet_mass,
    compute_cumulative_outlet_mass,
    compute_domain_mass,
)
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import RarefactionWave


def test_rarefaction_concentration_at_point_clamps_at_tail_boundary():
    """At v = v_tail (machine-precision), c is clamped to c_tail instead of None.

    Regression for the FP-imprecision defect that caused freundlich_nhalf to
    miss the modified-rarefaction emission at the shock-rarefaction
    collision (400 mass deficit on the original main baseline).
    """
    sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3)
    raref = RarefactionWave(theta_start=500.0, v_start=0.0, c_head=0.0, c_tail=4.0, sorption=sorption)

    theta = 2504.9999999999977
    v = raref.tail_position_at_theta(theta)
    assert v is not None
    c = raref.concentration_at_point(v, theta)
    assert c == 4.0, f"expected exact c_tail=4.0 at FP-tail boundary, got {c!r}"


def test_outlet_crossing_no_fp_duplicate_per_wave():
    """No (wave, outlet) pair emits two outlet_crossing events within FP of each other.

    Legitimate distinct crossings (e.g., a rarefaction's head and tail) are
    separated by macroscopic Δθ. The FP-duplicate defect in
    ``find_outlet_crossing`` produced two crossings at ``|Δθ| ≈ 1 ULP`` for
    the *same* boundary, which this test catches.
    """
    sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3)
    cin = np.zeros(200)
    cin[5:15] = 4.0
    flow = np.full(200, 100.0)
    tedges = pd.date_range("2020-01-01", periods=201, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=200.0, sorption=sorption)
    tr.run(max_iterations=100000)

    per_wave: dict = {}
    for ev in tr.state.events:
        if ev["type"] != "outlet_crossing":
            continue
        key = (id(ev["wave"]), ev["location"])
        per_wave.setdefault(key, []).append(float(ev["theta"]))

    for key, thetas in per_wave.items():
        thetas_sorted = sorted(thetas)
        for a, b in pairwise(thetas_sorted):
            tol = 1e-9 * max(abs(a), abs(b), 1.0)
            assert (b - a) > tol, f"FP-duplicate outlet_crossing for wave {key}: θ={a!r} and θ={b!r}"


def test_compute_first_front_arrival_theta_constant_retardation_analytic():
    """Hard-coded analytic answer for constant retardation: θ = θ_emit + V·R.

    Defends against silent-rewrite of the function itself (snapshot test
    `test_theta_first_arrival_matches` only confirms determinism, not
    physical correctness).
    """
    cin = np.array([0.0] * 5 + [10.0] * 10)
    theta_edges = np.concatenate(([0.0], np.cumsum(np.full(15, 100.0))))
    sorption = ConstantRetardation(retardation_factor=2.5)
    theta_first = compute_first_front_arrival_theta(
        cin=cin, theta_edges=theta_edges, aquifer_pore_volume=500.0, sorption=sorption
    )
    assert theta_first == 500.0 + 500.0 * 2.5  # θ_edges[5] + V·R = 500 + 1250 = 1750


def test_mass_balance_constant_retardation_machine_precision():
    """Mass balance holds to machine precision for the linear (conservative) case.

    Samples include θ values **mid-transit** (where m_dom > 0) so that a
    silent mutation of ``compute_domain_mass`` cannot pass the invariant.
    """
    sorption = ConstantRetardation(retardation_factor=2.0)
    v_outlet = 200.0
    cin = np.zeros(100)
    cin[5:15] = 4.0
    flow = np.full(100, 100.0)
    tedges = pd.date_range("2020-01-01", periods=101, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    theta_edges = tr.state.theta_edges
    # Mid-transit θ: pulse emits at θ ∈ [500, 1500], propagates at speed 1/R = 0.5,
    # so leading shock crosses outlet at θ = 500 + V·R = 500 + 400 = 900 and trailing
    # at θ = 1500 + 400 = 1900. Pick interior θ where m_dom > 0.
    interior_thetas = [700.0, 1000.0, 1400.0, 1700.0]
    post_transit_thetas = [float(frac * theta_edges[-1]) for frac in (0.25, 0.5, 0.75, 1.0)]

    saw_nonzero_domain = False
    for theta in [*interior_thetas, *post_transit_thetas]:
        m_in = compute_cumulative_inlet_mass(theta=theta, cin=cin, theta_edges=theta_edges)
        m_dom = compute_domain_mass(theta=theta, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
        m_out = compute_cumulative_outlet_mass(theta=theta, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
        if m_dom > 0:
            saw_nonzero_domain = True
        err = abs((m_dom + m_out) - m_in)
        tol = 1e-14 * max(m_in, 1.0)
        assert err <= tol, f"mass-balance violation at θ={theta}: err={err:.6e} > tol={tol:.6e}"

    assert saw_nonzero_domain, "Sample window must include θ where m_dom > 0 to exercise compute_domain_mass"


def test_mass_balance_freundlich_n2_canonical_pulse_machine_precision():
    """Mass balance holds to machine precision for n=2 Freundlich canonical pulse.

    Exercises the full Phase 2 step 4 closed-form chain: DecayingShockWave
    trajectory + ``integrate_fan_exact`` (temporal) + ``integrate_fan_spatial_exact``
    (spatial in ``compute_domain_mass``). Samples θ values covering:

    - pre-shock-arrival (m_dom > 0, m_out = 0): mass fully in domain
    - during outlet drainage (m_dom > 0 and m_out > 0): both contribute
    - post-fan (m_dom small): asymptotic decay region

    Prior to step 4 this test would XFAIL at ~5e-7 due to the c_min clamp in
    ``total_concentration`` biasing Rankine-Hugoniot shock speeds.
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    cin = np.zeros(500)
    cin[5:15] = 4.0
    flow = np.full(500, 100.0)
    tedges = pd.date_range("2020-01-01", periods=501, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    # Theta checkpoints spanning the pulse evolution.
    checkpoints = [4000.0, 6000.0, 9000.0, 15000.0, 25000.0]
    saw_dom = False
    saw_out = False
    for theta in checkpoints:
        m_in = compute_cumulative_inlet_mass(theta=theta, cin=cin, theta_edges=tr.state.theta_edges)
        m_dom = compute_domain_mass(theta=theta, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
        m_out = compute_cumulative_outlet_mass(
            theta=theta, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption
        )
        if m_dom > 1.0:
            saw_dom = True
        if m_out > 1.0:
            saw_out = True
        err = abs((m_dom + m_out) - m_in)
        tol = 1e-14 * max(m_in, 1.0)
        assert err <= tol, f"mass-balance violation at θ={theta}: err={err:.6e} > tol={tol:.6e}"

    assert saw_dom, "Sample window must include θ where m_dom > 1 to exercise the closed-form fan integral"
    assert saw_out, "Sample window must include θ where m_out > 1 to exercise compute_cumulative_outlet_mass"


def test_theta_constant_across_zero_flow_bin():
    """θ is constant across every zero-flow bin: ``theta_edges[i+1] == theta_edges[i]``.

    The (V, θ) refactor's flow-change machinery is exactly the precomputed
    ``theta_edges`` array; a zero-flow bin must produce a zero-width θ-segment.
    A regression that silently advances θ across a flow=0 bin would be invisible
    to the breakthrough-curve checks (cout is unchanged in θ-space) but would
    corrupt all θ→t translations afterwards.
    """
    flow = np.array([100.0] * 20 + [0.0] * 3 + [100.0] * 17)  # zero-flow bins at indices [20, 23)
    cin = np.zeros(len(flow))
    cin[5:15] = 4.0
    tedges = pd.date_range("2020-01-01", periods=len(flow) + 1, freq="D")
    tr = FrontTracker(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=200.0,
        sorption=ConstantRetardation(retardation_factor=2.0),
    )
    theta_edges = tr.state.theta_edges
    for i in range(len(flow)):
        if flow[i] == 0.0:
            assert theta_edges[i + 1] == theta_edges[i], (
                f"Zero-flow bin {i}: θ advanced by {theta_edges[i + 1] - theta_edges[i]:.6e} (must be 0)"
            )


def test_theta_at_t_roundtrip_machine_precision():
    """``theta_at_t`` and ``t_at_theta`` invert each other to machine precision.

    The (V, θ) refactor's central claim is that flow-change behavior is absorbed
    into the θ(t) map at the API boundary; this map MUST be a proper bijection
    on the non-zero-flow portion of the domain (it is many-to-one in zero-flow
    bins where θ is constant — handled by right-continuous t_at_theta).
    """
    # Three flow profiles covering the refactor's load-bearing cases.
    profiles = [
        # (label, flow array). All use 100 daily bins so tedges_days = [0, 1, ..., 100].
        ("constant_flow", np.full(100, 100.0)),
        ("step_change_flow", np.array([100.0] * 40 + [50.0] * 30 + [200.0] * 30)),
    ]
    for label, flow in profiles:
        cin = np.zeros(len(flow))
        cin[5:15] = 4.0  # any nonzero is fine; θ_edges don't depend on cin
        tedges = pd.date_range("2020-01-01", periods=len(flow) + 1, freq="D")
        tr = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=200.0,
            sorption=ConstantRetardation(retardation_factor=2.0),
        )
        state = tr.state
        # θ → t → θ inverts everywhere (excluding boundary endpoints to avoid
        # the searchsorted clip-edge case).
        theta_samples = np.linspace(state.theta_edges[1], state.theta_edges[-1] - 0.5, 50)
        for theta in theta_samples:
            theta_recovered = state.theta_at_t(state.t_at_theta(float(theta)))
            err = abs(theta_recovered - float(theta))
            tol = 1e-12 * max(abs(theta), 1.0)
            assert err <= tol, (
                f"[{label}] θ→t→θ roundtrip failed at θ={theta}: recovered={theta_recovered}, err={err:.3e} > {tol:.3e}"
            )
