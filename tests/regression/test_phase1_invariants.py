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
import pytest
import scipy.integrate

from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    LangmuirSorption,
    compute_first_front_arrival_theta,
)
from gwtransport.fronttracking.output import (
    compute_breakthrough_curve,
    compute_cumulative_inlet_mass,
    compute_cumulative_outlet_mass,
    compute_domain_mass,
    compute_total_outlet_mass,
    integrate_fan_spatial_exact,
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


def test_integrate_fan_spatial_langmuir_clamps_below_u_zero():
    """Langmuir spatial fan integral is clamped below ``u_zero`` where C(u)=0.

    The Langmuir fan has finite extent in u: ``C(u) = sqrt(a·u/(κ−u)) − K_L``
    is non-negative only for ``u ≥ u_zero = K_L²·κ/(a + K_L²)``. Below
    ``u_zero`` the closed-form antiderivative would evaluate at negative
    c (unphysical). The function must clamp both bounds at ``u_zero``
    before evaluating; mirrors the upper-bound ``θ_zero`` clamp in the
    temporal counterpart ``_integrate_fan_exact_langmuir``.

    Two invariants are asserted:

    1. The clamped path agrees with ``scipy.integrate.quad`` on the
       valid range ``[u_zero, κ]`` at machine precision (closed-form
       fan integrand vs numerical quadrature of ``C_total(u)``).
    2. Integration over ``[u_start, κ]`` is invariant to ``u_start`` for
       all ``u_start ≤ u_zero`` (lower-bound clamp test): pulling the
       lower bound below ``u_zero`` must NOT contribute to the integral.

    Catches the five mutation classes flagged by the test reviewer:
    no-clamp (negative result), wrong-side clamp (upper instead of
    lower), off-by-factor on u_zero, sign flip on the sqrt term, and
    swapped sqrt-end/sqrt-start.
    """
    sorption = LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3)
    kappa = 10.0
    a_coeff = sorption.a_coeff
    k_l = sorption.k_l
    u_zero = k_l * k_l * kappa / (a_coeff + k_l * k_l)

    # Reference: numerical quadrature of the true (clipped) C_total integrand on [u_zero, κ].
    def c_total_clipped(u: float) -> float:
        if u <= u_zero or u >= kappa:
            return 0.0
        c = np.sqrt(a_coeff * u / (kappa - u)) - k_l
        c = max(c, 0.0)
        return float(sorption.total_concentration(c))

    # Reference integral on the valid sub-range [u_zero, 0.99·κ] (avoid the κ singularity).
    u_ref_lo = float(u_zero)
    u_ref_hi = 0.99 * kappa
    closed_form = integrate_fan_spatial_exact(
        theta_origin=0.0,
        v_origin=0.0,
        v_start=u_ref_lo,
        v_end=u_ref_hi,
        theta=kappa,
        sorption=sorption,
    )
    quad_ref, _ = scipy.integrate.quad(c_total_clipped, u_ref_lo, u_ref_hi, limit=200)
    rel_err = abs(closed_form - quad_ref) / max(abs(quad_ref), 1.0)
    assert rel_err < 1e-12, (
        f"Langmuir spatial closed-form disagrees with scipy.quad on valid range: "
        f"closed={closed_form:.10e}, quad={quad_ref:.10e}, rel_err={rel_err:.3e}"
    )

    # Lower-bound clamp invariant: pulling u_start below u_zero must NOT change the result.
    for frac in (0.0, 0.1, 0.5, 0.9, 1.0):
        u_start = frac * u_zero
        clamped = integrate_fan_spatial_exact(
            theta_origin=0.0,
            v_origin=0.0,
            v_start=u_start,
            v_end=u_ref_hi,
            theta=kappa,
            sorption=sorption,
        )
        # Equality is exact: the clamp short-circuits below u_zero, so the
        # antiderivative is evaluated at IDENTICAL endpoints.
        assert clamped == closed_form, (
            f"clamp invariant failed at u_start={u_start} (={frac}·u_zero): got {clamped!r}, expected {closed_form!r}"
        )


@pytest.mark.xfail(strict=True, reason="n<1 mirror outlet-history loss; fix deferred to Step 5b.")
def test_mass_balance_freundlich_nhalf_mirror_canonical_pulse_xfails():
    """Locked-failure: n=0.5 mirror canonical pulse mass balance is known broken.

    The trailing-side fan head reaches v_outlet BEFORE the DSW forms; the
    parent rarefaction is deactivated at DSW formation, so
    ``identify_outlet_segments`` (iterating only active waves) loses the
    rarefaction's pre-DSW outlet contribution.

    Marked ``xfail(strict=True)`` so the day Step 5b fixes the
    outlet-history loss, this test XPASSes and CI fails loudly — forcing
    the developer to flip the marker. Without this guard, a silent
    "accidentally-fixed" regression would go unnoticed. Empirical failure
    is rel_err ≈ -99% (catastrophic loss of mass).
    """
    sorption = FreundlichSorption(k_f=0.01, n=0.5, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    # Mirrored canonical pulse: ambient c=4, dip to c=0 in pulse bins.
    cin = np.full(500, 4.0)
    cin[5:15] = 0.0
    flow = np.full(500, 100.0)
    tedges = pd.date_range("2020-01-01", periods=501, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    mass_in = float(np.sum(cin * np.diff(tr.state.theta_edges)))
    mass_out, _ = compute_total_outlet_mass(v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
    assert np.isclose(mass_out, mass_in, rtol=1e-12), (
        f"n=0.5 mirror mass balance: mass_in={mass_in:.4f}, mass_out={mass_out:.4f}"
    )


@pytest.mark.xfail(strict=True, reason="Multi-DSW interaction at v_outlet; fix deferred to Step 5b.")
def test_mass_balance_freundlich_n2_multipulse_xfails():
    """Locked-failure: two-pulse n=2 canonical mass balance is known broken.

    DSW1's continuing fan past DSW2's arrival is not accounted for;
    ``concentration_at_point`` returns DSW1's c, but the segment is owned
    by DSW2 in the new partition. Empirical rel_err ≈ -32% (significant
    mass loss). Marked ``xfail(strict=True)`` — same purpose as the n=0.5
    mirror test.
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    cin = np.zeros(500)
    cin[5:15] = 4.0
    cin[40:50] = 6.0  # second pulse
    flow = np.full(500, 100.0)
    tedges = pd.date_range("2020-01-01", periods=501, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    mass_in = float(np.sum(cin * np.diff(tr.state.theta_edges)))
    mass_out, _ = compute_total_outlet_mass(v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
    assert np.isclose(mass_out, mass_in, rtol=1e-12), (
        f"multi-pulse n=2 mass balance: mass_in={mass_in:.4f}, mass_out={mass_out:.4f}"
    )


def test_pointwise_breakthrough_match_freundlich_n2_canonical():
    """Outlet concentration c(t) matches the analytical post-DSW-arrival fan profile.

    For the canonical n=2 pulse (k_f=0.01, ρ_b=1500, n_por=0.3, c_step=4,
    flow=100, V_out=200, pulse=bins[5:15]) the post-DSW outlet
    concentration is the analytical fan profile

        c(t) = 2500 / (t − 17)²       valid for t ≥ t_DSW_arrival ≈ 79.5 d

    The 17-day offset is the fan-profile time-axis intercept: numerically
    c(t)·(t−17)² = 2500 holds to ~1e-16 across the asserted t-range. It
    is NOT a shock-arrival time (the bare leading shock never arrives
    separately — it merges with the rarefaction head at θ=2580 into the
    DSW). See Step 5 re-evaluation §Q3 (physics-math reviewer round 2)
    for the full derivation against R(c) = (θ−θ_apex)/(v−v_apex).

    Mass-balance integrated tests can hide a "right total, wrong shape"
    mutation; this pointwise comparison is the highest-leverage test in
    the parametric suite per the test reviewer.
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    cin = np.zeros(500)
    cin[5:15] = 4.0
    flow = np.full(500, 100.0)
    tedges = pd.date_range("2020-01-01", periods=501, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    # DSW arrival at outlet in t-space is approximately 79.5 d; sample 20+ points well past arrival.
    t_arrival = 80.0
    t_samples = np.linspace(t_arrival + 1.0, t_arrival + 100.0, 25)
    theta_samples = np.array([tr.state.theta_at_t(float(t)) for t in t_samples])

    c_numerical = compute_breakthrough_curve(theta_samples, v_outlet, tr.state.waves, sorption)
    c_analytical = 2500.0 / (t_samples - 17.0) ** 2

    # Empirical rel_err ≈ 2e-16 (physics-math reviewer round 2 verified); 1e-13
    # leaves ~5000× headroom while catching sub-percent shape mutations that
    # a 1e-12 budget would absorb.
    np.testing.assert_allclose(
        c_numerical,
        c_analytical,
        rtol=1e-13,
        err_msg=f"pointwise breakthrough mismatch:\nt={t_samples}\nnum={c_numerical}\nana={c_analytical}",
    )


@pytest.mark.xfail(strict=True, reason="n≈1 limitation in _integrate_fan_exact_freundlich; fix deferred to Step 5c.")
def test_freundlich_n_just_above_1_reduces_to_constant_velocity_shock():
    """Freundlich n=1.001: the DSW closed-form should reduce to a constant-velocity shock.

    The closed-form invariant ``θ·u^n = K·(n·u^(n−1) + α)`` has a smooth
    limit as n→1⁺ (K → θ_c·u_c/(1+α), DSW velocity → 1/(1+α)). At n=1 the
    Freundlich isotherm is linear, so the rarefaction collapses to a single
    characteristic and the DSW degenerates to a regular constant-velocity
    shock. The test verifies (a) mass balance holds and (b) the outlet
    concentration varies slowly past DSW arrival.

    Currently XFAILs with ZeroDivisionError in
    ``_integrate_fan_exact_freundlich``: the formula uses
    ``alpha ** (1/beta) * kappa_theta * exponent`` with
    ``beta = 1/n − 1`` and ``exponent = 1/n − 1`` — both zero at n=1.
    Physics-math reviewer (round 2) confirmed the math has a smooth limit;
    this is a numerical-implementation limitation, not a math problem.
    """
    sorption = FreundlichSorption(k_f=0.01, n=1.001, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    cin = np.zeros(500)
    cin[5:15] = 4.0
    flow = np.full(500, 100.0)
    tedges = pd.date_range("2020-01-01", periods=501, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    mass_in = float(np.sum(cin * np.diff(tr.state.theta_edges)))
    mass_out, _ = compute_total_outlet_mass(v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
    rel_err = abs(mass_out - mass_in) / max(mass_in, 1.0)
    assert rel_err < 1e-12, f"n=1.001 mass balance: rel_err={rel_err:.3e}"

    t_samples = np.linspace(85.0, 120.0, 10)
    theta_samples = np.array([tr.state.theta_at_t(float(t)) for t in t_samples])
    c_out = compute_breakthrough_curve(theta_samples, v_outlet, tr.state.waves, sorption)
    c_mean = float(np.mean(c_out))
    c_std = float(np.std(c_out))
    relative_spread = c_std / max(abs(c_mean), 1e-10)
    assert relative_spread < 0.1, (
        f"n=1.001 outlet c should be near-constant past DSW arrival; got mean={c_mean:.6f}, "
        f"std={c_std:.6f}, relative_spread={relative_spread:.6f}"
    )
