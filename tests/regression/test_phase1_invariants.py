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

import mpmath as mp
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
    integrate_fan_exact,
    integrate_fan_spatial_exact,
    integrate_rarefaction_exact,
)
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import DecayingShockWave, RarefactionWave


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
        m_out = compute_cumulative_outlet_mass(
            theta=theta,
            v_outlet=v_outlet,
            waves=tr.state.waves,
            sorption=sorption,
            cin=cin,
            theta_edges=tr.state.theta_edges,
        )
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
    ``u_zero`` the retardation ``R = κ/u`` exceeds the maximal ``R(0)``, so
    ``concentration_from_retardation`` returns 0 and the universal
    antiderivative ``G = C_total·u − κ·c`` is identically 0 there; pulling the
    lower bound below ``u_zero`` therefore adds nothing to the integral.

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


@pytest.mark.parametrize("n", [0.25, 0.5])
def test_mass_balance_freundlich_nhalf_mirror_pointwise_breakthrough(n):
    """n<1 mirror canonical pulse: pointwise breakthrough matches analytical fan.

    Step 5b (conservation-law pivot): outlet mass derived from
    ``m_out = m_in - m_dom``. The original identify_outlet_segments +
    integrate_fan_exact path couldn't handle the n<1 mirror geometry
    (parent rarefaction deactivated at DSW formation, deactivated wave's
    history lost). The pivot to retrospective ``was_active_at`` queries in
    ``compute_domain_mass`` (round 2 fix) is required for correct m_dom at
    intermediate θ.

    Step 5b round 5: rewritten after round-4 reviewer flagged the original
    per-checkpoint ``(m_dom + m_out) - m_in`` identity as tautological —
    ``compute_cumulative_outlet_mass`` returns ``m_in - m_dom`` literally,
    so the assertion is algebraically zero regardless of mutations. The
    new assertion compares ``compute_breakthrough_curve`` against an
    analytical rarefaction fan formula at θ-values spanning pre-DSW
    (rarefaction dispatch) and immediately-post-DSW (DSW closed-form
    fan-lookup) regimes. The analytical c is computed directly from the
    Freundlich isotherm parameters; mutations to either
    ``RarefactionWave.concentration_at_point`` or to the DSW's
    ``concentration_at_point`` fan-interior branch surface as a numerical
    mismatch (verified: a 0.5% rarefaction-formula mutation produces
    rel_err≈5e-3, well above the test's 1e-10 rtol). This test does NOT
    catch ``was_active_at → is_active`` reverts — that mutation is caught
    by ``test_compute_domain_mass_matches_inlet_pre_outlet_arrival``,
    which exercises a different dispatch path.

    Parameter range restricted to n ∈ {0.25, 0.5} — these are the n<1
    values that actually exercise the DSW closed-form solver. For
    n ∈ {0.75, 0.9} the simulator either falls through to the
    non-canonical Phase-1 overlay (no DSW formed; the wave list explodes
    to >200000 entries for n=0.75) or produces only regular shocks.

    Asserts:
    1. Pointwise breakthrough match against analytical fan at pre-DSW θ.
    2. Asymptotic total outlet mass: ``m_out_total = m_in_total -
       C_T(c_∞)·V_outlet`` where ``c_∞ = cin[-1] = 4`` is the sustained
       ambient. For canonical n<1 mirror this is NOT equal to mass_in —
       the aquifer fills to steady state at c=4 and never empties.
    """
    sorption = FreundlichSorption(k_f=0.01, n=n, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    # Mirrored canonical pulse: ambient c=4, dip to c=0 in pulse bins.
    cin = np.full(500, 4.0)
    cin[5:15] = 0.0
    flow = np.full(500, 100.0)
    tedges = pd.date_range("2020-01-01", periods=501, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    # Pointwise breakthrough match: the rarefaction from the cin=0→cin=4
    # step at θ=0 (apex at V=0, θ=0) has self-similar profile
    # ``R(c) = θ / V``. Solving for c at v=v_outlet: c is the value
    # satisfying ``retardation(c) = θ_query / v_outlet``. For Freundlich:
    # ``1 + α · c^(1/n - 1) = θ/V``  →  ``c = ((θ/V - 1) / α)^(1/(1/n - 1))``
    # where α = (bulk_density · k_f) / (porosity · n).
    # The rarefaction head (c=0) reaches v_outlet at θ = v_outlet.
    # Sample θ values past head-arrival but before DSW formation. DSW
    # formation θ is determined by shock vs raref tail collision.
    alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)
    exponent = (1.0 / sorption.n) - 1.0

    # The DSW inherits the parent rarefaction's apex (V=0, θ=0), so the
    # self-similar profile ``R(c) = θ/V`` continues to hold at v_outlet
    # AFTER DSW formation — UNTIL the DSW's shock interface (V_s) propagates
    # past v_outlet and v_outlet falls onto the c_fixed=0 side. For the
    # n=0.25 case this happens around θ ≈ 2000, for n=0.5 around θ ≈ 3000.
    # We sample both pre- and immediately-post-DSW θ values to exercise
    # both the rarefaction dispatch and the DSW closed-form fan lookup.
    # DSW forms at θ_dsw — n=0.25 ≈ 667, n=0.5 ≈ 1003.
    theta_samples = np.array([
        v_outlet + 100.0,  # pre-DSW (rarefaction dispatch)
        500.0,  # pre-DSW
        800.0 if n == 0.25 else 1100.0,  # just after DSW formation
        1200.0 if n == 0.25 else 1500.0,  # post-DSW, fan still covers v_outlet
        2000.0 if n == 0.25 else 2500.0,  # latest post-DSW where fan still active
    ])
    c_numerical = compute_breakthrough_curve(theta_samples, v_outlet, tr.state.waves, sorption)
    c_analytical = ((theta_samples / v_outlet - 1.0) / alpha) ** (1.0 / exponent)

    # rtol=1e-10 gives ~1000× headroom past the analytical formula's FP
    # noise. Tighter values trip on the DSW's brentq inversion (closed-form
    # invariant solver) which converges to ~1e-12 in practice.
    np.testing.assert_allclose(
        c_numerical,
        c_analytical,
        rtol=1e-10,
        err_msg=f"n={n} pointwise breakthrough mismatch:\nθ={theta_samples}\nnum={c_numerical}\nana={c_analytical}",
    )

    # Asymptotic total: m_out_total = m_in - C_T(c_∞) · V_outlet (c_∞=4 here).
    # NOTE: compute_total_outlet_mass uses the asymptotic formula directly
    # (no wave list), so this is a smoke test for the formula wiring — it
    # catches sign flips or missing c_∞ retrieval but not wave-list bugs.
    mass_in = float(np.sum(cin * np.diff(tr.state.theta_edges)))
    mass_out = compute_total_outlet_mass(
        v_outlet=v_outlet,
        sorption=sorption,
        cin=cin,
        theta_edges=tr.state.theta_edges,
    )
    c_inf = float(cin[-1])
    expected_m_out = mass_in - float(sorption.total_concentration(c_inf)) * v_outlet
    assert np.isclose(mass_out, expected_m_out, rtol=1e-12), (
        f"n={n} mirror asymptotic: mass_out={mass_out:.4f}, expected={expected_m_out:.4f}"
    )


def test_mass_balance_freundlich_n2_multipulse():
    """Two-pulse n=2 canonical mass balance at machine precision via conservation.

    Step 5b (conservation-law pivot): outlet mass derived from
    ``m_out = m_in - m_dom``, sidestepping the multi-DSW outlet dispatch
    that the original identify_outlet_segments + integrate_fan_exact path
    couldn't handle (newer DSW's swept-up region miscounted as separate
    DSW1 fan contribution).
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
    mass_out = compute_total_outlet_mass(
        v_outlet=v_outlet,
        sorption=sorption,
        cin=cin,
        theta_edges=tr.state.theta_edges,
    )
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


def test_freundlich_n_just_above_1_plateau_holds_inlet_c():
    """Freundlich n=1.001 limit: between leading-shock and rarefaction-head arrival
    at v_outlet, c at v_outlet equals c_inlet exactly.

    At n=1 exactly the Freundlich isotherm collapses to linear retardation
    and the trailing rarefaction degenerates to a single characteristic
    carrying ``c = c_inlet``. For finite ``n > 1`` close to 1 the
    rarefaction still has finite width but moves so slightly faster than
    the leading shock that any DSW collision occurs well past a typical
    ``v_outlet``. At ``v_outlet`` the breakthrough therefore consists of:

    1. ``c = 0`` until the leading shock arrives.
    2. ``c = c_inlet`` while ``v_outlet`` sits between the leading shock
       and the rarefaction head — this is the n→1 "constant-velocity
       shock" plateau.
    3. A narrow self-similar fan as the rarefaction passes ``v_outlet``,
       collapsing to a step in the strict ``n → 1`` limit.

    This test asserts (1) mass balance at machine precision and (2)
    ``c(v_outlet, θ) = c_inlet`` to machine precision for ``θ`` strictly
    inside the plateau window ``(θ_shock_arrival, θ_raref_head_arrival)``.
    The plateau-window endpoints are computed from Rankine-Hugoniot
    (shock) and ``1/R(c_inlet)`` (rarefaction-head characteristic speed),
    not read out of the wave list — so a regression that shifts either
    boundary breaks the assertion.

    Replaces a previous xfail whose ``relative_spread < 0.1`` premise was
    wrong: at n=1.001 the breakthrough at v_outlet is a step (no smooth
    fan reaches v_outlet before the integration window ends), so any wide
    sampling window sees both 0s and ``c_inlet``s and a large std.
    """
    c_inlet = 4.0
    sorption = FreundlichSorption(k_f=0.01, n=1.001, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    cin = np.zeros(500)
    cin[5:15] = c_inlet
    flow = np.full(500, 100.0)
    tedges = pd.date_range("2020-01-01", periods=501, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    mass_in = float(np.sum(cin * np.diff(tr.state.theta_edges)))
    mass_out = compute_total_outlet_mass(
        v_outlet=v_outlet,
        sorption=sorption,
        cin=cin,
        theta_edges=tr.state.theta_edges,
    )
    np.testing.assert_allclose(mass_out, mass_in, rtol=1e-12)

    # Plateau-window endpoints: in (V, θ) the leading 0 → c_inlet shock
    # leaves the inlet at ``theta_edges[5]`` with Rankine-Hugoniot speed
    # ``s_shock = c_inlet / C_T(c_inlet)``; the trailing rarefaction's head
    # (the ``c = c_inlet`` characteristic) leaves at ``theta_edges[15]``
    # with speed ``1/R(c_inlet)``. Both travel a θ-distance ``v_outlet``
    # to reach the outlet.
    theta_shock_emit = float(tr.state.theta_edges[5])
    theta_raref_emit = float(tr.state.theta_edges[15])
    s_shock = c_inlet / float(sorption.total_concentration(c_inlet))
    r_head = float(sorption.retardation(c_inlet))
    theta_shock_at_outlet = theta_shock_emit + v_outlet / s_shock
    theta_raref_head_at_outlet = theta_raref_emit + v_outlet * r_head
    assert theta_shock_at_outlet < theta_raref_head_at_outlet, (
        "test premise violated: at n=1.001 the rarefaction head must trail the "
        f"leading shock at v_outlet (got θ_shock={theta_shock_at_outlet}, "
        f"θ_raref={theta_raref_head_at_outlet})"
    )

    # Sample strictly inside the plateau; 1% margin avoids FP brushing of the boundaries.
    margin = 0.01 * (theta_raref_head_at_outlet - theta_shock_at_outlet)
    theta_samples = np.linspace(
        theta_shock_at_outlet + margin,
        theta_raref_head_at_outlet - margin,
        5,
    )
    c_out = compute_breakthrough_curve(theta_samples, v_outlet, tr.state.waves, sorption)
    np.testing.assert_allclose(c_out, c_inlet, rtol=1e-12)


def test_mass_balance_freundlich_n2_c_fixed_gt_0_pre_filled_aquifer():
    """Step 5c regression: c_fixed > 0 DSW mass balance at machine precision.

    The canonical c_R > 0 scenario (pulse ``c_low → c_H → c_low``) cannot
    test the DSW c_fixed > 0 closed form directly because the simulator
    initializes the aquifer at c=0, and the 0→c_low fill-up shock at θ=0
    merges with the leading pulse shock before any DSW could form (the
    canonical_head dispatch check ``raref.c_tail == shock.c_right`` fails
    when c_right has been "knocked down" to 0 by the merger).

    Workaround: pre-fill the aquifer with c=c_low for long enough that the
    fill-up shock exits before the pulse arrives (θ_fillup_exit ≈ 10200
    for our params). Use a SHORT pulse (3 bins) so the trailing fan head
    catches the leading shock INSIDE the domain, forming a DSW with
    c_fixed = c_low > 0 (decay_side='left'). Then assert per-checkpoint
    ``m_in = m_dom + m_out`` at machine precision.

    Without the Step 5c V_tail/θ_tail clamps in ``integrate_fan_exact`` /
    ``integrate_fan_spatial_exact``, the fan formula extrapolates past
    the physical fan range and underestimates m_out / m_dom by 37–84% at
    intermediate θ (per physics-math reviewer round 2 diagnosis).
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    v_outlet = 200.0
    # Pre-fill: 150 bins at c=1 lets the 0→1 fillup shock exit (θ_exit ≈ 10200, at bin 102).
    # Short 3-bin pulse so fan head catches shock at V≈120 (inside V_outlet=200).
    n_bins = 1000
    cin = np.full(n_bins, 1.0)
    cin[150:153] = 4.0
    flow = np.full(n_bins, 100.0)
    tedges = pd.date_range("2020-01-01", periods=n_bins + 1, freq="D")

    tr = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=v_outlet, sorption=sorption)
    tr.run(max_iterations=100000)

    # Verify a DSW with c_fixed > 0 actually formed.
    dsws = [w for w in tr.state.waves if isinstance(w, DecayingShockWave) and w.c_fixed > 0.0]
    assert len(dsws) == 1, f"expected exactly one DSW with c_fixed > 0; got {len(dsws)}"
    assert dsws[0].decay_side == "left", f"expected decay_side='left'; got {dsws[0].decay_side!r}"
    assert np.isclose(dsws[0].c_fixed, 1.0), f"expected c_fixed=1; got {dsws[0].c_fixed}"

    # Per-checkpoint mass balance at machine precision. Sample θ past the
    # fillup-exit (θ=10200) so the fillup transient doesn't perturb m_dom.
    theta_max = float(tr.state.theta_edges[-1])
    for frac in (0.25, 0.5, 0.75, 0.99):
        theta = float(frac * theta_max)
        m_in = compute_cumulative_inlet_mass(theta=theta, cin=cin, theta_edges=tr.state.theta_edges)
        m_dom = compute_domain_mass(theta=theta, v_outlet=v_outlet, waves=tr.state.waves, sorption=sorption)
        m_out = compute_cumulative_outlet_mass(
            theta=theta,
            v_outlet=v_outlet,
            waves=tr.state.waves,
            sorption=sorption,
            cin=cin,
            theta_edges=tr.state.theta_edges,
        )
        err = abs((m_dom + m_out) - m_in)
        # Empirical max abs_err = 3.6e-12 at frac=0.25 (per test-reviewer Q5);
        # rtol=1e-14·m_in + atol=1e-11 covers the worst case with ~3× margin.
        tol = 1e-14 * max(m_in, 1.0) + 1e-11
        assert err <= tol, f"c_fixed>0 mass balance at θ={theta}: err={err:.6e} > tol={tol:.6e}"


def test_integrate_fan_exact_c_apex_constant_region_freundlich_n2():
    """Temporal fan integral: c_apex > 0 splits ``[θ_start, θ_end]`` at ``θ_tail``.

    For Freundlich n=2 with c_apex=1, the fan c at v_outlet is bounded
    below by c_apex; the formula extrapolates to c < c_apex past
    ``θ_tail = θ_origin + (v_outlet − v_origin) · R(c_apex)``. The
    integral should clamp at θ_tail and add c_apex·(θ_end − θ_tail) for
    any θ_end > θ_tail.

    Hand-derived: for sorption(k_f=0.01, ρ_b=1500, n_por=0.3, n=2),
    R(c=1) = 1 + (ρ_b·k_f/(n_por·n))·c^(-1/2) = 1 + 25 = 26. So with
    θ_origin=0, v_origin=0, v_outlet=10: θ_tail = 10·26 = 260. For
    θ_end = 500 (well past θ_tail), the constant-region contribution
    is c_apex · (500 − 260) = 1 · 240 = 240.
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    v_outlet = 10.0
    theta_origin = 0.0
    v_origin = 0.0
    c_apex = 1.0

    # R(c_apex)·(v_outlet - v_origin) = θ_tail offset from θ_origin.
    r_c_apex = float(sorption.retardation(c_apex))
    theta_tail = theta_origin + (v_outlet - v_origin) * r_c_apex
    assert np.isclose(theta_tail, 260.0), f"hand-derived θ_tail check: {theta_tail}"

    # Integration over [θ_origin, θ_tail]: pure fan portion (no constant region
    # below θ_tail). Compared to scipy.quad over the closed-form fan integrand.
    def c_fan(theta_val):
        # c(θ) = [(R(c) - 1) / α]^(1/β) where R(c) = (θ-θ_origin)/(v_outlet-v_origin).
        # For n=2: c = [(r-1) / α]^(1/(1/n-1)) = [(r-1)/α]^(-2). α = 25 here.
        alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)
        r = (theta_val - theta_origin) / (v_outlet - v_origin)
        base = r - 1.0
        if base <= 0:
            return 0.0
        # 1/β = 1/(1/n - 1) = 1/(-0.5) = -2 for n=2 → c = (base/α)^(-2)
        return (base / alpha) ** (-2)

    fan_only_quad, _ = scipy.integrate.quad(c_fan, 100.0, theta_tail, limit=200)
    fan_only_closed = integrate_fan_exact(theta_origin, v_origin, v_outlet, 100.0, theta_tail, sorption, c_apex=c_apex)
    assert np.isclose(fan_only_closed, fan_only_quad, rtol=1e-10), (
        f"fan-only portion mismatch: closed={fan_only_closed}, quad={fan_only_quad}"
    )

    # Integration over [100, 500]: fan portion + constant c_apex region.
    full_closed = integrate_fan_exact(theta_origin, v_origin, v_outlet, 100.0, 500.0, sorption, c_apex=c_apex)
    expected_constant = c_apex * (500.0 - theta_tail)
    assert np.isclose(full_closed - fan_only_closed, expected_constant, rtol=1e-14), (
        f"constant-region contribution: full={full_closed}, fan={fan_only_closed}, "
        f"diff={full_closed - fan_only_closed}, expected={expected_constant}"
    )

    # Sanity: c_apex=0 default reproduces the original behavior on the same range
    # for [100, θ_tail] (where the fan formula is valid). For θ > θ_tail the
    # c_apex=0 path gives c < c_apex (extrapolated, unphysical), so the default
    # path agrees with the c_apex>0 path only up to θ_tail.
    default_closed_fan_only = integrate_fan_exact(theta_origin, v_origin, v_outlet, 100.0, theta_tail, sorption)
    assert np.isclose(default_closed_fan_only, fan_only_closed, rtol=1e-14)


def test_integrate_fan_spatial_exact_c_apex_constant_region_freundlich_n2():
    """Spatial fan integral: c_apex > 0 splits ``[v_start, v_end]`` at ``v_tail``.

    Mirror of the temporal test. For Freundlich n=2 c_apex=1 at θ=500
    with apex at (0, 0): u_tail = κ/R(c_apex) = 500/26 ≈ 19.23. For
    v_end = 100 (well past u_tail), the [0, u_tail] segment contributes
    C_total(c_apex) · u_tail = C_T(1) · 19.23 = 51 · 19.23 = 980.77.
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    theta = 500.0
    theta_origin = 0.0
    v_origin = 0.0
    c_apex = 1.0
    kappa = theta - theta_origin

    r_c_apex = float(sorption.retardation(c_apex))
    u_tail = kappa / r_c_apex
    c_total_apex = float(sorption.total_concentration(c_apex))
    expected_constant = c_total_apex * u_tail

    # Below u_tail only: should equal C_T(c_apex) · u_tail (no fan portion).
    below_tail = integrate_fan_spatial_exact(theta_origin, v_origin, 0.0, u_tail, theta, sorption, c_apex=c_apex)
    assert np.isclose(below_tail, expected_constant, rtol=1e-14), (
        f"[0, u_tail] constant region: got {below_tail}, expected {expected_constant}"
    )

    # Fan portion only ([u_tail, 50]): should match the c_apex=0 default since
    # u_start is already at u_tail (no constant-region trigger inside).
    fan_only_with_apex = integrate_fan_spatial_exact(
        theta_origin, v_origin, u_tail, 50.0, theta, sorption, c_apex=c_apex
    )
    fan_only_default = integrate_fan_spatial_exact(theta_origin, v_origin, u_tail, 50.0, theta, sorption)
    assert np.isclose(fan_only_with_apex, fan_only_default, rtol=1e-14), (
        f"fan-only [u_tail, 50] should not depend on c_apex when u_start >= u_tail; "
        f"got with_apex={fan_only_with_apex}, default={fan_only_default}"
    )

    # Full [0, 50]: constant region + fan region.
    full = integrate_fan_spatial_exact(theta_origin, v_origin, 0.0, 50.0, theta, sorption, c_apex=c_apex)
    assert np.isclose(full, expected_constant + fan_only_default, rtol=1e-14)

    # u_end < u_tail (entirely inside the constant region). Catches a
    # ``min(u_end, u_tail) - u_start`` → ``u_tail - u_start`` mutation that
    # would otherwise be invisible when u_end ≥ u_tail.
    half_u_tail = 0.5 * u_tail
    partial = integrate_fan_spatial_exact(theta_origin, v_origin, 0.0, half_u_tail, theta, sorption, c_apex=c_apex)
    expected_partial = c_total_apex * half_u_tail
    assert np.isclose(partial, expected_partial, rtol=1e-14), (
        f"[0, u_tail/2] constant region: got {partial}, expected {expected_partial}"
    )

    # Non-zero theta_origin (test-reviewer Gap 2): same hand-derivation translated
    # by theta_origin. Catches a sign error or wrong subtraction on theta_origin
    # that the theta_origin=0 cases above can't see.
    theta_origin_shifted = 100.0
    theta_shifted = theta + theta_origin_shifted  # preserve κ = theta - theta_origin = 500
    u_tail_shifted = kappa / r_c_apex
    full_shifted = integrate_fan_spatial_exact(
        theta_origin_shifted, v_origin, 0.0, 50.0, theta_shifted, sorption, c_apex=c_apex
    )
    # κ is invariant under (theta, theta_origin) translation; integral matches the theta_origin=0 case.
    assert np.isclose(full_shifted, full, rtol=1e-14), (
        f"theta_origin>0 case (u_tail={u_tail_shifted}): got {full_shifted}, expected {full}"
    )


def test_integrate_rarefaction_exact_passes_c_tail_as_c_apex():
    """The ``integrate_rarefaction_exact`` wrapper plumbs ``c_apex=raref.c_tail``.

    Closes test-reviewer Gap 3: the wrapper at output.py:541 passes
    ``c_apex=raref.c_tail`` to ``integrate_fan_exact``. A mutation that
    silently drops the wiring (``c_apex=0.0``) would break c_tail>0 cases
    end-to-end but is invisible to the existing canonical c_tail=0
    rarefaction tests. This synthetic test exercises the wrapper directly.

    For a Freundlich n=2 rarefaction with c_tail=1.0, c_head=4.0
    spanning v=0 to v=v_outlet=10, the fan integral at v_outlet over
    [θ_start, θ_end] with θ_end well past θ_tail must include the
    constant c_tail contribution past θ_tail. Without the
    ``c_apex=raref.c_tail`` wiring, the wrapper would extrapolate the
    fan formula and underestimate the integral.
    """
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    raref = RarefactionWave(theta_start=0.0, v_start=0.0, c_head=4.0, c_tail=1.0, sorption=sorption)
    v_outlet = 10.0

    # θ_tail at v_outlet for c_tail=1: θ_tail = 0 + 10·R(1) = 10·26 = 260.
    r_c_tail = float(sorption.retardation(raref.c_tail))
    theta_tail_expected = raref.theta_start + (v_outlet - raref.v_start) * r_c_tail
    assert np.isclose(theta_tail_expected, 260.0)

    # Past θ_tail: wrapper must equal explicit-c_apex call. Diverges if c_apex
    # wiring drops to 0.
    wrapper_result = integrate_rarefaction_exact(raref, v_outlet, 100.0, 500.0, sorption)
    explicit_result = integrate_fan_exact(
        raref.theta_start, raref.v_start, v_outlet, 100.0, 500.0, sorption, c_apex=raref.c_tail
    )
    assert np.isclose(wrapper_result, explicit_result, rtol=1e-14), (
        f"wrapper plumbing: got {wrapper_result}, expected {explicit_result}"
    )

    # And the wrapper must NOT equal the c_apex=0 default (which would mean the
    # wiring was dropped). Empirical diff for these params is ≈ 117.55 = 240
    # (constant c_apex·Δθ past θ_tail) minus the fan_integral over [θ_tail, 500]
    # that the c_apex=0 path includes but the c_apex>0 path clamps away. Sign
    # and order-of-magnitude check suffice to catch the wiring-drop mutation.
    default_result = integrate_fan_exact(raref.theta_start, raref.v_start, v_outlet, 100.0, 500.0, sorption)
    assert wrapper_result > default_result, (
        f"wrapper without c_apex wiring would underestimate; got wrapper={wrapper_result}, default={default_result}"
    )
    assert wrapper_result - default_result > 100.0, (
        f"wrapper-vs-default diff too small to confirm c_apex wiring; got {wrapper_result - default_result}"
    )


@pytest.mark.parametrize(
    "sorption",
    [
        FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3),
        FreundlichSorption(k_f=0.05, n=0.5, bulk_density=1200.0, porosity=0.35),
        LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3),
    ],
)
def test_universal_fan_integrators_match_mpmath(sorption):
    """The single universal IBP integrator equals high-precision quadrature for every isotherm.

    All sorptions route through ``_integrate_fan_exact_universal`` /
    ``_integrate_rarefaction_spatial_universal``. This pins the unified path against an
    independent ``mpmath`` (50-digit) reference for Freundlich (n>1 and n<1) and Langmuir,
    so a regression in the universal kernel ``R·c − C_T`` is caught directly (not only via
    downstream mass-balance tests).
    """
    mp.mp.dps = 50
    v_outlet, v_origin, theta_origin = 10.0, 0.0, 0.0
    dv = v_outlet - v_origin

    # Fan concentration bounds chosen well inside each isotherm's range; derive the θ window
    # from the bounds so the geometry is valid for n<1 and Langmuir alike.
    c_head, c_tail = 8.0, 0.5
    r_head = float(sorption.retardation(c_head))
    r_tail = float(sorption.retardation(c_tail))
    theta_a = theta_origin + dv * min(r_head, r_tail)
    theta_b = theta_origin + dv * max(r_head, r_tail)

    # mpmath reference: ∫ c(θ) dθ via 50-digit quadrature of the self-similar fan,
    # c(θ) = concentration_from_retardation((θ − θ_origin)/Δv).
    def c_of_theta(theta):
        r = (float(theta) - theta_origin) / dv
        return mp.mpf(float(sorption.concentration_from_retardation(r)))

    ref_temporal = mp.quad(c_of_theta, [theta_a, theta_b])
    got_temporal = integrate_fan_exact(theta_origin, v_origin, v_outlet, theta_a, theta_b, sorption)
    assert abs(got_temporal - float(ref_temporal)) <= 1e-11 * abs(float(ref_temporal)), (
        f"temporal: universal={got_temporal}, mpmath={float(ref_temporal)}"
    )

    # Spatial fan integral at fixed θ: ∫ C_T(c(u)) du, u from κ/r_tail to κ/r_head.
    theta_fixed = theta_b
    kappa = theta_fixed - theta_origin
    u_lo = kappa / max(r_head, r_tail)
    u_hi = kappa / min(r_head, r_tail)

    def ct_of_u(u):
        c = float(sorption.concentration_from_retardation(kappa / float(u)))
        return mp.mpf(float(sorption.total_concentration(c)))

    ref_spatial = mp.quad(ct_of_u, [u_lo, u_hi])
    got_spatial = integrate_fan_spatial_exact(theta_origin, v_origin, u_lo, u_hi, theta_fixed, sorption)
    assert abs(got_spatial - float(ref_spatial)) <= 1e-11 * abs(float(ref_spatial)), (
        f"spatial: universal={got_spatial}, mpmath={float(ref_spatial)}"
    )


@pytest.mark.parametrize(
    "sorption",
    [
        FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3),
        LangmuirSorption(s_max=0.1, k_l=5.0, bulk_density=1500.0, porosity=0.3),
    ],
)
def test_universal_spatial_fan_apex_matches_mpmath(sorption):
    """Spatial fan integral from the apex (u_start=0) equals high-precision quadrature.

    A fan segment whose lower bound is the apex (``v_start = v_origin``, so
    ``u_start = 0``) contributes ``G(u_end) − G(0)`` with ``G(0) = 0`` (the apex
    carries ``c = 0`` hence ``C_total = 0``). This pins that boundary value
    directly for both Freundlich (n>1) and Langmuir against an independent
    ``mpmath`` reference, so a regression in the apex handling is caught without
    relying on the Langmuir lower-bound clamp invariant.
    """
    mp.mp.dps = 50
    kappa = 4500.0
    # Fan spans c=0 at the apex (u→0, R→∞) to c_head at u_end = κ/R(c_head).
    u_end = kappa / float(sorption.retardation(2.0))
    # c clamps to 0 below u_zero = κ/R(0) where R(0) is the maximal retardation;
    # pass it as a quadrature breakpoint when interior so mpmath resolves the kink.
    u_zero = kappa / float(sorption.retardation(0.0))
    breakpoints = [0.0, u_zero, u_end] if 0.0 < u_zero < u_end else [0.0, u_end]

    def ct_of_u(u):
        c = float(sorption.concentration_from_retardation(kappa / float(u)))
        return mp.mpf(float(sorption.total_concentration(c)))

    ref = mp.quad(ct_of_u, breakpoints)
    # v_start = v_origin = 0 routes through the u_start ≤ 0 apex branch.
    got = integrate_fan_spatial_exact(0.0, 0.0, 0.0, u_end, kappa, sorption)
    assert abs(got - float(ref)) <= 1e-11 * abs(float(ref)), f"spatial apex: integrator={got}, mpmath={float(ref)}"


def test_freundlich_n_below_1_plus_inf_fan_integral_raises():
    """Freundlich n<1 fan integral diverges at θ=+∞; the universal integrator rejects it.

    Preserves the divergence guard that the dedicated integrator enforced. (Production never
    hits this path — DecayingShockWave.mass_after_outlet_arrival returns 0 for the n<1 mirror
    before calling — but the guard protects a direct caller.)
    """
    sorption = FreundlichSorption(k_f=0.05, n=0.5, bulk_density=1200.0, porosity=0.35)
    with pytest.raises(ValueError, match="diverges"):
        integrate_fan_exact(0.0, 0.0, 10.0, 100.0, float("inf"), sorption)
