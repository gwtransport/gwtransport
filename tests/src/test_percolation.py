"""
Unit and integration tests for :mod:`gwtransport.percolation`.

Tests cover:

- Section A: known-answer constitutive ground-truth (hand-derived literals, never
  recomputed from the class under test) — anchors BC/vG against a Mualem→Burdine
  exponent swap and a dropped Mualem square.
- Section B: end-to-end percolation with BC and vG (empty, first-arrival, rarefaction,
  pure-shock breakthrough, mass-balance).
- Section C: K-scaling identity, time-rescaling, validation.
- Section D: vG characteristic-speed monotonicity (property test).
- Section E: smoke (multi-column, perf budget).
- Section M: missing-test additions (dry days, saturation, idempotence, sign-flip invariance, two-step ramp).
- Section N: degenerate-shape inputs (scalar 0-d outlet pore volume, single input bin).
- Section P: transient global mass balance (BC closed-form, vG exact ``L=0`` and default ``L=0.5`` branches).

Tolerances: machine precision for Brooks-Corey closed-form paths; brentq-bounded
for vG; explicitly relaxed where physical reasoning permits (e.g. mass balance
under the approximately-mass-conserving fan-shock fallback regime).

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import BrooksCoreyConductivity, VanGenuchtenMualemConductivity
from gwtransport.fronttracking.output import (
    compute_domain_mass,
    concentration_at_point,
)
from gwtransport.fronttracking.waves import DecayingShockWave, RarefactionWave
from gwtransport.percolation import root_zone_to_water_table_kinematic_wave

# Soil O05 (coarse sand) parameters — used as the default test fixture.
O05 = {"theta_r": 0.01, "theta_s": 0.337, "k_s": 0.174, "brooks_corey_lambda": 0.25}
O05_VG = {"theta_r": 0.01, "theta_s": 0.337, "k_s": 0.174, "van_genuchten_n": 2.28}


def _make_tedges(n_days: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n_days + 1, freq="D")


# =============================================================================
# Section A — Known-answer constitutive ground-truth
# =============================================================================
#
# Every literal below is hand-derived (independent CAS / mpmath at 50 digits)
# from the closed-form constitutive curves — NEVER produced by running the
# class under test. The Brooks-Corey curve uses the Mualem exponent
# ``a = 3 + 2/λ``; for soil O05 (λ = 0.25) this is ``a = 11``. The Burdine
# variant ``a = 2 + 3/λ = 14`` gives DIFFERENT numbers (recorded below as the
# discriminator), so these tests fail under a Mualem→Burdine swap. The vG
# ``_k_se`` literals fail under a dropped Mualem square (``u·u → u``).


class TestConstitutiveGroundTruth:
    """A-series: hardcoded known-answer literals for BC and vG constitutive curves."""

    def test_a1_bc_total_concentration_known_answer(self):
        """A1: ``BrooksCoreyConductivity.total_concentration(0.002)`` equals the hand-derived literal.

        ``C_T(C) = Δθ · (C/k_s)^(1/a)`` with ``Δθ = 0.327``, ``k_s = 0.174``,
        ``a = 3 + 2/λ = 11`` (Mualem). Evaluated at ``C = 0.002`` this is
        ``0.327 · (0.002/0.174)^(1/11) = 0.21788524470301235`` (mpmath, 50 digits).
        The Burdine exponent ``a = 14`` would give ``0.2376898645655214`` instead,
        so this literal distinguishes Mualem from Burdine.
        """
        sorption = BrooksCoreyConductivity(**O05)
        # Hand-derived literal (Mualem a=11); Burdine a=14 would give 0.2376898645655214.
        np.testing.assert_allclose(sorption.total_concentration(0.002), 0.21788524470301235, rtol=1e-13)

    def test_a2_bc_retardation_known_answer(self):
        """A2: ``BrooksCoreyConductivity.retardation(0.002)`` equals the hand-derived literal.

        ``R(C) = (Δθ / (a·k_s)) · (C/k_s)^(1/a − 1)`` with ``a = 11`` (Mualem).
        At ``C = 0.002`` this is ``9.903874759227834`` (mpmath, 50 digits).
        """
        sorption = BrooksCoreyConductivity(**O05)
        np.testing.assert_allclose(sorption.retardation(0.002), 9.903874759227834, rtol=1e-13)

    def test_a3_bc_shock_speed_known_answer(self):
        """A3: ``BrooksCoreyConductivity.shock_speed`` between two nonzero levels equals the literal.

        Rankine-Hugoniot ``(C₂ − C₁)/(C_T(C₂) − C_T(C₁))`` with
        ``C₁ = 0.05·k_s = 0.0087``, ``C₂ = 0.3·k_s = 0.0522`` and the Mualem
        ``a = 11`` total-concentration curve evaluates to ``0.9873688299795546``
        (mpmath, 50 digits). This is a chord speed, distinct from either
        single-side characteristic speed.
        """
        sorption = BrooksCoreyConductivity(**O05)
        c1 = 0.05 * O05["k_s"]
        c2 = 0.3 * O05["k_s"]
        np.testing.assert_allclose(sorption.shock_speed(c1, c2), 0.9873688299795546, rtol=1e-13)

    def test_a4_vg_k_se_known_answers(self):
        """A4: ``VanGenuchtenMualemConductivity._k_se`` at fixed ``S_e`` equals hand-derived literals.

        ``K_M(S_e) = k_s · S_e^L · [1 − (1 − S_e^{1/m})^m]^2`` with ``k_s = 0.174``,
        ``L = 0.5`` (default Mualem), ``m = 1 − 1/n_vG`` and ``n_vG = 2.28``.
        Literals at ``S_e ∈ {0.3, 0.5, 0.7}`` from mpmath (50 digits). Dropping
        the Mualem square (``u·u → u``) would give {0.00644, 0.02160, 0.05027}
        instead, so these literals catch that mutation.
        """
        sorption = VanGenuchtenMualemConductivity(**O05_VG)
        # Hand-derived literals (Mualem, squared term). Dropped-square would give
        # 0.00643692421945469 / 0.0215963614450827 / 0.050269334712603586.
        expected = {0.3: 0.00043475733403324765, 0.5: 0.0037907655426169216, 0.7: 0.017358332655388515}
        for se, k_expected in expected.items():
            np.testing.assert_allclose(sorption._k_se(se), k_expected, rtol=1e-12)  # noqa: SLF001

    def test_a5_vg_k_se_burdine_branch_known_answers(self):
        """A5: ``_k_se`` with ``L = 0`` (Burdine/exact branch) equals hand-derived literals.

        Same curve with ``L = 0``: ``K_M(S_e) = k_s · [1 − (1 − S_e^{1/m})^m]^2``.
        Literals at ``S_e ∈ {0.3, 0.5, 0.7}`` from mpmath (50 digits). Pins the
        root-finding-free branch used by the vG ``mualem_l = 0`` mass-balance test.
        """
        sorption = VanGenuchtenMualemConductivity(
            theta_r=O05_VG["theta_r"],
            theta_s=O05_VG["theta_s"],
            k_s=O05_VG["k_s"],
            van_genuchten_n=O05_VG["van_genuchten_n"],
            mualem_l=0.0,
        )
        expected = {0.3: 0.0007937546629693939, 0.5: 0.005360952042145455, 0.7: 0.020747175800063807}
        for se, k_expected in expected.items():
            np.testing.assert_allclose(sorption._k_se(se), k_expected, rtol=1e-12)  # noqa: SLF001

    def test_a6_vg_total_concentration_public_round_trip(self):
        """A6: the PUBLIC ``total_concentration`` round-trips ``_k_se`` to ``Δθ · S_e``.

        A4/A5 pin the private ``_k_se`` curve shape; this anchors the public path
        so a bug confined to the ``total_concentration`` wrapper (e.g. a wrong
        ``Δθ`` multiply) is also caught. By construction ``C_T(C) = Δθ · S_e(C)``
        with ``S_e`` the inverse of ``_k_se``, so feeding ``C = _k_se(S_e)`` back
        through ``total_concentration`` must return ``Δθ · S_e`` exactly. With
        ``Δθ = θ_s − θ_r = 0.327`` the hand-derived literals at
        ``S_e ∈ {0.3, 0.5, 0.7}`` are ``{0.0981, 0.1635, 0.2289}``. Tolerance is
        brentq-bounded (the inverse ``S_e(C)`` uses brentq).
        """
        sorption = VanGenuchtenMualemConductivity(**O05_VG)
        delta_theta = O05_VG["theta_s"] - O05_VG["theta_r"]
        # Δθ · S_e hand-derived literals (Δθ = 0.327).
        expected = {0.3: 0.0981, 0.5: 0.1635, 0.7: 0.2289}
        for se, ct_expected in expected.items():
            c = sorption._k_se(se)  # noqa: SLF001
            np.testing.assert_allclose(sorption.total_concentration(c), ct_expected, rtol=1e-11)
            np.testing.assert_allclose(ct_expected, delta_theta * se, rtol=1e-13)


# =============================================================================
# Section B — End-to-end percolation
# =============================================================================


class TestEndToEnd:
    """B-series tests: empty input, first-arrival, rarefaction, mass balance."""

    def test_b1_empty_input_gives_zero_output(self):
        """B1: ``q_root_zone ≡ 0`` ⇒ ``q_water_table ≡ 0`` (both BC and vG)."""
        tedges = _make_tedges(60)
        q_zero = np.zeros(60)
        v_out = np.array([O05["theta_s"] * 1.0])
        q_wt_bc, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_zero,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
        )
        q_wt_vg, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_zero,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05_VG,
        )
        np.testing.assert_array_equal(q_wt_bc, np.zeros(60))
        np.testing.assert_array_equal(q_wt_vg, np.zeros(60))

    def test_b3_first_arrival_analytic_bc(self):
        """B3 (BC): the closed-form first-arrival DIAGNOSTIC matches the analytic shock formula.

        Checks ``structures[0]["theta_first_arrival"]`` — the value returned by
        :func:`gwtransport.fronttracking.math.compute_first_front_arrival_theta`,
        a closed-form *diagnostic* of when ``c_first`` reaches the outlet. This is
        NOT the solver's tracked wetting-front (an actual ``ShockWave``); the
        pure-shock end-to-end breakthrough that exercises the tracked front is
        covered by ``test_b6_constant_flux_pure_shock_known_answer_bc``. The
        diagnostic equals ``V_out · C_T(q0) / q0`` with ``C_T`` from Brooks-Corey
        (value-anchored by Section A / B6 — this first-arrival check is a geometric
        diagnostic, not the constitutive ground truth, which lives in those tests).
        """
        tedges = _make_tedges(400)
        q0 = 0.002  # m/day
        z_wt = 0.5  # m
        v_out = np.array([O05["theta_s"] * z_wt])
        _, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=np.full(400, q0),
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
        )
        sorption = BrooksCoreyConductivity(
            theta_r=O05["theta_r"],
            theta_s=O05["theta_s"],
            k_s=O05["k_s"],
            brooks_corey_lambda=O05["brooks_corey_lambda"],
        )
        # Analytic shock arrival in (V, θ_V): V_out · C_T(q0) / q0.
        # C_T(q0) = θ_m(q0) − θ_r = Δθ · (q0/k_s)^(1/a).
        ct = sorption.total_concentration(q0)
        theta_v_arrival = v_out[0] * ct / q0
        np.testing.assert_allclose(structures[0]["theta_first_arrival"], theta_v_arrival, rtol=1e-13)

    def test_b3_first_arrival_analytic_vg(self):
        """B3' (vG): the closed-form first-arrival DIAGNOSTIC matches the analytic shock formula.

        As in ``test_b3_first_arrival_analytic_bc`` this checks the closed-form
        ``theta_first_arrival`` diagnostic (``compute_first_front_arrival_theta``),
        not the solver's tracked front. vG ``C_T`` requires a brentq inversion, so
        the tolerance is brentq-bounded. Value-anchored by Section A / B7 — this
        first-arrival check is a geometric diagnostic, not the constitutive ground
        truth, which lives in those tests.
        """
        tedges = _make_tedges(400)
        q0 = 0.002
        z_wt = 0.5
        v_out = np.array([O05_VG["theta_s"] * z_wt])
        _, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=np.full(400, q0),
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05_VG,
        )
        sorption = VanGenuchtenMualemConductivity(
            theta_r=O05_VG["theta_r"],
            theta_s=O05_VG["theta_s"],
            k_s=O05_VG["k_s"],
            van_genuchten_n=O05_VG["van_genuchten_n"],
        )
        ct = sorption.total_concentration(q0)
        theta_v_arrival = v_out[0] * ct / q0
        np.testing.assert_allclose(structures[0]["theta_first_arrival"], theta_v_arrival, rtol=1e-12)

    def test_b4_rarefaction_self_similar_profile(self):
        """B4: the drying-step fan's interior matches the self-similar law ``R(c) = κ/u``.

        After a wetting-then-drying sequence the drying rarefaction is caught
        by the leading wetting shock and consumed into a ``DecayingShockWave``
        (the exact post-collision behaviour — the parent ``RarefactionWave`` is
        deactivated at the collision). The DSW inherits the rarefaction's apex
        ``(v_origin, theta_origin)`` and exposes the identical self-similar fan
        profile via ``DecayingShockWave.concentration_at_point``. Sampling its
        fan-interior at a θ inside the DSW's active lifetime, verify
        ``sorption.retardation(c) * (v - v_origin) == θ - θ_origin`` to machine
        precision.
        """
        n = 150
        tedges = _make_tedges(n)
        q_root = np.zeros(n)
        q_root[:30] = 0.003  # wet phase: emits a wetting-front shock
        q_root[30:] = 0.0005  # dry phase: emits a rarefaction (n>1 ⇒ higher c is faster ⇒ slow tail of low c)
        # Long-enough column that the rarefaction is fully resolved inside.
        v_out_val = O05["theta_s"] * 5.0
        _, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=np.array([v_out_val]),
            **O05,
        )
        state = structures[0]["tracker_state"]
        sorption = state.sorption
        # The drying rarefaction is consumed into a DecayingShockWave that
        # carries the self-similar profile. Its parent rarefaction is
        # deactivated at the collision.
        dsws = [w for w in state.waves if isinstance(w, DecayingShockWave)]
        assert dsws, "Expected a DecayingShockWave subsuming the drying-step rarefaction"
        dsw = dsws[0]
        assert dsw.decay_side == "left"
        # Probe a θ strictly inside the DSW's active lifetime so its fan has
        # spatial extent and concentration_at_point queries are live.
        theta_lo = dsw.theta_start
        theta_hi = dsw.theta_deactivation if np.isfinite(dsw.theta_deactivation) else dsw.theta_start * 2.0
        theta_probe = theta_lo + 0.5 * (theta_hi - theta_lo)
        # Fan-interior spans (upstream of V_s) from the shock face (c_decay) to
        # the fan's far boundary (c_fan_tail). Sample between the corresponding
        # self-similar positions, excluding the tight endpoints to avoid clipping.
        c_decay = dsw.c_decay_at_theta(theta_probe)
        assert c_decay is not None
        v_face = dsw.v_origin + (theta_probe - dsw.theta_origin) / float(sorption.retardation(c_decay))
        v_tail = dsw.v_origin + (theta_probe - dsw.theta_origin) / float(sorption.retardation(dsw.c_fan_tail))
        v_lo, v_hi = min(v_face, v_tail), max(v_face, v_tail)
        v_samples = np.linspace(v_lo + 0.05 * (v_hi - v_lo), v_hi - 0.05 * (v_hi - v_lo), 8)
        for v in v_samples:
            c = dsw.concentration_at_point(float(v), theta_probe)
            assert c is not None, f"DSW fan did not return c at interior point v={v}"
            r_from_sorption = float(sorption.retardation(c))
            r_from_self_similar = (theta_probe - dsw.theta_origin) / (v - dsw.v_origin)
            # Self-similar relation R(c) = (θ - θ_origin) / (v - v_origin) — machine precision for BC closed form.
            np.testing.assert_allclose(r_from_sorption, r_from_self_similar, rtol=1e-13)

    @staticmethod
    def _assert_bare_rarefaction_self_similar(sorption_kwargs, rtol, *, n=120, wet_days=30, mult=50.0):
        """Probe an uncollided ``RarefactionWave`` and assert ``R(c)·(v − v_start) == θ − θ_start``.

        Uses a long column so the drying-step rarefaction never reaches the
        outlet, and probes a θ strictly *before* the wetting shock collides with
        the fan (so the wave is still a bare :class:`RarefactionWave`, not the
        post-collision DSW that B4 inspects). The fan apex of a bare rarefaction
        is ``(v_start, theta_start)``.

        Parameters
        ----------
        sorption_kwargs : dict
            Constitutive-model keyword arguments forwarded to the solver.
        rtol : float
            Relative tolerance for the self-similar identity (machine precision
            for the Brooks-Corey closed form, brentq-bounded for van Genuchten).
        n : int, optional
            Total number of daily input bins. Default 120. A shorter drying tail
            (smaller ``n``) keeps the vG caller cheap while still producing a
            finitely-deactivating bare rarefaction.
        wet_days : int, optional
            Number of leading wet days (the wetting-front shock that later
            collides with the fan). Default 30.
        mult : float, optional
            Column-length multiplier ``v_out = theta_s · mult``. Default 50.0
            (long enough that the fan never reaches the outlet before collision).
        """
        tedges = _make_tedges(n)
        q_root = np.zeros(n)
        q_root[:wet_days] = 0.003  # wetting phase
        q_root[wet_days:] = 0.0005  # drying phase → emits a rarefaction
        v_out_val = sorption_kwargs["theta_s"] * mult  # long column: fan never reaches the outlet
        with warnings.catch_warnings():
            # The long column pushes some output θ-bins past the inlet window; the
            # resulting back-transform warning is incidental to this wave-geometry
            # probe (which reads the wave list directly, not the bin-average).
            warnings.simplefilter("ignore")
            _, structures = root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=np.array([v_out_val]),
                **sorption_kwargs,
            )
        state = structures[0]["tracker_state"]
        sorption = state.sorption
        rarefactions = [w for w in state.waves if isinstance(w, RarefactionWave)]
        assert rarefactions, "Expected a bare RarefactionWave from the drying step"
        rw = rarefactions[0]
        # Probe strictly inside [theta_start, theta_deactivation): the fan has
        # spatial extent and has not yet been consumed by the wetting shock.
        assert np.isfinite(rw.theta_deactivation), "Rarefaction should deactivate at the shock collision"
        theta_probe = rw.theta_start + 0.5 * (rw.theta_deactivation - rw.theta_start)
        v_tail = rw.tail_position_at_theta(theta_probe)
        v_head = rw.head_position_at_theta(theta_probe)
        assert v_tail is not None
        assert v_head is not None
        v_samples = np.linspace(v_tail + 0.05 * (v_head - v_tail), v_head - 0.05 * (v_head - v_tail), 8)
        for v in v_samples:
            c = rw.concentration_at_point(float(v), theta_probe)
            assert c is not None, f"bare rarefaction did not return c at interior point v={v}"
            r_from_sorption = float(sorption.retardation(c))
            r_from_self_similar = (theta_probe - rw.theta_start) / (v - rw.v_start)
            np.testing.assert_allclose(r_from_sorption, r_from_self_similar, rtol=rtol)

    def test_b4b_bare_rarefaction_self_similar_bc(self):
        """B4b (BC): an uncollided ``RarefactionWave`` obeys ``R(c)·(v − v_start) == θ − θ_start``.

        Complements B4 (which inspects the post-collision DSW): here the fan is
        probed before any collision, on a long column. Machine precision for the
        Brooks-Corey closed form.
        """
        self._assert_bare_rarefaction_self_similar(O05, rtol=1e-13)

    def test_b4c_bare_rarefaction_self_similar_vg(self):
        """B4c (vG): the bare-``RarefactionWave`` self-similar law holds for van Genuchten.

        Same probe as B4b with the vG-Mualem curve; tolerance is brentq-bounded
        (the self-similar inversion ``c = K_M^{-1}(Δθ/R)`` uses brentq). Uses a
        short drying tail (``n=25``) — the per-bin vG brentq inversions make a
        long tail expensive (~9 s at ``n=120``); ``n=25`` still produces a
        finitely-deactivating bare rarefaction and catches the ``r_target``
        self-similar mutation.
        """
        self._assert_bare_rarefaction_self_similar(O05_VG, rtol=1e-11, n=25, wet_days=12, mult=50.0)

    def test_b6_constant_flux_pure_shock_known_answer_bc(self):
        """B6 (BC): constant ``q0`` from rest → single wetting shock arriving at a hardcoded θ.

        A constant ``q_root_zone = q0`` from the dry initial state (``θ = θ_r``)
        emits one wetting-front shock; its arrival θ at the outlet is
        ``V_out · C_T(q0) / q0``. With ``q0 = 0.002`` and ``V_out = θ_s · 0.5``
        for soil O05, the hand-derived literal is ``18.35683186622879`` (mpmath,
        50 digits). The Burdine exponent would give ``20.025…`` instead — this
        literal is setup-specific (it pins both ``q0`` and ``V_out``), so the
        breakthrough is a clean 0→q0 step. Confirms the solver's tracked front,
        not just the closed-form diagnostic (which B3 checks).
        """
        n = 400
        tedges = _make_tedges(n)
        q0 = 0.002  # m/day — PINNED (literal below is setup-specific)
        v_out_val = O05["theta_s"] * 0.5  # V_out PINNED
        q_wt, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=np.full(n, q0),
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=np.array([v_out_val]),
            **O05,
        )
        # Hand-derived arrival θ (Mualem a=11); Burdine a=14 would give 20.02537108964518.
        np.testing.assert_allclose(structures[0]["theta_first_arrival"], 18.35683186622879, rtol=1e-13)
        # The breakthrough is a clean 0→q0 step: exactly one partial bin straddling
        # the arrival, zeros before, q0 after.
        partial = np.where((q_wt > 1e-12) & (q_wt < q0 * (1.0 - 1e-9)))[0]
        assert partial.size == 1, f"expected exactly one partial bin in a clean 0→q0 step, got {partial.size}"
        k = int(partial[0])
        np.testing.assert_array_equal(q_wt[:k], np.zeros(k))
        np.testing.assert_allclose(q_wt[k + 1 :], q0, rtol=1e-13)

    def test_b7_constant_flux_pure_shock_known_answer_vg(self):
        """B7 (vG): constant-flux pure-shock arrival for van Genuchten, own hardcoded literal.

        vG analogue of B6 with the same ``q0 = 0.002`` and ``V_out = θ_s · 0.5``.
        The arrival θ is ``V_out · C_T(q0) / q0`` with vG ``C_T(q0) = Δθ · S_e(q0)``
        and ``S_e(q0)`` from ``K_M(S_e) = q0``; the hand-derived literal is
        ``11.875931658164122`` (mpmath, 50 digits). Tolerance brentq-bounded.
        """
        n = 400
        tedges = _make_tedges(n)
        q0 = 0.002  # PINNED
        v_out_val = O05_VG["theta_s"] * 0.5  # PINNED
        q_wt, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=np.full(n, q0),
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=np.array([v_out_val]),
            **O05_VG,
        )
        # Hand-derived vG arrival θ (mpmath, 50 digits).
        np.testing.assert_allclose(structures[0]["theta_first_arrival"], 11.875931658164122, rtol=1e-12)
        partial = np.where((q_wt > 1e-12) & (q_wt < q0 * (1.0 - 1e-9)))[0]
        assert partial.size == 1, f"expected exactly one partial bin in a clean 0→q0 step, got {partial.size}"
        k = int(partial[0])
        np.testing.assert_array_equal(q_wt[:k], np.zeros(k))
        np.testing.assert_allclose(q_wt[k + 1 :], q0, rtol=1e-12)

    def test_b5a_compute_domain_mass_against_analytic_steady_state(self):
        """B5a: ``compute_domain_mass`` against analytic steady-state, *not* the conservation tautology.

        At steady state under constant ``q_root_zone = q0``, the column moisture
        is uniform at ``theta_m(q0)``, so the analytic domain mass is
        ``C_T(q0) * V_out = (theta_m(q0) - theta_r) * V_out``. The framework's
        ``compute_domain_mass`` integrates spatially via the IBP integrator
        (or, for constant regions, directly via ``C_T * dv``); both paths
        must give this exact value.

        This test *directly* exercises ``compute_domain_mass``/the IBP machinery
        — it is NOT the conservation identity ``m_in_domain + m_out = m_in``
        (which is true by construction; reviewer flagged the previous B5a as
        a tautology).
        """
        tedges = _make_tedges(400)
        n = 400
        q0 = 0.003  # m/day, well below k_s for soil O05
        q_root = np.full(n, q0)
        z_wt = 0.3
        v_out_val = O05["theta_s"] * z_wt
        _, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=np.array([v_out_val]),
            **O05,
        )
        sorption = BrooksCoreyConductivity(
            theta_r=O05["theta_r"],
            theta_s=O05["theta_s"],
            k_s=O05["k_s"],
            brooks_corey_lambda=O05["brooks_corey_lambda"],
        )
        # Analytic shock arrival θ_v = V_out * C_T(q0) / q0; the column is at steady state
        # for any θ > θ_arrival. Pick a probe θ well past arrival so the entire column is
        # uniformly at c = q0; compute_domain_mass then takes the constant-region path
        # (no fan, no shock inside the domain).
        ct_q0 = float(sorption.total_concentration(q0))
        theta_arrival = v_out_val * ct_q0 / q0
        state = structures[0]["tracker_state"]
        theta_probe = 2.0 * theta_arrival  # well past arrival

        analytic_domain_mass = ct_q0 * v_out_val  # C_T(q0) * V_out — closed form for BC
        numerical_domain_mass = compute_domain_mass(
            theta=theta_probe, v_outlet=v_out_val, waves=state.waves, sorption=state.sorption
        )
        np.testing.assert_allclose(numerical_domain_mass, analytic_domain_mass, rtol=1e-13)

    @staticmethod
    def _assert_domain_mass_straddling_rarefaction(sorption_kwargs, theta_probe, rtol):
        """Drive a fan that STRADDLES ``v_outlet`` and quantitatively pin the IBP integrator.

        Wet-then-dry forcing emits a bare rarefaction. At ``theta_probe`` the fan
        straddles the outlet (``v_tail < v_outlet < v_head``), so
        :func:`compute_domain_mass` partitions ``[0, v_outlet]`` into a constant
        upstream region ``[0, v_tail]`` at ``c_tail`` plus the fan interior
        ``[v_tail, v_outlet]`` — the latter routed through the universal IBP
        antiderivative ``_integrate_rarefaction_spatial_universal`` (the
        constant region is NOT). The reference is

        ``C_T(c_tail)·v_tail + ∫_{v_tail}^{v_outlet} C_T(c(v)) dv``,

        with the integral an INDEPENDENT trapezoidal rule over
        :func:`concentration_at_point` on a fine v-grid — so it exercises the IBP
        integrator (not just the constant-region ``C_T·Δv`` path).

        Parameters
        ----------
        sorption_kwargs : dict
            Constitutive-model keyword arguments forwarded to the solver.
        theta_probe : float
            Cumulative flow at which the fan straddles the outlet (asserted).
        rtol : float
            Relative tolerance: trapezoid truncation for the closed-form
            Brooks-Corey path, brentq-bounded for van Genuchten.
        """
        n = 200
        tedges = _make_tedges(n)
        q_root = np.zeros(n)
        q_root[:50] = 0.003  # wet phase: wetting-front shock
        q_root[50:] = 0.0005  # drying phase: emits a rarefaction
        v_out_val = sorption_kwargs["theta_s"] * 0.5  # outlet inside the fan's straddle window
        with warnings.catch_warnings():
            # Long-tail forcing pushes some output θ-bins past the inlet window;
            # the resulting back-transform warning is incidental here (we read the
            # wave list / compute_domain_mass directly, not the bin-average).
            warnings.simplefilter("ignore")
            _, structures = root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=np.array([v_out_val]),
                **sorption_kwargs,
            )
        assert structures[0]["n_rarefactions"] >= 1, "Expected a rarefaction after the drying step"
        state = structures[0]["tracker_state"]
        sorption = state.sorption
        rarefactions = [w for w in state.waves if isinstance(w, RarefactionWave) and w.was_active_at(theta_probe)]
        assert rarefactions, "Expected an active bare RarefactionWave at theta_probe"
        rw = rarefactions[0]
        v_tail = rw.tail_position_at_theta(theta_probe)
        v_head = rw.head_position_at_theta(theta_probe)
        assert v_tail is not None
        assert v_head is not None
        # The fan must STRADDLE v_outlet, else the IBP integrator is not exercised.
        assert v_tail < v_out_val < v_head, (
            f"fan must straddle v_outlet={v_out_val}: got v_tail={v_tail}, v_head={v_head} at θ={theta_probe}"
        )

        # Independent reference: constant upstream region at c_tail + trapezoidal
        # integration of the fan interior over [v_tail, v_outlet]. The trapezoid
        # over concentration_at_point is independent of the IBP antiderivative.
        m_constant = float(sorption.total_concentration(rw.c_tail)) * v_tail
        v_grid = np.linspace(v_tail, v_out_val, 4001)
        c_grid = np.array([concentration_at_point(float(v), theta_probe, state.waves, sorption) for v in v_grid])
        ct_grid = np.array([float(sorption.total_concentration(float(c))) for c in c_grid])
        m_fan_ref = float(np.trapezoid(ct_grid, v_grid))
        m_reference = m_constant + m_fan_ref
        # Sanity: the fan term is a non-trivial share of the total (so a κ-mutation
        # in the IBP antiderivative materially shifts the result).
        assert m_fan_ref > 0.1 * m_reference, "fan-interior term is too small to exercise the IBP integrator"

        m_domain = compute_domain_mass(theta=theta_probe, v_outlet=v_out_val, waves=state.waves, sorption=sorption)
        np.testing.assert_allclose(m_domain, m_reference, rtol=rtol)

    def test_b5a_universal_ibp_integrator_active_in_rarefaction_bc(self):
        """B5a-aux (BC): ``compute_domain_mass`` over a straddling fan exercises the IBP integrator.

        At ``θ = 18.7`` the bare drying rarefaction straddles ``v_outlet = θ_s·0.5``,
        so ``compute_domain_mass`` routes ``[v_tail, v_outlet]`` through the
        universal IBP antiderivative ``_integrate_rarefaction_spatial_universal``.
        Compared against an independent ``np.trapezoid`` over
        :func:`concentration_at_point` (4001 points). ``rtol = 1e-6`` accommodates
        the trapezoid truncation; the verified baseline relerr is ≈ 5e-10. A
        ``κ → 0.5·κ`` mutation in the antiderivative ``g = C_T·u − κ·c`` shifts the
        fan term by ~4% and fails this bound.
        """
        self._assert_domain_mass_straddling_rarefaction(O05, theta_probe=18.7, rtol=1e-6)

    def test_b5a_universal_ibp_integrator_active_in_rarefaction_vg(self):
        """B5a-aux (vG): the straddling-fan IBP-integrator check for van Genuchten.

        vG counterpart of the BC test at ``θ = 20.0`` (where the fan straddles
        ``v_outlet = θ_s·0.5``). The fan-interior ``c(v)`` inversion uses brentq,
        so the tolerance is brentq-bounded; the verified baseline relerr is
        ≈ 6e-10 (trapezoid- and brentq-limited), comfortably under ``rtol = 1e-6``.
        """
        self._assert_domain_mass_straddling_rarefaction(O05_VG, theta_probe=20.0, rtol=1e-6)

    def test_b5b_multi_pulse_smoke_no_crash(self):
        """B5b (multi-pulse): smoke test — multi-pulse input doesn't crash the solver.

        Multi-pulse input creates fan-shock collisions that hit the non-canonical
        fallback in ``handlers.py:314-390`` and exercise the closed-form fan
        integrators in ``output.py``. The existing framework's closed-form
        integrators (Freundlich, Langmuir) do not extend to ``BrooksCoreyConductivity``
        or ``VanGenuchtenMualemConductivity``; the generic numerical-quadrature
        fallback added in this module covers single-fan cases but does not yet
        compose correctly through multi-fan collisions (see plan v3 "Out of
        scope §1"). Exact multi-pulse mass balance for BC/vG is a v2 deliverable.

        The exact bin-average is therefore NOT asserted (global mass balance is
        ~18 % off on this fallback regime). What IS asserted is the physical
        envelope every admissible result must satisfy regardless of the fallback's
        approximation: ``0 ≤ q_wt ≤ q_root.max()`` — a flux at the water table can
        neither go negative nor exceed the largest root-zone leakage. A regression
        that scales the multi-fan fallback output (e.g. by 3×) pushes ``q_wt.max()``
        to ~3× ``q_root.max()`` and fails this envelope, so this is no longer a
        pure no-crash smoke test.
        """
        n = 90
        tedges = _make_tedges(n)
        rng = np.random.default_rng(42)
        q_root = np.zeros(n)
        for start in (10, 40, 70):
            q_root[start : start + 3] = rng.uniform(0.0005, 0.0015, 3)
        v_out_val = O05["theta_s"] * 5.0
        with warnings.catch_warnings():
            # The long column pushes some output θ-bins past the inlet window; the
            # resulting out-of-window clamp warning is incidental to this envelope
            # probe (and the multi-fan fallback is not exact here anyway).
            warnings.simplefilter("ignore")
            q_wt, _ = root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=np.array([v_out_val]),
                **O05,
            )
        assert q_wt.shape == (n,)
        assert not np.any(np.isnan(q_wt))
        assert not np.any(np.isinf(q_wt))
        # Physical envelope: 0 ≤ q_wt ≤ q_root.max() (a few ULPs of slack at the
        # upper edge where a bin-average grazes the largest input plateau).
        assert np.all(q_wt >= 0.0), f"q_wt went negative: min={q_wt.min()}"
        upper = float(q_root.max())
        assert np.all(q_wt <= upper + 1e3 * np.finfo(float).eps * upper), (
            f"q_wt exceeded q_root.max()={upper}: max={q_wt.max()}"
        )


# =============================================================================
# Section C — K-scaling
# =============================================================================


class TestKScaling:
    """C-series tests: identity, time-rescaling, validation."""

    def test_c1_identity_scaling_matches_none(self):
        """C1: ``k_scaling = ones`` is bit-for-bit identical to ``k_scaling = None``."""
        tedges = _make_tedges(120)
        q_root = np.full(120, 0.001)
        v_out = np.array([O05["theta_s"] * 0.3])
        q_wt_a, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
        )
        q_wt_b, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
            k_scaling=np.ones(120),
        )
        np.testing.assert_array_equal(q_wt_a, q_wt_b)

    def test_c2_time_rescaling_identity_constant_q(self):
        """C2 (corrected): ``q_wt_B(t) = α · q_wt_A(α·t)`` for matched (A: cin=q₀/α; B: K-scaling α, cin=q₀).

        The cin_solver is identical between A and B (q₀/α), so the solver's
        state in (V, θ_V) coordinates is identical. Only the θ_V → t mapping
        differs by the factor α (since flow_solver_B = α · flow_solver_A).
        Compares first-arrival times (less sensitive to bin discretization
        than amplitude comparison).
        """
        n = 365
        tedges = _make_tedges(n)
        q0 = 0.002
        alpha = 2.0
        z_wt = 0.5
        v_out = np.array([O05["theta_s"] * z_wt])
        _, struct_a = root_zone_to_water_table_kinematic_wave(
            q_root_zone=np.full(n, q0 / alpha),
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
        )
        _, struct_b = root_zone_to_water_table_kinematic_wave(
            q_root_zone=np.full(n, q0),
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
            k_scaling=np.full(n, alpha),
        )
        # In (V, θ_V) space the arrival θ_V is the SAME for A and B (same cin_solver,
        # same sorption). The wall-clock t = θ_V / flow_solver differs by α.
        # Convert: t_A = θ_V_A / θ_s; t_B = θ_V_B / (θ_s · α).
        # Since θ_V_A == θ_V_B, t_B = t_A / α.
        np.testing.assert_allclose(struct_a[0]["theta_first_arrival"], struct_b[0]["theta_first_arrival"], rtol=1e-13)

    @pytest.mark.parametrize(
        "invalid",
        [
            {"k_scaling": np.array([1.0, -0.5, 1.0])},  # negative
            {"k_scaling": np.array([1.0, 0.0, 1.0])},  # zero
            {"k_scaling": np.array([1.0, np.nan, 1.0])},  # NaN
            {"k_scaling": np.array([1.0, 1.0])},  # wrong length (3 bins expected)
        ],
    )
    def test_c4_invalid_k_scaling_raises(self, invalid):
        """C4: validation rejects malformed ``k_scaling``."""
        tedges = _make_tedges(3)
        q_root = np.full(3, 0.001)
        with pytest.raises(ValueError):
            root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=np.array([0.1]),
                **O05,
                **invalid,
            )

    def test_c5_k_scaling_requires_matching_output_grid(self):
        """C5: ``q_water_table_tedges != tedges`` while ``k_scaling`` is set ⇒ ``ValueError``."""
        tedges = _make_tedges(60)
        out_tedges = pd.date_range("2020-01-01", periods=21, freq="3D")
        with pytest.raises(ValueError, match="must equal tedges"):
            root_zone_to_water_table_kinematic_wave(
                q_root_zone=np.full(60, 0.001),
                tedges=tedges,
                q_water_table_tedges=out_tedges,
                cumulative_pore_volumes_outlet=np.array([0.1]),
                **O05,
                k_scaling=np.ones(60),
            )


class TestValidatorBranches:
    """C6: each input-validation branch raises ``ValueError`` with an informative message.

    Covers the eight core validator branches not exercised by C4 (``k_scaling``
    shape/NaN), C5 (output-grid mismatch under ``k_scaling``) or M3
    (saturation/ponding). Each case overrides exactly one keyword of an otherwise
    valid call so the matched message pins the intended branch.
    """

    @staticmethod
    def _valid_kwargs(n=5):
        """Return a fully valid keyword set for an ``n``-bin call (overridable per case)."""
        tedges = _make_tedges(n)
        return {
            "q_root_zone": np.full(n, 0.001),
            "tedges": tedges,
            "q_water_table_tedges": tedges,
            "cumulative_pore_volumes_outlet": np.array([O05["theta_s"] * 0.3]),
            "theta_r": O05["theta_r"],
            "theta_s": O05["theta_s"],
            "k_s": O05["k_s"],
            "brooks_corey_lambda": O05["brooks_corey_lambda"],
        }

    @pytest.mark.parametrize(
        ("override", "match"),
        [
            # 1. tedges length != len(q_root_zone) + 1.
            ({"tedges": _make_tedges(6)}, r"tedges must have length"),
            # 2. output edges reach beyond the input window.
            (
                {"q_water_table_tedges": pd.date_range("2020-01-02", periods=6, freq="D")},
                r"must lie within the input window",
            ),
            # 3. negative q_root_zone.
            ({"q_root_zone": np.array([0.001, -0.001, 0.001, 0.001, 0.001])}, r"non-negative"),
            # 4. NaN in q_root_zone.
            ({"q_root_zone": np.array([0.001, np.nan, 0.001, 0.001, 0.001])}, r"must not contain NaN"),
            # 5. non-positive / empty cumulative_pore_volumes_outlet.
            ({"cumulative_pore_volumes_outlet": np.array([0.1, -0.2])}, r"non-empty with all entries positive"),
            ({"cumulative_pore_volumes_outlet": np.array([])}, r"non-empty with all entries positive"),
            # 6. theta ordering / range violated.
            ({"theta_r": 0.4, "theta_s": 0.337}, r"theta_r, theta_s must satisfy"),
            # 7. non-positive k_s.
            ({"k_s": 0.0}, r"k_s must be positive"),
            # 8. neither / both sorption-parameter groups.
            ({"brooks_corey_lambda": None}, r"Exactly one of"),
            ({"van_genuchten_n": 2.28}, r"Exactly one of"),
        ],
    )
    def test_c6_validator_branch_raises(self, override, match):
        """C6: a single malformed keyword trips the matching validator branch."""
        kwargs = {**self._valid_kwargs(), **override}
        with pytest.raises(ValueError, match=match):
            root_zone_to_water_table_kinematic_wave(**kwargs)


# =============================================================================
# Section D — vG characteristic-speed monotonicity (property test)
# =============================================================================


class TestCharacteristicSpeedMonotonicity:
    """D-series: vG characteristic speeds increase monotonically with θ (property test).

    The previous D2 also asserted a 5×-vs-article band against the article's
    plotted vG curve (``n_vG`` not pinned to the Heinen 2020 Staringreeks row).
    A factor-of-5 "agreement" is not evidence and was dropped (PERC-M4); the
    quantitative ground-truth for the vG curve now lives in the hand-derived
    ``_k_se`` literals (``TestConstitutiveGroundTruth.test_a4_*``). What remains
    here is the genuine structural property: ``V_c(θ) = 1/R(K_vG(θ))`` must be
    strictly monotone increasing in θ.
    """

    def test_d2_characteristic_speeds_monotone_increasing_with_theta(self):
        """D2: ``V_c(θ) = 1/R(K_vG(θ))`` is strictly increasing in θ (wetter soil drains faster).

        Pure monotonicity property — no article-band comparison. Spans a wide θ
        range so the check is non-trivial; wetter soil (higher θ, higher K)
        must have a faster characteristic speed.
        """
        sorption = VanGenuchtenMualemConductivity(**O05_VG)
        theta_values = np.array([0.04, 0.06, 0.09, 0.11, 0.15, 0.20])  # increasing θ
        s_e = (theta_values - O05_VG["theta_r"]) / (O05_VG["theta_s"] - O05_VG["theta_r"])
        k_values = np.array([sorption._k_se(float(s)) for s in s_e])  # noqa: SLF001
        v_c = 1.0 / sorption.retardation(k_values)
        assert np.all(np.diff(v_c) > 0), f"characteristic speed V_c = 1/R must increase strictly with θ; got {v_c}"


# =============================================================================
# Section E — Smoke / regression
# =============================================================================


class TestSmoke:
    """E-series smoke tests: multi-column averaging."""

    def test_e1_multi_column_arithmetic_mean(self):
        """E1: result equals the arithmetic mean of single-column runs."""
        tedges = _make_tedges(60)
        q_root = np.full(60, 0.002)
        v_outs = np.array([0.5, 1.0, 1.5]) * O05["theta_s"]

        q_wt_multi, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_outs,
            **O05,
        )

        singles = []
        for v in v_outs:
            q_wt_single, _ = root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=np.array([v]),
                **O05,
            )
            singles.append(q_wt_single)
        expected = np.mean(np.stack(singles), axis=0)
        np.testing.assert_array_equal(q_wt_multi, expected)


# =============================================================================
# Section M — Additional tests from reviewer findings
# =============================================================================


class TestMissingCoverage:
    """M-series tests: dry days, saturation, idempotence, sign-flip invariance, two-step ramp."""

    def test_m2_dry_days_handled(self):
        """M2: dry intervals (q=0) interspersed with positive q → no NaN/inf, physical bounds hold.

        Constructs a wet-then-dry-then-wet sequence on a short column so the
        signal exits *inside* the inlet θ-window (PERC-M2 regime). Asserts the
        physical bounds ``0 ≤ q_wt ≤ q_root.max()`` (a flux at the water table can
        neither go negative nor exceed the largest root-zone leakage), not merely
        the absence of NaN/inf. The clamp-to-zero ``UserWarning`` is exercised
        separately by ``test_m2b_no_clamp_warning_inside_inlet_window``.
        """
        tedges = _make_tedges(60)
        q_root = np.full(60, 0.001)
        q_root[20:30] = 0.0  # dry period (full zero — the genuine dry-days case)
        v_out = np.array([O05["theta_s"] * 0.3])
        # The full-zero dry gap pushes the drying tail past the inlet window,
        # tripping the benign out-of-window clamp warning (the subject of M2b,
        # which uses an in-window forcing). Match it explicitly via pytest.warns
        # rather than blanket-silencing, so any unrelated new warning still
        # surfaces as an error here.
        with pytest.warns(UserWarning, match="clamp threshold"):
            q_wt, _ = root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=v_out,
                **O05,
            )
        assert not np.any(np.isnan(q_wt))
        assert not np.any(np.isinf(q_wt))
        # Physical bounds: 0 ≤ q_wt ≤ q_root.max(). The upper edge can sit a few
        # ULPs above q_root.max() where the bin-average grazes the steady plateau
        # (measured overshoot ≈ 1.3e-17 for q_root.max()=1e-3); a scaled
        # machine-epsilon slack rejects any 1e-12-scale spurious super-source.
        assert np.all(q_wt >= 0.0), f"q_wt went negative: min={q_wt.min()}"
        upper = float(q_root.max())
        assert np.all(q_wt <= upper + 1e3 * np.finfo(float).eps * upper), (
            f"q_wt exceeded q_root.max()={upper}: max={q_wt.max()}"
        )

    def test_m2b_no_clamp_warning_inside_inlet_window(self):
        """M2b: a wet-then-dry run *inside* the inlet θ-window emits no clamp-to-zero ``UserWarning``.

        The benign FP guard at ``advection.py`` clamps tiny negative bin-averages
        to zero and warns when output θ-bins exceed the inlet window (inconsistent
        θ ranges). Per the physics review this is NOT a code bug and must not be
        NaN-masked; the fix is to keep the response inside the inlet window. This
        test pins that contract: a wet-then-dry sequence on a short column,
        executed under ``warnings.simplefilter("error")``, must complete without
        raising any ``UserWarning``.
        """
        n = 200
        tedges = _make_tedges(n)
        q_root = np.zeros(n)
        q_root[:30] = 0.003  # wet phase
        q_root[30:] = 0.0008  # dry phase (small but positive → column stays wet, θ bounded)
        v_out = np.array([O05["theta_s"] * 0.2])  # short column: signal exits within the inlet window
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning becomes an exception
            root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=v_out,
                **O05,
            )

    @pytest.mark.parametrize(
        ("f_val", "q_factor", "should_raise"),
        [
            (1.5, 1.2, False),  # f · k_s = 1.5·k_s > 1.2·k_s ⇒ admissible
            (0.5, 0.6, True),  # f · k_s = 0.5·k_s < 0.6·k_s ⇒ rejected
            (1.0, 1.1, True),  # plain ponding: q > k_s with no scaling
            (1.0, 1.0, False),  # edge: q = k_s exactly is admissible
        ],
    )
    def test_m3_saturation_admissibility(self, f_val, q_factor, should_raise):
        """M3: validator enforces ``q_root ≤ f · k_s`` (parametrised accept/reject)."""
        tedges = _make_tedges(10)
        n = 10
        q_root = np.full(n, q_factor * O05["k_s"])
        k_scaling = np.full(n, f_val)
        kwargs = {} if f_val == 1.0 else {"k_scaling": k_scaling}
        if should_raise:
            with pytest.raises(ValueError, match=r"saturation|ponding"):
                root_zone_to_water_table_kinematic_wave(
                    q_root_zone=q_root,
                    tedges=tedges,
                    q_water_table_tedges=tedges,
                    cumulative_pore_volumes_outlet=np.array([0.05]),
                    **O05,
                    **kwargs,
                )
        else:
            # Must not raise.
            root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=np.array([0.05]),
                **O05,
                **kwargs,
            )

    def test_m4_idempotence(self):
        """M4: same inputs → same outputs bit-for-bit (no mutable state in sorption classes)."""
        tedges = _make_tedges(60)
        q_root = np.full(60, 0.001)
        v_out = np.array([O05["theta_s"] * 0.5])

        q1, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
        )
        q2, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
        )
        np.testing.assert_array_equal(q1, q2)

    def test_m6_sign_flip_invariance(self):
        """M6: under K-scaling the OUTPUT back-transform ``q_wt = f·cout`` is correctly oriented.

        The percolation wrapper inverts the inlet as ``cin = q/f`` and recovers
        the outlet as ``q_wt = f·cout``. The *inlet* leg (``q/f`` vs ``q·f``) is
        what C2's time-rescaling identity guards (the arrival θ shifts under a
        wrong inlet inversion). This test guards the *output* leg: a bug pairing
        the correct inlet ``cin = q/f`` with a wrong output ``q_wt = cout/f``
        cancels in C2 and in mass balance, but breaks the steady-state amplitude.

        Concrete check: run with constant ``k_scaling = 2.0`` and constant
        ``q_root``. Once the wetting front passes the outlet, ``q_water_table``
        must return to ``q_root_zone`` exactly — steady state is preserved under
        any time-only K-scaling only if the back-transform multiplies (not
        divides) by ``f``.
        """
        n = 365
        tedges = _make_tedges(n)
        q_root_val = 0.002
        z_wt = 0.3  # short enough to reach steady state in 1 year
        v_out = np.array([O05["theta_s"] * z_wt])
        q_wt, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=np.full(n, q_root_val),
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
            k_scaling=np.full(n, 2.0),
        )
        # The last 50 days should be at steady state ≈ q_root_val.
        steady_state = q_wt[-50:].mean()
        np.testing.assert_allclose(steady_state, q_root_val, rtol=1e-13)

    def test_m7_two_step_ramp_shock_speed(self):
        """M7: shock between two non-zero levels uses Rankine-Hugoniot ``ΔK/Δθ``, not ``dK/dθ``.

        Bug-catch: a mis-implemented ``shock_speed`` returning ``dK/dθ`` at θ_2 passes the
        single-step test (where θ_R = θ_r ⇒ shock = (K_2 - 0)/(θ_2 - θ_r) is consistent with
        a chord at θ_2) but fails when θ_R > θ_r (two-step ramp).

        Verifies the shock-speed contract of BrooksCoreyConductivity directly.
        """
        sorption = BrooksCoreyConductivity(**O05)
        # Two arbitrary intermediate concentrations
        c1 = 0.05 * O05["k_s"]
        c2 = 0.3 * O05["k_s"]
        # Closed-form Rankine-Hugoniot in (C, C_T) variables (= same as ΔK/Δθ in θ):
        # shock_speed = (C_2 - C_1) / (C_T(C_2) - C_T(C_1))
        ct1 = sorption.total_concentration(c1)
        ct2 = sorption.total_concentration(c2)
        expected = (c2 - c1) / (ct2 - ct1)
        actual = sorption.shock_speed(c1, c2)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
        # Also confirm: this is NOT just the characteristic speed at either side.
        char_speed_1 = 1.0 / sorption.retardation(c1)
        char_speed_2 = 1.0 / sorption.retardation(c2)
        # Both single-side characteristic speeds differ from the chord/shock speed by O(a) factor.
        assert not np.isclose(actual, char_speed_1, rtol=1e-3)
        assert not np.isclose(actual, char_speed_2, rtol=1e-3)
        # Lax entropy: c_left = c2 (upstream wetter) should satisfy admissibility.
        assert sorption.check_entropy_condition(c2, c1, sorption.shock_speed(c2, c1))


class TestDegenerateInputs:
    """Degenerate-shape end-to-end probes: scalar (0-d) outlet, single input bin."""

    def test_n1_scalar_cumulative_pore_volume_outlet(self):
        """N1: a scalar (0-d) ``cumulative_pore_volumes_outlet`` is promoted to a single column.

        The docstring advertises ``cumulative_pore_volumes_outlet`` as array-like;
        a lone scalar is a valid single column. The validator promotes it via
        ``np.atleast_1d`` so the per-column loop and ``len()`` are well-defined
        (previously a 0-d input crashed with an opaque ``TypeError: len() of
        unsized object``). The scalar result must equal the equivalent
        single-element 1-d call bit-for-bit.
        """
        tedges = _make_tedges(60)
        q_root = np.full(60, 0.001)
        v_scalar = O05["theta_s"] * 0.5
        q_wt_scalar, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_scalar,
            **O05,
        )
        q_wt_1d, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=np.array([v_scalar]),
            **O05,
        )
        assert q_wt_scalar.shape == (60,)
        np.testing.assert_array_equal(q_wt_scalar, q_wt_1d)

    def test_n2_single_input_bin(self):
        """N2: a single input bin (``n = 1``, two ``tedges``) runs and stays physically bounded.

        Exercises the degenerate ``n = 1`` path that the bin-edge / cumulative-flow
        machinery (``flow_tedges_days`` / ``cout_tedges_days`` construction, the
        single-interval ``θ_at_t`` mapping) distinguishes from the multi-bin case.
        A high constant flux over a very short column drives a partial breakthrough
        in the lone bin, so the bound ``0 ≤ q_wt ≤ q0`` is non-trivial (not
        always-zero). Runs under ``simplefilter("error")`` to keep it in-window.
        """
        tedges = _make_tedges(1)
        q0 = 0.1  # high flux, still < k_s = 0.174
        v_out = np.array([O05["theta_s"] * 0.005])  # very short column → partial breakthrough in one day
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            q_wt, _ = root_zone_to_water_table_kinematic_wave(
                q_root_zone=np.array([q0]),
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=v_out,
                **O05,
            )
        assert q_wt.shape == (1,)
        assert np.all(np.isfinite(q_wt))
        assert 0.0 <= q_wt[0] <= q0 * (1.0 + 1e-12), f"single-bin q_wt out of [0, q0]: {q_wt[0]}"
        # Non-trivial: the front partially breaks through (not a degenerate all-zero bin).
        assert q_wt[0] > 0.0, f"expected a partial breakthrough in the single bin, got {q_wt[0]}"


class TestPlottingClaim:
    """Anchors the notebook's plotting claim in a unit test (independent of the notebook)."""

    def test_t2_bin_average_is_flow_weighted_mean_of_exact_curve(self):
        """T2: each ``q_water_table`` bin equals the flow-weighted mean of the exact outlet curve.

        The notebook plots the bin-averaged output as steps and the exact breakthrough as a
        continuous line; the step value over a bin must equal the flow-weighted mean of the
        exact ``concentration_at_point`` curve across that bin. With constant ``flow_solver``
        (no K-scaling), flow-weighting reduces to a time average. Sampling a fine grid and
        comparing to the solver's reported bin value pins this claim at ``rtol=1e-4``
        (trapezoid error across the wetting-front jump inside the chosen bin). The
        sample count is 100001 (verified residual ≈ 1.7e-5, a ~6x margin); a 5e-4
        output error fails this bound.
        """
        tedges = _make_tedges(300)
        q0 = 0.002
        q_root = np.full(300, q0)
        v_out_val = O05["theta_s"] * 0.4
        # Coarse 5-day output bins so each bin spans several solver days (non-trivial average).
        out_tedges = pd.date_range("2020-01-01", periods=61, freq="5D")
        q_wt, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=out_tedges,
            cumulative_pore_volumes_outlet=np.array([v_out_val]),
            **O05,
        )
        state = structures[0]["tracker_state"]
        out_days = ((out_tedges - out_tedges[0]) / pd.Timedelta(days=1)).to_numpy()

        # Find the output bin straddling the wetting-front arrival: 0 < q_wt[k] < q0
        # (a partial average — the exact curve jumps from 0 to q0 inside the bin).
        partial = np.where((q_wt > 1e-9) & (q_wt < q0 * (1.0 - 1e-9)))[0]
        assert partial.size > 0, "expected an output bin straddling the arrival shock"
        k = int(partial[0])

        # Flow-weighted mean of the exact outlet curve over bin k. flow_solver is constant
        # (no scaling) so flow-weighting == time-weighting: mean = (1/Δt)∫ c_exact(t) dt.
        t_lo, t_hi = out_days[k], out_days[k + 1]
        tt = np.linspace(t_lo, t_hi, 100001)
        c_exact = np.array([
            concentration_at_point(v_out_val, state.theta_at_t(float(t)), state.waves, state.sorption) for t in tt
        ])
        mean_exact = np.trapezoid(c_exact, tt) / (t_hi - t_lo)
        np.testing.assert_allclose(q_wt[k], mean_exact, rtol=1e-4)


# =============================================================================
# Section R — Regressions for the shipped fan-shock collision bug
# =============================================================================
#
# Before the DecayingShockWave fix, a Brooks-Corey wetting-then-drying run
# routed every shock<->rarefaction collision through an approximate piecewise
# overlay that never terminated cleanly: the solver chewed through ~10000
# events (hitting max_iterations) and the bin-averaged output overshot the
# exact breakthrough by ~500%. The tests below pin the exact post-fix
# behaviour so the bug cannot silently return.

# Canonical wetting-then-drying Brooks-Corey forcing that drives a
# shock<->rarefaction collision (a DecayingShockWave). 40 wet days then a long
# drying tail; the tail is long enough that the breakthrough at v_out lands
# inside the inlet θ-window so the exact pointwise curve is well-defined there.
_R_Q_ROOT = np.array([0.003] * 40 + [0.0005] * 1200)


class TestShippedCollisionBugRegression:
    """R-series: regressions that would have caught the shipped collision bug."""

    def _run_canonical(self, q_root=None):
        q = _R_Q_ROOT if q_root is None else q_root
        n = len(q)
        tedges = _make_tedges(n)
        v_out = np.array([O05["theta_s"]])  # 0.337
        return root_zone_to_water_table_kinematic_wave(
            q_root_zone=q,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=v_out,
            **O05,
        )

    def test_r1_solver_converges_no_max_iterations_warning(self, caplog):
        """R1: the canonical BC wetting-then-drying run converges (no max_iterations warning).

        The shipped bug logged ``logging.warning("Reached max_iterations=...")``
        from ``gwtransport.fronttracking.solver`` because the approximate
        overlay produced an unterminating cascade of collision events. The
        exact DecayingShockWave path converges in a handful of events, so the
        warning must NOT fire. caplog (NOT recwarn) is required because this is
        a ``logging`` warning, not a ``warnings.warn`` record.
        """
        with caplog.at_level(logging.WARNING, logger="gwtransport.fronttracking.solver"):
            self._run_canonical()
        assert not any("max_iterations" in r.getMessage() for r in caplog.records), (
            "solver hit max_iterations — the unterminating-collision bug has returned: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    def test_r2_hard_event_bound(self):
        """R2: the canonical run produces fewer than 100 solver events.

        Literal bound, not relative: the exact path produces a single
        collision (DSW) plus a fan-exhaustion and a few outlet crossings —
        well under 100. The bug produced ~10000 (capped at max_iterations).
        """
        _, structures = self._run_canonical()
        n_events = structures[0]["n_events"]
        assert n_events < 100, f"expected < 100 solver events, got {n_events} (collision-cascade bug signature)"
        # The scenario must actually exercise the collision path, else the
        # bound is vacuous.
        n_dsw = sum(1 for w in structures[0]["tracker_state"].waves if isinstance(w, DecayingShockWave))
        assert n_dsw >= 1, "scenario did not form a DecayingShockWave; the regression target is not exercised"

    def test_r5_bin_average_matches_flow_weighted_exact_on_colliding_run(self):
        """R5: bin-averaged ``q_water_table`` equals the flow-weighted mean of the exact curve.

        This is the 500%-error symptom: on the colliding (DSW) run the
        bin-averaged output must equal the flow-weighted (== time-weighted, no
        K-scaling) mean of the pointwise ``concentration_at_point`` curve over
        each bin. The sampled bins are smooth (DSW-controlled), so the
        trapezoid residual is ≈ 4e-14 and the check holds at ``rtol = 1e-9``
        with a five-order margin — tight enough to catch sub-percent
        regressions, not just the shipped ~500% overshoot.
        """
        q_wt, structures = self._run_canonical()
        state = structures[0]["tracker_state"]
        v_out_val = float(O05["theta_s"])
        theta_max_inlet = float(state.theta_edges[-1])

        out_tedges = _make_tedges(len(_R_Q_ROOT))
        out_days = ((out_tedges - out_tedges[0]) / pd.Timedelta(days=1)).to_numpy()
        theta_out_edges = np.array([state.theta_at_t(float(t)) for t in out_days])

        # The DSW-controlled outlet response begins when the wetting front
        # reaches v_out (the single outlet_crossing of this run). Compare bins
        # at/after that arrival time: pre-arrival bins sit in the early
        # transient where the inlet-integral back-transform and the pointwise
        # wave query are not expected to agree (and are not the regression
        # target). Also require bins fully inside the inlet θ-window (beyond it
        # the inlet integral and the wave list live on inconsistent θ ranges).
        outlet_crossings = [ev["theta"] for ev in state.events if ev["type"] == "outlet_crossing"]
        assert outlet_crossings, "expected an outlet crossing on the colliding run"
        t_arrival = state.t_at_theta(float(min(outlet_crossings)))

        inside = theta_out_edges[1:] <= theta_max_inlet + 1e-9
        post_arrival = out_days[:-1] >= t_arrival - 1e-9
        nonzero = q_wt > 1e-9
        eligible = np.where(nonzero & inside & post_arrival)[0]
        assert eligible.size >= 5, f"expected several eligible bins, got {eligible.size}"

        # Sample the first post-arrival bin plus four spread-out bins covering
        # the DSW-controlled and steady regimes. These bins are smooth
        # (DSW-controlled), so 2001 sample points keep the trapezoid residual
        # ≈ 4e-14 (≪ rtol=1e-9), while a sub-percent bin-average error — and a
        # fortiori the shipped ~500% overshoot — fails at rtol=1e-9.
        idx = [int(eligible[0]), *(int(eligible[j]) for j in np.linspace(1, eligible.size - 1, 4).astype(int))]
        for k in idx:
            t_lo, t_hi = out_days[k], out_days[k + 1]
            tt = np.linspace(t_lo, t_hi, 2001)
            c_exact = np.array([
                concentration_at_point(v_out_val, state.theta_at_t(float(t)), state.waves, state.sorption) for t in tt
            ])
            mean_exact = np.trapezoid(c_exact, tt) / (t_hi - t_lo)
            np.testing.assert_allclose(
                q_wt[k],
                mean_exact,
                rtol=1e-9,
                err_msg=f"bin {k}: bin-average {q_wt[k]:.6e} vs flow-weighted exact {mean_exact:.6e}",
            )


# =============================================================================
# Section P — Transient global mass balance
# =============================================================================
#
# Reuses the Plan 1 independent-reference pattern: m_in and m_out are summed
# DIRECTLY from the public-API arrays (q_root_zone, q_water_table) — never from
# the solver's internal mass functions — and only the *stored* term reads
# compute_domain_mass.
#
# UNITS: the percolation public API is in water-depth flux [m/day], not solute
# mass. With Δt in days, m_in = Σ q_root·Δt and m_out = Σ q_wt·Δt are both in
# metres of water depth.
#
# STORED-TERM SCALING (physics-verified): compute_domain_mass integrates the
# conserved storage C_T = θ − θ_r over the solver V-coordinate, where dV = θ_s·dz.
# It therefore returns θ_s × the physical stored depth, so the closing term is
# compute_domain_mass(θ) / θ_s (DIVIDE). Multiplying by θ_s, or omitting the
# factor, leaves an O(0.03–0.1) residual — both are plausible-looking traps.


class TestTransientMassBalance:
    """P-series: transient global water-balance closure for percolation (BC and vG)."""

    # Wet-then-dry forcing whose response exits *inside* the inlet θ-window
    # (avoids the out-of-window degradation regime). The "dry" phase is small but
    # positive so the column stays wet and the final θ stays bounded.
    _WET_THEN_DRY = np.array([0.003] * 30 + [0.0008] * 170)
    _Z_WT = 0.2  # short column [m]
    _DT_DAYS = 1.0

    def _run(self, sorption_kwargs):
        q_root = self._WET_THEN_DRY
        n = len(q_root)
        tedges = _make_tedges(n)
        v_out_val = sorption_kwargs["theta_s"] * self._Z_WT
        # The class forcing is chosen so the response stays inside the inlet
        # θ-window (no out-of-window clamp). Enforce that contract: any warning
        # (e.g. the clamp UserWarning that would signal an out-of-window, possibly
        # clamped-to-zero, distorted balance) becomes an error rather than being
        # silently swallowed — mirroring test_m2b_no_clamp_warning_inside_inlet_window.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            q_wt, structures = root_zone_to_water_table_kinematic_wave(
                q_root_zone=q_root,
                tedges=tedges,
                q_water_table_tedges=tedges,
                cumulative_pore_volumes_outlet=np.array([v_out_val]),
                **sorption_kwargs,
            )
        return q_root, q_wt, structures[0]["tracker_state"], v_out_val

    def test_p1_global_water_balance_bc_closed_form(self):
        """P1 (BC): ``Σ q_root·Δt == Σ q_wt·Δt + compute_domain_mass(θ)/θ_s`` to machine precision.

        Brooks-Corey is closed-form throughout, so the balance closes at
        ``rtol ≈ 1e-12`` (the verified residual is ≈ 5e-17). The stored term
        DIVIDES ``compute_domain_mass`` by ``θ_s`` (see module note); the
        ``×θ_s`` and ``×1`` traps are checked explicitly below to fail.
        """
        q_root, q_wt, state, v_out_val = self._run(O05)
        theta_s = O05["theta_s"]
        theta_query = float(state.theta_edges[-1])  # end of the inlet window
        m_in = float(np.sum(q_root) * self._DT_DAYS)
        m_out = float(np.sum(q_wt) * self._DT_DAYS)
        m_dom = compute_domain_mass(theta=theta_query, v_outlet=v_out_val, waves=state.waves, sorption=state.sorption)
        stored = m_dom / theta_s  # DIVIDE — physics-verified
        np.testing.assert_allclose(m_in, m_out + stored, rtol=1e-12)
        # The two plausible-but-wrong scalings must NOT close.
        assert not np.isclose(m_in, m_out + m_dom, rtol=1e-6), "m_dom (no θ_s factor) should not close the balance"
        assert not np.isclose(m_in, m_out + m_dom * theta_s, rtol=1e-6), "m_dom·θ_s should not close the balance"

    def test_p2_global_water_balance_vg_exact_branch(self):
        """P2 (vG, ``mualem_l = 0``): the exact-branch global water balance closes to ``rtol ≈ 1e-12``.

        The Burdine (``L = 0``) vG branch has a closed-form ``S_e(C)`` inverse and
        is free of root-finding, so the balance closes to machine precision
        (verified residual ≈ 3e-17). This is the only end-to-end conservation
        test on the vG exact branch (vG otherwise appears in B1/B3'/B7/D2).
        """
        vg_l0 = {
            "theta_r": O05_VG["theta_r"],
            "theta_s": O05_VG["theta_s"],
            "k_s": O05_VG["k_s"],
            "van_genuchten_n": O05_VG["van_genuchten_n"],
            "mualem_l": 0.0,
        }
        q_root, q_wt, state, v_out_val = self._run(vg_l0)
        theta_s = vg_l0["theta_s"]
        theta_query = float(state.theta_edges[-1])
        m_in = float(np.sum(q_root) * self._DT_DAYS)
        m_out = float(np.sum(q_wt) * self._DT_DAYS)
        m_dom = compute_domain_mass(theta=theta_query, v_outlet=v_out_val, waves=state.waves, sorption=state.sorption)
        np.testing.assert_allclose(m_in, m_out + m_dom / theta_s, rtol=1e-12)

    def test_p3_global_water_balance_vg_default_mualem_branch(self):
        """P3 (vG, default ``mualem_l = 0.5``): the DEFAULT brentq branch global water balance closes.

        P2 covers only the ``L = 0`` (Burdine, root-finding-free) branch. The
        DEFAULT and most common usage is ``van_genuchten_l = 0.5``, which inverts
        ``K_M(S_e) = C`` with brentq throughout. This is the only end-to-end
        conservation test on that iterative branch: it catches a brentq-only
        inlet/outlet inversion error, or a ``compute_domain_mass`` scaling bug,
        that the closed-form ``L = 0`` path (P2) and Brooks-Corey path (P1) would
        both mask. On the in-window ``_WET_THEN_DRY`` forcing the balance closes to
        the brentq-bounded ``rtol = 1e-11`` (the verified residual is ≈ 0).
        """
        vg_default = {
            "theta_r": O05_VG["theta_r"],
            "theta_s": O05_VG["theta_s"],
            "k_s": O05_VG["k_s"],
            "van_genuchten_n": O05_VG["van_genuchten_n"],
        }  # mualem_l omitted → default 0.5 (brentq inversions)
        q_root, q_wt, state, v_out_val = self._run(vg_default)
        theta_s = vg_default["theta_s"]
        theta_query = float(state.theta_edges[-1])
        m_in = float(np.sum(q_root) * self._DT_DAYS)
        m_out = float(np.sum(q_wt) * self._DT_DAYS)
        m_dom = compute_domain_mass(theta=theta_query, v_outlet=v_out_val, waves=state.waves, sorption=state.sorption)
        np.testing.assert_allclose(m_in, m_out + m_dom / theta_s, rtol=1e-11)
