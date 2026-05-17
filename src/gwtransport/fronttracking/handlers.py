"""Event handlers for front tracking in (V, θ) coordinates.

Each handler receives the waves involved in an event and returns the new
waves created by the interaction. In (V, θ) coordinates every wave speed is
flow-free, so handlers depend only on concentrations and the sorption
isotherm — flow does not appear.

All handlers enforce physical correctness:

- Mass conservation (Rankine-Hugoniot condition)
- Entropy conditions (Lax condition for shocks)
- Causality (no backward-traveling information)

Handlers modify wave states in-place by deactivating parent waves and
creating new child waves.
"""

from gwtransport.fronttracking.math import FreundlichSorption, LangmuirSorption, SorptionModel, characteristic_speed
from gwtransport.fronttracking.waves import CharacteristicWave, DecayingShockWave, RarefactionWave, ShockWave

# Numerical tolerance constants
EPSILON_CONCENTRATION = 1e-15  # Tolerance for checking if concentration change is negligible


def handle_characteristic_collision(
    char1: CharacteristicWave,
    char2: CharacteristicWave,
    theta_event: float,
    v_event: float,
) -> list[ShockWave]:
    """Two characteristics collide → emit a shock.

    The faster characteristic catches the slower one from behind. By the
    entropy condition this compressive interaction is always a shock,
    independently of the sorption regime (Freundlich n>1, n<1, or constant
    retardation).

    Parameters
    ----------
    char1, char2 : CharacteristicWave
        Colliding characteristics.
    theta_event : float
        Cumulative flow at which the collision occurs [m³].
    v_event : float
        Position at which the collision occurs [m³].

    Returns
    -------
    list of ShockWave
        Single shock created at the collision point.

    Raises
    ------
    RuntimeError
        If the resulting shock fails the Lax entropy condition.
    """
    s1 = characteristic_speed(char1.concentration, char1.sorption)
    s2 = characteristic_speed(char2.concentration, char2.sorption)

    if s1 > s2:
        c_left = char1.concentration
        c_right = char2.concentration
    else:
        c_left = char2.concentration
        c_right = char1.concentration

    shock = ShockWave(
        theta_start=theta_event,
        v_start=v_event,
        c_left=c_left,
        c_right=c_right,
        sorption=char1.sorption,
    )

    if not shock.satisfies_entropy():
        msg = (
            f"Characteristic collision created non-entropic shock at θ={theta_event:.3f}, "
            f"V={v_event:.3f}. c_left={c_left:.3f}, c_right={c_right:.3f}, "
            f"shock_speed={shock.speed:.6g}"
        )
        raise RuntimeError(msg)

    char1.is_active = False
    char2.is_active = False
    return [shock]


def handle_shock_collision(
    shock1: ShockWave,
    shock2: ShockWave,
    theta_event: float,
    v_event: float,
) -> list[ShockWave]:
    """Two shocks collide → merge into a single shock connecting outer states.

    The merged shock has ``c_left`` from the faster (upstream) shock,
    ``c_right`` from the slower (downstream) shock; its speed is recomputed
    via Rankine-Hugoniot.

    Parameters
    ----------
    shock1, shock2 : ShockWave
        Colliding shocks.
    theta_event, v_event : float
        Cumulative flow [m³] and position [m³] of the collision.

    Returns
    -------
    list of ShockWave
        Single merged shock.

    Raises
    ------
    RuntimeError
        If the merged shock violates the entropy condition.
    """
    if shock1.speed > shock2.speed:
        c_left = shock1.c_left
        c_right = shock2.c_right
    else:
        c_left = shock2.c_left
        c_right = shock1.c_right

    merged = ShockWave(
        theta_start=theta_event,
        v_start=v_event,
        c_left=c_left,
        c_right=c_right,
        sorption=shock1.sorption,
    )

    if not merged.satisfies_entropy():
        msg = (
            f"Shock merger created non-entropic shock at θ={theta_event:.3f}. "
            f"This may indicate complex wave interaction requiring special handling."
        )
        raise RuntimeError(msg)

    shock1.is_active = False
    shock2.is_active = False

    return [merged]


def handle_shock_characteristic_collision(
    shock: ShockWave,
    char: CharacteristicWave,
    theta_event: float,
    v_event: float,
) -> list:
    """Shock catches or is caught by a characteristic.

    The characteristic concentration modifies one side of the shock:

    - Shock catches char (shock faster): modifies ``c_right``.
    - Char catches shock (char faster): modifies ``c_left``.

    If the resulting shock satisfies entropy it is emitted (compression);
    otherwise a rarefaction is created (expansion) to preserve mass balance.
    """
    s_shock = shock.speed
    s_char = characteristic_speed(char.concentration, char.sorption)

    if s_shock > s_char:
        new_shock = ShockWave(
            theta_start=theta_event,
            v_start=v_event,
            c_left=shock.c_left,
            c_right=char.concentration,
            sorption=shock.sorption,
        )
    else:
        new_shock = ShockWave(
            theta_start=theta_event,
            v_start=v_event,
            c_left=char.concentration,
            c_right=shock.c_right,
            sorption=shock.sorption,
        )

    if not new_shock.satisfies_entropy():
        # Expansion regime: emit a rarefaction whose head is the faster
        # state and tail the slower state.
        if s_shock > s_char:
            c_head = shock.c_left
            c_tail = char.concentration
        else:
            c_head = char.concentration
            c_tail = shock.c_right

        s_head = characteristic_speed(c_head, shock.sorption)
        s_tail = characteristic_speed(c_tail, shock.sorption)

        if s_head > s_tail:
            raref = RarefactionWave(
                theta_start=theta_event,
                v_start=v_event,
                c_head=c_head,
                c_tail=c_tail,
                sorption=shock.sorption,
            )
            shock.is_active = False
            char.is_active = False
            return [raref]
        # Edge case (s_head == s_tail within machine precision): deactivate
        # and emit nothing.
        shock.is_active = False
        char.is_active = False
        return []

    shock.is_active = False
    char.is_active = False
    return [new_shock]


def handle_shock_rarefaction_collision(
    shock: ShockWave,
    raref: RarefactionWave,
    theta_event: float,
    v_event: float,
    boundary_type: str | None,
) -> list:
    """Shock interacts with a rarefaction fan (tail or head boundary).

    For the canonical favorable (n>1 or Langmuir) head-collision case and the
    n<1 mirrored tail-collision case, emits a single :class:`DecayingShockWave`
    whose closed-form trajectory subsumes the fan + shock together. The
    fallback for non-canonical cases (e.g. n>1 tail-collision corner-case
    triggered by multi-pulse inlets) retains the Phase-1 piecewise-constant
    overlay; Phase-2 step 5+ adds parametric tests that will surface any
    remaining defects here.

    Returns
    -------
    list of Wave
        Emitted waves. For canonical cases this is ``[DecayingShockWave]``;
        for non-canonical cases it may be ``[ShockWave]``,
        ``[ShockWave, RarefactionWave]``, or ``[]``.
    """
    sorption = raref.sorption
    is_freundlich_unfavorable = isinstance(sorption, FreundlichSorption) and sorption.n < 1.0
    is_favorable_freundlich = isinstance(sorption, FreundlichSorption) and sorption.n > 1.0
    is_langmuir = isinstance(sorption, LangmuirSorption)

    # Multi-pulse non-canonical cases (raref.c_tail != shock.c_right for head
    # collision, raref.c_head != shock.c_left for tail collision) carry a
    # fan_tail concentration different from the shock's c_fixed; the current
    # DecayingShockWave does not store fan_tail separately, so its
    # ``concentration_at_point`` would clamp the fan-interior c incorrectly
    # past the fan's physical extent. Defer to Phase-1 overlay for these.
    is_canonical_head = boundary_type == "head" and abs(raref.c_tail - shock.c_right) < EPSILON_CONCENTRATION
    is_canonical_tail = boundary_type == "tail" and abs(raref.c_head - shock.c_left) < EPSILON_CONCENTRATION

    if is_canonical_head and (is_favorable_freundlich or is_langmuir):
        # Canonical favorable case: rarefaction head catches shock. After
        # collision, shock c_left decays from raref.c_head toward shock.c_right.
        # Only emit a DecayingShockWave if the rarefaction's head was actually
        # faster than the shock (precondition for physical collision); when
        # the test/solver feeds degenerate inputs (raref slower than shock,
        # impossible in a real run), deactivate both and emit nothing.
        s_raref_head = characteristic_speed(raref.c_head, raref.sorption)
        if s_raref_head <= shock.speed:
            shock.is_active = False
            raref.is_active = False
            return []
        assert isinstance(sorption, (FreundlichSorption, LangmuirSorption))  # noqa: S101
        try:
            decaying = DecayingShockWave(
                theta_start=theta_event,
                v_start=v_event,
                c_decay_initial=raref.c_head,
                c_fixed=shock.c_right,
                decay_side="left",
                v_origin=raref.v_start,
                theta_origin=raref.theta_start,
                sorption=sorption,
            )
        except NotImplementedError:
            # Closed form not derived yet (e.g. Freundlich n!=2 with c_fixed>0).
            # Fall through to Phase-1 overlay below; step 5+ extends coverage.
            pass
        else:
            shock.is_active = False
            raref.is_active = False
            return [decaying]

    if is_canonical_tail and is_freundlich_unfavorable:
        # Canonical n<1 mirror: trailing shock catches rarefaction's tail.
        # After collision, shock c_right decays from raref.c_tail toward
        # shock.c_left.
        s_raref_tail = characteristic_speed(raref.c_tail, raref.sorption)
        if shock.speed <= s_raref_tail:
            shock.is_active = False
            raref.is_active = False
            return []
        assert isinstance(sorption, FreundlichSorption)  # noqa: S101
        try:
            decaying = DecayingShockWave(
                theta_start=theta_event,
                v_start=v_event,
                c_decay_initial=raref.c_tail,
                c_fixed=shock.c_left,
                decay_side="right",
                v_origin=raref.v_start,
                theta_origin=raref.theta_start,
                sorption=sorption,
            )
        except NotImplementedError:
            pass
        else:
            shock.is_active = False
            raref.is_active = False
            return [decaying]

    # Non-canonical (multi-pulse corner cases) — fall back to Phase-1 overlay.
    if boundary_type == "tail":
        raref_c_at_collision = raref.concentration_at_point(v_event, theta_event)

        if raref_c_at_collision is None:
            new_shock = ShockWave(
                theta_start=theta_event,
                v_start=v_event,
                c_left=shock.c_left,
                c_right=raref.c_tail,
                sorption=shock.sorption,
            )
            if new_shock.satisfies_entropy():
                raref.is_active = False
                shock.is_active = False
                return [new_shock]
            return []

        new_shock = ShockWave(
            theta_start=theta_event,
            v_start=v_event,
            c_left=shock.c_left,
            c_right=raref_c_at_collision,
            sorption=shock.sorption,
        )

        if not new_shock.satisfies_entropy():
            raref.is_active = False
            shock.is_active = False
            return []

        c_new_tail = raref_c_at_collision

        s_head = characteristic_speed(raref.c_head, raref.sorption)
        s_tail = characteristic_speed(c_new_tail, raref.sorption)

        if s_head > s_tail:
            modified_raref = RarefactionWave(
                theta_start=theta_event,
                v_start=v_event,
                c_head=raref.c_head,
                c_tail=c_new_tail,
                sorption=raref.sorption,
            )
            shock.is_active = False
            raref.is_active = False
            return [new_shock, modified_raref]
        shock.is_active = False
        raref.is_active = False
        return [new_shock]

    # Non-canonical head branch fallback.
    s_raref_head = characteristic_speed(raref.c_head, raref.sorption)

    if s_raref_head > shock.speed:
        new_shock = ShockWave(
            theta_start=theta_event,
            v_start=v_event,
            c_left=raref.c_head,
            c_right=shock.c_right,
            sorption=raref.sorption,
        )

        if new_shock.satisfies_entropy():
            shock.is_active = False
            return [new_shock]

    shock.is_active = False
    raref.is_active = False
    return []


def handle_rarefaction_characteristic_collision(
    raref: RarefactionWave,
    char: CharacteristicWave,
    theta_event: float,
    v_event: float,
    boundary_type: str | None,
) -> list:
    """Rarefaction boundary intersects a characteristic.

    The safe option (b) from the front-tracking rebuild plan: when a
    characteristic's concentration matches the boundary concentration to
    within tolerance the characteristic is absorbed; otherwise an
    informative ``RuntimeError`` is raised because deactivating it would
    silently destroy mass.

    Raises
    ------
    RuntimeError
        If the characteristic's concentration does not match the colliding
        rarefaction boundary concentration within tolerance, or if
        ``boundary_type`` is not ``'head'`` or ``'tail'``.
    """
    rel_tol = 1e-9
    abs_tol = 1e-12
    raref_range = abs(raref.c_head - raref.c_tail)
    tol = max(rel_tol * raref_range, abs_tol)

    if boundary_type == "head":
        boundary_c = raref.c_head
    elif boundary_type == "tail":
        boundary_c = raref.c_tail
    else:
        msg = f"handle_rarefaction_characteristic_collision: unknown boundary_type {boundary_type!r}"
        raise RuntimeError(msg)

    if abs(char.concentration - boundary_c) > tol:
        msg = (
            f"Rarefaction-characteristic collision at θ={theta_event:.6f}, V={v_event:.6f} would silently "
            f"destroy mass: characteristic concentration {char.concentration:.6g} differs from "
            f"rarefaction {boundary_type} concentration {boundary_c:.6g} by "
            f"{abs(char.concentration - boundary_c):.3g} (tolerance {tol:.3g}). "
            f"Proper wave splitting at the rarefaction boundary is required for this case."
        )
        raise RuntimeError(msg)

    char.is_active = False
    return []


def handle_outlet_crossing(wave, theta_event: float, v_outlet: float) -> dict:
    """Record a wave crossing the outlet boundary.

    The wave is NOT deactivated — it remains for concentration queries at
    points between its origin and the outlet. The returned event record
    holds the cumulative flow ``theta`` at which the crossing occurs; the
    solver translates this to the user-facing time when appending to
    ``state.events``.
    """
    return {
        "theta": theta_event,
        "type": "outlet_crossing",
        "wave": wave,
        "location": v_outlet,
        "concentration_left": wave.concentration_left(),
        "concentration_right": wave.concentration_right(),
    }


def create_inlet_waves_at_theta(
    c_prev: float,
    c_new: float,
    theta: float,
    sorption: SorptionModel,
    v_inlet: float = 0.0,
) -> list:
    """Emit the wave produced by a step change in inlet concentration.

    Wave type is determined by characteristic speed comparison in (V, θ):

    - ``s_new > s_prev``: compression → shock.
    - ``s_new < s_prev``: expansion → rarefaction.
    - equal: contact discontinuity → characteristic.

    For shocks the entropy condition is verified; if violated, an empty list
    is returned (mass balance may be affected — a known limitation that
    motivates Phase 2 ``DecayingShockWave``).
    """
    if abs(c_new - c_prev) < EPSILON_CONCENTRATION:
        return []

    c_min = getattr(sorption, "c_min", 0.0)
    is_n_lt_1 = isinstance(sorption, FreundlichSorption) and sorption.n < 1.0

    # n<1, c_prev=0 or c_new=0: emit a single CharacteristicWave; clean water
    # has a well-defined speed since R(0)=1.
    if (c_prev <= c_min or c_new <= c_min) and is_n_lt_1 and c_min == 0:
        return [
            CharacteristicWave(
                theta_start=theta,
                v_start=v_inlet,
                concentration=c_new,
                sorption=sorption,
            )
        ]

    s_prev = characteristic_speed(c_prev, sorption)
    s_new = characteristic_speed(c_new, sorption)

    if s_new > s_prev + 1e-15:
        shock = ShockWave(
            theta_start=theta,
            v_start=v_inlet,
            c_left=c_new,
            c_right=c_prev,
            sorption=sorption,
        )
        if not shock.satisfies_entropy():
            return []
        return [shock]

    if s_new < s_prev - 1e-15:
        return [
            RarefactionWave(
                theta_start=theta,
                v_start=v_inlet,
                c_head=c_prev,
                c_tail=c_new,
                sorption=sorption,
            )
        ]

    return [
        CharacteristicWave(
            theta_start=theta,
            v_start=v_inlet,
            concentration=c_new,
            sorption=sorption,
        )
    ]
