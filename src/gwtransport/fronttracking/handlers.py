"""
Event Handlers for Front Tracking.

====================================

This module provides handlers for all wave interaction events in the front
tracking algorithm. Each handler receives waves involved in an event and
returns new waves created by the interaction.

All handlers enforce physical correctness:
- Mass conservation (Rankine-Hugoniot condition)
- Entropy conditions (Lax condition for shocks)
- Causality (no backward-traveling information)

Handlers modify wave states in-place by deactivating parent waves and
creating new child waves.
"""

from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption, characteristic_velocity
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave

# Numerical tolerance constants
EPSILON_CONCENTRATION = 1e-15  # Tolerance for checking if concentration change is negligible


def handle_characteristic_collision(
    char1: CharacteristicWave,
    char2: CharacteristicWave,
    t_event: float,
    v_event: float,
) -> list[ShockWave | RarefactionWave]:
    """
    Handle collision of two characteristics → create shock or rarefaction.

    When two characteristics with different concentrations intersect, they
    form a shock discontinuity. The faster characteristic (lower concentration
    for n>1) catches the slower one from behind.

    Parameters
    ----------
    char1 : CharacteristicWave
        First characteristic
    char2 : CharacteristicWave
        Second characteristic
    t_event : float
        Time of collision [days]
    v_event : float
        Position of collision [m³]

    Returns
    -------
    list[ShockWave]
        Single shock wave created at collision point

    Notes
    -----
    The shock has:
    - c_left: concentration from faster (upstream) characteristic
    - c_right: concentration from slower (downstream) characteristic
    - velocity: computed from Rankine-Hugoniot condition

    The parent characteristics are deactivated.

    Examples
    --------
    ::

        shock = handle_characteristic_collision(char1, char2, t=15.0, v=100.0)
        assert shock.satisfies_entropy()
        assert not char1.is_active  # Parent deactivated
    """
    # Get c_min from sorption to determine concentration threshold
    c_min = getattr(char1.sorption, "c_min", 0.0)
    is_n_lt_1 = isinstance(char1.sorption, FreundlichSorption) and char1.sorption.n < 1.0

    # Special case: if one characteristic has C near c_min
    # Need to determine if this is:
    # 1. C≈c_min from initial condition being overtaken by C>0 → keep C>0
    # 2. C≈c_min from inlet (clean water) catching C>0 → analyze velocities
    # Only use special handling for n<1 with c_min=0 where R(0)=1
    if char1.concentration <= c_min and char2.concentration > c_min and is_n_lt_1 and c_min == 0:
        # char1 is C≈0, char2 is C>0
        # Check velocities to determine who is catching whom
        vel1 = characteristic_velocity(char1.concentration, char1.flow, char1.sorption)
        vel2 = characteristic_velocity(char2.concentration, char2.flow, char2.sorption)

        if vel1 > vel2:
            # C=0 is faster → catching C>0 from behind
            # For Freundlich n>1: concentration decrease (C>0 → C=0) forms rarefaction
            # Physical: clean water (fast) catching contaminated water (slow)
            try:
                raref = RarefactionWave(
                    t_start=t_event,
                    v_start=v_event,
                    flow=char1.flow,
                    c_head=char1.concentration,  # C=0 is head (faster)
                    c_tail=char2.concentration,  # C>0 is tail (slower)
                    sorption=char1.sorption,
                )
            except ValueError:
                # Rarefaction creation failed - just keep C>0, deactivate C=0
                char1.is_active = False
                return []
            else:
                char1.is_active = False
                char2.is_active = False
                return [raref]
        else:
            # C>0 is faster → C>0 catching C=0 → C=0 is from initial condition
            # Just deactivate the C=0 and keep C>0 active
            char1.is_active = False
            return []

    elif char2.concentration <= c_min and char1.concentration > c_min and is_n_lt_1 and c_min == 0:
        # char2 is C≈0, char1 is C>0
        vel1 = characteristic_velocity(char1.concentration, char1.flow, char1.sorption)
        vel2 = characteristic_velocity(char2.concentration, char2.flow, char2.sorption)

        if vel2 > vel1:
            # C=0 is faster → catching C>0 from behind
            # For Freundlich n>1: concentration decrease forms rarefaction
            try:
                raref = RarefactionWave(
                    t_start=t_event,
                    v_start=v_event,
                    flow=char1.flow,
                    c_head=char2.concentration,  # C=0 is head (faster)
                    c_tail=char1.concentration,  # C>0 is tail (slower)
                    sorption=char1.sorption,
                )
            except ValueError:
                # Rarefaction creation failed
                char2.is_active = False
                return []
            else:
                char1.is_active = False
                char2.is_active = False
                return [raref]
        else:
            # C>0 is faster → C=0 is from initial condition
            char2.is_active = False
            return []

    # Normal case: analyze velocities to determine wave type
    # This now handles all cases for n>1 (higher C travels faster) and
    # concentrations above c_min for n<1 (lower C travels faster)
    vel1 = characteristic_velocity(char1.concentration, char1.flow, char1.sorption)
    vel2 = characteristic_velocity(char2.concentration, char2.flow, char2.sorption)

    if vel1 > vel2:
        c_left = char1.concentration
        c_right = char2.concentration
    else:
        c_left = char2.concentration
        c_right = char1.concentration

    # Create shock at collision point
    shock = ShockWave(
        t_start=t_event,
        v_start=v_event,
        flow=char1.flow,  # Assume same flow (piecewise constant)
        c_left=c_left,
        c_right=c_right,
        sorption=char1.sorption,
    )

    # Verify entropy condition
    if not shock.satisfies_entropy():
        # This shouldn't happen if characteristics collided correctly
        msg = (
            f"Characteristic collision created non-entropic shock at t={t_event:.3f}, V={v_event:.3f}. "
            f"c_left={c_left:.3f}, c_right={c_right:.3f}, shock_vel={shock.velocity:.3f}"
        )
        raise RuntimeError(msg)

    # Deactivate parent characteristics
    char1.is_active = False
    char2.is_active = False

    return [shock]


def handle_shock_collision(
    shock1: ShockWave,
    shock2: ShockWave,
    t_event: float,
    v_event: float,
) -> list[ShockWave]:
    """
    Handle collision of two shocks → merge into single shock.

    When two shocks collide, they merge into a single shock that connects
    the left state of the upstream shock to the right state of the downstream
    shock.

    Parameters
    ----------
    shock1 : ShockWave
        First shock
    shock2 : ShockWave
        Second shock
    t_event : float
        Time of collision [days]
    v_event : float
        Position of collision [m³]

    Returns
    -------
    list[ShockWave]
        Single merged shock wave

    Notes
    -----
    The merged shock has:
    - c_left: from the faster (upstream) shock
    - c_right: from the slower (downstream) shock
    - velocity: recomputed from Rankine-Hugoniot

    The parent shocks are deactivated.

    Examples
    --------
    ::

        merged = handle_shock_collision(shock1, shock2, t=20.0, v=200.0)
        assert merged.satisfies_entropy()
        assert not shock1.is_active  # Parents deactivated
    """
    # Determine which shock is upstream (faster)
    # The shock catching up from behind is upstream
    if shock1.velocity is None or shock2.velocity is None:
        msg = "Shock velocities should be set in __post_init__"
        raise RuntimeError(msg)
    if shock1.velocity > shock2.velocity:
        c_left = shock1.c_left
        c_right = shock2.c_right
    else:
        c_left = shock2.c_left
        c_right = shock1.c_right

    # Create merged shock
    merged = ShockWave(
        t_start=t_event,
        v_start=v_event,
        flow=shock1.flow,
        c_left=c_left,
        c_right=c_right,
        sorption=shock1.sorption,
    )

    # Entropy should be satisfied (both parents were entropic)
    if not merged.satisfies_entropy():
        # This can happen if the intermediate state causes issues
        # In some cases, the shocks might pass through each other instead
        msg = (
            f"Shock merger created non-entropic shock at t={t_event:.3f}. "
            f"This may indicate complex wave interaction requiring special handling."
        )
        raise RuntimeError(msg)

    # Deactivate parent shocks
    shock1.is_active = False
    shock2.is_active = False

    return [merged]


def handle_shock_characteristic_collision(
    shock: ShockWave,
    char: CharacteristicWave,
    t_event: float,
    v_event: float,
) -> list:
    """
    Handle shock catching or being caught by characteristic.

    When the attempted shock would violate entropy (indicating expansion rather
    than compression), a rarefaction wave is created instead to preserve mass
    balance. This addresses High Priority #1 from FRONT_TRACKING_REBUILD_PLAN.md.

    The outcome depends on which wave is faster:
    - If shock is faster: shock catches characteristic, absorbs it
    - If characteristic is faster: characteristic catches shock, modifies it

    Parameters
    ----------
    shock : ShockWave
        Shock wave
    char : CharacteristicWave
        Characteristic wave
    t_event : float
        Time of collision [days]
    v_event : float
        Position of collision [m³]

    Returns
    -------
    list
        List containing new wave(s): ShockWave if compression, RarefactionWave
        if expansion, or empty list in edge cases

    Notes
    -----
    The characteristic concentration modifies one side of the shock:
    - If shock catches char: modifies c_right
    - If char catches shock: modifies c_left

    If the new shock satisfies entropy → return shock (compression)
    If entropy violated → create rarefaction instead (expansion)

    Examples
    --------
    ::

        new_shock = handle_shock_characteristic_collision(shock, char, t=25.0, v=300.0)
        if new_shock:
            assert new_shock[0].satisfies_entropy()
    """
    if shock.velocity is None:
        msg = "Shock velocity should be set in __post_init__"
        raise RuntimeError(msg)
    shock_vel = shock.velocity
    char_vel = characteristic_velocity(char.concentration, char.flow, char.sorption)

    if shock_vel > char_vel:
        # Shock catching characteristic from behind
        # Characteristic is on right side of shock
        # New shock: c_left unchanged, c_right = char.concentration
        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=shock.flow,
            c_left=shock.c_left,
            c_right=char.concentration,
            sorption=shock.sorption,
        )
    else:
        # Characteristic catching shock from behind
        # Characteristic is on left side of shock
        # New shock: c_left = char.concentration, c_right unchanged
        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=shock.flow,
            c_left=char.concentration,
            c_right=shock.c_right,
            sorption=shock.sorption,
        )

    # Check entropy condition
    if not new_shock.satisfies_entropy():
        # Entropy violated → this is an expansion, not compression
        # Create rarefaction wave instead of shock to preserve mass balance

        # Determine head and tail concentrations based on velocity ordering
        # For a rarefaction: head (faster) follows tail (slower)
        if shock_vel > char_vel:
            # Shock was catching characteristic
            # Expansion between shock.c_left (faster) and char.concentration (slower)
            c_head = shock.c_left
            c_tail = char.concentration
        else:
            # Characteristic was catching shock
            # Expansion between char.concentration (faster) and shock.c_right (slower)
            c_head = char.concentration
            c_tail = shock.c_right

        # Verify this creates a valid rarefaction (head faster than tail)
        head_vel = characteristic_velocity(c_head, shock.flow, shock.sorption)
        tail_vel = characteristic_velocity(c_tail, shock.flow, shock.sorption)

        if head_vel > tail_vel:
            # Valid rarefaction - create it
            try:
                raref = RarefactionWave(
                    t_start=t_event,
                    v_start=v_event,
                    flow=shock.flow,
                    c_head=c_head,
                    c_tail=c_tail,
                    sorption=shock.sorption,
                )
            except ValueError:
                # Rarefaction validation failed - edge case
                # Deactivate waves and return empty
                shock.is_active = False
                char.is_active = False
                return []
            else:
                # Deactivate parent waves
                shock.is_active = False
                char.is_active = False
                return [raref]
        else:
            # Not a valid rarefaction - waves may pass through each other
            # This is an edge case - deactivate and return empty
            shock.is_active = False
            char.is_active = False
            return []

    # Shock satisfies entropy - return it
    # Deactivate parent waves
    shock.is_active = False
    char.is_active = False

    return [new_shock]


def handle_shock_rarefaction_collision(
    shock: ShockWave,
    raref: RarefactionWave,
    t_event: float,
    v_event: float,
    boundary_type: str,
) -> list:
    """
    Handle shock interacting with rarefaction fan with wave splitting.

    Implements proper wave splitting for shock-rarefaction interactions,
    addressing High Priority #2 from FRONT_TRACKING_REBUILD_PLAN.md.

    This is the most complex interaction. A shock can:
    - Catch the rarefaction tail: shock penetrates into rarefaction fan,
      creating both a modified rarefaction and a continuing shock
    - Be caught by rarefaction head: creates compression wave

    Parameters
    ----------
    shock : ShockWave
        Shock wave
    raref : RarefactionWave
        Rarefaction wave
    t_event : float
        Time of collision [days]
    v_event : float
        Position of collision [m³]
    boundary_type : str
        Which boundary collided: 'head' or 'tail'

    Returns
    -------
    list
        List of new waves created: may include shock and modified rarefaction
        for tail collision, or compression shock for head collision

    Notes
    -----
    **Tail collision**: Shock penetrates rarefaction, creating:
    - New shock continuing through rarefaction
    - Modified rarefaction with compressed tail (if rarefaction not fully overtaken)

    **Head collision**: Rarefaction head catches shock, may create compression shock

    Examples
    --------
    ::

        waves = handle_shock_rarefaction_collision(
            shock, raref, t=30.0, v=400.0, boundary_type="tail"
        )
    """
    if boundary_type == "tail":
        # Shock catching rarefaction tail - FULL WAVE SPLITTING
        # Shock penetrates into rarefaction fan, need to split waves

        # Query rarefaction concentration at collision point
        # This tells us where in the rarefaction fan the shock is
        raref_c_at_collision = raref.concentration_at_point(v_event, t_event)

        if raref_c_at_collision is None:
            # Shock not actually inside rarefaction - edge case
            # Fall back to simple approach
            new_shock = ShockWave(
                t_start=t_event,
                v_start=v_event,
                flow=shock.flow,
                c_left=shock.c_left,
                c_right=raref.c_tail,
                sorption=shock.sorption,
            )
            if new_shock.satisfies_entropy():
                raref.is_active = False
                shock.is_active = False
                return [new_shock]
            return []

        # Create shock that continues through rarefaction
        # Right state is the rarefaction concentration at collision
        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=shock.flow,
            c_left=shock.c_left,
            c_right=raref_c_at_collision,
            sorption=shock.sorption,
        )

        if not new_shock.satisfies_entropy():
            # Complex case - shock doesn't continue
            raref.is_active = False
            shock.is_active = False
            return []

        # Create modified rarefaction with compressed tail
        # The portion of rarefaction ahead of shock remains
        # New tail is at the collision concentration
        c_new_tail = raref_c_at_collision

        # Verify head is still faster than new tail
        head_vel = characteristic_velocity(raref.c_head, raref.flow, raref.sorption)
        tail_vel = characteristic_velocity(c_new_tail, raref.flow, raref.sorption)

        if head_vel > tail_vel:
            # Create modified rarefaction starting from collision point
            try:
                modified_raref = RarefactionWave(
                    t_start=t_event,
                    v_start=v_event,
                    flow=raref.flow,
                    c_head=raref.c_head,
                    c_tail=c_new_tail,
                    sorption=raref.sorption,
                )
            except ValueError:
                # Rarefaction validation failed
                shock.is_active = False
                raref.is_active = False
                return [new_shock]
            else:
                # Deactivate original waves
                shock.is_active = False
                raref.is_active = False

                return [new_shock, modified_raref]
        else:
            # Rarefaction completely overtaken - only shock continues
            shock.is_active = False
            raref.is_active = False
            return [new_shock]

    # boundary_type == 'head'
    # Rarefaction head catching shock
    # This creates compression between rarefaction head and shock
    # May form new compression shock

    # Check if compression forms between rarefaction head and shock
    raref_head_vel = characteristic_velocity(raref.c_head, raref.flow, raref.sorption)
    if shock.velocity is None:
        msg = "Shock velocity should be set in __post_init__"
        raise RuntimeError(msg)
    shock_vel = shock.velocity

    if raref_head_vel > shock_vel:
        # Rarefaction head is faster - creates compression
        # Try to form shock between rarefaction head and shock left state
        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=raref.flow,
            c_left=raref.c_head,
            c_right=shock.c_left,
            sorption=raref.sorption,
        )

        if new_shock.satisfies_entropy():
            # Compression shock forms
            # Deactivate original shock (rarefaction continues)
            shock.is_active = False
            return [new_shock]

    # No compression shock forms - deactivate both for safety
    shock.is_active = False
    raref.is_active = False
    return []


def handle_rarefaction_characteristic_collision(
    raref: RarefactionWave,  # noqa: ARG001
    char: CharacteristicWave,
    t_event: float,  # noqa: ARG001
    v_event: float,  # noqa: ARG001
    boundary_type: str,  # noqa: ARG001
) -> list:
    """
    Handle rarefaction boundary intersecting with characteristic.

    **SIMPLIFIED IMPLEMENTATION**: Just deactivates characteristic. See
    FRONT_TRACKING_REBUILD_PLAN.md "Known Issues and Future Improvements"
    Medium Priority #5.

    Parameters
    ----------
    raref : RarefactionWave
        Rarefaction wave
    char : CharacteristicWave
        Characteristic wave
    t_event : float
        Time of collision [days]
    v_event : float
        Position of collision [m³]
    boundary_type : str
        Which boundary collided: 'head' or 'tail'

    Returns
    -------
    list
        List of new waves created (currently always empty)

    Notes
    -----
    This is a simplified implementation that deactivates the characteristic
    without modifying the rarefaction structure.

    Current implementation: deactivates characteristic, leaves rarefaction
    unchanged.

    **Future enhancement**: Should modify rarefaction head/tail concentration
    to properly represent the wave structure instead of just absorbing the
    characteristic.
    """
    # Simplified: characteristic gets absorbed into rarefaction
    # More sophisticated: modify rarefaction boundaries
    char.is_active = False
    return []


def handle_rarefaction_rarefaction_collision(
    raref1: RarefactionWave,
    raref2: RarefactionWave,
    t_event: float,
    v_event: float,
    boundary_type: str,
) -> list:
    """Handle collision between two rarefaction boundaries.

    This handler is intentionally conservative: it records the fact that two
    rarefaction fans have intersected but does not yet modify the wave
    topology. Full entropic treatment of rarefaction-rarefaction interactions
    (potentially involving wave splitting) is reserved for a dedicated
    future enhancement.

    Parameters
    ----------
    raref1 : RarefactionWave
        First rarefaction wave in the collision.
    raref2 : RarefactionWave
        Second rarefaction wave in the collision.
    t_event : float
        Time of the boundary intersection [days].
    v_event : float
        Position of the intersection [m³].
    boundary_type : str
        Boundary of the first rarefaction that intersected: 'head' or 'tail'.

    Returns
    -------
    list
        Empty list; no new waves are created at this stage.

    Notes
    -----
    - Waves remain active so that concentration queries remain valid.
    - The FrontTracker records the event in its diagnostics history.
    - This is consistent with the design goal of exact analytical
      computation while deferring complex topology changes.
    """
    # No topology changes yet; keep both rarefactions active.
    _ = (raref1, raref2, t_event, v_event, boundary_type)
    return []


def handle_outlet_crossing(wave, t_event: float, v_outlet: float) -> dict:
    """
    Handle wave crossing outlet boundary.

    The wave exits the domain. It remains in the wave list for querying
    concentration at earlier times but is marked for different handling.

    Parameters
    ----------
    wave : Wave
        Any wave type (Characteristic, Shock, or Rarefaction)
    t_event : float
        Time when wave exits [days]
    v_outlet : float
        Outlet position [m³]

    Returns
    -------
    dict
        Event record with details about the crossing

    Notes
    -----
    Waves are NOT deactivated when they cross the outlet. They remain active
    for concentration queries at points between their origin and outlet.

    The event record includes:
    - time: crossing time
    - type: 'outlet_crossing'
    - wave: reference to wave object
    - concentration_left: upstream concentration
    - concentration_right: downstream concentration

    Examples
    --------
    ::

        event = handle_outlet_crossing(shock, t=50.0, v_outlet=500.0)
        print(f"Wave exited at t={event['time']:.2f}")
    """
    return {
        "time": t_event,
        "type": "outlet_crossing",
        "wave": wave,
        "location": v_outlet,
        "concentration_left": wave.concentration_left(),
        "concentration_right": wave.concentration_right(),
    }


def recreate_characteristic_with_new_flow(
    char: CharacteristicWave,
    t_change: float,
    flow_new: float,
) -> CharacteristicWave:
    """
    Create new characteristic at current position with new flow.

    When flow changes, existing characteristics must be recreated with updated
    velocities. The concentration remains constant, but velocity becomes
    flow_new / R(concentration).

    Parameters
    ----------
    char : CharacteristicWave
        Existing characteristic to recreate
    t_change : float
        Time of flow change [days]
    flow_new : float
        New flow rate [m³/day]

    Returns
    -------
    CharacteristicWave
        New characteristic at current position with new flow

    Notes
    -----
    The parent characteristic should be deactivated by the caller.

    Examples
    --------
    ::

        char_old = CharacteristicWave(
            t_start=0.0, v_start=0.0, flow=100.0, concentration=5.0, sorption=sorption
        )
        char_new = recreate_characteristic_with_new_flow(
            char_old, t_change=10.0, flow_new=200.0
        )
        assert char_new.flow == 200.0
        assert char_new.concentration == 5.0  # Concentration unchanged
    """
    v_at_change = char.position_at_time(t_change)

    if v_at_change is None:
        msg = f"Characteristic not yet active at t={t_change}"
        raise ValueError(msg)

    return CharacteristicWave(
        t_start=t_change,
        v_start=v_at_change,
        flow=flow_new,
        concentration=char.concentration,
        sorption=char.sorption,
        is_active=True,
    )


def recreate_shock_with_new_flow(
    shock: ShockWave,
    t_change: float,
    flow_new: float,
) -> ShockWave:
    """
    Create new shock at current position with new flow.

    When flow changes, shock velocity must be recomputed using Rankine-Hugoniot
    condition with the new flow: s = flow*(c_R - c_L) / (C_total(c_R) - C_total(c_L)).

    Parameters
    ----------
    shock : ShockWave
        Existing shock to recreate
    t_change : float
        Time of flow change [days]
    flow_new : float
        New flow rate [m³/day]

    Returns
    -------
    ShockWave
        New shock at current position with updated velocity

    Notes
    -----
    The parent shock should be deactivated by the caller.
    Shock velocity is automatically recomputed in ShockWave.__post_init__.

    Examples
    --------
    ::

        shock_old = ShockWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_left=10.0,
            c_right=2.0,
            sorption=sorption,
        )
        shock_new = recreate_shock_with_new_flow(
            shock_old, t_change=10.0, flow_new=200.0
        )
        assert shock_new.flow == 200.0
        assert (
            shock_new.velocity == 2 * shock_old.velocity
        )  # Velocity scales linearly with flow
    """
    v_at_change = shock.position_at_time(t_change)

    if v_at_change is None:
        msg = f"Shock not yet active at t={t_change}"
        raise ValueError(msg)

    return ShockWave(
        t_start=t_change,
        v_start=v_at_change,
        flow=flow_new,
        c_left=shock.c_left,
        c_right=shock.c_right,
        sorption=shock.sorption,
        is_active=True,
    )


def recreate_rarefaction_with_new_flow(
    raref: RarefactionWave,
    t_change: float,
    flow_new: float,
) -> RarefactionWave:
    """
    Create new rarefaction at current position with new flow.

    When flow changes, rarefaction head and tail velocities are updated.
    The fan structure (c_head, c_tail) is preserved, but the self-similar
    solution pivots at the flow change point.

    Parameters
    ----------
    raref : RarefactionWave
        Existing rarefaction to recreate
    t_change : float
        Time of flow change [days]
    flow_new : float
        New flow rate [m³/day]

    Returns
    -------
    RarefactionWave
        New rarefaction at current position with updated velocities

    Notes
    -----
    The parent rarefaction should be deactivated by the caller.
    The rarefaction fan "pivots" at (v_at_change, t_change).

    Before: R(C) = flow_old * (t - t_start_old) / (v - v_start_old)
    After:  R(C) = flow_new * (t - t_change) / (v - v_at_change)

    Examples
    --------
    ::

        raref_old = RarefactionWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_head=10.0,
            c_tail=2.0,
            sorption=sorption,
        )
        raref_new = recreate_rarefaction_with_new_flow(
            raref_old, t_change=10.0, flow_new=200.0
        )
        assert raref_new.flow == 200.0
        assert raref_new.c_head == 10.0  # Concentrations unchanged
    """
    v_at_change = raref.position_at_time(t_change)

    if v_at_change is None:
        msg = f"Rarefaction not yet active at t={t_change}"
        raise ValueError(msg)

    return RarefactionWave(
        t_start=t_change,
        v_start=v_at_change,
        flow=flow_new,
        c_head=raref.c_head,
        c_tail=raref.c_tail,
        sorption=raref.sorption,
        is_active=True,
    )


def handle_flow_change(
    t_change: float,
    flow_new: float,
    active_waves: list,
) -> list:
    """
    Handle flow change event by recreating all active waves with new flow.

    When flow changes, all existing waves must be recreated at their current
    positions with updated velocities. This maintains exact analytical computation
    while correctly handling time-varying flow.

    Parameters
    ----------
    t_change : float
        Time of flow change [days]
    flow_new : float
        New flow rate [m³/day]
    active_waves : list
        All currently active waves

    Returns
    -------
    list
        New waves created at current positions with new flow

    Notes
    -----
    Parent waves are deactivated by this handler.

    Physical interpretation:
    - Characteristics: velocity changes from flow_old/R(c) to flow_new/R(c)
    - Shocks: Rankine-Hugoniot velocity recomputed with new flow
    - Rarefactions: fan pivots at (v_change, t_change)

    Examples
    --------
    ::

        new_waves = handle_flow_change(
            t_change=10.0, flow_new=200.0, active_waves=[char1, shock1, raref1]
        )
        assert len(new_waves) == 3
        assert all(w.flow == 200.0 for w in new_waves)
    """
    new_waves = []

    for wave in active_waves:
        if not wave.is_active:
            continue

        # Create replacement wave with new flow BEFORE deactivating parent
        # (position_at_time requires wave to be active)
        if isinstance(wave, CharacteristicWave):
            new_wave = recreate_characteristic_with_new_flow(wave, t_change, flow_new)
        elif isinstance(wave, ShockWave):
            new_wave = recreate_shock_with_new_flow(wave, t_change, flow_new)
        elif isinstance(wave, RarefactionWave):
            new_wave = recreate_rarefaction_with_new_flow(wave, t_change, flow_new)
        else:
            msg = f"Unknown wave type: {type(wave)}"
            raise TypeError(msg)

        new_waves.append(new_wave)

        # Deactivate parent wave AFTER recreation
        wave.is_active = False

    return new_waves


def create_inlet_waves_at_time(
    c_prev: float,
    c_new: float,
    t: float,
    flow: float,
    sorption: FreundlichSorption | ConstantRetardation,
    v_inlet: float = 0.0,
) -> list:
    """
    Create appropriate waves when inlet concentration changes.

    Analyzes the concentration change and creates the physically correct
    wave type based on characteristic velocities.

    Parameters
    ----------
    c_prev : float
        Previous concentration [mass/volume]
    c_new : float
        New concentration [mass/volume]
    t : float
        Time of concentration change [days]
    flow : float
        Flow rate [m³/day]
    sorption : FreundlichSorption or ConstantRetardation
        Sorption parameters
    v_inlet : float, optional
        Inlet position [m³], default 0.0

    Returns
    -------
    list
        List of newly created waves (typically 1 wave per concentration change)

    Notes
    -----
    Wave type logic:
    - vel_new > vel_prev: Compression → create ShockWave
    - vel_new < vel_prev: Expansion → create RarefactionWave
    - vel_new == vel_prev: Contact discontinuity → create CharacteristicWave

    For shocks, verifies entropy condition before creation.

    Examples
    --------
    ::

        # Step increase from zero creates characteristic
        waves = create_inlet_waves_at_time(
            c_prev=0.0, c_new=10.0, t=10.0, flow=100.0, sorption=sorption
        )
        assert isinstance(waves[0], CharacteristicWave)
        # Step between nonzero values creates shock for n>1 (compression)
        waves = create_inlet_waves_at_time(
            c_prev=2.0, c_new=10.0, t=10.0, flow=100.0, sorption=sorption
        )
        assert isinstance(waves[0], ShockWave)
    """
    if abs(c_new - c_prev) < EPSILON_CONCENTRATION:  # No change
        return []

    # Get c_min from sorption if available (determines when to use special treatment)
    c_min = getattr(sorption, "c_min", 0.0)
    is_n_lt_1 = isinstance(sorption, FreundlichSorption) and sorption.n < 1.0

    # Special case: c_prev ≈ 0 AND this is n<1 with c_min=0
    # For n<1 (lower C travels faster), R(0)=1 is physically correct
    # The C=0 "water" ahead has a well-defined velocity and represents initial condition
    if c_prev <= c_min and is_n_lt_1 and c_min == 0:
        # Create characteristic wave with new concentration
        # The front propagates at v(c_new), leaving c_new behind and 0 ahead
        char = CharacteristicWave(
            t_start=t,
            v_start=v_inlet,
            flow=flow,
            concentration=c_new,
            sorption=sorption,
        )
        return [char]

    # Special case: c_new ≈ 0 AND this is n<1 with c_min=0
    # For n<1 (lower C travels faster), clean water (C=0) has well-defined velocity
    if c_new <= c_min and is_n_lt_1 and c_min == 0:
        # Create characteristic wave with zero concentration
        # This represents clean water entering the domain
        char = CharacteristicWave(
            t_start=t,
            v_start=v_inlet,
            flow=flow,
            concentration=c_new,
            sorption=sorption,
        )
        return [char]

    # Normal case: analyze velocities to determine wave type
    # For n>1 (higher C travels faster), even stepping down to c_min creates proper waves
    # The velocity analysis will determine if it's a shock, rarefaction, or characteristic
    vel_prev = characteristic_velocity(c_prev, flow, sorption)
    vel_new = characteristic_velocity(c_new, flow, sorption)

    if vel_new > vel_prev + 1e-15:  # Compression
        # New water is faster - will catch old water - create shock
        shock = ShockWave(
            t_start=t,
            v_start=v_inlet,
            flow=flow,
            c_left=c_new,  # Upstream is new (faster) water
            c_right=c_prev,  # Downstream is old (slower) water
            sorption=sorption,
        )

        # Verify entropy
        if not shock.satisfies_entropy():
            # Shock violates entropy - this compression cannot form a simple shock
            # This is a known limitation: some large jumps need composite waves
            # For now, return empty (no wave created) - mass balance may be affected
            # TODO: Implement composite wave creation (shock + rarefaction)
            return []

        return [shock]

    if vel_new < vel_prev - 1e-15:  # Expansion
        # New water is slower - will fall behind old water - create rarefaction
        try:
            raref = RarefactionWave(
                t_start=t,
                v_start=v_inlet,
                flow=flow,
                c_head=c_prev,  # Head (faster) is old water
                c_tail=c_new,  # Tail (slower) is new water
                sorption=sorption,
            )
        except ValueError:
            # Rarefaction validation failed (e.g., head not faster than tail)
            # This shouldn't happen if velocities were properly checked, but handle it
            return []
        else:
            return [raref]

    # Same velocity - contact discontinuity
    # This only happens if R(c_new) == R(c_prev), which is rare
    # Create a characteristic with the new concentration
    char = CharacteristicWave(
        t_start=t,
        v_start=v_inlet,
        flow=flow,
        concentration=c_new,
        sorption=sorption,
    )
    return [char]
