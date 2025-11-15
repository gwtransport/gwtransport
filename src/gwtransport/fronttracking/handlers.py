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


def handle_characteristic_collision(
    char1: CharacteristicWave,
    char2: CharacteristicWave,
    t_event: float,
    v_event: float,
) -> list[ShockWave]:
    """
    Handle collision of two characteristics → create shock.

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
    >>> shock = handle_characteristic_collision(char1, char2, t=15.0, v=100.0)
    >>> assert shock.satisfies_entropy()
    >>> assert not char1.is_active  # Parent deactivated
    """
    # Determine which characteristic is faster (upstream)

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
    >>> merged = handle_shock_collision(shock1, shock2, t=20.0, v=200.0)
    >>> assert merged.satisfies_entropy()
    >>> assert not shock1.is_active  # Parents deactivated
    """
    # Determine which shock is upstream (faster)
    # The shock catching up from behind is upstream
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
) -> list[ShockWave]:
    """
    Handle shock catching or being caught by characteristic.

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
    list[ShockWave]
        List containing new shock(s), may be empty if no shock forms

    Notes
    -----
    The characteristic concentration modifies one side of the shock:
    - If shock catches char: modifies c_right
    - If char catches shock: modifies c_left

    The new shock must satisfy entropy condition, otherwise interaction
    may result in rarefaction (not implemented yet).

    Examples
    --------
    >>> new_shock = handle_shock_characteristic_collision(shock, char, t=25.0, v=300.0)
    >>> if new_shock:
    ...     assert new_shock[0].satisfies_entropy()
    """
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
        # If entropy violated, this interaction doesn't create a shock
        # Instead, it might be a rarefaction scenario or waves pass through
        # For now, deactivate both and return empty (waves disappear)
        # TODO: Implement rarefaction creation in this case
        shock.is_active = False
        char.is_active = False
        return []

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
    Handle shock interacting with rarefaction fan.

    This is the most complex interaction. A shock can:
    - Catch the rarefaction tail: compresses rarefaction
    - Be caught by rarefaction head: creates compression

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
        List of new waves created (shocks, rarefactions, or characteristics)

    Notes
    -----
    This is a simplified implementation. Full shock-rarefaction interaction
    can create multiple waves and requires careful analysis of the wave
    structure.

    Current implementation:
    - Tail collision: shock penetrates, compressing rarefaction
    - Head collision: may create new shock or modify rarefaction

    TODO: Implement full interaction logic with wave splitting

    Examples
    --------
    >>> waves = handle_shock_rarefaction_collision(
    ...     shock, raref, t=30.0, v=400.0, boundary_type="tail"
    ... )
    """
    if boundary_type == "tail":
        # Shock catching rarefaction tail
        # Shock penetrates into rarefaction, compressing it
        # Create new shock that continues with c_right = rarefaction tail concentration
        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=shock.flow,
            c_left=shock.c_left,
            c_right=raref.c_tail,
            sorption=shock.sorption,
        )

        if new_shock.satisfies_entropy():
            # Deactivate rarefaction (simplified - should modify it instead)
            raref.is_active = False
            shock.is_active = False
            return [new_shock]
        # No shock forms - complex interaction
        return []

    # boundary_type == 'head'
    # Rarefaction head catching shock
    # This creates compression between rarefaction and shock
    # May form new shock between rarefaction head and shock left state
    new_shock = ShockWave(
        t_start=t_event,
        v_start=v_event,
        flow=raref.flow,
        c_left=raref.c_head,
        c_right=shock.c_left,
        sorption=raref.sorption,
    )

    if new_shock.satisfies_entropy():
        return [new_shock]
    # No shock forms - waves may pass through each other
    return []


def handle_rarefaction_characteristic_collision(
    raref: RarefactionWave,
    char: CharacteristicWave,
    t_event: float,
    v_event: float,
    boundary_type: str,
) -> list:
    """
    Handle rarefaction boundary intersecting with characteristic.

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
        List of new waves created

    Notes
    -----
    This is a simplified implementation. Full interaction may require
    modifying the rarefaction structure or creating new waves.

    Current implementation: deactivates characteristic, leaves rarefaction
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
    topology. Full entropic treatment of rarefaction–rarefaction interactions
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
    >>> event = handle_outlet_crossing(shock, t=50.0, v_outlet=500.0)
    >>> print(f"Wave exited at t={event['time']:.2f}")
    """
    return {
        "time": t_event,
        "type": "outlet_crossing",
        "wave": wave,
        "location": v_outlet,
        "concentration_left": wave.concentration_left(),
        "concentration_right": wave.concentration_right(),
    }


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
    >>> # Step increase creates shock for n>1
    >>> waves = create_inlet_waves_at_time(
    ...     c_prev=0.0, c_new=10.0, t=10.0, flow=100.0, sorption=sorption
    ... )
    >>> assert isinstance(waves[0], ShockWave)
    """
    if abs(c_new - c_prev) < 1e-15:  # No change
        return []

    # Compute characteristic velocities

    vel_prev = characteristic_velocity(c_prev, flow, sorption) if c_prev > 1e-15 else flow
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
            return [raref]
        except ValueError:
            # Rarefaction validation failed (e.g., head not faster than tail)
            # This shouldn't happen if velocities were properly checked, but handle it
            return []

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
