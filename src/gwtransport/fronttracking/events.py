"""
Event Detection for Front Tracking.

====================================

This module provides exact analytical event detection for the front tracking
algorithm. All intersections are computed using closed-form formulas with no
numerical iteration or tolerance-based checks.

Events include:
- Characteristic-characteristic collisions
- Shock-shock collisions
- Shock-characteristic collisions
- Rarefaction boundary interactions
- Outlet crossings

All calculations return exact floating-point results with machine precision.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from gwtransport.fronttracking.math import characteristic_position, characteristic_velocity
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave

# Numerical tolerance constants
EPSILON_VELOCITY = 1e-15  # Tolerance for checking if two velocities are equal (machine precision)


class EventType(Enum):
    """
    All possible event types in front tracking simulation.

    Attributes
    ----------
    CHAR_CHAR_COLLISION : str
        Two characteristics intersect (will form shock)
    SHOCK_SHOCK_COLLISION : str
        Two shocks collide (will merge)
    SHOCK_CHAR_COLLISION : str
        Shock catches or is caught by characteristic
    RAREF_CHAR_COLLISION : str
        Rarefaction boundary intersects with characteristic
    SHOCK_RAREF_COLLISION : str
        Shock intersects with rarefaction boundary
    RAREF_RAREF_COLLISION : str
        Rarefaction boundary intersects with another rarefaction boundary
    OUTLET_CROSSING : str
        Wave crosses outlet boundary
    INLET_CHANGE : str
        Inlet concentration changes (creates new wave)
    FLOW_CHANGE : str
        Flow rate changes (all waves get new velocities)
    """

    CHAR_CHAR_COLLISION = "characteristic_collision"
    SHOCK_SHOCK_COLLISION = "shock_collision"
    SHOCK_CHAR_COLLISION = "shock_characteristic_collision"
    RAREF_CHAR_COLLISION = "rarefaction_characteristic_collision"
    SHOCK_RAREF_COLLISION = "shock_rarefaction_collision"
    RAREF_RAREF_COLLISION = "rarefaction_rarefaction_collision"
    OUTLET_CROSSING = "outlet_crossing"
    INLET_CHANGE = "inlet_concentration_change"
    FLOW_CHANGE = "flow_change"


@dataclass
class Event:
    """
    Represents a single event in the simulation.

    Events are ordered by time for processing in chronological order.

    Parameters
    ----------
    time : float
        Time when event occurs [days]
    event_type : EventType
        Type of event
    waves_involved : list
        List of wave objects involved in this event
    location : float
        Volumetric position where event occurs [m³]
    flow_new : float, optional
        New flow rate for FLOW_CHANGE events [m³/day]

    Examples
    --------
    ::

        event = Event(
            time=15.5,
            event_type=EventType.SHOCK_CHAR_COLLISION,
            waves_involved=[shock1, char1],
            location=250.0,
        )
        print(f"Event at t={event.time}: {event.event_type.value}")
    """

    time: float
    event_type: EventType
    waves_involved: list  # List[Wave] - can't type hint due to circular import
    location: float
    flow_new: Optional[float] = None

    def __lt__(self, other):
        """Events ordered by time (for priority queue)."""
        return self.time < other.time

    def __repr__(self):
        """Return string representation of Event."""
        return (
            f"Event(t={self.time:.3f}, type={self.event_type.value}, "
            f"location={self.location:.3f}, n_waves={len(self.waves_involved)})"
        )


def find_characteristic_intersection(char1, char2, t_current: float) -> Optional[tuple[float, float]]:
    """
    Find exact analytical intersection of two characteristics.

    Solves the linear system:
        V1 = v1_start + vel1*(t - t1_start)
        V2 = v2_start + vel2*(t - t2_start)
        V1 = V2

    This reduces to:
        t = (v2_start - v1_start - vel2*t2_start + vel1*t1_start) / (vel1 - vel2)

    Parameters
    ----------
    char1 : CharacteristicWave
        First characteristic
    char2 : CharacteristicWave
        Second characteristic
    t_current : float
        Current simulation time [days]

    Returns
    -------
    tuple[float, float] or None
        (t_intersect, v_intersect) if intersection exists in future, None otherwise

    Notes
    -----
    Returns None if:
    - Characteristics are parallel (velocities equal within machine precision)
    - Intersection would occur in the past (t <= t_current)
    - Either characteristic is not yet active at intersection time

    The algorithm uses exact floating-point arithmetic with no tolerance checks
    except for detecting parallel lines (|vel1 - vel2| < 1e-15).

    Examples
    --------
    ::

        result = find_characteristic_intersection(char1, char2, t_current=10.0)
        if result:
            t_int, v_int = result
            print(f"Intersection at t={t_int:.6f}, V={v_int:.6f}")
    """
    # Import here to avoid circular dependency

    # Get velocities
    vel1 = characteristic_velocity(char1.concentration, char1.flow, char1.sorption)
    vel2 = characteristic_velocity(char2.concentration, char2.flow, char2.sorption)

    # Check if parallel (within machine precision)
    if abs(vel1 - vel2) < EPSILON_VELOCITY:
        return None

    # Both characteristics must be active at some common time
    t_both_active = max(char1.t_start, char2.t_start, t_current)

    # Compute positions when both are active

    v1 = characteristic_position(
        char1.concentration, char1.flow, char1.sorption, char1.t_start, char1.v_start, t_both_active
    )
    v2 = characteristic_position(
        char2.concentration, char2.flow, char2.sorption, char2.t_start, char2.v_start, t_both_active
    )

    if v1 is None or v2 is None:
        return None

    # Time until intersection from t_both_active
    # Solve: v1 + vel1*dt = v2 + vel2*dt
    dt = (v2 - v1) / (vel1 - vel2)

    if dt <= 0:  # Intersection in past or at current time
        return None

    t_intersect = t_both_active + dt
    v_intersect = v1 + vel1 * dt

    return (t_intersect, v_intersect)


def find_shock_shock_intersection(shock1, shock2, t_current: float) -> Optional[tuple[float, float]]:
    """
    Find exact analytical intersection of two shocks.

    Similar to characteristic intersection but uses shock velocities from
    Rankine-Hugoniot condition.

    Parameters
    ----------
    shock1 : ShockWave
        First shock
    shock2 : ShockWave
        Second shock
    t_current : float
        Current simulation time [days]

    Returns
    -------
    tuple[float, float] or None
        (t_intersect, v_intersect) if intersection exists in future, None otherwise

    Notes
    -----
    Shock velocities are constant (already computed from Rankine-Hugoniot),
    so this is a simple linear intersection problem.

    Examples
    --------
    ::

        result = find_shock_shock_intersection(shock1, shock2, t_current=10.0)
        if result:
            t_int, v_int = result
            print(f"Shocks collide at t={t_int:.6f}, V={v_int:.6f}")
    """
    vel1 = shock1.velocity
    vel2 = shock2.velocity

    # Check if parallel
    if abs(vel1 - vel2) < EPSILON_VELOCITY:
        return None

    t_both_active = max(shock1.t_start, shock2.t_start, t_current)

    # Compute positions when both are active
    v1_ref = shock1.v_start + shock1.velocity * (t_both_active - shock1.t_start)
    v2_ref = shock2.v_start + shock2.velocity * (t_both_active - shock2.t_start)

    if not shock1.is_active or not shock2.is_active:
        return None

    # Time until intersection from t_both_active
    dt = (v2_ref - v1_ref) / (vel1 - vel2)

    if dt <= 0:
        return None

    t_intersect = t_both_active + dt
    v_intersect = v1_ref + vel1 * dt

    return (t_intersect, v_intersect)


def find_shock_characteristic_intersection(shock, char, t_current: float) -> Optional[tuple[float, float]]:
    """
    Find exact analytical intersection of shock and characteristic.

    Parameters
    ----------
    shock : ShockWave
        Shock wave
    char : CharacteristicWave
        Characteristic wave
    t_current : float
        Current simulation time [days]

    Returns
    -------
    tuple[float, float] or None
        (t_intersect, v_intersect) if intersection exists in future, None otherwise

    Examples
    --------
    ::

        result = find_shock_characteristic_intersection(shock, char, t_current=10.0)
        if result:
            t_int, v_int = result
            print(f"Shock catches characteristic at t={t_int:.6f}, V={v_int:.6f}")
    """
    vel_shock = shock.velocity
    vel_char = characteristic_velocity(char.concentration, char.flow, char.sorption)

    # Check if parallel
    if abs(vel_shock - vel_char) < EPSILON_VELOCITY:
        return None

    t_both_active = max(shock.t_start, char.t_start, t_current)

    # Positions when both are active
    v_shock = shock.v_start + shock.velocity * (t_both_active - shock.t_start)

    v_char = characteristic_position(
        char.concentration, char.flow, char.sorption, char.t_start, char.v_start, t_both_active
    )

    if v_char is None or not shock.is_active or not char.is_active:
        return None

    # Time until intersection
    dt = (v_char - v_shock) / (vel_shock - vel_char)

    if dt <= 0:
        return None

    t_intersect = t_both_active + dt
    v_intersect = v_shock + vel_shock * dt

    return (t_intersect, v_intersect)


def find_rarefaction_boundary_intersections(raref, other_wave, t_current: float) -> list[tuple[float, float, str]]:
    """
    Find intersections of rarefaction head/tail with another wave.

    A rarefaction has two boundaries (head and tail), each traveling at
    characteristic velocities. This function finds intersections of both
    boundaries with the given wave.

    Parameters
    ----------
    raref : RarefactionWave
        Rarefaction wave
    other_wave : Wave
        Any other wave (Characteristic, Shock, or Rarefaction)
    t_current : float
        Current simulation time [days]

    Returns
    -------
    list[tuple[float, float, str]]
        List of (t_intersect, v_intersect, boundary_type) where boundary_type
        is either 'head' or 'tail'

    Notes
    -----
    The head travels at velocity corresponding to c_head, and the tail at
    velocity corresponding to c_tail. Both are treated as characteristics
    for intersection calculation.

    Examples
    --------
    ::

        intersections = find_rarefaction_boundary_intersections(
            raref, char, t_current=10.0
        )
        for t, v, boundary in intersections:
            print(f"{boundary} intersects at t={t:.3f}, V={v:.3f}")
    """
    # Import wave classes to avoid circular dependency

    intersections = []

    # Create temporary characteristics for head and tail boundaries
    head_char = CharacteristicWave(
        t_start=raref.t_start,
        v_start=raref.v_start,
        flow=raref.flow,
        concentration=raref.c_head,
        sorption=raref.sorption,
        is_active=raref.is_active,
    )

    tail_char = CharacteristicWave(
        t_start=raref.t_start,
        v_start=raref.v_start,
        flow=raref.flow,
        concentration=raref.c_tail,
        sorption=raref.sorption,
        is_active=raref.is_active,
    )

    # Check intersections based on other wave type
    if isinstance(other_wave, CharacteristicWave):
        # Head intersection
        result = find_characteristic_intersection(head_char, other_wave, t_current)
        if result:
            intersections.append((result[0], result[1], "head"))

        # Tail intersection
        result = find_characteristic_intersection(tail_char, other_wave, t_current)
        if result:
            intersections.append((result[0], result[1], "tail"))

    elif isinstance(other_wave, ShockWave):
        # Head intersection
        result = find_shock_characteristic_intersection(other_wave, head_char, t_current)
        if result:
            intersections.append((result[0], result[1], "head"))

        # Tail intersection
        result = find_shock_characteristic_intersection(other_wave, tail_char, t_current)
        if result:
            intersections.append((result[0], result[1], "tail"))

    elif isinstance(other_wave, RarefactionWave):
        # Rarefaction-rarefaction intersections: treat all boundaries as
        # characteristics and reuse the analytical intersection helpers.

        other_head_char = CharacteristicWave(
            t_start=other_wave.t_start,
            v_start=other_wave.v_start,
            flow=other_wave.flow,
            concentration=other_wave.c_head,
            sorption=other_wave.sorption,
            is_active=other_wave.is_active,
        )

        other_tail_char = CharacteristicWave(
            t_start=other_wave.t_start,
            v_start=other_wave.v_start,
            flow=other_wave.flow,
            concentration=other_wave.c_tail,
            sorption=other_wave.sorption,
            is_active=other_wave.is_active,
        )

        # head(head) and head(tail)
        result = find_characteristic_intersection(head_char, other_head_char, t_current)
        if result:
            intersections.append((result[0], result[1], "head"))

        result = find_characteristic_intersection(head_char, other_tail_char, t_current)
        if result:
            intersections.append((result[0], result[1], "head"))

        # tail(head) and tail(tail)
        result = find_characteristic_intersection(tail_char, other_head_char, t_current)
        if result:
            intersections.append((result[0], result[1], "tail"))

        result = find_characteristic_intersection(tail_char, other_tail_char, t_current)
        if result:
            intersections.append((result[0], result[1], "tail"))

    return intersections


def find_outlet_crossing(wave, v_outlet: float, t_current: float) -> Optional[float]:
    """
    Find exact analytical time when wave crosses outlet.

    For characteristics and shocks, solves:
        v_start + velocity*(t - t_start) = v_outlet

    For rarefactions, finds when head (leading edge) crosses.

    Parameters
    ----------
    wave : Wave
        Any wave type (Characteristic, Shock, or Rarefaction)
    v_outlet : float
        Outlet position [m³]
    t_current : float
        Current simulation time [days]

    Returns
    -------
    float or None
        Time when wave crosses outlet, or None if:
        - Wave already past outlet
        - Wave moving away from outlet
        - Wave not yet active

    Notes
    -----
    This function assumes waves always move in positive V direction (toward outlet).
    Negative velocities would indicate unphysical backward flow.

    Examples
    --------
    ::

        t_cross = find_outlet_crossing(shock, v_outlet=500.0, t_current=10.0)
        if t_cross:
            print(f"Shock exits at t={t_cross:.3f} days")
    """
    if not wave.is_active:
        return None

    if isinstance(wave, CharacteristicWave):
        # Get current position (use wave start time if not yet active)

        t_eval = max(t_current, wave.t_start)
        v_current = characteristic_position(
            wave.concentration, wave.flow, wave.sorption, wave.t_start, wave.v_start, t_eval
        )

        if v_current is None or v_current >= v_outlet:
            return None  # Already past outlet

        # Get velocity
        vel = characteristic_velocity(wave.concentration, wave.flow, wave.sorption)

        if vel <= 0:
            return None  # Moving backward (unphysical)

        # Solve: v_current + vel*(t - t_eval) = v_outlet
        dt = (v_outlet - v_current) / vel
        return t_eval + dt

    if isinstance(wave, ShockWave):
        # Current position (use wave start time if not yet active)
        t_eval = max(t_current, wave.t_start)
        v_current = wave.v_start + wave.velocity * (t_eval - wave.t_start)

        if v_current >= v_outlet:
            return None  # Already past outlet

        if wave.velocity <= 0:
            return None  # Moving backward (unphysical)

        # Solve: v_current + velocity*(t - t_eval) = v_outlet
        dt = (v_outlet - v_current) / wave.velocity
        return t_eval + dt

    if isinstance(wave, RarefactionWave):
        # Head crosses first (leading edge)
        t_eval = max(t_current, wave.t_start)
        vel_head = characteristic_velocity(wave.c_head, wave.flow, wave.sorption)

        v_head = characteristic_position(wave.c_head, wave.flow, wave.sorption, wave.t_start, wave.v_start, t_eval)

        if v_head is None or v_head >= v_outlet:
            return None

        if vel_head <= 0:
            return None

        dt = (v_outlet - v_head) / vel_head
        return t_eval + dt

    return None
