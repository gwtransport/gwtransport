"""Event detection for front tracking in (V, θ) coordinates.

All intersections are pure line/line geometry in the (V, θ) plane because
every wave speed dV/dθ is independent of flow. Functions return θ-coordinates
of intersections; the solver translates to user-facing t at the API boundary.

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

from gwtransport.fronttracking.math import characteristic_position, characteristic_speed
from gwtransport.fronttracking.waves import CharacteristicWave, DecayingShockWave, RarefactionWave, ShockWave

# Numerical tolerance constants
EPSILON_SPEED = 1e-15  # Tolerance for checking if two speeds are equal (machine precision)


class EventType(Enum):
    """All possible event types in front tracking simulation."""

    CHAR_CHAR_COLLISION = "characteristic_collision"
    """Two characteristics intersect (will form shock)."""
    SHOCK_SHOCK_COLLISION = "shock_collision"
    """Two shocks collide (will merge)."""
    SHOCK_CHAR_COLLISION = "shock_characteristic_collision"
    """Shock catches or is caught by characteristic."""
    RAREF_CHAR_COLLISION = "rarefaction_characteristic_collision"
    """Rarefaction boundary intersects with characteristic."""
    SHOCK_RAREF_COLLISION = "shock_rarefaction_collision"
    """Shock intersects with rarefaction boundary."""
    RAREF_RAREF_COLLISION = "rarefaction_rarefaction_collision"
    """Rarefaction boundary intersects with another rarefaction boundary."""
    DSW_FAN_EXHAUSTED = "decaying_shock_fan_exhausted"
    """A decaying shock's fan is exhausted (c_decay reached c_fan_tail)."""
    OUTLET_CROSSING = "outlet_crossing"
    """Wave crosses outlet boundary."""


@dataclass
class Event:
    """A single event in the simulation, ordered by cumulative flow θ.

    Parameters
    ----------
    theta : float
        Cumulative flow at which the event occurs [m³].
    event_type : EventType
        Type of event.
    waves_involved : list
        List of wave objects involved in this event.
    location : float
        Volumetric position at which the event occurs [m³].
    boundary_type : str or None
        Which rarefaction boundary collided: ``'head'`` or ``'tail'``.
        Set for rarefaction collision events.
    """

    theta: float
    event_type: EventType
    waves_involved: list  # List[Wave] - can't type hint due to circular import
    location: float
    boundary_type: str | None = None

    def __lt__(self, other):  # noqa: D105
        return self.theta < other.theta

    def __repr__(self):  # noqa: D105
        return (
            f"Event(θ={self.theta:.3f}, type={self.event_type.value}, "
            f"location={self.location:.3f}, n_waves={len(self.waves_involved)})"
        )


def find_characteristic_intersection(char1, char2, theta_current: float) -> tuple[float, float] | None:
    """Find exact analytical intersection of two characteristics in (V, θ).

    Returns (θ_intersect, V_intersect) if the intersection lies in the future
    (θ > θ_current) and both characteristics are active there; otherwise None.
    """
    s1 = characteristic_speed(char1.concentration, char1.sorption)
    s2 = characteristic_speed(char2.concentration, char2.sorption)

    if abs(s1 - s2) < EPSILON_SPEED:
        return None

    theta_both_active = max(char1.theta_start, char2.theta_start, theta_current)

    v1 = characteristic_position(
        char1.concentration, char1.sorption, char1.theta_start, char1.v_start, theta_both_active
    )
    v2 = characteristic_position(
        char2.concentration, char2.sorption, char2.theta_start, char2.v_start, theta_both_active
    )

    if v1 is None or v2 is None:
        return None

    # v1 + s1*dθ = v2 + s2*dθ
    dtheta = (v2 - v1) / (s1 - s2)

    if dtheta <= 0:
        return None

    theta_intersect = theta_both_active + dtheta
    v_intersect = v1 + s1 * dtheta

    return (theta_intersect, v_intersect)


def find_shock_shock_intersection(shock1, shock2, theta_current: float) -> tuple[float, float] | None:
    """Find exact analytical intersection of two shocks in (V, θ)."""
    s1 = shock1.speed
    s2 = shock2.speed

    if abs(s1 - s2) < EPSILON_SPEED:
        return None

    if not shock1.is_active or not shock2.is_active:
        return None

    theta_both_active = max(shock1.theta_start, shock2.theta_start, theta_current)

    v1_ref = shock1.v_start + shock1.speed * (theta_both_active - shock1.theta_start)
    v2_ref = shock2.v_start + shock2.speed * (theta_both_active - shock2.theta_start)

    dtheta = (v2_ref - v1_ref) / (s1 - s2)

    if dtheta <= 0:
        return None

    theta_intersect = theta_both_active + dtheta
    v_intersect = v1_ref + s1 * dtheta

    return (theta_intersect, v_intersect)


def find_shock_characteristic_intersection(shock, char, theta_current: float) -> tuple[float, float] | None:
    """Find exact analytical intersection of a shock and a characteristic in (V, θ)."""
    s_shock = shock.speed
    s_char = characteristic_speed(char.concentration, char.sorption)

    if abs(s_shock - s_char) < EPSILON_SPEED:
        return None

    theta_both_active = max(shock.theta_start, char.theta_start, theta_current)

    v_shock = shock.v_start + shock.speed * (theta_both_active - shock.theta_start)

    v_char = characteristic_position(
        char.concentration, char.sorption, char.theta_start, char.v_start, theta_both_active
    )

    if v_char is None or not shock.is_active or not char.is_active:
        return None

    dtheta = (v_char - v_shock) / (s_shock - s_char)

    if dtheta <= 0:
        return None

    theta_intersect = theta_both_active + dtheta
    v_intersect = v_shock + s_shock * dtheta

    return (theta_intersect, v_intersect)


def find_rarefaction_boundary_intersections(raref, other_wave, theta_current: float) -> list[tuple[float, float, str]]:
    """Intersections of a rarefaction's head/tail with another wave.

    Both rarefaction boundaries propagate at characteristic speeds (head at
    ``1/R(c_head)``, tail at ``1/R(c_tail)``), so we synthesize temporary
    ``CharacteristicWave`` instances and reuse the analytical helpers.

    Returns
    -------
    list of tuple
        ``(θ_intersect, V_intersect, boundary_type)`` for each intersection,
        where boundary_type is ``'head'`` or ``'tail'``.
    """
    intersections = []

    head_char = CharacteristicWave(
        theta_start=raref.theta_start,
        v_start=raref.v_start,
        concentration=raref.c_head,
        sorption=raref.sorption,
        is_active=raref.is_active,
    )

    tail_char = CharacteristicWave(
        theta_start=raref.theta_start,
        v_start=raref.v_start,
        concentration=raref.c_tail,
        sorption=raref.sorption,
        is_active=raref.is_active,
    )

    if isinstance(other_wave, CharacteristicWave):
        result = find_characteristic_intersection(head_char, other_wave, theta_current)
        if result:
            intersections.append((result[0], result[1], "head"))

        result = find_characteristic_intersection(tail_char, other_wave, theta_current)
        if result:
            intersections.append((result[0], result[1], "tail"))

    elif isinstance(other_wave, ShockWave):
        result = find_shock_characteristic_intersection(other_wave, head_char, theta_current)
        if result:
            intersections.append((result[0], result[1], "head"))

        result = find_shock_characteristic_intersection(other_wave, tail_char, theta_current)
        if result:
            intersections.append((result[0], result[1], "tail"))

    elif isinstance(other_wave, RarefactionWave):
        other_head_char = CharacteristicWave(
            theta_start=other_wave.theta_start,
            v_start=other_wave.v_start,
            concentration=other_wave.c_head,
            sorption=other_wave.sorption,
            is_active=other_wave.is_active,
        )

        other_tail_char = CharacteristicWave(
            theta_start=other_wave.theta_start,
            v_start=other_wave.v_start,
            concentration=other_wave.c_tail,
            sorption=other_wave.sorption,
            is_active=other_wave.is_active,
        )

        result = find_characteristic_intersection(head_char, other_head_char, theta_current)
        if result:
            intersections.append((result[0], result[1], "head"))

        result = find_characteristic_intersection(head_char, other_tail_char, theta_current)
        if result:
            intersections.append((result[0], result[1], "head"))

        result = find_characteristic_intersection(tail_char, other_head_char, theta_current)
        if result:
            intersections.append((result[0], result[1], "tail"))

        result = find_characteristic_intersection(tail_char, other_tail_char, theta_current)
        if result:
            intersections.append((result[0], result[1], "tail"))

    return intersections


def find_outlet_crossing(wave, v_outlet: float, theta_current: float) -> float | None:
    """Find the cumulative flow θ at which the wave crosses ``v_outlet``.

    Assumes positive flow (waves always move toward larger V). For rarefactions,
    returns the θ at which the head (leading edge) crosses. Returns None if
    the wave has already passed the outlet, is not active, or moves backward.
    The "already past" check uses a relative tolerance so that a wave whose
    crossing event has just been processed (and is at v_outlet ± a few ULPs)
    does not re-emit a duplicate crossing one ULP later.
    """
    if not wave.is_active:
        return None

    # Suppress re-emission when v_current is within FP of v_outlet: the
    # crossing was already recorded on the prior iteration.
    tol = 1e-12 * max(abs(v_outlet), abs(wave.v_start), 1.0)

    if isinstance(wave, CharacteristicWave):
        theta_eval = max(theta_current, wave.theta_start)
        v_current = characteristic_position(
            wave.concentration, wave.sorption, wave.theta_start, wave.v_start, theta_eval
        )

        if v_current is None or v_current >= v_outlet - tol:
            return None

        speed = characteristic_speed(wave.concentration, wave.sorption)

        if speed <= 0:
            return None

        dtheta = (v_outlet - v_current) / speed
        return theta_eval + dtheta

    if isinstance(wave, ShockWave):
        theta_eval = max(theta_current, wave.theta_start)
        v_current = wave.v_start + wave.speed * (theta_eval - wave.theta_start)

        if v_current >= v_outlet - tol:
            return None

        if wave.speed <= 0:
            return None

        dtheta = (v_outlet - v_current) / wave.speed
        return theta_eval + dtheta

    if isinstance(wave, RarefactionWave):
        theta_eval = max(theta_current, wave.theta_start)
        s_head = characteristic_speed(wave.c_head, wave.sorption)
        v_head = characteristic_position(wave.c_head, wave.sorption, wave.theta_start, wave.v_start, theta_eval)

        if v_head is None or v_head >= v_outlet - tol:
            return None

        if s_head <= 0:
            return None

        dtheta = (v_outlet - v_head) / s_head
        return theta_eval + dtheta

    if isinstance(wave, DecayingShockWave):
        # Closed-form inverse V_s(theta) = v_outlet.
        theta_cross = wave.outlet_crossing_theta(v_outlet)
        if theta_cross is None:
            return None
        # Suppress re-emission within FP of the prior crossing (same convention
        # as the linear-shock branch above).
        if theta_cross <= theta_current + 1e-15 * max(abs(theta_current), 1.0):
            return None
        return theta_cross

    return None
