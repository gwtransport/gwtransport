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
from gwtransport.fronttracking.waves import (
    CharacteristicWave,
    DecayingShockWave,
    DoubleFanShockWave,
    RarefactionWave,
    ShockWave,
)

# Numerical tolerance constants
EPSILON_SPEED = 1e-15  # Tolerance for checking if two speeds are equal (machine precision)
# A boundary state at/below the c_min retardation floor whose floored retardation
# exceeds this value is "pinned": for the n>1 dry-soil singularity R(c_min) is
# inflated to ~1e6, so the state advects orders of magnitude slower than any
# physical wave and its outlet crossing lands at a non-physical θ (~1e8) that only
# pollutes the diagnostic event record. n<1 clean water (R(c_min) ≈ 1, fast) stays
# well below this threshold and is NOT pinned, so its outlet crossing is kept.
OUTLET_PIN_RETARDATION = 1e4


def is_outlet_crossing_pinned(concentration: float, sorption) -> bool:
    """Whether a boundary state is pinned by the ``c_min`` retardation floor.

    A crossing scheduled for such a state is a non-physical artifact (its speed is
    a floor artifact, not physics); the caller drops it so it does not pollute the
    solver's event record / ``theta_current``.

    Parameters
    ----------
    concentration : float
        Boundary-state concentration [mass/volume].
    sorption : SorptionModel
        Sorption model (supplies ``c_min`` and ``retardation``).

    Returns
    -------
    bool
        ``True`` only when ``concentration`` is at/below ``c_min`` AND the floored
        retardation ``R(c_min)`` is inflated past :data:`OUTLET_PIN_RETARDATION`.
    """
    c_min = getattr(sorption, "c_min", 0.0)
    if concentration > c_min:
        return False
    return float(sorption.retardation(c_min)) > OUTLET_PIN_RETARDATION


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
    WAVE_MERGE = "wave_merge"
    """Two faces overtake (universal merge): any interaction involving a decaying/doubly-fed
    shock — fan-entry, doubly-fed formation, same-apex annihilation, side exhaustion (a
    doubly-fed shock crossing its own fan boundary line), and their compositions."""
    OUTLET_CROSSING = "outlet_crossing"
    """Wave crosses outlet boundary."""


@dataclass
class Event:
    """A single event in the simulation, ordered by cumulative flow θ.

    The solver's priority queue orders ``(theta, counter, ...)`` tuples, not
    ``Event`` objects, so this dataclass intentionally defines no ordering.

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
    faces: tuple | None = None  # (Face, Face) for WAVE_MERGE

    def __repr__(self):  # noqa: D105
        return (
            f"Event(θ={self.theta:.3f}, type={self.event_type.value}, "
            f"location={self.location:.3f}, n_waves={len(self.waves_involved)})"
        )


def _line_intersection(
    theta_start_a: float,
    v_start_a: float,
    speed_a: float,
    theta_start_b: float,
    v_start_b: float,
    speed_b: float,
    theta_current: float,
) -> tuple[float, float] | None:
    """First future crossing of two straight (V, θ) fronts, or ``None``.

    Every characteristic/shock/rarefaction-boundary travels at a flow-free constant
    speed, so each is a line ``V = v_start + speed·(θ − θ_start)``. Both are evaluated
    from the shared reference ``θ_both = max(θ_start_a, θ_start_b, θ_current)`` and the
    crossing is returned only when strictly in the future (``dθ > 0``).

    ``V_intersect`` is evaluated on line ``a``: the ``dθ`` is invariant under an a↔b swap
    (exact IEEE negation of numerator and denominator) but ``V_intersect`` is not, so the
    four public wrappers pass the same operand order their pre-refactor bodies used.
    """
    if abs(speed_a - speed_b) < EPSILON_SPEED:
        return None
    theta_both = max(theta_start_a, theta_start_b, theta_current)
    v_a = v_start_a + speed_a * (theta_both - theta_start_a)
    v_b = v_start_b + speed_b * (theta_both - theta_start_b)
    dtheta = (v_b - v_a) / (speed_a - speed_b)
    if dtheta <= 0:
        return None
    return (theta_both + dtheta, v_a + speed_a * dtheta)


def find_characteristic_intersection(char1, char2, theta_current: float) -> tuple[float, float] | None:
    """Find exact analytical intersection of two characteristics in (V, θ).

    Returns (θ_intersect, V_intersect) if the intersection lies in the future
    (θ > θ_current) and both characteristics are active there; otherwise None.
    """
    return _line_intersection(
        char1.theta_start, char1.v_start, char1.speed(), char2.theta_start, char2.v_start, char2.speed(), theta_current
    )


def find_shock_shock_intersection(shock1, shock2, theta_current: float) -> tuple[float, float] | None:
    """Find exact analytical intersection of two shocks in (V, θ)."""
    return _line_intersection(
        shock1.theta_start,
        shock1.v_start,
        shock1.speed,
        shock2.theta_start,
        shock2.v_start,
        shock2.speed,
        theta_current,
    )


def find_shock_characteristic_intersection(shock, char, theta_current: float) -> tuple[float, float] | None:
    """Find exact analytical intersection of a shock and a characteristic in (V, θ)."""
    return _line_intersection(
        shock.theta_start, shock.v_start, shock.speed, char.theta_start, char.v_start, char.speed(), theta_current
    )


def find_rarefaction_boundary_intersections(raref, other_wave, theta_current: float) -> list[tuple[float, float, str]]:
    """Intersections of a rarefaction's head/tail with another wave.

    Both rarefaction boundaries propagate at characteristic speeds (head at
    ``1/R(c_head)``, tail at ``1/R(c_tail)``), so each is a straight (V, θ) line
    fed directly into :func:`_line_intersection` — no temporary ``CharacteristicWave``
    objects. The operand order matches the per-branch order the closed-form helpers
    used before the refactor (``a`` is the raref boundary vs a characteristic, but the
    SHOCK vs a raref boundary, so ``V_intersect`` — the new wave's ``v_start`` — is
    bit-identical).

    Returns
    -------
    list of tuple
        ``(θ_intersect, V_intersect, boundary_type)`` for each intersection,
        where boundary_type is ``'head'`` or ``'tail'``.
    """
    intersections = []
    raref_boundaries = ((raref.head_speed(), "head"), (raref.tail_speed(), "tail"))

    if isinstance(other_wave, CharacteristicWave):
        for s_raref, tag in raref_boundaries:
            hit = _line_intersection(
                raref.theta_start,
                raref.v_start,
                s_raref,
                other_wave.theta_start,
                other_wave.v_start,
                other_wave.speed(),
                theta_current,
            )
            if hit:
                intersections.append((hit[0], hit[1], tag))

    elif isinstance(other_wave, ShockWave):
        # a=SHOCK, b=raref boundary (matches find_shock_characteristic_intersection's order).
        for s_raref, tag in raref_boundaries:
            hit = _line_intersection(
                other_wave.theta_start,
                other_wave.v_start,
                other_wave.speed,
                raref.theta_start,
                raref.v_start,
                s_raref,
                theta_current,
            )
            if hit:
                intersections.append((hit[0], hit[1], tag))

    elif isinstance(other_wave, RarefactionWave):
        other_boundaries = (other_wave.head_speed(), other_wave.tail_speed())
        for s_raref, tag in raref_boundaries:
            for s_other in other_boundaries:
                hit = _line_intersection(
                    raref.theta_start,
                    raref.v_start,
                    s_raref,
                    other_wave.theta_start,
                    other_wave.v_start,
                    s_other,
                    theta_current,
                )
                if hit:
                    intersections.append((hit[0], hit[1], tag))

    return intersections


def find_outlet_crossing(wave, v_outlet: float, theta_current: float) -> float | None:
    """Find the cumulative flow θ at which the wave crosses ``v_outlet``.

    Handles ``CharacteristicWave``, ``ShockWave``, and ``DecayingShockWave``.
    Rarefaction outlet crossings are handled by the callers directly (the
    solver and ``output.py`` split them into head/tail boundary crossings), so
    a ``RarefactionWave`` never reaches this function and returns ``None``.

    Assumes positive flow (waves always move toward larger V). Returns None if
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

        # A c_min-floored (pinned) characteristic — R(c_min) inflated for n>1,
        # c→0 — advects too slowly to cross at any physical θ; suppress the
        # artifact crossing rather than scheduling it at θ~1e8.
        if speed <= 0 or is_outlet_crossing_pinned(wave.concentration, wave.sorption):
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

    if isinstance(wave, (DecayingShockWave, DoubleFanShockWave)):
        # Closed-form / cached-trajectory inverse V_s(theta) = v_outlet.
        theta_cross = wave.outlet_crossing_theta(v_outlet)
        if theta_cross is None:
            return None
        # Suppress re-emission within FP of the prior crossing (same convention
        # as the linear-shock branch above).
        if theta_cross <= theta_current + 1e-15 * max(abs(theta_current), 1.0):
            return None
        return theta_cross

    return None
