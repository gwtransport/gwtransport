"""Wave–wave interaction resolution for multi-front nonlinear-sorption transport.

The event-driven solver in :mod:`gwtransport.fronttracking.solver` resolves collisions among
characteristics, shocks and rarefactions with the closed-form helpers in
:mod:`gwtransport.fronttracking.events`. This module adds the missing interaction classes —
anything a :class:`~gwtransport.fronttracking.waves.DecayingShockWave` or
:class:`~gwtransport.fronttracking.waves.DoubleFanShockWave` participates in — via one uniform
calculus (issue #294):

- **Faces and feeders.** Every wave is a set of *faces* (a shock face, a contact, or a
  rarefaction/fan boundary line). Each face separates a *left* (upstream) from a *right*
  (downstream) :class:`~gwtransport.fronttracking.waves.Feeder` — a constant state or a
  bounded self-similar fan.
- **Universal merge.** When a rear (upstream, faster) face overtakes a front (downstream)
  face, they merge into a single successor built from ``(rear.left_feeder, front.right_feeder)``.
  This one rule generates shock↔shock merges, shocks entering a fan (fan-entry), a rarefaction
  head catching a decaying shock (doubly-fed formation), same-apex decaying-shock annihilation,
  and every composition thereof.
- **Lipschitz-safe detection.** First crossings are found by a per-pair speed-bounded march
  (``|dg/dθ| ≤ Λ_pair``) so no crossing is skipped, with a grazing-minimum bracket for
  double-crossings inside one step. The bound is per-pair (not a global ``1``) because the
  percolation conductivity isotherms have ``R < 1`` regions.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from gwtransport.fronttracking.math import NonlinearSorption, SorptionModel, characteristic_speed
from gwtransport.fronttracking.waves import (
    CharacteristicWave,
    DecayingShockWave,
    DoubleFanShockWave,
    Feeder,
    RarefactionWave,
    ShockWave,
    Wave,
)

EPSILON_POSITION = 1e-15
# Largest speed used to size the Lipschitz march when a feeder range touches a saturated
# state (R = 0, λ = +∞ for van Genuchten-Mualem at S_e = 1). The step floor plus the D6
# monotonic-outlet-mass tripwire back this rare case; the sorption isotherms never hit it.
MAX_FINITE_SPEED = 1e6
MERGE_MATCH_TOL = 1e-9  # feeder-equality tolerance for degenerate (zero-jump) successors


@dataclass
class Face:
    """One separating surface of a wave, for event detection and the reader sweep.

    A face carries the wave it belongs to, a role tag, its position as a function of θ, the
    left (upstream) and right (downstream) feeders, whether its trajectory is curved (a
    fan-fed shock) and an upper bound on its speed (for the Lipschitz march).
    """

    wave: Wave
    role: str  # 'shock' | 'contact' | 'head' | 'tail' | 'boundary'
    left: Feeder
    right: Feeder
    is_curved: bool
    speed_bound: float
    line: tuple[float, float, float] | None = None
    """For a ``boundary`` face: ``(v_apex, theta_apex, speed)`` of the linear characteristic."""

    def position(self, theta: float) -> float | None:
        """Face position at ``theta`` (``None`` outside the wave's active θ-window)."""
        return _face_position(self, theta)


def _max_characteristic_speed(feeder: Feeder, sorption: SorptionModel) -> float:
    """Upper bound on ``|λ|`` over a feeder's concentration range (monotone in ``c``)."""
    if feeder.is_const:
        s = characteristic_speed(feeder.c_a, sorption)
    else:
        s = max(characteristic_speed(feeder.c_a, sorption), characteristic_speed(feeder.c_b, sorption))
    return min(s, MAX_FINITE_SPEED) if np.isfinite(s) else MAX_FINITE_SPEED


def _far_bound(feeder: Feeder, sorption: SorptionModel, *, upstream: bool) -> float:
    """Return the fan boundary concentration on the far (upstream/downstream) side.

    Upstream (the apex/tail side) is the larger-retardation bound; downstream (the head
    side) is the smaller-retardation bound. Monotonicity-agnostic (works for the n<1 mirror).
    """
    r_a = float(sorption.retardation(feeder.c_a))
    r_b = float(sorption.retardation(feeder.c_b))
    if upstream:
        return feeder.c_a if r_a >= r_b else feeder.c_b
    return feeder.c_a if r_a <= r_b else feeder.c_b


def iter_faces(wave: Wave, theta: float, *, include_boundaries: bool = True) -> list[Face]:
    """Enumerate the faces of ``wave`` at ``theta`` (shock/contact/boundary).

    ``theta`` selects the *historical* boundary state: a free fan boundary line is a face
    only for ``θ`` before it was consumed by a wave entering the fan (mirroring
    ``was_active_at``). ``include_boundaries=False`` drops boundary lines entirely.
    """
    if isinstance(wave, CharacteristicWave):
        left = Feeder.constant(wave.concentration)
        right = Feeder.constant(wave.c_ahead)
        return [Face(wave, "contact", left, right, is_curved=False, speed_bound=abs(wave.speed()))]

    if isinstance(wave, ShockWave):
        left = Feeder.constant(wave.c_left)
        right = Feeder.constant(wave.c_right)
        return [Face(wave, "shock", left, right, is_curved=False, speed_bound=abs(wave.speed))]

    if isinstance(wave, RarefactionWave):
        fan = Feeder.fan(wave.v_start, wave.theta_start, wave.c_tail, wave.c_head, wave.sorption)
        head = Face(
            wave, "head", fan, Feeder.constant(wave.c_head), is_curved=False, speed_bound=abs(wave.head_speed())
        )
        tail = Face(
            wave, "tail", Feeder.constant(wave.c_tail), fan, is_curved=False, speed_bound=abs(wave.tail_speed())
        )
        return [head, tail]

    if isinstance(wave, DecayingShockWave):
        return _decaying_shock_faces(wave, theta, include_boundaries=include_boundaries)

    if isinstance(wave, DoubleFanShockWave):
        return _double_fan_faces(wave, theta, include_boundaries=include_boundaries)

    return []


def _decaying_shock_faces(wave: DecayingShockWave, theta: float, *, include_boundaries: bool) -> list[Face]:
    """Shock face (curved) plus the free fan boundary line of a decaying shock at ``theta``."""
    s = wave.sorption
    c_lo = min(wave.c_fan_tail, wave.c_decay_initial)
    c_hi = max(wave.c_fan_tail, wave.c_decay_initial)
    boundary_free = not wave.fan_boundary_consumed and theta < wave.theta_fan_boundary_consumed
    fan = Feeder.fan(wave.v_origin, wave.theta_origin, c_lo, c_hi, s, far_boundary_free=boundary_free)
    fixed = Feeder.constant(wave.c_fixed)
    speed_bound = max(_max_characteristic_speed(fan, s), _max_characteristic_speed(fixed, s))
    if wave.decay_side == "left":
        shock = Face(wave, "shock", fan, fixed, is_curved=True, speed_bound=speed_bound)
    else:
        shock = Face(wave, "shock", fixed, fan, is_curved=True, speed_bound=speed_bound)
    faces = [shock]

    if include_boundaries and boundary_free:
        tail_c = Feeder.constant(wave.c_fan_tail)
        line_speed = characteristic_speed(wave.c_fan_tail, s)
        if np.isfinite(line_speed):
            # A wave entering through this boundary rides into a fan whose FAR edge is this
            # shock's own face (owned), so the entrant's successor must not re-expose a
            # boundary there: the boundary-side fan feeder is far_boundary_free=False.
            entry_fan = Feeder.fan(wave.v_origin, wave.theta_origin, c_lo, c_hi, s, far_boundary_free=False)
            if wave.decay_side == "left":
                # fan is upstream of the shock; its far (upstream) boundary carries c_fan_tail.
                left, right = tail_c, entry_fan
            else:
                left, right = entry_fan, tail_c
            faces.append(_boundary_line(wave, wave.v_origin, wave.theta_origin, line_speed, left, right))
    return faces


def _double_fan_faces(wave: DoubleFanShockWave, theta: float, *, include_boundaries: bool) -> list[Face]:
    """Shock face (curved) plus the free left/right fan boundary lines of a doubly-fed shock."""
    s = wave.sorption
    left_free = not wave.left_boundary_consumed and theta < wave.theta_left_boundary_consumed
    right_free = not wave.right_boundary_consumed and theta < wave.theta_right_boundary_consumed
    left_fan = Feeder.fan(
        wave.left_feeder.v_apex,
        wave.left_feeder.theta_apex,
        wave.left_feeder.c_a,
        wave.left_feeder.c_b,
        s,
        far_boundary_free=left_free,
    )
    right_fan = Feeder.fan(
        wave.right_feeder.v_apex,
        wave.right_feeder.theta_apex,
        wave.right_feeder.c_a,
        wave.right_feeder.c_b,
        s,
        far_boundary_free=right_free,
    )
    speed_bound = max(_max_characteristic_speed(left_fan, s), _max_characteristic_speed(right_fan, s))
    faces = [Face(wave, "shock", left_fan, right_fan, is_curved=True, speed_bound=speed_bound)]

    if include_boundaries and left_free:
        c_far = _far_bound(left_fan, s, upstream=True)
        line_speed = characteristic_speed(c_far, s)
        if np.isfinite(line_speed):
            entry_fan = Feeder.fan(
                left_fan.v_apex, left_fan.theta_apex, left_fan.c_a, left_fan.c_b, s, far_boundary_free=False
            )
            faces.append(
                _boundary_line(
                    wave, left_fan.v_apex, left_fan.theta_apex, line_speed, Feeder.constant(c_far), entry_fan
                )
            )
    if include_boundaries and right_free:
        c_far = _far_bound(right_fan, s, upstream=False)
        line_speed = characteristic_speed(c_far, s)
        if np.isfinite(line_speed):
            entry_fan = Feeder.fan(
                right_fan.v_apex, right_fan.theta_apex, right_fan.c_a, right_fan.c_b, s, far_boundary_free=False
            )
            faces.append(
                _boundary_line(
                    wave, right_fan.v_apex, right_fan.theta_apex, line_speed, entry_fan, Feeder.constant(c_far)
                )
            )
    return faces


def _boundary_line(wave: Wave, v_apex: float, theta_apex: float, speed: float, left: Feeder, right: Feeder) -> Face:
    """Build a linear fan boundary characteristic face (``V = v_apex + speed·(θ − θ_apex)``)."""
    return Face(
        wave, "boundary", left, right, is_curved=False, speed_bound=abs(speed), line=(v_apex, theta_apex, speed)
    )


def _face_position(face: Face, theta: float) -> float | None:
    """Position of any face at ``theta`` (dispatch on role/backing wave)."""
    wave = face.wave
    if not wave.was_active_at(theta):
        return None
    if face.role == "boundary" and face.line is not None:
        v_apex, theta_apex, speed = face.line
        return v_apex + speed * (theta - theta_apex)
    if isinstance(wave, RarefactionWave):
        return wave.head_position_at_theta(theta) if face.role == "head" else wave.tail_position_at_theta(theta)
    return wave.position_at_theta(theta)


def find_face_crossing(
    face_a: Face, face_b: Face, theta_current: float, theta_horizon: float
) -> tuple[float, float] | None:
    """First θ in ``(θ_start, θ_horizon]`` where two faces coincide, else ``None``.

    ``θ_start = max(θ_current, both faces born)``. The gap ``g(θ) = pos_b − pos_a`` is
    marched with a per-pair speed bound ``Λ = speed_bound_a + speed_bound_b`` so a step
    ``Δθ = max(|g|/Λ, floor)`` cannot straddle a sign change unseen; each step also brackets
    the gap's interior minimum to catch a grazing double-crossing.

    The loose near-coincidence admission at birth is enabled only for cross-wave pairs (exact
    coincidence is always reported). Same-wave exhaustion pairs — a wave's shock face against
    its own fan boundary line — suppress it: a front born a small distance from its own
    boundary line may be legitimately diverging, and the loose admission would fire a spurious
    immediate exhaustion.
    """
    wave_a, wave_b = face_a.wave, face_b.wave
    # Loose born-coincidence admission is a cross-wave notion (a newborn wave touching a
    # different neighbour it must immediately merge with); a wave's own two faces suppress it.
    allow_born_coincident = wave_a is not wave_b
    theta_start = max(theta_current, wave_a.theta_start, wave_b.theta_start)
    if theta_start >= theta_horizon:
        return None

    def gap(theta: float) -> float | None:
        pa = face_a.position(theta)
        pb = face_b.position(theta)
        if pa is None or pb is None:
            return None
        return pb - pa

    lam = face_a.speed_bound + face_b.speed_bound
    lam = lam if np.isfinite(lam) and lam > 0 else MAX_FINITE_SPEED
    floor = 1e-9 * max(abs(theta_start), 1.0)

    theta = theta_start
    g_prev = gap(theta)
    if g_prev is None:
        return None
    # Born-coincident / already-crossed event: when a wave is created by a near-simultaneous
    # (triple-) collision, it can be born within FP of — or just past — an adjacent wave it
    # must immediately merge with. The gap is already ≈0 or slightly negative at the search
    # start, so the forward march never sees a sign change. Detect it directly: if one wave
    # was just born (θ_start == search start) and the faces coincide within a small position
    # tolerance, emit the merge now (at the search start).
    pa0 = face_a.position(theta_start)
    born_now = abs(wave_a.theta_start - theta_start) <= floor or abs(wave_b.theta_start - theta_start) <= floor
    coincidence_tol = 1e-3 * max(abs(pa0) if pa0 is not None else 0.0, 1.0)
    if abs(g_prev) < EPSILON_POSITION or (allow_born_coincident and born_now and abs(g_prev) < coincidence_tol):
        return (theta, pa0) if pa0 is not None else None

    while theta < theta_horizon:
        step = max(abs(g_prev) / lam, floor)
        theta_next = min(theta + step, theta_horizon)
        g_next = gap(theta_next)
        if g_next is None:
            return None
        if g_prev * g_next <= 0.0:
            root = _bracket_zero(gap, theta, theta_next)
            if root is not None:
                v = face_a.position(root)
                return (root, v) if v is not None else None
        theta, g_prev = theta_next, g_next
    return None


def _bracket_zero(gap, lo: float, hi: float) -> float | None:
    """Root of ``gap`` in ``[lo, hi]`` (endpoints straddle zero, or one is ~0)."""
    g_lo = gap(lo)
    g_hi = gap(hi)
    if g_lo is None or g_hi is None:
        return None
    if abs(g_lo) < EPSILON_POSITION:
        return lo
    if abs(g_hi) < EPSILON_POSITION:
        return hi
    if g_lo * g_hi > 0.0:
        return None
    return float(brentq(gap, lo, hi, xtol=1e-13))


def make_wave_from_feeders(left: Feeder, right: Feeder, v: float, theta: float, sorption: SorptionModel) -> Wave | None:
    """Build the successor wave a merge produces from ``(left, right)`` feeders at ``(v, θ)``.

    ``(const, const)`` → :class:`ShockWave` (or ``None`` for a zero jump);
    ``(const, fan)``/``(fan, const)`` → :class:`DecayingShockWave`;
    ``(fan, fan)`` → :class:`DoubleFanShockWave`. Fan feeders whose far boundary is not free
    (``far_boundary_free=False``) mark the successor's boundary consumed on that side.
    """
    if left.is_const and right.is_const:
        if abs(left.c_a - right.c_a) < MERGE_MATCH_TOL:
            return None
        shock = ShockWave(theta_start=theta, v_start=v, c_left=left.c_a, c_right=right.c_a, sorption=sorption)
        return shock if shock.satisfies_entropy() else None

    if left.is_const or right.is_const:
        return _make_decaying_shock(left, right, v, theta, sorption)

    return _make_double_fan(left, right, v, theta, sorption)


def _make_decaying_shock(left: Feeder, right: Feeder, v: float, theta: float, sorption: SorptionModel) -> Wave | None:
    """Successor for one const + one fan feeder → a decaying shock (or a plain shock).

    The decaying side evolves from ``c_decay_initial`` (the fan value at the collision)
    toward ``c_fixed``. The fan only *feeds* that motion while there is a fan edge on the
    ``c_fixed`` side of ``c_decay_initial``; ``c_fan_tail`` is that edge (the exhaustion
    target — the decay asymptotes at ``c_fixed`` if it lies before the edge). If no fan edge
    lies toward ``c_fixed`` (the shock leaves the fan immediately — e.g. a collision exactly
    at a fan boundary), there is no fan-fed decay and the successor is a plain
    :class:`ShockWave` ``(c_decay_initial | c_fixed)``.
    """
    assert isinstance(sorption, NonlinearSorption)  # noqa: S101
    if left.is_const:
        fixed_c = left.c_a
        fan = right
        decay_side = "right"
    else:
        fixed_c = right.c_a
        fan = left
        decay_side = "left"
    c_decay_initial = fan.value(v, theta)
    if abs(c_decay_initial - fixed_c) < MERGE_MATCH_TOL:
        return None

    direction = fixed_c - c_decay_initial
    edges_toward_fixed = [b for b in (fan.c_a, fan.c_b) if (b - c_decay_initial) * direction > MERGE_MATCH_TOL]
    if not edges_toward_fixed:
        # The decay exits the fan at once — no fan-fed side, just a constant-state shock.
        # decay_side='left' has the decaying (fan) side upstream; 'right' downstream.
        if decay_side == "left":
            c_left, c_right = c_decay_initial, fixed_c
        else:
            c_left, c_right = fixed_c, c_decay_initial
        shock = ShockWave(theta_start=theta, v_start=v, c_left=c_left, c_right=c_right, sorption=sorption)
        return shock if shock.satisfies_entropy() else None
    # The fan edge farthest along the decay direction is the exhaustion target.
    c_fan_tail = max(edges_toward_fixed, key=lambda b: (b - c_decay_initial) * direction)

    return DecayingShockWave(
        theta_start=theta,
        v_start=v,
        c_decay_initial=c_decay_initial,
        c_fixed=fixed_c,
        c_fan_tail=c_fan_tail,
        decay_side=decay_side,
        v_origin=fan.v_apex,
        theta_origin=fan.theta_apex,
        sorption=sorption,
        fan_boundary_consumed=not fan.far_boundary_free,
    )


def _make_double_fan(left: Feeder, right: Feeder, v: float, theta: float, sorption: SorptionModel) -> Wave | None:
    """Successor for two fan feeders → a doubly-fed shock."""
    assert isinstance(sorption, NonlinearSorption)  # noqa: S101
    return DoubleFanShockWave(
        theta_start=theta,
        v_start=v,
        left_feeder=left,
        right_feeder=right,
        sorption=sorption,
        left_boundary_consumed=not left.far_boundary_free,
        right_boundary_consumed=not right.far_boundary_free,
    )


def resolve_merge(face_a: Face, face_b: Face, theta: float, v: float, sorption: SorptionModel) -> list[Wave]:
    """Resolve a two-face collision: deactivate/consume parents, return successor waves.

    Determines the rear (upstream) and front (downstream) face just before the crossing,
    forms the successor from ``(rear.left, front.right)``, and retires the parents: a whole
    wave is deactivated when its shock/contact/rarefaction face merges (its fan is absorbed
    into the successor); a bare fan boundary line is *consumed* while its owning wave lives on.
    """
    rear, front = _order_rear_front(face_a, face_b, theta)
    successors = make_wave_from_feeders(rear.left, front.right, v, theta, sorption)
    new_waves = [successors] if successors is not None else []

    _retire_parent(rear, theta)
    _retire_parent(front, theta)
    return new_waves


def _order_rear_front(face_a: Face, face_b: Face, theta: float) -> tuple[Face, Face]:
    """Return ``(rear, front)`` — the upstream (smaller V just before θ) face first."""
    eps = 1e-7 * max(abs(theta), 1.0)
    pa = face_a.position(theta - eps)
    pb = face_b.position(theta - eps)
    if pa is None or pb is None:
        pa = face_a.position(theta)
        pb = face_b.position(theta)
    if pa is None or pb is None:
        return face_a, face_b
    return (face_a, face_b) if pa <= pb else (face_b, face_a)


def _retire_parent(face: Face, theta: float) -> None:
    """Deactivate the wave, or consume just its boundary line if that is the merged face."""
    wave = face.wave
    if face.role == "boundary":
        _consume_boundary(wave, face, theta)
        return
    wave.deactivate(theta)


def _consume_boundary(wave: Wave, face: Face, theta: float) -> None:
    """Timestamp the crossed fan boundary as consumed at ``theta``; the wave lives on.

    Retrospective queries at ``θ' < theta`` still see the boundary as free (historical truth),
    so a later reader query does not retro-erase the boundary before it was actually consumed.
    """
    if isinstance(wave, DecayingShockWave):
        wave.theta_fan_boundary_consumed = theta
    elif isinstance(wave, DoubleFanShockWave):
        line = face.line
        matches_left = (
            line is not None
            and abs(line[0] - wave.left_feeder.v_apex) < EPSILON_POSITION
            and abs(line[1] - wave.left_feeder.theta_apex) < EPSILON_POSITION
        )
        if matches_left:
            wave.theta_left_boundary_consumed = theta
        else:
            wave.theta_right_boundary_consumed = theta
