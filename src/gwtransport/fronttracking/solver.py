"""Event-driven front-tracking solver in (V, θ) coordinates.

The simulation runs entirely in cumulative-flow space θ. Every public
output — wave attributes, ``state.events[i]['theta']``,
``theta_first_arrival`` — is in θ. Translation to user-facing time t is
the caller's responsibility via ``state.t_at_theta``.
Time-varying flow is absorbed into the precomputed ``theta_edges`` array
at ``__init__``; there is no flow-change event.

Algorithm:

1. Initialize waves from inlet boundary conditions (one per cin step at θ_edges[i]).
2. Find next event (earliest collision or outlet crossing in θ).
3. Advance θ to event.
4. Handle event (create new waves, deactivate old ones).
5. Repeat until no more events.

All calculations are exact analytical with machine precision.
"""

import logging
from dataclasses import dataclass
from operator import itemgetter

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport._time import tedges_to_days
from gwtransport.fronttracking.events import (
    Event,
    EventType,
    find_characteristic_intersection,
    find_outlet_crossing,
    find_rarefaction_boundary_intersections,
    find_shock_characteristic_intersection,
    find_shock_shock_intersection,
    is_outlet_crossing_pinned,
)
from gwtransport.fronttracking.handlers import (
    EPSILON_CONCENTRATION,
    create_inlet_waves_at_theta,
    handle_characteristic_collision,
    handle_outlet_crossing,
    handle_rarefaction_characteristic_collision,
    handle_shock_characteristic_collision,
    handle_shock_collision,
    handle_shock_rarefaction_collision,
)
from gwtransport.fronttracking.interactions import (
    find_face_crossing,
    iter_faces,
    resolve_merge,
)
from gwtransport.fronttracking.math import (
    SorptionModel,
    compute_first_front_arrival_theta,
)
from gwtransport.fronttracking.output import (
    FP_CANCELLATION_CLAMP,
    compute_cumulative_outlet_mass,
)
from gwtransport.fronttracking.waves import (
    CharacteristicWave,
    DecayingShockWave,
    DoubleFanShockWave,
    RarefactionWave,
    ShockWave,
    Wave,
)

logger = logging.getLogger(__name__)


@dataclass
class FrontTrackerState:
    """Complete state of the front-tracking simulation in (V, θ).

    Parameters
    ----------
    waves : list of Wave
        All waves created during simulation (includes inactive waves).
    events : list of dict
        Event history. Records use the ``"theta"`` key carrying the
        cumulative flow at which the event occurred [m³]. Callers translate
        to user-facing time via ``FrontTrackerState.t_at_theta``.
    theta_current : float
        Current simulation cumulative flow [m³].
    v_outlet : float
        Outlet position [m³].
    sorption : SorptionModel
        Sorption model.
    cin : numpy.ndarray
        Inlet concentration time series [mass/volume].
    flow : numpy.ndarray
        Flow rate time series [m³/day], one value per bin.
    tedges : pandas.DatetimeIndex
        Time bin edges.
    tedges_days : numpy.ndarray
        ``tedges`` as days from ``tedges[0]``, length ``len(flow) + 1``.
    theta_edges : numpy.ndarray
        Cumulative flow at every bin edge. ``theta_edges[i] = sum_{k<i} flow[k] *
        (tedges_days[k+1] - tedges_days[k])``. Length ``len(flow) + 1``.
    """

    waves: list[Wave]
    events: list[dict]
    theta_current: float
    v_outlet: float
    sorption: SorptionModel
    cin: np.ndarray
    flow: np.ndarray
    tedges: pd.DatetimeIndex
    tedges_days: npt.NDArray[np.floating]
    theta_edges: npt.NDArray[np.floating]
    theta_horizon: float = float("inf")

    def t_at_theta(self, theta: float) -> float:
        """Translate cumulative flow θ back to user-facing time t [days].

        Piecewise linear inversion of the (tedges_days → theta_edges) map.
        Implementation note on the (rare) zero-flow case: when a bin has
        ``flow[i] == 0``, θ is constant across ``[tedges_days[i], tedges_days[i+1])``;
        ``np.searchsorted(..., side='right') - 1`` lands on the rightmost
        such bin, so this function returns ``tedges_days[i]`` for the
        right-most i sharing that θ. Convention: events at a zero-flow θ
        plateau map to the END of the zero-flow interval.
        """
        if theta <= self.theta_edges[0]:
            return float(self.tedges_days[0])
        if theta >= self.theta_edges[-1]:
            # Extrapolate: pretend the last bin continues at its current flow.
            last_flow = float(self.flow[-1])
            if last_flow > 0:
                return float(self.tedges_days[-1] + (theta - self.theta_edges[-1]) / last_flow)
            return float(self.tedges_days[-1])

        # Find bin index i with theta_edges[i] <= theta < theta_edges[i+1].
        # np.searchsorted with side='right' returns the smallest i+1 such that
        # theta_edges[i+1] > theta, so subtracting 1 gives i.
        # The boundary returns above guarantee strictly-interior theta here, so
        # searchsorted lands in [0, len(flow)-1] without an extra index clamp.
        i = int(np.searchsorted(self.theta_edges, theta, side="right")) - 1
        flow_i = float(self.flow[i])
        if flow_i <= 0:
            return float(self.tedges_days[i])
        return float(self.tedges_days[i] + (theta - self.theta_edges[i]) / flow_i)

    def theta_at_t(self, t: float) -> float:
        """Translate user-facing time t [days] to cumulative flow θ [m³].

        Piecewise linear forward map. Outside the input range the boundary
        flow is extrapolated.
        """
        if t <= self.tedges_days[0]:
            return float(self.theta_edges[0])
        if t >= self.tedges_days[-1]:
            return float(self.theta_edges[-1] + (t - self.tedges_days[-1]) * float(self.flow[-1]))

        # Boundary returns above guarantee strictly-interior t, so searchsorted
        # lands in [0, len(flow)-1] without an extra index clamp.
        i = int(np.searchsorted(self.tedges_days, t, side="right")) - 1
        return float(self.theta_edges[i] + (t - self.tedges_days[i]) * float(self.flow[i]))

    def theta_at_t_array(self, t: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Vectorized ``theta_at_t``: map an array of times t [days] to θ [m³].

        Element-wise identical to :meth:`theta_at_t`; replaces per-scalar loops
        in the plotting/output breakthrough routines.

        Parameters
        ----------
        t : array-like
            User-facing time points [days].

        Returns
        -------
        ndarray
            Cumulative flow θ at each ``t`` [m³].
        """
        t_arr = np.asarray(t, dtype=float)
        # Interior map: i = searchsorted(tedges_days, t, 'right') - 1, clipped to
        # a valid bin so the boundary branches can overwrite the extrapolated tails.
        i = np.clip(np.searchsorted(self.tedges_days, t_arr, side="right") - 1, 0, len(self.flow) - 1)
        theta = self.theta_edges[i] + (t_arr - self.tedges_days[i]) * self.flow[i]
        # Left of the first edge: clamp to theta_edges[0].
        theta = np.where(t_arr <= self.tedges_days[0], self.theta_edges[0], theta)
        # Right of the last edge: extrapolate at the final-bin flow.
        return np.where(
            t_arr >= self.tedges_days[-1],
            self.theta_edges[-1] + (t_arr - self.tedges_days[-1]) * float(self.flow[-1]),
            theta,
        )


class FrontTracker:
    """Event-driven front-tracking solver for nonlinear sorption transport.

    Parameters
    ----------
    cin : numpy.ndarray
        Inlet concentration time series [mass/volume]; length ``n``.
    flow : numpy.ndarray
        Flow rate time series [m³/day]; length ``n`` (one value per bin).
    tedges : pandas.DatetimeIndex
        Time bin edges (length ``n+1``).
    aquifer_pore_volume : float
        Total pore volume [m³] — used as the outlet position.
    sorption : SorptionModel
        Sorption model.

    Attributes
    ----------
    state : FrontTrackerState
        Complete simulation state.
    theta_first_arrival : float
        Cumulative flow θ at which the first nonzero-concentration wave reaches
        the outlet [m³]. Translate to user-facing time via
        ``state.t_at_theta(theta_first_arrival)``.

    Notes
    -----
    The solver works exclusively in cumulative flow θ; events appended to
    ``state.events`` carry ``"theta"``. Translation to user-facing time t is
    the caller's responsibility (use ``state.t_at_theta``).
    """

    def __init__(
        self,
        cin: npt.ArrayLike,
        flow: npt.ArrayLike,
        tedges: pd.DatetimeIndex,
        aquifer_pore_volume: float,
        sorption: SorptionModel,
        theta_horizon: float | None = None,
    ):
        cin = np.asarray(cin, dtype=float)
        flow = np.asarray(flow, dtype=float)
        if len(tedges) != len(cin) + 1:
            msg = f"tedges must have length len(cin) + 1, got {len(tedges)} vs {len(cin) + 1}"
            raise ValueError(msg)
        if len(flow) != len(cin):
            msg = f"flow must have same length as cin, got {len(flow)} vs {len(cin)}"
            raise ValueError(msg)
        if np.any(cin < 0):
            msg = "cin must be non-negative"
            raise ValueError(msg)
        if np.any(flow < 0):
            msg = "flow must be non-negative (negative flow not supported)"
            raise ValueError(msg)
        if aquifer_pore_volume <= 0:
            msg = "aquifer_pore_volume must be positive"
            raise ValueError(msg)

        tedges_days = tedges_to_days(tedges)
        dt_days = np.diff(tedges_days)
        bin_volumes = flow * dt_days
        theta_edges = np.concatenate(([0.0], np.cumsum(bin_volumes)))

        # Global interaction resolution runs to a θ-horizon: an unresolved crossing at
        # θ > horizon cannot affect any reader query at θ ≤ horizon (information propagates
        # strictly downstream, all speeds ≥ 0). Default = the last inlet edge; the public
        # API passes the last requested output edge so beyond-outlet merges that still feed
        # an in-window outlet query are resolved.
        resolved_horizon = float(theta_edges[-1]) if theta_horizon is None else float(theta_horizon)

        self.state = FrontTrackerState(
            waves=[],
            events=[],
            theta_current=0.0,
            v_outlet=aquifer_pore_volume,
            sorption=sorption,
            cin=cin,
            flow=flow,
            tedges=tedges,
            tedges_days=tedges_days,
            theta_edges=theta_edges,
            theta_horizon=resolved_horizon,
        )

        self.theta_first_arrival = compute_first_front_arrival_theta(cin, theta_edges, aquifer_pore_volume, sorption)

        self._initialize_inlet_waves()

    def _initialize_inlet_waves(self):
        """Emit one wave per nonzero inlet step at the corresponding ``theta_edges[i]``."""
        c_prev = 0.0
        theta_edges = self.state.theta_edges

        for i in range(len(self.state.cin)):
            # A zero-flow (pump-off) bin carries no water: its θ-interval has zero
            # width, so it is transport-invisible. Skipping it carries the inlet
            # step through the gap (effective transition c_before_gap → c_after_gap)
            # instead of emitting coincident spurious waves at the degenerate θ.
            if theta_edges[i + 1] <= theta_edges[i]:
                continue

            c_new = float(self.state.cin[i])

            if abs(c_new - c_prev) > EPSILON_CONCENTRATION:
                new_waves = create_inlet_waves_at_theta(
                    c_prev=c_prev,
                    c_new=c_new,
                    theta=float(theta_edges[i]),
                    sorption=self.state.sorption,
                )
                self.state.waves.extend(new_waves)

            c_prev = c_new

    def find_next_event(self) -> Event | None:
        """Return the next event in θ-order, or ``None`` if none."""
        # Each call collects every candidate and selects the single earliest via
        # ``min`` over the (theta, counter, ...) tuples; the unique ``counter``
        # breaks θ-ties deterministically (and stops comparison before the
        # non-orderable EventType/Wave fields). A heap would only pay for its
        # invariant to extract one minimum, so a flat list + ``min`` is leaner.
        candidates: list[tuple] = []
        counter = 0  # Unique counter to break θ-ties deterministically

        active_waves = [w for w in self.state.waves if w.is_active]
        theta_current = self.state.theta_current

        # Defense-in-depth loop guard: reject collision candidates at or below
        # θ_current + tol so a degenerate geometry cannot re-fire the identical
        # event forever (same FP-tolerance scale as the outlet-crossing guard
        # below). The structural fix is routing all shock↔rarefaction collisions
        # through DecayingShockWave; this backstops any future near-coincident
        # event from looping the solver.
        collision_tol = 1e-12 * max(abs(theta_current), 1.0)

        def push_collision(theta, event_type, waves, v, boundary):
            nonlocal counter
            # Global resolution (issue #294 D5): collisions are resolved wherever they occur,
            # up to θ_horizon — not only within [0, v_outlet]. The single-owner reader needs a
            # globally interaction-consistent front order, since the nearest downstream face
            # of an in-domain query may lie beyond the outlet (e.g. a rarefaction head that
            # catches a shock past v_outlet still bounds the in-domain fan). Information
            # propagates strictly downstream (all speeds ≥ 0), so an event at θ > θ_horizon
            # cannot affect any reader query at θ ≤ θ_horizon.
            if v < -collision_tol:
                return
            if theta <= theta_current + collision_tol or theta > self.state.theta_horizon:
                return
            candidates.append((theta, counter, event_type, waves, v, boundary, None))
            counter += 1

        chars = [w for w in active_waves if isinstance(w, CharacteristicWave)]
        for i, w1 in enumerate(chars):
            for w2 in chars[i + 1 :]:
                result = find_characteristic_intersection(w1, w2, theta_current)
                if result:
                    theta, v = result
                    push_collision(theta, EventType.CHAR_CHAR_COLLISION, [w1, w2], v, None)

        shocks = [w for w in active_waves if isinstance(w, ShockWave)]
        for i, w1 in enumerate(shocks):
            for w2 in shocks[i + 1 :]:
                result = find_shock_shock_intersection(w1, w2, theta_current)
                if result:
                    theta, v = result
                    push_collision(theta, EventType.SHOCK_SHOCK_COLLISION, [w1, w2], v, None)

        for shock in shocks:
            for char in chars:
                result = find_shock_characteristic_intersection(shock, char, theta_current)
                if result:
                    theta, v = result
                    push_collision(theta, EventType.SHOCK_CHAR_COLLISION, [shock, char], v, None)

        rarefs = [w for w in active_waves if isinstance(w, RarefactionWave)]
        for raref in rarefs:
            for char in chars:
                intersections = find_rarefaction_boundary_intersections(raref, char, theta_current)
                for theta, v, boundary in intersections:
                    push_collision(theta, EventType.RAREF_CHAR_COLLISION, [raref, char], v, boundary)

        for shock in shocks:
            for raref in rarefs:
                intersections = find_rarefaction_boundary_intersections(raref, shock, theta_current)
                for theta, v, boundary in intersections:
                    push_collision(theta, EventType.SHOCK_RAREF_COLLISION, [shock, raref], v, boundary)

        for i, raref1 in enumerate(rarefs):
            for raref2 in rarefs[i + 1 :]:
                intersections = find_rarefaction_boundary_intersections(raref1, raref2, theta_current)
                for theta, v, boundary in intersections:
                    push_collision(theta, EventType.RAREF_RAREF_COLLISION, [raref1, raref2], v, boundary)

        # Interaction events (issue #294): every face pair with at least one decaying/doubly-fed
        # shock. The closed-form loops above cover char/shock/rarefaction pairs; here a face of a
        # DSW/DFSW (its curved shock face or a free fan boundary line) meets any other wave's face.
        # Resolution is GLOBAL — allowed beyond v_outlet up to θ_horizon — because the sweep reader
        # needs a globally interaction-consistent front order for the nearest-downstream lookup.
        special_types = (DecayingShockWave, DoubleFanShockWave)
        if any(isinstance(w, special_types) for w in active_waves):
            all_faces = [face for wave in active_waves for face in iter_faces(wave, theta_current)]
            theta_horizon = self.state.theta_horizon
            for i, fa in enumerate(all_faces):
                for fb in all_faces[i + 1 :]:
                    if not (isinstance(fa.wave, special_types) or isinstance(fb.wave, special_types)):
                        continue
                    same_wave = fa.wave is fb.wave
                    if same_wave and not (
                        isinstance(fa.wave, DoubleFanShockWave) and {fa.role, fb.role} == {"shock", "boundary"}
                    ):
                        # Same-wave crossings other than a doubly-fed shock meeting its own fan
                        # boundary line never occur (a DSW's fan exhaustion has its own exact
                        # detector; two boundary lines of one wave diverge).
                        continue
                    # A doubly-fed shock crossing its OWN fan boundary line is that side's
                    # exhaustion: the fan between them is spent, and the universal merge
                    # (const far-bound feeder | surviving fan feeder) degrades the wave to a
                    # DecayingShockWave while retiring the boundary — uniformly with every
                    # other face merge. Entropy (λ(c_L) ≥ σ ≥ λ(c_R)) makes these crossings
                    # the only finite-θ side endings: the left fan's slow (upstream-far) char
                    # can only catch the shock from behind, the shock can only catch the
                    # right fan's downstream-far char — exactly the two exposed lines.
                    crossing = find_face_crossing(fa, fb, theta_current, theta_horizon)
                    if crossing is None:
                        continue
                    theta, v = crossing
                    if v < -collision_tol or theta > self.state.theta_horizon:
                        continue
                    # A born-coincident merge is reported at theta == theta_current (a wave
                    # created this step is already touching/past a neighbour it must merge
                    # with). Admit it — resolve_merge deactivates both parents, so it cannot
                    # re-fire — but keep the strict-future filter for every ordinary crossing
                    # so a just-processed event is not re-emitted one ULP later.
                    born_coincident = theta <= theta_current + collision_tol and (
                        abs(fa.wave.theta_start - theta_current) <= collision_tol
                        or abs(fb.wave.theta_start - theta_current) <= collision_tol
                    )
                    if theta <= theta_current + collision_tol and not born_coincident:
                        continue
                    candidates.append((theta, counter, EventType.WAVE_MERGE, [fa.wave, fb.wave], v, None, (fa, fb)))
                    counter += 1

        # Fan-exhaustion: a DecayingShockWave is valid only while c_decay stays
        # above c_fan_tail. When c_decay reaches c_fan_tail the fan is spent and
        # the wave hands off to a regular ShockWave(c_fan_tail, c_fixed).
        for wave in active_waves:
            if isinstance(wave, DecayingShockWave):
                theta_exhaust = wave.theta_at_fan_exhaustion()
                if theta_exhaust is None or theta_exhaust <= theta_current + collision_tol:
                    continue
                if theta_exhaust > self.state.theta_horizon:
                    continue
                v_exhaust = wave.position_at_theta(theta_exhaust)
                if v_exhaust is None or v_exhaust < -collision_tol:
                    continue
                candidates.append((theta_exhaust, counter, EventType.DSW_FAN_EXHAUSTED, [wave], v_exhaust, None, None))
                counter += 1

        v_outlet = self.state.v_outlet
        # Same FP-tolerance discipline as events.find_outlet_crossing — prevents
        # re-emitting an outlet crossing for a boundary that's at v_outlet ± ULPs.
        outlet_tol = 1e-12 * max(abs(v_outlet), 1.0)

        for wave in active_waves:
            if isinstance(wave, RarefactionWave):
                theta_eval = max(theta_current, wave.theta_start)
                for c_boundary, pos_fn, speed_fn in (
                    (wave.c_head, wave.head_position_at_theta, wave.head_speed),
                    (wave.c_tail, wave.tail_position_at_theta, wave.tail_speed),
                ):
                    v_pos = pos_fn(theta_eval)
                    if v_pos is None or v_pos >= v_outlet - outlet_tol:
                        continue
                    s = speed_fn()
                    # Skip a c_min-floored (pinned) boundary — R(c_min) inflated
                    # for n>1, c→0: its crossing lands at a non-physical θ~1e8 and
                    # only pollutes the event record.
                    if s <= 0 or is_outlet_crossing_pinned(c_boundary, wave.sorption):
                        continue
                    theta_cross = theta_eval + (v_outlet - v_pos) / s
                    if theta_cross > theta_current:
                        candidates.append((
                            theta_cross,
                            counter,
                            EventType.OUTLET_CROSSING,
                            [wave],
                            v_outlet,
                            None,
                            None,
                        ))
                        counter += 1
            else:
                theta_cross = find_outlet_crossing(wave, self.state.v_outlet, theta_current)
                if theta_cross and theta_cross > theta_current:
                    candidates.append((
                        theta_cross,
                        counter,
                        EventType.OUTLET_CROSSING,
                        [wave],
                        self.state.v_outlet,
                        None,
                        None,
                    ))
                    counter += 1

        if not candidates:
            return None

        theta_event, _, event_type, waves, v, extra, faces = min(candidates, key=itemgetter(slice(2)))

        raref_types = {
            EventType.RAREF_CHAR_COLLISION,
            EventType.SHOCK_RAREF_COLLISION,
            EventType.RAREF_RAREF_COLLISION,
        }
        boundary_type = extra if event_type in raref_types else None

        return Event(
            theta=theta_event,
            event_type=event_type,
            waves_involved=waves,
            location=v,
            boundary_type=boundary_type,
            faces=faces,
        )

    def handle_event(self, event: Event):
        """Dispatch an event to its handler and record it (with t translated from θ)."""
        new_waves: list = []

        if event.event_type == EventType.CHAR_CHAR_COLLISION:
            new_waves = handle_characteristic_collision(
                event.waves_involved[0], event.waves_involved[1], event.theta, event.location
            )

        elif event.event_type == EventType.SHOCK_SHOCK_COLLISION:
            new_waves = handle_shock_collision(
                event.waves_involved[0], event.waves_involved[1], event.theta, event.location
            )

        elif event.event_type == EventType.SHOCK_CHAR_COLLISION:
            new_waves = handle_shock_characteristic_collision(
                event.waves_involved[0], event.waves_involved[1], event.theta, event.location
            )

        elif event.event_type == EventType.RAREF_CHAR_COLLISION:
            new_waves = handle_rarefaction_characteristic_collision(
                event.waves_involved[0],
                event.waves_involved[1],
                event.theta,
                event.location,
                boundary_type=event.boundary_type,
            )

        elif event.event_type == EventType.SHOCK_RAREF_COLLISION:
            new_waves = handle_shock_rarefaction_collision(
                event.waves_involved[0],
                event.waves_involved[1],
                event.theta,
                event.location,
                boundary_type=event.boundary_type,
            )

        elif event.event_type == EventType.RAREF_RAREF_COLLISION:
            # Conservative: rarefaction-rarefaction collision records the event
            # but makes no topology change. Both rarefactions remain active.
            new_waves = []

        elif event.event_type == EventType.WAVE_MERGE:
            assert event.faces is not None  # noqa: S101  # WAVE_MERGE always carries its two faces
            face_a, face_b = event.faces
            new_waves = resolve_merge(face_a, face_b, event.theta, event.location, self.state.sorption)

        elif event.event_type == EventType.DSW_FAN_EXHAUSTED:
            new_waves = self._handle_fan_exhaustion(event.waves_involved[0], event.theta, event.location)

        elif event.event_type == EventType.OUTLET_CROSSING:
            event_record = handle_outlet_crossing(event.waves_involved[0], event.theta, event.location)
            self.state.events.append(event_record)
            return

        self.state.waves.extend(new_waves)

        self.state.events.append({
            "theta": event.theta,
            "type": event.event_type.value,
            "location": event.location,
            "waves_before": event.waves_involved,
            "waves_after": new_waves,
        })

    def _handle_fan_exhaustion(self, dsw: DecayingShockWave, theta_event: float, v_event: float) -> list[Wave]:
        """Hand a fan-exhausted decaying shock off to a regular shock.

        When ``c_decay`` reaches ``c_fan_tail`` the decaying side is no longer
        fed by the fan; the wave continues as a constant-speed
        ``ShockWave(c_fan_tail, c_fixed)`` (sides assigned per ``decay_side``).
        The handoff is C1-continuous (``dV_s/dθ → S(c_fan_tail, c_fixed)`` = the
        spawned shock's speed). The decaying shock is deactivated.

        Returns
        -------
        list of Wave
            ``[ShockWave]`` for the continuation, or ``[]`` if it fails entropy.
        """
        if dsw.decay_side == "left":
            c_left, c_right = dsw.c_fan_tail, dsw.c_fixed
        else:
            c_left, c_right = dsw.c_fixed, dsw.c_fan_tail

        dsw.deactivate(theta_event)

        if abs(c_left - c_right) < EPSILON_CONCENTRATION:
            # Fan decayed onto the fixed state — no discontinuity remains.
            return []

        shock = ShockWave(
            theta_start=theta_event,
            v_start=v_event,
            c_left=c_left,
            c_right=c_right,
            sorption=self.state.sorption,
        )
        if not shock.satisfies_entropy():
            return []
        return [shock]

    def run(self, max_iterations: int = 10000, *, verbose: bool = False):
        """Process events in θ-order until the queue is empty or ``max_iterations`` is reached."""
        iteration = 0

        if verbose:
            logger.info("Starting simulation at θ=%.3f", self.state.theta_current)
            logger.info("Initial waves: %d", len(self.state.waves))
            logger.info("First arrival: θ=%.3f", self.theta_first_arrival)

        while iteration < max_iterations:
            event = self.find_next_event()

            if event is None:
                if verbose:
                    logger.info("Simulation complete after %d events at θ=%.6f", iteration, self.state.theta_current)
                break

            self.state.theta_current = event.theta

            try:
                self.handle_event(event)
            except Exception:
                logger.exception("Error handling event at θ=%.3f", event.theta)
                raise

            if iteration % 100 == 0:
                self.verify_physics()

            if verbose and iteration % 10 == 0:
                active = sum(1 for w in self.state.waves if w.is_active)
                logger.debug("Iteration %d: θ=%.3f, active_waves=%d", iteration, event.theta, active)

            iteration += 1

        if iteration >= max_iterations:
            logger.warning("Reached max_iterations=%d", max_iterations)

        if verbose:
            logger.info("Final statistics:")
            logger.info("  Total events: %d", len(self.state.events))
            logger.info("  Total waves created: %d", len(self.state.waves))
            logger.info("  Active waves: %d", sum(1 for w in self.state.waves if w.is_active))
            logger.info("  First arrival: θ=%.6f", self.theta_first_arrival)

    def verify_physics(self):
        """Verify physical correctness: every active shock satisfies Lax entropy.

        Mass conservation is intentionally NOT checked here. The closed-form
        identity ``m_out(θ) = m_in(θ) − m_dom(θ)`` makes any runtime
        ``m_in_domain + m_out_cumulative == m_in_cumulative`` test tautological
        (residual identically zero, regardless of any ``compute_domain_mass``
        bug), so it cannot catch a conservation error. The non-tautological,
        integral-based conservation check (an independent breakthrough integral
        compared to the inlet mass) lives in
        :func:`gwtransport.fronttracking.validation.verify_physics` check 7 and
        is exercised by ``TestEndToEndConservation`` /
        ``TestIndependentDomainMass``.

        Raises
        ------
        RuntimeError
            If an active shock violates the Lax entropy condition.
        """
        for wave in self.state.waves:
            if isinstance(wave, ShockWave) and wave.is_active and not wave.satisfies_entropy():
                msg = (
                    f"Shock at θ_start={wave.theta_start:.3f} violates entropy! "
                    f"c_left={wave.c_left:.3f}, c_right={wave.c_right:.3f}, "
                    f"speed={wave.speed:.6g}"
                )
                raise RuntimeError(msg)


def find_unresolved_interaction(state: FrontTrackerState) -> str | None:
    """Tripwire for a solver-left inconsistency in the resolved wave field (issue #294 D6).

    The solver now resolves *every* wave interaction (shock↔shock, fan-entry, doubly-fed
    formation, same-apex annihilation, and their compositions), so the wave list is
    interaction-consistent and the single-owner sweep reader is exact — overlapping fans are
    normal and correct. This function is therefore no longer an input filter but an internal
    invariant check: the cumulative outlet mass ``m_out(θ) = m_in(θ) − m_dom(θ)`` must be
    non-decreasing in θ (mass leaves the column, it never re-enters). A decrease beyond the
    FP-cancellation band means the reader's domain-mass field transiently over-counts stored
    mass — the fingerprint of an interaction the solver failed to resolve (a bug). The public
    API turns a non-``None`` return into a fail-loud ``RuntimeError`` rather than returning a
    silently wrong ``cout``.

    The geometric fan-overlap scan of the previous (input-refusing) version is intentionally
    dropped: with interactions resolved, overlapping fans read identically from either
    neighbour and are not an error, so that scan is now only a false positive.

    Parameters
    ----------
    state : FrontTrackerState
        Completed simulation state (after :meth:`FrontTracker.run`).

    Returns
    -------
    str or None
        A short description (θ and mechanism) of the first monotonicity violation, or
        ``None`` when the resolved field conserves mass (the normal case).

    Notes
    -----
    The scan stays strictly inside the inlet θ-window, so the benign out-of-window saturation
    clamp (a run whose output bins extend past the last injected mass) does not trip it.
    """
    waves = state.waves
    v_outlet = state.v_outlet
    theta_hi = float(state.theta_edges[-1])
    thetas = np.linspace(theta_hi / 400.0, theta_hi, 400)

    # Conservation symptom: cumulative outlet mass must be monotone non-decreasing. The band
    # mirrors ``compute_bin_averaged_concentration_exact``'s FP-cancellation clamp (same
    # constant, same ``max(scale, 1.0)`` floor) scaled to the total injected mass, so
    # pre-breakthrough cancellation dust stays silent while a genuine over-count (orders of
    # magnitude above the band) is caught.
    grid = np.concatenate(([0.0], thetas))
    m_out = np.array([
        compute_cumulative_outlet_mass(
            float(theta), v_outlet, waves, state.sorption, cin=state.cin, theta_edges=state.theta_edges
        )
        for theta in grid
    ])
    m_in_total = float(np.sum(state.cin * np.diff(state.theta_edges)))
    # Tolerance band: the larger of the FP-cancellation floor and a 1e-6 relative transient
    # allowance. The latter absorbs the benign ``c_min``-floor artifact at a fan-entry (the
    # decaying side is born at ``c ≈ 1e-12`` rather than exactly 0, perturbing the near-apex
    # fan integral by ~1e-3 for one θ-sample). A genuine unresolved interaction over-counts by
    # O(a pulse's mass) — orders of magnitude above this band — so the tripwire still fires.
    band = max(FP_CANCELLATION_CLAMP * np.finfo(float).eps, 1e-6) * max(m_in_total, 1.0)
    # A real over-count is SUSTAINED — m_out stays below its running maximum across a θ-range.
    # A single isolated sub-maximum sample is a measure-zero knife-edge at an exact collision θ
    # (the parents are deactivated and the successor just born); it does not affect the
    # integral-based bin-averaged cout, so require two consecutive samples below the running
    # max before flagging. (The actual cout the API returns never sees the knife-edge.)
    running_max = np.maximum.accumulate(m_out)
    below = m_out < running_max - band
    sustained = below[:-1] & below[1:]
    if np.any(sustained):
        idx = int(np.argmax(sustained))
        drop = float(running_max[idx] - m_out[idx + 1])
        return (
            f"cumulative outlet mass drops by {drop:.4g} near θ={grid[idx + 1]:.4g} m³ "
            "(the reader over-counts stored mass there — an interaction the solver failed to resolve)"
        )
    return None
