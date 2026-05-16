"""Event-driven front-tracking solver in (V, θ) coordinates.

The simulation runs entirely in cumulative-flow space θ; only the public
state.events records and ``t_first_arrival`` expose user-facing time t.
Time-varying flow is absorbed into the precomputed ``theta_edges`` array
at ``__init__`` — there is no flow-change event.

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
from heapq import heappop, heappush

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.fronttracking.events import (
    Event,
    EventType,
    find_characteristic_intersection,
    find_outlet_crossing,
    find_rarefaction_boundary_intersections,
    find_shock_characteristic_intersection,
    find_shock_shock_intersection,
)
from gwtransport.fronttracking.handlers import (
    create_inlet_waves_at_theta,
    handle_characteristic_collision,
    handle_outlet_crossing,
    handle_rarefaction_characteristic_collision,
    handle_shock_characteristic_collision,
    handle_shock_collision,
    handle_shock_rarefaction_collision,
)
from gwtransport.fronttracking.math import (
    SorptionModel,
    compute_first_front_arrival_time,
)
from gwtransport.fronttracking.output import (
    compute_cumulative_inlet_mass,
    compute_cumulative_outlet_mass,
    compute_domain_mass,
)
from gwtransport.fronttracking.waves import (
    CharacteristicWave,
    RarefactionWave,
    ShockWave,
    Wave,
)

logger = logging.getLogger(__name__)

# Numerical tolerance constants
EPSILON_CONCENTRATION = 1e-15  # Tolerance for concentration changes


@dataclass
class FrontTrackerState:
    """Complete state of the front-tracking simulation in (V, θ).

    Parameters
    ----------
    waves : list of Wave
        All waves created during simulation (includes inactive waves).
    events : list of dict
        Event history with details about each event. Records use ``"time"``
        key carrying user-facing days (translated from θ at append time).
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

    @property
    def t_current(self) -> float:
        """User-facing time corresponding to ``theta_current`` [days from tedges[0]]."""
        return self.t_at_theta(self.theta_current)

    def t_at_theta(self, theta: float) -> float:
        """Translate cumulative flow θ back to user-facing time t [days].

        Piecewise linear inversion of the (tedges_days → theta_edges) map.
        For zero-flow bins θ is constant across ``[tedges_days[i], tedges_days[i+1])``;
        the inversion is right-continuous, returning the leftmost t with
        ``theta_at_t(t) == θ`` (i.e. ``tedges_days[i]``).
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
        i = int(np.searchsorted(self.theta_edges, theta, side="right")) - 1
        i = max(0, min(i, len(self.flow) - 1))
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

        i = int(np.searchsorted(self.tedges_days, t, side="right")) - 1
        i = max(0, min(i, len(self.flow) - 1))
        return float(self.theta_edges[i] + (t - self.tedges_days[i]) * float(self.flow[i]))


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
    t_first_arrival : float
        First arrival time (end of spin-up period) [days from tedges[0]].

    Notes
    -----
    Internally the solver works exclusively in cumulative flow θ; events
    appended to ``state.events`` carry user-facing time t (translated from
    θ via ``state.t_at_theta``).
    """

    def __init__(
        self,
        cin: np.ndarray,
        flow: np.ndarray,
        tedges: pd.DatetimeIndex,
        aquifer_pore_volume: float,
        sorption: SorptionModel,
    ):
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

        tedges_days = np.asarray((tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)
        dt_days = np.diff(tedges_days)
        bin_volumes = np.asarray(flow, dtype=float) * dt_days
        theta_edges = np.concatenate(([0.0], np.cumsum(bin_volumes)))

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
        )

        self.t_first_arrival = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        self._initialize_inlet_waves()

    def _initialize_inlet_waves(self):
        """Emit one wave per nonzero inlet step at the corresponding ``theta_edges[i]``."""
        c_prev = 0.0
        theta_edges = self.state.theta_edges

        for i in range(len(self.state.cin)):
            c_new = float(self.state.cin[i])
            theta_change = float(theta_edges[i])

            if abs(c_new - c_prev) > EPSILON_CONCENTRATION:
                new_waves = create_inlet_waves_at_theta(
                    c_prev=c_prev,
                    c_new=c_new,
                    theta=theta_change,
                    sorption=self.state.sorption,
                    v_inlet=0.0,
                )
                self.state.waves.extend(new_waves)

            c_prev = c_new

    def find_next_event(self) -> Event | None:
        """Return the next event in θ-order, or ``None`` if none."""
        candidates: list[tuple] = []
        counter = 0  # Unique counter to break ties in the heap

        active_waves = [w for w in self.state.waves if w.is_active]
        theta_current = self.state.theta_current

        chars = [w for w in active_waves if isinstance(w, CharacteristicWave)]
        for i, w1 in enumerate(chars):
            for w2 in chars[i + 1 :]:
                result = find_characteristic_intersection(w1, w2, theta_current)
                if result:
                    theta, v = result
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (theta, counter, EventType.CHAR_CHAR_COLLISION, [w1, w2], v, None))
                        counter += 1

        shocks = [w for w in active_waves if isinstance(w, ShockWave)]
        for i, w1 in enumerate(shocks):
            for w2 in shocks[i + 1 :]:
                result = find_shock_shock_intersection(w1, w2, theta_current)
                if result:
                    theta, v = result
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (theta, counter, EventType.SHOCK_SHOCK_COLLISION, [w1, w2], v, None))
                        counter += 1

        for shock in shocks:
            for char in chars:
                result = find_shock_characteristic_intersection(shock, char, theta_current)
                if result:
                    theta, v = result
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (theta, counter, EventType.SHOCK_CHAR_COLLISION, [shock, char], v, None))
                        counter += 1

        rarefs = [w for w in active_waves if isinstance(w, RarefactionWave)]
        for raref in rarefs:
            for char in chars:
                intersections = find_rarefaction_boundary_intersections(raref, char, theta_current)
                for theta, v, boundary in intersections:
                    if 0 <= v <= self.state.v_outlet:
                        heappush(
                            candidates,
                            (theta, counter, EventType.RAREF_CHAR_COLLISION, [raref, char], v, boundary),
                        )
                        counter += 1

        for shock in shocks:
            for raref in rarefs:
                intersections = find_rarefaction_boundary_intersections(raref, shock, theta_current)
                for theta, v, boundary in intersections:
                    if 0 <= v <= self.state.v_outlet:
                        heappush(
                            candidates,
                            (theta, counter, EventType.SHOCK_RAREF_COLLISION, [shock, raref], v, boundary),
                        )
                        counter += 1

        for i, raref1 in enumerate(rarefs):
            for raref2 in rarefs[i + 1 :]:
                intersections = find_rarefaction_boundary_intersections(raref1, raref2, theta_current)
                for theta, v, boundary in intersections:
                    if 0 <= v <= self.state.v_outlet:
                        heappush(
                            candidates,
                            (theta, counter, EventType.RAREF_RAREF_COLLISION, [raref1, raref2], v, boundary),
                        )
                        counter += 1

        for wave in active_waves:
            if isinstance(wave, RarefactionWave):
                # Both head and tail crossings of the outlet are tracked.
                theta_eval = max(theta_current, wave.theta_start)
                v_head = wave.head_position_at_theta(theta_eval)
                if v_head is not None and v_head < self.state.v_outlet:
                    s_head = wave.head_speed()
                    if s_head > 0:
                        theta_cross_head = theta_eval + (self.state.v_outlet - v_head) / s_head
                        if theta_cross_head > theta_current:
                            heappush(
                                candidates,
                                (
                                    theta_cross_head,
                                    counter,
                                    EventType.OUTLET_CROSSING,
                                    [wave],
                                    self.state.v_outlet,
                                    None,
                                ),
                            )
                            counter += 1

                v_tail = wave.tail_position_at_theta(theta_eval)
                if v_tail is not None and v_tail < self.state.v_outlet:
                    s_tail = wave.tail_speed()
                    if s_tail > 0:
                        theta_cross_tail = theta_eval + (self.state.v_outlet - v_tail) / s_tail
                        if theta_cross_tail > theta_current:
                            heappush(
                                candidates,
                                (
                                    theta_cross_tail,
                                    counter,
                                    EventType.OUTLET_CROSSING,
                                    [wave],
                                    self.state.v_outlet,
                                    None,
                                ),
                            )
                            counter += 1
            else:
                theta_cross = find_outlet_crossing(wave, self.state.v_outlet, theta_current)
                if theta_cross and theta_cross > theta_current:
                    heappush(
                        candidates,
                        (theta_cross, counter, EventType.OUTLET_CROSSING, [wave], self.state.v_outlet, None),
                    )
                    counter += 1

        if not candidates:
            return None

        theta_event, _, event_type, waves, v, extra = heappop(candidates)

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

        elif event.event_type == EventType.OUTLET_CROSSING:
            event_record = handle_outlet_crossing(event.waves_involved[0], event.theta, event.location)
            event_record["time"] = self.state.t_at_theta(event_record.pop("theta"))
            self.state.events.append(event_record)
            return

        # Add new waves to state
        self.state.waves.extend(new_waves)

        self.state.events.append({
            "time": self.state.t_at_theta(event.theta),
            "type": event.event_type.value,
            "location": event.location,
            "waves_before": event.waves_involved,
            "waves_after": new_waves,
        })

    def run(self, max_iterations: int = 10000, *, verbose: bool = False):
        """Process events in θ-order until the queue is empty or ``max_iterations`` is reached."""
        iteration = 0

        if verbose:
            logger.info("Starting simulation at θ=%.3f", self.state.theta_current)
            logger.info("Initial waves: %d", len(self.state.waves))
            logger.info("First arrival time: %.3f days", self.t_first_arrival)

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
            logger.info("  First arrival time: %.6f days", self.t_first_arrival)

    def verify_physics(self, *, check_mass_balance: bool = False, mass_balance_rtol: float = 1e-12):
        """Verify physical correctness: shock entropy and (optionally) mass balance.

        Mass balance equation::

            mass_in_domain(t) + mass_out_cumulative(t) = mass_in_cumulative(t)

        Raises
        ------
        RuntimeError
            If an active shock violates entropy or if mass balance fails to
            ``mass_balance_rtol`` at the current simulation time.
        """
        for wave in self.state.waves:
            if isinstance(wave, ShockWave) and wave.is_active and not wave.satisfies_entropy():
                msg = (
                    f"Shock at θ_start={wave.theta_start:.3f} violates entropy! "
                    f"c_left={wave.c_left:.3f}, c_right={wave.c_right:.3f}, "
                    f"speed={wave.speed:.6g}"
                )
                raise RuntimeError(msg)

        if check_mass_balance:
            t_current = self.state.t_current
            tedges_days = self.state.tedges_days

            mass_in_domain = compute_domain_mass(
                t=t_current,
                v_outlet=self.state.v_outlet,
                waves=self.state.waves,
                sorption=self.state.sorption,
            )

            mass_in_cumulative = compute_cumulative_inlet_mass(
                t=t_current,
                cin=self.state.cin,
                flow=self.state.flow,
                tedges_days=tedges_days,
            )

            mass_out_cumulative = compute_cumulative_outlet_mass(
                t=t_current,
                v_outlet=self.state.v_outlet,
                waves=self.state.waves,
                sorption=self.state.sorption,
                flow=self.state.flow,
                tedges_days=tedges_days,
            )

            mass_balance_error = (mass_in_domain + mass_out_cumulative) - mass_in_cumulative

            if mass_in_cumulative > 0:
                relative_error = abs(mass_balance_error) / mass_in_cumulative
            else:
                relative_error = abs(mass_balance_error)

            if relative_error > mass_balance_rtol:
                msg = (
                    f"Mass balance violation at t={t_current:.6f}! "
                    f"mass_in_domain={mass_in_domain:.6e}, "
                    f"mass_out={mass_out_cumulative:.6e}, "
                    f"mass_in={mass_in_cumulative:.6e}, "
                    f"error={mass_balance_error:.6e}, "
                    f"relative_error={relative_error:.6e} > {mass_balance_rtol:.6e}"
                )
                raise RuntimeError(msg)
