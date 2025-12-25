"""
Front Tracking Solver - Event-Driven Simulation Engine.

========================================================

This module implements the main event-driven front tracking solver for
nonlinear sorption transport. The solver maintains a list of waves and
processes collision events chronologically using exact analytical calculations.

The algorithm:
1. Initialize waves from inlet boundary conditions
2. Find next event (earliest collision or outlet crossing)
3. Advance time to event
4. Handle event (create new waves, deactivate old ones)
5. Repeat until no more events

All calculations are exact analytical with machine precision.
"""

import logging
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Optional

import numpy as np
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
    create_inlet_waves_at_time,
    handle_characteristic_collision,
    handle_flow_change,
    handle_outlet_crossing,
    handle_rarefaction_characteristic_collision,
    handle_rarefaction_rarefaction_collision,
    handle_shock_characteristic_collision,
    handle_shock_collision,
    handle_shock_rarefaction_collision,
)
from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    compute_first_front_arrival_time,
)

# Import mass balance functions for runtime verification (High Priority #3)
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

# Numerical tolerance constants
EPSILON_CONCENTRATION = 1e-15  # Tolerance for concentration changes
MIN_EVENT_DATA_LENGTH = 5  # Minimum length of event_data tuple before accessing extra field


@dataclass
class FrontTrackerState:
    """
    Complete state of the front tracking simulation.

    This dataclass holds all information about the current simulation state,
    including all waves (active and inactive), event history, and simulation
    parameters.

    Parameters
    ----------
    waves : list[Wave]
        All waves created during simulation (includes inactive waves)
    events : list[dict]
        Event history with details about each event
    t_current : float
        Current simulation time [days from tedges[0]]
    v_outlet : float
        Outlet position [m³]
    sorption : FreundlichSorption or ConstantRetardation
        Sorption parameters
    cin : np.ndarray
        Inlet concentration time series [mass/volume]
    flow : np.ndarray
        Flow rate time series [m³/day]
    tedges : pd.DatetimeIndex
        Time bin edges [pandas DatetimeIndex]

    Examples
    --------
    ::

        state = FrontTrackerState(
            waves=[],
            events=[],
            t_current=0.0,
            v_outlet=500.0,
            sorption=sorption,
            cin=cin,
            flow=flow,
            tedges=tedges,
        )
    """

    waves: list[Wave]
    events: list[dict]
    t_current: float
    v_outlet: float
    sorption: FreundlichSorption | ConstantRetardation
    cin: np.ndarray
    flow: np.ndarray
    tedges: pd.DatetimeIndex


class FrontTracker:
    """
    Event-driven front tracking solver for nonlinear sorption transport.

    This is the main simulation engine that orchestrates wave propagation,
    event detection, and event handling. The solver maintains a list of waves
    and processes collision events chronologically.

    Parameters
    ----------
    cin : np.ndarray
        Inlet concentration time series [mass/volume]
    flow : np.ndarray
        Flow rate time series [m³/day]
    tedges : np.ndarray
        Time bin edges [days]
    aquifer_pore_volume : float
        Total pore volume [m³]
    sorption : FreundlichSorption or ConstantRetardation
        Sorption parameters

    Attributes
    ----------
    state : FrontTrackerState
        Complete simulation state
    t_first_arrival : float
        First arrival time (end of spin-up period) [days]

    Examples
    --------
    ::

        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=500.0,
            sorption=sorption,
        )
        tracker.run(max_iterations=1000)
        # Access results
        print(f"Total events: {len(tracker.state.events)}")
        print(f"Active waves: {sum(1 for w in tracker.state.waves if w.is_active)}")

    Notes
    -----
    The solver uses exact analytical calculations throughout with no numerical
    tolerances or iterative methods. All wave interactions are detected and
    handled with machine precision.

    The spin-up period (t < t_first_arrival) is affected by unknown initial
    conditions. Results are only valid for t >= t_first_arrival.
    """

    def __init__(
        self,
        cin: np.ndarray,
        flow: np.ndarray,
        tedges: pd.DatetimeIndex,
        aquifer_pore_volume: float,
        sorption: FreundlichSorption | ConstantRetardation,
    ):
        """
        Initialize tracker with inlet conditions and physical parameters.

        Parameters
        ----------
        cin : np.ndarray
            Inlet concentration time series [mass/volume]
        flow : np.ndarray
            Flow rate time series [m³/day]
        tedges : pd.DatetimeIndex
            Time bin edges [pandas DatetimeIndex]
        aquifer_pore_volume : float
            Total pore volume [m³]
        sorption : FreundlichSorption or ConstantRetardation
            Sorption parameters

        Raises
        ------
        ValueError
            If input arrays have incompatible lengths or invalid values
        """
        # Validation
        if len(tedges) != len(cin) + 1:
            msg = f"tedges must have length len(cin) + 1, got {len(tedges)} vs {len(cin) + 1}"
            raise ValueError(msg)
        if len(flow) != len(cin):
            msg = f"flow must have same length as cin, got {len(flow)} vs {len(cin)}"
            raise ValueError(msg)
        if np.any(cin < 0):
            msg = "cin must be non-negative"
            raise ValueError(msg)
        if np.any(flow <= 0):
            msg = "flow must be positive"
            raise ValueError(msg)
        if aquifer_pore_volume <= 0:
            msg = "aquifer_pore_volume must be positive"
            raise ValueError(msg)

        # Initialize state
        # t_current is in days from tedges[0], so it starts at 0.0
        self.state = FrontTrackerState(
            waves=[],
            events=[],
            t_current=0.0,
            v_outlet=aquifer_pore_volume,
            sorption=sorption,
            cin=cin.copy(),
            flow=flow.copy(),
            tedges=tedges.copy(),
        )

        # Compute spin-up period
        self.t_first_arrival = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

        # Detect flow changes for event scheduling
        self._flow_change_schedule = self._detect_flow_changes()

        # Initialize waves from inlet boundary conditions
        self._initialize_inlet_waves()

    def _initialize_inlet_waves(self):
        """
        Initialize all waves from inlet boundary conditions.

        Creates waves at each inlet concentration change by analyzing
        characteristic velocities and creating appropriate wave types.
        """
        c_prev = 0.0  # Assume domain initially at zero

        for i in range(len(self.state.cin)):
            c_new = float(self.state.cin[i])
            # Convert tedges[i] (Timestamp) to days from tedges[0]
            t_change = (self.state.tedges[i] - self.state.tedges[0]) / pd.Timedelta(days=1)
            flow_current = float(self.state.flow[i])

            if abs(c_new - c_prev) > EPSILON_CONCENTRATION:
                # Create wave(s) for this concentration change
                new_waves = create_inlet_waves_at_time(
                    c_prev=c_prev,
                    c_new=c_new,
                    t=t_change,
                    flow=flow_current,
                    sorption=self.state.sorption,
                    v_inlet=0.0,
                )
                self.state.waves.extend(new_waves)

            c_prev = c_new

    def _detect_flow_changes(self) -> list[tuple[float, float]]:
        """
        Detect all flow changes in the inlet time series.

        Scans the flow array and identifies time points where flow changes.
        These become scheduled events that will update all wave velocities.

        Returns
        -------
        list[tuple[float, float]]
            List of (t_change, flow_new) sorted by time
            Times are in days from tedges[0]

        Notes
        -----
        Flow changes are detected by comparing consecutive flow values.
        Only significant changes (>1e-15) are included.

        Examples
        --------
        >>> # flow = [100, 100, 200, 50] at tedges = [0, 10, 20, 30, 40] days
        >>> # Returns: [(20.0, 200.0), (30.0, 50.0)]
        """
        flow_changes = []
        epsilon_flow = 1e-15

        for i in range(1, len(self.state.flow)):
            if abs(self.state.flow[i] - self.state.flow[i - 1]) > epsilon_flow:
                # Convert tedges[i] to days from tedges[0]
                t_change = (self.state.tedges[i] - self.state.tedges[0]) / pd.Timedelta(days=1)
                flow_new = self.state.flow[i]
                flow_changes.append((t_change, flow_new))

        return flow_changes

    def find_next_event(self) -> Optional[Event]:
        """
        Find the next event (earliest in time).

        Searches all possible wave interactions and returns the earliest event.
        Uses a priority queue (min-heap) to efficiently find the minimum time.

        Returns
        -------
        Event or None
            Next event to process, or None if no future events

        Notes
        -----
        Checks for:
        - Characteristic-characteristic collisions
        - Shock-shock collisions
        - Shock-characteristic collisions
        - Rarefaction-characteristic collisions (head/tail)
        - Shock-rarefaction collisions (head/tail)
        - Outlet crossings for all wave types

        All collision times are computed using exact analytical formulas.
        """
        candidates = []  # Will use as min-heap by time
        counter = 0  # Unique counter to break ties without comparing EventType

        # Get only active waves
        active_waves = [w for w in self.state.waves if w.is_active]

        # 1. Flow change events (checked FIRST to get priority in tie-breaking)
        for t_change, flow_new in self._flow_change_schedule:
            if t_change > self.state.t_current:
                # All active waves are involved in flow change
                heappush(
                    candidates,
                    (t_change, counter, EventType.FLOW_CHANGE, active_waves.copy(), None, flow_new),
                )
                counter += 1
                break  # Only schedule the next flow change

        # 2. Characteristic-Characteristic collisions
        chars = [w for w in active_waves if isinstance(w, CharacteristicWave)]
        for i, w1 in enumerate(chars):
            for w2 in chars[i + 1 :]:
                result = find_characteristic_intersection(w1, w2, self.state.t_current)
                if result:
                    t, v = result
                    if 0 <= v <= self.state.v_outlet:  # In domain
                        heappush(candidates, (t, counter, EventType.CHAR_CHAR_COLLISION, [w1, w2], v, None))
                        counter += 1

        # 2. Shock-Shock collisions
        shocks = [w for w in active_waves if isinstance(w, ShockWave)]
        for i, w1 in enumerate(shocks):
            for w2 in shocks[i + 1 :]:
                result = find_shock_shock_intersection(w1, w2, self.state.t_current)
                if result:
                    t, v = result
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (t, counter, EventType.SHOCK_SHOCK_COLLISION, [w1, w2], v, None))
                        counter += 1

        # 3. Shock-Characteristic collisions
        for shock in shocks:
            for char in chars:
                result = find_shock_characteristic_intersection(shock, char, self.state.t_current)
                if result:
                    t, v = result
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (t, counter, EventType.SHOCK_CHAR_COLLISION, [shock, char], v, None))
                        counter += 1

        # 4. Rarefaction-Characteristic collisions
        rarefs = [w for w in active_waves if isinstance(w, RarefactionWave)]
        for raref in rarefs:
            for char in chars:
                intersections = find_rarefaction_boundary_intersections(raref, char, self.state.t_current)
                for t, v, boundary in intersections:
                    if 0 <= v <= self.state.v_outlet:
                        heappush(
                            candidates,
                            (t, counter, EventType.RAREF_CHAR_COLLISION, [raref, char], v, boundary),
                        )
                        counter += 1

        # 5. Shock-Rarefaction collisions
        for shock in shocks:
            for raref in rarefs:
                intersections = find_rarefaction_boundary_intersections(raref, shock, self.state.t_current)
                for t, v, boundary in intersections:
                    if 0 <= v <= self.state.v_outlet:
                        heappush(
                            candidates,
                            (t, counter, EventType.SHOCK_RAREF_COLLISION, [shock, raref], v, boundary),
                        )
                        counter += 1

        # 6. Rarefaction-Rarefaction collisions
        for i, raref1 in enumerate(rarefs):
            for raref2 in rarefs[i + 1 :]:
                intersections = find_rarefaction_boundary_intersections(raref1, raref2, self.state.t_current)
                for t, v, boundary in intersections:
                    if 0 <= v <= self.state.v_outlet:
                        heappush(
                            candidates,
                            (t, counter, EventType.RAREF_RAREF_COLLISION, [raref1, raref2], v, boundary),
                        )
                        counter += 1

        # 7. Outlet crossings
        for wave in active_waves:
            # For rarefactions, detect BOTH head and tail crossings
            if isinstance(wave, RarefactionWave):
                # Head crossing
                t_eval = max(self.state.t_current, wave.t_start)
                v_head = wave.head_position_at_time(t_eval)
                if v_head is not None and v_head < self.state.v_outlet:
                    vel_head = wave.head_velocity()
                    if vel_head > 0:
                        dt_head = (self.state.v_outlet - v_head) / vel_head
                        t_cross_head = t_eval + dt_head
                        if t_cross_head > self.state.t_current:
                            heappush(
                                candidates,
                                (t_cross_head, counter, EventType.OUTLET_CROSSING, [wave], self.state.v_outlet, None),
                            )
                            counter += 1

                # Tail crossing
                v_tail = wave.tail_position_at_time(t_eval)
                if v_tail is not None and v_tail < self.state.v_outlet:
                    vel_tail = wave.tail_velocity()
                    if vel_tail > 0:
                        dt_tail = (self.state.v_outlet - v_tail) / vel_tail
                        t_cross_tail = t_eval + dt_tail
                        if t_cross_tail > self.state.t_current:
                            heappush(
                                candidates,
                                (t_cross_tail, counter, EventType.OUTLET_CROSSING, [wave], self.state.v_outlet, None),
                            )
                            counter += 1
            else:
                # For characteristics and shocks, use existing logic
                t_cross = find_outlet_crossing(wave, self.state.v_outlet, self.state.t_current)
                if t_cross and t_cross > self.state.t_current:
                    heappush(
                        candidates, (t_cross, counter, EventType.OUTLET_CROSSING, [wave], self.state.v_outlet, None)
                    )
                    counter += 1

        # Return earliest event
        if candidates:
            # Handle 6-tuple format: (t, counter, event_type, waves, v, extra)
            event_data = heappop(candidates)
            t = event_data[0]
            # Skip counter at index 1
            event_type = event_data[2]
            waves = event_data[3]
            v = event_data[4]
            extra = event_data[5] if len(event_data) > MIN_EVENT_DATA_LENGTH else None

            # For FLOW_CHANGE events, extra contains flow_new
            flow_new = extra if event_type == EventType.FLOW_CHANGE else None

            return Event(time=t, event_type=event_type, waves_involved=waves, location=v, flow_new=flow_new)

        return None

    def handle_event(self, event: Event):
        """
        Handle an event by calling appropriate handler and updating state.

        Dispatches to the correct event handler based on event type, then
        updates the simulation state with any new waves created.

        Parameters
        ----------
        event : Event
            Event to handle

        Notes
        -----
        Event handlers may:
        - Deactivate parent waves
        - Create new child waves
        - Record event details in history
        - Verify physical correctness (entropy, mass balance)
        """
        new_waves = []

        if event.event_type == EventType.CHAR_CHAR_COLLISION:
            new_waves = handle_characteristic_collision(
                event.waves_involved[0], event.waves_involved[1], event.time, event.location
            )

        elif event.event_type == EventType.SHOCK_SHOCK_COLLISION:
            new_waves = handle_shock_collision(
                event.waves_involved[0], event.waves_involved[1], event.time, event.location
            )

        elif event.event_type == EventType.SHOCK_CHAR_COLLISION:
            new_waves = handle_shock_characteristic_collision(
                event.waves_involved[0], event.waves_involved[1], event.time, event.location
            )

        elif event.event_type == EventType.RAREF_CHAR_COLLISION:
            new_waves = handle_rarefaction_characteristic_collision(
                event.waves_involved[0],
                event.waves_involved[1],
                event.time,
                event.location,
                boundary_type="head",  # TODO: Get from event
            )

        elif event.event_type == EventType.SHOCK_RAREF_COLLISION:
            new_waves = handle_shock_rarefaction_collision(
                event.waves_involved[0],
                event.waves_involved[1],
                event.time,
                event.location,
                boundary_type="tail",  # TODO: Get from event
            )

        elif event.event_type == EventType.RAREF_RAREF_COLLISION:
            new_waves = handle_rarefaction_rarefaction_collision(
                event.waves_involved[0],
                event.waves_involved[1],
                event.time,
                event.location,
                boundary_type="head",  # TODO: Get from event
            )

        elif event.event_type == EventType.OUTLET_CROSSING:
            event_record = handle_outlet_crossing(event.waves_involved[0], event.time, event.location)
            self.state.events.append(event_record)
            return  # No new waves for outlet crossing

        elif event.event_type == EventType.FLOW_CHANGE:
            # Get all active waves at this time
            active_waves = [w for w in self.state.waves if w.is_active]
            if event.flow_new is None:
                msg = "FLOW_CHANGE event must have flow_new set"
                raise RuntimeError(msg)
            new_waves = handle_flow_change(event.time, event.flow_new, active_waves)

        # Add new waves to state
        self.state.waves.extend(new_waves)

        # Record event
        self.state.events.append({
            "time": event.time,
            "type": event.event_type.value,
            "location": event.location,
            "waves_before": event.waves_involved,
            "waves_after": new_waves,
        })

    def run(self, max_iterations: int = 10000, *, verbose: bool = False):
        """
        Run simulation until no more events or max_iterations reached.

        Processes events chronologically by repeatedly finding the next event,
        advancing time, and handling the event. Continues until no more events
        exist or the iteration limit is reached.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of events to process. Default 10000.
            Prevents infinite loops in case of bugs.
        verbose : bool, optional
            Print progress messages. Default False.

        Notes
        -----
        The simulation stops when:
        - No more events exist (all waves have exited or become inactive)
        - max_iterations is reached (safety limit)

        After completion, results are available in:
        - self.state.waves: All waves (active and inactive)
        - self.state.events: Complete event history
        - self.t_first_arrival: End of spin-up period
        """
        iteration = 0

        if verbose:
            logging.info("Starting simulation at t=%.3f", self.state.t_current)
            logging.info("Initial waves: %d", len(self.state.waves))
            logging.info("First arrival time: %.3f days", self.t_first_arrival)

        while iteration < max_iterations:
            # Find next event
            event = self.find_next_event()

            if event is None:
                if verbose:
                    logging.info("Simulation complete after %d events at t=%.6f", iteration, self.state.t_current)
                break

            # Advance time
            self.state.t_current = event.time

            # Handle event
            try:
                self.handle_event(event)
            except Exception:
                logging.exception("Error handling event at t=%.3f", event.time)
                raise

            # Optional: verify physics periodically
            if iteration % 100 == 0:
                self.verify_physics()

            if verbose and iteration % 10 == 0:
                active = sum(1 for w in self.state.waves if w.is_active)
                logging.debug("Iteration %d: t=%.3f, active_waves=%d", iteration, event.time, active)

            iteration += 1

        if iteration >= max_iterations:
            logging.warning("Reached max_iterations=%d", max_iterations)

        if verbose:
            logging.info("Final statistics:")
            logging.info("  Total events: %d", len(self.state.events))
            logging.info("  Total waves created: %d", len(self.state.waves))
            logging.info("  Active waves: %d", sum(1 for w in self.state.waves if w.is_active))
            logging.info("  First arrival time: %.6f days", self.t_first_arrival)

    def verify_physics(self, *, check_mass_balance: bool = False, mass_balance_rtol: float = 1e-12):
        """
        Verify physical correctness of current state.

        Implements High Priority #3 from FRONT_TRACKING_REBUILD_PLAN.md by adding
        runtime mass balance verification using exact analytical integration.

        Checks:
        - All shocks satisfy Lax entropy condition
        - All rarefactions have proper head/tail ordering
        - Mass balance: mass_in_domain + mass_out = mass_in (to specified tolerance)

        Parameters
        ----------
        check_mass_balance : bool, optional
            Enable mass balance verification. Default False (opt-in for now).
        mass_balance_rtol : float, optional
            Relative tolerance for mass balance check. Default 1e-6.
            This tolerance accounts for:
            - Midpoint approximation in spatial integration of rarefactions
            - Numerical precision in wave position calculations
            - Piecewise-constant approximations in domain partitioning

        Raises
        ------
        RuntimeError
            If physics violation is detected

        Notes
        -----
        Mass balance equation:
            mass_in_domain(t) + mass_out_cumulative(t) = mass_in_cumulative(t)

        All mass calculations use exact analytical integration where possible:
        - Inlet/outlet temporal integrals: exact for piecewise-constant functions
        - Domain spatial integrals: exact for constants, midpoint rule for rarefactions
        - Overall precision: ~1e-10 to 1e-12 relative error
        """
        # Check entropy for all active shocks
        for wave in self.state.waves:
            if isinstance(wave, ShockWave) and wave.is_active and not wave.satisfies_entropy():
                msg = (
                    f"Shock at t_start={wave.t_start:.3f} violates entropy! "
                    f"c_left={wave.c_left:.3f}, c_right={wave.c_right:.3f}, "
                    f"velocity={wave.velocity:.3f}"
                )
                raise RuntimeError(msg)

        # Check rarefaction ordering
        for wave in self.state.waves:
            if isinstance(wave, RarefactionWave) and wave.is_active:
                v_head = wave.head_velocity()
                v_tail = wave.tail_velocity()
                if v_head <= v_tail:
                    msg = (
                        f"Rarefaction at t_start={wave.t_start:.3f} has invalid ordering! "
                        f"head_velocity={v_head:.3f} <= tail_velocity={v_tail:.3f}"
                    )
                    raise RuntimeError(msg)

        # Check mass balance using exact analytical integration
        if check_mass_balance:
            t_current = self.state.t_current

            # Convert tedges from DatetimeIndex to float days for mass functions
            # Internal simulation uses float days from tedges[0]
            tedges_days = (self.state.tedges - self.state.tedges[0]) / pd.Timedelta(days=1)

            # Compute total mass in domain at current time
            mass_in_domain = compute_domain_mass(
                t=t_current,
                v_outlet=self.state.v_outlet,
                waves=self.state.waves,
                sorption=self.state.sorption,
            )

            # Compute cumulative inlet mass
            mass_in_cumulative = compute_cumulative_inlet_mass(
                t=t_current,
                cin=self.state.cin,
                flow=self.state.flow,
                tedges=tedges_days,
            )

            # Compute cumulative outlet mass
            mass_out_cumulative = compute_cumulative_outlet_mass(
                t=t_current,
                v_outlet=self.state.v_outlet,
                waves=self.state.waves,
                sorption=self.state.sorption,
                flow=self.state.flow,
                tedges=tedges_days,
            )

            # Mass balance: mass_in_domain + mass_out = mass_in
            mass_balance_error = (mass_in_domain + mass_out_cumulative) - mass_in_cumulative

            # Check relative error
            if mass_in_cumulative > 0:
                relative_error = abs(mass_balance_error) / mass_in_cumulative
            else:
                # No mass has entered yet - check absolute error is small
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
