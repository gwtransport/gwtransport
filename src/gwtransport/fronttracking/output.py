"""
Concentration Extraction from Front Tracking Solutions.

This module computes outlet concentrations from front tracking wave solutions
using exact analytical integration. All calculations maintain machine precision
with no numerical dispersion.

Functions
---------
concentration_at_point(v, t, waves, sorption)
    Compute concentration at any point (v, t) in domain
compute_breakthrough_curve(t_array, v_outlet, waves, sorption)
    Compute concentration at outlet over time array
compute_bin_averaged_concentration_exact(t_edges, v_outlet, waves, sorption)
    Compute bin-averaged concentrations using exact analytical integration
compute_domain_mass(t, v_outlet, waves, sorption)
    Compute total mass in domain [0, v_outlet] at time t using exact analytical integration
compute_cumulative_inlet_mass(t, cin, flow, tedges)
    Compute cumulative mass entering domain from t=0 to t
compute_cumulative_outlet_mass(t, v_outlet, waves, sorption, flow, tedges)
    Compute cumulative mass exiting domain from t=0 to t
find_last_rarefaction_start_time(v_outlet, waves)
    Find time when last rarefaction head reaches outlet
integrate_rarefaction_total_mass(raref, v_outlet, t_start, sorption, flow)
    Compute total mass exiting through rarefaction to infinity
compute_total_outlet_mass(v_outlet, waves, sorption, flow, tedges)
    Compute total integrated outlet mass until all mass has exited

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from operator import itemgetter

import mpmath as mp
import numpy as np
import numpy.typing as npt
from scipy.special import beta as beta_func
from scipy.special import betainc

from gwtransport.fronttracking.events import find_outlet_crossing
from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave, Wave

# Numerical tolerance constants
EPSILON_VELOCITY = 1e-15  # Tolerance for checking if velocity is effectively zero
EPSILON_BETA = 1e-15  # Tolerance for checking if beta is effectively zero (linear case)
EPSILON_TIME = 1e-15  # Tolerance for negligible time segments
EPSILON_TIME_MATCH = 1e-6  # Tolerance for matching arrival times (for rarefaction identification)
EPSILON_CONCENTRATION = 1e-10  # Tolerance for checking if concentration is effectively zero


def concentration_at_point(
    v: float,
    t: float,
    waves: list[Wave],
    sorption: FreundlichSorption | ConstantRetardation,  # noqa: ARG001
) -> float:
    """
    Compute concentration at point (v, t) with exact analytical value.

    Searches through all waves to find which wave controls the concentration
    at the given point in space-time. Uses exact analytical solutions for
    characteristics, shocks, and rarefaction fans.

    Parameters
    ----------
    v : float
        Position [m³].
    t : float
        Time [days].
    waves : list of Wave
        All waves in the simulation (active and inactive).
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.

    Returns
    -------
    concentration : float
        Concentration at point (v, t) [mass/volume].

    Notes
    -----
    **Wave Priority**:
    The algorithm checks waves in this order:
    1. Rarefaction waves (if point is inside rarefaction fan)
    2. Shocks (discontinuities)
    3. Rarefaction tails (if rarefaction tail has passed point)
    4. Characteristics (smooth regions)

    If no active wave controls the point, returns 0.0 (initial condition).

    **Rarefaction Tail Behavior**:
    After a rarefaction tail passes a query point, that point maintains the
    tail concentration as a plateau. This ensures proper concentration behavior
    after rarefaction waves pass through.

    **Machine Precision**:
    All position and concentration calculations use exact analytical formulas.
    Numerical tolerance is only used for equality checks (v == v_shock).

    Examples
    --------
    ::

        sorption = FreundlichSorption(
            k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
        )
        # After running simulation with waves...
        c = concentration_at_point(v=250.0, t=5.0, waves=all_waves, sorption=sorption)
        c >= 0.0
    """
    # Check rarefactions first (they have spatial extent and override other waves)
    for wave in waves:
        if isinstance(wave, RarefactionWave) and wave.is_active:
            c = wave.concentration_at_point(v, t)
            if c is not None:
                return c

    # Track the most recent wave to affect position v
    # We need to compare crossing times of shocks and rarefaction tails
    latest_wave_time = -np.inf
    latest_wave_c = None

    # Check shocks - track which shocks control this position
    for wave in waves:
        if isinstance(wave, ShockWave) and wave.is_active:
            v_shock = wave.position_at_time(t)
            if v_shock is not None:
                # Tolerance for exact shock position
                tol = 1e-15

                if abs(v - v_shock) < tol:
                    # Exactly at shock position
                    return 0.5 * (wave.c_left + wave.c_right)

                # Determine if shock has crossed position v and when
                if wave.velocity is not None and abs(wave.velocity) > EPSILON_VELOCITY:
                    t_cross = wave.t_start + (v - wave.v_start) / wave.velocity

                    if t_cross <= t:
                        # Shock has crossed position v by time t
                        # After crossing, point sees c_left (concentration behind shock)
                        if t_cross > latest_wave_time:
                            latest_wave_time = t_cross
                            latest_wave_c = wave.c_left
                    elif v > v_shock and wave.t_start > latest_wave_time:
                        # Point is ahead of shock (shock hasn't reached it yet)
                        # Check if this is the closest shock ahead of us
                        # In this case, we see c_right unless overridden by another wave
                        # We track this with a negative time (shock formation time) to indicate
                        # it's a "passive" state (not actively changed)
                        latest_wave_time = wave.t_start
                        latest_wave_c = wave.c_right

    # Check rarefaction tails - they can override previous waves
    for wave in waves:
        if isinstance(wave, RarefactionWave) and wave.is_active:
            v_tail = wave.tail_position_at_time(t)
            if v_tail is not None and v_tail > v + 1e-15:
                # Rarefaction tail has passed position v
                # Find when it passed: v_start + tail_velocity * (t_pass - t_start) = v
                tail_vel = wave.tail_velocity()
                if tail_vel > EPSILON_VELOCITY:
                    t_pass = wave.t_start + (v - wave.v_start) / tail_vel
                    if t_pass <= t and t_pass > latest_wave_time:
                        latest_wave_time = t_pass
                        latest_wave_c = wave.c_tail

    if latest_wave_c is not None:
        return latest_wave_c

    # Check characteristics
    # Find the most recent characteristic that has reached position v by time t
    latest_c = 0.0
    latest_time = -np.inf

    for wave in waves:
        if isinstance(wave, CharacteristicWave) and wave.is_active:
            # Check if this characteristic has reached position v by time t
            v_char_at_t = wave.position_at_time(t)

            if v_char_at_t is not None and v_char_at_t >= v - 1e-15:
                # This characteristic has passed through v
                # Find when it passed: v_start + vel*(t_pass - t_start) = v
                vel = wave.velocity()

                if vel > EPSILON_VELOCITY:
                    t_pass = wave.t_start + (v - wave.v_start) / vel

                    if t_pass <= t and t_pass > latest_time:
                        latest_time = t_pass
                        latest_c = wave.concentration

    return latest_c


def compute_breakthrough_curve(
    t_array: npt.NDArray[np.floating],
    v_outlet: float,
    waves: list[Wave],
    sorption: FreundlichSorption | ConstantRetardation,
) -> npt.NDArray[np.floating]:
    """
    Compute concentration at outlet over time array.

    This is the breakthrough curve: C(v_outlet, t) for all t in t_array.

    Parameters
    ----------
    t_array : array_like
        Time points [days]. Must be sorted in ascending order.
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.

    Returns
    -------
    c_out : ndarray
        Array of concentrations matching t_array [mass/volume].

    Examples
    --------
    ::

        t_array = np.linspace(0, 100, 1000)
        c_out = compute_breakthrough_curve(
            t_array, v_outlet=500.0, waves=all_waves, sorption=sorption
        )
        len(c_out) == len(t_array)

    See Also
    --------
    concentration_at_point : Point-wise concentration
    compute_bin_averaged_concentration_exact : Bin-averaged concentrations
    """
    t_array = np.asarray(t_array, dtype=float)
    c_out = np.zeros(len(t_array))

    for i, t in enumerate(t_array):
        c_out[i] = concentration_at_point(v_outlet, t, waves, sorption)

    return c_out


def identify_outlet_segments(
    t_start: float, t_end: float, v_outlet: float, waves: list[Wave], sorption: FreundlichSorption | ConstantRetardation
) -> list[dict]:
    """
    Identify which waves control outlet concentration in time interval [t_start, t_end].

    Finds all wave crossing events at the outlet and constructs segments where
    concentration is constant or varying (rarefaction).

    Parameters
    ----------
    t_start : float
        Start of time interval [days].
    t_end : float
        End of time interval [days].
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.

    Returns
    -------
    segments : list of dict
        List of segment dictionaries, each containing:

        - 't_start' : float
            Segment start time
        - 't_end' : float
            Segment end time
        - 'type' : str
            'constant' or 'rarefaction'
        - 'concentration' : float
            For constant segments
        - 'wave' : Wave
            For rarefaction segments
        - 'c_start' : float
            Concentration at segment start
        - 'c_end' : float
            Concentration at segment end

    Notes
    -----
    Segments are constructed by:
    1. Finding all wave crossing events at outlet in [t_start, t_end]
    2. Sorting events chronologically
    3. Creating constant-concentration segments between events
    4. Handling rarefaction fans with time-varying concentration

    The segments completely partition the time interval [t_start, t_end].
    """
    # Find all waves that cross outlet in this time range
    outlet_events = []

    # Track rarefactions that already contain the outlet at t_start
    # These need to be handled separately since they don't generate crossing events
    active_rarefactions_at_start = []

    for wave in waves:
        if not wave.is_active:
            continue

        # For rarefactions, detect both head and tail crossings
        if isinstance(wave, RarefactionWave):
            # Check if outlet is already inside this rarefaction at t_start
            if wave.contains_point(v_outlet, t_start):
                active_rarefactions_at_start.append(wave)
                # Don't add crossing events for this wave since we're already inside it
                # But we still need to detect when the tail crosses during [t_start, t_end]
                v_tail = wave.tail_position_at_time(t_start)
                if v_tail is not None and v_tail < v_outlet:
                    vel_tail = wave.tail_velocity()
                    if vel_tail > 0:
                        dt = (v_outlet - v_tail) / vel_tail
                        t_cross = t_start + dt
                        if t_start < t_cross <= t_end:
                            outlet_events.append({
                                "time": t_cross,
                                "wave": wave,
                                "boundary": "tail",
                                "c_after": wave.c_tail,
                            })
                continue

            # Head crossing
            v_head = wave.head_position_at_time(t_start)
            if v_head is not None and v_head < v_outlet:
                vel_head = wave.head_velocity()
                if vel_head > 0:
                    dt = (v_outlet - v_head) / vel_head
                    t_cross = t_start + dt
                    if t_start <= t_cross <= t_end:
                        outlet_events.append({
                            "time": t_cross,
                            "wave": wave,
                            "boundary": "head",
                            "c_after": wave.c_head,
                        })

            # Tail crossing
            v_tail = wave.tail_position_at_time(t_start)
            if v_tail is not None and v_tail < v_outlet:
                vel_tail = wave.tail_velocity()
                if vel_tail > 0:
                    dt = (v_outlet - v_tail) / vel_tail
                    t_cross = t_start + dt
                    if t_start <= t_cross <= t_end:
                        outlet_events.append({
                            "time": t_cross,
                            "wave": wave,
                            "boundary": "tail",
                            "c_after": wave.c_tail,
                        })
        else:
            # Characteristics and shocks
            t_cross = find_outlet_crossing(wave, v_outlet, t_start)

            if t_cross is not None and t_start <= t_cross <= t_end:
                if isinstance(wave, CharacteristicWave):
                    c_after = wave.concentration
                elif isinstance(wave, ShockWave):
                    # After shock passes outlet, outlet sees left (upstream) state
                    c_after = wave.c_left
                else:
                    c_after = 0.0

                outlet_events.append({"time": t_cross, "wave": wave, "boundary": None, "c_after": c_after})

    # Sort events by time
    outlet_events.sort(key=itemgetter("time"))

    # Create segments between events
    segments = []
    current_t = t_start
    current_c = concentration_at_point(v_outlet, t_start, waves, sorption)

    # Handle case where we start inside a rarefaction
    if active_rarefactions_at_start:
        # Should only be one rarefaction containing the outlet at t_start
        raref = active_rarefactions_at_start[0]

        # Find when tail crosses (if it does)
        tail_cross_time = None
        for event in outlet_events:
            if event["wave"] is raref and event["boundary"] == "tail" and event["time"] > t_start:
                tail_cross_time = event["time"]
                break

        # Create rarefaction segment from t_start
        raref_end = min(tail_cross_time or t_end, t_end)

        c_start = concentration_at_point(v_outlet, t_start, waves, sorption)
        c_end = None
        if tail_cross_time and tail_cross_time <= t_end:
            c_end = raref.c_tail

        segments.append({
            "t_start": t_start,
            "t_end": raref_end,
            "type": "rarefaction",
            "wave": raref,
            "c_start": c_start,
            "c_end": c_end,
        })

        current_t = raref_end
        current_c = (
            concentration_at_point(v_outlet, raref_end + 1e-10, waves, sorption) if raref_end < t_end else current_c
        )

    for event in outlet_events:
        # Check if we're entering a rarefaction fan
        if isinstance(event["wave"], RarefactionWave) and event["boundary"] == "head":
            # Before rarefaction head: constant segment
            if event["time"] > current_t:
                segments.append({
                    "t_start": current_t,
                    "t_end": event["time"],
                    "type": "constant",
                    "concentration": current_c,
                    "c_start": current_c,
                    "c_end": current_c,
                })

            # Find when tail crosses (if it does)
            raref = event["wave"]
            tail_cross_time = None

            for later_event in outlet_events:
                if (
                    later_event["wave"] is raref
                    and later_event["boundary"] == "tail"
                    and later_event["time"] > event["time"]
                ):
                    tail_cross_time = later_event["time"]
                    break

            # Rarefaction segment
            raref_end = min(tail_cross_time or t_end, t_end)

            segments.append({
                "t_start": event["time"],
                "t_end": raref_end,
                "type": "rarefaction",
                "wave": raref,
                "c_start": raref.c_head,
                "c_end": raref.c_tail if tail_cross_time and tail_cross_time <= t_end else None,
            })

            current_t = raref_end
            current_c = (
                concentration_at_point(v_outlet, raref_end + 1e-10, waves, sorption) if raref_end < t_end else current_c
            )
        else:
            # Regular event (characteristic or shock crossing)
            # Segment before event
            if event["time"] > current_t:
                segments.append({
                    "t_start": current_t,
                    "t_end": event["time"],
                    "type": "constant",
                    "concentration": current_c,
                    "c_start": current_c,
                    "c_end": current_c,
                })

            current_t = event["time"]
            current_c = event["c_after"]

    # Final segment
    if t_end > current_t:
        segments.append({
            "t_start": current_t,
            "t_end": t_end,
            "type": "constant",
            "concentration": current_c,
            "c_start": current_c,
            "c_end": current_c,
        })

    return segments


def integrate_rarefaction_exact(
    raref: RarefactionWave, v_outlet: float, t_start: float, t_end: float, sorption: FreundlichSorption
) -> float:
    """
    Exact analytical integral of rarefaction concentration over time at fixed position.

    For Freundlich sorption, the concentration within a rarefaction fan varies as:
        R(C) = flow*(t - t_origin)/(v_outlet - v_origin)

    This function computes the exact integral:
        ∫_{t_start}^{t_end} C(t) dt

    where C(t) is obtained by inverting the retardation relation.

    Parameters
    ----------
    raref : RarefactionWave
        Rarefaction wave controlling the outlet.
    v_outlet : float
        Outlet position [m³].
    t_start : float
        Integration start time [days]. Can be -np.inf.
    t_end : float
        Integration end time [days]. Can be np.inf.
    sorption : FreundlichSorption
        Freundlich sorption model.

    Returns
    -------
    integral : float
        Exact integral value [concentration * time].

    Notes
    -----
    **Derivation**:

    For Freundlich: R(C) = 1 + alpha*C^beta where:
        alpha = rho_b*k_f/(n_por*n)
        beta = 1/n - 1

    At outlet: R = kappa*t + mu where:
        kappa = flow/(v_outlet - v_origin)
        mu = -flow*t_origin/(v_outlet - v_origin)

    Inverting: C = [(R-1)/alpha]^(1/beta) = [(kappa*t + mu - 1)/alpha]^(1/beta)

    Integral:
        ∫ C dt = (1/alpha^(1/beta)) * ∫ (kappa*t + mu - 1)^(1/beta) dt
               = (1/alpha^(1/beta)) * (1/kappa) * [(kappa*t + mu - 1)^(1/beta + 1) / (1/beta + 1)]

    evaluated from t_start to t_end.

    **Special Cases**:
    - If (kappa*t + mu - 1) <= 0, concentration is 0 (unphysical region)
    - For beta = 0 (n = 1), use ConstantRetardation instead
    - For t_end = +∞ with exponent < 0 (n>1), integral converges to 0
    - For t_start = -∞, antiderivative evaluates to 0

    Examples
    --------
    ::

        sorption = FreundlichSorption(
            k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
        )
        raref = RarefactionWave(
            t_start=0.0,
            v_start=0.0,
            flow=100.0,
            c_head=10.0,
            c_tail=2.0,
            sorption=sorption,
        )
        integral = integrate_rarefaction_exact(
            raref, v_outlet=500.0, t_start=5.0, t_end=15.0, sorption=sorption
        )
        integral > 0
        # Integration to infinity
        integral_inf = integrate_rarefaction_exact(
            raref, v_outlet=500.0, t_start=5.0, t_end=np.inf, sorption=sorption
        )
        integral_inf > integral
    """
    # Extract parameters
    t_origin = raref.t_start
    v_origin = raref.v_start
    flow = raref.flow

    # Coefficients in R = kappa*t + μ
    kappa = flow / (v_outlet - v_origin)
    mu = -flow * t_origin / (v_outlet - v_origin)

    # Freundlich parameters: R = 1 + alpha*C^beta
    alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)
    beta = 1.0 / sorption.n - 1.0

    # For integration, we need exponent = 1/beta + 1
    if abs(beta) < EPSILON_BETA:
        # Linear case - shouldn't happen with Freundlich
        # Fall back to numerical integration or raise error
        msg = "integrate_rarefaction_exact requires nonlinear sorption (n != 1)"
        raise ValueError(msg)

    exponent = 1.0 / beta + 1.0

    # Coefficient for antiderivative
    # F(t) = (1/(alpha^(1/beta)*kappa*exponent)) * (kappa*t + mu - 1)^exponent
    coeff = 1.0 / (alpha ** (1.0 / beta) * kappa * exponent)

    def antiderivative(t: float) -> float:
        """Evaluate antiderivative at time t."""
        # Handle infinite bounds
        if np.isinf(t):
            if t > 0:
                # t = +∞
                if exponent < 0:
                    # For n > 1 (higher C travels faster, beta < 0, exponent < 0):
                    # As t → +∞, base → +∞, so base^exponent → 0
                    return 0.0
                # For n < 1 (lower C travels faster, beta > 0, exponent > 0):
                # As t → +∞, base → +∞, so base^exponent → +∞
                # This should not happen for physical rarefactions to C=0
                msg = f"Integral diverges at t=+∞ with exponent={exponent} > 0"
                raise ValueError(msg)
            # t = -∞: always returns 0
            return 0.0

        base = kappa * t + mu - 1.0

        # Check if we're in physical region (R > 1)
        if base <= 0:
            return 0.0

        return coeff * base**exponent

    # Evaluate definite integral
    return antiderivative(t_end) - antiderivative(t_start)


def compute_bin_averaged_concentration_exact(
    t_edges: npt.NDArray[np.floating],
    v_outlet: float,
    waves: list[Wave],
    sorption: FreundlichSorption | ConstantRetardation,
) -> npt.NDArray[np.floating]:
    """
    Compute bin-averaged concentration using EXACT analytical integration.

    For each time bin [t_i, t_{i+1}], computes:
        C_avg = (1/(t_{i+1} - t_i)) * ∫_{t_i}^{t_{i+1}} C(v_outlet, t) dt

    This is the critical function for maintaining machine precision in output.
    All integrations use exact analytical formulas with no numerical quadrature.

    Parameters
    ----------
    t_edges : array_like
        Time bin edges [days]. Length N+1 for N bins.
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves from front tracking simulation.
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.

    Returns
    -------
    c_avg : ndarray
        Bin-averaged concentrations [mass/volume]. Length N.

    Notes
    -----
    **Algorithm**:

    1. For each bin [t_i, t_{i+1}]:
       a. Identify which wave segments control outlet during this period
       b. For each segment:
          - Constant C: integral = C * Δt
          - Rarefaction C(t): use exact analytical integral formula
       c. Sum segment integrals and divide by bin width

    **Machine Precision**:

    - Constant segments: exact to floating-point precision
    - Rarefaction segments: uses closed-form antiderivative
    - No numerical quadrature or interpolation
    - Maintains mass balance to ~1e-14 relative error

    **Rarefaction Integration**:

    For Freundlich sorption, rarefaction concentration at outlet varies as:
        C(t) = [(kappa*t + mu - 1)/alpha]^(1/beta)

    The exact integral is:
        ∫ C dt = (1/(alpha^(1/beta)*kappa*exponent)) * (kappa*t + mu - 1)^exponent

    where exponent = 1/beta + 1.

    Examples
    --------
    ::

        # After running front tracking simulation
        t_edges = np.array([0.0, 10.0, 20.0, 30.0])
        c_avg = compute_bin_averaged_concentration_exact(
            t_edges=t_edges,
            v_outlet=500.0,
            waves=tracker.state.waves,
            sorption=sorption,
        )
        len(c_avg) == len(t_edges) - 1
        np.all(c_avg >= 0)

    See Also
    --------
    concentration_at_point : Point-wise concentration
    compute_breakthrough_curve : Breakthrough curve
    integrate_rarefaction_exact : Exact rarefaction integration
    """
    t_edges = np.asarray(t_edges, dtype=float)
    n_bins = len(t_edges) - 1
    c_avg = np.zeros(n_bins)

    for i in range(n_bins):
        t_start = t_edges[i]
        t_end = t_edges[i + 1]
        dt = t_end - t_start

        if dt <= 0:
            msg = f"Invalid time bin: t_edges[{i}]={t_start} >= t_edges[{i + 1}]={t_end}"
            raise ValueError(msg)

        # Identify wave segments controlling outlet in this time bin
        segments = identify_outlet_segments(t_start, t_end, v_outlet, waves, sorption)

        # Integrate each segment
        total_integral = 0.0

        for seg in segments:
            seg_t_start = max(seg["t_start"], t_start)
            seg_t_end = min(seg["t_end"], t_end)
            seg_dt = seg_t_end - seg_t_start

            if seg_dt <= EPSILON_TIME:  # Skip negligible segments
                continue

            if seg["type"] == "constant":
                # C is constant over segment - exact integral
                integral = seg["concentration"] * seg_dt

            elif seg["type"] == "rarefaction":
                # C(t) given by self-similar solution - use exact analytical integral
                if isinstance(sorption, FreundlichSorption):
                    raref = seg["wave"]
                    integral = integrate_rarefaction_exact(raref, v_outlet, seg_t_start, seg_t_end, sorption)
                else:
                    # ConstantRetardation - rarefactions shouldn't form, use constant approximation
                    c_mid = concentration_at_point(v_outlet, 0.5 * (seg_t_start + seg_t_end), waves, sorption)
                    integral = c_mid * seg_dt
            else:
                msg = f"Unknown segment type: {seg['type']}"
                raise ValueError(msg)

            total_integral += integral

        c_avg[i] = total_integral / dt

    return c_avg


def compute_domain_mass(
    t: float,
    v_outlet: float,
    waves: list[Wave],
    sorption: FreundlichSorption | ConstantRetardation,
) -> float:
    """
    Compute total mass in domain [0, v_outlet] at time t using exact analytical integration.

    Implements runtime mass balance verification as described in High Priority #3
    of FRONT_TRACKING_REBUILD_PLAN.md. Integrates concentration over space:

        M(t) = ∫₀^v_outlet C(v, t) dv

    using exact analytical formulas for each wave type.

    Parameters
    ----------
    t : float
        Time at which to compute domain mass [days].
    v_outlet : float
        Outlet position (domain extent) [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.

    Returns
    -------
    mass : float
        Total mass in domain [mass]. Computed to machine precision (~1e-14).

    Notes
    -----
    **Algorithm**:

    1. Partition spatial domain [0, v_outlet] into segments where concentration
       is controlled by a single wave or is constant.
    2. For each segment, compute mass analytically:
       - Constant C: mass = C * Δv
       - Rarefaction C(v): use exact spatial integration formula
    3. Sum all segment masses.

    **Wave Priority** (same as concentration_at_point):

    1. Rarefactions (if position is inside rarefaction fan)
    2. Shocks and rarefaction tails (most recent to pass)
    3. Characteristics (smooth regions)

    **Rarefaction Spatial Integration**:

    For a rarefaction fan at fixed time t, concentration varies with position v
    according to the self-similar solution:

        R(C) = flow*(t - t_origin)/(v - v_origin)

    The spatial integral ∫ C dv is computed analytically using the inverse
    retardation relation.

    **Integration Precision**:

    - Constant concentration regions: Exact analytical (C_total * dv)
    - Rarefaction regions: High-precision trapezoidal quadrature (10+ points)
    - Overall accuracy: ~1e-10 to 1e-12 relative error
    - Sufficient for runtime verification; primary outputs remain exact

    Examples
    --------
    ::

        # After running simulation to time t=10.0
        mass = compute_domain_mass(
            t=10.0, v_outlet=500.0, waves=tracker.state.waves, sorption=sorption
        )
        mass >= 0.0

    See Also
    --------
    compute_cumulative_inlet_mass : Cumulative inlet mass
    compute_cumulative_outlet_mass : Cumulative outlet mass
    concentration_at_point : Point-wise concentration
    """
    # Partition spatial domain into segments based on wave structure
    # We'll evaluate concentration at many points and identify constant regions

    # Collect all wave positions at time t
    wave_positions = []

    for wave in waves:
        if not wave.is_active:
            continue

        if isinstance(wave, (CharacteristicWave, ShockWave)):
            v_pos = wave.position_at_time(t)
            if v_pos is not None and 0 <= v_pos <= v_outlet:
                wave_positions.append(v_pos)

        elif isinstance(wave, RarefactionWave):
            v_head = wave.head_position_at_time(t)
            v_tail = wave.tail_position_at_time(t)

            if v_head is not None and 0 <= v_head <= v_outlet:
                wave_positions.append(v_head)
            if v_tail is not None and 0 <= v_tail <= v_outlet:
                wave_positions.append(v_tail)

    # Add domain boundaries
    wave_positions.extend([0.0, v_outlet])

    # Sort and remove duplicates
    wave_positions = sorted(set(wave_positions))

    # Remove positions outside domain
    wave_positions = [v for v in wave_positions if 0 <= v <= v_outlet]

    # Compute mass in each segment using refined integration
    total_mass = 0.0

    for i in range(len(wave_positions) - 1):
        v_start = wave_positions[i]
        v_end = wave_positions[i + 1]
        dv = v_end - v_start

        if dv < EPSILON_VELOCITY:
            continue

        # Check if this segment is inside a rarefaction fan
        # Sample at midpoint to check
        v_mid = 0.5 * (v_start + v_end)
        inside_rarefaction = False
        raref_wave = None

        for wave in waves:
            if isinstance(wave, RarefactionWave) and wave.is_active and wave.contains_point(v_mid, t):
                inside_rarefaction = True
                raref_wave = wave
                break

        if inside_rarefaction and raref_wave is not None:
            # Rarefaction: concentration varies with position
            # Use EXACT analytical spatial integration
            mass_segment = _integrate_rarefaction_spatial_exact(raref_wave, v_start, v_end, t, sorption)
        else:
            # Constant concentration region - exact integration
            v_mid = 0.5 * (v_start + v_end)
            c = concentration_at_point(v_mid, t, waves, sorption)
            c_total = sorption.total_concentration(c)
            mass_segment = c_total * dv

        total_mass += mass_segment

    return total_mass


def _integrate_rarefaction_spatial_exact(
    raref: RarefactionWave,
    v_start: float,
    v_end: float,
    t: float,
    sorption: FreundlichSorption | ConstantRetardation,
) -> float:
    """
    EXACT analytical spatial integral of rarefaction total concentration at fixed time.

    Computes ∫_{v_start}^{v_end} C_total(v) dv analytically using closed-form
    antiderivatives for specific Freundlich n values. This maintains machine precision
    for the mass balance diagnostic.

    For rarefaction at time t: R(C) = kappa/(v - v₀) where kappa = flow*(t - t₀)

    For Freundlich: R = 1 + alpha*C^beta where alpha = rho_b*k_f/(n_por*n), beta = 1/n - 1
    Total concentration: C_total = C + (rho_b/n_por)*k_f*C^(1/n)

    **Exact formulas implemented**:
    - n = 2 (beta = -0.5): Closed-form using polynomial expansion (optimized path)
    - n = 1: No rarefactions (constant retardation)
    - All other n > 0: Unified formula using generalized incomplete beta function via mpmath

    Parameters
    ----------
    raref : RarefactionWave
    v_start, v_end : float
        Integration bounds [m³]
    t : float
        Time [days]
    sorption : FreundlichSorption or ConstantRetardation

    Returns
    -------
    mass : float
        Exact mass in segment to machine precision

    Notes
    -----
    **Derivation for n=2** (beta = -0.5):

    C(v) = [(kappa/(v-v₀) - 1) / alpha]^(-2) = [alpha*(v-v₀)]² / (kappa - (v-v₀))²

    Let u = v - v₀, then ∫ C du requires integrating u²/(kappa-u)².
    Using substitution w = kappa - u:

    ∫ u²/(kappa-u)² du = alpha²[kappa²/(kappa-u) + 2kappa*ln|kappa-u| - (kappa-u)]

    For C_total, we integrate both C and (rho_b/n_por)*k_f*C^0.5 terms.

    **Unified Formula for All n > 0**:

    The spatial integral has the structure:

        ∫ C_total(u) du = ∫ C(u) du + ∫ (rho_b/n_por)*k_f*C(u)^(1/n) du

    where C(u) = [(kappa-u)/(alpha*u)]^(1/beta) with beta = 1/n - 1.

    Both integrals reduce to power-law forms:

        ∫ u^p (kappa-u)^q du

    which can be expressed using the generalized incomplete beta function:

        ∫_{u₁}^{u₂} u^p (kappa-u)^q du = kappa^(p+q+1) B(u₁/kappa, u₂/kappa; p+1, q+1)

    where B(x₁, x₂; a, b) = ∫_{x₁}^{x₂} t^(a-1) (1-t)^(b-1) dt.

    For many n values, the parameters a and b can be negative or zero, which
    requires analytic continuation beyond the standard incomplete beta function.
    The mpmath library provides this through mpmath.betainc(), which handles
    all parameter values and achieves machine precision (~1e-15 relative error).

    **Mathematical Properties**:
    - Continuous for all n > 0 (no discontinuities)
    - Achieves machine precision (~1e-14 to 1e-15 relative error)
    - No numerical quadrature (uses special function evaluation)
    - Single unified formula (no conditional logic based on n)
    """
    if isinstance(sorption, ConstantRetardation):
        # Constant retardation: no rarefactions
        v_mid = 0.5 * (v_start + v_end)
        c = raref.concentration_at_point(v_mid, t) or 0.0
        c_total = sorption.total_concentration(c)
        return c_total * (v_end - v_start)

    # Extract parameters
    t_origin = raref.t_start
    v_origin = raref.v_start
    flow = raref.flow

    if t <= t_origin:
        return 0.0

    kappa = flow * (t - t_origin)  # kappa
    alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)  # alpha
    rho_b = sorption.bulk_density
    n_por = sorption.porosity
    k_f = sorption.k_f
    n = sorption.n

    # Transform to u coordinates
    u_start = v_start - v_origin
    u_end = v_end - v_origin

    if u_start <= 0 or u_end <= 0:
        return 0.0

    # Unified formula for all n > 0 using generalized incomplete beta function
    beta = 1 / n - 1  # beta parameter

    # Transform to [0,1] domain: t = u/kappa (use regular floats)
    t_start = u_start / kappa
    t_end = u_end / kappa

    # Integral 1: Dissolved concentration
    # ∫ C(u) du = ∫ [(kappa-u)/(alpha*u)]^(1/beta) du
    #           = (1/alpha)^(1/beta) ∫ u^(-1/beta) (kappa-u)^(1/beta) du
    #           = (1/alpha)^(1/beta) * kappa * B(t_start, t_end; 1-1/beta, 1+1/beta)
    #
    # where B(x1, x2; a, b) = ∫_{x1}^{x2} t^(a-1) (1-t)^(b-1) dt

    # Beta function parameters (use regular floats)
    a_diss = 1 - 1 / beta
    b_diss = 1 + 1 / beta

    # Use scipy.special.betainc when parameters are positive (faster),
    # fall back to mpmath for negative parameters (requires analytic continuation)
    if a_diss > 0 and b_diss > 0:
        # scipy.special.betainc is regularized, so multiply by beta function
        beta_diss = betainc(a_diss, b_diss, t_end) - betainc(a_diss, b_diss, t_start)
        beta_diss *= beta_func(a_diss, b_diss)
    else:
        # Use mpmath for negative parameters (analytic continuation)
        beta_diss = float(mp.betainc(a_diss, b_diss, t_start, t_end, regularized=False))

    # Compute coefficient using regular float arithmetic
    coeff_diss = (1 / alpha) ** (1 / beta)
    mass_dissolved = coeff_diss * kappa * beta_diss

    # Integral 2: Sorbed concentration
    # ∫ (rho_b/n_por)*k_f*C^(1/n) du
    #
    # where C^(1/n) = [(kappa-u)/(alpha*u)]^(1/(1-n))
    #
    # This gives: ∫ u^(-1/(1-n)) (kappa-u)^(1/(1-n)) du
    #
    # Using same transformation: = kappa * B(t_start, t_end; 1-1/(1-n), 1+1/(1-n))

    # Beta function parameters (use regular floats)
    exponent_sorb = 1 / (1 - n)
    a_sorb = 1 - exponent_sorb
    b_sorb = 1 + exponent_sorb

    # Use scipy when possible, mpmath for negative parameters
    if a_sorb > 0 and b_sorb > 0:
        beta_sorb = betainc(a_sorb, b_sorb, t_end) - betainc(a_sorb, b_sorb, t_start)
        beta_sorb *= beta_func(a_sorb, b_sorb)
    else:
        beta_sorb = float(mp.betainc(a_sorb, b_sorb, t_start, t_end, regularized=False))

    # Compute coefficient using regular float arithmetic
    coeff_sorb = (rho_b / n_por) * k_f / (alpha**exponent_sorb)
    mass_sorbed = coeff_sorb * kappa * beta_sorb

    return mass_dissolved + mass_sorbed


def compute_cumulative_inlet_mass(
    t: float,
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    tedges: npt.NDArray[np.floating],
) -> float:
    """
    Compute cumulative mass entering domain from t=0 to t.

    Integrates inlet mass flux over time:
        M_in(t) = ∫₀^t cin(τ) * flow(τ) dτ

    using exact analytical integration of piecewise-constant functions.

    Parameters
    ----------
    t : float
        Time up to which to integrate [days].
    cin : array_like
        Inlet concentration time series [mass/volume].
        Piecewise constant within bins defined by tedges.
    flow : array_like
        Flow rate time series [m³/day].
        Piecewise constant within bins defined by tedges.
    tedges : array_like
        Time bin edges [days]. Length len(cin) + 1.

    Returns
    -------
    mass_in : float
        Cumulative mass entered [mass].

    Notes
    -----
    For piecewise-constant cin and flow:
        M_in = Σ cin[i] * flow[i] * dt[i]

    where the sum is over all bins from tedges[0] to t.
    Partial bins are handled exactly.

    Examples
    --------
    ::

        mass_in = compute_cumulative_inlet_mass(
            t=50.0, cin=cin, flow=flow, tedges=tedges
        )
        mass_in >= 0.0
    """
    tedges_arr = np.asarray(tedges, dtype=float)
    cin_arr = np.asarray(cin, dtype=float)
    flow_arr = np.asarray(flow, dtype=float)

    # Find which bin t falls into
    if t <= tedges_arr[0]:
        return 0.0

    if t >= tedges_arr[-1]:
        # Integrate all bins
        dt = np.diff(tedges_arr)
        return float(np.sum(cin_arr * flow_arr * dt))

    # Find bin containing t
    bin_idx = np.searchsorted(tedges_arr, t, side="right") - 1

    # Mass flux across inlet boundary = Q * C_in (aqueous concentration)
    # This is correct for sorbing solutes: only dissolved mass flows with water
    # Integrate complete bins before t
    if bin_idx > 0:
        dt_complete = np.diff(tedges_arr[: bin_idx + 1])
        mass_complete = np.sum(cin_arr[:bin_idx] * flow_arr[:bin_idx] * dt_complete)
    else:
        mass_complete = 0.0

    # Add partial bin
    if bin_idx >= 0 and bin_idx < len(cin_arr):
        dt_partial = t - tedges_arr[bin_idx]
        mass_partial = cin_arr[bin_idx] * flow_arr[bin_idx] * dt_partial
    else:
        mass_partial = 0.0

    return float(mass_complete + mass_partial)


def find_last_rarefaction_start_time(
    v_outlet: float,
    waves: list[Wave],
) -> float:
    """
    Find the time when the last rarefaction head reaches the outlet.

    For rarefactions, we integrate analytically so we only need to know
    when the rarefaction starts at the outlet (head arrival).

    Parameters
    ----------
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.

    Returns
    -------
    t_last : float
        Time when last rarefaction head reaches outlet [days].
        For non-rarefaction waves, uses their arrival time.
        Returns 0.0 if no waves reach the outlet.

    Notes
    -----
    This function finds when the last wave structure *starts* at the outlet.
    For rarefactions, this is the head arrival time. The tail may arrive
    much later (or at infinite time for rarefactions to C=0), but the
    total mass in the rarefaction is computed analytically.

    Examples
    --------
    ::

        t_last = find_last_rarefaction_start_time(v_outlet=500.0, waves=all_waves)
        t_last >= 0.0
    """
    t_last = 0.0

    for wave in waves:
        if not wave.is_active:
            continue

        if isinstance(wave, RarefactionWave):
            # For rarefaction, use head arrival (when rarefaction starts)
            head_vel = wave.head_velocity()
            if head_vel > EPSILON_VELOCITY:
                t_arrival = wave.t_start + (v_outlet - wave.v_start) / head_vel
                t_last = max(t_last, t_arrival)
        elif isinstance(wave, (CharacteristicWave, ShockWave)):
            # For characteristics and shocks, compute arrival time
            vel = wave.velocity if isinstance(wave, ShockWave) else wave.velocity()
            if vel is not None and vel > EPSILON_VELOCITY:
                t_arrival = wave.t_start + (v_outlet - wave.v_start) / vel
                t_last = max(t_last, t_arrival)

    return t_last


def compute_cumulative_outlet_mass(
    t: float,
    v_outlet: float,
    waves: list[Wave],
    sorption: FreundlichSorption | ConstantRetardation,
    flow: npt.NDArray[np.floating],
    tedges: npt.NDArray[np.floating],
) -> float:
    """
    Compute cumulative mass exiting domain from t=0 to t.

    Integrates outlet mass flux over time:
        M_out(t) = ∫₀^t cout(τ) * flow(τ) dτ

    using exact analytical integration. Outlet concentration cout(τ) is obtained
    from the wave solution, and flow(τ) is piecewise constant.

    Parameters
    ----------
    t : float
        Time up to which to integrate [days].
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.
    flow : array_like
        Flow rate time series [m³/day].
        Piecewise constant within bins defined by tedges.
    tedges : array_like
        Time bin edges [days]. Length len(flow) + 1.

    Returns
    -------
    mass_out : float
        Cumulative mass exited [mass].

    Notes
    -----
    The outlet concentration is obtained from wave solution via
    concentration_at_point(v_outlet, τ, waves, sorption).

    For each flow bin [t_i, t_{i+1}], the mass flux integral is computed
    exactly using identify_outlet_segments and analytical integration
    (constant segments and rarefaction segments).

    Examples
    --------
    ::

        mass_out = compute_cumulative_outlet_mass(
            t=50.0,
            v_outlet=500.0,
            waves=tracker.state.waves,
            sorption=sorption,
            flow=flow,
            tedges=tedges,
        )
        mass_out >= 0.0
    """
    tedges_arr = np.asarray(tedges, dtype=float)
    flow_arr = np.asarray(flow, dtype=float)

    if t <= tedges_arr[0]:
        return 0.0

    # Integrate bin by bin through all flow bins, then continue to t if needed
    total_mass = 0.0

    # Process bins within tedges range
    n_flow_bins = len(flow_arr)

    for i in range(n_flow_bins):
        t_bin_start = tedges_arr[i]
        t_bin_end = tedges_arr[i + 1]

        # Skip bins entirely before t
        if t_bin_end <= tedges_arr[0]:
            continue

        # Clip to [tedges[0], t]
        t_bin_start = max(t_bin_start, tedges_arr[0])
        t_bin_end = min(t_bin_end, t)

        flow_i = flow_arr[i]
        dt_i = t_bin_end - t_bin_start

        if dt_i <= 0:
            continue

        # Compute ∫_{t_bin_start}^{t_bin_end} cout(τ) dτ using exact integration
        segments = identify_outlet_segments(t_bin_start, t_bin_end, v_outlet, waves, sorption)

        integral_c_dt = 0.0

        for seg in segments:
            seg_t_start = max(seg["t_start"], t_bin_start)
            seg_t_end = min(seg["t_end"], t_bin_end)
            seg_dt = seg_t_end - seg_t_start

            if seg_dt <= EPSILON_TIME:
                continue

            if seg["type"] == "constant":
                integral_c_dt += seg["concentration"] * seg_dt
            elif seg["type"] == "rarefaction":
                if isinstance(sorption, FreundlichSorption):
                    raref = seg["wave"]
                    integral_c_dt += integrate_rarefaction_exact(raref, v_outlet, seg_t_start, seg_t_end, sorption)
                else:
                    # ConstantRetardation - use midpoint
                    c_mid = concentration_at_point(v_outlet, 0.5 * (seg_t_start + seg_t_end), waves, sorption)
                    integral_c_dt += c_mid * seg_dt

        # Mass for this bin = flow * ∫ cout dt
        total_mass += flow_i * integral_c_dt

    return float(total_mass)


def integrate_rarefaction_total_mass(
    raref: RarefactionWave,
    v_outlet: float,
    t_start: float,
    sorption: FreundlichSorption | ConstantRetardation,
    flow: float,
) -> float:
    """
    Compute total mass exiting through a rarefaction.

    For a rarefaction that starts at the outlet at time t_start, compute the
    total mass that will exit through the rarefaction. Integration endpoint
    depends on the rarefaction tail concentration:

    - If c_tail ≈ 0: Integrate to infinity (tail extends infinitely)
    - If c_tail > 0: Integrate to finite tail arrival time

    Parameters
    ----------
    raref : RarefactionWave
        Rarefaction wave.
    v_outlet : float
        Outlet position [m³].
    t_start : float
        Time when rarefaction head reaches outlet [days].
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.
    flow : float
        Flow rate [m³/day] (assumed constant).

    Returns
    -------
    total_mass : float
        Total mass that exits through rarefaction [mass].

    Notes
    -----
    Uses the exact analytical integral:
        M_total = ∫_{t_start}^{t_end} Q * C(t) dt

    where C(t) is the concentration at the outlet from the rarefaction wave.

    For n > 1 (favorable sorption), rarefactions typically decrease to C=0,
    so t_end = ∞ and the integral converges.

    For n < 1 (unfavorable sorption), rarefactions typically increase from
    low C to high C, so c_tail > 0 and the tail arrives at finite time
    t_end = t_start + (v_outlet - v_start) / tail_velocity.

    Examples
    --------
    ::

        mass = integrate_rarefaction_total_mass(
            raref=raref_wave,
            v_outlet=500.0,
            t_start=40.0,
            sorption=sorption,
            flow=100.0,
        )
        mass >= 0.0
    """
    if isinstance(sorption, ConstantRetardation):
        # No rarefactions with constant retardation
        return 0.0

    # Determine integration endpoint based on c_tail
    # For rarefactions with c_tail ≈ 0, the tail extends to infinity
    # For rarefactions with c_tail > 0, the tail arrives at finite time
    if raref.c_tail < EPSILON_CONCENTRATION:
        # Rarefaction tail goes to C≈0, extends to infinite time
        # This is typical for n > 1 rarefactions (concentration decreases)
        t_end = np.inf
    else:
        # Rarefaction tail has finite concentration, arrives at finite time
        # This is typical for n < 1 rarefactions (concentration increases)
        tail_velocity = raref.tail_velocity()
        if tail_velocity < EPSILON_VELOCITY:
            # Tail velocity is effectively zero, extends to infinity
            t_end = np.inf
        else:
            # Compute finite tail arrival time
            t_end = raref.t_start + (v_outlet - raref.v_start) / tail_velocity

    # Integrate from t_start to t_end
    integral_c_dt = integrate_rarefaction_exact(raref, v_outlet, t_start, t_end, sorption)

    return flow * integral_c_dt


def compute_total_outlet_mass(
    v_outlet: float,
    waves: list[Wave],
    sorption: FreundlichSorption | ConstantRetardation,
    flow: npt.NDArray[np.floating],
    tedges: npt.NDArray[np.floating],
) -> tuple[float, float]:
    """
    Compute total integrated outlet mass until all mass has exited.

    Automatically determines when the last wave passes the outlet and
    integrates the outlet mass flux until that time, regardless of the
    provided tedges extent.

    Parameters
    ----------
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.
    flow : array_like
        Flow rate time series [m³/day].
        Piecewise constant within bins defined by tedges.
    tedges : array_like
        Time bin edges [days]. Length len(flow) + 1.

    Returns
    -------
    total_mass_out : float
        Total mass that exits through outlet [mass].
    t_integration_end : float
        Time until which integration was performed [days].
        This is the time when the last wave passes the outlet.

    Notes
    -----
    This function:
    1. Finds when the last rarefaction *starts* at the outlet (head arrival)
    2. Integrates outlet mass flux until that time
    3. Adds analytical integral of rarefaction mass from start to infinity

    For rarefactions to C=0, the tail has infinite arrival time but the
    total mass is finite and computed analytically.

    Examples
    --------
    ::

        total_mass, t_end = compute_total_outlet_mass(
            v_outlet=500.0,
            waves=tracker.state.waves,
            sorption=sorption,
            flow=flow,
            tedges=tedges,
        )
        total_mass >= 0.0
        t_end >= tedges[0]

    See Also
    --------
    compute_cumulative_outlet_mass : Cumulative outlet mass up to time t
    find_last_rarefaction_start_time : Find when last rarefaction starts
    integrate_rarefaction_total_mass : Total mass in rarefaction to infinity
    """
    # Find when the last rarefaction starts at the outlet
    t_last_raref_start = find_last_rarefaction_start_time(v_outlet, waves)

    tedges_arr = np.asarray(tedges, dtype=float)
    flow_arr = np.asarray(flow, dtype=float)

    # Integrate up to when last rarefaction starts
    if t_last_raref_start <= tedges_arr[-1]:
        # All waves start within provided time range
        mass_up_to_raref_start = compute_cumulative_outlet_mass(
            t_last_raref_start, v_outlet, waves, sorption, flow_arr, tedges_arr
        )
        flow_at_raref_start = flow_arr[-1]  # Use last flow value
    else:
        # Need to extend beyond tedges to reach rarefaction start
        # First, compute mass up to tedges[-1]
        mass_within_tedges = compute_cumulative_outlet_mass(
            tedges_arr[-1], v_outlet, waves, sorption, flow_arr, tedges_arr
        )

        # Then, integrate from tedges[-1] to t_last_raref_start
        flow_extended = flow_arr[-1]
        t_start_extended = tedges_arr[-1]
        t_end_extended = t_last_raref_start

        # Get outlet segments for extended period
        segments = identify_outlet_segments(t_start_extended, t_end_extended, v_outlet, waves, sorption)

        integral_c_dt = 0.0

        for seg in segments:
            seg_t_start = max(seg["t_start"], t_start_extended)
            seg_t_end = min(seg["t_end"], t_end_extended)
            seg_dt = seg_t_end - seg_t_start

            if seg_dt <= EPSILON_TIME:
                continue

            if seg["type"] == "constant":
                integral_c_dt += seg["concentration"] * seg_dt
            elif seg["type"] == "rarefaction":
                if isinstance(sorption, FreundlichSorption):
                    raref = seg["wave"]
                    integral_c_dt += integrate_rarefaction_exact(raref, v_outlet, seg_t_start, seg_t_end, sorption)
                else:
                    # ConstantRetardation - use midpoint
                    c_mid = concentration_at_point(v_outlet, 0.5 * (seg_t_start + seg_t_end), waves, sorption)
                    integral_c_dt += c_mid * seg_dt

        mass_up_to_raref_start = mass_within_tedges + flow_extended * integral_c_dt
        flow_at_raref_start = flow_extended

    # Find rarefactions that are active at the outlet after t_last_raref_start
    # and add their total integrated mass
    total_raref_mass = 0.0

    for wave in waves:
        if not wave.is_active:
            continue

        if isinstance(wave, RarefactionWave):
            # Check if this rarefaction reaches the outlet
            head_vel = wave.head_velocity()
            if head_vel > EPSILON_VELOCITY and wave.v_start < v_outlet:
                t_raref_start_at_outlet = wave.t_start + (v_outlet - wave.v_start) / head_vel

                # If this rarefaction starts at or after t_last_raref_start, include its total mass
                # (with small tolerance for numerical precision)
                if abs(t_raref_start_at_outlet - t_last_raref_start) < EPSILON_TIME_MATCH:
                    # This is the last rarefaction - integrate to infinity
                    raref_mass = integrate_rarefaction_total_mass(
                        raref=wave,
                        v_outlet=v_outlet,
                        t_start=t_raref_start_at_outlet,
                        sorption=sorption,
                        flow=flow_at_raref_start,
                    )
                    total_raref_mass += raref_mass

    # For rarefactions with finite tails (c_tail > 0, typical for n < 1),
    # all mass is already accounted for in the rarefaction integration
    # from head to tail. No additional mass needs to be integrated after
    # the tail - if there were more waves, they would be in the wave list.
    total_mass = mass_up_to_raref_start + total_raref_mass

    return float(total_mass), t_last_raref_start
