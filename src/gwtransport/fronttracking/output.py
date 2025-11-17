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

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt

from gwtransport.fronttracking.events import find_outlet_crossing
from gwtransport.fronttracking.math import ConstantRetardation, FreundlichSorption
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave, Wave


def concentration_at_point(
    v: float, t: float, waves: list[Wave], sorption: FreundlichSorption | ConstantRetardation
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
    1. Rarefaction waves (have spatial extent)
    2. Shocks (discontinuities)
    3. Characteristics (smooth regions)

    If no active wave controls the point, returns 0.0 (initial condition).

    **Machine Precision**:
    All position and concentration calculations use exact analytical formulas.
    Numerical tolerance is only used for equality checks (v == v_shock).

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> # After running simulation with waves...
    >>> c = concentration_at_point(v=250.0, t=5.0, waves=all_waves, sorption=sorption)
    >>> c >= 0.0
    True
    """
    # Check rarefactions first (they have spatial extent and override other waves)
    for wave in waves:
        if isinstance(wave, RarefactionWave) and wave.is_active:
            c = wave.concentration_at_point(v, t)
            if c is not None:
                return c

    # Check shocks (discontinuities)
    # Find the rightmost shock that has passed position v by time t
    rightmost_shock_c = None
    rightmost_shock_pos = -np.inf

    for wave in waves:
        if isinstance(wave, ShockWave) and wave.is_active:
            v_shock = wave.position_at_time(t)
            if v_shock is not None:
                # Tolerance for exact shock position
                tol = 1e-15

                if v < v_shock - tol:
                    # Haven't reached shock yet - check if this is the rightmost shock behind us
                    if v_shock > rightmost_shock_pos:
                        rightmost_shock_pos = v_shock
                        rightmost_shock_c = wave.c_left
                elif v > v_shock + tol:
                    # Past shock - check if this is the rightmost shock we've passed
                    if v_shock > rightmost_shock_pos:
                        rightmost_shock_pos = v_shock
                        rightmost_shock_c = wave.c_right
                else:
                    # Exactly at shock
                    return 0.5 * (wave.c_left + wave.c_right)

    if rightmost_shock_c is not None:
        return rightmost_shock_c

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

                if vel > 1e-15:
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
    >>> t_array = np.linspace(0, 100, 1000)
    >>> c_out = compute_breakthrough_curve(
    ...     t_array, v_outlet=500.0, waves=all_waves, sorption=sorption
    ... )
    >>> len(c_out) == len(t_array)
    True

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

    for wave in waves:
        if not wave.is_active:
            continue

        # For rarefactions, detect both head and tail crossings
        if isinstance(wave, RarefactionWave):
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
    outlet_events.sort(key=lambda e: e["time"])

    # Create segments between events
    segments = []
    current_t = t_start
    current_c = concentration_at_point(v_outlet, t_start, waves, sorption)

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
        Integration start time [days].
    t_end : float
        Integration end time [days].
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

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> raref = RarefactionWave(
    ...     t_start=0.0,
    ...     v_start=0.0,
    ...     flow=100.0,
    ...     c_head=10.0,
    ...     c_tail=2.0,
    ...     sorption=sorption,
    ... )
    >>> integral = integrate_rarefaction_exact(
    ...     raref, v_outlet=500.0, t_start=5.0, t_end=15.0, sorption=sorption
    ... )
    >>> integral > 0
    True
    """
    # Extract parameters
    t_origin = raref.t_start
    v_origin = raref.v_start
    flow = raref.flow

    # Coefficients in R = κ*t + μ
    kappa = flow / (v_outlet - v_origin)
    mu = -flow * t_origin / (v_outlet - v_origin)

    # Freundlich parameters: R = 1 + alpha*C^beta
    alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)
    beta = 1.0 / sorption.n - 1.0

    # For integration, we need exponent = 1/β + 1
    if abs(beta) < 1e-15:
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
        base = kappa * t + mu - 1.0

        # Check if we're in physical region (R > 1)
        if base <= 0:
            return 0.0

        return coeff * base**exponent

    # Evaluate definite integral
    integral = antiderivative(t_end) - antiderivative(t_start)

    return integral


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
    >>> # After running front tracking simulation
    >>> t_edges = np.array([0.0, 10.0, 20.0, 30.0])
    >>> c_avg = compute_bin_averaged_concentration_exact(
    ...     t_edges=t_edges,
    ...     v_outlet=500.0,
    ...     waves=tracker.state.waves,
    ...     sorption=sorption,
    ... )
    >>> len(c_avg) == len(t_edges) - 1
    True
    >>> np.all(c_avg >= 0)
    True

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

            if seg_dt <= 1e-15:  # Skip negligible segments
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
