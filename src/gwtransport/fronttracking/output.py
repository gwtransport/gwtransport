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
compute_cumulative_inlet_mass(t, cin, flow, tedges_days)
    Compute cumulative mass entering domain from t=0 to t
compute_cumulative_outlet_mass(t, v_outlet, waves, sorption, flow, tedges_days)
    Compute cumulative mass exiting domain from t=0 to t
find_last_rarefaction_start_theta(v_outlet, waves)
    Find θ at which last rarefaction head reaches outlet
integrate_rarefaction_total_mass(raref, v_outlet, theta_start, sorption)
    Compute total mass exiting through rarefaction to infinity (in θ)
compute_total_outlet_mass(v_outlet, waves, sorption, flow, tedges_days)
    Compute total integrated outlet mass until all mass has exited

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from collections.abc import Sequence
from operator import itemgetter

import mpmath as mp
import numpy as np
import numpy.typing as npt
from scipy.special import beta as beta_func
from scipy.special import betainc

from gwtransport.fronttracking.events import find_outlet_crossing
from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    LangmuirSorption,
    NonlinearSorption,
    SorptionModel,
)
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave, Wave

# Numerical tolerance constants
EPSILON_VELOCITY = 1e-15  # Tolerance for checking if velocity is effectively zero
EPSILON_BETA = 1e-15  # Tolerance for checking if beta is effectively zero (linear case)
EPSILON_TIME = 1e-15  # Tolerance for negligible time segments
EPSILON_TIME_MATCH = 1e-6  # Tolerance for matching arrival times (for rarefaction identification)
EPSILON_CONCENTRATION = 1e-10  # Tolerance for checking if concentration is effectively zero


def concentration_at_point(
    v: float,
    theta: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,  # noqa: ARG001
) -> float:
    """Compute concentration at point (v, θ) with exact analytical value.

    The function works entirely in (V, θ) coordinates: public callers must
    translate user-facing time t → θ at the API boundary (e.g., via
    ``FrontTrackerState.theta_at_t``).

    Parameters
    ----------
    v : float
        Position [m³].
    theta : float
        Cumulative flow [m³].
    waves : list of Wave
        All waves in the simulation (active and inactive).
    sorption : SorptionModel
        Sorption model (unused — kept for API symmetry; wave methods carry
        their own sorption reference).

    Returns
    -------
    concentration : float
        Concentration at point (v, θ) [mass/volume].

    Notes
    -----
    **Wave priority**: rarefaction fans first (spatial extent), then most
    recently crossing shock or rarefaction tail, then characteristics. If
    no active wave controls the point, returns 0.0 (initial condition).
    """
    for wave in waves:
        if isinstance(wave, RarefactionWave) and wave.is_active:
            c = wave.concentration_at_point(v, theta)
            if c is not None:
                return c

    latest_wave_theta = -np.inf
    latest_wave_c = None

    for wave in waves:
        if isinstance(wave, ShockWave) and wave.is_active:
            v_shock = wave.position_at_theta(theta)
            if v_shock is not None:
                tol = 1e-15

                if abs(v - v_shock) < tol:
                    return 0.5 * (wave.c_left + wave.c_right)

                if wave.speed is not None and abs(wave.speed) > EPSILON_VELOCITY:
                    theta_cross = wave.theta_start + (v - wave.v_start) / wave.speed

                    if theta_cross <= theta:
                        if theta_cross > latest_wave_theta:
                            latest_wave_theta = theta_cross
                            latest_wave_c = wave.c_left
                    elif v > v_shock and wave.theta_start > latest_wave_theta:
                        latest_wave_theta = wave.theta_start
                        latest_wave_c = wave.c_right

    for wave in waves:
        if isinstance(wave, RarefactionWave) and wave.is_active:
            v_tail = wave.tail_position_at_theta(theta)
            if v_tail is not None and v_tail > v + 1e-15:
                tail_speed = wave.tail_speed()
                if tail_speed > EPSILON_VELOCITY:
                    theta_pass = wave.theta_start + (v - wave.v_start) / tail_speed
                    if theta_pass <= theta and theta_pass > latest_wave_theta:
                        latest_wave_theta = theta_pass
                        latest_wave_c = wave.c_tail

    if latest_wave_c is not None:
        return latest_wave_c

    latest_c = 0.0
    latest_theta = -np.inf

    for wave in waves:
        if isinstance(wave, CharacteristicWave) and wave.is_active:
            v_char_at_theta = wave.position_at_theta(theta)

            if v_char_at_theta is not None and v_char_at_theta >= v - 1e-15:
                speed = wave.speed()

                if speed > EPSILON_VELOCITY:
                    theta_pass = wave.theta_start + (v - wave.v_start) / speed

                    if theta_pass <= theta and theta_pass > latest_theta:
                        latest_theta = theta_pass
                        latest_c = wave.concentration

    return latest_c


def compute_breakthrough_curve(
    t_array: npt.NDArray[np.floating],
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
    *,
    theta_edges: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute concentration at outlet over time array.

    This is the breakthrough curve: C(v_outlet, t) for all t in t_array.

    Parameters
    ----------
    t_array : array-like
        Time points [days]. Must be sorted in ascending order.
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    c_out : numpy.ndarray
        Array of concentrations matching t_array [mass/volume].

    See Also
    --------
    concentration_at_point : Point-wise concentration
    compute_bin_averaged_concentration_exact : Bin-averaged concentrations

    Examples
    --------
    ::

        t_array = np.linspace(0, 100, 1000)
        c_out = compute_breakthrough_curve(
            t_array, v_outlet=500.0, waves=all_waves, sorption=sorption
        )
        len(c_out) == len(t_array)
    """
    t_array = np.asarray(t_array, dtype=float)
    theta_array = np.interp(t_array, tedges_days, theta_edges)
    c_out = np.zeros(len(t_array))

    for i, theta in enumerate(theta_array):
        c_out[i] = concentration_at_point(v_outlet, float(theta), waves, sorption)

    return c_out


def identify_outlet_segments(
    theta_start: float,
    theta_end: float,
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
) -> list[dict]:
    """Identify which waves control outlet concentration in θ-interval [theta_start, theta_end].

    Finds all wave crossing events at the outlet and constructs segments where
    concentration is constant or varying (rarefaction). All times are expressed
    as cumulative flow θ [m³].

    Parameters
    ----------
    theta_start : float
        Start of cumulative-flow interval [m³].
    theta_end : float
        End of cumulative-flow interval [m³].
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    segments : list of dict
        List of segment dictionaries, each containing:

        - 'theta_start' : float
            Segment start θ [m³]
        - 'theta_end' : float
            Segment end θ [m³]
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

    1. Finding all wave crossing events at the outlet for θ in [theta_start, theta_end].
    2. Sorting events by θ.
    3. Creating constant-concentration segments between events.
    4. Handling rarefaction fans with θ-varying concentration.

    The segments completely partition the interval [theta_start, theta_end].
    """
    # Find all waves that cross outlet in this θ-range
    outlet_events: list[dict] = []

    # Track rarefactions that already contain the outlet at theta_start
    # These need to be handled separately since they don't generate crossing events
    active_rarefactions_at_start: list[RarefactionWave] = []

    for wave in waves:
        if not wave.is_active:
            continue

        # For rarefactions, detect both head and tail crossings
        if isinstance(wave, RarefactionWave):
            # Check if outlet is already inside this rarefaction at theta_start
            if wave.contains_point(v_outlet, theta_start):
                active_rarefactions_at_start.append(wave)
                # Detect when the tail crosses during [theta_start, theta_end]
                tail_speed = wave.tail_speed()
                if tail_speed > EPSILON_VELOCITY:
                    theta_cross = wave.theta_start + (v_outlet - wave.v_start) / tail_speed
                    if theta_start < theta_cross <= theta_end:
                        outlet_events.append({
                            "theta": theta_cross,
                            "wave": wave,
                            "boundary": "tail",
                            "c_after": wave.c_tail,
                        })
                continue

            # Head crossing
            head_speed = wave.head_speed()
            if head_speed > EPSILON_VELOCITY and wave.v_start < v_outlet:
                theta_cross = wave.theta_start + (v_outlet - wave.v_start) / head_speed
                if theta_start <= theta_cross <= theta_end:
                    outlet_events.append({
                        "theta": theta_cross,
                        "wave": wave,
                        "boundary": "head",
                        "c_after": wave.c_head,
                    })

            # Tail crossing
            tail_speed = wave.tail_speed()
            if tail_speed > EPSILON_VELOCITY and wave.v_start < v_outlet:
                theta_cross = wave.theta_start + (v_outlet - wave.v_start) / tail_speed
                if theta_start <= theta_cross <= theta_end:
                    outlet_events.append({
                        "theta": theta_cross,
                        "wave": wave,
                        "boundary": "tail",
                        "c_after": wave.c_tail,
                    })
        else:
            # Characteristics and shocks
            theta_cross = find_outlet_crossing(wave, v_outlet, theta_start)

            if theta_cross is not None and theta_start <= theta_cross <= theta_end:
                if isinstance(wave, CharacteristicWave):
                    c_after = wave.concentration
                elif isinstance(wave, ShockWave):
                    # After shock passes outlet, outlet sees left (upstream) state
                    c_after = wave.c_left
                else:
                    c_after = 0.0

                outlet_events.append({"theta": theta_cross, "wave": wave, "boundary": None, "c_after": c_after})

    # Sort events by θ
    outlet_events.sort(key=itemgetter("theta"))

    # Create segments between events
    segments: list[dict] = []
    current_theta = theta_start
    current_c = concentration_at_point(v_outlet, theta_start, waves, sorption)

    # Handle case where we start inside a rarefaction
    if active_rarefactions_at_start:
        # Should only be one rarefaction containing the outlet at theta_start
        raref = active_rarefactions_at_start[0]

        # Find when tail crosses (if it does)
        tail_cross_theta = None
        for event in outlet_events:
            if event["wave"] is raref and event["boundary"] == "tail" and event["theta"] > theta_start:
                tail_cross_theta = event["theta"]
                break

        # Create rarefaction segment from theta_start
        raref_end = min(tail_cross_theta or theta_end, theta_end)

        c_start = concentration_at_point(v_outlet, theta_start, waves, sorption)
        c_end = None
        if tail_cross_theta and tail_cross_theta <= theta_end:
            c_end = raref.c_tail

        segments.append({
            "theta_start": theta_start,
            "theta_end": raref_end,
            "type": "rarefaction",
            "wave": raref,
            "c_start": c_start,
            "c_end": c_end,
        })

        current_theta = raref_end
        current_c = (
            concentration_at_point(v_outlet, raref_end + 1e-10, waves, sorption) if raref_end < theta_end else current_c
        )

    for event in outlet_events:
        # Check if we're entering a rarefaction fan
        if isinstance(event["wave"], RarefactionWave) and event["boundary"] == "head":
            # Before rarefaction head: constant segment
            if event["theta"] > current_theta:
                segments.append({
                    "theta_start": current_theta,
                    "theta_end": event["theta"],
                    "type": "constant",
                    "concentration": current_c,
                    "c_start": current_c,
                    "c_end": current_c,
                })

            # Find when tail crosses (if it does)
            raref = event["wave"]
            tail_cross_theta = None

            for later_event in outlet_events:
                if (
                    later_event["wave"] is raref
                    and later_event["boundary"] == "tail"
                    and later_event["theta"] > event["theta"]
                ):
                    tail_cross_theta = later_event["theta"]
                    break

            # Rarefaction segment
            raref_end = min(tail_cross_theta or theta_end, theta_end)

            segments.append({
                "theta_start": event["theta"],
                "theta_end": raref_end,
                "type": "rarefaction",
                "wave": raref,
                "c_start": raref.c_head,
                "c_end": raref.c_tail if tail_cross_theta and tail_cross_theta <= theta_end else None,
            })

            current_theta = raref_end
            current_c = (
                concentration_at_point(v_outlet, raref_end + 1e-10, waves, sorption)
                if raref_end < theta_end
                else current_c
            )
        else:
            # Regular event (characteristic or shock crossing)
            # Segment before event
            if event["theta"] > current_theta:
                segments.append({
                    "theta_start": current_theta,
                    "theta_end": event["theta"],
                    "type": "constant",
                    "concentration": current_c,
                    "c_start": current_c,
                    "c_end": current_c,
                })

            current_theta = event["theta"]
            current_c = event["c_after"]

    # Final segment
    if theta_end > current_theta:
        segments.append({
            "theta_start": current_theta,
            "theta_end": theta_end,
            "type": "constant",
            "concentration": current_c,
            "c_start": current_c,
            "c_end": current_c,
        })

    return segments


def integrate_rarefaction_exact(
    raref: RarefactionWave, v_outlet: float, theta_start: float, theta_end: float, sorption: SorptionModel
) -> float:
    """Exact θ-integral ``∫ c(θ) dθ`` of a rarefaction at the outlet.

    Dispatches to the isotherm-specific closed-form formula. The result is
    the mass-like quantity ``∫ c dθ`` (equal to ``∫ c·flow dt`` in time
    coordinates).

    Parameters
    ----------
    raref : RarefactionWave
        Rarefaction wave controlling the outlet.
    v_outlet : float
        Outlet position [m³].
    theta_start, theta_end : float
        Integration range in cumulative flow [m³]. Either can be ``±np.inf``.
    sorption : SorptionModel
        Sorption model (``FreundlichSorption`` or ``LangmuirSorption``).

    Returns
    -------
    integral : float
        ``∫ c(θ) dθ`` [mass — i.e. concentration × volume].

    Raises
    ------
    TypeError
        If sorption model does not support exact rarefaction integration.
    """
    if isinstance(sorption, FreundlichSorption):
        return _integrate_rarefaction_exact_freundlich(raref, v_outlet, theta_start, theta_end, sorption)
    if isinstance(sorption, LangmuirSorption):
        return _integrate_rarefaction_exact_langmuir(raref, v_outlet, theta_start, theta_end, sorption)
    msg = f"Exact rarefaction integration not supported for {type(sorption).__name__}"
    raise TypeError(msg)


def _integrate_rarefaction_exact_freundlich(
    raref: RarefactionWave, v_outlet: float, theta_start: float, theta_end: float, sorption: FreundlichSorption
) -> float:
    """Exact θ-integral ``∫ c(θ) dθ`` for a Freundlich rarefaction.

    In (V, θ), self-similar concentration inside the rarefaction is::

        c(θ) = [((θ - θ_origin)/(v_outlet - v_origin) - 1) / α]^(1/β)

    with ``α = ρ_b·k_f/(n_por·n)`` and ``β = 1/n - 1``. The antiderivative is
    ``F(θ) = coeff · base(θ)^(1/β + 1)`` where
    ``base(θ) = κ_θ·θ + μ_θ - 1`` with ``κ_θ = 1/(v_outlet - v_origin)``
    and ``μ_θ = -θ_origin/(v_outlet - v_origin)``.

    The user-facing mass integral is ``∫ c·flow dt = ∫ c dθ``; the user-facing
    time integral over a constant-flow bin is ``(1/flow) · ∫ c dθ``.

    Raises
    ------
    ValueError
        If sorption is linear (n = 1) or the integral diverges at θ=+∞.
    """
    theta_origin = raref.theta_start
    v_origin = raref.v_start

    kappa_theta = 1.0 / (v_outlet - v_origin)
    mu_theta = -theta_origin / (v_outlet - v_origin)

    alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)
    beta = 1.0 / sorption.n - 1.0

    if abs(beta) < EPSILON_BETA:
        msg = "integrate_rarefaction_exact requires nonlinear sorption (n != 1)"
        raise ValueError(msg)

    exponent = 1.0 / beta + 1.0
    coeff = 1.0 / (alpha ** (1.0 / beta) * kappa_theta * exponent)

    def antiderivative(theta: float) -> float:
        if np.isinf(theta):
            if theta > 0:
                if exponent < 0:
                    return 0.0
                msg = f"Integral diverges at θ=+∞ with exponent={exponent} > 0"
                raise ValueError(msg)
            return 0.0

        base = kappa_theta * theta + mu_theta - 1.0
        if base <= 0:
            return 0.0
        return coeff * base**exponent

    return antiderivative(theta_end) - antiderivative(theta_start)


def _integrate_rarefaction_exact_langmuir(
    raref: RarefactionWave, v_outlet: float, theta_start: float, theta_end: float, sorption: LangmuirSorption
) -> float:
    """Exact θ-integral ``∫ c(θ) dθ`` for a Langmuir rarefaction.

    Inside the fan, ``c(θ) = sqrt(A / B(θ)) - K_L`` with
    ``B(θ) = κ_θ·θ + μ_θ - 1``, ``κ_θ = 1/(v_outlet - v_origin)``,
    ``μ_θ = -θ_origin/(v_outlet - v_origin)``, and
    ``A = ρ_b·s_max·K_L/n_por``.

    Antiderivative: ``F(θ) = (2·sqrt(A)/κ_θ)·sqrt(B(θ)) - K_L·θ``.
    """
    theta_origin = raref.theta_start
    v_origin = raref.v_start

    kappa_theta = 1.0 / (v_outlet - v_origin)
    mu_theta = -theta_origin / (v_outlet - v_origin)
    a_coeff = sorption.a_coeff
    k_l = sorption.k_l

    coeff_sqrt = 2.0 * np.sqrt(a_coeff) / kappa_theta

    def antiderivative(theta: float) -> float:
        if np.isinf(theta):
            if theta > 0:
                msg = "Langmuir rarefaction integral diverges at θ=+∞"
                raise ValueError(msg)
            return 0.0

        base = kappa_theta * theta + mu_theta - 1.0
        if base <= 0:
            return 0.0
        return coeff_sqrt * np.sqrt(base) - k_l * theta

    return antiderivative(theta_end) - antiderivative(theta_start)


def compute_bin_averaged_concentration_exact(
    t_edges: npt.NDArray[np.floating],
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
    *,
    theta_edges: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    Compute bin-averaged concentration using EXACT analytical integration.

    For each time bin [t_i, t_{i+1}], computes:
        C_avg = (1/(t_{i+1} - t_i)) * ∫_{t_i}^{t_{i+1}} C(v_outlet, t) dt

    This is the critical function for maintaining machine precision in output.
    All integrations use exact analytical formulas with no numerical quadrature.

    Parameters
    ----------
    t_edges : array-like
        Time bin edges [days]. Length N+1 for N bins.
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves from front tracking simulation.
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    c_avg : numpy.ndarray
        Bin-averaged concentrations [mass/volume]. Length N.

    Raises
    ------
    ValueError
        If any time bin has non-positive width (``t_edges[i] >= t_edges[i+1]``),
        or if an unknown segment type is encountered during integration.

    See Also
    --------
    concentration_at_point : Point-wise concentration
    compute_breakthrough_curve : Breakthrough curve
    integrate_rarefaction_exact : Exact rarefaction integration

    Notes
    -----
    **Algorithm**:

    1. For each bin [t_i, t_{i+1}]:

       a. Identify which wave segments control outlet during this period
       b. For each segment, compute: Constant C gives integral = C * Δt,
          Rarefaction C(t) uses exact analytical integral formula
       c. Sum segment integrals and divide by bin width

    **Machine Precision**:

    - Constant segments: exact to floating-point precision
    - Rarefaction segments: uses closed-form antiderivative
    - No numerical quadrature or interpolation
    - Maintains mass balance to ~1e-14 relative error

    **Rarefaction Integration**:

    For Freundlich sorption, rarefaction concentration at outlet varies as::

        C(t) = [(kappa*t + mu - 1)/alpha]^(1/beta)

    The exact integral is::

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
    """
    t_edges = np.asarray(t_edges, dtype=float)
    tedges_days_arr = np.asarray(tedges_days, dtype=float)
    theta_edges_arr = np.asarray(theta_edges, dtype=float)
    n_bins = len(t_edges) - 1
    c_avg = np.zeros(n_bins)

    for i in range(n_bins):
        t_start = t_edges[i]
        t_end = t_edges[i + 1]
        dt = t_end - t_start

        if dt <= 0:
            msg = f"Invalid time bin: t_edges[{i}]={t_start} >= t_edges[{i + 1}]={t_end}"
            raise ValueError(msg)

        # identify_outlet_segments works in (V, θ). Translate the user-facing
        # bin [t_start, t_end] to a θ-range and back.
        theta_start = float(np.interp(t_start, tedges_days_arr, theta_edges_arr))
        theta_end = float(np.interp(t_end, tedges_days_arr, theta_edges_arr))
        segments = identify_outlet_segments(theta_start, theta_end, v_outlet, waves, sorption)

        # Sum ∫c dθ over segments; convert to ∫c dt via average flow over the bin.
        total_integral_theta = 0.0
        for seg in segments:
            seg_theta_start = max(seg["theta_start"], theta_start)
            seg_theta_end = min(seg["theta_end"], theta_end)
            seg_dtheta = seg_theta_end - seg_theta_start

            if seg_dtheta <= EPSILON_TIME:
                continue

            if seg["type"] == "constant":
                total_integral_theta += seg["concentration"] * seg_dtheta
            elif seg["type"] == "rarefaction":
                if isinstance(sorption, NonlinearSorption):
                    raref = seg["wave"]
                    total_integral_theta += integrate_rarefaction_exact(
                        raref, v_outlet, seg_theta_start, seg_theta_end, sorption
                    )
                else:
                    theta_mid = 0.5 * (seg_theta_start + seg_theta_end)
                    c_mid = concentration_at_point(v_outlet, theta_mid, waves, sorption)
                    total_integral_theta += c_mid * seg_dtheta
            else:
                msg = f"Unknown segment type: {seg['type']}"
                raise ValueError(msg)

        # ∫c dt = ∫c dθ / flow_avg where flow_avg = Δθ / Δt over [t_start, t_end].
        dtheta_bin = theta_end - theta_start
        if dtheta_bin > 0:
            c_avg[i] = total_integral_theta / dtheta_bin
        else:
            c_avg[i] = 0.0

    return c_avg


def compute_domain_mass(
    theta: float,
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
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
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    mass : float
        Total mass in domain [mass]. Computed to machine precision (~1e-14).

    See Also
    --------
    compute_cumulative_inlet_mass : Cumulative inlet mass
    compute_cumulative_outlet_mass : Cumulative outlet mass
    concentration_at_point : Point-wise concentration

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
    """
    # Partition spatial domain into segments based on wave structure
    # We'll evaluate concentration at many points and identify constant regions

    # Collect all wave positions at time t
    wave_positions = []

    for wave in waves:
        if not wave.is_active:
            continue

        if isinstance(wave, (CharacteristicWave, ShockWave)):
            v_pos = wave.position_at_theta(theta)
            if v_pos is not None and 0 <= v_pos <= v_outlet:
                wave_positions.append(v_pos)

        elif isinstance(wave, RarefactionWave):
            v_head = wave.head_position_at_theta(theta)
            v_tail = wave.tail_position_at_theta(theta)

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
            if isinstance(wave, RarefactionWave) and wave.is_active and wave.contains_point(v_mid, theta):
                inside_rarefaction = True
                raref_wave = wave
                break

        if inside_rarefaction and raref_wave is not None:
            # Rarefaction: concentration varies with position
            # Use EXACT analytical spatial integration
            mass_segment = _integrate_rarefaction_spatial_exact(raref_wave, v_start, v_end, theta, sorption)
        else:
            # Constant concentration region - exact integration
            v_mid = 0.5 * (v_start + v_end)
            c = concentration_at_point(v_mid, theta, waves, sorption)
            c_total = sorption.total_concentration(c)
            mass_segment = c_total * dv

        total_mass += mass_segment

    return total_mass


def _integrate_rarefaction_spatial_exact(
    raref: RarefactionWave,
    v_start: float,
    v_end: float,
    theta: float,
    sorption: SorptionModel,
) -> float:
    """Exact analytical spatial integral of rarefaction total concentration at fixed θ.

    In (V, θ) the self-similar fan satisfies ``R(C) = (θ - θ_origin)/(v - v_origin)``;
    define ``kappa = θ - θ_origin`` and ``u = v - v_origin``. The dissolved and sorbed
    contributions to ∫ C_total(v) dv each reduce to power-law forms in u that admit
    closed forms via incomplete beta functions (Freundlich) or elementary sqrt
    operations (Langmuir).

    Parameters
    ----------
    raref : RarefactionWave
        Rarefaction wave.
    v_start, v_end : float
        Integration range [m³].
    theta : float
        Cumulative flow at which to evaluate [m³].
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    mass : float
        Mass in the segment ``[v_start, v_end]``.

    Raises
    ------
    TypeError
        If sorption model does not support exact spatial integration.
    """
    if isinstance(sorption, ConstantRetardation):
        v_mid = 0.5 * (v_start + v_end)
        c = raref.concentration_at_point(v_mid, theta) or 0.0
        c_total = sorption.total_concentration(c)
        return c_total * (v_end - v_start)

    theta_origin = raref.theta_start
    v_origin = raref.v_start

    if theta <= theta_origin:
        return 0.0

    kappa = theta - theta_origin
    u_start = v_start - v_origin
    u_end = v_end - v_origin

    if u_start <= 0 or u_end <= 0:
        return 0.0

    if isinstance(sorption, LangmuirSorption):
        return _integrate_rarefaction_spatial_langmuir(sorption, kappa, u_start, u_end)

    if isinstance(sorption, FreundlichSorption):
        return _integrate_rarefaction_spatial_freundlich(sorption, kappa, u_start, u_end)

    msg = f"Exact spatial rarefaction integration not supported for {type(sorption).__name__}"
    raise TypeError(msg)


def _integrate_rarefaction_spatial_freundlich(
    sorption: FreundlichSorption, kappa: float, u_start: float, u_end: float
) -> float:
    """Exact spatial integral for Freundlich rarefaction using beta functions.

    Returns
    -------
    float
        Exact mass in segment.
    """
    alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)
    rho_b = sorption.bulk_density
    n_por = sorption.porosity
    k_f = sorption.k_f
    n = sorption.n

    beta = 1 / n - 1
    t_start_norm = u_start / kappa
    t_end_norm = u_end / kappa

    # Dissolved: ∫ C(u) du via incomplete beta function
    a_diss = 1 - 1 / beta
    b_diss = 1 + 1 / beta

    if a_diss > 0 and b_diss > 0:
        beta_diss = betainc(a_diss, b_diss, t_end_norm) - betainc(a_diss, b_diss, t_start_norm)
        beta_diss *= beta_func(a_diss, b_diss)
    else:
        beta_diss = float(mp.betainc(a_diss, b_diss, t_start_norm, t_end_norm, regularized=False))

    coeff_diss = (1 / alpha) ** (1 / beta)
    mass_dissolved = coeff_diss * kappa * beta_diss

    # Sorbed: ∫ (rho_b/n_por)*k_f*C^(1/n) du
    exponent_sorb = 1 / (1 - n)
    a_sorb = 1 - exponent_sorb
    b_sorb = 1 + exponent_sorb

    if a_sorb > 0 and b_sorb > 0:
        beta_sorb = betainc(a_sorb, b_sorb, t_end_norm) - betainc(a_sorb, b_sorb, t_start_norm)
        beta_sorb *= beta_func(a_sorb, b_sorb)
    else:
        beta_sorb = float(mp.betainc(a_sorb, b_sorb, t_start_norm, t_end_norm, regularized=False))

    coeff_sorb = (rho_b / n_por) * k_f / (alpha**exponent_sorb)
    mass_sorbed = coeff_sorb * kappa * beta_sorb

    return mass_dissolved + mass_sorbed


def _integrate_rarefaction_spatial_langmuir(
    sorption: LangmuirSorption, kappa: float, u_start: float, u_end: float
) -> float:
    """Exact spatial integral of C_total(v) for Langmuir rarefaction.

    Returns
    -------
    float
        Exact mass in segment.

    Notes
    -----
    With u = v - v_origin, kappa = flow*(t - t_origin):
        C(u) = sqrt(a_coeff*u/(kappa-u)) - K_L
        C_total = C + (rho_b/n_por) * s_max * C/(K_L + C)

    Since K_L + C = sqrt(a_coeff*u/(kappa-u)):
        C/(K_L+C) = 1 - K_L*sqrt((kappa-u)/(a_coeff*u))

    The integral simplifies to:
        integral C_total du = -2*sqrt(a_coeff)*[sqrt(u*(kappa-u))]_start^end
                              + (rho_b*s_max/n_por - K_L)*(u_end - u_start)
    """
    a_coeff = sorption.a_coeff
    k_l = sorption.k_l
    sorbed_max = sorption.bulk_density * sorption.s_max / sorption.porosity

    term_sqrt_end = np.sqrt(u_end * (kappa - u_end))
    term_sqrt_start = np.sqrt(u_start * (kappa - u_start))

    return -2.0 * np.sqrt(a_coeff) * (term_sqrt_end - term_sqrt_start) + (sorbed_max - k_l) * (u_end - u_start)


def compute_cumulative_inlet_mass(
    t: float,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges_days: npt.ArrayLike,
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
    cin : array-like
        Inlet concentration time series [mass/volume].
        Piecewise constant within bins defined by tedges_days.
    flow : array-like
        Flow rate time series [m³/day].
        Piecewise constant within bins defined by tedges_days.
    tedges_days : numpy.ndarray
        Time bin edges [days]. Length len(cin) + 1.

    Returns
    -------
    mass_in : float
        Cumulative mass entered [mass].

    Notes
    -----
    For piecewise-constant cin and flow:
        M_in = Σ cin[i] * flow[i] * dt[i]

    where the sum is over all bins from tedges_days[0] to t.
    Partial bins are handled exactly.

    Examples
    --------
    ::

        mass_in = compute_cumulative_inlet_mass(
            t=50.0, cin=cin, flow=flow, tedges_days=tedges_days
        )
        mass_in >= 0.0
    """
    tedges_arr = np.asarray(tedges_days, dtype=float)
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


def find_last_rarefaction_start_theta(
    v_outlet: float,
    waves: Sequence[Wave],
) -> float:
    """Return the θ at which the last rarefaction head reaches ``v_outlet``.

    For non-rarefaction waves, returns the θ at which the wave crosses the outlet.
    Returns 0.0 if no waves reach the outlet.
    """
    theta_last = 0.0

    for wave in waves:
        if not wave.is_active:
            continue

        if isinstance(wave, RarefactionWave):
            head_speed = wave.head_speed()
            if head_speed > EPSILON_VELOCITY:
                theta_arrival = wave.theta_start + (v_outlet - wave.v_start) / head_speed
                theta_last = max(theta_last, theta_arrival)
        elif isinstance(wave, (CharacteristicWave, ShockWave)):
            speed = wave.speed if isinstance(wave, ShockWave) else wave.speed()
            if speed is not None and speed > EPSILON_VELOCITY:
                theta_arrival = wave.theta_start + (v_outlet - wave.v_start) / speed
                theta_last = max(theta_last, theta_arrival)

    return theta_last


def compute_cumulative_outlet_mass(
    t: float,
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
    flow: npt.ArrayLike,
    tedges_days: npt.ArrayLike,
) -> float:
    """Compute cumulative mass exiting the domain from t=0 to ``t``.

    Internally evaluated in (V, θ) coordinates: ``mass = ∫ c(θ) dθ`` from 0 to
    ``θ(t)``. Because ``dθ = flow · dt``, the flow drops out of the integrand
    and the result equals ``∫ cout(τ) · flow(τ) dτ`` in time coordinates.

    Parameters
    ----------
    t : float
        User-facing time up to which to integrate [days from ``tedges_days[0]``].
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : SorptionModel
        Sorption model.
    flow : array-like
        Flow rate per bin [m³/day]; piecewise constant on ``tedges_days``.
    tedges_days : numpy.ndarray
        Time bin edges [days]. Length ``len(flow) + 1``.

    Returns
    -------
    mass_out : float
        Cumulative mass exited from t=0 to ``t`` [mass].

    Notes
    -----
    Algorithm:

    1. Build ``theta_edges = cumsum(flow · Δt)`` and translate ``t`` to
       ``theta_end`` by piecewise-linear interpolation against
       ``(tedges_days, theta_edges)``. Past ``tedges_days[-1]`` the last-bin
       flow is extrapolated.
    2. Call :func:`identify_outlet_segments` over ``[0, theta_end]``.
    3. For each segment: constant → ``c · Δθ``; rarefaction → exact analytic
       θ-integral via :func:`integrate_rarefaction_exact`.

    Examples
    --------
    ::

        mass_out = compute_cumulative_outlet_mass(
            t=50.0,
            v_outlet=500.0,
            waves=tracker.state.waves,
            sorption=sorption,
            flow=flow,
            tedges_days=tedges_days,
        )
        mass_out >= 0.0
    """
    tedges_arr = np.asarray(tedges_days, dtype=float)
    flow_arr = np.asarray(flow, dtype=float)

    if t <= tedges_arr[0]:
        return 0.0

    # Build θ-edges from flow × Δt, then translate the user-facing endpoint to θ.
    dt_days = np.diff(tedges_arr)
    theta_edges = np.concatenate([[0.0], np.cumsum(flow_arr * dt_days)])
    theta_start = float(theta_edges[0])

    if t >= tedges_arr[-1]:
        # Extrapolate past the last bin at the last-bin flow rate.
        theta_end = float(theta_edges[-1] + (t - tedges_arr[-1]) * flow_arr[-1])
    else:
        theta_end = float(np.interp(t, tedges_arr, theta_edges))

    if theta_end <= theta_start:
        return 0.0

    segments = identify_outlet_segments(theta_start, theta_end, v_outlet, waves, sorption)

    total_mass = 0.0

    for seg in segments:
        seg_theta_start = max(seg["theta_start"], theta_start)
        seg_theta_end = min(seg["theta_end"], theta_end)
        seg_dtheta = seg_theta_end - seg_theta_start

        if seg_dtheta <= EPSILON_TIME:
            continue

        if seg["type"] == "constant":
            total_mass += seg["concentration"] * seg_dtheta
        elif seg["type"] == "rarefaction":
            if isinstance(sorption, NonlinearSorption):
                raref = seg["wave"]
                total_mass += integrate_rarefaction_exact(raref, v_outlet, seg_theta_start, seg_theta_end, sorption)
            else:
                # ConstantRetardation - rarefactions shouldn't form; fall back to midpoint.
                c_mid = concentration_at_point(v_outlet, 0.5 * (seg_theta_start + seg_theta_end), waves, sorption)
                total_mass += c_mid * seg_dtheta

    return float(total_mass)


def integrate_rarefaction_total_mass(
    raref: RarefactionWave,
    v_outlet: float,
    theta_start: float,
    sorption: SorptionModel,
) -> float:
    """Total mass exiting through a rarefaction at the outlet (in (V, θ)).

    Mass equals ``∫ c(θ) dθ`` from ``theta_start`` (rarefaction-head outlet
    crossing) to either ``θ=+∞`` (if c_tail ≈ 0) or the θ at which the tail
    crosses the outlet.

    Parameters
    ----------
    raref : RarefactionWave
        Rarefaction wave.
    v_outlet : float
        Outlet position [m³].
    theta_start : float
        Cumulative flow at which the rarefaction head reaches the outlet [m³].
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    total_mass : float
        Total mass that exits through rarefaction [mass].
    """
    if isinstance(sorption, ConstantRetardation):
        return 0.0

    if raref.c_tail < EPSILON_CONCENTRATION:
        theta_end = np.inf
    else:
        tail_speed = raref.tail_speed()
        if tail_speed < EPSILON_VELOCITY:
            theta_end = np.inf
        else:
            theta_end = raref.theta_start + (v_outlet - raref.v_start) / tail_speed

    return integrate_rarefaction_exact(raref, v_outlet, theta_start, theta_end, sorption)


def compute_total_outlet_mass(
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
    flow: npt.ArrayLike,
    tedges_days: npt.ArrayLike,
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
    sorption : SorptionModel
        Sorption model.
    flow : array-like
        Flow rate time series [m³/day].
        Piecewise constant within bins defined by tedges_days.
    tedges_days : numpy.ndarray
        Time bin edges [days]. Length len(flow) + 1.

    Returns
    -------
    total_mass_out : float
        Total mass that exits through outlet [mass].
    t_integration_end : float
        Time until which integration was performed [days].
        This is the time when the last wave passes the outlet.

    See Also
    --------
    compute_cumulative_outlet_mass : Cumulative outlet mass up to time t
    find_last_rarefaction_start_theta : Find θ at which last rarefaction starts
    integrate_rarefaction_total_mass : Total mass in rarefaction to infinity

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
            tedges_days=tedges_days,
        )
        total_mass >= 0.0
        t_end >= tedges_days[0]
    """
    tedges_arr = np.asarray(tedges_days, dtype=float)
    flow_arr = np.asarray(flow, dtype=float)

    # Find the θ at which the last rarefaction head reaches the outlet
    theta_last_raref_start = find_last_rarefaction_start_theta(v_outlet, waves)

    # Build θ-edges from flow × Δt for the t ↔ θ translation
    dt_days = np.diff(tedges_arr)
    theta_edges = np.concatenate([[0.0], np.cumsum(flow_arr * dt_days)])

    # Translate theta_last_raref_start back to user-facing time, extending
    # past tedges_days[-1] using the last-bin flow if necessary.
    last_flow = float(flow_arr[-1])
    if theta_last_raref_start <= theta_edges[-1]:
        t_last_raref_start = float(np.interp(theta_last_raref_start, theta_edges, tedges_arr))
    elif last_flow > 0:
        t_last_raref_start = float(tedges_arr[-1] + (theta_last_raref_start - theta_edges[-1]) / last_flow)
    else:
        t_last_raref_start = float(tedges_arr[-1])

    # Mass up to theta_last_raref_start (∫ c dθ); reuse the cumulative routine
    # so the t ↔ θ extrapolation rules stay consistent.
    mass_up_to_raref_start = compute_cumulative_outlet_mass(
        t_last_raref_start, v_outlet, waves, sorption, flow_arr, tedges_arr
    )

    # Add the analytical tail-to-infinity contribution for every rarefaction
    # whose head crosses the outlet at exactly theta_last_raref_start.
    total_raref_mass = 0.0

    for wave in waves:
        if not wave.is_active:
            continue

        if isinstance(wave, RarefactionWave):
            head_speed = wave.head_speed()
            if head_speed > EPSILON_VELOCITY and wave.v_start < v_outlet:
                theta_head_crosses_outlet = wave.theta_start + (v_outlet - wave.v_start) / head_speed

                # Compare in θ-space (flow-free) — equivalent to the legacy time
                # tolerance but unit-agnostic with respect to flow rate.
                if abs(theta_head_crosses_outlet - theta_last_raref_start) < EPSILON_TIME_MATCH:
                    total_raref_mass += integrate_rarefaction_total_mass(
                        raref=wave,
                        v_outlet=v_outlet,
                        theta_start=theta_head_crosses_outlet,
                        sorption=sorption,
                    )

    total_mass = mass_up_to_raref_start + total_raref_mass

    return float(total_mass), t_last_raref_start
