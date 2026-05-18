"""Concentration extraction from front-tracking solutions (V, θ coordinates).

Every public function in this module takes θ (cumulative flow, m³). Callers
translate user-facing time t → θ at the API boundary via
``FrontTrackerState.theta_at_t``.

Functions
---------
concentration_at_point(v, theta, waves, sorption)
compute_breakthrough_curve(theta_array, v_outlet, waves, sorption)
compute_bin_averaged_concentration_exact(theta_bin_edges, v_outlet, waves, sorption, *, cin=None, theta_edges_inlet=None)
compute_domain_mass(theta, v_outlet, waves, sorption)
compute_cumulative_inlet_mass(theta, cin, theta_edges)
compute_cumulative_outlet_mass(theta, v_outlet, waves, sorption, *, cin, theta_edges)
compute_total_outlet_mass(v_outlet, sorption, *, cin, theta_edges) -> (mass, theta_integration_end)

Step 5b note: outlet-mass functions use the PDE conservation identity
``m_out(θ) = m_in(θ) − m_dom(θ)`` (Bear & Cheng 2010, §3.5). m_dom honors
historical wave activity via ``wave.was_active_at(theta)`` so retrospective
queries at θ before a collision event correctly attribute c at v_outlet.

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
from gwtransport.fronttracking.waves import CharacteristicWave, DecayingShockWave, RarefactionWave, ShockWave, Wave

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
    **Wave priority**: decaying shocks first (closed-form analytical), then
    rarefaction fans (spatial extent), then most recently crossing shock or
    rarefaction tail, then characteristics. If no active wave controls the
    point, returns 0.0 (initial condition).
    """
    for wave in waves:
        if isinstance(wave, DecayingShockWave) and wave.is_active:
            c = wave.concentration_at_point(v, theta)
            if c is not None:
                return c

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
    theta_array: npt.NDArray[np.floating],
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
) -> npt.NDArray[np.floating]:
    """Concentration at the outlet evaluated over a θ-array (breakthrough curve).

    Parameters
    ----------
    theta_array : array-like
        Cumulative-flow points at which to query the outlet concentration [m³].
        Must be sorted in ascending order. Callers translate from user-facing
        time via ``FrontTrackerState.theta_at_t`` before passing.
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    c_out : numpy.ndarray
        Concentration at ``v_outlet`` for each θ in ``theta_array`` [mass/volume].

    See Also
    --------
    concentration_at_point : Point-wise concentration
    compute_bin_averaged_concentration_exact : Bin-averaged concentrations

    Examples
    --------
    ::

        theta_array = np.linspace(0.0, tracker.state.theta_edges[-1], 1000)
        c_out = compute_breakthrough_curve(
            theta_array, v_outlet=500.0, waves=tracker.state.waves, sorption=sorption
        )
    """
    theta_arr = np.asarray(theta_array, dtype=float)
    c_out = np.zeros(len(theta_arr))
    for i, theta in enumerate(theta_arr):
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
            ``'constant'``, ``'rarefaction'``, or ``'decaying_fan'``.
            ``'decaying_fan'`` is owned by a :class:`DecayingShockWave` after
            its head crosses ``v_outlet``; c at ``v_outlet`` then follows the
            wave's self-similar fan profile.
        - 'concentration' : float
            For constant segments
        - 'wave' : Wave
            For rarefaction and decaying_fan segments
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
    4. Handling rarefaction and decaying-fan profiles with θ-varying concentration.

    The segments completely partition the interval [theta_start, theta_end].
    """
    # Find all waves that cross outlet in this θ-range
    outlet_events: list[dict] = []

    # Track rarefactions / decaying shocks that already contain the outlet at
    # theta_start (no crossing event in [theta_start, theta_end]).
    active_rarefactions_at_start: list[RarefactionWave | DecayingShockWave] = []

    for wave in waves:
        if not wave.is_active:
            continue

        if isinstance(wave, DecayingShockWave):
            # The wave's outlet crossing arrival behaves like a rarefaction head
            # arrival: before arrival, v_outlet is downstream (c=c_fixed for
            # decay_side='left'); after arrival, v_outlet is inside the fan
            # whose c follows the self-similar profile and asymptotes to the
            # fan's tail concentration.
            theta_cross = wave.outlet_crossing_theta(v_outlet)
            if theta_cross is None:
                continue
            if theta_cross <= theta_start:
                # Outlet already inside the fan at theta_start.
                active_rarefactions_at_start.append(wave)
            elif theta_cross <= theta_end:
                # c_after is the fan c just past arrival (the decay-side c at
                # the arrival θ). theta_cross > wave.theta_start by construction
                # (outlet_crossing_theta enforces v_outlet > v_start), so
                # c_decay_at_theta does not return None.
                c_after = wave.c_decay_at_theta(theta_cross)
                outlet_events.append({
                    "theta": theta_cross,
                    "wave": wave,
                    "boundary": "head",
                    "c_after": c_after,
                })
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

    # Handle case where we start inside a rarefaction or decaying-shock fan
    if active_rarefactions_at_start:
        raref = active_rarefactions_at_start[0]

        if isinstance(raref, RarefactionWave):
            # Find when tail crosses (if it does)
            tail_cross_theta = None
            for event in outlet_events:
                if event["wave"] is raref and event["boundary"] == "tail" and event["theta"] > theta_start:
                    tail_cross_theta = event["theta"]
                    break

            raref_end = min(tail_cross_theta or theta_end, theta_end)
            c_end = raref.c_tail if tail_cross_theta and tail_cross_theta <= theta_end else None

            segments.append({
                "theta_start": theta_start,
                "theta_end": raref_end,
                "type": "rarefaction",
                "wave": raref,
                "c_start": current_c,
                "c_end": c_end,
            })
        else:
            # DecayingShockWave fan extends to θ=+∞ (or asymptotes to c_fixed
            # for n>1 with c_min); treat the whole [theta_start, theta_end]
            # as one decaying-fan segment.
            raref_end = theta_end
            c_end = concentration_at_point(v_outlet, theta_end, waves, sorption)

            segments.append({
                "theta_start": theta_start,
                "theta_end": raref_end,
                "type": "decaying_fan",
                "wave": raref,
                "c_start": current_c,
                "c_end": c_end,
            })

        current_theta = raref_end
        current_c = (
            concentration_at_point(v_outlet, raref_end + 1e-10, waves, sorption) if raref_end < theta_end else current_c
        )

    for event in outlet_events:
        # Skip events that fall inside an already-emitted (typically rarefaction)
        # segment. By Phase-1 convention ``concentration_at_point`` lets active
        # rarefactions "win" over a behind-shock c_left; the segment list must
        # reflect the same convention to avoid double-counting.
        if event["theta"] < current_theta:
            continue

        if isinstance(event["wave"], RarefactionWave) and event["boundary"] == "head":
            if event["theta"] > current_theta:
                segments.append({
                    "theta_start": current_theta,
                    "theta_end": event["theta"],
                    "type": "constant",
                    "concentration": current_c,
                    "c_start": current_c,
                    "c_end": current_c,
                })

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
        elif isinstance(event["wave"], DecayingShockWave) and event["boundary"] == "head":
            if event["theta"] > current_theta:
                segments.append({
                    "theta_start": current_theta,
                    "theta_end": event["theta"],
                    "type": "constant",
                    "concentration": current_c,
                    "c_start": current_c,
                    "c_end": current_c,
                })

            decaying = event["wave"]
            # The decaying_fan segment ends at the next outlet-crossing event
            # (if any falls in the window) or at theta_end. Without this split,
            # multi-DSW pulses would have the first DSW's fan swallow every
            # later wave's arrival.
            seg_end = theta_end
            for later_event in outlet_events:
                if later_event["theta"] > event["theta"] and later_event["theta"] <= theta_end:
                    seg_end = later_event["theta"]
                    break
            c_end_val = concentration_at_point(v_outlet, seg_end, waves, sorption)

            segments.append({
                "theta_start": event["theta"],
                "theta_end": seg_end,
                "type": "decaying_fan",
                "wave": decaying,
                "c_start": event["c_after"],
                "c_end": c_end_val,
            })

            current_theta = seg_end
            current_c = c_end_val
        else:
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

    Convenience wrapper over :func:`integrate_fan_exact` that pulls the fan
    apex from ``raref.theta_start, raref.v_start``. Returns the mass-like
    quantity ``∫ c dθ`` (= ``∫ c·flow dt`` in time coordinates).

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
    """
    return integrate_fan_exact(
        raref.theta_start, raref.v_start, v_outlet, theta_start, theta_end, sorption, c_apex=raref.c_tail
    )


def integrate_fan_exact(
    theta_origin: float,
    v_origin: float,
    v_outlet: float,
    theta_start: float,
    theta_end: float,
    sorption: SorptionModel,
    c_apex: float = 0.0,
) -> float:
    """Exact θ-integral ``∫ c(θ) dθ`` for any self-similar fan at the outlet.

    Decoupled from the wave object so the same closed-form math applies to
    both :class:`RarefactionWave` (apex = ``theta_start, v_start``) and
    :class:`DecayingShockWave` (apex = ``theta_origin, v_origin``).

    Parameters
    ----------
    theta_origin, v_origin : float
        Cumulative flow and position at the fan's apex [m³].
    v_outlet : float
        Outlet position [m³].
    theta_start, theta_end : float
        Integration range in cumulative flow [m³]. ``theta_end`` may be
        ``+np.inf``; ``theta_start`` must be finite.
    sorption : SorptionModel
        Sorption model (``FreundlichSorption`` or ``LangmuirSorption``).
    c_apex : float, optional
        Concentration on the constant side at the fan apex. For
        ``RarefactionWave`` this is ``raref.c_tail``; for
        ``DecayingShockWave`` (decay_side='left') this is ``wave.c_fixed``.
        For ``c_apex > 0`` the fan formula extrapolates past the physical
        fan range; the integration is clamped at ``θ_tail`` (where
        ``c(θ_tail) = c_apex``) and the constant-c_apex region beyond
        contributes ``c_apex · (theta_end − θ_tail)``. Default 0.0
        preserves the c=0 apex behavior for canonical c_R=0 fans.

    Returns
    -------
    float
        Mass-like quantity ``∫ c(θ) dθ`` [mass — concentration × volume].

    Raises
    ------
    TypeError
        If the sorption model does not support exact fan integration.
    """
    if isinstance(sorption, FreundlichSorption):
        return _integrate_fan_exact_freundlich(
            theta_origin, v_origin, v_outlet, theta_start, theta_end, sorption, c_apex
        )
    if isinstance(sorption, LangmuirSorption):
        if c_apex > 0.0:
            # Langmuir DSW c_fixed>0 is not currently constructible (handlers.py only
            # creates Langmuir DSW with c_fixed=0). Defensive guard so a future caller
            # bypassing the constructor doesn't silently get a wrong answer.
            msg = f"integrate_fan_exact: c_apex > 0 not supported for Langmuir (got {c_apex})"
            raise NotImplementedError(msg)
        return _integrate_fan_exact_langmuir(theta_origin, v_origin, v_outlet, theta_start, theta_end, sorption)
    msg = f"Exact fan integration not supported for {type(sorption).__name__}"
    raise TypeError(msg)


def _integrate_fan_exact_freundlich(
    theta_origin: float,
    v_origin: float,
    v_outlet: float,
    theta_start: float,
    theta_end: float,
    sorption: FreundlichSorption,
    c_apex: float = 0.0,
) -> float:
    """Exact θ-integral ``∫ c(θ) dθ`` for a Freundlich fan with given apex.

    In (V, θ), self-similar concentration inside the fan is::

        c(θ) = [((θ - θ_origin)/(v_outlet - v_origin) - 1) / α]^(1/β)

    with ``α = ρ_b·k_f/(n_por·n)`` and ``β = 1/n - 1``. The antiderivative is
    ``F(θ) = coeff · base(θ)^(1/β + 1)`` where
    ``base(θ) = κ_θ·θ + μ_θ - 1`` with ``κ_θ = 1/(v_outlet - v_origin)``
    and ``μ_θ = -θ_origin/(v_outlet - v_origin)``.

    For ``n > 1`` the antiderivative naturally vanishes at +∞ (exponent < 0).
    For ``n < 1`` (exponent > 0) the antiderivative grows without bound at +∞;
    callers must pass a finite ``theta_end``. The n<1 mirror DecayingShockWave
    mass-to-+∞ semantics are different (c = c_fixed downstream of the shock
    after arrival, so the +∞ contribution is 0) and are handled at the
    ``mass_after_outlet_arrival`` / ``compute_total_outlet_mass`` level, not
    here — see those functions for the dispatch.

    When ``c_apex > 0`` the fan formula extrapolates to ``c < c_apex`` for
    ``θ > θ_tail = θ_origin + (v_outlet - v_origin) · R(c_apex)``; the
    integration is split into the fan portion ``[θ_start, min(θ_end, θ_tail)]``
    via the antiderivative, plus a constant-c_apex contribution
    ``c_apex · (θ_end - θ_tail)`` for any ``θ_end > θ_tail``.

    Raises
    ------
    ValueError
        If sorption is linear (n = 1), or if ``theta_end = +∞`` with n < 1.
    """
    kappa_theta = 1.0 / (v_outlet - v_origin)
    mu_theta = -theta_origin / (v_outlet - v_origin)

    alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)
    beta = 1.0 / sorption.n - 1.0

    if abs(beta) < EPSILON_BETA:
        msg = "integrate_fan_exact requires nonlinear sorption (n != 1)"
        raise ValueError(msg)

    exponent = 1.0 / beta + 1.0
    coeff = 1.0 / (alpha ** (1.0 / beta) * kappa_theta * exponent)

    # Clamp the upper bound at θ_tail where the fan c(θ) reaches c_apex; beyond
    # θ_tail the formula is unphysical (gives c < c_apex). The constant-c_apex
    # region's contribution is added back at the bottom.
    if c_apex > 0.0:
        theta_tail = theta_origin + (v_outlet - v_origin) * float(sorption.retardation(c_apex))
        theta_end_fan = min(theta_end, theta_tail)
    else:
        theta_tail = float("inf")
        theta_end_fan = theta_end

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

    fan_integral = antiderivative(theta_end_fan) - antiderivative(theta_start)
    constant_contrib = c_apex * max(theta_end - theta_tail, 0.0) if c_apex > 0.0 else 0.0
    return fan_integral + constant_contrib


def _integrate_fan_exact_langmuir(
    theta_origin: float,
    v_origin: float,
    v_outlet: float,
    theta_start: float,
    theta_end: float,
    sorption: LangmuirSorption,
) -> float:
    """Exact θ-integral ``∫ c(θ) dθ`` for a Langmuir fan with given apex.

    Inside the fan, ``c(θ) = sqrt(A / B(θ)) - K_L`` with
    ``B(θ) = κ_θ·θ + μ_θ - 1``, ``κ_θ = 1/(v_outlet - v_origin)``,
    ``μ_θ = -θ_origin/(v_outlet - v_origin)``, and
    ``A = ρ_b·s_max·K_L/n_por``.

    Antiderivative: ``F(θ) = (2·sqrt(A)/κ_θ)·sqrt(B(θ)) - K_L·θ``. The Langmuir
    fan c reaches 0 at a finite θ_zero (when ``B = A/K_L²``); the antiderivative
    is only physically meaningful while ``c ≥ 0``. For ``theta_end = +∞`` the
    upper bound is clamped to ``theta_zero`` since c=0 beyond that point.
    """
    kappa_theta = 1.0 / (v_outlet - v_origin)
    mu_theta = -theta_origin / (v_outlet - v_origin)
    a_coeff = sorption.a_coeff
    k_l = sorption.k_l

    coeff_sqrt = 2.0 * np.sqrt(a_coeff) / kappa_theta

    # c(θ) = 0 when B(θ) = A/K_L², i.e. κ_θ·θ + μ_θ - 1 = A/K_L².
    theta_zero = (a_coeff / (k_l * k_l) + 1.0 - mu_theta) / kappa_theta
    theta_end = theta_zero if (np.isinf(theta_end) and theta_end > 0) else min(theta_end, theta_zero)

    def antiderivative(theta: float) -> float:
        base = kappa_theta * theta + mu_theta - 1.0
        if base <= 0:
            return 0.0
        return coeff_sqrt * np.sqrt(base) - k_l * theta

    return antiderivative(theta_end) - antiderivative(theta_start)


def compute_bin_averaged_concentration_exact(
    theta_bin_edges: npt.NDArray[np.floating],
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
    *,
    cin: npt.ArrayLike | None = None,
    theta_edges_inlet: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """θ-bin-averaged outlet concentration.

    For each θ-bin ``[θ_i, θ_{i+1}]``::

        C_avg = (1 / Δθ) · ∫_{θ_i}^{θ_{i+1}} C(v_outlet, θ) dθ

    With ``cin`` + ``theta_edges_inlet`` provided (recommended for multi-DSW
    cases), uses the conservation-law identity
    ``C_avg = (Δm_in − Δm_dom) / Δθ`` per bin — analytical and explicit, no
    outlet-side fan dispatch. Otherwise falls back to the legacy
    outlet-segment integration (correct for canonical single-DSW cases;
    may miscount multi-DSW or n<1 mirror geometries).

    Parameters
    ----------
    theta_bin_edges : array-like
        Cumulative-flow OUTPUT bin edges [m³] (where C_avg is reported).
        Length N+1 for N bins. Callers translate t-bin edges with
        ``state.theta_at_t``.
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves from front tracking simulation.
    sorption : SorptionModel
        Sorption model.
    cin : array-like, optional (kw-only)
        Inlet concentration per inlet θ-bin. When provided with
        ``theta_edges_inlet``, the conservation form is used.
    theta_edges_inlet : ndarray, optional (kw-only)
        θ bin edges of the INLET (``state.theta_edges``), length
        ``len(cin) + 1``.

    Returns
    -------
    c_avg : numpy.ndarray
        Bin-averaged outlet concentrations [mass/volume]. Length N.

    Raises
    ------
    ValueError
        If any output θ-bin has non-positive width.

    See Also
    --------
    concentration_at_point : Point-wise concentration
    compute_breakthrough_curve : Breakthrough curve
    compute_cumulative_outlet_mass : Cumulative outlet mass via conservation
    """
    theta_edges_out = np.asarray(theta_bin_edges, dtype=float)

    if np.any(np.diff(theta_edges_out) <= 0):
        bad = int(np.argmin(np.diff(theta_edges_out)))
        msg = (
            f"Invalid θ-bin: theta_bin_edges[{bad}]={theta_edges_out[bad]} >= "
            f"theta_bin_edges[{bad + 1}]={theta_edges_out[bad + 1]}"
        )
        raise ValueError(msg)

    if cin is not None and theta_edges_inlet is not None:
        # Conservation form: c_avg = Δm_out/Δθ where m_out = m_in − m_dom.
        m_out_at_edges = np.array([
            compute_cumulative_outlet_mass(
                theta=float(theta_edges_out[i]),
                v_outlet=v_outlet,
                waves=waves,
                sorption=sorption,
                cin=cin,
                theta_edges=theta_edges_inlet,
            )
            for i in range(len(theta_edges_out))
        ])
        result = np.diff(m_out_at_edges) / np.diff(theta_edges_out)
        # FP-noise + extrapolation clamp: m_in − m_dom subtracts nearly-equal
        # large numbers, leaving ~1 ULP residuals on either sign. Also, for
        # queries past theta_edges_inlet[-1] the simulator's wave list has
        # not been extended with new inlet emissions, so m_dom and m_in are
        # mutually inconsistent and the conservation form gives unphysical
        # values. Clamp negative concentrations to 0 (physical non-negativity)
        # and clamp tiny positive FP-noise residuals to 0 too.
        max_c = float(np.max(np.abs(result))) if result.size else 0.0
        eps_clamp = 1e-12 * max(max_c, 1.0)
        result = np.where(np.abs(result) < eps_clamp, 0.0, result)
        return np.maximum(result, 0.0)

    # Legacy outlet-segment integration (compatible with hand-constructed
    # wave-list tests; correct for canonical single-DSW cases).
    n_bins = len(theta_edges_out) - 1
    c_avg = np.zeros(n_bins)
    for i in range(n_bins):
        theta_start = float(theta_edges_out[i])
        theta_end = float(theta_edges_out[i + 1])
        dtheta_bin = theta_end - theta_start
        segments = identify_outlet_segments(theta_start, theta_end, v_outlet, waves, sorption)
        total = 0.0
        for seg in segments:
            seg_a = max(seg["theta_start"], theta_start)
            seg_b = min(seg["theta_end"], theta_end)
            d = seg_b - seg_a
            if d <= EPSILON_TIME:
                continue
            if seg["type"] == "constant":
                total += seg["concentration"] * d
            elif seg["type"] == "rarefaction":
                if isinstance(sorption, NonlinearSorption):
                    total += integrate_rarefaction_exact(seg["wave"], v_outlet, seg_a, seg_b, sorption)
                else:
                    c_mid = concentration_at_point(v_outlet, 0.5 * (seg_a + seg_b), waves, sorption)
                    total += c_mid * d
            elif seg["type"] == "decaying_fan":
                w = seg["wave"]
                total += integrate_fan_exact(
                    w.theta_origin, w.v_origin, v_outlet, seg_a, seg_b, sorption, c_apex=w.c_fixed
                )
        c_avg[i] = total / dtheta_bin
    return c_avg


def compute_domain_mass(
    theta: float,
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
) -> float:
    """
    Compute total mass in domain [0, v_outlet] at cumulative flow θ.

    Integrates concentration over space::

        M(θ) = ∫₀^v_outlet C_total(v, θ) dv

    Exact analytical formulas for every wave type: constant regions
    (``C_total · Δv``), RarefactionWave fan interiors and DecayingShockWave fan
    interiors (closed-form via :func:`integrate_fan_spatial_exact`).

    Parameters
    ----------
    theta : float
        Cumulative flow at which to compute domain mass [m³].
    v_outlet : float
        Outlet position (domain extent) [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : SorptionModel
        Sorption model.

    Returns
    -------
    mass : float
        Total mass in domain [mass]. Closed-form analytical to machine precision.

    See Also
    --------
    compute_cumulative_inlet_mass : Cumulative inlet mass
    compute_cumulative_outlet_mass : Cumulative outlet mass
    concentration_at_point : Point-wise concentration
    integrate_fan_spatial_exact : Closed-form fan spatial integral

    Examples
    --------
    ::

        mass = compute_domain_mass(
            theta=2500.0, v_outlet=500.0, waves=tracker.state.waves, sorption=sorption
        )
        mass >= 0.0
    """
    # Partition spatial domain into segments based on wave structure
    # We'll evaluate concentration at many points and identify constant regions

    # Collect all wave positions at time t
    wave_positions = []

    for wave in waves:
        # Use was_active_at(theta) — not is_active — so historical wave geometries
        # contribute to retrospective m_dom queries. Without this, a wave deactivated
        # by a later collision event is skipped here, which propagates as a "cin echo
        # at the outlet" bug (m_out = m_in − 0 instead of m_in − m_dom_correct).
        if not wave.was_active_at(theta):
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

        elif isinstance(wave, DecayingShockWave):
            v_pos = wave.position_at_theta(theta)
            if v_pos is not None and 0 <= v_pos <= v_outlet:
                wave_positions.append(v_pos)
            if 0 <= wave.v_origin <= v_outlet:
                wave_positions.append(wave.v_origin)

    # Add domain boundaries
    wave_positions.extend([0.0, v_outlet])

    # Sort and remove duplicates; all entries are within [0, v_outlet] by
    # construction (each append site is guarded by the bounds check).
    wave_positions = sorted(set(wave_positions))

    # Compute mass in each segment using refined integration
    total_mass = 0.0

    for i in range(len(wave_positions) - 1):
        v_start = wave_positions[i]
        v_end = wave_positions[i + 1]
        dv = v_end - v_start

        if dv < EPSILON_VELOCITY:
            continue

        # Check whether the midpoint is inside any fan-bearing wave; the fan
        # spatial integral is closed-form for both RarefactionWave and
        # DecayingShockWave (the latter via the parameterised
        # ``integrate_fan_spatial_exact``).
        #
        # Multi-fan dispatch: when multiple DSWs (or rarefactions) nominally
        # contain v_mid, the NEWER one (later ``theta_start``) wins. The
        # planned exclusion rule v_mid ∈ [V_apex_W₂, V_s_W₁] (round-2 plan
        # review) is structurally equivalent to "newest wins" given the
        # iterate-newest-first sort — verified by code-opt impl review —
        # so the explicit exclusion loop has been collapsed.
        v_mid = 0.5 * (v_start + v_end)
        raref_wave: RarefactionWave | None = None
        decaying_wave: DecayingShockWave | None = None

        dsw_candidates: list[DecayingShockWave] = []
        for wave in waves:
            if not wave.was_active_at(theta):
                continue
            if isinstance(wave, DecayingShockWave):
                v_s = wave.position_at_theta(theta)
                if v_s is None:
                    continue
                # decay_side='left': fan is upstream of V_s (v < V_s);
                # decay_side='right': fan is downstream of V_s (v > V_s).
                in_fan = (wave.decay_side == "left" and v_mid < v_s) or (wave.decay_side == "right" and v_mid > v_s)
                if in_fan and v_mid != wave.v_origin:
                    dsw_candidates.append(wave)

        if dsw_candidates:
            decaying_wave = max(dsw_candidates, key=lambda w: w.theta_start)

        if decaying_wave is None:
            raref_candidates = [
                wave
                for wave in waves
                if wave.was_active_at(theta) and isinstance(wave, RarefactionWave) and wave.contains_point(v_mid, theta)
            ]
            if raref_candidates:
                raref_wave = max(raref_candidates, key=lambda w: w.theta_start)

        if raref_wave is not None:
            mass_segment = _integrate_rarefaction_spatial_exact(raref_wave, v_start, v_end, theta, sorption)
        elif decaying_wave is not None:
            mass_segment = integrate_fan_spatial_exact(
                decaying_wave.theta_origin,
                decaying_wave.v_origin,
                v_start,
                v_end,
                theta,
                sorption,
                c_apex=decaying_wave.c_fixed,
            )
        else:
            # Constant region: c at midpoint is exact for the segment.
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
    """Exact spatial integral of a rarefaction's total concentration at fixed θ.

    Thin wrapper over :func:`integrate_fan_spatial_exact` that pulls the fan
    apex from ``raref.theta_start, raref.v_start``. For ``ConstantRetardation``
    the rarefaction degenerates to a single c value (no fan), so we use the
    wave's ``concentration_at_point`` directly.

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
    float
        Mass in the segment ``[v_start, v_end]``.
    """
    if isinstance(sorption, ConstantRetardation):
        v_mid = 0.5 * (v_start + v_end)
        c = raref.concentration_at_point(v_mid, theta) or 0.0
        c_total = sorption.total_concentration(c)
        return c_total * (v_end - v_start)

    return integrate_fan_spatial_exact(
        raref.theta_start, raref.v_start, v_start, v_end, theta, sorption, c_apex=raref.c_tail
    )


def integrate_fan_spatial_exact(
    theta_origin: float,
    v_origin: float,
    v_start: float,
    v_end: float,
    theta: float,
    sorption: SorptionModel,
    c_apex: float = 0.0,
) -> float:
    """Exact spatial integral ``∫ C_total(v, θ) dv`` for any self-similar fan.

    Decoupled from the wave object so the same closed-form math applies to
    :class:`RarefactionWave` (apex = ``theta_start, v_start``) and
    :class:`DecayingShockWave` (apex = ``theta_origin, v_origin``).

    In (V, θ) the self-similar fan satisfies ``R(C) = (θ - θ_origin)/(v - v_origin)``;
    define ``kappa = θ - θ_origin`` and ``u = v - v_origin``. The dissolved and
    sorbed contributions reduce to power-law forms in ``u`` that admit closed
    forms via incomplete beta functions (Freundlich) or elementary sqrt
    operations (Langmuir).

    Parameters
    ----------
    theta_origin, v_origin : float
        Cumulative flow and position at the fan's apex [m³].
    v_start, v_end : float
        Integration range in v [m³].
    theta : float
        Cumulative flow at which to evaluate [m³].
    sorption : SorptionModel
        Sorption model (``FreundlichSorption`` or ``LangmuirSorption``).
    c_apex : float, optional
        Concentration on the constant side at the fan apex (typically the
        parent rarefaction's ``c_tail`` or the DSW's ``c_fixed`` for
        ``decay_side='left'``). For ``c_apex > 0`` the fan formula is
        unphysical for ``u < u_tail = kappa / R(c_apex)``; the integration
        is split into a constant-C_total(c_apex) region for
        ``u ∈ [u_start, u_tail]`` plus the fan integral for
        ``u ∈ [u_tail, u_end]``. Default 0.0 preserves the c=0 apex
        behavior for canonical c_R=0 rarefactions.

    Returns
    -------
    float
        Mass in the segment ``[v_start, v_end]``.

    Raises
    ------
    TypeError
        If the sorption model does not support exact spatial integration.
    """
    if theta <= theta_origin:
        return 0.0

    kappa = theta - theta_origin
    u_start = v_start - v_origin
    u_end = v_end - v_origin

    # The fan only exists for v > v_origin; clip u_start at 0 (the apex
    # contributes nothing to the integral since c(v_origin)=0 for n>1 and
    # the Beta-function form handles the lower-bound singularity for n<1).
    # If the whole segment is upstream of the apex, return 0.
    if u_end <= 0:
        return 0.0
    if u_start < 0:
        u_start = 0.0

    # Split off the constant-c_apex region near the apex for c_apex > 0.
    # The fan formula is only valid for u ≥ u_tail = kappa / R(c_apex);
    # below u_tail, c is clamped to c_apex (the parent's tail / DSW's fixed
    # concentration). Spatial counterpart of the temporal θ_tail clamp in
    # _integrate_fan_exact_freundlich.
    constant_contrib = 0.0
    if c_apex > 0.0:
        u_tail = kappa / float(sorption.retardation(c_apex))
        if u_start < u_tail:
            c_total_apex = float(sorption.total_concentration(c_apex))
            constant_contrib = c_total_apex * (min(u_end, u_tail) - u_start)
            u_start = u_tail
        if u_end <= u_start:
            return constant_contrib

    if isinstance(sorption, LangmuirSorption):
        return constant_contrib + _integrate_rarefaction_spatial_langmuir(sorption, kappa, u_start, u_end)

    if isinstance(sorption, FreundlichSorption):
        return constant_contrib + _integrate_rarefaction_spatial_freundlich(sorption, kappa, u_start, u_end)

    msg = f"Exact spatial fan integration not supported for {type(sorption).__name__}"
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
        Exact mass in segment [mass].

    Notes
    -----
    With u = v - v_origin, kappa = flow*(t - t_origin):
        C(u) = sqrt(a_coeff*u/(kappa-u)) - K_L
        C_total = C + (rho_b/n_por) * s_max * C/(K_L + C)

    Since K_L + C = sqrt(a_coeff*u/(kappa-u)):
        C/(K_L+C) = 1 - K_L*sqrt((kappa-u)/(a_coeff*u))

    The Langmuir fan has finite extent in u: C(u) ≥ 0 only for
    u ≥ u_zero = K_L²·κ/(a_coeff + K_L²). The integrand is unphysical
    (c<0, giving a negative C_total) for u < u_zero, so BOTH bounds are
    clamped at u_zero before evaluating the antiderivative. Spatial
    counterpart of the upper-bound θ_zero clamp in
    :func:`_integrate_fan_exact_langmuir`: there the fan has c=0 for
    θ ≥ θ_zero; here c=0 for u ≤ u_zero.

    The integral simplifies to:
        integral C_total du = -2*sqrt(a_coeff)*[sqrt(u*(kappa-u))]_start^end
                              + (rho_b*s_max/n_por - K_L)*(u_end - u_start)
    """
    a_coeff = sorption.a_coeff
    k_l = sorption.k_l
    sorbed_max = sorption.bulk_density * sorption.s_max / sorption.porosity

    u_zero = k_l * k_l * kappa / (a_coeff + k_l * k_l)
    # Clamp into the physical fan range [u_zero, kappa]: below u_zero, c<0;
    # at kappa, c diverges (NaN under sqrt for u_end > kappa).
    u_start = min(max(u_start, u_zero), kappa)
    u_end = min(max(u_end, u_zero), kappa)

    term_sqrt_end = np.sqrt(u_end * (kappa - u_end))
    term_sqrt_start = np.sqrt(u_start * (kappa - u_start))

    return -2.0 * np.sqrt(a_coeff) * (term_sqrt_end - term_sqrt_start) + (sorbed_max - k_l) * (u_end - u_start)


def compute_cumulative_inlet_mass(
    theta: float,
    cin: npt.ArrayLike,
    theta_edges: npt.ArrayLike,
) -> float:
    """Cumulative inlet mass entering the domain from θ=0 to ``theta``.

    In cumulative-flow coordinates ``M_in(θ) = ∫₀^θ cin(τ) dτ``; for
    piecewise-constant ``cin`` this is exact under summation over θ-bin
    widths.

    Parameters
    ----------
    theta : float
        Cumulative flow up to which to integrate [m³].
    cin : array-like
        Inlet concentration per θ-bin [mass/volume].
    theta_edges : array-like
        θ bin edges [m³], length ``len(cin) + 1``.

    Returns
    -------
    mass_in : float
        Cumulative inlet mass [mass].

    Examples
    --------
    ::

        mass_in = compute_cumulative_inlet_mass(
            theta=5000.0, cin=cin, theta_edges=theta_edges
        )
        mass_in >= 0.0
    """
    te = np.asarray(theta_edges, dtype=float)
    widths = np.clip(theta - te[:-1], 0.0, np.diff(te))
    return float(np.sum(np.asarray(cin, dtype=float) * widths))


def compute_cumulative_outlet_mass(
    theta: float,
    v_outlet: float,
    waves: Sequence[Wave],
    sorption: SorptionModel,
    *,
    cin: npt.ArrayLike,
    theta_edges: npt.NDArray[np.floating],
) -> float:
    """Cumulative mass exiting through the outlet from θ=0 to ``theta``.

    Computed analytically via the conservation-law identity::

        m_out(θ) = m_in(θ) − m_dom(θ)

    derived from integrating the PDE ``∂_θ C_T + ∂_V c = 0`` over the spatial
    domain ``[0, v_outlet]`` (Bear & Cheng 2010, §3.5; standard form for
    advection with sorption). This sidesteps the multi-fan dispatch problem
    that the outlet-segment integration faces when several DSWs cover
    v_outlet simultaneously — every term on the right is purely spatial or a
    closed-form inlet sum, no ownership priority needed.

    Parameters
    ----------
    theta : float
        Cumulative flow up to which to integrate [m³].
    v_outlet : float
        Outlet position [m³].
    waves : list of Wave
        All waves in the simulation.
    sorption : SorptionModel
        Sorption model.
    cin : array-like (kw-only)
        Inlet concentration per θ-bin [mass/volume].
    theta_edges : ndarray (kw-only)
        θ bin edges [m³], length ``len(cin) + 1``.

    Returns
    -------
    mass_out : float
        Cumulative outlet mass [mass].

    Examples
    --------
    ::

        mass_out = compute_cumulative_outlet_mass(
            theta=5000.0,
            v_outlet=500.0,
            waves=tracker.state.waves,
            sorption=sorption,
            cin=cin,
            theta_edges=tracker.state.theta_edges,
        )
        mass_out >= 0.0
    """
    if theta <= 0.0:
        return 0.0
    m_in = compute_cumulative_inlet_mass(theta=theta, cin=cin, theta_edges=theta_edges)
    m_dom = compute_domain_mass(theta=theta, v_outlet=v_outlet, waves=waves, sorption=sorption)
    return m_in - m_dom


def compute_total_outlet_mass(
    v_outlet: float,
    sorption: SorptionModel,
    *,
    cin: npt.ArrayLike,
    theta_edges: npt.NDArray[np.floating],
) -> tuple[float, float]:
    """Asymptotic total outlet mass via the conservation-law identity.

    Computed as ``m_in_total − m_dom_asymptotic`` where ``m_dom_asymptotic``
    is the aquifer's steady-state domain mass: ``C_T(c_∞) · v_outlet``, with
    ``c_∞ = cin[-1]`` (the final/sustained inlet boundary value). This is
    the asymptotic limit of ``m_in(θ) − m_dom(θ)`` as θ → ∞:

    - For ``c_∞ = 0`` (canonical c_R=0 pulse): m_dom_asymptotic = 0 and
      ``m_out_total = m_in_total`` — every injected mass unit eventually exits.
    - For ``c_∞ > 0`` (sustained ambient): m_dom_asymptotic = C_T(c_∞) · V_outlet
      stays in the domain at steady state; the rest exits.

    Sidesteps the multi-fan dispatch problem entirely — the integration is
    purely closed-form arithmetic over (cin, theta_edges) plus one
    ``sorption.total_concentration`` evaluation. The wave list is not
    needed.

    Parameters
    ----------
    v_outlet : float
        Outlet position [m³].
    sorption : SorptionModel
        Sorption model — used only for ``C_T(c_∞)``.
    cin : array-like (kw-only)
        Inlet concentration per θ-bin [mass/volume].
    theta_edges : ndarray (kw-only)
        θ bin edges [m³], length ``len(cin) + 1``.

    Returns
    -------
    total_mass_out : float
        Asymptotic outlet mass [mass]. Equals ``m_in_total`` for ``cin[-1]=0``.
    theta_integration_end : float
        ``theta_edges[-1]`` (the simulation end θ). Caller translates to
        user-facing time via ``FrontTrackerState.t_at_theta`` if needed.

    See Also
    --------
    compute_cumulative_outlet_mass : Cumulative outlet mass up to finite θ
    compute_domain_mass : Spatial integral of C_total in the aquifer
    """
    cin_arr = np.asarray(cin, dtype=float)
    te = np.asarray(theta_edges, dtype=float)
    m_in_total = float(np.sum(cin_arr * np.diff(te)))
    c_inf = float(cin_arr[-1]) if cin_arr.size > 0 else 0.0
    m_dom_asymptotic = float(sorption.total_concentration(c_inf)) * v_outlet
    return m_in_total - m_dom_asymptotic, float(te[-1])
