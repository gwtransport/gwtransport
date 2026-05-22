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
compute_total_outlet_mass(v_outlet, sorption, *, cin, theta_edges) -> float

Outlet-mass functions use the PDE conservation identity
``m_out(θ) = m_in(θ) − m_dom(θ)`` (Bear & Cheng 2010, Ch. 3: mass
conservation for transport with sorption). ``m_dom`` honors historical
wave activity via ``wave.was_active_at(theta)`` so retrospective queries
at θ before a collision event correctly attribute c at v_outlet.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import warnings
from collections.abc import Sequence
from operator import itemgetter

import numpy as np
import numpy.typing as npt

from gwtransport.fronttracking.events import find_outlet_crossing
from gwtransport.fronttracking.math import (
    ConstantRetardation,
    NonlinearSorption,
    SorptionModel,
)
from gwtransport.fronttracking.waves import CharacteristicWave, DecayingShockWave, RarefactionWave, ShockWave, Wave

# Numerical tolerance constants
EPSILON_VELOCITY = 1e-15  # Tolerance for checking if velocity is effectively zero
EPSILON_TIME = 1e-15  # Tolerance for negligible time segments


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
    # Multi-DSW dispatch: when several active DSW fans contain v, the newest
    # (largest ``theta_start``) wins — same rule as ``compute_domain_mass`` at
    # line ~1057. Iterating chronologically and short-circuiting picks the
    # older DSW's fan value, which is wrong for overlap regions. The fan
    # predicate mirrors the in-fan check used downstream.
    dsw_in_fan: list[DecayingShockWave] = []
    for wave in waves:
        if isinstance(wave, DecayingShockWave) and wave.was_active_at(theta):
            v_s = wave.position_at_theta(theta)
            if v_s is None:
                continue
            in_fan = (wave.decay_side == "left" and v < v_s) or (wave.decay_side == "right" and v > v_s)
            if in_fan and v != wave.v_origin:
                dsw_in_fan.append(wave)

    if dsw_in_fan:
        newest = max(dsw_in_fan, key=lambda w: w.theta_start)
        c = newest.concentration_at_point(v, theta)
        if c is not None:
            return c

    # Fallback: no DSW fan contains v — handle shock-face (v ≈ V_s) and
    # fixed-side (v on the c_fixed side) cases. Chronological iteration is
    # fine here because (a) shock-face is rare and unique by v ≈ V_s, and
    # (b) all DSWs in a typical multi-pulse share the same c_fixed baseline.
    for wave in waves:
        if isinstance(wave, DecayingShockWave) and wave.was_active_at(theta):
            c = wave.concentration_at_point(v, theta)
            if c is not None:
                return c

    # Multi-rarefaction dispatch: same shape as DSWs. RarefactionWave returns
    # None outside its fan, so collecting non-None candidates is equivalent to
    # an in-fan check. Newest wins.
    raref_candidates: list[tuple[float, RarefactionWave]] = []
    for wave in waves:
        if isinstance(wave, RarefactionWave) and wave.was_active_at(theta):
            c = wave.concentration_at_point(v, theta)
            if c is not None:
                raref_candidates.append((c, wave))

    if raref_candidates:
        return max(raref_candidates, key=lambda cw: cw[1].theta_start)[0]

    latest_wave_theta = -np.inf
    latest_wave_c = None
    # Stacked-shock geometry: for v right of multiple shocks, c at v is c_right
    # of the shock CLOSEST to v from the left (largest V_shock with V_shock < v).
    # ``theta_start`` would mis-rank stacked shocks (youngest = innermost ≠
    # closest-to-v), so we track V_shock directly.
    # Obstruction check: a shock's c_R applies at v only when no other active
    # wave (rarefaction tail/head, DSW V_s, other shock) sits between V_shock
    # and v. Otherwise c_R is the c in the immediate-right-of-shock zone, not
    # at v.
    rightmost_passed_v_shock = -np.inf
    rightmost_passed_c_right: float | None = None

    def _intervening_wave_between(v_a: float, v_b: float) -> bool:
        """Return True if any other active wave has a boundary V in (v_a, v_b)."""
        for other in waves:
            if not other.was_active_at(theta):
                continue
            if isinstance(other, RarefactionWave):
                v_t = other.tail_position_at_theta(theta)
                v_h = other.head_position_at_theta(theta)
                if v_t is not None and v_a < v_t < v_b:
                    return True
                if v_h is not None and v_a < v_h < v_b:
                    return True
            else:
                v_o = other.position_at_theta(theta)
                if v_o is not None and v_a < v_o < v_b:
                    return True
        return False

    for wave in waves:
        if isinstance(wave, ShockWave) and wave.was_active_at(theta):
            v_shock = wave.position_at_theta(theta)
            if v_shock is not None:
                tol = 1e-15

                if abs(v - v_shock) < tol:
                    return 0.5 * (wave.c_left + wave.c_right)

                if abs(wave.speed) > EPSILON_VELOCITY:
                    theta_cross = wave.theta_start + (v - wave.v_start) / wave.speed

                    if theta_cross <= theta:
                        if theta_cross > latest_wave_theta:
                            latest_wave_theta = theta_cross
                            latest_wave_c = wave.c_left
                    elif v > v_shock > rightmost_passed_v_shock and not _intervening_wave_between(v_shock, v):
                        rightmost_passed_v_shock = v_shock
                        rightmost_passed_c_right = wave.c_right

    for wave in waves:
        if isinstance(wave, RarefactionWave) and wave.was_active_at(theta):
            v_tail = wave.tail_position_at_theta(theta)
            if v_tail is not None and v_tail > v + 1e-15:
                tail_speed = wave.tail_speed()
                if tail_speed > EPSILON_VELOCITY:
                    theta_pass = wave.theta_start + (v - wave.v_start) / tail_speed
                    if theta_pass <= theta and theta_pass > latest_wave_theta:
                        latest_wave_theta = theta_pass
                        latest_wave_c = wave.c_tail

    # Priority: latest_wave_c is set by the IF-shock branch (theta_cross <=
    # theta -> c_L of closest-passing shock) or by the rarefaction-tail loop
    # (rarefaction tail downstream of v passed v at theta_pass -> c_tail).
    # rightmost_passed_c_right is set by the elif-shock branch (v > V_shock,
    # shock upstream of v, c_R of closest-from-left shock). The two represent
    # different geometric truths: latest_wave_c is "most recent event AT v",
    # rightmost_passed_c_right is "c just past the upstream shock". When a
    # rarefaction tail is downstream of the shock AND right of v, that
    # rarefaction's c_tail wins (the tail's passage at v is more recent than
    # the shock's contribution at v, geometrically). Prefer latest_wave_c.
    if latest_wave_c is not None:
        return latest_wave_c
    if rightmost_passed_c_right is not None:
        return rightmost_passed_c_right

    latest_c = 0.0
    latest_theta = -np.inf

    for wave in waves:
        if isinstance(wave, CharacteristicWave) and wave.was_active_at(theta):
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
        # Retrospective filter: ``identify_outlet_segments`` is called over
        # arbitrary [theta_start, theta_end] windows (e.g., plotting after the
        # simulation ends). ``is_active`` is the wave's *current* (end-of-sim)
        # state and skips waves that legitimately crossed v_outlet during the
        # window but were later deactivated by a collision. Skip only if the
        # wave's lifetime ended before the window started.
        if wave.theta_deactivation <= theta_start:
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

    # Handle case where we start inside a rarefaction or decaying-shock fan.
    # Multi-fan overlap: pick the newest (largest ``theta_start``) — matches
    # ``concentration_at_point`` and ``compute_domain_mass`` dispatch.
    if active_rarefactions_at_start:
        raref = max(active_rarefactions_at_start, key=lambda w: w.theta_start)

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
        # segment. ``concentration_at_point`` lets active rarefactions "win"
        # over a behind-shock c_left; the segment list must reflect the same
        # convention to avoid double-counting.
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
        Sorption model (any NonlinearSorption subclass).

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
        Sorption model (any NonlinearSorption subclass).
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
    # Every NonlinearSorption uses one universal IBP antiderivative, which evaluates the fan
    # kernel k = R·c − C_T at the segment endpoints via c_and_total_from_retardation.
    if isinstance(sorption, NonlinearSorption):
        return _integrate_fan_exact_universal(
            theta_origin, v_origin, v_outlet, theta_start, theta_end, sorption, c_apex
        )

    msg = f"Exact fan integration not supported for {type(sorption).__name__}"
    raise TypeError(msg)


def _integrate_fan_exact_universal(
    theta_origin: float,
    v_origin: float,
    v_outlet: float,
    theta_start: float,
    theta_end: float,
    sorption: NonlinearSorption,
    c_apex: float = 0.0,
) -> float:
    r"""Exact θ-integral ``∫ c(θ) dθ`` via the universal IBP antiderivative.

    For any ``NonlinearSorption`` (with ``R = dC_T/dC``), integration by
    parts on the self-similar fan ``R(c(θ)) = (θ − θ_origin)/Δv`` gives the
    closed-form antiderivative

    .. math::
        F(\\theta) = c(\\theta)\\,(\\theta - \\theta_{\\rm origin})
            - \\Delta v \\cdot C_T(c(\\theta)).

    The derivation uses ``∫ c\\,d\\theta = c·(\\theta-\\theta_0) − ∫(\\theta-\\theta_0)·dc
    = c·(\\theta-\\theta_0) − \\Delta v · ∫ R(c)\\,dc = c·(\\theta-\\theta_0) − \\Delta v · C_T(c)``,
    where the last equality is the definition of ``C_T`` as the antiderivative
    of ``R`` (``R = dC_T/dC``).

    This formula is exact for any sorption (Brooks-Corey, van Genuchten-Mualem,
    Freundlich, Langmuir). The only sorption-specific call is
    ``sorption.concentration_from_retardation`` at the two endpoints — for
    Brooks-Corey this is closed form; for van Genuchten-Mualem it is one
    ``brentq`` call per endpoint. No quadrature, no integration loop.

    Convergence at θ → ∞ for ``c_apex = 0``: for any monotone sorption with
    ``R(0) = ∞`` (BC, vG, Freundlich n > 1), ``c(∞) = 0`` and ``c·θ → 0``
    faster than ``Δv·C_T → 0`` (verified termwise from the closed-form
    asymptotic ``c ~ R^{-α}`` for some ``α > 1``), so ``F(∞) = 0``.

    For ``c_apex > 0`` the fan formula extrapolates to ``c < c_apex`` past
    ``θ_tail = θ_origin + Δv·R(c_apex)``; clamp the fan portion at
    ``θ_tail`` and add ``c_apex·(θ_end − θ_tail)`` for any ``θ_end > θ_tail``.
    """
    delta_v = v_outlet - v_origin
    if delta_v <= 0 or theta_end <= theta_start:
        return 0.0

    if c_apex > 0.0:
        theta_tail = theta_origin + delta_v * float(sorption.retardation(c_apex))
        theta_end_fan = min(theta_end, theta_tail)
    else:
        theta_tail = float("inf")
        theta_end_fan = theta_end

    # For a c_apex=0 fan the upper bound may be +∞. The integral converges only when
    # c → 0 as R → ∞ (so base·c → 0). Freundlich n<1 (c → ∞ as R → ∞) diverges — reject it
    # explicitly. (DecayingShockWave.mass_after_outlet_arrival returns 0 for the n<1 mirror
    # before reaching here, so this is a defensive guard for direct callers.)
    if theta_end_fan == float("inf") and not sorption.fan_converges_at_infinity():
        msg = "Fan integral diverges at θ=+∞ for this sorption (e.g. Freundlich n<1); pass a finite theta_end"
        raise ValueError(msg)

    def antiderivative(theta: float) -> float:
        if theta == float("inf"):
            # F(∞) = 0 for any sorption with c → 0 as R → ∞ (guarded above for divergent cases).
            return 0.0
        base = (theta - theta_origin) / delta_v
        if base <= 0.0:
            return 0.0
        c, ct = sorption.c_and_total_from_retardation(base)
        return c * (theta - theta_origin) - delta_v * ct

    fan_integral = antiderivative(theta_end_fan) - antiderivative(theta_start)
    constant_contrib = c_apex * max(theta_end - theta_tail, 0.0) if c_apex > 0.0 else 0.0
    return fan_integral + constant_contrib


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
    outlet-side fan dispatch. Otherwise falls back to outlet-segment
    integration (correct for canonical single-DSW cases; may miscount
    multi-DSW or n<1 mirror geometries).

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
        # FP-noise clamp: m_in − m_dom subtracts nearly-equal large numbers,
        # leaving ~1 ULP residuals on either sign. Clamp those to 0.
        max_c = float(np.max(np.abs(result))) if result.size else 0.0
        eps_clamp = 1e-12 * max(max_c, 1.0)
        result = np.where(np.abs(result) < eps_clamp, 0.0, result)
        # Large-negative diagnostic: residuals beyond the FP-noise band signal
        # a real conservation-form violation, most commonly the post-inlet
        # artifact — output edges exceed ``theta_edges_inlet[-1]``, m_in
        # caps at the last injected mass while the simulator's wave list
        # continues to evolve, producing FP-cancellation residuals from
        # inconsistent θ ranges. Surface as a UserWarning; clamp to 0 to
        # preserve the ``cout >= 0`` API contract.
        if result.size:
            min_val = float(np.min(result))
            if min_val < -eps_clamp:
                msg = (
                    f"compute_bin_averaged_concentration_exact produced concentrations "
                    f"as negative as {min_val:.3e} (clamp threshold -{eps_clamp:.3e}); "
                    f"likely caused by output θ-bin edges exceeding "
                    f"theta_edges_inlet[-1]={float(np.asarray(theta_edges_inlet)[-1]):.3f}, "
                    "putting the inlet integral and the wave list on inconsistent θ ranges. "
                    "Extend cin with trailing zeros to cover the output range, "
                    "or restrict output bins to within the inlet window."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
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
    # Partition spatial domain into segments at active wave positions.
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
        # contain v_mid, the newer one (later ``theta_start``) wins —
        # geometrically equivalent to the exclusion rule
        # ``v_mid ∈ [V_apex_W₂, V_s_W₁]`` for simulator-produced layouts.
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
        Sorption model (any NonlinearSorption subclass).
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
    # _integrate_fan_exact_universal.
    constant_contrib = 0.0
    if c_apex > 0.0:
        u_tail = kappa / float(sorption.retardation(c_apex))
        if u_start < u_tail:
            c_total_apex = float(sorption.total_concentration(c_apex))
            constant_contrib = c_total_apex * (min(u_end, u_tail) - u_start)
            u_start = u_tail
        if u_end <= u_start:
            return constant_contrib

    # One universal IBP antiderivative for every NonlinearSorption (see the rationale in
    # ``integrate_fan_exact``, the temporal counterpart).
    if isinstance(sorption, NonlinearSorption):
        return constant_contrib + _integrate_rarefaction_spatial_universal(sorption, kappa, u_start, u_end)

    msg = f"Exact spatial fan integration not supported for {type(sorption).__name__}"
    raise TypeError(msg)


def _integrate_rarefaction_spatial_universal(
    sorption: NonlinearSorption,
    kappa: float,
    u_start: float,
    u_end: float,
) -> float:
    r"""Exact spatial integral ``∫ C_T(c(u)) du`` via the universal IBP antiderivative.

    For any ``NonlinearSorption`` with ``R = dC_T/dC``, integration by parts
    on the self-similar fan ``R(c(u)) = κ/u`` gives the closed-form
    antiderivative

    .. math::
        G(u) = C_T(c(u))\\cdot u - \\kappa\\cdot c(u).

    Derivation: ``∫ C_T\\,du = C_T·u − ∫ u\\,dC_T = C_T·u − ∫ (κ/R)·R\\,dc =
    C_T·u − κ·c`` (the second equality uses ``u = κ/R`` and the third uses
    ``dC_T = R\\,dc``).

    Sorption-specific calls limited to ``concentration_from_retardation`` and
    ``total_concentration`` at the two endpoints. For Brooks-Corey both are
    closed form; for van Genuchten-Mualem ``concentration_from_retardation`` is
    one ``brentq`` per endpoint and ``total_concentration`` is also one
    ``brentq`` per endpoint (chained internally). No quadrature.

    At the apex (``u → 0``, ``R → ∞``) ``c → 0`` and ``C_T → 0`` so ``G(0) = 0``;
    a segment whose lower bound is at (or below, for a bounded-``R`` sorption like
    Langmuir where ``c = 0`` for ``u`` below the fan tail) the apex contributes
    ``G(u_end) − 0``.
    """
    if u_end <= 0.0 or u_end <= u_start or kappa <= 0.0:
        return 0.0
    if u_start <= 0.0:
        g_start = 0.0  # G(0) = C_T(0)·0 − κ·0 = 0 (apex)
    else:
        c_start, ct_start = sorption.c_and_total_from_retardation(kappa / u_start)
        g_start = ct_start * u_start - kappa * c_start
    c_end, ct_end = sorption.c_and_total_from_retardation(kappa / u_end)
    g_end = ct_end * u_end - kappa * c_end
    return g_end - g_start


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
    domain ``[0, v_outlet]`` (Bear & Cheng 2010, Ch. 3: mass conservation
    for advection with sorption). This sidesteps the multi-fan dispatch problem
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
) -> float:
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
    float
        Asymptotic outlet mass [mass]. Equals ``m_in_total`` for ``cin[-1]=0``.

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
    return m_in_total - m_dom_asymptotic
