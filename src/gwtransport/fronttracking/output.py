"""Concentration extraction from front-tracking solutions (V, θ coordinates).

Every public function in this module takes θ (cumulative flow, m³). Callers
translate user-facing time t → θ at the API boundary via
``FrontTrackerState.theta_at_t``.

Functions
---------
::

    concentration_at_point(v, theta, waves, sorption)
    compute_breakthrough_curve(theta_array, v_outlet, waves, sorption)
    compute_bin_averaged_concentration_exact(theta_bin_edges, v_outlet, waves, sorption, *, cin=None, theta_edges_inlet=None)
    compute_domain_mass(theta, v_outlet, waves, sorption)
    compute_cumulative_inlet_mass(theta, cin, theta_edges)
    compute_cumulative_outlet_mass(theta, v_outlet, waves, sorption, *, cin, theta_edges)
    compute_total_outlet_mass(*, cin, theta_edges) -> float
    identify_outlet_segments(theta_start, theta_end, v_outlet, waves, ...)
    integrate_rarefaction_exact(raref, v_outlet, theta_start, theta_end, sorption)
    integrate_fan_exact(theta_origin, v_origin, v_outlet, theta_start, theta_end, sorption, c_apex=0.0)
    integrate_fan_spatial_exact(theta_origin, v_origin, v_start, v_end, theta, sorption, c_apex=0.0)

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
from itertools import pairwise
from operator import itemgetter

import numpy as np
import numpy.typing as npt

from gwtransport.fronttracking.events import find_outlet_crossing
from gwtransport.fronttracking.interactions import Face, iter_faces
from gwtransport.fronttracking.math import (
    NonlinearSorption,
    SorptionModel,
)
from gwtransport.fronttracking.waves import (
    CharacteristicWave,
    DecayingShockWave,
    Feeder,
    RarefactionWave,
    ShockWave,
    Wave,
)

# Numerical tolerance constants
EPSILON_VELOCITY = 1e-15  # Tolerance for checking if velocity is effectively zero
EPSILON_TIME = 1e-15  # Tolerance for negligible time segments
EPSILON_VOLUME = 1e-15  # Tolerance for negligible spatial-segment widths Δv [m³]
EPSILON_POSITION = 1e-15  # Tolerance for shock-face proximity in position v [m³]
# Multiplier on the eps·max(|m_in|,|m_dom|)/Δθ cancellation scale below which a conservation-form
# residual is numerical zero. The m_dom fan-integral sum accumulates ~few·10² ULP over a
# multi-pulse record (measured max ratio ~460); 65536 clears that with wide margin while staying
# ~7 orders below any real breakthrough concentration (O(cin)).
FP_CANCELLATION_CLAMP = 65536.0


def _reader_faces(waves: Sequence[Wave], theta: float) -> list[tuple[float, Face]]:
    """``(position, face)`` for every active wave face at ``theta`` (reader sweep basis).

    Fan boundary lines are included: they mark where a downstream-opening fan ends and its
    plateau begins (a ``decay_side='right'`` decaying/doubly-fed shock has its fan on the
    downstream side, so its head boundary must partition the domain — the feeder clamp alone
    only reaches the apex-side plateau).
    """
    out: list[tuple[float, Face]] = []
    for wave in waves:
        if not wave.was_active_at(theta):
            continue
        for face in iter_faces(wave, theta, include_boundaries=True):
            pos = face.position(theta)
            if pos is not None:
                out.append((pos, face))
    return out


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
    **Nearest-downstream-face sweep.** ``C(v, θ)`` is the *left* (upstream) state of the
    nearest face strictly downstream of ``v`` among the waves active at ``θ``; if no face
    is downstream, the *right* state of the outermost face; ``0`` in a virgin domain. A
    face's feeder is a constant or a bounded self-similar fan (clamped to its extent, so the
    plateau beyond a fan reads correctly). Because every face crossing fires a solver event,
    the front list is interaction-consistent and each region has a single owner — no
    ownership heuristic is needed. This is exact for a single front (one owner everywhere)
    and for genuinely interacting multi-front inputs alike.
    """
    faces = _reader_faces(waves, theta)
    # Nearest face at or downstream of v (within FP): its left (upstream) feeder gives the
    # state just behind it, which is the state at v. A face strictly upstream of v does not
    # describe v. If nothing is at/downstream, the outermost face's right (downstream) feeder
    # holds — the state ahead of every front.
    downstream = [(p, f) for (p, f) in faces if p >= v - EPSILON_POSITION]
    if downstream:
        pos, face = min(downstream, key=itemgetter(0))
        # Exactly on a shock/fan face: return the average of the two sides (the infinitesimally
        # thin discontinuity convention). A contact (characteristic) carries its value on the
        # upstream side, so it returns that (the behind value) at its own position.
        if abs(pos - v) < EPSILON_POSITION and face.role != "contact":
            return 0.5 * (face.left.value(v, theta) + face.right.value(v, theta))
        return face.left.value(v, theta)
    if faces:
        _p, face = max(faces, key=itemgetter(0))
        return face.right.value(v, theta)
    return 0.0


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
    .. disable_try_examples

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
            ``'decaying_fan'`` is owned by a :class:`~gwtransport.fronttracking.waves.DecayingShockWave` after
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
    both :class:`~gwtransport.fronttracking.waves.RarefactionWave` (apex = ``theta_start, v_start``) and
    :class:`~gwtransport.fronttracking.waves.DecayingShockWave` (apex = ``theta_origin, v_origin``).

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
    # explicitly via the fan_converges_at_infinity guard.
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
    dtheta_out = np.diff(theta_edges_out)

    if np.any(dtheta_out <= 0):
        bad = int(np.argmin(dtheta_out))
        msg = (
            f"Invalid θ-bin: theta_bin_edges[{bad}]={theta_edges_out[bad]} >= "
            f"theta_bin_edges[{bad + 1}]={theta_edges_out[bad + 1]}"
        )
        raise ValueError(msg)

    if cin is not None and theta_edges_inlet is not None:
        # Conservation form: c_avg = Δm_out/Δθ where m_out = m_in − m_dom.
        # m_in(θ) = ∫₀^θ cin dτ is the piecewise-LINEAR (in θ) integral of the piecewise-constant
        # cin, so evaluate it at every output edge in O(N+M) from the inlet-bin cumulative sums
        # plus the partial bin containing θ — instead of the dense N_out×M_in clip-and-matmul
        # (≈245× slower, a 128 MB temporary at N=M=4000). Edges below te_in[0] contribute 0
        # (nothing is injected before the record starts, even if it starts mid-window); edges at
        # or past te_in[-1] saturate at the total. This mirrors ``compute_cumulative_inlet_mass``'s
        # clip exactly. m_dom stays a per-θ spatial geometry loop.
        te_in = np.asarray(theta_edges_inlet, dtype=float)
        cin_arr = np.asarray(cin, dtype=float)
        cum_in = np.concatenate([[0.0], np.cumsum(cin_arr * np.diff(te_in))])
        idx = np.clip(np.searchsorted(te_in, theta_edges_out, side="right") - 1, 0, len(cin_arr) - 1)
        m_in_at_edges = cum_in[idx] + cin_arr[idx] * (theta_edges_out - te_in[idx])
        m_in_at_edges = np.where(theta_edges_out < te_in[0], 0.0, m_in_at_edges)
        m_in_at_edges = np.where(theta_edges_out >= te_in[-1], cum_in[-1], m_in_at_edges)
        m_dom_at_edges = np.array([
            compute_domain_mass(theta=float(theta_e), v_outlet=v_outlet, waves=waves, sorption=sorption)
            for theta_e in theta_edges_out
        ])
        # ``compute_cumulative_outlet_mass`` short-circuits to 0 for θ ≤ 0;
        # replicate that clamp so non-positive output edges contribute no mass.
        m_out_at_edges = np.where(theta_edges_out <= 0.0, 0.0, m_in_at_edges - m_dom_at_edges)
        result = np.diff(m_out_at_edges) / dtheta_out
        # FP-noise clamp scaled to the m_in − m_dom CANCELLATION magnitude: each cumulative-mass
        # edge carries ~eps·max(|m_in|,|m_dom|) rounding (the m_dom fan-integral sum accumulates
        # several hundred ULP over a multi-pulse record), amplified by 1/Δθ. The former
        # 1e-12·max(cout) band keyed off the OUTPUT concentration, which collapses to ~0 before
        # breakthrough — far too tight, so ~1e-11 cancellation dust tripped the diagnostic on
        # fully in-range inputs. Residuals within this band are numerical zero: clamp and stay
        # silent.
        mass_scale = np.maximum(np.abs(m_in_at_edges), np.abs(m_dom_at_edges))
        # Band = the larger of the FP-cancellation floor and a 1e-6 relative transient floor.
        # The latter absorbs the benign ``c_min``-floor artifact at a fan-entry (a decaying
        # side born at ``c ≈ 1e-12`` rather than exactly 0 perturbs the near-apex fan integral
        # for one θ-sample); a genuine over-count is O(a pulse's mass), far above the band.
        eps_band = (
            max(FP_CANCELLATION_CLAMP * np.finfo(float).eps, 1e-6)
            * np.maximum(np.maximum(mass_scale[:-1], mass_scale[1:]), 1.0)
            / dtheta_out
        )
        result = np.where(np.abs(result) < eps_band, 0.0, result)
        # A residual MORE negative than the FP band is a genuine conservation-form violation with two
        # possible drivers, and BOTH can be present at once. Attribute per offending bin, not off the
        # single last edge: a bin whose right edge passes ``theta_edges_inlet[-1]`` sees m_in saturate
        # while the wave list keeps evolving (out-of-window); a fully in-window offending bin is a real
        # wave-model over-count (e.g. overlapping non-interacting waves from an oscillating inlet, see
        # ``compute_domain_mass`` Notes). Report every cause that actually produced a negative bin
        # instead of the previously unconditional — and often factually false — "edges exceed range"
        # message. Clamp to 0 to preserve the ``cout >= 0`` API contract either way.
        neg_mask = result < -eps_band
        if np.any(neg_mask):
            worst = float(np.min(result))
            te_in_last = float(te_in[-1])
            out_of_window = neg_mask & (theta_edges_out[1:] > te_in_last)
            causes = []
            if np.any(out_of_window):
                causes.append(
                    f"output θ-bin edges exceeding theta_edges_inlet[-1]={te_in_last:.3f} (m_in "
                    "saturates at the last injected mass while the wave list keeps evolving — extend "
                    "cin with trailing zeros to cover the output range, or restrict output bins to the "
                    "inlet window)"
                )
            if np.any(neg_mask & ~out_of_window):
                causes.append(
                    "the domain-mass integral growing faster than the inlet-mass integral within the "
                    "inlet window (an m_dom over-count — e.g. overlapping non-interacting waves from a "
                    "continuous/oscillating inlet — or a wave-list / cin inconsistency)"
                )
            warnings.warn(
                f"compute_bin_averaged_concentration_exact produced a concentration as negative as "
                f"{worst:.3e}, beyond the FP-cancellation band; likely cause(s): {'; '.join(causes)}.",
                UserWarning,
                stacklevel=2,
            )
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
    .. disable_try_examples

    ::

        mass = compute_domain_mass(
            theta=2500.0, v_outlet=500.0, waves=tracker.state.waves, sorption=sorption
        )
        mass >= 0.0
    """
    # Partition [0, v_outlet] at the active faces; each segment has a single owner (the
    # left feeder of its nearest downstream face). A constant owner integrates as
    # C_total·Δv; a fan owner integrates in closed form via integrate_fan_spatial_exact,
    # clamped at the fan's near-apex (larger-retardation) bound. Because the wave list is
    # interaction-consistent (every face crossing fires a solver event), this single-owner
    # sweep is exact for single-front and genuinely interacting multi-front layouts alike.
    faces = _reader_faces(waves, theta)
    boundaries = {0.0, v_outlet}
    boundaries.update(p for (p, _f) in faces if 0.0 < p < v_outlet)
    positions = sorted(boundaries)

    total_mass = 0.0
    for v_start, v_end in pairwise(positions):
        dv = v_end - v_start
        if dv < EPSILON_VOLUME:
            continue
        v_mid = 0.5 * (v_start + v_end)
        feeder = _owner_feeder(faces, v_mid)

        if feeder is None or feeder.is_const:
            c = feeder.value(v_mid, theta) if feeder is not None else 0.0
            total_mass += float(sorption.total_concentration(c)) * dv
        else:
            total_mass += integrate_fan_spatial_exact(
                feeder.theta_apex,
                feeder.v_apex,
                v_start,
                v_end,
                theta,
                sorption,
                c_apex=_near_apex_bound(feeder, sorption),
            )

    return float(total_mass)


def _owner_feeder(faces: list[tuple[float, Face]], v: float) -> Feeder | None:
    """Return the feeder controlling the state at ``v``: nearest-downstream face's left, else outermost right."""
    downstream = [(p, f) for (p, f) in faces if p > v + EPSILON_POSITION]
    if downstream:
        _p, face = min(downstream, key=itemgetter(0))
        return face.left
    if faces:
        _p, face = max(faces, key=itemgetter(0))
        return face.right
    return None


def _near_apex_bound(feeder: Feeder, sorption: SorptionModel) -> float:
    """Return the fan boundary concentration with the larger retardation (the near-apex / plateau side).

    This is the ``c_apex`` clamp for ``integrate_fan_spatial_exact``: the value the fan holds
    at (and beyond) its apex. Monotonicity-agnostic — the low-c bound for R-decreasing
    isotherms, the high-c bound for the Freundlich ``n<1`` mirror where R increases with c.
    """
    r_a = float(sorption.retardation(feeder.c_a))
    r_b = float(sorption.retardation(feeder.c_b))
    return feeder.c_a if r_a >= r_b else feeder.c_b


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
    :class:`~gwtransport.fronttracking.waves.RarefactionWave` (apex = ``theta_start, v_start``) and
    :class:`~gwtransport.fronttracking.waves.DecayingShockWave` (apex = ``theta_origin, v_origin``).

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
    .. disable_try_examples

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
    .. disable_try_examples

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
    *,
    cin: npt.ArrayLike,
    theta_edges: npt.NDArray[np.floating],
) -> float:
    """Total outlet mass over θ → ∞ (finite only for a returning-to-zero pulse).

    The final inlet value ``c_∞ = cin[-1]`` is the sustained boundary state as θ → ∞:

    - For ``c_∞ = 0`` (canonical c_R=0 pulse): injection ceases, the domain empties, and
      every injected mass unit eventually exits — ``m_out_total = m_in_total`` (the finite
      record integral ``Σ cin·Δθ``). The wave list is not needed.
    - For ``c_∞ > 0`` (sustained ambient): the inlet keeps injecting ``c_∞`` forever, so the
      cumulative outlet mass grows without bound — return ``+inf``. The previous formula
      ``m_in_total − C_T(c_∞)·v_outlet`` paired the FINITE record integral with the
      infinite-time steady-state fill and went **negative** whenever
      ``m_in_total < C_T(c_∞)·v_outlet``, which is not a physical outlet mass.

    Parameters
    ----------
    cin : array-like (kw-only)
        Inlet concentration per θ-bin [mass/volume].
    theta_edges : ndarray (kw-only)
        θ bin edges [m³], length ``len(cin) + 1``.

    Returns
    -------
    float
        ``m_in_total`` for ``cin[-1] = 0``; ``+inf`` for ``cin[-1] > 0``.

    See Also
    --------
    compute_cumulative_outlet_mass : Cumulative outlet mass up to a finite θ (use this for
        a sustained ``c_∞ > 0`` boundary, where the θ → ∞ total is unbounded).
    compute_domain_mass : Spatial integral of C_total in the aquifer
    """
    cin_arr = np.asarray(cin, dtype=float)
    # An empty cin is a malformed (no-bin) input by the cin/theta_edges contract;
    # let cin_arr[-1] raise IndexError rather than masking it as c_inf=0.
    if float(cin_arr[-1]) > 0.0:
        return float("inf")
    te = np.asarray(theta_edges, dtype=float)
    return float(np.sum(cin_arr * np.diff(te)))
