"""
Visualization functions for front tracking.

This module provides plotting utilities for visualizing front-tracking simulations:
- V-t diagrams showing wave propagation in space-time
- Breakthrough curves showing concentration at outlet over time

Internally the simulation uses cumulative-flow coordinates (V, θ). All plots
remain in user-facing time t (days). Translation is done via the state's
``t_at_theta`` / ``theta_at_t`` methods at the plotting boundary.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gwtransport._time import tedges_to_days
from gwtransport.fronttracking.output import compute_breakthrough_curve
from gwtransport.fronttracking.solver import FrontTrackerState
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave
from gwtransport.utils import step_plot_coords


def _wave_trajectory_in_t(
    state: FrontTrackerState,
    theta_start: float,
    v_start: float,
    speed: float,
    t_max: float,
    *,
    n_points: int = 100,
) -> tuple[list[float], list[float]]:
    """Convert a straight-in-θ wave trajectory into (t, V) samples for plotting.

    A wave with position ``V(θ) = v_start + speed * (θ - theta_start)`` is
    sampled at ``n_points`` θ-values between ``theta_start`` and the θ that
    corresponds to ``t_max`` (or the outlet, whichever comes first), then
    each sample is translated back to user-facing time via ``state.t_at_theta``.

    Parameters
    ----------
    state : FrontTrackerState
        Simulation state providing θ↔t translation.
    theta_start : float
        θ at which the wave forms [m³].
    v_start : float
        V at which the wave forms [m³].
    speed : float
        Wave speed dV/dθ.
    t_max : float
        Maximum user-facing time [days].
    n_points : int, optional
        Number of θ-samples (before clipping to outlet). Default 100.

    Returns
    -------
    t_samples : list of float
        User-facing times [days], monotonic.
    v_samples : list of float
        Wave positions at those times [m³], clipped to ``[0, v_outlet]``.
    """
    theta_max = state.theta_at_t(t_max)
    if theta_max <= theta_start:
        return [], []

    # If the wave will exit the domain before θ_max, clip θ to the outlet
    # crossing θ so we plot exactly up to V = v_outlet.
    if speed > 0:
        theta_outlet = theta_start + (state.v_outlet - v_start) / speed
        theta_end = min(theta_max, theta_outlet)
    else:
        theta_end = theta_max

    if theta_end <= theta_start:
        return [], []

    thetas = np.linspace(theta_start, theta_end, n_points)
    vs = v_start + speed * (thetas - theta_start)

    mask = (vs >= 0) & (vs <= state.v_outlet)
    t_arr = [state.t_at_theta(float(theta)) for theta in thetas[mask]]
    # Callers test truthiness of the returned lists (``if v_head:``), so keep
    # Python lists rather than arrays.
    return t_arr, vs[mask].tolist()


def plot_vt_diagram(
    state: FrontTrackerState,
    ax: Axes | None = None,
    *,
    t_max: float | None = None,
    figsize: tuple[float, float] = (14, 10),
    show_inactive: bool = False,
    show_events: bool = False,
) -> Axes:
    """
    Create V-t diagram showing all waves in space-time.

    Plots characteristics (blue lines), shocks (red lines), and rarefactions
    (green fans) in the (time, position) plane. This visualization shows how
    waves propagate and interact throughout the simulation.

    Internally the waves live in (V, θ); each wave's straight-line θ-trajectory
    is converted back to user-facing time t via ``state.t_at_theta`` before
    plotting.

    Parameters
    ----------
    state : FrontTrackerState
        Complete simulation state containing all waves.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into. If None, a new figure and axes are created
        using ``figsize``.
    t_max : float, optional
        Maximum time to plot [days]. If None, uses the input data time range.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default (14, 10).
    show_inactive : bool, optional
        Whether to show inactive waves (deactivated by interactions).
        Default False.
    show_events : bool, optional
        Whether to show wave interaction events as markers.
        Default False.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object containing the V-t diagram.

    See Also
    --------
    plot_front_tracking_summary : Multi-panel summary combining these views.
    gwtransport.advection.infiltration_to_extraction_nonlinear_sorption : Produces the tracker state.

    Notes
    -----
    - Characteristics appear as blue lines (constant speed in θ).
    - Shocks appear as thick red lines (jump discontinuities).
    - Rarefactions appear as green fans (smooth transition regions).
    - Outlet position is shown as a horizontal dashed line.
    - Only waves within domain [0, v_outlet] are plotted.

    Examples
    --------
    .. disable_try_examples

    ::

        from gwtransport.fronttracking.solver import FrontTracker

        tracker = FrontTracker(cin, flow, tedges, aquifer_pore_volume, sorption)
        tracker.run()
        ax = plot_vt_diagram(tracker.state)
        ax.figure.savefig("vt_diagram.png")
    """
    if t_max is None:
        t_max = float((state.tedges[-1] - state.tedges[0]) / pd.Timedelta(days=1))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    char_labeled = False
    shock_labeled = False
    raref_labeled = False
    event_labeled = False

    for wave in state.waves:
        if isinstance(wave, CharacteristicWave):
            if not wave.is_active and not show_inactive:
                continue

            t_plot, v_plot = _wave_trajectory_in_t(state, wave.theta_start, wave.v_start, wave.speed(), t_max)

            if len(v_plot) > 0:
                alpha = 0.3 if not wave.is_active else 0.7
                ax.plot(
                    t_plot,
                    v_plot,
                    "b-",
                    linewidth=0.5,
                    alpha=alpha,
                    label="Characteristic" if not char_labeled else "",
                )
                char_labeled = True

    for wave in state.waves:
        if isinstance(wave, ShockWave):
            if not wave.is_active and not show_inactive:
                continue

            t_plot, v_plot = _wave_trajectory_in_t(state, wave.theta_start, wave.v_start, wave.speed, t_max)

            if len(v_plot) > 0:
                alpha = 0.5 if not wave.is_active else 1.0
                ax.plot(
                    t_plot,
                    v_plot,
                    "r-",
                    linewidth=2,
                    alpha=alpha,
                    label="Shock" if not shock_labeled else "",
                )
                shock_labeled = True

    for wave in state.waves:
        if isinstance(wave, RarefactionWave):
            if not wave.is_active and not show_inactive:
                continue

            t_head, v_head = _wave_trajectory_in_t(state, wave.theta_start, wave.v_start, wave.head_speed(), t_max)
            t_tail, v_tail = _wave_trajectory_in_t(state, wave.theta_start, wave.v_start, wave.tail_speed(), t_max)

            alpha = 0.5 if not wave.is_active else 0.8
            label = "Rarefaction" if not raref_labeled else ""

            if v_head:
                ax.plot(t_head, v_head, "g-", linewidth=1.5, alpha=alpha, label=label)
                raref_labeled = True

            if v_tail:
                ax.plot(t_tail, v_tail, "g--", linewidth=1.5, alpha=alpha)

            # Fill between head and tail. Both are sampled from the same set of
            # θ values, so when neither is clipped at the outlet they correspond
            # one-to-one in time; sample lengths can differ once one boundary
            # hits the outlet earlier. Fill only the overlap region.
            if v_head and v_tail:
                n_fill = min(len(v_head), len(v_tail))
                if n_fill > 1:
                    ax.fill_between(
                        t_head[:n_fill],
                        v_head[:n_fill],
                        v_tail[:n_fill],
                        color="green",
                        alpha=0.1 if not wave.is_active else 0.2,
                    )

    ax.axhline(
        state.v_outlet,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Outlet (V={state.v_outlet:.1f} m³)",
    )

    ax.axhline(
        0.0,
        color="k",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Inlet (V=0)",
    )

    # Plot wave interaction events as markers. Event records carry ``"theta"``;
    # translate to user-facing t for display via ``state.t_at_theta``.
    if show_events and state.events:
        for event in state.events:
            if "theta" in event and "location" in event:
                t_event = state.t_at_theta(event["theta"])
                v_event = event["location"]
                if 0 <= t_event <= t_max and 0 <= v_event <= state.v_outlet:
                    # Determine marker style based on event type
                    event_type = event.get("type", "unknown")
                    if "shock" in event_type.lower() or "collision" in event_type.lower():
                        marker = "X"
                        color = "red"
                        size = 100
                    elif "rarefaction" in event_type.lower():
                        marker = "o"
                        color = "green"
                        size = 80
                    elif "outlet" in event_type.lower():
                        marker = "s"
                        color = "black"
                        size = 80
                    else:
                        marker = "D"
                        color = "gray"
                        size = 60

                    ax.scatter(
                        t_event,
                        v_event,
                        marker=marker,
                        s=size,
                        color=color,
                        edgecolors="black",
                        linewidths=1.5,
                        alpha=0.8,
                        zorder=10,
                        label="Event" if not event_labeled else "",
                    )
                    event_labeled = True

    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("Position (Pore Volume) [m³]", fontsize=12)
    ax.set_title("V-t Diagram: Front Tracking Simulation", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-state.v_outlet * 0.05, state.v_outlet * 1.05)

    return ax


def plot_inlet_concentration(
    tedges: pd.DatetimeIndex,
    cin: npt.ArrayLike,
    ax: Axes | None = None,
    *,
    t_first_arrival: float | None = None,
    color: str = "blue",
    t_max: float | None = None,
    figsize: tuple[float, float] = (8, 5),
) -> Axes:
    """
    Plot inlet concentration as a step function.

    Parameters
    ----------
    tedges : pandas.DatetimeIndex
        Time bin edges for inlet concentration.
        Length = len(cin) + 1.
    cin : array-like
        Inlet concentration values.
        Length = len(tedges) - 1.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into. If None, creates new figure.
    t_first_arrival : float, optional
        First arrival time to mark with vertical line [days].
    color : str, optional
        Color for inlet concentration line. Default 'blue'.
    t_max : float, optional
        Maximum time for x-axis [days]. If None, uses full range.
    figsize : tuple of float, optional
        Figure size if creating new figure. Default (8, 5).

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object.

    See Also
    --------
    plot_front_tracking_summary : Multi-panel summary that places this inlet panel.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    t_days = tedges_to_days(tedges)

    x_plot, y_plot = step_plot_coords(t_days, cin)
    ax.plot(x_plot, y_plot, linewidth=2, color=color, label="Inlet")

    if t_first_arrival is not None and np.isfinite(t_first_arrival):
        ax.axvline(
            t_first_arrival,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"First arrival ({t_first_arrival:.1f} days)",
        )

    ax.set_xlabel("Time [days]", fontsize=10)
    ax.set_ylabel("Concentration", fontsize=10)
    ax.set_title("Inlet Concentration", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if t_max is not None:
        ax.set_xlim(0, t_max)
    else:
        ax.set_xlim(0, t_days[-1])

    return ax


def _outlet_concentration_curve(
    state: FrontTrackerState,
    t_array: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Sample the exact outlet concentration at the given user-facing times.

    The ``t`` array is mapped to θ vectorially via ``state.theta_at_t_array``
    and delegated to :func:`compute_breakthrough_curve` (the outlet body of
    ``concentration_at_point`` over a θ-array).

    Parameters
    ----------
    state : FrontTrackerState
        Simulation state.
    t_array : numpy.ndarray
        User-facing time points [days].

    Returns
    -------
    c_out : numpy.ndarray
        Outlet concentrations matching ``t_array``.
    """
    theta_array = state.theta_at_t_array(t_array)
    return compute_breakthrough_curve(theta_array, state.v_outlet, state.waves, state.sorption)


def plot_front_tracking_summary(
    structure: dict,
    tedges: pd.DatetimeIndex,
    cin: npt.ArrayLike,
    cout_tedges: pd.DatetimeIndex,
    cout: npt.ArrayLike,
    *,
    figsize: tuple[float, float] = (16, 10),
    show_exact: bool = True,
    show_bin_averaged: bool = True,
    show_events: bool = True,
    show_inactive: bool = False,
    t_max: float | None = None,
    title: str | None = None,
) -> tuple[Figure, dict]:
    """
    Create comprehensive 3-panel summary figure for front tracking simulation.

    Creates a multi-panel visualization with:
    - Top-left: V-t diagram showing wave propagation
    - Top-right: Inlet concentration time series
    - Bottom: Outlet concentration (exact and/or bin-averaged)

    Parameters
    ----------
    structure : dict
        Structure returned from infiltration_to_extraction_nonlinear_sorption.
        Must contain keys: 'tracker_state', 'theta_first_arrival'.
    tedges : pandas.DatetimeIndex
        Time bin edges for inlet concentration.
        Length = len(cin) + 1.
    cin : array-like
        Inlet concentration values.
        Length = len(tedges) - 1.
    cout_tedges : pandas.DatetimeIndex
        Output time bin edges for bin-averaged concentration.
        Length = len(cout) + 1.
    cout : array-like
        Bin-averaged output concentration values.
        Length = len(cout_tedges) - 1.
    figsize : tuple of float, optional
        Figure size (width, height). Default (16, 10).
    show_exact : bool, optional
        Whether to show exact analytical breakthrough curve. Default True.
    show_bin_averaged : bool, optional
        Whether to show bin-averaged concentration. Default True.
    show_events : bool, optional
        Whether to show wave interaction events on V-t diagram. Default True.
    show_inactive : bool, optional
        Whether to show inactive waves on V-t diagram. Default False.
    t_max : float, optional
        Maximum time for plots [days]. If None, uses input data range.
    title : str, optional
        Overall figure title. If None, uses generic title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : dict
        Dictionary with keys 'vt', 'inlet', 'outlet' containing axes objects.

    See Also
    --------
    plot_vt_diagram : The top-left sub-panel.
    plot_inlet_concentration : The top-right sub-panel.
    gwtransport.advection.infiltration_to_extraction_nonlinear_sorption : Produces ``structure``.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    axes: dict = {}
    tracker_state: FrontTrackerState = structure["tracker_state"]

    if t_max is None:
        t_max = float((tedges[-1] - tedges[0]) / pd.Timedelta(days=1))

    # Top left: V-t diagram
    ax_vt = fig.add_subplot(gs[0, 0])
    plot_vt_diagram(
        tracker_state,
        ax=ax_vt,
        show_inactive=show_inactive,
        show_events=show_events,
        t_max=t_max,
    )
    ax_vt.set_title("V-t Diagram", fontsize=12, fontweight="bold")
    axes["vt"] = ax_vt

    # Top right: Inlet concentration
    ax_inlet = fig.add_subplot(gs[0, 1])
    plot_inlet_concentration(
        tedges,
        cin,
        ax=ax_inlet,
        t_first_arrival=tracker_state.t_at_theta(structure["theta_first_arrival"]),
        t_max=t_max,
    )
    axes["inlet"] = ax_inlet

    # Bottom: Outlet concentration (exact and bin-averaged)
    ax_outlet = fig.add_subplot(gs[1, :])

    if show_exact:
        t_exact = np.linspace(0, t_max, 1000)
        c_exact = _outlet_concentration_curve(tracker_state, t_exact)
        ax_outlet.plot(
            t_exact,
            c_exact,
            color="blue",
            linewidth=2.5,
            label="Exact outlet concentration",
            zorder=3,
        )

    if show_bin_averaged:
        # Share the exact curve's origin (tracker_state.tedges[0]); referencing the output
        # grid to its own first edge shifts the overlay by (cout_tedges[0] - tedges[0]) days.
        t_edges_days = tedges_to_days(cout_tedges, ref=tracker_state.tedges[0])
        xstep_cout, ystep_cout = step_plot_coords(t_edges_days, cout)
        ax_outlet.plot(
            xstep_cout,
            ystep_cout,
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Bin-averaged outlet",
            zorder=2,
        )

    t_first = tracker_state.t_at_theta(structure["theta_first_arrival"])
    if np.isfinite(t_first):
        ax_outlet.axvline(
            t_first,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"First arrival ({t_first:.1f} days)",
            zorder=1,
        )

    ax_outlet.set_xlabel("Time [days]", fontsize=11)
    ax_outlet.set_ylabel("Concentration", fontsize=11)
    ax_outlet.set_title("Outlet Concentration: Exact vs Bin-Averaged", fontsize=12, fontweight="bold")
    ax_outlet.grid(True, alpha=0.3)
    ax_outlet.legend(fontsize=9)
    ax_outlet.set_xlim(0, t_max)
    axes["outlet"] = ax_outlet

    if title is not None:
        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    return fig, axes


def plot_sorption_comparison(
    pulse_favorable_structure: dict,
    pulse_unfavorable_structure: dict,
    pulse_tedges: pd.DatetimeIndex,
    pulse_cin: npt.ArrayLike,
    dip_favorable_structure: dict,
    dip_unfavorable_structure: dict,
    dip_tedges: pd.DatetimeIndex,
    dip_cin: npt.ArrayLike,
    *,
    figsize: tuple[float, float] = (16, 12),
    t_max_pulse: float | None = None,
    t_max_dip: float | None = None,
) -> tuple[Figure, npt.NDArray]:
    """
    Compare how each inlet produces different outputs with n>1 vs n<1 sorption.

    Creates a 2x3 grid:
    - Row 1: Pulse inlet and its outputs with n>1 and n<1 sorption
    - Row 2: Dip inlet and its outputs with n>1 and n<1 sorption

    This demonstrates how the SAME inlet timeseries produces DIFFERENT breakthrough
    curves depending on the sorption isotherm.

    Parameters
    ----------
    pulse_favorable_structure : dict
        Structure from pulse inlet with n>1 (higher C travels faster).
    pulse_unfavorable_structure : dict
        Structure from pulse inlet with n<1 (lower C travels faster).
    pulse_tedges : pandas.DatetimeIndex
        Time bin edges for pulse inlet.
        Length = len(pulse_cin) + 1.
    pulse_cin : array-like
        Pulse inlet concentration (e.g., 0->10->0).
        Length = len(pulse_tedges) - 1.
    dip_favorable_structure : dict
        Structure from dip inlet with n>1 (higher C travels faster).
    dip_unfavorable_structure : dict
        Structure from dip inlet with n<1 (lower C travels faster).
    dip_tedges : pandas.DatetimeIndex
        Time bin edges for dip inlet.
        Length = len(dip_cin) + 1.
    dip_cin : array-like
        Dip inlet concentration (e.g., 10->2->10).
        Length = len(dip_tedges) - 1.
    figsize : tuple of float, optional
        Figure size (width, height). Default (16, 12).
    t_max_pulse : float, optional
        Max time for pulse plots [days]. If None, auto-computed.
    t_max_dip : float, optional
        Max time for dip plots [days]. If None, auto-computed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : numpy.ndarray
        2x3 array of axes objects.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        "Sorption Comparison: How Each Inlet Responds to n>1 vs n<1 Sorption",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    if t_max_pulse is None:
        t_max_pulse = float((pulse_tedges[-1] - pulse_tedges[0]) / pd.Timedelta(days=1))
    if t_max_dip is None:
        t_max_dip = float((dip_tedges[-1] - dip_tedges[0]) / pd.Timedelta(days=1))

    # Column 0 renders each inlet; columns 1-2 render that inlet's outlet response under the
    # favorable (n>1) and unfavorable (n<1) isotherm.
    for ax, panel_tedges, panel_cin, panel_title, panel_t_max in (
        (axes[0, 0], pulse_tedges, pulse_cin, "Pulse Inlet\n(0->10->0)", t_max_pulse),
        (axes[1, 0], dip_tedges, dip_cin, "Dip Inlet\n(10->2->10)", t_max_dip),
    ):
        x_step, y_step = step_plot_coords(tedges_to_days(panel_tedges), panel_cin)
        ax.plot(x_step, y_step, linewidth=2.5, color="black")
        ax.set_xlabel("Time [days]", fontsize=10)
        ax.set_ylabel("Concentration", fontsize=10)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, panel_t_max)

    for ax, structure, panel_t_max, favorable, wave_sequence, annotation in (
        (
            axes[0, 1],
            pulse_favorable_structure,
            t_max_pulse,
            True,
            "Shock->Rarefaction",
            "High C: FAST\nRise: Sharp\nFall: Smooth",
        ),
        (
            axes[0, 2],
            pulse_unfavorable_structure,
            t_max_pulse,
            False,
            "Rarefaction->Shock",
            "High C: SLOW\nRise: Smooth\nFall: Sharp",
        ),
        (
            axes[1, 1],
            dip_favorable_structure,
            t_max_dip,
            True,
            "Rarefaction->Shock",
            "High C: FAST\nDrop: Smooth\nRise: Sharp",
        ),
        (
            axes[1, 2],
            dip_unfavorable_structure,
            t_max_dip,
            False,
            "Shock->Rarefaction",
            "High C: SLOW\nDrop: Sharp\nRise: Smooth",
        ),
    ):
        line_style, title_color, box_color = (
            ("b-", "darkblue", "lightblue") if favorable else ("r-", "darkred", "lightcoral")
        )
        t_exact = np.linspace(0, panel_t_max, 1500)
        ax.plot(t_exact, _outlet_concentration_curve(structure["tracker_state"], t_exact), line_style, linewidth=2.5)
        ax.set_xlabel("Time [days]", fontsize=10)
        ax.set_ylabel("Concentration", fontsize=10)
        ax.set_title(
            f"{'n>1' if favorable else 'n<1'}\n{wave_sequence}", fontsize=11, fontweight="bold", color=title_color
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, panel_t_max)
        ax.text(
            0.05,
            0.95,
            annotation,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": box_color, "alpha": 0.7},
            fontsize=8,
        )

    return fig, axes
