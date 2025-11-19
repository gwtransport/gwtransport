"""
Visualization functions for front tracking.

This module provides plotting utilities for visualizing front-tracking simulations:
- V-t diagrams showing wave propagation in space-time
- Breakthrough curves showing concentration at outlet over time

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gwtransport.fronttracking.output import identify_outlet_segments
from gwtransport.fronttracking.solver import FrontTrackerState
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave


def plot_vt_diagram(
    state: FrontTrackerState,
    ax=None,
    t_max: float | None = None,
    figsize: tuple[float, float] = (14, 10),
    show_inactive: bool = False,
    show_events: bool = False,
):
    """
    Create V-t diagram showing all waves in space-time.

    Plots characteristics (blue lines), shocks (red lines), and rarefactions
    (green fans) in the (time, position) plane. This visualization shows how
    waves propagate and interact throughout the simulation.

    Parameters
    ----------
    state : FrontTrackerState
        Complete simulation state containing all waves.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into. If None, a new figure and axes are created
        using ``figsize``.
    t_max : float, optional
        Maximum time to plot [days]. If None, uses final simulation time * 1.2.
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
    fig : matplotlib.figure.Figure
        Figure object containing the V-t diagram.

    Notes
    -----
    - Characteristics appear as blue lines (constant velocity).
    - Shocks appear as thick red lines (jump discontinuities).
    - Rarefactions appear as green fans (smooth transition regions).
    - Outlet position is shown as a horizontal dashed line.
    - Only waves within domain [0, v_outlet] are plotted.

    Examples
    --------
    >>> from gwtransport.fronttracking.solver import FrontTracker
    >>> tracker = FrontTracker(cin, flow, tedges, aquifer_pore_volume, sorption)
    >>> tracker.run()
    >>> fig = plot_vt_diagram(tracker.state)
    >>> fig.savefig("vt_diagram.png")
    """
    # Track whether we created a new figure so we can return a consistent object
    created_fig = False

    if t_max is None:
        # Default to input data time range instead of simulation end time
        # Convert tedges[-1] from Timestamp to days from tedges[0]
        t_max = (state.tedges[-1] - state.tedges[0]) / pd.Timedelta(days=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    # Plot characteristics (blue lines)
    for wave in state.waves:
        if isinstance(wave, CharacteristicWave):
            if not wave.is_active and not show_inactive:
                continue

            t_plot = np.linspace(wave.t_start, t_max, 100)
            v_plot = []
            t_plot_used = []

            for t in t_plot:
                # Compute position manually for inactive waves when show_inactive=True
                if wave.is_active:
                    v = wave.position_at_time(t)
                else:
                    # Manually compute position for visualization
                    v = wave.v_start + wave.velocity() * (t - wave.t_start)

                if v is not None and 0 <= v <= state.v_outlet:
                    v_plot.append(v)
                    t_plot_used.append(t)
                elif v is not None and v > state.v_outlet:
                    # Wave crossed outlet - add exact intersection point
                    vel = wave.velocity()
                    if vel > 0:
                        t_outlet = wave.t_start + (state.v_outlet - wave.v_start) / vel
                        if wave.t_start <= t_outlet <= t_max:
                            v_plot.append(state.v_outlet)
                            t_plot_used.append(t_outlet)
                    break
                else:
                    break

            if len(v_plot) > 0:
                alpha = 0.3 if not wave.is_active else 0.7
                ax.plot(
                    t_plot_used,
                    v_plot,
                    "b-",
                    linewidth=0.5,
                    alpha=alpha,
                    label="Characteristic" if not hasattr(ax, "_char_labeled") else "",
                )
                ax._char_labeled = True

    # Plot shocks (red lines)
    for wave in state.waves:
        if isinstance(wave, ShockWave):
            if not wave.is_active and not show_inactive:
                continue

            t_plot = np.linspace(wave.t_start, t_max, 100)
            v_plot = []
            t_plot_used = []

            for t in t_plot:
                # Compute position manually for inactive waves when show_inactive=True
                if wave.is_active:
                    v = wave.position_at_time(t)
                else:
                    # Manually compute position for visualization
                    v = wave.v_start + wave.velocity * (t - wave.t_start)

                if v is not None and 0 <= v <= state.v_outlet:
                    v_plot.append(v)
                    t_plot_used.append(t)
                elif v is not None and v > state.v_outlet:
                    # Wave crossed outlet - add exact intersection point
                    vel = wave.velocity
                    if vel > 0:
                        t_outlet = wave.t_start + (state.v_outlet - wave.v_start) / vel
                        if wave.t_start <= t_outlet <= t_max:
                            v_plot.append(state.v_outlet)
                            t_plot_used.append(t_outlet)
                    break
                else:
                    break

            if len(v_plot) > 0:
                alpha = 0.5 if not wave.is_active else 1.0
                ax.plot(
                    t_plot_used,
                    v_plot,
                    "r-",
                    linewidth=2,
                    alpha=alpha,
                    label="Shock" if not hasattr(ax, "_shock_labeled") else "",
                )
                ax._shock_labeled = True

    # Plot rarefactions (green fans)
    for wave in state.waves:
        if isinstance(wave, RarefactionWave):
            if not wave.is_active and not show_inactive:
                continue

            t_plot = np.linspace(wave.t_start, t_max, 100)
            v_head_plot = []
            v_tail_plot = []
            t_plot_used = []
            head_crossed = False
            tail_crossed = False

            for t in t_plot:
                # Compute positions manually for inactive waves when show_inactive=True
                if wave.is_active:
                    v_head = wave.head_position_at_time(t)
                    v_tail = wave.tail_position_at_time(t)
                else:
                    # Manually compute positions for visualization
                    v_head = wave.v_start + wave.head_velocity() * (t - wave.t_start)
                    v_tail = wave.v_start + wave.tail_velocity() * (t - wave.t_start)

                # Track time points
                t_plot_used.append(t)

                # Handle head
                if v_head is not None and 0 <= v_head <= state.v_outlet:
                    v_head_plot.append(v_head)
                elif v_head is not None and v_head > state.v_outlet and not head_crossed:
                    # Add exact outlet intersection for head
                    head_vel = wave.head_velocity()
                    if head_vel > 0:
                        t_outlet_head = wave.t_start + (state.v_outlet - wave.v_start) / head_vel
                        if wave.t_start <= t_outlet_head <= t_max:
                            # Insert the exact crossing point
                            v_head_plot.append(state.v_outlet)
                            head_crossed = True
                    v_head_plot.append(None)
                else:
                    v_head_plot.append(None)

                # Handle tail
                if v_tail is not None and 0 <= v_tail <= state.v_outlet:
                    v_tail_plot.append(v_tail)
                elif v_tail is not None and v_tail > state.v_outlet and not tail_crossed:
                    # Add exact outlet intersection for tail
                    tail_vel = wave.tail_velocity()
                    if tail_vel > 0:
                        t_outlet_tail = wave.t_start + (state.v_outlet - wave.v_start) / tail_vel
                        if wave.t_start <= t_outlet_tail <= t_max:
                            # Insert the exact crossing point
                            v_tail_plot.append(state.v_outlet)
                            tail_crossed = True
                    v_tail_plot.append(None)
                else:
                    v_tail_plot.append(None)

            # Plot head and tail boundaries
            alpha = 0.5 if not wave.is_active else 0.8
            label = "Rarefaction" if not hasattr(ax, "_raref_labeled") else ""

            # Plot head (faster boundary)
            valid_head = [(t, v) for t, v in zip(t_plot_used, v_head_plot, strict=False) if v is not None]
            if valid_head:
                t_h, v_h = zip(*valid_head, strict=False)
                ax.plot(t_h, v_h, "g-", linewidth=1.5, alpha=alpha, label=label)
                ax._raref_labeled = True

            # Plot tail (slower boundary)
            valid_tail = [(t, v) for t, v in zip(t_plot_used, v_tail_plot, strict=False) if v is not None]
            if valid_tail:
                t_t, v_t = zip(*valid_tail, strict=False)
                ax.plot(t_t, v_t, "g--", linewidth=1.5, alpha=alpha)

            # Fill between head and tail
            if valid_head and valid_tail and len(valid_head) == len(valid_tail):
                ax.fill_between(
                    t_h,
                    v_h,
                    v_t,
                    color="green",
                    alpha=0.1 if not wave.is_active else 0.2,
                )

    # Plot outlet position
    ax.axhline(
        state.v_outlet,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Outlet (V={state.v_outlet:.1f} m³)",
    )

    # Plot inlet position
    ax.axhline(
        0.0,
        color="k",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Inlet (V=0)",
    )

    # Plot wave interaction events as markers
    if show_events and hasattr(state, "events") and state.events:
        for event in state.events:
            if "time" in event and "position" in event:
                t_event = event["time"]
                v_event = event["position"]
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
                        label="Event" if not hasattr(ax, "_event_labeled") else "",
                    )
                    ax._event_labeled = True

    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("Position (Pore Volume) [m³]", fontsize=12)
    ax.set_title("V-t Diagram: Front Tracking Simulation", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-state.v_outlet * 0.05, state.v_outlet * 1.05)

    fig.tight_layout()
    return fig


def plot_breakthrough_curve(
    state: FrontTrackerState,
    ax=None,
    t_max: float | None = None,
    n_rarefaction_points: int = 50,
    figsize: tuple[float, float] = (12, 6),
    t_first_arrival: float | None = None,
):
    """
    Plot exact analytical concentration breakthrough curve at outlet.

    Uses wave segment information to plot the exact analytical solution
    without discretization. Constant concentration regions are plotted
    as horizontal lines, and rarefaction regions are plotted using their
    exact self-similar solutions.

    Parameters
    ----------
    state : FrontTrackerState
        Complete simulation state containing all waves.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into. If None, a new figure and axes are created
        using ``figsize``.
    t_max : float, optional
        Maximum time to plot [days]. If None, uses final simulation time * 1.1.
    n_rarefaction_points : int, optional
        Number of points to use for plotting rarefaction segments (analytical
        curves). Default 50 per rarefaction segment.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default (12, 6).
    t_first_arrival : float, optional
        First arrival time for marking spin-up period [days]. If None, spin-up
        period is not plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the breakthrough curve.

    Notes
    -----
    - Uses identify_outlet_segments to get exact analytical segment boundaries
    - Constant concentration segments plotted as horizontal lines (no discretization)
    - Rarefaction segments plotted using exact self-similar solution
    - Shocks appear as instantaneous jumps at exact crossing times
    - No bin averaging or discretization artifacts

    Examples
    --------
    >>> from gwtransport.fronttracking.solver import FrontTracker
    >>> tracker = FrontTracker(cin, flow, tedges, aquifer_pore_volume, sorption)
    >>> tracker.run()
    >>> fig = plot_breakthrough_curve(tracker.state)
    >>> fig.savefig("exact_breakthrough.png")
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if t_max is None:
        # Default to input data time range instead of simulation end time
        # Convert tedges[-1] from Timestamp to days from tedges[0]
        import pandas as pd

        t_max = (state.tedges[-1] - state.tedges[0]) / pd.Timedelta(days=1)

    # Use exact analytical segments
    segments = identify_outlet_segments(0.0, t_max, state.v_outlet, state.waves, state.sorption)

    for i, segment in enumerate(segments):
        t_start = segment["t_start"]
        t_end = segment["t_end"]

        if segment["type"] == "constant":
            # Constant concentration segment - plot as horizontal line
            c_const = segment["concentration"]
            ax.plot(
                [t_start, t_end],
                [c_const, c_const],
                "b-",
                linewidth=2,
                label="Outlet concentration" if i == 0 else "",
            )
        elif segment["type"] == "rarefaction":
            # Rarefaction segment - plot exact analytical curve
            raref = segment["wave"]
            t_raref = np.linspace(t_start, t_end, n_rarefaction_points)
            c_raref = np.zeros_like(t_raref)

            for j, t in enumerate(t_raref):
                # Use the rarefaction wave's own concentration_at_point method
                c_at_point = raref.concentration_at_point(state.v_outlet, t)
                if c_at_point is not None:
                    c_raref[j] = c_at_point
                else:
                    # Fallback to boundary values if not in fan
                    c_raref[j] = segment.get("c_start", raref.c_tail)

            ax.plot(t_raref, c_raref, "b-", linewidth=2, label="Outlet concentration" if i == 0 else "")

    # Mark first arrival time if provided
    if t_first_arrival is not None and np.isfinite(t_first_arrival):
        ax.axvline(
            t_first_arrival,
            color="r",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"First arrival (t={t_first_arrival:.2f} days)",
        )

        # Shade spin-up region
        ax.axvspan(
            0,
            t_first_arrival,
            alpha=0.1,
            color="gray",
            label="Spin-up period",
        )

    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("Concentration [mass/volume]", fontsize=12)
    ax.set_title("Breakthrough Curve at Outlet (Exact Analytical)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_xlim(0, t_max)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


def plot_wave_interactions(
    state: FrontTrackerState,
    figsize: tuple[float, float] = (14, 8),
    ax=None,
):
    """
    Plot event timeline showing wave interactions.

    Creates a scatter plot showing when and where different types of wave
    interactions occur during the simulation.

    Parameters
    ----------
    state : FrontTrackerState
        Complete simulation state containing all events.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into. If None, a new figure and axes are created
        using ``figsize``.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default (14, 8).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the event timeline.

    Notes
    -----
    - Each event type is shown with a different color and marker.
    - Outlet crossings are shown separately from internal collisions.
    - Event locations are plotted in the (time, position) plane.

    Examples
    --------
    >>> from gwtransport.fronttracking.solver import FrontTracker
    >>> tracker = FrontTracker(cin, flow, tedges, aquifer_pore_volume, sorption)
    >>> tracker.run()
    >>> fig = plot_wave_interactions(tracker.state)
    >>> fig.savefig("wave_interactions.png")
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Group events by type
    event_types = {}
    for event_dict in state.events:
        event_type = event_dict["type"]
        if event_type not in event_types:
            event_types[event_type] = {"times": [], "locations": []}
        event_types[event_type]["times"].append(event_dict["time"])
        event_types[event_type]["locations"].append(event_dict.get("location", 0.0))

    # Define colors and markers for each event type
    event_style = {
        "CHAR_CHAR_COLLISION": {"color": "blue", "marker": "o", "label": "Char-Char"},
        "SHOCK_SHOCK_COLLISION": {"color": "red", "marker": "s", "label": "Shock-Shock"},
        "SHOCK_CHAR_COLLISION": {"color": "purple", "marker": "^", "label": "Shock-Char"},
        "RAREF_CHAR_COLLISION": {"color": "green", "marker": "v", "label": "Raref-Char"},
        "SHOCK_RAREF_COLLISION": {"color": "orange", "marker": "d", "label": "Shock-Raref"},
        "RAREF_RAREF_COLLISION": {"color": "cyan", "marker": "p", "label": "Raref-Raref"},
        "OUTLET_CROSSING": {"color": "black", "marker": "x", "label": "Outlet Crossing"},
        "INLET_CHANGE": {"color": "gray", "marker": "+", "label": "Inlet Change"},
    }

    # Plot each event type
    for event_type, data in event_types.items():
        style = event_style.get(event_type, {"color": "gray", "marker": "o", "label": event_type})
        ax.scatter(
            data["times"],
            data["locations"],
            c=style["color"],
            marker=style["marker"],
            s=100,
            alpha=0.7,
            label=f"{style['label']} ({len(data['times'])})",
        )

    # Plot outlet line for reference
    if state.events:
        ax.axhline(
            state.v_outlet,
            color="k",
            linestyle="--",
            linewidth=1,
            alpha=0.3,
            label=f"Outlet (V={state.v_outlet:.1f} m³)",
        )

    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("Position (Pore Volume) [m³]", fontsize=12)
    ax.set_title("Wave Interaction Events", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)

    if state.events:
        ax.set_xlim(left=0)
        ax.set_ylim(-state.v_outlet * 0.05, state.v_outlet * 1.05)

    fig.tight_layout()
    return fig
