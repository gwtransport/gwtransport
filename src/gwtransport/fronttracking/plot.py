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


def plot_inlet_concentration(
    dates,
    cin,
    ax=None,
    *,
    t_first_arrival=None,
    event_markers=None,
    color="blue",
    t_max=None,
    xlabel="Time [days]",
    ylabel="Concentration",
    title="Inlet Concentration",
    figsize=(8, 5),
    **step_kwargs,
):
    """
    Plot inlet concentration as step function with optional markers.

    Parameters
    ----------
    dates : pd.DatetimeIndex or array-like
        Time points for concentration values.
    cin : array-like
        Inlet concentration values.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot into. If None, creates new figure.
    t_first_arrival : float, optional
        First arrival time to mark with vertical line [days].
    event_markers : list of dict, optional
        Event markers to add. Each dict should have keys: 'time', 'label', 'color'.
    color : str, optional
        Color for inlet concentration line. Default 'blue'.
    t_max : float, optional
        Maximum time for x-axis [days]. If None, uses full range.
    xlabel : str, optional
        Label for x-axis. Default 'Time [days]'.
    ylabel : str, optional
        Label for y-axis. Default 'Concentration'.
    title : str, optional
        Plot title. Default 'Inlet Concentration'.
    figsize : tuple, optional
        Figure size if creating new figure. Default (8, 5).
    **step_kwargs
        Additional arguments passed to ax.step().

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Convert dates to days from start
    if hasattr(dates, "to_numpy"):
        dates_array = dates.to_numpy()
    else:
        dates_array = np.array(dates)

    t_days = (dates_array - dates_array[0]) / pd.Timedelta(days=1)

    # Plot inlet concentration
    ax.step(t_days, cin, where="post", linewidth=2, color=color, label="Inlet", **step_kwargs)

    # Add first arrival marker if provided
    if t_first_arrival is not None and np.isfinite(t_first_arrival):
        ax.axvline(
            t_first_arrival,
            color="green",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"First arrival ({t_first_arrival:.1f} days)",
        )

    # Add event markers if provided
    if event_markers is not None:
        for marker in event_markers:
            t = marker.get("time")
            label = marker.get("label", "")
            marker_color = marker.get("color", "gray")
            linestyle = marker.get("linestyle", "--")

            if t is not None:
                ax.axvline(
                    t,
                    color=marker_color,
                    linestyle=linestyle,
                    linewidth=1.5,
                    alpha=0.7,
                    label=label,
                )

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if t_max is not None:
        ax.set_xlim(0, t_max)
    else:
        ax.set_xlim(0, t_days[-1])

    return fig, ax


def plot_front_tracking_summary(
    structure,
    dates,
    cin,
    cout_tedges,
    cout,
    *,
    figsize=(16, 10),
    show_exact=True,
    show_bin_averaged=True,
    show_events=True,
    show_inactive=False,
    t_max=None,
    title=None,
    inlet_color="blue",
    outlet_exact_color="blue",
    outlet_binned_color="red",
    first_arrival_color="green",
):
    """
    Create comprehensive 3-panel summary figure for front tracking simulation.

    Creates a multi-panel visualization with:
    - Top-left: V-t diagram showing wave propagation
    - Top-right: Inlet concentration time series
    - Bottom: Outlet concentration (exact and/or bin-averaged)

    Parameters
    ----------
    structure : dict
        Structure returned from infiltration_to_extraction_front_tracking_detailed.
        Must contain keys: 'tracker_state', 't_first_arrival'.
    dates : pd.DatetimeIndex
        Time points for inlet concentration.
    cin : array-like
        Inlet concentration values.
    cout_tedges : pd.DatetimeIndex
        Output time edges for bin-averaged concentration.
    cout : array-like
        Bin-averaged output concentration values.
    figsize : tuple, optional
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
    inlet_color : str, optional
        Color for inlet concentration. Default 'blue'.
    outlet_exact_color : str, optional
        Color for exact outlet curve. Default 'blue'.
    outlet_binned_color : str, optional
        Color for bin-averaged outlet. Default 'red'.
    first_arrival_color : str, optional
        Color for first arrival marker. Default 'green'.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : dict
        Dictionary with keys 'vt', 'inlet', 'outlet' containing axes objects.
    """
    from gwtransport.fronttracking.output import concentration_at_point

    # Create figure with 3-panel layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    axes = {}

    # Top left: V-t diagram
    ax_vt = fig.add_subplot(gs[0, 0])
    plot_vt_diagram(
        structure["tracker_state"],
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
        dates,
        cin,
        ax=ax_inlet,
        t_first_arrival=structure["t_first_arrival"],
        color=inlet_color,
        t_max=t_max,
    )
    axes["inlet"] = ax_inlet

    # Bottom: Outlet concentration (exact and bin-averaged)
    ax_outlet = fig.add_subplot(gs[1, :])

    # Compute time range
    if t_max is None:
        t_max = (dates.to_numpy()[-1] - dates.to_numpy()[0]) / pd.Timedelta(days=1)

    # Exact breakthrough curve
    if show_exact:
        t_exact = np.linspace(0, t_max, 1000)
        c_exact = [
            concentration_at_point(
                structure["tracker_state"].v_outlet,
                t,
                structure["tracker_state"].waves,
                structure["tracker_state"].sorption,
            )
            for t in t_exact
        ]
        ax_outlet.plot(
            t_exact,
            c_exact,
            color=outlet_exact_color,
            linewidth=2.5,
            label="Exact outlet concentration",
            zorder=3,
        )

    # Bin-averaged outlet
    if show_bin_averaged:
        t_edges_days = ((cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)).values
        xstep_cout, ystep_cout = np.repeat(t_edges_days, 2)[1:-1], np.repeat(cout, 2)
        ax_outlet.plot(
            xstep_cout,
            ystep_cout,
            color=outlet_binned_color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Bin-averaged outlet",
            zorder=2,
        )

    # First arrival marker
    t_first = structure["t_first_arrival"]
    if np.isfinite(t_first):
        ax_outlet.axvline(
            t_first,
            color=first_arrival_color,
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

    # Overall title
    if title is not None:
        plt.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    plt.tight_layout()
    return fig, axes


def plot_comparison_favorable_unfavorable(
    structure_favorable,
    dates_favorable,
    cin_favorable,
    structure_unfavorable,
    dates_unfavorable,
    cin_unfavorable,
    *,
    figsize=(16, 10),
    annotate=True,
    t_max_favorable=None,
    t_max_unfavorable=None,
):
    """
    Create side-by-side comparison of favorable vs unfavorable sorption.

    DEPRECATED: Use plot_sorption_comparison instead for more flexibility.

    Creates a 2x2 grid comparing:
    - Top row: Inlet concentrations (favorable left, unfavorable right)
    - Bottom row: Outlet breakthrough curves

    Parameters
    ----------
    structure_favorable : dict
        Structure from front tracking with n>1 (favorable sorption).
    dates_favorable : pd.DatetimeIndex
        Inlet time points for favorable case.
    cin_favorable : array-like
        Inlet concentration for favorable case.
    structure_unfavorable : dict
        Structure from front tracking with n<1 (unfavorable sorption).
    dates_unfavorable : pd.DatetimeIndex
        Inlet time points for unfavorable case.
    cin_unfavorable : array-like
        Inlet concentration for unfavorable case.
    figsize : tuple, optional
        Figure size (width, height). Default (16, 10).
    annotate : bool, optional
        Whether to add arrows/labels for shocks and rarefactions. Default True.
    t_max_favorable : float, optional
        Max time for favorable plots [days]. If None, auto-computed.
    t_max_unfavorable : float, optional
        Max time for unfavorable plots [days]. If None, auto-computed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : ndarray
        2x2 array of axes objects.
    """
    from gwtransport.fronttracking.output import concentration_at_point

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(
        "Comparison: Favorable vs Unfavorable Sorption",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    # Compute time ranges
    if t_max_favorable is None:
        t_max_favorable = (dates_favorable.to_numpy()[-1] - dates_favorable.to_numpy()[0]) / pd.Timedelta(days=1)
    if t_max_unfavorable is None:
        t_max_unfavorable = (dates_unfavorable.to_numpy()[-1] - dates_unfavorable.to_numpy()[0]) / pd.Timedelta(days=1)

    # Left column: Favorable sorption
    ax_fav_inlet = axes[0, 0]
    t_days_fav = (dates_favorable.to_numpy() - dates_favorable.to_numpy()[0]) / pd.Timedelta(days=1)
    ax_fav_inlet.step(t_days_fav, cin_favorable, where="post", linewidth=2.5, color="blue")
    ax_fav_inlet.set_xlabel("Time [days]", fontsize=11)
    ax_fav_inlet.set_ylabel("Concentration", fontsize=11)
    ax_fav_inlet.set_title("Favorable: Inlet", fontsize=12, fontweight="bold", color="darkblue")
    ax_fav_inlet.grid(True, alpha=0.3)
    ax_fav_inlet.set_xlim(0, t_max_favorable)
    ax_fav_inlet.text(
        0.05,
        0.95,
        "High C travels FAST\n(low retardation)",
        transform=ax_fav_inlet.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        fontsize=9,
    )

    ax_fav_outlet = axes[1, 0]
    t_exact_fav = np.linspace(0, t_max_favorable, 2000)
    c_exact_fav = [
        concentration_at_point(
            structure_favorable["tracker_state"].v_outlet,
            t,
            structure_favorable["tracker_state"].waves,
            structure_favorable["tracker_state"].sorption,
        )
        for t in t_exact_fav
    ]
    ax_fav_outlet.plot(t_exact_fav, c_exact_fav, "b-", linewidth=2.5)
    ax_fav_outlet.set_xlabel("Time [days]", fontsize=11)
    ax_fav_outlet.set_ylabel("Concentration", fontsize=11)
    ax_fav_outlet.set_title("Favorable: Outlet", fontsize=12, fontweight="bold", color="darkblue")
    ax_fav_outlet.grid(True, alpha=0.3)
    ax_fav_outlet.set_xlim(0, t_max_favorable)

    # Right column: Unfavorable sorption
    ax_unfav_inlet = axes[0, 1]
    t_days_unfav = (dates_unfavorable.to_numpy() - dates_unfavorable.to_numpy()[0]) / pd.Timedelta(days=1)
    ax_unfav_inlet.step(t_days_unfav, cin_unfavorable, where="post", linewidth=2.5, color="darkred")
    ax_unfav_inlet.set_xlabel("Time [days]", fontsize=11)
    ax_unfav_inlet.set_ylabel("Concentration", fontsize=11)
    ax_unfav_inlet.set_title("Unfavorable: Inlet", fontsize=12, fontweight="bold", color="darkred")
    ax_unfav_inlet.grid(True, alpha=0.3)
    ax_unfav_inlet.set_xlim(0, t_max_unfavorable)
    ax_unfav_inlet.text(
        0.05,
        0.95,
        "High C travels SLOW\n(high retardation)",
        transform=ax_unfav_inlet.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightcoral", "alpha": 0.5},
        fontsize=9,
    )

    ax_unfav_outlet = axes[1, 1]
    t_exact_unfav = np.linspace(0, t_max_unfavorable, 2000)
    c_exact_unfav = [
        concentration_at_point(
            structure_unfavorable["tracker_state"].v_outlet,
            t,
            structure_unfavorable["tracker_state"].waves,
            structure_unfavorable["tracker_state"].sorption,
        )
        for t in t_exact_unfav
    ]
    ax_unfav_outlet.plot(t_exact_unfav, c_exact_unfav, "r-", linewidth=2.5)
    ax_unfav_outlet.set_xlabel("Time [days]", fontsize=11)
    ax_unfav_outlet.set_ylabel("Concentration", fontsize=11)
    ax_unfav_outlet.set_title("Unfavorable: Outlet", fontsize=12, fontweight="bold", color="darkred")
    ax_unfav_outlet.grid(True, alpha=0.3)
    ax_unfav_outlet.set_xlim(0, t_max_unfavorable)

    plt.tight_layout()
    return fig, axes


def plot_sorption_comparison(
    pulse_favorable_structure,
    pulse_unfavorable_structure,
    pulse_dates,
    pulse_cin,
    dip_favorable_structure,
    dip_unfavorable_structure,
    dip_dates,
    dip_cin,
    *,
    figsize=(16, 12),
    t_max_pulse=None,
    t_max_dip=None,
):
    """
    Compare how each inlet produces different outputs with favorable vs unfavorable sorption.

    Creates a 2x3 grid:
    - Row 1: Pulse inlet and its outputs with favorable and unfavorable sorption
    - Row 2: Dip inlet and its outputs with favorable and unfavorable sorption

    This demonstrates how the SAME inlet timeseries produces DIFFERENT breakthrough
    curves depending on the sorption isotherm.

    Parameters
    ----------
    pulse_favorable_structure : dict
        Structure from pulse inlet with favorable sorption (n>1).
    pulse_unfavorable_structure : dict
        Structure from pulse inlet with unfavorable sorption (n<1).
    pulse_dates : pd.DatetimeIndex
        Time points for pulse inlet.
    pulse_cin : array-like
        Pulse inlet concentration (e.g., 0→10→0).
    dip_favorable_structure : dict
        Structure from dip inlet with favorable sorption (n>1).
    dip_unfavorable_structure : dict
        Structure from dip inlet with unfavorable sorption (n<1).
    dip_dates : pd.DatetimeIndex
        Time points for dip inlet.
    dip_cin : array-like
        Dip inlet concentration (e.g., 10→2→10).
    figsize : tuple, optional
        Figure size (width, height). Default (16, 12).
    t_max_pulse : float, optional
        Max time for pulse plots [days]. If None, auto-computed.
    t_max_dip : float, optional
        Max time for dip plots [days]. If None, auto-computed.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    axes : ndarray
        2x3 array of axes objects.
    """
    from gwtransport.fronttracking.output import concentration_at_point

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(
        "Sorption Comparison: How Each Inlet Responds to Favorable vs Unfavorable Sorption",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )

    # Compute time ranges
    if t_max_pulse is None:
        t_max_pulse = (pulse_dates.to_numpy()[-1] - pulse_dates.to_numpy()[0]) / pd.Timedelta(days=1)
    if t_max_dip is None:
        t_max_dip = (dip_dates.to_numpy()[-1] - dip_dates.to_numpy()[0]) / pd.Timedelta(days=1)

    # === ROW 1: Pulse inlet (0→10→0) ===
    t_days_pulse = (pulse_dates.to_numpy() - pulse_dates.to_numpy()[0]) / pd.Timedelta(days=1)

    # Column 1: Pulse inlet
    ax_pulse_inlet = axes[0, 0]
    ax_pulse_inlet.step(t_days_pulse, pulse_cin, where="post", linewidth=2.5, color="black")
    ax_pulse_inlet.set_xlabel("Time [days]", fontsize=10)
    ax_pulse_inlet.set_ylabel("Concentration", fontsize=10)
    ax_pulse_inlet.set_title("Pulse Inlet\n(0→10→0)", fontsize=11, fontweight="bold")
    ax_pulse_inlet.grid(True, alpha=0.3)
    ax_pulse_inlet.set_xlim(0, t_max_pulse)

    # Column 2: Pulse → Favorable outlet
    ax_pulse_fav = axes[0, 1]
    t_exact_pulse_fav = np.linspace(0, t_max_pulse, 1500)
    c_exact_pulse_fav = [
        concentration_at_point(
            pulse_favorable_structure["tracker_state"].v_outlet,
            t,
            pulse_favorable_structure["tracker_state"].waves,
            pulse_favorable_structure["tracker_state"].sorption,
        )
        for t in t_exact_pulse_fav
    ]
    ax_pulse_fav.plot(t_exact_pulse_fav, c_exact_pulse_fav, "b-", linewidth=2.5)
    ax_pulse_fav.set_xlabel("Time [days]", fontsize=10)
    ax_pulse_fav.set_ylabel("Concentration", fontsize=10)
    ax_pulse_fav.set_title("Favorable (n>1)\nShock→Rarefaction", fontsize=11, fontweight="bold", color="darkblue")
    ax_pulse_fav.grid(True, alpha=0.3)
    ax_pulse_fav.set_xlim(0, t_max_pulse)
    ax_pulse_fav.text(
        0.05,
        0.95,
        "High C: FAST\nRise: Sharp\nFall: Smooth",
        transform=ax_pulse_fav.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
        fontsize=8,
    )

    # Column 3: Pulse → Unfavorable outlet
    ax_pulse_unfav = axes[0, 2]
    t_exact_pulse_unfav = np.linspace(0, t_max_pulse, 1500)
    c_exact_pulse_unfav = [
        concentration_at_point(
            pulse_unfavorable_structure["tracker_state"].v_outlet,
            t,
            pulse_unfavorable_structure["tracker_state"].waves,
            pulse_unfavorable_structure["tracker_state"].sorption,
        )
        for t in t_exact_pulse_unfav
    ]
    ax_pulse_unfav.plot(t_exact_pulse_unfav, c_exact_pulse_unfav, "r-", linewidth=2.5)
    ax_pulse_unfav.set_xlabel("Time [days]", fontsize=10)
    ax_pulse_unfav.set_ylabel("Concentration", fontsize=10)
    ax_pulse_unfav.set_title("Unfavorable (n<1)\nRarefaction→Shock", fontsize=11, fontweight="bold", color="darkred")
    ax_pulse_unfav.grid(True, alpha=0.3)
    ax_pulse_unfav.set_xlim(0, t_max_pulse)
    ax_pulse_unfav.text(
        0.05,
        0.95,
        "High C: SLOW\nRise: Smooth\nFall: Sharp",
        transform=ax_pulse_unfav.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightcoral", "alpha": 0.7},
        fontsize=8,
    )

    # === ROW 2: Dip inlet (10→2→10) ===
    t_days_dip = (dip_dates.to_numpy() - dip_dates.to_numpy()[0]) / pd.Timedelta(days=1)

    # Column 1: Dip inlet
    ax_dip_inlet = axes[1, 0]
    ax_dip_inlet.step(t_days_dip, dip_cin, where="post", linewidth=2.5, color="black")
    ax_dip_inlet.set_xlabel("Time [days]", fontsize=10)
    ax_dip_inlet.set_ylabel("Concentration", fontsize=10)
    ax_dip_inlet.set_title("Dip Inlet\n(10→2→10)", fontsize=11, fontweight="bold")
    ax_dip_inlet.grid(True, alpha=0.3)
    ax_dip_inlet.set_xlim(0, t_max_dip)

    # Column 2: Dip → Favorable outlet
    ax_dip_fav = axes[1, 1]
    t_exact_dip_fav = np.linspace(0, t_max_dip, 1500)
    c_exact_dip_fav = [
        concentration_at_point(
            dip_favorable_structure["tracker_state"].v_outlet,
            t,
            dip_favorable_structure["tracker_state"].waves,
            dip_favorable_structure["tracker_state"].sorption,
        )
        for t in t_exact_dip_fav
    ]
    ax_dip_fav.plot(t_exact_dip_fav, c_exact_dip_fav, "b-", linewidth=2.5)
    ax_dip_fav.set_xlabel("Time [days]", fontsize=10)
    ax_dip_fav.set_ylabel("Concentration", fontsize=10)
    ax_dip_fav.set_title("Favorable (n>1)\nRarefaction→Shock", fontsize=11, fontweight="bold", color="darkblue")
    ax_dip_fav.grid(True, alpha=0.3)
    ax_dip_fav.set_xlim(0, t_max_dip)
    ax_dip_fav.text(
        0.05,
        0.95,
        "High C: FAST\nDrop: Smooth\nRise: Sharp",
        transform=ax_dip_fav.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
        fontsize=8,
    )

    # Column 3: Dip → Unfavorable outlet
    ax_dip_unfav = axes[1, 2]
    t_exact_dip_unfav = np.linspace(0, t_max_dip, 1500)
    c_exact_dip_unfav = [
        concentration_at_point(
            dip_unfavorable_structure["tracker_state"].v_outlet,
            t,
            dip_unfavorable_structure["tracker_state"].waves,
            dip_unfavorable_structure["tracker_state"].sorption,
        )
        for t in t_exact_dip_unfav
    ]
    ax_dip_unfav.plot(t_exact_dip_unfav, c_exact_dip_unfav, "r-", linewidth=2.5)
    ax_dip_unfav.set_xlabel("Time [days]", fontsize=10)
    ax_dip_unfav.set_ylabel("Concentration", fontsize=10)
    ax_dip_unfav.set_title("Unfavorable (n<1)\nShock→Rarefaction", fontsize=11, fontweight="bold", color="darkred")
    ax_dip_unfav.grid(True, alpha=0.3)
    ax_dip_unfav.set_xlim(0, t_max_dip)
    ax_dip_unfav.text(
        0.05,
        0.95,
        "High C: SLOW\nDrop: Sharp\nRise: Smooth",
        transform=ax_dip_unfav.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "lightcoral", "alpha": 0.7},
        fontsize=8,
    )

    plt.tight_layout()
    return fig, axes
