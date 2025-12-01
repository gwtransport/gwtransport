"""
Tests for front tracking plotting functions.

These tests verify that plotting functions in gwtransport.fronttracking.plot
execute without errors. They focus on ensuring the functions run to completion
rather than verifying visual output quality.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import FreundlichSorption
from gwtransport.fronttracking.plot import (
    plot_breakthrough_curve,
    plot_front_tracking_summary,
    plot_inlet_concentration,
    plot_sorption_comparison,
    plot_vt_diagram,
    plot_wave_interactions,
)
from gwtransport.fronttracking.solver import FrontTracker

# Use non-interactive backend for testing
plt.switch_backend("Agg")


@pytest.fixture
def simple_pulse_input():
    """
    Create a simple pulse input for testing.

    Returns
    -------
    tuple
        (cin, flow, tedges) where cin is concentration [0, 10, 0],
        flow is constant 100 m³/day, and tedges spans 30 days.
    """
    tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-02-01"])
    cin = np.array([0.0, 10.0, 0.0])
    flow = np.array([100.0, 100.0, 100.0])
    return cin, flow, tedges


@pytest.fixture
def simple_step_input():
    """
    Create a simple step input for testing.

    Returns
    -------
    tuple
        (cin, flow, tedges) where cin is concentration [0, 10],
        flow is constant 100 m³/day, and tedges spans 30 days.
    """
    tedges = pd.to_datetime(["2020-01-01", "2020-01-16", "2020-02-01"])
    cin = np.array([0.0, 10.0])
    flow = np.array([100.0, 100.0])
    return cin, flow, tedges


@pytest.fixture
def freundlich_favorable():
    """Freundlich sorption with n>1 (higher C travels faster)."""
    return FreundlichSorption(k_f=0.01, n=1.5, bulk_density=1500.0, porosity=0.3)


@pytest.fixture
def freundlich_unfavorable():
    """Freundlich sorption with n<1 (lower C travels faster)."""
    return FreundlichSorption(k_f=0.01, n=0.7, bulk_density=1500.0, porosity=0.3)


@pytest.fixture
def tracker_state_pulse(simple_pulse_input, freundlich_favorable):
    """
    Create and run a FrontTracker with pulse input.

    Returns
    -------
    FrontTrackerState
        Completed simulation state with pulse input.
    """
    cin, flow, tedges = simple_pulse_input

    tracker = FrontTracker(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
        sorption=freundlich_favorable,
    )
    tracker.run()

    return tracker.state


@pytest.fixture
def tracker_state_step(simple_step_input, freundlich_favorable):
    """
    Create and run a FrontTracker with step input.

    Returns
    -------
    FrontTrackerState
        Completed simulation state with step input.
    """
    cin, flow, tedges = simple_step_input

    tracker = FrontTracker(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
        sorption=freundlich_favorable,
    )
    tracker.run()

    return tracker.state


class TestPlotVtDiagram:
    """Tests for plot_vt_diagram function."""

    def test_basic_plot(self, tracker_state_pulse):
        """Test basic V-t diagram plotting."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_vt_diagram(tracker_state_pulse, ax=ax)

        assert result is not None
        assert result == ax
        plt.close(fig)

    def test_plot_without_ax(self, tracker_state_pulse):
        """Test V-t diagram plotting without providing axes."""
        result = plot_vt_diagram(tracker_state_pulse)

        assert result is not None
        plt.close("all")

    def test_plot_with_t_max(self, tracker_state_pulse):
        """Test V-t diagram with custom t_max."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_vt_diagram(tracker_state_pulse, ax=ax, t_max=50.0)

        assert result is not None
        plt.close(fig)

    def test_plot_with_inactive_waves(self, tracker_state_pulse):
        """Test V-t diagram showing inactive waves."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_vt_diagram(tracker_state_pulse, ax=ax, show_inactive=True)

        assert result is not None
        plt.close(fig)

    def test_plot_with_events(self, tracker_state_pulse):
        """Test V-t diagram showing events."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_vt_diagram(tracker_state_pulse, ax=ax, show_events=True)

        assert result is not None
        plt.close(fig)

    def test_plot_custom_figsize(self, tracker_state_pulse):
        """Test V-t diagram with custom figure size."""
        result = plot_vt_diagram(tracker_state_pulse, figsize=(10, 8))

        assert result is not None
        plt.close("all")


class TestPlotBreakthroughCurve:
    """Tests for plot_breakthrough_curve function."""

    def test_basic_plot(self, tracker_state_pulse):
        """Test basic breakthrough curve plotting."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_breakthrough_curve(tracker_state_pulse, ax=ax)

        assert result is not None
        assert result == ax
        plt.close(fig)

    def test_plot_without_ax(self, tracker_state_pulse):
        """Test breakthrough curve without providing axes."""
        result = plot_breakthrough_curve(tracker_state_pulse)

        assert result is not None
        plt.close("all")

    def test_plot_with_t_max(self, tracker_state_pulse):
        """Test breakthrough curve with custom t_max."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_breakthrough_curve(tracker_state_pulse, ax=ax, t_max=50.0)

        assert result is not None
        plt.close(fig)

    def test_plot_with_first_arrival(self, tracker_state_pulse):
        """Test breakthrough curve with first arrival time marked."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_breakthrough_curve(tracker_state_pulse, ax=ax, t_first_arrival=5.0)

        assert result is not None
        plt.close(fig)

    def test_plot_custom_rarefaction_points(self, tracker_state_pulse):
        """Test breakthrough curve with custom number of rarefaction points."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_breakthrough_curve(tracker_state_pulse, ax=ax, n_rarefaction_points=100)

        assert result is not None
        plt.close(fig)


class TestPlotWaveInteractions:
    """Tests for plot_wave_interactions function."""

    def test_basic_plot(self, tracker_state_pulse):
        """Test basic wave interactions plotting."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_wave_interactions(tracker_state_pulse, ax=ax)

        assert result is not None
        assert result == ax
        plt.close(fig)

    def test_plot_without_ax(self, tracker_state_pulse):
        """Test wave interactions without providing axes."""
        result = plot_wave_interactions(tracker_state_pulse)

        assert result is not None
        plt.close("all")

    def test_plot_custom_figsize(self, tracker_state_pulse):
        """Test wave interactions with custom figure size."""
        result = plot_wave_interactions(tracker_state_pulse, figsize=(12, 6))

        assert result is not None
        plt.close("all")


class TestPlotInletConcentration:
    """Tests for plot_inlet_concentration function."""

    def test_basic_plot(self, simple_pulse_input):
        """Test basic inlet concentration plotting."""
        cin, _, tedges = simple_pulse_input

        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_inlet_concentration(tedges, cin, ax=ax)

        assert result is not None
        assert result == ax
        plt.close(fig)

    def test_plot_without_ax(self, simple_pulse_input):
        """Test inlet concentration without providing axes."""
        cin, _, tedges = simple_pulse_input

        result = plot_inlet_concentration(tedges, cin)

        assert result is not None
        plt.close("all")

    def test_plot_with_first_arrival(self, simple_pulse_input):
        """Test inlet concentration with first arrival marker."""
        cin, _, tedges = simple_pulse_input

        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_inlet_concentration(tedges, cin, ax=ax, t_first_arrival=10.0)

        assert result is not None
        plt.close(fig)

    def test_plot_with_event_markers(self, simple_pulse_input):
        """Test inlet concentration with event markers."""
        cin, _, tedges = simple_pulse_input

        event_markers = [
            {"time": 5.0, "label": "Event 1", "color": "red"},
            {"time": 15.0, "label": "Event 2", "color": "blue"},
        ]

        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_inlet_concentration(tedges, cin, ax=ax, event_markers=event_markers)

        assert result is not None
        plt.close(fig)

    def test_plot_custom_colors_and_labels(self, simple_pulse_input):
        """Test inlet concentration with custom styling."""
        cin, _, tedges = simple_pulse_input

        fig = plt.figure()
        ax = fig.add_subplot(111)

        result = plot_inlet_concentration(
            tedges,
            cin,
            ax=ax,
            color="red",
            xlabel="Custom X",
            ylabel="Custom Y",
            title="Custom Title",
        )

        assert result is not None
        plt.close(fig)


class TestPlotFrontTrackingSummary:
    """Tests for plot_front_tracking_summary function."""

    def test_basic_summary_plot(self, simple_pulse_input, tracker_state_pulse):
        """Test basic front tracking summary plotting."""
        cin, _, tedges = simple_pulse_input

        # Create output tedges for bin-averaged concentrations
        cout_tedges = pd.date_range(start=tedges[0], periods=40, freq="D")
        cout = np.zeros(len(cout_tedges) - 1)  # Dummy output concentrations

        structure = {
            "tracker_state": tracker_state_pulse,
            "t_first_arrival": 5.0,
        }

        fig = plot_front_tracking_summary(structure, tedges, cin, cout_tedges, cout)

        assert fig is not None
        plt.close(fig)

    def test_summary_plot_exact_only(self, simple_pulse_input, tracker_state_pulse):
        """Test summary plot showing only exact solution."""
        cin, _, tedges = simple_pulse_input

        cout_tedges = pd.date_range(start=tedges[0], periods=40, freq="D")
        cout = np.zeros(len(cout_tedges) - 1)

        structure = {
            "tracker_state": tracker_state_pulse,
            "t_first_arrival": 5.0,
        }

        fig = plot_front_tracking_summary(
            structure,
            tedges,
            cin,
            cout_tedges,
            cout,
            show_exact=True,
            show_bin_averaged=False,
        )

        assert fig is not None
        plt.close(fig)

    def test_summary_plot_binned_only(self, simple_pulse_input, tracker_state_pulse):
        """Test summary plot showing only bin-averaged solution."""
        cin, _, tedges = simple_pulse_input

        cout_tedges = pd.date_range(start=tedges[0], periods=40, freq="D")
        cout = np.zeros(len(cout_tedges) - 1)

        structure = {
            "tracker_state": tracker_state_pulse,
            "t_first_arrival": 5.0,
        }

        fig = plot_front_tracking_summary(
            structure,
            tedges,
            cin,
            cout_tedges,
            cout,
            show_exact=False,
            show_bin_averaged=True,
        )

        assert fig is not None
        plt.close(fig)

    def test_summary_plot_with_events(self, simple_pulse_input, tracker_state_pulse):
        """Test summary plot with events shown."""
        cin, _, tedges = simple_pulse_input

        cout_tedges = pd.date_range(start=tedges[0], periods=40, freq="D")
        cout = np.zeros(len(cout_tedges) - 1)

        structure = {
            "tracker_state": tracker_state_pulse,
            "t_first_arrival": 5.0,
        }

        fig = plot_front_tracking_summary(
            structure,
            tedges,
            cin,
            cout_tedges,
            cout,
            show_events=True,
        )

        assert fig is not None
        plt.close(fig)

    def test_summary_plot_custom_colors(self, simple_pulse_input, tracker_state_pulse):
        """Test summary plot with custom colors."""
        cin, _, tedges = simple_pulse_input

        cout_tedges = pd.date_range(start=tedges[0], periods=40, freq="D")
        cout = np.zeros(len(cout_tedges) - 1)

        structure = {
            "tracker_state": tracker_state_pulse,
            "t_first_arrival": 5.0,
        }

        fig = plot_front_tracking_summary(
            structure,
            tedges,
            cin,
            cout_tedges,
            cout,
            inlet_color="green",
            outlet_exact_color="purple",
            outlet_binned_color="orange",
            first_arrival_color="red",
        )

        assert fig is not None
        plt.close(fig)


class TestPlotSorptionComparison:
    """Tests for plot_sorption_comparison function."""

    def test_basic_sorption_comparison(
        self,
        simple_pulse_input,
        freundlich_favorable,
        freundlich_unfavorable,
    ):
        """Test basic sorption comparison plotting."""
        # Pulse input
        pulse_cin, pulse_flow, pulse_tedges = simple_pulse_input

        # Create dip input (inverted pulse: 10→2→10)
        dip_tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-02-01"])
        dip_cin = np.array([10.0, 2.0, 10.0])

        # Run simulations for pulse
        pulse_fav_tracker = FrontTracker(
            cin=pulse_cin,
            flow=pulse_flow,
            tedges=pulse_tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_favorable,
        )
        pulse_fav_tracker.run()

        pulse_unfav_tracker = FrontTracker(
            cin=pulse_cin,
            flow=pulse_flow,
            tedges=pulse_tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_unfavorable,
        )
        pulse_unfav_tracker.run()

        # Run simulations for dip
        dip_fav_tracker = FrontTracker(
            cin=dip_cin,
            flow=pulse_flow,  # Same flow pattern
            tedges=dip_tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_favorable,
        )
        dip_fav_tracker.run()

        dip_unfav_tracker = FrontTracker(
            cin=dip_cin,
            flow=pulse_flow,  # Same flow pattern
            tedges=dip_tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_unfavorable,
        )
        dip_unfav_tracker.run()

        # Create structures
        pulse_fav_structure = {
            "tracker_state": pulse_fav_tracker.state,
            "t_first_arrival": 5.0,
        }
        pulse_unfav_structure = {
            "tracker_state": pulse_unfav_tracker.state,
            "t_first_arrival": 5.0,
        }
        dip_fav_structure = {
            "tracker_state": dip_fav_tracker.state,
            "t_first_arrival": 5.0,
        }
        dip_unfav_structure = {
            "tracker_state": dip_unfav_tracker.state,
            "t_first_arrival": 5.0,
        }

        # Plot comparison
        fig, axes = plot_sorption_comparison(
            pulse_fav_structure,
            pulse_unfav_structure,
            pulse_tedges,
            pulse_cin,
            dip_fav_structure,
            dip_unfav_structure,
            dip_tedges,
            dip_cin,
        )

        assert fig is not None
        assert axes is not None
        assert axes.shape == (2, 3)
        plt.close(fig)

    def test_sorption_comparison_custom_t_max(
        self,
        simple_pulse_input,
        freundlich_favorable,
        freundlich_unfavorable,
    ):
        """Test sorption comparison with custom t_max values."""
        # Pulse input
        pulse_cin, pulse_flow, pulse_tedges = simple_pulse_input

        # Create dip input
        dip_tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-02-01"])
        dip_cin = np.array([10.0, 2.0, 10.0])

        # Run minimal simulations
        pulse_fav_tracker = FrontTracker(
            cin=pulse_cin,
            flow=pulse_flow,
            tedges=pulse_tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_favorable,
        )
        pulse_fav_tracker.run()

        pulse_unfav_tracker = FrontTracker(
            cin=pulse_cin,
            flow=pulse_flow,
            tedges=pulse_tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_unfavorable,
        )
        pulse_unfav_tracker.run()

        dip_fav_tracker = FrontTracker(
            cin=dip_cin,
            flow=pulse_flow,
            tedges=dip_tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_favorable,
        )
        dip_fav_tracker.run()

        dip_unfav_tracker = FrontTracker(
            cin=dip_cin,
            flow=pulse_flow,
            tedges=dip_tedges,
            aquifer_pore_volume=500.0,
            sorption=freundlich_unfavorable,
        )
        dip_unfav_tracker.run()

        # Create structures
        pulse_fav_structure = {
            "tracker_state": pulse_fav_tracker.state,
            "t_first_arrival": 5.0,
        }
        pulse_unfav_structure = {
            "tracker_state": pulse_unfav_tracker.state,
            "t_first_arrival": 5.0,
        }
        dip_fav_structure = {
            "tracker_state": dip_fav_tracker.state,
            "t_first_arrival": 5.0,
        }
        dip_unfav_structure = {
            "tracker_state": dip_unfav_tracker.state,
            "t_first_arrival": 5.0,
        }

        # Plot with custom t_max
        fig, axes = plot_sorption_comparison(
            pulse_fav_structure,
            pulse_unfav_structure,
            pulse_tedges,
            pulse_cin,
            dip_fav_structure,
            dip_unfav_structure,
            dip_tedges,
            dip_cin,
            t_max_pulse=50.0,
            t_max_dip=60.0,
        )

        assert fig is not None
        assert axes.shape == (2, 3)
        plt.close(fig)
