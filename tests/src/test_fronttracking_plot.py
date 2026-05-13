"""
Tests for front tracking plotting functions.

Per issue #174 group 8 (P2.4): plot tests inspect rendered data via
``ax.get_lines()`` / ``ax.collections`` rather than just asserting ``result is not None``.
Selectors use labels / colors / counts that survive matplotlib version changes;
they do not index ``ax.get_lines()`` positionally.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gwtransport.fronttracking.math import FreundlichSorption
from gwtransport.fronttracking.plot import (
    plot_breakthrough_curve,
    plot_inlet_concentration,
    plot_vt_diagram,
    plot_wave_interactions,
)
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave

plt.switch_backend("Agg")


@pytest.fixture
def freundlich_favorable():
    return FreundlichSorption(k_f=0.01, n=1.5, bulk_density=1500.0, porosity=0.3)


@pytest.fixture
def tracker_step(freundlich_favorable):
    """Step input 0 -> 10 with constant flow; long enough that breakthrough is fully visible."""
    n_bins = 50
    tedges = pd.date_range("2020-01-01", periods=n_bins + 1, freq="10D")
    cin = np.array([0.0] + [10.0] * (n_bins - 1))
    flow = np.full(n_bins, 100.0)
    tracker = FrontTracker(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=50.0,
        sorption=freundlich_favorable,
    )
    tracker.run(max_iterations=500)
    return tracker


@pytest.fixture
def tracker_pulse(freundlich_favorable):
    """Pulse input 0 -> 10 -> 0."""
    tedges = pd.to_datetime(["2020-01-01", "2020-01-11", "2020-01-21", "2020-02-01"])
    cin = np.array([0.0, 10.0, 0.0])
    flow = np.array([100.0, 100.0, 100.0])
    tracker = FrontTracker(
        cin=cin,
        flow=flow,
        tedges=tedges,
        aquifer_pore_volume=500.0,
        sorption=freundlich_favorable,
    )
    tracker.run()
    return tracker


class TestPlotVtDiagramData:
    """V-t diagram renders at least one segment per active wave kind."""

    def test_segments_match_active_wave_counts(self, tracker_pulse):
        fig, ax = plt.subplots()
        plot_vt_diagram(tracker_pulse.state, ax=ax)

        active_chars = sum(1 for w in tracker_pulse.state.waves if isinstance(w, CharacteristicWave) and w.is_active)
        active_shocks = sum(1 for w in tracker_pulse.state.waves if isinstance(w, ShockWave) and w.is_active)
        active_rarefs = sum(1 for w in tracker_pulse.state.waves if isinstance(w, RarefactionWave) and w.is_active)

        # plot.py uses blue for characteristics, red for shocks; rarefactions render
        # as fill_between collections (not lines).
        if active_chars > 0:
            assert sum(1 for ln in ax.get_lines() if ln.get_color() == "b") >= active_chars
        if active_shocks > 0:
            assert sum(1 for ln in ax.get_lines() if ln.get_color() == "r") >= active_shocks
        if active_rarefs > 0:
            assert len(ax.collections) >= active_rarefs

        plt.close(fig)


class TestPlotBreakthroughCurveData:
    """Breakthrough plot's rendered y-values cover the simulation's outlet states."""

    def test_plotted_breakthrough_covers_inlet_and_plateau(self, tracker_step):
        fig, ax = plt.subplots()
        plot_breakthrough_curve(tracker_step.state, ax=ax)

        lines = list(ax.get_lines())
        assert lines, "expected at least one breakthrough line"

        all_y = np.unique(np.concatenate([np.asarray(ln.get_ydata(), dtype=float) for ln in lines]))

        # The breakthrough should reach the inlet plateau concentration (10).
        assert np.any(np.isclose(all_y, 10.0, atol=1e-10)), (
            f"breakthrough plot does not reach the inlet plateau; y-values seen: {all_y}"
        )
        # And it should also include the pre-arrival zero state.
        assert np.any(np.isclose(all_y, 0.0, atol=1e-10)), (
            f"breakthrough plot is missing the spin-up (c=0) segment; y-values seen: {all_y}"
        )

        plt.close(fig)


class TestPlotInletConcentrationData:
    """Inlet concentration plot's y-values include each cin level."""

    def test_step_y_values_include_cin_levels(self, tracker_step):
        fig, ax = plt.subplots()
        plot_inlet_concentration(tracker_step.state.tedges, tracker_step.state.cin, ax=ax)

        lines = [ln for ln in ax.get_lines() if np.asarray(ln.get_ydata()).size >= 2]
        assert lines, "expected a step line in inlet concentration plot"

        all_y = np.unique(np.concatenate([np.asarray(ln.get_ydata(), dtype=float) for ln in lines]))
        for level in np.unique(tracker_step.state.cin):
            assert np.any(np.isclose(all_y, level, atol=1e-10)), (
                f"cin level {level} not found in plotted y-values {all_y}"
            )

        plt.close(fig)


class TestPlotSmoke:
    """Plot functions whose data-content is covered indirectly (no API for it):
    confirm they render without raising."""

    def test_plot_vt_diagram_step_does_not_raise(self, tracker_step):
        fig, ax = plt.subplots()
        plot_vt_diagram(tracker_step.state, ax=ax)
        plt.close(fig)

    def test_plot_wave_interactions_does_not_raise(self, tracker_pulse):
        fig, ax = plt.subplots()
        plot_wave_interactions(tracker_pulse.state, ax=ax)
        plt.close(fig)
