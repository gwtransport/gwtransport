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
from matplotlib.figure import Figure

from gwtransport.fronttracking.math import FreundlichSorption
from gwtransport.fronttracking.output import concentration_at_point
from gwtransport.fronttracking.plot import (
    plot_breakthrough_curve,
    plot_front_tracking_summary,
    plot_inlet_concentration,
    plot_sorption_comparison,
    plot_vt_diagram,
    plot_wave_interactions,
)
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.validation import verify_physics
from gwtransport.fronttracking.waves import CharacteristicWave, DecayingShockWave, RarefactionWave, ShockWave

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

        state = tracker_pulse.state
        # Default t_max in plot_vt_diagram is (tedges[-1] - tedges[0]) in days,
        # corresponding to theta_max = state.theta_at_t(t_max). Waves whose
        # theta_start is past that are active in state but not visible — only
        # count waves whose trajectory begins within the visible θ window.
        t_max_days = float((state.tedges[-1] - state.tedges[0]) / pd.Timedelta(days=1))
        theta_max = state.theta_at_t(t_max_days)

        def _visible(w):
            return w.is_active and w.theta_start < theta_max

        visible_chars = sum(1 for w in state.waves if isinstance(w, CharacteristicWave) and _visible(w))
        visible_shocks = sum(1 for w in state.waves if isinstance(w, ShockWave) and _visible(w))
        visible_rarefs = sum(1 for w in state.waves if isinstance(w, RarefactionWave) and _visible(w))

        # plot.py uses blue ('b') for characteristics, red ('r') for shocks; rarefactions render
        # as green head/tail lines plus a fill_between collection.
        if visible_chars > 0:
            assert sum(1 for ln in ax.get_lines() if ln.get_color() == "b") >= visible_chars
        if visible_shocks > 0:
            assert sum(1 for ln in ax.get_lines() if ln.get_color() == "r") >= visible_shocks
        if visible_rarefs > 0:
            assert len(ax.collections) >= visible_rarefs

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

    def test_plotted_breakthrough_renders_decaying_fan(self):
        """The exact-segment plot renders a DecayingShockWave's fan at the outlet (issue #311).

        Canonical Freundlich n=2 pulse whose DSW crosses v_outlet=100 at θ=3525 (t=35.25 d,
        constant flow 100 m³/d): after arrival the outlet concentration follows the DSW's
        self-similar fan profile. The rendered breakthrough lines must contain samples strictly
        inside the fan interval whose y-values equal ``concentration_at_point`` (the post-#316
        reader) at the same t. Before the fix the ``decaying_fan`` segment was silently skipped
        (and a deactivated rarefaction's extrapolated crossing plotted its c_start=10 fallback).
        """
        sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
        n = 40
        cin = np.array([0.0, 10.0, 10.0, 0.0] + [0.0] * (n - 4))
        flow = np.full(n, 100.0)
        tedges = pd.date_range("2020-01-01", periods=n + 1, freq="D")
        tracker = FrontTracker(cin=cin, flow=flow, tedges=tedges, aquifer_pore_volume=100.0, sorption=sorption)
        tracker.run()
        state = tracker.state

        dsw = next(w for w in state.waves if isinstance(w, DecayingShockWave))
        theta_cross = dsw.outlet_crossing_theta(state.v_outlet)
        assert theta_cross is not None
        t_cross = state.t_at_theta(theta_cross)
        t_end = float((state.tedges[-1] - state.tedges[0]) / pd.Timedelta(days=1))
        assert t_cross < t_end

        fig, ax = plt.subplots()
        plot_breakthrough_curve(state, ax=ax)

        xy = np.concatenate([np.column_stack([ln.get_xdata(), ln.get_ydata()]).astype(float) for ln in ax.get_lines()])
        inside = xy[(xy[:, 0] > t_cross + 1e-6) & (xy[:, 0] < t_end - 1e-6)]
        assert len(inside) >= 5, "no rendered samples inside the decaying-fan interval"
        expected = np.array([
            concentration_at_point(state.v_outlet, float(state.theta_at_t(t)), state.waves, state.sorption)
            for t in inside[:, 0]
        ])
        assert np.all(expected > 0.0)  # the fan interval carries nonzero concentration
        np.testing.assert_allclose(inside[:, 1], expected, rtol=1e-9)

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


def _structure_from_tracker(tracker):
    """Minimal ``structure`` dict mirroring infiltration_to_extraction_nonlinear_sorption output."""
    return {
        "tracker_state": tracker.state,
        "theta_first_arrival": tracker.theta_first_arrival,
        "waves": tracker.state.waves,
        "events": tracker.state.events,
    }


class TestVerifyPhysicsSpinupMaskTimeOrigin:
    """verify_physics check 5 must build its spin-up mask on the *shared* time origin.

    The mask maps ``cout_tedges`` to θ via ``tracker_state.theta_at_t``, which expects
    days measured from ``tracker_state.tedges[0]`` (the input origin). If the output grid
    is offset from the input origin and the mask is built with each grid's own first edge
    as reference, the after-spin-up window is shifted earlier and NaNs that fall after the
    true first arrival are silently excluded.
    """

    def test_nan_after_spinup_detected_in_offset_output_window(self, freundlich_favorable):
        # Small step so the mass-balance check integrates a single shock cheaply.
        tedges = pd.date_range("2020-01-01", periods=9, freq="10D")
        cin = np.array([0.0] + [10.0] * 7)
        flow = np.full(8, 100.0)
        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=50.0,
            sorption=freundlich_favorable,
        )
        tracker.run(max_iterations=500)
        state = tracker.state

        theta_first = tracker.theta_first_arrival
        t_first = state.t_at_theta(theta_first)
        assert np.isfinite(t_first)

        # Output grid entirely AFTER first arrival but OFFSET from the input origin.
        offset_days = t_first + 20.0
        cout_tedges = pd.date_range(state.tedges[0] + pd.Timedelta(days=offset_days), periods=4, freq="5D")
        cout = np.full(len(cout_tedges) - 1, 10.0)
        cout[0] = np.nan  # NaN in the first (genuinely after-spin-up) output bin.

        structure = {
            "waves": state.waves,
            "theta_first_arrival": theta_first,
            "events": state.events,
            "tracker_state": state,
        }
        results = verify_physics(structure, cout, cout_tedges, cin, verbose=False)
        check5 = next(c for c in results["checks"] if c["name"] == "No NaN after spin-up")
        assert not check5["passed"], "NaN after spin-up in an offset output window must be detected"


class TestSummaryOverlayTimeOrigin:
    """The bin-averaged outlet overlay must share the exact curve's time origin (tedges[0])."""

    def test_bin_averaged_overlay_starts_at_shared_origin(self, tracker_step):
        state = tracker_step.state
        structure = _structure_from_tracker(tracker_step)

        # Output grid offset 100 days past the input origin.
        offset_days = 100.0
        cout_tedges = pd.date_range(state.tedges[0] + pd.Timedelta(days=offset_days), periods=6, freq="10D")
        cout = np.full(len(cout_tedges) - 1, 5.0)

        fig, axes = plot_front_tracking_summary(
            structure,
            state.tedges,
            state.cin,
            cout_tedges,
            cout,
            show_exact=False,
        )
        binned = [ln for ln in axes["outlet"].get_lines() if ln.get_label() == "Bin-averaged outlet"]
        assert binned, "bin-averaged overlay line not found"
        xmin = float(np.asarray(binned[0].get_xdata(), dtype=float).min())
        # Left edge sits at the offset when measured from the shared origin, not at 0.
        np.testing.assert_allclose(xmin, offset_days, atol=1e-9)
        plt.close(fig)


class TestPlotSummaryAndComparisonSmoke:
    """Multi-panel composites render and return the documented figure/axes containers."""

    def test_plot_front_tracking_summary_returns_expected_axes(self, tracker_step):
        structure = _structure_from_tracker(tracker_step)
        cout_tedges = tracker_step.state.tedges
        cout = np.full(len(cout_tedges) - 1, 5.0)

        fig, axes = plot_front_tracking_summary(
            structure,
            tracker_step.state.tedges,
            tracker_step.state.cin,
            cout_tedges,
            cout,
        )

        assert isinstance(fig, Figure)
        assert set(axes.keys()) == {"vt", "inlet", "outlet"}
        plt.close(fig)

    def test_plot_sorption_comparison_returns_2x3_axes(self, tracker_pulse):
        structure = _structure_from_tracker(tracker_pulse)
        tedges = tracker_pulse.state.tedges
        cin = tracker_pulse.state.cin

        fig, axes = plot_sorption_comparison(
            structure,
            structure,
            tedges,
            cin,
            structure,
            structure,
            tedges,
            cin,
        )

        assert isinstance(fig, Figure)
        assert axes.shape == (2, 3)
        plt.close(fig)
