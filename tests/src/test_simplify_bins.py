import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from gwtransport.utils import simplify_bins


class TestSimplifyBinsBasic:
    """Test basic merging behavior without flow."""

    def test_identical_values_merged(self):
        """All identical values should merge into a single bin."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([5.0, 5.0, 5.0, 5.0])
        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 4.0])
        assert_array_equal(new_values, [5.0])
        assert new_flow is None

    def test_all_different_values_no_merge(self):
        """Bins with large jumps and tol=0 should not merge."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 3.0, 1.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, edges)
        assert_array_equal(new_values, values)

    def test_partial_merge(self):
        """Adjacent identical bins merge, others stay."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([1.0, 1.0, 5.0, 5.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 2.0, 4.0])
        assert_array_equal(new_values, [1.0, 5.0])

    def test_single_bin(self):
        """A single bin should be returned unchanged."""
        edges = np.array([0.0, 1.0])
        values = np.array([7.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, edges)
        assert_array_equal(new_values, [7.0])

    def test_empty_input(self):
        """Empty values should return empty output."""
        edges = np.array([0.0])
        values = np.array([])
        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values)
        assert len(new_edges) == 1
        assert len(new_values) == 0
        assert new_flow is None


class TestSimplifyBinsTolerance:
    """Test tolerance-based merging."""

    def test_merge_within_tolerance(self):
        """Values within tol should be merged."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 1.05, 0.98])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values, tol=0.1)
        assert_array_equal(new_edges, [0.0, 3.0])
        assert_array_equal(new_values, [np.mean(values)])

    def test_tolerance_splits_correctly(self):
        """Groups exceeding tol should be split at the largest jump."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([1.0, 1.0, 5.0, 5.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values, tol=0.5)
        assert_array_equal(new_edges, [0.0, 2.0, 4.0])
        assert_array_equal(new_values, [1.0, 5.0])


class TestSimplifyBinsWidthWeighted:
    """Test width-weighted averaging (no flow)."""

    def test_unequal_widths(self):
        """Merged value should be width-weighted, not arithmetic mean."""
        edges = np.array([0.0, 3.0, 4.0])
        values = np.array([2.0, 2.0])
        _, new_values, _ = simplify_bins(edges=edges, values=values)
        # Both values identical, so average is 2.0 regardless of width
        assert_array_equal(new_values, [2.0])

    def test_unequal_widths_different_values_with_tol(self):
        """Width-weighted average for merged bins with different widths."""
        edges = np.array([0.0, 3.0, 4.0])  # widths: 3, 1
        values = np.array([1.0, 5.0])
        _, new_values, _ = simplify_bins(edges=edges, values=values, tol=10.0)
        # Width-weighted: (3*1 + 1*5) / (3+1) = 8/4 = 2.0
        assert_array_equal(new_values, [2.0])


class TestSimplifyBinsFlowWeighted:
    """Test flow-weighted (volume-weighted) averaging."""

    def test_uniform_flow_equals_width_weighted(self):
        """Uniform flow should give same result as no flow."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 1.0, 1.0])
        flow = np.array([2.0, 2.0, 2.0])
        _, new_values_no_flow, _ = simplify_bins(edges=edges, values=values)
        _, new_values_flow, _ = simplify_bins(edges=edges, values=values, flow=flow)
        assert_array_equal(new_values_flow, new_values_no_flow)

    def test_flow_weighted_average(self):
        """Flow-weighted average should account for volume = flow x width."""
        edges = np.array([0.0, 1.0, 2.0])  # widths: 1, 1
        values = np.array([2.0, 8.0])
        flow = np.array([3.0, 1.0])
        # volumes: 3*1=3, 1*1=1. weighted avg: (3*2 + 1*8) / (3+1) = 14/4 = 3.5
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values, flow=flow, tol=10.0)
        assert_array_equal(new_edges, [0.0, 2.0])
        assert_array_equal(new_values, [3.5])

    def test_flow_and_width_combined(self):
        """Volume weighting uses flow x width as the weight."""
        edges = np.array([0.0, 2.0, 3.0])  # widths: 2, 1
        values = np.array([1.0, 4.0])
        flow = np.array([3.0, 6.0])
        # volumes: 3*2=6, 6*1=6. weighted avg: (6*1 + 6*4) / (6+6) = 30/12 = 2.5
        _, new_values, _ = simplify_bins(edges=edges, values=values, flow=flow, tol=10.0)
        assert_array_equal(new_values, [2.5])

    def test_flow_does_not_affect_splitting(self):
        """Splitting is based on values only, not on flow."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 10.0, 1.0])
        flow = np.array([100.0, 0.01, 100.0])
        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
        # All values differ, so no merging occurs
        assert_array_equal(new_edges, edges)
        assert_array_equal(new_values, values)
        assert_array_equal(new_flow, flow)


class TestSimplifyBinsFlowSimplification:
    """Test that flow is simplified with time-weighted averaging."""

    def test_flow_time_weighted_equal_widths(self):
        """Merged flow should be time-weighted average with equal bin widths."""
        edges = np.array([0.0, 1.0, 2.0])  # widths: 1, 1
        values = np.array([5.0, 5.0])
        flow = np.array([2.0, 8.0])
        # time-weighted: (1*2 + 1*8) / (1+1) = 10/2 = 5.0
        _, _, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
        assert_array_equal(new_flow, [5.0])

    def test_flow_time_weighted_unequal_widths(self):
        """Merged flow should be time-weighted average with unequal bin widths."""
        edges = np.array([0.0, 3.0, 4.0])  # widths: 3, 1
        values = np.array([5.0, 5.0])
        flow = np.array([2.0, 6.0])
        # time-weighted: (3*2 + 1*6) / (3+1) = 12/4 = 3.0
        _, _, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
        assert_array_equal(new_flow, [3.0])

    def test_flow_no_merge_unchanged(self):
        """Flow should be unchanged when no bins are merged."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 5.0, 1.0])
        flow = np.array([2.0, 4.0, 6.0])
        _, _, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
        assert_array_equal(new_flow, flow)

    def test_flow_none_when_not_provided(self):
        """Returned flow should be None when flow is not provided."""
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 1.0])
        _, _, new_flow = simplify_bins(edges=edges, values=values)
        assert new_flow is None

    def test_flow_volume_conservation(self):
        """Total volume (flow x width) should be conserved after simplification."""
        edges = np.array([0.0, 1.0, 3.0, 4.0, 7.0, 8.0])
        values = np.array([2.0, 2.0, 5.0, 5.0, 5.0])
        flow = np.array([3.0, 1.0, 4.0, 2.0, 6.0])
        widths = np.diff(edges)
        volume_before = np.sum(flow * widths)
        new_edges, _, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
        new_widths = np.diff(new_edges)
        volume_after = np.sum(new_flow * new_widths)
        assert_array_equal(volume_after, volume_before)

    def test_flow_mass_conservation(self):
        """Total mass (flow x width x value) should be conserved."""
        edges = np.array([0.0, 2.0, 3.0, 5.0, 6.0])
        values = np.array([3.0, 3.0, 7.0, 7.0])
        flow = np.array([2.0, 4.0, 1.0, 3.0])
        widths = np.diff(edges)
        mass_before = np.sum(flow * widths * values)
        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
        new_widths = np.diff(new_edges)
        mass_after = np.sum(new_flow * new_widths * new_values)
        assert_array_equal(mass_after, mass_before)


class TestSimplifyBinsMassConservation:
    """Test that total mass is conserved after simplification."""

    def test_mass_conservation_no_flow(self):
        """Total widthxvalue should be conserved."""
        edges = np.array([0.0, 1.0, 3.0, 4.0, 7.0, 8.0])
        values = np.array([2.0, 2.0, 5.0, 5.0, 5.0])
        widths = np.diff(edges)
        mass_before = np.sum(widths * values)
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        new_widths = np.diff(new_edges)
        mass_after = np.sum(new_widths * new_values)
        assert_array_equal(mass_after, mass_before)

    def test_mass_conservation_with_flow(self):
        """Total flowxwidthxvalue (= mass) should be conserved."""
        edges = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
        values = np.array([3.0, 3.0, 7.0, 7.0])
        flow = np.array([2.0, 4.0, 1.0, 3.0])
        widths = np.diff(edges)
        mass_before = np.sum(flow * widths * values)
        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
        new_widths = np.diff(new_edges)
        mass_after = np.sum(new_flow * new_widths * new_values)
        assert_array_equal(mass_after, mass_before)

    def test_mass_conservation_with_tolerance(self):
        """Mass conservation should hold even when merging with tolerance."""
        rng = np.random.default_rng(42)
        edges = np.arange(11, dtype=float)
        values = rng.uniform(0, 1, size=10)
        flow = rng.uniform(0.5, 2.0, size=10)
        widths = np.diff(edges)
        mass_before = np.sum(flow * widths * values)

        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow, tol=0.3)
        new_widths = np.diff(new_edges)
        mass_after = np.sum(new_flow * new_widths * new_values)
        # Summation order differs after merging, so allow machine-precision tolerance
        assert_allclose(mass_after, mass_before, atol=0.0, rtol=float(np.finfo(float).eps))


class TestSimplifyBinsEdgeCases:
    """Test edge cases and special inputs."""

    def test_two_bins_identical(self):
        """Two identical bins should merge."""
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([3.0, 3.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 2.0])
        assert_array_equal(new_values, [3.0])

    def test_two_bins_different(self):
        """Two different bins should not merge with tol=0."""
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 2.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, edges)
        assert_array_equal(new_values, values)

    def test_datetime_edges(self):
        """Function should work with pandas DatetimeIndex edges."""
        edges = pd.date_range("2020-01-01", periods=5, freq="D")
        values = np.array([1.0, 1.0, 5.0, 5.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        assert isinstance(new_edges, pd.DatetimeIndex)
        assert len(new_edges) == 3
        assert new_edges[0] == edges[0]
        assert new_edges[-1] == edges[-1]
        assert_array_equal(new_values, [1.0, 5.0])

    def test_ndarray_edges_preserved(self):
        """Numeric ndarray edges should return ndarray."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 1.0, 1.0])
        new_edges, _, _ = simplify_bins(edges=edges, values=values)
        assert isinstance(new_edges, np.ndarray)

    def test_direction_independence(self):
        """Result should be the same regardless of which side has more bins.

        The recursive split approach ensures direction independence by always
        splitting at the largest jump.
        """
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        values = np.array([1.0, 1.0, 1.0, 5.0, 5.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 3.0, 5.0])
        assert_array_equal(new_values, [1.0, 5.0])
