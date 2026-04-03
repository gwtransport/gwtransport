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
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 4.0])
        assert_array_equal(new_values, [5.0])

    def test_all_different_values_no_merge(self):
        """Bins with large jumps and tol=0 should not merge."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 3.0, 1.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, edges)
        assert_array_equal(new_values, values)

    def test_partial_merge(self):
        """Adjacent identical bins merge, others stay."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([1.0, 1.0, 5.0, 5.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 2.0, 4.0])
        assert_array_equal(new_values, [1.0, 5.0])

    def test_single_bin(self):
        """A single bin should be returned unchanged."""
        edges = np.array([0.0, 1.0])
        values = np.array([7.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, edges)
        assert_array_equal(new_values, [7.0])

    def test_empty_input(self):
        """Empty values should return empty output."""
        edges = np.array([0.0])
        values = np.array([])
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert len(new_edges) == 1
        assert len(new_values) == 0


class TestSimplifyBinsTolerance:
    """Test tolerance-based merging."""

    def test_merge_within_tolerance(self):
        """Values within tol should be merged."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 1.05, 0.98])
        new_edges, new_values = simplify_bins(edges=edges, values=values, tol=0.1)
        assert_array_equal(new_edges, [0.0, 3.0])
        assert_array_equal(new_values, [np.mean(values)])

    def test_tolerance_splits_correctly(self):
        """Groups exceeding tol should be split at the largest jump."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([1.0, 1.0, 5.0, 5.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values, tol=0.5)
        assert_array_equal(new_edges, [0.0, 2.0, 4.0])
        assert_array_equal(new_values, [1.0, 5.0])


class TestSimplifyBinsWidthWeighted:
    """Test width-weighted averaging (no flow)."""

    def test_unequal_widths(self):
        """Merged value should be width-weighted, not arithmetic mean."""
        edges = np.array([0.0, 3.0, 4.0])
        values = np.array([2.0, 2.0])
        _, new_values = simplify_bins(edges=edges, values=values)
        # Both values identical, so average is 2.0 regardless of width
        assert_array_equal(new_values, [2.0])

    def test_unequal_widths_different_values_with_tol(self):
        """Width-weighted average for merged bins with different widths."""
        edges = np.array([0.0, 3.0, 4.0])  # widths: 3, 1
        values = np.array([1.0, 5.0])
        _, new_values = simplify_bins(edges=edges, values=values, tol=10.0)
        # Width-weighted: (3*1 + 1*5) / (3+1) = 8/4 = 2.0
        assert_array_equal(new_values, [2.0])


class TestSimplifyBinsFlowWeighted:
    """Test flow-weighted (volume-weighted) averaging."""

    def test_uniform_flow_equals_width_weighted(self):
        """Uniform flow should give same result as no flow."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 1.0, 1.0])
        flow = np.array([2.0, 2.0, 2.0])
        _, new_values_no_flow = simplify_bins(edges=edges, values=values)
        _, new_values_flow = simplify_bins(edges=edges, values=values, flow=flow)
        assert_array_equal(new_values_flow, new_values_no_flow)

    def test_flow_weighted_average(self):
        """Flow-weighted average should account for volume = flow x width."""
        edges = np.array([0.0, 1.0, 2.0])  # widths: 1, 1
        values = np.array([2.0, 8.0])
        flow = np.array([3.0, 1.0])
        # volumes: 3*1=3, 1*1=1. weighted avg: (3*2 + 1*8) / (3+1) = 14/4 = 3.5
        new_edges, new_values = simplify_bins(edges=edges, values=values, flow=flow, tol=10.0)
        assert_array_equal(new_edges, [0.0, 2.0])
        assert_array_equal(new_values, [3.5])

    def test_flow_and_width_combined(self):
        """Volume weighting uses flow x width as the weight."""
        edges = np.array([0.0, 2.0, 3.0])  # widths: 2, 1
        values = np.array([1.0, 4.0])
        flow = np.array([3.0, 6.0])
        # volumes: 3*2=6, 6*1=6. weighted avg: (6*1 + 6*4) / (6+6) = 30/12 = 2.5
        _, new_values = simplify_bins(edges=edges, values=values, flow=flow, tol=10.0)
        assert_array_equal(new_values, [2.5])

    def test_flow_does_not_affect_splitting(self):
        """Splitting is based on values only, not on flow."""
        edges = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 10.0, 1.0])
        flow = np.array([100.0, 0.01, 100.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values, flow=flow)
        # All values differ, so no merging occurs
        assert_array_equal(new_edges, edges)
        assert_array_equal(new_values, values)


class TestSimplifyBinsMassConservation:
    """Test that total mass is conserved after simplification."""

    def test_mass_conservation_no_flow(self):
        """Total widthxvalue should be conserved."""
        edges = np.array([0.0, 1.0, 3.0, 4.0, 7.0, 8.0])
        values = np.array([2.0, 2.0, 5.0, 5.0, 5.0])
        widths = np.diff(edges)
        mass_before = np.sum(widths * values)
        new_edges, new_values = simplify_bins(edges=edges, values=values)
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
        new_edges, new_values = simplify_bins(edges=edges, values=values, flow=flow)
        # Recompute mass using merged bins
        s = np.searchsorted(edges, new_edges[:-1])
        new_volumes = np.add.reduceat(flow * widths, s)
        mass_after = np.sum(new_volumes * new_values)
        assert_array_equal(mass_after, mass_before)  # Exact: same summation order

    def test_mass_conservation_with_tolerance(self):
        """Mass conservation should hold even when merging with tolerance."""
        rng = np.random.default_rng(42)
        edges = np.arange(11, dtype=float)
        values = rng.uniform(0, 1, size=10)
        flow = rng.uniform(0.5, 2.0, size=10)
        widths = np.diff(edges)
        mass_before = np.sum(flow * widths * values)

        new_edges, new_values = simplify_bins(edges=edges, values=values, flow=flow, tol=0.3)
        # Mass via the simplified series: for each merged bin, the volume-weighted
        # average preserves total mass by construction
        s = np.searchsorted(edges, new_edges[:-1])
        new_volumes = np.add.reduceat(flow * widths, s)
        mass_after = np.sum(new_volumes * new_values)
        # Summation order differs after merging, so allow machine-precision tolerance
        assert_allclose(mass_after, mass_before, atol=0.0, rtol=np.finfo(float).eps)


class TestSimplifyBinsEdgeCases:
    """Test edge cases and special inputs."""

    def test_two_bins_identical(self):
        """Two identical bins should merge."""
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([3.0, 3.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 2.0])
        assert_array_equal(new_values, [3.0])

    def test_two_bins_different(self):
        """Two different bins should not merge with tol=0."""
        edges = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 2.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, edges)
        assert_array_equal(new_values, values)

    def test_datetime_edges(self):
        """Function should work with pandas DatetimeIndex edges."""
        edges = pd.date_range("2020-01-01", periods=5, freq="D")
        values = np.array([1.0, 1.0, 5.0, 5.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert len(new_edges) == 3
        assert new_edges[0] == edges[0]
        assert new_edges[-1] == edges[-1]
        assert_array_equal(new_values, [1.0, 5.0])

    def test_direction_independence(self):
        """Result should be the same regardless of which side has more bins.

        The recursive split approach ensures direction independence by always
        splitting at the largest jump.
        """
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        values = np.array([1.0, 1.0, 1.0, 5.0, 5.0])
        new_edges, new_values = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 3.0, 5.0])
        assert_array_equal(new_values, [1.0, 5.0])
