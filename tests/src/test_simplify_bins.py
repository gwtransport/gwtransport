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
        """Known-answer merge pins the split boundary AND the time-weighted flow per bin.

        The value series ``[2, 2, 5, 5, 5]`` merges only at the single 2->5 jump (edge 3.0),
        which is NOT a midpoint: a wrong split direction would land elsewhere and fail the
        pinned ``new_edges``. The merged flows are the hand-derived time-weighted averages:
        group 1 widths (1, 2) flow (3, 1) -> (1*3 + 2*1) / 3 = 5/3; group 2 widths (1, 3, 1)
        flow (4, 2, 6) -> (1*4 + 3*2 + 1*6) / 5 = 16/5.
        """
        edges = np.array([0.0, 1.0, 3.0, 4.0, 7.0, 8.0])
        values = np.array([2.0, 2.0, 5.0, 5.0, 5.0])
        flow = np.array([3.0, 1.0, 4.0, 2.0, 6.0])
        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow)
        assert_array_equal(new_edges, [0.0, 3.0, 8.0])
        assert_array_equal(new_values, [2.0, 5.0])
        assert_array_equal(new_flow, [5.0 / 3.0, 16.0 / 5.0])
        # Conservation follows from the literals: total volume is preserved.
        widths = np.diff(edges)
        new_widths = np.diff(new_edges)
        assert_allclose(np.sum(new_flow * new_widths), np.sum(flow * widths), atol=0.0, rtol=float(np.finfo(float).eps))


class TestSimplifyBinsMassConservation:
    """Test that total mass is conserved after simplification."""

    def test_mass_conservation_no_flow(self):
        """Known-answer width-weighted merge: the split boundary and merged values are pinned.

        ``[2, 2, 5, 5, 5]`` merges only at the 2->5 jump (edge 3.0). Within each group all
        values are identical, so the width-weighted averages are exactly 2 and 5 -- a wrong
        split would mix the 2s and 5s and change these literals.
        """
        edges = np.array([0.0, 1.0, 3.0, 4.0, 7.0, 8.0])
        values = np.array([2.0, 2.0, 5.0, 5.0, 5.0])
        new_edges, new_values, _ = simplify_bins(edges=edges, values=values)
        assert_array_equal(new_edges, [0.0, 3.0, 8.0])
        assert_array_equal(new_values, [2.0, 5.0])
        # Conservation follows from the literals: total width x value is preserved.
        assert_array_equal(np.sum(np.diff(new_edges) * new_values), np.sum(np.diff(edges) * values))

    def test_mass_conservation_with_flow(self):
        """Total flowxwidthxvalue (= mass) is conserved when a merge group mixes values.

        Values differ within the (tol-merged) group and flow vs width pull the weighting
        in opposite directions: volumes are 6*1=6 and 1*2=2, so the flow-weighted merged
        value is (6*2 + 2*8)/(6+2) = 3.5, distinct from the width-only weighting (6.0) and
        the unweighted mean (5.0). Mass conservation therefore exercises the flow-vs-width
        value-weighting rather than merging only identical values.
        """
        edges = np.array([0.0, 1.0, 3.0])  # widths: 1, 2
        values = np.array([2.0, 8.0])
        flow = np.array([6.0, 1.0])
        widths = np.diff(edges)
        mass_before = np.sum(flow * widths * values)
        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow, tol=10.0)
        # Single merged bin with the flow-volume-weighted value.
        assert_array_equal(new_edges, [0.0, 3.0])
        assert_array_equal(new_values, [3.5])
        new_widths = np.diff(new_edges)
        mass_after = np.sum(new_flow * new_widths * new_values)
        assert_array_equal(mass_after, mass_before)

    def test_mass_conservation_with_tolerance(self):
        """Known-answer tolerance merge: the split boundaries are pinned, not just conserved.

        Pinning ``new_edges`` to the literal merge boundaries makes the test sensitive to the
        split algorithm: a wrong split would merge a different set of bins and fail here. The
        merged values/flows are then checked against an independent per-group weighted average
        recomputed from the input literals (so the result is not compared against itself).
        """
        rng = np.random.default_rng(42)
        edges = np.arange(11, dtype=float)
        values = rng.uniform(0, 1, size=10)
        flow = rng.uniform(0.5, 2.0, size=10)
        widths = np.diff(edges)

        new_edges, new_values, new_flow = simplify_bins(edges=edges, values=values, flow=flow, tol=0.3)
        assert new_flow is not None  # flow supplied -> flow_out is an array

        # Pin the (non-obvious) merge boundaries: only bins 2-3 and 5-6-7 merge under tol=0.3.
        assert_array_equal(new_edges, [0.0, 1.0, 2.0, 4.0, 5.0, 8.0, 9.0, 10.0])

        # Independent per-group reference (volume-weighted value, time-weighted flow).
        groups = [[0], [1], [2, 3], [4], [5, 6, 7], [8], [9]]
        expected_values = np.array([
            np.sum(flow[g] * widths[g] * values[g]) / np.sum(flow[g] * widths[g]) for g in groups
        ])
        expected_flow = np.array([np.sum(flow[g] * widths[g]) / np.sum(widths[g]) for g in groups])
        assert_allclose(new_values, expected_values, atol=0.0, rtol=float(np.finfo(float).eps))
        assert_allclose(new_flow, expected_flow, atol=0.0, rtol=float(np.finfo(float).eps))

        # Mass (flow x width x value) is conserved up to summation-order rounding.
        mass_before = np.sum(flow * widths * values)
        mass_after = np.sum(new_flow * np.diff(new_edges) * new_values)
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


class TestSimplifyBinsIdempotenceAndEdgeAlignment:
    """Idempotence under repeated application and exact preservation of the first/last edges."""

    def test_idempotence_no_flow(self):
        """``simplify_bins(simplify_bins(x))`` equals ``simplify_bins(x)`` without flow."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0])
        values = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 2.0])
        new_edges_1, new_values_1, _ = simplify_bins(edges=edges, values=values, tol=0.1)
        new_edges_2, new_values_2, _ = simplify_bins(edges=new_edges_1, values=new_values_1, tol=0.1)
        assert_array_equal(new_edges_2, new_edges_1)
        assert_array_equal(new_values_2, new_values_1)

    def test_idempotence_with_flow(self):
        """``simplify_bins(simplify_bins(x))`` equals ``simplify_bins(x)`` with flow weighting."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0])
        values = np.array([1.0, 1.05, 5.0, 4.98, 5.02, 2.0])
        flow = np.array([3.0, 1.0, 2.0, 4.0, 1.5, 0.8])
        new_edges_1, new_values_1, new_flow_1 = simplify_bins(edges=edges, values=values, flow=flow, tol=0.1)
        new_edges_2, new_values_2, new_flow_2 = simplify_bins(
            edges=new_edges_1, values=new_values_1, flow=new_flow_1, tol=0.1
        )
        assert_array_equal(new_edges_2, new_edges_1)
        assert_array_equal(new_values_2, new_values_1)
        assert_array_equal(new_flow_2, new_flow_1)

    def test_edge_alignment_preserved_exactly_numeric(self):
        """First and last numeric edges are preserved bitwise, not just within tolerance."""
        edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([1.0, 1.0, 5.0, 5.0])
        new_edges, _, _ = simplify_bins(edges=edges, values=values)
        # Exact preservation: bitwise equality. ``assert_array_equal`` enforces this.
        assert_array_equal(new_edges[[0, -1]], edges[[0, -1]])

    def test_edge_alignment_preserved_exactly_with_flow(self):
        """First and last numeric edges are preserved bitwise even when flow drives merging."""
        edges = np.array([0.0, 1.5, 2.7, 3.9, 6.1])
        values = np.array([1.0, 1.0, 1.0, 1.0])
        flow = np.array([2.0, 3.0, 1.0, 4.0])
        new_edges, _, _ = simplify_bins(edges=edges, values=values, flow=flow)
        assert_array_equal(new_edges[[0, -1]], edges[[0, -1]])
