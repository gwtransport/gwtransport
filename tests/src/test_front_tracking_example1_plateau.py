"""Regression-style check for Example 1 breakthrough plateau.

This test mirrors the Example 1 setup in
`examples/08_Front_Tracking_Exact_Solution.ipynb` and verifies that the
front-tracking solver produces a non-trivial outlet breakthrough curve
with a visible plateau near the inlet step concentration (10.0).

The goal is not to enforce an exact analytical shape here (that is
covered by lower-level tests), but to guard against regressions where
almost all outlet bins remain zero due to parameter choices or
bin-averaging issues.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gwtransport.advection import infiltration_to_extraction_front_tracking_detailed
from gwtransport.utils import compute_time_edges


def test_example1_breakthrough_has_plateau() -> None:
    """Example 1 should yield a non-trivial outlet plateau near 10.0.

    We reproduce the Example 1 configuration used in the notebook, with
    parameters chosen so that the outlet sees a substantial portion of
    the step input within the 150-day output window.
    """

    # Inlet time grid and step input
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    cin = np.zeros(len(dates))
    cin[5:] = 10.0

    flow = np.full(len(dates), 100.0)

    # Aquifer and sorption parameters chosen for an early, visible plateau
    aquifer_pore_volume = 50.0
    freundlich_k = 0.001
    freundlich_n = 2.0
    bulk_density = 1500.0
    porosity = 0.3

    # Output grid
    cout_dates = pd.date_range(start=dates[0], periods=150, freq="D")
    cout_tedges = compute_time_edges(tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates))

    cout, structure = infiltration_to_extraction_front_tracking_detailed(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=np.array([aquifer_pore_volume]),
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
    )

    # Basic sanity: some bins must be non-zero after first arrival
    t_first = structure[0]["t_first_arrival"]
    cout_tedges_days = ((cout_tedges - cout_tedges[0]) / pd.Timedelta(days=1)).values
    mask_after = cout_tedges_days[:-1] >= t_first
    cout_after = cout[mask_after]

    assert cout_after.size > 0
    assert np.any(cout_after > 0.0)

    # Plateau property: a subset of bins far after first arrival should
    # cluster around the inlet step value 10.0. We take bins starting
    # 20 days after first arrival as "late-time".
    late_mask = cout_tedges_days[:-1] >= (t_first + 20.0)
    cout_late = cout[late_mask]

    # There should be several late-time bins to assess a plateau.
    assert cout_late.size >= 5

    # Their mean should be close to 10.0, well within a tight tolerance
    # given the exact analytical solver.
    mean_late = float(np.mean(cout_late))
    assert np.isclose(mean_late, 10.0, rtol=1e-3, atol=1e-3)

    # And their spread should be small compared to the mean, indicating
    # a flat plateau rather than a transient tail.
    std_late = float(np.std(cout_late))
    assert std_late <= 0.1
