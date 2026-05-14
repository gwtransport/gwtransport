"""Tests for `gwtransport.examples` synthetic data generators."""

import numpy as np
import pandas as pd
import pytest

from gwtransport.examples import (
    generate_ec_example_data,
    generate_example_data,
    generate_example_deposition_timeseries,
    generate_temperature_example_data,
)

GENERATORS = [
    generate_example_data,
    generate_temperature_example_data,
    generate_ec_example_data,
    generate_example_deposition_timeseries,
]


@pytest.mark.parametrize("generator", GENERATORS, ids=lambda g: g.__name__)
def test_each_public_generator_runs_with_defaults(generator):
    """Smoke test: every public generator must run with default kwargs."""
    df, tedges = generator()
    assert df is not None
    assert tedges is not None


@pytest.mark.parametrize("generator", GENERATORS, ids=lambda g: g.__name__)
def test_generated_arrays_match_bin_edge_convention(generator):
    """tedges has one more element than the data series and is a DatetimeIndex."""
    df, tedges = generator()
    assert isinstance(tedges, pd.DatetimeIndex)
    assert len(tedges) == len(df) + 1


def test_event_dates_outside_range_raises():
    """Out-of-range event_dates must raise ValueError rather than silently mapping to the nearest edge."""
    with pytest.raises(ValueError, match="outside the dates range"):
        generate_example_deposition_timeseries(event_dates=["2099-01-01"])


def test_aquifer_pore_volumes_and_gamma_mutually_exclusive():
    """Passing both discrete pore volumes and gamma parameters must raise ValueError."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        generate_example_data(
            aquifer_pore_volumes=np.array([100.0, 200.0]),
            aquifer_pore_volume_gamma_mean=500.0,
        )


def test_event_decay_scale_zero_raises():
    """event_decay_scale=0 produced silent NaN before; pin the new explicit ValueError."""
    with pytest.raises(ValueError, match="event_decay_scale must be positive"):
        generate_example_deposition_timeseries(event_decay_scale=0.0)


def test_event_duration_zero_raises():
    """event_duration=0 has no meaningful behaviour; pin the explicit ValueError."""
    with pytest.raises(ValueError, match="event_duration must be positive"):
        generate_example_deposition_timeseries(event_duration=0)


def _event_loop_reference(*, dates, event_dates_index, event_magnitude, event_duration, event_decay_scale):
    """Reference implementation: the per-event Python loop the vectorization replaced."""
    n_dates = len(dates)
    event = np.zeros(n_dates)
    for event_date in event_dates_index:
        event_idx = dates.get_indexer([event_date], method="nearest")[0]
        event_indices = np.arange(event_idx, min(event_idx + event_duration, n_dates))
        decay_pattern = event_magnitude * np.exp(-np.arange(len(event_indices)) / event_decay_scale)
        event[event_indices] += decay_pattern
    return event


@pytest.mark.parametrize("event_duration", [5, 30, 90])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_vectorized_event_accumulation_matches_loop(seed, event_duration):
    """Byte-for-byte equality of the vectorised event-decay block versus the original loop.

    Isolates the event component by zeroing base / seasonal / noise / non-negative clipping.
    Includes events near the end of the series (to exercise the boundary clip) and overlapping
    events (to exercise ``np.add.at``'s repeat-index accumulation).
    """
    date_start = "2018-01-01"
    date_end = "2018-06-30"

    # Mix of in-range, near-end, and overlapping event dates.
    event_dates = ["2018-01-05", "2018-03-15", "2018-03-16", "2018-06-25"]

    actual_series, _ = generate_example_deposition_timeseries(
        date_start=date_start,
        date_end=date_end,
        base=0.0,
        seasonal_amplitude=0.0,
        noise_scale=0.0,
        event_dates=event_dates,
        event_duration=event_duration,
        event_decay_scale=10.0,
        event_magnitude=3.0,
        ensure_non_negative=False,
        rng=seed,
    )

    dates = pd.date_range(date_start, date_end, freq="D").tz_localize("UTC")
    event_dates_index = pd.DatetimeIndex(pd.to_datetime(event_dates)).tz_localize(dates.tz)
    expected = _event_loop_reference(
        dates=dates,
        event_dates_index=event_dates_index,
        event_magnitude=3.0,
        event_duration=event_duration,
        event_decay_scale=10.0,
    )

    np.testing.assert_array_equal(actual_series.to_numpy(), expected)
