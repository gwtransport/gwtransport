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


def test_cin_method_constant_produces_constant_noiseless_signal():
    """cin_method='constant' must build cin around cin_mean (noiseless mean equals cin_mean)."""
    df, _ = generate_example_data(cin_method="constant", cin_mean=7.0, measurement_noise=0.0, rng=0)
    np.testing.assert_allclose(df["cin"].to_numpy(), 7.0)
    assert df.attrs["cin_method"] == "constant"


def test_unknown_cin_method_raises():
    """An unrecognised cin_method must raise ValueError."""
    with pytest.raises(ValueError, match="Unknown cin_method"):
        generate_example_data(cin_method="not_a_method")


@pytest.mark.parametrize(
    ("molecular_diffusivity", "longitudinal_dispersivity", "streamline_length"),
    [
        (0.05, None, None),
        (None, 1.0, None),
        (None, None, 100.0),
        (0.05, 1.0, None),
    ],
)
def test_partial_diffusion_parameters_raise(molecular_diffusivity, longitudinal_dispersivity, streamline_length):
    """Providing only some of the three diffusion parameters must raise ValueError."""
    with pytest.raises(ValueError, match="must all be provided together"):
        generate_example_data(
            molecular_diffusivity=molecular_diffusivity,
            longitudinal_dispersivity=longitudinal_dispersivity,
            streamline_length=streamline_length,
        )


def test_discrete_aquifer_pore_volumes_advection_path():
    """Discrete pore volumes (no diffusion) use the advection path and record the parameterization."""
    pore_volumes = np.array([500.0, 1000.0, 1500.0])
    df, tedges = generate_example_data(aquifer_pore_volumes=pore_volumes, rng=0)
    assert df.attrs["aquifer_pore_volume_parameterization"] == "discrete"
    np.testing.assert_array_equal(df.attrs["aquifer_pore_volumes"], pore_volumes)
    assert "aquifer_pore_volume_gamma_alpha" not in df.attrs
    assert len(df["cout"]) == len(df)
    assert len(tedges) == len(df) + 1


def test_discrete_aquifer_pore_volumes_diffusion_path():
    """Discrete pore volumes with diffusion parameters use the diffusion path."""
    pore_volumes = np.array([500.0, 1000.0, 1500.0])
    df, _ = generate_example_data(
        aquifer_pore_volumes=pore_volumes,
        molecular_diffusivity=0.05,
        longitudinal_dispersivity=1.0,
        streamline_length=100.0,
        rng=0,
    )
    assert df.attrs["aquifer_pore_volume_parameterization"] == "discrete"
    np.testing.assert_array_equal(df.attrs["aquifer_pore_volumes"], pore_volumes)
    assert df.attrs["molecular_diffusivity"] == 0.05
    assert len(df["cout"]) == len(df)


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


def test_deposition_seasonal_is_annual_for_nondaily_freq():
    """The deposition seasonal must have a one-year period for any ``freq``, not one year of samples.

    Isolates the seasonal component (base/noise/events zeroed) at weekly resolution and compares
    against the closed form ``seasonal_amplitude * sin(2*pi*elapsed_days/365.25)``. Before the fix the
    code used the sample index, so one year after the start the seasonal was ~0.78 instead of ~0.
    """
    date_start, date_end, freq = "2018-01-01", "2019-12-31", "W"
    seasonal_amplitude = 1.0

    series, _ = generate_example_deposition_timeseries(
        date_start=date_start,
        date_end=date_end,
        freq=freq,
        base=0.0,
        seasonal_amplitude=seasonal_amplitude,
        noise_scale=0.0,
        event_dates=[],
        ensure_non_negative=False,
        rng=0,
    )

    dates = pd.date_range(date_start, date_end, freq=freq).tz_localize("UTC")
    elapsed_days = (dates - dates[0]) / pd.Timedelta(days=1)
    expected = seasonal_amplitude * np.sin(2 * np.pi * np.asarray(elapsed_days) / 365.25)

    np.testing.assert_allclose(series.to_numpy(), expected)

    # One year after the start the annual seasonal must return to ~0 (the sample-index bug gave ~0.78).
    one_year = int(np.argmin(np.abs(np.asarray(elapsed_days) - 365.25)))
    assert abs(series.to_numpy()[one_year]) < 0.03


def test_flow_seasonal_varies_within_day_for_subdaily_freq():
    """Sub-daily flow must resolve the seasonal within a calendar day, not stair-step per integer day.

    With ``flow_noise=0`` and a short (spill-free) window the flow column equals the noiseless
    seasonal floored at 5 m3/day. Before the fix the integer day count held the seasonal constant
    across the four 6-hourly samples of each day; the closed form below distinguishes the two.
    """
    date_start, date_end, date_freq = "2020-01-01", "2020-01-05", "6h"
    flow_mean, flow_amplitude = 100.0, 30.0

    df, _ = generate_example_data(
        date_start=date_start,
        date_end=date_end,
        date_freq=date_freq,
        flow_mean=flow_mean,
        flow_amplitude=flow_amplitude,
        flow_noise=0.0,
        measurement_noise=0.0,
        rng=0,
    )

    dates = pd.date_range(date_start, date_end, freq=date_freq).tz_localize("UTC")
    frac_days = (dates - dates[0]) / pd.Timedelta(days=1)
    expected_flow = np.maximum(
        flow_mean + flow_amplitude * np.sin(2 * np.pi * np.asarray(frac_days) / 365 + np.pi), 5.0
    )

    np.testing.assert_allclose(df["flow"].to_numpy(), expected_flow)

    # The four 6-hourly samples of the first day must differ (the bug made them identical).
    assert np.unique(df["flow"].to_numpy()[:4]).size == 4
