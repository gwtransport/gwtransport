"""Tests for :mod:`gwtransport.arrival` against hand-derived crossing times.

All expected timestamps are derived by hand from the bin-center attribution:
bin centers of daily bins starting 2020-01-01 fall at 12:00 of each day, and a
linear interpolation between two straddling centers is computed analytically.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from gwtransport.arrival import first_arrival, peak_concentration, time_to_peak, time_to_threshold

TEDGES_DAILY = pd.date_range("2020-01-01", periods=5, freq="D")  # 4 daily bins
RAMP = np.array([0.0, 0.0, 2.0, 4.0])  # centers: 01-01 12:00, 01-02 12:00, 01-03 12:00, 01-04 12:00


def test_module_importable_from_package():
    arrival = importlib.import_module("gwtransport.arrival")

    assert callable(arrival.first_arrival)


def test_peak_concentration_is_max():
    np.testing.assert_allclose(peak_concentration(cout=RAMP), 4.0)


def test_peak_concentration_ignores_nan():
    np.testing.assert_allclose(peak_concentration(cout=[np.nan, 5.0, 2.0]), 5.0)


def test_time_to_peak_is_last_bin_center():
    # Maximum (4.0) sits in the last daily bin [2020-01-04, 2020-01-05) -> center 2020-01-04 12:00.
    assert time_to_peak(cout=RAMP, tedges=TEDGES_DAILY) == pd.Timestamp("2020-01-04 12:00:00")


def test_time_to_peak_ignores_nan():
    tedges = pd.date_range("2020-01-01", periods=4, freq="D")
    assert time_to_peak(cout=[np.nan, 5.0, 2.0], tedges=tedges) == pd.Timestamp("2020-01-02 12:00:00")


def test_time_to_threshold_interpolates_within_bin():
    # Straddling centers: 2020-01-02 12:00 (cout=0) and 2020-01-03 12:00 (cout=2).
    # threshold=1 -> fraction (1-0)/(2-0) = 1/2 of the 24 h center spacing -> 2020-01-03 00:00.
    crossing = time_to_threshold(cout=RAMP, tedges=TEDGES_DAILY, threshold=1.0)
    np.testing.assert_allclose(crossing.value, pd.Timestamp("2020-01-03 00:00:00").value)


def test_time_to_threshold_uneven_bins():
    # Bin widths 1, 2, 4, 8 days -> centers 01-01 12:00, 01-03 00:00, 01-06 00:00, 01-12 00:00.
    tedges = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-08", "2020-01-16"])
    cout = [0.0, 1.0, 4.0, 2.0]
    # First bin above threshold=2 is bin 2; fraction (2-1)/(4-1) = 1/3 of the 3-day
    # spacing between centers 01-03 and 01-06 -> 2020-01-04 00:00.
    crossing = time_to_threshold(cout=cout, tedges=tedges, threshold=2.0)
    np.testing.assert_allclose(crossing.value, pd.Timestamp("2020-01-04 00:00:00").value)


def test_first_arrival_default_level_zero():
    # First bin strictly above 0 is bin 2; interpolation between (0, 2) at level 0
    # lands on the previous bin center, 2020-01-02 12:00.
    arrival_time = first_arrival(cout=RAMP, tedges=TEDGES_DAILY)
    np.testing.assert_allclose(arrival_time.value, pd.Timestamp("2020-01-02 12:00:00").value)


def test_first_arrival_matches_time_to_threshold():
    # Both share one crossing path: identical results for identical levels.
    assert first_arrival(cout=RAMP, tedges=TEDGES_DAILY, level=1.0) == time_to_threshold(
        cout=RAMP, tedges=TEDGES_DAILY, threshold=1.0
    )


def test_crossing_never_exceeded_returns_nat():
    assert time_to_threshold(cout=RAMP, tedges=TEDGES_DAILY, threshold=10.0) is pd.NaT
    assert first_arrival(cout=RAMP, tedges=TEDGES_DAILY, level=4.0) is pd.NaT  # strict: 4 > 4 is False


def test_crossing_in_first_bin_returns_left_edge():
    # cout is above the level throughout the first bin -> earliest time in the record.
    assert time_to_threshold(cout=[3.0, 4.0, 5.0, 6.0], tedges=TEDGES_DAILY, threshold=1.0) == TEDGES_DAILY[0]


def test_crossing_after_nan_returns_left_edge_of_exceeding_bin():
    # NaN in the preceding bin: no straddling pair, bin 2 is above throughout -> its left edge.
    crossing = time_to_threshold(cout=[0.0, np.nan, 2.0, 4.0], tedges=TEDGES_DAILY, threshold=1.0)
    assert crossing == TEDGES_DAILY[2]


def test_tedges_parity_validated():
    with pytest.raises(ValueError, match="tedges must have one more element than cout"):
        time_to_threshold(cout=RAMP, tedges=TEDGES_DAILY[:-1], threshold=1.0)
    with pytest.raises(ValueError, match="tedges must have one more element than cout"):
        time_to_peak(cout=RAMP, tedges=TEDGES_DAILY[:-1])
