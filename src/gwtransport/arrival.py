"""
Arrival metrics for extraction-concentration breakthrough curves.

This module reduces a modeled or observed extraction-concentration series (``cout``)
to the arrival metrics an operator asks for during a contamination event:

- :func:`first_arrival` -- earliest time the concentration rises above a detection level.
- :func:`time_to_threshold` -- first time the concentration crosses a given threshold
  (e.g. a regulatory limit).
- :func:`peak_concentration` -- maximum concentration of the series.
- :func:`time_to_peak` -- time at which the maximum occurs.

Bin-edge convention
-------------------
``cout`` follows the package-wide bin-edge pattern: ``n`` values that are flow-weighted
bin averages, constant over the intervals ``[tedges[i], tedges[i+1])`` defined by the
``n + 1`` edges in ``tedges``. For crossing-time estimates each bin average is attributed
to its bin center, and the crossing time is linearly interpolated between the two bin
centers that straddle the level -- it is not snapped to a bin edge. When the level is
already exceeded in the first bin, or the bin preceding the crossing holds NaN, no
straddling pair exists; the left edge of the exceeding bin is returned because under the
piecewise-constant convention the concentration is above the level throughout that bin.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.typing import NaTType

from gwtransport._validation import _validate_tedges_parity


def _crossing_time(*, cout: npt.ArrayLike, tedges: pd.DatetimeIndex, level: float) -> pd.Timestamp | NaTType:
    """Interpolate the first time ``cout`` rises strictly above ``level``.

    Parameters
    ----------
    cout : array-like
        Concentration series, constant per bin (n values). NaN entries are treated
        as not exceeding ``level``.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n values).
    level : float
        Level to cross.

    Returns
    -------
    Timestamp
        Crossing time; ``pd.NaT`` if ``cout`` never exceeds ``level``.
    """
    cout = np.asarray(cout, dtype=float)
    _validate_tedges_parity(tedges, cout, tedges_name="tedges", values_name="cout")

    above = cout > level  # NaN compares False
    if not above.any():
        return pd.NaT
    i = int(np.argmax(above))
    if i == 0 or not np.isfinite(cout[i - 1]):
        # No straddling pair to interpolate between; the whole bin i is above level.
        return tedges[i]
    centers = tedges[:-1] + (tedges[1:] - tedges[:-1]) / 2
    frac = (level - cout[i - 1]) / (cout[i] - cout[i - 1])
    return centers[i - 1] + frac * (centers[i] - centers[i - 1])


def first_arrival(*, cout: npt.ArrayLike, tedges: pd.DatetimeIndex, level: float = 0.0) -> pd.Timestamp | NaTType:
    """Earliest time the concentration rises strictly above a detection level.

    The crossing time is linearly interpolated between the centers of the bin that
    first exceeds ``level`` and the preceding bin (see module docstring for the
    bin-center attribution and its edge cases).

    Parameters
    ----------
    cout : array-like
        Concentration of the extracted water, constant per bin (n values).
        NaN entries are treated as not exceeding ``level``.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n values).
    level : float, optional
        Detection level (same units as ``cout``). Default 0.0.

    Returns
    -------
    Timestamp
        First-arrival time; ``pd.NaT`` if ``cout`` never exceeds ``level``.

    See Also
    --------
    time_to_threshold : Same crossing estimate for a regulatory threshold.
    time_to_peak : Time at which the maximum concentration occurs.
    """
    return _crossing_time(cout=cout, tedges=tedges, level=level)


def time_to_threshold(*, cout: npt.ArrayLike, tedges: pd.DatetimeIndex, threshold: float) -> pd.Timestamp | NaTType:
    """First time the concentration rises strictly above a threshold.

    The crossing time is linearly interpolated between the centers of the bin that
    first exceeds ``threshold`` and the preceding bin (see module docstring for the
    bin-center attribution and its edge cases).

    Parameters
    ----------
    cout : array-like
        Concentration of the extracted water, constant per bin (n values).
        NaN entries are treated as not exceeding ``threshold``.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n values).
    threshold : float
        Threshold concentration (same units as ``cout``), e.g. a regulatory limit.

    Returns
    -------
    Timestamp
        Threshold-crossing time; ``pd.NaT`` if ``cout`` never exceeds ``threshold``.

    See Also
    --------
    first_arrival : Same crossing estimate for a small detection level.
    peak_concentration : Maximum concentration of the series.
    """
    return _crossing_time(cout=cout, tedges=tedges, level=threshold)


def peak_concentration(*, cout: npt.ArrayLike) -> float:
    """Maximum concentration of the series, ignoring NaN.

    Parameters
    ----------
    cout : array-like
        Concentration of the extracted water, constant per bin (n values).

    Returns
    -------
    float
        Maximum of ``cout``.

    See Also
    --------
    time_to_peak : Time at which the maximum occurs.
    """
    return float(np.nanmax(np.asarray(cout, dtype=float)))


def time_to_peak(*, cout: npt.ArrayLike, tedges: pd.DatetimeIndex) -> pd.Timestamp:
    """Time at which the maximum concentration occurs, ignoring NaN.

    The peak is attributed to the center of the bin holding the first occurrence
    of the maximum, consistent with ``cout`` being a flow-weighted bin average.

    Parameters
    ----------
    cout : array-like
        Concentration of the extracted water, constant per bin (n values).
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n values).

    Returns
    -------
    Timestamp
        Bin-center timestamp of the maximum of ``cout``.

    See Also
    --------
    peak_concentration : Maximum concentration of the series.
    first_arrival : Earliest time the concentration exceeds a detection level.
    """
    cout = np.asarray(cout, dtype=float)
    _validate_tedges_parity(tedges, cout, tedges_name="tedges", values_name="cout")
    i = int(np.nanargmax(cout))
    return tedges[i] + (tedges[i + 1] - tedges[i]) / 2
