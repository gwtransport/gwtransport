"""
Time-axis conversion atoms for transport-module entry points.

The transport modules in :mod:`gwtransport.advection`,
:mod:`gwtransport.diffusion`, :mod:`gwtransport.diffusion_fast`,
:mod:`gwtransport.deposition`, and :mod:`gwtransport.residence_time` repeatedly
convert a :class:`pandas.DatetimeIndex` of bin edges into a float64 array of
days relative to a reference timestamp. The two helpers here factor that
idiom once so the conversion (and its object-dtype-on-old-pandas contract)
lives in a single place.

Both helpers return ``float64`` arrays. ``tedges_to_days`` measures each edge
relative to ``ref`` (defaulting to the first edge); ``dt_to_days`` returns the
successive bin widths in days. The ``ref`` keyword is load-bearing: cross-array
conversions (e.g. output edges measured against the input-flow reference) must
share a common origin.

This module has no public API; importers are the transport modules themselves
plus ``tests/src/test_utils.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd


def tedges_to_days(tedges: pd.DatetimeIndex, *, ref: pd.Timestamp | None = None) -> npt.NDArray[np.floating]:
    """Convert time-bin edges to days relative to a reference timestamp.

    Parameters
    ----------
    tedges : DatetimeIndex
        Time-bin edges to convert.
    ref : Timestamp or None, optional
        Reference timestamp mapped to day zero. Defaults to ``tedges[0]`` when
        ``None``. Pass a shared reference when converting a second edge array
        that must align to the same origin.

    Returns
    -------
    ndarray
        Float64 array of days since ``ref``, one value per edge.
    """
    origin = tedges[0] if ref is None else ref
    return ((tedges - origin) / pd.Timedelta(days=1)).to_numpy(dtype=float)


def dt_to_days(t: pd.DatetimeIndex) -> npt.NDArray[np.floating]:
    """Convert successive time-bin widths to days.

    Parameters
    ----------
    t : DatetimeIndex
        Time-bin edges (n+1 edges for n bins).

    Returns
    -------
    ndarray
        Float64 array of bin widths in days (length ``len(t) - 1``).
    """
    return (np.diff(t) / pd.Timedelta(days=1)).astype(float)
