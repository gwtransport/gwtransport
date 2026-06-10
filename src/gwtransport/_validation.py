"""
Composable input-validation atoms for transport-module entry points.

The public ``infiltration_to_extraction`` / ``extraction_to_infiltration`` functions in
:mod:`gwtransport.advection`, :mod:`gwtransport.diffusion`, and
:mod:`gwtransport.deposition` share a small set of input-validation invariants
(bin-edge parity, NaN-free arrays, non-negative flow, positive physical
parameters). The atoms here factor those invariants once so that each module
exposes a single ``_validate_<module>_inputs`` wrapper composing the atoms with
module-specific error-message wording and ordering.

Each atom is keyword-only (except for the array under test) and raises
``ValueError`` with a default message; the optional ``message`` keyword lets the
module wrapper preserve the historical wording verbatim so that downstream
``pytest.raises(..., match=...)`` tests keep passing without modification.

This module has no public API; importers are the transport modules themselves
plus ``tests/src/test_validation.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd  # noqa: TC002  -- pandas is a hard runtime dependency; import unconditionally


def _validate_tedges_parity(
    tedges: pd.DatetimeIndex,
    values: npt.ArrayLike,
    *,
    tedges_name: str,
    values_name: str,
) -> None:
    """Validate bin-edge parity: ``len(tedges) == len(values) + 1``.

    Parameters
    ----------
    tedges : DatetimeIndex
        Bin edges (length ``n + 1``).
    values : array-like
        Bin-constant values (length ``n``).
    tedges_name, values_name : str
        Names used in the error message, e.g. ``"tedges"`` and ``"cin"``.

    Raises
    ------
    ValueError
        If ``len(tedges) != len(values) + 1``.
    """
    n_values = np.asarray(values).shape[0]
    if len(tedges) != n_values + 1:
        msg = f"{tedges_name} must have one more element than {values_name}"
        raise ValueError(msg)


def _validate_no_nan(
    arr: npt.ArrayLike,
    *,
    name: str,
    message: str | None = None,
) -> None:
    """Validate that ``arr`` contains no NaN values.

    Parameters
    ----------
    arr : array-like
        Array to check.
    name : str
        Variable name used in the default error message.
    message : str, optional
        Override the default ``"{name} contains NaN values, which are not allowed"``
        wording. Used by module wrappers that need to preserve historical strings
        pinned by ``pytest.raises(..., match=...)`` tests.

    Raises
    ------
    ValueError
        If any element of ``arr`` is NaN.
    """
    if np.any(np.isnan(np.asarray(arr))):
        msg = message if message is not None else f"{name} contains NaN values, which are not allowed"
        raise ValueError(msg)


def _validate_non_negative_array(
    arr: npt.ArrayLike,
    *,
    name: str,
    message: str | None = None,
) -> None:
    """Validate that ``arr`` has no strictly negative elements.

    Zeros are allowed. The companion ``_validate_positive_array`` rejects zeros too.

    Parameters
    ----------
    arr : array-like
        Array to check (any shape).
    name : str
        Variable name used in the default error message.
    message : str, optional
        Override the default ``"{name} must be non-negative"`` wording.

    Raises
    ------
    ValueError
        If any element of ``arr`` is strictly negative.
    """
    if np.any(np.asarray(arr) < 0):
        msg = message if message is not None else f"{name} must be non-negative"
        raise ValueError(msg)


def _validate_positive_array(
    arr: npt.ArrayLike,
    *,
    name: str,
    message: str | None = None,
) -> None:
    """Validate that every element of ``arr`` is strictly positive (``> 0``).

    Parameters
    ----------
    arr : array-like
        Array to check.
    name : str
        Variable name used in the default error message.
    message : str, optional
        Override the default ``"{name} must be positive"`` wording.

    Raises
    ------
    ValueError
        If any element of ``arr`` is ``<= 0``.
    """
    if np.any(np.asarray(arr) <= 0):
        msg = message if message is not None else f"{name} must be positive"
        raise ValueError(msg)


def _validate_positive_scalar(
    value: float,
    *,
    name: str,
    message: str | None = None,
) -> None:
    """Validate that ``value`` is strictly positive (``> 0``).

    Parameters
    ----------
    value : float
        Scalar to check.
    name : str
        Variable name used in the default error message.
    message : str, optional
        Override the default ``"{name} must be positive, got {value}"`` wording.

    Raises
    ------
    ValueError
        If ``value <= 0``.
    """
    if value <= 0:
        msg = message if message is not None else f"{name} must be positive, got {value}"
        raise ValueError(msg)


def _validate_scalar_or_matching_length(
    arr: npt.ArrayLike,
    *,
    name: str,
    expected_len: int,
    ref_name: str,
) -> None:
    """Validate length: ``len(arr) == expected_len``.

    Intended for inputs that the caller accepts as either a scalar or an
    array matching some reference length; the caller is expected to have
    broadcast size-1 arrays to ``expected_len`` *before* calling this atom.
    The error message refers to the user-facing scalar-or-matching contract.

    Parameters
    ----------
    arr : array-like
        Array to check (already broadcast from scalar form by the caller).
    name : str
        Variable name used in the error message.
    expected_len : int
        Required length.
    ref_name : str
        Name of the reference array whose length is the contract.

    Raises
    ------
    ValueError
        If ``len(arr) != expected_len``.
    """
    if np.asarray(arr).shape[0] != expected_len:
        msg = f"{name} must be a scalar or have same length as {ref_name}"
        raise ValueError(msg)
