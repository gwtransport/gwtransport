"""Tests for the input-validation atoms in :mod:`gwtransport._validation`.

One pair (silent-on-good + parametrized-bad) per atom; the ``message=`` override
path is covered for each atom that exposes it so that a future refactor cannot
drop the override without the test catching it.
"""

import numpy as np
import pandas as pd
import pytest

from gwtransport._validation import (
    _validate_no_nan,
    _validate_non_negative_array,
    _validate_positive_array,
    _validate_positive_scalar,
    _validate_retardation_factor,
    _validate_scalar_or_matching_length,
    _validate_tedges_parity,
)

# ---------------------------------------------------------------------------
# _validate_tedges_parity
# ---------------------------------------------------------------------------


def test_tedges_parity_silent_on_good_input():
    tedges = pd.date_range("2020-01-01", periods=6, freq="D")
    values = np.zeros(5)
    _validate_tedges_parity(tedges, values, tedges_name="tedges", values_name="cin")


@pytest.mark.parametrize(
    ("n_tedges", "n_values", "tedges_name", "values_name"),
    [
        (6, 4, "tedges", "cin"),  # tedges too long
        (5, 5, "tedges", "flow"),  # tedges off-by-one
        (4, 5, "cout_tedges", "cout"),  # tedges too short
    ],
)
def test_tedges_parity_rejects_mismatch(n_tedges, n_values, tedges_name, values_name):
    tedges = pd.date_range("2020-01-01", periods=n_tedges, freq="D")
    values = np.zeros(n_values)
    msg = f"{tedges_name} must have one more element than {values_name}"
    with pytest.raises(ValueError, match=msg):
        _validate_tedges_parity(tedges, values, tedges_name=tedges_name, values_name=values_name)


# ---------------------------------------------------------------------------
# _validate_no_nan
# ---------------------------------------------------------------------------


def test_no_nan_silent_on_good_input():
    _validate_no_nan(np.array([0.0, 1.0, 2.0]), name="cin")


def test_no_nan_rejects_default_message():
    with pytest.raises(ValueError, match="cin contains NaN values, which are not allowed"):
        _validate_no_nan(np.array([0.0, np.nan, 2.0]), name="cin")


def test_no_nan_rejects_override_message():
    with pytest.raises(ValueError, match="flow array cannot contain NaN values"):
        _validate_no_nan(np.array([0.0, np.nan, 2.0]), name="flow", message="flow array cannot contain NaN values")


# ---------------------------------------------------------------------------
# _validate_non_negative_array
# ---------------------------------------------------------------------------


def test_non_negative_array_silent_on_good_input():
    _validate_non_negative_array(np.array([0.0, 1.0, 2.0]), name="flow")


def test_non_negative_array_silent_on_all_zeros():
    """Zero is non-negative; the atom must NOT reject it."""
    _validate_non_negative_array(np.zeros(5), name="flow")


def test_non_negative_array_rejects_default_message():
    with pytest.raises(ValueError, match="flow must be non-negative"):
        _validate_non_negative_array(np.array([1.0, -0.5, 2.0]), name="flow")


def test_non_negative_array_rejects_override_message():
    with pytest.raises(ValueError, match=r"flow must be non-negative \(negative flow not supported\)"):
        _validate_non_negative_array(
            np.array([-1.0]),
            name="flow",
            message="flow must be non-negative (negative flow not supported)",
        )


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_non_negative_array_rejects_non_finite(bad):
    """NaN and +inf pass every ``< 0`` comparison; the atom must still reject them."""
    with pytest.raises(ValueError, match="molecular_diffusivity must be non-negative"):
        _validate_non_negative_array(np.array([0.0, bad]), name="molecular_diffusivity")


# ---------------------------------------------------------------------------
# _validate_positive_array
# ---------------------------------------------------------------------------


def test_positive_array_silent_on_good_input():
    _validate_positive_array(np.array([1.0, 2.0]), name="aquifer_pore_volumes")


def test_positive_array_rejects_zero():
    """Zero is NOT positive; ``_validate_positive_array`` must reject it."""
    with pytest.raises(ValueError, match="aquifer_pore_volumes must be positive"):
        _validate_positive_array(np.array([1.0, 0.0]), name="aquifer_pore_volumes")


def test_positive_array_rejects_negative():
    with pytest.raises(ValueError, match="streamline_length must be positive"):
        _validate_positive_array(np.array([1.0, -0.5]), name="streamline_length")


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_positive_array_rejects_non_finite(bad):
    """NaN and +inf pass every ``<= 0`` comparison; the atom must still reject them."""
    with pytest.raises(ValueError, match="aquifer_pore_volumes must be positive"):
        _validate_positive_array(np.array([1.0, bad]), name="aquifer_pore_volumes")


# ---------------------------------------------------------------------------
# _validate_positive_scalar
# ---------------------------------------------------------------------------


def test_positive_scalar_silent_on_good_input():
    _validate_positive_scalar(5.0, name="thickness")


def test_positive_scalar_rejects_default_message():
    with pytest.raises(ValueError, match=r"thickness must be positive, got 0\.0"):
        _validate_positive_scalar(0.0, name="thickness")


def test_positive_scalar_rejects_override_message():
    """Module wrappers preserve historical capitalized phrasings via ``message=``."""
    with pytest.raises(ValueError, match=r"Thickness must be positive, got -1\.0"):
        _validate_positive_scalar(-1.0, name="thickness", message="Thickness must be positive, got -1.0")


@pytest.mark.parametrize("bad", [np.inf, np.nan])
def test_positive_scalar_rejects_non_finite(bad):
    """NaN and +inf are neither ``<= 0``; the atom must still reject them."""
    with pytest.raises(ValueError, match="thickness must be positive"):
        _validate_positive_scalar(bad, name="thickness")


# ---------------------------------------------------------------------------
# _validate_scalar_or_matching_length
# ---------------------------------------------------------------------------


def test_scalar_or_matching_length_silent_when_matching():
    arr = np.array([1.0, 2.0, 3.0])
    _validate_scalar_or_matching_length(
        arr, name="molecular_diffusivity", expected_len=3, ref_name="aquifer_pore_volumes"
    )


def test_scalar_or_matching_length_rejects_wrong_length():
    arr = np.array([1.0, 2.0])
    with pytest.raises(
        ValueError,
        match="molecular_diffusivity must be a scalar or have same length as aquifer_pore_volumes",
    ):
        _validate_scalar_or_matching_length(
            arr, name="molecular_diffusivity", expected_len=3, ref_name="aquifer_pore_volumes"
        )


# ---------------------------------------------------------------------------
# _validate_retardation_factor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("good", [1.0, 1.5, 100.0, np.inf])
def test_retardation_factor_silent_on_ge_one(good):
    """Any value ``>= 1`` (including +inf) is physical and must pass silently."""
    _validate_retardation_factor(good)


@pytest.mark.parametrize("bad", [0.0, 0.5, 0.999, np.nan])
def test_retardation_factor_rejects_below_one_and_nan(bad):
    """Values ``< 1`` and NaN both fail ``>= 1``; the atom must reject them."""
    with pytest.raises(ValueError, match=r"retardation_factor must be >= 1\.0"):
        _validate_retardation_factor(bad)
