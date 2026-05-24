r"""
Rainfall contribution to the temperature of extracted water.

This module provides one public function:

- :func:`rainfall_to_extracted_temperature` — temperature of extracted water
  as an energy-conserving mix of advectively transported aquifer heat and the
  heat carried by rain falling onto the infiltration footprint.

Model
-----
The extracted-water temperature is the energy-weighted (volume-weighted) mean
of two contributions per output bin:

.. math::

    T_\\text{ext} = \\frac{V_\\text{rain}\\,T_\\text{rain} + V_\\text{ext}\\,T_\\text{adv}}
                         {V_\\text{rain} + V_\\text{ext}}

where

- ``T_adv`` is the temperature of the advectively transported aquifer water,
  computed by treating heat as a retarded tracer through
  :func:`gwtransport.advection.gamma_infiltration_to_extraction` (default
  ``retardation_factor = 2.0`` for heat);
- ``T_rain`` is the temperature of the rain water;
- ``V_ext`` is the extracted water volume in the output bin (flow integrated
  over the bin);
- ``V_rain`` is the rain volume reaching the infiltration footprint over the
  output bin, ``rainfall · infiltration_area · Δt``.

Energy-conserving derivation and the rho*c cancellation
-------------------------------------------------------
Conservation of thermal energy for two streams that mix into one extracted
stream is, writing :math:`\rho c` for the volumetric heat capacity
(J m\ :sup:`-3` K\ :sup:`-1`) and taking a common reference temperature,

.. math::

    \rho c_\text{rain}\,V_\text{rain}\,T_\text{rain}
      + \rho c_\text{water}\,V_\text{ext}\,T_\text{adv}
    = \rho c_\text{mix}\,(V_\text{rain} + V_\text{ext})\,T_\text{ext}.

Both contributions are liquid water, so :math:`\rho c_\text{rain} =
\rho c_\text{water} = \rho c_\text{mix}` and the factor cancels exactly,
leaving the volume-weighted mean above. The volumetric heat capacity is
therefore **not** a parameter of this v1 function. Allowing a distinct
rain heat capacity (:math:`\rho c_\text{rain} \ne \rho c_\text{water}`,
e.g. for a different solute load) is a trivial future extension: reinstate
the three :math:`\rho c` factors and they no longer cancel.

The issue's original "energy per unit volume converted to Kelvin" formulation
is dimensionally inconsistent (J m\ :sup:`-3` is not a temperature); the
energy-weighted mean above is the dimensionally correct statement of the same
intent.

v1 mixing assumption
--------------------
Rain is blended with the advected aquifer water **at the extraction point**.
This is a lumped simplification: in reality rain infiltrates at the surface,
advects through the column, and exchanges heat with the matrix along the way.
The lumped model is appropriate when the rain-contributed volume is a small
perturbation on the extracted volume, or when only the bulk energy balance
(not the in-column thermal history) is of interest. See
:ref:`assumption-well-mixed-extraction`.

Forward-only
------------
Only the forward (infiltration-to-extraction) direction is provided. The
reverse problem is non-unique: a single extracted temperature constrains one
equation but there are two temperature unknowns (the infiltration temperature
feeding ``T_adv`` and the rain temperature ``T_rain``), so it cannot be
inverted without additional assumptions.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport._time import tedges_to_days
from gwtransport._validation import (
    _validate_no_nan,
    _validate_non_negative_array,
    _validate_positive_scalar,
    _validate_tedges_parity,
)
from gwtransport.advection import gamma_infiltration_to_extraction
from gwtransport.utils import cumulative_flow_volume, linear_interpolate


def _validate_rainfall_inputs(
    *,
    cin: npt.NDArray[np.floating],
    t_rain: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    rainfall: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    infiltration_area: float,
) -> None:
    """Validate inputs for :func:`rainfall_to_extracted_temperature`.

    Composes the shared validation atoms from :mod:`gwtransport._validation`:
    bin-edge parity for ``cin``/``flow``/``rainfall`` against ``tedges`` and for
    ``t_rain`` against ``cout_tedges``, non-negative ``flow`` and ``rainfall``,
    a strictly positive ``infiltration_area``, and NaN-free inputs.

    Parameters
    ----------
    cin : ndarray
        Infiltration water temperature per ``tedges`` bin (K).
    t_rain : ndarray
        Rain water temperature per ``cout_tedges`` bin (K).
    flow : ndarray
        Flow rate per ``tedges`` bin (m³/day).
    rainfall : ndarray
        Rainfall rate per ``tedges`` bin (m/day).
    tedges : DatetimeIndex
        Time bin edges for ``cin``, ``flow``, and ``rainfall`` (n+1 edges).
    cout_tedges : DatetimeIndex
        Output time bin edges for ``t_rain`` and the result (m+1 edges).
    infiltration_area : float
        Plan-view infiltration footprint (m²).

    Raises
    ------
    ValueError
        If any bin-edge parity fails, ``flow`` or ``rainfall`` is negative,
        ``infiltration_area`` is not strictly positive, or any input is NaN.
    """
    _validate_tedges_parity(tedges, cin, tedges_name="tedges", values_name="cin")
    _validate_tedges_parity(tedges, flow, tedges_name="tedges", values_name="flow")
    _validate_tedges_parity(tedges, rainfall, tedges_name="tedges", values_name="rainfall")
    _validate_tedges_parity(cout_tedges, t_rain, tedges_name="cout_tedges", values_name="t_rain")
    _validate_non_negative_array(flow, name="flow", message="flow must be non-negative (negative flow not supported)")
    _validate_non_negative_array(rainfall, name="rainfall", message="rainfall must be non-negative")
    _validate_positive_scalar(infiltration_area, name="infiltration_area")
    _validate_no_nan(cin, name="cin")
    _validate_no_nan(t_rain, name="t_rain")
    _validate_no_nan(flow, name="flow")
    _validate_no_nan(rainfall, name="rainfall")


def rainfall_to_extracted_temperature(
    *,
    cin: npt.ArrayLike,
    t_rain: npt.ArrayLike,
    flow: npt.ArrayLike,
    rainfall: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    infiltration_area: float,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    retardation_factor: float = 2.0,
) -> npt.NDArray[np.floating]:
    r"""
    Temperature of extracted water as an energy-conserving rain/aquifer mix.

    The extracted-water temperature is the volume-weighted (energy-conserving)
    mean of two contributions:

    .. math::

        T_\text{ext} = \frac{V_\text{rain}\,T_\text{rain}
                              + V_\text{ext}\,T_\text{adv}}
                             {V_\text{rain} + V_\text{ext}}.

    ``T_adv`` is the advectively transported aquifer-water temperature, obtained
    by treating heat as a retarded tracer through
    :func:`gwtransport.advection.gamma_infiltration_to_extraction` with the
    infiltration temperature ``cin`` and a gamma-distributed aquifer pore volume.
    ``V_ext`` is the extracted volume per output bin (``flow`` integrated over
    each ``cout_tedges`` interval); ``V_rain = rainfall · infiltration_area · Δt``
    is the rain volume reaching the footprint over the same bin. The shared
    volumetric heat capacity of liquid water cancels (see module docstring), so
    it is not a parameter here.

    Provide either (``mean``, ``std``) or (``alpha``, ``beta``) for the gamma
    aquifer pore volume distribution; ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cin : array-like
        Temperature of the infiltrating water (K). Constant over each interval
        ``[tedges[i], tedges[i+1])``. Length ``len(tedges) - 1``.
    t_rain : array-like
        Temperature of the rain water (K). Constant over each output interval
        ``[cout_tedges[i], cout_tedges[i+1])``. Length ``len(cout_tedges) - 1``.
    flow : array-like
        Flow rate of water in the aquifer (m³/day). Constant over each interval
        ``[tedges[i], tedges[i+1])``. Length ``len(tedges) - 1``.
    rainfall : array-like
        Rainfall rate (m/day). Constant over each interval
        ``[tedges[i], tedges[i+1])``. Length ``len(tedges) - 1``. Non-negative.
    tedges : pandas.DatetimeIndex
        Time bin edges for ``cin``, ``flow``, and ``rainfall``. Length one more
        than those arrays.
    cout_tedges : pandas.DatetimeIndex
        Output time bin edges for ``t_rain`` and the returned temperature.
        Length one more than the desired output.
    infiltration_area : float
        Plan-view area of the infiltration footprint (m²) over which rain is
        collected. This is a surface (map-view) area, **not** a pore-geometry
        parameter; it is supplied directly and never derived from porosity or
        aquifer thickness. Must be strictly positive.
    mean : float, optional
        Mean of the gamma aquifer pore volume distribution. Must be strictly
        greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma aquifer pore volume distribution
        (invariant under the ``loc`` shift).
    loc : float, optional
        Location (minimum pore volume) of the gamma distribution. Must satisfy
        ``0 <= loc < mean``. Default is ``0.0``.
    alpha : float, optional
        Shape parameter of the gamma distribution (must be > 0).
    beta : float, optional
        Scale parameter of the gamma distribution (must be > 0).
    n_bins : int, optional
        Number of bins used to discretize the gamma distribution. Default 100.
    retardation_factor : float, optional
        Retardation factor of heat in the aquifer. Default 2.0 (typical for
        heat transport in saturated porous media).

    Returns
    -------
    numpy.ndarray
        Flow-weighted extracted-water temperature (K), aligned to
        ``cout_tedges``. Length ``len(cout_tedges) - 1``.

        Output bins where ``gamma_infiltration_to_extraction`` returns NaN
        (insufficient flow history; see its ``spinup`` policy) propagate NaN.
        Output bins where both contributions vanish (``V_rain == 0`` and
        ``V_ext == 0``) are NaN (0/0, undefined). In the pure limits the result
        is exact: ``V_rain == 0`` returns ``T_adv`` and ``V_ext == 0`` returns
        ``t_rain``, each bit-for-bit.

    Raises
    ------
    ValueError
        If any bin-edge parity fails, ``flow`` or ``rainfall`` is negative,
        ``infiltration_area`` is not strictly positive, any input is NaN, or
        the gamma parameterization is under- or over-specified (raised by
        :func:`gwtransport.gamma.bins`).

    See Also
    --------
    gwtransport.advection.gamma_infiltration_to_extraction : Advective heat transport (``T_adv``).
    gwtransport.residence_time.residence_time : Residence times underlying the advective transport.
    :ref:`concept-residence-time` : Time spent in the aquifer.
    :ref:`concept-retardation-factor` : Retardation of heat relative to water.
    :ref:`assumption-thermal-retardation` : Heat-as-retarded-tracer assumption.
    :ref:`assumption-well-mixed-extraction` : Lumped extraction-point mixing.

    Notes
    -----
    The mixing is energy-conserving: with a single liquid-water volumetric heat
    capacity, the energy balance
    ``rho*c (V_rain T_rain + V_ext T_adv) = rho*c (V_rain + V_ext) T_ext`` reduces
    to the volume-weighted mean above (the ``rho*c`` factor cancels). Blending rain at the
    extraction point is a lumped v1 assumption; rain that physically advects
    through the column is not resolved. Only the forward direction is provided
    because the reverse is non-unique (two temperature unknowns, one equation).

    ``V_ext`` is computed from the cumulative flow volume on ``tedges`` evaluated
    at the ``cout_tedges`` edges, so output bins need not align with the input
    grid. ``V_rain`` uses the same ``cout_tedges`` widths after averaging the
    rain volume rate onto the output grid.

    Examples
    --------
    Equal rain and extraction volumes give the arithmetic mean of the two
    temperatures:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.rainfall import rainfall_to_extracted_temperature
    >>>
    >>> dates = pd.date_range(start="2020-01-01", end="2020-03-31", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>> n = len(dates)
    >>> cin = np.full(n, 290.0)  # infiltration temperature (K)
    >>> flow = np.full(n, 100.0)  # m3/day
    >>> rainfall = np.full(n, 0.0)  # no rain -> result equals T_adv
    >>> t_rain = np.full(n, 280.0)  # rain temperature (K)
    >>> t_ext = rainfall_to_extracted_temperature(
    ...     cin=cin,
    ...     t_rain=t_rain,
    ...     flow=flow,
    ...     rainfall=rainfall,
    ...     tedges=tedges,
    ...     cout_tedges=tedges,
    ...     infiltration_area=1000.0,
    ...     mean=500.0,
    ...     std=200.0,
    ...     n_bins=20,
    ... )
    >>> t_ext.shape
    (91,)
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    cin = np.asarray(cin, dtype=float)
    t_rain = np.asarray(t_rain, dtype=float)
    flow = np.asarray(flow, dtype=float)
    rainfall = np.asarray(rainfall, dtype=float)

    _validate_rainfall_inputs(
        cin=cin,
        t_rain=t_rain,
        flow=flow,
        rainfall=rainfall,
        tedges=tedges,
        cout_tedges=cout_tedges,
        infiltration_area=infiltration_area,
    )

    # Advective aquifer-heat temperature, already aligned to cout_tedges.
    t_adv = gamma_infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        mean=mean,
        std=std,
        loc=loc,
        alpha=alpha,
        beta=beta,
        n_bins=n_bins,
        retardation_factor=retardation_factor,
    )

    # Extracted volume per output bin: integrate flow (on tedges) over each
    # cout_tedges interval via the cumulative-flow-volume machinery. Using a
    # shared reference (tedges[0]) keeps the two edge arrays on one origin so
    # cout_tedges need not coincide with tedges.
    tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    flow_cum = cumulative_flow_volume(flow, np.diff(tedges_days))
    flow_cum_at_cout = linear_interpolate(x_ref=tedges_days, y_ref=flow_cum, x_query=cout_tedges_days)
    v_ext = np.diff(flow_cum_at_cout)

    # Rain volume per output bin: same cumulative-then-difference treatment so
    # the rain rate (on tedges) is integrated over each cout_tedges interval.
    rain_volume_rate = rainfall * infiltration_area  # m3/day on tedges bins
    rain_cum = cumulative_flow_volume(rain_volume_rate, np.diff(tedges_days))
    rain_cum_at_cout = linear_interpolate(x_ref=tedges_days, y_ref=rain_cum, x_query=cout_tedges_days)
    v_rain = np.diff(rain_cum_at_cout)

    # Energy-conserving (volume-weighted) mix. Explicit zero-volume branches make
    # the pure limits exact: V_rain == 0 -> T_adv, V_ext == 0 -> t_rain; both
    # zero -> NaN (0/0, undefined).
    total_volume = v_rain + v_ext
    with np.errstate(invalid="ignore", divide="ignore"):
        mixed = (v_rain * t_rain + v_ext * t_adv) / total_volume
    mixed = np.where(v_rain == 0.0, t_adv, mixed)
    mixed = np.where(v_ext == 0.0, t_rain, mixed)
    return np.where(total_volume == 0.0, np.nan, mixed)
