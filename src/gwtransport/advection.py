"""
Advective Transport Modeling Along Aquifer Pore Volumes.

Water infiltrates and is transported in parallel along multiple aquifer pore volumes to
extraction. For each aquifer pore volume, transport is 1D advection with linear or non-linear
sorption; there is no microdispersion or molecular diffusion, while the spread across aquifer
pore volumes provides macrodispersion. Forward and backward modeling are supported. No assumption
is made about whether the flow is radial or orthogonal.

Available functions:

- :func:`infiltration_to_extraction` - Arbitrary pore volume distribution, flow-weighted averaging.
  Supports explicit distribution of aquifer pore volumes with flow-weighted averaging.
  Flexible output time resolution via cout_tedges. Use case: Known pore volume distribution
  from streamline analysis.

- :func:`gamma_infiltration_to_extraction` - Gamma-distributed pore volumes, flow-weighted averaging.
  Models aquifer heterogeneity with 2-parameter gamma distribution. Parameterizable via
  (alpha, beta) or (mean, std). Discretizes gamma distribution into equal-probability bins.
  Use case: Heterogeneous aquifer with calibrated gamma parameters.

- :func:`extraction_to_infiltration` - Arbitrary pore volume distribution, deconvolution.
  Inverts forward transport for arbitrary pore volume distributions. Symmetric inverse of
  infiltration_to_extraction. Flow-weighted averaging in reverse direction. Use case:
  Estimating infiltration history from extraction data.

- :func:`gamma_extraction_to_infiltration` - Gamma-distributed pore volumes, deconvolution.
  Inverts forward transport for gamma-distributed pore volumes. Symmetric inverse of
  gamma_infiltration_to_extraction. Use case: Calibrating infiltration conditions from
  extraction measurements.

- :func:`infiltration_to_extraction_nonlinear_sorption` - Exact front tracking with nonlinear sorption.
  Event-driven algorithm that solves 1D advective transport with Freundlich or Langmuir isotherm
  using analytical integration of shock and rarefaction waves. Machine-precision physics (no
  numerical dispersion). Returns bin-averaged concentrations together with the full piecewise
  analytical structure (events, segments, wave list) for downstream analysis. Use case: Sharp
  concentration fronts with exact mass balance required, across a distribution of aquifer
  pore volumes (macrodispersion). Forward modeling only; nonlinear sorption has no inverse.

Note on dispersion: The spreading from the pore volume distribution (APVD) represents
macrodispersion—aquifer-scale velocity heterogeneity that depends on both aquifer
properties and hydrological boundary conditions. To add microdispersion and molecular
diffusion separately (when APVD comes from streamline analysis), use :mod:`gwtransport.diffusion`.
See :ref:`concept-dispersion-scales` for details.

Note on cross-compound calibration: When APVD is calibrated from measurements of one
compound (e.g., temperature with D_m ~ 0.1 m²/day) and used to predict another (e.g., a
solute with D_m ~ 1e-4 m²/day), the molecular diffusion contribution is baked into the
calibrated std. The cleanest fix is to calibrate with :mod:`gwtransport.diffusion_fast`
instead, which keeps the three contributions separate.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from collections import Counter

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport import gamma
from gwtransport._time import tedges_to_days
from gwtransport._validation import (
    _validate_no_nan,
    _validate_non_negative_array,
    _validate_retardation_factor,
    _validate_tedges_parity,
)
from gwtransport.advection_utils import (
    _infiltration_to_extraction_weights,
    _resolve_spinup_inputs,
    _resolve_spinup_mask,
)
from gwtransport.fronttracking.math import (
    EPSILON_FREUNDLICH_N,
    ConstantRetardation,
    FreundlichSorption,
    LangmuirSorption,
    SorptionModel,
)
from gwtransport.fronttracking.output import compute_bin_averaged_concentration_exact
from gwtransport.fronttracking.solver import FrontTracker, find_unresolved_interaction
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave
from gwtransport.utils import solve_inverse_transport_banded


def _validate_advection_inputs(
    *,
    tedges: pd.DatetimeIndex,
    flow: np.ndarray,
    retardation_factor: float,
    aquifer_pore_volumes: npt.ArrayLike | None = None,
    cin_values: np.ndarray | None = None,
    cout_values: np.ndarray | None = None,
    cout_tedges: pd.DatetimeIndex | None = None,
) -> None:
    """Validate inputs common to advection forward / reverse entry points.

    Path selection via mutually-exclusive kwargs:

    - ``cin_values`` provided => forward path. ``tedges`` parities cin and flow.
    - ``cout_values`` + ``cout_tedges`` provided => reverse path. ``tedges`` parities
      flow; ``cout_tedges`` parities cout.

    All shared checks fire on both paths. ``flow >= 0`` is enforced in both
    directions (the previous reverse prologue omitted this; see issue #187), and
    ``aquifer_pore_volumes`` (when passed) must be finite and strictly positive --
    a negative or zero volume would source a cout bin from future infiltration.
    Every other error-message string is preserved verbatim from the prior
    duplicated prologue so that ``match=`` regex tests do not break.

    Raises
    ------
    ValueError
        If any check fails. The message identifies which invariant was violated.
    """
    if cin_values is not None:
        _validate_tedges_parity(tedges, cin_values, tedges_name="tedges", values_name="cin")
        _validate_tedges_parity(tedges, flow, tedges_name="tedges", values_name="flow")
        _validate_no_nan(cin_values, name="cin")
    elif cout_values is not None and cout_tedges is not None:
        _validate_tedges_parity(tedges, flow, tedges_name="tedges", values_name="flow")
        _validate_tedges_parity(cout_tedges, cout_values, tedges_name="cout_tedges", values_name="cout")
        _validate_no_nan(cout_values, name="cout")
    else:
        msg = "must provide cin_values (forward) or both cout_values and cout_tedges (reverse)"
        raise ValueError(msg)
    # The docstrings promise a ValueError for non-monotonic infiltration time edges, which would
    # otherwise silently corrupt the cumulative-volume mapping.
    if np.any(np.diff(tedges.asi8) <= 0):
        msg = "tedges must be strictly increasing"
        raise ValueError(msg)
    _validate_no_nan(flow, name="flow")
    _validate_non_negative_array(flow, name="flow", message="flow must be non-negative (negative flow not supported)")
    _validate_retardation_factor(retardation_factor)
    if aquifer_pore_volumes is not None:
        apv = np.asarray(aquifer_pore_volumes, dtype=float)
        # A negative or zero pore volume back-projects a cout bin to *future*
        # infiltration (anti-causal); a non-finite one poisons the whole solve.
        # The nonlinear and diffusion paths already reject these.
        if np.any(~np.isfinite(apv)) or np.any(apv <= 0.0):
            msg = "aquifer_pore_volumes must be positive"
            raise ValueError(msg)


def gamma_infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    retardation_factor: float = 1.0,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute the concentration of the extracted water by shifting cin with its residence time.

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer. The aquifer pore volume is approximated by a (shifted) gamma distribution
    parameterized by either (mean, std, loc) or (alpha, beta, loc).

    This function represents infiltration to extraction modeling by flow-weighted averaging.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water or temperature of infiltrating
        water. The model assumes this value is constant over each interval
        ``[tedges[i], tedges[i+1])``.
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. The model assumes this value is
        constant over each interval ``[tedges[i], tedges[i+1])``.
    tedges : pandas.DatetimeIndex
        Time edges for both cin and flow data. Used to compute the cumulative concentration.
        Has a length of one more than `cin` and `flow`.
    cout_tedges : pandas.DatetimeIndex
        Time edges for the output data. Used to compute the cumulative concentration.
        Has a length of one more than the desired output length.
    mean : float, optional
        Mean of the gamma distribution of the aquifer pore volume. Must be strictly
        greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma distribution of the aquifer pore volume
        (invariant under the ``loc`` shift).
    loc : float, optional
        Location (minimum pore volume) of the gamma distribution. Must satisfy
        ``0 <= loc < mean``. Default is ``0.0``.
    alpha : float, optional
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0).
    beta : float, optional
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0).
    n_bins : int, optional
        Number of bins to discretize the gamma distribution. Default 100.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.
    spinup : {"constant"} | float in [0, 1] | None, optional
        Forwarded to :func:`infiltration_to_extraction`. Default
        ``"constant"`` warm-starts the system before ``tedges[0]``.

    Returns
    -------
    numpy.ndarray
        Concentration of the compound in the extracted water, or temperature. Same units as cin.

    See Also
    --------
    infiltration_to_extraction : Transport with explicit pore volume distribution
    gamma_extraction_to_infiltration : Reverse operation (deconvolution)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.residence_time.full : Compute residence times
    gwtransport.diffusion.infiltration_to_extraction : Add microdispersion and molecular diffusion
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    :ref:`assumption-gamma-distribution` : When gamma distribution is adequate

    Notes
    -----
    The APVD is only time-invariant under the steady-streamlines assumption
    (see :ref:`assumption-steady-streamlines`).

    The spreading from the gamma-distributed pore volumes represents macrodispersion
    (aquifer-scale heterogeneity). When ``std`` comes from calibration on measurements,
    it absorbs all mixing: macrodispersion, microdispersion, and an average molecular
    diffusion contribution. When calibrating with the diffusion module, these three
    components are taken into account separately. When ``std`` comes from streamline
    analysis, it represents macrodispersion only; microdispersion and molecular diffusion
    can be added via :mod:`gwtransport.diffusion_fast` or :mod:`gwtransport.diffusion`.

    For cross-compound prediction (calibrating on temperature and predicting a solute),
    calibrate with :mod:`gwtransport.diffusion_fast` so the three contributions are
    tracked separately rather than lumped into a single calibrated ``std``.
    See :ref:`concept-dispersion-scales` for background.

    Examples
    --------
    Basic usage with alpha and beta parameters:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import gamma_infiltration_to_extraction
    >>>
    >>> # Create input data with aligned time edges
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Create output time edges (can be different alignment)
    >>> cout_dates = pd.date_range(start="2020-01-05", end="2020-01-15", freq="D")
    >>> cout_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
    ... )
    >>>
    >>> # Input concentration and flow (same length, aligned with tedges)
    >>> cin = pd.Series(np.ones(len(dates)), index=dates)
    >>> flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # 100 m³/day
    >>>
    >>> # Run gamma_infiltration_to_extraction with alpha/beta parameters
    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     n_bins=5,
    ... )
    >>> cout.shape
    (11,)

    Using mean and std parameters instead:

    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=100.0,
    ...     std=20.0,
    ...     n_bins=5,
    ... )

    With retardation factor:

    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     retardation_factor=2.0,  # Doubles residence time
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        retardation_factor=retardation_factor,
        spinup=spinup,
    )


def gamma_extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    mean: float | None = None,
    std: float | None = None,
    loc: float = 0.0,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    retardation_factor: float = 1.0,
    regularization_strength: float = 1e-10,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute the concentration of the infiltrating water from extracted water (deconvolution).

    The compound is retarded in the aquifer with a retardation factor. The residence
    time is computed based on the flow rate of the water in the aquifer and the pore volume
    of the aquifer. The aquifer pore volume is approximated by a (shifted) gamma distribution
    parameterized by either (mean, std, loc) or (alpha, beta, loc).

    This function inverts the forward flow-weighted averaging (deconvolution).
    It is symmetric to gamma_infiltration_to_extraction.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water or temperature of extracted
        water. The model assumes this value is constant over each interval
        ``[cout_tedges[i], cout_tedges[i+1])``.
    flow : array-like
        Flow rate of water in the aquifer [m³/day]. The model assumes this value is
        constant over each interval ``[tedges[i], tedges[i+1])``.
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data.
        Has a length of one more than `flow`.
    cout_tedges : pandas.DatetimeIndex
        Time edges for the cout data.
        Has a length of one more than `cout`.
    mean : float, optional
        Mean of the gamma distribution of the aquifer pore volume. Must be strictly
        greater than ``loc``.
    std : float, optional
        Standard deviation of the gamma distribution of the aquifer pore volume
        (invariant under the ``loc`` shift).
    loc : float, optional
        Location (minimum pore volume) of the gamma distribution. Must satisfy
        ``0 <= loc < mean``. Default is ``0.0``.
    alpha : float, optional
        Shape parameter of gamma distribution of the aquifer pore volume (must be > 0).
    beta : float, optional
        Scale parameter of gamma distribution of the aquifer pore volume (must be > 0).
    n_bins : int, optional
        Number of bins to discretize the gamma distribution. Default 100.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.
    regularization_strength : float, optional
        Tikhonov regularization parameter λ. See
        :func:`extraction_to_infiltration` for details. Default is 1e-10.
    spinup : {"constant"} | float in [0, 1] | None, optional
        Forwarded to :func:`extraction_to_infiltration`. Default
        ``"constant"`` warm-starts the system before ``tedges[0]``.

    Returns
    -------
    numpy.ndarray
        Concentration of the compound in the infiltrating water, or temperature. Same units as cout.

    See Also
    --------
    extraction_to_infiltration : Deconvolution with explicit pore volume distribution
    gamma_infiltration_to_extraction : Forward operation (flow-weighted averaging)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.diffusion.extraction_to_infiltration : Deconvolution with microdispersion and molecular diffusion
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    :ref:`assumption-gamma-distribution` : When gamma distribution is adequate

    Notes
    -----
    The APVD is only time-invariant under the steady-streamlines assumption
    (see :ref:`assumption-steady-streamlines`).

    The spreading from the gamma-distributed pore volumes represents macrodispersion
    (aquifer-scale heterogeneity). When ``std`` comes from calibration on measurements,
    it absorbs all mixing: macrodispersion, microdispersion, and an average molecular
    diffusion contribution. When calibrating with the diffusion module, these three
    components are taken into account separately. When ``std`` comes from streamline
    analysis, it represents macrodispersion only; microdispersion and molecular diffusion
    can be added via :mod:`gwtransport.diffusion_fast` or :mod:`gwtransport.diffusion`.

    For cross-compound prediction (calibrating on temperature and predicting a solute),
    calibrate with :mod:`gwtransport.diffusion_fast` so the three contributions are
    tracked separately rather than lumped into a single calibrated ``std``.
    See :ref:`concept-dispersion-scales` for background.

    Examples
    --------
    Basic usage with alpha and beta parameters:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import gamma_extraction_to_infiltration
    >>>
    >>> # Create cin/flow time edges
    >>> cin_dates = pd.date_range(start="2019-12-25", end="2020-01-15", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates)
    ... )
    >>>
    >>> # Create cout time edges
    >>> cout_dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
    ... )
    >>>
    >>> # Input concentration and flow
    >>> cout = np.ones(len(cout_dates))
    >>> flow = np.ones(len(cin_dates)) * 100  # 100 m³/day
    >>>
    >>> # Run gamma_extraction_to_infiltration with alpha/beta parameters
    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     n_bins=5,
    ... )
    >>> cin.shape
    (22,)

    Using mean and std parameters instead:

    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=100.0,
    ...     std=20.0,
    ...     n_bins=5,
    ... )

    With retardation factor:

    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     alpha=10.0,
    ...     beta=10.0,
    ...     retardation_factor=2.0,  # Doubles residence time
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        retardation_factor=retardation_factor,
        regularization_strength=regularization_strength,
        spinup=spinup,
    )


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    retardation_factor: float = 1.0,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute the concentration of the extracted water using flow-weighted advection.

    This function implements an infiltration to extraction advection model where cin and flow values
    correspond to the same aligned time bins defined by tedges.

    Pure advection is volume-stationary, so the weights are built on the
    cumulative-throughflow-volume axis rather than by inverting residence times:

    1. Map the cin and cout time edges to cumulative throughflow volume.
    2. Back-project each cout bin by every retarded pore volume to its
       infiltration-time source window. The window spans one cout bin's worth of
       volume, so it overlaps only a narrow band of cin bins.
    3. Compute the flow-weighted time overlap of each window with those cin bins,
       normalize per streamtube (each row sums to 1), and average over the
       streamtubes whose source window lies fully inside the cin range.


    Parameters
    ----------
    cin : array-like
        Concentration values of infiltrating water or temperature [concentration units].
        Length must match the number of time bins defined by tedges. The model assumes
        this value is constant over each interval ``[tedges[i], tedges[i+1])``.
    flow : array-like
        Flow rate values in the aquifer [m³/day].
        Length must match cin and the number of time bins defined by tedges. The model
        assumes this value is constant over each interval ``[tedges[i], tedges[i+1])``.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cin and flow data. Has length of
        len(cin) + 1 and len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
        Can have different time alignment and resolution than tedges.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m³] representing the distribution
        of residence times in the aquifer system.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.
    spinup : {"constant"} | float in [0, 1] | None, optional
        How to treat cout bins where one or more streamtube source windows
        fall outside the cin time range. Default is ``"constant"``.

        - ``"constant"`` — warm-start: shift ``tedges[0]`` backward by
          ``retardation_factor * max(aquifer_pore_volumes) / flow[0]`` and
          treat cin and flow as constant at their first value over the
          extended window. The forward strict-validity logic then has no
          NaN cout bins from spin-up; right-edge spin-up (cout extending
          past the cin range) is unchanged.
        - ``None`` — strict mass-conservation: NaN whenever any streamtube
          has not fully broken through into the cin range, or extraction
          flow during the bin is zero. Bundle row sums to 1 across cin.
        - float in [0, 1] — fraction threshold: emit cout when at least
          ``spinup * n_pv`` streamtubes have contributed; the bundle is
          then a count-mean over the contributing subset. *Warning:* this
          conserves mass per row but NOT cin → cout mass; with a delta
          cin pulse and ``spinup=0.0`` you reproduce the issue #161
          over-attribution (Σ cout > Σ cin).

    Returns
    -------
    numpy.ndarray
        Flow-weighted concentration in the extracted water. Same units
        as cin. Length equals ``len(cout_tedges) - 1``. NaN values mark
        cout bins where the chosen ``spinup`` policy is not satisfied:
        the default ``"constant"`` leaves NaN for any cout bin extending
        past the end of the flow record (a cout edge beyond
        ``tedges[-1]``, whose back-projected source window leaves the cin
        range) and for zero-throughflow bins; ``spinup=None`` additionally
        NaNs left-edge spin-up bins; a float threshold relaxes either case
        in exchange for non-mass-conserving count-mean output.

    Raises
    ------
    ValueError
        If tedges length doesn't match cin/flow arrays plus one, or if
        infiltration time edges become non-monotonic (invalid input conditions).

    See Also
    --------
    gamma_infiltration_to_extraction : Transport with gamma-distributed pore volumes
    extraction_to_infiltration : Reverse operation (deconvolution)
    gwtransport.residence_time.full : Compute residence times from flow and pore volume
    gwtransport.residence_time.freundlich_retardation : Compute concentration-dependent retardation
    :ref:`concept-pore-volume-distribution` : Background on aquifer heterogeneity modeling
    :ref:`concept-transport-equation` : Flow-weighted averaging approach

    Examples
    --------
    Basic usage with pandas Series:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import infiltration_to_extraction
    >>>
    >>> # Create input data
    >>> dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=dates, number_of_bins=len(dates)
    ... )
    >>>
    >>> # Create output time edges (different alignment)
    >>> cout_dates = pd.date_range(start="2020-01-05", end="2020-01-15", freq="D")
    >>> cout_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
    ... )
    >>>
    >>> # Input concentration and flow
    >>> cin = pd.Series(np.ones(len(dates)), index=dates)
    >>> flow = pd.Series(np.ones(len(dates)) * 100, index=dates)  # 100 m³/day
    >>>
    >>> # Define distribution of aquifer pore volumes
    >>> aquifer_pore_volumes = np.array([50, 100, 200])  # m³
    >>>
    >>> # Run infiltration_to_extraction
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )
    >>> cout.shape
    (11,)

    Using array inputs instead of pandas Series:

    >>> # Convert to arrays
    >>> cin_values = cin.values
    >>> flow_values = flow.values
    >>>
    >>> cout = infiltration_to_extraction(
    ...     cin=cin_values,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )

    With constant retardation factor (linear sorption):

    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     retardation_factor=2.0,  # Compound moves twice as slowly
    ... )

    Note: For concentration-dependent retardation (nonlinear sorption),
    use `infiltration_to_extraction_nonlinear_sorption` instead, as this
    function only supports constant (float) retardation factors.

    Using single pore volume:

    >>> single_volume = np.array([100])  # Single 100 m³ pore volume
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=single_volume,
    ... )
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    # Convert to arrays for vectorized operations
    cin = np.asarray(cin)
    flow = np.asarray(flow)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes)

    _validate_advection_inputs(
        tedges=tedges,
        flow=flow,
        retardation_factor=retardation_factor,
        aquifer_pore_volumes=aquifer_pore_volumes,
        cin_values=cin,
    )

    weight_tedges, weight_flow, weight_cin, threshold, _ = _resolve_spinup_inputs(
        spinup,
        tedges=tedges,
        flow=flow,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        cin=cin,
    )
    assert weight_cin is not None  # noqa: S101 -- narrowed: cin was passed in
    band_vals, col_start, contributing_bins, zero_flow_cout = _infiltration_to_extraction_weights(
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=weight_flow,
        retardation_factor=retardation_factor,
    )
    weights, _, invalid_mask = _resolve_spinup_mask(
        band_vals=band_vals,
        col_start=col_start,
        contributing_bins=contributing_bins,
        zero_flow_cout=zero_flow_cout,
        n_pv=len(aquifer_pore_volumes),
        spinup=threshold,
    )

    # Banded flow-weighted average: row k contributes cin over its narrow band only.
    # Out-of-range band slots carry zero weight, so the clipped gather is harmless.
    n_cin = len(weight_cin)
    cols = np.clip(col_start[:, None] + np.arange(weights.shape[1]), 0, n_cin - 1)
    out = np.einsum("kb,kb->k", weights, weight_cin[cols])

    # Invalid rows (cout bins where the spin-up policy is not satisfied or
    # where extraction flow was zero) become NaN.
    out[invalid_mask] = np.nan

    return out


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    retardation_factor: float = 1.0,
    regularization_strength: float = 1e-10,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute the concentration of the infiltrating water from extracted water (deconvolution).

    Inverts the forward transport model by solving the linear system
    ``W_forward @ cin = cout`` where ``W_forward`` is the weight matrix from
    :func:`infiltration_to_extraction`. Uses Tikhonov regularization to
    smoothly blend data fitting with a physically motivated target
    (transpose-and-normalize of the forward matrix).

    Well-determined modes (large singular values relative to √λ) are
    dominated by the data; poorly-determined modes are pulled toward the
    target. This avoids edge oscillations and is less sensitive to the
    regularization parameter than truncated SVD (``rcond``).

    Parameters
    ----------
    cout : array-like
        Concentration values of extracted water [concentration units].
        Length must match the number of time bins defined by cout_tedges. The model
        assumes this value is constant over each interval
        ``[cout_tedges[i], cout_tedges[i+1])``.
    flow : array-like
        Flow rate values in the aquifer [m³/day].
        Length must match the number of time bins defined by tedges. The model assumes
        this value is constant over each interval ``[tedges[i], tedges[i+1])``.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cin (output) and flow data. Has length of
        len(flow) + 1. Output cin has length len(tedges) - 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Has length of len(cout) + 1.
        Can have different time alignment and resolution than tedges.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m³] representing the distribution
        of residence times in the aquifer system.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption/interaction.
    regularization_strength : float, optional
        Tikhonov regularization parameter λ. Controls the tradeoff between
        fitting the data (``||W cin - cout||²``) and staying close to the
        regularization target (``λ ||cin - cin_target||²``). The target is
        the transpose-and-normalize of the forward matrix applied to cout.

        Larger values trust the target more (smoother, more biased); smaller
        values trust the data more (noisier, less biased). The solution
        varies continuously with λ. Default is 1e-10.

        A good starting value for noisy data is
        ``λ ≈ (noise_std / signal_amplitude)²``. For example, temperature
        data with 0.05 °C noise and ~10 °C seasonal amplitude suggests
        ``regularization_strength ≈ (0.05 / 10)² ≈ 2.5e-5``. Increase by
        a factor of 2-10 for additional smoothing. For noiseless synthetic
        data (e.g., roundtrip tests), the default 1e-10 preserves machine
        precision.
    spinup : {"constant"} | float in [0, 1] | None, optional
        Spin-up policy applied when building the forward weight matrix
        used to set up the inverse problem. Same semantics as in
        :func:`infiltration_to_extraction`; default ``"constant"`` shifts
        ``tedges[0]`` backward by ``retardation_factor *
        max(aquifer_pore_volumes) / flow[0]`` so the inverse problem has
        no spin-up zero-rows for cout bins inside the original tedges
        range. The warm-start prefix is solved for internally but dropped
        before returning, so the output cin stays aligned with the
        user-provided ``tedges`` (length ``len(tedges) - 1``), not the
        padded grid. Passing ``None`` keeps the strict-validity behavior
        (zero-rows in W from incomplete breakthrough).

    Returns
    -------
    numpy.ndarray
        Concentration in the infiltrating water. Same units as cout.
        Length equals len(tedges) - 1 (unchanged whether or not
        ``spinup="constant"`` shifted ``tedges[0]``). NaN values indicate
        cin bins with no temporal overlap with the extraction data. The
        forward weight matrix used to set up the inverse problem treats
        spin-up and zero-flow cout bins as zero-rows according to the
        ``spinup`` policy.

    Raises
    ------
    ValueError
        If tedges length doesn't match flow plus one, if cout_tedges length
        doesn't match cout plus one, or if inputs contain NaN.

    See Also
    --------
    gamma_extraction_to_infiltration : Deconvolution with gamma-distributed pore volumes
    infiltration_to_extraction : Forward operation (flow-weighted averaging)
    gwtransport.residence_time.full : Compute residence times from flow and pore volume
    gwtransport.utils.solve_tikhonov : Solver used for inversion
    :ref:`concept-pore-volume-distribution` : Background on aquifer heterogeneity modeling
    :ref:`concept-transport-equation` : Flow-weighted averaging approach

    Notes
    -----
    NaN values in ``cout`` are rejected. The Tikhonov solver here does not
    mask NaN rows, so any NaN in ``cout`` would poison the solution. This
    differs from :func:`gwtransport.deposition.extraction_to_deposition`,
    whose regularized solver excludes NaN ``cout`` rows by construction.

    Examples
    --------
    Basic usage with pandas Series:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.utils import compute_time_edges
    >>> from gwtransport.advection import extraction_to_infiltration
    >>>
    >>> # Create cin/flow time edges
    >>> cin_dates = pd.date_range(start="2019-12-25", end="2020-01-15", freq="D")
    >>> tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cin_dates, number_of_bins=len(cin_dates)
    ... )
    >>>
    >>> # Create cout time edges
    >>> cout_dates = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = compute_time_edges(
    ...     tedges=None, tstart=None, tend=cout_dates, number_of_bins=len(cout_dates)
    ... )
    >>>
    >>> # Input concentration and flow
    >>> cout = np.ones(len(cout_dates))
    >>> flow = np.ones(len(cin_dates)) * 100  # 100 m³/day
    >>>
    >>> # Define distribution of aquifer pore volumes
    >>> aquifer_pore_volumes = np.array([50, 100, 200])  # m³
    >>>
    >>> # Run extraction_to_infiltration
    >>> cin = extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )
    >>> cin.shape
    (22,)

    Round-trip reconstruction (symmetric with infiltration_to_extraction).
    The default ``spinup="constant"`` warm-starts the left edge; the cout
    window must therefore stay inside the cin window with margin matching
    the longest residence time on the right (forward NaN at the right
    edge would otherwise be rejected by ``extraction_to_infiltration``):

    >>> from gwtransport.advection import infiltration_to_extraction
    >>> rt_cout_dates = pd.date_range(start="2020-01-01", end="2020-01-10", freq="D")
    >>> rt_cout_tedges = compute_time_edges(
    ...     tedges=None,
    ...     tstart=None,
    ...     tend=rt_cout_dates,
    ...     number_of_bins=len(rt_cout_dates),
    ... )
    >>> cin_original = np.sin(np.linspace(0, 2 * np.pi, len(cin_dates))) + 2
    >>> cout_rt = infiltration_to_extraction(
    ...     cin=cin_original,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=rt_cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )
    >>> cin_recovered = extraction_to_infiltration(
    ...     cout=cout_rt,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=rt_cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ... )
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    # Convert to arrays for vectorized operations
    cout = np.asarray(cout)
    flow = np.asarray(flow)

    _validate_advection_inputs(
        tedges=tedges,
        flow=flow,
        retardation_factor=retardation_factor,
        aquifer_pore_volumes=aquifer_pore_volumes,
        cout_values=cout,
        cout_tedges=cout_tedges,
    )

    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes)

    weight_tedges, weight_flow, _, threshold, n_pad = _resolve_spinup_inputs(
        spinup,
        tedges=tedges,
        flow=flow,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
    )
    n_cin_padded = len(weight_tedges) - 1

    band_vals, col_start, contributing_bins, zero_flow_cout = _infiltration_to_extraction_weights(
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=weight_flow,
        retardation_factor=retardation_factor,
    )
    band_vals, _, _ = _resolve_spinup_mask(
        band_vals=band_vals,
        col_start=col_start,
        contributing_bins=contributing_bins,
        zero_flow_cout=zero_flow_cout,
        n_pv=len(aquifer_pore_volumes),
        spinup=threshold,
    )

    cin_padded = solve_inverse_transport_banded(
        band_vals=band_vals,
        col_start=col_start,
        observed=cout,
        n_output=n_cin_padded,
        regularization_strength=regularization_strength,
    )
    # Drop warm-start prefix so the output aligns with the user-provided tedges.
    return cin_padded[n_pad:]


def _validate_front_tracking_inputs(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    freundlich_k: float | None,
    freundlich_n: float | None,
    bulk_density: float | None,
    porosity: float | None,
    retardation_factor: float | None,
    langmuir_s_max: float | None,
    langmuir_k_l: float | None,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    pd.DatetimeIndex,
    pd.DatetimeIndex,
    npt.NDArray[np.float64],
    SorptionModel,
    npt.NDArray[np.floating],
]:
    """Validate inputs and create sorption object for front tracking functions.

    Returns
    -------
    tuple
        Validated and converted inputs: (cin, flow, tedges, cout_tedges,
        aquifer_pore_volumes, sorption, cout_tedges_days).

    Raises
    ------
    ValueError
        If array lengths are inconsistent, values are non-physical (negative
        concentrations, non-positive flows, NaN values, non-positive pore
        volumes), retardation_factor < 1, Freundlich or Langmuir parameters
        are missing or non-positive, freundlich_n equals 1, or physical
        parameters are invalid.
    """
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    if len(tedges) != len(cin) + 1:
        msg = "tedges must have length len(cin) + 1"
        raise ValueError(msg)
    if len(flow) != len(cin):
        msg = "flow must have same length as cin"
        raise ValueError(msg)
    if np.any(cin < 0):
        msg = "cin must be non-negative"
        raise ValueError(msg)
    if np.any(np.isnan(cin)) or np.any(np.isnan(flow)):
        msg = "cin and flow must not contain NaN"
        raise ValueError(msg)
    if np.any(flow < 0):
        msg = "flow must be non-negative (negative flow not supported)"
        raise ValueError(msg)
    if np.any(aquifer_pore_volumes <= 0):
        msg = "aquifer_pore_volumes must be positive"
        raise ValueError(msg)

    # Convert cout_tedges to days (relative to tedges[0]) for output computation
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])

    # Determine which sorption model is requested
    has_retardation = retardation_factor is not None
    has_freundlich = freundlich_k is not None or freundlich_n is not None
    has_langmuir = langmuir_s_max is not None or langmuir_k_l is not None
    n_models = has_retardation + has_freundlich + has_langmuir

    if n_models == 0:
        msg = (
            "Must provide one of: retardation_factor, Freundlich parameters "
            "(freundlich_k, freundlich_n, bulk_density, porosity), or Langmuir parameters "
            "(langmuir_s_max, langmuir_k_l, bulk_density, porosity)"
        )
        raise ValueError(msg)
    if n_models > 1:
        msg = "Only one sorption model can be specified (retardation_factor, Freundlich, or Langmuir)"
        raise ValueError(msg)

    # Create sorption object
    if retardation_factor is not None:
        _validate_retardation_factor(retardation_factor)
        sorption: SorptionModel = ConstantRetardation(retardation_factor=retardation_factor)
    elif has_freundlich:
        if freundlich_k is None or freundlich_n is None or bulk_density is None or porosity is None:
            msg = "All Freundlich parameters required (freundlich_k, freundlich_n, bulk_density, porosity)"
            raise ValueError(msg)
        if freundlich_k <= 0 or freundlich_n <= 0:
            msg = "Freundlich parameters must be positive"
            raise ValueError(msg)
        if abs(freundlich_n - 1.0) < EPSILON_FREUNDLICH_N:
            msg = "freundlich_n = 1 not supported (use retardation_factor for linear case)"
            raise ValueError(msg)
        if bulk_density <= 0 or not 0 < porosity < 1:
            msg = "Invalid physical parameters"
            raise ValueError(msg)

        sorption = FreundlichSorption(
            k_f=freundlich_k,
            n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity,
        )
    else:
        if langmuir_s_max is None or langmuir_k_l is None or bulk_density is None or porosity is None:
            msg = "All Langmuir parameters required (langmuir_s_max, langmuir_k_l, bulk_density, porosity)"
            raise ValueError(msg)
        if langmuir_s_max <= 0 or langmuir_k_l <= 0:
            msg = "Langmuir parameters must be positive"
            raise ValueError(msg)
        if bulk_density <= 0 or not 0 < porosity < 1:
            msg = "Invalid physical parameters"
            raise ValueError(msg)

        sorption = LangmuirSorption(
            s_max=langmuir_s_max,
            k_l=langmuir_k_l,
            bulk_density=bulk_density,
            porosity=porosity,
        )

    return cin, flow, tedges, cout_tedges, aquifer_pore_volumes, sorption, cout_tedges_days


def _theta_at_last_output_edge(
    cout_tedges_days: npt.NDArray[np.floating],
    flow_tedges_days: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
) -> float:
    """Cumulative flow θ at the last requested output edge (the interaction-resolution horizon).

    Piecewise-linear ``t → θ`` map (identical to ``FrontTrackerState.theta_at_t``),
    extrapolating the final-bin flow when the output window extends past the flow record.

    Returns
    -------
    float
        Cumulative flow θ [m³] at ``cout_tedges_days[-1]``.
    """
    theta_edges = np.concatenate(([0.0], np.cumsum(flow * np.diff(flow_tedges_days))))
    t = float(cout_tedges_days[-1])
    if t <= flow_tedges_days[0]:
        return float(theta_edges[0])
    if t >= flow_tedges_days[-1]:
        return float(theta_edges[-1] + (t - flow_tedges_days[-1]) * float(flow[-1]))
    i = int(np.searchsorted(flow_tedges_days, t, side="right")) - 1
    return float(theta_edges[i] + (t - flow_tedges_days[i]) * float(flow[i]))


def _flow_weighted_front_tracking_output(
    cout_tedges_days: npt.NDArray[np.floating],
    flow_tedges_days: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    v_outlet: float,
    waves: list,
    sorption: SorptionModel,
    theta_edges: npt.NDArray[np.floating],
    cin: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Compute flow-weighted bin-averaged concentration from front-tracking output.

    Splits output bins at flow boundaries so that Q is constant within each
    sub-bin, then combines sub-bins with flow-weighting:
    ``c_avg = Σ(Q_k · c_k · dt_k) / Σ(Q_k · dt_k)``.

    Internally translates the output ``t``-bin edges to θ via the same
    ``(flow_tedges_days, theta_edges)`` map the tracker built, and calls
    :func:`compute_bin_averaged_concentration_exact` in θ-coordinates.

    Parameters
    ----------
    cout_tedges_days : ndarray
        Output time bin edges [days from reference].
    flow_tedges_days : ndarray
        Flow time bin edges [days from reference] (length ``len(flow) + 1``).
    flow : ndarray
        Flow rate per flow bin [m³/day].
    v_outlet : float
        Outlet volume position [m³].
    waves : list
        Wave list from front tracking simulation.
    sorption : object
        Sorption model.
    theta_edges : ndarray
        Cumulative-flow edges at the flow-bin boundaries [m³]
        (length ``len(flow) + 1``).
    cin : ndarray
        Infiltration concentration values, one per flow bin. Passed directly to
        :func:`compute_bin_averaged_concentration_exact`.

    Returns
    -------
    ndarray
        Flow-weighted bin-averaged concentrations. Length = len(cout_tedges_days) - 1.
    """
    inner_flow_edges = flow_tedges_days[
        (flow_tedges_days > cout_tedges_days[0]) & (flow_tedges_days < cout_tedges_days[-1])
    ]
    fine_edges = np.unique(np.concatenate([cout_tedges_days, inner_flow_edges]))

    # np.interp clips on both sides; extrapolate the θ map past either flow edge at
    # the adjacent-bin flow (matches the FrontTrackerState.theta_at_t rule). Without
    # this, out-of-window fine_edges collapse to a duplicate θ. A θ edge ≤ 0
    # downstream reads back as m = 0 → 0.0 (the documented out-of-range contract).
    fine_theta_edges = np.interp(fine_edges, flow_tedges_days, theta_edges)
    underflow = fine_edges < flow_tedges_days[0]
    if underflow.any():
        fine_theta_edges[underflow] = theta_edges[0] - (flow_tedges_days[0] - fine_edges[underflow]) * float(flow[0])
    overflow = fine_edges > flow_tedges_days[-1]
    if overflow.any():
        fine_theta_edges[overflow] = theta_edges[-1] + (fine_edges[overflow] - flow_tedges_days[-1]) * float(flow[-1])

    # A zero-flow input span leaves θ stationary, so its sub-bins have zero width in
    # θ and zero q·dt weight. Drop them before the exact averaging (which rejects
    # non-positive-width bins); their concentration cannot affect the flow-weighted
    # mean, so they read back as 0 and carry no weight. Consecutive kept bins stay
    # contiguous because the dropped bins share their neighbours' θ value.
    theta_lo, theta_hi = fine_theta_edges[:-1], fine_theta_edges[1:]
    nondegenerate = theta_hi > theta_lo
    c_fine = np.zeros(theta_lo.shape)
    if nondegenerate.any():
        kept_edges = np.concatenate([theta_lo[nondegenerate], theta_hi[nondegenerate][-1:]])
        c_fine[nondegenerate] = compute_bin_averaged_concentration_exact(
            theta_bin_edges=kept_edges,
            v_outlet=v_outlet,
            waves=waves,
            sorption=sorption,
            cin=cin,
            theta_edges_inlet=theta_edges,
        )

    # Map each fine sub-bin to its flow value. side="right" enforces the
    # half-open [t_k, t_{k+1}) bin convention if a midpoint ever lands
    # exactly on an inner flow edge (does not happen for np.unique-derived
    # midpoints in practice, but is defensible against floating-point drift).
    fine_mids = (fine_edges[:-1] + fine_edges[1:]) / 2
    flow_idx = np.searchsorted(flow_tedges_days[1:], fine_mids, side="right")
    flow_idx = np.clip(flow_idx, 0, len(flow) - 1)
    q_fine = flow[flow_idx]
    dt_fine = np.diff(fine_edges)

    # Map each fine sub-bin to its original output bin. Same side="right"
    # rationale as above.
    cout_bin_idx = np.searchsorted(cout_tedges_days[1:], fine_mids, side="right")
    cout_bin_idx = np.clip(cout_bin_idx, 0, len(cout_tedges_days) - 2)

    # Vectorized per-bin flow-weighted average:
    # c_out[k] = sum_i (Q_i * c_i * dt_i) / sum_i (Q_i * dt_i) for fine sub-bins i in bin k
    n_cout = len(cout_tedges_days) - 1
    qdt_product = q_fine * dt_fine
    cqdt_product = c_fine * qdt_product
    denominator = np.bincount(cout_bin_idx, weights=qdt_product, minlength=n_cout)
    numerator = np.bincount(cout_bin_idx, weights=cqdt_product, minlength=n_cout)
    # A zero-throughflow output bin (all overlapping input bins have zero flow) has
    # an undefined flow-weighted average: emit NaN, matching the linear sibling.
    # Pre-record bins keep positive throughflow, so their 0-mass windows read as 0.0.
    c_out = np.full(n_cout, np.nan)
    valid = denominator > 0
    c_out[valid] = numerator[valid] / denominator[valid]
    return c_out


def infiltration_to_extraction_nonlinear_sorption(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    freundlich_k: float | None = None,
    freundlich_n: float | None = None,
    bulk_density: float | None = None,
    porosity: float | None = None,
    retardation_factor: float | None = None,
    langmuir_s_max: float | None = None,
    langmuir_k_l: float | None = None,
    max_iterations: int = 10000,
) -> tuple[npt.NDArray[np.floating], list[dict]]:
    """
    Compute extracted concentration with complete diagnostic information.

    Returns both bin-averaged concentrations and detailed simulation structure for each pore volume.

    Exactly one sorption model must be specified:

    - ``retardation_factor`` for constant (linear) retardation.
    - ``freundlich_k`` + ``freundlich_n`` + ``bulk_density`` + ``porosity`` for
      Freundlich isotherm.
    - ``langmuir_s_max`` + ``langmuir_k_l`` + ``bulk_density`` + ``porosity`` for
      Langmuir isotherm.

    Parameters
    ----------
    cin : array-like
        Infiltration concentration [mg/L or any units].
        Length = len(tedges) - 1. The model assumes this value is constant over each
        interval ``[tedges[i], tedges[i+1])``.
    flow : array-like
        Flow rate [m³/day]. Must be non-negative.
        Length = len(tedges) - 1. The model assumes this value is constant over each
        interval ``[tedges[i], tedges[i+1])``.
    tedges : pandas.DatetimeIndex
        Time bin edges. Length = len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Output time bin edges. Can be different from tedges.
        Length = number of output bins + 1 (n+1 edges for n output values).
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m³] representing the distribution
        of residence times in the aquifer system. Each pore volume must be positive.
    freundlich_k : float, optional
        Freundlich coefficient [(m³/kg)^(1/n)]. Must be positive. Same convention (isotherm
        ``s = k_f * C^(1/n)``) as :func:`gwtransport.residence_time.freundlich_retardation`.
    freundlich_n : float, optional
        Freundlich exponent [-]. Must be positive and != 1.
    bulk_density : float, optional
        Bulk density [kg/m³]. Must be positive.
        Shared by Freundlich and Langmuir models.
    porosity : float, optional
        Porosity [-]. Must be in (0, 1).
        Shared by Freundlich and Langmuir models.
    retardation_factor : float, optional
        Constant retardation factor [-]. Must be >= 1.0.
    langmuir_s_max : float, optional
        Langmuir maximum sorption capacity [mg/kg]. Must be positive.
    langmuir_k_l : float, optional
        Langmuir half-saturation constant [mg/L]. Must be positive.
    max_iterations : int, optional
        Maximum number of events. Default 10000.

    Returns
    -------
    cout : numpy.ndarray
        Flow-weighted concentrations averaged across all pore volumes. Output
        bins whose source window leaves the inlet flow record (e.g. cout bins
        before first breakthrough, or extending past the flow record) are
        returned as ``0.0``, not NaN; the front-tracking solver clamps such
        out-of-range windows to the last known state rather than masking them.
        An output bin with zero throughflow (every overlapping input bin has
        zero flow) has an undefined flow-weighted average and is returned as
        NaN, matching :func:`infiltration_to_extraction`.

    structures : list of dict
        List of detailed simulation structures, one for each pore volume, with keys:

        - 'waves': List[Wave] - All wave objects created during simulation
        - 'events': List[dict] - All events; each record carries ``"theta"`` (m³)
          and ``"type"``. Translate to user-facing time t via
          ``tracker_state.t_at_theta(event["theta"])`` if needed.
        - 'theta_first_arrival': float - Cumulative flow at first nonzero arrival [m³]
        - 'n_events': int - Total number of events
        - 'n_shocks': int - Number of shocks created
        - 'n_rarefactions': int - Number of rarefactions created
        - 'n_characteristics': int - Number of characteristics created
        - 'theta_current': float - Final simulation cumulative flow [m³]
        - 'sorption': SorptionModel - Sorption object
        - 'tracker_state': FrontTrackerState - Complete simulation state
        - 'aquifer_pore_volume': float - Pore volume for this simulation

    Raises
    ------
    RuntimeError
        If the front-tracking solver leaves an unresolved wave interaction (the domain-mass
        field over-counts stored mass). This is an internal-consistency tripwire, not an
        input limitation — every wave interaction is now composed; a firing indicates a bug.

    See Also
    --------
    infiltration_to_extraction : Convolution-based approach for linear retardation
    gamma_infiltration_to_extraction : For distributions of pore volumes
    :ref:`concept-nonlinear-sorption` : Freundlich isotherm and front-tracking theory
    :ref:`assumption-advection-dominated` : When diffusion/dispersion is negligible

    Examples
    --------
    .. disable_try_examples

    ::

        cout, structures = infiltration_to_extraction_nonlinear_sorption(
            cin=cin,
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=np.array([500.0]),
            freundlich_k=0.01,
            freundlich_n=2.0,
            bulk_density=1500.0,
            porosity=0.3,
        )

        # Access spin-up period for first pore volume
        theta_first = structures[0]["theta_first_arrival"]
        t_first = structures[0]["tracker_state"].t_at_theta(theta_first)
        print(f"First arrival: θ={theta_first:.2f} m³ (t={t_first:.2f} days)")

        # Analyze events for first pore volume
        for event in structures[0]["events"]:
            print(f"θ={event['theta']:.2f}: {event['type']}")
    """
    cin, flow, tedges, cout_tedges, aquifer_pore_volumes, sorption, cout_tedges_days = _validate_front_tracking_inputs(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        freundlich_k=freundlich_k,
        freundlich_n=freundlich_n,
        bulk_density=bulk_density,
        porosity=porosity,
        retardation_factor=retardation_factor,
        langmuir_s_max=langmuir_s_max,
        langmuir_k_l=langmuir_k_l,
    )

    # Flow time edges in days (same reference as cout_tedges_days)
    flow_tedges_days = tedges_to_days(tedges)

    # Each pore-volume bin from the gamma distribution is an equal-mass streamtube,
    # so all streamtubes carry equal flow at the outlet. The bundle outlet
    # concentration is the simple arithmetic mean over streamtubes. Accumulate the
    # per-streamtube output into a running sum so peak memory stays O(n_cout)
    # rather than O(n_pv * n_cout).
    cout_sum = np.zeros(len(cout_tedges) - 1)
    structures = []

    # Resolve wave interactions out to the last requested output edge (extrapolating the
    # final-bin flow if the output window extends past the flow record), so beyond-outlet
    # merges that still feed an in-window outlet query are composed.
    theta_horizon = _theta_at_last_output_edge(cout_tedges_days, flow_tedges_days, flow)

    for aquifer_pore_volume in aquifer_pore_volumes:
        tracker = FrontTracker(
            cin=cin,
            flow=flow,
            tedges=tedges,
            aquifer_pore_volume=aquifer_pore_volume,
            sorption=sorption,
            theta_horizon=theta_horizon,
        )

        tracker.run(max_iterations=max_iterations)

        # The solver now resolves every wave interaction (shock↔shock, fan-entry, doubly-fed
        # formation, same-apex annihilation and their compositions), so the wave list is
        # interaction-consistent and the single-owner reader is exact. This detector is now a
        # tripwire: a firing means the solver left an unresolved interaction (an internal bug),
        # not an unsupported input — fail loud rather than return a silently wrong cout.
        interaction = find_unresolved_interaction(tracker.state)
        if interaction is not None:
            msg = (
                "infiltration_to_extraction_nonlinear_sorption: the front-tracking solver left an "
                f"unresolved wave interaction ({interaction}). This is an internal inconsistency; "
                "please report it with the cin/flow/pore-volume inputs at "
                "https://github.com/gwtransport/gwtransport/issues."
            )
            raise RuntimeError(msg)

        cout_sum += _flow_weighted_front_tracking_output(
            cout_tedges_days=cout_tedges_days,
            flow_tedges_days=flow_tedges_days,
            flow=flow,
            v_outlet=aquifer_pore_volume,
            waves=tracker.state.waves,
            sorption=sorption,
            theta_edges=tracker.state.theta_edges,
            cin=cin,
        )

        wave_counts = Counter(type(w) for w in tracker.state.waves)
        structure = {
            "waves": tracker.state.waves,
            "events": tracker.state.events,
            "theta_first_arrival": tracker.theta_first_arrival,
            "n_events": len(tracker.state.events),
            "n_shocks": wave_counts[ShockWave],
            "n_rarefactions": wave_counts[RarefactionWave],
            "n_characteristics": wave_counts[CharacteristicWave],
            "theta_current": tracker.state.theta_current,
            "sorption": sorption,
            "tracker_state": tracker.state,
            "aquifer_pore_volume": aquifer_pore_volume,
        }
        structures.append(structure)

    return cout_sum / len(aquifer_pore_volumes), structures
