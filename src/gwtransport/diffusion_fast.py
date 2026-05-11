"""
Fast diffusive transport via V-coordinate Gaussian smoothing.

This module computes 1D solute (or thermal) transport with longitudinal
dispersion using moment-averaged Gaussian smoothing applied in
cumulative-volume (V) coordinates, then composed with the exact-advection
streamtube weight matrix. Two branches are exposed:

- ``flow_out is None`` — smooth-then-advect on the natural input V-grid.
- ``flow_out is not None`` — advect-then-smooth on the natural output V-grid.

In both cases the smoothing operator is built by resampling onto a uniform
V-grid, applying a row-normalised Gaussian (sigma in V-units), and rebinning
back; both the resampling and the on-uniform Gaussian preserve constants
exactly, and the V-weighted column sum residual is bounded by the variable-
sigma_V slope (typically below ``1e-2`` for realistic aquifer parameters,
and exactly zero when ``sigma_V`` is V-independent, i.e. when only
longitudinal dispersivity acts).

The reported outlet concentration is an approximation of Bear's *resident*
concentration ``C_R``. The Kreft-Zuber (1978) flux concentration
``C_F = C_R + (D_L/v) * N_x`` adds a Gaussian-density correction local to
the breakthrough; the slow :mod:`gwtransport.diffusion` module integrates
``C_F`` cell-wise via 16-point Gauss-Legendre quadrature and remains
strictly non-negative. Use that module when the Kreft-Zuber flux semantic
matters; this fast module trades that ``O(1/Pe)`` shape correction for a
simpler sparse-matrix operator that respects the convex-combination bound.

Both ``diffusion_fast`` and :mod:`gwtransport.diffusion` add microdispersion
and molecular diffusion on top of macrodispersion captured by the aquifer
pore-volume distribution (APVD). See :ref:`concept-dispersion` for
background.

Available functions:

- :func:`infiltration_to_extraction` — forward transport.
- :func:`extraction_to_infiltration` — inverse via Tikhonov regularisation.
- :func:`gamma_infiltration_to_extraction` — convenience wrapper for a
  gamma-distributed APVD (forward).
- :func:`gamma_extraction_to_infiltration` — same, inverse.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse
from scipy.special import erf

from gwtransport import gamma
from gwtransport.advection_utils import (
    _infiltration_to_extraction_weights,
    _resolve_spinup_inputs_wide_edges,
    _resolve_spinup_mask,
)
from gwtransport.residence_time import residence_time_mean
from gwtransport.utils import solve_inverse_transport

# Minimum coefficient sum to consider a row valid
_EPSILON_COEFF_SUM = 1e-10


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute diffusion during 1D transport using fast Gaussian smoothing.

    Combines advection (via pore volume distribution) with diffusion (via Gaussian
    smoothing) to produce extraction concentrations on the ``cout_tedges`` time grid.
    This is the fast approximate counterpart of
    :func:`gwtransport.diffusion.infiltration_to_extraction`.

    When ``flow_out`` is provided, diffusion is applied on the output grid
    (advect-then-smooth); otherwise it is applied on the input grid
    (smooth-then-advect). Both branches operate in cumulative-volume (V)
    coordinates via uniform-V resampling, so non-uniform ``tedges`` and
    non-uniform ``cout_tedges`` are handled without an explicit uniformity
    check.

    Parameters
    ----------
    cin : array-like
        Concentration or temperature of the compound in the infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Has length len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3] representing the distribution of
        flow paths.
    mean_streamline_length : float
        Mean travel distance [m]. Must be positive.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity [m2/day]. Must be non-negative.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations.
        Default is 3.0.
    flow_out : array-like or None, optional
        Flow rate [m3/day] on the output grid (aligned to ``cout_tedges``).
        Required when ``tedges`` has non-uniform time steps. Length must
        equal ``len(cout_tedges) - 1``. Default is None.
    spinup : {"constant"} | float in [0, 1] | None, optional
        Forwarded to the underlying advection weight computation. Default
        ``"constant"`` prepends warm-start bins (each of width
        ``tedges[1] - tedges[0]``) so that left-edge spin-up does not
        produce NaN cout. The first observed cin and flow are used as
        the constant warm-start values.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in extracted water. Length equals
        len(cout_tedges) - 1. NaN values indicate time periods with no valid
        contributions from the infiltration data.

    See Also
    --------
    gwtransport.diffusion.infiltration_to_extraction : Physically rigorous analytical solution
        that supports per-pore-volume arrays for streamline_length, molecular_diffusivity,
        and longitudinal_dispersivity.
    extraction_to_infiltration : Inverse operation
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    A single ``mean_streamline_length`` is shared across all pore volumes. This
    assumes streamline-length heterogeneity is captured solely through the
    pore-volume distribution: the effective cross-sectional area
    ``A_eff = V_p / L_mean`` varies with V_p while the path length L is held
    fixed. This is appropriate for many systems but breaks down for
    partially-penetrating wells or wedge-shaped capture zones, where path
    length itself varies between streamtubes. In those cases, use
    :func:`gwtransport.diffusion.infiltration_to_extraction` with a per-streamtube
    ``streamline_length`` array instead.
    """
    # Convert inputs
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    tedges = pd.DatetimeIndex(tedges)
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    if flow_out is not None:
        flow_out = np.asarray(flow_out, dtype=float)

    n_pore_volumes = len(aquifer_pore_volumes)

    # Input validation
    _validate_inputs(
        cin_or_cout=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        is_forward=True,
        flow_out=flow_out,
    )

    # Dispersion warning
    if n_pore_volumes > 1 and mean_longitudinal_dispersivity > 0 and not suppress_dispersion_warning:
        _warn_dispersion()

    # Apply spinup policy to inputs (extend boundary tedges for "constant").
    # diffusion_fast uses diffusion-style wide-edge padding rather than
    # advection's physics-precise n_pad bins, because the Gaussian smoothing
    # kernel adds a tail that R*V_max/Q alone does not capture.
    weight_tedges, threshold = _resolve_spinup_inputs_wide_edges(spinup, tedges=tedges)

    # Build advection weight matrix (needed in both branches)
    accumulated_weights, contributing_bins, zero_flow_cout = _infiltration_to_extraction_weights(
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=flow,
        retardation_factor=retardation_factor,
    )
    w_adv, adv_invalid_mask = _resolve_spinup_mask(
        accumulated_weights=accumulated_weights,
        contributing_bins=contributing_bins,
        zero_flow_cout=zero_flow_cout,
        n_pv=len(aquifer_pore_volumes),
        spinup=threshold,
    )

    # Cumulative-volume edges on the user's natural ``tedges`` (no wide
    # spin-up). The smoothing operator runs on this grid so the uniform-V
    # resampling does not have to span a 100-year warm-start interval.
    weight_dt_days = (np.diff(weight_tedges) / pd.Timedelta("1D")).astype(float)
    v_in_widedge_edges = np.concatenate(([0.0], np.cumsum(flow * weight_dt_days)))
    natural_dt_days = (np.diff(tedges) / pd.Timedelta("1D")).astype(float)
    v_in_natural_edges = np.concatenate(([0.0], np.cumsum(flow * natural_dt_days)))

    # Output-side V-edges expressed in the SAME cumulative-volume reference
    # frame as ``v_in_widedge_edges`` (anchored at ``weight_tedges[0]``).
    # ``flow_out`` is used only for sigma_V; the V-grid is set by the
    # underlying flow so it is consistent with ``w_adv``.
    cout_dt_days = (np.diff(cout_tedges) / pd.Timedelta("1D")).astype(float)
    tedges_days = ((weight_tedges - weight_tedges[0]) / pd.Timedelta("1D")).to_numpy(dtype=float)
    cout_tedges_days = ((cout_tedges - weight_tedges[0]) / pd.Timedelta("1D")).to_numpy(dtype=float)
    v_out_edges = np.interp(cout_tedges_days, tedges_days, v_in_widedge_edges)
    if flow_out is None:
        # Per-cout-bin flux inferred from the V-edge increments. Matches
        # the slow ``diffusion`` module's ``flow_during_cout``.
        flow_out_arr = np.where(cout_dt_days > 0.0, np.diff(v_out_edges) / cout_dt_days, 0.0)
    else:
        flow_out_arr = np.asarray(flow_out, dtype=float)

    # Moment-averaged sigma_V in V-coordinate units [m^3] on the output
    # grid: used by both the K-Z correction and (when applicable) the
    # output-side smoothing matrix.
    sigma_v_out = _compute_sigma_v(
        flow=flow_out_arr,
        tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=mean_streamline_length,
        molecular_diffusivity=mean_molecular_diffusivity,
        longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )

    if flow_out is None:
        # Smooth-then-advect: V-coord smoothing on the natural input grid,
        # then exact advection (with wide-edge spin-up) maps to the output.
        sigma_v_in = _compute_sigma_v(
            flow=flow,
            tedges=tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=mean_streamline_length,
            molecular_diffusivity=mean_molecular_diffusivity,
            longitudinal_dispersivity=mean_longitudinal_dispersivity,
            retardation_factor=retardation_factor,
            direction="infiltration_to_extraction",
        )
        m_v_in = _build_v_smooth_matrix(
            v_edges=v_in_natural_edges,
            sigma_v_per_bin=sigma_v_in,
            asymptotic_cutoff_sigma=asymptotic_cutoff_sigma if asymptotic_cutoff_sigma is not None else 4.0,
        )
        cin_diffused = m_v_in @ np.asarray(cin, dtype=float)
        cout = w_adv @ cin_diffused
    else:
        # Advect-then-smooth: pure advection produces ``cout_unsmoothed`` on
        # the output grid, then V-coord smoothing applies the dispersive
        # spread. Validity normalisation handles cout bins that pre-date
        # streamtube breakthrough or fall in zero-flow windows.
        cout_unsmoothed = w_adv @ np.asarray(cin, dtype=float)
        total_coeff = np.sum(w_adv, axis=1)
        zero_row_mask = total_coeff < _EPSILON_COEFF_SUM
        cout_unsmoothed[zero_row_mask] = 0.0
        validity = np.where(zero_row_mask, 0.0, 1.0)

        m_v_out = _build_v_smooth_matrix(
            v_edges=v_out_edges,
            sigma_v_per_bin=sigma_v_out,
            asymptotic_cutoff_sigma=asymptotic_cutoff_sigma if asymptotic_cutoff_sigma is not None else 4.0,
        )
        smoothed = m_v_out @ cout_unsmoothed
        smoothed_validity = m_v_out @ validity
        with np.errstate(invalid="ignore"):
            cout = np.where(smoothed_validity > _EPSILON_COEFF_SUM, smoothed / smoothed_validity, np.nan)

    # NOTE: This module reports an approximation of Bear's resident
    # concentration C_R. The full Kreft-Zuber flux concentration
    # C_F = C_R + (D_L/v) * N_x adds a Gaussian-density correction
    # localised near the breakthrough; including it via step-differences
    # would shift the reported concentration by O(1/Pe) and break the
    # convex-combination bound for sharp pulse inputs (cf. slow
    # ``gwtransport.diffusion``, which integrates C_F per cell via
    # 16-point Gauss-Legendre and stays strictly non-negative). Use the
    # slow module when the Kreft-Zuber flux semantic matters; this fast
    # variant prioritises monotone-friendly C_R smoothing.

    # Mark invalid rows as NaN: incomplete streamtube breakthrough (spin-up) or
    # zero flow during the cout bin.
    cout[adv_invalid_mask] = np.nan

    return cout


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    regularization_strength: float = 1e-10,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """Reconstruct infiltration concentration from extracted water via Tikhonov inversion.

    Inverts the forward transport model (Gaussian diffusion + advection) by building
    the combined forward matrix and solving via Tikhonov regularization. This is the
    fast approximate counterpart of
    :func:`gwtransport.diffusion.extraction_to_infiltration`.

    When ``flow_out`` is provided, smoothing is applied on the output V-grid;
    otherwise on the input V-grid. Both paths handle non-uniform ``tedges``
    and ``cout_tedges`` without an explicit uniformity check.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Has length of len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Has length of len(cout) + 1.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m3].
    mean_streamline_length : float
        Mean travel distance [m]. Must be positive.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity [m2/day]. Must be non-negative.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations.
        Default is 3.0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-10.
    flow_out : array-like or None, optional
        Flow rate [m3/day] on the output grid (aligned to ``cout_tedges``).
        Required when ``tedges`` has non-uniform time steps. Length must
        equal ``len(cout_tedges) - 1``. Default is None.
    spinup : {"constant"} | float in [0, 1] | None, optional
        Forwarded to the underlying advection weight computation. Default
        ``"constant"`` warm-starts the inverse problem by extending
        ``tedges`` backward; the recovered cin is sliced back to the
        original tedges length so the public output shape is unchanged.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in infiltrating water. Length equals
        len(tedges) - 1. NaN values indicate time periods with no valid
        contributions from the extraction data.

    Warns
    -----
    UserWarning
        When the forward matrix is rank-deficient. This occurs with constant
        flow when the residence time is an integer multiple of the time step
        width. To fix, adjust ``aquifer_pore_volumes`` slightly (e.g.,
        multiply by 1.001).

    See Also
    --------
    infiltration_to_extraction : Forward operation
    gwtransport.diffusion.extraction_to_infiltration : Analytically correct deconvolution
        that supports per-pore-volume arrays for streamline_length, molecular_diffusivity,
        and longitudinal_dispersivity.
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    A single ``mean_streamline_length`` is shared across all pore volumes. This
    assumes streamline-length heterogeneity is captured solely through the
    pore-volume distribution: the effective cross-sectional area
    ``A_eff = V_p / L_mean`` varies with V_p while the path length L is held
    fixed. This is appropriate for many systems but breaks down for
    partially-penetrating wells or wedge-shaped capture zones, where path
    length itself varies between streamtubes. In those cases, use
    :func:`gwtransport.diffusion.extraction_to_infiltration` with a per-streamtube
    ``streamline_length`` array instead.
    """
    # Convert inputs
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    cout = np.asarray(cout, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    if flow_out is not None:
        flow_out = np.asarray(flow_out, dtype=float)

    n_pore_volumes = len(aquifer_pore_volumes)

    # Input validation
    _validate_inputs(
        cin_or_cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        is_forward=False,
        flow_out=flow_out,
    )

    # Dispersion warning
    if n_pore_volumes > 1 and mean_longitudinal_dispersivity > 0 and not suppress_dispersion_warning:
        _warn_dispersion()

    # Apply spinup policy: extend boundary tedges by 100 years on each side
    # for "constant" mode (diffusion-style wide-edge padding). Lengths are
    # preserved, so no slicing is needed on the recovered cin.
    weight_tedges, threshold = _resolve_spinup_inputs_wide_edges(spinup, tedges=tedges)
    n_cin = len(weight_tedges) - 1

    # Build advection weight matrix on the (possibly padded) inputs.
    accumulated_weights, contributing_bins, zero_flow_cout = _infiltration_to_extraction_weights(
        tedges=weight_tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        flow=flow,
        retardation_factor=retardation_factor,
    )
    w_adv, _ = _resolve_spinup_mask(
        accumulated_weights=accumulated_weights,
        contributing_bins=contributing_bins,
        zero_flow_cout=zero_flow_cout,
        n_pv=len(aquifer_pore_volumes),
        spinup=threshold,
    )

    # Cumulative-volume edges shared between the smoothing matrices on the
    # input and output sides; see the comment block in
    # ``infiltration_to_extraction`` for the reference-frame rationale.
    weight_dt_days = (np.diff(weight_tedges) / pd.Timedelta("1D")).astype(float)
    v_in_widedge_edges = np.concatenate(([0.0], np.cumsum(flow * weight_dt_days)))
    natural_dt_days = (np.diff(tedges) / pd.Timedelta("1D")).astype(float)
    v_in_natural_edges = np.concatenate(([0.0], np.cumsum(flow * natural_dt_days)))

    cout_dt_days = (np.diff(cout_tedges) / pd.Timedelta("1D")).astype(float)
    tedges_days = ((weight_tedges - weight_tedges[0]) / pd.Timedelta("1D")).to_numpy(dtype=float)
    cout_tedges_days = ((cout_tedges - weight_tedges[0]) / pd.Timedelta("1D")).to_numpy(dtype=float)
    v_out_edges = np.interp(cout_tedges_days, tedges_days, v_in_widedge_edges)
    if flow_out is None:
        flow_out_arr = np.where(cout_dt_days > 0.0, np.diff(v_out_edges) / cout_dt_days, 0.0)
    else:
        flow_out_arr = np.asarray(flow_out, dtype=float)

    sigma_v_out = _compute_sigma_v(
        flow=flow_out_arr,
        tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=mean_streamline_length,
        molecular_diffusivity=mean_molecular_diffusivity,
        longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )

    if flow_out is None:
        # Smooth-then-advect: W_forward = w_adv @ M_v_in
        sigma_v_in = _compute_sigma_v(
            flow=flow,
            tedges=tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=mean_streamline_length,
            molecular_diffusivity=mean_molecular_diffusivity,
            longitudinal_dispersivity=mean_longitudinal_dispersivity,
            retardation_factor=retardation_factor,
            direction="infiltration_to_extraction",
        )
        m_v_in = _build_v_smooth_matrix(
            v_edges=v_in_natural_edges,
            sigma_v_per_bin=sigma_v_in,
            asymptotic_cutoff_sigma=asymptotic_cutoff_sigma if asymptotic_cutoff_sigma is not None else 4.0,
        )
        # Force dense ndarray (avoid scipy's np.matrix from sparse @ dense).
        w_forward = np.asarray(w_adv @ m_v_in)
    else:
        # Advect-then-smooth: W_forward = M_v_out @ w_adv
        m_v_out = _build_v_smooth_matrix(
            v_edges=v_out_edges,
            sigma_v_per_bin=sigma_v_out,
            asymptotic_cutoff_sigma=asymptotic_cutoff_sigma if asymptotic_cutoff_sigma is not None else 4.0,
        )
        w_forward = np.asarray(m_v_out @ w_adv)

    # No K-Z correction is applied (see note in ``infiltration_to_extraction``):
    # the forward matrix here is the same C_R-approximating operator used by
    # the forward, ensuring round-trip consistency.

    return solve_inverse_transport(
        w_forward=w_forward,
        observed=cout,
        n_output=n_cin,
        regularization_strength=regularization_strength,
        warn_rank_deficient=True,
    )


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
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute extracted concentration with fast Gaussian diffusion for gamma-distributed pore volumes.

    Combines advective transport (based on gamma-distributed pore volumes) with
    fast Gaussian diffusion. This is a convenience wrapper around
    :func:`infiltration_to_extraction` that parameterizes the aquifer pore volume
    distribution as a (shifted) gamma distribution.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Has length len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
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
        Number of bins to discretize the gamma distribution. Default is 100.
    mean_streamline_length : float
        Mean travel distance through the aquifer [m] averaged over all aquifer
        pore volumes. Must be positive. For per-pore-volume arrays, use
        :func:`gwtransport.diffusion.gamma_infiltration_to_extraction`.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity [m2/day] averaged over all
        aquifer pore volumes. Must be non-negative. For per-pore-volume
        arrays, use :func:`gwtransport.diffusion.gamma_infiltration_to_extraction`.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity [m] averaged over all aquifer pore
        volumes. Must be non-negative. For per-pore-volume arrays, use
        :func:`gwtransport.diffusion.gamma_infiltration_to_extraction`.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations.
        Default is 3.0.
    flow_out : array-like or None, optional
        Flow rate [m3/day] on the output grid (aligned to ``cout_tedges``).
        Required when ``tedges`` has non-uniform time steps. Length must
        equal ``len(cout_tedges) - 1``. Default is None.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in extracted water. Length equals
        len(cout_tedges) - 1. NaN values indicate time periods with no valid
        contributions from the infiltration data.

    See Also
    --------
    infiltration_to_extraction : Transport with explicit pore volume distribution
    gamma_extraction_to_infiltration : Reverse operation (deconvolution)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.diffusion.gamma_infiltration_to_extraction : Physically rigorous analytical
        solution that supports per-pore-volume arrays for streamline_length,
        molecular_diffusivity, and longitudinal_dispersivity.
    gwtransport.advection.gamma_infiltration_to_extraction : Pure advection (no dispersion)
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The APVD is only time-invariant under the steady-streamlines assumption
    (see :ref:`assumption-steady-streamlines`).

    The spreading from the gamma-distributed pore volumes represents macrodispersion
    (aquifer-scale heterogeneity). When ``std`` comes from calibration on measurements,
    it absorbs all mixing: macrodispersion, microdispersion, and an average molecular
    diffusion contribution. When ``std`` comes from streamline analysis, it represents
    macrodispersion only; microdispersion and molecular diffusion can be added via the
    dispersion parameters.
    See :ref:`concept-dispersion-scales` for guidance on when to add microdispersion.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion_fast import gamma_infiltration_to_extraction
    >>>
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>> cin = np.zeros(len(tedges) - 1)
    >>> cin[5:10] = 1.0
    >>> flow = np.ones(len(tedges) - 1) * 100.0
    >>>
    >>> cout = gamma_infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=500.0,
    ...     std=100.0,
    ...     n_bins=5,
    ...     mean_streamline_length=100.0,
    ...     mean_molecular_diffusivity=1e-4,
    ...     mean_longitudinal_dispersivity=1.0,
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        suppress_dispersion_warning=suppress_dispersion_warning,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
        flow_out=flow_out,
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
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    suppress_dispersion_warning: bool = False,
    asymptotic_cutoff_sigma: float | None = 3.0,
    regularization_strength: float = 1e-10,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Reconstruct infiltration concentration from extracted water for gamma-distributed pore volumes.

    Inverts the forward transport model (fast Gaussian diffusion + advection with
    gamma-distributed pore volumes) via Tikhonov regularization. This is a convenience
    wrapper around :func:`extraction_to_infiltration` that parameterizes the aquifer
    pore volume distribution as a (shifted) gamma distribution.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Has length of len(flow) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Has length of len(cout) + 1.
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
        Number of bins to discretize the gamma distribution. Default is 100.
    mean_streamline_length : float
        Mean travel distance through the aquifer [m] averaged over all aquifer
        pore volumes. Must be positive. For per-pore-volume arrays, use
        :func:`gwtransport.diffusion.gamma_extraction_to_infiltration`.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity [m2/day] averaged over all
        aquifer pore volumes. Must be non-negative. For per-pore-volume
        arrays, use :func:`gwtransport.diffusion.gamma_extraction_to_infiltration`.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity [m] averaged over all aquifer pore
        volumes. Must be non-negative. For per-pore-volume arrays, use
        :func:`gwtransport.diffusion.gamma_extraction_to_infiltration`.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    suppress_dispersion_warning : bool, optional
        Suppress warning about combining multiple pore volumes with dispersivity.
        Default is False.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations.
        Default is 3.0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-10.
    flow_out : array-like or None, optional
        Flow rate [m3/day] aligned to ``cout_tedges``. When provided, the
        Gaussian matrix operates on the output grid. Length must equal
        ``len(cout_tedges) - 1``. Default is None.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in infiltrating water. Length equals
        len(tedges) - 1. NaN values indicate time periods with no valid
        contributions from the extraction data.

    See Also
    --------
    extraction_to_infiltration : Deconvolution with explicit pore volume distribution
    gamma_infiltration_to_extraction : Forward operation (convolution)
    gwtransport.gamma.bins : Create gamma distribution bins
    gwtransport.diffusion.gamma_extraction_to_infiltration : Physically rigorous analytical
        solution that supports per-pore-volume arrays for streamline_length,
        molecular_diffusivity, and longitudinal_dispersivity.
    gwtransport.advection.gamma_extraction_to_infiltration : Pure advection (no dispersion)
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The APVD is only time-invariant under the steady-streamlines assumption
    (see :ref:`assumption-steady-streamlines`).

    The spreading from the gamma-distributed pore volumes represents macrodispersion
    (aquifer-scale heterogeneity). When ``std`` comes from calibration on measurements,
    it absorbs all mixing: macrodispersion, microdispersion, and an average molecular
    diffusion contribution. When ``std`` comes from streamline analysis, it represents
    macrodispersion only; microdispersion and molecular diffusion can be added via the
    dispersion parameters.
    See :ref:`concept-dispersion-scales` for guidance on when to add microdispersion.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion_fast import gamma_extraction_to_infiltration
    >>>
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>> cout = np.zeros(len(cout_tedges) - 1)
    >>> cout[5:10] = 1.0
    >>> flow = np.ones(len(tedges) - 1) * 100.0
    >>>
    >>> cin = gamma_extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     mean=500.0,
    ...     std=100.0,
    ...     n_bins=5,
    ...     mean_streamline_length=100.0,
    ...     mean_molecular_diffusivity=1e-4,
    ...     mean_longitudinal_dispersivity=1.0,
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        suppress_dispersion_warning=suppress_dispersion_warning,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
        regularization_strength=regularization_strength,
        flow_out=flow_out,
        spinup=spinup,
    )


def _compute_sigma_v(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float,
    direction: str,
) -> npt.NDArray[np.floating]:
    """Compute Gaussian sigma in cumulative-volume (V) coordinate units [m3].

    The per-streamtube V-coord variance for pore volume V is

        sigma_V^2(V) = (R*V/L)^2 * (2*D_m*tau(V) + 2*alpha_L*L).

    Moment-averaged over the pore-volume distribution by the law of total
    variance,

        E[sigma_V^2] = (R/L)^2 * (2*D_m*tau_bar/V_bar * E[V^3] + 2*alpha_L*L * E[V^2]).

    The result is in V-coordinate units (cubic meters), which is what the
    V-coordinate smoothing operator expects.

    Parameters
    ----------
    flow : array-like
        Flow rate [m3/day].
    tedges : DatetimeIndex
        Time edges for flow data.
    aquifer_pore_volumes : array-like
        Pore volumes [m3]. Moments E[V^2] and E[V^3] are computed from this
        array.
    streamline_length : float
        Streamline length L [m].
    molecular_diffusivity : float
        Effective molecular diffusivity D_m [m2/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].
    retardation_factor : float
        Retardation factor [-].
    direction : {'infiltration_to_extraction', 'extraction_to_infiltration'}
        Direction for residence time computation. ``"infiltration_to_extraction"``
        gives tau at the infiltration bin (parcel's full journey to outlet);
        ``"extraction_to_infiltration"`` gives tau at the extraction bin
        (parcel arriving here).

    Returns
    -------
    ndarray
        sigma_V per bin in V-coordinate units [m3].

    See Also
    --------
    _build_v_smooth_matrix : Builds the V-coordinate smoothing operator
        using these sigma_V values.
    """
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)

    mean_pv = float(np.mean(aquifer_pore_volumes))
    ev2 = float(np.mean(aquifer_pore_volumes**2))
    ev3 = float(np.mean(aquifer_pore_volumes**3))

    rt_array = residence_time_mean(
        flow=flow,
        flow_tedges=tedges,
        tedges_out=tedges,
        aquifer_pore_volumes=mean_pv,
        retardation_factor=retardation_factor,
        direction=direction,
    )[0]

    valid_mask = ~np.isnan(rt_array)
    if np.any(valid_mask):
        rt_array = np.interp(np.arange(len(rt_array)), np.where(valid_mask)[0], rt_array[valid_mask])

    mol_var_x = 2.0 * molecular_diffusivity * rt_array
    disp_var_x = 2.0 * longitudinal_dispersivity * streamline_length

    var_numerator = mol_var_x / mean_pv * ev3 + disp_var_x * ev2

    sigma_v_sq = (retardation_factor / streamline_length) ** 2 * var_numerator
    return np.sqrt(np.maximum(sigma_v_sq, 0.0))


def _build_rebin_matrix(
    *,
    v_src_edges: npt.NDArray[np.floating],
    v_dst_edges: npt.NDArray[np.floating],
) -> sparse.csr_matrix:
    """Build a V-mass-preserving rebinning matrix between two V-grids.

    For piecewise-constant concentration c_src on V-bins defined by
    ``v_src_edges``, the rebinned concentration c_dst on bins defined by
    ``v_dst_edges`` is ``c_dst = R @ c_src``, where

        R[i, k] = (V-overlap of src bin k with dst bin i) / dV_dst[i].

    Properties (assuming dst bins are fully covered by src bins):

    - ``sum_k R[i, k] = 1`` for every i (constant concentration preserved).
    - ``sum_i dV_dst[i] * R[i, k] = dV_src[k]`` for every k (V-mass preserved
      to machine precision).

    Parameters
    ----------
    v_src_edges : ndarray, shape (n_src + 1,)
        Source V-grid edges (monotonically increasing).
    v_dst_edges : ndarray, shape (n_dst + 1,)
        Destination V-grid edges (monotonically increasing).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n_dst, n_src)
        Sparse rebinning matrix.
    """
    v_src = np.asarray(v_src_edges, dtype=float)
    v_dst = np.asarray(v_dst_edges, dtype=float)
    n_src = len(v_src) - 1
    n_dst = len(v_dst) - 1
    dv_dst = np.diff(v_dst)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for k in range(n_src):
        lo = np.maximum(v_src[k], v_dst[:-1])
        hi = np.minimum(v_src[k + 1], v_dst[1:])
        overlap = np.maximum(0.0, hi - lo)
        active = np.where(overlap > 0.0)[0]
        if len(active) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(dv_dst[active] > 0.0, overlap[active] / dv_dst[active], 0.0)
        rows.extend(active.tolist())
        cols.extend([k] * len(active))
        data.extend(weights.tolist())
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_dst, n_src))


def _build_v_smooth_matrix(
    *,
    v_edges: npt.NDArray[np.floating],
    sigma_v_per_bin: npt.NDArray[np.floating],
    refinement: int = 4,
    asymptotic_cutoff_sigma: float | None = 4.0,
) -> sparse.csr_matrix:
    """Build the V-coordinate Gaussian smoothing matrix via uniform-V resampling.

    For piecewise-constant signal c on the (possibly non-uniform) V-grid given
    by ``v_edges``, apply Gaussian smoothing with width ``sigma_v_per_bin[k]``
    [m^3] at each V-bin. Implemented as ``M = R_from @ G @ R_to`` where:

    1. ``R_to`` rebins the input V-grid to a uniform V-grid (mass-preserving).
    2. ``G`` is the standard row-normalized Gaussian on the uniform V-grid,
       with sigma in uniform-bin-index units.
    3. ``R_from`` rebins back to the input V-grid (mass-preserving).

    Properties of M:

    - **Constant preservation**: ``M @ ones = ones`` to machine precision,
      because every operator in the composition has row sums = 1.
    - **Mass conservation**: ``sum_i dV[i] * M[i, k] = dV[k]`` to O(d sigma_V/dV)
      precision, dominated by the variable-sigma discretization error of the
      uniform-V Gaussian step.

    Parameters
    ----------
    v_edges : ndarray, shape (n + 1,)
        Cumulative-volume edges of the input grid.
    sigma_v_per_bin : ndarray, shape (n,)
        Gaussian width in V-coordinate units [m^3] for each bin.
    refinement : int, optional
        Number of uniform-V bins per minimum input bin width. Higher
        refinement reduces V-grid discretization error in the rebinning
        step at increased cost. Default is 4.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the Gaussian kernel at this many standard deviations on
        the uniform-V grid. Default is 4.0.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (n, n)
        Sparse V-coordinate smoothing matrix.
    """
    v_edges = np.asarray(v_edges, dtype=float)
    sigma = np.asarray(sigma_v_per_bin, dtype=float)
    n_orig = len(sigma)
    dv_orig = np.diff(v_edges)

    # Identity when there is no smoothing to do.
    if not np.any(sigma > 0.0):
        return sparse.csr_matrix(sparse.eye(n_orig, dtype=float))

    # Uniform-V grid spacing: min interior bin width / refinement, capped
    # to keep n_uniform tractable when the input grid contains huge
    # warm-start bins from spin-up padding.
    finite_dv = dv_orig[dv_orig > 0.0]
    if finite_dv.size == 0:
        return sparse.csr_matrix(sparse.eye(n_orig, dtype=float))
    dv_uniform_target = float(np.min(finite_dv)) / max(int(refinement), 1)

    v_min, v_max = float(v_edges[0]), float(v_edges[-1])
    span = v_max - v_min
    if span <= 0.0:
        return sparse.csr_matrix(sparse.eye(n_orig, dtype=float))

    # Cap n_uniform to avoid pathological refinement when wide spin-up
    # bins are present. For typical use cases this cap is not reached.
    n_uniform = int(np.ceil(span / dv_uniform_target))
    n_uniform_cap = 50 * max(n_orig, 1)
    n_uniform = max(1, min(n_uniform, n_uniform_cap))
    v_uniform_edges = np.linspace(v_min, v_max, n_uniform + 1)
    dv_uniform_actual = span / n_uniform

    # Interpolate sigma_V onto uniform-grid centers.
    v_orig_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    v_uniform_centers = 0.5 * (v_uniform_edges[:-1] + v_uniform_edges[1:])
    sigma_uniform_v = np.interp(v_uniform_centers, v_orig_centers, sigma)

    # Convert V-space sigma to uniform-bin-index units for the standard
    # Gaussian convolution matrix builder.
    sigma_uniform_idx = sigma_uniform_v / dv_uniform_actual

    r_to = _build_rebin_matrix(v_src_edges=v_edges, v_dst_edges=v_uniform_edges)
    g_uniform = _build_gaussian_matrix(
        n=n_uniform,
        sigma_array=sigma_uniform_idx,
        asymptotic_cutoff_sigma=asymptotic_cutoff_sigma,
    )
    r_from = _build_rebin_matrix(v_src_edges=v_uniform_edges, v_dst_edges=v_edges)

    return (r_from @ g_uniform @ r_to).tocsr()


def _build_gaussian_matrix(
    *, n: int, sigma_array: npt.ArrayLike, asymptotic_cutoff_sigma: float | None = 3.0
) -> sparse.csr_matrix:
    """Build sparse Gaussian convolution matrix with CDF-integrated bin weights.

    Each row represents a position-specific Gaussian kernel where weights are
    computed by integrating the Gaussian CDF over each bin, rather than
    point-sampling the PDF. This is more accurate for small sigma values
    (< ~2 bins) where the PDF peak can fall between bin centers.

    Parameters
    ----------
    n : int
        Length of the signal (number of bins).
    sigma_array : array-like
        Array of sigma values of length n.
    asymptotic_cutoff_sigma : float or None, optional
        Truncate the kernel at this many standard deviations. Set to None to
        disable truncation (use full signal length). Default is 3.0.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse convolution matrix of shape (n, n).
    """
    sigma_array = np.asarray(sigma_array)

    # Handle zero sigma values
    zero_mask = sigma_array == 0
    if np.all(zero_mask):
        return sparse.csr_matrix(sparse.eye(n, dtype=float))  # type: ignore[reportReturnType]

    # Get maximum kernel size and create position arrays
    max_sigma = np.max(sigma_array)
    truncate = asymptotic_cutoff_sigma if asymptotic_cutoff_sigma is not None else n / max(max_sigma, 1.0)
    max_radius = int(truncate * max_sigma + 0.5)

    # Create arrays for all possible kernel positions
    positions = np.arange(-max_radius, max_radius + 1)

    # Create a mask for valid sigma values
    valid_sigma = ~zero_mask
    valid_indices = np.where(valid_sigma)[0]

    # Shape: (n_valid_points, 1)
    center_positions = valid_indices[:, np.newaxis]
    # Shape: (1, max_kernel_size)
    kernel_positions = positions[np.newaxis, :]

    # Calculate CDF-integrated weights for proper bin averaging.
    # Compute erf at unique bin edges, then take differences. Adjacent bins
    # share an edge, so this halves the number of erf evaluations.
    sigmas = sigma_array[valid_sigma][:, np.newaxis]
    sqrt2 = np.sqrt(2)
    edges = np.arange(-max_radius - 0.5, max_radius + 1.5)  # (2*max_radius + 2,)
    erf_at_edges = erf(edges[np.newaxis, :] / (sigmas * sqrt2))  # (n_valid, 2*max_radius+2)
    weights = 0.5 * np.diff(erf_at_edges, axis=1)  # (n_valid, 2*max_radius+1)

    # Normalize each kernel
    weights /= np.sum(weights, axis=1)[:, np.newaxis]

    # Calculate absolute positions in the signal
    absolute_positions = center_positions + kernel_positions

    # Handle boundary conditions (nearest-neighbor extrapolation)
    absolute_positions = np.clip(absolute_positions, 0, n - 1)

    # Create coordinate arrays for sparse matrix
    rows = np.repeat(center_positions, weights.shape[1])
    cols = absolute_positions.ravel()
    data = weights.ravel()

    # Remove zero weights to save memory
    nonzero_mask = data != 0
    rows = rows[nonzero_mask]
    cols = cols[nonzero_mask]
    data = data[nonzero_mask]

    # Add identity matrix elements for zero-sigma positions
    if np.any(zero_mask):
        zero_indices = np.where(zero_mask)[0]
        rows = np.concatenate([rows, zero_indices])
        cols = np.concatenate([cols, zero_indices])
        data = np.concatenate([data, np.ones(len(zero_indices))])

    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


def _validate_inputs(
    *,
    cin_or_cout: np.ndarray,
    flow: np.ndarray,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: np.ndarray,
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    is_forward: bool,
    flow_out: np.ndarray | None = None,
) -> None:
    """Validate inputs for infiltration_to_extraction and extraction_to_infiltration.

    Raises
    ------
    ValueError
        If array lengths are inconsistent, mean_molecular_diffusivity or
        mean_longitudinal_dispersivity are negative, cin or cout or flow
        contain NaN values, aquifer_pore_volumes contains non-positive
        values, or mean_streamline_length is non-positive.
    """
    if is_forward:
        if len(tedges) != len(cin_or_cout) + 1:
            msg = "tedges must have one more element than cin"
            raise ValueError(msg)
    elif len(cout_tedges) != len(cin_or_cout) + 1:
        msg = "cout_tedges must have one more element than cout"
        raise ValueError(msg)
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if mean_molecular_diffusivity < 0:
        msg = "mean_molecular_diffusivity must be non-negative"
        raise ValueError(msg)
    if mean_longitudinal_dispersivity < 0:
        msg = "mean_longitudinal_dispersivity must be non-negative"
        raise ValueError(msg)
    if np.any(np.isnan(cin_or_cout)):
        msg = f"{'cin' if is_forward else 'cout'} contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(np.isnan(flow)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(flow < 0):
        msg = "flow must be non-negative (negative flow not supported)"
        raise ValueError(msg)
    if np.any(aquifer_pore_volumes <= 0):
        msg = "aquifer_pore_volumes must be positive"
        raise ValueError(msg)
    if mean_streamline_length <= 0:
        msg = "mean_streamline_length must be positive"
        raise ValueError(msg)
    if flow_out is not None:
        n_cout = len(cout_tedges) - 1
        if len(flow_out) != n_cout:
            msg = f"flow_out must have length len(cout_tedges) - 1 = {n_cout}, got {len(flow_out)}"
            raise ValueError(msg)
        if np.any(np.isnan(flow_out)):
            msg = "flow_out contains NaN values, which are not allowed"
            raise ValueError(msg)
        if np.any(flow_out < 0):
            msg = "flow_out must be non-negative (negative flow not supported)"
            raise ValueError(msg)


def _warn_dispersion() -> None:
    """Emit warning about combining multiple pore volumes with dispersivity."""
    msg = (
        "Using multiple aquifer_pore_volumes with non-zero mean_longitudinal_dispersivity. "
        "Both represent spreading from velocity heterogeneity at different scales.\n\n"
        "This is appropriate when:\n"
        "  - APVD comes from streamline analysis (explicit geometry)\n"
        "  - You want to add unresolved microdispersion and molecular diffusion\n\n"
        "This may double-count spreading when:\n"
        "  - APVD was calibrated from measurements (microdispersion already included)\n\n"
        "For gamma-parameterized APVD, consider using the 'equivalent APVD std' approach\n"
        "from 05_Diffusion_Dispersion.ipynb to combine both effects in the faster advection\n"
        "module. Note: this variance combination is only valid for continuous (gamma)\n"
        "distributions, not for discrete streamline volumes.\n"
        "Suppress this warning with suppress_dispersion_warning=True."
    )
    warnings.warn(msg, UserWarning, stacklevel=3)
