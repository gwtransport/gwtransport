"""
Fast closed-form 1D advection-dispersion transport (Kreft-Zuber flux concentration).

This module computes the same physics as :mod:`gwtransport.diffusion` -- the Kreft-Zuber
(1978) flux concentration ``C_F`` at the outlet of a bundle of 1D streamtubes -- but
evaluates the bin-averaged breakthrough in closed form instead of by Gauss-Legendre
quadrature.

For each streamtube (one aquifer pore volume) the resident concentration in moving-frame
cumulative-volume (V) coordinates is the Gaussian CDF
``C_R = 0.5 * erfc((L - xi) / (2 * sqrt(D_t)))``, with ``D_t = D_m * tau + alpha_L * xi``
the moving-frame dispersion product. Its bin-average over a cout bin has the closed-form
antiderivative ``I(x) = 0.5*x + 0.5*[x*erf(x/s) + (s/sqrt(pi))*exp(-(x/s)^2)]``,
``s = 2*sqrt(D_t)``. Evaluating ``I`` once per cout edge with ``D_t`` carried *per edge*
and differencing yields the flux concentration ``C_F`` directly -- not merely ``C_R`` --
because ``dD_t/dx = D_m/v + alpha_L = D_L/v`` is exactly the Kreft-Zuber flux coefficient
(using ``d(tau)/dx = 1/v``). The dispersive boundary-flux correction therefore emerges from
the ``D_t`` variation across the bin; no explicit correction term is added.

The elapsed time ``tau`` and travel distance ``xi`` are read directly from the time and
cumulative-volume edges (``tau_ij = t_cout_i - t_cin_j``, ``xi`` geometric), so no per-cell
quadrature and no residence-time inversion is needed. The result reproduces
:mod:`gwtransport.diffusion` to machine precision when the cout grid aligns with the flow
grid (supply ``flow_out`` on the output grid), and the build cost is independent of the
dispersion strength.

Streamtube assumption (no cross-sectional area parameter)
---------------------------------------------------------

Each entry in ``aquifer_pore_volumes`` is an independent 1D streamtube sharing the mean
streamline length ``L``; molecular diffusion enters the V-space variance through
``D_m * tau`` and mechanical dispersion through ``alpha_L * xi``. Per-streamtube
``streamline_length`` / ``molecular_diffusivity`` / ``longitudinal_dispersivity`` arrays
are not supported here -- use :mod:`gwtransport.diffusion` for those.

Available functions:

- :func:`infiltration_to_extraction` -- forward transport.
- :func:`extraction_to_infiltration` -- inverse via Tikhonov regularisation.
- :func:`gamma_infiltration_to_extraction` -- gamma-distributed APVD (forward).
- :func:`gamma_extraction_to_infiltration` -- same, inverse.

References
----------
Kreft, A., & Zuber, A. (1978). On the physical meaning of the dispersion equation and its
solutions for different initial and boundary conditions. Chemical Engineering Science,
33(11), 1471-1480.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import erf

from gwtransport import gamma
from gwtransport._time import dt_to_days, tedges_to_days
from gwtransport.residence_time import residence_time
from gwtransport.utils import cumulative_flow_volume, solve_inverse_transport

# Minimum coefficient sum to consider an output bin valid.
_EPSILON_COEFF_SUM = 1e-10

# sqrt(pi), used in the closed-form breakthrough antiderivative.
_SQRT_PI = np.sqrt(np.pi)

# Floor on the moving-frame dispersion product D_t [m^2] to keep the erf argument finite
# for pre-breakthrough / zero-dispersion edges (where D_t -> 0).
_DT_FLOOR = 1e-30


def _flux_breakthrough_fraction(
    *,
    step_widths: npt.NDArray[np.floating],
    tau: npt.NDArray[np.floating],
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    streamline_len: float,
    retardation_factor: float,
    velocity: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    r"""Bin-averaged Kreft-Zuber flux concentration over each cout bin, per cin edge.

    Closed-form evaluation of the bin-averaged outlet breakthrough. For a unit step
    injected at cin edge *j*, the bin-averaged resident concentration over cout bin *i* is

    .. math::

        \frac{1}{\Delta x}\int C_R\,dx,\qquad
        C_R = \tfrac12\bigl(1 + \operatorname{erf}(x/(2\sqrt{D_t}))\bigr),

    with antiderivative
    :math:`I(x) = \tfrac12 x + \tfrac12[x\,\operatorname{erf}(x/s) + (s/\sqrt\pi)e^{-(x/s)^2}]`,
    :math:`s = 2\sqrt{D_t}`. Evaluating :math:`I` once per cout *edge* and differencing,
    with the moving-frame dispersion product :math:`D_t = D_m\,\tau + \alpha_L\,\xi`
    carried *per edge*, yields the **flux** concentration :math:`C_F` (Kreft & Zuber 1978),
    not merely :math:`C_R`. This is exact, not coincidental: the antiderivative difference
    picks up a :math:`\partial I/\partial D_t \cdot (dD_t/dx)` term, and

    .. math::

        \frac{dD_t}{dx} = D_m\frac{d\tau}{dx} + \alpha_L
                        = \frac{D_m}{v} + \alpha_L = \frac{D_L}{v},

    which is precisely the Kreft-Zuber flux coefficient (using :math:`d\tau/dx = 1/v`). The
    dispersive boundary-flux correction therefore emerges from the ``D_t`` variation across
    the bin; no explicit flux term is added.

    Parameters
    ----------
    step_widths : ndarray, shape (n_cout_edges, n_cin_edges)
        ``x = (V_cout - V_cin - R*V_pore) * L / (R*V_pore)`` at each (cout-edge, cin-edge);
        equals :math:`\xi - L`.
    tau : ndarray, shape (n_cout_edges, n_cin_edges)
        Elapsed time ``t_cout - t_cin`` [day] at each (cout-edge, cin-edge), clipped at 0.
    molecular_diffusivity : float
        Effective molecular diffusivity D_m [m^2/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].
    streamline_len : float
        Streamline length L [m].
    retardation_factor : float
        Retardation factor R [-]. Triggers the K-Z flux-coefficient correction when
        ``R != 1`` and ``D_m > 0`` (see notes in the body).
    velocity : ndarray, shape (n_cout_bins,)
        Fluid (unretarded) velocity ``Q*L/V_pore`` [m/day] in each cout bin, used by the
        retardation correction.

    Returns
    -------
    ndarray, shape (n_cout_bins, n_cin_edges)
        Bin-averaged flux concentration for the step injected at each cin edge.

    References
    ----------
    Kreft, A., & Zuber, A. (1978). On the physical meaning of the dispersion equation and
    its solutions for different initial and boundary conditions. Chemical Engineering
    Science, 33(11), 1471-1480.
    """
    x_lo = step_widths[:-1]
    x_hi = step_widths[1:]
    dx = x_hi - x_lo

    # No dispersion: C_R is the step function H(x); its exact bin-average is the fraction
    # of the cout bin with x > 0 (the limit of the erf form as D_t -> 0).
    if molecular_diffusivity == 0.0 and longitudinal_dispersivity == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = (np.maximum(x_hi, 0.0) - np.maximum(x_lo, 0.0)) / dx
        return np.where(dx > 0.0, frac, 0.5 + 0.5 * np.sign(x_lo))

    xi = step_widths + streamline_len
    dt_var = molecular_diffusivity * tau + longitudinal_dispersivity * np.maximum(xi, 0.0)
    dt_var = np.maximum(dt_var, _DT_FLOOR)
    s = 2.0 * np.sqrt(dt_var)
    with np.errstate(over="ignore", invalid="ignore"):
        u = step_widths / s
        gaussian = np.exp(-(u * u))
        antideriv = 0.5 * step_widths + 0.5 * (step_widths * erf(u) + (s / _SQRT_PI) * gaussian)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(dx > 0.0, np.diff(antideriv, axis=0) / dx, 0.0)

    # Retardation correction. The per-edge antiderivative generates an effective flux
    # coefficient dD_t/dx = R*D_m/v + alpha_L (since d(tau)/dx = R/v under retardation), but
    # the Kreft-Zuber coefficient is D_L/v = D_m/v + alpha_L. Remove the excess
    # (R-1)*D_m/v times the bin-averaged Gaussian density dC_R/dx|_{D_t} = e^{-u^2}/(s*sqrt(pi)).
    # This term is identically zero when R == 1 or D_m == 0, so the exact per-edge result is
    # preserved there; elsewhere it restores agreement with the rigorous module.
    if retardation_factor != 1.0 and molecular_diffusivity > 0.0:
        density = gaussian / (s * _SQRT_PI)
        density_binavg = 0.5 * (density[:-1] + density[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            excess = np.where(velocity > 0.0, (retardation_factor - 1.0) * molecular_diffusivity / velocity, 0.0)
        frac -= excess[:, None] * density_binavg
    return frac


def _extend_tedges_flag(spinup: str | float | None) -> bool:
    """Translate the public ``spinup`` parameter to the internal extend flag.

    ``"constant"`` (default) extends ``tedges`` by 100 years on each side so a constant
    warm-start fills the left-edge spin-up region; ``None`` disables the extension (spin-up
    cout becomes NaN). Mirrors :func:`gwtransport.diffusion._diffusion_extend_tedges_flag`.

    Returns
    -------
    bool
        True if ``tedges`` should be extended (warm-start), False otherwise.

    Raises
    ------
    ValueError
        If ``spinup`` is a string other than ``"constant"``.
    NotImplementedError
        If ``spinup`` is a float (fraction-threshold mode is not implemented).
    """
    if spinup is None:
        return False
    if isinstance(spinup, str):
        if spinup != "constant":
            msg = f"spinup string must be 'constant'; got {spinup!r}"
            raise ValueError(msg)
        return True
    msg = f"diffusion_fast's spinup only supports None or 'constant'; float thresholds are not implemented (got {spinup!r})"
    raise NotImplementedError(msg)


def _closed_form_coeff_matrix(
    *,
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    flow_out: npt.NDArray[np.floating] | None,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    mean_streamline_length: float,
    mean_molecular_diffusivity: float,
    mean_longitudinal_dispersivity: float,
    retardation_factor: float,
    extend_tedges: bool,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Build the forward coefficient matrix W (``cout = W @ cin``) via the closed-form C_F.

    Mirrors :func:`gwtransport.diffusion._infiltration_to_extraction_coeff_matrix`
    (per-streamtube loop over pore volumes, 100-year warm-start extension, residence-time
    validity) but computes the bin-averaged flux concentration in closed form
    (:func:`_flux_breakthrough_fraction`) instead of 16-point Gauss-Legendre quadrature. The
    result reproduces the slow module's C_F to machine precision when the cout grid aligns
    with the flow grid.

    Returns
    -------
    coeff_matrix : ndarray, shape (n_cout_bins, n_cin_bins)
        Forward coefficient matrix, NaN replaced with zero.
    valid_cout_bins : ndarray of bool, shape (n_cout_bins,)
        Output bins with complete breakthrough information for every streamtube.
    """
    work_tedges = tedges
    if extend_tedges:
        work_tedges = pd.DatetimeIndex([
            tedges[0] - pd.Timedelta("36500D"),
            *list(tedges[1:-1]),
            tedges[-1] + pd.Timedelta("36500D"),
        ])

    tedges_days = tedges_to_days(work_tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=work_tedges[0])

    # Cumulative through-flow volume on a common axis. cout-edge volumes come from flow_out
    # when provided (the user-specified extraction-side flow), placed on the infiltration
    # volume axis by anchoring at the first cout edge inside the flow record (so an output
    # window that starts before the input data stays correctly aligned); otherwise
    # interpolated from the infiltration curve.
    cumulative_volume_at_cin = cumulative_flow_volume(flow, dt_to_days(work_tedges))
    if flow_out is not None:
        cumsum_out = np.concatenate(([0.0], np.cumsum(flow_out * dt_to_days(cout_tedges))))
        in_range = (cout_tedges_days >= tedges_days[0]) & (cout_tedges_days <= tedges_days[-1])
        i0 = int(np.argmax(in_range)) if np.any(in_range) else 0
        v_at_i0 = float(np.interp(cout_tedges_days[i0], tedges_days, cumulative_volume_at_cin))
        cumulative_volume_at_cout = v_at_i0 + (cumsum_out - cumsum_out[i0])
    else:
        cumulative_volume_at_cout = np.interp(cout_tedges_days, tedges_days, cumulative_volume_at_cin)

    # Residence time identifies cout bins with complete breakthrough (NaN beyond data).
    rt_at_cout_tedges = residence_time(
        flow=flow,
        flow_tedges=work_tedges,
        index=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        retardation_factor=retardation_factor,
        direction="extraction_to_infiltration",
    )
    valid_cout_bins = ~np.any(np.isnan(rt_at_cout_tedges[:, :-1]) | np.isnan(rt_at_cout_tedges[:, 1:]), axis=0)

    # Elapsed time tau[i, j] = t_cout_i - t_cin_j (the moving-frame age). Geometric xi and
    # tau are all the closed form needs -- no per-cell residence-time inversion.
    tau = np.maximum(cout_tedges_days[:, None] - tedges_days[None, :], 0.0)

    # Fluid (unretarded) velocity in each cout bin: q_cout * L / V_pore (V_pore folded in
    # per streamtube below). q_cout = dV_cout / dt_cout.
    dv_cout = np.diff(cumulative_volume_at_cout)
    dt_cout = np.diff(cout_tedges_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_cout = np.where(dt_cout > 0, dv_cout / dt_cout, 0.0)

    n_cout_bins = len(cout_tedges) - 1
    n_cin_bins = len(flow)
    accumulated_coeff = np.zeros((n_cout_bins, n_cin_bins))
    for v_pore in aquifer_pore_volumes:
        r_vpv = retardation_factor * v_pore
        step_widths = (
            (cumulative_volume_at_cout[:, None] - cumulative_volume_at_cin[None, :] - r_vpv)
            * mean_streamline_length
            / r_vpv
        )
        frac = _flux_breakthrough_fraction(
            step_widths=step_widths,
            tau=tau,
            molecular_diffusivity=mean_molecular_diffusivity,
            longitudinal_dispersivity=mean_longitudinal_dispersivity,
            streamline_len=mean_streamline_length,
            retardation_factor=retardation_factor,
            velocity=q_cout * mean_streamline_length / v_pore,
        )
        accumulated_coeff += frac[:, :-1] - frac[:, 1:]

    coeff_matrix = accumulated_coeff / len(aquifer_pore_volumes)
    return np.nan_to_num(coeff_matrix, nan=0.0), valid_cout_bins


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
    flow_out: npt.ArrayLike | None = None,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute extracted concentration with advection and longitudinal dispersion.

    Fast closed-form counterpart of :func:`gwtransport.diffusion.infiltration_to_extraction`.
    Reports the Kreft-Zuber (1978) flux concentration ``C_F`` and reproduces the slow module
    to machine precision when the cout grid aligns with the flow grid (supply ``flow_out``).

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in the infiltrating water. Length ``len(tedges) - 1``.
    flow : array-like
        Flow rate of water in the aquifer [m3/day]. Length ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Length ``len(cin) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Length ``len(output) + 1``.
    aquifer_pore_volumes : array-like
        Aquifer pore volumes [m3] -- one independent streamtube per entry.
    mean_streamline_length : float
        Mean travel distance L [m]. Must be positive.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity D_m [m2/day]. Must be non-negative.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity alpha_L [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid (aligned to ``cout_tedges``,
        length ``len(cout_tedges) - 1``). Defines the cout-edge volumes; supply it for
        machine-precision agreement with :mod:`gwtransport.diffusion` when ``cout_tedges``
        differs from ``tedges``. Default None (cout volumes interpolated from ``flow``).
    spinup : {"constant"} | None, optional
        ``"constant"`` (default) extends ``tedges`` by 100 years on each side so a constant
        warm-start fills the left-edge spin-up region; ``None`` leaves spin-up cout as NaN.

    Returns
    -------
    numpy.ndarray
        Bin-averaged Kreft-Zuber flux concentration ``C_F`` in the extracted water. Length
        ``len(cout_tedges) - 1``. NaN where no infiltration data has broken through.

    See Also
    --------
    gwtransport.diffusion.infiltration_to_extraction : Quadrature reference; supports
        per-streamtube streamline_length, molecular_diffusivity, longitudinal_dispersivity.
    extraction_to_infiltration : Inverse operation.
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion.
    """
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    tedges = pd.DatetimeIndex(tedges)
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    if flow_out is not None:
        flow_out = np.asarray(flow_out, dtype=float)

    _validate_inputs(
        cin_or_cout=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        is_forward=True,
        flow_out=flow_out,
    )

    coeff_matrix, valid_cout_bins = _closed_form_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow_out=flow_out,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        extend_tedges=_extend_tedges_flag(spinup),
    )

    cout = coeff_matrix @ cin

    # Mark output bins invalid where no input has broken through (spin-up) or the output
    # bin extends beyond the input data range.
    total_coeff = np.sum(coeff_matrix, axis=1)
    cout[(total_coeff < _EPSILON_COEFF_SUM) | ~valid_cout_bins] = np.nan
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
    regularization_strength: float = 1e-10,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """Reconstruct infiltration concentration from extracted water (deconvolution).

    Inverts the forward model by building the same closed-form flux-concentration matrix as
    :func:`infiltration_to_extraction` and solving ``W @ cin = cout`` via Tikhonov
    regularization. Fast closed-form counterpart of
    :func:`gwtransport.diffusion.extraction_to_infiltration`.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water. Length ``len(cout_tedges) - 1``.
    flow : array-like
        Flow rate of water in the aquifer [m3/day]. Length ``len(tedges) - 1``.
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Length ``len(flow) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Length ``len(cout) + 1``.
    aquifer_pore_volumes : array-like
        Aquifer pore volumes [m3] -- one independent streamtube per entry.
    mean_streamline_length : float
        Mean travel distance L [m]. Must be positive.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity D_m [m2/day]. Must be non-negative.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity alpha_L [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    regularization_strength : float, optional
        Tikhonov regularization parameter (default 1e-10).
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid (aligned to ``cout_tedges``).
        See :func:`infiltration_to_extraction`. Default None.
    spinup : {"constant"} | None, optional
        See :func:`infiltration_to_extraction`. Default ``"constant"``.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Length ``len(tedges) - 1``.
        NaN where no extraction data constrains the bin.

    Warns
    -----
    UserWarning
        When the forward matrix is rank-deficient (constant flow with residence time an
        integer multiple of the time step). Adjust ``aquifer_pore_volumes`` slightly
        (e.g. multiply by 1.001) to fix.

    See Also
    --------
    infiltration_to_extraction : Forward operation.
    gwtransport.diffusion.extraction_to_infiltration : Quadrature reference.
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion.
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    cout = np.asarray(cout, dtype=float)
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    if flow_out is not None:
        flow_out = np.asarray(flow_out, dtype=float)

    _validate_inputs(
        cin_or_cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        is_forward=False,
        flow_out=flow_out,
    )

    n_cin = len(tedges) - 1
    w_forward, valid_cout_bins = _closed_form_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        flow_out=flow_out,
        aquifer_pore_volumes=aquifer_pore_volumes,
        mean_streamline_length=mean_streamline_length,
        mean_molecular_diffusivity=mean_molecular_diffusivity,
        mean_longitudinal_dispersivity=mean_longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        extend_tedges=_extend_tedges_flag(spinup),
    )

    return solve_inverse_transport(
        w_forward=w_forward,
        observed=cout,
        n_output=n_cin,
        regularization_strength=regularization_strength,
        valid_rows=valid_cout_bins,
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
    flow_out: npt.ArrayLike | None = None,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """Compute extracted concentration for a gamma-distributed pore volume distribution.

    Convenience wrapper around :func:`infiltration_to_extraction` that discretizes a
    (shifted) gamma aquifer pore-volume distribution into ``n_bins`` equal-probability
    streamtubes. Provide either (mean, std) or (alpha, beta); ``loc`` defaults to 0.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin and flow data. Length ``len(cin) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins.
    mean, std : float, optional
        Mean and standard deviation of the gamma pore-volume distribution.
    loc : float, optional
        Location (minimum pore volume), ``0 <= loc < mean``. Default 0.0.
    alpha, beta : float, optional
        Shape and scale parameters of the gamma distribution (alternative to mean/std).
    n_bins : int, optional
        Number of equal-probability streamtubes. Default 100.
    mean_streamline_length : float
        Mean travel distance L [m]. Must be positive.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity D_m [m2/day]. Must be non-negative.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity alpha_L [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid. See
        :func:`infiltration_to_extraction`. Default None.
    spinup : {"constant"} | None, optional
        See :func:`infiltration_to_extraction`. Default ``"constant"``.

    Returns
    -------
    numpy.ndarray
        Bin-averaged Kreft-Zuber flux concentration ``C_F`` in the extracted water.

    See Also
    --------
    infiltration_to_extraction : Transport with an explicit pore volume distribution.
    gamma_extraction_to_infiltration : Reverse operation.
    gwtransport.gamma.bins : Create gamma distribution bins.
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model.
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
    regularization_strength: float = 1e-10,
    flow_out: npt.ArrayLike | None = None,
    spinup: str | float | None = "constant",
) -> npt.NDArray[np.floating]:
    """Reconstruct infiltration concentration for a gamma-distributed pore volume distribution.

    Convenience wrapper around :func:`extraction_to_infiltration` that discretizes a
    (shifted) gamma aquifer pore-volume distribution into ``n_bins`` equal-probability
    streamtubes. Provide either (mean, std) or (alpha, beta); ``loc`` defaults to 0.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water.
    flow : array-like
        Flow rate of water in the aquifer [m3/day].
    tedges : pandas.DatetimeIndex
        Time edges for cin (output) and flow data. Length ``len(flow) + 1``.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Length ``len(cout) + 1``.
    mean, std : float, optional
        Mean and standard deviation of the gamma pore-volume distribution.
    loc : float, optional
        Location (minimum pore volume), ``0 <= loc < mean``. Default 0.0.
    alpha, beta : float, optional
        Shape and scale parameters of the gamma distribution (alternative to mean/std).
    n_bins : int, optional
        Number of equal-probability streamtubes. Default 100.
    mean_streamline_length : float
        Mean travel distance L [m]. Must be positive.
    mean_molecular_diffusivity : float
        Mean effective molecular diffusivity D_m [m2/day]. Must be non-negative.
    mean_longitudinal_dispersivity : float
        Mean longitudinal dispersivity alpha_L [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0).
    regularization_strength : float, optional
        Tikhonov regularization parameter (default 1e-10).
    flow_out : array-like or None, optional
        Extraction flow rate [m3/day] on the output grid. See
        :func:`infiltration_to_extraction`. Default None.
    spinup : {"constant"} | None, optional
        See :func:`infiltration_to_extraction`. Default ``"constant"``.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Length ``len(tedges) - 1``.

    See Also
    --------
    extraction_to_infiltration : Deconvolution with an explicit pore volume distribution.
    gamma_infiltration_to_extraction : Forward operation.
    gwtransport.gamma.bins : Create gamma distribution bins.
    :ref:`concept-gamma-distribution` : Two-parameter pore volume model.
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
        regularization_strength=regularization_strength,
        flow_out=flow_out,
        spinup=spinup,
    )


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
    retardation_factor: float,
    is_forward: bool,
    flow_out: np.ndarray | None = None,
) -> None:
    """Validate inputs for infiltration_to_extraction and extraction_to_infiltration.

    Raises
    ------
    ValueError
        If array lengths are inconsistent, mean_molecular_diffusivity or
        mean_longitudinal_dispersivity are negative, cin or cout or flow contain NaN values,
        aquifer_pore_volumes contains non-positive values, mean_streamline_length is
        non-positive, or retardation_factor is below 1 (anti-retardation is not physical for
        the supported sorption isotherms).
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
    if retardation_factor < 1.0:
        msg = "retardation_factor must be >= 1.0"
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
