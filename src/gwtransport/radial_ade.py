r"""Exact radial advection-dispersion transport for a single well (push-pull / ASR).

Computes the extracted flux concentration ``cout`` at a single fully-penetrating well driven by an
arbitrary signed flow schedule (positive = injection, negative = extraction, zero = rest) and an
arbitrary injected concentration ``cin``. The physics is the exact radial advection-dispersion of the
radial-ADE knowledge base: volume coordinate ``V(r) = pi b n (r^2 - r_w^2)``, Scheidegger
velocity-dependent dispersion ``D = alpha_L |u| + D_m``, Kreft-Zuber flux boundary conditions, and
the exact Airy / Whittaker per-phase kernels. Nothing is reduced to a Gaussian; the exact
non-Gaussian breakthrough (with the correct skewness) is carried.

Two engines back the forward map. A single inject-then-extract cycle with ``D_m = 0`` uses the
closed-form echo operator (:mod:`gwtransport._radial_compose`, KB Sec. 10a) -- exact for arbitrary
within-phase variable flow, with the exact temporal moments. Any other signed-flow schedule (more
reversals / multi-cycle ASR) and any ``D_m > 0`` case use an implicit finite-volume solve of the
*same* exact conservative V-form PDE (:mod:`gwtransport._radial_fv`, KB Sec. 9) -- same equation,
same boundary conditions, with a controlled ~1% discretization error (a spectral-domain accelerator
for the multi-cycle case is a possible future optimization, not a different physics). The engine is
chosen automatically; cycles are expressed through the flow sign pattern, not an argument.

The reported ``cout`` is the flow-weighted average over each output bin -- defined on extraction bins
(``flow < 0``) and ``NaN`` on injection / rest bins (nothing is recovered there).

Macrodispersion is an **ensemble over disk heights** (KB addendum Sec. A6): ``pore_height`` may be an
array of disk thicknesses, each an independent radial cell carrying the full flow, and the output is
the (weight-weighted) average of the per-disk breakthroughs. A spread of heights spreads the
arrival times -- the height-parameterised analogue of the package's pore-volume APVD.

Available functions:

- :func:`infiltration_to_extraction` -- forward transport (cin -> cout).
- :func:`extraction_to_infiltration` -- inverse via Tikhonov regularization (cout -> cin).
- :func:`gamma_infiltration_to_extraction` -- gamma-distributed disk height (forward).
- :func:`gamma_extraction_to_infiltration` -- same, inverse.

References
----------
Kreft, A., & Zuber, A. (1978). On the physical meaning of the dispersion equation and its solutions
for different initial and boundary conditions. Chemical Engineering Science, 33(11), 1471-1480.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport import gamma
from gwtransport._radial_compose import single_cycle_echo_matrix
from gwtransport._radial_fv import fv_cout_deviation
from gwtransport._time import dt_to_days

# Finite-volume discretization for the general (multi-cycle / D_m>0) engine.
_FV_N_CELLS = 400
_FV_N_SUB = 8


def _is_single_cycle(flow: npt.NDArray[np.floating]) -> bool:
    """Return True if the schedule is a single injection block followed by a single extraction block.

    Such schedules (one flow reversal, injection first, ``D_m = 0``) use the exact closed-form echo
    operator; any other signed-flow pattern (more reversals, extraction first) and any ``D_m > 0``
    case fall to the finite-volume engine.

    Returns
    -------
    bool
        Whether ``flow`` is a single inject-then-extract cycle.
    """
    signs = np.sign(flow[flow != 0.0])
    n_changes = int(np.sum(np.diff(signs) != 0)) if signs.size else 0
    return n_changes <= 1 and (signs.size == 0 or signs[0] > 0)


def _validate(
    *,
    cin_or_cout: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    pore_height: npt.NDArray[np.floating],
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    weights: npt.NDArray[np.floating] | None,
) -> None:
    """Validate inputs for the radial single-well transport functions (signed flow is allowed).

    Raises
    ------
    ValueError
        On inconsistent lengths, non-positive geometry, out-of-range porosity, negative dispersion,
        ``retardation_factor < 1``, mismatched ``weights``, NaN in ``flow``, or a ``cout_tedges`` that
        differs from ``tedges`` (a distinct output grid is not yet supported).
    """
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if len(cin_or_cout) != len(flow):
        msg = "cin/cout must have the same length as flow"
        raise ValueError(msg)
    if not tedges.equals(cout_tedges):
        msg = "cout_tedges must equal tedges (a distinct output grid is not yet supported)"
        raise ValueError(msg)
    if np.any(np.isnan(flow)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(pore_height <= 0.0):
        msg = "pore_height must be positive"
        raise ValueError(msg)
    if not 0.0 < porosity <= 1.0:
        msg = "porosity must be in (0, 1]"
        raise ValueError(msg)
    if well_radius <= 0.0:
        msg = "well_radius must be positive"
        raise ValueError(msg)
    if longitudinal_dispersivity <= 0.0:
        msg = "longitudinal_dispersivity must be positive (the dispersion kernel requires alpha_L > 0)"
        raise ValueError(msg)
    if molecular_diffusivity < 0.0:
        msg = "molecular_diffusivity must be non-negative"
        raise ValueError(msg)
    if retardation_factor < 1.0:
        msg = "retardation_factor must be >= 1.0"
        raise ValueError(msg)
    if weights is not None and len(weights) != len(pore_height):
        msg = "weights must have the same length as pore_height"
        raise ValueError(msg)


def _forward_operator(
    *,
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    pore_height: npt.NDArray[np.floating],
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    weights: npt.NDArray[np.floating] | None,
    n_quad: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Build the ensemble-averaged echo operator ``W`` (``cout' = W @ cin'_inj``) and the phase masks.

    Returns
    -------
    w_ens : ndarray, shape (n_ext, n_inj)
        Weight-weighted average over the disk-height ensemble of the per-disk echo matrices.
    inj_mask, ext_mask : ndarray of bool
        Injection (``flow > 0``) and extraction (``flow < 0``) bin masks.
    """
    inj_mask, ext_mask = flow > 0.0, flow < 0.0
    dt = dt_to_days(tedges)
    inj_vol = np.concatenate(([0.0], np.cumsum((flow * dt)[inj_mask])))  # 0 .. S_inj
    ext_vol = np.concatenate(([0.0], np.cumsum((-flow * dt)[ext_mask])))  # 0 .. T_end
    inj_flow_scale = float(np.mean(flow[inj_mask])) if np.any(inj_mask) else 1.0
    ext_flow_scale = float(np.mean(-flow[ext_mask])) if np.any(ext_mask) else 1.0

    heights = np.atleast_1d(np.asarray(pore_height, dtype=float))
    w = np.ones(len(heights)) if weights is None else np.asarray(weights, dtype=float)
    w_ens = np.zeros((int(np.sum(ext_mask)), int(np.sum(inj_mask))))
    for b_i, w_i in zip(heights, w, strict=True):
        w_ens += w_i * single_cycle_echo_matrix(
            inj_volume_edges=inj_vol,
            ext_volume_edges=ext_vol,
            c_geo=np.pi * b_i * porosity,
            r_w=well_radius,
            alpha_l=longitudinal_dispersivity,
            inj_flow_scale=inj_flow_scale,
            ext_flow_scale=ext_flow_scale,
            retardation_factor=retardation_factor,
            molecular_diffusivity=molecular_diffusivity,
            n_quad=n_quad,
        )
    return w_ens / np.sum(w), inj_mask, ext_mask


def _fv_forward(
    *,
    cin_deviation: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    pore_height: npt.NDArray[np.floating],
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    weights: npt.NDArray[np.floating] | None,
) -> npt.NDArray[np.floating]:
    """Ensemble finite-volume extracted-flux deviation per bin (general signed flow / D_m > 0).

    Runs the implicit FV solver once per disk and returns the weight-weighted average; NaN on
    injection / rest bins.

    Returns
    -------
    ndarray, shape (n,)
        Weighted-average extracted-flux deviation; NaN on non-extraction bins.
    """
    heights = np.atleast_1d(np.asarray(pore_height, dtype=float))
    w = np.ones(len(heights)) if weights is None else np.asarray(weights, dtype=float)
    acc = np.zeros(len(flow))
    for b_i, w_i in zip(heights, w, strict=True):
        c_i = fv_cout_deviation(
            cin_deviation=cin_deviation,
            flow=flow,
            dt_days=dt_days,
            c_geo=np.pi * b_i * porosity,
            r_w=well_radius,
            alpha_l=longitudinal_dispersivity,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
            n_cells=_FV_N_CELLS,
            n_sub=_FV_N_SUB,
        )
        acc += w_i * np.nan_to_num(c_i)
    out = np.full(len(flow), np.nan)
    ext_mask = flow < 0.0
    out[ext_mask] = acc[ext_mask] / np.sum(w)
    return out


def _fv_operator(
    *,
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    pore_height: npt.NDArray[np.floating],
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    weights: npt.NDArray[np.floating] | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Build the dense forward operator ``W`` (``cout' = W @ cin'_inj``) by FV unit-pulse columns.

    One ensemble FV solve per injection bin (a unit deviation pulse there) gives that column of the
    linear forward map -- used by the reverse solve for the general signed-flow / ``D_m > 0`` engine.

    Returns
    -------
    w : ndarray, shape (n_ext, n_inj)
        Dense forward operator.
    inj_mask, ext_mask : ndarray of bool
        Injection / extraction bin masks.
    """
    inj_mask, ext_mask = flow > 0.0, flow < 0.0
    inj_idx = np.flatnonzero(inj_mask)
    w = np.zeros((int(np.sum(ext_mask)), len(inj_idx)))
    for j, idx in enumerate(inj_idx):
        pulse = np.zeros(len(flow))
        pulse[idx] = 1.0
        cout_dev = _fv_forward(
            cin_deviation=pulse,
            flow=flow,
            dt_days=dt_days,
            pore_height=pore_height,
            porosity=porosity,
            well_radius=well_radius,
            longitudinal_dispersivity=longitudinal_dispersivity,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
            weights=weights,
        )
        w[:, j] = cout_dev[ext_mask]
    return w, inj_mask, ext_mask


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    pore_height: npt.ArrayLike,
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    weights: npt.ArrayLike | None = None,
    background: float = 0.0,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    """Compute the extracted flux concentration at a radial well for a signed flow schedule.

    Parameters
    ----------
    cin : array-like, shape (n,)
        Injected concentration per time bin (used only on injection bins, ``flow > 0``).
    flow : array-like, shape (n,)
        Signed flow per time bin [m^3/day]: ``> 0`` injection, ``< 0`` extraction, ``0`` rest.
    tedges : DatetimeIndex
        Time bin edges (``n + 1`` for ``n`` bins).
    cout_tedges : DatetimeIndex
        Output time bin edges; must equal ``tedges``. Output is NaN on injection / rest bins.
    pore_height : array-like
        Disk thickness ``b`` [m]; a scalar, or an array of heights for the ensemble-over-heights
        macrodispersion.
    porosity : float
        Porosity ``n`` [-].
    well_radius : float
        Well (screen) radius ``r_w`` [m].
    longitudinal_dispersivity : float
        Longitudinal dispersivity ``alpha_L`` [m].
    molecular_diffusivity : float, optional
        Molecular diffusivity ``D_m`` [m^2/day]. Default 0. ``D_m > 0`` (and any multi-reversal
        schedule) routes to the finite-volume engine (~1% discretization error).
    retardation_factor : float, optional
        Linear retardation ``R >= 1``. Default 1.
    weights : array-like, optional
        Per-disk averaging weights (same length as ``pore_height``). Default equal weights.
    background : float, optional
        Ambient aquifer concentration ``c_bg``. The deviation ``cin - c_bg`` is transported and
        ``c_bg`` is added back; constant ``cin = c_bg`` returns ``cout = c_bg``. Default 0.
    n_quad : int, optional
        Gauss-Legendre node count for the resident-profile superposition. Default 240.

    Returns
    -------
    ndarray, shape (n,)
        Extracted flux concentration; NaN on injection and rest bins.
    """
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    pore_height = np.atleast_1d(np.asarray(pore_height, dtype=float))
    weights_arr = None if weights is None else np.atleast_1d(np.asarray(weights, dtype=float))
    _validate(
        cin_or_cout=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        pore_height=pore_height,
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=weights_arr,
    )
    if _is_single_cycle(flow) and molecular_diffusivity == 0.0:
        w_ens, inj_mask, ext_mask = _forward_operator(
            flow=flow,
            tedges=tedges,
            pore_height=pore_height,
            porosity=porosity,
            well_radius=well_radius,
            longitudinal_dispersivity=longitudinal_dispersivity,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
            weights=weights_arr,
            n_quad=n_quad,
        )
        cout = np.full(len(flow), np.nan)
        cout[ext_mask] = background + w_ens @ (cin[inj_mask] - background)
        return cout
    # General signed-flow / D_m > 0 engine: implicit finite volume (one ensemble solve).
    cout_dev = _fv_forward(
        cin_deviation=cin - background,
        flow=flow,
        dt_days=dt_to_days(tedges),
        pore_height=pore_height,
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=weights_arr,
    )
    return background + cout_dev


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    pore_height: npt.ArrayLike,
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    weights: npt.ArrayLike | None = None,
    background: float = 0.0,
    regularization_strength: float = 1e-10,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    """Recover the injected concentration from extracted-water measurements (Tikhonov inverse).

    Inverts the forward echo operator built by :func:`infiltration_to_extraction`. Returns the
    injected concentration on injection bins (NaN on extraction / rest bins).

    Parameters
    ----------
    cout : array-like, shape (n,)
        Measured extracted concentration (used on extraction bins, ``flow < 0``).
    flow, tedges, cout_tedges, pore_height, porosity, well_radius, longitudinal_dispersivity,
    molecular_diffusivity, retardation_factor, weights, background, n_quad
        As in :func:`infiltration_to_extraction`.
    regularization_strength : float, optional
        Tikhonov parameter. Default ``1e-10``.

    Returns
    -------
    ndarray, shape (n,)
        Recovered injected concentration; NaN on extraction / rest bins.
    """
    cout = np.asarray(cout, dtype=float)
    flow = np.asarray(flow, dtype=float)
    pore_height = np.atleast_1d(np.asarray(pore_height, dtype=float))
    weights_arr = None if weights is None else np.atleast_1d(np.asarray(weights, dtype=float))
    _validate(
        cin_or_cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        pore_height=pore_height,
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=weights_arr,
    )
    if _is_single_cycle(flow) and molecular_diffusivity == 0.0:
        w_ens, inj_mask, ext_mask = _forward_operator(
            flow=flow,
            tedges=tedges,
            pore_height=pore_height,
            porosity=porosity,
            well_radius=well_radius,
            longitudinal_dispersivity=longitudinal_dispersivity,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
            weights=weights_arr,
            n_quad=n_quad,
        )
    else:
        w_ens, inj_mask, ext_mask = _fv_operator(
            flow=flow,
            dt_days=dt_to_days(tedges),
            pore_height=pore_height,
            porosity=porosity,
            well_radius=well_radius,
            longitudinal_dispersivity=longitudinal_dispersivity,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
            weights=weights_arr,
        )
    # Tikhonov least-squares min ||W x - (cout-bg)||^2 + lambda ||x||^2 via the stable augmented
    # system [W; sqrt(lambda) I] x = [cout-bg; 0]. The package's solve_inverse_transport assumes the
    # advection rows-sum-to-1 convention; the echo operator instead has column sums ~1 (mass
    # conservation per injection bin) and overdetermined rows, so a direct Tikhonov fit is used.
    n_inj = w_ens.shape[1]
    augmented = np.vstack([w_ens, np.sqrt(regularization_strength) * np.eye(n_inj)])
    rhs = np.concatenate([cout[ext_mask] - background, np.zeros(n_inj)])
    cin_dev = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    cin = np.full(len(flow), np.nan)
    cin[inj_mask] = background + cin_dev
    return cin


def gamma_infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    mean: float | None = None,
    std: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    background: float = 0.0,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    """Single-well radial transport with a gamma-distributed disk height (ensemble macrodispersion).

    Discretizes the gamma distribution of disk height ``b`` into ``n_bins`` equal-probability bins via
    :func:`gwtransport.gamma.bins` and runs :func:`infiltration_to_extraction` over the bin heights,
    weighting each by its probability mass.

    Parameters
    ----------
    mean, std, alpha, beta : float, optional
        Gamma parameters of the disk height ``b`` (either ``mean``/``std`` or ``alpha``/``beta``).
    n_bins : int, optional
        Number of equal-probability height bins. Default 100.
    cin, flow, tedges, cout_tedges, porosity, well_radius, longitudinal_dispersivity,
    molecular_diffusivity, retardation_factor, background, n_quad
        As in :func:`infiltration_to_extraction`.

    Returns
    -------
    ndarray, shape (n,)
        Extracted flux concentration; NaN on injection / rest bins.
    """
    height_bins = gamma.bins(mean=mean, std=std, alpha=alpha, beta=beta, n_bins=n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        pore_height=height_bins["expected_values"],
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=height_bins["probability_mass"],
        background=background,
        n_quad=n_quad,
    )


def gamma_extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    mean: float | None = None,
    std: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    n_bins: int = 100,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    background: float = 0.0,
    regularization_strength: float = 1e-10,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    """Inverse of :func:`gamma_infiltration_to_extraction` (gamma-distributed disk height).

    Returns
    -------
    ndarray, shape (n,)
        Recovered injected concentration; NaN on extraction / rest bins.
    """
    height_bins = gamma.bins(mean=mean, std=std, alpha=alpha, beta=beta, n_bins=n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        pore_height=height_bins["expected_values"],
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=height_bins["probability_mass"],
        background=background,
        regularization_strength=regularization_strength,
        n_quad=n_quad,
    )
