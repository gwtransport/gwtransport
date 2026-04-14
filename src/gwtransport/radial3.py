"""
Radial Push-Pull Well Transport Model -- analytical Bessel kernel.

Evaluates an analytical kernel derived from the 2-D radial heat
equation Green's function at the well screen, transformed to the
``V = scale * r**2`` coordinate, on a Gauss-Legendre partition of the
union of all bin edges. The heavy lifting lives in
:func:`gwtransport.radial3_utils._radial_bessel_matrix`, whose module
docstring explains the physics and quadrature scheme in detail.

Characteristics:

1. **No FV grid, no CFL sub-stepping.** Cost scales with
   ``n_quad * n_sub_intervals * n_layers`` and is independent of any
   discretization parameter other than the GL order.
2. **Row sums come out analytically.** A single rate ``lambda`` is
   assigned per extraction-time node so the per-injection-bin
   exponential contributions telescope, giving analytic row sums
   without any post-hoc clipping or normalization.
3. **Physical well-face boundary.** The kernel is an exponential in
   ``V_src`` supported on ``[0, V_tau]``, so no mass leaks to the
   unphysical region ``V > V_tau`` above the LIFO-matched parcel -- a
   key correction over a symmetric Gaussian-in-V kernel.
4. **Exact LIFO fallback.** With ``molecular_diffusivity == 0`` and
   ``longitudinal_dispersivity == 0`` the path delegates to the LIFO
   matrix used elsewhere in the package.

Public API: ``push_pull``, ``push_pull_inverse``, ``gamma_push_pull``,
``gamma_push_pull_inverse``.

This file is part of gwtransport which is released under AGPL-3.0
license. See the ./LICENSE file or
https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full
license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport import gamma
from gwtransport.radial3_utils import _push_pull_advection_matrix, _radial_bessel_matrix
from gwtransport.utils import solve_inverse_transport

# Default molecular diffusivity for solute transport in water at room
# temperature: ~1e-9 m**2/s = 8.64e-5 m**2/day. Typical small-ion
# diffusivities (Cl-, Na+, NO3-) range from 1.0e-9 to 2.0e-9 m**2/s, so
# this default is a sensible starting point. Users with non-aqueous
# solutes or temperature-dependent rates should pass an explicit value.
_DEFAULT_DM = 8.64e-5


def push_pull(
    *,
    flow: npt.ArrayLike,
    cin: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    layer_heights: npt.ArrayLike,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = _DEFAULT_DM,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
    n_quad: int = 6,
) -> npt.NDArray[np.floating]:
    """Compute extraction concentration for a radial push-pull well.

    Evaluates the 2-D radial Bessel kernel at the well screen via
    Gauss-Legendre quadrature on the partition obtained from the
    union of all bin edges; see
    :func:`gwtransport.radial3_utils._radial_bessel_matrix` for the
    derivation.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m**3/day]. Positive = injection,
        negative = extraction, zero = rest.
    cin : array-like, shape (n,)
        Injection concentration per time bin. Values during
        extraction/rest bins are ignored.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    cout_tedges : DatetimeIndex
        Output time bin edges. Returns NaN for bins without extraction.
    layer_heights : array-like, shape (N,)
        Height of each horizontal streamtube [m].
    porosity : float
        Porosity ``n`` [-].
    retardation_factor : float, optional
        Retardation factor ``R`` [-]. Default 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity ``D_m`` [m**2/day]. Default ``8.64e-5``
        (~ 1e-9 m**2/s, typical small-ion diffusivity in water at room
        temperature).
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity ``alpha_L`` [m]. Default 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water. Default 0.
    n_quad : int, optional
        Gauss-Legendre order per sub-interval. Default 6 gives spectral
        convergence for the smooth Gaussian integrand once the partition
        has resolved every flow-bin discontinuity.

    Returns
    -------
    ndarray, shape (n_cout,)
        Flow-weighted extraction concentration. NaN for bins without
        extraction.

    Raises
    ------
    ValueError
        If ``flow`` and ``cin`` have different lengths, or if ``tedges``
        does not have ``len(flow) + 1`` entries.

    See Also
    --------
    push_pull_inverse : Inverse solver sharing the same forward operator.
    gamma_push_pull : Convenience wrapper with gamma-distributed layer heights.
    """
    flow = np.asarray(flow, dtype=float)
    cin = np.asarray(cin, dtype=float)
    layer_heights = np.asarray(layer_heights, dtype=float)
    n = len(flow)

    if len(cin) != n:
        msg = f"flow and cin must have the same length, got {n} and {len(cin)}"
        raise ValueError(msg)
    if len(tedges) != n + 1:
        msg = f"tedges must have length len(flow) + 1, got {len(tedges)} and {n}"
        raise ValueError(msg)

    flow, cin, tedges_days, cout_tedges_days = _prepend_background(
        flow=flow, cin=cin, tedges=tedges, cout_tedges=cout_tedges, c_background=c_background
    )
    n = len(flow)

    has_diffusion = molecular_diffusivity > 0 or longitudinal_dispersivity > 0

    if not has_diffusion:
        dt = np.diff(tedges_days)
        w, has_extraction = _push_pull_advection_matrix(
            flow=flow, dt=dt, tedges_days=tedges_days, cout_tedges_days=cout_tedges_days
        )
    else:
        w, has_extraction = _radial_bessel_matrix(
            flow=flow,
            tedges_days=tedges_days,
            cout_tedges_days=cout_tedges_days,
            layer_heights=layer_heights,
            porosity=porosity,
            retardation_factor=retardation_factor,
            molecular_diffusivity=molecular_diffusivity,
            longitudinal_dispersivity=longitudinal_dispersivity,
            n_quad=n_quad,
        )

    is_injection = flow > 0
    cin_clean = np.where(is_injection, cin, 0.0)

    cout = w @ cin_clean
    cout[~has_extraction] = np.nan
    return cout


def push_pull_inverse(
    *,
    flow: npt.ArrayLike,
    cout: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    layer_heights: npt.ArrayLike,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = _DEFAULT_DM,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
    regularization_strength: float = 1e-3,
    n_quad: int = 6,
) -> npt.NDArray[np.floating]:
    """Estimate injection concentration from extraction measurements.

    Inverse counterpart of :func:`push_pull`. Builds the same forward
    operator and inverts via :func:`gwtransport.utils.solve_inverse_transport`.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m**3/day].
    cout : array-like, shape (n_cout,)
        Measured extraction concentration on the ``cout_tedges`` grid.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    cout_tedges : DatetimeIndex
        Output time bin edges.
    layer_heights : array-like, shape (N,)
        Height of each horizontal streamtube [m].
    porosity : float
        Porosity ``n`` [-].
    retardation_factor : float, optional
        Retardation factor ``R`` [-]. Default 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity ``D_m`` [m**2/day]. Default ``8.64e-5``.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity ``alpha_L`` [m]. Default 0.
    c_background : float, optional
        Background concentration. Default 0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default 1e-3.
    n_quad : int, optional
        Gauss-Legendre order per sub-interval. Default 6.

    Returns
    -------
    ndarray, shape (n,)
        Estimated injection concentration. NaN for extraction and rest
        bins, and for injection bins the solver could not constrain.

    See Also
    --------
    push_pull : Forward model.
    """
    flow = np.asarray(flow, dtype=float)
    cout = np.asarray(cout, dtype=float)
    layer_heights = np.asarray(layer_heights, dtype=float)
    n = len(flow)

    flow, _, tedges_days, cout_tedges_days = _prepend_background(
        flow=flow,
        cin=np.zeros(n),
        tedges=tedges,
        cout_tedges=cout_tedges,
        c_background=c_background,
    )
    n = len(flow)

    has_diffusion = molecular_diffusivity > 0 or longitudinal_dispersivity > 0

    if not has_diffusion:
        dt = np.diff(tedges_days)
        w, has_extraction = _push_pull_advection_matrix(
            flow=flow, dt=dt, tedges_days=tedges_days, cout_tedges_days=cout_tedges_days
        )
    else:
        w, has_extraction = _radial_bessel_matrix(
            flow=flow,
            tedges_days=tedges_days,
            cout_tedges_days=cout_tedges_days,
            layer_heights=layer_heights,
            porosity=porosity,
            retardation_factor=retardation_factor,
            molecular_diffusivity=molecular_diffusivity,
            longitudinal_dispersivity=longitudinal_dispersivity,
            n_quad=n_quad,
        )

    is_injection = flow > 0
    valid_cout = has_extraction & np.isfinite(cout)
    bg_contribution = w[:, 0] * c_background
    observed = np.where(valid_cout, cout - bg_contribution, 0.0)

    cin_recovered = solve_inverse_transport(
        w_forward=w[:, 1:],
        observed=observed,
        n_output=n - 1,
        regularization_strength=regularization_strength,
        valid_rows=valid_cout,
    )

    out = np.full(n - 1, np.nan)
    idx = np.flatnonzero(is_injection[1:])
    out[idx] = cin_recovered[idx]
    return out


def gamma_push_pull(
    *,
    flow: npt.ArrayLike,
    cin: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = _DEFAULT_DM,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
    n_quad: int = 6,
) -> npt.NDArray[np.floating]:
    """Compute extraction concentration with gamma-distributed layer heights.

    Convenience wrapper around :func:`push_pull` that discretizes a
    gamma distribution into equal-probability bins for layer heights.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m**3/day].
    cin : array-like, shape (n,)
        Injection concentration per time bin.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    cout_tedges : DatetimeIndex
        Output time bin edges.
    alpha, beta : float, optional
        Shape and scale parameters of the gamma distribution.
    mean, std : float, optional
        Mean and standard deviation of the gamma distribution.
    n_bins : int, optional
        Number of bins for gamma discretization. Default 100.
    porosity : float
        Porosity ``n`` [-].
    retardation_factor : float, optional
        Retardation factor ``R`` [-]. Default 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity ``D_m`` [m**2/day]. Default ``8.64e-5``.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity ``alpha_L`` [m]. Default 0.
    c_background : float, optional
        Background concentration. Default 0.
    n_quad : int, optional
        Gauss-Legendre order per sub-interval. Default 6.

    Returns
    -------
    ndarray, shape (n_cout,)
        Flow-weighted extraction concentration.

    See Also
    --------
    push_pull : Base function with explicit layer heights.
    gwtransport.gamma.bins : Gamma distribution discretization.
    """
    gamma_bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    layer_heights = gamma_bins["expected_values"]

    return push_pull(
        flow=flow,
        cin=cin,
        tedges=tedges,
        cout_tedges=cout_tedges,
        layer_heights=layer_heights,
        porosity=porosity,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        c_background=c_background,
        n_quad=n_quad,
    )


def gamma_push_pull_inverse(
    *,
    flow: npt.ArrayLike,
    cout: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = _DEFAULT_DM,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
    regularization_strength: float = 1e-3,
    n_quad: int = 6,
) -> npt.NDArray[np.floating]:
    """Estimate injection concentration with gamma-distributed layer heights.

    Convenience wrapper around :func:`push_pull_inverse`.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m**3/day].
    cout : array-like, shape (n_cout,)
        Measured extraction concentration.
    tedges : DatetimeIndex
        Time bin edges.
    cout_tedges : DatetimeIndex
        Output time bin edges.
    alpha, beta : float, optional
        Shape and scale parameters of the gamma distribution.
    mean, std : float, optional
        Mean and standard deviation of the gamma distribution.
    n_bins : int, optional
        Number of bins for gamma discretization. Default 100.
    porosity : float
        Porosity ``n`` [-].
    retardation_factor : float, optional
        Retardation factor ``R`` [-]. Default 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity ``D_m`` [m**2/day]. Default ``8.64e-5``.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity ``alpha_L`` [m]. Default 0.
    c_background : float, optional
        Background concentration. Default 0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default 1e-3.
    n_quad : int, optional
        Gauss-Legendre order per sub-interval. Default 6.

    Returns
    -------
    ndarray, shape (n,)
        Estimated injection concentration.

    See Also
    --------
    push_pull_inverse : Base function with explicit layer heights.
    gwtransport.gamma.bins : Gamma distribution discretization.
    """
    gamma_bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    layer_heights = gamma_bins["expected_values"]

    return push_pull_inverse(
        flow=flow,
        cout=cout,
        tedges=tedges,
        cout_tedges=cout_tedges,
        layer_heights=layer_heights,
        porosity=porosity,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        c_background=c_background,
        regularization_strength=regularization_strength,
        n_quad=n_quad,
    )


def _prepend_background(
    *,
    flow: npt.NDArray[np.floating],
    cin: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    c_background: float,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Prepend a large background-injection bin and convert tedges to days.

    The background bin sits at index 0 with a volume ``1000x`` the
    peak ``|v_cum|`` so that over-extraction and the outer plume edge
    both draw from ``c_background``.

    Returns
    -------
    flow_full : ndarray, shape (n+1,)
        Signed flow rate with the background bin prepended at index 0.
    cin_full : ndarray, shape (n+1,)
        Injection concentration with ``c_background`` at index 0.
    tedges_days : ndarray, shape (n+2,)
        Time edges in days from the start of the background bin.
    cout_tedges_days : ndarray, shape (n_cout+1,)
        Output time edges in days from the start of the background bin.
    """
    dt = np.diff(tedges) / pd.Timedelta("1D")
    v_cum_test = np.cumsum(flow * dt)
    max_abs_v = max(float(np.max(np.abs(np.concatenate(([0.0], v_cum_test))))), 1.0)
    prepend_volume = 1000.0 * max_abs_v
    flow_full = np.concatenate(([prepend_volume], flow))
    cin_full = np.concatenate(([c_background], cin))
    tedges_full = tedges.insert(0, tedges[0] - pd.Timedelta("1D"))

    tedges_days = ((tedges_full - tedges_full[0]) / pd.Timedelta("1D")).values.astype(float)
    cout_tedges_days = ((cout_tedges - tedges_full[0]) / pd.Timedelta("1D")).values.astype(float)
    return flow_full, cin_full, tedges_days, cout_tedges_days
