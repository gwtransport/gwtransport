"""
Radial Push-Pull Well Transport Model.

This module provides functions to model solute transport in a single well with radial
flow in a vertically heterogeneous aquifer. The model handles bidirectional flow
(injection and extraction) through a push-pull transport mechanism, where the most
recently injected water is the first to be extracted.

The key difference from the linear infiltration-to-extraction models in
:mod:`gwtransport.advection` and :mod:`gwtransport.diffusion` is:

- **Bidirectional flow**: Water is injected and then extracted from the same well.
  Transport follows push-pull ordering — the last parcel injected is the first extracted.
- **Radial geometry**: The radial distance to the advective front scales as
  r = √(V / (N · π · h · n · R)), where V is cumulative volume, N is the
  number of layers, h is the layer height, n is porosity, and R is retardation.

**Heterogeneity model**: The aquifer is divided into N horizontal streamtubes
flowing to the well screen, each carrying equal flow but having a different
height h_i. Thinner streamtubes push the tracer front further radially for
the same injected volume, leading to more diffusive spreading and irreversible
mixing. Without diffusion, streamtube heterogeneity is invisible — all
streamtubes produce identical breakthrough (perfect push-pull recovery).

Available functions:

- :func:`push_pull` - Forward model: compute extraction concentration for a
  radial push-pull well given injection concentration and flow history.

- :func:`push_pull_inverse` - Inverse model: estimate injection concentration
  from measured extraction concentration using Tikhonov regularization.

- :func:`gamma_push_pull` - Convenience wrapper using gamma-distributed layer
  heights. Parameterizable via (alpha, beta) or (mean, std).

- :func:`gamma_push_pull_inverse` - Convenience wrapper for the inverse model
  with gamma-distributed layer heights.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport import gamma
from gwtransport.radial_utils import _push_pull_advection_matrix, _push_pull_diffusion_matrix
from gwtransport.utils import solve_inverse_transport


def push_pull(
    *,
    flow: npt.ArrayLike,
    cin: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    layer_heights: npt.ArrayLike,
    porosity: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute extraction concentration for a radial push-pull well.

    Models solute transport in a single well with radial flow in a vertically
    heterogeneous aquifer. Injection (flow > 0) pushes tracer outward; extraction
    (flow < 0) pulls it back in push-pull (last-in-first-out) order.

    Without diffusion, all layers produce identical breakthrough (layer
    heterogeneity is invisible). With diffusion, layers with smaller height
    push the front further radially, causing more irreversible spreading.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cin : array-like, shape (n,)
        Injection concentration per time bin [concentration units].
        Values during extraction/rest bins are ignored.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    cout_tedges : DatetimeIndex
        Output time bin edges (n_cout+1 edges for n_cout bins). The
        extraction concentration is flow-weighted averaged onto this grid.
        NaN for bins without extraction.
    layer_heights : array-like, shape (N,)
        Height of each horizontal streamtube flowing to the well screen [m].
        The well screen is divided into N streamtubes of equal flow but
        different heights. Thinner streamtubes push the advective front
        further radially for the same injected volume, causing more
        diffusive spreading along the flow path.
    porosity : float
        Porosity n [-].
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water [concentration
        units]. When extraction pulls water from beyond the injection zone,
        the unaccounted fraction is assigned this concentration. Default is 0.

    Returns
    -------
    ndarray, shape (n_cout,)
        Flow-weighted extraction concentration. NaN for bins without extraction.
        Length equals ``len(cout_tedges) - 1``.

    Raises
    ------
    ValueError
        If flow and cin have different lengths, or tedges has wrong length.

    See Also
    --------
    push_pull_inverse : Estimate injection concentration from extraction data.
    gamma_push_pull : Convenience wrapper with gamma-distributed layer heights.
    :ref:`radial-kernel-derivation` : First-principles derivation of the
        radial diffusion kernel used internally.
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

    dt = np.diff(tedges) / pd.Timedelta("1D")

    # Always prepend a large background injection to establish ambient
    # aquifer concentration. The prepended bin serves two purposes:
    # (1) It represents the ambient aquifer water that contacts the outer
    #     edge of the injected plume, so extraction beyond the injection
    #     zone correctly pulls water at ``c_background``.
    # (2) Downstream normalization attributes any numerical erf boundary
    #     artifact to the existing weight distribution, guaranteeing that
    #     a constant injection concentration equal to ``c_background``
    #     produces a constant extraction concentration at machine precision.
    v_cum_test = np.cumsum(flow * dt)
    max_abs_v = max(np.max(np.abs(np.concatenate(([0.0], v_cum_test)))), 1.0)
    prepend_volume = 1000.0 * max_abs_v
    flow = np.concatenate(([prepend_volume], flow))
    cin = np.concatenate(([c_background], cin))
    dt = np.concatenate(([1.0], dt))
    tedges = tedges.insert(0, tedges[0] - pd.Timedelta("1D"))  # type: ignore[assignment]
    n += 1

    n_layers = len(layer_heights)
    has_diffusion = molecular_diffusivity > 0 or longitudinal_dispersivity > 0

    tedges_days = ((tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)
    n_cout = len(cout_tedges) - 1
    has_extraction = np.zeros(n_cout, dtype=bool)

    if not has_diffusion:
        # Pure advection: push-pull attribution, layer heterogeneity invisible
        w, has_extraction = _push_pull_advection_matrix(
            flow=flow,
            dt=dt,
            tedges_days=tedges_days,
            cout_tedges_days=cout_tedges_days,
        )
    else:
        # With diffusion: average coefficient matrices across layers
        v_cum = np.concatenate(([0.0], np.cumsum(flow * dt)))

        # Compute once: shared across all layers
        v_max_after = np.maximum.accumulate(v_cum[::-1])[::-1]

        w = np.zeros((n_cout, n))
        for h in layer_heights:
            w_layer, has_extraction = _push_pull_diffusion_matrix(
                flow=flow,
                tedges_days=tedges_days,
                cout_tedges_days=cout_tedges_days,
                v_cum=v_cum,
                layer_height=h,
                porosity=porosity,
                retardation_factor=retardation_factor,
                n_layers=n_layers,
                molecular_diffusivity=molecular_diffusivity,
                longitudinal_dispersivity=longitudinal_dispersivity,
                v_max_after=v_max_after,
            )
            w += w_layer
        w /= n_layers

    # Sanitize ``cin`` at bins whose values are physically ignored
    # (non-injection bins) so that any user-supplied NaN there cannot
    # contaminate valid extraction rows through the matmul. We do NOT
    # overwrite NaN at injection bins: a NaN injection must propagate to
    # every extraction bin it contributes to so the caller can see the
    # missing measurement.
    is_injection = flow > 0
    cin_clean = np.where(is_injection, cin, 0.0)

    cout = w @ cin_clean

    # NaN where no extraction volume contributes to the output bin
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
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
    regularization_strength: float = 1e-3,
) -> npt.NDArray[np.floating]:
    """Estimate injection concentration from extraction measurements.

    Solves the inverse transport problem for the radial push-pull well model
    using Tikhonov regularization via :func:`gwtransport.utils.solve_inverse_transport`.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cout : array-like, shape (n_cout,)
        Measured extraction concentration [concentration units].
        Has shape (n_cout,) matching the ``cout_tedges`` grid.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    cout_tedges : DatetimeIndex
        Output time bin edges (n_cout+1 edges for n_cout bins). The ``cout``
        array is assumed to live on this grid.
    layer_heights : array-like, shape (N,)
        Height of each horizontal streamtube flowing to the well screen [m].
        The well screen is divided into N streamtubes of equal flow but
        different heights. Thinner streamtubes push the advective front
        further radially for the same injected volume, causing more
        diffusive spreading along the flow path.
    porosity : float
        Porosity n [-].
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water [concentration
        units]. Default is 0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-3.

    Returns
    -------
    ndarray, shape (n,)
        Estimated injection concentration. NaN for extraction and rest bins.

    See Also
    --------
    push_pull : Forward model.
    gwtransport.utils.solve_inverse_transport : Tikhonov solver.
    :ref:`radial-kernel-derivation` : First-principles derivation of the
        radial diffusion kernel used internally.
    """
    flow = np.asarray(flow, dtype=float)
    cout = np.asarray(cout, dtype=float)
    layer_heights = np.asarray(layer_heights, dtype=float)
    n = len(flow)

    dt = np.diff(tedges) / pd.Timedelta("1D")

    # Always prepend a large background injection (mirror of forward model).
    v_cum_test = np.cumsum(flow * dt)
    max_abs_v = max(np.max(np.abs(np.concatenate(([0.0], v_cum_test)))), 1.0)
    prepend_volume = 1000.0 * max_abs_v
    flow = np.concatenate(([prepend_volume], flow))
    dt = np.concatenate(([1.0], dt))
    tedges = tedges.insert(0, tedges[0] - pd.Timedelta("1D"))  # type: ignore[assignment]
    n += 1

    n_layers = len(layer_heights)
    has_diffusion = molecular_diffusivity > 0 or longitudinal_dispersivity > 0

    tedges_days = ((tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)
    cout_tedges_days = ((cout_tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)
    n_cout = len(cout_tedges) - 1
    has_extraction = np.zeros(n_cout, dtype=bool)

    if not has_diffusion:
        w, has_extraction = _push_pull_advection_matrix(
            flow=flow,
            dt=dt,
            tedges_days=tedges_days,
            cout_tedges_days=cout_tedges_days,
        )
    else:
        v_cum = np.concatenate(([0.0], np.cumsum(flow * dt)))

        # Compute once: shared across all layers
        v_max_after = np.maximum.accumulate(v_cum[::-1])[::-1]

        w = np.zeros((n_cout, n))
        for h in layer_heights:
            w_layer, has_extraction = _push_pull_diffusion_matrix(
                flow=flow,
                tedges_days=tedges_days,
                cout_tedges_days=cout_tedges_days,
                v_cum=v_cum,
                layer_height=h,
                porosity=porosity,
                retardation_factor=retardation_factor,
                n_layers=n_layers,
                molecular_diffusivity=molecular_diffusivity,
                longitudinal_dispersivity=longitudinal_dispersivity,
                v_max_after=v_max_after,
            )
            w += w_layer
        w /= n_layers

    is_injection = flow > 0

    # An output bin is usable for the inverse solve only if the forward
    # model attributes extraction volume to it AND the caller actually
    # supplied a finite measurement there. User-supplied NaN at an
    # otherwise-valid bin means "no measurement" and must be masked out
    # of the solver rather than silently coerced to zero, which would
    # bias the Tikhonov target.
    valid_cout = has_extraction & np.isfinite(cout)

    # Subtract the prepended background's contribution from the observed
    # concentration, then solve for the real (non-prepended) injection
    # bins. Invalid rows carry 0.0 as a harmless placeholder; the solver
    # ignores them via ``valid_rows=valid_cout``.
    bg_contribution = w[:, 0] * c_background
    observed = np.where(valid_cout, cout - bg_contribution, 0.0)

    cin_recovered = solve_inverse_transport(
        w_forward=w[:, 1:],
        observed=observed,
        n_output=n - 1,
        regularization_strength=regularization_strength,
        valid_rows=valid_cout,
    )

    # ``cin_recovered`` already carries NaN for injection columns that
    # the solver could not constrain (inactive in valid rows). Start
    # from NaN and only copy the recovered value into real injection
    # bins, so rest/extraction bins remain NaN and unrecoverable
    # injection bins stay NaN as well.
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
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute extraction concentration with gamma-distributed layer heights.

    Convenience wrapper around :func:`push_pull` that discretizes a gamma
    distribution into equal-probability bins for layer heights.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cin : array-like, shape (n,)
        Injection concentration per time bin [concentration units].
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    cout_tedges : DatetimeIndex
        Output time bin edges (n_cout+1 edges for n_cout bins). The
        extraction concentration is flow-weighted averaged onto this grid.
    alpha : float, optional
        Shape parameter of gamma distribution for layer heights.
    beta : float, optional
        Scale parameter of gamma distribution for layer heights.
    mean : float, optional
        Mean layer height [m].
    std : float, optional
        Standard deviation of layer heights [m].
    n_bins : int, optional
        Number of bins for gamma discretization. Default is 100.
    porosity : float
        Porosity n [-].
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water [concentration
        units]. Default is 0.

    Returns
    -------
    ndarray, shape (n_cout,)
        Flow-weighted extraction concentration. NaN for bins without extraction.
        Length equals ``len(cout_tedges) - 1``.

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
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    c_background: float = 0.0,
    regularization_strength: float = 1e-3,
) -> npt.NDArray[np.floating]:
    """Estimate injection concentration with gamma-distributed layer heights.

    Convenience wrapper around :func:`push_pull_inverse` that discretizes
    a gamma distribution into equal-probability bins for layer heights.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cout : array-like, shape (n_cout,)
        Measured extraction concentration [concentration units].
        Has shape (n_cout,) matching the ``cout_tedges`` grid.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    cout_tedges : DatetimeIndex
        Output time bin edges (n_cout+1 edges for n_cout bins). The ``cout``
        array is assumed to live on this grid.
    alpha : float, optional
        Shape parameter of gamma distribution for layer heights.
    beta : float, optional
        Scale parameter of gamma distribution for layer heights.
    mean : float, optional
        Mean layer height [m].
    std : float, optional
        Standard deviation of layer heights [m].
    n_bins : int, optional
        Number of bins for gamma discretization. Default is 100.
    porosity : float
        Porosity n [-].
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    c_background : float, optional
        Background concentration of the ambient aquifer water [concentration
        units]. Default is 0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-3.

    Returns
    -------
    ndarray, shape (n,)
        Estimated injection concentration. NaN for extraction and rest bins.

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
    )
