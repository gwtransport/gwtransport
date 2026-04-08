"""
Radial Push-Pull Well Transport Model.

This module provides functions to model solute transport in a single well with radial
flow in a vertically heterogeneous aquifer. The model handles bidirectional flow
(injection and extraction) through a LIFO (last-in-first-out) transport mechanism,
where the most recently injected water is extracted first.

The key difference from the linear infiltration-to-extraction models in
:mod:`gwtransport.advection` and :mod:`gwtransport.diffusion` is:

- **Bidirectional flow**: Water is injected and then extracted from the same well.
  Transport follows LIFO ordering — the last parcel injected is the first extracted.
- **Radial geometry**: The radial distance to the advective front scales as
  r² = r_w² + V / (N · π · h · n · R), where V is cumulative volume, N is the
  number of layers, h is the layer height, n is porosity, and R is retardation.

**Heterogeneity model**: The aquifer is divided into N layers with equal flow weight
but different heights h_i. Layers with smaller h push the tracer front further
radially for the same injected volume, leading to more diffusive spreading and
irreversible mixing. Without diffusion, layer heterogeneity is invisible — all
layers produce identical breakthrough (perfect LIFO recovery).

Available functions:

- :func:`push_pull_well` - Forward model: compute extraction concentration for a
  radial push-pull well given injection concentration and flow history.

- :func:`push_pull_well_inverse` - Inverse model: estimate injection concentration
  from measured extraction concentration using Tikhonov regularization.

- :func:`gamma_push_pull_well` - Convenience wrapper using gamma-distributed layer
  heights. Parameterizable via (alpha, beta) or (mean, std).

- :func:`gamma_push_pull_well_inverse` - Convenience wrapper for the inverse model
  with gamma-distributed layer heights.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special

from gwtransport import gamma
from gwtransport.utils import solve_inverse_transport

# Gauss-Legendre quadrature nodes and weights for volume-space integration
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


def _push_pull_advection_matrix(
    *,
    flow: npt.NDArray[np.floating],
    dt: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Build LIFO stack coefficient matrix for pure advection.

    Processes bins chronologically: injection bins push volume onto a stack,
    extraction bins pop from the stack top. The resulting matrix W satisfies
    ``cout = W @ cin``.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day]. Positive = injection, negative = extraction.
    dt : ndarray, shape (n,)
        Time bin widths [days].

    Returns
    -------
    ndarray, shape (n, n)
        Coefficient matrix. Rows sum to 1 for extraction bins that fully
        recover injected volume, < 1 for over-extraction, and 0 for
        injection/rest bins.
    """
    n = len(flow)
    w = np.zeros((n, n))
    # Stack entries: [bin_index, remaining_volume]
    stack: list[list[float]] = []

    for i in range(n):
        vol = flow[i] * dt[i]
        if vol > 0:
            # Injection: push onto stack
            stack.append([float(i), vol])
        elif vol < 0:
            # Extraction: pop from stack (LIFO)
            vol_needed = -vol
            extract_vol = vol_needed  # total extraction volume for normalization
            while vol_needed > 0 and stack:
                j_idx, vol_avail = stack[-1]
                consumed = min(vol_avail, vol_needed)
                w[i, int(j_idx)] += consumed / extract_vol
                vol_needed -= consumed
                stack[-1][1] -= consumed
                if stack[-1][1] <= 0:
                    stack.pop()
        # vol == 0: rest period, no action

    return w


def _signed_radial_distance(
    *,
    delta_volume: npt.NDArray[np.floating],
    layer_height: float,
    porosity: float,
    retardation_factor: float,
    well_radius: float,
    n_layers: int,
) -> npt.NDArray[np.floating]:
    r"""Compute signed radial distance from volume difference.

    .. math::

        x = \text{sign}(\Delta V) \cdot \left(
            \sqrt{r_w^2 + \frac{|\Delta V|}{N \cdot \pi \cdot h \cdot n \cdot R}}
            - r_w
        \right)

    Parameters
    ----------
    delta_volume : ndarray
        Volume difference [m³]. Positive means front has been extracted past
        the well screen.
    layer_height : float
        Layer height h [m].
    porosity : float
        Porosity n [-].
    retardation_factor : float
        Retardation factor R [-].
    well_radius : float
        Well radius r_w [m].
    n_layers : int
        Number of layers N.

    Returns
    -------
    ndarray
        Signed radial distance [m]. Same shape as delta_volume.
    """
    dv = np.asarray(delta_volume, dtype=float)
    scale = n_layers * np.pi * layer_height * porosity * retardation_factor
    r = np.sqrt(well_radius**2 + np.abs(dv) / scale) - well_radius
    return np.sign(dv) * r


def _erf_mean_volume_radial(
    *,
    v_cum: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    layer_height: float,
    porosity: float,
    well_radius: float,
    retardation_factor: float,
    n_layers: int,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
) -> npt.NDArray[np.floating]:
    r"""Compute mean erf in volume space for radial geometry.

    For each (cout_bin i, cin_edge j), computes:

    .. math::

        \text{mean\_erf}_{i,j} = \frac{1}{|\Delta V_i|}
        \int_{V_i}^{V_{i+1}} \text{erf}\!\left(
            \frac{x(V)}{2\sqrt{D_{\text{eff}} \cdot \tau_{\text{eff}}}}
        \right) dV

    where x(V) is the signed radial distance (nonlinear in V) and the
    effective diffusion-time product accounts for both molecular diffusion
    and mechanical dispersion along the radial path.

    Parameters
    ----------
    v_cum : ndarray, shape (n+1,)
        Cumulative volume at time edges [m³].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days from reference.
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day].
    layer_height : float
        Layer height h [m].
    porosity : float
        Porosity n [-].
    well_radius : float
        Well radius r_w [m].
    retardation_factor : float
        Retardation factor R [-].
    n_layers : int
        Number of layers N.
    molecular_diffusivity : float
        Molecular diffusivity D_m [m²/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].

    Returns
    -------
    ndarray, shape (n_cout_bins, n_cin_edges)
        Mean erf value for each (extraction bin, injection edge) pair.
        NaN for inactive or injection bins.
    """
    n = len(flow)
    n_edges = n + 1
    scale = n_layers * np.pi * layer_height * porosity * retardation_factor

    # Identify extraction bins (flow < 0)
    is_extraction = flow < 0

    # Mean erf: shape (n_cout_bins, n_cin_edges)
    # cout bins = extraction bins only contribute, but we compute for all
    # cin edges = all edges (injection boundaries)
    mean_erf = np.full((n, n_edges), np.nan)

    if not np.any(is_extraction):
        return mean_erf

    # Precompute v_max_after[j] = max(v_cum[j:]) for path length calculation
    v_max_after = np.maximum.accumulate(v_cum[::-1])[::-1]

    # D = 0 case: exact step function
    no_diffusion = (molecular_diffusivity == 0.0) and (longitudinal_dispersivity == 0.0)

    for i in range(n):
        if not is_extraction[i]:
            continue

        v_lo = v_cum[i]
        v_hi = v_cum[i + 1]
        dv_bin = v_hi - v_lo  # negative for extraction

        if dv_bin == 0.0:
            continue

        for j in range(n_edges):
            v_j = v_cum[j]

            if no_diffusion:
                # Step function: erf = sign(V_j - V)
                # For extraction bin i, V goes from v_lo to v_hi (v_hi < v_lo)
                # sign(V_j - V) is +1 when V < V_j (front extracted past well)
                # and -1 when V > V_j (front still in aquifer)
                if v_j <= v_hi:
                    mean_erf[i, j] = -1.0  # All volume beyond front
                elif v_j >= v_lo:
                    mean_erf[i, j] = 1.0  # All volume before front
                else:
                    frac_positive = (v_lo - v_j) / (v_lo - v_hi)
                    mean_erf[i, j] = 1.0 - 2.0 * frac_positive
            else:
                # GL quadrature over extraction bin volume
                v_a = min(v_lo, v_hi)
                v_b = max(v_lo, v_hi)
                dv = v_b - v_a

                v_mid = (v_a + v_b) / 2.0
                v_half = dv / 2.0

                # GL nodes in volume space
                v_nodes = v_mid + v_half * _GL_NODES  # shape (16,)

                # Signed volume difference: V_j - V(t)
                # Positive when front has been extracted past well
                delta_v = v_j - v_nodes
                x_nodes = _signed_radial_distance(
                    delta_volume=delta_v,
                    layer_height=layer_height,
                    porosity=porosity,
                    retardation_factor=retardation_factor,
                    well_radius=well_radius,
                    n_layers=n_layers,
                )

                # Time at each GL node: linear interpolation within bin i
                # (v_cum is non-monotonic, so np.interp on v_cum doesn't work)
                t_lo = tedges_days[i]
                t_hi = tedges_days[i + 1]
                t_nodes = t_lo + (v_nodes - v_lo) / (v_hi - v_lo) * (t_hi - t_lo)

                # Time of injection edge j
                t_j = tedges_days[j]

                # Diffusion time: elapsed since injection
                tau = np.abs(t_nodes - t_j)

                # Path length for dispersivity
                # Max push-out distance for parcels injected at edge j
                v_max_j = v_max_after[j]
                x_max = np.sqrt(well_radius**2 + max(v_max_j - v_j, 0.0) / scale) - well_radius

                # Current signed radial position of the front
                # net_vol = V(t) - V_j: positive when still in aquifer
                net_vol = v_nodes - v_j
                x_cur = _signed_radial_distance(
                    delta_volume=net_vol,
                    layer_height=layer_height,
                    porosity=porosity,
                    retardation_factor=retardation_factor,
                    well_radius=well_radius,
                    n_layers=n_layers,
                )

                # Total path: out to x_max, then back to x_cur
                # path_len = x_max + (x_max - x_cur) = 2*x_max - x_cur
                path_len = np.maximum(2.0 * x_max - x_cur, 0.0)

                # D_eff * tau_eff = D_m * tau + alpha_L * path_len
                d_tau = molecular_diffusivity * tau + longitudinal_dispersivity * path_len

                # erf argument: x / (2 * sqrt(d_tau))
                with np.errstate(divide="ignore", invalid="ignore"):
                    arg = x_nodes / (2.0 * np.sqrt(d_tau))
                erf_vals = np.where(np.isfinite(arg), special.erf(arg), np.sign(x_nodes))

                # Integrate via GL quadrature
                mean_erf[i, j] = np.dot(erf_vals, _GL_WEIGHTS) / 2.0

    return mean_erf


def _push_pull_diffusion_matrix(
    *,
    flow: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    v_cum: npt.NDArray[np.floating],
    layer_height: float,
    porosity: float,
    well_radius: float,
    retardation_factor: float,
    n_layers: int,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
) -> npt.NDArray[np.floating]:
    """Build coefficient matrix with radial diffusion for one layer.

    For each (cout_bin, cin_edge), computes flow-weighted mean erf via
    :func:`_erf_mean_volume_radial`. Converts to coefficients.

    In the push-pull model, frac increases with edge index j (later injection
    = front closer to well = extracted sooner), so the coefficient is
    ``frac_end - frac_start`` (reversed from the linear model). Only injection
    bin columns are populated; extraction/rest columns remain zero.

    Parameters
    ----------
    flow : ndarray, shape (n,)
        Flow rate per bin [m³/day].
    tedges_days : ndarray, shape (n+1,)
        Time edges in days.
    v_cum : ndarray, shape (n+1,)
        Cumulative volume at time edges [m³].
    layer_height : float
        Layer height h [m].
    porosity : float
        Porosity n [-].
    well_radius : float
        Well radius r_w [m].
    retardation_factor : float
        Retardation factor R [-].
    n_layers : int
        Number of layers N.
    molecular_diffusivity : float
        Molecular diffusivity D_m [m²/day].
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m].

    Returns
    -------
    ndarray, shape (n, n)
        Coefficient matrix W such that ``cout = W @ cin``.
    """
    n = len(flow)
    is_injection = flow > 0

    response = _erf_mean_volume_radial(
        v_cum=v_cum,
        tedges_days=tedges_days,
        flow=flow,
        layer_height=layer_height,
        porosity=porosity,
        well_radius=well_radius,
        retardation_factor=retardation_factor,
        n_layers=n_layers,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
    )

    # Convert mean_erf (n, n+1) -> coefficients (n, n)
    # frac[i,j] = probability that water injected before edge j has been
    # extracted in bin i. In the push-pull LIFO model, frac increases with j
    # (later injection edges = closer to well = extracted sooner).
    frac = 0.5 * (1.0 + response)
    frac_filled = np.nan_to_num(frac, nan=0.0)

    coeff = np.zeros((n, n))
    for j in range(n):
        if is_injection[j]:
            # coeff[i,j] = frac at right edge of bin j minus frac at left edge
            coeff[:, j] = frac_filled[:, j + 1] - frac_filled[:, j]

    return np.maximum(coeff, 0.0)


def push_pull_well(
    *,
    flow: npt.ArrayLike,
    cin: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    layer_heights: npt.ArrayLike,
    porosity: float,
    well_radius: float = 0.0,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute extraction concentration for a radial push-pull well.

    Models solute transport in a single well with radial flow in a vertically
    heterogeneous aquifer. Injection (flow > 0) pushes tracer outward; extraction
    (flow < 0) pulls it back in LIFO order.

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
    layer_heights : array-like, shape (N,)
        Layer heights h_i [m]. Layers have equal flow weight.
    porosity : float
        Porosity n [-].
    well_radius : float, optional
        Well radius r_w [m]. Default is 0.
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.

    Returns
    -------
    ndarray, shape (n,)
        Extraction concentration. NaN for injection and rest bins.

    Raises
    ------
    ValueError
        If flow and cin have different lengths, or tedges has wrong length.

    See Also
    --------
    push_pull_well_inverse : Estimate injection concentration from extraction data.
    gamma_push_pull_well : Convenience wrapper with gamma-distributed layer heights.
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
    n_layers = len(layer_heights)
    has_diffusion = molecular_diffusivity > 0 or longitudinal_dispersivity > 0

    if not has_diffusion:
        # Pure advection: LIFO stack, layer heterogeneity invisible
        w = _push_pull_advection_matrix(flow=flow, dt=dt)
    else:
        # With diffusion: average coefficient matrices across layers
        tedges_days = ((tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)
        v_cum = np.concatenate(([0.0], np.cumsum(flow * dt)))

        w = np.zeros((n, n))
        for h in layer_heights:
            w_layer = _push_pull_diffusion_matrix(
                flow=flow,
                tedges_days=tedges_days,
                v_cum=v_cum,
                layer_height=h,
                porosity=porosity,
                well_radius=well_radius,
                retardation_factor=retardation_factor,
                n_layers=n_layers,
                molecular_diffusivity=molecular_diffusivity,
                longitudinal_dispersivity=longitudinal_dispersivity,
            )
            w += w_layer
        w /= n_layers

    # Compute output concentration
    cout = w @ cin
    is_extraction = flow < 0
    cout[~is_extraction] = np.nan

    return cout


def push_pull_well_inverse(
    *,
    flow: npt.ArrayLike,
    cout: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    layer_heights: npt.ArrayLike,
    porosity: float,
    well_radius: float = 0.0,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
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
    cout : array-like, shape (n,)
        Measured extraction concentration [concentration units].
        Values during injection/rest bins are ignored.
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
    layer_heights : array-like, shape (N,)
        Layer heights h_i [m]. Layers have equal flow weight.
    porosity : float
        Porosity n [-].
    well_radius : float, optional
        Well radius r_w [m]. Default is 0.
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-3.

    Returns
    -------
    ndarray, shape (n,)
        Estimated injection concentration. NaN for extraction and rest bins.

    See Also
    --------
    push_pull_well : Forward model.
    gwtransport.utils.solve_inverse_transport : Tikhonov solver.
    """
    flow = np.asarray(flow, dtype=float)
    cout = np.asarray(cout, dtype=float)
    layer_heights = np.asarray(layer_heights, dtype=float)
    n = len(flow)

    dt = np.diff(tedges) / pd.Timedelta("1D")
    n_layers = len(layer_heights)
    has_diffusion = molecular_diffusivity > 0 or longitudinal_dispersivity > 0

    if not has_diffusion:
        w = _push_pull_advection_matrix(flow=flow, dt=dt)
    else:
        tedges_days = ((tedges - tedges[0]) / pd.Timedelta("1D")).values.astype(float)
        v_cum = np.concatenate(([0.0], np.cumsum(flow * dt)))

        w = np.zeros((n, n))
        for h in layer_heights:
            w_layer = _push_pull_diffusion_matrix(
                flow=flow,
                tedges_days=tedges_days,
                v_cum=v_cum,
                layer_height=h,
                porosity=porosity,
                well_radius=well_radius,
                retardation_factor=retardation_factor,
                n_layers=n_layers,
                molecular_diffusivity=molecular_diffusivity,
                longitudinal_dispersivity=longitudinal_dispersivity,
            )
            w += w_layer
        w /= n_layers

    is_extraction = flow < 0
    is_injection = flow > 0

    cin_recovered = solve_inverse_transport(
        w_forward=w,
        observed=np.where(is_extraction, cout, 0.0),
        n_output=n,
        regularization_strength=regularization_strength,
        valid_rows=is_extraction,
    )

    out = np.full(n, np.nan)
    injection_mask = is_injection
    idx = np.flatnonzero(injection_mask)
    out[idx] = cin_recovered[idx]
    return out


def gamma_push_pull_well(
    *,
    flow: npt.ArrayLike,
    cin: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    porosity: float,
    well_radius: float = 0.0,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
) -> npt.NDArray[np.floating]:
    """Compute extraction concentration with gamma-distributed layer heights.

    Convenience wrapper around :func:`push_pull_well` that discretizes a gamma
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
    well_radius : float, optional
        Well radius r_w [m]. Default is 0.
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.

    Returns
    -------
    ndarray, shape (n,)
        Extraction concentration. NaN for injection and rest bins.

    See Also
    --------
    push_pull_well : Base function with explicit layer heights.
    gwtransport.gamma.bins : Gamma distribution discretization.
    """
    gamma_bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    layer_heights = gamma_bins["expected_values"]

    return push_pull_well(
        flow=flow,
        cin=cin,
        tedges=tedges,
        layer_heights=layer_heights,
        porosity=porosity,
        well_radius=well_radius,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
    )


def gamma_push_pull_well_inverse(
    *,
    flow: npt.ArrayLike,
    cout: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,
    porosity: float,
    well_radius: float = 0.0,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    longitudinal_dispersivity: float = 0.0,
    regularization_strength: float = 1e-3,
) -> npt.NDArray[np.floating]:
    """Estimate injection concentration with gamma-distributed layer heights.

    Convenience wrapper around :func:`push_pull_well_inverse` that discretizes
    a gamma distribution into equal-probability bins for layer heights.

    Parameters
    ----------
    flow : array-like, shape (n,)
        Flow rate per time bin [m³/day]. Positive = injection,
        negative = extraction, zero = rest.
    cout : array-like, shape (n,)
        Measured extraction concentration [concentration units].
    tedges : DatetimeIndex
        Time bin edges (n+1 edges for n bins).
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
    well_radius : float, optional
        Well radius r_w [m]. Default is 0.
    retardation_factor : float, optional
        Retardation factor R [-]. Default is 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity D_m [m²/day]. Default is 0.
    longitudinal_dispersivity : float, optional
        Longitudinal dispersivity alpha_L [m]. Default is 0.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-3.

    Returns
    -------
    ndarray, shape (n,)
        Estimated injection concentration. NaN for extraction and rest bins.

    See Also
    --------
    push_pull_well_inverse : Base function with explicit layer heights.
    gwtransport.gamma.bins : Gamma distribution discretization.
    """
    gamma_bins = gamma.bins(alpha=alpha, beta=beta, mean=mean, std=std, n_bins=n_bins)
    layer_heights = gamma_bins["expected_values"]

    return push_pull_well_inverse(
        flow=flow,
        cout=cout,
        tedges=tedges,
        layer_heights=layer_heights,
        porosity=porosity,
        well_radius=well_radius,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        regularization_strength=regularization_strength,
    )
