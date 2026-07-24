r"""
Analytical solutions for 1D advection-dispersion transport.

Water infiltrates and is transported in parallel along multiple aquifer pore volumes to
extraction. For each aquifer pore volume, transport is 1D advection with microdispersion,
molecular diffusion, and linear sorption; the spread across aquifer pore volumes provides
macrodispersion. Forward and backward modeling are supported. The flow is assumed orthogonal.

The orthogonal-flow (Cartesian) geometry is what makes the Kreft-Zuber breakthrough the exact
1D solution used below.

Key functions:

- :func:`infiltration_to_extraction` - Main transport function combining advection,
  microdispersion, and molecular diffusion with explicit pore volume distribution and
  streamline lengths.

- :func:`extraction_to_infiltration` - Inverse operation (deconvolution with dispersion).

- :func:`gamma_infiltration_to_extraction` - Gamma-distributed pore volumes with dispersion.
  Models aquifer heterogeneity with 2-parameter gamma distribution. Parameterizable via
  (alpha, beta) or (mean, std). Discretizes gamma distribution into equal-probability bins.

- :func:`gamma_extraction_to_infiltration` - Gamma-distributed pore volumes, deconvolution
  with dispersion. Symmetric inverse of gamma_infiltration_to_extraction.

When to choose this module vs :mod:`gwtransport.diffusion_fast`
---------------------------------------------------------------

This is the reference implementation: it evaluates the bin-averaged Kreft-Zuber flux
concentration by resolution-aware composite Gauss-Legendre quadrature (splitting at
flow-bin boundaries, with extra front-centred panels wherever a sharp breakthrough front
is otherwise under-resolved).
Prefer it only when the output grid is coarser than the flow detail -- it integrates the
full within-bin flow, which the closed-form :mod:`gwtransport.diffusion_fast` approximates as
constant per output bin. Otherwise that module computes the same physics to machine
precision for *every* parameter regime (including ``retardation_factor != 1`` with non-zero
molecular diffusivity, whose flux correction it also evaluates in closed form) and is
~80-90x faster (no quadrature, no residence-time inversion). Both modules accept
per-streamtube ``streamline_length`` / ``molecular_diffusivity`` /
``longitudinal_dispersivity`` arrays (heterogeneous flow paths -- partially-penetrating
wells, wedge-shaped capture zones).

Reported outlet concentration: Kreft-Zuber (1978) flux concentration
---------------------------------------------------------------------

The outlet concentration reported by this module is the **flux concentration**

    C_F(L, t) = C_R(L, t) - (D_s / v_s) * dC_R/dx \|_{x=L}

with the solute-front (retarded-frame) velocity v_s = Q L / (R V_pore) and the
dispersion D_s = D_m + alpha_L * v_s, so the flux coefficient is
D_s / v_s = D_m / v_s + alpha_L = R D_m / v_fluid + alpha_L (with the fluid
velocity v_fluid = Q L / V_pore). The resident profile C_R solves the retarded
ADE with advection v_s and dispersion D_s, so its flux-vs-resident correction
must use v_s — not v_fluid; pairing v_s with the moving-frame variance below is
what conserves mass for R > 1 with D_m > 0.

— the solute mass flux at the outlet divided by the volumetric fluid flux. This
is what is measured when sampling the extracted fluid. The resident
concentration ``C_R`` is Bear (1972) eq. 10.6.4, the variable-flow moving-frame
Ogata-Banks solution

    C_R(L, V; t_j) = 0.5 * erfc((L - xi_j(V)) / (2 * sqrt(D_t(V))))

with the dispersion variance accumulated in the moving (Lagrangian) frame:

    D_t(V) = sigma^2(V) / 2 = D_m * tau(V) + alpha_L * xi(V)

where:

- D_m is the effective molecular (or thermal) diffusivity [m²/day]
- alpha_L is the longitudinal dispersivity [m]
- tau(V) is the elapsed time since infiltration [day], with V the cumulative
  extracted volume
- xi(V) = L (V - V_j) / (R V_pore) is the distance the parcel has actually
  travelled [m]

The K-Z flux-correction term is what makes the column-sum invariant
``integral Q c_out dt = integral Q c_in dt`` hold under arbitrary variable Q.
Without it, the leading-order C_R loses O(1/Pe) per column under variable Q +
pure D_m (issue #180).

Implementation: the bin-averaged C_F is computed by resolution-aware composite
Gauss-Legendre quadrature in volume space, split at flow-bin boundaries so each
sub-interval sees a linear t(V). Within a sub-interval the erf-like front has
width ``sqrt(4*D_t)`` (in volume units); for near-zero dispersivity this can be
orders of magnitude below the flow-bin width, so a single fixed-order rule
cannot resolve it. Sub-intervals whose front is under-resolved are therefore
tiled with front-centred panels (fine near the front, flat tails outside),
which restores the column-mass invariant to ~1e-11 for every dispersion regime;
smooth/already-resolved sub-intervals keep the plain single 16-point rule. The
variance is evaluated at each quadrature node from the parcel's own tau and xi
histories — never capped at the residence time. The K-Z identity requires
Bear's formula to satisfy the variable-coefficient ADE exactly, which holds only
when D_t is allowed to keep growing past breakthrough.

Macrodispersion vs microdispersion
----------------------------------

This module adds microdispersion (alpha_L) and molecular diffusion (D_m) on top of
macrodispersion captured by the pore volume distribution (APVD). Both represent velocity
heterogeneity at different scales. Microdispersion is an aquifer property; macrodispersion
depends additionally on hydrological boundary conditions. See :ref:`concept-dispersion-scales`
for guidance on when to use each approach and how to avoid double-counting spreading effects.

Streamtube assumption (no cross-sectional area parameter)
---------------------------------------------------------

Each entry in ``aquifer_pore_volumes`` is treated as an independent 1D streamtube. There is
no cross-sectional area parameter: the variance budget uses ``2 D_m tau`` (molecular
diffusion in time) and ``2 alpha_L xi`` (microdispersion in travelled distance), with
the streamline length ``L`` and the pore volume ``V_pore`` together fixing the implicit
streamtube cross-section ``A = V_pore / L``. Callers who need distributed-area effects must
provide multiple streamtubes (via ``aquifer_pore_volumes`` or the gamma-parameterised
wrappers).

References
----------
Bear, J. (1972). Dynamics of Fluids in Porous Media. American Elsevier
Publishing Company. Equation 10.6.4 (variable-flow Ogata-Banks form). Provides
the resident concentration ``C_R``.

Kreft, A., & Zuber, A. (1978). On the physical meaning of the dispersion
equation and its solutions for different initial and boundary conditions.
Chemical Engineering Science, 33(11), 1471-1480. Eq. 2 gives the resident-to-
flux concentration transformation; Eq. 1 is the mass-balance identity that
makes the column-sum invariant exact.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import special

from gwtransport import gamma
from gwtransport._time import dt_to_days, tedges_to_days
from gwtransport._validation import (
    _validate_no_nan,
    _validate_non_negative_array,
    _validate_positive_array,
    _validate_retardation_factor,
    _validate_scalar_or_matching_length,
    _validate_tedges_parity,
)
from gwtransport.residence_time import fraction_explained_full
from gwtransport.utils import cumulative_flow_volume, solve_inverse_transport

# Numerical tolerance for coefficient sum to determine valid output bins
EPSILON_COEFF_SUM = 1e-10

# Gauss-Legendre quadrature nodes and weights for volume-space integration
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)

# Resolution-aware composite-quadrature template for the erf-like breakthrough front.
# When the front width sqrt(4*D_t) (in volume units) is much smaller than a
# (cell, flow-bin) sub-interval, plain 16-point GL cannot resolve it. Cells flagged
# as under-resolved get panels placed at ``V_front + front_width * _FRONT_OFFSETS``
# (clipped to the sub-interval); the two outer panels then cover the flat erf tails,
# where 16-point GL is already exact. Panels near the front are ~1 front-width wide,
# spanning +-6 front-widths (erf and the flux-correction Gaussian are flat to ~1e-16
# beyond that). A cell is refined only when its front lies inside the sub-interval and
# the sub-interval is wider than _REFINE_RATIO front-widths, so adaptivity triggers
# only near sharp fronts (smooth regimes keep the plain single-panel cost and answer).
_FRONT_REACH = 6.0
_FRONT_OFFSETS = np.arange(-_FRONT_REACH, _FRONT_REACH + 0.5, 1.0)
_REFINE_RATIO = 4.0


def _cfrac_mean_volume(
    *,
    step_widths: npt.NDArray[np.floating],
    cumulative_volume_at_cout_tedges: npt.NDArray[np.floating],
    cumulative_volume_at_cin_tedges: npt.NDArray[np.floating],
    tedges_days: npt.NDArray[np.floating],
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    r_vpv: float,
    streamline_len: float,
) -> npt.NDArray[np.floating]:
    r"""Compute bin-averaged flux concentration at the outlet for each cell.

    For each cell (cout-bin *i*, cin-edge *j*), computes the flow-weighted
    average of the Kreft-Zuber (1978) **flux concentration** at the outlet:

    .. math::

        \text{frac}_{i,j} = \frac{1}{\Delta V_i}
        \int_{V_i}^{V_{i+1}} C_F\!\left(L,\,V;\,t_j\right) dV

    where :math:`C_F = C_R - (D_s / v_s) \, \partial_x C_R\big|_{x=L}` and
    :math:`C_R` is Bear's (1972) moving-frame solution:

    .. math::

        C_R(L, V; t_j) &= \tfrac{1}{2}\,
            \mathrm{erfc}\!\left( \frac{L - \xi_j(V)}{2\sqrt{D_t(V)}} \right) \\
        \xi_j(V) &= L \cdot (V - V_j) \,/\, (R\,V_\text{pore}) \\
        D_t(V) &= D_m\,\tau_j(V) + \alpha_L\,\xi_j(V),
            \quad \tau_j(V) = t(V) - t_j \\
        D_s &= D_m + \alpha_L\,v_s(t), \quad v_s(t) = Q(t)\,L\,/\,(R\,V_\text{pore}).

    The solute-front velocity :math:`v_s` (advection speed of the retarded ADE
    that :math:`C_R` solves), not the fluid velocity :math:`Q L / V_\text{pore}`,
    sets the flux coefficient :math:`D_s/v_s = D_m/v_s + \alpha_L`. The added
    flux-correction term

    .. math::

        \frac{D_s}{v_s(t(V))} \cdot
        \frac{1}{\sqrt{4\pi\,D_t(V)}}\,
        \exp\!\left( -\frac{(L - \xi_j(V))^2}{4\,D_t(V)} \right)

    converts Bear's *resident* concentration to a *flux* concentration. This
    makes the coefficient matrix conserve mass under the criterion
    ``integral Q c_out dt = integral Q c_in dt`` — the relevant invariant for
    tracer measurements taken in the extracted fluid (Kreft & Zuber, 1978,
    Eq. 5 and Eq. 1). Without this correction, Bear's leading-order kernel
    misses the dispersive boundary flux at the outlet and column-sum mass
    conservation fails by O(1/Pe) under variable Q.

    Implementation: resolution-aware composite Gauss-Legendre quadrature in
    volume space, split at flow-bin boundaries so that within each sub-interval
    :math:`t(V)` is linear. The erf-like front has width :math:`\sqrt{4 D_t}`
    (in volume units); a sub-interval whose front is under-resolved by a single
    16-point rule (front width far below the sub-interval width) is tiled with
    front-centred panels (see ``_FRONT_OFFSETS``), while smooth/already-resolved
    sub-intervals keep the plain single 16-point rule (bit-identical to it). No
    "fully capped" branch: the moving-frame variance keeps growing past
    breakthrough, and the K-Z identity requires Bear's formula to satisfy the
    variable-coefficient ADE exactly (which it does only without capping).

    Parameters
    ----------
    step_widths : ndarray, shape (n_cout_edges, n_cin_edges)
        x-position ``x(V_cout, V_cin) = (V_cout - V_cin - r_vpv) * L / r_vpv``
        at each (cout-edge, cin-edge). NaN for inactive cells. Equals
        :math:`\xi - L`.
    cumulative_volume_at_cout_tedges : ndarray, shape (n_cout_edges,)
        Cumulative extracted volume at each cout time edge [m³].
    cumulative_volume_at_cin_tedges : ndarray, shape (n_cin_edges,)
        Cumulative volume at each cin (flow) time edge [m³].
    tedges_days : ndarray, shape (n_cin_edges,)
        Flow time edges in days.
    molecular_diffusivity : float
        Effective (retarded-frame) molecular diffusivity D_m [m²/day].
        Contributes ``D_m * tau`` to the dispersion product ``D_t``.
    longitudinal_dispersivity : float
        Longitudinal dispersivity alpha_L [m]. Contributes ``alpha_L * xi``
        to the dispersion product ``D_t``.
    r_vpv : float
        Retardation factor times pore volume = R * V_pore [m³].
    streamline_len : float
        Streamline length L [m].

    Returns
    -------
    ndarray, shape (n_cout_bins, n_cin_edges)
        Bin-averaged flux concentration for each cell. NaN for inactive cells.

    References
    ----------
    Kreft, A., & Zuber, A. (1978). On the physical meaning of the dispersion
    equation and its solutions for different initial and boundary conditions.
    Chemical Engineering Science, 33(11), 1471-1480.
    """
    n_cout_edges, n_cin_edges = step_widths.shape
    n_cout_bins = n_cout_edges - 1

    x_lo = step_widths[:-1]
    x_hi = step_widths[1:]
    dx = x_hi - x_lo

    v_lo_arr = cumulative_volume_at_cout_tedges[:-1]
    v_hi_arr = cumulative_volume_at_cout_tedges[1:]

    is_valid = ~np.isnan(x_lo) & ~np.isnan(x_hi)

    frac = np.full((n_cout_bins, n_cin_edges), np.nan)

    # --- No dispersion: C_F = C_R = step function (no dispersive flux) ---
    if molecular_diffusivity == 0.0 and longitudinal_dispersivity == 0.0:
        with np.errstate(divide="ignore", invalid="ignore"):
            cr_no_disp = 0.5 + 0.5 * (np.abs(x_hi) - np.abs(x_lo)) / dx
        cr_no_disp = np.where(dx == 0.0, 0.5 + 0.5 * np.sign(x_lo), cr_no_disp)
        return np.where(is_valid, cr_no_disp, frac)

    # --- Pre-compute solute-front velocity and K-Z coefficient (D_s/v_s) per flow bin ---
    dv_per_bin = np.diff(cumulative_volume_at_cin_tedges)
    dt_per_bin = np.diff(tedges_days)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_per_bin = np.where(dt_per_bin > 0, dv_per_bin / dt_per_bin, 0.0)
        # Solute-front velocity v_s = Q L / (R V_pore) -- the advection speed of the retarded ADE
        # that C_R actually solves. Kreft-Zuber requires the flux coefficient D_s/v_s = D_m/v_s +
        # alpha_L to use THAT velocity (not the fluid velocity Q L / V_pore); pairing it with the
        # moving-frame variance D_t = D_m tau + alpha_L xi is what conserves mass for R>1, D_m>0.
        v_per_bin = q_per_bin * streamline_len / r_vpv
        # (D_s/v_s) = D_m/v_s + alpha_L. At v_s=0 the bin has dV=0 and is skipped below; the
        # surrounding errstate suppresses the divide warning for those lanes.
        dl_over_v_per_bin = np.where(
            v_per_bin > 0,
            molecular_diffusivity / v_per_bin + longitudinal_dispersivity,
            0.0,
        )

    # --- Resolution-aware composite Gauss-Legendre quadrature, split by flow bins ---
    # The integration window per (cell, flow-bin) is the intersection of
    # [V_lo, V_hi] (cell), [ve_lo, ve_hi] (flow bin), and [V_j, infty) (parcel
    # entered). Within each sub-interval, t(V) is linear. Where the sub-interval is
    # much wider than the front width sqrt(4*D_t), the erf-like front is under-resolved
    # by a single 16-point GL rule; such sub-intervals are tiled with front-centred
    # panels (see _FRONT_OFFSETS). Smooth sub-intervals keep the single panel and are
    # bit-identical to the plain 16-point rule.
    idx_i, idx_j = np.nonzero(is_valid)
    if len(idx_i) == 0:
        return frac

    v_lo_cells = v_lo_arr[idx_i]
    v_hi_cells = v_hi_arr[idx_i]
    v_cin_cells = cumulative_volume_at_cin_tedges[idx_j]
    t_j_cells = tedges_days[idx_j]
    total_dv = v_hi_cells - v_lo_cells
    valid_cells = total_dv > 0

    # Post-injection lower bound is loop-invariant: max(V_lo, V_cin) (D2 hoist).
    v_lo_or_cin = np.maximum(v_lo_cells, v_cin_cells)

    integral_cf = np.zeros(len(idx_i))

    vol_edges = cumulative_volume_at_cin_tedges
    for k in range(len(vol_edges) - 1):
        ve_lo, ve_hi = vol_edges[k], vol_edges[k + 1]
        if v_per_bin[k] <= 0.0:
            continue
        dl_over_v_k = dl_over_v_per_bin[k]
        dt_sub_bin = tedges_days[k + 1] - tedges_days[k]
        dv_sub_edge = ve_hi - ve_lo

        # Intersection of cell, flow-bin, and post-injection range
        sub_lo = np.maximum(v_lo_or_cin, ve_lo)
        sub_hi = np.minimum(v_hi_cells, ve_hi)
        overlap = (sub_hi > sub_lo) & valid_cells
        if not np.any(overlap):
            continue

        lo = sub_lo[overlap]
        hi = sub_hi[overlap]
        vcin = v_cin_cells[overlap]
        tj = t_j_cells[overlap]

        # Front centre (x = 0 => xi = L => V = V_cin + r_vpv) and its width in volume.
        # front_width = sqrt(4*D_t_front) * r_vpv / L, with D_t_front = D_m*tau_front +
        # alpha_L*L (xi = L at the front). t(V) is linear within flow bin k.
        v_front = vcin + r_vpv
        t_front = tedges_days[k] + (v_front - ve_lo) * (dt_sub_bin / dv_sub_edge)
        tau_front = np.maximum(t_front - tj, 0.0)
        dt_front = molecular_diffusivity * tau_front + longitudinal_dispersivity * streamline_len
        front_width = np.sqrt(4.0 * dt_front) * r_vpv / streamline_len

        # Refine only where a sharp front region intersects this sub-interval and is
        # under-resolved by a single 16-point rule; elsewhere the integrand is flat
        # or already resolved, so one panel is exact. The front-region test (not just
        # "centre in this bin") also catches a front whose tail spills across a
        # flow-bin boundary. A spuriously large extrapolated front_width far from the
        # front fails ``underresolved`` and so cannot trigger refinement.
        front_hits = (v_front + _FRONT_REACH * front_width > lo) & (v_front - _FRONT_REACH * front_width < hi)
        underresolved = (hi - lo) > _REFINE_RATIO * front_width
        if np.any(front_hits & underresolved):
            inner = np.clip(
                v_front[:, np.newaxis] + front_width[:, np.newaxis] * _FRONT_OFFSETS[np.newaxis, :],
                lo[:, np.newaxis],
                hi[:, np.newaxis],
            )
            edges = np.concatenate([lo[:, np.newaxis], inner, hi[:, np.newaxis]], axis=1)
        else:
            edges = np.stack([lo, hi], axis=1)

        p_lo = edges[:, :-1]
        p_hi = edges[:, 1:]
        p_mid = 0.5 * (p_lo + p_hi)
        p_half = 0.5 * (p_hi - p_lo)

        # GL nodes over every panel: shape (n_cell, n_panel, n_gl)
        v_nodes = p_mid[:, :, np.newaxis] + p_half[:, :, np.newaxis] * _GL_NODES[np.newaxis, np.newaxis, :]

        # Geometry: x = xi - L = (V - V_j - r_vpv) * L / r_vpv (parcel position
        # relative to outlet); xi = parcel travel distance.
        x_nodes = (v_nodes - vcin[:, np.newaxis, np.newaxis] - r_vpv) * streamline_len / r_vpv
        xi_nodes = x_nodes + streamline_len

        t_nodes = tedges_days[k] + (v_nodes - ve_lo) * (dt_sub_bin / dv_sub_edge)
        # tau >= 0 by construction (lo >= v_cin); clip for safety.
        tau_nodes = np.maximum(t_nodes - tj[:, np.newaxis, np.newaxis], 0.0)

        # Bear's variance accumulator (sigma^2/2) — NO capping at RT/L
        dt_nodes = molecular_diffusivity * tau_nodes + longitudinal_dispersivity * xi_nodes

        with np.errstate(divide="ignore", invalid="ignore"):
            arg = x_nodes / (2.0 * np.sqrt(dt_nodes))
        # C_R = 0.5 * (1 + erf(arg)) = 0.5 * erfc((L-xi)/(2*sqrt(Dt)))
        erf_vals = np.where(np.isfinite(arg), special.erf(arg), np.sign(x_nodes))
        cr_vals = 0.5 * (1.0 + erf_vals)

        # K-Z flux correction: FC = (D_s/v_s) * (1/sqrt(4 pi D_t)) * exp(-arg^2)
        with np.errstate(divide="ignore", invalid="ignore"):
            gauss_vals = np.where(
                dt_nodes > 0.0,
                np.exp(-(arg**2)) / np.sqrt(4.0 * np.pi * dt_nodes),
                0.0,
            )
        cf_vals = cr_vals + dl_over_v_k * gauss_vals

        # Integrate: GL-weight over nodes, then sum panel contributions per cell.
        # The weight contraction is done in 2D so a single-panel (non-refined)
        # sub-interval is bit-identical to a plain 16-point rule.
        cf_weighted = (cf_vals.reshape(-1, cf_vals.shape[-1]) @ _GL_WEIGHTS).reshape(cf_vals.shape[:-1])
        integral_cf[overlap] += (p_half * cf_weighted).sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        frac_cells = np.where(valid_cells, integral_cf / total_dv, np.nan)
    frac[idx_i, idx_j] = frac_cells

    return frac


def _diffusion_extend_tedges_flag(spinup: object) -> bool:
    """Translate the public ``spinup`` parameter to the internal extend flag.

    The diffusion module's existing warm-start behavior is to extend
    ``tedges`` by 100 years on each side. The public ``spinup`` parameter
    maps onto this binary toggle: ``"constant"`` enables the extension
    (default; preserves legacy behavior), ``None`` disables it (cout in
    spin-up region becomes NaN). The float fraction-threshold mode of
    other modules is not implemented here.

    Returns
    -------
    bool
        True if tedges should be extended (warm-start), False if not.

    Raises
    ------
    ValueError
        If ``spinup`` is a string other than ``"constant"``.
    NotImplementedError
        If ``spinup`` is a float (fraction-threshold mode is not
        implemented for the diffusion module).
    """
    if spinup is None:
        return False
    if isinstance(spinup, str):
        if spinup != "constant":
            msg = f"spinup string must be 'constant'; got {spinup!r}"
            raise ValueError(msg)
        return True
    msg = (
        "diffusion's spinup parameter only supports None or 'constant'; "
        f"float thresholds are not yet implemented (got {spinup!r})"
    )
    raise NotImplementedError(msg)


def _validate_diffusion_inputs(
    *,
    tedges: pd.DatetimeIndex,
    flow: npt.NDArray[np.floating],
    aquifer_pore_volumes: npt.NDArray[np.floating],
    streamline_length: npt.NDArray[np.floating],
    molecular_diffusivity: npt.NDArray[np.floating],
    longitudinal_dispersivity: npt.NDArray[np.floating],
    retardation_factor: float,
    cin_values: npt.NDArray[np.floating] | None = None,
    cout_values: npt.NDArray[np.floating] | None = None,
    cout_tedges: pd.DatetimeIndex | None = None,
) -> None:
    """Validate inputs common to diffusion forward / reverse entry points.

    Path selection via mutually-exclusive kwargs:

    - ``cin_values`` provided => forward path. ``tedges`` parities cin and flow.
    - ``cout_values`` + ``cout_tedges`` provided => reverse path. ``tedges`` parities
      flow; ``cout_tedges`` parities cout.

    Raises
    ------
    ValueError
        If any check fails. The message identifies which invariant was violated.
    """
    n_pore_volumes = len(aquifer_pore_volumes)

    if cin_values is not None:
        _validate_tedges_parity(tedges, cin_values, tedges_name="tedges", values_name="cin")
        _validate_tedges_parity(tedges, flow, tedges_name="tedges", values_name="flow")
    elif cout_values is not None and cout_tedges is not None:
        _validate_tedges_parity(tedges, flow, tedges_name="tedges", values_name="flow")
        _validate_tedges_parity(cout_tedges, cout_values, tedges_name="cout_tedges", values_name="cout")
    else:
        msg = "must provide cin_values (forward) or both cout_values and cout_tedges (reverse)"
        raise ValueError(msg)
    if len(aquifer_pore_volumes) != len(streamline_length):
        msg = "aquifer_pore_volumes and streamline_length must have the same length"
        raise ValueError(msg)
    _validate_scalar_or_matching_length(
        molecular_diffusivity,
        name="molecular_diffusivity",
        expected_len=n_pore_volumes,
        ref_name="aquifer_pore_volumes",
    )
    _validate_scalar_or_matching_length(
        longitudinal_dispersivity,
        name="longitudinal_dispersivity",
        expected_len=n_pore_volumes,
        ref_name="aquifer_pore_volumes",
    )
    _validate_non_negative_array(molecular_diffusivity, name="molecular_diffusivity")
    _validate_non_negative_array(longitudinal_dispersivity, name="longitudinal_dispersivity")
    # Forward cin must be NaN-free; reverse cout may contain NaN (measurement
    # gaps) -- the inverse solver excludes those rows, matching deposition (#321).
    if cin_values is not None:
        _validate_no_nan(cin_values, name="cin")
    _validate_no_nan(flow, name="flow")
    _validate_non_negative_array(flow, name="flow", message="flow must be non-negative (negative flow not supported)")
    _validate_positive_array(aquifer_pore_volumes, name="aquifer_pore_volumes")
    _validate_positive_array(streamline_length, name="streamline_length")
    _validate_retardation_factor(retardation_factor)


def _prepare_diffusion_arrays(
    *,
    flow: npt.ArrayLike,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: npt.ArrayLike,
    molecular_diffusivity: npt.ArrayLike,
    longitudinal_dispersivity: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
]:
    """Coerce flow / geometry / dispersion inputs to broadcasted float arrays.

    Each per-streamtube parameter (``streamline_length``, ``molecular_diffusivity``,
    ``longitudinal_dispersivity``) may be passed as a scalar; it is broadcast to one
    value per pore volume. The returned arrays are read-only views when broadcast (none
    is mutated downstream).

    Returns
    -------
    tuple of ndarray
        ``(flow, aquifer_pore_volumes, streamline_length, molecular_diffusivity,
        longitudinal_dispersivity)`` as float arrays.
    """
    flow = np.asarray(flow, dtype=float)
    aquifer_pore_volumes = np.asarray(aquifer_pore_volumes, dtype=float)
    streamline_length = np.atleast_1d(np.asarray(streamline_length, dtype=float))
    molecular_diffusivity = np.atleast_1d(np.asarray(molecular_diffusivity, dtype=float))
    longitudinal_dispersivity = np.atleast_1d(np.asarray(longitudinal_dispersivity, dtype=float))

    n_pore_volumes = len(aquifer_pore_volumes)
    if streamline_length.size == 1:
        streamline_length = np.broadcast_to(streamline_length, (n_pore_volumes,))
    if molecular_diffusivity.size == 1:
        molecular_diffusivity = np.broadcast_to(molecular_diffusivity, (n_pore_volumes,))
    if longitudinal_dispersivity.size == 1:
        longitudinal_dispersivity = np.broadcast_to(longitudinal_dispersivity, (n_pore_volumes,))

    return flow, aquifer_pore_volumes, streamline_length, molecular_diffusivity, longitudinal_dispersivity


def _infiltration_to_extraction_coeff_matrix(
    *,
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    streamline_length: npt.NDArray[np.floating],
    molecular_diffusivity: npt.NDArray[np.floating],
    longitudinal_dispersivity: npt.NDArray[np.floating],
    retardation_factor: float,
    extend_tedges: bool = True,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
    """Build the forward coefficient matrix for diffusion transport.

    Constructs the matrix W such that ``cout = W @ cin``, accounting for
    advection, microdispersion, and molecular diffusion. NaN entries in the raw coefficient
    matrix are replaced with zero.

    Parameters
    ----------
    flow : ndarray
        Flow rate of water [m³/day]. Already validated.
    tedges : DatetimeIndex
        Cin/flow time edges (not yet extended for spin-up).
    cout_tedges : DatetimeIndex
        Cout time edges.
    aquifer_pore_volumes : ndarray
        Pore volumes [m³]. Already validated.
    streamline_length : ndarray
        Travel distances [m]. Already validated.
    molecular_diffusivity : ndarray
        Effective molecular diffusivities [m²/day]. Already broadcasted.
        See :func:`infiltration_to_extraction` for physical interpretation.
    longitudinal_dispersivity : ndarray
        Longitudinal dispersivities [m]. Already broadcasted.
    retardation_factor : float
        Retardation factor.

    Returns
    -------
    coeff_matrix : ndarray
        Filled coefficient matrix of shape (n_cout, n_cin). NaN replaced
        with zero.
    valid_cout_bins : ndarray
        Boolean mask of shape (n_cout,) indicating valid output bins.
    """
    if extend_tedges:
        # Extend tedges by 100 years on each side to provide warm-start cin
        # and post-data flow for cout bins near the boundaries. Equivalent to
        # the public ``spinup="constant"`` policy in other modules.
        tedges = pd.DatetimeIndex([
            tedges[0] - pd.Timedelta("36500D"),
            *list(tedges[1:-1]),
            tedges[-1] + pd.Timedelta("36500D"),
        ])

    # Compute the cumulative flow at tedges
    cumulative_volume_at_cin_tedges = cumulative_flow_volume(flow, dt_to_days(tedges))  # m³

    # Compute the cumulative flow at cout_tedges. Both edge arrays are first reduced to a shared
    # day axis: np.interp coerces each datetime64 operand to int64 in its own resolution, so a
    # cout_tedges / tedges unit mismatch (e.g. ns vs us) would send every query out of range and
    # silently return all-NaN.
    tedges_days_arr = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(cout_tedges, ref=tedges[0])
    cumulative_volume_at_cout_tedges = np.interp(
        cout_tedges_days, tedges_days_arr, cumulative_volume_at_cin_tedges
    ).astype(float)

    # Output bin valid where every streamtube's advective look-back is in-record across the whole
    # bin -- i.e. advective coverage == 1 for all pore volumes (NaN outside the record -> invalid).
    # This is the advective validity gate only; the dispersive informedness is the captured kernel
    # mass (total_coeff) applied downstream.
    valid_cout_bins = np.all(
        fraction_explained_full(
            flow=flow,
            tedges=tedges,
            cout_tedges=cout_tedges,
            aquifer_pore_volumes=aquifer_pore_volumes,
            retardation_factor=retardation_factor,
            direction="extraction_to_infiltration",
        )
        >= 1.0,
        axis=0,
    )

    # Initialize coefficient matrix accumulator
    n_cout_bins = len(cout_tedges) - 1
    n_cin_bins = len(flow)
    accumulated_coeff = np.zeros((n_cout_bins, n_cin_bins))

    # Loop over each pore volume
    for i_pv in range(len(aquifer_pore_volumes)):
        r_vpv = retardation_factor * aquifer_pore_volumes[i_pv]

        delta_volume = cumulative_volume_at_cout_tedges[:, None] - cumulative_volume_at_cin_tedges[None, :] - r_vpv

        step_widths = delta_volume / r_vpv * streamline_length[i_pv]

        frac = _cfrac_mean_volume(
            step_widths=step_widths,
            cumulative_volume_at_cout_tedges=cumulative_volume_at_cout_tedges,
            cumulative_volume_at_cin_tedges=cumulative_volume_at_cin_tedges,
            tedges_days=tedges_days_arr,
            molecular_diffusivity=float(molecular_diffusivity[i_pv]),
            longitudinal_dispersivity=float(longitudinal_dispersivity[i_pv]),
            r_vpv=r_vpv,
            streamline_len=streamline_length[i_pv],
        )

        accumulated_coeff += frac[:, :-1] - frac[:, 1:]

    coeff_matrix_filled = np.nan_to_num(accumulated_coeff / len(aquifer_pore_volumes), nan=0.0)

    return coeff_matrix_filled, valid_cout_bins


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: npt.ArrayLike,
    molecular_diffusivity: npt.ArrayLike,
    longitudinal_dispersivity: npt.ArrayLike,
    retardation_factor: float = 1.0,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute extracted concentration with advection, microdispersion, and molecular diffusion.

    This function models 1D solute transport through an aquifer system along orthogonal
    (Cartesian) flow paths. Each aquifer pore volume is an independent streamline carrying
    advection with microdispersion (alpha_L) and molecular diffusion (D_m); the spread across
    the pore volume distribution provides macrodispersion. Linear sorption enters via the
    retardation factor.

    The physical model assumes:

    1. Water infiltrates with concentration cin at time t_in
    2. Water travels distance L through aquifer with residence time tau = V_pore / Q
    3. During transport, microdispersion and molecular diffusion spread each streamline,
       while the spread across pore volumes provides macrodispersion
    4. At extraction, the concentration is a dispersed breakthrough curve

    The reported extracted concentration is the Kreft-Zuber (1978) **flux
    concentration** at the outlet, defined as the solute mass flux divided
    by the volumetric fluid flux. This is what is measured when sampling the
    outflowing fluid. Compared to Bear's leading-order resident concentration,
    it includes the dispersive boundary flux ``-D_s * dC_R/dx`` at
    ``x = L`` (with the solute-front dispersion ``D_s = D_m + alpha_L * v_s``
    and velocity ``v_s = Q L / (R V_pore)``), which is what makes the column-sum
    invariant ``integral Q c_out dt = integral Q c_in dt`` hold exactly under
    variable flow.

    Microdispersion and molecular diffusion enter as the moving-frame variance

        sigma^2(V) = 2 * D_m * tau(V) + 2 * alpha_L * xi(V),

    where ``tau(V)`` is the elapsed time since infiltration and ``xi(V)`` is
    the distance the parcel has actually travelled. Evaluating sigma^2 at each
    quadrature node — and avoiding any artificial capping past breakthrough —
    keeps Bear's formula an exact solution of the variable-coefficient ADE,
    which the Kreft-Zuber identity relies on.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water [concentration units].
        Length must match the number of time bins defined by tedges. The model assumes
        this value is constant over each interval ``[tedges[i], tedges[i+1])``.
    flow : array-like
        Flow rate of water in the aquifer [m³/day].
        Length must match cin and the number of time bins defined by tedges. The model
        assumes this value is constant over each interval ``[tedges[i], tedges[i+1])``.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for both cin and flow data. Has length of
        len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for output data bins. Has length of desired output + 1.
        The output concentration is averaged over each bin.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m³] representing the distribution
        of flow paths. Each pore volume determines the residence time for
        that flow path: tau = V_pore / Q.
    streamline_length : array-like
        Array of travel distances [m] corresponding to each pore volume.
        Must have the same length as aquifer_pore_volumes.
    molecular_diffusivity : float or array-like
        Effective (retarded-frame) molecular diffusivity [m²/day]. Can be a
        scalar (same for all pore volumes) or an array with the same length as
        aquifer_pore_volumes. Must be non-negative. For solute transport, this is
        the molecular diffusion coefficient D_m [m²/day] — typically ~1e-5 m²/day,
        negligible compared to microdispersion. For heat transport, pass the
        thermal diffusivity D_th = lambda / (rho*c)_eff [m²/day], typically
        0.01-0.1 m²/day.

        Internally, this contributes ``2 * molecular_diffusivity * tau`` to the
        variance, where ``tau`` is the elapsed time in days (no extra factor of
        R). The retardation factor instead enters the flux coefficient
        ``D_s/v_s = R D_m / v_fluid + alpha_L`` through the solute-front velocity
        ``v_s = Q L / (R V_pore)``. For heat transport, the thermal diffusivity
        already represents the effective diffusivity D_eff in the porous matrix;
        for solutes the contribution is typically negligible.
    longitudinal_dispersivity : float or array-like
        Longitudinal dispersivity [m]. Can be a scalar (same for all pore
        volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative. Represents microdispersion from pore-scale velocity variations.
        Set to 0 for pure molecular diffusion.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption.
    spinup : {'constant'} or None, optional
        Spin-up policy (default ``'constant'``). ``'constant'`` extends tedges by
        100 years on each side so that output bins near the boundary are always
        informed. ``None`` disables the extension; output bins without sufficient
        upstream data become NaN. Float fraction-threshold mode is not implemented
        and raises ``NotImplementedError``.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the extracted water. Same units as cin.
        Length equals len(cout_tedges) - 1. NaN values indicate time periods
        with no valid contributions from the infiltration data.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, if diffusivity is negative,
        or if aquifer_pore_volumes and streamline_length have different lengths.

    See Also
    --------
    extraction_to_infiltration : Inverse operation (deconvolution)
    gwtransport.advection.infiltration_to_extraction : Pure advection (no dispersion)
    gwtransport.diffusion_fast.infiltration_to_extraction : Fast closed-form equivalent
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The algorithm constructs a coefficient matrix W where cout = W @ cin:

    1. For each pore volume, build a cell grid in cumulative volume space:

       - cells span ``(V_cout[i], V_cout[i+1]) x V_cin[j]`` for each
         (cout-bin i, cin-edge j)
       - delta_volume = V_cout - V_cin - r_vpv encodes the parcel's offset
         from the outlet at each (cout-edge, cin-edge)

    2. For each cell, compute the bin-averaged Kreft-Zuber flux concentration
       ``frac[i, j] = (1/dV_i) * integral C_F(L, V; t_j) dV`` by resolution-aware
       composite Gauss-Legendre quadrature in volume space, split at flow-bin
       boundaries so that ``t(V)`` is linear within each sub-interval. Where the
       erf-like front (width ``sqrt(4*D_t)`` in volume units) is under-resolved by
       a single 16-point rule -- as for near-zero dispersivity -- the sub-interval
       is tiled with front-centred panels; smooth sub-intervals keep the single
       rule. The moving-frame variance ``D_t = D_m*tau + alpha_L*xi`` is evaluated
       at each quadrature node (never capped at the residence time).

    3. Coefficient for bin: ``coeff[i,j] = frac[i, j] - frac[i, j+1]``. This
       is the contribution of cin[j] to cout[i] in the W matrix.

    4. Average coefficients across all pore volumes.

    The K-Z flux-correction term in C_F = C_R - (D_s/v_s) * dC_R/dx (solute-front
    velocity v_s = Q L / (R V_pore), dispersion D_s = D_m + alpha_L * v_s) is what
    makes the column-sum invariant exact under variable Q; see the module
    docstring for the derivation.

    Examples
    --------
    Basic usage with constant flow:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion import infiltration_to_extraction
    >>>
    >>> # Create time edges
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>>
    >>> # Input concentration (step function) and constant flow
    >>> cin = np.zeros(len(tedges) - 1)
    >>> cin[5:10] = 1.0  # Pulse of concentration
    >>> flow = np.ones(len(tedges) - 1) * 100.0  # 100 m³/day
    >>>
    >>> # Single pore volume of 500 m³, travel distance 100 m
    >>> aquifer_pore_volumes = np.array([500.0])
    >>> streamline_length = np.array([100.0])
    >>>
    >>> # Compute with dispersion (molecular diffusion + dispersivity)
    >>> # Scalar values broadcast to all pore volumes
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     molecular_diffusivity=1e-4,  # m²/day, same for all pore volumes
    ...     longitudinal_dispersivity=1.0,  # m, same for all pore volumes
    ... )

    With multiple pore volumes (heterogeneous aquifer):

    >>> # Distribution of pore volumes and corresponding travel distances
    >>> aquifer_pore_volumes = np.array([400.0, 500.0, 600.0])
    >>> streamline_length = np.array([80.0, 100.0, 120.0])
    >>>
    >>> # Scalar diffusion parameters broadcast to all pore volumes
    >>> cout = infiltration_to_extraction(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     molecular_diffusivity=1e-4,  # m²/day
    ...     longitudinal_dispersivity=1.0,  # m
    ... )
    """
    cout_tedges = pd.DatetimeIndex(cout_tedges)
    tedges = pd.DatetimeIndex(tedges)

    cin = np.asarray(cin, dtype=float)
    flow, aquifer_pore_volumes, streamline_length, molecular_diffusivity, longitudinal_dispersivity = (
        _prepare_diffusion_arrays(
            flow=flow,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=molecular_diffusivity,
            longitudinal_dispersivity=longitudinal_dispersivity,
        )
    )

    _validate_diffusion_inputs(
        tedges=tedges,
        flow=flow,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        cin_values=cin,
    )

    extend_tedges = _diffusion_extend_tedges_flag(spinup)
    coeff_matrix, valid_cout_bins = _infiltration_to_extraction_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        extend_tedges=extend_tedges,
    )

    cout = coeff_matrix @ cin

    # Output bins are invalid where the coefficient sum is near zero (no cin has broken
    # through yet) or the bin extends beyond the input data range (valid_cout_bins).
    total_coeff = np.sum(coeff_matrix, axis=1)
    no_valid_contribution = (total_coeff < EPSILON_COEFF_SUM) | ~valid_cout_bins
    cout[no_valid_contribution] = np.nan

    return cout


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,
    streamline_length: npt.ArrayLike,
    molecular_diffusivity: npt.ArrayLike,
    longitudinal_dispersivity: npt.ArrayLike,
    retardation_factor: float = 1.0,
    regularization_strength: float = 1e-10,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute infiltration concentration from extracted water (deconvolution with dispersion).

    Inverts the forward transport model by building the forward coefficient
    matrix ``W_forward`` from :func:`infiltration_to_extraction` and solving
    ``W_forward @ cin = cout`` via Tikhonov regularization. Well-determined
    modes are dominated by the data; poorly-determined modes are pulled
    toward the physically motivated target (transpose-and-normalize of the
    forward matrix).

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water [concentration units].
        Length must match the number of time bins defined by cout_tedges.
    flow : array-like
        Flow rate of water in the aquifer [m³/day].
        Length must match the number of time bins defined by tedges.
    tedges : pandas.DatetimeIndex
        Time edges defining bins for cin (output) and flow data.
        Has length of len(flow) + 1. Output cin has length len(tedges) - 1.
    cout_tedges : pandas.DatetimeIndex
        Time edges for cout data bins. Has length of len(cout) + 1.
        Can have different time alignment and resolution than tedges.
    aquifer_pore_volumes : array-like
        Array of aquifer pore volumes [m³] representing the distribution
        of flow paths. Each pore volume determines the residence time for
        that flow path: tau = V_pore / Q.
    streamline_length : array-like
        Array of travel distances [m] corresponding to each pore volume.
        Must have the same length as aquifer_pore_volumes.
    molecular_diffusivity : float or array-like
        Effective molecular diffusivity [m²/day]. Can be a scalar (same for all
        pore volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative. See :func:`infiltration_to_extraction` for
        details on the physical interpretation and the interaction with
        retardation_factor.
    longitudinal_dispersivity : float or array-like
        Longitudinal dispersivity [m]. Can be a scalar (same for all pore
        volumes) or an array with the same length as aquifer_pore_volumes.
        Must be non-negative.
    retardation_factor : float, optional
        Retardation factor of the compound in the aquifer (default 1.0).
        Values > 1.0 indicate slower transport due to sorption.
    regularization_strength : float, optional
        Tikhonov regularization parameter λ. See
        :func:`gwtransport.advection.extraction_to_infiltration` for details.
        Default is 1e-10.
    spinup : {'constant'} or None, optional
        Spin-up policy (default ``'constant'``). ``'constant'`` extends tedges by
        100 years on each side so that output bins near the boundary are always
        informed. ``None`` disables the extension; output bins without sufficient
        upstream data become NaN. Float fraction-threshold mode is not implemented
        and raises ``NotImplementedError``.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Same units as cout.
        Length equals len(tedges) - 1. NaN values indicate time periods
        with no valid contributions from the extraction data.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, if diffusivity is negative,
        or if aquifer_pore_volumes and streamline_length have different lengths.

    See Also
    --------
    infiltration_to_extraction : Forward operation (convolution)
    gwtransport.advection.extraction_to_infiltration : Pure advection (no dispersion)
    :ref:`concept-dispersion-scales` : Macrodispersion vs microdispersion

    Notes
    -----
    The algorithm builds the forward coefficient matrix ``W_forward`` (same as
    used by :func:`infiltration_to_extraction`) and solves ``W_forward @ cin = cout``
    using :func:`gwtransport.utils.solve_tikhonov`. This ensures mathematical
    consistency between forward and inverse operations.

    NaN values in ``cout`` mark measurement gaps (e.g. sparse lab samples).
    Their rows are excluded from the Tikhonov solve, matching
    :func:`gwtransport.deposition.extraction_to_deposition`; cin bins
    constrained only by gapped ``cout`` bins are returned as NaN.

    Examples
    --------
    Basic usage with constant flow:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from gwtransport.diffusion import extraction_to_infiltration
    >>>
    >>> # Create time edges: tedges for cin/flow, cout_tedges for cout
    >>> tedges = pd.date_range(start="2020-01-01", end="2020-01-20", freq="D")
    >>> cout_tedges = pd.date_range(start="2020-01-05", end="2020-01-25", freq="D")
    >>>
    >>> # Extracted concentration and constant flow
    >>> cout = np.zeros(len(cout_tedges) - 1)
    >>> cout[5:10] = 1.0  # Observed pulse at extraction
    >>> flow = np.ones(len(tedges) - 1) * 100.0  # 100 m³/day
    >>>
    >>> # Single pore volume of 500 m³, travel distance 100 m
    >>> aquifer_pore_volumes = np.array([500.0])
    >>> streamline_length = np.array([100.0])
    >>>
    >>> # Reconstruct infiltration concentration
    >>> cin = extraction_to_infiltration(
    ...     cout=cout,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volumes=aquifer_pore_volumes,
    ...     streamline_length=streamline_length,
    ...     molecular_diffusivity=1e-4,
    ...     longitudinal_dispersivity=1.0,
    ... )
    """
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    cout = np.asarray(cout, dtype=float)
    flow, aquifer_pore_volumes, streamline_length, molecular_diffusivity, longitudinal_dispersivity = (
        _prepare_diffusion_arrays(
            flow=flow,
            aquifer_pore_volumes=aquifer_pore_volumes,
            streamline_length=streamline_length,
            molecular_diffusivity=molecular_diffusivity,
            longitudinal_dispersivity=longitudinal_dispersivity,
        )
    )

    _validate_diffusion_inputs(
        tedges=tedges,
        flow=flow,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        cout_values=cout,
        cout_tedges=cout_tedges,
    )

    n_cin = len(tedges) - 1

    # Build forward weight matrix: W_forward @ cin = cout
    extend_tedges = _diffusion_extend_tedges_flag(spinup)
    w_forward, valid_cout_bins = _infiltration_to_extraction_coeff_matrix(
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=aquifer_pore_volumes,
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        extend_tedges=extend_tedges,
    )

    return solve_inverse_transport(
        w_forward=w_forward,
        observed=cout,
        n_output=n_cin,
        regularization_strength=regularization_strength,
        valid_rows=valid_cout_bins,
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
    streamline_length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute extracted concentration with advection and dispersion for gamma-distributed pore volumes.

    Combines advection with microdispersion and molecular diffusion along each streamline
    (gamma-distributed pore volumes, whose spread provides macrodispersion). This is a
    convenience wrapper around :func:`infiltration_to_extraction` that parameterizes
    the aquifer pore volume distribution as a (shifted) gamma distribution.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cin : array-like
        Concentration of the compound in infiltrating water.
    flow : array-like
        Flow rate of water in the aquifer [m³/day].
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
    streamline_length : float
        Travel distance through the aquifer [m]. Applied uniformly to all
        gamma-discretized pore volumes.
    molecular_diffusivity : float
        Effective molecular diffusivity [m²/day]. Must be non-negative.
        See :func:`infiltration_to_extraction` for details on the interaction
        with retardation_factor.
    longitudinal_dispersivity : float
        Longitudinal dispersivity [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    spinup : {'constant'} or None, optional
        Spin-up policy (default ``'constant'``). ``'constant'`` extends tedges by
        100 years on each side so that output bins near the boundary are always
        informed. ``None`` disables the extension; output bins without sufficient
        upstream data become NaN. Float fraction-threshold mode is not implemented
        and raises ``NotImplementedError``.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the extracted water. Length equals
        len(cout_tedges) - 1. NaN values indicate time periods with no valid
        contributions from the infiltration data.

    See Also
    --------
    infiltration_to_extraction : Transport with explicit pore volume distribution
    gamma_extraction_to_infiltration : Reverse operation (deconvolution)
    gwtransport.gamma.bins : Create gamma distribution bins
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
    >>> from gwtransport.diffusion import gamma_infiltration_to_extraction
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
    ...     streamline_length=100.0,
    ...     molecular_diffusivity=1e-4,
    ...     longitudinal_dispersivity=1.0,
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
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
    streamline_length: float,
    molecular_diffusivity: float,
    longitudinal_dispersivity: float,
    retardation_factor: float = 1.0,
    regularization_strength: float = 1e-10,
    spinup: str | None = "constant",
) -> npt.NDArray[np.floating]:
    """
    Compute infiltration concentration from extracted water for gamma-distributed pore volumes.

    Inverts the forward transport model (advection + dispersion with gamma-distributed
    pore volumes) via Tikhonov regularization. This is a convenience wrapper around
    :func:`extraction_to_infiltration` that parameterizes the aquifer pore volume
    distribution as a (shifted) gamma distribution.

    Provide either (mean, std) or (alpha, beta); ``loc`` is optional and defaults to 0.

    Parameters
    ----------
    cout : array-like
        Concentration of the compound in extracted water.
    flow : array-like
        Flow rate of water in the aquifer [m³/day].
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
    streamline_length : float
        Travel distance through the aquifer [m]. Applied uniformly to all
        gamma-discretized pore volumes.
    molecular_diffusivity : float
        Effective molecular diffusivity [m²/day]. Must be non-negative.
        See :func:`infiltration_to_extraction` for details on the interaction
        with retardation_factor.
    longitudinal_dispersivity : float
        Longitudinal dispersivity [m]. Must be non-negative.
    retardation_factor : float, optional
        Retardation factor (default 1.0). Values > 1.0 indicate slower transport.
    regularization_strength : float, optional
        Tikhonov regularization parameter. Default is 1e-10.
    spinup : {'constant'} or None, optional
        Spin-up policy (default ``'constant'``). ``'constant'`` extends tedges by
        100 years on each side so that output bins near the boundary are always
        informed. ``None`` disables the extension; output bins without sufficient
        upstream data become NaN. Float fraction-threshold mode is not implemented
        and raises ``NotImplementedError``.

    Returns
    -------
    numpy.ndarray
        Bin-averaged concentration in the infiltrating water. Length equals
        len(tedges) - 1. NaN values indicate time periods with no valid
        contributions from the extraction data.

    See Also
    --------
    extraction_to_infiltration : Deconvolution with explicit pore volume distribution
    gamma_infiltration_to_extraction : Forward operation (convolution)
    gwtransport.gamma.bins : Create gamma distribution bins
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
    >>> from gwtransport.diffusion import gamma_extraction_to_infiltration
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
    ...     streamline_length=100.0,
    ...     molecular_diffusivity=1e-4,
    ...     longitudinal_dispersivity=1.0,
    ... )
    """
    bins = gamma.bins(mean=mean, std=std, loc=loc, alpha=alpha, beta=beta, n_bins=n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        aquifer_pore_volumes=bins["expected_values"],
        streamline_length=streamline_length,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        retardation_factor=retardation_factor,
        regularization_strength=regularization_strength,
        spinup=spinup,
    )
