r"""Composition of constant-Q radial phases into the well observable (single cycle).

For a single inject-then-extract cycle at one well (no intermediate flow reversal) the extracted
flux concentration is built grid-free from the per-phase kernels of :mod:`gwtransport._radial_asr_kernels`
(the KB Sec. 10a pipeline), with everything carried in the flushed-volume clock so that arbitrary
within-phase variable flow is exact for ``D_m = 0`` (the S-clock convolution theorem):

1. **Injection -> resident profile.** A piecewise-constant injected deviation ``cin'`` (concentration
   minus background) over injection volume bins ``[sigma_j, sigma_{j+1}]`` leaves, after flushing the
   total injected volume ``S_inj``, the resident profile

   ``f(V') = sum_j cin'_j [G1(S_inj - sigma_j; V') - G1(S_inj - sigma_{j+1}; V')]``,

   where ``G1(S; V') = L^{-1}[ghat_FR(p; V')/p](S)`` is the flux-resident step response in flushed
   volume (FR mode: flux injection at the well, resident detection at volume ``V'``).

2. **Extraction -> echo.** Each resident parcel at ``V'`` returns to the well with the duality arrival
   kernel whose flushed-extraction-volume Laplace transform is the same ``ghat_FR(p; V')`` (KB Sec. 7,
   ``|Q| h_bar = ghat_FR``). The flow-weighted average over an output (extraction) volume bin
   ``[T_i, T_{i+1}]`` is therefore ``[G1(T_{i+1}; V') - G1(T_i; V')]/(T_{i+1}-T_i)``, and superposing
   over the resident profile gives the bin-averaged echo. The whole map is the weight matrix

   ``W_{ij} = int_0^inf [G1(S_inj-sigma_j;V') - G1(S_inj-sigma_{j+1};V')]
              * [G1(T_{i+1};V') - G1(T_i;V')]/(T_{i+1}-T_i) dV'``,   ``cout'_i = sum_j W_{ij} cin'_j``.

The flushed-volume FR transfer function is the autonomous form: ``ghat`` depends on the Laplace
variable only through ``beta = s/(alpha_L A_0)``, and the S-clock substitution makes
``beta = 2 c_geo R p / alpha_L`` (``c_geo = pi b n``, ``p`` conjugate to flushed volume), independent
of the flow magnitude. So the kernel is evaluated as ``transfer_function(s = 2 c_geo p, a0 = 1, R)``.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt

from gwtransport._radial_asr_dehoog import dehoog_inverse
from gwtransport._radial_asr_kernels import transfer_function

# Default de Hoog series length and front-anchored scaling margin for the radial-ASR inversions: the FR
# step response (here) and the field propagators (_radial_asr_reuse) both import these so the two never
# silently desync on de Hoog resolution.
_DEHOOG_TERMS = 44
_SCALE_MARGIN = 1.3


def _fr_step_response(
    v_prime: float,
    corner_volumes: npt.NDArray[np.floating],
    *,
    c_geo: float,
    r_w: float,
    alpha_l: float,
    retardation_factor: float,
    flow_scale: float,
    molecular_diffusivity: float,
) -> npt.NDArray[np.floating]:
    r"""Flux-resident step response ``G1(S; V') = L^{-1}[ghat_FR(p; V')/p](S)`` at one ``V'``.

    The flushed-volume Laplace variable ``p`` enters the transfer function as ``s = flow_scale * p``
    with ``A_0 = flow_scale / (2 c_geo)``. For ``D_m = 0`` the flow magnitude cancels (``ghat``
    depends only on ``beta = 2 c_geo p / alpha_L``), so the response is the flow-independent S-clock
    kernel -- exact for arbitrary within-phase variable flow. For ``D_m > 0`` the kernel depends on
    ``A_0`` separately, so ``flow_scale`` must be the (constant) phase flow magnitude.

    The de Hoog half-period is anchored to the FR arrival-volume mean at ``V'``
    (``mu = R c_geo[(r'+alpha_L)^2 + alpha_L^2 - r_w^2]``, KB Sec. 7 -- the breakthrough front),
    bounded below by the requested corner volumes, so the front is resolved even when the output
    extends far past it. Corners ``<= 0`` map to ``0`` (no breakthrough yet).

    Returns
    -------
    ndarray
        ``G1(S; V')`` for each ``S`` in ``corner_volumes`` (same shape).
    """
    r_p = np.sqrt(r_w**2 + v_prime / c_geo)
    mu = retardation_factor * c_geo * ((r_p + alpha_l) ** 2 + alpha_l**2 - r_w**2)

    def f_hat(p: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        return (
            transfer_function(
                s=flow_scale * p,
                r=r_p,
                r_w=r_w,
                alpha_l=alpha_l,
                a0=flow_scale / (2.0 * c_geo),
                d_m=molecular_diffusivity,
                retardation_factor=retardation_factor,
                inject="flux",
                detect="resident",
            )
            / p
        )

    cv = np.asarray(corner_volumes, dtype=float)
    out = np.zeros_like(cv)
    positive = cv > 0.0
    if np.any(positive):
        scaling = _SCALE_MARGIN * max(mu, float(cv[positive].max()))
        out[positive] = dehoog_inverse(f_hat=f_hat, t=cv[positive], n_terms=_DEHOOG_TERMS, scaling=scaling)
    return out


def single_cycle_echo_matrix(
    *,
    inj_volume_edges: npt.NDArray[np.floating],
    ext_volume_edges: npt.NDArray[np.floating],
    c_geo: float,
    r_w: float,
    alpha_l: float,
    inj_flow_scale: float,
    ext_flow_scale: float,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 0.0,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    r"""Echo weight matrix ``W`` (``cout' = W @ cin'``) for one inject-then-extract cycle, one disk.

    The injection builds the resident profile with the FR kernel at the injection flow magnitude and
    the extraction reads it out at the extraction flow magnitude. For ``D_m = 0`` the flow magnitudes
    cancel (the S-clock kernel is flow-independent), so arbitrary within-phase variable flow is exact;
    for ``D_m > 0`` the flow scales must be the (constant) per-phase magnitudes.

    Parameters
    ----------
    inj_volume_edges : ndarray
        Cumulative flushed-volume edges of the injection bins (length ``n_inj + 1``), increasing.
    ext_volume_edges : ndarray
        Cumulative flushed-volume edges of the extraction (output) bins (length ``n_ext + 1``),
        increasing, measured from the start of extraction.
    c_geo : float
        Geometry constant ``pi b n`` so that ``V(r) = c_geo (r^2 - r_w^2)`` (m^3 per m^2).
    r_w : float
        Well radius (m).
    alpha_l : float
        Longitudinal dispersivity (m).
    inj_flow_scale, ext_flow_scale : float
        Flow magnitudes ``|Q|`` (m^3/day) of the injection and extraction phases (only relevant when
        ``molecular_diffusivity > 0``; ignored to within rounding for ``D_m = 0``).
    retardation_factor : float, optional
        Linear retardation ``R >= 1``. Default 1.
    molecular_diffusivity : float, optional
        Molecular diffusivity (m^2/day). Default 0 (Airy S-clock; exact for variable flow).
    n_quad : int, optional
        Number of Gauss-Legendre nodes for the ``V'`` superposition integral. Default 240.

    Returns
    -------
    ndarray, shape (n_ext, n_inj)
        ``W`` with ``cout'_i = sum_j W_{ij} cin'_j`` (deviation from background).
    """
    sigma = np.asarray(inj_volume_edges, dtype=float) - inj_volume_edges[0]  # 0 .. S_inj
    s_inj = sigma[-1]
    t_edges = np.asarray(ext_volume_edges, dtype=float) - ext_volume_edges[0]  # 0 .. T_end
    dt = np.diff(t_edges)

    # V' quadrature window: the resident profile f(V') = G1(S_inj; V') has its front at the retarded solute
    # radius r_front (where the FR arrival mean equals S_inj) and a dispersive tail of breakthrough width
    # ~sqrt(alpha_L r_front). A flat 4 S_inj/R margin is Peclet-blind and truncates that tail at low Peclet
    # (r_front/alpha_L << 1), losing mass (int f dV' < S_inj/R). Mirror the reuse engine's 12-sigma advective
    # reach plus a 6-sigma molecular reach over the cycle duration -- the same grid that conserves the mass
    # to ~1e-5. D_m dispatch is inside transfer_function (Airy for D_m = 0, Riccati log-derivative for D_m > 0).
    r_front = np.sqrt(r_w**2 + s_inj / (retardation_factor * c_geo))
    total_time = s_inj / inj_flow_scale + t_edges[-1] / ext_flow_scale
    r_max = r_front + 12.0 * np.sqrt(alpha_l * r_front + alpha_l**2) + 6.0 * np.sqrt(molecular_diffusivity * total_time)
    v_max = c_geo * (r_max**2 - r_w**2)
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    v_nodes = 0.5 * v_max * (nodes + 1.0)
    v_weights = 0.5 * v_max * weights

    inj_corners = s_inj - sigma  # G1 evaluated at S_inj - sigma_j (descending, last is 0)
    w = np.zeros((len(dt), len(sigma) - 1))
    for v, vw in zip(v_nodes, v_weights, strict=True):
        g_inj = _fr_step_response(
            v,
            inj_corners,
            c_geo=c_geo,
            r_w=r_w,
            alpha_l=alpha_l,
            retardation_factor=retardation_factor,
            flow_scale=inj_flow_scale,
            molecular_diffusivity=molecular_diffusivity,
        )
        g_ext = _fr_step_response(
            v,
            t_edges,
            c_geo=c_geo,
            r_w=r_w,
            alpha_l=alpha_l,
            retardation_factor=retardation_factor,
            flow_scale=ext_flow_scale,
            molecular_diffusivity=molecular_diffusivity,
        )
        f_contrib = g_inj[:-1] - g_inj[1:]  # length n_inj: resident-profile contribution of each cin bin
        ext_avg = (g_ext[1:] - g_ext[:-1]) / dt  # length n_ext: bin-averaged arrival per output bin
        w += vw * np.outer(ext_avg, f_contrib)
    # Retardation amplitude: the mobile profile integrates to S_inj/R (sorbed mass is immobile) and the
    # arrival kernel's CDF plateaus at 1, so the bare readout under-recovers by 1/R. Each extracted
    # mobile parcel mobilizes its sorbed companion (total solute = R x mobile), so scale the readout by R.
    return retardation_factor * w
