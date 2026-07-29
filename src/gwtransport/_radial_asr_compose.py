r"""Flux-resident step response of a constant-Q radial phase in the flushed-volume clock.

The single injection/readout primitive the composition engines are built from
(:mod:`gwtransport._radial_asr_reuse`): ``G1(S; V') = L^{-1}[ghat_FR(p; V')/p](S)``, the flux-resident
step response in flushed volume (FR mode: flux injection at the well, resident detection at volume
``V'``). It carries both directions of the transport:

1. **Injection -> resident profile.** A piecewise-constant injected deviation ``cin'`` (concentration
   minus background) over injection volume bins ``[sigma_j, sigma_{j+1}]`` leaves, after flushing the
   total injected volume ``S_inj``, the resident profile

   ``f(V') = sum_j cin'_j [G1(S_inj - sigma_j; V') - G1(S_inj - sigma_{j+1}; V')]``.

2. **Extraction -> arrival.** Each resident parcel at ``V'`` returns to the well with the duality
   arrival kernel whose flushed-extraction-volume Laplace transform is the same ``ghat_FR(p; V')``
   (``|Q| h_bar = ghat_FR``). The flow-weighted average over an output (extraction) volume bin
   ``[T_i, T_{i+1}]`` is therefore ``[G1(T_{i+1}; V') - G1(T_i; V')]/(T_{i+1}-T_i)``.

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

    For ``D_m = 0`` the flushed-volume Airy kernel depends on the Laplace variable ``p`` only through
    ``beta = 2 c_geo R p / alpha_L``, so it is evaluated directly in the flow-free canonical form
    (``s = 2 c_geo p``, ``A_0 = 1``) -- exact for arbitrary within-phase variable flow and bit-independent of
    ``flow_scale``. For ``D_m > 0`` the kernel depends on ``A_0 = flow_scale / (2 c_geo)`` separately, so the
    Laplace variable enters as ``s = flow_scale * p`` and ``flow_scale`` must be the (constant) phase flow
    magnitude.

    The de Hoog half-period is anchored to the FR arrival-volume mean at ``V'``
    (``mu = R c_geo[(r'+alpha_L)^2 + alpha_L^2 - r_w^2]`` -- the breakthrough front), bounded below by
    the requested corner volumes, so the front is resolved even when the output extends far past it.
    Corners ``<= 0`` map to ``0`` (no breakthrough yet).

    Returns
    -------
    ndarray
        ``G1(S; V')`` for each ``S`` in ``corner_volumes`` (same shape).
    """
    r_p = np.sqrt(r_w**2 + v_prime / c_geo)
    mu = retardation_factor * c_geo * ((r_p + alpha_l) ** 2 + alpha_l**2 - r_w**2)
    # D_m = 0: the Airy S-clock kernel depends on p only through beta = 2 c_geo R p / alpha_L, so evaluate it
    # in the flow-free canonical form (a0 = 1, s = 2 c_geo p) -- routing flow_scale through s and a0 would
    # round-trip it and leave ~1-ulp bit-noise the de Hoog QD stage amplifies. D_m > 0: the kernel depends on
    # A_0 = flow_scale/(2 c_geo) separately, so use the (constant) phase flow magnitude.
    s_mult = 2.0 * c_geo if molecular_diffusivity == 0.0 else flow_scale
    a0 = s_mult / (2.0 * c_geo)  # 1.0 exactly for D_m = 0 (flow-free), flow_scale/(2 c_geo) for D_m > 0

    def f_hat(p: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
        return (
            transfer_function(
                s=s_mult * p,
                r=r_p,
                r_w=r_w,
                alpha_l=alpha_l,
                a0=a0,
                d_m=molecular_diffusivity,
                retardation_factor=retardation_factor,
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
