r"""Vectorized double-precision de Hoog numerical Laplace inversion.

This private module provides :func:`dehoog_inverse`, an implementation of the de Hoog, Knight &
Stokes (1982) accelerated Fourier-series method for inverting a Laplace transform
:math:`\bar f(s) \to f(t)`. It is the foundational numerical primitive of the exact radial
advection-dispersion module: the per-phase transfer functions (:mod:`gwtransport._radial_asr_kernels`)
are known in closed form only in the Laplace domain, and the bin-level observable needs the
real-time (here: real-flushed-volume) kernel and its antiderivatives, obtained by inverting
:math:`\hat g/s` and :math:`\hat g/s^2`.

Why de Hoog and not Talbot
--------------------------
The radial ASR observables include step-like resident profiles and sharp breakthroughs. The Talbot
(deformed-contour) family is built for smooth, monotone-decaying responses and produces large
spurious oscillations -- even negative concentrations -- on these step-like inputs. The de Hoog
quotient-difference / Pade acceleration of the Bromwich-Fourier series handles them reliably. This
is verified in the test suite against ``mpmath.invertlaplace`` on both smooth and step-like
transforms; Talbot is intentionally not offered.

Algorithm
---------
For a real evaluation time ``t`` the inverse is the Bromwich integral, discretized as the
Fourier series (Crump, 1976)

.. math::

    f(t) \approx \frac{e^{\gamma t}}{T}\Big[\tfrac12 \bar f(\gamma)
        + \sum_{k=1}^{\infty}\operatorname{Re}\big\{\bar f(\gamma + i k\pi/T)\,e^{i k\pi t/T}\big\}\Big],

with abscissa :math:`\gamma = \alpha - \ln(\text{tol})/(2T)` placed to the right of every
singularity of :math:`\bar f` (``alpha`` = the largest real part of any singularity; ``0`` for the
decaying radial kernels, whose poles lie on the non-positive real axis). de Hoog et al. accelerate
the truncated power series :math:`\sum_k a_k z^k`, :math:`z=e^{i\pi t/T}`,
:math:`a_k=\bar f(\gamma+ik\pi/T)` (with :math:`a_0` halved), by its continued-fraction (Pade)
representation whose coefficients come from the quotient-difference (QD) recurrence, plus a tail
remainder estimate for the last convergent. The continued fraction is evaluated by the standard
three-term recurrence.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def dehoog_inverse(
    *,
    f_hat: Callable[[npt.NDArray[np.complexfloating]], npt.NDArray[np.complexfloating]],
    t: npt.ArrayLike,
    n_terms: int = 24,
    scaling: float | None = None,
    alpha: float = 0.0,
    tol: float = 1e-9,
) -> npt.NDArray[np.floating]:
    r"""Invert a single Laplace transform at an array of times via the de Hoog method.

    The transform ``f_hat`` is sampled once on ``2 * n_terms + 1`` complex Bromwich nodes; the
    continued-fraction acceleration and its evaluation are vectorized over ``t``. Only ``t > 0`` is
    meaningful; ``t <= 0`` returns ``0`` (the inverse of a one-sided transform vanishes for negative
    time, and the series is undefined at ``t = 0``).

    Parameters
    ----------
    f_hat : callable
        Vectorized Laplace transform :math:`\bar f(s)`. Receives a complex ``ndarray`` of shape
        ``(2 * n_terms + 1,)`` and must return a complex ``ndarray`` of the same shape.
    t : array-like
        Evaluation times (same clock as the inverse, e.g. flushed volume). 1-D or scalar.
    n_terms : int, optional
        Number of acceleration terms ``M``; the transform is evaluated at ``2*M + 1`` nodes.
        Default 24.
    scaling : float, optional
        The half-period ``T`` of the Bromwich-Fourier approximation. The result is accurate for
        ``0 < t < 2T``, best for ``t <~ T``. Defaults to ``2 * max(t)`` (so every requested time
        sits in the well-resolved first half).
    alpha : float, optional
        Largest real part of any singularity of ``f_hat`` (the Bromwich abscissa is placed to its
        right). Default 0 (correct for the decaying radial kernels and for ``f_hat`` with poles only
        at ``s <= 0``, including the ``1/s`` and ``1/s^2`` antiderivative factors).
    tol : float, optional
        Target relative accuracy controlling the abscissa offset. Default ``1e-9``.

    Returns
    -------
    ndarray
        Real inverse ``f(t)``, same shape as ``t`` (0-d input yields a 0-d array).

    Raises
    ------
    ValueError
        If ``f_hat`` does not return an array of shape ``(2 * n_terms + 1,)``.

    Notes
    -----
    The QD recurrence is run on 1-D arrays of decreasing length (the standard rhombus rules of de
    Hoog, Knight & Stokes 1982); the continued fraction is then evaluated by the three-term
    recurrence ``A_n = A_{n-1} + d_n z A_{n-2}``, ``B_n = B_{n-1} + d_n z B_{n-2}`` with a final
    remainder term that sums the truncated tail.

    References
    ----------
    de Hoog, F. R., Knight, J. H., & Stokes, A. N. (1982). An improved method for numerical
    inversion of Laplace transforms. SIAM Journal on Scientific and Statistical Computing, 3(3),
    357-366.
    """
    t_arr = np.asarray(t, dtype=float)
    scalar_input = t_arr.ndim == 0
    t_flat = np.atleast_1d(t_arr)

    m = int(n_terms)
    n_nodes = 2 * m + 1
    t_max = float(np.max(t_flat)) if t_flat.size else 1.0
    if t_max <= 0.0:
        # No positive evaluation times: the one-sided inverse is identically zero there.
        out = np.zeros_like(t_flat)
        return out.reshape(t_arr.shape) if not scalar_input else out[0]
    big_t = float(scaling) if scaling is not None else 2.0 * t_max
    gamma = alpha - np.log(tol) / (2.0 * big_t)

    # Sample the transform on the Bromwich nodes s_k = gamma + i k pi / T, k = 0 .. 2M.
    k = np.arange(n_nodes)
    s = gamma + 1j * k * np.pi / big_t
    a = np.asarray(f_hat(s), dtype=complex)
    if a.shape != (n_nodes,):
        msg = f"f_hat must return shape ({n_nodes},), got {a.shape}"
        raise ValueError(msg)
    a = a.copy()
    a[0] *= 0.5

    # Quotient-difference algorithm -> continued-fraction coefficients d[0 .. 2M].
    d = np.empty(n_nodes, dtype=complex)
    d[0] = a[0]
    q = a[1:] / a[:-1]  # q_1^{(i)}, length 2M
    d[1] = -q[0]
    e = q[1:] - q[:-1]  # e_1^{(i)} = e_0^{(i+1)} + q_1^{(i+1)} - q_1^{(i)}, length 2M-1
    d[2] = -e[0]
    for r in range(2, m + 1):
        q = q[1:-1] * e[1:] / e[:-1]  # q_r^{(i)}, length 2M-2r+2
        d[2 * r - 1] = -q[0]
        e = e[1:-1] + q[1:] - q[:-1]  # e_r^{(i)}, length 2M-2r+1
        d[2 * r] = -e[0]

    # Evaluate the continued fraction at z = exp(i pi t / T), vectorized over t, by the three-term
    # recurrence A_n = A_{n-1} + d_n z A_{n-2}, B_n = B_{n-1} + d_n z B_{n-2}. The loop stops one
    # coefficient early (n up to 2M-1) so the de Hoog tail remainder can replace the bare last
    # coefficient d_{2M} in the final convergent -- the "improved" estimate of the truncated
    # continued-fraction tail. Keeping the two trailing convergents avoids any division when forming it.
    z = np.exp(1j * np.pi * t_flat / big_t)
    a_pp = np.zeros_like(z)  # A_{-1}
    b_pp = np.ones_like(z)  # B_{-1}
    a_p = np.full_like(z, d[0])  # A_0
    b_p = np.ones_like(z)  # B_0
    for n in range(1, n_nodes - 1):
        dz = d[n] * z
        a_pp, a_p = a_p, a_p + dz * a_pp
        b_pp, b_p = b_p, b_p + dz * b_pp
    # a_p, b_p hold the (2M-1) convergent; a_pp, b_pp the (2M-2) convergent.
    rem = 0.5 * (1.0 + z * (d[n_nodes - 2] - d[n_nodes - 1]))
    rem = -rem * (1.0 - np.sqrt(1.0 + z * d[n_nodes - 1] / (rem * rem)))
    a_final = a_p + rem * a_pp
    b_final = b_p + rem * b_pp

    result = (np.exp(gamma * t_flat) / big_t) * np.real(a_final / b_final)
    result = np.where(t_flat > 0.0, result, 0.0)
    return result.reshape(t_arr.shape) if scalar_input else result
