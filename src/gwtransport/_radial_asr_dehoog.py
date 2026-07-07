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
    r"""Invert a Laplace transform -- optionally a whole batch of them -- at an array of times.

    The transform ``f_hat`` is sampled once on ``2 * n_terms + 1`` complex Bromwich nodes; the
    continued-fraction acceleration and its evaluation are vectorized over ``t`` and over any trailing
    batch axes ``f_hat`` carries, so one call inverts every entry of a propagator matrix in a single
    QD/continued-fraction pass. Only ``t > 0`` is meaningful; ``t <= 0`` returns ``0`` (the inverse of a
    one-sided transform vanishes for negative time, and the series is undefined at ``t = 0``).

    Parameters
    ----------
    f_hat : callable
        Vectorized Laplace transform :math:`\bar f(s)`. Receives a complex ``ndarray`` of shape
        ``(2 * n_terms + 1,)`` and returns a complex ``ndarray`` whose leading axis is ``2 * n_terms + 1``,
        optionally with trailing batch axes (e.g. all ``n_quad**2`` entries of a propagator matrix).
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
        Real inverse ``f(t)``, shape ``t.shape + batch`` where ``batch`` is the trailing shape of
        ``f_hat``'s output (a scalar ``t`` with no batch yields a 0-d array).

    Raises
    ------
    ValueError
        If ``f_hat``'s returned array does not have leading axis ``2 * n_terms + 1``.

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
    n_t = t_flat.size

    m = int(n_terms)
    n_nodes = 2 * m + 1
    t_max = float(np.max(t_flat)) if t_flat.size else 1.0
    big_t = float(scaling) if scaling is not None else 2.0 * t_max
    if big_t <= 0.0:
        big_t = 1.0  # all t <= 0: the result is masked to 0 below, this only keeps the nodes finite
    gamma = alpha - np.log(tol) / (2.0 * big_t)

    # Sample the transform on the Bromwich nodes s_k = gamma + i k pi / T, k = 0 .. 2M. ``f_hat`` may
    # carry a trailing batch shape; the QD/continued-fraction pass below broadcasts over it, so one call
    # inverts a whole batch (e.g. every entry of a propagator matrix) at once.
    k = np.arange(n_nodes)
    s = gamma + 1j * k * np.pi / big_t
    a = np.asarray(f_hat(s), dtype=complex)
    if a.shape[0] != n_nodes:
        msg = f"f_hat must return leading axis {n_nodes}, got {a.shape}"
        raise ValueError(msg)
    batch = a.shape[1:]
    nb = len(batch)
    a = a.copy()
    # A column is *underflow-degenerate* when its transform stays FINITE but decays to the double-precision
    # floor -- identically zero (a decoupled azimuthal mode with no source) or underflowing toward zero at
    # the high-frequency nodes (a heavily damped far-field propagator entry: rest / Airy / Riccati). Both
    # drive the quotient-difference recurrence into 0/0 or x/0 and poison the column with NaN, yet the
    # physical inverse is ~0. Such columns are detected here (finite transform) and their non-finite output
    # is zeroed below. A column whose transform OVERFLOWED (inf/nan already in ``a`` -- an ill-scaled kernel)
    # is a genuine breakdown and is deliberately NOT masked: it must still surface as NaN so a real failure
    # can never masquerade as a physical zero.
    finite_transform = np.all(np.isfinite(a), axis=0)
    a[0] *= 0.5

    # Quotient-difference algorithm -> continued-fraction coefficients d[0 .. 2M] (per batch entry; the
    # rhombus rules slice the leading node axis and broadcast over the trailing batch). The degenerate
    # columns above form 0/0 / x/0 here and in the continued fraction, so every division is evaluated under
    # errstate; their non-finite output is masked to zero after the assembly.
    d = np.empty((n_nodes, *batch), dtype=complex)
    d[0] = a[0]
    with np.errstate(all="ignore"):  # degenerate columns form 0/0, x/0 and over/underflow; masked below
        q = a[1:] / a[:-1]  # q_1^{(i)}, length 2M
        d[1] = -q[0]
        e = q[1:] - q[:-1]  # e_1^{(i)} = e_0^{(i+1)} + q_1^{(i+1)} - q_1^{(i)}, length 2M-1
        d[2] = -e[0]
        for r in range(2, m + 1):
            q = q[1:-1] * e[1:] / e[:-1]  # q_r^{(i)}, length 2M-2r+2
            d[2 * r - 1] = -q[0]
            e = e[1:-1] + q[1:] - q[:-1]  # e_r^{(i)}, length 2M-2r+1
            d[2 * r] = -e[0]

    # Evaluate the continued fraction at z = exp(i pi t / T) by the three-term recurrence
    # A_n = A_{n-1} + d_n z A_{n-2}, B_n = B_{n-1} + d_n z B_{n-2}, broadcasting time (leading axis) against
    # the batch (trailing axes). The loop stops one coefficient early (n up to 2M-1) so the de Hoog tail
    # remainder can replace the bare last coefficient d_{2M} in the final convergent -- the "improved"
    # estimate of the truncated tail. Keeping the two trailing convergents avoids any division forming it.
    time_shape = (n_t, *([1] * nb))  # broadcast the leading time axis against the trailing batch
    z = np.exp(1j * np.pi * t_flat / big_t).reshape(time_shape)  # (n_t, 1..1)
    a_pp = np.zeros((n_t, *batch), dtype=complex)  # A_{-1}
    b_pp = np.ones((n_t, *batch), dtype=complex)  # B_{-1}
    a_p = np.broadcast_to(d[0], (n_t, *batch))  # A_0 (read-only view; only ever read below, never mutated)
    b_p = np.ones((n_t, *batch), dtype=complex)  # B_0
    with np.errstate(all="ignore"):  # degenerate columns propagate Inf/NaN through the CF; masked below
        for n in range(1, n_nodes - 1):
            dz = d[n] * z  # d[n] (batch) broadcasts against z (time, 1..1) -> (n_t, *batch)
            a_pp, a_p = a_p, a_p + dz * a_pp
            b_pp, b_p = b_p, b_p + dz * b_pp
        # a_p, b_p hold the (2M-1) convergent; a_pp, b_pp the (2M-2) convergent.
        rem = 0.5 * (1.0 + z * (d[n_nodes - 2] - d[n_nodes - 1]))
        rem = -rem * (1.0 - np.sqrt(1.0 + z * d[n_nodes - 1] / (rem * rem)))
        a_final = a_p + rem * a_pp
        b_final = b_p + rem * b_pp
        result = (np.exp(gamma * t_flat).reshape(time_shape) / big_t) * np.real(a_final / b_final)
    result = np.where(t_flat.reshape(time_shape) > 0.0, result, 0.0)
    # Zero the underflow-degenerate columns (finite transform, non-finite QD output). Overflowed columns
    # (non-finite transform) are left as NaN so a genuine breakdown surfaces rather than reads as zero.
    result = np.where(~np.isfinite(result) & finite_transform, 0.0, result)
    if scalar_input:
        return result[0]
    return result.reshape(t_arr.shape + batch)
