"""Independent Godunov upwind finite-volume oracle for front-tracking transport.

The front tracker resolves the scalar conservation law ``∂_θ C_T(c) + ∂_V c = 0`` exactly
(method of characteristics with shock/rarefaction bookkeeping). This module solves the *same*
law by a completely different method — first-order Godunov upwind finite volumes — to provide
an independent reference for the tracker's outlet concentration and stored mass.

One path for every isotherm. Because the flux ``f(U) = c`` is monotone increasing in the
conserved ``U = C_T(c)`` and the characteristic speed ``λ = df/dU = 1/R(c) > 0`` everywhere
(one-directional flow), the Godunov interface flux reduces to pure **upwind**:
``F_{i+1/2} = c_i``. The only isotherm-specific step is inverting ``U → c``; it is done by
vectorised bisection on the monotone ``total_concentration`` (the tracker's forward map,
which is not part of the interaction logic under test), so the oracle reimplements none of
the shock/rarefaction/merge machinery it checks.

First-order upwind is monotone and TVD, so it converges to the entropy solution as the grid
refines; its error is O(ΔV) (numerical diffusion), which sets the agreement tolerance.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt


def c_from_total(
    sorption,
    u: npt.NDArray[np.floating],
    *,
    c_hi: float,
    c_guess: npt.NDArray[np.floating] | None = None,
    tol: float = 1e-12,
    maxiter: int = 60,
) -> npt.NDArray[np.floating]:
    """Invert ``U = C_T(c)`` for ``c`` by safeguarded vectorised Newton.

    The Newton derivative is free: ``dC_T/dc = R(c)`` is exactly the retardation factor, so
    the update ``c <- c - (C_T(c) - U)/R(c)`` uses only the isotherm's forward maps (no
    reimplementation of any inverse). Warm-started from ``c_guess`` (the previous timestep's
    cells, which barely move), it converges in a handful of iterations; a fixed bisection
    fallback would be ~10x slower for the same accuracy over a long march.

    Parameters
    ----------
    sorption : SorptionModel
        Any model with ``total_concentration`` and ``retardation``.
    u : ndarray
        Conserved totals ``C_T`` to invert. Non-negative.
    c_hi : float
        Upper clamp for ``c`` (kept for safety; ``c ≤ U`` since sorption only adds mass).
    c_guess : ndarray, optional
        Warm-start (previous cells). Defaults to ``min(U, c_hi)`` (a valid upper bracket).
    tol : float, optional
        Absolute convergence tolerance on ``|C_T(c) - U|`` (default 1e-12).
    maxiter : int, optional
        Newton iteration cap (default 60).

    Returns
    -------
    ndarray
        Dissolved concentration ``c`` with ``C_T(c) = u``.
    """
    u = np.asarray(u, dtype=float)
    c = np.clip(np.minimum(u, c_hi) if c_guess is None else np.asarray(c_guess, dtype=float), 0.0, c_hi)
    for _ in range(maxiter):
        ct = np.asarray(sorption.total_concentration(c), dtype=float)
        resid = ct - u
        if np.max(np.abs(resid)) <= tol:
            break
        r = np.maximum(np.asarray(sorption.retardation(c), dtype=float), 1e-300)
        c = np.clip(c - resid / r, 0.0, c_hi)
    return c


def upwind_fv_outlet(
    sorption,
    cin: npt.NDArray[np.floating],
    theta_edges: npt.NDArray[np.floating],
    v_outlet: float,
    *,
    n_cells: int = 1500,
    cfl: float = 0.4,
    c_hi: float | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Upwind FV march of ``U = C_T`` on ``[0, v_outlet]``; return outlet ``(θ, c)`` samples.

    The column starts empty (``c = 0``). The inlet ghost cell carries ``cin(θ)`` at ``V = 0``.
    The step ``Δθ`` is set from the CFL limit using the maximum characteristic speed
    ``max 1/R`` over the inlet concentration range, then snapped so an integer number of steps
    reaches ``θ_edges[-1]``.

    Parameters
    ----------
    sorption : SorptionModel
        Sorption/conductivity model.
    cin : ndarray
        Inlet concentration, constant per input bin; aligned with ``theta_edges``.
    theta_edges : ndarray
        Cumulative-flow bin edges (length ``len(cin) + 1``).
    v_outlet : float
        Outlet position (pore volume).
    n_cells : int, optional
        Number of V-cells (default 1500).
    cfl : float, optional
        Courant number (default 0.4).
    c_hi : float, optional
        Initial bracket for the ``U → c`` inversion; defaults to ``4·max(cin)``.

    Returns
    -------
    theta : ndarray
        θ at each recorded step.
    cout : ndarray
        Dissolved concentration in the outlet cell at each step.
    """
    cin = np.asarray(cin, dtype=float)
    theta_edges = np.asarray(theta_edges, dtype=float)
    c_max = float(cin.max())
    c_hi = c_hi if c_hi is not None else max(c_max * 4.0, 1e-6)

    dv = v_outlet / n_cells
    u = np.zeros(n_cells)

    c_probe = np.linspace(1e-9 * max(c_max, 1e-9), max(c_max, 1e-9), 200)
    lam_max = float(np.max(1.0 / np.asarray(sorption.retardation(c_probe), dtype=float)))
    theta_end = float(theta_edges[-1])
    n_steps = max(int(np.ceil(theta_end / (cfl * dv / lam_max))), 1)
    dtheta = theta_end / n_steps

    theta = 0.0
    rec_theta = np.empty(n_steps)
    rec_cout = np.empty(n_steps)
    c_cells = np.zeros(n_cells)  # warm-start carrier
    for k in range(n_steps):
        c_cells = c_from_total(sorption, u, c_hi=c_hi, c_guess=c_cells)  # concentrations at current θ
        rec_theta[k] = theta
        rec_cout[k] = c_cells[-1]  # outlet cell at current θ
        idx = min(max(int(np.searchsorted(theta_edges, theta, side="right") - 1), 0), len(cin) - 1)
        c_left = np.empty(n_cells)
        c_left[0] = float(cin[idx])  # inlet ghost at V=0
        c_left[1:] = c_cells[:-1]
        u = np.maximum(u - (dtheta / dv) * (c_cells - c_left), 0.0)
        theta += dtheta
    return rec_theta, rec_cout
