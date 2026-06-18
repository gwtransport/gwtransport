r"""Finite-volume solver of the exact radial advection-dispersion PDE (general signed flow + D_m).

This is the general engine for cases the closed-form single-cycle path does not cover: multi-cycle
schedules (more than one flow reversal) and molecular diffusion ``D_m > 0``. It integrates the exact
conservative V-coordinate PDE

    d_t C + Q d_V C = d_V(D_V d_V C),   D_V = 2 alpha_L |Q| sqrt(pi b n (V+V_w)) + 4 pi b n D_m (V+V_w),

with the Kreft-Zuber flux / Danckwerts boundary pair, by an implicit (backward-Euler) finite-volume
scheme on a uniform-in-radius grid (KB Sec. 9). The physics is not simplified -- the exact equation,
exact boundary conditions, and both dispersion mechanisms are carried; only the spatio-temporal
discretization is approximate (controlled by ``n_cells`` / ``n_sub``, validated against the exact
single-cycle kernels). Backward Euler is L-stable, so the near-well cell-Courant blow-up of a
uniform-r grid does not destabilise it (KB Sec. 10b), and reporting ``cout`` as the flow-weighted
bin average averages out any residual ringing.

Working in the radius form, the cell volume is ``dV_k = pi b n (r_{k+1/2}^2 - r_{k-1/2}^2)`` and the
finite-volume flux at interface radius ``rho`` is ``F = Q C^up - (alpha_L |Q| + 2 pi b n D_m rho)
(C_{k+1}-C_k)/dr`` with upwind ``C^up`` (KB Sec. 9). Linear retardation multiplies the storage term
by ``R`` (the residence time scales by ``R``). Boundary conditions: injection (``Q>0``) a third-type
flux ``F_well = Q c_in``; extraction (``Q<0``) Danckwerts (zero dispersive flux) ``F_well = Q C_0``
with extracted flux concentration ``C_0``; rest (``Q=0``) no flux. The outer boundary uses a zero
(background-deviation) ghost cell placed beyond the plume.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import solve_banded

# Outer-boundary safety margin: the zero-deviation ghost is placed where the plume volume reaches
# this multiple of the largest net injected volume (plus the dispersive reach), so the outer BC never
# clips the plume. Generous because an implicit solve makes extra cells cheap.
_R_MAX_VOLUME_FACTOR = 3.0


def fv_cout_deviation(
    *,
    cin_deviation: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geo: float,
    r_w: float,
    alpha_l: float,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    n_cells: int = 300,
    n_sub: int = 6,
) -> npt.NDArray[np.floating]:
    """Extracted-flux deviation per flow bin for one disk, by implicit finite volume.

    Solves the conservative V-form PDE for the concentration deviation ``c' = c - c_bg`` (initial
    condition zero) over the full signed-flow schedule and returns the flow-weighted bin average of
    the well-face flux concentration on extraction bins (NaN on injection / rest bins).

    Parameters
    ----------
    cin_deviation : ndarray, shape (n,)
        Injected concentration deviation per bin (used on injection bins, ``flow > 0``).
    flow : ndarray, shape (n,)
        Signed flow per bin [m^3/day].
    dt_days : ndarray, shape (n,)
        Bin widths [day].
    c_geo : float
        Geometry constant ``pi b n`` (``V = c_geo (r^2 - r_w^2)``).
    r_w : float
        Well radius [m].
    alpha_l : float
        Longitudinal dispersivity [m].
    molecular_diffusivity : float, optional
        Molecular diffusivity [m^2/day]. Default 0.
    retardation_factor : float, optional
        Linear retardation ``R >= 1``. Default 1.
    n_cells : int, optional
        Number of uniform-in-radius cells. Default 300.
    n_sub : int, optional
        Backward-Euler sub-steps per flow bin. Default 6.

    Returns
    -------
    ndarray, shape (n,)
        Extracted-flux deviation per bin; NaN on injection / rest bins.
    """
    # Outer radius: contain the plume (largest net injected volume) plus a dispersive reach.
    net_volume = np.concatenate(([0.0], np.cumsum(flow * dt_days)))
    peak_volume = max(float(net_volume.max()), 0.0)
    # ~6 sigma radial reach beyond the advective front: mechanical (alpha_L) + molecular (D_m over the
    # whole schedule), so the zero-deviation outer ghost never clips the plume even at large D_m.
    total_time = float(np.sum(dt_days))
    dispersive_reach = 6.0 * alpha_l * np.sqrt(max(peak_volume / c_geo, 1.0)) + 6.0 * np.sqrt(
        molecular_diffusivity * total_time
    )
    r_max = np.sqrt(r_w**2 + _R_MAX_VOLUME_FACTOR * peak_volume / c_geo) + dispersive_reach + r_w

    edges = np.linspace(r_w, r_max, n_cells + 1)  # cell faces (radius), edges[0] = r_w
    dr = (r_max - r_w) / n_cells
    cell_volume = c_geo * (edges[1:] ** 2 - edges[:-1] ** 2)  # dV_k
    r_int = edges[1:-1]  # interior interface radii (between consecutive cells), length n_cells - 1
    r_out = edges[-1]  # outer face radius

    conc = np.zeros(n_cells)  # deviation field, IC 0
    cout = np.full(len(flow), np.nan)

    for i, (q, dt_bin, cin_i) in enumerate(zip(flow, dt_days, cin_deviation, strict=True)):
        dt = dt_bin / n_sub
        abs_q = abs(q)
        # Interface dispersion coefficient / dr (interior): (alpha_L |Q| + 2 c_geo D_m rho) / dr.
        disp_int = (alpha_l * abs_q + 2.0 * c_geo * molecular_diffusivity * r_int) / dr
        disp_out = (alpha_l * abs_q + 2.0 * c_geo * molecular_diffusivity * r_out) / dr

        # Per-interface flux Jacobian d F_i / d C_i (a) and d F_i / d C_{i+1} (b), upwind on Q.
        if q >= 0.0:
            a_int = q + disp_int
            b_int = -disp_int
        else:
            a_int = disp_int
            b_int = q - disp_int

        # Assemble the tridiagonal A (banded) for the implicit step. Storage R dV/dt on the diagonal;
        # interface i contributes +F_i to cell i and -F_i to cell i+1.
        diag = retardation_factor * cell_volume / dt
        diag[:-1] += a_int
        diag[1:] += -b_int
        upper = b_int.copy()  # A[i, i+1]
        lower = -a_int.copy()  # A[i+1, i]
        # Outer boundary (cell n_cells-1 gains +F_out; F_out depends only on C_{N-1} via the zero ghost).
        diag[-1] += (q + disp_out) if q >= 0.0 else disp_out
        # Inner (well) boundary contribution to cell 0 (-F_well).
        if q < 0.0:  # extraction: Danckwerts F_well = Q C_0 -> -F_well adds -Q to the diagonal
            diag[0] += -q
        ab = np.zeros((3, n_cells))
        ab[0, 1:] = upper
        ab[1, :] = diag
        ab[2, :-1] = lower

        well_face_sum = 0.0
        for _ in range(n_sub):
            rhs = (retardation_factor * cell_volume / dt) * conc
            if q > 0.0:  # injection: third-type flux F_well = Q cin -> +Q cin on cell 0 RHS
                rhs[0] += q * cin_i
            conc = solve_banded((1, 1), ab, rhs)
            well_face_sum += conc[0]
        if q < 0.0:  # extracted flux concentration is the well-face cell value, bin-averaged
            cout[i] = well_face_sum / n_sub
    return cout
