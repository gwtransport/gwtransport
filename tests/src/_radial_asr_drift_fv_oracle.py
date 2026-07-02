r"""Independent 2-D finite-volume oracle for single-well ASR with steady regional drift (tests only).

The test oracle for the drift engine (:mod:`gwtransport._radial_asr_drift_kernels`): an independent
polar finite-volume discretization of the same 2-D advection-dispersion PDE the grid-free engine
solves, used only to cross-check it. It deliberately reintroduces the spatio-temporal discretization
the engine avoids, so it carries a first-order (``~O(1/n_cells)``) error and is expected to *converge
to* the engine under refinement (Richardson). At ``regional_flux = 0`` it reduces cell-by-cell to the
1-D radial oracle (:mod:`tests.src._radial_asr_fv_oracle`), hence to the radial engine, to machine
precision.

The seepage velocity superposes the radial well field on the uniform regional drift,

    v = v_d x_hat + A(t) r_hat / r,   A(t) = Q(t) / (2 pi b n),   v_d = U / n,

and the mechanical dispersion is the rank-1 Scheidegger tensor plus isotropic molecular diffusion
(transverse mechanical dispersion neglected, ``alpha_T = 0``),

    D = alpha_L (v outer v) / |v| + D_m I.

A conservative backward-Euler scheme on a polar ``(r, theta)`` grid integrates it. The radial/tangential
diagonal dispersion (``D_rr``, ``D_thth``) and the upwind advection are implicit (a 5-point stencil,
tridiagonal in ``r`` and periodic-tridiagonal in ``theta``); the ``D_rtheta`` cross term is added by
deferred correction (explicit, from the previous sub-step). Boundary conditions mirror the 1-D oracle
per ``theta``-face: injection (``Q>0``) a Kreft-Zuber third-type flux ``F_well = Q_r,well c_in``;
extraction (``Q<0``) Danckwerts (zero dispersive flux); rest (``Q=0``) a zero-deviation ghost so the
drift advects past a shut well. The extracted flux concentration is the flow-weighted (m=0) average of
the well-face value over the ``theta``-faces.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.linalg import splu

# Outer-boundary safety: place the zero-deviation ghost where the plume volume reaches this multiple of
# the peak net injected volume, then add the drift translation and the dispersive reach on top.
_R_MAX_VOLUME_FACTOR = 3.0


def fv2d_cout_deviation(
    *,
    cin_deviation: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    b: float,
    porosity: float,
    r_w: float,
    alpha_l: float,
    regional_flux: float = 0.0,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    n_r: int = 160,
    n_theta: int = 48,
    n_sub: int = 6,
    cross_term: bool = True,
) -> npt.NDArray[np.floating]:
    r"""Extracted-flux deviation per flow bin for one streamtube, by implicit 2-D polar finite volume.

    Solves the 2-D advection-dispersion PDE for the concentration deviation ``c' = c - c_bg`` (initial
    condition zero) over the full signed-flow schedule and returns the flow-weighted (m=0) well-face
    flux concentration on extraction bins (NaN on injection / rest bins).

    Parameters
    ----------
    cin_deviation : ndarray, shape (n,)
        Injected concentration deviation per bin (used on injection bins, ``flow > 0``).
    flow : ndarray, shape (n,)
        Signed flow per bin [m^3/day]: ``> 0`` injection, ``< 0`` extraction, ``0`` rest.
    dt_days : ndarray, shape (n,)
        Bin widths [day].
    b : float
        Aquifer / screen thickness [m].
    porosity : float
        Porosity ``n`` [-].
    r_w : float
        Well radius [m].
    alpha_l : float
        Longitudinal dispersivity [m].
    regional_flux : float, optional
        Uniform regional Darcy flux ``U`` [m/day] in ``+x`` (drift seepage ``v_d = U / n``). Default 0.
    molecular_diffusivity : float, optional
        Molecular diffusivity ``D_m`` [m^2/day]. Default 0.
    retardation_factor : float, optional
        Linear retardation ``R >= 1``. Default 1.
    n_r, n_theta : int, optional
        Radial cell count and azimuthal cell count. Defaults 160, 48.
    n_sub : int, optional
        Backward-Euler sub-steps per flow bin. Default 6.
    cross_term : bool, optional
        Include the ``D_rtheta`` cross dispersion by deferred correction. Default True.

    Returns
    -------
    ndarray, shape (n,)
        Extracted-flux deviation per bin; NaN on injection / rest bins.
    """
    flow = np.asarray(flow, dtype=float)
    dt_days = np.asarray(dt_days, dtype=float)
    cin_deviation = np.asarray(cin_deviation, dtype=float)
    nporo = porosity
    nb = nporo * b
    c_geo = np.pi * b * nporo
    v_d = regional_flux / nporo
    d_m = molecular_diffusivity

    # Domain: contain the plume (peak net injected volume) plus the drift translation and dispersive reach.
    net_vol = np.concatenate(([0.0], np.cumsum(flow * dt_days)))
    peak_vol = max(float(net_vol.max()), 0.0)
    total_time = float(np.sum(dt_days))
    r_front = np.sqrt(max(peak_vol / c_geo, 1.0))
    drift_total = abs(v_d) * total_time
    reach = 6.0 * alpha_l * r_front + 6.0 * np.sqrt(d_m * total_time)
    r_max = np.sqrt(r_w**2 + _R_MAX_VOLUME_FACTOR * peak_vol / c_geo) + drift_total + reach + r_w

    er = np.linspace(r_w, r_max, n_r + 1)  # radial faces
    dr = (r_max - r_w) / n_r
    rc = 0.5 * (er[:-1] + er[1:])  # cell-center radii
    dth = 2.0 * np.pi / n_theta
    thc = (np.arange(n_theta) + 0.5) * dth  # cell-center angles
    thf = np.arange(n_theta) * dth  # theta faces (face j sits between cells j-1 and j)
    vol = c_geo * (er[1:] ** 2 - er[:-1] ** 2) / n_theta  # (n_r,) cell pore volume
    cosc, sinc, cosf, sinf = np.cos(thc), np.sin(thc), np.cos(thf), np.sin(thf)
    ncell = n_r * n_theta

    # Flattened cell index helpers (i in 0..n_r-1, j in 0..n_theta-1).
    ij = np.arange(ncell)
    ii = ij // n_theta

    conc = np.zeros((n_r, n_theta))
    cout = np.full(len(flow), np.nan)

    for ibin, (q, dt_bin, cin_i) in enumerate(zip(flow, dt_days, cin_deviation, strict=True)):
        a0 = q / (2.0 * c_geo)
        dt = dt_bin / n_sub
        rows_l, cols_l, vals_l = [], [], []

        # --- interior radial faces at er[k+1], k = 0..n_r-2 (vectorized over (k, j)) ---
        rf = er[1:-1][:, None]  # (n_r-1, 1)
        vr = a0 / rf + v_d * cosc[None, :]
        vth = -v_d * sinc[None, :]
        speed = np.sqrt(vr**2 + vth**2) + 1e-300
        drr = alpha_l * vr**2 / speed + d_m
        qr = nb * dth * (a0 + v_d * rf * cosc[None, :])
        gr = drr * nb * rf * dth / dr
        up, dn = np.where(qr >= 0.0, qr, 0.0), np.where(qr < 0.0, qr, 0.0)
        a_idx = (np.arange(n_r - 1)[:, None] * n_theta + np.arange(n_theta)[None, :]).ravel()
        b_idx = ((np.arange(n_r - 1) + 1)[:, None] * n_theta + np.arange(n_theta)[None, :]).ravel()
        for r_, c_, v_ in (
            (a_idx, a_idx, (up + gr).ravel()),
            (a_idx, b_idx, (dn - gr).ravel()),
            (b_idx, a_idx, -(up + gr).ravel()),
            (b_idx, b_idx, -(dn - gr).ravel()),
        ):
            rows_l.append(r_)
            cols_l.append(c_)
            vals_l.append(v_)

        # --- theta faces at thf[j] between (i, j-1) and (i, j) (vectorized over (i, j)) ---
        vthf = -v_d * sinf[None, :]
        vrf = a0 / rc[:, None] + v_d * cosf[None, :]
        speedt = np.sqrt(vrf**2 + vthf**2) + 1e-300
        dthth = alpha_l * vthf**2 / speedt + d_m
        qth = (-nb * dr * v_d * sinf[None, :]) * np.ones((n_r, 1))
        gth = dthth * nb * dr / (rc[:, None] * dth)
        upt, dnt = np.where(qth >= 0.0, qth, 0.0), np.where(qth < 0.0, qth, 0.0)
        jm1 = (np.arange(n_theta) - 1) % n_theta
        at_idx = (np.arange(n_r)[:, None] * n_theta + jm1[None, :]).ravel()
        bt_idx = (np.arange(n_r)[:, None] * n_theta + np.arange(n_theta)[None, :]).ravel()
        for r_, c_, v_ in (
            (at_idx, at_idx, (upt + gth).ravel()),
            (at_idx, bt_idx, (dnt - gth).ravel()),
            (bt_idx, at_idx, -(upt + gth).ravel()),
            (bt_idx, bt_idx, -(dnt - gth).ravel()),
        ):
            rows_l.append(r_)
            cols_l.append(c_)
            vals_l.append(v_)

        # --- inner well boundary (i = 0, face at r_w) ---
        qr_well = nb * dth * (a0 + v_d * r_w * cosc)  # outward (>0 inject)
        well0 = np.arange(n_theta)
        if q < 0.0:  # extraction: Danckwerts (zero dispersive flux)
            rows_l.append(well0)
            cols_l.append(well0)
            vals_l.append(-qr_well)
        elif q == 0.0:  # rest: zero-deviation ghost, drift past a shut well
            rows_l.append(well0)
            cols_l.append(well0)
            vals_l.append(np.maximum(qr_well, 0.0))

        # --- outer boundary (i = n_r-1, face at r_max), zero-deviation ghost ---
        vr_out = a0 / r_max + v_d * cosc
        drr_out = alpha_l * vr_out**2 / (np.sqrt(vr_out**2 + (v_d * sinc) ** 2) + 1e-300) + d_m
        qr_out = nb * dth * (a0 + v_d * r_max * cosc)
        gr_out = drr_out * nb * r_max * dth / dr
        out0 = (n_r - 1) * n_theta + np.arange(n_theta)
        rows_l.append(out0)
        cols_l.append(out0)
        vals_l.append(np.maximum(qr_out, 0.0) + gr_out)

        # storage diagonal + assemble
        diag0 = retardation_factor * vol[ii] / dt
        rows = np.concatenate([*rows_l, ij])
        cols = np.concatenate([*cols_l, ij])
        vals = np.concatenate([*vals_l, diag0])
        lu = splu(sp.csc_matrix((vals, (rows, cols)), shape=(ncell, ncell)))

        # D_rtheta cross-dispersion coefficients at the faces (deferred correction below).
        drth_rf = alpha_l * vr * vth / speed  # at radial faces (n_r-1, n_theta)
        drth_tf = alpha_l * vrf * vthf / speedt  # at theta faces (n_r, n_theta)
        use_cross = cross_term and v_d != 0.0

        wt = np.abs(qr_well)
        wtsum = wt.sum()
        well_sum = 0.0
        for _ in range(n_sub):
            rhs = (retardation_factor * vol[:, None] / dt) * conc
            if q > 0.0:
                rhs[0, :] += qr_well * cin_i
            if use_cross:
                rhs += _cross_flux(conc, drth_rf, drth_tf, nb, dr, dth, n_r, n_theta)
            conc = lu.solve(rhs.ravel()).reshape(n_r, n_theta)
            if q < 0.0:
                well_sum += float((wt * conc[0, :]).sum() / wtsum)
        if q < 0.0:
            cout[ibin] = well_sum / n_sub
    return cout


def _cross_flux(
    conc: npt.NDArray[np.floating],
    drth_rf: npt.NDArray[np.floating],
    drth_tf: npt.NDArray[np.floating],
    nb: float,
    dr: float,
    dth: float,
    n_r: int,
    n_theta: int,
) -> npt.NDArray[np.floating]:
    r"""Explicit deferred-correction divergence of the ``D_rtheta`` cross fluxes (added to the RHS).

    The radial-face cross flux uses the tangential gradient ``(1/r) d c/d theta``; the theta-face cross
    flux uses the radial gradient ``d c/d r``. Returns the net influx per cell (storage-RHS sign).

    Returns
    -------
    ndarray, shape (n_r, n_theta)
        Net cross-flux contribution per cell.
    """
    out = np.zeros((n_r, n_theta))
    dc_dth = 0.5 * (np.roll(conc, -1, axis=1) - np.roll(conc, 1, axis=1)) / dth  # centered, at cell centers
    tang_face = 0.5 * (dc_dth[:-1, :] + dc_dth[1:, :])  # average to radial face k
    fr = -drth_rf * nb * dth * tang_face  # area r dth, tangential grad (1/r) dC/dth -> r cancels
    # fr is the outward (+r) cross flux at the face between cells k and k+1: cell k loses it, cell k+1
    # gains it. The distribution sign is load-bearing: inverted, the oracle disperses along the
    # y-mirrored velocity -- a different rank-1 tensor with its own y-symmetric solution, wrong at
    # O(eps^2) of the breakthrough. The engine-vs-FV loss tests pin it.
    out[:-1, :] -= fr
    out[1:, :] += fr
    dc_dr = np.zeros((n_r, n_theta))
    dc_dr[1:-1, :] = (conc[2:, :] - conc[:-2, :]) / (2 * dr)
    rad_face = 0.5 * (dc_dr + np.roll(dc_dr, 1, axis=1))  # average to theta face j
    fth = -drth_tf * nb * dr * rad_face
    # fth is the (+theta) cross flux at face j between cells j-1 and j: cell j-1 loses it, cell j gains.
    out += fth
    out -= np.roll(fth, -1, axis=1)
    return out
