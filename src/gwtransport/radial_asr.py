r"""Exact radial advection-dispersion transport for a single well (push-pull / ASR).

Water is injected in an infinite aquifer at a single fully-penetrating well and later recovered at
the same well under a signed flow schedule (push-pull / ASR). Transport is radial advection with
microdispersion, molecular diffusion, and linear sorption; the spread of velocities across the well
screen provides macrodispersion. Forward and backward modeling are supported.

Computes the extracted flux concentration ``cout`` at a single fully-penetrating well driven by an
arbitrary signed flow schedule (positive = injection, negative = extraction, zero = rest) and an
arbitrary injected concentration ``cin``. The physics is the exact radial advection-dispersion of the
radial ASR knowledge base: volume coordinate ``V(r) = pi b n (r^2 - r_w^2)``, Scheidegger
velocity-dependent dispersion ``D = alpha_L |u| + D_m`` (microdispersion ``alpha_L |u|`` plus molecular diffusion ``D_m``), Kreft-Zuber flux boundary conditions, and
the exact per-phase kernels (Airy for ``D_m = 0``; the log-derivative Riccati ODE for ``D_m > 0``).
Nothing is reduced to a Gaussian; the exact
non-Gaussian breakthrough (with the correct skewness) is carried.

The forward map is **grid-free** end to end -- no PDE is discretized, so none of the finite-volume
artefacts appear. A single inject-then-extract cycle with no intervening rest uses the closed-form echo operator
(``gwtransport._radial_asr_compose``, KB Sec. 10a) -- exact for arbitrary within-phase variable flow,
with the exact temporal moments. Any other signed-flow schedule (more reversals / multi-cycle ASR, or a
single cycle with a rest under nonzero ``D_m``) uses the reused-propagator-matrix engine
(``gwtransport._radial_asr_reuse``, KB addendum Sec. A1-A7), which composes the exact per-phase kernels
(Airy / Riccati / Bessel) through the interior two-point Green's functions. Each per-reversal field
hand-off ``f_out = P @ f`` is a bounded linear operator; its matrix ``P`` is built once per distinct
``(direction, phase volume)`` from a single batched de Hoog inversion and reused at every recurrence, so
the special-function + inversion cost is ``O(distinct phase volumes)`` rather than ``O(reversals)``. It is
bit-equivalent, to the de Hoog floor, to the per-reversal grid-free composition. Molecular diffusion during
pumping (the ``D_m > 0`` Whittaker kernel) is evaluated through the log-derivative Riccati ODE
(``gwtransport._radial_asr_kernels.resolvent_riccati``) -- exact to the de Hoog inversion floor at any
``A_0/D_m``, with no special-function precision cap, and reducing continuously to the Airy branch as
``D_m -> 0``. During a **rest** (``Q = 0``) advection and microdispersion vanish and molecular
diffusion acts alone on the wall-clock clock; it is carried exactly by the order-0 modified Bessel
pure-diffusion kernel, the dominant mixing for seasonal storage / ATES. The only
numerical steps are Gauss-Legendre quadrature and de Hoog Laplace inversion of exact special-function
kernels. An independent finite-volume solve of the same PDE (``tests/src/_radial_asr_fv_oracle.py``,
KB Sec. 9) is used only as a test oracle. The propagator matrices are assembled on the Bromwich
contour (``Re s > 0``), where the field hand-off is well-conditioned at any Peclet. The engine is chosen
automatically; cycles are expressed through the flow sign pattern, not
an argument.

The reported ``cout`` is the flow-weighted average over each output bin -- defined on extraction bins
(``flow < 0``) and ``NaN`` on injection / rest bins (nothing is recovered there).

Macrodispersion within the well screen
--------------------------------------
The well screen has a **known** height; macrodispersion is the spread of arrival times caused by
*velocity heterogeneity across the screen*. It is modelled as parallel streamtubes (``pore_heights``):
each streamtube is an independent radial cell carrying the full flow, with an effective pore height
that sets its velocity, and the output is the weight-averaged breakthrough. A streamtube of effective
height ``b`` has velocity ``proportional to 1/b`` (its pore volume to radius ``r`` is
``pi b n (r^2 - r_w^2)``), so smaller ``b`` means faster breakthrough.
:func:`gamma_infiltration_to_extraction` builds this ensemble from a gamma distribution of the layer
**velocity** within the fixed screen height (see that function); the mean velocity is set by the screen
height and the spread by a velocity coefficient of variation. The spread is a within-screen velocity
distribution -- velocity heterogeneity across the well screen -- not an aquifer pore-volume distribution.

Regional background flow (drift)
--------------------------------
With a steady uniform regional Darcy flux ``regional_flux`` (``U``, drift seepage ``v_d = U/n``) the well
field is superimposed on a regional gradient, so the stored bubble drifts and recovery degrades. The
radial symmetry is broken and the transport is solved by an **azimuthal Fourier-mode** expansion
``c(r, theta) = sum_m c_m(r) e^{i m theta}`` (``m = 0`` is the radial engine; drift couples ``m`` to
``m +- 1``), composed through the same per-phase interior Green's functions
(``gwtransport._radial_asr_drift_kernels``). ``regional_flux = 0`` (default) dispatches to the radial path
bit-for-bit. The engine is for the **slow-drift** envelope -- the plume must stay well inside the
stagnation radius ``r_s = |A_0|/|v_d|`` (else a ``ValueError``); rest phases under drift are not yet
supported. The drift-induced recovery loss is validated against an independent 2-D finite-volume oracle.

Available functions:

- :func:`infiltration_to_extraction` -- forward transport (cin -> cout).
- :func:`extraction_to_infiltration` -- inverse via Tikhonov regularization (cout -> cin).
- :func:`gamma_infiltration_to_extraction` -- gamma-distributed screen velocity (forward).
- :func:`gamma_extraction_to_infiltration` -- same, inverse.

References
----------
The references below give the published closed-form solutions for the **single-phase** radial *injection*
problem (steady divergent flow from one well) -- the per-phase forward kernel this module composes. The
convergent-extraction dual (KB Sec. 7) and the multi-cycle push-pull / ASR composition across flow
reversals are built on top of those kernels here and are not in the single-injection references. All
share the assumptions used here: a single fully-penetrating well in a homogeneous medium with steady
divergent flow ``v = Q / (2 pi b n r)``, plus retardation.

The ``D_m = 0`` kernel (velocity-proportional microdispersion ``D = alpha_L |u|``, Airy functions)
is the classical radial-dispersion problem: Tang & Babu (1979) under a Dirichlet (resident-concentration)
well boundary, and Chen (1987) under the Cauchy / third-type (flux) boundary used here -- explicitly the
Kreft-Zuber flux concentration, with transfer function ``Ai(Y) / [Ai(Y0)/2 - p^(1/3) Ai'(Y0)]`` equal to
the flux operator this module evaluates. The ``D_m > 0`` kernel (``D = alpha_L |u| + D_m``, Kummer /
confluent-hypergeometric functions) under the same flux boundary, with retardation, is Aichi & Akitaya
(2018) -- whose well operator ``U(a,b) + 2a U(a+1,b+1)`` is this module's Whittaker flux boundary; they
record the ``D_m -> 0`` reduction to Chen (1987) as an open problem, which this module performs
continuously -- the log-derivative Riccati kernel reduces smoothly to the Airy branch as ``D_m -> 0``.
The ``alpha_L = 0`` limit (constant diffusion, drift-dominated radial transport, Whittaker equation) is
Akanji & Falade (2019). Each is an injection-only solution; none treats extraction or multi-cycle push-pull.

Kreft, A., & Zuber, A. (1978). On the physical meaning of the dispersion equation and its solutions
for different initial and boundary conditions. Chemical Engineering Science, 33(11), 1471-1480.

Tang, D. H., & Babu, D. K. (1979). Analytical solution of a velocity dependent dispersion problem.
Water Resources Research, 15(6), 1471-1478.

Chen, C.-S. (1987). Analytical solutions for radial dispersion with Cauchy boundary at injection well.
Water Resources Research, 23(7), 1217-1224.

Aichi, M., & Akitaya, K. (2018). Analytical solution for a radial advection-dispersion equation
including both mechanical dispersion and molecular diffusion for a steady-state flow field in a
horizontal aquifer caused by a constant rate injection from a well. Hydrological Research Letters,
12(3), 23-27.

Akanji, L. T., & Falade, G. K. (2019). Closed-form solution of radial transport of tracers in porous
media influenced by linear drift. Energies, 12(1), 29.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport import gamma
from gwtransport._radial_asr_compose import single_cycle_echo_matrix
from gwtransport._radial_asr_drift_kernels import block_cout_deviation
from gwtransport._radial_asr_reuse import cout_deviation
from gwtransport._time import dt_to_days


def _is_single_cycle(flow: npt.NDArray[np.floating]) -> bool:
    """Return True if the schedule is a single injection block followed by a single extraction block.

    Such schedules (one flow reversal, injection first) use the exact closed-form echo operator; any
    other signed-flow pattern (more reversals, extraction first) uses the reused-propagator-matrix engine.

    Returns
    -------
    bool
        Whether ``flow`` is a single inject-then-extract cycle.
    """
    signs = np.sign(flow[flow != 0.0])
    n_changes = int(np.sum(np.diff(signs) != 0)) if signs.size else 0
    return n_changes <= 1 and (signs.size == 0 or signs[0] > 0)


def _validate(
    *,
    cin_or_cout: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    pore_heights: npt.NDArray[np.floating],
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    weights: npt.NDArray[np.floating] | None,
    regional_flux: float,
    n_modes: int | None,
) -> None:
    """Validate inputs for the radial single-well transport functions (signed flow is allowed).

    Raises
    ------
    ValueError
        On inconsistent lengths, non-positive geometry, out-of-range porosity, negative dispersion,
        ``retardation_factor < 1``, mismatched ``weights``, NaN in ``flow``, a non-finite ``regional_flux``,
        an ``n_modes < 1``, or a ``cout_tedges`` that differs from ``tedges`` (a distinct output grid is not
        yet supported).
    """
    if len(tedges) != len(flow) + 1:
        msg = "tedges must have one more element than flow"
        raise ValueError(msg)
    if len(cin_or_cout) != len(flow):
        msg = "cin/cout must have the same length as flow"
        raise ValueError(msg)
    if not tedges.equals(cout_tedges):
        msg = "cout_tedges must equal tedges (a distinct output grid is not yet supported)"
        raise ValueError(msg)
    if np.any(np.isnan(flow)):
        msg = "flow contains NaN values, which are not allowed"
        raise ValueError(msg)
    if np.any(pore_heights <= 0.0):
        msg = "pore_heights must be positive"
        raise ValueError(msg)
    if not 0.0 < porosity <= 1.0:
        msg = "porosity must be in (0, 1]"
        raise ValueError(msg)
    if well_radius <= 0.0:
        msg = "well_radius must be positive"
        raise ValueError(msg)
    if longitudinal_dispersivity <= 0.0:
        msg = "longitudinal_dispersivity must be positive (the dispersion kernel requires alpha_L > 0)"
        raise ValueError(msg)
    if molecular_diffusivity < 0.0:
        msg = "molecular_diffusivity must be non-negative"
        raise ValueError(msg)
    if retardation_factor < 1.0:
        msg = "retardation_factor must be >= 1.0"
        raise ValueError(msg)
    if weights is not None and len(weights) != len(pore_heights):
        msg = "weights must have the same length as pore_heights"
        raise ValueError(msg)
    if not np.isfinite(regional_flux):
        msg = "regional_flux must be finite"
        raise ValueError(msg)
    if n_modes is not None and n_modes < 1:
        msg = "n_modes must be >= 1"
        raise ValueError(msg)


def _echo_operator(
    *,
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    c_geos: npt.NDArray[np.floating],
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    weights: npt.NDArray[np.floating],
    n_quad: int,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Weight-averaged single-cycle echo operator ``W`` (``cout' = W @ cin'_inj``) over the streamtubes.

    Builds the closed-form echo matrix per streamtube (geometry constant ``c_geo = pi b n``) and
    averages by ``weights``. Used by both the forward (``cout = W @ cin``) and the reverse (Tikhonov).

    Returns
    -------
    w_ens : ndarray, shape (n_ext, n_inj)
        Weight-averaged echo operator.
    inj_mask, ext_mask : ndarray of bool
        Injection (``flow > 0``) and extraction (``flow < 0``) bin masks.
    """
    inj_mask, ext_mask = flow > 0.0, flow < 0.0
    dt = dt_to_days(tedges)
    inj_vol = np.concatenate(([0.0], np.cumsum((flow * dt)[inj_mask])))  # 0 .. S_inj
    ext_vol = np.concatenate(([0.0], np.cumsum((-flow * dt)[ext_mask])))  # 0 .. T_end
    inj_flow_scale = float(np.mean(flow[inj_mask])) if np.any(inj_mask) else 1.0
    ext_flow_scale = float(np.mean(-flow[ext_mask])) if np.any(ext_mask) else 1.0
    w_ens = np.zeros((int(np.sum(ext_mask)), int(np.sum(inj_mask))))
    for c_geo, w_i in zip(c_geos, weights, strict=True):
        w_ens += w_i * single_cycle_echo_matrix(
            inj_volume_edges=inj_vol,
            ext_volume_edges=ext_vol,
            c_geo=c_geo,
            r_w=well_radius,
            alpha_l=longitudinal_dispersivity,
            inj_flow_scale=inj_flow_scale,
            ext_flow_scale=ext_flow_scale,
            retardation_factor=retardation_factor,
            molecular_diffusivity=molecular_diffusivity,
            n_quad=n_quad,
        )
    return w_ens / np.sum(weights), inj_mask, ext_mask


def _reuse_ensemble(
    cin_deviation: npt.NDArray[np.floating],
    *,
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geos: npt.NDArray[np.floating],
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    weights: npt.NDArray[np.floating],
    n_quad: int,
) -> npt.NDArray[np.floating]:
    """Weight-averaged multi-cycle extracted-flux deviation over the streamtubes.

    Runs the reused-propagator-matrix multi-cycle engine once per streamtube (geometry constant
    ``c_geo = pi b n``) and averages by ``weights``. Used by the forward (with ``cin``) and the reverse
    (with unit pulses).

    Returns
    -------
    ndarray, shape (n,)
        Weight-averaged extracted-flux deviation on extraction bins, ``0`` elsewhere.
    """
    acc = np.zeros(len(flow))
    for c_geo, w_i in zip(c_geos, weights, strict=True):
        acc += w_i * np.nan_to_num(
            cout_deviation(
                cin_deviation=cin_deviation,
                flow=flow,
                dt_days=dt_days,
                c_geo=c_geo,
                r_w=well_radius,
                alpha_l=longitudinal_dispersivity,
                molecular_diffusivity=molecular_diffusivity,
                retardation_factor=retardation_factor,
                n_quad=n_quad,
            )
        )
    return acc / np.sum(weights)


def _auto_n_modes(
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geo: float,
    well_radius: float,
    v_d: float,
) -> int:
    r"""Azimuthal truncation ``M`` sized from the plume-front drift ratio ``eps(R_b) = v_d R_b / A_0``.

    The mode amplitudes decay geometrically, ``|c_m| ~ eps^|m|``, so keeping modes ``-M .. M`` truncates the
    azimuthal field at ``O(eps^{M+1})``. ``M`` is chosen so that tail is below ``~5e-3`` and clamped to
    ``[2, 6]`` (the slow-drift envelope -- beyond ``eps ~ 0.6`` the far-field escape this engine does not
    model dominates anyway). ``A_0`` uses the **smallest** pumping magnitude (the worst-case largest ``eps``,
    consistent with the stagnation-radius envelope guard), ``R_b`` the peak net injected radius. This
    function is only reached for nonzero drift, so ``eps > 0`` and the ``[2, 6]`` clamp covers the rest.

    Returns
    -------
    int
        Azimuthal truncation ``M``.
    """
    a0 = float(np.min(np.abs(flow[flow != 0.0]))) / (2.0 * c_geo)
    net_volume = np.concatenate(([0.0], np.cumsum(flow * dt_days)))
    peak_volume = max(float(net_volume.max()), 0.0)
    r_b = np.sqrt(well_radius**2 + peak_volume / c_geo)
    eps = min(abs(v_d) * r_b / abs(a0), 0.6)
    return int(np.clip(np.ceil(np.log(5e-3) / np.log(eps)), 2, 6))


def _block_ensemble(
    cin_deviation: npt.NDArray[np.floating],
    *,
    flow: npt.NDArray[np.floating],
    dt_days: npt.NDArray[np.floating],
    c_geos: npt.NDArray[np.floating],
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float,
    retardation_factor: float,
    regional_flux: float,
    n_modes: int | None,
    weights: npt.NDArray[np.floating],
    n_quad: int,
) -> npt.NDArray[np.floating]:
    """Weight-averaged multi-cycle extracted-flux deviation with regional drift over the streamtubes.

    Runs the azimuthal-mode block engine (:func:`gwtransport._radial_asr_drift_kernels.block_cout_deviation`)
    once per streamtube (geometry constant ``c_geo = pi b n``) and averages by ``weights``. The drift
    seepage ``v_d = U / n`` is the same for every streamtube (a regional Darcy flux through the porosity);
    only the radial strength ``A_0 ~ 1/c_geo`` varies, so faster (thinner) streamtubes see a smaller drift
    ratio. ``n_modes`` is auto-sized per streamtube from its drift ratio when not given.

    Returns
    -------
    ndarray, shape (n,)
        Weight-averaged extracted-flux deviation on extraction bins, ``0`` elsewhere.
    """
    v_d = regional_flux / porosity
    acc = np.zeros(len(flow))
    for c_geo, w_i in zip(c_geos, weights, strict=True):
        m = n_modes if n_modes is not None else _auto_n_modes(flow, dt_days, c_geo, well_radius, v_d)
        acc += w_i * np.nan_to_num(
            block_cout_deviation(
                cin_deviation=cin_deviation,
                flow=flow,
                dt_days=dt_days,
                c_geo=c_geo,
                r_w=well_radius,
                alpha_l=longitudinal_dispersivity,
                v_d=v_d,
                molecular_diffusivity=molecular_diffusivity,
                retardation_factor=retardation_factor,
                n_modes=m,
                n_quad=n_quad,
            )
        )
    return acc / np.sum(weights)


def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    pore_heights: npt.ArrayLike,
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    weights: npt.ArrayLike | None = None,
    background: float = 0.0,
    regional_flux: float = 0.0,
    n_modes: int | None = None,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    """Compute the extracted flux concentration at a radial well for a signed flow schedule.

    Parameters
    ----------
    cin : array-like, shape (n,)
        Injected concentration per time bin (used only on injection bins, ``flow > 0``).
    flow : array-like, shape (n,)
        Signed flow per time bin [m^3/day]: ``> 0`` injection, ``< 0`` extraction, ``0`` rest.
    tedges : DatetimeIndex
        Time bin edges (``n + 1`` for ``n`` bins).
    cout_tedges : DatetimeIndex
        Output time bin edges; must equal ``tedges``. Output is NaN on injection / rest bins.
    pore_heights : array-like
        Effective streamtube pore height(s) ``b`` [m] -- a scalar (one homogeneous screen) or an array
        of streamtube heights for the velocity-heterogeneity macrodispersion ensemble (each streamtube
        carries the full flow; smaller ``b`` = faster). See the module docstring and
        :func:`gamma_infiltration_to_extraction`.
    porosity : float
        Porosity ``n`` [-].
    well_radius : float
        Well (screen) radius ``r_w`` [m].
    longitudinal_dispersivity : float
        Longitudinal dispersivity ``alpha_L`` [m].
    molecular_diffusivity : float, optional
        Molecular diffusivity ``D_m`` [m^2/day]. Default 0. ``D_m = 0`` uses the vectorized Airy branch;
        ``D_m > 0`` uses the log-derivative Riccati kernel -- exact to the de Hoog floor at any ``A_0/D_m``
        with no precision cap, reducing continuously to the Airy branch as ``D_m -> 0``.
    retardation_factor : float, optional
        Linear retardation ``R >= 1``. Default 1.
    weights : array-like, optional
        Per-streamtube averaging weights (same length as ``pore_heights``). Default equal weights.
    background : float, optional
        Ambient aquifer concentration ``c_bg``. The deviation ``cin - c_bg`` is transported and
        ``c_bg`` is added back; constant ``cin = c_bg`` returns ``cout = c_bg``. Default 0.
    regional_flux : float, optional
        Steady uniform regional background Darcy flux ``U`` [m/day] in ``+x`` (drift seepage
        ``v_d = U / n``). ``0`` (default) reproduces the radial-symmetric engine bit-for-bit. A nonzero
        value engages the azimuthal-mode block engine, which captures the drift-induced recovery loss
        (the down-gradient plume is partly swept past the well). The slow-drift envelope requires the
        plume to stay well inside the stagnation radius ``r_s = |A_0| / |v_d|`` (a ``ValueError`` is raised
        otherwise). Rest phases (``flow == 0``) combined with nonzero drift are not yet supported
        (``NotImplementedError``); the rest-with-drift propagator is future work.
    n_modes : int, optional
        Azimuthal truncation ``M`` for the drift engine (keeps modes ``-M .. M``). Default ``None``
        auto-sizes ``M`` from the plume-front drift ratio ``eps = v_d R_b / A_0`` (clamped to ``[2, 6]``).
        Ignored when ``regional_flux == 0``.
    n_quad : int, optional
        Gauss-Legendre node count for the resident-profile superposition. Default 240.

    Returns
    -------
    ndarray, shape (n,)
        Extracted flux concentration; NaN on injection and rest bins.
    """
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    pore_heights = np.atleast_1d(np.asarray(pore_heights, dtype=float))
    weights_arr = np.ones(len(pore_heights)) if weights is None else np.atleast_1d(np.asarray(weights, dtype=float))
    _validate(
        cin_or_cout=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        pore_heights=pore_heights,
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=None if weights is None else weights_arr,
        regional_flux=regional_flux,
        n_modes=n_modes,
    )
    c_geos = np.pi * pore_heights * porosity
    cout = np.full(len(flow), np.nan)
    if regional_flux != 0.0:
        # Steady regional drift breaks radial symmetry: the azimuthal-mode block engine carries it.
        ext_mask = flow < 0.0
        cout_dev = _block_ensemble(
            cin - background,
            flow=flow,
            dt_days=dt_to_days(tedges),
            c_geos=c_geos,
            porosity=porosity,
            well_radius=well_radius,
            longitudinal_dispersivity=longitudinal_dispersivity,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
            regional_flux=regional_flux,
            n_modes=n_modes,
            weights=weights_arr,
            n_quad=n_quad,
        )
        cout[ext_mask] = background + cout_dev[ext_mask]
        return cout
    # A rest phase combined with molecular diffusion (seasonal storage / ATES) cannot use the
    # flushed-volume echo operator, which is blind to a rest's wall-clock diffusion; route it to the
    # reuse engine (which propagates the rest with the Bessel pure-diffusion kernel).
    use_echo = _is_single_cycle(flow) and not (molecular_diffusivity > 0.0 and np.any(flow == 0.0))
    if use_echo:
        w_ens, inj_mask, ext_mask = _echo_operator(
            flow=flow,
            tedges=tedges,
            c_geos=c_geos,
            well_radius=well_radius,
            longitudinal_dispersivity=longitudinal_dispersivity,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
            weights=weights_arr,
            n_quad=n_quad,
        )
        cout[ext_mask] = background + w_ens @ (cin[inj_mask] - background)
        return cout
    ext_mask = flow < 0.0
    cout_dev = _reuse_ensemble(
        cin - background,
        flow=flow,
        dt_days=dt_to_days(tedges),
        c_geos=c_geos,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=weights_arr,
        n_quad=n_quad,
    )
    cout[ext_mask] = background + cout_dev[ext_mask]
    return cout


def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    pore_heights: npt.ArrayLike,
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    weights: npt.ArrayLike | None = None,
    background: float = 0.0,
    regional_flux: float = 0.0,
    n_modes: int | None = None,
    regularization_strength: float = 1e-10,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    """Recover the injected concentration from extracted-water measurements (Tikhonov inverse).

    Inverts the forward operator built by :func:`infiltration_to_extraction`. Returns the injected
    concentration on injection bins (NaN on extraction / rest bins).

    Parameters
    ----------
    cout : array-like, shape (n,)
        Measured extracted concentration (used on extraction bins, ``flow < 0``).
    flow, tedges, cout_tedges, pore_heights, porosity, well_radius, longitudinal_dispersivity
        As in :func:`infiltration_to_extraction`.
    molecular_diffusivity, retardation_factor, weights, background, regional_flux, n_modes, n_quad
        As in :func:`infiltration_to_extraction`.
    regularization_strength : float, optional
        Tikhonov parameter. Default ``1e-10``.

    Returns
    -------
    ndarray, shape (n,)
        Recovered injected concentration; NaN on extraction / rest bins.
    """
    cout = np.asarray(cout, dtype=float)
    flow = np.asarray(flow, dtype=float)
    pore_heights = np.atleast_1d(np.asarray(pore_heights, dtype=float))
    weights_arr = np.ones(len(pore_heights)) if weights is None else np.atleast_1d(np.asarray(weights, dtype=float))
    _validate(
        cin_or_cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        pore_heights=pore_heights,
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=None if weights is None else weights_arr,
        regional_flux=regional_flux,
        n_modes=n_modes,
    )
    c_geos = np.pi * pore_heights * porosity
    # A rest phase with molecular diffusion routes to the reuse engine (see the forward function); regional
    # drift never uses the radial echo operator (it breaks the azimuthal symmetry the echo relies on).
    use_echo = (
        regional_flux == 0.0 and _is_single_cycle(flow) and not (molecular_diffusivity > 0.0 and np.any(flow == 0.0))
    )
    if use_echo:
        w_ens, inj_mask, ext_mask = _echo_operator(
            flow=flow,
            tedges=tedges,
            c_geos=c_geos,
            well_radius=well_radius,
            longitudinal_dispersivity=longitudinal_dispersivity,
            molecular_diffusivity=molecular_diffusivity,
            retardation_factor=retardation_factor,
            weights=weights_arr,
            n_quad=n_quad,
        )
    else:
        # Build the dense forward operator W by reuse-engine unit-pulse columns (one ensemble solve per
        # injection bin); the reverse cannot reuse the cheap single-solve forward path.
        inj_mask, ext_mask = flow > 0.0, flow < 0.0
        inj_idx = np.flatnonzero(inj_mask)
        dt_days = dt_to_days(tedges)
        w_ens = np.zeros((int(np.sum(ext_mask)), len(inj_idx)))
        for j, idx in enumerate(inj_idx):
            pulse = np.zeros(len(flow))
            pulse[idx] = 1.0
            if regional_flux != 0.0:
                col = _block_ensemble(
                    pulse,
                    flow=flow,
                    dt_days=dt_days,
                    c_geos=c_geos,
                    porosity=porosity,
                    well_radius=well_radius,
                    longitudinal_dispersivity=longitudinal_dispersivity,
                    molecular_diffusivity=molecular_diffusivity,
                    retardation_factor=retardation_factor,
                    regional_flux=regional_flux,
                    n_modes=n_modes,
                    weights=weights_arr,
                    n_quad=n_quad,
                )
            else:
                col = _reuse_ensemble(
                    pulse,
                    flow=flow,
                    dt_days=dt_days,
                    c_geos=c_geos,
                    well_radius=well_radius,
                    longitudinal_dispersivity=longitudinal_dispersivity,
                    molecular_diffusivity=molecular_diffusivity,
                    retardation_factor=retardation_factor,
                    weights=weights_arr,
                    n_quad=n_quad,
                )
            w_ens[:, j] = col[ext_mask]
    # Tikhonov least-squares min ||W x - (cout-bg)||^2 + lambda ||x||^2 via the stable augmented
    # system [W; sqrt(lambda) I] x = [cout-bg; 0]. The echo / reuse operator has column sums ~1
    # (mass conservation per injection bin) and overdetermined rows, so a direct Tikhonov fit is used.
    n_inj = w_ens.shape[1]
    augmented = np.vstack([w_ens, np.sqrt(regularization_strength) * np.eye(n_inj)])
    rhs = np.concatenate([cout[ext_mask] - background, np.zeros(n_inj)])
    cin_dev = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    cin = np.full(len(flow), np.nan)
    cin[inj_mask] = background + cin_dev
    return cin


def gamma_infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    screen_height: float,
    velocity_cv: float,
    n_bins: int = 100,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    background: float = 0.0,
    regional_flux: float = 0.0,
    n_modes: int | None = None,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    """Radial transport with gamma-distributed screen velocity (within-screen macrodispersion).

    The well screen has a **known** height ``screen_height``; macrodispersion is the spread of arrival
    times from velocity heterogeneity across that fixed height. The layer velocity is gamma-distributed
    with mean equal to the homogeneous value (a streamtube at the mean velocity has effective pore
    height ``screen_height``) and coefficient of variation ``velocity_cv``. A streamtube with velocity
    ratio ``rho`` (gamma, mean 1) has effective pore height ``screen_height / rho`` -- faster layers are
    thinner and break through sooner. The gamma is discretized into ``n_bins`` equal-probability bins
    (:func:`gwtransport.gamma.bins`) and averaged by probability mass via
    :func:`infiltration_to_extraction`.

    Parameters
    ----------
    screen_height : float
        Known well-screen height ``H`` [m] (the fixed total; the mean streamtube velocity is set by it).
    velocity_cv : float
        Coefficient of variation of the layer velocity (the macrodispersion strength). ``0`` is a
        homogeneous screen (a single streamtube, sharp breakthrough); typically ``< 1`` -- larger values
        give a heavy slow-velocity tail.
    n_bins : int, optional
        Number of equal-probability velocity bins. Default 100.
    cin, flow, tedges, cout_tedges, porosity, well_radius, longitudinal_dispersivity
        As in :func:`infiltration_to_extraction`.
    molecular_diffusivity, retardation_factor, background, regional_flux, n_modes, n_quad
        As in :func:`infiltration_to_extraction`.

    Returns
    -------
    ndarray, shape (n,)
        Extracted flux concentration; NaN on injection / rest bins.
    """
    pore_heights, weights = _velocity_gamma_streamtubes(screen_height, velocity_cv, n_bins)
    return infiltration_to_extraction(
        cin=cin,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        pore_heights=pore_heights,
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=weights,
        background=background,
        regional_flux=regional_flux,
        n_modes=n_modes,
        n_quad=n_quad,
    )


def gamma_extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    porosity: float,
    well_radius: float,
    longitudinal_dispersivity: float,
    screen_height: float,
    velocity_cv: float,
    n_bins: int = 100,
    molecular_diffusivity: float = 0.0,
    retardation_factor: float = 1.0,
    background: float = 0.0,
    regional_flux: float = 0.0,
    n_modes: int | None = None,
    regularization_strength: float = 1e-10,
    n_quad: int = 240,
) -> npt.NDArray[np.floating]:
    """Inverse of :func:`gamma_infiltration_to_extraction` (gamma-distributed screen velocity).

    Returns
    -------
    ndarray, shape (n,)
        Recovered injected concentration; NaN on extraction / rest bins.
    """
    pore_heights, weights = _velocity_gamma_streamtubes(screen_height, velocity_cv, n_bins)
    return extraction_to_infiltration(
        cout=cout,
        flow=flow,
        tedges=tedges,
        cout_tedges=cout_tedges,
        pore_heights=pore_heights,
        porosity=porosity,
        well_radius=well_radius,
        longitudinal_dispersivity=longitudinal_dispersivity,
        molecular_diffusivity=molecular_diffusivity,
        retardation_factor=retardation_factor,
        weights=weights,
        background=background,
        regional_flux=regional_flux,
        n_modes=n_modes,
        regularization_strength=regularization_strength,
        n_quad=n_quad,
    )


def _velocity_gamma_streamtubes(
    screen_height: float, velocity_cv: float, n_bins: int
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Streamtube pore heights and weights for a gamma-distributed screen velocity (mean velocity <-> H).

    The layer velocity ratio ``rho`` is gamma(mean 1, std ``velocity_cv``); the effective pore height is
    ``screen_height / rho`` (velocity ~ 1/height), so the mean velocity corresponds to height ``H``.

    Returns
    -------
    pore_heights : ndarray
        Effective streamtube pore heights ``screen_height / rho`` per velocity bin.
    weights : ndarray
        Probability mass per velocity bin.

    Raises
    ------
    ValueError
        If ``screen_height`` or ``velocity_cv`` is not positive.
    """
    if screen_height <= 0.0:
        msg = "screen_height must be positive"
        raise ValueError(msg)
    if velocity_cv <= 0.0:
        msg = "velocity_cv must be positive (use a single homogeneous screen for no macrodispersion)"
        raise ValueError(msg)
    bins = gamma.bins(mean=1.0, std=velocity_cv, n_bins=n_bins)
    return screen_height / bins["expected_values"], bins["probability_mass"]
