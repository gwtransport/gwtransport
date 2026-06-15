"""
Percolation through thick unsaturated zones via the Kinematic Wave method.

This module provides one public function:

- :func:`root_zone_to_water_table_kinematic_wave` — exact front-tracking
  solver for gravity-driven percolation between the bottom of the root
  zone and the water table, following the Kinematic-Wave method described
  in Olsthoorn (2026, *Stromingen* 32(1)). Supports Brooks-Corey and
  van Genuchten-Mualem constitutive curves and a time-varying
  multiplicative scaling of K(θ) (e.g. for temperature-corrected
  viscosity).

**Forward-only.** Inverse mapping ``water_table_to_root_zone`` is not
provided. The KW unsaturated-zone problem is fundamentally one-way under
gravity: multiple ``q_root_zone(t)`` series produce indistinguishable
``q_water_table(t)`` after the column's intrinsic low-pass response,
making the inverse ill-posed. Users wanting an inverse should formulate
it as a regularised inverse problem outside this package.

**Cumulative pore-volume coordinate.** The position axis is *cumulative
pore volume per unit cross-sectional area* (units of length), not
geometric depth. For a soil of constant porosity ``n_p ≡ θ_s`` and
water-table depth ``z_wt``, the conversion is ``V_out = θ_s · z_wt``.
The docstring of :func:`root_zone_to_water_table_kinematic_wave`
spells out the recovery rule and the layered-porosity generalisation.

References
----------
.. [1] Olsthoorn, T.N. (2026). Percolation through thick unsaturated
   zones — Munsflow vs. the Kinematic Wave. *Stromingen* 32(1).
.. [2] Heinen, M., Bakker, G., Wösten, J.M.H. (2020). Waterretentie en
   Doorlatendheidskarakteristieken van boven- en ondergronden in
   Nederland: de Staringreeks. Update 2018. Wageningen Environmental
   Research, Report 2978.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport._time import tedges_to_days
from gwtransport.advection import _flow_weighted_front_tracking_output
from gwtransport.fronttracking.math import (
    BrooksCoreyConductivity,
    VanGenuchtenMualemConductivity,
)
from gwtransport.fronttracking.solver import FrontTracker
from gwtransport.fronttracking.waves import CharacteristicWave, RarefactionWave, ShockWave


def root_zone_to_water_table_kinematic_wave(
    *,
    q_root_zone: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    q_water_table_tedges: pd.DatetimeIndex,
    cumulative_pore_volumes_outlet: npt.ArrayLike,
    theta_r: float,
    theta_s: float,
    k_s: float,
    brooks_corey_lambda: float | None = None,
    van_genuchten_n: float | None = None,
    mualem_l: float = 0.5,
    k_scaling: npt.ArrayLike | None = None,
    max_iterations: int = 10000,
) -> tuple[npt.NDArray[np.floating], list[dict]]:
    r"""Percolation flux at the water table by exact Kinematic-Wave front tracking.

    Solves the nonlinear scalar conservation law

    .. math::
        \\frac{\\partial \\theta_m}{\\partial t} +
        \\frac{\\partial K(\\theta_m)}{\\partial z} = 0

    exactly via :class:`gwtransport.fronttracking.solver.FrontTracker`,
    using either a Brooks-Corey or a van Genuchten-Mualem constitutive
    curve. Implements the Kinematic-Wave method described in
    Olsthoorn (2026). The capillary term ``∂ψ/∂z`` is dropped (gravity
    drainage only); real fronts are slightly smoothed by capillarity, so
    if smoothing matters use the Munsflow-style approach in
    :mod:`gwtransport.diffusion` instead.

    Parameters
    ----------
    q_root_zone : array-like
        Root-zone leakage entering the unsaturated zone at the top
        boundary [length/time, e.g. m/day]. Piecewise constant over each
        ``[tedges[i], tedges[i+1])`` bin. Non-negative.
        Length = ``len(tedges) - 1``. At any bin, ``q_root_zone <= f·K_s``
        must hold (with ``f = k_scaling`` or 1) for the inlet inversion
        to be well-defined; the validator raises ``ValueError`` otherwise.
    tedges : pandas.DatetimeIndex
        Time bin edges of the input series. Length ``n + 1`` for ``n`` bins.
    q_water_table_tedges : pandas.DatetimeIndex
        Output time bin edges. Free monotone index when ``k_scaling`` is
        None; **must equal** ``tedges`` when ``k_scaling`` is set (the
        back-transform ``q_wt = f · cout`` is exact only on the input grid).
        Must lie within the input window ``[tedges[0], tedges[-1]]`` (the flow
        series defines the system only there); querying beyond it raises.
    cumulative_pore_volumes_outlet : array-like
        Cumulative pore volume per unit cross-sectional area at the water
        table [length]. For a soil of constant porosity (``n_p ≡ θ_s``)
        and water-table depth ``z_wt``, this is ``θ_s · z_wt``. For
        layered porosity, ``∫₀^{z_wt} n_p(z') dz'``. The geometric depth
        is recovered as ``z_wt = V_out / θ_s`` (uniform case). Array-like
        to support a distribution of column lengths in parallel
        (analogous to :func:`gamma_infiltration_to_extraction`); each
        entry must be positive.
    theta_r : float
        Residual volumetric moisture content [-]. Must satisfy
        ``0 <= theta_r < theta_s``.
    theta_s : float
        Saturated volumetric moisture content [-]. Equal to the porosity
        for typical soils. Must satisfy ``theta_r < theta_s < 1``.
    k_s : float
        Saturated hydraulic conductivity [length/time]. Positive.
    brooks_corey_lambda : float or None, optional
        Brooks-Corey pore-size distribution index [-]. Set to use the
        Brooks-Corey branch. Mutually exclusive with ``van_genuchten_n``.
    van_genuchten_n : float or None, optional
        Van Genuchten shape parameter ``n_vG > 1``. Set to use the
        van Genuchten-Mualem branch (numerical inversion via brentq).
        Mutually exclusive with ``brooks_corey_lambda``.
    mualem_l : float, optional
        Mualem pore-connectivity parameter ``L``. Default 0.5
        (standard Mualem). Honored only when ``van_genuchten_n`` is set.
    k_scaling : array-like or None, optional
        Dimensionless time-only multiplicative factor ``f(t)`` applied
        to the entire ``K(θ)`` curve:
        ``K(θ, t) = f(t) · K_reference(θ)``. Length ``n``. Default None
        means ``f ≡ 1``. All entries must be strictly positive.

        The cumulative-flow trick in the underlying front-tracking solver
        absorbs ``f(t)`` exactly: wave dynamics in cumulative effective
        time remain flow-free. Typical usage is a temperature-corrected
        viscosity ``f(t) = μ_ref / μ(T(t))``; ``μ`` varies ~60% between
        5 °C and 25 °C, so seasonal swings of 30-50% in effective ``K_s``
        are realistic for shallow soils.
    max_iterations : int, optional
        Maximum number of solver events. Default 10000.

    Returns
    -------
    q_water_table : ndarray
        Bin-averaged percolation flux at the water table [same units as
        ``q_root_zone``], length ``len(q_water_table_tedges) - 1``,
        averaged across the columns in ``cumulative_pore_volumes_outlet``.
    structures : list of dict
        Per-column simulation structures (same schema as
        :func:`gwtransport.advection.infiltration_to_extraction_nonlinear_sorption`,
        with ``aquifer_pore_volume`` renamed to
        ``cumulative_pore_volume_outlet``):

        - ``waves`` — all wave objects.
        - ``events`` — event history; each record has ``"theta"`` (cumulative
          effective time) and ``"type"`` keys. Translate ``theta`` to wall-clock
          time via ``tracker_state.t_at_theta(event["theta"])``.
        - ``theta_first_arrival`` — cumulative effective time at which
          the first nonzero arrival reaches the outlet.
        - ``n_events``, ``n_shocks``, ``n_rarefactions``,
          ``n_characteristics`` — counts.
        - ``theta_current`` — final cumulative effective time.
        - ``sorption`` — the sorption object.
        - ``tracker_state`` — complete :class:`FrontTrackerState` for the
          column (use ``state.t_at_theta`` to translate ``θ → t``).
        - ``cumulative_pore_volume_outlet`` — the V_out for this column.

    Raises
    ------
    ValueError
        If inputs are inconsistent (wrong lengths, NaN, negative ``q_root_zone``
        or ``k_scaling``, non-positive ``cumulative_pore_volumes_outlet`` or
        ``k_s``), if neither or both sorption-parameter groups are supplied,
        if ``q_root_zone > f(t) * k_s`` at any bin (saturation/ponding limit),
        or if ``q_water_table_tedges`` does not equal ``tedges`` while
        ``k_scaling`` is provided.

    Warns
    -----
    UserWarning
        If output θ-bins extend beyond the inlet θ-window (i.e. the drying tail
        of ``q_root_zone`` reaches zero and the column has not yet equilibrated
        by the last output bin). Bin averages in that region are clamped to zero.

    See Also
    --------
    gwtransport.advection.infiltration_to_extraction_nonlinear_sorption :
        Solute transport with nonlinear sorption (analogous front-tracking
        algorithm in the saturated-zone domain).
    gwtransport.diffusion :
        Munsflow-style linearised advection-diffusion (complementary;
        smoothed fronts).
    gwtransport.fronttracking.math.BrooksCoreyConductivity :
        Brooks-Corey constitutive class.
    gwtransport.fronttracking.math.VanGenuchtenMualemConductivity :
        van Genuchten-Mualem constitutive class.
    :ref:`concept-kinematic-wave` : Background on the Kinematic-Wave method for
        unsaturated-zone percolation.

    Notes
    -----
    **Cumulative pore-volume coordinate.** The internal V axis is
    ``V(z) = int_0^z n_p(z') dz'`` (units of length). For a uniform soil
    with ``n_p = theta_s``, ``V = theta_s * z``; depth is recovered as
    ``z = V / theta_s``. The solver-side identification
    ``flow = theta_s * f(t)`` (with ``f`` the optional K-scaling) follows
    from the chain rule ``d/dz = theta_s * d/dV``.

    **Inlet boundary inversion.** The solver works in a reference frame
    where ``K = K_ref(theta_m)``; the time-varying scaling is moved to the
    boundary as ``cin_solver(t) = q_root_zone(t) / f(t)`` and recovered
    at the outlet as ``q_water_table(t) = f(t) * cout(t)``. The
    requirement ``cin_solver <= k_s`` (i.e. ``q_root_zone <= f * k_s``)
    is the saturation/ponding admissibility check enforced by the
    validator.

    **The KW approximation.** Capillary stresses are neglected; flow
    is gravity-only. Wetting fronts are sharp shocks satisfying
    Rankine-Hugoniot ``V_f = (K_1 - K_2)/(theta_1 - theta_2)``. Drying tails are
    self-similar rarefaction fans. Real fronts are slightly capillary-
    smoothed; if that smoothing matters, use Munsflow-style
    advection-diffusion (the article's Munsflow method, mapped to
    :mod:`gwtransport.diffusion` in this package).

    **Initial condition.** The column starts at ``theta_m = theta_r`` (i.e.
    ``K = 0``) everywhere. To start from field capacity or a long-term
    equilibrium, prepend a constant-q spin-up to the input series.

    **Exact mass conservation.** Both Brooks-Corey and van Genuchten-Mualem
    fan integrals use a closed-form integration-by-parts antiderivative
    derived from the universal identity ``R = dC_T/dC``: for the spatial
    fan integral ``G(u) = C_T(c) * u - kappa * c``, and for the temporal
    fan integral ``F(theta) = c * (theta - theta_origin) - Delta_v * C_T(c)``.
    For Brooks-Corey both ``c`` and ``C_T`` at the endpoints are closed
    form; for van Genuchten-Mualem they require a single ``brentq`` call
    per endpoint (transcendental ``K(theta)``). The Burdine variant
    (``mualem_l = 0``) admits a closed-form inverse and is fully
    free of root-finding.

    References
    ----------
    .. [1] Olsthoorn, T.N. (2026). Percolation through thick unsaturated
       zones — Munsflow vs. the Kinematic Wave. *Stromingen* 32(1).
    .. [2] Heinen, M., Bakker, G., Wösten, J.M.H. (2020). *Waterretentie
       en Doorlatendheidskarakteristieken van boven- en ondergronden in
       Nederland: de Staringreeks. Update 2018.* Wageningen Environmental
       Research, Report 2978.
    .. [3] Charbeneau, R.J. (2000). *Groundwater Hydraulics and Pollutant
       Transport.* Prentice Hall.

    Examples
    --------
    Reproduce a 10-year step-response for the article's soil O05
    (coarse sand, Brooks-Corey)::

        import numpy as np
        import pandas as pd
        from gwtransport.percolation import (
            root_zone_to_water_table_kinematic_wave,
        )

        tedges = pd.date_range("1995-01-01", "2005-01-01", freq="D")
        q_root = np.full(len(tedges) - 1, 1e-3)  # 1 mm/day

        q_wt, structures = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=np.array([0.337 * 20.0]),
            theta_r=0.01,
            theta_s=0.337,
            k_s=0.174,
            brooks_corey_lambda=0.25,
        )

    With time-varying water viscosity::

        days = ((tedges[:-1] - tedges[0]) / pd.Timedelta(days=1)).values
        T = 10.0 + 5.0 * np.sin(2 * np.pi * days / 365.25)  # °C
        mu_ref, dmu_dT = 1.31, -0.027  # mPa·s, linear around 10 °C
        mu = mu_ref + dmu_dT * (T - 10.0)
        k_scaling = mu_ref / mu

        q_wt_visc, _ = root_zone_to_water_table_kinematic_wave(
            q_root_zone=q_root,
            tedges=tedges,
            q_water_table_tedges=tedges,
            cumulative_pore_volumes_outlet=np.array([0.337 * 20.0]),
            theta_r=0.01,
            theta_s=0.337,
            k_s=0.174,
            brooks_corey_lambda=0.25,
            k_scaling=k_scaling,
        )
    """
    q_root_zone_arr = np.asarray(q_root_zone, dtype=float)
    # Promote a scalar / 0-d outlet pore volume to a single-column 1-d array so the
    # downstream per-column iteration and len() are well-defined (a lone scalar is a
    # valid single column, consistent with the "array-like" contract in the docstring).
    cumulative_pore_volumes_outlet_arr = np.atleast_1d(np.asarray(cumulative_pore_volumes_outlet, dtype=float))
    tedges = pd.DatetimeIndex(tedges)
    q_water_table_tedges = pd.DatetimeIndex(q_water_table_tedges)

    n_bins = len(q_root_zone_arr)
    if len(tedges) != n_bins + 1:
        msg = f"tedges must have length len(q_root_zone) + 1, got {len(tedges)} vs {n_bins + 1}"
        raise ValueError(msg)
    if q_water_table_tedges[0] < tedges[0] or q_water_table_tedges[-1] > tedges[-1]:
        msg = (
            f"q_water_table_tedges must lie within the input window [{tedges[0]}, {tedges[-1]}], got "
            f"[{q_water_table_tedges[0]}, {q_water_table_tedges[-1]}]. The flow series defines the "
            "system only over the input window; querying beyond it is ill-posed (extend q_root_zone instead)."
        )
        raise ValueError(msg)
    if np.any(q_root_zone_arr < 0):
        msg = "q_root_zone must be non-negative"
        raise ValueError(msg)
    if np.any(np.isnan(q_root_zone_arr)):
        msg = "q_root_zone must not contain NaN"
        raise ValueError(msg)
    if cumulative_pore_volumes_outlet_arr.size == 0 or np.any(cumulative_pore_volumes_outlet_arr <= 0):
        msg = "cumulative_pore_volumes_outlet must be non-empty with all entries positive"
        raise ValueError(msg)
    if not (0.0 <= theta_r < theta_s < 1.0):
        msg = f"theta_r, theta_s must satisfy 0 <= theta_r < theta_s < 1, got theta_r={theta_r}, theta_s={theta_s}"
        raise ValueError(msg)
    if k_s <= 0:
        msg = f"k_s must be positive, got {k_s}"
        raise ValueError(msg)
    if (brooks_corey_lambda is None) == (van_genuchten_n is None):
        msg = "Exactly one of brooks_corey_lambda or van_genuchten_n must be provided"
        raise ValueError(msg)

    if k_scaling is None:
        f = np.ones(n_bins, dtype=float)
    else:
        f = np.asarray(k_scaling, dtype=float)
        if f.shape != (n_bins,):
            msg = f"k_scaling must have shape ({n_bins},), got {f.shape}"
            raise ValueError(msg)
        if np.any(np.isnan(f)) or np.any(f <= 0):
            msg = "k_scaling must be strictly positive and contain no NaN"
            raise ValueError(msg)
        if len(q_water_table_tedges) != len(tedges) or not (q_water_table_tedges == tedges).all():
            msg = (
                "q_water_table_tedges must equal tedges when k_scaling is provided "
                "(the back-transform q_wt = f * cout is exact only on the input grid)"
            )
            raise ValueError(msg)

    # Saturation/ponding admissibility: K_ref(θ_m at inlet) = q_root/f must be <= k_s.
    cin_solver = q_root_zone_arr / f
    if float(cin_solver.max()) > k_s:
        bin_idx = int(cin_solver.argmax())
        msg = (
            f"Inlet saturation/ponding limit exceeded at bin {bin_idx}: "
            f"q_root_zone/k_scaling = {cin_solver[bin_idx]:.6g} > k_s = {k_s:.6g}. "
            "Reduce q_root_zone or increase k_scaling (warmer water → lower viscosity → higher k_s effective)."
        )
        raise ValueError(msg)

    if brooks_corey_lambda is not None:
        sorption: BrooksCoreyConductivity | VanGenuchtenMualemConductivity = BrooksCoreyConductivity(
            theta_r=theta_r, theta_s=theta_s, k_s=k_s, brooks_corey_lambda=brooks_corey_lambda
        )
    else:
        assert van_genuchten_n is not None  # noqa: S101  # narrowed by validation above
        sorption = VanGenuchtenMualemConductivity(
            theta_r=theta_r, theta_s=theta_s, k_s=k_s, van_genuchten_n=van_genuchten_n, mualem_l=mualem_l
        )

    # Solver-frame arrays: flow_solver = θ_s · f(t) (porosity ≡ θ_s for unsaturated KW);
    # cin_solver = q_root/f was already formed for the admissibility check above.
    flow_solver = theta_s * f

    flow_tedges_days = tedges_to_days(tedges)
    cout_tedges_days = tedges_to_days(q_water_table_tedges, ref=tedges[0])
    n_out = len(q_water_table_tedges) - 1

    q_wt_all = np.zeros((len(cumulative_pore_volumes_outlet_arr), n_out))
    structures: list[dict] = []

    for i, v_out in enumerate(cumulative_pore_volumes_outlet_arr):
        tracker = FrontTracker(
            cin=cin_solver,
            flow=flow_solver,
            tedges=tedges,
            aquifer_pore_volume=float(v_out),
            sorption=sorption,
        )
        tracker.run(max_iterations=max_iterations)

        cout_ref = _flow_weighted_front_tracking_output(
            cout_tedges_days=cout_tedges_days,
            flow_tedges_days=flow_tedges_days,
            flow=flow_solver,
            v_outlet=float(v_out),
            waves=tracker.state.waves,
            sorption=sorption,
            theta_edges=tracker.state.theta_edges,
            cin=cin_solver,
        )

        # Back-transform to physical flux: q_wt = f · cout_ref. When k_scaling is set the
        # validator requires q_water_table_tedges == tedges, so f aligns with cout_ref. When
        # k_scaling is None, f ≡ 1 and the output grid may be coarser than the input, so
        # cout_ref is used directly (multiplying by the input-length f would mis-broadcast).
        q_wt_all[i, :] = cout_ref if k_scaling is None else f * cout_ref

        structures.append({
            "waves": tracker.state.waves,
            "events": tracker.state.events,
            "theta_first_arrival": tracker.theta_first_arrival,
            "n_events": len(tracker.state.events),
            "n_shocks": sum(1 for w in tracker.state.waves if isinstance(w, ShockWave)),
            "n_rarefactions": sum(1 for w in tracker.state.waves if isinstance(w, RarefactionWave)),
            "n_characteristics": sum(1 for w in tracker.state.waves if isinstance(w, CharacteristicWave)),
            "theta_current": tracker.state.theta_current,
            "sorption": sorption,
            "tracker_state": tracker.state,
            "cumulative_pore_volume_outlet": float(v_out),
        })

    return np.mean(q_wt_all, axis=0), structures
