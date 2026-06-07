.. _concepts:

Core Concepts
=============

Groundwater transport involves the movement of solutes and heat through porous media. This guide introduces the fundamental concepts underlying ``gwtransport``.

.. _concept-pore-volume-distribution:

The Central Concept: Pore Volume Distribution
---------------------------------------------

The central innovation of ``gwtransport`` is reducing complex 3D aquifer geometry to a **pore volume distribution**. This distribution captures the essential heterogeneity of an aquifer system.

Why This Matters
~~~~~~~~~~~~~~~~

- Heterogeneous aquifers have multiple flow paths with different travel times
- Water extracted from a well is a mixture from all these paths
- The pore volume distribution describes how much of the aquifer is "fast" vs "slow"
- A gamma distribution provides a flexible, physically meaningful approximation

**Key insight:** The pore volume distribution is constant over time, while flow rates and concentrations vary. Once calibrated, the same distribution can predict transport of any solute. However, the APVD is only well-defined and time-invariant under the :ref:`steady-streamlines assumption <assumption-steady-streamlines>`: it is a property of the aquifer *for a given streamline configuration*, not an intrinsic aquifer property in general. When flow magnitude changes but streamline geometry is fixed, the APVD remains constant. When boundary conditions change in a way that redirects flow (e.g., a new pumping well activates), the streamlines rearrange and the APVD is no longer valid.

Key Parameters
~~~~~~~~~~~~~~

- **Mean pore volume**: Average volume of water in flow paths (m³)
- **Standard deviation**: Variability in pore volumes across different paths (m³)
- **Distribution shape**: Commonly approximated using a two-parameter gamma distribution

The gamma distribution model is implemented in :py:func:`gwtransport.advection.gamma_infiltration_to_extraction`. For cases with known streamline geometry, pore volumes can be computed directly and passed to :py:func:`gwtransport.advection.infiltration_to_extraction`.

For assumptions about the gamma distribution, see :ref:`assumption-gamma-distribution`.

.. _concept-residence-time:

Residence Time
~~~~~~~~~~~~~~

Residence time is the duration a water parcel (or solute) spends in the aquifer between infiltration and extraction points. For a given streamline with pore volume :math:`V` and flow rate :math:`Q`:

.. math::

   t_r = \frac{V \cdot R}{Q}

where :math:`R` is the retardation factor. Residence time depends on:

- **Pore volume** of the flow path (m³)
- **Flow rate** through the system (m³/day)
- **Retardation factor** of the compound (dimensionless)

The distribution of residence times directly reflects the pore volume distribution. Use :py:func:`gwtransport.residence_time.residence_time_full` to compute residence times from flow rates and pore volumes. See the :doc:`/examples/02_Residence_Time_Analysis` example for practical applications.

.. _concept-retardation-factor:

Retardation Factor
~~~~~~~~~~~~~~~~~~

The retardation factor :math:`R` quantifies how much slower a compound moves compared to the bulk water flow. It accounts for interactions between the transported substance and the aquifer matrix:

- **Conservative tracers** (:math:`R = 1.0`): Move at the same velocity as water (e.g., chloride, bromide, salts)
- **Temperature** (:math:`R \approx 2.0`): Retarded by heat exchange with the solid matrix; exact value depends on porosity and heat capacity ratios
- **Sorbing solutes** (:math:`R > 1`): Delayed by adsorption to aquifer materials; magnitude depends on distribution coefficient :math:`K_d`

For temperature, the retardation factor can be estimated from aquifer properties (see :doc:`/examples/01_Aquifer_Characterization_Temperature`) or calibrated alongside pore volume parameters. For reactive solutes, :math:`R = 1 + \frac{\rho_b K_d}{\theta}` where :math:`\rho_b` is bulk density and :math:`\theta` is porosity.

For assumptions about retardation, see :ref:`assumption-linear-retardation` and :ref:`assumption-thermal-retardation`.

Temperature as a Natural Tracer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temperature variations in infiltrated water serve as an effective natural tracer for aquifer characterization. Unlike artificial tracers, temperature:

- **Requires no injection**: Ambient seasonal variations provide the tracer signal
- **Enables continuous monitoring**: High-frequency temperature sensors are cost-effective
- **Has predictable behavior**: Retardation factor can be estimated from physical properties
- **Reflects transport processes**: Subject to the same advection and dispersion as solutes

The key limitation is that temperature undergoes diffusive heat exchange with the aquifer matrix, requiring a retardation factor correction. Once pore volumes are calibrated using temperature data, conservative solutes can be predicted using :math:`R = 1.0`. See :doc:`/examples/01_Aquifer_Characterization_Temperature` for a complete calibration workflow.

.. _concept-transport-physics:

Transport Physics
-----------------

.. _concept-transport-equation:

Core Transport Equation
~~~~~~~~~~~~~~~~~~~~~~~

The extracted concentration is a flow-weighted average over all flow paths:

.. math::

   C_{out}(t) = \sum_{i} w_i \cdot C_{in}(t - \tau_i)

Where:

- :math:`w_i` = weight of flow path i (from pore volume distribution)
- :math:`\tau_i` = residence time of flow path i
- :math:`C_{in}` = infiltrated concentration

This is mathematically equivalent to convolution, but implemented as discrete weighted averaging. The concentration at the extraction point is the flow-weighted average across all streamlines:

.. math::

   C_{out}(t) = \frac{\sum_i Q_i \cdot C_i(t)}{\sum_i Q_i}

where :math:`C_i(t)` is the concentration on streamline :math:`i` and :math:`Q_i` is the flow through that streamline. See :py:mod:`gwtransport.advection` for implementation details.

For assumptions about the transport framework, see :ref:`assumption-advection-dominated`, :ref:`assumption-steady-streamlines`, and :ref:`assumption-no-transverse-mixing`.

.. _concept-dispersion:

Macrodispersion and Microdispersion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key advantage of ``gwtransport`` is how it handles dispersion by distinguishing two forms:

**Macrodispersion** arises naturally from the pore volume distribution (APVD):

- Fast flow paths deliver early arrivals
- Slow flow paths deliver late arrivals
- The mixture at the well shows a "dispersed" breakthrough curve
- Depends on both aquifer properties **and** hydrological boundary conditions — when streamlines change due to changed boundary conditions, macrodispersion changes

**Microdispersion** is mechanical dispersion at the pore scale (:math:`\alpha_L`):

- Spreading within individual streamlines due to pore-scale velocity variations
- An **aquifer property** — determined by pore structure, independent of hydrological boundary conditions

**Molecular diffusion** (:math:`D_m`) is a separate process (Brownian motion), distinct from both macro- and microdispersion.

Both microdispersion and molecular diffusion can be added via the :mod:`gwtransport.diffusion` or :mod:`gwtransport.diffusion_fast` modules.

**No numerical dispersion** because:

- No spatial discretization (no grid cells)
- Analytical time integration
- Machine-precision mass balance

This is fundamentally different from traditional numerical transport models where dispersion must be parameterized separately and grid resolution affects results.

.. _concept-dispersion-scales:

Macrodispersion and Microdispersion as Scale-Dependent Heterogeneity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All spreading in porous media transport arises from **velocity heterogeneity** at different scales:

1. **Molecular scale**: Brownian motion causes spreading even in uniform flow (:math:`D_m`) — **molecular diffusion**
2. **Pore scale** (~mm to cm): Velocity varies within and between pores (:math:`\alpha_L`, longitudinal dispersivity) — **microdispersion**
3. **Aquifer scale** (~m to km): Different streamlines have different path lengths, reflecting conductivity contrasts from local layers to aquifer-wide heterogeneity (captured by APVD) — **macrodispersion**

**Microdispersion** (scale 2) is an aquifer property determined by pore structure, independent of hydrological boundary conditions. **Macrodispersion** (scale 3) depends on both aquifer properties and hydrological boundary conditions (the streamline geometry). **Molecular diffusion** (scale 1) is a separate process, distinct from both.

The boundaries between these scales are **not sharp**. This is why measured dispersivity (:math:`\alpha_L`) famously increases with experiment scale—larger measurements "see" more heterogeneity.

**gwtransport's approach:**

- The **pore volume distribution (APVD)** captures macrodispersion explicitly
- The **diffusion module** adds microdispersion (:math:`\alpha_L`) and molecular diffusion (:math:`D_m`)

**Dispersion coefficient vs. breakthrough spreading:**

The longitudinal dispersion coefficient :math:`D_L = D_m + \alpha_L \cdot v` increases with flow velocity. However, the actual spreading of a breakthrough curve at a fixed travel distance does **not** increase with flow:

- Mechanical dispersion (:math:`\alpha_L`): spatial spreading :math:`\sigma_x^2 = 2 \alpha_L L` is **independent of velocity** — the plume traverses the same pore-scale heterogeneity regardless of speed
- Molecular diffusion (:math:`D_m`): spreading **decreases** with higher flow — less time for diffusion to act

In pore volume units, :math:`\sigma_{V,disp}` is flow-independent, while :math:`\sigma_{V,diff}` decreases with higher flow (proportional to :math:`1/\sqrt{Q}`).

**When calibrating APVD from measurements**, the fitted :math:`\sigma_{apv}` is an *effective* parameter: it already lumps together macrodispersion, microdispersion, and an average molecular diffusion contribution from the calibration window, so downstream calculations treat it as the total. When calibrating with the diffusion module, these three components are tracked separately.

**Some parameters are solute-specific.** The retardation factor :math:`R` and the molecular diffusivity :math:`D_m` are properties of the transported compound, not of the aquifer geometry. A :math:`\sigma_{apv}` calibrated from measurements of one compound (e.g. temperature, :math:`R \approx 2`, :math:`D_m \approx 0.1` m²/day) therefore bakes in *that* compound's retardation and diffusion, and is not directly transferable to another (e.g. a solute, :math:`D_m \approx 10^{-4}` m²/day). For a different compound, recalibrate or re-derive its molecular-diffusion contribution :math:`\sigma_{V,\mathrm{diff}}` with the target :math:`R` and :math:`D_m`; the mechanical-dispersion term :math:`\sigma_{V,\mathrm{disp}}` depends only on geometry (:math:`\alpha_L`, :math:`L`) and carries over unchanged. For discrete-volume APVDs, apply :mod:`gwtransport.diffusion` on top of advection instead. See :ref:`theory-variance-derivations` for the :math:`\sigma_{V,\mathrm{diff}}` derivation.

**When computing APVD from streamline analysis**, only macrodispersion (aquifer-scale path length variation) is captured. Microdispersion (:math:`\alpha_L`) and molecular diffusion (:math:`D_m`) must be added using the :mod:`gwtransport.diffusion_fast` or :mod:`gwtransport.diffusion` modules:

- **Gamma-parameterized APVD**: Use :mod:`gwtransport.diffusion_fast` (standard approach).
- **Discrete streamline volumes**: Use :mod:`gwtransport.diffusion_fast` or :mod:`gwtransport.diffusion`.
- **Fast approximate forward (any flow)**: Use :mod:`gwtransport.diffusion_fast_fast` when a small error (~3e-4 with microdispersion present) is acceptable in exchange for speed.

.. _concept-variance-components:

Variance Components of Breakthrough Spreading
""""""""""""""""""""""""""""""""""""""""""""""

The total variance of the output breakthrough curve (in volume units) is the sum of three independent components. Since the streamtubes are independent, the correct micro/diff contribution is the average of the per-streamtube variances (law of total variance), not the variance evaluated at the mean pore volume:

.. list-table::
   :header-rows: 1
   :widths: 14 18 20 22 26

   * - Method
     - Macrodispersion
     - Microdispersion
     - Molecular diffusion
     - Total
   * - Mean PV
     - :math:`\sigma_{apv}^2`
     - :math:`\frac{2 \alpha_L \bar{V}^2}{L}`
     - :math:`\frac{2 D_m R \bar{V}^3}{L^2 Q}`
     - :math:`\sigma_{apv}^2 + \frac{2 \bar{V}^2}{L}\left(\alpha_L + \frac{D_m R \bar{V}}{L Q}\right)`
   * - Discrete bins
     - :math:`\sigma_{apv}^2`
     - :math:`\frac{2 \alpha_L}{L} \overline{V^2}`
     - :math:`\frac{2 D_m R}{L^2 Q} \overline{V^3}`
     - :math:`\sigma_{apv}^2 + \frac{2}{L}\left(\alpha_L \overline{V^2} + \frac{D_m R}{L Q} \overline{V^3}\right)`
   * - Gamma
     - :math:`\sigma_{apv}^2`
     - :math:`\frac{2 \alpha_L}{L}\left(\bar{V}^2 + \sigma_{apv}^2\right)`
     - :math:`\frac{2 D_m R \left(\bar{V}^2 + \sigma_{apv}^2\right)\left(\bar{V}^2 + 2\sigma_{apv}^2\right)}{L^2 Q \bar{V}}`
     - :math:`\sigma_{apv}^2 + \frac{2\left(\bar{V}^2 + \sigma_{apv}^2\right)}{L}\left(\alpha_L + \frac{D_m R \left(\bar{V}^2 + 2\sigma_{apv}^2\right)}{L Q \bar{V}}\right)`

where :math:`\overline{V^k} = \frac{1}{n}\sum_i V_i^k` is the :math:`k`-th sample moment over the discrete pore volume bins. For a gamma distribution, :math:`\mathbb{E}[V^2] = \bar{V}^2 + \sigma_{apv}^2` and :math:`\mathbb{E}[V^3] = (\bar{V}^2 + \sigma_{apv}^2)(\bar{V}^2 + 2\sigma_{apv}^2)/\bar{V}`.

Both ``diffusion_fast`` and ``diffusion`` resolve the per-streamtube breakthrough directly (the **Discrete bins** / **Gamma** treatment); the **Mean PV** column is the cruder moment-averaged alternative, shown for comparison. Per-streamtube resolution is more accurate because :math:`\sigma^2(V)` is nonlinear in :math:`V`; the Mean PV correction is of order :math:`\text{CV}^2 = (\sigma_{apv}/\bar{V})^2`.

Calibration Approaches
----------------------

Temperature Tracer Test (No Groundwater Model Needed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temperature variations in surface water propagate through the aquifer to extraction wells. By fitting modeled extraction temperature to observations, the gamma distribution parameters can be estimated.

**Advantages:**

- No artificial tracer injection required
- Uses naturally occurring temperature signals
- Continuous, low-cost monitoring with standard sensors
- Predictable thermal behavior (retardation factor ~2.0 for heat)

**Workflow:**

1. Measure T_in, T_out, Q over time
2. Optimize gamma(mean, std) to match observed extraction temperatures
3. Calibrated model ready for predictions

See :doc:`/examples/01_Aquifer_Characterization_Temperature` for a complete example.

Streamline Analysis (When Flow Model Exists)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When detailed flow field data are available (e.g., from numerical groundwater models), pore volumes can be computed directly without assuming a parametric distribution:

1. Compute streamlines from infiltration to extraction points using flow field data
2. Calculate cross-sectional areas between adjacent streamlines
3. Convert 2D streamline areas to 3D pore volumes: :math:`V_i = A_i \times d \times \theta`, where :math:`d` is aquifer depth and :math:`\theta` is porosity
4. Pass volumes directly to :py:func:`gwtransport.advection.infiltration_to_extraction`

This approach captures the actual distribution of flow paths, including multi-modal or irregular patterns that cannot be represented by a gamma distribution. The tradeoff is requiring detailed flow field information.

Model Approaches
----------------

.. _concept-gamma-distribution:

Gamma Distribution Model
~~~~~~~~~~~~~~~~~~~~~~~~

The gamma distribution provides a flexible two-parameter approximation for aquifer pore volume heterogeneity. The probability density function is:

.. math::

   f(V) = \frac{1}{\Gamma(k)\theta^k} V^{k-1} e^{-V/\theta}

where:

- :math:`k` is the shape parameter (dimensionless)
- :math:`\theta` is the scale parameter (m³)
- Mean pore volume: :math:`\mu = k \cdot \theta`
- Standard deviation: :math:`\sigma = \sqrt{k} \cdot \theta`

In practice, ``gwtransport`` parameterizes using mean and standard deviation directly (see :py:func:`gwtransport.gamma.bins`), which are more intuitive than shape and scale. The gamma model works well for moderately heterogeneous aquifers but may not capture multi-modal distributions or extreme heterogeneity.

For assumptions about the gamma distribution, see :ref:`assumption-gamma-distribution`.

.. _concept-gamma-loc:

Shifted Gamma Distribution (Minimum Pore Volume)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When every flow path has a guaranteed minimum pore volume (for example, a fixed sand-bed or gravel-pack region through which all water must travel before entering the heterogeneous aquifer), the two-parameter gamma in :ref:`concept-gamma-distribution` cannot represent this floor. ``gwtransport`` supports a three-parameter *shifted* gamma distribution with an optional location parameter ``loc`` that horizontally shifts the entire distribution by a fixed amount:

.. math::

   f(V; k, \theta, V_0) = \frac{1}{\Gamma(k)\theta^k} (V - V_0)^{k-1} e^{-(V - V_0)/\theta}, \qquad V \geq V_0

where :math:`V_0 = \text{loc}` is the minimum pore volume (m³). Mathematical properties:

- Mean pore volume: :math:`\mu = k \cdot \theta + V_0` (shifted up by ``loc``)
- Standard deviation: :math:`\sigma = \sqrt{k} \cdot \theta` (invariant under the shift)
- Support: :math:`V \in [V_0, \infty)` (the minimum pore volume is ``loc``, not zero)

In gwtransport, the constraint ``0 <= loc < mean`` is enforced. Setting ``loc = 0`` recovers the standard two-parameter gamma distribution exactly. When calibrating the three parameters, :math:`\sigma` is held fixed and the "excess" mean :math:`\mu - V_0` is re-scaled into :math:`(k, \theta)` via

.. math::

   k = \left(\frac{\mu - V_0}{\sigma}\right)^2, \qquad \theta = \frac{\sigma^2}{\mu - V_0}

For log removal analytics with a shifted residence-time distribution, the shift adds a constant :math:`\mu_\lambda \cdot (V_0/Q)` to the effective mean log removal, where :math:`\mu_\lambda` is the log10 decay rate and :math:`Q` is the flow rate. See :py:func:`gwtransport.logremoval.gamma_mean` and :py:func:`gwtransport.logremoval.gamma_find_flow_for_target_mean`.

.. _concept-nonlinear-sorption:

Non-Linear Sorption: Exact Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contaminants with concentration-dependent retardation, ``gwtransport`` provides exact analytical solutions using front-tracking. Two non-linear sorption isotherms are supported:

**Freundlich isotherm** — unbounded sorption:

.. math::

   s(C) = k_f C^{1/n}, \qquad R(C) = 1 + \frac{\rho_b k_f}{\theta n} C^{(1/n)-1}

- For n > 1 (favorable): Higher C travels faster → sharp rise, gradual decline
- For n < 1 (unfavorable): Higher C travels slower → gradual rise, sharp decline

**Langmuir isotherm** — bounded sorption with maximum capacity:

.. math::

   s(C) = s_{\max} \frac{C}{K_L + C}, \qquad R(C) = 1 + \frac{\rho_b \, s_{\max} \, K_L}{\theta \, (K_L + C)^2}

- Always favorable: R decreases with increasing C (higher C travels faster)
- R(0) is finite (no minimum-concentration threshold needed)
- R → 1 as C → ∞ (all sorption sites saturated)

**Wave physics (both isotherms):**

- **Shocks** form when faster concentrations overtake slower ones.
- **Rarefaction waves** form when concentrations spread apart.
- **Decaying shocks** form when a rarefaction fan catches up to a shock — the shock keeps moving but its strength decreases as the fan feeds into its trailing side. ``gwtransport`` resolves the decay exactly — closed form for Freundlich, Langmuir, and Brooks-Corey; quadrature for van Genuchten — so a long-running simulation does not lose accuracy as the shock weakens.

The solver tracks these waves analytically, eliminating numerical artifacts. Use :py:func:`gwtransport.advection.infiltration_to_extraction_nonlinear_sorption` for non-linear sorption (pass Freundlich or Langmuir parameters).

**Robustness under variable flow.** The wave dynamics are formulated in cumulative-flow coordinates :math:`\theta = \int Q(t')\,dt'` rather than wall-clock time, so changes in pumping rate enter only through the :math:`\theta(t)` mapping at the API boundary — wave geometries do not need to be rebuilt when :math:`Q(t)` varies. The same simulation routine handles steady, ramped, and rapidly fluctuating flow without recalibration.

**Mass conservation.** Total outlet mass is computed from the conservation identity :math:`m_{\text{out}}(\theta) = m_{\text{in}}(\theta) - m_{\text{dom}}(\theta)`, where :math:`m_{\text{dom}}` is the spatial integral of total concentration over the domain. The identity is exact, so reported mass balance is at machine precision (relative error :math:`\lesssim 10^{-13}` for typical canonical cases).

See :doc:`/examples/10_Advection_with_non_linear_sorption` for a complete example.

.. _concept-kinematic-wave:

Kinematic-Wave Percolation Through Thick Unsaturated Zones
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :mod:`gwtransport.percolation` module computes the percolation flux from the bottom of the root zone to the water table in a thick unsaturated zone. It drops the capillary-suction term from Richards' equation, retaining gravity-driven flow only — the Kinematic-Wave (KW) approximation studied by Olsthoorn (2026, *Stromingen* 32(1)) and Charbeneau (2000):

.. math::

   \frac{\partial \theta_m}{\partial t} + \frac{\partial K(\theta_m)}{\partial z} = 0,

where :math:`\theta_m(z, t)` is volumetric moisture content and :math:`K(\theta_m)` is the unsaturated hydraulic conductivity (Brooks-Corey or van Genuchten-Mualem).

**Mapping to the front-tracking framework.** The KW equation is structurally identical to the nonlinear advection-with-sorption PDE that the front-tracking solver already integrates, under the identification

- :math:`C \equiv K(\theta_m)` (flux variable)
- :math:`C_T \equiv \theta_m - \theta_r` (conserved storage, with residual offset)
- :math:`V(z) = \int_0^z n_p(z')\,dz' = \theta_s \cdot z` (cumulative pore volume per unit cross-sectional area, units of length)
- ``flow_solver(t) = θ_s · f(t)`` where :math:`f(t)` is an optional time-only K-scaling.

The mapping turns the soil's unsaturated conductivity curve into a "sorption" object (:class:`gwtransport.fronttracking.math.BrooksCoreyConductivity`, :class:`gwtransport.fronttracking.math.VanGenuchtenMualemConductivity`), and the existing wave kernel solves the full KW dynamics exactly: sharp wetting-front shocks via Rankine-Hugoniot, drying-tail rarefaction fans, decaying-shock collisions, machine-precision mass balance.

**K-scaling for viscosity.** Water viscosity ``μ(T)`` varies with temperature; since ``K_s ∝ 1/μ``, a seasonal 10 ± 5 °C swing produces roughly 30-50% variation in effective ``K_s``. The cumulative-flow trick in the front-tracking solver absorbs an arbitrary positive time-only scaling ``f(t)`` of the conductivity curve exactly:

.. math::

   \frac{\partial \theta_m}{\partial t} + n_p \cdot f(t) \cdot \frac{\partial K_{\rm ref}(\theta_m)}{\partial V} = 0
   \qquad\Longleftrightarrow\qquad
   \frac{\partial C_T}{\partial \theta_V} + \frac{\partial C}{\partial V} = 0,
   \qquad
   \theta_V(t) = \int_0^t n_p\,f(\tau)\,d\tau.

The boundary inversion ``cin_solver(t) = q_root_zone(t) / f(t)`` and back-transform ``q_water_table(t) = f(t) · cout(t)`` recover the physical flux from the reference-frame solver state.

**Forward-only.** No inverse mapping ``water_table_to_root_zone`` is provided: KW unsaturated percolation is one-way under gravity, and multiple ``q_root_zone(t)`` series produce indistinguishable ``q_water_table(t)`` after the column's intrinsic low-pass response.

See :py:func:`gwtransport.percolation.root_zone_to_water_table_kinematic_wave` and :doc:`/examples/11_Percolation_Unsaturated_Zone` for a complete walkthrough.

Comparison to Complex Transport Models
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - gwtransport
     - Full 3D Model
   * - Parameters
     - 2 (mean, std of PV)
     - Many (K, porosity, dispersivity, boundary conditions...)
   * - Calibration data needed
     - Temperature/concentration time series
     - Extensive spatial data
   * - Numerical dispersion
     - None
     - Grid-dependent
   * - Computation time
     - Seconds
     - Hours to days
   * - Uncertainty
     - Transparent (2 parameters)
     - Complex (parameter correlation)
   * - When to use
     - Initial assessment, design screening
     - Detailed site characterization

**Key insight:** More complex models require more data to constrain additional parameters. If you don't have that data, added complexity doesn't improve predictions.

What gwtransport Does NOT Do
----------------------------

1. **Does not solve flow equations** - Requires flow rates as input
2. **Does not model 3D geometry explicitly** - Reduces to pore volume distribution
3. **Does not handle reactions** - Only retardation/sorption (see :ref:`assumption-no-reactions`)
4. **Does not model multi-species interactions** - Single compound at a time
5. **Does not include density-dependent flow** - Assumes fixed streamlines (see :ref:`assumption-steady-streamlines`)

These simplifications are intentional: they make the model tractable while capturing the essential physics for many practical problems.

For a complete discussion of all assumptions and when they apply, see :doc:`assumptions`.

Applications
------------

Bank Filtration and Managed Aquifer Recharge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict pathogen removal efficiency in bank filtration systems by coupling residence time distributions with pathogen attenuation rates. See :doc:`/examples/03_Pathogen_Removal_Bank_Filtration` and :doc:`/examples/04_Deposition_Analysis_Bank_Filtration`. Use :py:func:`gwtransport.logremoval.residence_time_to_log_removal` to convert residence times to log removal values.

Contaminant Transport Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast contaminant arrival times and breakthrough curves at extraction wells. Once pore volume parameters are calibrated, predict transport of conservative solutes under varying flow conditions. Useful for risk assessment and treatment design.

Aquifer Characterization
~~~~~~~~~~~~~~~~~~~~~~~~

Estimate effective pore volume distributions from temperature tracer tests (:doc:`/examples/01_Aquifer_Characterization_Temperature`). Infer aquifer heterogeneity without costly artificial tracer tests. Validate numerical groundwater models against observed transport behavior.

Digital Twin Systems
~~~~~~~~~~~~~~~~~~~~

Implement real-time water quality monitoring by continuously updating model predictions with incoming sensor data. Enable early warning for contamination events. Support operational decisions for drinking water utilities by forecasting impacts of changing infiltration conditions.
