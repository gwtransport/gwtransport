Core Concepts
=============

Groundwater transport involves the movement of solutes and heat through porous media. This guide introduces the fundamental concepts underlying ``gwtransport``.

The Central Concept: Pore Volume Distribution
---------------------------------------------

The central innovation of ``gwtransport`` is reducing complex 3D aquifer geometry to a **pore volume distribution**. This distribution captures the essential heterogeneity of an aquifer system.

Why This Matters
~~~~~~~~~~~~~~~~

- Heterogeneous aquifers have multiple flow paths with different travel times
- Water extracted from a well is a mixture from all these paths
- The pore volume distribution describes how much of the aquifer is "fast" vs "slow"
- A gamma distribution provides a flexible, physically meaningful approximation

**Key insight:** The pore volume distribution is constant over time (it's a property of the aquifer geometry), while flow rates and concentrations vary. Once calibrated, the same distribution can predict transport of any solute.

Key Parameters
~~~~~~~~~~~~~~

- **Mean pore volume**: Average volume of water in flow paths (m³)
- **Standard deviation**: Variability in pore volumes across different paths (m³)
- **Distribution shape**: Commonly approximated using a two-parameter gamma distribution

The gamma distribution model is implemented in :py:func:`gwtransport.advection.gamma_infiltration_to_extraction`. For cases with known streamline geometry, pore volumes can be computed directly using :py:func:`gwtransport.surfacearea.compute_average_heights` and passed to :py:func:`gwtransport.advection.infiltration_to_extraction`.

For assumptions about the gamma distribution, see :ref:`assumption-gamma-distribution`.

Residence Time
~~~~~~~~~~~~~~

Residence time is the duration a water parcel (or solute) spends in the aquifer between infiltration and extraction points. For a given streamline with pore volume :math:`V` and flow rate :math:`Q`:

.. math::

   t_r = \frac{V \cdot R}{Q}

where :math:`R` is the retardation factor. Residence time depends on:

- **Pore volume** of the flow path (m³)
- **Flow rate** through the system (m³/day)
- **Retardation factor** of the compound (dimensionless)

The distribution of residence times directly reflects the pore volume distribution. Use :py:func:`gwtransport.residence_time.residence_time` to compute residence times from flow rates and pore volumes. See the :doc:`/examples/02_Residence_Time_Analysis` example for practical applications.

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

Transport Physics
-----------------

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

Dispersion Without Numerical Dispersion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key advantage of ``gwtransport`` is how it handles dispersion:

**Macroscopic dispersion** arises naturally from the pore volume distribution:

- Fast flow paths deliver early arrivals
- Slow flow paths deliver late arrivals
- The mixture at the well shows a "dispersed" breakthrough curve

**No numerical dispersion** because:

- No spatial discretization (no grid cells)
- Analytical time integration
- Machine-precision mass balance

This is fundamentally different from traditional numerical transport models where dispersion must be parameterized separately and grid resolution affects results.

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
2. Calculate cross-sectional areas between adjacent streamlines (:py:func:`gwtransport.surfacearea.compute_average_heights`)
3. Convert 2D streamline areas to 3D pore volumes: :math:`V_i = A_i \times d \times \theta`, where :math:`d` is aquifer depth and :math:`\theta` is porosity
4. Pass volumes directly to :py:func:`gwtransport.advection.infiltration_to_extraction`

This approach captures the actual distribution of flow paths, including multi-modal or irregular patterns that cannot be represented by a gamma distribution. The tradeoff is requiring detailed flow field information.

Model Approaches
----------------

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

Non-Linear Sorption: Exact Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For contaminants with concentration-dependent retardation (Freundlich isotherm), ``gwtransport`` provides exact analytical solutions using front-tracking:

.. math::

   R(C) = 1 + \frac{\rho_b}{\theta n} k_f C^{(1/n)-1}

**Wave physics:**

- **Shocks** form when faster concentrations overtake slower ones
- **Rarefaction waves** form when concentrations spread apart
- For n > 1 (favorable sorption): Sharp rise, gradual decline
- For n < 1 (unfavorable sorption): Gradual rise, sharp decline

The solver tracks these waves analytically, eliminating numerical artifacts. Use :py:func:`gwtransport.advection.infiltration_to_extraction_front_tracking` for non-linear sorption.

See :doc:`/examples/10_Advection_with_non_linear_sorption` for a complete example.

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
