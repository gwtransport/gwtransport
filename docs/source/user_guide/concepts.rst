Core Concepts
=============

Groundwater transport involves the movement of solutes and heat through porous media. This guide introduces the fundamental concepts underlying ``gwtransport``.

Pore Volume Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

The pore volume distribution describes how water flows through different pathways in a heterogeneous aquifer. Aquifer heterogeneity creates preferential flow paths with varying pore volumes, leading to a distribution of travel times even under steady flow conditions.

Key parameters:

- **Mean pore volume**: Average volume of water in flow paths (m³)
- **Standard deviation**: Variability in pore volumes across different paths (m³)
- **Distribution shape**: Commonly approximated using a two-parameter gamma distribution

The gamma distribution model is implemented in :py:func:`gwtransport.advection.gamma_infiltration_to_extraction`. For cases with known streamline geometry, pore volumes can be computed directly using :py:func:`gwtransport.surfacearea.surface_area_between_streamlines` and passed to :py:func:`gwtransport.advection.distribution_infiltration_to_extraction`.

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

- **Conservative tracers** (:math:`R = 1.0`): Move at the same velocity as water (e.g., chloride, bromide)
- **Temperature** (:math:`R \approx 2.0`): Retarded by heat exchange with the solid matrix; exact value depends on porosity and heat capacity ratios
- **Sorbing solutes** (:math:`R > 1`): Delayed by adsorption to aquifer materials; magnitude depends on distribution coefficient :math:`K_d`

For temperature, the retardation factor can be estimated from aquifer properties (see :doc:`/examples/01_Aquifer_Characterization_Temperature`) or calibrated alongside pore volume parameters. For reactive solutes, :math:`R = 1 + \frac{\rho_b K_d}{\theta}` where :math:`\rho_b` is bulk density and :math:`\theta` is porosity.

Temperature as a Natural Tracer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temperature variations in infiltrated water serve as an effective natural tracer for aquifer characterization. Unlike artificial tracers, temperature:

- **Requires no injection**: Ambient seasonal variations provide the tracer signal
- **Enables continuous monitoring**: High-frequency temperature sensors are cost-effective
- **Has predictable behavior**: Retardation factor can be estimated from physical properties
- **Reflects transport processes**: Subject to the same advection and dispersion as solutes

The key limitation is that temperature undergoes diffusive heat exchange with the aquifer matrix, requiring a retardation factor correction. Once pore volumes are calibrated using temperature data, conservative solutes can be predicted using :math:`R = 1.0`. See :doc:`/examples/01_Aquifer_Characterization_Temperature` for a complete calibration workflow.

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

In practice, ``gwtransport`` parameterizes using mean and standard deviation directly (see :py:func:`gwtransport.gamma.gamma_distribution`), which are more intuitive than shape and scale. The gamma model works well for moderately heterogeneous aquifers but may not capture multi-modal distributions or extreme heterogeneity.

Streamline Analysis
~~~~~~~~~~~~~~~~~~~

When detailed flow field data are available (e.g., from numerical groundwater models), pore volumes can be computed directly without assuming a parametric distribution:

1. Compute streamlines from infiltration to extraction points using flow field data
2. Calculate cross-sectional areas between adjacent streamlines (:py:func:`gwtransport.surfacearea.surface_area_between_streamlines`)
3. Convert 2D streamline areas to 3D pore volumes: :math:`V_i = A_i \times d \times \theta`, where :math:`d` is aquifer depth and :math:`\theta` is porosity
4. Pass volumes directly to :py:func:`gwtransport.advection.distribution_infiltration_to_extraction`

This approach captures the actual distribution of flow paths, including multi-modal or irregular patterns that cannot be represented by a gamma distribution. The tradeoff is requiring detailed flow field information.

Transport Framework
~~~~~~~~~~~~~~~~~~~

``gwtransport`` uses a streamtube convolution approach where:

- **Advection** is the primary transport mechanism along discrete streamlines
- **Macroscopic dispersion** emerges naturally from the distribution of pore volumes across streamlines
- **Retardation** is applied uniformly across all streamlines via the retardation factor

This framework differs from traditional advection-dispersion equations by explicitly representing flow path heterogeneity. The concentration at the extraction point is the flow-weighted average across all streamlines:

.. math::

   C_{out}(t) = \frac{\sum_i Q_i \cdot C_i(t)}{\sum_i Q_i}

where :math:`C_i(t)` is the concentration on streamline :math:`i` and :math:`Q_i` is the flow through that streamline. See :py:mod:`gwtransport.advection` for implementation details.

Applications
------------

Bank Filtration and Managed Aquifer Recharge
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predict pathogen removal efficiency in bank filtration systems by coupling residence time distributions with pathogen attenuation rates. See :doc:`/examples/03_Pathogen_Removal_Bank_Filtration` and :doc:`/examples/04_Deposition_Analysis_Bank_Filtration`. Use :py:func:`gwtransport.logremoval.residence_time_to_log_removal` to convert residence times to log removal values.

Contaminant Transport Forecasting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecast contaminant arrival times and breakthrough curves at extraction wells. Once pore volume parameters are calibrated, predict transport of conservative solutes under varying flow conditions. Useful for risk assessment and treatment design.

Aquifer Characterization
~~~~~~~~~~~~~~~~~~~~~~~~~

Estimate effective pore volume distributions from temperature tracer tests (:doc:`/examples/01_Aquifer_Characterization_Temperature`). Infer aquifer heterogeneity without costly artificial tracer tests. Validate numerical groundwater models against observed transport behavior.

Digital Twin Systems
~~~~~~~~~~~~~~~~~~~~

Implement real-time water quality monitoring by continuously updating model predictions with incoming sensor data. Enable early warning for contamination events. Support operational decisions for drinking water utilities by forecasting impacts of changing infiltration conditions.