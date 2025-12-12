.. _assumptions:

Assumptions
===========

This page provides a complete overview of the assumptions made by the ``gwtransport`` package. Understanding these assumptions helps practitioners determine when ``gwtransport`` is appropriate for their application and how to validate results.

.. contents:: Contents
   :local:
   :depth: 2

Module Reference
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Module
     - Primary Functions
   * - ``advection``
     - :py:func:`~gwtransport.advection.gamma_infiltration_to_extraction`, :py:func:`~gwtransport.advection.infiltration_to_extraction`, :py:func:`~gwtransport.advection.infiltration_to_extraction_front_tracking`
   * - ``residence_time``
     - :py:func:`~gwtransport.residence_time.residence_time`, :py:func:`~gwtransport.residence_time.residence_time_mean`, :py:func:`~gwtransport.residence_time.freundlich_retardation`
   * - ``deposition``
     - :py:func:`~gwtransport.deposition.deposition_to_extraction`, :py:func:`~gwtransport.deposition.extraction_to_deposition`
   * - ``logremoval``
     - :py:func:`~gwtransport.logremoval.residence_time_to_log_removal`, :py:func:`~gwtransport.logremoval.gamma_mean`, :py:func:`~gwtransport.logremoval.parallel_mean`
   * - ``diffusion``
     - :py:func:`~gwtransport.diffusion.infiltration_to_extraction`
   * - ``gamma``
     - :py:func:`~gwtransport.gamma.bins`, :py:func:`~gwtransport.gamma.mean_std_to_alpha_beta`

Physical/Hydrogeological Assumptions
------------------------------------

.. _assumption-advection-dominated:

1. Advection-Dominated Transport
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** Molecular diffusion and mechanical dispersion are negligible compared to advective spreading caused by aquifer heterogeneity.

**Applies to:** ``advection`` module (all functions), ``residence_time``, ``deposition``, ``logremoval``

**Does NOT apply when using:** ``diffusion`` module (which explicitly adds diffusive spreading)

**What this means:** The spreading of solute plumes is dominated by the variation in flow path lengths (different streamlines have different travel times), not by diffusive/dispersive processes within individual streamlines.

**When it holds:**

- Heterogeneous aquifers with significant pore volume variability
- Moderate to high flow velocities
- Short to moderate flow path lengths

**When it fails:**

- Homogeneous aquifers (low advective spreading)
- Very long flow paths (diffusion accumulates)
- Low flow velocities (more time for diffusion)
- High diffusivity materials

**Testable:** Yes - variance ratio test

.. math::

   \frac{\sigma^2_{\text{diffusion}}}{\sigma^2_{\text{advection}}} < 0.1 \implies \text{Diffusion negligible}

.. _assumption-steady-streamlines:

2. Steady Streamlines
~~~~~~~~~~~~~~~~~~~~~

**Assumption:** The geometry of flow paths (streamlines) remains fixed over time; only flow rates vary.

**Applies to:** All modules (``advection``, ``residence_time``, ``deposition``, ``logremoval``, ``diffusion``)

**What this means:** When pumping rate doubles, water moves twice as fast along the same paths—the paths themselves don't change. The pore volume distribution is a time-invariant property of the aquifer geometry.

**When it holds:**

- Uniform scaling of flow field with pumping rate
- Stable boundary conditions (river stage, recharge patterns)
- No significant changes in aquifer properties

**When it fails:**

- Changing boundary conditions that redirect flow
- Multiple pumping wells with varying schedules
- Seasonal changes in recharge distribution
- Density-dependent flow (saltwater intrusion)

**Testable:** Yes - cross-validation across flow regimes. Calibrate on high-flow periods, validate on low-flow periods (or vice versa). If the same pore volume distribution works across flow regimes, the assumption holds.

.. _assumption-saturated-flow:

3. Saturated Flow
~~~~~~~~~~~~~~~~~

**Assumption:** The aquifer is fully saturated; there is no vadose (unsaturated) zone transport.

**Applies to:** All modules

**What this means:** All pore space is filled with water. The package does not model variably saturated conditions, capillary effects, or air-water interfaces.

**When it holds:**

- Confined aquifers
- Unconfined aquifers below the water table
- Bank filtration systems with permanent saturation

**When it fails:**

- Vadose zone transport
- Perched aquifers
- Intermittently saturated zones
- Artificial recharge through unsaturated zone

**Testable:** No - this defines the domain of applicability. If your system includes significant unsaturated zone transport, ``gwtransport`` is not the appropriate tool.

.. _assumption-single-porosity:

4. Single Porosity
~~~~~~~~~~~~~~~~~~

**Assumption:** All water flows through a single connected pore network; there is no matrix diffusion or dual-porosity behavior.

**Applies to:** All modules

**What this means:** Water in the aquifer is mobile and participates in flow. There are no stagnant zones where solutes can diffuse in and out, causing tailing.

**When it holds:**

- Granular aquifers (sand, gravel)
- Well-connected pore networks
- Systems where matrix diffusion timescales >> residence times

**When it fails:**

- Fractured rock aquifers
- Karst systems
- Clay-rich aquitards with diffusive exchange
- Dual-porosity media

**Testable:** No - this defines the domain of applicability. Characteristic signature of dual-porosity: extended tailing in breakthrough curves that cannot be explained by pore volume heterogeneity alone.

.. _assumption-no-reactions:

5. No Reactions or Decay
~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** Transport is conservative (no mass loss or gain). The package models only physical transport processes (advection, sorption), not chemical or biological reactions.

**Applies to:** ``advection``, ``residence_time``, ``deposition``, ``diffusion``

**Partially relaxed in:** ``logremoval`` module, which applies empirical decay coefficients as post-processing to residence times

**What this means:** What infiltrates will eventually be extracted (accounting for retardation). There is no:

- First-order decay (pathogens, radionuclides)
- Biodegradation
- Chemical reactions (precipitation, dissolution, redox)
- Volatilization

**When it holds:**

- Conservative tracers (chloride, bromide)
- Temperature (thermal transport)
- Solutes that are stable over residence time scales
- Short residence times where decay is minimal

**When it fails:**

- Pathogen transport (die-off occurs)
- Biodegradable contaminants
- Redox-sensitive species
- Radionuclides with significant decay

**Testable:** Partially - compare mass balance between infiltration and extraction. If extracted mass << infiltrated mass (after accounting for retardation and mixing), reactions/decay are significant.

.. note::

   The log-removal calculations in the :doc:`/examples/03_Pathogen_Removal_Bank_Filtration` example use an empirical rate coefficient to represent pathogen inactivation. This is a post-processing step, not part of the core transport model. For reactive transport, ``gwtransport`` predictions are conservative (over-predict concentrations).

.. _assumption-no-transverse-mixing:

6. No Transverse Mixing
~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** Each streamtube transports independently; there is no mixing or dispersion between adjacent streamlines.

**Applies to:** All modules (fundamental to the streamtube approach)

**What this means:** A parcel of water stays on its streamline from infiltration to extraction. Concentration differences between streamlines are preserved.

**When it holds:**

- Stratified aquifers
- Laminar flow conditions
- Short to moderate travel distances

**When it fails:**

- Highly heterogeneous aquifers with tortuous flow
- Long travel distances allowing transverse dispersion
- Turbulent flow (rare in porous media)

**Testable:** Difficult - requires spatially distributed concentration data. Indirect test: if model systematically over-predicts peak concentrations and under-predicts tails, transverse mixing may be significant.

.. _assumption-incompressible-flow:

7. Incompressible Flow
~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** Water is incompressible; mass is conserved; flow in equals flow out.

**Applies to:** All modules (fundamental assumption)

**What this means:** There is no storage change in the aquifer over the timescales of interest. The flow rate measured at extraction represents the flow through the entire system.

**When it holds:**

- Steady-state or quasi-steady flow
- Confined aquifers with minimal storage
- Timescales longer than aquifer response time

**When it fails:**

- Rapid transients (pumping tests)
- Significant aquifer storage effects
- Compressible aquifer materials

**Testable:** No - this is a fundamental assumption of the streamtube approach.

Model Parameterization Assumptions
----------------------------------

.. _assumption-gamma-distribution:

8. Gamma Distribution Adequacy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** The two-parameter gamma distribution adequately represents the actual pore volume heterogeneity.

**Applies to:** Functions with ``gamma_`` prefix only:

- :py:func:`~gwtransport.advection.gamma_infiltration_to_extraction`
- :py:func:`~gwtransport.advection.gamma_extraction_to_infiltration`
- :py:func:`~gwtransport.logremoval.gamma_mean`, :py:func:`~gwtransport.logremoval.gamma_pdf`, :py:func:`~gwtransport.logremoval.gamma_cdf`, :py:func:`~gwtransport.logremoval.gamma_find_flow_for_target_mean`

**Does NOT apply to:**

- :py:func:`~gwtransport.advection.infiltration_to_extraction` — accepts explicit pore volumes
- :py:func:`~gwtransport.advection.infiltration_to_extraction_front_tracking` — accepts explicit pore volumes

**What this means:** The shape of the pore volume distribution can be captured by specifying only mean and standard deviation. The gamma distribution is unimodal and right-skewed.

**When it holds:**

- Moderately heterogeneous aquifers
- Unimodal pore volume distributions
- When detailed pore volume data are not available

**When it fails:**

- Multi-modal distributions (e.g., multiple aquifer layers)
- Extreme heterogeneity
- When streamline analysis provides actual pore volumes

**Testable:** Yes

- If pore volumes are known: Kolmogorov-Smirnov test, Q-Q plot
- If only time series available: Check residuals for systematic patterns suggesting distribution mismatch

**Alternative:** Use :py:func:`~gwtransport.advection.infiltration_to_extraction` with explicit pore volumes from streamline analysis instead of :py:func:`~gwtransport.advection.gamma_infiltration_to_extraction`.

.. _assumption-linear-retardation:

9. Linear Retardation
~~~~~~~~~~~~~~~~~~~~~

**Assumption:** The retardation factor R is constant and does not depend on concentration.

**Applies to:** Most functions that accept a ``retardation_factor`` parameter:

- :py:func:`~gwtransport.advection.gamma_infiltration_to_extraction`, :py:func:`~gwtransport.advection.infiltration_to_extraction`
- :py:func:`~gwtransport.advection.gamma_extraction_to_infiltration`, :py:func:`~gwtransport.advection.extraction_to_infiltration`
- :py:func:`~gwtransport.residence_time.residence_time`, :py:func:`~gwtransport.residence_time.residence_time_mean`
- :py:func:`~gwtransport.deposition.deposition_to_extraction`, :py:func:`~gwtransport.deposition.extraction_to_deposition`
- :py:func:`~gwtransport.diffusion.infiltration_to_extraction`

**Does NOT apply to:**

- :py:func:`~gwtransport.advection.infiltration_to_extraction_front_tracking` — supports Freundlich (non-linear) sorption
- :py:func:`~gwtransport.residence_time.freundlich_retardation` — computes concentration-dependent R

**What this means:** All concentrations travel at the same retarded velocity. The sorption isotherm is linear.

**When it holds:**

- Conservative tracers (R = 1)
- Temperature (R ≈ 2, constant)
- Linear sorption isotherms
- Low concentration ranges where isotherms are approximately linear

**When it fails:**

- Non-linear (Freundlich, Langmuir) sorption
- High concentrations approaching sorption capacity
- Concentration-dependent retardation

**Testable:** Yes - compare predictions with constant R vs. concentration-dependent R (Freundlich). The package supports non-linear sorption via :py:func:`~gwtransport.advection.infiltration_to_extraction_front_tracking`.

.. note::

   ``gwtransport`` provides exact solutions for Freundlich sorption using front-tracking. If non-linear sorption is suspected, use the front-tracking functions instead of assuming linear retardation.

.. _assumption-thermal-retardation:

10. Thermal Retardation Factor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** For temperature transport, the retardation factor R ≈ 2.0 (or another specified value) is appropriate.

**Applies to:** Any module when used for temperature transport (user must specify appropriate R)

**What this means:** Heat exchanges with the aquifer matrix, causing temperature signals to travel slower than water. The retardation factor depends on the ratio of heat capacities:

.. math::

   R = 1 + \frac{(1-n) \rho_s c_s}{n \rho_w c_w}

where n is porosity, ρ is density, and c is specific heat capacity.

**When it holds:**

- Known or typical aquifer thermal properties
- Uniform thermal properties along flow path

**When it fails:**

- Highly variable thermal properties
- Aquifers with unusual mineralogy
- Very high or low porosity

**Testable:** Yes - calibrate R as a free parameter and check if:

1. The optimized value is physically plausible (typically 1.5-3.0)
2. The confidence interval is reasonable
3. The value is consistent with known thermal properties

Data and Input Assumptions
--------------------------

.. _assumption-representative-input:

11. Representative Input Signal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** The measured infiltration concentration/temperature represents the actual input to the aquifer system.

**Applies to:** All modules that take ``cin`` (infiltration concentration) as input:

- ``advection`` module (all forward functions)
- :py:func:`~gwtransport.diffusion.infiltration_to_extraction`
- :py:func:`~gwtransport.deposition.deposition_to_extraction` (for deposition rates)

**What this means:** The monitoring point captures the true input signal without significant bias, lag, or spatial averaging issues.

**When it holds:**

- Monitoring point is at or near infiltration location
- Infiltration zone is spatially uniform
- No significant lag between surface water and infiltration

**When it fails:**

- Monitoring distant from actual infiltration
- Spatially heterogeneous infiltration zone
- Significant travel time through riverbed/bank
- Multiple infiltration sources with different signals

**Testable:** Yes - sensitivity analysis. Perturb input signal timing (shift by days) and magnitude (scale by %) and assess impact on predictions.

.. _assumption-well-mixed-extraction:

12. Well-Mixed Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** All streamlines contribute to extracted water simultaneously as a flow-weighted average.

**Applies to:** All modules that compute extraction concentrations:

- ``advection`` module (all functions computing ``cout``)
- :py:func:`~gwtransport.diffusion.infiltration_to_extraction`
- :py:func:`~gwtransport.deposition.deposition_to_extraction`
- ``logremoval`` module (log-removal averaged across flow paths)

**What this means:** The extraction well integrates flow from all contributing streamlines. There is no vertical stratification or preferential flow to specific screen intervals.

**When it holds:**

- Fully screened wells
- Long well screens relative to aquifer thickness
- Vertically uniform aquifer properties

**When it fails:**

- Partially penetrating wells
- Short screens in thick aquifers
- Vertically stratified aquifers
- Point extraction (e.g., piezometers)

**Testable:** Difficult - requires multi-depth sampling or flow profiling. Compare predictions for different assumed screen configurations if data available.

.. _assumption-adequate-time-resolution:

13. Adequate Time Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** The input data time resolution captures the relevant dynamics of the system.

**Applies to:** All modules that use ``tedges`` for temporal discretization

**What this means:** Temporal discretization (tedges) is fine enough that sub-interval variations don't significantly affect results.

**When it holds:**

- Daily data for systems with residence times >> days
- High-frequency data for rapid transients

**When it fails:**

- Coarse temporal resolution relative to residence time
- Rapidly varying input signals
- Short residence times with sub-daily dynamics

**Testable:** Yes - resample input data to coarser resolution and compare results. If results change significantly, finer resolution may be needed.

Numerical Implementation
------------------------

.. _assumption-adequate-discretization:

14. Adequate Discretization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Assumption:** The number of bins (n_bins) used to discretize the gamma distribution is sufficient for accurate results.

**Applies to:** Functions with ``gamma_`` prefix that use ``n_bins`` parameter:

- :py:func:`~gwtransport.advection.gamma_infiltration_to_extraction`
- :py:func:`~gwtransport.advection.gamma_extraction_to_infiltration`
- :py:func:`~gwtransport.gamma.bins`

**Does NOT apply to:**

- Functions using explicit pore volume arrays (already discretized by user)
- ``logremoval.gamma_*`` functions (use analytical formulas, not discretization)

**What this means:** The continuous gamma distribution is approximated by discrete bins. Too few bins may cause numerical artifacts; too many is computationally wasteful.

**When it holds:**

- n_bins ≥ 50 for most applications
- Default n_bins = 100 is usually adequate

**When it fails:**

- Very narrow distributions (high precision needed)
- Very wide distributions (need coverage of tails)
- Sharp input signals requiring fine temporal resolution

**Testable:** Yes - convergence test. Run with n_bins = 50, 100, 200, 500 and check that results converge. If results change significantly between 100 and 200, use more bins.

Summary: When is gwtransport Appropriate?
-----------------------------------------

Good Candidates
~~~~~~~~~~~~~~~

- Bank filtration systems
- Managed aquifer recharge in granular aquifers
- Alluvial aquifers with moderate heterogeneity
- Systems where advection dominates dispersion
- Conservative tracer or temperature transport
- Preliminary assessment before detailed modeling

Poor Candidates
~~~~~~~~~~~~~~~

- Fractured rock or karst systems
- Vadose zone transport
- Reactive contaminants with significant decay
- Highly transient flow with changing streamline geometry
- Dual-porosity systems with matrix diffusion

When in Doubt
~~~~~~~~~~~~~

1. Start with ``gwtransport`` as a first assessment
2. Test key assumptions (especially :ref:`advection dominance <assumption-advection-dominated>` and :ref:`gamma adequacy <assumption-gamma-distribution>`)
3. If assumptions fail or predictions don't match observations, consider more complex models
4. Use ``gwtransport`` results to identify which additional complexity is needed

Testing Framework
-----------------

The assumptions can be organized into testable categories:

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Category
     - Assumptions
     - Modules Affected
     - Test Methods
   * - Domain Applicability
     - :ref:`Saturated flow <assumption-saturated-flow>`, :ref:`Single porosity <assumption-single-porosity>`, :ref:`No reactions <assumption-no-reactions>`
     - All
     - Not testable - defines when to use gwtransport
   * - Transport Physics
     - :ref:`Advection-dominated <assumption-advection-dominated>`
     - All except ``diffusion``
     - Variance ratio
   * - Transport Physics
     - :ref:`Steady streamlines <assumption-steady-streamlines>`, :ref:`No transverse mixing <assumption-no-transverse-mixing>`, :ref:`Incompressible <assumption-incompressible-flow>`
     - All
     - Cross-validation, Residual analysis
   * - Parameterization
     - :ref:`Gamma adequate <assumption-gamma-distribution>`
     - ``gamma_*`` functions only
     - Goodness-of-fit, Q-Q plot
   * - Parameterization
     - :ref:`Linear retardation <assumption-linear-retardation>`
     - All except ``front_tracking``
     - Compare with non-linear model
   * - Parameterization
     - :ref:`Thermal R <assumption-thermal-retardation>`
     - Temperature applications
     - Sensitivity analysis, Physical bounds
   * - Data Quality
     - :ref:`Representative input <assumption-representative-input>`, :ref:`Well-mixed extraction <assumption-well-mixed-extraction>`, :ref:`Time resolution <assumption-adequate-time-resolution>`
     - All
     - Sensitivity analysis, Resampling
   * - Numerical
     - :ref:`Adequate discretization <assumption-adequate-discretization>`
     - ``gamma_*`` functions only
     - Convergence tests

.. See Example 5 (Testing Assumptions) for implementation of these tests.

Quick Reference: Assumptions by Module
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Module
     - Key Assumptions
     - Can Be Relaxed By
   * - ``advection`` (gamma functions)
     - :ref:`Gamma distribution <assumption-gamma-distribution>`, :ref:`Linear R <assumption-linear-retardation>`, :ref:`Advection-dominated <assumption-advection-dominated>`
     - Use explicit pore volumes, front-tracking
   * - ``advection`` (explicit pore volumes)
     - :ref:`Linear R <assumption-linear-retardation>`, :ref:`Advection-dominated <assumption-advection-dominated>`
     - Use front-tracking for non-linear R
   * - ``advection`` (front-tracking)
     - :ref:`Advection-dominated <assumption-advection-dominated>`
     - — (supports non-linear sorption)
   * - ``residence_time``
     - :ref:`Steady streamlines <assumption-steady-streamlines>`, :ref:`Linear R <assumption-linear-retardation>`
     - Use :py:func:`~gwtransport.residence_time.freundlich_retardation` for non-linear
   * - ``deposition``
     - :ref:`Steady streamlines <assumption-steady-streamlines>`, :ref:`Linear R <assumption-linear-retardation>`, :ref:`No decay <assumption-no-reactions>`
     - —
   * - ``logremoval``
     - :ref:`Steady streamlines <assumption-steady-streamlines>`
     - — (adds empirical decay)
   * - ``diffusion``
     - :ref:`Steady streamlines <assumption-steady-streamlines>`
     - — (relaxes advection-only assumption)
