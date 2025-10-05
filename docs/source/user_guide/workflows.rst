Common Workflows
================

This section describes typical workflows for using ``gwtransport`` in different scenarios. Each workflow is illustrated with code examples and links to complete demonstrations.

Temperature Tracer Test Workflow
--------------------------------

Calibrate aquifer pore volume distribution using temperature measurements from infiltration and extraction points. Temperature provides a continuous, non-invasive tracer signal for aquifer characterization.

**See the complete example:** :doc:`/examples/01_Aquifer_Characterization_Temperature`

Step 1: Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~~

Organize temperature and flow measurements into the format required by ``gwtransport``. Time series are represented using ``tedges`` (time edges) with values defined between consecutive edges.

.. code-block:: python

   import numpy as np
   import pandas as pd
   from gwtransport.advection import gamma_infiltration_to_extraction

   # Load temperature and flow data
   data = pd.read_csv('temperature_data.csv', parse_dates=['time'])  # doctest: +SKIP

   # Extract time series (values represent intervals between tedges)
   cin_data = data['temp_infiltration'].values  # °C
   cout_observed = data['temp_extraction'].values  # °C (for calibration)
   flow_data = data['flow_rate'].values  # m³/day

   # Create time edges (one more element than data arrays)
   tedges = pd.DatetimeIndex(data['time'])

Step 2: Model Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize pore volume parameters (mean, std) and temperature retardation factor by minimizing mismatch between observed and predicted extraction temperatures. See :py:func:`gwtransport.advection.gamma_infiltration_to_extraction` for the forward model.

.. code-block:: python

   from scipy.optimize import minimize

   def objective_function(params):
       mean_vol, std_vol, retardation = params

       # Compute model prediction
       cout_model = gamma_infiltration_to_extraction(
           cin=cin_data,
           flow=flow_data,
           tedges=tedges,
           cout_tedges=tedges,
           mean=mean_vol,
           std=std_vol,
           retardation_factor=retardation,
       )

       # Sum of squared errors (skip NaN values at start)
       mask = ~np.isnan(cout_model)
       error = np.sum((cout_model[mask] - cout_observed[mask]) ** 2)
       return error

   # Initial guess: [mean (m³), std (m³), retardation (-)]
   initial_params = [30000.0, 8100.0, 2.0]

   # Optimize using Nelder-Mead (derivative-free)
   result = minimize(objective_function, initial_params, method='Nelder-Mead')
   mean_opt, std_opt, R_opt = result.x
   print(f"Optimized: mean={mean_opt:.0f} m³, std={std_opt:.0f} m³, R={R_opt:.2f}")

Step 3: Prediction for Conservative Solutes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use calibrated pore volumes to predict transport of conservative solutes (e.g., contaminants). Conservative solutes have :math:`R = 1.0` unlike temperature's :math:`R \approx 2.0`.

.. code-block:: python

   # Predict conservative solute transport using calibrated pore volumes
   # but with retardation_factor=1.0
   solute_concentration = gamma_infiltration_to_extraction(
       cin=cin_solute,  # Solute concentration at infiltration [mg/L]
       flow=flow_data,
       tedges=tedges,
       cout_tedges=tedges,
       mean=mean_opt,  # Use calibrated mean pore volume
       std=std_opt,    # Use calibrated std pore volume
       retardation_factor=1.0,  # Conservative tracer (not temperature!)
   )

The key insight: calibrate using temperature (R ≈ 2.0), then predict solutes (R = 1.0) using the same pore volume distribution. See :doc:`/examples/02_Residence_Time_Analysis` for applications.

Streamline Analysis Workflow
----------------------------

Compute pore volumes directly from numerical flow fields when detailed groundwater models are available. This bypasses the gamma distribution assumption.

Step 1: Extract Pore Volumes from Flow Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gwtransport.advection import infiltration_to_extraction

   # Compute streamline geometry from your flow model
   # (implementation depends on your modeling software)
   streamline_coords = extract_streamlines_from_model(flow_model)

   # Calculate cross-sectional areas between adjacent streamlines
   areas = surface_area_between_streamlines(streamline_coords)

   # Convert 2D areas to 3D pore volumes
   depth_aquifer = 200.0  # [m] vertical extent
   porosity = 0.35  # [-] effective porosity
   aquifer_pore_volumes = areas * depth_aquifer * porosity

Step 2: Transport Calculation Without Distribution Assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use actual pore volume distribution (no gamma assumption)
   cout = infiltration_to_extraction(
       cin=cin_data,
       flow=flow_data,
       tedges=tedges,
       cout_tedges=tedges,
       aquifer_pore_volumes=aquifer_pore_volumes,
       retardation_factor=1.0,
   )

This approach is more accurate when the true pore volume distribution is multi-modal or highly irregular. See :py:func:`gwtransport.advection.infiltration_to_extraction` for details.

Residence Time Analysis Workflow
--------------------------------

Compute residence time distributions to understand water age and assess treatment effectiveness. Residence times are essential for pathogen removal calculations.

**See the complete example:** :doc:`/examples/02_Residence_Time_Analysis`

Step 1: Compute Residence Times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

   import pandas as pd
   import numpy as np
   from gwtransport.residence_time import residence_time

   # Example data
   tedges = pd.date_range("2023-01-01", "2023-01-05", freq="D")
   flow_data = np.array([100.0, 120.0, 110.0, 105.0])  # m³/day
   mean_opt = 250.0  # m³

   # Compute residence time at each time step
   rt = residence_time(
       flow=flow_data,
       flow_tedges=tedges,
       aquifer_pore_volume=mean_opt,  # Can be scalar or array
       retardation_factor=1.0,  # For conservative tracers
       direction='extraction_to_infiltration',
       index=tedges,
   )

   print(f"Residence times (days): {rt}")

.. testoutput::
   :options: +ELLIPSIS

   Residence times (days): ...

For gamma-distributed pore volumes, the mean residence time is :math:`\overline{t_r} = \frac{\mu \cdot R}{Q}` where :math:`\mu` is the mean pore volume.

Step 2: Scenario Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluate how residence times change under varying operational conditions:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Define flow scenarios
   scenarios = {
       'Low flow (50%)': flow_data * 0.5,
       'Normal flow': flow_data,
       'High flow (200%)': flow_data * 2.0,
   }

   # Compute residence times for each scenario
   for name, flows in scenarios.items():
       rt_scenario = residence_time(
           flow=flows,
           flow_tedges=tedges,
           aquifer_pore_volume=mean_opt,
           retardation_factor=1.0,
           index=tedges,
       )
       plt.plot(tedges, rt_scenario, label=name)

   plt.xlabel('Time')
   plt.ylabel('Residence Time [days]')
   plt.legend()

Residence time directly impacts pathogen removal efficiency (next section).

Pathogen Removal Analysis Workflow
----------------------------------

Assess pathogen removal efficiency in bank filtration systems by combining residence time distributions with pathogen attenuation rates.

**See complete examples:** :doc:`/examples/03_Pathogen_Removal_Bank_Filtration` and :doc:`/examples/04_Deposition_Analysis_Bank_Filtration`

Step 1: Compute Log Removal from Residence Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gwtransport.logremoval import residence_time_to_log_removal

   # Define pathogen-specific log removal rate
   # Typical values: 0.5-2.0 for bacteria, 1.0-3.0 for viruses
   log_removal_rate = 1.5  # [dimensionless]

   # Compute log removal from residence times
   log_removal = residence_time_to_log_removal(
       residence_times=rt,  # [days] from previous section
       log_removal_rate=log_removal_rate,
   )

Log removal represents orders of magnitude reduction: LR=3 means 99.9% (3-log) removal.

Step 2: Assess Treatment Effectiveness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert log removal to removal efficiency percentage
   removal_efficiency = 1 - 10**(-log_removal)

   # Check compliance with treatment targets
   target_log_removal = 4.0  # Example: 4-log virus removal requirement
   meets_target = log_removal >= target_log_removal

   print(f"Log removal: {log_removal:.2f}")
   print(f"Removal efficiency: {removal_efficiency:.2%}")
   print(f"Meets 4-log target: {meets_target}")

For gamma-distributed residence times, use :py:func:`gwtransport.logremoval.gamma_mean` to compute mean log removal analytically, or :py:func:`gwtransport.logremoval.gamma_find_flow_for_target_mean` to determine required flow rates for treatment targets.

Best Practices
--------------

Data Quality and Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**High-resolution measurements**: Temperature and flow data should have sufficient temporal resolution to capture dynamic variations. Daily or sub-daily measurements are typically required.

**Handle missing data**: Use :py:func:`gwtransport.utils.linear_interpolate` for small gaps, but avoid interpolating across long periods that may introduce bias.

**Validate sensor accuracy**: Temperature sensor drift can systematically bias calibrated parameters. Cross-check against independent measurements.

**Account for seasonal cycles**: Ensure calibration data spans sufficient time to capture seasonal temperature variations in natural systems.

Model Selection and Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Choose appropriate parameterization**: Use gamma distribution for simple cases; use direct pore volume distributions when flow heterogeneity is complex or multi-modal.

**Validate on independent data**: Reserve a portion of data for validation. Calibrate on one time period, validate on another.

**Check residual patterns**: Systematic residuals indicate model structural error. Random residuals suggest adequate model complexity.

**Sensitivity analysis**: Test how predictions change with parameter variations. Identify which parameters most strongly influence results.

**Physical plausibility**: Verify that calibrated parameters are physically reasonable (e.g., pore volumes consistent with aquifer geometry, retardation factors within expected ranges).

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Parameter uncertainty**: Use ensemble methods or Bayesian calibration to quantify uncertainty in mean, std, and retardation factor.

**Propagate uncertainty**: Run models with parameter samples to generate prediction intervals, not just point predictions.

**Report limitations**: Document model assumptions (e.g., gamma distribution, neglecting transverse dispersion, steady-state flow approximation).

Workflow Documentation
~~~~~~~~~~~~~~~~~~~~~~

**Version control**: Track code, parameters, and data provenance using git or similar tools.

**Reproducible scripts**: Ensure analyses can be reproduced by others. Use Jupyter notebooks or documented scripts.

**Save calibrated parameters**: Store optimized parameter values with metadata (calibration period, objective function, convergence criteria).

**Link to examples**: Reference the example notebooks in your documentation to demonstrate usage patterns.