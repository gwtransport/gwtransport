Common Workflows
================

This section describes typical workflows for using gwtransport in different scenarios.

Temperature Tracer Test Workflow
---------------------------------

This is the most common workflow for aquifer characterization using temperature data.

Step 1: Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~~

Collect and prepare your data:

.. code-block:: python

   import numpy as np
   import pandas as pd
   from gwtransport.advection import gamma_infiltration_to_extraction
   
   # Load your data
   data = pd.read_csv('temperature_data.csv')
   
   # Extract time series
   cin_data = data['temp_infiltration'].values
   flow_data = data['flow_rate'].values
   time_data = data['time'].values
   
   # Create time edges
   tedges = np.concatenate([[time_data[0]], time_data])

Step 2: Model Calibration
~~~~~~~~~~~~~~~~~~~~~~~~~

Fit model parameters to observed data:

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
       
       # Compare with observations
       error = np.sum((cout_model - cout_observed) ** 2)
       return error
   
   # Initial parameter guess
   initial_params = [30000, 8100, 2.0]
   
   # Optimize parameters
   result = minimize(objective_function, initial_params, method='Nelder-Mead')
   optimal_params = result.x

Step 3: Validation and Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate the model and make predictions:

.. code-block:: python

   # Use optimized parameters for predictions
   mean_vol, std_vol, retardation = optimal_params
   
   # Predict for new conditions
   cout_prediction = gamma_infiltration_to_extraction(
       cin=cin_new,
       flow=flow_new,
       tedges=tedges_new,
       cout_tedges=tedges_new,
       mean=mean_vol,
       std=std_vol,
       retardation_factor=retardation,
   )

Streamline Analysis Workflow
-----------------------------

For cases where you have detailed flow field data.

Step 1: Compute Streamlines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gwtransport.advection import infiltration_to_extraction

   # Compute areas between streamlines (from flow modeling)
   areas_between_streamlines = compute_streamline_areas(flow_field)

   # Convert to 3D pore volumes
   depth_aquifer = 200.0  # [m]
   aquifer_pore_volumes = areas_between_streamlines * depth_aquifer

Step 2: Direct Transport Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use pore volumes directly
   cout = infiltration_to_extraction(
       cin=cin_data,
       flow=flow_data,
       tedges=tedges,
       cout_tedges=tedges,
       aquifer_pore_volumes=aquifer_pore_volumes,
       retardation_factor=1.0,
   )

Residence Time Analysis Workflow
---------------------------------

Analyze residence time distributions for different scenarios.

Step 1: Compute Base Residence Times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gwtransport.residence_time import compute_residence_time_distribution
   
   # Compute residence time distribution
   residence_times = compute_residence_time_distribution(
       aquifer_pore_volumes=aquifer_pore_volumes,
       flow_rates=flow_data,
       tedges=tedges,
   )

Step 2: Scenario Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze different flow scenarios
   scenarios = {
       'low_flow': flow_data * 0.5,
       'normal_flow': flow_data,
       'high_flow': flow_data * 2.0,
   }
   
   residence_results = {}
   for scenario, flows in scenarios.items():
       residence_results[scenario] = compute_residence_time_distribution(
           aquifer_pore_volumes=aquifer_pore_volumes,
           flow_rates=flows,
           tedges=tedges,
       )

Pathogen Removal Analysis Workflow
-----------------------------------

Assess pathogen removal efficiency for treatment design.

Step 1: Define Pathogen Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gwtransport.logremoval import compute_log_removal
   
   # Define pathogen removal parameters
   pathogen_params = {
       'decay_rate': 0.1,  # [1/day]
       'attachment_rate': 0.05,  # [1/day]
       'detachment_rate': 0.01,  # [1/day]
   }

Step 2: Compute Removal Efficiency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute log removal for different residence times
   log_removal = compute_log_removal(
       residence_times=residence_times,
       decay_rate=pathogen_params['decay_rate'],
       attachment_rate=pathogen_params['attachment_rate'],
       detachment_rate=pathogen_params['detachment_rate'],
   )
   
   # Assess treatment effectiveness
   removal_efficiency = 1 - 10**(-log_removal)
   print(f"Pathogen removal efficiency: {removal_efficiency:.2%}")

Best Practices
--------------

Data Quality
~~~~~~~~~~~~

- Ensure high-resolution time series data
- Check for data gaps and outliers
- Validate measurement accuracy
- Consider seasonal variations

Model Validation
~~~~~~~~~~~~~~~~

- Use independent data for validation
- Check residual patterns
- Perform sensitivity analysis
- Compare with physical expectations

Parameter Uncertainty
~~~~~~~~~~~~~~~~~~~~~

- Quantify parameter uncertainty
- Propagate uncertainty to predictions
- Use ensemble approaches when appropriate
- Document assumptions and limitations

Documentation
~~~~~~~~~~~~~

- Document data sources and processing
- Record model assumptions
- Save parameter values and fits
- Create reproducible workflows