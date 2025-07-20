Getting Started
===============

Installation
------------

Install gwtransport from PyPI:

.. code-block:: bash

   pip install gwtransport

Requirements
~~~~~~~~~~~~

- Python 3.10 or higher
- NumPy
- SciPy
- Pandas
- Matplotlib

Basic Concepts
--------------

gwtransport provides two main approaches to characterize groundwater systems:

1. **Temperature Tracer Test**
   
   Use natural temperature variations as tracers to estimate aquifer properties.
   This approach fits a two-parameter gamma distribution to represent the pore volume distribution.

2. **Streamline Analysis**
   
   Directly compute pore volumes from flow field data using streamline analysis.
   This provides more detailed spatial information about the aquifer structure.

Core Workflow
-------------

1. **Data Collection**
   
   Collect time series data of:
   - Temperature of infiltrated water
   - Temperature of extracted water
   - Flow rates
   - Time measurements

2. **Model Calibration**
   
   Fit model parameters to match observed temperature breakthrough curves.

3. **Prediction**
   
   Use calibrated model to predict:
   - Residence time distributions
   - Contaminant transport
   - Pathogen removal efficiency

Quick Example
-------------

Here's a simple example using temperature tracer test data:

.. code-block:: python

   import numpy as np
   from gwtransport.advection import gamma_infiltration_to_extraction

   # Measurement data
   cin_data = [11.0, 12.0, 13.0]  # Temperature infiltrated water [°C]
   flow_data = [100.0, 150.0, 100.0]  # Flow rates [m³/day]
   tedges = [0, 1, 2, 3]  # Time edges [days]
   
   # Model parameters (to be calibrated)
   mean_pore_volume = 30000  # [m³]
   std_pore_volume = 8100   # [m³]
   retardation_factor = 2.0  # [-]
   
   # Compute model prediction
   cout_model = gamma_infiltration_to_extraction(
       cin=cin_data,
       flow=flow_data,
       tedges=tedges,
       cout_tedges=tedges,
       mean=mean_pore_volume,
       std=std_pore_volume,
       retardation_factor=retardation_factor,
   )
   
   print(f"Predicted temperature: {cout_model}")

Next Steps
----------

- Explore the :doc:`examples/01_Aquifer_Characterization_Temperature` to see detailed workflows
- Check the :doc:`api/modules` for complete function documentation
- See :doc:`user_guide/index` for advanced usage patterns