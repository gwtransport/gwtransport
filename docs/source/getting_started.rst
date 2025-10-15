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

.. testcode::

   import numpy as np
   from gwtransport.advection import gamma_infiltration_to_extraction

   # Measurement data
   tedges = pd.date_range(start="2020-01-01", end="2020-01-07", freq="D")

   cin = np.ones(len(tedges) - 1) * 1.0     # Initial concentration [g/m³]
   cin[2:] = 2.0                            # Step change on day 3 (January 3)
   flow = np.ones(len(tedges) - 1) * 100.0  # Flow rates [m³/day]

   # Compute model prediction
   cout_model = gamma_infiltration_to_extraction(
         cin=cin,
         tedges=tedges,
         cout_tedges=tedges,
         flow=flow,
         mean=100.0,  # [m³]
         std=30.0,    # [m³]
         retardation_factor=1.0,  # [-]
         n_bins=20,
   )

   print(f"Predicted temperature: {cout_model}")

.. testoutput::
   :options: +ELLIPSIS

   Predicted temperature between cout_tedges: [       nan 1.         1.11874999 1.88125001 2.         2.        ]

Next Steps
----------

- Explore the :doc:`examples/01_Aquifer_Characterization_Temperature` to see detailed workflows
- Check the :doc:`api/modules` for complete function documentation
- See :doc:`user_guide/index` for advanced usage patterns