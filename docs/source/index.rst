gwtransport Documentation
==========================

**Characterize groundwater systems and predict contaminant transport from field temperature data**

``gwtransport`` provides timeseries analysis of groundwater transport of solutes and temperature. 
Estimate two aquifer properties from a temperature tracer test, predict residence times and transport 
of other solutes, and assess pathogen removal efficiency. Alternatively, the aquifer properties can 
be estimated directly from the streamlines.

.. image:: https://img.shields.io/pypi/v/gwtransport.svg
   :target: https://pypi.org/project/gwtransport/
   :alt: PyPI Version

.. image:: https://img.shields.io/pypi/pyversions/gwtransport.svg
   :target: https://pypi.org/project/gwtransport/
   :alt: Python Versions

.. image:: https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml/badge.svg
   :target: https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml
   :alt: Tests

Quick Start
-----------

Install gwtransport with pip:

.. code-block:: bash

   pip install gwtransport

Basic usage example:

.. code-block:: python

   from gwtransport.advection import gamma_infiltration_to_extraction

   # Temperature tracer test analysis
   cout_model = gamma_infiltration_to_extraction(
       cin=[11.0, 12.0, 13.0],        # Temperature infiltrated water
       flow=[100.0, 150.0, 100.0],    # Flow rates
       tedges=[0, 1, 2, 3],           # Time edges
       cout_tedges=[0, 1, 2, 3],      # Output time edges
       mean=30000,                    # Mean pore volume [m³]
       std=8100,                      # Standard deviation [m³]
       retardation_factor=2.0,        # Retardation factor
   )

What You Can Do
---------------

Once you have calibrated the aquifer pore volume distribution, you can:

- **Predict residence time distributions** under varying flow conditions
- **Forecast contaminant arrival times** and transport pathways
- **Design treatment systems** with quantified pathogen removal efficiency
- **Assess groundwater vulnerability** to contamination
- **Early warning systems** as digital twin for drinking water protection

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   getting_started
   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/01_Aquifer_Characterization_Temperature.nblink
   examples/02_Residence_Time_Analysis.nblink
   examples/03_Pathogen_Removal_Bank_Filtration.nblink

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`