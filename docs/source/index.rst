gwtransport
===========

``gwtransport`` provides timeseries analysis of groundwater transport of
solutes and temperature. Estimate two aquifer properties from a
temperature tracer test, predict residence times and transport of other
solutes, and assess pathogen removal efficiency. Alternatively, the
aquifer properties can be estimated directly from the streamlines.

+------------------------+--------------------------------------------+
| Testing of source code | |Functional Testing| |Test Coverage|       |
|                        | |Linting| |Build and release package|      |
+------------------------+--------------------------------------------+
| Testing of examples    | |Testing of examples|                      |
|                        |                                            |
+------------------------+--------------------------------------------+
| Package                | |PyPI - Python Version| |PyPI - Version|   |
|                        | |GitHub commits since latest release|      |
+------------------------+--------------------------------------------+

.. _gwtransport-1:

What you can do with a calibrated model
---------------------------------------

Once you have calibrated the aquifer pore volume distribution, you can:

-  **Predict residence time distributions** under varying flow
   conditions
-  **Forecast contaminant arrival times** and transport pathways
-  **Design treatment systems** with quantified pathogen removal
   efficiency
-  **Assess groundwater vulnerability** to contamination
-  **Early warning systems** as digital twin for drinking water
   protection

Two ways to obtain model parameters
-----------------------------------

The aquifer pore volume distribution can be obtained using:

.. _1-streamline-analysis:

1. Streamline Analysis
~~~~~~~~~~~~~~~~~~~~~~

Compute the area between streamlines from flow field data to directly
estimate the pore volume distribution parameters.

.. code:: python

   from gwtransport.advection import infiltration_to_extraction

   # Measurements
   cin_data = [1.0, 2.0, 3.0]  # Example concentration infiltrated water
   flow_data = [100.0, 150.0, 100.0]  # Example flow rates
   tedges = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")  # Example time edges

   areas_between_streamlines = np.array([100.0, 90.0, 110.0])  # Example areas
   depth_aquifer = 2.0  # Convert areas between 2d streamlines to 3d aquifer pore volumes.
   aquifer_pore_volumes = areas_between_streamlines * depth_aquifer

   cout = infiltration_to_extraction(
       cin=cin_data,
       flow=flow_data,
       tedges=tedges,
       cout_tedges=tedges,
       aquifer_pore_volumes=aquifer_pore_volumes,
       retardation_factor=1.0,
   )

   # Note that the first values are NaN, as no cin values have fully passed through the aquifer yet.

.. _2-temperature-tracer-test:

2. Temperature Tracer Test
~~~~~~~~~~~~~~~~~~~~~~~~~~

Approximate the aquifer pore volume distribution with a two-parameter
gamma distribution. Estimate these parameters from the measured
temperature of the infiltrated and extracted water. Temperature acts as
a natural tracer, revealing how water flows through different paths in
heterogeneous aquifers through calibration.

.. code:: python

   from gwtransport.advection import gamma_infiltration_to_extraction

   # Measurements
   cin_data = [11.0, 12.0, 13.0]  # Example temperature infiltrated water
   flow_data = [100.0, 150.0, 100.0]  # Example flow rates
   tedges = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")  # Example time edges

   cout_data = [10.5, 11.0, 11.5]  # Example temperature extracted water. Only required for the calibration period.

   cout_model = gamma_infiltration_to_extraction(
       cin=cin_data,
       flow=flow_data,
       tedges=tedges,
       cout_tedges=tedges,
       mean=200.0,  # [m3] Adjust such that cout_model matches the measured cout
       std=16.0,    # [m3] Adjust such that cout_model matches the measured cout
       retardation_factor=2.0,  # [-] Retardation factor for the temperature tracer
   )

   # Compare model output with measured data to calibrate the mean and std parameters. See Example 1.

Installation
------------

.. code:: bash

   pip install gwtransport

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
   examples/04_Deposition_Analysis_Bank_Filtration.nblink
   examples/10_Advection_with_non_linear_sorption.nblink

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules

License
-------

GNU Affero General Public License v3.0

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |Functional Testing| image:: https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml/badge.svg?branch=main
   :target: https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml
.. |Test Coverage| image:: https://gwtransport.github.io/gwtransport/coverage-badge.svg
   :target: https://gwtransport.github.io/gwtransport/htmlcov/
.. |Linting| image:: https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml/badge.svg?branch=main
   :target: https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml
.. |Build and release package| image:: https://github.com/gwtransport/gwtransport/actions/workflows/release.yml/badge.svg?branch=main
   :target: https://github.com/gwtransport/gwtransport/actions/workflows/release.yml
.. |Testing of examples| image:: https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml/badge.svg?branch=main
   :target: https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml
.. |PyPI - Python Version| image:: https://img.shields.io/pypi/pyversions/gwtransport.svg?logo=python&label=Python&logoColor=gold
   :target: https://pypi.org/project/gwtransport/
.. |PyPI - Version| image:: https://img.shields.io/pypi/v/gwtransport.svg?logo=pypi&label=PyPI&logoColor=gold
   :target: https://pypi.org/project/gwtransport/
.. |GitHub commits since latest release| image:: https://img.shields.io/github/commits-since/gwtransport/gwtransport/latest?logo=github&logoColor=lightgrey
   :target: https://github.com/gwtransport/gwtransport/compare/
