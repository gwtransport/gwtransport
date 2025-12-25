gwtransport
===========

``gwtransport`` enables timeseries analysis of groundwater transport of solutes and temperature. 1) Estimate aquifer properties either from temperature tracer tests or streamline analysis. 2) Predict residence times and solute transport, and assess pathogen removal efficiency in bank filtration systems.

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

What you can do with a calibrated model
---------------------------------------

Calibration refers to the estimation of the aquifer pore volume distribution. Once you have estimated this, you can:

-  **Predict residence time distributions** under varying flow conditions
-  **Forecast contaminant arrival times** and extracted concentrations
-  **Design treatment systems** with quantified pathogen removal efficiency
-  **Assess groundwater vulnerability** to contamination with a retardation that may depend on its concentration (non-linear sorption)
-  **Enable early warning systems** as digital twins for drinking water protection

Two ways to obtain model parameters
-----------------------------------

The aquifer pore volume distribution is constant over time and can be obtained using:

1. Streamline Analysis
~~~~~~~~~~~~~~~~~~~~~~

You already have computed the groundwater flow with an analytical solution or a numerical groundwater flow model. Compute the area between streamlines from flow field data to directly estimate the pore volume distribution parameters.

.. code:: python

   from gwtransport.advection import infiltration_to_extraction

   # Measurements
   cin = [1.0, 2.0, 3.0]  # Concentration infiltrated water [g/l]
   flow = [100.0, 150.0, 100.0]  # Flow rates [m3/day]
   tedges = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")  # Time edges for cin and flow bins

   # Information from groundwater flow model
   areas_between_streamlines = np.array([100.0, 90.0, 110.0])  # [m2]
   depth_aquifer = 2.0  # [m] Convert areas between 2D streamlines to 3D aquifer pore volumes
   porosity = 0.35  # [-]
   aquifer_pore_volumes = areas_between_streamlines * depth_aquifer * porosity  # [m3]

   cout = infiltration_to_extraction(
       cin=cin,
       flow=flow,
       tedges=tedges,
       cout_tedges=tedges,
       aquifer_pore_volumes=aquifer_pore_volumes,
       retardation_factor=1.0,
   )  # [g/l] same units as cin

   # Note: Initial output values are NaN until the first cin value has fully passed the aquifer

2. Temperature Tracer Test
~~~~~~~~~~~~~~~~~~~~~~~~~~

Approximate the aquifer pore volume distribution with a two-parameter gamma distribution. Estimate these parameters from measured temperatures of infiltrated and extracted water. Temperature acts as a natural tracer, revealing flow paths through heterogeneous aquifers via calibration. No groundwater model is required.

.. code:: python

   from gwtransport.advection import gamma_infiltration_to_extraction

   # Measurements
   cin = [11.0, 12.0, 13.0]  # [degC] Measured temperature infiltrated water
   flow = [100.0, 150.0, 100.0]  # Flow rates [m3/day]
   tedges = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")  # Time edges for cin and flow bins

   cout_measured = [10.5, 11.0, 11.5]  # [degC] Measured temperature extracted water (required for calibration only)

   cout_model = gamma_infiltration_to_extraction(
       cin=cin,
       flow=flow,
       tedges=tedges,
       cout_tedges=tedges,
       mean=200.0,  # [m3] Adjust such that cout_model matches the measured cout
       std=16.0,    # [m3] Adjust such that cout_model matches the measured cout
       retardation_factor=2.0,  # [-] Retardation factor for the temperature tracer
   )

   # Compare model output with measured data to calibrate mean and std parameters (see example notebook 1)

Installation
------------

.. code:: bash

   pip install gwtransport

Documentation Contents
----------------------

For an in-depth discussion of the core modeling approach, see :doc:`user_guide/concepts`. To determine when ``gwtransport`` is appropriate for your application and understand the underlying assumptions, see :doc:`user_guide/assumptions`.

.. toctree::
   :maxdepth: 2
   :caption: Guide
   :hidden:

   user_guide/concepts
   user_guide/assumptions

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/01_Aquifer_Characterization_Temperature.nblink
   examples/02_Residence_Time_Analysis.nblink
   examples/03_Pathogen_Removal_Bank_Filtration.nblink
   examples/04_Deposition_Analysis_Bank_Filtration.nblink
   examples/05_Diffusion_Dispersion.nblink
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
