Core Concepts
=============

Understanding Groundwater Transport
------------------------------------

Groundwater transport involves the movement of solutes through porous media. The key concepts in gwtransport include:

Pore Volume Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

The pore volume distribution describes how water flows through different pathways in a heterogeneous aquifer. 
This distribution is characterized by:

- **Mean pore volume**: The average volume of water in flow paths
- **Standard deviation**: The variability in pore volumes across different paths
- **Shape**: Often approximated using a gamma distribution

Residence Time
~~~~~~~~~~~~~~

Residence time is the time water spends in the aquifer between infiltration and extraction points. 
It depends on:

- Flow velocity
- Path length
- Porosity
- Pore volume distribution

Retardation Factor
~~~~~~~~~~~~~~~~~~

The retardation factor accounts for processes that slow down solute transport relative to water flow:

- **Conservative tracers** (R = 1): Move with the same velocity as water
- **Temperature** (R > 1): Often retarded due to heat exchange with the solid matrix
- **Reactive solutes** (R > 1): Delayed by sorption or chemical reactions

Temperature as a Natural Tracer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Temperature variations in infiltrated water act as natural tracers because:

- They are non-invasive and cost-effective
- They provide continuous monitoring capability
- They have known retardation properties
- They are affected by the same transport processes as other solutes

Model Approaches
----------------

Gamma Distribution Model
~~~~~~~~~~~~~~~~~~~~~~~~

The gamma distribution model assumes that pore volumes follow a gamma distribution with parameters:

.. math::

   f(V) = \frac{1}{\Gamma(k)\theta^k} V^{k-1} e^{-V/\theta}

Where:
- :math:`k` is the shape parameter
- :math:`\theta` is the scale parameter
- Mean = :math:`k \cdot \theta`
- Variance = :math:`k \cdot \theta^2`

Streamline Analysis
~~~~~~~~~~~~~~~~~~~

Direct computation of pore volumes from flow field data:

1. Compute streamlines from infiltration to extraction points
2. Calculate areas between adjacent streamlines
3. Convert 2D areas to 3D pore volumes using aquifer depth
4. Use these volumes directly in transport calculations

Advection-Dispersion Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transport equations are solved using an advection-dispersion framework that accounts for:

- **Advection**: Bulk movement with the flow
- **Dispersion**: Spreading due to velocity variations
- **Retardation**: Delayed transport due to physical/chemical processes

Applications
------------

Water Quality Management
~~~~~~~~~~~~~~~~~~~~~~~~

- Predicting contaminant arrival times
- Designing treatment systems
- Assessing vulnerability to contamination

Aquifer Characterization
~~~~~~~~~~~~~~~~~~~~~~~~

- Estimating hydraulic properties
- Understanding flow heterogeneity
- Validating groundwater models

Early Warning Systems
~~~~~~~~~~~~~~~~~~~~~

- Real-time monitoring of water quality
- Automated alerts for contamination events
- Digital twin applications for water utilities