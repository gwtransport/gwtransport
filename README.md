# gwtransport

`gwtransport` enables timeseries analysis of groundwater transport of solutes and temperature. Estimate aquifer properties from temperature tracer tests or streamline analysis, predict residence times and solute transport, and assess pathogen removal efficiency in bank filtration systems.

|                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Testing of source code | [![Functional Testing](https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml) [![Test Coverage](https://gwtransport.github.io/gwtransport/coverage-badge.svg)](https://gwtransport.github.io/gwtransport/htmlcov/) [![Linting](https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml) [![Build and release package](https://github.com/gwtransport/gwtransport/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/release.yml) |
| Testing of examples    | [![Testing of examples](https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml)htmlcov_examples/)                                                                                                                                                                                                                                                                                                                                                                           |
| Package                | [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gwtransport.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/gwtransport/) [![PyPI - Version](https://img.shields.io/pypi/v/gwtransport.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/gwtransport/) [![GitHub commits since latest release](https://img.shields.io/github/commits-since/gwtransport/gwtransport/latest?logo=github&logoColor=lightgrey)](https://github.com/gwtransport/gwtransport/compare/)                                                                                                                                                                                                                                    |

## What you can do with a calibrated model

Once you have calibrated the aquifer pore volume distribution, you can:

- **Predict residence time distributions** under varying flow conditions
- **Forecast contaminant arrival times** and extracted concentrations
- **Design treatment systems** with quantified pathogen removal efficiency
- **Assess groundwater vulnerability** to contamination
- **Enable early warning systems** as digital twins for drinking water protection

## Two ways to obtain model parameters

The aquifer pore volume distribution can be obtained using:

### 1. Streamline Analysis

Compute the area between streamlines from flow field data to directly estimate the pore volume distribution parameters.

```python
from gwtransport.advection import infiltration_to_extraction

# Measurements
cin_data = [1.0, 2.0, 3.0]  # Example concentration infiltrated water
flow_data = [100.0, 150.0, 100.0]  # Example flow rates
tedges = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")  # Example time edges

areas_between_streamlines = np.array([100.0, 90.0, 110.0])  # Example areas
depth_aquifer = 2.0  # Convert areas between 2D streamlines to 3D aquifer pore volumes
porosity = 0.35
aquifer_pore_volumes = areas_between_streamlines * depth_aquifer * porosity

cout = infiltration_to_extraction(
    cin=cin_data,
    flow=flow_data,
    tedges=tedges,
    cout_tedges=tedges,
    aquifer_pore_volumes=aquifer_pore_volumes,
    retardation_factor=1.0,
)

# Note: Initial output values are NaN until cin values fully pass through the aquifer
```

### 2. Temperature Tracer Test

Approximate the aquifer pore volume distribution with a two-parameter gamma distribution. Estimate these parameters from measured temperatures of infiltrated and extracted water. Temperature acts as a natural tracer, revealing flow paths through heterogeneous aquifers via calibration.

```python
from gwtransport.advection import gamma_infiltration_to_extraction

# Measurements
cin_data = [11.0, 12.0, 13.0]  # Example temperature infiltrated water
flow_data = [100.0, 150.0, 100.0]  # Example flow rates
tedges = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")  # Example time edges

cout_data = [10.5, 11.0, 11.5]  # Example temperature extracted water (required for calibration only)

cout_model = gamma_infiltration_to_extraction(
    cin=cin_data,
    flow=flow_data,
    tedges=tedges,
    cout_tedges=tedges,
    mean=200.0,  # [m3] Adjust such that cout_model matches the measured cout
    std=16.0,    # [m3] Adjust such that cout_model matches the measured cout
    retardation_factor=2.0,  # [-] Retardation factor for the temperature tracer
)

# Compare model output with measured data to calibrate mean and std parameters (see Example 1)
```

## Installation

```bash
pip install gwtransport
```

## Examples and Documentation

Examples:

- [Estimate aquifer pore volume from temperature response](https://gwtransport.github.io/gwtransport/examples/01_Aquifer_Characterization_Temperature.html)
- [Estimate the residence time distribution](https://gwtransport.github.io/gwtransport/examples/02_Residence_Time_Analysis.html)
- [Log removal efficiency analysis](https://gwtransport.github.io/gwtransport/examples/03_Pathogen_Removal_Bank_Filtration.html)
- [Deposition analysis](https://gwtransport.github.io/gwtransport/examples/04_Deposition_Analysis_Bank_Filtration.html)

Full documentation: [gwtransport.github.io/gwtransport](https://gwtransport.github.io/gwtransport)

## License

GNU Affero General Public License v3.0
