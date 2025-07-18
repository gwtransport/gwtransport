# gwtransport

**Characterize groundwater systems and predict contaminant transport from field temperature data**

`gwtransport` provides timeseries analysis of groundwater transport of solu†es and temperature. Estimate two aquifer properties from a temperature tracer test, predict residence times and transport of other solutes, and assess pathogen removal efficiency. Alternatively, the aquifer properties can be estimated directly from the streamlines.

|                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Testing of source code | [![Functional Testing](https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml) [![Test Coverage](https://gwtransport.github.io/gwtransport/coverage-badge.svg)](https://gwtransport.github.io/gwtransport/htmlcov/) [![Linting](https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml) [![Build and release package](https://github.com/gwtransport/gwtransport/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/release.yml) |
| Testing of examples    | [![Testing of examples](https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml) [![Coverage by examples](https://gwtransport.github.io/gwtransport/coverage_examples-badge.svg)](https://gwtransport.github.io/gwtransport/htmlcov_examples/)                                                                                                                                                                                                                                                                                                                                                                           |
| Package                | [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gwtransport.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/gwtransport/) [![PyPI - Version](https://img.shields.io/pypi/v/gwtransport.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/gwtransport/) [![GitHub commits since latest release](https://img.shields.io/github/commits-since/gwtransport/gwtransport/latest?logo=github&logoColor=lightgrey)](https://github.com/gwtransport/gwtransport/compare/)                                                                                                                                                                                                                                    |

# gwtransport

**Timeseries analysis of groundwater transport of solutes and temperature**

## What You Can Do With a Calibrated Model

Once you have calibrated the aquifer pore volume distribution, you can:

- **Predict residence time distributions** under varying flow conditions
- **Forecast contaminant arrival times** and transport pathways  
- **Design treatment systems** with quantified pathogen removal efficiency
- **Assess groundwater vulnerability** to contamination
- **Early warning systems** as digital twin for drinking water protection

## Two Ways to obtain model parameters

The aquifer pore volume distribution can be obtained using:

### 1. Streamline Analysis  
Compute the area between streamlines from flow field data to directly estimate the pore volume distribution parameters.

```python
from gwtransport.advection import distribution_infiltration_to_extraction

# Measurements
cin_data = [1.0, 2.0, 3.0]  # Example concentration infiltrated water
flow_data = [100.0, 150.0, 100.0]  # Example flow rates
tedges = [0, 1, 2, 3]  # Example time edges

areas_between_streamlines = np.array([100.0, 200.0, 150.0])  # Example areas
depth_aquifer = 200.0  # Convert areas between 2d streamlines to 3d aquifer pore volumes.
aquifer_pore_volumes = areas_between_streamlines * depth_aquifer

cout = distribution_infiltration_to_extraction(
    cin=cin_data,
    flow=flow_data,
    tedges=tedges,
    cout_tedges=tedges,
    aquifer_pore_volumes=aquifer_pore_volumes,
    retardation_factor=1.0,
)
```

### 2. Temperature Tracer Test
Approximate the aquifer pore volume distribution with a two-parameter gamma distribution. Estimate these parameters from the measured temperature of the infiltrated and extracted water. Temperature acts as a natural tracer, revealing how water flows through different paths in heterogeneous aquifers through calibration.

```python
from gwtransport.advection import gamma_infiltration_to_extraction

# Measurements
cin_data = [11.0, 12.0, 13.0]  # Example temperature infiltrated water
flow_data = [100.0, 150.0, 100.0]  # Example flow rates
tedges = [0, 1, 2, 3]  # Example time edges

cout_data = [10.5, 11.0, 11.5]  # Example temperature extracted water. Only required for the calibration period.

cout_model = gamma_infiltration_to_extraction(
    cin=cin_data,
    flow=flow_data,
    tedges=tedges,
    cout_tedges=tedges,
    mean=30000,  # [m3] Adjust such that cout_model matches the measured cout
    std=8100,    # [m3] Adjust such that cout_model matches the measured cout
    retardation_factor=2.0,  # [-] Retardation factor for the temperature tracer
)

# Compare model output with measured data to calibrate the mean and std parameters. See Example 1.
```

## Installation

```bash
pip install gwtransport
```

## Examples and Documentation

Examples:
- [Estimate aquifer pore volume from temperature response](https://gwtransport.github.io/gwtransport/examples/01_Estimate_aquifer_pore_volume_from_temperature_response.html)
- [Estimate the residence time distribution](https://gwtransport.github.io/gwtransport/examples/02_Estimate_the_residence_time_distribution.html)
- [Log removal efficiency analysis](https://gwtransport.github.io/gwtransport/examples/03_Log_removal.html)

Full documentation: [gwtransport.github.io/gwtransport](https://gwtransport.github.io/gwtransport)

## License

GNU Affero General Public License v3.0