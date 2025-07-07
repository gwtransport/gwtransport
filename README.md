# gwtransport

**Characterize groundwater systems and predict contaminant transport from field temperature data**

`gwtransport` provides a complete workflow for analyzing heterogeneous aquifer systems - from field measurements to treatment design. Using temperature as a natural tracer, estimate aquifer properties, predict residence times, and assess pathogen removal efficiency.

|                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Testing of source code | [![Functional Testing](https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml) [![Test Coverage](https://gwtransport.github.io/gwtransport/coverage-badge.svg)](https://gwtransport.github.io/gwtransport/htmlcov/) [![Linting](https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml) [![Build and release package](https://github.com/gwtransport/gwtransport/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/release.yml) |
| Testing of examples    | [![Testing of examples](https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml) [![Coverage by examples](https://gwtransport.github.io/gwtransport/coverage_examples-badge.svg)](https://gwtransport.github.io/gwtransport/htmlcov_examples/)                                                                                                                                                                                                                                                                                                                                                                           |
| Package                | [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gwtransport.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/gwtransport/) [![PyPI - Version](https://img.shields.io/pypi/v/gwtransport.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/gwtransport/) [![GitHub commits since latest release](https://img.shields.io/github/commits-since/gwtransport/gwtransport/latest?logo=github&logoColor=lightgrey)](https://github.com/gwtransport/gwtransport/compare/)                                                                                                                                                                                                                                    |

## Three-Step Workflow

### 1. Characterize Aquifer Heterogeneity

**From temperature measurements to aquifer properties**

Use temperature breakthrough curves to estimate pore volume distributions through inverse modeling. Temperature acts as a natural tracer, revealing how water flows through different paths in heterogeneous aquifers.

- **Input**: Temperature time series, flow rates
- **Output**: Statistical parameters of aquifer pore volume distribution
- **Applications**: Groundwater vulnerability assessment, aquifer characterization

![Temperature Response Analysis](examples/01_Temperature_response.png)

_Temperature breakthrough curves reveal aquifer heterogeneity - faster breakthrough indicates shorter flow paths_

### 2. Predict Residence Time Distributions

**From aquifer properties to water travel times**

Calculate how long water spends in the aquifer under varying flow conditions. Essential for predicting when contamination will arrive or when treatment processes will be effective.

- **Input**: Aquifer parameters (from Step 1), flow time series
- **Output**: Residence time distributions over time
- **Applications**: Contaminant transport forecasting, early warning systems

![Residence Time Analysis](examples/02_Forward_residence_time.png)

_Residence time distributions show when infiltrating water will be extracted - critical for contamination timing_

### 3. Design Treatment Systems

**From residence times to pathogen removal efficiency**

Evaluate pathogen log-removal efficiency in bank filtration systems. Design optimal flow rates and assess parallel treatment configurations for safe drinking water production.

- **Input**: Residence times (from Step 2), pathogen removal kinetics
- **Output**: Log-removal efficiency, design flow rates
- **Applications**: Drinking water treatment design, regulatory compliance

![Log-Removal Analysis](examples/03_log_removal_analysis.png)

_Parallel treatment systems require weighted averaging - simple averaging overestimates removal efficiency_

## Key Features

- **🌡️ Temperature-based characterization**: Use readily available temperature data as natural tracers
- **📊 Statistical framework**: Gamma distributions model aquifer heterogeneity realistically
- **⏱️ Residence time analysis**: Predict water travel times under varying flow conditions
- **🦠 Pathogen removal assessment**: Calculate log-removal efficiency for treatment design
- **🔧 Design tools**: Find optimal flow rates for target removal efficiency
- **📈 Risk assessment**: Evaluate treatment performance under uncertainty

## Installation & Quick Start

```bash
pip install gwtransport
```

Run the complete workflow examples:

```bash
# Locate the examples directory
cd gwtransport/examples

# Run all three examples
python 01_Estimate_aquifer_pore_volume_from_temperature_response.py
python 02_Estimate_the_residence_time_distribution.py
python 03_Log_removal.py
```

## Core Functions

```python
# Aquifer characterization (Example 1)
from gwtransport.advection import gamma_forward

# Residence time analysis (Example 2)
from gwtransport.residence_time import residence_time

# Treatment design (Example 3)
from gwtransport.logremoval import parallel_mean, gamma_find_flow_for_target_mean
```

## Applications

- **Bank filtration systems**: Design and optimize natural treatment processes
- **Groundwater vulnerability**: Assess contamination risks and travel times
- **Drinking water safety**: Ensure pathogen removal meets regulatory standards
- **Aquifer characterization**: Quantify heterogeneity without expensive drilling
- **Treatment monitoring**: Evaluate performance of existing systems

## Scientific Basis

The package models aquifer heterogeneity using gamma distributions of pore volumes, representing the reality that groundwater follows preferential flow paths. Temperature serves as an ideal natural tracer because:

- It's continuously measured in most groundwater systems
- It follows known thermal retardation processes
- It provides quantitative data for inverse modeling
- It's non-reactive and conservative

**Key assumptions**: Advection-dominated transport (Pe >> 1), stationary pore volume distributions, well-mixed recharge conditions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits and License

Created by Bas des Tombe and maintained by many contributors. Licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.
