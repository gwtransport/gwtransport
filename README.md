|                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Testing of source code | [![Functional Testing](https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/functional_testing.yml) [![Test Coverage](https://gwtransport.github.io/gwtransport/coverage-badge.svg)](https://gwtransport.github.io/gwtransport/htmlcov/) [![Linting](https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/linting.yml) [![Build and release package](https://github.com/gwtransport/gwtransport/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/release.yml) |
| Testing of examples    | [![Testing of examples](https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml/badge.svg?branch=main)](https://github.com/gwtransport/gwtransport/actions/workflows/examples_testing.yml) [![Coverage by examples](https://gwtransport.github.io/gwtransport/coverage_examples-badge.svg)](https://gwtransport.github.io/gwtransport/htmlcov_examples/)                                                                                                                                                                                                                                                                                                                                                                           |
| Package                | [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gwtransport.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/gwtransport/) [![PyPI - Version](https://img.shields.io/pypi/v/gwtransport.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/gwtransport/) [![GitHub commits since latest release](https://img.shields.io/github/commits-since/gwtransport/gwtransport/latest?logo=github&logoColor=lightgrey)](https://github.com/gwtransport/gwtransport/compare/)                                                                                                                                                                                                                                    |
|                        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

## Overview

`gwtransport` is a Python package for modeling the transport of temperature and contaminants in groundwater flow systems. It provides practical tools for analyzing advection, diffusion, deposition processes, and calculating residence time distributions in aquifer systems.

### Aquifer Pore Volume Distribution

The aquifer pore volume distribution is a probability density curve that holds the fraction of flow that passes a certain pore volume between infiltration and extraction. This package relies on the estimating this distribution and uses it to compute many different transport processes related to residence time. In the examples here, the aquifer pore volume distribution is approximated by a Gamma distribution of which the two parameters are estimated from the flow rate, the temperature of the infiltration and extracted water. Alternatively, you can derive the aquifer pore volume distribution from the streamlines of your groundwater model. In all examples the assumption is made that the aquifer pore volume distribution is constant over time and all water infiltrates with the same temperature.

### Calibration of the aquifer pore volume distribution (Example 1)

A practical approach to calibrate the two parameters of the Gamma distribution is to use the temperature response of the aquifer. This requires the following measurements over a limited time period:

- Infiltration and extraction temperature,
- Extraction rate or a quantity that is proportional to the extraction rate, such as the head gradient.

See: [Example 1](https://github.com/gwtransport/gwtransport/blob/main/examples/01_Estimate_aquifer_pore_volume_from_temperature_response.py).

### Examples that make use of the aquifer pore volume distribution

The aquifer pore volume distribution can be used to compute:

- [Contaminent transport with different retardation factors](https://github.com/gwtransport/gwtransport/blob/main/examples/01_Estimate_aquifer_pore_volume_from_temperature_response.py)
- [The residence time](https://github.com/gwtransport/gwtransport/blob/main/examples/02_Estimate_the_residence_time_distribution.py)
- [Log-removal of pathogens](https://github.com/gwtransport/gwtransport/blob/main/examples/03_Log_removal.py)

## Installation

```bash
pip install gwtransport
```

## Usage Examples

### Example 1: Estimate Aquifer Pore Volume from Temperature Response

This example demonstrates how to use temperature data to estimate the aquifer pore volume distribution:

```python
from scipy.optimize import curve_fit
from gwtransport import advection

# Load your temperature and flow data
# ...

# Define objective function for curve fitting
def objective(time, mean, std):
    cout = advection.gamma_forward(
        cin=temperature_data,
        flow=flow_data,
        mean=mean,
        std=std,
        n_bins=200,
        retardation_factor=2.0
    )
    return cout.values

# Perform curve fitting
(apv_gamma_mean, apv_gamma_std), pcov = curve_fit(
    objective,
    temperature_data.index,
    extracted_temperature_data,
    p0=(7000.0, 500.0),
)

# Print the fitted parameters
print(
  f"Fitted Gamma distribution for the aquifer pore volume has a mean of: {apv_gamma_mean:.1f} m³ and ",
  f"a standard deviation of: {apv_gamma_std:.1f} m³")
```

### Example 2: Estimate Residence Time Distribution

Once you've characterized your aquifer, you can calculate residence time distributions:

```python
import pandas as pd
from gwtransport import advection, gamma

# Set up your parameters
mean = 8000.0  # m3
std = 400.0  # m3
retardation_factor = 1.0
flow_data = pd.Series(...)  # Your flow data

# Calculate residence time
bins = gamma.bins(mean=mean, std=std, n_bins=1000)

# Calculate forward residence time (infiltration to extraction)
rt_forward = advection.residence_time(
    flow_data,
    bins["expected_value"],
    retardation_factor=1.0,
    direction="infiltration"
)
```

### Example 3: Log Removal of Pathogens

Calculate the log removal of pathogens based on residence time:

```python
import numpy as np
from gwtransport import logremoval

# Calculate log-removal of the extracted water
rt_alpha = 2.0  # Shape parameter of the gamma distribution for the residence time
rt_beta = 10.0  # Scale parameter of the gamma distribution for the residence time
log_removal_rate = 0.5  # Log-removal rate [1/day]
mean_removal = logremoval.gamma_mean(rt_alpha, rt_beta, log_removal_rate)
print(f"Mean log removal: {mean_removal:.2f}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits and License

This project is initially created by Bas des Tombe and is maintained and extended by many contributors. It is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.
