|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![Functional Testing](https://github.com/bdestombe/python_gwtransport1d/actions/workflows/functional_testing.yml/badge.svg?branch=main)](https://github.com/bdestombe/python_gwtransport1d/actions/workflows/functional_testing.yml) [![Linting](https://github.com/bdestombe/python_gwtransport1d/actions/workflows/linting.yml/badge.svg?branch=main)](https://github.com/bdestombe/python_gwtransport1d/actions/workflows/linting.yml) [![Build and release package](https://github.com/bdestombe/python-gwtransport1d/actions/workflows/release.yml/badge.svg?branch=main)](https://github.com/bdestombe/python-gwtransport1d/actions/workflows/release.yml) |
| Package | [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gwtransport1d.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/gwtransport1d/) [![PyPI - Version](https://img.shields.io/pypi/v/gwtransport1d.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/gwtransport1d/) [![GitHub commits since latest release](https://img.shields.io/github/commits-since/bdestombe/python-gwtransport1d/latest?logo=github&logoColor=lightgrey)](https://github.com/bdestombe/python-gwtransport1d/compare/)                                                                                                                   |
|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

# gwtransport1d

Transport of temperature and contaminants in 1D groundwater flow system. This code is used for the bank filtration systems operated by PWN, drinkingwater company in the Netherlands.

- Compute residence times with forward and backward tracking
- Compute the temperature / concentration of a contaminant at the extraction well if advection is the main transport mechanism
- Compute the areal deposition that explains the difference between the inlet and outlet concentration

## Multi-1d transport

Aquifers are often not 1D, but 2D or 3D. The code is extended to multi-1D transport, where the 1D transport is computed for each layer. Currently implemented is gamma distribution to compute the residence time distribution in the aquifer. It assumes the flow is distributed as a gamma distribution over the layers, resulting in a gamma distribution of the residence times.

## Aquifer pore volume

Flow may vary over time while the aquifer pore volume does not. The aquifer pore volume, or its distribution in multi-1d, is used to compute the residence time distribution.

## Installation

```bash
pip install gwtransport1d
```
