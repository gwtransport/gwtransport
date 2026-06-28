"""
Example Data Generation for Groundwater Transport Modeling.

This module provides utilities to generate synthetic datasets for demonstrating
and testing groundwater transport models. It creates realistic flow patterns,
concentration/temperature time series, and deposition events suitable for testing
advection, diffusion, and deposition analysis functions.

Available functions:

- :func:`generate_example_data` - Generate comprehensive synthetic dataset with flow and
  concentration time series. Creates seasonal flow patterns with optional spill events,
  input concentration data via synthetic sinusoidal patterns, constant values, or real KNMI
  soil temperature, and extracted concentration computed through gamma-distributed pore volume
  transport. When diffusion parameters are provided, uses the diffusion module instead of
  pure advection. Returns DataFrame with flow, cin, cout columns plus attrs containing
  generation parameters and aquifer properties, and time edges (tedges).

- :func:`generate_temperature_example_data` - Convenience wrapper around
  :func:`generate_example_data` with sensible defaults for temperature transport including
  thermal retardation, thermal diffusivity, and longitudinal dispersivity.

- :func:`generate_ec_example_data` - Convenience wrapper around
  :func:`generate_example_data` with sensible defaults for electrical conductivity (EC)
  transport. EC is a conservative tracer (retardation factor 1.0) with negligible molecular
  diffusivity compared to thermal transport.

- :func:`generate_example_deposition_timeseries` - Generate synthetic deposition time series
  for pathogen/contaminant deposition analysis. Combines baseline deposition, seasonal patterns,
  random noise, and episodic contamination events with exponential decay. Returns Series with
  deposition rates [ng/m²/day] and attrs containing generation parameters, and time edges
  (tedges). Useful for testing extraction_to_deposition deconvolution and
  deposition_to_extraction convolution functions.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.advection import gamma_infiltration_to_extraction, infiltration_to_extraction
from gwtransport.diffusion_fast import gamma_infiltration_to_extraction as diffusion_gamma_infiltration_to_extraction
from gwtransport.diffusion_fast import infiltration_to_extraction as diffusion_infiltration_to_extraction
from gwtransport.gamma import mean_std_loc_to_alpha_beta
from gwtransport.utils import compute_time_edges

_DEFAULT_GAMMA_MEAN = 1000.0  # m³
_DEFAULT_GAMMA_STD = 200.0  # m³
_DEFAULT_GAMMA_LOC = 0.0  # m³, minimum pore volume
_DEFAULT_GAMMA_NBINS = 250

# KNMI station 260 (De Bilt) daily-mean soil temperature at 20 cm depth (column TB3, degrees
# Celsius), UTC, from 2020-01-01 onward. Stored inline -- rather than downloaded from KNMI via
# gwtransport.utils.get_soil_temperature -- so the cin_method="soil_temperature" examples run
# without network access (e.g. in the browser under Pyodide/JupyterLite). Rounded to 0.01 degrees.
_SOIL_TEMPERATURE_TB3_START = "2020-01-01"
_SOIL_TEMPERATURE_TB3 = (5.65, 5.62, 6.2, 6.43, 6.68, 6.8, 6.55, 7.07, 7.95, 8.4, 7.38, 7.07, 7.33, 7.65, 8.25, 7.7, 7.47, 7.28, 6.43, 5.8, 5.68, 5.68, 6.32, 5.98, 5.9, 5.8, 6.45, 6.82, 6.38, 6.45, 7.28, 8.05, 7.93, 8.0, 7.73, 6.85, 6.53, 6.32, 6.05, 6.78, 7.57, 6.93, 6.38, 6.03, 6.22, 6.68, 7.75, 8.1, 7.6, 7.18, 7.22, 7.18, 7.25, 7.8, 7.35, 7.47, 6.72, 6.2, 5.82, 6.1, 6.4, 6.38, 6.32, 6.2, 6.47, 6.57, 6.5, 6.82, 7.22, 7.65, 8.5, 8.85, 8.18, 7.75, 7.98, 8.07, 7.93, 8.0, 8.4, 8.38, 7.75, 7.12, 6.72, 6.53, 6.27, 6.43, 6.57, 7.12, 7.18, 6.53, 6.7, 6.6, 6.9, 7.43, 7.3, 7.5, 8.33, 9.07, 9.5, 10.18, 10.2, 10.18, 10.2, 10.35, 9.68, 9.4, 9.93, 10.43, 10.38, 10.55, 10.43, 10.45, 10.7, 11.1, 11.57, 11.55, 11.3, 11.43, 11.62, 11.2, 11.32, 11.2, 11.15, 11.18, 11.8, 11.73, 11.75, 12.07, 12.6, 13.0, 13.35, 12.65, 11.93, 11.6, 11.23, 11.25, 11.68, 11.73, 12.45, 13.38, 14.02, 14.38, 15.0, 14.55, 14.18, 13.93, 14.65, 15.38, 15.32, 15.18, 15.6, 15.82, 15.95, 16.4, 16.88, 16.52, 14.77, 13.9, 13.85, 14.3, 14.88, 14.6, 14.73, 15.5, 16.8, 17.42, 16.92, 17.15, 17.05, 17.4, 17.42, 17.17, 17.05, 17.27, 17.52, 18.33, 18.77, 19.27, 19.62, 18.67, 17.7, 17.02, 17.12, 17.33, 17.45, 17.17, 16.98, 16.8, 16.5, 16.7, 16.65, 16.7, 16.52, 16.58, 16.95, 17.55, 16.83, 17.2, 17.33, 18.0, 18.3, 18.15, 17.55, 17.17, 17.3, 17.62, 17.58, 17.73, 17.58, 17.55, 17.35, 17.33, 18.05, 18.85, 18.25, 17.67, 17.35, 17.73, 18.3, 19.08, 19.62, 20.15, 20.33, 20.45, 21.0, 20.95, 20.95, 20.95, 20.9, 20.38, 19.58, 19.1, 19.62, 19.95, 19.55, 18.73, 18.4, 17.75, 17.52, 17.15, 17.58, 17.15, 16.88, 16.98, 16.8, 16.52, 16.62, 17.12, 17.17, 16.62, 16.4, 16.67, 17.05, 16.65, 16.4, 16.12, 16.12, 16.42, 17.0, 17.5, 16.9, 16.1, 15.85, 15.77, 15.62, 15.6, 15.67, 15.82, 15.4, 14.93, 15.15, 15.38, 15.35, 15.57, 15.68, 14.85, 14.85, 14.52, 14.18, 14.18, 14.18, 14.18, 14.25, 13.65, 12.95, 12.82, 12.9, 12.75, 12.57, 12.45, 12.32, 12.02, 12.12, 11.98, 12.55, 13.2, 12.82, 12.95, 13.2, 12.65, 12.27, 12.18, 11.9, 12.32, 12.95, 12.83, 13.55, 13.1, 11.95, 10.98, 11.05, 10.73, 10.32, 10.73, 11.32, 11.9, 11.8, 11.15, 11.27, 11.62, 11.5, 11.43, 11.68, 11.55, 10.68, 10.05, 10.48, 9.93, 9.47, 9.48, 9.6, 9.92, 9.5, 9.1, 7.82, 8.02, 8.62, 8.6, 8.38, 7.5, 6.75, 6.93, 7.0, 6.7, 6.47, 6.12, 6.53, 7.3, 7.7, 8.38, 8.3, 8.38, 8.32, 8.1, 8.25, 8.07, 8.73, 9.2, 9.23, 8.07, 7.22, 7.2, 7.0, 6.88, 6.72, 6.55, 6.05, 6.18, 6.4, 6.18, 5.93, 5.9, 5.8, 5.75, 4.97, 4.28, 4.7, 5.6, 5.3, 5.12, 4.65, 4.53, 4.45, 4.65, 5.22, 6.03, 6.33, 6.05, 5.47, 5.3, 4.9, 4.88, 5.03, 5.53, 5.95, 5.95, 4.68, 4.1, 4.45, 5.53, 6.15, 6.25, 6.4, 5.4, 4.53, 4.07, 3.75, 3.47, 3.15, 2.85, 2.58, 2.43, 2.55, 4.18, 5.35, 5.55, 6.0, 6.65, 6.62, 7.15, 7.82, 7.97, 8.03, 7.2, 6.65, 6.57, 6.45, 6.32, 6.57, 6.18, 5.3, 4.88, 5.25, 5.97, 5.97, 6.43, 6.68, 6.65, 6.45, 6.75, 6.88, 6.9, 6.68, 6.57, 6.4, 6.85, 6.7, 6.88, 6.82, 7.15, 7.4, 7.62, 7.4, 7.88, 8.15, 8.52, 9.25, 8.52, 8.07, 8.18, 8.02, 7.4, 6.88, 6.8, 7.1, 7.6, 7.2, 6.88, 6.72, 6.72, 6.93, 7.03, 7.38, 7.73, 8.35, 8.65, 8.88, 8.57, 8.33, 8.45, 8.45, 8.48, 8.7, 8.93, 9.07, 8.77, 8.93, 8.93, 8.73, 9.05, 8.85, 8.88, 8.8, 8.73, 9.68, 11.3, 11.5, 11.6, 11.35, 11.77, 11.88, 11.83, 11.9, 11.85, 11.88, 11.93, 12.12, 11.95, 11.8, 12.12, 12.35, 12.1, 12.07, 12.02, 12.77, 13.2, 13.85, 14.05, 14.43, 15.15, 15.5, 15.6, 15.23, 14.95, 15.18, 15.75, 15.93, 16.17, 16.45, 15.9, 16.12, 16.65, 16.62, 17.45, 18.05, 18.12, 17.42, 17.1, 16.33, 16.15, 16.2, 16.1, 16.28, 16.52, 16.4, 16.62, 16.08, 15.62, 15.6, 16.08, 16.48, 16.65, 16.5, 16.52, 16.73, 16.85, 17.08, 17.12, 17.22, 17.4, 17.27, 17.33, 16.98, 17.52, 18.3, 18.5, 18.2, 18.45, 18.77, 18.33, 17.9, 18.1, 18.42, 18.27, 18.02, 17.75, 17.52, 17.52, 17.52, 17.48, 17.22, 17.08, 17.73, 18.2, 17.95, 17.85, 17.67, 17.8, 17.92, 18.33, 18.45, 18.1, 18.1, 18.28, 17.23, 17.15, 17.33, 17.33, 17.75, 18.92, 19.02, 18.65, 18.33, 18.4, 17.92, 17.77, 17.6, 17.62, 17.65, 17.67, 17.75, 17.6, 17.62, 17.38, 17.25, 17.58, 17.77, 17.83, 18.1, 18.12, 18.08, 17.48, 17.25, 17.58, 17.38, 17.2, 17.0, 16.92, 16.7, 16.0, 15.85, 15.8, 16.1, 16.52, 16.52, 16.65, 16.12, 15.75, 15.02, 14.98, 15.1, 15.1, 14.9, 14.7, 14.07, 14.2, 14.07, 13.8, 13.4, 13.68, 13.7, 13.18, 13.48, 13.88, 13.32, 13.27, 13.17, 13.52, 14.1, 13.55, 12.7, 12.07, 11.75, 11.57, 12.15, 12.7, 12.62, 12.32, 12.48, 12.45, 11.88, 11.5, 11.12, 11.1, 10.55, 10.68, 11.05, 10.77, 10.5, 10.73, 10.95, 11.02, 10.62, 11.1, 11.05, 10.52, 10.27, 10.15, 10.68, 11.0, 10.93, 10.0, 9.27, 9.4, 9.43, 9.07, 7.62, 7.03, 6.93, 6.97, 8.05, 7.62, 6.85, 6.97, 7.15, 7.22, 6.28, 6.38, 6.38, 6.45, 6.35, 6.72, 7.7, 8.12, 8.48, 8.48, 8.18, 8.42, 8.45, 8.38, 7.18, 5.78, 5.17, 6.05, 6.55, 5.28, 5.07, 6.25, 7.18, 7.85, 8.73, 9.02, 9.23, 9.27, 8.98, 7.95, 7.25, 6.88, 6.45, 5.57, 5.5, 5.38, 5.6, 6.12, 6.35, 6.55, 6.32, 6.5, 6.43, 6.25, 6.1, 5.65, 5.9, 6.4, 6.57, 6.47, 6.22, 6.35, 6.45, 6.72, 6.93, 6.4, 6.28, 6.72, 7.12, 7.28, 6.88, 6.95, 6.43, 6.7, 7.35, 7.7, 7.35, 6.3, 5.82, 6.43, 6.85, 7.35, 8.3, 7.75, 7.35, 6.9, 7.1, 6.7, 6.78, 6.95, 6.65, 6.2, 5.75, 5.55, 5.5, 5.9, 5.68, 5.45, 5.28, 5.18, 4.93, 4.85, 4.88, 5.05, 5.6, 6.25, 6.98, 7.53, 7.38, 7.15, 7.62, 7.2, 7.32, 7.2, 6.72, 7.2, 7.5, 7.62, 7.57, 7.68, 8.25, 8.35, 8.55, 8.45, 8.07, 7.2, 6.53, 6.25, 6.47, 7.5, 7.93, 8.12, 7.78, 7.65, 7.43, 7.62, 8.18, 9.05, 9.5, 9.7, 9.73, 9.48, 9.52, 9.62, 9.82, 9.9, 10.18, 10.3, 10.38, 10.18, 10.0, 9.88, 9.77, 9.98, 9.8, 9.75, 10.1, 10.27, 10.43, 10.68, 11.15, 11.65, 11.85, 11.73, 12.25, 12.52, 12.77, 12.6, 12.65, 12.93, 13.35, 13.48, 13.95, 15.48, 15.95, 15.65, 14.93, 15.55, 15.85, 15.0, 14.85, 15.1, 14.85, 14.48, 14.05, 13.98, 14.15, 14.32, 14.7, 14.98, 15.67, 16.75, 16.17, 16.1, 16.0, 16.1, 16.3, 16.35, 16.27, 15.88, 15.98, 16.27, 16.55, 17.17, 17.03, 16.42, 16.3, 16.5, 16.95, 18.27, 18.1, 17.98, 17.7, 17.4, 17.75, 18.02, 18.05, 17.83, 17.95, 17.73, 17.55, 17.27, 17.38, 17.12, 17.45, 17.17, 17.3, 17.5, 18.15, 18.15, 17.8, 17.73, 17.62, 17.92, 18.52, 19.05, 19.15, 18.55, 18.45, 18.7, 18.98, 18.67, 18.08, 17.75, 18.0, 18.4, 18.52, 18.4, 18.33, 18.83, 19.3, 19.25, 18.48, 18.4, 18.33, 18.17, 18.6, 18.95, 19.25, 19.33, 19.38, 19.75, 19.48, 19.38, 19.58, 19.3, 19.27, 18.8, 18.83, 19.1, 19.4, 19.73, 19.8, 19.1, 18.55, 18.25, 18.3, 18.25, 18.15, 18.12, 18.25, 18.33, 18.35, 18.7, 18.73, 18.4, 17.95, 17.8, 17.67, 17.55, 17.85, 17.83, 17.48, 17.02, 16.5, 16.08, 15.45, 15.6, 15.43, 15.12, 14.98, 15.38, 14.93, 14.77, 14.23, 13.88, 13.62, 13.43, 13.32, 13.88, 13.65, 13.77, 14.32, 14.3, 13.68, 13.82, 13.05, 12.68, 12.35, 12.3, 12.65, 13.0, 13.18, 13.38, 13.77, 13.85, 12.62, 12.25, 12.9, 13.32, 13.1, 13.57, 13.52, 13.23, 13.48, 13.55, 13.68, 13.75, 13.38, 13.27, 13.05, 12.65, 12.52, 11.58, 11.7, 11.77, 12.07, 12.02, 11.85, 11.85, 10.9, 10.23, 9.45, 9.7, 10.7, 10.65, 10.23, 9.73, 7.68, 6.82, 7.82, 8.45, 8.73, 8.9, 8.25, 8.32, 8.55, 9.12, 9.1, 8.1, 7.43, 7.07, 6.5, 6.57, 7.1, 6.82, 6.65, 6.07, 5.45, 5.47, 5.32, 4.43, 3.8, 3.22, 2.9, 2.8, 2.75, 2.7, 4.15, 6.07, 6.65, 7.25, 7.68, 7.4, 7.68, 6.6, 6.82, 7.7, 7.25, 8.23, 9.18, 9.2, 8.05, 8.2, 8.85, 9.02, 8.98, 8.65, 8.0, 7.6, 8.0, 8.35, 8.57, 7.93, 7.82, 6.98, 6.18, 4.9, 4.55, 4.38, 4.0, 3.75, 3.9, 4.33, 4.25, 4.12, 4.83, 4.8, 4.85, 5.28, 5.3, 5.38, 6.03, 6.7, 6.88, 7.1, 6.28, 5.28, 4.18, 3.5, 3.3, 4.53, 5.75, 6.15, 5.57, 5.0, 5.38, 6.8, 7.53, 7.95, 7.48, 7.65, 7.62, 7.75, 6.93, 6.4, 5.43, 4.75, 4.65, 4.38, 4.15, 4.7, 5.6, 5.88, 5.8, 5.8, 5.03, 4.78, 4.8, 4.43, 4.68, 6.38, 7.28, 6.53, 6.53, 6.95, 8.2, 8.65, 8.5, 8.62, 8.77, 9.18, 9.7, 9.42, 9.07, 8.25, 7.38, 7.68, 8.88, 9.43, 9.65, 9.38, 8.55, 7.88, 7.92, 7.97, 8.27, 8.98, 9.75, 9.8, 9.7, 9.9, 9.7, 9.35, 9.75, 9.77, 9.9, 10.02, 10.02, 9.88, 9.85, 10.3, 10.52, 10.82, 10.3, 9.8, 9.57, 10.23, 10.65, 10.93, 11.35, 11.52, 11.18, 11.48, 12.43, 12.65, 13.68, 13.98, 14.15, 13.85, 13.8, 13.9, 14.35, 14.93, 14.85, 13.98, 13.88, 13.88, 14.1, 14.57, 15.03, 15.85, 16.1, 14.95, 14.6, 14.57, 14.88, 15.48, 15.43, 15.18, 15.27, 15.68, 15.2, 15.3, 15.7, 15.98, 16.12, 16.45, 16.67, 17.08, 17.75, 18.2, 18.6, 18.58, 18.42, 18.33, 18.4, 18.45, 18.77, 18.58, 19.17, 19.48, 19.2, 18.7, 19.4, 20.1, 20.33, 19.45, 18.92, 18.75, 18.35, 18.25, 17.77, 17.73, 17.23, 17.02, 16.7, 17.45, 18.75, 19.83, 19.62, 19.9, 19.98, 19.27, 18.8, 19.23, 18.9, 18.75, 18.42, 18.85, 18.8, 18.73, 18.15, 17.67, 18.08, 17.6, 17.62, 17.67, 18.12, 18.52, 18.2, 18.0, 17.85, 17.75, 18.08, 18.0, 17.92, 17.48, 17.2, 17.35, 17.58, 17.92, 18.55, 19.22, 18.98, 18.65, 19.1, 18.77, 18.98, 19.12, 20.0, 19.78, 20.08, 19.9, 19.52, 19.67, 19.65, 19.1, 18.3, 17.88, 17.92, 17.3, 17.05, 17.2, 17.62, 17.65, 17.85, 18.33, 18.95, 19.23, 19.3, 19.58, 19.85, 20.2, 20.05, 19.67, 18.55, 17.98, 17.85, 18.23, 18.0, 17.67, 17.42, 17.35, 16.92, 16.65, 16.17, 15.98, 16.6, 16.62, 17.05, 16.92, 16.17, 16.52, 16.83, 17.1, 16.15, 15.93, 15.88, 16.25, 16.62, 16.4, 16.62, 16.55, 16.5, 16.48, 16.2, 14.57, 13.93, 13.5, 12.95, 13.43, 14.02, 13.6, 13.8, 13.3, 13.48, 12.92, 12.68, 12.88, 12.7, 12.85, 12.62, 12.62, 12.7, 12.55, 12.0, 11.65, 11.6, 11.55, 11.32, 11.23, 11.27, 10.93, 10.42, 10.2, 9.98, 10.68, 11.05, 10.93, 10.2, 9.5, 10.12, 10.82, 10.52, 9.73, 9.62, 9.98, 9.1, 8.45, 8.48, 7.85, 7.22, 6.9, 6.68, 5.98, 5.85, 5.83, 5.83, 6.05, 5.6, 5.6, 6.47, 7.22, 7.8, 7.9, 8.2, 8.25, 8.05, 8.1, 8.5, 8.35, 8.42, 8.32, 8.65, 8.5, 8.6, 9.0, 9.65, 9.7, 8.57, 8.57, 8.93, 8.65, 8.52, 8.42, 8.32, 8.92, 8.8, 8.5, 8.1, 7.15, 5.8, 4.82, 4.03, 3.47, 3.17, 3.38, 4.25, 4.43, 4.12, 3.88, 3.55, 3.2, 3.1, 3.07, 4.32, 5.22, 6.12, 6.18, 6.65, 6.05, 5.18, 4.93, 5.57, 6.05, 6.2, 6.1, 6.82, 7.47, 7.83, 7.97, 8.05, 7.05, 6.85, 7.78, 8.23, 8.02, 7.4, 7.68, 8.73, 9.12, 9.15, 9.27, 9.05, 8.68, 8.77, 9.0, 8.8, 7.97, 7.7, 7.68, 7.2, 6.97, 7.47, 7.9, 7.95, 8.25, 8.82, 8.23, 8.05, 8.07, 7.53, 7.32, 7.88, 8.1, 8.18, 8.73, 9.1, 9.75, 9.88, 9.32, 9.78, 10.07, 10.25, 10.6, 10.43, 10.05, 9.38, 9.0, 9.07, 9.4, 9.52, 9.43, 9.82, 9.77, 10.15, 9.95, 10.1, 10.43, 10.75, 11.12, 11.95, 12.38, 12.55, 12.07, 12.05, 12.3, 12.82, 13.1, 12.43, 11.32, 10.95, 10.73, 10.93, 10.9, 10.75, 10.5, 9.98, 10.35, 10.15, 10.25, 10.85, 11.52, 11.3, 12.43, 13.82, 15.18, 15.18, 13.68, 13.95, 14.0, 14.57, 14.7, 14.4, 14.7, 15.3, 15.77, 16.55, 16.8, 16.83, 16.75, 16.83, 16.72, 17.0, 16.8, 16.65, 16.77, 16.55, 16.33, 16.3, 16.33, 16.33, 16.25, 16.4, 16.43, 16.67, 16.62, 16.3, 16.27, 16.55, 16.92, 16.42, 16.05, 16.02, 15.83, 15.73, 15.25, 15.52, 15.52, 15.82, 15.88, 15.82, 15.98, 16.35, 16.33, 16.35, 16.6, 16.7, 16.9, 17.33, 18.02, 18.7, 19.3, 19.3, 18.65, 18.83, 18.55, 18.23, 17.58, 17.47, 17.45, 17.33, 17.15, 17.25, 18.17, 18.9, 18.83, 18.3, 17.65, 17.62, 18.1, 18.8, 18.67, 19.02, 19.67, 20.17, 20.5, 19.88, 20.07, 19.9, 19.58, 19.73, 19.6, 19.67, 19.65, 20.03, 20.38, 20.3, 20.15, 20.3, 19.98, 19.83, 19.95, 20.17, 19.88, 20.0, 19.65, 20.12, 20.52, 21.2, 21.38, 20.85, 20.55, 20.23, 20.12, 19.5, 19.45, 19.12, 18.62, 18.7, 18.7, 18.73, 18.27, 18.23, 18.58, 19.17, 19.05, 18.88, 19.0, 19.67, 19.98, 19.73, 19.48, 19.73, 19.92, 20.15, 19.38, 18.55, 16.95, 16.42, 16.0, 15.6, 15.6, 16.0, 16.17, 16.42, 16.83, 16.98, 16.9, 16.9, 17.2, 16.85, 16.67, 16.62, 16.25, 15.35, 14.93, 14.57, 14.7, 14.7, 14.2, 13.52, 13.1, 12.88, 13.52, 14.38, 14.68, 14.62, 14.05, 13.15, 13.1, 12.95, 12.32, 12.73, 13.82, 14.55, 14.73, 14.68, 14.73, 14.32, 13.77, 13.07, 13.02, 13.32, 13.75, 13.23, 13.52, 13.88, 14.07, 13.42, 13.27, 12.23, 11.2, 10.75, 10.93, 11.35, 10.88, 10.57, 10.85, 11.18, 11.1, 10.5, 10.9, 10.9, 10.88, 10.73, 9.95, 9.32, 8.45, 7.75, 7.12, 6.95, 7.75, 9.23, 9.25, 8.75, 8.7, 8.12, 7.28, 7.47, 8.05, 8.52, 7.72, 7.53, 8.08, 8.18, 8.25, 8.07, 7.88, 7.6, 7.35, 7.25, 6.82, 7.0, 7.98, 8.23, 8.35, 9.02, 8.1, 7.88, 8.07, 7.4, 7.05, 7.55, 8.2, 8.07, 7.35, 6.97, 7.15, 7.4, 7.32, 7.2, 6.28, 5.82, 5.55, 6.5, 6.53, 5.88, 5.45, 5.15, 4.68, 4.2, 4.22, 3.9, 4.75, 5.72, 5.57, 5.15, 4.8, 4.62, 4.5, 4.32, 4.78, 5.35, 5.93, 5.68, 5.78, 6.1, 6.47, 6.5, 5.8, 4.82, 4.1, 3.5, 3.22, 3.75, 4.72, 4.72, 4.6, 4.7, 4.6, 4.4, 4.77, 4.95, 4.3, 4.03, 4.05, 3.55, 3.05, 2.78, 2.85, 4.68, 6.2, 6.77, 7.25, 7.43, 7.47, 7.22, 7.05, 6.62, 6.0, 5.5, 5.22, 5.18, 5.35, 5.8, 6.2, 6.43, 6.5, 6.8, 6.55, 6.22, 5.8, 5.6, 5.57, 6.1, 5.95, 5.65, 6.23, 6.95, 7.97, 8.52, 8.85, 8.85, 8.88, 8.4, 8.15, 8.18, 8.45, 8.73, 8.62, 8.65, 9.07, 9.35, 9.65, 9.38, 9.0, 9.12, 9.27, 9.55, 9.68, 10.0, 10.7, 10.9, 11.18, 11.25, 10.92, 10.9, 10.85, 10.82, 11.25, 11.38, 11.57, 11.62, 11.62, 11.68, 11.88, 12.05, 12.2, 12.73, 13.27, 13.8, 13.77, 13.35, 12.55, 12.02, 12.07, 12.25, 12.27, 12.55, 13.07, 13.65, 13.65, 13.62, 13.7, 13.55, 13.88, 14.3, 14.1, 14.57, 14.73, 14.45, 14.02, 13.3, 13.68, 14.2, 14.27, 14.43, 14.4, 14.75, 15.65, 16.67, 16.12, 15.88, 15.75, 15.43, 15.5, 15.43, 15.4, 15.07, 15.38, 15.25, 16.3, 17.38, 18.67, 18.17, 17.8, 17.8, 18.02, 18.23, 18.23, 18.45, 18.77, 18.5, 18.02, 17.85, 18.55, 18.2, 18.35, 18.8, 19.55, 20.17, 20.92, 20.45, 19.58, 19.2, 18.73, 18.42, 17.98, 18.17, 18.73, 18.98, 19.2, 19.58, 19.48, 19.35, 18.95, 18.67, 18.95, 19.42, 19.7, 19.45, 19.3, 19.15, 19.25, 19.38, 19.17, 19.23, 18.73, 18.75, 18.22, 17.8, 17.45, 17.35, 17.12, 17.33, 17.48, 17.08, 17.17, 17.52, 17.8, 17.85, 18.12, 18.8, 19.12, 19.23, 18.8, 18.4, 17.3, 16.92, 17.23, 17.33, 16.83, 16.48, 16.17, 15.82, 15.7, 15.92, 16.3, 16.53, 16.12, 15.82, 16.08, 16.38, 16.77, 16.77, 17.22, 17.33, 17.12, 17.3, 17.83, 17.65, 16.45, 16.83, 16.4, 16.17, 15.93, 16.42, 16.12, 15.85, 16.17, 16.75, 17.0, 17.0, 15.85, 14.95, 14.7, 14.3, 14.32, 14.52, 14.57, 14.4, 14.32, 14.1, 13.93, 13.95, 13.98, 13.75, 13.7, 14.27, 14.8, 14.73, 14.62, 14.85, 14.95, 15.15, 15.07, 14.95, 14.65, 14.45, 14.23, 13.52, 13.65, 14.12, 14.12, 13.88, 13.28, 12.73, 12.25, 11.82, 11.52, 11.88, 12.1, 11.7, 11.73, 11.77, 11.57, 11.8, 12.12, 11.98, 11.58, 11.25, 11.57, 11.27, 11.32, 11.3, 11.25, 11.68, 11.65, 11.23, 10.2, 9.38, 9.18, 8.75, 8.25, 7.47, 7.03, 7.05, 7.45, 7.53, 7.22, 7.9, 8.62, 8.8, 8.27, 8.05, 8.27, 8.27, 8.23, 8.07, 8.6, 9.25, 9.77, 10.07, 9.88, 9.43, 9.4, 8.6, 8.68, 8.5, 8.57, 8.65, 9.15, 8.7, 8.48, 8.3, 7.72, 7.07, 5.9, 4.95, 4.4, 4.32, 4.4, 4.65, 4.4, 4.82, 4.97, 4.65, 4.6, 4.55, 4.5, 4.43, 4.3, 3.97, 3.78, 3.55, 3.4, 4.28, 5.3, 5.45, 6.07, 6.2, 5.7, 5.15, 4.85, 4.7, 4.6, 4.53, 4.78, 4.38, 3.8, 3.88, 4.32, 4.43, 4.17, 4.45, 5.07, 5.3, 4.65, 4.47, 4.82, 4.65, 5.22, 5.43, 5.55, 5.22, 5.75, 6.53, 6.57, 5.62, 4.55, 4.2, 4.7, 4.85, 4.5, 4.6, 5.47, 6.62, 7.22, 7.53, 8.1, 8.27, 8.4, 8.75, 8.18, 8.15, 7.83, 7.83, 7.72, 7.72, 8.2, 8.6, 8.43, 8.45, 8.62, 8.3, 8.42, 8.2, 7.65, 7.8, 8.05, 8.55, 8.3, 8.4, 8.27, 8.27, 8.52, 8.77, 8.98, 8.35, 7.7, 8.02, 8.02, 8.12, 8.0, 8.4, 8.88, 8.57, 9.15, 9.88, 9.52, 9.3, 9.38, 9.93, 10.52, 10.07, 9.98, 9.95, 9.82, 10.07, 10.73, 11.05, 11.38, 11.02, 10.77, 10.85, 10.73, 10.88, 11.12, 11.35, 11.52, 11.52, 11.68, 11.68, 11.75, 11.95, 12.23, 13.05, 13.25, 13.32, 12.77, 12.43, 12.57, 12.6, 12.73, 12.65, 12.0, 12.0, 12.12, 11.8, 11.7, 11.67, 12.23, 12.68, 13.05, 13.77, 14.57, 15.5, 16.25, 16.15, 16.42, 16.8, 16.35, 16.67, 16.9, 17.05, 16.67, 16.77, 16.48, 16.35, 16.2, 16.38, 16.2, 16.23, 16.0, 15.65, 15.4, 15.55, 16.1, 15.77, 15.82, 16.27, 17.35, 18.0, 19.4, 20.8)  # fmt: skip  # noqa: FURB152


def generate_example_data(
    *,
    date_start: str = "2020-01-01",
    date_end: str = "2021-12-31",
    date_freq: str = "D",
    flow_mean: float = 100.0,  # m3/day
    flow_amplitude: float = 30.0,  # m3/day
    flow_noise: float = 10.0,  # m3/day
    cin_method: str = "synthetic",
    cin_mean: float = 12.0,
    cin_amplitude: float = 8.0,
    measurement_noise: float = 1.0,
    aquifer_pore_volumes: npt.ArrayLike | None = None,
    aquifer_pore_volume_gamma_mean: float | None = None,
    aquifer_pore_volume_gamma_std: float | None = None,
    aquifer_pore_volume_gamma_loc: float | None = None,
    aquifer_pore_volume_gamma_nbins: int | None = None,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float | None = None,
    longitudinal_dispersivity: float | None = None,
    streamline_length: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Generate synthetic concentration/temperature and flow data for groundwater transport.

    Creates a synthetic dataset with seasonal flow patterns, input concentration (cin),
    and output concentration (cout) computed via gamma-distributed pore volume transport.
    When ``molecular_diffusivity``, ``longitudinal_dispersivity``, and ``streamline_length``
    are provided, the diffusion module is used instead of pure advection.

    Parameters
    ----------
    date_start, date_end : str
        Start and end dates for the generated time series (YYYY-MM-DD).
    date_freq : str, default "D"
        Frequency string for pandas.date_range.
    flow_mean : float, default 100.0
        Mean flow rate [m³/day].
    flow_amplitude : float, default 30.0
        Seasonal amplitude of flow rate [m³/day].
    flow_noise : float, default 10.0
        Random noise level for flow rate [m³/day].
    cin_method : str, default "synthetic"
        Method for generating infiltration concentration. Options:

        - ``"synthetic"``: Seasonal sinusoidal pattern defined by ``cin_mean`` and
          ``cin_amplitude``. Measurement noise is applied.
        - ``"constant"``: Constant value equal to ``cin_mean``. Measurement noise
          is still applied.
        - ``"soil_temperature"``: Real soil temperature data from KNMI station 260.
    cin_mean : float, default 12.0
        Mean value of infiltrating concentration.
    cin_amplitude : float, default 8.0
        Seasonal amplitude of infiltration concentration (only used for
        ``"synthetic"`` method).
    measurement_noise : float, default 1.0
        Standard deviation of the Gaussian measurement noise applied
        independently to ``cin`` and ``cout``. Because the two noise draws are
        independent, applying the forward operator to ``df['cin']`` does not
        exactly reproduce ``df['cout']`` when ``measurement_noise > 0``; the
        underlying noiseless signals remain consistent.
    aquifer_pore_volumes : array-like or None, default None
        Discrete aquifer pore volumes [m³] representing the distribution of
        residence times. When provided, the gamma distribution is bypassed and
        none of the ``aquifer_pore_volume_gamma_*`` parameters may be passed.
        When ``None``, the pore volume distribution is built from the gamma
        parameters below.
    aquifer_pore_volume_gamma_mean : float or None, default None
        Mean pore volume of the aquifer gamma distribution [m³] (default 1000.0
        when unset). Must be strictly greater than
        ``aquifer_pore_volume_gamma_loc``. Mutually exclusive with
        ``aquifer_pore_volumes``.
    aquifer_pore_volume_gamma_std : float or None, default None
        Standard deviation of aquifer pore volume gamma distribution [m³]
        (default 200.0 when unset; invariant under the ``loc`` shift).
        Mutually exclusive with ``aquifer_pore_volumes``.
    aquifer_pore_volume_gamma_loc : float or None, default None
        Location (minimum pore volume) of the aquifer gamma distribution [m³]
        (default 0.0 when unset). Must satisfy ``0 <= loc < mean``. Mutually
        exclusive with ``aquifer_pore_volumes``.
    aquifer_pore_volume_gamma_nbins : int or None, default None
        Number of bins to discretize the aquifer pore volume gamma distribution
        (default 250 when unset). Mutually exclusive with
        ``aquifer_pore_volumes``.
    retardation_factor : float, default 1.0
        Retardation factor for transport.
    molecular_diffusivity : float or None, default None
        Effective molecular diffusivity [m²/day]. When provided together with
        ``longitudinal_dispersivity`` and ``streamline_length``, the diffusion
        module is used instead of pure advection. For solutes, typically ~1e-5
        m²/day (negligible). For heat, use thermal diffusivity ~0.01-0.1 m²/day.
    longitudinal_dispersivity : float or None, default None
        Longitudinal dispersivity [m]. Must be provided together with
        ``molecular_diffusivity`` and ``streamline_length``.
    streamline_length : float or None, default None
        Travel distance along the streamline [m]. Must be provided together
        with ``molecular_diffusivity`` and ``longitudinal_dispersivity``.
    rng : numpy.random.Generator, int, or None, default None
        Source of randomness for the synthetic flow noise, spill events, and
        measurement noise. Accepted in any form supported by
        :func:`numpy.random.default_rng`. Pass an integer (or
        :class:`numpy.random.Generator`) for reproducible output; ``None``
        draws fresh entropy each call.

    Returns
    -------
    tuple
        A tuple containing:

        - pandas.DataFrame: DataFrame with columns ``'flow'``, ``'cin'``,
          ``'cout'`` and metadata attributes for the aquifer parameters.
        - pandas.DatetimeIndex: Time edges (tedges) used for the flow
          calculations.

    Raises
    ------
    ValueError
        If ``cin_method`` is not one of the supported methods, if only some
        of the diffusion parameters are provided, or if ``aquifer_pore_volumes``
        is passed together with any ``aquifer_pore_volume_gamma_*`` parameter.

    See Also
    --------
    generate_temperature_example_data : Wrapper with thermal transport defaults.
    generate_ec_example_data : Wrapper with EC transport defaults.
    """
    rng = np.random.default_rng(rng)

    dates = pd.date_range(start=date_start, end=date_end, freq=date_freq).tz_localize("UTC")
    days = (dates - dates[0]).days.values

    # Generate flow data with seasonal pattern (higher in winter)
    seasonal_flow = flow_mean + flow_amplitude * np.sin(2 * np.pi * days / 365 + np.pi)
    flow = seasonal_flow + rng.normal(0, flow_noise, len(dates))

    min_days_for_spills = 60
    if len(dates) > min_days_for_spills:  # Only add spills for longer time series
        n_spills = int(rng.integers(6, 16))
        for _ in range(n_spills):
            spill_start = int(rng.integers(0, len(dates) - 30))
            spill_duration = int(rng.integers(15, 45))
            spill_magnitude = float(rng.uniform(2.0, 5.0))

            flow[spill_start : spill_start + spill_duration] /= spill_magnitude

    # Enforce a positive flow floor after spills so residence times remain finite.
    flow = np.maximum(flow, 5.0)

    # Generate infiltration concentration. nonoise is needed to compute cout.
    if cin_method == "synthetic":
        # Seasonal pattern with noise
        cin_nonoise = cin_mean + cin_amplitude * np.sin(2 * np.pi * days / 365)
        cin_values = cin_nonoise + rng.normal(0, measurement_noise, len(dates))
    elif cin_method == "constant":
        # Constant value
        cin_nonoise = np.full(len(dates), cin_mean)
        cin_values = cin_nonoise + rng.normal(0, measurement_noise, len(dates))
    elif cin_method == "soil_temperature":
        # Use the inline KNMI soil temperature data (already includes measurement noise).
        soil_temperature = pd.Series(
            _SOIL_TEMPERATURE_TB3,
            index=pd.date_range(
                start=_SOIL_TEMPERATURE_TB3_START, periods=len(_SOIL_TEMPERATURE_TB3), freq="D", tz="UTC"
            ),
            name="TB3",
        )
        cin_nonoise = cin_values = soil_temperature.resample(date_freq).mean()[dates].values
    else:
        msg = f"Unknown cin_method: {cin_method}"
        raise ValueError(msg)

    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Validate pore volume parameterization: either discrete volumes or gamma parameters, not both.
    gamma_set_by_user = [
        name
        for name, value in {
            "aquifer_pore_volume_gamma_mean": aquifer_pore_volume_gamma_mean,
            "aquifer_pore_volume_gamma_std": aquifer_pore_volume_gamma_std,
            "aquifer_pore_volume_gamma_loc": aquifer_pore_volume_gamma_loc,
            "aquifer_pore_volume_gamma_nbins": aquifer_pore_volume_gamma_nbins,
        }.items()
        if value is not None
    ]
    if aquifer_pore_volumes is not None and gamma_set_by_user:
        msg = (
            "aquifer_pore_volumes is mutually exclusive with the aquifer_pore_volume_gamma_* "
            f"parameters; got both aquifer_pore_volumes and {gamma_set_by_user}."
        )
        raise ValueError(msg)

    # Validate diffusion parameterization: all three parameters provided or none.
    diffusion_provided = (molecular_diffusivity, longitudinal_dispersivity, streamline_length)
    n_diffusion = sum(1 for p in diffusion_provided if p is not None)
    if 0 < n_diffusion < len(diffusion_provided):
        msg = "molecular_diffusivity, longitudinal_dispersivity, and streamline_length must all be provided together."
        raise ValueError(msg)
    # Validation above forbids partial-set states, so this conjunction is equivalent to any single check;
    # writing it in full lets the type checker narrow all three params to non-None inside the branches below.
    use_diffusion = (
        molecular_diffusivity is not None and longitudinal_dispersivity is not None and streamline_length is not None
    )

    # Fill in gamma defaults so downstream callers see concrete values (not used when
    # aquifer_pore_volumes is supplied, but kept in scope for the attrs block below).
    gamma_mean = aquifer_pore_volume_gamma_mean if aquifer_pore_volume_gamma_mean is not None else _DEFAULT_GAMMA_MEAN
    gamma_std = aquifer_pore_volume_gamma_std if aquifer_pore_volume_gamma_std is not None else _DEFAULT_GAMMA_STD
    gamma_loc = aquifer_pore_volume_gamma_loc if aquifer_pore_volume_gamma_loc is not None else _DEFAULT_GAMMA_LOC
    gamma_nbins = (
        aquifer_pore_volume_gamma_nbins if aquifer_pore_volume_gamma_nbins is not None else _DEFAULT_GAMMA_NBINS
    )

    # Compute cout. Branch on pore volume parameterization, then on diffusion.
    if aquifer_pore_volumes is not None:
        aquifer_pore_volumes_array = np.asarray(aquifer_pore_volumes, dtype=float)
        if use_diffusion:
            cout_values = diffusion_infiltration_to_extraction(
                cin=cin_nonoise,
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                aquifer_pore_volumes=aquifer_pore_volumes_array,
                streamline_length=streamline_length,
                molecular_diffusivity=molecular_diffusivity,
                longitudinal_dispersivity=longitudinal_dispersivity,
                retardation_factor=retardation_factor,
            )
        else:
            cout_values = infiltration_to_extraction(
                cin=cin_nonoise,
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                aquifer_pore_volumes=aquifer_pore_volumes_array,
                retardation_factor=retardation_factor,
            )
    elif use_diffusion:
        cout_values = diffusion_gamma_infiltration_to_extraction(
            cin=cin_nonoise,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            mean=gamma_mean,
            std=gamma_std,
            loc=gamma_loc,
            n_bins=gamma_nbins,
            streamline_length=streamline_length,
            molecular_diffusivity=molecular_diffusivity,
            longitudinal_dispersivity=longitudinal_dispersivity,
            retardation_factor=retardation_factor,
        )
    else:
        cout_values = gamma_infiltration_to_extraction(
            cin=cin_nonoise,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            mean=gamma_mean,
            std=gamma_std,
            loc=gamma_loc,
            n_bins=gamma_nbins,
            retardation_factor=retardation_factor,
        )

    # Add some noise to represent measurement errors
    cout_values += rng.normal(0, measurement_noise, len(dates))

    df = pd.DataFrame(
        data={"flow": flow, "cin": cin_values, "cout": cout_values},
        index=dates,
    )
    df.attrs.update({
        "description": "Example data for groundwater transport modeling",
        "source": "Synthetic data generated by gwtransport.examples.generate_example_data",
        "retardation_factor": retardation_factor,
        "date_start": date_start,
        "date_end": date_end,
        "date_freq": date_freq,
        "flow_mean": flow_mean,
        "flow_amplitude": flow_amplitude,
        "flow_noise": flow_noise,
        "cin_method": cin_method,
        "cin_mean": cin_mean,
        "cin_amplitude": cin_amplitude,
        "measurement_noise": measurement_noise,
    })
    if aquifer_pore_volumes is not None:
        df.attrs["aquifer_pore_volume_parameterization"] = "discrete"
        df.attrs["aquifer_pore_volumes"] = aquifer_pore_volumes_array
    else:
        alpha, beta = mean_std_loc_to_alpha_beta(mean=gamma_mean, std=gamma_std, loc=gamma_loc)
        df.attrs.update({
            "aquifer_pore_volume_parameterization": "gamma",
            "aquifer_pore_volume_gamma_mean": gamma_mean,
            "aquifer_pore_volume_gamma_std": gamma_std,
            "aquifer_pore_volume_gamma_loc": gamma_loc,
            "aquifer_pore_volume_gamma_alpha": alpha,
            "aquifer_pore_volume_gamma_beta": beta,
            "aquifer_pore_volume_gamma_nbins": gamma_nbins,
        })
    if molecular_diffusivity is not None:
        df.attrs["molecular_diffusivity"] = molecular_diffusivity
        df.attrs["longitudinal_dispersivity"] = longitudinal_dispersivity
        df.attrs["streamline_length"] = streamline_length

    return df, tedges


def generate_temperature_example_data(
    *,
    date_start: str = "2020-01-01",
    date_end: str = "2021-12-31",
    date_freq: str = "D",
    flow_mean: float = 100.0,
    flow_amplitude: float = 30.0,
    flow_noise: float = 10.0,
    cin_method: str = "synthetic",
    cin_mean: float = 12.0,
    cin_amplitude: float = 8.0,
    measurement_noise: float = 1.0,
    aquifer_pore_volumes: npt.ArrayLike | None = None,
    aquifer_pore_volume_gamma_mean: float | None = None,
    aquifer_pore_volume_gamma_std: float | None = None,
    aquifer_pore_volume_gamma_loc: float | None = None,
    aquifer_pore_volume_gamma_nbins: int | None = None,
    retardation_factor: float = 2.0,
    molecular_diffusivity: float = 0.05,
    longitudinal_dispersivity: float = 1.0,
    streamline_length: float = 100.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Generate synthetic temperature and flow data for groundwater transport examples.

    Convenience wrapper around :func:`generate_example_data` with sensible
    defaults for temperature transport: thermal retardation factor, thermal
    diffusivity, longitudinal dispersivity, and streamline length.

    Typical parameter values for temperature transport in various sand types:

    +---------------------------------+------------+-------------+--------------------+
    | Parameter                       | Fine sand  | Medium sand | Coarse sand/gravel |
    +=================================+============+=============+====================+
    | retardation_factor R            | 2.0--3.0   | 1.5--2.5    | 1.2--2.0           |
    +---------------------------------+------------+-------------+--------------------+
    | molecular_diffusivity (m²/day)  | 0.03--0.06 | 0.05--0.08  | 0.08--0.12         |
    +---------------------------------+------------+-------------+--------------------+
    | longitudinal_dispersivity (m)   | 0.1--1.0   | 0.5--5.0    | 1.0--10.0          |
    +---------------------------------+------------+-------------+--------------------+
    | streamline_length (m)           | site-specific                                 |
    +---------------------------------+------------+-------------+--------------------+

    Parameters
    ----------
    retardation_factor : float, default 2.0
        Thermal retardation factor.
    molecular_diffusivity : float, default 0.05
        Thermal diffusivity [m²/day].
    longitudinal_dispersivity : float, default 1.0
        Longitudinal dispersivity [m].
    streamline_length : float, default 100.0
        Travel distance along the streamline [m].

    Returns
    -------
    tuple
        See :func:`generate_example_data`.

    See Also
    --------
    generate_example_data : Generic version with full parameter control.
    generate_ec_example_data : Wrapper with EC transport defaults.

    Notes
    -----
    All other parameters are forwarded unchanged to :func:`generate_example_data`;
    see that function for their descriptions.
    """
    return generate_example_data(
        date_start=date_start,
        date_end=date_end,
        date_freq=date_freq,
        flow_mean=flow_mean,
        flow_amplitude=flow_amplitude,
        flow_noise=flow_noise,
        cin_method=cin_method,
        cin_mean=cin_mean,
        cin_amplitude=cin_amplitude,
        measurement_noise=measurement_noise,
        aquifer_pore_volumes=aquifer_pore_volumes,
        aquifer_pore_volume_gamma_mean=aquifer_pore_volume_gamma_mean,
        aquifer_pore_volume_gamma_std=aquifer_pore_volume_gamma_std,
        aquifer_pore_volume_gamma_loc=aquifer_pore_volume_gamma_loc,
        aquifer_pore_volume_gamma_nbins=aquifer_pore_volume_gamma_nbins,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        streamline_length=streamline_length,
        rng=rng,
    )


def generate_ec_example_data(
    *,
    date_start: str = "2020-01-01",
    date_end: str = "2021-12-31",
    date_freq: str = "D",
    flow_mean: float = 100.0,
    flow_amplitude: float = 30.0,
    flow_noise: float = 10.0,
    cin_method: str = "synthetic",
    cin_mean: float = 500.0,
    cin_amplitude: float = 150.0,
    measurement_noise: float = 10.0,
    aquifer_pore_volumes: npt.ArrayLike | None = None,
    aquifer_pore_volume_gamma_mean: float | None = None,
    aquifer_pore_volume_gamma_std: float | None = None,
    aquifer_pore_volume_gamma_loc: float | None = None,
    aquifer_pore_volume_gamma_nbins: int | None = None,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 5e-5,
    longitudinal_dispersivity: float = 1.0,
    streamline_length: float = 100.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Generate synthetic electrical conductivity and flow data for groundwater transport examples.

    Convenience wrapper around :func:`generate_example_data` with sensible
    defaults for electrical conductivity (EC) transport. EC is a conservative
    tracer: dissolved ions travel at water velocity without retardation.

    Typical parameter values for EC (dissolved ion) transport in various sand
    types. The molecular diffusivity represents effective ionic diffusion in
    porous media (free-water D_0 reduced by porosity/tortuosity). It is
    negligible compared to microdispersion at field scale.

    +---------------------------------+----------------+----------------+--------------------+
    | Parameter                       | Fine sand      | Medium sand    | Coarse sand/gravel |
    +=================================+================+================+====================+
    | retardation_factor R            | 1.0            | 1.0            | 1.0                |
    +---------------------------------+----------------+----------------+--------------------+
    | molecular_diffusivity (m²/day)  | 3e-5 -- 5e-5   | 4e-5 -- 8e-5   | 5e-5 -- 1e-4       |
    +---------------------------------+----------------+----------------+--------------------+
    | longitudinal_dispersivity (m)   | 0.1--1.0       | 0.5--5.0       | 1.0--10.0          |
    +---------------------------------+----------------+----------------+--------------------+
    | streamline_length (m)           | site-specific                                        |
    +---------------------------------+----------------+----------------+--------------------+

    Parameters
    ----------
    cin_mean : float, default 500.0
        Mean infiltration EC [uS/cm, typical surface water EC].
    cin_amplitude : float, default 150.0
        Seasonal amplitude of infiltration EC [uS/cm].
    measurement_noise : float, default 10.0
        Standard deviation of the Gaussian measurement noise [uS/cm].
    retardation_factor : float, default 1.0
        Retardation factor (1.0 for a conservative tracer).
    molecular_diffusivity : float, default 5e-5
        Effective ionic diffusion [m²/day].
    longitudinal_dispersivity : float, default 1.0
        Longitudinal dispersivity [m].
    streamline_length : float, default 100.0
        Travel distance along the streamline [m].

    Returns
    -------
    tuple
        See :func:`generate_example_data`.

    See Also
    --------
    generate_example_data : Generic version with full parameter control.
    generate_temperature_example_data : Wrapper with thermal transport defaults.

    Notes
    -----
    All other parameters are forwarded unchanged to :func:`generate_example_data`;
    see that function for their descriptions.
    """
    return generate_example_data(
        date_start=date_start,
        date_end=date_end,
        date_freq=date_freq,
        flow_mean=flow_mean,
        flow_amplitude=flow_amplitude,
        flow_noise=flow_noise,
        cin_method=cin_method,
        cin_mean=cin_mean,
        cin_amplitude=cin_amplitude,
        measurement_noise=measurement_noise,
        aquifer_pore_volumes=aquifer_pore_volumes,
        aquifer_pore_volume_gamma_mean=aquifer_pore_volume_gamma_mean,
        aquifer_pore_volume_gamma_std=aquifer_pore_volume_gamma_std,
        aquifer_pore_volume_gamma_loc=aquifer_pore_volume_gamma_loc,
        aquifer_pore_volume_gamma_nbins=aquifer_pore_volume_gamma_nbins,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        streamline_length=streamline_length,
        rng=rng,
    )


def generate_example_deposition_timeseries(
    *,
    date_start: str = "2018-01-01",
    date_end: str = "2023-12-31",
    freq: str = "D",
    base: float = 0.8,
    seasonal_amplitude: float = 0.3,
    noise_scale: float = 0.1,
    event_dates: npt.ArrayLike | pd.DatetimeIndex | None = None,
    event_magnitude: float = 3.0,
    event_duration: int = 30,
    event_decay_scale: float = 10.0,
    ensure_non_negative: bool = True,
    rng: np.random.Generator | int | None = None,
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """
    Generate synthetic deposition timeseries for groundwater transport examples.

    Parameters
    ----------
    date_start, date_end : str
        Start and end dates for the generated time series (YYYY-MM-DD).
    freq : str
        Frequency string for pandas.date_range (default 'D').
    base : float
        Baseline deposition rate (ng/m²/day).
    seasonal_amplitude : float
        Amplitude of the annual seasonal sinusoidal pattern (ng/m²/day).
    noise_scale : float
        Standard deviation of Gaussian noise added to the signal (ng/m²/day).
    event_dates : list-like or None
        Dates (strings or pandas-compatible) at which to place episodic events.
        Time-zone-naive entries are interpreted as UTC to match the generated
        ``dates`` index. If None, a sensible default list is used.
    event_magnitude : float
        Peak deposition added at event onset (ng/m²/day). Decays exponentially
        over ``event_duration`` days at rate ``event_decay_scale``.
    event_duration : int
        Duration of each event in days.
    event_decay_scale : float
        Decay scale used in the exponential decay for event time series.
    ensure_non_negative : bool
        If True, negative values are clipped to zero.
    rng : numpy.random.Generator, int, or None, default None
        Source of randomness for the additive Gaussian noise. Accepted in any
        form supported by :func:`numpy.random.default_rng`. Pass an integer
        (or :class:`numpy.random.Generator`) for reproducible output; ``None``
        draws fresh entropy each call.

    Returns
    -------
    tuple
        A tuple containing:

        - pandas.Series: Deposition time series (ng/m²/day) indexed by UTC
          timestamps.
        - pandas.DatetimeIndex: Time bin edges (n+1 edges for n values).

    Raises
    ------
    ValueError
        If ``event_decay_scale`` or ``event_duration`` is not positive, or if any
        ``event_dates`` entry falls outside the generated ``dates`` range.

    See Also
    --------
    gwtransport.deposition.deposition_to_extraction : Forward operator consuming this data.
    gwtransport.deposition.extraction_to_deposition : Inverse operator.
    """
    if event_decay_scale <= 0:
        msg = f"event_decay_scale must be positive, got {event_decay_scale}"
        raise ValueError(msg)
    if event_duration <= 0:
        msg = f"event_duration must be positive, got {event_duration}"
        raise ValueError(msg)

    rng = np.random.default_rng(rng)

    dates = pd.date_range(date_start, date_end, freq=freq).tz_localize("UTC")
    n_dates = len(dates)
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n_dates)

    # Base deposition rate with seasonal and event patterns
    seasonal_pattern = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_dates) / 365.25)
    noise = noise_scale * rng.normal(0, 1, n_dates)

    # Default event dates if not provided
    if event_dates is None:
        event_dates = ["2020-06-15", "2021-03-20", "2021-09-10", "2022-07-05"]
    event_dates_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(event_dates)))
    # Match the timezone of `dates` so naive user input (and the string defaults)
    # can be compared against the tz-aware index in `get_indexer`.
    if event_dates_index.tz is None:
        event_dates_index = event_dates_index.tz_localize(dates.tz)
    else:
        event_dates_index = event_dates_index.tz_convert(dates.tz)

    out_of_range = (event_dates_index < dates[0]) | (event_dates_index > dates[-1])
    if out_of_range.any():
        msg = (
            f"event_dates contains {out_of_range.sum()} date(s) outside the dates range "
            f"[{dates[0]}, {dates[-1]}]: {event_dates_index[out_of_range].tolist()}"
        )
        raise ValueError(msg)

    # Vectorized event accumulation. For each event start, scatter a ``(n_events, event_duration)``
    # decay block into ``event`` via ``np.add.at`` so overlapping events sum correctly. The
    # boundary mask drops indices that fall past the end of the series (preserves the loop's
    # ``min(event_idx + event_duration, n_dates)`` clipping).
    starts = dates.get_indexer(event_dates_index, method="nearest")
    cols = np.arange(event_duration)
    flat_indices = starts[:, None] + cols[None, :]
    valid = flat_indices < n_dates
    decay_block = np.broadcast_to(event_magnitude * np.exp(-cols / event_decay_scale), flat_indices.shape)
    event = np.zeros(n_dates)
    np.add.at(event, flat_indices[valid], decay_block[valid])

    # Combine all components
    total = base + seasonal_pattern + noise + event
    if ensure_non_negative:
        total = np.maximum(total, 0.0)

    series = pd.Series(data=total, index=dates, name="deposition")
    series.attrs.update({
        "description": "Example deposition time series for groundwater transport modeling",
        "source": "Synthetic data generated by gwtransport.examples.generate_example_deposition_timeseries",
        "base": base,
        "seasonal_amplitude": seasonal_amplitude,
        "noise_scale": noise_scale,
        "event_dates": [str(d.date()) for d in event_dates_index],
        "event_magnitude": event_magnitude,
        "event_duration": event_duration,
        "event_decay_scale": event_decay_scale,
        "date_start": date_start,
        "date_end": date_end,
        "date_freq": freq,
    })

    return series, tedges
