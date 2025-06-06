"""
Example 3: Estimate the log removal of pathogens.

The pathogen removal in bank filtration systems is heavily
related to the residence time with a logarithmic relation.
In this example, the pathogen removal rate is assumed to:
- solely depend on residence time,
- follows a logarithmic curve,
- be constant; it does not vary in space nor time.

In this case, where the pathogen removal follows a logarithmic relation with the residence time,
the mean removal of the extracted water is heavily weighted towards the shorter residence times.

Two cases are explored here:
1. Varying flow: Timeseries of a bank filtration system in production
2. Constant flow: Deep dive into the effect of a bigger spread in the residence time
"""

import numpy as np
from example_data_generation import generate_synthetic_data

from gwtransport import advection
from gwtransport import gamma as gamma_utils

# %% 1. Varying flow
# ------------------
# Outline:
# - We generate a flow timeseries
# - with a log-removal-rate value from literature we compute the log-removal of
#   the extracted water.
mean, std = 8000.0, 400.0  # m3
mean_flow = 120.0  # m3/day

df = generate_synthetic_data(
    start_date="2020-01-01",
    end_date="2025-12-31",
    mean_flow=mean_flow,  # m3/day
    flow_amplitude=40.0,  # m3/day
    flow_noise=5.0,  # m3/day
    mean_temp_infiltration=12.0,  # °C
    temp_infiltration_amplitude=8.0,  # °C
    aquifer_pore_volume=mean,  # m3
    aquifer_pore_volume_std=std,  # m3
    retardation_factor=1.0,
)

# Discretize the aquifer pore volume distribution in bins
# alpha, beta = gamma_utils.mean_std_to_alpha_beta(mean, std)
bins = gamma_utils.bins(mean=mean, std=std, n_bins=1000)

# Compute the residence time, similar to Example 2
rt_forward_rf1 = advection.residence_time(
    flow=df.flow,
    flow_tend=df.index,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=1.0,  # Note that we are computing the rt of the water, not the heat transport
    direction="infiltration",
)

# Compute the log-removal of pathogens
log_removal_rate = 0.5  # Log-removal rate [1/day]
log_removal = np.log(10) * log_removal_rate * rt_forward_rf1
log_removal_mean = log_removal.mean(axis=0)

# %% 2. Constant flow
# -------------------
# Notes:
# If flow is constant, a gamma distribution of the aquifer
# pore volume results in a gamma distribution of the residence
# time, which allows for the direct calculation of the mean removal
# of pathogens.
