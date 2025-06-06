"""
Example 2: Estimation of the residence time distribution using synthetic data.

This notebook demonstrates how to estimate the residence time distribution from the
aquifer pore volume distribution (See: Example 1) and the flow rate. The residence time is the time
it takes for water to travel through the aquifer. It is a key parameter in
groundwater transport modeling and is important for understanding the fate and
transport of contaminants in groundwater systems.

Assumptions:
- The aquifer pore volume distribution is constant over time.
- Advection is the only transport process.

Two types of residence time are computed:
- Forward: In how many days from now is the water extracted?
- Backward: How many days ago was the water infiltrated?
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from example_data_generation import generate_synthetic_data

from gwtransport import advection
from gwtransport import gamma as gamma_utils

np.random.seed(42)  # For reproducibility
plt.style.use("seaborn-v0_8-whitegrid")

# %%
# 1. Generate synthetic data
# --------------------------
# We'll use our data generation function to create synthetic temperature and flow data

# Generate one year of daily data
mean, std = 8000.0, 400.0  # m3
retardation_factor = 2.0
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
    retardation_factor=retardation_factor,
)

# Discretize the aquifer pore volume distribution in bins
bins = gamma_utils.bins(mean=mean, std=std, n_bins=1000)

# %%
# 2. Forward: Compute and plot the residence time
# ---------------------------------------------------------------------
# Compute the residence time at the time of infiltration of the generated data for every bin of the aquifer pore volume distribion.
# Data returned is of shape (n_bins, n_days). First with retardation factor = 1.0, then with the
# retardation factor of the temperature in the aquifer (= 2).
rt_forward_rf1 = advection.residence_time(
    flow=df.flow,
    flow_tend=df.index,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=1.0,  # Note that we are computing the rt of the water, not the heat transport
    direction="infiltration",
)
rt_forward_rf2 = advection.residence_time(
    flow=df.flow,
    flow_tend=df.index,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=retardation_factor,
    direction="infiltration",
)

# The rt_forward_rf1 and rt_forward_rf2 arrays contain the residence time distribution at each timestamp of infiltration. This distribution varies over time.
# Here, we compute the mean residence time for each timestamp of infiltration, and certain quantiles to visualize the spread in residence time.
# Note that the residence time can not be computed at the outer ends.
quantiles = [1, 10, 90, 99]
quantile_headers = [f"rt_forward_rf1_{q}%" for q in quantiles]

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    df["rt_forward_rf1_mean"] = np.nanmean(rt_forward_rf1, axis=0)  # last values not defined
    df["rt_forward_rf2_mean"] = np.nanmean(rt_forward_rf2, axis=0)  # last values not defined

    df[quantile_headers] = np.nanpercentile(rt_forward_rf1, quantiles, axis=0).T  # last values not defined

# %%
# 3. Forward: Plot the results
# ----------------------------
# Forward: In how many days from now is the water extracted?
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(df.index, df.flow, label="Flow", color="C0")
ax[0].set_ylabel("Flow [m³/day]")
ax[0].legend(loc="upper left")

# Plot the residence time
ax[1].plot(df.index, df.rt_forward_rf1_mean, label="Mean (forward; retardation=1)")
ax[1].plot(df.index, df.rt_forward_rf2_mean, label=f"Mean (forward; retardation={retardation_factor:.1f})")

for q in quantiles:
    ax[1].plot(df.index, df[f"rt_forward_rf1_{q}%"], label=f"Quantile {q}% (forward; retardation=1)", ls="--", lw=0.8)

ax[1].set_title("Residence time in the aquifer")
ax[1].set_ylabel("Residence time [days]")
ax[1].legend(loc="upper left")
ax[1].set_xlabel("Date")

# Make a note about forward and backward residence time
ax[1].text(
    0.01,
    0.01,
    "Forward: In how many days from now is the water extracted?",
    ha="left",
    va="bottom",
    transform=ax[1].transAxes,
    fontsize=10,
)
plt.tight_layout()

# %%
# 4. Backward: Compute and plot the residence time
# ------------------------------------------------
# Compute the residence time at the time of extraction of the generated data for every bin of the aquifer pore volume distribion.
# Data returned is of shape (n_bins, n_days). First with retardation factor = 1.0, then with the
# retardation factor of the temperature in the aquifer (= 2).
rt_backward_rf1 = advection.residence_time(
    flow=df.flow,
    flow_tend=df.index,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=1.0,
    direction="extraction",
)
rt_backward_rf2 = advection.residence_time(
    flow=df.flow,
    flow_tend=df.index,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=retardation_factor,
    direction="extraction",
)
# The rt_backward_rf1 and rt_backward_rf2 arrays contain the residence time distribution at each timestamp of extraction. This distribution varies over time.
# Here, we compute the mean residence time for each timestamp of extraction, and certain quantiles to visualize the spread in residence time.
quantiles = [1, 10, 90, 99]
quantile_headers = [f"rt_backward_rf1_{q}%" for q in quantiles]
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    df["rt_backward_rf1_mean"] = np.nanmean(rt_backward_rf1, axis=0)  # last values not defined
    df["rt_backward_rf2_mean"] = np.nanmean(rt_backward_rf2, axis=0)  # last values not defined

    df[quantile_headers] = np.nanpercentile(rt_backward_rf1, quantiles, axis=0).T  # last values not defined

# %%
# 5. Backward: Plot the results
# ----------------------------
# Backward: How many days ago was the water infiltrated?
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(df.index, df.flow, label="Flow", color="C0")
ax[0].set_ylabel("Flow [m³/day]")
ax[0].legend(loc="upper left")
# Plot the residence time
ax[1].plot(df.index, df.rt_backward_rf1_mean, label="Mean (backward; retardation=1)")
ax[1].plot(df.index, df.rt_backward_rf2_mean, label=f"Mean (backward; retardation={retardation_factor:.1f})")
for q in quantiles:
    ax[1].plot(df.index, df[f"rt_backward_rf1_{q}%"], label=f"Quantile {q}% (backward; retardation=1)", ls="--", lw=0.8)
ax[1].set_title("Residence time in the aquifer")
ax[1].set_ylabel("Residence time [days]")
ax[1].legend(loc="upper left")
ax[1].set_xlabel("Date")
# Make a note about forward and backward residence time
ax[1].text(
    0.01,
    0.01,
    "Backward: How many days ago was the water infiltrated?",
    ha="left",
    va="bottom",
    transform=ax[1].transAxes,
    fontsize=10,
)
plt.tight_layout()
plt.show()
