"""
Example 2: Residence time distribution analysis.

This example demonstrates how to calculate residence time distributions from
aquifer pore volume distribution (from Example 1) and flow rates. Residence time
quantifies how long water spends in the aquifer, which is crucial for:
- Contaminant transport predictions
- Groundwater vulnerability assessment
- Treatment efficiency evaluation

Methodology:
- Input: pore volume distribution, flow time series
- Calculate travel times for different flow paths
- Analyze temporal variations in residence time

Two perspectives:
- Forward: How long until infiltrating water is extracted?
- Backward: How long ago was extracted water infiltrated?

Assumptions:
- Stationary pore volume distribution
- Advection-dominated transport
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from example_data_generation import generate_synthetic_data

from gwtransport import gamma as gamma_utils
from gwtransport.residence_time import residence_time

np.random.seed(42)  # For reproducibility
plt.style.use("seaborn-v0_8-whitegrid")

# %%
# 1. Generate synthetic data
# --------------------------
# Create flow and aquifer parameter data for residence time analysis

# Define aquifer parameters (typically from Example 1 optimization)
mean, std = 8000.0, 400.0  # Pore volume statistics [m³]
retardation_factor = 2.0  # For conservative tracer analysis
mean_flow = 120.0  # Base discharge rate [m³/day]

df, tedges = generate_synthetic_data(
    start_date="2020-01-01",
    end_date="2025-12-31",
    mean_flow=mean_flow,  # Base discharge [m³/day]
    flow_amplitude=40.0,  # Seasonal variation [m³/day]
    flow_noise=5.0,  # Daily fluctuations [m³/day]
    mean_temp_infiltration=12.0,  # Mean temperature [°C]
    temp_infiltration_amplitude=8.0,  # Seasonal range [°C]
    aquifer_pore_volume=mean,  # Mean pore volume [m³]
    aquifer_pore_volume_std=std,  # Pore volume variability [m³]
    retardation_factor=retardation_factor,  # Thermal retardation [-]
)

# Discretize pore volume distribution for residence time calculation
bins = gamma_utils.bins(mean=mean, std=std, n_bins=1000)  # High resolution for accuracy

# %%
# 2. Forward residence time analysis
# ---------------------------------
# Calculate how long infiltrating water takes to be extracted.
# Compute for all flow paths (bins) and compare water vs thermal transport.

# Use time bin edges returned from generate_synthetic_data
flow_tedges = tedges

# Water residence time (no retardation)
rt_forward_rf1 = residence_time(
    flow=df.flow,
    flow_tedges=flow_tedges,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=1.0,  # Conservative tracer (water flow)
    direction="infiltration",
)

# Thermal residence time (with retardation)
rt_forward_rf2 = residence_time(
    flow=df.flow,
    flow_tedges=flow_tedges,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=retardation_factor,  # Heat transport (slower)
    direction="infiltration",
)

# Statistical analysis of residence time distributions
# Arrays contain residence times for all flow paths at each time step
# Calculate mean and quantiles to characterize temporal variability

quantiles = [1, 10, 90, 99]  # Percentiles for uncertainty bounds
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

# Save the forward residence time plot
out_path = Path(__file__).parent / "02_Forward_residence_time.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

# %%
# 4. Backward: Compute and plot the residence time
# ------------------------------------------------
# Compute the residence time at the time of extraction of the generated data for every bin of the aquifer pore volume distribion.
# Data returned is of shape (n_bins, n_days). First with retardation factor = 1.0, then with the
# retardation factor of the temperature in the aquifer (= 2).
rt_backward_rf1 = residence_time(
    flow=df.flow,
    flow_tedges=flow_tedges,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=1.0,
    direction="extraction",
)
rt_backward_rf2 = residence_time(
    flow=df.flow,
    flow_tedges=flow_tedges,
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

# Save the backward residence time plot
out_path = Path(__file__).parent / "02_Backward_residence_time.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
