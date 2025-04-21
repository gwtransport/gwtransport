"""Example 2: Estimation of the residence time distribution using synthetic data."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from example_data_generation import generate_synthetic_data
from scipy.stats import gamma as gamma_dist

from gwtransport1d import advection
from gwtransport1d import gamma as gamma_utils

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
    random_seed=42,
)

# %%
# 2. Compute and plot the mean residence time of the generated data
# -----------------------------------------------------------------
# Compute the residence time of the generated data

# Discretize the aquifer pore volume distribution in bins
alpha, beta = gamma_utils.mean_std_to_alpha_beta(mean, std)
bins = gamma_utils.bins(alpha, beta, n_bins=1000)

# Compute the residence time for every bin. Data returned is of shape (n_bins, n_days)
rt_forward = advection.residence_time(
    df.flow,
    bins["expected_value"],
    retardation_factor=1.0,  # Note that we are computing the rt of the water, not the heat transport
    direction="infiltration",
)
rt_backward = advection.residence_time(
    df.flow,
    bins["expected_value"],
    retardation_factor=1.0,
    direction="extraction",
)

# Note that the residence time can not be computed at the outer ends.
with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", message="Mean of empty slice")
    warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
    df["rt_forward_mean"] = np.nanmean(rt_forward, axis=0)  # last values not defined
    df["rt_backward_mean"] = np.nanmean(rt_backward, axis=0)  # first values not defined

    df["rt_forward_fastest_5perc"] = np.nanpercentile(rt_forward, 5, axis=0)  # last values not defined
    df["rt_backward_fastest_5perc"] = np.nanpercentile(rt_backward, 5, axis=0)  # first values not defined

# %%
# 3. Plot the results
# -------------------

fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax[0].plot(df.index, df.flow, label="Flow", color="C0")
ax[0].set_ylabel("Flow [m³/day]")
ax[0].legend(loc="upper left")

# Plot the residence time
ax[1].plot(df.index, df.rt_forward_mean, label="Mean residence time (forward)", color="C1")
ax[1].plot(df.index, df.rt_backward_mean, label="Mean residence time (backward)", color="C2")
ax[1].plot(df.index, df.rt_forward_fastest_5perc, label="Fastest 5% residence time (forward)", color="C1", ls="--")
ax[1].plot(df.index, df.rt_backward_fastest_5perc, label="Fastest 5% residence time (backward)", color="C2", ls="--")
ax[1].set_title("Residence time of the aquifer")
ax[1].set_ylabel("Residence time [days]")
ax[1].legend(loc="upper left")
ax[1].set_xlabel("Date")

# Make a note about forward and backward residence time
ax[1].text(
    0.01,
    0.01,
    "Forward: After how many days is the water extracted?\nBackward: How many days ago was the water infiltrated?",
    ha="left",
    va="bottom",
    transform=ax[1].transAxes,
    fontsize=10,
)
plt.tight_layout()

# %%
# 4. Residence time as a function of flow
# ---------------------------------------
# If the flow is constant we can compute the residence time distribution directly from the aquifer pore volume distribution.

# Flows for which to compute the residence time
flows = np.linspace(0.1, mean_flow, 100)  # m3/day

alpha, beta = gamma_utils.mean_std_to_alpha_beta(mean, std)

quantiles = np.array([0.1, 0.5, 0.9])
scaled_beta = beta / flows
bin_edges = gamma_dist.ppf(quantiles[:, None], alpha, scale=scaled_beta[None, :])

# Plot on log-log scale


fig, ax = plt.subplots(figsize=(12, 6))
for i, q in enumerate(quantiles):
    ax.loglog(flows, bin_edges[i], label=f"Quantile {q:.0%}")
ax.set_xlabel("Flow [m³/day]")
ax.set_ylabel("Residence time [days]")
ax.set_title("Residence time as a function of flow")
ax.legend()
plt.tight_layout()
