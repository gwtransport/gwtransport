"""
Example 1: Estimation of aquifer pore volume using temperature response.

This notebook demonstrates how to use temperature data to estimate the aquifer pore volume distribution.
The aquifer pore volume distribution directly affects the distribution of residence time of
the extracted water. We seek a distribution of aquifer pore volume that retards the temperature of the
infiltrating water to match the temperature of the extracted water.

An estimate of the aquifer pore volume distribution is the starting point for:
- Forecasting the concentration/temperature of the extracted water.
- Estimating the residence time of the infiltrating/extracted water (Example 2).
- Log-removal of pathogens based on residence time.

Key concepts:
- Temperature of the infiltrating water, temperature of the extracted water and the flow rate are known.
- Fitting gamma distribution parameters of the aquifer pore volume.
- The distribution is discretized into bins that convey an equal fraction of the flow, similar to streamlines.
- Per bin, at every time of extraction, the time of infiltration is computed. See Example 2 for more details on residence time.
- The temperature of the extracted water is the average of all bins of the temperature at the time of infiltration.

Assumptions:
- The aquifer pore volume distribution is constant over time.
- Advection is the only transport process.
"""

import matplotlib.pyplot as plt
import numpy as np
from example_data_generation import generate_synthetic_data
from scipy.optimize import curve_fit
from scipy.stats import gamma as gamma_dist

from gwtransport import advection
from gwtransport import gamma as gamma_utils

np.random.seed(42)  # For reproducibility
plt.style.use("seaborn-v0_8-whitegrid")

# %%
# 1. Generate synthetic data
# --------------------------
# We'll use our data generation function to create synthetic temperature and flow data

# Generate one year of daily data
df = generate_synthetic_data(
    start_date="2020-01-01",
    end_date="2025-12-31",
    mean_flow=120.0,  # m3/day
    flow_amplitude=40.0,  # m3/day
    flow_noise=5.0,  # m3/day
    mean_temp_infiltration=12.0,  # °C
    temp_infiltration_amplitude=8.0,  # °C
    aquifer_pore_volume=8000.0,  # m3
    aquifer_pore_volume_std=400.0,  # m3
    retardation_factor=2.0,
)

print("Data summary:")
print(f"- Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"- Mean flow: {df['flow'].mean():.1f} m³/day")
print(f"- Mean infiltration temperature: {df['temp_infiltration'].mean():.1f} °C")
print(f"- Mean extraction temperature: {df['temp_extraction'].mean():.1f} °C")
print(f"- True mean of aquifer pore volume distribution: {df.attrs['aquifer_pore_volume_mean']:.1f} m³")
print(f"- True standard deviation of aquifer pore volume distribution: {df.attrs['aquifer_pore_volume_std']:.1f} m³")


# %%
# 3. Curve fitting to estimate aquifer pore volume distribution parameters
# ------------------------------------------------------------------------
# Perform the curve fitting on valid data. It takes some time to for large fractions
# of the infiltration temperature be present in the extracted water. For simplicity sake,
# we will use the first year as spin up time and only fit the data from 2021 onwards.
def objective(time, mean, std):  # noqa: ARG001, D103
    cout = advection.gamma_forward(
        cin=df.temp_infiltration,
        cin_tend=df.index,
        cout_tend=df.index,
        flow=df.flow,
        flow_tend=df.index,
        mean=mean,
        std=std,
        n_bins=200,
        retardation_factor=2.0,
    )
    return cout["2021-01-01":].values


(mean, std), pcov = curve_fit(
    objective,
    df.index,
    df["2021-01-01":].temp_extraction,
    p0=(7000.0, 500.0),
    bounds=([1000, 10], [10000, 1000]),  # Reasonable bounds for mean and std
    method="trf",  # Trust Region Reflective algorithm
    max_nfev=100,  # Limit number of function evaluations to keep runtime reasonable
)
df["temp_extraction_modeled"] = advection.gamma_forward(
    cin=df.temp_infiltration,
    cin_tend=df.index,
    cout_tend=df.index,
    flow=df.flow,
    flow_tend=df.index,
    mean=mean,
    std=std,
    n_bins=100,
    retardation_factor=2.0,
)

# Print the fitted parameters
print("\nFitted parameters:")
print(f"- Fitted mean of aquifer pore volume distribution: {mean:.1f} +/- {pcov[0, 0] ** 0.5:.1f} m³")
print(f"- Fitted standard deviation of aquifer pore volume distribution: {std:.1f} +/- {pcov[1, 1] ** 0.5:.1f} m³")

# %%
# 4. Plot the measured and modeled temperatures
# -------------------
fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), nrows=2, ncols=1, sharex=True)

ax1.set_title("Estimation of aquifer pore volume using temperature response")
ax1.plot(df.index, df.flow, label="Flow rate", color="C0", alpha=0.8, linewidth=0.8)
ax1.set_ylabel("Flow rate (m³/day)")
ax1.legend()

ax2.plot(df.index, df.temp_infiltration, label="Infiltration water: measured", color="C0", alpha=0.8, linewidth=0.8)
ax2.plot(df.index, df.temp_extraction, label="Extracted water: measured", color="C1", alpha=0.8, linewidth=0.8)
ax2.plot(df.index, df.temp_extraction_modeled, label="Extracted water: modeled", color="C2", alpha=0.8, linewidth=0.8)
ax2.set_xlabel("Date")
ax2.set_ylabel("Temperature (°C)")
ax2.legend()

plt.tight_layout()

# %%
# 5. Plot the fitted distribution of the aquifer pore volume
# -------------------
# The number of bins is reduced here for demonstration (dramatic) purposes.
#
# Note that the gamma distribution is parameterized here in two ways:
# - Shape and scale (alpha, beta)
# - Mean and standard deviation (mean, std)
# The two parameterizations are related by the following equations:
# - mean = alpha * beta
# - std = beta * sqrt(alpha)
n_bins = 10
alpha, beta = gamma_utils.mean_std_to_alpha_beta(mean, std)
gbins = gamma_utils.bins(alpha=alpha, beta=beta, n_bins=n_bins)

print(f"Gamma distribution (alpha={alpha:.1f}, beta={beta:.1f}) divided into {n_bins} equal-volume bins:")
print("-" * 80)
print(f"{'Bin':3s} {'Lower':10s} {'Upper':10s} {'E[X|bin]':10s} {'P(bin)':10s}")
print("-" * 80)

for i in range(n_bins):
    upper = f"{gbins['upper_bound'][i]:.3f}" if not np.isinf(gbins["upper_bound"][i]) else "∞"
    lower = f"{gbins['lower_bound'][i]:.3f}"
    expected = f"{gbins['expected_value'][i]:.3f}"
    prob = f"{gbins['probability_mass'][i]:.3f}"
    print(f"{i:3d} {lower:10s} {upper:10s} {expected:10s} {prob:10s}")

# Verify total probability is exactly 1
print(f"\nTotal probability mass: {gbins['probability_mass'].sum():.6f}")

# Verify expected value is close to the mean of the distribution
mean = alpha * beta
expected_value = np.sum(gbins["expected_value"] * gbins["probability_mass"])
print(f"Mean of distribution: {mean:.3f}")
print(f"Expected value of bins: {expected_value:.3f}")

mass_per_bin = gamma_utils.bin_masses(alpha, beta, gbins["edges"])
print(f"Total probability mass: {mass_per_bin.sum():.6f}")
print("Probability mass per bin:")
print(mass_per_bin)

# plot the gamma distribution and the bins
x = np.linspace(0, 1.1 * gbins["expected_value"][-1], 1000)
y = gamma_dist.pdf(x, alpha, scale=beta)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(
    f"Gamma distribution (alpha={alpha:.1f}, beta={beta:.1f}, mean={mean:.1f}, std={std:.1f}) divided into {n_bins} equal-volume bins:"
)
ax.plot(x, y, label="Gamma PDF", color="C0", alpha=0.8, linewidth=0.8)
pdf_at_lower_bound = gamma_dist.pdf(gbins["lower_bound"], alpha, scale=beta)
ax.vlines(gbins["lower_bound"], 0, pdf_at_lower_bound, color="C1", alpha=0.8, linewidth=0.8)
ax.set_xlabel("Aquifer pore volume (m³)")
ax.set_ylabel("Probability density (-)")
ax.legend()
plt.show()
