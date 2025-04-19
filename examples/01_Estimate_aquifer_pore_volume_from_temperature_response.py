"""
Example 1: Estimation of aquifer pore volume using temperature response.

This notebook demonstrates how to use temperature data to estimate the aquifer
pore volume by fitting a gamma distribution to the temperature response of the aquifer.

Key concepts:
- Using temperature time series to infer aquifer properties
- Fitting gamma distribution parameters (alpha/beta or mean/std)
- Visualizing the fitted distributions and transport results
"""

import matplotlib.pyplot as plt
import numpy as np
from example_data_generation import generate_synthetic_data
from scipy.optimize import curve_fit

from gwtransport1d import advection

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
    random_seed=42,
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
        cin=df.temp_infiltration, flow=df.flow, mean=mean, std=std, n_bins=100, retardation_factor=2.0
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
    cin=df.temp_infiltration, flow=df.flow, mean=mean, std=std, n_bins=100, retardation_factor=2.0
)

# Print the fitted parameters
print("\nFitted parameters:")
print(f"- Fitted mean of aquifer pore volume distribution: {mean:.1f} +/- {pcov[0, 0] ** 0.5:.1f} m³")
print(f"- Fitted standard deviation of aquifer pore volume distribution: {std:.1f} +/- {pcov[1, 1] ** 0.5:.1f} m³")

# %%
# 4. Plot the results
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
plt.show()
