"""
Example 1: Aquifer pore volume estimation using temperature response.

This example demonstrates inverse modeling to estimate aquifer pore volume distribution
from temperature breakthrough curves. Temperature acts as a conservative tracer with
known thermal retardation, allowing characterization of flow paths and residence times.

Applications:
- Groundwater vulnerability assessment
- Residence time distribution analysis (Example 2)
- Contaminant transport forecasting

Methodology:
- Input: infiltration temperature, extraction temperature, flow rates
- Model: gamma-distributed pore volumes with thermal retardation
- Optimization: curve fitting to match observed temperature breakthrough
- Output: statistical parameters of pore volume distribution

Key assumptions:
- Stationary pore volume distribution
- Advection-dominated transport (Pe >> 1)
- Thermal retardation factor = 2.0 (typical for saturated media)
"""

from pathlib import Path

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
# Create synthetic temperature and flow time series to demonstrate the inverse modeling approach

# Generate 6 years of daily data with seasonal patterns
df, tedges = generate_synthetic_data(
    start_date="2020-01-01",
    end_date="2025-12-31",
    mean_flow=120.0,  # Base flow rate [m³/day]
    flow_amplitude=40.0,  # Seasonal flow variation [m³/day]
    flow_noise=5.0,  # Random daily fluctuations [m³/day]
    mean_temp_infiltration=12.0,  # Annual mean temperature [°C]
    temp_infiltration_amplitude=8.0,  # Seasonal temperature range [°C]
    aquifer_pore_volume=8000.0,  # True mean pore volume [m³]
    aquifer_pore_volume_std=400.0,  # True standard deviation [m³]
    retardation_factor=2.0,  # Thermal retardation factor [-]
)

print("Data summary:")
print(f"- Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"- Mean flow: {df['flow'].mean():.1f} m³/day")
print(f"- Mean infiltration temperature: {df['temp_infiltration'].mean():.1f} °C")
print(f"- Mean extraction temperature: {df['temp_extraction'].mean():.1f} °C")
print(f"- True mean of aquifer pore volume distribution: {df.attrs['aquifer_pore_volume_mean']:.1f} m³")
print(f"- True standard deviation of aquifer pore volume distribution: {df.attrs['aquifer_pore_volume_std']:.1f} m³")


# %%
# 2. Parameter estimation via optimization
# ----------------------------------------
# Implement inverse modeling to estimate gamma distribution parameters.
# Use spin-up period to allow thermal breakthrough to stabilize.

# Time bin edges are already computed and returned from generate_synthetic_data

# Define training dataset (exclude spin-up period)
train_data = df["2021-01-01":].temp_extraction  # Skip first year for thermal equilibration
train_data = train_data.dropna()  # Remove NaN values that may occur at series boundaries
train_length = len(train_data)


def objective(_xdata, mean, std):
    """Forward model for temperature breakthrough with gamma-distributed pore volumes."""
    print(f"Optimizing with parameters: mean={mean:.1f} m³, std={std:.1f} m³")
    cout = advection.gamma_forward(
        cin=df.temp_infiltration,
        flow=df.flow,
        tedges=tedges,
        cout_tedges=tedges,
        mean=mean,  # Mean pore volume [m³]
        std=std,  # Standard deviation [m³]
        n_bins=25,  # Discretization resolution
        retardation_factor=2.0,  # Thermal retardation factor
    )
    # Match the training data indices and remove NaN values
    return cout[df.index >= "2021-01-01"]


# Nonlinear least squares optimization
(mean, std), pcov = curve_fit(
    objective,
    df.index,
    train_data.values,  # Use the cleaned training data values
    p0=(7500.0, 450.0),  # Initial parameter estimates [m³]
    bounds=([5000, 200], [10000, 600]),  # Physical constraints [m³]
    method="trf",  # Trust Region Reflective algorithm
    max_nfev=250,  # Limit function evaluations for computational efficiency
)

# Generate model predictions using optimized parameters
df["temp_extraction_modeled"] = advection.gamma_forward(
    cin=df.temp_infiltration,
    flow=df.flow,
    tedges=tedges,
    cout_tedges=tedges,
    mean=mean,  # Fitted mean pore volume
    std=std,  # Fitted standard deviation
    n_bins=250,  # Computational resolution
    retardation_factor=2.0,  # Thermal retardation
)

# Report optimization results with uncertainty estimates
print("\nParameter estimation results:")
print(f"- Mean pore volume: {mean:.1f} ± {pcov[0, 0] ** 0.5:.1f} m³")
print(f"- Standard deviation: {std:.1f} ± {pcov[1, 1] ** 0.5:.1f} m³")
print(f"- Coefficient of variation: {std / mean:.2f}")

# %%
# 3. Model validation and visualization
# ------------------------------------
fig, (ax1, ax2) = plt.subplots(figsize=(10, 6), nrows=2, ncols=1, sharex=True)

ax1.set_title("Temperature-based aquifer characterization")
ax1.plot(df.index, df.flow, label="Discharge rate", color="C0", alpha=0.8, linewidth=0.8)
ax1.set_ylabel("Discharge [m³/day]")
ax1.legend()

ax2.plot(df.index, df.temp_infiltration, label="Recharge temperature", color="C0", alpha=0.8, linewidth=0.8)
ax2.plot(df.index, df.temp_extraction, label="Discharge temperature (observed)", color="C1", alpha=0.8, linewidth=0.8)
ax2.plot(
    df.index, df.temp_extraction_modeled, label="Discharge temperature (modeled)", color="C2", alpha=0.8, linewidth=0.8
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Temperature [°C]")
ax2.legend()

fig.tight_layout()

# Save the temperature response plot
out_path = Path(__file__).parent / "01_Temperature_response.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

# %%
# 4. Pore volume distribution analysis
# -----------------------------------
# Visualize the fitted gamma distribution representing spatial heterogeneity
# in pore volume. Each bin represents a different flow path through the aquifer.

# Discretize gamma distribution into flow path bins
n_bins = 10  # Reduced for visualization clarity
alpha, beta = gamma_utils.mean_std_to_alpha_beta(mean, std)  # Convert parameterization
gbins = gamma_utils.bins(alpha=alpha, beta=beta, n_bins=n_bins)  # Equal-probability bins

print(f"Gamma distribution (α={alpha:.1f}, β={beta:.1f}) discretized into {n_bins} equiprobable bins:")
print("-" * 80)
print(f"{'Bin':3s} {'Lower [m³]':10s} {'Upper [m³]':10s} {'E[V|bin]':10s} {'P(bin)':10s}")
print("-" * 80)

for i in range(n_bins):
    upper = f"{gbins['upper_bound'][i]:.3f}" if not np.isinf(gbins["upper_bound"][i]) else "∞"
    lower = f"{gbins['lower_bound'][i]:.3f}"
    expected = f"{gbins['expected_value'][i]:.3f}"
    prob = f"{gbins['probability_mass'][i]:.3f}"
    print(f"{i:3d} {lower:10s} {upper:10s} {expected:10s} {prob:10s}")

# Verify discretization accuracy
print("\nDiscretization verification:")
print(f"Total probability mass: {gbins['probability_mass'].sum():.6f}")
mean_analytical = alpha * beta
mean_numerical = np.sum(gbins["expected_value"] * gbins["probability_mass"])
print(f"Analytical mean: {mean_analytical:.3f} m³")
print(f"Numerical mean: {mean_numerical:.3f} m³")
print(f"Relative error: {abs(mean_analytical - mean_numerical) / mean_analytical * 100:.2f}%")

mass_per_bin = gamma_utils.bin_masses(alpha, beta, gbins["edges"])
print(f"Total probability mass: {mass_per_bin.sum():.6f}")
print("Probability mass per bin:")
print(mass_per_bin)

# plot the gamma distribution and the bins
x = np.linspace(0, 1.1 * gbins["expected_value"][-1], 1000)
y = gamma_dist.pdf(x, alpha, scale=beta)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"Fitted pore volume distribution (α={alpha:.1f}, β={beta:.1f}, μ={mean:.1f} m³, σ={std:.1f} m³)")
ax.plot(x, y, label="Probability density function", color="C0", alpha=0.8, linewidth=2)
pdf_at_lower_bound = gamma_dist.pdf(gbins["lower_bound"], alpha, scale=beta)
ax.vlines(gbins["lower_bound"], 0, pdf_at_lower_bound, color="C1", alpha=0.6, linewidth=1, label="Bin boundaries")
ax.set_xlabel("Pore volume [m³]")
ax.set_ylabel("Probability density [m⁻³]")
ax.legend()
fig.tight_layout()

# Save the pore volume distribution plot
out_path = Path(__file__).parent / "01_Pore_volume_distribution.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
