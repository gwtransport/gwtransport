"""
Example 3: Pathogen Removal in Bank Filtration Systems.

This example shows how to calculate pathogen removal efficiency in groundwater
treatment systems using log-removal analysis. Understanding pathogen removal is
crucial for safe drinking water production from riverbank filtration.

What you'll learn:
🔬 How pathogens are removed as water flows through aquifer sediments
💧 Why residence time (how long water stays underground) matters
📊 How to calculate removal efficiency for different system designs
⚖️ Why averaging parallel systems requires special care

Real-world context:
Bank filtration is widely used in Europe for drinking water treatment. River water
infiltrates through riverbank sediments, where pathogens are naturally filtered out
through physical straining and biological decay. The longer water stays underground,
the more pathogens are removed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from example_data_generation import generate_synthetic_data

from gwtransport import gamma as gamma_utils
from gwtransport.logremoval import (
    gamma_find_flow_for_target_mean,
    gamma_mean,
    parallel_mean,
    residence_time_to_log_removal,
)
from gwtransport.residence_time import residence_time

np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")

# %%
# Section 1: Understanding Basic Log-Removal
# ==========================================
#
# The "log-removal" concept:
# - Measures pathogen removal on a logarithmic scale
# - 1 log10 = 90% removal (1 in 10 pathogens remain)
# - 2 log10 = 99% removal (1 in 100 pathogens remain)
# - 3 log10 = 99.9% removal (1 in 1000 pathogens remain)
#
# The key relationship: Log-removal = k * log10(residence_time)
# Where k is the removal rate constant (depends on pathogen type and aquifer)

print("=== Section 1: Basic Log-Removal Calculation ===")
print("Simulating a small bank filtration system...")
print()

# Define aquifer characteristics (imagine a riverbank aquifer)
mean_pore_volume = 1000.0  # m³ (total water-filled space in aquifer)
std_pore_volume = 300.0  # m³ (variability in pore volume)
flow_rate = 50.0  # m³/day (water extraction rate)
log_removal_rate = 3.5  # log10 removal per log10(day) (typical for bacteria)

# Step 1: Convert aquifer properties to mathematical distribution
# (The gamma distribution describes the variability in flow paths)
alpha, beta = gamma_utils.mean_std_to_alpha_beta(mean_pore_volume, std_pore_volume)

# Step 2: Calculate residence time statistics
# Residence time = pore volume / flow rate
rt_alpha = alpha
rt_beta = beta / flow_rate

# Step 3: Calculate pathogen log-removal
mean_log_removal = gamma_mean(rt_alpha, rt_beta, log_removal_rate)
removal_efficiency = (1 - 10 ** (-mean_log_removal)) * 100

# Step 4: Display results in understandable terms
mean_residence_time = mean_pore_volume / flow_rate
print("Aquifer setup:")
print(f"  • Pore volume: {mean_pore_volume:.0f} ± {std_pore_volume:.0f} m³")
print(f"  • Flow rate: {flow_rate} m³/day")
print(f"  • Mean residence time: {mean_residence_time:.1f} days")
print()
print("Pathogen removal results:")
print(f"  • Log-removal: {mean_log_removal:.2f} log10")
print(f"  • Removal efficiency: {removal_efficiency:.1f}%")
print(f"  • This means {100 - removal_efficiency:.1f}% of pathogens remain")

# %%
# Section 2: Why Parallel Systems Need Special Treatment
# =====================================================
#
# Imagine three wells extracting from the same aquifer, each with different
# flow paths and residence times. How do we calculate the overall removal?
#
# KEY INSIGHT: You CANNOT simply average log-removal values!
# Log-removal represents exponential processes, so averaging must account for
# the fact that shorter residence times dominate the overall performance.

print("\n=== Section 2: Parallel Systems - A Common Mistake ===")
print("Scenario: Three extraction wells with different residence times")
print()

# Three parallel extraction points with different log-removal efficiencies
unit_removals = np.array([0.5, 1.0, 1.5])  # log₁₀ values for each well

# CORRECT method: Use parallel_mean() which accounts for exponential nature
combined_removal = parallel_mean(unit_removals)

# WRONG method: Simple arithmetic average (commonly done incorrectly!)
simple_average = np.mean(unit_removals)

print("Individual extraction wells:")
for i, removal in enumerate(unit_removals):
    efficiency = (1 - 10 ** (-removal)) * 100
    print(f"  Well {i + 1}: {removal:.1f} log10 → {efficiency:.1f}% removal")

print()
combined_efficiency = (1 - 10 ** (-combined_removal)) * 100
simple_efficiency = (1 - 10 ** (-simple_average)) * 100

print("Combined system performance:")
print(f"  ✅ Correct method: {combined_removal:.2f} log10 → {combined_efficiency:.1f}% removal")
print(f"  ❌ Wrong method:   {simple_average:.2f} log10 → {simple_efficiency:.1f}% removal")
print()
print("📝 Note: The correct method gives LOWER removal because the worst-performing")
print("   well (shortest residence time) dominates the overall system performance.")

# %%
# Section 3: Design Application - Meeting Safety Standards
# =======================================================
#
# Water treatment facilities must meet strict pathogen removal standards.
# For example, the WHO recommends at least 2 log10 (99%) removal for bacteria.
#
# Question: What flow rate should we use to achieve this target?

print("\n=== Section 3: Design Application ===")
print("Design challenge: Meet WHO standards for safe drinking water")
print()

# WHO recommendation: minimum 2 log10 (99%) removal for bacteria
target_removal = 2.0
target_efficiency = (1 - 10 ** (-target_removal)) * 100

print(f"Target: {target_removal} log10 removal ({target_efficiency:.0f}% efficiency)")

# Find the maximum flow rate that still achieves our target
required_flow = gamma_find_flow_for_target_mean(
    target_mean=target_removal, apv_alpha=alpha, apv_beta=beta, log_removal_rate=log_removal_rate
)

required_residence_time = mean_pore_volume / required_flow

print()
print("Design solution:")
print(f"  • Maximum flow rate: {required_flow:.1f} m³/day")
print(f"  • Required residence time: {required_residence_time:.1f} days")
print(f"  • Daily water production: {required_flow:.1f} m³ = {required_flow * 1000:.0f} liters")
print()
print("💡 Engineering insight: Higher flow rates mean shorter residence times,")
print("   which reduces pathogen removal. There's always a trade-off between")
print("   water production capacity and treatment effectiveness.")

# %%
# Section 4: Visualization - Seeing the Difference
# ===============================================

print("\n=== Section 4: Visualization ===")
print("Creating comparison chart to show why averaging method matters...")
print()

# Create a clear, educational visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Individual well performance
well_names = ["Well 1\n(Short path)", "Well 2\n(Medium path)", "Well 3\n(Long path)"]
well_removals = unit_removals
well_efficiencies = [(1 - 10 ** (-r)) * 100 for r in well_removals]

bars1 = ax1.bar(well_names, well_efficiencies, color=["lightcoral", "lightblue", "lightgreen"], alpha=0.8)
ax1.set_ylabel("Removal Efficiency (%)")
ax1.set_title("Individual Well Performance")
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3, axis="y")

# Add percentage labels
for bar, efficiency in zip(bars1, well_efficiencies, strict=False):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{efficiency:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# Right plot: Comparison of averaging methods
methods = ["Correct Method\n(Parallel)", "Wrong Method\n(Simple Average)"]
combined_efficiencies = [(1 - 10 ** (-combined_removal)) * 100, (1 - 10 ** (-simple_average)) * 100]
colors = ["gold", "red"]

bars2 = ax2.bar(methods, combined_efficiencies, color=colors, alpha=0.8)
ax2.set_ylabel("Combined Removal Efficiency (%)")
ax2.set_title("Combined System Performance")
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3, axis="y")

# Add percentage and log-removal labels
labels = [
    f"{eff:.1f}%\n({log_val:.2f} log10)"
    for eff, log_val in zip(combined_efficiencies, [combined_removal, simple_average], strict=False)
]
for bar, label in zip(bars2, labels, strict=False):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 1, label, ha="center", va="bottom", fontweight="bold")

plt.tight_layout()

# Save the plot
out_path = Path(__file__).parent / "03_log_removal_analysis.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"📊 Visualization saved to: {out_path}")
print("   This chart clearly shows why the correct averaging method is crucial!")

# %%
# Section 5: Real-World Scenario - Seasonal Variations
# ===================================================
#
# In reality, river flows change seasonally, affecting bank filtration performance.
# Let's simulate a multi-year bank filtration system to see how log-removal
# varies with changing river levels and extraction rates.

print("\n=== Section 5: Real-World Time Series Analysis ===")
print("Simulating 6 years of bank filtration with seasonal flow variations...")
print()

# Generate realistic flow data with seasonal patterns
df, tedges = generate_synthetic_data(
    start_date="2020-01-01",
    end_date="2025-12-31",
    mean_flow=120.0,  # Base flow rate [m³/day]
    flow_amplitude=40.0,  # Seasonal flow variation [m³/day]
    flow_noise=5.0,  # Random daily fluctuations [m³/day]
    mean_temp_infiltration=12.0,  # Annual mean temperature [°C]
    temp_infiltration_amplitude=8.0,  # Seasonal temperature range [°C]
    aquifer_pore_volume=8000.0,  # Larger aquifer for this scenario [m³]
    aquifer_pore_volume_std=400.0,  # Standard deviation [m³]
    retardation_factor=2.0,  # Thermal retardation factor [-]
)

# Set up aquifer characteristics for this larger system
mean_pv, std_pv = 8000.0, 400.0  # Pore volume statistics [m³]
bins = gamma_utils.bins(mean=mean_pv, std=std_pv, n_bins=1000)  # High resolution

# Calculate residence times and log-removal for the time series
print("Computing pathogen removal over time...")

# Calculate residence time distribution for water flow
rt_forward_rf1 = residence_time(
    flow=df.flow,
    flow_tedges=tedges,
    aquifer_pore_volume=bins["expected_value"],
    retardation_factor=1.0,  # Water (conservative tracer)
    direction="infiltration",
)

# Compute log-removal for each flow path and time point
log_removal_array = residence_time_to_log_removal(residence_times=rt_forward_rf1, log_removal_rate=log_removal_rate)

# Average across all flow paths (using correct parallel averaging)
df["log_removal"] = parallel_mean(log_removal_array, axis=0)
df["removal_efficiency"] = (1 - 10 ** (-df["log_removal"])) * 100

# Create an informative time series plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Flow rate over time
ax1.plot(df.index, df.flow, color="steelblue", linewidth=1.5)
ax1.set_ylabel("Flow Rate (m³/day)")
ax1.set_title("Bank Filtration System Performance Over Time")
ax1.grid(True, alpha=0.3)

# Plot 2: Log-removal over time
ax2.plot(df.index, df.log_removal, color="forestgreen", linewidth=1.5)
ax2.set_ylabel("Log-removal (log10)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the time series plot
out_path = Path(__file__).parent / "03_log_removal_time_series.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"📈 Time series plot saved to: {out_path}")

# Calculate and display summary statistics
min_removal = df["log_removal"].min()
max_removal = df["log_removal"].max()
mean_removal = df["log_removal"].mean()
min_efficiency = df["removal_efficiency"].min()
max_efficiency = df["removal_efficiency"].max()

print()
print("📊 Performance summary over 6 years:")
print(f"  • Log-removal range: {min_removal:.2f} - {max_removal:.2f} log10")
print(f"  • Efficiency range: {min_efficiency:.1f}% - {max_efficiency:.1f}%")
print(f"  • Average log-removal: {mean_removal:.2f} log10")
print()
print("🔍 Observation: Higher flows → shorter residence times → less pathogen removal")
print("   This seasonal variation is important for water treatment plant design!")
# %%
# Summary: Key Lessons for Hydrology Students
# ==========================================

print("\n" + "=" * 60)
print("🎓 KEY LESSONS FOR BANK FILTRATION DESIGN")
print("=" * 60)
print()
print("1. 📐 FUNDAMENTAL RELATIONSHIP:")
print("   Log-removal = k * log10(residence_time)")
print("   → Longer underground residence = better pathogen removal")
print()
print("2. ⚖️ FLOW RATE TRADE-OFF:")
print("   Higher pumping rates → shorter residence times → less removal")
print("   → Engineers must balance water demand with treatment efficiency")
print()
print("3. 🚫 AVERAGING MISTAKE TO AVOID:")
print("   NEVER use simple arithmetic averaging for parallel systems!")
print("   → Use parallel_mean() - the shortest path dominates performance")
print()
print("4. 🔧 DESIGN TOOLS:")
print("   Use gamma_find_flow_for_target_mean() to find maximum safe flow rates")
print("   → Essential for meeting drinking water standards")
print("=" * 60)
