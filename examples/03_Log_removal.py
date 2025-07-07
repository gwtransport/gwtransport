"""
Example 3: Pathogen log-removal computation in bank filtration systems.

This example demonstrates how to calculate pathogen log-removal efficiency using
the functions in gwtransport.logremoval. Log-removal represents pathogen reduction
on a logarithmic scale (e.g., 3 log10 = 99.9% removal).

Key concepts:
- Log-removal = k * log10(residence_time), where k is the removal rate
- Parallel systems require weighted averaging, not simple averaging
- Design applications: finding flow rates for target removal efficiency
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gwtransport import gamma as gamma_utils
from gwtransport.logremoval import (
    gamma_find_flow_for_target_mean,
    gamma_mean,
    parallel_mean,
)

np.random.seed(42)
plt.style.use("seaborn-v0_8-whitegrid")

# %%
# 1. Basic log-removal calculation
# --------------------------------

print("=== Log-Removal Calculation ===")

# Aquifer parameters
mean_pore_volume = 1000.0  # m³
std_pore_volume = 300.0  # m³
flow_rate = 50.0  # m³/day
log_removal_rate = 1.0  # log10 removal per log10(day)

# Convert to gamma distribution parameters
alpha, beta = gamma_utils.mean_std_to_alpha_beta(mean_pore_volume, std_pore_volume)

# Calculate residence time distribution parameters
rt_alpha = alpha
rt_beta = beta / flow_rate

# Calculate mean log-removal
mean_log_removal = gamma_mean(rt_alpha, rt_beta, log_removal_rate)
removal_efficiency = (1 - 10 ** (-mean_log_removal)) * 100

print(f"Flow rate: {flow_rate} m³/day")
print(f"Mean log-removal: {mean_log_removal:.2f} log10")
print(f"Pathogen removal efficiency: {removal_efficiency:.2f}%")

# %%
# 2. Parallel systems comparison
# ------------------------------

print("\n=== Parallel Systems ===")

# Three parallel treatment units with different removal rates
unit_removals = np.array([2.5, 3.5, 4.0])  # log10

# Correct approach: weighted averaging
combined_removal = parallel_mean(unit_removals)

# Incorrect approach: simple averaging
simple_average = np.mean(unit_removals)

print("Individual units:")
for i, removal in enumerate(unit_removals):
    efficiency = (1 - 10 ** (-removal)) * 100
    print(f"  Unit {i + 1}: {removal:.1f} log10 ({efficiency:.1f}% removal)")

combined_efficiency = (1 - 10 ** (-combined_removal)) * 100
simple_efficiency = (1 - 10 ** (-simple_average)) * 100

print(f"Combined (correct): {combined_removal:.2f} log10 ({combined_efficiency:.1f}% removal)")
print(f"Simple average (wrong): {simple_average:.2f} log10 ({simple_efficiency:.1f}% removal)")

# %%
# 3. Design application
# ---------------------

print("\n=== Design Application ===")

# Find flow rate needed for 99% removal (2 log10)
target_removal = 2.0
required_flow = gamma_find_flow_for_target_mean(
    target_mean=target_removal, apv_alpha=alpha, apv_beta=beta, log_removal_rate=log_removal_rate
)

print(f"For {target_removal} log10 (99% removal):")
print(f"Required flow rate: {required_flow:.1f} m³/day")
print(f"Required mean residence time: {mean_pore_volume / required_flow:.1f} days")

# %%
# 4. Visualization
# ----------------

print("\n=== Visualization ===")
print("Creating bar chart comparison...")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Parallel systems comparison
units = ["Unit 1", "Unit 2", "Unit 3", "Parallel mean\nof 3 units", "Simple Avg (wrong)\nof 3 units"]
removals = [*unit_removals, combined_removal, simple_average]
colors = ["skyblue", "lightgreen", "lightcoral", "gold", "red"]
bars = ax.bar(units, removals, color=colors, alpha=0.7)
ax.set_ylabel("Log-removal [log10]")
ax.set_title("Parallel Systems: Correct vs Incorrect Averaging")
ax.grid(True, alpha=0.3, axis="y")

# Add value labels on bars
for bar, removal in zip(bars, removals, strict=False):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.05, f"{removal:.2f}", ha="center", va="bottom")

plt.tight_layout()

# Save the plot
out_path = Path(__file__).parent / "03_log_removal_analysis.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {out_path}")

# %%
# Summary
# -------

print("\n=== Key Takeaways ===")
print("1. Log-removal depends on residence time: R = k * log10(t)")
print("2. Higher flow rates reduce residence time and log-removal")
print(
    "3. Use parallel_mean() for parallel systems, not simple averaging. Log-removal is weighted towards the shorter residence times."
)
print("4. Design applications: gamma_find_flow_for_target_mean() finds required flow")
