"""Example of residence time estimation using the advection model with a Gamma distribution for the aquifer pore volume.

means, stds
IK93: 9000, 6000
IK94:
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma

from gwtransport1d.gamma import cout_advection_gamma, gamma_equal_mass_bins, gamma_mean_std_to_alpha_beta

fp = Path(
    "/Users/bdestombe/Projects/bdestombe/python-pwn-productiecapaciteit-infiltratiegebieden/productiecapaciteit/data/Merged/IK94.feather"
)
df = pd.read_feather(fp).set_index("Datum")
# df = df.groupby(df.index.date).mean()
df.index = pd.to_datetime(df.index)
df.Q *= 24.0  # m3/day
df.spui *= 24.0  # m3/day

isspui = ~np.isclose(df.spui, 0.0)

# Define Gamma distribution for aquifer pore volume
alpha, beta, n_bins = 10.0, 140.0 * 4, 100
retardation_factor = 2.0
explainable_fraction = 0.95  # Fraction of the spui that can be explained by the Q
flow_min = 10.0  # Minimum flow rate to consider otherwise temperature in well resembles atmo

means = 216000, 216000, 216000
stds = 96000, 144000, 192000
tout = np.zeros(shape=(len(means), len(df)), dtype=float)
tout_masks = np.zeros(shape=(len(means), len(df)), dtype=bool)
flow_min = 10.0  # Minimum flow rate to consider otherwise temperature in well resembles atmo

means = 3 * [6750]
stds = 4000, 5000, 6000

tout = np.zeros(shape=(len(means), len(df)), dtype=float)
tout_masks = np.zeros(shape=(len(means), len(df)), dtype=bool)

for i, (mean, std) in enumerate(zip(means, stds, strict=False)):
    alpha, beta = gamma_mean_std_to_alpha_beta(mean, std)
    tout[i] = cout_advection_gamma(df.T_bodem, df.Q, alpha, beta, n_bins=100, retardation_factor=2.0).values
    tout[i] = cout_advection_gamma(df.T_bodem, df.Q, alpha, beta, n_bins=100, retardation_factor=2.0).values
    spuiout_fraction = cout_advection_gamma(
        isspui.astype(float), df.Q, alpha, beta, n_bins=n_bins, retardation_factor=2.0
    )
    isspuiout = spuiout_fraction > (1 - explainable_fraction)
    tout_masks[i, isspuiout] = True

tout_masks[:, flow_min > df.Q] = True

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, sharey=False)
secax = ax3.secondary_xaxis("top", functions=(lambda x: x / df.Q.median(), lambda x: x * df.Q.median()))
lkwargs = {"lw": 0.8}
ax2.axhline(0.0, c="black", lw=0.5)

for i, (mean, std) in enumerate(zip(means, stds, strict=False)):
    c = f"C{i}"
    label = f"mean={mean:.0f}, std={std:.0f}"
    ax1.plot(df.index, tout[i], alpha=0.3, c=c, **lkwargs)
    tout[i, tout_masks[i]] = np.nan
    ax1.plot(df.index, tout[i], **lkwargs, label=label, c=c)

    err = tout[i] - df.gwt0.values
    ax2.plot(df.index, err, label=label, c=c)

c = f"C{i + 1}"
ax1.plot(df.index, df.T_bodem, alpha=0.3, **lkwargs, c=c)
df.loc[isspui, "T_bodem"] = np.nan
ax1.plot(df.index, df.T_bodem, label="Infiltratie", lw=0.4, c=c)

c = f"C{i + 2}"
ax1.plot(df.index, df.gwt0, alpha=0.3, label="gwt0", **lkwargs, c=c)
a = df.gwt0.values.copy()
df.loc[tout_masks.sum(axis=0).astype(bool), "gwt0"] = np.nan
ax1.plot(df.index, df.gwt0, **lkwargs, c=c)

# c = f"C{i + 3}"
# ax1.plot(df.index, df.spui, alpha=0.3, label="spui", **lkwargs, c=c)

# c = f"C{i + 4}"
# ax1.plot(df.index, df.Q, alpha=0.3, label="Q", **lkwargs, c=c)

for i, (mean, std) in enumerate(zip(means, stds, strict=False)):
    c = f"C{i}"
    label = f"mean={mean:.0f}, std={std:.0f}"
    alpha, beta = gamma_mean_std_to_alpha_beta(mean, std)
    tout_masks[i, isspuiout] = True

tout_masks[:, flow_min > df.Q] = True

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, sharey=False)
secax = ax3.secondary_xaxis("top", functions=(lambda x: x / df.Q.median(), lambda x: x * df.Q.median()))
lkwargs = {"lw": 0.8}
ax2.axhline(0.0, c="black", lw=0.5)

for i, (mean, std) in enumerate(zip(means, stds, strict=False)):
    c = f"C{i}"
    label = f"mean={mean:.0f}, std={std:.0f}"
    ax1.plot(df.index, tout[i], alpha=0.3, c=c, **lkwargs)
    tout[i, tout_masks[i]] = np.nan
    ax1.plot(df.index, tout[i], **lkwargs, label=label, c=c)

    err = tout[i] - df.gwt0.values
    ax2.plot(df.index, err, label=label, c=c)

c = f"C{i + 1}"
ax1.plot(df.index, df.T_bodem, alpha=0.3, **lkwargs, c=c)
df.loc[isspui, "T_bodem"] = np.nan
ax1.plot(df.index, df.T_bodem, label="Infiltratie", lw=0.4, c=c)

c = f"C{i + 2}"
ax1.plot(df.index, df.gwt0, alpha=0.3, label="gwt0", **lkwargs, c=c)
a = df.gwt0.values.copy()
df.loc[tout_masks.sum(axis=0).astype(bool), "gwt0"] = np.nan
ax1.plot(df.index, df.gwt0, **lkwargs, c=c)

# c = f"C{i + 3}"
# ax1.plot(df.index, df.spui, alpha=0.3, label="spui", **lkwargs, c=c)

# c = f"C{i + 4}"
# ax1.plot(df.index, df.Q, alpha=0.3, label="Q", **lkwargs, c=c)

for i, (mean, std) in enumerate(zip(means, stds, strict=False)):
    c = f"C{i}"
    label = f"mean={mean:.0f}, std={std:.0f}"
    alpha, beta = gamma_mean_std_to_alpha_beta(mean, std)

    # plot distribution
    bins = gamma_equal_mass_bins(alpha, beta, n_bins)
    x = np.linspace(0.0, gamma.ppf(0.99, alpha, scale=beta), 100)
    ax3.plot(x, gamma.pdf(x, alpha, scale=beta), c=c, lw=0.5, label=label)
    ax3.plot(x, gamma.pdf(x, alpha, scale=beta), c=c, lw=0.5, label=label)

ax1.xaxis.tick_top()
secax.tick_params(axis="x", direction="in", pad=-15)
secax.set_xlabel("Residence time with constant median flow [days]")
ax1.set_ylabel("Temperature [°C]")
ax2.set_ylabel("Temperature [°C]")
ax3.set_ylabel("Probability density of flow [-]")
ax3.set_xlabel("Aquifer pore volume [m$^3$]")
ax2.set_ylabel("Temperature [°C]")
ax3.set_ylabel("Probability density of flow [-]")
ax3.set_xlabel("Aquifer pore volume [m$^3$]")
ax1.legend(fontsize="x-small", loc="upper right")
ax2.legend(fontsize="x-small", loc="upper right")
ax3.legend(fontsize="x-small", loc="lower right")
ax2.legend(fontsize="x-small", loc="upper right")
ax3.legend(fontsize="x-small", loc="lower right")
plt.savefig("testje.png", dpi=300)
