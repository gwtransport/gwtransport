"""Example of residence time estimation using the advection model with a Gamma distribution for the aquifer pore volume."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma

from gwtransport1d.gamma import cout_advection_gamma, gamma_equal_mass_bins, gamma_mean_std_to_alpha_beta

fp = Path(
    "/Users/bdestombe/Projects/bdestombe/python-pwn-productiecapaciteit-infiltratiegebieden/productiecapaciteit/data/Merged/IK93.feather"
)
df = pd.read_feather(fp).set_index("Datum")
# df = df.groupby(df.index.date).mean()
df.index = pd.to_datetime(df.index)

isspui = ~np.isclose(df.spui, 0.0)

# Define Gamma distribution for aquifer pore volume
alpha, beta, n_bins = 10.0, 140.0 * 4, 100
retardation_factor = 2.0
explainable_fraction = 0.95  # Fraction of the spui that can be explained by the Q

spuiin_fraction = np.clip(a=df.spui / df.Q, a_min=0.0, a_max=1.0, where=df.Q.values > 0.0)

means = 9000, 9000, 9000
stds = 4000, 6000, 8000

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, sharey=False)
secax = ax2.secondary_xaxis("top", functions=(lambda x: x / df.Q.median(), lambda x: x * df.Q.median()))

for i, (mean, std) in enumerate(zip(means, stds, strict=False)):
    c = f"C{i}"
    label = f"mean={mean:.0f}, std={std:.0f}"
    alpha, beta = gamma_mean_std_to_alpha_beta(mean, std)

    tout = cout_advection_gamma(df.T_bodem, df.Q, alpha, beta, n_bins=100, retardation_factor=2.0)

    spuiout_fraction = cout_advection_gamma(
        isspui.astype(float), df.Q, alpha, beta, n_bins=n_bins, retardation_factor=2.0
    )
    isspuiout = spuiout_fraction > (1 - explainable_fraction)
    tout[isspuiout] = np.nan

    err = ((tout - df.T_bodem) ** 2).sum()
    ax1.plot(df.index, tout, lw=0.5, label=label, c=c)

    # plot distribution
    bins = gamma_equal_mass_bins(alpha, beta, n_bins)
    x = np.linspace(0.0, gamma.ppf(0.99, alpha, scale=beta), 100)
    ax2.plot(x, gamma.pdf(x, alpha, scale=beta), c=c, lw=0.5, label=label)

ax1.plot(df.index, df.gwt0, c="C3", lw=0.5, label="gwt0")

ax1.xaxis.tick_top()
secax.tick_params(axis="x", direction="in", pad=-15)
secax.set_xlabel("Residence time with constant median flow [days]")
ax1.set_ylabel("Temperature [°C]")
ax2.set_ylabel("Probability density of flow [-]")
ax2.set_xlabel("Aquifer pore volume [m$^3$]")
ax1.legend(fontsize="x-small", loc="upper right")
ax2.legend(fontsize="x-small", loc="lower right")
plt.savefig("testje.png", dpi=300)

# print("done")
