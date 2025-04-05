import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma as gamma_dist

from gwtransport1d.gamma import gamma_equal_mass_bins, gamma_mean_std_to_alpha_beta


def plot_box(ax, box, color="k", alpha=0.5):
    """
    Plot a box on the given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    box : list of float
        The box coordinates [x0, y0, x1, y1].
    color : str
        The color of the box.
    alpha : float
        The transparency of the box.
    """
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor=color, facecolor=color, alpha=alpha))


def max_height_of_gamma(alpha, beta):
    """
    Calculate the maximum height of the gamma distribution.

    Parameters
    ----------
    alpha : float
        Shape parameter of the gamma distribution.
    beta : float
        Scale parameter of the gamma distribution.

    Returns
    -------
    float
        The maximum height of the gamma distribution.

    Example
    -------
    >>> max_height_of_gamma(2.25, 96000)
    3.4815052308495704e-06
    """
    return gamma_dist.pdf(alpha - 1, alpha, scale=beta)


means = 216000, 216000, 216000
stds = 96000, 144000, 192000
i = 1
mean, std = means[i], stds[i]
alpha, beta = gamma_mean_std_to_alpha_beta(mean, std)

n_bins = 8

bins = gamma_equal_mass_bins(alpha, beta, n_bins)
fig, (ax, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 8), sharey=True)
ax.set_title("Gamma distribution")

line_props = {
    "linestyle": "-",
    "linewidth": 0.5,
    "color": "black",
    "alpha": 0.5,
    "zorder": 1,
}


# ax2.axvline(0, color="black", alpha=0.5, linewidth=0.3)
ax.add_patch(plt.Rectangle((0, 0), 1, bins["edges"][-2], edgecolor="none", facecolor="papayawhip", alpha=1))
y_array = np.tile(bins["edges"][None], (2, 1))
ax.plot([0, 1], y_array, color="C0", alpha=0.5)

x = np.linspace(0, bins["edges"][-2], 1000)
y_cdf = 1 - gamma_dist.cdf(x, alpha, scale=beta)
y_pdf = gamma_dist.pdf(x, alpha, scale=beta)
ax2.plot(y_cdf, x, label="Gamma CDF")
ax2.plot(y_pdf / y_pdf.max(), x, label="Gamma PDF")  # normalize PDF to max height of 1.

# ax h/v lines
for p, edge in zip(bins["probability_mass"].cumsum(), bins["edges"][1:], strict=True):
    # ax2.axhline(edge, xmax=p, **line_props)
    ax2.axvline(x=p, ymin=edge, **line_props)

# ax2.hlines(bins["edges"][1:], xmax=bins["probability_mass"].cumsum(), **line_props)
# ax2.vlines(bins["probability_mass"].cumsum(), ymax=bins["edges"][1:], **line_props)
# for edge in bins["edges"]:
#     ax2.axhline(edge, **line_props)

# ax2.axvline(0.0, **line_props)
# for p in bins["probability_mass"].cumsum():
#     ax2.axvline(p, **line_props)

ax.set_ylim(ax.get_ylim()[::-1])

plt.show()

heigth_fraction = bins["expected_value"] / bins["expected_value"].sum()
heigth_fraction_cumsum = heigth_fraction[::-1].cumsum()[::-1]

# plot distribution
cdf_at_edges = bins["edges"]
probability_at_edges = 1 - np.concatenate(([0.0], bins["probability_mass"])).cumsum()

fig, (ax, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))
ax.add_patch(plt.Rectangle((0, 0), 1, 1, edgecolor="black", facecolor="papayawhip", alpha=1))

y_array = np.tile(bins["edges"][None], (2, 1))
ax.plot([0, 1], y_array, color="C0", alpha=0.5)

fig, ax = plt.subplots(figsize=(8, 4))

# plot the gamma distribution and the bins
x = np.linspace(0, bins["edges"][-2], 1000)
y_cdf = 1 - gamma_dist.cdf(x, alpha, scale=beta)
y_pdf = gamma_dist.pdf(x, alpha, scale=beta)
ax.plot(y_cdf, x, label="Gamma CDF")
ax.plot(y_pdf / y_pdf.max(), x, label="Gamma PDF")
ax.scatter(probability_at_edges, cdf_at_edges, color="C0", alpha=0.5)
ax.set_ylim(ax.get_ylim()[::-1])

for i in range(n_bins):
    lower = bins["lower_bound"][i]
    upper = bins["upper_bound"][i]
    expected = bins["expected_value"][i]
