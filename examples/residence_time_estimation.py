"""Example of residence time estimation using the advection model with a Gamma distribution for the aquifer pore volume."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gwtransport1d.gamma import cout_advection_gamma
from gwtransport1d.residence_time import residence_time_retarded
from gwtransport1d.utils import find_consecutive_true_ranges, interp_series

fp = Path(
    "/Users/bdestombe/Projects/bdestombe/python-pwn-productiecapaciteit-infiltratiegebieden/productiecapaciteit/data/Merged/IK93.feather"
)
df = pd.read_feather(fp).set_index("Datum")
# df = df.groupby(df.index.date).mean()
df.index = pd.to_datetime(df.index)

isspui = ~np.isclose(df.spui, 0.0)
# spui_period_indices = find_consecutive_true_ranges(isspui, timestamp_at_end_of_index=True)
# spui_period_times = df.index[spui_period_indices.flatten()]
# df["rt_infiltration"] = residence_time_retarded(df.Q, 1300 * 8, 2, direction="infiltration")[0]
# rt = pd.to_timedelta(interp_series(df["rt_infiltration"], spui_period_times), unit="D")
# projected_period_times = (spui_period_times + rt).values.reshape((-1, 2))


# Define Gamma distribution for aquifer pore volume
alpha, beta, n_bins = 10.0, 140.0 * 4, 100
retardation_factor = 2.0
rt_est = alpha * beta / df["Q"].mean() * retardation_factor


tout = cout_advection_gamma(df.T_bodem, df.Q, alpha, beta, n_bins=100, retardation_factor=2.0)

alphas = [8, 8, 8]
betas = [1300, 1000, 1500]

for _alpha, _beta in zip(alphas, betas, strict=True):
    rt = _alpha * _beta / df["Q"].mean() * retardation_factor
    tout = cout_advection_gamma(df.T_bodem, df.Q, _alpha, _beta, n_bins=100, retardation_factor=2.0)
    err = ((tout - df.T_bodem) ** 2).sum()
    print(rt, _alpha, _beta, err)  # noqa: T201
    plt.plot(df.index, tout, label=f"alpha={_alpha:.0f}, beta={_beta:.0f}")

plt.plot(df.index, df.gwt0, c="C3", label="gwt0")
plt.legend()
plt.show()

# plt.figure()
# plt.plot(df.index, df.T_bodem, c="C2", label="T_bodem")
# plt.plot(df.index, df.gwt0, c="C3", label="gwt0")
# plt.plot(df.index, tout, c="C1", label="mean1")
# plt.legend()

# plt.show()

# def fun(x):
#     print(x)
#     _alpha, _beta = x
#     out = (get_cout_advection_gamma(df.T_bodem, df.Q, _alpha, _beta, n_bins=100, retardation_factor=2.0) - df.T_bodem).values
#     out[np.isnan(out)] = 0.0
#     out[np.isinf(out)] = 0.0
#     return out

# res = fsolve(fun, x0=[alpha, beta])
# res = least_squares(fun, x0=[alpha, beta], loss='cauchy', gtol=None, verbose=2)
# print(res)
