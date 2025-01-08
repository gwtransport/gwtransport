"""Example of residence time estimation using the advection model with a Gamma distribution for the aquifer pore volume."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from gwtransport1d.advection import get_cout_advection_gamma

fp = Path(
    "/Users/bdestombe/Projects/bdestombe/python-pwn-productiecapaciteit-infiltratiegebieden/productiecapaciteit/data/Merged/IK93.feather"
)
df = pd.read_feather(fp).set_index("Datum")
# df = df.groupby(df.index.date).mean()
df.index = pd.to_datetime(df.index)

# Define Gamma distribution for aquifer pore volume
alpha, beta, n_bins = 10.0, 140.0 * 4, 100
retardation_factor = 2.0
rt_est = alpha * beta / df["Q"].mean() * retardation_factor

tout = get_cout_advection_gamma(df.T_bodem, df.Q, alpha, beta, n_bins=100, retardation_factor=2.0)

alphas = [8, 8, 8]
betas = [1300, 1000, 1500]

for _alpha, _beta in zip(alphas, betas, strict=True):
    rt = _alpha * _beta / df["Q"].mean() * retardation_factor
    tout = get_cout_advection_gamma(df.T_bodem, df.Q, _alpha, _beta, n_bins=100, retardation_factor=2.0)
    err = ((tout - df.T_bodem) ** 2).sum()
    print(rt, _alpha, _beta, err)
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
