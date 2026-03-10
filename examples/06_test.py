import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import gamma as gamma_dist

from gwtransport import advection
from gwtransport import gamma as gamma_utils
from gwtransport.examples import generate_example_data
from gwtransport.logremoval import parallel_mean, residence_time_to_log_removal
from gwtransport.residence_time import fraction_explained, residence_time

np.random.seed(42)
mean = 8000.0
std = 400.0
n_bins = 250

# Generate synthetic bank filtration data
df, tedges = generate_example_data(
    date_start="2020-01-01",
    date_end="2025-05-31",
    flow_mean=120.0,
    flow_amplitude=40.0,
    flow_noise=5.0,
    temp_infiltration_method="soil_temperature",
    aquifer_pore_volume_gamma_mean=mean,
    aquifer_pore_volume_gamma_std=std,
    aquifer_pore_volume_gamma_nbins=n_bins,
    retardation_factor=2.0,
    temp_measurement_noise=0.1,
)

print(f"Period: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Mean flow: {df['flow'].mean():.1f} m³/day")

traindata2 = df["2021-01-01":].copy()
traindata2_tedges = pd.date_range(start=traindata2.index[0], end=traindata2.index[-1] + pd.Timedelta(days=1), freq="D")

cin = advection.gamma_extraction_to_infiltration(
    cout=traindata2.temp_extraction,
    flow=traindata2.flow,
    tedges=traindata2_tedges,
    cout_tedges=traindata2_tedges,
    mean=8000.0,
    std=400.0,
    n_bins=1000,
    retardation_factor=2.0,
    regularization=0.01,
)
traindata2["temp_infiltration_modeled"] = cin
traindata2["temp_infiltration_modeled"].plot()
df.temp_infiltration.plot()
# cin should be very similar to traindata2["temp_infiltration"]
print("Mean absolute error:", np.abs(cin - traindata2.temp_infiltration).mean())