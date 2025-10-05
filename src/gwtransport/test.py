
import numpy as np
import pandas as pd
from gwtransport.advection import gamma_infiltration_to_extraction
from gwtransport.gamma import alpha_beta_to_mean_std
tedges = pd.date_range(start="2020-01-01", end="2020-01-07", freq="D")

# Use a step function to see retardation effects
cin = np.ones(len(tedges) - 1)
cin[2:] = 2.0  # Step change on day 3 (January 3)
flow = np.ones(len(tedges) - 1) * 100.0

# Compare results with different retardation factors
cout1 = gamma_infiltration_to_extraction(
    cin=cin,
    tedges=tedges,
    cout_tedges=tedges,
    flow=flow,
    mean=100.0,
    std=30.0,
    retardation_factor=1.0,
    n_bins=20,
)
print(cout1)