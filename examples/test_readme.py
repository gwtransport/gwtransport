"""Example script testing README functionality."""

import numpy as np
import pandas as pd

from gwtransport.advection import distribution_infiltration_to_extraction

# Measurements
cin_data = [1.0, 2.0, 3.0]  # Example concentration infiltrated water
flow_data = [100.0, 150.0, 100.0]  # Example flow rates
tedges = pd.date_range(start="2020-01-05", end="2020-01-08", freq="D")  # Example time edges

areas_between_streamlines = np.array([100.0, 90.0, 110.0])  # Example areas
depth_aquifer = 2.0  # Convert areas between 2d streamlines to 3d aquifer pore volumes.
aquifer_pore_volumes = areas_between_streamlines * depth_aquifer

cout = distribution_infiltration_to_extraction(
    cin=cin_data,
    flow=flow_data,
    tedges=tedges,
    cout_tedges=tedges,
    aquifer_pore_volumes=aquifer_pore_volumes,
    retardation_factor=1.0,
)
