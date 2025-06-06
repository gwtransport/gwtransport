"""
Synthetic Data Generation Functions for Groundwater Transport Examples.

This module provides functions to generate synthetic temperature and flow data
for demonstrating groundwater transport models. It creates realistic time series
with seasonal patterns and random variations.
"""

import numpy as np
import pandas as pd

from gwtransport import advection, gamma


def generate_synthetic_data(
    start_date="2020-01-01",
    end_date="2021-12-31",
    mean_flow=100.0,  # m3/day
    flow_amplitude=30.0,  # m3/day
    flow_noise=10.0,  # m3/day
    mean_temp_infiltration=12.0,  # °C
    temp_infiltration_amplitude=8.0,  # °C
    temp_infiltration_noise=1.0,  # °C
    aquifer_pore_volume=1000.0,  # m3
    aquifer_pore_volume_std=200.0,  # m3
    retardation_factor=1.0,
):
    """
    Generate synthetic temperature and flow data for groundwater transport examples.

    Parameters
    ----------
    start_date : str
        Start date for the time series in 'YYYY-MM-DD' format
    end_date : str
        End date for the time series in 'YYYY-MM-DD' format
    mean_flow : float
        Mean flow rate in m3/day
    flow_amplitude : float
        Seasonal amplitude of flow rate in m3/day
    flow_noise : float
        Random noise level for flow rate in m3/day
    mean_temp_infiltration : float
        Mean temperature of infiltrating water in °C
    temp_infiltration_amplitude : float
        Seasonal amplitude of infiltration temperature in °C
    temp_infiltration_noise : float
        Random noise level for infiltration temperature in °C
    aquifer_pore_volume : float
        Mean pore volume of the aquifer in m3
    aquifer_pore_volume_std : float
        Standard deviation of aquifer pore volume in m3 (for generating heterogeneity)
    retardation_factor : float
        Retardation factor for temperature transport

    Returns
    -------
    pandas.DataFrame
        DataFrame containing dates, flow rates, infiltration temperature, and
        extracted water temperature
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    days = (dates - dates[0]).days.values

    # Generate flow data with seasonal pattern (higher in winter)
    seasonal_flow = mean_flow + flow_amplitude * np.sin(2 * np.pi * days / 365 + np.pi)
    flow = seasonal_flow + np.random.normal(0, flow_noise, len(dates))
    flow = np.maximum(flow, 5.0)  # Ensure flow is not too small or negative

    n_spills = np.random.randint(6, 16)
    for _ in range(n_spills):
        spill_start = np.random.randint(0, len(dates) - 30)
        spill_duration = np.random.randint(15, 45)
        spill_magnitude = np.random.uniform(2.0, 5.0)

        flow[spill_start : spill_start + spill_duration] /= spill_magnitude

    # Generate infiltration temperature with seasonal pattern
    infiltration_temp = mean_temp_infiltration + temp_infiltration_amplitude * np.sin(2 * np.pi * days / 365)
    infiltration_temp += np.random.normal(0, temp_infiltration_noise, len(dates))

    # Create data frame
    df = pd.DataFrame(
        data={
            "flow": flow,
            "temp_infiltration": infiltration_temp,
        },
        index=dates,
    )
    df["temp_extraction"] = advection.gamma_forward(
        cin=df["temp_infiltration"],
        cin_tend=df["temp_infiltration"].index,
        cout_tend=df["temp_infiltration"].index,
        flow=df["flow"],
        flow_tend=df["flow"].index,
        mean=aquifer_pore_volume,
        std=aquifer_pore_volume_std,
        n_bins=1000,
        retardation_factor=retardation_factor,
    )

    # Add some noise to represent measurement errors and other factors
    df["temp_extraction"] += np.random.normal(0, 0.1, len(df))

    # Add some spills (periods with lower extraction temperature due to external factors)
    # Simulate 2-3 spill events of varying duration
    # n_spills = np.random.randint(2, 4)
    # for _ in range(n_spills):
    #     spill_start = np.random.randint(0, len(dates) - 30)
    #     spill_duration = np.random.randint(5, 15)
    #     spill_magnitude = np.random.uniform(2.0, 5.0)
    #     df.iloc[spill_start : spill_start + spill_duration, df.columns.get_loc("temp_extraction")] -= spill_magnitude

    # Add metadata for reference
    alpha, beta = gamma.mean_std_to_alpha_beta(mean=aquifer_pore_volume, std=aquifer_pore_volume_std)
    df.attrs["aquifer_pore_volume_mean"] = aquifer_pore_volume
    df.attrs["aquifer_pore_volume_std"] = aquifer_pore_volume_std
    df.attrs["aquifer_pore_volume_gamma_alpha"] = alpha
    df.attrs["aquifer_pore_volume_gamma_beta"] = beta
    df.attrs["retardation_factor"] = retardation_factor

    return df
