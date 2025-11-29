"""
Example Data Generation for Groundwater Transport Modeling.

This module provides utilities to generate synthetic datasets for demonstrating
and testing groundwater transport models. It creates realistic flow patterns,
temperature/concentration time series, and deposition events suitable for testing
advection, diffusion, and deposition analysis functions.

Available functions:

- :func:`generate_example_data` - Generate comprehensive synthetic dataset with flow and
  temperature time series. Creates seasonal flow patterns with optional spill events,
  temperature data via synthetic sinusoidal patterns or real KNMI soil temperature, and
  extracted temperature computed through gamma-distributed pore volume transport. Returns
  DataFrame with flow, temp_infiltration, temp_extraction columns plus attrs containing
  generation parameters and aquifer properties. Temperature generation methods: 'synthetic'
  (seasonal sinusoidal pattern), 'constant' (constant temperature with noise), or
  'soil_temperature' (real data from KNMI station 260).

- :func:`generate_example_deposition_timeseries` - Generate synthetic deposition time series
  for pathogen/contaminant deposition analysis. Combines baseline deposition, seasonal patterns,
  random noise, and episodic contamination events with exponential decay. Returns Series with
  deposition rates [ng/m²/day] and attrs containing generation parameters. Useful for testing
  extraction_to_deposition deconvolution and deposition_to_extraction convolution functions.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.advection import gamma_infiltration_to_extraction
from gwtransport.gamma import mean_std_to_alpha_beta
from gwtransport.utils import compute_time_edges, get_soil_temperature


def generate_example_data(
    *,
    date_start: str = "2020-01-01",
    date_end: str = "2021-12-31",
    date_freq: str = "D",
    flow_mean: float = 100.0,  # m3/day
    flow_amplitude: float = 30.0,  # m3/day
    flow_noise: float = 10.0,  # m3/day
    temp_infiltration_method: str = "synthetic",  # Method for generating infiltration temperature
    temp_infiltration_mean: float = 12.0,  # °C
    temp_infiltration_amplitude: float = 8.0,  # °C
    temp_measurement_noise: float = 1.0,  # °C
    aquifer_pore_volume_gamma_mean: float = 1000.0,  # m3
    aquifer_pore_volume_gamma_std: float = 200.0,  # m3
    aquifer_pore_volume_gamma_nbins: int = 250,  # Discretization resolution
    retardation_factor: float = 1.0,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Generate synthetic temperature and flow data for groundwater transport examples.

    Parameters
    ----------
    date_start, date_end : str
        Start and end dates for the generated time series (YYYY-MM-DD).
    date_freq : str, default "D"
        Frequency string for pandas.date_range (default 'D').
    flow_mean : float, default 100.0
        Mean flow rate in m3/day
    flow_amplitude : float, default 30.0
        Seasonal amplitude of flow rate in m3/day
    flow_noise : float, default 10.0
        Random noise level for flow rate in m3/day
    temp_infiltration_method : str, default "synthetic"
        Method for generating infiltration temperature. Options:
        - "synthetic": Seasonal pattern. temp_infiltration_mean and temp_infiltration_amplitude define the pattern. temp_measurement_noise is applied.
        - "constant": Constant temperature equal to temp_infiltration_mean. temp_measurement_noise is still applied.
        - "soil_temperature": Use real soil temperature data from KNMI station
    temp_infiltration_mean : float, default 12.0
        Mean temperature of infiltrating water in °C
    temp_infiltration_amplitude : float, default 8.0
        Seasonal amplitude of infiltration temperature in °C (only used for "synthetic" method)
    temp_measurement_noise : float, default 1.0
        Random noise level for infiltration temperature in °C
    aquifer_pore_volume_gamma_mean : float, default 1000.0
        Mean pore volume of the aquifer gamma distribution in m3
    aquifer_pore_volume_gamma_std : float, default 200.0
        Standard deviation of aquifer pore volume gamma distribution in m3
    aquifer_pore_volume_gamma_nbins : int, default 250
        Number of bins to discretize the aquifer pore volume gamma distribution
    retardation_factor : float, default 1.0
        Retardation factor for temperature transport

    Returns
    -------
    tuple
        A tuple containing:
        - pandas.DataFrame: DataFrame with columns 'flow', 'temp_infiltration', 'temp_extraction'
          and metadata attributes for the aquifer parameters
        - pandas.DatetimeIndex: Time edges (tedges) used for the flow calculations

    Raises
    ------
    ValueError
        If temp_infiltration_method is not one of the supported methods
    """
    # Create date range
    dates = pd.date_range(start=date_start, end=date_end, freq=date_freq).tz_localize("UTC")
    days = (dates - dates[0]).days.values

    # Generate flow data with seasonal pattern (higher in winter)
    seasonal_flow = flow_mean + flow_amplitude * np.sin(2 * np.pi * days / 365 + np.pi)
    flow = seasonal_flow + np.random.normal(0, flow_noise, len(dates))
    flow = np.maximum(flow, 5.0)  # Ensure flow is not too small or negative

    min_days_for_spills = 60
    if len(dates) > min_days_for_spills:  # Only add spills for longer time series
        n_spills = np.random.randint(6, 16)
        for _ in range(n_spills):
            spill_start = np.random.randint(0, len(dates) - 30)
            spill_duration = np.random.randint(15, 45)
            spill_magnitude = np.random.uniform(2.0, 5.0)

            flow[spill_start : spill_start + spill_duration] /= spill_magnitude

    # Generate infiltration temperature
    if temp_infiltration_method == "synthetic":
        # Seasonal pattern with noise
        infiltration_temp = temp_infiltration_mean + temp_infiltration_amplitude * np.sin(2 * np.pi * days / 365)
        infiltration_temp += np.random.normal(0, temp_measurement_noise, len(dates))
    elif temp_infiltration_method == "constant":
        # Constant temperature
        infiltration_temp = np.full(len(dates), temp_infiltration_mean)
        infiltration_temp += np.random.normal(0, temp_measurement_noise, len(dates))
    elif temp_infiltration_method == "soil_temperature":
        # Use soil temperature data already includes measurement noise
        infiltration_temp = (
            get_soil_temperature(
                station_number=260,  # Example station number
                interpolate_missing_values=True,
            )["TB3"]
            .resample(date_freq)
            .mean()[dates]
            .values
        )
    else:
        msg = f"Unknown temperature method: {temp_infiltration_method}"
        raise ValueError(msg)

    # Compute tedges for the flow series
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    temp_extraction = gamma_infiltration_to_extraction(
        cin=infiltration_temp,
        flow=flow,
        tedges=tedges,
        cout_tedges=tedges,
        mean=aquifer_pore_volume_gamma_mean,  # Use mean pore volume
        std=aquifer_pore_volume_gamma_std,  # Use standard deviation for heterogeneity
        n_bins=aquifer_pore_volume_gamma_nbins,  # Discretization resolution
        retardation_factor=retardation_factor,
    )

    # Add some noise to represent measurement errors
    temp_extraction += np.random.normal(0, temp_measurement_noise, len(dates))

    # Create data frame
    alpha, beta = mean_std_to_alpha_beta(mean=aquifer_pore_volume_gamma_mean, std=aquifer_pore_volume_gamma_std)
    df = pd.DataFrame(
        data={"flow": flow, "temp_infiltration": infiltration_temp, "temp_extraction": temp_extraction},
        index=dates,
    )
    df.attrs = {
        "description": "Example data for groundwater transport modeling",
        "source": "Synthetic data generated by gwtransport.examples.generate_example_data",
        "aquifer_pore_volume_gamma_mean": aquifer_pore_volume_gamma_mean,
        "aquifer_pore_volume_gamma_std": aquifer_pore_volume_gamma_std,
        "aquifer_pore_volume_gamma_alpha": alpha,
        "aquifer_pore_volume_gamma_beta": beta,
        "aquifer_pore_volume_gamma_nbins": aquifer_pore_volume_gamma_nbins,
        "retardation_factor": retardation_factor,
        "date_start": date_start,
        "date_end": date_end,
        "date_freq": date_freq,
        "flow_mean": flow_mean,
        "flow_amplitude": flow_amplitude,
        "flow_noise": flow_noise,
        "temp_infiltration_method": temp_infiltration_method,
        "temp_infiltration_mean": temp_infiltration_mean,
        "temp_infiltration_amplitude": temp_infiltration_amplitude,
        "temp_measurement_noise": temp_measurement_noise,
    }
    return df, tedges


def generate_example_deposition_timeseries(
    *,
    date_start: str = "2018-01-01",
    date_end: str = "2023-12-31",
    freq: str = "D",
    base: float = 0.8,
    seasonal_amplitude: float = 0.3,
    noise_scale: float = 0.1,
    event_dates: npt.ArrayLike | pd.DatetimeIndex | None = None,
    event_magnitude: float = 3.0,
    event_duration: int = 30,
    event_decay_scale: float = 10.0,
    ensure_non_negative: bool = True,
) -> tuple[pd.Series, pd.DatetimeIndex]:
    """
    Generate synthetic deposition timeseries for groundwater transport examples.

    Parameters
    ----------
    date_start, date_end : str
        Start and end dates for the generated time series (YYYY-MM-DD).
    freq : str
        Frequency string for pandas.date_range (default 'D').
    base : float
        Baseline deposition rate (ng/m^2/day).
    seasonal_amplitude : float
        Amplitude of the annual seasonal sinusoidal pattern.
    noise_scale : float
        Standard deviation scale for Gaussian noise added to the signal.
    event_dates : list-like or None
        Dates (strings or pandas-compatible) at which to place episodic events. If None,
        a sensible default list is used.
    event_magnitude : float
        Peak magnitude multiplier for events.
    event_duration : int
        Duration of each event in days.
    event_decay_scale : float
        Decay scale used in the exponential decay for event time series.
    ensure_non_negative : bool
        If True, negative values are clipped to zero.

    Returns
    -------
    pandas.Series
        Time series of deposition values indexed by daily timestamps.
    """
    # Create synthetic deposition time series - needs to match flow period
    dates = pd.date_range(date_start, date_end, freq=freq).tz_localize("UTC")
    n_dates = len(dates)
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n_dates)

    # Base deposition rate with seasonal and event patterns
    seasonal_pattern = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_dates) / 365.25)
    noise = noise_scale * np.random.normal(0, 1, n_dates)

    # Default event dates if not provided
    if event_dates is None:
        event_dates = ["2020-06-15", "2021-03-20", "2021-09-10", "2022-07-05"]
    # Convert to DatetimeIndex - handles list, array, or DatetimeIndex input
    if isinstance(event_dates, pd.DatetimeIndex):
        event_dates_index = event_dates
    else:
        # Convert ArrayLike to list for pd.to_datetime
        event_dates_list = event_dates if isinstance(event_dates, list) else list(np.asarray(event_dates))
        event_dates_index = pd.DatetimeIndex(pd.to_datetime(event_dates_list))

    event = np.zeros(n_dates)
    for event_date in event_dates_index:
        event_idx = dates.get_indexer([event_date], method="nearest")[0]
        event_indices = np.arange(event_idx, min(event_idx + event_duration, n_dates))
        decay_pattern = event_magnitude * np.exp(-np.arange(len(event_indices)) / event_decay_scale)
        event[event_indices] += decay_pattern

    # Combine all components
    total = base + seasonal_pattern + noise + event
    if ensure_non_negative:
        total = np.maximum(total, 0.0)

    series = pd.Series(data=total, index=dates, name="deposition")
    series.attrs = {
        "description": "Example deposition time series for groundwater transport modeling",
        "source": "Synthetic data generated by gwtransport.examples.generate_example_deposition_timeseries",
        "base": base,
        "seasonal_amplitude": seasonal_amplitude,
        "noise_scale": noise_scale,
        "event_dates": [str(d.date()) for d in event_dates_index],
        "event_magnitude": event_magnitude,
        "event_duration": event_duration,
        "event_decay_scale": event_decay_scale,
        "date_start": date_start,
        "date_end": date_end,
        "date_freq": freq,
    }

    # Create deposition series
    return series, tedges
