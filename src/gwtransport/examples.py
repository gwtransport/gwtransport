"""
Example Data Generation for Groundwater Transport Modeling.

This module provides utilities to generate synthetic datasets for demonstrating
and testing groundwater transport models. It creates realistic flow patterns,
concentration/temperature time series, and deposition events suitable for testing
advection, diffusion, and deposition analysis functions.

Available functions:

- :func:`generate_example_data` - Generate comprehensive synthetic dataset with flow and
  concentration time series. Creates seasonal flow patterns with optional spill events,
  input concentration data via synthetic sinusoidal patterns, constant values, or real KNMI
  soil temperature, and extracted concentration computed through gamma-distributed pore volume
  transport. When diffusion parameters are provided, uses the diffusion module instead of
  pure advection. Returns DataFrame with flow, cin, cout columns plus attrs containing
  generation parameters and aquifer properties.

- :func:`generate_temperature_example_data` - Convenience wrapper around
  :func:`generate_example_data` with sensible defaults for temperature transport including
  thermal retardation, thermal diffusivity, and longitudinal dispersivity.

- :func:`generate_ec_example_data` - Convenience wrapper around
  :func:`generate_example_data` with sensible defaults for electrical conductivity (EC)
  transport. EC is a conservative tracer (retardation factor 1.0) with negligible molecular
  diffusivity compared to thermal transport.

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
from gwtransport.diffusion_fast import infiltration_to_extraction as diffusion_infiltration_to_extraction
from gwtransport.gamma import bins as gamma_bins
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
    cin_method: str = "synthetic",
    cin_mean: float = 12.0,
    cin_amplitude: float = 8.0,
    measurement_noise: float = 1.0,
    aquifer_pore_volume_gamma_mean: float = 1000.0,  # m3
    aquifer_pore_volume_gamma_std: float = 200.0,  # m3
    aquifer_pore_volume_gamma_nbins: int = 250,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float | None = None,
    longitudinal_dispersivity: float | None = None,
    streamline_length: float | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Generate synthetic concentration/temperature and flow data for groundwater transport.

    Creates a synthetic dataset with seasonal flow patterns, input concentration (cin),
    and output concentration (cout) computed via gamma-distributed pore volume transport.
    When ``molecular_diffusivity``, ``longitudinal_dispersivity``, and ``streamline_length``
    are provided, the diffusion module is used instead of pure advection.

    Parameters
    ----------
    date_start, date_end : str
        Start and end dates for the generated time series (YYYY-MM-DD).
    date_freq : str, default "D"
        Frequency string for pandas.date_range.
    flow_mean : float, default 100.0
        Mean flow rate [m3/day].
    flow_amplitude : float, default 30.0
        Seasonal amplitude of flow rate [m3/day].
    flow_noise : float, default 10.0
        Random noise level for flow rate [m3/day].
    cin_method : str, default "synthetic"
        Method for generating infiltration concentration. Options:

        - ``"synthetic"``: Seasonal sinusoidal pattern defined by ``cin_mean`` and
          ``cin_amplitude``. Measurement noise is applied.
        - ``"constant"``: Constant value equal to ``cin_mean``. Measurement noise
          is still applied.
        - ``"soil_temperature"``: Real soil temperature data from KNMI station 260.
    cin_mean : float, default 12.0
        Mean value of infiltrating concentration.
    cin_amplitude : float, default 8.0
        Seasonal amplitude of infiltration concentration (only used for
        ``"synthetic"`` method).
    measurement_noise : float, default 1.0
        Random noise level applied to both cin and cout to represent
        measurement errors.
    aquifer_pore_volume_gamma_mean : float, default 1000.0
        Mean pore volume of the aquifer gamma distribution [m3].
    aquifer_pore_volume_gamma_std : float, default 200.0
        Standard deviation of aquifer pore volume gamma distribution [m3].
    aquifer_pore_volume_gamma_nbins : int, default 250
        Number of bins to discretize the aquifer pore volume gamma distribution.
    retardation_factor : float, default 1.0
        Retardation factor for transport.
    molecular_diffusivity : float or None, default None
        Effective molecular diffusivity [m2/day]. When provided together with
        ``longitudinal_dispersivity`` and ``streamline_length``, the diffusion
        module is used instead of pure advection. For solutes, typically ~1e-5
        m2/day (negligible). For heat, use thermal diffusivity ~0.01-0.1 m2/day.
    longitudinal_dispersivity : float or None, default None
        Longitudinal dispersivity [m]. Must be provided together with
        ``molecular_diffusivity`` and ``streamline_length``.
    streamline_length : float or None, default None
        Travel distance along the streamline [m]. Must be provided together
        with ``molecular_diffusivity`` and ``longitudinal_dispersivity``.

    Returns
    -------
    tuple
        A tuple containing:

        - pandas.DataFrame: DataFrame with columns ``'flow'``, ``'cin'``,
          ``'cout'`` and metadata attributes for the aquifer parameters.
        - pandas.DatetimeIndex: Time edges (tedges) used for the flow
          calculations.

    Raises
    ------
    ValueError
        If ``cin_method`` is not one of the supported methods, or if only
        some of the diffusion parameters are provided.

    See Also
    --------
    generate_temperature_example_data : Wrapper with thermal transport defaults.
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

    # Generate infiltration concentration. nonoise is needed to compute cout.
    if cin_method == "synthetic":
        # Seasonal pattern with noise
        cin_nonoise = cin_mean + cin_amplitude * np.sin(2 * np.pi * days / 365)
        cin_values = cin_nonoise + np.random.normal(0, measurement_noise, len(dates))
    elif cin_method == "constant":
        # Constant value
        cin_nonoise = np.full(len(dates), cin_mean)
        cin_values = cin_nonoise + np.random.normal(0, measurement_noise, len(dates))
    elif cin_method == "soil_temperature":
        # Use soil temperature data (already includes measurement noise)
        cin_nonoise = cin_values = (
            get_soil_temperature(
                station_number=260,  # Example station number
                interpolate_missing_values=True,
            )["TB3"]
            .resample(date_freq)
            .mean()[dates]
            .values
        )
    else:
        msg = f"Unknown cin_method: {cin_method}"
        raise ValueError(msg)

    # Compute tedges for the flow series
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Compute alpha, beta for gamma distribution
    alpha, beta = mean_std_to_alpha_beta(mean=aquifer_pore_volume_gamma_mean, std=aquifer_pore_volume_gamma_std)

    # Compute cout using diffusion or advection
    diffusion_params = (molecular_diffusivity, longitudinal_dispersivity, streamline_length)
    if any(p is not None for p in diffusion_params):
        if not all(p is not None for p in diffusion_params):
            msg = (
                "molecular_diffusivity, longitudinal_dispersivity, and streamline_length must all be provided together."
            )
            raise ValueError(msg)

        gb = gamma_bins(alpha=alpha, beta=beta, n_bins=aquifer_pore_volume_gamma_nbins)

        cout_values = diffusion_infiltration_to_extraction(
            cin=cin_nonoise,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            aquifer_pore_volumes=gb["expected_values"],
            streamline_length=streamline_length,
            molecular_diffusivity=molecular_diffusivity,
            longitudinal_dispersivity=longitudinal_dispersivity,
            retardation_factor=retardation_factor,
            suppress_dispersion_warning=True,
        )
    else:
        cout_values = gamma_infiltration_to_extraction(
            cin=cin_nonoise,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            mean=aquifer_pore_volume_gamma_mean,
            std=aquifer_pore_volume_gamma_std,
            n_bins=aquifer_pore_volume_gamma_nbins,
            retardation_factor=retardation_factor,
        )

    # Add some noise to represent measurement errors
    cout_values += np.random.normal(0, measurement_noise, len(dates))

    # Create data frame
    df = pd.DataFrame(
        data={"flow": flow, "cin": cin_values, "cout": cout_values},
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
        "cin_method": cin_method,
        "cin_mean": cin_mean,
        "cin_amplitude": cin_amplitude,
        "measurement_noise": measurement_noise,
    }
    if molecular_diffusivity is not None:
        df.attrs["molecular_diffusivity"] = molecular_diffusivity
        df.attrs["longitudinal_dispersivity"] = longitudinal_dispersivity
        df.attrs["streamline_length"] = streamline_length

    return df, tedges


def generate_temperature_example_data(**kwargs: object) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Generate synthetic temperature and flow data for groundwater transport examples.

    Convenience wrapper around :func:`generate_example_data` with sensible
    defaults for temperature transport: thermal retardation factor, thermal
    diffusivity, longitudinal dispersivity, and streamline length.

    Typical parameter values for temperature transport in various sand types:

    +---------------------------------+------------+-------------+--------------------+
    | Parameter                       | Fine sand  | Medium sand | Coarse sand/gravel |
    +=================================+============+=============+====================+
    | retardation_factor R            | 2.0--3.0   | 1.5--2.5    | 1.2--2.0           |
    +---------------------------------+------------+-------------+--------------------+
    | molecular_diffusivity (m2/day)  | 0.03--0.06 | 0.05--0.08  | 0.08--0.12         |
    +---------------------------------+------------+-------------+--------------------+
    | longitudinal_dispersivity (m)   | 0.1--1.0   | 0.5--5.0    | 1.0--10.0          |
    +---------------------------------+------------+-------------+--------------------+
    | streamline_length (m)           | site-specific                                 |
    +---------------------------------+------------+-------------+--------------------+

    Parameters
    ----------
    **kwargs
        All keyword arguments are forwarded to :func:`generate_example_data`.
        Defaults that differ from ``generate_example_data``:

        - ``retardation_factor`` : 2.0 (thermal retardation)
        - ``molecular_diffusivity`` : 0.05 m2/day (thermal diffusivity)
        - ``longitudinal_dispersivity`` : 1.0 m
        - ``streamline_length`` : 100.0 m

    Returns
    -------
    tuple
        See :func:`generate_example_data`.

    See Also
    --------
    generate_example_data : Generic version with full parameter control.
    """
    defaults = {
        "retardation_factor": 2.0,
        "molecular_diffusivity": 0.05,
        "longitudinal_dispersivity": 1.0,
        "streamline_length": 100.0,
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    return generate_example_data(**kwargs)


def generate_ec_example_data(**kwargs: object) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Generate synthetic electrical conductivity and flow data for groundwater transport examples.

    Convenience wrapper around :func:`generate_example_data` with sensible
    defaults for electrical conductivity (EC) transport. EC is a conservative
    tracer: dissolved ions travel at water velocity without retardation.

    Typical parameter values for EC (dissolved ion) transport in various sand
    types. The molecular diffusivity represents effective ionic diffusion in
    porous media (free-water D_0 reduced by porosity/tortuosity). It is
    negligible compared to mechanical dispersion at field scale.

    +---------------------------------+----------------+----------------+--------------------+
    | Parameter                       | Fine sand      | Medium sand    | Coarse sand/gravel |
    +=================================+================+================+====================+
    | retardation_factor R            | 1.0            | 1.0            | 1.0                |
    +---------------------------------+----------------+----------------+--------------------+
    | molecular_diffusivity (m2/day)  | 3e-5 -- 5e-5   | 4e-5 -- 8e-5   | 5e-5 -- 1e-4       |
    +---------------------------------+----------------+----------------+--------------------+
    | longitudinal_dispersivity (m)   | 0.1--1.0       | 0.5--5.0       | 1.0--10.0          |
    +---------------------------------+----------------+----------------+--------------------+
    | streamline_length (m)           | site-specific                                        |
    +---------------------------------+----------------+----------------+--------------------+

    Parameters
    ----------
    **kwargs
        All keyword arguments are forwarded to :func:`generate_example_data`.
        Defaults that differ from ``generate_example_data``:

        - ``cin_mean`` : 500.0 (uS/cm, typical surface water EC)
        - ``cin_amplitude`` : 150.0 (uS/cm, seasonal variation)
        - ``measurement_noise`` : 10.0 (uS/cm)
        - ``retardation_factor`` : 1.0 (conservative tracer)
        - ``molecular_diffusivity`` : 5e-5 m2/day (effective ionic diffusion)
        - ``longitudinal_dispersivity`` : 1.0 m
        - ``streamline_length`` : 100.0 m

    Returns
    -------
    tuple
        See :func:`generate_example_data`.

    See Also
    --------
    generate_example_data : Generic version with full parameter control.
    generate_temperature_example_data : Wrapper with thermal transport defaults.
    """
    defaults = {
        "cin_mean": 500.0,
        "cin_amplitude": 150.0,
        "measurement_noise": 10.0,
        "retardation_factor": 1.0,
        "molecular_diffusivity": 5e-5,
        "longitudinal_dispersivity": 1.0,
        "streamline_length": 100.0,
    }
    for key, value in defaults.items():
        kwargs.setdefault(key, value)
    return generate_example_data(**kwargs)


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
