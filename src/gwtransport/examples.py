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
  generation parameters and aquifer properties, and time edges (tedges).

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
  deposition rates [ng/m²/day] and attrs containing generation parameters, and time edges
  (tedges). Useful for testing extraction_to_deposition deconvolution and
  deposition_to_extraction convolution functions.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.advection import gamma_infiltration_to_extraction, infiltration_to_extraction
from gwtransport.diffusion_fast import gamma_infiltration_to_extraction as diffusion_gamma_infiltration_to_extraction
from gwtransport.diffusion_fast import infiltration_to_extraction as diffusion_infiltration_to_extraction
from gwtransport.gamma import mean_std_loc_to_alpha_beta
from gwtransport.utils import compute_time_edges, get_soil_temperature

_DEFAULT_GAMMA_MEAN = 1000.0  # m³
_DEFAULT_GAMMA_STD = 200.0  # m³
_DEFAULT_GAMMA_LOC = 0.0  # m³, minimum pore volume
_DEFAULT_GAMMA_NBINS = 250


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
    aquifer_pore_volumes: npt.ArrayLike | None = None,
    aquifer_pore_volume_gamma_mean: float | None = None,
    aquifer_pore_volume_gamma_std: float | None = None,
    aquifer_pore_volume_gamma_loc: float | None = None,
    aquifer_pore_volume_gamma_nbins: int | None = None,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float | None = None,
    longitudinal_dispersivity: float | None = None,
    streamline_length: float | None = None,
    rng: np.random.Generator | int | None = None,
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
        Mean flow rate [m³/day].
    flow_amplitude : float, default 30.0
        Seasonal amplitude of flow rate [m³/day].
    flow_noise : float, default 10.0
        Random noise level for flow rate [m³/day].
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
        Standard deviation of the Gaussian measurement noise applied
        independently to ``cin`` and ``cout``. Because the two noise draws are
        independent, applying the forward operator to ``df['cin']`` does not
        exactly reproduce ``df['cout']`` when ``measurement_noise > 0``; the
        underlying noiseless signals remain consistent.
    aquifer_pore_volumes : array-like or None, default None
        Discrete aquifer pore volumes [m³] representing the distribution of
        residence times. When provided, the gamma distribution is bypassed and
        none of the ``aquifer_pore_volume_gamma_*`` parameters may be passed.
        When ``None``, the pore volume distribution is built from the gamma
        parameters below.
    aquifer_pore_volume_gamma_mean : float or None, default None
        Mean pore volume of the aquifer gamma distribution [m³] (default 1000.0
        when unset). Must be strictly greater than
        ``aquifer_pore_volume_gamma_loc``. Mutually exclusive with
        ``aquifer_pore_volumes``.
    aquifer_pore_volume_gamma_std : float or None, default None
        Standard deviation of aquifer pore volume gamma distribution [m³]
        (default 200.0 when unset; invariant under the ``loc`` shift).
        Mutually exclusive with ``aquifer_pore_volumes``.
    aquifer_pore_volume_gamma_loc : float or None, default None
        Location (minimum pore volume) of the aquifer gamma distribution [m³]
        (default 0.0 when unset). Must satisfy ``0 <= loc < mean``. Mutually
        exclusive with ``aquifer_pore_volumes``.
    aquifer_pore_volume_gamma_nbins : int or None, default None
        Number of bins to discretize the aquifer pore volume gamma distribution
        (default 250 when unset). Mutually exclusive with
        ``aquifer_pore_volumes``.
    retardation_factor : float, default 1.0
        Retardation factor for transport.
    molecular_diffusivity : float or None, default None
        Effective molecular diffusivity [m²/day]. When provided together with
        ``longitudinal_dispersivity`` and ``streamline_length``, the diffusion
        module is used instead of pure advection. For solutes, typically ~1e-5
        m²/day (negligible). For heat, use thermal diffusivity ~0.01-0.1 m²/day.
    longitudinal_dispersivity : float or None, default None
        Longitudinal dispersivity [m]. Must be provided together with
        ``molecular_diffusivity`` and ``streamline_length``.
    streamline_length : float or None, default None
        Travel distance along the streamline [m]. Must be provided together
        with ``molecular_diffusivity`` and ``longitudinal_dispersivity``.
    rng : numpy.random.Generator, int, or None, default None
        Source of randomness for the synthetic flow noise, spill events, and
        measurement noise. Accepted in any form supported by
        :func:`numpy.random.default_rng`. Pass an integer (or
        :class:`numpy.random.Generator`) for reproducible output; ``None``
        draws fresh entropy each call.

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
        If ``cin_method`` is not one of the supported methods, if only some
        of the diffusion parameters are provided, or if ``aquifer_pore_volumes``
        is passed together with any ``aquifer_pore_volume_gamma_*`` parameter.

    See Also
    --------
    generate_temperature_example_data : Wrapper with thermal transport defaults.
    generate_ec_example_data : Wrapper with EC transport defaults.
    """
    rng = np.random.default_rng(rng)

    dates = pd.date_range(start=date_start, end=date_end, freq=date_freq).tz_localize("UTC")
    days = (dates - dates[0]).days.values

    # Generate flow data with seasonal pattern (higher in winter)
    seasonal_flow = flow_mean + flow_amplitude * np.sin(2 * np.pi * days / 365 + np.pi)
    flow = seasonal_flow + rng.normal(0, flow_noise, len(dates))

    min_days_for_spills = 60
    if len(dates) > min_days_for_spills:  # Only add spills for longer time series
        n_spills = int(rng.integers(6, 16))
        for _ in range(n_spills):
            spill_start = int(rng.integers(0, len(dates) - 30))
            spill_duration = int(rng.integers(15, 45))
            spill_magnitude = float(rng.uniform(2.0, 5.0))

            flow[spill_start : spill_start + spill_duration] /= spill_magnitude

    # Enforce a positive flow floor after spills so residence times remain finite.
    flow = np.maximum(flow, 5.0)

    # Generate infiltration concentration. nonoise is needed to compute cout.
    if cin_method == "synthetic":
        # Seasonal pattern with noise
        cin_nonoise = cin_mean + cin_amplitude * np.sin(2 * np.pi * days / 365)
        cin_values = cin_nonoise + rng.normal(0, measurement_noise, len(dates))
    elif cin_method == "constant":
        # Constant value
        cin_nonoise = np.full(len(dates), cin_mean)
        cin_values = cin_nonoise + rng.normal(0, measurement_noise, len(dates))
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

    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=len(dates))

    # Validate pore volume parameterization: either discrete volumes or gamma parameters, not both.
    gamma_set_by_user = [
        name
        for name, value in {
            "aquifer_pore_volume_gamma_mean": aquifer_pore_volume_gamma_mean,
            "aquifer_pore_volume_gamma_std": aquifer_pore_volume_gamma_std,
            "aquifer_pore_volume_gamma_loc": aquifer_pore_volume_gamma_loc,
            "aquifer_pore_volume_gamma_nbins": aquifer_pore_volume_gamma_nbins,
        }.items()
        if value is not None
    ]
    if aquifer_pore_volumes is not None and gamma_set_by_user:
        msg = (
            "aquifer_pore_volumes is mutually exclusive with the aquifer_pore_volume_gamma_* "
            f"parameters; got both aquifer_pore_volumes and {gamma_set_by_user}."
        )
        raise ValueError(msg)

    # Validate diffusion parameterization: all three parameters provided or none.
    diffusion_provided = (molecular_diffusivity, longitudinal_dispersivity, streamline_length)
    n_diffusion = sum(1 for p in diffusion_provided if p is not None)
    if 0 < n_diffusion < len(diffusion_provided):
        msg = "molecular_diffusivity, longitudinal_dispersivity, and streamline_length must all be provided together."
        raise ValueError(msg)
    # Validation above forbids partial-set states, so this conjunction is equivalent to any single check;
    # writing it in full lets the type checker narrow all three params to non-None inside the branches below.
    use_diffusion = (
        molecular_diffusivity is not None and longitudinal_dispersivity is not None and streamline_length is not None
    )

    # Fill in gamma defaults so downstream callers see concrete values (not used when
    # aquifer_pore_volumes is supplied, but kept in scope for the attrs block below).
    gamma_mean = aquifer_pore_volume_gamma_mean if aquifer_pore_volume_gamma_mean is not None else _DEFAULT_GAMMA_MEAN
    gamma_std = aquifer_pore_volume_gamma_std if aquifer_pore_volume_gamma_std is not None else _DEFAULT_GAMMA_STD
    gamma_loc = aquifer_pore_volume_gamma_loc if aquifer_pore_volume_gamma_loc is not None else _DEFAULT_GAMMA_LOC
    gamma_nbins = (
        aquifer_pore_volume_gamma_nbins if aquifer_pore_volume_gamma_nbins is not None else _DEFAULT_GAMMA_NBINS
    )

    # Compute cout. Branch on pore volume parameterization, then on diffusion.
    if aquifer_pore_volumes is not None:
        aquifer_pore_volumes_array = np.asarray(aquifer_pore_volumes, dtype=float)
        if use_diffusion:
            cout_values = diffusion_infiltration_to_extraction(
                cin=cin_nonoise,
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                aquifer_pore_volumes=aquifer_pore_volumes_array,
                streamline_length=streamline_length,
                molecular_diffusivity=molecular_diffusivity,
                longitudinal_dispersivity=longitudinal_dispersivity,
                retardation_factor=retardation_factor,
            )
        else:
            cout_values = infiltration_to_extraction(
                cin=cin_nonoise,
                flow=flow,
                tedges=tedges,
                cout_tedges=tedges,
                aquifer_pore_volumes=aquifer_pore_volumes_array,
                retardation_factor=retardation_factor,
            )
    elif use_diffusion:
        cout_values = diffusion_gamma_infiltration_to_extraction(
            cin=cin_nonoise,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            mean=gamma_mean,
            std=gamma_std,
            loc=gamma_loc,
            n_bins=gamma_nbins,
            streamline_length=streamline_length,
            molecular_diffusivity=molecular_diffusivity,
            longitudinal_dispersivity=longitudinal_dispersivity,
            retardation_factor=retardation_factor,
        )
    else:
        cout_values = gamma_infiltration_to_extraction(
            cin=cin_nonoise,
            flow=flow,
            tedges=tedges,
            cout_tedges=tedges,
            mean=gamma_mean,
            std=gamma_std,
            loc=gamma_loc,
            n_bins=gamma_nbins,
            retardation_factor=retardation_factor,
        )

    # Add some noise to represent measurement errors
    cout_values += rng.normal(0, measurement_noise, len(dates))

    df = pd.DataFrame(
        data={"flow": flow, "cin": cin_values, "cout": cout_values},
        index=dates,
    )
    df.attrs.update({
        "description": "Example data for groundwater transport modeling",
        "source": "Synthetic data generated by gwtransport.examples.generate_example_data",
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
    })
    if aquifer_pore_volumes is not None:
        df.attrs["aquifer_pore_volume_parameterization"] = "discrete"
        df.attrs["aquifer_pore_volumes"] = aquifer_pore_volumes_array
    else:
        alpha, beta = mean_std_loc_to_alpha_beta(mean=gamma_mean, std=gamma_std, loc=gamma_loc)
        df.attrs.update({
            "aquifer_pore_volume_parameterization": "gamma",
            "aquifer_pore_volume_gamma_mean": gamma_mean,
            "aquifer_pore_volume_gamma_std": gamma_std,
            "aquifer_pore_volume_gamma_loc": gamma_loc,
            "aquifer_pore_volume_gamma_alpha": alpha,
            "aquifer_pore_volume_gamma_beta": beta,
            "aquifer_pore_volume_gamma_nbins": gamma_nbins,
        })
    if molecular_diffusivity is not None:
        df.attrs["molecular_diffusivity"] = molecular_diffusivity
        df.attrs["longitudinal_dispersivity"] = longitudinal_dispersivity
        df.attrs["streamline_length"] = streamline_length

    return df, tedges


def generate_temperature_example_data(
    *,
    date_start: str = "2020-01-01",
    date_end: str = "2021-12-31",
    date_freq: str = "D",
    flow_mean: float = 100.0,
    flow_amplitude: float = 30.0,
    flow_noise: float = 10.0,
    cin_method: str = "synthetic",
    cin_mean: float = 12.0,
    cin_amplitude: float = 8.0,
    measurement_noise: float = 1.0,
    aquifer_pore_volumes: npt.ArrayLike | None = None,
    aquifer_pore_volume_gamma_mean: float | None = None,
    aquifer_pore_volume_gamma_std: float | None = None,
    aquifer_pore_volume_gamma_loc: float | None = None,
    aquifer_pore_volume_gamma_nbins: int | None = None,
    retardation_factor: float = 2.0,
    molecular_diffusivity: float = 0.05,
    longitudinal_dispersivity: float = 1.0,
    streamline_length: float = 100.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
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
    | molecular_diffusivity (m²/day)  | 0.03--0.06 | 0.05--0.08  | 0.08--0.12         |
    +---------------------------------+------------+-------------+--------------------+
    | longitudinal_dispersivity (m)   | 0.1--1.0   | 0.5--5.0    | 1.0--10.0          |
    +---------------------------------+------------+-------------+--------------------+
    | streamline_length (m)           | site-specific                                 |
    +---------------------------------+------------+-------------+--------------------+

    Parameters
    ----------
    retardation_factor : float, default 2.0
        Thermal retardation factor.
    molecular_diffusivity : float, default 0.05
        Thermal diffusivity [m²/day].
    longitudinal_dispersivity : float, default 1.0
        Longitudinal dispersivity [m].
    streamline_length : float, default 100.0
        Travel distance along the streamline [m].

    Returns
    -------
    tuple
        See :func:`generate_example_data`.

    See Also
    --------
    generate_example_data : Generic version with full parameter control.
    generate_ec_example_data : Wrapper with EC transport defaults.

    Notes
    -----
    All other parameters are forwarded unchanged to :func:`generate_example_data`;
    see that function for their descriptions.
    """
    return generate_example_data(
        date_start=date_start,
        date_end=date_end,
        date_freq=date_freq,
        flow_mean=flow_mean,
        flow_amplitude=flow_amplitude,
        flow_noise=flow_noise,
        cin_method=cin_method,
        cin_mean=cin_mean,
        cin_amplitude=cin_amplitude,
        measurement_noise=measurement_noise,
        aquifer_pore_volumes=aquifer_pore_volumes,
        aquifer_pore_volume_gamma_mean=aquifer_pore_volume_gamma_mean,
        aquifer_pore_volume_gamma_std=aquifer_pore_volume_gamma_std,
        aquifer_pore_volume_gamma_loc=aquifer_pore_volume_gamma_loc,
        aquifer_pore_volume_gamma_nbins=aquifer_pore_volume_gamma_nbins,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        streamline_length=streamline_length,
        rng=rng,
    )


def generate_ec_example_data(
    *,
    date_start: str = "2020-01-01",
    date_end: str = "2021-12-31",
    date_freq: str = "D",
    flow_mean: float = 100.0,
    flow_amplitude: float = 30.0,
    flow_noise: float = 10.0,
    cin_method: str = "synthetic",
    cin_mean: float = 500.0,
    cin_amplitude: float = 150.0,
    measurement_noise: float = 10.0,
    aquifer_pore_volumes: npt.ArrayLike | None = None,
    aquifer_pore_volume_gamma_mean: float | None = None,
    aquifer_pore_volume_gamma_std: float | None = None,
    aquifer_pore_volume_gamma_loc: float | None = None,
    aquifer_pore_volume_gamma_nbins: int | None = None,
    retardation_factor: float = 1.0,
    molecular_diffusivity: float = 5e-5,
    longitudinal_dispersivity: float = 1.0,
    streamline_length: float = 100.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
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
    | molecular_diffusivity (m²/day)  | 3e-5 -- 5e-5   | 4e-5 -- 8e-5   | 5e-5 -- 1e-4       |
    +---------------------------------+----------------+----------------+--------------------+
    | longitudinal_dispersivity (m)   | 0.1--1.0       | 0.5--5.0       | 1.0--10.0          |
    +---------------------------------+----------------+----------------+--------------------+
    | streamline_length (m)           | site-specific                                        |
    +---------------------------------+----------------+----------------+--------------------+

    Parameters
    ----------
    cin_mean : float, default 500.0
        Mean infiltration EC [uS/cm, typical surface water EC].
    cin_amplitude : float, default 150.0
        Seasonal amplitude of infiltration EC [uS/cm].
    measurement_noise : float, default 10.0
        Standard deviation of the Gaussian measurement noise [uS/cm].
    retardation_factor : float, default 1.0
        Retardation factor (1.0 for a conservative tracer).
    molecular_diffusivity : float, default 5e-5
        Effective ionic diffusion [m²/day].
    longitudinal_dispersivity : float, default 1.0
        Longitudinal dispersivity [m].
    streamline_length : float, default 100.0
        Travel distance along the streamline [m].

    Returns
    -------
    tuple
        See :func:`generate_example_data`.

    See Also
    --------
    generate_example_data : Generic version with full parameter control.
    generate_temperature_example_data : Wrapper with thermal transport defaults.

    Notes
    -----
    All other parameters are forwarded unchanged to :func:`generate_example_data`;
    see that function for their descriptions.
    """
    return generate_example_data(
        date_start=date_start,
        date_end=date_end,
        date_freq=date_freq,
        flow_mean=flow_mean,
        flow_amplitude=flow_amplitude,
        flow_noise=flow_noise,
        cin_method=cin_method,
        cin_mean=cin_mean,
        cin_amplitude=cin_amplitude,
        measurement_noise=measurement_noise,
        aquifer_pore_volumes=aquifer_pore_volumes,
        aquifer_pore_volume_gamma_mean=aquifer_pore_volume_gamma_mean,
        aquifer_pore_volume_gamma_std=aquifer_pore_volume_gamma_std,
        aquifer_pore_volume_gamma_loc=aquifer_pore_volume_gamma_loc,
        aquifer_pore_volume_gamma_nbins=aquifer_pore_volume_gamma_nbins,
        retardation_factor=retardation_factor,
        molecular_diffusivity=molecular_diffusivity,
        longitudinal_dispersivity=longitudinal_dispersivity,
        streamline_length=streamline_length,
        rng=rng,
    )


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
    rng: np.random.Generator | int | None = None,
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
        Baseline deposition rate (ng/m²/day).
    seasonal_amplitude : float
        Amplitude of the annual seasonal sinusoidal pattern (ng/m²/day).
    noise_scale : float
        Standard deviation of Gaussian noise added to the signal (ng/m²/day).
    event_dates : list-like or None
        Dates (strings or pandas-compatible) at which to place episodic events.
        Time-zone-naive entries are interpreted as UTC to match the generated
        ``dates`` index. If None, a sensible default list is used.
    event_magnitude : float
        Peak deposition added at event onset (ng/m²/day). Decays exponentially
        over ``event_duration`` days at rate ``event_decay_scale``.
    event_duration : int
        Duration of each event in days.
    event_decay_scale : float
        Decay scale used in the exponential decay for event time series.
    ensure_non_negative : bool
        If True, negative values are clipped to zero.
    rng : numpy.random.Generator, int, or None, default None
        Source of randomness for the additive Gaussian noise. Accepted in any
        form supported by :func:`numpy.random.default_rng`. Pass an integer
        (or :class:`numpy.random.Generator`) for reproducible output; ``None``
        draws fresh entropy each call.

    Returns
    -------
    tuple
        A tuple containing:

        - pandas.Series: Deposition time series (ng/m²/day) indexed by UTC
          timestamps.
        - pandas.DatetimeIndex: Time bin edges (n+1 edges for n values).

    Raises
    ------
    ValueError
        If ``event_decay_scale`` or ``event_duration`` is not positive, or if any
        ``event_dates`` entry falls outside the generated ``dates`` range.

    See Also
    --------
    gwtransport.deposition.deposition_to_extraction : Forward operator consuming this data.
    gwtransport.deposition.extraction_to_deposition : Inverse operator.
    """
    if event_decay_scale <= 0:
        msg = f"event_decay_scale must be positive, got {event_decay_scale}"
        raise ValueError(msg)
    if event_duration <= 0:
        msg = f"event_duration must be positive, got {event_duration}"
        raise ValueError(msg)

    rng = np.random.default_rng(rng)

    dates = pd.date_range(date_start, date_end, freq=freq).tz_localize("UTC")
    n_dates = len(dates)
    tedges = compute_time_edges(tedges=None, tstart=None, tend=dates, number_of_bins=n_dates)

    # Base deposition rate with seasonal and event patterns
    seasonal_pattern = seasonal_amplitude * np.sin(2 * np.pi * np.arange(n_dates) / 365.25)
    noise = noise_scale * rng.normal(0, 1, n_dates)

    # Default event dates if not provided
    if event_dates is None:
        event_dates = ["2020-06-15", "2021-03-20", "2021-09-10", "2022-07-05"]
    event_dates_index = pd.DatetimeIndex(pd.to_datetime(np.asarray(event_dates)))
    # Match the timezone of `dates` so naive user input (and the string defaults)
    # can be compared against the tz-aware index in `get_indexer`.
    if event_dates_index.tz is None:
        event_dates_index = event_dates_index.tz_localize(dates.tz)
    else:
        event_dates_index = event_dates_index.tz_convert(dates.tz)

    out_of_range = (event_dates_index < dates[0]) | (event_dates_index > dates[-1])
    if out_of_range.any():
        msg = (
            f"event_dates contains {out_of_range.sum()} date(s) outside the dates range "
            f"[{dates[0]}, {dates[-1]}]: {event_dates_index[out_of_range].tolist()}"
        )
        raise ValueError(msg)

    # Vectorized event accumulation. For each event start, scatter a ``(n_events, event_duration)``
    # decay block into ``event`` via ``np.add.at`` so overlapping events sum correctly. The
    # boundary mask drops indices that fall past the end of the series (preserves the loop's
    # ``min(event_idx + event_duration, n_dates)`` clipping).
    starts = dates.get_indexer(event_dates_index, method="nearest")
    cols = np.arange(event_duration)
    flat_indices = starts[:, None] + cols[None, :]
    valid = flat_indices < n_dates
    decay_block = np.broadcast_to(event_magnitude * np.exp(-cols / event_decay_scale), flat_indices.shape)
    event = np.zeros(n_dates)
    np.add.at(event, flat_indices[valid], decay_block[valid])

    # Combine all components
    total = base + seasonal_pattern + noise + event
    if ensure_non_negative:
        total = np.maximum(total, 0.0)

    series = pd.Series(data=total, index=dates, name="deposition")
    series.attrs.update({
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
    })

    return series, tedges
