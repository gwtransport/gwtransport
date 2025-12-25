"""
Mathematical Foundation for Front Tracking with Nonlinear Sorption.

This module provides exact analytical computations for:
- Freundlich and constant retardation models
- Shock velocities via Rankine-Hugoniot condition
- Characteristic velocities and positions
- First arrival time calculations
- Entropy condition verification

All computations are exact analytical formulas with no numerical tolerances.

This file is part of gwtransport which is released under AGPL-3.0 license.
See the ./LICENSE file or go to https://github.com/gwtransport/gwtransport/blob/main/LICENSE for full license details.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

# Numerical tolerance constants
EPSILON_FREUNDLICH_N = 1e-10  # Tolerance for checking if n ≈ 1.0
EPSILON_EXPONENT = 1e-10  # Tolerance for checking if exponent ≈ 0
EPSILON_DENOMINATOR = 1e-18  # Tolerance for near-zero denominators in shock velocity


@dataclass
class FreundlichSorption:
    """
    Freundlich sorption isotherm with exact analytical methods.

    The Freundlich isotherm is: s(C) = k_f * C^(1/n)

    where:
    - s is sorbed concentration [mass/mass of solid]
    - C is dissolved concentration [mass/volume of water]
    - k_f is Freundlich coefficient [(volume/mass)^(1/n)]
    - n is Freundlich exponent (dimensionless)

    For n > 1: Higher C travels faster
    For n < 1: Higher C travels slower
    For n = 1: linear (not supported, use ConstantRetardation instead)

    Parameters
    ----------
    k_f : float
        Freundlich coefficient [(m³/kg)^(1/n)]. Must be positive.
    n : float
        Freundlich exponent [-]. Must be positive and != 1.
    bulk_density : float
        Bulk density of porous medium [kg/m³]. Must be positive.
    porosity : float
        Porosity [-]. Must be in (0, 1).
    c_min : float, optional
        Minimum concentration threshold. For n>1, prevents infinite retardation
        as C→0. Default: 0.1 for n>1, 0.0 for n<1 (set automatically if not provided).

    Attributes
    ----------
    k_f : float
        Freundlich coefficient
    n : float
        Freundlich exponent
    bulk_density : float
        Bulk density
    porosity : float
        Porosity
    c_min : float
        Minimum concentration threshold

    Methods
    -------
    retardation(c)
        Compute retardation factor R(C)
    total_concentration(c)
        Compute total concentration C_total(C) = C + (rho_b/n_por)*s(C)
    concentration_from_retardation(r)
        Invert R → C analytically
    shock_velocity(c_left, c_right, flow)
        Compute shock velocity via Rankine-Hugoniot condition
    check_entropy_condition(c_left, c_right, shock_vel, flow)
        Verify Lax entropy condition

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> r = sorption.retardation(5.0)
    >>> c_back = sorption.concentration_from_retardation(r)
    >>> bool(np.isclose(c_back, 5.0))
    True

    Notes
    -----
    The retardation factor is defined as:
        R(C) = 1 + (rho_b/n_por) * ds/dC
             = 1 + (rho_b*k_f)/(n_por*n) * C^((1/n)-1)

    For Freundlich sorption, R depends on C, which creates nonlinear wave behavior.

    For n>1 (higher C travels faster), R(C)→∞ as C→0, which can cause extremely slow
    wave propagation. The c_min parameter prevents this by enforcing a minimum
    concentration, making R(C) finite for all C≥0.
    """

    k_f: float
    n: float
    bulk_density: float
    porosity: float
    c_min: float = 1e-12

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.k_f <= 0:
            msg = f"k_f must be positive, got {self.k_f}"
            raise ValueError(msg)
        if self.n <= 0:
            msg = f"n must be positive, got {self.n}"
            raise ValueError(msg)
        if abs(self.n - 1.0) < EPSILON_FREUNDLICH_N:
            msg = "n = 1 (linear case) not supported, use ConstantRetardation instead"
            raise ValueError(msg)
        if self.bulk_density <= 0:
            msg = f"bulk_density must be positive, got {self.bulk_density}"
            raise ValueError(msg)
        if not 0 < self.porosity < 1:
            msg = f"porosity must be in (0, 1), got {self.porosity}"
            raise ValueError(msg)
        if self.c_min < 0:
            msg = f"c_min must be non-negative, got {self.c_min}"
            raise ValueError(msg)

    def retardation(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Compute retardation factor R(C).

        The retardation factor relates concentration velocity to pore water velocity:
            v_C = flow / R(C)

        For Freundlich sorption:
            R(C) = 1 + (rho_b*k_f)/(n_por*n) * C^((1/n)-1)

        Parameters
        ----------
        c : float or array_like
            Dissolved concentration [mass/volume]. Non-negative.

        Returns
        -------
        r : float or ndarray
            Retardation factor [-]. Always >= 1.0.

        Notes
        -----
        - For n > 1: R decreases with increasing C (higher C travels faster)
        - For n < 1: R increases with increasing C (higher C travels slower)
        - Concentrations at or below c_min return R=1 if c_min=0, else are clipped to c_min
        """
        is_array = isinstance(c, np.ndarray)
        c_arr = np.asarray(c)

        if self.c_min == 0 and self.n < 1.0:
            # Only for n<1 (lower C travels faster) where R(0)=1 is physically correct
            result = np.where(c_arr <= 0, 1.0, self._compute_retardation(c_arr))
        else:
            c_eff = np.maximum(c_arr, self.c_min)
            result = self._compute_retardation(c_eff)

        return result if is_array else float(result)

    def _compute_retardation(self, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Compute retardation for positive concentrations."""
        exponent = (1.0 / self.n) - 1.0
        coefficient = (self.bulk_density * self.k_f) / (self.porosity * self.n)
        return 1.0 + coefficient * (c**exponent)

    def total_concentration(self, c: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Compute total concentration (dissolved + sorbed per unit pore volume).

        Total concentration includes both dissolved and sorbed mass:
            C_total = C + (rho_b/n_por) * s(C)
                    = C + (rho_b/n_por) * k_f * C^(1/n)

        Parameters
        ----------
        c : float or array_like
            Dissolved concentration [mass/volume]. Non-negative.

        Returns
        -------
        c_total : float or ndarray
            Total concentration [mass/volume]. Always >= c.

        Notes
        -----
        This is the conserved quantity in the transport equation:
            ∂C_total/∂t + ∂(flow*C)/∂v = 0

        The flux term only includes dissolved concentration because sorbed mass
        is immobile. Concentrations at or below c_min return C if c_min=0, else use c_min.
        """
        is_array = isinstance(c, np.ndarray)
        c_arr = np.asarray(c)

        if self.c_min == 0 and self.n < 1.0:
            # Only for n<1 (lower C travels faster) where C=0 is physically valid
            sorbed = np.where(
                c_arr <= 0, 0.0, (self.bulk_density / self.porosity) * self.k_f * (c_arr ** (1.0 / self.n))
            )
        else:
            c_eff = np.maximum(c_arr, self.c_min)
            sorbed = (self.bulk_density / self.porosity) * self.k_f * (c_eff ** (1.0 / self.n))

        result = c_arr + sorbed
        return result if is_array else float(result)

    def concentration_from_retardation(self, r: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Invert retardation factor to obtain concentration analytically.

        Given R, solves R = retardation(C) for C. This is used in rarefaction waves
        where the self-similar solution gives R as a function of position and time.

        Parameters
        ----------
        r : float or array_like
            Retardation factor [-]. Must be >= 1.0.

        Returns
        -------
        c : float or ndarray
            Dissolved concentration [mass/volume]. Non-negative.

        Notes
        -----
        This inverts the relation:
            R = 1 + (rho_b*k_f)/(n_por*n) * C^((1/n)-1)

        The analytical solution is:
            C = [(R-1) * n_por*n / (rho_b*k_f)]^(n/(1-n))

        For n = 1 (linear sorption), the exponent n/(1-n) is undefined, which is
        why linear sorption must use ConstantRetardation class instead.

        Examples
        --------
        >>> sorption = FreundlichSorption(
        ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
        ... )
        >>> r = sorption.retardation(5.0)
        >>> c = sorption.concentration_from_retardation(r)
        >>> bool(np.isclose(c, 5.0, rtol=1e-14))
        True
        """
        is_array = isinstance(r, np.ndarray)
        r_arr = np.asarray(r)

        exponent = (1.0 / self.n) - 1.0

        if abs(exponent) < EPSILON_EXPONENT:
            msg = "Cannot invert linear retardation (n=1)"
            raise ValueError(msg)

        coefficient = (self.bulk_density * self.k_f) / (self.porosity * self.n)
        base = (r_arr - 1.0) / coefficient

        inversion_exponent = 1.0 / exponent
        c = base**inversion_exponent
        result = np.maximum(c, self.c_min)
        result = np.where(r_arr <= 1.0, self.c_min, result)
        result = np.where(base <= 0, self.c_min, result)

        return result if is_array else float(result)

    def shock_velocity(self, c_left: float, c_right: float, flow: float) -> float:
        """
        Compute shock velocity via Rankine-Hugoniot condition.

        The Rankine-Hugoniot condition ensures mass conservation across the shock:
            s_shock = [flux(C_R) - flux(C_L)] / [C_total(C_R) - C_total(C_L)]

        where flux(C) = flow * C (only dissolved species are transported).

        Parameters
        ----------
        c_left : float
            Concentration upstream (behind) shock [mass/volume].
        c_right : float
            Concentration downstream (ahead of) shock [mass/volume].
        flow : float
            Flow rate [volume/time]. Must be positive.

        Returns
        -------
        s_shock : float
            Shock velocity [volume/time].

        Notes
        -----
        The Rankine-Hugoniot condition is derived from integrating the conservation
        law across the shock discontinuity. It ensures that the total mass flux
        (advective transport) is conserved.

        For physical shocks with n > 1 (higher C travels faster):
        - c_left > c_right (concentration decreases across shock)
        - The shock velocity is between the characteristic velocities

        Examples
        --------
        >>> sorption = FreundlichSorption(
        ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
        ... )
        >>> v_shock = sorption.shock_velocity(c_left=10.0, c_right=2.0, flow=100.0)
        >>> v_shock > 0
        True
        """
        # Flux = flow * C (only dissolved species flow)
        flux_left = flow * c_left
        flux_right = flow * c_right

        # Total concentration (dissolved + sorbed)
        c_total_left = self.total_concentration(c_left)
        c_total_right = self.total_concentration(c_right)

        # Rankine-Hugoniot condition
        # s_shock = Δflux / ΔC_total
        denom = c_total_right - c_total_left

        # Guard against degenerate "shock" states where the total
        # concentration jump tends to zero. In the analytic limit
        # ΔC_total → 0, the Rankine-Hugoniot speed approaches the
        # characteristic velocity, so we fall back to that value
        # instead of dividing by an extremely small number.
        if abs(denom) < EPSILON_DENOMINATOR:
            return float(flow / self.retardation(c_left))

        return float((flux_right - flux_left) / denom)

    def check_entropy_condition(self, c_left: float, c_right: float, shock_vel: float, flow: float) -> bool:
        """
        Verify Lax entropy condition for physical admissibility of shock.

        The Lax entropy condition ensures that characteristics flow INTO the shock
        from both sides, which is required for physical shocks:
            λ(C_L) > s_shock > λ(C_R)

        where λ(C) = flow / R(C) is the characteristic velocity.

        Parameters
        ----------
        c_left : float
            Concentration upstream of shock [mass/volume].
        c_right : float
            Concentration downstream of shock [mass/volume].
        shock_vel : float
            Shock velocity [volume/time].
        flow : float
            Flow rate [volume/time].

        Returns
        -------
        satisfies : bool
            True if shock satisfies entropy condition (is physical).

        Notes
        -----
        Shocks that violate the entropy condition are unphysical and should be
        replaced by rarefaction waves. The entropy condition prevents non-physical
        expansion shocks.

        For n > 1 (higher C travels faster):
        - Physical shocks have c_left > c_right
        - Characteristic from left is faster: λ(c_left) > λ(c_right)
        - Shock velocity is between them

        For n < 1 (lower C travels faster):
        - Physical shocks have c_left < c_right
        - Characteristic from left is slower: λ(c_left) < λ(c_right)
        - Shock velocity is still between them

        Examples
        --------
        >>> sorption = FreundlichSorption(
        ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
        ... )
        >>> # Physical shock (compression for n>1)
        >>> v_shock = sorption.shock_velocity(10.0, 2.0, 100.0)
        >>> sorption.check_entropy_condition(10.0, 2.0, v_shock, 100.0)
        True
        >>> # Unphysical shock (expansion for n>1)
        >>> v_shock_bad = sorption.shock_velocity(2.0, 10.0, 100.0)
        >>> sorption.check_entropy_condition(2.0, 10.0, v_shock_bad, 100.0)
        False
        """
        # Characteristic velocities
        lambda_left = flow / self.retardation(c_left)
        lambda_right = flow / self.retardation(c_right)

        # If any of the velocities are non-finite (can occur for
        # test-generated edge states), treat the entropy condition as
        # violated rather than propagating RuntimeWarnings.
        if not np.isfinite(lambda_left) or not np.isfinite(lambda_right) or not np.isfinite(shock_vel):
            return False

        # Lax condition: λ(C_L) > s_shock > λ(C_R)
        # Use small tolerance for floating-point comparison
        tolerance = 1e-14 * max(abs(lambda_left), abs(lambda_right), abs(shock_vel))

        return bool((lambda_left > shock_vel + tolerance) and (shock_vel > lambda_right - tolerance))


@dataclass
class ConstantRetardation:
    """
    Constant (linear) retardation model.

    For linear sorption: s(C) = K_d * C
    This gives constant retardation: R(C) = 1 + (rho_b/n_por) * K_d = constant

    This is a special case where concentration-dependent behavior disappears.
    Used for conservative tracers or as approximation for weak sorption.

    Parameters
    ----------
    retardation_factor : float
        Constant retardation factor [-]. Must be >= 1.0.
        R = 1.0 means no retardation (conservative tracer).

    Attributes
    ----------
    retardation_factor : float
        Constant retardation value

    Methods
    -------
    retardation(c)
        Return constant retardation (independent of c)
    total_concentration(c)
        Compute C_total = C * retardation_factor
    concentration_from_retardation(r)
        Not applicable for constant R (raises error)
    shock_velocity(c_left, c_right, flow)
        Compute shock velocity (simplified for constant R)
    check_entropy_condition(c_left, c_right, shock_vel, flow)
        Check entropy (always satisfied for compression shocks)

    Notes
    -----
    With constant retardation:
    - All concentrations travel at same velocity: flow / R
    - No rarefaction waves form (all concentrations travel together)
    - Shocks occur only at concentration discontinuities at inlet
    - Solution reduces to simple time-shifting

    This is equivalent to using `infiltration_to_extraction_series` in the
    gwtransport package.

    Examples
    --------
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> sorption.retardation(5.0)
    2.0
    >>> sorption.retardation(10.0)
    2.0
    """

    retardation_factor: float

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.retardation_factor < 1.0:
            msg = f"retardation_factor must be >= 1.0, got {self.retardation_factor}"
            raise ValueError(msg)

    def retardation(self, c: float) -> float:  # noqa: ARG002
        """
        Return constant retardation factor (independent of concentration).

        Parameters
        ----------
        c : float
            Dissolved concentration (not used for constant retardation).

        Returns
        -------
        r : float
            Constant retardation factor.
        """
        return self.retardation_factor

    def total_concentration(self, c: float) -> float:
        """
        Compute total concentration for linear sorption.

        For constant retardation:
            C_total = C * R

        Parameters
        ----------
        c : float
            Dissolved concentration [mass/volume].

        Returns
        -------
        c_total : float
            Total concentration [mass/volume].
        """
        return c * self.retardation_factor

    def concentration_from_retardation(self, r: float) -> float:
        """
        Not applicable for constant retardation.

        With constant R, all concentrations have the same retardation, so
        inversion is not meaningful. This method raises an error.

        Raises
        ------
        NotImplementedError
            Always raised for constant retardation.
        """
        msg = "concentration_from_retardation not applicable for ConstantRetardation (R is independent of C)"
        raise NotImplementedError(msg)

    def shock_velocity(self, c_left: float, c_right: float, flow: float) -> float:  # noqa: ARG002
        """
        Compute shock velocity for constant retardation.

        With constant R, the shock velocity simplifies to:
            s_shock = flow / R

        This is the same as all characteristic velocities.

        Parameters
        ----------
        c_left : float
            Concentration upstream of shock (not used for constant R).
        c_right : float
            Concentration downstream of shock (not used for constant R).
        flow : float
            Flow rate [volume/time].

        Returns
        -------
        s_shock : float
            Shock velocity [volume/time].
        """
        return flow / self.retardation_factor

    def check_entropy_condition(self, c_left: float, c_right: float, shock_vel: float, flow: float) -> bool:  # noqa: ARG002, PLR6301
        """
        Check entropy condition for constant retardation.

        With constant R, all characteristic velocities are equal, so the
        entropy condition is trivially satisfied for any shock (or rather,
        shocks don't really exist - they're just contact discontinuities).

        Parameters
        ----------
        c_left : float
            Concentration upstream.
        c_right : float
            Concentration downstream.
        shock_vel : float
            Shock velocity.
        flow : float
            Flow rate.

        Returns
        -------
        satisfies : bool
            Always True for constant retardation.
        """
        return True


def characteristic_velocity(c: float, flow: float, sorption: FreundlichSorption | ConstantRetardation) -> float:
    """
    Compute characteristic velocity for given concentration.

    In smooth regions of the solution, concentration travels at velocity:
        v = flow / R(C)

    Parameters
    ----------
    c : float
        Dissolved concentration [mass/volume].
    flow : float
        Flow rate [volume/time].
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.

    Returns
    -------
    velocity : float
        Characteristic velocity [volume/time].

    Examples
    --------
    >>> sorption = FreundlichSorption(
    ...     k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3
    ... )
    >>> v = characteristic_velocity(c=5.0, flow=100.0, sorption=sorption)
    >>> v > 0
    True
    """
    return float(flow / sorption.retardation(c))


def characteristic_position(
    c: float, flow: float, sorption: FreundlichSorption | ConstantRetardation, t_start: float, v_start: float, t: float
) -> float | None:
    """
    Compute exact position of characteristic at time t.

    Characteristics propagate linearly in time:
        V(t) = v_start + velocity * (t - t_start)

    where velocity = flow / R(C) is constant along the characteristic.

    Parameters
    ----------
    c : float
        Concentration carried by characteristic [mass/volume].
    flow : float
        Flow rate [volume/time].
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.
    t_start : float
        Time when characteristic starts [days].
    v_start : float
        Starting position [volume].
    t : float
        Time at which to compute position [days].

    Returns
    -------
    position : float or None
        Position at time t [volume], or None if t < t_start.

    Examples
    --------
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> v = characteristic_position(
    ...     c=5.0, flow=100.0, sorption=sorption, t_start=0.0, v_start=0.0, t=10.0
    ... )
    >>> bool(np.isclose(v, 500.0))  # v = 100/2 * 10 = 500
    True
    """
    if t < t_start:
        return None

    velocity = characteristic_velocity(c, flow, sorption)
    dt_days = t - t_start

    return v_start + velocity * dt_days


def compute_first_front_arrival_time(
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    sorption: FreundlichSorption | ConstantRetardation,
) -> float:
    """
    Compute exact time when first wave reaches outlet (v_max).

    This function returns the precise moment when the first non-zero
    concentration wave from the inlet arrives at the outlet. This marks
    the end of the spin-up period.

    For the typical case where the first inlet change creates a characteristic
    (e.g., 0→C transition), this is when that characteristic reaches v_max.

    For cases with rarefaction waves:
    - n>1 (higher C travels faster): The head of a rarefaction
      (higher C) arrives first.
    - n<1 (lower C travels faster): The head of a rarefaction
      (lower C) arrives first.

    Algorithm:
    1. Find first index where cin > 0
    2. Compute residence time for this concentration from inlet to outlet
    3. Account for piecewise constant flow during transit
    4. Return arrival time in days from tedges[0]

    Parameters
    ----------
    cin : numpy.ndarray
        Inlet concentration [mass/volume]. Length = len(tedges) - 1.
    flow : numpy.ndarray
        Flow rate [volume/time]. Length = len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges. Length = len(cin) + 1.
        Expected to be DatetimeIndex.
    aquifer_pore_volume : float
        Total pore volume [volume]. Must be positive.
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.

    Returns
    -------
    t_first_arrival : float
        Time when first wave reaches outlet, measured in days from tedges[0].
        Returns np.inf if no concentration ever arrives.

    Notes
    -----
    The residence time accounts for retardation:
        residence_time = aquifer_pore_volume * R(C) / flow_avg

    For piecewise constant flow, we integrate:
        ∫₀^residence_time flow(t) dt = aquifer_pore_volume * R(C)

    This function computes the EXACT crossing time in days, not a bin edge.
    Use compute_first_fully_informed_bin_edge() to get the corresponding
    output bin edge for masking purposes.

    Examples
    --------
    >>> import pandas as pd
    >>> cin = np.array([0.0, 10.0] + [10.0] * 10)  # First bin zero, then nonzero
    >>> flow = np.array([100.0] * 12)  # Constant flow
    >>> tedges = pd.date_range("2020-01-01", periods=13, freq="D")
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> t_first = compute_first_front_arrival_time(cin, flow, tedges, 500.0, sorption)
    >>> # Result is in days from tedges[0]
    >>> bool(np.isclose(t_first, 11.0))  # 1 day (offset) + 10 days (travel time)
    True

    See Also
    --------
    compute_first_fully_informed_bin_edge : Get first valid output bin edge
    """
    # Find first non-zero concentration
    nonzero_indices = np.where(cin > 0)[0]

    if len(nonzero_indices) == 0:
        # No concentration ever arrives
        return np.inf

    idx_first = nonzero_indices[0]
    c_first = cin[idx_first]

    # Compute retardation for this concentration
    r_first = sorption.retardation(c_first)

    # Target: cumulative flow volume needed to reach outlet
    target_volume = aquifer_pore_volume * r_first

    # Integrate piecewise constant flow starting from idx_first
    # tedges is assumed to be DatetimeIndex, convert all times to days
    cumulative_volume = 0.0

    for i in range(idx_first, len(flow)):
        # Convert time interval to days
        dt_days = (tedges[i + 1] - tedges[i]) / pd.Timedelta(days=1)
        volume_in_bin = flow[i] * dt_days

        if cumulative_volume + volume_in_bin >= target_volume:
            # First arrival occurs during this bin
            remaining_volume = target_volume - cumulative_volume
            dt_partial = remaining_volume / flow[i]

            # Return time in days from tedges[0]
            t_bin_start_days = (tedges[i] - tedges[0]) / pd.Timedelta(days=1)
            return t_bin_start_days + dt_partial

        cumulative_volume += volume_in_bin

    # Never reaches outlet with given flow history
    return np.inf


def compute_first_fully_informed_bin_edge(
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    sorption: FreundlichSorption | ConstantRetardation,
    output_tedges: pd.DatetimeIndex,
) -> float:
    """
    Compute left edge of first output bin that is fully informed.

    A bin [t_i, t_{i+1}] is "fully informed" if all water exiting during
    that interval originated from known inlet conditions (not unknown
    initial conditions). This occurs when t_i >= first front arrival time.

    This function is useful for:
    - Masking output bins during spin-up period
    - Determining which output times are valid for analysis
    - Plotting valid vs spin-up regions

    Rarefaction handling:
    - For n>1: Rarefaction tail (lower C, slower) arrives after head.
      Once the first wave (head) arrives, subsequent bins are informed.
    - For n<1: Rarefaction tail (higher C, slower) arrives after head.
      Same principle applies.

    In both cases, once the leading edge of the inlet-generated wave structure
    reaches the outlet, all subsequent output is determined by inlet history.

    Parameters
    ----------
    cin : numpy.ndarray
        Inlet concentration [mass/volume]. Length = len(tedges) - 1.
    flow : numpy.ndarray
        Flow rate [volume/time]. Length = len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Inlet time bin edges. Length = len(cin) + 1.
        Expected to be DatetimeIndex.
    aquifer_pore_volume : float
        Total pore volume [volume]. Must be positive.
    sorption : FreundlichSorption or ConstantRetardation
        Sorption model.
    output_tedges : pandas.DatetimeIndex
        Output time bin edges. These are the bins for which we want
        to determine the first fully-informed bin.
        Expected to be DatetimeIndex.

    Returns
    -------
    t_first_bin : float
        Left edge of first output bin that is fully informed, measured in
        days from output_tedges[0].
        Returns last edge in days if no bin is fully informed.
        Returns np.inf if output_tedges is empty.

    Notes
    -----
    This differs from compute_first_front_arrival_time in that it returns
    a BIN EDGE (from output_tedges), not the exact crossing time.

    Both functions return time in days, but measured from different reference points:
    - compute_first_front_arrival_time: days from tedges[0]
    - compute_first_fully_informed_bin_edge: days from output_tedges[0]

    For masking output arrays::

        import pandas as pd

        t_first_bin = compute_first_fully_informed_bin_edge(...)
        # Convert output_tedges to days from output_tedges[0]
        tedges_days = (output_tedges - output_tedges[0]) / pd.Timedelta(days=1)
        mask = tedges_days[:-1] >= t_first_bin
        cout_valid = cout[mask]

    Examples
    --------
    >>> import pandas as pd
    >>> # Exact arrival at ~11 days from tedges[0]
    >>> cin = np.array([0.0, 10.0, 10.0])
    >>> flow = np.array([100.0, 100.0, 100.0])
    >>> tedges = pd.date_range("2020-01-01", periods=4, freq="D")
    >>> output_tedges = pd.date_range("2020-01-01", periods=20, freq="D")
    >>> sorption = ConstantRetardation(retardation_factor=2.0)
    >>> t_bin = compute_first_fully_informed_bin_edge(
    ...     cin, flow, tedges, 500.0, sorption, output_tedges
    ... )
    >>> # Result is in days from output_tedges[0]
    >>> t_bin >= 11.0  # First bin edge >= arrival time
    True

    See Also
    --------
    compute_first_front_arrival_time : Get exact arrival time
    """
    if len(output_tedges) == 0:
        return np.inf

    # Compute exact arrival time (in days from tedges[0])
    t_arrival_days = compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

    if not np.isfinite(t_arrival_days):
        # No arrival, return last edge in days from output_tedges[0]
        return (output_tedges[-1] - output_tedges[0]) / pd.Timedelta(days=1)

    # Convert output_tedges to days from output_tedges[0]
    # Find first bin edge >= t_arrival_days
    # Note: t_arrival_days is measured from tedges[0], but output_tedges might have different start
    # So we need to adjust the reference point

    # Convert t_arrival from "days from tedges[0]" to "days from output_tedges[0]"
    t_arrival_abs = tedges[0] + pd.Timedelta(days=t_arrival_days)
    t_arrival_output_ref = (t_arrival_abs - output_tedges[0]) / pd.Timedelta(days=1)

    # Find first output bin edge >= t_arrival
    for t_edge in output_tedges:
        t_edge_days = (t_edge - output_tedges[0]) / pd.Timedelta(days=1)
        if t_edge_days >= t_arrival_output_ref:
            return t_edge_days

    # If no edge is >= t_arrival, return the last edge
    # (This means all bins are before arrival)
    return (output_tedges[-1] - output_tedges[0]) / pd.Timedelta(days=1)
