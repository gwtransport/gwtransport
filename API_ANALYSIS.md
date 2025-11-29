# gwtransport API Analysis

**Version:** 0.21.0
**Date:** 2025-11-28
**Python:** >=3.11

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Concepts](#core-concepts)
3. [API Design Patterns](#api-design-patterns)
4. [Module-by-Module API](#module-by-module-api)
5. [Identified Inconsistencies](#identified-inconsistencies)
6. [Conventions and Customs](#conventions-and-customs)
7. [Recommendations](#recommendations)

---

## Executive Summary

`gwtransport` is a Python package for modeling 1D groundwater transport of solutes and temperature through aquifer systems. The API follows a physics-based design where:

- **Time series** are represented as discrete bins with edge timestamps (`tedges`)
- **Concentrations/flows** are values between edges (length = `len(tedges) - 1`)
- **Transport** is modeled as forward (infiltration→extraction) and reverse (extraction→infiltration)
- **Heterogeneity** is captured via gamma-distributed pore volumes or explicit distributions
- **Retardation** accounts for sorption (linear or nonlinear via Freundlich isotherm)

### Key Strengths

1. **Consistent time series representation**: Edge-based binning throughout
2. **Symmetric forward/reverse operations**: Convolution/deconvolution pairs
3. **Flexible parameterization**: Multiple ways to specify distributions (alpha/beta vs mean/std)
4. **Comprehensive documentation**: Extensive NumPy-style docstrings
5. **Physics-first design**: Clear mapping to physical processes

### Major Inconsistencies Found

1. **Parameter naming**: `aquifer_pore_volume` (singular) vs `aquifer_pore_volumes` (plural)
2. **Parameter ordering**: Inconsistent position of `retardation_factor` across functions
3. **Missing deconvolution**: Several reverse operations not implemented
4. **Mixed validation**: Some functions validate NaN, others don't
5. **Direction naming**: Inconsistent between `"extraction_to_infiltration"` and `"infiltration_to_extraction"`

---

## Core Concepts

### 1. Time Series Representation

**Convention**: Time bins defined by edges (`tedges`), values live between edges

```python
tedges = pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03'])
# Creates 2 bins: [2020-01-01, 2020-01-02) and [2020-01-02, 2020-01-03)
values = np.array([10.0, 15.0])  # len(values) = len(tedges) - 1
```

**Rationale**: Enables precise temporal integration and alignment

### 2. Transport Directions

| Direction | Meaning | Operation Type |
|-----------|---------|----------------|
| `infiltration_to_extraction` | Water enters → Water exits | Forward (convolution) |
| `extraction_to_infiltration` | Water exits → Water enters | Reverse (deconvolution) |

**Physical Interpretation**:
- **Forward**: Given infiltration history, predict extraction
- **Reverse**: Given extraction measurements, infer infiltration history

### 3. Aquifer Heterogeneity Models

#### Gamma Distribution (Parametric)
```python
# Two parameterizations:
# 1. Shape/scale parameters
gamma_infiltration_to_extraction(alpha=10.0, beta=50.0, ...)
# 2. Physical parameters
gamma_infiltration_to_extraction(mean=500.0, std=158.1, ...)
```

#### Explicit Distribution (Non-parametric)
```python
# Direct pore volume array
infiltration_to_extraction(aquifer_pore_volumes=[400, 500, 600], ...)
```

### 4. Retardation

**Linear (Constant Factor)**:
```python
retardation_factor = 2.0  # Compound moves 2x slower than water
```

**Nonlinear (Freundlich Isotherm)**:
```python
R = freundlich_retardation(
    concentration=cin,
    freundlich_k=0.02,
    freundlich_n=0.75,
    bulk_density=1600.0,
    porosity=0.35
)
```

**Physical Meaning**:
- R = 1.0: Conservative tracer (no interaction)
- R > 1.0: Sorption/thermal delay
- Temperature typically R ≈ 2.0

---

## API Design Patterns

### Pattern 1: Keyword-Only Arguments

**All public functions** use keyword-only arguments (enforced by `*` in signature):

```python
def infiltration_to_extraction(
    *,  # Forces keyword arguments
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    ...
) -> npt.NDArray[np.floating]:
```

**Rationale**: Prevents positional argument errors, improves readability

### Pattern 2: Symmetric Pairs

Many operations come in forward/reverse pairs:

| Forward (Convolution) | Reverse (Deconvolution) |
|-----------------------|-------------------------|
| `infiltration_to_extraction` | `extraction_to_infiltration` |
| `gamma_infiltration_to_extraction` | `gamma_extraction_to_infiltration` |
| `infiltration_to_extraction_series` | `extraction_to_infiltration_series` |
| `deposition_to_extraction` | `extraction_to_deposition` |
| `convolve_diffusion` | `deconvolve_diffusion` ⚠️ NOT IMPLEMENTED |

**Asymmetry**: Deconvolution operations are harder (ill-posed inverse problems)

### Pattern 3: Hierarchical Complexity

Functions organized by increasing sophistication:

```
Simple:  _series()           # Single pore volume, time shift only
Medium:  _to_extraction()    # Arbitrary distribution, convolution
Complex: gamma_*()           # Gamma-distributed, discretized
Expert:  front_tracking_*()  # Exact analytical solution, nonlinear
```

### Pattern 4: Optional Output Edges

**Flexible time resolution**:
```python
# Input and output on different time grids
gamma_infiltration_to_extraction(
    tedges=daily_tedges,        # Input grid
    cout_tedges=hourly_tedges,  # Output grid (different!)
    ...
)
```

**Exception**: `_series()` functions don't support custom output edges

### Pattern 5: Multiple Parameterizations

**Gamma distribution** accepts either:
```python
# Statistical parameters
bins(alpha=10.0, beta=50.0)
# Physical parameters
bins(mean=500.0, std=158.1)
```

**Validation**: Exactly one pair must be provided (XOR logic)

---

## Module-by-Module API

### advection.py

**Core transport functions for advective (flow-driven) compound movement**

#### Function Taxonomy

```
infiltration_to_extraction_series()      # Single pore volume, simple
extraction_to_infiltration_series()      # Reverse of above

infiltration_to_extraction()             # Arbitrary distribution
extraction_to_infiltration()             # Reverse of above

gamma_infiltration_to_extraction()       # Gamma-distributed
gamma_extraction_to_infiltration()       # Reverse of above

infiltration_to_extraction_front_tracking()          # Exact nonlinear
infiltration_to_extraction_front_tracking_detailed() # + diagnostics
```

#### Key Signatures

```python
# Simplest: Time shift only, no convolution
def infiltration_to_extraction_series(
    *,
    flow: npt.ArrayLike,              # [m3/day]
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,       # ⚠️ SINGULAR
    retardation_factor: float = 1.0,
) -> pd.DatetimeIndex:
    """Returns shifted tedges, cout values = cin values"""

# Standard: Arbitrary distribution
def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,               # Concentration [ng/m3]
    flow: npt.ArrayLike,              # [m3/day]
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,    # Different grid allowed!
    aquifer_pore_volumes: npt.ArrayLike,  # ⚠️ PLURAL (distribution)
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:

# Gamma-distributed
def gamma_infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    alpha: float | None = None,       # XOR with mean/std
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int = 100,                # Discretization
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:

# Front tracking (exact analytical)
def infiltration_to_extraction_front_tracking(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.ArrayLike,  # ⚠️ PLURAL
    # Freundlich parameters (XOR with retardation_factor)
    freundlich_k: float | None = None,
    freundlich_n: float | None = None,
    bulk_density: float | None = None,
    porosity: float | None = None,
    retardation_factor: float | None = None,  # ⚠️ Optional, not default!
    max_iterations: int = 10000,
) -> npt.NDArray[np.floating]:
```

#### Inconsistencies in advection.py

| Issue | Examples | Severity |
|-------|----------|----------|
| Singular/plural naming | `aquifer_pore_volume` (series) vs `aquifer_pore_volumes` (distribution) | High |
| Optional vs required retardation | `retardation_factor=1.0` vs `retardation_factor: float \| None = None` | Medium |
| Parameter order | `retardation_factor` position varies | Low |

---

### gamma.py

**Gamma distribution utilities for pore volume heterogeneity**

#### Functions

```python
def parse_parameters(
    *,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[float, float]:
    """Validate and convert to (alpha, beta). XOR constraint."""

def mean_std_to_alpha_beta(
    *, mean: float, std: float
) -> tuple[float, float]:
    """alpha = mean²/std², beta = std²/mean"""

def alpha_beta_to_mean_std(
    *, alpha: float, beta: float
) -> tuple[float, float]:
    """mean = alpha*beta, std = sqrt(alpha)*beta"""

def bins(
    *,
    alpha: float | None = None,
    beta: float | None = None,
    mean: float | None = None,
    std: float | None = None,
    n_bins: int | None = None,              # XOR with quantile_edges
    quantile_edges: np.ndarray | None = None,
) -> dict[str, npt.NDArray[np.floating]]:
    """
    Returns dict with keys:
    - 'lower_bound': array[n_bins]
    - 'upper_bound': array[n_bins]
    - 'edges': array[n_bins + 1]
    - 'expected_values': array[n_bins]  # Used in transport
    - 'probability_mass': array[n_bins]
    """

def bin_masses(
    *, alpha: float, beta: float, bin_edges: npt.ArrayLike
) -> npt.NDArray[np.floating]:
    """Probability mass between edges using incomplete gamma"""
```

#### Design Pattern: XOR Parameterization

```python
# Valid
bins(alpha=10, beta=50, n_bins=100)
bins(mean=500, std=158, n_bins=100)
bins(alpha=10, beta=50, quantile_edges=[0, 0.25, 0.5, 0.75, 1.0])

# Invalid (raises ValueError)
bins(alpha=10, mean=500, n_bins=100)  # Both pairs provided
bins(n_bins=100)                      # No parameters
bins(alpha=10, beta=50)               # No binning specified
```

---

### residence_time.py

**Compute residence times accounting for flow variability and retardation**

#### Functions

```python
def residence_time(
    *,
    flow: npt.ArrayLike | None = None,
    flow_tedges: pd.DatetimeIndex | np.ndarray | None = None,
    aquifer_pore_volume: npt.ArrayLike | None = None,  # ⚠️ Can be array!
    index: pd.DatetimeIndex | np.ndarray | None = None,
    retardation_factor: float = 1.0,
    direction: str = "extraction_to_infiltration",
) -> npt.NDArray[np.floating]:
    """
    Returns 2D array: shape (n_pore_volumes, n_time_points)
    NaN where insufficient flow history
    """

def residence_time_mean(
    *,
    flow: npt.ArrayLike,
    flow_tedges: pd.DatetimeIndex | np.ndarray,
    tedges_out: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volume: npt.ArrayLike,
    direction: str = "extraction_to_infiltration",
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Mean residence time over intervals"""

def fraction_explained(
    *,
    rt: npt.NDArray[np.floating] | None = None,
    flow: npt.ArrayLike | None = None,
    flow_tedges: pd.DatetimeIndex | np.ndarray | None = None,
    aquifer_pore_volume: npt.ArrayLike | None = None,
    index: pd.DatetimeIndex | np.ndarray | None = None,
    retardation_factor: float = 1.0,
    direction: str = "extraction_to_infiltration",
) -> npt.NDArray[np.floating]:
    """
    Returns fraction in [0, 1]
    1.0 = all pore volumes have valid residence time
    0.0 = none have valid residence time (spin-up period)
    """

def freundlich_retardation(
    *,
    concentration: npt.ArrayLike,
    freundlich_k: float,
    freundlich_n: float,
    bulk_density: float,
    porosity: float,
) -> npt.NDArray[np.floating]:
    """
    Concentration-dependent retardation
    R = 1 + (ρ_b/θ) * K_f * n * C^(n-1)
    """
```

#### Inconsistencies in residence_time.py

| Issue | Details |
|-------|---------|
| Parameter naming | `aquifer_pore_volume` accepts array but named singular |
| Parameter order | `retardation_factor` before `direction` in some, after in others |
| Optional parameters | Many parameters `| None` but marked as required in docstrings |

---

### utils.py

**General utilities for time series, interpolation, and numerical operations**

#### Time Series Functions

```python
def linear_interpolate(
    *,
    x_ref: npt.ArrayLike,
    y_ref: npt.ArrayLike,
    x_query: npt.ArrayLike,
    left: float | None = None,    # None = clamp, float = value
    right: float | None = None,
) -> npt.NDArray[np.floating]:
    """Auto-sorts unsorted reference data"""

def linear_average(
    *,
    x_data: npt.ArrayLike,
    y_data: npt.ArrayLike,
    x_edges: npt.ArrayLike,       # Can be 2D for batch processing!
    extrapolate_method: str = "nan",  # 'nan', 'outer', 'raise'
) -> npt.NDArray[np.float64]:
    """Average piecewise linear function over bins"""

def interp_series(
    *,
    series: pd.Series,
    index_new: pd.DatetimeIndex,
    **interp1d_kwargs: object
) -> pd.Series:
    """Pandas-aware interpolation"""

def compute_time_edges(
    *,
    tedges: pd.DatetimeIndex | None,
    tstart: pd.DatetimeIndex | None,
    tend: pd.DatetimeIndex | None,
    number_of_bins: int,
) -> pd.DatetimeIndex:
    """Construct tedges from explicit edges, starts, or ends"""
```

#### Bin Overlap Functions

```python
def partial_isin(
    *,
    bin_edges_in: npt.ArrayLike,
    bin_edges_out: npt.ArrayLike,
) -> npt.NDArray[np.floating]:
    """
    Matrix[i,j] = fraction of input bin i overlapping output bin j
    Returns dense matrix (n_bins_in, n_bins_out)
    """

def time_bin_overlap(
    *,
    tedges: npt.ArrayLike,
    bin_tedges: list[tuple],
) -> npt.NDArray[np.floating]:
    """Similar to partial_isin but for time ranges"""

def combine_bin_series(
    *,
    a: npt.ArrayLike,
    a_edges: npt.ArrayLike,
    b: npt.ArrayLike,
    b_edges: npt.ArrayLike,
    extrapolation: str | float = 0.0,  # 'nearest' or fill value
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map two binned series onto common unique edges"""
```

#### Underdetermined System Solver

```python
def solve_underdetermined_system(
    *,
    coefficient_matrix: npt.ArrayLike,
    rhs_vector: npt.ArrayLike,
    nullspace_objective: str | Callable = "squared_differences",
    # 'squared_differences', 'summed_differences', or custom callable
    optimization_method: str = "BFGS",
) -> npt.NDArray[np.floating]:
    """
    Solve Ax = b where m < n (underdetermined)
    Uses least-squares + nullspace regularization
    Handles NaN by row exclusion
    """
```

#### External Data

```python
def get_soil_temperature(
    *,
    station_number: int = 260,  # KNMI stations: 260, 273, 286, 323
    interpolate_missing_values: bool = True,
) -> pd.DataFrame:
    """
    Download and cache soil temperature from KNMI
    Daily cache prevents redundant downloads
    """
```

#### Inconsistencies in utils.py

| Issue | Details |
|-------|---------|
| 2D support | `linear_average` supports 2D `x_edges`, others don't |
| Extrapolation | Different methods across functions ('nan', 'outer', None, float) |
| Return types | Sometimes ndarray, sometimes tuple, sometimes DataFrame |

---

### deposition.py

**Model compound deposition during aquifer transport**

#### Functions

```python
def deposition_to_extraction(
    *,
    dep: npt.ArrayLike,           # Deposition rate [ng/m2/day]
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volume: float,   # ⚠️ SINGULAR (single volume only)
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Forward: deposition → concentration change"""

def extraction_to_deposition(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex | np.ndarray,
    cout: npt.ArrayLike,          # Concentration [ng/m3]
    cout_tedges: pd.DatetimeIndex | np.ndarray,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
    nullspace_objective: str = "squared_differences",
) -> npt.NDArray[np.floating]:
    """
    Reverse: concentration → deposition (inverse problem)
    Uses underdetermined system solver
    Handles NaN in cout
    """

def compute_deposition_weights(
    *,
    flow_values: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    porosity: float,
    thickness: float,
    retardation_factor: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Internal: weight matrix for deposition convolution"""

def spinup_duration(
    *,
    flow: np.ndarray,
    flow_tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    retardation_factor: float,
) -> float:
    """Time needed for system to become fully informed [days]"""
```

#### Deposition Physics

**Weight matrix relates deposition rate to concentration change**:
```
dC/dt = (contact_area / extracted_volume) * deposition_rate
```

Where contact area depends on:
- Streamline geometry (porosity, thickness)
- Residence time distribution
- Flow-weighted integration

---

### logremoval.py

**Pathogen log removal efficiency for water treatment**

#### Core Functions

```python
def residence_time_to_log_removal(
    *,
    residence_times: npt.ArrayLike,
    log_removal_rate: float,
) -> npt.NDArray[np.floating]:
    """
    LR = log_removal_rate * log10(residence_time)

    Examples:
    - LR=1 → 90% removal
    - LR=2 → 99% removal
    - LR=3 → 99.9% removal
    """

def parallel_mean(
    *,
    log_removals: npt.ArrayLike,
    flow_fractions: npt.ArrayLike | None = None,  # None = equal
    axis: int | None = None,
) -> npt.NDArray[np.floating]:
    """
    Weighted average for parallel treatment paths
    LR_total = -log10(sum(F_i * 10^(-LR_i)))

    Note: For series systems, sum log removals directly
    """
```

#### Gamma Distribution Functions

```python
def gamma_pdf(
    *, r: npt.ArrayLike, rt_alpha: float, rt_beta: float,
    log_removal_rate: float
) -> npt.NDArray[np.floating]:
    """PDF of log removal for gamma-distributed residence time"""

def gamma_cdf(
    *, r: npt.ArrayLike, rt_alpha: float, rt_beta: float,
    log_removal_rate: float
) -> npt.NDArray[np.floating]:
    """CDF of log removal"""

def gamma_mean(
    *, rt_alpha: float, rt_beta: float, log_removal_rate: float
) -> float:
    """Expected log removal: E[R] = (c/ln10) * (ψ(α) + ln(β))"""

def gamma_find_flow_for_target_mean(
    *,
    target_mean: float,
    apv_alpha: float,
    apv_beta: float,
    log_removal_rate: float,
) -> float:
    """
    Inverse design: Find flow rate for target log removal
    Q = β * exp(ψ(α) - (ln(10)*target)/c)
    """
```

#### Naming Convention

| Parameter | Prefix | Meaning |
|-----------|--------|---------|
| `rt_alpha`, `rt_beta` | `rt_` | Residence time distribution |
| `apv_alpha`, `apv_beta` | `apv_` | Aquifer pore volume distribution |

**Relationship**: `gamma(rt_α, rt_β) = gamma(apv_α, apv_β/flow)`

---

### diffusion.py

**Diffusive/dispersive transport corrections**

#### Forward Operation

```python
def infiltration_to_extraction(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    diffusivity: float = 0.1,        # [m2/day]
    retardation_factor: float = 1.0,
    aquifer_length: float = 80.0,
    porosity: float = 0.35,
) -> npt.NDArray[np.floating]:
    """
    Apply Gaussian smoothing with position-dependent sigma
    Typical diffusivity for heat in saturated sand:
    - Fine sand/silt: 0.007-0.01 m²/day
    - Typical sand: 0.01-0.05 m²/day
    - Coarse sand/gravel: 0.05-0.10 m²/day
    """
```

#### Reverse Operation

```python
def extraction_to_infiltration(
    *,
    cout: npt.ArrayLike,
    flow: npt.ArrayLike,
    aquifer_pore_volume: float,
    diffusivity: float = 0.1,
    retardation_factor: float = 1.0,
    aquifer_length: float = 80.0,
    porosity: float = 0.35,
) -> npt.NDArray[np.floating]:
    """⚠️ NOT IMPLEMENTED - raises NotImplementedError

    Reason: Deconvolution is ill-posed, requires regularization
    """
```

#### Helper Functions

```python
def compute_sigma_array(
    *,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    diffusivity: float = 0.1,
    retardation_factor: float = 1.0,
    aquifer_length: float = 80.0,
    porosity: float = 0.35,
) -> npt.NDArray[np.floating]:
    """
    σ = sqrt(2 * D * t_residence) / dx
    Clipped to [0, 100] for stability
    """

def convolve_diffusion(
    *,
    input_signal: npt.ArrayLike,
    sigma_array: npt.ArrayLike,
    truncate: float = 4.0,
) -> npt.NDArray[np.floating]:
    """
    Variable-sigma Gaussian filter using sparse matrix
    Boundary: mode='nearest' (repeat edge values)
    """

def deconvolve_diffusion(
    *,
    output_signal: npt.ArrayLike,
    sigma_array: npt.ArrayLike,
    truncate: float = 4.0,
) -> npt.NDArray[np.floating]:
    """⚠️ NOT IMPLEMENTED"""
```

#### Inconsistencies

- Missing reverse operations (deconvolution)
- `infiltration_to_extraction` in both `diffusion.py` and `advection.py` (name collision!)

---

### surfacearea.py

**Geometric utilities for streamline analysis**

```python
def compute_average_heights(
    *,
    x_edges: npt.ArrayLike,         # 1D array (n_x,)
    y_edges: npt.ArrayLike,         # 2D array (n_y, n_x)
    y_lower: float,                 # Clipping bounds
    y_upper: float,
) -> npt.NDArray[np.floating]:
    """
    Compute area/width of clipped trapezoids

    Returns: shape (n_y-1, n_x-1)

    Used for direct pore volume estimation from streamlines
    Alternative to gamma distribution approximation
    """
```

**Algorithm**: Shoelace formula for quadrilaterals with edge intersection corrections

---

### advection_utils.py

**Internal helpers (not part of public API)**

```python
def _infiltration_to_extraction_weights(
    *,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cin: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.floating]:
    """
    Compute normalized weight matrix for convolution
    Optimizations:
    - Temporal overlap clipping
    - Flow-weighted averaging
    """

def _extraction_to_infiltration_weights(
    *,
    tedges: pd.DatetimeIndex,
    cin_tedges: pd.DatetimeIndex,
    aquifer_pore_volumes: npt.NDArray[np.floating],
    cout: npt.NDArray[np.floating],
    flow: npt.NDArray[np.floating],
    retardation_factor: float,
) -> npt.NDArray[np.floating]:
    """Symmetric to forward weights (transposed temporal overlap)"""
```

**Note**: Underscore prefix indicates internal use only

---

## Identified Inconsistencies

### 1. Parameter Naming: Singular vs Plural

**Issue**: `aquifer_pore_volume` sometimes accepts arrays but uses singular name

| Function | Parameter | Actual Type | Correct Name? |
|----------|-----------|-------------|---------------|
| `infiltration_to_extraction_series()` | `aquifer_pore_volume` | `float` | ✅ Correct |
| `infiltration_to_extraction()` | `aquifer_pore_volumes` | `ArrayLike` | ✅ Correct |
| `residence_time()` | `aquifer_pore_volumes` | `ArrayLike` | ✅ Correct (FIXED) |
| `deposition_to_extraction()` | `aquifer_pore_volume` | `float` | ✅ Correct |
| `front_tracking()` | `aquifer_pore_volumes` | `ArrayLike` | ✅ Correct |

**Status**: ✅ **FIXED** - `residence_time()` parameter renamed to `aquifer_pore_volumes`

### 2. Parameter Order Inconsistency

**Issue**: `retardation_factor` position varies across functions

```python
# Pattern A: retardation_factor before direction
residence_time(..., retardation_factor=1.0, direction="...")

# Pattern B: direction before retardation_factor
residence_time_mean(..., direction="...", retardation_factor=1.0)

# Pattern C: retardation_factor as last parameter
infiltration_to_extraction(..., retardation_factor=1.0)
```

**Recommendation**: Standardize to alphabetical or logical grouping

### 3. Optional vs Required Parameters

**Issue**: Type hints show `| None` but docstrings say "required"

```python
def residence_time(
    *,
    flow: npt.ArrayLike | None = None,  # Type says optional
    ...
):
    """
    Parameters
    ----------
    flow : array-like
        Flow rate ... (required)  # Docstring says required
    """
```

**Recommendation**: Remove `| None` if truly required, or add validation logic

### 4. Direction String Values

**Issue**: Two similar but inverted strings

```python
direction = "extraction_to_infiltration"  # Backwards in time
direction = "infiltration_to_extraction"  # Forwards in time
```

**Problems**:
- Easy to mix up
- Typos not caught until runtime
- No IDE autocomplete

**Recommendation**: Use Enum or Literal type

```python
from typing import Literal

Direction = Literal["extraction_to_infiltration", "infiltration_to_extraction"]

def residence_time(..., direction: Direction = "extraction_to_infiltration"):
```

### 5. NaN Validation Inconsistency

**Issue**: Some functions validate NaN inputs, others don't

| Function | Validates NaN? |
|----------|----------------|
| `infiltration_to_extraction()` | ✅ Yes (raises ValueError) |
| `extraction_to_deposition()` | ✅ Yes for flow, ⚠️ allows NaN in cout |
| `deposition_to_extraction()` | ✅ Yes (raises ValueError) |
| `residence_time()` | ❌ No validation |
| `linear_average()` | ⚠️ Filters out NaN silently |

**Recommendation**: Document NaN handling strategy consistently

### 6. Missing Deconvolution Operations

**Issue**: Several reverse operations not implemented

| Forward | Reverse | Status |
|---------|---------|--------|
| `convolve_diffusion()` | `deconvolve_diffusion()` | ❌ NotImplementedError |
| `diffusion.infiltration_to_extraction()` | `diffusion.extraction_to_infiltration()` | ❌ NotImplementedError |

**Reason**: Deconvolution is ill-posed (requires regularization)

**Recommendation**: Document why missing, or implement with regularization

### 7. Function Name Collision

**Issue**: `infiltration_to_extraction()` exists in TWO modules

- `advection.infiltration_to_extraction()` - advective transport
- `diffusion.infiltration_to_extraction()` - diffusive transport

**Problem**: Confusing imports, unclear which to use

```python
# Which one?
from gwtransport.advection import infiltration_to_extraction
from gwtransport.diffusion import infiltration_to_extraction
```

**Recommendation**: Rename diffusion function to `apply_diffusion()` or `diffusion_correction()`

### 8. Inconsistent Return Types

**Issue**: Similar operations return different types

```python
infiltration_to_extraction_series() -> pd.DatetimeIndex  # Time edges
infiltration_to_extraction() -> np.ndarray               # Concentrations
```

**This is actually correct** (different purposes), but could confuse users

### 9. Default Value Inconsistencies

**Issue**: `n_bins` default varies

```python
gamma_infiltration_to_extraction(..., n_bins: int = 100)
bins(..., n_bins: int | None = None)  # No default
```

**Recommendation**: Standardize default to 100 across all functions

### 10. Docstring Parameter Descriptions

**Issue**: Some parameters lack units, others inconsistent

```python
# Good
flow : array-like
    Flow rate of water in the aquifer [m3/day].

# Missing units
diffusivity : float, optional
    diffusivity of the compound. Default is 0.1.  # ⚠️ What units?
```

**Recommendation**: All physical quantities should include units

---

## Conventions and Customs

### 1. Time Handling

**Universal Pattern**: Edge-based time bins

```python
tedges = pd.DatetimeIndex([...])  # n+1 edges
values = np.array([...])           # n values
```

**Alignment**:
- Values represent **average** over interval
- Interval: `[tedges[i], tedges[i+1])`
- Half-open intervals (includes left, excludes right)

### 2. Flow Direction Semantics

**Physical Interpretation**:

```
Infiltration → Aquifer → Extraction
   (cin)        (pore)      (cout)
```

- **Forward**: cin + flow → cout (prediction)
- **Reverse**: cout + flow → cin (inversion)

### 3. Retardation Factor Convention

**Values**:
- `R = 1.0`: Conservative tracer (water speed)
- `R > 1.0`: Retarded (slower than water)
- `R < 1.0`: Invalid (physically impossible)

**Typical Values**:
- Temperature: `R ≈ 2.0` (thermal retardation in saturated sand)
- Conservative solute: `R = 1.0` (chloride, bromide)
- Sorbing compound: `R = 1 + (ρ_b/θ) * K_d`

### 4. Concentration Units

**Not enforced**, but documentation suggests:
- Concentrations: `[ng/m³]` or `[mg/L]`
- Temperature: `[°C]`
- Deposition: `[ng/m²/day]`

**Important**: Units must be consistent within calculation

### 5. Gamma Distribution Discretization

**Default Strategy**: Equal-probability bins

```python
bins(alpha=10, beta=50, n_bins=100)
# Creates 100 bins, each with p = 0.01
```

**Custom Strategy**: Explicit quantiles

```python
bins(alpha=10, beta=50, quantile_edges=[0, 0.25, 0.5, 0.75, 1.0])
# Creates 4 bins with custom probabilities
```

### 6. NaN Semantics

**Multiple meanings**:

1. **Insufficient history**: Residence time calculation before spin-up
2. **Invalid data**: Missing measurements (explicitly allowed in some functions)
3. **Out of bounds**: Extrapolation beyond valid range

**No universal convention** - check docstrings

### 7. Front Tracking vs Convolution

**When to use each**:

| Method | Use Case | Limitations |
|--------|----------|-------------|
| Convolution (`infiltration_to_extraction`) | Linear sorption, approximate | Numerical dispersion |
| Front tracking (`*_front_tracking`) | Nonlinear sorption, exact | Single pore volume, slower |

### 8. Validation Strategy

**Implemented checks**:
- Array length matching (`len(tedges) = len(values) + 1`)
- Positive flows (sometimes)
- XOR parameter constraints (gamma parameters)
- NaN validation (inconsistent)

**Missing checks**:
- Unit consistency
- Physical range validation (e.g., porosity ∈ (0,1))
- Monotonic time edge validation

### 9. Type Hints

**Consistent patterns**:
```python
npt.ArrayLike          # Input (flexible)
npt.NDArray[np.floating]  # Output (specific)
pd.DatetimeIndex | np.ndarray  # Time inputs
float | None           # Optional parameters
```

**Good practice**: Accept flexible inputs, return specific types

### 10. Error Messages

**Pattern**: Descriptive messages with `msg` variable

```python
if condition_violated:
    msg = "Detailed explanation of what went wrong"
    raise ValueError(msg)
```

**Recommendation**: All error messages follow this pattern

---

## Recommendations

### High Priority

1. **~~Fix `residence_time()` parameter name~~** ✅ **COMPLETED**
   ```python
   # ✅ DONE - Parameter renamed from aquifer_pore_volume to aquifer_pore_volumes
   residence_time(..., aquifer_pore_volumes=array)
   ```

2. **Add Direction Literal type**
   ```python
   from typing import Literal
   Direction = Literal["extraction_to_infiltration", "infiltration_to_extraction"]
   ```

3. **Rename diffusion function to avoid collision**
   ```python
   # Change
   diffusion.infiltration_to_extraction()
   # To
   diffusion.apply_diffusion_correction()
   ```

4. **Standardize `n_bins` default**
   ```python
   # All functions should use
   n_bins: int = 100
   ```

5. **Document NaN handling strategy**
   - Add section to each docstring
   - Standardize behavior across modules

### Medium Priority

6. **Standardize parameter order**
   - Group by logical category
   - Alphabetize within category
   - Document ordering convention

7. **Add units to all docstrings**
   - Physical quantities need `[units]`
   - Check consistency across package

8. **Implement missing deconvolution or document why not**
   - Either implement with regularization
   - Or clearly document limitations

9. **Add validation for physical ranges**
   ```python
   if not 0 < porosity < 1:
       raise ValueError("Porosity must be in (0, 1)")
   ```

10. **Fix Optional type hints**
    - Remove `| None` if required
    - Add validation if truly optional

### Low Priority

11. **Standardize error message format**
12. **Add input shape validation helpers**
13. **Consider returning NamedTuples instead of dicts**
14. **Add more type aliases for readability**
15. **Consider deprecation warnings for inconsistent names**

---

## Conclusion

The `gwtransport` API demonstrates **strong physics-based design** with **consistent time series handling** and **comprehensive documentation**. The main inconsistencies are:

1. **Naming** (singular/plural, function collisions)
2. **Parameter ordering** (minor)
3. **Validation** (incomplete)
4. **Missing features** (deconvolution)

These issues are **cosmetic** rather than architectural. The core design is **solid and well-thought-out**. Addressing the High Priority recommendations would significantly improve user experience with minimal code changes.

**Overall Assessment**: 8.5/10 - Excellent physics modeling with room for API polish.
