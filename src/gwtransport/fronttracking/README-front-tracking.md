# Front Tracking in `gwtransport`

This document describes the **exact analytical front-tracking implementation** in the `gwtransport` package and how to use the public API from user code.

The implementation follows the design in `FRONT_TRACKING_REBUILD_PLAN.md` and is
summarized in `src/gwtransport/fronttracking/KNOWLEDGE_BASE.md`. All
calculations are closed-form analytical expressions (no iterative solvers, no
time-stepping) and are validated by 174 tests.

---

## 1. What problem this solves

The front-tracking solver computes **extracted concentrations** along a
streamline (e.g., river-to-well path in bank filtration) for 1D advection with
nonlinear Freundlich sorption or constant retardation.

Conceptually, we solve a conservation law in pore-volume coordinates
\(v\) (volume) and time \(t\):

$$
\frac{\partial C_{\text{total}}}{\partial t} + \frac{\partial (\text{flow}\,C)}{\partial v} = 0,
$$

where:

- \(C\) is dissolved concentration,
- \(C\_{\text{total}}(C) = C + \text{(sorbed mass per unit fluid volume)}\),
- `flow` is the volumetric flow rate along the streamline (m³/day).

Instead of discretizing this equation in space and time, the implementation
**tracks waves exactly**:

- **Characteristics** in regions where concentration is smooth.
- **Shocks** where faster water overtakes slower water.
- **Rarefactions** where slower water enters behind faster water.

All wave speeds, interaction times, and concentrations are computed using exact
analytical formulas.

---

## 2. Design principles

The implementation is governed by the following principles (see
`FRONT_TRACKING_REBUILD_PLAN.md`):

1. **Exact analytical computation**  
   All calculations use closed-form analytical expressions. There are **no**
   iterative solvers, no tolerance-driven loops, and no numerical quadrature.

2. **Physical correctness**  
   Every wave satisfies conservation laws and entropy conditions exactly.
   Shocks use the Rankine–Hugoniot condition and must satisfy the Lax entropy
   inequality.

3. **Spin-up period handling**  
   The solver computes the first arrival time \(t*\text{first_arrival}\). All
   times before this are "spin-up" and may depend on unknown initial
   conditions in the aquifer. For \(t \ge t*\text{first_arrival}\), the
   solution is determined purely by the inlet history and sorption model.

4. **Detailed diagnostics**  
   The solver records all waves and all events (collisions, outlet crossings,
   etc.) so that the full history of the solution can be inspected.

5. **Multiple-streamline architecture**  
   The current implementation solves a **single** streamline with
   `aquifer_pore_volume` as the outlet coordinate, but the architecture is
   designed to extend to distributions of pore volumes.

6. **Consistent API**  
   The public interface uses standard `gwtransport` terminology:
   `cin`, `flow`, `tedges`, `aquifer_pore_volume`, etc.

7. **No capital variable names**  
   Implementation uses Python-style names (`c_left`, `v_outlet`,
   `t_first_arrival`) rather than capitalized symbols.

8. **Meaningful and explicit tests**  
   Tests do not contain control-flow or `try/except` logic and use very tight
   tolerances (rtol ~ 1e-14). Masked arrays are only used when the unmasked
   values suffice to prove the intended property.

For a detailed breakdown by module and phase, see
`src/gwtransport/fronttracking/KNOWLEDGE_BASE.md`.

---

## 3. Public API

The main entry points live in `src/gwtransport/advection.py`.

### 3.1 `infiltration_to_extraction_front_tracking`

```python
from gwtransport.advection import infiltration_to_extraction_front_tracking
```

```python
cout = infiltration_to_extraction_front_tracking(
      *,
      cin,
      flow,
      tedges,
      cout_tedges,
      aquifer_pore_volume,
      freundlich_k,
      freundlich_n,
      bulk_density,
      porosity,
      retardation_factor=None,
      max_iterations=10000,
)
```

**Purpose**

Compute **bin-averaged extraction concentration** at the outlet using exact
front tracking with nonlinear sorption (Freundlich) or constant retardation.

**Parameters (keywords only)**

- `cin`: array-like, infiltration concentration per inlet time bin.  
   Length = `len(tedges) - 1`. Must be non-negative.
- `flow`: array-like, flow rate (m³/day) per inlet time bin.  
   Length = `len(tedges) - 1`. Must be strictly positive.
- `tedges`: `pandas.DatetimeIndex` or array-like of time edges for `cin`/`flow`.
  Length = `len(cin) + 1`. Defines inlet bins.
- `cout_tedges`: `pandas.DatetimeIndex` (or array-like) of output time edges.
  Length determines the number of output bins. Can differ from `tedges`.
- `aquifer_pore_volume`: float, total pore volume (m³). Also the outlet
  coordinate `v_outlet`.
- `freundlich_k`: float, Freundlich coefficient [(m³/kg)^(1/n)], used when
  `retardation_factor` is `None`.
- `freundlich_n`: float, Freundlich exponent (>0, ≠1), used when
  `retardation_factor` is `None`.
- `bulk_density`: float, bulk density (kg/m³), used when
  `retardation_factor` is `None`.
- `porosity`: float in (0, 1), used when `retardation_factor` is `None`.
- `retardation_factor`: optional float ≥ 1.0. If provided, uses a constant
  retardation model instead of Freundlich sorption.
- `max_iterations`: safety limit on number of events in the solver loop.

**Returns**

- `cout`: `numpy.ndarray` of bin-averaged extraction concentrations
  corresponding to `cout_tedges` (length = `len(cout_tedges) - 1`).

**Physical behavior**

- Characteristics, shocks, and rarefactions are tracked exactly using
  `FrontTracker`.
- Outlet concentration over time is computed via exact evaluation of
  `concentration_at_point` and exact analytic integration in
  `compute_bin_averaged_concentration_exact`.
- Spin-up is handled internally via `compute_first_front_arrival_time`; values of
  `cout` before `t_first_arrival` are influenced by unknown initial conditions
  and should be interpreted accordingly.

### 3.2 `infiltration_to_extraction_front_tracking_detailed`

```python
from gwtransport.advection import (
      infiltration_to_extraction_front_tracking_detailed,
)

result = infiltration_to_extraction_front_tracking_detailed(
      cin=cin,
      flow=flow,
      tedges=tedges,
      cout_tedges=cout_tedges,
      aquifer_pore_volume=aquifer_pore_volume,
      freundlich_k=freundlich_k,
      freundlich_n=freundlich_n,
      bulk_density=bulk_density,
      porosity=porosity,
      retardation_factor=retardation_factor,
)

cout = result["cout"]
waves = result["waves"]
events = result["events"]
t_first_arrival = result["t_first_arrival"]
tracker_state = result["tracker_state"]
```

In addition to `cout`, the detailed function returns a diagnostics structure
containing:

- `waves`: all waves (characteristics, shocks, rarefactions) with their
  parameters and activity flags.
- `events`: a chronological log of events (wave collisions, outlet
  crossings, etc.).
- `t_first_arrival`: spin-up boundary (see §4).
- `n_events`, `n_shocks`, `n_rarefactions`: simple counts for quick sanity
  checks.
- `tracker_state`: full `FrontTrackerState` object.

This is the recommended entry point when you want to inspect wave dynamics,
plot V–t diagrams, or perform custom diagnostics.

---

## 4. Spin-up and first arrival time

The function `compute_first_front_arrival_time` analytically determines the first
time that non-zero concentration reaches the outlet, given `cin`, `flow`,
`tedges`, and `aquifer_pore_volume`.

- For a simple constant case, this reduces to:

  $$
  t_\text{first\_arrival} = \frac{\text{aquifer\_pore\_volume} \cdot R(C)}{\text{flow}}.
  $$

- For general, piecewise-constant `flow`, the algorithm integrates exactly
  until the cumulative volume equals `aquifer_pore_volume * R(c_first)`.

Interpretation:

- \(t < t\_\text{first_arrival}\): solution depends on unknown initial
  concentration in the domain (spin-up).
- \(t \ge t\_\text{first_arrival}\): solution depends only on inlet history
  (`cin`, `flow`, `tedges`) and sorption parameters.

The detailed API exposes `t_first_arrival` so you can filter out spin-up
bins in downstream analysis.

---

## 5. Internals (very short overview)

The internals are fully documented in
`src/gwtransport/fronttracking/KNOWLEDGE_BASE.md`. At a high level:

- `front_tracking_math.py` defines the sorption models and exact formulas for
  characteristic speeds, shock speeds, and rarefaction inversion/integration.
- `front_tracking_waves.py` defines `Wave`, `CharacteristicWave`,
  `ShockWave`, and `RarefactionWave`.
- `front_tracking_events.py` computes analytical intersection times between
  waves and detects outlet crossings.
- `front_tracking_inlet.py` creates waves at the inlet for each change in
  `cin`.
- `front_tracking_handlers.py` implements all wave interaction rules.
- `front_tracking_solver.py` implements the event-driven `FrontTracker`.
- `front_tracking_output.py` implements `concentration_at_point`,
  `compute_breakthrough_curve`, and
  `compute_bin_averaged_concentration_exact`.
- `front_tracking_plot.py` provides V–t diagram and breakthrough-curve
  plotting utilities.

You generally do **not** need to interact with these modules directly unless
you are extending the solver.

---

## 6. Example: pulse injection with Freundlich sorption

```python
import numpy as np
import pandas as pd

from gwtransport.advection import (
      infiltration_to_extraction_front_tracking,
)

# Inlet time bins (3 bins → 4 edges)
tedges = pd.date_range("2020-01-01", periods=4, freq="10D")

# Pulse injection: 0 → 10 → 0 mg/L
cin = np.array([0.0, 10.0, 0.0])

# Constant flow (m³/day)
flow = np.array([100.0, 100.0, 100.0])

# Output bin edges (can be different from tedges)
cout_tedges = pd.date_range("2020-01-01", periods=10, freq="5D")

cout = infiltration_to_extraction_front_tracking(
      cin=cin,
      flow=flow,
      tedges=tedges,
      cout_tedges=cout_tedges,
      aquifer_pore_volume=500.0,
      freundlich_k=0.01,
      freundlich_n=2.0,
      bulk_density=1500.0,
      porosity=0.3,
)

print(cout)
```

This computes a breakthrough curve at the outlet with bin-averaged
concentrations matching `cout_tedges`.

---

## 7. Tests and verification

Front-tracking functionality is covered by

- `tests/src/test_front_tracking_math_phase1.py` (math + sorption),
- `tests/src/test_front_tracking_waves_phase2.py`,
- `tests/src/test_front_tracking_events_phase3.py`,
- `tests/src/test_front_tracking_handlers_phase4.py`,
- `tests/src/test_front_tracking_solver_phase5.py`,
- `tests/src/test_front_tracking_output_phase6.py`,
- `tests/src/test_advection_api_phase7.py`,
- `tests/src/test_front_tracking_phase8.py` (integration),

for a total of 174 passing tests.

To run all front-tracking tests:

```bash
PYTHONPATH=src uv run pytest \
   tests/src/test_front_tracking*.py \
   tests/src/test_advection_api_phase7.py -v
```

These tests verify:

- Exactness of characteristic, shock, and rarefaction formulas.
- Entropy conditions for shocks.
- Analytical correctness of rarefactions (self-similar solution).
- Linear (constant retardation) and nonlinear (Freundlich) cases.
- Correct wave-type creation at the inlet.
- Analytical breakthrough curves and bin-averaged concentrations.

---

## 8. Where to go next

- For implementation details and design rationale, read
  `src/gwtransport/fronttracking/KNOWLEDGE_BASE.md`.
- For a worked example with plots and mass balance verification, open
  `examples/08_Front_Tracking_Exact_Solution.ipynb`.
- For integration in your own workflows, use the two public functions in
  `gwtransport.advection` documented in §3.
