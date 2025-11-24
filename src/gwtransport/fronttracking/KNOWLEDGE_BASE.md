# Front-tracking Knowledge Base

This file captures the current design and implementation of the exact analytical front-tracking solver in `gwtransport`. It is intended as persistent context for future assistant sessions and for human maintainers.

---

## 1. High-level goal

- Implementation lives in:
  - `src/gwtransport/advection.py`
  - `src/gwtransport/fronttracking/math.py`
  - `src/gwtransport/fronttracking/waves.py`
  - `src/gwtransport/fronttracking/events.py`
  - `src/gwtransport/fronttracking/inlet.py`
  - `src/gwtransport/fronttracking/handlers.py`
  - `src/gwtransport/fronttracking/solver.py`
  - `src/gwtransport/fronttracking/output.py`
  - `src/gwtransport/fronttracking/plot.py`
- Goal: exact analytical, event-driven front tracking for 1D advection with sorption:
  - Nonlinear Freundlich sorption and linear constant-retardation case
  - Machine-precision, no numerical tolerances, no iterative solvers
  - Physically correct shocks/rarefactions, strict entropy and conservation

All 9 phases in `FRONT_TRACKING_REBUILD_PLAN.md` are implemented and passing tests (174 tests total).

---

## 2. Modules and responsibilities

### 2.1 `math.py` (Phase 1 – core math)

**Classes and functions**

- `FreundlichSorption`
  - Parameters: `k_f`, `n`, `bulk_density`, `porosity`
  - Methods:
    - `retardation(c)`:
      $R(C) = 1 + \frac{\rho_b k_f}{n_{\text{por}} n} C^{(1/n) - 1}$
    - `total_concentration(c)`:
      $C_\text{total} = C + \frac{\rho_b}{n_{\text{por}}} k_f C^{1/n}$
    - `concentration_from_retardation(r)`: exact analytic inverse of `retardation`.
    - `shock_velocity(c_left, c_right, flow)`: Rankine–Hugoniot,
      $$s = \frac{\text{flow}(c_\text{right} - c_\text{left})}{C_\text{total}(c_\text{right}) - C_\text{total}(c_\text{left})}.$$
    - `check_entropy_condition(c_left, c_right, shock_vel, flow)`: Lax entropy check using characteristic speeds.
- `ConstantRetardation`
  - Parameter: `retardation_factor`.
  - Same interface as `FreundlichSorption`, with linear/constant versions of the methods.
- Characteristic helpers:
  - `characteristic_velocity(c, flow, sorption) = flow / R(c)`.
  - `characteristic_position(c, flow, sorption, t_start, v_start, t)`.
- Rarefaction math:
  - `rarefaction_concentration_at_point(v, t, v_origin, t_origin, flow, sorption)`
    - Uses self-similar relation:
      $$R(C) = \frac{\text{flow} (t - t_0)}{v - v_0}$$
      then analytic inverse via sorption model.
  - `integrate_rarefaction_exact(raref, v_outlet, t_start, t_end, sorption)`
    - Exact analytic $\int C(t)\,dt$ along outlet for a rarefaction (Freundlich case), using closed-form power-law integration.
- Spin-up computation:
  - `compute_first_front_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)`
    - Finds first index `i` where `cin[i] > 0` and defines `t_start = tedges[i]`.
    - Computes retardation `R(c_first)` for this first non-zero concentration using the provided sorption model.
    - Integrates the piecewise-constant flow history exactly from `t_start` forward until the cumulative advected volume reaches `aquifer_pore_volume * R(c_first)`.
    - Uses an analytic partial-bin calculation: if the target is reached within bin `i`, solves `remaining_volume = flow[i] * dt_partial` exactly.
    - Returns `t_first_arrival = t_start + (tedges[i] - t_start) + dt_partial`, which simplifies to the event time measured from the first non-zero inlet time.
    - Returns `np.inf` if there is no non-zero concentration or if the flow history is insufficient to advect the required volume.
    - Defines "spin-up" period: $t < t_\text{first\_arrival}$.

All math is closed-form; no root-finding and no numerical tolerances.

---

### 2.2 `waves.py` (Phase 2 – wave representation)

Common base:

- `Wave` (abstract base class)
  - Attributes: `t_start`, `v_start`, `flow`, `is_active`.
  - Abstract methods:
    - `position_at_time(t)`.
    - `concentration_left()`.
    - `concentration_right()`.
    - `concentration_at_point(v, t)`.

Specific wave types:

- `CharacteristicWave(Wave)`
  - Fields: `concentration`, `sorption`.
  - `velocity() = flow / R(concentration)`.
  - `position_at_time(t)`: linear motion if `t >= t_start` and `is_active`.
  - `concentration_left/right()`: both equal to `concentration`.
  - `concentration_at_point(v, t)`: returns `concentration` when this characteristic controls `(v, t)`.

- `ShockWave(Wave)`
  - Fields: `c_left`, `c_right`, `sorption`, `velocity` (computed in `__post_init__`).
  - `velocity`: computed via `sorption.shock_velocity` (Rankine–Hugoniot).
  - `position_at_time(t)`: linear motion if active.
  - `concentration_left/right()`: shock states.
  - `concentration_at_point(v, t)`:
    - `c_left` upstream, `c_right` downstream, average if exactly on the shock.
  - `satisfies_entropy()`: uses `sorption.check_entropy_condition`.

- `RarefactionWave(Wave)`
  - Fields: `c_head`, `c_tail`, `sorption`.
  - `head_velocity()`, `tail_velocity()`: characteristic velocities at `c_head`, `c_tail`.
  - `head_position_at_time(t)`, `tail_position_at_time(t)`.
  - `contains_point(v, t)`: tests if `(v, t)` lies between head and tail.
  - `concentration_at_point(v, t)`:
    - Uses self-similar solution
      $$R(C) = \frac{\text{flow}(t - t_0)}{v - v_0}$$
      and analytic inversion via sorption.
    - Enforces `c` within `[min(c_tail, c_head), max(c_tail, c_head)]`.
  - `concentration_left() = c_tail`, `concentration_right() = c_head`.

Wave kinematics are exact linear functions of time with analytically determined speeds.

---

### 2.3 `events.py` (Phase 3 – event detection)

**Core structures**

- `EventType` enum:
  - `CHAR_CHAR_COLLISION`, `SHOCK_SHOCK_COLLISION`, `SHOCK_CHAR_COLLISION`,
    `RAREF_CHAR_COLLISION`, `SHOCK_RAREF_COLLISION`, `OUTLET_CROSSING`, `INLET_CHANGE`.
- `Event` dataclass:
  - Fields: `time`, `event_type`, `waves_involved`, `location` (v-position).
  - `__lt__` orders events by time (for heaps/priority queues).

**Analytical intersection functions**

All intersections are computed analytically (no iteration):

- `find_characteristic_intersection(char1, char2, t_current)`.
- `find_shock_shock_intersection(shock1, shock2, t_current)`.
- `find_shock_characteristic_intersection(shock, char, t_current)`.
- `find_rarefaction_boundary_intersections(raref, other_wave, t_current)`:
  - Wraps rarefaction head/tail as temporary characteristics and reuses characteristic/shock intersection logic.
- `find_outlet_crossing(wave, v_outlet, t_current)`:
  - Solves linear crossing times exactly for characteristics and shocks.
  - Uses rarefaction head for rarefaction crossings.

---

### 2.4 `handlers.py` (Phase 4 – wave creation & interactions)

**Inlet wave creation**

- `create_inlet_waves_at_time(c_prev, c_new, t, flow, sorption, v_inlet=0.0)`:
  - If `c_new == c_prev`: no waves.
  - **Special case for C=0 transitions** (added to fix Example 1 issues):
    - If `c_prev ≈ 0` or `c_new ≈ 0`: create `CharacteristicWave` with the new concentration.
    - Physical interpretation: C=0 represents the background/initial condition without a well-defined velocity in Freundlich sorption. When injecting C>0 into C=0 (or vice versa), the solute propagates as a characteristic, leaving its concentration behind and the background ahead.
    - This avoids entropy violations that occur when trying to create shocks from/to C=0 with Freundlich n>1.
  - For nonzero concentrations, compute velocities via sorption model:
    - Compression (`vel_new > vel_prev`): create `ShockWave` with `c_left=c_new`, `c_right=c_prev` and enforce entropy.
    - Expansion (`vel_new < vel_prev`): create `RarefactionWave` with `c_head=c_prev`, `c_tail=c_new`.
    - Same velocity: create `CharacteristicWave` with `c_new`.
- `initialize_all_inlet_waves(cin, flow, tedges, sorption)`:
  - Start from `c_prev = 0.0` (handled by spin-up for physical interpretation).
  - For each bin where `cin[i]` changes, call `create_inlet_waves_at_time` at `tedges[i]`.

**Wave interaction handlers**

- `handle_characteristic_collision(char1, char2, t_event, v_event)`:
  - **Special case for C=0 collisions** (added to fix Example 1 issues):
    - If one characteristic has C≈0 and the other has C>0: deactivate the C=0 characteristic and keep the C>0 characteristic active (no new waves created).
    - Physical interpretation: C=0 represents background; when C>0 water catches up to C=0 region, the C>0 continues propagating.
  - For nonzero characteristic collisions: replaces two characteristics by an entropic `ShockWave` with appropriate left/right states.
  - Deactivates parent characteristics (or just the C=0 one in the special case).
- `handle_shock_collision(shock1, shock2, t_event, v_event)`:
  - Merges shocks into a single `ShockWave` with left state from upstream shock and right state from downstream shock.
  - Deactivates parent shocks.
- `handle_shock_characteristic_collision(shock, char, t_event, v_event)`:
  - Handles shock catching a characteristic or vice versa.
  - Produces a new entropic shock when possible and deactivates parent waves.
- `handle_shock_rarefaction_collision(shock, raref, t_event, v_event, collision_type)`:
  - Simplified but physically motivated treatment of shock–rarefaction interactions.
  - Uses `collision_type` (`"head"` or `"tail"`) to decide the resulting shock or wave modifications.
- `handle_outlet_crossing(wave, t_event, v_outlet)`:
  - Records outlet crossing events without deactivating the wave (waves remain needed for concentration queries).

These handlers are the only place where wave topology changes; all operations are deterministic and analytic.

---

### 2.5 `solver.py` (Phase 5 – main solver)

**Core structures**

- `FrontTrackerState`
  - `waves`: all waves (active and inactive).
  - `events`: list of event dicts (time, type, waves_before/after, etc.).
  - `t_current`: current simulation time.
  - `v_outlet`: outlet position (equal to `aquifer_pore_volume`).
  - `sorption`: `FreundlichSorption` or `ConstantRetardation` instance.
  - Inlet data: `cin`, `flow`, `tedges`.

- `FrontTracker`
  - `__init__(cin, flow, tedges, aquifer_pore_volume, sorption)`:
    - Validates shapes, positivity, and consistency of inputs.
    - Sets `v_outlet = aquifer_pore_volume`.
    - Computes `t_first_arrival` via `compute_first_front_arrival_time` (spin-up boundary).
    - Initializes inlet waves via `initialize_all_inlet_waves`.
  - `find_next_event()`:
    - Uses analytical intersection functions to collect candidate events involving all active waves.
    - Rejects events outside domain `0 <= v <= v_outlet` or in the past.
    - Includes outlet crossings via `find_outlet_crossing`.
    - Returns earliest event as an `Event` instance or `None` if no more events.
  - `handle_event(event)`:
    - Dispatches to the appropriate handler (`handle_characteristic_collision`, `handle_shock_collision`, etc.).
    - Updates `state.waves` with new waves and appends an event record to `state.events`.
  - `run(max_iterations=10000)`:
    - Event-driven loop: repeatedly finds the next event, advances `t_current` to the event time, and handles the event.
    - Optional physics verification via `verify_physics()` every N events.
    - Prints summary statistics (events, waves, active waves, first arrival time).
  - `verify_physics(*, check_mass_balance=False, mass_balance_rtol=1e-12)`:
    - Enforces entropy condition for all active shocks and rarefactions.
    - Optional mass balance verification (when `check_mass_balance=True`):
      - Computes mass_in_domain + mass_out_cumulative and compares to mass_in_cumulative.
      - Uses exact analytical integration functions from `output.py`.
      - Raises `RuntimeError` if relative error exceeds `mass_balance_rtol`.
      - Mass balance equation:
        $$\text{mass\_in\_domain}(t) + \text{mass\_out\_cumulative}(t) = \text{mass\_in\_cumulative}(t)$$

  - Known limitation (Phase 5 implementation detail):
    - For rarefaction-related collisions (`RAREF_CHAR_COLLISION`, `SHOCK_RAREF_COLLISION`,
      `RAREF_RAREF_COLLISION`), the event detection layer computes which rarefaction boundary
      (head vs tail) participates in the interaction.
    - The current `handle_event` implementation in `solver.py` does not yet
      propagate this boundary information through the `Event` object and instead passes a
      hard-coded `boundary_type` (e.g. "head" or "tail") to the corresponding handler.
    - Existing Phase 4 and 8 tests cover the physically relevant interaction patterns used
      so far, but future refinements should propagate the exact boundary type through
      `Event` and into the handlers for full generality.

The solver is fully event-driven and analytical; there is no time-stepping.

---

### 2.6 `output.py` (Phase 6 – concentration extraction)

- `concentration_at_point(v, t, waves, sorption)`:
  - Exact concentration $C(v, t)$ based on the wave set.
  - Priority:
    1. Rarefactions (if point is inside rarefaction fan; use rarefaction self-similar solution).
    2. Shocks and rarefaction tails (most recent wave to pass through point).
       - Shocks: track crossing time and concentration (c_left after crossing, c_right ahead of shock).
       - Rarefaction tails: track when tail passed point and return tail concentration.
       - Return concentration from whichever passed most recently.
    3. Characteristics (latest characteristic that has passed through `(v, t)`).
  - If no wave controls `(v, t)`, returns `0.0` (initial condition).
  - **Critical fix (2025-01-18)**: Rarefaction tails now properly establish final plateaus after passing a point. Previously, after a rarefaction tail passed, the code incorrectly reverted to an earlier characteristic's concentration.

- `compute_breakthrough_curve(t_array, v_outlet, waves, sorption)`:
  - Evaluates `concentration_at_point(v_outlet, t, ...)` for each `t` in `t_array`.

- `identify_outlet_segments(t_start, t_end, v_outlet, waves)`:
  - Decomposes `[t_start, t_end]` into segments where the outlet concentration is controlled by a single wave or wave type.
  - Returns a list of segments with:
    - `t_start`, `t_end`.
    - `type` (e.g. `"constant"`, `"rarefaction"`).
    - `concentration` (for constant segments) or `wave` reference.

- `integrate_rarefaction_exact(raref, v_outlet, t_start, t_end, sorption)`:
  - Computes exact integral of rarefaction-controlled concentration at the outlet.

- `compute_bin_averaged_concentration_exact(t_edges, v_outlet, waves, sorption)`:
  - For each bin `[t_i, t_{i+1}]`, uses `identify_outlet_segments` and analytic integrals:
    - Constant segments: `integral = C * Δt`.
    - Rarefaction segments: `integral = integrate_rarefaction_exact(...)`.
  - Returns bin-averaged concentrations for each bin with no numerical quadrature.

**Mass balance verification functions** (added for High Priority #3):

- `compute_domain_mass(t, v_outlet, waves, sorption)`:
  - Computes total mass (dissolved + sorbed) in domain [0, v_outlet] at time t.
  - Uses **exact analytical spatial integration** (no numerical quadrature):
    - Partitions domain by wave positions into constant-concentration regions and rarefaction fans.
    - For constant regions: exact integration using $C_{\text{total}} \times \Delta v$.
    - For rarefactions: calls `_integrate_rarefaction_spatial_exact()`.
  - Total concentration: $C_{\text{total}} = C + \frac{\rho_b}{n_{\text{por}}} k_f C^{1/n}$ (dissolved + sorbed).
  - Achieves machine precision for spatial integrals.

- `_integrate_rarefaction_spatial_exact(raref, v_start, v_end, t, sorption)`:
  - **Exact analytical spatial integral** of rarefaction concentration over [v_start, v_end].
  - **Unified formula for ALL n > 0** using generalized incomplete beta function via mpmath:
    - Both dissolved and sorbed integrals reduce to power-law forms: $\int u^p (\kappa-u)^q du$
    - Expressed using generalized incomplete beta function: $\kappa^{p+q+1} B(u_1/\kappa, u_2/\kappa; p+1, q+1)$
    - Uses `mpmath.betainc()` which handles negative parameters via analytic continuation
    - Achieves machine precision (~1e-15 relative error) for all positive real n > 0
    - No conditional logic or special cases (except optimized path for n=2)
  - **Optimized path for n=2** (β = -0.5) using explicit polynomial antiderivatives:
    - Rarefaction concentration: $C(u) = \frac{\alpha^2 u^2}{(\kappa - u)^2}$ where $u = v - v_{\text{origin}}$, $\kappa = \text{flow} \times (t - t_{\text{origin}})$, $\alpha = \frac{\rho_b k_f}{n_{\text{por}} n}$.
    - Dissolved antiderivative: $\alpha^2 \left[ \frac{\kappa^2}{\kappa-u} + 2\kappa \ln(\kappa-u) - (\kappa-u) \right]$.
    - Sorbed antiderivative: $\frac{\rho_b}{n_{\text{por}}} k_f \alpha \left[ -u - \kappa \ln(\kappa-u) \right]$.
  - **Mathematical properties**:
    - Continuous for all n > 0 (no discontinuities)
    - True closed-form antiderivative (no numerical quadrature)
    - Single unified formula (implemented 2025-01-24)

- `compute_cumulative_inlet_mass(t, cin, flow, tedges)`:
  - Computes total mass that has entered the domain from t=0 to t=t.
  - Exact piecewise-constant integration: $\int_0^t C_{\text{in}}(s) \times \text{flow}(s) \, ds$.
  - No numerical quadrature.

- `compute_cumulative_outlet_mass(t, v_outlet, waves, sorption, flow, tedges)`:
  - Computes total mass that has exited the domain from t=0 to t=t.
  - Uses existing exact outlet integration infrastructure (`identify_outlet_segments`, `integrate_rarefaction_exact`).
  - Integrates outlet concentration weighted by flow rate: $\int_0^t C_{\text{out}}(s) \times \text{flow}(s) \, ds$.

This ensures machine-precision mass balance and no numerical dispersion.

---

### 2.7 `plot.py` and `examples/08_Front_Tracking_Exact_Solution.ipynb` (Phase 9)

- `plot_vt_diagram(tracker_state, ...)`:
  - Visualizes characteristics (blue), shocks (red), rarefactions (green) in the $(v, t)$ plane.
- `plot_breakthrough_curve(...)`:
  - Builds breakthrough curves at the outlet using `concentration_at_point` and plots them.
- `examples/08_Front_Tracking_Exact_Solution.ipynb`:
  - Demonstrates step inputs, pulses, V–t diagrams, mass balance verification, spin-up handling, and comparison with a convolution approach.

---

### 2.8 Public API in `advection.py` (Phase 7)

- `infiltration_to_extraction_front_tracking(...)`
  - Keyword-only parameters:
    - `cin`, `flow`, `tedges`, `cout_tedges`, `aquifer_pore_volume`.
    - Freundlich parameters: `freundlich_k`, `freundlich_n`, `bulk_density`, `porosity`.
    - Optional `retardation_factor` for constant retardation.
    - `max_iterations`.
  - Behavior:
    - Constructs appropriate sorption model (`FreundlichSorption` if `retardation_factor` is `None`, otherwise `ConstantRetardation`).
    - Initializes `FrontTracker`, runs the event-driven simulation.
    - Uses `compute_bin_averaged_concentration_exact` to compute bin-averaged extraction concentration aligned with `cout_tedges`.
    - Computes and internally stores `t_first_arrival`.
  - Returns:
    - `cout`: bin-averaged extraction concentration.

- `infiltration_to_extraction_front_tracking_detailed(...)`
  - Same inputs as the main function.
  - Returns:
    - `cout` plus a diagnostics structure with:
      - `waves`, `events`, `t_first_arrival`, `n_events`, `n_shocks`, `n_rarefactions`, and `tracker_state`.

All associated tests in `tests/src/test_front_tracking_*.py` and `tests/src/test_advection_api_phase7.py` pass with tight tolerances (rtol ~ 1e-14).

---

## 3. Critical invariants and conventions

When editing or extending the front-tracking code, the following invariants and conventions must be preserved.

### 3.1 Exact analytical computation

- No iterative solvers (no root-finding, no numerical integration, no tolerance-driven loops).
- All crossing times, velocities, and concentrations must come from closed-form expressions.

### 3.2 Physical correctness

- All shocks must satisfy the Lax entropy condition.
- No negative concentrations; outlet concentrations must never exceed inlet maxima.
- Rarefactions use the self-similar solution consistent with the conservation law.

### 3.3 Spin-up period handling

- `compute_first_front_arrival_time` is the canonical source of `t_first_arrival`.
- Times `t < t_first_arrival` are "spin-up" and depend on unknown initial conditions.
- For `t >= t_first_arrival`, the solution is fully determined by inlet history and the sorption model.

### 3.4 Diagnostics and observability

- All events are stored in `FrontTrackerState.events` as plain dicts.
- All waves (including inactive ones) are retained in `FrontTrackerState.waves` with `is_active` flags.
- The detailed API (`infiltration_to_extraction_front_tracking_detailed`) must expose enough structure to reconstruct wave and event histories for analysis and debugging.

### 3.5 Multiple-streamline architecture compatibility

- Current implementation is for a single streamline with pore volume as the spatial coordinate.
- Design and APIs must remain compatible with future extensions to multiple streamlines / pore-volume distributions (e.g., do not hard-code one-streamline assumptions into public APIs if avoidable).

### 3.6 API and naming consistency

- Use consistent names matching `gwtransport` terminology:
  - `cin`, `flow`, `tedges`, `cout_tedges`, `aquifer_pore_volume`, `freundlich_k`, `freundlich_n`, `bulk_density`, `porosity`, `retardation_factor`, `t_first_arrival`, etc.
- Units:
  - Time: days.
  - Volume: m³.
  - Flow: m³/day.
- Time representation convention:
  - **Input/Output**: `tedges` and `cout_tedges` are always `pd.DatetimeIndex` (pandas Timestamps).
  - **Internal simulation**: All times (`t_current`, `t_start`, `t_first_arrival`, event times, wave times) are floats representing **days from `tedges[0]`**.
  - **No flexibility**: Code should NOT use try-except blocks to accept both floats and Timestamps. Each function has a clear contract: internal functions use floats in days, public APIs accept DatetimeIndex.
  - Conversion: When initializing from `tedges`, convert to days: `(tedges[i] - tedges[0]) / pd.Timedelta(days=1)`.
- Input timeseries data is provided as bin-averaged numpy arrays; The time edges of the bin are `tedges` and are pandas `DatetimeIndex`, and are of length `len(cin) + 1`.
- No capital variable names in the implementation:
  - Use `c_left`, `v_outlet`, `t_first_arrival` rather than `C_left`, `V_outlet`, `TFirstArrival`.
- Plotting of binned values must be plotted as steps with:
  - `x, y = np.repeat(tedges, 2)[1:-1], np.repeat(cout, 2)`

### 3.7 Testing discipline

- Tests must be explicit and simple:
  - No control flow (`if`, `for`) or `try/except` within test bodies, except where strictly necessary and then only with clear justification.
  - Each comparison must directly serve the test purpose.
- Masked arrays may be used only when the unmasked values are enough to fully validate the test purpose.
- Numerical tolerances in tests should be extremely tight (typical rtol ~ 1e-14) consistent with the exact analytical implementation.

---

## 4. How to use this knowledge base

- Treat this file as the authoritative high-level description of the front-tracking implementation.
- When changing code:
  - Check which module and phase are affected.
  - Verify that the change respects the invariants in Section 3.
  - Consider how the change interacts with event detection, wave interactions, and spin-up handling.
- When using an AI assistant:
  - Provide this file (and `KNOWLEDGE_BASE_PROMPT.md`) as context.
  - Ask the assistant explicitly to respect all constraints in Section 3 when proposing changes.
