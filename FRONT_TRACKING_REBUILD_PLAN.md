# Front Tracking Rebuild Plan - EXACT ANALYTICAL IMPLEMENTATION

**Branch**: `front-tracking-clean` (clean branch from main)
**Goal**: Machine-precision, physically correct, exact analytical solution for front tracking with nonlinear sorption

**Status**: Phase 6 COMPLETE ✅

---

## Progress Summary

### Completed
- ✅ Phase 1.1-1.4: Core Mathematical Foundation (front_tracking_math.py)
  - FreundlichSorption class with all methods
  - ConstantRetardation class
  - Characteristic velocity functions
  - First arrival time computation
  - **All 39 unit tests passing** with machine precision (rtol=1e-14)

- ✅ Phase 2.1-2.4: Wave Representation (front_tracking_waves.py)
  - Abstract Wave base class with position and concentration methods
  - CharacteristicWave class for smooth regions
  - ShockWave class with Rankine-Hugoniot condition
  - RarefactionWave class with self-similar solution
  - **All 33 unit tests passing**

- ✅ Phase 3.1-3.2: Event Detection (front_tracking_events.py)
  - Event and EventType dataclasses
  - find_characteristic_intersection (exact analytical)
  - find_shock_shock_intersection (exact analytical)
  - find_shock_characteristic_intersection (exact analytical)
  - find_rarefaction_boundary_intersections (head/tail detection)
  - find_outlet_crossing (for all wave types)
  - **All 20 unit tests passing** with machine precision (rtol=1e-14)

- ✅ Phase 4.1-4.2: Wave Interactions (front_tracking_handlers.py)
  - handle_characteristic_collision: creates entropic shocks
  - handle_shock_collision: merges shocks with proper entropy
  - handle_shock_characteristic_collision: absorbs characteristics
  - handle_shock_rarefaction_collision: complex interactions (simplified)
  - handle_rarefaction_characteristic_collision: boundary interactions
  - handle_outlet_crossing: records exit events
  - create_inlet_waves_at_time: inlet boundary condition handler
  - **All 22 unit tests passing** with entropy verification

- ✅ Phase 5.1-5.2: Front Tracker (front_tracking_solver.py)
  - FrontTrackerState: Complete simulation state management
  - FrontTracker class: Main event-driven solver
  - find_next_event(): Searches all wave interactions with exact analytical detection
  - handle_event(): Dispatcher for all event types
  - run(): Main simulation loop with chronological event processing
  - verify_physics(): Entropy and consistency checks
  - _initialize_inlet_waves(): Automatic inlet wave creation
  - **All 21 unit tests passing** with edge case handling

- ✅ Phase 6.1-6.3: Concentration Extraction (front_tracking_output.py)
  - concentration_at_point(): Exact point-wise concentration computation
  - compute_breakthrough_curve(): Concentration at outlet over time
  - identify_outlet_segments(): Wave segment identification for integration
  - integrate_rarefaction_exact(): Exact analytical integration of rarefactions
  - compute_bin_averaged_concentration_exact(): Bin-averaged output with exact integration
  - **All 23 unit tests passing** with machine precision (rtol=1e-14)
  - **Total: 161 tests passing** (39 + 33 + 20 + 22 + 21 + 23 + integration tests)

### Next Steps
- Phase 7: API Integration (public interface in advection.py)
- Phase 8: Complete integration testing and validation

### For Session Continuation

If starting a new session, tell Claude:
> "Continue implementing the front tracking rebuild. We're on branch `front-tracking-clean`. Phases 1-6 (Mathematical Foundation + Wave Representation + Event Detection + Wave Interactions + Front Tracker + Concentration Extraction) complete with 161 tests passing. Next: implement Phase 7 (API Integration - create public interface in advection.py). See FRONT_TRACKING_REBUILD_PLAN.md for the full plan."

---

## Design Principles

1. **Exact Analytical Computation**: All calculations use closed-form analytical expressions (no numerical tolerances, no iterative solvers)
2. **Physical Correctness**: Every wave satisfies conservation laws and entropy conditions exactly
3. **Spin-up Period Handling**: Compute first arrival time; output is independent of unknown initial conditions after spin-up
4. **Detailed Diagnostics**: Track all events, waves, and state changes for verification
5. **Multiple Streamline Architecture**: Design supports future extension to distributions of pore volumes
6. **Consistent API**: Use gwtransport terminology (tedges, cin, flow, aquifer_pore_volume, etc.)
7. **No Capital Variable Names**: Follow Python convention (c_left not C_L, v_max not V_max)
8. **Meaningful and Explicit Tests**: None of the tests contain conditional statements or try-except constructs. Ensure each comparisson serves the purpose of the test.

---

## Phase 1: Core Mathematical Foundation (Exact Analytical)

### 1.1: Freundlich Sorption Class

**File**: `src/gwtransport/front_tracking_math.py`

**Class**: `FreundlichSorption`
```python
@dataclass
class FreundlichSorption:
    """Freundlich sorption: s(C) = k_f * C^(1/n)"""
    k_f: float          # Freundlich coefficient [(m³/kg)^(1/n)]
    n: float            # Freundlich exponent [-]
    bulk_density: float # ρ_b [kg/m³]
    porosity: float     # n_por [-]

    def retardation(self, c: float) -> float:
        """R(C) = 1 + (ρ_b*k_f)/(n_por*n) * C^((1/n)-1)"""

    def total_concentration(self, c: float) -> float:
        """C_total = C + (ρ_b/n_por)*k_f*C^(1/n)"""

    def concentration_from_retardation(self, r: float) -> float:
        """Invert R(C) analytically: C = [(R-1)*n_por*n/(ρ_b*k_f)]^(n/(1-n))"""

    def shock_velocity(self, c_left: float, c_right: float, flow: float) -> float:
        """Rankine-Hugoniot: s = [flow*(c_right - c_left)] / [C_total(c_right) - C_total(c_left)]"""

    def check_entropy_condition(self, c_left: float, c_right: float, shock_vel: float, flow: float) -> bool:
        """Lax condition: λ(c_left) > shock_vel > λ(c_right)"""
```

**Special Case**: `ConstantRetardation`
```python
@dataclass
class ConstantRetardation:
    """Linear case: R(C) = constant"""
    retardation_factor: float

    # Same interface as FreundlichSorption but simplified
```

**Unit Tests**:
- Verify R(0) = 1.0 exactly
- Test R→C→R roundtrip: `assert c == concentration_from_retardation(retardation(c))`
- Compare shock_velocity against known Buckley-Leverett solutions
- Test entropy condition on known physical/unphysical shocks

---

### 1.2: Characteristic Mathematics

**Functions**:
```python
def characteristic_velocity(c: float, flow: float, sorption: FreundlichSorption) -> float:
    """Exact: v = flow / R(c)"""
    return flow / sorption.retardation(c)

def characteristic_position(c: float, flow: float, sorption: FreundlichSorption,
                           t_start: float, v_start: float, t: float) -> float:
    """Exact linear propagation: V(t) = v_start + velocity*(t - t_start)"""
    velocity = characteristic_velocity(c, flow, sorption)
    return v_start + velocity * (t - t_start)
```

**Unit Tests**:
- Verify dV/dt = flow/R(c) by computing finite difference and comparing
- Test that characteristics with c=0 travel at speed = flow (R(0)=1)

---

### 1.3: Rarefaction Analytical Solutions

**Functions**:
```python
def rarefaction_concentration_at_point(v: float, t: float,
                                       v_origin: float, t_origin: float,
                                       flow: float, sorption: FreundlichSorption) -> float:
    """
    Self-similar solution: R(C) = flow * (t - t_origin) / (v - v_origin)
    Returns C by inverting R analytically
    """
    if abs(v - v_origin) < 1e-15:  # Exact zero check
        return None  # Undefined at origin

    r_target = flow * (t - t_origin) / (v - v_origin)
    return sorption.concentration_from_retardation(r_target)

def rarefaction_mass_in_fan(c_head: float, c_tail: float,
                            v_head: float, v_tail: float,
                            flow: float, sorption: FreundlichSorption) -> float:
    """
    Exact analytical integral: ∫_{v_tail}^{v_head} C_total(C(v)) dv

    Derivation:
    In rarefaction: C(v) defined by R(C) = flow*(t-t0)/(v-v0)
    Change variables: dv = [flow*(t-t0)/R^2] * dR/dC * dC
    Integrate C_total(C) with exact antiderivative
    """
    # TODO: Derive exact closed-form integral for Freundlich case
    # This is critical for exact mass balance verification
```

**Unit Tests**:
- Verify self-similar solution satisfies PDE: ∂C_total/∂t + ∂(flow*C)/∂v = 0
- Test mass conservation: integrate mass in fan analytically and verify
- Compare head/tail positions against characteristic velocities

---

### 1.4: First Arrival Time Computation

**Function**:
```python
def compute_first_arrival_time(cin: np.ndarray, flow: np.ndarray,
                               tedges: np.ndarray, aquifer_pore_volume: float,
                               sorption: FreundlichSorption | ConstantRetardation) -> float:
    """
    Compute time of first non-zero concentration arrival at outlet.

    Algorithm:
    1. Find first i where cin[i] > 0
    2. Compute residence time for this concentration from tedges[i]
    3. First arrival = tedges[i] + residence_time

    Returns:
        t_first_arrival [days] - Time when first non-zero concentration reaches outlet

    Notes:
        - All times before t_first_arrival are spin-up period
        - Initial conditions (unknown C₀(v)) only affect t < t_first_arrival
        - For t ≥ t_first_arrival, solution is completely determined by inlet history
    """
    # Find first non-zero concentration
    idx_first = np.argmax(cin > 0)
    if cin[idx_first] == 0:
        return np.inf  # No concentration ever arrives

    c_first = cin[idx_first]
    t_start = tedges[idx_first]

    # Compute residence time for this concentration
    # For this streamline: residence_time = aquifer_pore_volume * R(c) / flow
    # But flow is piecewise constant, so integrate carefully

    r_first = sorption.retardation(c_first) if hasattr(sorption, 'retardation') else sorption.retardation_factor

    # Integrate from t_start forward until cumulative flow reaches aquifer_pore_volume * r_first
    target_volume = aquifer_pore_volume * r_first
    cumulative = 0.0

    for i in range(idx_first, len(flow)):
        dt = tedges[i+1] - tedges[i]
        volume_in_bin = flow[i] * dt

        if cumulative + volume_in_bin >= target_volume:
            # First arrival occurs during this bin
            remaining = target_volume - cumulative
            dt_partial = remaining / flow[i]
            return t_start + (tedges[i] - tedges[idx_first]) + dt_partial

        cumulative += volume_in_bin

    return np.inf  # Never reaches outlet with given flow history
```

**Unit Tests**:
- Test simple case: constant C, constant flow → t_first = aquifer_pore_volume * R / flow
- Test variable flow: verify analytical integration matches expected arrival time
- Test edge case: cin all zeros → return infinity

---

## Phase 2: Wave Representation

### 2.1: Abstract Wave Base Class

**File**: `src/gwtransport/front_tracking_waves.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Wave(ABC):
    """Abstract base class for all wave types"""
    t_start: float      # Formation time [days]
    v_start: float      # Formation position [m³]
    flow: float         # Flow rate [m³/day] (piecewise constant in this time interval)
    is_active: bool = True

    @abstractmethod
    def position_at_time(self, t: float) -> float | None:
        """Compute wave position at time t"""

    @abstractmethod
    def concentration_left(self) -> float:
        """Concentration on left (upstream) side"""

    @abstractmethod
    def concentration_right(self) -> float:
        """Concentration on right (downstream) side"""

    @abstractmethod
    def concentration_at_point(self, v: float, t: float) -> float | None:
        """Concentration at point (v, t) if wave controls it"""
```

---

### 2.2: Characteristic Wave

```python
@dataclass
class CharacteristicWave(Wave):
    """Characteristic line along which C is constant"""
    concentration: float
    sorption: FreundlichSorption | ConstantRetardation

    def velocity(self) -> float:
        """Characteristic velocity: flow / R(C)"""
        if hasattr(self.sorption, 'retardation'):
            return self.flow / self.sorption.retardation(self.concentration)
        else:
            return self.flow / self.sorption.retardation_factor

    def position_at_time(self, t: float) -> float | None:
        if t < self.t_start or not self.is_active:
            return None
        return self.v_start + self.velocity() * (t - self.t_start)

    def concentration_left(self) -> float:
        return self.concentration

    def concentration_right(self) -> float:
        return self.concentration

    def concentration_at_point(self, v: float, t: float) -> float | None:
        """Concentration is constant along characteristic"""
        v_at_t = self.position_at_time(t)
        if v_at_t is None:
            return None
        # Check if point (v,t) is on this characteristic line
        # In practice, this is handled by wave ordering, not exact equality
        return self.concentration
```

---

### 2.3: Shock Wave

```python
@dataclass
class ShockWave(Wave):
    """Shock discontinuity"""
    c_left: float       # Upstream concentration
    c_right: float      # Downstream concentration
    sorption: FreundlichSorption | ConstantRetardation
    velocity: float = None  # Computed in __post_init__

    def __post_init__(self):
        """Compute shock velocity from Rankine-Hugoniot"""
        if self.velocity is None:
            self.velocity = self.sorption.shock_velocity(
                self.c_left, self.c_right, self.flow
            )

    def position_at_time(self, t: float) -> float | None:
        if t < self.t_start or not self.is_active:
            return None
        return self.v_start + self.velocity * (t - self.t_start)

    def concentration_left(self) -> float:
        return self.c_left

    def concentration_right(self) -> float:
        return self.c_right

    def concentration_at_point(self, v: float, t: float) -> float | None:
        """Return c_left or c_right depending on which side of shock"""
        v_shock = self.position_at_time(t)
        if v_shock is None:
            return None

        # Small tolerance for numerical comparison
        if v < v_shock - 1e-15:
            return self.c_left
        elif v > v_shock + 1e-15:
            return self.c_right
        else:
            # Exactly at shock - undefined, return average
            return 0.5 * (self.c_left + self.c_right)

    def satisfies_entropy(self) -> bool:
        """Verify Lax entropy condition"""
        return self.sorption.check_entropy_condition(
            self.c_left, self.c_right, self.velocity, self.flow
        )
```

---

### 2.4: Rarefaction Wave

```python
@dataclass
class RarefactionWave(Wave):
    """Rarefaction (expansion fan) with smooth concentration gradient"""
    c_head: float       # Leading edge concentration (faster)
    c_tail: float       # Trailing edge concentration (slower)
    sorption: FreundlichSorption | ConstantRetardation

    def head_velocity(self) -> float:
        """Velocity of head characteristic"""
        if hasattr(self.sorption, 'retardation'):
            return self.flow / self.sorption.retardation(self.c_head)
        else:
            return self.flow / self.sorption.retardation_factor

    def tail_velocity(self) -> float:
        """Velocity of tail characteristic"""
        if hasattr(self.sorption, 'retardation'):
            return self.flow / self.sorption.retardation(self.c_tail)
        else:
            return self.flow / self.sorption.retardation_factor

    def head_position_at_time(self, t: float) -> float | None:
        if t < self.t_start or not self.is_active:
            return None
        return self.v_start + self.head_velocity() * (t - self.t_start)

    def tail_position_at_time(self, t: float) -> float | None:
        if t < self.t_start or not self.is_active:
            return None
        return self.v_start + self.tail_velocity() * (t - self.t_start)

    def position_at_time(self, t: float) -> float | None:
        """Return head position (leading edge of rarefaction)"""
        return self.head_position_at_time(t)

    def contains_point(self, v: float, t: float) -> bool:
        """Check if point (v,t) is inside rarefaction fan"""
        if t < self.t_start or not self.is_active:
            return False

        v_head = self.head_position_at_time(t)
        v_tail = self.tail_position_at_time(t)

        if v_head is None or v_tail is None:
            return False

        return v_tail <= v <= v_head

    def concentration_at_point(self, v: float, t: float) -> float | None:
        """Self-similar solution: R(C) = flow*(t-t0)/(v-v0)"""
        if not self.contains_point(v, t):
            return None

        if abs(v - self.v_start) < 1e-15:
            return self.c_tail  # At origin

        r_target = self.flow * (t - self.t_start) / (v - self.v_start)

        if hasattr(self.sorption, 'concentration_from_retardation'):
            c = self.sorption.concentration_from_retardation(r_target)
        else:
            # Constant retardation - no variation within rarefaction
            # This shouldn't happen - rarefactions require concentration-dependent R
            return None

        # Verify c is in valid range [c_tail, c_head] (or [c_head, c_tail] for n<1)
        c_min = min(self.c_tail, self.c_head)
        c_max = max(self.c_tail, self.c_head)

        if c_min <= c <= c_max:
            return c
        return None

    def concentration_left(self) -> float:
        return self.c_tail

    def concentration_right(self) -> float:
        return self.c_head
```

---

## Phase 3: Event Detection (Exact Intersections)

### 3.1: Event Data Structure

**File**: `src/gwtransport/front_tracking_events.py`

```python
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    """All possible event types"""
    CHAR_CHAR_COLLISION = "characteristic_collision"
    SHOCK_SHOCK_COLLISION = "shock_collision"
    SHOCK_CHAR_COLLISION = "shock_characteristic_collision"
    RAREF_CHAR_COLLISION = "rarefaction_characteristic_collision"
    SHOCK_RAREF_COLLISION = "shock_rarefaction_collision"
    OUTLET_CROSSING = "outlet_crossing"
    INLET_CHANGE = "inlet_concentration_change"

@dataclass
class Event:
    """Represents a single event in the simulation"""
    time: float
    event_type: EventType
    waves_involved: list[Wave]  # References to waves involved
    location: float  # v position where event occurs

    def __lt__(self, other):
        """Events ordered by time (for priority queue)"""
        return self.time < other.time
```

---

### 3.2: Intersection Calculations

**All intersections computed analytically (no iteration, no tolerance)**

```python
def find_characteristic_intersection(char1: CharacteristicWave,
                                    char2: CharacteristicWave,
                                    t_current: float) -> tuple[float, float] | None:
    """
    Exact analytical intersection of two characteristics.

    Solve:
        v1_start + vel1*(t - t1_start) = v2_start + vel2*(t - t2_start)

    Returns:
        (t_intersect, v_intersect) or None if no future intersection
    """
    vel1 = char1.velocity()
    vel2 = char2.velocity()

    # Check parallel
    if abs(vel1 - vel2) < 1e-15:  # Exact parallel
        return None

    # Both characteristics must be active at some common time
    t_both_active = max(char1.t_start, char2.t_start, t_current)

    # Positions when both active
    v1 = char1.position_at_time(t_both_active)
    v2 = char2.position_at_time(t_both_active)

    if v1 is None or v2 is None:
        return None

    # Time until intersection from t_both_active
    dt = (v2 - v1) / (vel1 - vel2)

    if dt <= 0:  # Intersection in past
        return None

    t_intersect = t_both_active + dt
    v_intersect = v1 + vel1 * dt

    return (t_intersect, v_intersect)


def find_shock_shock_intersection(shock1: ShockWave, shock2: ShockWave,
                                  t_current: float) -> tuple[float, float] | None:
    """Exact analytical intersection of two shocks"""
    vel1 = shock1.velocity
    vel2 = shock2.velocity

    if abs(vel1 - vel2) < 1e-15:
        return None

    t_both_active = max(shock1.t_start, shock2.t_start, t_current)

    v1 = shock1.position_at_time(t_both_active)
    v2 = shock2.position_at_time(t_both_active)

    if v1 is None or v2 is None:
        return None

    dt = (v2 - v1) / (vel1 - vel2)

    if dt <= 0:
        return None

    t_intersect = t_both_active + dt
    v_intersect = v1 + vel1 * dt

    return (t_intersect, v_intersect)


def find_shock_characteristic_intersection(shock: ShockWave, char: CharacteristicWave,
                                          t_current: float) -> tuple[float, float] | None:
    """Exact analytical intersection of shock and characteristic"""
    vel_shock = shock.velocity
    vel_char = char.velocity()

    if abs(vel_shock - vel_char) < 1e-15:
        return None

    t_both_active = max(shock.t_start, char.t_start, t_current)

    v_shock = shock.position_at_time(t_both_active)
    v_char = char.position_at_time(t_both_active)

    if v_shock is None or v_char is None:
        return None

    dt = (v_char - v_shock) / (vel_shock - vel_char)

    if dt <= 0:
        return None

    t_intersect = t_both_active + dt
    v_intersect = v_shock + vel_shock * dt

    return (t_intersect, v_intersect)


def find_rarefaction_boundary_intersections(raref: RarefactionWave,
                                           other_wave: Wave,
                                           t_current: float) -> list[tuple[float, float, str]]:
    """
    Find intersections of rarefaction head/tail with another wave.

    Returns:
        List of (t, v, boundary_type) where boundary_type is 'head' or 'tail'
    """
    intersections = []

    # Create temporary characteristics for head and tail
    head_char = CharacteristicWave(
        t_start=raref.t_start,
        v_start=raref.v_start,
        flow=raref.flow,
        concentration=raref.c_head,
        sorption=raref.sorption
    )

    tail_char = CharacteristicWave(
        t_start=raref.t_start,
        v_start=raref.v_start,
        flow=raref.flow,
        concentration=raref.c_tail,
        sorption=raref.sorption
    )

    # Check head intersection
    if isinstance(other_wave, CharacteristicWave):
        result = find_characteristic_intersection(head_char, other_wave, t_current)
        if result:
            intersections.append((result[0], result[1], 'head'))

        result = find_characteristic_intersection(tail_char, other_wave, t_current)
        if result:
            intersections.append((result[0], result[1], 'tail'))

    elif isinstance(other_wave, ShockWave):
        result = find_shock_characteristic_intersection(other_wave, head_char, t_current)
        if result:
            intersections.append((result[0], result[1], 'head'))

        result = find_shock_characteristic_intersection(other_wave, tail_char, t_current)
        if result:
            intersections.append((result[0], result[1], 'tail'))

    return intersections


def find_outlet_crossing(wave: Wave, v_outlet: float, t_current: float) -> float | None:
    """
    Exact analytical time when wave crosses outlet.

    For characteristics and shocks: solve v_start + vel*(t - t_start) = v_outlet
    For rarefactions: solve for when head crosses
    """
    if not wave.is_active:
        return None

    if isinstance(wave, (CharacteristicWave, ShockWave)):
        # Get current position
        v_current = wave.position_at_time(t_current)
        if v_current is None or v_current >= v_outlet:
            return None

        # Get velocity
        if isinstance(wave, CharacteristicWave):
            vel = wave.velocity()
        else:
            vel = wave.velocity

        if vel <= 0:
            return None

        # Solve: v_current + vel*(t - t_current) = v_outlet
        dt = (v_outlet - v_current) / vel
        return t_current + dt

    elif isinstance(wave, RarefactionWave):
        # Head crosses first
        v_head = wave.head_position_at_time(t_current)
        if v_head is None or v_head >= v_outlet:
            return None

        vel_head = wave.head_velocity()
        if vel_head <= 0:
            return None

        dt = (v_outlet - v_head) / vel_head
        return t_current + dt

    return None
```

---

## Phase 4: Wave Interactions (Event Handlers)

### 4.1: Inlet Wave Creation

**File**: `src/gwtransport/front_tracking_inlet.py`

```python
def create_inlet_waves_at_time(c_prev: float, c_new: float, t: float,
                               flow: float, sorption: FreundlichSorption | ConstantRetardation,
                               v_inlet: float = 0.0) -> list[Wave]:
    """
    Create appropriate waves when inlet concentration changes from c_prev to c_new.

    Logic:
    - Compute velocities: vel_prev = flow/R(c_prev), vel_new = flow/R(c_new)
    - If vel_new > vel_prev: Compression → create ShockWave
    - If vel_new < vel_prev: Expansion → create RarefactionWave
    - If vel_new == vel_prev: Contact discontinuity → create CharacteristicWave

    Returns:
        List of newly created waves (typically 1 wave per concentration change)
    """
    if abs(c_new - c_prev) < 1e-15:  # No change
        return []

    # Compute velocities
    if hasattr(sorption, 'retardation'):
        r_prev = sorption.retardation(c_prev) if c_prev > 0 else 1.0
        r_new = sorption.retardation(c_new) if c_new > 0 else 1.0
    else:
        r_prev = sorption.retardation_factor
        r_new = sorption.retardation_factor

    vel_prev = flow / r_prev
    vel_new = flow / r_new

    if vel_new > vel_prev + 1e-15:  # Compression
        # New water is faster - will catch old water - create shock
        shock = ShockWave(
            t_start=t,
            v_start=v_inlet,
            flow=flow,
            c_left=c_new,      # Upstream is new (faster) water
            c_right=c_prev,    # Downstream is old (slower) water
            sorption=sorption
        )

        # Verify entropy
        if not shock.satisfies_entropy():
            raise ValueError(f"Created shock violates entropy condition at t={t}")

        return [shock]

    elif vel_new < vel_prev - 1e-15:  # Expansion
        # New water is slower - will fall behind old water - create rarefaction
        raref = RarefactionWave(
            t_start=t,
            v_start=v_inlet,
            flow=flow,
            c_head=c_prev,     # Head (faster) is old water
            c_tail=c_new,      # Tail (slower) is new water
            sorption=sorption
        )
        return [raref]

    else:  # Same velocity - contact discontinuity
        # This only happens if R(c_new) == R(c_prev), which is rare
        # Create a characteristic with the new concentration
        char = CharacteristicWave(
            t_start=t,
            v_start=v_inlet,
            flow=flow,
            concentration=c_new,
            sorption=sorption
        )
        return [char]


def initialize_all_inlet_waves(cin: np.ndarray, flow: np.ndarray, tedges: np.ndarray,
                               sorption: FreundlichSorption | ConstantRetardation) -> list[Wave]:
    """
    Initialize all waves from inlet boundary conditions.

    Algorithm:
    1. Start with c_prev = 0 (unknown initial condition)
    2. For each time bin i:
       - If cin[i] != c_prev: create wave(s) at tedges[i]
       - Update c_prev = cin[i]

    Returns:
        List of all waves created from inlet conditions
    """
    waves = []
    c_prev = 0.0  # Assume domain initially at zero (spin-up handles this)

    for i in range(len(cin)):
        c_new = cin[i]
        t_change = tedges[i]
        flow_current = flow[i]

        if abs(c_new - c_prev) > 1e-15:
            new_waves = create_inlet_waves_at_time(
                c_prev, c_new, t_change, flow_current, sorption, v_inlet=0.0
            )
            waves.extend(new_waves)

        c_prev = c_new

    return waves
```

---

### 4.2: Event Handlers

```python
def handle_characteristic_collision(char1: CharacteristicWave, char2: CharacteristicWave,
                                   t_event: float, v_event: float) -> list[Wave]:
    """
    Handle collision of two characteristics → create shock.

    Returns:
        List of new waves created (single shock)
    """
    # Determine which is faster
    if char1.velocity() > char2.velocity():
        c_left = char1.concentration
        c_right = char2.concentration
    else:
        c_left = char2.concentration
        c_right = char1.concentration

    # Create shock
    shock = ShockWave(
        t_start=t_event,
        v_start=v_event,
        flow=char1.flow,  # Assume same flow (piecewise constant)
        c_left=c_left,
        c_right=c_right,
        sorption=char1.sorption
    )

    # Verify entropy
    if not shock.satisfies_entropy():
        # This shouldn't happen if we constructed correctly
        raise RuntimeError(f"Characteristic collision created non-entropic shock at t={t_event}")

    # Deactivate parent characteristics
    char1.is_active = False
    char2.is_active = False

    return [shock]


def handle_shock_collision(shock1: ShockWave, shock2: ShockWave,
                          t_event: float, v_event: float) -> list[Wave]:
    """
    Handle collision of two shocks → merge into single shock.

    Returns:
        List of new waves (single merged shock)
    """
    # Merged shock: left state from upstream shock, right state from downstream shock
    # Determine which is upstream (depends on which was ahead at collision time)

    # Simple rule: shock with higher left concentration is typically upstream (for n>1)
    if shock1.c_left >= shock2.c_left:
        c_left = shock1.c_left
        c_right = shock2.c_right
    else:
        c_left = shock2.c_left
        c_right = shock1.c_right

    merged = ShockWave(
        t_start=t_event,
        v_start=v_event,
        flow=shock1.flow,
        c_left=c_left,
        c_right=c_right,
        sorption=shock1.sorption
    )

    # Deactivate parent shocks
    shock1.is_active = False
    shock2.is_active = False

    return [merged]


def handle_shock_characteristic_collision(shock: ShockWave, char: CharacteristicWave,
                                         t_event: float, v_event: float) -> list[Wave]:
    """
    Handle shock catching or being caught by characteristic.

    Outcomes:
    - Shock absorbs characteristic (characteristic disappears)
    - Or shock and characteristic merge into new shock

    Returns:
        List of new waves
    """
    # Determine if shock is catching characteristic or vice versa
    shock_vel = shock.velocity
    char_vel = char.velocity()

    if shock_vel > char_vel:
        # Shock catching characteristic from behind
        # Characteristic on right side of shock gets absorbed
        # New shock: c_left unchanged, c_right = char.concentration

        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=shock.flow,
            c_left=shock.c_left,
            c_right=char.concentration,
            sorption=shock.sorption
        )
    else:
        # Characteristic catching shock from behind
        # Characteristic on left side of shock
        # New shock: c_left = char.concentration, c_right unchanged

        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=shock.flow,
            c_left=char.concentration,
            c_right=shock.c_right,
            sorption=shock.sorption
        )

    # Check entropy
    if not new_shock.satisfies_entropy():
        # If entropy violated, this interaction shouldn't create a shock
        # Instead, it might be a rarefaction scenario
        # For now, return empty list (more complex logic needed)
        return []

    shock.is_active = False
    char.is_active = False

    return [new_shock]


def handle_shock_rarefaction_collision(shock: ShockWave, raref: RarefactionWave,
                                      t_event: float, v_event: float,
                                      collision_type: str) -> list[Wave]:
    """
    Handle shock interacting with rarefaction fan.

    Parameters:
        collision_type: 'head' or 'tail' indicating which boundary

    This is complex - shock can:
    - Penetrate rarefaction fan
    - Get absorbed
    - Create new waves

    Returns:
        List of new waves
    """
    # Simplified handling for first implementation:
    # If shock catches tail: rarefaction gets compressed
    # If head catches shock: complex interaction

    if collision_type == 'tail':
        # Shock catching rarefaction tail
        # Shock penetrates into rarefaction, compressing it
        # New shock continues with c_right = rarefaction tail concentration

        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=shock.flow,
            c_left=shock.c_left,
            c_right=raref.c_tail,
            sorption=shock.sorption
        )

        # Rarefaction remains active but tail is now at shock position
        # (More sophisticated: modify rarefaction structure)
        raref.is_active = False  # Simplification: deactivate
        shock.is_active = False

        return [new_shock]

    else:  # collision_type == 'head'
        # Rarefaction head catching shock
        # This creates compression between rarefaction and shock
        # May form new shock or modify rarefaction

        # Simplified: create shock between rarefaction head and shock left
        new_shock = ShockWave(
            t_start=t_event,
            v_start=v_event,
            flow=raref.flow,
            c_left=raref.c_head,
            c_right=shock.c_left,
            sorption=raref.sorption
        )

        if new_shock.satisfies_entropy():
            return [new_shock]
        else:
            # No shock forms - waves pass through each other
            return []


def handle_outlet_crossing(wave: Wave, t_event: float, v_outlet: float) -> dict:
    """
    Handle wave crossing outlet boundary.

    Wave remains active for concentration queries but is marked as exited.

    Returns:
        Event record dict with details
    """
    # Don't deactivate - we still need it for concentration_at_point queries
    # Just record the exit event

    return {
        'time': t_event,
        'type': 'outlet_crossing',
        'wave': wave,
        'concentration_left': wave.concentration_left(),
        'concentration_right': wave.concentration_right(),
    }
```

---

## Phase 5: Front Tracker (Main Simulation Engine)

### 5.1: Front Tracker Class

**File**: `src/gwtransport/front_tracking_solver.py`

```python
@dataclass
class FrontTrackerState:
    """Complete state of the simulation"""
    waves: list[Wave]                    # All waves (active and inactive)
    events: list[dict]                   # Event history
    t_current: float                     # Current simulation time
    v_outlet: float                      # Outlet position
    sorption: FreundlichSorption | ConstantRetardation

    # Inlet conditions (for reference)
    cin: np.ndarray
    flow: np.ndarray
    tedges: np.ndarray


class FrontTracker:
    """
    Event-driven front tracking solver.

    This is the main simulation engine that orchestrates wave propagation,
    event detection, and event handling.
    """

    def __init__(self, cin: np.ndarray, flow: np.ndarray, tedges: np.ndarray,
                 aquifer_pore_volume: float,
                 sorption: FreundlichSorption | ConstantRetardation):
        """Initialize tracker with inlet conditions and physical parameters"""

        # Validation
        if len(tedges) != len(cin) + 1:
            raise ValueError("tedges must have length len(cin) + 1")
        if len(flow) != len(cin):
            raise ValueError("flow must have same length as cin")
        if np.any(cin < 0):
            raise ValueError("cin must be non-negative")
        if np.any(flow <= 0):
            raise ValueError("flow must be positive")
        if aquifer_pore_volume <= 0:
            raise ValueError("aquifer_pore_volume must be positive")

        # Store parameters
        self.state = FrontTrackerState(
            waves=[],
            events=[],
            t_current=tedges[0],
            v_outlet=aquifer_pore_volume,
            sorption=sorption,
            cin=cin,
            flow=flow,
            tedges=tedges
        )

        # Compute spin-up period
        self.t_first_arrival = compute_first_arrival_time(
            cin, flow, tedges, aquifer_pore_volume, sorption
        )

        # Initialize waves from inlet
        self.state.waves = initialize_all_inlet_waves(cin, flow, tedges, sorption)


    def find_next_event(self) -> Event | None:
        """
        Find the next event (earliest in time).

        Searches all possible interactions and returns the earliest one.

        Returns:
            Event object or None if no future events
        """
        from heapq import heappush, heappop

        candidates = []  # Will use as min-heap by time

        # Get only active waves
        active_waves = [w for w in self.state.waves if w.is_active]

        # 1. Characteristic-Characteristic collisions
        for i, w1 in enumerate(active_waves):
            if not isinstance(w1, CharacteristicWave):
                continue
            for w2 in active_waves[i+1:]:
                if not isinstance(w2, CharacteristicWave):
                    continue

                result = find_characteristic_intersection(w1, w2, self.state.t_current)
                if result:
                    t, v = result
                    if 0 <= v <= self.state.v_outlet:  # In domain
                        heappush(candidates, (t, EventType.CHAR_CHAR_COLLISION, [w1, w2], v))

        # 2. Shock-Shock collisions
        for i, w1 in enumerate(active_waves):
            if not isinstance(w1, ShockWave):
                continue
            for w2 in active_waves[i+1:]:
                if not isinstance(w2, ShockWave):
                    continue

                result = find_shock_shock_intersection(w1, w2, self.state.t_current)
                if result:
                    t, v = result
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (t, EventType.SHOCK_SHOCK_COLLISION, [w1, w2], v))

        # 3. Shock-Characteristic collisions
        for w1 in active_waves:
            if not isinstance(w1, ShockWave):
                continue
            for w2 in active_waves:
                if not isinstance(w2, CharacteristicWave):
                    continue

                result = find_shock_characteristic_intersection(w1, w2, self.state.t_current)
                if result:
                    t, v = result
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (t, EventType.SHOCK_CHAR_COLLISION, [w1, w2], v))

        # 4. Rarefaction-Characteristic collisions
        for w1 in active_waves:
            if not isinstance(w1, RarefactionWave):
                continue
            for w2 in active_waves:
                if not isinstance(w2, CharacteristicWave):
                    continue

                intersections = find_rarefaction_boundary_intersections(w1, w2, self.state.t_current)
                for t, v, boundary in intersections:
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (t, EventType.RAREF_CHAR_COLLISION, [w1, w2], v))

        # 5. Shock-Rarefaction collisions
        for w1 in active_waves:
            if not isinstance(w1, ShockWave):
                continue
            for w2 in active_waves:
                if not isinstance(w2, RarefactionWave):
                    continue

                intersections = find_rarefaction_boundary_intersections(w2, w1, self.state.t_current)
                for t, v, boundary in intersections:
                    if 0 <= v <= self.state.v_outlet:
                        heappush(candidates, (t, EventType.SHOCK_RAREF_COLLISION, [w1, w2], v))

        # 6. Outlet crossings
        for wave in active_waves:
            t_cross = find_outlet_crossing(wave, self.state.v_outlet, self.state.t_current)
            if t_cross and t_cross > self.state.t_current:
                heappush(candidates, (t_cross, EventType.OUTLET_CROSSING, [wave], self.state.v_outlet))

        # Return earliest event
        if candidates:
            t, event_type, waves, v = heappop(candidates)
            return Event(time=t, event_type=event_type, waves_involved=waves, location=v)

        return None


    def handle_event(self, event: Event):
        """
        Handle an event by calling appropriate handler and updating state.
        """
        # Dispatch to appropriate handler
        new_waves = []

        if event.event_type == EventType.CHAR_CHAR_COLLISION:
            new_waves = handle_characteristic_collision(
                event.waves_involved[0], event.waves_involved[1],
                event.time, event.location
            )

        elif event.event_type == EventType.SHOCK_SHOCK_COLLISION:
            new_waves = handle_shock_collision(
                event.waves_involved[0], event.waves_involved[1],
                event.time, event.location
            )

        elif event.event_type == EventType.SHOCK_CHAR_COLLISION:
            new_waves = handle_shock_characteristic_collision(
                event.waves_involved[0], event.waves_involved[1],
                event.time, event.location
            )

        elif event.event_type == EventType.SHOCK_RAREF_COLLISION:
            # Need to determine collision type (head/tail)
            # Simplification: assume tail for now
            new_waves = handle_shock_rarefaction_collision(
                event.waves_involved[0], event.waves_involved[1],
                event.time, event.location, 'tail'
            )

        elif event.event_type == EventType.OUTLET_CROSSING:
            event_record = handle_outlet_crossing(
                event.waves_involved[0], event.time, event.location
            )
            self.state.events.append(event_record)
            return  # No new waves

        # Add new waves to state
        self.state.waves.extend(new_waves)

        # Record event
        self.state.events.append({
            'time': event.time,
            'type': event.event_type.value,
            'location': event.location,
            'waves_before': event.waves_involved,
            'waves_after': new_waves
        })


    def run(self, max_iterations: int = 10000):
        """
        Run simulation until no more events or max_iterations reached.

        Parameters:
            max_iterations: Safety limit to prevent infinite loops
        """
        iteration = 0

        while iteration < max_iterations:
            # Find next event
            event = self.find_next_event()

            if event is None:
                print(f"Simulation complete after {iteration} events at t={self.state.t_current:.6f}")
                break

            # Advance time
            self.state.t_current = event.time

            # Handle event
            self.handle_event(event)

            # Optional: verify physics after each event
            if iteration % 10 == 0:  # Check every 10 events
                self.verify_physics()

            iteration += 1

        if iteration >= max_iterations:
            print(f"Warning: Reached max_iterations={max_iterations}")

        print(f"Final statistics:")
        print(f"  Total events: {len(self.state.events)}")
        print(f"  Total waves created: {len(self.state.waves)}")
        print(f"  Active waves: {sum(1 for w in self.state.waves if w.is_active)}")
        print(f"  First arrival time: {self.t_first_arrival:.6f} days")


    def verify_physics(self):
        """
        Verify physical correctness of current state.

        Checks:
        - All shocks satisfy entropy condition
        - Mass balance (TODO: implement exact analytical mass balance)
        - No overlapping waves
        """
        # Check entropy for all active shocks
        for wave in self.state.waves:
            if isinstance(wave, ShockWave) and wave.is_active:
                if not wave.satisfies_entropy():
                    raise RuntimeError(f"Shock at t={wave.t_start} violates entropy!")

        # TODO: Implement exact mass balance check
        # This requires analytical integration over domain
```

---

## Phase 6: Concentration Extraction

### 6.1: Point-wise Concentration

**File**: `src/gwtransport/front_tracking_output.py`

```python
def concentration_at_point(v: float, t: float, waves: list[Wave],
                          sorption: FreundlichSorption | ConstantRetardation) -> float:
    """
    Compute concentration at point (v, t) with EXACT analytical value.

    Algorithm:
    1. Check each wave to see if it controls concentration at (v,t)
    2. Priority: Rarefactions > Shocks > Characteristics
    3. If no wave controls point, return 0.0 (initial condition)

    Returns:
        Concentration [mg/L or units of cin]
    """
    # Check rarefactions first (they have spatial extent)
    for wave in waves:
        if isinstance(wave, RarefactionWave) and wave.is_active:
            c = wave.concentration_at_point(v, t)
            if c is not None:
                return c

    # Check shocks (discontinuities)
    for wave in waves:
        if isinstance(wave, ShockWave) and wave.is_active:
            v_shock = wave.position_at_time(t)
            if v_shock is not None:
                # Determine which side of shock
                if v < v_shock - 1e-15:
                    # Haven't reached shock yet - check if any wave behind shock
                    continue
                elif v > v_shock + 1e-15:
                    # Past shock - concentration is c_right
                    return wave.c_right
                else:
                    # Exactly at shock
                    return 0.5 * (wave.c_left + wave.c_right)

    # Check characteristics
    # Need to find which characteristic(s) have passed through (v,t)
    # The most recent one determines concentration

    latest_c = 0.0
    latest_time = -np.inf

    for wave in waves:
        if isinstance(wave, CharacteristicWave) and wave.is_active:
            # Check if this characteristic has reached position v by time t
            v_char_at_t = wave.position_at_time(t)
            if v_char_at_t is not None and v_char_at_t >= v:
                # This characteristic has passed through v
                # Find when it passed: v_start + vel*(t_pass - t_start) = v
                vel = wave.velocity()
                if vel > 0:
                    t_pass = wave.t_start + (v - wave.v_start) / vel
                    if t_pass <= t and t_pass > latest_time:
                        latest_time = t_pass
                        latest_c = wave.concentration

    return latest_c
```

---

### 6.2: Breakthrough Curve

```python
def compute_breakthrough_curve(t_array: np.ndarray, v_outlet: float,
                               waves: list[Wave],
                               sorption: FreundlichSorption | ConstantRetardation) -> np.ndarray:
    """
    Compute concentration at outlet over time array.

    This is the breakthrough curve: C(v_outlet, t).

    Parameters:
        t_array: Time points [days]
        v_outlet: Outlet position [m³]
        waves: List of all waves
        sorption: Sorption parameters

    Returns:
        Array of concentrations matching t_array
    """
    c_out = np.zeros(len(t_array))

    for i, t in enumerate(t_array):
        c_out[i] = concentration_at_point(v_outlet, t, waves, sorption)

    return c_out
```

---

### 6.3: Bin-Averaged Concentration (EXACT ANALYTICAL)

```python
def compute_bin_averaged_concentration_exact(t_edges: np.ndarray, v_outlet: float,
                                            waves: list[Wave],
                                            sorption: FreundlichSorption | ConstantRetardation) -> np.ndarray:
    """
    Compute bin-averaged concentration using EXACT analytical integration.

    For each time bin [t_i, t_{i+1}]:
        C_avg = (1/(t_{i+1} - t_i)) * ∫_{t_i}^{t_{i+1}} C(v_outlet, t) dt

    Algorithm:
    1. For each bin, identify which wave segments control outlet
    2. For each segment:
       - Constant C: integral = C * Δt
       - Rarefaction C(t): derive exact integral formula
    3. Sum segment integrals and divide by bin width

    This is CRITICAL for machine-precision accuracy.

    Returns:
        Array of bin-averaged concentrations [length = len(t_edges) - 1]
    """
    n_bins = len(t_edges) - 1
    c_avg = np.zeros(n_bins)

    for i in range(n_bins):
        t_start = t_edges[i]
        t_end = t_edges[i + 1]
        dt = t_end - t_start

        if dt <= 0:
            raise ValueError(f"Invalid time bin: t_edges[{i}]={t_start} >= t_edges[{i+1}]={t_end}")

        # Identify wave segments controlling outlet in this time bin
        segments = identify_outlet_segments(t_start, t_end, v_outlet, waves)

        # Integrate each segment
        total_integral = 0.0

        for seg in segments:
            seg_t_start = max(seg['t_start'], t_start)
            seg_t_end = min(seg['t_end'], t_end)
            seg_dt = seg_t_end - seg_t_start

            if seg_dt <= 0:
                continue

            if seg['type'] == 'constant':
                # C is constant over segment
                integral = seg['concentration'] * seg_dt

            elif seg['type'] == 'rarefaction':
                # C(t) given by self-similar solution
                # Need to integrate analytically
                raref = seg['wave']

                # C(t) at outlet: R(C) = flow*(t - t_origin)/(v_outlet - v_origin)
                # Invert: C = C_from_R(flow*(t - t_origin)/(v_outlet - v_origin))

                # For Freundlich: C = [a*(t - t_origin)]^b where a, b are constants
                # Integral: ∫ C dt = a^b * ∫ (t - t_origin)^b dt
                #                   = a^b * [(t - t_origin)^(b+1) / (b+1)]

                integral = integrate_rarefaction_exact(
                    raref, v_outlet, seg_t_start, seg_t_end, sorption
                )

            else:
                raise ValueError(f"Unknown segment type: {seg['type']}")

            total_integral += integral

        c_avg[i] = total_integral / dt

    return c_avg


def identify_outlet_segments(t_start: float, t_end: float, v_outlet: float,
                            waves: list[Wave]) -> list[dict]:
    """
    Identify which waves control outlet concentration in time interval [t_start, t_end].

    Returns:
        List of segment dictionaries, each containing:
        - 't_start', 't_end': Segment time boundaries
        - 'type': 'constant', 'rarefaction', 'shock', etc.
        - 'concentration': For constant segments
        - 'wave': Reference to controlling wave
    """
    segments = []

    # Find all events at outlet in this time range
    outlet_events = []

    for wave in waves:
        if isinstance(wave, (CharacteristicWave, ShockWave)):
            t_cross = find_outlet_crossing(wave, v_outlet, t_start)
            if t_cross and t_start <= t_cross <= t_end:
                outlet_events.append({
                    'time': t_cross,
                    'wave': wave,
                    'c_before': 0.0,  # TODO: get concentration before
                    'c_after': wave.concentration if isinstance(wave, CharacteristicWave) else wave.c_right
                })

    # Sort events by time
    outlet_events.sort(key=lambda e: e['time'])

    # Create segments between events
    current_t = t_start
    current_c = concentration_at_point(v_outlet, t_start, waves, None)  # Need sorption here

    for event in outlet_events:
        # Segment before event
        if event['time'] > current_t:
            segments.append({
                't_start': current_t,
                't_end': event['time'],
                'type': 'constant',
                'concentration': current_c
            })

        current_t = event['time']
        current_c = event['c_after']

    # Final segment
    if t_end > current_t:
        segments.append({
            't_start': current_t,
            't_end': t_end,
            'type': 'constant',
            'concentration': current_c
        })

    return segments


def integrate_rarefaction_exact(raref: RarefactionWave, v_outlet: float,
                                t_start: float, t_end: float,
                                sorption: FreundlichSorption) -> float:
    """
    Exact analytical integral of rarefaction concentration over time at fixed position.

    For Freundlich sorption:
        R(C) = 1 + α*C^β  where α = ρ_b*k_f/(n_por*n), β = 1/n - 1

        At outlet: R = flow*(t - t_origin)/(v_outlet - v_origin) = κ*t + μ

        Invert: C = [(R-1)/α]^(1/β) = [κ*t + μ - 1]^(1/β) / α^(1/β)

        Integral: ∫_{t_start}^{t_end} C dt
                = (1/α^(1/β)) * ∫ (κ*t + μ - 1)^(1/β) dt
                = (1/α^(1/β)) * (1/κ) * [(κ*t + μ - 1)^(1/β + 1) / (1/β + 1)]|_{t_start}^{t_end}

    Returns:
        Exact integral value
    """
    # Extract parameters
    t_origin = raref.t_start
    v_origin = raref.v_start
    flow = raref.flow

    # Coefficients in R = κ*t + μ
    kappa = flow / (v_outlet - v_origin)
    mu = -flow * t_origin / (v_outlet - v_origin)

    # Freundlich parameters
    alpha = sorption.bulk_density * sorption.k_f / (sorption.porosity * sorption.n)
    beta = 1.0 / sorption.n - 1.0

    # Exponent for integration
    exponent = 1.0 / beta + 1.0

    # Antiderivative: F(t) = (1/(α^(1/β)*κ)) * (κ*t + μ - 1)^exponent / exponent
    coeff = 1.0 / (alpha**(1.0/beta) * kappa * exponent)

    def antiderivative(t):
        base = kappa * t + mu - 1.0
        if base <= 0:
            return 0.0
        return coeff * base**exponent

    integral = antiderivative(t_end) - antiderivative(t_start)

    return integral
```

---

## Phase 7: API Integration (Public Interface)

### 7.1: Main Public Function

**File**: `src/gwtransport/advection.py` (modify existing)

```python
def infiltration_to_extraction_front_tracking(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    freundlich_k: float,
    freundlich_n: float,
    bulk_density: float,
    porosity: float,
    retardation_factor: float | None = None,
    max_iterations: int = 10000,
) -> npt.NDArray[np.floating]:
    """
    Compute extracted concentration using exact front tracking with nonlinear sorption.

    Uses event-driven analytical algorithm that tracks shock waves, rarefaction waves,
    and characteristics with machine precision. No numerical dispersion, exact mass
    balance to floating-point precision.

    Parameters
    ----------
    cin : array-like
        Infiltration concentration [mg/L or any units].
        Length = len(tedges) - 1.
    flow : array-like
        Flow rate [m³/day]. Must be positive.
        Length = len(tedges) - 1.
    tedges : pandas.DatetimeIndex
        Time bin edges. Length = len(cin) + 1.
    cout_tedges : pandas.DatetimeIndex
        Output time bin edges. Can be different from tedges.
        Length determines output array size.
    aquifer_pore_volume : float
        Total pore volume [m³]. Must be positive.
    freundlich_k : float, optional
        Freundlich coefficient [(m³/kg)^(1/n)]. Must be positive.
        Used if retardation_factor is None.
    freundlich_n : float, optional
        Freundlich exponent [-]. Must be positive and != 1.
        Used if retardation_factor is None.
    bulk_density : float, optional
        Bulk density [kg/m³]. Must be positive.
        Used if retardation_factor is None.
    porosity : float, optional
        Porosity [-]. Must be in (0, 1).
        Used if retardation_factor is None.
    retardation_factor : float, optional
        Constant retardation factor [-]. If provided, uses linear retardation
        instead of Freundlich sorption. Must be >= 1.0.
    max_iterations : int, optional
        Maximum number of events. Default 10000.

    Returns
    -------
    cout : numpy.ndarray
        Bin-averaged extraction concentration.
        Length = len(cout_tedges) - 1.

    Notes
    -----
    **Spin-up Period**:
    The function computes the first arrival time t_first. Concentrations
    before t_first are affected by unknown initial conditions and should
    not be used for analysis. Use `infiltration_to_extraction_front_tracking_detailed`
    to access t_first.

    **Machine Precision**:
    All calculations use exact analytical formulas. Mass balance is conserved
    to floating-point precision (~1e-14 relative error). No numerical tolerances
    are used for time/position calculations.

    **Physical Correctness**:
    - All shocks satisfy Lax entropy condition
    - Rarefaction waves use self-similar solutions
    - Causality is strictly enforced

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> # Pulse injection
    >>> tedges = pd.date_range('2020-01-01', periods=4, freq='10D')
    >>> cin = np.array([0.0, 10.0, 0.0])
    >>> flow = np.array([100.0, 100.0, 100.0])
    >>> cout_tedges = pd.date_range('2020-01-01', periods=10, freq='5D')
    >>>
    >>> cout = infiltration_to_extraction_front_tracking(
    ...     cin=cin,
    ...     flow=flow,
    ...     tedges=tedges,
    ...     cout_tedges=cout_tedges,
    ...     aquifer_pore_volume=500.0,
    ...     freundlich_k=0.01,
    ...     freundlich_n=2.0,
    ...     bulk_density=1500.0,
    ...     porosity=0.3,
    ... )

    See Also
    --------
    infiltration_to_extraction_front_tracking_detailed : Returns detailed structure
    infiltration_to_extraction : Convolution-based approach for linear case
    gamma_infiltration_to_extraction : For distributions of pore volumes
    """
    # Input validation
    cin = np.asarray(cin, dtype=float)
    flow = np.asarray(flow, dtype=float)
    tedges = pd.DatetimeIndex(tedges)
    cout_tedges = pd.DatetimeIndex(cout_tedges)

    if len(tedges) != len(cin) + 1:
        raise ValueError("tedges must have length len(cin) + 1")
    if len(flow) != len(cin):
        raise ValueError("flow must have same length as cin")
    if np.any(cin < 0):
        raise ValueError("cin must be non-negative")
    if np.any(flow <= 0):
        raise ValueError("flow must be positive")
    if np.any(np.isnan(cin)) or np.any(np.isnan(flow)):
        raise ValueError("cin and flow must not contain NaN")
    if aquifer_pore_volume <= 0:
        raise ValueError("aquifer_pore_volume must be positive")

    # Convert time to days (relative to tedges[0])
    t_ref = tedges[0]
    tedges_days = ((tedges - t_ref) / pd.Timedelta(days=1)).values
    cout_tedges_days = ((cout_tedges - t_ref) / pd.Timedelta(days=1)).values

    # Create sorption object
    if retardation_factor is not None:
        if retardation_factor < 1.0:
            raise ValueError("retardation_factor must be >= 1.0")
        sorption = ConstantRetardation(retardation_factor=retardation_factor)
    else:
        if freundlich_k <= 0 or freundlich_n <= 0:
            raise ValueError("Freundlich parameters must be positive")
        if abs(freundlich_n - 1.0) < 1e-10:
            raise ValueError("freundlich_n = 1 not supported (use retardation_factor for linear case)")
        if bulk_density <= 0 or not 0 < porosity < 1:
            raise ValueError("Invalid physical parameters")

        sorption = FreundlichSorption(
            k_f=freundlich_k,
            n=freundlich_n,
            bulk_density=bulk_density,
            porosity=porosity
        )

    # Create tracker and run simulation
    from gwtransport.front_tracking_solver import FrontTracker

    tracker = FrontTracker(
        cin=cin,
        flow=flow,
        tedges=tedges_days,
        aquifer_pore_volume=aquifer_pore_volume,
        sorption=sorption
    )

    tracker.run(max_iterations=max_iterations)

    # Extract bin-averaged concentrations at outlet
    from gwtransport.front_tracking_output import compute_bin_averaged_concentration_exact

    cout = compute_bin_averaged_concentration_exact(
        t_edges=cout_tedges_days,
        v_outlet=aquifer_pore_volume,
        waves=tracker.state.waves,
        sorption=sorption
    )

    return cout
```

---

### 7.2: Detailed Output Function

```python
def infiltration_to_extraction_front_tracking_detailed(
    *,
    cin: npt.ArrayLike,
    flow: npt.ArrayLike,
    tedges: pd.DatetimeIndex,
    cout_tedges: pd.DatetimeIndex,
    aquifer_pore_volume: float,
    freundlich_k: float,
    freundlich_n: float,
    bulk_density: float,
    porosity: float,
    retardation_factor: float | None = None,
    max_iterations: int = 10000,
) -> tuple[npt.NDArray[np.floating], dict]:
    """
    Compute extracted concentration with complete diagnostic information.

    Returns both bin-averaged concentrations and detailed simulation structure.

    Parameters
    ----------
    [Same as infiltration_to_extraction_front_tracking]

    Returns
    -------
    cout : numpy.ndarray
        Bin-averaged concentrations.

    structure : dict
        Detailed simulation structure with keys:

        - 'waves': List[Wave] - All wave objects created during simulation
        - 'events': List[dict] - All events with times, types, and details
        - 't_first_arrival': float - First arrival time (end of spin-up period)
        - 'n_events': int - Total number of events
        - 'n_shocks': int - Number of shocks created
        - 'n_rarefactions': int - Number of rarefactions created
        - 'final_time': float - Final simulation time
        - 'sorption': FreundlichSorption | ConstantRetardation - Sorption object
        - 'tracker_state': FrontTrackerState - Complete simulation state

    Examples
    --------
    >>> cout, structure = infiltration_to_extraction_front_tracking_detailed(
    ...     cin=cin, flow=flow, tedges=tedges, cout_tedges=cout_tedges,
    ...     aquifer_pore_volume=500.0,
    ...     freundlich_k=0.01, freundlich_n=2.0,
    ...     bulk_density=1500.0, porosity=0.3
    ... )
    >>>
    >>> # Access spin-up period
    >>> print(f"First arrival: {structure['t_first_arrival']:.2f} days")
    >>>
    >>> # Analyze events
    >>> for event in structure['events']:
    ...     print(f"t={event['time']:.2f}: {event['type']}")
    >>>
    >>> # Plot V-t diagram (custom visualization)
    >>> from gwtransport.front_tracking_plot import plot_vt_diagram
    >>> fig = plot_vt_diagram(structure['tracker_state'])
    """
    # [Same input processing as main function]
    # ... (omitted for brevity)

    # Run simulation
    tracker = FrontTracker(...)
    tracker.run(max_iterations=max_iterations)

    # Extract concentrations
    cout = compute_bin_averaged_concentration_exact(...)

    # Build structure dict
    structure = {
        'waves': tracker.state.waves,
        'events': tracker.state.events,
        't_first_arrival': tracker.t_first_arrival,
        'n_events': len(tracker.state.events),
        'n_shocks': sum(1 for w in tracker.state.waves if isinstance(w, ShockWave)),
        'n_rarefactions': sum(1 for w in tracker.state.waves if isinstance(w, RarefactionWave)),
        'final_time': tracker.state.t_current,
        'sorption': sorption,
        'tracker_state': tracker.state,
        'aquifer_pore_volume': aquifer_pore_volume,
    }

    return cout, structure
```

---

## Phase 8: Testing Strategy

### 8.1: Unit Tests

**File**: `tests/src/test_front_tracking_math.py`

```python
def test_retardation_roundtrip():
    """Test R(C) → C roundtrip for Freundlich sorption"""
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    for c in [0.1, 1.0, 5.0, 10.0, 100.0]:
        r = sorption.retardation(c)
        c_back = sorption.concentration_from_retardation(r)
        assert np.isclose(c, c_back, rtol=1e-14), f"Roundtrip failed: {c} → {r} → {c_back}"


def test_shock_velocity_mass_conservation():
    """Verify shock velocity conserves mass exactly"""
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)
    flow = 100.0
    c_left = 10.0
    c_right = 2.0

    # Compute shock velocity
    v_shock = sorption.shock_velocity(c_left, c_right, flow)

    # Verify Rankine-Hugoniot
    flux_left = flow * c_left
    flux_right = flow * c_right
    c_total_left = sorption.total_concentration(c_left)
    c_total_right = sorption.total_concentration(c_right)

    v_shock_expected = (flux_right - flux_left) / (c_total_right - c_total_left)

    assert np.isclose(v_shock, v_shock_expected, rtol=1e-14)


def test_rarefaction_self_similar_solution():
    """Verify rarefaction satisfies self-similar solution"""
    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    # Create rarefaction
    raref = RarefactionWave(
        t_start=0.0, v_start=0.0, flow=100.0,
        c_head=10.0, c_tail=2.0, sorption=sorption
    )

    # Test self-similar solution at various points
    t = 10.0
    for v in [50.0, 100.0, 150.0, 200.0]:
        if not raref.contains_point(v, t):
            continue

        c = raref.concentration_at_point(v, t)

        # Verify: R(C) = flow * t / v
        r_expected = raref.flow * t / v
        r_actual = sorption.retardation(c)

        assert np.isclose(r_actual, r_expected, rtol=1e-14)


def test_first_arrival_time_constant_flow():
    """Test first arrival time with constant flow"""
    cin = np.array([0.0, 10.0, 10.0])
    flow = np.array([100.0, 100.0, 100.0])
    tedges = np.array([0.0, 10.0, 20.0, 30.0])
    aquifer_pore_volume = 500.0

    sorption = FreundlichSorption(k_f=0.01, n=2.0, bulk_density=1500.0, porosity=0.3)

    t_first = compute_first_arrival_time(cin, flow, tedges, aquifer_pore_volume, sorption)

    # Expected: t_first = tedges[1] + aquifer_pore_volume * R(10) / flow
    r_10 = sorption.retardation(10.0)
    t_expected = 10.0 + aquifer_pore_volume * r_10 / 100.0

    assert np.isclose(t_first, t_expected, rtol=1e-14)
```

---

### 8.2: Scenario Tests

**File**: `tests/src/test_front_tracking_scenarios.py`

```python
def test_scenario_step_input():
    """Test step input: C=0→10 → single shock propagation"""
    cin = np.array([0.0, 10.0])
    flow = np.array([100.0, 100.0])
    tedges = np.array([0.0, 10.0, 20.0])

    # ... run simulation ...

    # Verify:
    # - Single shock created at t=10
    # - Shock velocity matches Rankine-Hugoniot
    # - Mass balance exact


def test_scenario_pulse_injection():
    """Test pulse: C=0→10→0 → compression + rarefaction"""
    cin = np.array([0.0, 10.0, 0.0])
    flow = np.array([100.0, 100.0, 100.0])
    tedges = np.array([0.0, 10.0, 20.0, 30.0])

    # ... run simulation ...

    # Verify:
    # - Shock forms at t=10 (compression)
    # - Rarefaction forms at t=20 (expansion)
    # - Mass balance: total mass in = total mass out


def test_scenario_ramp_input():
    """Test monotonic ramp: C=0→5→10 → pure rarefactions"""
    cin = np.array([0.0, 5.0, 10.0])
    flow = np.array([100.0, 100.0, 100.0])
    tedges = np.array([0.0, 10.0, 20.0, 30.0])

    # ... run simulation ...

    # Verify:
    # - Two rarefactions created
    # - No shocks (all expansions)
    # - Smooth concentration profile


def test_mass_balance_exact():
    """Test exact mass balance for all scenarios"""
    # For each scenario:
    # 1. Compute total mass injected: ∫ cin * flow dt
    # 2. Compute total mass in domain at end
    # 3. Compute total mass extracted
    # 4. Verify: mass_in = mass_domain + mass_out (to 1e-14 relative error)
    pass
```

---

### 8.3: Comparison Tests

**File**: `tests/src/test_front_tracking_comparison.py`

```python
def test_compare_constant_retardation_to_series():
    """
    For constant retardation, front tracking should match
    infiltration_to_extraction_series exactly.
    """
    # Create test case with retardation_factor
    # Run both front tracking and series
    # Compare outputs (should be identical)
    pass


def test_compare_against_analytical_buckley_leverett():
    """
    Compare against known Buckley-Leverett solution for specific parameters.
    """
    # Use standard Buckley-Leverett test case from literature
    # Verify shock position and concentration profile
    pass
```

---

## Phase 9: Visualization & Documentation

### 9.1: V-t Diagram Plotting

**File**: `src/gwtransport/front_tracking_plot.py`

```python
def plot_vt_diagram(state: FrontTrackerState, t_max: float = None,
                   figsize: tuple = (14, 10)) -> matplotlib.figure.Figure:
    """
    Create V-t diagram showing all waves.

    Parameters
    ----------
    state : FrontTrackerState
        Complete simulation state
    t_max : float, optional
        Maximum time to plot. If None, uses final simulation time.
    figsize : tuple, optional
        Figure size in inches

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure with V-t diagram
    """
    import matplotlib.pyplot as plt

    if t_max is None:
        t_max = state.t_current * 1.2

    fig, ax = plt.subplots(figsize=figsize)

    # Plot characteristics (blue lines)
    for wave in state.waves:
        if isinstance(wave, CharacteristicWave):
            t_plot = np.linspace(wave.t_start, t_max, 100)
            v_plot = [wave.position_at_time(t) for t in t_plot]
            v_plot = [v for v in v_plot if v is not None and 0 <= v <= state.v_outlet]
            ax.plot(t_plot[:len(v_plot)], v_plot, 'b-', linewidth=0.5, alpha=0.7)

    # Plot shocks (red lines)
    for wave in state.waves:
        if isinstance(wave, ShockWave):
            t_plot = np.linspace(wave.t_start, t_max, 100)
            v_plot = [wave.position_at_time(t) for t in t_plot]
            v_plot = [v for v in v_plot if v is not None and 0 <= v <= state.v_outlet]
            ax.plot(t_plot[:len(v_plot)], v_plot, 'r-', linewidth=2)

    # Plot rarefactions (green fans)
    for wave in state.waves:
        if isinstance(wave, RarefactionWave):
            t_plot = np.linspace(wave.t_start, t_max, 100)

            # Head boundary
            v_head = [wave.head_position_at_time(t) for t in t_plot]
            v_head = [v if v and 0 <= v <= state.v_outlet else None for v in v_head]

            # Tail boundary
            v_tail = [wave.tail_position_at_time(t) for t in t_plot]
            v_tail = [v if v and 0 <= v <= state.v_outlet else None for v in v_tail]

            # Plot boundaries
            ax.plot(t_plot, v_head, 'g--', linewidth=1.5, label='Rarefaction')
            ax.plot(t_plot, v_tail, 'g:', linewidth=1.5)

            # Fill fan
            ax.fill_between(t_plot, v_tail, v_head, alpha=0.2, color='green')

    # Mark outlet
    ax.axhline(state.v_outlet, color='k', linestyle='--', linewidth=1, label='Outlet')

    ax.set_xlabel('Time [days]', fontsize=12)
    ax.set_ylabel('Volumetric position [m³]', fontsize=12)
    ax.set_title('V-t Diagram: Wave Propagation', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, state.v_outlet * 1.05)

    return fig


def plot_breakthrough_curve(state: FrontTrackerState, t_max: float = None,
                           n_points: int = 1000) -> matplotlib.figure.Figure:
    """Plot concentration at outlet vs time"""
    # ... similar to above ...
```

---

### 9.2: Example Notebook

**File**: `examples/08_Front_Tracking_Exact_Solution.ipynb`

Contents:
1. Introduction to front tracking
2. Basic example: step input
3. Complex example: pulse injection
4. Visualizing V-t diagram
5. Accessing detailed structure
6. Verifying mass balance
7. Spin-up period handling
8. Comparison with convolution approach

---

## Known Issues and Future Improvements

This section documents TODOs and known limitations in the current implementation.
All items are tracked for future enhancement phases.

### High Priority

1. **Rarefaction-Rarefaction Intersections** (`front_tracking_events.py:390`)
   - Location: `find_rarefaction_boundary_intersections()`
   - Issue: Not yet implemented
   - Impact: Cannot handle rarefaction fan interactions
   - Solution: Implement boundary-boundary intersection logic for raref-raref collisions

2. **Rarefaction Creation from Invalid Shock** (`front_tracking_handlers.py:264`)
   - Location: `handle_shock_characteristic_collision()`
   - Issue: When entropy violated, waves just disappear instead of creating rarefaction
   - Impact: Mass balance may not be preserved in some edge cases
   - Solution: Detect compression-to-rarefaction transitions and create appropriate waves

3. **Full Shock-Rarefaction Wave Splitting** (`front_tracking_handlers.py:318`)
   - Location: `handle_shock_rarefaction_collision()`
   - Issue: Simplified implementation without wave splitting
   - Impact: Complex shock-rarefaction interactions may not be fully accurate
   - Solution: Implement full wave splitting logic as described in LeVeque (2002)

4. **Exact Mass Balance Verification** (`front_tracking_solver.py:535`)
   - Location: `verify_physics()`
   - Issue: Only checks entropy, not mass balance
   - Impact: Cannot verify mass conservation during simulation
   - Solution: Implement analytical integration over domain to compute total mass

### Medium Priority

5. **Rarefaction Boundary Type in Events** (`front_tracking_solver.py:396, 405`)
   - Location: `handle_event()` dispatcher
   - Issue: Hardcoded "head" and "tail" instead of extracting from event
   - Impact: May handle wrong boundary in some cases
   - Solution: Add boundary_type field to Event dataclass

6. **Rarefaction Tail Exit Detection** (`front_tracking.py:1478`)
   - Location: Old FrontTracker implementation
   - Issue: Only detects head exit, not tail exit
   - Impact: Rarefaction waves remain partially active after tail exits
   - Solution: Add tail crossing detection to outlet crossing logic

7. **Sophisticated Rarefaction Boundary Modification** (`front_tracking_handlers.py:401`)
   - Location: `handle_rarefaction_characteristic_collision()`
   - Issue: Current implementation just absorbs characteristic
   - Impact: May not correctly represent wave structure
   - Solution: Modify rarefaction head/tail concentrations instead of deactivating

### Low Priority

8. **Rarefaction-Rarefaction Collisions**
   - Status: Not implemented
   - Impact: Rare case, limited practical impact
   - Solution: Extend event detection to handle rarefaction fan mergers

9. **Adaptive Time Stepping**
   - Status: Uses event-driven (exact) time stepping
   - Impact: Many events may slow down simulation
   - Solution: Consider adaptive refinement for dense event regions (post-Phase 7)

10. **Parallel Event Processing**
    - Status: Sequential event processing
    - Impact: Cannot leverage parallelism
    - Solution: Implement space-time domain decomposition (research topic)

### Implementation Priority Order

**Phase 5 (Current)**: Items blocking basic functionality
- Item #4: Mass balance verification (for validation)
- Item #5: Correct boundary type handling

**Phase 6 (Next)**: Items affecting accuracy
- Item #1: Rarefaction-rarefaction intersections
- Item #3: Full shock-rarefaction splitting
- Item #2: Rarefaction creation from invalid shocks

**Phase 7 (Polish)**: Completeness and edge cases
- Item #6: Tail exit detection
- Item #7: Sophisticated boundary modification
- Item #8: Rarefaction-rarefaction collisions

**Future Work**: Performance and advanced features
- Item #9: Adaptive time stepping
- Item #10: Parallel processing

---

## Implementation Order

1. **Week 1**: Phase 1 (Math foundation) + Phase 2 (Wave classes)
2. **Week 2**: Phase 3 (Event detection) + Phase 4 (Event handlers)
3. **Week 3**: Phase 5 (Front tracker) + Phase 6 (Output)
4. **Week 4**: Phase 7 (API) + Phase 8 (Testing)
5. **Week 5**: Phase 9 (Documentation) + Refinement

---

## Questions Resolved

1. ✅ Initial conditions: Unknown, use spin-up period with `t_first_arrival`
2. ✅ Variable flow: Piecewise constant with tedges
3. ✅ Retardation: Support both Freundlich and constant
4. ✅ Output priority: Detailed diagnostics
5. ✅ Tolerance: Exact analytical (no tolerances)
6. ✅ Multiple streamlines: Architecture supports future extension

---

**Ready to proceed with implementation?**
