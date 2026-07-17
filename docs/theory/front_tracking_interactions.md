# Front-tracking wave interactions — the authoritative catalogue

This document is the reference truth table for the nonlinear-sorption / kinematic-wave
front tracker in `src/gwtransport/fronttracking/`. It answers two questions definitively:

1. **From the physics of scalar conservation laws, which wave–wave interactions are
   possible?** Every possible interaction must be represented and tested; every
   interaction the code _represents_ but that cannot physically occur must be confirmed
   non-existent and its dead path removed.
2. **Which interaction resolutions are exact vs numerical, and are the numerical ones
   acceptable?**

Every claim below is either derived from the governing law or verified numerically
against the source (`.venv-claude`); the verification scripts live in `scratchpad/` on the
`audit/294-interaction-completeness` branch. Line references are into the worktree source.

---

## 1. Governing law and elementary waves

### 1.1 The scalar conservation law

In cumulative-flow coordinates `θ = ∫ flow dt` and pore-volume position `V`, transport of a
single sorbing solute (or of water content in kinematic-wave percolation) is the **scalar
conservation law**

```
∂_θ C_T(c) + ∂_V c = 0
```

with conserved quantity `U = C_T(c)` (total = dissolved + sorbed, or storage `θ_m − θ_r`)
and flux `f(U) = c` (dissolved concentration, or hydraulic conductivity `K`). Flow drops out
entirely; the wave structure is a property of the isotherm alone (`math.py:62`
`shock_speed`, `math.py:1099` `characteristic_speed`).

The **characteristic (celerity)** is

```
λ(c) = df/dU = dc/dC_T = 1 / R(c) ,     R(c) = dC_T/dc  (the retardation factor),
```

so `λ = 1/R` (`math.py:1099`). Because flow is one-directional, `λ ≥ 0` always: `R ≥ 0`, with
`R = 0` only at full saturation (`S_e = 1` for vG-M/Brooks–Corey, where `dK/dS_e → ∞`), at
which `λ = +∞` — the `MAX_FINITE_SPEED` / `R = 0` edge special-cased at `math.py:94,147`.

A **shock** between states `c_L` (upstream/behind) and `c_R` (downstream/ahead) moves at the
Rankine–Hugoniot speed

```
dV_s/dθ = (c_R − c_L) / (C_T(c_R) − C_T(c_L))     (math.py:62)
```

and is admissible iff it satisfies the **Lax entropy condition** in θ-space,
`λ(c_L) ≥ dV_s/dθ ≥ λ(c_R)` (`math.py:123`). A **rarefaction** is the self-similar fan
`R(c) = (θ − θ_0)/(V − v_0)` connecting `c_tail` to `c_head` with `λ(c_head) > λ(c_tail)`
(`waves.py:439`; the class raises if head ≤ tail, i.e. a compression).

### 1.2 Genuine nonlinearity and flux curvature

The Riemann outcome for a scalar law is fixed by the **curvature of the flux**,
`f''(U)`. Differentiating `λ = 1/R`,

```
f''(U) = dλ/dU = d(1/R)/dc · dc/dU = (−R'/R²)·(1/R) = −R'(c) / R(c)³ ,
```

so

```
sign f''(U) = − sign R'(c) .        (verified: scratchpad/repro_section0.py)
```

Genuine nonlinearity (`f'' ≠ 0`) holds exactly where `R'(c) ≠ 0`. The flux is:

| flux class                       | condition                      | elementary waves             | Riemann outcome                            |
| -------------------------------- | ------------------------------ | ---------------------------- | ------------------------------------------ |
| genuinely nonlinear, **convex**  | `R' < 0` everywhere            | shock, rarefaction           | a single shock **or** a single rarefaction |
| genuinely nonlinear, **concave** | `R' > 0` everywhere            | shock, rarefaction (mirror)  | a single shock **or** a single rarefaction |
| **linearly degenerate**          | `R' ≡ 0` (ConstantRetardation) | contact only                 | contact                                    |
| **non-convex**                   | `R'` changes sign              | compound (shock+rarefaction) | Oleinik convex-hull construction           |

The last row is the only one that would require machinery beyond shocks and rarefactions.
**Section 2 proves it never occurs for any isotherm in this package.**

---

## 2. Per-isotherm classification — every flux is single-curvature

For each sorption/conductivity model the sign of `R'(c)` is derived and confirmed
numerically over the full concentration range. **Result: not one isotherm is non-convex.**

| isotherm (class, math.py)                                       | `R(c)`                                      | `R'` sign | flux class                | measured over range                 |
| --------------------------------------------------------------- | ------------------------------------------- | --------- | ------------------------- | ----------------------------------- |
| **Freundlich `n>1`** (`FreundlichSorption`:163)                 | `1 + (ρ_b k_f)/(φ n)·c^{1/n−1}`             | `R' < 0`  | **convex** (favorable)    | `R'` single-signed `<0`; `R ≥ 1`    |
| **Freundlich `n<1`**                                            | same, `1/n−1 > 0`                           | `R' > 0`  | **concave** (unfavorable) | `R'` single-signed `>0`; `R ≥ 1`    |
| **Langmuir** (`LangmuirSorption`:537)                           | `1 + A/(K_L+c)²`                            | `R' < 0`  | **convex** (favorable)    | `R'` single-signed `<0`; `R ≥ 1`    |
| **Constant** (`ConstantRetardation`:401)                        | `R = const`                                 | `R' ≡ 0`  | **linear / degenerate**   | contacts only, no fans              |
| **Brooks–Corey** (`BrooksCoreyConductivity`:730)                | `(Δθ/(a K_s))·(c/K_s)^{1/a−1}`, `a=3+2/λ>3` | `R' < 0`  | **convex** (favorable)    | `f ∝ U^a, a>3 ⇒ f''>0`; `R<1` real  |
| **van Genuchten–Mualem** (`VanGenuchtenMualemConductivity`:861) | `Δθ / (dK/dS_e)`                            | `R' < 0`  | **convex** (favorable)    | `d²K/dS_e² > 0` ∀ params (see §2.2) |

Two physical regimes must be kept distinct because the retardation constraint differs:

- **Solute sorption** (`R ≥ 1` always): Freundlich, Langmuir, Constant. `λ = 1/R ≤ 1`.
- **Unsaturated-zone percolation** (`R < 1` realistic): Brooks–Corey, vG-Mualem.
  Carsel–Parrish soils give `minR` down to `~2e-5` (coarse sand), i.e. celerity `λ = 1/R` up
  to `~4e4`. This is why the interaction detector marches with a **per-pair**
  Lipschitz bound `Λ = speed_bound_a + speed_bound_b` rather than a global `1`
  (`interactions.py:236` `find_face_crossing`, `:75` `_max_characteristic_speed`).

### 2.1 Freundlich, Langmuir, Constant, Brooks–Corey

- **Freundlich** `R(c) = 1 + (ρ_b k_f)/(φ n)·c^{1/n−1}`. The exponent `1/n−1` is negative for
  `n>1` (⇒ `R' < 0`, convex, higher `c` faster, `R→∞` as `c→0`) and positive for `n<1`
  (⇒ `R' > 0`, concave, higher `c` slower). Single-signed in both cases — no inflection.
- **Langmuir** `R(c) = 1 + A/(K_L+c)²` with `A = ρ_b s_max K_L/φ > 0`. `R'(c) = −2A/(K_L+c)³ < 0`
  for all `c ≥ 0`. Convex, favorable, `R(0)=1+A/K_L²` finite.
- **Constant** `R' ≡ 0`: linearly degenerate. Every characteristic speed equals every shock
  speed (`math.py:502`); only contacts exist, never fans.
- **Brooks–Corey** (recast): flux `f = c = K`, conserved `U = C_T = Δθ (c/K_s)^{1/a}`, so
  `c = K_s (U/Δθ)^a` with `a = 3 + 2/λ > 3`. Then `f(U) ∝ U^a` and `f''(U) ∝ a(a−1) U^{a−2} > 0`:
  strictly convex for every `λ > 0`. Equivalently `R' < 0` single-signed.

### 2.2 van Genuchten–Mualem is convex for ALL parameters (correction of a prior claim)

`K(S_e) = K_s · S_e^L · [1 − (1 − S_e^{1/m})^m]^2`, `m = 1 − 1/n_vG`, `n_vG > 1`, `L ≥ 0`.
With `U = C_T = Δθ·S_e` and `f = K`, the flux curvature is `f''(U) ∝ d²K/dS_e²`. The flux is
non-convex (compound waves possible) **iff `d²K/dS_e²` changes sign** on `S_e ∈ (0,1)`.

**High-precision result (`scratchpad/repro_vgm_stable.py`, dps=200): `d²K/dS_e² > 0` strictly,
for every `n_vG ∈ [1.001, 10]` and `L ∈ [0, 3]`.** The van Genuchten–Mualem flux is convex
everywhere; there are **no** compound waves.

A prior analysis reported vG-M as non-convex for clay (`n_vG ≈ 1.09`). That was a
**numerical artefact**, not physics:

- A naive `np.gradient(R(c))` (or double-differencing the brentq-based `C_T`) produces
  spurious `R'` / `d²K` sign flips near the singular endpoints (`R → ∞` as `c → 0`,
  `R → 0` as `c → K_s`).
- Any **low-precision differencing** of `K` near `S_e → 0` is unreliable because
  `U = 1 − (1 − S_e^{1/m})^m` suffers **catastrophic cancellation** when `S_e^{1/m}`
  underflows the working precision (e.g. for `n_vG = 1.09`, `m ≈ 0.083`, `1/m ≈ 12`, and at
  `S_e = 1e-8`, `S_e^{1/m} ≈ 1e-97`; at `n_vG → 1`, `1/m` is huge and `0.8^{1/m} ≈ 1e-97`).
  Then `1 − (tiny)` rounds to exactly `1` and `U` collapses to `0` (or a value with all
  significant digits lost) — the resulting differenced curvature is meaningless (sign flips
  either way), not a real inflection.

The cure is a **numerically stable** `U = −expm1(m · log1p(−S_e^{1/m}))`, which keeps `U`
accurate to full precision even when `S_e^{1/m} ~ 1e-300`. With that, `d²K/dS_e² > 0`
everywhere. The **asymptotic** argument closes it rigorously: as `S_e → 0`,
`U ≈ m·S_e^{1/m}` so `K ≈ m²·S_e^{L+2/m}`, a convex power law (exponent `L + 2/m > 1`), hence
`d²K/dS_e² > 0` in the tail.

Consequences:

- **No detect-and-reject scan is needed** (there is nothing non-convex to reject).
- The `__post_init__` 2-point monotonicity guard (samples `S_e ∈ {0.5, 0.99}`) was
  **vestigial** and is removed: both sample points lie in the float64-well-resolved band
  where `_dk_dse` is cleanly increasing, so `_dk_dse(0.5) < _dk_dse(0.99)` for every valid
  Carsel–Parrish soil and the guard never fired. Its safety does **not** rest on `_dk_dse`
  being monotone over the whole search range `[_C_MIN, 1]` — in fact `_dk_dse` is
  double-precision _non_-monotone over a low-`S_e` sub-interval (the same catastrophic
  cancellation), but that deep tail maps to `c < _C_MIN`, is unreachable by brentq, and is
  therefore harmless.
- The vG-M `R↔c` inversion degrades only at `c → K_s` (saturation, `S_e → 1`, `R → 0`, where
  `dK/dS_e → ∞` cannot be bracketed in `[_C_MIN, 1]`). This is an expected endpoint edge
  (the `MAX_FINITE_SPEED` / `R = 0` regime, `interactions.py:48`), not a mid-range defect;
  round-trips are exact for `c ≤ 0.5·K_s`.

**Bottom line:** every flux is convex, concave, or linear — all single-curvature. The
solver's shock + rarefaction repertoire is _complete_; compound waves are physically absent
from this package.

---

## 3. Complete pairwise interaction table

A scalar law has a **single characteristic family**. Two consequences fix the entire
interaction catalogue:

1. **Same-family simple waves do not interact.** Two rarefactions separated by a constant
   state are bounded by characteristics of equal speed (the shared boundary concentration),
   so the gap between them never closes (Rhee–Aris–Amundson, Vol. II, simple-wave theory).
   **RAREF–RAREF cannot collide.**
2. Interactions occur only when a **rear (upstream) wave is faster** than the front
   (downstream) wave it chases. The primitives are:

| rear ＼ front   | shock                                              | rarefaction                                                                          |
| --------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **shock**       | merge → one shock (Lax)                            | shock overtakes fan → curved (decaying) shock; may weaken to nothing or exit the fan |
| **rarefaction** | fan head overtakes shock → curved (decaying) shock | (impossible — same family, no interaction)                                           |

Contacts (linear flux) never interact (all speeds equal). Thus the **only** genuine
interactions are: **shock↔shock**, **shock→fan**, and **fan→shock**. A curved shock fed by
one fan can subsequently be fed by a second fan (**doubly-fed**), and two curved shocks can
merge — but these are compositions of the same three primitives, not new ones.

### 3.1 Mapping the code onto the table

Wave objects (`waves.py`) and events (`events.py`, `EventType` at `:67`) are
_representations_ of the primitives above:

| code object / event     | file:line                             | primitive realized                                                                           | reachable?                                |
| ----------------------- | ------------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `CharacteristicWave`    | `waves.py:256`                        | contact / constant-state characteristic                                                      | yes (smooth regions)                      |
| `ShockWave`             | `waves.py:342`                        | shock                                                                                        | yes                                       |
| `RarefactionWave`       | `waves.py:440`                        | rarefaction fan                                                                              | yes                                       |
| `DecayingShockWave`     | `waves.py:604`                        | shock adjacent to **one** fan                                                                | yes (shock↔fan)                           |
| `DoubleFanShockWave`    | `waves.py:1079`                       | shock adjacent to **two** fans                                                               | yes (doubly-fed formation)                |
| `CHAR_CHAR_COLLISION`   | handler `handlers.py:32`              | two contacts compress → shock (genesis)                                                      | yes                                       |
| `SHOCK_SHOCK_COLLISION` | `handlers.py:95`                      | shock↔shock merge                                                                            | yes                                       |
| `SHOCK_CHAR_COLLISION`  | `handlers.py:152`                     | contact into shock (may emit a fan)                                                          | yes                                       |
| `RAREF_CHAR_COLLISION`  | `handlers.py:305`                     | contact absorbed at a fan boundary                                                           | yes                                       |
| `SHOCK_RAREF_COLLISION` | `handlers.py:223`                     | shock→fan (tail) and fan→shock (head) → `DecayingShockWave`                                  | yes                                       |
| `WAVE_MERGE`            | `interactions.py:394` `resolve_merge` | universal merge (any DSW/DFSW face pair, incl. a DFSW's own boundary line — side exhaustion) | yes (transitively, multi-front)           |
| `DSW_FAN_EXHAUSTED`     | `solver.py:594`                       | degradation: `DecayingShockWave` → plain `ShockWave`                                         | yes                                       |
| `RAREF_RAREF_COLLISION` | **no-op** `solver.py:560`             | (impossible interaction)                                                                     | detected, correctly not resolved — see §4 |
| `OUTLET_CROSSING`       | `handlers.py:354`                     | boundary bookkeeping                                                                         | yes                                       |

The universal merge calculus (`interactions.py`): every wave exposes **faces** (`Face`,
`:53`) separating a left/right **`Feeder`** (a constant state or a bounded self-similar fan,
`waves.py:55`). When a rear face overtakes a front face they merge into a single successor
built from `(rear.left_feeder, front.right_feeder)` via `make_wave_from_feeders`
(`interactions.py:308`): `(const,const)→ShockWave`, `(const,fan)/(fan,const)→DecayingShockWave`,
`(fan,fan)→DoubleFanShockWave`. This one rule generates shock–shock merges, fan-entry,
doubly-fed formation, same-apex annihilation, and every composition thereof.

---

## 4. Non-existence and reachability claims

### 4.1 RAREF–RAREF is a correct no-op — CONFIRMED

`RAREF_RAREF_COLLISION` is detected geometrically (`solver.py:383–387`) but the handler makes no
topology change and leaves both rarefactions active (`solver.py:560`). This is **correct**:
in a scalar (single-family) law two rarefactions never interact (§3, point 1). Numerically
confirmed (`scratchpad/repro_raref_raref.py`): a `5→3→1` Freundlich `n=2` inlet emits two
same-family rarefactions; the solver produces **no** `rarefaction_rarefaction` topology
event, `find_unresolved_interaction` returns `None`, and the outlet `cout` matches the
independent Godunov FV oracle to `0.07%` (first-order FV error). The event is retained only
as a detection record; the reader sweep treats overlapping fans as normal (they read
identically from either neighbour).

_Action:_ keep the no-op; pin it with a non-existence regression test asserting the solver
never emits a resolved raref–raref successor on a randomized same-family sweep.

### 4.2 Compound waves — CONFIRMED non-existent

No isotherm is non-convex (§2), so the Oleinik compound (shock+rarefaction) wave never
forms. The solver's shock/rarefaction repertoire is complete.

_Action:_ no compound-wave handling and no "detect-and-reject non-convex flux" scan are
required. Document convexity; treat the vestigial vG-M 2-point guard per §2.2.

### 4.3 Doubly-fed side exhaustion — RESOLVED through the universal merge (issue #317)

A `DoubleFanShockWave` side ends in finite θ **exactly** when the shock face crosses one of
its own fan boundary lines. Lax entropy (`λ(c_L) ≥ σ ≥ λ(c_R)`) admits only two such
crossings: the left fan's slow upstream-far characteristic catching the shock from behind
(the `n<1` mirror, where the fan's far ray outruns the shock), or the shock catching the
right fan's fast downstream-far characteristic. Both are the **same primitive** as every
other face merge — a boundary line overtaking (or being overtaken by) a shock face — so they
are resolved by the one universal `WAVE_MERGE` path: the merge builds the degraded successor
from `(const far-bound feeder | surviving fan feeder)` and retires the exhausted boundary
line. The face-pair loop admits the crossing of a DFSW's shock face with its **own** boundary
line (`solver.py`); the loose born-coincident admission is disabled for same-wave pairs so a
front born a small distance from its own boundary and diverging from it is not spuriously
retired (exact coincidence still fires).

This replaces the earlier dedicated `DFSW_SIDE_EXHAUSTED` detector
(`theta_at_side_exhaustion` and its handler), which was **incomplete**: it targeted the fan
edge _farther_ from the current side value (wrong whenever the wave is born mid-fan) and its
residual was flattened to exactly zero by the feeder's fan-extent clamp, so the bracket never
fired. That left the exhausted fan's boundary line exposed _beyond_ the shock — a ghost face
the sweep reader misattributed, breaking outlet-mass monotonicity: the loud decline below.
Geometry-, not regime-dependent (favorable Langmuir hits the same missed crossing).

- **DFSW × DFSW** and **characteristic × fan** (via `WAVE_MERGE`): structurally reachable
  (the face-pair loop admits them) but not separately pinned. Construct a producing
  input or prove impossibility.

### 4.4 Former multi-front gap — unfavorable (`n<1`) and some Langmuir (issue #317, RESOLVED)

The audit found the multi-front solver **declined** (left an unresolved interaction → the
public API raised `RuntimeError`, never a silently-wrong `cout`) for some multi-pulse inputs:

- **Freundlich `n<1`** (concave / mirror geometry): a simple two-pulse input sufficed —
  minimal reproducer `cin=[0,8,8,8,0,0,2,2,0]`, `V=10.6`.
- **Langmuir**: some specific multi-pulse configs.
- **Freundlich `n>1`** (favorable): robustly supported (0 declines over broad sweeps).

Root cause: the missed doubly-fed side exhaustion of §4.3, not a mishandled shock↔fan merge —
the merge calculus and every DSW/DFSW trajectory are correct in the `n<1` mirror (verified to
`<1e-6` relative against independent DOP853 integration). Resolved by routing side exhaustion
through the universal merge (§4.3). Pinned by `TestSideExhaustion`: the `n<1` reproducer with
the exhaustion at the closed-form `θ=778.75`, `V=5/28` plus an FV-oracle cross-check, the
public-API end-to-end resolution, and the two formerly-declining favorable Langmuir configs.

---

## 5. Exactness of interaction resolution

Two families of resolution paths exist. The direction of the audit is **one universal path**,
not per-isotherm closed forms (leanness "one path"): the Feeder/Face merge calculus plus the
universal numerical decay trajectory already resolve every `NonlinearSorption` and every
Freundlich `n`.

| resolution path                                                                                                                                      | file:line                      | accuracy                                                                                                                 | status                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| Sweep-reader fan integrals (universal IBP antiderivative)                                                                                            | `output.py:592,601,1047`       | exact                                                                                                                    | keep (already single universal path)                      |
| Merge calculus (`resolve_merge`/`make_wave_from_feeders`)                                                                                            | `interactions.py:308,394`      | exact (algebraic)                                                                                                        | keep                                                      |
| `DecayingShockWave` universal decay profile (`_build_decay_profile`, Gauss–Legendre-10, 6000-node spline)                                            | `waves.py:1369`                | ~`1e-2` in the Freundlich `c_fixed>0` **growing** branch near the `c→0` floor (central-FD `R'`, `R'→∞`); to be tightened | keep + tighten (§ below)                                  |
| `DoubleFanShockWave` universal RK4 (`_ensure_numerical`, fixed 512-step)                                                                             | `waves.py:1202`                | discretization-limited; convergence-tested                                                                               | keep                                                      |
| Freundlich `n=2` DFSW shared-apex quadratic (`_v_closed`, `_alpha`, `_k`)                                                                            | `waves.py:1145,1166,1172`      | exact                                                                                                                    | **parallel special case → delete after equivalence test** |
| Freundlich decaying-shock closed forms (`_compute_k_freundlich`, `_c_decay_freundlich`, `_invert_freundlich_cr_zero`, `_outlet_crossing_freundlich`) | `waves.py:1494,1560,1607,1653` | exact                                                                                                                    | **delete after equivalence test**                         |
| Langmuir decaying-shock closed forms (`_compute_k_langmuir`, `_c_decay_langmuir`, `_outlet_crossing_langmuir`)                                       | `waves.py:1534,1635,1701`      | exact                                                                                                                    | **delete after equivalence test**                         |
| Brooks–Corey decaying-shock closed form (`_c_decay_brooks_corey`)                                                                                    | `waves.py:1456`                | exact                                                                                                                    | **delete after equivalence test**                         |
| dispatch flags (`_freundlich_cf`/`_langmuir_cf`/`_brooks_corey_cf`/`_numerical`, `_closed_form`)                                                     | `waves.py:773–778,1145`        | —                                                                                                                        | **delete with their branches**                            |

**Tightening the universal path** (prerequisite to deleting the closed forms): the `~1e-2`
growing-branch error is dominated by the **central finite-difference `R'`** in
`_build_decay_profile` (`waves.py:1417`) where `R'→∞` at the `c→0` floor, compounded by a
c-uniform node grid that under-resolves the steep integrand there. Every isotherm has an
**analytic `R'`**; replacing the FD `R'` with the closed-form derivative (and refining the
near-floor grid) drives the universal quadrature to a documented tolerance for all isotherms,
after which the per-`n` closed forms are redundant and deleted (each behind a one-time
equivalence test vs the universal path, with the analytic `n=2` solution and a DOP853
trajectory kept as **test-only** anchors).

Controlled approximations that remain (documented): the `c_min` dry-soil floor
(`math.py:32`, `_C_MIN = 1e-12`) and the born-coincident `1e-3·V` triple-collision tolerance
(`interactions.py:275`).

---

## 6. Literature

- **Rhee, Aris & Amundson**, _First-Order Partial Differential Equations_, Vols. I–II —
  chromatographic wave interactions and same-family simple-wave theory (the definitive
  source for §3–§4).
- **Helfferich & Klein**, _Multicomponent Chromatography_ — interference/coherence.
- **LeVeque**, _Finite Volume Methods for Hyperbolic Problems_ — convex vs non-convex scalar
  laws, Oleinik entropy condition, compound waves; Godunov upwind (the FV oracle basis).
- **Smith (1983)**, _The theory of the kinematic wave in unsaturated flow_; **Charbeneau
  (1984)**; **Sisson, Ferguson & van Genuchten (1980)** — kinematic-wave percolation,
  convex conductivity flux, drainage/wetting fronts.
- **Carsel & Parrish (1988)** — statistical distributions of soil hydraulic parameters
  (the `n_vG`, `λ`, `θ_r`, `θ_s`, `K_s` grids used for the percolation regime).
- **van Genuchten (1980)**; **Mualem (1976)**; **Brooks & Corey (1964)** — the retention /
  conductivity constitutive models.
