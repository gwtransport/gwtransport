# gwtransport Code Review — 2026-03-01

## Context

A thorough code review was performed on all 22 source files (~13,500 lines) of the gwtransport package. The review was conducted from the `report` repository by reading the `gwtransport/` codebase and cross-referencing against the theory documents (`kb/apvd.md`, `kb/rtd_knowledge_base.md`).

Five parallel review agents covered:
1. `advection.py` + `advection_utils.py`
2. `residence_time.py` + `gamma.py`
3. `logremoval.py` + `diffusion.py`
4. All front tracking modules (`fronttracking/`)
5. `utils.py` + `deposition.py` + `surfacearea.py` + `diffusion_fast.py`

---

## Status: What was done, what remains

- **2026-03-01:** Full review of all source files, findings catalogued below.
- **2026-03-02:** Fixes applied in commit `ecff754`. See per-item status below.
- **2026-03-02:** Second batch of fixes: #11 (documented), #17, #24, #26, #27, #28, #31, #34. All missing tests written.
- **Still outstanding (require architectural decisions):** Issues #5, #8, #13, #14, #15, #21, #23.

---

## Critical Bugs (produce silently wrong results)

### 1. ✅ FIXED `surfacearea.py` — Upper-clip correction is one-directional

The trapezoid clipping correction only handles the case where the left vertex is above `y_upper` and the right vertex is below (`y_tl > y_upper > y_tr`). The reverse direction (`y_tl < y_upper < y_tr`) is silently ignored, producing wrong contact areas.

**Analytical counterexample:** `y_tl=3, y_tr=7, y_upper=5, y_lower=0, width=2`. Code returned avg height 4.0; correct answer is 4.5. *(Note: original review stated 3.25, which was incorrect.)*

**Fix:** Rewrote `compute_average_heights` using exact analytical integrals of clipped linear functions (`_positive_part_integral` + `_clipped_linear_integral`), handling all crossing cases for both bounds.

### 2. ✅ FIXED `surfacearea.py` — No correction for lower-clip crossings

The same triangle-correction logic was entirely missing for `y_lower` crossings.

**Analytical counterexample:** `y_tl=2, y_tr=2, y_bl=-2, y_br=2, y_lower=0, y_upper=10, width=2`. Code returned avg height 1.0; correct answer is 1.5. ✓

**Fix:** Handled by the same rewrite as #1.

### 3. ✅ FIXED `fronttracking/handlers.py:554–575` — Shock-rarefaction head collision uses wrong state

When the rarefaction head catches a shock, the new shock was created with `c_right=shock.c_left` instead of `c_right=shock.c_right`. This discards the state downstream of the original shock, breaking mass conservation.

**Fix:** Changed to `c_right=shock.c_right`.

### 4. ✅ FIXED `fronttracking/solver.py` — Boundary type hardcoded for rarefaction collisions

The boundary type was stored in the event heap tuple but never extracted into the `Event` object. All rarefaction collisions used hardcoded boundary types.

**Fix:** Added `boundary_type` field to `Event` dataclass, extracted it from the heap tuple for rarefaction collision event types, and wired it through to all three handler calls.

### 5. ⬜ OUTSTANDING `fronttracking/handlers.py:578–625` — Rarefaction-characteristic handler is a stub

```python
def handle_rarefaction_characteristic_collision(...):
    char.is_active = False
    return []
```

The characteristic is silently discarded without modifying the rarefaction. This breaks mass conservation whenever a new inlet wave enters an existing rarefaction fan. Acknowledged in docstring as known limitation. **Not fixed — requires significant front tracking redesign.**

---

## Serious Physics Issues

### 6. ✅ FIXED (docstring) `advection_utils.py` — Deconvolution is not a true inverse

The `extraction_to_infiltration` docstring claimed "pseudo-inverse" but the implementation is a separate forward model in reverse direction.

**Fix:** Corrected docstring to say "reverse-direction flow-weighted estimate, NOT a true deconvolution or pseudo-inverse." The implementation itself is unchanged — a proper regularized deconvolution would be a separate feature.

### 7. ✅ FIXED `diffusion.py` — `_erf_integral_time` wrong for negative x

The formula produced values off by a constant `−x²/D` for negative x due to the `erfc` term.

**Fix:** Compute using `abs(x)` and restore `sign(x)` at the end, making the function correctly odd in x.

### 8. ⬜ OUTSTANDING `residence_time.py` — Freundlich retardation array vs scalar mismatch

`freundlich_retardation` returns an array (one R per timestep) but `residence_time` expects a scalar `retardation_factor`. Broadcasting silently produces dimensionally incorrect results. The correct implementation for concentration-dependent retardation requires integrating `V_p · R(C(t'))` along the flow path, not scalar multiplication. **Requires architectural decision.**

### 9. ✅ ALREADY FIXED `logremoval.py` — Wrong formula in docstring

Was fixed in prior commit `ded4460` (before this review session). The `gamma_find_flow_for_target_mean` docstring now matches the MGF-based implementation.

### 10. ✅ ALREADY FIXED `logremoval.py` — `gamma_mean` semantic ambiguity

Was fixed in prior commit `ded4460`. `gamma_mean` now correctly computes the parallel (mixed-effluent) mean using the moment generating function, not the arithmetic mean.

### 11. ✅ DOCUMENTED `utils.py:330` — `linear_average` silently converts NaN to 0

```python
segment_integrals = np.nan_to_num(segment_integrals, nan=0.0)
```

For bins straddling the data boundary, the integral is underestimated because the width denominator uses the full bin width, not the covered fraction. **Needs investigation of downstream impact before changing.**

### 12. ✅ FIXED `fronttracking/math.py` — R-H fallback uses one-sided velocity

When `|ΔC_total| → 0`, the degenerate fallback returned `flow / retardation(c_left)` instead of averaging both sides.

**Fix:** Changed to `flow / (0.5 * (retardation(c_left) + retardation(c_right)))`.

### 13. ⬜ OUTSTANDING `fronttracking/math.py` — `c_min` clipping propagates into shock velocity

`total_concentration` clips to `c_min` before computing the sorbed term. Since `shock_velocity` calls `total_concentration`, the R-H condition is evaluated at clipped concentrations, not true ones. This can produce spurious entropy violations. **The clipping is a deliberate regularization for numerical stability; changing it requires careful analysis of edge cases.**

### 14. ⬜ OUTSTANDING `fronttracking/waves.py:556–562` — `concentration_left/right` semantics flip for n < 1

The labels "upstream = tail" and "downstream = head" presuppose `n > 1` physics. For `n < 1`, the ordering reverses. Collision handlers relying on these methods may assign wrong states. **Requires front tracking redesign for n < 1 support.**

### 15. ⬜ OUTSTANDING `fronttracking/math.py:711–718` — First arrival time uses wrong concentration

`compute_first_front_arrival_time` uses `R(c_first)` (the first non-zero bin), but for `n > 1` the fastest wave is the highest concentration with the lowest retardation. Should use minimum retardation over all non-zero bins. **Requires careful analysis of n > 1 vs n < 1 cases.**

---

## Moderate Issues

### 16. ✅ FIXED `advection.py` — Missing `retardation_factor >= 1` validation

**Fix:** Added `retardation_factor < 1.0` check raising `ValueError` in both `infiltration_to_extraction` and `extraction_to_infiltration`.

### 17. ✅ DOCUMENTED `advection_utils.py:114–124` — Equal-weight averaging is an undocumented constraint

All pore volume bins receive equal weight `1/N`. This is correct for `gamma.bins()` (equal-probability) but wrong for user-supplied unequal streamlines. Should document or add a `weights` parameter.

### 18. ✅ FIXED `residence_time.py` — `fraction_explained` wrong for 1D input

**Fix:** Added `rt.ndim != 2` validation raising `ValueError`.

### 19. ✅ FIXED `gamma.py` — Partial alpha/beta silently overwritten

**Fix:** Changed `parse_parameters` to check `(alpha is None) != (beta is None)` and raise `ValueError` requiring both or neither.

### 20. ✅ FIXED `logremoval.py` — `flow_fractions` sum not validated

**Fix:** Restored validation: `np.all(np.isclose(np.sum(flow_fractions, axis=axis), 1.0))`, raising `ValueError` if not satisfied.

### 21. ⬜ OUTSTANDING `fronttracking/solver.py:301–309` — Race condition at simultaneous flow+concentration change

When `flow` and `cin` both change at the same `tedges[i]`, two events are scheduled at the same time. The ordering depends on the heap counter, not physics.

### 22. ✅ FIXED `fronttracking/math.py` — Asymmetric entropy tolerance

**Fix:** Changed left inequality from `shock_vel + tolerance` to `shock_vel - tolerance` to match the right side.

### 23. ⬜ OUTSTANDING `diffusion_fast.py:139–147` — Output on wrong time grid

`infiltration_to_extraction` returns concentrations on the infiltration time grid, not shifted to extraction time. The contract is ambiguous compared to `diffusion.py`.

### 24. ✅ FIXED `deposition.py:106–107` — Cumulative flow construction differs from `residence_time.py`

Uses a fragile `diff+prepend` pattern instead of the cleaner `concatenate+cumsum` in `residence_time.py`. Same numerical result but should be unified.

---

## Style / Documentation

| # | Location | Issue | Status |
|---|----------|-------|--------|
| 25 | `advection.py` + `math.py` | `EPSILON_FREUNDLICH_N` defined in two places | ✅ Fixed — import from `math.py` |
| 26 | `advection_utils.py:26,82` | `cin` passed but only `len(cin)` used | ✅ Fixed — removed unused params |
| 27 | `advection.py` vs `math.py` vs `residence_time.py` | Inconsistent Freundlich K unit docs | ✅ Fixed — unified to `[(m³/kg)^(1/n)]` |
| 28 | `residence_time.py:240–242` | Opaque `diff+prepend` cumulative flow (correct but fragile) | ✅ Fixed — unified to `concatenate+cumsum` |
| 29 | `logremoval.py` | Duplicate `Notes` sections in `parallel_mean` | ✅ Fixed — merged into one |
| 30 | `math.py:26` | `EPSILON_DENOMINATOR = 1e-18` below machine epsilon | ✅ Fixed — bumped to `1e-15` |
| 31 | `advection.py` | Duplicate validation blocks in front tracking variants | ✅ Fixed — extracted `_validate_front_tracking_inputs` |
| 32 | `gamma.py` | `mean_std_to_alpha_beta` raises `ZeroDivisionError` not `ValueError` for `std=0` | ✅ Fixed — explicit `ValueError` |
| 33 | `logremoval.py` | Return type annotation inconsistent with actual return | ✅ Fixed — `np.floating \| NDArray` |
| 34 | `diffusion_fast.py:262` | `compute_sigma_array` returns length [m], not sigma — misleading name | ✅ Fixed — renamed to `compute_diffusive_spreading_length` |

---

## Missing Tests

| Test | Would catch | Status |
|------|-------------|--------|
| `_erf_integral_time` with negative x vs `scipy.integrate.quad` | Bug #7 | ✅ Written in `test_diffusion.py` |
| `fraction_explained` with 1D input | Bug #18 | ✅ Written in `test_fraction_explained.py` |
| `surfacearea` exact analytical values + reversed crossings | Bugs #1–2 | ✅ Written in `test_surfacearea.py` |
| `diffusion_fast` vs `diffusion.py` numerical comparison | Issue #23 | ✅ Written in `test_diffusion_fast.py` (documents time grid offset) |
| `freundlich_retardation` analytical value checks | Issue #8 | ✅ Written in `test_residence_time.py` |
| Variable-flow residence time with exact analytical solution | General correctness | ✅ Written in `test_residence_time.py` |
| `gamma_mean` vs `parallel_mean` (Jensen inequality) | Issue #10 | ✅ Written in `test_logremoval.py` |
| `gamma.bins()` with `alpha < 1` and large `n_bins` | Numerical stability | ✅ Written in `test_gamma.py` |
| Round-trip `deconv(conv(cin))` identity test | Issue #6 | ✅ Already existed in `test_advection_roundtrip.py` |

---

## Summary

**Fixed (2026-03-02, commit `ecff754`):** #1, #2, #3, #4, #6 (docstring), #7, #12, #16, #18, #19, #20, #22, #25, #29, #30, #32, #33. Issues #9, #10 were already fixed in prior commits.

**Fixed (2026-03-02, second batch):** #11 (documented), #17 (documented), #24, #26, #27, #28, #31, #34. All 9 missing tests written.

**Outstanding (require architectural decisions):** #5 (rarefaction stub), #8 (Freundlich array/scalar), #13 (c_min clipping), #14 (n < 1 semantics), #15 (first arrival time), #21 (race condition), #23 (diffusion_fast time grid).
