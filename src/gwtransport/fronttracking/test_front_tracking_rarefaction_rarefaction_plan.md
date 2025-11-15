# Rarefaction–Rarefaction Interaction Work Package

This note captures the current state of `tests/src/test_front_tracking_events.py` and a prompt you can use to continue work on rarefaction–rarefaction interactions in a new session.

---

## 1. File reference

Target file (already in the repo):

- `tests/src/test_front_tracking_events.py`

The relevant test class is:

- `TestRarefactionRarefactionIntersections`

It currently contains:

- `test_head_or_tail_boundary_intersects_other_rarefaction`
- `test_parallel_rarefaction_boundaries_do_not_intersect`

Both are already regime-aware (`n>1` vs `n<1`) and passing.

---

## 2. Physics summary for use in tests

- Freundlich sorption with exponent `n` defines retardation and characteristic speeds.
- Characteristic speed is `flow / R(c)`, where retardation `R(c)` increases with concentration differently depending on `n`.
- Regime behaviour:
  - For \(n > 1\): higher concentration ⇒ faster characteristics.
  - For \(0 < n < 1\): higher concentration ⇒ slower characteristics.
- Valid rarefaction waves must have **head velocity > tail velocity**:
  - For \(n > 1\): typically `c_head > c_tail`.
  - For \(0 < n < 1\): must have `c_tail > c_head`.
- All spatial coordinates `v` are non-negative, and intersections must satisfy `t_int >= t_current` and `v_int >= 0`.

---

## 3. Plan for rarefaction–rarefaction interactions

Work inside `TestRarefactionRarefactionIntersections` in `tests/src/test_front_tracking_events.py`.

### 3.1 Scenarios to cover

For \(n > 1\):

1. **Generic intersecting rarefactions (already present)**
   - Two rarefactions (`raref1`, `raref2`) with overlapping characteristic fans so that at least one boundary (head or tail of `raref1`) intersects some boundary of `raref2`.
   - Expectations:
     - At least one intersection in the future (`t_int > t_current`, `v_int >= 0`).
     - Returned boundary label in `{ "head", "tail" }`.
     - Position of the selected boundary of `raref1` at `t_int` matches `v_int` via `characteristic_position`.

2. **Parallel rarefaction boundaries (already present)**
   - Two rarefactions with identical `(c_head, c_tail)` and `t_start` but offset `v_start`, so head and tail speeds are equal for both waves.
   - Expectations:
     - At most a single boundary intersection (typically head–head) where appropriate.
     - If an intersection exists, it is labelled as `"head"`, and `v_int` matches the head position from `characteristic_position`.

3. **Head–tail vs tail–head interaction (optional improvement)**
   - Configure `raref1` and `raref2` so that `raref1`'s head intersects `raref2`'s tail (or vice versa), and verify that the solver reports the correct boundary and location.

For \(0 < n < 1\):

4. **Invalid rarefactions (already partially covered)**
   - Any `RarefactionWave` with `c_tail < c_head` is invalid and must raise `ValueError("Not a rarefaction:")`.
   - Existing tests already check this behaviour for selected parameter sets.

5. **Valid rarefaction–rarefaction intersection (new positive test)**
   - Construct two **valid** rarefactions with `c_tail > c_head`:
     - `raref1`: `c_head1 < c_tail1`.
     - `raref2`: `c_head2 < c_tail2`.
   - Choose concentrations so that a boundary of `raref1` intersects a boundary of `raref2` in the future.
   - Expectations:
     - `find_rarefaction_boundary_intersections(raref1, raref2, t_current)` returns at least one intersection.
     - For the first `(t_int, v_int, boundary)`:
       - `t_int >= t_current`, `v_int >= 0`.
       - Using the indicated boundary of `raref1` (head or tail) and `characteristic_position`, you can reproduce `v_int` to machine precision.

6. **Valid rarefactions with no intersection (new negative test)**
   - Construct two valid rarefactions (`c_tail > c_head` for both) such that their fans never intersect in the future (e.g. one is entirely ahead and faster).
   - Expect `find_rarefaction_boundary_intersections` to return an empty list.

Implementation constraint: tests should remain regime-aware using

```python
if freundlich_sorption.n < 1.0:
    ...  # n<1 behaviour
elif freundlich_sorption.n > 1.0:
    ...  # n>1 behaviour
```

so that a hypothetical `n == 1` is not implicitly classified.

---

## 4. Ready-to-use prompt for a new session

When you open a new session and want to continue this work, paste the following prompt (adjust the date if you like):

> I am working in the `gwtransport` repository on branch `front-tracking-clean` on macOS. My current file of interest is `tests/src/test_front_tracking_events.py`, and I want to continue refining the **rarefaction–rarefaction interaction tests**. The project already has a knowledge base in `src/gwtransport/fronttracking/KNOWLEDGE_BASE.md` that describes the front-tracking physics.
>
> There is a helper note at `tests/src/test_front_tracking_rarefaction_rarefaction_plan.md` that summarises the physics and desired scenarios. Please:
>
> 1. Open `tests/src/test_front_tracking_events.py` and focus on the `TestRarefactionRarefactionIntersections` class.
> 2. Use the plan in `tests/src/test_front_tracking_rarefaction_rarefaction_plan.md` to:
>    - Ensure the existing tests correctly cover intersecting and parallel rarefactions for `n>1`, with accurate use of `characteristic_position` and strict machine-precision checks.
>    - Add **two additional tests for `0 < n < 1`**:
>      - One positive test where two valid rarefactions (with `c_tail > c_head`) have at least one boundary intersection, checking `t_int`, `v_int`, the reported `boundary`, and consistency with `characteristic_position`.
>      - One negative test where two valid rarefactions (with `c_tail > c_head`) do **not** intersect in the future, asserting that the intersection list is empty.
>    - Optionally, add one more explicit `n>1` case where `raref1`'s **tail** is the boundary that intersects, and assert that the solver returns `boundary == "tail"` and that positions match.
> 3. Keep the regime handling explicit using `if freundlich_sorption.n < 1.0:` / `elif freundlich_sorption.n > 1.0:` so `n==1` is not misclassified.
> 4. After making changes, run `pytest tests/src/test_front_tracking_events.py -k "RarefactionRarefaction"` and iterate until all tests pass with correct physics and no unnecessary tolerances beyond the existing `np.isclose` uses.
>
> Make a short plan first, then implement the changes directly in the test file.

---

With this file and prompt, you can quickly restore context and continue developing the rarefaction–rarefaction interaction tests in a fresh session.