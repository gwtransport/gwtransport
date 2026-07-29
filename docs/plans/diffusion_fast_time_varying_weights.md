# Plan: time-varying streamtube weights in `diffusion_fast`

Implementation brief for extending `gwtransport.diffusion_fast` with time-varying
flow shares across streamtubes ("pipes with valves"). Pore volumes stay constant;
only the apportionment of flow across the fixed streamtube family varies in time.

## Model

Fixed atoms `aquifer_pore_volumes` `V_i` (apparent volumes, current convention).
New input: relative activity `rho` of shape `(n_pv, n_bins)`, aligned with `tedges`
bins (flux-like: constant per bin), with

- `rho >= 0`, and
- `mean_i rho[i, t] == 1` for every bin `t` (this **is** mass conservation — validate
  to a tight tolerance, do not silently normalize).

Tube `i` sees the effective flow `rho[i] * flow`; its residence threshold is
unchanged (`R * V_i` against the tube's own cumulative volume). `rho = 1`
everywhere reproduces the current model exactly.

Physics that makes this work in this module (no new approximation): the kernel's
variance `D_t = D_m * tau + alpha_L * xi` uses only endpoint quantities — `tau` is
clock time, `xi` comes from the tube's cumulative-volume displacement — and
velocity stays spatially uniform within each constant-volume tube. A parked tube
(`rho_i = 0` stretch) is a zero-flow plateau of that tube's record: advection
freezes while `D_m * tau` keeps growing, which the existing stagnation machinery
already implements for shared-flow gaps.

## Code touchpoints (`_closed_form_coeff_matrix`, diffusion_fast.py)

The build is already a two-pass per-streamtube loop, and both workers take the
cumulative arrays as arguments; the values pass contains no flow variable at all.
Changes are confined to the caller:

1. `cumulative_volume_at_cin` → per tube: `cumulative_flow_volume(rho[i] * flow, dt)`.
2. `cumulative_volume_at_cout` → per tube, sampled at the cout edges (reject
   `flow_out` together with weights for now — keep scope tight).
3. `min_cin_flow` → per tube (min positive effective flow); throttled tubes get
   conservatively wide bands — a cost, not a correctness issue.
4. `stagnant_time_at_cout` → per tube, from the tube's own `dv == 0` bins. The gap
   correction inside `_pv_band_values` detects gaps from the cumulative it is
   given (`np.diff(v_cin) <= 0`) and needs no change.
5. Final mix: replace `band_vals /= n_pv` with per-row mixing weights applied at
   the scatter,

   `m_i[row] = ∫_bin rho[i] * flow dt / (n_pv * ∫_bin flow dt)`,

   so `sum_i m_i[row] == 1`. Skip stripe evaluation where `m_i[row] == 0` and
   guard `0 * NaN`. When `rho is None`, keep the existing `/= n_pv` path so the
   default stays bit-identical.

6. Validity: `valid_cout_bins` currently demands every tube in-record
   (`np.all(fraction_explained_full(...) >= 1)`). Compute coverage per tube on the
   tube's own effective flow, and gate on the **share-weighted** covered mass
   (strict `np.all` would NaN large swaths whenever one tube idles long). Mirror
   the `spinup` float-threshold semantics used elsewhere in the package.
7. Spin-up: the warm-start `extend_tedges` pad holds each tube's boundary `rho`
   column constant. A tube whose record starts closed (`rho = 0` boundary) has
   unknowable initial storage — strict-invalid until its own breakthrough is the
   correct, honest behavior.

`_pv_band_geometry`, `_pv_band_values`, `_breakthrough_antideriv`, and
`_solve_reverse_banded` are unchanged. The reverse entry point works by threading
`rho` through the same operator build.

## API

Add `streamtube_weights: npt.ArrayLike | None = None` (the `rho` above) to
`infiltration_to_extraction` and `extraction_to_infiltration`; for the `gamma_*`
entry points, weight row `i` maps to bin atom `i` of `gamma.bins` (ascending
volumes). Follow NumPy docstring style; document the mean-1 convention and the
per-bin alignment. Do **not** touch `diffusion_fast_fast` (its frozen-mean-flow
molecular approximation collapses for intermittent tubes) beyond a docstring
warning, and do not touch `advection`.

## Test contract (`tests/src/test_diffusion_fast.py`)

Machine precision throughout (`np.testing.assert_allclose(actual, expected)`,
per project convention).

1. **Default equivalence**: `streamtube_weights=None` is bit-identical to current
   output; all-ones weights match to machine precision.
2. **Disjoint-bank oracle** (the load-bearing test — it validates per-tube
   cumulatives, parked-tube stagnation, and mixing against existing tested code
   in one shot). Partition tubes into banks; bank `k` has `n_k` tubes and valve
   share `lambda_k(t)`, `sum_k lambda_k = 1`. Weighted run:
   `rho[i] = (n_pv / n_k) * lambda_k(t)` for tube `i` in bank `k`. Oracle: one
   **current-code** run per bank with `flow = lambda_k * flow` and rescaled atoms
   `V_i * n_k / n_pv`, mixed per cout bin with weights
   `∫_bin lambda_k * flow dt / ∫_bin flow dt` (mask, don't multiply, where a
   bank's share is 0). The two must agree to machine precision — including with
   `D_m > 0`, `R > 1`, and `lambda_k = 0` stretches (parked-bank aging), and in
   the zero-dispersion limit.
3. **Fixed point**: constant `cin` gives `cout ≡ cin` on fully covered bins, for
   any valid `rho`, any `R`, any dispersion (rows sum to 1).
4. **Mass budget**: a `cin` pulse is fully recovered (flux-weighted) once every
   tube has broken through.
5. **Validation**: negative weights, wrong shape, column mean ≠ 1, or
   `flow_out` combined with weights raise `ValueError`.

## Documentation

- Docstrings state that weights partially relax
  `assumption-steady-streamlines` while preserving incompressibility and
  no-transverse-mixing.
- Honesty note: a parked tube neglects transverse exchange for its whole idle
  time; negligible for solutes, material for heat
  (`sqrt(D_m * tau_park)` vs streamtube spacing) beyond weeks of parking.
- Weights must come from observables (per-well flows, clogging index, stage) —
  never free-fit per time.

## Workflow

Follow `CLAUDE.md`: run unit tests, `ruff format`/`ruff check --fix`, prettier,
and `ty check` before committing. Write the oracle test first and run it against
an intermediate implementation early — its bank/atom rescaling arithmetic is the
easiest place to lose a day.
