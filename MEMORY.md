# Reviewer memory for gwtransport

Curated, non-obvious notes from past reviews. Prune aggressively; do not restate the system prompt.

## Environment / process notes
- Bash tool: plain `sleep`/`until` polling loops are capped at 2 minutes wall time regardless of
  loop logic. For long-running commands (e.g. the drift test suite), use `run_in_background: true`
  and poll the output file, or pass an explicit large `timeout`.
- `pytest` has `-n=auto` (xdist) as a default addopt (`pyproject.toml`) — always parallel, so a
  single slow test can dominate wall time on whichever worker draws it (LoadScheduling has no
  cost model).
- `uv tool run ty check .` needs `--python .venv-claude` (or whatever venv exists) explicitly — it
  does not auto-discover a non-`.venv` environment name, and without it every third-party import
  fails to resolve (`pytest`, etc.), producing noise unrelated to the diff under review.
- `uv tool run --from sphinx ... sphinx-build` fails on this box's system Python (3.11) vs the
  project's `Python>=3.12` requirement; use `uv run --extra docs sphinx-build ...` inside the
  project venv instead.

## gwtransport.radial_asr / drift engine (`_radial_asr_drift_kernels.py`)
- **Performance, not correctness, is the main risk here.** The azimuthal block engine solves an
  `(2M+1)x(2M+1)` matrix Riccati ODE (DOP853, rtol=1e-9/atol=1e-10) per phase; cost blows up sharply
  with `n_modes` (M=2 finishes a 12-bin toy schedule in <90s; M=4 the same schedule can take
  minutes). Measured: `tests/src/test_radial_asr_drift.py` (29 tests, deliberately "light" params)
  took ~27 min wall clock with 4 xdist workers, individual tests up to ~170s each — vs ~2 min total
  for the comparable pre-existing `test_radial_asr.py` (39 tests). Expect this file to dominate CI
  time for this test directory; don't assume a "stuck" test process is hung — check accumulated CPU
  time before killing it (real work, just slow).
- `regional_flux` (U, Darcy flux, m/day) and `v_d` (seepage velocity, m/day) differ by porosity:
  `v_d = U / n`. The module docstrings/table describe scenarios in terms of `v_d`; don't pass a
  `v_d` value straight into `regional_flux=` when reproducing a documented number — multiply by
  porosity first. (Cost me a false "docs table is wrong" alarm during review; it wasn't.)
- The `docs/source/user_guide/modules.rst` "Feasibility envelope" table numbers were spot-verified
  end-to-end (public `infiltration_to_extraction`, both the conservative-solute and thermal-retardation
  rows, years=1, all drift rates) and match to the displayed precision. No automated test pins these
  numbers (each cell costs ~50-100s to reproduce at n_quad=60); re-verify a couple of cells manually
  if a future PR touches `_auto_n_modes`, `_rest_drift_field`, or `field_grid`, since nothing in CI
  would catch a silent drift in these values.
- `field_grid`'s stagnation-radius cap `_RS_FRAC = 0.6` (in `_radial_asr_drift_kernels.py`) is
  duplicated as a bare `0.6` literal in `radial_asr._auto_n_modes` rather than imported — watch for
  this magic-number duplication if either threshold ever changes.
- Retardation is correctly R-independent in the stagnation radius (`r_s = |A_0|/|v_d|`) and in the
  eps decay ratio (both are water-flow-field/steady-state properties); it correctly *does* enter the
  rest-phase translation (`delta = v_d t / R`) and the Gaussian spread variances. `_auto_n_modes`
  deliberately ignores retardation in its rest-displacement heuristic (assumes R=1, conservative
  over-estimate of mode count) — that's intentional, not a bug.
- The scalar (non-drift) engine (`_radial_asr_reuse.py`/`_radial_asr_kernels.py`, unmodified by the
  drift PR) is itself very slow/warns of float overflow for small `molecular_diffusivity`
  (~1e-4 m2/day) combined with a rest phase at aquifer scales of tens of meters — pre-existing,
  out of scope for a drift-only diff, but explains why the drift PR's docs footnote deliberately
  uses "negligible regional_flux" (routes through the block engine) rather than exactly
  `regional_flux=0.0` (routes through the slow/fragile scalar reuse path) for its "no drift"
  reference column.
