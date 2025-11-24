# Front-tracking Knowledge Base Prompt

This prompt is designed for (re)starting an AI-assisted coding session on the
**exact analytical front-tracking implementation** in `gwtransport`.

You can paste this prompt (optionally together with the contents of
`KNOWLEDGE_BASE.md`) into a new session to give the assistant the necessary
context and hard constraints before describing a concrete task.

---

## Prompt template

> You are working on the `gwtransport` repository, branch `front-tracking-clean`.
> The front-tracking implementation is complete and tested (174 tests) and
> follows the design in `FRONT_TRACKING_REBUILD_PLAN.md`.
>
> There is a detailed knowledge base file at
> `src/gwtransport/fronttracking/KNOWLEDGE_BASE.md` describing the current
> implementation, modules, algorithms, and invariants.
>
> Please:
> 1. Load and interpret the information in
>    `src/gwtransport/fronttracking/KNOWLEDGE_BASE.md` as the high-level
>    specification of how front tracking is implemented in:
>    - `src/gwtransport/advection.py`
>    - `src/gwtransport/fronttracking/math.py`
>    - `src/gwtransport/fronttracking/waves.py`
>    - `src/gwtransport/fronttracking/events.py`
>    - `src/gwtransport/fronttracking/handlers.py`
>    - `src/gwtransport/fronttracking/solver.py`
>    - `src/gwtransport/fronttracking/output.py`
>    - `src/gwtransport/fronttracking/plot.py`
> 2. Strictly enforce the front-tracking implementation principles (from the
>    rebuild plan):
>    - **Exact Analytical Computation**: All calculations use closed-form
>      analytical expressions (no numerical tolerances, no iterative solvers,
>      no numerical quadrature).
>    - **Physical Correctness**: Every wave satisfies conservation laws and entropy conditions exactly.
>    - **Spin-up Period Handling**: Compute first arrival time; output is
>      independent of unknown initial conditions after spin-up.
>    - **Detailed Diagnostics**: Track all events, waves, and state changes for
>      verification.
>    - **Multiple Streamline Architecture**: Design supports future extension
>      to distributions of pore volumes.
>    - **Consistent API**: Use `gwtransport` terminology (`tedges`, `cin`,
>      `flow`, `aquifer_pore_volume`, etc.).
>    - **No Capital Variable Names**: Follow Python convention (e.g., `c_left`,
>      not `C_L`; `v_max`, not `V_max`).
>    - **Meaningful and Explicit Tests**: Tests should not contain (1)
>      conditional statements other than those created by `pytest.mark.parametrize`
>      or (2) `try/except` constructs unless absolutely necessary. Each
>      comparison must directly serve the test purpose. Masked array
>      comparisons are only allowed when the unmasked values are sufficient to
>      validate the test purpose.
> 3. Before proposing or editing any code, check that your changes:
>    - Preserve analytic closed-form formulas and do not introduce numerical
>      solvers, tolerance-based iteration, or ad-hoc numerical hacks.
>    - Preserve entropy conditions and physical constraints.
>    - Preserve or extend diagnostics and testability.
>    - Keep the public API consistent with the definitions documented in
>      `KNOWLEDGE_BASE.md`.
> 4. When you need details about how something is supposed to work, consult and
>    align with `KNOWLEDGE_BASE.md` first, then read the corresponding source
>    file(s) to ensure you are consistent with the current implementation.
> 5. Update this prompt itself and/or `KNOWLEDGE_BASE.md` if you make any changes to the implementation details.
>
> ## Recent Progress: Unified Spatial Integration Formula (Completed ✅)
>
> **High Priority #3 from `FRONT_TRACKING_REBUILD_PLAN.md` - FULLY COMPLETED**:
>
> ### Phase 1: Runtime Mass Balance Verification (Completed 2025-01-23)
> - Added `compute_domain_mass()`, `compute_cumulative_inlet_mass()`, and
>   `compute_cumulative_outlet_mass()` to `output.py`
> - Extended `verify_physics()` with optional mass balance checking
> - Implemented **exact analytical spatial integration** for Freundlich n=2
>   using closed-form antiderivatives (no numerical quadrature)
> - Mass balance equation: mass_in_domain(t) + mass_out_cumulative(t) = mass_in_cumulative(t)
> - Tests pass with ~1e-6 relative error for n=2 (limited by time discretization,
>   not spatial integration which achieves machine precision)
>
> ### Phase 2: Unified Formula for All n > 0 (Completed 2025-01-24) ✅
> - **Derived and implemented ONE unified analytical formula** using generalized
>   incomplete beta function via mpmath
> - Works for **ALL positive real n > 0** with no conditional logic or special cases
> - Uses `mpmath.betainc()` with analytic continuation for negative parameters
> - Achieves **machine precision (~1e-15 relative error)** for all n values
> - Mathematical formulation: ∫ u^p (κ-u)^q du = κ^(p+q+1) B(u₁/κ, u₂/κ; p+1, q+1)
> - **All 285 tests pass** with representative n values (0.5, 0.8, 1.5, 2.0, 3.0, 5.0)
> - **Dependency added**: `mpmath>=1.3.0` to project dependencies
>
> ## Current Task: [Replace with your task description]
>
> Given this context and these constraints, here is the task I want you to
> work on now:
>
> [Paste your specific task here - the unified formula task above is now complete]
>
> ---
>
> **NOTE**: The above task has been completed. This template is preserved for
> reference and can be adapted for future tasks.
---

## How to use this prompt

1. Open a new AI assistant session in your editor.
2. Paste the contents of this file (and optionally `KNOWLEDGE_BASE.md`) into
	the chat.
3. Replace the placeholder task description at the end with your concrete,
	narrowly scoped task.
4. Let the assistant read `KNOWLEDGE_BASE.md` (and other referenced source
	files) from the repo and proceed under the constraints described above.
