# Front-tracking Knowledge Base Prompt

This prompt is designed for restarting an AI-assisted coding session on the exact analytical front-tracking implementation in `gwtransport`.

You can paste this prompt (optionally together with the contents of `KNOWLEDGE_BASE.md`) into a new session to give the assistant the necessary context and constraints.

---

## Prompt template

> You are working on the `gwtransport` repository, branch `front-tracking-clean`.
> The front-tracking implementation is complete and tested (174 tests) and follows the design in `FRONT_TRACKING_REBUILD_PLAN.md`.
>
> There is a detailed knowledge base file at `src/gwtransport/fronttracking/KNOWLEDGE_BASE.md` describing the current implementation, modules, algorithms, and invariants. Assume that file is up to date and authoritative.
>
> Please:
> 1. Load and interpret the information in `src/gwtransport/fronttracking/KNOWLEDGE_BASE.md` as the high-level specification of how front tracking is implemented in:
>    - `src/gwtransport/advection.py`
>    - `src/gwtransport/front_tracking_math.py`
>    - `src/gwtransport/front_tracking_waves.py`
>    - `src/gwtransport/front_tracking_events.py`
>    - `src/gwtransport/front_tracking_inlet.py`
>    - `src/gwtransport/front_tracking_handlers.py`
>    - `src/gwtransport/front_tracking_solver.py`
>    - `src/gwtransport/front_tracking_output.py`
>    - `src/gwtransport/front_tracking_plot.py`
> 2. Strictly enforce the front-tracking implementation principles (from the rebuild plan):
>    - **Exact Analytical Computation**: All calculations use closed-form analytical expressions (no numerical tolerances, no iterative solvers).
>    - **Physical Correctness**: Every wave satisfies conservation laws and entropy conditions exactly.
>    - **Spin-up Period Handling**: Compute first arrival time; output is independent of unknown initial conditions after spin-up.
>    - **Detailed Diagnostics**: Track all events, waves, and state changes for verification.
>    - **Multiple Streamline Architecture**: Design supports future extension to distributions of pore volumes.
>    - **Consistent API**: Use `gwtransport` terminology (`tedges`, `cin`, `flow`, `aquifer_pore_volume`, etc.).
>    - **No Capital Variable Names**: Follow Python convention (e.g., `c_left`, not `C_L`; `v_max`, not `V_max`).
>    - **Meaningful and Explicit Tests**: Tests should not contain conditional statements or `try/except` constructs unless absolutely necessary. Each comparison must directly serve the test purpose. Masked array comparisons are only allowed when the unmasked values are sufficient to validate the test purpose.
> 3. Before proposing or editing any code, check that your changes:
>    - Preserve analytic closed-form formulas and do not introduce numerical solvers or tolerance-based logic.
>    - Preserve entropy conditions and physical constraints.
>    - Preserve or extend diagnostics and testability.
>    - Keep the public API consistent with the definitions documented in `KNOWLEDGE_BASE.md`.
> 4. When you need details about how something is supposed to work, consult and align with `KNOWLEDGE_BASE.md` first, then read the corresponding source file(s) to ensure you are consistent with the current implementation.
>
> Given this context and these constraints, here is the task I want you to work on now:
> [describe your task here, e.g., "Update `README-front-tracking.md` to match the final implementation", or "Add a new integration test for a piecewise-constant flow with two pulses", etc.]

---

## How to use this prompt

1. Open a new AI assistant session in your editor.
2. Paste the contents of this file (and optionally `KNOWLEDGE_BASE.md`) into the chat.
3. Replace the placeholder task description at the end with your concrete task.
4. Let the assistant read `KNOWLEDGE_BASE.md` from the repo and proceed under the constraints described above.
