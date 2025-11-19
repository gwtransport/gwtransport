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
> Given this context and these constraints, here is the task I want you to
> work on now. Please scope it as narrowly and concretely as possible
> (e.g. "add a specific test", "adjust a particular wave interaction", or
> "update this doc section"):
> I am currently working on `examples/08_Front_Tracking_Exact_Solution.ipynb` It looks okay but it does not include rarefactions. Please create an additional example notebook that demonstrates front-tracking with rarefaction waves. Make sure to include plots of the V-t diagram and breakthrough curves, similar to the existing notebook. 

---

## How to use this prompt

1. Open a new AI assistant session in your editor.
2. Paste the contents of this file (and optionally `KNOWLEDGE_BASE.md`) into
	the chat.
3. Replace the placeholder task description at the end with your concrete,
	narrowly scoped task.
4. Let the assistant read `KNOWLEDGE_BASE.md` (and other referenced source
	files) from the repo and proceed under the constraints described above.
