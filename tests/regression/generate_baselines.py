"""Regenerate fronttracking snapshot baselines.

Run from the repo root:

    env UV_PROJECT_ENVIRONMENT=.venv-claude uv run -q python -m tests.regression.generate_baselines

This script is idempotent. The pickled outputs are tracked in git; only re-run
when intentionally locking in a new baseline. Phase 1 of the (V, θ) refactor
asserts these snapshots are bit-identical — do NOT regenerate during the
refactor.
"""

from __future__ import annotations

import sys
import time

from tests.regression.scenarios import SCENARIOS, run_scenario, save_baseline


def main() -> int:
    """Run every scenario, pickle its outputs, and print a one-line summary per scenario."""
    for scenario in SCENARIOS:
        t0 = time.perf_counter()
        result = run_scenario(scenario)
        elapsed = time.perf_counter() - t0
        save_baseline(scenario, result)
        cout = result["cout"]
        print(  # noqa: T201 — CLI utility, print is the intended output channel
            f"  {scenario.name:32s} | {elapsed:5.2f}s | "
            f"events={result['n_events']:4d} | mass={result['total_outlet_mass']:12.6f} | "
            f"cout_sum={float(cout.sum()):12.6f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
