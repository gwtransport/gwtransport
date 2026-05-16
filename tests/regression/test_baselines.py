"""Snapshot-regression tests for the fronttracking public API.

The pickled baselines in ``baselines/`` were captured on ``main`` (commit
103aab3) before the (V, θ) refactor. Phase 1 must reproduce every captured
output bit-for-bit so the architectural rewrite is provably behavior-preserving.

Regeneration: see ``generate_baselines.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.regression.scenarios import SCENARIOS, Scenario, load_baseline, run_scenario


@pytest.fixture(scope="module", params=SCENARIOS, ids=lambda s: s.name)
def scenario_run(request) -> tuple[Scenario, dict, dict]:
    scenario: Scenario = request.param
    current = run_scenario(scenario)
    baseline = load_baseline(scenario)
    return scenario, current, baseline


def test_t_first_arrival_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["t_first_arrival"] == baseline["t_first_arrival"]


def test_n_waves_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["n_waves"] == baseline["n_waves"]


def test_cout_matches(scenario_run):
    _, current, baseline = scenario_run
    # Phase 1 reformulates mass as ∫c dθ (was Σ flow·∫c dt); the rarefaction
    # antiderivative's coefficient absorbs flow differently. Mathematically
    # identical, FP-summation-equivalent to ~few ULPs (physics-math review
    # measured 9 ULPs at magnitude 3750 → ~1.2e-12 rtol). Keep rtol very tight
    # so the test still catches any genuine math regression.
    np.testing.assert_allclose(current["cout"], baseline["cout"], rtol=1e-11, atol=0.0)


def test_domain_mass_matches(scenario_run):
    _, current, baseline = scenario_run
    np.testing.assert_allclose(current["domain_mass"], baseline["domain_mass"], rtol=1e-11, atol=0.0)


def test_total_outlet_mass_matches(scenario_run):
    _, current, baseline = scenario_run
    np.testing.assert_allclose(current["total_outlet_mass"], baseline["total_outlet_mass"], rtol=1e-11, atol=0.0)


def test_t_integration_end_matches(scenario_run):
    _, current, baseline = scenario_run
    np.testing.assert_allclose(current["t_integration_end"], baseline["t_integration_end"], rtol=1e-11, atol=0.0)


# n_events and event_summary are NOT part of the public-output contract.
# Phase 1 deleted FLOW_CHANGE events (handle_flow_change removed), so varying-flow
# scenarios genuinely have a different event stream. The breakthrough curve and
# mass remain bit-identical (within ULP) — the user-visible contract.
