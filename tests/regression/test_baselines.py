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


def test_n_events_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["n_events"] == baseline["n_events"]


def test_event_summary_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["event_summary"] == baseline["event_summary"]


def test_cout_matches(scenario_run):
    _, current, baseline = scenario_run
    np.testing.assert_array_equal(current["cout"], baseline["cout"])


def test_domain_mass_matches(scenario_run):
    _, current, baseline = scenario_run
    np.testing.assert_array_equal(current["domain_mass"], baseline["domain_mass"])


def test_total_outlet_mass_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["total_outlet_mass"] == baseline["total_outlet_mass"]


def test_t_integration_end_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["t_integration_end"] == baseline["t_integration_end"]
