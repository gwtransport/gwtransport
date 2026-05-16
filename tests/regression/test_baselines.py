"""Snapshot-regression tests for the fronttracking public API.

Baselines were regenerated from the (V, θ) branch after the Phase-1 refactor.
They are NOT main-branch snapshots: main had two known bugs the refactor fixes
— (1) `concentration_at_point` returning a shock's `c_right` in the post-collision
"void" interval before the modified rarefaction arrives (~400 spurious mass on
the `freundlich_nhalf_pulse` baseline) and (2) `handle_flow_change` skipping the
recreate-with-new-flow step for waves whose head has passed the outlet, freezing
the outdated flow into c(v_outlet, t > flow_change_time) (drift on
`step_change_flow_n2`). The (V, θ) refactor sidesteps both defects: the
recorded θ-domain quantities reflect the physically correct profile.

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


def test_theta_first_arrival_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["theta_first_arrival"] == baseline["theta_first_arrival"]


def test_n_waves_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["n_waves"] == baseline["n_waves"]


def test_cout_matches(scenario_run):
    _, current, baseline = scenario_run
    np.testing.assert_array_equal(current["cout"], baseline["cout"])


def test_domain_mass_matches(scenario_run):
    _, current, baseline = scenario_run
    np.testing.assert_array_equal(current["domain_mass"], baseline["domain_mass"])


def test_total_outlet_mass_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["total_outlet_mass"] == baseline["total_outlet_mass"]


def test_theta_integration_end_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["theta_integration_end"] == baseline["theta_integration_end"]


def test_event_summary_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["event_summary"] == baseline["event_summary"]


def test_n_events_matches(scenario_run):
    _, current, baseline = scenario_run
    assert current["n_events"] == baseline["n_events"]
