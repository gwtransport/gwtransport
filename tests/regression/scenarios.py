"""Scenario specifications for fronttracking snapshot regression.

Each `Scenario` is a stable, hashable description of inputs (cin, flow, tedges,
v_pore, sorption). The `run_scenario` function executes a `FrontTracker`
simulation and captures user-facing outputs (breakthrough curve, domain mass,
total outlet mass, event-ordering summary).

The captured outputs are pickled to ``baselines/<name>.pkl`` by
``generate_baselines.py`` and asserted bit-identical by ``test_baselines.py``.
"""

from __future__ import annotations

import pickle  # noqa: S403 — baselines are trusted local artifacts, not external data
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from gwtransport.fronttracking.math import (
    ConstantRetardation,
    FreundlichSorption,
    LangmuirSorption,
    SorptionModel,
)
from gwtransport.fronttracking.output import (
    compute_breakthrough_curve,
    compute_domain_mass,
    compute_total_outlet_mass,
)
from gwtransport.fronttracking.solver import FrontTracker

BASELINES_DIR = Path(__file__).parent / "baselines"


@dataclass(frozen=True)
class Scenario:
    """Frozen specification of a fronttracking simulation scenario.

    Parameters
    ----------
    name : str
        Unique identifier used as the baseline-pickle filename stem.
    cin : tuple of float
        Inlet concentration per bin.
    flow : tuple of float
        Flow rate per bin [m³/day].
    v_pore : float
        Outlet position / aquifer pore volume [m³].
    sorption_kind : {'freundlich', 'langmuir', 'constant'}
        Sorption model class.
    sorption_params : tuple of tuple
        Ordered ((name, value), ...) pairs for the sorption constructor.
    t_start : str
        ISO-format date for ``tedges[0]``.
    t_sample_count : int
        Number of evenly-spaced query times in ``[0, n_bins]`` for the
        breakthrough curve.
    domain_mass_times : tuple of float
        Times [days from tedges[0]] at which to evaluate ``compute_domain_mass``.
    """

    name: str
    cin: tuple[float, ...]
    flow: tuple[float, ...]
    v_pore: float
    sorption_kind: str
    sorption_params: tuple[tuple[str, float], ...]
    t_start: str = "2020-01-01"
    t_sample_count: int = 1001
    domain_mass_times: tuple[float, ...] = ()


def _make_sorption(scenario: Scenario) -> SorptionModel:
    params = dict(scenario.sorption_params)
    if scenario.sorption_kind == "freundlich":
        return FreundlichSorption(**params)
    if scenario.sorption_kind == "langmuir":
        return LangmuirSorption(**params)
    if scenario.sorption_kind == "constant":
        return ConstantRetardation(**params)
    msg = f"Unknown sorption_kind: {scenario.sorption_kind!r}"
    raise ValueError(msg)


def _build_inputs(scenario: Scenario) -> dict:
    cin = np.asarray(scenario.cin, dtype=float)
    flow = np.asarray(scenario.flow, dtype=float)
    n_bins = len(cin)
    if len(flow) != n_bins:
        msg = f"cin and flow length mismatch: {len(cin)} vs {len(flow)}"
        raise ValueError(msg)
    tedges = pd.date_range(scenario.t_start, periods=n_bins + 1, freq="D")
    tedges_days = np.asarray((tedges - tedges[0]) / pd.Timedelta(days=1), dtype=float)
    t_sample = np.linspace(0.0, float(n_bins), scenario.t_sample_count)
    domain_mass_times = (
        np.asarray(scenario.domain_mass_times, dtype=float)
        if scenario.domain_mass_times
        else np.linspace(0.0, float(n_bins), 7)[1:-1]
    )
    return {
        "cin": cin,
        "flow": flow,
        "tedges": tedges,
        "tedges_days": tedges_days,
        "t_sample": t_sample,
        "domain_mass_times": domain_mass_times,
    }


def run_scenario(scenario: Scenario) -> dict:
    """Run the scenario through the current public API and capture outputs.

    The capture is the contract for snapshot regression: anything in the returned
    dict must be reproducible bit-identically by the refactored code.
    """
    inputs = _build_inputs(scenario)
    sorption = _make_sorption(scenario)

    tracker = FrontTracker(
        cin=inputs["cin"],
        flow=inputs["flow"],
        tedges=inputs["tedges"],
        aquifer_pore_volume=scenario.v_pore,
        sorption=sorption,
    )
    tracker.run(max_iterations=100000, verbose=False)

    cout: npt.NDArray[np.floating] = compute_breakthrough_curve(
        inputs["t_sample"],
        scenario.v_pore,
        tracker.state.waves,
        sorption,
        theta_edges=tracker.state.theta_edges,
        tedges_days=inputs["tedges_days"],
    )

    domain_mass = np.array([
        compute_domain_mass(tracker.state.theta_at_t(float(t)), scenario.v_pore, tracker.state.waves, sorption)
        for t in inputs["domain_mass_times"]
    ])

    total_mass, t_integration_end = compute_total_outlet_mass(
        scenario.v_pore,
        tracker.state.waves,
        sorption,
        inputs["flow"],
        inputs["tedges_days"],
    )

    event_summary = tuple((float(ev["time"]), str(ev["type"])) for ev in tracker.state.events)

    return {
        "scenario_name": scenario.name,
        "t_first_arrival": float(tracker.t_first_arrival),
        "n_waves": len(tracker.state.waves),
        "n_events": len(tracker.state.events),
        "event_summary": event_summary,
        "cout": cout,
        "domain_mass_times": inputs["domain_mass_times"],
        "domain_mass": domain_mass,
        "total_outlet_mass": float(total_mass),
        "t_integration_end": float(t_integration_end),
        "t_sample": inputs["t_sample"],
    }


def baseline_path(scenario: Scenario) -> Path:
    """Return the on-disk path for the scenario's pickled baseline."""
    return BASELINES_DIR / f"{scenario.name}.pkl"


def save_baseline(scenario: Scenario, result: dict) -> None:
    """Pickle ``result`` to the scenario's baseline path, creating dirs as needed."""
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    with baseline_path(scenario).open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_baseline(scenario: Scenario) -> dict:
    """Load the previously captured baseline dict for ``scenario``."""
    with baseline_path(scenario).open("rb") as f:
        return pickle.load(f)  # noqa: S301 — baselines are trusted local artifacts


_FREUNDLICH_N2_PARAMS: tuple[tuple[str, float], ...] = (
    ("k_f", 0.01),
    ("n", 2.0),
    ("bulk_density", 1500.0),
    ("porosity", 0.3),
)

_FREUNDLICH_NHALF_PARAMS: tuple[tuple[str, float], ...] = (
    ("k_f", 0.01),
    ("n", 0.5),
    ("bulk_density", 1500.0),
    ("porosity", 0.3),
)

_LANGMUIR_PARAMS: tuple[tuple[str, float], ...] = (
    ("s_max", 0.1),
    ("k_l", 2.0),
    ("bulk_density", 1500.0),
    ("porosity", 0.3),
)

_CONSTANT_PARAMS: tuple[tuple[str, float], ...] = (("retardation_factor", 2.0),)


def _pulse_cin(n_bins: int, *, start: int, end: int, c_step: float) -> tuple[float, ...]:
    arr = np.zeros(n_bins)
    arr[start:end] = c_step
    return tuple(arr.tolist())


def _constant_flow(n_bins: int, flow_val: float) -> tuple[float, ...]:
    return tuple([float(flow_val)] * n_bins)


def _step_flow(n_bins: int, base: float, scale: float, step_at: int) -> tuple[float, ...]:
    arr = np.full(n_bins, float(base))
    arr[step_at:] = float(base) * float(scale)
    return tuple(arr.tolist())


def _with_zeros_flow(n_bins: int, flow_val: float, zero_ranges: tuple[tuple[int, int], ...]) -> tuple[float, ...]:
    arr = np.full(n_bins, float(flow_val))
    for lo, hi in zero_ranges:
        arr[lo:hi] = 0.0
    return tuple(arr.tolist())


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="canonical_pulse_n2",
        cin=_pulse_cin(500, start=5, end=15, c_step=4.0),
        flow=_constant_flow(500, 100.0),
        v_pore=200.0,
        sorption_kind="freundlich",
        sorption_params=_FREUNDLICH_N2_PARAMS,
        domain_mass_times=(20.0, 50.0, 100.0, 200.0, 300.0),
    ),
    Scenario(
        name="freundlich_nhalf_pulse",
        cin=_pulse_cin(200, start=5, end=15, c_step=4.0),
        flow=_constant_flow(200, 100.0),
        v_pore=200.0,
        sorption_kind="freundlich",
        sorption_params=_FREUNDLICH_NHALF_PARAMS,
        domain_mass_times=(20.0, 50.0, 80.0, 120.0, 160.0),
    ),
    Scenario(
        name="langmuir_pulse",
        cin=_pulse_cin(200, start=5, end=15, c_step=4.0),
        flow=_constant_flow(200, 100.0),
        v_pore=200.0,
        sorption_kind="langmuir",
        sorption_params=_LANGMUIR_PARAMS,
        domain_mass_times=(20.0, 50.0, 80.0, 120.0, 160.0),
    ),
    Scenario(
        name="constant_retardation_pulse",
        cin=_pulse_cin(100, start=5, end=15, c_step=4.0),
        flow=_constant_flow(100, 100.0),
        v_pore=200.0,
        sorption_kind="constant",
        sorption_params=_CONSTANT_PARAMS,
        domain_mass_times=(10.0, 25.0, 50.0, 75.0, 90.0),
    ),
    Scenario(
        name="step_change_flow_n2",
        cin=_pulse_cin(200, start=5, end=15, c_step=4.0),
        flow=_step_flow(200, 100.0, 1.5, step_at=80),
        v_pore=200.0,
        sorption_kind="freundlich",
        sorption_params=_FREUNDLICH_N2_PARAMS,
        domain_mass_times=(20.0, 50.0, 80.0, 120.0, 160.0),
    ),
    Scenario(
        name="zero_flow_bins_n2",
        cin=_pulse_cin(150, start=55, end=65, c_step=4.0),
        flow=_with_zeros_flow(150, 100.0, ((10, 13), (25, 28))),
        v_pore=200.0,
        sorption_kind="freundlich",
        sorption_params=_FREUNDLICH_N2_PARAMS,
        domain_mass_times=(30.0, 60.0, 80.0, 100.0, 130.0),
    ),
)
