"""End-to-end regression tests for all shipped scenarios."""

from __future__ import annotations

import time

import pytest

import scenarios
from executor import OnlineNAMO


SCENARIOS = (
    "corridor",
    "maze_doors_complex",
    "moving_depot",
    "strategy_demo",
)


def _format_result(name, result, wall_time: float) -> str:
    events = ", ".join(result.world_events) if result.world_events else "none"
    return (
        f"\n[scenario] {name}\n"
        f"  success       : {result.success} ({result.message})\n"
        f"  wall time     : {wall_time:.3f} s\n"
        f"  objective C   : {result.C:,.4f}\n"
        f"  energy J      : {result.J:,.4f}\n"
        f"  simulated T   : {result.T:,.4f} s\n"
        f"  replans       : {result.cycles:,}\n"
        f"  expansions    : {result.total_expansions:,}\n"
        f"  moved         : {result.removed or 'none'}\n"
        f"  world events  : {events}\n"
    )


@pytest.mark.scenario
@pytest.mark.parametrize("scenario_name", SCENARIOS, ids=SCENARIOS)
def test_scenario_reaches_goal(scenario_name, monkeypatch, tmp_path):
    """Run one scenario offline and require the robot to reach its goal."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    scenario = scenarios.load(scenario_name)
    cfg = scenario["cfg"]
    cfg.deepseek_api_key = ""
    cfg.use_llm_ordering = False
    cfg.plan_time_in_clock = False
    cfg.save_frames = False
    cfg.verbose = False
    cfg.out_dir = str(tmp_path)

    simulation = OnlineNAMO(
        scenario["workspace"],
        scenario["static"],
        scenario["movable"],
        scenario["start"],
        scenario["goal"],
        cfg,
        events=scenario.get("dynamics"),
    )

    started = time.perf_counter()
    result = simulation.run()
    wall_time = time.perf_counter() - started

    print(_format_result(scenario_name, result, wall_time), flush=True)

    assert result.llm_mode == "heuristic"
    assert result.llm_calls == 0
    assert result.success, result.message

