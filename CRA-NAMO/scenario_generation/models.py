"""Data contracts used by the random-map pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shapely.geometry import Polygon

from config import Config
from obstacle import MovableObstacle, StaticObstacle


@dataclass(frozen=True)
class RandomScenarioRequest:
    """Everything that may intentionally change a generated scenario."""

    seed: int
    profile: str = "balanced"
    width: float = 24.0
    height: float = 14.0
    topology: str | None = None
    min_decision_points: int = 3
    max_decision_points: int = 4
    obstacle_count: int | tuple[int, int] = (8, 12)
    event_count: int | tuple[int, int] = (0, 1)
    max_generation_attempts: int = 40

    def __post_init__(self) -> None:
        if self.width < 16.0 or self.height < 10.0:
            raise ValueError("random maps require at least a 16 x 10 m workspace")
        if self.min_decision_points < 1:
            raise ValueError("min_decision_points must be positive")
        if self.max_decision_points < self.min_decision_points:
            raise ValueError("max_decision_points must be >= min_decision_points")
        if self.max_decision_points < 3 or self.max_decision_points > 4:
            raise ValueError("generator v1 supports three or four decision points")
        lo, hi = ((self.obstacle_count, self.obstacle_count)
                  if isinstance(self.obstacle_count, int)
                  else self.obstacle_count)
        if lo < 1 or hi < lo:
            raise ValueError("obstacle_count must be a non-empty increasing range")
        elo, ehi = ((self.event_count, self.event_count)
                    if isinstance(self.event_count, int)
                    else self.event_count)
        if elo < 0 or ehi < elo:
            raise ValueError("event_count must be a non-negative increasing range")
        if self.max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be positive")


@dataclass
class DecisionSpec:
    name: str
    kind: str
    involved_oids: list[int]
    alternatives: list[str]
    expected_tradeoff: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def runtime_record(self) -> dict[str, Any]:
        record = {"name": self.name, "kind": self.kind}
        record.update(self.metadata)
        return record


@dataclass
class TopologyResult:
    workspace: Polygon
    walls: list[StaticObstacle]
    start: tuple[float, float]
    goal: tuple[float, float]
    anchors: dict[str, Any]
    reserved_regions: list[Polygon] = field(default_factory=list)


@dataclass
class CandidateScenario:
    workspace: Polygon
    static: list[StaticObstacle]
    movable: list[MovableObstacle]
    start: tuple[float, float]
    goal: tuple[float, float]
    cfg: Config
    events: list[Any] = field(default_factory=list)
    decision_points: list[DecisionSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_scenario_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "static": self.static,
            "movable": self.movable,
            "start": self.start,
            "goal": self.goal,
            "cfg": self.cfg,
            "dynamics": self.events,
            "decision_points": [p.runtime_record() for p in self.decision_points],
            "generation": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationIssue:
    stage: str
    code: str
    message: str
    fatal: bool = True


@dataclass
class ValidationReport:
    accepted: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    scenario: CandidateScenario
    manifest: dict[str, Any]
    validation: ValidationReport
    attempts: int
    fingerprint: str

    def to_scenario_dict(self) -> dict[str, Any]:
        result = self.scenario.to_scenario_dict()
        result["generation"].update({
            "attempts": self.attempts,
            "fingerprint": self.fingerprint,
            "validation": dict(self.validation.metrics),
        })
        return result


class ScenarioGenerationError(RuntimeError):
    """Raised after every deterministic candidate for a request is rejected."""
