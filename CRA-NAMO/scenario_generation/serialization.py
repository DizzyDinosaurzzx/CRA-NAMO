"""Lossless JSON persistence for generated scenarios."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from shapely.geometry import Polygon

from config import Config
from dynamics import AfterMoved, AtTime, Event, Halt, MoveTo, Mutate, NearPoint
from obstacle import MovableObstacle, StaticObstacle
from scenario_generation.fingerprint import fingerprint_manifest
from scenario_generation.models import CandidateScenario, DecisionSpec

GENERATOR_VERSION = 1


def _config_dict(cfg: Config) -> dict:
    result = {}
    for item in fields(cfg):
        if not item.init or item.name.startswith("_"):
            continue
        value = getattr(cfg, item.name)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[item.name] = value
    # Credentials are runtime configuration, never scenario data.
    result.pop("deepseek_api_key", None)
    # Corpus size does not describe an individual map and must not affect its
    # content fingerprint.
    for name in (
            "random_map_experiment_count", "random_map_generate_images",
            "random_map_run_strategies", "random_map_timeout_seconds",
            "random_map_resume", "random_map_seed_start",
            "random_map_output_dir"):
        result.pop(name, None)
    return result


def _trigger_dict(trigger) -> dict:
    if isinstance(trigger, AtTime):
        return {"type": "at_time", "t": trigger.t}
    if isinstance(trigger, NearPoint):
        return {"type": "near_point", "at": list(trigger.at),
                "radius": trigger.radius}
    if isinstance(trigger, AfterMoved):
        return {"type": "after_moved", "oid": trigger.oid}
    raise TypeError(f"unsupported generated trigger {type(trigger).__name__}")


def _effect_dict(effect) -> dict:
    if isinstance(effect, MoveTo):
        return {"type": "move_to", "oid": effect.oid,
                "goal": list(effect.goal), "speed": effect.speed}
    if isinstance(effect, Halt):
        return {"type": "halt", "oid": effect.oid}
    if isinstance(effect, Mutate):
        result = {"type": "mutate", "oid": effect.oid}
        for name in ("material", "l", "d", "h", "difficulty", "contact_reveals"):
            value = getattr(effect, name)
            if value is not None:
                result[name] = value
        return result
    raise TypeError(f"unsupported generated effect {type(effect).__name__}")


def manifest_from_candidate(candidate: CandidateScenario, *, seed: int,
                            profile: str, attempt: int,
                            validation: dict) -> dict:
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "seed": int(seed),
        "profile": profile,
        "attempt": int(attempt),
        "topology": {
            "name": candidate.metadata.get("topology", "unknown"),
            "family": candidate.metadata.get("topology_family", "unknown"),
            "metrics": dict(candidate.metadata.get("topology_metrics", {})),
        },
        "calibration": dict(candidate.metadata.get("calibration", {})),
        "workspace": list(candidate.workspace.bounds),
        "walls": [{
            "name": wall.name,
            "theta": wall.theta,
            "polygon": [list(p) for p in list(wall.polygon.exterior.coords)[:-1]],
        } for wall in candidate.static],
        "obstacles": [{
            "oid": obs.oid, "x": obs.x, "y": obs.y,
            "l": obs.l, "d": obs.d, "h": obs.h, "theta": obs.theta,
            "material": obs.material, "difficulty": obs.difficulty,
            "contact_reveals": obs.contact_reveals,
            "interacts_with": list(obs.interacts_with),
            "interaction_risk": obs.interaction_risk,
        } for obs in candidate.movable],
        "start": list(candidate.start),
        "goal": list(candidate.goal),
        "events": [{
            "name": event.name,
            "trigger": _trigger_dict(event.trigger),
            "effect": _effect_dict(event.effect),
        } for event in candidate.events],
        "decision_points": [{
            "name": point.name,
            "kind": point.kind,
            "involved_oids": point.involved_oids,
            "alternatives": point.alternatives,
            "expected_tradeoff": point.expected_tradeoff,
            "metadata": point.metadata,
        } for point in candidate.decision_points],
        "config": _config_dict(candidate.cfg),
        "validation": validation,
    }
    manifest["fingerprint"] = fingerprint_manifest(manifest)
    return manifest


def _trigger_from(data: dict):
    kind = data["type"]
    if kind == "at_time":
        return AtTime(float(data["t"]))
    if kind == "near_point":
        return NearPoint(tuple(data["at"]), float(data["radius"]))
    if kind == "after_moved":
        return AfterMoved(int(data["oid"]))
    raise ValueError(f"unknown trigger type {kind!r}")


def _effect_from(data: dict):
    kind = data["type"]
    if kind == "move_to":
        return MoveTo(int(data["oid"]), tuple(data["goal"]), data.get("speed"))
    if kind == "halt":
        return Halt(int(data["oid"]))
    if kind == "mutate":
        values = dict(data)
        values.pop("type")
        return Mutate(**values)
    raise ValueError(f"unknown effect type {kind!r}")


def candidate_from_manifest(manifest: dict) -> CandidateScenario:
    if int(manifest.get("generator_version", -1)) != GENERATOR_VERSION:
        raise ValueError("unsupported random-map generator version")
    expected = manifest.get("fingerprint")
    actual = fingerprint_manifest(manifest)
    if expected and expected != actual:
        raise ValueError(f"manifest fingerprint mismatch: expected {expected}, got {actual}")
    xmin, ymin, xmax, ymax = manifest["workspace"]
    workspace = Polygon(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))
    walls = [StaticObstacle(Polygon(row["polygon"]), row["name"], row.get("theta"))
             for row in manifest["walls"]]
    movable = [MovableObstacle(
        oid=int(row["oid"]), x=float(row["x"]), y=float(row["y"]),
        l=float(row["l"]), d=float(row["d"]), h=float(row["h"]),
        theta=float(row["theta"]), material=row["material"],
        difficulty=float(row["difficulty"]),
        contact_reveals=row.get("contact_reveals", ""),
        interacts_with=tuple(row.get("interacts_with", ())),
        interaction_risk=row.get("interaction_risk", ""))
        for row in manifest["obstacles"]]
    events = [Event(name=row.get("name", ""),
                    trigger=_trigger_from(row["trigger"]),
                    effect=_effect_from(row["effect"]))
              for row in manifest.get("events", ())]
    decisions = [DecisionSpec(
        name=row["name"], kind=row["kind"],
        involved_oids=list(row.get("involved_oids", ())),
        alternatives=list(row.get("alternatives", ())),
        expected_tradeoff=row.get("expected_tradeoff", ""),
        metadata=dict(row.get("metadata", {})))
        for row in manifest.get("decision_points", ())]
    cfg = Config(**manifest.get("config", {}))
    topology = manifest.get("topology", {})
    return CandidateScenario(
        workspace=workspace, static=walls, movable=movable,
        start=tuple(manifest["start"]), goal=tuple(manifest["goal"]),
        cfg=cfg, events=events, decision_points=decisions,
        metadata={
            "seed": manifest["seed"], "profile": manifest["profile"],
            "manifest": True,
            "topology": topology.get("name", "unknown"),
            "topology_family": topology.get("family", "unknown"),
            "topology_metrics": dict(topology.get("metrics", {})),
            "calibration": dict(manifest.get("calibration", {})),
        })


def save_manifest(manifest: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
