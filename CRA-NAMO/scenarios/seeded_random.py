"""Scenario adapter for deterministic decision-oriented random maps."""

from __future__ import annotations

from config import Config
from scenario_generation import RandomScenarioRequest, ScenarioGenerator
from scenario_generation.fingerprint import fingerprint_manifest
from scenario_generation.serialization import candidate_from_manifest, load_manifest


def create(*, seed: int = 0, profile: str = "balanced",
           manifest_path: str | None = None,
           generation_attempts: int = 40) -> dict:
    if manifest_path:
        manifest = load_manifest(manifest_path)
        candidate = candidate_from_manifest(manifest)
        result = candidate.to_scenario_dict()
        result["manifest"] = manifest
        result["generation"].update({
            "fingerprint": fingerprint_manifest(manifest),
            "manifest_path": manifest_path,
        })
        return result

    generation_cfg = Config()
    generated = ScenarioGenerator().generate(RandomScenarioRequest(
        seed=seed, profile=profile,
        obstacle_count=generation_cfg.random_map_obstacle_count,
        event_count=generation_cfg.random_map_dynamic_obstacle_count,
        max_generation_attempts=generation_attempts))
    result = generated.to_scenario_dict()
    result["manifest"] = generated.manifest
    return result
