"""Centralised generation preferences; samplers contain no profile magic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    topology_weights: dict[str, float]
    wall_angle_sigma: float
    dynamic_probability: float
    hidden_probability: float
    background_density: float
    time_importance: float


PROFILES: dict[str, GenerationProfile] = {
    "balanced": GenerationProfile(
        "balanced", {"room_graph": 0.50, "junction": 0.25,
                     "branching": 0.15, "corridor": 0.10},
        wall_angle_sigma=0.035, dynamic_probability=0.55,
        hidden_probability=0.65, background_density=1.0,
        time_importance=0.25),
    "dynamic": GenerationProfile(
        "dynamic", {"room_graph": 0.35, "junction": 0.45,
                    "corridor": 0.20},
        wall_angle_sigma=0.025, dynamic_probability=1.0,
        hidden_probability=0.35, background_density=0.8,
        time_importance=0.45),
    "risk": GenerationProfile(
        "risk", {"room_graph": 0.55, "junction": 0.20,
                 "branching": 0.25},
        wall_angle_sigma=0.03, dynamic_probability=0.35,
        hidden_probability=0.55, background_density=0.9,
        time_importance=0.2),
    "manipulation": GenerationProfile(
        "manipulation", {"room_graph": 0.35, "corridor": 0.45,
                         "branching": 0.20},
        wall_angle_sigma=0.06, dynamic_probability=0.25,
        hidden_probability=0.4, background_density=1.15,
        time_importance=0.15),
    "adversarial": GenerationProfile(
        "adversarial", {"room_graph": 0.55, "junction": 0.25,
                        "corridor": 0.20},
        wall_angle_sigma=0.07, dynamic_probability=0.8,
        hidden_probability=0.85, background_density=1.25,
        time_importance=0.35),
    "showcase": GenerationProfile(
        "showcase", {"room_graph": 0.60, "junction": 0.40},
        wall_angle_sigma=0.02, dynamic_probability=1.0,
        hidden_probability=1.0, background_density=0.75,
        time_importance=0.3),
    "benchmark": GenerationProfile(
        "benchmark", {"room_graph": 0.65, "junction": 0.35},
        wall_angle_sigma=0.045, dynamic_probability=0.5,
        hidden_probability=0.5, background_density=1.0,
        time_importance=0.25),
}


def get_profile(name: str) -> GenerationProfile:
    key = str(name).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown random-map profile {name!r}; available: "
                         + ", ".join(sorted(PROFILES)))
    return PROFILES[key]


def weighted_choice(weights: dict[str, float], rng) -> str:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0.0:
        raise ValueError("at least one topology weight must be positive")
    pick = rng.random() * total
    for name, weight in weights.items():
        pick -= max(0.0, float(weight))
        if pick <= 0.0:
            return name
    return next(reversed(weights))
