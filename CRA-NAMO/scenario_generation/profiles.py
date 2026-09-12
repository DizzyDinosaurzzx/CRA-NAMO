"""Centralised generation preferences; samplers contain no profile magic.

A profile answers four questions about a corpus of generated maps: which
domain the obstacles come from, how many choices the robot is made to face,
how often those choices are rigged against the offline heuristic, and how much
the world moves while it decides.

The profiles are deliberately close to one another on everything except theme.
The experiment compares planning strategies, so the maps they run on should
differ in vocabulary and layout, not in how hard the corpus is.
"""

from __future__ import annotations

from dataclasses import dataclass

from scenario_generation.materials import THEMES

# Every generated map poses this many static choices, plus at most one more
# from a dynamic event.  Six keeps a 30 x 18 m room graph busy without making
# the route so long that a single run stops being readable.
DEFAULT_GATE_DECISIONS = 6


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    note: str
    # Material themes this profile may draw from; one is picked per map.
    themes: tuple[str, ...]
    topology_weights: dict[str, float]
    wall_angle_sigma: float
    # How many gate decisions to build.  A dynamic event may add one more.
    gate_decisions: int = DEFAULT_GATE_DECISIONS
    # Probability that a gate blocker is drawn from the heuristic-blind
    # vocabulary rather than the plainly-labelled one.
    trap_probability: float = 0.55
    # Probability that the map carries a moving obstacle at all.
    dynamic_probability: float = 0.35
    # Probability that the hidden-difficulty gate really hides something.
    hidden_probability: float = 0.6
    background_density: float = 1.0
    time_importance: float = 0.25


PROFILES: dict[str, GenerationProfile] = {
    "depot": GenerationProfile(
        "depot", "moving_depot and warehouse: pallets, trolleys, unlabelled chemistry",
        themes=("depot",),
        topology_weights={"room_graph": 0.45, "junction": 0.25,
                          "corridor": 0.20, "branching": 0.10},
        wall_angle_sigma=0.035, trap_probability=0.55,
        dynamic_probability=0.40, hidden_probability=0.60,
        background_density=1.0, time_importance=0.25),
    "home": GenerationProfile(
        "home", "home and maze: furniture, and the two things you never shove",
        themes=("home",),
        topology_weights={"room_graph": 0.60, "junction": 0.25,
                          "branching": 0.15},
        wall_angle_sigma=0.030, trap_probability=0.50,
        dynamic_probability=0.25, hidden_probability=0.65,
        background_density=1.1, time_importance=0.20),
    "hospital": GenerationProfile(
        "hospital", "hospital: ward furniture whose danger is procedural",
        themes=("hospital",),
        topology_weights={"room_graph": 0.50, "junction": 0.30,
                          "corridor": 0.20},
        wall_angle_sigma=0.025, trap_probability=0.65,
        dynamic_probability=0.35, hidden_probability=0.55,
        background_density=0.9, time_importance=0.32),
    "earthquake": GenerationProfile(
        "earthquake", "earthquake: the cheapest push is the structural one",
        themes=("earthquake",),
        topology_weights={"room_graph": 0.55, "branching": 0.25,
                          "junction": 0.20},
        wall_angle_sigma=0.060, trap_probability=0.70,
        dynamic_probability=0.30, hidden_probability=0.70,
        background_density=1.15, time_importance=0.20),
    "benchmark": GenerationProfile(
        "benchmark", "one corpus spanning all four vocabularies",
        themes=("depot", "home", "hospital", "earthquake"),
        topology_weights={"room_graph": 0.55, "junction": 0.30,
                          "corridor": 0.15},
        wall_angle_sigma=0.040, trap_probability=0.60,
        dynamic_probability=0.35, hidden_probability=0.60,
        background_density=1.0, time_importance=0.25),
}

for _profile in PROFILES.values():
    unknown = [name for name in _profile.themes if name not in THEMES]
    if unknown:
        raise ValueError(
            f"profile {_profile.name!r} names unknown material themes {unknown}")
    if _profile.gate_decisions < 1:
        raise ValueError(f"profile {_profile.name!r} needs at least one gate")
del _profile


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


def pick_theme(profile: GenerationProfile, rng) -> str:
    """Choose this map's material vocabulary."""
    return profile.themes[rng.randrange(len(profile.themes))]
