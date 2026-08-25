"""Estimate manipulation risk and convert it to a cost surcharge."""

from __future__ import annotations

import os
import re
from typing import Dict, Optional

import requests

from config import Config

LOW = "low"
MEDIUM = "medium"
MEDIUM_HIGH = "medium_high"
HIGH = "high"
EXTREME = "extreme"

LEVELS = (LOW, MEDIUM, MEDIUM_HIGH, HIGH, EXTREME)
_ORDER = {name: i for i, name in enumerate(LEVELS)}

# What each level costs, as the detour in metres that would be worth taking to
# avoid it. Multiplied by lambda_distance in `cost`, so the ladder stays in
# proportion to the map whatever the robot's driving resistance is set to.
#
# The steps are wide on purpose. `low` is free: ordinary furniture is what a
# NAMO robot is *for*. `extreme` is far past any detour a finite map can offer,
# which is the point — a load-bearing column in a damaged building is not a
# trade-off, and the planner should route around it or fail rather than price it.
RISK_DETOUR_EQUIV_M: Dict[str, float] = {
    LOW: 0.0,
    MEDIUM: 20.0,
    MEDIUM_HIGH: 80.0,
    HIGH: 400.0,
    EXTREME: 5000.0,
}

# Whole labels, for objects whose name settles the question outright.
RISK_LABELS: Dict[str, str] = {
    # low: everyday furniture, empty containers, nothing depending on it
    "chair": LOW, "plastic_chair": LOW, "stool": LOW, "wooden_table": LOW,
    "desk": LOW, "cardboard_box": LOW, "styrofoam_box": LOW, "foam_mat": LOW,
    "empty_cart": LOW, "empty_shelf": LOW, "trash_bin": LOW, "pallet": LOW,
    "wooden_crate": LOW, "sofa": LOW, "cabinet": LOW, "concrete_block": LOW,
    "steel_shelf": LOW, "filing_cabinet": LOW, "loaded_pallet": LOW,
    "shelf": LOW, "cart": LOW, "steel_safe": LOW,
    # medium: contents that spill, break, or are worth something
    "water_cup": MEDIUM, "full_cup": MEDIUM, "aquarium": MEDIUM,
    "paint_bucket": MEDIUM, "glassware_crate": MEDIUM, "server_rack": MEDIUM,
    "lab_bench": MEDIUM, "medicine_cabinet": MEDIUM,
    # medium_high: hazardous contents, or a structure already under stress
    "gas_cylinder": MEDIUM_HIGH, "fuel_drum": MEDIUM_HIGH,
    "cracked_pillar": MEDIUM_HIGH, "earthquake_pillar": MEDIUM_HIGH,
    "debris_pile": MEDIUM_HIGH, "chemical_drum": MEDIUM_HIGH,
    "electrical_cabinet": MEDIUM_HIGH,
    # high: a person is involved, or a life-critical machine
    "occupied_wheelchair": HIGH, "wheelchair_with_person": HIGH,
    "occupied_bed": HIGH, "hospital_bed": HIGH, "stretcher": HIGH,
    "incubator": HIGH, "ventilator": HIGH, "person": HIGH,
    # extreme: something is holding the building up
    "load_bearing_column": EXTREME, "support_pillar": EXTREME,
    "structural_column": EXTREME, "shoring_prop": EXTREME,
    "collapsed_beam": EXTREME,
}

# Keywords, for labels the table has never seen. A token raises the risk to at
# least its level and never lowers it, so "occupied_wheelchair_broken" still
# reads as high even though no entry matches it whole. Order within a level does
# not matter; the maximum wins.
RISK_KEYWORDS: Dict[str, str] = {
    # medium
    "water": MEDIUM, "liquid": MEDIUM, "full": MEDIUM, "glass": MEDIUM,
    "fragile": MEDIUM, "cup": MEDIUM, "bottle": MEDIUM, "tank": MEDIUM,
    "server": MEDIUM, "lab": MEDIUM, "specimen": MEDIUM,
    # medium_high
    "gas": MEDIUM_HIGH, "fuel": MEDIUM_HIGH, "chemical": MEDIUM_HIGH,
    "explosive": MEDIUM_HIGH, "cracked": MEDIUM_HIGH, "earthquake": MEDIUM_HIGH,
    "damaged": MEDIUM_HIGH, "unstable": MEDIUM_HIGH, "debris": MEDIUM_HIGH,
    "electrical": MEDIUM_HIGH, "hazard": MEDIUM_HIGH,
    # high
    "person": HIGH, "people": HIGH, "occupied": HIGH, "patient": HIGH,
    "child": HIGH, "wheelchair": HIGH, "stretcher": HIGH, "casualty": HIGH,
    "victim": HIGH, "medical": HIGH, "ventilator": HIGH,
    # extreme
    "load_bearing": EXTREME, "loadbearing": EXTREME, "structural": EXTREME,
    "support_column": EXTREME, "shoring": EXTREME, "pillar": EXTREME,
    "column": EXTREME, "beam": EXTREME, "strut": EXTREME,
}

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _normalise(name) -> str:
    return _NON_WORD.sub("_", str(name).strip().lower()).strip("_")


def higher(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Whichever of two levels is the more dangerous. None counts as no opinion."""
    if a is None:
        return b
    if b is None:
        return a
    return a if _ORDER[a] >= _ORDER[b] else b


def keyword_level(label) -> str:
    """Risk from the label alone: whole-table first, then the worst keyword hit.

    Keywords only ever raise the level. A label that names something safe but
    contains one alarming word is treated as alarming — the failure that matters
    here is moving something that should not have been moved, not detouring
    around something that turned out to be fine.
    """
    key = _normalise(label)
    if key in RISK_LABELS:
        return RISK_LABELS[key]
    level = None
    tokens = key.split("_")
    joined = "_".join(tokens)
    for word, word_level in RISK_KEYWORDS.items():
        if word in tokens or ("_" in word and word in joined):
            level = higher(level, word_level)
    return level if level is not None else LOW


def detour_equivalent_m(level: Optional[str]) -> float:
    """Metres of detour this level is worth avoiding. Unknown levels read as low."""
    if level is None:
        return 0.0
    return RISK_DETOUR_EQUIV_M.get(level, 0.0)


class RiskEstimator:
    """Risk per obstacle, assessed on sight and re-assessed on contact.

    Holds one level per oid. `assess` fills it in the first time and is a no-op
    afterwards; `reassess` overwrites it with the better-informed verdict once
    the robot has actually touched the thing.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.level: Dict[int, str] = {}          # oid -> current level
        self.source: Dict[int, str] = {}         # oid -> how it was decided
        self.on_contact: set[int] = set()        # oids re-assessed after touching
        self.label_cache: Dict[str, str] = {}    # label -> level, to spare repeat calls
        self.calls = 0
        self.mode = "deepseek" if self.api_key else "heuristic"

    def assess(self, observation: dict) -> str:
        """First look, from the visual label. Cached: one verdict per obstacle."""
        oid = observation["oid"]
        if oid in self.level:
            return self.level[oid]
        level, source = self._decide(observation, difficulty=None)
        self.level[oid] = level
        self.source[oid] = source
        return level

    def reassess(self, observation: dict, difficulty: float) -> str:
        """Second look, once the robot has touched it and knows what it weighs.

        Replaces the first verdict — that is the point of touching. Runs once per
        obstacle; a second collision tells us nothing the first did not.
        """
        oid = observation["oid"]
        if oid in self.on_contact:
            return self.level[oid]
        level, source = self._decide(observation, difficulty=difficulty)
        self.level[oid] = level
        self.source[oid] = f"{source}-contact"
        self.on_contact.add(oid)
        return level

    def forget(self, oid: int):
        """Drop the verdict on an obstacle that is no longer what it was.

        Including the contact verdict: it was passed on an object with a
        different label or a different size, so touching it again is warranted.
        The label cache stays — labels have not changed their meaning.
        """
        self.level.pop(oid, None)
        self.source.pop(oid, None)
        self.on_contact.discard(oid)

    def level_of(self, oid: int) -> Optional[str]:
        """Current verdict, or None for an obstacle never assessed."""
        return self.level.get(oid)

    def _decide(self, o: dict, difficulty: Optional[float]):
        label = self._label(o, difficulty)
        cached = self.label_cache.get(label)
        if cached is not None:
            return cached, "cache"
        if self.api_key:
            level = self._deepseek(o, label, difficulty)
            if level is not None:
                self.label_cache[label] = level
                return level, "deepseek"
        level = keyword_level(label)
        self.label_cache[label] = level
        return level, "keyword"

    @staticmethod
    def _label(o: dict, difficulty: Optional[float]) -> str:
        """What the robot currently calls this thing.

        Contact can resolve the label into something more specific — the scenario
        supplies that as `contact_reveals`, and it is only ever visible once the
        robot has touched the obstacle.
        """
        if difficulty is not None and o.get("contact_reveals"):
            return _normalise(o["contact_reveals"])
        return _normalise(o.get("material", "unknown"))

    def _build_prompt(self, o: dict, label: str,
                      difficulty: Optional[float]) -> str:
        ladder = "\n".join(
            f"  {name:<12s} worth a detour of {RISK_DETOUR_EQUIV_M[name]:g} m to avoid"
            for name in LEVELS)
        examples = "\n".join(
            f"  {lbl:<24s} -> {lvl}" for lbl, lvl in (
                ("wooden_table", LOW),
                ("cardboard_box", LOW),
                ("water_cup", MEDIUM),
                ("gas_cylinder", MEDIUM_HIGH),
                ("cracked_pillar", MEDIUM_HIGH),
                ("occupied_wheelchair", HIGH),
                ("load_bearing_column", EXTREME),
            ))
        if difficulty is None:
            stage = ("The robot has only SEEN this obstacle, from a distance. "
                     "Judge from the label and size alone.")
        else:
            stage = ("The robot has now physically TOUCHED this obstacle and "
                     f"measured the force needed to push it: {difficulty:g} N. "
                     "A body far heavier than its label suggests may be holding "
                     "something up, or may be full rather than empty; a body far "
                     "lighter may be empty after all. Revise accordingly.")
        return (
            "Rate how dangerous it is for a mobile robot to push this obstacle "
            "out of its way. This is NOT about how hard it is to move — it is "
            "about what happens to people and to the building if it is moved.\n\n"
            "Answer with exactly one of these level names:\n"
            f"{ladder}\n\n"
            "Guidance:\n"
            "  low          ordinary furniture and empty containers; nothing "
            "spills, nobody is hurt, nothing collapses.\n"
            "  medium       contents that spill, break, or are costly.\n"
            "  medium_high  hazardous contents, or a structure already damaged "
            "or under stress.\n"
            "  high         a person is in, on, or dependent on it.\n"
            "  extreme      it is holding the structure up; moving it risks a "
            "collapse.\n\n"
            f"Calibrated examples:\n{examples}\n\n"
            f"{stage}\n\n"
            f"Obstacle label: '{label}', size l x d x h = "
            f"{float(o['l']):g} x {float(o['d']):g} x {float(o.get('h', 1.0)):g} m.\n"
            "Output ONLY the level name, nothing else."
        )

    def _parse(self, text: str) -> Optional[str]:
        found = _normalise(text)
        # longest first, so "medium_high" is not swallowed by "medium"
        for name in sorted(LEVELS, key=len, reverse=True):
            if name in found:
                return name
        return None

    def _deepseek(self, o: dict, label: str,
                  difficulty: Optional[float]) -> Optional[str]:
        prompt = self._build_prompt(o, label, difficulty)
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        body = {
            "model": self.cfg.deepseek_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
            "thinking": {"type": "enabled" if self.cfg.deepseek_thinking
                         else "disabled"},
        }
        if self.cfg.llm_max_tokens:
            body["max_tokens"] = int(self.cfg.llm_max_tokens)
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                self.calls += 1
                r = requests.post(self.cfg.deepseek_base_url, headers=headers,
                                  json=body, timeout=self.cfg.llm_timeout)
                data = r.json()
                if r.status_code >= 400:
                    error = data.get("error", data) if isinstance(data, dict) else data
                    self.cfg.log(f"[risk] HTTP {r.status_code}: {error}")
                    if not (r.status_code >= 500 or r.status_code in {408, 409, 429}):
                        return None
                    continue
                text = data["choices"][0]["message"].get("content") or ""
                level = self._parse(text)
                if level is not None:
                    return level
                self.cfg.log(f"[risk] no level in response: {text!r}")
            except Exception as e:
                self.cfg.log(f"[risk] call failed ({attempt}): {e}")
        return None
