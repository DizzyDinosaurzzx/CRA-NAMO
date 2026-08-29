"""Estimate manipulation risk and convert it to a cost surcharge."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Optional, Sequence

import requests

from config import Config
from llm_difficulty import friction_force, material_mu_rho

LOW = "low"
MEDIUM = "medium"
MEDIUM_HIGH = "medium_high"
HIGH = "high"
EXTREME = "extreme"

LEVELS = (LOW, MEDIUM, MEDIUM_HIGH, HIGH, EXTREME)
_ORDER = {name: i for i, name in enumerate(LEVELS)}

# Detour-equivalent surcharge for each risk level.
RISK_DETOUR_EQUIV_M: Dict[str, float] = {
    LOW: 0.0,
    MEDIUM: 20.0,
    MEDIUM_HIGH: 80.0,
    HIGH: 400.0,
    EXTREME: 5000.0,
}

# Exact labels with known risk levels.
RISK_LABELS: Dict[str, str] = {
    "chair": LOW, "plastic_chair": LOW, "stool": LOW, "wooden_table": LOW,
    "desk": LOW, "cardboard_box": LOW, "styrofoam_box": LOW, "foam_mat": LOW,
    "empty_cart": LOW, "empty_shelf": LOW, "trash_bin": LOW, "pallet": LOW,
    "wooden_crate": LOW, "sofa": LOW, "cabinet": LOW, "concrete_block": LOW,
    "steel_shelf": LOW, "filing_cabinet": LOW, "loaded_pallet": LOW,
    "shelf": LOW, "cart": LOW, "steel_safe": LOW,
    "water_cup": MEDIUM, "full_cup": MEDIUM, "aquarium": MEDIUM,
    "paint_bucket": MEDIUM, "glassware_crate": MEDIUM, "server_rack": MEDIUM,
    "lab_bench": MEDIUM, "medicine_cabinet": MEDIUM,
    "gas_cylinder": MEDIUM_HIGH, "fuel_drum": MEDIUM_HIGH,
    "cracked_pillar": MEDIUM_HIGH, "earthquake_pillar": MEDIUM_HIGH,
    "debris_pile": MEDIUM_HIGH, "chemical_drum": MEDIUM_HIGH,
    "electrical_cabinet": MEDIUM_HIGH,
    "occupied_wheelchair": HIGH, "wheelchair_with_person": HIGH,
    "occupied_bed": HIGH, "hospital_bed": HIGH, "stretcher": HIGH,
    "incubator": HIGH, "ventilator": HIGH, "person": HIGH,
    "load_bearing_column": EXTREME, "support_pillar": EXTREME,
    "structural_column": EXTREME, "shoring_prop": EXTREME,
    "collapsed_beam": EXTREME,
}

# Keyword fallback for labels absent from the exact table.
RISK_KEYWORDS: Dict[str, str] = {
    "water": MEDIUM, "liquid": MEDIUM, "full": MEDIUM, "glass": MEDIUM,
    "fragile": MEDIUM, "cup": MEDIUM, "bottle": MEDIUM, "tank": MEDIUM,
    "server": MEDIUM, "lab": MEDIUM, "specimen": MEDIUM,
    "gas": MEDIUM_HIGH, "fuel": MEDIUM_HIGH, "chemical": MEDIUM_HIGH,
    "explosive": MEDIUM_HIGH, "cracked": MEDIUM_HIGH, "earthquake": MEDIUM_HIGH,
    "damaged": MEDIUM_HIGH, "unstable": MEDIUM_HIGH, "debris": MEDIUM_HIGH,
    "electrical": MEDIUM_HIGH, "hazard": MEDIUM_HIGH,
    "person": HIGH, "people": HIGH, "occupied": HIGH, "patient": HIGH,
    "child": HIGH, "wheelchair": HIGH, "stretcher": HIGH, "casualty": HIGH,
    "victim": HIGH, "medical": HIGH, "ventilator": HIGH,
    "load_bearing": EXTREME, "loadbearing": EXTREME, "structural": EXTREME,
    "support_column": EXTREME, "shoring": EXTREME, "pillar": EXTREME,
    "column": EXTREME, "beam": EXTREME, "strut": EXTREME,
}

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _normalise(name) -> str:
    return _NON_WORD.sub("_", str(name).strip().lower()).strip("_")


def step_up(level: Optional[str]) -> str:
    """One rung further up the ladder, or the top if already there."""
    if level is None:
        return LEVELS[0]
    return LEVELS[min(_ORDER[level] + 1, len(LEVELS) - 1)]


def higher(a: Optional[str], b: Optional[str]) -> Optional[str]:
    """Whichever of two levels is the more dangerous. None counts as no opinion."""
    if a is None:
        return b
    if b is None:
        return a
    return a if _ORDER[a] >= _ORDER[b] else b


def keyword_level(label) -> str:
    """Return the highest risk implied by an exact label or keyword."""
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


def label_of(o: dict, difficulty: Optional[float] = None) -> str:
    """Return the currently observable label, including contact revelations."""
    if difficulty is not None and o.get("contact_reveals"):
        return _normalise(o["contact_reveals"])
    return _normalise(o.get("material", "unknown"))


class RiskEstimator:
    """Estimate one risk level per obstacle and revise it after contact."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.level: Dict[int, str] = {}
        self.source: Dict[int, str] = {}
        self.on_contact: set[int] = set()
        # Cache keys include every prompt input.
        self.verdict_cache: Dict[tuple, str] = {}
        self.calls = 0
        self.perception_calls = 0
        self._perception_started: Optional[float] = None
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

    def assess_many(self, observations: List[dict]) -> Dict[int, str]:
        """Assess newly visible obstacles in one request, within a stage budget."""
        unresolved: Dict[tuple, List[dict]] = {}
        for observation in observations:
            oid = observation["oid"]
            if oid in self.level:
                continue
            cached = self.verdict_cache.get(self._cache_key(observation, None))
            if cached is not None:
                self.level[oid] = cached
                self.source[oid] = "cache"
            else:
                unresolved.setdefault(
                    self._cache_key(observation, None), []).append(observation)

        groups = list(unresolved.items())
        decided = (self._deepseek_many([(k[0], g[0]) for k, g in groups])
                   if self.api_key else {})
        for i, (key, group) in enumerate(groups):
            level = decided.get(i)
            source = "deepseek-batch"
            if level is None:
                level = keyword_level(key[0])
                source = "keyword"
            self.verdict_cache[key] = level
            for observation in group:
                oid = observation["oid"]
                self.level[oid] = level
                self.source[oid] = source
        return {o["oid"]: self.level[o["oid"]] for o in observations}

    def reassess(self, observation: dict, difficulty: float) -> str:
        """Reassess an obstacle once after contact reveals its difficulty."""
        oid = observation["oid"]
        if oid in self.on_contact:
            return self.level[oid]
        level, source = self._decide(observation, difficulty=difficulty)
        self.level[oid] = level
        self.source[oid] = f"{source}-contact"
        self.on_contact.add(oid)
        return level

    def forget(self, oid: int):
        """Drop per-obstacle verdicts while retaining reusable question results."""
        self.level.pop(oid, None)
        self.source.pop(oid, None)
        self.on_contact.discard(oid)

    def level_of(self, oid: int, partners: Sequence[int] = ()) -> Optional[str]:
        """Return the obstacle's verdict, raised by any coupled partners."""
        level = self.level.get(oid)
        for other in partners:
            level = higher(level, self.level.get(other))
        return level

    def forbids(self, level: Optional[str]) -> bool:
        """Is this level one the robot is not allowed to disturb at any price?"""
        bar = _normalise(self.cfg.risk_forbidden_level)
        if level is None or bar not in _ORDER:
            return False
        return _ORDER[level] >= _ORDER[bar]

    def _decide(self, o: dict, difficulty: Optional[float]):
        label = self._label(o, difficulty)
        key = self._cache_key(o, difficulty)
        cached = self.verdict_cache.get(key)
        if cached is not None:
            return cached, "cache"
        if self.api_key:
            level = self._deepseek(o, label, difficulty)
            if level is not None:
                self.verdict_cache[key] = level
                return level, "deepseek"
        level = (keyword_level(label) if difficulty is None
                 else self._weighed(o, label, difficulty))
        self.verdict_cache[key] = level
        return level, "keyword"

    def _cache_key(self, o: dict, difficulty: Optional[float]) -> tuple:
        """Build a cache key from every value that reaches the risk prompt."""
        return (label_of(o, difficulty),
                round(float(o["l"]), 3), round(float(o["d"]), 3),
                round(float(o.get("h", 1.0)), 3),
                None if difficulty is None else round(float(difficulty), 3))

    def _weighed(self, o: dict, label: str, difficulty: float) -> str:
        """Raise the keyword verdict when measured force exceeds expectation."""
        level = keyword_level(label)
        volume = float(o["l"]) * float(o["d"]) * float(o.get("h", 1.0))
        expected = friction_force(material_mu_rho(label), volume)
        if expected > 0.0 and difficulty >= expected * self.cfg.risk_heavy_ratio:
            return step_up(level)
        return level

    @staticmethod
    def _label(o: dict, difficulty: Optional[float]) -> str:
        return label_of(o, difficulty)

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

    def _deepseek_many(self, items) -> Dict[int, str]:
        if not items or self.cfg.perception_llm_max_calls <= 0:
            return {}
        if self._perception_started is None:
            self._perception_started = time.monotonic()
        rows = [
            {"id": str(i), "label": label,
             "size_m": [float(o["l"]), float(o["d"]),
                        float(o.get("h", 1.0))]}
            for i, (label, o) in enumerate(items)
        ]
        prompt = (
            "Rate the consequence of a mobile robot pushing each obstacle. "
            "Risk is about harm to people, contents, or the building, not push "
            "force. Levels: low=ordinary furniture; medium=spill/break/cost; "
            "medium_high=hazardous or damaged structure; high=person involved; "
            "extreme=load-bearing/collapse risk. Return only one JSON object "
            "mapping every id to one of low, medium, medium_high, high, extreme.\n"
            f"Items: {json.dumps(rows, ensure_ascii=False)}"
        )
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
            elapsed = time.monotonic() - self._perception_started
            remaining = self.cfg.perception_llm_timeout - elapsed
            if (self.perception_calls >= self.cfg.perception_llm_max_calls
                    or remaining <= 0.0):
                self.cfg.log("[risk] perception LLM budget exhausted; using keywords")
                return {}
            timeout = min(self.cfg.llm_timeout, remaining)
            try:
                self.calls += 1
                self.perception_calls += 1
                response = requests.post(self.cfg.deepseek_base_url,
                                         headers=headers, json=body,
                                         timeout=timeout)
                data = response.json()
                if response.status_code >= 400:
                    error = data.get("error", data) if isinstance(data, dict) else data
                    self.cfg.log(f"[risk] HTTP {response.status_code}: {error}")
                    if not (response.status_code >= 500
                            or response.status_code in {408, 409, 429}):
                        return {}
                    continue
                content = data["choices"][0]["message"].get("content") or ""
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                parsed = json.loads(match.group(0) if match else content)
                result = {}
                for i, _item in enumerate(items):
                    level = self._parse(str(parsed.get(str(i), "")))
                    if level is not None:
                        result[i] = level
                return result
            except Exception as e:
                self.cfg.log(f"[risk] batch call failed ({attempt}): {e}")
        return {}

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
