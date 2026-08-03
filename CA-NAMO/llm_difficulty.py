from __future__ import annotations
import os
import re
import time
from typing import Dict
import requests
from config import Config

# Estimate: difficulty ~= density * volume (l * d * h).
MATERIAL_DENSITY: Dict[str, float] = {
    # ---- very light ---- #
    "styrofoam_box": 0.004,
    "foam_mat": 0.05,
    # ---- light ---- #
    "cardboard_box": 0.07,
    "empty_cart": 0.08,
    "plastic_chair": 0.10,
    "trash_bin": 0.15,
    "stool": 0.18,
    "chair": 0.20,
    "empty_shelf": 0.21,
    "cart": 0.30,
    # ---- medium ---- #
    "wooden_table": 0.50,
    "wooden_crate": 0.75,
    "shelf": 0.80,
    "sofa": 0.85,
    "cabinet": 0.90,
    "pallet": 1.00,
    "loaded_pallet": 1.10,
    # ---- heavy ---- #
    "filing_cabinet": 2.50,
    "steel_shelf": 4.20,
    "steel_safe": 5.50,
    # ---- very heavy ---- #
    "concrete_block": 25.0,
    "industrial_machine": 37.5,
    # ---- fallback ---- #
    "unknown": 1,
}
# Typical height per material, used as the ground-truth h in the scenarios.
MATERIAL_HEIGHT: Dict[str, float] = {
    "styrofoam_box": 1.0,
    "foam_mat": 0.1,
    "cardboard_box": 0.8,
    "empty_cart": 1.0,
    "plastic_chair": 0.9,
    "trash_bin": 1.0,
    "stool": 0.5,
    "chair": 0.9,
    "empty_shelf": 1.8,
    "cart": 1.1,
    "wooden_table": 0.75,
    "wooden_crate": 1.0,
    "shelf": 1.8,
    "sofa": 0.85,
    "cabinet": 1.8,
    "pallet": 0.15,
    "loaded_pallet": 1.2,
    "filing_cabinet": 1.3,
    "steel_shelf": 2.0,
    "steel_safe": 1.5,
    "concrete_block": 1.0,
    "industrial_machine": 1.6,
    "unknown": 1.0,
}
# Synonym table
MATERIAL_ALIASES: Dict[str, str] = {
    "box": "cardboard_box",
    "carton": "cardboard_box",
    "crate": "wooden_crate",
    "table": "wooden_table",
    "desk": "wooden_table",
    "bin": "trash_bin",
    "pallet_loaded": "loaded_pallet",
    "safe": "steel_safe",
    "machine": "industrial_machine",
    "concrete": "concrete_block",
    "foam": "foam_mat",
    "styrofoam": "styrofoam_box",
}
PROMPT_ANCHORS = tuple(
    name for name in sorted(MATERIAL_DENSITY, key=lambda k: MATERIAL_DENSITY[k])
    if name != "unknown"
)

_NON_WORD = re.compile(r"[^a-z0-9]+")

def _normalise(name) -> str:  # normalise material name
    return _NON_WORD.sub("_", str(name).strip().lower()).strip("_")


def _canonical_anchor(name) -> str | None:
    key = _normalise(name)
    anchor_names = {anchor for anchor in PROMPT_ANCHORS if anchor != "unknown"}
    if key in anchor_names:
        return key
    alias_target = MATERIAL_ALIASES.get(key)
    return alias_target if alias_target in anchor_names else None

def _volume(o: dict) -> float:
    volume = o.get("volume")
    if not volume:
        volume = float(o["l"]) * float(o["d"]) * float(o.get("h", 1.0))
    return float(volume)

def _lookup(table: Dict[str, float], name) -> float:
    key = _normalise(name)
    if key in table:
        return table[key]
    if key in MATERIAL_ALIASES:
        return table[MATERIAL_ALIASES[key]]
    tokens = set(key.split("_"))
    hits = [v for k, v in table.items()
            if k != "unknown" and tokens & set(k.split("_"))]
    hits += [table[canon] for alias, canon in MATERIAL_ALIASES.items()
             if tokens & set(alias.split("_"))]
    return max(hits) if hits else table["unknown"]

def material_density(name) -> float:
    return _lookup(MATERIAL_DENSITY, name)

def material_height(name) -> float:
    return _lookup(MATERIAL_HEIGHT, name)

class DifficultyEstimator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.cache: Dict[int, float] = {}
        self.density_cache: Dict[int, float] = {}
        self.material_density_cache: Dict[str, float] = {}
        self.material_source_cache: Dict[str, str] = {}
        self.source_cache: Dict[int, str] = {}
        self.object_cache_hits = 0
        self.material_cache_hits = 0
        self.calls = 0
        self.mode = "deepseek" if self.api_key else "heuristic"

    # ------------- Public interface -------------- #
    def estimate(self, obs_obs: dict) -> float:  # estimate difficulty from perceived material
        oid = obs_obs["oid"]
        if oid in self.cache:
            self.object_cache_hits += 1
            return self.cache[oid]

        material = _normalise(obs_obs.get("material", "unknown"))
        canonical = _canonical_anchor(material)
        if canonical is not None:
            density = MATERIAL_DENSITY[canonical]
            source = "anchor"
        elif material in self.material_density_cache:
            density = self.material_density_cache[material]
            source = f"{self.material_source_cache[material]}-cache"
            self.material_cache_hits += 1
        elif self.api_key:
            density = self._deepseek(obs_obs)
            if density is None:
                density = material_density(material)
                source = "heuristic-fallback"
            else:
                source = "deepseek"
            self.material_density_cache[material] = density
            self.material_source_cache[material] = source
        else:
            density = material_density(material)
            source = "heuristic"
            self.material_density_cache[material] = density
            self.material_source_cache[material] = source

        density = max(0.0, float(density))
        difficulty = max(0.01, round(density * _volume(obs_obs), 3))
        self.density_cache[oid] = round(density, 6)
        self.source_cache[oid] = source
        self.cache[oid] = difficulty
        return difficulty

    # ------------- Heuristic method --------------- #
    def _heuristic(self, o: dict) -> float:
        density = material_density(o.get("material", "unknown"))
        return density * _volume(o)

    # ------------- LLM density estimation --------------- #
    def _build_prompt(self, o: dict) -> str:
        material = _normalise(o.get("material", "unknown"))
        canonical = _canonical_anchor(material)

        anchors = "\n".join(
            f"  {name:<18s} {MATERIAL_DENSITY[name]:g}"
            for name in sorted(PROMPT_ANCHORS, key=lambda k: MATERIAL_DENSITY[k])
            if name != "unknown"
        )

        if canonical is not None:
            density = MATERIAL_DENSITY[canonical]
            material_instruction = (
                f"The input label '{material}' is an exact calibrated match for "
                f"'{canonical}'. You MUST use density={density:g}; do not replace "
                "it with real-world intuition or another reference value.\n"
                f"Return exactly {density:g}."
            )
        else:
            material_instruction = (
                f"The input label '{material}' is NOT in PROMPT_ANCHORS. Infer its "
                "project-scale density by combining the PROMPT_ANCHORS table with "
                "ordinary real-world knowledge about this object.\n"
                "Internally follow these rules:\n"
                "1. Infer the likely object category, construction, filled/empty "
                "state, mass, floor contact, wheels, and pushing friction from the "
                "label. Brand or product names must be interpreted as their actual "
                "real-world object type.\n"
                "2. Select 2 to 4 anchors with the closest expected pushing "
                "resistance. Use their numeric values to bracket and interpolate. "
                "Extrapolate only when the object is clearly outside that bracket.\n"
                "3. PROMPT_ANCHORS define the numeric scale and take priority over "
                "raw physical units. Do not copy a convenient midpoint or generic "
                "fallback. In particular, return 0.5 only when the object is "
                "independently judged comparable to wooden_table.\n"
                "4. Return ONE size-independent density coefficient. Object volume "
                "must not affect it; Python will multiply density by volume later."
            )

        return (
            "Estimate a project-specific pushing-resistance density coefficient for "
            "a mobile robot moving an obstacle aside.\n"
            "Use this project-specific calibrated scale, even when it differs from "
            "ordinary physical density. The caller will calculate:\n\n"
            "    difficulty = density * volume        # volume = l * d * h\n\n"
            "`density` is a per-material coefficient (difficulty per cubic metre). "
            "The following PROMPT_ANCHORS values are DENSITIES, not final "
            "difficulties:\n\n"
            f"{anchors}\n\n"
            f"{material_instruction}\n\n"
            f"Obstacle label: '{o.get('material')}', measured size "
            f"l x d x h = {float(o['l']):g} x {float(o['d']):g} x "
            f"{float(o.get('h', 1.0)):g} m (volume = {_volume(o):g} m^3). "
            "The size tells you what kind of object this is; the density you "
            "return must still be size-independent.\n"
            "Output ONLY the density number, no words or units."
        )

    def _deepseek(self, o: dict):
        prompt = self._build_prompt(o)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = {
            "model": self.cfg.deepseek_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": False,
            "thinking": {"type": "disabled"},
        }
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                self.calls += 1
                r = requests.post(self.cfg.deepseek_base_url, headers=headers,
                                  json=body, timeout=self.cfg.llm_timeout)
                data = r.json()
                if r.status_code >= 400:
                    error = data.get("error", data) if isinstance(data, dict) else data
                    self.cfg.log(f"[LLM] HTTP {r.status_code}: {error}")
                    retryable = r.status_code >= 500 or r.status_code in {408, 409, 429}
                    if not retryable:
                        return None
                    if attempt < self.cfg.llm_max_retries:
                        time.sleep(2.0)
                    continue
                if "choices" not in data:
                    self.cfg.log(f"[LLM] unexpected response: {data}")
                    if attempt < self.cfg.llm_max_retries:
                        time.sleep(2.0)
                    continue
                choice = data["choices"][0]
                text = choice.get("message", {}).get("content") or ""
                m = re.search(
                    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                    text,
                )
                if m:
                    density = float(m.group())
                    if density >= 0:
                        return density
                self.cfg.log(
                    "[LLM] no valid density in response "
                    f"(finish_reason={choice.get('finish_reason')!r}, text={text!r})")
                if attempt < self.cfg.llm_max_retries:
                    time.sleep(2.0)
            except Exception as e:
                self.cfg.log(f"[LLM] call failed ({attempt}): {e}")
                if attempt < self.cfg.llm_max_retries:
                    time.sleep(2.0)
        return None
