from __future__ import annotations
import os
import re
import time
from typing import Dict
import requests
from config import Config

# material -> difficulty coefficient per unit footprint area; estimate difficulty ~= density * (l * d).
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
    "unknown": 0.50,
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
# Calibration anchors provided to the LLM
PROMPT_ANCHORS = (
    "styrofoam_box",
    "cardboard_box",
    "chair",
    "wooden_table",
    "loaded_pallet",
    "filing_cabinet",
)

_NON_WORD = re.compile(r"[^a-z0-9]+")

def _normalise(name) -> str:  # normalise material name
    return _NON_WORD.sub("_", str(name).strip().lower()).strip("_")

def material_density(name) -> float: 
    key = _normalise(name)
    if key in MATERIAL_DENSITY:
        return MATERIAL_DENSITY[key]
    if key in MATERIAL_ALIASES:
        return MATERIAL_DENSITY[MATERIAL_ALIASES[key]]
    tokens = set(key.split("_"))
    hits = [v for k, v in MATERIAL_DENSITY.items()
            if k != "unknown" and tokens & set(k.split("_"))]
    hits += [MATERIAL_DENSITY[canon] for alias, canon in MATERIAL_ALIASES.items()
             if tokens & set(alias.split("_"))]
    return max(hits) if hits else MATERIAL_DENSITY["unknown"]

class DifficultyEstimator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.cache: Dict[int, float] = {}
        self.calls = 0
        self.mode = "deepseek" if self.api_key else "heuristic"

    # ------------- Public interface -------------- #
    def estimate(self, obs_obs: dict) -> float:  # estimate from perceived material
        oid = obs_obs["oid"]
        if oid in self.cache:
            return self.cache[oid]
        if self.api_key:
            val = self._deepseek(obs_obs)
            if val is None:                       # fallback on failure
                val = self._heuristic(obs_obs)
        else:
            val = self._heuristic(obs_obs)
        val = max(0.01, round(float(val), 3))
        self.cache[oid] = val
        return val

    # ------------- Heuristic method --------------- #
    def _heuristic(self, o: dict) -> float:
        density = material_density(o.get("material", "unknown"))
        area = o.get("area")
        if not area:
            area = o["l"] * o["d"]
        return density * float(area)

    # ------------- LLM estimation --------------- #
    def _build_prompt(self, o: dict) -> str:
        asked = set(_normalise(o.get("material", "")).split("_"))
        anchors = "\n".join(
            f"  {name:<18s} {MATERIAL_DENSITY[name]:g}"
            for name in sorted(PROMPT_ANCHORS, key=lambda k: MATERIAL_DENSITY[k])
            if not (asked & set(name.split("_")))
        )
        return (
            "You estimate how hard a mobile robot must work to push an obstacle "
            "aside.\n"
            "The manipulation difficulty coefficient is the work required per unit "
            "of push distance. Model it as:\n\n"
            "    difficulty = density * footprint_area\n\n"
            "`density` is a per-material coefficient (difficulty per square metre). "
            "Reference points on the intended scale:\n\n"
            f"{anchors}\n\n"
            "The material below is NOT in that list. Judge its density yourself "
            "from what the material implies about mass and friction, staying "
            "consistent with the scale above; heavy industrial items may go well "
            "beyond the largest reference point. Then multiply by the footprint "
            "area and return the RESULTING DIFFICULTY, not the density.\n\n"
            f"Obstacle: material='{o.get('material')}', "
            f"footprint={o['l']}x{o['d']} (area={o.get('area')}).\n"
            "Output ONLY the number, no words or units."
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
            "max_tokens": 12,
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
                text = data["choices"][0]["message"]["content"]
                m = re.search(r"[-+]?\d*\.?\d+", text)
                if m:
                    return float(m.group())
            except Exception as e:           
                self.cfg.log(f"[LLM] call failed ({attempt}): {e}")
                if attempt < self.cfg.llm_max_retries:
                    time.sleep(2.0)
        return None
