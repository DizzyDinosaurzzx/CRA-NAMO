from __future__ import annotations
import os
import re
import time
from typing import Dict
import requests
from config import Config

# Difficulty is the sliding friction force that resists pushing the obstacle:
#
#     difficulty = f = mu * m * g = (mu * rho) * V * g        [N]
#
# with V = l * d * h the bounding-box volume. Because V is the bounding box and
# not the solid volume, rho is a *bulk* density (total mass / bounding volume),
# so a mostly-empty shelf is much lighter per cubic metre than solid concrete.
# Wheeled objects use an effective rolling-resistance coefficient for mu, which
# is why a cart is far easier to move than its mass alone suggests.
G = 9.81                       # gravitational acceleration [m/s^2]

# Sliding friction coefficient against the floor (rolling resistance if wheeled).
MATERIAL_MU: Dict[str, float] = {
    "styrofoam_box": 0.35,
    "foam_mat": 0.50,
    "cardboard_box": 0.35,
    "empty_cart": 0.02,        # wheels
    "plastic_chair": 0.40,
    "trash_bin": 0.40,
    "stool": 0.40,
    "chair": 0.45,
    "empty_shelf": 0.45,
    "cart": 0.03,              # wheels, loaded
    "wooden_table": 0.40,
    "wooden_crate": 0.45,
    "shelf": 0.45,
    "sofa": 0.50,
    "cabinet": 0.45,
    "pallet": 0.40,
    "loaded_pallet": 0.40,
    "filing_cabinet": 0.45,
    "steel_shelf": 0.50,
    "steel_safe": 0.50,
    "concrete_block": 0.60,
    "industrial_machine": 0.50,
    "unknown": 0.40,
}
# Bulk density [kg/m^3] = total mass / bounding-box volume (l * d * h).
MATERIAL_RHO: Dict[str, float] = {
    # ---- very light ---- #
    "styrofoam_box": 15.0,     # EPS foam
    "plastic_chair": 22.0,
    "wooden_table": 26.0,
    "foam_mat": 30.0,
    "chair": 31.0,
    # ---- light ---- #
    "empty_shelf": 35.0,
    "cardboard_box": 40.0,
    "trash_bin": 42.0,
    "empty_cart": 50.0,
    "stool": 50.0,
    "sofa": 52.0,
    "wooden_crate": 60.0,
    # ---- medium ---- #
    "cabinet": 100.0,
    "cart": 150.0,             # loaded
    "shelf": 167.0,            # loaded
    "pallet": 174.0,
    # ---- heavy ---- #
    "steel_shelf": 300.0,      # loaded
    "filing_cabinet": 308.0,
    "loaded_pallet": 434.0,
    "industrial_machine": 700.0,
    "steel_safe": 800.0,
    # ---- very heavy ---- #
    "concrete_block": 2400.0,  # solid concrete
    # ---- fallback ---- #
    "unknown": 100.0,
}
# mu * rho [kg/m^3] -- the per-material coefficient the estimator works with.
MATERIAL_MU_RHO: Dict[str, float] = {
    name: round(MATERIAL_MU[name] * MATERIAL_RHO[name], 4)
    for name in MATERIAL_RHO
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
    name for name in sorted(MATERIAL_MU_RHO, key=lambda k: MATERIAL_MU_RHO[k])
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

def material_mu(name) -> float:
    return _lookup(MATERIAL_MU, name)

def material_rho(name) -> float:
    return _lookup(MATERIAL_RHO, name)

def material_mu_rho(name) -> float:
    return _lookup(MATERIAL_MU_RHO, name)

def material_height(name) -> float:
    return _lookup(MATERIAL_HEIGHT, name)

def friction_force(mu_rho: float, volume: float) -> float:
    """f = mu * rho * V * g  [N] -- the push resistance charged per metre."""
    return mu_rho * volume * G

class DifficultyEstimator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.cache: Dict[int, float] = {}
        self.mu_rho_cache: Dict[int, float] = {}
        self.material_mu_rho_cache: Dict[str, float] = {}
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
            mu_rho = MATERIAL_MU_RHO[canonical]
            source = "anchor"
        elif material in self.material_mu_rho_cache:
            mu_rho = self.material_mu_rho_cache[material]
            source = f"{self.material_source_cache[material]}-cache"
            self.material_cache_hits += 1
        elif self.api_key:
            mu_rho = self._deepseek(obs_obs)
            if mu_rho is None:
                mu_rho = material_mu_rho(material)
                source = "heuristic-fallback"
            else:
                source = "deepseek"
            self.material_mu_rho_cache[material] = mu_rho
            self.material_source_cache[material] = source
        else:
            mu_rho = material_mu_rho(material)
            source = "heuristic"
            self.material_mu_rho_cache[material] = mu_rho
            self.material_source_cache[material] = source

        mu_rho = max(0.0, float(mu_rho))
        difficulty = max(0.01, round(friction_force(mu_rho, _volume(obs_obs)), 3))
        self.mu_rho_cache[oid] = round(mu_rho, 6)
        self.source_cache[oid] = source
        self.cache[oid] = difficulty
        return difficulty

    # ------------- Heuristic method --------------- #
    def _heuristic(self, o: dict) -> float:
        mu_rho = material_mu_rho(o.get("material", "unknown"))
        return friction_force(mu_rho, _volume(o))

    # ------------- LLM mu*rho estimation --------------- #
    def _build_prompt(self, o: dict) -> str:
        material = _normalise(o.get("material", "unknown"))
        canonical = _canonical_anchor(material)

        anchors = "\n".join(
            f"  {name:<18s} mu={MATERIAL_MU[name]:<5g} rho={MATERIAL_RHO[name]:<7g} "
            f"mu*rho={MATERIAL_MU_RHO[name]:g}"
            for name in sorted(PROMPT_ANCHORS, key=lambda k: MATERIAL_MU_RHO[k])
            if name != "unknown"
        )

        if canonical is not None:
            mu_rho = MATERIAL_MU_RHO[canonical]
            material_instruction = (
                f"The input label '{material}' is an exact calibrated match for "
                f"'{canonical}'. You MUST use mu*rho={mu_rho:g}; do not replace "
                "it with real-world intuition or another reference value.\n"
                f"Return exactly {mu_rho:g}."
            )
        else:
            material_instruction = (
                f"The input label '{material}' is NOT in PROMPT_ANCHORS. Infer its "
                "mu and rho by combining the PROMPT_ANCHORS table with ordinary "
                "real-world knowledge about this object.\n"
                "Internally follow these rules:\n"
                "1. Infer the likely object category, construction, filled/empty "
                "state, total mass, floor contact and wheels from the label. Brand "
                "or product names must be interpreted as their actual real-world "
                "object type.\n"
                "2. Estimate rho as total mass divided by the BOUNDING-BOX volume, "
                "not the density of the raw material: a shelf is mostly air, so its "
                "bulk rho is far below the density of steel or wood.\n"
                "3. Estimate mu as the friction coefficient against a hard floor. "
                "If the object rolls on wheels or castors, use an effective rolling "
                "resistance coefficient (~0.02-0.05) instead.\n"
                "4. Select 2 to 4 anchors with the closest expected mu*rho and use "
                "their numbers to bracket and interpolate. Extrapolate only when "
                "the object is clearly outside that bracket.\n"
                "5. Return ONE size-independent number: the product mu*rho in "
                "kg/m^3. Object volume must not affect it; Python multiplies by "
                "volume and g later."
            )

        return (
            "Estimate the pushing-resistance coefficient mu*rho for a mobile robot "
            "sliding an obstacle aside. The push resistance is the sliding friction "
            "force, so the caller will calculate:\n\n"
            "    difficulty = mu * m * g = (mu * rho) * volume * g   # [N]\n"
            "    volume = l * d * h (bounding box), g = 9.81 m/s^2\n\n"
            "`mu` is the floor friction coefficient (dimensionless) and `rho` is the "
            "BULK density in kg/m^3, i.e. total mass divided by the bounding-box "
            "volume. The following PROMPT_ANCHORS list mu, rho and their product; "
            "the value you return is the PRODUCT mu*rho in kg/m^3, not a final "
            "difficulty in newtons:\n\n"
            f"{anchors}\n\n"
            f"{material_instruction}\n\n"
            f"Obstacle label: '{o.get('material')}', measured size "
            f"l x d x h = {float(o['l']):g} x {float(o['d']):g} x "
            f"{float(o.get('h', 1.0)):g} m (volume = {_volume(o):g} m^3). "
            "The size tells you what kind of object this is; the mu*rho you "
            "return must still be size-independent.\n"
            "Output ONLY the mu*rho number, no words or units."
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
            "temperature": 0.0,
            "stream": False,
            "thinking": {"type": "enabled" if self.cfg.deepseek_thinking
                         else "disabled"},
        }
        # Omitted entirely when unset: a cap below the reasoning length does not
        # truncate the answer, it returns an empty content with
        # finish_reason='length', which reads here as "no number" and silently
        # degrades to the heuristic.
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
                # Last number, not the first: a reasoning reply may restate mu
                # and rho before committing to their product, and it is the
                # final figure that answers the question.
                nums = re.findall(
                    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
                    text,
                )
                if nums:
                    mu_rho = float(nums[-1])
                    if mu_rho >= 0:
                        return mu_rho
                if choice.get("finish_reason") == "length":
                    self.cfg.log(
                        "[LLM] reply hit the token cap before answering; raise "
                        f"Config.llm_max_tokens (currently {self.cfg.llm_max_tokens!r})")
                else:
                    self.cfg.log(
                        "[LLM] no valid mu*rho in response "
                        f"(finish_reason={choice.get('finish_reason')!r}, text={text!r})")
                if attempt < self.cfg.llm_max_retries:
                    time.sleep(2.0)
            except Exception as e:
                self.cfg.log(f"[LLM] call failed ({attempt}): {e}")
                if attempt < self.cfg.llm_max_retries:
                    time.sleep(2.0)
        return None
