"""使用LLM估计障碍物移动难度"""

from __future__ import annotations
import os
import re
import time
from typing import Dict
import requests
from config import Config

# 材质 -> 单位底面积的操作难度系数，估计值 difficulty ~= density * (l * d)。
MATERIAL_DENSITY: Dict[str, float] = {
    # ---- 极轻：几乎没有阻力 ----
    "styrofoam_box": 0.004,
    "foam_mat": 0.05,
    # ---- 轻：单手可推 ----
    "cardboard_box": 0.07,
    "empty_cart": 0.08,
    "plastic_chair": 0.10,
    "trash_bin": 0.15,
    "stool": 0.18,
    "chair": 0.20,
    "empty_shelf": 0.21,
    "cart": 0.30,
    # ---- 中：需要身体发力 ----
    "wooden_table": 0.50,
    "wooden_crate": 0.75,
    "shelf": 0.80,
    "sofa": 0.85,
    "cabinet": 0.90,
    "pallet": 1.00,
    "loaded_pallet": 1.10,
    # ---- 重：勉强推得动 ----
    "filing_cabinet": 2.50,
    "steel_shelf": 4.20,
    "steel_safe": 5.50,
    # ---- 极重：实际上等同于墙 ----
    "concrete_block": 25.0,
    "industrial_machine": 37.5,
    # ---- 兜底 ----
    "unknown": 0.50,
}
# 同义词表格
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
# 注入 LLM prompt 的标定锚点：跨越整个量级、彼此区分度高的少数几项。
# 只用来对齐量纲——被查询的材质会在构造 prompt 时从中剔除，避免直接泄漏答案。
PROMPT_ANCHORS = (
    "styrofoam_box",
    "cardboard_box",
    "chair",
    "wooden_table",
    "loaded_pallet",
    "filing_cabinet",
)

_NON_WORD = re.compile(r"[^a-z0-9]+")

def _normalise(name) -> str:
    """'Wooden Crate' / 'wooden-crate' -> 'wooden_crate'。"""
    return _NON_WORD.sub("_", str(name).strip().lower()).strip("_")

def material_density(name) -> float:
    """材质名 -> 难度系数：精确匹配 -> 别名 -> 词元匹配 -> unknown。"""
    key = _normalise(name)
    if key in MATERIAL_DENSITY:
        return MATERIAL_DENSITY[key]
    if key in MATERIAL_ALIASES:
        return MATERIAL_DENSITY[MATERIAL_ALIASES[key]]

    # 组合词回退："reinforced_steel_safe" -> 命中 steel_safe。
    # 取命中项里最重的一档：多出来的修饰词通常意味着更难推（loaded/steel/...）。
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

    # -------------公共接口-------------- #
    def estimate(self, obs_obs: dict) -> float:
        """Return an estimated difficulty for a *perceived* obstacle observation."""
        oid = obs_obs["oid"]
        if oid in self.cache:
            return self.cache[oid]
        if self.api_key:
            val = self._deepseek(obs_obs)
            if val is None:                       # 失败回退
                val = self._heuristic(obs_obs)
        else:
            val = self._heuristic(obs_obs)
        val = max(0.01, round(float(val), 3))
        self.cache[oid] = val
        return val

    # -------------启发式方法--------------- #
    def _heuristic(self, o: dict) -> float:
        """difficulty ~= 材质系数 x 底面积。只依赖感知得到的材质标签与几何尺寸。"""
        density = material_density(o.get("material", "unknown"))
        area = o.get("area")
        if not area:
            area = o["l"] * o["d"]
        return density * float(area)

    # -------------LLM估计--------------- #
    def _build_prompt(self, o: dict) -> str:
        """构造询问 prompt。

        只给少量跨量级的标定锚点，用来对齐量纲；**被查询材质本身及与它共享词元的
        材质会被剔除**，否则 LLM 只需查表相乘就能复现启发式，等于白问。
        """
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
                    # 参数、鉴权等 4xx 错误重复请求不会自行恢复；限流和超时除外。
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
            except Exception as e:            # noqa: BLE001 - 任意失败都回退
                self.cfg.log(f"[LLM] call failed ({attempt}): {e}")
                if attempt < self.cfg.llm_max_retries:
                    time.sleep(2.0)
        return None
