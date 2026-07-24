"""
使用 LLM（DeepSeek）估计操作难度，并提供离线启发式回退方案。
* 如果有可用的 DeepSeek API 密钥（Config.deepseek_api_key 或 DEEPSEEK_API_KEY 环境变量），则使用 DeepSeek-V3 进行估计。
* 否则使用确定性的启发式方法（材质密度代理值 x 面积）
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict

import requests

from config import Config

"""
离线回退方案中“单位面积质量/摩擦力”的粗略代理值。数值仅用于相对比较；
由于估计值只用于排序，具体尺度并不重要。
"""

MATERIAL_DENSITY: Dict[str, float] = {
    "empty_cart": 0.08,
    "cardboard_box": 0.12,
    "plastic_chair": 0.15,
    "chair": 0.20,
    "trash_bin": 0.20,
    "stool": 0.18,
    "cart": 0.35,
    "wooden_table": 0.50,
    "table": 0.50,
    "shelf": 0.70,
    "cabinet": 0.90,
    "sofa": 0.85,
    "pallet_loaded": 1.10,
    "pallet": 1.00,
    "crate": 0.75,
    "unknown": 0.50,
}


class DifficultyEstimator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.cache: Dict[int, float] = {}
        self.calls = 0
        self.mode = "deepseek" if self.api_key else "heuristic"

    # ------------------------------------------------------------------ 公共接口
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

    # -------------------------------------------------------------- 启发式方法
    def _heuristic(self, o: dict) -> float:
        density = MATERIAL_DENSITY.get(str(o.get("material", "unknown")).lower(),
                                       MATERIAL_DENSITY["unknown"])
        return density * float(o.get("area", o["l"] * o["d"]))

    # --------------------------------------------------------------- DeepSeek
    def _deepseek(self, o: dict):
        prompt = (
            "You estimate how hard a mobile robot must work to push an obstacle aside.\n"
            "Return a SINGLE positive number: the manipulation difficulty coefficient, "
            "i.e. the work required per unit of push distance (higher = heavier / "
            "harder to move). Typical scale: an empty cart ~0.1, a wooden chair ~0.5, "
            "a table ~3, a loaded pallet or heavy sofa ~10.\n\n"
            f"Obstacle: material='{o.get('material')}', "
            f"footprint={o['l']}x{o['d']} (area={o.get('area')}).\n"
            "Output ONLY the number, no words or units."
        )
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
        }
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                self.calls += 1
                r = requests.post(self.cfg.deepseek_base_url, headers=headers,
                                  json=body, timeout=self.cfg.llm_timeout)
                data = r.json()
                if "choices" not in data:
                    self.cfg.log(f"[LLM] unexpected response: {data}")
                    time.sleep(2.0)
                    continue
                text = data["choices"][0]["message"]["content"]
                m = re.search(r"[-+]?\d*\.?\d+", text)
                if m:
                    return float(m.group())
            except Exception as e:            # noqa: BLE001 - 任意失败都回退
                self.cfg.log(f"[LLM] call failed ({attempt}): {e}")
                time.sleep(2.0)
        return None
