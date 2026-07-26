"""
使用 LLM（DeepSeek）根据物体标签估计移动难度 eta = mu * m，并提供离线回退方案。
* 如果有可用的 DeepSeek API 密钥（Config.deepseek_api_key 或 DEEPSEEK_API_KEY 环境变量），则使用 DeepSeek-V4-Flash 进行估计。
* 否则使用确定性的“物体标签 -> eta”映射。
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict

import requests

from config import Config

"""
离线回退方案直接按物体标签给出 eta = mu * m 的粗略估计。
这些数值会进入 W = eta * g * d，因此与 LLM 提示词使用同一尺度。
"""

MATERIAL_ETA: Dict[str, float] = {
    "empty_cart": 0.3,
    "cardboard_box": 0.2,
    "plastic_chair": 0.4,
    "chair": 0.4,
    "trash_bin": 0.5,
    "stool": 0.3,
    "cart": 1.0,
    "wooden_table": 1.2,
    "table": 1.2,
    "shelf": 5.0,
    "cabinet": 6.0,
    "sofa": 10.0,
    "loaded_pallet": 20.0,
    "pallet_loaded": 20.0,
    "pallet": 8.0,
    "wooden_crate": 2.0,
    "crate": 2.0,
    "steel_safe": 200.0,
    "unknown": 1.0,
}


class DifficultyEstimator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.deepseek_api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.cache: Dict[int, float] = {}
        self.calls = 0
        self.mode = "deepseek" if self.api_key else "heuristic"

    # ------------------------------------------------------------------ 公共接口
    def estimate(self, oid: int, label: str) -> float:
        """仅根据物体标签返回 eta = mu * m 的估计值。"""
        if oid in self.cache:
            return self.cache[oid]
        if self.api_key:
            val = self._deepseek(label)
            if val is None:                       # 失败回退
                val = self._heuristic(label)
        else:
            val = self._heuristic(label)
        val = max(0.01, round(float(val), 3))
        self.cache[oid] = val
        return val

    # -------------------------------------------------------------- 启发式方法
    def _heuristic(self, label: str) -> float:
        return MATERIAL_ETA.get(str(label).lower(), MATERIAL_ETA["unknown"])

    # --------------------------------------------------------------- DeepSeek
    def _deepseek(self, label: str):
        prompt = (
            "Estimate the manipulation difficulty eta = mu * m for the object label "
            "below, where mu is the friction coefficient and m is the object mass. "
            "Use only the supplied label. Return a SINGLE positive number. "
            "Reference scale: empty_cart ~0.3, cardboard_box ~0.2, chair ~0.4, "
            "wooden_table ~1.2, wooden_crate ~2, loaded_pallet ~20, "
            "steel_safe ~200.\n\n"
            f"Object label: '{label}'.\n"
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
