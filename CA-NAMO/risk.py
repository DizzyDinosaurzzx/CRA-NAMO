"""触碰障碍物后评估搬运它的次生风险（结构坍塌、掩埋、误伤等）。

难度模型（llm_difficulty.py）只算摩擦力，看不出"这是灾区里一根还在承重的柱子，
挪开可能引发二次坍塌"这类代价——那是次生风险，跟推不推得动是两件事。默认关闭
（见 config.risk_assessment_enabled），只在真正接触障碍物之后才评估一次、缓存
结果；档位 -> 附加代价的换算表由 config.risk_tier_penalty 唯一定义，本模块只
负责问出档位是哪一个。风险代价如何计入 C 见 cost.removal_cost。
"""

from __future__ import annotations
import os
import re
import time
from typing import Dict, Optional

import requests

from config import Config

# 没有 API key 或调用失败时的兜底：context/material 里出现这些词才判有风险，
# 宁可漏判（回退到 "none"，行为等同功能关闭）也不要在没有信号时瞎报高风险。
_HAZARD_KEYWORDS = (
    "load-bearing", "load bearing", "structural", "collapse", "support",
    "unstable", "debris", "rubble", "trapped", "buried",
)


class RiskEstimator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = cfg.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.cache: Dict[int, str] = {}      # oid -> 档位，同一障碍物只问一次
        self.calls = 0

    def penalty(self, tier: str) -> float:
        return self.cfg.risk_tier_penalty.get(tier, 0.0)

    def assess(self, obs_obs: dict) -> str:
        """返回风险档位，必是 cfg.risk_tier_penalty 的键之一。obs_obs 取自
        MovableObstacle.observation()，需含 material/context/l/d/h。"""
        if not self.cfg.risk_assessment_enabled:
            return "none"
        oid = obs_obs["oid"]
        if oid in self.cache:
            return self.cache[oid]

        tier = self._heuristic(obs_obs)
        if self.api_key:
            llm_tier = self._deepseek(obs_obs)
            if llm_tier is not None:
                tier = llm_tier
            else:
                self.cfg.log(f"[risk] oid={oid} LLM 无有效回复，退回启发式档位 {tier!r}")
        self.cache[oid] = tier
        return tier

    # --- 启发式兜底 ---
    def _heuristic(self, o: dict) -> str:
        text = f"{o.get('material', '')} {o.get('context', '')}".lower()
        return "high" if any(k in text for k in _HAZARD_KEYWORDS) else "none"

    # --- LLM 判断 ---
    def _build_prompt(self, o: dict) -> str:
        tiers = list(self.cfg.risk_tier_penalty)     # 档位顺序即此处的权威定义
        context = o.get("context") or "(no extra context given)"
        return (
            "A mobile robot is about to push/drag an obstacle aside to clear its "
            "path. Judge the SECONDARY risk of moving it -- not how hard it is to "
            "push (that is handled separately by a friction model), but whether "
            "moving it could cause harm beyond the push itself: structural "
            "collapse, burying or trapping someone, releasing hazardous contents, "
            "destabilising what it is currently supporting, etc.\n\n"
            f"Object label: '{o.get('material')}'\n"
            f"Context: {context}\n"
            f"Size l x d x h = {float(o['l']):g} x {float(o['d']):g} x "
            f"{float(o.get('h', 1.0)):g} m\n\n"
            f"Answer with exactly one word, the risk tier, from this list ordered "
            f"least to most severe: {', '.join(tiers)}.\n"
            "- none: an ordinary object, nothing else depends on it.\n"
            "- low: minor plausible harm, unlikely to matter in practice.\n"
            "- moderate: could injure lightly or damage something if mishandled.\n"
            "- high: real chance of secondary harm (e.g. destabilising nearby "
            "structure, spilling hazardous material).\n"
            "- critical: moving it can plausibly trigger collapse, burial, or "
            "other severe harm (e.g. a load-bearing column in a damaged "
            "building).\n"
            "Output ONLY the single tier word, nothing else."
        )

    def _deepseek(self, o: dict) -> Optional[str]:
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
        if self.cfg.llm_max_tokens:
            body["max_tokens"] = int(self.cfg.llm_max_tokens)
        valid = set(self.cfg.risk_tier_penalty)
        for attempt in range(self.cfg.llm_max_retries + 1):
            try:
                self.calls += 1
                r = requests.post(self.cfg.deepseek_base_url, headers=headers,
                                  json=body, timeout=self.cfg.llm_timeout)
                data = r.json()
                if r.status_code >= 400:
                    error = data.get("error", data) if isinstance(data, dict) else data
                    self.cfg.log(f"[risk] HTTP {r.status_code}: {error}")
                    retryable = r.status_code >= 500 or r.status_code in {408, 409, 429}
                    if not retryable:
                        return None
                    if attempt < self.cfg.llm_max_retries:
                        time.sleep(2.0)
                    continue
                if "choices" not in data:
                    self.cfg.log(f"[risk] unexpected response: {data}")
                    if attempt < self.cfg.llm_max_retries:
                        time.sleep(2.0)
                    continue
                choice = data["choices"][0]
                text = (choice.get("message", {}).get("content") or "").lower()
                # 取回复里最后一个合法档位词：推理式回复可能先复述选项再给结论
                hits = [w for w in re.findall(r"[a-z]+", text) if w in valid]
                if hits:
                    return hits[-1]
                self.cfg.log(
                    f"[risk] no valid tier in response "
                    f"(finish_reason={choice.get('finish_reason')!r}, text={text!r})")
                if attempt < self.cfg.llm_max_retries:
                    time.sleep(2.0)
            except Exception as e:
                self.cfg.log(f"[risk] call failed ({attempt}): {e}")
                if attempt < self.cfg.llm_max_retries:
                    time.sleep(2.0)
        return None
