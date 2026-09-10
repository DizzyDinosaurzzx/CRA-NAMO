"""让 LLM 在几何层验证过的候选方案中做离散选择（L2 策略）。

与 llm_difficulty / risk 不同，这里的 LLM 不返回任何数值，而是直接给出决策。
提示词只提供**可观测量**——材料标签、尺寸、距离、方位、路线长度——而刻意隐去
目标函数的一切输出（J、W、风险附加项、以牛顿为单位的推动难度）。要测的正是
LLM 的语义先验能否替代显式代价模型，把算好的代价喂给它会让对比退化为读数比大小。
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional, Tuple

import requests

from config import Config

def geometry_bearing(reference: float, course: float) -> float:
    """返回 course 相对 reference 的夹角，落在 [-pi, pi]。"""
    return (course - reference + math.pi) % (2.0 * math.pi) - math.pi


class ActionChooser:
    """在候选方案中选一个；没有可用 LLM 时交还给调用方按代价排序。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.api_key = ((cfg.deepseek_api_key
                         or os.getenv("DEEPSEEK_API_KEY", ""))
                        if cfg.llm_choice else "")
        self.calls = 0
        self.reused = 0
        self.log: List[str] = []
        # 候选集合相同时沿用上次决定，按候选身份而非下标记录。
        self._decided: Dict[frozenset, tuple] = {}
        self.mode = "deepseek" if self.api_key else "heuristic"

    # ---------- 候选描述 ----------

    @staticmethod
    def _route_length(plan) -> float:
        return sum(float(a.get("dist", 0.0))
                   for a in plan.actions if a["type"] == "move")

    @staticmethod
    def _push_distance(plan, oid: int) -> float:
        return sum(float(a.get("dist", 0.0)) for a in plan.actions
                   if a["type"] == "remove" and a["oid"] == oid)

    def describe(self, options: List[dict], belief, robot_xy,
                 goal_xy) -> List[dict]:
        """把候选翻译成只含可观测量的描述。"""
        rx, ry = robot_xy
        goal_course = math.atan2(goal_xy[1] - ry, goal_xy[0] - rx)
        rows = []
        for i, opt in enumerate(options):
            plan = opt["plan"]
            travel = self._route_length(plan)
            if not opt["oids"]:
                detail = (f"drive around everything, moving nothing; "
                          f"{travel:,.1f} m of travel")
            else:
                parts = []
                for oid in opt["oids"]:
                    obs = belief.obstacle(oid)
                    bearing = math.degrees(geometry_bearing(
                        goal_course, math.atan2(obs.y - ry, obs.x - rx)))
                    parts.append(
                        f"obstacle {oid} ('{obs.material}', "
                        f"{obs.l:,.2f} x {obs.d:,.2f} x {obs.h:,.2f} m, "
                        f"{math.hypot(obs.x - rx, obs.y - ry):,.1f} m away, "
                        f"{abs(bearing):,.0f} deg "
                        f"{'left' if bearing >= 0 else 'right'} of the goal "
                        f"direction, pushed {self._push_distance(plan, oid):,.1f} m "
                        f"aside)")
                detail = ("push " + " then ".join(parts)
                          + f"; {travel:,.1f} m of travel")
            rows.append({"index": i, "detail": detail})
        return rows

    # ---------- 决策 ----------

    @staticmethod
    def _identity(option: dict) -> tuple:
        """候选的稳定身份，与它在列表中的位置无关。"""
        return (option["kind"], tuple(sorted(option["oids"])))

    @classmethod
    def _key(cls, options: List[dict]) -> frozenset:
        """候选集合的身份。机器人移动会改变候选顺序，但选择本身没有变，
        因此按集合而非序列缓存，并记住选中的候选而不是它的下标。"""
        return frozenset(cls._identity(o) for o in options)

    def choose(self, options: List[dict], rows: List[dict],
               goal_distance: float) -> Tuple[Optional[int], str]:
        """返回 (选中的候选下标, 来源)；下标为 None 表示交还给代价排序。"""
        if not options:
            return None, "empty"
        if len(options) == 1:
            return 0, "only-option"

        key = self._key(options)
        if self.cfg.llm_choice_reuse_decision and key in self._decided:
            wanted = self._decided[key]
            for i, option in enumerate(options):
                if self._identity(option) == wanted:
                    self.reused += 1
                    return i, "reuse"

        if not self.api_key:
            return None, "heuristic"

        pick = self._deepseek(rows, goal_distance)
        if pick is None or not 0 <= pick < len(options):
            self.cfg.log("[choice] no usable answer; falling back to lowest cost")
            return None, "fallback"
        self._decided[key] = self._identity(options[pick])
        return pick, "deepseek"

    def _build_prompt(self, rows: List[dict], goal_distance: float) -> str:
        listing = "\n".join(f"  {r['index']}. {r['detail']}" for r in rows)
        return (
            "A mobile robot must reach its goal in a cluttered indoor space. It "
            "can drive around obstacles, or push a movable obstacle aside and go "
            "straight through. Pushing costs energy and time, and some objects "
            "should not be disturbed at all because of what moving them does to "
            "people, to their contents, or to the building.\n\n"
            f"The goal is {goal_distance:,.1f} m away in a straight line.\n\n"
            "Each option below is already known to be physically executable. "
            "Choose the one you judge best overall, trading off the travel "
            "involved against what pushing that particular object would mean:\n\n"
            f"{listing}\n\n"
            "Output ONLY the number of the option you choose, nothing else."
        )

    def _deepseek(self, rows: List[dict], goal_distance: float) -> Optional[int]:
        prompt = self._build_prompt(rows, goal_distance)
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        body = {
            "model": self.cfg.deepseek_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "stream": False,
            "thinking": {"type": "enabled" if self.cfg.deepseek_thinking
                         else "disabled"},
            "reasoning_effort": self.cfg.deepseek_reasoning_effort,
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
                    self.cfg.log(f"[choice] HTTP {r.status_code}: {error}")
                    if not (r.status_code >= 500
                            or r.status_code in {408, 409, 429}):
                        return None
                    continue
                text = data["choices"][0]["message"].get("content") or ""
                # 取最后一个整数，推理过程可能先复述其他编号。
                nums = re.findall(r"\d+", text)
                if nums:
                    return int(nums[-1])
                self.cfg.log(f"[choice] no option number in response: {text!r}")
            except Exception as e:
                self.cfg.log(f"[choice] call failed ({attempt}): {e}")
        return None
