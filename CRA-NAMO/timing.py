"""任务耗时模型唯一所在地：T = T_motion + T_decision；模块外不得自行用距离除以速度。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from config import Config

XY = Tuple[float, float]

# 小于此值视为接触规划器的数值噪声（障碍物旋转时抓取点几乎没动），而非真实行驶
_MIN_SEGMENT_M = 1e-9
# 朝向变化小于此值视为同一直线的延续，不截断直线段、不计转向
_MIN_TURN_RAD = 1e-6


def trapezoid_time(distance: float, v_max: float, a_max: float) -> float:
    """从静止到静止走完 distance 的秒数（|v|<=v_max、|a|<=a_max）；代入弧度同样适用于转向。"""
    if distance <= 0.0:
        return 0.0
    # 加速到 v_max 再减速回零所消耗的距离
    ramp = v_max * v_max / a_max
    if distance >= ramp:
        return distance / v_max + v_max / a_max      # 加速-巡航-减速
    return 2.0 * math.sqrt(distance / a_max)         # 三角形：始终达不到 v_max


def _wrap_pi(angle: float) -> float:
    """模 2*pi 的最短带符号转角；勿用 geometry.wrap_dtheta（那是模 pi 的，而朝向半圈并不重合）。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def heading_of(a: XY, b: XY) -> Optional[float]:
    """a 到 b 的行进方向；两点重合时返回 None。"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if math.hypot(dx, dy) <= _MIN_SEGMENT_M:
        return None
    return math.atan2(dy, dx)


# --- 规划侧估计 ---
# 搜索须在事前给一条未必会走的边估价，且不知道到达朝向（按朝向计价就得搜 (节点, 朝向) 而非节点），
# 故按巡航速度、无加减速、无转向估计——结果构成执行器数值的下界，保证 A* 启发式可采纳；
# 上报的任务时间始终取自执行器。
def drive_seconds(cfg: Config, distance: float) -> float:
    """自由空间行驶 distance 米的估计秒数。"""
    return distance / cfg.v_max


def turn_seconds(cfg: Config, dtheta: float) -> float:
    """自由空间原地转 dtheta 弧度的秒数；此估计是精确的（执行器同式计费），单独存在只因搜索到扩张循环才看得到拐角。"""
    return trapezoid_time(abs(dtheta), cfg.w_max, cfg.alpha_max)


def turn_between(h_in: Optional[float], h_out: Optional[float]) -> float:
    """从朝向 h_in 转到 h_out 的转角大小；任一朝向未知则为 0。"""
    if h_in is None or h_out is None:
        return 0.0
    return abs(_wrap_pi(h_out - h_in))


def removal_seconds(cfg: Config, contact_travel: float) -> float:
    """估计搬走一个障碍物所需秒数：机器人全程贴身护送，按其自身接送路程计时。"""
    return contact_travel / cfg.v_max_contact + 2.0 * cfg.grip_time


@dataclass(frozen=True)
class MotionProfile:
    """自由空间与接触状态下的速度、加速度限制。"""
    v_max: float
    a_max: float
    w_max: float
    alpha_max: float
    v_max_contact: float
    a_max_contact: float
    w_max_contact: float
    alpha_max_contact: float
    grip_time: float

    @classmethod
    def from_config(cls, cfg: Config) -> "MotionProfile":
        return cls(v_max=cfg.v_max, a_max=cfg.a_max,
                   w_max=cfg.w_max, alpha_max=cfg.alpha_max,
                   v_max_contact=cfg.v_max_contact, a_max_contact=cfg.a_max_contact,
                   w_max_contact=cfg.w_max_contact,
                   alpha_max_contact=cfg.alpha_max_contact,
                   grip_time=cfg.grip_time)

    def limits(self, in_contact: bool) -> Tuple[float, float, float, float]:
        if in_contact:
            return (self.v_max_contact, self.a_max_contact,
                    self.w_max_contact, self.alpha_max_contact)
        return self.v_max, self.a_max, self.w_max, self.alpha_max


class MotionTimer:
    """随执行器实际行驶逐段累计仿真秒数；共线同模式段合并为一次直线运行，结束才折算时间，flush 前总计值不完整。"""

    def __init__(self, profile: MotionProfile, heading: float = 0.0):
        self.profile = profile
        self.heading = heading
        self.total = 0.0            # 机器人全部仿真运动时间 [s]
        self.contact_total = 0.0    # 其中抓着障碍物的部分 [s]
        # 尚未折算成时间的未闭合直线段
        self._run_dist = 0.0
        self._run_contact = False

    @property
    def elapsed(self) -> float:
        """含未闭合直线段的已耗时秒数；刻意只读：此处 flush 会截断直线段、使同一直线计两次。"""
        if self._run_dist <= 0.0:
            return self.total
        v_max, a_max, _w, _alpha = self.profile.limits(self._run_contact)
        return self.total + trapezoid_time(self._run_dist, v_max, a_max)

    # --- 累计 ---
    def travel(self, distance: float, heading: Optional[float] = None,
               in_contact: bool = False) -> None:
        """沿朝向 heading 行驶 distance 米；heading=None 表示倒退不计转向（差速底盘倒车无需掉头，但倒车前仍须停车截断直线段）。"""
        if distance <= _MIN_SEGMENT_M:
            return
        if heading is None:
            self.flush()
        elif (abs(_wrap_pi(heading - self.heading)) > _MIN_TURN_RAD
                or in_contact != self._run_contact):
            self.flush()
            self.turn_to(heading, in_contact)
        self._run_dist += distance
        self._run_contact = in_contact

    def turn_to(self, heading: float, in_contact: bool = False) -> None:
        """原地转向至指定朝向。"""
        self.flush()
        dtheta = abs(_wrap_pi(heading - self.heading))
        _v, _a, w_max, alpha_max = self.profile.limits(in_contact)
        self._add(trapezoid_time(dtheta, w_max, alpha_max), in_contact)
        self.heading = heading

    def hold(self, seconds: float, in_contact: bool = True) -> None:
        """机器人原地不动但被占用的时间——抓取或松开。"""
        if seconds <= 0.0:
            return
        self.flush()
        self._add(seconds, in_contact)

    def grip(self) -> None:
        """抓上并松开障碍物各一次的耗时。"""
        self.hold(2.0 * self.profile.grip_time)

    def flush(self) -> None:
        """闭合当前直线段并折算成时间。"""
        if self._run_dist <= 0.0:
            self._run_dist = 0.0
            return
        v_max, a_max, _w, _alpha = self.profile.limits(self._run_contact)
        self._add(trapezoid_time(self._run_dist, v_max, a_max), self._run_contact)
        self._run_dist = 0.0

    def _add(self, seconds: float, in_contact: bool) -> None:
        self.total += seconds
        if in_contact:
            self.contact_total += seconds
