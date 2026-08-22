"""障碍物自主运动策略：地图自己会变的口子，与机器人推拉障碍物(manip_/move_)完全无关。

脚本化时刻表 / 事件触发 / 反应式三种触发方式，全部收敛成同一个 step 接口：给定经过的
时长与当前世界快照，返回该障碍物这一 tick 的新位姿，或 None 表示不动。三种方式不是
互相独立的三套代码——event() 内部就是复用 scripted()，触发后把"相对触发时刻的时间表"
交给它插值；reactive() 是唯一真正需要每 tick 自己做决策(而非查表)的一种。

默认不给任何 MovableObstacle 挂 drift，行为与不存在本模块时完全一致——这是硬约束，
调用方见 executor._tick。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import geometry

Pose = Tuple[float, float, float]     # x, y, theta
XY = Tuple[float, float]
Condition = Callable[[object, "DriftState"], bool]


@dataclass
class DriftState:
    """驱动一次 tick 所需的世界只读快照。"""
    t: float                          # 当前仿真时刻 [s]，等于 executor 里 self.timer.elapsed
    robot_xy: Tuple[float, float]
    obstacles: Dict[object, object]   # oid -> MovableObstacle，供反应式策略读取周边状态


@dataclass
class DriftPolicy:
    """自主运动策略的统一外壳；step(obs, dt, state) -> 新位姿或 None(本 tick 不动)。"""
    step: Callable[[object, float, DriftState], Optional[Pose]]


def scripted(keyframes: Sequence[Tuple[float, Pose]], loop: bool = True) -> DriftPolicy:
    """按 (仿真时刻, 位姿) 关键帧分段线性插值；loop=True 时超出末尾按周期折返重放整张时刻表。
    朝向插值走 geometry.wrap_dtheta 的最短转角，而非对存储角直接线性插值——矩形朝向
    仅模 pi 有意义，直接插值在跨越回绕点时会绕远路，与 contact.py 里同一类处理同源。
    """
    if len(keyframes) < 2:
        raise ValueError("scripted drift needs at least two keyframes")
    times = [t for t, _ in keyframes]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("scripted drift keyframes must be strictly increasing in time")
    span = times[-1] - times[0]

    def _pose_at(t: float) -> Pose:
        if loop and span > 0.0:
            t = times[0] + (t - times[0]) % span
        if t <= times[0]:
            return keyframes[0][1]
        if t >= times[-1]:
            return keyframes[-1][1]
        i = 0
        while times[i + 1] < t:
            i += 1
        t0, p0 = keyframes[i]
        t1, p1 = keyframes[i + 1]
        frac = (t - t0) / (t1 - t0)
        x = p0[0] + (p1[0] - p0[0]) * frac
        y = p0[1] + (p1[1] - p0[1]) * frac
        theta = p0[2] + geometry.wrap_dtheta(p0[2], p1[2]) * frac
        return (x, y, theta)

    def _step(obs, dt: float, state: DriftState) -> Optional[Pose]:
        pose = _pose_at(state.t)
        if (abs(pose[0] - obs.x) < 1e-9 and abs(pose[1] - obs.y) < 1e-9
                and abs(pose[2] - obs.theta) < 1e-9):
            return None
        return pose

    return DriftPolicy(step=_step)


# --- 事件条件：给 event() 用的可组合谓词 ---
def robot_within(point: XY, radius: float) -> Condition:
    """条件：机器人当前位置进入 point 半径 radius 以内。"""
    return lambda obs, state: math.dist(state.robot_xy, point) <= radius


def obstacle_removed(oid) -> Condition:
    """条件：另一个障碍物已被机器人真正搬运过(obs.removed)；不含自主漂移导致的位移，
    见 executor._drift_relocate 不置 removed 的说明。"""
    def _cond(obs, state: DriftState) -> bool:
        other = state.obstacles.get(oid)
        return other is not None and other.removed
    return _cond


def after(t_threshold: float) -> Condition:
    """条件：仿真时刻超过 t_threshold。与 scripted() 的绝对时刻表等价，这里作为可组合的
    条件提供，方便用 all_of/any_of 和其它条件一起构造复合触发。"""
    return lambda obs, state: state.t >= t_threshold


def all_of(*conditions: Condition) -> Condition:
    return lambda obs, state: all(c(obs, state) for c in conditions)


def any_of(*conditions: Condition) -> Condition:
    return lambda obs, state: any(c(obs, state) for c in conditions)


# --- 事件触发式 ---
def event(condition: Condition, keyframes: Sequence[Tuple[float, Pose]],
         loop: bool = False) -> DriftPolicy:
    """触发前不动；condition(obs, state) 首次为真时记下触发时刻，此后把 keyframes
    (相对触发时刻的 (Δt, 位姿) 序列，Δt 须从 0 开始严格递增)交给 scripted() 插值——
    事件触发本质是"原点换成触发时刻的脚本化"，不是另一套逻辑。
    """
    if keyframes[0][0] != 0.0:
        raise ValueError("event drift keyframes must start at relative dt=0")
    inner = scripted(keyframes, loop=loop)
    trigger: Dict[str, float] = {}

    def _step(obs, dt: float, state: DriftState) -> Optional[Pose]:
        if "t0" not in trigger:
            if not condition(obs, state):
                return None
            trigger["t0"] = state.t
        rel_state = DriftState(t=state.t - trigger["t0"], robot_xy=state.robot_xy,
                               obstacles=state.obstacles)
        return inner.step(obs, dt, rel_state)

    return DriftPolicy(step=_step)


# --- 反应式随机游走 ---
def reactive(rng_seed: int, speed: float, bounds: Tuple[float, float, float, float],
            avoid_radius: float = 1.5, resample_tries: int = 8) -> DriftPolicy:
    """在 bounds=(xmin,xmax,ymin,ymax) 内随机挑目标点、匀速朝它走，到达就重新采样；
    采样时跳过会让目标点落在机器人 avoid_radius 半径内的候选——只是"尽量不主动凑近"，
    不是物理意义上的碰撞保证，真撞上仍由 executor 现有的碰撞检测/重规划兜底，见
    drift-clock-integration 备忘里"碰撞语义"那条决定。用 rng_seed 播种保证同一场景
    每次跑结果一致。
    """
    rng = random.Random(rng_seed)
    xmin, xmax, ymin, ymax = bounds

    def _sample(state: DriftState) -> XY:
        p = (xmin, ymin)
        for _ in range(resample_tries):
            p = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
            if math.dist(p, state.robot_xy) >= avoid_radius:
                break
        return p

    target: Dict[str, XY] = {}

    def _step(obs, dt: float, state: DriftState) -> Optional[Pose]:
        if "xy" not in target:
            target["xy"] = _sample(state)
        tx, ty = target["xy"]
        dx, dy = tx - obs.x, ty - obs.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            target["xy"] = _sample(state)
            return None
        step_len = min(speed * dt, dist)
        theta = math.atan2(dy, dx)
        nx = obs.x + dx / dist * step_len
        ny = obs.y + dy / dist * step_len
        if step_len >= dist - 1e-9:
            target.pop("xy", None)     # 到达，下个 tick 重新采样
        return (nx, ny, theta)

    return DriftPolicy(step=_step)
