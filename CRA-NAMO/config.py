from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import kinematics

class StrategyFlags(NamedTuple):
    """一个策略 arm 打开的能力。"""

    llm_cost: bool       # LLM 估计推动代价 mu*rho，否则查材料表。
    llm_risk: bool       # LLM 评估风险等级，否则查关键词表。
    shortest: bool       # 忽略搬移代价，只按几何最短路前进。
    llm_choice: bool     # 由 LLM 在几何层验证过的候选中做离散选择。


# 关闭 LLM 的一项表示改用离线启发式，而不是不估计。
STRATEGIES: dict[str, StrategyFlags] = {
    # LLM 只提供代价/风险数值，仍由 A* 决策。
    "llm-cost-risk": StrategyFlags(True, True, False, False),
    "llm-cost": StrategyFlags(True, False, False, False),
    "llm-risk": StrategyFlags(False, True, False, False),
    "no-llm": StrategyFlags(False, False, False, False),
    # 不做取舍的下界基线。
    "shortest": StrategyFlags(False, False, True, False),
    # L2：几何层枚举并验证候选，LLM 直接选一个。
    "llm-choice": StrategyFlags(True, True, False, True),
}
DEFAULT_STRATEGY = "shortest"


def validate_strategy(value: str) -> str:
    name = str(value).strip().lower().replace("_", "-")
    if name not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {value!r}; available: {', '.join(STRATEGIES)}")
    return name


# DeepSeek 的推理强度档位。medium/xhigh 在服务端并入 high，这里先行归一，
# 避免日志里出现一个实际不生效的档位名。
REASONING_EFFORTS = ("low", "high", "max")
_EFFORT_ALIASES = {"medium": "high", "xhigh": "high"}


def validate_reasoning_effort(value: str) -> str:
    name = str(value).strip().lower()
    name = _EFFORT_ALIASES.get(name, name)
    if name not in REASONING_EFFORTS:
        raise ValueError(
            f"unknown reasoning effort {value!r}; available: "
            + ", ".join(REASONING_EFFORTS))
    return name


def validate_time_importance(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("time_importance must lie in [0, 1]")
    return value


def validate_lambda(value: float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError("lambda_distance must be a positive number")
    return value


@dataclass
class Config:
    """规划、执行和可视化共用的配置。"""

    robot_radius: float = 0.3

    # 空载和载物行驶的运动限制。
    robot_v_max: float = 0.6         # [米/秒]
    robot_a_max: float = 0.5         # [米/秒^2]
    robot_w_max: float = 1.5         # [弧度/秒]
    robot_alpha_max: float = 2.0     # [弧度/秒^2]
    robot_v_max_loaded: float = 0.25
    robot_a_max_loaded: float = 0.2
    robot_w_max_loaded: float = 0.6
    robot_alpha_max_loaded: float = 0.8

    # 目标函数：C = (1 - w)J + w(time_value * T) + R。
    time_importance: float = 0.0     # w 的范围为 [0, 1]
    time_value: float = 100.0        # [焦耳/秒]

    lambda_distance: float = 350.0   # 等效行驶阻力 [牛]

    risk_weight: float = 1.0
    # 达到此等级及以上的风险禁止搬移。
    risk_forbidden_level: str = "extreme"
    # 接触力超过该比例时，后备风险等级上调一级。
    risk_heavy_ratio: float = 3.0

    R_perc: float = 10.0             # 感知半径 [米]
    sight_width: float = 0.1         # 视线宽度 [米]

    R_manip: float = 5.0             # 搬移搜索半径 [米]
    # 轻度偏好前向放置姿态；设为零可关闭。
    manip_forward_penalty: float = 2.0
    # 按放下之后机器人还要走的路来挑落点；关掉则只看搬移本身的代价。
    manip_lookahead: bool = True
    manip_max_frames_per_action: int = 30
    # 因降低路线图净空而增加的惩罚，单位为障碍物移动距离。
    manip_blocked_edge_penalty_m: float = 0.0

    contact_required: bool = True
    contact_station_spacing: float = 0.3   # 周边采样间距 [米]
    contact_max_slide: float = 0.6         # 每步最大握持滑移 [米]
    contact_clearance: float = 0.01        # 接触容差 [米]
    # 测得的力与规划值相差达到该比例时重新规划。
    contact_replan_ratio: float = 1.25
    # 相对摩擦力的最大施力比例；设为零可关闭限制。
    contact_max_force_ratio: float = 1.0
    # 这些限制描述物理可行性；任一项设为零可关闭对应限制。
    robot_max_push_force: float = 200_000.0
    robot_push_height: float = 0.15
    push_friction_mu: float = 0.5

    se2_cell: float = 0.15
    se2_n_theta: int = 12
    se2_connectivity: int = 8
    se2_containment: str = "centroid"
    se2_rot_weight: float | None = None
    se2_goal_candidates: int = 24
    # 初始放置姿态全部被拒时扩大候选列表。
    se2_goal_widen: int = 1

    # 世界运动与路径代价使用相同的旋转折算距离单位。
    dynamic_speed: float = 0.3          # 障碍物默认移动速度 [米/秒]
    dynamic_step: float = 0.25          # 每个运动子步模拟的秒数
    dynamic_block_patience: float = 2.0  # 等待后开始寻找其他路线的秒数
    dynamic_give_up: float = 30.0       # 卡住后原地停车前的秒数
    dynamic_replan_backoff: float = 2.0  # 再次请求失败路线前等待的秒数

    dynamic_wait_step: float = 2.0      # 每次无法规划时机器人的等待秒数
    dynamic_max_wait: float = 90.0      # 判定通道关闭前的总等待秒数

    grid_step: float = 0.3          # 路线图节点间距 [米]
    conn_radius: float = 0.6        # 路线图连接半径 [米]

    # 五选一的规划策略；决定 LLM 是否参与代价/风险估计，见 STRATEGIES。
    strategy: str = DEFAULT_STRATEGY
    use_llm_ordering: bool = True
    max_expansions: int = 100000

    # belief 没有更新时最多连续执行的边数；0 表示不限，只在 belief 更新时重规划。
    step_execute_edges: int = 0
    max_replans: int = 10000

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4.1-flash-expires-on-0910"
    deepseek_thinking: bool = True
    deepseek_reasoning_effort: str = "low"
    llm_max_tokens: int | None = None
    llm_timeout: float = 300.0
    llm_max_retries: int = 2
    perception_llm_timeout: float = 60.0
    perception_llm_max_calls: int = 8

    # llm-choice：每次决策最多给 LLM 看几个候选，按离机器人的距离取近的。
    llm_choice_max_options: int = 10
    # 候选集合不变时沿用上次决定，避免每条 0.3 m 边都重新提问。
    llm_choice_reuse_decision: bool = True

    out_dir: str = "img"
    save_frames: bool = True
    gif_fps: float = 10
    gif_end_hold_s: float = 2
    gif_dpi: int = 300
    gif_time_step: float = 1.0
    gif_max_frames: int = 400
    verbose: bool = True
    _log_sink: Callable[[str], None] = field(default=print, repr=False)
    _log_flush: Callable[[], None] = field(default=lambda: None, repr=False)

    def __post_init__(self):
        self.lambda_distance = validate_lambda(self.lambda_distance)
        self.time_importance = validate_time_importance(self.time_importance)
        self.strategy = validate_strategy(self.strategy)
        self.deepseek_reasoning_effort = validate_reasoning_effort(
            self.deepseek_reasoning_effort)

    @property
    def use_llm_cost(self) -> bool:
        """LLM 是否估计搬移代价（mu*rho）？否则使用材料表启发式。"""
        return STRATEGIES[self.strategy].llm_cost

    @property
    def use_llm_risk(self) -> bool:
        """LLM 是否评估搬移风险等级？否则使用关键词启发式。"""
        return STRATEGIES[self.strategy].llm_risk

    @property
    def shortest_path_mode(self) -> bool:
        """是否忽略搬移代价，只按最短路径前进并清除沿途障碍物？"""
        return STRATEGIES[self.strategy].shortest

    @property
    def llm_choice(self) -> bool:
        """是否由 LLM 在候选方案中做决策，而不是由 A* 比较代价？"""
        return STRATEGIES[self.strategy].llm_choice

    def free_profile(self) -> kinematics.MotionProfile:
        """返回空载运动参数。"""
        return kinematics.MotionProfile(
            self.robot_v_max, self.robot_a_max,
            self.robot_w_max, self.robot_alpha_max)

    def loaded_profile(self) -> kinematics.MotionProfile:
        """返回载物运动参数。"""
        return kinematics.MotionProfile(
            self.robot_v_max_loaded, self.robot_a_max_loaded,
            self.robot_w_max_loaded, self.robot_alpha_max_loaded)

    def log(self, *args):
        if self.verbose:
            self._log_sink(" ".join(str(a) for a in args))

    def set_logger(self, sink: Callable[[str], None],
                   flush: Callable[[], None]) -> None:
        """设置规划和执行使用的控制台日志函数。"""
        self._log_sink = sink
        self._log_flush = flush

    def flush_log(self) -> None:
        self._log_flush()
