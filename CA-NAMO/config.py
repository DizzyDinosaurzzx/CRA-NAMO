"""全局配置与参数校验。"""
from __future__ import annotations
from dataclasses import dataclass, field

def validate_lambda(value: float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError("lambda_distance must be a positive number")
    return value


def validate_time_importance(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("time_importance must be between 0 and 1")
    return value

@dataclass
class Config:
    # --- 机器人 ---
    robot_radius: float = 0.3    # 机器人半径

    # --- 机器人动力学（只影响任务时间，不进入 J） ---
    v_max: float = 0.8           # 空载巡航速度 [m/s]
    a_max: float = 0.5           # 空载线加速度 [m/s^2]
    w_max: float = 1.5           # 空载原地转向角速度 [rad/s]
    alpha_max: float = 2.0       # 空载角加速度 [rad/s^2]
    # 随行搬运时机器人贴着负载须防滑脱，行驶与转向都要温和得多。
    v_max_contact: float = 0.25      # [m/s]
    a_max_contact: float = 0.20      # [m/s^2]
    w_max_contact: float = 0.50      # [rad/s]
    alpha_max_contact: float = 0.80  # [rad/s^2]
    grip_time: float = 2.0       # 抓上与松开障碍物各需这么多秒

    # --- 代价函数 J = lambda * D + W ---
    # 难度是真实摩擦力 f = mu*rho*V*g [N]，W 单位为焦耳；lambda_distance 是标定出的等效行驶阻力 [N]
    # （并非裸滚动阻力），为的是让"绕行 vs 搬开"的权衡仍偏向搬开轻障碍物。
    lambda_distance: float = 350.0   # 等效行驶阻力 [N]

    # --- 搜索优化目标：C = (1-w)*J + w*(lambda*v_max)*T ---
    # 求快与求省的权重：0 为纯能量 J（与旧模型完全一致），1 为纯最快 T。
    # 一秒按 lambda*v_max [W] 计价——即巡航一秒本就要花的能量，无需另行标定，见 cost.time_price。
    time_importance: float = 1

    # --- 感知 ---
    R_perc: float = 100       # 感知半径
    sight_width: float = 0.1     # 视线宽度
    # 额外的重新感知触发阈值(见 executor._perc_poll)：累计位移/累计耗时任一超过就
    # 补扫一次世界；默认 inf 即关闭，不影响现有场景——今天的按边/按动作感知点已经够用，
    # 只有配合会自主移动的障碍物(drift.py)时才需要收紧。
    perc_step: float = float("inf")        # 触发重新感知的累计位移 [m]
    perc_time_step: float = float("inf")   # 触发重新感知的累计仿真耗时 [s]（不含 plan_time）

    # --- 自主移动障碍物挡路时的应对 ---
    # 机器人走到一半被挂了 drift 策略的障碍物挡住时，与其立刻重规划绕路，不如先原地
    # 等它自己让开——这不是给 A* 加时空维度的"等待"动作(那需要预测 drift 的未来轨迹，
    # 代价和风险都大得多，故未做)，只是执行层面"先等等看"的轻量策略。默认 0 即不等，
    # 不影响任何不挂 drift 的场景；只在挡路的障碍物确实挂了 drift 时才会触发，见
    # executor._wait / _any_drifting。
    wait_on_block_s: float = 0.0       # 被 drift 障碍物挡路时先等待的秒数
    wait_substep_s: float = 0.5        # 等待按多大步长切分推进(避免大 Δt 让漂移跳步)

    # --- 搬运 ---
    R_manip: float = 5.0      # 障碍物只能移到当前位姿周围该半径之内
    manip_forward_penalty: float = 2.0
    manip_max_frames_per_action: int = 30
    check_obstacle_collision: bool = True  # 障碍物之间的碰撞检测
    full_reveal_on_contact: bool = False

    # --- 机器人与障碍物接触 ---
    contact_station_spacing: float = 0.3   # 障碍物周边候选抓取点的间距 [m]
    contact_max_slide: float = 0.6         # 每个搬运子步内抓取点可滑移的距离 [m]
    contact_clearance: float = 0.01        # 区分"接触"与"重叠"的容差 [m]

    # --- SE(2) 路径规划器（障碍物自身的路线） ---
    se2_cell: float = 0.15
    se2_n_theta: int = 12
    se2_connectivity: int = 8
    se2_containment: str = "centroid"
    se2_rot_weight: float | None = None

    # --- 次生风险评估（可选，默认关闭；见 risk.py） ---
    risk_assessment_enabled: bool = False
    risk_tier_penalty: dict = field(default_factory=lambda: {
        "none": 0.0,
        "low": 100.0,
        "moderate": 1000.0,
        "high":10000.0,
        "critical": 100000.0,
    })

    # --- 路网 ---
    grid_step: float = 0.3    # 路网节点网格间距
    conn_radius: float =0.6   # 路网节点连接半径

    # --- 搜索 ---
    strategy: str = "normal"    # "normal" | "shortest"
    use_llm_ordering: bool = True
    max_expansions: int = 100000

    # --- 在线循环 ---
    step_execute_edges: int = 1     # 重新感知的频率
    max_replans: int = 10000

    # --- LLM ---
    deepseek_api_key: str = "sk-c1ea9b080fc444ceb1f5fa7901e3b92f"
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-flash-vision-exp"
    deepseek_thinking: bool = True
    llm_max_tokens: int | None = None  # None 表示不设上限；推理约需 3-4k tokens
    llm_timeout: float = 300.0
    llm_max_retries: int = 2

    # --- 其他 ---
    rng_seed: int = 0
    out_dir: str = "img"
    save_frames: bool = False   # 是否保存逐步动画（GIF）
    gif_speed: float = 1.0      # 播放速度倍率，相对仿真时间
    gif_fps: float = 5.0       # 单帧最短停留对应的帧率上限
    gif_max_frame_s: float = 5.0   # 单帧最长停留时长(仿真秒数按 gif_speed 折算后)[s]
    gif_end_hold_s: float = 2  # 循环前最后一帧的停留时长
    gif_dpi: int = 300         # GIF 内每帧的渲染分辨率
    verbose: bool = True

    def __post_init__(self):
        self.lambda_distance = validate_lambda(self.lambda_distance)
        self.time_importance = validate_time_importance(self.time_importance)
        self._log_last: str | None = None   # 折叠连续重复的 log() 内容，避免刷屏
        self._log_repeat: int = 0

    def log(self, *args):
        if not self.verbose:
            return
        msg = " ".join(str(a) for a in args)
        if msg == self._log_last:
            self._log_repeat += 1
            return
        self.flush_log()
        print(msg)
        self._log_last = msg

    def flush_log(self):
        if self._log_repeat:
            print(f"  ↳ 上一条重复了 {self._log_repeat} 次，已折叠")
            self._log_repeat = 0
        self._log_last = None

