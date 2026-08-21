"""全局配置与参数校验。"""
from __future__ import annotations
from dataclasses import dataclass

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
    # 差速驱动圆盘：先原地转向再直线行驶；数值取室内服务机器人量级（TurtleBot/Jackal 级），换算成秒见 timing.py。
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
    time_importance: float = 0.5

    # --- 感知 ---
    R_perc: float = 10.0       # 感知半径
    sight_width: float = 0.1     # 视线宽度

    # --- 搬运 ---
    # "搬运"统指沿任意方向推或拉障碍物：机器人可在其周边任一点抓取，无优先方向。
    R_manip: float = 5.0      # 障碍物只能移到当前位姿周围该半径之内
    # 软性偏好把障碍物放在机器人前方而非身后，免得挡住回路；只是对候选放置点的偏置而非限制，置 0 可彻底去掉。
    manip_forward_penalty: float = 2.0
    manip_max_frames_per_action: int = 30
    check_obstacle_collision: bool = True  # 障碍物之间的碰撞检测
    full_reveal_on_contact: bool = False

    # --- 机器人与障碍物接触 ---
    # 障碍物移动全程机器人须贴住它：可在周边任一点抓取（任意方向推/拉），移动中抓取点还可沿表面滑移；
    # 自身行程与普通行驶一样按 lambda_distance 计费。contact_required=False 退回旧模型（机器人在路网点上原地等待）。
    contact_required: bool = True
    contact_station_spacing: float = 0.3   # 障碍物周边候选抓取点的间距 [m]
    contact_max_slide: float = 0.6         # 每个搬运子步内抓取点可滑移的距离 [m]
    contact_clearance: float = 0.01        # 区分"接触"与"重叠"的容差 [m]

    # --- SE(2) 路径规划器（障碍物自身的路线） ---
    se2_use_planner: bool = True
    se2_cell: float = 0.15
    se2_n_theta: int = 12
    se2_connectivity: int = 8
    se2_containment: str = "centroid"
    se2_rot_weight: float | None = None

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
    # 推理必须开着：关掉时模型退化成照抄 prompt 锚点表的一行，典型误差 6.1x、
    # 46% 答案塌缩到同一值；开着则降到 1.33x（Spearman 0.95）。详见 bench/llm_test_out/。
    deepseek_api_key: str = "sk-c1ea9b080fc444ceb1f5fa7901e3b92f"
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_thinking: bool = True    # False 会复现照抄表格一行的旧行为
    llm_max_tokens: int | None = None  # None 表示不设上限；推理约需 3-4k tokens
    # 单次推理实测可达 ~80 s；超时不会报错而是静默回退启发式，故取远高于最坏观测值。
    llm_timeout: float = 300.0
    llm_max_retries: int = 2

    # --- 其他 ---
    rng_seed: int = 0
    out_dir: str = "img"
    save_frames: bool = False   # 是否保存逐步动画（GIF）
    gif_fps: float = 5.0       # 动画速度（帧/秒）
    gif_end_hold_s: float = 2  # 循环前最后一帧的停留时长
    gif_dpi: int = 300         # GIF 内每帧的渲染分辨率
    verbose: bool = True

    def __post_init__(self):
        self.lambda_distance = validate_lambda(self.lambda_distance)
        self.time_importance = validate_time_importance(self.time_importance)

    def log(self, *args):
        if self.verbose:
            print(*args)

