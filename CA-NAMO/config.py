"""CA-NAMO的全局配置"""

from __future__ import annotations
from dataclasses import dataclass


def validate_lambda(value: float) -> float:
    """lambda_distance 的唯一校验入口（构造时与命令行覆盖时共用）。"""
    value = float(value)
    if value < 0:
        raise ValueError("lambda_distance 必须为非负数")
    return value


@dataclass
class Config:
    # -------- 机器人 -------- #
    robot_radius: float = 0.35 # 机器人尺寸半径
    touch_margin: float = 0.15 # 触摸感知余量，真实触摸圆半径 = robot_radius + touch_margin

    # -------- 代价函数 J = lambda * D + W -------- #
    lambda_distance: float = 1.0     # 机器人单位移动距离的做功系数lambda

    # -------- 感知 -------- #
    R_perc: float = 8.0       # 感知半径（以机器人为中心的感知圆半径）
    sight_width: float = 0.2  # 视线宽度

    # -------- 操作 -------- #
    R_push: float = 5.0       # 障碍物只能在当前位姿周围的该半径内重新放置
    drop_ring_samples: int = 24     # 在障碍物周围尝试的候选放置方向数
    drop_radius_steps: int = 6      # 尝试的候选放置距离数（0..R_push）
    check_obstacle_collision: bool = True  # 放置障碍物时是否检测与其他障碍物的碰撞
    push_clear_scope: str = "repeat"
    # 推开障碍物的代价如何记账（三者的放置硬约束/计费方式不同）：
    #   "edge"   = 只腾空当前边，且该 oid 之后所有边免费  -> 系统性低估（实测约 8x）
    #   "repeat" = 只腾空当前边，但【每条被挡的边都照实计一次推动费】-> 诚实、保守
    #   "union"  = 一次推到全清该障碍物挡住的所有通道       -> 记账自洽但实际做功更大
    full_reveal_on_contact: bool = False   # 推动碰撞后：True=获知被撞障碍物全部信息；False=只知"此处有物"

    # -------- SE2 Push Planner --------
    push_use_planner: bool = True
    push_cell: float = 0.15
    push_n_theta: int = 12
    push_connectivity: int = 8
    push_containment: str = "centroid"
    # 1 弧度旋转折合多少米平移。None = 自动取矩形的摩擦力矩臂 r̄（见
    # push_planner.mean_rotation_radius），使 W = difficulty * se2_cost
    # 恰好等于 μmg·d + μmg·θ·r̄ 这一物理做功模型。
    push_rot_weight: float | None = None
    push_sample_theta: bool = True
    push_max_frames_per_action: int = 30

    # -------- 路网 -------- #
    grid_step: float = 1    # 静态自由空间中路网节点的网格间距
    conn_radius: float = 2   # 两个节点距离不超过此值且视线无阻时建立连接

    # -------- 搜索 -------- #
    use_llm_ordering: bool = True
    max_expansions: int = 100000    

    # -------- 在线循环 -------- #
    step_execute_edges: int = 1     # 重新感知的刷新频率（走几步就重新更新一遍感知）
    max_replans: int = 1000         

    # -------- LLM-------- #
    deepseek_api_key: str = ""     
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-flash"
    llm_timeout: float = 30.0
    llm_max_retries: int = 2

    # -------- 其他 -------- #
    rng_seed: int = 0
    out_dir: str = "img"       # 可视化结果的输出目录
    save_frames: bool = True   # 是否保存过程帧（img/frames_<地图名>/）
    verbose: bool = True

    def __post_init__(self):
        self.lambda_distance = validate_lambda(self.lambda_distance)

    def log(self, *args):
        if self.verbose:
            print(*args)
