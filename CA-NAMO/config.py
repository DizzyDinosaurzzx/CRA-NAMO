from __future__ import annotations
from dataclasses import dataclass

def validate_lambda(value: float) -> float: #lambda_distance 正数检验
    value = float(value)
    if value <= 0:
        raise ValueError("lambda_distance 必须为正数")
    return value

@dataclass
class Config:
    # -------- 机器人 -------- #
    robot_radius: float = 0.3    # 机器人半径

    # -------- 代价函数 J = lambda * D + W -------- #
    lambda_distance: float = 100000    # 移动做工系数

    # -------- 感知 -------- #
    R_perc: float = 1000.0       # 感知半径
    sight_width: float = 0.2     # 视线宽度

    # -------- 操作 -------- #
    R_push: float = 5.0       # 障碍物只能在当前位姿周围的该半径内重新放置
    push_forward_penalty: float = 2.0
    check_obstacle_collision: bool = True  # 障碍物之间碰撞检测
    full_reveal_on_contact: bool = False

    # -------- SE2 Push Planner --------
    push_use_planner: bool = True
    push_cell: float = 0.15
    push_n_theta: int = 12
    push_connectivity: int = 8
    push_containment: str = "centroid"
    push_rot_weight: float | None = None
    push_max_frames_per_action: int = 30

    # -------- 路网 -------- #
    grid_step: float = 1    # 路网节点网格间距
    conn_radius: float = 2  # 路网节点的连接半径

    # -------- 搜索 -------- #
    use_llm_ordering: bool = True
    max_expansions: int = 100000    

    # -------- 在线循环 -------- #
    step_execute_edges: int = 1     # 重新感知频率
    max_replans: int = 10000         

    # -------- LLM-------- #
    deepseek_api_key: str = ""     
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-flash"
    llm_timeout: float = 30.0
    llm_max_retries: int = 2

    # -------- 其他 -------- #
    rng_seed: int = 0
    out_dir: str = "img"       
    save_frames: bool = True   # 是否保存过程帧
    verbose: bool = True

    def __post_init__(self):
        self.lambda_distance = validate_lambda(self.lambda_distance)

    def log(self, *args):
        if self.verbose:
            print(*args)
