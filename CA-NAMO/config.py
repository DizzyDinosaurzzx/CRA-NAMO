"""CA-NAMO的全局配置"""

from dataclasses import dataclass, field
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
    phi_0: float = 0.05       # 传感器最小可分辨角

    # -------- 操作 -------- #
    R_push: float = 5.0       # 障碍物只能在当前位姿周围的该半径内重新放置
    drop_ring_samples: int = 24     # 在障碍物周围尝试的候选放置方向数
    drop_radius_steps: int = 6      # 尝试的候选放置距离数（0..R_push）
    check_obstacle_collision: bool = True  # 放置障碍物时是否检测与其他障碍物的碰撞
    full_reveal_on_contact: bool = False   # 推动碰撞后：True=获知被撞障碍物全部信息；False=只知“此处有物”

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
    out_dir: str = "img"       
    save_frames: bool = True   
    verbose: bool = True

    def __post_init__(self):
        if self.lambda_distance < 0:
            raise ValueError("lambda_distance 必须为非负数")

    def log(self, *args):
        if self.verbose:
            print(*args)
