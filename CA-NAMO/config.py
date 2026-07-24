"""
代价感知 NAMO 的全局配置
"""

from dataclasses import dataclass, field

@dataclass
class Config:
    # -------- 代价函数 J --------
    lambda_d: float = 1.0     # 行驶距离权重（“惜路”）
    lambda_w: float = 1.0     # 操作功权重（“惜力”）

    # -------- 感知 --------
    R_perc: float = 8.0       # 感知半径（以机器人为中心的感知圆半径）

    # -------- 操作 --------
    R_push: float = 6.0       # 障碍物只能在当前位姿周围的该半径内重新放置
    drop_ring_samples: int = 24     # 在障碍物周围尝试的候选放置方向数
    drop_radius_steps: int = 6      # 尝试的候选放置距离数（0..R_push）

    # 优化障碍物放置的逻辑：是否使用LLM指导？是否实时更新位置？

    # -------- 路网 --------
    grid_step: float = 2.0    # 静态自由空间中路网节点的网格间距
    conn_radius: float = 3.0  # 两个节点距离不超过此值且视线无阻时建立连接
    robot_radius: float = 0.4 # 机器人圆盘半径；边按此值膨胀

    # -------- 搜索 --------
    use_llm_ordering: bool = True
    max_expansions: int = 200000    # A* 扩展次数上限

    # -------- 在线循环 --------
    step_execute_edges: int = 1     # 重新感知的刷新频率（走几步就重新更新一遍感知）
    max_replans: int = 400          # 规划-执行-感知-重规划循环上限

    # -------- LLM（DeepSeek）--------
    deepseek_api_key: str = ""             # 为空时使用启发式回退方案
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-chat"  # DeepSeek-V3
    llm_timeout: float = 30.0
    llm_max_retries: int = 2

    # -------- 其他 --------
    rng_seed: int = 0
    out_dir: str = "img"       # 可视化结果的输出目录
    verbose: bool = True

    def log(self, *args):
        if self.verbose:
            print(*args)
