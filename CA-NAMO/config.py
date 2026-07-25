"""
代价感知 NAMO 的全局配置
"""

from dataclasses import dataclass, field

@dataclass
class Config:
    # -------- 机器人 --------
    robot_radius: float = 0.35 # 机器人尺寸半径

    # -------- 代价函数 J --------
    lambda_d: float = 1.0     # 行驶距离权重（“惜路”）
    lambda_w: float = 1.0     # 操作功权重（“惜力”）

    # -------- 感知 --------
    R_perc: float = 8.0       # 感知半径（以机器人为中心的感知圆半径）
    sight_width: float = 0.2  # 视线宽度：>0 时视线是有该宽度的“走廊”，需完整穿过自由空间
                              # 且不碰障碍物才算看得见；0=零宽度射线（再窄的缝也能看穿）。
                              # 调大可禁止“细缝偷窥”（缝宽 < sight_width 就看不穿）。

    # -------- 操作 --------
    R_push: float = 5.0       # 障碍物只能在当前位姿周围的该半径内重新放置
    drop_ring_samples: int = 24     # 在障碍物周围尝试的候选放置方向数
    drop_radius_steps: int = 6      # 尝试的候选放置距离数（0..R_push）
    check_obstacle_collision: bool = True  # 放置障碍物时是否检测与其他障碍物的碰撞
    full_reveal_on_contact: bool = False   # 推动碰撞后：True=直接获知被撞障碍物全部信息；
                                           # False(更真实)=只知“此处有物”，移开遮挡后才完整感知

    # -------- 路网 --------
    grid_step: float = 1    # 静态自由空间中路网节点的网格间距
    conn_radius: float = 2   # 两个节点距离不超过此值且视线无阻时建立连接

    # -------- 搜索 --------
    use_llm_ordering: bool = True
    max_expansions: int = 200000    # A* 扩展次数上限

    # -------- 在线循环 --------
    step_execute_edges: int = 1     # 重新感知的刷新频率（走几步就重新更新一遍感知）
    max_replans: int = 2000          # 规划-执行-感知-重规划循环上限

    # -------- LLM（DeepSeek）--------
    deepseek_api_key: str = ""             # 为空时使用启发式回退方案
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-chat"  # DeepSeek-V3
    llm_timeout: float = 30.0
    llm_max_retries: int = 2

    # -------- 其他 --------
    rng_seed: int = 0
    out_dir: str = "img"       # 可视化结果的输出目录
    save_frames: bool = True   # 是否逐步保存机器人每一步运动的帧图片（img/frames/）
    verbose: bool = True

    def log(self, *args):
        if self.verbose:
            print(*args)
