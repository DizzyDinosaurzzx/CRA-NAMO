from __future__ import annotations
from dataclasses import dataclass

def validate_lambda(value: float) -> float: 
    value = float(value)
    if value <= 0:
        raise ValueError("lambda_distance must be a positive number")
    return value

@dataclass
class Config:
    # -------- Robot -------- #
    robot_radius: float = 0.3    # robot radius

    # -------- Cost function J = lambda * D + W -------- #
    lambda_distance: float = 1    # motion work coefficient

    # -------- Perception -------- #
    R_perc: float = 10.0       # perception radius
    sight_width: float = 0.1     # line-of-sight width

    # -------- Manipulation -------- #
    R_push: float = 5.0       # obstacles can only be relocated within this radius around current pose
    push_forward_penalty: float = 2.0
    check_obstacle_collision: bool = True  # collision detection between obstacles
    full_reveal_on_contact: bool = False

    # -------- SE2 Push Planner --------
    push_use_planner: bool = True
    push_cell: float = 0.15
    push_n_theta: int = 12
    push_connectivity: int = 8
    push_containment: str = "centroid"
    push_rot_weight: float | None = None
    push_max_frames_per_action: int = 30

    # -------- Roadmap -------- #
    grid_step: float = 0.3    # roadmap node grid spacing
    conn_radius: float =0.6   # roadmap node connection radius

    # -------- Search -------- #
    strategy: str = "normal"    # "normal" | "shortest" | "easiest"
    use_llm_ordering: bool = True
    max_expansions: int = 100000

    # -------- Online loop -------- #
    step_execute_edges: int = 1     # re-perception frequency
    max_replans: int = 10000

    # -------- LLM -------- #
    deepseek_api_key: str = "sk-08760d6dfd7c46d6928bebd3689b25c7"
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-pro"
    llm_timeout: float = 30.0
    llm_max_retries: int = 2

    # -------- Other -------- #
    rng_seed: int = 0
    out_dir: str = "img"
    save_frames: bool = True   # whether to save the per-step animation (GIF)
    gif_fps: float = 5.0       # animation speed, frames per second
    gif_end_hold_s: float = 2  # hold the last frame this long before looping
    gif_dpi: int = 300         # per-frame render resolution inside the GIF
    verbose: bool = True

    def __post_init__(self):
        self.lambda_distance = validate_lambda(self.lambda_distance)

    def log(self, *args):
        if self.verbose:
            print(*args)
