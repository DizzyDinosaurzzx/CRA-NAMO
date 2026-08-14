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
    # Obstacle difficulty is a real friction force f = mu*rho*V*g [N], so W is in
    # joules and lambda_distance is the equivalent driving resistance [N] the robot
    # pays per metre travelled. It is calibrated (not the robot's bare rolling
    # resistance) so that the detour-vs-push trade-off stays in a regime where
    # pushing light obstacles is still worthwhile.
    lambda_distance: float = 350.0   # equivalent driving resistance [N]

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
    # Reasoning is on, and the completion length is uncapped, because the prompt
    # asks for a multi-step derivation (category -> mass -> bulk density -> mu ->
    # product). Denied the room to run it, the model degenerates into copying a
    # row out of the anchor table in the prompt: measured 6.1x typical error with
    # 46% of answers collapsing onto a single value, and the collapse target
    # moves when the table is reordered — i.e. the answer tracked the prompt
    # layout, not the object. With reasoning enabled the same model on the same
    # prompt reaches 1.33x typical error (Spearman 0.95). See llm_test_out/.
    deepseek_api_key: str = "sk-c1ea9b080fc444ceb1f5fa7901e3b92f"
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_thinking: bool = True    # False reproduces the old copy-a-row behaviour
    llm_max_tokens: int | None = None  # None -> omit the cap; reasoning needs ~3-4k
    # Reasoning takes seconds, not milliseconds: a single call was measured up to
    # ~80 s. A short timeout here does not fail loudly, it silently falls back to
    # the heuristic, so it is set well above the observed worst case.
    llm_timeout: float = 300.0
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
