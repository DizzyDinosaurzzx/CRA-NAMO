from __future__ import annotations
from dataclasses import dataclass

def validate_lambda(value: float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError("lambda_distance must be a positive number")
    return value

@dataclass
class Config:
    # --- Robot ---
    robot_radius: float = 0.3    # robot radius

    # --- Cost function J = lambda * D + W ---
    # Obstacle difficulty is a real friction force f = mu*rho*V*g [N], so W is in
    # joules and lambda_distance is the equivalent driving resistance [N] the robot
    # pays per metre travelled. It is calibrated (not the robot's bare rolling
    # resistance) so that the detour-vs-move trade-off stays in a regime where
    # moving light obstacles is still worthwhile.
    lambda_distance: float = 350.0   # equivalent driving resistance [N]

    # --- Perception ---
    R_perc: float = 10.0       # perception radius
    sight_width: float = 0.1     # line-of-sight width

    # --- Manipulation ---
    # "Manipulation" throughout means moving an obstacle in any direction, by
    # pushing or by pulling — the robot grips wherever it likes on the perimeter,
    # so there is no privileged direction. (The code used to say "push"
    # everywhere, from when that was the only option.)
    R_manip: float = 5.0      # obstacles can only be relocated within this radius around current pose
    # Soft preference for dropping the obstacle ahead of the robot rather than
    # behind it, so it does not end up blocking the way back. It is a bias on the
    # candidate drop poses, not a restriction: the robot may still move an
    # obstacle in any direction, and set this to 0 to remove even the bias.
    manip_forward_penalty: float = 2.0
    manip_max_frames_per_action: int = 30
    check_obstacle_collision: bool = True  # collision detection between obstacles
    full_reveal_on_contact: bool = False

    # --- Robot–obstacle contact ---
    # The robot has to stay flush against the obstacle for the whole time the
    # obstacle is moving. It may grip anywhere on the perimeter (so it can push
    # or pull, in any direction) and may slide that grip along the surface while
    # the obstacle moves. Its own travel is charged at lambda_distance, exactly
    # like ordinary driving. Set contact_required = False to recover the older
    # model where obstacles moved while the robot stood on its roadmap node.
    contact_required: bool = True
    contact_station_spacing: float = 0.3   # spacing of candidate grip points around the obstacle [m]
    contact_max_slide: float = 0.6         # how far the grip may slide per manipulation sub-step [m]
    contact_clearance: float = 0.01        # tolerance separating "touching" from "overlapping" [m]

    # --- SE(2) path planner (the obstacle's own route) ---
    se2_use_planner: bool = True
    se2_cell: float = 0.15
    se2_n_theta: int = 12
    se2_connectivity: int = 8
    se2_containment: str = "centroid"
    se2_rot_weight: float | None = None

    # --- Roadmap ---
    grid_step: float = 0.3    # roadmap node grid spacing
    conn_radius: float =0.6   # roadmap node connection radius

    # --- Search ---
    strategy: str = "normal"    # "normal" | "shortest" | "easiest"
    use_llm_ordering: bool = True
    max_expansions: int = 100000

    # --- Online loop ---
    step_execute_edges: int = 1     # re-perception frequency
    max_replans: int = 10000

    # --- LLM ---
    # Reasoning is on, and the completion length is uncapped, because the prompt
    # asks for a multi-step derivation (category -> mass -> bulk density -> mu ->
    # product). Denied the room to run it, the model degenerates into copying a
    # row out of the anchor table in the prompt: measured 6.1x typical error with
    # 46% of answers collapsing onto a single value, and the collapse target
    # moves when the table is reordered — i.e. the answer tracked the prompt
    # layout, not the object. With reasoning enabled the same model on the same
    # prompt reaches 1.33x typical error (Spearman 0.95). See bench/llm_test_out/.
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

    # --- Other ---
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

