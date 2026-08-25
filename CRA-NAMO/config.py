"""Runtime configuration and validation."""

from __future__ import annotations
from dataclasses import dataclass

import kinematics
import log


STRATEGIES = ("normal", "shortest")


def validate_strategy(value: str) -> str:
    if value not in STRATEGIES:
        raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
    return value


def validate_time_importance(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError("time_importance must lie in [0, 1]")
    return value


def validate_lambda(value: float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError("lambda_distance must be a positive number")
    return value


@dataclass
class Config:
    """Configuration shared by planning, execution, and visualization."""

    robot_radius: float = 0.2

    # Free and loaded motion limits for the turn-then-drive model.
    robot_v_max: float = 0.6         # [m/s]
    robot_a_max: float = 0.5         # [m/s^2]
    robot_w_max: float = 1.5         # [rad/s]
    robot_alpha_max: float = 2.0     # [rad/s^2]
    robot_v_max_loaded: float = 0.25
    robot_a_max_loaded: float = 0.2
    robot_w_max_loaded: float = 0.6
    robot_alpha_max_loaded: float = 0.8

    # C = (1 - w)J + w(time_value * T) + R.
    time_importance: float = 0.0     # w in [0, 1]
    time_value: float = 100.0        # [J/s]

    lambda_distance: float = 350.0   # equivalent driving resistance [N]

    risk_weight: float = 1.0

    R_perc: float = 10.0             # perception radius [m]
    sight_width: float = 0.1         # line-of-sight width [m]

    R_manip: float = 5.0             # relocation search radius [m]
    # Soft preference only; zero permits unbiased placement.
    manip_forward_penalty: float = 2.0
    manip_max_frames_per_action: int = 30

    contact_required: bool = True
    contact_station_spacing: float = 0.3   # perimeter sample spacing [m]
    contact_max_slide: float = 0.6         # maximum grip slide per step [m]
    contact_clearance: float = 0.01        # contact tolerance [m]
    # Maximum applied force relative to friction; zero disables the limit.
    contact_max_force_ratio: float = 1.0

    se2_cell: float = 0.15
    se2_n_theta: int = 12
    se2_connectivity: int = 8
    se2_containment: str = "centroid"
    se2_rot_weight: float | None = None
    # Candidate drop poses are checked in ascending cost order.
    se2_goal_candidates: int = 24

    grid_step: float = 0.3          # roadmap node spacing [m]
    conn_radius: float = 0.6        # roadmap connection radius [m]

    strategy: str = "normal"    # "normal" | "shortest"
    use_llm_ordering: bool = True
    max_expansions: int = 100000

    step_execute_edges: int = 1     # edges executed before re-perception
    max_replans: int = 10000

    deepseek_api_key: str = "sk-c1ea9b080fc444ceb1f5fa7901e3b92f"
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-flash-vision-exp"
    deepseek_thinking: bool = True
    llm_max_tokens: int | None = None
    llm_timeout: float = 300.0
    llm_max_retries: int = 2

    rng_seed: int = 0
    out_dir: str = "img"
    save_frames: bool = True
    gif_fps: float = 10
    gif_end_hold_s: float = 2
    gif_dpi: int = 300
    # Sample animation frames on simulated time, capped by gif_max_frames.
    gif_time_step: float = 1.0
    gif_max_frames: int = 400
    verbose: bool = True

    def __post_init__(self):
        self.lambda_distance = validate_lambda(self.lambda_distance)
        self.time_importance = validate_time_importance(self.time_importance)
        self.strategy = validate_strategy(self.strategy)

    def free_profile(self) -> kinematics.MotionProfile:
        """Return the unloaded motion profile."""
        return kinematics.MotionProfile(
            self.robot_v_max, self.robot_a_max,
            self.robot_w_max, self.robot_alpha_max)

    def loaded_profile(self) -> kinematics.MotionProfile:
        """Return the loaded motion profile."""
        return kinematics.MotionProfile(
            self.robot_v_max_loaded, self.robot_a_max_loaded,
            self.robot_w_max_loaded, self.robot_alpha_max_loaded)

    def log(self, *args):
        if self.verbose:
            log.emit(" ".join(str(a) for a in args))
