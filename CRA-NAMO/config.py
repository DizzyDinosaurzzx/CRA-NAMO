"""Runtime configuration and validation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import kinematics


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

    # World motion. Obstacles may travel under their own steam; speed is in the
    # rotation-folded units of cost.se2_path_length, so one number covers both
    # sliding and turning. Scenarios without events never touch any of this.
    dynamic_speed: float = 0.3          # default obstacle travel speed [m/s]
    dynamic_step: float = 0.25          # simulated seconds per motion sub-step
    dynamic_block_patience: float = 2.0  # seconds waiting before seeking another route
    dynamic_give_up: float = 30.0       # seconds stuck before it parks where it is
    # Planning takes real seconds and by default they are spent on the shared
    # clock like any others. With a world that moves, that makes a run depend on
    # how fast the machine planning it happens to be — a loaded machine gives the
    # obstacles longer to travel. Set False to take planning off the clock: it is
    # still measured and reported, the world then advances only with the robot,
    # and a dynamic run repeats exactly.
    plan_time_in_clock: bool = True

    dynamic_wait_step: float = 2.0      # seconds the robot waits per unplannable cycle
    dynamic_max_wait: float = 90.0      # total waiting before the way counts as shut

    grid_step: float = 0.3          # roadmap node spacing [m]
    conn_radius: float = 0.6        # roadmap connection radius [m]

    strategy: str = "normal"    # "normal" | "shortest"
    use_llm_ordering: bool = True
    max_expansions: int = 100000

    step_execute_edges: int = 1     # edges executed before re-perception
    max_replans: int = 10000

    deepseek_api_key: str = ""
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
    _log_sink: Callable[[str], None] = field(default=print, repr=False)
    _log_flush: Callable[[], None] = field(default=lambda: None, repr=False)

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
            self._log_sink(" ".join(str(a) for a in args))

    def set_logger(self, sink: Callable[[str], None],
                   flush: Callable[[], None]) -> None:
        """Set the console logger used by planning and execution."""
        self._log_sink = sink
        self._log_flush = flush

    def flush_log(self) -> None:
        self._log_flush()
