"""Runtime configuration and validation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

import kinematics


class StrategyFlags(NamedTuple):
    llm_cost: bool
    llm_risk: bool
    shortest: bool
    llm_choice: bool


STRATEGIES: dict[str, StrategyFlags] = {
    # Primary CRA-NAMO method: LLM estimates both manipulation cost and risk.
    "cra-namo": StrategyFlags(True, True, False, False),
    "llm-cost-risk": StrategyFlags(True, True, False, False),
    "llm-cost": StrategyFlags(True, False, False, False),
    "llm-risk": StrategyFlags(False, True, False, False),
    "no-llm": StrategyFlags(False, False, False, False),
    "shortest": StrategyFlags(False, False, True, False),
    "llm-choice": StrategyFlags(True, True, False, True),
}
DEFAULT_STRATEGY = "shortest"


def validate_strategy(value: str) -> str:
    name = str(value).strip().lower().replace("_", "-")
    if name not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {value!r}; available: {', '.join(STRATEGIES)}")
    return name


REASONING_EFFORTS = ("low", "high", "max")
_EFFORT_ALIASES = {"medium": "high", "xhigh": "high"}


def validate_reasoning_effort(value: str) -> str:
    name = _EFFORT_ALIASES.get(str(value).strip().lower(),
                               str(value).strip().lower())
    if name not in REASONING_EFFORTS:
        raise ValueError(
            f"unknown reasoning effort {value!r}; available: "
            + ", ".join(REASONING_EFFORTS))
    return name

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

    robot_radius: float = 0.1

    # Seeded-random map generation. obstacle_count is the total number of
    # movable obstacles (decision, dynamic, and background obstacles combined).
    random_map_obstacle_count: int = 10
    # Number of independently moving obstacles on each generated map.
    random_map_dynamic_obstacle_count: int = 5
    # Number of consecutive maps produced by benchmarks/random_maps.py.
    random_map_experiment_count: int = 10
    # Whether the batch benchmark writes a PNG and GIF for each strategy run.
    random_map_generate_images: bool = True
    # Complete batch-experiment configuration.  The benchmark entry point reads
    # these values directly; no command-line switches are required.
    random_map_run_strategies: tuple[str, ...] = (
        "no-llm", "shortest", "cra-namo")
    random_map_timeout_seconds: float = 300
    random_map_resume: bool = True
    random_map_seed_start: int = 0
    random_map_output_dir: str = "img/random_experiments"

    # Motion limits for unloaded and loaded driving.
    robot_v_max: float = 0.6         # [m/s]
    robot_a_max: float = 0.5         # [m/s^2]
    robot_w_max: float = 1.5         # [rad/s]
    robot_alpha_max: float = 2.0     # [rad/s^2]
    robot_v_max_loaded: float = 0.25
    robot_a_max_loaded: float = 0.2
    robot_w_max_loaded: float = 0.6
    robot_alpha_max_loaded: float = 0.8

    # Objective: C = (1 - w)J + w(time_value * T) + R.
    time_importance: float = 0.0     # w in [0, 1]
    time_value: float = 100.0        # [J/s]

    lambda_distance: float = 350.0   # equivalent driving resistance [N]

    risk_weight: float = 1.0
    # Risks at or above this level are forbidden when the value is non-empty.
    risk_forbidden_level: str = "extreme"
    # A contact force above this ratio raises the fallback risk by one level.
    risk_heavy_ratio: float = 3.0

    R_perc: float = 10.0             # perception radius [m]
    sight_width: float = 0.1         # line-of-sight width [m]

    R_manip: float = 5.0             # relocation search radius [m]
    # Soft preference for forward drop poses; zero disables it.
    manip_forward_penalty: float = 2.0
    # Score candidate drop poses using the remaining robot route.
    manip_lookahead: bool = True
    manip_max_frames_per_action: int = 30
    # Penalty, in metres of obstacle travel, for reducing roadmap clearance.
    manip_blocked_edge_penalty_m: float = 0.0

    contact_required: bool = True
    contact_station_spacing: float = 0.3   # perimeter sample spacing [m]
    contact_max_slide: float = 0.6         # maximum grip slide per step [m]
    contact_clearance: float = 0.01        # contact tolerance [m]
    # Replan when measured force differs from the planned value by this ratio.
    contact_replan_ratio: float = 1.25
    # Maximum applied force relative to friction; zero disables the limit.
    contact_max_force_ratio: float = 1.0
    # These limits model physical feasibility; zero disables either limit.
    robot_max_push_force: float = 200_000.0
    robot_push_height: float = 0.15
    push_friction_mu: float = 0.5

    se2_cell: float = 0.15
    se2_n_theta: int = 12
    se2_connectivity: int = 8
    se2_containment: str = "centroid"
    se2_rot_weight: float | None = None
    se2_goal_candidates: int = 24
    # Widen the candidate shortlist when every initial drop pose is rejected.
    se2_goal_widen: int = 1

    # World motion uses the same rotation-folded distance units as path cost.
    dynamic_speed: float = 0.3          # default obstacle travel speed [m/s]
    dynamic_step: float = 0.25          # simulated seconds per motion sub-step
    dynamic_block_patience: float = 2.0  # seconds waiting before seeking another route
    dynamic_give_up: float = 30.0       # seconds stuck before it parks where it is
    dynamic_replan_backoff: float = 2.0  # seconds before re-asking for a route that was not there

    dynamic_wait_step: float = 2.0      # seconds the robot waits per unplannable cycle
    dynamic_max_wait: float = 90.0      # total waiting before the way counts as shut

    grid_step: float = 0.3          # roadmap node spacing [m]
    conn_radius: float = 0.6        # roadmap connection radius [m]

    strategy: str = DEFAULT_STRATEGY
    use_llm_ordering: bool = True
    max_expansions: int = 100000

    step_execute_edges: int = 1     # edges executed before re-perception
    max_replans: int = 10000

    deepseek_api_key: str = "sk-4bb1a"
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-flash-vision-exp"
    deepseek_thinking: bool = True
    deepseek_reasoning_effort: str = "low"
    llm_max_tokens: int | None = None
    llm_timeout: float = 300.0
    llm_max_retries: int = 2
    perception_llm_timeout: float = 60.0
    perception_llm_max_calls: int = 8

    llm_choice_max_options: int = 10
    llm_choice_reuse_decision: bool = True

    out_dir: str = "img"
    save_frames: bool = True
    # Viewer-only: draw every real obstacle in animation frames even before
    # the robot perceives it. This does not change the planner's belief state.
    show_global_obstacles: bool = False
    gif_fps: float = 10
    gif_end_hold_s: float = 2
    gif_dpi: int = 300
    gif_time_step: float = 1.0
    gif_max_frames: int = 400
    verbose: bool = True
    _log_sink: Callable[[str], None] = field(default=print, repr=False)
    _log_flush: Callable[[], None] = field(default=lambda: None, repr=False)

    def __post_init__(self):
        if self.random_map_obstacle_count < 4:
            raise ValueError(
                "random_map_obstacle_count must be at least 4")
        if self.random_map_dynamic_obstacle_count < 0:
            raise ValueError(
                "random_map_dynamic_obstacle_count must be non-negative")
        if (self.random_map_obstacle_count
                < 4 + self.random_map_dynamic_obstacle_count):
            raise ValueError(
                "random_map_obstacle_count must leave room for the four "
                "decision obstacles and configured dynamic obstacle")
        if self.random_map_experiment_count < 1:
            raise ValueError(
                "random_map_experiment_count must be positive")
        if not self.random_map_run_strategies:
            raise ValueError("random_map_run_strategies must not be empty")
        self.random_map_run_strategies = tuple(
            validate_strategy(value) for value in self.random_map_run_strategies)
        if len(set(self.random_map_run_strategies)) != len(
                self.random_map_run_strategies):
            raise ValueError("random_map_run_strategies must not contain duplicates")
        if self.random_map_timeout_seconds <= 0:
            raise ValueError("random_map_timeout_seconds must be positive")
        self.random_map_seed_start = int(self.random_map_seed_start)
        if not str(self.random_map_output_dir).strip():
            raise ValueError("random_map_output_dir must not be empty")
        self.lambda_distance = validate_lambda(self.lambda_distance)
        self.time_importance = validate_time_importance(self.time_importance)
        self.strategy = validate_strategy(self.strategy)
        self.deepseek_reasoning_effort = validate_reasoning_effort(
            self.deepseek_reasoning_effort)

    @property
    def use_llm_cost(self) -> bool:
        return STRATEGIES[self.strategy].llm_cost

    @property
    def use_llm_risk(self) -> bool:
        return STRATEGIES[self.strategy].llm_risk

    @property
    def shortest_path_mode(self) -> bool:
        return STRATEGIES[self.strategy].shortest

    @property
    def llm_choice(self) -> bool:
        return STRATEGIES[self.strategy].llm_choice

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
