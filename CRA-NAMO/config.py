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
    # --- Robot ---
    robot_radius: float = 0.2    # robot radius

    # --- Robot kinematics ---
    # What the robot can physically do, which is what turns a route into a
    # duration. The model is turn-in-place-then-drive (see `kinematics`), so the
    # angular limits are as load-bearing as the linear ones: a route made of many
    # short segments at sharp angles can easily spend more time turning than
    # driving, and that is the whole reason a time-aware search picks different
    # routes from a distance-aware one.
    robot_v_max: float = 0.6         # [m/s]
    robot_a_max: float = 0.5         # [m/s^2]
    robot_w_max: float = 1.5         # [rad/s]
    robot_alpha_max: float = 2.0     # [rad/s^2]
    # Holding an obstacle, the robot is slower on every axis: it is pushing a
    # loaded mass through floor friction, and it has to keep its grip. These are
    # the limits that apply from grip to release.
    robot_v_max_loaded: float = 0.25
    robot_a_max_loaded: float = 0.2
    robot_w_max_loaded: float = 0.6
    robot_alpha_max_loaded: float = 0.8

    # --- Time vs. energy: C = (1 - w) * J + w * (time_value * T) + R ---
    # w = 0 recovers the pure-energy objective exactly, and is the default: every
    # result predating the time axis still reproduces.
    #
    # `time_value` is what converts seconds into the joules J is measured in.
    # Adding a raw count of seconds to a five-figure joule total would leave w
    # doing nothing until it was within a whisker of 1 — with these maps J is
    # ~1e4 and T is ~1e2, so the honest exchange rate is ~1e2 J/s. Read it as the
    # power the robot burns just by being switched on: idling for a second costs
    # `time_value` joules, so time and work are traded at a rate with units,
    # rather than by adding a length to a mass. Set it to 1.0 for the literal
    # C = (1-w)J + wT.
    time_importance: float = 0.0     # w in [0, 1]
    time_value: float = 100.0        # [J/s], i.e. watts

    # --- Cost function J = lambda * D + W ---
    # Obstacle difficulty is a real friction force f = mu*rho*V*g [N], so W is in
    # joules and lambda_distance is the equivalent driving resistance [N] the robot
    # pays per metre travelled. It is calibrated (not the robot's bare rolling
    # resistance) so that the detour-vs-move trade-off stays in a regime where
    # moving light obstacles is still worthwhile.
    lambda_distance: float = 350.0   # equivalent driving resistance [N]

    # --- Risk ---
    risk_weight: float = 1.0

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
    se2_cell: float = 0.15
    se2_n_theta: int = 12
    se2_connectivity: int = 8
    se2_containment: str = "centroid"
    se2_rot_weight: float | None = None

    # --- Roadmap ---
    grid_step: float = 0.3    # roadmap node grid spacing
    conn_radius: float = 0.6   # roadmap node connection radius

    # --- Search ---
    strategy: str = "normal"    # "normal" | "shortest"
    use_llm_ordering: bool = True
    max_expansions: int = 100000

    # --- Online loop ---
    step_execute_edges: int = 1     # re-perception frequency
    max_replans: int = 10000

    # --- LLM ---
    deepseek_api_key: str = "sk-c1ea9b080fc444ceb1f5fa7901e3b92f"
    deepseek_base_url: str = "https://api.deepseek.com/chat/completions"
    deepseek_model: str = "deepseek-v4-flash-vision-exp"
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
    # The animation runs on the simulated clock, not on the event list: one frame
    # per `gif_time_step` seconds, so a slow loaded drag takes visibly longer on
    # screen than the same distance driven free. `gif_max_frames` stretches the
    # step when a run is long enough that honouring it would produce a GIF nobody
    # can open.
    gif_time_step: float = 1.0   # simulated seconds per animation frame
    gif_max_frames: int = 400
    verbose: bool = True

    def __post_init__(self):
        self.lambda_distance = validate_lambda(self.lambda_distance)
        self.time_importance = validate_time_importance(self.time_importance)
        # a stale "easiest" would otherwise be silently treated as "normal"
        self.strategy = validate_strategy(self.strategy)

    # --- kinematics ---
    def free_profile(self) -> kinematics.MotionProfile:
        """What the robot can do driving on its own."""
        return kinematics.MotionProfile(
            self.robot_v_max, self.robot_a_max,
            self.robot_w_max, self.robot_alpha_max)

    def loaded_profile(self) -> kinematics.MotionProfile:
        """What it can do from the moment it grips an obstacle until it lets go."""
        return kinematics.MotionProfile(
            self.robot_v_max_loaded, self.robot_a_max_loaded,
            self.robot_w_max_loaded, self.robot_alpha_max_loaded)

    def log(self, *args):
        if self.verbose:
            log.emit(" ".join(str(a) for a in args))

