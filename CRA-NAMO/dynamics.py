"""Drive the world's own motion: obstacles that move and change on their own.

A scenario may hand the simulator a list of `Event`s. Each pairs a *trigger* —
a moment on the clock, the robot moving some obstacle, the robot arriving
somewhere — with an *effect*: send an obstacle off towards a pose of its own, or
change what it is made of, how big it is, how hard it is to shift.

Everything in here reads and writes ground truth only. The robot is told
nothing: it finds out that something has moved or changed the same way it finds
out anything else, by looking at it or by running into it. Keeping that
separation is the point of the module boundary — `Belief` never imports this.

An obstacle under way follows an SE(2) route planned for it in real time. If
something gets in the way it stops and re-plans on the next tick, so a route is
never trusted for longer than one step. Being picked up by the robot suspends
it: for as long as the robot has hold of it, it is an ordinary movable obstacle
sitting wherever the robot has put it. On release it sets off again from there,
along a route worked out from the pose it was dropped at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Point
from shapely.ops import unary_union

import geometry
import manipulation
from config import Config
from obstacle import MovableObstacle

Pose = Tuple[float, float, float]
XY = Tuple[float, float]

_ARRIVED_EPS = 1e-6


# --- triggers ---
@dataclass
class AtTime:
    """Fire once the simulated clock passes `t` seconds."""
    t: float

    def ready(self, clock: float, robot_xy: XY, moved: set) -> bool:
        return clock >= self.t


@dataclass
class AfterMoved:
    """Fire once the robot has moved obstacle `oid`."""
    oid: int

    def ready(self, clock: float, robot_xy: XY, moved: set) -> bool:
        return self.oid in moved


@dataclass
class NearPoint:
    """Fire once the robot comes within `radius` of `at`."""
    at: XY
    radius: float = 1.0

    def ready(self, clock: float, robot_xy: XY, moved: set) -> bool:
        return math.dist(robot_xy, self.at) <= self.radius


# --- effects ---
@dataclass
class MoveTo:
    """Send an obstacle off towards a pose of its own choosing.

    `speed` is measured in the same units `cost.se2_path_length` uses, so an
    angle counts as the distance its mean radius sweeps and one number covers
    both halves of the motion. None takes `cfg.dynamic_speed`.
    """
    oid: int
    goal: Pose
    speed: Optional[float] = None

    def apply(self, dyn: "WorldDynamics") -> str:
        dyn.send(self.oid, self.goal, self.speed)
        return (f"obstacle {self.oid} sets off for "
                f"({self.goal[0]:,.1f}, {self.goal[1]:,.1f})")


@dataclass
class Halt:
    """Stop an obstacle wherever it has got to."""
    oid: int

    def apply(self, dyn: "WorldDynamics") -> str:
        dyn.stop(self.oid)
        return f"obstacle {self.oid} stops"


@dataclass
class Mutate:
    """Change what an obstacle is, in place.

    Anything left None is left alone. `l`, `d`, `h` and `material` are there to
    be seen and the robot picks them up the moment it looks; `difficulty` and
    `contact_reveals` are not, and cost it a fresh touch to learn.
    """
    oid: int
    material: Optional[str] = None
    l: Optional[float] = None
    d: Optional[float] = None
    h: Optional[float] = None
    difficulty: Optional[float] = None
    contact_reveals: Optional[str] = None

    def apply(self, dyn: "WorldDynamics") -> str:
        obs = dyn.obstacle(self.oid)
        if obs is None:
            return f"obstacle {self.oid} is not in the world"
        changed = []
        for name in ("material", "l", "d", "h", "difficulty", "contact_reveals"):
            value = getattr(self, name)
            if value is None or getattr(obs, name) == value:
                continue
            setattr(obs, name, value)
            changed.append(name)
        if changed:
            dyn.mark_stale(self.oid)
        return (f"obstacle {self.oid} changes {', '.join(changed)}" if changed
                else f"obstacle {self.oid} unchanged")


@dataclass
class Event:
    """One trigger, one effect, fired at most once."""
    trigger: object
    effect: object
    name: str = ""
    fired: bool = False


# --- an obstacle under way ---
@dataclass
class Actor:
    oid: int
    goal: Pose
    speed: float
    path: Optional[List[Pose]] = None
    leg: int = 0                     # index of the path pose already passed
    replan: bool = True              # route is stale and must be worked out again
    suspended: bool = False          # the robot has hold of it
    arrived: bool = False
    blocked_for: float = 0.0         # seconds since it last sought another route
    waited: float = 0.0              # seconds since it last made any progress
    avoid_robot: bool = False        # waited long enough to route around it


class WorldDynamics:
    """Advance the world's own motion alongside the robot's."""

    def __init__(self, world: List[MovableObstacle], static_obstacles,
                 workspace, events: Optional[Sequence[Event]], cfg: Config):
        self.world = world
        self.static_obstacles = static_obstacles
        self.cfg = cfg
        self.events: List[Event] = list(events or ())
        self.actors: Dict[int, Actor] = {}
        self.moved_by_robot: set = set()
        self.moved_on_own: set = set()
        self.log: List[Tuple[float, str]] = []      # (clock, what happened)
        self.version = 0            # bumped whenever ground truth changes
        bounds = workspace.bounds
        self._bounds = (bounds[0], bounds[2], bounds[1], bounds[3])

    @property
    def active(self) -> bool:
        """Does this scenario have any world motion at all?

        A scenario without events is left bit-for-bit as it was before the world
        could move: nothing is stepped, nothing is re-planned, no extra frames.
        """
        return bool(self.events) or bool(self.actors)

    @property
    def moving(self) -> set:
        """Obstacles actually under way this instant."""
        return {oid for oid, a in self.actors.items()
                if not (a.arrived or a.suspended) and a.path}

    def obstacle(self, oid: int) -> Optional[MovableObstacle]:
        for w in self.world:
            if w.oid == oid:
                return w
        return None

    def mark_stale(self, oid: int):
        """Ground truth for this obstacle changed; any route it was on is void."""
        self.version += 1
        actor = self.actors.get(oid)
        if actor is not None:
            actor.replan = True

    # --- what the executor tells it ---
    def send(self, oid: int, goal: Pose, speed: Optional[float] = None):
        """Give an obstacle somewhere to be."""
        if self.obstacle(oid) is None:
            return
        self.actors[oid] = Actor(oid, tuple(goal),
                                 float(self.cfg.dynamic_speed if speed is None
                                       else speed))

    def stop(self, oid: int):
        self.actors.pop(oid, None)

    def note_moved(self, oid: int):
        """The robot has moved this obstacle; `AfterMoved` triggers may fire."""
        self.moved_by_robot.add(oid)

    def suspend(self, oid: int):
        """The robot has taken hold of it: the script is off until it lets go."""
        actor = self.actors.get(oid)
        if actor is not None:
            actor.suspended = True

    def release(self, oid: int):
        """The robot has let go: carry on from wherever it was put down."""
        actor = self.actors.get(oid)
        if actor is None:
            return
        actor.suspended = False
        actor.replan = True          # the route it was on started somewhere else
        actor.blocked_for = 0.0

    # --- the clock ---
    def advance(self, seconds: float, clock: float, robot_xy: XY) -> List[str]:
        """Let `seconds` pass in the world. Returns what happened, for the log."""
        if not self.active or seconds <= 0.0:
            return []
        notes = self._fire_events(clock, robot_xy)
        step = max(self.cfg.dynamic_step, 1e-3)
        left = seconds
        while left > 1e-9:
            dt = min(step, left)
            left -= dt
            if self._step(dt, robot_xy):
                self.version += 1
        for note in notes:
            self.log.append((clock, note))
        return notes

    def _fire_events(self, clock: float, robot_xy: XY) -> List[str]:
        notes = []
        for ev in self.events:
            if ev.fired or not ev.trigger.ready(clock, robot_xy,
                                                self.moved_by_robot):
                continue
            ev.fired = True
            what = ev.effect.apply(self)
            notes.append(f"{ev.name or 'event'}: {what}")
            self.version += 1
        return notes

    # --- motion ---
    def _step(self, dt: float, robot_xy: XY) -> bool:
        moved = False
        for actor in list(self.actors.values()):
            if actor.arrived or actor.suspended:
                continue
            if self._advance_actor(actor, dt, robot_xy):
                moved = True
        return moved

    def _blockers(self, oid: int, robot_xy: XY):
        """What the route could not have accounted for.

        Only the robot and the other movable bodies. The walls are deliberately
        left out: the route came from a planner that has them in its C-space and
        keeps its own margin from them, and re-testing the same route against
        them here with a swept-area test of a different tolerance does not make
        it safer — it makes a body threading a doorway refuse its own validated
        route a few centimetres in, and stand in the door for the rest of the run
        arguing with the planner that put it there.
        """
        polys = [w.polygon for w in self.world if w.oid != oid]
        polys.append(Point(robot_xy).buffer(self.cfg.robot_radius))
        return [(p, p.bounds) for p in polys]

    def _others_union(self, oid: int, robot_xy: Optional[XY] = None):
        """What a route for this obstacle has to steer around.

        The robot is left out normally: it is about to move on, and putting it in
        would give every step of its journey a C-space of its own. It goes in
        only for a body that has waited for it and given up, which is the one
        case where a route that ignores it is no use.
        """
        polys = [w.polygon for w in self.world if w.oid != oid]
        if robot_xy is not None:
            polys.append(Point(robot_xy).buffer(self.cfg.robot_radius))
        return unary_union(polys) if polys else None

    def _advance_actor(self, actor: Actor, dt: float, robot_xy: XY) -> bool:
        obs = self.obstacle(actor.oid)
        if obs is None:
            actor.arrived = True
            return False
        if actor.replan or not actor.path:
            actor.path = manipulation.plan_route_se2(
                obs, actor.goal, self.static_obstacles, self._bounds, self.cfg,
                others_polys=self._others_union(
                    actor.oid, robot_xy if actor.avoid_robot else None))
            actor.leg = 0
            actor.replan = False
            actor.avoid_robot = False
            if not actor.path:
                # nowhere to go from here; try again once the world has changed
                actor.blocked_for += dt
                actor.waited += dt
                self._give_up_if_stuck(actor)
                return False
            actor.blocked_for = 0.0

        budget = actor.speed * dt
        blockers = self._blockers(actor.oid, robot_xy)
        rot_w = self._rot_weight(obs)
        pose: Pose = (obs.x, obs.y, obs.theta)
        moved = False
        while budget > _ARRIVED_EPS and actor.leg + 1 < len(actor.path):
            nxt = actor.path[actor.leg + 1]
            span = self._leg_cost(pose, nxt, rot_w)
            if span <= _ARRIVED_EPS:
                actor.leg += 1
                continue
            if span <= budget:
                candidate, done_leg = nxt, True
            else:
                candidate, done_leg = self._lerp(pose, nxt, budget / span), False
            if not manipulation.path_is_clear_against(obs, [pose, candidate],
                                                     blockers):
                # Something is in the way. Wait where it is — the robot it is
                # most often waiting for will move on — and only go looking for
                # another route once the wait has gone on long enough.
                actor.blocked_for += dt
                actor.waited += dt
                if actor.blocked_for >= self.cfg.dynamic_block_patience:
                    actor.replan = True
                    actor.avoid_robot = True
                    actor.blocked_for = 0.0
                self._give_up_if_stuck(actor)
                break
            budget -= span if done_leg else budget
            pose = candidate
            moved = True
            if done_leg:
                actor.leg += 1

        if moved:
            obs.x, obs.y, obs.theta = pose
            self.moved_on_own.add(actor.oid)
            actor.waited = 0.0
            if actor.leg + 1 >= len(actor.path):
                actor.arrived = True
        return moved

    def _give_up_if_stuck(self, actor: Actor):
        """Stop trying, rather than spin on a route that will never open.

        A body that has stood still for this long is not going to get where it
        was going, and a simulation is better off with it parked than with it
        re-planning the same blocked route for the rest of the run.
        """
        if actor.waited < self.cfg.dynamic_give_up:
            return
        actor.arrived = True
        self.cfg.log(f"[dynamics] oid={actor.oid} gives up on "
                     f"({actor.goal[0]:,.1f}, {actor.goal[1]:,.1f})")

    def _rot_weight(self, obs: MovableObstacle) -> float:
        return (geometry.mean_rotation_radius(obs.l, obs.d)
                if self.cfg.se2_rot_weight is None
                else float(self.cfg.se2_rot_weight))

    @staticmethod
    def _leg_cost(a: Pose, b: Pose, rot_weight: float) -> float:
        return (math.hypot(b[0] - a[0], b[1] - a[1])
                + rot_weight * abs(geometry.wrap_dtheta(a[2], b[2])))

    @staticmethod
    def _lerp(a: Pose, b: Pose, s: float) -> Pose:
        return (a[0] + (b[0] - a[0]) * s,
                a[1] + (b[1] - a[1]) * s,
                a[2] + geometry.wrap_dtheta(a[2], b[2]) * s)


# --- shorthands for scenario files ---
def at_time(t: float) -> AtTime:
    return AtTime(float(t))


def after_moved(oid: int) -> AfterMoved:
    return AfterMoved(int(oid))


def near_point(at: XY, radius: float = 1.0) -> NearPoint:
    return NearPoint((float(at[0]), float(at[1])), float(radius))
