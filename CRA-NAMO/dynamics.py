"""Simulate obstacle motion and state changes driven by scenario events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

import geometry
import manipulation
from config import Config
from obstacle import MovableObstacle

Pose = Tuple[float, float, float]
XY = Tuple[float, float]

_ARRIVED_EPS = 1e-6


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


@dataclass
class MoveTo:
    """Send an obstacle toward a target pose."""
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
    """Change selected obstacle properties in place."""
    oid: int
    material: Optional[str] = None
    l: Optional[float] = None
    d: Optional[float] = None
    h: Optional[float] = None
    difficulty: Optional[float] = None
    contact_reveals: Optional[str] = None

    def __post_init__(self):
        for name in ("l", "d", "h"):
            value = getattr(self, name)
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"Mutate {name} must be positive")
        if self.difficulty is not None and float(self.difficulty) < 0.0:
            raise ValueError("Mutate difficulty must be non-negative")

    def apply(self, dyn: "WorldDynamics") -> str:
        """Apply the mutation, rejecting size growth into occupied space."""
        obs = dyn.obstacle(self.oid)
        if obs is None:
            return f"obstacle {self.oid} is not in the world"
        fields = ["material", "l", "d", "h", "difficulty", "contact_reveals"]
        refused = ""
        if self.l is not None or self.d is not None:
            l = obs.l if self.l is None else float(self.l)
            d = obs.d if self.d is None else float(self.d)
            refused = dyn.room_to_grow(obs, l, d)
            if refused:
                fields = [f for f in fields if f not in ("l", "d")]
        changed = []
        for name in fields:
            value = getattr(self, name)
            if value is None or getattr(obs, name) == value:
                continue
            setattr(obs, name, value)
            changed.append(name)
        if changed:
            dyn.mark_stale(self.oid)
        what = (f"obstacle {self.oid} changes {', '.join(changed)}" if changed
                else f"obstacle {self.oid} unchanged")
        return what if not refused else f"{what} (cannot grow into {refused})"


@dataclass
class Event:
    """One trigger, one effect, fired at most once."""
    trigger: object
    effect: object
    name: str = ""
    fired: bool = False


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
    retry_at: float = -math.inf      # clock before which re-planning is pointless
    tried_version: int = -1          # world version the last failed attempt saw


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
        self.log: List[Tuple[float, str]] = []
        self.version = 0
        self.clock = 0.0
        self.robot_xy: XY = (0.0, 0.0)
        self.stale: set = set()
        bounds = workspace.bounds
        self._bounds = (bounds[0], bounds[2], bounds[1], bounds[3])

    @property
    def active(self) -> bool:
        """Return whether events or moving actors need simulation."""
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
        self.stale.add(oid)
        actor = self.actors.get(oid)
        if actor is not None:
            actor.replan = True
            actor.retry_at = -math.inf

    def drain_stale(self) -> set:
        """Which obstacles have changed since this was last asked."""
        changed, self.stale = self.stale, set()
        return changed

    def room_to_grow(self, obs: MovableObstacle, l: float, d: float) -> str:
        """Return the first object that would block a proposed size increase."""
        grown = (Polygon(geometry.rect_corners(obs.x, obs.y, l, d, obs.theta))
                 .difference(obs.polygon))
        if grown.is_empty or grown.area <= geometry.CONTACT_AREA_EPS:
            return ""
        xmin, xmax, ymin, ymax = self._bounds
        if not box(xmin, ymin, xmax, ymax).contains(grown):
            return "the world's edge"
        for so in self.static_obstacles:
            if grown.intersection(so.polygon).area > geometry.CONTACT_AREA_EPS:
                return f"wall {so.name}"
        for w in self.world:
            if w.oid == obs.oid:
                continue
            if grown.intersection(w.polygon).area > geometry.CONTACT_AREA_EPS:
                return f"obstacle {w.oid}"
        robot = Point(self.robot_xy).buffer(self.cfg.robot_radius)
        if grown.intersection(robot).area > geometry.CONTACT_AREA_EPS:
            return "the robot"
        return ""

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
        actor.replan = True
        actor.blocked_for = 0.0
        actor.retry_at = -math.inf

    def advance(self, seconds: float, clock: float,
                robot_xy: XY) -> List[Tuple[float, str]]:
        """Advance world time in bounded steps and return event log entries."""
        if not self.active or seconds <= 0.0:
            return []
        self.robot_xy = robot_xy
        step = max(self.cfg.dynamic_step, 1e-3)
        notes: List[Tuple[float, str]] = []
        t = clock
        left = seconds
        while left > 1e-9:
            self.clock = t
            notes += self._fire_events(t, robot_xy)
            dt = min(step, left, max(self._until_next_time_trigger(t), 1e-6))
            if self._step(dt, t, robot_xy):
                self.version += 1
            t += dt
            left -= dt
        self.clock = t
        notes += self._fire_events(t, robot_xy)
        self.log += notes
        return notes

    def _fire_events(self, clock: float, robot_xy: XY) -> List[Tuple[float, str]]:
        notes = []
        for ev in self.events:
            if ev.fired or not ev.trigger.ready(clock, robot_xy,
                                                self.moved_by_robot):
                continue
            ev.fired = True
            what = ev.effect.apply(self)
            notes.append((clock, f"{ev.name or 'event'}: {what}"))
            self.version += 1
        return notes

    def _until_next_time_trigger(self, clock: float) -> float:
        """Return the time until the next unfired timed event."""
        gap = math.inf
        for ev in self.events:
            if not ev.fired and isinstance(ev.trigger, AtTime):
                gap = min(gap, ev.trigger.t - clock)
        return gap

    def _step(self, dt: float, clock: float, robot_xy: XY) -> bool:
        moved = False
        for actor in list(self.actors.values()):
            if actor.arrived or actor.suspended:
                continue
            if self._advance_actor(actor, dt, clock, robot_xy):
                moved = True
        return moved

    def _blockers(self, oid: int, robot_xy: XY):
        """Return dynamic blockers for an actor's next route segment."""
        polys = [w.polygon for w in self.world if w.oid != oid]
        polys.append(Point(robot_xy).buffer(self.cfg.robot_radius))
        return [(p, p.bounds) for p in polys]

    def _others_union(self, oid: int, robot_xy: Optional[XY] = None):
        """Return the union of obstacles that an actor must avoid."""
        polys = [w.polygon for w in self.world if w.oid != oid]
        if robot_xy is not None:
            polys.append(Point(robot_xy).buffer(self.cfg.robot_radius))
        return unary_union(polys) if polys else None

    def _advance_actor(self, actor: Actor, dt: float, clock: float,
                       robot_xy: XY) -> bool:
        obs = self.obstacle(actor.oid)
        if obs is None:
            actor.arrived = True
            return False
        if actor.replan or not actor.path:
            if not self._may_replan(actor, clock):
                actor.blocked_for += dt
                actor.waited += dt
                self._give_up_if_stuck(actor)
                return False
            actor.path = manipulation.plan_route_se2(
                obs, actor.goal, self.static_obstacles, self._bounds, self.cfg,
                others_polys=self._others_union(
                    actor.oid, robot_xy if actor.avoid_robot else None))
            actor.leg = 0
            actor.replan = False
            actor.avoid_robot = False
            if not actor.path:
                # Retry after the world changes or the back-off interval expires.
                actor.retry_at = clock + max(self.cfg.dynamic_replan_backoff, 0.0)
                actor.tried_version = self.version
                actor.blocked_for += dt
                actor.waited += dt
                self._give_up_if_stuck(actor)
                return False
            actor.retry_at = -math.inf
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
                # Wait before replanning around the blocker.
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

    def _may_replan(self, actor: Actor, clock: float) -> bool:
        """Return whether the actor's route may be planned again."""
        return (actor.retry_at == -math.inf
                or actor.tried_version != self.version
                or clock >= actor.retry_at)

    def _give_up_if_stuck(self, actor: Actor):
        """Park an actor that has exceeded the configured wait limit."""
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


# Convenience constructors for scenario files.
def at_time(t: float) -> AtTime:
    return AtTime(float(t))


def after_moved(oid: int) -> AfterMoved:
    return AfterMoved(int(oid))


def near_point(at: XY, radius: float = 1.0) -> NearPoint:
    return NearPoint((float(at[0]), float(at[1])), float(radius))
