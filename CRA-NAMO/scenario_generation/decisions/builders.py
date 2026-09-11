"""Place semantic blockers before incidental background clutter."""

from __future__ import annotations

from dataclasses import dataclass

from obstacle import MovableObstacle
from scenarios._realism import MU_CASTORS, MU_WOOD, push_force
from scenario_generation.models import DecisionSpec, TopologyResult


@dataclass
class OidAllocator:
    next_oid: int = 1000

    def take(self) -> int:
        oid = self.next_oid
        self.next_oid += 1
        return oid


def _blocker(gate: dict, pose, oid: int, *, material: str,
             density: float, mu: float, h: float,
             contact_reveals: str = "") -> MovableObstacle:
    l = gate["door_height"] - 0.22
    d = gate["wall_thickness"] + 0.22
    return MovableObstacle(
        x=pose[0], y=pose[1], l=l, d=d, h=h, theta=pose[2],
        material=material,
        difficulty=push_force(density * l * d * h, mu),
        contact_reveals=contact_reveals, oid=oid)


def build_gate_decisions(topology: TopologyResult, allocator: OidAllocator,
                         rng, *, include_hidden: bool = True):
    """Build three qualitatively different choices on successive gates."""
    gates = topology.anchors["gates"]
    obstacles: list[MovableObstacle] = []
    decisions: list[DecisionSpec] = []

    # Gate 0: a cheap direct push competes with a geometrical detour.
    gate = gates[0]
    oid = allocator.take()
    obstacles.append(_blocker(
        gate, gate["direct"], oid, material="empty_cart",
        density=rng.uniform(35.0, 60.0), mu=MU_CASTORS, h=0.9))
    decisions.append(DecisionSpec(
        name="gate_0_move_or_detour", kind="move_or_detour",
        involved_oids=[oid], alternatives=["move", "detour"],
        expected_tradeoff="short direct push versus a longer open route",
        metadata={"obstacles": [oid]}))

    # Gate 1: both doors are blocked; the direct object carries semantic risk.
    gate = gates[1]
    risky_oid, safe_oid = allocator.take(), allocator.take()
    obstacles.extend([
        _blocker(gate, gate["direct"], risky_oid,
                 material="shoring_prop", density=310.0,
                 mu=MU_WOOD, h=1.25),
        _blocker(gate, gate["bypass"], safe_oid,
                 material="empty_cart", density=45.0,
                 mu=MU_CASTORS, h=0.9),
    ])
    decisions.append(DecisionSpec(
        name="gate_1_risk_or_distance", kind="risk_or_distance",
        involved_oids=[risky_oid, safe_oid],
        alternatives=["risky_short_route", "safe_bypass"],
        expected_tradeoff="short risky push versus safe extra travel",
        metadata={"risky": risky_oid, "safer_alternative": safe_oid}))

    # Gate 2: visually ordinary cartons may reveal a much heavier load.
    gate = gates[2]
    hidden_oid = allocator.take()
    density = rng.uniform(360.0, 460.0) if include_hidden else rng.uniform(35.0, 55.0)
    obstacles.append(_blocker(
        gate, gate["direct"], hidden_oid, material="cardboard_box",
        density=density, mu=MU_WOOD, h=1.05,
        contact_reveals="cartons_of_books" if include_hidden else ""))
    hidden_kind = "hidden_difficulty" if include_hidden else "move_or_detour"
    decisions.append(DecisionSpec(
        name=("gate_2_hidden_difficulty" if include_hidden
              else "gate_2_move_or_detour"), kind=hidden_kind,
        involved_oids=[hidden_oid], alternatives=["try_push", "detour"],
        expected_tradeoff="visual estimate followed by contact-driven replanning",
        metadata={"obstacles": [hidden_oid]}))

    return obstacles, decisions
