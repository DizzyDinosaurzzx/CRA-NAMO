"""Place semantic blockers before incidental background clutter.

Each gate on the robot's shortest route is turned into one choice.  The kinds
below differ in what the robot has to work out, not in how the geometry looks:

``move_or_detour``
    The plain baseline.  A cheap, honestly-labelled object against a detour.

``risk_or_distance``
    Both doors blocked.  The direct one carries a label the keyword table
    already reads as dangerous, so every arm avoids it; the choice is between
    the safe push and the longer way round.

``hidden_difficulty``
    An honest label over a dishonest weight.  Nobody can see it coming; the
    point is whether the planner replans once contact reveals the truth.

``blind_risk``
    The label reads as ordinary to the keyword table and as dangerous to anyone
    who knows the domain.  The push is priced to be *slightly* cheaper than the
    detour, so an arm that cannot read the label takes it, makes contact, and is
    charged the surcharge the label was warning about.

``blind_weight``
    The label reads light and the object is not.  The offline table prices the
    push from the label and commits; the world charges the real newtons.

``risk_or_risk``
    Both doors blocked by something worth avoiding, one lexically obvious and
    one only semantically so.  There is no free option, only a least-bad one.

``false_alarm``
    The opposite mistake.  The label trips an alarming keyword while naming
    something harmless, so the offline arm pays a detour to avoid a cardboard
    box.  Without this kind, "never push anything" would score well on every
    other kind and the corpus would stop measuring understanding.

The last four are what separate an LLM arm from the offline tables; see
``scenario_generation/materials.py`` for why the labels are misread.
"""

from __future__ import annotations

from dataclasses import dataclass

from obstacle import MovableObstacle
from scenarios._realism import push_force
from scenario_generation.materials import Material, Theme
from scenario_generation.models import DecisionSpec, TopologyResult

# Gate kinds in the order they are laid out before shuffling.  Every corpus
# therefore covers all six as soon as a map carries six gates.
GATE_PATTERN = (
    "blind_risk",
    "false_alarm",
    "move_or_detour",
    "blind_weight",
    "hidden_difficulty",
    "risk_or_distance",
    "risk_or_risk",
)

# Kinds a plain gate may be upgraded into, all of which need a semantic read.
_UPGRADES = ("blind_risk", "blind_weight", "false_alarm")

# Kinds that block two doorways and so need the wall to have two.  Every other
# kind gets a single doorway: a gate with one blocked door and one open one
# beside it is not a decision, because the robot simply walks through the
# second door without ever weighing the push.
TWO_DOOR_KINDS = ("risk_or_distance", "risk_or_risk")


def doors_needed(kinds) -> list[int]:
    return [2 if kind in TWO_DOOR_KINDS else 1 for kind in kinds]


def plan_gate_kinds(count: int, rng, *, include_hidden: bool,
                    trap_probability: float) -> list[str]:
    """Settle the kind of every gate before the walls are built."""
    kinds = []
    for kind in gate_kinds(count, rng):
        if kind == "move_or_detour" and rng.random() < trap_probability:
            kind = _UPGRADES[rng.randrange(len(_UPGRADES))]
        if kind == "hidden_difficulty" and not include_hidden:
            kind = "move_or_detour"
        kinds.append(kind)
    return kinds

# Labels the keyword table already reads as dangerous, used where a decision
# is supposed to be obvious to every arm.
_OBVIOUS_HAZARD = "shoring_prop"


@dataclass
class OidAllocator:
    next_oid: int = 1000

    def take(self) -> int:
        oid = self.next_oid
        self.next_oid += 1
        return oid


def gate_kinds(count: int, rng) -> list[str]:
    """Return one kind per gate, covering the pattern before repeating it."""
    kinds: list[str] = []
    while len(kinds) < count:
        block = list(GATE_PATTERN)
        rng.shuffle(block)
        kinds.extend(block)
    return kinds[:count]


def _fit(gate: dict) -> tuple[float, float]:
    """Return the (length, depth) a blocker needs to sit inside a doorway."""
    return gate["door_height"] - 0.22, gate["wall_thickness"] + 0.22


def _blocker(gate: dict, pose, oid: int, material: Material, rng, *,
             density: float | None = None,
             reveals: str | None = None) -> MovableObstacle:
    """Build one doorway blocker with the material's ground-truth physics."""
    l, d = _fit(gate)
    _, _, h = material.sample_size(rng)
    rho = material.sample_density(rng) if density is None else density
    return MovableObstacle(
        x=pose[0], y=pose[1], l=l, d=d, h=h, theta=pose[2],
        material=material.label,
        difficulty=push_force(rho * l * d * h, material.mu),
        contact_reveals=(material.reveals if reveals is None else reveals),
        oid=oid)


def _pick(materials, rng, used: set[str]) -> Material:
    """Prefer a label this map has not used yet, so six gates read differently."""
    fresh = [m for m in materials if m.label not in used]
    pool = fresh or list(materials)
    material = pool[rng.randrange(len(pool))]
    used.add(material.label)
    return material


def _cheap_plain(theme: Theme, rng, used: set[str]) -> Material:
    """One of the lightest honestly-labelled things in the theme."""
    ranked = sorted(theme.plain, key=lambda m: m.mu * sum(m.density))
    return _pick(tuple(ranked[:3]), rng, used)


def build_gate_decisions(topology: TopologyResult, allocator: OidAllocator,
                         rng, theme: Theme, kinds):
    """Build one choice per gate, drawing labels from ``theme``.

    ``kinds`` is settled before the topology is built, because a gate's kind
    decides how many doorways its wall needs.
    """
    gates = topology.anchors["gates"]
    obstacles: list[MovableObstacle] = []
    decisions: list[DecisionSpec] = []
    used: set[str] = set()

    for index, kind in enumerate(kinds[:len(gates)]):
        gate = gates[index]
        builder = _BUILDERS[kind]
        built, spec = builder(gate, index, allocator, rng, theme, used)
        obstacles.extend(built)
        spec.metadata["gate_index"] = index
        decisions.append(spec)

    return obstacles, decisions


# --------------------------------------------------------------------------
# One builder per gate kind.  Each returns (obstacles, decision).
# --------------------------------------------------------------------------

def _build_move_or_detour(gate, index, allocator, rng, theme, used):
    oid = allocator.take()
    material = _cheap_plain(theme, rng, used)
    obstacle = _blocker(gate, gate["direct"], oid, material, rng)
    return [obstacle], DecisionSpec(
        name=f"gate_{index}_move_or_detour", kind="move_or_detour",
        involved_oids=[oid], alternatives=["move", "detour"],
        expected_tradeoff="short direct push versus a longer open route",
        metadata={"obstacles": [oid], "material": material.label})


def _bypass(gate):
    return gate.get("bypass") or gate["direct"]


def _build_risk_or_distance(gate, index, allocator, rng, theme, used):
    risky_oid, safe_oid = allocator.take(), allocator.take()
    safe_material = _cheap_plain(theme, rng, used)
    l, d = _fit(gate)
    risky = MovableObstacle(
        x=gate["direct"][0], y=gate["direct"][1], l=l, d=d, h=1.25,
        theta=gate["direct"][2], material=_OBVIOUS_HAZARD,
        difficulty=push_force(310.0 * l * d * 1.25, 0.35), oid=risky_oid)
    safe = _blocker(gate, _bypass(gate), safe_oid, safe_material, rng)
    return [risky, safe], DecisionSpec(
        name=f"gate_{index}_risk_or_distance", kind="risk_or_distance",
        involved_oids=[risky_oid, safe_oid],
        alternatives=["risky_short_route", "safe_bypass"],
        expected_tradeoff="short risky push versus safe extra travel",
        metadata={"risky": risky_oid, "safer_alternative": safe_oid,
                  "material": _OBVIOUS_HAZARD})


def _build_hidden_difficulty(gate, index, allocator, rng, theme, used):
    oid = allocator.take()
    material = _pick(theme.plain, rng, used)
    # The label is honest; the load behind it is not.
    heavy = material.density[1] * rng.uniform(6.0, 9.0)
    obstacle = _blocker(gate, gate["direct"], oid, material, rng,
                        density=heavy, reveals="unexpected_heavy_load")
    return [obstacle], DecisionSpec(
        name=f"gate_{index}_hidden_difficulty", kind="hidden_difficulty",
        involved_oids=[oid], alternatives=["try_push", "detour"],
        expected_tradeoff="visual estimate followed by contact-driven replanning",
        metadata={"obstacles": [oid], "material": material.label})


def _build_blind_risk(gate, index, allocator, rng, theme, used):
    oid = allocator.take()
    material = _pick(theme.risk_traps, rng, used)
    obstacle = _blocker(gate, gate["direct"], oid, material, rng)
    return [obstacle], DecisionSpec(
        name=f"gate_{index}_blind_risk", kind="blind_risk",
        involved_oids=[oid], alternatives=["move", "detour"],
        expected_tradeoff=("a cheap push the keyword table calls safe and the "
                           "label calls dangerous"),
        metadata={"obstacles": [oid], "material": material.label,
                  "true_risk": material.risk,
                  "reveals": material.reveals})


def _build_blind_weight(gate, index, allocator, rng, theme, used):
    oid = allocator.take()
    material = _pick(theme.weight_traps, rng, used)
    obstacle = _blocker(gate, gate["direct"], oid, material, rng)
    return [obstacle], DecisionSpec(
        name=f"gate_{index}_blind_weight", kind="blind_weight",
        involved_oids=[oid], alternatives=["move", "detour"],
        expected_tradeoff=("a push the offline table prices from a light-"
                           "sounding label"),
        metadata={"obstacles": [oid], "material": material.label,
                  "true_risk": material.risk})


def _build_risk_or_risk(gate, index, allocator, rng, theme, used):
    blind_oid, obvious_oid = allocator.take(), allocator.take()
    blind_material = _pick(theme.risk_traps, rng, used)
    blind = _blocker(gate, gate["direct"], blind_oid, blind_material, rng)
    l, d = _fit(gate)
    obvious = MovableObstacle(
        x=_bypass(gate)[0], y=_bypass(gate)[1], l=l, d=d, h=1.2,
        theta=_bypass(gate)[2], material="glassware_crate",
        difficulty=push_force(120.0 * l * d * 1.2, 0.45), oid=obvious_oid)
    return [blind, obvious], DecisionSpec(
        name=f"gate_{index}_risk_or_risk", kind="risk_or_risk",
        involved_oids=[blind_oid, obvious_oid],
        alternatives=["move_blind_hazard", "move_known_hazard", "detour"],
        expected_tradeoff=("two blocked doors, one dangerous only to a reader "
                           "of the label"),
        metadata={"risky": blind_oid, "safer_alternative": obvious_oid,
                  "material": blind_material.label,
                  "true_risk": blind_material.risk,
                  "reveals": blind_material.reveals})


def _build_false_alarm(gate, index, allocator, rng, theme, used):
    oid = allocator.take()
    material = _pick(theme.false_alarms, rng, used)
    obstacle = _blocker(gate, gate["direct"], oid, material, rng)
    return [obstacle], DecisionSpec(
        name=f"gate_{index}_false_alarm", kind="false_alarm",
        involved_oids=[oid], alternatives=["move", "detour"],
        expected_tradeoff=("a harmless object behind an alarming word: the "
                           "detour is the expensive mistake"),
        metadata={"obstacles": [oid], "material": material.label,
                  "true_risk": material.risk})


_BUILDERS = {
    "move_or_detour": _build_move_or_detour,
    "risk_or_distance": _build_risk_or_distance,
    "hidden_difficulty": _build_hidden_difficulty,
    "blind_risk": _build_blind_risk,
    "blind_weight": _build_blind_weight,
    "risk_or_risk": _build_risk_or_risk,
    "false_alarm": _build_false_alarm,
}
