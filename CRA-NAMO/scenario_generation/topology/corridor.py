"""Longer corridor topology with narrower, more manipulation-heavy gates."""

from scenario_generation.topology.base import staged_topology


def build_corridor(request, rng, profile=None, door_counts=None):
    return staged_topology(
        request, rng, corridor=True,
        wall_angle_sigma=(None if profile is None else profile.wall_angle_sigma),
        gate_count=(3 if profile is None else profile.gate_decisions),
        door_counts=door_counts)
