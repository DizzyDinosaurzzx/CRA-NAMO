"""Multi-gate topology with meaningful short-route/bypass choices."""

from scenario_generation.topology.base import staged_topology


def build_branching(request, rng, profile=None, door_counts=None):
    return staged_topology(
        request, rng, corridor=False,
        wall_angle_sigma=(None if profile is None else profile.wall_angle_sigma),
        gate_count=(3 if profile is None else profile.gate_decisions),
        door_counts=door_counts)
