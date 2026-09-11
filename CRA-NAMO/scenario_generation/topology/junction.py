"""Dense room graph biased toward loops and high-degree junctions."""

from scenario_generation.topology.room_graph import build_room_graph


def build_junction(request, rng, profile=None):
    return build_room_graph(request, rng, profile, junction_heavy=True)

