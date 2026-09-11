"""Topology templates for generated maps."""

from scenario_generation.topology.branching import build_branching
from scenario_generation.topology.corridor import build_corridor
from scenario_generation.topology.junction import build_junction
from scenario_generation.topology.room_graph import build_room_graph

BUILDERS = {
    "branching": build_branching,
    "corridor": build_corridor,
    "room_graph": build_room_graph,
    "junction": build_junction,
}

__all__ = ["BUILDERS", "build_branching", "build_corridor",
           "build_junction", "build_room_graph"]
