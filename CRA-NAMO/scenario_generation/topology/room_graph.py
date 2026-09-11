"""Random orthogonal room graph with loops, junctions, and dead ends."""

from __future__ import annotations

import heapq
import math
from collections import deque
from itertools import combinations

from shapely.geometry import LineString, Point, box

from obstacle import StaticObstacle
from scenario_generation.models import TopologyResult
from scenario_generation.topology.base import shell

Node = tuple[int, int]
Edge = tuple[Node, Node]


def _edge(a: Node, b: Node) -> Edge:
    return (a, b) if a < b else (b, a)


def _adjacency(edges: set[Edge]) -> dict[Node, set[Node]]:
    graph: dict[Node, set[Node]] = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)
    return graph


def _path(edges: set[Edge], start: Node, goal: Node,
          excluded: set[Edge] | None = None) -> list[Node]:
    graph = _adjacency(edges - (excluded or set()))
    queue = deque([start])
    parent: dict[Node, Node | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == goal:
            result = []
            while node is not None:
                result.append(node)
                node = parent[node]
            return list(reversed(result))
        for nxt in sorted(graph.get(node, ())) :
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    return []


def _diameter(edges: set[Edge], nodes: list[Node]) -> int:
    longest = 0
    for start in nodes:
        for goal in nodes:
            route = _path(edges, start, goal)
            if route:
                longest = max(longest, len(route) - 1)
    return longest


def _weighted_distance(edges: set[Edge], start: Node, goal: Node,
                       centers: dict[Node, tuple[float, float]],
                       excluded: set[Edge] | None = None) -> float:
    graph = _adjacency(edges - (excluded or set()))
    queue = [(0.0, start)]
    best = {start: 0.0}
    while queue:
        distance, node = heapq.heappop(queue)
        if node == goal:
            return distance
        if distance > best[node] + 1e-12:
            continue
        for nxt in graph.get(node, ()):
            candidate = distance + math.dist(centers[node], centers[nxt])
            if candidate + 1e-12 < best.get(nxt, math.inf):
                best[nxt] = candidate
                heapq.heappush(queue, (candidate, nxt))
    return math.inf


def _all_edges(cols: int, rows: int) -> list[Edge]:
    result = []
    for col in range(cols):
        for row in range(rows):
            node = (col, row)
            if col + 1 < cols:
                result.append(_edge(node, (col + 1, row)))
            if row + 1 < rows:
                result.append(_edge(node, (col, row + 1)))
    return result


def _random_graph(cols: int, rows: int, start: Node, goal: Node,
                  rng, removal_probability: float) -> tuple[set[Edge], list[Node]]:
    """Remove grid edges while retaining a multi-route start/goal problem."""
    complete = _all_edges(cols, rows)
    edges = set(complete)
    order = list(complete)
    rng.shuffle(order)
    for candidate in order:
        if rng.random() > removal_probability:
            continue
        trial = edges - {candidate}
        route = _path(trial, start, goal)
        if len(route) < 4:
            continue
        edges = trial

    # At least three shortest-route edges must be non-bridges. Restore random
    # omitted edges until that counterfactual condition holds.
    omitted = [edge for edge in complete if edge not in edges]
    rng.shuffle(omitted)
    while True:
        route = _path(edges, start, goal)
        route_edges = [_edge(a, b) for a, b in zip(route, route[1:])]
        nonbridges = [edge for edge in route_edges
                      if _path(edges, start, goal, {edge})]
        if len(nonbridges) >= 3:
            return edges, route
        if not omitted:
            return set(complete), _path(set(complete), start, goal)
        edges.add(omitted.pop())


def _decision_edges(edges: set[Edge], route: list[Node],
                    start: Node, goal: Node) -> tuple[list[Edge], list[int], int]:
    """Mine three central edges whose individual detours preserve each other."""
    route_edges = [_edge(a, b) for a, b in zip(route, route[1:])]
    candidates = []
    base_hops = len(route_edges)
    for index, edge in enumerate(route_edges):
        alternate = _path(edges, start, goal, {edge})
        if alternate:
            alt_edges = {_edge(a, b) for a, b in zip(alternate, alternate[1:])}
            candidates.append((edge, index, len(alternate) - 1, alt_edges))
    if len(candidates) < 3:
        raise ValueError("room graph lacks three non-bridge decision edges")

    best = None
    for chosen in combinations(candidates, 3):
        chosen_edges = {item[0] for item in chosen}
        # A good counterfactual bypasses one decision while rejoining before
        # the others, rather than making all three decisions disappear at once.
        preserved = sum(len((chosen_edges - {item[0]}) & item[3])
                        for item in chosen)
        detour_gain = sum(max(0, item[2] - base_hops) for item in chosen)
        spread = max(item[1] for item in chosen) - min(item[1] for item in chosen)
        score = (preserved, detour_gain, spread)
        if best is None or score > best[0]:
            best = (score, chosen)
    selected = sorted(best[1], key=lambda item: item[1])
    return ([item[0] for item in selected],
            [item[2] for item in selected], best[0][0])


def _cuts(length: float, count: int, rng, jitter: float) -> list[float]:
    result = [0.0]
    for index in range(1, count):
        result.append(length * index / count + rng.uniform(-jitter, jitter))
    result.append(length)
    return result


def _wall_pieces(axis: str, fixed: float, lo: float, hi: float,
                 openings: list[tuple[float, float]], thickness: float,
                 name: str) -> list[StaticObstacle]:
    spans = []
    cursor = lo
    for center, width in sorted(openings):
        left, right = max(lo, center - width / 2), min(hi, center + width / 2)
        if left > cursor + 1e-8:
            spans.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < hi - 1e-8:
        spans.append((cursor, hi))
    walls = []
    for index, (a, b) in enumerate(spans):
        if axis == "vertical":
            walls.append(StaticObstacle.rect(
                fixed, (a + b) / 2, thickness, b - a, 0.0,
                f"{name}_{index}"))
        else:
            walls.append(StaticObstacle.rect(
                (a + b) / 2, fixed, b - a, thickness, 0.0,
                f"{name}_{index}"))
    return walls


def _shared_segment(edge: Edge, xs: list[float], ys: list[float]):
    (ca, ra), (cb, rb) = edge
    if ca != cb:
        col, row = min(ca, cb), ra
        return "vertical", xs[col + 1], ys[row], ys[row + 1]
    col, row = ca, min(ra, rb)
    return "horizontal", ys[row + 1], xs[col], xs[col + 1]


def _room_center(node: Node, xs: list[float], ys: list[float]):
    col, row = node
    return ((xs[col] + xs[col + 1]) / 2,
            (ys[row] + ys[row + 1]) / 2)


def _portal_pose(axis: str, fixed: float, center: float):
    return ((fixed, center, math.pi / 2)
            if axis == "vertical" else (center, fixed, 0.0))


def build_room_graph(request, rng, profile=None, *, junction_heavy: bool = False):
    width, height = request.width, request.height
    cols = rng.choice((4, 5))
    rows = rng.choice((3, 4))
    xs = _cuts(width, cols, rng, min(0.45, width / cols * 0.10))
    ys = _cuts(height, rows, rng, min(0.30, height / rows * 0.09))
    start_node = (0, rng.randrange(rows))
    goal_choices = [r for r in range(rows) if r != start_node[1]] or [start_node[1]]
    goal_node = (cols - 1, rng.choice(goal_choices))
    removal = 0.13 if junction_heavy else rng.uniform(0.24, 0.44)
    edges, shortest = _random_graph(
        cols, rows, start_node, goal_node, rng, removal)
    shortest_edges = [_edge(a, b) for a, b in zip(shortest, shortest[1:])]
    if len(shortest_edges) < 3:
        raise ValueError("room graph did not produce three decision edges")
    decision_edges, detour_hops, independence = _decision_edges(
        edges, shortest, start_node, goal_node)

    wall_t = 0.34
    workspace = box(0.0, 0.0, width, height)
    walls = shell(width, height, wall_t)
    gates = []
    portal_by_edge = {}
    reserved = [Point(_room_center(start_node, xs, ys)).buffer(0.8),
                Point(_room_center(goal_node, xs, ys)).buffer(0.8)]

    for index, edge in enumerate(_all_edges(cols, rows)):
        axis, fixed, lo, hi = _shared_segment(edge, xs, ys)
        span = hi - lo
        openings: list[tuple[float, float]] = []
        poses = []
        if edge in edges:
            if edge in decision_edges:
                door = max(1.05, min(1.35, (span - 0.42) / 2))
                offset = door / 2 + 0.16
                centers = [(lo + hi) / 2 - offset, (lo + hi) / 2 + offset]
                openings = [(center, door) for center in centers]
                poses = [_portal_pose(axis, fixed, center) for center in centers]
                gates.append({
                    "index": decision_edges.index(edge),
                    "direct": poses[0], "bypass": poses[1],
                    "door_height": door, "wall_thickness": wall_t,
                    "tilt": 0.0, "graph_edge": edge,
                })
            else:
                door = min(1.55, max(1.15, span * 0.33))
                center = (lo + hi) / 2 + rng.uniform(-span * 0.08, span * 0.08)
                openings = [(center, door)]
                poses = [_portal_pose(axis, fixed, center)]
            portal_by_edge[edge] = poses
            a_center, b_center = (_room_center(edge[0], xs, ys),
                                  _room_center(edge[1], xs, ys))
            for pose in poses:
                reserved.append(LineString([a_center, pose[:2], b_center]).buffer(0.52))
        walls.extend(_wall_pieces(
            axis, fixed, lo, hi, openings, wall_t, f"room_boundary_{index}"))

    gates.sort(key=lambda gate: gate["index"])
    noncritical = [
        edge for edge in edges
        if edge not in decision_edges
        and _path(edges, start_node, goal_node, {edge})
    ]
    if not noncritical:
        noncritical = [edge for edge in edges if edge not in decision_edges]
    dynamic_edge = rng.choice(noncritical)
    dynamic_pose = portal_by_edge[dynamic_edge][0]
    dynamic_home = _room_center(dynamic_edge[0], xs, ys)
    dynamic_site = {
        "home": dynamic_home,
        "target": dynamic_pose,
        "door_height": min(1.4, max(1.1, 0.32 * (
            _shared_segment(dynamic_edge, xs, ys)[3]
            - _shared_segment(dynamic_edge, xs, ys)[2]))),
        "wall_thickness": wall_t,
    }

    graph = _adjacency(edges)
    nodes = [(c, r) for c in range(cols) for r in range(rows)]
    centers = {node: _room_center(node, xs, ys) for node in nodes}
    degree = {node: len(graph.get(node, ())) for node in nodes}
    cycle_rank = len(edges) - cols * rows + 1
    base_distance = _weighted_distance(edges, start_node, goal_node, centers)
    detour_metres = [round(max(
        0.0,
        _weighted_distance(edges, start_node, goal_node, centers, {edge})
        - base_distance), 3) for edge in decision_edges]
    edge_lengths = [math.dist(centers[a], centers[b]) for a, b in edges]
    metrics = {
        "room_count": cols * rows,
        "graph_edge_count": len(edges),
        "cycle_rank": cycle_rank,
        "dead_end_count": sum(value == 1 for value in degree.values()),
        "junction_count": sum(value >= 3 for value in degree.values()),
        "shortest_hops": len(shortest_edges),
        "decision_detour_hops": detour_hops,
        "decision_graph_detour_m": detour_metres,
        "decision_independence_score": independence,
        "graph_diameter": _diameter(edges, nodes),
        "mean_degree": round(2.0 * len(edges) / len(nodes), 3),
        "mean_edge_length_m": round(sum(edge_lengths) / len(edge_lengths), 3),
        "row_count": rows,
        "column_count": cols,
    }
    zones = [box(xs[c] + 0.45, ys[r] + 0.45,
                 xs[c + 1] - 0.45, ys[r + 1] - 0.45)
             for c in range(cols) for r in range(rows)
             if xs[c + 1] - xs[c] > 1.2 and ys[r + 1] - ys[r] > 1.2]
    # Free-standing angled walls add SE(2) variation without corrupting the
    # room graph. They are rejected whenever they touch a graph connector.
    angled_walls = []
    target_islands = rng.randint(1, 3)
    for attempt in range(80):
        if len(angled_walls) >= target_islands:
            break
        zone = rng.choice(zones)
        minx, miny, maxx, maxy = zone.bounds
        length = rng.uniform(0.8, min(1.8, max(maxx - minx, maxy - miny)))
        theta = rng.uniform(-math.pi / 2, math.pi / 2)
        wall = StaticObstacle.rect(
            rng.uniform(minx, maxx), rng.uniform(miny, maxy),
            length, rng.uniform(0.16, 0.24), theta,
            f"angled_island_{len(angled_walls)}")
        if not workspace.covers(wall.polygon):
            continue
        if any(wall.polygon.intersection(other.polygon).area > 1e-7
               for other in [*walls, *angled_walls]):
            continue
        if any(wall.polygon.intersects(region) for region in reserved):
            continue
        angled_walls.append(wall)
    walls.extend(angled_walls)
    metrics["angled_wall_count"] = len(angled_walls)
    return TopologyResult(
        workspace=workspace, walls=walls,
        start=_room_center(start_node, xs, ys),
        goal=_room_center(goal_node, xs, ys),
        anchors={
            "gates": gates,
            "background_zones": zones,
            "dynamic_site": dynamic_site,
            "topology_metrics": metrics,
            "topology_family": "junction" if junction_heavy else "room_graph",
            "graph_edges": list(edges),
        },
        reserved_regions=reserved)
