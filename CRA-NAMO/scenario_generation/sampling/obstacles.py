"""Generate physically coherent incidental clutter."""

from __future__ import annotations

import math

from obstacle import MovableObstacle
from scenarios._realism import push_force


def _angle(rng) -> float:
    draw = rng.random()
    if draw < 0.60:
        base = rng.choice((0.0, math.pi / 2))
        return base + rng.gauss(0.0, math.radians(5.0))
    if draw < 0.85:
        return rng.choice((-1.0, 1.0)) * rng.uniform(math.radians(15), math.radians(30))
    return rng.uniform(-math.pi / 2, math.pi / 2)


def fill_background(topology, existing, allocator, rng, target_count: int,
                    theme) -> list:
    """Rejection-sample clutter away from authored decision corridors.

    Clutter is drawn from the map's own vocabulary, traps included: a corpus
    where every heuristic-blind label sits in a doorway would be solvable by
    noticing which objects block doors rather than by reading them.
    """
    result = []
    walls = topology.walls
    zones = topology.anchors["background_zones"]
    catalog = theme.all_materials()
    attempts = 0
    while len(existing) + len(result) < target_count and attempts < target_count * 80:
        attempts += 1
        material = catalog[rng.randrange(len(catalog))]
        l, d, h = material.sample_size(rng)
        zone = rng.choice(zones)
        minx, miny, maxx, maxy = zone.bounds
        if maxx - minx <= l or maxy - miny <= d:
            continue
        obs = MovableObstacle(
            x=rng.uniform(minx + l / 2, maxx - l / 2),
            y=rng.uniform(miny + d / 2, maxy - d / 2),
            l=l, d=d, h=h, theta=_angle(rng), material=material.label,
            difficulty=push_force(material.sample_density(rng) * l * d * h,
                                  material.mu),
            contact_reveals=material.reveals, oid=allocator.take())
        if not topology.workspace.covers(obs.polygon):
            continue
        if any(obs.polygon.intersection(w.polygon).area > 1e-7 for w in walls):
            continue
        if any(obs.polygon.intersection(o.polygon).area > 1e-7
               for o in [*existing, *result]):
            continue
        if any(obs.polygon.intersects(region) for region in topology.reserved_regions):
            continue
        result.append(obs)
    return result

