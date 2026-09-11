"""Generate physically coherent incidental clutter."""

from __future__ import annotations

import math

from obstacle import MovableObstacle
from scenarios._realism import push_force


_CATALOG = (
    # material, l range, d range, h range, density range, friction range
    ("cardboard_box", (0.45, 0.9), (0.4, 0.8), (0.5, 1.1), (30, 60), (0.28, 0.38)),
    ("wooden_crate", (0.6, 1.2), (0.55, 1.0), (0.7, 1.2), (100, 190), (0.35, 0.48)),
    ("empty_cart", (0.7, 1.2), (0.45, 0.8), (0.8, 1.2), (35, 65), (0.02, 0.05)),
    ("filing_cabinet", (0.55, 0.9), (0.45, 0.75), (1.1, 1.6), (180, 330), (0.38, 0.5)),
    ("loaded_pallet", (0.8, 1.4), (0.7, 1.2), (0.7, 1.3), (260, 480), (0.32, 0.45)),
)


def _angle(rng) -> float:
    draw = rng.random()
    if draw < 0.60:
        base = rng.choice((0.0, math.pi / 2))
        return base + rng.gauss(0.0, math.radians(5.0))
    if draw < 0.85:
        return rng.choice((-1.0, 1.0)) * rng.uniform(math.radians(15), math.radians(30))
    return rng.uniform(-math.pi / 2, math.pi / 2)


def fill_background(topology, existing, allocator, rng, target_count: int) -> list:
    """Rejection-sample clutter away from authored decision corridors."""
    result = []
    walls = topology.walls
    zones = topology.anchors["background_zones"]
    attempts = 0
    while len(existing) + len(result) < target_count and attempts < target_count * 80:
        attempts += 1
        material, lr, dr, hr, rr, mur = rng.choice(_CATALOG)
        l, d, h = rng.uniform(*lr), rng.uniform(*dr), rng.uniform(*hr)
        zone = rng.choice(zones)
        minx, miny, maxx, maxy = zone.bounds
        if maxx - minx <= l or maxy - miny <= d:
            continue
        obs = MovableObstacle(
            x=rng.uniform(minx + l / 2, maxx - l / 2),
            y=rng.uniform(miny + d / 2, maxy - d / 2),
            l=l, d=d, h=h, theta=_angle(rng), material=material,
            difficulty=push_force(rng.uniform(*rr) * l * d * h,
                                  rng.uniform(*mur)), oid=allocator.take())
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

