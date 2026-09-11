"""Generate a valid transient obstruction tied to route timing."""

from __future__ import annotations

import math

from dynamics import Event, MoveTo, Mutate, after_moved, at_time
from obstacle import MovableObstacle
from scenarios._realism import MU_CASTORS, push_force
from scenario_generation.models import DecisionSpec


def build_dynamic_event(topology, existing, allocator, rng, *,
                        allow_state_mutation: bool = True):
    site = topology.anchors.get("dynamic_site")
    if site is not None:
        templates = (["temporary_portal", "reactive_portal", "state_mutation"]
                     if allow_state_mutation
                     else ["temporary_portal", "reactive_portal"])
        weights = ([0.50, 0.25, 0.25] if allow_state_mutation
                   else [0.50, 0.25])
        template = rng.choices(templates, weights=weights, k=1)[0]
        if template == "state_mutation":
            target = rng.choice(existing)
            event = Event(
                name="seeded load changes after settling",
                trigger=at_time(rng.uniform(14.0, 28.0)),
                effect=Mutate(
                    oid=target.oid, material="loaded_pallet",
                    difficulty=target.difficulty * rng.uniform(2.0, 3.5),
                    contact_reveals="unexpected_heavy_load"))
            decision = DecisionSpec(
                name="dynamic_state_reassessment", kind="state_change_replan",
                involved_oids=[target.oid],
                alternatives=["continue", "reassess_and_replan"],
                expected_tradeoff="stale physical belief versus online reassessment",
                metadata={"obstacles": [target.oid]})
            return [], [event], decision, None

        home_x, home_y = site["home"]
        target_x, target_y, target_theta = site["target"]
        oid = allocator.take()
        actor = MovableObstacle(
            x=home_x, y=home_y,
            l=site["door_height"] - 0.22,
            d=site["wall_thickness"] + 0.18,
            h=1.0, theta=target_theta, material="empty_cart",
            difficulty=push_force(48.0, MU_CASTORS), oid=oid)
        speed = rng.uniform(0.22, 0.38)
        trigger_t = rng.uniform(12.0, 20.0)
        trigger = (after_moved(rng.choice(existing).oid)
                   if template == "reactive_portal" else at_time(trigger_t))
        enter = Event(
            name=("seeded trolley reacts to a moved obstacle"
                  if template == "reactive_portal"
                  else "seeded trolley blocks a graph portal"),
            trigger=trigger,
            effect=MoveTo(
                oid=oid, goal=(target_x, target_y, target_theta), speed=speed))
        events = [enter]
        if template == "temporary_portal":
            events.append(Event(
                name="seeded trolley clears the graph portal",
                trigger=at_time(trigger_t + rng.uniform(10.0, 16.0)),
                effect=MoveTo(
                    oid=oid, goal=(home_x, home_y, target_theta), speed=speed)))
        decision = DecisionSpec(
            name=("dynamic_reactive_replan" if template == "reactive_portal"
                  else "dynamic_wait_or_replan"),
            kind=("coupled_replan" if template == "reactive_portal"
                  else "wait_or_replan"),
            involved_oids=[oid], alternatives=["wait", "replan"],
            expected_tradeoff="temporary portal closure versus a graph detour",
            metadata={"temporary_obstacles": [oid]})
        return [actor], events, decision, ((actor.x, actor.y), (target_x, target_y))

    gates = topology.anchors["gates"]
    left = gates[0]["direct"][0]
    right = gates[1]["direct"][0]
    x = (left + right) / 2.0
    start_y = topology.workspace.bounds[3] - 1.35
    target_y = (topology.anchors["direct_y"] + topology.anchors["bypass_y"]) / 2
    oid = allocator.take()
    actor = MovableObstacle(
        x=x, y=start_y, l=0.95, d=0.58, h=1.0,
        theta=math.pi / 2, material="empty_cart",
        difficulty=push_force(48.0, MU_CASTORS), oid=oid)
    speed = rng.uniform(0.22, 0.38)
    trigger_t = rng.uniform(12.0, 20.0)
    enter = Event(
        name="seeded trolley crosses the route",
        trigger=at_time(trigger_t),
        effect=MoveTo(oid=oid, goal=(x, target_y, math.pi / 2), speed=speed))
    leave = Event(
        name="seeded trolley clears the route",
        trigger=at_time(trigger_t + rng.uniform(10.0, 16.0)),
        effect=MoveTo(oid=oid, goal=(x, start_y, math.pi / 2), speed=speed))
    decision = DecisionSpec(
        name="dynamic_wait_or_replan", kind="wait_or_replan",
        involved_oids=[oid], alternatives=["wait", "replan"],
        expected_tradeoff="temporary moving obstruction versus route change",
        metadata={"temporary_obstacles": [oid]})
    return [actor], [enter, leave], decision, (
        (actor.x, actor.y), enter.effect.goal[:2])


def build_additional_dynamic_event(topology, existing, allocator, rng):
    """Create a room-local moving cart for counts beyond the portal actor."""
    zones = topology.anchors.get("background_zones", ())
    for _ in range(300):
        zone = rng.choice(zones)
        minx, miny, maxx, maxy = zone.bounds
        l, d = 0.72, 0.48
        if maxx - minx <= l + 0.8 or maxy - miny <= d + 0.8:
            continue
        x = rng.uniform(minx + l / 2 + 0.4, maxx - l / 2 - 0.4)
        y = rng.uniform(miny + d / 2 + 0.4, maxy - d / 2 - 0.4)
        target = (x + rng.uniform(-0.8, 0.8), y + rng.uniform(-0.8, 0.8), 0.0)
        actor = MovableObstacle(
            x=x, y=y, l=l, d=d, h=1.0, theta=0.0,
            material="empty_cart", difficulty=push_force(48.0, MU_CASTORS),
            oid=allocator.take())
        target_polygon = actor.polygon_at(*target)
        if not topology.workspace.covers(actor.polygon) or not topology.workspace.covers(target_polygon):
            continue
        if any(actor.polygon.intersects(w.polygon) or target_polygon.intersects(w.polygon)
               for w in topology.walls):
            continue
        if any(actor.polygon.intersects(o.polygon) or target_polygon.intersects(o.polygon)
               for o in existing):
            continue
        event = Event(
            name="seeded cart patrols a background zone",
            trigger=at_time(rng.uniform(10.0, 24.0)),
            effect=MoveTo(oid=actor.oid, goal=target,
                          speed=rng.uniform(0.18, 0.32)))
        return actor, [event]
    return None, []
